// Base
#include "Hydro2.H"
// Parsing and Input Handeling
#include "AMReX_MultiFab.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "BC/Nothing.H"
#include "BC/NSCBC.H"
#include "BC/NSCBC4.H"
#include "Numeric/Stencil.H"
#include "IC/Constant.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLL.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/HLLC_LM.H"
#include "Solver/Local/FluidRiemann/HLLC_ADC.H"
//#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
//#include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"


// Limiters
//#include "Solver/Local/Limiter/Minmod.H"
//#include "Solver/Local/Limiter/VanLeer.H"

//EOS
#include "Solver/EOS/EOS.H"
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"


#include <AMReX_Math.H>
#include "AMReX_TimeIntegrator.H"


//#if AMREX_SPACEDIM == 2

namespace Integrator
{

// Spalding mass-transfer number  B_M = (Y_local - Y_inf) / (1 - Y_local).
// Single source of truth for both the Mix() and the RHS() pre-source loop --
// avoids the previous bug where the two computations had drifted to a wrong
// (1 + Y_inf) denominator. (Spalding 1953; Sirignano 2010 Eq. 2.18;
// Abramzon-Sirignano IJHMT 1989 Eq. 18.)
//
// `Y_local` here is whatever vapor mass-fraction the caller chose to use --
// in the simplified Spalding closure used in this code, that is the cell
// mass fraction Y(i,j,k). A more rigorous implementation would evaluate
// the saturation mass fraction via Antoine + Clausius-Clapeyron at the
// interface temperature; out of scope for this fix (see F-2 in
// bin/Spalding_Rep.md).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SpaldingBM(Set::Scalar Y_local, Set::Scalar Y_inf, Set::Scalar small)
{
    return (Y_local - Y_inf) / (1.0 - Y_local + small);
}

// ----------------------------------------------------------------------------
// Per-cell scaling factor for the Hu-Adams-Shu positivity-preserving flux
// limiter. Returns the largest theta in [0,1] such that the trial conserved
// state  U(theta) = B + theta * (s * dF)  stays in the admissible set
//   G = { rho >= eps_rho,  p >= eps_p }   (eps_p = floor on absolute pressure).
// B = (rho, Mx, My, E) is the source-inclusive low-order baseline for the cell,
// dF = (mass, Mx, My, E) is the high-minus-low face flux correction, and s folds
// in the convexity-split factor and signed lambda = ± (2*DIM) * dt/dx_dir.
//
// Density is a linear constraint; internal energy UE = E - |M|^2/(2 rho) is
// concave along the line in theta, so with h(0) >= 0 (baseline in G) a single
// bisection brackets the largest safe theta. The absolute-pressure floor
// p >= eps_p is equivalent to UE >= ue_floor =
// (eps_p + gamma_eff*p0_eff - pref)/(gamma_eff - 1) for frozen mixture
// gamma_eff, p0_eff.  (Perthame-Shu 1996; Zhang-Shu 2010; Hu-Adams-Shu 2013.)
//
// Optional per-phase guard (guard_phase): the mixture mass change s*dF[0] splits
// across phases by the upwind face fraction ef, so the partial densities change
// by s*ef*dF[0] and s*(1-ef)*dF[0] per unit theta. Both are linear, so we just
// add two more upper-bound clamps on t to keep rho*eta0, rho*eta1 >= phase_floor
// (= the same `small` the post-update partial-density floor would otherwise
// inject mass to enforce). re0_base/re1_base are the partial baselines (Bbase
// comps 4,5). The pressure (UE) constraint depends only on the mixture, so the
// bisection below is unchanged -- it just runs on the tightened [0,t_rho].
// ----------------------------------------------------------------------------
// Phase-dependent pressure floor lives in Solver::EOS::EOS::PressureFloor (so
// the EOS-facing MixedPressure/diagnostics share one definition). Gas
// (eta -> 1) is held at p >= eps_p; the stiffened liquid (eta -> 0) may tension
// down to -p_cav, with a hyperbolicity guard (p -> -p0_eff makes c -> 0).
// ----------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar PPThetaCell(const Set::Scalar B[4], Set::Scalar s, const Set::Scalar dF[4],
                               Set::Scalar gamma_eff, Set::Scalar p0_eff,
                               Set::Scalar eps_rho, Set::Scalar eps_p, Set::Scalar pref,
                               bool guard_phase = false,
                               Set::Scalar re0_base = 0.0, Set::Scalar re1_base = 0.0,
                               Set::Scalar ef = 0.0, Set::Scalar phase_floor = 0.0)
{
    // Per-unit-theta change vector.
    const Set::Scalar V0 = s * dF[0]; // rho
    const Set::Scalar V1 = s * dF[1]; // Mx
    const Set::Scalar V2 = s * dF[2]; // My
    const Set::Scalar V3 = s * dF[3]; // E

    // Per-phase per-unit-theta change (partial mass = upwind ef * mixture mass).
    const Set::Scalar V_re0 = s * ef * dF[0];
    const Set::Scalar V_re1 = s * (1.0 - ef) * dF[0];

    const Set::Scalar r0 = B[0];
    const Set::Scalar ue_floor = (eps_p + gamma_eff * p0_eff - pref) / (gamma_eff - 1.0);

    // Internal energy along the line; valid only where density stays positive.
    auto UE = [&](Set::Scalar t) -> Set::Scalar {
        Set::Scalar r  = r0 + t * V0;
        Set::Scalar mx = B[1] + t * V1;
        Set::Scalar my = B[2] + t * V2;
        Set::Scalar e  = B[3] + t * V3;
        return e - 0.5 * (mx * mx + my * my) / r;
    };

    // Baseline must already be admissible; otherwise the limiter cannot help
    // (flux blending only adds the correction) and we fall back to pure HLL.
    if (r0 <= eps_rho || UE(0.0) < ue_floor) return 0.0;
    if (guard_phase && (re0_base < phase_floor || re1_base < phase_floor)) return 0.0;

    // Density: largest t keeping rho >= eps_rho.
    Set::Scalar t_rho = 1.0;
    if (V0 < 0.0) t_rho = (r0 - eps_rho) / (-V0);
    if (t_rho > 1.0) t_rho = 1.0;
    if (t_rho < 0.0) t_rho = 0.0;

    // Per-phase densities: tighten t so each partial stays >= phase_floor.
    if (guard_phase)
    {
        if (V_re0 < 0.0) { Set::Scalar t0 = (re0_base - phase_floor) / (-V_re0); if (t0 < t_rho) t_rho = t0; }
        if (V_re1 < 0.0) { Set::Scalar t1 = (re1_base - phase_floor) / (-V_re1); if (t1 < t_rho) t_rho = t1; }
        if (t_rho < 0.0) t_rho = 0.0;
    }

    // Internal energy: concave in t, h(0) >= 0. If the whole [0,t_rho] is safe
    // take t_rho, else bisect for the crossing.
    if (UE(t_rho) >= ue_floor) return t_rho;

    Set::Scalar lo = 0.0, hi = t_rho;
    for (int it = 0; it < 50; ++it)
    {
        Set::Scalar mid = 0.5 * (lo + hi);
        if ((r0 + mid * V0) > eps_rho && UE(mid) >= ue_floor) lo = mid;
        else hi = mid;
    }
    return lo;
}

Hydro2::Hydro2(IO::ParmParse& pp) : Hydro2()
{
    pp.queryclass(*this);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// PARSE ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Parse(Hydro2& value, IO::ParmParse& pp)
{
    BL_PROFILE("Integrator::Hydro2::Hydro2()");
    {
        // REFINEMENT CRITERION
        pp.query_default("eta_refinement_criterion", value.eta_refinement_criterion, 0.001);   // eta-based refinement
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01); // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01); // velocity gradient-based refinement
        pp.query_default("divu_refinement_criterion", value.divu_refinement_criterion, 0.05);   // Ducros-weighted compression (shock) refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);    // density-based refinement

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);                // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl);  // cfl condition
        pp_query_default("pref", value.pref, 0.0);          // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);       // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-6);   // eta cutoff value

        // Positivity-preserving flux limiter (Hu-Adams-Shu)
        pp_query_default("pp_flux_limiter", value.pp_flux_limiter, 1); // 1: on, 0: off
        pp_query_default("pp_source_limit", value.pp_source_limit, 1); // 1: fold source into PP baseline, 0: limit flux only
        pp_query_default("pp_source_limiter", value.pp_source_limiter, 0); // 1: scale source for positivity + per-phase flux guard (needs pp_flux_limiter); 0: off
        if (value.pp_source_limiter != 0 && value.pp_flux_limiter == 0)
            Util::Abort(INFO, "pp_source_limiter requires pp_flux_limiter=1 (it lives in the PP-limiter Pass B/C/D); got pp_flux_limiter=0");
        pp_query_default("eps_p", value.eps_p, 1.0);          // floor on absolute pressure p (gas side, eta -> 1)
        pp_query_default("eps_rho", value.eps_rho, 1.0E-10);  // density floor
        pp_query_default("p_cav", value.p_cav, 0.0);          // liquid cavitation floor: hold the liquid (eta -> 0) at p >= -p_cav (gas stays p >= eps_p). 0 = vacuum floor, no tension.
        if (value.p_cav < 0.0)
            Util::Abort(INFO, "p_cav is the magnitude of the liquid tensile/cavitation floor (liquid held at p >= -p_cav); it must be >= 0");

        // HLLC-ADC carbuncle cure: drive the omega blend from a Ducros-weighted
        // div(u) compression shock tag instead of the internal pressure-ratio
        // sensor (carbuncles are pressure-quiet). A cell is tagged when the
        // sensor exceeds adc_shock_threshold; the tag is then grown by adc_grow
        // cells so HLL dissipation covers the post-shock band, not just the
        // pressure jump. adc_grow < 0 disables the tag (use the pressure sensor).
        pp_query_default("adc_shock_threshold", value.adc_shock_threshold, 0.05);
        pp_query_default("adc_grow", value.adc_grow, 1);
        pp_query_default("lagrange", value.lagrange, 0.0);  // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);            // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix");  // Roe solver entropy fix
        pp_query_default("Spec_Vol", value.Spec_Vol, 1);    // 0: Solve Energy via specific mass | 1: Solve Energy via specific volume

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, false); // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("apply_weight", value.apply_weight, false);                  // Apply weight when solving, default: false --> "No Weight"
        pp_query_default("apply_vaporization", value.apply_vaporization, false);       // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"
        pp_query_default("static_eta", value.static_eta, false);                      // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // FLUID 0
        pp_query_required("mu0", value.mu0); // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0); // bulk viscosity coefficient
        
        // FLUID 1
        pp_query_required("mu1", value.mu1); // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0); // bulk viscosity coefficient

        // EOS
        Solver::EOS::Tammann::Parse(value.eos0, pp, "eos0.");
        Solver::EOS::Tammann::Parse(value.eos1, pp, "eos1.");

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0);    // Surface tension condition
        pp_query_default("Dv", value.Dv, 0.0);          // Vapor Diffusivity
        // HRM bulk cavitation (liquid -> vapor when p < p_sat). Off by default.
        pp_query_default("apply_cavitation", value.apply_cavitation, 0);
        pp_query_default("tau_cav", value.tau_cav, 1.0e-4);            // cavitation relaxation time [s]
        pp_query_default("antoine_A", value.antoine_A, 4.10549);      // log10(p_sat[bar]) = A - B/(T[K]+C)
        pp_query_default("antoine_B", value.antoine_B, 1625.928);
        pp_query_default("antoine_C", value.antoine_C, -92.839);
        pp_query_default("L_vap", value.L_vap, 256.0e3);              // latent heat [J/kg] (Phase B energy coupling, unused in Phase A)
        if (value.apply_cavitation != 0 && value.tau_cav <= 0.0)
            Util::Abort(INFO, "tau_cav must be > 0 when apply_cavitation=1 (it is the cavitation relaxation time)");
        pp_query_required("epsilon", value.epsilon);    // diffuse interface thickness Y_infinity
        pp_query_default("Y_infinity", value.Y_infinity, 0.0); // Far Field Vapor Mass Fraction
        pp_query_default("Mob", value.Mob_user, 0.0);   // CH mobility scale M0: M = M0 * epsilon^2
        if (value.epsilon <= 0.0)
        {
            Util::Abort(INFO, "epsilon must be positive for Hydro2 Cahn-Hilliard mobility; got ", value.epsilon);
        }

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 1); // 1: Smooth Normals (default)  2: legacy Hessian-based
        pp_query_default("smooth_kernel_size", value.smooth_kernel_size, 3); // Gaussian normal-smoothing kernel: 3 (3x3) or 5 (5x5)

        // ===========================================================
        // 7-EQUATION (Baer-Nunziato) SCAFFOLDING SWITCHES
        //   equation_count = 5  -> existing Allaire-style model (default)
        //   equation_count = 7  -> per-phase mass/momentum/energy + alpha
        //
        // Phase A: parsing wired, MultiFabs registered, Riemann/NSCBC stubs
        //          available, but the RHS, Mix, and BC paths still operate
        //          on the 5-eqn state. Selecting equation_count=7 will
        //          abort with a clear "not yet implemented" message in the
        //          first integrator stage that touches the 7-eqn path.
        // ===========================================================
        pp_query_default("equation_count",       value.equation_count,       5);
        pp_query_default("relax_pressure_stiff", value.relax_pressure_stiff, 1);
        pp_query_default("relax_velocity_stiff", value.relax_velocity_stiff, 1);
        pp_query_default("surface_tension_form", value.surface_tension_form, 0);
        if (value.equation_count != 5 && value.equation_count != 7)
        {
            Util::Abort(INFO, "equation_count must be 5 or 7; got ", value.equation_count);
        }

        // INTERFACE COMPRESSION
        pp_query_default("apply_sharpening", value.apply_sharpening, false);
        pp_query_default("sharpening_frequency", value.sharpening_frequency, 10);
        pp_query_default("reinit_max_iter", value.reinit_max_iter, 10);
        pp_query_default("reinit_tolerance", value.reinit_tolerance, 1e-6);
        pp_query_default("density_max_iter", value.density_max_iter, value.reinit_max_iter); // Density correction iterations (papers use 5-10 iterations)
        pp_query_default("density_tolerance", value.density_tol, value.reinit_tolerance);    // Density correction tolerance
        pp_query_default("density_relax", value.omega_relax, 0.5);                  // Relaxation parameter (0.3-0.7 typical)

    
        // Boundry Conditions
        pp_query_default("nghost", value.nghost, 2); // Number of Ghost Cells

        // The curvature pipeline fills the derived fields (eta_x, smoothed
        // normals) into the ghost layer so kappa is coarse-fine/physical-
        // boundary correct. That needs nghost >= krad+2 (3 for a 3x3 smoothing
        // kernel, 4 for 5x5). Below that, patch-edge curvature ghosts fall back
        // to same-level FillBoundary (today's behavior) and can be stale.
        {
            const int krad_req = (value.smooth_kernel_size >= 5) ? 2 : 1;
            if (value.nghost < krad_req + 2)
                Util::Message(INFO, "WARNING: nghost=", value.nghost,
                              " < krad+2=", krad_req + 2,
                              " for smooth_kernel_size=", value.smooth_kernel_size,
                              "; curvature at coarse-fine/physical boundaries falls back to FillBoundary (same-level only).");
        }

        bool uses_nscbc = false;
        std::vector<std::string> bc_faces = { "xlo", "xhi", "ylo", "yhi" };
#if AMREX_SPACEDIM == 3
        bc_faces.push_back("zlo");
        bc_faces.push_back("zhi");
#endif

        for (const auto &face : bc_faces)
        {
            std::string bc_type_str;
            pp.query(("density.bc.type." + face).c_str(), bc_type_str);
            int bc_type = BC::BCUtil::ReadString(bc_type_str);
            if (BC::BCUtil::IsNSCBC(bc_type))
            {
                uses_nscbc = true;
                break;
            }
        }

        Util::Message(INFO, "uses_nscbc=", uses_nscbc);

        // Initialize boundary conditions based on whether NSCBC is used
        if (uses_nscbc)
        {
            if (value.nghost == 2)
            {
                value.nscbc_bc = new BC::NSCBC(pp);
                value.nscbc4_bc = nullptr;
                Util::Message(INFO, "Parsing NSCBC (2-cell)");
            }
            else if (value.nghost == 4)
            {
                value.nscbc_bc = nullptr;
                value.nscbc4_bc = new BC::NSCBC4(pp);
                Util::Message(INFO, "Parsing NSCBC4 (4-cell)");
            }
            else
            {
                Util::Abort(INFO, "NSCBC requires nghost = 2 or 4");
            }

            // Use BC::Nothing for standard BC pointers
            value.density_bc = &value.bc_nothing;
            value.energy_bc = &value.bc_nothing;
            value.momentum_bc = &value.bc_nothing;

            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
            Util::Message(INFO, "nscbc4_bc Pointer=", value.nscbc4_bc);
        }
        else
        {
            // Standard mode: Use Expression BCs
            value.nscbc_bc = nullptr;

            value.density_bc = new BC::Expression(1, pp, "density.bc");
            value.energy_bc = new BC::Expression(1, pp, "energy.bc");
            value.momentum_bc = new BC::Expression(2, pp, "momentum.bc");

            Util::Message(INFO, "Parsing Reg");
            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
        }

        // Eta BC: parse from "eta.bc" if user provides it, otherwise zero neumann.
        // Eta is a volume fraction transported by advection + Cahn-Hilliard;
        // zero-neumann (ghost = interior) is the physically correct default.
        if (pp.contains("eta.bc.type.xlo") || pp.contains("eta.bc.type.ylo"))
        {
            value.eta_bc = new BC::Expression(1, pp, "eta.bc");
        }
        else
        {
            value.eta_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
        }
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,           value.eta_bc, 1, nghost, "eta", true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.eta_bc, 1, nghost, "eta_old", false, true);
        value.RegisterNewFab(value.rho_eta0_mf,      value.density_bc, 1, nghost, "rho_eta0", true, true);
        value.RegisterNewFab(value.rho_eta1_mf,      value.density_bc, 1, nghost, "rho_eta1", true, true);
        value.RegisterNewFab(value.rho_eta0_old_mf,  value.density_bc, 1, nghost, "rho_eta0_old", false, true);
        value.RegisterNewFab(value.rho_eta1_old_mf,  value.density_bc, 1, nghost, "rho_eta1_old", false, true);

        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing, 1, 0, "etadot", true, false);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing, 4, 0, "hess_eta", false, false, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing, 2, 0, "n_hat", false, false, { "x", "y" });

        // FLUID 0
        value.RegisterNewFab(value.density0_mf,     value.density_bc,   1, nghost, "density0",     false, false );
        value.RegisterNewFab(value.density0_old_mf, value.density_bc,   1, nghost, "density0_old", false, false);

        value.RegisterNewFab(value.energy0_mf,      value.energy_bc,    1, nghost, "energy0", false, false);
        value.RegisterNewFab(value.energy0_old_mf,  value.energy_bc,    1, nghost, "energy0_old" , false, false);

        value.RegisterNewFab(value.momentum0_mf,    value.momentum_bc,  2, nghost, "momentum0", false, false, { "x", "y" });
        value.RegisterNewFab(value.momentum0_old_mf,value.momentum_bc,  2, nghost, "momentum0_old", false, false);
 
        //value.RegisterNewFab(value.T0_mf,           value.temperature_bc, 1, nghost, "T0", false, false);
        //value.RegisterNewFab(value.k0_thermal_mf,   &value.bc_nothing, 1, nghost, "k0_thermal", false, false);
        //value.RegisterNewFab(value.h0_thermal_mf,   &value.bc_nothing, 1, nghost, "h0_thermal", false, false);

        value.RegisterNewFab(value.pressure0_mf,    value.energy_bc,  1, nghost, "pressure0", false, false);
        value.RegisterNewFab(value.velocity0_mf,    &value.bc_nothing,  2, nghost, "velocity0", false, false, { "x", "y" });

        // FLUID 1
        value.RegisterNewFab(value.density1_mf,     value.density_bc,   1, nghost, "density1", false, false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc,   1, nghost, "density1_old", false, false);

        value.RegisterNewFab(value.energy1_mf,      value.energy_bc,    1, nghost, "energy1", false, false);
        value.RegisterNewFab(value.energy1_old_mf,  value.energy_bc,    1, nghost, "energy1_old", false, false);

        value.RegisterNewFab(value.momentum1_mf,    value.momentum_bc,  2, nghost, "momentum1", false, false, { "x", "y" });
        value.RegisterNewFab(value.momentum1_old_mf,value.momentum_bc,  2, nghost, "momentum1_old", false, false);

        //value.RegisterNewFab(value.T1_mf,           value.temperature_bc, 1, nghost, "T1", false, false);
        //value.RegisterNewFab(value.k1_thermal_mf,   &value.bc_nothing, 1, nghost, "k1_thermal", false, false);
        //value.RegisterNewFab(value.h1_thermal_mf,   &value.bc_nothing, 1, nghost, "h1_thermal", false, false);

        value.RegisterNewFab(value.pressure1_mf,    value.energy_bc,  1, nghost, "pressure1", false, true);
        value.RegisterNewFab(value.velocity1_mf,    &value.bc_nothing,  2, nghost, "velocity1", false, true, { "x", "y" });

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,     value.energy_bc, 1, nghost, "pressure", true, false);
        value.RegisterNewFab(value.velocity_mf,     &value.bc_nothing,  2, nghost, "velocity", true, false, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf,    &value.bc_nothing,  1, 0, "vorticity", true, false);
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true, false);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false, false);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost, "energy_per_vol", true, true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost, "energy_per_mass", true, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost, "energy_vol_old", false, true);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost, "energy_mas_old", false, true);
        value.RegisterNewFab(value.momentum_mf,     value.momentum_bc,  2, nghost, "momentum", true, true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc,  2, nghost, "momentum_old", false, true, { "x", "y" });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing, 1, 0, "m0", false, false);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing, 2, 0, "u0", false, false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing, 2, 0, "q0", false, false, { "x", "y" });
        value.RegisterNewFab(value.Source_mf,       &value.bc_nothing,  4, 0, "Source", true, false, { "_rho", "_Mx", "_My","_E" });
        value.RegisterNewFab(value.Fsv_mf,          &value.bc_nothing,  2, 0, "Fsv", true, false, { "x", "y" });  // Surface Tension
        value.RegisterNewFab(value.Fw_mf,           &value.bc_nothing,  2, 0, "Fw", true, false, { "x", "y" });   // Weight
        value.RegisterNewFab(value.Ldot_mf,         &value.bc_nothing,  2, 0, "Ldot", true, false, { "x", "y" });  // Ldot
        value.RegisterNewFab(value.T_mf,            value.energy_bc,  1, nghost, "T", true, false);                  // Temperature
        value.RegisterNewFab(value.cp_mf,           &value.bc_nothing,  1, nghost, "cp", false, true);         // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,           &value.bc_nothing,  1, nghost, "cv", false, true);         // Constant Volume Specific Heat
        //value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false, true);         // Thermal Conductivity
        //value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false, true);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc, 1, nghost, "gamma", true, false);                 // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf,           value.energy_bc, 1, nghost, "p0", true, true);                    // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.eta_bc, 1, nghost, "mu_chem", true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true, false);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, false, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   value.energy_bc,  1, nghost, "UE_per_vol", true, false);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   value.energy_bc,  1, nghost, "UE_per_mass", true, false);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   value.energy_bc,  1, nghost, "KE_per_vol", true, false);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   value.energy_bc,  1, nghost, "KE_per_mass", true, false);        // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,           &value.bc_nothing,  1, nghost, "Spadling_Number", true, false);    // Spalding Number
        value.RegisterNewFab(value.Y_mf,            &value.bc_nothing,  1, nghost, "Mass_Fraction", true, false);       // Mass Fraction

        // Curvature pipeline fields
        value.RegisterNewFab(value.eta_x_mf,            &value.bc_nothing,  1, nghost, "eta_x", false, false);
        value.RegisterNewFab(value.eta_y_mf,            &value.bc_nothing,  1, nghost, "eta_y", false, false);
        value.RegisterNewFab(value.gradmag_mf,          &value.bc_nothing,  1, nghost, "gradmag", false, false);
        value.RegisterNewFab(value.nx_smoothed_mf,      &value.bc_nothing,  1, nghost, "nx_smoothed", false, false);
        value.RegisterNewFab(value.ny_smoothed_mf,      &value.bc_nothing,  1, nghost, "ny_smoothed", false, false);
        value.RegisterNewFab(value.kappa_SF_mf,         &value.bc_nothing,  1, nghost, "kappa_SF", true, false);

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, nghost, "grad_eta", true, false, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, false, { "Active", "SN", "Legacy" });
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", false, false, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,     &value.bc_nothing,  1, 0, "rho_flux", true, false);                    // Density Flux
        value.RegisterNewFab(value.M_flux_mf,       &value.bc_nothing,  2, 0, "M_flux", true, false, { "x", "y" });        // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,       &value.bc_nothing,  1, 0, "E_flux", true, false);                      // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,      &value.bc_nothing,  2, 0, "div_tau", true, false, { "x", "y" });            // Energy Flux
        value.RegisterNewFab(value.hess_u_mf,       &value.bc_nothing,  8, 0, "hess_u", false, false, {
                                                                                                     "000","001",
                                                                                                     "010","011",
                                                                                                     "100","101",
                                                                                                     "110","111",
                                                                                                    }); // hess_u Flux
        value.RegisterNewFab(value.Vap_dot_mf, &value.bc_nothing, 5, 0, "Vap_dot", true, false, { "_eta", "_rho", "_Mx", "_My", "_E" }); // Momentum Flux
        value.RegisterNewFab(value.omega_adc_mf, &value.bc_nothing, 2, 0, "omega_adc", true, false, { "_xhi", "_yhi" }); // HLLC-ADC shock locator (per hi-face omega)
        value.RegisterNewFab(value.flux_dev_mf, &value.bc_nothing, 2, 0, "flux_dev", true, false, { "_xhi", "_yhi" }); // |F_used - F_HLL| rel. dev. per hi-face
        value.RegisterNewFab(value.shock_sensor_mf, &value.bc_nothing, 1, 0, "shock_sensor", true, false); // raw Ducros-weighted compression sensor (compared to adc_shock_threshold)
        value.RegisterNewFab(value.pp_theta_mf,  &value.bc_nothing, 2, 0, "pp_theta", true, false, { "_xhi", "_yhi" }); // PP limiter blend factor per hi-face (1=HLLC, 0=HLL)
        value.RegisterNewFab(value.pp_dmass_mf,  &value.bc_nothing, 1, 0, "pp_dmass", true, false); // antidiffusive mass flux removed by PP limiter at hi-faces (interface desync source)
        value.RegisterNewFab(value.pp_efloor_mf, &value.bc_nothing, 1, 0, "pp_efloor", true, false); // internal energy injected by the eps_p positivity backstop (pressure flooring)

        // ====================================================================
        // 7-EQUATION (Baer-Nunziato) PER-PHASE CONSERVATIVES
        //
        // Registered only when equation_count == 7 so the 5-eqn build is
        // bit-identical to the pre-conversion behavior. Components match
        // the layout declared in Hydro2.H.
        // ====================================================================
        if (value.equation_count == 7)
        {
            // Phase 0 (liquid)
            value.RegisterNewFab(value.alpha_rho_0_mf,     value.density_bc,  1, nghost, "alpha_rho_0",      true,  true);
            value.RegisterNewFab(value.alpha_rho_0_old_mf, value.density_bc,  1, nghost, "alpha_rho_0_old",  false, true);
            value.RegisterNewFab(value.alpha_rho_M0_mf,    value.momentum_bc, 2, nghost, "alpha_rho_M0",     true,  true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_M0_old_mf,value.momentum_bc, 2, nghost, "alpha_rho_M0_old", false, true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_E0_mf,    value.energy_bc,   1, nghost, "alpha_rho_E0",     true,  true);
            value.RegisterNewFab(value.alpha_rho_E0_old_mf,value.energy_bc,   1, nghost, "alpha_rho_E0_old", false, true);

            // Phase 1 (gas)
            value.RegisterNewFab(value.alpha_rho_1_mf,     value.density_bc,  1, nghost, "alpha_rho_1",      true,  true);
            value.RegisterNewFab(value.alpha_rho_1_old_mf, value.density_bc,  1, nghost, "alpha_rho_1_old",  false, true);
            value.RegisterNewFab(value.alpha_rho_M1_mf,    value.momentum_bc, 2, nghost, "alpha_rho_M1",     true,  true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_M1_old_mf,value.momentum_bc, 2, nghost, "alpha_rho_M1_old", false, true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_E1_mf,    value.energy_bc,   1, nghost, "alpha_rho_E1",     true,  true);
            value.RegisterNewFab(value.alpha_rho_E1_old_mf,value.energy_bc,   1, nghost, "alpha_rho_E1_old", false, true);

            // Volume fraction (advected, non-conservative). Distinct field from
            // eta_mf so the existing 5-eqn diagnostics keep working unchanged.
            value.RegisterNewFab(value.alpha0_mf,     value.energy_bc, 1, nghost, "alpha0",     true,  true);
            value.RegisterNewFab(value.alpha0_old_mf, value.energy_bc, 1, nghost, "alpha0_old", false, true);
        }
    }

    // INITIAL CONDITIONS
    // Eta
    pp.select_default<IC::Constant,IC::Laminate,IC::Expression,IC::BMP,IC::PNG>("eta.ic",value.eta_ic,value.geom);
    // Fluid 0
    pp.select_default<IC::Constant,IC::Expression>("velocity0.ic",      value.velocity0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure0.ic",      value.pressure0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density0.ic",       value.density0_ic,  value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("temperature0.ic",  value.temperature0_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("k0_thermal.ic",    value.k0_thermal_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h0_thermal_ic, value.geom);

    // Fluid 1
    pp.select_default<IC::Constant,IC::Expression>("velocity1.ic",      value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure1.ic",      value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density1.ic",       value.density1_ic,  value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("temperature1.ic",  value.temperature1_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("k1_thermal.ic",    value.k1_thermal_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h1_thermal_ic, value.geom);

    // DIFFUSE BOUNDARY SOURCES
    // diffuse boundary prescribed mass flux 
    pp.select_default<IC::Constant,IC::Expression>("m0.ic",value.ic_m0,value.geom);
    // diffuse boundary prescribed velocity
    pp.select_default<IC::Constant,IC::Expression>("u0.ic",value.ic_u0,value.geom);
    // diffuse boundary prescribed heat flux 
    pp.select_default<IC::Constant,IC::Expression>("q.ic",value.ic_q,value.geom);
    

    // SOLVERS
    // Riemann solver
    std::string solver_name;
    pp.query("Riemann_Solver.type", solver_name);
    Util::Message(INFO, "Input file has Riemann_Solver.type = ", solver_name);
    pp.select_default<Solver::Local::FluidRiemann::Roe,
                      Solver::Local::FluidRiemann::HLL,
                      Solver::Local::FluidRiemann::HLLE,
                      Solver::Local::FluidRiemann::HLLC,
                      Solver::Local::FluidRiemann::HLLCE,
                      //Solver::Local::FluidRiemann::HLLCE_WENO5, // Never verified but updated
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman, // Can't remember if this has been verified
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro,
                      Solver::Local::FluidRiemann::HLLC_LM,
                      Solver::Local::FluidRiemann::HLLC_ADC
                      //Solver::Local::FluidRiemann::Upwind,
                      //Solver::Local::FluidRiemann::Lax_Friedrich
    >("Riemann_Solver", value.riemannsolver);
    Util::Message(INFO, "Selected Riemann solver: ", typeid(*value.riemannsolver).name());

    // Low-order, positivity-preserving flux for the PP flux limiter. Always an
    // HLL solver (Davis wave speeds), independent of the selected high-order
    // solver; only consumed when pp_flux_limiter != 0.
    value.hll_solver = new Solver::Local::FluidRiemann::HLL();

    // Mirror HLLC-ADC's exponent so the omega_adc_mf diagnostic uses the same
    // sensor sharpness as the active solver. Defaults to 3.0 if not set.
    {
        // HLLC-ADC reads its parameters under the fully-qualified prefix
        // "Riemann_Solver.hllc_adc" (select_default appends ".<type>"); mirror
        // that here so the omega_adc diagnostic uses the solver's actual alpha.
        IO::ParmParse pp_rs("Riemann_Solver.hllc_adc");
        pp_rs.query_default("alpha", value.omega_adc_alpha, 3.0);
    }


    // LIMITERS
    pp_query_default("Limiter", value.Limiter, 0); // Type of solver
    if (value.Limiter == 0)
    {
        // No Limiter
    }
    else if (value.Limiter == 1)
    {
        // pp.select_default<Solver::Local::Limiter::Minmod>("Limiter", value.limiter_minmod);
    }
    else if (value.Limiter == 2)
    {
        // pp.select_default<Solver::Local::Limiter::VanLeer>("Limiter", value.limiter_vanleer);
    }
    else
    {
        Util::ParallelMessage(INFO, "-------------------------------");
        Util::ParallelMessage(INFO, "Invalid Limiter: ", value.Limiter);
        Util::ParallelMessage(INFO, "Acceptable Methods: ");
        Util::ParallelMessage(INFO, "None       : 0");
        Util::ParallelMessage(INFO, "MinMod     : 1");
        Util::ParallelMessage(INFO, "Van Leer   : 2");
        Util::Abort(INFO);
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");

    // Initialize step counter (shared between 5-eqn and 7-eqn paths)
    if (step_counter.size() <= (size_t)lev)
    {
        step_counter.resize(lev + 1, 0);
    }
    step_counter[lev] = 0;

    // Ensure flux register and face-flux vectors are large enough.
    if ((int)flux_reg.size() <= lev)
        flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev)
        cc_fluxes.resize(lev + 1);
    if ((int)pp_scratch.size() <= lev)
        pp_scratch.resize(lev + 1);

    // Initialize individual fluid variables
    // DIFFUSIVE BOUNDRY
    eta_ic          ->Initialize(lev, eta_mf, 0.0);
    eta_ic          ->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]  ->setVal(0.0);
    hess_eta_mf[lev]->setVal(0.0);

    // FLUID 0
    velocity0_ic    ->Initialize(lev, velocity0_mf, 0.0);
    pressure0_ic    ->Initialize(lev, pressure0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_old_mf, 0.0);
    //temperature0_ic ->Initialize(lev, T0_mf, 0.0);
    //k0_thermal_ic   ->Initialize(lev, k0_thermal_mf, 0.0);
    //h0_thermal_ic   ->Initialize(lev, h0_thermal_mf, 0.0);

    // FLUID 1
    velocity1_ic    ->Initialize(lev, velocity1_mf, 0.0);
    pressure1_ic    ->Initialize(lev, pressure1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_old_mf, 0.0);
    //temperature1_ic ->Initialize(lev, T1_mf, 0.0);
    //k1_thermal_ic   ->Initialize(lev, k1_thermal_mf, 0.0);
    //h1_thermal_ic   ->Initialize(lev, h1_thermal_mf, 0.0);

    // FORCED SOURCE
    ic_m0           ->Initialize(lev, m0_mf, 0.0);
    ic_u0           ->Initialize(lev, u0_mf, 0.0);
    ic_q            ->Initialize(lev, q_mf, 0.0);

    // FILLING GHOST CELLS
    rho_eta0_mf[lev]->setVal(0.0);
    rho_eta1_mf[lev]->setVal(0.0);
    rho_eta0_old_mf[lev]->setVal(0.0);
    rho_eta1_old_mf[lev]->setVal(0.0);

    energy_per_vol_mf[lev]->setVal(0.0);
    energy_per_mas_mf[lev]->setVal(0.0);
    energy_per_vol_old_mf[lev]->setVal(0.0);
    energy_per_mas_old_mf[lev]->setVal(0.0);

    UE_per_vol_mf[lev]->setVal(0.0);
    UE_per_mas_mf[lev]->setVal(0.0);
    KE_per_vol_mf[lev]->setVal(0.0);
    KE_per_mas_mf[lev]->setVal(0.0);

    // Calculate mixed variables based on individual fluid variables
    Mix(lev);

    // Zero Common Fields
    ZeroDerivedScratchFields(lev);

    // Initialize Riemander
    a_mf[lev]           ->setVal(0.0);
    Ma_mf[lev]          ->setVal(0.0);
    UE_per_vol_mf[lev]  ->setVal(0.0);
    UE_per_mas_mf[lev]  ->setVal(0.0);
    KE_per_vol_mf[lev]  ->setVal(0.0);
    KE_per_mas_mf[lev]  ->setVal(0.0);

    // Use BoxArray/dmap from a registered MultiFab (density_mf) because
    // grids[lev]/dmap[lev] may not yet be set when Initialize runs.
    const amrex::BoxArray& ba_init = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm_init = density_mf[lev]->DistributionMap();

    // Build FluxRegister for this level if it is a fine level.
    if (lev > 0)
    {
        int ncomp_reflux = 2 + AMREX_SPACEDIM + 1; // rho0 + rho1 + momentum + energy
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba_init, dm_init, refRatio(lev - 1), lev, ncomp_reflux);
    }

    // Cell-centered flux MultiFabs: cell (i,j) stores the hi-face Riemann flux.
    // 1 ghost cell so FillBoundary can propagate fluxes across box boundaries
    // for the cell-to-face conversion in Advance().
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 2, 1);
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba_init, dm_init, AMREX_SPACEDIM, 1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 1, 1);

        cc_fluxes[lev].mass[d]->setVal(0.0);
        cc_fluxes[lev].mom[d]->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
    }

    // PP flux-limiter scratch (see LimiterScratch). 1 ghost like cc_fluxes.
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        pp_scratch[lev].Fhi[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 5, 1);
        pp_scratch[lev].Flo[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 5, 1);
        pp_scratch[lev].Fhi[d]->setVal(0.0);
        pp_scratch[lev].Flo[d]->setVal(0.0);
    }
    pp_scratch[lev].Bbase = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 6, 1);
    pp_scratch[lev].theta = std::make_unique<amrex::MultiFab>(ba_init, dm_init, AMREX_SPACEDIM, 1);
    pp_scratch[lev].Bbase->setVal(0.0);
    pp_scratch[lev].theta->setVal(0.0);

    Util::ParallelMessage(INFO, "Finished initialization, begginning time iteration");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();
    
    // Function is for the diffusive mixing terms. I.E: rho = eta*rho0 + (1-eta)*rho1
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0_old = rho_eta0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1_old = rho_eta1_old_mf.Patch(lev, mfi);

        // FLUID 0
        Set::Patch<const Set::Scalar>   v0          = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p0          = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0        = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0_old    = density0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0          = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0_old      = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E0          = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E0_old      = energy0_old_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   T0          = T0_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   k0_thermal  = k0_thermal_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   h0_thermal  = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E1          = energy1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E1_old      = energy1_old_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   T1          = T1_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   k1_thermal  = k1_thermal_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   h1_thermal  = h1_thermal_mf.Patch(lev, mfi);

        // MIXTURE 
        Set::Patch<Set::Scalar>         v           = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         press       = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho         = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho_old     = density_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M           = momentum_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M_old       = momentum_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E_vol       = energy_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E_mas       = energy_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E_vol_old   = energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E_mas_old   = energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         T           = T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cp          = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cv          = cv_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar>         k_thermal   = k_thermal_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar>         h_thermal   = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         gammaf      = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         p0_eff      = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         mu_chem_    = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Bm          = Bm_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Y           = Y_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar>         a           = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Ma          = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_vol      = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_mas      = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_vol      = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_mas      = KE_per_mas_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Derivative Function Calls
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // Calculate State Variables 
            rho(i, j, k) = eta(i, j, k) * rho0(i, j, k) + (1.0 - eta(i, j, k)) * rho1(i, j, k);
            //rho(i, j, k) = 1.0 / (eta(i, j, k) / (rho0(i, j, k)) + (1.0 - eta(i, j, k)) / (rho1(i, j, k)));
            rho_old(i, j, k) = rho(i, j, k);  

            rho_eta0(i, j, k) = rho(i, j, k) * eta(i, j, k);
            rho_eta1(i, j, k) = rho(i, j, k) * (1.0 - eta(i, j, k));

            rho_eta0_old(i, j, k) = rho_eta0(i, j, k);
            rho_eta1_old(i, j, k) = rho_eta1(i, j, k);

            M(i, j, k, 0) = (rho0(i, j, k) * v0(i, j, k, 0)) * eta(i, j, k) + (rho1(i, j, k) * v1(i, j, k, 0)) * (1.0 - eta(i, j, k));
            M(i, j, k, 1) = (rho0(i, j, k) * v0(i, j, k, 1)) * eta(i, j, k) + (rho1(i, j, k) * v1(i, j, k, 1)) * (1.0 - eta(i, j, k));
            M_old(i, j, k, 0) = M(i, j, k, 0);
            M_old(i, j, k, 1) = M(i, j, k, 1);

            // Kinetic Energy
            KE_vol(i, j, k) = (0.5 * ((v0(i, j, k, 0) * v0(i, j, k, 0)) + (v0(i, j, k, 1) * v0(i, j, k, 1))) * rho0(i, j, k)) * eta(i, j, k)
                            + (0.5 * ((v1(i, j, k, 0) * v1(i, j, k, 0)) + (v1(i, j, k, 1) * v1(i, j, k, 1))) * rho1(i, j, k)) * (1.0 - eta(i, j, k));
            KE_mas(i, j, k) = (0.5 * ((v0(i, j, k, 0) * v0(i, j, k, 0)) + (v0(i, j, k, 1) * v0(i, j, k, 1)))) * eta(i, j, k)
                            + (0.5 * ((v1(i, j, k, 0) * v1(i, j, k, 0)) + (v1(i, j, k, 1) * v1(i, j, k, 1)))) * (1.0 - eta(i, j, k));

            // Internal Energy
            Set::Scalar p_eff = p0(i, j, k) * (eta(i, j, k)) + p1(i, j, k) * (1.0 - eta(i, j, k));
            UE_vol(i, j, k) = Solver::EOS::EOS::MixedInternalEnergy(p_eff, eta(i, j, k), eos0_local, eos1_local, pref, small);
            UE_mas(i, j, k) = UE_vol(i, j, k) / rho(i, j, k);
            
            // Kinetic Energy
            E_vol(i, j, k) = KE_vol(i, j, k) + UE_vol(i, j, k);
            E_vol_old(i, j, k) = E_vol(i, j, k);
            E_mas(i, j, k) = KE_mas(i, j, k) + UE_mas(i, j, k);
            E_mas_old(i, j, k) = E_mas(i, j, k);

            // Initialize extra fields - (not directly used to solve)
            // Velocity
            v(i, j, k, 0) = v0(i, j, k, 0) * eta(i, j, k) + v1(i, j, k, 0) * (1.0 - eta(i, j, k));
            v(i, j, k, 1) = v0(i, j, k, 1) * eta(i, j, k) + v1(i, j, k, 1) * (1.0 - eta(i, j, k));

            // Specific Heat Ratio
            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);

            // Pressure
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y))
            Bm(i, j, k) = SpaldingBM(Y(i, j, k), Y_infinity, small);

            // Temperature
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Thermal Conductivity
            //k_thermal(i, j, k) = eta(i, j, k) * k0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * k1_thermal(i, j, k);

            // Thermal Convectivity
            //h_thermal(i, j, k) = eta(i, j, k) * h0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * h1_thermal(i, j, k);

            // Speed of Sound
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gammaf(i, j, k), p0_eff(i, j, k), small);

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / a(i, j, k);
            Ma(i, j, k, 1) = v(i, j, k, 1) / a(i, j, k);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(0, lev, i, j, k, "ERROR IN Mix(): Primative Field Calculation", {
                { "rho_eta0", rho_eta0(i, j, k) }, 
                { "rho_eta1", rho_eta1(i, j, k) }, 
                { "rho", rho(i, j, k) }, 
                { "M[0]", M(i, j, k, 0) }, 
                { "M[1]", M(i, j, k, 1) }, 
                { "E_vol", E_vol(i, j, k) }, 
                { "E_mas", E_mas(i, j, k) }, 
                { "UE_vol", UE_vol(i, j, k) }, 
                { "UE_mas", UE_mas(i, j, k) }, 
                { "KE_vol", KE_vol(i, j, k) }, 
                { "KE_mas", KE_mas(i, j, k) }, 
                { "eta", eta(i, j, k) }, 
                { "gammaf", gammaf(i, j, k) }, 
                { "v[0]", v(i, j, k, 0) }, 
                { "v[1]", v(i, j, k, 1) }, 
                { "press", press(i, j, k) },
                { "p0_eff", p0_eff(i, j, k) },
                { "T", T(i, j, k) },
                { "a", a(i, j, k) },
                { "Y", Y(i, j, k) },
                { "Bm", Bm(i, j, k) }
            }); // end check4nans

        });
    }
    c_max = 0.0;
    vx_max = 0.0;
    vy_max = 0.0;
    F_max = 0.0;
    rho_min = 1e10;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// TIMESTEPBEGIN ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::TimeStepBegin(Set::Scalar, int /*iter*/)
{

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// TIMESTEPCOMPLETE ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::TimeStepComplete(Set::Scalar time, int lev)
{
    if (dynamictimestep.on)
    {
        Integrator::DynamicTimestep_Update();
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// EquationOfState ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/*
void Hydro2::EquationOfState(Set::Scalar time, int lev)
{
    // Tammann-EOS Stiffened Gas Equation of State
}
*/


///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// SurfaceTemsion ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
/*
void Hydro2::SurfaceTension(Set::Scalar time, int lev)
{
}
*/


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// RHS /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RHS(int lev,
    Set::Scalar time,
    Set::Scalar dt,
    amrex::MultiFab &rho_eta0_rhs_mf,
    amrex::MultiFab &rho_eta1_rhs_mf,
    amrex::MultiFab &M_rhs_mf,
    amrex::MultiFab &E_rhs_mf,
    amrex::MultiFab &eta_rhs_mf,
    const amrex::MultiFab &rho_eta0_mf_in,
    const amrex::MultiFab &rho_eta1_mf_in,
    const amrex::MultiFab &M_mf_in,
    const amrex::MultiFab &E_mf_in,
    const amrex::MultiFab &eta_mf_in)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Converting Array to mf
    amrex::MultiFab::Copy(*rho_eta0_mf[lev], rho_eta0_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*rho_eta1_mf[lev], rho_eta1_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*momentum_mf[lev], M_mf_in, 0, 0, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*eta_mf[lev], eta_mf_in, 0, 0, 1, 0);

    // Eta Fields
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }
    
    // Primitive Fields (with BC)
    FillGhost4BC(lev, time);

    // Curvature pipeline (smooth-normals method fills kappas_mf before the RHS loop)
    if (kappa_method == 1)
        ComputeKappas(lev);

    // Pre-Source Terms
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto const rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        auto const M = momentum_mf[lev]->array(mfi);
        auto const E = energy_per_vol_mf[lev]->array(mfi);

        // PRIMITIVE
        Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> a = a_mf.Patch(lev, mfi);

        // SINGLE PHASE
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);

        // SOURCE - ish
        Set::Patch<const Set::Scalar> KE = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> UE = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T = T_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Y = Y_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Derivative Function Calls
            // Normal Compute
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            //Set::Scalar mu_chem = -epsilon * lap_eta + f_prime / epsilon;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y))
            Bm(i, j, k) = SpaldingBM(Y(i, j, k), Y_infinity, small);

            // Curvature (legacy Hessian-based method — stored in component 2 for diagnostics)
            if (kappa_method != 1)
            {
                Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
                Set::Vector t1;
                if (abs(n_hat(0)) > abs(n_hat(1)))
                    t1 = Set::Vector(-n_hat(1), n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                else
                    t1 = Set::Vector(n_hat(1), -n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);

                Set::Scalar kappa1 = -n_hat.dot(hess_eta * n_hat);
                Set::Scalar kappa2 = -t1.dot(hess_eta * t1) * 2.0 * epsilon;
                kappas(i, j, k, 0) = kappa2;
                kappas(i, j, k, 1) = kappa1;
                kappas(i, j, k, 2) = kappa2;
            }

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN RHS(): Primative Field Calculation", {
                { "rho_eta0", rho_eta0(i, j, k) }, 
                { "rho_eta1", rho_eta1(i, j, k) }, 
                { "rho", rho(i, j, k) }, 
                { "M[0]", M(i, j, k, 0) }, 
                { "M[1]", M(i, j, k, 1) }, 
                { "E", E(i, j, k) }, 
                { "eta", eta(i, j, k) }, 
                { "gammaf", gammaf(i, j, k) }, 
                { "v[0]", v(i, j, k, 0) }, 
                { "v[1]", v(i, j, k, 1) }, 
                { "KE", KE(i, j, k) }, 
                { "UE", UE(i, j, k) }, 
                { "press", press(i, j, k) },
                { "p0_eff", p0_eff(i, j, k) },
                { "T", T(i, j, k) },
                { "a", a(i, j, k) },
                { "Y", Y(i, j, k) },
                { "Bm", Bm(i, j, k) },
                { "mu_chem", mu_chem_(i, j, k) },
                { "grad_eta[0]", grad_eta(0) },
                { "grad_eta[1]", grad_eta(1) },
                { "grad_eta_mag", grad_eta_mag },
                { "lap_eta", lap_eta },
                { "hess_eta[0,0]", hess_eta(0, 0) },
                { "hess_eta[0,1]", hess_eta(0, 1) },
                { "hess_eta[1,0]", hess_eta(1, 0) },
                { "hess_eta[1,1]", hess_eta(1, 1) },
                { "kappas[0]", kappas(i, j, k, 0) },
                { "kappas[1]", kappas(i, j, k, 1) },
                { "kappas[2]", kappas(i, j, k, 2) }
            }); // end check4nans

        }); // end parallelfor
    } // end Primative field

    // Fill mu_chem ghost cells with zero-neumann BCs before the main loop
    // computes lap(mu_chem) via central differences. Must use eta_bc
    // (zero-neumann), NOT energy_bc — energy_bc has dirichlet values
    // (e.g. 86M at xlo) that would create a massive spurious lap(mu_chem)
    // and corrupt eta across the entire domain via the CH term.
    FillBoundariesWithBC(lev, time, eta_bc, { mu_chem_mf[lev].get() });

    // ------------------------------------------------------------
    // HLLC-ADC carbuncle shock tag: a Ducros-weighted div(u) compression
    // sensor, thresholded per cell then grown by adc_grow cells. Below it
    // drives a per-face omega (0 = full HLL on the ADC-controlled components,
    // 1 = full HLLC) into the Riemann states, replacing the solver's
    // pressure-ratio sensor. Carbuncles are pressure-quiet, so a div(u) tag
    // plus dilation lands the HLL dissipation on the post-shock band where the
    // ripple lives, not just on the thin pressure jump. adc_grow < 0 disables
    // the tag and lets the solver use its internal pressure sensor.
    // ------------------------------------------------------------
    const bool use_shock_tag = (adc_grow >= 0);
    const int  tag_grow      = std::max(adc_grow, 0);
    amrex::MultiFab shock_tag_local(velocity_mf[lev]->boxArray(),
                                    velocity_mf[lev]->DistributionMap(), 1, 1);
    shock_tag_local.setVal(0.0);
    shock_sensor_mf[lev]->setVal(0.0);
    pp_theta_mf[lev]->setVal(0.0);
    pp_dmass_mf[lev]->setVal(0.0);
    if (use_shock_tag)
    {
        amrex::MultiFab shock_raw(velocity_mf[lev]->boxArray(),
                                  velocity_mf[lev]->DistributionMap(), 1, tag_grow + 1);
        shock_raw.setVal(0.0);

        const Set::Scalar dr      = std::sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));
        const Set::Scalar thresh  = adc_shock_threshold;
        const Set::Scalar small_l = small;

        // Raw shock indicator per valid cell.
        for (amrex::MFIter mfi(shock_raw, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& sbx = mfi.validbox();
            auto raw = shock_raw.array(mfi);
            auto vv  = velocity_mf[lev]->array(mfi);
            auto aa  = a_mf[lev]->array(mfi);
            auto sensor_out = shock_sensor_mf[lev]->array(mfi);
            amrex::ParallelFor(sbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto sten = Numeric::GetStencil(i, j, k, domain);
                Set::Matrix grad_u = Numeric::Gradient(vv, i, j, k, DX, sten);
                Set::Scalar div_u = grad_u.trace();
                Set::Scalar comp = std::max(-div_u, 0.0);
#if AMREX_SPACEDIM == 2
                Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
                Set::Scalar curl_mag2 = wz * wz;
#else
                Set::Scalar wx = grad_u(2, 1) - grad_u(1, 2);
                Set::Scalar wy = grad_u(0, 2) - grad_u(2, 0);
                Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
                Set::Scalar curl_mag2 = wx * wx + wy * wy + wz * wz;
#endif
                Set::Scalar ducros = comp / std::sqrt(div_u * div_u + curl_mag2 + small_l * small_l);
                Set::Scalar c = std::max(aa(i, j, k), small_l);
                Set::Scalar sensor = ducros * comp * dr / c;
                sensor_out(i, j, k) = sensor;
                raw(i, j, k) = (sensor > thresh) ? 1.0 : 0.0;
            });
        }
        shock_raw.FillBoundary(geom[lev].periodicity());

        // Morphological dilation: tag a cell if any raw cell within tag_grow is set.
        for (amrex::MFIter mfi(shock_tag_local, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& sbx = mfi.validbox();
            auto tag = shock_tag_local.array(mfi);
            auto raw = shock_raw.array(mfi);
            const int g = tag_grow;
            amrex::ParallelFor(sbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar m = 0.0;
                for (int dj = -g; dj <= g; dj++)
                    for (int di = -g; di <= g; di++)
#if AMREX_SPACEDIM == 3
                        for (int dk = -g; dk <= g; dk++)
                            m = std::max(m, raw(i + di, j + dj, k + dk));
#else
                        m = std::max(m, raw(i + di, j + dj, k));
#endif
                tag(i, j, k) = m;
            });
        }
        shock_tag_local.FillBoundary(geom[lev].periodicity());
    }

    // Main time integration loop
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        // PRIMARY FLUIDS
        // FLUID 0
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto const rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        auto const M = momentum_mf[lev]->array(mfi);
        auto const E = energy_per_vol_mf[lev]->array(mfi);

        // OUTPUTS
        Set::Patch<Set::Scalar> rho_eta0_rhs = rho_eta0_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_eta1_rhs = rho_eta1_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs       = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs       = E_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> eta_rhs     = eta_rhs_mf.array(mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);

        // Cell-centered flux arrays for reflux at coarse-fine boundaries.
        // cc_fluxes use density_mf's BA/dmap, so mfi access is compatible.
        bool have_cc_fluxes = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y;
        amrex::Array4<Set::Scalar> ff_mom_x,  ff_mom_y;
        amrex::Array4<Set::Scalar> ff_ene_x,  ff_ene_y;
        if (have_cc_fluxes)
        {
            ff_mass_x = cc_fluxes[lev].mass[0]->array(mfi);
            ff_mass_y = cc_fluxes[lev].mass[1]->array(mfi);
            ff_mom_x  = cc_fluxes[lev].mom[0]->array(mfi);
            ff_mom_y  = cc_fluxes[lev].mom[1]->array(mfi);
            ff_ene_x  = cc_fluxes[lev].energy[0]->array(mfi);
            ff_ene_y  = cc_fluxes[lev].energy[1]->array(mfi);
        }

        // PP flux-limiter scratch: store the selected (high-order) and HLL
        // (low-order) hi-face fluxes for use in the limiter passes below.
        const bool pp_on = (pp_flux_limiter != 0 && lev < (int)pp_scratch.size()
                            && pp_scratch[lev].Fhi[0]);
        amrex::Array4<Set::Scalar> Fhi_x, Fhi_y, Flo_x, Flo_y;
        if (pp_on)
        {
            Fhi_x = pp_scratch[lev].Fhi[0]->array(mfi);
            Fhi_y = pp_scratch[lev].Fhi[1]->array(mfi);
            Flo_x = pp_scratch[lev].Flo[0]->array(mfi);
            Flo_y = pp_scratch[lev].Flo[1]->array(mfi);
        }

        Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> a = a_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T = T_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Y = Y_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fsv = Fsv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fw = Fw_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ldot_ = Ldot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Vap_dot = Vap_dot_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q0 = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        // DEBUGGING PLOTS
        Set::Patch<const Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = hess_u_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> omega_adc = omega_adc_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> flux_dev = flux_dev_mf.Patch(lev, mfi);
        const Set::Scalar omega_alpha_local = omega_adc_alpha;
        amrex::Array4<const Set::Scalar> stag = shock_tag_local.const_array(mfi);

        const auto bx_lo = amrex::lbound(bx);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);

            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            // Extract velocity from momentum and density
            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Vector u0 = Set::Vector(_u0(i, j, k, 0), _u0(i, j, k, 1));

            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));

            Set::Vector q0_ = Set::Vector(q0(i, j, k, 0), q0(i, j, k, 1));

            /// Calculate Source Terms
            // Shear:
            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0_.dot(grad_eta);

            Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
            Set::Matrix3 hess_u = Set::Matrix3::Zero();

            for (int p = 0; p < 2; p++)
                for (int q = 0; q < 2; q++)
                    for (int r = 0; r < 2; r++)
                    {
                        hess_u(r, p, q) = (hess_M(r, p, q) - gradu(r, q) * gradrho(p) - gradu(r, p) * gradrho(q) - u(r) * hess_rho(p, q))
                                          / (rho(i, j, k));
                    }
            
            // WIP: Debugging feild for hess_u
            hess_u_(i, j, k, 0) = hess_u(0, 0, 0);
            hess_u_(i, j, k, 1) = hess_u(0, 0, 1);
            hess_u_(i, j, k, 2) = hess_u(0, 1, 0);
            hess_u_(i, j, k, 3) = hess_u(0, 1, 1);
            hess_u_(i, j, k, 4) = hess_u(1, 0, 0);
            hess_u_(i, j, k, 5) = hess_u(1, 0, 1);
            hess_u_(i, j, k, 6) = hess_u(1, 1, 0);
            hess_u_(i, j, k, 7) = hess_u(1, 1, 1);

            // ------------------------------------------------------------
            // Divergence of Stress
            // ------------------------------------------------------------
            Set::Vector Ldot = Set::Vector::Zero();
            Set::Vector div_tau = Set::Vector::Zero();

            // Effective Viscosities
            Set::Scalar mu_eff = eta(i, j, k) * mu0 + (1.0 - eta(i, j, k)) * mu1;
            Set::Scalar lambda_eff = eta(i, j, k) * mu0_b + (1.0 - eta(i, j, k)) * mu1_b;
            Set::Vector grad_mu = (mu0 - mu1) * grad_eta;
            Set::Vector grad_lambda = (mu0_b - mu1_b) * grad_eta;

            // Solving
            for (int p = 0; p < 2; p++)             // i
                for (int q = 0; q < 2; q++)         // j
                    for (int r = 0; r < 2; r++)     // k
                        for (int s = 0; s < 2; s++) // l
                        {
                            Set::Scalar Mpqrs = 0.0;
                            Set::Scalar dMpqrs = 0.0;
                            if ((p == r) and (q == s))
                            {
                                Mpqrs += mu_eff;
                                dMpqrs += grad_mu(q);
                            }
                            if ((p == s) and (q == r))
                            {
                                Mpqrs += mu_eff;
                                dMpqrs += grad_mu(q);
                            }
                            if ((p == q) and (r == s))
                            {
                                Mpqrs += lambda_eff - (2.0 / 3.0) * mu_eff;
                                dMpqrs += grad_lambda(q) - (2.0 / 3.0) * grad_mu(q);
                            }

                            div_tau(p) += Mpqrs * hess_u(r, s, q);
                            Ldot(p) += 0.5 * Mpqrs * (u(r) - u0(r)) * hess_eta(q, s);

                            // Grad visc terms
                            div_tau(p) += dMpqrs * gradu(r, s);
                        }

            // Debugging feild for div_tau and Ldot
            div_tau_(i, j, k, 0) = div_tau(0);
            div_tau_(i, j, k, 1) = div_tau(1);
            Ldot_(i, j, k, 0) = Ldot(0);
            Ldot_(i, j, k, 1) = Ldot(1);

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Viscosity solving", {
                { "Ldot[0]", Ldot(0) },
                { "Ldot[1]", Ldot(1) },
                { "div_tau[0]", div_tau(0) },
                { "div_tau[1]", div_tau(1) }
            }); // end check4nans
            
            // ------------------------------------------------------------
            // Surface Tension
            // ------------------------------------------------------------
            // Fsv =  simga * kappa * grad_eta
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);
            if (apply_surface_tension)
            {
                // Optimization, only calc surface tension if on interface
                if (grad_eta_mag > 0.0)
                {
                    Set::Scalar kappa = kappas(i, j, k, 0);
                    Set::Scalar sigma_eff = sigma;
                    Set::Scalar alpha = 6 * sqrt(2);
                    //Set::Scalar UFFDA = epsilon * alpha * grad_eta_mag * grad_eta_mag;                // What I oringially had, did not reach max amplitude
                    //Set::Scalar UFFDA = grad_eta_mag * grad_eta_mag / (epsilon * sqrt(2.0));          // Forces an interface thickness
                    //Set::Scalar UFFDA = grad_eta_mag * grad_eta_mag;                                  // More natural
                    Set::Scalar UFFDA = epsilon * grad_eta_mag / 0.02;           // Working the best
                    //Set::Scalar UFFDA = grad_eta_mag;           // More natural
                    //Fsv_vector(0) = sigma_eff * kappa * n_hat(0) * UFFDA;    // / (grad_eta_mag + small)); // / (DX[0] + small);
                    //Fsv_vector(1) = sigma_eff * kappa * n_hat(1) * UFFDA; // / (grad_eta_mag + small)); // / (DX[1] + small);
                    
                    Fsv_vector(0) = sigma_eff * kappa * grad_eta(0); // * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
                    Fsv_vector(1) = sigma_eff * kappa * grad_eta(1); // * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
                } 
            }
            Fsv(i, j, k, 0) = Fsv_vector(0);
            Fsv(i, j, k, 1) = Fsv_vector(1);

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fsv_vector[0]", Fsv_vector(0) },
                { "Fsv_vector[1]", Fsv_vector(1) }
            }); // end check4nans
            

            // ------------------------------------------------------------
            // Weight
            // ------------------------------------------------------------
            // Fw = - rho * g
            Set::Vector Fw_vector = Set::Vector(0.0, 0.0);
            if (apply_weight)
            {
                Fw_vector(0) = 0.0;
                Fw_vector(1) = -rho(i, j, k) * g;
            }
            Fw(i, j, k, 0) = Fw_vector(0);
            Fw(i, j, k, 1) = Fw_vector(1);

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Weight solving", {
                { "Fw_vector[0]", Fw_vector(0) },
                { "Fw_vector[1]", Fw_vector(1) }
            }); // end check4nans

            // ------------------------------------------------------------
            // Conservative Allen-Cahn
            // ------------------------------------------------------------
            // d(eta)/dt = -u*grad(eta) + Mob * laplacian(mu)
            // Laplacian of Chemical Potential (conservative form)
            /*
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            Set::Scalar phi = eta(i, j, k); // Dummy Variable bc it gets UGGGLLLYYYYY
            Set::Scalar kappa = kappas(i, j, k, 0);
            Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0]; // Mob = u_max * epsilon,  (epsilon = 0.7*DX) Chiu & Lin (2011)
            Set::Scalar advection = -u.dot(grad_eta);
            //Set::Scalar diffusion = Mob * lap_mu_chem; // Allen-Cahn Alternative - it has to be curve fit
            Set::Scalar diffusion = Mob * ( lap_eta - ((phi * (1.0-phi) * (1.0-2.0*phi)) / (epsilon*epsilon + small)) - (grad_eta_mag * kappa) ); // Chiu & Lin (2011) URL: https://www.sciencedirect.com/science/article/pii/S0021999110005243
            Set::Scalar eta_dot_AC = advection + diffusion;
            */

            // ------------------------------------------------------------
            // Cahn-Hilliard
            // ------------------------------------------------------------
            // Dr. Q mobility scaling:
            //   Pe_CH = (U / epsilon) / (M * sigma / epsilon^3)
            //         = U * epsilon^2 / (M * sigma).
            // Setting Pe_CH = O(1) gives M = M0 * epsilon^2 with
            // M0 = O(U / sigma).  The chemical potential stored here is
            // mu = f'(eta) - epsilon^2 lap(eta), so the dimensional
            // coefficient multiplying lap(mu) is M * sigma / epsilon.
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            Set::Scalar M0_CH = Mob_user;
            Set::Scalar M_CH = M0_CH * epsilon * epsilon;
            Set::Scalar Mob = M_CH * sigma / epsilon;
            Set::Scalar eta_dot_CH = Mob * lap_mu_chem;

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Cahn-Hillard solving", {
                { "mu_chem",  mu_chem_(i, j, k) },
                { "lap_mu_chem",  lap_mu_chem },
                { "a",  a(i, j, k) },
                { "M0_CH",  M0_CH },
                { "M_CH",  M_CH },
                { "Mob",  Mob },
                { "eta_dot_CH",  eta_dot_CH }
            }); // end check4nans

            // ------------------------------------------------------------
            // Vaporization
            // ------------------------------------------------------------
            // Spalding Vaporization
            // NOTE: rho0 (OR eta = 1) SHOULD BE THE GAS PHASE
            Set::Scalar eta_dot_Vap = 0.0;
            Set::Scalar m_dot_Vap = 0.0;
            Set::Scalar E_dot_Vap = 0.0;
            if (apply_vaporization == 1)
            {
                //m_dot_Vap = (rho0(i, j, k) * Dv * (Bm(i, j, k) / (1.0 + Bm(i, j, k) + small)) * grad_eta_mag);
                //eta_dot_Vap += (1.0 / (rho(i, j, k) * epsilon)) * m_dot_vap;

                // Interface temperature - use gas side temperature
                /*
                Set::Scalar T_s = T(i, j, k);         // Interface temperature [K]
                Set::Scalar T_celsius = T_s - 273.15; // Convert to Celsius for Antoine equation

                // Saturation pressure from Antoine equation
                // log10(p_sat[mmHg]) = A - B/(C + T[*C])
                Set::Scalar log10_psat_mmHg = vap_Antoine_A - vap_Antoine_B / (vap_Antoine_C + T_celsius);
                Set::Scalar p_sat = std::pow(10.0, log10_psat_mmHg) * 133.322; // Convert mmHg to Pa

                // Mole fraction of vapor at interface (saturation)
                Set::Scalar x_vs = p_sat / (press(i, j, k) + small);
                x_vs = std::min(x_vs, 0.99); // Clamp to avoid numerical issues

                // Mass fraction of vapor at surface (Equation 5 of document):
                // Y_v,s = W_v * x_v,s / (W_v * x_v,s + W_g * (1 - x_v,s))
                Set::Scalar numerator_Y = vap_W_v * x_vs;
                Set::Scalar Y_vs = numerator_Y / (numerator_Y + vap_W_g * (1.0 - x_vs) + small);
                */

                // Mass fraction of vapor at surface
                Set::Scalar Y_vs = Y(i, j, k); // rho_eta0(i, j, k) / (rho0(i, j, k) + rho1(i, j, k));

                // Spalding mass transfer number
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // Gas density from fluid 0 (eta=1 corresponds to fluid 0)
                Set::Scalar rho_g = rho_eta0(i, j, k);

                // Mass-transfer rate (volumetric) -- simplified Spalding form.
                // m_dot_Vap = rho_g * D_v * (B_M / (1+B_M)) * |grad(eta)|     [kg/m^3/s]
                // (We keep the simplified (B_M/(1+B_M)) factor instead of ln(1+B_M)
                // and omit the Sherwood/length-scale prefactor for now -- see F-3.)
                m_dot_Vap = rho_g * Dv * (B_M / (1.0 + B_M + small)) * grad_eta_mag;

                // Volume-fraction source from phase change (F-4 fix).
                // Canonical 5-eq Allaire form (Saurel-Petitpas-Abgrall JFM 2008
                // Eq. 44; Le Metayer-Massoni-Saurel IJMF 2013 Eq. 28; Schmidmayer
                // JCP 2020 Eq. 8):
                //     D(alpha_gas)/Dt |_phase = m_dot_vol * (1/rho_l - 1/rho_g)
                // For evaporation (m_dot_Vap > 0) with rho_l > rho_g, this is
                // positive, so the gas volume fraction grows. Phase 0 = gas
                // (per the comment above); phase 1 = liquid in this code's
                // convention.
                Set::Scalar inv_rho_g = 1.0 / std::max(rho_eta0(i, j, k), small);
                Set::Scalar inv_rho_l = 1.0 / std::max(rho_eta1(i, j, k), small);
                eta_dot_Vap = m_dot_Vap * (inv_rho_l - inv_rho_g);

                // Energy "flux" diagnostic -- NOT applied to Source[3] (commented
                // out at line 1259-ish below). Kept inactive per F-6 ignore;
                // formula reproduced below without referencing the deleted
                // M_dot_Vap (F-5) so it stays compileable. Mathematically
                // identical to the previous u.dot(u * m_dot_Vap * |grad_eta|) * |grad_eta|.
                E_dot_Vap = m_dot_Vap * (u(0)*u(0) + u(1)*u(1)) * grad_eta_mag * grad_eta_mag;
            }

            // ------------------------------------------------------------
            // HRM bulk cavitation (Phase A: mass + void; latent heat deferred)
            // ------------------------------------------------------------
            // When the local pressure falls below saturation, convert liquid ->
            // vapor at a finite rate (Bilicki / Downar-Zapolski HRM): the gas phase
            // here IS n-dodecane vapor, so cavitation is just pressure-triggered
            // vaporization. Folds into the same m_dot_Vap/eta_dot_Vap, so the
            // partial-density update preserves action-reaction (gas +m, liquid -m).
            // Evaporation only (drive clamped >= 0; no condensation of the ambient
            // vapor) and it vanishes in pure gas since the rate scales with the
            // liquid partial density. The tension is relieved through the
            // eta-blended EOS (p0_eff drops as vapor forms); no latent-heat term is
            // applied yet (mixture energy held fixed -- Phase B, pending theory).
            if (apply_cavitation == 1)
            {
                Set::Scalar p_sat = std::pow(10.0, antoine_A - antoine_B / (T(i, j, k) + antoine_C)) * 1.0e5; // Antoine (bar) -> Pa
                Set::Scalar drive = (p_sat - press(i, j, k)) / (p_sat + small);
                drive = (drive < 0.0) ? 0.0 : ((drive > 1.0) ? 1.0 : drive);
                Set::Scalar m_dot_cav = (rho_eta1(i, j, k) / tau_cav) * drive; // liquid -> vapor [kg/m^3/s]
                Set::Scalar inv_rho_g = 1.0 / std::max(rho_eta0(i, j, k), small);
                Set::Scalar inv_rho_l = 1.0 / std::max(rho_eta1(i, j, k), small);
                m_dot_Vap   += m_dot_cav;
                eta_dot_Vap += m_dot_cav * (inv_rho_l - inv_rho_g);
            }

            // Vaporization Trackers (Vap_dot[2..3] used to hold M_dot_Vap which
            // has been removed per F-5; left as zero so the plotfile field
            // shape is preserved without changing the registration).
            Vap_dot(i, j, k, 0) = eta_dot_Vap;
            Vap_dot(i, j, k, 1) = m_dot_Vap;
            Vap_dot(i, j, k, 2) = 0.0;
            Vap_dot(i, j, k, 3) = 0.0;
            Vap_dot(i, j, k, 4) = E_dot_Vap;


            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv_vector(0) + Fw_vector(0),
                                                  Fsv_vector(1) + Fw_vector(1));
            
            Source(i, j, k, 0) = mdot0;

            // Momentum source from phase change is ZERO in 5-eq velocity-equilibrium
            // (vapor leaves at the local fluid velocity -- Saurel-Petitpas JFM 2008
            // Eq. 3). M_dot_Vap was removed per F-5.
            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force);// + E_dot_Vap;

            // Lagrange terms to enforce no-penetration
            Source(i, j, k, 1) = Source(i, j, k, 1) - lagrange * u.dot(grad_eta) * grad_eta(0);
            Source(i, j, k, 2) = Source(i, j, k, 2) - lagrange * u.dot(grad_eta) * grad_eta(1);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Source solving", { 
                { "Total_Force[0]",  Total_Force(0) },
                { "Total_Force[1]",  Total_Force(1) },
                { "Source[0]",  Source(i, j, k, 0) },
                { "Source[1]",  Source(i, j, k, 1) },
                { "Source[2]",  Source(i, j, k, 2) },
                { "Source[3]",  Source(i, j, k, 3) }
            }); // end check4nans

            // Riemann solver for mixed fluid
            const int X = 0, Y = 1;

            // Create arrays to store cell states for reconstruction
            std::vector<Solver::Local::FluidRiemann::State> x_states(3);
            std::vector<Solver::Local::FluidRiemann::State> y_states(3);

            // Fill the arrays with cell states
            x_states[0] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i - 1, j, k, X); // x_lo
            x_states[1] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i, j, k, X);     // x
            x_states[2] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i + 1, j, k, X); // x_hi

            y_states[0] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i, j - 1, k, Y); // y_lo
            y_states[1] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i, j, k, Y);     // y
            y_states[2] = Solver::Local::FluidRiemann::State(rho, M, E, gammaf, p0_eff, T, i, j + 1, k, Y); // y_hi

            // Variables to store reconstructed states at interfaces
            std::vector<Solver::Local::FluidRiemann::State> x_leftStates(3), x_rightStates(3);
            std::vector<Solver::Local::FluidRiemann::State> y_leftStates(3), y_rightStates(3);

            if (Limiter == 0)
            {
                // No limiter - use cell-centered values directly
                // x_leftStates.resize(3);
                // x_rightStates.resize(3);
                // y_leftStates.resize(3);
                // y_rightStates.resize(3);

                // For x-direction
                x_leftStates[1] = x_states[0];  // i-1/2
                x_rightStates[1] = x_states[1]; // i
                x_leftStates[2] = x_states[1];  // i
                x_rightStates[2] = x_states[2]; // i+1/2

                // For y-direction
                y_leftStates[1] = y_states[0];  // j-1/2
                y_rightStates[1] = y_states[1]; // j
                y_leftStates[2] = y_states[1];  // j
                y_rightStates[2] = y_states[2]; // j+1/2
            }
            else if (Limiter == 1)
            {
                // Minmod limiter
                // limiter_minmod->reconstructCharacteristicStates(x_states, x_leftStates, x_rightStates, pref, small);
                // limiter_minmod->reconstructCharacteristicStates(y_states, y_leftStates, y_rightStates, pref, small);
            }
            else if (Limiter == 2)
            {
                // Van Leer limiter
                // limiter_vanleer->reconstructCharacteristicStates(x_states, x_leftStates, x_rightStates, pref, small);
                // limiter_vanleer->reconstructCharacteristicStates(y_states, y_leftStates, y_rightStates, pref, small);
            }

            // Populate the perpendicular-neighbor pressures on each face state.
            // Consumed by shock-aware solvers (HLLC-ADC) for the multidim sensor;
            // ignored by all other solvers. For an x-face the perpendiculars are
            // y-neighbors; for a y-face they are x-neighbors.
            // x-face at i-1/2: L is cell (i-1,j), R is cell (i,j)
            x_leftStates[1].p_cell    = press(i - 1, j, k);
            x_leftStates[1].p_perp_lo = press(i - 1, j - 1, k);
            x_leftStates[1].p_perp_hi = press(i - 1, j + 1, k);
            x_rightStates[1].p_cell    = press(i, j, k);
            x_rightStates[1].p_perp_lo = press(i, j - 1, k);
            x_rightStates[1].p_perp_hi = press(i, j + 1, k);
            // x-face at i+1/2: L is cell (i,j), R is cell (i+1,j)
            x_leftStates[2].p_cell    = press(i, j, k);
            x_leftStates[2].p_perp_lo = press(i, j - 1, k);
            x_leftStates[2].p_perp_hi = press(i, j + 1, k);
            x_rightStates[2].p_cell    = press(i + 1, j, k);
            x_rightStates[2].p_perp_lo = press(i + 1, j - 1, k);
            x_rightStates[2].p_perp_hi = press(i + 1, j + 1, k);
            // y-face at j-1/2: L is cell (i,j-1), R is cell (i,j)
            y_leftStates[1].p_cell    = press(i, j - 1, k);
            y_leftStates[1].p_perp_lo = press(i - 1, j - 1, k);
            y_leftStates[1].p_perp_hi = press(i + 1, j - 1, k);
            y_rightStates[1].p_cell    = press(i, j, k);
            y_rightStates[1].p_perp_lo = press(i - 1, j, k);
            y_rightStates[1].p_perp_hi = press(i + 1, j, k);
            // y-face at j+1/2: L is cell (i,j), R is cell (i,j+1)
            y_leftStates[2].p_cell    = press(i, j, k);
            y_leftStates[2].p_perp_lo = press(i - 1, j, k);
            y_leftStates[2].p_perp_hi = press(i + 1, j, k);
            y_rightStates[2].p_cell    = press(i, j + 1, k);
            y_rightStates[2].p_perp_lo = press(i - 1, j + 1, k);
            y_rightStates[2].p_perp_hi = press(i + 1, j + 1, k);

            // Drive the HLLC-ADC blend from the div(u) shock tag: a face inside
            // the (dilated) tagged region gets omega = 0 (full HLL on the
            // controlled components), otherwise omega = 1 (full HLLC). Both
            // sides of a face carry the same value. When the tag is disabled
            // (adc_grow < 0) omega_ext stays -1 and the solver uses its
            // internal pressure-ratio sensor.
            if (use_shock_tag)
            {
                Set::Scalar tc  = stag(i, j, k);
                Set::Scalar txm = stag(i - 1, j, k);
                Set::Scalar txp = stag(i + 1, j, k);
                Set::Scalar tym = stag(i, j - 1, k);
                Set::Scalar typ = stag(i, j + 1, k);
                Set::Scalar om_xlo = (txm > 0.5 || tc > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_xhi = (tc > 0.5 || txp > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_ylo = (tym > 0.5 || tc > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_yhi = (tc > 0.5 || typ > 0.5) ? 0.0 : 1.0;
                x_leftStates[1].omega_ext = x_rightStates[1].omega_ext = om_xlo;
                x_leftStates[2].omega_ext = x_rightStates[2].omega_ext = om_xhi;
                y_leftStates[1].omega_ext = y_rightStates[1].omega_ext = om_ylo;
                y_leftStates[2].omega_ext = y_rightStates[2].omega_ext = om_yhi;
            }

            // ------------------------------------------------------------
            // HLLC-ADC shock locator diagnostic.
            // Compute omega at the cell's two hi-faces (i+1/2, j) and
            // (i, j+1/2) using the same 5-face pressure-ratio min that
            // HLLC_ADC consumes. Written every step so the locator can be
            // inspected regardless of which Riemann solver is selected.
            //   omega = ( min_k f_k )^alpha,
            //   f_k   = min(p_a^k, p_b^k) / max(p_a^k, p_b^k).
            // ------------------------------------------------------------
            {
                const Set::Scalar omega_small = 1e-30;
                auto omega_face_ratio = [&](Set::Scalar pa, Set::Scalar pb) -> Set::Scalar
                {
                    Set::Scalar pmin = std::min(pa, pb);
                    Set::Scalar pmax = std::max(pa, pb);
                    return pmin / (pmax + omega_small);
                };
                auto omega_at_face = [&](const Solver::Local::FluidRiemann::State& L,
                                         const Solver::Local::FluidRiemann::State& R) -> Set::Scalar
                {
                    Set::Scalar fm = omega_face_ratio(L.p_cell, R.p_cell);
                    if (L.p_cell > 0.0 && L.p_perp_lo > 0.0 && L.p_perp_hi > 0.0)
                    {
                        fm = std::min(fm, omega_face_ratio(L.p_perp_lo, L.p_cell));
                        fm = std::min(fm, omega_face_ratio(L.p_cell,    L.p_perp_hi));
                    }
                    if (R.p_cell > 0.0 && R.p_perp_lo > 0.0 && R.p_perp_hi > 0.0)
                    {
                        fm = std::min(fm, omega_face_ratio(R.p_perp_lo, R.p_cell));
                        fm = std::min(fm, omega_face_ratio(R.p_cell,    R.p_perp_hi));
                    }
                    return std::pow(fm, omega_alpha_local);
                };
                if (use_shock_tag)
                {
                    // Report the tag-based omega actually fed to the solver.
                    omega_adc(i, j, k, 0) = (stag(i, j, k) > 0.5 || stag(i + 1, j, k) > 0.5) ? 0.0 : 1.0;
                    omega_adc(i, j, k, 1) = (stag(i, j, k) > 0.5 || stag(i, j + 1, k) > 0.5) ? 0.0 : 1.0;
                }
                else
                {
                    omega_adc(i, j, k, 0) = omega_at_face(x_leftStates[2], x_rightStates[2]); // x-hi face
                    omega_adc(i, j, k, 1) = omega_at_face(y_leftStates[2], y_rightStates[2]); // y-hi face
                }
            }

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            
            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Conservative Variable Check", { 
                { "eta", eta(i, j, k) },
                { "rho_eta0", rho_eta0(i, j, k) },
                { "rho_eta1", rho_eta1(i, j, k) },
                { "rho", rho(i, j, k) },
                { "M[0]", M(i, j, k, 0) },
                { "M[1]", M(i, j, k, 1) },
                { "E", E(i, j, k) },
                { "gammaf", gammaf(i, j, k) },
                { "press", press(i, j, k) },
            }); // end check4nans


            try
            {
                flux_xlo = riemannsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = riemannsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = riemannsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = riemannsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN RIEMANN SOLVERS");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);

                Util::ParallelMessage(INFO, "x_states[0]=", x_states[0]);
                Util::ParallelMessage(INFO, "x_states[1]=", x_states[1]);
                Util::ParallelMessage(INFO, "x_states[2]=", x_states[2]);

                Util::ParallelMessage(INFO, "y_states[0]=", y_states[0]);
                Util::ParallelMessage(INFO, "y_states[1]=", y_states[1]);
                Util::ParallelMessage(INFO, "y_states[2]=", y_states[2]);

                Util::Abort(INFO);
            }

            // ------------------------------------------------------------
            // PP flux limiter: store the high-order (selected) and low-order
            // (HLL) hi-face fluxes in fixed (x,y) momentum frame so the limiter
            // passes can form the blend F^l + theta*(F^h - F^l). Lo-face fluxes
            // are written into the lo ghost at box boundaries (FillBoundary
            // overwrites interior box boundaries; physical boundaries keep the
            // BC-based flux). Components: 0=mass,1=x-mom,2=y-mom,3=energy,
            // 4=u_interface (Fhi only).
            // ------------------------------------------------------------
            if (pp_on)
            {
                Solver::Local::FluidRiemann::Flux hll_xlo = hll_solver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_ylo = hll_solver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_xhi = hll_solver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_yhi = hll_solver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);

                // x hi-face (i+1/2): normal=x, tangent=y
                Fhi_x(i, j, k, 0) = flux_xhi.mass;
                Fhi_x(i, j, k, 1) = flux_xhi.momentum_normal;
                Fhi_x(i, j, k, 2) = flux_xhi.momentum_tangent;
                Fhi_x(i, j, k, 3) = flux_xhi.energy;
                Fhi_x(i, j, k, 4) = flux_xhi.u_interface;
                Flo_x(i, j, k, 0) = hll_xhi.mass;
                Flo_x(i, j, k, 1) = hll_xhi.momentum_normal;
                Flo_x(i, j, k, 2) = hll_xhi.momentum_tangent;
                Flo_x(i, j, k, 3) = hll_xhi.energy;
                Flo_x(i, j, k, 4) = hll_xhi.u_interface;

                // y hi-face (j+1/2): normal=y, tangent=x
                Fhi_y(i, j, k, 0) = flux_yhi.mass;
                Fhi_y(i, j, k, 1) = flux_yhi.momentum_tangent;
                Fhi_y(i, j, k, 2) = flux_yhi.momentum_normal;
                Fhi_y(i, j, k, 3) = flux_yhi.energy;
                Fhi_y(i, j, k, 4) = flux_yhi.u_interface;
                Flo_y(i, j, k, 0) = hll_yhi.mass;
                Flo_y(i, j, k, 1) = hll_yhi.momentum_tangent;
                Flo_y(i, j, k, 2) = hll_yhi.momentum_normal;
                Flo_y(i, j, k, 3) = hll_yhi.energy;
                Flo_y(i, j, k, 4) = hll_yhi.u_interface;

                if (i == bx_lo.x) {
                    Fhi_x(i - 1, j, k, 0) = flux_xlo.mass;
                    Fhi_x(i - 1, j, k, 1) = flux_xlo.momentum_normal;
                    Fhi_x(i - 1, j, k, 2) = flux_xlo.momentum_tangent;
                    Fhi_x(i - 1, j, k, 3) = flux_xlo.energy;
                    Fhi_x(i - 1, j, k, 4) = flux_xlo.u_interface;
                    Flo_x(i - 1, j, k, 0) = hll_xlo.mass;
                    Flo_x(i - 1, j, k, 1) = hll_xlo.momentum_normal;
                    Flo_x(i - 1, j, k, 2) = hll_xlo.momentum_tangent;
                    Flo_x(i - 1, j, k, 3) = hll_xlo.energy;
                    Flo_x(i - 1, j, k, 4) = hll_xlo.u_interface;
                }
                if (j == bx_lo.y) {
                    Fhi_y(i, j - 1, k, 0) = flux_ylo.mass;
                    Fhi_y(i, j - 1, k, 1) = flux_ylo.momentum_tangent;
                    Fhi_y(i, j - 1, k, 2) = flux_ylo.momentum_normal;
                    Fhi_y(i, j - 1, k, 3) = flux_ylo.energy;
                    Fhi_y(i, j - 1, k, 4) = flux_ylo.u_interface;
                    Flo_y(i, j - 1, k, 0) = hll_ylo.mass;
                    Flo_y(i, j - 1, k, 1) = hll_ylo.momentum_tangent;
                    Flo_y(i, j - 1, k, 2) = hll_ylo.momentum_normal;
                    Flo_y(i, j - 1, k, 3) = hll_ylo.energy;
                    Flo_y(i, j - 1, k, 4) = hll_ylo.u_interface;
                }

                // Diagnostic: relative deviation of the used flux from pure HLL
                // at the two hi-faces, max over conserved components. ~0 at a
                // tagged face confirms it actually produced an HLL flux.
                auto rel_dev = [&](const Solver::Local::FluidRiemann::Flux& fh,
                                   const Solver::Local::FluidRiemann::Flux& fl_) -> Set::Scalar
                {
                    Set::Scalar num = std::max(std::max(std::abs(fh.mass            - fl_.mass),
                                                        std::abs(fh.momentum_normal - fl_.momentum_normal)),
                                               std::max(std::abs(fh.momentum_tangent- fl_.momentum_tangent),
                                                        std::abs(fh.energy          - fl_.energy)));
                    Set::Scalar den = std::max(std::max(std::abs(fl_.mass),            std::abs(fl_.momentum_normal)),
                                               std::max(std::abs(fl_.momentum_tangent),std::abs(fl_.energy)));
                    return num / (den + small);
                };
                flux_dev(i, j, k, 0) = rel_dev(flux_xhi, hll_xhi);
                flux_dev(i, j, k, 1) = rel_dev(flux_yhi, hll_yhi);
            }
            else
            {
                flux_dev(i, j, k, 0) = 0.0;
                flux_dev(i, j, k, 1) = 0.0;
            }

            // Upwind volume fractions (face-centered alpha, advected by HLLC contact wave speed u*)
            Set::Scalar eta_face_xlo = (flux_xlo.u_interface > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
            Set::Scalar eta_face_xhi = (flux_xhi.u_interface > 0.0) ? eta(i, j, k) : eta(i + 1, j, k);
            Set::Scalar eta_face_ylo = (flux_ylo.u_interface > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);
            Set::Scalar eta_face_yhi = (flux_yhi.u_interface > 0.0) ? eta(i, j, k) : eta(i, j + 1, k);

            // -----------------------------------------------------------
            // Store hi-face Riemann fluxes at cell centers for reflux.
            // Cell (i,j) stores the flux at face (i+1/2,j) for d=0
            // and at face (i,j+1/2) for d=1.  Converted to face-centered
            // MultiFabs in Advance() for CrseInit/FineAdd.
            // -----------------------------------------------------------
            if (have_cc_fluxes)
            {
                // x-direction hi-face: per-phase mass fluxes
                ff_mass_x(i, j, k, 0) = eta_face_xhi * flux_xhi.mass;           // rho_eta0
                ff_mass_x(i, j, k, 1) = (1.0 - eta_face_xhi) * flux_xhi.mass;  // rho_eta1
                ff_mom_x(i, j, k, 0)  = flux_xhi.momentum_normal;   // x-mom
                ff_mom_x(i, j, k, 1)  = flux_xhi.momentum_tangent;  // y-mom
                ff_ene_x(i, j, k)     = flux_xhi.energy;

                // y-direction hi-face: per-phase mass fluxes (swap mom to fixed x,y)
                ff_mass_y(i, j, k, 0) = eta_face_yhi * flux_yhi.mass;           // rho_eta0
                ff_mass_y(i, j, k, 1) = (1.0 - eta_face_yhi) * flux_yhi.mass;  // rho_eta1
                ff_mom_y(i, j, k, 0)  = flux_yhi.momentum_tangent;  // x-mom
                ff_mom_y(i, j, k, 1)  = flux_yhi.momentum_normal;   // y-mom
                ff_ene_y(i, j, k)     = flux_yhi.energy;

                // Write lo-face fluxes into ghost cells at box boundaries.
                // At coarse-fine interfaces FillBoundary has no neighbor to
                // copy from, so the ghost would stay zero without this.
                // At interior box boundaries FillBoundary overwrites later.
                if (i == bx_lo.x) {
                    ff_mass_x(i - 1, j, k, 0) = eta_face_xlo * flux_xlo.mass;
                    ff_mass_x(i - 1, j, k, 1) = (1.0 - eta_face_xlo) * flux_xlo.mass;
                    ff_mom_x(i - 1, j, k, 0)  = flux_xlo.momentum_normal;
                    ff_mom_x(i - 1, j, k, 1)  = flux_xlo.momentum_tangent;
                    ff_ene_x(i - 1, j, k)     = flux_xlo.energy;
                }
                if (j == bx_lo.y) {
                    ff_mass_y(i, j - 1, k, 0) = eta_face_ylo * flux_ylo.mass;
                    ff_mass_y(i, j - 1, k, 1) = (1.0 - eta_face_ylo) * flux_ylo.mass;
                    ff_mom_y(i, j - 1, k, 0)  = flux_ylo.momentum_tangent;
                    ff_mom_y(i, j - 1, k, 1)  = flux_ylo.momentum_normal;
                    ff_ene_y(i, j - 1, k)     = flux_ylo.energy;
                }
            }

            // ------------------------------------------------------------
            // Non-conservative volume-fraction advection (Saurel & Abgrall 1999, Eq. 41)
            //   D(eta)/Dt = deta/dt + u*grad(eta) = 0
            // Discretized as
            //   deta/dt = -div(u eta) + eta * div(u)
            // ------------------------------------------------------------
            {
                Set::Scalar div_uA_x = (flux_xhi.u_interface * eta_face_xhi
                                      - flux_xlo.u_interface * eta_face_xlo) / DX[0];
                Set::Scalar div_uA_y = (flux_yhi.u_interface * eta_face_yhi
                                      - flux_ylo.u_interface * eta_face_ylo) / DX[1];
                Set::Scalar div_u_x  = (flux_xhi.u_interface - flux_xlo.u_interface) / DX[0];
                Set::Scalar div_u_y  = (flux_yhi.u_interface - flux_ylo.u_interface) / DX[1];
                Set::Scalar eta_adv  = -(div_uA_x + div_uA_y)
                                       + eta(i, j, k) * (div_u_x + div_u_y);
                // When the PP limiter is active the conserved fluxes are blended
                // toward HLL per face; the volume fraction must advect with the
                // SAME blended interface velocity, or eta stays razor-sharp while
                // rho diffuses and the stiffened-liquid EOS amplifies the
                // mismatch into a spurious pressure wave. Defer the advective
                // term to Pass D (recomputed with the blended u); keep only the
                // non-advective sources here. With the limiter off, advect now.
                eta_rhs(i, j, k) = (pp_on ? 0.0 : eta_adv)
                                   + eta_dot_Vap
                                   + eta_dot_CH; // CHAN HILLARD (CH)
            }


            // UPDATE MIXED FLUID VARIABLES
            rho_flux(i, j, k) = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]);
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]);
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]);
            E_flux(i, j, k) = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]);

            // Density
            Set::Scalar F_rho_eta0_xlo = eta_face_xlo * flux_xlo.mass;
            Set::Scalar F_rho_eta0_xhi = eta_face_xhi * flux_xhi.mass;
            Set::Scalar F_rho_eta0_ylo = eta_face_ylo * flux_ylo.mass;
            Set::Scalar F_rho_eta0_yhi = eta_face_yhi * flux_yhi.mass;

            Set::Scalar F_rho_eta1_xlo = (1.0 - eta_face_xlo) * flux_xlo.mass;
            Set::Scalar F_rho_eta1_xhi = (1.0 - eta_face_xhi) * flux_xhi.mass;
            Set::Scalar F_rho_eta1_ylo = (1.0 - eta_face_ylo) * flux_ylo.mass;
            Set::Scalar F_rho_eta1_yhi = (1.0 - eta_face_yhi) * flux_yhi.mass;

            Set::Scalar rho_eta0_flux = (F_rho_eta0_xlo - F_rho_eta0_xhi) / DX[0]
                                        + (F_rho_eta0_ylo - F_rho_eta0_yhi) / DX[1];

            Set::Scalar rho_eta1_flux = (F_rho_eta1_xlo - F_rho_eta1_xhi) / DX[0]
                                        + (F_rho_eta1_ylo - F_rho_eta1_yhi) / DX[1];

            // Mass
            rho_eta0_rhs(i, j, k) = rho_eta0_flux + Source(i, j, k, 0) * (eta(i, j, k)) + m_dot_Vap;
            rho_eta1_rhs(i, j, k) = rho_eta1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k)) - m_dot_Vap;
            
            // Momentum
            M_rhs(i, j, k, 0) = M_flux(i, j, k, 0) + Source(i, j, k, 1); //(mu * (lap_ux * eta(i, j, k))) +
            M_rhs(i, j, k, 1) = M_flux(i, j, k, 1) + Source(i, j, k, 2); //(mu * (lap_uy * eta(i, j, k))) +

            // Energy
            E_rhs(i, j, k) = E_flux(i, j, k) + Source(i, j, k, 3);

           // ------------------------------------------------------------
           // Error Checking
           // ------------------------------------------------------------
           if ( (M_rhs(i, j, k, 0) != M_rhs(i, j, k, 0))
                or (M_rhs(i, j, k, 1) != M_rhs(i, j, k, 1))
                or (E_rhs(i, j, k) != E_rhs(i, j, k))
                or (rho_eta0_rhs(i, j, k) != rho_eta0_rhs(i, j, k))
                or (rho_eta1_rhs(i, j, k) != rho_eta1_rhs(i, j, k)))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HYDRO2");
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, ", j=", j);
                Util::ParallelMessage(INFO, "flux_xlo.mass=", flux_xlo.mass);
                Util::ParallelMessage(INFO, "flux_xhi.mass=", flux_xhi.mass);
                Util::ParallelMessage(INFO, "flux_ylo.mass=", flux_ylo.mass);
                Util::ParallelMessage(INFO, "flux_yhi.mass=", flux_yhi.mass);
                Util::ParallelMessage(INFO, "Source=", Source(i, j, k, 0), ", ", Source(i, j, k, 1), ", ", Source(i, j, k, 2), ", ", Source(i, j, k, 3));
                Util::ParallelMessage(INFO, "x_states[1] ", x_states[1]);           // Center cell in x-direction
                Util::ParallelMessage(INFO, "y_states[1] ", y_states[1]);           // Center cell in y-direction
                Util::ParallelMessage(INFO, "x_rightStates[2] ", x_rightStates[2]); // Right interface in x-direction
                Util::ParallelMessage(INFO, "y_rightStates[2] ", y_rightStates[2]); // Right interface in y-direction
                Util::ParallelMessage(INFO, "x_rightStates[1] ", x_rightStates[1]); // Left interface in x-direction
                Util::ParallelMessage(INFO, "y_rightStates[1] ", y_rightStates[1]); // Left interface in y-direction
                Util::ParallelMessage(INFO, "gamma_eff=", gammaf(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta0/dt=", rho_eta0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", rho_eta1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                Util::Abort(INFO);
            }

            if ((time <= 1e-7 && i == 0 && j == 0) && false)
            { // First timestep~ish~, first cell
                Util::ParallelMessage(INFO, "=== FIRST CELL DIAGNOSTICS ===");
                Util::ParallelMessage(INFO, "eta = ", eta(i, j, k));
                Util::ParallelMessage(INFO, "rho = ", rho(i, j, k));
                Util::ParallelMessage(INFO, "M = ", M(i, j, k, 0), ", ", M(i, j, k, 1));
                Util::ParallelMessage(INFO, "E = ", E(i, j, k));
                Util::ParallelMessage(INFO, "press = ", press(i, j, k));

                // Check flux calculation
                Util::ParallelMessage(INFO, "flux_xlo.mass = ", flux_xlo.mass);
                Util::ParallelMessage(INFO, "flux_xhi.mass = ", flux_xhi.mass);
                Util::ParallelMessage(INFO, "flux_xlo.momentum_normal = ", flux_xlo.momentum_normal);
                Util::ParallelMessage(INFO, "flux_xhi.momentum_normal = ", flux_xhi.momentum_normal);
                Util::ParallelMessage(INFO, "flux_xlo.energy = ", flux_xlo.energy);
                Util::ParallelMessage(INFO, "flux_xhi.energy = ", flux_xhi.energy);

                Util::ParallelMessage(INFO, "drhoeta0/dt=", rho_eta0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", rho_eta1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                
            }

            // Calculate vorticity for visualization
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));
        });
    }

    // ====================================================================
    // Positivity-preserving flux limiter (Hu-Adams-Shu 2013).
    // Pass A above stored the selected (Fhi) and HLL (Flo) hi-face fluxes.
    // Here we (B) build the source-inclusive low-order baseline, (C) compute
    // the per-face blend factor theta, and (D) recompute the conservative RHS
    // (and reflux fluxes) from the blended flux. Pass D also re-adds the
    // non-conservative eta advection using the SAME blended interface velocity
    // (Pass A left only the eta sources in eta_rhs when the limiter is on).
    // ====================================================================
    const bool pp_on_lev = (pp_flux_limiter != 0 && lev < (int)pp_scratch.size()
                            && pp_scratch[lev].Fhi[0]);
    if (pp_on_lev)
    {
        const Set::Scalar eps_rho_l = eps_rho;
        const Set::Scalar eps_p_l   = eps_p;
        const Set::Scalar p_cav_l   = p_cav;
        const Set::Scalar pref_l    = pref;
        const Set::Scalar small_l   = small;
        const Set::Scalar pp_factor = 2.0 * AMREX_SPACEDIM;
        // Source limiter on -> guard the per-phase partial densities in theta and
        // scale the source for positivity in Pass D. Its positivity baseline must
        // be source-free (the source is governed by s, not by theta), so override
        // the legacy source-into-baseline folding.
        const bool src_limit_on = (pp_source_limiter != 0);

        // Make each cell's lo-face flux available from its neighbor's hi-face.
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            pp_scratch[lev].Fhi[d]->FillBoundary(geom[lev].periodicity());
            pp_scratch[lev].Flo[d]->FillBoundary(geom[lev].periodicity());
        }

        // ---- Pass B: B = U_low + dt*flux_div (+ dt*Source if pp_source_limit) ----
        // With pp_source_limit set, the source is folded into the positivity
        // baseline so theta also guards against the source; cleared, theta
        // limits the flux only and the source is added unlimited in Pass D.
        const Set::Scalar src_fac = src_limit_on ? 0.0 : ((pp_source_limit != 0) ? 1.0 : 0.0);
        // Seed the baseline over valid cells AND one ghost layer from the
        // already ghost-filled conserved state. FillGhost4BC (run at the top of
        // RHS) FillPatches rho_eta0/rho_eta1/M/E across coarse-fine boundaries
        // and applies the physical BCs, and recomputes the primitives over the
        // grown box -- so density_mf/momentum_mf/energy_per_vol_mf hold valid
        // ghosts here. The valid cells are overwritten just below with the
        // source-inclusive low-order update, and FillBoundary then overwrites
        // same-level ghosts with the neighbor's updated baseline. This leaves
        // coarse-fine and physical-domain ghosts holding a real, positive
        // neighbor state instead of zero. Without it, PPThetaCell reads an empty
        // (rho=0) baseline across every patch boundary and returns theta=0,
        // dumping spurious full-HLL dissipation on grid lines (the pp_theta=0
        // streaks seen on patch edges).
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& gbx = mfi.growntilebox(1);
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = rho_eta0_mf[lev]->array(mfi);
            auto re1 = rho_eta1_mf[lev]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Bb(i, j, k, 0) = rho(i, j, k);
                Bb(i, j, k, 1) = Mom(i, j, k, 0);
                Bb(i, j, k, 2) = Mom(i, j, k, 1);
                Bb(i, j, k, 3) = Ene(i, j, k);
                Bb(i, j, k, 4) = re0(i, j, k);
                Bb(i, j, k, 5) = re1(i, j, k);
            });
        }
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = rho_eta0_mf[lev]->array(mfi);
            auto re1 = rho_eta1_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto Src = Source_mf[lev]->array(mfi);
            auto Vap = Vap_dot_mf[lev]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Fhx = pp_scratch[lev].Fhi[0]->array(mfi);
            auto Fhy = pp_scratch[lev].Fhi[1]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar fdiv_rho = (Flx(i - 1, j, k, 0) - Flx(i, j, k, 0)) / DX[0] + (Fly(i, j - 1, k, 0) - Fly(i, j, k, 0)) / DX[1];
                Set::Scalar fdiv_mx  = (Flx(i - 1, j, k, 1) - Flx(i, j, k, 1)) / DX[0] + (Fly(i, j - 1, k, 1) - Fly(i, j, k, 1)) / DX[1];
                Set::Scalar fdiv_my  = (Flx(i - 1, j, k, 2) - Flx(i, j, k, 2)) / DX[0] + (Fly(i, j - 1, k, 2) - Fly(i, j, k, 2)) / DX[1];
                Set::Scalar fdiv_E   = (Flx(i - 1, j, k, 3) - Flx(i, j, k, 3)) / DX[0] + (Fly(i, j - 1, k, 3) - Fly(i, j, k, 3)) / DX[1];
                Bb(i, j, k, 0) = rho(i, j, k) + dt * fdiv_rho + src_fac * dt * Src(i, j, k, 0);
                Bb(i, j, k, 1) = Mom(i, j, k, 0) + dt * fdiv_mx + src_fac * dt * Src(i, j, k, 1);
                Bb(i, j, k, 2) = Mom(i, j, k, 1) + dt * fdiv_my + src_fac * dt * Src(i, j, k, 2);
                Bb(i, j, k, 3) = Ene(i, j, k) + dt * fdiv_E + src_fac * dt * Src(i, j, k, 3);

                // Per-phase low-order baselines. The partial mass flux is the
                // upwind face fraction (by the high-order u_interface, matching
                // Pass D's re*_div split) times the low-order mixture mass flux,
                // so Bb[4]+Bb[5] == Bb[0] at theta=0. The source is folded in only
                // when the source limiter is off (src_fac); otherwise s guards it.
                Set::Scalar ef_xlo = (Fhx(i - 1, j, k, 4) > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
                Set::Scalar ef_xhi = (Fhx(i, j, k, 4)     > 0.0) ? eta(i, j, k)     : eta(i + 1, j, k);
                Set::Scalar ef_ylo = (Fhy(i, j - 1, k, 4) > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);
                Set::Scalar ef_yhi = (Fhy(i, j, k, 4)     > 0.0) ? eta(i, j, k)     : eta(i, j + 1, k);
                Set::Scalar fdiv_re0 = (ef_xlo * Flx(i - 1, j, k, 0) - ef_xhi * Flx(i, j, k, 0)) / DX[0]
                                     + (ef_ylo * Fly(i, j - 1, k, 0) - ef_yhi * Fly(i, j, k, 0)) / DX[1];
                Set::Scalar fdiv_re1 = ((1.0 - ef_xlo) * Flx(i - 1, j, k, 0) - (1.0 - ef_xhi) * Flx(i, j, k, 0)) / DX[0]
                                     + ((1.0 - ef_ylo) * Fly(i, j - 1, k, 0) - (1.0 - ef_yhi) * Fly(i, j, k, 0)) / DX[1];
                Set::Scalar mdv = Vap(i, j, k, 1);
                Bb(i, j, k, 4) = re0(i, j, k) + dt * fdiv_re0 + src_fac * dt * (Src(i, j, k, 0) * eta(i, j, k)         + mdv);
                Bb(i, j, k, 5) = re1(i, j, k) + dt * fdiv_re1 + src_fac * dt * (Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv);
            });
        }
        pp_scratch[lev].Bbase->FillBoundary(geom[lev].periodicity());

        // ---- Pass C: per-face blend factor theta ----
        // Default to 1 (no blend) so any face not explicitly set -- e.g. hi-side
        // ghosts Pass D never reads -- stays high-order rather than full HLL.
        pp_scratch[lev].theta->setVal(1.0);
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            // Grow one cell on the low side so each box also fills the theta of
            // its lo-edge faces (held in the lo ghost as that ghost cell's hi
            // face). Pass D reads a cell's lo-face theta from there; across a
            // coarse-fine boundary FillBoundary has no same-level neighbor to
            // supply it, so we compute it from the now-valid baseline and the
            // lo-face fluxes Pass A wrote into the ghost. Growing only the low
            // side keeps every (i+1,j+1) neighbor read inside the 1-cell halo.
            amrex::Box bx = mfi.validbox();
            bx.growLo(0, 1);
            bx.growLo(1, 1);
            auto Fhx = pp_scratch[lev].Fhi[0]->array(mfi);
            auto Fhy = pp_scratch[lev].Fhi[1]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            auto gam = gamma_mf[lev]->array(mfi);
            auto p0  = p0_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto Th  = pp_scratch[lev].theta->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // x hi-face: L = (i,j), R = (i+1,j), lambda = dt/dx. ef = upwind
                // face fraction (by high-order u_interface) shared by both cells.
                {
                    Set::Scalar dF[4] = { Fhx(i, j, k, 0) - Flx(i, j, k, 0), Fhx(i, j, k, 1) - Flx(i, j, k, 1),
                                          Fhx(i, j, k, 2) - Flx(i, j, k, 2), Fhx(i, j, k, 3) - Flx(i, j, k, 3) };
                    Set::Scalar BL[4] = { Bb(i, j, k, 0), Bb(i, j, k, 1), Bb(i, j, k, 2), Bb(i, j, k, 3) };
                    Set::Scalar BR[4] = { Bb(i + 1, j, k, 0), Bb(i + 1, j, k, 1), Bb(i + 1, j, k, 2), Bb(i + 1, j, k, 3) };
                    Set::Scalar lam = dt / DX[0];
                    Set::Scalar ef  = (Fhx(i, j, k, 4) > 0.0) ? eta(i, j, k) : eta(i + 1, j, k);
                    Set::Scalar pfL = Solver::EOS::EOS::PressureFloor(eta(i, j, k),     p0(i, j, k),     eps_p_l, p_cav_l);
                    Set::Scalar pfR = Solver::EOS::EOS::PressureFloor(eta(i + 1, j, k), p0(i + 1, j, k), eps_p_l, p_cav_l);
                    Set::Scalar tL = PPThetaCell(BL, -pp_factor * lam, dF, gam(i, j, k),     p0(i, j, k),     eps_rho_l, pfL, pref_l,
                                                 src_limit_on, Bb(i, j, k, 4),     Bb(i, j, k, 5),     ef, small_l);
                    Set::Scalar tR = PPThetaCell(BR, +pp_factor * lam, dF, gam(i + 1, j, k), p0(i + 1, j, k), eps_rho_l, pfR, pref_l,
                                                 src_limit_on, Bb(i + 1, j, k, 4), Bb(i + 1, j, k, 5), ef, small_l);
                    Set::Scalar th = std::min(tL, tR);
                    Th(i, j, k, 0) = (th < 0.0) ? 0.0 : ((th > 1.0) ? 1.0 : th);
                }
                // y hi-face: L = (i,j), R = (i,j+1), lambda = dt/dy
                {
                    Set::Scalar dF[4] = { Fhy(i, j, k, 0) - Fly(i, j, k, 0), Fhy(i, j, k, 1) - Fly(i, j, k, 1),
                                          Fhy(i, j, k, 2) - Fly(i, j, k, 2), Fhy(i, j, k, 3) - Fly(i, j, k, 3) };
                    Set::Scalar BL[4] = { Bb(i, j, k, 0), Bb(i, j, k, 1), Bb(i, j, k, 2), Bb(i, j, k, 3) };
                    Set::Scalar BR[4] = { Bb(i, j + 1, k, 0), Bb(i, j + 1, k, 1), Bb(i, j + 1, k, 2), Bb(i, j + 1, k, 3) };
                    Set::Scalar lam = dt / DX[1];
                    Set::Scalar ef  = (Fhy(i, j, k, 4) > 0.0) ? eta(i, j, k) : eta(i, j + 1, k);
                    Set::Scalar pfL = Solver::EOS::EOS::PressureFloor(eta(i, j, k),     p0(i, j, k),     eps_p_l, p_cav_l);
                    Set::Scalar pfR = Solver::EOS::EOS::PressureFloor(eta(i, j + 1, k), p0(i, j + 1, k), eps_p_l, p_cav_l);
                    Set::Scalar tL = PPThetaCell(BL, -pp_factor * lam, dF, gam(i, j, k),     p0(i, j, k),     eps_rho_l, pfL, pref_l,
                                                 src_limit_on, Bb(i, j, k, 4),     Bb(i, j, k, 5),     ef, small_l);
                    Set::Scalar tR = PPThetaCell(BR, +pp_factor * lam, dF, gam(i, j + 1, k), p0(i, j + 1, k), eps_rho_l, pfR, pref_l,
                                                 src_limit_on, Bb(i, j + 1, k, 4), Bb(i, j + 1, k, 5), ef, small_l);
                    Set::Scalar th = std::min(tL, tR);
                    Th(i, j, k, 1) = (th < 0.0) ? 0.0 : ((th > 1.0) ? 1.0 : th);
                }
            });
        }
        pp_scratch[lev].theta->FillBoundary(geom[lev].periodicity());

        // ---- Pass D: blended flux divergence -> conservative RHS + reflux ----
        const bool have_cc = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            const auto bxlo = amrex::lbound(bx);
            auto eta = eta_mf[lev]->array(mfi);
            auto Src = Source_mf[lev]->array(mfi);
            auto Vap = Vap_dot_mf[lev]->array(mfi);
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = rho_eta0_mf[lev]->array(mfi);
            auto re1 = rho_eta1_mf[lev]->array(mfi);
            auto gam = gamma_mf[lev]->array(mfi);
            auto p0  = p0_mf[lev]->array(mfi);
            auto Fhx = pp_scratch[lev].Fhi[0]->array(mfi);
            auto Fhy = pp_scratch[lev].Fhi[1]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Th  = pp_scratch[lev].theta->array(mfi);
            auto re0rhs = rho_eta0_rhs_mf.array(mfi);
            auto re1rhs = rho_eta1_rhs_mf.array(mfi);
            auto Mrhs   = M_rhs_mf.array(mfi);
            auto Erhs   = E_rhs_mf.array(mfi);
            auto etarhs = eta_rhs_mf.array(mfi);
            auto rho_flux = rho_flux_mf[lev]->array(mfi);
            auto M_flux   = M_flux_mf[lev]->array(mfi);
            auto E_flux   = E_flux_mf[lev]->array(mfi);
            auto pp_th_diag = pp_theta_mf[lev]->array(mfi);
            auto pp_dm_diag = pp_dmass_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y, ff_mom_x, ff_mom_y, ff_ene_x, ff_ene_y;
            if (have_cc)
            {
                ff_mass_x = cc_fluxes[lev].mass[0]->array(mfi);
                ff_mass_y = cc_fluxes[lev].mass[1]->array(mfi);
                ff_mom_x  = cc_fluxes[lev].mom[0]->array(mfi);
                ff_mom_y  = cc_fluxes[lev].mom[1]->array(mfi);
                ff_ene_x  = cc_fluxes[lev].energy[0]->array(mfi);
                ff_ene_y  = cc_fluxes[lev].energy[1]->array(mfi);
            }
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar th_xhi = Th(i, j, k, 0);
                Set::Scalar th_xlo = Th(i - 1, j, k, 0);
                Set::Scalar th_yhi = Th(i, j, k, 1);
                Set::Scalar th_ylo = Th(i, j - 1, k, 1);

                // Diagnostics: blend factor and the antidiffusive mass flux the
                // limiter removed at this cell's hi-faces (the desync the
                // unblended eta advection never sees).
                pp_th_diag(i, j, k, 0) = th_xhi;
                pp_th_diag(i, j, k, 1) = th_yhi;
                pp_dm_diag(i, j, k) = std::max((1.0 - th_xhi) * std::abs(Fhx(i, j, k, 0) - Flx(i, j, k, 0)),
                                               (1.0 - th_yhi) * std::abs(Fhy(i, j, k, 0) - Fly(i, j, k, 0)));

                // Blended face fluxes F^l + theta*(F^h - F^l), comps 0=mass,1=Mx,2=My,3=E.
                Set::Scalar Fxhi[4], Fxlo[4], Fyhi[4], Fylo[4];
                for (int c = 0; c < 4; c++)
                {
                    Fxhi[c] = Flx(i, j, k, c)     + th_xhi * (Fhx(i, j, k, c)     - Flx(i, j, k, c));
                    Fxlo[c] = Flx(i - 1, j, k, c) + th_xlo * (Fhx(i - 1, j, k, c) - Flx(i - 1, j, k, c));
                    Fyhi[c] = Fly(i, j, k, c)     + th_yhi * (Fhy(i, j, k, c)     - Fly(i, j, k, c));
                    Fylo[c] = Fly(i, j - 1, k, c) + th_ylo * (Fhy(i, j - 1, k, c) - Fly(i, j - 1, k, c));
                }

                // Upwinded face volume fractions from the high-order u_interface.
                Set::Scalar uxhi = Fhx(i, j, k, 4),     uxlo = Fhx(i - 1, j, k, 4);
                Set::Scalar uyhi = Fhy(i, j, k, 4),     uylo = Fhy(i, j - 1, k, 4);
                Set::Scalar ef_xhi = (uxhi > 0.0) ? eta(i, j, k) : eta(i + 1, j, k);
                Set::Scalar ef_xlo = (uxlo > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
                Set::Scalar ef_yhi = (uyhi > 0.0) ? eta(i, j, k) : eta(i, j + 1, k);
                Set::Scalar ef_ylo = (uylo > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);

                Set::Scalar rho_div = (Fxlo[0] - Fxhi[0]) / DX[0] + (Fylo[0] - Fyhi[0]) / DX[1];
                Set::Scalar mx_div  = (Fxlo[1] - Fxhi[1]) / DX[0] + (Fylo[1] - Fyhi[1]) / DX[1];
                Set::Scalar my_div  = (Fxlo[2] - Fxhi[2]) / DX[0] + (Fylo[2] - Fyhi[2]) / DX[1];
                Set::Scalar E_div   = (Fxlo[3] - Fxhi[3]) / DX[0] + (Fylo[3] - Fyhi[3]) / DX[1];

                Set::Scalar re0_div = (ef_xlo * Fxlo[0] - ef_xhi * Fxhi[0]) / DX[0]
                                    + (ef_ylo * Fylo[0] - ef_yhi * Fyhi[0]) / DX[1];
                Set::Scalar re1_div = ((1.0 - ef_xlo) * Fxlo[0] - (1.0 - ef_xhi) * Fxhi[0]) / DX[0]
                                    + ((1.0 - ef_ylo) * Fylo[0] - (1.0 - ef_yhi) * Fyhi[0]) / DX[1];

                // Volume-fraction advection with the SAME per-face blend factor as
                // the conserved fluxes: u_blend = u_HLL + theta*(u_HLLC - u_HLL).
                // Pass A deferred this term (leaving only sources in eta_rhs) when
                // the limiter is on, so eta now tracks the blended mass transport
                // instead of advecting at the unblended HLLC contact speed -- the
                // desync that floored the pressure in the stiff liquid. theta=1
                // faces recover full HLLC advection identically to the limiter-off
                // path.
                Set::Scalar ub_xhi = Flx(i, j, k, 4)     + th_xhi * (Fhx(i, j, k, 4)     - Flx(i, j, k, 4));
                Set::Scalar ub_xlo = Flx(i - 1, j, k, 4) + th_xlo * (Fhx(i - 1, j, k, 4) - Flx(i - 1, j, k, 4));
                Set::Scalar ub_yhi = Fly(i, j, k, 4)     + th_yhi * (Fhy(i, j, k, 4)     - Fly(i, j, k, 4));
                Set::Scalar ub_ylo = Fly(i, j - 1, k, 4) + th_ylo * (Fhy(i, j - 1, k, 4) - Fly(i, j - 1, k, 4));

                Set::Scalar ebf_xhi = (ub_xhi > 0.0) ? eta(i, j, k)     : eta(i + 1, j, k);
                Set::Scalar ebf_xlo = (ub_xlo > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
                Set::Scalar ebf_yhi = (ub_yhi > 0.0) ? eta(i, j, k)     : eta(i, j + 1, k);
                Set::Scalar ebf_ylo = (ub_ylo > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);

                Set::Scalar div_uA_b = (ub_xhi * ebf_xhi - ub_xlo * ebf_xlo) / DX[0]
                                     + (ub_yhi * ebf_yhi - ub_ylo * ebf_ylo) / DX[1];
                Set::Scalar div_u_b  = (ub_xhi - ub_xlo) / DX[0]
                                     + (ub_yhi - ub_ylo) / DX[1];
                etarhs(i, j, k) += -div_uA_b + eta(i, j, k) * div_u_b;

                Set::Scalar mdv = Vap(i, j, k, 1); // m_dot_Vap stored in Pass A

                // Positivity-preserving SOURCE limiter. The flux-blended update is
                // already admissible (the per-phase theta guaranteed it); find the
                // largest single scale s in [0,1] so that adding dt*s*Source keeps
                // the cell admissible (mixture rho,p AND both partial densities).
                // One scalar s for every component preserves the action-reaction
                // split (mdv cancels in the mixture; total mass source = s*Src0).
                Set::Scalar s_src = 1.0;
                if (src_limit_on)
                {
                    Set::Scalar Bf[4] = { rho(i, j, k) + dt * rho_div, Mom(i, j, k, 0) + dt * mx_div,
                                          Mom(i, j, k, 1) + dt * my_div, Ene(i, j, k) + dt * E_div };
                    Set::Scalar Sv[4] = { Src(i, j, k, 0), Src(i, j, k, 1), Src(i, j, k, 2), Src(i, j, k, 3) };
                    // Mixture (rho, p): reuse PPThetaCell with scale dt, source as dF.
                    Set::Scalar pf = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0(i, j, k), eps_p_l, p_cav_l);
                    Set::Scalar s = PPThetaCell(Bf, dt, Sv, gam(i, j, k), p0(i, j, k), eps_rho_l, pf, pref_l);
                    // Per-phase partial densities (linear in s).
                    Set::Scalar re0f = re0(i, j, k) + dt * re0_div;
                    Set::Scalar re1f = re1(i, j, k) + dt * re1_div;
                    Set::Scalar src_re0 = Src(i, j, k, 0) * eta(i, j, k)         + mdv;
                    Set::Scalar src_re1 = Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv;
                    if (src_re0 < 0.0) { Set::Scalar s0 = (re0f - small_l) / (-dt * src_re0); if (s0 < s) s = s0; }
                    if (src_re1 < 0.0) { Set::Scalar s1 = (re1f - small_l) / (-dt * src_re1); if (s1 < s) s = s1; }
                    s_src = (s < 0.0) ? 0.0 : ((s > 1.0) ? 1.0 : s);
                }

                re0rhs(i, j, k) = re0_div + s_src * (Src(i, j, k, 0) * eta(i, j, k)         + mdv);
                re1rhs(i, j, k) = re1_div + s_src * (Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv);
                Mrhs(i, j, k, 0) = mx_div + s_src * Src(i, j, k, 1);
                Mrhs(i, j, k, 1) = my_div + s_src * Src(i, j, k, 2);
                Erhs(i, j, k)    = E_div  + s_src * Src(i, j, k, 3);

                rho_flux(i, j, k) = rho_div;
                M_flux(i, j, k, 0) = mx_div;
                M_flux(i, j, k, 1) = my_div;
                E_flux(i, j, k) = E_div;

                if (have_cc)
                {
                    ff_mass_x(i, j, k, 0) = ef_xhi * Fxhi[0];
                    ff_mass_x(i, j, k, 1) = (1.0 - ef_xhi) * Fxhi[0];
                    ff_mom_x(i, j, k, 0) = Fxhi[1];
                    ff_mom_x(i, j, k, 1) = Fxhi[2];
                    ff_ene_x(i, j, k) = Fxhi[3];

                    ff_mass_y(i, j, k, 0) = ef_yhi * Fyhi[0];
                    ff_mass_y(i, j, k, 1) = (1.0 - ef_yhi) * Fyhi[0];
                    ff_mom_y(i, j, k, 0) = Fyhi[1];
                    ff_mom_y(i, j, k, 1) = Fyhi[2];
                    ff_ene_y(i, j, k) = Fyhi[3];

                    if (i == bxlo.x) {
                        ff_mass_x(i - 1, j, k, 0) = ef_xlo * Fxlo[0];
                        ff_mass_x(i - 1, j, k, 1) = (1.0 - ef_xlo) * Fxlo[0];
                        ff_mom_x(i - 1, j, k, 0) = Fxlo[1];
                        ff_mom_x(i - 1, j, k, 1) = Fxlo[2];
                        ff_ene_x(i - 1, j, k) = Fxlo[3];
                    }
                    if (j == bxlo.y) {
                        ff_mass_y(i, j - 1, k, 0) = ef_ylo * Fylo[0];
                        ff_mass_y(i, j - 1, k, 1) = (1.0 - ef_ylo) * Fylo[0];
                        ff_mom_y(i, j - 1, k, 0) = Fylo[1];
                        ff_mom_y(i, j - 1, k, 1) = Fylo[2];
                        ff_ene_y(i, j - 1, k) = Fylo[3];
                    }
                }
            });
        }
    } // end pp_on_lev
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    step_counter[lev]++;

    // Swapping pointers
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    std::swap(eta_old_mf[lev], eta_mf[lev]);
    std::swap(rho_eta0_old_mf[lev], rho_eta0_mf[lev]);
    std::swap(rho_eta1_old_mf[lev], rho_eta1_mf[lev]);

    // ------------------------------------------------------------
    // Time Integration
    // ------------------------------------------------------------

    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*rho_eta0_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_eta1_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*eta_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*rho_eta0_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_eta1_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*eta_old_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        // rhs_mf:      [0]=rho_eta0_rhs, [1]=rho_eta1_rhs, [2]=M_rhs, [3]=E_rhs, [4]=eta_rhs
        // solution_mf: [0]=rho_eta0,     [1]=rho_eta1,     [2]=M,     [3]=E,     [4]=eta
        RHS(lev, time, dt,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], rhs_mf[4],
            solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3], solution_mf[4]);
    });

    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        // Copy stage data to working arrays
        amrex::MultiFab::Copy(*rho_eta0_mf[lev],       stage_mf[0], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*rho_eta1_mf[lev],       stage_mf[1], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*momentum_mf[lev],       stage_mf[2], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], stage_mf[3], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*eta_mf[lev],            stage_mf[4], 0, 0, 1,              nghost);

        // Clamp eta in domain prior to ghost fill (state can drift slightly outside [0,1])
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            });
        }

        // Fill all ghost cells
        FillGhost4BC(lev, time);

        // Copy back
        amrex::MultiFab::Copy(stage_mf[0], *rho_eta0_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[1], *rho_eta1_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[2], *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(stage_mf[3], *energy_per_vol_mf[lev], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[4], *eta_mf[lev],            0, 0, 1,              nghost);

    });


    // Zero face-flux MultiFabs so reflux accumulation starts clean each step.
    if (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0])
    {
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            cc_fluxes[lev].mass[d]->setVal(0.0);
            cc_fluxes[lev].mom[d]->setVal(0.0);
            cc_fluxes[lev].energy[d]->setVal(0.0);
        }
    }

    timeintegrator.advance(solution_old, solution_new, time, dt);

    // ------------------------------------------------------------------
    // Feed FluxRegister for reflux at coarse-fine boundaries.
    // Convert cell-centered hi-face fluxes to face-centered MultiFabs
    // required by CrseInit/FineAdd, then accumulate.
    // ------------------------------------------------------------------
    bool need_crse = (lev < finest_level
                      && lev + 1 < (int)flux_reg.size() && flux_reg[lev + 1]
                      && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
    bool need_fine = (lev > 0
                      && lev < (int)flux_reg.size() && flux_reg[lev]
                      && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);

    if (need_crse || need_fine)
    {
        if (need_crse)
            flux_reg[lev + 1]->setVal(0.0);

        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            // FillBoundary so ghost cells have neighbor's hi-face flux
            // (needed for the lo-face of each box in the conversion below).
            cc_fluxes[lev].mass[d]->FillBoundary(geom[lev].periodicity());
            cc_fluxes[lev].mom[d]->FillBoundary(geom[lev].periodicity());
            cc_fluxes[lev].energy[d]->FillBoundary(geom[lev].periodicity());

            // Build temporary face-centered MultiFabs for FluxRegister.
            // Use cc_fluxes' own BA/dmap to stay consistent.
            amrex::BoxArray face_ba = cc_fluxes[lev].mass[d]->boxArray();
            face_ba.surroundingNodes(d);
            const amrex::DistributionMapping& dm_cc = cc_fluxes[lev].mass[d]->DistributionMap();
            amrex::MultiFab face_mass(face_ba, dm_cc, 2, 0);
            amrex::MultiFab face_mom(face_ba, dm_cc, AMREX_SPACEDIM, 0);
            amrex::MultiFab face_ene(face_ba, dm_cc, 1, 0);

            // Convert: face(f,j,k) = cc(f-1,j,k) for d=0, etc.
            // The hi-face flux of cell (f-1) IS the flux at face f.
            int nlocal = (int)cc_fluxes[lev].mass[d]->local_size();
            for (int li = 0; li < nlocal; li++)
            {
                auto cc_m = cc_fluxes[lev].mass[d]->atLocalIdx(li).array();
                auto cc_p = cc_fluxes[lev].mom[d]->atLocalIdx(li).array();
                auto cc_e = cc_fluxes[lev].energy[d]->atLocalIdx(li).array();

                auto f_m = face_mass.atLocalIdx(li).array();
                auto f_p = face_mom.atLocalIdx(li).array();
                auto f_e = face_ene.atLocalIdx(li).array();

                const amrex::Box& fbx = face_mass.atLocalIdx(li).box();

                int dd = d;
                amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int ii = (dd == 0) ? i - 1 : i;
                    int jj = (dd == 1) ? j - 1 : j;
                    f_m(i, j, k, 0) = cc_m(ii, jj, k, 0);
                    f_m(i, j, k, 1) = cc_m(ii, jj, k, 1);
                    for (int n = 0; n < AMREX_SPACEDIM; n++)
                        f_p(i, j, k, n) = cc_p(ii, jj, k, n);
                    f_e(i, j, k) = cc_e(ii, jj, k);
                });
            }

            // Face area MultiFab for correct area-weighted flux accumulation.
            // For Cartesian grids: x-face area = dy, y-face area = dx.
            const amrex::Real* dx_lev = geom[lev].CellSize();
            amrex::Real face_area = (d == 0) ? dx_lev[1] : dx_lev[0];
            amrex::MultiFab area_mf(face_ba, dm_cc, 1, 0);
            area_mf.setVal(face_area);

            if (need_crse)
            {
                flux_reg[lev + 1]->CrseInit(face_mass, area_mf, d, 0, 0, 2, -dt);
                flux_reg[lev + 1]->CrseInit(face_mom, area_mf, d, 0, 2, AMREX_SPACEDIM, -dt);
                flux_reg[lev + 1]->CrseInit(face_ene, area_mf, d, 0, 2 + AMREX_SPACEDIM, 1, -dt);
            }
            if (need_fine)
            {
                flux_reg[lev]->FineAdd(face_mass, area_mf, d, 0, 0, 2, dt);
                flux_reg[lev]->FineAdd(face_mom, area_mf, d, 0, 2, AMREX_SPACEDIM, dt);
                flux_reg[lev]->FineAdd(face_ene, area_mf, d, 0, 2 + AMREX_SPACEDIM, 1, dt);
            }
        }
    }

    // ENFORCE POSITIVITY after time advance. The deficit healed here is mass
    // created from nothing (non-conservative) -- accumulate it as a diagnostic
    // so the source/flux limiter's effect on conservation is measurable.
    Set::Scalar floor_mass_local = 0.0;
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Scalar small_local = small;

        amrex::ParallelFor(bx, [=, &floor_mass_local] AMREX_GPU_DEVICE(int i, int j, int k) {
            floor_mass_local += std::max(small_local - rho_eta0(i, j, k), 0.0)
                              + std::max(small_local - rho_eta1(i, j, k), 0.0);
            rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), small_local);
            rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), small_local);
        });
    }



    // ------------------------------------------------------------
    // Interface Sharpenging
    // ------------------------------------------------------------
    if (apply_sharpening && (step_counter[lev] > 10) && (step_counter[lev] % sharpening_frequency == 0))
    {
        InterfaceSharpening(lev, dt);
    }

    // ------------------------------------------------------------
    // Mixed Fields
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta_new(i, j, k) = std::max(0.0, std::min(1.0, eta_new(i, j, k)));

            check4nans(time, lev, i, j, k, "ERROR IN Advance(): Conservative Variable Check", {
                { "eta_new", eta_new(i, j, k) },
                { "rho_eta0", rho_eta0(i, j, k) },
                { "rho_eta1", rho_eta1(i, j, k) },
                { "rho", rho(i, j, k) }
            });
        });
    } // end rho, eta solver loop

    // ------------------------------------------------------------
    // Compute CFL for next time step on this level
    // ------------------------------------------------------------
    Set::Scalar c_max_local = 0.0;
    Set::Scalar vx_max_local = 0.0;
    Set::Scalar vy_max_local = 0.0;
    Set::Scalar F_max_local = 0.0;
    Set::Scalar rho_min_local = 1e10;
    Set::Scalar floor_energy_local = 0.0; // internal energy injected by the eps_p backstop

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
    
        Set::Patch<const Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
    
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_mas = energy_per_mas_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> a = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ma = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_vol = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_mas = KE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_vol = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_mas = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> pp_efloor_ = pp_efloor_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
    
        amrex::ParallelFor(bx, [=, &c_max_local, &vx_max_local, &vy_max_local, &F_max_local, &rho_min_local, &floor_energy_local] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);
        
            Set::Vector grad_eta = Numeric::Gradient(eta_new, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta_new, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta_new, i, j, k, 0, DX);

            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
        
            etadot(i,j,k) = (eta_new(i,j,k) - eta(i,j,k)) / dt;
        
            v(i,j,k,0) = M(i,j,k,0) / (rho(i,j,k));
            v(i,j,k,1) = M(i,j,k,1) / (rho(i,j,k));

            // Limiting Velocity
            Set::Scalar u_limit = 1e8;
            v(i, j, k, 0) = (v(i, j, k, 0) < 0.0) ? std::max(v(i, j, k, 0), -u_limit) : std::min(v(i, j, k, 0), u_limit);
            v(i, j, k, 1) = (v(i, j, k, 1) < 0.0) ? std::max(v(i, j, k, 1), -u_limit) : std::min(v(i, j, k, 1), u_limit);
        
            KE_vol(i,j,k) = 0.5 * rho(i,j,k) * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
            KE_mas(i,j,k) = 0.5 * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
        
            UE_vol(i,j,k) = E_vol(i,j,k) - KE_vol(i,j,k);

            // EOS-aware positivity backstop (final safety net behind the PP flux
            // limiter). Enforce a PHASE-DEPENDENT physical pressure floor: the gas
            // (eta -> 1) cannot tension, p >= eps_p; the stiffened liquid (eta -> 0)
            // may be pulled into tension down to a cavitation limit, p >= -p_cav
            // (PressureFloor blends the two). This replaces the old uniform p >= eps_p
            // (which clamped legitimate lee-side liquid tension to 1 Pa and injected
            // energy every step). UE_vol >= (p_floor + gamma_eff*p0_eff - pref)/(gamma_eff-1).
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
            {
                Set::Scalar p_floor  = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0_eff(i, j, k), eps_p, p_cav);
                Set::Scalar ue_floor = (p_floor + gammaf(i, j, k) * p0_eff(i, j, k) - pref) / (gammaf(i, j, k) - 1.0);
                // Diagnostic: energy injected by the floor (0 if not floored).
                pp_efloor_(i, j, k) = std::max(ue_floor - UE_vol(i, j, k), 0.0);
                floor_energy_local += pp_efloor_(i, j, k);
                UE_vol(i, j, k) = std::max(UE_vol(i, j, k), ue_floor);
                // Heal the CONSERVED energy too, not just the derived UE/pressure.
                // Without this E_vol keeps its sub-floor (negative-internal-energy)
                // value: only the displayed pressure is clamped while the bad
                // conserved energy persists and propagates. Writing it back makes
                // the backstop a true (non-conservative) energy floor so the
                // conserved state stays consistent with the floored pressure.
                E_vol(i, j, k) = UE_vol(i, j, k) + KE_vol(i, j, k);
            }
            E_mas(i,j,k) = E_vol(i,j,k) / (rho(i,j,k) + small);
            UE_mas(i,j,k) = E_mas(i,j,k) - KE_mas(i,j,k);

            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);
        
            Set::Scalar f_prime = 4.0 * eta_new(i,j,k) * (eta_new(i,j,k) - 0.5) * (eta_new(i,j,k) - 1.0);
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i,j,k) = mu_chem;
        
            Bm(i,j,k) = eta(i,j,k) / (1.0 - eta(i,j,k) + small);
        
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gammaf(i, j, k), p0_eff(i, j, k), small);

            Ma(i,j,k,0) = v(i,j,k,0) / (a(i,j,k) + small);
            Ma(i,j,k,1) = v(i,j,k,1) / (a(i,j,k) + small);
        
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
            grad_eta_(i,j,k,0) = grad_eta(0);
            grad_eta_(i,j,k,1) = grad_eta(1);
        
            if (grad_eta_mag < 1e-4) {
                n_hat_(i,j,k,0) = 0.0;
                n_hat_(i,j,k,1) = 0.0;
            } else {
                n_hat_(i,j,k,0) = n_hat(0);
                n_hat_(i,j,k,1) = n_hat(1);
            }
        
            check4nans(time, lev, i, j, k, "ERROR IN Advance(): Visualization", { 
                {"eta_new", eta_new(i,j,k)},
                {"rho_eta0", rho_eta0(i,j,k)},
                {"rho_eta1", rho_eta1(i,j,k)},
                {"gammaf", gammaf(i,j,k)},
                {"etadot", etadot(i,j,k)},
                {"M[0]", M(i,j,k,0)},
                {"M[1]", M(i,j,k,1)},
                {"v[0]", v(i,j,k,0)},
                {"v[1]", v(i,j,k,1)},
                {"KE_vol", KE_vol(i,j,k)},
                {"KE_mas", KE_mas(i,j,k)},
                {"UE_vol", UE_vol(i,j,k)},
                {"UE_mas", UE_mas(i,j,k)},
                {"E_vol", E_vol(i,j,k)},
                {"E_mas", E_mas(i,j,k)},
                {"p0_eff", p0_eff(i,j,k)},
                {"press", press(i,j,k)},
                {"mu_chem_", mu_chem_(i,j,k)},
                {"Bm", Bm(i,j,k)},
                {"a", a(i,j,k)},
                {"Ma[0]", Ma(i,j,k,0)},
                {"Ma[1]", Ma(i,j,k,1)},
                {"grad_eta_[0]", grad_eta_(i,j,k,0)},
                {"grad_eta_[1]", grad_eta_(i,j,k,1)},
                {"n_hat_[0]", n_hat_(i,j,k,0)},
                {"n_hat_[1]", n_hat_(i,j,k,1)}
            });
        
            // Track CFL quantities (NOW WORKS - captured by reference!)
            c_max_local = std::max(c_max_local, a(i,j,k));
            vx_max_local = std::max(vx_max_local, std::abs(v(i,j,k,0)));
            vy_max_local = std::max(vy_max_local, std::abs(v(i,j,k,1)));
        
            Set::Scalar F_mag = sqrt(Source(i,j,k,1) * Source(i,j,k,1) + 
                                    Source(i,j,k,2) * Source(i,j,k,2));
            F_max_local = std::max(F_max_local, F_mag);
            rho_min_local = std::min(rho_min_local, rho(i,j,k));
        });
    } // end Mixed Fields loop

    // Parallel Reduction
    amrex::ParallelDescriptor::ReduceRealMax(c_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(F_max_local);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min_local);

    c_max = c_max_local;
    vx_max = vx_max_local;
    vy_max = vy_max_local;
    F_max = F_max_local;
    rho_min = rho_min_local;

    // Conservation diagnostic: total mass and internal energy this level created
    // out of nothing by the post-update positivity floors this step (cell-volume
    // weighted). Both should trend to ~0 as the source/flux limiter does its job;
    // a persistently large mass figure is the droplet-mass leak.
    amrex::ParallelDescriptor::ReduceRealSum(floor_mass_local);
    amrex::ParallelDescriptor::ReduceRealSum(floor_energy_local);
    const Set::Scalar cell_vol = DX[0] * DX[1];
    if ((floor_mass_local > 0.0 || floor_energy_local > 0.0) && amrex::ParallelDescriptor::IOProcessor())
        Util::ParallelMessage(INFO, "[pp-floor] lev=", lev,
                              " mass_injected=", floor_mass_local * cell_vol,
                              " energy_injected=", floor_energy_local * cell_vol);

    // Computing dt for next time step on all levels
    Set::Scalar dx_min = std::min(DX[0], DX[1]);

    Set::Scalar wave_speed = c_max + sqrt(vx_max * vx_max + vy_max * vy_max);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    Set::Scalar mu_max = std::max(mu0, mu1);
    Set::Scalar dt_viscous = cfl_v * rho_min * dx_min * dx_min / (mu_max + small);

    Set::Scalar a_max = F_max / (rho_min + small);
    Set::Scalar dt_force = cfl_v * sqrt(dx_min / (a_max + small));

    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    Set::Scalar dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn });
    dt_max = dt_max * 0.9;

    // Debugging to report cfl constants used. Change bool to show
    if ((step_counter[lev] % 10 == 0) && false)
    {
        Util::Message(INFO, "=== CFL DIAGNOSTICS Level ", lev, " ===");
        Util::Message(INFO, "  c_max = ", c_max, " m/s");
        Util::Message(INFO, "  vx_max = ", vx_max, " m/s");
        Util::Message(INFO, "  vy_max = ", vy_max, " m/s");
        Util::Message(INFO, "  dt_max = ", dt_max, " s");
    }

    if (dynamictimestep.on)
    {
        this->DynamicTimestep_SyncTimeStep(lev, dt_max);
    }

} 


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// REGRIDDING //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Regrid(int lev, Set::Scalar regrid_time)
{
    BL_PROFILE("Integrator::Hydro2::Regrid");

    // Zero fill fields
    ZeroDerivedScratchFields(lev);

    // Ensure vectors are large enough for this level.
    if ((int)flux_reg.size() <= lev)
        flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev)
        cc_fluxes.resize(lev + 1);
    if ((int)pp_scratch.size() <= lev)
        pp_scratch.resize(lev + 1);

    // Use BA/dmap from a registered MultiFab (same reason as Initialize).
    const amrex::BoxArray& ba_reg = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm_reg = density_mf[lev]->DistributionMap();

    // Rebuild FluxRegister on regridded level.
    if (lev > 0)
    {
        int ncomp_reflux = 2 + AMREX_SPACEDIM + 1;
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba_reg, dm_reg, refRatio(lev - 1), lev, ncomp_reflux);
    }

    // Rebuild cell-centered flux storage (1 ghost for FillBoundary in Advance).
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 2, 1);
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, AMREX_SPACEDIM, 1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 1, 1);

        cc_fluxes[lev].mass[d]->setVal(0.0);
        cc_fluxes[lev].mom[d]->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
    }

    // Rebuild PP flux-limiter scratch (1 ghost, mirrors cc_fluxes).
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        pp_scratch[lev].Fhi[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 5, 1);
        pp_scratch[lev].Flo[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 5, 1);
        pp_scratch[lev].Fhi[d]->setVal(0.0);
        pp_scratch[lev].Flo[d]->setVal(0.0);
    }
    pp_scratch[lev].Bbase = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 6, 1); // 6 comp: rho, Mx, My, E, rho*eta0, rho*eta1 (must match Initialize)
    pp_scratch[lev].theta = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, AMREX_SPACEDIM, 1);
    pp_scratch[lev].Bbase->setVal(0.0);
    pp_scratch[lev].theta->setVal(0.0);

    // Apply BC
    FillGhost4BC(lev, regrid_time);

    Util::Message(INFO, "Regridding on level", lev);
}// end regrid

//void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray& a_tags, Set::Scalar, int)
{
    BL_PROFILE("Integrator::Flame::TagCellsForRefinement");

    const Set::Scalar* DX = geom[lev].CellSize();
    Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));
    amrex::Box domain = geom[lev].Domain();

    // Eta criterion for refinement
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& eta = (*eta_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            if (grad_eta.lpNorm<2>() * dr * 2 > eta_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Vorticity criterion for refinement
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, 0, DX, sten);
            if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }
    
    // Gradu criterion for refinement
    for (amrex::MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& v = (*velocity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);
            if (grad_u.lpNorm<2>() * dr * 2 > gradu_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Divergence (shock) criterion for refinement.
    // Shocks are strongly compressive (div(u) << 0); clean material interfaces
    // and contacts keep u continuous so div(u) stays small there. A Ducros
    // switch additionally suppresses tagging in shear/vorticity-dominated
    // regions, so the gas-liquid interface no longer triggers refinement the
    // way the old pressure-gradient sensor did.
    for (amrex::MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& v = (*velocity_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const& a = (*a_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);

            // div(u) = trace; compressive part only.
            Set::Scalar div_u = grad_u.trace();
            Set::Scalar comp = std::max(-div_u, 0.0);

            // |curl(u)|^2 from the off-diagonal velocity gradients.
#if AMREX_SPACEDIM == 2
            Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
            Set::Scalar curl_mag2 = wz * wz;
#else
            Set::Scalar wx = grad_u(2, 1) - grad_u(1, 2);
            Set::Scalar wy = grad_u(0, 2) - grad_u(2, 0);
            Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
            Set::Scalar curl_mag2 = wx * wx + wy * wy + wz * wz;
#endif

            // Ducros switch in [0,1]: ~1 in a clean shock, ~0 in shear.
            Set::Scalar ducros = comp / std::sqrt(div_u * div_u + curl_mag2 + small * small);

            // Dimensionless shock strength: compression across a cell relative
            // to the local sound speed, gated by the Ducros switch.
            Set::Scalar c = std::max(a(i, j, k), small);
            Set::Scalar sensor = ducros * comp * dr / c;

            if (sensor > divu_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Density criterion for refinement
    for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& rho = (*density_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Vector grad_rho = Numeric::Gradient(rho, i, j, k, 0, DX, sten);
            if (grad_rho.lpNorm<2>() * dr * 2 > rho_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

}



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// INTERFACE SHARPENING /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::InterfaceSharpening(int lev, Set::Scalar dt_physical)
{
    BL_PROFILE("Integrator::Hydro2::InterfaceSharpening");

    if (!apply_sharpening)
        return;

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    Util::Message(INFO, "=== INTERFACE SHARPENING START ===");
    Util::Message(INFO, "Level: ", lev);

    // Check if interface exists
    Set::Scalar eta_min = eta_mf[lev]->min(0);
    Set::Scalar eta_max = eta_mf[lev]->max(0);

    Util::Message(INFO, "eta range: [", eta_min, ", ", eta_max, "]");

    if (eta_max - eta_min < 0.1)
    {
        Util::Message(INFO, "No significant interface detected, skipping sharpening");
        return;
    }

    // Temporary MultiFabs for sharpening procedure
    amrex::MultiFab psi_mf(rho_eta0_mf[lev]->boxArray(), rho_eta0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab psi_reinit_mf(rho_eta0_mf[lev]->boxArray(), rho_eta0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab phi_sharp_mf(rho_eta0_mf[lev]->boxArray(), rho_eta0_mf[lev]->DistributionMap(), 1, 2);

    // ============================================================================
    // STEP 1: Transform phi to psi (Equation 6)
    // psi = epsilon * ln(phi / (1-phi))
    // Only operate on INTERIOR cells (exclude boundaries)
    // ============================================================================
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx_full = mfi.validbox();

        // Shrink box to exclude boundaries
        amrex::Box bx = bx_full;
        bx.grow(0); // Exclude 2 layers from all boundaries

        if (bx.isEmpty())
        {
            Util::Message(INFO, "Box too small for sharpening, skipping");
            continue;
        }

        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> psi = psi_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Compute mixture density
            Set::Scalar rho = rho_eta0(i, j, k) + rho_eta1(i, j, k);

            // Compute volume fraction
            Set::Scalar phi = rho_eta0(i, j, k) / (rho + small);

            // Clamp to avoid log(0)
            phi = std::max(1e-10, std::min(1.0 - 1e-10, phi));

            // Equation (6): psi = epsilon * ln(phi/(1-phi))
            psi(i, j, k) = epsilon * std::log(phi / (1.0 - phi));
        });
    }

    // Fill boundaries for psi using custom BC function
    FillBoundariesWithBC(lev, 0.0, energy_bc, { &psi_mf });

    // ============================================================================
    // STEP 2: Reinitialize psi to signed distance function (Equation 10)
    // d(psi)/d(tau) = S(psi) * (1 - |grad(psi)|)
    // ============================================================================
    amrex::MFIter::allowMultipleMFIters(true);
    ReinitializeSignedDistance(lev, psi_reinit_mf, psi_mf, reinit_max_iter);
    amrex::MFIter::allowMultipleMFIters(false);

    // Fill boundaries for reinitialized psi
    FillBoundariesWithBC(lev, 0.0, energy_bc, { &psi_reinit_mf });

    // ============================================================================
    // STEP 3: Transform psi back to phi_sharp (Inverse of Equation 6)
    // phi = 1 / (1 + exp(-psi/epsilon))
    // Only operate on INTERIOR cells
    // ============================================================================
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx_full = mfi.validbox();

        // Shrink box to exclude boundaries
        amrex::Box bx = bx_full;
        bx.grow(0);

        if (bx.isEmpty())
            continue;

        Set::Patch<const Set::Scalar> psi_reinit = psi_reinit_mf.array(mfi);
        Set::Patch<Set::Scalar> phi_sharp = phi_sharp_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Inverse of Eq. (6): phi = 1/(1 + exp(-psi/epsilon))
            phi_sharp(i, j, k) = 1.0 / (1.0 + std::exp(-psi_reinit(i, j, k) / epsilon));

            // Clamp to [0,1]
            phi_sharp(i, j, k) = std::max(0.0, std::min(1.0, phi_sharp(i, j, k)));
        });
    }

    // Fill boundaries for phi_sharp
    Util::ParallelMessage(INFO, "Filling Shrp Interface");

    FillBoundariesWithBC(lev, 0.0, energy_bc, { &phi_sharp_mf });

    // ============================================================================
    // STEP 4: Density Correction with Compression Operators (Equations 15-16)
    // This is the key step that prevents density anomalies!
    // ============================================================================

    // Create working copies for iterative correction
    amrex::MultiFab rho_eta0_work(rho_eta0_mf[lev]->boxArray(), rho_eta0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab rho_eta1_work(rho_eta1_mf[lev]->boxArray(), rho_eta1_mf[lev]->DistributionMap(), 1, 2);

    // Initialize with current values
    amrex::MultiFab::Copy(rho_eta0_work, *rho_eta0_mf[lev], 0, 0, 1, 2);
    amrex::MultiFab::Copy(rho_eta1_work, *rho_eta1_mf[lev], 0, 0, 1, 2);

    // Pseudo-timestep for density correction (Equation 20a)
    // From knowledge: dt <= 2*h^2 for well-resolved interface (epsilon >= h/2)
    Set::Scalar h = std::min(DX[0], DX[1]);
    Set::Scalar dt_compression = 0.5 * h * h;

    Set::Scalar omega_relax = 0.5;

    

    Util::Message(INFO, "Starting density correction iterations...");

    for (int density_iter = 0; density_iter < density_max_iter; density_iter++)
    {
        // Store old values for convergence check
        amrex::MultiFab rho_eta0_old(rho_eta0_work.boxArray(), rho_eta0_work.DistributionMap(), 1, 2);
        amrex::MultiFab rho_eta1_old(rho_eta1_work.boxArray(), rho_eta1_work.DistributionMap(), 1, 2);

        amrex::MultiFab::Copy(rho_eta0_old, rho_eta0_work, 0, 0, 1, 2);
        amrex::MultiFab::Copy(rho_eta1_old, rho_eta1_work, 0, 0, 1, 2);

        // Apply compression operators (INTERIOR CELLS ONLY)
        for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx_full = mfi.validbox();

            // Shrink box to exclude boundaries
            amrex::Box bx = bx_full;
            bx.grow(0);

            if (bx.isEmpty())
                continue;

            Set::Patch<const Set::Scalar> phi_sharp = phi_sharp_mf.array(mfi);
            Set::Patch<Set::Scalar> rho_eta0 = rho_eta0_work.array(mfi);
            Set::Patch<Set::Scalar> rho_eta1 = rho_eta1_work.array(mfi);
            Set::Patch<const Set::Scalar> rho_eta0_original = rho_eta0_mf[lev]->array(mfi);
            Set::Patch<const Set::Scalar> rho_eta1_original = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto sten = Numeric::GetStencil(i, j, k, domain);

                // ============================================================
                // Compute geometric quantities from sharpened interface
                // ============================================================

                // Gradient of phi_sharp
                Set::Vector grad_phi = Numeric::Gradient(phi_sharp, i, j, k, 0, DX);
                Set::Scalar grad_phi_mag = grad_phi.lpNorm<2>();

                // Narrow band check (papers use phi(1-phi) > 0.001)
                Set::Scalar phi_val = phi_sharp(i, j, k);
                Set::Scalar narrow_band_indicator = phi_val * (1.0 - phi_val);

                if (narrow_band_indicator < 0.001)
                {
                    // Outside narrow band - don't modify density
                    return;
                }

                if (grad_phi_mag < 1e-10)
                {
                    // No interface gradient - don't modify
                    return;
                }

                // Normal vector
                Set::Vector n_hat = grad_phi / grad_phi_mag;

                // Laplacian of phi_sharp
                Set::Scalar lap_phi = Numeric::Laplacian(phi_sharp, i, j, k, 0, DX);

                // Curvature
                Set::Scalar kappa = -lap_phi / (grad_phi_mag + small);

                // ============================================================
                // Maximum Principle Check (Equation 23)
                // From knowledge: prevents spurious oscillations
                // Condition: |kappa| + |(1-2*phi)|/epsilon <= sqrt(2)/h
                // ============================================================

                Set::Scalar max_principle_lhs = std::abs(kappa) + std::abs(1.0 - 2.0 * phi_val) / epsilon;
                Set::Scalar max_principle_rhs = std::sqrt(2.0) / h;

                if (max_principle_lhs > max_principle_rhs)
                {
                    // Maximum principle violated - DON'T modify density
                    // This preserves thermodynamically consistent state from flow solver
                    return;
                }

                // ============================================================
                // Heaviside Function (from knowledge)
                // Localizes density correction to interface region
                // H = tanh(phi*(1-phi) / 0.01)
                // ============================================================

                Set::Scalar H = std::tanh(narrow_band_indicator / 0.01);

                // ============================================================
                // Compression Operator for Fluid 0 (Liquid) - Equation (15)
                // R_l = H * n dot [grad(epsilon * n dot grad(rho_0*phi_0))
                //                  - (1-2*phi)*grad(rho_0*phi_0)]
                // ============================================================

                // Gradient of rho_eta0
                Set::Vector grad_rho_eta0 = Numeric::Gradient(rho_eta0, i, j, k, 0, DX);

                // n dot grad(rho_0*phi_0)
                Set::Scalar n_dot_grad_rho_eta0 = n_hat.dot(grad_rho_eta0);

                // Compute grad(epsilon * n dot grad(rho_0*phi_0))
                // Need n dot grad(rho_0*phi_0) at neighboring cells
                Set::Scalar n_dot_grad_rho_eta0_ip = 0.0, n_dot_grad_rho_eta0_im = 0.0;
                Set::Scalar n_dot_grad_rho_eta0_jp = 0.0, n_dot_grad_rho_eta0_jm = 0.0;

                // At i+1,j
                if (i + 1 <= domain.bigEnd(0))
                {
                    Set::Vector grad_phi_ip = Numeric::Gradient(phi_sharp, i + 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_ip = grad_phi_ip.lpNorm<2>();
                    if (grad_phi_mag_ip > 1e-10)
                    {
                        Set::Vector n_hat_ip = grad_phi_ip / grad_phi_mag_ip;
                        Set::Vector grad_rho_eta0_ip = Numeric::Gradient(rho_eta0, i + 1, j, k, 0, DX);
                        n_dot_grad_rho_eta0_ip = n_hat_ip.dot(grad_rho_eta0_ip);
                    }
                }

                // At i-1,j
                if (i - 1 >= domain.smallEnd(0))
                {
                    Set::Vector grad_phi_im = Numeric::Gradient(phi_sharp, i - 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_im = grad_phi_im.lpNorm<2>();
                    if (grad_phi_mag_im > 1e-10)
                    {
                        Set::Vector n_hat_im = grad_phi_im / grad_phi_mag_im;
                        Set::Vector grad_rho_eta0_im = Numeric::Gradient(rho_eta0, i - 1, j, k, 0, DX);
                        n_dot_grad_rho_eta0_im = n_hat_im.dot(grad_rho_eta0_im);
                    }
                }

                // At i,j+1
                if (j + 1 <= domain.bigEnd(1))
                {
                    Set::Vector grad_phi_jp = Numeric::Gradient(phi_sharp, i, j + 1, k, 0, DX);
                    Set::Scalar grad_phi_mag_jp = grad_phi_jp.lpNorm<2>();
                    if (grad_phi_mag_jp > 1e-10)
                    {
                        Set::Vector n_hat_jp = grad_phi_jp / grad_phi_mag_jp;
                        Set::Vector grad_rho_eta0_jp = Numeric::Gradient(rho_eta0, i, j + 1, k, 0, DX);
                        n_dot_grad_rho_eta0_jp = n_hat_jp.dot(grad_rho_eta0_jp);
                    }
                }

                // At i,j-1
                if (j - 1 >= domain.smallEnd(1))
                {
                    Set::Vector grad_phi_jm = Numeric::Gradient(phi_sharp, i, j - 1, k, 0, DX);
                    Set::Scalar grad_phi_mag_jm = grad_phi_jm.lpNorm<2>();
                    if (grad_phi_mag_jm > 1e-10)
                    {
                        Set::Vector n_hat_jm = grad_phi_jm / grad_phi_mag_jm;
                        Set::Vector grad_rho_eta0_jm = Numeric::Gradient(rho_eta0, i, j - 1, k, 0, DX);
                        n_dot_grad_rho_eta0_jm = n_hat_jm.dot(grad_rho_eta0_jm);
                    }
                }

                // grad(epsilon * n dot grad(rho_0*phi_0))
                Set::Scalar grad_term0_x = epsilon * (n_dot_grad_rho_eta0_ip - n_dot_grad_rho_eta0_im) / (2.0 * DX[0]);
                Set::Scalar grad_term0_y = epsilon * (n_dot_grad_rho_eta0_jp - n_dot_grad_rho_eta0_jm) / (2.0 * DX[1]);

                Set::Scalar term1_0 = n_hat(0) * grad_term0_x + n_hat(1) * grad_term0_y;

                // (1-2*phi) * grad(rho_0*phi_0)
                Set::Scalar term2_0 = (1.0 - 2.0 * phi_val) * n_dot_grad_rho_eta0;

                // Compression operator R_l (Equation 15)
                Set::Scalar R_l = H * (term1_0 - term2_0);

                // ============================================================
                // Compression Operator for Fluid 1 (Gas) - Equation (16)
                // R_g = H * n dot [grad(epsilon * n dot grad(rho_1*phi_1))
                //                  - (1-2*phi)*grad(rho_1*phi_1)]
                // ============================================================

                // Gradient of rho_eta1
                Set::Vector grad_rho_eta1 = Numeric::Gradient(rho_eta1, i, j, k, 0, DX);

                // n dot grad(rho_1*phi_1)
                Set::Scalar n_dot_grad_rho_eta1 = n_hat.dot(grad_rho_eta1);

                // Compute grad(epsilon * n dot grad(rho_1*phi_1)) at neighbors
                Set::Scalar n_dot_grad_rho_eta1_ip = 0.0, n_dot_grad_rho_eta1_im = 0.0;
                Set::Scalar n_dot_grad_rho_eta1_jp = 0.0, n_dot_grad_rho_eta1_jm = 0.0;

                // At i+1,j
                if (i + 1 <= domain.bigEnd(0))
                {
                    Set::Vector grad_phi_ip = Numeric::Gradient(phi_sharp, i + 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_ip = grad_phi_ip.lpNorm<2>();
                    if (grad_phi_mag_ip > 1e-10)
                    {
                        Set::Vector n_hat_ip = grad_phi_ip / grad_phi_mag_ip;
                        Set::Vector grad_rho_eta1_ip = Numeric::Gradient(rho_eta1, i + 1, j, k, 0, DX);
                        n_dot_grad_rho_eta1_ip = n_hat_ip.dot(grad_rho_eta1_ip);
                    }
                }

                // At i-1,j
                if (i - 1 >= domain.smallEnd(0))
                {
                    Set::Vector grad_phi_im = Numeric::Gradient(phi_sharp, i - 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_im = grad_phi_im.lpNorm<2>();
                    if (grad_phi_mag_im > 1e-10)
                    {
                        Set::Vector n_hat_im = grad_phi_im / grad_phi_mag_im;
                        Set::Vector grad_rho_eta1_im = Numeric::Gradient(rho_eta1, i - 1, j, k, 0, DX);
                        n_dot_grad_rho_eta1_im = n_hat_im.dot(grad_rho_eta1_im);
                    }
                }

                // At i,j+1
                if (j + 1 <= domain.bigEnd(1))
                {
                    Set::Vector grad_phi_jp = Numeric::Gradient(phi_sharp, i, j + 1, k, 0, DX);
                    Set::Scalar grad_phi_mag_jp = grad_phi_jp.lpNorm<2>();
                    if (grad_phi_mag_jp > 1e-10)
                    {
                        Set::Vector n_hat_jp = grad_phi_jp / grad_phi_mag_jp;
                        Set::Vector grad_rho_eta1_jp = Numeric::Gradient(rho_eta1, i, j + 1, k, 0, DX);
                        n_dot_grad_rho_eta1_jp = n_hat_jp.dot(grad_rho_eta1_jp);
                    }
                }

                // At i,j-1
                if (j - 1 >= domain.smallEnd(1))
                {
                    Set::Vector grad_phi_jm = Numeric::Gradient(phi_sharp, i, j - 1, k, 0, DX);
                    Set::Scalar grad_phi_mag_jm = grad_phi_jm.lpNorm<2>();
                    if (grad_phi_mag_jm > 1e-10)
                    {
                        Set::Vector n_hat_jm = grad_phi_jm / grad_phi_mag_jm;
                        Set::Vector grad_rho_eta1_jm = Numeric::Gradient(rho_eta1, i, j - 1, k, 0, DX);
                        n_dot_grad_rho_eta1_jm = n_hat_jm.dot(grad_rho_eta1_jm);
                    }
                }

                // grad(epsilon * n dot grad(rho_1*phi_1))
                Set::Scalar grad_term1_x = epsilon * (n_dot_grad_rho_eta1_ip - n_dot_grad_rho_eta1_im) / (2.0 * DX[0]);
                Set::Scalar grad_term1_y = epsilon * (n_dot_grad_rho_eta1_jp - n_dot_grad_rho_eta1_jm) / (2.0 * DX[1]);

                Set::Scalar term1_1 = n_hat(0) * grad_term1_x + n_hat(1) * grad_term1_y;

                // (1-2*phi) * grad(rho_1*phi_1)
                Set::Scalar term2_1 = (1.0 - 2.0 * phi_val) * n_dot_grad_rho_eta1;

                // Compression operator R_g (Equation 16)
                Set::Scalar R_g = H * (term1_1 - term2_1);

                // ============================================================
                // Update with relaxation (pseudo-time stepping)
                // ============================================================

                rho_eta0(i, j, k) = rho_eta0(i, j, k) - omega_relax * dt_compression * R_l;
                rho_eta1(i, j, k) = rho_eta1(i, j, k) - omega_relax * dt_compression * R_g;

                // Ensure positivity
                rho_eta0(i, j, k) = std::max(small, rho_eta0(i, j, k));
                rho_eta1(i, j, k) = std::max(small, rho_eta1(i, j, k));

                // ============================================================
                // Enforce exact mass conservation
                // Total mass must equal original total mass
                // ============================================================

                Set::Scalar rho_total_original = rho_eta0_original(i, j, k) + rho_eta1_original(i, j, k);
                Set::Scalar rho_total_new = rho_eta0(i, j, k) + rho_eta1(i, j, k);

                Set::Scalar mass_error = std::abs(rho_total_new - rho_total_original);

                if (mass_error > 1e-12)
                {
                    // Renormalize to ensure exact mass conservation
                    Set::Scalar scale = rho_total_original / (rho_total_new + small);
                    rho_eta0(i, j, k) *= scale;
                    rho_eta1(i, j, k) *= scale;
                }
            });
        }

        // Fill boundaries after each iteration using custom BC function
        Util::ParallelMessage(INFO, "Filling Shrp Interface: Density Correction");

        FillBoundariesWithBC(lev, 0.0, density_bc, { &rho_eta0_work, &rho_eta1_work });

        // ========================================================================
        // Check convergence of density correction
        // ========================================================================

        amrex::MultiFab residual0(rho_eta0_work.boxArray(), rho_eta0_work.DistributionMap(), 1, 0);
        amrex::MultiFab residual1(rho_eta1_work.boxArray(), rho_eta1_work.DistributionMap(), 1, 0);

        amrex::MultiFab::Copy(residual0, rho_eta0_work, 0, 0, 1, 0);
        amrex::MultiFab::Copy(residual1, rho_eta1_work, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(residual0, rho_eta0_old, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(residual1, rho_eta1_old, 0, 0, 1, 0);

        Set::Scalar max_residual = std::max(residual0.norm0(), residual1.norm0());

        if (max_residual < density_tol)
        {
            Util::Message(INFO, "  Density correction converged in ", density_iter + 1, " iterations");
            break;
        }

        if (density_iter == density_max_iter - 1)
        {
            Util::Message(INFO, "  Density correction reached max iterations (", density_max_iter, ")");
            Util::Message(INFO, "  Final residual: ", max_residual);
        }
    }

    // ============================================================================
    // STEP 5: Copy corrected densities back to main arrays
    // ============================================================================

    amrex::MultiFab::Copy(*rho_eta0_mf[lev], rho_eta0_work, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*rho_eta1_mf[lev], rho_eta1_work, 0, 0, 1, 0);

    // ============================================================================
    // STEP 6: Update eta from corrected densities
    // ============================================================================
    // WARNING: This recovery (eta = rho_eta0 / rho_total) is the legacy
    // mass-fraction form that the new state model deliberately abandons.
    // It is dead code while apply_sharpening=0. If sharpening is re-enabled,
    // the sharpening algorithm needs to be reworked to operate on the
    // independent volume-fraction state (eta_mf) directly rather than via
    // the conservative phase masses; otherwise it will silently overwrite
    // the volume fraction with the mass fraction every sharpening pass.
    // ============================================================================

    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar rho_total = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            eta(i, j, k) = rho_eta0(i, j, k) / (rho_total + small);

            // Simple clamping (NO cutoff transformation)
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ============================================================================
    // FILL BOUNDARIES WITH CUSTOM BC (FINAL - ONLY ONCE)
    // ============================================================================
    Util::ParallelMessage(INFO, "Filling Shrp Interface: Density Correction, COMPELTE");

    // Eta: use eta_bc (not energy_bc or density_bc, which have wrong dirichlet values)
    FillBoundariesWithBC(lev, 0.0, eta_bc, { eta_mf[lev].get() });
    // Density: fill total density, then partition by eta (see FillGhost4BC fix)
    FillBoundariesWithBC(lev, 0.0, density_bc, { density_mf[lev].get() });
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);
        auto rho  = density_mf[lev]->array(mfi);
        auto eta  = eta_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho0(i, j, k) = rho(i, j, k) * eta(i, j, k);
            rho1(i, j, k) = rho(i, j, k) * (1.0 - eta(i, j, k));
        });
    }

    Util::Message(INFO, "=== INTERFACE SHARPENING COMPLETE ===");
}




///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// REINITIALIZE SIGNED DISTANCE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::ReinitializeSignedDistance(int lev,
                                        amrex::MultiFab &psi_mf,
                                        const amrex::MultiFab &psi_init_mf,
                                        int max_iter)
{
    BL_PROFILE("Integrator::Hydro2::ReinitializeSignedDistance");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Copy initial condition
    amrex::MultiFab::Copy(psi_mf, psi_init_mf, 0, 0, 1, 2);

    // Pseudo-time step (CFL condition for Equation 10)
    Set::Scalar dt_reinit = 0.5 * std::min(DX[0], DX[1]);

    // Temporary storage for iteration
    amrex::MultiFab psi_old(psi_mf.boxArray(), psi_mf.DistributionMap(), 1, 2);

    // Iterative reinitialization (Equation 10)
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Copy current state
        amrex::MultiFab::Copy(psi_old, psi_mf, 0, 0, 1, 2);

        // Update
        for (amrex::MFIter mfi(psi_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> psi_init = psi_init_mf.array(mfi);
            Set::Patch<const Set::Scalar> psi_o = psi_old.array(mfi);
            Set::Patch<Set::Scalar> psi_n = psi_mf.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto sten = Numeric::GetStencil(i, j, k, domain);

                // Smoothed sign function (from Equation 9)
                // S(psi) ~= tanh(psi/(2eps_h))
                Set::Scalar sign_psi = std::tanh(psi_init(i, j, k) / (2.0 * epsilon));

                // Godunov upwind gradient magnitude
                Set::Scalar grad_mag = 0.0;

                if (sign_psi > 0.0)
                {
                    // Backward differences
                    Set::Scalar Dx_minus = (psi_o(i, j, k) - psi_o(i - 1, j, k)) / DX[0];
                    Set::Scalar Dx_plus = (psi_o(i + 1, j, k) - psi_o(i, j, k)) / DX[0];
                    Set::Scalar Dy_minus = (psi_o(i, j, k) - psi_o(i, j - 1, k)) / DX[1];
                    Set::Scalar Dy_plus = (psi_o(i, j + 1, k) - psi_o(i, j, k)) / DX[1];

                    Set::Scalar grad_x = std::max(Dx_minus, 0.0) * std::max(Dx_minus, 0.0)
                                         + std::min(Dx_plus, 0.0) * std::min(Dx_plus, 0.0);
                    Set::Scalar grad_y = std::max(Dy_minus, 0.0) * std::max(Dy_minus, 0.0)
                                         + std::min(Dy_plus, 0.0) * std::min(Dy_plus, 0.0);

                    grad_mag = std::sqrt(grad_x + grad_y);
                }
                else
                {
                    // Forward differences
                    Set::Scalar Dx_minus = (psi_o(i, j, k) - psi_o(i - 1, j, k)) / DX[0];
                    Set::Scalar Dx_plus = (psi_o(i + 1, j, k) - psi_o(i, j, k)) / DX[0];
                    Set::Scalar Dy_minus = (psi_o(i, j, k) - psi_o(i, j - 1, k)) / DX[1];
                    Set::Scalar Dy_plus = (psi_o(i, j + 1, k) - psi_o(i, j, k)) / DX[1];

                    Set::Scalar grad_x = std::min(Dx_minus, 0.0) * std::min(Dx_minus, 0.0)
                                         + std::max(Dx_plus, 0.0) * std::max(Dx_plus, 0.0);
                    Set::Scalar grad_y = std::min(Dy_minus, 0.0) * std::min(Dy_minus, 0.0)
                                         + std::max(Dy_plus, 0.0) * std::max(Dy_plus, 0.0);

                    grad_mag = std::sqrt(grad_x + grad_y);
                }

                // Update using Equation (10): dpsi/dtau_ = S(psi)(1 - |grad_psi|)
                psi_n(i, j, k) = psi_o(i, j, k) + dt_reinit * sign_psi * (1.0 - grad_mag);
            });
        }

        // Fill boundaries
        psi_mf.FillBoundary(geom[lev].periodicity());

        // Check convergence
        amrex::MultiFab residual(psi_mf.boxArray(), psi_mf.DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(residual, psi_mf, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(residual, psi_old, 0, 0, 1, 0);

        Set::Scalar max_residual = residual.norm0();

        if (max_residual < reinit_tolerance)
        {
            Util::Message(INFO, "  Reinitialization converged in ", iter + 1, " iterations");
            break;
        }
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////// ERROR CHECKING ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Inputs:
//      time : integer of timestep
//      lev  : level integer
//      i, j, k : Cell posistions
//      msg  : string message to display. Use this to denote which section of the code the funciton is being called.
//      { "name",name(i,j,k) } : Format Name Value Pair. Duplicate this pair for all values to check. Name shows the output when it fails, value is the actual value.
// Outputs:
//      Will display an error message showing ALL variables with the user defined message to which part of the code failed.
void Hydro2::check4nans(int time, int lev, int i, int j, int k, const std::string &message, std::initializer_list<std::pair<std::string, double> > vars)
{
    // NaNs
    bool hasNaN = false;
    for (const auto &[name, value] : vars)
    {
        if (value != value)
        { // NaN check
            hasNaN = true;
            break;
        }
    }

    // NaNs
    bool hasInf = false;
    for (const auto &[name, value] : vars)
    {
        if (!std::isfinite(value))
        { // Inf check
            hasInf = true;
            break;
        }
    }

    if (hasNaN || hasInf)
    {
        Util::ParallelMessage(INFO, "-------------------------------");
        Util::ParallelMessage(INFO, message);
        Util::ParallelMessage(INFO, "time=", time);
        Util::ParallelMessage(INFO, "lev=", lev);
        Util::ParallelMessage(INFO, "i=", i, ", j=", j, ", k=", k);

        for (const auto &[name, value] : vars)
        {
            Util::ParallelMessage(INFO, name, "=", value);
        }

        Util::Abort(INFO);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// BOUNDRY CONDITIONS //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// FillBoundaries(): Fill Boundries with Periodicity
// Example Usage: FillBoundaries(lev, time, {x_mf[lev].get(), y_mf[lev].get(), z_mf[lev].get()});
void Hydro2::FillBoundaries(int lev, std::initializer_list<amrex::MultiFab *> mfs)
{
    BL_PROFILE("Integrator::Hydro2::FillBoundaries");
    for (auto *mf : mfs)
    {
        if (mf != nullptr)
        {
            mf->FillBoundary(geom[lev].periodicity());
        }
        
        // Checking
        bool err_verbose = false;
        if (err_verbose && (mf->contains_nan()))
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundaries");
            Util::Abort(INFO);
        }
    }
}

// FillBoundariesWithBC(): Fill Boundries with BC
// Example Usage: FillBoundariesWithBC(lev, time, x_bc, {x_mf[lev].get(), y_mf[lev].get(), z_mf[lev].get()});
void Hydro2::FillBoundariesWithBC(int lev, Set::Scalar time, BC::BC<Set::Scalar> *bc, std::initializer_list<amrex::MultiFab *> mfs)
{
    BL_PROFILE("Integrator::Hydro2::FillBoundariesWithBC");
    for (auto *mf : mfs)
    {
        if (mf != nullptr)
        {
            mf->FillBoundary(geom[lev].periodicity());
            if (bc != nullptr)
            {
                bc->FillBoundary(*mf, 0, mf->nComp(), time, 0);
            }
        }

        // Checking
        bool err_verbose = false;
        if (err_verbose && (mf->contains_nan()))
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundariesWithBC");
            Util::Abort(INFO);
        }
    }
}

// ZeroDerivedScratchFields(): Single source of truth for the list of
// derived/scratch MultiFabs that need to be zeroed at IC time and after
// every regrid
void Hydro2::ZeroDerivedScratchFields(int lev)
{
    BL_PROFILE("Integrator::Hydro2::ZeroDerivedScratchFields");
    FillBoundariesWithZero(lev, {
        Source_mf[lev].get(),               // RHS source / forcing
        Fsv_mf[lev].get(),                  // surface tension
        Fw_mf[lev].get(),                   // weight / body force
        Ldot_mf[lev].get(),                 // Lagrange / IB term
        Vap_dot_mf[lev].get(),              // vaporization tracker
        rho_flux_mf[lev].get(),             // Riemann mass flux divergence
        M_flux_mf[lev].get(),               // Riemann momentum flux divergence
        E_flux_mf[lev].get(),               // Riemann energy flux divergence
        div_tau_mf[lev].get(),              // viscous stress divergence
        hess_u_mf[lev].get(),               // velocity Hessian (debug)
        grad_eta_mf[lev].get(),             // grad(eta)
        kappas_mf[lev].get(),               // curvature
        grad_mag_grad_eta_mf[lev].get(),    // |grad(eta)| gradient
        eta_x_mf[lev].get(),               // eta gradient x
        eta_y_mf[lev].get(),               // eta gradient y
        gradmag_mf[lev].get(),             // |grad(eta)|
        nx_smoothed_mf[lev].get(),         // smoothed normal x
        ny_smoothed_mf[lev].get(),         // smoothed normal y
        kappa_SF_mf[lev].get(),            // smooth-normal curvature
        n_hat_mf[lev].get(),                // interface normal
        hess_eta_mf[lev].get(),             // Hessian of eta
        mu_chem_mf[lev].get(),              // chemical potential
        Bm_mf[lev].get(),                   // Spalding number
        Y_mf[lev].get()                     // mass fraction (diagnostic)
    });
}

// FillBoundariesWithZero(): Fill Boundries with Zero
// Example Usage: FillBoundariesWithZero(lev, { x_mf[lev].get(), y_mf[lev].get() });
void Hydro2::FillBoundariesWithZero(int lev, std::initializer_list<amrex::MultiFab *> mfs)
{
    BL_PROFILE("Integrator::Hydro2::FillBoundariesWithZero");
    (void)lev;
    for (auto *mf : mfs)
    {
        if (mf != nullptr)
        {
            mf->setVal(0.0);
        }

        // Checking
        bool err_verbose = false;
        if (err_verbose && mf != nullptr && mf->contains_nan())
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundariesWithZero");
            Util::Abort(INFO);
        }
    }
}


/// ============================================================================
/// FillGhost4BC(): Unified ghost cell filling for all boundary conditions
/// ============================================================================
///
/// This function handles ghost cell filling for both NSCBC and standard BCs.
/// It ensures proper ordering: compute domain quantities -> fill conservatives
/// -> compute ghost primitives, maintaining consistency between all fields.
///
/// CRITICAL ORDERING (to prevent NaN propagation):
///   1. Compute rho, eta in domain
///   2. Fill eta in ghosts (NSCBC needs this for EOS)
///   3. Compute primitives in domain (NSCBC needs these for characteristics)
///   4. Call NSCBC or standard BCs to fill conservative ghosts
///   5. Update rho_eta0, rho_eta1 from NSCBC-modified rho_total
///   6. Compute primitives in ghosts (using NSCBC's gamma/p if available)
///   7. Enforce consistency and check for NaN/Inf
///
/// @param lev   AMR level
/// @param time  Current simulation time
/// ============================================================================
void Hydro2::FillGhost4BC(int lev, Set::Scalar time)
{
    BL_PROFILE("Integrator::Hydro2::FillGhost4BC");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Ensure eta_bc has a valid geometry for FillBoundary calls.
    // The base Integrator calls define(geom) on registered BCs after
    // Initialize(), but eta_bc may not yet be defined on this level
    // if this is the first call. define() is idempotent.
    eta_bc->define(geom[lev]);

    // ------------------------------------------------------------
    // STEP 1: Determine BC strategy based on nghost and NSCBC flag
    // ------------------------------------------------------------
    bool use_nscbc = (nscbc_bc != nullptr) || (nscbc4_bc != nullptr);
    int effective_nghost = nghost;

    if (use_nscbc)
    {
        // NSCBC objects were already created in Parse()
        // Just verify the correct one exists for this nghost
        if (nghost == 2 && nscbc_bc == nullptr)
        {
            Util::Abort(INFO, "nghost=2 but nscbc_bc is null - check Parse()");
        }
        else if (nghost == 4 && nscbc4_bc == nullptr)
        {
            Util::Abort(INFO, "nghost=4 but nscbc4_bc is null - check Parse()");
        }

        if (nghost == 2)
        {
            Util::Message(INFO, "FillGhost4BC: Using NSCBC with 2 ghost cells");
        }
        else if (nghost == 4)
        {
            Util::Message(INFO, "FillGhost4BC: Using NSCBC4 with 4 ghost cells");
        }
    }
    else
    {
        effective_nghost = 2;
    }

    // Fill Patches for regridding
    if (lev > 0)
    {
        FillPatch(lev, time, rho_eta0_mf,        *rho_eta0_mf[lev],       *density_bc,   0);
        FillPatch(lev, time, rho_eta1_mf,        *rho_eta1_mf[lev],       *density_bc,   0);
        FillPatch(lev, time, momentum_mf,        *momentum_mf[lev],       *momentum_bc,  0);
        FillPatch(lev, time, energy_per_vol_mf,  *energy_per_vol_mf[lev], *energy_bc,    0);
        FillPatch(lev, time, eta_mf,             *eta_mf[lev],            *eta_bc,       0);
    }

    // ------------------------------------------------------------
    // STEP 2: Compute total density in DOMAIN; clamp eta.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ------------------------------------------------------------
    // STEP 3: Fill Eta ghost cells
    // ------------------------------------------------------------
    // Eta must use its own BC (eta_bc), NOT energy_bc.
    // energy_bc has dirichlet values meant for the total energy field
    // (e.g. 1685653 at xlo), which would overwrite eta ghost cells with
    // nonsensical values (clamped to 1.0 or 0.0). eta_bc defaults to
    // zero-neumann: the flow carries eta into the domain via advection;
    // ghost cells should mirror the interior gradient.
    FillBoundariesWithBC(lev, time, eta_bc, {
            eta_mf[lev].get()
     });


    // ------------------------------------------------------------
    // STEP 4: Compute primitives in DOMAIN
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY

        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);
        auto UE = UE_per_vol_mf[lev]->array(mfi);
        auto KE = KE_per_vol_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Velocity
            v(i, j, k, 0) = M(i, j, k, 0) / rho(i, j, k);
            v(i, j, k, 1) = M(i, j, k, 1) / rho(i, j, k);

            // Limiting Velocity
            /*
            Set::Scalar u_limit = 1e8;
            v(i, j, k, 0) = (v(i, j, k, 0) < 0.0) ? std::max(v(i, j, k, 0), -u_limit) : std::min(v(i, j, k, 0), u_limit);
            v(i, j, k, 1) = (v(i, j, k, 1) < 0.0) ? std::max(v(i, j, k, 1), -u_limit) : std::min(v(i, j, k, 1), u_limit);
            */

            // Kinetic energy
            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            // Mixed EOS properties
            gamma(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);

            // Internal energy, floored to the SAME eps_p-equivalent ue_floor the
            // post-update backstop uses. The old clamp to `small` only caught
            // UE < 0; but for the stiffened liquid p = (gamma-1)UE - gamma*p0 is
            // already hugely negative for any UE below ue_floor (~7e8), so the
            // weak clamp let negative pressure into the Riemann reconstruction
            // even while the plotted (post-update) pressure read eps_p. This is
            // the pressure the solver actually consumes, so it must carry the
            // same floor or the backstop never reaches the flux computation.
            Set::Scalar p_floor = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0_eff(i, j, k), eps_p, p_cav);
            Set::Scalar ue_floor = (p_floor + gamma(i, j, k) * p0_eff(i, j, k) - pref) / (gamma(i, j, k) - 1.0);
            UE(i, j, k) = std::max(E(i, j, k) - KE(i, j, k), ue_floor);

            // Pressure from Tammann EOS
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);

            // Temperature
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Sound speed
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gamma(i, j, k), p0_eff(i, j, k), small);
        });
    }

    // ------------------------------------------------------------
    // STEP 5: Fill CONSERVATIVE ghost cells (rho, M, E)
    // ------------------------------------------------------------
    if (use_nscbc)
    {
        // ====================================================================
        // NSCBC PATH: Use characteristic-based boundary conditions
        // ====================================================================

        // NSCBC operates on total density, not phase densities
        // Compute rho_total in domain + ghosts
        amrex::MultiFab rho_total(rho_eta0_mf[lev]->boxArray(),
                                  rho_eta0_mf[lev]->DistributionMap(),
                                  1,
                                  nghost);

        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.growntilebox(nghost);
            auto rho = rho_total.array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }

        // Create working copies for NSCBC (it modifies them in-place)
        amrex::MultiFab M_copy(momentum_mf[lev]->boxArray(),
                               momentum_mf[lev]->DistributionMap(),
                               AMREX_SPACEDIM,
                               nghost);
        amrex::MultiFab E_copy(energy_per_vol_mf[lev]->boxArray(),
                               energy_per_vol_mf[lev]->DistributionMap(),
                               1,
                               nghost);

        amrex::MultiFab::Copy(M_copy, *momentum_mf[lev], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(E_copy, *energy_per_vol_mf[lev], 0, 0, 1, nghost);

        // ====================================================================
        // CALL NSCBC TO FILL GHOST CELLS
        // ====================================================================
        // NSCBC will:
        //   - Read eta, gamma, p0, pressure from domain (computed above)
        //   - Compute characteristic waves from interior gradients
        //   - Modify incoming waves based on BC type
        //   - Fill rho_total, M, E in ghost cells
        //   - Write gamma, p0, pressure to ghost cells
        // ====================================================================
        if (nghost == 2 && nscbc_bc != nullptr)
        {
            nscbc_bc->FillBoundary(rho_total, M_copy, E_copy, *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev], eos0, eos1, geom[lev], time, pref);
        }
        else if (nghost == 4 && nscbc4_bc != nullptr)
        {
            nscbc4_bc->FillBoundary(rho_total, M_copy, E_copy, *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev], eos0, eos1, geom[lev], time, pref);
        }

        // Copy modified conservatives back to main arrays
        amrex::MultiFab::Copy(*momentum_mf[lev], M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_copy, 0, 0, 1, nghost);

        // ====================================================================
        // Update phase densities from NSCBC-modified rho_total
        // ====================================================================
        // NSCBC filled rho_total in ghosts, but we need rho_eta0, rho_eta1
        // Partition using eta: rho_eta0 = rho_total * eta
        // ====================================================================
        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);

            auto rho = rho_total.array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // Partition total density by volume fraction
                rho0(i, j, k) = rho(i, j, k) * eta(i, j, k);
                rho1(i, j, k) = rho(i, j, k) * (1.0 - eta(i, j, k));

                // Enforce positivity
                /*
                rho0(i, j, k) = std::max(rho0(i, j, k), small);
                rho1(i, j, k) = std::max(rho1(i, j, k), small);
                */
            });
        }

        // Update density_mf from rho_total
        amrex::MultiFab::Copy(*density_mf[lev], rho_total, 0, 0, 1, nghost);
    }
    else
    {
        // Density: apply density_bc to the TOTAL density (density_mf),
        // then partition into rho_eta0/rho_eta1 using eta.
        // The user-specified dirichlet value (e.g. 39.0 at xlo) is the
        // total mixture density. Applying it directly to both rho_eta0
        // AND rho_eta1 would double the total density in ghost cells.
        // Inter-fab exchange for phase densities still needed:
        rho_eta0_mf[lev]->FillBoundary(geom[lev].periodicity());
        rho_eta1_mf[lev]->FillBoundary(geom[lev].periodicity());
        FillBoundariesWithBC(lev, time, density_bc, {
            density_mf[lev].get()
        });
        // Partition total density by eta in ghost cells (mirrors NSCBC path)
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho  = density_mf[lev]->array(mfi);
            auto eta  = eta_mf[lev]->array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho0(i, j, k) = rho(i, j, k) * eta(i, j, k);
                rho1(i, j, k) = rho(i, j, k) * (1.0 - eta(i, j, k));
            });
        }
        // Momentum
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        // Energy
        FillBoundariesWithBC(lev, time, energy_bc, {
            energy_per_vol_mf[lev].get()
        });

        // Zero Gradient Fill
        if (nghost > effective_nghost)
        {
            for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &validbox = mfi.validbox();
                const amrex::Box &ghostEffbox = mfi.growntilebox(effective_nghost);
                const amrex::Box &ghostNbox = mfi.growntilebox(nghost);

                auto rho0 = rho_eta0_mf[lev]->array(mfi);
                auto rho1 = rho_eta1_mf[lev]->array(mfi);
                auto M = momentum_mf[lev]->array(mfi);
                auto E = energy_per_vol_mf[lev]->array(mfi);

                amrex::ParallelFor(ghostNbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // If in outer ghost layers (beyond layer 2)
                    if (!ghostEffbox.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                    {
                        // Find nearest layer-2 ghost cell and copy its value
                        int i_copy = i;
                        int j_copy = j;

                        // Clamp to layer-2 boundary
                        if (i < ghostEffbox.smallEnd(0))
                            i_copy = ghostEffbox.smallEnd(0);
                        if (i > ghostEffbox.bigEnd(0))
                            i_copy = ghostEffbox.bigEnd(0);
                        if (j < ghostEffbox.smallEnd(1))
                            j_copy = ghostEffbox.smallEnd(1);
                        if (j > ghostEffbox.bigEnd(1))
                            j_copy = ghostEffbox.bigEnd(1);

                        // Zero-gradient extrapolation (copy from layer 2)
                        rho0(i, j, k) = rho0(i_copy, j_copy, k);
                        rho1(i, j, k) = rho1(i_copy, j_copy, k);
                        M(i, j, k, 0) = M(i_copy, j_copy, k, 0);
                        M(i, j, k, 1) = M(i_copy, j_copy, k, 1);
                        E(i, j, k) = E(i_copy, j_copy, k);
                    }
                });
            }
        }
    }

    // ------------------------------------------------------------
    // STEP 6: Update total density in GHOST CELLS; clamp eta.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ------------------------------------------------------------
    // STEP 7: Compute primitives in GHOST CELLS
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);
        auto UE = UE_per_vol_mf[lev]->array(mfi);
        auto KE = KE_per_vol_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        // Compute 
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            v(i, j, k, 0) = M(i, j, k, 0) / rho(i, j, k);
            v(i, j, k, 1) = M(i, j, k, 1) / rho(i, j, k);

            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            UE(i, j, k) = E(i, j, k) - KE(i, j, k);
            UE(i, j, k) = (UE(i, j, k) < 0.0) ? small : UE(i, j, k);

            //if (!(use_nscbc))
            //{
                gamma(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
                p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
                press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);
            //}
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gamma(i, j, k), p0_eff(i, j, k), small);
        });
    }

    // ------------------------------------------------------------
    // STEP 8: Repair any remaining NaN in ALL cells (domain + ghosts)
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Check for NaN in any field
            bool has_nan = !std::isfinite(rho_eta0(i, j, k)) || !std::isfinite(rho_eta1(i, j, k)) || !std::isfinite(rho(i, j, k)) || !std::isfinite(eta(i, j, k)) || !std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1)) || !std::isfinite(E(i, j, k)) || !std::isfinite(press(i, j, k)) || !std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1)) || !std::isfinite(T(i, j, k)) || !std::isfinite(a(i, j, k)) || !std::isfinite(gamma(i, j, k)) || !std::isfinite(p0_eff(i, j, k));

            if (has_nan)
            {
                // Repair by copying from nearest valid neighbor

                // Try left neighbor (i-1)
                if (i > ghostbox.smallEnd(0))
                {
                    bool left_valid = std::isfinite(rho_eta0(i - 1, j, k)) && std::isfinite(E(i - 1, j, k)) && std::isfinite(press(i - 1, j, k));
                    if (left_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i - 1, j, k);
                        rho_eta1(i, j, k) = rho_eta1(i - 1, j, k);
                        rho(i, j, k) = rho(i - 1, j, k);
                        eta(i, j, k) = eta(i - 1, j, k);
                        M(i, j, k, 0) = M(i - 1, j, k, 0);
                        M(i, j, k, 1) = M(i - 1, j, k, 1);
                        E(i, j, k) = E(i - 1, j, k);
                        press(i, j, k) = press(i - 1, j, k);
                        v(i, j, k, 0) = v(i - 1, j, k, 0);
                        v(i, j, k, 1) = v(i - 1, j, k, 1);
                        T(i, j, k) = T(i - 1, j, k);
                        a(i, j, k) = a(i - 1, j, k);
                        gamma(i, j, k) = gamma(i - 1, j, k);
                        p0_eff(i, j, k) = p0_eff(i - 1, j, k);
                        return; // Fixed, exit early
                    }
                }

                // Try right neighbor (i+1)
                if (i < ghostbox.bigEnd(0))
                {
                    bool right_valid = std::isfinite(rho_eta0(i + 1, j, k)) && std::isfinite(E(i + 1, j, k)) && std::isfinite(press(i + 1, j, k));
                    if (right_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i + 1, j, k);
                        rho_eta1(i, j, k) = rho_eta1(i + 1, j, k);
                        rho(i, j, k) = rho(i + 1, j, k);
                        eta(i, j, k) = eta(i + 1, j, k);
                        M(i, j, k, 0) = M(i + 1, j, k, 0);
                        M(i, j, k, 1) = M(i + 1, j, k, 1);
                        E(i, j, k) = E(i + 1, j, k);
                        press(i, j, k) = press(i + 1, j, k);
                        v(i, j, k, 0) = v(i + 1, j, k, 0);
                        v(i, j, k, 1) = v(i + 1, j, k, 1);
                        T(i, j, k) = T(i + 1, j, k);
                        a(i, j, k) = a(i + 1, j, k);
                        gamma(i, j, k) = gamma(i + 1, j, k);
                        p0_eff(i, j, k) = p0_eff(i + 1, j, k);
                        return;
                    }
                }

                // Try bottom neighbor (j-1)
                if (j > ghostbox.smallEnd(1))
                {
                    bool bottom_valid = std::isfinite(rho_eta0(i, j - 1, k)) && std::isfinite(E(i, j - 1, k)) && std::isfinite(press(i, j - 1, k));
                    if (bottom_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i, j - 1, k);
                        rho_eta1(i, j, k) = rho_eta1(i, j - 1, k);
                        rho(i, j, k) = rho(i, j - 1, k);
                        eta(i, j, k) = eta(i, j - 1, k);
                        M(i, j, k, 0) = M(i, j - 1, k, 0);
                        M(i, j, k, 1) = M(i, j - 1, k, 1);
                        E(i, j, k) = E(i, j - 1, k);
                        press(i, j, k) = press(i, j - 1, k);
                        v(i, j, k, 0) = v(i, j - 1, k, 0);
                        v(i, j, k, 1) = v(i, j - 1, k, 1);
                        T(i, j, k) = T(i, j - 1, k);
                        a(i, j, k) = a(i, j - 1, k);
                        gamma(i, j, k) = gamma(i, j - 1, k);
                        p0_eff(i, j, k) = p0_eff(i, j - 1, k);
                        return;
                    }
                }

                // Try top neighbor (j+1)
                if (j < ghostbox.bigEnd(1))
                {
                    bool top_valid = std::isfinite(rho_eta0(i, j + 1, k)) && std::isfinite(E(i, j + 1, k)) && std::isfinite(press(i, j + 1, k));
                    if (top_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i, j + 1, k);
                        rho_eta1(i, j, k) = rho_eta1(i, j + 1, k);
                        rho(i, j, k) = rho(i, j + 1, k);
                        eta(i, j, k) = eta(i, j + 1, k);
                        M(i, j, k, 0) = M(i, j + 1, k, 0);
                        M(i, j, k, 1) = M(i, j + 1, k, 1);
                        E(i, j, k) = E(i, j + 1, k);
                        press(i, j, k) = press(i, j + 1, k);
                        v(i, j, k, 0) = v(i, j + 1, k, 0);
                        v(i, j, k, 1) = v(i, j + 1, k, 1);
                        T(i, j, k) = T(i, j + 1, k);
                        a(i, j, k) = a(i, j + 1, k);
                        gamma(i, j, k) = gamma(i, j + 1, k);
                        p0_eff(i, j, k) = p0_eff(i, j + 1, k);
                        return;
                    }
                }

                Util::Message(INFO, "=== NaN DETECTED IN FillGhost4BC ===");
                Util::Message(INFO, "Level: ", lev);
                Util::Message(INFO, "Time: ", time);
                Util::Message(INFO, "NSCBC mode: ", use_nscbc);
                Util::Message(INFO, "All neighbors are NaNs, unable to patch cell");
                Util::Abort(INFO, "NaN detected in FillGhost4BC - see details above");
            }
        });
    }
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
        });
    }

    // ------------------------------------------------------------
    // STEP 9: Enforce consistency and positivity in ALL cells
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Enforce rho = rho_eta0 + rho_eta1
            Set::Scalar rho_calc = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            rho(i, j, k) = rho_calc;

            // Enforce positivity
            rho(i, j, k) = std::max(rho(i, j, k), small);
            rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), small);
            rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), small);
            press(i, j, k) = std::max(press(i, j, k), 1.0e-6);
            E(i, j, k) = std::max(E(i, j, k), 1.0e-10);

            // Clamp eta to [0, 1]
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ========================================================================
    // STEP 10: NaN/Inf detection and reporting
    // ========================================================================
    // Check all critical fields for NaN/Inf in ghost cells.
    // If found, report location and abort with diagnostic info.
    // ========================================================================
    bool found_nan = false;
    std::string nan_details = "";

    const amrex::Box &domain_box = geom[lev].Domain();
    int domain_ilo = domain_box.smallEnd(0);
    int domain_ihi = domain_box.bigEnd(0);
    int domain_jlo = domain_box.smallEnd(1);
    int domain_jhi = domain_box.bigEnd(1);

    // Lambda to describe cell location
    auto describe_location = [&](int i, int j, int k, const amrex::Box &validbox) -> std::string {
        std::stringstream ss;

        // Distance from domain boundaries
        int dist_from_xlo = i - domain_ilo;
        int dist_from_xhi = domain_ihi - i;
        int dist_from_ylo = j - domain_jlo;
        int dist_from_yhi = domain_jhi - j;

        // Distance from valid box boundaries
        int dist_from_valid_xlo = i - validbox.smallEnd(0);
        int dist_from_valid_xhi = validbox.bigEnd(0) - i;
        int dist_from_valid_ylo = j - validbox.smallEnd(1);
        int dist_from_valid_yhi = validbox.bigEnd(1) - j;

        ss << "Cell (" << i << "," << j << "," << k << ")\n";
        ss << "  Domain: [" << domain_ilo << "," << domain_ihi << "] x ["
           << domain_jlo << "," << domain_jhi << "]\n";
        ss << "  ValidBox: [" << validbox.smallEnd(0) << "," << validbox.bigEnd(0) << "] x ["
           << validbox.smallEnd(1) << "," << validbox.bigEnd(1) << "]\n";

        // Determine location type
        bool in_domain = (i >= domain_ilo && i <= domain_ihi && j >= domain_jlo && j <= domain_jhi);
        bool in_valid = validbox.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));

        if (in_valid)
        {
            ss << "  Location: INTERIOR (valid box)\n";
        }
        else if (in_domain)
        {
            ss << "  Location: DOMAIN but outside valid box\n";
        }
        else
        {
            ss << "  Location: GHOST CELL\n";

            // Determine which boundary
            if (i < domain_ilo)
            {
                ss << "    X-LO boundary (ghost layer " << (domain_ilo - i) << ")\n";
            }
            else if (i > domain_ihi)
            {
                ss << "    X-HI boundary (ghost layer " << (i - domain_ihi) << ")\n";
            }

            if (j < domain_jlo)
            {
                ss << "    Y-LO boundary (ghost layer " << (domain_jlo - j) << ")\n";
            }
            else if (j > domain_jhi)
            {
                ss << "    Y-HI boundary (ghost layer " << (j - domain_jhi) << ")\n";
            }

            // Check if corner
            if ((i < domain_ilo || i > domain_ihi) && (j < domain_jlo || j > domain_jhi))
            {
                ss << "    CORNER CELL\n";
            }
        }

        ss << "  Distance from domain boundaries:\n";
        ss << "    X-LO: " << dist_from_xlo << " cells\n";
        ss << "    X-HI: " << dist_from_xhi << " cells\n";
        ss << "    Y-LO: " << dist_from_ylo << " cells\n";
        ss << "    Y-HI: " << dist_from_yhi << " cells\n";

        return ss.str();
    };

    // Check each field and report first NaN location
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &validbox = mfi.validbox();
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto p = pressure_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        // Scan for NaN (CPU loop for detailed reporting)
        amrex::LoopOnCpu(ghostbox, [&](int i, int j, int k) {
            // Check density
            if (!found_nan && !std::isfinite(rho(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: density\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(rho(i, j, k)) + "\n";
            }

            // Check momentum
            if (!found_nan && (!std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1))))
            {
                found_nan = true;
                nan_details = "Field: momentum\n" + describe_location(i, j, k, validbox);
                nan_details += "M[0]: " + std::to_string(M(i, j, k, 0)) + "\n";
                nan_details += "M[1]: " + std::to_string(M(i, j, k, 1)) + "\n";
            }

            // Check energy
            if (!found_nan && !std::isfinite(E(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: energy\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(E(i, j, k)) + "\n";
            }

            // Check pressure
            if (!found_nan && !std::isfinite(p(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: pressure\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(p(i, j, k)) + "\n";
            }

            // Check velocity
            if (!found_nan && (!std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1))))
            {
                found_nan = true;
                nan_details = "Field: velocity\n" + describe_location(i, j, k, validbox);
                nan_details += "v[0]: " + std::to_string(v(i, j, k, 0)) + "\n";
                nan_details += "v[1]: " + std::to_string(v(i, j, k, 1)) + "\n";
            }

            // Check eta
            if (!found_nan && !std::isfinite(eta(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: eta\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(eta(i, j, k)) + "\n";
            }
        });

        // If NaN found, stop scanning
        if (found_nan)
            break;
    }

    if (found_nan)
    {
        Util::Message(INFO, "=== NaN DETECTED IN FillGhost4BC ===");
        Util::Message(INFO, "Level: ", lev);
        Util::Message(INFO, "Time: ", time);
        Util::Message(INFO, "NSCBC mode: ", use_nscbc);
        Util::Message(INFO, "\n" + nan_details);
        Util::Abort(INFO, "NaN detected in FillGhost4BC - see details above");
    }
} // end FillGhost4BC()


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// POST-SUBCYCLE REFLUX /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::PostSubcycleReflux(int lev, Set::Scalar /*time*/, Set::Scalar /*dt_coarse*/)
{
    BL_PROFILE("Integrator::Hydro2::PostSubcycleReflux");

    if (lev >= finest_level) return;

    int fine_lev = lev + 1;
    if (fine_lev >= (int)flux_reg.size() || !flux_reg[fine_lev]) return;

    // Apply the reflux correction to coarse-level conserved variables.
    // The FluxRegister accumulated:
    //   fine_flux * dt_fine  (positive, from FineAdd across all sub-steps)
    //   coarse_flux * dt_coarse (negative, from CrseInit)
    // Reflux() applies:  U_coarse += scale * (1/vol) * sum_faces(register)
    // Register layout: [rho_eta0, rho_eta1, mom_x, mom_y, energy]

    // Reflux per-phase densities directly
    flux_reg[fine_lev]->Reflux(*rho_eta0_mf[lev],
                               1.0,        // scale
                               0,          // src component in register
                               0,          // dst component
                               1,          // ncomp
                               geom[lev]);

    flux_reg[fine_lev]->Reflux(*rho_eta1_mf[lev],
                               1.0,
                               1,          // src component in register
                               0,          // dst component
                               1,
                               geom[lev]);

    // Reflux momentum
    flux_reg[fine_lev]->Reflux(*momentum_mf[lev],
                               1.0,
                               2,          // src component in register
                               0,          // dst component
                               AMREX_SPACEDIM,
                               geom[lev]);

    // Reflux energy
    flux_reg[fine_lev]->Reflux(*energy_per_vol_mf[lev],
                               1.0,
                               2 + AMREX_SPACEDIM,  // src component in register
                               0,                    // dst component
                               1,
                               geom[lev]);

    // Recompute derived fields from the refluxed conserved variables.
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto rho  = density_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);
        auto mom  = momentum_mf[lev]->array(mfi);
        auto vel  = velocity_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            vel(i, j, k, 0) = mom(i, j, k, 0) / rho(i, j, k);
            vel(i, j, k, 1) = mom(i, j, k, 1) / rho(i, j, k);
        });
    }
}


///////////////////////////////////////////////////////////////////////////
///////////////////////// CURVATURE PIPELINE ////////////////////////////
///////////////////////////////////////////////////////////////////////////

void Hydro2::ComputeGradEta(int lev)
{
    BL_PROFILE("Hydro2::ComputeGradEta");

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    // Compute eta gradients into the ghost layer (not just valid cells) so the
    // smoothing/curvature stages downstream are coarse-fine- and physical-
    // boundary-correct: eta is FillPatch'd to nghost in FillGhost4BC, but the
    // derived fields only get same-level FillBoundary, which leaves patch-edge
    // ghosts stale. gE is the ghost depth we fill; reading eta at +/-1 needs eta
    // valid to gE+1 (<= nghost). Targets normals 1-deep -> eta_x krad+1 deep.
    // Non-tiled MFIter avoids the grown-tile write race.
    const int krad = (smooth_kernel_size >= 5) ? 2 : 1;
    const int gE = std::max(0, std::min(krad + 1, nghost - 1));

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.growntilebox(gE);

        auto eta_arr  = eta_mf[lev]->const_array(mfi);
        auto etax_arr = eta_x_mf[lev]->array(mfi);
        auto etay_arr = eta_y_mf[lev]->array(mfi);
        auto gmag_arr = gradmag_mf[lev]->array(mfi);
        auto ge_arr   = grad_eta_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Set::Vector G = Numeric::Gradient(eta_arr, i, j, k, 0, dx);
            amrex::Real gx = G(0);
            amrex::Real gy = G(1);

            etax_arr(i,j,k,0) = gx;
            etay_arr(i,j,k,0) = gy;
            gmag_arr(i,j,k,0) = std::sqrt(gx*gx + gy*gy);
            ge_arr(i,j,k,0)   = gx;
            ge_arr(i,j,k,1)   = gy;
        });
    }

    eta_x_mf[lev]->FillBoundary(geom.periodicity());
    eta_y_mf[lev]->FillBoundary(geom.periodicity());
    gradmag_mf[lev]->FillBoundary(geom.periodicity());
    grad_eta_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeSmoothNormals(int lev)
{
    BL_PROFILE("Hydro2::ComputeSmoothNormals");

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    const int krad = (smooth_kernel_size >= 5) ? 2 : 1;

    // Smooth normals one ghost cell deep so ComputeKappas' divergence (reads n
    // at +/-1) is coarse-fine/physical-boundary correct. Needs eta_x valid to
    // gN+krad, which ComputeGradEta fills (gE = krad+1 when nghost >= krad+2).
    // Falls back to 0 (valid-only, same as before) when the halo is too small.
    const int gN = std::max(0, std::min(1, (nghost - 1) - krad));

    amrex::GpuArray<amrex::Real, 3> gw;
    gw[0] = 1.0;
    gw[1] = std::exp(-0.5);
    gw[2] = std::exp(-2.0);

    for (amrex::MFIter mfi(*nx_smoothed_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.growntilebox(gN);

        auto eta_x   = eta_x_mf[lev]->const_array(mfi);
        auto eta_y   = eta_y_mf[lev]->const_array(mfi);
        auto gradmag = gradmag_mf[lev]->const_array(mfi);
        auto nx_s    = nx_smoothed_mf[lev]->array(mfi);
        auto ny_s    = ny_smoothed_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real accx = 0.0, accy = 0.0, wsum = 0.0;
            const amrex::Box tb_g = amrex::grow(tb, krad);

            for (int di = -krad; di <= krad; ++di)
            for (int dj = -krad; dj <= krad; ++dj)
            {
                int ii = i + di, jj = j + dj;
                if (!tb_g.contains(amrex::IntVect(AMREX_D_DECL(ii, jj, 0)))) continue;

                amrex::Real w    = gw[std::abs(di)] * gw[std::abs(dj)];
                amrex::Real gmN  = gradmag(ii, jj, k) + 1e-14;
                accx += w * eta_x(ii, jj, k) / gmN;
                accy += w * eta_y(ii, jj, k) / gmN;
                wsum += w;
            }

            amrex::Real nx = accx / (wsum + 1e-14);
            amrex::Real ny = accy / (wsum + 1e-14);
            amrex::Real m  = std::sqrt(nx*nx + ny*ny) + 1e-14;
            nx_s(i, j, k) = nx / m;
            ny_s(i, j, k) = ny / m;
        });
    }

    nx_smoothed_mf[lev]->FillBoundary(geom.periodicity());
    ny_smoothed_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeKappas(int lev)
{
    BL_PROFILE("Hydro2::ComputeKappas");

    ComputeGradEta(lev);
    ComputeSmoothNormals(lev);

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    const amrex::Real dx_eff = std::max({dx[0], dx[1], epsilon});
    const amrex::Real Cg = 0.1 / dx_eff;
    const amrex::Real small = 1e-14;

    for (amrex::MFIter mfi(*kappas_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.tilebox();

        auto gradmag_arr = gradmag_mf[lev]->const_array(mfi);
        auto nx_s     = nx_smoothed_mf[lev]->const_array(mfi);
        auto ny_s     = ny_smoothed_mf[lev]->const_array(mfi);
        auto kappas   = kappas_mf[lev]->array(mfi);
        auto kappa_sf = kappa_SF_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real gm = gradmag_arr(i,j,k);

            if (gm < Cg)
            {
                kappas(i,j,k,0) = 0.0;
                kappas(i,j,k,1) = 0.0;
                kappas(i,j,k,2) = 0.0;
                kappa_sf(i,j,k) = 0.0;
                return;
            }

            int il = amrex::max(i-1, domain.smallEnd(0));
            int ir = amrex::min(i+1, domain.bigEnd(0));
            int jl = amrex::max(j-1, domain.smallEnd(1));
            int jr = amrex::min(j+1, domain.bigEnd(1));

            amrex::Real nx_x = 0.5 * (nx_s(ir,j,k) - nx_s(il,j,k)) / dx[0];
            amrex::Real ny_y = 0.5 * (ny_s(i,jr,k) - ny_s(i,jl,k)) / dx[1];

            amrex::Real kSN = -(nx_x + ny_y);

            const amrex::Real kappa_max = 2.0 / amrex::min(dx[0], dx[1]);
            kSN = amrex::max(-kappa_max, amrex::min(kappa_max, kSN));

            kappas(i,j,k,0) = kSN;
            kappas(i,j,k,1) = kSN;
            kappas(i,j,k,2) = 0.0;
            kappa_sf(i,j,k) = kSN;
        });
    }

    kappas_mf[lev]->FillBoundary(geom.periodicity());
}


} // end of Integrator namespace

//#endif
