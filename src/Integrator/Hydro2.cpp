// Base
#include "Hydro2.H"
#include <memory>
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
#include "IC/STL.H"   // 3D STL solid geometry (self-guarded: no-op unless USE_EB + 3D)
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"
// Limiters / primitive-variable reconstruction
#include "Solver/Local/Limiter/Limiter.H"
#include "Solver/Local/Limiter/Godunov.H"
#include "Solver/Local/Limiter/Minmod.H"
#include "Solver/Local/Limiter/VanLeer.H"
#include "Solver/Local/Limiter/WENO3.H"
#include "Solver/Local/Limiter/WENO5.H"
//EOS
#include "Solver/EOS/EOS.H"
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"
// Generic
#include <AMReX_Math.H>
#include "AMReX_TimeIntegrator.H"


namespace Integrator
{

// ----------------------------------------------------------------------
// CAPILLARY ENERGY CLOSURE helpers (see Hydro2.H, capillary_closure).
//
// e_cap = sigma_eff |grad eta| is a STATE FUNCTION of (eta, Gamma), never an
// independently transported field -- so it can be recomputed identically at
// every site that assembles or inverts rho E, and can never drift out of
// step with eta.  Both helpers mirror the sigma_eff evaluation in RHS
// exactly; keep them in sync if the Marmottant branch structure changes.
// ----------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SigmaEffFromGamma(Set::Scalar Gamma, Set::Scalar grad_eta_mag,
                                     int marm, Set::Scalar sig_const,
                                     Set::Scalar chi_, Set::Scalar Gb,
                                     Set::Scalar sbrk, Set::Scalar sigw)
{
    if (grad_eta_mag <= 1.0e-10) return 0.0;   // off the interface: no shell
    if (!marm) return sig_const;
    Set::Scalar se = 0.0;
    if (Gamma > 0.0 && Gamma < Gb)             // stretched -> elastic branch
    {
        const Set::Scalar el = chi_ * (Gb / Gamma - 1.0);
        se = (el >= sbrk) ? sigw : el;
    }
    return se;                                 // Gamma >= Gb: buckled, sigma = 0
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar CapEnergyAt(const amrex::Array4<const Set::Scalar> &eta,
                               const amrex::Array4<const Set::Scalar> &shell,
                               int i, int j, int k, const Set::Scalar *DX,
                               int marm, Set::Scalar sig_const, Set::Scalar chi_,
                               Set::Scalar Gb, Set::Scalar sbrk, Set::Scalar sigw)
{
    // The centred gradient reads i+-1, so the outermost ghost ring has no
    // valid stencil.  e_cap is identically zero away from the interface, and
    // the interface never reaches the outer ghosts in any case, so returning
    // 0 there is exact rather than merely safe.
    if (!eta.contains(i - 1, j, k) || !eta.contains(i + 1, j, k)) return 0.0;
#if AMREX_SPACEDIM > 1
    if (!eta.contains(i, j - 1, k) || !eta.contains(i, j + 1, k)) return 0.0;
#endif
#if AMREX_SPACEDIM > 2
    if (!eta.contains(i, j, k - 1) || !eta.contains(i, j, k + 1)) return 0.0;
#endif
    const Set::Vector ge = Numeric::Gradient(eta, i, j, k, 0, DX);
    const Set::Scalar gem = ge.lpNorm<2>();
    const Set::Scalar G = marm ? shell(i, j, k) : 0.0;
    return SigmaEffFromGamma(G, gem, marm, sig_const, chi_, Gb, sbrk, sigw) * gem;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SpaldingBM(Set::Scalar Y_local, Set::Scalar Y_inf, Set::Scalar small)
{
    return (Y_local - Y_inf) / (1.0 - Y_local + small);
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
        pp.query_default("eta_refinement_criterion", value.eta_refinement_criterion, 0.001);        // eta-based refinement
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01);     // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01);     // velocity gradient-based refinement
        pp.query_default("p_refinement_criterion", value.p_refinement_criterion, 1e-3);             // pressure-based refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);         // density-based refinement
        pp.query_default("phi_refinement_criterion", value.embedded.refinement_criterion, 0.001);   // solid-boundary refinement
        {   // optional geometric refinement box: force refinement to max_level inside [lo,hi]
            amrex::ParmParse ppb("refine_box");
            std::vector<Set::Scalar> lo, hi;
            if (ppb.queryarr("lo", lo) && ppb.queryarr("hi", hi)
                && (int)lo.size() >= AMREX_SPACEDIM && (int)hi.size() >= AMREX_SPACEDIM)
            {
                value.refine_box_on = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { value.refine_box_lo[d] = lo[d]; value.refine_box_hi[d] = hi[d]; }
            }
        }

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);                // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl);  // cfl condition (viscous specific)
        pp_query_default("pref", value.pref, 0.0);          // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);     // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-8);   // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0);  // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);            // Gravitational Acceletation

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, false);  // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("capillary_closure", value.capillary_closure, 0);
        pp_query_default("apply_weight", value.apply_weight, false);                    // Apply weight when solving, default: false --> "No Weight"
        pp_query_default("apply_vaporization", value.apply_vaporization, false);        // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // EMBEDDED SOLID BOUNDARY (see FlowWedge for examples)
        pp_query_default("apply_embedded_solid", value.embedded.apply, 0);  // Apply Solid Boundry 1 --> "Domain has solid boundry"
        pp_query_default("solid.brinkman", value.embedded.brinkman, 0.0);   // Yang(2023) momentum-only Brinkman no-penetration (0=off)
        pp_query_default("solid.slip", value.embedded.slip, 0);             // 1 = slip wall (penalize only wall-normal momentum); 0 = no-slip
        if (value.embedded.apply)
        {
            if (value.embedded.brinkman <= 0.0)
                Util::Message(INFO, "embedded solid: Yang full-flux wall, auto momentum projection (solid.brinkman not set)");
            else
                Util::Message(INFO, "embedded solid: Yang full-flux porous wall, solid.brinkman=", value.embedded.brinkman);
            // Skips pressure relaxation within the solid (acts likes Hydro do-not-solve type solid)
            value.embedded.relax_skip = 0.5;
        }

        // Artificial heat exchange (AHE) on per-phase internal energy rows.
        // Sources:
        //  Schmidmayer 2020 eq. 13 r-source
        //  Saurel-Petitpas-Berry 2009 Sec 5.3 Fig. 21
        // ahe.method: top-level switch that maps to (apply_ahe, ahe.form).
        //   0 = Sch20 stiff-limit only (no explicit AHE source)
        //   1 = Sch20 finite-mu r-source (+/- mu p_I Dp)
        //   2 = Sau09 sec.5.3 q*@u/@x form
        // When set (>= 0) overrides the legacy apply_ahe / ahe.form parses.
        int ahe_method_in = -1;
        pp_query_default("ahe.method",          ahe_method_in, -1);
        pp_query_default("apply_ahe",           value.apply_ahe,           0);
        pp_query_default("ahe.use_const_mu",    value.ahe_use_const_mu,    0);
        pp_query_default("ahe.mu_const",        value.ahe_mu_const,        0.0);
        pp_query_default("ahe.mu_scale",        value.ahe_mu_scale,        1.0);
        pp_query_default("ahe.mu_a",            value.ahe_mu_a,           -5.64e7);
        pp_query_default("ahe.mu_b",            value.ahe_mu_b,            5.34e3);
        pp_query_default("ahe.mu_c",            value.ahe_mu_c,           -25.6);
        pp_query_default("ahe.v_min",           value.ahe_v_min,           2.65e-4);
        pp_query_default("ahe.v_max",           value.ahe_v_max,           4.61e-4);
        pp_query_default("ahe.compression_only",value.ahe_compression_only,0);
        pp_query_default("ahe.apply_alpha_src", value.ahe_apply_alpha_src, 1);
        pp_query_default("ahe.max_frac",        value.ahe_max_frac,        0.1);
        pp_query_default("ahe.form",            value.ahe_form,            1);

        // Apply ahe.method override if the user set it.
        if (ahe_method_in == 0)
        {
            value.ahe_method = 0;
            value.apply_ahe  = 0;            // Sch20 stiff limit -- RelaxAndReinit only (default)
        }
        else if (ahe_method_in == 1)
        {
            value.ahe_method = 1;
            value.apply_ahe  = 1;
            value.ahe_form   = 0;            // Sch20 r-source (antisymmetric)
        }
        else if (ahe_method_in == 2)
        {
            value.ahe_method = 2;
            value.apply_ahe  = 1;
            value.ahe_form   = 1;            // Sau09 q*du/dx
        }
        //else (ahe_method_in == -1): unset -> fall back to legacy flags as parsed.

        // FLUID 0
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient
        
        // FLUID 1
        pp_query_required("mu1", value.mu1);            // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0);    // bulk viscosity coefficient

        // EOS
        Solver::EOS::Tammann::Parse(value.eos0, pp, "eos0.");
        Solver::EOS::Tammann::Parse(value.eos1, pp, "eos1.");

        // PeleC EOS implementation (only works for non-tamman fluids "p0=0")
        //   eos.backend = tammann   (default; equivalent to "native")
        //   eos.backend = pelephysics
        std::string eos_backend_str = "tammann";
        pp_query_default("eos.backend", eos_backend_str, "tammann");
        Solver::EOS::SetBackend(eos_backend_str);

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0);            // Surface tension condition (= sigma_water when marmottant=1)
        // Marmottant (2005) coated-bubble surface tension -- ADVECTED-SHELL form.
        // sigma is a function of the shell's areal strain Gamma_0/Gamma, carried by
        // the conserved shell density c (see Hydro2.H).  No curvature is involved,
        // so the eta/kappa/sigma smoothing passes and the height-function curvature
        // that the old sigma(R(x)) path needed are GONE.
        pp_query_default("marmottant", value.marmottant, 0);                       // 0=const sigma, 1=Marmottant sigma(Gamma)
        pp_query_default("marmottant.R_buckling", value.marmottant_R_buckling, 0.0);   // buckling radius [m]
        pp_query_default("marmottant.chi",        value.marmottant_chi, 0.0);          // shell elastic modulus [N/m]
        pp_query_default("marmottant.sigma_break",value.marmottant_sigma_break, 1.0e30); // rupture tension (default huge = never rupture)
        pp_query_default("marmottant.R0",         value.marmottant_R0, 0.0);           // reference radius: shell starts UNSTRAINED here
        if (value.marmottant)
        {
            if (!(value.marmottant_R0 > 0.0))
                Util::Abort(INFO, "marmottant=1 requires marmottant.R0 > 0 "
                                  "(the radius at which the shell is unstrained, Gamma=Gamma_0=1)");
            if (!(value.marmottant_R_buckling > 0.0))
                Util::Abort(INFO, "marmottant=1 requires marmottant.R_buckling > 0");
            // Gamma_0 == 1 by construction (c is initialized to |grad eta|), so the
            // buckling areal density follows from Marmottant's A/A_buck = (R/R_buck)^2
            // evaluated at the reference state.  This reproduces the previous
            // calibration exactly:  sigma(t=0) = chi ((R0/R_buck)^2 - 1).
            value.marmottant_Gamma_buck = (value.marmottant_R0 * value.marmottant_R0)
                                        / (value.marmottant_R_buckling * value.marmottant_R_buckling);
            Util::Message(INFO, "Marmottant ON (advected shell): R0=", value.marmottant_R0,
                          " R_buckling=", value.marmottant_R_buckling,
                          " chi=", value.marmottant_chi,
                          " Gamma_buck=", value.marmottant_Gamma_buck,
                          " sigma(t=0)=", value.marmottant_chi * (value.marmottant_Gamma_buck - 1.0),
                          " sigma_break=", value.marmottant_sigma_break,
                          " sigma_water=", value.sigma);
        }
        pp_query_default("Dv", value.Dv, 0.0);                  // Vapor Diffusivity
        pp_query_default("Lv", value.Lv, 0.0);                  // Latent heat of vaporization [J/kg] (evaporative cooling sink)
        pp_query_required("epsilon", value.epsilon);            // diffuse interface thickness Y_infinity
        pp_query_default("Y_infinity", value.Y_infinity, 0.0);  // Far Field Vapor Mass Fraction
        pp_query_default("Mob", value.Mob_user, 0.0);           // CH mobility scale M0: M = M0 * epsilon^2
        if (value.epsilon <= 0.0)
        {
            Util::Abort(INFO, "epsilon must be positive for Hydro2 Cahn-Hilliard mobility; got ", value.epsilon);
        }

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 1); // Method to solve for curvature

        // IC pressure equalization during initialization step to prevenent spurious oscilation at the start
        pp_query_default("equalize_ic_pressure", value.equalize_ic_pressure, 0);
        
        // Newton diagnostic for stiff pressure relaxation.
        pp_query_default("nan_check", value.nan_check, 0);  // 1 = per-cell NaN/Inf guard (slow; debugging only)
        pp_query_default("verbose", value.verbose, 0);    // 1 = per-call informational messages
        pp_query_default("relax_diag", value.relax_diag, 0); // 1 = print per-stage {max_iters, max_residual, count_unconverged}.
        pp_query_default("clip_ghost_only", value.clip_ghost_only, 0); // 1 = FillGhost STEP-9 positivity clip touches GHOST cells only (per-phase mass conservation)

        // Symmetry-face detection: a domain face whose normal-momentum BC is
        // REFLECT_ODD is a symmetry plane; its advective fluxes are enforced
        // to the exact u*=0 solution in RHS (see symmetrize()).
        {
            const char* faces[6] = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};
            for (int f = 0; f < 2 * AMREX_SPACEDIM; ++f)
            {
                const int d = f / 2;
                const bool is_lo = (f % 2 == 0);
                std::vector<std::string> mom_bc;
                std::string key = std::string("momentum.bc.type.") + faces[d * 2 + (is_lo ? 0 : 1)];
                if (pp.queryarr(key.c_str(), mom_bc) && (int)mom_bc.size() > d)
                {
                    std::string t = mom_bc[d];
                    for (auto &c : t) c = toupper(c);
                    const bool sym = (t == "REFLECT_ODD");
                    if (is_lo) value.sym_face_lo[d] = sym;
                    else       value.sym_face_hi[d] = sym;
                    if (sym)
                        Util::Message(INFO, "symmetry face detected: ", faces[d * 2 + (is_lo ? 0 : 1)],
                                      " (exact u*=0 flux enforced)");
                }
            }
        }

        // INTERFACE COMPRESSION
        pp_query_default("density_relax", value.omega_relax, 0.5);                          // Relaxation parameter (0.3-0.7 typical)

        // BOUNDRY CONDITITIONS
        pp_query_default("nghost", value.nghost, 2); // Number of Ghost Cells (NOTE: NSCBC can only use 2 or 4 nghost) ### WIP ### Add nghost cabability for nghost

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

        // Primitive (density, momentum, pressure, eta) vs conservative (density, momentum, energy) BC path. Default 0 keeps the existing conservative
        pp_query_default("bc.primitive", value.bc_primitive, 0);

        // Initialize boundary conditions based on whether NSCBC is used
        Util::Message(INFO, "uses_nscbc=", uses_nscbc);
        if (uses_nscbc)
        {
            if (value.nghost == 2)
            {
#if AMREX_SPACEDIM == 3
                // The simple (nghost=2) NSCBC variant is 2D-only: single-tangential
                // transverse LODI + a 2D corner closure that were never ported to 3D/
                // Use NSCBC4 (nghost=4) for 3D, which is validated (faces+edges+corners).
                Util::Abort(INFO, "Simple NSCBC (nghost=2) is 2D-only; set nghost=4 to use NSCBC4 for 3D runs.");
#endif
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
            value.momentum_bc = new BC::Expression(AMREX_SPACEDIM, pp, "momentum.bc");

            // Primitive path: prescribe pressure at the boundary and reconstruct
            // per-phase energies from the EOS in FillGhost4BC. Left nullptr in
            // the conservative path (delete nullptr is safe).
            if (value.bc_primitive)
                value.pressure_bc = new BC::Expression(1, pp, "pressure.bc");

            Util::Message(INFO, "Parsing Reg");
            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
        }

        // Eta BC: parse from "eta.bc" if user provides it, otherwise zero neumann.
        // Eta is a volume fraction transported by advection + Cahn-Hilliard;
        if (pp.contains("eta.bc.type.xlo") || pp.contains("eta.bc.type.ylo"))
        {
            value.eta_bc = new BC::Expression(1, pp, "eta.bc");
        }
        else
        {
            value.eta_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
        }

        // Phi (solid indicator) BC.  The solid geometry is static, so the
        // indicator should not vary at the domain edge: zero-neumann
        // (ghost = interior) is the correct default.  An explicit
        // "solid.phi.bc" expression may override it.
        if (value.embedded.apply)
        {
            if (pp.contains("solid.phi.bc.type.xlo") || pp.contains("solid.phi.bc.type.ylo"))
                value.embedded.phi_bc = new BC::Expression(1, pp, "solid.phi.bc");
            else
                value.embedded.phi_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
        }
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // BC carried by the (derived) pressure fields for AMR coarse-fine
        // FillPatch. In the primitive path this is the user pressure BC; else
        // fall back to energy_bc (pressure fields are non-evolving, so this is
        // mostly cosmetic -- the physical-boundary fill happens in FillGhost4BC).
        BC::BC<Set::Scalar>* pbc =
            (value.bc_primitive && value.pressure_bc) ? value.pressure_bc : value.energy_bc;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,           value.eta_bc,      1, nghost,  "eta",          true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.eta_bc,      1, nghost,  "eta_old",      false,true);
        value.RegisterNewFab(value.rho_eta0_mf,      value.density_bc,  1, nghost,  "rho_eta0",     true, true);
        value.RegisterNewFab(value.rho_eta1_mf,      value.density_bc,  1, nghost,  "rho_eta1",     true, true);
        value.RegisterNewFab(value.rho_eta0_old_mf,  value.density_bc,  1, nghost,  "rho_eta0_old", false,true);
        value.RegisterNewFab(value.rho_eta1_old_mf,  value.density_bc,  1, nghost,  "rho_eta1_old", false,true);

        // Advected shell density c = Gamma|grad eta| (8th RK primary) + its
        // diagnostic areal density Gamma = c/|grad eta|.  Shares eta's BC: c is a
        // passive advected scalar with the same zero-gradient far-field behaviour.
        value.RegisterNewFab(value.shell_mf,         value.eta_bc,      1, nghost,  "shell",        true, true);
        value.RegisterNewFab(value.shell_old_mf,     value.eta_bc,      1, nghost,  "shell_old",    false,true);
        // Colour function c (Schmidmayer 2017 eq. 3): 9th RK primary, pure advection.
        value.RegisterNewFab(value.cfun_mf,          value.eta_bc,      1, nghost,  "cfun",         true, true);
        value.RegisterNewFab(value.cfun_old_mf,      value.eta_bc,      1, nghost,  "cfun_old",     false,true);
        value.RegisterNewFab(value.Gamma_mf,        &value.bc_nothing,  1, 0,       "Gamma",        true, false);

        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing,  1, 0,       "etadot",       true, false);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing,  4, 0,       "hess_eta",     false,false, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing,  AMREX_SPACEDIM, 0,       "n_hat",        false,false, { AMREX_D_DECL("x", "y", "z") });

        // EMBEDDED SOLID BOUNDARY
        if (value.embedded.apply)
        {
            value.RegisterNewFab(value.embedded.phi_mf,         value.embedded.phi_bc, 1, nghost,   "phi",      true, false);
            value.RegisterNewFab(value.embedded.phi_old_mf,     value.embedded.phi_bc, 1, nghost,   "phi_old",  false,false);
            value.RegisterNewFab(value.embedded.grad_phi_mf,   &value.bc_nothing,      2, 0,        "grad_phi", true, false, { "x", "y" });

            // Prescribed per-phase solid target state (quiescent solid).
            value.RegisterNewFab(value.embedded.density0_mf, value.density_bc,  1, nghost, "solid_rho_eta0",    false,false);
            value.RegisterNewFab(value.embedded.density1_mf, value.density_bc,  1, nghost, "solid_rho_eta1",    false,false);
            value.RegisterNewFab(value.embedded.momentum_mf, value.momentum_bc, AMREX_SPACEDIM, nghost, "solid_momentum",    false,false, { AMREX_D_DECL("x", "y", "z") });
            value.RegisterNewFab(value.embedded.energy0_mf,  value.energy_bc,   1, nghost, "solid_energy0",     false,false);
            value.RegisterNewFab(value.embedded.energy1_mf,  value.energy_bc,   1, nghost, "solid_energy1",     false,false);
        }

        // FLUID 0
        value.RegisterNewFab(value.density0_mf,     value.density_bc,   1, nghost,  "density0",     false,false);
        value.RegisterNewFab(value.density0_old_mf, value.density_bc,   1, nghost,  "density0_old", false,false);

        // E_0 / E_1 are 6-eq conserved primaries (per-phase internal energies).
        value.RegisterNewFab(value.energy0_mf,      value.energy_bc,    1, nghost, "energy0",       true, true);
        value.RegisterNewFab(value.energy0_old_mf,  value.energy_bc,    1, nghost, "energy0_old",   false,true);

        // PER-PHASE momenta are diagnostic only (read-only patches M0/M1) and are
        // built from the 2-component per-phase velocity ICs; kept 2-component in 3D.
        // They use bc_nothing (component-agnostic): momentum_bc now has SPACEDIM
        // components and would assert on these 2-component fabs in 3D.
        value.RegisterNewFab(value.momentum0_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum0",     false,false, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum0_old_mf,&value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum0_old", false,false);
 
        //value.RegisterNewFab(value.T0_mf,           value.temperature_bc, 1, nghost, "T0", false, false);
        //value.RegisterNewFab(value.k0_thermal_mf,   &value.bc_nothing, 1, nghost, "k0_thermal", false, false);
        //value.RegisterNewFab(value.h0_thermal_mf,   &value.bc_nothing, 1, nghost, "h0_thermal", false, false);

        value.RegisterNewFab(value.pressure0_mf,    pbc,                1, nghost, "pressure0",     false,false);
        value.RegisterNewFab(value.velocity0_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "velocity0",     false,false, { AMREX_D_DECL("x", "y", "z") });

        // FLUID 1
        value.RegisterNewFab(value.density1_mf,     value.density_bc,   1, nghost, "density1",      false,false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc,   1, nghost, "density1_old",  false,false);

        value.RegisterNewFab(value.energy1_mf,      value.energy_bc,    1, nghost, "energy1",       true, true);
        value.RegisterNewFab(value.energy1_old_mf,  value.energy_bc,    1, nghost, "energy1_old",   false,true);

        value.RegisterNewFab(value.momentum1_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum1",     false,false, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum1_old_mf,&value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum1_old", false,false);

        //value.RegisterNewFab(value.T1_mf,           value.temperature_bc, 1, nghost, "T1", false, false);
        //value.RegisterNewFab(value.k1_thermal_mf,   &value.bc_nothing, 1, nghost, "k1_thermal", false, false);
        //value.RegisterNewFab(value.h1_thermal_mf,   &value.bc_nothing, 1, nghost, "h1_thermal", false, false);

        value.RegisterNewFab(value.pressure1_mf,    pbc,                1, nghost, "pressure1",     false,true);
        value.RegisterNewFab(value.velocity1_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "velocity1",     false,true, { AMREX_D_DECL("x", "y", "z") });

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,    pbc,                 1,              nghost,  "pressure",    true, false);
        value.RegisterNewFab(value.velocity_mf,    &value.bc_nothing,   AMREX_SPACEDIM, nghost,  "velocity",    true, false, { AMREX_D_DECL("x", "y", "z") });
        // Vorticity is the curl of velocity: a scalar (omega_z) in 2D, a full
        // 3-vector (omega_x, omega_y, omega_z) in 3D.
#if AMREX_SPACEDIM == 2
        value.RegisterNewFab(value.vorticity_mf,           &value.bc_nothing,   1, 0,       "vorticity",        true, false);
#else
        value.RegisterNewFab(value.vorticity_mf,           &value.bc_nothing,   3, 0,       "vorticity",        true, false, { "x", "y", "z" });
#endif
        value.RegisterNewFab(value.density_mf,              value.density_bc,   1, nghost,  "density",          true, false);
        value.RegisterNewFab(value.density_old_mf,          value.density_bc,   1, nghost,  "density_old",      false,false);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost,  "energy_per_vol",   true, true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost,  "energy_per_mass",  true, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost,  "energy_vol_old",   false,true);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost,  "energy_mas_old",   false,true);
        value.RegisterNewFab(value.momentum_mf,             value.momentum_bc,  AMREX_SPACEDIM, nghost,  "momentum",         true, true, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum_old_mf,         value.momentum_bc,  AMREX_SPACEDIM, nghost,  "momentum_old",     false,true, { AMREX_D_DECL("x", "y", "z") });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,          &value.bc_nothing,   1, 0,       "m0",               false,false);
        value.RegisterNewFab(value.u0_mf,          &value.bc_nothing,   2, 0,       "u0",               false,false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,           &value.bc_nothing,   2, 0,       "q0",               false,false, { "x", "y" });
        value.RegisterNewFab(value.Source_mf,      &value.bc_nothing,   AMREX_SPACEDIM + 2, 0,  "Source",true, false, { "_rho", AMREX_D_DECL("_Mx", "_My", "_Mz"), "_E" });
        value.RegisterNewFab(value.Fsv_mf,         &value.bc_nothing,   AMREX_SPACEDIM, 0,      "Fsv",  true, false, { AMREX_D_DECL("x", "y", "z") }); // Surface Tension
        value.RegisterNewFab(value.Fw_mf,          &value.bc_nothing,   AMREX_SPACEDIM, 0,      "Fw",   true, false, { AMREX_D_DECL("x", "y", "z") }); // Weight
        value.RegisterNewFab(value.Ldot_mf,        &value.bc_nothing,   AMREX_SPACEDIM, 0, "Ldot",      true, false, { AMREX_D_DECL("x", "y", "z") }); // Ldot (3-comp: z-momentum source develops dynamically)
        value.RegisterNewFab(value.T_mf,            value.energy_bc,    1, nghost, "T",                 true, false);               // Temperature
        value.RegisterNewFab(value.cp_mf,          &value.bc_nothing,   1, nghost, "cp",                false,true);                // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,          &value.bc_nothing,   1, nghost, "cv",                false,true);                // Constant Volume Specific Heat
        //value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false, true);         // Thermal Conductivity
        //value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false, true);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc,    1, nghost, "gamma",             true, false);               // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf,           value.energy_bc,    1, nghost, "p0",                true, true);                // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc,    1, nghost, "mu_chem",           true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,           &value.bc_nothing,   1, nghost, "a",                 true, false);               // Speed of sound
        value.RegisterNewFab(value.Ma_mf,          &value.bc_nothing,   2, nghost, "Ma",                true, false, { "x", "y" }); // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   value.energy_bc,    1, nghost, "UE_per_vol",        true, false);               // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   value.energy_bc,    1, nghost, "UE_per_mass",       true, false);               // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   value.energy_bc,    1, nghost, "KE_per_vol",        true, false);               // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   value.energy_bc,    1, nghost, "KE_per_mass",       true, false);               // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,          &value.bc_nothing,   1, nghost, "Spadling_Number",   true, false);               // Spalding Number
        value.RegisterNewFab(value.Y_mf,           &value.bc_nothing,   1, nghost, "Mass_Fraction",     true, false);               // Mass Fraction

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,         &value.bc_nothing,  AMREX_SPACEDIM, 0, "grad_eta",           true, false, { AMREX_D_DECL("x", "y", "z") }); // grad(eta)
        value.RegisterNewFab(value.kappas_mf,           &value.bc_nothing,  3,              0, "kappa",              true, false, { "Avg", "1", "2" });             // Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf,&value.bc_nothing,  AMREX_SPACEDIM, 0, "grad_mag_grad_eta",  false,false, { AMREX_D_DECL("x", "y", "z") }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,         &value.bc_nothing,  1,              0, "rho_flux",           true, false);                                  // Density Flux
        value.RegisterNewFab(value.M_flux_mf,           &value.bc_nothing,  AMREX_SPACEDIM, 0, "M_flux",             true, false, { AMREX_D_DECL("x", "y", "z") }); // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,           &value.bc_nothing,  1,              0, "E_flux",             true, false);                                  // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,          &value.bc_nothing,  AMREX_SPACEDIM, 0, "div_tau",            true, false, { AMREX_D_DECL("x", "y", "z") }); // Energy Flux
        value.RegisterNewFab(value.hess_u_mf,           &value.bc_nothing,  8,              0, "hess_u",             false,false, {"000","001",
                                                                                                                                   "010","011",
                                                                                                                                   "100","101",
                                                                                                                                   "110","111"});                   // hess_u Flux
        value.RegisterNewFab(value.Vap_dot_mf, &value.bc_nothing, AMREX_SPACEDIM + 3, 0, "Vap_dot", true, false, { "_eta", "_rho", AMREX_D_DECL("_Mx", "_My", "_Mz"), "_E" });    // Momentum Flux
        
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

    // EMBEDDED SOLID BOUNDARY initial conditions.
    //   solid.phi.ic       -> indicator field (1 fluid, 0 solid), e.g. a tanh
    //                         circle/airfoil expression (see tests/FlowCylinder).
    //   solid.density{0,1} -> prescribed phase masses (alpha_k rho_k) in solid.
    //   solid.momentum     -> prescribed mixture momentum in solid (0 = static wall).
    //   solid.energy{0,1}  -> prescribed per-phase internal energies in solid.
    if (value.embedded.apply)
    {
#if (AMREX_SPACEDIM == 3)
        // 3D build: also offer "stl" geometry (IC::STL -> self-contained Util::STL, no EB).
        pp.select_default<IC::Constant,IC::Expression,IC::BMP,IC::PNG,IC::STL>("solid.phi.ic", value.embedded.phi_ic, value.geom);
#else
        pp.select_default<IC::Constant,IC::Expression,IC::BMP,IC::PNG>("solid.phi.ic",       value.embedded.phi_ic,      value.geom);
#endif
        // Single-phase solid input: total density + pressure (+ momentum).  The
        // per-phase target slots are derived from these by InitEmbeddedSolidTarget.
        pp.select_default<IC::Constant,IC::Expression>("solid.density.ic",                  value.embedded.density_ic,  value.geom);
        pp.select_default<IC::Constant,IC::Expression>("solid.pressure.ic",                 value.embedded.pressure_ic, value.geom);
        pp.select_default<IC::Constant,IC::Expression>("solid.momentum.ic",                 value.embedded.momentum_ic, value.geom);
    }


    // SOLVERS
    // Riemann solver
    std::string solver_name;
    pp.query("Riemann_Solver.type", solver_name);
    Util::Message(INFO, "Input file has Riemann_Solver.type = ", solver_name);
    pp.select_default<Solver::Local::FluidRiemann::Roe,
                      Solver::Local::FluidRiemann::HLLE,
                      Solver::Local::FluidRiemann::HLLC,
                      Solver::Local::FluidRiemann::HLLCE,
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman,   // Doesn't really work
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro
                      //Solver::Local::FluidRiemann::Upwind,            // Super bad - do not use
                      //Solver::Local::FluidRiemann::Lax_Friedrich      // Super bad - do not use
    >("Riemann_Solver", value.riemannsolver);
    Util::Message(INFO, "Selected Riemann solver: ", typeid(*value.riemannsolver).name());


    // LIMITER / primitive-variable reconstruction.
    // Selected by name: Limiter.type = godunov | minmod | vanleer | weno3 | weno5
    // Default = godunov (no reconstruction; first-order behavior preserved).
    {
        std::string limiter_name;
        pp.query("Limiter.type", limiter_name);
        Util::Message(INFO, "Input file has Limiter.type = ", limiter_name);
        pp.select_default<Solver::Local::Limiter::Godunov,  // 1st Order (i.e. no limiter)
                          Solver::Local::Limiter::Minmod,
                          Solver::Local::Limiter::VanLeer,
                          Solver::Local::Limiter::WENO3,    // 3rd Order
                          Solver::Local::Limiter::WENO5     // 5th Order
        >("Limiter", value.limiter);
        Util::Message(INFO, "Selected Limiter: ", typeid(*value.limiter).name());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");
    // Initialize step counter
    if (step_counter.size() <= (size_t)lev)
    {
        step_counter.resize(lev + 1, 0);
    }
    step_counter[lev] = 0;


    // Initialize individual fluid variables
    // DIFFUSIVE BOUNDRY
    eta_ic          ->Initialize(lev, eta_mf, 0.0);
    eta_ic          ->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]  ->setVal(0.0);
    hess_eta_mf[lev]->setVal(0.0);

    // ------------------------------------------------------------------
    // SHELL (Marmottant): the AREAL shell density Gamma is the transported
    // primary.  The shell starts unstrained everywhere, Gamma_0 = 1, so
    // sigma(t=0) = chi(Gamma_buck - 1) exactly and uniformly across the whole
    // band.  Setting it to 1 in the bulk too costs nothing -- Gamma is only ever
    // read where |grad eta| is significant -- and avoids a spurious Gamma
    // gradient at the band edge that advection would otherwise pull inward.
    // ------------------------------------------------------------------
    shell_mf[lev]    ->setVal(1.0);
    shell_old_mf[lev]->setVal(1.0);
    // c is initialised FROM eta and normalised by its jump.  eta already runs
    // 0 -> 1 across the band, so [c] = 1 and c = eta at t = 0.  They diverge
    // afterwards: eta is relaxed and clamped, c only advects.
    amrex::MultiFab::Copy(*cfun_mf[lev],     *eta_mf[lev], 0, 0, 1, nghost);
    amrex::MultiFab::Copy(*cfun_old_mf[lev], *eta_mf[lev], 0, 0, 1, nghost);
    Gamma_mf[lev]    ->setVal(0.0);

    // SOURCE TERMS
    Fw_mf[lev]      ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Vap_dot_mf[lev] ->setVal(0.0);

    // Pre-zero per-phase velocity/momentum so inputs that specify only the in-plane
    // components (region0/region1) leave the out-of-plane (z) component at 0.  The
    // expression IC writes exactly as many components as regionN entries provided,
    // so a 3-component field driven by a 2-region IC keeps vz = 0 here; supply
    // velocity{0,1}.ic.expression.region2 to drive a genuine 3D velocity field.
    velocity0_mf[lev]->setVal(0.0);
    velocity1_mf[lev]->setVal(0.0);
    momentum0_mf[lev]->setVal(0.0);
    momentum1_mf[lev]->setVal(0.0);

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

    // EMBEDDED SOLID BOUNDARY: indicator + prescribed solid target state.
    if (embedded.apply)
    {
        embedded.phi_ic     ->Initialize(lev, embedded.phi_mf, 0.0);
        embedded.phi_ic     ->Initialize(lev, embedded.phi_old_mf, 0.0);
        embedded.grad_phi_mf[lev]    ->setVal(0.0);
        // Derive the per-phase solid target (density0/1, energy0/1, momentum)
        // from the single solid density + pressure, split by local eta.
        // Must run BEFORE the velocity blend below (it reads density0/1, M).
        InitEmbeddedSolidTarget(lev, 0.0);

        // ----------------------------------------------------------------
        // Initialize the embedded solid AT REST (Mix-style velocity blend).
        // ----------------------------------------------------------------
        // The fluid IC sets the velocity over the WHOLE domain, including
        // inside the solid.  Blend it toward the solid velocity u_solid by phi
        //   velocity <- phi*velocity + (1-phi)*u_solid
        // so the solid region starts quiescent.  Without this, the Yang
        // full-flux porous wall (no cutoff freeze) starts the dense solid
        // interior moving at the freestream and shocks itself apart at startup
        // (the low-Mach viscous Riemann abort).  u_solid = solid_M/rho_solid
        // (= 0 for a static solid).  Mix() below then builds the conserved
        // state from the blended primitives.
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi   = embedded.phi_mf[lev]->array(mfi);
            auto vel0  = velocity0_mf[lev]->array(mfi);
            auto vel1  = velocity1_mf[lev]->array(mfi);
            auto s_re0 = embedded.density0_mf[lev]->array(mfi);
            auto s_re1 = embedded.density1_mf[lev]->array(mfi);
            auto s_M   = embedded.momentum_mf[lev]->array(mfi);
            const Set::Scalar small_loc = small;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar w     = std::min(std::max(phi(i, j, k), 0.0), 1.0);
                const Set::Scalar rho_s = std::max(s_re0(i, j, k) + s_re1(i, j, k), small_loc);
                const Set::Scalar us_x  = s_M(i, j, k, 0) / rho_s;
                const Set::Scalar us_y  = s_M(i, j, k, 1) / rho_s;
                vel0(i, j, k, 0) = w * vel0(i, j, k, 0) + (1.0 - w) * us_x;
                vel0(i, j, k, 1) = w * vel0(i, j, k, 1) + (1.0 - w) * us_y;
                vel1(i, j, k, 0) = w * vel1(i, j, k, 0) + (1.0 - w) * us_x;
                vel1(i, j, k, 1) = w * vel1(i, j, k, 1) + (1.0 - w) * us_y;
            });
        }

        // grad(phi) diagnostic (static; for the boundary-refinement plot).
        const Set::Scalar *DXp = geom[lev].CellSize();
        embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto grad_phi = embedded.grad_phi_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector g = Numeric::Gradient(phi, i, j, k, 0, DXp);
                grad_phi(i, j, k, 0) = g(0);
                grad_phi(i, j, k, 1) = g(1);
            });
        }
    }

    // FILLING GHOST CELLS
    rho_eta0_mf[lev]->setVal(0.0);
    rho_eta1_mf[lev]->setVal(0.0);
    rho_eta0_old_mf[lev]->setVal(0.0);
    rho_eta1_old_mf[lev]->setVal(0.0);

    // Pre-zero the mixture momentum/velocity (like the conserved fields above) so
    // every cell of the grown box is finite before the time=0 Mix() runs over it
    // (Mix() then overwrites valid+ghost from the per-phase fields).
    momentum_mf[lev]->setVal(0.0);
    momentum_old_mf[lev]->setVal(0.0);
    velocity_mf[lev]->setVal(0.0);

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

    // Reflux scratch: FluxRegister (lev>0) + per-direction cc_fluxes.
    // Initialize covers lev=0 (Regrid is never called on level 0); Regrid
    // covers lev>0 via MakeNewLevelFromCoarse / RemakeLevel.
    AllocateRefluxScratch(lev);

    Util::ParallelMessage(INFO, "Finished initialization, begginning time iteration");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    // Six-equation initial-condition construction.
    // ------------------------------------------------------------------------
    // Inputs (per-phase IC fields populated by Initialize()):
    //   eta_mf       = alpha_1            (Schmidmayer 2020 eq. 13 first row)
    //   density{0,1} = rho_k (pure)       (per-phase IC)
    //   velocity{0,1}= u_k    (per-phase IC; mechanical equilibrium assumed)
    //   pressure{0,1}= p_k    (per-phase IC; usually equal at IC)
    //
    // Outputs (canonical 6-eq conservative primaries):
    //   eta            = alpha_1                          (Sch20 eq. 13)
    //   rho_eta0       = (alpha_1 rho_1)                  (canonical phase mass)
    //   rho_eta1       = (alpha_2 rho_2)
    //   density (mix)  = alpha_1 rho_1 + alpha_2 rho_2    (Sch20 eq. 8)
    //   M              = mixture momentum  rho u
    //   energy0        = alpha_1 (p + gam0 pi0)/(gam0-1)  (Sau09 eq. III.5)
    //   energy1        = alpha_2 (p + gam1 pi1)/(gam1-1)
    //   energy_per_vol = E0 + E1 + KE     = redundant rho E   (Sch20 eq. 16)
    // ------------------------------------------------------------------------
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();
    (void)domain;

    // Function is for the diffusive mixing terms (6-eq canonical form).
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // Capillary-energy closure inputs (see Hydro2.H).
        const int cap_close = capillary_closure, marm_ic = marmottant;
        const Set::Scalar sig_ic = sigma, chi_ic = marmottant_chi;
        const Set::Scalar Gb_ic = marmottant_Gamma_buck;
        const Set::Scalar sbrk_ic = marmottant_sigma_break, sigw_ic = sigma;
        amrex::Array4<const Set::Scalar> const &shell_ic = shell_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> const &cfun_ic  = cfun_mf[lev]->const_array(mfi);

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0_old = rho_eta0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1_old = rho_eta1_old_mf.Patch(lev, mfi);

        // FLUID 0 (per-phase IC and primary canonical (alpha rho e)_0 written)
        Set::Patch<const Set::Scalar>   v0          = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p0          = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0        = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0_old    = density0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0          = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0_old      = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0_arr      = energy0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0_old_arr  = energy0_old_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   T0          = T0_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   k0_thermal  = k0_thermal_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   h0_thermal  = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1 (per-phase IC and primary canonical (alpha rho e)_1 written)
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_arr      = energy1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_old_arr  = energy1_old_mf.Patch(lev, mfi);
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

        // EOS constants (Tammann/SG) for the per-phase initialization.
        const Set::Scalar gam0 = eos0_local.Gamma();
        const Set::Scalar pi0_ = eos0_local.P0();
        const Set::Scalar gam1 = eos1_local.Gamma();
        const Set::Scalar pi1_ = eos1_local.P0();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

            // Eta laplacian (used below for the chemical-potential diagnostic).
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // --- 6-equation canonical state at IC (Schmidmayer 2020 Sec2.3) ---
            // Volume fractions:
            const Set::Scalar a1 = eta(i, j, k);              // alpha_1
            const Set::Scalar a2 = 1.0 - a1;                  // alpha_2

            // Phase masses (canonical):
            rho_eta0(i, j, k)     = a1 * rho0(i, j, k);       // (alpha_1 rho_1)
            rho_eta1(i, j, k)     = a2 * rho1(i, j, k);       // (alpha_2 rho_2)
            rho_eta0_old(i, j, k) = rho_eta0(i, j, k);
            rho_eta1_old(i, j, k) = rho_eta1(i, j, k);

            // Mixture density (Schmidmayer 2020 eq. 8):
            rho(i, j, k)     = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            rho_old(i, j, k) = rho(i, j, k);

            // Mixture momentum from per-phase mass-weighted velocities:
            //   M = (alpha_1 rho_1) u_0 + (alpha_2 rho_2) u_1
            // PER-PHASE velocity fields (v0/v1) are full AMREX_SPACEDIM vectors; the
            // IC may drive every component (region0/region1/region2).  Build all
            // momentum components from them -- in 3D the z-momentum is a genuine DOF.
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                M(i, j, k, d) = rho_eta0(i, j, k) * v0(i, j, k, d) + rho_eta1(i, j, k) * v1(i, j, k, d);

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                M_old(i, j, k, d) = M(i, j, k, d);

            // Mixture velocity (diagnostic):
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            // Mixture kinetic energy (consistent with rho E):
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM( v(i, j, k, 0) * v(i, j, k, 0),
                                                                + v(i, j, k, 1) * v(i, j, k, 1),
                                                                + v(i, j, k, 2) * v(i, j, k, 2)));
            KE_mas(i, j, k) = (rho(i, j, k) > small) ? KE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // ===========================================================
            // MECHANICAL-EQUILIBRIUM INITIAL CONDITION
            //   p_mix = alpha_1 p0_ic + alpha_2 p1_ic    (Schmidmayer eq. 8)           
            //   (alpha rho e)_k = alpha_k (p_mix + gamma_k pi_k) / (gamma_k - 1)
            //                                              (Sau09 eq. III.5)
            // equalize_ic_pressure:
            //   0 (legacy)  : p_mix = a1*p0 + a2*p1                 (linear average)
            //   1 (matches  : p_mix = (a1*p0/(g0-1) + a2*p1/(g1-1)) /
            //     RelaxAnd                (a1/(g0-1) + a2/(g1-1))
            //     Reinit)     gamma-weighted form -- this is exactly the
            //                 pressure that the energy-conserving relaxation
            //                 (Sch20 eq. 26 / Sau09 eq. III.5) would yield
            //                 from these per-phase IC pressures, so step-1
            //                 RelaxAndReinit produces no IC pressure kick.
            // ===========================================================
            const Set::Scalar p0_ij = p0(i, j, k);
            const Set::Scalar p1_ij = p1(i, j, k);
            Set::Scalar p_mix_IC;
            if (equalize_ic_pressure == 0)
            {
                p_mix_IC = a1 * p0_ij + a2 * p1_ij;
            }
            else
            {
                const Set::Scalar denom_a = a1 / (gam0 - 1.0) + a2 / (gam1 - 1.0);
                const Set::Scalar numer_a = a1 * p0_ij / (gam0 - 1.0)
                                          + a2 * p1_ij / (gam1 - 1.0);
                p_mix_IC = numer_a / std::max(denom_a, small);
            }

            E0_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(p_mix_IC, a1, gam0, pi0_, small);
            E1_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(p_mix_IC, a2, gam1, pi1_, small);
            E0_old_arr(i, j, k) = E0_arr(i, j, k);
            E1_old_arr(i, j, k) = E1_arr(i, j, k);

            // Mixture internal energy (sum of canonical per-phase energies):
            UE_vol(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);
            UE_mas(i, j, k) = (rho(i, j, k) > small) ? UE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // Redundant total energy rho E (Schmidmayer 2020 eq. 16), plus the
            // capillary energy when the closure is on (Schmidmayer 2017).
            E_vol(i, j, k)     = KE_vol(i, j, k) + UE_vol(i, j, k)
                               + (cap_close ? CapEnergyAt(cfun_ic, shell_ic, i, j, k, DX,
                                                          marm_ic, sig_ic, chi_ic, Gb_ic,
                                                          sbrk_ic, sigw_ic) : 0.0);
            E_vol_old(i, j, k) = E_vol(i, j, k);
            E_mas(i, j, k)     = KE_mas(i, j, k) + UE_mas(i, j, k);
            E_mas_old(i, j, k) = E_mas(i, j, k);

            // Diagnostic specific-heat-ratio and ref-pressure (for plotfile only).
            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);

            // Mixture pressure
            press(i, j, k) = p_mix_IC;

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y))
            Bm(i, j, k) = SpaldingBM(Y(i, j, k), Y_infinity, small);

            // Temperature (diagnostic; not used by hyperbolic step)
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Speed of sound -- FROZEN mixture sound speed.
            // Schmidmayer 2020 eq. 17 / Saurel 2009 eq. III.2:
            //   c^2 = Y_1 c_1^2 + Y_2 c_2^2
            // Use the mechanical-equilibrium IC pressure p_mix_IC (both
            // per-phase pressures are = p_mix_IC at this point).
            {
                const Set::Scalar Y0 = rho_eta0(i, j, k) / std::max(rho(i, j, k), small);
                const Set::Scalar Y1 = 1.0 - Y0;
                const Set::Scalar rho1pure = rho_eta0(i, j, k) / std::max(a1, small);
                const Set::Scalar rho2pure = rho_eta1(i, j, k) / std::max(a2, small);
                const Set::Scalar c0_ph = Solver::EOS::EOS::PhasicSoundSpeed(rho1pure, p_mix_IC, gam0, pi0_, small);
                const Set::Scalar c1_ph = Solver::EOS::EOS::PhasicSoundSpeed(rho2pure, p_mix_IC, gam1, pi1_, small);
                a(i, j, k) = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_ph, c1_ph);
            }

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / a(i, j, k);
            Ma(i, j, k, 1) = v(i, j, k, 1) / a(i, j, k);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(0, lev, i, j, k, "ERROR IN Mix(): Primative Field Calculation", {
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
    vz_max = 0.0;
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
///////////////////////////////////////////////// RHS /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RHS(int lev,
    Set::Scalar time,
    amrex::MultiFab &rho_eta0_rhs_mf,
    amrex::MultiFab &rho_eta1_rhs_mf,
    amrex::MultiFab &M_rhs_mf,
    amrex::MultiFab &E_rhs_mf,
    amrex::MultiFab &eta_rhs_mf,
    amrex::MultiFab &E0_rhs_mf,
    amrex::MultiFab &E1_rhs_mf,
    amrex::MultiFab &shell_rhs_mf,
    amrex::MultiFab &cfun_rhs_mf,
    const amrex::MultiFab &rho_eta0_mf_in,
    const amrex::MultiFab &rho_eta1_mf_in,
    const amrex::MultiFab &M_mf_in,
    const amrex::MultiFab &E_mf_in,
    const amrex::MultiFab &eta_mf_in,
    const amrex::MultiFab &E0_mf_in,
    const amrex::MultiFab &E1_mf_in,
    const amrex::MultiFab &shell_mf_in,
    const amrex::MultiFab &cfun_mf_in)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Converting Array to mf  (6-eq primaries -- Sch20 eq. 13 + 16)
    amrex::MultiFab::Copy(*rho_eta0_mf[lev],       rho_eta0_mf_in, 0, 0, 1,              0);
    amrex::MultiFab::Copy(*rho_eta1_mf[lev],       rho_eta1_mf_in, 0, 0, 1,              0);
    amrex::MultiFab::Copy(*momentum_mf[lev],       M_mf_in,        0, 0, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_mf_in,        0, 0, 1,              0);
    amrex::MultiFab::Copy(*eta_mf[lev],            eta_mf_in,      0, 0, 1,              0);
    amrex::MultiFab::Copy(*energy0_mf[lev],        E0_mf_in,       0, 0, 1,              0);
    amrex::MultiFab::Copy(*energy1_mf[lev],        E1_mf_in,       0, 0, 1,              0);
    amrex::MultiFab::Copy(*shell_mf[lev],          shell_mf_in,    0, 0, 1,              0);
    shell_mf[lev]->FillBoundary(geom[lev].periodicity());
    amrex::MultiFab::Copy(*cfun_mf[lev],           cfun_mf_in,     0, 0, 1,              0);
    cfun_mf[lev]->FillBoundary(geom[lev].periodicity());

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

    // Primitive Fields (with BCs)
    FillGhost4BC(lev, time);

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

            // Curvature
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector
            if (false)//(grad_eta_mag < 1e-4)
            {
                n_hat(0) = 0.0;
                n_hat(1) = 0.0;
            }
            else
            {
                n_hat(0) = n_hat(0);
                n_hat(1) = n_hat(1);

                // (grad|grad eta|)_a = (1/|grad eta|) sum_b H_ab (grad eta)_b = (H grad_eta)/|grad eta|.
                // Matrix-vector form is dimension-generic and reduces to the original
                // explicit 2-component expression (hess_eta is symmetric).
                Set::Vector grad_mag_grad_eta = (1.0 / (grad_eta_mag + small)) * (hess_eta * grad_eta);

                Set::Scalar kappa, kappa1, kappa2 = 0.0;
                if (kappa_method == 1)
                {
                    kappa = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / (grad_eta_mag * grad_eta_mag + small)));

                    kappas(i, j, k, 0) = kappa;  // Mean or selected curvature
                    kappas(i, j, k, 1) = kappa1; // First principal curvature
                    kappas(i, j, k, 2) = kappa2; // Second principal curvature
                }
                else if (kappa_method == 2)
                {
#if AMREX_SPACEDIM == 2
                    // 2D Marmottant path (unchanged -- bit-identical to the 2D solver).
                    // Orthogonal Basis
                    Set::Vector t1, t2;

                    if (abs(n_hat(0)) > abs(n_hat(1)))
                    {
                        t1 = Set::Vector(-n_hat(1), n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                    }
                    else
                    {
                        t1 = Set::Vector(n_hat(1), -n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                    }

                    kappa1 = n_hat.dot(hess_eta * n_hat); // Normal Curvature
                    kappa2 = t1.dot(hess_eta * t1);       // Tangential Curvature

                    kappa1 = -kappa1;
                    kappa2 = -kappa2 * 2.0 * epsilon;

                    // Regularization
                    Set::Scalar K23 = kappa2 * kappa2;     // K23 Regularization
                    Set::Scalar K_Gauss = kappa1 * kappa2; // Gauss Regularization
                    // Mean
                    Set::Scalar K_mean = (kappa1 + kappa2) / 2.0; // Mean Curvature
                    // Assign the curvature you want to use
                    kappa = kappa2; // Or use another curvature measure as needed
#else
                    // 3D: build a genuine orthonormal tangent basis {t1, t2} in the
                    // plane perpendicular to n_hat (t2 = n_hat x t1).  A surface in 3D
                    // has TWO tangential principal directions, so kappa1/kappa2 are the
                    // normal-section curvatures along t1/t2 (vs the 2D single tangent).
                    // Seed t1 from whichever axis is least aligned with n_hat, then
                    // Gram-Schmidt + cross product.  Same -2*epsilon scaling as 2D.
                    Set::Vector seed = Set::Vector::Unit(0);
                    if (std::abs(n_hat(0)) >= std::abs(n_hat(1)) && std::abs(n_hat(0)) >= std::abs(n_hat(2)))
                        seed = Set::Vector::Unit(1);
                    Set::Vector t1 = seed - seed.dot(n_hat) * n_hat;
                    t1 /= (t1.norm() + small);
                    Set::Vector t2 = n_hat.cross(t1);   // already unit (n_hat, t1 orthonormal)

                    kappa1 = -t1.dot(hess_eta * t1) * 2.0 * epsilon; // 1st principal curvature
                    kappa2 = -t2.dot(hess_eta * t2) * 2.0 * epsilon; // 2nd principal curvature

                    // For 3D CSF the mean curvature drives surface tension.  (The exact
                    // selection that reproduces the Marmottant 2D piecewise-sigma model
                    // in 3D is TODO when 3D surface-tension bubbles are run; mean
                    // curvature is the standard, well-posed choice meanwhile.)
                    kappa = 0.5 * (kappa1 + kappa2);
#endif
                    // Store curvature values
                    kappas(i, j, k, 0) = kappa;  // Mean or selected curvature
                    kappas(i, j, k, 1) = kappa1; // First principal curvature
                    kappas(i, j, k, 2) = kappa2; // Second principal curvature
                }
            }

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN RHS(): Primative Field Calculation", {
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

    // Fill Boundries
    FillBoundariesWithBC(lev, time, energy_bc, { mu_chem_mf[lev].get() });

    // ------------------------------------------------------------
    // MARMOTTANT (2005) coated-bubble surface tension -- ADVECTED SHELL
    // ------------------------------------------------------------
    // sigma is a function of the shell's AREAL STRAIN, not of the local curvature.
    // The lipid monolayer is a MATERIAL carried by the interface, so we transport a
    // conserved shell density  c = Gamma |grad eta|  as an RK primary
    // (d(c)/dt + div(c u) = 0, discretized with the interface velocity below), and
    //
    //     Gamma = c / |grad eta|                        shell per unit AREA
    //     sigma = chi (Gamma_buck/Gamma - 1)            elastic
    //           = 0                                     buckled  (Gamma >= Gamma_buck)
    //           = sigma_water                           ruptured (elastic >= sigma_break)
    //
    // Gamma_buck/Gamma IS Marmottant's area ratio A/A_buck, so this is the same
    // sigma(A) law -- with A carried by the shell instead of re-derived from a
    // second derivative of eta every step.  What that buys, vs. the old sigma(R(x)):
    //
    //  * No curvature enters the tension, so there is nothing for the elastic law
    //    to amplify.  sigma(R) had d(sigma)/sigma = 2(R/Rb)^2/((R/Rb)^2-1) dR/R
    //    = 10.5x at R0/Rb = 1.11, which turned a 1% kappa error into 10% tension.
    //  * Gamma is uniform ACROSS the band by construction: c and |grad eta| share
    //    the same profile, so c/|grad eta| is flat through the band thickness.  The
    //    old per-cell R = (d-1)/kappa gave every cell in the band its OWN level-set
    //    radius, so with the band spanning R0 +- 3eps and R_buck inside that range,
    //    ~24% of band cells were permanently mis-classified "buckled" (sigma=0) even
    //    on a perfect circle -- and that fraction did NOT converge under refinement.
    //  * Shell material is conserved, so stretching genuinely dilutes Gamma and the
    //    rupture branch is reached for the right physical reason.
    //  * Tangential (Marangoni) redistribution of shell material is represented;
    //    eta alone cannot express it, since sliding material along a level set
    //    leaves eta identically unchanged.
    //  * Nonspherical shells and multiple bubbles need no labelling or geometry.
    //
    // sigma_eff carries one ghost cell so the capillary tensor Omega below can be
    // built in the ghost ring too (kappas_mf has zero ghosts).  It is evaluated on
    // the GROWN box directly from eta/shell (both of which have filled ghosts), so
    // the ghost ring is genuinely computed rather than left zero at physical faces.
    amrex::MultiFab sig_eff_mf;

    if (marmottant)
    {
        const amrex::BoxArray &ba = eta_mf[lev]->boxArray();
        const amrex::DistributionMapping &dm = eta_mf[lev]->DistributionMap();
        sig_eff_mf.define(ba, dm, 1, 1);
        sig_eff_mf.setVal(0.0);

        const Set::Scalar chi_ = marmottant_chi, Gb = marmottant_Gamma_buck;
        const Set::Scalar sbrk = marmottant_sigma_break, sigw = sigma;
        for (amrex::MFIter mfi(sig_eff_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &vbx = mfi.validbox();
            const amrex::Box gbx  = mfi.growntilebox(1);
            // The interface gate must use the SAME field the capillary tensor
            // is built from, or the shell tension and Omega end up gated on
            // two different interfaces once c and eta separate.
            amrex::Array4<const Set::Scalar> const &et = capillary_closure
                                                       ? cfun_mf[lev]->const_array(mfi)
                                                       : eta_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const &cs = shell_mf[lev]->const_array(mfi);
            amrex::Array4<Set::Scalar> const &sg  = sig_eff_mf.array(mfi);
            amrex::Array4<Set::Scalar> const &Gm  = Gamma_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &kap = kappas_mf[lev]->array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                Set::Scalar gem = ge.lpNorm<2>();
                Set::Scalar se = 0.0, G = 0.0;
                // Gate on |grad eta|: off the interface there is no shell and no
                // area to normalise by.  Omega is identically zero there anyway.
                if (gem > 1.0e-10)
                {
                    // Gamma is a PRIMARY -- read straight off the advected field.
                    // No division by |grad eta|, which is what made the previous
                    // c/|grad eta| reconstruction blow up in the band tails.
                    G = cs(i, j, k);
                    if (G > 0.0 && G < Gb)             // stretched -> elastic branch
                    {
                        Set::Scalar el = chi_ * (Gb / G - 1.0);
                        se = (el >= sbrk) ? sigw : el;
                    }
                    // G >= Gb: compressed past buckling -> sigma = 0 (se stays 0)
                }
                sg(i, j, k) = se;
                if (vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                {
                    Gm(i, j, k)     = G;
                    kap(i, j, k, 2) = se;   // sigma_eff diagnostic (plottable)
                }
            });
        }
        // Interior/periodic ghosts from neighbours; physical-face ghosts keep the
        // values computed on the grown box above.
        sig_eff_mf.FillBoundary(geom[lev].periodicity());
    }

    // ------------------------------------------------------------
    // CAPILLARY STRESS TENSOR  (Schmidmayer et al. 2017, Eq.3-4) -- the conservative
    // "continuum surface stress" form of surface tension.  The momentum force is
    //   F = div(Omega),   Omega_ij = sigma_eff (|grad eta| delta_ij - d_i eta d_j eta/|grad eta|).
    // div(Omega) reduces to sigma*kappa*grad(eta) for CONSTANT sigma, but for a VARYING
    // sigma it also carries the Marangoni (grad sigma) term automatically -- i.e. the
    // physically complete surface stress.  sigma_eff is the local Marmottant tension
    // kappas(.,2) (marmottant) or the constant sigma.  [The capillary energy
    // eps_sigma = sigma|grad eta| in the energy flux is the next step toward full
    // discrete conservation; it does not change the static momentum/Laplace balance.]
    // Symmetric-tensor storage:  2D [xx,yy,xy] ; 3D [xx,yy,zz,xy,xz,yz].
    amrex::MultiFab Omega;
    if (apply_surface_tension)
    {
        const amrex::BoxArray &baO = eta_mf[lev]->boxArray();
        const amrex::DistributionMapping &dmO = eta_mf[lev]->DistributionMap();
        const int nO = (AMREX_SPACEDIM == 2) ? 3 : 6;
        Omega.define(baO, dmO, nO, 1);
        Omega.setVal(0.0);
        const Set::Scalar sig0 = sigma; const int marm = marmottant;
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.growntilebox(1);
            amrex::Array4<const Set::Scalar> const &et = eta_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const &sg = marm ? sig_eff_mf.const_array(mfi)
                                                              : amrex::Array4<const Set::Scalar>{};
            amrex::Array4<Set::Scalar> const &om = Omega.array(mfi);
            // Schmidmayer 2017 builds Omega from the COLOUR FUNCTION gradient.
            // eta is rewritten by relaxation/clamping without moving material,
            // so grad(eta) makes the capillary force respond to thermodynamic
            // adjustment; grad(c) does not.  Gated so cap_c=0 keeps the old
            // behaviour exactly.
            amrex::Array4<const Set::Scalar> const &cf = cfun_mf[lev]->const_array(mfi);
            const int use_c = capillary_closure;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = use_c ? Numeric::Gradient(cf, i, j, k, 0, DX)
                                       : Numeric::Gradient(et, i, j, k, 0, DX);
                Set::Scalar gem = ge.lpNorm<2>();
                if (gem < 1.0e-10) return; // Omega = 0 off the interface
                Set::Scalar se = marm ? sg(i, j, k) : sig0;
                om(i, j, k, 0) = se * (gem - ge(0) * ge(0) / gem); // xx
                om(i, j, k, 1) = se * (gem - ge(1) * ge(1) / gem); // yy
#if AMREX_SPACEDIM == 2
                om(i, j, k, 2) = se * (-ge(0) * ge(1) / gem);      // xy
#else
                om(i, j, k, 2) = se * (gem - ge(2) * ge(2) / gem); // zz
                om(i, j, k, 3) = se * (-ge(0) * ge(1) / gem);      // xy
                om(i, j, k, 4) = se * (-ge(0) * ge(2) / gem);      // xz
                om(i, j, k, 5) = se * (-ge(1) * ge(2) / gem);      // yz
#endif
            });
        }
        Omega.FillBoundary(geom[lev].periodicity());
    }

    // Main time integration loop
    // [DIAG RHSDECOMP] per-cell boundary-flux + source stash
    amrex::MultiFab rhsp_fxlo_mf(rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
    amrex::MultiFab rhsp_fxhi_mf(rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
    amrex::MultiFab rhsp_fylo_mf(rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
    amrex::MultiFab rhsp_fyhi_mf(rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
    amrex::MultiFab rhsp_src_mf (rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
    rhsp_fxlo_mf.setVal(0.0); rhsp_fxhi_mf.setVal(0.0); rhsp_fylo_mf.setVal(0.0);
    rhsp_fyhi_mf.setVal(0.0); rhsp_src_mf.setVal(0.0);
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
        auto rhsp_fxlo = rhsp_fxlo_mf.array(mfi);
        auto rhsp_fxhi = rhsp_fxhi_mf.array(mfi);
        auto rhsp_fylo = rhsp_fylo_mf.array(mfi);
        auto rhsp_fyhi = rhsp_fyhi_mf.array(mfi);
        auto rhsp_src  = rhsp_src_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs       = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs       = E_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> eta_rhs     = eta_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> shell_rhs   = shell_rhs_mf.array(mfi);
        Set::Patch<const Set::Scalar> shell = shell_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cfun_rhs    = cfun_rhs_mf.array(mfi);
        Set::Patch<const Set::Scalar> cfun  = cfun_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E0_rhs      = E0_rhs_mf.array(mfi);    // per-phase internal energy
        Set::Patch<Set::Scalar> E1_rhs      = E1_rhs_mf.array(mfi);

        // PER-PHASE INTERNAL ENERGIES (canonical primaries)
        Set::Patch<const Set::Scalar> E0_arr = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1_arr = energy1_mf.Patch(lev, mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);

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
        amrex::Array4<const Set::Scalar> om_cap; // capillary stress tensor (if surface tension on)
        if (apply_surface_tension) om_cap = Omega.const_array(mfi);
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = hess_u_mf.Patch(lev, mfi);

        // ---------------- Cell-centered hi-face flux storage for reflux ---
        // cc_fluxes use density_mf's BA/dmap (same as this MFIter source),
        // so mfi access is compatible.  Each cell stores the Riemann flux
        // at its HI face (i+1/2 for d=0, j+1/2 for d=1); at box-lo
        // boundaries we also write into ghost cell (i-1) / (j-1) so
        // coarse-fine boundary fluxes survive FillBoundary (Matt Q
        // commit 8a1633977 "Fix missing lo-face fluxes...").
        bool have_cc_fluxes = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y;
        amrex::Array4<Set::Scalar> ff_mom_x,  ff_mom_y;
        amrex::Array4<Set::Scalar> ff_ene_x,  ff_ene_y;
        amrex::Array4<Set::Scalar> ff_ene_k_x, ff_ene_k_y;
#if AMREX_SPACEDIM == 3
        amrex::Array4<Set::Scalar> ff_mass_z, ff_mom_z, ff_ene_z, ff_ene_k_z;
#endif
        if (have_cc_fluxes)
        {
            ff_mass_x  = cc_fluxes[lev].mass  [0]->array(mfi);
            ff_mass_y  = cc_fluxes[lev].mass  [1]->array(mfi);
            ff_mom_x   = cc_fluxes[lev].mom   [0]->array(mfi);
            ff_mom_y   = cc_fluxes[lev].mom   [1]->array(mfi);
            ff_ene_x   = cc_fluxes[lev].energy[0]->array(mfi);
            ff_ene_y   = cc_fluxes[lev].energy[1]->array(mfi);
            ff_ene_k_x = cc_fluxes[lev].ene_k [0]->array(mfi);
            ff_ene_k_y = cc_fluxes[lev].ene_k [1]->array(mfi);
#if AMREX_SPACEDIM == 3
            ff_mass_z  = cc_fluxes[lev].mass  [2]->array(mfi);
            ff_mom_z   = cc_fluxes[lev].mom   [2]->array(mfi);
            ff_ene_z   = cc_fluxes[lev].energy[2]->array(mfi);
            ff_ene_k_z = cc_fluxes[lev].ene_k [2]->array(mfi);
#endif
        }
        const auto bx_lo = amrex::lbound(bx);

        // EMBEDDED SOLID BOUNDARY patches (empty Array4 when the feature is
        // off; never dereferenced unless apply_embedded_solid is set).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);          // solid indicator (1 fluid, 0 solid)
        Set::Patch<const Set::Scalar> s_re0  = embedded.density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_re1  = embedded.density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_M    = embedded.momentum_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_E0   = embedded.energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_E1   = embedded.energy1_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);

            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            // Extract velocity from momentum and density.  The mixture velocity u
            // is fully dimensioned (momentum spine carries z); the prescribed-flow
            // u0 keeps a 0.0 z-placeholder (its IC field stays 2-component).
            Set::Vector u = Set::Vector(AMREX_D_DECL(v(i, j, k, 0), v(i, j, k, 1), v(i, j, k, 2)));
            Set::Vector u0 = Set::Vector(AMREX_D_DECL(_u0(i, j, k, 0), _u0(i, j, k, 1), 0.0));

            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));

            Set::Vector q0_ = Set::Vector(AMREX_D_DECL(q0(i, j, k, 0), q0(i, j, k, 1), 0.0));  // q0 IC field stays 2-component

            /// Calculate Source Terms
            // Shear:
            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0_.dot(grad_eta);

            Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
            Set::Matrix3 hess_u = Set::Matrix3::Zero();

            for (int p = 0; p < AMREX_SPACEDIM; p++)
                for (int q = 0; q < AMREX_SPACEDIM; q++)
                    for (int r = 0; r < AMREX_SPACEDIM; r++)
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

            // DIAGNOSTIC interface-localised artificial viscosity (see Hydro2.H).
            // Body-force pair: momentum gets F_art, energy gets u.F_art below.

            // EMBEDDED SOLID
            Set::Scalar phi_c = 1.0;
            if (embedded.apply)
            {
                phi_c = embedded.clampPhi(phisol(i, j, k));
            }

            // Solving
            for (int p = 0; p < AMREX_SPACEDIM; p++)             // i
                for (int q = 0; q < AMREX_SPACEDIM; q++)         // j
                    for (int r = 0; r < AMREX_SPACEDIM; r++)     // k
                        for (int s = 0; s < AMREX_SPACEDIM; s++) // l
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
                            Ldot(p) += 0.0;

                            // Grad visc terms
                            div_tau(p) += dMpqrs * gradu(r, s);
                        }

            // Debugging feild for div_tau and Ldot
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                div_tau_(i, j, k, d) = div_tau(d);

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Ldot_(i, j, k, d) = Ldot(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Viscosity solving", {
                { "Ldot[0]", Ldot(0) },
                { "Ldot[1]", Ldot(1) },
                { "div_tau[0]", div_tau(0) },
                { "div_tau[1]", div_tau(1) }
            }); // end check4nans
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Ldot[2]", Ldot(2) },
                { "div_tau[2]", div_tau(2) },
            }); // end check4nans
#endif  
            
            // ------------------------------------------------------------
            // Surface Tension
            // ------------------------------------------------------------
            // Fsv =  simga * kappa * grad_eta
            Set::Vector Fsv_vector = Set::Vector::Zero();
            // Under the split scheme the capillary terms belong to L_cap
            // (Schmidmayer eq. 17); applying them here too would double-count.
            if (apply_surface_tension && !capillary_closure)
            {
                // Conservative continuum-surface-stress force  F = div(Omega)
                // (Schmidmayer 2017).  Centered divergence of the capillary stress
                // tensor stored in om_cap.
                const Set::Scalar idx = 0.5 / DX[0], idy = 0.5 / DX[1];
#if AMREX_SPACEDIM == 2
                Fsv_vector(0) = (om_cap(i+1,j,k,0) - om_cap(i-1,j,k,0)) * idx
                              + (om_cap(i,j+1,k,2) - om_cap(i,j-1,k,2)) * idy;
                Fsv_vector(1) = (om_cap(i+1,j,k,2) - om_cap(i-1,j,k,2)) * idx
                              + (om_cap(i,j+1,k,1) - om_cap(i,j-1,k,1)) * idy;
#else
                const Set::Scalar idz = 0.5 / DX[2];
                Fsv_vector(0) = (om_cap(i+1,j,k,0) - om_cap(i-1,j,k,0)) * idx
                              + (om_cap(i,j+1,k,3) - om_cap(i,j-1,k,3)) * idy
                              + (om_cap(i,j,k+1,4) - om_cap(i,j,k-1,4)) * idz;
                Fsv_vector(1) = (om_cap(i+1,j,k,3) - om_cap(i-1,j,k,3)) * idx
                              + (om_cap(i,j+1,k,1) - om_cap(i,j-1,k,1)) * idy
                              + (om_cap(i,j,k+1,5) - om_cap(i,j,k-1,5)) * idz;
                Fsv_vector(2) = (om_cap(i+1,j,k,4) - om_cap(i-1,j,k,4)) * idx
                              + (om_cap(i,j+1,k,5) - om_cap(i,j-1,k,5)) * idy
                              + (om_cap(i,j,k+1,2) - om_cap(i,j,k-1,2)) * idz;
#endif
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Fsv(i, j, k, d) = Fsv_vector(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fsv_vector[0]", Fsv_vector(0) },
                { "Fsv_vector[1]", Fsv_vector(1) }
            });
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fsv_vector[2]", Fsv_vector(2) }
            }); // end check4nans
#endif  

            // ------------------------------------------------------------
            // Weight
            // ------------------------------------------------------------
            // Fw = - rho * g
            Set::Vector Fw_vector = Set::Vector::Zero();
            if (apply_weight)
            {
                // Gravity acts along the y-axis (component 1), as in the 2D model.
                Fw_vector(1) = -rho(i, j, k) * g;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Fw(i, j, k, d) = Fw_vector(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Weight solving", {
                { "Fw_vector[0]", Fw_vector(0) },
                { "Fw_vector[1]", Fw_vector(1) }
            });
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fw_vector[2]", Fw_vector(2) }
            }); // end check4nans
#endif

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
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Cahn-Hillard solving", {
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
                // Spalding mass transfer number (from the vapor-mass-fraction
                // surrogate Y = (alpha rho)_0 / rho computed above).
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // PURE-phase densities rho_k = (alpha rho)_k / alpha_k (the
                // density0/1_mf primitives from FillGhost4BC).  The old code
                // used the PARTIAL densities (alpha rho)_k here, which (i)
                // scaled the evaporation rate by a spurious factor alpha_g and
                // (ii) made the volume-transfer term blow up where a phase
                // vanishes (1/max(alpha*rho, small) -> 1e8).
                Set::Scalar rho_g = std::max(rho0(i, j, k), small);   // gas    (fluid 0; eta = alpha_gas)
                Set::Scalar rho_l = std::max(rho1(i, j, k), small);   // liquid (fluid 1)

                // Mass-transfer rate (volumetric) -- simplified Spalding form,
                // localized to the band by |grad(eta)|.  Dv is a calibrated
                // input that carries the film-length division of the classical
                // d^2-law rate (rho_g D_v / delta_film) ln(1+B_M).
                m_dot_Vap = rho_g * Dv * (B_M / (1.0 + B_M + small)) * grad_eta_mag;

                // Volume-fraction source: evaporation (m_dot > 0) GROWS the
                // gas volume fraction -- gas gains volume at 1/rho_g per unit
                // transferred mass while liquid loses it at 1/rho_l:
                //     d(alpha_g)/dt += m_dot (1/rho_g - 1/rho_l) > 0.
                // (The old (1/rho_l - 1/rho_g) partial-density form had the
                // sign INVERTED: evaporation shrank the gas volume fraction.)
                eta_dot_Vap = m_dot_Vap * (1.0 / rho_g - 1.0 / rho_l);

                // Evaporative cooling: the stiffened-gas EOSs carry no heat-of-
                // vaporization offset, so the latent heat enters as an explicit
                // sink -m_dot * Lv [W/m^3] on the mixture energy (applied in
                // the Source energy row below; Lv = 0 disables).  Replaces the
                // old m_dot |u|^2 |grad eta|^2 term, which was dimensionally
                // inconsistent and never actually applied.
                E_dot_Vap = -m_dot_Vap * Lv;
            }
            // Vaporization Trackers.  Layout: [_eta, _rho, M..., _E].
            Vap_dot(i, j, k, 0) = eta_dot_Vap;
            Vap_dot(i, j, k, 1) = m_dot_Vap;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Vap_dot(i, j, k, 2 + d) = 0.0;
            Vap_dot(i, j, k, AMREX_SPACEDIM + 2) = E_dot_Vap;


            // Total:
            Set::Vector Total_Force = Fsv_vector + Fw_vector;

            // ------------------------------------------------------------
            // Energy work terms in conservative (divergence) form.
            //
            // Viscous:   div(tau.u) = u.div(tau) + tau:grad(u).  The second
            // term is the dissipation Phi >= 0 (arrested KE -> heat); with
            // only u.div(tau) the total energy is not conserved.
            //   Phi = 2 mu_eff S:S + (lambda_eff - 2/3 mu_eff) (div u)^2,
            //   S = sym(grad u).
            //
            // Capillary: the FLUID energy (rho E = internal + kinetic, which
            // does NOT contain the surface energy sigma|grad eta|) receives
            // exactly the capillary-force work u.div(Omega) -- already inside
            // u.dot(Total_Force) below.  The identity
            //   u.div(Omega) = div(Omega.u) - Omega:grad(u)
            // says the fluid absorbs the transported capillary work MINUS the
            // Omega:grad(u) exchange, which is the rate the SURFACE reservoir
            // gains:  d(sigma|grad eta|)/dt + div(sigma|grad eta| u)
            //       = Omega:grad(u)   (exact for advected eta).
            // A previous revision added cap_work = Omega:grad(u) ON TOP of
            // u.dot(Total_Force), handing the fluid div(Omega.u) total -- i.e.
            // the surface-exchange leg was double-counted into rho E, CREATING
            // energy whenever the interface stretched and destroying it on
            // shrink (measured on the oscillating-bubble test as a per-cycle
            // amplitude pump).  The reservoir itself needs no PDE here: it is
            // the state function sigma_eff |grad eta| (auditable offline from
            // the plotted kappa[2] and grad_eta fields).  Folding it INTO the
            // conserved state (full Schmidmayer 2017 E_t) remains the deeper
            // upgrade -- it touches the Riemann state, EOS inversions, and
            // relaxation, and is deferred deliberately.
            // ------------------------------------------------------------
            Set::Matrix Ssym = 0.5 * (gradu + gradu.transpose());
            Set::Scalar divu_cell = gradu.trace();
            Set::Scalar visc_diss = 2.0 * mu_eff * Ssym.squaredNorm()
                                  + (lambda_eff - (2.0 / 3.0) * mu_eff) * divu_cell * divu_cell;

            // SOURCES.  Layout: [0]=mdot, [1..SD]=momentum, [SD+1]=energy.
            Source(i, j, k, 0) = mdot0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Source(i, j, k, 1 + d) = Pdot0(d) + Ldot(d) + div_tau(d) + Total_Force(d);
            // Capillary work for the LEGACY body-force path (capillary_closure=0).
            // With the split closure on, Fsv_vector is zero here (the capillary
            // terms belong to L_cap), so this contributes nothing.
            const Set::Scalar cap_w = u.dot(Fsv_vector);
            Source(i, j, k, AMREX_SPACEDIM + 1) = qdot0 + u.dot(div_tau) + visc_diss + u.dot(Ldot)
                                                + u.dot(Fw_vector) + cap_w + E_dot_Vap;

            // Lagrange terms to enforce no-penetration
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Source(i, j, k, 1 + d) -= lagrange * u.dot(grad_eta) * grad_eta(d);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Source solving", { 
                { "Total_Force[0]",  Total_Force(0) },
                { "Total_Force[1]",  Total_Force(1) },
                { "Source[0]",  Source(i, j, k, 0) },
                { "Source[1]",  Source(i, j, k, 1) },
                { "Source[2]",  Source(i, j, k, 2) },
                { "Source[3]",  Source(i, j, k, 3) }
            }); // end check4nans

            // ============================================================
            // 6-equation HLLC face fluxes (Saurel 2009 Sec 3.1.2 / Schmidmayer 2020 Sec 3).
            // ============================================================
            const int X_dir = 0, Y_dir = 1, Z_dir = 2;

            // Build per-face State (6-eq).  EOS constants are per-phase and identical L/R per cell (same eos0/eos1).
            auto make_state = [&](int ii, int jj, int kk, int dir)
                -> Solver::Local::FluidRiemann::State
            {
                Solver::Local::FluidRiemann::State s;
                s.alpha       = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                s.alpha_rho_0 = rho_eta0(ii, jj, kk);
                s.alpha_rho_1 = rho_eta1(ii, jj, kk);
                // `dir` is the face-normal direction (0=x, 1=y, 2=z).  The tangents
                // are the cyclically-next momentum components -- the SAME convention
                // used when the Riemann momentum fluxes are scattered back into
                // M_flux below, so the round-trip is self-consistent in any dim.
                // In 2D this reduces exactly to the old normal/tangent swap.
                s.M_normal   = M(ii, jj, kk, dir);
                s.M_tangent  = M(ii, jj, kk, (dir + 1) % AMREX_SPACEDIM);
#if AMREX_SPACEDIM == 3
                s.M_tangent2 = M(ii, jj, kk, (dir + 2) % AMREX_SPACEDIM);
#endif
                s.E0      = E0_arr(ii, jj, kk);
                s.E1      = E1_arr(ii, jj, kk);
                s.E_total = E(ii, jj, kk);
                s.gamma0  = eos0.Gamma();
                s.pi0     = eos0.P0();
                s.gamma1  = eos1.Gamma();
                s.pi1     = eos1.P0();
                return s;
            };

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Conservative Variable Check", {
                { "eta", eta(i, j, k) },
                { "rho_eta0", rho_eta0(i, j, k) },
                { "rho_eta1", rho_eta1(i, j, k) },
                { "rho", rho(i, j, k) },
                { "M[0]", M(i, j, k, 0) },
                { "M[1]", M(i, j, k, 1) },
                { "E", E(i, j, k) },
                { "E0", E0_arr(i, j, k) },
                { "E1", E1_arr(i, j, k) },
                { "press", press(i, j, k) },
            }); // end check4nans

            // ------------------------------------------------------------
            // Reconstruct face states using the selected limiter (Sch20
            // Sec 3.2: PRIMITIVE-variable reconstruction; never conservatives).
            // For face between cell `lo` and cell `hi` in normal direction
            // `dir`, gather 6-cell window {lo-2, lo-1, lo, hi, hi+1, hi+2},
            // convert to primitives, and reconstruct Q_L (right edge of
            // `lo`) and Q_R (left edge of `hi`, via reversed-stencil trick).
            // Default Limiter=Godunov returns the cell-center value
            // unchanged -- equivalent to the original first-order flux.
            // ------------------------------------------------------------
            auto compute_face = [&](int lo_i, int lo_j, int lo_k, int hi_i, int hi_j, int hi_k, int dir)
                -> Solver::Local::FluidRiemann::Flux
            {
                const int di = hi_i - lo_i;
                const int dj = hi_j - lo_j;
                const int dk = hi_k - lo_k;
                Solver::Local::Limiter::Primitive prim[6];
                for (int s = -2; s <= 3; ++s)
                {
                    Solver::Local::FluidRiemann::State raw =
                        make_state(lo_i + s * di, lo_j + s * dj, lo_k + s * dk, dir);
                    prim[s + 2] = Solver::Local::Limiter::ToPrimitive(raw, small);
                }
                // Q_L: right-edge of cell `lo` (stencil centered on prim[2]).
                Solver::Local::Limiter::Primitive stencil_L[5] =
                    { prim[0], prim[1], prim[2], prim[3], prim[4] };
                // Q_R: left-edge of cell `hi` (stencil centered on prim[3],
                //      reversed so right-edge reconstruction returns left-edge).
                Solver::Local::Limiter::Primitive stencil_R[5] =
                    { prim[5], prim[4], prim[3], prim[2], prim[1] };
                Solver::Local::Limiter::Primitive pL = limiter->Reconstruct(stencil_L);
                Solver::Local::Limiter::Primitive pR = limiter->Reconstruct(stencil_R);
                Solver::Local::FluidRiemann::State sL_face = Solver::Local::Limiter::ToState(pL, small);
                Solver::Local::FluidRiemann::State sR_face = Solver::Local::Limiter::ToState(pR, small);

                return riemannsolver->Solve(sL_face, sR_face, pref, small);
            };

            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
#if AMREX_SPACEDIM == 3
            Solver::Local::FluidRiemann::Flux flux_zlo, flux_zhi;
#endif
            try
            {
                flux_xlo = compute_face(i - 1, j,     k,     i,     j,     k,     X_dir);
                flux_xhi = compute_face(i,     j,     k,     i + 1, j,     k,     X_dir);
                flux_ylo = compute_face(i,     j - 1, k,     i,     j,     k,     Y_dir);
                flux_yhi = compute_face(i,     j,     k,     i,     j + 1, k,     Y_dir);
#if AMREX_SPACEDIM == 3
                flux_zlo = compute_face(i,     j,     k - 1, i,     j,     k,     Z_dir);
                flux_zhi = compute_face(i,     j,     k,     i,     j,     k + 1, Z_dir);
#endif

                // ------------------------------------------------------
                // SYMMETRY-FACE FLUX ENFORCEMENT.  At a REFLECT domain
                // face the exact Riemann solution has u* = 0: the only
                // nonzero flux component is the normal-momentum (pressure)
                // term.  Computing it numerically instead leaves a
                // roundoff residual of order eps*|S_L|*rho_eta (the stiff
                // sound speed amplifies machine eps ~1e-16 to ~1e-13),
                // applied with the SAME SIGN at the SAME cells every step
                // -- a coherent mass/energy pump through the symmetry
                // plane that feeds back exponentially in driven octant/
                // quadrant runs (measured: 100%% of the gas-mass
                // instability enters via the reflect faces; RHSDECOMP
                // budget closes to 1e-25 with them zeroed analytically).
                // ------------------------------------------------------
                auto symmetrize = [](Solver::Local::FluidRiemann::Flux &f) {
                    f.mass0 = 0.0;
                    f.mass1 = 0.0;
                    f.momentum_tangent  = 0.0;
                    f.momentum_tangent2 = 0.0;
                    f.energy_total = 0.0;
                    f.energy0 = 0.0;
                    f.energy1 = 0.0;
                };
                if (sym_face_lo[0] && i == domain.smallEnd(0)) symmetrize(flux_xlo);
                if (sym_face_hi[0] && i == domain.bigEnd(0))   symmetrize(flux_xhi);
                if (sym_face_lo[1] && j == domain.smallEnd(1)) symmetrize(flux_ylo);
                if (sym_face_hi[1] && j == domain.bigEnd(1))   symmetrize(flux_yhi);
#if AMREX_SPACEDIM == 3
                if (sym_face_lo[2] && k == domain.smallEnd(2)) symmetrize(flux_zlo);
                if (sym_face_hi[2] && k == domain.bigEnd(2))   symmetrize(flux_zhi);
#endif
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN RIEMANN SOLVERS (6-eq)");
                Util::ParallelMessage(INFO, "lev=", lev, " i=", i, " j=", j, " k=", k);
                Util::Abort(INFO);
            }


            // NOTE: two rearrangements of the capillary force were implemented and
            // REMOVED.  (1) div(Omega) from face-evaluated Omega; (2) Omega carried
            // in the momentum flux, F = rho u u + p I - Omega, subtracted from the
            // Riemann fluxes.  These are ALGEBRAICALLY IDENTICAL -- differencing
            // face fluxes reproduces the face divergence exactly -- and measured
            // identical to 5 s.f. at t=20us on the static R0=24mm bubble (|u|max
            // 0.05015 both).  Both are WORSE than the cell-centred 2*dx source
            // (|u|max at 0.3ms: 29.9 source vs 78.0 face-Fsv), and (2) additionally
            // crashed before 0.18 ms because subtracting Omega AFTER the Riemann
            // solve leaves the solver's wave speeds unaware of it, so the fluxes
            // stop being consistent with the reconstructed states.
            // A real flux-form treatment needs Omega inside the WAVE STRUCTURE
            // (modified signal speeds), i.e. a change to the Riemann solver itself
            // -- not a rearrangement of where div(Omega) is evaluated.

            // ============================================================
            // Saurel-Abgrall non-conservative discretization for alpha and
            // per-phase internal energies (Saurel 2009 lines 836 / 852):
            //
            //   div(alpha u) computed from FACE-UPWIND alpha_face and
            //                S_M (= flux.u_interface).
            //   div(u)       from S_M.
            //   eta_rhs      = -div(alpha u) + alpha_C * div(u)
            //                = - u . grad(alpha)             (Sch20 row 1)
            //   E_k_rhs      = -div(alpha rho e_k u)
            //                  - alpha_k_C * p_k_C * div(u)  (Sau09 line 852)
            // where C subscript indicates CELL-CENTERED, TIME-n FROZEN
            // (the Saurel-Abgrall trick that makes the Abgrall uniform-flow
            //  test pass exactly).
            // ============================================================

            // Cell-centered per-phase pressures at time n (frozen for the
            // non-conservative source).
            const Set::Scalar a1_C   = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a2_C   = 1.0 - a1_C;
            const Set::Scalar p0_C   = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1_C, eos0.Gamma(), eos0.P0(), small);
            const Set::Scalar p1_C   = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2_C, eos1.Gamma(), eos1.P0(), small);

            // Face-upwind alpha (constant across acoustic; advected at S_M).
            Set::Scalar a_face_xlo = (flux_xlo.u_interface > 0.0) ? eta(i - 1, j, k) : eta(i,     j, k);
            Set::Scalar a_face_xhi = (flux_xhi.u_interface > 0.0) ? eta(i,     j, k) : eta(i + 1, j, k);
            Set::Scalar a_face_ylo = (flux_ylo.u_interface > 0.0) ? eta(i, j - 1, k) : eta(i, j,     k);
            Set::Scalar a_face_yhi = (flux_yhi.u_interface > 0.0) ? eta(i, j,     k) : eta(i, j + 1, k);
            a_face_xlo = std::min(std::max(a_face_xlo, 0.0), 1.0);
            a_face_xhi = std::min(std::max(a_face_xhi, 0.0), 1.0);
            a_face_ylo = std::min(std::max(a_face_ylo, 0.0), 1.0);
            a_face_yhi = std::min(std::max(a_face_yhi, 0.0), 1.0);
            // Colour function c: face-upwind on the SAME contact velocity as
            // alpha, so c and eta share one discrete advection operator.
            // Schmidmayer 2017 sec. 4.2.1: the c row of H_nc is -c, exactly as
            // the alpha_1 row is -alpha_1, and both are driven by the Riemann
            // star velocity u*.  Advecting c with the cell-centred u instead
            // (as the first implementation did) gives it a different truncation
            // error from eta, and the two separate -- measured at 2.28% in R
            // over 15 ms.  NOT clamped to [0,1]: c is a marker, and clipping it
            // would inject exactly the non-material change we are avoiding.
            const Set::Scalar c_face_xlo = (flux_xlo.u_interface > 0.0) ? cfun(i - 1, j, k) : cfun(i,     j, k);
            const Set::Scalar c_face_xhi = (flux_xhi.u_interface > 0.0) ? cfun(i,     j, k) : cfun(i + 1, j, k);
            const Set::Scalar c_face_ylo = (flux_ylo.u_interface > 0.0) ? cfun(i, j - 1, k) : cfun(i, j,     k);
            const Set::Scalar c_face_yhi = (flux_yhi.u_interface > 0.0) ? cfun(i, j,     k) : cfun(i, j + 1, k);
#if AMREX_SPACEDIM == 3
            const Set::Scalar c_face_zlo = (flux_zlo.u_interface > 0.0) ? cfun(i, j, k - 1) : cfun(i, j, k);
            const Set::Scalar c_face_zhi = (flux_zhi.u_interface > 0.0) ? cfun(i, j, k)     : cfun(i, j, k + 1);
            Set::Scalar a_face_zlo = (flux_zlo.u_interface > 0.0) ? eta(i, j, k - 1) : eta(i, j, k);
            Set::Scalar a_face_zhi = (flux_zhi.u_interface > 0.0) ? eta(i, j, k)     : eta(i, j, k + 1);
            a_face_zlo = std::min(std::max(a_face_zlo, 0.0), 1.0);
            a_face_zhi = std::min(std::max(a_face_zhi, 0.0), 1.0);
#endif

            // -----------------------------------------------------------
            // Store hi-face Riemann fluxes at cell centers for reflux.
            // Cell (i,j) -> face (i+1/2,j) flux for d=0, (i,j+1/2) for d=1.
            // HEAD's Riemann solver already returns per-phase mass and
            // per-phase energy components, so we feed the FluxRegister
            // directly without an extra eta-weighted split (Matt Q's
            // a5783ab32 split is built into our flux struct).
            //
            // At box-lo boundaries also write the lo-face flux into the
            // (i-1) / (j-1) ghost cell -- FillBoundary in Advance() will
            // overwrite this from a neighbor where one exists, but at
            // coarse-fine boundaries no neighbor exists and these ghost
            // values are what FineAdd picks up (Matt Q 8a1633977 fix).
            // -----------------------------------------------------------
            // ff_mom_d stores, in component c, the flux of mixture-momentum
            // component c through the d-face: component d gets momentum_normal,
            // (d+1)%SD gets momentum_tangent, (d+2)%SD gets momentum_tangent2 --
            // the SAME cyclic mapping as make_state / the M_flux scatter, so the
            // reflux is self-consistent in any dimension.  In 2D this reduces to
            // the old explicit (x: 0=n,1=t ; y: 0=t,1=n) layout.
            if (have_cc_fluxes)
            {
                // x-direction hi-face (normal->0, tangent->1, tangent2->2)
                ff_mass_x  (i, j, k, 0) = flux_xhi.mass0;
                ff_mass_x  (i, j, k, 1) = flux_xhi.mass1;
                ff_mom_x   (i, j, k, 0) = flux_xhi.momentum_normal;
                ff_mom_x   (i, j, k, 1) = flux_xhi.momentum_tangent;
                ff_ene_x   (i, j, k)    = flux_xhi.energy_total;
                ff_ene_k_x (i, j, k, 0) = flux_xhi.energy0;
                ff_ene_k_x (i, j, k, 1) = flux_xhi.energy1;

                // y-direction hi-face (normal->1, tangent->2%SD, tangent2->0)
                ff_mass_y  (i, j, k, 0) = flux_yhi.mass0;
                ff_mass_y  (i, j, k, 1) = flux_yhi.mass1;
                ff_mom_y   (i, j, k, 1)                  = flux_yhi.momentum_normal;
                ff_mom_y   (i, j, k, 2 % AMREX_SPACEDIM) = flux_yhi.momentum_tangent;
                ff_ene_y   (i, j, k)    = flux_yhi.energy_total;
                ff_ene_k_y (i, j, k, 0) = flux_yhi.energy0;
                ff_ene_k_y (i, j, k, 1) = flux_yhi.energy1;
#if AMREX_SPACEDIM == 3
                ff_mom_x   (i, j, k, 2) = flux_xhi.momentum_tangent2;  // x tangent2 -> 2
                ff_mom_y   (i, j, k, 0) = flux_yhi.momentum_tangent2;  // y tangent2 -> 0

                // z-direction hi-face (normal->2, tangent->0, tangent2->1)
                ff_mass_z  (i, j, k, 0) = flux_zhi.mass0;
                ff_mass_z  (i, j, k, 1) = flux_zhi.mass1;
                ff_mom_z   (i, j, k, 2) = flux_zhi.momentum_normal;
                ff_mom_z   (i, j, k, 0) = flux_zhi.momentum_tangent;
                ff_mom_z   (i, j, k, 1) = flux_zhi.momentum_tangent2;
                ff_ene_z   (i, j, k)    = flux_zhi.energy_total;
                ff_ene_k_z (i, j, k, 0) = flux_zhi.energy0;
                ff_ene_k_z (i, j, k, 1) = flux_zhi.energy1;
#endif

                if (i == bx_lo.x) {
                    ff_mass_x  (i - 1, j, k, 0) = flux_xlo.mass0;
                    ff_mass_x  (i - 1, j, k, 1) = flux_xlo.mass1;
                    ff_mom_x   (i - 1, j, k, 0) = flux_xlo.momentum_normal;
                    ff_mom_x   (i - 1, j, k, 1) = flux_xlo.momentum_tangent;
                    ff_ene_x   (i - 1, j, k)    = flux_xlo.energy_total;
                    ff_ene_k_x (i - 1, j, k, 0) = flux_xlo.energy0;
                    ff_ene_k_x (i - 1, j, k, 1) = flux_xlo.energy1;
#if AMREX_SPACEDIM == 3
                    ff_mom_x   (i - 1, j, k, 2) = flux_xlo.momentum_tangent2;
#endif
                }
                if (j == bx_lo.y) {
                    ff_mass_y  (i, j - 1, k, 0) = flux_ylo.mass0;
                    ff_mass_y  (i, j - 1, k, 1) = flux_ylo.mass1;
                    ff_mom_y   (i, j - 1, k, 1)                  = flux_ylo.momentum_normal;
                    ff_mom_y   (i, j - 1, k, 2 % AMREX_SPACEDIM) = flux_ylo.momentum_tangent;
                    ff_ene_y   (i, j - 1, k)    = flux_ylo.energy_total;
                    ff_ene_k_y (i, j - 1, k, 0) = flux_ylo.energy0;
                    ff_ene_k_y (i, j - 1, k, 1) = flux_ylo.energy1;
#if AMREX_SPACEDIM == 3
                    ff_mom_y   (i, j - 1, k, 0) = flux_ylo.momentum_tangent2;
#endif
                }
#if AMREX_SPACEDIM == 3
                if (k == bx_lo.z) {
                    ff_mass_z  (i, j, k - 1, 0) = flux_zlo.mass0;
                    ff_mass_z  (i, j, k - 1, 1) = flux_zlo.mass1;
                    ff_mom_z   (i, j, k - 1, 2) = flux_zlo.momentum_normal;
                    ff_mom_z   (i, j, k - 1, 0) = flux_zlo.momentum_tangent;
                    ff_mom_z   (i, j, k - 1, 1) = flux_zlo.momentum_tangent2;
                    ff_ene_z   (i, j, k - 1)    = flux_zlo.energy_total;
                    ff_ene_k_z (i, j, k - 1, 0) = flux_zlo.energy0;
                    ff_ene_k_z (i, j, k - 1, 1) = flux_zlo.energy1;
                }
#endif
            }

            // div(alpha u) and div(u) (from S_M = flux.u_interface).
            const Set::Scalar div_uA_x = (flux_xhi.u_interface * a_face_xhi
                                        - flux_xlo.u_interface * a_face_xlo) / DX[0];
            const Set::Scalar div_uA_y = (flux_yhi.u_interface * a_face_yhi
                                        - flux_ylo.u_interface * a_face_ylo) / DX[1];
            const Set::Scalar div_u_x  = (flux_xhi.u_interface - flux_xlo.u_interface) / DX[0];
            const Set::Scalar div_u_y  = (flux_yhi.u_interface - flux_ylo.u_interface) / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar div_uA_z = (flux_zhi.u_interface * a_face_zhi
                                        - flux_zlo.u_interface * a_face_zlo) / DX[2];
            const Set::Scalar div_u_z  = (flux_zhi.u_interface - flux_zlo.u_interface) / DX[2];
#endif
            const Set::Scalar div_u    = AMREX_D_TERM(div_u_x, + div_u_y, + div_u_z);

            // ------------------------------------------------------------
            // Volume fraction (alpha_1) row -- Saurel 2009 line 836:
            //   d(alpha)/dt + u . grad(alpha) = 0
            //   discretized as -div(alpha u) + alpha_C div(u).
            // The 6-eq model REPLACES the 5-eq Kapila K-source with the
            // stiff relaxation source (handled in the post-stage hook).
            // ------------------------------------------------------------
            const Set::Scalar eta_advect = -(AMREX_D_TERM(div_uA_x, + div_uA_y, + div_uA_z)) + eta(i, j, k) * div_u;
            eta_rhs(i, j, k) = eta_advect + eta_dot_Vap;

            // ------------------------------------------------------------
            // SHELL row -- the AREAL density Gamma is the primary, transported by
            // the exact surfactant-on-a-deforming-interface law (Stone 1990):
            //
            //     D(Gamma)/Dt = -Gamma (div_s u)      div_s u = div(u) - n.grad(u).n
            //  => Gamma_rhs   = -div(Gamma u) + Gamma (n.grad(u).n)
            //
            // WHY THIS FORM.  The previous version stored the volumetric density
            // c = Gamma|grad eta| and recovered Gamma = c/|grad eta|.  That
            // division is ill-conditioned exactly where |grad eta| -> 0, so any
            // residual mismatch between how c and eta are discretely transported
            // showed up first in the band tails: measured Gamma -> 1.79 mean /
            // 11.9 max in the gas-side tail at 5 ms while the band core was still
            // 0.95-1.01, after which the piecewise sigma(Gamma) rectified that
            // noise into force and it leaked inward.  Damping or gating the tail
            // only masks it.
            //
            // Here NO |grad eta| appears at all.  The only coupling to eta is the
            // UNIT normal n_hat, which stays well-conditioned right through the
            // tails, and Gamma uses the same face-upwind values and the same
            // interface velocity S_M as the eta row -- so the two are transported
            // by the identical operator and cannot drift apart.  It also makes
            // Gamma immune to NON-MATERIAL changes in eta (pressure relaxation,
            // clamping, ghost reconstruction) for free, since Gamma is never
            // reconstructed from eta.
            //
            // Trade-off: this is the non-conservative form, so total shell mass is
            // conserved only to truncation rather than to machine precision.  The
            // conservative form conserved it exactly and still failed, because a
            // well-conditioned Gamma -- not exact conservation -- is what the
            // sigma(Gamma) map actually needs.
            // Gamma is reconstructed at faces with the SAME limiter eta uses for
            // its Riemann reconstruction (Limiter.type = godunov|minmod|vanleer|
            // weno3|weno5).  Every limiter reconstructs componentwise, so we route
            // Gamma through the existing Primitive API in the `alpha` slot -- the
            // other fields are inert.  With Limiter=Godunov this collapses exactly
            // to the first-order upwind difference, so the scheme is unchanged at
            // the default and only improves when a real limiter is selected.
            //
            // The VELOCITY stays cell-centered, matching gradu in the stretch term
            // below.  An earlier version paired the face interface velocity S_M
            // with cell-centered gradu; the O(dx) disagreement biased Gamma upward
            // ~2%/ms (sigma decayed to 0.54 at 10 ms while the bubble expanded --
            // the shell read as COMPRESSING while its surface grew).  Only the
            // Gamma INTERPOLATION is raised in order here, never the velocity.
            auto shell_face = [&](int di, int dj, int dk,
                                  Set::Scalar &qL, Set::Scalar &qR)
            {
                Solver::Local::Limiter::Primitive prim[6];
                for (int sft = -2; sft <= 3; ++sft)
                    prim[sft + 2].alpha = shell(i + sft * di, j + sft * dj, k + sft * dk);
                Solver::Local::Limiter::Primitive stL[5] =
                    { prim[0], prim[1], prim[2], prim[3], prim[4] };
                Solver::Local::Limiter::Primitive stR[5] =
                    { prim[5], prim[4], prim[3], prim[2], prim[1] };
                qL = limiter->Reconstruct(stL).alpha;   // right edge of the `lo` cell
                qR = limiter->Reconstruct(stR).alpha;   // left  edge of the `hi` cell
            };

            //   D(Gamma)/Dt = -Gamma (div u - n.grad(u).n)
            // Upwind-biased high-order gradient: for u > 0 difference the two
            // left-reconstructed faces, for u < 0 the two right-reconstructed ones.
            const Set::Scalar ux = u(0), uy = u(1);
            Set::Scalar aL, aR, bL, bR;
            shell_face(1, 0, 0, aL, aR);            // face i+1/2 (lo = i)
            Set::Scalar cL, cR;
            {   // face i-1/2 (lo = i-1): shift the stencil one cell down
                Solver::Local::Limiter::Primitive prim[6];
                for (int sft = -2; sft <= 3; ++sft) prim[sft + 2].alpha = shell(i - 1 + sft, j, k);
                Solver::Local::Limiter::Primitive stL[5] = { prim[0], prim[1], prim[2], prim[3], prim[4] };
                Solver::Local::Limiter::Primitive stR[5] = { prim[5], prim[4], prim[3], prim[2], prim[1] };
                cL = limiter->Reconstruct(stL).alpha;
                cR = limiter->Reconstruct(stR).alpha;
            }
            const Set::Scalar dGx = (ux > 0.0) ? (aL - cL) / DX[0] : (aR - cR) / DX[0];

            shell_face(0, 1, 0, bL, bR);            // face j+1/2 (lo = j)
            Set::Scalar dL, dR;
            {   // face j-1/2 (lo = j-1)
                Solver::Local::Limiter::Primitive prim[6];
                for (int sft = -2; sft <= 3; ++sft) prim[sft + 2].alpha = shell(i, j - 1 + sft, k);
                Solver::Local::Limiter::Primitive stL[5] = { prim[0], prim[1], prim[2], prim[3], prim[4] };
                Solver::Local::Limiter::Primitive stR[5] = { prim[5], prim[4], prim[3], prim[2], prim[1] };
                dL = limiter->Reconstruct(stL).alpha;
                dR = limiter->Reconstruct(stR).alpha;
            }
            const Set::Scalar dGy = (uy > 0.0) ? (bL - dL) / DX[1] : (bR - dR) / DX[1];

#if AMREX_SPACEDIM == 3
            const Set::Scalar uz = u(2);
            Set::Scalar eL, eR, fL, fR;
            shell_face(0, 0, 1, eL, eR);            // face k+1/2 (lo = k)
            {   // face k-1/2 (lo = k-1)
                Solver::Local::Limiter::Primitive prim[6];
                for (int sft = -2; sft <= 3; ++sft) prim[sft + 2].alpha = shell(i, j, k - 1 + sft);
                Solver::Local::Limiter::Primitive stL[5] = { prim[0], prim[1], prim[2], prim[3], prim[4] };
                Solver::Local::Limiter::Primitive stR[5] = { prim[5], prim[4], prim[3], prim[2], prim[1] };
                fL = limiter->Reconstruct(stL).alpha;
                fR = limiter->Reconstruct(stR).alpha;
            }
            const Set::Scalar dGz = (uz > 0.0) ? (eL - fL) / DX[2] : (eR - fR) / DX[2];
#endif
            const Set::Scalar u_dot_gradG = AMREX_D_TERM(ux * dGx, + uy * dGy, + uz * dGz);
            // Surface divergence.  In the bulk grad_eta -> 0 so n_hat -> 0 and this
            // reduces to div(u); Gamma there is unused.
            // NOTE: differencing the ratio u = M/rho directly, instead of the
            // product-rule form (grad M - u grad rho)/rho that `gradu` uses, was
            // tried on the suspicion that the 10:1 density swing across the band
            // spoils the reconstruction.  It changed nothing measurable (Gamma
            // response 82.4% / 83.1% vs 83.0%), so gradu is not the issue and the
            // simpler shared expression is kept.
            const Set::Scalar div_s_u = gradu.trace() - n_hat.dot(gradu * n_hat);
            shell_rhs(i, j, k) = -u_dot_gradG - shell(i, j, k) * div_s_u;

            // ------------------------------------------------------------
            // COLOUR FUNCTION  dc/dt + u.grad(c) = 0   (Schmidmayer 2017 eq. 3)
            // ------------------------------------------------------------
            // Discretised EXACTLY as the alpha_1 row above: -div(c u*) + c div(u*),
            // with u* the HLLC contact velocity.  Same flux, same star velocity,
            // same truncation error -- so c and eta cannot drift apart for
            // numerical reasons, only through the physical difference between
            // them (alpha_1 carries the relaxation source mu(P1-P2); c does not).
            const Set::Scalar div_uC_x = (flux_xhi.u_interface * c_face_xhi
                                        - flux_xlo.u_interface * c_face_xlo) / DX[0];
            const Set::Scalar div_uC_y = (flux_yhi.u_interface * c_face_yhi
                                        - flux_ylo.u_interface * c_face_ylo) / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar div_uC_z = (flux_zhi.u_interface * c_face_zhi
                                        - flux_zlo.u_interface * c_face_zlo) / DX[2];
#endif
            cfun_rhs(i, j, k) = -(AMREX_D_TERM(div_uC_x, + div_uC_y, + div_uC_z))
                              + cfun(i, j, k) * div_u;

            // NOTE: a surface-diffusion term  + D_s grad_s^2(Gamma)  (Stone 1990)
            // was implemented and tested here, then removed.  It does not fix the
            // Gamma feedback instability.  Measured at eps = 4 finest cells / 4 AMR
            // levels, sigma spread at t = 0.5 ms:
            //   tangential grad_s^2:  D_s = 0 / 1e-5 / 1e-4 / 1e-3 -> 226 / 225 / 204 / 192 %
            //   full Laplacian:       D_s = 1e-4 / 1e-3 / 1e-2     -> 194 / 168 / 73 %
            // Only D_s = 1e-2 bites, and that sits AT the explicit diffusion limit
            // (dt_max = dx^2/4D_s = 9.5e-7 vs the subcycled fine dt of 6.9e-7) while
            // still growing exponentially -- delayed, not stabilized, at a diffusivity
            // ~10 orders above real lipid values.  The unstable modes live in the
            // band-NORMAL degrees of freedom of Gamma, which are unphysical for a
            // surface field; damping cannot fix a mode whose root cause is that the
            // degree of freedom should not exist.


            // ------------------------------------------------------------
            // Phase-mass rows (pure conservation, no source from h):
            //   d(alpha rho)_k / dt + div((alpha rho)_k u) = 0
            // ------------------------------------------------------------
            const Set::Scalar rho_eta0_flux = AMREX_D_TERM(
                                              (flux_xlo.mass0 - flux_xhi.mass0) / DX[0],
                                            + (flux_ylo.mass0 - flux_yhi.mass0) / DX[1],
                                            + (flux_zlo.mass0 - flux_zhi.mass0) / DX[2]);
            const Set::Scalar rho_eta1_flux = AMREX_D_TERM(
                                              (flux_xlo.mass1 - flux_xhi.mass1) / DX[0],
                                            + (flux_ylo.mass1 - flux_yhi.mass1) / DX[1],
                                            + (flux_zlo.mass1 - flux_zhi.mass1) / DX[2]);

            rho_eta0_rhs(i, j, k) = rho_eta0_flux + Source(i, j, k, 0) * (eta(i, j, k))         + m_dot_Vap;
            rho_eta1_rhs(i, j, k) = rho_eta1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k))   - m_dot_Vap;
            // [DIAG RHSDECOMP] stash boundary-face mass1 fluxes + source terms
            if (rhsp_fxlo.contains(i, j, k))
            {
                rhsp_fxlo(i, j, k) = flux_xlo.mass1;
                rhsp_fxhi(i, j, k) = flux_xhi.mass1;
                rhsp_fylo(i, j, k) = flux_ylo.mass1;
                rhsp_fyhi(i, j, k) = flux_yhi.mass1;
                rhsp_src (i, j, k) = Source(i, j, k, 0) * (1.0 - eta(i, j, k)) - m_dot_Vap;
            }

            // Diagnostic mass flux (kept for plotfile compatibility):
            rho_flux(i, j, k) = rho_eta0_flux + rho_eta1_flux;

            // ------------------------------------------------------------
            // Mixture momentum (pure conservation, with body sources):
            // ------------------------------------------------------------
            // In RHS we accumulate as (F_lo - F_hi)/dx.  On each face with normal
            // `dir`, momentum_normal feeds M-component `dir` and the tangential
            // momentum fluxes feed the cyclically-next components (dir+1, dir+2) --
            // mirroring make_state's normal/tangent mapping so the round-trip is
            // self-consistent.  The `% AMREX_SPACEDIM` keeps the 2D path identical
            // (x-tangent->1, y-tangent->0) while the 3D path closes the cycle.
            for (int n = 0; n < AMREX_SPACEDIM; ++n) M_flux(i, j, k, n) = 0.0;
            // x-faces (normal->0, tangent->1)
            M_flux(i, j, k, 0)                  += (flux_xlo.momentum_normal  - flux_xhi.momentum_normal ) / DX[0];
            M_flux(i, j, k, 1 % AMREX_SPACEDIM) += (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / DX[0];
            // y-faces (normal->1, tangent->2%SD)
            M_flux(i, j, k, 1)                  += (flux_ylo.momentum_normal  - flux_yhi.momentum_normal ) / DX[1];
            M_flux(i, j, k, 2 % AMREX_SPACEDIM) += (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / DX[1];
#if AMREX_SPACEDIM == 3
            // second tangents of the x/y faces, plus the full z-faces.
            M_flux(i, j, k, 2) += (flux_xlo.momentum_tangent2 - flux_xhi.momentum_tangent2) / DX[0]; // x t2 -> 2
            M_flux(i, j, k, 0) += (flux_ylo.momentum_tangent2 - flux_yhi.momentum_tangent2) / DX[1]; // y t2 -> 0
            // z-faces (normal->2, tangent->0, tangent2->1)
            M_flux(i, j, k, 2) += (flux_zlo.momentum_normal   - flux_zhi.momentum_normal  ) / DX[2];
            M_flux(i, j, k, 0) += (flux_zlo.momentum_tangent  - flux_zhi.momentum_tangent ) / DX[2];
            M_flux(i, j, k, 1) += (flux_zlo.momentum_tangent2 - flux_zhi.momentum_tangent2) / DX[2];
#endif

            for (int n = 0; n < AMREX_SPACEDIM; ++n)
                M_rhs(i, j, k, n) = M_flux(i, j, k, n) + Source(i, j, k, 1 + n);

            // ------------------------------------------------------------
            // Redundant mixture total energy (pure conservation):
            //   d(rho E)/dt + div((rho E + p) u) = 0     (Sch20 eq. 16)
            // ------------------------------------------------------------
            E_flux(i, j, k) = AMREX_D_TERM(
                              (flux_xlo.energy_total - flux_xhi.energy_total) / DX[0],
                            + (flux_ylo.energy_total - flux_yhi.energy_total) / DX[1],
                            + (flux_zlo.energy_total - flux_zhi.energy_total) / DX[2]);

            // CAPILLARY ENERGY CLOSURE (Schmidmayer 2017): the energy flux
            // carries -Omega.u, i.e. the RHS gains +div(Omega.u).  G is built
            // cell-centred and averaged to faces, so the face value is shared
            // between neighbours and the sum telescopes -- discretely
            // conservative.  Note the (hi - lo) ordering here is the OPPOSITE
            // of the (lo - hi) convention above, because that term is -div(F)
            // while this one is +div(G).
            if (false && apply_surface_tension)   // superseded by L_cap
            {
                auto Gcap = [&](int ii, int jj, int kk, int d) -> Set::Scalar {
#if AMREX_SPACEDIM == 2
                    const Set::Scalar oxx = om_cap(ii,jj,kk,0), oyy = om_cap(ii,jj,kk,1);
                    const Set::Scalar oxy = om_cap(ii,jj,kk,2);
                    const Set::Scalar ux = v(ii,jj,kk,0), uy = v(ii,jj,kk,1);
                    return (d == 0) ? (oxx * ux + oxy * uy) : (oxy * ux + oyy * uy);
#else
                    const Set::Scalar oxx = om_cap(ii,jj,kk,0), oyy = om_cap(ii,jj,kk,1);
                    const Set::Scalar ozz = om_cap(ii,jj,kk,2), oxy = om_cap(ii,jj,kk,3);
                    const Set::Scalar oxz = om_cap(ii,jj,kk,4), oyz = om_cap(ii,jj,kk,5);
                    const Set::Scalar ux = v(ii,jj,kk,0), uy = v(ii,jj,kk,1), uz = v(ii,jj,kk,2);
                    if (d == 0) return oxx * ux + oxy * uy + oxz * uz;
                    if (d == 1) return oxy * ux + oyy * uy + oyz * uz;
                    return              oxz * ux + oyz * uy + ozz * uz;
#endif
                };
                Set::Scalar cap_div = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    const Set::Scalar g_hi = 0.5 * (Gcap(i, j, k, d)
                                                  + Gcap(i + di, j + dj, k + dk, d));
                    const Set::Scalar g_lo = 0.5 * (Gcap(i - di, j - dj, k - dk, d)
                                                  + Gcap(i, j, k, d));
                    cap_div += (g_hi - g_lo) / DX[d];
                }
                E_flux(i, j, k) += cap_div;
            }
            E_rhs(i, j, k)  = E_flux(i, j, k) + Source(i, j, k, AMREX_SPACEDIM + 1);

            // ------------------------------------------------------------
            // Per-phase internal energies (Sch20 eq. 13 last two rows /
            // Sau09 line 852):
            //   d(alpha rho e)_k/dt + div((alpha rho e)_k u)
            //                       + alpha_k_C p_k_C div(u) = 0
            // (The +/- mu p_I (p_1-p_2) relaxation source is deferred to
            //  the post-stage hook, in the stiff-relaxation limit.)
            // ------------------------------------------------------------
            const Set::Scalar E0_flux_div = AMREX_D_TERM(
                                            (flux_xlo.energy0 - flux_xhi.energy0) / DX[0],
                                          + (flux_ylo.energy0 - flux_yhi.energy0) / DX[1],
                                          + (flux_zlo.energy0 - flux_zhi.energy0) / DX[2]);
            const Set::Scalar E1_flux_div = AMREX_D_TERM(
                                            (flux_xlo.energy1 - flux_xhi.energy1) / DX[0],
                                          + (flux_ylo.energy1 - flux_yhi.energy1) / DX[1],
                                          + (flux_zlo.energy1 - flux_zhi.energy1) / DX[2]);

            E0_rhs(i, j, k) = (E0_flux_div - a1_C * p0_C * div_u);
            E1_rhs(i, j, k) = (E1_flux_div - a2_C * p1_C * div_u);

            // ------------------------------------------------------------
            // Artificial heat exchange (AHE) -- Schmidmayer 2020 eq. 13
            // r-source on per-phase internal energies:
            //     E0_rhs += -mu p_I (p_0 - p_1)
            //     E1_rhs += +mu p_I (p_0 - p_1)
            //
            // Sch20 does NOT specify a finite mu
            // (they take the stiff limit inside operator-split Newton); the
            // Saurel-Petitpas-Berry 2009 sec. 5.3 curve fit used here for
            // mu(v) was originally calibrated for the q*@u/@x form, NOT the
            // antisymmetric Sch20 form.  Magnitudes are a starting point
            // and must be re-calibrated for the target mixture.
            //
            // NOT the most conservative method in this codebase.  Pairwise
            // conservative across the two energy rows (sources sum to zero),
            // but per-phase energy is only conserved pairwise; total mixture
            // energy is maintained via the redundant rho-E reinit step.  If
            // the reinit step were disabled the per-phase energies would
            // drift -- the AHE source intentionally trades that property
            // for non-stiff pressure relaxation dynamics.
            // ------------------------------------------------------------
            if (apply_ahe)
            {
                // ----------------------------------------------------------
                // Local quantities shared by both AHE forms.
                // ----------------------------------------------------------
                const Set::Scalar rho0_pure = std::max(rho_eta0(i,j,k) / std::max(a1_C, small), small);
                const Set::Scalar rho1_pure = std::max(rho_eta1(i,j,k) / std::max(a2_C, small), small);

                const Set::Scalar c0_C = Solver::EOS::EOS::PhasicSoundSpeed(
                    rho0_pure, p0_C, eos0.Gamma(), eos0.P0(), small);
                const Set::Scalar c1_C = Solver::EOS::EOS::PhasicSoundSpeed(
                    rho1_pure, p1_C, eos1.Gamma(), eos1.P0(), small);

                // mu (or q*-coefficient in Sau09 form) -- either constant or
                // Sau09 Sec.5.3 curve fit.
                Set::Scalar mu_ahe;
                if (ahe_use_const_mu)
                {
                    mu_ahe = ahe_mu_const;
                }
                else
                {
                    const Set::Scalar rho_mix_loc = std::max(rho_eta0(i,j,k) + rho_eta1(i,j,k), small);
                    const Set::Scalar v_mix       = 1.0 / rho_mix_loc;
                    mu_ahe = ahe_mu_scale * Solver::EOS::EOS::AHE_mu_Saurel(
                        v_mix, ahe_mu_a, ahe_mu_b, ahe_mu_c, ahe_v_min, ahe_v_max);
                }

                // Acoustic-CFL based source cap (used by both forms).
                const Set::Scalar E0_loc = std::max(E0_arr(i, j, k), small);
                const Set::Scalar E1_loc = std::max(E1_arr(i, j, k), small);
                const Set::Scalar E_min  = std::min(E0_loc, E1_loc);
                const Set::Scalar rho_loc_lim = std::max(rho_eta0(i,j,k) + rho_eta1(i,j,k), small);
                const Set::Scalar u_mag_lim = std::sqrt(AMREX_D_TERM(  M(i, j, k, 0) * M(i, j, k, 0),
                                                                    +  M(i, j, k, 1) * M(i, j, k, 1),
                                                                    +  M(i, j, k, 2) * M(i, j, k, 2))) / rho_loc_lim;
                const Set::Scalar c_local = std::max(c0_C, c1_C);
                const Set::Scalar dx_min  = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});
                const Set::Scalar src_cap = ahe_max_frac * E_min * (u_mag_lim + c_local) / std::max(dx_min, small);

                if (ahe_form == 0)
                {
                    // ======================================================
                    // Sch20 r-source form (antisymmetric +/- mu p_I Dp).
                    // ======================================================
                    const Set::Scalar Z0_C = rho0_pure * c0_C;
                    const Set::Scalar Z1_C = rho1_pure * c1_C;
                    const Set::Scalar p_I  = Solver::EOS::EOS::InterfacialPressureZ(
                        p0_C, p1_C, Z0_C, Z1_C, small);
                    const Set::Scalar delta_p = p0_C - p1_C;

                    const bool gate_on = (!ahe_compression_only) || (div_u < 0.0);
                    Set::Scalar ahe_src = gate_on ? (mu_ahe * p_I * delta_p) : 0.0;
                    ahe_src = std::max(std::min(ahe_src, src_cap), -src_cap);

                    E0_rhs(i, j, k) += -ahe_src;
                    E1_rhs(i, j, k) += +ahe_src;

                    if (ahe_apply_alpha_src && gate_on)
                    {
                        Set::Scalar eta_src = mu_ahe * delta_p;
                        const Set::Scalar eta_src_cap = ahe_max_frac * (u_mag_lim + c_local) / std::max(dx_min, small);
                        eta_src = std::max(std::min(eta_src, eta_src_cap), -eta_src_cap);
                        eta_rhs(i, j, k) += eta_src;
                    }
                }
                else
                {
                    // ======================================================
                    // Sau09 sec.5.3 q*@u/@x form (default).
                    //
                    // Sau09 eq. (above 5.3):
                    //   d(a_k rho_k e_k)/dt + d(a_k rho_k e_k u)/dx
                    //                       + (a_k p_k + q*) du/dx = 0,
                    //   q* = g(du/dx) * mu(v),
                    //   g  = 1 if du/dx < 0 else 0       (compression only).
                    //
                    // Both phases get the SAME -q* div(u) on the RHS (NOT
                    // antisymmetric).  q* acts as an extra effective pressure
                    // active only in the shock layer.  The Newton inside
                    // RelaxAndReinit sees Sau09-modified per-phase energies
                    // and produces a consistent shocked state -- this is
                    // the form used to obtain Sau09 Fig. 22 convergence.
                    //
                    // No alpha source in this form.
                    // ======================================================
                    const Set::Scalar g_ind = (div_u < 0.0) ? 1.0 : 0.0;
                    const Set::Scalar q_star = g_ind * mu_ahe;

                    Set::Scalar q_src = q_star * div_u;   // = -q*|div_u| <= 0 in compression
                    q_src = std::max(std::min(q_src, src_cap), -src_cap);

                    E0_rhs(i, j, k) += -q_src;
                    E1_rhs(i, j, k) += -q_src;
                }
            }

            // ============================================================
            // EMBEDDED SOLID BOUNDARY -- Yang(2023) diffuse-domain
            // no-penetration via MOMENTUM-ONLY Brinkman penalization.
            // ------------------------------------------------------------
            // MOMENTUM diffuse-interface no-penetration (Yang et al. 2023,
            // term phi0/kappa (u_S - u); orig. Angot 1999 / Liu & Vasilyev
            // 2007)
            // ============================================================
            if (embedded.apply && embedded.brinkman > 0.0)
            {
                const Set::Scalar chi = 1.0 - phi_c;
                if (chi > 0.0)
                {
                    const Set::Scalar lam = embedded.brinkman * chi;
                    Set::Vector nrm = embedded.slip ? Numeric::Gradient(phisol, i, j, k, 0, DX)
                                                    : Set::Vector::Zero();
                    const Set::Scalar nmag = embedded.slip ? nrm.norm() : 0.0;
                    if (embedded.slip && nmag > 1e-8)
                    {
                        // SLIP wall: penalize only the wall-NORMAL momentum
                        // (no-penetration), leaving tangential free.  Normal from
                        // grad(phi).  (Deep in the solid grad~0 -> falls through to
                        // the full no-slip branch below to hold the static solid.)
                        const Set::Vector nhat = nrm / nmag;
                        Set::Scalar dMn = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            dMn += (M(i, j, k, d) - s_M(i, j, k, d)) * nhat(d);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            M_rhs(i, j, k, d) += -lam * dMn * nhat(d);
                    }
                    else
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            M_rhs(i, j, k, d) += -lam * (M(i, j, k, d) - s_M(i, j, k, d));
                    }
                    // momentum-only: no mass/energy penalty (pressure builds,
                    // arrested KE -> internal energy; energy-conserving).
                }
            }

           // ------------------------------------------------------------
           // Error Checking
           // ------------------------------------------------------------
           if ( (M_rhs(i, j, k, 0) != M_rhs(i, j, k, 0))
                or (M_rhs(i, j, k, 1) != M_rhs(i, j, k, 1))
                or (E_rhs(i, j, k) != E_rhs(i, j, k))
                or (E0_rhs(i, j, k) != E0_rhs(i, j, k))
                or (E1_rhs(i, j, k) != E1_rhs(i, j, k))
                or (eta_rhs(i, j, k) != eta_rhs(i, j, k))
                or (rho_eta0_rhs(i, j, k) != rho_eta0_rhs(i, j, k))
                or (rho_eta1_rhs(i, j, k) != rho_eta1_rhs(i, j, k)))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HYDRO2 (6-eq RHS)");
                Util::ParallelMessage(INFO, "time=", time, " lev=", lev, " i=", i, " j=", j);
                Util::ParallelMessage(INFO, "flux_xlo: ", flux_xlo);
                Util::ParallelMessage(INFO, "flux_xhi: ", flux_xhi);
                Util::ParallelMessage(INFO, "flux_ylo: ", flux_ylo);
                Util::ParallelMessage(INFO, "flux_yhi: ", flux_yhi);
                Util::ParallelMessage(INFO, "Source=", Source(i, j, k, 0), ", ", Source(i, j, k, 1), ", ", Source(i, j, k, 2), ", ", Source(i, j, k, 3));
                Util::ParallelMessage(INFO, "drhoeta0/dt=", rho_eta0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", rho_eta1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dE0/dt=", E0_rhs(i, j, k), " dE1/dt=", E1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "deta/dt=", eta_rhs(i, j, k));
                Util::Abort(INFO);
            }

            // Calculate vorticity (curl of velocity) for visualization.
#if AMREX_SPACEDIM == 2
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));           // omega_z
#else
            omega(i, j, k, 0) = gradu(2, 1) - gradu(1, 2);          // omega_x = du_z/dy - du_y/dz
            omega(i, j, k, 1) = gradu(0, 2) - gradu(2, 0);          // omega_y = du_x/dz - du_z/dx
            omega(i, j, k, 2) = gradu(1, 0) - gradu(0, 1);          // omega_z = du_y/dx - du_x/dy
#endif
        });
    }
    // [DIAG RHSDECOMP] mass1 RHS budget: sum(rhs)*dV must equal
    // net boundary influx + interior sources.  Printed every 300 calls.
    if (relax_diag)
    {
        const amrex::Box &dom = geom[lev].Domain();
        Set::Scalar bx_lo = 0, bx_hi = 0, by_lo = 0, by_hi = 0, src_sum = 0;
        for (amrex::MFIter pmfi(rhsp_fxlo_mf, false); pmfi.isValid(); ++pmfi)
        {
            const amrex::Box &pbx = pmfi.validbox();
            auto flo = rhsp_fxlo_mf.array(pmfi);
            auto fhi = rhsp_fxhi_mf.array(pmfi);
            auto glo = rhsp_fylo_mf.array(pmfi);
            auto ghi = rhsp_fyhi_mf.array(pmfi);
            auto sr  = rhsp_src_mf.array(pmfi);
            amrex::LoopOnCpu(pbx, [&](int i, int j, int k) {
                if (i == dom.smallEnd(0)) bx_lo += flo(i, j, k) * DX[1];
                if (i == dom.bigEnd(0))   bx_hi -= fhi(i, j, k) * DX[1];
                if (j == dom.smallEnd(1)) by_lo += glo(i, j, k) * DX[0];
                if (j == dom.bigEnd(1))   by_hi -= ghi(i, j, k) * DX[0];
                src_sum += sr(i, j, k) * DX[0] * DX[1];
            });
        }
        Set::Scalar rhs_sum = rho_eta1_rhs_mf.sum(0) * DX[0] * DX[1];
        static int _rd_n = 0;
        if ((_rd_n++ % 300) == 0)
        {
            char b[260];
            snprintf(b, 260, "RHSDECOMP rhs=%.6e bxlo=%.6e bxhi=%.6e bylo=%.6e byhi=%.6e src=%.6e gap=%.3e",
                     rhs_sum, bx_lo, bx_hi, by_lo, by_hi, src_sum,
                     rhs_sum - (bx_lo + bx_hi + by_lo + by_hi + src_sum));
            Util::Message(INFO, b);
        }
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    step_counter[lev]++;
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK advance_entry lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }

    // Swapping pointers (6-eq primaries -- canonical set)
    std::swap(density_old_mf[lev],         density_mf[lev]);
    std::swap(momentum_old_mf[lev],        momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev],  energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev],  energy_per_mas_mf[lev]);
    std::swap(eta_old_mf[lev],             eta_mf[lev]);
    std::swap(rho_eta0_old_mf[lev],        rho_eta0_mf[lev]);
    std::swap(rho_eta1_old_mf[lev],        rho_eta1_mf[lev]);
    std::swap(energy0_old_mf[lev],         energy0_mf[lev]);
    std::swap(energy1_old_mf[lev],         energy1_mf[lev]);
    std::swap(shell_old_mf[lev],           shell_mf[lev]);
    std::swap(cfun_old_mf[lev],            cfun_mf[lev]);

    // ------------------------------------------------------------
    // Time Integration
    //
    // 6-equation primary set (Schmidmayer 2020 eqs. 13 + 16):
    //   [0] rho_eta0       = alpha_1 rho_1
    //   [1] rho_eta1       = alpha_2 rho_2
    //   [2] momentum       = rho u (2-component)
    //   [3] energy_per_vol = rho E (redundant)
    //   [4] eta            = alpha_1
    //   [5] energy0        = alpha_1 rho_1 e_1
    //   [6] energy1        = alpha_2 rho_2 e_2
    // ------------------------------------------------------------

    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*rho_eta0_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_eta1_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(),          amrex::MakeType::make_alias, 0, AMREX_SPACEDIM);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(),    amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*eta_mf[lev].get(),               amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy0_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy1_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*shell_mf[lev].get(),             amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*cfun_mf[lev].get(),              amrex::MakeType::make_alias, 0, 1);

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*rho_eta0_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_eta1_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, AMREX_SPACEDIM);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(),amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*eta_old_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy0_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy1_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*shell_old_mf[lev].get(),         amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*cfun_old_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        // rhs_mf:      [0]=rho_eta0_rhs, [1]=rho_eta1_rhs, [2]=M_rhs, [3]=E_rhs, [4]=eta_rhs, [5]=E0_rhs, [6]=E1_rhs, [7]=shell_rhs
        // solution_mf: [0]=rho_eta0,     [1]=rho_eta1,     [2]=M,     [3]=E,     [4]=eta,     [5]=E0,     [6]=E1,     [7]=shell
        RHS(lev, time,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], rhs_mf[4], rhs_mf[5], rhs_mf[6], rhs_mf[7], rhs_mf[8],
            solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3], solution_mf[4], solution_mf[5], solution_mf[6], solution_mf[7], solution_mf[8]);
    });

    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        // Copy stage data to working arrays
        amrex::MultiFab::Copy(*rho_eta0_mf[lev],       stage_mf[0], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*rho_eta1_mf[lev],       stage_mf[1], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*momentum_mf[lev],       stage_mf[2], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], stage_mf[3], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*eta_mf[lev],            stage_mf[4], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*energy0_mf[lev],        stage_mf[5], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*energy1_mf[lev],        stage_mf[6], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*shell_mf[lev],          stage_mf[7], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*cfun_mf[lev],           stage_mf[8], 0, 0, 1,              nghost);

        // Everything from here to the end of this hook changes eta WITHOUT moving
        // material (clamping to [0,1], per-phase ghost reconstruction, stiff
        // pressure relaxation inside FillGhost4BC).  Hold the shell's areal
        // density Gamma fixed across all of it; c is rescaled to the new |grad eta|.

        // Clamp eta in domain prior to ghost fill (state can drift slightly outside [0,1])
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            });
        }

        // Fill all ghost cells (uses 6-eq primitive recovery internally)
        FillGhost4BC(lev, time);

        // Copy back
        amrex::MultiFab::Copy(stage_mf[0], *rho_eta0_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[1], *rho_eta1_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[2], *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(stage_mf[3], *energy_per_vol_mf[lev], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[4], *eta_mf[lev],            0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[5], *energy0_mf[lev],        0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[6], *energy1_mf[lev],        0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[7], *shell_mf[lev],          0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[8], *cfun_mf[lev],           0, 0, 1,              nghost);

    });


    // Zero face-flux scratch so reflux accumulation starts clean each step.
    if (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0])
    {
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            cc_fluxes[lev].mass[d]  ->setVal(0.0);
            cc_fluxes[lev].mom[d]   ->setVal(0.0);
            cc_fluxes[lev].energy[d]->setVal(0.0);
            cc_fluxes[lev].ene_k[d] ->setVal(0.0);
        }
    }

    // ------------------------------------------------------------------
    // OPERATOR CHAIN  (Schmidmayer 2017 sec. 4.2):
    //     U^{n+1} = L_relax ( L_cap ( L_hyper (U^n) ) )
    // L_hyper is the SSPRK3 advance of the hyperbolic submodel (16) -- the
    // capillary terms are gated out of RHS above.  L_cap and L_relax then run
    // ONCE per step, in that order.
    // ------------------------------------------------------------------
    defer_relax = (capillary_closure != 0);
    timeintegrator.advance(solution_old, solution_new, time, dt);
    if (capillary_closure)
    {
        defer_relax = false;
        CapillaryOperator(lev, dt);          // L_cap
        RelaxAndReinit(lev);                 // L_relax
        FillGhost4BC(lev, time + dt);        // ghosts consistent with the relaxed state
        // No copy-back: solution_new aliases momentum_mf / energy_per_vol_mf /
        // eta_mf / energy{0,1}_mf (MakeType::make_alias above), so the in-place
        // updates from L_cap and L_relax are already visible in the state.
    }

    // ------------------------------------------------------------------
    // Feed FluxRegister for reflux at coarse-fine boundaries.
    // Convert cell-centered hi-face fluxes (written by RHS) to face-centered
    // MultiFabs, then accumulate into the appropriate register.
    //
    // Register layout matches PostSubcycleReflux:
    //   [0]               rho_eta0
    //   [1]               rho_eta1
    //   [2..2+SD-1]       momentum_k
    //   [2+SD]            rho*E
    //   [3+SD..4+SD]      E_0, E_1
    // ------------------------------------------------------------------
    {
        const bool need_crse = (lev < finest_level
                                && lev + 1 < (int)flux_reg.size() && flux_reg[lev + 1]
                                && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        const bool need_fine = (lev > 0
                                && lev < (int)flux_reg.size() && flux_reg[lev]
                                && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);

        if (need_crse || need_fine)
        {
            if (need_crse) flux_reg[lev + 1]->setVal(0.0);

            constexpr int IDX_RHO_ETA0 = 0;
            constexpr int IDX_RHO_ETA1 = 1;
            constexpr int IDX_M        = 2;
            constexpr int IDX_E_VOL    = 2 + AMREX_SPACEDIM;
            constexpr int IDX_E0       = 3 + AMREX_SPACEDIM;
            constexpr int IDX_E1       = 4 + AMREX_SPACEDIM;

            for (int d = 0; d < AMREX_SPACEDIM; d++)
            {
                // FillBoundary so ghost cells inherit a neighbor's hi-face
                // flux for the cc->face conversion below.  At c/f boundaries
                // FillBoundary has nothing to copy from -- the lo-face values
                // written by RHS (commit 8a1633977 fix) persist there.
                cc_fluxes[lev].mass[d]  ->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].mom[d]   ->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].energy[d]->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].ene_k[d] ->FillBoundary(geom[lev].periodicity());

                // Build face-centered MFs on cc_fluxes' BA/dmap (surroundingNodes).
                amrex::BoxArray face_ba = cc_fluxes[lev].mass[d]->boxArray();
                face_ba.surroundingNodes(d);
                const amrex::DistributionMapping& dm_cc = cc_fluxes[lev].mass[d]->DistributionMap();
                amrex::MultiFab face_mass (face_ba, dm_cc, 2,              0);
                amrex::MultiFab face_mom  (face_ba, dm_cc, AMREX_SPACEDIM, 0);
                amrex::MultiFab face_ene  (face_ba, dm_cc, 1,              0);
                amrex::MultiFab face_ene_k(face_ba, dm_cc, 2,              0);

                // Convert cell-centered hi-face flux to face-centered:
                //   face(f, j) = cc(f - 1, j)   for d=0
                // i.e., the hi-face flux of cell f-1 IS the flux at face f.
                int nlocal = (int)cc_fluxes[lev].mass[d]->local_size();
                for (int li = 0; li < nlocal; li++)
                {
                    auto cc_m  = cc_fluxes[lev].mass  [d]->atLocalIdx(li).array();
                    auto cc_p  = cc_fluxes[lev].mom   [d]->atLocalIdx(li).array();
                    auto cc_e  = cc_fluxes[lev].energy[d]->atLocalIdx(li).array();
                    auto cc_ek = cc_fluxes[lev].ene_k [d]->atLocalIdx(li).array();

                    auto f_m  = face_mass  .atLocalIdx(li).array();
                    auto f_p  = face_mom   .atLocalIdx(li).array();
                    auto f_e  = face_ene   .atLocalIdx(li).array();
                    auto f_ek = face_ene_k .atLocalIdx(li).array();

                    const amrex::Box& fbx = face_mass.atLocalIdx(li).box();
                    const int dd = d;
                    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        const int ii = (dd == 0) ? i - 1 : i;
                        const int jj = (dd == 1) ? j - 1 : j;
                        const int kk = (dd == 2) ? k - 1 : k;   // z-face shift (3D)
                        f_m (i, j, k, 0) = cc_m (ii, jj, kk, 0);
                        f_m (i, j, k, 1) = cc_m (ii, jj, kk, 1);
                        for (int n = 0; n < AMREX_SPACEDIM; n++)
                            f_p(i, j, k, n) = cc_p(ii, jj, kk, n);
                        f_e (i, j, k)    = cc_e (ii, jj, kk);
                        f_ek(i, j, k, 0) = cc_ek(ii, jj, kk, 0);
                        f_ek(i, j, k, 1) = cc_ek(ii, jj, kk, 1);
                    });
                }

                // Face area for area-weighted accumulation: product of the cell
                // sizes in all directions EXCEPT the face normal d.
                //   2D: x-face area = dy, y-face area = dx.
                //   3D: x-face = dy*dz, y-face = dx*dz, z-face = dx*dy.
                const amrex::Real* dx_lev = geom[lev].CellSize();
                amrex::Real face_area = 1.0;
                for (int dd2 = 0; dd2 < AMREX_SPACEDIM; ++dd2)
                    if (dd2 != d) face_area *= dx_lev[dd2];
                amrex::MultiFab area_mf(face_ba, dm_cc, 1, 0);
                area_mf.setVal(face_area);

                // Sign convention: CrseInit gets -dt (subtracts coarse flux),
                // FineAdd gets +dt (adds fine flux). Reflux = (fine - coarse).
                if (need_crse)
                {
                    flux_reg[lev + 1]->CrseInit(face_mass,  area_mf, d, 0, IDX_RHO_ETA0, 2,              -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_mom,   area_mf, d, 0, IDX_M,        AMREX_SPACEDIM, -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_ene,   area_mf, d, 0, IDX_E_VOL,    1,              -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_ene_k, area_mf, d, 0, IDX_E0,       2,              -dt, amrex::FluxRegister::ADD);
                }
                if (need_fine)
                {
                    flux_reg[lev]->FineAdd(face_mass,  area_mf, d, 0, IDX_RHO_ETA0, 2,              dt);
                    flux_reg[lev]->FineAdd(face_mom,   area_mf, d, 0, IDX_M,        AMREX_SPACEDIM, dt);
                    flux_reg[lev]->FineAdd(face_ene,   area_mf, d, 0, IDX_E_VOL,    1,              dt);
                    flux_reg[lev]->FineAdd(face_ene_k, area_mf, d, 0, IDX_E0,       2,              dt);
                }
            }
        }
    }

    // ENFORCE POSITIVITY after time advance
    // [DIAG] measure clip-created mass (CLIPLEDGER, gated on relax_diag)
    Set::Scalar _clip0 = 0.0, _clip1 = 0.0;
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        if (relax_diag)
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                if (rho_eta0(i, j, k) < 0.0) _clip0 -= rho_eta0(i, j, k);
                if (rho_eta1(i, j, k) < 0.0) _clip1 -= rho_eta1(i, j, k);
            });

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // NEGATIVITY clip only -- flooring at `small` (1e-6) parks an
            // artificial discontinuity in the trace-phase partial density at
            // the contour where the tanh tail crosses 1e-6 (eta ~ 1e-9);
            // the recovery rho_k = arh_k/alpha then reads a fake 1e6 kg/m^3
            // liquid there and the biased per-phase fluxes PUMP gas mass at
            // that contour (measured: +16%/period gas-mass creation in
            // UT-RA on the pre-fix code, concentrated in the two columns
            // straddling the floor contour).  Divides downstream are all
            // DIV_FLOOR/max-protected, so zero is safe here.
            rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), 0.0);
            rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), 0.0);
        });
    }
    if (relax_diag && (_clip0 > 0.0 || _clip1 > 0.0))
    {
        static Set::Scalar _clip0_cum = 0.0, _clip1_cum = 0.0;
        static int _clip_n = 0;
        _clip0_cum += _clip0; _clip1_cum += _clip1;
        if ((_clip_n++ % 200) == 0)
        {
            char b[200];
            snprintf(b, 200, "CLIPLEDGER step-created0=%.4e step-created1=%.4e cum0=%.4e cum1=%.4e",
                     _clip0, _clip1, _clip0_cum, _clip1_cum);
            Util::Message(INFO, b);
        }
    }
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK advance_exit  lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
    // ============================================================
    // POST-INTEGRATION PRIMITIVE REFRESH.
    // ============================================================
    FillGhost4BC(lev, time + dt);
    


    // ------------------------------------------------------------
    // Interface Sharpenging
    // ------------------------------------------------------------

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

            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Advance(): Conservative Variable Check", {
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
    Set::Scalar vz_max_local = 0.0;
    Set::Scalar F_max_local = 0.0;
    Set::Scalar rho_min_local = 1e10;
    // Max LOCAL kinematic viscosity nu = mu(eta)/rho.  Reduced exactly rather
    // than estimated as mu_max/rho_min: that pairing takes the liquid's mu with
    // the gas's rho, two values that never occur in the same cell, and for the
    // standard setup (mu 0.15/0.015, rho 10/1) it overestimates nu by 10x --
    // costing a factor of ~6 in dt once the capillary limit takes over.
    Set::Scalar nu_max_local = 0.0;

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
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
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
        Set::Patch<const Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        // 6-eq per-phase internal energies (canonical primaries)
        Set::Patch<const Set::Scalar> E0_p = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1_p = energy1_mf.Patch(lev, mfi);

        // Capillary-energy closure inputs (see Hydro2.H).  NOTE: this loop
        // aliases `eta` to eta_OLD; rho E here is current, so e_cap must be
        // built from eta_new.
        const int cap_close_p = capillary_closure, marm_p = marmottant;
        const Set::Scalar sig_p = sigma, chi_p = marmottant_chi;
        const Set::Scalar Gb_p = marmottant_Gamma_buck;
        const Set::Scalar sbrk_p = marmottant_sigma_break, sigw_p = sigma;
        amrex::Array4<const Set::Scalar> const &shell_p = shell_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> const &cfun_p  = cfun_mf[lev]->const_array(mfi);

        // EMBEDDED SOLID indicator (empty Array4 when feature is off).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
        const Set::Scalar gam0_ = eos0_local.Gamma();
        const Set::Scalar pi0_  = eos0_local.P0();
        const Set::Scalar gam1_ = eos1_local.Gamma();
        const Set::Scalar pi1_  = eos1_local.P0();
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;
        // Local copies: class members are not addressable inside a GPU lambda.
        const Set::Scalar mu0_l = mu0, mu1_l = mu1, small_l = small;

        amrex::ParallelFor(bx, [=, &c_max_local, &vx_max_local, &vy_max_local, &vz_max_local, &F_max_local, &rho_min_local, &nu_max_local] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Set::Vector grad_eta = Numeric::Gradient(eta_new, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta_new, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta_new, i, j, k, 0, DX);

            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);

            etadot(i,j,k) = (eta_new(i,j,k) - eta(i,j,k)) / dt;

            // Limiting Velocity
            Set::Scalar u_limit = 1e8;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                v(i, j, k, d) = M(i, j, k, d) / (rho(i, j, k));
                v(i, j, k, d) = (v(i, j, k, d) < 0.0) ? std::max(v(i, j, k, d), -u_limit) : std::min(v(i, j, k, d), u_limit);
            }

            KE_vol(i,j,k) = 0.5 * rho(i,j,k) * (AMREX_D_TERM(v(i,j,k,0) * v(i,j,k,0), + v(i,j,k,1) * v(i,j,k,1), + v(i,j,k,2) * v(i,j,k,2)));
            KE_mas(i,j,k) = 0.5 * (AMREX_D_TERM(v(i,j,k,0) * v(i,j,k,0), + v(i,j,k,1) * v(i,j,k,1), + v(i,j,k,2) * v(i,j,k,2)));

            // Debit the capillary energy: with the closure on, rho E carries
            // e_cap, so the internal energy is rho E - KE - e_cap.
            UE_vol(i,j,k) = E_vol(i,j,k) - KE_vol(i,j,k)
                          - (cap_close_p ? CapEnergyAt(cfun_p, shell_p, i, j, k, DX,
                                                       marm_p, sig_p, chi_p, Gb_p,
                                                       sbrk_p, sigw_p) : 0.0);
            UE_vol(i, j, k) = (UE_vol(i, j, k) < 0.0) ? small : UE_vol(i, j, k);
            E_mas(i,j,k) = E_vol(i,j,k) / (rho(i,j,k) + small);
            UE_mas(i,j,k) = E_mas(i,j,k) - KE_mas(i,j,k);

            // 6-eq mixture pressure and frozen sound speed.
            // p   = alpha_1 p_1 + alpha_2 p_2          (Sch20 eq. 8)
            // c^2 = Y_1 c_1^2  + Y_2 c_2^2             (Sch20 eq. 17)
            {
                Set::Scalar a1 = std::min(std::max(eta_new(i, j, k), alpha_floor), 1.0 - alpha_floor);
                Set::Scalar a2 = 1.0 - a1;
                Set::Scalar arh0 = std::max(rho_eta0(i, j, k), small);
                Set::Scalar arh1 = std::max(rho_eta1(i, j, k), small);
                Set::Scalar r0p  = arh0 / a1;
                Set::Scalar r1p  = arh1 / a2;
                Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_p(i, j, k), a1, gam0_, pi0_, small);
                Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_p(i, j, k), a2, gam1_, pi1_, small);
                // TRACE-PHASE SLAVING: recovering a vanished phase's pressure
                // from E_k/alpha_k is ill-conditioned by gam*pi/p (~2.4e4 for
                // water at 1 bar) -- any alpha inconsistency becomes ~1e5 Pa of
                // garbage that the limiter reconstructs across the band tail
                // (the residual secular leak after the alpha-floor fix).  A
                // phase below the eta cutoff is dynamically irrelevant: slave
                // its pressure to the dominant phase (mechanical equilibrium).
                {
                    // SMOOTH slaving: a hard threshold leaves a discontinuous
                    // seam in the p_k field at the switch contour, which the
                    // limiter reads as a real pressure feature (measured: a
                    // persistent -30% dip parked at the eta = 1e-2 contour
                    // pumping gas from the bubble center into the band).
                    // Blend raw -> dominant over alpha in [SLAVE_LO, SLAVE_HI]
                    // with a C1 smoothstep: no seam, fully slaved below 1%,
                    // fully raw above 10%.
                    const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                    Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                    w0 = w0 * w0 * (3.0 - 2.0 * w0);
                    Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                    w1 = w1 * w1 * (3.0 - 2.0 * w1);
                    const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                    p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                    p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
                }
                Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(r0p, p0_loc, gam0_, pi0_, small);
                Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(r1p, p1_loc, gam1_, pi1_, small);
                Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
                Set::Scalar Y1 = 1.0 - Y0;
                press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
                a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);
                p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local); // diagnostic only
            }

            Set::Scalar f_prime = 4.0 * eta_new(i,j,k) * (eta_new(i,j,k) - 0.5) * (eta_new(i,j,k) - 1.0);
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i,j,k) = mu_chem;

            // Spalding number from the vapor MASS fraction Y = (alpha rho)_0 / rho
            // (same definition as RHS); the old eta/(1-eta) volume-fraction ratio
            // was not the Spalding B_M and clobbered the RHS diagnostic.
            Bm(i,j,k) = SpaldingBM(rho_eta0(i,j,k) / std::max(rho(i,j,k), small), Y_infinity, small);

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
        
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Advance(): Visualization", { 
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
        
            // Track CFL quantities.  Skip solid cells (their frozen state must
            // not drive the global timestep).
            const bool is_fluid = (!embedded.apply) || (phisol(i, j, k) > embedded.relax_skip);
            if (is_fluid)
            {
                c_max_local = std::max(c_max_local, a(i,j,k));
                vx_max_local = std::max(vx_max_local, std::abs(v(i,j,k,0)));
                vy_max_local = std::max(vy_max_local, std::abs(v(i,j,k,1)));
#if AMREX_SPACEDIM == 3
                vz_max_local = std::max(vz_max_local, std::abs(v(i,j,k,2)));
#endif

                Set::Scalar F_mag = sqrt(Source(i,j,k,1) * Source(i,j,k,1) +
                                         Source(i,j,k,2) * Source(i,j,k,2));
                F_max_local = std::max(F_max_local, F_mag);
                rho_min_local = std::min(rho_min_local, rho(i,j,k));
                {
                    const Set::Scalar a1n = std::min(std::max(eta_new(i,j,k), 0.0), 1.0);
                    const Set::Scalar mu_c = a1n * mu0_l + (1.0 - a1n) * mu1_l;
                    nu_max_local = std::max(nu_max_local, mu_c / (rho(i,j,k) + small_l));
                }
            }
        });
    } // end Mixed Fields loop

    // Parallel Reduction
    amrex::ParallelDescriptor::ReduceRealMax(c_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vz_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(F_max_local);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min_local);
    amrex::ParallelDescriptor::ReduceRealMax(nu_max_local);

    c_max = c_max_local;
    vx_max = vx_max_local;
    vy_max = vy_max_local;
    vz_max = vz_max_local;
    F_max = F_max_local;
    rho_min = rho_min_local;
    nu_max  = nu_max_local;

    // Computing dt for next time step on all levels
    Set::Scalar dx_min = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});

    Set::Scalar wave_speed = c_max + sqrt(AMREX_D_TERM(vx_max * vx_max, + vy_max * vy_max, + vz_max * vz_max));
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    // VISCOUS (explicit diffusion).  The stability limit for a (2d+1)-point
    // Laplacian in d dimensions is dt <= dx^2 / (2 d nu), nu = mu/rho.  The
    // previous form omitted the 2d, so with the default cfl_v = cfl = 0.3 it was
    // ~20%% too permissive in 2D (limit coefficient 0.25) and 6x too permissive
    // in 3D.  That is why mu0=mu1=0.05 blew up immediately at eps/dx=4 -- dt
    // stayed at the acoustic 6.9e-7 against a true gas-phase limit of 3.8e-7,
    // Total_Force reached -3.7e7 and dt collapsed to the 1e-12 floor.
    // rho_min with mu_max is the conservative pairing: the binding phase is the
    // one with the largest nu = mu/rho.
    Set::Scalar mu_max = std::max(mu0, mu1);
    // The diagnostic artificial viscosity mu_art = C rho a dx w has kinematic
    // value nu_art = C a dx w <= C c_max dx_min, independent of rho, so it adds
    // directly to nu.  Without this the C > ~1 runs would blow up on a stale
    // timestep rather than on the physics under test.
    // nu_max is the exact max of mu(eta)/rho over the domain (reduced above).
    // Fall back to the old conservative pairing if it was never set.
    Set::Scalar nu_phys = (nu_max > 0.0) ? nu_max : (mu_max / (rho_min + small));
    Set::Scalar nu_total = nu_phys;
    Set::Scalar dt_viscous = cfl_v * dx_min * dx_min
                           / (2.0 * AMREX_SPACEDIM * (nu_total + small));

    // CAPILLARY (Brackbill, Kothe & Zemach, JCP 100 (1992) 335).  An explicit
    // surface-tension force cannot resolve capillary waves faster than
    //     dt <= sqrt(rho_bar dx^3 / (2 pi sigma)).
    // This scales as dx^1.5 while the acoustic CFL scales as dx, so REFINING a
    // surface-tension problem eventually violates it no matter how small the
    // acoustic dt is.  Measured on the static R0=24mm bubble (inviscid,
    // sigma=7.28): dt_acoustic/dt_cap = 0.70 / 0.99 / 1.40 at eps/dx = 2 / 4 / 8,
    // tracking the observed parasitic-current growth 3365 / 9630 / 103169 /s.
    // NOTE this constraint alone does NOT cure those currents -- capping dt at
    // 0.25 dt_cap cut the growth rate only 33%% (eps/dx=4) and 9%% (eps/dx=8).
    // It is a correctness fix; the residual grid-aligned (n=4) force imbalance is
    // a separate, unsolved problem.
    // sigma is the largest tension the capillary term can apply: with
    // marmottant=1 the elastic branch is clamped at sigma_break -> sigma_water,
    // which is this same `sigma`.
    Set::Scalar dt_capillary = 1.0e30;
    if (apply_surface_tension && sigma > 0.0)
        dt_capillary = cfl_v * std::sqrt(rho_min * dx_min * dx_min * dx_min
                                       / (2.0 * Set::Constant::Pi * sigma));

    Set::Scalar a_max = F_max / (rho_min + small);
    Set::Scalar dt_force = cfl_v * sqrt(dx_min / (a_max + small));

    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    Set::Scalar dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn, dt_capillary });
    dt_max = dt_max * 0.9;

    // Debugging to report cfl constants used. Change bool to show
    if ((step_counter[lev] % 10 == 0) && false)
    {
        Util::Message(INFO, "=== CFL DIAGNOSTICS Level ", lev, " ===");
        Util::Message(INFO, "  c_max = ", c_max, " m/s");
        Util::Message(INFO, "  vx_max = ", vx_max, " m/s");
        Util::Message(INFO, "  vy_max = ", vy_max, " m/s");
        Util::Message(INFO, "  vz_max = ", vz_max, " m/s");
        Util::Message(INFO, "  dt_acoustic  = ", dt_acoustic,  " s");
        Util::Message(INFO, "  dt_viscous   = ", dt_viscous,   " s");
        Util::Message(INFO, "  dt_capillary = ", dt_capillary, " s");
        Util::Message(INFO, "  dt_max = ", dt_max, " s");
    }

    if (dynamictimestep.on)
    {
        this->DynamicTimestep_SyncTimeStep(lev, dt_max);
    }

} 


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// EMBEDDED-SOLID TARGET DERIVE //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build the per-phase embedded-solid target (density0/1_mf, energy0/1_mf,
// momentum_mf) from the SINGLE-phase solid input (density_ic, pressure_ic,
// momentum_ic), splitting mass + energy by the LOCAL eta (option 3a) and using
// the stiffened-gas EOS for the per-phase internal energies:
//   rho_eta0 = eta*rho_s ,  rho_eta1 = (1-eta)*rho_s
//   E_k      = alpha_k*(p_s + gamma_k*pi_k)/(gamma_k - 1)
// Called at IC time and after every regrid.  Requires eta_mf populated first.
void Hydro2::InitEmbeddedSolidTarget(int lev, Set::Scalar time)
{
    if (!embedded.apply) return;

    // Single-phase solid inputs.  Stage the total density into the phase-0
    // density slot and the pressure into the phase-0 energy slot as scratch,
    // then split per-cell (read into locals before overwriting).
    embedded.momentum_ic->Initialize(lev, embedded.momentum_mf, time);
    embedded.density_ic ->Initialize(lev, embedded.density0_mf, time);   // scratch: rho_s
    embedded.pressure_ic->Initialize(lev, embedded.energy0_mf, time);    // scratch: p_s

    const Set::Scalar g0 = eos0.Gamma(), pi0 = eos0.P0();
    const Set::Scalar g1 = eos1.Gamma(), pi1 = eos1.P0();
    const Set::Scalar sm = small;
    for (amrex::MFIter mfi(*embedded.density0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox(nghost);
        auto eta = eta_mf[lev]->array(mfi);
        auto d0  = embedded.density0_mf[lev]->array(mfi);
        auto d1  = embedded.density1_mf[lev]->array(mfi);
        auto e0  = embedded.energy0_mf[lev]->array(mfi);
        auto e1  = embedded.energy1_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar rho_s = d0(i, j, k);   // scratch (total solid density)
            const Set::Scalar p_s   = e0(i, j, k);   // scratch (solid pressure)
            const Set::Scalar a1 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a2 = 1.0 - a1;
            d0(i, j, k) = a1 * rho_s;
            d1(i, j, k) = a2 * rho_s;
            e0(i, j, k) = a1 * (p_s + g0 * pi0) / std::max(g0 - 1.0, sm);
            e1(i, j, k) = a2 * (p_s + g1 * pi1) / std::max(g1 - 1.0, sm);
        });
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

    // Re-allocate reflux scratch on the freshly-regridded level. Any previously-accumulated fluxes are discarded.
    AllocateRefluxScratch(lev);

    // EMBEDDED SOLID: Re-evaluate phi from its IC on the new grid rather
    // than relying on coarse-fine interpolation, which would smear the
    // sharp solid boundary and create spurious partial-solid cells.
    if (embedded.apply)
    {
        embedded.phi_ic   ->Initialize(lev, embedded.phi_mf,            regrid_time);
        embedded.phi_ic   ->Initialize(lev, embedded.phi_old_mf,        regrid_time);
        // Re-derive the per-phase solid target from the single density+pressure
        // (split by the regridded eta).
        InitEmbeddedSolidTarget(lev, regrid_time);

        // grad(phi) diagnostic (static; recomputed for the new grid).
        const Set::Scalar *DX = geom[lev].CellSize();
        embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto grad_phi = embedded.grad_phi_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector g = Numeric::Gradient(phi, i, j, k, 0, DX);
                grad_phi(i, j, k, 0) = g(0);
                grad_phi(i, j, k, 1) = g(1);
            });
        }
    }

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

    // Geometric refinement box: force refinement to max_level inside [lo,hi]
    // (targets a region, e.g. the trailing edge, without refining the whole domain)
    if (refine_box_on)
    {
        const Set::Scalar* problo = geom[lev].ProbLo();
        const Set::Scalar bl0 = refine_box_lo[0], bh0 = refine_box_hi[0];
        const Set::Scalar bl1 = refine_box_lo[1], bh1 = refine_box_hi[1];
#if AMREX_SPACEDIM == 3
        const Set::Scalar bl2 = refine_box_lo[2], bh2 = refine_box_hi[2];
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<char> const& tags = a_tags.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar x = problo[0] + (i + 0.5) * DX[0];
                Set::Scalar y = problo[1] + (j + 0.5) * DX[1];
                bool inside = (x >= bl0 && x <= bh0 && y >= bl1 && y <= bh1);
#if AMREX_SPACEDIM == 3
                Set::Scalar z = problo[2] + (k + 0.5) * DX[2];
                inside = inside && (z >= bl2 && z <= bh2);
#endif
                if (inside) tags(i, j, k) = amrex::TagBox::SET;
            });
        }
    }

    // EMBEDDED SOLID: refine the diffuse solid boundary
    if (embedded.apply)
    {
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], true); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<char> const& tags = a_tags.array(mfi);
            amrex::Array4<const Set::Scalar> const& phi = (*embedded.phi_mf[lev]).array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector grad_phi = Numeric::Gradient(phi, i, j, k, 0, DX);
                if (grad_phi.lpNorm<2>() * dr * 2 > embedded.refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
            });
        }
    }

    // Vorticity criterion for refinement
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
#if AMREX_SPACEDIM == 2
            Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, 0, DX, sten);
            if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
#else
            // 3D vorticity has 3 components; tag if any component's gradient is steep.
            for (int c = 0; c < AMREX_SPACEDIM; ++c)
            {
                Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, c, DX, sten);
                if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion)
                {
                    tags(i, j, k) = amrex::TagBox::SET;
                    break;
                }
            }
#endif
        });
    }
    
    // Gradu criterion for refinement
    for (amrex::MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& v = (*velocity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);
            if (grad_u.lpNorm<2>() * dr * 2 > gradu_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Pressure criterion for refinement
    for (amrex::MFIter mfi(*pressure_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& press = (*pressure_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_p = Numeric::Gradient(press, i, j, k, 0, DX, sten);
            if (grad_p.lpNorm<2>() * dr * 2 > p_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Density criterion for refinement
    for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& rho = (*density_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_rho = Numeric::Gradient(rho, i, j, k, 0, DX, sten);
            if (grad_rho.lpNorm<2>() * dr * 2 > rho_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

}



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// INTERFACE SHARPENING /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// REINITIALIZE SIGNED DISTANCE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


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
/// ============================================================================
void Hydro2::FillGhost4BC(int lev, Set::Scalar time)
{
    BL_PROFILE("Integrator::Hydro2::FillGhost4BC");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Ensure eta_bc has a valid geometry for FillBoundary calls.
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
            if (verbose) Util::Message(INFO, "FillGhost4BC: Using NSCBC with 2 ghost cells");
        }
        else if (nghost == 4)
        {
            if (verbose) Util::Message(INFO, "FillGhost4BC: Using NSCBC4 with 4 ghost cells");
        }
    }
    else
    {
        effective_nghost = 2;
    }

    // Fill Patches for regridding -- AMR coarse-fine ghost interpolation
    // for the FULL 6-eq conservative state.  Missing energy0_mf or
    // energy1_mf here causes the flux-loop make_state to pick up STALE
    // (unfilled) per-phase E_k ghost values at coarse-fine interfaces;
    // when a wave reaches a fine patch whose ghosts sit at the corner
    // of (coarse-fine) and (physical-boundary), the BC fills neighbor
    // ghosts on top of those garbage E_k cells -> garbage flux -> NaN.
    if (lev > 0)
    {
        FillPatch(lev, time, rho_eta0_mf,        *rho_eta0_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, rho_eta1_mf,        *rho_eta1_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, momentum_mf,        *momentum_mf[lev],       *momentum_bc, 0);
        FillPatch(lev, time, energy_per_vol_mf,  *energy_per_vol_mf[lev], *energy_bc,   0);
        FillPatch(lev, time, eta_mf,             *eta_mf[lev],            *eta_bc,      0);
        FillPatch(lev, time, shell_mf,           *shell_mf[lev],          *eta_bc,      0);
        FillPatch(lev, time, cfun_mf,            *cfun_mf[lev],           *eta_bc,      0);
        FillPatch(lev, time, energy0_mf,         *energy0_mf[lev],        *energy_bc,   0);
        FillPatch(lev, time, energy1_mf,         *energy1_mf[lev],        *energy_bc,   0);
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
    // STEP 2b: EMBEDDED SOLID BOUNDARY auto wall + ghost fill.
    // ------------------------------------------------------------
    // Fill the static indicator's ghosts, then (only when solid.brinkman was
    // not provided) apply a dt-independent, unconditionally-stable momentum-
    // only projection toward the solid velocity, graded by phi.  When
    // solid.brinkman > 0 the explicit RHS penalty is the wall and this is
    // skipped.  This runs BEFORE the STEP 4 primitive recovery so the EOS sees
    // a sensible (finite p, c) state inside the solid, and it is IDEMPOTENT --
    // safe to call from RHS, the post-stage hook, and the post-integration
    // refresh.
    if (embedded.apply)
    {
        embedded.phi_bc->define(geom[lev]);
        FillBoundariesWithBC(lev, time, embedded.phi_bc, { embedded.phi_mf[lev].get() });

        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto M        = momentum_mf[lev]->array(mfi);
            auto E        = energy_per_vol_mf[lev]->array(mfi);
            auto E0_arr   = energy0_mf[lev]->array(mfi);
            auto E1_arr   = energy1_mf[lev]->array(mfi);
            auto rho      = density_mf[lev]->array(mfi);
            auto s_M      = embedded.momentum_mf[lev]->array(mfi);

            const Set::Scalar small_  = small;
            const Set::Scalar brink_  = embedded.brinkman;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // Auto wall (used only when solid.brinkman not provided): a
                // dt-independent, unconditionally-stable momentum-only
                // projection toward the solid velocity, graded by phi:
                //   M <- phi*M + (1-phi)*M_solid  (= the rest-init blend applied
                // every step).  When solid.brinkman > 0 the explicit RHS penalty
                // is the wall instead and this is skipped.  Mass/per-phase
                // energy keep their evolved (full-flux) values; only the
                // momentum is projected and the redundant total energy E is
                // recomputed consistently (rho already = rho_eta0+rho_eta1 from
                // STEP 2 above).
                if (brink_ <= 0.0)
                {
                    const Set::Scalar w = std::min(std::max(phi(i, j, k), 0.0), 1.0);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        M(i, j, k, d) = w * M(i, j, k, d) + (1.0 - w) * s_M(i, j, k, d);
                    const Set::Scalar KE = 0.5 * (AMREX_D_TERM(M(i, j, k, 0) * M(i, j, k, 0),
                                                             + M(i, j, k, 1) * M(i, j, k, 1),
                                                             + M(i, j, k, 2) * M(i, j, k, 2))) / std::max(rho(i, j, k), small_);
                    E(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k) + KE;
                }
            });
        }
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
    // STEP 4: Compute primitives in DOMAIN (6-equation recovery).
    //
    // From the canonical primaries (eta, (alpha rho)_k, M, rho E, E_k) we
    // derive per-phase rho_k, e_k, p_k, c_k and the mixture rho, u, p, c.
    //   - p_k uses Sau09 eq. (II.3) inverted (PhasicPressureFromEnergy).
    //   - p   uses Schmidmayer 2020 eq. (8): p = alpha_1 p_1 + alpha_2 p_2.
    //   - c   uses Schmidmayer 2020 eq. (17): c^2 = Y_1 c_1^2 + Y_2 c_2^2.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY

        auto rho       = density_mf[lev]->array(mfi);
        auto eta       = eta_mf[lev]->array(mfi);
        auto arh0_arr  = rho_eta0_mf[lev]->array(mfi);
        auto arh1_arr  = rho_eta1_mf[lev]->array(mfi);
        auto M         = momentum_mf[lev]->array(mfi);
        auto E         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_arr    = energy0_mf[lev]->array(mfi);
        auto E1_arr    = energy1_mf[lev]->array(mfi);
        auto v         = velocity_mf[lev]->array(mfi);
        auto press     = pressure_mf[lev]->array(mfi);
        auto T         = T_mf[lev]->array(mfi);
        auto a         = a_mf[lev]->array(mfi);
        auto gamma     = gamma_mf[lev]->array(mfi);
        auto p0_eff    = p0_mf[lev]->array(mfi);
        auto UE        = UE_per_vol_mf[lev]->array(mfi);
        auto KE        = KE_per_vol_mf[lev]->array(mfi);
        auto rho0_arr  = density0_mf[lev]->array(mfi);
        auto rho1_arr  = density1_mf[lev]->array(mfi);
        auto p0_arr    = pressure0_mf[lev]->array(mfi);
        auto p1_arr    = pressure1_mf[lev]->array(mfi);
        auto v0_arr    = velocity0_mf[lev]->array(mfi);
        auto v1_arr    = velocity1_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
        const Set::Scalar gam0 = eos0_local.Gamma();
        const Set::Scalar pi0_ = eos0_local.P0();
        const Set::Scalar gam1 = eos1_local.Gamma();
        const Set::Scalar pi1_ = eos1_local.P0();
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Volume fraction with floor.
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            // Phase masses (canonical).
            Set::Scalar arh0 = std::max(arh0_arr(i, j, k), small);
            Set::Scalar arh1 = std::max(arh1_arr(i, j, k), small);

            // Mixture density (Sch20 eq. 8 RHS).
            rho(i, j, k) = arh0 + arh1;

            // Velocity from mixture momentum.
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            // Kinetic energy.
            KE(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                          + v(i, j, k, 1) * v(i, j, k, 1),
                                                          + v(i, j, k, 2) * v(i, j, k, 2)));

            // Pure-phase densities for per-phase EOS evaluation.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;

            // Per-phase pressures from canonical (alpha rho e)_k (Sau09 II.3).
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);
            // TRACE-PHASE SLAVING (see flux-loop recovery for rationale):
            // a vanished phase's E_k/alpha_k pressure is gam*pi/p-amplified
            // noise; slave it to the dominant phase below the eta cutoff.
            {
                // SMOOTH slaving (see flux-loop recovery block for rationale).
                const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w0 = w0 * w0 * (3.0 - 2.0 * w0);
                Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w1 = w1 * w1 * (3.0 - 2.0 * w1);
                const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
            }

            // Per-phase sound speeds.
            Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho0_pure, p0_loc, gam0, pi0_, small);
            Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho1_pure, p1_loc, gam1, pi1_, small);

            // Mass fractions.
            Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
            Set::Scalar Y1 = 1.0 - Y0;

            // Mixture pressure (Sch20 eq. 8) and frozen sound speed (Sch20 eq. 17).
            press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
            a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);

            // Per-phase diagnostic primitives (plotfile).
            rho0_arr(i, j, k) = rho0_pure;
            rho1_arr(i, j, k) = rho1_pure;
            p0_arr(i, j, k)   = p0_loc;
            p1_arr(i, j, k)   = p1_loc;
            v0_arr(i, j, k, 0) = v(i, j, k, 0);   // mechanical equilibrium (one velocity)
            v0_arr(i, j, k, 1) = v(i, j, k, 1);
            v1_arr(i, j, k, 0) = v(i, j, k, 0);
            v1_arr(i, j, k, 1) = v(i, j, k, 1);

            // Internal energy (sum of canonical per-phase energies).
            UE(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);

            // Diagnostic gamma / p0_eff (NOT used by hyperbolic step).
            gamma(i, j, k)  = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);
            T(i, j, k)      = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), a1, eos0_local, eos1_local, pref);
        });
    }

    // ============================================================
    // 6-eq STIFF PRESSURE RELAXATION + REINIT (Sch20 Sec.3.3 / Sau09 Sec.3.5)
    // Primitives are computed first (STEP 4 above), then the Newton enforces 
    // p_0 = p_1 on thecanonical primaries (eta, E_0, E_1) before any source/flux
    // ============================================================
    // MERGE (AcousticBC): mass-tracking diagnostic, gated on relax_diag.  This
    // supersedes the unconditional pre-relax Message that was here on the
    // Marmottant side -- that one printed every call and swamped the logs.
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK fillghost_in  lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
    // MERGE (Marmottant): Schmidmayer 2017 splitting -- U^{n+1} = L_relax L_cap
    // L_hyper(U^n), so relaxation runs ONCE per step after the capillary
    // operator, not between RK stages.  defer_relax is false unless the split
    // scheme is active, so the AcousticBC behaviour is unchanged by default.
    if (!defer_relax) RelaxAndReinit(lev);

    // ------------------------------------------------------------
    // STEP 5: Fill CONSERVATIVE ghost cells (rho, M, E)
    // ------------------------------------------------------------
    if (use_nscbc)
    {
        // ====================================================================
        // NSCBC PATH: Use characteristic-based boundary conditions
        // ====================================================================

        const int ib_lo = geom[lev].Domain().smallEnd(0);
        const int ib_hi = geom[lev].Domain().bigEnd(0);
        const int jb_lo = geom[lev].Domain().smallEnd(1);
        const int jb_hi = geom[lev].Domain().bigEnd(1);
        const bool x_periodic = geom[lev].isPeriodic(0);
        const bool y_periodic = geom[lev].isPeriodic(1);
#if AMREX_SPACEDIM == 3
        const int kb_lo = geom[lev].Domain().smallEnd(2);
        const int kb_hi = geom[lev].Domain().bigEnd(2);
        const bool z_periodic = geom[lev].isPeriodic(2);
#endif

        // --------------------------------------------------------------------
        // PRE-NSCBC: refresh periodic ghosts on ALL 6-eq conservative
        // primaries.  Without this, periodic ghosts of (alpha rho)_k,
        // momentum, rho E, E_0, E_1 carry stale values from the previous
        // timestep (the post-stage MultiFab::Copy copies the integrator's
        // stage ghosts which are never advanced).  The downstream RHS reads
        // ghost cells at y-edge interior cells to build Riemann states; if
        // those are stale, the flux divergence is wrong, the error feeds
        // back into the domain, and after O(60) steps the state explodes.
        //
        // FillBoundariesWithBC unconditionally runs mf->FillBoundary(
        // periodicity) on each MF; the bc->FillBoundary call is a no-op
        // for nscbc_outflow / nscbc_inflow faces (Expression doesn't handle
        // those types) — so this DOES fill periodic ghosts and is harmless
        // for the NSCBC faces (NSCBC overwrites those further down).
        // --------------------------------------------------------------------
        FillBoundariesWithBC(lev, time, density_bc, {
            rho_eta0_mf[lev].get(),
            rho_eta1_mf[lev].get()
        });
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        FillBoundariesWithBC(lev, time, energy_bc, {
            energy_per_vol_mf[lev].get(),
            energy0_mf[lev].get(),
            energy1_mf[lev].get()
        });

        // --------------------------------------------------------------------
        // PRE-NSCBC: extrapolate eta into NSCBC ghost cells by constant
        // copy from the boundary cell.  Periodic-direction ghosts are
        // skipped because they are already filled by FillBoundary's
        // periodic copy in STEP 3 above.
        // --------------------------------------------------------------------
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto eta = eta_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const bool x_outside = (i < ib_lo) || (i > ib_hi);
                const bool y_outside = (j < jb_lo) || (j > jb_hi);
#if AMREX_SPACEDIM == 3
                const bool z_outside = (k < kb_lo) || (k > kb_hi);
#else
                const bool z_outside = false;
#endif
                if (!x_outside && !y_outside && !z_outside) return;  // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone
#if AMREX_SPACEDIM == 3
                if (z_outside && z_periodic)  return;                // periodic — leave alone
                const int kb = std::min(std::max(k, kb_lo), kb_hi);
#else
                const int kb = k;
#endif
                const int ib = std::min(std::max(i, ib_lo), ib_hi);
                const int jb = std::min(std::max(j, jb_lo), jb_hi);
                eta(i, j, k) = eta(ib, jb, kb);
            });
        }

        // --------------------------------------------------------------------
        // Build mixture rho_total in domain + ghosts.  NSCBC's LODI uses
        // ONLY mixture quantities; the per-phase fields are re-derived after.
        // --------------------------------------------------------------------
        amrex::MultiFab rho_total(rho_eta0_mf[lev]->boxArray(),
                                  rho_eta0_mf[lev]->DistributionMap(),
                                  1,
                                  nghost);

        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.growntilebox(nghost);
            auto rho  = rho_total.array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }

        // Working copies for momentum and total energy (NSCBC modifies in-place).
        amrex::MultiFab M_copy(momentum_mf[lev]->boxArray(),
                               momentum_mf[lev]->DistributionMap(),
                               AMREX_SPACEDIM,
                               nghost);
        amrex::MultiFab E_copy(energy_per_vol_mf[lev]->boxArray(),
                               energy_per_vol_mf[lev]->DistributionMap(),
                               1,
                               nghost);
        amrex::MultiFab::Copy(M_copy, *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(E_copy, *energy_per_vol_mf[lev], 0, 0, 1,              nghost);

        // --------------------------------------------------------------------
        // Run NSCBC LODI.  Uses the mixture-effective EOS internally (Tammann
        // form, validated against the 5-equation code path).  Modifies
        // rho_total, M_copy, E_copy in ghosts; writes gamma_mf, p0_mf,
        // pressure_mf in ghosts as well.
        // --------------------------------------------------------------------
        if (nghost == 2 && nscbc_bc != nullptr)
        {
            nscbc_bc->FillBoundary(rho_total, M_copy, E_copy,
                                   *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev],
                                   eos0, eos1, geom[lev], time, pref);
        }
        else if (nghost == 4 && nscbc4_bc != nullptr)
        {
            nscbc4_bc->FillBoundary(rho_total, M_copy, E_copy,
                                    *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev],
                                    eos0, eos1, geom[lev], time, pref);
        }

        amrex::MultiFab::Copy(*momentum_mf[lev],       M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_copy, 0, 0, 1,              nghost);

        // --------------------------------------------------------------------
        // POST-NSCBC partition: build the 6-eq per-phase ghosts from the
        // NSCBC-filled mixture and the (pre-extrapolated) eta.  Identical
        // ideology to the 5-equation form — split mass and internal energy
        // by alpha alone, with mechanical-equilibrium pressure:
        //
        //   (alpha rho)_0_ghost = rho_total_ghost * eta_ghost
        //   (alpha rho)_1_ghost = rho_total_ghost * (1 - eta_ghost)
        //   E_0_ghost           = eta_ghost     * (p_g + gamma_0 pi_0)/(gamma_0 - 1)
        //   E_1_ghost           = (1-eta_ghost) * (p_g + gamma_1 pi_1)/(gamma_1 - 1)
        //
        // The E_k formulas are Sau09 eq. III.5 / Sch20 eq. 13 init evaluated
        // at the NSCBC-set ghost pressure and the boundary-extrapolated eta.
        // For periodic-direction ghosts the fields are already correct via
        // FillBoundary periodicity — skip.
        // --------------------------------------------------------------------
        const Set::Scalar g0_ns = eos0.Gamma();
        const Set::Scalar pi0_ns = eos0.P0();
        const Set::Scalar g1_ns = eos1.Gamma();
        const Set::Scalar pi1_ns = eos1.P0();
        const Set::Scalar small_ns = small;

        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho   = rho_total.array(mfi);
            auto eta   = eta_mf[lev]->array(mfi);
            auto rho0  = rho_eta0_mf[lev]->array(mfi);
            auto rho1  = rho_eta1_mf[lev]->array(mfi);
            auto press = pressure_mf[lev]->array(mfi);
            auto E0    = energy0_mf[lev]->array(mfi);
            auto E1    = energy1_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const bool x_outside = (i < ib_lo) || (i > ib_hi);
                const bool y_outside = (j < jb_lo) || (j > jb_hi);
#if AMREX_SPACEDIM == 3
                const bool z_outside = (k < kb_lo) || (k > kb_hi);
#else
                const bool z_outside = false;
#endif
                if (!x_outside && !y_outside && !z_outside) return;  // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone
#if AMREX_SPACEDIM == 3
                if (z_outside && z_periodic)  return;                // periodic — leave alone
#endif

                const Set::Scalar a1 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2 = 1.0 - a1;
                const Set::Scalar rho_g = std::max(rho(i, j, k), small_ns);
                const Set::Scalar p_g   = std::max(press(i, j, k), small_ns);

                // Per-phase densities: preserve the (BC-filled) per-phase RATIO and
                // rescale to the LODI-updated mixture rho_g, instead of re-splitting
                // rho_g by eta -- the latter mis-assigns liquid mass to the gas at a
                // mixed ghost (alpha_g*alpha_l*(rho_l-rho_g)); pure-phase ghosts are
                // identical either way.  Energies stay the equilibrium re-derivation.
                const Set::Scalar r0   = std::max(rho0(i, j, k), 0.0);
                const Set::Scalar r1   = std::max(rho1(i, j, k), 0.0);
                const Set::Scalar scale = rho_g / std::max(r0 + r1, small_ns);
                rho0(i, j, k) = r0 * scale;
                rho1(i, j, k) = r1 * scale;
                E0(i, j, k)   = a1 * (p_g + g0_ns * pi0_ns) / std::max(g0_ns - 1.0, small_ns);
                E1(i, j, k)   = a2 * (p_g + g1_ns * pi1_ns) / std::max(g1_ns - 1.0, small_ns);
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
        // Per-phase density GHOSTS via their own BC.  Do NOT re-partition the mixture
        // density by eta: rho_eta_k = rho_mix*(eta | 1-eta) is exact only at PURE-phase
        // cells.  At a mixed-eta INTERFACE cell it injects alpha_g*alpha_l*(rho_l-rho_g)
        // (~250 for water/air at 50/50) of LIQUID mass into the GAS partial density --
        // and growntilebox(nghost) clobbered the VALID, flux-conserved cells too, so it
        // was a per-step gas-mass SOURCE at the interface that destroyed bubble collapse
        // (gas mass blew up ~119x; see tests/FlowRayleighPlesset mass-conservation check).
        // rho_eta0/1 are conserved -- fill only their ghosts, directly, per phase.
        FillBoundariesWithBC(lev, time, density_bc, {
            rho_eta0_mf[lev].get(), rho_eta1_mf[lev].get()
        });
        // Mixture density is the consistent sum of the partial densities.
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho  = density_mf[lev]->array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);
            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }
        // Mixture momentum.
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        if (!bc_primitive)
        {
            // CONSERVATIVE PATH: redundant total energy AND per-phase internal
            // energies (6-eq primaries -- Sch20 eq. 13 last two rows + eq. 16).
            FillBoundariesWithBC(lev, time, energy_bc, {
                energy_per_vol_mf[lev].get(),
                energy0_mf[lev].get(),
                energy1_mf[lev].get()
            });
        }
        else
        {
            // PRIMITIVE PATH: prescribe pressure at the boundary, then
            // reconstruct per-phase internal energies from (alpha, p) via the
            // EOS -- identical formula to the NSCBC4 branch above. eta ghosts
            // are already filled (STEP 3); rho/momentum ghosts from above.
            // rho E (energy_per_vol) is rebuilt in STEP 7 as E0+E1+KE.
            FillBoundariesWithBC(lev, time, pressure_bc, {
                pressure_mf[lev].get()
            });

            const Set::Scalar g0p = eos0.Gamma(), pi0p = eos0.P0();
            const Set::Scalar g1p = eos1.Gamma(), pi1p = eos1.P0();
            const Set::Scalar smp = small;
            for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &ghostbox = mfi.growntilebox(nghost);
                auto eta   = eta_mf[lev]->array(mfi);
                auto press = pressure_mf[lev]->array(mfi);
                auto E0    = energy0_mf[lev]->array(mfi);
                auto E1    = energy1_mf[lev]->array(mfi);
                amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const Set::Scalar a1  = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                    const Set::Scalar a2  = 1.0 - a1;
                    const Set::Scalar p_g = std::max(press(i, j, k), smp);
                    E0(i, j, k) = a1 * (p_g + g0p * pi0p) / std::max(g0p - 1.0, smp);
                    E1(i, j, k) = a2 * (p_g + g1p * pi1p) / std::max(g1p - 1.0, smp);
                });
            }
        }

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
                auto eta_zg = eta_mf[lev]->array(mfi);
                auto E0_zg = energy0_mf[lev]->array(mfi);
                auto E1_zg = energy1_mf[lev]->array(mfi);

                amrex::ParallelFor(ghostNbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // If in outer ghost layers (beyond layer 2)
                    if (!ghostEffbox.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                    {
                        // Find nearest layer-2 ghost cell and copy its value
                        int i_copy = i;
                        int j_copy = j;
                        int k_copy = k;

                        // Clamp to layer-2 boundary
                        if (i < ghostEffbox.smallEnd(0))
                            i_copy = ghostEffbox.smallEnd(0);
                        if (i > ghostEffbox.bigEnd(0))
                            i_copy = ghostEffbox.bigEnd(0);
                        if (j < ghostEffbox.smallEnd(1))
                            j_copy = ghostEffbox.smallEnd(1);
                        if (j > ghostEffbox.bigEnd(1))
                            j_copy = ghostEffbox.bigEnd(1);
#if AMREX_SPACEDIM == 3
                        if (k < ghostEffbox.smallEnd(2))
                            k_copy = ghostEffbox.smallEnd(2);
                        if (k > ghostEffbox.bigEnd(2))
                            k_copy = ghostEffbox.bigEnd(2);
#endif

                        // Zero-gradient extrapolation (copy from layer 2).
                        // MUST cover the FULL 6-eq state: this block originally
                        // copied only {rho_eta, M, E}, leaving eta/E0/E1 in
                        // ghost layers 3-4 FROZEN AT t=0 while the copied
                        // fields evolved.  WENO5's boundary-face stencil reads
                        // layer 3; ToPrimitive on the mismatched state (gam*pi
                        // amplified) corrupted the boundary Riemann solve until
                        // the OUTFLOW face pumped mass INTO the domain (UT_ADV:
                        // xlo boundary flux drifted from the exact -8.0 to
                        // +11.3 while adjacent cells stayed healthy).  Third
                        // defect fixed in this block (after the 3D k-clamp and
                        // the missing z-momentum).
                        rho0(i, j, k) = rho0(i_copy, j_copy, k_copy);
                        rho1(i, j, k) = rho1(i_copy, j_copy, k_copy);
                        M(i, j, k, 0) = M(i_copy, j_copy, k_copy, 0);
                        M(i, j, k, 1) = M(i_copy, j_copy, k_copy, 1);
#if AMREX_SPACEDIM == 3
                        M(i, j, k, 2) = M(i_copy, j_copy, k_copy, 2);
#endif
                        E(i, j, k) = E(i_copy, j_copy, k_copy);
                        eta_zg(i, j, k) = eta_zg(i_copy, j_copy, k_copy);
                        E0_zg(i, j, k) = E0_zg(i_copy, j_copy, k_copy);
                        E1_zg(i, j, k) = E1_zg(i_copy, j_copy, k_copy);
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
        auto M = momentum_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            // VACUUM-CONSISTENT MOMENTUM: a cell whose density sits at/near
            // the floor carries no physical momentum -- retaining finite M
            // there manufactures u = M/rho_floor ~ 1e8 velocities (CFL
            // collapse to dt ~ 1e-12, then NaN; the divide-floor disease,
            // 4th appearance -- observed as the focal-corner vacuum crash
            // in the driven quadrant/octant).  1e-5 kg/m^3 is 4+ orders
            // below any physical state in the water/air system.
            if (rho(i, j, k) < 1.0e-5)
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    M(i, j, k, d) = 0.0;
        });
    }

    // ------------------------------------------------------------
    // STEP 7: Compute primitives in GHOST CELLS (6-eq primitive recovery).
    //
    // Also OVERWRITE the BC-supplied rho E in ghosts with the
    // thermodynamically consistent value  rho E = E0 + E1 + KE.
    // This avoids the prior BC-inconsistency bug class where a
    // user-supplied rho E in ghosts disagrees with the per-phase
    // energies, leading the reinit step to overwrite p with garbage.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        // Capillary-energy closure inputs (see Hydro2.H).
        const int cap_close_g = capillary_closure, marm_g = marmottant;
        const Set::Scalar sig_g = sigma, chi_g = marmottant_chi;
        const Set::Scalar Gb_g = marmottant_Gamma_buck;
        const Set::Scalar sbrk_g = marmottant_sigma_break, sigw_g = sigma;
        auto shell_g   = shell_mf[lev]->const_array(mfi);
        auto cfun_g    = cfun_mf[lev]->const_array(mfi);

        auto rho       = density_mf[lev]->array(mfi);
        auto eta       = eta_mf[lev]->array(mfi);
        auto arh0_arr  = rho_eta0_mf[lev]->array(mfi);
        auto arh1_arr  = rho_eta1_mf[lev]->array(mfi);
        auto M         = momentum_mf[lev]->array(mfi);
        auto E         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_arr    = energy0_mf[lev]->array(mfi);
        auto E1_arr    = energy1_mf[lev]->array(mfi);
        auto v         = velocity_mf[lev]->array(mfi);
        auto press     = pressure_mf[lev]->array(mfi);
        auto T         = T_mf[lev]->array(mfi);
        auto a         = a_mf[lev]->array(mfi);
        auto gamma     = gamma_mf[lev]->array(mfi);
        auto p0_eff    = p0_mf[lev]->array(mfi);
        auto UE        = UE_per_vol_mf[lev]->array(mfi);
        auto KE        = KE_per_vol_mf[lev]->array(mfi);
        auto rho0_arr  = density0_mf[lev]->array(mfi);
        auto rho1_arr  = density1_mf[lev]->array(mfi);
        auto p0_arr    = pressure0_mf[lev]->array(mfi);
        auto p1_arr    = pressure1_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
        const Set::Scalar gam0 = eos0_local.Gamma();
        const Set::Scalar pi0_ = eos0_local.P0();
        const Set::Scalar gam1 = eos1_local.Gamma();
        const Set::Scalar pi1_ = eos1_local.P0();
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            Set::Scalar arh0 = std::max(arh0_arr(i, j, k), small);
            Set::Scalar arh1 = std::max(arh1_arr(i, j, k), small);
            rho(i, j, k) = arh0 + arh1;

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            KE(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                          + v(i, j, k, 1) * v(i, j, k, 1),
                                                          + v(i, j, k, 2) * v(i, j, k, 2)));

            // Per-phase pure densities, pressures, and sound speeds.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);
            // TRACE-PHASE SLAVING (see flux-loop recovery for rationale).
            {
                // SMOOTH slaving (see flux-loop recovery block for rationale).
                const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w0 = w0 * w0 * (3.0 - 2.0 * w0);
                Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w1 = w1 * w1 * (3.0 - 2.0 * w1);
                const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
            }
            Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho0_pure, p0_loc, gam0, pi0_, small);
            Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho1_pure, p1_loc, gam1, pi1_, small);

            Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
            Set::Scalar Y1 = 1.0 - Y0;

            press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
            a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);

            // Total energy in the ghost CONSISTENT with the per-phase energies:
            // rho E = sum_k (alpha rho e)_k + 0.5 rho |u|^2  (Sch20 eq. 16).
            // This overwrite is the BC-consistency guarantee for the 6-eq model.
            UE(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);
            E(i, j, k)  = UE(i, j, k) + KE(i, j, k)
                        + (cap_close_g ? CapEnergyAt(cfun_g, shell_g, i, j, k, DX,
                                                     marm_g, sig_g, chi_g, Gb_g,
                                                     sbrk_g, sigw_g) : 0.0);

            // Diagnostic per-phase primitives.
            rho0_arr(i, j, k) = rho0_pure;
            rho1_arr(i, j, k) = rho1_pure;
            p0_arr(i, j, k)   = p0_loc;
            p1_arr(i, j, k)   = p1_loc;

            // Diagnostic mixture gamma / p0 / T (NOT used by hyperbolic step).
            gamma(i, j, k)  = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);
            T(i, j, k)      = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), a1, eos0_local, eos1_local, pref);
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

        const amrex::Box vbx8 = mfi.validbox();
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Check for NaN in any field
            bool has_nan = !std::isfinite(rho_eta0(i, j, k)) || !std::isfinite(rho_eta1(i, j, k)) || !std::isfinite(rho(i, j, k)) || !std::isfinite(eta(i, j, k)) || !std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1)) || !std::isfinite(E(i, j, k)) || !std::isfinite(press(i, j, k)) || !std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1)) || !std::isfinite(T(i, j, k)) || !std::isfinite(a(i, j, k)) || !std::isfinite(gamma(i, j, k)) || !std::isfinite(p0_eff(i, j, k));

            if (has_nan)
            {
                // [DIAG] STEP-8 repair ledger: this overwrite REPLACES the
                // cell's per-phase MASSES with a neighbor's -- silent mass
                // creation/destruction if it ever fires on valid cells.
                {
                    char b[220];
                    snprintf(b, 220, "STEP8REPAIR lev=%d i=%d j=%d valid=%d re0=%.3e re1=%.3e "
                             "finite[re0=%d re1=%d rho=%d p=%d v=%d T=%d a=%d gam=%d]",
                             lev, i, j,
                             (int)vbx8.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))),
                             (double)rho_eta0(i, j, k), (double)rho_eta1(i, j, k),
                             (int)std::isfinite(rho_eta0(i, j, k)), (int)std::isfinite(rho_eta1(i, j, k)),
                             (int)std::isfinite(rho(i, j, k)), (int)std::isfinite(press(i, j, k)),
                             (int)std::isfinite(v(i, j, k, 0)), (int)std::isfinite(T(i, j, k)),
                             (int)std::isfinite(a(i, j, k)), (int)std::isfinite(gamma(i, j, k)));
                    Util::Message(INFO, b);
                }
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

        // [DIAG] STEP-9 clip ledger on VALID cells (the uninstrumented mass
        // source: this clip runs EVERY RK STAGE via post_stage FillGhost and
        // growntilebox INCLUDES the valid region).  Gated on relax_diag.
        if (relax_diag)
        {
            const amrex::Box &vbx9 = mfi.validbox();
            Set::Scalar c0 = 0.0, c1 = 0.0;
            amrex::LoopOnCpu(vbx9, [&](int i, int j, int k) {
                if (rho_eta0(i, j, k) < 0.0) c0 -= rho_eta0(i, j, k);
                if (rho_eta1(i, j, k) < 0.0) c1 -= rho_eta1(i, j, k);
            });
            if (c0 > 0.0 || c1 > 0.0)
            {
                static Set::Scalar cum0 = 0.0, cum1 = 0.0;
                static int nrep = 0;
                cum0 += c0; cum1 += c1;
                if ((nrep++ % 500) == 0)
                {
                    char b[200];
                    snprintf(b, 200, "STEP9CLIP valid-created0=%.4e valid-created1=%.4e cum0=%.4e cum1=%.4e",
                             c0, c1, cum0, cum1);
                    Util::Message(INFO, b);
                }
            }
        }
        // FIX (gated on clip_ghost_only): restrict the STEP-9 positivity clip
        // to GHOST cells.  Valid-cell mid-stage undershoots are transient RK
        // stage states -- zeroing them CREATES phase mass every stage (the
        // quadrant/octant driven-bubble mass instability); the final combined
        // state is protected by the post-advance clip, which is measured and
        // near-silent.
        const amrex::Box vbx_keep = mfi.validbox();
        const int ghost_only = clip_ghost_only;
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const bool in_valid = vbx_keep.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));
            // Enforce rho = rho_eta0 + rho_eta1
            Set::Scalar rho_calc = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            rho(i, j, k) = rho_calc;

            // Enforce positivity (rho_eta: NEGATIVITY clip only; and with
            // clip_ghost_only=1, ghosts only -- valid-cell stage states pass
            // through untouched to preserve per-phase mass conservation)
            rho(i, j, k) = std::max(rho(i, j, k), small);
            if (!ghost_only || !in_valid)
            {
                rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), 0.0);
                rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), 0.0);
            }
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

    // ================================================================
    // REFLECT-MIRROR AUDIT (temporary diagnostic, gated on relax_diag):
    // at xlo/ylo domain faces with REFLECT BCs, ghost layer -g must be
    // the exact mirror of interior layer +g-1: even fields equal, normal
    // momentum negated.  Any asymmetry here seeds a nonzero mass flux
    // through the symmetry plane (quadrant/octant gas-mass instability).
    // ================================================================
    if (relax_diag)
    {
        const amrex::Box &dom = geom[lev].Domain();
        Set::Scalar a_eta = 0, a_r0 = 0, a_r1 = 0, a_Mn = 0, a_E0 = 0, a_E1 = 0;
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &vbx = mfi.validbox();
            if (vbx.smallEnd(0) != dom.smallEnd(0)) continue;
            auto ge = eta_mf[lev]->array(mfi);
            auto g0 = rho_eta0_mf[lev]->array(mfi);
            auto g1 = rho_eta1_mf[lev]->array(mfi);
            auto gM = momentum_mf[lev]->array(mfi);
            auto gE0 = energy0_mf[lev]->array(mfi);
            auto gE1 = energy1_mf[lev]->array(mfi);
            const int klo = (AMREX_SPACEDIM == 3) ? vbx.smallEnd(2) : 0;
            const int khi = (AMREX_SPACEDIM == 3) ? vbx.bigEnd(2) : 0;
            for (int k = klo; k <= khi; ++k)
                for (int j = vbx.smallEnd(1); j <= vbx.bigEnd(1); ++j)
                    for (int g = 1; g <= nghost; ++g)
                    {
                        const int ig = dom.smallEnd(0) - g;      // ghost
                        const int im = dom.smallEnd(0) + g - 1;  // mirror
                        a_eta = std::max(a_eta, std::abs(ge(ig,j,k) - ge(im,j,k)));
                        a_r0  = std::max(a_r0,  std::abs(g0(ig,j,k) - g0(im,j,k)));
                        a_r1  = std::max(a_r1,  std::abs(g1(ig,j,k) - g1(im,j,k)));
                        a_Mn  = std::max(a_Mn,  std::abs(gM(ig,j,k,0) + gM(im,j,k,0)));
                        a_E0  = std::max(a_E0,  std::abs(gE0(ig,j,k) - gE0(im,j,k)));
                        a_E1  = std::max(a_E1,  std::abs(gE1(ig,j,k) - gE1(im,j,k)));
                    }
        }
        static int _ma_count = 0;
        if ((_ma_count++ % 200) == 0 && (a_eta + a_r0 + a_r1 + a_Mn + a_E0 + a_E1) > 0)
        {
            char b[240];
            snprintf(b, 240, "MIRRORAUDIT lev=%d asym: eta=%.3e r0=%.3e r1=%.3e Mn=%.3e E0=%.3e E1=%.3e",
                     lev, a_eta, a_r0, a_r1, a_Mn, a_E0, a_E1);
            Util::Message(INFO, b);
        }
    }
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK fillghost_out lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
} // end FillGhost4BC()


////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// RELAXATION + REINITIALIZATION //////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
//
// Stiff pressure-relaxation (mu -> infinity) + reinitialization step for the
// 6-equation diffuse-interface model.
//
//   Step 1 (relaxation, Schmidmayer 2020 Sec.3.3, eq. (24)+(25); Saurel 2009 Sec.3.3,
//   eq. (III.4)):  per-cell Newton on
//                     f(p) = sum_k (alpha rho)_k v_k(p) - 1 = 0
//   with v_k from Saurel eq. (III.4) (frozen-p_hat_I form).  Updates alpha_k
//   holding (alpha rho)_k, momentum, and rho E fixed.
//
//   Step 2 (reinit, Schmidmayer 2020 eq. (26); Saurel 2009 eq. (III.5)):
//   recompute mixture p from the redundant rho E using the post-relaxation
//   alpha_k; then reset each per-phase (alpha rho e)_k from
//   (alpha rho e)_k = alpha_k (p + gamma_k pi_k)/(gamma_k - 1).
//
// Gamma is now a transported PRIMARY that never references |grad eta|, so it is
// already immune to non-material changes in eta and needs no snapshot/restore.
// Kept as no-ops so the call sites document where eta is rewritten without any
// material motion, should a future formulation need them again.



// ======================================================================
// L_cap : Schmidmayer 2017 eq. (17) / sec. 4.2.2.  See Hydro2.H.
// ======================================================================
void Hydro2::CapillaryOperator(int lev, Set::Scalar dt)
{
    if (!apply_surface_tension) return;

    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar sig_const = sigma;
    const int marm = marmottant;
    const Set::Scalar chi_ = marmottant_chi, Gb = marmottant_Gamma_buck;
    const Set::Scalar sbrk = marmottant_sigma_break, sigw = sigma;

    // Cell-centred capillary primitives, one ghost so the face averages below
    // have both neighbours.  Layout: [0]=||w||, [1]=w1^2/||w||,
    // [2]=w2^2/||w||, [3]=w1w2/||w||, [4]=sigma_eff.
    const int NQ = 5;
    amrex::MultiFab Q(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), NQ, 1);
    Q.setVal(0.0);
    for (amrex::MFIter mfi(Q, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = mfi.growntilebox(1);
        auto q  = Q.array(mfi);
        auto cf = cfun_mf[lev]->const_array(mfi);
        auto sh = shell_mf[lev]->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (!cf.contains(i - 1, j, k) || !cf.contains(i + 1, j, k)) return;
            if (!cf.contains(i, j - 1, k) || !cf.contains(i, j + 1, k)) return;
            Set::Vector w = Numeric::Gradient(cf, i, j, k, 0, DX);
            const Set::Scalar wn = w.lpNorm<2>();
            if (wn < 1.0e-10) return;
            const Set::Scalar se = SigmaEffFromGamma(marm ? sh(i, j, k) : 0.0, wn,
                                                     marm, sig_const, chi_, Gb, sbrk, sigw);
            q(i, j, k, 0) = wn;
            q(i, j, k, 1) = w(0) * w(0) / wn;
            q(i, j, k, 2) = w(1) * w(1) / wn;
            q(i, j, k, 3) = w(0) * w(1) / wn;
            q(i, j, k, 4) = se;
        });
    }
    Q.FillBoundary(geom[lev].periodicity());

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto q  = Q.const_array(mfi);
        auto M  = momentum_mf[lev]->array(mfi);
        auto E  = energy_per_vol_mf[lev]->array(mfi);
        auto v  = velocity_mf[lev]->const_array(mfi);
        const Set::Scalar dtl = dt, dxl = DX[0], dyl = DX[1];
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Face value = arithmetic average of the two neighbouring cells
            // (Schmidmayer sec. 4.2.2).  Shared between neighbours, so the
            // sum telescopes and the update is discretely conservative.
            auto fx = [&](int c, int s) { return 0.5 * (q(i, j, k, c) + q(i + s, j, k, c)); };
            auto fy = [&](int c, int s) { return 0.5 * (q(i, j, k, c) + q(i, j + s, k, c)); };
            // sigma_eff at faces, so a spatially varying (Marmottant) tension
            // carries the Marangoni contribution through the same averaging.
            const Set::Scalar sxp = 0.5 * (q(i, j, k, 4) + q(i + 1, j, k, 4));
            const Set::Scalar sxm = 0.5 * (q(i, j, k, 4) + q(i - 1, j, k, 4));
            const Set::Scalar syp = 0.5 * (q(i, j, k, 4) + q(i, j + 1, k, 4));
            const Set::Scalar sym = 0.5 * (q(i, j, k, 4) + q(i, j - 1, k, 4));

            const Set::Scalar dNx = (sxp * fx(0, 1) - sxm * fx(0, -1)) / dxl;
            const Set::Scalar dNy = (syp * fy(0, 1) - sym * fy(0, -1)) / dyl;
            const Set::Scalar d11x = (sxp * fx(1, 1) - sxm * fx(1, -1)) / dxl;
            const Set::Scalar d12x = (sxp * fx(3, 1) - sxm * fx(3, -1)) / dxl;
            const Set::Scalar d12y = (syp * fy(3, 1) - sym * fy(3, -1)) / dyl;
            const Set::Scalar d22y = (syp * fy(2, 1) - sym * fy(2, -1)) / dyl;

            M(i, j, k, 0) -= dtl * (-dNx + d11x + d12y);
            M(i, j, k, 1) -= dtl * (-dNy + d12x + d22y);

            // Energy: d_x[(w1^2 u + w1 w2 v)/||w||] + d_y[(w1 w2 u + w2^2 v)/||w||],
            // with u,v also face-averaged so the flux is single-valued.
            auto uf = [&](int d, int s) {
                return (d == 0) ? 0.5 * (v(i, j, k, 0) + v(i + s, j, k, 0))
                                : 0.5 * (v(i, j, k, 0) + v(i, j + s, k, 0)); };
            auto vf = [&](int d, int s) {
                return (d == 0) ? 0.5 * (v(i, j, k, 1) + v(i + s, j, k, 1))
                                : 0.5 * (v(i, j, k, 1) + v(i, j + s, k, 1)); };
            const Set::Scalar Ex = (sxp * (fx(1, 1) * uf(0, 1) + fx(3, 1) * vf(0, 1))
                                  - sxm * (fx(1, -1) * uf(0, -1) + fx(3, -1) * vf(0, -1))) / dxl;
            const Set::Scalar Ey = (syp * (fy(3, 1) * uf(1, 1) + fy(2, 1) * vf(1, 1))
                                  - sym * (fy(3, -1) * uf(1, -1) + fy(2, -1) * vf(1, -1))) / dyl;
            E(i, j, k) -= dtl * (Ex + Ey);
        });
    }
}

void Hydro2::RelaxAndReinit(int lev)
{
    BL_PROFILE("Integrator::Hydro2::RelaxAndReinit");
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK relax_entry   lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }

    // Constants for iteration
    const Set::Scalar alpha_floor = 1.0e-12;
    const Set::Scalar DIV_FLOOR   = 1.0e-30;
    const Set::Scalar small_loc   = small;        // kept for legacy callers
    const Set::Scalar cutoff_loc  = cutoff;       // pure-cell guard threshold
    const int         max_iter    = 30;
    const Set::Scalar newton_tol  = 1.0e-10;
    const int         bisect_max  = 120;
    const Set::Scalar bisect_tol  = 1.0e-12;
    // Practical |f| threshold for the "unconverged_cells" diagnostic.
    // Newton's strict newton_tol=1e-10 over-flags bisection cells that
    // are physically converged at |f| ~ 1e-8.  unconv_threshold relaxes
    // the warning to a value that matches engineering convergence
    // (volume constraint Sum_k (alpha rho)_k v_k(p) = 1 satisfied to
    // 1 part in 10^6).
    const Set::Scalar unconv_threshold = 1.0e-6;

    const Set::Scalar gam0 = eos0.Gamma();
    const Set::Scalar pi0_ = eos0.P0();
    const Set::Scalar gam1 = eos1.Gamma();
    const Set::Scalar pi1_ = eos1.P0();

    // Optional Newton diagnostic.  Per cell write {iter_count, |f(p_final)|};
    // after the ParallelFor reduce to {max iters, max residual, unconverged
    // cell count} and print a one-line summary.
    // MERGE (Marmottant): preserve the shell's areal density across the
    // pressure relaxation below (it moves volume between phases, not lipid).

    // MERGE (AcousticBC): honour relax_diag instead of the hardcoded `true`
    // that was on the Marmottant side.  That leftover forced the Newton
    // diagnostic on for every run and is why the logs were full of
    // RelaxAndReinit lines.
    const bool diag_on = (relax_diag != 0);
    std::unique_ptr<amrex::MultiFab> diag_mf;
    if (diag_on)
    {
        // 3 components: [0]=Newton iter count, [1]=|f(p_relaxed)|,
        //               [2]=|p_relaxed - p_reinit|
        // Component 2 detects whether the eq.26 reinit step is amplifying
        // cell-local noise.  If max(|p_relaxed - p_reinit|) is large or
        // correlates with checker artifacts, the reinit step is the noise
        // source (Newton's volume-constraint p disagrees with the energy-
        // consistent p from rho E and the new alpha).
        diag_mf = std::make_unique<amrex::MultiFab>(
            eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 3, 0);
        diag_mf->setVal(0.0);
    }

    // ------------------------------------------------------------------
    // CAPILLARY ENERGY SNAPSHOT (see Hydro2.H, capillary_closure).
    //
    // Schmidmayer 2017 splits the operators as L_relax(L_cap(L_hyper(U))),
    // so the capillary energy entering the relaxation is already fixed; and
    // their eps_sigma = sigma|grad c| is built from a colour function the
    // relaxation never touches.  Here eps_sigma is built from eta = alpha_1,
    // which the relaxation DOES rewrite (eta = a1_new below).  Evaluating it
    // inside the kernel is then a race: e_cap at cell i reads eta at i+-1,
    // which may or may not already have been relaxed depending on traversal
    // order.  Freeze it here, from the pre-relaxation eta, to reproduce the
    // same invariant deterministically.
    std::unique_ptr<amrex::MultiFab> ecap_snap;
    if (capillary_closure)
    {
        ecap_snap = std::make_unique<amrex::MultiFab>(
            eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, 0);
        const int marm_s = marmottant;
        const Set::Scalar sig_s = sigma, chi_s = marmottant_chi;
        const Set::Scalar Gb_s = marmottant_Gamma_buck;
        const Set::Scalar sbrk_s = marmottant_sigma_break, sigw_s = sigma;
        const Set::Scalar *DXs = geom[lev].CellSize();
        for (amrex::MFIter mfi(*ecap_snap, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto ec = ecap_snap->array(mfi);
            auto cf = cfun_mf[lev]->const_array(mfi);
            auto sh = shell_mf[lev]->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                ec(i, j, k) = CapEnergyAt(cf, sh, i, j, k, DXs, marm_s, sig_s,
                                          chi_s, Gb_s, sbrk_s, sigw_s);
            });
        }
    }

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        auto eta   = eta_mf[lev]->array(mfi);
        auto arh0  = rho_eta0_mf[lev]->array(mfi);
        auto arh1  = rho_eta1_mf[lev]->array(mfi);
        auto M_    = momentum_mf[lev]->array(mfi);
        auto E_    = energy_per_vol_mf[lev]->array(mfi);
        auto E0_   = energy0_mf[lev]->array(mfi);
        auto E1_   = energy1_mf[lev]->array(mfi);
        // Capillary-energy closure: read the frozen pre-relaxation snapshot
        // built above, never eta directly (the kernel rewrites eta).
        const int cap_close_x = capillary_closure;
        amrex::Array4<const Set::Scalar> ecap_s;
        if (capillary_closure) ecap_s = ecap_snap->const_array(mfi);

        // EMBEDDED SOLID indicator (empty Array4 when feature is off).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);

        // Diagnostic array (only used if diag_on).
        amrex::Array4<Set::Scalar> diag_arr = diag_on
            ? (*diag_mf)[mfi].array()
            : amrex::Array4<Set::Scalar>{};

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Skip solid cells: pressure relaxation must not touch the
            // frozen solid state (its per-phase energies are prescribed,
            // not in thermodynamic equilibrium with the alpha there).
            if (embedded.apply && embedded.isSolid(phisol(i, j, k))) return;
            
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            // Conservative partial densities
            Set::Scalar arh0_loc = std::max(arh0(i, j, k), 0.0);
            Set::Scalar arh1_loc = std::max(arh1(i, j, k), 0.0);

            // Pure-phase densities
            Set::Scalar rho0_pre = arh0_loc / std::max(a1, DIV_FLOOR);
            Set::Scalar rho1_pre = arh1_loc / std::max(a2, DIV_FLOOR);

            // Pre-relax per-phase pressures (Sau09 eq. II.3 inverted)
            Set::Scalar p0_pre = (gam0 - 1.0) * E0_(i, j, k) / std::max(a1, DIV_FLOOR) - gam0 * pi0_;
            Set::Scalar p1_pre = (gam1 - 1.0) * E1_(i, j, k) / std::max(a2, DIV_FLOOR) - gam1 * pi1_;
            // SG positivity floor (Sau09 Sec.3): p_k + pi_k > 0.
            p0_pre = std::max(p0_pre, -pi0_ + DIV_FLOOR);
            p1_pre = std::max(p1_pre, -pi1_ + DIV_FLOOR);

            // -----------------------------------------------------------
            // TRACE-PHASE SLAVING (relax): same ill-conditioning as the
            // recovery paths -- p_pre = (gam-1)E_k/alpha - gam*pi turns a
            // 4e-5 relative E/alpha inconsistency into a 100% pressure
            // error at gam*pi/p ~ 2.4e4.  Below the guard cutoff those
            // cells skip the Newton entirely, but in the band tail ABOVE
            // cutoff (eta ~ 1e-4..1e-2) the Newton ingests the mispriced
            // trace p_pre and MOVES alpha off its garbage relaxation
            // volume -- a secular interface-erosion loop (the residual
            // exponential drain after the recovery slaving landed).
            // Slave the trace phase's p_pre to the dominant phase's:
            // with p0_pre == p1_pre the volume constraint is satisfied
            // at p_pre exactly (v_k = v_k^0 -> f = a1+a2-1 = 0), so the
            // Newton converges immediately and alpha is NOT moved --
            // which is the correct physical outcome for a cell whose
            // trace phase has no independent pressure to relax against.
            // -----------------------------------------------------------
            {
                // SMOOTH blend (see recovery blocks): hard switches leave a
                // seam the dynamics lock onto.
                const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                const Set::Scalar er = eta(i, j, k);
                Set::Scalar w0 = std::min(std::max((er - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w0 = w0 * w0 * (3.0 - 2.0 * w0);
                Set::Scalar w1 = std::min(std::max(((1.0 - er) - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w1 = w1 * w1 * (3.0 - 2.0 * w1);
                const Set::Scalar p0_raw = p0_pre, p1_raw = p1_pre;
                p0_pre = w0 * p0_raw + (1.0 - w0) * p1_raw;
                p1_pre = w1 * p1_raw + (1.0 - w1) * p0_raw;
            }

            // -----------------------------------------------------------
            // PURE-CELL GUARD: relaxation is an interface operation
            // (Sau09/Sch20 relax alpha between phases -- meaningless when
            // one phase is below the eta cutoff).  In near-pure cells the
            // trace phase's E_k is alpha-amplified noise: p_pre =
            // (gam-1)E_k/alpha - gam pi turns roundoff into MPa at
            // alpha ~ 1e-12, and when the trace pressure floors at -pi
            // the volume constraint f(p) goes CONSTANT (plateau at
            // |f| ~ 1/gamma_gas: no root exists, Newton and bisection
            // both spin to max_iter).  The floored state is NOT confined
            // to sub-cutoff cells -- band-TAIL cells (eta ~ 1e-4..1e-2)
            // hit it through advection over/undershoot of the steep
            // trace E_k profile -- so the guard fires on EITHER
            // (a) eta beyond cutoff, or (b) either phase's pre-relax
            // pressure sitting at its positivity floor (that phase's
            // v(p) has lost EOS meaning: the solve is garbage-in).
            // Skip the solve: freeze alpha AND eta (no eta stomp to the
            // alpha_floor clamp), and do only the energy-consistent
            // reinit so both per-phase energies return to mechanical
            // equilibrium with the conserved mixture state
            // (self-healing: next step's p_pre is then consistent and
            // the cell rejoins the Newton path).
            // -----------------------------------------------------------
            const bool p0_floored = (p0_pre <= -pi0_ + std::max(2.0 * DIV_FLOOR, 1.0e-9 * pi0_));
            const bool p1_floored = (p1_pre <= -pi1_ + std::max(2.0 * DIV_FLOOR, 1.0e-9 * pi1_));
            if (eta(i, j, k) < cutoff_loc || eta(i, j, k) > 1.0 - cutoff_loc
                || p0_floored || p1_floored)
            {
                Set::Scalar rho_p  = arh0_loc + arh1_loc;
                Set::Scalar ke_p   = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                       + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                       + M_(i, j, k, 2) * M_(i, j, k, 2))) / std::max(rho_p, small_loc);
                Set::Scalar rhoe_p = std::max(E_(i, j, k) - ke_p, small_loc);
                Set::Scalar p_pure = Solver::EOS::EOS::ReinitMixturePressure(rhoe_p, a1, a2,
                                                                            gam0, pi0_, gam1, pi1_, small_loc);
                p_pure = std::max(p_pure, -std::min(pi0_, pi1_) + small_loc);
                // Write E_k with the FROZEN raw eta (clamped only to [0,1]),
                // NOT the alpha_floor-clamped a1: eta stays frozen in this
                // branch, so the recovery divide E_k/eta only cancels exactly
                // when E_k was built with the same eta.  Using the 1e-12-
                // clamped a1 here re-creates the divide-floor mispricing in
                // the band tail (eta ~ 1e-14..1e-6) that this guard exists
                // to prevent.
                const Set::Scalar a1g = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2g = 1.0 - a1g;
                E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a1g, gam0, pi0_, small_loc);
                E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a2g, gam1, pi1_, small_loc);
                if (diag_on)
                {
                    diag_arr(i, j, k, 0) = 0.0;
                    diag_arr(i, j, k, 1) = 0.0;
                    diag_arr(i, j, k, 2) = 0.0;
                }
                return;
            }

            // -----------------------------------------------------------
            // Newton on f(p) = arh0*v0(p) + arh1*v1(p) - 1 = 0, with
            // bisection fallback when Newton stalls.
            //
            // SC form (Sch20 eq. 24+25):
            //   v_k(p) = v_k^0 * [p_k^0 + gam pi + (gam-1) p] / [gam (p + pi)]
            //
            // Monotonicity (proven in Pressure_Relaxation.tex):
            //   dv_k/dp = -v_k^0 (p_k^0+pi_k)/(gam_k (p+pi_k)^2) < 0
            // strictly, given SG positivity p_k^0 + pi_k > 0.  Hence
            // f(p) is monotone-decreasing and its root is UNIQUE on
            // (-min_k(pi_k), +infty).  Therefore bisection is bulletproof
            // for cells where Newton fails to converge -- which it does
            // when the Tammann liquid Jacobian arh0*dv0/dp ~ 4e-13
            // dwarfs the convergence test scale (problematic for
            // interface cells with conflicting per-phase pressures).
            // -----------------------------------------------------------
            const Set::Scalar p_min = -std::min(pi0_, pi1_) + DIV_FLOOR;

            // Sch20 eq. 8 mixture pressure as the initial guess --
            // exact for the mechanical-equilibrium IC (p0_pre == p1_pre
            // -> p_init == p_mix == root), and a sensible interpolation
            // for general states.  Bounded away from p=0 by virtue of
            // SG positivity on both pre-relax phases.
            Set::Scalar p_init = a1 * p0_pre + a2 * p1_pre;
            // Lower bracket: a hair above p_min (avoid the gas v -> inf
            // pole at p -> -pi_gas).  For typical gas/liquid: p_min ~ 0,
            // p_lo ~ small positive.  For all-liquid pair: p_min ~ -pi,
            // p_lo > p_min.
            Set::Scalar p_lo = std::max(p_min, 1.0e-3 * std::max(std::abs(p_init), 1.0));
            // Upper bracket: generously above the maximum pre-relax p.
            Set::Scalar p_hi = 1.0e4 * std::max(std::max(std::abs(p0_pre), std::abs(p1_pre)), 1.0);

            // Local evaluator (captures pre-relax state from this cell).
            auto eval_f = [&](Set::Scalar p) -> Set::Scalar
            {
                Set::Scalar v0 = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1 = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                return arh0_loc * v0 + arh1_loc * v1 - 1.0;
            };

            // Start Newton at p_init clamped into [p_lo, p_hi].
            Set::Scalar p = std::max(std::min(p_init, p_hi), p_lo);

            int iters_used = max_iter;
            Set::Scalar f_final = eval_f(p);
            bool newton_ok = false;
            for (int it = 0; it < max_iter; ++it)
            {
                Set::Scalar v0_p = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1_p = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                Set::Scalar f    = arh0_loc * v0_p + arh1_loc * v1_p - 1.0;
                f_final = f;

                Set::Scalar dv0 = Solver::EOS::EOS::RelaxationVolume_SG_SC_dvdp(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar dv1 = Solver::EOS::EOS::RelaxationVolume_SG_SC_dvdp(p, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                Set::Scalar df  = arh0_loc * dv0 + arh1_loc * dv1;

                if (std::abs(df) < DIV_FLOOR) break;       // bail to bisection
                Set::Scalar dp = -f / df;
                Set::Scalar p_new = p + dp;
                // Clamp Newton step into the bracket -- prevents the
                // Tammann-stiff cells from taking 10^9-scale steps that
                // overshoot into the gas singularity at p -> 0.
                if (p_new < p_lo) p_new = 0.5 * (p + p_lo);
                if (p_new > p_hi) p_new = 0.5 * (p + p_hi);
                if (std::abs(dp) < newton_tol * std::max(std::abs(p_new), 1.0))
                {
                    p = p_new;
                    f_final = eval_f(p);
                    iters_used = it + 1;
                    newton_ok = true;
                    break;
                }
                p = p_new;
            }

            // -----------------------------------------------------------
            // Bisection fallback.  Triggered when Newton ran the full
            // max_iter without satisfying the convergence test OR when
            // Newton bailed early on a degenerate Jacobian.  f(p) is
            // monotone-decreasing so bracketing is robust: we just need
            // f(p_lo) > 0 and f(p_hi) < 0.  If the bracket is broken
            // (rare degenerate states), expand outward.
            // -----------------------------------------------------------
            if (!newton_ok)
            {
                Set::Scalar f_lo = eval_f(p_lo);
                Set::Scalar f_hi = eval_f(p_hi);

                // Walk the upper bound out if we don't bracket -- happens
                // if pre-relax state has anomalously high partial densities.
                int expand = 0;
                while (f_hi > 0.0 && expand < 20)
                {
                    p_hi *= 10.0;
                    f_hi  = eval_f(p_hi);
                    ++expand;
                }
                // Walk lower bound in if needed.
                while (f_lo < 0.0 && p_lo > p_min && expand < 40)
                {
                    p_lo = std::max(0.5 * p_lo, p_min);
                    f_lo = eval_f(p_lo);
                    ++expand;
                }

                if (f_lo * f_hi <= 0.0)
                {
                    // Standard bisection -- f monotone-decreasing.
                    Set::Scalar p_mid = 0.5 * (p_lo + p_hi);
                    Set::Scalar f_mid = 0.0;
                    int bit = 0;
                    for (bit = 0; bit < bisect_max; ++bit)
                    {
                        p_mid = 0.5 * (p_lo + p_hi);
                        f_mid = eval_f(p_mid);
                        if (f_mid > 0.0) p_lo = p_mid;
                        else             p_hi = p_mid;
                        if ((p_hi - p_lo) < bisect_tol * std::max(std::abs(p_mid), 1.0)) break;
                    }
                    p = p_mid;
                    f_final = f_mid;
                    iters_used = max_iter + bit + 1;   // distinguishable from Newton-only count
                }
                // else: degenerate (no root in bracket); leave p at last Newton iterate.
            }

            Set::Scalar p_relaxed = p;

            // Write diagnostic: (iter count, |f| at final p).
            if (diag_on)
            {
                diag_arr(i, j, k, 0) = static_cast<Set::Scalar>(iters_used);
                diag_arr(i, j, k, 1) = std::abs(f_final);
            }

            // Post-relax volume fractions (using self-consistent v_k(p)).
            // DIV_FLOOR matches the pre-relax / Newton-loop convention.
            // ROBUST FALLBACK: if the volume-constraint solve did NOT converge
            // (unphysical per-phase input -- e.g. a coarse cell whose per-phase
            // ENERGY split came out with the wrong sign, while the conserved
            // MIXTURE energy E is still correct), do NOT trust the failed solve's
            // alpha.  Keep the prior alpha and let the energy-consistent reinit
            // (Sch20 eq.26, below) rebuild a valid per-phase state from E.  Without
            // this the masked-garbage alpha poisons the interface (rho_eta0/1) and
            // the per-phase energies at the bubble center on coarse grids.
            const bool relax_failed = std::abs(f_final) > unconv_threshold;
            Set::Scalar a1_new, a2_new;
            if (relax_failed)
            {
                a1_new = a1;   // prior (clamped) volume fraction -- trust the conserved E instead
                a2_new = a2;
            }
            else
            {
                Set::Scalar v0_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                a1_new = arh0_loc * v0_r;
                a2_new = arh1_loc * v1_r;
                Set::Scalar asum = a1_new + a2_new;
                if (asum > DIV_FLOOR) { a1_new /= asum; a2_new /= asum; }
            }
            a1_new = std::min(std::max(a1_new, alpha_floor), 1.0 - alpha_floor);
            a2_new = 1.0 - a1_new;

            // -----------------------------------------------------------
            // Reinit: recompute mixture p from the (CONSERVED) redundant
            // rho E using the POST-relaxation alpha_k.
            // Schmidmayer 2020 eq. 26 / Saurel 2009 eq. III.5.
            // -----------------------------------------------------------
            Set::Scalar rho_loc = arh0_loc + arh1_loc;
            Set::Scalar ke      = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                    + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                    + M_(i, j, k, 2) * M_(i, j, k, 2))) / std::max(rho_loc, small_loc);
            // Debit the capillary energy before the EOS inversion: with the
            // closure on, rho E carries e_cap, and reading it as internal
            // energy inflates p at the interface (dt collapse).  This is the
            // THIRD rho E inversion site -- it names its local `rho_e`, which
            // is why the E_vol - KE grep did not surface it.
            Set::Scalar rho_e   = std::max(E_(i, j, k) - ke
                                  - (cap_close_x ? ecap_s(i, j, k) : 0.0), small_loc);

            Set::Scalar p_reinit = Solver::EOS::EOS::ReinitMixturePressure(rho_e, a1_new, a2_new,
                                                                          gam0, pi0_, gam1, pi1_, small_loc);

            // Floor: keep p + gam_k pi_k > 0 for both phases (SG positivity).
            const Set::Scalar p_min_ph = -std::min(pi0_, pi1_) + small_loc;
            p_reinit = std::max(p_reinit, p_min_ph);

            // Diagnostic: gap between Newton's converged p and reinit p.
            // Nonzero gap means the volume-constraint root and the energy-
            // consistent root disagree -- per-cell noise from the upwind
            // flux feeding into a cell-local Newton, or vice versa.
            if (diag_on)
            {
                diag_arr(i, j, k, 2) = std::abs(p_relaxed - p_reinit);
            }

            // -----------------------------------------------------------
            // Reset per-phase canonical energies from p_reinit.
            // (Sau09 eq. III.5: e_k = (p + gam pi)/((gam-1) rho_k), so
            //  (alpha rho e)_k = alpha_k (p + gam pi)/(gam - 1).)
            // -----------------------------------------------------------
            eta(i, j, k) = a1_new;
            E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_reinit, a1_new, gam0, pi0_, small_loc);
            E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_reinit, a2_new, gam1, pi1_, small_loc);

            // rho_eta{0,1}, momentum, energy_per_vol are NOT touched
            // (they are conserved through relaxation by construction).
        });
    }


    // ----------------------------------------------------------------
    // Diagnostic summary.
    // ----------------------------------------------------------------
    if (diag_on)
    {
        const Set::Scalar max_iters = diag_mf->max(0);   // already MPI-reduced
        const Set::Scalar max_res   = diag_mf->max(1);
        const Set::Scalar max_pgap  = diag_mf->max(2);   // |p_relaxed - p_reinit|

        // Count cells that hit max_iter without converging.
        int unconv = 0;
        for (amrex::MFIter mfi(*diag_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto arr = (*diag_mf)[mfi].array();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        // Count as truly unconverged only if iters hit the max AND |f| exceeds the physical-engineering tolerance.
                        if (static_cast<int>(arr(i, j, k, 0)) >= max_iter
                            && arr(i, j, k, 1) > unconv_threshold)
                            ++unconv;
                    }
        }
        amrex::ParallelDescriptor::ReduceIntSum(unconv);

        Util::Message(INFO, "RelaxAndReinit lev=", lev,
                      " step=", step_counter[lev],
                      " max_iters=", static_cast<int>(max_iters),
                      "/", max_iter,
                      " max_residual=", max_res,
                      " max|p_relaxed-p_reinit|=", max_pgap,
                      " unconverged_cells=", unconv);

        if (unconv > 0)
        {
            Util::Message(INFO,
                "  ** WARNING: ", unconv,
                " cell(s) hit max_iter without satisfying |f(p)| < ",
                newton_tol, ".  Newton is failing to relax pressure in those cells.");
        }
    }
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK relax_exit    lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
} // end RelaxAndReinit()


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// REFLUX SCRATCH ALLOC ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conserved primaries refluxed at coarse-fine boundaries:
//   rho_eta0, rho_eta1, M_x, M_y, rho*E, E_0, E_1   (7 components total)
//
// We split mass per-phase (eta*F, (1-eta)*F at the face) rather than refluxing
// total mass + repartitioning, because rho_eta_k are the actual evolved fields
// (not density_mf, which is derived).  See Matt Q's commit a5783ab32.
//
// E_0, E_1 are 6-eq additions not present in the 5-eq line; refluxed alongside
// the mixture energy so per-phase internal energies stay consistent across c/f.
//
// Uses density_mf[lev]'s BoxArray / DistributionMap because grids[lev]/dmap[lev]
// is not guaranteed populated when this runs (see memory note
// feedback_amr_perlevel_alloc_must_cover_level0.md -- prior reflux attempt
// segfaulted by reading empty grids[lev] inside MakeNewLevelFromScratch).
// density_mf[lev] is guaranteed allocated by RegisterNewFab before Initialize.
//
void Hydro2::AllocateRefluxScratch(int lev)
{
    BL_PROFILE("Integrator::Hydro2::AllocateRefluxScratch");

    if ((int)flux_reg.size() <= lev)  flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev) cc_fluxes.resize(lev + 1);

    const amrex::BoxArray& ba           = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm = density_mf[lev]->DistributionMap();

    // FluxRegister sits between lev-1 (coarse) and lev (fine).
    if (lev > 0)
    {
        const int ncomp_reflux = 2 + AMREX_SPACEDIM + 1 + 2; // rho0, rho1, M_x, M_y, rhoE, E_0, E_1
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_reflux);
        flux_reg[lev]->setVal(0.0);
    }

    // Cell-centered hi-face flux scratch.  1 ghost so FillBoundary can
    // propagate the hi-face flux of one box to the lo-face of its neighbor
    // when converting to face-centered MultiFabs in Advance().
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba, dm, 2,                1);
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba, dm, AMREX_SPACEDIM,   1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba, dm, 1,                1);
        cc_fluxes[lev].ene_k[d]  = std::make_unique<amrex::MultiFab>(ba, dm, 2,                1);

        cc_fluxes[lev].mass[d]  ->setVal(0.0);
        cc_fluxes[lev].mom[d]   ->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
        cc_fluxes[lev].ene_k[d] ->setVal(0.0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// POST-SUBCYCLE REFLUX /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Called from base Integrator::TimeStep AFTER all fine subcycles complete
// and BEFORE average_down.  Applies the FluxRegister correction to the
// coarse-level conserved primaries, then re-derives density/velocity from
// the corrected conserved state.
//
// Register layout (must match component offsets in Advance/RHS):
//   [0]               rho_eta0       (per-phase mass 0)
//   [1]               rho_eta1       (per-phase mass 1)
//   [2 .. 2+SD-1]     momentum_k     (k = x, y, ...)
//   [2+SD]            energy_per_vol (total rho*E)
//   [3+SD .. 4+SD]    E_0, E_1       (per-phase internal energies; 6-eq)
//
void Hydro2::PostSubcycleReflux(int lev, Set::Scalar /*time*/, Set::Scalar /*dt_coarse*/)
{
    BL_PROFILE("Integrator::Hydro2::PostSubcycleReflux");

    if (lev >= finest_level) return;
    const int fine_lev = lev + 1;
    if (fine_lev >= (int)flux_reg.size() || !flux_reg[fine_lev]) return;

    constexpr int IDX_RHO_ETA0 = 0;
    constexpr int IDX_RHO_ETA1 = 1;
    constexpr int IDX_M        = 2;
    constexpr int IDX_E_VOL    = 2 + AMREX_SPACEDIM;
    constexpr int IDX_E0       = 3 + AMREX_SPACEDIM;
    constexpr int IDX_E1       = 4 + AMREX_SPACEDIM;

    flux_reg[fine_lev]->Reflux(*rho_eta0_mf[lev],       1.0, IDX_RHO_ETA0, 0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*rho_eta1_mf[lev],       1.0, IDX_RHO_ETA1, 0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*momentum_mf[lev],       1.0, IDX_M,        0, AMREX_SPACEDIM, geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy_per_vol_mf[lev], 1.0, IDX_E_VOL,    0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy0_mf[lev],        1.0, IDX_E0,       0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy1_mf[lev],        1.0, IDX_E1,       0, 1,              geom[lev]);

    // Recompute derived fields from the refluxed conserved state.
    // density and velocity are not refluxed directly -- they're derived.
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto rho  = density_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);
        auto M    = momentum_mf[lev]->array(mfi);
        auto v    = velocity_mf[lev]->array(mfi);
        const Set::Scalar small_local = small;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            const Set::Scalar rho_safe = std::max(rho(i, j, k), small_local);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / rho_safe;
        });
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// POST-AVERAGE-DOWN //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// AMReX average_down independently averages every conserved primary into
// each coarse-under-fine cell.  For the 6-eq state that includes E_0 and E_1
// (per-phase internal energies), which are nonlinear functions of (rho*E,
// rho_eta_k, alpha) via the EOS.  Independent averaging breaks the EOS
// relation: the averaged (E_0, E_1) don't satisfy mech equilibrium with the
// averaged (alpha, rho_eta_k, rho*E). 
// 
// After average_down, recompute (E_0, E_1) on the coarse level from
// the averaged mixture state via Saurel III.5 mech-equilibrium:
//   e_int = sum_k alpha_k * (p_mix + gamma_k * pi_k) / (gamma_k - 1)
//   solve for p_mix, then E_k = alpha_k * (p_mix + gamma_k * pi_k) / (gamma_k - 1).
//
void Hydro2::PostAverageDown(int coarse_lev)
{
    BL_PROFILE("Integrator::Hydro2::PostAverageDown");

    // Only coarse-under-fine cells got touched by average_down -- those are
    // the cells whose EOS consistency we need to restore.  Other coarse
    // cells are already at mech equilibrium from the prior RelaxAndReinit;
    // overwriting them introduces a small per-step bias that compounds in
    // low-Mach tests (~11 um interface drift over 3 us in pure advection).
    //
    // makeFineMask builds an iMultiFab on the coarse grid: 1 where covered
    // by the fine BoxArray (coarsened to coarse resolution), 0 elsewhere.
    if (coarse_lev + 1 > finest_level) return;

    amrex::iMultiFab fine_cover_mask = amrex::makeFineMask(
        eta_mf[coarse_lev]->boxArray(),
        eta_mf[coarse_lev]->DistributionMap(),
        eta_mf[coarse_lev + 1]->boxArray(),
        refRatio(coarse_lev),
        /*crse_value=*/ 0,
        /*fine_value=*/ 1);

    const Set::Scalar gam0 = eos0.Gamma();
    const Set::Scalar pi0_ = eos0.P0();
    const Set::Scalar gam1 = eos1.Gamma();
    const Set::Scalar pi1_ = eos1.P0();
    const Set::Scalar small_local = small;

    for (amrex::MFIter mfi(*eta_mf[coarse_lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto eta   = eta_mf[coarse_lev]        ->array(mfi);
        auto rho0  = rho_eta0_mf[coarse_lev]   ->array(mfi);
        auto rho1  = rho_eta1_mf[coarse_lev]   ->array(mfi);
        auto M     = momentum_mf[coarse_lev]   ->array(mfi);
        auto E_vol = energy_per_vol_mf[coarse_lev]->array(mfi);
        auto E0    = energy0_mf[coarse_lev]    ->array(mfi);
        auto E1    = energy1_mf[coarse_lev]    ->array(mfi);
        auto rho   = density_mf[coarse_lev]    ->array(mfi);
        auto v     = velocity_mf[coarse_lev]   ->array(mfi);
        auto mask  = fine_cover_mask.const_array(mfi);
        // Capillary-energy closure inputs (see Hydro2.H).
        const int cap_close_r = capillary_closure, marm_r = marmottant;
        const Set::Scalar sig_r = sigma, chi_r = marmottant_chi;
        const Set::Scalar Gb_r = marmottant_Gamma_buck;
        const Set::Scalar sbrk_r = marmottant_sigma_break, sigw_r = sigma;
        auto shell_r = shell_mf[coarse_lev]->const_array(mfi);
        auto cfun_r  = cfun_mf[coarse_lev]->const_array(mfi);
        const Set::Scalar *DXr = geom[coarse_lev].CellSize();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Skip cells that average_down didn't touch.
            if (mask(i, j, k) == 0) return;
            // Re-derive density and velocity from refluxed/averaged conserved.
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            const Set::Scalar rho_safe = std::max(rho(i, j, k), small_local);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / rho_safe;

            // Volume fractions (clamped).
            const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a1 = 1.0 - a0;

            // Mixture internal energy per volume:  e_int = rho*E - KE
            const Set::Scalar KE = 0.5 * rho_safe * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                               + v(i, j, k, 1) * v(i, j, k, 1),
                                                               + v(i, j, k, 2) * v(i, j, k, 2)));
            const Set::Scalar e_int = E_vol(i, j, k) - KE
                                    - (cap_close_r ? CapEnergyAt(cfun_r, shell_r, i, j, k, DXr,
                                                                 marm_r, sig_r, chi_r, Gb_r,
                                                                 sbrk_r, sigw_r) : 0.0);

            // Sau09 III.5:  e_int = sum_k alpha_k * (p + gamma_k * pi_k) / (gamma_k - 1)
            //   p_mix * [a0/(g0-1) + a1/(g1-1)] = e_int - [a0*g0*pi0/(g0-1) + a1*g1*pi1/(g1-1)]
            const Set::Scalar denom = a0 / std::max(gam0 - 1.0, small_local)
                                    + a1 / std::max(gam1 - 1.0, small_local);
            const Set::Scalar stiff_offset = a0 * gam0 * pi0_ / std::max(gam0 - 1.0, small_local)
                                           + a1 * gam1 * pi1_ / std::max(gam1 - 1.0, small_local);
            const Set::Scalar p_mix = (e_int - stiff_offset) / std::max(denom, small_local);

            // Recompute E_k consistent with p_mix.  This overwrites whatever
            // independent average_down produced.
            E0(i, j, k) = a0 * (p_mix + gam0 * pi0_) / std::max(gam0 - 1.0, small_local);
            E1(i, j, k) = a1 * (p_mix + gam1 * pi1_) / std::max(gam1 - 1.0, small_local);
        });
    }
}


} // end of Integrator namespace

//#endif
