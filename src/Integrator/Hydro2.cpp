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
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
//#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
//#include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"


// Limiters / primitive-variable reconstruction (Sch20 §3.2)
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
        pp.query_default("p_refinement_criterion", value.p_refinement_criterion, 1e-3);         // pressure-based refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);    // density-based refinement

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);                // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl);  // cfl condition
        pp_query_default("pref", value.pref, 0.0);          // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);       // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-6);   // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0);  // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);            // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix");  // Roe solver entropy fix
        pp_query_default("Spec_Vol", value.Spec_Vol, 1);    // 0: Solve Energy via specific mass | 1: Solve Energy via specific volume

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, false); // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("apply_weight", value.apply_weight, false);                  // Apply weight when solving, default: false --> "No Weight"
        pp_query_default("apply_vaporization", value.apply_vaporization, false);       // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"
        pp_query_default("static_eta", value.static_eta, false);                      // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // Artificial heat exchange (AHE) on per-phase internal energy rows.
        // See bin/AHE.md for derivation: Schmidmayer 2020 eq. 13 r-source
        // structural form with Saurel-Petitpas-Berry 2009 §5.3 Fig. 21
        // curve fit for mu(v).
        // ahe.method: top-level switch that maps to (apply_ahe, ahe.form).
        //   0 = Sch20 stiff-limit only (no explicit AHE source)
        //   1 = Sch20 finite-mu r-source (+/- mu p_I Dp)
        //   2 = Sau09 sec.5.3 q*@u/@x form
        // When set (>= 0) overrides the legacy apply_ahe / ahe.form parses.
        int ahe_method_in = -1;
        pp_query_default("ahe.method", ahe_method_in, -1);

        pp_query_default("apply_ahe",             value.apply_ahe,             0);
        pp_query_default("ahe.use_const_mu",      value.ahe_use_const_mu,      0);
        pp_query_default("ahe.mu_const",          value.ahe_mu_const,          0.0);
        pp_query_default("ahe.mu_scale",          value.ahe_mu_scale,          1.0);
        pp_query_default("ahe.mu_a",              value.ahe_mu_a,             -5.64e7);
        pp_query_default("ahe.mu_b",              value.ahe_mu_b,              5.34e3);
        pp_query_default("ahe.mu_c",              value.ahe_mu_c,            -25.6);
        pp_query_default("ahe.v_min",             value.ahe_v_min,             2.65e-4);
        pp_query_default("ahe.v_max",             value.ahe_v_max,             4.61e-4);
        pp_query_default("ahe.compression_only",  value.ahe_compression_only,  0);
        pp_query_default("ahe.apply_alpha_src",   value.ahe_apply_alpha_src,   1);
        pp_query_default("ahe.max_frac",          value.ahe_max_frac,          0.1);
        pp_query_default("ahe.form",              value.ahe_form,              1);

        // Apply ahe.method override if the user set it.
        if (ahe_method_in == 0)
        {
            value.ahe_method = 0;
            value.apply_ahe  = 0;            // Sch20 stiff limit -- RelaxAndReinit only
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
        // else (ahe_method_in == -1): unset -> fall back to legacy flags as parsed.

        // FLUID 0
        pp_query_required("mu0", value.mu0); // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0); // bulk viscosity coefficient
        
        // FLUID 1
        pp_query_required("mu1", value.mu1); // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0); // bulk viscosity coefficient

        // EOS
        Solver::EOS::Tammann::Parse(value.eos0, pp, "eos0.");
        Solver::EOS::Tammann::Parse(value.eos1, pp, "eos1.");

        // EOS backend selection: routes the gas-branch PhasicPressureFromEnergy /
        // PhasicEnergyFromPressure / PhasicSoundSpeed calls (pi_k == 0) to either
        // the native Tammann+CPG implementation (default) or the PelePhysics
        // shim (Solver::EOS::PelePhysicsEOS, requires PELE_ENABLED build).
        // Tammann liquid (pi_k > 0) always stays native.
        //   eos.backend = tammann   (default; equivalent to "native")
        //   eos.backend = pelephysics
        // See bin/PeleC.md for the integration plan.
        std::string eos_backend_str = "tammann";
        pp_query_default("eos.backend", eos_backend_str, "tammann");
        Solver::EOS::SetBackend(eos_backend_str);

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0);    // Surface tension condition
        pp_query_default("Dv", value.Dv, 0.0);          // Vapor Diffusivity
        pp_query_required("epsilon", value.epsilon);    // diffuse interface thickness Y_infinity
        pp_query_default("Y_infinity", value.Y_infinity, 0.0); // Far Field Vapor Mass Fraction

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 2); // Method to solve for curvature

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

        // Newton diagnostic for stiff pressure relaxation.
        // 1 = print per-stage {max_iters, max_residual, count_unconverged}.
        pp_query_default("relax_diag", value.relax_diag, 0);

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

        // NSCBC ported to the 6-equation model: characteristic decomposition
        // now uses the FROZEN mixture sound speed (Sch20 eq. 17 / Sau09
        // eq. III.2) inside NSCBC::compute_prim, consistent with the
        // hyperbolic-step HLLC.  After NSCBC modifies the mixture (rho, M, ρE)
        // ghosts, FillGhost4BC re-derives per-phase (αρ)_k and E_k from the
        // updated mixture state and the extrapolated alpha (linearly
        // degenerate; see post-NSCBC block in FillGhost4BC).

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
    
            // Use BC::Nothing (does nothing when called)
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
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,           value.energy_bc, 1, nghost, "eta", true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.energy_bc, 1, nghost, "eta_old", false, true);
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
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true, true);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false, true);
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
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc, 1, nghost, "mu_chem", true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true, false);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, false, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   value.energy_bc,  1, nghost, "UE_per_vol", true, false);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   value.energy_bc,  1, nghost, "UE_per_mass", true, false);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   value.energy_bc,  1, nghost, "KE_per_vol", true, false);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   value.energy_bc,  1, nghost, "KE_per_mass", true, false);        // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,           &value.bc_nothing,  1, nghost, "Spadling_Number", true, false);    // Spalding Number
        value.RegisterNewFab(value.Y_mf,            &value.bc_nothing,  1, nghost, "Mass_Fraction", true, false);       // Mass Fraction

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, 0, "grad_eta", true, false, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, 0, "kappa", true, false, { "Avg", "1", "2" }); // To Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, 0, "grad_mag_grad_eta", false, false, { "x", "y" }); // grad( | grad(eta) | )
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
                      Solver::Local::FluidRiemann::HLLE,
                      Solver::Local::FluidRiemann::HLLC,
                      Solver::Local::FluidRiemann::HLLCE,
                      //Solver::Local::FluidRiemann::HLLCE_WENO5, // Never verified but updated
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman, // Can't remember if this has been verified
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro
                      //Solver::Local::FluidRiemann::Upwind,
                      //Solver::Local::FluidRiemann::Lax_Friedrich
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

            // --- 6-equation canonical state at IC (Schmidmayer 2020 §2.3) ---
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
            M(i, j, k, 0) = rho_eta0(i, j, k) * v0(i, j, k, 0) + rho_eta1(i, j, k) * v1(i, j, k, 0);
            M(i, j, k, 1) = rho_eta0(i, j, k) * v0(i, j, k, 1) + rho_eta1(i, j, k) * v1(i, j, k, 1);
            M_old(i, j, k, 0) = M(i, j, k, 0);
            M_old(i, j, k, 1) = M(i, j, k, 1);

            // Mixture velocity (diagnostic):
            v(i, j, k, 0) = M(i, j, k, 0) / std::max(rho(i, j, k), small);
            v(i, j, k, 1) = M(i, j, k, 1) / std::max(rho(i, j, k), small);

            // Mixture kinetic energy (consistent with rho E):
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0)
                                                  + v(i, j, k, 1) * v(i, j, k, 1));
            KE_mas(i, j, k) = (rho(i, j, k) > small) ? KE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // ===========================================================
            // MECHANICAL-EQUILIBRIUM INITIAL CONDITION (Schmidmayer 2020
            // §3.3, line 538-541: "the conservative variables follow from
            // simple mixture relations, allowing thermodynamic consistency").
            //
            // In input files the per-phase IC pressures (pressure0_ic,
            // pressure1_ic) are typically set to the bulk values of the
            // two pure phases (e.g., Garrick eq. 68: p0 = 3.059e-4 liquid,
            // p1 = 2.753 gas).  In a diffuse-interface cell with intermediate
            // alpha the literal per-phase IC pressures are NOT in mechanical
            // equilibrium.  Schmidmayer's 6-eq model assumes mechanical-
            // equilibrium IC where p_1 = p_2 = p_mix.  Build that here:
            //
            //   p_mix = alpha_1 p0_ic + alpha_2 p1_ic    (Schmidmayer eq. 8)
            //   (alpha rho e)_k = alpha_k (p_mix + gamma_k pi_k) / (gamma_k - 1)
            //                                              (Sau09 eq. III.5)
            //
            // In pure-phase cells (alpha = 0 or 1) this reduces to the
            // active-phase IC pressure, so pure-phase tests (Toro1a) are
            // unchanged.  In diffuse-interface cells it produces the
            // correct mechanical-equilibrium IC that the 5-eq reference
            // uses.
            // ===========================================================
            const Set::Scalar p_mix_IC = a1 * p0(i, j, k) + a2 * p1(i, j, k);

            E0_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(p_mix_IC, a1, gam0, pi0_, small);
            E1_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(p_mix_IC, a2, gam1, pi1_, small);
            E0_old_arr(i, j, k) = E0_arr(i, j, k);
            E1_old_arr(i, j, k) = E1_arr(i, j, k);

            // Mixture internal energy (sum of canonical per-phase energies):
            UE_vol(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);
            UE_mas(i, j, k) = (rho(i, j, k) > small) ? UE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // Redundant total energy rho E (Schmidmayer 2020 eq. 16):
            E_vol(i, j, k)     = KE_vol(i, j, k) + UE_vol(i, j, k);
            E_vol_old(i, j, k) = E_vol(i, j, k);
            E_mas(i, j, k)     = KE_mas(i, j, k) + UE_mas(i, j, k);
            E_mas_old(i, j, k) = E_mas(i, j, k);

            // Diagnostic specific-heat-ratio and ref-pressure (for plotfile only;
            // hyperbolic step uses per-phase constants directly, not these).
            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);

            // Mixture pressure (Schmidmayer 2020 eq. 8).  In mechanical
            // equilibrium IC this equals p_mix_IC (both per-phase pressures
            // are the same here).
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
    amrex::MultiFab &rho_eta0_rhs_mf,
    amrex::MultiFab &rho_eta1_rhs_mf,
    amrex::MultiFab &M_rhs_mf,
    amrex::MultiFab &E_rhs_mf,
    amrex::MultiFab &eta_rhs_mf,
    amrex::MultiFab &E0_rhs_mf,
    amrex::MultiFab &E1_rhs_mf,
    const amrex::MultiFab &rho_eta0_mf_in,
    const amrex::MultiFab &rho_eta1_mf_in,
    const amrex::MultiFab &M_mf_in,
    const amrex::MultiFab &E_mf_in,
    const amrex::MultiFab &eta_mf_in,
    const amrex::MultiFab &E0_mf_in,
    const amrex::MultiFab &E1_mf_in)
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

    // Primitive Fields (with BC) -- RelaxAndReinit is called inside
    // FillGhost4BC after STEP 4 primitive recovery (see comment block
    // there).  Was briefly called here at top of RHS but moved into
    // FillGhost4BC to match the user's 5-eq workflow ordering.
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

                Set::Vector grad_mag_grad_eta = Set::Vector(1 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                                                            1 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));

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
                    // Store curvature values
                    kappas(i, j, k, 0) = kappa;  // Mean or selected curvature
                    kappas(i, j, k, 1) = kappa1; // First principal curvature
                    kappas(i, j, k, 2) = kappa2; // Second principal curvature
                } 
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
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = hess_u_mf.Patch(lev, mfi);

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
            // d(eta)/dt = -u*grad(eta) + div( M*grad(mu) )
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            // Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0]; // Mob = u_max * epsilon
            Set::Scalar Mob = a(i, j, k) * epsilon;// *epsilon / DX[0]; // Mob = u_max * epsilon
            //Set::Scalar advection = -u.dot(grad_eta);
            Set::Scalar eta_dot_CH = Mob * lap_mu_chem * 0.2;

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Cahn-Hillard solving", {
                { "mu_chem",  mu_chem_(i, j, k) },
                { "lap_mu_chem",  lap_mu_chem },
                { "a",  a(i, j, k) },
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

            // ============================================================
            // 6-equation HLLC face fluxes (Saurel 2009 §3.1.2 /
            // Schmidmayer 2020 §3).
            // ============================================================
            const int X = 0, Y_dir = 1;

            // Build per-face State (6-eq).  EOS constants are per-phase and
            // identical L/R per cell (same eos0/eos1).
            auto make_state = [&](int ii, int jj, int kk, int dir)
                -> Solver::Local::FluidRiemann::State
            {
                Solver::Local::FluidRiemann::State s;
                s.alpha       = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                s.alpha_rho_0 = rho_eta0(ii, jj, kk);
                s.alpha_rho_1 = rho_eta1(ii, jj, kk);
                if (dir == X)
                {
                    s.M_normal  = M(ii, jj, kk, 0);
                    s.M_tangent = M(ii, jj, kk, 1);
                }
                else
                {
                    s.M_normal  = M(ii, jj, kk, 1);
                    s.M_tangent = M(ii, jj, kk, 0);
                }
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
            // Error Checking (must come before face reconstruction so we
            // catch NaN in the cell BEFORE limiter touches it).
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Conservative Variable Check", {
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
            // §3.2: PRIMITIVE-variable reconstruction; never conservatives).
            // For face between cell `lo` and cell `hi` in normal direction
            // `dir`, gather 6-cell window {lo-2, lo-1, lo, hi, hi+1, hi+2},
            // convert to primitives, and reconstruct Q_L (right edge of
            // `lo`) and Q_R (left edge of `hi`, via reversed-stencil trick).
            // Default Limiter=Godunov returns the cell-center value
            // unchanged -- equivalent to the original first-order flux.
            // ------------------------------------------------------------
            auto compute_face = [&](int lo_i, int lo_j, int hi_i, int hi_j, int dir)
                -> Solver::Local::FluidRiemann::Flux
            {
                const int di = hi_i - lo_i;
                const int dj = hi_j - lo_j;
                Solver::Local::Limiter::Primitive prim[6];
                for (int s = -2; s <= 3; ++s)
                {
                    Solver::Local::FluidRiemann::State raw =
                        make_state(lo_i + s * di, lo_j + s * dj, k, dir);
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
                return riemannsolver->Solve(sL_face, sR_face, pref, small, Spec_Vol);
            };

            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            try
            {
                flux_xlo = compute_face(i - 1, j,     i,     j,     X    );
                flux_xhi = compute_face(i,     j,     i + 1, j,     X    );
                flux_ylo = compute_face(i,     j - 1, i,     j,     Y_dir);
                flux_yhi = compute_face(i,     j,     i,     j + 1, Y_dir);
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN RIEMANN SOLVERS (6-eq)");
                Util::ParallelMessage(INFO, "lev=", lev, " i=", i, " j=", j);
                Util::Abort(INFO);
            }

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

            // div(alpha u) and div(u) (from S_M = flux.u_interface).
            const Set::Scalar div_uA_x = (flux_xhi.u_interface * a_face_xhi
                                        - flux_xlo.u_interface * a_face_xlo) / DX[0];
            const Set::Scalar div_uA_y = (flux_yhi.u_interface * a_face_yhi
                                        - flux_ylo.u_interface * a_face_ylo) / DX[1];
            const Set::Scalar div_u_x  = (flux_xhi.u_interface - flux_xlo.u_interface) / DX[0];
            const Set::Scalar div_u_y  = (flux_yhi.u_interface - flux_ylo.u_interface) / DX[1];
            const Set::Scalar div_u    = div_u_x + div_u_y;

            // ------------------------------------------------------------
            // Volume fraction (alpha_1) row -- Saurel 2009 line 836:
            //   d(alpha)/dt + u . grad(alpha) = 0
            //   discretized as -div(alpha u) + alpha_C div(u).
            // The 6-eq model REPLACES the 5-eq Kapila K-source with the
            // stiff relaxation source (handled in the post-stage hook).
            // ------------------------------------------------------------
            const Set::Scalar eta_advect = -(div_uA_x + div_uA_y) + eta(i, j, k) * div_u;
            eta_rhs(i, j, k) = eta_advect + eta_dot_Vap;

            // ------------------------------------------------------------
            // Phase-mass rows (pure conservation, no source from h):
            //   d(alpha rho)_k / dt + div((alpha rho)_k u) = 0
            // ------------------------------------------------------------
            const Set::Scalar rho_eta0_flux = (flux_xlo.mass0 - flux_xhi.mass0) / DX[0]
                                            + (flux_ylo.mass0 - flux_yhi.mass0) / DX[1];
            const Set::Scalar rho_eta1_flux = (flux_xlo.mass1 - flux_xhi.mass1) / DX[0]
                                            + (flux_ylo.mass1 - flux_yhi.mass1) / DX[1];

            rho_eta0_rhs(i, j, k) = rho_eta0_flux + Source(i, j, k, 0) * (eta(i, j, k))         + m_dot_Vap;
            rho_eta1_rhs(i, j, k) = rho_eta1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k))   - m_dot_Vap;

            // Diagnostic mass flux (kept for plotfile compatibility):
            rho_flux(i, j, k) = rho_eta0_flux + rho_eta1_flux;

            // ------------------------------------------------------------
            // Mixture momentum (pure conservation, with body sources):
            // ------------------------------------------------------------
            // In RHS we accumulate as (F_lo - F_hi)/dx.  X-direction:
            //   M[0] takes momentum_normal from x-faces and momentum_tangent
            //   from y-faces.
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal  - flux_xhi.momentum_normal ) / DX[0]
                               + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / DX[1];
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / DX[0]
                               + (flux_ylo.momentum_normal  - flux_yhi.momentum_normal ) / DX[1];

            M_rhs(i, j, k, 0) = M_flux(i, j, k, 0) + Source(i, j, k, 1);
            M_rhs(i, j, k, 1) = M_flux(i, j, k, 1) + Source(i, j, k, 2);

            // ------------------------------------------------------------
            // Redundant mixture total energy (pure conservation):
            //   d(rho E)/dt + div((rho E + p) u) = 0     (Sch20 eq. 16)
            // ------------------------------------------------------------
            E_flux(i, j, k) = (flux_xlo.energy_total - flux_xhi.energy_total) / DX[0]
                            + (flux_ylo.energy_total - flux_yhi.energy_total) / DX[1];
            E_rhs(i, j, k)  = E_flux(i, j, k) + Source(i, j, k, 3);

            // ------------------------------------------------------------
            // Per-phase internal energies (Sch20 eq. 13 last two rows /
            // Sau09 line 852):
            //   d(alpha rho e)_k/dt + div((alpha rho e)_k u)
            //                       + alpha_k_C p_k_C div(u) = 0
            // (The +/- mu p_I (p_1-p_2) relaxation source is deferred to
            //  the post-stage hook, in the stiff-relaxation limit.)
            // ------------------------------------------------------------
            const Set::Scalar E0_flux_div = (flux_xlo.energy0 - flux_xhi.energy0) / DX[0]
                                          + (flux_ylo.energy0 - flux_yhi.energy0) / DX[1];
            const Set::Scalar E1_flux_div = (flux_xlo.energy1 - flux_xhi.energy1) / DX[0]
                                          + (flux_ylo.energy1 - flux_yhi.energy1) / DX[1];

            E0_rhs(i, j, k) = E0_flux_div - a1_C * p0_C * div_u;
            E1_rhs(i, j, k) = E1_flux_div - a2_C * p1_C * div_u;

            // ------------------------------------------------------------
            // Artificial heat exchange (AHE) -- Schmidmayer 2020 eq. 13
            // r-source on per-phase internal energies:
            //     E0_rhs += -mu p_I (p_0 - p_1)
            //     E1_rhs += +mu p_I (p_0 - p_1)
            //
            // HYBRID (see bin/AHE.md): Sch20 does NOT specify a finite mu
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
                // Sau09 sec.5.3 curve fit.
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
                const Set::Scalar u_mag_lim = std::sqrt(  M(i, j, k, 0) * M(i, j, k, 0)
                                                       +  M(i, j, k, 1) * M(i, j, k, 1)) / rho_loc_lim;
                const Set::Scalar c_local = std::max(c0_C, c1_C);
                const Set::Scalar dx_min  = std::min(DX[0], DX[1]);
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

            // Calculate vorticity for visualization
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));
        });
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

    // Swapping pointers (6-eq primaries -- canonical set)
    std::swap(density_old_mf[lev],         density_mf[lev]);
    std::swap(momentum_old_mf[lev],        momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev],  energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev],  energy_per_mas_mf[lev]);
    std::swap(eta_old_mf,                  eta_mf);
    std::swap(rho_eta0_old_mf,             rho_eta0_mf);
    std::swap(rho_eta1_old_mf,             rho_eta1_mf);
    std::swap(energy0_old_mf,              energy0_mf);
    std::swap(energy1_old_mf,              energy1_mf);

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
    solution_new.emplace_back(*rho_eta0_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_eta1_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*eta_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy0_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy1_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*rho_eta0_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_eta1_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*eta_old_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy0_old_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy1_old_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        // rhs_mf:      [0]=rho_eta0_rhs, [1]=rho_eta1_rhs, [2]=M_rhs, [3]=E_rhs, [4]=eta_rhs, [5]=E0_rhs, [6]=E1_rhs
        // solution_mf: [0]=rho_eta0,     [1]=rho_eta1,     [2]=M,     [3]=E,     [4]=eta,     [5]=E0,     [6]=E1
        RHS(lev, time,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], rhs_mf[4], rhs_mf[5], rhs_mf[6],
            solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3], solution_mf[4], solution_mf[5], solution_mf[6]);
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

        // Clamp eta in domain prior to ghost fill (state can drift slightly outside [0,1])
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            });
        }

        // ===========================================================
        // 6-eq STIFF PRESSURE RELAXATION + REINITIALIZATION.
        // MOVED to top of RHS().  AMReX_RKIntegrator.H:217 only invokes
        // post_stage_action when stage index i > 0, so this lambda is
        // never called for forward-Euler and never after the final stage
        // of any RK scheme -- i.e., this whole block was dead code.
        // The probe message below confirmed it was never firing.
        // ===========================================================
        // Util::Message(INFO, "RelaxAndReinit -- vvvvv");  // dead probe
        // RelaxAndReinit(lev);                              // -> moved into RHS()

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

    });


    timeintegrator.advance(solution_old, solution_new, time, dt);

    // ENFORCE POSITIVITY after time advance
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Scalar small_local = small;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), small_local);
            rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), small_local);
        });
    }

    // ============================================================
    // POST-INTEGRATION PRIMITIVE REFRESH.
    // AMReX's RKIntegrator never re-evaluates RHS at the FINAL
    // assembled state (S_new = S_old + sum w_i dt F_i); the last
    // RHS call was at the last INTERMEDIATE stage state.  So all
    // diagnostic primitives -- pressure_mf, density_mf, velocity_mf,
    // pressure0/1_mf, density0/1_mf, T_mf, etc. -- are still those
    // of the last substage (e.g., t + dt/2 for SSPRK3, not t + dt).
    // Plotfiles written between Advance() calls would show this
    // stale pressure.  FillGhost4BC re-derives all primitives from
    // the current (final-step) conservative state.
    //
    // Side effect: RelaxAndReinit is invoked once more here (since
    // it lives inside FillGhost4BC).  This is essentially a no-op
    // when the state is already in mechanical equilibrium from the
    // last substage's relax (Newton converges in 0-1 iters), so
    // the cost is the ghost-fill work only.  It also enforces a
    // clean p_0 = p_1 in the final assembled state, which serves
    // as the start of the next step.
    // ============================================================
    FillGhost4BC(lev, time + dt);

    // ============================================================
    // AMR conservative reflux: accumulate post-step face fluxes
    // into the FluxRegister(s) on the (lev-1, lev) and (lev, lev+1)
    // boundaries. The actual Reflux correction is applied by
    // Integrator::TimeStep -> Hydro2::Reflux(lev) after the fine
    // level's subcycling completes.
    // ============================================================
    ComputeAndAccumulateFaceFluxes(lev, dt);


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

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
        const Set::Scalar gam0_ = eos0_local.Gamma();
        const Set::Scalar pi0_  = eos0_local.P0();
        const Set::Scalar gam1_ = eos1_local.Gamma();
        const Set::Scalar pi1_  = eos1_local.P0();
        const Set::Scalar alpha_floor = 1.0e-6;

        amrex::ParallelFor(bx, [=, &c_max_local, &vx_max_local, &vy_max_local, &F_max_local, &rho_min_local] AMREX_GPU_DEVICE(int i, int j, int k)
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

            Bm(i,j,k) = eta(i,j,k) / (1.0 - eta(i,j,k) + small);

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

    // Apply BC
    FillGhost4BC(lev, regrid_time);

    // ------------------------------------------------------------------
    // Allocate / re-allocate the AMR reflux machinery for THIS level.
    //
    // flux_reg[lev]   : holds the coarse/fine flux mismatch at the
    //                   boundary between (lev-1) coarse and lev fine.
    //                   Only meaningful for lev > 0.
    // face_flux_x/y_mf[lev] : scratch face-centered fluxes for the 5
    //                   conserved variables on level lev.
    //
    // Resize the per-level containers if this is the first Regrid we
    // see at this level, then (re-)build the structures using the new
    // BoxArray / DistributionMap for level lev.
    // ------------------------------------------------------------------
    const int nlevs_max = maxLevel() + 1;
    if ((int)flux_reg.size()        < nlevs_max) flux_reg.resize(nlevs_max);
    if ((int)face_flux_x_mf.size()  < nlevs_max) face_flux_x_mf.resize(nlevs_max);
    if ((int)face_flux_y_mf.size()  < nlevs_max) face_flux_y_mf.resize(nlevs_max);

    // Face-centered flux storage on level lev.
    {
        amrex::BoxArray ba_x = grids[lev]; ba_x.surroundingNodes(0);
        amrex::BoxArray ba_y = grids[lev]; ba_y.surroundingNodes(1);
        face_flux_x_mf[lev].define(ba_x, dmap[lev], n_conserved, 0);
        face_flux_y_mf[lev].define(ba_y, dmap[lev], n_conserved, 0);
        face_flux_x_mf[lev].setVal(0.0);
        face_flux_y_mf[lev].setVal(0.0);
    }

    // FluxRegister at the boundary between (lev-1) and lev.
    if (lev > 0)
    {
        flux_reg[lev].reset(new amrex::FluxRegister(grids[lev], dmap[lev],
                                                    refRatio(lev - 1),
                                                    lev, n_conserved));
        flux_reg[lev]->setVal(0.0);
    }

    Util::Message(INFO, "Regridding on level", lev);
}// end regrid

///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// AMR REFLUX (FluxRegister hooks) //////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// The Hydro2 integrator wires amrex::FluxRegister to fix non-conservation
// at coarse-fine boundaries. The base class Integrator::TimeStep calls
// Reflux(lev) AFTER the fine sub-stepping at level lev+1 has completed
// and BEFORE the average_down. Our Reflux applies the accumulated flux
// mismatch on the (lev, lev+1) boundary to the coarse conservative state
// at level lev for all 5 conserved variables.
//
// Flux accumulation:
//   - In Hydro2::Advance, after the RK time integration completes, we
//     evaluate the face fluxes ONCE from the final post-step state via
//     ComputeAndAccumulateFaceFluxes(lev, dt[lev]) and use them to
//     populate the FluxRegister(s). This is first-order-in-time for the
//     reflux correction (since we use one face evaluation rather than
//     properly weighted Butcher-tableau accumulation across RK stages),
//     but the underlying time integration retains its full RK order.
//     What matters for preventing droplet "spraying" at c-f boundaries
//     is divergence-form conservation, not the time-accuracy of the
//     correction itself.
//
// Flux registration convention (matches amrex::FluxRegister docs):
//   - When level lev finishes its Advance(), it acts as:
//       * the FINE level for flux_reg[lev]      (if lev > 0)
//         -> FineAdd of the just-computed level-lev face fluxes
//            scaled by +dt[lev] / dx_perp
//       * the COARSE level for flux_reg[lev+1]  (if lev < finest_level)
//         -> CrseInit of the just-computed level-lev face fluxes
//            scaled by -dt[lev] / dx_perp
//
//   The "dx_perp" division gives the correct units for a divergence
//   correction; amrex::FluxRegister handles this internally if we pass
//   the geometry.CellSize() factors as the scale argument convention.
//   Here we use the API form FineAdd(mf, dir, scomp, dcomp, ncomp, scale)
//   which expects scale = dt; the register applies the dx scaling at
//   Reflux time using the provided coarse geometry.
//
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Reflux(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Reflux");

    // Only fine levels have a flux register; lev here is the COARSE
    // level adjacent to the (just-completed) fine level lev+1.
    if (lev + 1 > finest_level) return;
    if ((int)flux_reg.size() <= lev + 1) return;
    if (!flux_reg[lev + 1])             return;

    // Apply the accumulated flux mismatch (fine - coarse) at the
    // (lev, lev+1) boundary to the COARSE conserved variables on
    // level lev. Component layout in the FluxRegister matches
    // ConservedIdx { rho_eta0, rho_eta1, M_x, M_y, E }.
    const Set::Scalar scale = 1.0 / geom[lev].CellSize(0); // dx normalisation
    (void)scale; // amrex::FluxRegister::Reflux uses geometry internally below.

    amrex::FluxRegister &fr = *flux_reg[lev + 1];

    // Reflux each conserved variable into its corresponding coarse MF.
    fr.Reflux(*rho_eta0_mf[lev],       1.0, IDX_RHO_ETA0, 0, 1, geom[lev]);
    fr.Reflux(*rho_eta1_mf[lev],       1.0, IDX_RHO_ETA1, 0, 1, geom[lev]);
    fr.Reflux(*momentum_mf[lev],       1.0, IDX_M_X,      0, 1, geom[lev]);
    fr.Reflux(*momentum_mf[lev],       1.0, IDX_M_Y,      1, 1, geom[lev]);
    fr.Reflux(*energy_per_vol_mf[lev], 1.0, IDX_E,        0, 1, geom[lev]);

    // Zero the register for the next coarse step.
    fr.setVal(0.0);

    // After the conservative state was corrected on level lev, the
    // mixture density and the per-phase mass primaries are out of
    // sync along the strip of coarse cells adjacent to the c-f
    // boundary. Re-derive primitives so the next Advance starts from
    // a consistent state on level lev.
    FillGhost4BC(lev, t_new[lev]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/// ComputeAndAccumulateFaceFluxes
///
/// Evaluate the face-centered Riemann fluxes for the 5 conserved variables
/// (alpha_rho_0, alpha_rho_1, M_x, M_y, E_total) using the CURRENT (post-RK)
/// state on level lev, and accumulate them into flux_reg[lev] (as the fine
/// contribution) and flux_reg[lev+1] (as the coarse contribution), each
/// scaled by `weight` (the level-lev dt).
///
/// The per-face Riemann solve mirrors the inline lambda inside RHS so the
/// numerical flux that crosses the c-f boundary in the FluxRegister matches
/// what the per-cell divergence used during the RK time integration. We
/// reuse the same EOS, limiter, and Riemann solver objects.
///
/// We deliberately compute fluxes ONCE per face (face-centered iteration
/// using surroundingNodes box arrays) rather than the per-cell 4x recompute
/// that RHS does today. This is purely for the FluxRegister accumulation;
/// the RHS hot path is unchanged.
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::ComputeAndAccumulateFaceFluxes(int lev, Set::Scalar weight)
{
    BL_PROFILE("Integrator::Hydro2::ComputeAndAccumulateFaceFluxes");

    // No fine/coarse partners -> nothing to register, but still cheap to
    // skip the work entirely.
    const bool has_fine_partner   = (lev < finest_level)
                                    && (int)flux_reg.size() > lev + 1
                                    && flux_reg[lev + 1];
    const bool has_coarse_partner = (lev > 0)
                                    && (int)flux_reg.size() > lev
                                    && flux_reg[lev];
    if (!has_fine_partner && !has_coarse_partner) return;

    if ((int)face_flux_x_mf.size() <= lev) return;
    if ((int)face_flux_y_mf.size() <= lev) return;

    // Make sure ghost cells of the conservative state are filled so the
    // face Riemann solves at the boundary of the valid box have proper
    // neighbors. FillGhost4BC was already called by Advance() before us;
    // a no-op rebuild here just guards against callers that didn't.

    face_flux_x_mf[lev].setVal(0.0);
    face_flux_y_mf[lev].setVal(0.0);

    const Set::Scalar small_local      = small;
    const Set::Scalar pref_local       = pref;
    const int         spec_vol_local   = Spec_Vol;
    Solver::Local::FluidRiemann::FluidRiemann *rs_local = riemannsolver;
    Solver::Local::Limiter::Limiter           *lim_local = limiter;
    const Solver::EOS::Tammann eos0_local = eos0;
    const Solver::EOS::Tammann eos1_local = eos1;

    // X-direction faces.
    for (amrex::MFIter mfi(face_flux_x_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // face-centered (x-nodal)

        auto eta_a       = eta_mf[lev]->array(mfi);
        auto rho_eta0_a  = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1_a  = rho_eta1_mf[lev]->array(mfi);
        auto M_a         = momentum_mf[lev]->array(mfi);
        auto E_a         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_a        = energy0_mf[lev]->array(mfi);
        auto E1_a        = energy1_mf[lev]->array(mfi);
        auto flux_x      = face_flux_x_mf[lev].array(mfi);

        amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
        {
            // Face at i sits between cell (i-1) and cell (i) in x-direction.
            auto make_state = [&](int ii, int jj, int kk) -> Solver::Local::FluidRiemann::State
            {
                Solver::Local::FluidRiemann::State s;
                s.alpha       = std::min(std::max(eta_a(ii, jj, kk), 0.0), 1.0);
                s.alpha_rho_0 = rho_eta0_a(ii, jj, kk);
                s.alpha_rho_1 = rho_eta1_a(ii, jj, kk);
                s.M_normal    = M_a(ii, jj, kk, 0);
                s.M_tangent   = M_a(ii, jj, kk, 1);
                s.E0          = E0_a(ii, jj, kk);
                s.E1          = E1_a(ii, jj, kk);
                s.E_total     = E_a(ii, jj, kk);
                s.gamma0      = eos0_local.Gamma();
                s.pi0         = eos0_local.P0();
                s.gamma1      = eos1_local.Gamma();
                s.pi1         = eos1_local.P0();
                return s;
            };

            Solver::Local::Limiter::Primitive prim[6];
            for (int s = -2; s <= 3; ++s)
            {
                Solver::Local::FluidRiemann::State raw = make_state(i - 1 + s, j, k);
                prim[s + 2] = Solver::Local::Limiter::ToPrimitive(raw, small_local);
            }
            Solver::Local::Limiter::Primitive stencil_L[5] =
                { prim[0], prim[1], prim[2], prim[3], prim[4] };
            Solver::Local::Limiter::Primitive stencil_R[5] =
                { prim[5], prim[4], prim[3], prim[2], prim[1] };
            Solver::Local::Limiter::Primitive pL = lim_local->Reconstruct(stencil_L);
            Solver::Local::Limiter::Primitive pR = lim_local->Reconstruct(stencil_R);
            Solver::Local::FluidRiemann::State sL = Solver::Local::Limiter::ToState(pL, small_local);
            Solver::Local::FluidRiemann::State sR = Solver::Local::Limiter::ToState(pR, small_local);

            Solver::Local::FluidRiemann::Flux f;
            try { f = rs_local->Solve(sL, sR, pref_local, small_local, spec_vol_local); }
            catch (...) { return; }

            flux_x(i, j, k, IDX_RHO_ETA0) = f.mass0;
            flux_x(i, j, k, IDX_RHO_ETA1) = f.mass1;
            flux_x(i, j, k, IDX_M_X)      = f.momentum_normal;
            flux_x(i, j, k, IDX_M_Y)      = f.momentum_tangent;
            flux_x(i, j, k, IDX_E)        = f.energy_total;
        });
    }

    // Y-direction faces.
    for (amrex::MFIter mfi(face_flux_y_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // face-centered (y-nodal)

        auto eta_a       = eta_mf[lev]->array(mfi);
        auto rho_eta0_a  = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1_a  = rho_eta1_mf[lev]->array(mfi);
        auto M_a         = momentum_mf[lev]->array(mfi);
        auto E_a         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_a        = energy0_mf[lev]->array(mfi);
        auto E1_a        = energy1_mf[lev]->array(mfi);
        auto flux_y      = face_flux_y_mf[lev].array(mfi);

        amrex::LoopOnCpu(bx, [=] (int i, int j, int k) noexcept
        {
            // For Y-direction faces, the Riemann solver expects M_normal=M_y,
            // M_tangent=M_x. We swap on input and unswap on the output.
            auto make_state = [&](int ii, int jj, int kk) -> Solver::Local::FluidRiemann::State
            {
                Solver::Local::FluidRiemann::State s;
                s.alpha       = std::min(std::max(eta_a(ii, jj, kk), 0.0), 1.0);
                s.alpha_rho_0 = rho_eta0_a(ii, jj, kk);
                s.alpha_rho_1 = rho_eta1_a(ii, jj, kk);
                s.M_normal    = M_a(ii, jj, kk, 1);
                s.M_tangent   = M_a(ii, jj, kk, 0);
                s.E0          = E0_a(ii, jj, kk);
                s.E1          = E1_a(ii, jj, kk);
                s.E_total     = E_a(ii, jj, kk);
                s.gamma0      = eos0_local.Gamma();
                s.pi0         = eos0_local.P0();
                s.gamma1      = eos1_local.Gamma();
                s.pi1         = eos1_local.P0();
                return s;
            };

            Solver::Local::Limiter::Primitive prim[6];
            for (int s = -2; s <= 3; ++s)
            {
                Solver::Local::FluidRiemann::State raw = make_state(i, j - 1 + s, k);
                prim[s + 2] = Solver::Local::Limiter::ToPrimitive(raw, small_local);
            }
            Solver::Local::Limiter::Primitive stencil_L[5] =
                { prim[0], prim[1], prim[2], prim[3], prim[4] };
            Solver::Local::Limiter::Primitive stencil_R[5] =
                { prim[5], prim[4], prim[3], prim[2], prim[1] };
            Solver::Local::Limiter::Primitive pL = lim_local->Reconstruct(stencil_L);
            Solver::Local::Limiter::Primitive pR = lim_local->Reconstruct(stencil_R);
            Solver::Local::FluidRiemann::State sL = Solver::Local::Limiter::ToState(pL, small_local);
            Solver::Local::FluidRiemann::State sR = Solver::Local::Limiter::ToState(pR, small_local);

            Solver::Local::FluidRiemann::Flux f;
            try { f = rs_local->Solve(sL, sR, pref_local, small_local, spec_vol_local); }
            catch (...) { return; }

            // Unswap normal/tangent so the FluxRegister sees (M_x, M_y).
            flux_y(i, j, k, IDX_RHO_ETA0) = f.mass0;
            flux_y(i, j, k, IDX_RHO_ETA1) = f.mass1;
            flux_y(i, j, k, IDX_M_X)      = f.momentum_tangent;
            flux_y(i, j, k, IDX_M_Y)      = f.momentum_normal;
            flux_y(i, j, k, IDX_E)        = f.energy_total;
        });
    }

    // ---- Accumulate into the FluxRegister(s). --------------------------
    // amrex::FluxRegister::FineAdd / CrseInit signature:
    //   (MultiFab const& flux, int dir, int srccomp, int dstcomp, int ncomp, Real scale)
    //
    // Sign convention (per AMReX docs):
    //   CrseInit takes (-1) * coarse flux into the register at scale = dt.
    //   FineAdd  adds (+1) * fine flux into the register at scale = dt.
    // Reflux then applies (fine - coarse) / dx_coarse to the coarse cells.
    if (has_coarse_partner)
    {
        // We are the FINE side of flux_reg[lev].
        flux_reg[lev]->FineAdd(face_flux_x_mf[lev], 0, 0, 0, n_conserved, weight);
        flux_reg[lev]->FineAdd(face_flux_y_mf[lev], 1, 0, 0, n_conserved, weight);
    }
    if (has_fine_partner)
    {
        // We are the COARSE side of flux_reg[lev+1].
        flux_reg[lev + 1]->CrseInit(face_flux_x_mf[lev], 0, 0, 0, n_conserved, weight,
                                    amrex::FluxRegister::ADD);
        flux_reg[lev + 1]->CrseInit(face_flux_y_mf[lev], 1, 0, 0, n_conserved, weight,
                                    amrex::FluxRegister::ADD);
    }
}

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

    // Vorticity criterion for refinement
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
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

    FillBoundariesWithBC(lev, 0.0, density_bc, { rho_eta0_mf[lev].get(), rho_eta1_mf[lev].get(), eta_mf[lev].get() });
    FillBoundariesWithBC(lev, 0.0, energy_bc, { eta_mf[lev].get() });

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

    // Fill Patches for regridding -- AMR coarse-fine ghost interpolation
    // for the FULL 6-eq conservative state.  Missing energy0_mf or
    // energy1_mf here causes the flux-loop make_state to pick up STALE
    // (unfilled) per-phase E_k ghost values at coarse-fine interfaces;
    // when a wave reaches a fine patch whose ghosts sit at the corner
    // of (coarse-fine) and (physical-boundary), the BC fills neighbor
    // ghosts on top of those garbage E_k cells -> garbage flux -> NaN.
    // Symptom: AMR-only NaNs in FillGhost4BC when waves hit the boundary,
    // BC-agnostic (Neumann and NSCBC both affected).  See bin/AMR_NaN.md
    // / memory feedback_amr_fillpatch_missing_E_k.md.
    if (lev > 0)
    {
        FillPatch(lev, time, rho_eta0_mf,        *rho_eta0_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, rho_eta1_mf,        *rho_eta1_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, momentum_mf,        *momentum_mf[lev],       *momentum_bc, 0);
        FillPatch(lev, time, energy_per_vol_mf,  *energy_per_vol_mf[lev], *energy_bc,   0);
        // Volume fraction eta is NOT a conservative flux quantity (it is
        // bounded in [0,1] and represents a sharp tanh interface). Using
        // cell_cons_interp on it produces over/undershoots at coarse-fine
        // boundaries that drive spurious surface-tension forces and cause
        // the droplet to "spray" instead of forming ligaments under shock
        // impact. Use piecewise-constant interpolation here so the bounds
        // and monotonicity of eta are preserved across c-f boundaries.
        FillPatch(lev, time, eta_mf,             *eta_mf[lev],            *energy_bc,   0,
                  &amrex::pc_interp);
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
    // STEP 3: Fill Eta ghost cells
    // ------------------------------------------------------------
    FillBoundariesWithBC(lev, time, energy_bc, {
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
        const Set::Scalar alpha_floor = 1.0e-6;

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
            v(i, j, k, 0) = M(i, j, k, 0) / std::max(rho(i, j, k), small);
            v(i, j, k, 1) = M(i, j, k, 1) / std::max(rho(i, j, k), small);

            // Kinetic energy.
            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0)
                                              + v(i, j, k, 1) * v(i, j, k, 1));

            // Pure-phase densities for per-phase EOS evaluation.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;

            // Per-phase pressures from canonical (alpha rho e)_k (Sau09 II.3).
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);

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
    // 6-eq STIFF PRESSURE RELAXATION + REINIT (Sch20 §3.3 / Sau09 §3.5)
    // Placed here per the user's 5-eq workflow: primitives are computed
    // first (STEP 4 above), then the Newton enforces p_0 = p_1 on the
    // canonical primaries (eta, E_0, E_1) before any source/flux work.
    //
    // CAVEAT: subsequent STEPs 5-9 below (NSCBC ghost fill, ghost-cell
    // primitive recovery, NaN repair, positivity) read pressure_mf /
    // pressure0_mf / pressure1_mf, which are now one-relax-step stale
    // relative to the post-relax conservatives.  Downstream RHS source
    // and flux loops re-derive p_k locally from E_k per cell so they
    // see the fresh state -- but if a future change makes a downstream
    // consumer rely on pressure_mf, refresh primitives after the relax.
    // ============================================================
    Util::Message(INFO, "FillGhost4BC pre-relax: relax_diag=", relax_diag, " lev=", lev);
    RelaxAndReinit(lev);

    // ------------------------------------------------------------
    // STEP 5: Fill CONSERVATIVE ghost cells (rho, M, E)
    // ------------------------------------------------------------
    if (use_nscbc)
    {
        // ====================================================================
        // NSCBC PATH: Use characteristic-based boundary conditions
        // ====================================================================

        // ====================================================================
        // 6-eq <-> NSCBC bridge.  Strategy: NSCBC operates on MIXTURE
        // quantities only (rho_total, M, E_total, gamma_mix, pi_mix,
        // pressure_mix) — exactly as in the validated 5-equation code path.
        // The per-phase 6-eq primaries ((alpha rho)_k, E_k) are reconstructed
        // here from (rho_total, eta, p_ghost) AFTER NSCBC finishes.  Eta is
        // a linearly-degenerate field at the boundary, so we extrapolate it
        // from the boundary cell into the NSCBC ghosts before NSCBC runs
        // (Expression::FillBoundary is a no-op for nscbc_* types and would
        // otherwise leave eta uninitialized).
        // ====================================================================

        const int ib_lo = geom[lev].Domain().smallEnd(0);
        const int ib_hi = geom[lev].Domain().bigEnd(0);
        const int jb_lo = geom[lev].Domain().smallEnd(1);
        const int jb_hi = geom[lev].Domain().bigEnd(1);
        const bool x_periodic = geom[lev].isPeriodic(0);
        const bool y_periodic = geom[lev].isPeriodic(1);

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
                if (!x_outside && !y_outside) return;                // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone

                const int ib = std::min(std::max(i, ib_lo), ib_hi);
                const int jb = std::min(std::max(j, jb_lo), jb_hi);
                eta(i, j, k) = eta(ib, jb, k);
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
                if (!x_outside && !y_outside) return;                // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone

                const Set::Scalar a1 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2 = 1.0 - a1;
                const Set::Scalar rho_g = std::max(rho(i, j, k), small_ns);
                const Set::Scalar p_g   = std::max(press(i, j, k), small_ns);

                rho0(i, j, k) = a1 * rho_g;
                rho1(i, j, k) = a2 * rho_g;
                E0(i, j, k)   = a1 * (p_g + g0_ns * pi0_ns) / std::max(g0_ns - 1.0, small_ns);
                E1(i, j, k)   = a2 * (p_g + g1_ns * pi1_ns) / std::max(g1_ns - 1.0, small_ns);
            });
        }

        // Update density_mf from rho_total
        amrex::MultiFab::Copy(*density_mf[lev], rho_total, 0, 0, 1, nghost);
    }
    else
    {
        // Phase masses (canonical).
        FillBoundariesWithBC(lev, time, density_bc, {
            rho_eta0_mf[lev].get(),
            rho_eta1_mf[lev].get()
        });
        // Mixture momentum.
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        // Redundant total energy AND per-phase internal energies (6-eq primaries
        // -- Schmidmayer 2020 eq. 13 last two rows + eq. 16).
        FillBoundariesWithBC(lev, time, energy_bc, {
            energy_per_vol_mf[lev].get(),
            energy0_mf[lev].get(),
            energy1_mf[lev].get()
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
        const Set::Scalar alpha_floor = 1.0e-6;

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            Set::Scalar arh0 = std::max(arh0_arr(i, j, k), small);
            Set::Scalar arh1 = std::max(arh1_arr(i, j, k), small);
            rho(i, j, k) = arh0 + arh1;

            v(i, j, k, 0) = M(i, j, k, 0) / std::max(rho(i, j, k), small);
            v(i, j, k, 1) = M(i, j, k, 1) / std::max(rho(i, j, k), small);

            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0)
                                              + v(i, j, k, 1) * v(i, j, k, 1));

            // Per-phase pure densities, pressures, and sound speeds.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);
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
            E(i, j, k)  = UE(i, j, k) + KE(i, j, k);

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


////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// RELAXATION + REINITIALIZATION //////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
//
// Stiff pressure-relaxation (mu -> infinity) + reinitialization step for the
// 6-equation diffuse-interface model.
//
//   Step 1 (relaxation, Schmidmayer 2020 §3.3, eq. (24)+(25); Saurel 2009 §3.3,
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
// Schmidmayer line 466-470: "performed at each stage. Thus there is only one
// pressure at the end of each stage."
//
void Hydro2::RelaxAndReinit(int lev)
{
    BL_PROFILE("Integrator::Hydro2::RelaxAndReinit");
    Util::Message(INFO, "RelaxAndReinit");


    // alpha_floor TIGHTENED to 1e-12 (was 1e-6) to match the Limiter
    // framework's convention.  At the old 1e-6 floor, pure-other-phase
    // cells saw rho_k = arh_k_floored / 1e-6 wildly different from the
    // physical pure-phase density, producing bad pre-relax sound speeds,
    // a wildly wrong impedance-weighted pHat0 initial guess, and Newton
    // divergence on ~400 interface cells per call (Sch20 Oscillating
    // post-mortem: see bin/log.txt -- residual=1330, |p_relaxed-p_reinit|
    // =102350 Pa, reinit then injected that 1e5 Pa perturbation into the
    // bubble dynamics every step).  DIV_FLOOR (1e-30) is the divide-by-
    // zero floor used in inlined EOS calls -- decoupled from the
    // integrator's `small` (~1e-8), which is sized for HLLC flux
    // positivity guards and is far too coarse to use as a divisor for
    // pure-phase-density inversion of canonical (alpha rho)_k.
    const Set::Scalar alpha_floor = 1.0e-12;
    const Set::Scalar DIV_FLOOR   = 1.0e-30;
    const Set::Scalar small_loc   = small;        // kept for legacy callers
    const int         max_iter    = 30;
    const Set::Scalar newton_tol  = 1.0e-10;
    const int         bisect_max  = 120;          // generous; AMR lev=4 needs headroom
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
    const bool diag_on = true; //(relax_diag != 0.0);
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

        // Diagnostic array (only used if diag_on).
        amrex::Array4<Set::Scalar> diag_arr = diag_on
            ? (*diag_mf)[mfi].array()
            : amrex::Array4<Set::Scalar>{};

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // -----------------------------------------------------------
            // Pre-relax state.  ALL divides use DIV_FLOOR (1e-30), NEVER
            // `small`.  Same lesson as the Limiter Garrick post-mortem:
            // using `small`=1e-8 as a divisor for canonical (alpha rho)_k
            // -> rho_k produces 10000x discontinuities in p_k between
            // cells whose alpha straddles the small floor, which then
            // destroys the Newton's initial guess and convergence.
            // -----------------------------------------------------------
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            // Conservative partial densities -- NO `small` floor here.
            Set::Scalar arh0_loc = std::max(arh0(i, j, k), 0.0);
            Set::Scalar arh1_loc = std::max(arh1(i, j, k), 0.0);

            // Pure-phase densities with DIV_FLOOR (1e-30).  In pure-other-
            // phase cells where alpha_k = alpha_floor (1e-12) and arh_k
            // is also ~1e-12 (consistent IC), this gives rho_k ~ physical
            // density.  Old code's `arh = max(arh,small=1e-8)` made
            // rho_k = 1e-8/1e-12 = 1e4 = garbage.
            Set::Scalar rho0_pre = arh0_loc / std::max(a1, DIV_FLOOR);
            Set::Scalar rho1_pre = arh1_loc / std::max(a2, DIV_FLOOR);

            // Pre-relax per-phase pressures (Sau09 eq. II.3 inverted) --
            // inlined so we control the divide-floor directly instead of
            // routing through EOS::PhasicPressureFromEnergy which uses
            // the integrator's `small` for BOTH divide-floor and SG
            // positivity floor (conflating two concerns).
            Set::Scalar p0_pre = (gam0 - 1.0) * E0_(i, j, k) / std::max(a1, DIV_FLOOR) - gam0 * pi0_;
            Set::Scalar p1_pre = (gam1 - 1.0) * E1_(i, j, k) / std::max(a2, DIV_FLOOR) - gam1 * pi1_;
            // SG positivity floor (Sau09 §3): p_k + pi_k > 0.
            p0_pre = std::max(p0_pre, -pi0_ + DIV_FLOOR);
            p1_pre = std::max(p1_pre, -pi1_ + DIV_FLOOR);

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
            Set::Scalar v0_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
            Set::Scalar v1_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
            Set::Scalar a1_new = arh0_loc * v0_r;
            Set::Scalar a2_new = arh1_loc * v1_r;
            Set::Scalar asum   = a1_new + a2_new;
            if (asum > DIV_FLOOR) { a1_new /= asum; a2_new /= asum; }
            a1_new = std::min(std::max(a1_new, alpha_floor), 1.0 - alpha_floor);
            a2_new = 1.0 - a1_new;

            // -----------------------------------------------------------
            // Reinit: recompute mixture p from the (CONSERVED) redundant
            // rho E using the POST-relaxation alpha_k.
            // Schmidmayer 2020 eq. 26 / Saurel 2009 eq. III.5.
            // -----------------------------------------------------------
            Set::Scalar rho_loc = arh0_loc + arh1_loc;
            Set::Scalar ke      = 0.5 * (M_(i, j, k, 0) * M_(i, j, k, 0)
                                       + M_(i, j, k, 1) * M_(i, j, k, 1)) / std::max(rho_loc, small_loc);
            Set::Scalar rho_e   = std::max(E_(i, j, k) - ke, small_loc);

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
                        // Count as truly unconverged only if iters hit
                        // the max AND |f| exceeds the physical-engineering
                        // tolerance.  Bisection routinely lands at
                        // |f| ~ 1e-8 -- physically converged; flagging
                        // those with the old newton_tol=1e-10 produces
                        // spurious "unconverged" reports on AMR runs.
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
} // end RelaxAndReinit()




} // end of Integrator namespace

//#endif
