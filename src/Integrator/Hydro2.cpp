// Base
#include "Hydro2.H"
// Parsing and Input Handeling
#include "AMReX_MultiFab.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "Numeric/Stencil.H"
#include "IC/Constant.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
#include "AMReX_Geometry.H"
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


// Limiters
//#include "Solver/Local/Limiter/Minmod.H"
//#include "Solver/Local/Limiter/VanLeer.H"

#include <AMReX_Math.H>
#include "AMReX_TimeIntegrator.H"
#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MFIter.H>
#include <AMReX_Array4.H>
#include "Thermo_Interp.H"
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H>


using namespace amrex;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"

//#if AMREX_SPACEDIM == 2

namespace Integrator
{

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
        pp_query_default("cutoff", value.cutoff, 1.0E-8);   // eta cutoff value
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
        pp_query_required("gamma0", value.gamma0);      // gamma for gamma law
        pp_query_default("pi_0", value.pi_0, 0.0);      // pi for Tammann EOS
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient
        pp_query_default("cp0", value.cp0, 0.0);        // Constant Pressure Specific Heat [J/kg]
        pp_query_default("cv0", value.cv0, 0.0);        // Constant Volume Specific Heat [J/kg]
        // pp_query_required("R0", value.R0);              // Specific Gas Constant
        // pp_query_required("MW0", value.MW0);            // Molecular Weight

        // FLUID 1
        pp_query_required("gamma1", value.gamma1);      // gamma for gamma law
        pp_query_default("pi_1", value.pi_1, 0.0);      // pi for Tammann EOS
        pp_query_required("mu1", value.mu1);            // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0);    // bulk viscosity coefficient
        pp_query_default("cp1", value.cp1, 0.0);        // Constant Pressure Specific Heat [J/kg]
        pp_query_default("cv1", value.cv1, 0.0);        // Constant Volume Specific Heat [J/kg]
        // pp_query_required("R1", value.R1);              // Specific Gas Constant
        // pp_query_required("MW1", value.MW1);            // Molecular Weight

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0);    // Surface tension condition
        pp_query_default("Dv", value.Dv, 0.0);          // Vapor Diffusivity
        pp_query_required("epsilon", value.epsilon);    // diffuse interface thickness Y_infinity
        pp_query_default("Y_infinity", value.Y_infinity, 0.0); // Far Field Vapor Mass Fraction

        // CURVATURE
        pp_query_default("kappa_method",        value.kappa_method,        1); // 1:"Smooth Normals" (default)  2:"Height Function"  3:"Hybrid"
        pp_query_default("smooth_kernel_size",  value.smooth_kernel_size,  3); // Gaussian normal-smoothing kernel: 3 (3x3) or 5 (5x5)
        pp_query_default("nghost",              value.nghost,              2); // Ghost cell depth: controls HF column integration band and stencil reach

        // IMPLICIT CAHN-HILLIARD
        // 0 = explicit, 1 = Eyre double-Helmholtz MLMG, 2 = Newton-MLMG (full nonlinear)
        pp_query_default("implicit_ch",      value.implicit_ch,      0);        // 0: explicit  1: Eyre MLMG  2: Newton-MLMG
        pp_query_default("ch_mobility_nom",  value.ch_mobility_nom,  0.0);      // Nominal mobility [m^2/s] (0 = auto: 0.2*ε*c_max)
        pp_query_default("ch_newton_iters",  value.ch_newton_iters,  5);        // Max Newton iterations (implicit_ch=2 only)
        pp_query_default("ch_newton_tol",    value.ch_newton_tol,    1.0e-10);  // Newton convergence tolerance
        pp_query_default("ch_W_scale",       value.ch_W_scale,       1.0);      // double-well amplitude (equilibrium width = √2·ε/√W_scale)

        // Boundry Conditions
        value.density_bc = new BC::Expression(1, pp, "density.bc");
        value.energy_bc = new BC::Constant(1, pp, "energy.bc");
        value.momentum_bc = new BC::Expression(2, pp, "momentum.bc");
        //value.eta_bc = new BC::Constant(1, pp, "pf.eta.bc");
        value.temperature_bc = new BC::Constant(1, pp, "energy.bc"); // Change to be different if needed? ___TEMP___
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,          value.energy_bc, 1, nghost, "eta", true, true);
        value.RegisterNewFab(value.eta_old_mf,      value.energy_bc, 1, nghost, "eta_old", false, true);
        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing, 1, nghost, "etadot", true, false);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing, 4, nghost, "hess_eta", false, false, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing,  2, nghost, "n_hat", false, false, { "x", "y" });

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

        value.RegisterNewFab(value.pressure0_mf,    value.energy_bc,  1, nghost, "pressure0", false, true);
        value.RegisterNewFab(value.velocity0_mf,    &value.bc_nothing,  2, nghost, "velocity0", false, false, { "x", "y" });
        value.RegisterNewFab(value.vorticity0_mf,   &value.bc_nothing,  1, nghost, "vorticity0", false, false);

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
        value.RegisterNewFab(value.vorticity1_mf,   &value.bc_nothing,  1, nghost, "vorticity1", false, true);

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,     value.energy_bc, 1, nghost, "pressure", true, true);
        value.RegisterNewFab(value.velocity_mf,     &value.bc_nothing,  2, nghost, "velocity", true, false, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf,    &value.bc_nothing,  1, nghost, "vorticity", true, false);
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true, true);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false, true);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost, "energy_per_vol", true, true);
        value.RegisterNewFab(value.energy_per_mass_mf,       value.energy_bc,    1, nghost, "energy_per_mass", true, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost, "energy_vol_old", false, true);
        value.RegisterNewFab(value.energy_per_mass_old_mf,   value.energy_bc,    1, nghost, "energy_mass_old", false, true);
        value.RegisterNewFab(value.momentum_mf,     value.momentum_bc,  2, nghost, "momentum", true, true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc,  2, nghost, "momentum_old", false, true, { "x", "y" });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing,  1, nghost, "m0", false, false);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing, 2, nghost, "u0", false, false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing, 2, nghost, "q0", false, false, { "x", "y" });
        value.RegisterNewFab(value.Source_mf,       &value.bc_nothing,  4, nghost, "Source", true, false, { "_rho", "_Mx", "_My","_E" });
        value.RegisterNewFab(value.Fsv_mf,          &value.bc_nothing,  2, nghost, "Fsv", true, false, { "x", "y" });  // Surface Tension
        value.RegisterNewFab(value.Fw_mf,           &value.bc_nothing,  2, nghost, "Fw", true, false, { "x", "y" });   // Weight
        value.RegisterNewFab(value.Ldot_mf,         &value.bc_nothing,  2, nghost, "Ldot", true, false, { "x", "y" });  // Ldot
        value.RegisterNewFab(value.T_mf,            value.energy_bc,  1, nghost, "T", true, false);                  // Temperature
        value.RegisterNewFab(value.cp_mf,           &value.bc_nothing,  1, nghost, "cp", false, true);         // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,           &value.bc_nothing,  1, nghost, "cv", false, true);         // Constant Volume Specific Heat
        //value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false, true);         // Thermal Conductivity
        //value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false, true);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc, 1, nghost, "gamma", true, false);                 // Specific Heat Ratio
        value.RegisterNewFab(value.pi_mf,   value.energy_bc, 1, nghost, "Tamann_pi", true, true);                    // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc, 1, nghost, "mu_chem", true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true, false);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, false, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   &value.bc_nothing,  1, nghost, "UE_per_vol", true, false);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mass_mf,   &value.bc_nothing,  1, nghost, "UE_per_mass", true, false);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   &value.bc_nothing,  1, nghost, "KE_per_vol", true, false);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mass_mf,   &value.bc_nothing,  1, nghost, "KE_per_mass", true, false);        // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,           &value.bc_nothing,  1, nghost, "Spadling_Number", true, false);    // Spalding Number
        value.RegisterNewFab(value.Y_mf,            &value.bc_nothing,  1, nghost, "Mass_Fraction", true, false);       // Mass Fraction

        // Kappa and curvature related fields
        value.RegisterNewFab(value.kappa_HF_mf,            &value.bc_nothing,  1, nghost, "kappa_HF", true, false);
        value.RegisterNewFab(value.kappa_SF_mf,            &value.bc_nothing,  1, nghost, "kappa_SF", true, false);
        value.RegisterNewFab(value.h_eta_mf,               &value.bc_nothing,  1, nghost, "h_eta", true, false);
        value.RegisterNewFab(value.nx_smoothed_mf,         &value.bc_nothing,  1, nghost, "nx_smoothed", true, false);
        value.RegisterNewFab(value.ny_smoothed_mf,         &value.bc_nothing,  1, nghost, "ny_smoothed", true, false);
        value.RegisterNewFab(value.gradmag_mf,             &value.bc_nothing,  1, nghost, "gradmag", true, false);
        value.RegisterNewFab(value.eta_x_mf,               &value.bc_nothing,  1, nghost, "eta_x", true, false);
        value.RegisterNewFab(value.eta_y_mf,               &value.bc_nothing,  1, nghost, "eta_y", true, false);

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, nghost, "grad_eta", false, false, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, false, { "Active", "HF", "SN" }); // Active=selected method, HF=height function, SN=smooth normals (raw div)
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", false, false, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,     &value.bc_nothing,  1, nghost, "rho_flux", true, false);                    // Density Flux
        value.RegisterNewFab(value.M_flux_mf,       &value.bc_nothing,  2, nghost, "M_flux", true, false, { "x", "y" });        // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,       &value.bc_nothing,  1, nghost, "E_flux", true, false);                      // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,      &value.bc_nothing,  2, nghost, "div_tau", true, false, { "x", "y" });
        value.RegisterNewFab(value.tau_mf,          &value.bc_nothing,  3, nghost, "tau", false, false, { "xx", "xy", "yy" }); // viscous stress tensor
        value.RegisterNewFab(value.hess_u_mf,       &value.bc_nothing,  8, nghost, "hess_u", false, false, {
                                                                                                     "000","001",
                                                                                                     "010","011",
                                                                                                     "100","101",
                                                                                                     "110","111",
                                                                                                    }); // hess_u Flux
        value.RegisterNewFab(value.Vap_dot_mf, &value.bc_nothing, 5, nghost, "Vap_dot", true, false, { "_eta", "_rho", "_Mx", "_My", "_E" }); // Momentum Flux



    }

    // INITIAL CONDITIONS
    // Eta
    pp.select_default<IC::Constant,IC::Laminate,IC::Expression,IC::BMP,IC::PNG>("eta.ic",value.eta_ic,value.geom);
    // Fluid 0
    pp.select_default<IC::Constant,IC::Expression>("velocity0.ic",      value.velocity0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure0.ic",      value.pressure0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density0.ic",       value.density0_ic,  value.geom);
    // pp.select_default<IC::Constant,IC::Expression>("energy0.ic",       value.energy0_ic,  value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("temperature0.ic",  value.temperature0_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("k0_thermal.ic",    value.k0_thermal_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h0_thermal_ic, value.geom);


    // Fluid 1
    pp.select_default<IC::Constant,IC::Expression>("velocity1.ic",      value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure1.ic",      value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density1.ic",       value.density1_ic,  value.geom);
    // pp.select_default<IC::Constant,IC::Expression>("energy1.ic",       value.energy1_ic,  value.geom);
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
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro,
                      Solver::Local::FluidRiemann::Upwind,
                      Solver::Local::FluidRiemann::Lax_Friedrich
    >("Riemann_Solver", value.riemannsolver);
    Util::Message(INFO, "Selected Riemann solver: ", typeid(*value.riemannsolver).name());


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

    // -----------------------------------------------------------------------
    // GC-NSCBC parameters (Motheau 2017)
    // -----------------------------------------------------------------------
    pp_query_default("nscbc.enabled", value.nscbc.enabled, false); // Enable GC-NSCBC boundary conditions

    if (value.nscbc.enabled)
    {
        using FT = Hydro2::NSCBCParams::FaceType;
        auto parse_face_type = [](const std::string& s) -> FT {
            if (s == "outflow") return FT::OUTFLOW;
            if (s == "inflow")  return FT::INFLOW;
            return FT::NONE;
        };

        pp_query_default("nscbc.sigma", value.nscbc.sigma, 0.25); // Outflow pressure relaxation strength (Motheau sigma)
        pp_query_default("nscbc.beta",  value.nscbc.beta,  0.0);  // Transverse term weight: 0 = full Lodato (recommended), 1 = pure LODI
        pp_query_default("nscbc.Lx",    value.nscbc.Lx,    0.0);  // x-domain length for K (0 = auto from geometry)
        pp_query_default("nscbc.Ly",    value.nscbc.Ly,    0.0);  // y-domain length for K (0 = auto from geometry)

        // xlo face
        { std::string s = "none"; pp.query("nscbc.xlo", s); // Face type: none | outflow | inflow
          value.nscbc.face_type[0] = parse_face_type(s); }
        pp_query_default("nscbc.xlo.p_target",  value.nscbc.p_t[0],      0.0);   // Target pressure at xlo [Pa]
        pp_query_default("nscbc.xlo.un_target", value.nscbc.un_t[0],     0.0);   // Target normal velocity at xlo [m/s]
        pp_query_default("nscbc.xlo.ut_target", value.nscbc.ut_t[0],     0.0);   // Target tangential velocity at xlo [m/s]
        pp_query_default("nscbc.xlo.T_target",  value.nscbc.T_t[0],    300.0);   // Target temperature at xlo [K]
        pp_query_default("nscbc.xlo.eta_relax", value.nscbc.eta_relax[0], 2.0);  // Inflow relaxation factor at xlo

        // xhi face
        { std::string s = "none"; pp.query("nscbc.xhi", s); // Face type: none | outflow | inflow
          value.nscbc.face_type[1] = parse_face_type(s); }
        pp_query_default("nscbc.xhi.p_target",  value.nscbc.p_t[1],      0.0);   // Target pressure at xhi [Pa]
        pp_query_default("nscbc.xhi.un_target", value.nscbc.un_t[1],     0.0);   // Target normal velocity at xhi [m/s]
        pp_query_default("nscbc.xhi.ut_target", value.nscbc.ut_t[1],     0.0);   // Target tangential velocity at xhi [m/s]
        pp_query_default("nscbc.xhi.T_target",  value.nscbc.T_t[1],    300.0);   // Target temperature at xhi [K]
        pp_query_default("nscbc.xhi.eta_relax", value.nscbc.eta_relax[1], 2.0);  // Inflow relaxation factor at xhi

        // ylo face
        { std::string s = "none"; pp.query("nscbc.ylo", s); // Face type: none | outflow | inflow
          value.nscbc.face_type[2] = parse_face_type(s); }
        pp_query_default("nscbc.ylo.p_target",  value.nscbc.p_t[2],      0.0);   // Target pressure at ylo [Pa]
        pp_query_default("nscbc.ylo.un_target", value.nscbc.un_t[2],     0.0);   // Target normal velocity at ylo [m/s]
        pp_query_default("nscbc.ylo.ut_target", value.nscbc.ut_t[2],     0.0);   // Target tangential velocity at ylo [m/s]
        pp_query_default("nscbc.ylo.T_target",  value.nscbc.T_t[2],    300.0);   // Target temperature at ylo [K]
        pp_query_default("nscbc.ylo.eta_relax", value.nscbc.eta_relax[2], 2.0);  // Inflow relaxation factor at ylo

        // yhi face
        { std::string s = "none"; pp.query("nscbc.yhi", s); // Face type: none | outflow | inflow
          value.nscbc.face_type[3] = parse_face_type(s); }
        pp_query_default("nscbc.yhi.p_target",  value.nscbc.p_t[3],      0.0);   // Target pressure at yhi [Pa]
        pp_query_default("nscbc.yhi.un_target", value.nscbc.un_t[3],     0.0);   // Target normal velocity at yhi [m/s]
        pp_query_default("nscbc.yhi.ut_target", value.nscbc.ut_t[3],     0.0);   // Target tangential velocity at yhi [m/s]
        pp_query_default("nscbc.yhi.T_target",  value.nscbc.T_t[3],    300.0);   // Target temperature at yhi [K]
        pp_query_default("nscbc.yhi.eta_relax", value.nscbc.eta_relax[3], 2.0);  // Inflow relaxation factor at yhi
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();
    const Box& domain = geom.Domain();

    // Initialize individual fluid variables
    // DIFFUSE BOUNDARY
    eta_ic          ->Initialize(lev, eta_mf, 0.0);
    eta_ic          ->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]  ->setVal(0.0);
    hess_eta_mf[lev]->setVal(0.0);

    // FLUID 0
    velocity0_ic    ->Initialize(lev, velocity0_mf, 0.0);
    pressure0_ic    ->Initialize(lev, pressure0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_old_mf, 0.0);

    // FLUID 1
    velocity1_ic    ->Initialize(lev, velocity1_mf, 0.0);
    pressure1_ic    ->Initialize(lev, pressure1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_old_mf, 0.0);

    // FORCED SOURCE
    ic_m0           ->Initialize(lev, m0_mf, 0.0);
    ic_u0           ->Initialize(lev, u0_mf, 0.0);
    ic_q            ->Initialize(lev, q_mf, 0.0);

    // Debugging: verify IC values before mixing
    amrex::Print() << "p0 AFTER IC, BEFORE MIX: "
                << pressure0_mf[lev]->min(0) << " "
                << pressure0_mf[lev]->max(0) << "\n";
    amrex::Print() << "p1 AFTER IC, BEFORE MIX: "
                << pressure1_mf[lev]->min(0) << " "
                << pressure1_mf[lev]->max(0) << "\n";
    amrex::Print() << "rho0 AFTER IC, BEFORE MIX: "
                << density0_mf[lev]->min(0) << " "
                << density0_mf[lev]->max(0) << "\n";
    amrex::Print() << "eta AFTER IC, BEFORE MIX: "
                << eta_mf[lev]->min(0) << " "
                << eta_mf[lev]->max(0) << "\n";
    amrex::Print() << "eta NaN after IC (incl ghost): " << eta_mf[lev]->contains_nan() << "\n";
    amrex::Print() << "eta NaN valid-only after IC:   " << eta_mf[lev]->contains_nan(0,1,0) << "\n";

    // Calculate mixed variables based on individual fluid variables
    Mix(lev);
    amrex::Print() << "eta NaN after Mix (incl ghost):" << eta_mf[lev]->contains_nan() << "\n";

    // Make sure ghost cells are consistent with initial conditions
    eta_mf[lev]->FillBoundary(geom.periodicity());
    density_mf[lev]->FillBoundary(geom.periodicity());
    momentum_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_vol_mf[lev]->FillBoundary(geom.periodicity());
    amrex::Print() << "eta NaN after FillBoundary (incl ghost): " << eta_mf[lev]->contains_nan() << "\n";

    // NATURAL SOURCE
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Fw_mf[lev]      ->setVal(0.0);
    Ldot_mf[lev]    ->setVal(0.0);
    Vap_dot_mf[lev] ->setVal(0.0);
    tau_mf[lev]     ->setVal(0.0);

    // BOUNDRY CURVATURE AND THINGS
    kappas_mf[lev]  ->setVal(0.0);
    grad_mag_grad_eta_mf[lev]->setVal(0.0);
    Bm_mf[lev]      ->setVal(0.0);  // Spalding Number

    // MIXED PROPERTIES (setVal also clears ghost cells, preventing NaN from debug-mode allocation)
    mu_chem_mf[lev]     ->setVal(0.0);
    T_mf[lev]           ->setVal(0.0);
    a_mf[lev]           ->setVal(0.0);
    Ma_mf[lev]          ->setVal(0.0);
    UE_per_vol_mf[lev]  ->setVal(0.0);
    UE_per_mass_mf[lev] ->setVal(0.0);
    KE_per_vol_mf[lev]  ->setVal(0.0);
    KE_per_mass_mf[lev] ->setVal(0.0);

    Util::ParallelMessage(INFO, "Finished initialization, begginning time iteration");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    BL_PROFILE("Hydro2::Mix");

    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();

    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // Inputs
        auto eta    = eta_mf[lev]->const_array(mfi);

        auto rho0   = density0_mf[lev]->const_array(mfi);
        auto rho1   = density1_mf[lev]->const_array(mfi);

        auto v0     = velocity0_mf[lev]->const_array(mfi);
        auto v1     = velocity1_mf[lev]->const_array(mfi);

        auto p0     = pressure0_mf[lev]->const_array(mfi);
        auto p1     = pressure1_mf[lev]->const_array(mfi);

        // Outputs (phase energies)
        auto E0     = energy0_mf[lev]->array(mfi);
        auto E1     = energy1_mf[lev]->array(mfi);

        // Mixture fields
        auto rho      = density_mf[lev]->array(mfi);
        auto rho_old  = density_old_mf[lev]->array(mfi);

        auto M        = momentum_mf[lev]->array(mfi);
        auto M_old    = momentum_old_mf[lev]->array(mfi);

        auto E_vol      = energy_per_vol_mf[lev]->array(mfi);
        auto E_vol_old  = energy_per_vol_old_mf[lev]->array(mfi);

        auto E_mass     = energy_per_mass_mf[lev]->array(mfi);
        auto E_mass_old = energy_per_mass_old_mf[lev]->array(mfi);

        auto gamma_mix = gamma_mf[lev]->array(mfi);
        auto pi_mix    = pi_mf[lev]->array(mfi);
        auto p_mix     = pressure_mf[lev]->array(mfi);

        auto v_mix     = velocity_mf[lev]->array(mfi);

        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i,int j,int k)
        {
            Real e = eta(i,j,k);

            //--------------------------------------------------------------
            // 1. Compute energy of each phase from pressure (your request)
            //--------------------------------------------------------------

            // ---------------- Phase 0 ----------------
            Real r0 = rho0(i,j,k);
            Real u0x = v0(i,j,k,0);
            Real u0y = v0(i,j,k,1);

            Real KE0 = 0.5 * r0 * (u0x*u0x + u0y*u0y);
            Real UE0 = (p0(i,j,k) + gamma0*pi_0) / (gamma0 - 1.0);
            if (UE0 < 0) UE0 = 0;

            E0(i,j,k) = KE0 + UE0;       // total energy per volume


            // ---------------- Phase 1 ----------------
            Real r1 = rho1(i,j,k);
            Real u1x = v1(i,j,k,0);
            Real u1y = v1(i,j,k,1);

            Real KE1 = 0.5 * r1 * (u1x*u1x + u1y*u1y);
            Real UE1 = (p1(i,j,k) + gamma1*pi_1) / (gamma1 - 1.0);
            if (UE1 < 0) UE1 = 0;

            E1(i,j,k) = KE1 + UE1;       // total energy per volume


            //--------------------------------------------------------------
            // 2. Mixture density (linear in η)
            //--------------------------------------------------------------
            Real rm = e*r0 + (1.0-e)*r1;
            rho(i,j,k)     = rm;
            rho_old(i,j,k) = rm;


            //--------------------------------------------------------------
            // 3. Mixture momentum (conservative mixing)
            //--------------------------------------------------------------
            Real Mx = e*r0*u0x + (1.0-e)*r1*u1x;
            Real My = e*r0*u0y + (1.0-e)*r1*u1y;

            M(i,j,k,0)      = Mx;
            M(i,j,k,1)      = My;
            M_old(i,j,k,0)  = Mx;
            M_old(i,j,k,1)  = My;


            //--------------------------------------------------------------
            // 4. Mixture velocity
            //--------------------------------------------------------------
            Real vx = Mx / rm;
            Real vy = My / rm;

            v_mix(i,j,k,0) = vx;
            v_mix(i,j,k,1) = vy;


            //--------------------------------------------------------------
            // 5. Mixture total energy (linear in η)
            //--------------------------------------------------------------
            Real Ev = e*E0(i,j,k) + (1.0-e)*E1(i,j,k);

            E_vol(i,j,k)      = Ev;
            E_vol_old(i,j,k)  = Ev;

            Real Em = Ev / rm;
            E_mass(i,j,k)     = Em;
            E_mass_old(i,j,k) = Em;


            //--------------------------------------------------------------
            // 6. Mixture gamma and pi (use your interpolation function)
            //--------------------------------------------------------------
            double gmix, pimix;
            Thermo_Interp::InterpolateGammaPi_Stiffened(
                e, gamma1, gamma0, pi_1, pi_0, gmix, pimix);

            gamma_mix(i,j,k) = gmix;
            pi_mix(i,j,k)    = pimix;


            //--------------------------------------------------------------
            // 7. Mixture pressure from stiffened EOS (energy-primary)
            //--------------------------------------------------------------
            Real KE = 0.5 * rm * (vx*vx + vy*vy);
            Real UE = Ev - KE;
            if (UE < 0) UE = 0;

            Real p = (gmix - 1.0)*UE - gmix*pimix + pref;
            if (p < 0) p = 1e-6;

            p_mix(i,j,k) = p;
        });
    }

    // Fill ghost cells
    density_mf[lev]->FillBoundary(geom.periodicity());
    momentum_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_vol_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_mass_mf[lev]->FillBoundary(geom.periodicity());
    pressure_mf[lev]->FillBoundary(geom.periodicity());
    velocity_mf[lev]->FillBoundary(geom.periodicity());
    gamma_mf[lev]->FillBoundary(geom.periodicity());
    pi_mf[lev]->FillBoundary(geom.periodicity());

    // Compute mixture vorticity: ω = ∂v/∂x − ∂u/∂y
    // Use boundary-aware stencils to avoid reading uninitialized ghost cells
    // at non-periodic domain boundaries.
    {
        const Box domain = geom.Domain();
        for (MFIter mfi(*vorticity_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.validbox();
            auto v_arr = velocity_mf[lev]->const_array(mfi);
            auto omega = vorticity_mf[lev]->array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                auto sten = Numeric::GetStencil(i, j, k, domain);
                Set::Matrix grad_v = Numeric::Gradient(v_arr, i, j, k, DX, sten);
                omega(i, j, k) = grad_v(1, 0) - grad_v(0, 1); // dv/dx - du/dy
            });
        }
    }
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
            amrex::MultiFab &eta_rhs_mf,
            amrex::MultiFab &rho_rhs_mf,
            amrex::MultiFab &M_rhs_mf,
            amrex::MultiFab &E_rhs_mf,
            const amrex::MultiFab &eta_mf_in,
            const amrex::MultiFab &rho_mf_in,
            const amrex::MultiFab &M_mf_in,
            const amrex::MultiFab &E_mf_in)
{
    BL_PROFILE("Hydro2::RHS");

    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();
    const Box& domain = geom.Domain();

    //---------------------------------------------------------------------------
    // 0. Curvature pipeline — always run; kappa_method selects which result
    //    goes into kappas_mf component 0 (used by the surface tension force).
    //    Component 1 = HF, Component 2 = SN (raw) are always stored for output.
    //---------------------------------------------------------------------------
    ComputeKappas(lev);

    //---------------------------------------------------------------------------
    // 1. FIRST LOOP: compute primitive fields and geometry-dependent things
    //---------------------------------------------------------------------------
    for (MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // CONSERVATIVE (ALL Array4)
        auto eta_arr = eta_mf_in.array(mfi);
        auto rho_arr = rho_mf_in.array(mfi);
        auto M_arr   = M_mf_in.array(mfi);
        auto E_arr   = E_mf_in.array(mfi);

        // PRIMITIVE/OUTPUT
        auto v_arr     = velocity_mf[lev]->array(mfi);
        auto p_arr     = pressure_mf[lev]->array(mfi);
        auto a_arr     = a_mf[lev]->array(mfi);

        auto KE_arr    = KE_per_vol_mf[lev]->array(mfi);
        auto UE_arr    = UE_per_vol_mf[lev]->array(mfi);

        auto cp_arr    = cp_mf[lev]->array(mfi);
        auto cv_arr    = cv_mf[lev]->array(mfi);
        auto T_arr     = T_mf[lev]->array(mfi);

        auto gamma_mix = gamma_mf[lev]->array(mfi);
        auto pi_mix    = pi_mf[lev]->array(mfi);

        auto mu_arr    = mu_chem_mf[lev]->array(mfi);
        auto Bm_arr    = Bm_mf[lev]->array(mfi);
        auto Y_arr     = Y_mf[lev]->array(mfi);

        auto kappas_arr = kappas_mf[lev]->array(mfi);

        auto rho0_arr = density0_mf[lev]->array(mfi);
        auto rho1_arr = density1_mf[lev]->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            //------------------------------------------------------------------
            // REBUILD PRIMITIVES
            //------------------------------------------------------------------
            const Real eta = eta_arr(i,j,k);
            const Real rho = rho_arr(i,j,k);
            const Real mx  = M_arr(i,j,k,0);
            const Real my  = M_arr(i,j,k,1);

            // Velocity — guard against rho=0 (e.g. freshly created AMR cells or
            // cells where the density update produced an exact zero).
            // Without this, vx = mx/0 = NaN (if mx=0) or Inf (if mx≠0), which
            // propagates to KE = 0*Inf = NaN and then to pressure = NaN,
            // corrupting the DynamicTimestep CFL computation.
            Real safe_rho = (rho > Real(1e-20)) ? rho : Real(1e-20);
            Real vx = mx / safe_rho;
            Real vy = my / safe_rho;
            v_arr(i,j,k,0) = vx;
            v_arr(i,j,k,1) = vy;

            // Kinetic Energy
            Real KE = 0.5 * rho * (vx*vx + vy*vy);
            KE_arr(i,j,k) = KE;

            // Internal Energy — guard against NaN/Inf and runaway accumulation
            Real E = E_arr(i,j,k);
            Real UE = E - KE;
            if (!std::isfinite(UE) || UE < 0.0) UE = 0.0;
            if (UE > Real(1e20)) UE = Real(1e20); // cap runaway energy
            UE_arr(i,j,k) = UE;

            //------------------------------------------------------------------
            // Gamma-law mixture + Tammann EOS
            //------------------------------------------------------------------
            Real A =   eta/(gamma0-1.0)
                     + (1.0 - eta)/(gamma1-1.0);

            Real B = ( eta * gamma0 * pi_0 )/(gamma0-1.0)
                   + ((1.0 - eta) * gamma1 * pi_1)/(gamma1-1.0);

            // Real gmix = 1.0 + 1.0/A;
            // gamma_mix(i,j,k) = gmix;

            // // Tammann pressure offset
            // Real pimix = (B/A)/gmix;
            // pi_mix  (i,j,k) = pimix;

            double gmix, pimix;
            Thermo_Interp::InterpolateGammaPi_Stiffened(
                eta, gamma1, gamma0, pi_1, pi_0, gmix, pimix);
            gamma_mix(i,j,k) = gmix;
            pi_mix(i,j,k) = pimix;

            // Pressure — clamp both from below (non-negative) and from above
            // (prevents T overflow during AMR interpolation / FillCoarsePatch).
            Real p_mix_local = (gmix - 1.0)*UE - gmix*pimix + pref;
            if (!std::isfinite(p_mix_local) || p_mix_local < 0.0) p_mix_local = 1e-6;
            if (p_mix_local > Real(1e15)) p_mix_local = Real(1e15); // upper cap
            p_arr(i,j,k) = p_mix_local;

            // Specific heats
            cp_arr(i,j,k) = eta*cp0 + (1.0 - eta)*cp1;
            cv_arr(i,j,k) = eta*cv0 + (1.0 - eta)*cv1;

            // Temperature — guard against Inf/NaN and unrealistically large values
            // (large T can overflow to Inf during AMR FillCoarsePatch interpolation).
            {
                Set::Scalar T_denom = rho * cv_arr(i,j,k) * (gmix - 1.0) + 1e-14;
                if (T_denom <= Real(0.0)) T_denom = Real(1e-14);
                Set::Scalar T_val = (p_mix_local + pimix) / T_denom;
                if (!std::isfinite(T_val) || T_val > Real(1e15)) T_val = Real(0.0);
                T_arr(i,j,k) = T_val;
            }

            // Speed of sound — use safe_rho to prevent astronomical a when rho→0,
            // which would collapse DynamicTimestep to dt_min indefinitely.
            a_arr(i,j,k) = std::sqrt(gmix * (p_mix_local + pimix) / safe_rho);

            //------------------------------------------------------------------
            // Chemical potential + mass fraction
            //------------------------------------------------------------------
            Real gm_eta = Numeric::Gradient(eta_arr, i,j,k, 0, DX).lpNorm<2>();
            Real lap_eta = Numeric::Laplacian(eta_arr, i,j,k, 0, DX);

            Real fprime = ch_W_scale * 4.0 * eta * (eta-0.5) * (eta-1.0);
            mu_arr(i,j,k) = -epsilon*epsilon*lap_eta + fprime;

            Y_arr(i,j,k) = rho0_arr(i,j,k)*eta / (rho + 1e-14);

            Bm_arr(i,j,k) = (Y_arr(i,j,k) - Y_infinity) / (1.0 + Y_infinity + 1e-14);
        });
    }

    //---------------------------------------------------------------------------
    // Fill ghost cells of derived fields computed in the first loop so that
    // neighbor accesses in the second loop (e.g. Laplacian of mu_chem) don't
    // read stale or NaN ghost cells at coarse-fine boundaries.
    //---------------------------------------------------------------------------
    mu_chem_mf[lev]->FillBoundary(geom.periodicity());

    // Fill ghost cells of EOS fields computed in the first loop.
    // The Riemann solver in the second loop reads neighbor cells (i±1, j±1)
    // of gamma_mf and pi_mf.  Without this exchange, those ghost cells hold
    // stale values from the previous timestep, giving wrong wave speeds and
    // pressures at every MPI rank boundary.
    gamma_mf[lev]->FillBoundary(geom.periodicity());
    pi_mf[lev]->FillBoundary(geom.periodicity());
    T_mf[lev]->FillBoundary(geom.periodicity());

    //---------------------------------------------------------------------------
    // 1b. PASS A: compute viscous stress tensor tau at every cell.
    //     tau is stored in tau_mf, then FillBoundary'd so that Pass B can
    //     differentiate it with a single first-order FD — avoiding hess_u
    //     and explicit grad(mu) terms entirely.
    //---------------------------------------------------------------------------
    for (MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const Box& bx    = mfi.validbox();
        auto eta_arr_tau = eta_mf_in.array(mfi);
        auto v_arr_tau   = velocity_mf[lev]->array(mfi);
        auto tau_arr     = tau_mf[lev]->array(mfi);

        Real mu0_    = mu0;
        Real mu1_    = mu1;
        Real mu0_b_  = mu0_b;
        Real mu1_b_  = mu1_b;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Real eta_loc    = eta_arr_tau(i,j,k);
            Real mu_eff     = eta_loc * mu0_  + (1.0 - eta_loc) * mu1_;
            Real lambda_eff = eta_loc * mu0_b_ + (1.0 - eta_loc) * mu1_b_;

            Set::Matrix gradu = Numeric::Gradient(v_arr_tau, i, j, k, DX, sten);
            Real div_u = gradu(0,0) + gradu(1,1);
            Real bulk  = lambda_eff - (2.0/3.0)*mu_eff;

            tau_arr(i,j,k,0) = 2.0*mu_eff*gradu(0,0) + bulk*div_u;           // tau_xx
            tau_arr(i,j,k,1) = mu_eff*(gradu(0,1) + gradu(1,0));              // tau_xy = tau_yx
            tau_arr(i,j,k,2) = 2.0*mu_eff*gradu(1,1) + bulk*div_u;           // tau_yy
        });
    }
    tau_mf[lev]->FillBoundary(geom.periodicity());

    //---------------------------------------------------------------------------
    // 2. SECOND LOOP: compute fluxes using Riemann solver
    //---------------------------------------------------------------------------
    for (MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // CONSERVATIVE
        auto eta_arr = eta_mf_in.array(mfi);
        auto rho_arr = rho_mf_in.array(mfi);
        auto M_arr   = M_mf_in.array(mfi);
        auto E_arr   = E_mf_in.array(mfi);

        // Primitive arrays
        auto v_arr     = velocity_mf[lev]->array(mfi);
        auto p_arr     = pressure_mf[lev]->array(mfi);
        auto a_arr     = a_mf[lev]->array(mfi);

        auto gamma_mix = gamma_mf[lev]->array(mfi);
        auto pi_mix    = pi_mf[lev]->array(mfi);
        auto T_arr     = T_mf[lev]->array(mfi);

        // Output
        auto eta_rhs = eta_rhs_mf.array(mfi);
        auto rho_rhs = rho_rhs_mf.array(mfi);
        auto M_rhs   = M_rhs_mf.array(mfi);
        auto E_rhs   = E_rhs_mf.array(mfi);

        // Source terms
        auto S_arr       = Source_mf[lev]->array(mfi);
        auto kappas_arr  = kappas_mf[lev]->array(mfi);
        auto mu_chem_arr = mu_chem_mf[lev]->array(mfi);
        auto Bm_arr      = Bm_mf[lev]->array(mfi);
        auto Fsv_arr     = Fsv_mf[lev]->array(mfi);
        auto Fw_arr      = Fw_mf[lev]->array(mfi);
        auto Ldot_arr    = Ldot_mf[lev]->array(mfi);
        auto Vap_dot_arr = Vap_dot_mf[lev]->array(mfi);
        auto div_tau_arr = div_tau_mf[lev]->array(mfi);
        auto hess_u_arr  = hess_u_mf[lev]->array(mfi);
        auto tau_arr     = tau_mf[lev]->const_array(mfi);
        auto m0_arr      = m0_mf[lev]->array(mfi);
        auto u0_arr      = u0_mf[lev]->array(mfi);
        auto q0_arr      = q_mf[lev]->array(mfi);

        // For diagnostic fields / vaporization
        auto rho0_arr = density0_mf[lev]->array(mfi);
        auto rho1_arr = density1_mf[lev]->array(mfi);

        // Capture implicit_ch as a plain bool so the GPU lambda can use it
        bool do_implicit_ch = implicit_ch;

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            //------------------------------------------------------------------
            // Neighbors — read directly including ghost cells.
            // Ghost cells at physical boundaries are filled by ApplyNSCBC (when
            // NSCBC is enabled) or by FillBoundary via the standard BC objects.
            // Ghost cells at MPI rank boundaries are filled by FillBoundary.
            // No clamping: the NSCBC ghost cells must be read by the Riemann
            // solver for the BC to have any effect on the solution.
            //------------------------------------------------------------------
            int il = i-1;
            int ir = i+1;
            int jl = j-1;
            int jr = j+1;

            //------------------------------------------------------------------
            // Build Riemann states ONLY from Array4 fields
            //------------------------------------------------------------------

            namespace FR = Solver::Local::FluidRiemann;

            FR::State Sxm = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                il, j, k, 0);    // x-direction

            FR::State Sxc = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                i, j, k, 0);

            FR::State Sxp = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                ir, j, k, 0);

            FR::State Sym = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                i, jl, k, 1);   // y-direction

            FR::State Syc = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                i, j, k, 1);

            FR::State Syp = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_mix, pi_mix  , T_arr,
                i, jr, k, 1);

            //------------------------------------------------------------------
            // No limiter: take cell-centered as left/right states
            //------------------------------------------------------------------
            FR::State xl = Sxm;
            FR::State xr = Sxc;

            FR::State yl = Sym;
            FR::State yr = Syc;

            FR::State xr_hi = Sxc;
            FR::State xl_hi = Sxp;

            FR::State yr_hi = Syc;
            FR::State yl_hi = Syp;

            //------------------------------------------------------------------
            // Solve Riemann problems
            //------------------------------------------------------------------
            FR::Flux fxm = riemannsolver->Solve(xl, xr, pref, small, Spec_Vol);
            FR::Flux fym = riemannsolver->Solve(yl, yr, pref, small, Spec_Vol);

            FR::Flux fxp = riemannsolver->Solve(xr_hi, xl_hi, pref, small, Spec_Vol);
            FR::Flux fyp = riemannsolver->Solve(yr_hi, yl_hi, pref, small, Spec_Vol);

            //------------------------------------------------------------------
            // Divergence of fluxes
            //------------------------------------------------------------------
            Real drho = (fxm.mass - fxp.mass)/DX[0]
                      + (fym.mass - fyp.mass)/DX[1];

            Real dMx  = (fxm.momentum_normal - fxp.momentum_normal)/DX[0]
                      + (fym.momentum_tangent - fyp.momentum_tangent)/DX[1];

            Real dMy  = (fxm.momentum_tangent - fxp.momentum_tangent)/DX[0]
                      + (fym.momentum_normal - fyp.momentum_normal)/DX[1];

            Real dE   = (fxm.energy - fxp.energy)/DX[0]
                      + (fym.energy - fyp.energy)/DX[1];

            //------------------------------------------------------------------
            // Source terms
            //------------------------------------------------------------------
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Real eta_loc    = eta_arr(i,j,k);
            // Guard against rho=0
            Real rho_loc    = amrex::max(rho_arr(i,j,k), Real(1.0e-14));
            // Guard against Inf/NaN velocity (e.g. from a prior step with near-zero rho
            // that was fixed by safe_rho; Inf*0=NaN in grad/source terms)
            Real vx_loc = v_arr(i,j,k,0);  if (!std::isfinite(vx_loc)) vx_loc = Real(0.0);
            Real vy_loc = v_arr(i,j,k,1);  if (!std::isfinite(vy_loc)) vy_loc = Real(0.0);
            Set::Vector u   = Set::Vector(vx_loc, vy_loc);
            Set::Vector u0v = Set::Vector(u0_arr(i,j,k,0), u0_arr(i,j,k,1));

            Set::Vector grad_eta    = Numeric::Gradient(eta_arr, i, j, k, 0, DX, sten);
            Real grad_eta_mag       = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta    = Numeric::Hessian(eta_arr, i, j, k, 0, DX, sten);

            // External source fields
            Real mdot0 = -m0_arr(i,j,k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Vector q0_ = Set::Vector(q0_arr(i,j,k,0), q0_arr(i,j,k,1));
            Real qdot0 = q0_.dot(grad_eta);

            // Viscous stress divergence: differentiate the pre-computed tau field.
            // This avoids explicit grad(mu) and hess_u terms; variable viscosity
            // is captured naturally through the tau values at neighboring cells.
            Set::Vector grad_tau_xx = Numeric::Gradient(tau_arr, i, j, k, 0, DX, sten);
            Set::Vector grad_tau_xy = Numeric::Gradient(tau_arr, i, j, k, 1, DX, sten);
            Set::Vector grad_tau_yy = Numeric::Gradient(tau_arr, i, j, k, 2, DX, sten);

            Set::Vector div_tau;
            div_tau(0) = grad_tau_xx(0) + grad_tau_xy(1);
            div_tau(1) = grad_tau_xy(0) + grad_tau_yy(1);

            div_tau_arr(i,j,k,0) = div_tau(0);
            div_tau_arr(i,j,k,1) = div_tau(1);

            // hess_u is no longer computed; zero out diagnostic field
            for (int c = 0; c < 8; c++) hess_u_arr(i,j,k,c) = 0.0;

            // Interface Lagrangian term (Ldot) — needs local M tensor and hess_eta
            Set::Vector Ldot = Set::Vector::Zero();
            Real mu_eff     = eta_loc * mu0  + (1.0 - eta_loc) * mu1;
            Real lambda_eff = eta_loc * mu0_b + (1.0 - eta_loc) * mu1_b;
            for (int p = 0; p < 2; p++)
                for (int q = 0; q < 2; q++)
                    for (int r = 0; r < 2; r++)
                        for (int s = 0; s < 2; s++)
                        {
                            Real Mpqrs = 0.0;
                            if ((p==r) && (q==s)) Mpqrs += mu_eff;
                            if ((p==s) && (q==r)) Mpqrs += mu_eff;
                            if ((p==q) && (r==s)) Mpqrs += lambda_eff - (2.0/3.0)*mu_eff;
                            Ldot(p) += 0.5 * Mpqrs * (u(r) - u0v(r)) * hess_eta(q,s);
                        }

            Ldot_arr(i,j,k,0) = Ldot(0);
            Ldot_arr(i,j,k,1) = Ldot(1);

            // Surface tension: F_sv = sigma * kappa * grad(eta) * epsilon
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);
            if (apply_surface_tension && grad_eta_mag > 0.0)
            {
                Real kappa = kappas_arr(i,j,k,0);
                Fsv_vector(0) = sigma * kappa * grad_eta(0) * epsilon;
                Fsv_vector(1) = sigma * kappa * grad_eta(1) * epsilon;
            }
            Fsv_arr(i,j,k,0) = Fsv_vector(0);
            Fsv_arr(i,j,k,1) = Fsv_vector(1);

            // Gravity: F_w = -rho * g (downward)
            Set::Vector Fw_vector = Set::Vector(0.0, 0.0);
            if (apply_weight)
            {
                Fw_vector(0) = 0.0;
                Fw_vector(1) = -rho_loc * g;
            }
            Fw_arr(i,j,k,0) = Fw_vector(0);
            Fw_arr(i,j,k,1) = Fw_vector(1);

            // Vaporization (Spalding model)
            Real eta_dot_Vap = 0.0;
            Real m_dot_Vap   = 0.0;
            Set::Vector M_dot_Vap = Set::Vector(0.0, 0.0);
            Real E_dot_Vap   = 0.0;
            if (apply_vaporization == 1)
            {
                Real B_M      = Bm_arr(i,j,k);
                Real rho_eta0 = eta_loc * rho_loc;
                Real rho_g    = rho_eta0 / std::max(eta_loc, 1e-14);
                Real vap_coeff = (rho_g * Dv / (rho_eta0 + 1e-14)) * (B_M / (1.0 + B_M + 1e-14));
                eta_dot_Vap = (1.0 / epsilon) * vap_coeff * grad_eta_mag;
                m_dot_Vap   = rho_g * Dv * (B_M / (1.0 + B_M + 1e-14));
                M_dot_Vap   = u * m_dot_Vap;
                E_dot_Vap   = u.dot(M_dot_Vap);
            }
            Vap_dot_arr(i,j,k,0) = eta_dot_Vap;
            Vap_dot_arr(i,j,k,1) = m_dot_Vap;
            Vap_dot_arr(i,j,k,2) = M_dot_Vap(0);
            Vap_dot_arr(i,j,k,3) = M_dot_Vap(1);
            Vap_dot_arr(i,j,k,4) = E_dot_Vap;

            // Total body force
            Set::Vector Total_Force = Fsv_vector + Fw_vector;

            // Assemble source vector
            S_arr(i,j,k,0) = mdot0 + m_dot_Vap;
            S_arr(i,j,k,1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0) + M_dot_Vap(0);
            S_arr(i,j,k,2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1) + M_dot_Vap(1);
            S_arr(i,j,k,3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force) + E_dot_Vap;

            // Lagrange no-penetration enforcement
            S_arr(i,j,k,1) -= lagrange * u.dot(grad_eta) * grad_eta(0);
            S_arr(i,j,k,2) -= lagrange * u.dot(grad_eta) * grad_eta(1);

            //------------------------------------------------------------------
            // Apply to RHS
            // Guard flux divergences and source terms against NaN/Inf: a NaN
            // entering the time advance produces NaN in conserved variables which
            // then propagates indefinitely.  Setting the RHS to 0 at those cells
            // leaves them unchanged this step — incorrect but bounded.
            //------------------------------------------------------------------
            if (!std::isfinite(drho))          drho = Real(0.0);
            if (!std::isfinite(dMx))           dMx  = Real(0.0);
            if (!std::isfinite(dMy))           dMy  = Real(0.0);
            if (!std::isfinite(dE))            dE   = Real(0.0);
            if (!std::isfinite(S_arr(i,j,k,0))) S_arr(i,j,k,0) = Real(0.0);
            if (!std::isfinite(S_arr(i,j,k,1))) S_arr(i,j,k,1) = Real(0.0);
            if (!std::isfinite(S_arr(i,j,k,2))) S_arr(i,j,k,2) = Real(0.0);
            if (!std::isfinite(S_arr(i,j,k,3))) S_arr(i,j,k,3) = Real(0.0);
            rho_rhs(i,j,k) = drho + S_arr(i,j,k,0);
            M_rhs(i,j,k,0) = dMx  + S_arr(i,j,k,1);
            M_rhs(i,j,k,1) = dMy  + S_arr(i,j,k,2);
            E_rhs(i,j,k)   = dE   + S_arr(i,j,k,3);

            //------------------------------------------------------------------
            // ETA equation: advection + Cahn-Hilliard + vaporization
            // When implicit_ch is enabled, the biharmonic term is handled
            // after the explicit step via ApplyImplicitCH(); only advection
            // and vaporization remain in the explicit RHS.
            //------------------------------------------------------------------
            Real adv = -(u(0)*grad_eta(0) + u(1)*grad_eta(1));

            Real eta_dot_CH = 0.0;
            if (!do_implicit_ch) {
                Real lap_mu_chem = Numeric::Laplacian(mu_chem_arr, i, j, k, 0, DX, sten);
                Real Mob         = a_arr(i,j,k) * epsilon;
                eta_dot_CH       = Mob * lap_mu_chem * 0.2;
            }

            eta_rhs(i,j,k) = adv + eta_dot_CH + eta_dot_Vap;
        });
    }
}

// IMPLICIT CAHN-HILLIARD BIHARMONIC
///////////////////////////////////////////////////////////////////////////////////////////////////////
// ApplyImplicitCH: operator-split implicit treatment of the CH biharmonic.
//
// The CH eta equation is:  ∂η/∂t = M(∇²f'(η) − κ∇⁴η)
// After the explicit advance gives η*, we solve (implicitly for the biharmonic):
//
//   (I + dt·M·κ·∇⁴) η^{n+1} = η* + dt·M·∇²f'(η*)
//
// Using the approximate double-Helmholtz factorization with γ = sqrt(dt·M·κ):
//
//   (I + γ·(−∇²)) ψ       = η* + dt·M·∇²f'(η*)   [Solve 1]
//   (I + γ·(−∇²)) η^{n+1} = ψ                      [Solve 2]
//
// Each solve is a standard Helmholtz equation handled by MLABecLaplacian + MLMG.
// Combined operator: (I + γ(−∇²))² = I + 2γ(−∇²) + γ²∇⁴, which approximates
// I + γ²∇⁴ = I + dt·M·κ·∇⁴ with an O(dt) error — acceptable for a 1st-order integrator.
// Both factors are symmetric positive definite, so the method is unconditionally stable.
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared setup helper used by both ApplyImplicitCH modes:
//   - BC arrays for MLMG (Neumann at walls, periodic where applicable)
//   - Face-centred b-coefficient arrays (uniform = 1)
//   - solveHelmholtz lambda: solves (I + γ(−∇²)) sol = rhs via MLABecLaplacian+MLMG
//
// Returns gamma_CH = sqrt(dt · M · κ).
// Callers pass the lambda by reference into their own scope.
// ------------------------------------------------------------------
// NOTE: this is a file-local helper, not a member function.
static Real CH_SetupAndSolveHelmholtz(
    int lev,
    const Geometry& geom,
    const amrex::BoxArray& ba,
    const amrex::DistributionMapping& dm,
    Real gamma_CH,
    Array<LinOpBCType, AMREX_SPACEDIM>& lo_bc,
    Array<LinOpBCType, AMREX_SPACEDIM>& hi_bc,
    Array<MultiFab, AMREX_SPACEDIM>& bcoef,
    MultiFab& sol,
    const MultiFab& f_rhs)
{
    LPInfo info;
    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);
    mlabec.setMaxOrder(2);
    mlabec.setDomainBC(lo_bc, hi_bc);
    mlabec.setLevelBC(0, &sol);
    mlabec.setScalars(1.0, gamma_CH);
    mlabec.setACoeffs(0, 1.0);
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));
    MLMG mlmg(mlabec);
    mlmg.setMaxIter(200);
    mlmg.setMaxFmgIter(0);
    mlmg.setVerbose(0);
    mlmg.solve({&sol}, {&f_rhs}, 1.0e-11, 0.0);
    return 0.0; // unused return, struct for reuse
}

// Fills a multifab of μ = f'(η) − ε²∇²η on valid cells, then fills boundaries.
// eta_mf must already have valid ghost cells before calling.
static void CH_ComputeMu(
    MultiFab& mu_mf,
    const MultiFab& eta_mf_in,
    BC::BC<Set::Scalar>* energy_bc,
    const Geometry& geom,
    const Real* DX,
    const Box& domain,
    Real kappa_CH,
    Real W_scale)
{
    for (MFIter mfi(eta_mf_in, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto eta = eta_mf_in.const_array(mfi);
        auto mu  = mu_mf.array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten  = Numeric::GetStencil(i, j, k, domain);
            Real e     = eta(i,j,k,0);
            Real fp    = W_scale * 4.0 * e * (e - 0.5) * (e - 1.0);
            Real lap_e = Numeric::Laplacian(eta, i, j, k, 0, DX, sten);
            mu(i,j,k,0) = fp - kappa_CH * lap_e;
        });
    }
    energy_bc->FillBoundary(mu_mf, 0, 1, 0.0, 0);
    mu_mf.FillBoundary(geom.periodicity());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// ApplyImplicitCH  (implicit_ch = 1)
//
// Single-step Eyre-split implicit biharmonic:
//   (I + dt·M·ε²·∇⁴) η^{n+1} ≈ η* + dt·M·∇²f'(η*)
// via the double-Helmholtz factorization
//   (I + γ(−∇²))² ≈ I + γ²∇⁴  where γ = sqrt(dt·M·ε²).
//
// Ghost-cell fix applied: f_prime ghost cells are filled via energy_bc before
// computing ∇²f', so boundary-adjacent cells don't read uninitialized data.
///////////////////////////////////////////////////////////////////////////////////////////////////////

/// Zero-gradient extrapolation for any NaN ghost cells not reached by FillBoundary
/// (e.g. INT_DIR/NSCBC faces where physbc does nothing).
/// Reads from the nearest valid cell of the same fab — equivalent to Neumann BC.
static void FillNaNGhostsZeroGrad(amrex::MultiFab& mf)
{
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const amrex::Box gbx = mfi.fabbox();
        const amrex::Box vbx = mfi.validbox();
        auto arr = mf.array(mfi);
        int  nc  = mf.nComp();
        amrex::ParallelFor(gbx, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
            bool is_valid = AMREX_D_TERM(
                (i >= vbx.smallEnd(0) && i <= vbx.bigEnd(0)),
             && (j >= vbx.smallEnd(1) && j <= vbx.bigEnd(1)),
             && (k >= vbx.smallEnd(2) && k <= vbx.bigEnd(2)));
            if (!is_valid && !std::isfinite(arr(i,j,k,n))) {
                int ic = amrex::max(vbx.smallEnd(0), amrex::min(vbx.bigEnd(0), i));
                int jc = amrex::max(vbx.smallEnd(1), amrex::min(vbx.bigEnd(1), j));
#if AMREX_SPACEDIM == 3
                int kc = amrex::max(vbx.smallEnd(2), amrex::min(vbx.bigEnd(2), k));
#else
                int kc = k;
#endif
                arr(i,j,k,n) = arr(ic,jc,kc,n);
            }
        });
    }
}

void Hydro2::ApplyImplicitCH(int lev, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ApplyImplicitCH");

    const Geometry& geom   = this->geom[lev];
    const Real*     DX     = geom.CellSize();
    const Box&      domain = geom.Domain();
    const int nghost = eta_mf[lev]->nGrow();

    const Real kappa_CH = epsilon * epsilon;
    const Real M_nom    = (ch_mobility_nom > 0.0) ? ch_mobility_nom
                                                   : 0.2 * epsilon * (c_max + 1.0);
    const Real gamma_CH = std::sqrt(dt * M_nom * kappa_CH);

    // BC arrays (Neumann at walls)
    Array<LinOpBCType, AMREX_SPACEDIM> lo_bc, hi_bc;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bc[d] = geom.isPeriodic(d) ? LinOpBCType::Periodic : LinOpBCType::Neumann;
        hi_bc[d] = geom.isPeriodic(d) ? LinOpBCType::Periodic : LinOpBCType::Neumann;
    }

    // Face-centred b-coefficient arrays (b = 1 everywhere)
    Array<MultiFab, AMREX_SPACEDIM> bcoef;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        BoxArray ba = eta_mf[lev]->boxArray(); ba.surroundingNodes(d);
        bcoef[d].define(ba, eta_mf[lev]->DistributionMap(), 1, 0);
        bcoef[d].setVal(1.0);
    }

    auto solveHelmholtz = [&](MultiFab& sol, const MultiFab& f_rhs) {
        CH_SetupAndSolveHelmholtz(lev, geom, eta_mf[lev]->boxArray(),
                                  eta_mf[lev]->DistributionMap(), gamma_CH,
                                  lo_bc, hi_bc, bcoef, sol, f_rhs);
    };

    // ------------------------------------------------------------------
    // Ensure η* ghost cells are valid before computing f' and its Laplacian
    // ------------------------------------------------------------------
    energy_bc->FillBoundary(*eta_mf[lev], 0, 1, 0.0, 0);
    eta_mf[lev]->FillBoundary(geom.periodicity());

    // ------------------------------------------------------------------
    // Compute f'(η*) on valid cells, then fill ghost cells properly
    // (BUG FIX: previously used growntilebox reading stale eta ghost cells,
    //  and only called periodicity FillBoundary — physical boundary ghosts
    //  were uninitialized, corrupting ∇²f' near walls)
    // ------------------------------------------------------------------
    MultiFab f_prime_mf(eta_mf[lev]->boxArray(),
                        eta_mf[lev]->DistributionMap(), 1, nghost);
    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto eta = eta_mf[lev]->const_array(mfi);
        auto fp  = f_prime_mf.array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Real e = eta(i,j,k,0);
            fp(i,j,k,0) = ch_W_scale * 4.0 * e * (e - 0.5) * (e - 1.0);
        });
    }
    energy_bc->FillBoundary(f_prime_mf, 0, 1, 0.0, 0);   // physical BCs
    f_prime_mf.FillBoundary(geom.periodicity());            // periodic + neighbor exchange
    FillNaNGhostsZeroGrad(f_prime_mf);  // INT_DIR/NSCBC faces: physbc does nothing, fill by extrapolation

    // ------------------------------------------------------------------
    // rhs = η* + dt·M·∇²f'(η*)
    // ------------------------------------------------------------------
    MultiFab rhs_mf(eta_mf[lev]->boxArray(),
                    eta_mf[lev]->DistributionMap(), 1, 0);
    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto eta = eta_mf[lev]->const_array(mfi);
        auto fp  = f_prime_mf.const_array(mfi);
        auto rhs = rhs_mf.array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten   = Numeric::GetStencil(i, j, k, domain);
            Real lap_fp = Numeric::Laplacian(fp, i, j, k, 0, DX, sten);
            rhs(i,j,k,0) = eta(i,j,k,0) + dt * M_nom * lap_fp;
        });
    }

    // Solve 1: (I + γ(−∇²)) ψ = rhs
    MultiFab psi_mf(eta_mf[lev]->boxArray(),
                    eta_mf[lev]->DistributionMap(), 1, nghost);
    psi_mf.setVal(0.0);
    solveHelmholtz(psi_mf, rhs_mf);
    energy_bc->FillBoundary(psi_mf, 0, 1, 0.0, 0);

    // Solve 2: (I + γ(−∇²)) η^{n+1} = ψ
    MultiFab eta_new_mf(eta_mf[lev]->boxArray(),
                        eta_mf[lev]->DistributionMap(), 1, nghost);
    MultiFab::Copy(eta_new_mf, *eta_mf[lev], 0, 0, 1, nghost);
    solveHelmholtz(eta_new_mf, psi_mf);

    // Clamp and write back
    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto eta     = eta_mf[lev]->array(mfi);
        auto eta_new = eta_new_mf.const_array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            eta(i,j,k,0) = amrex::max(0.0, amrex::min(1.0, eta_new(i,j,k,0)));
        });
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// ApplyImplicitCH_Newton  (implicit_ch = 2)
//
// Newton-MLMG: solve the full nonlinear CH system
//   F(η) ≡ η − η* − dt·M·∇²μ(η) = 0,   μ(η) = f'(η) − ε²∇²η
//
// Newton iteration with simplified Jacobian (I + dt·M·ε²·∇⁴):
//   (I + dt·M·ε²·∇⁴) δη = −F(η^k) = η* − η^k + dt·M·∇²μ(η^k)
//   η^{k+1} = clamp(η^k + δη)
//
// Key advantage over impl_ch=1: μ is re-evaluated at η^k each iteration, so
// f'(η) is not frozen at η* (no Eyre splitting error). Converges quadratically
// in 3–5 iterations. The inner biharmonic solve still uses the double-Helmholtz
// factorization but only as a preconditioner — the Newton residual drives
// convergence to the correct nonlinear solution.
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::ApplyImplicitCH_Newton(int lev, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ApplyImplicitCH_Newton");

    const Geometry& geom   = this->geom[lev];
    const Real*     DX     = geom.CellSize();
    const Box&      domain = geom.Domain();
    const int nghost = eta_mf[lev]->nGrow();

    const Real kappa_CH = epsilon * epsilon;
    const Real M_nom    = (ch_mobility_nom > 0.0) ? ch_mobility_nom
                                                   : 0.2 * epsilon * (c_max + 1.0);
    const Real gamma_CH = std::sqrt(dt * M_nom * kappa_CH);

    // BC and coefficient setup (same as ApplyImplicitCH)
    Array<LinOpBCType, AMREX_SPACEDIM> lo_bc, hi_bc;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        lo_bc[d] = geom.isPeriodic(d) ? LinOpBCType::Periodic : LinOpBCType::Neumann;
        hi_bc[d] = geom.isPeriodic(d) ? LinOpBCType::Periodic : LinOpBCType::Neumann;
    }
    Array<MultiFab, AMREX_SPACEDIM> bcoef;
    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        BoxArray ba = eta_mf[lev]->boxArray(); ba.surroundingNodes(d);
        bcoef[d].define(ba, eta_mf[lev]->DistributionMap(), 1, 0);
        bcoef[d].setVal(1.0);
    }
    auto solveHelmholtz = [&](MultiFab& sol, const MultiFab& f_rhs) {
        CH_SetupAndSolveHelmholtz(lev, geom, eta_mf[lev]->boxArray(),
                                  eta_mf[lev]->DistributionMap(), gamma_CH,
                                  lo_bc, hi_bc, bcoef, sol, f_rhs);
    };

    // ------------------------------------------------------------------
    // FIX 1: Pre-clamp eta to [0,1] before saving eta_star.
    // The explicit Riemann/surface-tension advance (step 6) may push eta
    // outside [0,1].  If eta_star stores an exploded value the Newton rhs
    // (eta_star - eta^k + ...) is permanently huge regardless of convergence.
    // The global clamp in Advance step 9 hasn't run yet at this point.
    // ------------------------------------------------------------------
    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& vbx = mfi.validbox();
        auto eta = eta_mf[lev]->array(mfi);
        ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            eta(i,j,k,0) = amrex::max(0.0, amrex::min(1.0, eta(i,j,k,0)));
        });
    }

    // ------------------------------------------------------------------
    // Save η* — Newton iterates modify eta_mf[lev] in place
    // ------------------------------------------------------------------
    MultiFab eta_star(eta_mf[lev]->boxArray(),
                      eta_mf[lev]->DistributionMap(), 1, nghost);
    MultiFab::Copy(eta_star, *eta_mf[lev], 0, 0, 1, nghost);

    // Scratch MultiFabs allocated once outside the loop
    MultiFab mu_k    (eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, nghost);
    // FIX 2: Initialize mu_k to 0 so coarse-fine boundary ghost cells that
    // FillBoundary cannot fill (same-level only) contain 0 rather than
    // uninitialized garbage.  The Laplacian stencil in the residual reads
    // ghost cells; garbage there → ∇²μ ~ 1e+80 → persistent huge residual.
    mu_k.setVal(0.0);
    MultiFab rhs_mf  (eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, 0);
    MultiFab psi_mf  (eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, nghost);
    MultiFab delta_mf(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, nghost);

    // ------------------------------------------------------------------
    // Newton loop
    // ------------------------------------------------------------------
    for (int iter = 0; iter < ch_newton_iters; ++iter)
    {
        // Ensure η^k ghost cells are valid for Laplacian inside CH_ComputeMu
        energy_bc->FillBoundary(*eta_mf[lev], 0, 1, 0.0, 0);
        eta_mf[lev]->FillBoundary(geom.periodicity());
        FillNaNGhostsZeroGrad(*eta_mf[lev]);  // INT_DIR/NSCBC: extrapolate so ∇²η is finite

        // μ^k = f'(η^k) − ε²∇²η^k  (valid cells + filled ghosts)
        CH_ComputeMu(mu_k, *eta_mf[lev], energy_bc, geom, DX, domain, kappa_CH, ch_W_scale);
        FillNaNGhostsZeroGrad(mu_k);  // ensure Laplacian of μ^k reads finite ghost cells

        // Residual r = η^k − η* − dt·M·∇²μ^k
        // rhs_newton = −r = η* − η^k + dt·M·∇²μ^k
        // ||rhs_newton||_∞ → 0 at convergence
        Real res_norm = 0.0;
        for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();
            auto eta  = eta_mf[lev]->const_array(mfi);
            auto es   = eta_star.const_array(mfi);
            auto mu   = mu_k.const_array(mfi);
            auto rhs  = rhs_mf.array(mfi);
            ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto sten   = Numeric::GetStencil(i, j, k, domain);
                Real lap_mu = Numeric::Laplacian(mu, i, j, k, 0, DX, sten);
                rhs(i,j,k,0) = es(i,j,k,0) - eta(i,j,k,0) + dt * M_nom * lap_mu;
            });
        }
        res_norm = rhs_mf.norm0();
        // Util::ParallelMessage(INFO, "  CH Newton iter ", iter, " residual: ", res_norm);
        if (res_norm < ch_newton_tol) break;

        // Solve (I + dt·M·ε²·∇⁴) δη = rhs_newton via double-Helmholtz
        // Solve 1: (I + γ(−∇²)) ψ = rhs_newton
        psi_mf.setVal(0.0);
        solveHelmholtz(psi_mf, rhs_mf);
        energy_bc->FillBoundary(psi_mf, 0, 1, 0.0, 0);

        // Solve 2: (I + γ(−∇²)) δη = ψ
        delta_mf.setVal(0.0);
        solveHelmholtz(delta_mf, psi_mf);

        // Update: η^{k+1} = clamp(η^k + δη)
        for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& vbx = mfi.validbox();
            auto eta = eta_mf[lev]->array(mfi);
            auto de  = delta_mf.const_array(mfi);
            ParallelFor(vbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i,j,k,0) = amrex::max(0.0, amrex::min(1.0,
                                           eta(i,j,k,0) + de(i,j,k,0)));
            });
        }
    }
    // Ghost cells filled by step 8 FillBoundary in Advance
}

Set::Scalar InterpolateInterface(Set::Scalar eta, Set::Scalar val1, Set::Scalar val2) {
    // Ensure eta is clamped between 0 and 1
    eta = std::max(0.0, std::min(1.0, eta));
    // Perform linear interpolation
    return (1.0 - eta) * val1 + eta * val2;
}

// CURVATURE CALCULATION
void Hydro2::ComputeHeightFunction(int lev)
{
    BL_PROFILE("Hydro2::ComputeHeightFunction");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();
    const Box& domain = geom.Domain();

    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);

    // Ensure eta ghost cells are NaN-free before the column integration reads them.
    FillNaNGhostsZeroGrad(*eta_mf[lev]);

    // Column-integration height function (VOF analog for diffuse interfaces).
    //
    // Rather than searching for the η=0.5 iso-contour in each column, we sum η
    // over a local band of ±N_int cells in the dominant normal direction.  This
    // is the direct diffuse-interface analog of summing volume fractions in VOF:
    //
    //   h_int(i) = Σ_{k=-N_int}^{N_int}  η(i, j+k) · dy      [y-dominant]
    //
    // For a tanh profile centered at y₀, this sum ≈ y_hi_band − y₀ (a smooth,
    // always-defined quantity proportional to the interface position).
    //
    // Sign convention: to match the iso-contour sign expected by
    // ComputeHFcurvature_MUSCL (h varies like the interface coordinate),
    // we store  h = −sign(η_dominant) · h_int.  This gives:
    //   top surface (η_y > 0):  h ≈ y_interface(i) − const  → h_xx same sign as d²y/dx²
    //   bottom surface (η_y < 0): same property by symmetry
    // ComputeHFcurvature_MUSCL is therefore completely unchanged.
    //
    // N_int equals nGrow so the band uses the full allocated ghost region.
    // Increasing nghost to 4 gives a 9-cell band; blocking_factor must be >= nghost
    // so that FillBoundary can supply all ghost layers from neighbor valid cells.
    const int N_int = eta_mf[lev]->nGrow();

    h_eta_mf[lev]->setVal(0.0);

    for (MFIter mfi(*h_eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb  = mfi.tilebox();
        const Box& fab = mfi.fabbox();

        const int ilo_f = fab.smallEnd(0);
        const int ihi_f = fab.bigEnd(0);
        const int jlo_f = fab.smallEnd(1);
        const int jhi_f = fab.bigEnd(1);

        auto eta_arr = eta_mf[lev]->const_array(mfi);
        auto eta_x   = eta_x_mf[lev]->const_array(mfi);
        auto eta_y   = eta_y_mf[lev]->const_array(mfi);
        auto h_arr   = h_eta_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real ax = std::abs(eta_x(i,j,0));
            Real ay = std::abs(eta_y(i,j,0));

            if (ay >= ax)
            {
                // y-dominant: integrate η over column i in the j-direction.
                // Flip sign so h ≈ y_interface(i) − const (same convention as
                // iso-contour h), making ComputeHFcurvature_MUSCL unchanged.
                Real sn = (eta_y(i,j,0) >= 0.0) ? -1.0 : 1.0;

                int j0 = amrex::max(amrex::max(jlo, jlo_f), j - N_int);
                int j1 = amrex::min(amrex::min(jhi, jhi_f), j + N_int);

                Real h_int = 0.0;
                for (int jj = j0; jj <= j1; ++jj)
                    h_int += eta_arr(i, jj, 0) * dx[1];

                h_arr(i, j, 0) = sn * h_int;
            }
            else
            {
                // x-dominant: integrate η over row j in the i-direction.
                Real sn = (eta_x(i,j,0) >= 0.0) ? -1.0 : 1.0;

                int i0 = amrex::max(amrex::max(ilo, ilo_f), i - N_int);
                int i1 = amrex::min(amrex::min(ihi, ihi_f), i + N_int);

                Real h_int = 0.0;
                for (int ii = i0; ii <= i1; ++ii)
                    h_int += eta_arr(ii, j, 0) * dx[0];

                h_arr(i, j, 0) = sn * h_int;
            }
        });
    }

    h_eta_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeHFcurvature_MUSCL(int lev)
{
    BL_PROFILE("Hydro2::ComputeHFcurvature_MUSCL");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();

    for (MFIter mfi(*h_eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& fab = mfi.fabbox();
        const int ilo_f = fab.smallEnd(0);
        const int ihi_f = fab.bigEnd(0);
        const int jlo_f = fab.smallEnd(1);
        const int jhi_f = fab.bigEnd(1);

        auto h_arr   = h_eta_mf[lev]->const_array(mfi);
        auto eta_x   = eta_x_mf[lev]->const_array(mfi);
        auto eta_y   = eta_y_mf[lev]->const_array(mfi);
        auto kHF_arr = kappa_HF_mf[lev]->array(mfi);

        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            Real ax = std::abs(eta_x(i,j,0));
            Real ay = std::abs(eta_y(i,j,0));

            Real kHF;

            // Sentinel value written by ComputeHeightFunction for cells where
            // no eta=0.5 crossing was found within the N=2 search band.
            // If the center cell or either stencil neighbor has the sentinel,
            // the curvature stencil is invalid — zero kHF and let
            // ComputeHybridCurvature fall back to the smooth-normal estimate.
            const Real h_sentinel = 1.0e30;

            if (ay >= ax)
            {
                // y-dominant: h(i,j) = y-crossing, varies across x.
                int il = (i > ilo_f) ? i-1 : i;
                int ir = (i < ihi_f) ? i+1 : i;

                Real hC = h_arr(i,  j, 0);
                Real hL = h_arr(il, j, 0);
                Real hR = h_arr(ir, j, 0);

                // Require center and both neighbors to be (a) found (not sentinel)
                // and (b) y-dominant.  A neighbor that found an x-crossing has a
                // physically different h than a y-crossing; mixing them gives wrong hxx.
                bool neigh_ok =
                    (hC < h_sentinel*0.5) &&
                    (hL < h_sentinel*0.5) && (std::abs(eta_y(il,j,0)) >= std::abs(eta_x(il,j,0))) &&
                    (hR < h_sentinel*0.5) && (std::abs(eta_y(ir,j,0)) >= std::abs(eta_x(ir,j,0)));

                if (!neigh_ok)
                {
                    kHF = 0.0;
                }
                else
                {
                    Real dl = hC - hL;
                    Real dr = hR - hC;
                    Real slope = (dl*dr > 0.0) ? (2.0*dl*dr)/(dl+dr) : 0.0;

                    Real hx  = slope / dx[0];
                    Real hxx = (hR - 2.0*hC + hL) / (dx[0]*dx[0]);

                    // sign(eta_y) makes curvature consistent around the interface:
                    // without it, h_xx is negative at the top (concave-down) and
                    // positive at the bottom (concave-up) giving opposite signs for
                    // the same physical curvature.
                    Real sn = (eta_y(i,j,0) >= 0.0) ? 1.0 : -1.0;
                    kHF = sn * hxx / std::pow(1.0 + hx*hx, 1.5);
                }
            }
            else
            {
                // x-dominant: h(i,j) = x-crossing, varies across y.
                int jd = (j > jlo_f) ? j-1 : j;
                int ju = (j < jhi_f) ? j+1 : j;

                Real hC = h_arr(i, j,  0);
                Real hD = h_arr(i, jd, 0);
                Real hU = h_arr(i, ju, 0);

                bool neigh_ok =
                    (hC < h_sentinel*0.5) &&
                    (hD < h_sentinel*0.5) && (std::abs(eta_x(i,jd,0)) >  std::abs(eta_y(i,jd,0))) &&
                    (hU < h_sentinel*0.5) && (std::abs(eta_x(i,ju,0)) >  std::abs(eta_y(i,ju,0)));

                if (!neigh_ok)
                {
                    kHF = 0.0;
                }
                else
                {
                    Real dl = hC - hD;
                    Real dr = hU - hC;
                    Real slope = (dl*dr > 0.0) ? (2.0*dl*dr)/(dl+dr) : 0.0;

                    Real hy  = slope / dx[1];
                    Real hyy = (hU - 2.0*hC + hD) / (dx[1]*dx[1]);

                    Real sn = (eta_x(i,j,0) >= 0.0) ? 1.0 : -1.0;
                    kHF = sn * hyy / std::pow(1.0 + hy*hy, 1.5);
                }
            }

            kHF_arr(i,j,0) = kHF;
        });
    }

    kappa_HF_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeSmoothNormals(int lev)
{
    BL_PROFILE("Hydro2::ComputeSmoothNormals");

    const Geometry& geom = this->geom[lev];

    // Kernel radius: 1 → 3x3, 2 → 5x5.
    // The 5x5 kernel reads ghost cells at distance 2, which fits within nghost=2.
    const int krad = (smooth_kernel_size >= 5) ? 2 : 1;

    // Separable 1D Gaussian weights: g[d] = exp(-d^2/2), sigma=1 grid cell.
    //   krad=1: uses g[0]=1, g[1]=exp(-0.5)≈0.6065
    //   krad=2: additionally g[2]=exp(-2.0)≈0.1353
    // 2D weight at (di,dj) = g[|di|] * g[|dj|]; normalization via wsum per cell.
    amrex::GpuArray<Real, 3> gw;
    gw[0] = 1.0;
    gw[1] = std::exp(-0.5);
    gw[2] = std::exp(-2.0);  // only accessed when krad=2

    for (MFIter mfi(*nx_smoothed_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto eta_x   = eta_x_mf[lev]->const_array(mfi);
        auto eta_y   = eta_y_mf[lev]->const_array(mfi);
        auto gradmag = gradmag_mf[lev]->const_array(mfi);
        auto nx_s    = nx_smoothed_mf[lev]->array(mfi);
        auto ny_s    = ny_smoothed_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real accx = 0.0, accy = 0.0, wsum = 0.0;

            // Use the grown tilebox (valid cells + up to krad ghost layers) so
            // that MPI-neighbor ghost cells filled by FillBoundary are included.
            // Only skip cells that are outside the allocated ghost region.
            // Physical-boundary ghost cells beyond the domain are initialized to
            // zero by RegisterNewFab and contribute zero weight — correct behavior.
            const Box tb_g = amrex::grow(tb, krad);

            for (int di = -krad; di <= krad; ++di)
            for (int dj = -krad; dj <= krad; ++dj)
            {
                int ii = i + di, jj = j + dj;
                if (!tb_g.contains(IntVect(AMREX_D_DECL(ii, jj, 0)))) continue;

                Real w    = gw[std::abs(di)] * gw[std::abs(dj)];
                Real gmN  = gradmag(ii, jj, k) + 1e-14;
                accx += w * eta_x(ii, jj, k) / gmN;
                accy += w * eta_y(ii, jj, k) / gmN;
                wsum += w;
            }

            Real nx = accx / (wsum + 1e-14);
            Real ny = accy / (wsum + 1e-14);
            Real m  = std::sqrt(nx*nx + ny*ny) + 1e-14;
            nx_s(i, j, k) = nx / m;
            ny_s(i, j, k) = ny / m;
        });
    }

    nx_smoothed_mf[lev]->FillBoundary(geom.periodicity());
    ny_smoothed_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeHybridCurvature(int lev)
{
    BL_PROFILE("Hydro2::ComputeHybridCurvature");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();
    const Box& dom = geom.Domain();

    // Cg: gradient magnitude threshold to identify interface cells.
    // When epsilon >= dx (resolved), expected gm ~ 1/epsilon.
    // When epsilon < dx (under-resolved), effective gm ~ 1/dx.
    // Use the coarser scale so the threshold is never too tight.
    const Real dx_eff = std::max({dx[0], dx[1], epsilon});
    const Real Cg = 0.1 / dx_eff;
    const Real small = 1e-14;
    const int km = kappa_method;  // capture for GPU lambda

    for (MFIter mfi(*kappas_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto gradmag  = gradmag_mf[lev]->const_array(mfi);
        auto nx_s     = nx_smoothed_mf[lev]->const_array(mfi);
        auto ny_s     = ny_smoothed_mf[lev]->const_array(mfi);
        auto kHF_arr  = kappa_HF_mf[lev]->const_array(mfi);
        auto h_arr    = h_eta_mf[lev]->const_array(mfi);

        auto kappas      = kappas_mf[lev]->array(mfi);
        auto kappa_SF    = kappa_SF_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i,int j,int k)
        {
            Real gm = gradmag(i,j,k);

            if (gm < Cg)
            {
                kappas(i,j,k,0) = 0.0;
                kappas(i,j,k,1) = 0.0;
                kappas(i,j,k,2) = 0.0;
                kappa_SF(i,j,k) = 0.0;
                return;
            }

            //------------------------------------------------------
            // Sg
            //------------------------------------------------------
            Real Sg = gm / (gm + Cg);

            //------------------------------------------------------
            // Compute SN curvature: kSN = d(nx)/dx + d(ny)/dy
            //------------------------------------------------------
            // Clamp to the DOMAIN boundary, not the tilebox boundary.
            // Ghost cells of nx_s / ny_s / h_arr are valid (FillBoundary was
            // called) for MPI-neighbor cells, so we should read them.  Only
            // clamp when the cell is at the physical domain edge where no ghost
            // data exists beyond the boundary.
            int il = amrex::max(i-1, dom.smallEnd(0));
            int ir = amrex::min(i+1, dom.bigEnd(0));
            int jl = amrex::max(j-1, dom.smallEnd(1));
            int jr = amrex::min(j+1, dom.bigEnd(1));

            Real nx_x = 0.5 * (nx_s(ir,j,k) - nx_s(il,j,k)) / dx[0];
            Real ny_y = 0.5 * (ny_s(i,jr,k) - ny_s(i,jl,k)) / dx[1];

            Real kSN = nx_x + ny_y;

            //------------------------------------------------------
            // HF curvature at same (i,j,k)
            //------------------------------------------------------
            Real kHF = kHF_arr(i,j,k);

            //------------------------------------------------------
            // Sm monotonicity test: check neighbors in the direction
            // perpendicular to the dominant normal (i.e. along the
            // interface tangent) with the corresponding grid scale.
            //------------------------------------------------------
            const Real h_sentinel = 1.0e30;
            Real hC = h_arr(i,j,0);
            Real Sm = 1.0;

            // Any sentinel in center or tangential neighbors → kHF not valid here
            if (hC > h_sentinel*0.5) { Sm = 0.0; }
            else if (std::abs(ny_s(i,j,k)) >= std::abs(nx_s(i,j,k)))
            {
                // y-dominant: h is y-crossing varying in x
                if (i > tb.smallEnd(0))
                {
                    Real hN = h_arr(i-1,j,0);
                    if (hN > h_sentinel*0.5 || std::abs(hN - hC) > 3.0*dx[1]) Sm = 0.0;
                }
                if (i < tb.bigEnd(0))
                {
                    Real hN = h_arr(i+1,j,0);
                    if (hN > h_sentinel*0.5 || std::abs(hN - hC) > 3.0*dx[1]) Sm = 0.0;
                }
            }
            else
            {
                // x-dominant: h is x-crossing varying in y
                if (j > tb.smallEnd(1))
                {
                    Real hN = h_arr(i,j-1,0);
                    if (hN > h_sentinel*0.5 || std::abs(hN - hC) > 3.0*dx[0]) Sm = 0.0;
                }
                if (j < tb.bigEnd(1))
                {
                    Real hN = h_arr(i,j+1,0);
                    if (hN > h_sentinel*0.5 || std::abs(hN - hC) > 3.0*dx[0]) Sm = 0.0;
                }
            }

            //------------------------------------------------------
            // Ss smoothness: depends on |kHF - kSN_neg|
            //
            // Sign convention note:
            //   kHF = sign(eta_dominant) * h_xx/(1+hx^2)^1.5
            //       → negative for a convex droplet (eta=0 inside, eta=1 outside)
            //   kSN = div(∇η/|∇η|) = +1/R for the same convex droplet
            //
            // Both represent the same physical curvature but with opposite signs.
            // Flip kSN before blending so both conventions match: kSN_neg ≈ kHF.
            // This keeps Ss high (methods agree) and gives a consistent kH.
            //------------------------------------------------------
            Real kSN_neg = -kSN;

            Real Cs = amrex::max(std::abs(kHF), 1e-12);
            Real dk = std::abs(kHF - kSN_neg);

            Real Ss = std::exp(-dk / (Cs + small));

            //------------------------------------------------------
            // Final hybrid curvature weight
            //------------------------------------------------------
            Real w = Sg * Sm * Ss;
            w = amrex::max(0.0, amrex::min(1.0, w));

            Real kH = w * kHF + (1.0 - w) * kSN_neg;

            // Clamp to prevent runaway curvature from shock-compressed or
            // under-resolved interface regions driving enormous surface tension forces.
            const Real kappa_max = 2.0 / amrex::min(dx[0], dx[1]);
            kH = amrex::max(-kappa_max, amrex::min(kappa_max, kH));

            //------------------------------------------------------
            // Store results.
            // Component 0 = active curvature (selected by kappa_method):
            //   1 → kSN_neg  (smooth normals, negated to HF sign convention)
            //   2 → kHF      (height function)
            //   3 → kH       (hybrid blend)
            // Components 1,2 always store HF and raw SN for diagnostic output.
            //------------------------------------------------------
            Real kappa_active = (km == 1) ? kSN_neg
                              : (km == 2) ? kHF
                              :             kH;
            kappas(i,j,k,0) = kappa_active;
            kappas(i,j,k,1) = kHF;
            kappas(i,j,k,2) = kSN;
            kappa_SF(i,j,k) = kSN_neg;
        });
    }

    kappas_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeGradEta(int lev)
{
    BL_PROFILE("Hydro2::ComputeGradEta");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();
    const Box& domain = geom.Domain();

    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto eta_arr  = eta_mf[lev]->const_array(mfi);
        auto etax_arr = eta_x_mf[lev]->array(mfi);
        auto etay_arr = eta_y_mf[lev]->array(mfi);
        auto gmag_arr = gradmag_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            auto sten = Numeric::GetStencil(i,j,k,domain);

            // Correct overload: Gradient(Array4<const Real>, ...)
            Set::Vector G = Numeric::Gradient(eta_arr, i,j,k, 0, dx);

            Real gx = G(0);
            Real gy = G(1);

            etax_arr(i,j,k,0) = gx;
            etay_arr(i,j,k,0) = gy;
            gmag_arr(i,j,k,0) = std::sqrt(gx*gx + gy*gy);
        });
    }

    eta_x_mf[lev]->FillBoundary(geom.periodicity());
    eta_y_mf[lev]->FillBoundary(geom.periodicity());
    gradmag_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeKappas(int lev)
{
    BL_PROFILE("Hydro2::ComputeKappas");

    //
    // Full curvature pipeline — all three methods are always computed so that
    // all components of kappas_mf are available for diagnostic output:
    //   kappas[0] = active curvature (selected by kappa_method, used in force)
    //   kappas[1] = kHF  (height function)
    //   kappas[2] = kSN  (smooth normals, raw divergence convention)
    //
    // kappa_method:
    //   1 → kSN_neg = −kSN  (smooth normals, HF sign convention, default)
    //   2 → kHF             (height function)
    //   3 → kH              (weighted hybrid blend of HF and SN)
    //

    // 1. gradient of eta (uses your Numeric::Gradient)
    ComputeGradEta(lev);

    // 2. height function for HF method
    ComputeHeightFunction(lev);

    // 3. HF curvature with MUSCL slope limiter
    ComputeHFcurvature_MUSCL(lev);

    // 4. Gaussian smoothed normals + SN curvature
    ComputeSmoothNormals(lev);   // you named it this in your header

    // 5. Hybrid curvature (HF + SN with S_g, S_m, S_s)
    ComputeHybridCurvature(lev);

    // That’s it. Output curvature lives in kappas_mf:
    //    component 0 = hybrid curvature
    //    component 1 = HF curvature
    //    component 2 = SN curvature
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// GC-NSCBC /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Ghost Cell Navier-Stokes Characteristic Boundary Conditions
// Reference: Motheau, Almgren & Bell, J. Comput. Phys. 333 (2017) 1-24
//
// Algorithm overview (per face):
//   1. Compute one-sided 2nd-order FD of primitive variables at the boundary cell.
//   2. Compute LODI wave amplitudes L from the primitive gradients.
//   3. Replace the incoming acoustic L with the BC-specific value:
//      - OUTFLOW: relax incoming acoustic toward target pressure (Eq. 35).
//      - INFLOW:  relax all three incoming waves toward target state (Eqs. 42-44).
//   4. Reconstruct modified primitive gradients via dQ/dn = S * Lambda^{-1} * L.
//   5. Fill 1st and 2nd ghost cells from the gradient (Eqs. 27-28 or 31-32).
//   6. Convert primitive ghost values (rho, un, ut, p) to conserved (rho, M, E).
//
// Stiffened-gas adaptation: the characteristic speed is a = sqrt(gamma*(p+pi)/rho).
// The LODI relations retain the standard ideal-gas structure with (p+pi) in place of p
// for pressure-velocity coupling. The ghost cells store the PHYSICAL pressure p.
//
// Corner treatment (Motheau Sec. IV.B, Eqs. 51-67):
//   At corners where two NSCBC faces meet, a 2x2 coupled linear system is solved for
//   the two incoming acoustic wave amplitudes, then ghost cells are filled from both
//   normal-direction gradients simultaneously.
//
void Hydro2::ApplyNSCBC(
    int lev,
    amrex::MultiFab& eta_mf_in,
    amrex::MultiFab& rho_mf_in,
    amrex::MultiFab& M_mf_in,
    amrex::MultiFab& E_mf_in,
    amrex::Real /*time*/)
{
    BL_PROFILE("Integrator::Hydro2::ApplyNSCBC");

    using FT = Hydro2::NSCBCParams::FaceType;
    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();
    const Box& domain = geom.Domain();

    // Domain lengths for K relaxation coefficient. Auto-detect from geometry if 0.
    Real Lx_ = nscbc.Lx > 0.0 ? nscbc.Lx :
                (domain.bigEnd(0) - domain.smallEnd(0) + 1) * DX[0];
    Real Ly_ = nscbc.Ly > 0.0 ? nscbc.Ly :
                (domain.bigEnd(1) - domain.smallEnd(1) + 1) * DX[1];

    // Capture EOS and NSCBC scalars for the device kernels.
    Real g0_ = gamma0, g1_ = gamma1;
    Real pi0_ = pi_0, pi1_ = pi_1;
    Real pref_ = pref;
    Real cv0_ = cv0, cv1_ = cv1;

    Real nscbc_sigma_ = nscbc.sigma;
    Real nscbc_beta_  = nscbc.beta;

    // Face parameter arrays (captured by copy into lambdas)
    Real face_p_t [4], face_un_t[4], face_ut_t[4];
    Real face_T_t [4], face_eta_r[4];
    for (int f = 0; f < 4; ++f) {
        face_p_t [f] = nscbc.p_t[f];
        face_un_t[f] = nscbc.un_t[f];
        face_ut_t[f] = nscbc.ut_t[f];
        face_T_t [f] = nscbc.T_t[f];
        face_eta_r[f]= nscbc.eta_relax[f];
    }

    FT face_types[4];
    for (int f = 0; f < 4; ++f) face_types[f] = nscbc.face_type[f];

    // -------------------------------------------------------------------------
    // GPU-callable helper: compute primitives (rho, un, ut, p, a, gmix, pimix)
    // at a given cell from the conserved arrays.
    //
    // dir   = normal direction (0=x, 1=y)
    // For dir=0: un = vx, ut = vy
    // For dir=1: un = vy, ut = vx
    // -------------------------------------------------------------------------
    auto compute_prim = [=] AMREX_GPU_HOST_DEVICE (
        const Array4<const Real>& eta_a,
        const Array4<const Real>& rho_a,
        const Array4<const Real>& M_a,
        const Array4<const Real>& E_a,
        int ii, int jj, int kk, int dir_,
        Real& rho_out, Real& un_out, Real& ut_out,
        Real& p_out,   Real& a_out,
        Real& gm_out,  Real& pi_out) -> void
    {
        Real eta_ = eta_a(ii,jj,kk);
        Real rho_ = amrex::max(rho_a(ii,jj,kk), Real(1.0e-14));
        Real mx_  = M_a(ii,jj,kk,0);
        Real my_  = M_a(ii,jj,kk,1);
        Real E_   = E_a(ii,jj,kk);

        Real vx_ = mx_ / rho_;
        Real vy_ = my_ / rho_;
        // Guard against huge velocities from near-zero rho (rho was clamped to 1e-14).
        // Without this, vx_=mx_/1e-14 can be enormous, making T-terms Inf and
        // then dp_dn_new=Inf → ghost pressure=Inf → Riemann solver hits Inf-Inf=NaN.
        const Real vel_lim_ = Real(1.0e7);
        if (!std::isfinite(vx_) || std::abs(vx_) > vel_lim_) vx_ = 0.0;
        if (!std::isfinite(vy_) || std::abs(vy_) > vel_lim_) vy_ = 0.0;
        Real KE_ = 0.5 * rho_ * (vx_*vx_ + vy_*vy_);
        Real UE_ = E_ - KE_;
        if (!(UE_ > 0.0)) UE_ = 0.0;  // handles NaN as well as < 0

        // Stiffened-gas EOS mixing (same order as line 739 in RHS)
        Real gm_,pi_;
        Thermo_Interp::InterpolateGammaPi_Stiffened(
            eta_, g1_, g0_, pi1_, pi0_, gm_, pi_);

        Real p_ = (gm_ - 1.0)*UE_ - gm_*pi_ + pref_;
        if (!(p_ > 0.0)) p_ = 1.0e-6;  // handles NaN as well as < 0

        Real a_ = std::sqrt(gm_ * (p_ + pi_) / rho_);

        rho_out = rho_;
        un_out  = (dir_ == 0) ? vx_ : vy_;
        ut_out  = (dir_ == 0) ? vy_ : vx_;
        p_out   = p_;
        a_out   = a_;
        gm_out  = gm_;
        pi_out  = pi_;
    };

    // -------------------------------------------------------------------------
    // GPU-callable helper: write primitive ghost values back to conserved arrays.
    // Uses the eta already set in the ghost cell by the prior FillBoundary call.
    // -------------------------------------------------------------------------
    auto set_ghost_cons = [=] AMREX_GPU_HOST_DEVICE (
        const Array4<const Real>& eta_a,
        const Array4<      Real>& rho_a,
        const Array4<      Real>& M_a,
        const Array4<      Real>& E_a,
        const Array4<      Real>& gam_a_,
        const Array4<      Real>& pi_a_,
        const Array4<      Real>& p_a_,
        int ii, int jj, int kk, int dir_,
        Real rho_, Real un_, Real ut_, Real p_) -> void
    {
        Real eta_ = eta_a(ii,jj,kk);
        // Guard: if eta at the ghost cell is NaN or out of range (e.g. FillBoundary
        // extrapolated from a NaN boundary cell), clamp it to [0,1].  A NaN eta
        // flows through InterpolateGammaPi and makes gm_/pi_ NaN, which then makes
        // UE_ = (p + NaN - pref)/(NaN-1) = NaN even when p and rho are finite.
        if (!(eta_ >= 0.0 && eta_ <= 1.0)) eta_ = amrex::max(amrex::min(eta_, Real(1.0)), Real(0.0));
        if (amrex::isnan(eta_)) eta_ = Real(0.0);
        Real gm_, pi_;
        Thermo_Interp::InterpolateGammaPi_Stiffened(
            eta_, g1_, g0_, pi1_, pi0_, gm_, pi_);

        // Inverse EOS: UE from p
        Real UE_ = (p_ + gm_*pi_ - pref_) / (gm_ - 1.0);
        if (UE_ < 0.0) UE_ = 0.0;

        Real vx_ = (dir_ == 0) ? un_ : ut_;
        Real vy_ = (dir_ == 0) ? ut_ : un_;
        Real KE_ = 0.5 * rho_ * (vx_*vx_ + vy_*vy_);

        rho_a  (ii,jj,kk)   = rho_;
        M_a    (ii,jj,kk,0) = rho_ * vx_;
        M_a    (ii,jj,kk,1) = rho_ * vy_;
        E_a    (ii,jj,kk)   = UE_ + KE_;
        gam_a_ (ii,jj,kk)   = gm_;
        pi_a_  (ii,jj,kk)   = pi_;
        p_a_   (ii,jj,kk)   = p_;
    };

    // =========================================================================
    // Loop over tiles, process each active NSCBC face.
    //
    // Face layout:  0 = xlo (dir=0, side=lo)
    //               1 = xhi (dir=0, side=hi)
    //               2 = ylo (dir=1, side=lo)
    //               3 = yhi (dir=1, side=hi)
    // =========================================================================
    for (MFIter mfi(rho_mf_in, false); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();

        auto eta_a = eta_mf_in.array(mfi);
        auto rho_a = rho_mf_in.array(mfi);
        auto M_a   = M_mf_in  .array(mfi);
        auto E_a   = E_mf_in  .array(mfi);
        auto gam_a = gamma_mf   [lev]->array(mfi);
        auto pi_a  = pi_mf      [lev]->array(mfi);
        auto p_a   = pressure_mf[lev]->array(mfi);

        // Read-only versions for the compute_prim helper (ghost cells are readable)
        auto eta_r = eta_mf_in.const_array(mfi);
        auto rho_r = rho_mf_in.const_array(mfi);
        auto M_r   = M_mf_in  .const_array(mfi);
        auto E_r   = E_mf_in  .const_array(mfi);

        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
        {
            Real dhn  = DX[dir];           // cell size in normal direction
            Real dht  = DX[1 - dir];       // cell size in transverse direction
            Real Ln_  = (dir == 0) ? Lx_ : Ly_;  // domain length in normal dir

            for (int side = 0; side < 2; ++side)  // side=0 → lo, side=1 → hi
            {
                // face index: xlo=0, xhi=1, ylo=2, yhi=3
                int face_idx = dir * 2 + side;

                if (face_types[face_idx] == FT::NONE) continue;

                // Check if this tile touches the physical domain boundary on this face
                int ibdry;
                if (side == 0) {
                    if (vbx.smallEnd(dir) != domain.smallEnd(dir)) continue;
                    ibdry = domain.smallEnd(dir);
                } else {
                    if (vbx.bigEnd(dir) != domain.bigEnd(dir)) continue;
                    ibdry = domain.bigEnd(dir);
                }

                // Extract per-face BC parameters as scalars for GPU capture.
                // C-style arrays cannot be reliably captured by value in GPU lambdas.
                Real p_t_    = face_p_t [face_idx];
                Real un_t_   = face_un_t[face_idx];
                Real ut_t_   = face_ut_t[face_idx];
                Real T_t_fi  = face_T_t [face_idx]; // temperature target (scalar for GPU)
                Real eta_r_  = face_eta_r[face_idx];
                FT   ftype   = face_types[face_idx];
                int  side_  = side;  // 0=lo,1=hi (copy to avoid capture of loop var)
                int  dir_   = dir;

                // Build a 1-cell-thick box along the face for the ParallelFor
                Box face_box = vbx;
                face_box.setSmall(dir, ibdry);
                face_box.setBig  (dir, ibdry);
                int ng_total = rho_mf_in.nGrow(); // captured by value in lambda below

                ParallelFor(face_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    // Boundary cell index in normal direction
                    // (i or j depending on dir_ -- the lambda processes general (i,j,k))

                    // --------------------------------------------------------
                    // Step 1: compute primitives at boundary (b) and two interior
                    // neighbors (m1, m2) for one-sided normal FD.
                    //
                    // Neighbor offsets in normal direction:
                    //   hi side (side_=1): m1 = N-1, m2 = N-2  (interior)
                    //   lo side (side_=0): m1 = N+1, m2 = N+2  (interior)
                    // --------------------------------------------------------
                    int di = (dir_ == 0) ? 1 : 0;
                    int dj = (dir_ == 1) ? 1 : 0;

                    // Offsets for "interior" direction (away from boundary)
                    int sign_in = (side_ == 1) ? -1 : 1;  // -1 for hi, +1 for lo

                    int im1 = i + sign_in * di;
                    int jm1 = j + sign_in * dj;
                    int im2 = i + sign_in * 2 * di;
                    int jm2 = j + sign_in * 2 * dj;

                    Real rho_b, un_b, ut_b, p_b, a_b, gm_b, pi_b;
                    compute_prim(eta_r, rho_r, M_r, E_r, i, j, k, dir_,
                                 rho_b, un_b, ut_b, p_b, a_b, gm_b, pi_b);

                    // If the boundary cell or its interior neighbors have bad energy,
                    // the NSCBC formulas will produce junk ghost values that can cause
                    // the Riemann solver to blow up in adjacent interior cells.
                    // Skip NSCBC and leave ghost cells as-is from FillBoundary.
                    if (!std::isfinite(E_r(i,   j,   k))  ||
                        !std::isfinite(E_r(im1, jm1, k))  ||
                        !std::isfinite(M_r(i,   j,   k,0))||
                        !std::isfinite(M_r(i,   j,   k,1)))
                        return;

                    Real rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1;
                    compute_prim(eta_r, rho_r, M_r, E_r, im1, jm1, k, dir_,
                                 rho_m1, un_m1, ut_m1, p_m1, a_m1, gm_m1, pi_m1);

                    Real rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2;
                    compute_prim(eta_r, rho_r, M_r, E_r, im2, jm2, k, dir_,
                                 rho_m2, un_m2, ut_m2, p_m2, a_m2, gm_m2, pi_m2);

                    Real a2_b = a_b * a_b;
                    Real M_loc = std::abs(un_b) / (a_b + 1.0e-14);

                    // --------------------------------------------------------
                    // Step 2: one-sided 1st-order FD in normal direction.
                    //   hi side: df/dn = (f_b - f_m1) / dhn
                    //   lo side: df/dn = (f_m1 - f_b) / dhn  [= -(f_b-f_m1)/dhn]
                    //
                    // NOTE: we intentionally use 1st-order (not 2nd-order) here.
                    // The 2nd-order one-sided stencil (3f_b - 4f_{m1} + f_{m2})
                    // has a leading coefficient of 3, which amplifies boundary-
                    // localized perturbations by 3× before the L-wave computation.
                    // Combined with the acoustic impedance Z = ρa (large for
                    // stiffened-EOS fluids), this creates a ~1.5× growth factor
                    // per timestep, causing exponential blow-up along boundary
                    // cells in ~25 steps.  The 1st-order stencil (coefficient 1)
                    // eliminates this amplification while giving the same accuracy
                    // as 2nd-order for smooth acoustic waves passing the boundary.
                    // --------------------------------------------------------
                    Real fd_sign = (side_ == 1) ? 1.0 : -1.0;
                    Real drho_dn = fd_sign * (rho_b - rho_m1) / dhn;
                    Real dun_dn  = fd_sign * (un_b  - un_m1 ) / dhn;
                    Real dut_dn  = fd_sign * (ut_b  - ut_m1 ) / dhn;
                    Real dp_dn   = fd_sign * (p_b   - p_m1  ) / dhn;
                    // Here is the 2nd-order version we are NOT using:
                    // Hopefully we can go back to this once we have a more robust corner treatment that can handle the amplified perturbations.  See Motheau et al. JCP 2017 for their use of the 2nd-order one-sided stencil.
                    // Real drho_dn = fd_sign * (3.0*rho_b - 4.0*rho_m1 + rho_m2) / (2.0*dhn);
                    // Real dun_dn  = fd_sign * (3.0*un_b  - 4.0*un_m1  + un_m2 ) / (2.0*dhn);
                    // Real dut_dn  = fd_sign * (3.0*ut_b  - 4.0*ut_m1  + ut_m2 ) / (2.0*dhn);
                    // Real dp_dn   = fd_sign * (3.0*p_b   - 4.0*p_m1   + p_m2  ) / (2.0*dhn);


                    // --------------------------------------------------------
                    // Step 3: central 2nd-order FD in transverse direction.
                    // --------------------------------------------------------
                    int ti = (dir_ == 1) ? 1 : 0;
                    int tj = (dir_ == 0) ? 1 : 0;
                    int itp = amrex::min(i + ti, domain.bigEnd (0));
                    int itm = amrex::max(i - ti, domain.smallEnd(0));
                    int jtp = amrex::min(j + tj, domain.bigEnd (1));
                    int jtm = amrex::max(j - tj, domain.smallEnd(1));
                    // dt_denom = 2*dht for central diff, 1*dht for one-sided.
                    // At face edges a transverse neighbor is clamped to the boundary cell
                    // (e.g. j=0: jtm=j, so we only have a one-sided step of 1 cell).
                    // Use the actual span between the two transverse samples.
                    int nsteps_t = (itp - itm) + (jtp - jtm);
                    Real dt_denom = dht * amrex::max(nsteps_t, 1);

                    Real rho_tp, un_tp, ut_tp, p_tp, a_tp, gm_tp, pi_tp;
                    compute_prim(eta_r, rho_r, M_r, E_r, itp, jtp, k, dir_,
                                 rho_tp, un_tp, ut_tp, p_tp, a_tp, gm_tp, pi_tp);

                    Real rho_tm, un_tm, ut_tm, p_tm, a_tm, gm_tm, pi_tm;
                    compute_prim(eta_r, rho_r, M_r, E_r, itm, jtm, k, dir_,
                                 rho_tm, un_tm, ut_tm, p_tm, a_tm, gm_tm, pi_tm);

                    Real drho_dt = (rho_tp - rho_tm) / dt_denom;
                    Real dun_dt  = (un_tp  - un_tm ) / dt_denom;
                    Real dut_dt  = (ut_tp  - ut_tm ) / dt_denom;
                    Real dp_dt   = (p_tp   - p_tm  ) / dt_denom;

                    // --------------------------------------------------------
                    // Step 4: LODI wave amplitudes from interior values.
                    // L1 = (un-a)/(2a^2) * (dp/dn - rho*a*dun/dn)  [backward acoustic]
                    // L2 = un/a^2 * (a^2*drho/dn - dp/dn)           [entropy]
                    // L3 = un * dut/dn                               [shear]
                    // L4 = (un+a)/(2a^2) * (dp/dn + rho*a*dun/dn)  [forward acoustic]
                    // --------------------------------------------------------
                    Real L1 = (un_b - a_b) / (2.0*a2_b) * (dp_dn - rho_b*a_b*dun_dn);
                    Real L2 = (std::abs(un_b) > 1.0e-14) ?
                               un_b / a2_b * (a2_b*drho_dn - dp_dn) : 0.0;
                    Real L3 = (std::abs(un_b) > 1.0e-14) ?
                               un_b * dut_dn : 0.0;
                    Real L4 = (un_b + a_b) / (2.0*a2_b) * (dp_dn + rho_b*a_b*dun_dn);

                    // --------------------------------------------------------
                    // Step 5: transverse source terms (Motheau Eqs. 36-37,45-46).
                    // These are the "T" terms included in the incoming wave mod.
                    // T1 involves the backward acoustic direction (L1 wave).
                    // T4 involves the forward acoustic direction (L4 wave).
                    // T2 and T3 are for entropy and shear.
                    //
                    // Derived from: T = Lambda * S^{-1} * B_t * dQ/dt
                    // where B_t is the transverse Jacobian.
                    // --------------------------------------------------------
                    Real T1 = (un_b - a_b) / (2.0*a2_b) *
                              (-rho_b*a_b*ut_b*dun_dt + a2_b*rho_b*dut_dt + ut_b*dp_dt);
                    Real T4 = (un_b + a_b) / (2.0*a2_b) *
                              ( rho_b*a_b*ut_b*dun_dt + a2_b*rho_b*dut_dt + ut_b*dp_dt);
                    Real T2 = (std::abs(un_b) > 1.0e-14) ?
                               un_b * ut_b / a2_b * (a2_b*drho_dt - dp_dt) : 0.0;
                    Real T3 = (std::abs(un_b) > 1.0e-14) ?
                               un_b * (ut_b*dut_dt + (1.0/rho_b)*dp_dt) : 0.0;

                    // --------------------------------------------------------
                    // Step 6: apply BC-specific modification to incoming wave(s).
                    //
                    // Convention (subsonic, standard orientation):
                    //   hi side (xhi/yhi): L1 is the incoming acoustic (speed un-a < 0).
                    //   lo side (xlo/ylo): L4 is the incoming acoustic (speed un+a > 0).
                    //
                    // For OUTFLOW: damp the single incoming acoustic wave (Motheau Eq. 35).
                    //   K = sigma * a * (1 - M^2) / L_domain
                    //   L_in = K*(p - p_t) - (1-beta)*T_in
                    //
                    // For INFLOW: specify all three incoming waves (Motheau Eqs. 42-44).
                    //   Acoustic: L_in = eta_relax*(rho*a^2*(1-M^2)/L)*(un - un_t) - T_in
                    //   Entropy:  L2   = eta_relax*(rho*a*(gm-1)/L) *(T - T_t)  - T2
                    //   Shear:    L3   = eta_relax*(a/L)             *(ut - ut_t) - T3
                    //   (for lo side L4 is incoming; for hi side L1 is incoming)
                    // --------------------------------------------------------
                    // K has units s/m² to match the L-amplitude convention
                    // (L_k = λ_k/(2a²) * characteristic_gradient, so K = σ/(a*L) not σ*a/L)
                    Real K = nscbc_sigma_ * (1.0 - M_loc*M_loc) / (a_b * Ln_);

                    // Local temperature (needed for inflow entropy relaxation, Motheau Eq. 43)
                    Real eta_b  = eta_r(i,j,k);
                    Real cv_mix = eta_b * cv0_ + (1.0 - eta_b) * cv1_;
                    Real T_b = (p_b + pi_b) / (rho_b * cv_mix * (gm_b - 1.0) + 1.0e-14);

                    if (ftype == FT::OUTFLOW) {
                        // hi side: incoming = L1; lo side: incoming = L4
                        if (side_ == 1) {  // hi boundary (xhi or yhi)
                            L1 = K * (p_b - p_t_) - (1.0 - nscbc_beta_) * T1;
                        } else {           // lo boundary (xlo or ylo)
                            L4 = K * (p_b - p_t_) - (1.0 - nscbc_beta_) * T4;
                        }
                    }
                    else if (ftype == FT::INFLOW) {
                        Real er = eta_r_;
                        Real K_acoustic = er * rho_b * (1.0 - M_loc*M_loc) / Ln_;
                        Real K_entropy  = er * rho_b * a_b * (gm_b - 1.0) / Ln_;
                        Real K_shear    = er * a_b / Ln_;
                        if (side_ == 1) {  // hi boundary: L1 is incoming
                            L1 = -K_acoustic * (un_b - un_t_) - T1;
                            L2 = -K_entropy  * (T_b  - T_t_fi) - T2;
                            L3 = -K_shear    * (ut_b - ut_t_) - T3;
                        } else {           // lo boundary: L4 is incoming
                            L4 = K_acoustic * (un_b - un_t_) - T4;
                            L2 = K_entropy  * (T_b  - T_t_fi) - T2;
                            L3 = K_shear    * (ut_b - ut_t_) - T3;
                        }
                    }

                    // --------------------------------------------------------
                    // Step 7: reconstruct modified primitive gradients.
                    //   dQ/dn = S * Lambda^{-1} * L
                    //
                    //   xi1 = L1/(un-a),  xi2 = L2/un,  xi3 = L3/un,  xi4 = L4/(un+a)
                    //
                    //   drho/dn = xi1 + xi2 + xi4
                    //   dun/dn  = (-a/rho)*xi1 + (a/rho)*xi4
                    //   dut/dn  = xi3
                    //   dp/dn   = a^2*(xi1 + xi4)
                    // --------------------------------------------------------
                    Real xi1 = L1 / (un_b - a_b);   // un_b - a_b != 0 for subsonic
                    Real xi2 = (std::abs(un_b) > 1.0e-14) ? L2 / un_b : 0.0;
                    Real xi3 = (std::abs(un_b) > 1.0e-14) ? L3 / un_b : 0.0;
                    Real xi4 = L4 / (un_b + a_b);   // un_b + a_b != 0

                    Real drho_dn_new = xi1 + xi2 + xi4;
                    Real dun_dn_new  = (-a_b/rho_b)*xi1 + (a_b/rho_b)*xi4;
                    Real dut_dn_new  = xi3;
                    Real dp_dn_new   = a2_b * (xi1 + xi4);
                    // Guard: if any upstream value was NaN or Inf (e.g. from huge
                    // velocities at near-zero-rho cells), zero the gradient so the
                    // ghost cell gets a flat extrapolation rather than Inf pressure.
                    if (!std::isfinite(drho_dn_new)) drho_dn_new = 0.0;
                    if (!std::isfinite(dun_dn_new))  dun_dn_new  = 0.0;
                    if (!std::isfinite(dut_dn_new))  dut_dn_new  = 0.0;
                    if (!std::isfinite(dp_dn_new))   dp_dn_new   = 0.0;

                    // --------------------------------------------------------
                    // Step 8: fill 1st and 2nd ghost cells from gradient.
                    //
                    // hi side (right/top): ghost at N+1 and N+2
                    //   Q_{N+1} = Q_{N-1} + 2*dhn * dQ/dn  (Eq. 31)
                    //   Q_{N+2} = -2*Q_{N-1} - 3*Q_N + 6*Q_{N+1} - 6*dhn*dQ/dn (Eq. 32)
                    //
                    // lo side (left/bottom): ghost at -1 and -2
                    //   Q_{-1}  = Q_{+1}  - 2*dhn * dQ/dn  (Eq. 27)
                    //   Q_{-2}  = -2*Q_{+1} - 3*Q_0 + 6*Q_{-1} + 6*dhn*dQ/dn (Eq. 28)
                    // --------------------------------------------------------
                    // Ghost cell positions (in the normal direction)
                    int ig1_i = i + (-sign_in) * di;   // 1st ghost cell
                    int jg1_j = j + (-sign_in) * dj;
                    int ig2_i = i + (-sign_in) * 2*di; // 2nd ghost cell
                    int jg2_j = j + (-sign_in) * 2*dj;

                    // 1st ghost (Eqs. 27/31): use Q_{m1} (1 cell interior) as reference
                    Real sign_gh = (side_ == 1) ? 1.0 : -1.0; // +1 for hi, -1 for lo
                    Real rho_g1 = rho_m1 + sign_gh * 2.0*dhn * drho_dn_new;
                    Real un_g1  = un_m1  + sign_gh * 2.0*dhn * dun_dn_new;
                    Real ut_g1  = ut_m1  + sign_gh * 2.0*dhn * dut_dn_new;
                    Real p_g1   = p_m1   + sign_gh * 2.0*dhn * dp_dn_new;

                    // Fallback: if the NSCBC formula produces a ghost-cell pressure
                    // that is far below the boundary-cell pressure (e.g. a gas-phase
                    // cell at the boundary with a steep interface gradient), the formula
                    // has broken down.  Use zero-gradient (copy boundary cell state) so
                    // the Riemann solver sees a smooth state rather than a vacuum.
                    if (!std::isfinite(p_g1) || p_g1 < Real(1.0e-3) * p_b) {
                        set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                       ig1_i, jg1_j, k, dir_, rho_b, un_b, ut_b, p_b);
                        set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                       ig2_i, jg2_j, k, dir_, rho_b, un_b, ut_b, p_b);
                        // Fill g3..nGrow by constant extrapolation (copy boundary state)
                        for (int ig = 3; ig <= ng_total; ++ig) {
                            set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                           i + (-sign_in)*ig*di, j + (-sign_in)*ig*dj,
                                           k, dir_, rho_b, un_b, ut_b, p_b);
                        }
                        return;
                    }

                    // Use !(x>0) pattern which catches NaN; amrex::max(NaN,x) is not
                    // guaranteed to return x on all GPU backends.
                    if (!(rho_g1 > 0.0))         rho_g1 = rho_b;
                    if (!std::isfinite(un_g1))   un_g1 = 0.0;
                    if (!std::isfinite(ut_g1))   ut_g1 = 0.0;

                    // 2nd ghost: simple linear extrapolation g2 = 2*g1 - b.
                    // Algebraically equivalent to Lodato Eq. 32 for a linear profile,
                    // but with error amplification of 2 instead of 6, preventing the
                    // Eq. 32 coefficient-6 instability from collapsing the CFL timestep.
                    Real rho_g2 = 2.0*rho_g1 - rho_b;
                    Real un_g2  = 2.0*un_g1  - un_b;
                    Real ut_g2  = 2.0*ut_g1  - ut_b;
                    Real p_g2   = 2.0*p_g1   - p_b;
                    if (!(rho_g2 > 0.0))         rho_g2 = rho_b;
                    if (!std::isfinite(p_g2) || !(p_g2 > 0.0)) p_g2 = p_b;
                    if (!std::isfinite(un_g2))   un_g2 = 0.0;
                    if (!std::isfinite(ut_g2))   ut_g2 = 0.0;

                    // Write ghost cells to conserved arrays and EOS fields
                    set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                   ig1_i, jg1_j, k, dir_, rho_g1, un_g1, ut_g1, p_g1);
                    set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                   ig2_i, jg2_j, k, dir_, rho_g2, un_g2, ut_g2, p_g2);
                    // Fill g3..nGrow by constant extrapolation (copy g2 state)
                    for (int ig = 3; ig <= ng_total; ++ig) {
                        set_ghost_cons(eta_r, rho_a, M_a, E_a, gam_a, pi_a, p_a,
                                       i + (-sign_in)*ig*di, j + (-sign_in)*ig*dj,
                                       k, dir_, rho_g2, un_g2, ut_g2, p_g2);
                    }
                });
            }  // side
        }  // dir

        // =====================================================================
        // Corner treatment (Motheau Sec. IV.B, Eqs. 51-67).
        //
        // For cells that lie at the intersection of two NSCBC faces, we solve a
        // 2x2 coupled system for the two incoming acoustic wave amplitudes and
        // fill the corner ghost cells from both normal-direction gradients.
        //
        // Corner ghost cell positions (for top-right corner, xhi+yhi):
        //   Physical corner: (N, M)  -- already a valid cell
        //   Ghost row:  (N+1,M), (N+2,M)  -- filled by xhi kernel above
        //   Ghost col:  (N,M+1), (N,M+2)  -- filled by yhi kernel above
        //   Ghost corner: (N+1,M+1), (N+2,M+1), (N+1,M+2), (N+2,M+2)  -- need corner fill
        // =====================================================================
        for (int cx = 0; cx < 2; ++cx)  // cx=0 → xlo, cx=1 → xhi
        for (int cy = 0; cy < 2; ++cy)  // cy=0 → ylo, cy=1 → yhi
        {
            int fx = cx;           // face index for x: 0=xlo, 1=xhi
            int fy = 2 + cy;       // face index for y: 2=ylo, 3=yhi

            if (face_types[fx] == FT::NONE) continue;
            if (face_types[fy] == FT::NONE) continue;

            // Check if this tile touches both corner-contributing boundaries
            int ibdry_x = (cx == 0) ? domain.smallEnd(0) : domain.bigEnd(0);
            int ibdry_y = (cy == 0) ? domain.smallEnd(1) : domain.bigEnd(1);

            bool touch_x = (cx == 0) ? (vbx.smallEnd(0) == domain.smallEnd(0))
                                      : (vbx.bigEnd  (0) == domain.bigEnd  (0));
            bool touch_y = (cy == 0) ? (vbx.smallEnd(1) == domain.smallEnd(1))
                                      : (vbx.bigEnd  (1) == domain.bigEnd  (1));

            if (!touch_x || !touch_y) continue;

            // fill all ghost cells in each direction
            int ng = rho_mf_in.nGrow();

            // Build a box over the corner ghost cells.
            // sign_x: +1 for xhi ghosts, -1 for xlo ghosts
            int sx = (cx == 1) ?  1 : -1;
            int sy = (cy == 1) ?  1 : -1;

            Box corner_box;
            corner_box.setSmall(0, (cx == 1) ? ibdry_x + 1 : ibdry_x - ng);
            corner_box.setBig  (0, (cx == 1) ? ibdry_x + ng: ibdry_x - 1);
            corner_box.setSmall(1, (cy == 1) ? ibdry_y + 1 : ibdry_y - ng);
            corner_box.setBig  (1, (cy == 1) ? ibdry_y + ng: ibdry_y - 1);
#if AMREX_SPACEDIM == 3
            corner_box.setSmall(2, domain.smallEnd(2));
            corner_box.setBig  (2, domain.bigEnd  (2));
#endif

            // Capture per-face scalars for the corner kernel
            Real px_t_  = face_p_t[fx];
            Real py_t_  = face_p_t[fy];
            FT   ftx    = face_types[fx];
            FT   fty    = face_types[fy];
            Real etar_x = face_eta_r[fx];
            Real etar_y = face_eta_r[fy];
            Real un_t_x = face_un_t[fx];
            Real un_t_y = face_un_t[fy];

            ParallelFor(corner_box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Physical corner cell (at the actual domain boundary intersection)
                int ic = ibdry_x, jc = ibdry_y;

                // --- Primitives at corner boundary cell ---
                Real rho_c, un_c, ut_c, p_c, a_c, gm_c, pi_c;
                compute_prim(eta_r, rho_r, M_r, E_r, ic, jc, k, 0,
                             rho_c, un_c, ut_c, p_c, a_c, gm_c, pi_c);
                // For y-direction, "un" = vy, "ut" = vx
                Real vx_c = un_c, vy_c = ut_c;  // from dir=0 call: un=vx, ut=vy
                Real a2_c = a_c * a_c;
                Real Mx_c = std::abs(vx_c) / (a_c + 1.0e-14);
                Real My_c = std::abs(vy_c) / (a_c + 1.0e-14);

                // --- x-direction: one-sided FD at corner (ic, jc) ---
                // Interior neighbors in x
                int im1 = ic - sx, im2 = ic - 2*sx;
                Real rho_xm1, un_xm1, ut_xm1, p_xm1, a_xm1, gm_xm1, pi_xm1;
                compute_prim(eta_r, rho_r, M_r, E_r, im1, jc, k, 0,
                             rho_xm1, un_xm1, ut_xm1, p_xm1, a_xm1, gm_xm1, pi_xm1);
                Real rho_xm2, un_xm2, ut_xm2, p_xm2, a_xm2, gm_xm2, pi_xm2;
                compute_prim(eta_r, rho_r, M_r, E_r, im2, jc, k, 0,
                             rho_xm2, un_xm2, ut_xm2, p_xm2, a_xm2, gm_xm2, pi_xm2);

                Real fdx = (Real)sx;  // positive for hi (backward diff), negative for lo (forward diff)
                // 1st-order one-sided FD (see face loop comment for stability rationale)
                Real drho_dx_c = fdx * (rho_c - rho_xm1) / DX[0];
                Real dvx_dx_c  = fdx * (vx_c  - un_xm1 ) / DX[0];
                Real dvy_dx_c  = fdx * (vy_c  - ut_xm1 ) / DX[0];
                Real dp_dx_c   = fdx * (p_c   - p_xm1  ) / DX[0];
                // Here is the 2nd-order version we are NOT using:
                // Hopefully we can go back to this once we have a more robust corner treatment that can handle the amplified perturbations.  See Motheau et al. JCP 2017 for their use of the 2nd-order one-sided stencil.
                // Real drho_dx_c = fdx * (3.0*rho_c - 4.0*rho_xm1 + rho_xm2) / (2.0*DX[0]);
                // Real dvx_dx_c  = fdx * (3.0*vx_c  - 4.0*un_xm1  + un_xm2 ) / (2.0*DX[0]);
                // Real dvy_dx_c  = fdx * (3.0*vy_c  - 4.0*ut_xm1  + ut_xm2 ) / (2.0*DX[0]);
                // Real dp_dx_c   = fdx * (3.0*p_c   - 4.0*p_xm1   + p_xm2  ) / (2.0*DX[0]);


                // --- y-direction: one-sided FD at corner (ic, jc) ---
                int jm1 = jc - sy, jm2 = jc - 2*sy;
                Real rho_ym1, vn_ym1, vt_ym1, p_ym1, a_ym1, gm_ym1, pi_ym1;
                compute_prim(eta_r, rho_r, M_r, E_r, ic, jm1, k, 1,
                             rho_ym1, vn_ym1, vt_ym1, p_ym1, a_ym1, gm_ym1, pi_ym1);
                Real rho_ym2, vn_ym2, vt_ym2, p_ym2, a_ym2, gm_ym2, pi_ym2;
                compute_prim(eta_r, rho_r, M_r, E_r, ic, jm2, k, 1,
                             rho_ym2, vn_ym2, vt_ym2, p_ym2, a_ym2, gm_ym2, pi_ym2);
                // For y-direction (dir=1): un=vy, ut=vx in compute_prim output
                // 1st-order one-sided FD (see face loop comment for stability rationale)
                Real drho_dy_c = (Real)sy * (rho_c - rho_ym1) / DX[1];
                Real dvy_dy_c  = (Real)sy * (vy_c  - vn_ym1 ) / DX[1];
                Real dvx_dy_c  = (Real)sy * (vx_c  - vt_ym1 ) / DX[1];
                Real dp_dy_c   = (Real)sy * (p_c   - p_ym1  ) / DX[1];
                // Here is the 2nd-order version we are NOT using:
                // Hopefully we can go back to this once we have a more robust corner treatment that can
                // Real drho_dy_c = (Real)sy * (3.0*rho_c - 4.0*rho_ym1 + rho_ym2) / (2.0*DX[1]);
                // Real dvy_dy_c  = (Real)sy * (3.0*vy_c  - 4.0*vn_ym1  + vn_ym2 ) / (2.0*DX[1]);
                // Real dvx_dy_c  = (Real)sy * (3.0*vx_c  - 4.0*vt_ym1  + vt_ym2 ) / (2.0*DX[1]);
                // Real dp_dy_c   = (Real)sy * (3.0*p_c   - 4.0*p_ym1   + p_ym2  ) / (2.0*DX[1]);


                // --- LODI wave amplitudes from x-direction at corner ---
                // L1_x (incoming at hi, outgoing at lo), L4_x (incoming at lo, outgoing at hi)
                Real L1_x = (vx_c - a_c) / (2.0*a2_c) * (dp_dx_c - rho_c*a_c*dvx_dx_c);
                Real L2_x = (std::abs(vx_c) > 1.0e-14) ?
                             vx_c / a2_c * (a2_c*drho_dx_c - dp_dx_c) : 0.0;
                Real L3_x = (std::abs(vx_c) > 1.0e-14) ?
                             vx_c * dvy_dx_c : 0.0;
                Real L4_x = (vx_c + a_c) / (2.0*a2_c) * (dp_dx_c + rho_c*a_c*dvx_dx_c);

                // --- LODI wave amplitudes from y-direction at corner ---
                Real L1_y = (vy_c - a_c) / (2.0*a2_c) * (dp_dy_c - rho_c*a_c*dvy_dy_c);
                Real L2_y = (std::abs(vy_c) > 1.0e-14) ?
                             vy_c / a2_c * (a2_c*drho_dy_c - dp_dy_c) : 0.0;
                Real L3_y = (std::abs(vy_c) > 1.0e-14) ?
                             vy_c * dvx_dy_c : 0.0;
                Real L4_y = (vy_c + a_c) / (2.0*a2_c) * (dp_dy_c + rho_c*a_c*dvy_dy_c);

                // --- Identify incoming acoustic waves for this corner ---
                // hi-x: L1_x incoming; lo-x: L4_x incoming
                // hi-y: L1_y incoming; lo-y: L4_y incoming
                Real& Lx_in = (cx == 1) ? L1_x : L4_x;
                Real& Ly_in = (cy == 1) ? L1_y : L4_y;

                Real Kx = nscbc_sigma_ * (1.0 - Mx_c*Mx_c) / (a_c * Lx_);
                Real Ky = nscbc_sigma_ * (1.0 - My_c*My_c) / (a_c * Ly_);
                Real ab = (1.0 - nscbc_beta_) / 2.0;

                // --- Lodato (2008) Eq. 42: T̃ transverse corrections for corner ---
                // T_5: pressure-dilatation transverse term for each face direction.
                //   x-face (normal=x, transverse=y): T5_x = ρa² ∂vy/∂y
                //   y-face (normal=y, transverse=x): T5_y = ρa² ∂vx/∂x
                Real T5_x = a2_c * dvy_dy_c;
                Real T5_y = a2_c * dvx_dx_c;

                // Outgoing acoustic waves at the corner (complement of the incoming references)
                Real Lx_out = (cx == 1) ? L4_x : L1_x;
                Real Ly_out = (cy == 1) ? L4_y : L1_y;

                // Entropy T-term for x-LODI at corner (normal=x, transverse=y derivative)
                Real T2_xface = (std::abs(vx_c) > 1.0e-14) ?
                    vx_c * vy_c / a2_c * (a2_c*drho_dy_c - dp_dy_c) : 0.0;
                // Shear T-term for y-LODI at corner (normal=y, transverse=x derivative)
                Real T3_yface = (std::abs(vy_c) > 1.0e-14) ?
                    vy_c * (vx_c * dvx_dx_c + (1.0/rho_c)*dp_dx_c) : 0.0;

                // ζ: sign of the outgoing acoustic eigenvalue (+1 at hi, -1 at lo)
                Real zeta_x = (cx == 1) ? 1.0 : -1.0;
                Real zeta_y = (cy == 1) ? 1.0 : -1.0;

                // T̃ correction terms (Lodato 2008 Eqs. 42–43):
                //   T̃_x = T5_x − Ly_out/2 − ζ_x·ρ·a·(L2_y − T2_xface)
                //   T̃_y = T5_y − Lx_out/2 − ζ_y·ρ·a·(L3_x − T3_yface)
                Real Ttilde_x = T5_x - 0.5*Ly_out
                              - zeta_x * rho_c*a_c * (L2_y - T2_xface);
                Real Ttilde_y = T5_y - 0.5*Lx_out
                              - zeta_y * rho_c*a_c * (L3_x - T3_yface);

                if (ftx == FT::OUTFLOW && fty == FT::OUTFLOW)
                {
                    // Lodato Eq. 42 / Motheau Eq. 56: coupled outflow/outflow system
                    // with transverse correction terms added to RHS:
                    //   Lx_in + ab * Ly_in = Kx*(p-pt) + (1-β)*T̃_x
                    //   ab * Lx_in + Ly_in = Ky*(p-pt) + (1-β)*T̃_y
                    Real bx = Kx * (p_c - px_t_) + (1.0 - nscbc_beta_) * Ttilde_x;
                    Real by = Ky * (p_c - py_t_) + (1.0 - nscbc_beta_) * Ttilde_y;
                    Real det_inv = 1.0 / (1.0 - ab*ab + 1.0e-20);
                    Lx_in = det_inv * (bx - ab*by);
                    Ly_in = det_inv * (by - ab*bx);
                }
                else if (ftx == FT::OUTFLOW && fty == FT::INFLOW)
                {
                    // Motheau Eq. 58: compatibility Ly_in = 0; Lodato adds T5 to Lx_in
                    Ly_in = 0.0;
                    Lx_in = Kx * (p_c - px_t_) + (1.0 - nscbc_beta_) * T5_x;
                }
                else if (ftx == FT::INFLOW && fty == FT::OUTFLOW)
                {
                    Lx_in = 0.0;
                    Ly_in = Ky * (p_c - py_t_) + (1.0 - nscbc_beta_) * T5_y;
                }
                else if (ftx == FT::INFLOW && fty == FT::INFLOW)
                {
                    Real Kax = etar_x * rho_c * (1.0 - Mx_c*Mx_c) / Lx_;
                    Real Kay = etar_y * rho_c * (1.0 - My_c*My_c) / Ly_;
                    // x incoming: relax vx toward un_t_x; y incoming: relax vy toward un_t_y
                    Lx_in = (cx == 1) ? (-Kax * (vx_c - un_t_x))
                                      : ( Kax * (vx_c - un_t_x));
                    Ly_in = (cy == 1) ? (-Kay * (vy_c - un_t_y))
                                      : ( Kay * (vy_c - un_t_y));
                }

                // --- Reconstruct normal-direction gradients ---
                // X-direction
                Real xi1_x = L1_x / (vx_c - a_c);
                Real xi2_x = (std::abs(vx_c) > 1.0e-14) ? L2_x / vx_c : 0.0;
                Real xi3_x = (std::abs(vx_c) > 1.0e-14) ? L3_x / vx_c : 0.0;
                Real xi4_x = L4_x / (vx_c + a_c);

                Real drho_dx_new = xi1_x + xi2_x + xi4_x;
                Real dvx_dx_new  = (-a_c/rho_c)*xi1_x + (a_c/rho_c)*xi4_x;
                Real dvy_dx_new  = xi3_x;
                Real dp_dx_new   = a2_c * (xi1_x + xi4_x);
                if (!std::isfinite(drho_dx_new)) drho_dx_new = 0.0;
                if (!std::isfinite(dvx_dx_new))  dvx_dx_new  = 0.0;
                if (!std::isfinite(dvy_dx_new))  dvy_dx_new  = 0.0;
                if (!std::isfinite(dp_dx_new))   dp_dx_new   = 0.0;

                // Y-direction
                Real xi1_y = L1_y / (vy_c - a_c);
                Real xi2_y = (std::abs(vy_c) > 1.0e-14) ? L2_y / vy_c : 0.0;
                Real xi3_y = (std::abs(vy_c) > 1.0e-14) ? L3_y / vy_c : 0.0;
                Real xi4_y = L4_y / (vy_c + a_c);

                Real drho_dy_new = xi1_y + xi2_y + xi4_y;
                Real dvy_dy_new  = (-a_c/rho_c)*xi1_y + (a_c/rho_c)*xi4_y;
                Real dvx_dy_new  = xi3_y;
                Real dp_dy_new   = a2_c * (xi1_y + xi4_y);
                if (amrex::isnan(drho_dy_new)) drho_dy_new = 0.0;
                if (amrex::isnan(dvy_dy_new))  dvy_dy_new  = 0.0;
                if (amrex::isnan(dvx_dy_new))  dvx_dy_new  = 0.0;
                if (amrex::isnan(dp_dy_new))   dp_dy_new   = 0.0;

                // --- Fill this ghost corner cell from both direction gradients ---
                // Linear extrapolation from the physical corner boundary cell (ic, jc)
                // using the modified gradients from both normal directions.
                // Ghost offset from corner:
                //   ox = ±1 or ±2 (x-normal direction)
                //   oy = ±1 or ±2 (y-normal direction)
                int ox = i - ic;
                int oy = j - jc;

                Real rho_g = rho_c + (Real)ox*DX[0]*drho_dx_new
                                   + (Real)oy*DX[1]*drho_dy_new;
                Real vx_g  = vx_c  + (Real)ox*DX[0]*dvx_dx_new
                                   + (Real)oy*DX[1]*dvx_dy_new;
                Real vy_g  = vy_c  + (Real)ox*DX[0]*dvy_dx_new
                                   + (Real)oy*DX[1]*dvy_dy_new;
                Real p_g   = p_c   + (Real)ox*DX[0]*dp_dx_new
                                   + (Real)oy*DX[1]*dp_dy_new;

                if (!(rho_g > 0.0))          rho_g = rho_c;
                if (!std::isfinite(p_g) || !(p_g > 0.0)) p_g = p_c;
                if (!std::isfinite(vx_g))    vx_g  = 0.0;
                if (!std::isfinite(vy_g))    vy_g  = 0.0;

                // Store as conserved (use dir=0 since we already have vx,vy separately)
                Real eta_g = eta_r(i, j, k);
                if (amrex::isnan(eta_g) || !(eta_g >= 0.0 && eta_g <= 1.0))
                    eta_g = amrex::max(amrex::min(eta_g, Real(1.0)), Real(0.0));
                if (amrex::isnan(eta_g)) eta_g = 0.0;
                Real gm_g, pi_g;
                Thermo_Interp::InterpolateGammaPi_Stiffened(
                    eta_g, g1_, g0_, pi1_, pi0_, gm_g, pi_g);
                Real UE_g = (p_g + gm_g*pi_g - pref_) / (gm_g - 1.0);
                if (!(UE_g > 0.0)) UE_g = 0.0;
                Real KE_g = 0.5 * rho_g * (vx_g*vx_g + vy_g*vy_g);

                rho_a(i,j,k)   = rho_g;
                M_a  (i,j,k,0) = rho_g * vx_g;
                M_a  (i,j,k,1) = rho_g * vy_g;
                E_a  (i,j,k)   = UE_g + KE_g;
                gam_a(i,j,k)   = gm_g;
                pi_a (i,j,k)   = pi_g;
                p_a  (i,j,k)   = p_g;
            });
        }  // corner (cx, cy)

    }  // MFIter

    // =========================================================================
    // DIAGNOSTIC: scan ghost cells for NaN and print (i,j,k) location.
    // Remove once the ghost-cell coverage bug is identified.
    // =========================================================================
    amrex::Gpu::synchronize();
    for (amrex::MFIter mfi(rho_mf_in, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box vbx  = mfi.validbox();
        const amrex::Box gbx  = amrex::grow(vbx, rho_mf_in.nGrow());
        auto const& rho_scan  = rho_mf_in.array(mfi);
        auto const& E_scan    = E_mf_in.array(mfi);
        auto const& eta_scan = eta_mf_in.array(mfi);
        amrex::LoopOnCpu(gbx, [&](int i, int j, int k)
        {
            bool in_valid = vbx.contains(amrex::IntVect(AMREX_D_DECL(i,j,k)));
            if (amrex::isnan(rho_scan(i,j,k)))
                amrex::Print() << "[NSCBC-NaN] rho NaN at ("
                    << i << "," << j << "," << k << ")"
                    << (in_valid ? " [interior]" : " [ghost]") << "\n";
            if (amrex::isnan(E_scan(i,j,k)))
                amrex::Print() << "[NSCBC-NaN] E NaN at ("
                    << i << "," << j << "," << k << ")"
                    << (in_valid ? " [interior]" : " [ghost]") << "\n";
            if (amrex::isnan(eta_scan(i,j,k)))
                amrex::Print() << "[NSCBC-NaN] eta NaN at ("
                    << i << "," << j << "," << k << ")"
                    << (in_valid ? " [interior]" : " [ghost]") << "\n";
        });
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::Advance");

    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();
    const Box& domain = geom.Domain();

    // // check MF are defined correctly
    // amrex::Print() << "CHECK MF CONSISTENCY:\n";
    // amrex::Print() << "eta        nComp=" << eta_mf[lev]->nComp()
    //             << "  nGrow=" << eta_mf[lev]->nGrow() << "\n";
    // amrex::Print() << "etaOld     nComp=" << eta_old_mf[lev]->nComp()
    //             << "  nGrow=" << eta_old_mf[lev]->nGrow() << "\n";
    // amrex::Print() << "density    nComp=" << density_mf[lev]->nComp()
    //             << "  nGrow=" << density_mf[lev]->nGrow() << "\n";
    // amrex::Print() << "densityOld nComp=" << density_old_mf[lev]->nComp()
    //             << "  nGrow=" << density_old_mf[lev]->nGrow() << "\n";

    // amrex::Print() << "momentum    nComp=" << momentum_mf[lev]->nComp()
    //             << "  nGrow=" << momentum_mf[lev]->nGrow() << "\n";
    // amrex::Print() << "momentumOld nComp=" << momentum_old_mf[lev]->nComp()
    //             << "  nGrow=" << momentum_old_mf[lev]->nGrow() << "\n";

    // amrex::Print() << "energyVol    nComp=" << energy_per_vol_mf[lev]->nComp()
    //             << "  nGrow=" << energy_per_vol_mf[lev]->nGrow() << "\n";
    // amrex::Print() << "energyVolOld nComp=" << energy_per_vol_old_mf[lev]->nComp()
    //             << "  nGrow=" << energy_per_vol_old_mf[lev]->nGrow() << "\n";

    // if (density_mf[lev]->boxArray() != (density_old_mf[lev]->boxArray()))
    //     amrex::Abort("density_mf and density_old_mf boxArrays do NOT match!");

    // if (momentum_mf[lev]->boxArray() != (momentum_old_mf[lev]->boxArray()))
    //     amrex::Abort("momentum_mf and momentum_old_mf boxArrays do NOT match!");

    // if (energy_per_vol_mf[lev]->boxArray() != (energy_per_vol_old_mf[lev]->boxArray()))
    //     amrex::Abort("energy_per_vol_mf and energy_per_vol_old_mf boxArrays do NOT match!");

    //==============================================================
    // 1. Swap old/new pointers for the *real* solution MultiFabs
    //==============================================================
    // std::swap(eta_old_mf[lev],          eta_mf[lev]);
    // std::swap(density_old_mf[lev],      density_mf[lev]);
    // std::swap(momentum_old_mf[lev],     momentum_mf[lev]);
    // std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    // std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    // actually copy instead of swap
    eta_old_mf[lev]->ParallelCopy(*eta_mf[lev]);
    density_old_mf[lev]->ParallelCopy(*density_mf[lev]);
    momentum_old_mf[lev]->ParallelCopy(*momentum_mf[lev]);
    energy_per_vol_old_mf[lev]->ParallelCopy(*energy_per_vol_mf[lev]);
    energy_per_mass_old_mf[lev]->ParallelCopy(*energy_per_mass_mf[lev]);

    //==============================================================
    // 2. Build *fresh*, fully allocated MultiFabs for TimeIntegrator
    //    No aliasing. No component mismatch. No ghost mismatch.
    //==============================================================
    amrex::Vector<amrex::MultiFab> solution_new;
    amrex::Vector<amrex::MultiFab> solution_old;

    // Build solution_new from the current-time MultiFabs.
    // Use src_ngrow=nGrow so that the coarse-fine ghost cells (filled by
    // FillPatch in Integrator::timeStep before calling Advance) are preserved.
    // eta
    solution_new.emplace_back(eta_mf[lev]->boxArray(),
                              eta_mf[lev]->DistributionMap(),
                              eta_mf[lev]->nComp(),
                              eta_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*eta_mf[lev], 0, 0, eta_mf[lev]->nComp(),
                                     eta_mf[lev]->nGrow(), eta_mf[lev]->nGrow());

    // density
    solution_new.emplace_back(density_mf[lev]->boxArray(),
                              density_mf[lev]->DistributionMap(),
                              density_mf[lev]->nComp(),
                              density_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*density_mf[lev], 0, 0, density_mf[lev]->nComp(),
                                     density_mf[lev]->nGrow(), density_mf[lev]->nGrow());

    // momentum (2 components)
    solution_new.emplace_back(momentum_mf[lev]->boxArray(),
                              momentum_mf[lev]->DistributionMap(),
                              momentum_mf[lev]->nComp(),
                              momentum_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*momentum_mf[lev], 0, 0, momentum_mf[lev]->nComp(),
                                     momentum_mf[lev]->nGrow(), momentum_mf[lev]->nGrow());

    // energy per volume
    solution_new.emplace_back(energy_per_vol_mf[lev]->boxArray(),
                              energy_per_vol_mf[lev]->DistributionMap(),
                              energy_per_vol_mf[lev]->nComp(),
                              energy_per_vol_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*energy_per_vol_mf[lev], 0, 0, energy_per_vol_mf[lev]->nComp(),
                                     energy_per_vol_mf[lev]->nGrow(), energy_per_vol_mf[lev]->nGrow());


    //==============================================================
    // 3. Fill ghosts for solution_new BEFORE integration.
    //    coarse-fine ghost cells were already included via src_ngrow=nGrow
    //    in the ParallelCopy above; FillBoundary handles physical BCs
    //    and same-level neighbor communication.
    //==============================================================
    // DIAGNOSTIC: check for NaN right after ParallelCopy, before FillBoundary
    {
        const char* fnames[4] = {"eta","density","momentum","energy"};
        for (int n = 0; n < 4; n++) {
            if (solution_new[n].contains_nan(0, solution_new[n].nComp(), solution_new[n].nGrowVect()))
                amrex::Print() << "[DIAG-post-copy] NaN in " << fnames[n]
                               << " (incl ghosts) BEFORE FillBoundary lev=" << lev << " t=" << time << "\n";
            if (solution_new[n].contains_nan()) {
                amrex::Print() << "[DIAG-post-copy] NaN in " << fnames[n]
                               << " (valid only) BEFORE FillBoundary lev=" << lev << " t=" << time << "\n";
                // Print specific NaN cell locations (CPU scan, diagnostic only)
                for (amrex::MFIter mfi(solution_new[n], false); mfi.isValid(); ++mfi) {
                    auto const& arr = solution_new[n].array(mfi);
                    const amrex::Box vbx = mfi.validbox();
                    amrex::LoopOnCpu(vbx, [&](int i, int j, int k) {
                        for (int c = 0; c < solution_new[n].nComp(); ++c)
                            if (amrex::isnan(arr(i,j,k,c)))
                                amrex::Print() << "[NaN-LOC] " << fnames[n]
                                               << " comp=" << c
                                               << " at (" << i << "," << j << "," << k << ")"
                                               << " lev=" << lev << " t=" << time << "\n";
                    });
                }
            }
        }
        // Also check pressure_mf (a separately tracked field, not in solution_new)
        if (pressure_mf[lev]->contains_nan(0, 1, amrex::IntVect(AMREX_D_DECL(0,0,0)))) {
            amrex::Print() << "[DIAG-post-copy] NaN in pressure_mf (valid only) lev=" << lev << " t=" << time << "\n";
            for (amrex::MFIter mfi(*pressure_mf[lev], false); mfi.isValid(); ++mfi) {
                auto const& parr = pressure_mf[lev]->array(mfi);
                const amrex::Box vbx = mfi.validbox();
                amrex::LoopOnCpu(vbx, [&](int i, int j, int k) {
                    if (amrex::isnan(parr(i,j,k)))
                        amrex::Print() << "[NaN-LOC] pressure at (" << i << "," << j << "," << k
                                       << ") lev=" << lev << " t=" << time << "\n";
                });
            }
        }
    }
    energy_bc->FillBoundary(solution_new[0], 0, solution_new[0].nComp(), time, 0);
    density_bc->FillBoundary(solution_new[1], 0, solution_new[1].nComp(), time, 0);
    momentum_bc->FillBoundary(solution_new[2], 0, solution_new[2].nComp(), time, 0);
    energy_bc->FillBoundary(solution_new[3], 0, solution_new[3].nComp(), time, 0);

    // Fill any NaN ghost cells not reached by same-level FillBoundary
    // (coarse-fine gap regions when nghost > 2) by zero-gradient extrapolation
    // from the nearest valid cell in the same fab.
    {
        auto fillNaNGhosts = [](amrex::MultiFab& mf) {
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const amrex::Box gbx = mfi.fabbox();
                const amrex::Box vbx = mfi.validbox();
                auto arr = mf.array(mfi);
                int  nc  = mf.nComp();
                amrex::ParallelFor(gbx, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    bool is_valid = AMREX_D_TERM(
                        (i >= vbx.smallEnd(0) && i <= vbx.bigEnd(0)),
                     && (j >= vbx.smallEnd(1) && j <= vbx.bigEnd(1)),
                     && (k >= vbx.smallEnd(2) && k <= vbx.bigEnd(2)));
                    if (!is_valid && !std::isfinite(arr(i,j,k,n))) {
                        int ic = amrex::max(vbx.smallEnd(0), amrex::min(vbx.bigEnd(0), i));
                        int jc = amrex::max(vbx.smallEnd(1), amrex::min(vbx.bigEnd(1), j));
#if AMREX_SPACEDIM == 3
                        int kc = amrex::max(vbx.smallEnd(2), amrex::min(vbx.bigEnd(2), k));
#else
                        int kc = k;
#endif
                        arr(i,j,k,n) = arr(ic,jc,kc,n);
                    }
                });
            }
        };
        for (auto& mf : solution_new) fillNaNGhosts(mf);

        // Sanitize any remaining NaN in valid cells (can arise from FillCoarsePatch
        // using NaN-contaminated coarse ghost cells during AMR initialization).
        auto fillNaNValid = [](amrex::MultiFab& mf) {
            for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
                const amrex::Box vbx = mfi.validbox();
                auto arr = mf.array(mfi);
                int nc = mf.nComp();
                amrex::ParallelFor(vbx, nc, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                    if (!std::isfinite(arr(i,j,k,n))) arr(i,j,k,n) = amrex::Real(0.0);
                });
            }
        };
        for (auto& mf : solution_new) fillNaNValid(mf);
    }

    // DIAGNOSTIC: check for NaN introduced by FillBoundary
    {
        const char* fnames[4] = {"eta","density","momentum","energy"};
        for (int n = 0; n < 4; n++)
            if (solution_new[n].contains_nan(0, solution_new[n].nComp(), solution_new[n].nGrowVect()))
                amrex::Print() << "[DIAG-pre-NSCBC] NaN in " << fnames[n]
                               << " (incl ghosts) AFTER FillBoundary lev=" << lev << " t=" << time << "\n";
    }
    if (nscbc.enabled)
        ApplyNSCBC(lev, solution_new[0], solution_new[1], solution_new[2], solution_new[3], time);
    // DIAGNOSTIC: check for NaN after ApplyNSCBC
    if (nscbc.enabled) {
        const char* fnames[4] = {"eta","density","momentum","energy"};
        for (int n = 0; n < 4; n++) {
            if (solution_new[n].contains_nan(0, solution_new[n].nComp(), solution_new[n].nGrowVect()))
                amrex::Print() << "[DIAG-post-NSCBC] NaN in " << fnames[n]
                               << " (incl ghosts) t=" << time << "\n";
            if (solution_new[n].contains_nan())
                amrex::Print() << "[DIAG-post-NSCBC] NaN in " << fnames[n]
                               << " (valid cells only) t=" << time << "\n";
        }
    }

    //==============================================================
    // Build solution_old as a complete copy of solution_new (valid
    // cells + ghost cells).  The FEIntegrator copies S_old→S_new at
    // the start of advance(), so solution_old MUST have valid ghost
    // cells or it will overwrite the valid ghost cells we just built.
    // For multi-step RK methods solution_old is the beginning-of-step
    // state, which at this point equals solution_new, so this is
    // always correct.
    //==============================================================
    for (int n = 0; n < (int)solution_new.size(); n++) {
        solution_old.emplace_back(solution_new[n].boxArray(),
                                  solution_new[n].DistributionMap(),
                                  solution_new[n].nComp(),
                                  solution_new[n].nGrow());
        amrex::MultiFab::Copy(solution_old.back(), solution_new[n], 0, 0,
                              solution_new[n].nComp(), solution_new[n].nGrow());
    }

    //==============================================================
    // 4. Set up AMReX TimeIntegrator
    //==============================================================
    amrex::TimeIntegrator timeintegrator(solution_new, time);

    // RHS functor
    timeintegrator.set_rhs([&](
        amrex::Vector<amrex::MultiFab>& rhs_mf,
        amrex::Vector<amrex::MultiFab>& sol_mf,
        const Real t)
    {
        // Each sol_mf[i] has correct structure & ghost cells
        RHS(lev, t,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3],
            sol_mf[0], sol_mf[1], sol_mf[2], sol_mf[3]);
    });

    //==============================================================
    // 5. Fill ghosts after each stage (REQUIRED for multi-stage methods)
    //==============================================================
    timeintegrator.set_post_stage_action(
        [&](amrex::Vector<amrex::MultiFab>& stage, Real t)
    {
        // Apply physical BCs (which also calls FillBoundary internally)
        energy_bc->FillBoundary(stage[0], 0, stage[0].nComp(), t, 0);   // eta
        density_bc->FillBoundary(stage[1], 0, stage[1].nComp(), t, 0);  // density
        momentum_bc->FillBoundary(stage[2], 0, stage[2].nComp(), t, 0); // momentum
        energy_bc->FillBoundary(stage[3], 0, stage[3].nComp(), t, 0);   // energy
        // DIAGNOSTIC: check for NaN introduced by FillBoundary (stage)
        {
            const char* snames[4] = {"eta","density","momentum","energy"};
            for (int n = 0; n < 4; n++)
                if (stage[n].contains_nan(0, stage[n].nComp(), stage[n].nGrowVect()))
                    amrex::Print() << "[DIAG-stage-pre-NSCBC] NaN in " << snames[n]
                                   << " (incl ghosts) t=" << t << "\n";
        }
        if (nscbc.enabled)
            ApplyNSCBC(lev, stage[0], stage[1], stage[2], stage[3], t);
        // DIAGNOSTIC: check for NaN after ApplyNSCBC (stage)
        if (nscbc.enabled) {
            const char* snames[4] = {"eta","density","momentum","energy"};
            for (int n = 0; n < 4; n++) {
                if (stage[n].contains_nan(0, stage[n].nComp(), stage[n].nGrowVect()))
                    amrex::Print() << "[DIAG-stage-post-NSCBC] NaN in " << snames[n]
                                   << " (incl ghosts) t=" << t << "\n";
                if (stage[n].contains_nan())
                    amrex::Print() << "[DIAG-stage-post-NSCBC] NaN in " << snames[n]
                                   << " (valid cells only) t=" << t << "\n";
            }
        }
    });

    //==============================================================
    // 6. Integrate in time
    //==============================================================
    timeintegrator.advance(solution_old, solution_new, time, dt);

    //==============================================================
    // 7. Copy new state back into the owning MultiFabs
    //==============================================================
    eta_mf[lev]->ParallelCopy(solution_new[0]);
    density_mf[lev]->ParallelCopy(solution_new[1]);
    momentum_mf[lev]->ParallelCopy(solution_new[2]);
    energy_per_vol_mf[lev]->ParallelCopy(solution_new[3]);

    //==============================================================
    // 7b. Implicit Cahn-Hilliard biharmonic step (operator split)
    //     Solves (I + sqrt_alpha*(-∇²))^2 η = η* + dt*M*∇²f'(η*)
    //     unconditionally stable — no dx^4 timestep penalty.
    //==============================================================
    if (!static_eta) {
        if      (implicit_ch == 1) ApplyImplicitCH(lev, dt);
        else if (implicit_ch == 2) ApplyImplicitCH_Newton(lev, dt);
    }

    //==============================================================
    // 8. Fill ghosts after final update
    //==============================================================
    eta_mf[lev]->FillBoundary(geom.periodicity());
    density_mf[lev]->FillBoundary(geom.periodicity());
    momentum_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_vol_mf[lev]->FillBoundary(geom.periodicity());

    //==============================================================
    // 9. Compute visualization + CFL quantities
    //==============================================================
    c_max   = 0.0;
    vx_max  = 0.0;
    vy_max  = 0.0;
    F_max   = 0.0;
    rho_min = 1e10;

    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Eta
        auto eta        = eta_mf[lev]->array(mfi);
        auto eta_old    = eta_old_mf[lev]->const_array(mfi);
        auto etadot     = etadot_mf[lev]->array(mfi);
        auto grad_eta   = grad_eta_mf[lev]->const_array(mfi);
        auto n_hat      = n_hat_mf[lev]->const_array(mfi);
        auto kappas     = kappas_mf[lev]->const_array(mfi);
        auto grad_mag_grad_eta  = grad_mag_grad_eta_mf[lev]->const_array(mfi);

        // Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        // Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf.Patch(lev, mfi);

        // Mixture
        auto rho        = density_mf[lev]->array(mfi);
        auto E_vol      = energy_per_vol_mf[lev]->const_array(mfi);
        auto E_mass     = energy_per_mass_mf[lev]->const_array(mfi);
        auto M          = momentum_mf[lev]->array(mfi);

        // Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        // Set::Patch<const Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
        // Set::Patch<const Set::Scalar> E_mas = energy_per_mas_mf.Patch(lev, mfi);
        // Set::Patch<const Set::Scalar> M = momentum_mf.Patch(lev, mfi);

        auto a       = a_mf[lev]->array(mfi);
        auto Ma      = Ma_mf[lev]->array(mfi);
        auto KE_vol  = KE_per_vol_mf[lev]->array(mfi);
        auto KE_mass = KE_per_mass_mf[lev]->array(mfi);
        auto UE_vol  = UE_per_vol_mf[lev]->array(mfi);
        auto UE_mass = UE_per_mass_mf[lev]->array(mfi);

        // Set::Patch<Set::Scalar> a = a_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> Ma = Ma_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> KE_vol = KE_per_vol_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> KE_mas = KE_per_mas_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> UE_vol = UE_per_vol_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> UE_mas = UE_per_mas_mf.Patch(lev, mfi);

        auto v      = velocity_mf[lev]->array(mfi);
        auto p      = pressure_mf[lev]->array(mfi);
        auto Source = Source_mf[lev]->const_array(mfi);

        // Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        // Set::Patch<const Set::Scalar> Source = Source_mf.Patch(lev, mfi);


        // //Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        // //Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        // //Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        // //Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);

        auto gamma_mix  = gamma_mf[lev]->array(mfi);
        auto pi_mix     = pi_mf[lev]->array(mfi);

        // Set::Patch<Set::Scalar> gamma_mix = gamma_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> pi_mix = pi_mix_mf.Patch(lev, mfi);

        auto mu_chem    = mu_chem_mf[lev]->array(mfi);
        auto Bm         = Bm_mf[lev]->array(mfi);

        // Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        // Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // CUTOFFS
            eta(i, j, k) = std::max( 0.0,std::min( 1.0,(eta(i, j, k) - cutoff) / (1.0 - 2.0 * cutoff) ) );
            if (!std::isfinite(rho(i, j, k)) || rho(i, j, k) < small) rho(i, j, k) = small;

            // Derivatice Function Calls
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // gamma
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * pi_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * pi_1) / (gamma1 - 1.0);
            // gamma(i, j, k) = 1.0 + (1.0 / A);

            double gmix, pimix;
            Thermo_Interp::InterpolateGammaPi_Stiffened(
                eta(i,j,k), gamma1, gamma0, pi_1, pi_0, gmix, pimix);
            gamma_mix(i,j,k) = gmix;
            pi_mix(i,j,k) = pimix;

            // etadot
            etadot(i, j, k) = (eta(i, j, k) - eta_old(i, j, k)) / dt;

            // Velocity = M ./ (DX*DY*rho)
            // Guard against rho=0 (freshly created AMR cells or exact-zero density
            // after time advance) to prevent 0/0=NaN propagating to pressure.
            {
                Set::Scalar safe_rho_ = (rho(i,j,k) > Set::Scalar(1e-20))
                                      ? rho(i,j,k) : Set::Scalar(1e-20);
                Set::Scalar vx_ = M(i, j, k, 0) / safe_rho_;
                Set::Scalar vy_ = M(i, j, k, 1) / safe_rho_;
                // Clamp Inf/NaN: M/1e-20 can be huge-but-finite when rho→0;
                // Inf in velocity_mf causes Inf*0=NaN in source terms next step.
                if (!std::isfinite(vx_)) vx_ = Set::Scalar(0.0);
                if (!std::isfinite(vy_)) vy_ = Set::Scalar(0.0);
                // Cap huge-but-finite velocity to prevent KE overflow (v²→Inf):
                // bad NSCBC corner ghost cells can accumulate M over many steps.
                const Set::Scalar v_cap = Set::Scalar(1e8); // unphysical upper bound
                if (vx_ >  v_cap) vx_ =  v_cap;
                if (vx_ < -v_cap) vx_ = -v_cap;
                if (vy_ >  v_cap) vy_ =  v_cap;
                if (vy_ < -v_cap) vy_ = -v_cap;
                v(i, j, k, 0) = vx_;
                v(i, j, k, 1) = vy_;
            }

            // Kinetic Energy
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            if (!std::isfinite(KE_vol(i, j, k))) KE_vol(i, j, k) = Set::Scalar(0.0);
            KE_mass(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            if (!std::isfinite(KE_mass(i, j, k))) KE_mass(i, j, k) = Set::Scalar(0.0);

            // Internal Energy — NaN-safe clamp (UE < 0 check misses NaN: NaN < 0 = false)
            UE_vol(i, j, k) = E_vol(i, j, k) - KE_vol(i, j, k);
            if (!std::isfinite(UE_vol(i, j, k)) || UE_vol(i, j, k) < 0.0)
                UE_vol(i, j, k) = 0.0;
            UE_mass(i, j, k) = E_mass(i, j, k) - KE_mass(i, j, k);
            if (!std::isfinite(UE_mass(i, j, k)) || UE_mass(i, j, k) < 0.0)
                UE_mass(i, j, k) = 0.0;

            // Pressure — NaN-safe clamp
            p(i, j, k) = (gmix - 1.0) * UE_vol(i, j, k) - gmix * pimix + pref; // pressure Tammann EOS modification
            if (!std::isfinite(p(i, j, k)) || p(i, j, k) < 0.0)
                p(i, j, k) = 1e-6;

            // Chemical Potential
            Set::Scalar f_prime = ch_W_scale * 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: W_scale*f'(eta)
            Set::Scalar mu_chem_local = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem(i, j, k) = mu_chem_local;

            // Spalding Number
            Bm(i, j, k) = eta(i, j, k) / (1.0 - eta(i, j, k) + small);

            // Speed of sound:
            {
                Set::Scalar safe_rho_a = (std::isfinite(rho(i,j,k)) && rho(i,j,k) > small) ? rho(i,j,k) : small;
                a(i, j, k) = sqrt(gamma_mix(i, j, k) * (p(i, j, k) + pi_mix(i, j, k)) / safe_rho_a);
                if (!std::isfinite(a(i, j, k))) a(i, j, k) = Set::Scalar(0.0);
            }

            // Mach Number
            {
                Set::Scalar max_ = v(i, j, k, 0) / (a(i, j, k) + small);
                Set::Scalar may_ = v(i, j, k, 1) / (a(i, j, k) + small);
                Ma(i, j, k, 0) = std::isfinite(max_) ? max_ : Set::Scalar(0.0);
                Ma(i, j, k, 1) = std::isfinite(may_) ? may_ : Set::Scalar(0.0);
            }

            // // Curvature
            // Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            // grad_eta_(i, j, k, 0) = grad_eta(0);
            // grad_eta_(i, j, k, 1) = grad_eta(1);

            // // Debugging, would like to delete condition
            // if (grad_eta_mag < 1e-4)
            // {
            //     n_hat_(i, j, k, 0) = 0.0;
            //     n_hat_(i, j, k, 1) = 0.0;
            // }
            // else
            // {
            //     n_hat_(i, j, k, 0) = n_hat(0);
            //     n_hat_(i, j, k, 1) = n_hat(1);
            // }


            // ------------------------------------------------------------
            // Adaptive Timestepping
            // ------------------------------------------------------------
            // Solving for new states
            Set::Scalar A_new = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B_new = (eta(i, j, k) * gamma0 * pi_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * pi_1) / (gamma1 - 1.0);
            // Set::Scalar gmix_new = 1.0 + (1.0 / A_new);
            Set::Scalar safe_rho_dt = (std::isfinite(rho(i,j,k)) && rho(i,j,k) > small) ? rho(i,j,k) : small;
            Set::Vector v_new = Set::Vector(M(i, j, k, 0) / safe_rho_dt, M(i, j, k, 1) / safe_rho_dt);
            if (!std::isfinite(v_new(0))) v_new(0) = Set::Scalar(0.0);
            if (!std::isfinite(v_new(1))) v_new(1) = Set::Scalar(0.0);
            if (v_new(0) >  Set::Scalar(1e8)) v_new(0) =  Set::Scalar(1e8);
            if (v_new(0) < -Set::Scalar(1e8)) v_new(0) = -Set::Scalar(1e8);
            if (v_new(1) >  Set::Scalar(1e8)) v_new(1) =  Set::Scalar(1e8);
            if (v_new(1) < -Set::Scalar(1e8)) v_new(1) = -Set::Scalar(1e8);
            Set::Scalar KE_vol_new = 0.5 * safe_rho_dt * (v_new(0) * v_new(0) + v_new(1) * v_new(1));

            Set::Scalar UE_vol_new = E_vol(i, j, k) - KE_vol_new;
            if (!std::isfinite(UE_vol_new) || UE_vol_new < 0.0)
            {
                UE_vol_new = 0.0;
            }
            Set::Scalar p_new = (UE_vol_new - B_new) / A_new;
            if (!std::isfinite(p_new) || p_new < 0.0)
            {
                p_new = 1e-6;
            }
            // Set::Scalar pi_mix_new = (B_new / A_new) / gmix_new;
            double gmix_new, pimix_new;
            Thermo_Interp::InterpolateGammaPi_Stiffened(
                eta(i,j,k), gamma1, gamma0, pi_1, pi_0, gmix_new, pimix_new);
            // gamma_mix(i,j,k) = gmix;
            // pi_mix_new(i,j,k) = pimix;
            Set::Scalar sound_speed_new = sqrt(gmix_new * (p_new + pimix_new) / safe_rho_dt);

            if (std::isfinite(sound_speed_new)) c_max = std::max(c_max, sound_speed_new);
            if (std::isfinite(v_new(0))) vx_max = std::max(vx_max, std::abs(v_new(0))); // vx
            if (std::isfinite(v_new(1))) vy_max = std::max(vy_max, std::abs(v_new(1))); // vy

            // Track maximum force magnitude (not acceleration yet)
            Set::Scalar F_mag = sqrt(Source(i, j, k, 1) * Source(i, j, k, 1) + Source(i, j, k, 2) * Source(i, j, k, 2));
            if (std::isfinite(F_mag)) F_max = std::max(F_max, F_mag);
            if (std::isfinite(rho(i, j, k))) rho_min = std::min(rho_min, rho(i, j, k));

        });
    }

    // Diagnostic: check Ma_mf immediately after visualization kernel
    if (Ma_mf[lev]->contains_nan(0, Ma_mf[lev]->nComp(), 0))
        amrex::Print() << "[POST-VIZ] Ma_mf lev=" << lev << " has VALID-CELL NaN after viz kernel\n";
    else
        amrex::Print() << "[POST-VIZ] Ma_mf lev=" << lev << " clean after viz kernel\n";

    // ------------------------------------------------------------
    // Dynamic Timesteping
    // ------------------------------------------------------------
    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();

    // Update adaptive timestep
    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);
    amrex::ParallelDescriptor::ReduceRealMax(F_max);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min);

    // Compute timestep constraints
    Set::Scalar dx_min = std::min(DX[0], DX[1]);

    // 1. Acoustic CFL
    Set::Scalar vel_mag = sqrt(vx_max * vx_max + vy_max * vy_max);
    if (!std::isfinite(vel_mag)) vel_mag = Set::Scalar(0.0);
    Set::Scalar wave_speed = c_max + vel_mag;
    if (!std::isfinite(wave_speed)) wave_speed = Set::Scalar(0.0);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    // 2. Viscous CFL
    Set::Scalar mu_max = std::max(mu0, mu1);
    Set::Scalar dt_viscous = cfl_v * rho_min * dx_min * dx_min / (mu_max + small);

    // 3. Force CFL (guard: if F_max=0 no force constraint; if a_max=Inf skip)
    Set::Scalar dt_force = std::numeric_limits<Set::Scalar>::max();
    if (F_max > Set::Scalar(0.0) && rho_min > Set::Scalar(0.0)) {
        Set::Scalar a_max = F_max / rho_min;
        if (std::isfinite(a_max) && a_max > Set::Scalar(0.0))
            dt_force = cfl_v * sqrt(dx_min / a_max);
    }

    // 4. Allen-Cahn diffusion CFL
    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    // Take minimum
    dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn });

    // Safety factor
    dt_max = dt_max * 0.9;

    // Timestep diagnostics
    bool timestep_verbose = false;
    if (timestep_verbose == true)
    {
        Util::ParallelMessage(INFO, "\n=== CFL TIMESTEP DIAGNOSTICS ===");
        Util::ParallelMessage(INFO, "Grid spacing: ", dx_min, " m");
        Util::ParallelMessage(INFO, "Sound speed max: ", c_max, " m/s");
        Util::ParallelMessage(INFO, "Velocity max: ", std::max(vx_max, vy_max), " m/s");
        Util::ParallelMessage(INFO, "Force max: ", F_max, " N/m^3");
        Util::ParallelMessage(INFO, "Density min: ", rho_min, " kg/m^3");
        Util::ParallelMessage(INFO, "");
        Util::ParallelMessage(INFO, "dt_acoustic: ", dt_acoustic, " s");
        Util::ParallelMessage(INFO, "dt_viscous: ", dt_viscous, " s");
        Util::ParallelMessage(INFO, "dt_force: ", dt_force, " s");
        Util::ParallelMessage(INFO, "dt_allen_cahn: ", dt_allen_cahn, " s");
        Util::ParallelMessage(INFO, "");
        Util::ParallelMessage(INFO, "Final dt_max: ", dt_max, " s");
        Util::ParallelMessage(INFO, "================================\n");
    }
    if (dynamictimestep.on)
    {
        this->DynamicTimestep_SyncTimeStep(lev, dt_max);
    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// REGRIDDING //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Regrid(int lev, Set::Scalar /* time */)
{
    BL_PROFILE("Integrator::Hydro2::Regrid");
    Source_mf[lev]->setVal(0.0);
    if (lev < finest_level) return;

    Util::Message(INFO, "Regridding on level", lev);
}//end regrid

//void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray& a_tags, Set::Scalar, int)
{
    BL_PROFILE("Integrator::Flame::TagCellsForRefinement");

    const Set::Scalar* DX = geom[lev].CellSize();
    Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));
    const Box& domain = geom[lev].Domain();

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

    // Pressure criterion for refinement
    for (amrex::MFIter mfi(*pressure_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& p_mix = (*pressure_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Vector grad_p = Numeric::Gradient(p_mix, i, j, k, 0, DX, sten);
            if (grad_p.lpNorm<2>() * dr * 2 > p_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
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

}

#pragma GCC diagnostic pop

//#endif
