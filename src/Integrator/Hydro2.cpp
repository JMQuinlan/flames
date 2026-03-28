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

using namespace amrex;


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
        pp_query_default("p0_0", value.p0_0, 0.0);      // p0 for Tammann EOS
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient
        pp_query_default("cp0", value.cp0, 0.0);        // Constant Pressure Specific Heat [J/kg]
        pp_query_default("cv0", value.cv0, 0.0);        // Constant Volume Specific Heat [J/kg]
        // pp_query_required("R0", value.R0);              // Specific Gas Constant
        // pp_query_required("MW0", value.MW0);            // Molecular Weight
        
        // FLUID 1
        pp_query_required("gamma1", value.gamma1);      // gamma for gamma law
        pp_query_default("p0_1", value.p0_1, 0.0);      // p0 for Tammann EOS
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
        pp_query_default("kappa_method", value.kappa_method, 2); // Method to solve for curvature
        
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
        int nghost = 2;

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

        value.RegisterNewFab(value.pressure0_mf,    value.energy_bc,  1, nghost, "pressure0", false, false);
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
        value.RegisterNewFab(value.pressure_mf,     value.energy_bc, 1, nghost, "pressure", true, false);
        value.RegisterNewFab(value.velocity_mf,     &value.bc_nothing,  2, nghost, "velocity", true, false, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf,    &value.bc_nothing,  1, nghost, "vorticity", true, false);
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true, true);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false, true);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost, "energy_per_vol", true, true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost, "energy_per_mass", true, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost, "energy_vol_old", false, true);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost, "energy_mas_old", false, true);
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
        value.RegisterNewFab(value.p0_mf,           value.energy_bc, 1, nghost, "p0", true, true);                    // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc, 1, nghost, "mu_chem", true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true, false);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, false, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   &value.bc_nothing,  1, nghost, "UE_per_vol", true, false);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   &value.bc_nothing,  1, nghost, "UE_per_mass", true, false);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   &value.bc_nothing,  1, nghost, "KE_per_vol", true, false);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   &value.bc_nothing,  1, nghost, "KE_per_mass", true, false);        // Kinetic Energy (per unit mass)
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
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, false, { "Avg", "1", "2" }); // To Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", false, false, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,     &value.bc_nothing,  1, nghost, "rho_flux", true, false);                    // Density Flux
        value.RegisterNewFab(value.M_flux_mf,       &value.bc_nothing,  2, nghost, "M_flux", true, false, { "x", "y" });        // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,       &value.bc_nothing,  1, nghost, "E_flux", true, false);                      // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,      &value.bc_nothing,  2, nghost, "div_tau", true, false, { "x", "y" });            // Energy Flux
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

    // Calculate mixed variables based on individual fluid variables
    Mix(lev);

    // Make sure ghost cells are consistent with initial conditions
    eta_mf[lev]->FillBoundary(geom.periodicity());
    density_mf[lev]->FillBoundary(geom.periodicity());
    momentum_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_vol_mf[lev]->FillBoundary(geom.periodicity());

    // NATURAL SOURCE
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Fw_mf[lev]      ->setVal(0.0); 
    Ldot_mf[lev]    ->setVal(0.0);
    Vap_dot_mf[lev] ->setVal(0.0); 

    // BOUNDRY CURVATURE AND THINGS
    kappas_mf[lev]  ->setVal(0.0);
    grad_mag_grad_eta_mf[lev]->setVal(0.0);
    Bm_mf[lev]      ->setVal(0.0);  // Spalding Number

    // MIXED PROPERTIES
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
    BL_PROFILE("Hydro2::Mix");

    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();

    //----------------------------------------------------------------------
    // 1. Compute mixtures only on valid region
    //----------------------------------------------------------------------
    for (MFIter mfi(*eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        auto eta  = eta_mf[lev]->const_array(mfi);

        auto v0   = velocity0_mf[lev]->const_array(mfi);
        auto v1   = velocity1_mf[lev]->const_array(mfi);

        auto p0   = pressure0_mf[lev]->const_array(mfi);
        auto p1   = pressure1_mf[lev]->const_array(mfi);

        auto rho0 = density0_mf[lev]->const_array(mfi);
        auto rho1 = density1_mf[lev]->const_array(mfi);

        auto E0   = energy0_mf[lev]->const_array(mfi);
        auto E1   = energy1_mf[lev]->const_array(mfi);

        auto rho      = density_mf[lev]->array(mfi);
        auto rho_old  = density_old_mf[lev]->array(mfi);

        auto M        = momentum_mf[lev]->array(mfi);
        auto M_old    = momentum_old_mf[lev]->array(mfi);

        auto E_vol    = energy_per_vol_mf[lev]->array(mfi);
        auto E_vol_old= energy_per_vol_old_mf[lev]->array(mfi);

        auto E_mas    = energy_per_mas_mf[lev]->array(mfi);
        auto E_mas_old= energy_per_mas_old_mf[lev]->array(mfi);

        auto press    = pressure_mf[lev]->array(mfi);
        auto v_mix    = velocity_mf[lev]->array(mfi);
        auto gammaf   = gamma_mf[lev]->array(mfi);
        auto p0eff    = p0_mf[lev]->array(mfi);
        auto T_arr    = T_mf[lev]->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            //------------------------------------------------------------------
            // Mixture interpolation
            //------------------------------------------------------------------
            Real e  = eta(i,j,k);
            Real r0 = rho0(i,j,k);
            Real r1 = rho1(i,j,k);

            Real u0x = v0(i,j,k,0);
            Real u0y = v0(i,j,k,1);
            Real u1x = v1(i,j,k,0);
            Real u1y = v1(i,j,k,1);

            Real rho_mix = e*r0 + (1.0-e)*r1;
            rho(i,j,k)      = rho_mix;
            rho_old(i,j,k)  = rho_mix;

            Real Mx = e*r0*u0x + (1.0-e)*r1*u1x;
            Real My = e*r0*u0y + (1.0-e)*r1*u1y;
            M(i,j,k,0)      = Mx;
            M(i,j,k,1)      = My;
            M_old(i,j,k,0)  = Mx;
            M_old(i,j,k,1)  = My;

            Real vx = Mx/rho_mix;
            Real vy = My/rho_mix;
            v_mix(i,j,k,0) = vx;
            v_mix(i,j,k,1) = vy;

            //------------------------------------------------------------------
            // Energy (per volume)
            //------------------------------------------------------------------
            Real E0v = E0(i,j,k);
            Real E1v = E1(i,j,k);
            Real Ev_mix = e*E0v + (1.0-e)*E1v;

            Real KEv = 0.5*rho_mix*(vx*vx + vy*vy);
            Real UEv = Ev_mix - KEv;
            if (UEv < 0) UEv = 0;

            E_vol(i,j,k)      = Ev_mix;
            E_vol_old(i,j,k)  = Ev_mix;

            Real Emas = Ev_mix/rho_mix;
            E_mas(i,j,k)      = Emas;
            E_mas_old(i,j,k)  = Emas;

            //------------------------------------------------------------------
            // EOS mixture
            //------------------------------------------------------------------
            Real A = e/(gamma0-1.0) + (1.0-e)/(gamma1-1.0);
            Real B = (e * gamma0 * p0_0)/(gamma0-1.0)
                   + ((1.0-e) * gamma1 * p0_1)/(gamma1-1.0);

            Real gam = 1.0 + 1.0/A;
            gammaf(i,j,k) = gam;

            Real p0eff_local = (B/A)/gam;
            p0eff(i,j,k) = p0eff_local;

            Real p = (gam-1.0)*UEv - gam*p0eff_local + pref;
            if (p < 0) p = 1e-10;
            press(i,j,k) = p;

            // T_arr(i,j,k) =
            //     (p + p0eff_local)/(rho_mix*cv_arr(i,j,k)*(gam-1.0) + 1e-14);
            T_arr(i,j,k) =
                (p + p0eff_local)/(rho_mix*cv0*(gam-1.0) + 1e-14);
        });
    }

    //----------------------------------------------------------------------
    // 2. Fill same-level ghosts
    //----------------------------------------------------------------------
    density_mf[lev]->FillBoundary(geom.periodicity());
    momentum_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_vol_mf[lev]->FillBoundary(geom.periodicity());
    energy_per_mas_mf[lev]->FillBoundary(geom.periodicity());
    velocity_mf[lev]->FillBoundary(geom.periodicity());
    pressure_mf[lev]->FillBoundary(geom.periodicity());
    gamma_mf[lev]->FillBoundary(geom.periodicity());
    p0_mf[lev]->FillBoundary(geom.periodicity());
    T_mf[lev]->FillBoundary(geom.periodicity());
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
    // 0. (Optional) Curvature pipeline
    //---------------------------------------------------------------------------
    if (kappa_method == 3)
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
        auto press_arr = pressure_mf[lev]->array(mfi);
        auto a_arr     = a_mf[lev]->array(mfi);

        auto KE_arr    = KE_per_vol_mf[lev]->array(mfi);
        auto UE_arr    = UE_per_vol_mf[lev]->array(mfi);

        auto cp_arr    = cp_mf[lev]->array(mfi);
        auto cv_arr    = cv_mf[lev]->array(mfi);
        auto T_arr     = T_mf[lev]->array(mfi);

        auto gamma_arr = gamma_mf[lev]->array(mfi);
        auto p0_arr    = p0_mf[lev]->array(mfi);

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

            // Velocity
            Real vx = mx / rho;
            Real vy = my / rho;
            v_arr(i,j,k,0) = vx;
            v_arr(i,j,k,1) = vy;

            // Kinetic Energy
            Real KE = 0.5 * rho * (vx*vx + vy*vy);
            KE_arr(i,j,k) = KE;

            // Internal Energy
            Real E = E_arr(i,j,k);
            Real UE = E - KE;
            if (UE < 0.0) UE = 0.0;
            UE_arr(i,j,k) = UE;

            //------------------------------------------------------------------
            // Gamma-law mixture + Tammann EOS
            //------------------------------------------------------------------
            Real A =   eta/(gamma0-1.0)
                     + (1.0 - eta)/(gamma1-1.0);

            Real B = ( eta * gamma0 * p0_0 )/(gamma0-1.0)
                   + ((1.0 - eta) * gamma1 * p0_1)/(gamma1-1.0);

            Real gamma_eff = 1.0 + 1.0/A;
            gamma_arr(i,j,k) = gamma_eff;

            // Tammann pressure offset
            Real p0_eff = (B/A)/gamma_eff;
            p0_arr(i,j,k) = p0_eff;

            // Pressure
            Real press = (gamma_eff - 1.0)*UE - gamma_eff*p0_eff + pref;
            if (press < 0.0) press = 1e-10;
            press_arr(i,j,k) = press;

            // Specific heats
            cp_arr(i,j,k) = eta*cp0 + (1.0 - eta)*cp1;
            cv_arr(i,j,k) = eta*cv0 + (1.0 - eta)*cv1;

            // Temperature
            T_arr(i,j,k) = (press + p0_eff) / (rho * cv_arr(i,j,k) * (gamma_eff - 1.0) + 1e-14);

            // Speed of sound
            a_arr(i,j,k) = std::sqrt(gamma_eff * (press + p0_eff) / rho);

            //------------------------------------------------------------------
            // Chemical potential + mass fraction
            //------------------------------------------------------------------
            Real gm_eta = Numeric::Gradient(eta_arr, i,j,k, 0, DX).lpNorm<2>();
            Real lap_eta = Numeric::Laplacian(eta_arr, i,j,k, 0, DX);

            Real fprime = 4.0 * eta * (eta-0.5) * (eta-1.0);
            mu_arr(i,j,k) = -epsilon*epsilon*lap_eta + fprime;

            Y_arr(i,j,k) = rho0_arr(i,j,k)*eta / (rho + 1e-14);

            Bm_arr(i,j,k) = (Y_arr(i,j,k) - Y_infinity) / (1.0 + Y_infinity + 1e-14);
        });
    }

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
        auto press_arr = pressure_mf[lev]->array(mfi);
        auto a_arr     = a_mf[lev]->array(mfi);

        auto gamma_arr = gamma_mf[lev]->array(mfi);
        auto p0_arr    = p0_mf[lev]->array(mfi);
        auto T_arr     = T_mf[lev]->array(mfi);

        // Output
        auto eta_rhs = eta_rhs_mf.array(mfi);
        auto rho_rhs = rho_rhs_mf.array(mfi);
        auto M_rhs   = M_rhs_mf.array(mfi);
        auto E_rhs   = E_rhs_mf.array(mfi);

        // Source terms
        auto S_arr    = Source_mf[lev]->array(mfi);

        // For diagnostic fields
        auto rho0_arr = density0_mf[lev]->array(mfi);
        auto rho1_arr = density1_mf[lev]->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            //------------------------------------------------------------------
            // Neighbors (safe indexing)
            //------------------------------------------------------------------
            int il = amrex::max(i-1, bx.smallEnd(0));
            int ir = amrex::min(i+1, bx.bigEnd(0));
            int jl = amrex::max(j-1, bx.smallEnd(1));
            int jr = amrex::min(j+1, bx.bigEnd(1));

            //------------------------------------------------------------------
            // Build Riemann states ONLY from Array4 fields
            //------------------------------------------------------------------

            namespace FR = Solver::Local::FluidRiemann;

            FR::State Sxm = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
                il, j, k, 0);    // x-direction

            FR::State Sxc = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
                i, j, k, 0);

            FR::State Sxp = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
                ir, j, k, 0);

            FR::State Sym = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
                i, jl, k, 1);   // y-direction

            FR::State Syc = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
                i, j, k, 1);

            FR::State Syp = FR::State(
                rho_arr, M_arr, E_arr,
                gamma_arr, p0_arr, T_arr,
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
            // Add source terms
            //------------------------------------------------------------------
            rho_rhs(i,j,k) = drho + S_arr(i,j,k,0);
            M_rhs(i,j,k,0) = dMx  + S_arr(i,j,k,1);
            M_rhs(i,j,k,1) = dMy  + S_arr(i,j,k,2);
            E_rhs(i,j,k)   = dE   + S_arr(i,j,k,3);

            //------------------------------------------------------------------
            // ETA equation
            //------------------------------------------------------------------
            // Build grad_eta properly from Array4
            Real eta = eta_arr(i,j,k);
            auto grad   = Numeric::Gradient(eta_arr, i,j,k, 0, DX);
            Real lapeta = Numeric::Laplacian(eta_arr, i,j,k, 0, DX);

            Real adv = -( (M_arr(i,j,k,0)/rho_arr(i,j,k))*grad(0)
                         +(M_arr(i,j,k,1)/rho_arr(i,j,k))*grad(1) );

            Real Mob = 0.0; // as you had in your original
            Real diff= Mob * lapeta;

            eta_rhs(i,j,k) = adv + diff;  // + vaporization if needed
        });
    }
}
// CURVATURE CALCULATION
void Hydro2::ComputeHeightFunction (int lev)
{
    BL_PROFILE("Hydro2::ComputeHeightFunction");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();
    const Box& domain = geom.Domain();
    const auto problo = geom.ProbLo();

    const int ilo = domain.smallEnd(0);
    const int ihi = domain.bigEnd(0);
    const int jlo = domain.smallEnd(1);
    const int jhi = domain.bigEnd(1);

    // ------------------------------------------------------------
    // (1) Make a SINGLE-FAB scratch MF covering the entire domain
    // ------------------------------------------------------------
    BoxArray ba_single(domain);

    // Correct: generates a valid 1-element DM
    DistributionMapping dm_single(ba_single);

    MultiFab eta_single(ba_single, dm_single, 1, 0);
    eta_single.setVal(0.0);

    // Copy data from tiled MF into single FAB MF
    eta_single.ParallelCopy(*eta_mf[lev]);

    MultiFab h_single(ba_single, dm_single, 1, 0);

    // ------------------------------------------------------------
    // (2) Now we can safely access full-domain array
    // ------------------------------------------------------------
    auto eta_arr = eta_single.const_array(0);
    auto h_arr   = h_single.array(0);

    // ------------------------------------------------------------
    // (3) Compute heights globally, full domain
    // ------------------------------------------------------------
    ParallelFor(ihi - ilo + 1, [=] AMREX_GPU_DEVICE(int ii)
    {
        int i = ii + ilo;

        Real best = 1e20;
        int jbest = jlo;

        for (int j = jlo; j <= jhi; ++j)
        {
            Real d = amrex::Math::abs(eta_arr(i,j,0) - 0.5);
            if (d < best) { best = d; jbest = j; }
        }

        Real y = problo[1] + (jbest + 0.5)*dx[1];

        // Fill all j for later tile-safe curvature use
        for (int j = jlo; j <= jhi; ++j)
            h_arr(i,j,0) = y;
    });

    // ------------------------------------------------------------
    // (4) Scatter back to tiled MF
    // ------------------------------------------------------------
    h_eta_mf[lev]->ParallelCopy(h_single);
    h_eta_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeHFcurvature_MUSCL(int lev)
{
    BL_PROFILE("Hydro2::ComputeHFcurvature_MUSCL");

    const Geometry& geom = this->geom[lev];
    const Real* dx = geom.CellSize();

    for (MFIter mfi(*h_eta_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto h_arr   = h_eta_mf[lev]->const_array(mfi);
        auto kHF_arr = kappa_HF_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            int il = (i > tb.smallEnd(0)) ? i-1 : i;
            int ir = (i < tb.bigEnd(0))   ? i+1 : i;

            Real hC = h_arr(i,j,0);
            Real hL = h_arr(il,j,0);
            Real hR = h_arr(ir,j,0);

            Real dl = hC - hL;
            Real dr = hR - hC;

            Real slope = (dl*dr > 0.0 ? (2*dl*dr)/(dl+dr) : 0.0);
            Real hx  = slope / dx[0];
            Real hxx = (hR - 2*hC + hL) / (dx[0]*dx[0]);

            kHF_arr(i,j,0) = hxx / std::pow(1.0 + hx*hx, 1.5);
        });
    }

    kappa_HF_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeSmoothNormals(int lev)
{
    BL_PROFILE("Hydro2::ComputeSmoothNormals");

    const Geometry& geom = this->geom[lev];

    for (MFIter mfi(*nx_smoothed_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto eta_x  = eta_x_mf[lev]->const_array(mfi);
        auto eta_y  = eta_y_mf[lev]->const_array(mfi);
        auto gradmag= gradmag_mf[lev]->const_array(mfi);

        auto nx_s = nx_smoothed_mf[lev]->array(mfi);
        auto ny_s = ny_smoothed_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            Real gm = gradmag(i,j,k) + 1e-14;
            Real nx0 = eta_x(i,j,k)/gm;
            Real ny0 = eta_y(i,j,k)/gm;

            Real accx=0, accy=0, wsum=0;
            const Real w0=0.27901, w1=0.44198, w2=0.27901;

            for (int di=-1; di<=1; ++di)
            for (int dj=-1; dj<=1; ++dj)
            {
                int ii=i+di, jj=j+dj;

                if (!tb.contains(IntVect(AMREX_D_DECL(ii,jj,0))))
                    continue;

                Real w = ((di==0 && dj==0)?  w1
                       : (std::abs(di)+std::abs(dj)==1)? w0 : w2);

                Real gmN = gradmag(ii,jj,k)+1e-14;
                Real nxN = eta_x(ii,jj,k)/gmN;
                Real nyN = eta_y(ii,jj,k)/gmN;

                accx += w*nxN;
                accy += w*nyN;
                wsum += w;
            }

            Real nx=accx/(wsum+1e-14);
            Real ny=accy/(wsum+1e-14);

            Real m = std::sqrt(nx*nx + ny*ny) + 1e-14;

            nx_s(i,j,k) = nx/m;
            ny_s(i,j,k) = ny/m;
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

    const Real Cg = 0.1/epsilon;
    const Real small = 1e-14;

    for (MFIter mfi(*kappas_mf[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& tb = mfi.tilebox();

        auto gradmag  = gradmag_mf[lev]->const_array(mfi);
        auto nx_s     = nx_smoothed_mf[lev]->const_array(mfi);
        auto ny_s     = ny_smoothed_mf[lev]->const_array(mfi);
        auto kHF_arr  = kappa_HF_mf[lev]->const_array(mfi);
        auto h_arr    = h_eta_mf[lev]->const_array(mfi);

        auto kappas   = kappas_mf[lev]->array(mfi);

        ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i,int j,int k)
        {
            Real gm = gradmag(i,j,k);

            if (gm < Cg)
            {
                kappas(i,j,k,0) = 0.0;
                kappas(i,j,k,1) = 0.0;
                kappas(i,j,k,2) = 0.0;
                return;
            }

            //------------------------------------------------------
            // Sg
            //------------------------------------------------------
            Real Sg = gm / (gm + Cg);

            //------------------------------------------------------
            // Compute SN curvature: kSN = d(nx)/dx + d(ny)/dy
            //------------------------------------------------------
            int il = (i > tb.smallEnd(0)) ? i-1 : i;
            int ir = (i < tb.bigEnd(0))   ? i+1 : i;
            int jl = (j > tb.smallEnd(1)) ? j-1 : j;
            int jr = (j < tb.bigEnd(1))   ? j+1 : j;

            Real nx_x = 0.5 * (nx_s(ir,j,k) - nx_s(il,j,k)) / dx[0];
            Real ny_y = 0.5 * (ny_s(i,jr,k) - ny_s(i,jl,k)) / dx[1];

            Real kSN = nx_x + ny_y;

            //------------------------------------------------------
            // HF curvature at same (i,j,k)
            //------------------------------------------------------
            Real kHF = kHF_arr(i,j,k);

            //------------------------------------------------------
            // Sm monotonicity test — must use j, not 0
            //------------------------------------------------------
            Real hC = h_arr(i,j,0);
            Real Sm = 1.0;

            if (i > tb.smallEnd(0))
            {
                if (std::abs(h_arr(i-1,j,0) - hC) > 3*dx[1])
                    Sm = 0.0;
            }

            if (i < tb.bigEnd(0))
            {
                if (std::abs(h_arr(i+1,j,0) - hC) > 3*dx[1])
                    Sm = 0.0;
            }

            //------------------------------------------------------
            // Ss smoothness: depends on |kHF - kSN|
            //------------------------------------------------------
            Real Cs = amrex::max(std::abs(kHF), 1e-12);
            Real dk = std::abs(kHF - kSN);

            Real Ss = std::exp(-dk / (Cs + small));

            //------------------------------------------------------
            // Final hybrid curvature weight
            //------------------------------------------------------
            Real w = Sg * Sm * Ss;
            w = amrex::max(0.0, amrex::min(1.0, w));

            Real kH = w * kHF + (1.0 - w) * kSN;

            //------------------------------------------------------
            // Store results
            //------------------------------------------------------
            kappas(i,j,k,0) = kH;
            kappas(i,j,k,1) = kHF;
            kappas(i,j,k,2) = kSN;
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
    // IMPORTANT: curvature pipeline depends on the following sequence:
    //
    //  1. Compute ∇η (eta_x, eta_y) and |∇η| into gradmag_mf
    //  2. Build height function h(i)
    //  3. Compute HF curvature using MUSCL limited slopes
    //  4. Compute smoothed-normal curvature (SN curvature)
    //  5. Blend HF and SN to get hybrid curvature
    //
    // All MultiFabs involved must be already allocated via RegisterNewFab().
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
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::Advance");

    const Geometry& geom = this->geom[lev];
    const Real* DX = geom.CellSize();

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
    energy_per_mas_old_mf[lev]->ParallelCopy(*energy_per_mas_mf[lev]);

    //==============================================================
    // 2. Build *fresh*, fully allocated MultiFabs for TimeIntegrator
    //    No aliasing. No component mismatch. No ghost mismatch.
    //==============================================================
    amrex::Vector<amrex::MultiFab> solution_new;
    amrex::Vector<amrex::MultiFab> solution_old;

    // eta
    solution_new.emplace_back(eta_mf[lev]->boxArray(),
                              eta_mf[lev]->DistributionMap(),
                              eta_mf[lev]->nComp(),
                              eta_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*eta_mf[lev]);

    solution_old.emplace_back(eta_old_mf[lev]->boxArray(),
                              eta_old_mf[lev]->DistributionMap(),
                              eta_old_mf[lev]->nComp(),
                              eta_old_mf[lev]->nGrow());
    solution_old.back().ParallelCopy(*eta_old_mf[lev]);


    // density
    solution_new.emplace_back(density_mf[lev]->boxArray(),
                              density_mf[lev]->DistributionMap(),
                              density_mf[lev]->nComp(),
                              density_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*density_mf[lev]);

    solution_old.emplace_back(density_old_mf[lev]->boxArray(),
                              density_old_mf[lev]->DistributionMap(),
                              density_old_mf[lev]->nComp(),
                              density_old_mf[lev]->nGrow());
    solution_old.back().ParallelCopy(*density_old_mf[lev]);


    // momentum (2 components)
    solution_new.emplace_back(momentum_mf[lev]->boxArray(),
                              momentum_mf[lev]->DistributionMap(),
                              momentum_mf[lev]->nComp(),
                              momentum_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*momentum_mf[lev]);

    solution_old.emplace_back(momentum_old_mf[lev]->boxArray(),
                              momentum_old_mf[lev]->DistributionMap(),
                              momentum_old_mf[lev]->nComp(),
                              momentum_old_mf[lev]->nGrow());
    solution_old.back().ParallelCopy(*momentum_old_mf[lev]);


    // energy per volume
    solution_new.emplace_back(energy_per_vol_mf[lev]->boxArray(),
                              energy_per_vol_mf[lev]->DistributionMap(),
                              energy_per_vol_mf[lev]->nComp(),
                              energy_per_vol_mf[lev]->nGrow());
    solution_new.back().ParallelCopy(*energy_per_vol_mf[lev]);

    solution_old.emplace_back(energy_per_vol_old_mf[lev]->boxArray(),
                              energy_per_vol_old_mf[lev]->DistributionMap(),
                              energy_per_vol_old_mf[lev]->nComp(),
                              energy_per_vol_old_mf[lev]->nGrow());
    solution_old.back().ParallelCopy(*energy_per_vol_old_mf[lev]);


    //==============================================================
    // 3. Fill ghosts for *both* solution vectors BEFORE integration
    //==============================================================
    for (auto& mf : solution_new)
        mf.FillBoundary(geom.periodicity());

    for (auto& mf : solution_old)
        mf.FillBoundary(geom.periodicity());

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
    // 5. Fill ghosts after each RK stage (REQUIRED)
    //==============================================================
    timeintegrator.set_post_stage_action(
        [&](amrex::Vector<amrex::MultiFab>& stage, Real t)
    {
        for (int n = 0; n < stage.size(); ++n)
        {
            stage[n].FillBoundary(geom.periodicity());
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

    // (KEEP your final diagnostic block as is—it was correct)
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

}


//#endif
