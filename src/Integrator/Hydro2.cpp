// Base
#include "Hydro2.H"
// Parsing and Input Handeling
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "Numeric/Stencil.H"
#include "IC/Constant.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
// Solvers
//#include "Solver/Local/Riemann/Roe.H"
#include "Solver/Local/Riemann/HLLC.H"
//#include "Solver/Local/Riemann/HLLC_WENO5.H"
//#include "Solver/Local/Riemann/HLLE.H"
//#include "Solver/Local/Riemann/HLLE_WENO5.H"
//#include "Solver/Local/Riemann/HLLCE.H"
//#include "Solver/Local/Riemann/HLLCE_WENO5.H"
//#include "Solver/Local/Riemann/PartiallyParabolic.H"
// Limiters
//#include "Solver/Local/Limiter/Minmod.H"
//#include "Solver/Local/Limiter/VanLeer.H"

#if AMREX_SPACEDIM == 2

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
        pp_query_default("small", value.small, 1E-8);       // small regularization value
        pp_query_default("cutoff", value.cutoff, 1E-100);   // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0);  // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);            // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix");  // Roe solver entropy fix
        pp_query_default("scheme", value.scheme, 0);        // 0: Forward Euler | 1: RK4
        pp_query_default("Spec_Vol", value.Spec_Vol, 1);    // 0: Solve Energy via specific mass | 1: Solve Energy via specific volume

        // ADAPTIVE TIMESTEP
        // Swtiched pointer names to fix weird time stepping issue
        /*
        pp_query_default("adaptive_timestep", value.adaptive_timestep, false); 
        pp_query_default("dt_min", value.dt_min, 1E-9);
        pp_query_default("dt_max", value.dt_max, 1E-3);
        pp_query_default("dt_growth", value.dt_growth, 1.2);
        */
        pp_forbid("adaptive_timestep", "--> dynamictimestep.on"); 
        pp_forbid("dt_min", "--> dynamictimestep.min"); 
        pp_forbid("dt_max", "--> dynamictimestep.max"); 
        pp_forbid("dt_growth", "--> REMOVED");
        pp_forbid("dt_nprev", "--> dynamictimestep.nprevious");
        


        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, true); // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("apply_buoyancy", value.apply_buoyancy, false);              // Apply buoyancy when solving, default: false --> "No Buoyancy"
        pp_query_default("apply_weight", value.apply_weight, false);                  // Apply weight when solving, default: false --> "No Weight"
        pp_query_default("static_eta", value.static_eta, false);                      // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // NEW SOLVER FORBIDS
        pp_forbid("gamma", "--> gamma0 and gamma1");
        pp_forbid("mu", "--> mu0 and mu1");

        // FLUID 0
        pp_query_required("gamma0", value.gamma0);      // gamma for gamma law
        pp_query_default("p0_0", value.p0_0, 0.0);      // p0 for Tammann EOS
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient
        // pp_query_required("R0", value.R0);              // Specific Gas Constant
        // pp_query_required("MW0", value.MW0);            // Molecular Weight
        
        // FLUID 1
        pp_query_required("gamma1", value.gamma1);      // gamma for gamma law
        pp_query_default("p0_1", value.p0_1, 0.0);      // p0 for Tammann EOS
        pp_query_required("mu1", value.mu1);            // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0);    // bulk viscosity coefficient
        // pp_query_required("R1", value.R1);              // Specific Gas Constant
        // pp_query_required("MW1", value.MW1);            // Molecular Weight


        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 70.0);   // surface tension condition
        pp_query_required("epsilon", value.epsilon);    // diffuse interface thickness

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 2); // Method to solve for curvature
        
        // Boundry Conditions
        pp_forbid("rho.bc","--> density.bc");
        pp_forbid("p.bc","--> pressure.bc");
        pp_forbid("v.bc","--> velocity.bc");
        pp_forbid("pressure.bc", "--> energy.bc");
        pp_forbid("velocity.bc", "--> momentum.bc");
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
        value.RegisterNewFab(value.eta_mf,          value.energy_bc, 1, nghost, "eta", true);
        value.RegisterNewFab(value.eta_old_mf,      value.energy_bc, 1, nghost, "eta_old", true);
        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing, 1, nghost, "etadot", true);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing,  4, nghost, "hess_eta", true, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing,  2, nghost, "n_hat", true, { "x", "y" });

        // FLUID 0
        value.RegisterNewFab(value.density0_mf,     value.density_bc,   1, nghost, "density0",     false );
        value.RegisterNewFab(value.density0_old_mf, value.density_bc,   1, nghost, "density0_old", false);

        value.RegisterNewFab(value.energy0_mf,      value.energy_bc,    1, nghost, "energy0", false);
        value.RegisterNewFab(value.energy0_old_mf,  value.energy_bc,    1, nghost, "energy0_old" , false);

        value.RegisterNewFab(value.momentum0_mf,    value.momentum_bc,  2, nghost, "momentum0", false, { "x", "y" });
        value.RegisterNewFab(value.momentum0_old_mf,value.momentum_bc,  2, nghost, "momentum0_old", false);
 
        value.RegisterNewFab(value.T0_mf,           value.temperature_bc, 1, nghost, "T0", false);
        value.RegisterNewFab(value.cp0_mf,          &value.bc_nothing, 1, nghost, "cp0", false);
        value.RegisterNewFab(value.cv0_mf,          &value.bc_nothing, 1, nghost, "cv0", false);
        value.RegisterNewFab(value.k0_thermal_mf,   &value.bc_nothing, 1, nghost, "k0_thermal", false);
        value.RegisterNewFab(value.h0_thermal_mf,   &value.bc_nothing, 1, nghost, "h0_thermal", false);

        value.RegisterNewFab(value.pressure0_mf,    &value.bc_nothing,  1, nghost, "pressure0", false);
        value.RegisterNewFab(value.velocity0_mf,    &value.bc_nothing,  2, nghost, "velocity0", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity0_mf,   &value.bc_nothing,  1, nghost, "vorticity0", false);

        // FLUID 1
        value.RegisterNewFab(value.density1_mf,     value.density_bc,   1, nghost, "density1", false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc,   1, nghost, "density1_old", false);

        value.RegisterNewFab(value.energy1_mf,      value.energy_bc,    1, nghost, "energy1", false);
        value.RegisterNewFab(value.energy1_old_mf,  value.energy_bc,    1, nghost, "energy1_old", false);

        value.RegisterNewFab(value.momentum1_mf,    value.momentum_bc,  2, nghost, "momentum1", false, { "x", "y" });
        value.RegisterNewFab(value.momentum1_old_mf,value.momentum_bc,  2, nghost, "momentum1_old", false);

        value.RegisterNewFab(value.T1_mf,           value.temperature_bc, 1, nghost, "T1", false);
        value.RegisterNewFab(value.cp1_mf,          &value.bc_nothing, 1, nghost, "cp1", false);
        value.RegisterNewFab(value.cv1_mf,          &value.bc_nothing, 1, nghost, "cv1", false);
        value.RegisterNewFab(value.k1_thermal_mf,   &value.bc_nothing, 1, nghost, "k1_thermal", false);
        value.RegisterNewFab(value.h1_thermal_mf,   &value.bc_nothing, 1, nghost, "h1_thermal", false);

        value.RegisterNewFab(value.pressure1_mf,    &value.bc_nothing,  1, nghost, "pressure1", false);
        value.RegisterNewFab(value.velocity1_mf,    &value.bc_nothing,  2, nghost, "velocity1", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity1_mf,   &value.bc_nothing,  1, nghost, "vorticity1", false);

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,     &value.bc_nothing,  1, nghost, "pressure", true);
        value.RegisterNewFab(value.velocity_mf,     &value.bc_nothing,  2, nghost, "velocity", true, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf,    &value.bc_nothing,  1, nghost, "vorticity", true);
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost, "energy_per_vol", true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost, "energy_per_mass", true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost, "energy_vol_old", false);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost, "energy_mas_old", false);
        value.RegisterNewFab(value.momentum_mf,     value.momentum_bc,  2, nghost, "momentum", true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc,  2, nghost, "momentum_old", false, { "x", "y" });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing,  1, 0, "m0", true);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing,  2, 0, "u0", true, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing,  2, 0, "q0", true, { "x", "y" });
        value.RegisterNewFab(value.Source_mf,       &value.bc_nothing,  4, nghost, "Source", true);
        value.RegisterNewFab(value.Fsv_mf,          &value.bc_nothing,  2, nghost, "Fsv", true, { "x", "y" });  // Surface Tension
        value.RegisterNewFab(value.Fb_mf,           &value.bc_nothing,  2, nghost, "Fb", true, { "x", "y" });   // Buoyancy
        value.RegisterNewFab(value.Fw_mf,           &value.bc_nothing,  2, nghost, "Fw", true, { "x", "y" });   // Weight
        value.RegisterNewFab(value.T_mf,            &value.bc_nothing,  1, nghost, "T", true);                  // Temperature
        value.RegisterNewFab(value.cp_mf,           &value.bc_nothing,  1, nghost, "cp", false);                // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,           &value.bc_nothing,  1, nghost, "cv", false);                // Constant Volume Specific Heat
        value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false);         // Thermal Conductivity
        value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc, 1, nghost, "gamma", true);                 // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf,           value.energy_bc, 1, nghost, "p0", true);                    // Tamman Pressure
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true);                  // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   &value.bc_nothing,  1, nghost, "UE_per_vol", true);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   &value.bc_nothing,  1, nghost, "UE_per_mass", true);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   &value.bc_nothing,  1, nghost, "KE_per_vol", true);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   &value.bc_nothing,  1, nghost, "KE_per_mass", true);        // Kinetic Energy (per unit mass)

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, nghost, "grad_eta", true, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, { "Avg", "1", "2" }); // To Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", true, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,     &value.bc_nothing,  1, nghost, "rho_flux", true);                    // Density Flux
        value.RegisterNewFab(value.M_flux_mf,       &value.bc_nothing,  2, nghost, "M_flux", true, { "x", "y" });        // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,       &value.bc_nothing,  1, nghost, "E_flux", true);                      // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,      &value.bc_nothing,  2, nghost, "div_tau", true, { "x", "y" });            // Energy Flux
        value.RegisterNewFab(value.hess_u_mf,       &value.bc_nothing,  8, nghost, "hess_u", true, {
                                                                                                     "000",
                                                                                                     "001",
                                                                                                     "010",
                                                                                                     "011",
                                                                                                     "100",
                                                                                                     "101",
                                                                                                     "110",
                                                                                                     "111",
                                                                                                    }); // hess_u Flux
    }

    // NEW SOLVER FORBIDS
    pp_forbid("velocity.ic.type", "--> velocity0.ic.type or velocity1.ic.type");
    pp_forbid("pressure.ic", "--> pressure0.ic or pressure1.ic");
    pp_forbid("density.ic.type", "--> density0.ic.type or density1.ic.type");


    // INITIAL CONDITIONS
    // Eta
    pp.select_default<IC::Constant,IC::Laminate,IC::Expression,IC::BMP,IC::PNG>("eta.ic",value.eta_ic,value.geom);
    // Fluid 0
    pp.select_default<IC::Constant,IC::Expression>("velocity0.ic",      value.velocity0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure0.ic",      value.pressure0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density0.ic",       value.density0_ic,  value.geom);
    pp.select_default<IC::Constant, IC::Expression>("temperature0.ic",  value.temperature0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cp0.ic",           value.cp0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cv0.ic",           value.cv0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("k0_thermal.ic",    value.k0_thermal_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h0_thermal_ic, value.geom);


    // Fluid 1
    pp.select_default<IC::Constant,IC::Expression>("velocity1.ic",      value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure1.ic",      value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density1.ic",       value.density1_ic,  value.geom);
    pp.select_default<IC::Constant, IC::Expression>("temperature1.ic",  value.temperature1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cp1.ic",           value.cp1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cv1.ic",           value.cv1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("k1_thermal.ic",    value.k1_thermal_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h1_thermal_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("gamma1.ic",        value.gamma1_ic, value.geom);


    // DIFFUSE BOUNDARY SOURCES
    // diffuse boundary prescribed mass flux 
    pp.select_default<IC::Constant,IC::Expression>("m0.ic",value.ic_m0,value.geom);
    // diffuse boundary prescribed velocity
    pp.select_default<IC::Constant,IC::Expression>("u0.ic",value.ic_u0,value.geom);
    // diffuse boundary prescribed heat flux 
    pp.select_default<IC::Constant,IC::Expression>("q.ic",value.ic_q,value.geom);
    

    // SOLVERS
    // Riemann solver
    pp_query_default("Riemann_Solver", value.Riemann_Solver, 0); // Type of solver
    if (value.Riemann_Solver == 0)
    {
        // pp.select_default<Solver::Local::Riemann::Roe>("solver", value.roesolver);
    }
    else if (value.Riemann_Solver == 1)
    {
        pp.select_default<Solver::Local::Riemann::HLLC>("solver", value.hllcsolver);
    }
    else if (value.Riemann_Solver == 2)
    {
        // pp.select_default<Solver::Local::Riemann::HLLE>("solver", value.hllesolver);
    }
    else if (value.Riemann_Solver == 3)
    {
        // pp.select_default<Solver::Local::Riemann::HLLCE>("solver", value.hllcesolver);
    }
    else if (value.Riemann_Solver == 35)
    {
        // pp.select_default<Solver::Local::Riemann::HLLC_WENO5>("solver", value.hllc_weno5solver);
    }
    else if (value.Riemann_Solver == 36)
    {
        // pp.select_default<Solver::Local::Riemann::PartiallyParabolic>("solver", value.partiallyparabolicsolver);
    }
    else if (value.Riemann_Solver == 99)
    {
        // DEBUG, Just a place holder
    }
    else
    {
        Util::ParallelMessage(INFO, "-------------------------------");
        Util::ParallelMessage(INFO, "Invalid solver method: ", value.Riemann_Solver);
        Util::ParallelMessage(INFO, "Acceptable Methods: ");
        Util::ParallelMessage(INFO, "Roe        : 0");
        Util::ParallelMessage(INFO, "HLLC       : 1");
        Util::ParallelMessage(INFO, "HLLE       : 2");
        Util::ParallelMessage(INFO, "HLLCE      : 3");
        Util::ParallelMessage(INFO, "Under Testing:");
        Util::ParallelMessage(INFO, "HLLC_WENO5 : 35");
        Util::ParallelMessage(INFO, "PPM        : 36");
        Util::ParallelMessage(INFO, "HLLE DEBUG : 99");
        Util::Exception(INFO);
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
        Util::ParallelMessage(INFO, "Invalid Limiter: ", value.Riemann_Solver);
        Util::ParallelMessage(INFO, "Acceptable Methods: ");
        Util::ParallelMessage(INFO, "None       : 0");
        Util::ParallelMessage(INFO, "MinMod     : 1");
        Util::ParallelMessage(INFO, "Van Leer   : 2");
        Util::Exception(INFO);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");

    // Initialize individual fluid variables
    // DIFFUSIVE BOUNDRY
    eta_ic          ->Initialize(lev, eta_mf, 0.0);
    eta_ic          ->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]  ->setVal(0.0);

    // FLUID 0
    velocity0_ic    ->Initialize(lev, velocity0_mf, 0.0);
    pressure0_ic    ->Initialize(lev, pressure0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_mf, 0.0);
    density0_ic     ->Initialize(lev, density0_old_mf, 0.0);
    temperature0_ic ->Initialize(lev, T0_mf, 0.0);
    cp0_ic          ->Initialize(lev, cp0_mf, 0.0);
    cv0_ic          ->Initialize(lev, cv0_mf, 0.0);
    k0_thermal_ic   ->Initialize(lev, k0_thermal_mf, 0.0);
    h0_thermal_ic   ->Initialize(lev, h0_thermal_mf, 0.0);

    // FLUID 1
    velocity1_ic    ->Initialize(lev, velocity1_mf, 0.0);
    pressure1_ic    ->Initialize(lev, pressure1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_old_mf, 0.0);
    temperature1_ic ->Initialize(lev, T1_mf, 0.0);
    cp1_ic          ->Initialize(lev, cp1_mf, 0.0);
    cv1_ic          ->Initialize(lev, cv1_mf, 0.0);
    k1_thermal_ic   ->Initialize(lev, k1_thermal_mf, 0.0);
    h1_thermal_ic   ->Initialize(lev, h1_thermal_mf, 0.0);

    // SOURCE
    ic_m0           ->Initialize(lev, m0_mf, 0.0);
    ic_u0           ->Initialize(lev, u0_mf, 0.0);
    ic_q            ->Initialize(lev, q_mf, 0.0);
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    Fb_mf[lev]      ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    Fw_mf[lev]      ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    kappas_mf[lev]  ->setVal(0.0);
    grad_mag_grad_eta_mf[lev]->setVal(0.0);

    a_mf[lev]       ->setVal(0.0);
    Ma_mf[lev]      ->setVal(0.0);
    UE_per_vol_mf[lev]->setVal(0.0);
    UE_per_mas_mf[lev]->setVal(0.0);
    KE_per_vol_mf[lev]->setVal(0.0);
    KE_per_mas_mf[lev]->setVal(0.0);


    // Calculate mixed variables based on individual fluid variables
    Mix(lev);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    // Function is for the diffusive mixing terms. I.E: rho = eta*rho0 + (1-eta)*rho1
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        // FLUID 0
        Set::Patch<const Set::Scalar>   v0          = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p0          = pressure0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho0        = density0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho0_old    = density0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M0          = momentum0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M0_old      = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0          = energy0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0_old      = energy0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         T0          = T0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cp0         = cp0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cv0         = cv0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         k0_thermal  = k0_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         h0_thermal  = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1          = energy1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_old      = energy1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         T1          = T1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cp1         = cp1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cv1         = cv1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         k1_thermal  = k1_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         h1_thermal  = h1_thermal_mf.Patch(lev, mfi);

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
        Set::Patch<Set::Scalar>         k_thermal   = k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         h_thermal   = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         gammaf      = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         p0_eff      = p0_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar>         a           = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Ma          = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_vol      = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_mas      = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_vol      = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_mas      = KE_per_mas_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Calculate State Variables 
            rho(i, j, k) = eta(i, j, k) * rho0(i, j, k) + (1.0 - eta(i, j, k)) * rho1(i, j, k);
            rho_old(i, j, k) = rho(i, j, k);  

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
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0); 
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            UE_vol(i, j, k) = (p_eff + pref)*A + B;
            UE_mas(i, j, k) = (UE_vol(i, j, k)) / (rho(i, j, k) + small);
            
            // Kinetic Energy
            //  TODO: Get rid of thermally perfect assumption. Involve temperature
            E_vol(i, j, k) = KE_vol(i, j, k) + UE_vol(i, j, k);
            E_vol_old(i, j, k) = E_vol(i, j, k);
            E_mas(i, j, k) = KE_mas(i, j, k) + UE_mas(i, j, k);
            E_mas_old(i, j, k) = E_mas(i, j, k);

            // Initialize extra fields - (not directly used to solve)
            // Velocity
            v(i, j, k, 0) = v0(i, j, k, 0) * eta(i, j, k) + v1(i, j, k, 0) * (1.0 - eta(i, j, k));
            v(i, j, k, 1) = v0(i, j, k, 1) * eta(i, j, k) + v1(i, j, k, 1) * (1.0 - eta(i, j, k));

            // Specific Heat Ratio
            Set::Scalar gamma_eff = 1.0 + (1.0 / A);
            gammaf(i, j, k) = gamma_eff;

            // Pressure
            press(i, j, k) = (UE_vol(i, j, k) - B) / A - pref;
            p0_eff(i, j, k) = (B / A) / gamma_eff;

            // Temperature
            T(i, j, k) = T0(i, j, k) * eta(i, j, k) + T1(i, j, k) * (1.0 - eta(i, j, k));

            // Constant Pressure Specific Heat
            cp(i, j, k) = eta(i, j, k) * cp0(i, j, k) + (1.0 - eta(i, j, k)) * cp1(i, j, k);

            // Constant Volume Specific Heat
            cv(i, j, k) = eta(i, j, k) * cv0(i, j, k) + (1.0 - eta(i, j, k)) * cv1(i, j, k);

            // Thermal Conductivity
            k_thermal(i, j, k) = eta(i, j, k) * k0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * k1_thermal(i, j, k);

            // Thermal Convectivity
            h_thermal(i, j, k) = eta(i, j, k) * h0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * h1_thermal(i, j, k);

            // Speed of Sound
            a(i, j, k) = (std::sqrt(gamma0 * p0(i, j, k) / (rho0(i, j, k) + small))) * (eta(i, j, k)) + (std::sqrt(gamma1 * p1(i, j, k) / (rho1(i, j, k) + small))) * (1.0 - eta(i, j, k));

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / a(i, j, k);
            Ma(i, j, k, 1) = v(i, j, k, 1) / a(i, j, k);

        });
    }
    c_max = 0.0;
    vx_max = 0.0;
    vy_max = 0.0;
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
        Integrator::DynamicTimestep_Update();
    return;

    const Set::Scalar *DX = geom[lev].CellSize();

    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);

    Set::Scalar new_timestep = cfl / ((c_max + vx_max) / (DX[0]) + (c_max + vy_max) / (DX[1]) + small);

    // DEBUGGING VERBOSE
    // Ensure dt_min is valid
    if (std::isnan(new_timestep) || std::isinf(new_timestep) || new_timestep <= 0.0)
    {
        amrex::Print() << "WARNING: Invalid new_timestep calculated: " << new_timestep
                       << " at time " << time << " on level " << lev << "\n";
        amrex::Print() << "  c_max = " << c_max << ", vx_max = " << vx_max
                       << ", vy_max = " << vy_max << "\n";
        new_timestep = dynamictimestep.min; // Use the minimum timestep from parameters
    }
    Util::Message(INFO, "  CFL Timestep = ", new_timestep);
    //amrex::Print() << "  CFL Timestep = " << new_timestep << "\n";

    Util::Assert(INFO, TEST(AMREX_SPACEDIM == 2));

    SetTimestep(new_timestep);
    
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// RHS /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RHS(int lev, Set::Scalar time, amrex::MultiFab &rho_rhs_mf, amrex::MultiFab &M_rhs_mf, amrex::MultiFab &E_rhs_mf, amrex::MultiFab &eta_rhs_mf, const amrex::MultiFab &rho_mf_in, const amrex::MultiFab &M_mf_in, const amrex::MultiFab &E_mf_in, const amrex::MultiFab &eta_mf_in) //, const amrex::MultiFab &velocity_mf_in, const amrex::MultiFab &pressure_mf_in, const amrex::MultiFab &T_mf_in)
{
    // Deleted to condence code - have copy saved locally we can reimplemant after validation
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    // Swaping pointers
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    std::swap(eta_old_mf, eta_mf);

    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();


    // First loop to show plotting fields and calculate intermediate items
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        //const amrex::Box &bx = mfi.growntilebox(); // Will return NaNs for Eta
        const amrex::Box &bx = mfi.validbox(); 


        // Eta
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta_new = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_eta_ = hess_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf.Patch(lev, mfi);
        


        // Mixture
        Set::Patch<const Set::Scalar> rho = density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_old_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> a = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ma = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_vol = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_mas = KE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_vol = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_mas = UE_per_mas_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);

        

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // gamma
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            Set::Scalar gamma_eff = 1.0 + (1.0 / A);
            gammaf(i, j, k) = gamma_eff;

            // etadot
            etadot(i, j, k) = (eta_new(i, j, k) - eta(i, j, k)) / dt;

            // /////////////////////////////////////////////////////
            // INTEGRATOR METHODS:
            // 0: Forward Euler
            // 1: Runge-Kutta 4th Order (RK4)

            if (scheme == 0)
            {
            }
            else if (scheme == 1)
            {
            }
            else
            {
                Util::ParallelMessage(INFO, "ERROR in Hydro2::Advance() : Integrator Methods");
                Util::ParallelMessage(INFO, "Method ", scheme, " is unknown.");
                Util::Exception(INFO);
            }

            // Velocity = M ./ (DX*DY*rho)
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k) + small);
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k) + small);

            // Kinetic Energy
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            KE_mas(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            // Potential Energy
            UE_vol(i, j, k) = E_vol(i, j, k) - KE_vol(i, j, k);
            UE_mas(i, j, k) = E_mas(i, j, k) - KE_mas(i, j, k);
            
            // Pressure
            press(i, j, k) = (UE_vol(i, j, k) - B) / A;
            p0_eff(i,j,k) = (B / A) / gamma_eff;

            // Speed of sound:
            a(i, j, k) = std::sqrt(gammaf(i,j,k) * (press(i, j, k) + p0_eff(i,j,k)) / (rho(i, j, k) + small));

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / (a(i, j, k) + small);
            Ma(i, j, k, 1) = v(i, j, k, 1) / (a(i, j, k) + small);




            // DEBUGGING
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);

            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

            // Debugging, would like to delete condition
            if (grad_eta_mag < 1e-4)
            {
                n_hat_(i, j, k, 0) = 0.0;
                n_hat_(i, j, k, 1) = 0.0;

                hess_eta_(i, j, k, 0) = 0.0;
                hess_eta_(i, j, k, 1) = 0.0;
                hess_eta_(i, j, k, 2) = 0.0;
                hess_eta_(i, j, k, 3) = 0.0;
            }
            else
            {
                n_hat_(i, j, k, 0) = n_hat(0);
                n_hat_(i, j, k, 1) = n_hat(1);

                hess_eta_(i, j, k, 0) = hess_eta(0, 0);
                hess_eta_(i, j, k, 1) = hess_eta(0, 1);
                hess_eta_(i, j, k, 2) = hess_eta(1, 0);
                hess_eta_(i, j, k, 3) = hess_eta(1, 1);
            }

            Set::Vector grad_mag_grad_eta = Set::Vector(1 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                                                        1 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));    
            
            grad_mag_grad_eta_(i, j, k, 0) = grad_mag_grad_eta(0);
            grad_mag_grad_eta_(i, j, k, 1) = grad_mag_grad_eta(1);

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

                if (std::abs(n_hat(0)) > std::abs(n_hat(1)))
                {
                    t1 = Set::Vector(-n_hat(1), n_hat(0)) / std::sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                }
                else
                {
                    t1 = Set::Vector(n_hat(1), -n_hat(0)) / std::sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                }

                // t1 = Set::Vector(-n_hat(1), n_hat(0)) / std::sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
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

            // DEBUG Tool
            if ((Ma(i, j, k, 0) != Ma(i, j, k, 0))
                or (Ma(i, j, k, 1) != Ma(i, j, k, 1))
                or (press(i, j, k) != press(i, j, k))
                or (v(i, j, k) != v(i, j, k))
                or (KE_vol(i, j, k) != KE_vol(i, j, k)) 
                or (UE_vol(i, j, k) != UE_vol(i, j, k)) 
                or (press(i, j, k) > 1E1000) )
            {
                Util::ParallelMessage(INFO, "v=", v(i, j, k, 0), ", ", v(i, j, k, 1));
                Util::ParallelMessage(INFO, "press=", press(i, j, k));
                Util::ParallelMessage(INFO, "p_eff=", p0_eff(i,j,k));
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k, 0), ", ", M(i, j, k, 1));
                Util::ParallelMessage(INFO, "E=", E_vol(i, j, k));
                Util::ParallelMessage(INFO, "KE=", KE_vol(i, j, k));
                Util::ParallelMessage(INFO, "UE=", UE_vol(i, j, k));
                Util::ParallelMessage(INFO, "Ma=", Ma(i, j, k, 0), ", ", Ma(i, j, k, 1));
                Util::ParallelMessage(INFO, "a=", a(i, j, k));
                Util::ParallelMessage(INFO, "gamma=", gammaf(i, j, k));
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                Util::ParallelMessage(INFO, "etadot=", etadot(i, j, k));
                Util::Exception(INFO);
            }

        });
    }

    // ============= SYNCHRONIZATION =============
    // Eta
    (*hess_eta_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*kappas_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*grad_mag_grad_eta_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*n_hat_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*grad_eta_mf[lev]).FillBoundary(geom[lev].periodicity());
    // Mixutre
    (*a_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*Ma_mf[lev]).FillBoundary(geom[lev].periodicity());
    /*
    (*KE_vol_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*KE_mas_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*UE_vol_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*UE_mas_mf[lev]).FillBoundary(geom[lev].periodicity());
    */


    (*velocity_mf[lev]).FillBoundary(geom[lev].periodicity());
    (*pressure_mf[lev]).FillBoundary(geom[lev].periodicity());

    // Not adding to yet so we can leave these our
    /*
    Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
    Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
    Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
    Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
    Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
    Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
    */


    // Ensure all MPI ranks complete
    amrex::ParallelDescriptor::Barrier();


    // Main time integration loop
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        // MIXTURE
        Set::Patch<const Set::Scalar> rho = density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_new = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_vol_new = energy_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_mas_new = energy_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_new = momentum_mf.Patch(lev, mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> T = T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fsv = Fsv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fb = Fb_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fw = Fw_mf.Patch(lev, mfi);

        // DEBUGGING PLOTS
        Set::Patch<const Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> hess_eta_ = hess_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = hess_u_mf.Patch(lev, mfi);

        Set::Scalar *dt_max_handle = &dt_max;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
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
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k) + small);

            Set::Vector q0 = Set::Vector(q(i, j, k, 0), q(i, j, k, 1));

            /// Calculate Source Terms
            // Shear:
            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0.dot(grad_eta);

            Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
            Set::Matrix3 hess_u = Set::Matrix3::Zero();
            for (int p = 0; p < 2; p++)
                for (int q = 0; q < 2; q++)
                    for (int r = 0; r < 2; r++)
                    {
                        hess_u(r, p, q) = (hess_M(r, p, q) - gradu(r, q) * gradrho(p) - gradu(r, p) * gradrho(q) - u(r) * hess_rho(p, q))
                                          / (rho(i, j, k) + small);
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

            // Stress Tensors:
            Set::Vector Ldot0 = Set::Vector::Zero();
            Set::Vector div_tau = Set::Vector::Zero();
            Set::Scalar div_u = gradu(0, 0) + gradu(1, 1); // Divergence of velocity

            // Calculate effective viscosities
            Set::Scalar mu_eff = eta(i, j, k) * mu0 + (1.0 - eta(i, j, k)) * mu1;       // Effective dynamic viscosity
            Set::Scalar mu_b_eff = eta(i, j, k) * mu0_b + (1.0 - eta(i, j, k)) * mu1_b; // Effective bulk viscosity

            for (int p = 0; p < 2; p++)             // Dimension Component
                for (int q = 0; q < 2; q++)         // Dimension Component
                    for (int r = 0; r < 2; r++)     // X
                        for (int s = 0; s < 2; s++) // Y
                        {
                            Set::Scalar Mpqrs = 0.0;

                            // Newtonian fluid terms (shear viscosity)
                            if (p == r && q == s)
                                Mpqrs += 0.5 * mu_eff;
                            if (p == s && q == r)
                                Mpqrs += 0.5 * mu_eff;

                            // Bulk viscosity
                            if (p == q && r == s)
                                Mpqrs += (1.0 / 3.0) * mu_b_eff;

                            Ldot0(p) += 0.5 * Mpqrs * (u(r) - u0(r)) * hess_eta(q, s);
                            div_tau(p) += 2.0 * Mpqrs * hess_u(r, s, q);
                        }
            // NEW Formulation of div_tau
            // Calculate gradient of divergence
            Set::Vector grad_div_u = Set::Vector::Zero();
            grad_div_u[0] = hess_u(0, 0, 0) + hess_u(1, 0, 1);
            grad_div_u[1] = hess_u(0, 1, 0) + hess_u(1, 1, 1);
            // Calculate div_tau
            //div_tau(0) = mu_eff * (hess_u(0, 0, 0) + hess_u(0, 1, 1) + (1.0 / 3.0) * grad_div_u[0]);
            //div_tau(1) = mu_eff * (hess_u(1, 0, 0) + hess_u(1, 1, 1) + (1.0 / 3.0) * grad_div_u[1]);
            // Add bulk viscosity contribution
            div_tau(0) += mu_b_eff * grad_div_u(0);
            div_tau(1) += mu_b_eff * grad_div_u(1);

            // WIP: Debugging feild for div_tau
            div_tau_(i, j, k, 0) = div_tau(0);
            div_tau_(i, j, k, 1) = div_tau(1);

            // Curvature
            Set::Scalar kappa = kappas(i, j, k, 0);
            Set::Vector grad_mag_grad_eta = Set::Vector(grad_mag_grad_eta_(i, j, k, 0), grad_mag_grad_eta_(i, j, k, 1));
            
            // Surface Tension:
            // Fsv =  simga * kappa * n_hat
            Fsv(i, j, k) = (0.0, 0.0);
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);
            if (apply_surface_tension)
            {
                // Optimization, only calc surface tension if on interface
                if (grad_eta_mag > 0.01)
                {
                    Set::Scalar sigma_eff = sigma;
                    Set::Scalar alpha = 6 * std::sqrt(2);
                    Set::Scalar UFFDA = epsilon * alpha * grad_eta_mag * grad_eta_mag; 
                    Fsv(i, j, k, 0) = sigma_eff * kappa * n_hat(0) * UFFDA; // / (grad_eta_mag + small)); // / (DX[0] + small);
                    Fsv(i, j, k, 1) = sigma_eff * kappa * n_hat(1) * UFFDA; // / (grad_eta_mag + small)); // / (DX[1] + small);

                }
                Fsv_vector = Set::Vector(Fsv(i, j, k, 0), Fsv(i, j, k, 1));
            }

            // Weight:
            Fw(i, j, k) = (0.0, 0.0);
            Set::Vector Fw_vector = Set::Vector(0.0, 0.0);
            if (apply_weight)
            {
                Fw(i, j, k, 0) = 0.0;
                Fw(i, j, k, 1) = -rho(i, j, k) * g;
                Fw_vector = Set::Vector(Fw(i, j, k, 0), Fw(i, j, k, 1)); // or multiply by cell area if needed
            }

            // Buoyancy:
            Fb(i, j, k) = (0.0, 0.0);
            Set::Vector Fb_vector = Set::Vector(0.0, 0.0);
            if (apply_buoyancy)
            {
                Fb_vector = Set::Vector(0.0, 0.0); // replace mass with actual value if available
            }

            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv(i, j, k, 0) + Fb_vector(0) + Fw_vector(0),
                                                  Fsv(i, j, k, 1) + Fb_vector(1) + Fw_vector(1));

            Source(i, j, k, 0) = mdot0;
            Source(i, j, k, 1) = Pdot0(0) - Ldot0(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) - Ldot0(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(Total_Force); //+ u.dot(Ldot0)

           // Lagrange terms to enforce no-penetration
            Source(i, j, k, 1) -= lagrange * u.dot(grad_eta) * grad_eta(0);
            Source(i, j, k, 2) -= lagrange * u.dot(grad_eta) * grad_eta(1);

            // Riemann solver for mixed fluid
            const int X = 0, Y = 1;

            // Create arrays to store cell states for reconstruction
            std::vector<Solver::Local::Riemann::State> x_states(3);
            std::vector<Solver::Local::Riemann::State> y_states(3);

            // Fill the arrays with cell states
            if (Spec_Vol == 1)
            {
                x_states[0] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i - 1, j, k, X); // x_lo
                x_states[1] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, X);     // x
                x_states[2] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i + 1, j, k, X); // x_hi

                y_states[0] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j - 1, k, Y); // y_lo
                y_states[1] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, Y);     // y
                y_states[2] = Solver::Local::Riemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j + 1, k, Y); // y_hi
            }
            else 
            {
                x_states[0] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i - 1, j, k, X); // x_lo
                x_states[1] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i, j, k, X);     // x
                x_states[2] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i + 1, j, k, X); // x_hi

                y_states[0] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i, j - 1, k, Y); // y_lo
                y_states[1] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i, j, k, Y);     // y
                y_states[2] = Solver::Local::Riemann::State(rho, M, E_mas, gammaf, p0_eff, T, i, j + 1, k, Y); // y_hi
            }
            

            // Variables to store reconstructed states at interfaces
            std::vector<Solver::Local::Riemann::State> x_leftStates(3), x_rightStates(3);
            std::vector<Solver::Local::Riemann::State> y_leftStates(3), y_rightStates(3);

            if (Limiter == 0)
            {
                // No limiter - use cell-centered values directly
                //x_leftStates.resize(3);
                //x_rightStates.resize(3);
                //y_leftStates.resize(3);
                //y_rightStates.resize(3);

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
                //limiter_minmod->reconstructCharacteristicStates(x_states, x_leftStates, x_rightStates, pref, small);
                //limiter_minmod->reconstructCharacteristicStates(y_states, y_leftStates, y_rightStates, pref, small);
            }
            else if (Limiter == 2)
            {
                // Van Leer limiter
                //limiter_vanleer->reconstructCharacteristicStates(x_states, x_leftStates, x_rightStates, pref, small);
                //limiter_vanleer->reconstructCharacteristicStates(y_states, y_leftStates, y_rightStates, pref, small);
            }

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::Riemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;

            try
            {
                if (Riemann_Solver == 0)
                {
                    // Calculate fluxes for the mixed fluid using ROE
                    //flux_xlo = roesolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                    //flux_ylo = roesolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                    //flux_xhi = roesolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                    //flux_yhi = roesolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
                }
                else if (Riemann_Solver == 1)
                {
                    // Calculate fluxes for the mixed fluid using HLLC
                    flux_xlo = hllcsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                    flux_ylo = hllcsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                    flux_xhi = hllcsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                    flux_yhi = hllcsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
                }
                else if (Riemann_Solver == 2)
                {
                    // Calculate fluxes for the mixed fluid using HLLE
                    //flux_xlo = hllesolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                    //flux_ylo = hllesolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                    //flux_xhi = hllesolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                    //flux_yhi = hllesolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
                }
                else if (Riemann_Solver == 3)
                {
                    // Calculate fluxes for the mixed fluid using HLLCE
                    //flux_xlo = hllcesolver->Solve(x_leftStates[1], x_rightStates[1], pref, small);
                    //flux_ylo = hllcesolver->Solve(y_leftStates[1], y_rightStates[1], pref, small);
                    //flux_xhi = hllcesolver->Solve(x_leftStates[2], x_rightStates[2], pref, small);
                    //flux_yhi = hllcesolver->Solve(y_leftStates[2], y_rightStates[2], pref, small);
                }
                else if (Riemann_Solver == 35)
                {
                    // Calculate fluxes for the mixed fluid using HLLC_WENO5
                    /*
                    flux_xlo = hllc_weno5solver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                    flux_ylo = hllc_weno5solver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                    flux_xhi = hllc_weno5solver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                    flux_yhi = hllc_weno5solver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
                    */
                }
                else if (Riemann_Solver == 36)
                {
                    // Calculate fluxes for the mixed fluid using PPM
                    //flux_xlo = partiallyparabolicsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, mu_eff, k_thermal(i, j, k), dt, DX[0]);
                    //flux_ylo = partiallyparabolicsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, mu_eff, k_thermal(i, j, k), dt, DX[1]);
                    //flux_xhi = partiallyparabolicsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, mu_eff, k_thermal(i, j, k), dt, DX[0]);
                    //flux_yhi = partiallyparabolicsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, mu_eff, k_thermal(i, j, k), dt, DX[1]);
                }
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::Abort(INFO);
            }


            // UPDATE MIXED FLUID VARIABLES

            // Update Source Terms to account for moving boundry
            // Delete me if does not worky :(
            // Source(i, j, k, 0) = Source(i, j, k, 0) - rho(i, j, k) * deta_dt;
            // Source(i, j, k, 1) = Source(i, j, k, 1) - M(i, j, k, 0) * deta_dt;
            // Source(i, j, k, 2) = Source(i, j, k, 2) - M(i, j, k, 1) * deta_dt;
            // Source(i, j, k, 3) = Source(i, j, k, 3) - E_vol(i, j, k) * deta_dt;

            // DEBUGGING:
            rho_flux(i, j, k) = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]);
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]);
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]);
            E_flux(i, j, k) = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]);

            // Density
            Set::Scalar drho_dt = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]) + Source(i, j, k, 0);

            rho_new(i, j, k) = rho(i, j, k) + (drho_dt)*dt;
            if (rho_new(i, j, k) < (0.0 + small)) {
                rho_new(i, j, k) = 0.0 + small;
            }

            // Momentum
            Set::Scalar dMx_dt = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]) + div_tau(0) +
                                 //(mu * (lap_ux * eta(i, j, k))) +
                                 Source(i, j, k, 1);
            M_new(i, j, k, 0) = M(i, j, k, 0) + dMx_dt * dt;

            Set::Scalar dMy_dt = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]) + div_tau(1) +
                                 //(mu * (lap_uy * eta(i, j, k))) +
                                 Source(i, j, k, 2);
            M_new(i, j, k, 1) = M(i, j, k, 1) + dMy_dt * dt;

            // Energy
            Set::Scalar dE_dt = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]) + Source(i, j, k, 3);
            if (Spec_Vol == 1)
            {
                E_vol_new(i, j, k) = E_vol(i, j, k) + dE_dt * dt;
                E_mas_new(i, j, k) = E_vol_new(i, j, k) / (rho(i, j, k) + small);
            }
            else
            {
                E_mas_new(i, j, k) = E_mas(i, j, k) + dE_dt * dt;
                E_vol_new(i, j, k) = E_mas_new(i, j, k) * (rho(i, j, k));
            }
           
            
            /// Eta:
            // Material Derivative
            Set::Vector u_new = Set::Vector(M_new(i, j, k, 0) / rho_new(i, j, k), M_new(i, j, k, 1) / rho_new(i, j, k));
            Set::Scalar deta_dt = -u_new.dot(grad_eta);
            // Cahn-Hillard
            // Set::Scalar deta_dt = -u.dot(grad_eta) + M.dot(grad_eta)
            // Set::Matrix tmp =
            // Set::Scalar deta_dt = -1.0 / (rho(i, j, k) * (u_mag**2))
            // Set::Matrix gradu = (gradM - u * gradrho.transpose()) / rho(i, j, k);

            // Set::Scalar deta_dt = -; //https://www.sciencedirect.com/science/article/pii/S002199912100005X

            //  Either Allen-Cahn or Cahn-Hillar
            // IDK anymore, all are the same
            // Please work
            Set::Scalar Mob = 0.0; // Mobility
            Set::Vector Ugly = Set::Vector(0.0, 0.0);
            Ugly(0) = epsilon * grad_mag_grad_eta(0);
            Ugly(1) = epsilon * grad_mag_grad_eta(1);
            deta_dt = deta_dt + Mob * n_hat.dot(Ugly);

            if (static_eta == 1)
            {
                eta_new(i, j, k) = eta(i, j, k);
            }
            else
            {
                eta_new(i, j, k) = eta(i, j, k) + deta_dt * dt;
                if (eta_new(i, j, k) <= cutoff)
                {
                    eta_new(i, j, k) = 0.0;
                }
                else if (eta_new(i, j, k) >= (1.0 - cutoff))
                {
                    eta_new(i, j, k) = 1.0;
                }
            }
            

            // ERROR CHECKING
            if ((rho_new(i, j, k) != rho_new(i, j, k))
                or (M_new(i, j, k, 0) != M_new(i, j, k, 0))
                or (M_new(i, j, k, 1) != M_new(i, j, k, 1))
                or (E_vol_new(i, j, k) != E_vol_new(i, j, k))
                or (eta_new(i, j, k) != eta_new(i, j, k)))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HYDRO2");
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, ", j=", j);
                Util::ParallelMessage(INFO, "drho_dt=", drho_dt); // dies
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
                Util::ParallelMessage(INFO, "rho=", rho_new(i, j, k));
                Util::ParallelMessage(INFO, "M=", M_new(i, j, k, 0), ", ", M_new(i, j, k, 1));
                Util::ParallelMessage(INFO, "E=", E_vol_new(i, j, k));
                Util::ParallelMessage(INFO, "eta=", eta_new(i, j, k));
                Util::Exception(INFO);
            }

            if (time <= 1e-7 && i == 0 && j == 0)
            { // First timestep, first cell
                Util::ParallelMessage(INFO, "=== FIRST CELL DIAGNOSTICS ===");
                Util::ParallelMessage(INFO, "eta = ", eta(i, j, k));
                Util::ParallelMessage(INFO, "rho = ", rho(i, j, k));
                Util::ParallelMessage(INFO, "M = ", M(i, j, k, 0), ", ", M(i, j, k, 1));
                Util::ParallelMessage(INFO, "E_vol = ", E_vol(i, j, k));
                Util::ParallelMessage(INFO, "press = ", press(i, j, k));

                // Check flux calculation
                Util::ParallelMessage(INFO, "flux_xlo.mass = ", flux_xlo.mass);
                Util::ParallelMessage(INFO, "flux_xhi.mass = ", flux_xhi.mass);
                Util::ParallelMessage(INFO, "flux_xlo.momentum_normal = ", flux_xlo.momentum_normal);
                Util::ParallelMessage(INFO, "flux_xhi.momentum_normal = ", flux_xhi.momentum_normal);
                Util::ParallelMessage(INFO, "flux_xlo.energy = ", flux_xlo.energy);
                Util::ParallelMessage(INFO, "flux_xhi.energy = ", flux_xhi.energy);

                Util::ParallelMessage(INFO, "drho_dt = ", drho_dt);
                Util::ParallelMessage(INFO, "dMx_dt = ", dMx_dt);
                Util::ParallelMessage(INFO, "dE_dt = ", dE_dt);
            }

            // Set::Vector grad_ux = Numeric::Gradient(v, i, j, k, 0, DX);
            // Set::Vector grad_uy = Numeric::Gradient(v, i, j, k, 1, DX);

            // Adaptive Timestep
            Set::Scalar sound_speed = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k) + small));

            c_max = std::max(c_max, sound_speed);
            vx_max = std::max(vx_max, std::abs(v(i, j, k, 0)));
            vy_max = std::max(vy_max, std::abs(v(i, j, k, 1)));

            *dt_max_handle = std::fabs(cfl * DX[0] / (u(0) + small));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl * DX[1] / (u(1) + small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[0] * DX[0] / (Source(i, j, k, 1) + small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[1] * DX[1] / (Source(i, j, k, 2) + small)));

            // Calculate vorticity for visualization
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));
        });
    }
    // Update adaptive timestep
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


#endif
