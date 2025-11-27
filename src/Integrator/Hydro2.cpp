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
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing, 4, nghost, "hess_eta", true, { "00", "01", "10", "11" });
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
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing,  1, nghost, "m0", true);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing, 2, nghost, "u0", true, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing, 2, nghost, "q0", true, { "x", "y" });
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
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc, 1, nghost, "mu_chem", true);                    // Tamman Pressure
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

    // FORCED SOURCE
    ic_m0           ->Initialize(lev, m0_mf, 0.0);
    ic_u0           ->Initialize(lev, u0_mf, 0.0);
    //ic_q->Initialize(lev, q_mf, 0.0);
    q_mf[lev]->setVal(0.0);

    /*
    m0_mf[lev]->FillBoundary(geom[lev].periodicity());
    u0_mf[lev]->FillBoundary(geom[lev].periodicity());
    q_mf[lev]->FillBoundary(geom[lev].periodicity());
    */
    amrex::ParallelDescriptor::Barrier();


    // NATURAL SOURCE
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Fb_mf[lev]      ->setVal(0.0);
    Fw_mf[lev]      ->setVal(0.0); 
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
    amrex::Box domain = geom[lev].Domain();


    // Function is for the diffusive mixing terms. I.E: rho = eta*rho0 + (1-eta)*rho1
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        // FLUID 0
        Set::Patch<const Set::Scalar>   v0          = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p0          = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0        = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0_old    = density0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0          = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0_old      = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E0          = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E0_old      = energy0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   T0          = T0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   cp0         = cp0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   cv0         = cv0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   k0_thermal  = k0_thermal_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   h0_thermal  = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E1          = energy1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   E1_old      = energy1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   T1          = T1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   cp1         = cp1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   cv1         = cv1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   k1_thermal  = k1_thermal_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   h1_thermal  = h1_thermal_mf.Patch(lev, mfi);

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
        Set::Patch<Set::Scalar>         mu_chem_    = mu_chem_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar>         a           = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Ma          = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_vol      = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_mas      = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_vol      = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_mas      = KE_per_mas_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);



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

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

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
Hydro2::RHS(int lev, Set::Scalar time, amrex::MultiFab &rho_rhs_mf, amrex::MultiFab &M_rhs_mf, amrex::MultiFab &E_mas_rhs_mf, amrex::MultiFab &E_vol_rhs_mf, amrex::MultiFab &eta_rhs_mf, const amrex::MultiFab &rho_mf_in, const amrex::MultiFab &M_mf_in, const amrex::MultiFab &E_mas_mf_in, const amrex::MultiFab &E_vol_mf_in, const amrex::MultiFab &eta_mf_in)
{
    BL_PROFILE("Hydro2::RHS");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Create non-const copies for boundary filling
    amrex::MultiFab rho_temp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, rho_mf_in.nGrow());
    amrex::MultiFab M_temp(M_mf_in.boxArray(), M_mf_in.DistributionMap(), 2, M_mf_in.nGrow());
    amrex::MultiFab E_mas_temp(E_mas_mf_in.boxArray(), E_mas_mf_in.DistributionMap(), 1, E_mas_mf_in.nGrow());
    amrex::MultiFab E_vol_temp(E_vol_mf_in.boxArray(), E_vol_mf_in.DistributionMap(), 1, E_vol_mf_in.nGrow());
    amrex::MultiFab eta_temp(eta_mf_in.boxArray(), eta_mf_in.DistributionMap(), 1, eta_mf_in.nGrow());

    amrex::MultiFab::Copy(rho_temp, rho_mf_in, 0, 0, 1, rho_mf_in.nGrow());
    amrex::MultiFab::Copy(M_temp, M_mf_in, 0, 0, 2, M_mf_in.nGrow());
    amrex::MultiFab::Copy(E_mas_temp, E_mas_mf_in, 0, 0, 1, E_mas_mf_in.nGrow());
    amrex::MultiFab::Copy(E_vol_temp, E_vol_mf_in, 0, 0, 1, E_vol_mf_in.nGrow());
    amrex::MultiFab::Copy(eta_temp, eta_mf_in, 0, 0, 1, eta_mf_in.nGrow());

    // Fill boundary conditions
    rho_temp.FillBoundary(geom[lev].periodicity());
    M_temp.FillBoundary(geom[lev].periodicity());
    E_mas_temp.FillBoundary(geom[lev].periodicity());
    E_vol_temp.FillBoundary(geom[lev].periodicity());
    eta_temp.FillBoundary(geom[lev].periodicity());

    density_bc->FillBoundary(rho_temp, 0, 1, time, 0);
    momentum_bc->FillBoundary(M_temp, 0, 2, time, 0);
    energy_bc->FillBoundary(E_mas_temp, 0, 1, time, 0);
    energy_bc->FillBoundary(E_vol_temp, 0, 1, time, 0);
    energy_bc->FillBoundary(eta_temp, 0, 1, time, 0);


    // Zero out RHS arrays
    rho_rhs_mf.setVal(0.0);
    M_rhs_mf.setVal(0.0);
    E_mas_rhs_mf.setVal(0.0);
    E_vol_rhs_mf.setVal(0.0);
    eta_rhs_mf.setVal(0.0);

    // ============================================
    // STEP 1: Calculate derived quantities
    // (Lines 1095-1165 from NewCode.pdf)
    // ============================================
    for (amrex::MFIter mfi(eta_temp, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        Set::Patch<const Set::Scalar> eta = eta_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> rho = rho_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> E_mas = E_mas_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> E_vol = E_vol_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> M = M_temp.const_array(mfi);

        Set::Patch<Set::Scalar> v = velocity_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> press = pressure_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> eta_rhs = eta_rhs_mf.array(mfi);

        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar eta_val = eta(i, j, k);

            // Effective properties (lines 1099-1103)
            Set::Scalar gamma_eff = eta_val * gamma0 + (1.0 - eta_val) * gamma1;
            Set::Scalar p0_eff = eta_val * p0_0 + (1.0 - eta_val) * p0_1;
            Set::Scalar mu_eff = eta_val * mu0 + (1.0 - eta_val) * mu1;
            Set::Scalar mu_b_eff = eta_val * mu0_b + (1.0 - eta_val) * mu1_b;

            // Velocity (lines 1105-1107)
            Set::Scalar rho_reg = std::max(rho(i, j, k), small);
            v(i, j, k, 0) = M(i, j, k, 0) / rho_reg;
            v(i, j, k, 1) = M(i, j, k, 1) / rho_reg;

            // Kinetic energy (lines 1109-1111)
            Set::Scalar KE_mas = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            Set::Scalar KE_vol = rho(i, j, k) * KE_mas;

            // Internal energy (lines 1113-1115)
            Set::Scalar UE_mas = E_mas(i, j, k) - KE_mas;
            Set::Scalar UE_vol = E_vol(i, j, k) - KE_vol;

            // Pressure - Tammann EOS (lines 1117-1120)
            Set::Scalar A = 1.0 / (gamma_eff - 1.0);
            Set::Scalar B = p0_eff * A;
            press(i, j, k) = (UE_vol - B) / A - pref;

            // Gradients for eta (lines 1122-1125)
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX, sten);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();

            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

            // Hessian and curvature (lines 1127-1145)
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = hess_eta(0, 0) + hess_eta(1, 1);

            Set::Scalar kappa = 0.0;
            if (kappa_method == 1)
            {
                Set::Vector grad_mag_grad_eta = Set::Vector(
                    1.0 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                    1.0 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));

                grad_mag_grad_eta_(i, j, k, 0) = grad_mag_grad_eta(0);
                grad_mag_grad_eta_(i, j, k, 1) = grad_mag_grad_eta(1);

                kappa = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / (grad_eta_mag * grad_eta_mag + small)));
            }
            else if (kappa_method == 2)
            {
                if (grad_eta_mag > small)
                {
                    Set::Vector n = grad_eta / grad_eta_mag;
                    kappa = -(hess_eta(0, 0) * n(0) * n(0) + 2.0 * hess_eta(0, 1) * n(0) * n(1) + hess_eta(1, 1) * n(1) * n(1));
                }
            }

            kappas(i, j, k, 0) = kappa;

            // Eta RHS (lines 1147-1149)
            eta_rhs(i, j, k) = -(v(i, j, k, 0) * grad_eta(0) + v(i, j, k, 1) * grad_eta(1));

            // Pressure check (lines 1151-1156)
            if (press(i, j, k) > 1E100 || std::isnan(press(i, j, k)))
            {
                printf("ERROR at (%d,%d,%d): press=%e, rho=%e, E_vol=%e, E_mas=%e, eta=%e\n",
                       i,
                       j,
                       k,
                       press(i, j, k),
                       rho(i, j, k),
                       E_vol(i, j, k),
                       E_mas(i, j, k),
                       eta(i, j, k));
            }
        });
    }

    // ============================================
    // STEP 2: Compute fluxes and forces
    // (Lines 1167-1310 from NewCode.pdf)
    // ============================================
    for (amrex::MFIter mfi(eta_temp, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<const Set::Scalar> rho = rho_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> E_mas = E_mas_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> E_vol = E_vol_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> M = M_temp.const_array(mfi);
        Set::Patch<const Set::Scalar> eta = eta_temp.const_array(mfi);

        Set::Patch<const Set::Scalar> v = velocity_mf[lev]->const_array(mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf[lev]->const_array(mfi);

        Set::Patch<Set::Scalar> rho_rhs = rho_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_mas_rhs = E_mas_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_vol_rhs = E_vol_rhs_mf.array(mfi);

        Set::Patch<const Set::Scalar> m0 = m0_mf[lev]->const_array(mfi);
        Set::Patch<const Set::Scalar> q = q_mf[lev]->const_array(mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf[lev]->const_array(mfi);
        Set::Patch<Set::Scalar> Source = Source_mf[lev]->array(mfi);

        // YOUR force arrays (lines 1181-1183)
        Set::Patch<Set::Scalar> Fsv = Fsv_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> Fb = Fb_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> Fw = Fw_mf[lev]->array(mfi);

        Set::Patch<const Set::Scalar> grad_eta_ = grad_eta_mf[lev]->const_array(mfi);
        Set::Patch<const Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf[lev]->const_array(mfi);
        Set::Patch<const Set::Scalar> kappas = kappas_mf[lev]->const_array(mfi);

        // Viscosity debugging arrays (lines 1184-1186)
        Set::Patch<Set::Scalar> grad_div_u_ = grad_div_u_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);

            Set::Scalar eta_val = eta(i, j, k);
            Set::Scalar gamma_eff = eta_val * gamma0 + (1.0 - eta_val) * gamma1;
            Set::Scalar p0_eff = eta_val * p0_0 + (1.0 - eta_val) * p0_1;
            Set::Scalar mu_eff = eta_val * mu0 + (1.0 - eta_val) * mu1;
            Set::Scalar mu_b_eff = eta_val * mu0_b + (1.0 - eta_val) * mu1_b;

            // ============================================
            // RECONSTRUCT STATES (lines 1188-1220)
            // ============================================

            Solver::Local::Riemann::State x_states[3];
            Solver::Local::Riemann::State y_states[3];

            for (int s = 0; s < 3; s++)
            {
                int ii = i + s - 1;

                if (Spec_Vol == 1)
                {
                    x_states[s] = Solver::Local::Riemann::State(rho, M, E_vol, gamma_eff, p0_eff, T, ii, j, k, X);
                }
                else
                {
                    // For mass-specific, need to convert
                    x_states[s] = Solver::Local::Riemann::State(rho, M, E_vol, gamma_eff, p0_eff, T, ii, j, k, X);
                }
            }

            for (int s = 0; s < 3; s++)
            {
                int jj = j + s - 1;

                if (Spec_Vol == 1)
                {
                    y_states[s] = Solver::Local::Riemann::State(rho, M, E_vol, gamma_eff, p0_eff, T, i, jj, k, Y);
                }
                else
                {
                    y_states[s] = Solver::Local::Riemann::State(rho, M, E_vol, gamma_eff, p0_eff, T, i, jj, k, Y);
                }
            }


            // ============================================
            // COMPUTE FLUXES (lines 1222-1228)
            // ============================================

            Solver::Local::Riemann::Flux flux_xlo, flux_xhi, flux_ylo, flux_yhi;

            if (Riemann_Solver == 1) // HLLC
            {
                flux_xlo = hllcsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = hllcsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = hllcsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = hllcsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            // Add other solvers here as needed (commented out as requested)

            // ============================================
            // VISCOSITY (lines 1000-1052 from NewCode.pdf)
            // ============================================

            Set::Scalar rho_reg = std::max(rho(i, j, k), small);

            // Velocity gradients
            Set::Vector grad_u = Numeric::Gradient(M, i, j, k, 0, DX, sten) / rho_reg;
            Set::Vector grad_v = Numeric::Gradient(M, i, j, k, 1, DX, sten) / rho_reg;

            // Divergence of velocity
            Set::Scalar div_u = grad_u(0) + grad_v(1);

            // Strain rate tensor
            Set::Scalar S_xx = grad_u(0);
            Set::Scalar S_yy = grad_v(1);
            Set::Scalar S_xy = 0.5 * (grad_u(1) + grad_v(0));

            // Deviatoric stress tensor
            Set::Scalar tau_xx = 2.0 * mu_eff * (S_xx - div_u / 3.0);
            Set::Scalar tau_yy = 2.0 * mu_eff * (S_yy - div_u / 3.0);
            Set::Scalar tau_xy = 2.0 * mu_eff * S_xy;

            // Divergence of stress tensor
            Set::Vector div_tau;
            Set::Scalar tau_xx_arr[3][3]; // Local stencil storage
            Set::Scalar tau_xy_arr[3][3];
            Set::Scalar tau_yy_arr[3][3];

            // Fill stencil
            for (int di = -1; di <= 1; di++)
            {
                for (int dj = -1; dj <= 1; dj++)
                {
                    int ii = i + di;
                    int jj = j + dj;

                    // Compute velocity gradients at (ii, jj)
                    auto sten_local = Numeric::GetStencil(ii, jj, k, bx);
                    Set::Scalar rho_local = std::max(rho(ii, jj, k), small);
                    Set::Vector grad_u_local = Numeric::Gradient(M, ii, jj, k, 0, DX, sten_local) / rho_local;
                    Set::Vector grad_v_local = Numeric::Gradient(M, ii, jj, k, 1, DX, sten_local) / rho_local;
                    Set::Scalar div_u_local = grad_u_local(0) + grad_v_local(1);

                    Set::Scalar eta_local = eta(ii, jj, k);
                    Set::Scalar mu_eff_local = eta_local * mu0 + (1.0 - eta_local) * mu1;
                    Set::Scalar mu_b_eff_local = eta_local * mu0_b + (1.0 - eta_local) * mu1_b;

                    tau_xx_arr[di + 1][dj + 1] = 2.0 * mu_eff_local * (grad_u_local(0) - div_u_local / 3.0);
                    tau_yy_arr[di + 1][dj + 1] = 2.0 * mu_eff_local * (grad_v_local(1) - div_u_local / 3.0);
                    tau_xy_arr[di + 1][dj + 1] = 2.0 * mu_eff_local * 0.5 * (grad_u_local(1) + grad_v_local(0));
                }
            }

            // Now compute divergence using finite differences
            Set::Scalar dtau_xx_dx = (tau_xx_arr[2][1] - tau_xx_arr[0][1]) / (2.0 * DX[0]);
            Set::Scalar dtau_xy_dy = (tau_xy_arr[1][2] - tau_xy_arr[1][0]) / (2.0 * DX[1]);
            Set::Vector div_tau;
            div_tau(0) = dtau_xx_dx + dtau_xy_dy;

            Set::Scalar dtau_xy_dx = (tau_xy_arr[2][1] - tau_xy_arr[0][1]) / (2.0 * DX[0]);
            Set::Scalar dtau_yy_dy = (tau_yy_arr[1][2] - tau_yy_arr[1][0]) / (2.0 * DX[1]);
            div_tau(1) = dtau_xy_dx + dtau_yy_dy;
            //div_tau(0) += mu_b_eff * ddiv_u_dx;
            //div_tau(1) += mu_b_eff * ddiv_u_dy;

            div_tau_(i, j, k, 0) = div_tau(0);
            div_tau_(i, j, k, 1) = div_tau(1);

            // ============================================
            // SURFACE TENSION (lines 1062-1092 from NewCode.pdf)
            // ============================================

            Set::Scalar kappa = kappas(i, j, k, 0);
            Set::Vector grad_eta = Set::Vector(grad_eta_(i, j, k, 0), grad_eta_(i, j, k, 1));
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();

            Fsv(i, j, k, 0) = 0.0;
            Fsv(i, j, k, 1) = 0.0;
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);

            if (apply_surface_tension)
            {
                if (grad_eta_mag > 0.01)
                {
                    Set::Scalar sigma_eff = sigma;
                    Set::Scalar alpha = std::tanh((eta_val - 0.5) / epsilon);
                    sigma_eff = sigma * alpha;

                    Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
                    Fsv_vector = sigma_eff * kappa * n_hat;

                    Fsv(i, j, k, 0) = Fsv_vector(0);
                    Fsv(i, j, k, 1) = Fsv_vector(1);
                }
            }

            // ============================================
            // BUOYANCY (lines 953-970 from NewCode.pdf)
            // ============================================

            Fb(i, j, k, 0) = 0.0;
            Fb(i, j, k, 1) = 0.0;
            Set::Vector Fb_vector = Set::Vector(0.0, 0.0);

            if (apply_buoyancy)
            {
                Set::Scalar rho_ref = lagrange;
                Fb_vector(1) = (rho(i, j, k) - rho_ref) * g;

                Fb(i, j, k, 0) = Fb_vector(0);
                Fb(i, j, k, 1) = Fb_vector(1);
            }

            // ============================================
            // WEIGHT (lines 972-989 from NewCode.pdf)
            // ============================================

            Fw(i, j, k, 0) = 0.0;
            Fw(i, j, k, 1) = 0.0;
            Set::Vector Fw_vector = Set::Vector(0.0, 0.0);

            if (apply_weight)
            {
                Fw_vector(1) = rho(i, j, k) * g;

                Fw(i, j, k, 0) = Fw_vector(0);
                Fw(i, j, k, 1) = Fw_vector(1);
            }

            // ============================================
            // FLUX DIVERGENCE (lines 1269-1278)
            // ============================================

            Set::Scalar drho_dt = (flux_xlo.mass - flux_xhi.mass) / DX[0] + (flux_ylo.mass - flux_yhi.mass) / DX[1];

            Set::Scalar dMx_dt = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / DX[0] + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / DX[1];

            Set::Scalar dMy_dt = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / DX[0] + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / DX[1];

            Set::Scalar dE_dt = (flux_xlo.energy - flux_xhi.energy) / DX[0] + (flux_ylo.energy - flux_yhi.energy) / DX[1];

            // ============================================
            // SOURCE TERMS (lines 1230-1233)
            // ============================================

            Set::Scalar m0_val = m0(i, j, k);
            Set::Vector u0_val(0.0, 0.0);
            u0_val(0) = _u0(i, j, k, 0);
            u0_val(1) = _u0(i, j, k, 1);
            Set::Vector q_val(0.0, 0.0);
            q_val(0) = q(i, j, k, 0);
            q_val(1) = q(i, j, k, 1);

            Source(i, j, k, 0) = m0_val;
            Source(i, j, k, 1) = m0_val * u0_val(0);
            Source(i, j, k, 2) = m0_val * u0_val(1);
            Source(i, j, k, 3) = m0_val * 0.5 * (u0_val(0) * u0_val(0) + u0_val(1) * u0_val(1)) + q_val(0) + q_val(1);

            // ============================================
            // FINAL RHS (lines 1280-1310)
            // ============================================

            // Density
            rho_rhs(i, j, k) = drho_dt + Source(i, j, k, 0);

            // Momentum (with viscosity and forces)
            M_rhs(i, j, k, 0) = dMx_dt + div_tau(0) + Fsv_vector(0) + Fb_vector(0) + Fw_vector(0) + Source(i, j, k, 1);
            M_rhs(i, j, k, 1) = dMy_dt + div_tau(1) + Fsv_vector(1) + Fb_vector(1) + Fw_vector(1) + Source(i, j, k, 2);

            // Energy
            if (Spec_Vol == 1)
            {
                E_vol_rhs(i, j, k) = dE_dt + Source(i, j, k, 3);
                E_mas_rhs(i, j, k) = (E_vol_rhs(i, j, k) - E_vol(i, j, k) * rho_rhs(i, j, k) / rho_reg) / rho_reg;
            }
            else
            {
                E_vol_rhs(i, j, k) = dE_dt + Source(i, j, k, 3);
                E_mas_rhs(i, j, k) = (E_vol_rhs(i, j, k) - E_mas(i, j, k) * rho_rhs(i, j, k)) / rho_reg;
            }
        });
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::Advance(int lev, amrex::Real time, amrex::Real dt)
{
    BL_PROFILE("Hydro2::Advance");

    // Get grid information
    const amrex::BoxArray &ba = density_mf[lev]->boxArray();
    const amrex::DistributionMapping &dm = density_mf[lev]->DistributionMap();
    const int ng = density_mf[lev]->nGrow();

    // Swap pointers for old and new values (YOUR original approach)
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(E_mas_old_mf[lev], E_mas_mf[lev]);
    std::swap(E_vol_old_mf[lev], E_vol_mf[lev]);
    std::swap(eta_old_mf[lev], eta_mf[lev]);

    // Now "old" points to previous timestep, "new" will be updated
    amrex::MultiFab &density_old = *density_old_mf[lev];
    amrex::MultiFab &momentum_old = *momentum_old_mf[lev];
    amrex::MultiFab &E_mas_old = *E_mas_old_mf[lev];
    amrex::MultiFab &E_vol_old = *E_vol_old_mf[lev];
    amrex::MultiFab &eta_old = *eta_old_mf[lev];

    amrex::MultiFab &density_new = *density_mf[lev];
    amrex::MultiFab &momentum_new = *momentum_mf[lev];
    amrex::MultiFab &E_mas_new = *E_mas_mf[lev];
    amrex::MultiFab &E_vol_new = *E_vol_mf[lev];
    amrex::MultiFab &eta_new = *eta_mf[lev];

    // Temporary storage for RHS
    amrex::MultiFab density_rhs(ba, dm, 1, ng);
    amrex::MultiFab momentum_rhs(ba, dm, 2, ng);
    amrex::MultiFab E_mas_rhs(ba, dm, 1, ng);
    amrex::MultiFab E_vol_rhs(ba, dm, 1, ng);
    amrex::MultiFab eta_rhs(ba, dm, 1, ng);

    // ============================================
    // TIME INTEGRATION
    // ============================================

    if (time_integration_scheme == 0) // Forward Euler
    {
        // Calculate RHS
        RHS(lev, time, density_rhs, momentum_rhs, E_mas_rhs, E_vol_rhs, eta_rhs, density_old, momentum_old, E_mas_old, E_vol_old, eta_old);

        // Update: y_new = y_old + dt * RHS
        amrex::MultiFab::LinComb(density_new, 1.0, density_old, 0, dt, density_rhs, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_new, 1.0, momentum_old, 0, dt, momentum_rhs, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(E_mas_new, 1.0, E_mas_old, 0, dt, E_mas_rhs, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(E_vol_new, 1.0, E_vol_old, 0, dt, E_vol_rhs, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_new, 1.0, eta_old, 0, dt, eta_rhs, 0, 0, 1, 0);
    }
    else if (time_integration_scheme == 1) // RK4
    {
        // Temporary storage for intermediate stages
        amrex::MultiFab density_temp(ba, dm, 1, ng);
        amrex::MultiFab momentum_temp(ba, dm, 2, ng);
        amrex::MultiFab E_mas_temp(ba, dm, 1, ng);
        amrex::MultiFab E_vol_temp(ba, dm, 1, ng);
        amrex::MultiFab eta_temp(ba, dm, 1, ng);

        // K arrays for each stage
        amrex::MultiFab density_k1(ba, dm, 1, ng), density_k2(ba, dm, 1, ng);
        amrex::MultiFab density_k3(ba, dm, 1, ng), density_k4(ba, dm, 1, ng);
        amrex::MultiFab momentum_k1(ba, dm, 2, ng), momentum_k2(ba, dm, 2, ng);
        amrex::MultiFab momentum_k3(ba, dm, 2, ng), momentum_k4(ba, dm, 2, ng);
        amrex::MultiFab E_mas_k1(ba, dm, 1, ng), E_mas_k2(ba, dm, 1, ng);
        amrex::MultiFab E_mas_k3(ba, dm, 1, ng), E_mas_k4(ba, dm, 1, ng);
        amrex::MultiFab E_vol_k1(ba, dm, 1, ng), E_vol_k2(ba, dm, 1, ng);
        amrex::MultiFab E_vol_k3(ba, dm, 1, ng), E_vol_k4(ba, dm, 1, ng);
        amrex::MultiFab eta_k1(ba, dm, 1, ng), eta_k2(ba, dm, 1, ng);
        amrex::MultiFab eta_k3(ba, dm, 1, ng), eta_k4(ba, dm, 1, ng);

        // Stage 1: K1 = RHS(t, y_old)
        RHS(lev, time, density_k1, momentum_k1, E_mas_k1, E_vol_k1, eta_k1, density_old, momentum_old, E_mas_old, E_vol_old, eta_old);

        // Stage 2: K2 = RHS(t + dt/2, y_old + dt/2 * K1)
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt / 2.0, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt / 2.0, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(E_mas_temp, 1.0, E_mas_old, 0, dt / 2.0, E_mas_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(E_vol_temp, 1.0, E_vol_old, 0, dt / 2.0, E_vol_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt / 2.0, eta_k1, 0, 0, 1, 0);

        RHS(lev, time + dt / 2.0, density_k2, momentum_k2, E_mas_k2, E_vol_k2, eta_k2, density_temp, momentum_temp, E_mas_temp, E_vol_temp, eta_temp);

        // Stage 3: K3 = RHS(t + dt/2, y_old + dt/2 * K2)
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt / 2.0, density_k2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt / 2.0, momentum_k2, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(E_mas_temp, 1.0, E_mas_old, 0, dt / 2.0, E_mas_k2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(E_vol_temp, 1.0, E_vol_old, 0, dt / 2.0, E_vol_k2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt / 2.0, eta_k2, 0, 0, 1, 0);

        RHS(lev, time + dt / 2.0, density_k3, momentum_k3, E_mas_k3, E_vol_k3, eta_k3, density_temp, momentum_temp, E_mas_temp, E_vol_temp, eta_temp);

        // Stage 4: K4 = RHS(t + dt, y_old + dt * K3)
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt, density_k3, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt, momentum_k3, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(E_mas_temp, 1.0, E_mas_old, 0, dt, E_mas_k3, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(E_vol_temp, 1.0, E_vol_old, 0, dt, E_vol_k3, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt, eta_k3, 0, 0, 1, 0);

        RHS(lev, time + dt, density_k4, momentum_k4, E_mas_k4, E_vol_k4, eta_k4, density_temp, momentum_temp, E_mas_temp, E_vol_temp, eta_temp);

        // Final update: y_new = y_old + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
        for (amrex::MFIter mfi(eta_new, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();

            auto const &rho_old_arr = density_old.const_array(mfi);
            auto const &rho_new_arr = density_new.array(mfi);
            auto const &rho_k1_arr = density_k1.const_array(mfi);
            auto const &rho_k2_arr = density_k2.const_array(mfi);
            auto const &rho_k3_arr = density_k3.const_array(mfi);
            auto const &rho_k4_arr = density_k4.const_array(mfi);

            auto const &M_old_arr = momentum_old.const_array(mfi);
            auto const &M_new_arr = momentum_new.array(mfi);
            auto const &M_k1_arr = momentum_k1.const_array(mfi);
            auto const &M_k2_arr = momentum_k2.const_array(mfi);
            auto const &M_k3_arr = momentum_k3.const_array(mfi);
            auto const &M_k4_arr = momentum_k4.const_array(mfi);

            auto const &E_mas_old_arr = E_mas_old.const_array(mfi);
            auto const &E_mas_new_arr = E_mas_new.array(mfi);
            auto const &E_mas_k1_arr = E_mas_k1.const_array(mfi);
            auto const &E_mas_k2_arr = E_mas_k2.const_array(mfi);
            auto const &E_mas_k3_arr = E_mas_k3.const_array(mfi);
            auto const &E_mas_k4_arr = E_mas_k4.const_array(mfi);

            auto const &E_vol_old_arr = E_vol_old.const_array(mfi);
            auto const &E_vol_new_arr = E_vol_new.array(mfi);
            auto const &E_vol_k1_arr = E_vol_k1.const_array(mfi);
            auto const &E_vol_k2_arr = E_vol_k2.const_array(mfi);
            auto const &E_vol_k3_arr = E_vol_k3.const_array(mfi);
            auto const &E_vol_k4_arr = E_vol_k4.const_array(mfi);

            auto const &eta_old_arr = eta_old.const_array(mfi);
            auto const &eta_new_arr = eta_new.array(mfi);
            auto const &eta_k1_arr = eta_k1.const_array(mfi);
            auto const &eta_k2_arr = eta_k2.const_array(mfi);
            auto const &eta_k3_arr = eta_k3.const_array(mfi);
            auto const &eta_k4_arr = eta_k4.const_array(mfi);

            const Set::Scalar cutoff_val = cutoff;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new_arr(i, j, k) = rho_old_arr(i, j, k) + dt / 6.0 * (rho_k1_arr(i, j, k) + 2.0 * rho_k2_arr(i, j, k) + 2.0 * rho_k3_arr(i, j, k) + rho_k4_arr(i, j, k));

                M_new_arr(i, j, k, 0) = M_old_arr(i, j, k, 0) + dt / 6.0 * (M_k1_arr(i, j, k, 0) + 2.0 * M_k2_arr(i, j, k, 0) + 2.0 * M_k3_arr(i, j, k, 0) + M_k4_arr(i, j, k, 0));

                M_new_arr(i, j, k, 1) = M_old_arr(i, j, k, 1) + dt / 6.0 * (M_k1_arr(i, j, k, 1) + 2.0 * M_k2_arr(i, j, k, 1) + 2.0 * M_k3_arr(i, j, k, 1) + M_k4_arr(i, j, k, 1));

                E_mas_new_arr(i, j, k) = E_mas_old_arr(i, j, k) + dt / 6.0 * (E_mas_k1_arr(i, j, k) + 2.0 * E_mas_k2_arr(i, j, k) + 2.0 * E_mas_k3_arr(i, j, k) + E_mas_k4_arr(i, j, k));

                E_vol_new_arr(i, j, k) = E_vol_old_arr(i, j, k) + dt / 6.0 * (E_vol_k1_arr(i, j, k) + 2.0 * E_vol_k2_arr(i, j, k) + 2.0 * E_vol_k3_arr(i, j, k) + E_vol_k4_arr(i, j, k));

                eta_new_arr(i, j, k) = eta_old_arr(i, j, k) + dt / 6.0 * (eta_k1_arr(i, j, k) + 2.0 * eta_k2_arr(i, j, k) + 2.0 * eta_k3_arr(i, j, k) + eta_k4_arr(i, j, k));

                // Apply eta cutoff (from YOUR NewCode.pdf)
                if (eta_new_arr(i, j, k) <= cutoff_val)
                {
                    eta_new_arr(i, j, k) = 0.0;
                }
                else if (eta_new_arr(i, j, k) >= (1.0 - cutoff_val))
                {
                    eta_new_arr(i, j, k) = 1.0;
                }
            });
        }
    }

    // ============================================
    // ERROR CHECKING (from YOUR code)
    // ============================================

    if (density_new.contains_nan() || momentum_new.contains_nan() || E_mas_new.contains_nan() || E_vol_new.contains_nan() || eta_new.contains_nan())
    {
        amrex::Print() << "NaN detected at time = " << time + dt << "\n";
        amrex::Print() << "  Max |rho| = " << density_new.norm0() << "\n";
        amrex::Print() << "  Max |M| = " << momentum_new.norm0() << "\n";
        amrex::Print() << "  Max |E_mas| = " << E_mas_new.norm0() << "\n";
        amrex::Print() << "  Max |E_vol| = " << E_vol_new.norm0() << "\n";
        amrex::Print() << "  Max |eta| = " << eta_new.norm0() << "\n";
        amrex::Abort("NaN values in solution");
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
