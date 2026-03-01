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
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
//#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
//#include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
//#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"

// Limiters
//#include "Solver/Local/Limiter/Minmod.H"
//#include "Solver/Local/Limiter/VanLeer.H"

#include <AMReX_Math.H>
#include "AMReX_TimeIntegrator.H"


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
                      //Solver::Local::FluidRiemann::HLLCE, // WIP - Need to update
                      //Solver::Local::FluidRiemann::HLLCE_WENO5, // Never verified but updated
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman, // Can't remember if this has been verified
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro,
                      Solver::Local::FluidRiemann::Upwind
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
            UE_mas(i, j, k) = (UE_vol(i, j, k)) / (rho(i, j, k));
            
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
            gammaf(i, j, k) = 1.0 + (1.0 / A);

            // Pressure
            p0_eff(i, j, k) = (B / A) / gammaf(i, j, k);
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE_vol(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref; // pressure Tammann EOS modification

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho0(i, j, k) * eta(i, j, k) / (rho(i, j, k));

            // Spalding Number
            Bm(i, j, k) = (Y(i, j, k) - Y_infinity) / (1 + Y_infinity + small);
            
            // Constant Pressure Specific Heat
            cp(i, j, k) = eta(i, j, k) * cp0 + (1.0 - eta(i, j, k)) * cp1;

            // Constant Volume Specific Heat
            cv(i, j, k) = eta(i, j, k) * cv0 + (1.0 - eta(i, j, k)) * cv1;

            // Temperature
            // T(i, j, k) = T0(i, j, k) * eta(i, j, k) + T1(i, j, k) * (1.0 - eta(i, j, k));
            T(i, j, k) = UE_vol(i, j, k) / (rho(i, j, k) * cv(i, j, k) + small); // Volume Specific
            // T(i, j, k) = UE_vol(i, j, k) / (cv(i, j, k) + small); // Mass Specific

            // Thermal Conductivity
            //k_thermal(i, j, k) = eta(i, j, k) * k0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * k1_thermal(i, j, k);

            // Thermal Convectivity
            //h_thermal(i, j, k) = eta(i, j, k) * h0_thermal(i, j, k) + (1.0 - eta(i, j, k)) * h1_thermal(i, j, k);

            // Speed of Sound
            a(i, j, k) = sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

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
    const amrex::MultiFab &E_mf_in) //, const amrex::MultiFab &velocity_mf_in, const amrex::MultiFab &pressure_mf_in, const amrex::MultiFab &T_mf_in)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Primitive Fields
    //for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    for (amrex::MFIter mfi(*(velocity_mf)[lev], true); mfi.isValid(); ++mfi)
        {
        const amrex::Box &bx = mfi.validbox();

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho = rho_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> M = M_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> E = E_mf_in.array(mfi);

        // PRIMITIVE
        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> a = a_mf.Patch(lev, mfi);

        // SINGLE PHASE
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);

        // SOURCE - ish
        Set::Patch<Set::Scalar> KE = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> T = T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Y = Y_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Derivatice Function Calls
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // gamma
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            gammaf(i, j, k) = 1.0 + (1.0 / A);

            // Velocity
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k));
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k));

            // Kinetic Energy
            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1)); // Per Vol
            // KE(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1)); // Per Mass

            // Potential Energy
            UE(i, j, k) = E(i, j, k) - KE(i, j, k);

            // Pressure
            p0_eff(i, j, k) = (B / A) / gammaf(i, j, k);
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref;                // Per Vol
            //press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE(i, j, k) * rho(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref; // Per Mass
            //press(i, j, k) = std::max(small, press(i, j, k));

            // Constant Pressure Specific Heat
            cp(i, j, k) = eta(i, j, k) * cp0 + (1.0 - eta(i, j, k)) * cp1;

            // Constant Volume Specific Heat
            cv(i, j, k) = eta(i, j, k) * cv0 + (1.0 - eta(i, j, k)) * cv1;

            // Temperature
            // T(i, j, k) = T0(i, j, k) * eta(i, j, k) + T1(i, j, k) * (1.0 - eta(i, j, k));
            T(i, j, k) = UE(i, j, k) / (rho(i, j, k) * cv(i, j, k) + small); // Volume Specific
            // T(i, j, k) = UE(i, j, k) / (cv(i, j, k) + small); // Mass Specific

            // Speed of sound:
            a(i, j, k) = sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho0(i, j, k) * eta(i, j, k) / (rho(i, j, k));
            
            // Spalding Number
            Bm(i, j, k) = (Y(i, j, k) - Y_infinity) / (1 + Y_infinity + small);

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

                    // t1 = Set::Vector(-n_hat(1), n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
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
        });
    }

    // Main time integration loop
    //for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    for (amrex::MFIter mfi(*(velocity_mf)[lev], true); mfi.isValid(); ++mfi)
        {
        const amrex::Box &bx = mfi.validbox();
        // PRIMARY FLUIDS
        // FLUID 0
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        /*
        Set::Patch<const Set::Scalar> v0 = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0 = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M0 = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E0 = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T0 = T0_mf.Patch(lev, mfi);
        */

        // FLUID 1
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);
        /*
        Set::Patch<const Set::Scalar> v1 = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p1 = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M1 = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1 = energy1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T1 = T1_mf.Patch(lev, mfi);
        */

        // Mixture
        Set::Patch<const Set::Scalar> eta = eta_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho = rho_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> M = M_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> E = E_mf_in.array(mfi);

        // OUTPUTS
        Set::Patch<Set::Scalar> eta_rhs = eta_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_rhs = rho_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs = E_rhs_mf.array(mfi);

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

            // DEBUG Tool
            if ((Ldot(0) != Ldot(0))
                or (Ldot(1) != Ldot(1))
                or (div_tau(0) != div_tau(0))
                or (div_tau(1) != div_tau(1)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Viscosity solving:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Ldot=", Ldot(0), ", ", Ldot(1));
                Util::ParallelMessage(INFO, "div_tau=", div_tau(0), ", ", div_tau(1));
                Util::Abort(INFO);
            }

            
            // ------------------------------------------------------------
            // Surface Tension
            // ------------------------------------------------------------
            // Fsv =  simga * kappa * n_hat
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
                    
                    Fsv_vector(0) = sigma_eff * kappa * grad_eta(0) * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
                    Fsv_vector(1) = sigma_eff * kappa * grad_eta(1) * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
                }
            }
            Fsv(i, j, k, 0) = Fsv_vector(0);
            Fsv(i, j, k, 1) = Fsv_vector(1);

            // DEBUG Tool
            if ((Fsv_vector(0) != Fsv_vector(0))
                or (Fsv_vector(1) != Fsv_vector(1)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Surface Tension solving:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Fsv_vector=", Fsv_vector(0), ", ", Fsv_vector(1));
                Util::Abort(INFO);
            }
            

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

            // DEBUG Tool
            if ((Fw_vector(0) != Fw_vector(0))
                or (Fw_vector(1) != Fw_vector(1)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Weight solving:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Fw_vector=", Fw_vector(0), ", ", Fw_vector(1));
                Util::Abort(INFO);
            }

            // ------------------------------------------------------------
            // Conservative Allen-Cahn
            // ------------------------------------------------------------
            // d(eta)/dt = -u·grad(eta) + Mob * laplacian(mu)
            // Laplacian of Chemical Potential (conservative form)
            /*
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            Set::Scalar phi = eta(i, j, k); // Dummy Variable bc it gets UGGGLLLYYYYY
            Set::Scalar kappa = kappas(i, j, k, 0);
            Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0]; // Mob = u_max * epsilon,  (epsilon = 0.7*DX) Chiu & Lin (2011)
            Set::Scalar advection = -u.dot(grad_eta);
            //Set::Scalar diffusion = Mob * lap_mu_chem; // Allen-Cahn Alternative - it has to be curve fit
            Set::Scalar diffusion = Mob * ( lap_eta - ((phi * (1.0-phi) * (1.0-2.0*phi)) / (epsilon*epsilon + small)) - (grad_eta_mag * kappa) ); // Chiu & Lin (2011) URL: https://www.sciencedirect.com/science/article/pii/S0021999110005243
            Set::Scalar eta_dot = advection + diffusion;
            */

            // ------------------------------------------------------------
            // Cahn–Hilliard
            // ------------------------------------------------------------
            // d(eta)/dt = -u·grad(eta) + div( M*grad(mu) )
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            //Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0]; // Mob = u_max * epsilon
            Set::Scalar Mob = 0.0;  // a(i, j, k) * epsilon * epsilon / DX[0]; // Mob = u_max * epsilon
            Set::Scalar advection = -u.dot(grad_eta);
            Set::Scalar diffusion = Mob * lap_mu_chem;
            Set::Scalar eta_dot = advection + diffusion;

            // DEBUG Tool
            if ((eta_dot != eta_dot))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Boundry Evolution:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "mu_chem_=", mu_chem_(i, j, k));
                Util::ParallelMessage(INFO, "lap_mu_chem=", lap_mu_chem);
                Util::ParallelMessage(INFO, "a=", a(i, j, k));
                Util::ParallelMessage(INFO, "Mob=", Mob);
                Util::ParallelMessage(INFO, "advection=", advection);
                Util::ParallelMessage(INFO, "diffusion=", diffusion);
                Util::ParallelMessage(INFO, "eta_dot=", eta_dot);
                Util::Abort(INFO);
            } 

            // ------------------------------------------------------------
            // Vaporization
            // ------------------------------------------------------------
            // Spalding Vaporization
            // NOTE: rho0 (OR eta = 1) SHOULD BE THE GAS PHASE
            Set::Scalar eta_dot_Vap = 0.0;
            Set::Scalar m_dot_Vap = 0.0;
            Set::Vector M_dot_Vap = Set::Vector(0.0, 0.0);
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
                Set::Scalar Y_vs = rho0(i, j, k) * eta(i, j, k) / (rho0(i, j, k) + rho1(i, j, k));

                // Spalding mass transfer number
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // Gas density from fluid 0 (eta=1 corresponds to fluid 0)
                Set::Scalar rho_g = rho0(i, j, k);

                // Scaling density choice: using rho_eta = rho_g makes RHS independent of mixture density
                // This is consistent with the document recommendation
                // Set::Scalar rho_eta = rho_g;
                Set::Scalar rho_eta = rho(i, j, k);//rho_g;

                // Vaporization source for eta equation (Equation 7):
                // source_vap = (1/epsilon) * (rho_g * D_v / rho_eta) * (B_M/(1+B_M)) * |grad(eta)|
                Set::Scalar vap_coeff = (rho_g * Dv / rho_eta + small) * (B_M / (1.0 + B_M + small));
                eta_dot_Vap = (1.0 / epsilon) * vap_coeff * grad_eta_mag;

                // FLUXES
                m_dot_Vap = rho_g * Dv * (B_M / (1.0 + B_M + small));   // Mass Flux
                M_dot_Vap = u * m_dot_Vap;                              // Momentum Flux
                E_dot_Vap = u.dot(M_dot_Vap);                           // Energy Flux

                
            }
            // Vaporization Trackers
            Vap_dot(i, j, k, 0) = eta_dot_Vap;
            Vap_dot(i, j, k, 1) = m_dot_Vap;
            Vap_dot(i, j, k, 2) = M_dot_Vap(0);
            Vap_dot(i, j, k, 3) = M_dot_Vap(1);
            Vap_dot(i, j, k, 4) = E_dot_Vap;


            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv_vector(0) + Fw_vector(0),
                                                  Fsv_vector(1) + Fw_vector(1));
            // DEBUG Tool
            if ((Total_Force(0) != Total_Force(0))
                or (Total_Force(1) != Total_Force(1)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Total Force:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Total_Force=", Total_Force(0), ", ", Total_Force(1));
                Util::Abort(INFO);
            }

            Source(i, j, k, 0) = mdot0 + m_dot_Vap;
            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0) + M_dot_Vap(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1) + M_dot_Vap(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force) + E_dot_Vap;

            // Lagrange terms to enforce no-penetration
            Source(i, j, k, 1) = Source(i, j, k, 1) - lagrange * u.dot(grad_eta) * grad_eta(0);
            Source(i, j, k, 2) = Source(i, j, k, 2) - lagrange * u.dot(grad_eta) * grad_eta(1);

            // DEBUG Tool
            if ((Source(i, j, k, 0) != Source(i, j, k, 0))
                or (Source(i, j, k, 1) != Source(i, j, k, 1))
                or (Source(i, j, k, 2) != Source(i, j, k, 2))
                or (Source(i, j, k, 3) != Source(i, j, k, 3)) )
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Total Source:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Source_rho=", Source(i, j, k, 0) );
                Util::ParallelMessage(INFO, "Source_M=", Source(i, j, k, 1), ", ", Source(i, j, k, 2));
                Util::ParallelMessage(INFO, "Source_E=", Source(i, j, k, 3) );
                Util::Abort(INFO);
            }

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

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            
            if (rho(i, j, k) != rho(i, j, k) || M(i, j, k, 0) != M(i, j, k, 0) || E(i, j, k) != E(i, j, k) || gammaf(i, j, k) != gammaf(i, j, k) || press(i, j, k) != press(i, j, k))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN CONSERVATIVE VARIABLES");
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "NaN detected at i=", i, " j=", j);
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k, 0), M(i, j, k, 1));
                Util::ParallelMessage(INFO, "E=", E(i, j, k));
                Util::ParallelMessage(INFO, "gamma=", gammaf(i, j, k));
                Util::ParallelMessage(INFO, "press=", press(i, j, k));
                Util::Abort(INFO);
            }

            if (rho(i - 1, j, k) != rho(i - 1, j, k) || rho(i + 1, j, k) != rho(i + 1, j, k) || rho(i, j - 1, k) != rho(i, j - 1, k) || rho(i, j + 1, k) != rho(i, j + 1, k))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR WITH GHOST CELLS");
                Util::ParallelMessage(INFO, "time=", time);
                Util::ParallelMessage(INFO, "NaN in neighbor at i=", i, " j=", j);
                Util::Abort(INFO);
            }

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

            // UPDATE MIXED FLUID VARIABLES
            rho_flux(i, j, k) = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]);
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]);
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]);
            E_flux(i, j, k) = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]);

            // Density
            rho_rhs(i, j, k) = rho_flux(i, j, k) + Source(i, j, k, 0);
            
            // Momentum
            M_rhs(i, j, k, 0) = M_flux(i, j, k, 0) + Source(i, j, k, 1); //(mu * (lap_ux * eta(i, j, k))) +
            M_rhs(i, j, k, 1) = M_flux(i, j, k, 1) + Source(i, j, k, 2); //(mu * (lap_uy * eta(i, j, k))) +

            // Energy
            E_rhs(i, j, k) = E_flux(i, j, k) + Source(i, j, k, 3);

            /// Eta:
            if (static_eta == 1)
            {
                eta_rhs(i, j, k) = 0.0;
            }
            else
            {
                eta_rhs(i, j, k) = eta_dot + eta_dot_Vap;
            }

            // ERROR CHECKING
            if ((rho_rhs(i, j, k) != rho_rhs(i, j, k))
                or (M_rhs(i, j, k, 0) != M_rhs(i, j, k, 0))
                or (M_rhs(i, j, k, 1) != M_rhs(i, j, k, 1))
                or (E_rhs(i, j, k) != E_rhs(i, j, k))
                or (eta_rhs(i, j, k) != eta_rhs(i, j, k)))
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
                Util::ParallelMessage(INFO, "drho/dt=", rho_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                Util::ParallelMessage(INFO, "deta/dt=", eta_rhs(i, j, k));
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

                Util::ParallelMessage(INFO, "drho/dt=", rho_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                Util::ParallelMessage(INFO, "deta/dt=", eta_rhs(i, j, k));
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

    // Swaping pointers
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    std::swap(eta_old_mf, eta_mf);

    // ------------------------------------------------------------
    // Time Integration
    // ------------------------------------------------------------
    // New Solution Refrences
    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*eta_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*density_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(), amrex::MakeType::make_alias, 0, 2);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);

    // Old Solution Refrences
    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*eta_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*density_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 2);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);

    // Create the time integrator
    amrex::TimeIntegrator timeintegrator(solution_new, time);

    // Set the time integrator RHS
    timeintegrator.set_rhs([&](
        amrex::Vector<amrex::MultiFab> &rhs_mf,
        amrex::Vector<amrex::MultiFab> &solution_mf, 
        const Set::Scalar time) {
            RHS(lev, time, rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3]);
    });

    // Filling boundaries during stages
    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        //eta_bc->FillBoundary(stage_mf[0], 0, 1, time, 0); // Might break since I don't have eta_bc specified directly
        energy_bc->FillBoundary(stage_mf[0], 0, 1, time, 0); // Might break since I don't have eta_bc specified directly
        stage_mf[0].FillBoundary(true);
        density_bc->FillBoundary(stage_mf[1], 0, 1, time, 0);
        stage_mf[1].FillBoundary(true);
        momentum_bc->FillBoundary(stage_mf[2], 0, 2, time, 0);
        stage_mf[2].FillBoundary(true);
        energy_bc->FillBoundary(stage_mf[3], 0, 1, time, 0);
        stage_mf[3].FillBoundary(true);
    });

    // Do the update
    timeintegrator.advance(solution_old, solution_new, time, dt);

    // ------------------------------------------------------------
    // Cutoffs and Visualization Fields
    // ------------------------------------------------------------
    // Dynamic Time Step Initialization
    c_max = 0.0;
    vx_max = 0.0;
    vy_max = 0.0;
    F_max = 0.0;
    rho_min = 1e10;
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Eta
        Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf.Patch(lev, mfi);

        // Mixture
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = energy_per_mas_mf.Patch(lev, mfi);
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


        //Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // CUTOFFS
            eta_new(i, j, k) = std::max( 0.0,std::min( 1.0,(eta_new(i, j, k) - cutoff) / (1.0 - 2.0 * cutoff) ) );
            rho(i, j, k) = std::max(rho(i, j, k), small);

            // Derivatice Function Calls
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // gamma
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            gammaf(i, j, k) = 1.0 + (1.0 / A);

            // etadot
            etadot(i, j, k) = (eta_new(i, j, k) - eta(i, j, k)) / dt;

            // Velocity = M ./ (DX*DY*rho)
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k));
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k));

            // Kinetic Energy
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            KE_mas(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            // Potential Energy
            UE_vol(i, j, k) = E_vol(i, j, k) - KE_vol(i, j, k);
            UE_mas(i, j, k) = E_mas(i, j, k) - KE_mas(i, j, k);

            // Pressure
            p0_eff(i, j, k) = (B / A) / gammaf(i, j, k);
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE_vol(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref; // pressure Tammann EOS modification

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Spalding Number
            Bm(i, j, k) = eta(i, j, k) / (1.0 - eta(i, j, k) + small);

            // Speed of sound:
            a(i, j, k) = sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / (a(i, j, k) + small);
            Ma(i, j, k, 1) = v(i, j, k, 1) / (a(i, j, k) + small);

            // Curvature
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

            // Debugging, would like to delete condition
            if (grad_eta_mag < 1e-4)
            {
                n_hat_(i, j, k, 0) = 0.0;
                n_hat_(i, j, k, 1) = 0.0;
            }
            else
            {
                n_hat_(i, j, k, 0) = n_hat(0);
                n_hat_(i, j, k, 1) = n_hat(1);
            }


            // ------------------------------------------------------------
            // Adaptive Timestepping
            // ------------------------------------------------------------
            // Solving for new states
            Set::Scalar A_new = (eta_new(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta_new(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B_new = (eta_new(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta_new(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            Set::Scalar gamma_eff_new = 1.0 + (1.0 / A_new);
            Set::Vector v_new = Set::Vector(M(i, j, k, 0) / (rho(i, j, k)), M(i, j, k, 1) / (rho(i, j, k)));
            Set::Scalar KE_vol_new = 0.5 * rho(i, j, k) * (v_new(0) * v_new(0) + v_new(1) * v_new(1));
            Set::Scalar UE_vol_new = E_vol(i, j, k) - KE_vol_new;
            Set::Scalar press_new = (UE_vol_new - B_new) / A_new;
            Set::Scalar p0_eff_new = (B_new / A_new) / gamma_eff_new;
            Set::Scalar sound_speed_new = sqrt(gamma_eff_new * (press_new + p0_eff_new) / (rho(i, j, k)));

            c_max = std::max(c_max, sound_speed_new);
            vx_max = std::max(vx_max, abs(v_new(0))); // vx
            vy_max = std::max(vy_max, abs(v_new(1))); // vy

            // Track maximum force magnitude (not acceleration yet)
            Set::Scalar F_mag = sqrt(Source(i, j, k, 1) * Source(i, j, k, 1) + Source(i, j, k, 2) * Source(i, j, k, 2));
            F_max = std::max(F_max, F_mag);
            rho_min = std::min(rho_min, rho(i, j, k));
                
        });
    }

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
    Set::Scalar wave_speed = c_max + sqrt(vx_max * vx_max + vy_max * vy_max);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    // 2. Viscous CFL
    Set::Scalar mu_max = std::max(mu0, mu1);
    Set::Scalar dt_viscous = cfl_v * rho_min * dx_min * dx_min / (mu_max + small);

    // 3. Force CFL
    Set::Scalar a_max = F_max / rho_min; // Maximum acceleration
    Set::Scalar dt_force = cfl_v * sqrt(dx_min / a_max);

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
