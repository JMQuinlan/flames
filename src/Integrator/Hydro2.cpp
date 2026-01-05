// Base
#include "Hydro2.H"

// Parsing and Input Handeling
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "IC/BMP.H"
#include "IC/Constant.H"
#include "IC/Expression.H"
#include "IC/Laminate.H"
#include "IC/PNG.H"
#include "IO/ParmParse.H"
#include "Numeric/Stencil.H"

// Solvers
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/Roe.H"
// #include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
// #include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
// #include "Solver/Local/FluidRiemann/HLLCE.H"
// #include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
// #include "Solver/Local/FluidRiemann/PartiallyParabolic.H"

// Limiters
// #include "Solver/Local/Limiter/Minmod.H"
// #include "Solver/Local/Limiter/VanLeer.H"

#if AMREX_SPACEDIM == 2

namespace Integrator
{

Hydro2::Hydro2(IO::ParmParse &pp) : Hydro2()
{
    pp.queryclass(*this);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////// PARSE ////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

void
Hydro2::Parse(Hydro2 &value, IO::ParmParse &pp)
{
    BL_PROFILE("Integrator::Hydro2::Hydro2()");
    {
        // REFINEMENT CRITERION
        pp.query_default("eta_refinement_criterion", value.eta_refinement_criterion, 0.001);    // eta-based refinement
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01); // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01); // velocity gradient-based refinement
        pp.query_default("p_refinement_criterion", value.p_refinement_criterion, 1e-3);         // pressure-based refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);     // density-based refinement

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);               // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl); // cfl condition
        pp_query_default("pref", value.pref, 0.0);         // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);    // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-8);  // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0); // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);           // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix"); // Roe solver entropy fix
        pp_query_default("scheme", value.scheme, 1);       // 1: Forward Euler | 2: RK2 | 3: RK3 | 4: RK4
        pp_query_default("Spec_Vol", value.Spec_Vol, 1);   // 0: Solve Energy via specific mass | 1: Solve Energy via specific volume

        // ADAPTIVE TIMESTEP
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
        pp_query_default("apply_vaporization", value.apply_vaporization, false);      // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // NEW SOLVER FORBIDS
        pp_forbid("gamma", "--> gamma0 and gamma1");
        pp_forbid("mu", "--> mu0 and mu1");

        // FLUID 0
        pp_query_required("gamma0", value.gamma0);   // gamma for gamma law
        pp_query_default("p0_0", value.p0_0, 0.0);   // p0 for Tammann EOS
        pp_query_required("mu0", value.mu0);         // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0); // bulk viscosity coefficient

        // FLUID 1
        pp_query_required("gamma1", value.gamma1);   // gamma for gamma law
        pp_query_default("p0_1", value.p0_1, 0.0);   // p0 for Tammann EOS
        pp_query_required("mu1", value.mu1);         // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0); // bulk viscosity coefficient

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0); // surface tension condition
        pp_query_default("Dv", value.Dv, 0.0);       // Vapor Diffusivity
        pp_query_required("epsilon", value.epsilon); // diffuse interface thickness

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 2); // Method to solve for curvature

        // Boundry Conditions
        pp_forbid("rho.bc", "--> density.bc");
        pp_forbid("p.bc", "--> pressure.bc");
        pp_forbid("v.bc", "--> velocity.bc");
        pp_forbid("pressure.bc", "--> energy.bc");
        pp_forbid("velocity.bc", "--> momentum.bc");

        value.density_bc = new BC::Expression(1, pp, "density.bc");
        value.energy_bc = new BC::Constant(1, pp, "energy.bc");
        value.momentum_bc = new BC::Expression(2, pp, "momentum.bc");
        value.temperature_bc = new BC::Constant(1, pp, "energy.bc"); // Change to be different if needed? ___TEMP___
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = 4;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf, value.energy_bc, 1, nghost, "eta", true);
        value.RegisterNewFab(value.eta_old_mf, value.energy_bc, 1, nghost, "eta_old", true);
        value.RegisterNewFab(value.etadot_mf, &value.bc_nothing, 1, nghost, "etadot", true);
        value.RegisterNewFab(value.hess_eta_mf, &value.bc_nothing, 4, nghost, "hess_eta", true, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf, &value.bc_nothing, 2, nghost, "n_hat", true, { "x", "y" });

        // FLUID 0
        value.RegisterNewFab(value.density0_mf, value.density_bc, 1, nghost, "density0", false);
        value.RegisterNewFab(value.density0_old_mf, value.density_bc, 1, nghost, "density0_old", false);
        value.RegisterNewFab(value.energy0_mf, value.energy_bc, 1, nghost, "energy0", false);
        value.RegisterNewFab(value.energy0_old_mf, value.energy_bc, 1, nghost, "energy0_old", false);
        value.RegisterNewFab(value.momentum0_mf, value.momentum_bc, 2, nghost, "momentum0", false, { "x", "y" });
        value.RegisterNewFab(value.momentum0_old_mf, value.momentum_bc, 2, nghost, "momentum0_old", false);
        value.RegisterNewFab(value.T0_mf, value.temperature_bc, 1, nghost, "T0", false);
        value.RegisterNewFab(value.cp0_mf, &value.bc_nothing, 1, nghost, "cp0", false);
        value.RegisterNewFab(value.cv0_mf, &value.bc_nothing, 1, nghost, "cv0", false);
        value.RegisterNewFab(value.k0_thermal_mf, &value.bc_nothing, 1, nghost, "k0_thermal", false);
        value.RegisterNewFab(value.h0_thermal_mf, &value.bc_nothing, 1, nghost, "h0_thermal", false);
        value.RegisterNewFab(value.pressure0_mf, value.energy_bc, 1, nghost, "pressure0", false);
        value.RegisterNewFab(value.velocity0_mf, &value.bc_nothing, 2, nghost, "velocity0", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity0_mf, &value.bc_nothing, 1, nghost, "vorticity0", false);

        // FLUID 1
        value.RegisterNewFab(value.density1_mf, value.density_bc, 1, nghost, "density1", false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc, 1, nghost, "density1_old", false);
        value.RegisterNewFab(value.energy1_mf, value.energy_bc, 1, nghost, "energy1", false);
        value.RegisterNewFab(value.energy1_old_mf, value.energy_bc, 1, nghost, "energy1_old", false);
        value.RegisterNewFab(value.momentum1_mf, value.momentum_bc, 2, nghost, "momentum1", false, { "x", "y" });
        value.RegisterNewFab(value.momentum1_old_mf, value.momentum_bc, 2, nghost, "momentum1_old", false);
        value.RegisterNewFab(value.T1_mf, value.temperature_bc, 1, nghost, "T1", false);
        value.RegisterNewFab(value.cp1_mf, &value.bc_nothing, 1, nghost, "cp1", false);
        value.RegisterNewFab(value.cv1_mf, &value.bc_nothing, 1, nghost, "cv1", false);
        value.RegisterNewFab(value.k1_thermal_mf, &value.bc_nothing, 1, nghost, "k1_thermal", false);
        value.RegisterNewFab(value.h1_thermal_mf, &value.bc_nothing, 1, nghost, "h1_thermal", false);
        value.RegisterNewFab(value.pressure1_mf, value.energy_bc, 1, nghost, "pressure1", false);
        value.RegisterNewFab(value.velocity1_mf, &value.bc_nothing, 2, nghost, "velocity1", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity1_mf, &value.bc_nothing, 1, nghost, "vorticity1", false);

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf, value.energy_bc, 1, nghost, "pressure", true);
        value.RegisterNewFab(value.velocity_mf, &value.bc_nothing, 2, nghost, "velocity", true, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf, &value.bc_nothing, 1, nghost, "vorticity", true);
        value.RegisterNewFab(value.density_mf, value.density_bc, 1, nghost, "density", true);
        value.RegisterNewFab(value.density_old_mf, value.density_bc, 1, nghost, "density_old", false);
        value.RegisterNewFab(value.energy_per_vol_mf, value.energy_bc, 1, nghost, "energy_per_vol", true);
        value.RegisterNewFab(value.energy_per_mas_mf, value.energy_bc, 1, nghost, "energy_per_mass", true);
        value.RegisterNewFab(value.energy_per_vol_old_mf, value.energy_bc, 1, nghost, "energy_vol_old", false);
        value.RegisterNewFab(value.energy_per_mas_old_mf, value.energy_bc, 1, nghost, "energy_mas_old", false);
        value.RegisterNewFab(value.momentum_mf, value.momentum_bc, 2, nghost, "momentum", true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc, 2, nghost, "momentum_old", false, { "x", "y" });

        // SOURCES
        value.RegisterNewFab(value.m0_mf, &value.bc_nothing, 1, nghost, "m0", true);
        value.RegisterNewFab(value.u0_mf, &value.bc_nothing, 2, nghost, "u0", true, { "x", "y" });
        value.RegisterNewFab(value.q_mf, &value.bc_nothing, 2, nghost, "q0", true, { "x", "y" });
        value.RegisterNewFab(value.Source_mf, &value.bc_nothing, 4, nghost, "Source", true);
        value.RegisterNewFab(value.Fsv_mf, &value.bc_nothing, 2, nghost, "Fsv", true, { "x", "y" });   // Surface Tension
        value.RegisterNewFab(value.Fb_mf, &value.bc_nothing, 2, nghost, "Fb", true, { "x", "y" });     // Buoyancy
        value.RegisterNewFab(value.Fw_mf, &value.bc_nothing, 2, nghost, "Fw", true, { "x", "y" });     // Weight
        value.RegisterNewFab(value.tau_xx_mf, value.density_bc, 1, nghost, "tau_xx", true, { "xx" });  // Stress Tensor
        value.RegisterNewFab(value.tau_xy_mf, value.density_bc, 1, nghost, "tau_xy", true, { "xy" });  // Stress Tensor
        value.RegisterNewFab(value.tau_yy_mf, value.density_bc, 1, nghost, "tau_yy", true, { "yy" });  // Stress Tensor
        value.RegisterNewFab(value.Ldot_mf, &value.bc_nothing, 2, nghost, "Ldot", true, { "x", "y" }); // Ldot
        value.RegisterNewFab(value.T_mf, &value.bc_nothing, 1, nghost, "T", true);                     // Temperature
        value.RegisterNewFab(value.cp_mf, &value.bc_nothing, 1, nghost, "cp", false);                  // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf, &value.bc_nothing, 1, nghost, "cv", false);                  // Constant Volume Specific Heat
        value.RegisterNewFab(value.k_thermal_mf, &value.bc_nothing, 1, nghost, "k_thermal", false);    // Thermal Conductivity
        value.RegisterNewFab(value.h_thermal_mf, &value.bc_nothing, 1, nghost, "h_thermal", false);    // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf, value.energy_bc, 1, nghost, "gamma", true);               // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf, value.energy_bc, 1, nghost, "p0", true);                     // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf, value.energy_bc, 1, nghost, "mu_chem", true);           // Tammann Pressure
        value.RegisterNewFab(value.a_mf, &value.bc_nothing, 1, nghost, "a", true);                     // Speed of sound
        value.RegisterNewFab(value.Ma_mf, &value.bc_nothing, 2, nghost, "Ma", true, { "x", "y" });     // Mach
        value.RegisterNewFab(value.UE_per_vol_mf, &value.bc_nothing, 1, nghost, "UE_per_vol", true);   // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf, &value.bc_nothing, 1, nghost, "UE_per_mass", true);  // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf, &value.bc_nothing, 1, nghost, "KE_per_vol", true);   // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf, &value.bc_nothing, 1, nghost, "KE_per_mass", true);  // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf, &value.bc_nothing, 1, nghost, "Spadling_Number", true);      // Spalding Number

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_eta", true, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf, &value.bc_nothing, 3, nghost, "kappa", true, { "Avg", "1", "2" });                 // To Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", true, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf, &value.bc_nothing, 1, nghost, "rho_flux", true);                                 // Density Flux
        value.RegisterNewFab(value.M_flux_mf, &value.bc_nothing, 2, nghost, "M_flux", true, { "x", "y" });                       // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf, &value.bc_nothing, 1, nghost, "E_flux", true);                                     // Energy Flux
        value.RegisterNewFab(value.div_tau_mf, &value.bc_nothing, 2, nghost, "div_tau", true, { "x", "y" });                     // Energy Flux
        value.RegisterNewFab(value.hess_u_mf, &value.bc_nothing, 8, nghost, "hess_u", true, {
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
    pp.select_default<IC::Constant, IC::Laminate, IC::Expression, IC::BMP, IC::PNG>("eta.ic", value.eta_ic, value.geom);

    // Fluid 0
    pp.select_default<IC::Constant, IC::Expression>("velocity0.ic", value.velocity0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("pressure0.ic", value.pressure0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("density0.ic", value.density0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("temperature0.ic", value.temperature0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cp0.ic", value.cp0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cv0.ic", value.cv0_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("k0_thermal.ic", value.k0_thermal_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic", value.h0_thermal_ic, value.geom);

    // Fluid 1
    pp.select_default<IC::Constant, IC::Expression>("velocity1.ic", value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("pressure1.ic", value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("density1.ic", value.density1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("temperature1.ic", value.temperature1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cp1.ic", value.cp1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("cv1.ic", value.cv1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("k1_thermal.ic", value.k1_thermal_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic", value.h1_thermal_ic, value.geom);

    // DIFFUSE BOUNDARY SOURCES
    pp.select_default<IC::Constant, IC::Expression>("m0.ic", value.ic_m0, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("u0.ic", value.ic_u0, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("q.ic", value.ic_q, value.geom);

    // SOLVERS
    // Riemann solver
    pp_query_default("Riemann_Solver", value.Riemann_Solver, 0); // Type of solver

    if (value.Riemann_Solver == 0)
    {
        pp.select_default<Solver::Local::FluidRiemann::Roe>("solver", value.roesolver);
    }
    else if (value.Riemann_Solver == 1)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLC>("solver", value.hllcsolver);
    }
    else if (value.Riemann_Solver == 2)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLE>("solver", value.hllesolver);
    }
    else if (value.Riemann_Solver == 37)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLC_All_Mach>("solver", value.hllc_All_Machsolver);
    }
    else if (value.Riemann_Solver == 38)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro>("solver", value.hllc_All_Mach_Furfarosolver);
    }
    else
    {
        Util::ParallelMessage(INFO, "-------------------------------");
        Util::ParallelMessage(INFO, "Invalid solver method: ", value.Riemann_Solver);
        Util::ParallelMessage(INFO, "Acceptable Methods: ");
        Util::ParallelMessage(INFO, "Roe        : 0");
        Util::ParallelMessage(INFO, "HLLC       : 1");
        Util::ParallelMessage(INFO, "HLLE       : 2");
        Util::ParallelMessage(INFO, "Under Testing:");
        Util::ParallelMessage(INFO, "HLLC_All_Mach : 37");
        Util::ParallelMessage(INFO, "HLLC_All_Mach_Furfaro : 38");
        Util::Exception(INFO);
    }

    // LIMITERS
    pp_query_default("Limiter", value.Limiter, 0); // Type of solver
    if (value.Limiter != 0)
    {
        Util::ParallelMessage(INFO, "-------------------------------");
        Util::ParallelMessage(INFO, "Invalid Limiter: ", value.Limiter);
        Util::ParallelMessage(INFO, "Acceptable Methods: ");
        Util::ParallelMessage(INFO, "None       : 0");
        Util::Exception(INFO);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");

    // Initialize individual fluid variables
    // DIFFUSIVE BOUNDRY
    eta_ic->Initialize(lev, eta_mf, 0.0);
    eta_ic->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]->setVal(0.0);

    // FLUID 0
    velocity0_ic->Initialize(lev, velocity0_mf, 0.0);
    pressure0_ic->Initialize(lev, pressure0_mf, 0.0);
    density0_ic->Initialize(lev, density0_mf, 0.0);
    density0_ic->Initialize(lev, density0_old_mf, 0.0);
    temperature0_ic->Initialize(lev, T0_mf, 0.0);
    cp0_ic->Initialize(lev, cp0_mf, 0.0);
    cv0_ic->Initialize(lev, cv0_mf, 0.0);
    k0_thermal_ic->Initialize(lev, k0_thermal_mf, 0.0);
    h0_thermal_ic->Initialize(lev, h0_thermal_mf, 0.0);

    // FLUID 1
    velocity1_ic->Initialize(lev, velocity1_mf, 0.0);
    pressure1_ic->Initialize(lev, pressure1_mf, 0.0);
    density1_ic->Initialize(lev, density1_mf, 0.0);
    density1_ic->Initialize(lev, density1_old_mf, 0.0);
    temperature1_ic->Initialize(lev, T1_mf, 0.0);
    cp1_ic->Initialize(lev, cp1_mf, 0.0);
    cv1_ic->Initialize(lev, cv1_mf, 0.0);
    k1_thermal_ic->Initialize(lev, k1_thermal_mf, 0.0);
    h1_thermal_ic->Initialize(lev, h1_thermal_mf, 0.0);

    // FORCED SOURCE
    ic_m0->Initialize(lev, m0_mf, 0.0);
    ic_u0->Initialize(lev, u0_mf, 0.0);
    q_mf[lev]->setVal(0.0);

    // NATURAL SOURCE
    Source_mf[lev]->setVal(0.0);
    Fsv_mf[lev]->setVal(0.0);
    Fb_mf[lev]->setVal(0.0);
    Fw_mf[lev]->setVal(0.0);
    tau_xx_mf[lev]->setVal(0.0);
    tau_xy_mf[lev]->setVal(0.0);
    tau_yy_mf[lev]->setVal(0.0);
    Ldot_mf[lev]->setVal(0.0);

    // BOUNDRY CURVATURE AND THINGS
    kappas_mf[lev]->setVal(0.0);
    grad_mag_grad_eta_mf[lev]->setVal(0.0);
    Bm_mf[lev]->setVal(0.0); // Spalding Number

    // MIXED PROPERTIES
    a_mf[lev]->setVal(0.0);
    Ma_mf[lev]->setVal(0.0);
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
void
Hydro2::Mix(int lev)
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
        Set::Patch<const Set::Scalar> v0 = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0 = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho0_old = density0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M0 = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M0_old = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E0 = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E0_old = energy0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T0 = T0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cp0 = cp0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cv0 = cv0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> k0_thermal = k0_thermal_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> h0_thermal = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar> v1 = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p1 = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho1_old = density1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M1 = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M1_old = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1 = energy1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1_old = energy1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T1 = T1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cp1 = cp1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cv1 = cv1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> k1_thermal = k1_thermal_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> h1_thermal = h1_thermal_mf.Patch(lev, mfi);

        // MIXTURE
        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M = momentum_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_mas = energy_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_vol_old = energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_mas_old = energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> T = T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar> a = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ma = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_vol = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_mas = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_vol = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_mas = KE_per_mas_mf.Patch(lev, mfi);

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

            UE_vol(i, j, k) = (p_eff + pref) * A + B;
            UE_mas(i, j, k) = (UE_vol(i, j, k)) / (rho(i, j, k));

            // Total Energy
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
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Spalding Number
            Bm(i, j, k) = eta(i, j, k) / (1.0 - eta(i, j, k) + small);

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
            a(i, j, k) = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

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

void
Hydro2::TimeStepBegin(Set::Scalar, int /*iter*/)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// TIMESTEPCOMPLETE ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

void
Hydro2::TimeStepComplete(Set::Scalar time, int lev)
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
            amrex::MultiFab &rho_rhs_mf,
            amrex::MultiFab &M_rhs_mf,
            amrex::MultiFab &E_rhs_mf,
            amrex::MultiFab &eta_rhs_mf,
            const amrex::MultiFab &rho_mf_in,
            const amrex::MultiFab &M_mf_in,
            const amrex::MultiFab &E_mf_in,
            const amrex::MultiFab &eta_mf_in,
            Set::Scalar dt,
            const Set::Scalar *DX)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    amrex::Box domain = geom[lev].Domain();

    // Create temporary MultiFabs for primitive and derived quantities
    int nghost = 4;
    amrex::MultiFab velocity_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 2, nghost);
    amrex::MultiFab pressure_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab gamma_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab p0_eff_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab a_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab T_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab grad_eta_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 2, nghost);
    amrex::MultiFab hess_eta_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 4, nghost);
    amrex::MultiFab n_hat_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 2, nghost);
    amrex::MultiFab kappa_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab mu_chem_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab tau_xx_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab tau_xy_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab tau_yy_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 1, nghost);
    amrex::MultiFab Ldot_tmp(rho_mf_in.boxArray(), rho_mf_in.DistributionMap(), 2, nghost);

    // Initialize RHS to zero
    rho_rhs_mf.setVal(0.0);
    M_rhs_mf.setVal(0.0);
    E_rhs_mf.setVal(0.0);
    eta_rhs_mf.setVal(0.0);

    // Fill boundaries on input state
    const_cast<amrex::MultiFab &>(rho_mf_in).FillBoundary(geom[lev].periodicity());
    const_cast<amrex::MultiFab &>(M_mf_in).FillBoundary(geom[lev].periodicity());
    const_cast<amrex::MultiFab &>(E_mf_in).FillBoundary(geom[lev].periodicity());
    const_cast<amrex::MultiFab &>(eta_mf_in).FillBoundary(geom[lev].periodicity());

    // ============================================================================
    // LOOP 1: Calculate primitive and derived variables
    // ============================================================================
    for (amrex::MFIter mfi(rho_mf_in, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<const Set::Scalar> const &rho = rho_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &M = M_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &E_vol = E_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &eta = eta_mf_in.array(mfi);

        amrex::Array4<Set::Scalar> const &v = velocity_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &press = pressure_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &gammaf = gamma_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &p0_eff = p0_eff_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &a = a_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &T = T_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &grad_eta = grad_eta_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &hess_eta = hess_eta_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &n_hat = n_hat_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &kappa = kappa_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &mu_chem = mu_chem_tmp.array(mfi);

        amrex::Array4<const Set::Scalar> const &m0 = (*m0_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const &u0 = (*u0_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const &q0 = (*q_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Velocity
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k) + small);
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k) + small);

            // Kinetic Energy
            Set::Scalar KE_vol = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            // Internal Energy
            Set::Scalar UE_vol = E_vol(i, j, k) - KE_vol;

            // Gamma and pressure
            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);

            gammaf(i, j, k) = 1.0 + (1.0 / A);
            p0_eff(i, j, k) = (B / A) / gammaf(i, j, k);
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE_vol - gammaf(i, j, k) * p0_eff(i, j, k) + pref;

            // Temperature (placeholder - adjust based on your EOS)
            T(i, j, k) = press(i, j, k) / (rho(i, j, k) * R + small);

            // Speed of sound
            a(i, j, k) = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k) + small));

            // Eta gradients and curvature
            Set::Vector grad_eta_vec = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta_vec.lpNorm<2>();
            Set::Matrix hess_eta_mat = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            grad_eta(i, j, k, 0) = grad_eta_vec(0);
            grad_eta(i, j, k, 1) = grad_eta_vec(1);

            hess_eta(i, j, k, 0) = hess_eta_mat(0, 0);
            hess_eta(i, j, k, 1) = hess_eta_mat(0, 1);
            hess_eta(i, j, k, 2) = hess_eta_mat(1, 0);
            hess_eta(i, j, k, 3) = hess_eta_mat(1, 1);

            // Normal vector
            Set::Vector n_hat_vec = grad_eta_vec / (grad_eta_mag + small);
            n_hat(i, j, k, 0) = n_hat_vec(0);
            n_hat(i, j, k, 1) = n_hat_vec(1);

            // Curvature
            if (kappa_method == 2 && grad_eta_mag > 0.01)
            {
                Set::Vector t1;
                if (std::abs(n_hat_vec(0)) > std::abs(n_hat_vec(1)))
                {
                    t1 = Set::Vector(-n_hat_vec(1), n_hat_vec(0)) / std::sqrt(n_hat_vec(0) * n_hat_vec(0) + n_hat_vec(1) * n_hat_vec(1) + small);
                }
                else
                {
                    t1 = Set::Vector(n_hat_vec(1), -n_hat_vec(0)) / std::sqrt(n_hat_vec(0) * n_hat_vec(0) + n_hat_vec(1) * n_hat_vec(1) + small);
                }

                Set::Scalar kappa1 = -n_hat_vec.dot(hess_eta_mat * n_hat_vec);
                Set::Scalar kappa2 = -t1.dot(hess_eta_mat * t1) * 2.0 * epsilon;
                kappa(i, j, k) = kappa2;
            }
            else
            {
                kappa(i, j, k) = 0.0;
            }

            // Chemical potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k));
            mu_chem(i, j, k) = -epsilon * epsilon * lap_eta + f_prime;
        });
    }

    // Fill boundaries on temporary fields
    velocity_tmp.FillBoundary(geom[lev].periodicity());
    pressure_tmp.FillBoundary(geom[lev].periodicity());
    gamma_tmp.FillBoundary(geom[lev].periodicity());
    p0_eff_tmp.FillBoundary(geom[lev].periodicity());
    a_tmp.FillBoundary(geom[lev].periodicity());
    T_tmp.FillBoundary(geom[lev].periodicity());
    grad_eta_tmp.FillBoundary(geom[lev].periodicity());
    hess_eta_tmp.FillBoundary(geom[lev].periodicity());
    n_hat_tmp.FillBoundary(geom[lev].periodicity());
    kappa_tmp.FillBoundary(geom[lev].periodicity());
    mu_chem_tmp.FillBoundary(geom[lev].periodicity());

    // ============================================================================
    // LOOP 2: Calculate stress tensor and viscous terms
    // ============================================================================
    for (amrex::MFIter mfi(rho_mf_in, true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        amrex::Array4<const Set::Scalar> const &rho = rho_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &M = M_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &eta = eta_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &v = velocity_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &grad_eta = grad_eta_tmp.array(mfi);

        amrex::Array4<Set::Scalar> const &tau_xx = tau_xx_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &tau_xy = tau_xy_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &tau_yy = tau_yy_tmp.array(mfi);
        amrex::Array4<Set::Scalar> const &Ldot = Ldot_tmp.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Velocity gradient
            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k) + small);

            // Strain rate tensor
            Set::Matrix eps = Set::Matrix::Zero();
            Set::Scalar div_u = gradu(0, 0) + gradu(1, 1);

            for (int p = 0; p < 2; ++p)
            {
                for (int q = 0; q < 2; ++q)
                {
                    eps(p, q) = 0.5 * (gradu(p, q) + gradu(q, p));
                }
            }

            // Effective viscosities
            Set::Scalar mu_eff = eta(i, j, k) * mu0 + (1.0 - eta(i, j, k)) * mu1;
            Set::Scalar lambda_eff = eta(i, j, k) * mu0_b + (1.0 - eta(i, j, k)) * mu1_b;
            Set::Vector grad_mu = (mu0 - mu1) * Set::Vector(grad_eta(i, j, k, 0), grad_eta(i, j, k, 1));
            Set::Vector grad_lambda = (mu0_b - mu1_b) * Set::Vector(grad_eta(i, j, k, 0), grad_eta(i, j, k, 1));

            // Stress tensor
            Set::Matrix tau = Set::Matrix::Zero();
            for (int p = 0; p < 2; ++p)
            {
                for (int q = 0; q < 2; ++q)
                {
                    tau(p, q) = 2.0 * mu_eff * eps(p, q) + lambda_eff * div_u * (p == q);
                }
            }

            tau_xx(i, j, k) = tau(0, 0);
            tau_xy(i, j, k) = tau(0, 1);
            tau_yy(i, j, k) = tau(1, 1);

            // Ldot term (viscosity gradient coupling)
            Set::Vector Ldot_vec = Set::Vector::Zero();
            for (int p = 0; p < 2; ++p)
            {
                for (int q = 0; q < 2; ++q)
                {
                    Ldot_vec(p) += grad_mu(q) * (gradu(p, q) + gradu(q, p));
                }
                Ldot_vec(p) += grad_lambda(p) * div_u;
            }

            Ldot(i, j, k, 0) = Ldot_vec(0);
            Ldot(i, j, k, 1) = Ldot_vec(1);
        });
    }

    // Fill boundaries on stress tensor fields
    tau_xx_tmp.FillBoundary(geom[lev].periodicity());
    tau_xy_tmp.FillBoundary(geom[lev].periodicity());
    tau_yy_tmp.FillBoundary(geom[lev].periodicity());
    Ldot_tmp.FillBoundary(geom[lev].periodicity());

    // ============================================================================
    // LOOP 3: Calculate RHS (fluxes + sources)
    // ============================================================================
    for (amrex::MFIter mfi(rho_mf_in, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        amrex::Array4<const Set::Scalar> const &rho = rho_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &M = M_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &E_vol = E_mf_in.array(mfi);
        amrex::Array4<const Set::Scalar> const &eta = eta_mf_in.array(mfi);

        amrex::Array4<const Set::Scalar> const &v = velocity_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &press = pressure_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &gammaf = gamma_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &p0_eff = p0_eff_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &T = T_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &grad_eta = grad_eta_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &n_hat = n_hat_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &kappa = kappa_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &mu_chem = mu_chem_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &tau_xx = tau_xx_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &tau_xy = tau_xy_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &tau_yy = tau_yy_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &Ldot = Ldot_tmp.array(mfi);
        amrex::Array4<const Set::Scalar> const &a = a_tmp.array(mfi); // ADD THIS

        amrex::Array4<const Set::Scalar> const &m0 = (*m0_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const &u0 = (*u0_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const &q0 = (*q_mf[lev]).array(mfi);

        amrex::Array4<Set::Scalar> const &rho_rhs = rho_rhs_mf.array(mfi);
        amrex::Array4<Set::Scalar> const &M_rhs = M_rhs_mf.array(mfi);
        amrex::Array4<Set::Scalar> const &E_rhs = E_rhs_mf.array(mfi);
        amrex::Array4<Set::Scalar> const &eta_rhs = eta_rhs_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const int X = 0, Y = 1;

            // ============================================================
            // RIEMANN SOLVER FLUXES
            // ============================================================

            // Create cell states for reconstruction
            std::vector<Solver::Local::FluidRiemann::State> x_states(3);
            std::vector<Solver::Local::FluidRiemann::State> y_states(3);

            x_states[0] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i - 1, j, k, X);
            x_states[1] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, X);
            x_states[2] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i + 1, j, k, X);

            y_states[0] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j - 1, k, Y);
            y_states[1] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, Y);
            y_states[2] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j + 1, k, Y);

            // No limiter - use cell-centered values
            std::vector<Solver::Local::FluidRiemann::State> x_leftStates(3), x_rightStates(3);
            std::vector<Solver::Local::FluidRiemann::State> y_leftStates(3), y_rightStates(3);

            x_leftStates[1] = x_states[0];
            x_rightStates[1] = x_states[1];
            x_leftStates[2] = x_states[1];
            x_rightStates[2] = x_states[2];

            y_leftStates[1] = y_states[0];
            y_rightStates[1] = y_states[1];
            y_leftStates[2] = y_states[1];
            y_rightStates[2] = y_states[2];

            // Calculate fluxes
            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;

            if (Riemann_Solver == 0)
            {
                flux_xlo = roesolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = roesolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = roesolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = roesolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            else if (Riemann_Solver == 1)
            {
                flux_xlo = hllcsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = hllcsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = hllcsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = hllcsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            else if (Riemann_Solver == 2)
            {
                flux_xlo = hllesolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = hllesolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = hllesolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = hllesolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            else if (Riemann_Solver == 37)
            {
                flux_xlo = hllc_All_Machsolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = hllc_All_Machsolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = hllc_All_Machsolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = hllc_All_Machsolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }
            else if (Riemann_Solver == 38)
            {
                flux_xlo = hllc_All_Mach_Furfarosolver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                flux_ylo = hllc_All_Mach_Furfarosolver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                flux_xhi = hllc_All_Mach_Furfarosolver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                flux_yhi = hllc_All_Mach_Furfarosolver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);
            }

            // ============================================================
            // SOURCE TERMS
            // ============================================================

            Set::Vector grad_eta_vec = Set::Vector(grad_eta(i, j, k, 0), grad_eta(i, j, k, 1));
            Set::Scalar grad_eta_mag = grad_eta_vec.lpNorm<2>();
            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Vector u0_vec = Set::Vector(u0(i, j, k, 0), u0(i, j, k, 1));
            Set::Vector q0_vec = Set::Vector(q0(i, j, k, 0), q0(i, j, k, 1));

            // Diffuse boundary sources
            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0_vec.dot(grad_eta_vec);

            // Divergence of stress tensor
            Set::Matrix grad_tau_xx = Numeric::Gradient(tau_xx, i, j, k, DX);
            Set::Matrix grad_tau_xy = Numeric::Gradient(tau_xy, i, j, k, DX);
            Set::Matrix grad_tau_yy = Numeric::Gradient(tau_yy, i, j, k, DX);

            Set::Vector div_tau = Set::Vector::Zero();
            div_tau(0) = grad_tau_xx(0, 0) + grad_tau_xy(0, 1);
            div_tau(1) = grad_tau_xy(1, 0) + grad_tau_yy(1, 1);

            Set::Vector Ldot_vec = Set::Vector(Ldot(i, j, k, 0), Ldot(i, j, k, 1));

            // Surface tension
            Set::Vector Fsv_vector = Set::Vector::Zero();
            if (apply_surface_tension && grad_eta_mag > 0.01)
            {
                Set::Scalar sigma_eff = sigma;
                Set::Scalar alpha = 6 * std::sqrt(2);
                Set::Scalar UFFDA = epsilon * alpha * grad_eta_mag * grad_eta_mag;
                Set::Vector n_hat_vec = Set::Vector(n_hat(i, j, k, 0), n_hat(i, j, k, 1));
                Fsv_vector = sigma_eff * kappa(i, j, k) * n_hat_vec * UFFDA;
            }

            // Weight
            Set::Vector Fw_vector = Set::Vector::Zero();
            if (apply_buoyancy)
            {
                Fw_vector(1) = -rho(i, j, k) * g;
            }

            // Buoyancy (placeholder)
            Set::Vector Fb_vector = Set::Vector::Zero();

            // Total force
            Set::Vector Total_Force = Fsv_vector + Fb_vector + Fw_vector;

            // ============================================================
            // ASSEMBLE RHS
            // ============================================================

            // Density RHS
            Set::Scalar rho_flux = (flux_xlo.mass - flux_xhi.mass) / DX[0] + (flux_ylo.mass - flux_yhi.mass) / DX[1];
            rho_rhs(i, j, k) = rho_flux + mdot0;

            // Momentum RHS
            Set::Scalar Mx_flux = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / DX[0] + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / DX[1];
            Set::Scalar My_flux = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / DX[0] + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / DX[1];

            M_rhs(i, j, k, 0) = Mx_flux + Pdot0(0) + Ldot_vec(0) + div_tau(0) + Total_Force(0);
            M_rhs(i, j, k, 1) = My_flux + Pdot0(1) + Ldot_vec(1) + div_tau(1) + Total_Force(1);

            // Lagrange no-penetration
            M_rhs(i, j, k, 0) -= lagrange * u.dot(grad_eta_vec) * grad_eta_vec(0);
            M_rhs(i, j, k, 1) -= lagrange * u.dot(grad_eta_vec) * grad_eta_vec(1);

            // Energy RHS
            Set::Scalar E_flux = (flux_xlo.energy - flux_xhi.energy) / DX[0] + (flux_ylo.energy - flux_yhi.energy) / DX[1];
            E_rhs(i, j, k) = E_flux + qdot0 + u.dot(div_tau) + u.dot(Ldot_vec) + u.dot(Total_Force);

            // Eta RHS (Allen-Cahn)
            if (static_eta == 1)
            {
                eta_rhs(i, j, k) = 0.0;
            }
            else
            {
                Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0];
                Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem, i, j, k, 0, DX);
                Set::Scalar advection = -u.dot(grad_eta_vec);
                Set::Scalar diffusion = 0.0; // Mob * lap_mu_chem; // Can enable if needed

                eta_rhs(i, j, k) = advection + diffusion;

                // Vaporization (Spalding)
                if (apply_vaporization == 1)
                {
                    Set::Scalar Bm = eta(i, j, k) / (1.0 - eta(i, j, k) + small);
                    Set::Scalar rho0_local = eta(i, j, k) * rho(i, j, k); // Approximate
                    eta_rhs(i, j, k) += (1.0 / (rho(i, j, k) * epsilon + small)) * (rho0_local * Dv * (Bm / (1.0 + Bm + small)) * grad_eta_mag);
                }
            }
        });
    }

    amrex::Gpu::synchronize();
    amrex::ParallelDescriptor::Barrier();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

void
Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::Advance");

    // Swapping pointers to save old state
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    std::swap(eta_old_mf[lev], eta_mf[lev]);

    // Get geometry
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Reset dynamic timestep tracking variables
    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();
    c_max = 0.0;
    vx_max = 0.0;
    vy_max = 0.0;
    F_max = 0.0;
    rho_min = 1e10;

    // Handle boundary conditions for source terms
    m0_mf[lev]->FillBoundary(geom[lev].periodicity());
    u0_mf[lev]->FillBoundary(geom[lev].periodicity());
    q_mf[lev]->FillBoundary(geom[lev].periodicity());

    // ============================================================================
    // TIME INTEGRATION SCHEMES
    // ============================================================================

    if (scheme == 1)
    {
        // ========================================
        // FORWARD EULER
        // ========================================

        int nghost = 4;
        amrex::MultiFab k1_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k1_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        // Compute k1 = RHS(old state)
        RHS(lev, time, k1_rho, k1_M, k1_E, k1_eta, *density_old_mf[lev], *momentum_old_mf[lev], *energy_per_vol_old_mf[lev], *eta_old_mf[lev], dt, DX);

        // Update: new = old + dt * k1
        amrex::MultiFab::LinComb(*density_mf[lev], 1.0, *density_old_mf[lev], 0, dt, k1_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(*momentum_mf[lev], 1.0, *momentum_old_mf[lev], 0, dt, k1_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(*energy_per_vol_mf[lev], 1.0, *energy_per_vol_old_mf[lev], 0, dt, k1_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(*eta_mf[lev], 1.0, *eta_old_mf[lev], 0, dt, k1_eta, 0, 0, 1, nghost);

        // Fill boundaries on new state
        density_mf[lev]->FillBoundary(geom[lev].periodicity());
        momentum_mf[lev]->FillBoundary(geom[lev].periodicity());
        energy_per_vol_mf[lev]->FillBoundary(geom[lev].periodicity());
        eta_mf[lev]->FillBoundary(geom[lev].periodicity());
    }
    else if (scheme == 2)
    {
        // ========================================
        // RK2 (Heun's method)
        // ========================================

        int nghost = 4;
        amrex::MultiFab k1_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k1_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k2_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k2_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab rho_tmp(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab M_tmp(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab E_tmp(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab eta_tmp(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        // Stage 1: k1 = RHS(old)
        RHS(lev, time, k1_rho, k1_M, k1_E, k1_eta, *density_old_mf[lev], *momentum_old_mf[lev], *energy_per_vol_old_mf[lev], *eta_old_mf[lev], dt, DX);

        // Intermediate state: tmp = old + dt * k1
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, dt, k1_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, dt, k1_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, dt, k1_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, dt, k1_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 2: k2 = RHS(tmp)
        RHS(lev, time + dt, k2_rho, k2_M, k2_E, k2_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Final update: new = old + dt/2 * (k1 + k2)
        for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();

            amrex::Array4<Set::Scalar> const &rho_new = density_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &M_new = momentum_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &E_new = energy_per_vol_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &eta_new = eta_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &rho_old = density_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &M_old = momentum_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &E_old = energy_per_vol_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &eta_old = eta_old_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &k1_rho_arr = k1_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_M_arr = k1_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_E_arr = k1_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_eta_arr = k1_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k2_rho_arr = k2_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_M_arr = k2_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_E_arr = k2_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_eta_arr = k2_eta.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new(i, j, k) = rho_old(i, j, k) + 0.5 * dt * (k1_rho_arr(i, j, k) + k2_rho_arr(i, j, k));
                M_new(i, j, k, 0) = M_old(i, j, k, 0) + 0.5 * dt * (k1_M_arr(i, j, k, 0) + k2_M_arr(i, j, k, 0));
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + 0.5 * dt * (k1_M_arr(i, j, k, 1) + k2_M_arr(i, j, k, 1));
                E_new(i, j, k) = E_old(i, j, k) + 0.5 * dt * (k1_E_arr(i, j, k) + k2_E_arr(i, j, k));
                eta_new(i, j, k) = eta_old(i, j, k) + 0.5 * dt * (k1_eta_arr(i, j, k) + k2_eta_arr(i, j, k));

                // Apply eta cutoff
                if (eta_new(i, j, k) <= cutoff)
                    eta_new(i, j, k) = 0.0;
                else if (eta_new(i, j, k) >= (1.0 - cutoff))
                    eta_new(i, j, k) = 1.0;
            });
        }

        density_mf[lev]->FillBoundary(geom[lev].periodicity());
        momentum_mf[lev]->FillBoundary(geom[lev].periodicity());
        energy_per_vol_mf[lev]->FillBoundary(geom[lev].periodicity());
        eta_mf[lev]->FillBoundary(geom[lev].periodicity());
    }
    else if (scheme == 3)
    {
        // ========================================
        // RK3 (Shu-Osher)
        // ========================================

        int nghost = 4;
        amrex::MultiFab k1_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k1_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k2_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k2_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k3_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k3_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k3_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k3_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab rho_tmp(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab M_tmp(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab E_tmp(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab eta_tmp(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        // Stage 1: k1 = RHS(old)
        RHS(lev, time, k1_rho, k1_M, k1_E, k1_eta, *density_old_mf[lev], *momentum_old_mf[lev], *energy_per_vol_old_mf[lev], *eta_old_mf[lev], dt, DX);

        // Intermediate state 1: tmp = old + dt * k1
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, dt, k1_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, dt, k1_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, dt, k1_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, dt, k1_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 2: k2 = RHS(tmp)
        RHS(lev, time + dt, k2_rho, k2_M, k2_E, k2_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Intermediate state 2: tmp = 3/4*old + 1/4*(old + dt*k2) = old + dt/4*k2
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, 0.25 * dt, k2_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, 0.25 * dt, k2_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, 0.25 * dt, k2_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, 0.25 * dt, k2_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 3: k3 = RHS(tmp)
        RHS(lev, time + 0.5 * dt, k3_rho, k3_M, k3_E, k3_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Final update: new = old + dt/6*(k1 + k2 + 4*k3)
        for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();

            amrex::Array4<Set::Scalar> const &rho_new = density_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &M_new = momentum_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &E_new = energy_per_vol_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &eta_new = eta_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &rho_old = density_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &M_old = momentum_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &E_old = energy_per_vol_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &eta_old = eta_old_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &k1_rho_arr = k1_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_M_arr = k1_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_E_arr = k1_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_eta_arr = k1_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k2_rho_arr = k2_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_M_arr = k2_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_E_arr = k2_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_eta_arr = k2_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k3_rho_arr = k3_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_M_arr = k3_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_E_arr = k3_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_eta_arr = k3_eta.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new(i, j, k) = rho_old(i, j, k) + dt / 6.0 * (k1_rho_arr(i, j, k) + k2_rho_arr(i, j, k) + 4.0 * k3_rho_arr(i, j, k));
                M_new(i, j, k, 0) = M_old(i, j, k, 0) + dt / 6.0 * (k1_M_arr(i, j, k, 0) + k2_M_arr(i, j, k, 0) + 4.0 * k3_M_arr(i, j, k, 0));
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + dt / 6.0 * (k1_M_arr(i, j, k, 1) + k2_M_arr(i, j, k, 1) + 4.0 * k3_M_arr(i, j, k, 1));
                E_new(i, j, k) = E_old(i, j, k) + dt / 6.0 * (k1_E_arr(i, j, k) + k2_E_arr(i, j, k) + 4.0 * k3_E_arr(i, j, k));
                eta_new(i, j, k) = eta_old(i, j, k) + dt / 6.0 * (k1_eta_arr(i, j, k) + k2_eta_arr(i, j, k) + 4.0 * k3_eta_arr(i, j, k));

                // Apply eta cutoff
                if (eta_new(i, j, k) <= cutoff)
                    eta_new(i, j, k) = 0.0;
                else if (eta_new(i, j, k) >= (1.0 - cutoff))
                    eta_new(i, j, k) = 1.0;
            });
        }

        density_mf[lev]->FillBoundary(geom[lev].periodicity());
        momentum_mf[lev]->FillBoundary(geom[lev].periodicity());
        energy_per_vol_mf[lev]->FillBoundary(geom[lev].periodicity());
        eta_mf[lev]->FillBoundary(geom[lev].periodicity());
    }
    else if (scheme == 4)
    {
        // ========================================
        // RK4 (Classical)
        // ========================================

        int nghost = 4;
        amrex::MultiFab k1_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k1_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k1_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k2_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k2_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k2_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k3_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k3_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k3_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k3_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab k4_rho(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k4_M(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab k4_E(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab k4_eta(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        amrex::MultiFab rho_tmp(density_old_mf[lev]->boxArray(), density_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab M_tmp(momentum_old_mf[lev]->boxArray(), momentum_old_mf[lev]->DistributionMap(), 2, nghost);
        amrex::MultiFab E_tmp(energy_per_vol_old_mf[lev]->boxArray(), energy_per_vol_old_mf[lev]->DistributionMap(), 1, nghost);
        amrex::MultiFab eta_tmp(eta_old_mf[lev]->boxArray(), eta_old_mf[lev]->DistributionMap(), 1, nghost);

        // Stage 1: k1 = RHS(old)
        RHS(lev, time, k1_rho, k1_M, k1_E, k1_eta, *density_old_mf[lev], *momentum_old_mf[lev], *energy_per_vol_old_mf[lev], *eta_old_mf[lev], dt, DX);

        // Intermediate state 1: tmp = old + dt/2 * k1
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, 0.5 * dt, k1_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, 0.5 * dt, k1_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, 0.5 * dt, k1_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, 0.5 * dt, k1_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 2: k2 = RHS(tmp)
        RHS(lev, time + 0.5 * dt, k2_rho, k2_M, k2_E, k2_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Intermediate state 2: tmp = old + dt/2 * k2
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, 0.5 * dt, k2_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, 0.5 * dt, k2_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, 0.5 * dt, k2_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, 0.5 * dt, k2_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 3: k3 = RHS(tmp)
        RHS(lev, time + 0.5 * dt, k3_rho, k3_M, k3_E, k3_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Intermediate state 3: tmp = old + dt * k3
        amrex::MultiFab::LinComb(rho_tmp, 1.0, *density_old_mf[lev], 0, dt, k3_rho, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(M_tmp, 1.0, *momentum_old_mf[lev], 0, dt, k3_M, 0, 0, 2, nghost);
        amrex::MultiFab::LinComb(E_tmp, 1.0, *energy_per_vol_old_mf[lev], 0, dt, k3_E, 0, 0, 1, nghost);
        amrex::MultiFab::LinComb(eta_tmp, 1.0, *eta_old_mf[lev], 0, dt, k3_eta, 0, 0, 1, nghost);

        rho_tmp.FillBoundary(geom[lev].periodicity());
        M_tmp.FillBoundary(geom[lev].periodicity());
        E_tmp.FillBoundary(geom[lev].periodicity());
        eta_tmp.FillBoundary(geom[lev].periodicity());

        // Stage 4: k4 = RHS(tmp)
        RHS(lev, time + dt, k4_rho, k4_M, k4_E, k4_eta, rho_tmp, M_tmp, E_tmp, eta_tmp, dt, DX);

        // Final update: new = old + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();

            amrex::Array4<Set::Scalar> const &rho_new = density_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &M_new = momentum_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &E_new = energy_per_vol_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &eta_new = eta_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &rho_old = density_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &M_old = momentum_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &E_old = energy_per_vol_old_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> const &eta_old = eta_old_mf[lev]->array(mfi);

            amrex::Array4<const Set::Scalar> const &k1_rho_arr = k1_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_M_arr = k1_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_E_arr = k1_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k1_eta_arr = k1_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k2_rho_arr = k2_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_M_arr = k2_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_E_arr = k2_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k2_eta_arr = k2_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k3_rho_arr = k3_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_M_arr = k3_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_E_arr = k3_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k3_eta_arr = k3_eta.array(mfi);

            amrex::Array4<const Set::Scalar> const &k4_rho_arr = k4_rho.array(mfi);
            amrex::Array4<const Set::Scalar> const &k4_M_arr = k4_M.array(mfi);
            amrex::Array4<const Set::Scalar> const &k4_E_arr = k4_E.array(mfi);
            amrex::Array4<const Set::Scalar> const &k4_eta_arr = k4_eta.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new(i, j, k) = rho_old(i, j, k) + dt / 6.0 * (k1_rho_arr(i, j, k) + 2.0 * k2_rho_arr(i, j, k) + 2.0 * k3_rho_arr(i, j, k) + k4_rho_arr(i, j, k));
                M_new(i, j, k, 0) = M_old(i, j, k, 0) + dt / 6.0 * (k1_M_arr(i, j, k, 0) + 2.0 * k2_M_arr(i, j, k, 0) + 2.0 * k3_M_arr(i, j, k, 0) + k4_M_arr(i, j, k, 0));
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + dt / 6.0 * (k1_M_arr(i, j, k, 1) + 2.0 * k2_M_arr(i, j, k, 1) + 2.0 * k3_M_arr(i, j, k, 1) + k4_M_arr(i, j, k, 1));
                E_new(i, j, k) = E_old(i, j, k) + dt / 6.0 * (k1_E_arr(i, j, k) + 2.0 * k2_E_arr(i, j, k) + 2.0 * k3_E_arr(i, j, k) + k4_E_arr(i, j, k));
                eta_new(i, j, k) = eta_old(i, j, k) + dt / 6.0 * (k1_eta_arr(i, j, k) + 2.0 * k2_eta_arr(i, j, k) + 2.0 * k3_eta_arr(i, j, k) + k4_eta_arr(i, j, k));

                // Apply eta cutoff
                if (eta_new(i, j, k) <= cutoff)
                    eta_new(i, j, k) = 0.0;
                else if (eta_new(i, j, k) >= (1.0 - cutoff))
                    eta_new(i, j, k) = 1.0;
            });
        }

        density_mf[lev]->FillBoundary(geom[lev].periodicity());
        momentum_mf[lev]->FillBoundary(geom[lev].periodicity());
        energy_per_vol_mf[lev]->FillBoundary(geom[lev].periodicity());
        eta_mf[lev]->FillBoundary(geom[lev].periodicity());
    }
    else
    {
        Util::ParallelMessage(INFO, "ERROR in Hydro2::Advance() : Integrator Methods");
        Util::ParallelMessage(INFO, "Method ", scheme, " is unknown.");
        Util::ParallelMessage(INFO, "Valid schemes: 1=Forward Euler, 2=RK2, 3=RK3, 4=RK4");
        Util::Exception(INFO);
    }

    // ============================================================================
    // UPDATE PLOTTING FIELDS FROM FINAL STATE
    // ============================================================================
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Eta
        Set::Patch<const Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_eta_ = hess_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_mag_grad_eta_ = grad_mag_grad_eta_mf.Patch(lev, mfi);

        // Mixture
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
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
        Set::Patch<Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Derivative Function Calls
            Set::Vector grad_eta = Numeric::Gradient(eta_new, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta_new, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta_new, i, j, k, 0, DX);

            // gamma
            Set::Scalar A = (eta_new(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta_new(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta_new(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta_new(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            gammaf(i, j, k) = 1.0 + (1.0 / A);

            // etadot
            etadot(i, j, k) = (eta_new(i, j, k) - eta(i, j, k)) / dt;

            // Velocity = M ./ rho
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
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE_vol(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref;

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta_new(i, j, k) * (eta_new(i, j, k) - 0.5) * (eta_new(i, j, k) - 1.0);
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Spalding Number
            Bm(i, j, k) = eta_new(i, j, k) / (1.0 - eta_new(i, j, k) + small);

            // Speed of sound:
            a(i, j, k) = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

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

                kappa1 = n_hat.dot(hess_eta * n_hat); // Normal Curvature
                kappa2 = t1.dot(hess_eta * t1);       // Tangential Curvature

                kappa1 = -kappa1;
                kappa2 = -kappa2 * 2.0 * epsilon;

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
                or (press(i, j, k) > 1E1000))
            {
                Util::ParallelMessage(INFO, "v=", v(i, j, k, 0), ", ", v(i, j, k, 1));
                Util::ParallelMessage(INFO, "press=", press(i, j, k));
                Util::ParallelMessage(INFO, "p_eff=", p0_eff(i, j, k));
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k, 0), ", ", M(i, j, k, 1));
                Util::ParallelMessage(INFO, "E=", E_vol(i, j, k));
                Util::ParallelMessage(INFO, "KE=", KE_vol(i, j, k));
                Util::ParallelMessage(INFO, "UE=", UE_vol(i, j, k));
                Util::ParallelMessage(INFO, "Ma=", Ma(i, j, k, 0), ", ", Ma(i, j, k, 1));
                Util::ParallelMessage(INFO, "a=", a(i, j, k));
                Util::ParallelMessage(INFO, "gamma=", gammaf(i, j, k));
                Util::ParallelMessage(INFO, "eta=", eta_new(i, j, k));
                Util::ParallelMessage(INFO, "etadot=", etadot(i, j, k));
                Util::Exception(INFO);
            }
        });
    }

    amrex::Gpu::synchronize();
    amrex::ParallelDescriptor::Barrier();

    // ============================================================================
    // ADAPTIVE TIMESTEP CALCULATION
    // ============================================================================

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);

        Set::Scalar *dt_max_handle = &dt_max;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Solving for new states
            Set::Scalar sound_speed_new = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

            c_max = std::max(c_max, sound_speed_new);
            vx_max = std::max(vx_max, std::abs(v(i, j, k, 0)));
            vy_max = std::max(vy_max, std::abs(v(i, j, k, 1)));
            rho_min = std::min(rho_min, rho(i, j, k));
        });
    }

    // Reduce maximums across all processors (if parallel)
    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);
    amrex::ParallelDescriptor::ReduceRealMax(F_max);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min);

    // Compute timestep constraints
    Set::Scalar dx_min = std::min(DX[0], DX[1]);

    // 1. Acoustic CFL
    Set::Scalar wave_speed = c_max + std::sqrt(vx_max * vx_max + vy_max * vy_max);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    // 2. Viscous CFL
    Set::Scalar mu_max = std::max(mu0, mu1);
    Set::Scalar dt_viscous = cfl_v * rho_min * dx_min * dx_min / (mu_max + small);

    // 3. Force CFL
    Set::Scalar a_max = F_max / (rho_min + small); // Maximum acceleration
    Set::Scalar dt_force = cfl_v * std::sqrt(dx_min / (a_max + small));

    // 4. Allen-Cahn diffusion CFL
    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    // Take minimum
    dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn });

    // Safety factor
    dt_max *= 0.9;

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

    amrex::Gpu::synchronize();
    amrex::ParallelDescriptor::Barrier();
}
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// REGRIDDING //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::Regrid(int lev, Set::Scalar /* time */)
{
    BL_PROFILE("Integrator::Hydro2::Regrid");

    Source_mf[lev]->setVal(0.0);

    if (lev < finest_level)
        return;

    Util::Message(INFO, "Regridding on level", lev);
}

void
Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar, int)
{
    BL_PROFILE("Integrator::Hydro2::TagCellsForRefinement");

    const Set::Scalar *DX = geom[lev].CellSize();
    Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));

    // Eta criterion for refinement
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<char> const &tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const &eta = (*eta_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            if (grad_eta.lpNorm<2>() * dr * 2 > eta_refinement_criterion)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Vorticity criterion for refinement
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<char> const &tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const &omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, 0, DX, sten);
            if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Gradu criterion for refinement
    for (amrex::MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<char> const &tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const &v = (*velocity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);
            if (grad_u.lpNorm<2>() * dr * 2 > gradu_refinement_criterion)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Pressure criterion for refinement
    for (amrex::MFIter mfi(*pressure_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<char> const &tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const &press = (*pressure_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_p = Numeric::Gradient(press, i, j, k, 0, DX, sten);
            if (grad_p.lpNorm<2>() * dr * 2 > p_refinement_criterion)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Density criterion for refinement
    for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        amrex::Array4<char> const &tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const &rho = (*density_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
            Set::Vector grad_rho = Numeric::Gradient(rho, i, j, k, 0, DX, sten);
            if (grad_rho.lpNorm<2>() * dr * 2 > rho_refinement_criterion)
                tags(i, j, k) = amrex::TagBox::SET;
        });
    }
}

}

#endif

