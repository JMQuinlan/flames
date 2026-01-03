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
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
//#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
//#include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
//#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
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
        pp_query_default("small", value.small, 1.0E-8);       // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-8);   // eta cutoff value
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
        pp_query_default("apply_vaporization", value.apply_vaporization, false);      // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

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
        pp_query_default("sigma", value.sigma, 0.0);   // surface tension condition
        pp_query_default("Dv", value.Dv, 0.0);   // Vapor Diffusivity
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
        int nghost = 3;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,          value.energy_bc, 1, nghost, "eta", true);
        value.RegisterNewFab(value.eta_old_mf,      value.energy_bc, 1, nghost, "eta_old", false);
        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing, 1, nghost, "etadot", true);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing, 4, nghost, "hess_eta", false, { "00", "01", "10", "11" });
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

        value.RegisterNewFab(value.pressure0_mf,    value.energy_bc,  1, nghost, "pressure0", false);
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

        value.RegisterNewFab(value.pressure1_mf,    value.energy_bc,  1, nghost, "pressure1", false);
        value.RegisterNewFab(value.velocity1_mf,    &value.bc_nothing,  2, nghost, "velocity1", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity1_mf,   &value.bc_nothing,  1, nghost, "vorticity1", false);

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,     value.energy_bc, 1, nghost, "pressure", true);
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
        value.RegisterNewFab(value.tau_xx_mf, value.density_bc, 1, nghost, "tau_xx", true, { "xx" });           // Stress Tensor
        value.RegisterNewFab(value.tau_xy_mf, value.density_bc, 1, nghost, "tau_xy", true, { "xy" });           // Stress Tensor
        value.RegisterNewFab(value.tau_yy_mf, value.density_bc, 1, nghost, "tau_yy", true, { "yy" });           // Stress Tensor
        value.RegisterNewFab(value.Ldot_mf,         &value.bc_nothing,  2, nghost, "Ldot", true, { "x", "y" });  // Ldot
        value.RegisterNewFab(value.T_mf,            &value.bc_nothing,  1, nghost, "T", true);                  // Temperature
        value.RegisterNewFab(value.cp_mf,           &value.bc_nothing,  1, nghost, "cp", false);                // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,           &value.bc_nothing,  1, nghost, "cv", false);                // Constant Volume Specific Heat
        value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false);         // Thermal Conductivity
        value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc, 1, nghost, "gamma", true);                 // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf,           value.energy_bc, 1, nghost, "p0", true);                    // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc, 1, nghost, "mu_chem", true);               // Tammann Pressure
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   &value.bc_nothing,  1, nghost, "UE_per_vol", true);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   &value.bc_nothing,  1, nghost, "UE_per_mass", true);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   &value.bc_nothing,  1, nghost, "KE_per_vol", true);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   &value.bc_nothing,  1, nghost, "KE_per_mass", true);        // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,           &value.bc_nothing,  1, nghost, "Spadling_Number", true);    // Spalding Number

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, nghost, "grad_eta", true, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, { "Avg", "1", "2" }); // To Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", true, { "x", "y" }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,     &value.bc_nothing,  1, nghost, "rho_flux", true);                    // Density Flux
        value.RegisterNewFab(value.M_flux_mf,       &value.bc_nothing,  2, nghost, "M_flux", true, { "x", "y" });        // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,       &value.bc_nothing,  1, nghost, "E_flux", true);                      // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,      &value.bc_nothing,  2, nghost, "div_tau", true, { "x", "y" });            // Viscous Stress
        value.RegisterNewFab(value.hess_u_mf,       &value.bc_nothing,  8, nghost, "hess_u", false, {
                                                                                                     "000","001",
                                                                                                     "010","011",
                                                                                                     "100","101",
                                                                                                     "110","111",
                                                                                                    }); // hess_u Flux


        // Stage storage for RK schemes
        value.RegisterNewFab(value.density_stage_mf, value.density_bc, 1, nghost, "density_stage", false);
        value.RegisterNewFab(value.momentum_stage_mf, value.momentum_bc, 2, nghost, "momentum_stage", false);
        value.RegisterNewFab(value.energy_per_vol_stage_mf, value.energy_bc, 1, nghost, "energy_per_vol_stage", false);
        value.RegisterNewFab(value.energy_per_mas_stage_mf, value.energy_bc, 1, nghost, "energy_per_mas_stage", false);
        value.RegisterNewFab(value.eta_stage_mf, value.energy_bc, 1, nghost, "eta_stage", false);

        // Stage-specific primitive/derived fields
        value.RegisterNewFab(value.velocity_stage_mf, &value.bc_nothing, 2, nghost, "velocity_stage", false);
        value.RegisterNewFab(value.pressure_stage_mf, value.energy_bc, 1, nghost, "pressure_stage", false);
        value.RegisterNewFab(value.gamma_stage_mf, value.energy_bc, 1, nghost, "gamma_stage", false);
        value.RegisterNewFab(value.p0_stage_mf, value.energy_bc, 1, nghost, "p0_stage", false);
        value.RegisterNewFab(value.a_stage_mf, &value.bc_nothing, 1, nghost, "a_stage", false);
        value.RegisterNewFab(value.Ma_stage_mf, &value.bc_nothing, 2, nghost, "Ma_stage", false);
        value.RegisterNewFab(value.mu_chem_stage_mf, value.energy_bc, 1, nghost, "mu_chem_stage", false);
        value.RegisterNewFab(value.Bm_stage_mf, &value.bc_nothing, 1, nghost, "Bm_stage", false);
        value.RegisterNewFab(value.T_stage_mf, &value.bc_nothing, 1, nghost, "T_stage", false);
        value.RegisterNewFab(value.cp_stage_mf, &value.bc_nothing, 1, nghost, "cp_stage", false);
        value.RegisterNewFab(value.cv_stage_mf, &value.bc_nothing, 1, nghost, "cv_stage", false);
        value.RegisterNewFab(value.k_thermal_stage_mf, &value.bc_nothing, 1, nghost, "k_thermal_stage", false);
        value.RegisterNewFab(value.h_thermal_stage_mf, &value.bc_nothing, 1, nghost, "h_thermal_stage", false);
        value.RegisterNewFab(value.tau_xx_stage_mf, value.density_bc, 1, nghost, "tau_xx_stage", false);
        value.RegisterNewFab(value.tau_xy_stage_mf, value.density_bc, 1, nghost, "tau_xy_stage", false);
        value.RegisterNewFab(value.tau_yy_stage_mf, value.density_bc, 1, nghost, "tau_yy_stage", false);
        value.RegisterNewFab(value.Ldot_stage_mf, &value.bc_nothing, 2, nghost, "Ldot_stage", false);
        value.RegisterNewFab(value.Source_stage_mf, &value.bc_nothing, 4, nghost, "Source_stage", false);
        value.RegisterNewFab(value.Fsv_stage_mf, &value.bc_nothing, 2, nghost, "Fsv_stage", false);
        value.RegisterNewFab(value.Fb_stage_mf, &value.bc_nothing, 2, nghost, "Fb_stage", false);
        value.RegisterNewFab(value.Fw_stage_mf, &value.bc_nothing, 2, nghost, "Fw_stage", false);
        value.RegisterNewFab(value.div_tau_stage_mf, &value.bc_nothing, 2, nghost, "div_tau_stage", false);
        value.RegisterNewFab(value.hess_u_stage_mf, &value.bc_nothing, 8, nghost, "hess_u_stage", false);
        value.RegisterNewFab(value.grad_eta_stage_mf, &value.bc_nothing, 2, nghost, "grad_eta_stage", false);
        value.RegisterNewFab(value.hess_eta_stage_mf, &value.bc_nothing, 4, nghost, "hess_eta_stage", false);
        value.RegisterNewFab(value.n_hat_stage_mf, &value.bc_nothing, 2, nghost, "n_hat_stage", false);
        value.RegisterNewFab(value.kappas_stage_mf, &value.bc_nothing, 3, nghost, "kappas_stage", false);
        value.RegisterNewFab(value.grad_mag_grad_eta_stage_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta_stage", false);
        value.RegisterNewFab(value.etadot_stage_mf, &value.bc_nothing, 1, nghost, "etadot_stage", false);
        value.RegisterNewFab(value.KE_per_vol_stage_mf, &value.bc_nothing, 1, nghost, "KE_per_vol_stage", false);
        value.RegisterNewFab(value.KE_per_mas_stage_mf, &value.bc_nothing, 1, nghost, "KE_per_mas_stage", false);
        value.RegisterNewFab(value.UE_per_vol_stage_mf, &value.bc_nothing, 1, nghost, "UE_per_vol_stage", false);
        value.RegisterNewFab(value.UE_per_mas_stage_mf, &value.bc_nothing, 1, nghost, "UE_per_mas_stage", false);

        // RK stage derivatives
        value.RegisterNewFab(value.k1_rho_mf, value.density_bc, 1, nghost, "k1_rho", false);
        value.RegisterNewFab(value.k1_M_mf, value.momentum_bc, 2, nghost, "k1_M", false);
        value.RegisterNewFab(value.k1_E_mf, value.energy_bc, 1, nghost, "k1_E", false);
        value.RegisterNewFab(value.k1_eta_mf, value.energy_bc, 1, nghost, "k1_eta", false);

        value.RegisterNewFab(value.k2_rho_mf, value.density_bc, 1, nghost, "k2_rho", false);
        value.RegisterNewFab(value.k2_M_mf, value.momentum_bc, 2, nghost, "k2_M", false);
        value.RegisterNewFab(value.k2_E_mf, value.energy_bc, 1, nghost, "k2_E", false);
        value.RegisterNewFab(value.k2_eta_mf, value.energy_bc, 1, nghost, "k2_eta", false);

        value.RegisterNewFab(value.k3_rho_mf, value.density_bc, 1, nghost, "k3_rho", false);
        value.RegisterNewFab(value.k3_M_mf, value.momentum_bc, 2, nghost, "k3_M", false);
        value.RegisterNewFab(value.k3_E_mf, value.energy_bc, 1, nghost, "k3_E", false);
        value.RegisterNewFab(value.k3_eta_mf, value.energy_bc, 1, nghost, "k3_eta", false);

        value.RegisterNewFab(value.k4_rho_mf, value.density_bc, 1, nghost, "k4_rho", false);
        value.RegisterNewFab(value.k4_M_mf, value.momentum_bc, 2, nghost, "k4_M", false);
        value.RegisterNewFab(value.k4_E_mf, value.energy_bc, 1, nghost, "k4_E", false);
        value.RegisterNewFab(value.k4_eta_mf, value.energy_bc, 1, nghost, "k4_eta", false);
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
    else if (value.Riemann_Solver == 3)
    {
        // pp.select_default<Solver::Local::FluidRiemann::HLLCE>("solver", value.hllcesolver);
    }
    else if (value.Riemann_Solver == 35)
    {
        // pp.select_default<Solver::Local::FluidRiemann::HLLC_WENO5>("solver", value.hllc_weno5solver);
    }
    else if (value.Riemann_Solver == 36)
    {
        // pp.select_default<Solver::Local::FluidRiemann::PartiallyParabolic>("solver", value.partiallyparabolicsolver);
    }
    else if (value.Riemann_Solver == 37)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLC_All_Mach>("solver", value.hllc_All_Machsolver);
    }
    else if (value.Riemann_Solver == 38)
    {
        pp.select_default<Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro>("solver", value.hllc_All_Mach_Furfarosolver);
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
        Util::ParallelMessage(INFO, "HLLC_All_Mach : 37");
        Util::ParallelMessage(INFO, "HLLC_All_Mach_Furfaro : 38");
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

    // NATURAL SOURCE
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Fb_mf[lev]      ->setVal(0.0);
    Fw_mf[lev]      ->setVal(0.0); 
    tau_xx_mf[lev]     ->setVal(0.0); 
    tau_xy_mf[lev]     ->setVal(0.0); 
    tau_yy_mf[lev]     ->setVal(0.0); 
    Ldot_mf[lev]    ->setVal(0.0); 

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
        Set::Patch<Set::Scalar>         Bm          = Bm_mf.Patch(lev, mfi);

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
Hydro2::RHS(int lev, Set::Scalar time, Set::Scalar dt, const Set::Scalar *DX, amrex::MultiFab &drho_dt_mf, amrex::MultiFab &dM_dt_mf, amrex::MultiFab &dE_dt_mf, amrex::MultiFab &deta_dt_mf, Set::Scalar &dt_max, bool use_stage)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    // ============================================================
    // PHASE 1: Calculate primitive fields and derived quantities
    // ============================================================
    PrimitiveFieldCalc(lev, dt, DX, use_stage);

    // Synchronize primitive fields before next phase
    if (use_stage)
    {
        (*velocity_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*pressure_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*gamma_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*p0_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*a_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*Ma_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*grad_eta_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*hess_eta_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*n_hat_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*kappas_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*mu_chem_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
    }
    else
    {
        (*velocity_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*pressure_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*gamma_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*p0_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*a_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*Ma_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*grad_eta_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*hess_eta_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*n_hat_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*kappas_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*mu_chem_mf[lev]).FillBoundary(geom[lev].periodicity());
    }

    // ============================================================
    // PHASE 2: Calculate natural/intermediate quantities
    // ============================================================
    NaturalCalc(lev, dt, DX, use_stage);

    // Synchronize stress tensor and viscous terms before next phase
    if (use_stage)
    {
        (*tau_xx_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*tau_xy_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*tau_yy_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*Ldot_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
    }
    else
    {
        (*tau_xx_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*tau_xy_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*tau_yy_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*Ldot_mf[lev]).FillBoundary(geom[lev].periodicity());
    }

    // ============================================================
    // PHASE 3: Calculate forced source terms
    // ============================================================
    ForcedCalc(lev, dt, DX, use_stage);

    // Synchronize source terms before flux calculation
    if (use_stage)
    {
        (*Source_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*div_tau_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
    }
    else
    {
        (*Source_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*div_tau_mf[lev]).FillBoundary(geom[lev].periodicity());
    }

    // ============================================================
    // PHASE 4: Calculate Riemann fluxes and compute derivatives
    // ============================================================
    RiemannFlux(lev, time, dt, DX, &drho_dt_mf, &dM_dt_mf, &dE_dt_mf, &deta_dt_mf, dt_max, use_stage);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////// ADVANCE ///////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::Advance");

    // Swap pointers to prepare for new timestep
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev], energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev], energy_per_mas_mf[lev]);
    std::swap(eta_old_mf, eta_mf);

    // Initialize dynamic timestep tracking
    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();
    c_max = 0.0;
    vx_max = 0.0;
    vy_max = 0.0;
    F_max = 0.0;
    rho_min = 1e10;

    // Geometry
    const Set::Scalar *DX = geom[lev].CellSize();

    // Ensure boundary sources are synchronized
    m0_mf[lev]->FillBoundary(geom[lev].periodicity());
    u0_mf[lev]->FillBoundary(geom[lev].periodicity());
    q_mf[lev]->FillBoundary(geom[lev].periodicity());

    // ============================================================
    // TIME INTEGRATION
    // ============================================================

    if (scheme == 0) // Forward Euler
    {
        Util::Message(INFO, "Using Forward Euler time integration");

        // Compute dU/dt at current state
        RHS(lev, time, dt, DX, *k1_rho_mf[lev], *k1_M_mf[lev], *k1_E_mf[lev], *k1_eta_mf[lev], dt_max, false);

        // Update: U^{n+1} = U^n + dt * k1
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> E_old = energy_per_vol_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> eta_old = eta_old_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k1_rho = k1_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_M = k1_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_E = k1_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_eta = k1_eta_mf.Patch(lev, mfi);

            Set::Patch<Set::Scalar> rho_new = density_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> M_new = momentum_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> E_new = energy_per_vol_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new(i, j, k) = rho_old(i, j, k) + dt * k1_rho(i, j, k, 0);
                rho_new(i, j, k) = std::max(rho_new(i, j, k), small);

                M_new(i, j, k, 0) = M_old(i, j, k, 0) + dt * k1_M(i, j, k, 0);
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + dt * k1_M(i, j, k, 1);

                E_new(i, j, k) = E_old(i, j, k) + dt * k1_E(i, j, k, 0);

                if (static_eta == 1)
                {
                    eta_new(i, j, k) = eta_old(i, j, k);
                }
                else
                {
                    eta_new(i, j, k) = eta_old(i, j, k) + dt * k1_eta(i, j, k, 0);
                    if (eta_new(i, j, k) <= cutoff)
                        eta_new(i, j, k) = 0.0;
                    else if (eta_new(i, j, k) >= (1.0 - cutoff))
                        eta_new(i, j, k) = 1.0;
                }
            });
        }
    }
    else if (scheme == 1) // RK4
    {
        Util::Message(INFO, "Using RK4 time integration");

        // Stage 1: k1 = RHS(U^n)
        RHS(lev, time, dt, DX, *k1_rho_mf[lev], *k1_M_mf[lev], *k1_E_mf[lev], *k1_eta_mf[lev], dt_max, false);

        // Stage 2: U_stage = U^n + 0.5*dt*k1, k2 = RHS(U_stage)
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> E_old = energy_per_vol_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> eta_old = eta_old_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k1_rho = k1_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_M = k1_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_E = k1_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_eta = k1_eta_mf.Patch(lev, mfi);

            Set::Patch<Set::Scalar> rho_stage = density_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> M_stage = momentum_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> E_stage = energy_per_vol_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> eta_stage = eta_stage_mf.Patch(lev, mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_stage(i, j, k) = rho_old(i, j, k) + 0.5 * dt * k1_rho(i, j, k, 0);
                M_stage(i, j, k, 0) = M_old(i, j, k, 0) + 0.5 * dt * k1_M(i, j, k, 0);
                M_stage(i, j, k, 1) = M_old(i, j, k, 1) + 0.5 * dt * k1_M(i, j, k, 1);
                E_stage(i, j, k) = E_old(i, j, k) + 0.5 * dt * k1_E(i, j, k, 0);
                eta_stage(i, j, k) = eta_old(i, j, k) + 0.5 * dt * k1_eta(i, j, k, 0);
            });
        }

        (*density_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*momentum_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*energy_per_vol_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*eta_stage_mf[lev]).FillBoundary(geom[lev].periodicity());

        RHS(lev, time + 0.5 * dt, dt, DX, *k2_rho_mf[lev], *k2_M_mf[lev], *k2_E_mf[lev], *k2_eta_mf[lev], dt_max, true);

        // Stage 3: U_stage = U^n + 0.5*dt*k2, k3 = RHS(U_stage)
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> E_old = energy_per_vol_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> eta_old = eta_old_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k2_rho = k2_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_M = k2_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_E = k2_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_eta = k2_eta_mf.Patch(lev, mfi);

            Set::Patch<Set::Scalar> rho_stage = density_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> M_stage = momentum_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> E_stage = energy_per_vol_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> eta_stage = eta_stage_mf.Patch(lev, mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_stage(i, j, k) = rho_old(i, j, k) + 0.5 * dt * k2_rho(i, j, k, 0);
                M_stage(i, j, k, 0) = M_old(i, j, k, 0) + 0.5 * dt * k2_M(i, j, k, 0);
                M_stage(i, j, k, 1) = M_old(i, j, k, 1) + 0.5 * dt * k2_M(i, j, k, 1);
                E_stage(i, j, k) = E_old(i, j, k) + 0.5 * dt * k2_E(i, j, k, 0);
                eta_stage(i, j, k) = eta_old(i, j, k) + 0.5 * dt * k2_eta(i, j, k, 0);
            });
        }

        (*density_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*momentum_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*energy_per_vol_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*eta_stage_mf[lev]).FillBoundary(geom[lev].periodicity());

        RHS(lev, time + 0.5 * dt, dt, DX, *k3_rho_mf[lev], *k3_M_mf[lev], *k3_E_mf[lev], *k3_eta_mf[lev], dt_max, true);

        // Stage 4: U_stage = U^n + dt*k3, k4 = RHS(U_stage)
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> E_old = energy_per_vol_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> eta_old = eta_old_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k3_rho = k3_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_M = k3_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_E = k3_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_eta = k3_eta_mf.Patch(lev, mfi);

            Set::Patch<Set::Scalar> rho_stage = density_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> M_stage = momentum_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> E_stage = energy_per_vol_stage_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> eta_stage = eta_stage_mf.Patch(lev, mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_stage(i, j, k) = rho_old(i, j, k) + dt * k3_rho(i, j, k, 0);
                M_stage(i, j, k, 0) = M_old(i, j, k, 0) + dt * k3_M(i, j, k, 0);
                M_stage(i, j, k, 1) = M_old(i, j, k, 1) + dt * k3_M(i, j, k, 1);
                E_stage(i, j, k) = E_old(i, j, k) + dt * k3_E(i, j, k, 0);
                eta_stage(i, j, k) = eta_old(i, j, k) + dt * k3_eta(i, j, k, 0);
            });
        }

        (*density_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*momentum_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*energy_per_vol_stage_mf[lev]).FillBoundary(geom[lev].periodicity());
        (*eta_stage_mf[lev]).FillBoundary(geom[lev].periodicity());

        RHS(lev, time + dt, dt, DX, *k4_rho_mf[lev], *k4_M_mf[lev], *k4_E_mf[lev], *k4_eta_mf[lev], dt_max, true);

        // Final update: U^{n+1} = U^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_old = density_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> E_old = energy_per_vol_old_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> eta_old = eta_old_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k1_rho = k1_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_M = k1_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_E = k1_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k1_eta = k1_eta_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k2_rho = k2_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_M = k2_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_E = k2_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k2_eta = k2_eta_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k3_rho = k3_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_M = k3_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_E = k3_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k3_eta = k3_eta_mf.Patch(lev, mfi);

            Set::Patch<const Set::Scalar> k4_rho = k4_rho_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k4_M = k4_M_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k4_E = k4_E_mf.Patch(lev, mfi);
            Set::Patch<const Set::Scalar> k4_eta = k4_eta_mf.Patch(lev, mfi);

            Set::Patch<Set::Scalar> rho_new = density_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> M_new = momentum_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> E_new = energy_per_vol_mf.Patch(lev, mfi);
            Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho_new(i, j, k) = rho_old(i, j, k) + (dt / 6.0) * (k1_rho(i, j, k, 0) + 2.0 * k2_rho(i, j, k, 0) + 2.0 * k3_rho(i, j, k, 0) + k4_rho(i, j, k, 0));
                rho_new(i, j, k) = std::max(rho_new(i, j, k), small);

                M_new(i, j, k, 0) = M_old(i, j, k, 0) + (dt / 6.0) * (k1_M(i, j, k, 0) + 2.0 * k2_M(i, j, k, 0) + 2.0 * k3_M(i, j, k, 0) + k4_M(i, j, k, 0));
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + (dt / 6.0) * (k1_M(i, j, k, 1) + 2.0 * k2_M(i, j, k, 1) + 2.0 * k3_M(i, j, k, 1) + k4_M(i, j, k, 1));

                E_new(i, j, k) = E_old(i, j, k) + (dt / 6.0) * (k1_E(i, j, k, 0) + 2.0 * k2_E(i, j, k, 0) + 2.0 * k3_E(i, j, k, 0) + k4_E(i, j, k, 0));

                if (static_eta == 1)
                {
                    eta_new(i, j, k) = eta_old(i, j, k);
                }
                else
                {
                    eta_new(i, j, k) = eta_old(i, j, k) + (dt / 6.0) * (k1_eta(i, j, k, 0) + 2.0 * k2_eta(i, j, k, 0) + 2.0 * k3_eta(i, j, k, 0) + k4_eta(i, j, k, 0));
                    if (eta_new(i, j, k) <= cutoff)
                        eta_new(i, j, k) = 0.0;
                    else if (eta_new(i, j, k) >= (1.0 - cutoff))
                        eta_new(i, j, k) = 1.0;
                }
            });
        }
    }
    else
    {
        Util::Abort(INFO, "Invalid time integration scheme: ", scheme);
    }

    // ============================================================
    // Update adaptive timestep if enabled
    // ============================================================
    if (dynamictimestep.on)
    {
        this->DynamicTimestep_SyncTimeStep(lev, dt_max);
    }

    // Final synchronization
    amrex::Gpu::synchronize();
    amrex::ParallelDescriptor::Barrier();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// PrimitiveFieldCalc //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::PrimitiveFieldCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage)
{
    BL_PROFILE("Integrator::Hydro2::PrimitiveFieldCalc");

    amrex::Box domain = geom[lev].Domain();

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Select input MultiFabs based on use_stage
        Set::Patch<const Set::Scalar> eta_new = use_stage ? eta_stage_mf.Patch(lev, mfi) : eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = use_stage ? eta_stage_mf.Patch(lev, mfi) : eta_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho = use_stage ? density_stage_mf.Patch(lev, mfi) : density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = use_stage ? energy_per_vol_stage_mf.Patch(lev, mfi) : energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = use_stage ? energy_per_mas_stage_mf.Patch(lev, mfi) : energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = use_stage ? momentum_stage_mf.Patch(lev, mfi) : momentum_old_mf.Patch(lev, mfi);

        // Select output MultiFabs based on use_stage
        Set::Patch<Set::Scalar> etadot = use_stage ? etadot_stage_mf.Patch(lev, mfi) : etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = use_stage ? grad_eta_stage_mf.Patch(lev, mfi) : grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_eta_ = use_stage ? hess_eta_stage_mf.Patch(lev, mfi) : hess_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = use_stage ? n_hat_stage_mf.Patch(lev, mfi) : n_hat_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = use_stage ? kappas_stage_mf.Patch(lev, mfi) : kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_mag_grad_eta_ = use_stage ? grad_mag_grad_eta_stage_mf.Patch(lev, mfi) : grad_mag_grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> a = use_stage ? a_stage_mf.Patch(lev, mfi) : a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ma = use_stage ? Ma_stage_mf.Patch(lev, mfi) : Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_vol = use_stage ? KE_per_vol_stage_mf.Patch(lev, mfi) : KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> KE_mas = use_stage ? KE_per_mas_stage_mf.Patch(lev, mfi) : KE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_vol = use_stage ? UE_per_vol_stage_mf.Patch(lev, mfi) : UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> UE_mas = use_stage ? UE_per_mas_stage_mf.Patch(lev, mfi) : UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> v = use_stage ? velocity_stage_mf.Patch(lev, mfi) : velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = use_stage ? pressure_stage_mf.Patch(lev, mfi) : pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = use_stage ? gamma_stage_mf.Patch(lev, mfi) : gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = use_stage ? p0_stage_mf.Patch(lev, mfi) : p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = use_stage ? mu_chem_stage_mf.Patch(lev, mfi) : mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = use_stage ? Bm_stage_mf.Patch(lev, mfi) : Bm_mf.Patch(lev, mfi);

        // Rest of the function is IDENTICAL to your current implementation
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            Set::Scalar A = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            gammaf(i, j, k) = 1.0 + (1.0 / A);

            etadot(i, j, k) = (eta_new(i, j, k) - eta(i, j, k)) / dt;

            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k));
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k));

            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));
            KE_mas(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            UE_vol(i, j, k) = E_vol(i, j, k) - KE_vol(i, j, k);
            UE_mas(i, j, k) = E_mas(i, j, k) - KE_mas(i, j, k);

            p0_eff(i, j, k) = (B / A) / gammaf(i, j, k);
            press(i, j, k) = (gammaf(i, j, k) - 1.0) * UE_vol(i, j, k) - gammaf(i, j, k) * p0_eff(i, j, k) + pref;

            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0);
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            Bm(i, j, k) = eta(i, j, k) / (1.0 - eta(i, j, k) + small);

            a(i, j, k) = std::sqrt(gammaf(i, j, k) * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

            Ma(i, j, k, 0) = v(i, j, k, 0) / (a(i, j, k) + small);
            Ma(i, j, k, 1) = v(i, j, k, 1) / (a(i, j, k) + small);

            Set::Vector n_hat = grad_eta / (grad_eta_mag + small);

            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

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

            Set::Vector grad_mag_grad_eta = Set::Vector(
                1 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                1 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));

            grad_mag_grad_eta_(i, j, k, 0) = grad_mag_grad_eta(0);
            grad_mag_grad_eta_(i, j, k, 1) = grad_mag_grad_eta(1);

            Set::Scalar kappa, kappa1, kappa2 = 0.0;

            if (kappa_method == 1)
            {
                kappa = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / (grad_eta_mag * grad_eta_mag + small)));
                kappas(i, j, k, 0) = kappa;
                kappas(i, j, k, 1) = kappa1;
                kappas(i, j, k, 2) = kappa2;
            }
            else if (kappa_method == 2)
            {
                Set::Vector t1;
                if (std::abs(n_hat(0)) > std::abs(n_hat(1)))
                {
                    t1 = Set::Vector(-n_hat(1), n_hat(0)) / std::sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                }
                else
                {
                    t1 = Set::Vector(n_hat(1), -n_hat(0)) / std::sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                }

                kappa1 = n_hat.dot(hess_eta * n_hat);
                kappa2 = t1.dot(hess_eta * t1);
                kappa1 = -kappa1;
                kappa2 = -kappa2 * 2.0 * epsilon;

                kappa = kappa2;

                kappas(i, j, k, 0) = kappa;
                kappas(i, j, k, 1) = kappa1;
                kappas(i, j, k, 2) = kappa2;
            }

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
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                Util::ParallelMessage(INFO, "etadot=", etadot(i, j, k));
                Util::Exception(INFO);
            }
        });
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// NaturalCalc /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::NaturalCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage)
{
    BL_PROFILE("Integrator::Hydro2::NaturalCalc");

    amrex::Box domain = geom[lev].Domain();

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Select input MultiFabs based on use_stage
        Set::Patch<const Set::Scalar> rho = use_stage ? density_stage_mf.Patch(lev, mfi) : density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = use_stage ? energy_per_vol_stage_mf.Patch(lev, mfi) : energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = use_stage ? energy_per_mas_stage_mf.Patch(lev, mfi) : energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = use_stage ? momentum_stage_mf.Patch(lev, mfi) : momentum_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = use_stage ? eta_stage_mf.Patch(lev, mfi) : eta_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> v = use_stage ? velocity_stage_mf.Patch(lev, mfi) : velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = use_stage ? pressure_stage_mf.Patch(lev, mfi) : pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = use_stage ? gamma_stage_mf.Patch(lev, mfi) : gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = use_stage ? p0_stage_mf.Patch(lev, mfi) : p0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> mu_chem_ = use_stage ? mu_chem_stage_mf.Patch(lev, mfi) : mu_chem_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Bm = use_stage ? Bm_stage_mf.Patch(lev, mfi) : Bm_mf.Patch(lev, mfi);

        // Boundary sources (always read from non-stage versions)
        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q0 = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        // Output MultiFabs based on use_stage
        Set::Patch<Set::Scalar> T = use_stage ? T_stage_mf.Patch(lev, mfi) : T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cp = use_stage ? cp_stage_mf.Patch(lev, mfi) : cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = use_stage ? cv_stage_mf.Patch(lev, mfi) : cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = use_stage ? k_thermal_stage_mf.Patch(lev, mfi) : k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = use_stage ? h_thermal_stage_mf.Patch(lev, mfi) : h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> tau_xx = use_stage ? tau_xx_stage_mf.Patch(lev, mfi) : tau_xx_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> tau_xy = use_stage ? tau_xy_stage_mf.Patch(lev, mfi) : tau_xy_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> tau_yy = use_stage ? tau_yy_stage_mf.Patch(lev, mfi) : tau_yy_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Ldot_ = use_stage ? Ldot_stage_mf.Patch(lev, mfi) : Ldot_mf.Patch(lev, mfi);

        // Debugging (read from stage if use_stage)
        Set::Patch<const Set::Scalar> grad_eta_ = use_stage ? grad_eta_stage_mf.Patch(lev, mfi) : grad_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> hess_eta_ = use_stage ? hess_eta_stage_mf.Patch(lev, mfi) : hess_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> n_hat_ = use_stage ? n_hat_stage_mf.Patch(lev, mfi) : n_hat_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> kappas = use_stage ? kappas_stage_mf.Patch(lev, mfi) : kappas_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> grad_mag_grad_eta_ = use_stage ? grad_mag_grad_eta_stage_mf.Patch(lev, mfi) : grad_mag_grad_eta_mf.Patch(lev, mfi);

        // Rest of the function is IDENTICAL to your current implementation
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small);

            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Vector u0 = Set::Vector(_u0(i, j, k, 0), _u0(i, j, k, 1));
            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));

            Set::Matrix eps = Set::Matrix::Zero();
            Set::Scalar div_u = gradu(0, 0) + gradu(1, 1);

            for (int p = 0; p < 2; ++p)
            {
                for (int q = 0; q < 2; ++q)
                {
                    eps(p, q) = 0.5 * (gradu(p, q) + gradu(q, p));
                }
            }

            Set::Scalar mu_eff = eta(i, j, k) * mu0 + (1.0 - eta(i, j, k)) * mu1;
            Set::Scalar lambda_eff = eta(i, j, k) * mu0_b + (1.0 - eta(i, j, k)) * mu1_b;
            Set::Vector grad_mu = (mu0 - mu1) * grad_eta;
            Set::Vector grad_lambda = (mu0_b - mu1_b) * grad_eta;

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

            Set::Vector Ldot = Set::Vector::Zero();
            for (int p = 0; p < 2; ++p)
            {
                for (int q = 0; q < 2; ++q)
                {
                    Ldot(p) = Ldot(p) + grad_mu(q) * (gradu(p, q) + gradu(q, p));
                }
                Ldot(p) = Ldot(p) + grad_lambda(p) * div_u;
                Ldot_(i, j, k, p) = Ldot(p);
            }

            if ((Ldot_(i, j, k, 0) != Ldot_(i, j, k, 0))
                or (Ldot_(i, j, k, 1) != Ldot_(i, j, k, 1))
                or (tau_xx(i, j, k) != tau_xx(i, j, k))
                or (tau_xy(i, j, k) != tau_xy(i, j, k))
                or (tau_yy(i, j, k) != tau_yy(i, j, k)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Intermediate time step loop:");
                Util::ParallelMessage(INFO, "eps=", eps(0, 0), ", ", eps(0, 1), "; ", eps(1, 0), ", ", eps(1, 1));
                Util::ParallelMessage(INFO, "Ldot=", Ldot_(i, j, k, 0), ", ", Ldot_(i, j, k, 1));
                Util::ParallelMessage(INFO, "tau_xx=", tau_xx(i, j, k));
                Util::ParallelMessage(INFO, "tau_xy=", tau_xy(i, j, k));
                Util::ParallelMessage(INFO, "tau_yy=", tau_yy(i, j, k));
                Util::Exception(INFO);
            }
        });
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// ForcedCalc //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::ForcedCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage)
{
    BL_PROFILE("Integrator::Hydro2::ForcedCalc");

    amrex::Box domain = geom[lev].Domain();

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Select input MultiFabs based on use_stage
        Set::Patch<const Set::Scalar> rho = use_stage ? density_stage_mf.Patch(lev, mfi) : density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = use_stage ? energy_per_vol_stage_mf.Patch(lev, mfi) : energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_mas = use_stage ? energy_per_mas_stage_mf.Patch(lev, mfi) : energy_per_mas_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = use_stage ? momentum_stage_mf.Patch(lev, mfi) : momentum_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = use_stage ? eta_stage_mf.Patch(lev, mfi) : eta_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> v = use_stage ? velocity_stage_mf.Patch(lev, mfi) : velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = use_stage ? pressure_stage_mf.Patch(lev, mfi) : pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> a = use_stage ? a_stage_mf.Patch(lev, mfi) : a_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Ma = use_stage ? Ma_stage_mf.Patch(lev, mfi) : Ma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = use_stage ? gamma_stage_mf.Patch(lev, mfi) : gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = use_stage ? p0_stage_mf.Patch(lev, mfi) : p0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> mu_chem_ = use_stage ? mu_chem_stage_mf.Patch(lev, mfi) : mu_chem_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Bm = use_stage ? Bm_stage_mf.Patch(lev, mfi) : Bm_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> tau_xx = use_stage ? tau_xx_stage_mf.Patch(lev, mfi) : tau_xx_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> tau_xy = use_stage ? tau_xy_stage_mf.Patch(lev, mfi) : tau_xy_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> tau_yy = use_stage ? tau_yy_stage_mf.Patch(lev, mfi) : tau_yy_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Ldot_ = use_stage ? Ldot_stage_mf.Patch(lev, mfi) : Ldot_mf.Patch(lev, mfi);

        // Boundary sources (always read from non-stage versions)
        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q0 = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        // Output MultiFabs based on use_stage
        Set::Patch<Set::Scalar> T = use_stage ? T_stage_mf.Patch(lev, mfi) : T_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cp = use_stage ? cp_stage_mf.Patch(lev, mfi) : cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cv = use_stage ? cv_stage_mf.Patch(lev, mfi) : cv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> k_thermal = use_stage ? k_thermal_stage_mf.Patch(lev, mfi) : k_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> h_thermal = use_stage ? h_thermal_stage_mf.Patch(lev, mfi) : h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Source = use_stage ? Source_stage_mf.Patch(lev, mfi) : Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fsv = use_stage ? Fsv_stage_mf.Patch(lev, mfi) : Fsv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fb = use_stage ? Fb_stage_mf.Patch(lev, mfi) : Fb_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fw = use_stage ? Fw_stage_mf.Patch(lev, mfi) : Fw_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> div_tau_ = use_stage ? div_tau_stage_mf.Patch(lev, mfi) : div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = use_stage ? hess_u_stage_mf.Patch(lev, mfi) : hess_u_mf.Patch(lev, mfi);

        // Debugging (read from stage if use_stage)
        Set::Patch<const Set::Scalar> grad_eta_ = use_stage ? grad_eta_stage_mf.Patch(lev, mfi) : grad_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> hess_eta_ = use_stage ? hess_eta_stage_mf.Patch(lev, mfi) : hess_eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> n_hat_ = use_stage ? n_hat_stage_mf.Patch(lev, mfi) : n_hat_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> kappas = use_stage ? kappas_stage_mf.Patch(lev, mfi) : kappas_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> grad_mag_grad_eta_ = use_stage ? grad_mag_grad_eta_stage_mf.Patch(lev, mfi) : grad_mag_grad_eta_mf.Patch(lev, mfi);

        // Rest of the function is IDENTICAL to your current implementation
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small);

            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Vector u0 = Set::Vector(_u0(i, j, k, 0), _u0(i, j, k, 1));
            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));

            Set::Vector q0_ = Set::Vector(q0(i, j, k, 0), q0(i, j, k, 1));

            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0_.dot(grad_eta);

            Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
            Set::Matrix3 hess_u = Set::Matrix3::Zero();

            Set::Scalar inv_rho = 1.0 / (rho(i, j, k));
            Set::Scalar inv_rho2 = inv_rho * inv_rho;
            Set::Scalar inv_rho3 = inv_rho2 * inv_rho;

            for (int r = 0; r < 2; ++r)
                for (int p = 0; p < 2; ++p)
                    for (int q = 0; q < 2; ++q)
                    {
                        hess_u(r, p, q) = inv_rho * hess_M(r, p, q)
                                          - inv_rho2 * (gradM(r, p) * gradrho(q) + gradM(r, q) * gradrho(p) + M(i, j, k, r) * hess_rho(p, q))
                                          + 2.0 * inv_rho3 * M(i, j, k, r) * gradrho(p) * gradrho(q);
                    }

            hess_u_(i, j, k, 0) = hess_u(0, 0, 0);
            hess_u_(i, j, k, 1) = hess_u(0, 0, 1);
            hess_u_(i, j, k, 2) = hess_u(0, 1, 0);
            hess_u_(i, j, k, 3) = hess_u(0, 1, 1);
            hess_u_(i, j, k, 4) = hess_u(1, 0, 0);
            hess_u_(i, j, k, 5) = hess_u(1, 0, 1);
            hess_u_(i, j, k, 6) = hess_u(1, 1, 0);
            hess_u_(i, j, k, 7) = hess_u(1, 1, 1);

            Set::Matrix grad_tau_xx = Numeric::Gradient(tau_xx, i, j, k, DX);
            Set::Matrix grad_tau_xy = Numeric::Gradient(tau_xy, i, j, k, DX);
            Set::Matrix grad_tau_yy = Numeric::Gradient(tau_yy, i, j, k, DX);

            Set::Vector div_tau = Set::Vector::Zero();
            div_tau(0) = grad_tau_xx(0, 0) + grad_tau_xy(0, 1);
            div_tau(1) = grad_tau_xy(1, 0) + grad_tau_yy(1, 1);

            Set::Vector Ldot = Set::Vector(Ldot_(i, j, k, 0), Ldot_(i, j, k, 1));

            if ((Ldot_(i, j, k, 0) != Ldot_(i, j, k, 0))
                or (Ldot_(i, j, k, 1) != Ldot_(i, j, k, 1))
                or (div_tau(0) != div_tau(0))
                or (div_tau(1) != div_tau(1)))
            {
                Util::ParallelMessage(INFO, "------------------------------------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN Hydro2(): Viscosity solving:");
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::ParallelMessage(INFO, "Ldot=", Ldot_(i, j, k, 0), ", ", Ldot_(i, j, k, 1));
                Util::ParallelMessage(INFO, "tau_xx=", tau_xx(i, j, k));
                Util::ParallelMessage(INFO, "tau_xy=", tau_xy(i, j, k));
                Util::ParallelMessage(INFO, "tau_yy=", tau_yy(i, j, k));
                Util::ParallelMessage(INFO, "grad_tau_xx=", grad_tau_xx(0, 0), ", ", grad_tau_xx(0, 1), "; ", grad_tau_xx(1, 0), ", ", grad_tau_xx(1, 1));
                Util::ParallelMessage(INFO, "grad_tau_xy=", grad_tau_xy(0, 0), ", ", grad_tau_xy(0, 1), "; ", grad_tau_xy(1, 0), ", ", grad_tau_xy(1, 1));
                Util::ParallelMessage(INFO, "grad_tau_yy=", grad_tau_yy(0, 0), ", ", grad_tau_yy(0, 1), "; ", grad_tau_yy(1, 0), ", ", grad_tau_yy(1, 1));
                Util::ParallelMessage(INFO, "div_tau=", div_tau(0), ", ", div_tau(1));
                Util::Exception(INFO);
            }

            div_tau_(i, j, k, 0) = div_tau(0);
            div_tau_(i, j, k, 1) = div_tau(1);

            Set::Scalar kappa = kappas(i, j, k, 0);
            Set::Vector grad_mag_grad_eta = Set::Vector(grad_mag_grad_eta_(i, j, k, 0), grad_mag_grad_eta_(i, j, k, 1));

            Fsv(i, j, k) = (0.0, 0.0);
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);

            if (apply_surface_tension)
            {
                if (grad_eta_mag > 0.01)
                {
                    Set::Scalar sigma_eff = sigma;
                    Set::Scalar alpha = 6 * std::sqrt(2);
                    Set::Scalar UFFDA = epsilon * alpha * grad_eta_mag * grad_eta_mag;

                    Fsv(i, j, k, 0) = sigma_eff * kappa * n_hat(0) * UFFDA;
                    Fsv(i, j, k, 1) = sigma_eff * kappa * n_hat(1) * UFFDA;
                }
                Fsv_vector = Set::Vector(Fsv(i, j, k, 0), Fsv(i, j, k, 1));
            }

            Fw(i, j, k) = (0.0, 0.0);
            Set::Vector Fw_vector = Set::Vector(0.0, 0.0);

            if (apply_weight)
            {
                Fw(i, j, k, 0) = 0.0;
                Fw(i, j, k, 1) = -rho(i, j, k) * g;
                Fw_vector = Set::Vector(Fw(i, j, k, 0), Fw(i, j, k, 1));
            }

            Fb(i, j, k) = (0.0, 0.0);
            Set::Vector Fb_vector = Set::Vector(0.0, 0.0);

            if (apply_buoyancy)
            {
                Fb_vector = Set::Vector(0.0, 0.0);
            }

            Set::Vector Total_Force = Set::Vector(Fsv(i, j, k, 0) + Fb_vector(0) + Fw_vector(0),
                                                  Fsv(i, j, k, 1) + Fb_vector(1) + Fw_vector(1));

            Source(i, j, k, 0) = mdot0;
            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force);

            Source(i, j, k, 1) = Source(i, j, k, 1) - lagrange * u.dot(grad_eta) * grad_eta(0);
            Source(i, j, k, 2) = Source(i, j, k, 2) - lagrange * u.dot(grad_eta) * grad_eta(1);
        });
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// RiemannFlux /////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RiemannFlux(int lev, Set::Scalar time, Set::Scalar dt, const Set::Scalar *DX, amrex::MultiFab *drho_dt_mf, amrex::MultiFab *dM_dt_mf, amrex::MultiFab *dE_dt_mf, amrex::MultiFab *deta_dt_mf, Set::Scalar &dt_max, bool use_stage)
{
    BL_PROFILE("Integrator::Hydro2::RiemannFlux");

    amrex::Box domain = geom[lev].Domain();

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // Select which MultiFabs to read from based on use_stage flag
        Set::Patch<const Set::Scalar> rho = use_stage ? density_stage_mf.Patch(lev, mfi) : density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = use_stage ? energy_per_vol_stage_mf.Patch(lev, mfi) : energy_per_vol_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = use_stage ? momentum_stage_mf.Patch(lev, mfi) : momentum_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = use_stage ? eta_stage_mf.Patch(lev, mfi) : eta_old_mf.Patch(lev, mfi);

        // Read from stage-specific derived fields
        Set::Patch<const Set::Scalar> v = use_stage ? velocity_stage_mf.Patch(lev, mfi) : velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = use_stage ? pressure_stage_mf.Patch(lev, mfi) : pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> a = use_stage ? a_stage_mf.Patch(lev, mfi) : a_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = use_stage ? gamma_stage_mf.Patch(lev, mfi) : gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = use_stage ? p0_stage_mf.Patch(lev, mfi) : p0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> mu_chem_ = use_stage ? mu_chem_stage_mf.Patch(lev, mfi) : mu_chem_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Bm = use_stage ? Bm_stage_mf.Patch(lev, mfi) : Bm_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Source = use_stage ? Source_stage_mf.Patch(lev, mfi) : Source_mf.Patch(lev, mfi);

        // Output: time derivatives
        /*
        Set::Patch<Set::Scalar> drho_dt = drho_dt_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> dM_dt = dM_dt_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> dE_dt = dE_dt_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> deta_dt = deta_dt_mf.Patch(lev, mfi);
        */

        Set::Patch<Set::Scalar> drho_dt = drho_dt_mf->array(mfi);
        Set::Patch<Set::Scalar> dM_dt = dM_dt_mf->array(mfi);
        Set::Patch<Set::Scalar> dE_dt = dE_dt_mf->array(mfi);
        Set::Patch<Set::Scalar> deta_dt = deta_dt_mf->array(mfi);

        // Debugging
        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);

        // Boundary sources
        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q0 = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        // Fluid 0 and 1 (for debugging - keep as is)
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T = use_stage ? T_stage_mf.Patch(lev, mfi) : T_mf.Patch(lev, mfi);

        Set::Scalar *dt_max_handle = &dt_max;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small);

            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));

            // Riemann solver (same as before)
            const int X = 0, Y = 1;
            std::vector<Solver::Local::FluidRiemann::State> x_states(3);
            std::vector<Solver::Local::FluidRiemann::State> y_states(3);

            x_states[0] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i - 1, j, k, X);
            x_states[1] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, X);
            x_states[2] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i + 1, j, k, X);
            y_states[0] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j - 1, k, Y);
            y_states[1] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j, k, Y);
            y_states[2] = Solver::Local::FluidRiemann::State(rho, M, E_vol, gammaf, p0_eff, T, i, j + 1, k, Y);

            std::vector<Solver::Local::FluidRiemann::State> x_leftStates(3), x_rightStates(3);
            std::vector<Solver::Local::FluidRiemann::State> y_leftStates(3), y_rightStates(3);

            if (Limiter == 0)
            {
                x_leftStates[1] = x_states[0];
                x_rightStates[1] = x_states[1];
                x_leftStates[2] = x_states[1];
                x_rightStates[2] = x_states[2];
                y_leftStates[1] = y_states[0];
                y_rightStates[1] = y_states[1];
                y_leftStates[2] = y_states[1];
                y_rightStates[2] = y_states[2];
            }

            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;

            try
            {
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
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::Abort(INFO);
            }

            // COMPUTE TIME DERIVATIVES (not state updates!)
            rho_flux(i, j, k) = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]);
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]);
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]);
            E_flux(i, j, k) = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]);

            // Store derivatives (flux divergence + sources)
            drho_dt(i, j, k, 0) = rho_flux(i, j, k) + Source(i, j, k, 0);
            dM_dt(i, j, k, 0) = M_flux(i, j, k, 0) + Source(i, j, k, 1);
            dM_dt(i, j, k, 1) = M_flux(i, j, k, 1) + Source(i, j, k, 2);
            dE_dt(i, j, k, 0) = E_flux(i, j, k) + Source(i, j, k, 3);

            // Eta time derivative
            Set::Scalar Mob = a(i, j, k) * 0.7 * DX[0];
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            Set::Vector u_current = Set::Vector(M(i, j, k, 0) / (rho(i, j, k)), M(i, j, k, 1) / (rho(i, j, k)));
            Set::Scalar advection = -u_current.dot(grad_eta);
            Set::Scalar diffusion = 0.0;
            Set::Scalar eta_source = advection + diffusion;

            if (apply_vaporization == 1)
            {
                eta_source += (1.0 / (rho(i, j, k) * epsilon)) * (rho0(i, j, k) * Dv * (Bm(i, j, k) / (1.0 + Bm(i, j, k) + small)) * grad_eta_mag);
            }

            if (static_eta == 1)
            {
                deta_dt(i, j, k, 0) = 0.0;
            }
            else
            {
                deta_dt(i, j, k, 0) = eta_source;
            }

            // Adaptive timestep tracking (same as before)
            Set::Scalar A_current = (eta(i, j, k)) / (gamma0 - 1.0) + (1.0 - eta(i, j, k)) / (gamma1 - 1.0);
            Set::Scalar B_current = (eta(i, j, k) * gamma0 * p0_0) / (gamma0 - 1.0) + ((1.0 - eta(i, j, k)) * gamma1 * p0_1) / (gamma1 - 1.0);
            Set::Scalar gamma_eff_current = 1.0 + (1.0 / A_current);
            Set::Scalar sound_speed_current = std::sqrt(gamma_eff_current * (press(i, j, k) + p0_eff(i, j, k)) / (rho(i, j, k)));

            c_max = std::max(c_max, sound_speed_current);
            vx_max = std::max(vx_max, std::abs(u(0)));
            vy_max = std::max(vy_max, std::abs(u(1)));

            Set::Scalar F_mag = std::sqrt(Source(i, j, k, 1) * Source(i, j, k, 1) + Source(i, j, k, 2) * Source(i, j, k, 2));
            F_max = std::max(F_max, F_mag);
            rho_min = std::min(rho_min, rho(i, j, k));
        });
    }

    // Adaptive timestep calculation (same as before)
    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);
    amrex::ParallelDescriptor::ReduceRealMax(F_max);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min);

    Set::Scalar dx_min = std::min(DX[0], DX[1]);
    Set::Scalar wave_speed = c_max + std::sqrt(vx_max * vx_max + vy_max * vy_max);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);
    Set::Scalar mu_max = std::max(mu0, mu1);
    Set::Scalar dt_viscous = cfl_v * rho_min * dx_min * dx_min / (mu_max + small);
    Set::Scalar a_max = F_max / rho_min;
    Set::Scalar dt_force = cfl_v * std::sqrt(dx_min / (a_max + small));
    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn }) * 0.9;
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
