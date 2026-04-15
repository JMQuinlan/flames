// Base
#include "Hydro2.H"
// Parsing and Input Handeling
#include "AMReX_MultiFab.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "BC/Nothing.H"
#include "BC/NSCBC.H"
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
        pp_query_default("nghost", value.nghost, 2);

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
            // NSCBC mode: Initialize NSCBC handler and use Nothing BCs
            value.nscbc_bc = new BC::NSCBC(pp);

            // Use BC::Nothing (does nothing when called)
            value.density_bc = &value.bc_nothing;
            value.energy_bc = &value.bc_nothing;
            value.momentum_bc = &value.bc_nothing;

            Util::Message(INFO, "Parsing NSCBC");
            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);

        }
        else
        {
            // Standard mode: Use Expression BCs
            value.nscbc_bc = nullptr;

            value.density_bc = new BC::Expression(1, pp, "density.bc");
            value.energy_bc = new BC::Constant(1, pp, "energy.bc");
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
        value.RegisterNewFab(value.eta_mf,           value.density_bc, 1, nghost, "eta", true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.density_bc, 1, nghost, "eta_old", false, true);
        value.RegisterNewFab(value.rho_eta0_mf,      value.density_bc, 1, nghost, "rho_eta0", true, false);
        value.RegisterNewFab(value.rho_eta1_mf,      value.density_bc, 1, nghost, "rho_eta1", true, false);
        value.RegisterNewFab(value.rho_eta0_old_mf,  value.density_bc, 1, nghost, "rho_eta0_old", false, false);
        value.RegisterNewFab(value.rho_eta1_old_mf,  value.density_bc, 1, nghost, "rho_eta1_old", false, false);

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
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing,  1, nghost, "m0", false, false);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing, 2, nghost, "u0", false, false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing, 2, nghost, "q0", false, false, { "x", "y" });
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
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small);

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number
            Bm(i, j, k) = (Y(i, j, k) - Y_infinity) / (1 + Y_infinity + small);
            
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
    amrex::MultiFab &rho_eta0_rhs_mf, 
    amrex::MultiFab &rho_eta1_rhs_mf, 
    amrex::MultiFab &M_rhs_mf, 
    amrex::MultiFab &E_rhs_mf, 
    const amrex::MultiFab &rho_eta0_mf_in,
    const amrex::MultiFab &rho_eta1_mf_in,
    const amrex::MultiFab &M_mf_in, 
    const amrex::MultiFab &E_mf_in) //, const amrex::MultiFab &velocity_mf_in, const amrex::MultiFab &pressure_mf_in, const amrex::MultiFab &T_mf_in)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Eta Fields
    // for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // CONSERVATIVE
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf_in.array(mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);

            // eta(i, j, k) = (rho(i, j, k) - rho_eta1(i, j, k)) / (rho_eta0(i, j, k) - rho_eta1(i, j, k) + small);
            eta(i, j, k) = rho_eta0(i, j, k) / rho(i, j, k);

            // CUTOFFS
            eta(i, j, k) = std::max(0.0, std::min(1.0, (eta(i, j, k) - cutoff) / (1.0 - 2.0 * cutoff)));
            
        });
    }

    if (nscbc_bc != nullptr)
    {
        FillBoundaries(lev, { 
            eta_mf[lev].get(), 
            density_mf[lev].get(), 
            rho_eta0_mf[lev].get(), 
            rho_eta1_mf[lev].get() });
    }
    else
    {
        // Copy interior + ghost cells TO working copies
        amrex::MultiFab rho_eta0_copy(rho_eta0_mf_in.boxArray(), rho_eta0_mf_in.DistributionMap(), 1, nghost);
        amrex::MultiFab rho_eta1_copy(rho_eta1_mf_in.boxArray(), rho_eta1_mf_in.DistributionMap(), 1, nghost);
        amrex::MultiFab M_copy(M_mf_in.boxArray(), M_mf_in.DistributionMap(), AMREX_SPACEDIM, nghost);
        amrex::MultiFab E_copy(E_mf_in.boxArray(), E_mf_in.DistributionMap(), 1, nghost);


        amrex::MultiFab::Copy(rho_eta0_copy, rho_eta0_mf_in, 0, 0, 1, nghost); // Include ghosts
        amrex::MultiFab::Copy(rho_eta1_copy, rho_eta1_mf_in, 0, 0, 1, nghost); // Include ghosts
        amrex::MultiFab::Copy(M_copy, M_mf_in, 0, 0, AMREX_SPACEDIM, nghost);  // Include ghosts
        amrex::MultiFab::Copy(E_copy, E_mf_in, 0, 0, 1, nghost);               // Include ghosts
        

        FillBoundariesWithBC(lev, time, density_bc, { 
            eta_mf[lev].get(), 
            density_mf[lev].get() 
            //&rho_eta0_copy,
            //&rho_eta1_copy
        });

        FillBoundariesWithBC(lev, time, momentum_bc, { 
            &M_copy
        });

        FillBoundariesWithBC(lev, time, energy_bc, { 
            &E_copy
        });

        // Copy back INCLUDING ALL 4 GHOST CELL LAYERS
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(rho_eta0_mf_in), rho_eta0_copy, 0, 0, 1, nghost);
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(rho_eta1_mf_in), rho_eta1_copy, 0, 0, 1, nghost);
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(M_mf_in), M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(E_mf_in), E_copy, 0, 0, 1, nghost);
        
        /*
        FillBoundariesWithBC(lev, time, density_bc, { 
            eta_mf[lev].get(), 
            density_mf[lev].get(), 
            rho_eta0_mf[lev].get(),
            rho_eta1_mf[lev].get() 
        });
        */
        

        
    }
        

    // Primitive Fields
    //for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
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

            // gamma
            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
            
            // Velocity
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k));
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k));

            // Kinetic Energy
            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1)); // Per Vol
            // KE(i, j, k) = 0.5 * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1)); // Per Mass

            // Potential Energy
            UE(i, j, k) = E(i, j, k) - KE(i, j, k);

            // Pressure
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small);

            // Temperature
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Speed of sound:
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gammaf(i, j, k), p0_eff(i, j, k), small);

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            //Set::Scalar mu_chem = -epsilon * lap_eta + f_prime / epsilon;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));
            
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


    // NSCBC SPECIFIC BOUNDRY CONDITION
    if (nscbc_bc != nullptr)
    {
        Util::Message(INFO, "Using NSCBC");

        // Compute total density from phase densities
        amrex::MultiFab rho_total(rho_eta0_mf_in.boxArray(),
                                  rho_eta0_mf_in.DistributionMap(),
                                  1,
                                  nghost);

        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.growntilebox(nghost);
            auto rho = rho_total.array(mfi);
            auto rho0 = rho_eta0_mf_in.const_array(mfi);
            auto rho1 = rho_eta1_mf_in.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }

        // Create working copies for NSCBC
        amrex::MultiFab M_copy(M_mf_in.boxArray(), M_mf_in.DistributionMap(), AMREX_SPACEDIM, nghost);
        amrex::MultiFab E_copy(E_mf_in.boxArray(), E_mf_in.DistributionMap(), 1, nghost);

        amrex::MultiFab::Copy(M_copy, M_mf_in, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(E_copy, E_mf_in, 0, 0, 1, nghost);

        // Apply NSCBC
        nscbc_bc->FillBoundary(rho_total,
                               M_copy,
                               E_copy,
                               *eta_mf[lev],
                               *gamma_mf[lev],
                               *p0_mf[lev],
                               *pressure_mf[lev],
                               eos0,
                               eos1,
                               geom[lev],
                               time,
                               pref);

        // Copy back
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(M_mf_in), M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(const_cast<amrex::MultiFab &>(E_mf_in), E_copy, 0, 0, 1, nghost);

        // Update density_mf from rho_total
        density_mf[lev]->ParallelCopy(rho_total);


        // Compute primitive variables in ghost cells from updated conservative variables
        for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghost_box = mfi.growntilebox(nghost);

            auto eta = eta_mf[lev]->array(mfi);
            auto rho = density_mf[lev]->array(mfi);
            auto rho_eta0 = const_cast<amrex::MultiFab &>(rho_eta0_mf_in).array(mfi);
            auto rho_eta1 = const_cast<amrex::MultiFab &>(rho_eta1_mf_in).array(mfi);
            auto M = const_cast<amrex::MultiFab &>(M_mf_in).array(mfi);
            auto E = const_cast<amrex::MultiFab &>(E_mf_in).array(mfi);
            auto v = velocity_mf[lev]->array(mfi);
            auto press = pressure_mf[lev]->array(mfi);
            auto T = T_mf[lev]->array(mfi);
            auto a = a_mf[lev]->array(mfi);
            auto gammaf = gamma_mf[lev]->array(mfi);
            auto p0_eff = p0_mf[lev]->array(mfi);
            auto UE = UE_per_vol_mf[lev]->array(mfi);
            auto KE = KE_per_vol_mf[lev]->array(mfi);

            const Solver::EOS::Tammann eos0_local = eos0;
            const Solver::EOS::Tammann eos1_local = eos1;

            amrex::ParallelFor(ghost_box, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // rho_eta Fields
                rho_eta0(i, j, k) = rho(i, j, k) * eta(i, j, k);
                rho_eta1(i, j, k) = rho(i, j, k) * (1.0 - eta(i, j, k));

                // Velocity from momentum
                v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k) + small);
                v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k) + small);

                // Kinetic energy
                KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

                // Internal energy
                UE(i, j, k) = E(i, j, k) - KE(i, j, k);

                // EOS properties
                gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
                p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
                press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small);
                T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);
                a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gammaf(i, j, k), p0_eff(i, j, k), small);
            });
        }
        ///  VALIDATION CHECK ///////////////////////////////
        if (step_counter[lev] == 1)
        { // First timestep only
            for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                const amrex::Box &domain_box = geom[lev].Domain();

                // Check X-LO boundary ghost cells
                if (bx.smallEnd(0) == domain_box.smallEnd(0))
                {
                    auto rho = density_mf[lev]->array(mfi);
                    auto M = const_cast<amrex::MultiFab &>(M_mf_in).array(mfi);
                    auto E = const_cast<amrex::MultiFab &>(E_mf_in).array(mfi);

                    // Sample one ghost cell at mid-height
                    int i_ghost = domain_box.smallEnd(0) - 1;
                    int j_mid = (bx.smallEnd(1) + bx.bigEnd(1)) / 2;

                    Set::Scalar rho_g = rho(i_ghost, j_mid, 0);
                    Set::Scalar Mx_g = M(i_ghost, j_mid, 0, 0);
                    Set::Scalar My_g = M(i_ghost, j_mid, 0, 1);
                    Set::Scalar E_g = E(i_ghost, j_mid, 0);
                    Set::Scalar vx_g = Mx_g / (rho_g + small);
                    Set::Scalar vy_g = My_g / (rho_g + small);

                    Util::Message(INFO, "=== NSCBC X-LO GHOST CHECK ===");
                    Util::Message(INFO, "Ghost cell (", i_ghost, ",", j_mid, "):");
                    Util::Message(INFO, "  rho=", rho_g);
                    Util::Message(INFO, "  Mx=", Mx_g, " My=", My_g);
                    Util::Message(INFO, "  E=", E_g);
                    Util::Message(INFO, "  vx=", vx_g, " vy=", vy_g);

                    // Compare to interior neighbor
                    int i_interior = domain_box.smallEnd(0);
                    Set::Scalar rho_i = rho(i_interior, j_mid, 0);
                    Set::Scalar vx_i = M(i_interior, j_mid, 0, 0) / (rho_i + small);
                    Set::Scalar vy_i = M(i_interior, j_mid, 0, 1) / (rho_i + small);

                    Util::Message(INFO, "Interior cell (", i_interior, ",", j_mid, "):");
                    Util::Message(INFO, "  rho=", rho_i);
                    Util::Message(INFO, "  vx=", vx_i, " vy=", vy_i);

                    // FAIL CONDITIONS
                    if (rho_g < 0.0 || rho_g > 10000.0)
                    {
                        Util::Message(INFO, "FAIL: Ghost density out of range!");
                    }
                    if (std::abs(vx_g) > 1000.0 || std::abs(vy_g) > 1000.0)
                    {
                        Util::Message(INFO, "FAIL: Ghost velocity too large!");
                    }
                    if (E_g < 0.0 || E_g > 1e10)
                    {
                        Util::Message(INFO, "FAIL: Ghost energy out of range!");
                    }

                    // EXPECTED: Ghost should be similar to interior (for outflow)
                    Set::Scalar rho_diff = std::abs(rho_g - rho_i) / (rho_i + small);
                    if (rho_diff > 0.5)
                    {
                        Util::Message(INFO, "WARNING: Ghost density differs by ", rho_diff * 100, "%");
                    }
                }
            }
        }

        ///////////////////////////////

    }
    else
    {
        // Primative Field Boundries
        FillBoundariesWithBC(lev, time, energy_bc, { 
            pressure_mf[lev].get(), 
            T_mf[lev].get(), 
            gamma_mf[lev].get(),
            p0_mf[lev].get() 
        });

        /*
        FillBoundariesWithBC(lev, time, energy_bc, { 
            energy_per_vol_mf[lev].get(), 
            energy_per_mas_mf[lev].get(), 
            energy_per_vol_old_mf[lev].get(), 
            energy_per_mas_old_mf[lev].get(),
            UE_per_vol_mf[lev].get(),
            UE_per_mas_mf[lev].get(), 
            KE_per_vol_mf[lev].get(), 
            KE_per_mas_mf[lev].get(), 
            pressure_mf[lev].get(), 
            gamma_mf[lev].get(), 
            p0_mf[lev].get(), 
            mu_chem_mf[lev].get(), 
            Bm_mf[lev].get(),
            Y_mf[lev].get(),
            T_mf[lev].get(),
            cp_mf[lev].get(),
            cv_mf[lev].get()
        });
        */

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

        // Mixture
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = M_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> E = E_mf_in.array(mfi);

        // OUTPUTS
        Set::Patch<Set::Scalar> rho_eta0_rhs = rho_eta0_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_eta1_rhs = rho_eta1_rhs_mf.array(mfi);
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
            Set::Scalar eta_dot_AC = advection + diffusion;
            */

            // ------------------------------------------------------------
            // Cahn–Hilliard
            // ------------------------------------------------------------
            // d(eta)/dt = -u·grad(eta) + div( M*grad(mu) )
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
                Set::Scalar Y_vs = Y(i, j, k); // rho_eta0(i, j, k) / (rho0(i, j, k) + rho1(i, j, k));

                // Spalding mass transfer number
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // Gas density from fluid 0 (eta=1 corresponds to fluid 0)
                Set::Scalar rho_g = rho_eta0(i, j, k) / std::max(eta(i, j, k), small);

                // Scaling density choice: using rho_eta = rho_g makes RHS independent of mixture density
                // This is consistent with the document recommendation
                // Set::Scalar rho_eta = rho_g;
                //Set::Scalar rho_eta = rho(i, j, k);//rho_g;

                // Vaporization source for eta equation (Equation 7):
                // source_vap = (1/epsilon) * (rho_g * D_v / rho_eta) * (B_M/(1+B_M)) * |grad(eta)|
                Set::Scalar vap_coeff = (rho_g * Dv / (rho_eta0(i, j, k) + small)) * (B_M / (1.0 + B_M + small));
                eta_dot_Vap = (1.0 / epsilon) * vap_coeff * grad_eta_mag;

                // FLUXES
                m_dot_Vap = rho_g * Dv * (B_M / (1.0 + B_M + small)) * grad_eta_mag; // Mass Flux
                M_dot_Vap = u * m_dot_Vap * grad_eta_mag;                            // Momentum Flux
                E_dot_Vap = u.dot(M_dot_Vap) * grad_eta_mag;                         // Energy Flux

                /*
                m_dot_Vap = m_dot_Vap * (1.0 / epsilon);                             // Mass Flux
                M_dot_Vap = M_dot_Vap * (1.0 / epsilon); // Momentum Flux
                E_dot_Vap = E_dot_Vap * (1.0 / epsilon);                             // Energy Flux
                */
                m_dot_Vap = m_dot_Vap;
                M_dot_Vap = M_dot_Vap;
                E_dot_Vap = E_dot_Vap;
                
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
            
            Source(i, j, k, 0) = mdot0 + m_dot_Vap;
            /*
            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0);// + M_dot_Vap(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1);// + M_dot_Vap(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force);// + E_dot_Vap;
            */

            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0) + M_dot_Vap(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1) + M_dot_Vap(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force) + E_dot_Vap;

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

            // Upwind volume fractions
            Set::Scalar eta_face_xlo = (flux_xlo.u_interface > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
            Set::Scalar eta_face_xhi = (flux_xhi.u_interface > 0.0) ? eta(i, j, k) : eta(i + 1, j, k);
            Set::Scalar eta_face_ylo = (flux_ylo.u_interface > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);
            Set::Scalar eta_face_yhi = (flux_yhi.u_interface > 0.0) ? eta(i, j, k) : eta(i, j + 1, k);


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

            rho_eta0_rhs(i, j, k) = rho_eta0_flux + Source(i, j, k, 0) * (eta(i, j, k));
            rho_eta1_rhs(i, j, k) = rho_eta1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k));
            
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
    std::swap(eta_old_mf, eta_mf);
    std::swap(rho_eta0_old_mf, rho_eta0_mf);
    std::swap(rho_eta1_old_mf, rho_eta1_mf);

    // ------------------------------------------------------------
    // Time Integration
    // ------------------------------------------------------------

    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*rho_eta0_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_eta1_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(), amrex::MakeType::make_alias, 0, 2);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*rho_eta0_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_eta1_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 2);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        RHS(lev, time, rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3]);
    });

    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        if (nscbc_bc != nullptr)
        {
            // Compute total density: rho = rho_eta0 + rho_eta1
            amrex::MultiFab rho_total(stage_mf[0].boxArray(), stage_mf[0].DistributionMap(), 1, nghost);

            for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.growntilebox(nghost);
                auto rho_arr = rho_total.array(mfi);
                auto rho_eta0_arr = stage_mf[0].array(mfi);
                auto rho_eta1_arr = stage_mf[1].array(mfi);

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    rho_arr(i, j, k) = rho_eta0_arr(i, j, k) + rho_eta1_arr(i, j, k);
                });
            }

            // Compute eta = rho_eta0 / rho_total
            amrex::MultiFab eta_stage(stage_mf[0].boxArray(), stage_mf[0].DistributionMap(), 1, nghost);

            for (amrex::MFIter mfi(eta_stage); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.growntilebox(nghost);
                auto eta_arr = eta_stage.array(mfi);
                auto rho_eta0_arr = stage_mf[0].array(mfi);
                auto rho_arr = rho_total.array(mfi);
                Set::Scalar small_local = small;

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    eta_arr(i, j, k) = rho_eta0_arr(i, j, k) / (rho_arr(i, j, k) + small_local);
                    eta_arr(i, j, k) = std::max(0.0, std::min(1.0, eta_arr(i, j, k)));
                });
            }

            // ACTUALLY CALL NSCBC
            nscbc_bc->FillBoundary(rho_total,
                                   stage_mf[2], // momentum
                                   stage_mf[3], // energy
                                   eta_stage,
                                   *gamma_mf[lev],
                                   *p0_mf[lev],
                                   *pressure_mf[lev],
                                   eos0,
                                   eos1,
                                   geom[lev],
                                   time,
                                   pref);

            // Update rho_eta0 and rho_eta1 from modified rho_total
            for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.growntilebox(nghost);
                auto rho_arr = rho_total.array(mfi);
                auto eta_arr = eta_stage.array(mfi);
                auto rho_eta0_arr = stage_mf[0].array(mfi);
                auto rho_eta1_arr = stage_mf[1].array(mfi);

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    rho_eta0_arr(i, j, k) = rho_arr(i, j, k) * eta_arr(i, j, k);
                    rho_eta1_arr(i, j, k) = rho_arr(i, j, k) * (1.0 - eta_arr(i, j, k));

                    // Enforce positivity
                    rho_eta0_arr(i, j, k) = std::max(rho_eta0_arr(i, j, k), small);
                    rho_eta1_arr(i, j, k) = std::max(rho_eta1_arr(i, j, k), small);
                });
            }
        }
        else
        {
            // Standard mode: Apply standard BCs
            density_bc->FillBoundary(stage_mf[0], 0, 1, time, 0);
            stage_mf[0].FillBoundary(true);
            density_bc->FillBoundary(stage_mf[1], 0, 1, time, 0);
            stage_mf[1].FillBoundary(true);
            momentum_bc->FillBoundary(stage_mf[2], 0, AMREX_SPACEDIM, time, 0);
            stage_mf[2].FillBoundary(true);
            energy_bc->FillBoundary(stage_mf[3], 0, 1, time, 0);
            stage_mf[3].FillBoundary(true);
        }

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

            rho(i, j, k) = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            eta_new(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k) + small);

            eta_new(i, j, k) = std::max(0.0, std::min(1.0, (eta_new(i, j, k) - cutoff) / (1.0 - 2.0 * cutoff)));
            rho(i, j, k) = std::max(rho(i, j, k), small);

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

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
    
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
        
            KE_vol(i,j,k) = 0.5 * rho(i,j,k) * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
            KE_mas(i,j,k) = 0.5 * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
        
            UE_vol(i,j,k) = E_vol(i,j,k) - KE_vol(i,j,k);
            E_mas(i,j,k) = E_vol(i,j,k) / (rho(i,j,k) + small);
            UE_mas(i,j,k) = E_mas(i,j,k) - KE_mas(i,j,k);
        
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small);
        
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

    Source_mf[lev]->setVal(0.0);

    if (lev < finest_level)
        return;

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
    FillBoundariesWithBC(lev, 0.0, density_bc, { &psi_mf });

    // ============================================================================
    // STEP 2: Reinitialize psi to signed distance function (Equation 10)
    // d(psi)/d(tau) = S(psi) * (1 - |grad(psi)|)
    // ============================================================================
    amrex::MFIter::allowMultipleMFIters(true);
    ReinitializeSignedDistance(lev, psi_reinit_mf, psi_mf, reinit_max_iter);
    amrex::MFIter::allowMultipleMFIters(false);

    // Fill boundaries for reinitialized psi
    FillBoundariesWithBC(lev, 0.0, density_bc, { &psi_reinit_mf });

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

    FillBoundariesWithBC(lev, 0.0, density_bc, { &phi_sharp_mf });

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
        /*
        if (mf->contains_nan())
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundaries");
            Util::Abort(INFO);
        }
        */
    }
}

// FillBoundariesWithBC(): Fill Boundries with BC
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
        /*
        if (mf->contains_nan())
        {
            Util::Paralle lMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundariesWithBC");
            Util::Abort(INFO);
        }
        */
    }
}
















} // end of Integrator namespace

//#endif
