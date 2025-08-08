
#include "Hydro2.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "Numeric/Stencil.H"
#include "IC/Constant.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
#include "Solver/Local/Riemann/Roe.H"
#include "Solver/Local/Riemann/HLLC.H"
#include "Solver/Local/Riemann/HLLC_WENO5.H"
#include "Solver/Local/Riemann/HLLE.H"
#include "Solver/Local/Riemann/HLLCE.H"

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
        pp_query_required("cfl", value.cfl);           // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl); // cfl condition
        pp_query_default("pref", value.pref, 1.0);     // reference pressure for Roe solver
        pp_query_default("small", value.small, 1E-8);  // small regularization value
        pp_query_default("cutoff", value.cutoff, 1E-100);  // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0); // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81); // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix"); // Roe solver entropy fix
        pp_query_default("scheme", value.scheme, 0); // 0: Forward Euler | 1: RK4

        // ADAPTIVE TIMESTEP
        pp_query_default("adaptive_timestep", value.adaptive_timestep, false); // Gravitational Acceletation
        pp_query_default("dt_min", value.dt_min, 1E-9);
        pp_query_default("dt_max", value.dt_max, 1E-3);
        pp_query_default("dt_growth", value.dt_growth, 1.2);

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, true); // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("apply_buoyancy", value.apply_buoyancy, false);              // Apply buoyancy when solving, default: false --> "No Buoyancy"
        pp_query_default("apply_weight", value.apply_weight, false);                  // Apply weight when solving, default: false --> "No Weight"

        // NEW SOLVER FORBIDS
        pp_forbid("gamma", "--> gamma0 and gamma1");
        pp_forbid("mu", "--> mu0 and mu1");

        // FLUID 0
        pp_query_required("gamma0", value.gamma0);      // gamma for gamma law
        pp_query_default("p0_0", value.p0_0, 0.0);           // p0 for Tammann EOS
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient

        // FLUID 1
        pp_query_required("gamma1", value.gamma1);      // gamma for gamma law
        pp_query_default("p0_1", value.p0_1, 0.0);           // p0 for Tammann EOS
        pp_query_required("mu1", value.mu1);            // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0);    // bulk viscosity coefficient

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 70.0); // surface tension condition
        pp_query_required("epsilon", value.epsilon); // diffuse interface thickness

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
        value.eta_bc = new BC::Constant(1, pp, "pf.eta.bc");

        
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = 2;

        // DIFFUSSIVE PARAMETERS
        value.RegisterNewFab(value.eta_mf,     value.eta_bc, 1, nghost, "eta",     true );
        value.RegisterNewFab(value.eta_old_mf, value.eta_bc, 1, nghost, "eta_old", true);
        value.RegisterNewFab(value.etadot_mf,  value.eta_bc, 1, nghost, "etadot",  true );
        value.RegisterNewFab(value.hess_eta_mf, &value.bc_nothing, 4, nghost, "hess_eta", true, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf, &value.bc_nothing, 2, nghost, "n_hat", true, { "x", "y" });

        // FLUID 0
        value.RegisterNewFab(value.density0_mf,     value.density_bc, 1, nghost, "density0",     false );
        value.RegisterNewFab(value.density0_old_mf, value.density_bc, 1, nghost, "density0_old", false);

        value.RegisterNewFab(value.energy0_mf, value.energy_bc, 1, nghost, "energy0", false);
        value.RegisterNewFab(value.energy0_old_mf, value.energy_bc, 1, nghost, "energy0_old" , false);

        value.RegisterNewFab(value.momentum0_mf, value.momentum_bc, 2, nghost, "momentum0", false, { "x", "y" });
        value.RegisterNewFab(value.momentum0_old_mf, value.momentum_bc, 2, nghost, "momentum0_old", false);
 
        value.RegisterNewFab(value.pressure0_mf, &value.bc_nothing, 1, nghost, "pressure0", false);
        value.RegisterNewFab(value.velocity0_mf, &value.bc_nothing, 2, nghost, "velocity0", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity0_mf, &value.bc_nothing, 1, nghost, "vorticity0", false);

        // FLUID 1
        value.RegisterNewFab(value.density1_mf, value.density_bc, 1, nghost, "density1", false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc, 1, nghost, "density1_old", false);

        value.RegisterNewFab(value.energy1_mf, value.energy_bc, 1, nghost, "energy1", false);
        value.RegisterNewFab(value.energy1_old_mf, value.energy_bc, 1, nghost, "energy1_old", false);

        value.RegisterNewFab(value.momentum1_mf, value.momentum_bc, 2, nghost, "momentum1", false, { "x", "y" });
        value.RegisterNewFab(value.momentum1_old_mf, value.momentum_bc, 2, nghost, "momentum1_old", false);

        value.RegisterNewFab(value.pressure1_mf, &value.bc_nothing, 1, nghost, "pressure1", false);
        value.RegisterNewFab(value.velocity1_mf, &value.bc_nothing, 2, nghost, "velocity1", false, { "x", "y" });
        value.RegisterNewFab(value.vorticity1_mf, &value.bc_nothing, 1, nghost, "vorticity1", false);

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf, &value.bc_nothing, 1, nghost, "pressure", true);
        value.RegisterNewFab(value.velocity_mf, &value.bc_nothing, 2, nghost, "velocity", true, { "x", "y" });
        value.RegisterNewFab(value.vorticity_mf, &value.bc_nothing, 1, nghost, "vorticity", true);
        value.RegisterNewFab(value.density_mf, value.density_bc, 1, nghost, "density", true);
        value.RegisterNewFab(value.density_old_mf, value.density_bc, 1, nghost, "density_old", false);
        value.RegisterNewFab(value.energy_mf, value.energy_bc, 1, nghost, "energy", true);
        value.RegisterNewFab(value.energy_old_mf, value.energy_bc, 1, nghost, "energy_old", false);
        value.RegisterNewFab(value.momentum_mf, value.momentum_bc, 2, nghost, "momentum", true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc, 2, nghost, "momentum_old", false);

        // SOURCE
        value.RegisterNewFab(value.m0_mf, &value.bc_nothing, 1, 0, "m0", true);
        value.RegisterNewFab(value.u0_mf, &value.bc_nothing, 2, 0, "u0", true, { "x", "y" });
        value.RegisterNewFab(value.q_mf, &value.bc_nothing, 2, 0, "q0", true, { "x", "y" });
        value.RegisterNewFab(value.Source_mf, &value.bc_nothing, 4, 0, "Source", true);
        value.RegisterNewFab(value.Fsv_mf, &value.bc_nothing, 2, nghost, "Fsv", true, { "x", "y" }); // To Track Surface Tension
        value.RegisterNewFab(value.Fb_mf, &value.bc_nothing, 2, nghost, "Fb", true, { "x", "y" }); // To Track Bouyancy
        value.RegisterNewFab(value.Fw_mf, &value.bc_nothing, 2, nghost, "Fw", true, { "x", "y" }); // To Track Weight
        value.RegisterNewFab(value.kappas_mf, &value.bc_nothing, 3, nghost, "kappa", true, { "Avg", "1", "2" }); // To Surface curvature


        // DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_eta", true, { "x", "y" });
    }

    // NEW SOLVER FORBIDS
    pp_forbid("velocity.ic.type", "--> velocity0.ic.type or velocity1.ic.type");
    pp_forbid("pressure.ic", "--> pressure0.ic or pressure1.ic");
    pp_forbid("density.ic.type", "--> density0.ic.type or density1.ic.type");


    // INITIAL CONDITIONS
    // Eta
    pp.select_default<IC::Constant,IC::Laminate,IC::Expression,IC::BMP,IC::PNG>("eta.ic",value.eta_ic,value.geom);
    // Fluid 0
    pp.select_default<IC::Constant,IC::Expression>("velocity0.ic",value.velocity0_ic,value.geom);
    pp.select_default<IC::Constant,IC::Expression>("pressure0.ic",value.pressure0_ic,value.geom);
    pp.select_default<IC::Constant,IC::Expression>("density0.ic",value.density0_ic,value.geom);
    // Fluid 1
    pp.select_default<IC::Constant, IC::Expression>("velocity1.ic", value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("pressure1.ic", value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant, IC::Expression>("density1.ic", value.density1_ic, value.geom);


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
        pp.select_default<Solver::Local::Riemann::Roe>("solver", value.roesolver);
    }
    else if (value.Riemann_Solver == 1)
    {
        pp.select_default<Solver::Local::Riemann::HLLC>("solver", value.hllcsolver);
    }
    else if (value.Riemann_Solver == 2)
    {
        pp.select_default<Solver::Local::Riemann::HLLE>("solver", value.hllesolver);
    }
    else if (value.Riemann_Solver == 3)
    {
        pp.select_default<Solver::Local::Riemann::HLLCE>("solver", value.hllcesolver);
    }
    else if (value.Riemann_Solver == 35)
    {
        pp.select_default<Solver::Local::Riemann::HLLC_WENO5>("solver", value.hllc_weno5solver);
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

    // FLUID 1
    velocity1_ic    ->Initialize(lev, velocity1_mf, 0.0);
    pressure1_ic    ->Initialize(lev, pressure1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_mf, 0.0);
    density1_ic     ->Initialize(lev, density1_old_mf, 0.0);

    // SOURCE
    ic_m0           ->Initialize(lev, m0_mf, 0.0);
    ic_u0           ->Initialize(lev, u0_mf, 0.0);
    ic_q            ->Initialize(lev, q_mf, 0.0);
    Source_mf[lev]  ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    Fb_mf[lev]      ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    Fw_mf[lev]      ->setVal(0.0); //->Initialize(lev, m0_mf, 0.0);
    kappas_mf[lev]  ->setVal(0.0);

    // ADAPTIVE TIMESTEP
    if (adaptive_timestep)
    {
        dynamictimestep.on = true;
        dynamictimestep.cfl = cfl;
        dynamictimestep.min = dt_min;
        dynamictimestep.max = dt_max;
        dynamictimestep.nprevious = 3; // Use 3 previous timesteps for averaging

        amrex::Print() << "Dynamic timestepping enabled with:\n"
                       << "  CFL = " << dynamictimestep.cfl << "\n"
                       << "  Min dt = " << dynamictimestep.min << "\n"
                       << "  Max dt = " << dynamictimestep.max << "\n"
                       << "  Growth factor = " << dt_growth << "\n"
                       << "  History length = " << dynamictimestep.nprevious << std::endl;
    }
    else
    {
        dynamictimestep.on = false;
    }

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

        // FLUID 1
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1          = energy1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_old      = energy1_old_mf.Patch(lev, mfi);

        // Mixture Initalization
        Set::Patch<Set::Scalar>         v           = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         press       = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho         = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         rho_old     = density_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M           = momentum_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         M_old       = momentum_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E           = energy_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E_old       = energy_old_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

            // Calculate Mixtures 
            rho(i, j, k) = eta(i, j, k) * rho0(i, j, k) + (1.0 - eta(i, j, k)) * rho1(i, j, k);
            rho_old(i, j, k) = rho(i, j, k);  

            M(i, j, k, 0) = (rho0(i, j, k) * v0(i, j, k, 0)) * eta(i, j, k) + (rho1(i, j, k) * v1(i, j, k, 0)) * (1.0 - eta(i, j, k));
            M(i, j, k, 1) = (rho0(i, j, k) * v0(i, j, k, 1)) * eta(i, j, k) + (rho1(i, j, k) * v1(i, j, k, 1)) * (1.0 - eta(i, j, k));
            M_old(i, j, k, 0) = M(i, j, k, 0);
            M_old(i, j, k, 1) = M(i, j, k, 1);

            //  TODO: Get rid of thermally perfect assumption. Involve temperature
            E(i, j, k) = (0.5 * ((v0(i, j, k, 0) * v0(i, j, k, 0)) + (v0(i, j, k, 1) * v0(i, j, k, 1))) * rho0(i, j, k) + p0(i, j, k) / (gamma0 - 1.0)) * eta(i, j, k)
                       + (0.5 * ((v1(i, j, k, 0) * v1(i, j, k, 0)) + (v1(i, j, k, 1) * v1(i, j, k, 1))) * rho1(i, j, k) + p1(i, j, k) / (gamma1 - 1.0)) * (1.0 - eta(i, j, k));
            E_old(i, j, k) = E(i, j, k);

            v(i, j, k, 0) = v0(i,j,k,0)*eta(i,j,k) + v1(i,j,k,0)*(1.0-eta(i,j,k)); 
            v(i, j, k, 1) = v0(i,j,k,1)*eta(i,j,k) + v1(i,j,k,1)*(1.0-eta(i,j,k)); 

            press(i, j, k) = p0(i,j,k)*eta(i,j,k) + p1(i,j,k)*(1.0-eta(i,j,k));

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
    // OLD METHOD
    /*
    Integrator::DynamicTimestep_Update();

    return;

    const Set::Scalar* DX = geom[lev].CellSize();

    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);

    Set::Scalar new_timestep = cfl / ((c_max + vx_max) / (DX[0]+small) + (c_max + vy_max) / (DX[1]+small) + small);

    Util::Assert(INFO, TEST(AMREX_SPACEDIM == 2));

    SetTimestep(new_timestep);
    */
    
    /*
    // OLD NEW METHOD
    // Calculate maximum wave speeds across all processors
    const Set::Scalar *DX = geom[lev].CellSize();

    // Reduce maximum values across all processors
    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);

    // Calculate minimum stable timestep based on CFL condition
    Set::Scalar dt_min_ = cfl / ((c_max + vx_max) / (DX[0] + small) + (c_max + vy_max) / (DX[1] + small) + small);
    //Set::Scalar dt_min = cfl * std::sqrt(DX[0] * DX[0] + DX[1] * DX[1]) / (std::sqrt(vx_max * vx_max + vy_max * vy_max) + small);
    
    // Ensure dt_min is valid
    if (std::isnan(dt_min_) || std::isinf(dt_min_) || dt_min_ <= 0.0)
    {
        amrex::Print() << "WARNING: Invalid dt_min calculated: " << dt_min_
                       << " at time " << time << " on level " << lev << "\n";
        amrex::Print() << "  c_max = " << c_max << ", vx_max = " << vx_max
                       << ", vy_max = " << vy_max << "\n";
        dt_min_ = dt_min; // Use the minimum timestep from parameters
    }

    // DEBUGGING VERBOSE:
    //amrex::Print() << "  c_max = " << c_max << ", vx_max = " << vx_max << ", vy_max = " << vy_max << "\n";
    amrex::Print() << "  Recommended dt_min = " << dt_min_ << "\n";

    // Sync this minimum timestep with the dynamic timestep system
    DynamicTimestep_SyncTimeStep(lev, dt_min_);

    // Let the parent class handle the rest of the dynamic timestepping
    Integrator::DynamicTimestep_Update();
    */

    // NEW NEW METHOD
    // From Dr. Runnels
    if (dynamictimestep.on)
        Integrator::DynamicTimestep_Update();
    return;

    const Set::Scalar *DX = geom[lev].CellSize();

    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);

    Set::Scalar new_timestep = cfl / ((c_max + vx_max) / (DX[0]) + (c_max + vy_max) / (DX[1]) + small);
    new_timestep = new_timestep * 2.0; // Temp fix till root caust is found. new_timestep is double the lowest recommended time step

    // DEBUGGING VERBOSE
    // Ensure dt_min is valid
    if (std::isnan(new_timestep) || std::isinf(new_timestep) || new_timestep <= 0.0)
    {
        amrex::Print() << "WARNING: Invalid new_timestep calculated: " << new_timestep
                       << " at time " << time << " on level " << lev << "\n";
        amrex::Print() << "  c_max = " << c_max << ", vx_max = " << vx_max
                       << ", vy_max = " << vy_max << "\n";
        new_timestep = dt_min; // Use the minimum timestep from parameters
    }
    Util::Message(INFO, "  CFL Timestep = ", new_timestep);
    //amrex::Print() << "  CFL Timestep = " << new_timestep << "\n";

    Util::Assert(INFO, TEST(AMREX_SPACEDIM == 2));

    SetTimestep(new_timestep);
    
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// SETADAPTIVETIMESTEPPARAMS //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::SetAdaptiveTimestepParams(Set::Scalar cfl_val, Set::Scalar min_dt, Set::Scalar max_dt, Set::Scalar growth_factor_val, bool enable_adaptive)
{
    cfl = cfl_val;
    dt_min = min_dt;
    dt_max = max_dt;
    dt_growth = growth_factor_val;       // Changed from dt_growth_factor to dt_growth
    adaptive_timestep = enable_adaptive; // Changed from use_adaptive_timestep to adaptive_timestep

    amrex::Print() << "Adaptive timestepping parameters set: CFL = " << cfl
                   << ", dt_min = " << dt_min
                   << ", dt_max = " << dt_max
                   << ", growth_factor = " << dt_growth                 // Changed from dt_growth_factor to dt_growth
                   << ", enabled = " << adaptive_timestep << std::endl; // Changed from use_adaptive_timestep to adaptive_timestep
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// RHS /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RHS(int lev, Set::Scalar time, amrex::MultiFab &rho_rhs_mf, amrex::MultiFab &M_rhs_mf, amrex::MultiFab &E_rhs_mf, amrex::MultiFab &eta_rhs_mf, const amrex::MultiFab &rho_mf_in, const amrex::MultiFab &M_mf_in, const amrex::MultiFab &E_mf_in, const amrex::MultiFab &eta_mf_in)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // Eta
        Set::Patch<const Set::Scalar> eta = eta_mf_in.array(mfi);

        // Mixture
        Set::Patch<const Set::Scalar> rho = rho_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> E = E_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> M = M_mf_in.array(mfi);
        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // gamma_eff
            Set::Scalar gamma_eff = eta(i, j, k) * gamma0 + (1.0 - eta(i, j, k)) * gamma1;

            // Velocity = M ./ (DX*DY*rho)
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k) + small);
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k) + small);

            // Pressure
            press(i, j, k) = (E(i, j, k) - (0.5 * ((M(i, j, k, 0) * M(i, j, k, 0)) + (M(i, j, k, 1) * M(i, j, k, 1))) / (rho(i, j, k) + small))) * (gamma_eff - 1.0) - pref;

            // DEBUG Tool
            if (press(i, j, k) > 1E1000)
            {
                Util::ParallelMessage(INFO, "v=", v(i, j, k));
                Util::ParallelMessage(INFO, "press=", press(i, j, k));
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k));
                Util::ParallelMessage(INFO, "E=", E(i, j, k));
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                Util::Exception(INFO);
            }
        });
    }

    // Main calculation loop
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // MIXTURE
        Set::Patch<const Set::Scalar> eta = eta_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> rho = rho_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> E = E_mf_in.array(mfi);
        Set::Patch<const Set::Scalar> M = M_mf_in.array(mfi);
        Set::Patch<Set::Scalar> eta_rhs = eta_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_rhs = rho_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs = E_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs = M_rhs_mf.array(mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);


        //Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Source = Source_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> Fsv = Fsv_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fb = Fb_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Fw = Fw_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);

        // DEBUGGING
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_eta_ = hess_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);
            Set::Vector n_hat = grad_eta / (grad_eta_mag + small); // Normal Vector

            // DEBUGGING
            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

            // Debugging, would like to delete condition
            if (grad_eta_mag < small)
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

                if ((hess_eta(0, 0) != hess_eta(0, 0))
                    or (hess_eta(0, 1) != hess_eta(0, 1))
                    or (hess_eta(1, 0) != hess_eta(1, 0))
                    or (hess_eta(1, 1) != hess_eta(1, 1)))
                {
                    Util::ParallelMessage(INFO, "ERROR in Hydro2::RHS() : hess_eta");
                    Util::ParallelMessage(INFO, "i=", i, ", j=", j);
                    Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                    Util::ParallelMessage(INFO, "M=", M(i, j, k));
                    Util::ParallelMessage(INFO, "E=", E(i, j, k));
                    Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                    Util::ParallelMessage(INFO, "hess_eta(0,0)=", hess_eta(0, 0));
                    Util::ParallelMessage(INFO, "hess_eta(0,1)=", hess_eta(0, 1));
                    Util::ParallelMessage(INFO, "hess_eta(1,0)=", hess_eta(1, 0));
                    Util::ParallelMessage(INFO, "hess_eta(1,1)=", hess_eta(1, 1));
                    Util::Exception(INFO);
                }
                else
                {
                    hess_eta_(i, j, k, 0) = hess_eta(0, 0);
                    hess_eta_(i, j, k, 1) = hess_eta(0, 1);
                    hess_eta_(i, j, k, 2) = hess_eta(1, 0);
                    hess_eta_(i, j, k, 3) = hess_eta(1, 1);
                }
            }

            // Extract velocity from momentum and density
            //Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            Set::Vector u = Set::Vector(M(i, j, k, 0) / (rho(i, j, k) + small), M(i, j, k, 1) / (rho(i, j, k) + small));
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
                                          / rho(i, j, k);
                    }

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
                            if (p == r && q == s) Mpqrs += 0.5 * mu_eff;
                            if (p == s && q == r) Mpqrs += 0.5 * mu_eff;
                            // Bulk viscosity
                            if (p == q && r == s) Mpqrs += (1.0 / 3.0) * mu_b_eff;
                            Ldot0(p) += 0.5 * Mpqrs * (u(r) - u0(r)) * hess_eta(q, s);
                        }

            // NEW Formulation of div_tau
            // Calculate gradient of divergence
            Set::Vector grad_div_u = Set::Vector::Zero();
            grad_div_u[0] = hess_u(0, 0, 0) + hess_u(1, 0, 1);
            grad_div_u[1] = hess_u(0, 1, 0) + hess_u(1, 1, 1);
            // Calculate div_tau
            div_tau(0) = mu_eff * (hess_u(0, 0, 0) + hess_u(0, 1, 1) + (1.0 / 3.0) * grad_div_u[0]);
            div_tau(1) = mu_eff * (hess_u(1, 0, 0) + hess_u(1, 1, 1) + (1.0 / 3.0) * grad_div_u[1]);

            // Surface Tension:
            Fsv(i, j, k) = (0.0, 0.0);
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);
            if (apply_surface_tension)
            {
                if (grad_eta_mag <= 0.01)
                {
                    Fsv(i, j, k, 0) = 0.0;
                    Fsv(i, j, k, 1) = 0.0;
                }
                else
                {
                    Set::Scalar sigma_eff = sigma;
                    Set::Scalar kappa = 0.0;
                    if (kappa_method == 1)
                    {
                        Set::Vector grad_mag_grad_eta = Set::Vector(1 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                                                                    1 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));
                        kappa = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / ((grad_eta_mag + small) * (grad_eta_mag + small))));

                        // Density Scaling
                        /*
                        Set::Scalar a = 0.49;               // Cutoff
                        Set::Scalar x = eta(i, j, k) - 0.5; // Eta centered at 0
                        Set::Scalar D = 0.0;                // Dirac Delta Value
                        Set::Scalar pi = 3.14159;           // Add decimals as needed
                        if ((-a < x) and (x < a)) D = 0.5 * (1 + x / a + 1 / pi * std::sin(pi * x / a)) else D = 0.0;
                        kappa = D * kappa; // Smoothened Curvature
                        */

                        // To Track Surface Curvature
                        kappas(i, j, k, 0) = kappa; // Mean
                        kappas(i, j, k, 1) = 0.0;   // 1
                        kappas(i, j, k, 2) = 0.0;   // 2
                    }
                    else if (kappa_method == 2)
                    {
                        // Here are a few different ways of calculating principle curvatures. Approach 1 and 2 give much larger curvatures than the kappa_method==1 method described above from the divergence of the normal vector. This brought the need of approach 3.
                        // To run any approach, uncomment your desired approach and run.
                        //
                        // Approach 1 uses the simplified dot product of the hessian matrix in the normal and tangent direction to extract principle curvatures
                        // Approach 2 uses matrix projection of the hessian onto the normal and tangent plane and extracts principle curvatures
                        // Approach 3 uses calculates the eigen values of the hessian to find the principle curvatures

                        // /////////////////////////////////////////////////////////////////////////////////////
                        // APPROACH 1: Dot product of Hessian and normal vector
                        // /*
                        // Normal Vector
                        Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
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
                        Set::Scalar kappa1 = n_hat.dot(hess_eta * n_hat); // Normal Curvature
                        Set::Scalar kappa2 = t1.dot(hess_eta * t1);       // Tangential Curvature

                        /*
                        if ((std::abs(kappa2) > std::abs(kappa1))
                            and !( (eta(i,j,k) > 0.40) and (eta(i,j,k) < 0.60)))
                        {
                            std::swap(kappa1, kappa2);
                        }
                        */

                        kappa1 = -kappa1;
                        kappa2 = -kappa2 * 2.0 * epsilon;

                        // TODO: Add check for normal vector

                        // */

                        // /////////////////////////////////////////////////////////////////////////////////////
                        // APPROACH 2: Hessian Projection
                        /*
                        // Normal Vector
                        Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
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
                        // Transformation matrix Q: n_hat | t1
                        Set::Matrix Q(2, 2);
                        Q(0, 0) = n_hat(0); Q(0, 1) = t1(0);
                        Q(1, 0) = n_hat(1); Q(1, 1) = t1(1);

                        // Hessian Projection
                        Set::Matrix H_projected = Q.transpose() * hess_eta * Q;

                        // Principal Curvatures
                        Set::Scalar kappa1 = H_projected(0, 0); // Normal Curvature
                        Set::Scalar kappa2 = H_projected(1, 1); // Tangential Curvature
                        */

                        // /////////////////////////////////////////////////////////////////////////////////////
                        // APPROACH 3: Eigen Values of Hessian
                        // TODO: only works in 2D
                        /*
                        Set::Scalar trace = hess_eta(0, 0) + hess_eta(1, 1);
                        Set::Scalar det = hess_eta(0, 0) * hess_eta(1, 1) - hess_eta(0, 1) * hess_eta(1, 0);
                        Set::Scalar discriminant = std::sqrt(trace * trace - 4 * det);

                        // Principal curvatures are eigenvalues
                        Set::Scalar kappa1 = (trace + discriminant) / 2.0; // Larger eigenvalue
                        Set::Scalar kappa2 = (trace - discriminant) / 2.0; // Smaller eigenvalue

                        // // Ensure kappa1 is the larger
                        // if (std::abs(kappa2) > std::abs(kappa1))
                        // {
                        //     std::swap(kappa1, kappa2);
                        // }

                        // Scale kappa2 from boundry thickness
                        kappa2 = kappa2; //*2.0 * epsilon;
                        */

                        // /////////////////////////////////////////////////////////////////////////////////////
                        // Extracting method output to be used
                        // Regularization
                        Set::Scalar K23 = kappa2 * kappa2;     // K23 Regularization
                        Set::Scalar K_Gauss = kappa1 * kappa2; // Gauss Regularization
                        // Mean
                        Set::Scalar K_mean = (kappa1 + kappa2) / 2.0; // Mean Curvature
                        // Set::Scalar K_mean = std::sqrt(std::abs((K23 + K_Gauss) / 2.0)); // Mean Curvature
                        // Set::Scalar K_mean = std::sqrt(std::abs((K23 + K_Gauss) / 2.0)) * kappa1/std::abs(kappa1); // Mean Curvature

                        // Assign the curvature you want to use
                        kappa = kappa2; // Or use another curvature measure as needed

                        // Density Scaling
                        /*
                        Set::Scalar a = 0.49;               // Cutoff
                        Set::Scalar x = eta(i, j, k) - 0.5; // Eta centered at 0
                        Set::Scalar D = 0.0;                // Dirac Delta Value
                        Set::Scalar pi = 3.14159;           // Add decimals as needed
                        if ((-a < x) and (x < a)) D = 0.5 * (1 + x / a + 1 / pi * std::sin(pi * x / a)) else D = 0.0;
                        kappa = D * kappa; // Smoothened Curvature
                        */

                        // Store curvature values
                        kappas(i, j, k, 0) = kappa;  // Mean or selected curvature
                        kappas(i, j, k, 1) = kappa1; // First principal curvature
                        kappas(i, j, k, 2) = kappa2; // Second principal curvature
                    }

                    Fsv(i, j, k, 0) = sigma_eff * (kappa * grad_eta(0));
                    Fsv(i, j, k, 1) = sigma_eff * (kappa * grad_eta(1));
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
                Fw_vector = Set::Vector(Fw(i, j, k, 0), Fw(i, j, k, 1));
            }

            // Buoyancy:
            Fb(i, j, k) = (0.0, 0.0);
            Set::Vector Fb_vector = Set::Vector(0.0, 0.0);
            if (apply_buoyancy)
            {
                // WIP: set to use refrence density
                Fb_vector = Set::Vector(0.0, 0.0);
            }

            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv(i, j, k, 0) + Fb_vector(0) + Fw_vector(0),
                                                  Fsv(i, j, k, 1) + Fb_vector(1) + Fw_vector(1));

            Source(i, j, k, 0) = mdot0;
            Source(i, j, k, 1) = Pdot0(0) - Ldot0(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) - Ldot0(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(Total_Force);

            // Lagrange terms to enforce no-penetration
            Source(i, j, k, 1) -= lagrange * u.dot(grad_eta) * grad_eta(0);
            Source(i, j, k, 2) -= lagrange * u.dot(grad_eta) * grad_eta(1);

            // Riemann solver for mixed fluid
            const int X = 0, Y = 1;
            Solver::Local::Riemann::State state_xlo(rho, M, E, i - 1, j, k, X);
            Solver::Local::Riemann::State state_x(rho, M, E, i, j, k, X);
            Solver::Local::Riemann::State state_xhi(rho, M, E, i + 1, j, k, X);
            Solver::Local::Riemann::State state_ylo(rho, M, E, i, j - 1, k, Y);
            Solver::Local::Riemann::State state_y(rho, M, E, i, j, k, Y);
            Solver::Local::Riemann::State state_yhi(rho, M, E, i, j + 1, k, Y);

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::Riemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            Set::Scalar gamma_eff = eta(i, j, k) * gamma0 + (1.0 - eta(i, j, k)) * gamma1;
            Set::Scalar p0_eff = eta(i, j, k) * p0_0 + (1.0 - eta(i, j, k)) * p0_1;

            try
            {
                // Calculate fluxes based on the selected Riemann solver
                if (Riemann_Solver == 0)
                {
                    flux_xlo = roesolver->Solve(state_xlo, state_x, gamma_eff, pref, small, p0_eff);
                    flux_ylo = roesolver->Solve(state_ylo, state_y, gamma_eff, pref, small, p0_eff);
                    flux_xhi = roesolver->Solve(state_x, state_xhi, gamma_eff, pref, small, p0_eff);
                    flux_yhi = roesolver->Solve(state_y, state_yhi, gamma_eff, pref, small, p0_eff);
                }
                else if (Riemann_Solver == 1)
                {
                    flux_xlo = hllcsolver->Solve(state_xlo, state_x, gamma_eff, pref, small, p0_eff);
                    flux_ylo = hllcsolver->Solve(state_ylo, state_y, gamma_eff, pref, small, p0_eff);
                    flux_xhi = hllcsolver->Solve(state_x, state_xhi, gamma_eff, pref, small, p0_eff);
                    flux_yhi = hllcsolver->Solve(state_y, state_yhi, gamma_eff, pref, small, p0_eff);
                }
                else if (Riemann_Solver == 2)
                {
                    flux_xlo = hllesolver->Solve(state_xlo, state_x, gamma_eff, pref, small, p0_eff);
                    flux_ylo = hllesolver->Solve(state_ylo, state_y, gamma_eff, pref, small, p0_eff);
                    flux_xhi = hllesolver->Solve(state_x, state_xhi, gamma_eff, pref, small, p0_eff);
                    flux_yhi = hllesolver->Solve(state_y, state_yhi, gamma_eff, pref, small, p0_eff);
                }
                else if (Riemann_Solver == 3)
                {
                    flux_xlo = hllcesolver->Solve(state_xlo, state_x, gamma_eff, pref, small, p0_eff);
                    flux_ylo = hllcesolver->Solve(state_ylo, state_y, gamma_eff, pref, small, p0_eff);
                    flux_xhi = hllcesolver->Solve(state_x, state_xhi, gamma_eff, pref, small, p0_eff);
                    flux_yhi = hllcesolver->Solve(state_y, state_yhi, gamma_eff, pref, small, p0_eff);
                }
                else if (Riemann_Solver == 35)
                {
                    flux_xlo = hllc_weno5solver->Solve(state_xlo, state_x, gamma_eff, pref, small, p0_eff);
                    flux_ylo = hllc_weno5solver->Solve(state_ylo, state_y, gamma_eff, pref, small, p0_eff);
                    flux_xhi = hllc_weno5solver->Solve(state_x, state_xhi, gamma_eff, pref, small, p0_eff);
                    flux_yhi = hllc_weno5solver->Solve(state_y, state_yhi, gamma_eff, pref, small, p0_eff);
                }
            }
            catch (...)
            {
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "dx=", DX[0], "dy=", DX[1]);
                Util::Abort(INFO);
            }

            // Calculate RHS for eta
            //Set::Scalar deta_dt = -v(i, j, k, 0) * grad_eta(0) - v(i, j, k, 1) * grad_eta(1);
            Set::Scalar deta_dt = -u(0) * grad_eta(0) - u(1) * grad_eta(1);
            eta_rhs(i, j, k) = deta_dt;

            // Calculate RHS for density
            Set::Scalar drho_dt = 
                (flux_xlo.mass - flux_xhi.mass) / (DX[0] + small) 
                + (flux_ylo.mass - flux_yhi.mass) / (DX[1] + small) 
                + Source(i, j, k, 0);
            rho_rhs(i, j, k) = drho_dt;

            // Calculate RHS for momentum
            Set::Scalar dMx_dt = 
                (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0] + small) 
                + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1] + small) 
                + div_tau(0) 
                + Source(i, j, k, 1);
            M_rhs(i, j, k, 0) = dMx_dt;

            Set::Scalar dMy_dt = 
                (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0] + small) 
                + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1] + small) 
                + div_tau(1) 
                + Source(i, j, k, 2);
            M_rhs(i, j, k, 1) = dMy_dt;

            // Calculate RHS for energy
            Set::Scalar dE_dt = 
                (flux_xlo.energy - flux_xhi.energy) / (DX[0] + small) 
                + (flux_ylo.energy - flux_yhi.energy) / (DX[1] + small) 
                + Source(i, j, k, 3);
            E_rhs(i, j, k) = dE_dt;

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
    // Swap pointers for old and new values
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_old_mf[lev], energy_mf[lev]);
    std::swap(eta_old_mf, eta_mf);

    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();
    const Set::Scalar *DX = geom[lev].CellSize();

    // Create temporary storage for RK stages if needed
    const amrex::BoxArray &ba = density_mf[lev]->boxArray();
    const amrex::DistributionMapping &dm = density_mf[lev]->DistributionMap();
    const int ng = density_mf[lev]->nGrow();

    // Handles to old solution
    amrex::MultiFab &density_old = *density_old_mf[lev];
    amrex::MultiFab &momentum_old = *momentum_old_mf[lev];
    amrex::MultiFab &energy_old = *energy_old_mf[lev];
    amrex::MultiFab &eta_old = *eta_old_mf[lev];

    // Handles to new solution
    amrex::MultiFab &density_new = *density_mf[lev];
    amrex::MultiFab &momentum_new = *momentum_mf[lev];
    amrex::MultiFab &energy_new = *energy_mf[lev];
    amrex::MultiFab &eta_new = *eta_mf[lev];

    // Temporary storage for RHS
    amrex::MultiFab density_rhs(ba, dm, 1, 0);
    amrex::MultiFab momentum_rhs(ba, dm, 2, 0);
    amrex::MultiFab energy_rhs(ba, dm, 1, 0);
    amrex::MultiFab eta_rhs(ba, dm, 1, 0);

    // Choose integration scheme
    if (scheme == 0) // Forward Euler
    {
        // Calculate RHS
        RHS(lev, time, density_rhs, momentum_rhs, energy_rhs, eta_rhs, density_old, momentum_old, energy_old, eta_old);

        // Update solution using Forward Euler
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();

            Set::Patch<const Set::Scalar> rho_rhs = density_rhs.array(mfi);
            Set::Patch<const Set::Scalar> M_rhs = momentum_rhs.array(mfi);
            Set::Patch<const Set::Scalar> E_rhs = energy_rhs.array(mfi);
            Set::Patch<const Set::Scalar> eta_rhs_patch = eta_rhs.array(mfi);

            Set::Patch<const Set::Scalar> rho_old = density_old.array(mfi);
            Set::Patch<const Set::Scalar> M_old = momentum_old.array(mfi);
            Set::Patch<const Set::Scalar> E_old = energy_old.array(mfi);
            Set::Patch<const Set::Scalar> eta_old_patch = eta_old.array(mfi);

            Set::Patch<Set::Scalar> rho_new = density_new.array(mfi);
            Set::Patch<Set::Scalar> M_new = momentum_new.array(mfi);
            Set::Patch<Set::Scalar> E_new = energy_new.array(mfi);
            Set::Patch<Set::Scalar> eta_new_patch = eta_new.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // Forward Euler update
                rho_new(i, j, k) = rho_old(i, j, k) + dt * rho_rhs(i, j, k);
                M_new(i, j, k, 0) = M_old(i, j, k, 0) + dt * M_rhs(i, j, k, 0);
                M_new(i, j, k, 1) = M_old(i, j, k, 1) + dt * M_rhs(i, j, k, 1);
                E_new(i, j, k) = E_old(i, j, k) + dt * E_rhs(i, j, k);

                // Update eta with cutoff handling
                eta_new_patch(i, j, k) = eta_old_patch(i, j, k) + dt * eta_rhs_patch(i, j, k);

                if (eta_new_patch(i, j, k) <= cutoff)
                {
                    eta_new_patch(i, j, k) = 0.0;
                }
                else if (eta_new_patch(i, j, k) >= (1.0 - cutoff))
                {
                    eta_new_patch(i, j, k) = 1.0;
                }
            });
        }
    }
    else if (scheme == 1) // RK4
    {
        // Temporary storage for RK stages
        amrex::MultiFab density_k1(ba, dm, 1, 0), momentum_k1(ba, dm, 2, 0), energy_k1(ba, dm, 1, 0), eta_k1(ba, dm, 1, 0);
        amrex::MultiFab density_k2(ba, dm, 1, 0), momentum_k2(ba, dm, 2, 0), energy_k2(ba, dm, 1, 0), eta_k2(ba, dm, 1, 0);
        amrex::MultiFab density_k3(ba, dm, 1, 0), momentum_k3(ba, dm, 2, 0), energy_k3(ba, dm, 1, 0), eta_k3(ba, dm, 1, 0);
        amrex::MultiFab density_k4(ba, dm, 1, 0), momentum_k4(ba, dm, 2, 0), energy_k4(ba, dm, 1, 0), eta_k4(ba, dm, 1, 0);

        // Temporary storage for intermediate states
        amrex::MultiFab density_temp(ba, dm, 1, ng), momentum_temp(ba, dm, 2, ng), energy_temp(ba, dm, 1, ng), eta_temp(ba, dm, 1, ng);

        // Fill ghost cells
        density_temp.ParallelCopyToGhost(density_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));
        momentum_temp.ParallelCopyToGhost(momentum_old, 0, 0, 2, amrex::IntVect(1), amrex::IntVect(1));
        energy_temp.ParallelCopyToGhost(energy_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));
        eta_temp.ParallelCopyToGhost(eta_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));

        // K1 = RHS(t, y_old)
        RHS(lev, time, density_k1, momentum_k1, energy_k1, eta_k1, density_old, momentum_old, energy_old, eta_old);

        // K2 = RHS(t + dt/2, y_old + dt/2 * K1)
        // First compute y_temp = y_old + dt/2 * K1
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt / 2.0, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt / 2.0, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_temp, 1.0, energy_old, 0, dt / 2.0, energy_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt / 2.0, eta_k1, 0, 0, 1, 0);

        // Fill boundary
        density_bc->FillBoundary(density_temp, 0, 1, time, 0);
        density_temp.FillBoundary(true);
        momentum_bc->FillBoundary(momentum_temp, 0, 2, time, 0);
        momentum_temp.FillBoundary(true);
        energy_bc->FillBoundary(energy_temp, 0, 1, time, 0);
        energy_temp.FillBoundary(true);
        eta_bc->FillBoundary(eta_temp, 0, 1, time, 0);
        eta_temp.FillBoundary(true);

        // Compute K2
        RHS(lev, time + dt / 2.0, density_k2, momentum_k2, energy_k2, eta_k2, density_temp, momentum_temp, energy_temp, eta_temp);

        // K3 = RHS(t + dt/2, y_old + dt/2 * K2)
        // First compute y_temp = y_old + dt/2 * K2
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt / 2.0, density_k2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt / 2.0, momentum_k2, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_temp, 1.0, energy_old, 0, dt / 2.0, energy_k2, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt / 2.0, eta_k2, 0, 0, 1, 0);

        // Fill boundary
        density_bc->FillBoundary(density_temp, 0, 1, time, 0);
        density_temp.FillBoundary(true);
        momentum_bc->FillBoundary(momentum_temp, 0, 2, time, 0);
        momentum_temp.FillBoundary(true);
        energy_bc->FillBoundary(energy_temp, 0, 1, time, 0);
        energy_temp.FillBoundary(true);
        eta_bc->FillBoundary(eta_temp, 0, 1, time, 0);
        eta_temp.FillBoundary(true);

        // Compute K3
        RHS(lev, time + dt / 2.0, density_k3, momentum_k3, energy_k3, eta_k3, density_temp, momentum_temp, energy_temp, eta_temp);

        // K4 = RHS(t + dt, y_old + dt * K3)
        // First compute y_temp = y_old + dt * K3
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt, density_k3, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt, momentum_k3, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_temp, 1.0, energy_old, 0, dt, energy_k3, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt, eta_k3, 0, 0, 1, 0);

        // Fill boundary
        density_bc->FillBoundary(density_temp, 0, 1, time, 0);
        density_temp.FillBoundary(true);
        momentum_bc->FillBoundary(momentum_temp, 0, 2, time, 0);
        momentum_temp.FillBoundary(true);
        energy_bc->FillBoundary(energy_temp, 0, 1, time, 0);
        energy_temp.FillBoundary(true);
        eta_bc->FillBoundary(eta_temp, 0, 1, time, 0);
        eta_temp.FillBoundary(true);

        // Compute K4
        RHS(lev, time + dt, density_k4, momentum_k4, energy_k4, eta_k4, density_temp, momentum_temp, energy_temp, eta_temp);

        // Combine to get final solution: y_new = y_old + dt/6 * (K1 + 2*K2 + 2*K3 + K4)
        // First: y_new = y_old + dt/6 * K1
        amrex::MultiFab::LinComb(density_new, 1.0, density_old, 0, dt / 6.0, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_new, 1.0, momentum_old, 0, dt / 6.0, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_new, 1.0, energy_old, 0, dt / 6.0, energy_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_new, 1.0, eta_old, 0, dt / 6.0, eta_k1, 0, 0, 1, 0);

        // Add dt/3 * K2 (2*dt/6)
        amrex::MultiFab::Saxpy(density_new, dt / 3.0, density_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_new, dt / 3.0, momentum_k2, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_new, dt / 3.0, energy_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_new, dt / 3.0, eta_k2, 0, 0, 1, 0);

        // Add dt/3 * K3 (2*dt/6)
        amrex::MultiFab::Saxpy(density_new, dt / 3.0, density_k3, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_new, dt / 3.0, momentum_k3, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_new, dt / 3.0, energy_k3, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_new, dt / 3.0, eta_k3, 0, 0, 1, 0);

        // Add dt/6 * K4
        amrex::MultiFab::Saxpy(density_new, dt / 6.0, density_k4, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_new, dt / 6.0, momentum_k4, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_new, dt / 6.0, energy_k4, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_new, dt / 6.0, eta_k4, 0, 0, 1, 0);

        // Apply cutoff to eta
        for (amrex::MFIter mfi(eta_new, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            Set::Patch<Set::Scalar> eta_new_patch = eta_new.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (eta_new_patch(i, j, k) <= cutoff)
                {
                    eta_new_patch(i, j, k) = 0.0;
                }
                else if (eta_new_patch(i, j, k) >= (1.0 - cutoff))
                {
                    eta_new_patch(i, j, k) = 1.0;
                }
            });
        }
    }
    else if (scheme == 2) // SSPRK3
    {
        // Butcher Tableau for SSPRK3
        //     |
        //  1  |  1
        // 1/2 | 1/4  1/4
        // ---------------------
        //     | 1/6  1/6  2/3

        Set::Scalar c2 = 1.0, a21 = 1.0;
        Set::Scalar c3 = 0.5, a31 = 0.25, a32 = 0.25;
        Set::Scalar b1 = 1. / 6, b2 = 1. / 6, b3 = 2. / 3;

        // Temporary storage for RK stages
        amrex::MultiFab density_k1(ba, dm, 1, 0), momentum_k1(ba, dm, 2, 0), energy_k1(ba, dm, 1, 0), eta_k1(ba, dm, 1, 0);
        amrex::MultiFab density_k2(ba, dm, 1, 0), momentum_k2(ba, dm, 2, 0), energy_k2(ba, dm, 1, 0), eta_k2(ba, dm, 1, 0);
        amrex::MultiFab density_k3(ba, dm, 1, 0), momentum_k3(ba, dm, 2, 0), energy_k3(ba, dm, 1, 0), eta_k3(ba, dm, 1, 0);

        // Temporary storage for intermediate states
        amrex::MultiFab density_temp(ba, dm, 1, ng), momentum_temp(ba, dm, 2, ng), energy_temp(ba, dm, 1, ng), eta_temp(ba, dm, 1, ng);

        // Fill ghost cells
        density_temp.ParallelCopyToGhost(density_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));
        momentum_temp.ParallelCopyToGhost(momentum_old, 0, 0, 2, amrex::IntVect(1), amrex::IntVect(1));
        energy_temp.ParallelCopyToGhost(energy_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));
        eta_temp.ParallelCopyToGhost(eta_old, 0, 0, 1, amrex::IntVect(1), amrex::IntVect(1));

        // K1 = RHS(t, y_old)
        RHS(lev, time, density_k1, momentum_k1, energy_k1, eta_k1, density_old, momentum_old, energy_old, eta_old);

        // K2 = RHS(t + c2*dt, y_old + dt*a21*k1)
        // First compute y_temp = y_old + dt*a21*k1
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt * a21, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt * a21, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_temp, 1.0, energy_old, 0, dt * a21, energy_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt * a21, eta_k1, 0, 0, 1, 0);

        // Fill boundary
        density_bc->FillBoundary(density_temp, 0, 1, time, 0);
        density_temp.FillBoundary(true);
        momentum_bc->FillBoundary(momentum_temp, 0, 2, time, 0);
        momentum_temp.FillBoundary(true);
        energy_bc->FillBoundary(energy_temp, 0, 1, time, 0);
        energy_temp.FillBoundary(true);
        eta_bc->FillBoundary(eta_temp, 0, 1, time, 0);
        eta_temp.FillBoundary(true);

        // Compute K2
        RHS(lev, time + c2 * dt, density_k2, momentum_k2, energy_k2, eta_k2, density_temp, momentum_temp, energy_temp, eta_temp);

        // K3 = RHS(t + c3*dt, y_old + dt*(a31*k1 + a32*k2))
        // First compute y_temp = y_old + dt*a31*k1
        amrex::MultiFab::LinComb(density_temp, 1.0, density_old, 0, dt * a31, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_temp, 1.0, momentum_old, 0, dt * a31, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_temp, 1.0, energy_old, 0, dt * a31, energy_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_temp, 1.0, eta_old, 0, dt * a31, eta_k1, 0, 0, 1, 0);

        // Add dt*a32*k2
        amrex::MultiFab::Saxpy(density_temp, dt * a32, density_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_temp, dt * a32, momentum_k2, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_temp, dt * a32, energy_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_temp, dt * a32, eta_k2, 0, 0, 1, 0);

        // Fill boundary
        density_bc->FillBoundary(density_temp, 0, 1, time, 0);
        density_temp.FillBoundary(true);
        momentum_bc->FillBoundary(momentum_temp, 0, 2, time, 0);
        momentum_temp.FillBoundary(true);
        energy_bc->FillBoundary(energy_temp, 0, 1, time, 0);
        energy_temp.FillBoundary(true);
        eta_bc->FillBoundary(eta_temp, 0, 1, time, 0);
        eta_temp.FillBoundary(true);

        // Compute K3
        RHS(lev, time + c3 * dt, density_k3, momentum_k3, energy_k3, eta_k3, density_temp, momentum_temp, energy_temp, eta_temp);

        // Combine to get final solution: y_new = y_old + dt*(b1*k1 + b2*k2 + b3*k3)
        // First: y_new = y_old + dt*b1*k1
        amrex::MultiFab::LinComb(density_new, 1.0, density_old, 0, dt * b1, density_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(momentum_new, 1.0, momentum_old, 0, dt * b1, momentum_k1, 0, 0, 2, 0);
        amrex::MultiFab::LinComb(energy_new, 1.0, energy_old, 0, dt * b1, energy_k1, 0, 0, 1, 0);
        amrex::MultiFab::LinComb(eta_new, 1.0, eta_old, 0, dt * b1, eta_k1, 0, 0, 1, 0);

        // Add dt*b2*k2
        amrex::MultiFab::Saxpy(density_new, dt * b2, density_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_new, dt * b2, momentum_k2, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_new, dt * b2, energy_k2, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_new, dt * b2, eta_k2, 0, 0, 1, 0);

        // Add dt*b3*k3
        amrex::MultiFab::Saxpy(density_new, dt * b3, density_k3, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(momentum_new, dt * b3, momentum_k3, 0, 0, 2, 0);
        amrex::MultiFab::Saxpy(energy_new, dt * b3, energy_k3, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(eta_new, dt * b3, eta_k3, 0, 0, 1, 0);

        // Apply cutoff to eta
        for (amrex::MFIter mfi(eta_new, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            Set::Patch<Set::Scalar> eta_new_patch = eta_new.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                if (eta_new_patch(i, j, k) <= cutoff)
                {
                    eta_new_patch(i, j, k) = 0.0;
                }
                else if (eta_new_patch(i, j, k) >= (1.0 - cutoff))
                {
                    eta_new_patch(i, j, k) = 1.0;
                }
            });
        }
    }
    else
    {
        Util::ParallelMessage(INFO, "ERROR in Hydro2::Advance() : Integrator Methods");
        Util::ParallelMessage(INFO, "Method ", scheme, " is unknown.");
        Util::Exception(INFO);
    }

    // Update velocity, pressure, and calculate maximum wave speeds for adaptive timestepping
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E = energy_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Source = Source_mf.Patch(lev, mfi);

        Set::Scalar *dt_max_handle = &dt_max;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Update velocity
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k) + small);
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k) + small);

            // Update pressure
            Set::Scalar gamma_eff = eta(i, j, k) * gamma0 + (1.0 - eta(i, j, k)) * gamma1;
            press(i, j, k) = (E(i, j, k) - (0.5 * ((M(i, j, k, 0) * M(i, j, k, 0)) + (M(i, j, k, 1) * M(i, j, k, 1))) / (rho(i, j, k) + small))) * (gamma_eff - 1.0) - pref;

            // Calculate vorticity
            Set::Matrix gradu = Numeric::Gradient(v, i, j, k, DX);
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));

            // Calculate maximum wave speeds for adaptive timestepping
            Set::Scalar sound_speed = std::sqrt(gamma_eff * press(i, j, k) / (rho(i, j, k) + small));
            c_max = std::max(c_max, sound_speed);
            vx_max = std::max(vx_max, std::abs(v(i, j, k, 0)));
            vy_max = std::max(vy_max, std::abs(v(i, j, k, 1)));

            // Calculate dt_max for adaptive timestepping
            *dt_max_handle =                          std::fabs(cfl * DX[0] / (v(i, j, k, 0) + small));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl * DX[1] / (v(i, j, k, 1) + small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[0] * DX[0] / (Source(i, j, k, 1) + small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[1] * DX[1] / (Source(i, j, k, 2) + small)));
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
