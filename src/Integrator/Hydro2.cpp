
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
#include "Solver/Local/Riemann/HLLE.H"

#if AMREX_SPACEDIM == 2

namespace Integrator
{

Hydro2::Hydro2(IO::ParmParse& pp) : Hydro2()
{
    pp.queryclass(*this);
}

void
Hydro2::Parse(Hydro2& value, IO::ParmParse& pp)
{
    BL_PROFILE("Integrator::Hydro2::Hydro2()");
    {
        // REFINEMENT CRITERION
        pp.query_default("eta_refinement_criterion",   value.eta_refinement_criterion  , 0.01); // eta-based refinement
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01); // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01); // velocity gradient-based refinement
        pp.query_default("p_refinement_criterion", value.p_refinement_criterion, 1e100);        // pressure-based refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e100);    // density-based refinement

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);           // cfl condition
        pp_query_default("cfl_v", value.cfl_v, 1E-100); // cfl condition
        pp_query_default("pref", value.pref, 1.0);     // reference pressure for Roe solver
        pp_query_default("small", value.small, 1E-8);  // small regularization value
        pp_query_default("cutoff", value.cutoff, 1E-100);  // eta cutoff value
        pp_query_default("lagrange", value.lagrange, 0.0); // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81); // Gravitational Acceletation
        pp_forbid("roefix", "--> solver.roe.entropy_fix"); // Roe solver entropy fix

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, true); // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        pp_query_default("apply_buoyancy", value.apply_buoyancy, false);              // Apply buoyancy when solving, default: false --> "No Buoyancy"
        pp_query_default("apply_weight", value.apply_weight, false);                  // Apply weight when solving, default: false --> "No Weight"

        // NEW SOLVER FORBIDS
        pp_forbid("gamma", "--> gamma0 and gamma1");
        pp_forbid("mu", "--> mu0 and mu1");

        // FLUID 0
        pp_query_required("gamma0", value.gamma0);  // gamma for gamma law
        pp_query_required("mu0", value.mu0);        // linear viscosity 

        // FLUID 1
        pp_query_required("gamma1", value.gamma1); // gamma for gamma law
        pp_query_required("mu1", value.mu1);       // linear viscosity coefficient

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 70.0); // surface tension condition
        
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
        value.RegisterNewFab(value.Fsv_mf, &value.bc_nothing, 2, nghost, "Fsv", true, {"x","y"}); // To Track Surface Tension
        value.RegisterNewFab(value.Fb_mf, &value.bc_nothing, 2, nghost, "Fb", true, {"x","y"}); // To Track Bouyancy
        value.RegisterNewFab(value.Fw_mf, &value.bc_nothing, 2, nghost, "Fw", true, { "x", "y" }); // To Track Weight

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
    
    // SOLVER
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
    
    
}


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

    // Calculate mixed variables based on individual fluid variables
    Mix(lev);
}

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

void Hydro2::UpdateEta(int lev, Set::Scalar time)
{
    eta_ic->Initialize(lev, eta_mf, time);
}

void Hydro2::TimeStepBegin(Set::Scalar, int /*iter*/)
{

}

void Hydro2::TimeStepComplete(Set::Scalar, int lev)
{
    Integrator::DynamicTimestep_Update();

    return;

    const Set::Scalar* DX = geom[lev].CellSize();

    amrex::ParallelDescriptor::ReduceRealMax(c_max);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max);

    Set::Scalar new_timestep = cfl / ((c_max + vx_max) / DX[0] + (c_max + vy_max) / DX[1]);

    Util::Assert(INFO, TEST(AMREX_SPACEDIM == 2));

    SetTimestep(new_timestep);
}

void Hydro2::Advance(int lev, Set::Scalar time, Set::Scalar dt)
{
    // Swaping pointers
    /*
    // FLUID 0
    std::swap(density0_old_mf[lev], density0_mf[lev]);
    std::swap(momentum0_old_mf[lev], momentum0_mf[lev]);
    std::swap(energy0_old_mf[lev], energy0_mf[lev]);
    
    // FLUID 1
    std::swap(density1_old_mf[lev], density1_mf[lev]);
    std::swap(momentum1_old_mf[lev], momentum1_mf[lev]);
    std::swap(energy1_old_mf[lev], energy1_mf[lev]);
    */

    // MIX
    std::swap(density_old_mf[lev], density_mf[lev]);
    std::swap(momentum_old_mf[lev], momentum_mf[lev]);
    std::swap(energy_old_mf[lev], energy_mf[lev]);
    std::swap(eta_old_mf, eta_mf);

    Set::Scalar dt_max = std::numeric_limits<Set::Scalar>::max();

    //UpdateEta(lev, time);
    const Set::Scalar *DX = geom[lev].CellSize();

    // Update etadot
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // Eta:
        Set::Patch<Set::Scalar> &eta_new = (*eta_mf[lev]).array(mfi);
        Set::Patch<const Set::Scalar> const &eta = (*eta_old_mf[lev]).array(mfi);
        Set::Patch<Set::Scalar> &etadot = (*etadot_mf[lev]).array(mfi);

        // Mixture
        Set::Patch<const Set::Scalar> rho = density_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E = energy_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> M = momentum_old_mf.Patch(lev, mfi);

        Set::Patch<Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        
        // Capture only the arrays, not the MFIter
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // gamma_eff
            Set::Scalar gamma_eff = eta(i, j, k) * gamma0 + (1.0 - eta(i, j, k)) * gamma1;

            // etadot
            etadot(i, j, k) = (eta_new(i, j, k) - eta(i, j, k)) / dt;

            // Velocity = M ./ (DX*DY*rho)
            v(i, j, k, 0) = M(i, j, k, 0) / (rho(i, j, k));
            v(i, j, k, 1) = M(i, j, k, 1) / (rho(i, j, k));

            // Pressure
            press(i, j, k) = (E(i, j, k) - (0.5 * ((M(i, j, k, 0) * M(i, j, k, 0)) + (M(i, j, k, 1) * M(i, j, k, 1))) / (rho(i, j, k)+small))) * (gamma_eff - 1.0) - pref; // NEEDS Verification


            // DEBUG Tool
            if (press(i, j, k) > 1E1000)
            {
                Util::ParallelMessage(INFO, "v=", v(i,j,k));
                Util::ParallelMessage(INFO, "press=", press(i,j,k));
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k));
                Util::ParallelMessage(INFO, "E=", E(i, j, k));
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                //Util::ParallelMessage(INFO, "etadot=", etadot(i, j, k));
                Util::Exception(INFO);
            }
        });
    }
   
    amrex::Box domain = geom[lev].Domain();

    // Main time integration loop
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        // MIXTURE
        Set::Patch<const Set::Scalar>   rho     = density_old_mf.Patch(lev,mfi);
        Set::Patch<const Set::Scalar>   E       = energy_old_mf.Patch(lev,mfi);
        Set::Patch<const Set::Scalar>   M       = momentum_old_mf.Patch(lev,mfi);
        Set::Patch<Set::Scalar>         rho_new = density_mf.Patch(lev,mfi);
        Set::Patch<Set::Scalar>         E_new   = energy_mf.Patch(lev,mfi);
        Set::Patch<Set::Scalar>         M_new   = momentum_mf.Patch(lev,mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);

        Set::Patch<const Set::Scalar>   eta     = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   v       = velocity_mf.Patch(lev, mfi); 

        Set::Patch<const Set::Scalar>   m0      = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   q       = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   _u0     = u0_mf.Patch(lev, mfi);

        amrex::Array4<Set::Scalar> const &Source = (*Source_mf[lev]).array(mfi);
        Set::Patch <Set::Scalar>        Fsv = Fsv_mf.Patch(lev, mfi);
        Set::Patch <Set::Scalar>        Fb = Fb_mf.Patch(lev, mfi);
        Set::Patch <Set::Scalar>        Fw = Fw_mf.Patch(lev, mfi);

        // DEBUGGING
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);

        Set::Scalar *dt_max_handle = &dt_max;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta, i, j, k, 0, DX);
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // DEBUGGING
            grad_eta_(i, j, k, 0) = grad_eta(0);
            grad_eta_(i, j, k, 1) = grad_eta(1);

            // Extract velocity from momentum and density
            //Set::Vector u  = Set::Vector(M(i, j, k, 0) / rho(i, j, k), M(i, j, k, 1) / rho(i, j, k));
            Set::Vector u = Set::Vector(v(i, j, k, 0), v(i, j, k, 1));
            //Set::Vector u_mag = u.lpNorm<2>();
            Set::Vector u0 = Set::Vector(_u0(i, j, k, 0), _u0(i, j, k, 1));

            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / rho(i, j, k);

            Set::Vector q0 = Set::Vector(q(i, j, k, 0), q(i, j, k, 1));

            // Calculate source terms
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

            
            Set::Vector Ldot0 = Set::Vector::Zero();
            Set::Vector div_tau = Set::Vector::Zero();
            for (int p = 0; p < 2; p++) // Dimension Component
                for (int q = 0; q < 2; q++) // X
                    for (int r = 0; r < 2; r++) // Y
                        for (int s = 0; s < 2; s++) // Z
                        {
                            Set::Scalar Mpqrs = 0.0;
                            /*
                            if (p == r && q == s) Mpqrs += 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic
                            if (p == s && q == r) Mpqrs += 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic
                            if (p == q && r == s) Mpqrs += 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic
                            */
                            if (p == r && q == s) Mpqrs = 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic
                            if (p == s && q == r) Mpqrs = 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic
                            if (p == q && r == s) Mpqrs = 0.5 * (eta(i, j, k) * mu0 + ((1.0 - eta(i, j, k)) * mu1)); // TODO: Assumes Isotropic

                            Ldot0(p) += 0.5 * Mpqrs * (u(r) - u0(r)) * hess_eta(q, s);
                            div_tau(p) += 2.0 * Mpqrs * hess_u(r, s, q);
                        }

            // Surface Tension:
            // Fsv =  simga * kappa * n_hat
            Fsv(i, j, k) = (0.0, 0.0);
            Set::Vector Fsv_vector = Set::Vector(0.0, 0.0);
            if (apply_surface_tension)
            {
                // Optimization, only calc surface tension if on interface
                //if ((eta(i, j, k) <= cutoff / 10.0) or (eta(i, j, k) >= 1.0 - cutoff / 10.0))
                if (((grad_eta(0) <= cutoff / 10.0) and (grad_eta(0) >= -cutoff / 10.0))
                    and ((grad_eta(1) <= cutoff / 10.0) and (grad_eta(1) >= -cutoff / 10.0)))
                {
                    Fsv(i, j, k, 0) = 0.0;
                    Fsv(i, j, k, 1) = 0.0;
                }
                else
                {
                    Set::Scalar sigma_eff = sigma;
                    Set::Vector grad_mag_grad_eta = Set::Vector(1 / (grad_eta_mag + small) * (grad_eta(0) * hess_eta(0, 0) + grad_eta(1) * hess_eta(0, 1)),
                                                                1 / (grad_eta_mag + small) * (grad_eta(1) * hess_eta(1, 1) + grad_eta(0) * hess_eta(1, 0)));
                    Set::Scalar kappa = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / ((grad_eta_mag + small) * (grad_eta_mag + small))));

                    Fsv(i, j, k, 0) = sigma_eff * (kappa * grad_eta(0) / (grad_eta_mag + small)); // / (DX[0] + small);
                    Fsv(i, j, k, 1) = sigma_eff * (kappa * grad_eta(1) / (grad_eta_mag + small)); // / (DX[1] + small);
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
                // Example: apply a downward weight force; modify as appropriate
                Fb_vector = Set::Vector(0.0, 0.0); // replace mass with actual value if available
            }

            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv(i, j, k, 0) + Fb_vector(0) + Fw_vector(0),
                                                  Fsv(i, j, k, 1) + Fb_vector(1) + Fw_vector(1));
            

            Source(i, j, k, 0) = mdot0;
            Source(i, j, k, 1) = Pdot0(0) - Ldot0(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) - Ldot0(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(Total_Force); //(Fsv(i, j, k, 0) * u(0) + Fsv(i, j, k, 1) * u(1)); //+ u.dot(Ldot0)

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

            // TODO: Enforce individual reimann states

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::Riemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            Set::Scalar gamma_eff = eta(i, j, k) * gamma0 + (1.0 - eta(i, j, k)) * gamma1;

            try
            {
                if (Riemann_Solver == 0)
                {
                    // Calculate fluxes for the mixed fluid
                    flux_xlo = roesolver->Solve(state_xlo, state_x, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_ylo = roesolver->Solve(state_ylo, state_y, gamma_eff, pref, small);     //  * eta(i,j,k)
                    flux_xhi = roesolver->Solve(state_x, state_xhi, gamma_eff, pref, small);     //  * eta(i,j,k)
                    flux_yhi = roesolver->Solve(state_y, state_yhi, gamma_eff, pref, small);     //  * eta(i,j,k)
                }
                else if (Riemann_Solver == 1)
                {
                    // Calculate fluxes for the mixed fluid
                    flux_xlo = hllcsolver->Solve(state_xlo, state_x, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_ylo = hllcsolver->Solve(state_ylo, state_y, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_xhi = hllcsolver->Solve(state_x, state_xhi, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_yhi = hllcsolver->Solve(state_y, state_yhi, gamma_eff, pref, small); //  * eta(i,j,k)
                }
                else if (Riemann_Solver == 2)
                {
                    // Calculate fluxes for the mixed fluid
                    flux_xlo = hllesolver->Solve(state_xlo, state_x, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_ylo = hllesolver->Solve(state_ylo, state_y, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_xhi = hllesolver->Solve(state_x, state_xhi, gamma_eff, pref, small); //  * eta(i,j,k)
                    flux_yhi = hllesolver->Solve(state_y, state_yhi, gamma_eff, pref, small); //  * eta(i,j,k)
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
            /// Eta:
            // Material Derivative
            Set::Scalar deta_dt = -u.dot(grad_eta);
            // Cahn-Hillard
            // Set::Scalar deta_dt = -u.dot(grad_eta) + M.dot(grad_eta)
            // Set::Matrix tmp =
            // Set::Scalar deta_dt = -1.0 / (rho(i, j, k) * (u_mag**2))
            // Set::Matrix gradu = (gradM - u * gradrho.transpose()) / rho(i, j, k);

            // Set::Scalar deta_dt = -; //https://www.sciencedirect.com/science/article/pii/S002199912100005X
            eta_new(i, j, k) = eta(i, j, k) + deta_dt * dt;
            if (eta_new(i, j, k) <= cutoff)
            {
                eta_new(i, j, k) = 0.0;
            }
            else if (eta_new(i, j, k) >= (1.0 - cutoff))
            {
                eta_new(i, j, k) = 1.0;
            }

            // Update Source Terms to account for moving boundry
            // Delete me if does not worky :(
            Source(i, j, k, 0) = Source(i, j, k, 0) - rho(i, j, k) * deta_dt;
            Source(i, j, k, 1) = Source(i, j, k, 1) - M(i, j, k, 0) * deta_dt;
            Source(i, j, k, 2) = Source(i, j, k, 2) - M(i, j, k, 1) * deta_dt;
            Source(i, j, k, 3) = Source(i, j, k, 3) - E(i, j, k) * deta_dt;


            // Density
            Set::Scalar drho_dt = 
                (flux_xlo.mass - flux_xhi.mass) / (DX[0]+small) + 
                (flux_ylo.mass - flux_yhi.mass) / (DX[1]+small) + 
                Source(i, j, k, 0);

            rho_new(i, j, k) = rho(i, j, k) + (drho_dt) * dt;

            if (rho_new(i, j, k) != rho_new(i, j, k))
            {
                Util::ParallelMessage(INFO, "lev=", lev);
                Util::ParallelMessage(INFO, "i=", i, "j=", j);
                Util::ParallelMessage(INFO, "drho_dt=", drho_dt); // dies
                Util::ParallelMessage(INFO, "flux_xlo.mass=", flux_xlo.mass);
                Util::ParallelMessage(INFO, "flux_xhi.mass=", flux_xhi.mass);
                Util::ParallelMessage(INFO, "flux_ylo.mass=", flux_ylo.mass);
                Util::ParallelMessage(INFO, "flux_yhi.mass=", flux_yhi.mass);
                Util::ParallelMessage(INFO, "eta=", eta(i, j, k));
                Util::ParallelMessage(INFO, "Source=", Source(i, j, k, 0));
                Util::ParallelMessage(INFO, "state_x ", state_x); // <<<<
                Util::ParallelMessage(INFO, "state_y ", state_y);
                Util::ParallelMessage(INFO, "state_xhi ", state_xhi); // <<<<
                Util::ParallelMessage(INFO, "state_yhi ", state_yhi);
                Util::ParallelMessage(INFO, "state_xlo ", state_xlo);
                Util::ParallelMessage(INFO, "state_ylo ", state_ylo);
                Util::ParallelMessage(INFO, "gamma_eff=", gamma_eff);
                Util::ParallelMessage(INFO, "rho=", rho(i, j, k));
                Util::ParallelMessage(INFO, "M=", M(i, j, k));
                Util::ParallelMessage(INFO, "E=", E(i, j, k));
                Util::Exception(INFO);
            }

            // Momentum
            Set::Scalar dMx_dt = 
                (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]+small) + 
                (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]+small) + 
                div_tau(0) +
                //(mu * (lap_ux * eta(i, j, k))) +
                Source(i, j, k, 1);

            M_new(i, j, k, 0) = M(i, j, k, 0) + dMx_dt * dt;

            Set::Scalar dMy_dt = 
                (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]+small) + 
                (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]+small) + 
                div_tau(1) +
                //(mu * (lap_uy * eta(i, j, k))) +
                Source(i, j, k, 2);

            M_new(i, j, k, 1) = M(i, j, k, 1) + dMy_dt * dt;

            // Energy
            Set::Scalar dE_dt = 
                (flux_xlo.energy - flux_xhi.energy) / (DX[0] + small) + 
                (flux_ylo.energy - flux_yhi.energy) / (DX[1] + small) + 
                Source(i, j, k, 3);

            E_new(i, j, k) = E(i, j, k) + dE_dt * dt;

            




            // Set::Vector grad_ux = Numeric::Gradient(v, i, j, k, 0, DX);
            // Set::Vector grad_uy = Numeric::Gradient(v, i, j, k, 1, DX);

            *dt_max_handle = std::fabs(cfl * DX[0] / (u(0)+small));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl * DX[1] / (u(1)+small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[0] * DX[0] / (Source(i, j, k, 1) + small)));
            *dt_max_handle = std::min(*dt_max_handle, std::fabs(cfl_v * DX[1] * DX[1] / (Source(i, j, k, 2) + small)));

            // Calculate vorticity for visualization
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));

        });
    }

    this->DynamicTimestep_SyncTimeStep(lev, dt_max);
} // end Advance

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
