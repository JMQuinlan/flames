#ifndef INTEGRATOR_Hydro2_H
#define INTEGRATOR_Hydro2_H

#include "Integrator/Integrator.H"
#include "IO/ParmParse.H"

#include "BC/BC.H"
#include "BC/Constant.H"
#include "BC/Nothing.H"
#include "IC/IC.H"

#include "Set/Base.H"
#include "Solver/Local/Riemann/Roe.H"
#include "Solver/Local/Riemann/HLLC.H"
#include "Solver/Local/Riemann/HLLC_WENO5.H"
#include "Solver/Local/Riemann/HLLE.H"
//#include "Solver/Local/Riemann/HLLE_WENO5.H"
#include "Solver/Local/Riemann/HLLCE.H"
//#include "Solver/Local/Riemann/HLLCE_WENO5.H"
#include "Solver/Local/Riemann/PartiallyParabolic.H"


#if AMREX_SPACEDIM==2

namespace Integrator
{
class Hydro2 : virtual public Integrator
{
public:

    Hydro2() {};
    Hydro2(IO::ParmParse& pp);
    static void Parse(Hydro2& value, IO::ParmParse& pp);
    virtual ~Hydro2()
    {
        /// Boundry Conditions
        delete density_bc;
        delete momentum_bc;
        delete energy_bc;
        delete eta_bc;

        /// Inital Conditions
        // ETA
        delete eta_ic;
        delete etadot_ic;
        // FLUID 0
        delete density0_ic;
        delete velocity0_ic;
        delete pressure0_ic;
        delete temperature0_ic; // Temperature
        // FLUID 1
        delete density1_ic;
        delete velocity1_ic;
        delete pressure1_ic;
        delete temperature1_ic; // Temperature
        // SOURCES
        delete ic_m0;
        delete ic_u0;
        delete ic_q;
        /// SOLVERS
        delete roesolver;
        delete hllcsolver;
        delete hllc_weno5solver;
        delete hllesolver;
        //delete hlle_weno5solver;
        delete hllcesolver;
        //delete hllce_weno5solver;
        delete partiallyparabolicsolver;

    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    void RHS(int lev, 
        Set::Scalar time, 
        amrex::MultiFab &rho_rhs_mf, 
        amrex::MultiFab &M_rhs_mf, 
        amrex::MultiFab &E_rhs_mf, 
        amrex::MultiFab &eta_rhs_mf, 
        const amrex::MultiFab &rho_mf, 
        const amrex::MultiFab &M_mf, 
        const amrex::MultiFab &E_mf, 
        const amrex::MultiFab &eta_mf_in );


    void SetAdaptiveTimestepParams(Set::Scalar cfl_val, Set::Scalar min_dt, Set::Scalar max_dt, Set::Scalar growth_factor_val, bool enable_adaptive);

    // virtual void UpdateEta(int lev, Set::Scalar time);

private:
    // FLUID 0
    Set::Field<Set::Scalar> density0_mf;
    Set::Field<Set::Scalar> density0_old_mf;

    Set::Field<Set::Scalar> energy0_mf;
    Set::Field<Set::Scalar> energy0_old_mf;

    Set::Field<Set::Scalar> momentum0_mf;
    Set::Field<Set::Scalar> momentum0_old_mf;

    Set::Field<Set::Scalar> T0_mf;

    Set::Field<Set::Scalar> velocity0_mf;
    Set::Field<Set::Scalar> pressure0_mf;
    Set::Field<Set::Scalar> vorticity0_mf;

    // FLUID 1
    Set::Field<Set::Scalar> density1_mf;
    Set::Field<Set::Scalar> density1_old_mf;

    Set::Field<Set::Scalar> energy1_mf;
    Set::Field<Set::Scalar> energy1_old_mf;

    Set::Field<Set::Scalar> momentum1_mf;
    Set::Field<Set::Scalar> momentum1_old_mf;

    Set::Field<Set::Scalar> T1_mf;

    Set::Field<Set::Scalar> velocity1_mf;
    Set::Field<Set::Scalar> pressure1_mf;
    Set::Field<Set::Scalar> vorticity1_mf;

    // Mixture
    Set::Field<Set::Scalar> density_mf;
    Set::Field<Set::Scalar> density_old_mf;
    Set::Field<Set::Scalar> vorticity_mf;
    Set::Field<Set::Scalar> momentum_mf;
    Set::Field<Set::Scalar> momentum_old_mf;
    Set::Field<Set::Scalar> energy_mf;
    Set::Field<Set::Scalar> energy_old_mf;
    Set::Field<Set::Scalar> velocity_mf;
    Set::Field<Set::Scalar> pressure_mf;

    Set::Field<Set::Scalar> T_mf;   // Temperature
    Set::Field<Set::Scalar> a_mf;   // Speed of sound
    Set::Field<Set::Scalar> Ma_mf;  // Mach
    Set::Field<Set::Scalar> UE_mf;  // Internal Energy
    Set::Field<Set::Scalar> KE_mf;  // Kinetic Energy

protected:
    Set::Field<Set::Scalar> eta_mf;
    Set::Field<Set::Scalar> eta_mf_in;
    Set::Field<Set::Scalar> etadot_mf;
    Set::Field<Set::Scalar> eta_old_mf;
    Set::Field<Set::Scalar> hess_eta_mf;
    Set::Field<Set::Scalar> n_hat_mf;

private:
   
    Set::Field<Set::Scalar> m0_mf;
    Set::Field<Set::Scalar> u0_mf;
    Set::Field<Set::Scalar> q_mf;

    //Set::Field<Set::Scalar> m1_mf;
    //Set::Field<Set::Scalar> u1_mf;
    //Set::Field<Set::Scalar> q1_mf;
    

    Set::Field<Set::Scalar> Source_mf;
    Set::Field<Set::Scalar> flux_mf;
    Set::Field<Set::Scalar> Fsv_mf;
    Set::Field<Set::Scalar> Fb_mf;
    Set::Field<Set::Scalar> Fw_mf;
    Set::Field<Set::Scalar> grad_eta_mf;
    Set::Field<Set::Scalar> kappas_mf;

    BC::BC<Set::Scalar>* density_bc = nullptr;
    BC::BC<Set::Scalar>* momentum_bc = nullptr;
    BC::BC<Set::Scalar>* energy_bc = nullptr;
    BC::BC<Set::Scalar>* eta_bc = nullptr;
    BC::BC<Set::Scalar>* temperature_bc = nullptr;

    IC::IC<Set::Scalar>* velocity0_ic = nullptr;
    IC::IC<Set::Scalar>* pressure0_ic = nullptr;
    IC::IC<Set::Scalar>* density0_ic  = nullptr;
    IC::IC<Set::Scalar>* temperature0_ic  = nullptr;

    IC::IC<Set::Scalar>* velocity1_ic = nullptr;
    IC::IC<Set::Scalar>* pressure1_ic = nullptr;
    IC::IC<Set::Scalar>* density1_ic = nullptr;
    IC::IC<Set::Scalar>* temperature1_ic = nullptr;

    IC::IC<Set::Scalar>* ic_m0 = nullptr;
    IC::IC<Set::Scalar>* ic_u0 = nullptr;
    IC::IC<Set::Scalar>* ic_q = nullptr;

    IC::IC<Set::Scalar>* eta_ic = nullptr;
    IC::IC<Set::Scalar>* etadot_ic = nullptr;

    BC::Nothing bc_nothing;

    BC::Constant neumann_bc_1 = BC::Constant::ZeroNeumann(1);
    BC::Constant neumann_bc_D = BC::Constant::ZeroNeumann(AMREX_SPACEDIM);

    Set::Scalar c_max = 0.0;
    Set::Scalar vx_max = 0.0;
    Set::Scalar vy_max = 0.0;

    Set::Scalar eta_refinement_criterion=NAN, omega_refinement_criterion=NAN, gradu_refinement_criterion=NAN, p_refinement_criterion=NAN, rho_refinement_criterion=NAN;
    // FLUID 0
    Set::Scalar gamma0 = NAN, p0_0 = NAN, mu0 = NAN, mu0_b = NAN, MW0 = NAN;

    // FLUID 1
    Set::Scalar gamma1 = NAN, p0_1 = NAN, mu1 = NAN, mu1_b = NAN, MW1 = NAN;

    // Interaction Parameters
    Set::Scalar sigma = NAN;
    Set::Scalar epsilon = NAN;
    Set::Scalar g = 9.81; // m/s^2
    // Set::Scalar R = 8.314; // [J]/[mol]-[K]
    Set::Scalar R = 8314.; // [J]/[kmol]-[K] - Probably more common to use

    // CFD Properties
    Set::Scalar cfl = NAN, cfl_v = NAN, pref = NAN;
    Set::Scalar small=NAN;
    Set::Scalar cutoff=NAN;
    Set::Scalar lagrange=NAN;

    // Booleans
    int apply_surface_tension = 1;
    int apply_buoyancy = 0;
    int apply_weight = 0;
    
    int Riemann_Solver = 0;
    int kappa_method = 2;

    // Adaptive Timestep
    bool adaptive_timestep = false;
    Set::Scalar dt_min = NAN, dt_max = NAN, dt_growth = NAN;

    // Solvers
    int scheme = 0;
    Solver::Local::Riemann::Roe *roesolver = nullptr;
    Solver::Local::Riemann::HLLC *hllcsolver = nullptr;
    Solver::Local::Riemann::HLLC_WENO5 *hllc_weno5solver = nullptr;
    Solver::Local::Riemann::HLLE *hllesolver = nullptr;
    //Solver::Local::Riemann::HLLE_WENO5 *hlle_weno5solver = nullptr;
    Solver::Local::Riemann::HLLCE *hllcesolver = nullptr;
    //Solver::Local::Riemann::HLLCE_WENO5 *hllce_weno5solver = nullptr;
    Solver::Local::Riemann::PartiallyParabolic *partiallyparabolicsolver = nullptr;

};
}

#endif
#endif
