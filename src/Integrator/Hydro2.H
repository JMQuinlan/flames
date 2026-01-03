#ifndef INTEGRATOR_Hydro2_H
#define INTEGRATOR_Hydro2_H

#include "Integrator/Integrator.H"
#include "IO/ParmParse.H"

#include "BC/BC.H"
#include "BC/Constant.H"
#include "BC/Nothing.H"
#include "IC/IC.H"

#include "Set/Base.H"

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
// #include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
// Limiters
// #include "Solver/Local/Limiter/Minmod.H"
// #include "Solver/Local/Limiter/VanLeer.H"


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
        //delete eta_bc;
        delete temperature_bc;

        /// Inital Conditions
        // ETA
        delete eta_ic;
        delete etadot_ic;
        // FLUID 0
        delete density0_ic;
        delete velocity0_ic;
        delete pressure0_ic;
        delete temperature0_ic; // Temperature
        delete cp0_ic;          // Constant Pressure Specific heat
        delete cv0_ic;          // Constant Volume Specific Heat
        delete k0_thermal_ic;   // Thermal Conductivity
        delete h0_thermal_ic;   // Thermal Convectivity
        //delete gamma0_ic;       // Specific Heat Ratio
        // FLUID 1
        delete density1_ic;
        delete velocity1_ic;
        delete pressure1_ic;
        delete temperature1_ic; // Temperature
        delete cp1_ic;          // Constant Pressure Specific heat
        delete cv1_ic;          // Constant Volume Specific Heat
        delete k1_thermal_ic;   // Thermal Conductivity
        delete h1_thermal_ic;   // Thermal Convectivity
        // delete gamma1_ic;       // Specific Heat Ratio
        // SOURCES
        delete ic_m0;
        delete ic_u0;
        delete ic_q;
        /// SOLVERS
        // Riemann Solvers
        delete roesolver;
        delete hllcsolver;
        delete hllc_All_Machsolver;
        delete hllc_All_Mach_Furfarosolver;
        //delete hllc_weno5solver;
        delete hllesolver;
        //delete hlle_weno5solver; 
        // delete hllcesolver;
        //delete hllce_weno5solver;
        // delete partiallyparabolicsolver;
        // Flux Limiters
        // delete limiter_minmod;
        // delete limiter_vanleer;

        
    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void PrimitiveFieldCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage);
    void NaturalCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage);
    void ForcedCalc(int lev, Set::Scalar dt, const Set::Scalar *DX, bool use_stage);
    void RiemannFlux(int lev, Set::Scalar time, Set::Scalar dt, const Set::Scalar *DX, amrex::MultiFab *drho_dt_mf, amrex::MultiFab *dM_dt_mf, amrex::MultiFab *dE_dt_mf, amrex::MultiFab *deta_dt_mf, Set::Scalar &dt_max, bool use_stage);

    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    void RHS(int lev,
             Set::Scalar time,
             Set::Scalar dt,
             const Set::Scalar *DX,
             amrex::MultiFab &drho_dt_mf,
             amrex::MultiFab &dM_dt_mf,
             amrex::MultiFab &dE_dt_mf,
             amrex::MultiFab &deta_dt_mf,
             Set::Scalar &dt_max,
             bool use_stage);



    void SetAdaptiveTimestepParams(Set::Scalar cfl_val, Set::Scalar min_dt, Set::Scalar max_dt, Set::Scalar growth_factor_val, bool enable_adaptive);

private:
    // FLUID 0
    Set::Field<Set::Scalar> density0_mf;
    Set::Field<Set::Scalar> density0_old_mf;

    Set::Field<Set::Scalar> energy0_mf;
    Set::Field<Set::Scalar> energy0_old_mf;

    Set::Field<Set::Scalar> momentum0_mf;
    Set::Field<Set::Scalar> momentum0_old_mf;

    Set::Field<Set::Scalar> T0_mf;
    Set::Field<Set::Scalar> cp0_mf;
    Set::Field<Set::Scalar> cv0_mf;
    Set::Field<Set::Scalar> k0_thermal_mf;
    Set::Field<Set::Scalar> h0_thermal_mf;
    //Set::Field<Set::Scalar> gamma0_mf;

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
    Set::Field<Set::Scalar> cp1_mf;
    Set::Field<Set::Scalar> cv1_mf;
    Set::Field<Set::Scalar> k1_thermal_mf;
    Set::Field<Set::Scalar> h1_thermal_mf;
    //Set::Field<Set::Scalar> gamma1_mf;
    Set::Field<Set::Scalar> velocity1_mf;
    Set::Field<Set::Scalar> pressure1_mf;
    Set::Field<Set::Scalar> vorticity1_mf;

    // Mixture
    Set::Field<Set::Scalar> density_mf;
    Set::Field<Set::Scalar> density_old_mf;
    Set::Field<Set::Scalar> vorticity_mf;
    Set::Field<Set::Scalar> momentum_mf;
    Set::Field<Set::Scalar> momentum_old_mf;
    Set::Field<Set::Scalar> energy_per_vol_mf;
    Set::Field<Set::Scalar> energy_per_mas_mf;
    Set::Field<Set::Scalar> energy_per_vol_old_mf;
    Set::Field<Set::Scalar> energy_per_mas_old_mf;
    Set::Field<Set::Scalar> velocity_mf;
    Set::Field<Set::Scalar> pressure_mf;

    Set::Field<Set::Scalar> T_mf;           // Temperature
    Set::Field<Set::Scalar> cp_mf;          // Constant Pressure Specific Heat
    Set::Field<Set::Scalar> cv_mf;          // Constant Volume Specific Heat
    Set::Field<Set::Scalar> k_thermal_mf;   // Thermal Conductivity
    Set::Field<Set::Scalar> h_thermal_mf;   // Thermal Convectivity
    Set::Field<Set::Scalar> gamma_mf;       // Specific Heat Ratio
    Set::Field<Set::Scalar> p0_mf;          // Tamman Pressure
    Set::Field<Set::Scalar> mu_chem_mf;     // Allen-Cahn Chemical Potential
    Set::Field<Set::Scalar> a_mf;           // Speed of sound
    Set::Field<Set::Scalar> Ma_mf;          // Mach
    Set::Field<Set::Scalar> UE_per_vol_mf;  // Internal Energy (per unit volume)
    Set::Field<Set::Scalar> UE_per_mas_mf;  // Internal Energy (per unit mass)
    Set::Field<Set::Scalar> KE_per_vol_mf;  // Kinetic Energy (per unit volume)
    Set::Field<Set::Scalar> KE_per_mas_mf;  // Kinetic Energy (per unit mass)




    // Stage storage for RK schemes
    Set::Field<Set::Scalar> density_stage_mf;
    Set::Field<Set::Scalar> momentum_stage_mf;
    Set::Field<Set::Scalar> energy_per_vol_stage_mf;
    Set::Field<Set::Scalar> energy_per_mas_stage_mf;
    Set::Field<Set::Scalar> eta_stage_mf;

    // Stage-specific primitive/derived fields
    Set::Field<Set::Scalar> velocity_stage_mf;
    Set::Field<Set::Scalar> pressure_stage_mf;
    Set::Field<Set::Scalar> gamma_stage_mf;
    Set::Field<Set::Scalar> p0_stage_mf;
    Set::Field<Set::Scalar> a_stage_mf;
    Set::Field<Set::Scalar> Ma_stage_mf;
    Set::Field<Set::Scalar> mu_chem_stage_mf;
    Set::Field<Set::Scalar> Bm_stage_mf;
    Set::Field<Set::Scalar> T_stage_mf;
    Set::Field<Set::Scalar> cp_stage_mf;
    Set::Field<Set::Scalar> cv_stage_mf;
    Set::Field<Set::Scalar> k_thermal_stage_mf;
    Set::Field<Set::Scalar> h_thermal_stage_mf;
    Set::Field<Set::Scalar> tau_xx_stage_mf;
    Set::Field<Set::Scalar> tau_xy_stage_mf;
    Set::Field<Set::Scalar> tau_yy_stage_mf;
    Set::Field<Set::Scalar> Ldot_stage_mf;
    Set::Field<Set::Scalar> Source_stage_mf;
    Set::Field<Set::Scalar> Fsv_stage_mf;
    Set::Field<Set::Scalar> Fb_stage_mf;
    Set::Field<Set::Scalar> Fw_stage_mf;
    Set::Field<Set::Scalar> div_tau_stage_mf;
    Set::Field<Set::Scalar> hess_u_stage_mf;
    Set::Field<Set::Scalar> grad_eta_stage_mf;
    Set::Field<Set::Scalar> hess_eta_stage_mf;
    Set::Field<Set::Scalar> n_hat_stage_mf;
    Set::Field<Set::Scalar> kappas_stage_mf;
    Set::Field<Set::Scalar> grad_mag_grad_eta_stage_mf;
    Set::Field<Set::Scalar> etadot_stage_mf;
    Set::Field<Set::Scalar> KE_per_vol_stage_mf;
    Set::Field<Set::Scalar> KE_per_mas_stage_mf;
    Set::Field<Set::Scalar> UE_per_vol_stage_mf;
    Set::Field<Set::Scalar> UE_per_mas_stage_mf;

    // RK stage derivatives (k1, k2, k3, k4)
    Set::Field<Set::Scalar> k1_rho_mf;
    Set::Field<Set::Scalar> k1_M_mf;
    Set::Field<Set::Scalar> k1_E_mf;
    Set::Field<Set::Scalar> k1_eta_mf;

    Set::Field<Set::Scalar> k2_rho_mf;
    Set::Field<Set::Scalar> k2_M_mf;
    Set::Field<Set::Scalar> k2_E_mf;
    Set::Field<Set::Scalar> k2_eta_mf;

    Set::Field<Set::Scalar> k3_rho_mf;
    Set::Field<Set::Scalar> k3_M_mf;
    Set::Field<Set::Scalar> k3_E_mf;
    Set::Field<Set::Scalar> k3_eta_mf;

    Set::Field<Set::Scalar> k4_rho_mf;
    Set::Field<Set::Scalar> k4_M_mf;
    Set::Field<Set::Scalar> k4_E_mf;
    Set::Field<Set::Scalar> k4_eta_mf;


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
    

    Set::Field<Set::Scalar> Source_mf;      // Sumation of Source Terms
    Set::Field<Set::Scalar> flux_mf;        // Riemann Solver Fluxes
    Set::Field<Set::Scalar> Fsv_mf;         // Surface Tension
    Set::Field<Set::Scalar> Fb_mf;          // Boyancy Force (INOP)
    Set::Field<Set::Scalar> Fw_mf;          // Weight Force
    Set::Field<Set::Scalar> Bm_mf;          // Spalding Number
    Set::Field<Set::Scalar> tau_xx_mf;         // Stress Tensor (xx)
    Set::Field<Set::Scalar> tau_xy_mf;         // Stress Tensor (xy)
    Set::Field<Set::Scalar> tau_yy_mf;         // Stress Tensor (yy)
    Set::Field<Set::Scalar> Ldot_mf;        // Ldot
    Set::Field<Set::Scalar> grad_eta_mf;
    Set::Field<Set::Scalar> kappas_mf;
    Set::Field<Set::Scalar> grad_mag_grad_eta_mf;

    Set::Field<Set::Scalar> rho_flux_mf;
    Set::Field<Set::Scalar> M_flux_mf;
    Set::Field<Set::Scalar> E_flux_mf;

    // Debugging
    Set::Field<Set::Scalar> div_tau_mf;
    Set::Field<Set::Scalar> hess_u_mf;


    BC::BC<Set::Scalar>* density_bc = nullptr;
    BC::BC<Set::Scalar>* momentum_bc = nullptr;
    BC::BC<Set::Scalar>* energy_bc = nullptr;
    //BC::BC<Set::Scalar>* eta_bc = nullptr;
    BC::BC<Set::Scalar>* temperature_bc = nullptr;

    IC::IC<Set::Scalar>* velocity0_ic = nullptr;
    IC::IC<Set::Scalar>* pressure0_ic = nullptr;
    IC::IC<Set::Scalar>* density0_ic  = nullptr;
    IC::IC<Set::Scalar>* temperature0_ic  = nullptr;
    IC::IC<Set::Scalar>* cp0_ic  = nullptr;
    IC::IC<Set::Scalar>* cv0_ic  = nullptr;
    IC::IC<Set::Scalar>* k0_thermal_ic  = nullptr;
    IC::IC<Set::Scalar>* h0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma0_ic  = nullptr;

    IC::IC<Set::Scalar>* velocity1_ic = nullptr;
    IC::IC<Set::Scalar>* pressure1_ic = nullptr;
    IC::IC<Set::Scalar>* density1_ic = nullptr;
    IC::IC<Set::Scalar>* temperature1_ic = nullptr;
    IC::IC<Set::Scalar>* cp1_ic  = nullptr;
    IC::IC<Set::Scalar>* cv1_ic  = nullptr;
    IC::IC<Set::Scalar>* k1_thermal_ic  = nullptr;
    IC::IC<Set::Scalar>* h1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma1_ic  = nullptr;

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
    Set::Scalar F_max = 0.0;
    Set::Scalar rho_min = 1e10;

    Set::Scalar eta_refinement_criterion=NAN, omega_refinement_criterion=NAN, gradu_refinement_criterion=NAN, p_refinement_criterion=NAN, rho_refinement_criterion=NAN;
    // FLUID 0
    Set::Scalar gamma0 = NAN, p0_0 = NAN, mu0 = NAN, mu0_b = NAN, MW0 = NAN;
    Set::Scalar b0 = NAN, q0EOS = NAN;

    // FLUID 1
    Set::Scalar gamma1 = NAN, p0_1 = NAN, mu1 = NAN, mu1_b = NAN, MW1 = NAN;
    Set::Scalar b1 = NAN, q1EOS = NAN;

    // Interaction Parameters
    Set::Scalar sigma = NAN;
    Set::Scalar epsilon = NAN;
    Set::Scalar g = 9.81; // m/s^2
    // Set::Scalar R = 8.314; // [J]/[mol]-[K]
    Set::Scalar R = 8314.; // [J]/[kmol]-[K] - Probably more common to use
    Set::Scalar Dv = NAN; // Vapor Diffusivity

    // CFD Properties
    Set::Scalar cfl = NAN, cfl_v = NAN, pref = NAN;
    Set::Scalar small=NAN;
    Set::Scalar cutoff=NAN;
    Set::Scalar lagrange=NAN;

    // Booleans
    int apply_surface_tension = 1;
    int apply_buoyancy = 0;
    int apply_weight = 0;
    int static_eta = 0;
    int apply_vaporization = 0;
    int Spec_Vol = 1;
    
    int Riemann_Solver = 0;
    int kappa_method = 2;

    // Adaptive Timestep
    bool adaptive_timestep = false;
    Set::Scalar dt_min = NAN, dt_max = NAN, dt_growth = NAN;

    // Solvers
    int scheme = 0;
    Solver::Local::FluidRiemann::Roe *roesolver = nullptr;
    Solver::Local::FluidRiemann::HLLC *hllcsolver = nullptr;
    Solver::Local::FluidRiemann::HLLC_All_Mach *hllc_All_Machsolver = nullptr;
    Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro *hllc_All_Mach_Furfarosolver = nullptr;
    //Solver::Local::FluidRiemann::HLLC_WENO5 *hllc_weno5solver = nullptr; 
    Solver::Local::FluidRiemann::HLLE *hllesolver = nullptr;
    //Solver::Local::FluidRiemann::HLLE_WENO5 *hlle_weno5solver = nullptr;
    //Solver::Local::FluidRiemann::HLLCE *hllcesolver = nullptr;
    //Solver::Local::FluidRiemann::HLLCE_WENO5 *hllce_weno5solver = nullptr;
    //Solver::Local::FluidRiemann::PartiallyParabolic *partiallyparabolicsolver = nullptr;


    // Limiters
    int Limiter = 0;
    //Solver::Local::Limiter::Minmod *limiter_minmod = nullptr;
    //Solver::Local::Limiter::VanLeer *limiter_vanleer = nullptr;

};
}

#endif
#endif
