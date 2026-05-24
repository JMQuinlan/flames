#ifndef INTEGRATOR_HYDRO2_H
#define INTEGRATOR_HYDRO2_H

#include "Integrator/Integrator.H"
#include "IO/ParmParse.H"

// Boundry Conditions
#include "BC/BC.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "BC/Nothing.H"
#include "BC/NSCBC.H"
#include "BC/NSCBC4.H"
#include "BC/NSCBC7.H"
#include "IC/IC.H"

#include "Set/Base.H"

// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC7.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
// #include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
// #include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
// #include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"

//EOS
#include "Solver/EOS/EOS.H"
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"

// Limiters
// #include "Solver/Local/Limiter/Minmod.H"
// #include "Solver/Local/Limiter/VanLeer.H"



namespace Integrator
{
class Hydro2 : virtual public Integrator
{
public:
    static constexpr const char *name = "hydro2";

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
        delete temperature_bc;

        /// Inital Conditions
        // ETA
        delete eta_ic;
        delete etadot_ic;
        // FLUID 0
        delete density0_ic;
        delete velocity0_ic;
        delete pressure0_ic;
        //delete temperature0_ic; // Temperature
        //delete k0_thermal_ic;   // Thermal Conductivity
        //delete h0_thermal_ic;   // Thermal Convectivity
        // FLUID 1
        delete density1_ic;
        delete velocity1_ic;
        delete pressure1_ic;
        //delete temperature1_ic; // Temperature
       // delete k1_thermal_ic;   // Thermal Conductivity
        //delete h1_thermal_ic;   // Thermal Convectivity
        // SOURCES
        delete ic_m0;
        delete ic_u0;
        delete ic_q;
        // Flux Limiters
        // delete limiter_minmod;
        // delete limiter_vanleer;

        delete riemannsolver;
        delete riemannsolver7;
        delete nscbc7_bc;
    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    void PostSubcycleReflux(int lev, Set::Scalar time, Set::Scalar dt_coarse) override;
    void RHS(int lev,
        Set::Scalar time,
        amrex::MultiFab &rho_eta0_rhs_mf,
        amrex::MultiFab &rho_eta1_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        amrex::MultiFab &eta_rhs_mf,
        const amrex::MultiFab &rho_eta0_mf_in,
        const amrex::MultiFab &rho_eta1_mf_in,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf,
        const amrex::MultiFab &eta_mf_in);

    // ========================================================================
    // 7-EQUATION (Baer-Nunziato) entry points.
    // Wired via dispatch in Initialize() / Advance() based on equation_count.
    // Phase-B implementation: phase-by-phase HLLC, NO relaxation, NO ST,
    // NO vaporization. Single-phase regions (alpha = 0 or 1) must reduce to
    // the 5-eqn Allaire result.
    // ========================================================================
    void Initialize7eqn(int lev);
    void PrimitivesFromConservatives7eqn(int lev);
    void Advance7eqn(int lev, Set::Scalar time, Set::Scalar dt);
    void RHS7eqn(int lev,
        Set::Scalar time,
        amrex::MultiFab &alpha_rho_0_rhs_mf,
        amrex::MultiFab &alpha_rho_M0_rhs_mf,
        amrex::MultiFab &alpha_rho_E0_rhs_mf,
        amrex::MultiFab &alpha_rho_1_rhs_mf,
        amrex::MultiFab &alpha_rho_M1_rhs_mf,
        amrex::MultiFab &alpha_rho_E1_rhs_mf,
        amrex::MultiFab &alpha0_rhs_mf,
        const amrex::MultiFab &alpha_rho_0_in,
        const amrex::MultiFab &alpha_rho_M0_in,
        const amrex::MultiFab &alpha_rho_E0_in,
        const amrex::MultiFab &alpha_rho_1_in,
        const amrex::MultiFab &alpha_rho_M1_in,
        const amrex::MultiFab &alpha_rho_E1_in,
        const amrex::MultiFab &alpha0_in);


    void SetAdaptiveTimestepParams(Set::Scalar cfl_val, Set::Scalar min_dt, Set::Scalar max_dt, Set::Scalar growth_factor_val, bool enable_adaptive);
    void check4nans(int time, int lev, int i, int j, int k, const std::string &message, std::initializer_list<std::pair<std::string, double> > vars);
    //void InterfaceCompression(int lev, Set::Scalar dt_physical);
    void InterfaceSharpening(int lev, Set::Scalar dt_physical);
private:
    void ReinitializeSignedDistance(int lev,
                                    amrex::MultiFab &psi_mf,
                                    const amrex::MultiFab &psi_init_mf,
                                    int max_iter = 50);
    // BOUNDRY CONDITION FUNCTIONS
    // For MultiFabs that only need periodic/processor boundaries
    void FillBoundaries(int lev, std::initializer_list<amrex::MultiFab *> mfs);
    // For MultiFabs that need physical BCs + periodic/processor boundaries
    void FillBoundariesWithBC(int lev, Set::Scalar time, BC::BC<Set::Scalar> *bc, std::initializer_list<amrex::MultiFab *> mfs);
    // For MultiFabs that only need zero fill boundaries
    void FillBoundariesWithZero(int lev, std::initializer_list<amrex::MultiFab *> mfs);
    // Wrapper Function for BCs and Primative Fields
    void FillGhost4BC(int lev, Set::Scalar time);
    // Zero every derived/scratch MultiFabs
    void ZeroDerivedScratchFields(int lev);

    // Curvature pipeline
    void ComputeGradEta(int lev);
    void ComputeSmoothNormals(int lev);
    void ComputeKappas(int lev);

protected:
    // FLUID 0
    Set::Field<Set::Scalar> density0_mf;
    Set::Field<Set::Scalar> density0_old_mf;

    Set::Field<Set::Scalar> energy0_mf;
    Set::Field<Set::Scalar> energy0_old_mf;

    Set::Field<Set::Scalar> momentum0_mf;
    Set::Field<Set::Scalar> momentum0_old_mf;

    //Set::Field<Set::Scalar> T0_mf;
    //Set::Field<Set::Scalar> k0_thermal_mf;
    //Set::Field<Set::Scalar> h0_thermal_mf;
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

    //Set::Field<Set::Scalar> T1_mf;
    //Set::Field<Set::Scalar> k1_thermal_mf;
    //Set::Field<Set::Scalar> h1_thermal_mf;
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

    Set::Field<Set::Scalar> Source_mf;      // Sumation of Source Terms
    Set::Field<Set::Scalar> flux_mf;        // Riemann Solver Fluxes
    Set::Field<Set::Scalar> Fsv_mf;         // Surface Tension
    Set::Field<Set::Scalar> Fw_mf;          // Weight Force
    Set::Field<Set::Scalar> Bm_mf;          // Spalding Number
    Set::Field<Set::Scalar> Y_mf;           // Mass Fraction
    Set::Field<Set::Scalar> Ldot_mf;        // Ldot
    Set::Field<Set::Scalar> grad_eta_mf;
    Set::Field<Set::Scalar> kappas_mf;
    Set::Field<Set::Scalar> grad_mag_grad_eta_mf;
    Set::Field<Set::Scalar> eta_x_mf;
    Set::Field<Set::Scalar> eta_y_mf;
    Set::Field<Set::Scalar> gradmag_mf;
    Set::Field<Set::Scalar> nx_smoothed_mf;
    Set::Field<Set::Scalar> ny_smoothed_mf;
    Set::Field<Set::Scalar> kappa_SF_mf;

    Set::Field<Set::Scalar> rho_flux_mf;
    Set::Field<Set::Scalar> M_flux_mf;
    Set::Field<Set::Scalar> E_flux_mf;

    // Debugging
    Set::Field<Set::Scalar> div_tau_mf;
    Set::Field<Set::Scalar> hess_u_mf;
    Set::Field<Set::Scalar> Vap_dot_mf;
    

    Set::Field<Set::Scalar> m0_mf;
    Set::Field<Set::Scalar> u0_mf;
    Set::Field<Set::Scalar> q_mf;

protected:
    Set::Field<Set::Scalar> eta_mf;

    Set::Field<Set::Scalar> rho_eta0_mf;
    Set::Field<Set::Scalar> rho_eta1_mf;
    Set::Field<Set::Scalar> rho_eta0_old_mf;
    Set::Field<Set::Scalar> rho_eta1_old_mf;

    // ========================================================================
    // 7-EQUATION (Baer-Nunziato) CONSERVATIVE STATE
    //
    // Allocated only when equation_count == 7. The 5-eqn fields above remain
    // populated as diagnostics so plot files and post-processing scripts
    // continue to work, but they are no longer the primary state.
    //
    // Layout (per-phase, 9 components in 2D, 11 in 3D):
    //     alpha_rho_k_mf      :  alpha_k * rho_k             [1 component]
    //     alpha_rho_Mk_mf     :  alpha_k * rho_k * u_k       [SPACEDIM]
    //     alpha_rho_Ek_mf     :  alpha_k * rho_k * E_k       [1 component]
    //     alpha0_mf           :  volume fraction of phase 0  [1 component]
    //
    // Phase 0 is the liquid (eos0, Tammann). Phase 1 is the gas (eos1).
    // alpha_1 = 1 - alpha_0 is implicit.
    // ========================================================================
    Set::Field<Set::Scalar> alpha_rho_0_mf,    alpha_rho_0_old_mf;
    Set::Field<Set::Scalar> alpha_rho_M0_mf,   alpha_rho_M0_old_mf;
    Set::Field<Set::Scalar> alpha_rho_E0_mf,   alpha_rho_E0_old_mf;
    Set::Field<Set::Scalar> alpha_rho_1_mf,    alpha_rho_1_old_mf;
    Set::Field<Set::Scalar> alpha_rho_M1_mf,   alpha_rho_M1_old_mf;
    Set::Field<Set::Scalar> alpha_rho_E1_mf,   alpha_rho_E1_old_mf;
    Set::Field<Set::Scalar> alpha0_mf,         alpha0_old_mf;

    //Set::Field<Set::Scalar> eta_mf_in;
    Set::Field<Set::Scalar> etadot_mf;
    Set::Field<Set::Scalar> eta_old_mf;
    Set::Field<Set::Scalar> hess_eta_mf;
    Set::Field<Set::Scalar> n_hat_mf;
protected:

    int n_step = 0;

    BC::BC<Set::Scalar>* density_bc = nullptr;
    BC::BC<Set::Scalar>* momentum_bc = nullptr;
    BC::BC<Set::Scalar>* energy_bc = nullptr;
    BC::BC<Set::Scalar>* eta_bc = nullptr;
    BC::BC<Set::Scalar>* temperature_bc = nullptr;

    IC::IC<Set::Scalar>* velocity0_ic = nullptr;
    IC::IC<Set::Scalar>* pressure0_ic = nullptr;
    IC::IC<Set::Scalar>* density0_ic  = nullptr;
    //IC::IC<Set::Scalar>* temperature0_ic  = nullptr;
    //IC::IC<Set::Scalar>* k0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma0_ic  = nullptr;

    IC::IC<Set::Scalar>* velocity1_ic = nullptr;
    IC::IC<Set::Scalar>* pressure1_ic = nullptr;
    IC::IC<Set::Scalar>* density1_ic = nullptr;
    //IC::IC<Set::Scalar>* temperature1_ic = nullptr;
    //IC::IC<Set::Scalar>* k1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma1_ic  = nullptr;

    IC::IC<Set::Scalar>* ic_m0 = nullptr;
    IC::IC<Set::Scalar>* ic_u0 = nullptr;
    IC::IC<Set::Scalar>* ic_q = nullptr;

    IC::IC<Set::Scalar>* eta_ic = nullptr;
    IC::IC<Set::Scalar>* etadot_ic = nullptr;

    BC::Nothing bc_nothing;

    BC::Constant neumann_bc_1 = BC::Constant::ZeroNeumann(1);
    BC::Constant neumann_bc_D = BC::Constant::ZeroNeumann(AMREX_SPACEDIM);

    BC::NSCBC *nscbc_bc = nullptr;
    BC::NSCBC4 *nscbc4_bc = nullptr;
    BC::NSCBC7 *nscbc7_bc = nullptr;   // 7-eqn per-phase NSCBC (Phase E)


    Set::Scalar c_max = 0.0;
    Set::Scalar vx_max = 0.0;
    Set::Scalar vy_max = 0.0;
    Set::Scalar F_max = 0.0;
    Set::Scalar rho_min = 1e10;

    Set::Scalar eta_refinement_criterion=NAN, omega_refinement_criterion=NAN, gradu_refinement_criterion=NAN, p_refinement_criterion=NAN, rho_refinement_criterion=NAN;
    // FLUID 0
    Set::Scalar gamma0 = NAN, p0_0 = NAN, mu0 = NAN, mu0_b = NAN, MW0 = NAN;
    Set::Scalar b0 = NAN, q0EOS = NAN;
    Set::Scalar cp0 = NAN, cv0 = NAN;

    // FLUID 1
    Set::Scalar gamma1 = NAN, p0_1 = NAN, mu1 = NAN, mu1_b = NAN, MW1 = NAN;
    Set::Scalar b1 = NAN, q1EOS = NAN;
    Set::Scalar cp1 = NAN, cv1 = NAN;

    // Interaction Parameters
    Set::Scalar sigma = NAN;
    Set::Scalar epsilon = NAN;
    Set::Scalar g = 9.81; // m/s^2
    // Set::Scalar R = 8.314; // [J]/[mol]-[K]
    Set::Scalar R = 8314.; // [J]/[kmol]-[K] - Probably more common to use
    Set::Scalar Dv = 0.0; // Vapor Diffusivity
    Set::Scalar Y_infinity = 0.0; // Far Feild Vapor Mass Fraction


    // CFD Properties
    Set::Scalar cfl = NAN, cfl_v = NAN, pref = NAN;
    // Cahn-Hilliard mobility scale M0 in M = M0 * epsilon^2.
    Set::Scalar Mob_user = NAN;
    Set::Scalar small=NAN;
    Set::Scalar cutoff=NAN;
    Set::Scalar lagrange=NAN;

    // Booleans
    int apply_surface_tension = 1;
    int apply_weight = 0;
    int static_eta = 0;
    int apply_vaporization = 0;
    int Spec_Vol = 1;

    int kappa_method = 1;
    int smooth_kernel_size = 3;

    // Adaptive Timestep
    bool adaptive_timestep = false;
    Set::Scalar dt_min = NAN, dt_max = NAN, dt_growth = NAN;

    // Limiters
    int Limiter = 0;
    //Solver::Local::Limiter::Minmod *limiter_minmod = nullptr;
    //Solver::Local::Limiter::VanLeer *limiter_vanleer = nullptr;

    // Solvers
    Solver::Local::FluidRiemann::FluidRiemann  *riemannsolver  = nullptr; // 5-eqn
    Solver::Local::FluidRiemann::FluidRiemann7 *riemannsolver7 = nullptr; // 7-eqn (Phase C)

    // IntegrationScheme
    enum IntegrationScheme
    {
        ForwardEuler,
        SSPRK3,
        RK4
    };
    IntegrationScheme scheme;

    // EOS Parameters
    Solver::EOS::Tammann eos0; // Fluid 0
    Solver::EOS::Tammann eos1; // Fluid 1 

    // Interface Compression Stuff
    bool apply_sharpening = false;
    int sharpening_frequency = 1;
    int reinit_max_iter = 50;
    Set::Scalar reinit_tolerance = 1e-6;
    std::vector<int> step_counter; // Track steps per level
    int density_max_iter = 10;
    Set::Scalar density_tol = 1e-8;
    Set::Scalar omega_relax = 0.5;


    // NGHOST
    int nghost = 2;

    // ------------------------------------------------------------------
    // Flux registers for conservative reflux at coarse-fine boundaries.
    // flux_reg[lev] sits between level lev-1 (coarse) and lev (fine).
    // Element [0] is unused; valid indices are [1 .. max_level].
    // ------------------------------------------------------------------
    amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg;

    // Cell-centered flux MultiFabs storing hi-face Riemann fluxes.
    // cc_fluxes[lev].mass[d]: cell (i,j) stores mass flux at face (i+1/2,j)
    //   for d=0, or (i,j+1/2) for d=1.  1 component.
    // cc_fluxes[lev].mom[d]:  AMREX_SPACEDIM components (x-mom, y-mom).
    // cc_fluxes[lev].energy[d]: 1 component.  1 ghost cell for FillBoundary.
    // Converted to temporary face-centered MultiFabs in Advance() for
    // FluxRegister CrseInit/FineAdd.
    struct CellFluxes {
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> mass;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> mom;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> energy;
    };
    amrex::Vector<CellFluxes> cc_fluxes;  // indexed by level

    // ========================================================================
    // 7-EQUATION SWITCHES (Phase A scaffolding)
    //
    //   equation_count            : 5 (default, current Allaire model) or 7
    //                               (Baer-Nunziato per-phase)
    //   relax_pressure_stiff      : when 7, use SPB-2009 instantaneous
    //                               pressure relaxation between RK stages
    //   relax_velocity_stiff      : same, for the velocity-equilibrium step
    //   surface_tension_form      : 0 = Brackbill source (current default,
    //                                   matches existing 5-eqn behavior)
    //                               1 = Schmidmayer 2017 conservative
    //                                   capillary tensor (Phase D)
    // ========================================================================
    int equation_count        = 5;
    int relax_pressure_stiff  = 1;
    int relax_velocity_stiff  = 1;
    int surface_tension_form  = 0;
};
}

#endif
