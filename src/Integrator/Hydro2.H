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
        delete hll_solver;
        delete nscbc7_bc;
    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void WriteIntegrals(Set::Scalar time);  // append volume-integrated conserved quantities to integrals.dat
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    void PostSubcycleReflux(int lev, Set::Scalar time, Set::Scalar dt_coarse) override;
    void RHS(int lev,
        Set::Scalar time,
        Set::Scalar dt,
        amrex::MultiFab &alpharho0_rhs_mf,
        amrex::MultiFab &alpharho1_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        amrex::MultiFab &eta_rhs_mf,
        amrex::MultiFab &rho_vap_rhs_mf,
        const amrex::MultiFab &alpharho0_mf_in,
        const amrex::MultiFab &alpharho1_mf_in,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho_vap_mf_in);

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
    Set::Field<Set::Scalar> Y_mf;           // Mass Fraction (gas mass fraction of total = alpharho0/rho)
    Set::Field<Set::Scalar> mass_frac_v_mf; // vapor mass fraction OF THE GAS = rho_vap/alpharho0 (Stage 3 diagnostic + Spalding/EOS input)
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
    // HLLC-ADC pressure-ratio shock locator output. Two components: omega at
    // the cell's hi-face in x (i+1/2,j) and y (i,j+1/2). Written every RHS
    // regardless of which Riemann solver is active so the shock locator can
    // be inspected as a diagnostic.
    Set::Field<Set::Scalar> omega_adc_mf;
    // Diagnostic: per hi-face relative deviation of the *used* Riemann flux from
    // the pure-HLL flux, max over the 4 conserved components. ~0 means the face
    // produced a genuine HLL flux (confirms a tagged/omega=0 cell took the HLL
    // path with all components blended); >0 means HLLC antidiffusion is still
    // present (untagged face, or adc_components<4 leaking energy/tangent).
    Set::Field<Set::Scalar> flux_dev_mf;
    // Diagnostic: the raw Ducros-weighted compression shock sensor per cell
    // (the continuous value compared against adc_shock_threshold), written so the
    // threshold can be tuned by inspecting the field directly.
    Set::Field<Set::Scalar> shock_sensor_mf;
    // PP-limiter diagnostics (only meaningful when pp_flux_limiter != 0).
    // pp_theta: per hi-face blend factor theta in [0,1] (1 = full high-order
    //   HLLC, 0 = full HLL). Shows where the limiter is dissipating.
    // pp_dmass: the antidiffusive mass flux removed at the cell's hi-faces,
    //   (1-theta)*|F_HLLC - F_HLL| (max over x/y). This is the mass-transport
    //   change the conserved (rho,M,E) update sees but the unblended eta
    //   advection does not -- the interface desync source.
    // pp_efloor: internal energy per volume injected by the EOS positivity
    //   backstop, max(ue_floor - UE_raw, 0). Nonzero = the absolute-pressure
    //   floor (eps_p) fired in that cell (the liquid "pressure flooring").
    Set::Field<Set::Scalar> pp_theta_mf;
    Set::Field<Set::Scalar> pp_dmass_mf;
    Set::Field<Set::Scalar> pp_efloor_mf;


    Set::Field<Set::Scalar> m0_mf;
    Set::Field<Set::Scalar> u0_mf;
    Set::Field<Set::Scalar> q_mf;

protected:
    Set::Field<Set::Scalar> eta_mf;

    Set::Field<Set::Scalar> alpharho0_mf;
    Set::Field<Set::Scalar> alpharho1_mf;
    Set::Field<Set::Scalar> alpharho0_old_mf;
    Set::Field<Set::Scalar> alpharho1_old_mf;

    // Stage 3 (inert carrier + Spalding): dodecane-vapor partial density within the
    // gas phase. gas (eta=1) = inert carrier (eos0) + dodecane vapor. rho_vap is
    // conserved (advects with the mixture mass flux + gains the evaporation source);
    // Y_v = rho_vap/alpharho0 is the vapor mass fraction of the gas; carrier mass =
    // alpharho0 - rho_vap. Active only when species_transport != 0.
    Set::Field<Set::Scalar> rho_vap_mf;
    Set::Field<Set::Scalar> rho_vap_old_mf;

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

    Set::Scalar eta_refinement_criterion=NAN, omega_refinement_criterion=NAN, gradu_refinement_criterion=NAN, divu_refinement_criterion=NAN, rho_refinement_criterion=NAN;
    Set::Scalar eta_band_refinement=NAN; // value-based band tag: refine where min(eta,1-eta) > this
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
    Set::Scalar L_film = 1.0; // Spalding film/boundary-layer thickness [m] in m'' = (rho_g*Dv/L_film)*B'. PHYSICAL diffusion length, distinct from the band localizer |grad eta|. Fixed L_film = quasi-steady (constant rate); transient sqrt(t) Stefan needs L -> delta(t)~sqrt(Dv*t). Default 1.0 reproduces the legacy under-dimensioned magnitude.
    // Chapman-Enskog variable gas diffusivity. When rhoD_ref > 0 the vapor
    // transport coefficient is the gas product rho*D(T) = rhoD_ref*(T/rhoD_Tref)^rhoD_exp
    // [kg/m/s] (CE binary: D~T^1.5/p, rho~p/T => rho*D~T^0.5, pressure-independent),
    // shared by BOTH the bulk Fickian vapor diffusion (face coeff eta*rho*D) and the
    // Spalding film source (rho*D at the interface T). rhoD_ref <= 0 keeps the legacy
    // constant-Dv path (bulk: alpharho0*Dv; source: rho_g*Dv).
    Set::Scalar rhoD_ref = 0.0;    // reference gas rho*D [kg/m/s] (0 = legacy Dv path)
    Set::Scalar rhoD_Tref = 300.0; // reference temperature [K] for rho*D(T)
    Set::Scalar rhoD_exp = 0.5;    // T exponent of rho*D; Chapman-Enskog = 0.5
    Set::Scalar k0_thermal = 0.0; // gas-phase thermal conductivity [W/m/K] (Fourier conduction; 0 = off)
    Set::Scalar k1_thermal = 0.0; // liquid-phase thermal conductivity [W/m/K] (Fourier conduction; 0 = off)
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
    int spalding_saturation = 0;       // Stage 2: 1 = Spalding driving force = Antoine saturation Y_s at interface T; 0 = legacy local cell-Y. See SaturationYs.
    int spalding_film_sink = 0;        // Stage 3d: 0 = constant far-field Y_infinity as the Spalding sink (fixed driving -> constant rate). 1 = sample the sink at a FILM distance into the gas (transport-responsive -> sqrt(t) Stefan recession). Requires species_transport.
    Set::Scalar film_eps_mult = 3.0;   // film sink distance ell = film_eps_mult * epsilon, sampled along +n_hat (into the gas). Needs dx >= ell/nghost so the sample clears the diffuse band.
    int apply_vap_const = 0;           // Verification: prescribed constant-rate phase change (bypasses Spalding closure). 0 = off.
    Set::Scalar vap_const_mdot = 0.0;  // imposed interfacial mass flux [kg/m^2/s]; m_dot_Vap = vap_const_mdot * |grad eta| (liquid->vapor if > 0)
    int apply_cavitation = 0;          // HRM bulk cavitation: liquid -> vapor at finite rate when p < p_sat(T). 0 = off.
    Set::Scalar tau_cav = 1.0e-4;      // cavitation relaxation time [s]
    Set::Scalar antoine_A = 4.10549;   // Antoine: log10(p_sat[bar]) = A - B/(T[K] + C)  (n-dodecane, 399-490 K)
    Set::Scalar antoine_B = 1625.928;
    Set::Scalar antoine_C = -92.839;
    Set::Scalar L_vap = 256.0e3;       // latent heat of vaporization [J/kg]; phase-change energy coupling (Stage 3c). Applied only when apply_latent_heat=1.
    int apply_latent_heat = 0;         // Stage 3c: subtract L_vap*m_dot_Vap from the mixture energy (evaporative cooling). 0 = off (Phase-A mass-only behavior).

    // Stage 3: inert-carrier (air) gas + transported dodecane-vapor species.
    int species_transport = 0;         // 0 = off (gas = eos0 directly). 1 = gas is carrier(eos0)+vapor, track rho_vap.
    Set::Scalar Yv_init = 0.0;         // initial vapor mass fraction in the gas (eta=1) region
    Set::Scalar cp_vap = 1994.85;      // dodecane-vapor cp [J/kg/K] (carrier props come from eos0)
    Set::Scalar cv_vap = 1950.0;       // dodecane-vapor cv [J/kg/K]
    int Spec_Vol = 1;

    int kappa_method = 1;
    int smooth_kernel_size = 3;
    Set::Scalar kappa_grad_frac = 0.02;

    int write_integrals = 1;  // 1: append volume-integrated conserved quantities to <plot_file>/integrals.dat each step
    // Cumulative liquid->vapor mass transformed by the phase-change source (INT INT m_dot_Vap dV dt),
    // accumulated trapezoidally in WriteIntegrals from the volume-integrated rate. IOProcessor copy only.
    Set::Scalar m_vap_transformed = 0.0;  // running total [kg] (integrals.dat column 11)
    Set::Scalar wi_mdot_prev = 0.0;       // previous volume-integrated m_dot_Vap [kg/s] (trapezoidal state)
    Set::Scalar wi_time_prev = -1.0;      // previous WriteIntegrals time [s] (-1 = uninitialized)
    // Cumulative positivity-floor mass injection [kg], split by phase partition, to measure the
    // density-ratio asymmetry of the rho_eta floor leak (integrals.dat columns 12/13).
    Set::Scalar m_floor_gas = 0.0;        // alpharho0 (gas partition) floor injection
    Set::Scalar m_floor_liq = 0.0;        // alpharho1 (liquid partition) floor injection
    Set::Scalar e_floor = 0.0;            // cumulative internal energy injected [J] by the eps_p
                                          // pressure backstop (integrals.dat column 14)
    // Phase-split of the backstop injection (eta>=0.5 gas-dominated, else liquid),
    // to tell whether the energy leak fires in expanding air or in tensioned
    // liquid, and how negative the pre-floor pressure gets in each. The two
    // p_preflr_min_* are all-time running minima of the pre-floor (unclamped)
    // pressure among cells the floor actually touched: a modest negative value on
    // the liquid side means physical tension is being clipped (raise p_cav); a
    // huge negative value (or near-zero/negative UE) means an upstream overshoot.
    Set::Scalar e_floor_gas = 0.0;        // cumulative injection [J] in gas-dominated cells (col 15)
    Set::Scalar e_floor_liq = 0.0;        // cumulative injection [J] in liquid-dominated cells (col 16)
    Set::Scalar p_preflr_min_gas = 1e30;  // all-time min pre-floor pressure [Pa], floored gas cells (col 17)
    Set::Scalar p_preflr_min_liq = 1e30;  // all-time min pre-floor pressure [Pa], floored liquid cells (col 18)

    // Exponent on the HLLC-ADC pressure-ratio shock locator. Reflected in the
    // omega_adc_mf diagnostic regardless of which Riemann solver is active.
    // Set via Riemann_Solver.alpha (same key consumed by the HLLC_ADC solver).
    Set::Scalar omega_adc_alpha = 3.0;

    // Positivity-preserving flux limiter (Hu-Adams-Shu 2013, building on
    // Perthame-Shu 1996 / Zhang-Shu 2010). Blends the selected Riemann flux
    // toward the HLL flux per face so the Forward-Euler update keeps every cell
    // in the admissible set { rho > eps_rho, p + p0 >= eps_p }.
    int pp_flux_limiter = 1;
    // When set, fold dt*Source into the positivity baseline B so theta also
    // guards against the source driving a cell out of the admissible set.
    // Clear (pp_source_limit = 0) to limit the flux only and add the source
    // unlimited, recovering the pure Hu-Adams-Shu flux limiter.
    int pp_source_limit = 1;
    // Optional positivity-preserving SOURCE limiter (requires pp_flux_limiter).
    // When set: (1) the flux baseline is taken source-free so theta limits the
    // flux only, (2) the per-face theta additionally keeps the per-phase partial
    // densities alpharho0, alpharho1 >= small (not just the mixture rho), and
    // (3) the cell source is scaled by the largest s in [0,1] that keeps the
    // flux-updated state admissible (mixture rho,p AND both partial densities).
    // This replaces hard pressure/partial-density flooring (which injects mass
    // and energy non-conservatively) with the minimal source throttle needed to
    // stay positive. Default 0 (off) -> legacy pp_source_limit behavior.
    int pp_source_limiter = 0;
    Set::Scalar eps_p = NAN;    // pressure floor on p (gas side, eta -> 1)
    Set::Scalar eps_rho = NAN;  // density floor
    Set::Scalar p_cav = 0.0;    // liquid cavitation/tensile floor: hold the liquid (eta -> 0) at p >= -p_cav while the gas (eta -> 1) stays p >= eps_p. 0 = vacuum floor (no tension).

    // HLLC-ADC carbuncle cure driven by a Ducros-weighted div(u) compression
    // shock tag (carbuncles are pressure-quiet, so the internal pressure-ratio
    // sensor misses them). A cell is tagged when the sensor exceeds
    // adc_shock_threshold, then the tag is grown by adc_grow cells so the HLL
    // dissipation covers the post-shock band. adc_grow < 0 disables the tag and
    // falls back to the solver's pressure-ratio omega.
    Set::Scalar adc_shock_threshold = 0.05;
    int adc_grow = 1;
    int hll_first_steps = 0; // force full HLL (omega=0) for the first N steps, then resume the shock-tag blend (skips the step-1 carbuncle seed)

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
    // Low-order, positivity-preserving flux for the PP flux limiter (always HLL).
    Solver::Local::FluidRiemann::FluidRiemann  *hll_solver     = nullptr;

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
    // cc_fluxes[lev].mass[d]: cell (i,j) stores per-phase mass flux at face
    //   (i+1/2,j) for d=0, or (i,j+1/2) for d=1.  3 components: alpharho0,
    //   alpharho1, rho_vap (vapor advective flux; comp 2 written only when
    //   species_transport != 0, otherwise it stays 0 and refluxes as a no-op).
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

    // ------------------------------------------------------------------
    // Scratch storage for the positivity-preserving flux limiter.
    // All cell-centered, storing the hi-face value (cell i holds face i+1/2
    // for direction d), 1 ghost so FillBoundary supplies each cell's lo-face.
    //   Fhi[d]: high-order (selected solver) flux, 5 comp = mass, x-mom,
    //           y-mom (fixed frame), energy, u_interface.
    //   Flo[d]: low-order HLL flux, 5 comp = mass, x-mom, y-mom, energy,
    //           u_interface (HLL contact speed, for blended eta advection).
    //   Bbase : low-order positivity baseline, 6 comp = rho, Mx, My, E,
    //           alpharho0, alpharho1 (last two only used by the source limiter's
    //           per-phase guard; source folded in unless pp_source_limiter).
    //   theta : per-face blend factor, AMREX_SPACEDIM comp (theta at i+1/2, j+1/2).
    // ------------------------------------------------------------------
    struct LimiterScratch {
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> Fhi;
        amrex::Array<std::unique_ptr<amrex::MultiFab>, AMREX_SPACEDIM> Flo;
        std::unique_ptr<amrex::MultiFab> Bbase;
        std::unique_ptr<amrex::MultiFab> theta;
    };
    amrex::Vector<LimiterScratch> pp_scratch;  // indexed by level

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
