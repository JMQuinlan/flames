#ifndef INTEGRATOR_HYDRO2_H
#define INTEGRATOR_HYDRO2_H

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
#include "Solver/Local/FluidRiemann/HLLC_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
// #include "Solver/Local/FluidRiemann/HLLE_WENO5.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
// #include "Solver/Local/FluidRiemann/HLLCE_WENO5.H"
// #include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"

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
        //delete eta_bc;
        delete temperature_bc;

        /// Inital Conditions
        // ETA
        delete eta_ic;
        delete etadot_ic;
        // PHASE 1 (liquid)
        delete density_1_ic;
        delete velocity_1_ic;
        delete pressure_1_ic;
        delete energy_1_ic;
        //delete temperature0_ic; // Temperature
        //delete k0_thermal_ic;   // Thermal Conductivity
        //delete h0_thermal_ic;   // Thermal Convectivity
        // PHASE 0 (gas)
        delete density_0_ic;
        delete velocity_0_ic;
        delete pressure_0_ic;
        delete energy_0_ic;
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



    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    // Stage 3: derive mixture (rho, M, E, p, v, a) from per-phase state.
    // Post-compute only; evolution uses per-phase state directly.
    void MixDiagnostic(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    void RHS(int lev,
        Set::Scalar time,
        amrex::MultiFab &eta_rhs_mf,
        amrex::MultiFab &rho_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho_mf,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf);

    // Stage 3: single-phase Euler RHS with a scalar (gamma, pi) EOS.
    // Produces drho/dt, dM/dt, dE/dt for one phase, using that phase's own
    // conserved state. No Allaire blend, no Kapila sources, no diagnostics.
    // Phase coupling (SQR source terms) is applied separately in Stage 4.
    // phase_idx: 0 ⇒ α_k = 1−η,  1 ⇒ α_k = η.  rho_other_mf is the OTHER
    // phase's (intrinsic) density field, needed at faces to form
    // ρ_mix_face for the mass-fraction Korteweg weight Y_k_face.
    void RHS_PerPhase(int lev,
        Set::Scalar time,
        int phase_idx,
        Set::Scalar gamma_k,
        Set::Scalar pi_k,
        Set::Scalar mu_k,
        Set::Scalar mu_k_b,
        amrex::MultiFab &rho_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        const amrex::MultiFab &rho_mf,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf,
        const amrex::MultiFab &eta_mf,
        const amrex::MultiFab &rho_other_mf);

    // Stage 3: η evolution (advection + CH + vaporization) driven by
    // per-phase conserved state. Advection uses a mass-flux-weighted
    // mixture velocity u_mix = (η·ρ1·u1 + (1-η)·ρ0·u0)/(η·ρ1 + (1-η)·ρ0)
    // with simple central-upwind at faces. No Kapila term.
    void RHS_Eta(int lev,
        Set::Scalar time,
        amrex::MultiFab &eta_rhs_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho0_mf_in,
        const amrex::MultiFab &M0_mf_in,
        const amrex::MultiFab &rho1_mf_in,
        const amrex::MultiFab &M1_mf_in);

    // Stage 4: Schmidt-Quinlan-Runnels (SQR) phase-coupling source terms, additive
    // contributions to the per-phase conservation laws over the diffuse
    // interface region. Uses sum-to-zero signs (phase 0 and phase 1
    // sources cancel in mixture sums) so total mixture mass/momentum/
    // energy are conserved identically. Inputs:
    //   m0_mf (prescribed interface mass flux, scalar)
    //   u0_mf (prescribed interface velocity, 2-comp)
    //   q_mf  (prescribed interface heat flux vector, 2-comp)
    // For inert fluid-fluid (m0.ic = 0, q.ic = 0) reduces to the
    // natural-BC pressure-balance momentum/energy coupling.
    void ApplySQRSource(int lev,
        Set::Scalar time,
        amrex::MultiFab &rho0_rhs_mf, amrex::MultiFab &M0_rhs_mf, amrex::MultiFab &E0_rhs_mf,
        amrex::MultiFab &rho1_rhs_mf, amrex::MultiFab &M1_rhs_mf, amrex::MultiFab &E1_rhs_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho0_mf_in, const amrex::MultiFab &M0_mf_in, const amrex::MultiFab &E0_mf_in,
        const amrex::MultiFab &rho1_mf_in, const amrex::MultiFab &M1_mf_in, const amrex::MultiFab &E1_mf_in);

    // Phase 6 (CH-Korteweg spec §1.8, §4.4): populate the CH-derived effective
    // SQR source fields (m0_eff_mf, u0_eff_mf, q_eff_mf) from the current
    // hydrodynamic + CH state. Always runs (called at the end of MixDiagnostic).
    // In SQRMode::CH_Emergent the integrator subsequently routes these into
    // m0_mf/u0_mf/q_mf so ApplySQRSource consumes them; in SQRMode::Prescribed
    // they are diagnostic-only (visible in plotfiles for comparison against
    // the prescribed values).
    void ComputeEffectiveSQRSources(int lev);

    // Phase 5 (CH-Korteweg spec §2.2, §4.3): well-balanced Korteweg capillary
    // force computed as the discrete divergence of the Korteweg stress tensor
    //   Sigma_K = -kappa * grad_eta (x) grad_eta
    //             + ( kappa * eta * lap_eta + (kappa/2) |grad_eta|^2 ) I
    // taken in face-flux form. First derivatives of eta are evaluated exactly
    // on faces (with vertex-averaging for the face-tangential component); the
    void WriteIntegrals(Set::Scalar time);
    void SetAdaptiveTimestepParams(Set::Scalar cfl_val, Set::Scalar min_dt, Set::Scalar max_dt, Set::Scalar growth_factor_val, bool enable_adaptive);
    void ApplyImplicitCH(int lev, Set::Scalar dt);
    void ApplyImplicitCH_Newton(int lev, Set::Scalar dt);
    // Phase 2 (CH-Korteweg spec §4.2): EOS-derived diagnostic
    //   dmu_bulk = f_g(rho_0, T_0) - f_l(rho_1, T_1)
    // and the spinodal-proximity ratio |dmu_bulk|/lambda. Purely diagnostic;
    // does NOT modify mu_chem_mf (that wiring belongs to Phase 4).
    void ComputeChemicalPotential(int lev);
    // Operator-split explicit-Euler heat-conduction step on per-phase energies.
    //   E_k(n+1) = E_k(n) + dt * k_k * Laplacian(T_k),   T_k = UE_k / (rho_k cv_k)
    // Per-phase k_eta{0,1} are constant scalars from input. No-op when
    // apply_heat_conduction == 0.
    void ApplyHeatConduction(int lev, Set::Scalar dt);
    // In-place floor on the per-phase conservative state. Enforces:
    //   rho_eta{0,1} >= small,    finite |M_eta{0,1}|,
    //   UE_0 = E_0 - KE_0 >= small             (CPG, ensures p_0 >= 0)
    //   UE_1 = E_1 - KE_1 >= pi_1 + small   (Tammann, ensures p_1 + pi >= 0
    //                                           so c_1^2 = gamma*(p+pi)/rho > 0)
    // Run after every operator-split write to the per-phase fabs. NaN-safe
    // (non-finite values are replaced, not just compared).
    void ClampPerPhaseState(int lev);
    void ComputeHybridCurvature(int lev);
    void ComputeSmoothNormals(int lev);
    void ComputeHeightFunction(int lev);
    void ComputeKappas(int lev);
    void ComputeGradEta(int lev);
    void ComputeHFcurvature_MUSCL(int lev);
    Set::Scalar InterpolateTamannP0(Set::Scalar eta, Set::Scalar Tamann_p_0_0, Set::Scalar Tamann_p_0_1);

    void ApplyNSCBC(int lev,
                    amrex::MultiFab& eta_mf,
                    amrex::MultiFab& rho_mf,
                    amrex::MultiFab& M_mf,
                    amrex::MultiFab& E_mf,
                    amrex::Real time);

protected:
    // PHASE 1 (liquid — stiffened gas)
    Set::Field<Set::Scalar> density_1_mf;
    Set::Field<Set::Scalar> density_1_old_mf;

    Set::Field<Set::Scalar> energy_1_mf;
    Set::Field<Set::Scalar> energy_1_old_mf;

    Set::Field<Set::Scalar> momentum_1_mf;
    Set::Field<Set::Scalar> momentum_1_old_mf;

    //Set::Field<Set::Scalar> T0_mf;
    //Set::Field<Set::Scalar> k0_thermal_mf;
    //Set::Field<Set::Scalar> h0_thermal_mf;
    //Set::Field<Set::Scalar> gamma_1_mf;

    Set::Field<Set::Scalar> velocity_1_mf;
    Set::Field<Set::Scalar> pressure_1_mf;
    Set::Field<Set::Scalar> vorticity_1_mf;
    Set::Field<Set::Scalar> c_1_mf;

    // PHASE 0 (gas — CPG)
    Set::Field<Set::Scalar> density_0_mf;
    Set::Field<Set::Scalar> density_0_old_mf;

    Set::Field<Set::Scalar> energy_0_mf;
    Set::Field<Set::Scalar> energy_0_old_mf;

    Set::Field<Set::Scalar> momentum_0_mf;
    Set::Field<Set::Scalar> momentum_0_old_mf;

    //Set::Field<Set::Scalar> T1_mf;
    //Set::Field<Set::Scalar> k1_thermal_mf;
    //Set::Field<Set::Scalar> h1_thermal_mf;
    //Set::Field<Set::Scalar> gamma_0_mf;
    Set::Field<Set::Scalar> velocity_0_mf;
    Set::Field<Set::Scalar> pressure_0_mf;
    Set::Field<Set::Scalar> vorticity_0_mf;
    Set::Field<Set::Scalar> c_0_mf;

    // Mixture
    Set::Field<Set::Scalar> density_mf;
    Set::Field<Set::Scalar> density_old_mf;
    Set::Field<Set::Scalar> vorticity_mf;
    Set::Field<Set::Scalar> momentum_mf;
    Set::Field<Set::Scalar> momentum_old_mf;
    Set::Field<Set::Scalar> energy_per_vol_mf;
    Set::Field<Set::Scalar> energy_per_mass_mf;
    Set::Field<Set::Scalar> energy_per_vol_old_mf;
    Set::Field<Set::Scalar> energy_per_mass_old_mf;
    Set::Field<Set::Scalar> velocity_mf;
    Set::Field<Set::Scalar> pressure_mf;

    Set::Field<Set::Scalar> T_mf;           // Temperature
    Set::Field<Set::Scalar> cp_mf;          // Constant Pressure Specific Heat
    Set::Field<Set::Scalar> cv_mf;          // Constant Volume Specific Heat
    Set::Field<Set::Scalar> k_thermal_mf;   // Thermal Conductivity
    Set::Field<Set::Scalar> h_thermal_mf;   // Thermal Convectivity
    Set::Field<Set::Scalar> gamma_mf;       // Specific Heat Ratio
    Set::Field<Set::Scalar> pi_mf;          // Tamman Pressure
    Set::Field<Set::Scalar> mu_chem_mf;     // Allen-Cahn Chemical Potential
    Set::Field<Set::Scalar> dmu_bulk_mf;       // EOS-only piece f_g(rho_0,T_0) - f_l(rho_1,T_1)  [diagnostic]
    Set::Field<Set::Scalar> dmu_over_lambda_mf;// |dmu_bulk| / ch_W_scale (spinodal-proximity, see spec §1.10)
    Set::Field<Set::Scalar> a_mf;           // Speed of sound
    Set::Field<Set::Scalar> Ma_mf;          // Mach
    Set::Field<Set::Scalar> UE_per_vol_mf;  // Internal Energy (per unit volume)
    Set::Field<Set::Scalar> UE_per_mass_mf; // Internal Energy (per unit mass)
    Set::Field<Set::Scalar> KE_per_vol_mf;  // Kinetic Energy (per unit volume)
    Set::Field<Set::Scalar> KE_per_mass_mf; // Kinetic Energy (per unit mass)

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

    Set::Field<Set::Scalar>     h_eta_mf;           // height field
    Set::Field<Set::Scalar>     nx_smoothed_mf;     // smoothed normals n_x
    Set::Field<Set::Scalar>     ny_smoothed_mf;     // smoothed normals n_y
    Set::Field<Set::Scalar>     eta_x_mf;           // gradient of eta in x direction
    Set::Field<Set::Scalar>     eta_y_mf;           // gradient of eta in y direction
    Set::Field<Set::Scalar>     gradmag_mf;         // mag(grad(eta))
    Set::Field<Set::Scalar>     kappa_SF_mf;        // curvature from smoothed normals
    Set::Field<Set::Scalar>     kappa_HF_mf;        // curvature from height function
    // Set::Field<Set::Scalar>     kappas_mf;          // weighted curvature

    Set::Field<Set::Scalar> rho_flux_mf;
    Set::Field<Set::Scalar> M_flux_mf;
    Set::Field<Set::Scalar> E_flux_mf;

    // Debugging
    Set::Field<Set::Scalar> div_tau_mf;
    Set::Field<Set::Scalar> hess_u_mf;
    Set::Field<Set::Scalar> tau_mf;     // viscous stress tensor (xx, xy, yy)
    Set::Field<Set::Scalar> Vap_dot_mf;


    Set::Field<Set::Scalar> m0_mf;
    Set::Field<Set::Scalar> u0_mf;
    Set::Field<Set::Scalar> q_mf;

    // Phase 6 (CH-Korteweg spec §1.8, §4.4): CH-emergent SQR source diagnostics.
    // Populated by Hydro2::ComputeEffectiveSQRSources at the end of MixDiagnostic
    // from the current CH state. In sqr_source_mode == CH_Emergent these are
    // copied into m0_mf/u0_mf/q_mf before ApplySQRSource runs; in Prescribed
    // mode they are populated as a diagnostic for offline comparison against
    // the user-prescribed values.
    //   m0_eff [kg/m^2/s] : from div(M*rho_mix*grad(mu_chem)) / max(|grad eta|,floor)
    //   u0_eff [m/s,2c]   : mass-flux-weighted mixture velocity
    //   q_eff  [W/m^2,2c] : m0_eff * L_vap(T) * n_hat
    Set::Field<Set::Scalar> m0_eff_mf;
    Set::Field<Set::Scalar> u0_eff_mf;
    Set::Field<Set::Scalar> q_eff_mf;

    // SQR source diagnostic plot fields (populated inside ApplySQRSource).
    //   sqr_dp_mf      [Pa,2c]      : per-phase pressure imbalance dp_k = ±(p_k - p_int)
    //   sqr_mom_src_mf [N/m^3,4c]   : per-phase momentum source dp_k * grad(eta)
    //   sqr_eng_src_mf [W/m^3,2c]   : per-phase energy source (p_k u_k - p* u*) * grad(eta)
    Set::Field<Set::Scalar> sqr_dp_mf;
    Set::Field<Set::Scalar> sqr_mom_src_mf;
    Set::Field<Set::Scalar> sqr_eng_src_mf;

protected:
    Set::Field<Set::Scalar> eta_mf;
    Set::Field<Set::Scalar> eta_mf_in;
    Set::Field<Set::Scalar> etadot_mf;
    Set::Field<Set::Scalar> eta_old_mf;
    Set::Field<Set::Scalar> hess_eta_mf;
    Set::Field<Set::Scalar> n_hat_mf;
protected:



    BC::BC<Set::Scalar>* density_bc = nullptr;
    BC::BC<Set::Scalar>* momentum_bc = nullptr;
    BC::BC<Set::Scalar>* energy_bc = nullptr;
    // Per-phase energy BCs. Default to a copy of energy_bc when no per-phase
    // input is given; problems like Stefan need different BCs on the spectator
    // liquid energy at a gas-side wall (and vice versa). When set,
    // energy_{0,1}_mf are registered against these instead of the shared
    // energy_bc; the mixture energy_per_vol_mf stays on energy_bc.
    BC::BC<Set::Scalar>* energy_0_bc = nullptr;
    BC::BC<Set::Scalar>* energy_1_bc = nullptr;
    // Per-phase density BCs. Same pattern as the per-phase energy BCs above:
    // default to the shared density_bc; problems like Stefan that pin gas-phase
    // density at a heated wall (to keep T = UE/(ρ·cv) sensible under fixed-UE
    // Dirichlet) need to override only one phase without polluting the spectator.
    BC::BC<Set::Scalar>* density_0_bc = nullptr;
    BC::BC<Set::Scalar>* density_1_bc = nullptr;
    //BC::BC<Set::Scalar>* eta_bc = nullptr;
    BC::BC<Set::Scalar>* temperature_bc = nullptr;

    IC::IC<Set::Scalar>* velocity_1_ic = nullptr;
    IC::IC<Set::Scalar>* pressure_1_ic = nullptr;
    IC::IC<Set::Scalar>* density_1_ic  = nullptr;
    IC::IC<Set::Scalar>* energy_1_ic = nullptr;
    //IC::IC<Set::Scalar>* temperature0_ic  = nullptr;
    //IC::IC<Set::Scalar>* k0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma_1_ic  = nullptr;

    IC::IC<Set::Scalar>* velocity_0_ic = nullptr;
    IC::IC<Set::Scalar>* pressure_0_ic = nullptr;
    IC::IC<Set::Scalar>* density_0_ic = nullptr;
    IC::IC<Set::Scalar>* energy_0_ic = nullptr;
    //IC::IC<Set::Scalar>* temperature1_ic = nullptr;
    //IC::IC<Set::Scalar>* k1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma_0_ic  = nullptr;

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
    // PHASE 1 (liquid — stiffened gas)
    Set::Scalar gamma_1 = NAN, pi_1 = NAN, mu_1 = NAN, mu_1_b = NAN, MW0 = NAN;
    Set::Scalar b0 = NAN, q0EOS = NAN;
    Set::Scalar cp_1 = NAN, cv_1 = NAN;
    // Free-energy calibration parameters for phase 1 (Tammann)
    // q_1: reference internal energy [J/kg]; qprime_1: entropy reference [J/(kg·K)];
    // rho_ref_1: reference density [kg/m^3] used in the ln(rho/rho_ref) term.
    Set::Scalar q_1 = 0.0, qprime_1 = 0.0, rho_ref_1 = 1.0;

    // PHASE 0 (gas — CPG)
    Set::Scalar gamma_0 = NAN, pi_0 = NAN, mu_0 = NAN, mu_0_b = NAN, MW1 = NAN;
    Set::Scalar b1 = NAN, q1EOS = NAN;
    Set::Scalar cp_0 = NAN, cv_0 = NAN;
    // Free-energy calibration parameters for phase 0 (CPG)
    // R_0: specific gas constant [J/(kg·K)] for the ideal-gas free energy.
    Set::Scalar q_0 = 0.0, qprime_0 = 0.0, rho_ref_0 = 1.0, R_0 = 287.0;

    // Shared free-energy reference temperature [K]
    Set::Scalar T_ref = 300.0;

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
    Set::Scalar small=NAN;
    Set::Scalar cutoff=NAN;
    Set::Scalar lagrange=NAN;

    // Booleans
    // Master interfacial-stress kill switch. Default = 1.
    // Set 0 to zero out the flux-form Korteweg + Q-trace contribution in
    // RHS_PerPhase (κ_face → 0, q_eta → 0 inside KortewegFaceData), which
    // disables ALL interfacial stresses delivered by the new HLLC path.
    // The legacy Brackbill block in ApplySQRSource is dead and not gated by
    // this flag.
    int apply_surface_tension = 1;
    int apply_weight = 0;
    int apply_viscous = 0;     // Per-phase viscous stress in RHS_PerPhase (constant mu_k, mu_k_b)
    // Per-phase heat conduction (operator-split explicit-Euler step).
    // CFL constraint: dt < 0.5 * dx^2 / max(k_k / (rho_k cv_k)).
    int apply_heat_conduction = 0;
    Set::Scalar k_0 = 0.0;  // gas-phase thermal conductivity [W/(m K)]
    Set::Scalar k_1 = 0.0;  // liquid-phase thermal conductivity [W/(m K)]
    int static_eta = 0;
    int apply_vaporization = 0;

    // Phase 6 (CH-Korteweg spec §3.3): SQR phase-coupling source mode.
    //   Off          : no phase-coupling sources at all (single-phase limit / debugging).
    //   Prescribed   : default; m0_mf/u0_mf/q_mf set from ic_m0/ic_u0/ic_q (legacy
    //                  apply_sqr_source=1 behavior).
    //   CH_Emergent  : m0_mf/u0_mf/q_mf are overwritten each step with the CH-derived
    //                  effective values (m0_eff_mf etc.) before ApplySQRSource runs.
    // Replaces the legacy boolean `apply_sqr_source`. The legacy key is still
    // accepted at parse time and mapped onto Off/Prescribed for back-compat.
    enum class SQRMode { Off = 0, Prescribed = 1, CH_Emergent = 2 };
    SQRMode sqr_source_mode = SQRMode::Prescribed;
    // If non-zero, re-evaluate ic_m0/ic_u0/ic_q each Advance step at the current
    // simulation time. Used for prescribed-flux validation problems (e.g. Stefan)
    // where the analytical mass flux scales as ~1/sqrt(t).
    int sqr_sources_time_dependent = 0;
    int Spec_Vol = 1;

    // Interface-pressure closure for the SQR source (p_int).
    // 0: OFF — no pressure-imbalance coupling (mass/heat/ST terms still apply)
    // 1: arithmetic mean          p_int = 0.5 (p_0 + p_1)
    // 2: eta-weighted             p_int = (1-eta) p_0 + eta p_1
    // 3: acoustic-impedance       p_int = (Z_1 p_0 + Z_0 p_1)/(Z_0 + Z_1), Z_k = rho_k c_k
    // Same weighting is applied to u_int used in the energy source.
    // Legacy path: only consulted when sqr_legacy_p_int_method = true (Phase B).
    int p_int_method = 0;

    // === Closure-strategy framework (Phase B onwards) ===
    // For each per-phase nonequilibrium quantity (rho, u_n, u_t, E — rho/u_t
    // added in Phase C/D) the interface flux J_Q is computed by one of
    // these strategies. The chosen closure must vanish at full equilibrium
    // (matching intensives across the interface), otherwise the per-phase
    // momentum/energy ground state is corrupted under the two-full-states LHS
    // (spurious-wave radiation in transients — see acoustic_1d).
    //
    //   Constitutive (default): symmetric pressure/velocity-imbalance flux,
    //                      S_M^k = (p_k - p_int) ∇η,
    //                      S_E^k = (p_k u_k - p_int u_int) · ∇η,
    //                      with the interface average (p_int, u_int) selected
    //                      per quantity by `pint_method`:
    //                        1: arithmetic mean (default)
    //                        2: eta-weighted
    //                        3: acoustic-impedance weighted
    //                        4: mass-fraction weighted
    //                      Vanishes at p_0=p_1 (resp. p_0u_0=p_1u_1), so the
    //                      per-phase pressure ground state survives. Same form
    //                      as the legacy p_int closure but lives inside the
    //                      new framework rather than gated on
    //                      sqr_legacy_p_int_method.
    //   Pillbox     (S1): theory's per-phase constitutive flux from the
    //                      pillbox SQR derivation. Asymmetric — phase 0 sees
    //                      -p_1 ∇η, phase 1 sees +p_0 ∇η; analogous H_k u_k
    //                      flux for energy. Theory-faithful in the eps -> 0
    //                      limit, but at finite eps it does *not* vanish at
    //                      mechanical equilibrium and forces ∇p_k = ±p_other
    //                      ∇η, driving phase pressures away from atmospheric
    //                      across the band. Research mode only.
    //   PillboxSymmetric (S1', Phase B.5):
    //                      antisymmetrized variant of Pillbox. p_0, p_1 are
    //                      replaced by p_bar = (p_0+p_1)/2 (and analogously
    //                      H_k u_k by their arithmetic mean) so the per-cell
    //                      momentum / energy split is strict action-reaction.
    //                      Same ground-state pathology as Pillbox — kept as a
    //                      research/comparison option.
    //   Jump        (S2): J_Q = -lambda * jump(intensive)  (Lagrange-multiplier
    //                      transfer coefficient on the conjugate intensive
    //                      variable: u_n, T, etc.). Symmetric — strict
    //                      action-reaction on the band by construction. Vanishes
    //                      at intensive equilibrium.
    //   Relax       (S3): J_Q = -C_Q * v_Q * jump(intensive)  (capacity-scaled
    //                      transfer flux; chi_Q = C_Q v_Q is a genuine per-area
    //                      transfer coefficient — heat-transfer coeff / impedance
    //                      / mass-transfer coeff). Symmetric. Through the |grad
    //                      eta| flux template the intensive jump decays at the
    //                      band-local rate v_Q*|grad eta| (emergent relaxation
    //                      time tau_loc ~ eps/v_Q). C_Q from harmonic (series)
    //                      means: rho_eff = rho_0 rho_1/(rho_0+rho_1) for u_n/u_t;
    //                      C_E = (rho_0 cv_0)(rho_1 cv_1)/(rho_0 cv_0+rho_1 cv_1) for E;
    //                      C_rho = chi_0 chi_1/(chi_0+chi_1), chi_k = rho_k^2/p_k for rho.
    //                      Vanishes at intensive equilibrium. v_Q -> inf is the
    //                      S1/continuity corner; for u_n (kinematic jump(u_n)=0 is
    //                      exact) finite v_Q is a stiff penalty, not a physical
    //                      finite-resistance interface.
    // Default is Constitutive — vanishes at equilibrium, so pressure agreement
    // emerges automatically through the EOS once velocity / temperature have
    // equilibrated. Pillbox / PillboxSymmetric are kept for research only.
    enum class ClosureStrategy {
        Constitutive     = 0,
        Jump             = 1,
        Relax            = 2,
        Pillbox          = 3,
        PillboxSymmetric = 4
    };
    struct ClosureSpec {
        ClosureStrategy strategy = ClosureStrategy::Constitutive;
        Set::Scalar lambda = 0.0;  // S2 transfer coefficient (must be > 0 if strategy=Jump)
        Set::Scalar vel    = 0.0;  // S3 transfer velocity v_Q (must be > 0 if strategy=Relax)
        int pint_method    = 1;    // Constitutive averaging: 1 arith, 2 eta, 3 Z, 4 mass-fraction
    };
    ClosureSpec un_close;   // normal-velocity closure (Phase B)
    ClosureSpec E_close;    // energy/temperature closure (Phase B; Fourier flux added in Phase E)
    ClosureSpec rho_close;  // density / phase-change closure: Constitutive = mdot0 from existing m0_mf (CH-emergent / prescribed); Jump/Relax = dmu_bulk-driven mass flux.
    ClosureSpec ut_close;   // tangential-velocity closure: Constitutive = angular-momentum no-slip (S1, currently a no-op stub — A10 not yet implemented); Jump/Relax = ⟦u_t⟧-driven flux.

    // If true, ApplySQRSource uses the legacy p_int_method-averaged momentum
    // and energy sources instead of the new theory-faithful S1/S2/S3 closure
    // path. Enables the kill-switch regression (set p_int_method=2 +
    // sqr_legacy_p_int_method=1 to reproduce the pre-Phase-B baseline
    // bit-for-bit). Mass and prescribed-heat-flux paths are unchanged.
    bool sqr_legacy_p_int_method = false;

    int kappa_method = 1;        // 1: Smooth Normals (default)  2: Height Function  3: Hybrid
    int smooth_kernel_size = 3;  // Gaussian kernel for normal smoothing: 3 (3x3, default) or 5 (5x5)
    int nghost = 2;              // Ghost cell depth for MultiFab allocation and HF integration band

    // Implicit CH
    int implicit_ch = 0;               // 0: explicit  1: Eyre MLMG  2: Newton-MLMG
    Set::Scalar ch_mobility_nom = 0.0; // nominal mobility for implicit solve (0 = auto)
    int ch_newton_iters = 5;           // max Newton iterations (implicit_ch=2)
    Set::Scalar ch_newton_tol = 1.0e-10; // Newton convergence tolerance
    Set::Scalar ch_W_scale = 1.0;        // double-well amplitude scale (W_scale=1 → equilibrium width=√2·ε)
    // Phase 2 / Phase 4: enables EOS-derived contribution dmu_bulk in both
    // the diagnostic field and (Phase 4) the f'(eta) used by the CH solver.
    // Default OFF: identical behavior to the legacy generic double-well.
    int ch_thermo_consistent = 0;
    // Phase 4 mobility scaling for the implicit CH solver (spec §1.7, §3.2):
    //   "magaletti"  -> M = ch_mobility_nom * epsilon^2  (default; matched-asymptotic)
    //   "allen_cahn" -> M = ch_mobility_nom * epsilon
    //   "explicit"   -> M = ch_mobility_nom (no scaling)
    std::string ch_mobility_scaling = "magaletti";
    Set::Scalar ch_mobility_factor  = 1.0;   // computed in Parse from above + epsilon

    // Degenerate (interface-localized) mobility: M(eta) = M_nom * eta^2*(1-eta)^2.
    // Peak at eta=0.5 is M_nom/16; bulk (eta=0 or 1) is exactly 0. This kills CH
    // dynamics in bulk regions where dmu_bulk can be non-zero from spectator-phase
    // drift (spec §6.3) and confines phase change to the interface.
    // Currently honored only by ApplyImplicitCH_Newton (implicit_ch=2).
    // TODO: extend to ApplyImplicitCH (Eyre, implicit_ch=1) and the explicit
    // RHS_Eta path (implicit_ch=0) when those modes are needed.
    int ch_mobility_degenerate = 0;

    // CH gradient-energy coefficient kappa (controls equilibrium interface width
    // delta = sqrt(kappa/(2*lambda)) and surface tension sigma = sqrt(2*kappa*lambda)/6).
    // Historically hardwired to epsilon^2; now an independent input so we can
    // raise lambda (=ch_W_scale) to satisfy the spinodal margin |dmu|/lambda < 0.3
    // without collapsing the interface to sub-grid widths.
    //
    // ch_kappa = 0 (default): auto = epsilon^2 (legacy behavior, backward compat).
    // ch_kappa > 0          : use this value directly.
    // ch_kappa_eff is the resolved value used at every CH site (computed in Parse).
    Set::Scalar ch_kappa     = 0.0;   // input; 0 means "auto = epsilon^2"
    Set::Scalar ch_kappa_eff = 0.0;   // resolved value used in code (set in Parse)

    int write_integrals = 1;

    // Adaptive Timestep
    bool adaptive_timestep = false;
    Set::Scalar dt_min = NAN, dt_max = NAN, dt_growth = NAN;

    // Limiters
    int Limiter = 0;
    //Solver::Local::Limiter::Minmod *limiter_minmod = nullptr;
    //Solver::Local::Limiter::VanLeer *limiter_vanleer = nullptr;

    // Solvers
    Solver::Local::FluidRiemann::FluidRiemann *riemannsolver = nullptr;

    // IntegrationScheme
    enum IntegrationScheme
    {
        ForwardEuler,
        SSPRK3,
        RK4
    };
    IntegrationScheme scheme;

    // -----------------------------------------------------------------------
    // Ghost Cell Navier-Stokes Characteristic Boundary Conditions (GC-NSCBC)
    // Reference: Motheau et al. (2017), doi:10.1016/j.jcp.2017.01.010
    // -----------------------------------------------------------------------
    struct NSCBCParams {
        enum class FaceType { NONE = 0, OUTFLOW = 1, INFLOW = 2 };

        bool enabled = false;

        // Per-face type. Index: 0=xlo, 1=xhi, 2=ylo, 3=yhi
        FaceType face_type[4] = {FaceType::NONE, FaceType::NONE,
                                  FaceType::NONE, FaceType::NONE};

        // Outflow relaxation parameters (Motheau Eq. 38)
        Set::Scalar sigma = 0.25;  // pressure relaxation strength
        Set::Scalar beta  = 0.0;   // transverse term weight: 0 = full Lodato (recommended), 1 = pure LODI (no transverse)

        // Domain lengths used for the K relaxation coefficient.
        // Auto-set from geometry if not specified (0 = auto).
        Set::Scalar Lx = 0.0, Ly = 0.0;

        // Per-face target values (physical units)
        Set::Scalar p_t [4] = {0.0, 0.0, 0.0, 0.0}; // target pressure
        Set::Scalar un_t[4] = {0.0, 0.0, 0.0, 0.0}; // target normal velocity
        Set::Scalar ut_t[4] = {0.0, 0.0, 0.0, 0.0}; // target tangential velocity
        Set::Scalar T_t [4] = {0.0, 0.0, 0.0, 0.0}; // target temperature

        // Per-face inflow relaxation damping (Motheau Eq. 42-44)
        Set::Scalar eta_relax[4] = {2.0, 2.0, 2.0, 2.0};
    } nscbc;

};
}

#endif
