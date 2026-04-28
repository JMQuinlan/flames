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
        // PHASE eta=1 (liquid)
        delete density_eta1_ic;
        delete velocity_eta1_ic;
        delete pressure_eta1_ic;
        delete energy_eta1_ic;
        //delete temperature0_ic; // Temperature
        //delete k0_thermal_ic;   // Thermal Conductivity
        //delete h0_thermal_ic;   // Thermal Convectivity
        // PHASE eta=0 (gas)
        delete density_eta0_ic;
        delete velocity_eta0_ic;
        delete pressure_eta0_ic;
        delete energy_eta0_ic;
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
    // Phase coupling (BS source terms) is applied separately in Stage 4.
    void RHS_PerPhase(int lev,
        Set::Scalar time,
        Set::Scalar gamma_k,
        Set::Scalar pi_k,
        Set::Scalar mu_k,
        Set::Scalar mu_k_b,
        amrex::MultiFab &rho_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        const amrex::MultiFab &rho_mf,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf);

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

    // Stage 4: Boyd-Schmidt (BS) phase-coupling source terms, additive
    // contributions to the per-phase conservation laws over the diffuse
    // interface region. Uses sum-to-zero signs (phase 0 and phase 1
    // sources cancel in mixture sums) so total mixture mass/momentum/
    // energy are conserved identically. Inputs:
    //   m0_mf (prescribed interface mass flux, scalar)
    //   u0_mf (prescribed interface velocity, 2-comp)
    //   q_mf  (prescribed interface heat flux vector, 2-comp)
    // For inert fluid-fluid (m0.ic = 0, q.ic = 0) reduces to the
    // natural-BC pressure-balance momentum/energy coupling.
    void ApplyBSSource(int lev,
        Set::Scalar time,
        amrex::MultiFab &rho0_rhs_mf, amrex::MultiFab &M0_rhs_mf, amrex::MultiFab &E0_rhs_mf,
        amrex::MultiFab &rho1_rhs_mf, amrex::MultiFab &M1_rhs_mf, amrex::MultiFab &E1_rhs_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho0_mf_in, const amrex::MultiFab &M0_mf_in, const amrex::MultiFab &E0_mf_in,
        const amrex::MultiFab &rho1_mf_in, const amrex::MultiFab &M1_mf_in, const amrex::MultiFab &E1_mf_in);

    // Phase 5 (CH-Korteweg spec §4.3): well-balanced Korteweg capillary force
    // from the face-flux form  F_cap_face = -eta_face * (mu_chem_j - mu_chem_i)/dx.
    // The cell-centered divergence is added to per-phase momentum (eta-weighted)
    // and per-phase energy gets F_cap . u_k. Reads mu_chem_mf as populated by
    // the post-step visualization block in Advance() (mu_chem = -kappa*lap_eta
    //  + W_scale*W'(eta)). Ghost cells are FillBoundary'd internally before use.
    // Does NOT recompute mu_chem from a different stencil — well-balancing
    // depends on a single source of truth.
    // Only fires when korteweg_well_balanced != 0; in that case the Brackbill
    // Fsv block in ApplyBSSource is skipped to avoid double-counting.
    void ApplyKortewegStress(int lev,
        Set::Scalar time,
        amrex::MultiFab &M0_rhs_mf, amrex::MultiFab &E0_rhs_mf,
        amrex::MultiFab &M1_rhs_mf, amrex::MultiFab &E1_rhs_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &rho0_mf_in, const amrex::MultiFab &M0_mf_in,
        const amrex::MultiFab &rho1_mf_in, const amrex::MultiFab &M1_mf_in);


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
    // PHASE eta=1 (liquid — stiffened gas)
    Set::Field<Set::Scalar> density_eta1_mf;
    Set::Field<Set::Scalar> density_eta1_old_mf;

    Set::Field<Set::Scalar> energy_eta1_mf;
    Set::Field<Set::Scalar> energy_eta1_old_mf;

    Set::Field<Set::Scalar> momentum_eta1_mf;
    Set::Field<Set::Scalar> momentum_eta1_old_mf;

    //Set::Field<Set::Scalar> T0_mf;
    //Set::Field<Set::Scalar> k0_thermal_mf;
    //Set::Field<Set::Scalar> h0_thermal_mf;
    //Set::Field<Set::Scalar> gamma_eta1_mf;

    Set::Field<Set::Scalar> velocity_eta1_mf;
    Set::Field<Set::Scalar> pressure_eta1_mf;
    Set::Field<Set::Scalar> vorticity_eta1_mf;
    Set::Field<Set::Scalar> c_eta1_mf;

    // PHASE eta=0 (gas — CPG)
    Set::Field<Set::Scalar> density_eta0_mf;
    Set::Field<Set::Scalar> density_eta0_old_mf;

    Set::Field<Set::Scalar> energy_eta0_mf;
    Set::Field<Set::Scalar> energy_eta0_old_mf;

    Set::Field<Set::Scalar> momentum_eta0_mf;
    Set::Field<Set::Scalar> momentum_eta0_old_mf;

    //Set::Field<Set::Scalar> T1_mf;
    //Set::Field<Set::Scalar> k1_thermal_mf;
    //Set::Field<Set::Scalar> h1_thermal_mf;
    //Set::Field<Set::Scalar> gamma_eta0_mf;
    Set::Field<Set::Scalar> velocity_eta0_mf;
    Set::Field<Set::Scalar> pressure_eta0_mf;
    Set::Field<Set::Scalar> vorticity_eta0_mf;
    Set::Field<Set::Scalar> c_eta0_mf;

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
    // energy_eta{0,1}_mf are registered against these instead of the shared
    // energy_bc; the mixture energy_per_vol_mf stays on energy_bc.
    BC::BC<Set::Scalar>* energy_eta0_bc = nullptr;
    BC::BC<Set::Scalar>* energy_eta1_bc = nullptr;
    // Per-phase density BCs. Same pattern as the per-phase energy BCs above:
    // default to the shared density_bc; problems like Stefan that pin gas-phase
    // density at a heated wall (to keep T = UE/(ρ·cv) sensible under fixed-UE
    // Dirichlet) need to override only one phase without polluting the spectator.
    BC::BC<Set::Scalar>* density_eta0_bc = nullptr;
    BC::BC<Set::Scalar>* density_eta1_bc = nullptr;
    //BC::BC<Set::Scalar>* eta_bc = nullptr;
    BC::BC<Set::Scalar>* temperature_bc = nullptr;

    IC::IC<Set::Scalar>* velocity_eta1_ic = nullptr;
    IC::IC<Set::Scalar>* pressure_eta1_ic = nullptr;
    IC::IC<Set::Scalar>* density_eta1_ic  = nullptr;
    IC::IC<Set::Scalar>* energy_eta1_ic = nullptr;
    //IC::IC<Set::Scalar>* temperature0_ic  = nullptr;
    //IC::IC<Set::Scalar>* k0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h0_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma_eta1_ic  = nullptr;

    IC::IC<Set::Scalar>* velocity_eta0_ic = nullptr;
    IC::IC<Set::Scalar>* pressure_eta0_ic = nullptr;
    IC::IC<Set::Scalar>* density_eta0_ic = nullptr;
    IC::IC<Set::Scalar>* energy_eta0_ic = nullptr;
    //IC::IC<Set::Scalar>* temperature1_ic = nullptr;
    //IC::IC<Set::Scalar>* k1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* h1_thermal_ic  = nullptr;
    //IC::IC<Set::Scalar>* gamma_eta0_ic  = nullptr;

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
    // PHASE eta=1 (liquid — stiffened gas)
    Set::Scalar gamma_eta1 = NAN, pi_eta1 = NAN, mu_eta1 = NAN, mu_eta1_b = NAN, MW0 = NAN;
    Set::Scalar b0 = NAN, q0EOS = NAN;
    Set::Scalar cp_eta1 = NAN, cv_eta1 = NAN;
    // Free-energy calibration parameters for phase 1 (Tammann)
    // q_eta1: reference internal energy [J/kg]; qprime_eta1: entropy reference [J/(kg·K)];
    // rho_ref_eta1: reference density [kg/m^3] used in the ln(rho/rho_ref) term.
    Set::Scalar q_eta1 = 0.0, qprime_eta1 = 0.0, rho_ref_eta1 = 1.0;

    // PHASE eta=0 (gas — CPG)
    Set::Scalar gamma_eta0 = NAN, pi_eta0 = NAN, mu_eta0 = NAN, mu_eta0_b = NAN, MW1 = NAN;
    Set::Scalar b1 = NAN, q1EOS = NAN;
    Set::Scalar cp_eta0 = NAN, cv_eta0 = NAN;
    // Free-energy calibration parameters for phase 0 (CPG)
    // R_eta0: specific gas constant [J/(kg·K)] for the ideal-gas free energy.
    Set::Scalar q_eta0 = 0.0, qprime_eta0 = 0.0, rho_ref_eta0 = 1.0, R_eta0 = 287.0;

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
    int apply_surface_tension = 1;
    int apply_weight = 0;
    int apply_viscous = 0;     // Per-phase viscous stress in RHS_PerPhase (constant mu_k, mu_k_b)
    // Per-phase heat conduction (operator-split explicit-Euler step).
    // CFL constraint: dt < 0.5 * dx^2 / max(k_k / (rho_k cv_k)).
    int apply_heat_conduction = 0;
    Set::Scalar k_eta0 = 0.0;  // gas-phase thermal conductivity [W/(m K)]
    Set::Scalar k_eta1 = 0.0;  // liquid-phase thermal conductivity [W/(m K)]
    int static_eta = 0;
    int apply_vaporization = 0;
    int apply_bs_source = 1;   // Stage 4: BS phase-coupling sources (set 0 to test single-phase limit)
    // If non-zero, re-evaluate ic_m0/ic_u0/ic_q each Advance step at the current
    // simulation time. Used for prescribed-flux validation problems (e.g. Stefan)
    // where the analytical mass flux scales as ~1/sqrt(t).
    int bs_sources_time_dependent = 0;
    int Spec_Vol = 1;

    // Interface-pressure closure for the BS source (p_int).
    // 0: OFF — no pressure-imbalance coupling (mass/heat/ST terms still apply)
    // 1: arithmetic mean          p_int = 0.5 (p_0 + p_1)
    // 2: eta-weighted             p_int = (1-eta) p_0 + eta p_1
    // 3: acoustic-impedance       p_int = (Z_1 p_0 + Z_0 p_1)/(Z_0 + Z_1), Z_k = rho_k c_k
    // Same weighting is applied to u_int used in the energy source.
    int p_int_method = 0;

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

    // Phase 5 (CH-Korteweg spec §2.2, §4.3): well-balanced Korteweg surface
    // tension via face-flux form  F_cap_face = -eta_face * (mu_chem_j - mu_chem_i)/dx.
    // At static equilibrium mu_chem is uniform => discrete face flux is
    // exactly zero => no parasitic currents. The legacy Brackbill form
    // Fsv = sigma * kappa_curv * epsilon * grad_eta in ApplyBSSource is NOT
    // well-balanced and produces parasitic flow ~ sigma/(rho*dx).
    //
    // 0 (default): legacy Brackbill Fsv path in ApplyBSSource (preserves
    //              backward compat with all existing tests).
    // 1          : skip the Brackbill block in ApplyBSSource and add the
    //              face-flux Korteweg force via ApplyKortewegStress.
    int korteweg_well_balanced = 0;

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
