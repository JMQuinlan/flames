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

//EOS
#include "Solver/EOS/EOS.H"
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"

// Limiters
#include "Solver/Local/Limiter/Limiter.H"
#include "Solver/Local/Limiter/Godunov.H"
#include "Solver/Local/Limiter/Minmod.H"
#include "Solver/Local/Limiter/VanLeer.H"
#include "Solver/Local/Limiter/WENO3.H"
#include "Solver/Local/Limiter/WENO5.H"



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
        // EMBEDDED SOLID
        delete phi_ic;
        delete phi_bc;
        delete solid.density0_ic;
        delete solid.density1_ic;
        delete solid.momentum_ic;
        delete solid.energy0_ic;
        delete solid.energy1_ic;
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
        delete limiter;
        delete riemannsolver;


        
    }

protected:
    void Initialize(int lev) override;
    void Mix(int lev);
    void TimeStepBegin(Set::Scalar a_time, int a_iter) override;
    void TimeStepComplete(Set::Scalar time, int lev) override;
    void Advance(int lev, Set::Scalar time, Set::Scalar dt) override;
    void TagCellsForRefinement(int lev, amrex::TagBoxArray& tags, amrex::Real /*time*/, int /*ngrow*/) override;
    void Regrid(int lev, Set::Scalar time) override;
    // RHS signature carries the 6-equation primaries:
    //   eta              = alpha_1
    //   rho_eta{0,1}     = (alpha_k rho_k)  (canonical phase mass, NOT rho*eta)
    //   M                = mixture momentum
    //   E                = redundant total energy rho E (for shock fix-up reinit)
    //   E{0,1}           = (alpha_k rho_k e_k)  per-phase internal energy primaries
    // Schmidmayer-Bryngelson-Colonius JCP 2020, eqs (13) + (16).
    void RHS(int lev,
        Set::Scalar time,
        amrex::MultiFab &rho_eta0_rhs_mf,
        amrex::MultiFab &rho_eta1_rhs_mf,
        amrex::MultiFab &M_rhs_mf,
        amrex::MultiFab &E_rhs_mf,
        amrex::MultiFab &eta_rhs_mf,
        amrex::MultiFab &E0_rhs_mf,
        amrex::MultiFab &E1_rhs_mf,
        const amrex::MultiFab &rho_eta0_mf_in,
        const amrex::MultiFab &rho_eta1_mf_in,
        const amrex::MultiFab &M_mf,
        const amrex::MultiFab &E_mf,
        const amrex::MultiFab &eta_mf_in,
        const amrex::MultiFab &E0_mf_in,
        const amrex::MultiFab &E1_mf_in);

    // Post-stage pressure relaxation + reinitialization (Saurel-Petitpas-Berry
    // 2009 §3.5 / Schmidmayer 2020 §3.3).  Runs in-place on the working
    // MultiFabs after each RK stage and before the ghost-cell fill.
    void RelaxAndReinit(int lev);


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

    //Set::Field<Set::Scalar> eta_mf_in;
    Set::Field<Set::Scalar> etadot_mf;
    Set::Field<Set::Scalar> eta_old_mf;
    Set::Field<Set::Scalar> hess_eta_mf;
    Set::Field<Set::Scalar> n_hat_mf;

    // ------------------------------------------------------------------
    // EMBEDDED SOLID BOUNDARY (diffuse-interface immersed boundary)
    // ------------------------------------------------------------------
    // phi is the SOLID indicator: phi = 1 -> fluid, phi = 0 -> solid.
    // It is DISTINCT from eta, which in Hydro2 is the fluid-fluid volume
    // fraction alpha_1.  (In the single-phase Hydro solver the field named
    // eta plays the role that phi plays here.)  Static in this first
    // implementation: phi is set from its IC and never evolved.
    Set::Field<Set::Scalar> phi_mf;
    Set::Field<Set::Scalar> phi_old_mf;
    Set::Field<Set::Scalar> grad_phi_mf;
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

    // EMBEDDED SOLID: indicator IC/BC + prescribed per-phase target state.
    // The solid target is the quiescent state held inside solid cells and
    // used to build a reflecting fluid state at the diffuse boundary.
    IC::IC<Set::Scalar>* phi_ic = nullptr;
    BC::BC<Set::Scalar>* phi_bc = nullptr;
    struct {
        Set::Field<Set::Scalar> density0_mf;   // prescribed (alpha_1 rho_1) inside solid
        Set::Field<Set::Scalar> density1_mf;   // prescribed (alpha_2 rho_2) inside solid
        Set::Field<Set::Scalar> momentum_mf;   // prescribed mixture momentum inside solid (=0 for a static wall)
        Set::Field<Set::Scalar> energy0_mf;    // prescribed (alpha_1 rho_1 e_1) inside solid
        Set::Field<Set::Scalar> energy1_mf;    // prescribed (alpha_2 rho_2 e_2) inside solid
        IC::IC<Set::Scalar>* density0_ic = nullptr;
        IC::IC<Set::Scalar>* density1_ic = nullptr;
        IC::IC<Set::Scalar>* momentum_ic = nullptr;
        IC::IC<Set::Scalar>* energy0_ic  = nullptr;
        IC::IC<Set::Scalar>* energy1_ic  = nullptr;
    } solid;

    BC::Nothing bc_nothing;

    BC::Constant neumann_bc_1 = BC::Constant::ZeroNeumann(1);
    BC::Constant neumann_bc_D = BC::Constant::ZeroNeumann(AMREX_SPACEDIM);

    BC::NSCBC *nscbc_bc = nullptr;
    BC::NSCBC4 *nscbc4_bc = nullptr;


    Set::Scalar c_max = 0.0;
    Set::Scalar vx_max = 0.0;
    Set::Scalar vy_max = 0.0;
    Set::Scalar F_max = 0.0;
    Set::Scalar rho_min = 1e10;

    Set::Scalar eta_refinement_criterion=NAN, omega_refinement_criterion=NAN, gradu_refinement_criterion=NAN, p_refinement_criterion=NAN, rho_refinement_criterion=NAN;
    Set::Scalar phi_refinement_criterion=NAN; // grad(phi) solid-boundary refinement
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

    // EMBEDDED SOLID BOUNDARY controls.
    //   apply_embedded_solid = 0 -> no immersed solid (default; behavior unchanged)
    //   apply_embedded_solid = 1 -> mask fluxes by phi + penalize toward solid state
    int apply_embedded_solid = 0;
    // Below this phi value a cell is frozen to the prescribed solid target.
    Set::Scalar solid_cutoff = 1.0e-6;
    // Large solid viscosity blended in by (1-phi) to enforce no-SLIP at the
    // diffuse wall via the viscous stress (the Brinkman penalty already gives
    // no-penetration + a velocity sink; high mu sharpens the tangential
    // boundary layer, as required in the single-phase Hydro solver).
    // NOTE: this enters the viscous CFL, so a large value shrinks dt.
    Set::Scalar solid_mu   = 100.0;
    Set::Scalar solid_mu_b = 0.0;
    // Optional phi-interface NORMAL no-penetration penalty coefficient (the
    // solid analog of the fluid `lagrange`).  Default 0 (off).  Tangential
    // no-slip is enforced by the always-on viscous interface traction
    // (0.5*Mpqrs*(u-u_solid)*hess_phi, using solid_mu); this adds an extra
    // normal penalty  -lagrange_solid*(u-u_solid).grad_phi * grad_phi.
    Set::Scalar lagrange_solid = 0.0;

    // ------------------------------------------------------------------
    // Artificial heat exchange (AHE) - Schmidmayer 2020 eq. 13 r-source
    //     E0_rhs += -mu p_I (p_0 - p_1)
    //     E1_rhs += +mu p_I (p_0 - p_1)
    // with mu from the Saurel-Petitpas-Berry 2009 sec. 5.3 Fig. 21 curve
    // fit mu(v) = exp(a v^2 + b v + c).  HYBRID: Sch20 does NOT specify
    // a finite mu (they take the stiff limit) and the Sau09 curve fit
    // was originally calibrated for the q*@u/@x form, not the Sch20
    // antisymmetric form.  See bin/AHE.md for the rationale.
    //
    // Pairwise conservative (sources sum to zero), but per-phase energy
    // is only conserved pairwise - total mixture energy is maintained
    // via the redundant rho-E reinit.  NOT the most conservative method
    // in this codebase; see the comment block in Hydro2::RHS.
    // ------------------------------------------------------------------
    // ahe.method (preferred top-level switch; overrides apply_ahe/ahe.form):
    //   0 = Sch20 stiff-limit only        (no explicit AHE source; RelaxAndReinit Newton)
    //   1 = Sch20 finite-mu r-source      (+/- mu p_I Dp; antisymmetric)
    //   2 = Sau09 sec.5.3 q*@u/@x form    (q* = mu(v), gated on div(u)<0)
    int         ahe_method           = 0;
    int         apply_ahe            = 0;
    int         ahe_use_const_mu     = 0;         // 1 = use ahe_mu_const, skip curve fit
    Set::Scalar ahe_mu_const         = 0.0;       // constant mu when ahe_use_const_mu=1
    Set::Scalar ahe_mu_scale         = 1.0;       // multiplicative scale on curve fit
    Set::Scalar ahe_mu_a             = -5.64e7;   // Sau09 Fig. 21 epoxy-spinel (dim'l)
    Set::Scalar ahe_mu_b             =  5.34e3;
    Set::Scalar ahe_mu_c             = -25.6;
    Set::Scalar ahe_v_min            =  2.65e-4;  // [m^3/kg] valid range lower
    Set::Scalar ahe_v_max            =  4.61e-4;  // [m^3/kg] valid range upper
    int         ahe_compression_only = 0;         // 1 = gate on div(u)<0 (Sch20 form only; Sau09 form is gated by design)
    int         ahe_apply_alpha_src  = 1;         // 1 = also apply +mu Dp to eta_rhs (Sch20 row 1)
    Set::Scalar ahe_max_frac         = 0.1;       // |dE_AHE| < max_frac * E_min * (|u|+c)/dx
    int         ahe_form             = 1;         // 0 = Sch20 r-source (+/- mu p_I Dp);
                                                  // 1 = Sau09 sec.5.3 q*@u/@x form (default).

    int kappa_method = 2;

    // Adaptive Timestep
    bool adaptive_timestep = false;
    Set::Scalar dt_min = NAN, dt_max = NAN, dt_growth = NAN;

    // Limiters / reconstruction (Sch20 §3.2 primitive-variable
    // reconstruction).  Selected by name in input via:
    //     Limiter.type = godunov | minmod | vanleer | weno3 | weno5
    Solver::Local::Limiter::Limiter *limiter = nullptr;

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

    // Newton diagnostic for RelaxAndReinit.
    //   relax_diag = 0  : off (default; no diagnostic overhead)
    //   relax_diag = 1  : every stage print {max_iters, max_residual}
    //                     plus AUTO-warn if any cell hits max_iter without converging.
    int relax_diag = 0;

};
}

#endif
