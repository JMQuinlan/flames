// Base
#include "Hydro2.H"
// Parsing and Input Handeling
#include "AMReX_MultiFab.H"
#include "IO/ParmParse.H"
#include "BC/Constant.H"
#include "BC/Expression.H"
#include "BC/Nothing.H"
#include "BC/NSCBC.H"
#include "BC/NSCBC4.H"
#include "Numeric/Stencil.H"
#include "IC/Constant.H"
#include "IC/Laminate.H"
#include "IC/Expression.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLL.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/HLLC_LM.H"
#include "Solver/Local/FluidRiemann/HLLC_ADC.H"
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

#include <fstream>
#include <iomanip>
#include "AMReX_MultiFabUtil.H"   // makeFineMask (WriteIntegrals)


#include <AMReX_Math.H>
#include "AMReX_TimeIntegrator.H"


//#if AMREX_SPACEDIM == 2

namespace Integrator
{

// Spalding mass-transfer number  B_M = (Y_local - Y_inf) / (1 - Y_local).
// Single source of truth for both the Mix() and the RHS() pre-source loop --
// avoids the previous bug where the two computations had drifted to a wrong
// (1 + Y_inf) denominator. (Spalding 1953; Sirignano 2010 Eq. 2.18;
// Abramzon-Sirignano IJHMT 1989 Eq. 18.)
//
// `Y_local` here is whatever vapor mass-fraction the caller chose to use:
// the legacy closure passes the local cell mass fraction Y(i,j,k) (which at a
// liquid-dominated interface is ~rho_g/rho_l, nearly zeroing the rate -- the
// "evaporation does nothing" bug); the Stage-2 closure passes the Antoine
// SATURATION mass fraction Y_s from SaturationYs() below (gated on
// spalding_saturation), the physically meaningful driving force.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SpaldingBM(Set::Scalar Y_local, Set::Scalar Y_inf, Set::Scalar small)
{
    return (Y_local - Y_inf) / (1.0 - Y_local + small);
}

// Saturation vapor mass fraction at a liquid->vapor interface (Stage 2 Spalding
// closure). The interface temperature T sets the vapor's saturation PARTIAL
// pressure p_sat via the same Antoine curve (bar) used by the HRM cavitation
// block, so the two share one saturation line; Raoult/Dalton give the saturation
// MOLE fraction x_s = p_sat/p; the mole->mass conversion needs the
// molecular-weight ratio, taken from the specific gas constants R = cp - cv
// (W = R_univ/R, so the universal constant cancels):
//     Y_s = (x_s/R_v) / ( x_s/R_v + (1-x_s)/R_g )
// with R_v the vapor and R_g the inert-carrier specific gas constant. No new
// molecular-weight inputs are needed -- R_v = cp_vap-cv_vap and R_g comes from the
// carrier eos0. x_s is clamped to [0, 0.99] (sub-critical / numerically safe).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SaturationYs(Set::Scalar T, Set::Scalar p,
                                Set::Scalar A, Set::Scalar B, Set::Scalar C,
                                Set::Scalar R_v, Set::Scalar R_g, Set::Scalar small)
{
    const Set::Scalar p_sat = std::pow(10.0, A - B / (T + C)) * 1.0e5; // Antoine bar -> Pa
    Set::Scalar x_s = p_sat / (p + small);
    x_s = (x_s < 0.0) ? 0.0 : ((x_s > 0.99) ? 0.99 : x_s);            // saturation mole fraction
    const Set::Scalar nv = x_s / (R_v + small);
    const Set::Scalar ng = (1.0 - x_s) / (R_g + small);
    return nv / (nv + ng + small);
}

// NaN-safe clamp of a vapor mass fraction to [0, hi]. A plain ternary clamp
// (x<lo?lo:(x>hi?hi:x)) LEAKS NaN: a NaN fails both comparisons and passes
// straight through, so one corrupted rho_vap silently poisons every EOS call
// that reads Yv (gamma -> p0_eff -> press -> sound speed all go NaN, and the
// crash then mis-points at gamma instead of the real culprit rho_vap). Mapping
// NaN -- and any sub-zero input -- to 0, the physically safe "no vapor" value,
// localizes such corruption. Reordered so the default (else) branch is 0.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar ClampYv(Set::Scalar Yv, Set::Scalar hi = 1.0)
{
    return (Yv > 0.0) ? ((Yv < hi) ? Yv : hi) : 0.0;
}

// Stage 3 (3a-2): per-cell effective GAS equation of state for a binary
// ideal-gas mixture of an inert carrier (the eos0 phase, e.g. air) and dodecane
// vapor at vapor mass fraction Yv = mass_frac_v = rho_vap / alpharho0. Frozen
// caloric mixing: cv and cp are mass-weighted and gamma = cp/cv; both components
// are ideal gases so the stiffened reference pressure p0 = 0. Reduces EXACTLY to
// the carrier at Yv = 0 (and, if the carrier's cv/cp equal the vapor's, to the
// carrier for all Yv -- so a "carrier == vapor" input is a no-op regression).
// The result is passed in place of eos0 to the Mixed* EOS calls so the gas
// thermodynamics track composition; the liquid (eos1) and the eta-blending are
// unchanged.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Solver::EOS::Tammann GasEOS_eff(Set::Scalar Yv,
                                       const Solver::EOS::Tammann &carrier,
                                       Set::Scalar cv_vap, Set::Scalar cp_vap)
{
    Yv = ClampYv(Yv); // NaN-safe clamp to [0,1] (a raw ternary clamp leaks NaN)
    const Set::Scalar cv_g = Yv * cv_vap + (1.0 - Yv) * carrier.Cv();
    const Set::Scalar cp_g = Yv * cp_vap + (1.0 - Yv) * carrier.Cp();
    return Solver::EOS::Tammann(cp_g / cv_g, 0.0, cv_g, cp_g);
}

// Chapman-Enskog gas transport product rho*D(T) [kg/m/s]. CE binary diffusion
// gives D ~ T^1.5/p while rho ~ p/T, so the PRODUCT rho*D ~ T^0.5 and is
// pressure-independent (robust across shocks). Parameterized as
//   rho*D(T) = rhoD_ref * (T/T_ref)^q,   q = 0.5 (CE) by default.
// The single coefficient is shared by the bulk vapor diffusion (face value
// eta*rho*D, eta = gas volume fraction) and the Spalding film source (rho*D at
// the interface T), so the resolved transport and the sub-grid closure stay
// thermodynamically consistent. T is clamped non-negative for pow() safety.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar RhoD_CE(Set::Scalar T, Set::Scalar rhoD_ref,
                           Set::Scalar T_ref, Set::Scalar q, Set::Scalar small)
{
    const Set::Scalar Tr = T / (T_ref + small);
    return rhoD_ref * std::pow(Tr > 0.0 ? Tr : 0.0, q);
}

// ----------------------------------------------------------------------------
// Per-cell scaling factor for the Hu-Adams-Shu positivity-preserving flux
// limiter. Returns the largest theta in [0,1] such that the trial conserved
// state  U(theta) = B + theta * (s * dF)  stays in the admissible set
//   G = { rho >= eps_rho,  p >= eps_p }   (eps_p = floor on absolute pressure).
// B = (rho, Mx, My, E) is the source-inclusive low-order baseline for the cell,
// dF = (mass, Mx, My, E) is the high-minus-low face flux correction, and s folds
// in the convexity-split factor and signed lambda = ± (2*DIM) * dt/dx_dir.
//
// Density is a linear constraint; internal energy UE = E - |M|^2/(2 rho) is
// concave along the line in theta, so with h(0) >= 0 (baseline in G) a single
// bisection brackets the largest safe theta. The absolute-pressure floor
// p >= eps_p is equivalent to UE >= ue_floor =
// (eps_p + gamma_eff*p0_eff - pref)/(gamma_eff - 1) for frozen mixture
// gamma_eff, p0_eff.  (Perthame-Shu 1996; Zhang-Shu 2010; Hu-Adams-Shu 2013.)
//
// Optional per-phase guard (guard_phase): the per-phase flux corrections are
// NOT proportional to the mixture correction dF[0] -- each component flux is
// split by its OWN donor-side (Larrouturou) upwind mass fraction, so the caller
// passes dF_re0 = Y0h*Fh[0] - Y0l*Fl[0], the gas-partial per-unit-theta change
// at this face (the liquid one is dF[0] - dF_re0 since Y0+Y1=1 per flux). Both
// are linear in theta, so we just add two more upper-bound clamps on t to keep
// alpharho0, alpharho1 >= phase_floor (= 0, matching the post-update
// floor-at-0; partials are legitimately exactly 0 in pure phases, so any
// positive threshold here would force theta=0 across the whole single-phase
// field). re0_base/re1_base are the partial baselines (Bbase comps 4,5). The
// pressure (UE) constraint depends only on the mixture, so the bisection below
// is unchanged -- it just runs on the tightened [0,t_rho].
// ----------------------------------------------------------------------------
// Phase-dependent pressure floor lives in Solver::EOS::EOS::PressureFloor (so
// the EOS-facing MixedPressure/diagnostics share one definition). Gas
// (eta -> 1) is held at p >= eps_p; the stiffened liquid (eta -> 0) may tension
// down to -p_cav, with a hyperbolicity guard (p -> -p0_eff makes c -> 0).
// ----------------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar PPThetaCell(const Set::Scalar B[4], Set::Scalar s, const Set::Scalar dF[4],
                               Set::Scalar gamma_eff, Set::Scalar p0_eff,
                               Set::Scalar eps_rho, Set::Scalar eps_p, Set::Scalar pref,
                               bool guard_phase = false,
                               Set::Scalar re0_base = 0.0, Set::Scalar re1_base = 0.0,
                               Set::Scalar dF_re0 = 0.0, Set::Scalar phase_floor = 0.0)
{
    // Per-unit-theta change vector.
    const Set::Scalar V0 = s * dF[0]; // rho
    const Set::Scalar V1 = s * dF[1]; // Mx
    const Set::Scalar V2 = s * dF[2]; // My
    const Set::Scalar V3 = s * dF[3]; // E

    // Per-phase per-unit-theta change (donor-split partial correction; the
    // liquid one is the mixture remainder, exact by Y0+Y1=1 per flux).
    const Set::Scalar V_re0 = s * dF_re0;
    const Set::Scalar V_re1 = s * (dF[0] - dF_re0);

    const Set::Scalar r0 = B[0];
    const Set::Scalar ue_floor = (eps_p + gamma_eff * p0_eff - pref) / (gamma_eff - 1.0);

    // Internal energy along the line; valid only where density stays positive.
    auto UE = [&](Set::Scalar t) -> Set::Scalar {
        Set::Scalar r  = r0 + t * V0;
        Set::Scalar mx = B[1] + t * V1;
        Set::Scalar my = B[2] + t * V2;
        Set::Scalar e  = B[3] + t * V3;
        return e - 0.5 * (mx * mx + my * my) / r;
    };

    // Baseline must already be admissible; otherwise the limiter cannot help
    // (flux blending only adds the correction) and we fall back to pure HLL.
    if (r0 <= eps_rho || UE(0.0) < ue_floor) return 0.0;
    if (guard_phase && (re0_base < phase_floor || re1_base < phase_floor)) return 0.0;

    // Density: largest t keeping rho >= eps_rho.
    Set::Scalar t_rho = 1.0;
    if (V0 < 0.0) t_rho = (r0 - eps_rho) / (-V0);
    if (t_rho > 1.0) t_rho = 1.0;
    if (t_rho < 0.0) t_rho = 0.0;

    // Per-phase densities: tighten t so each partial stays >= phase_floor.
    if (guard_phase)
    {
        if (V_re0 < 0.0) { Set::Scalar t0 = (re0_base - phase_floor) / (-V_re0); if (t0 < t_rho) t_rho = t0; }
        if (V_re1 < 0.0) { Set::Scalar t1 = (re1_base - phase_floor) / (-V_re1); if (t1 < t_rho) t_rho = t1; }
        if (t_rho < 0.0) t_rho = 0.0;
    }

    // Internal energy: concave in t, h(0) >= 0. If the whole [0,t_rho] is safe
    // take t_rho, else bisect for the crossing.
    if (UE(t_rho) >= ue_floor) return t_rho;

    Set::Scalar lo = 0.0, hi = t_rho;
    for (int it = 0; it < 50; ++it)
    {
        Set::Scalar mid = 0.5 * (lo + hi);
        if ((r0 + mid * V0) > eps_rho && UE(mid) >= ue_floor) lo = mid;
        else hi = mid;
    }
    return lo;
}

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
        pp.query_default("eta_refinement_criterion", value.eta_refinement_criterion, 0.001);   // eta-based refinement (|grad eta|)
        pp.query_default("eta_band_refinement", value.eta_band_refinement, 1e-3);              // value-based band tag: refine where min(eta,1-eta) > this (covers the low-|grad eta| band tails the gradient tag misses)
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01); // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01); // velocity gradient-based refinement
        pp.query_default("divu_refinement_criterion", value.divu_refinement_criterion, 0.05);   // Ducros-weighted compression (shock) refinement
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);    // density-based refinement

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);                // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl);  // cfl condition
        pp_query_default("pref", value.pref, 0.0);          // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);       // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-6);   // eta cutoff value

        // Positivity-preserving flux limiter (Hu-Adams-Shu)
        pp_query_default("pp_flux_limiter", value.pp_flux_limiter, 1); // 1: on, 0: off
        pp_query_default("pp_source_limit", value.pp_source_limit, 1); // 1: fold source into PP baseline, 0: limit flux only
        pp_query_default("pp_source_limiter", value.pp_source_limiter, 0); // 1: scale source for positivity + per-phase flux guard (needs pp_flux_limiter); 0: off
        if (value.pp_source_limiter != 0 && value.pp_flux_limiter == 0)
            Util::Abort(INFO, "pp_source_limiter requires pp_flux_limiter=1 (it lives in the PP-limiter Pass B/C/D); got pp_flux_limiter=0");
        pp_query_default("eps_p", value.eps_p, 1.0);          // floor on absolute pressure p (gas side, eta -> 1)
        pp_query_default("eps_rho", value.eps_rho, 1.0E-10);  // density floor
        pp_query_default("p_cav", value.p_cav, 0.0);          // liquid cavitation floor: hold the liquid (eta -> 0) at p >= -p_cav (gas stays p >= eps_p). 0 = vacuum floor, no tension.
        if (value.p_cav < 0.0)
            Util::Abort(INFO, "p_cav is the magnitude of the liquid tensile/cavitation floor (liquid held at p >= -p_cav); it must be >= 0");

        // HLLC-ADC carbuncle cure: drive the omega blend from a Ducros-weighted
        // div(u) compression shock tag instead of the internal pressure-ratio
        // sensor (carbuncles are pressure-quiet). A cell is tagged when the
        // sensor exceeds adc_shock_threshold; the tag is then grown by adc_grow
        // cells so HLL dissipation covers the post-shock band, not just the
        // pressure jump. adc_grow < 0 disables the tag (use the pressure sensor).
        pp_query_default("adc_shock_threshold", value.adc_shock_threshold, 0.05);
        pp_query_default("adc_grow", value.adc_grow, 1);
        // Interface-compression tag: OR into the raw shock tag any diffuse-band cell
        // (min(eta,1-eta) > adc_interface_band) that is also compressing (div_u < 0).
        // Lands HLL dissipation on the shock-compressed interface band that the
        // Ducros sensor shear-suppresses (cures the windward-contact energy overshoot
        // that drives the liquid pressure/energy floor). 0 = off.
        pp_query_default("adc_interface_band", value.adc_interface_band, 0.0);
        // Force full HLL (omega = 0 on every face) for the first hll_first_steps
        // step(s), then fall back to the normal shock-tag blend. The interface/
        // contact carbuncle is seeded only by the step-1 startup transient (the
        // surface-tension ring), which the div(u) shock locator cannot tag; running
        // HLL there skips the seed without globally over-diffusing the run. 0 = off.
        // (omega=0 is genuine all-component HLL only when adc_components covers all 4.)
        pp_query_default("hll_first_steps", value.hll_first_steps, 0);
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
        // Spalding film thickness L [m]. The Spalding areal flux is m'' = (rho_g*Dv/L)*B'
        // -- L is the diffusion boundary-layer thickness, a PHYSICAL length distinct from
        // the band localizer |grad eta|. Prescribing a fixed L is the QUASI-STEADY
        // stagnant-film model (valid when delta^2/Dv << t_change): with L and B_M fixed
        // the rate is CONSTANT (linear mass loss), NOT the transient Stefan sqrt(t)
        // (which needs L -> delta(t) ~ sqrt(Dv*t)). Default 1.0 reproduces the legacy
        // (under-dimensioned) magnitude; set to a real film thickness for a correct rate.
        pp_query_default("L_film", value.L_film, 1.0);   // Spalding film/boundary-layer thickness [m]
        // Chapman-Enskog variable gas diffusivity. rhoD_ref > 0 enables the gas
        // transport product rho*D(T) = rhoD_ref*(T/rhoD_Tref)^rhoD_exp [kg/m/s] for
        // BOTH the bulk vapor diffusion and the Spalding source (p-independent, ~T^0.5);
        // rhoD_ref <= 0 keeps the legacy constant-Dv behavior.
        pp_query_default("rhoD_ref", value.rhoD_ref, 0.0);     // gas rho*D [kg/m/s]; >0 enables CE variable diffusivity
        pp_query_default("rhoD_Tref", value.rhoD_Tref, 300.0); // reference T [K] for rho*D(T)
        pp_query_default("rhoD_exp", value.rhoD_exp, 0.5);     // T exponent of rho*D (Chapman-Enskog: D~T^1.5/p, rho~p/T => rho*D~T^0.5)
        // Floor on rho0 = alpharho0/eta in the CE species-dt diagnostic only (caps Deff
        // = rho*D(T)/rho0 so a collapsed-alpharho0 interface cell can't crush dt_species
        // to ~1e-11; the vapor operator is already ClampYv-bounded). 0 = off (raw rho0).
        pp_query_default("dt_species_rho0_floor", value.dt_species_rho0_floor, 0.0); // [kg/m^3]; ~1e-2 for air-dodecane
        pp_query_default("k0_thermal", value.k0_thermal, 0.0); // gas thermal conductivity [W/m/K] (Fourier conduction; 0 = off)
        pp_query_default("k1_thermal", value.k1_thermal, 0.0); // liquid thermal conductivity [W/m/K] (Fourier conduction; 0 = off)
        // Verification: prescribed constant-rate phase change (bypasses Spalding). Off by default.
        pp_query_default("apply_vap_const", value.apply_vap_const, 0);
        pp_query_default("vap_const_mdot", value.vap_const_mdot, 0.0); // interfacial mass flux [kg/m^2/s]; m_dot = vap_const_mdot*|grad eta|
        // HRM bulk cavitation (liquid -> vapor when p < p_sat). Off by default.
        pp_query_default("apply_cavitation", value.apply_cavitation, 0);
        pp_query_default("tau_cav", value.tau_cav, 1.0e-4);            // cavitation relaxation time [s]
        pp_query_default("antoine_A", value.antoine_A, 4.10549);      // log10(p_sat[bar]) = A - B/(T[K]+C)
        pp_query_default("antoine_B", value.antoine_B, 1625.928);
        pp_query_default("antoine_C", value.antoine_C, -92.839);
        // Stage 2: physical Spalding driving force. 1 = build Bm from the Antoine
        // SATURATION mass fraction Y_s at the interface T (carrier+vapor model);
        // 0 = legacy local cell-Y driving force. See SaturationYs / SpaldingBM.
        pp_query_default("spalding_saturation", value.spalding_saturation, 0);
        // Stage 3d: Spalding sink (Y_inf) closure. 0 = constant far-field Y_infinity
        // (fixed driving -> constant, non-Stefan rate). 1 = sample the sink at a film
        // distance ell = film_eps_mult*epsilon into the gas along +n_hat, so the
        // driving falls as vapor accumulates near the interface (transport-limited,
        // sqrt(t) recession). Sampling the gas EDGE of the band (not the adjacent
        // cell) avoids the self-quench from the source's own vapor.
        pp_query_default("spalding_film_sink", value.spalding_film_sink, 0);
        pp_query_default("film_eps_mult", value.film_eps_mult, 3.0);
        pp_query_default("L_vap", value.L_vap, 256.0e3);              // latent heat of vaporization [J/kg] (Stage 3c energy coupling)
        pp_query_default("apply_latent_heat", value.apply_latent_heat, 0); // 1 = subtract L_vap*m_dot_Vap from mixture energy (evaporative cooling); 0 = off
        // Stage 3: inert-carrier gas + transported dodecane-vapor species.
        pp_query_default("species_transport", value.species_transport, 0); // 1 = gas is carrier(eos0)+vapor, track rho_vap
        pp_query_default("Yv_init", value.Yv_init, 0.0);              // initial vapor mass fraction in the gas region
        pp_query_default("cp_vap", value.cp_vap, 1994.85);           // dodecane-vapor cp [J/kg/K]
        pp_query_default("cv_vap", value.cv_vap, 1950.0);            // dodecane-vapor cv [J/kg/K]
        if (value.spalding_film_sink != 0 && value.species_transport == 0)
            Util::Abort(INFO, "spalding_film_sink=1 requires species_transport=1 (the sink samples rho_vap/alpharho0)");
        // rho_vap is advected in BOTH the FE (no-limiter) path and the PP-limiter
        // Pass D, in each case riding the SAME mixture mass flux as alpharho0 so the
        // carrier (= alpharho0 - rho_vap) is conserved face-by-face. Its coarse-fine
        // ghosts are now FillPatch'd from the coarse level (FillGhost4BC) and its
        // advective face flux is refluxed at coarse-fine boundaries (cc_fluxes mass
        // comp 2 -> FluxRegister), so the multigrid handling matches alpharho0. The
        // remaining approximation under the limiter is that rho_vap carries no
        // positivity theta of its own (it inherits the per-face blend chosen for
        // the mixture); the diffusive (Dv) flux is also not refluxed, matching the
        // conduction treatment.
        if (value.species_transport != 0 && value.pp_flux_limiter != 0)
            Util::Message(INFO, "WARNING: species_transport=1 with pp_flux_limiter=1: "
                          "vapor advection inherits the mixture per-face positivity "
                          "blend (no separate rho_vap theta); CF ghosts are FillPatch'd "
                          "and the advective flux is refluxed.");
        if (value.apply_cavitation != 0 && value.tau_cav <= 0.0)
            Util::Abort(INFO, "tau_cav must be > 0 when apply_cavitation=1 (it is the cavitation relaxation time)");
        pp_query_required("epsilon", value.epsilon);    // diffuse interface thickness Y_infinity
        pp_query_default("Y_infinity", value.Y_infinity, 0.0); // Far Field Vapor Mass Fraction
        pp_query_default("Mob", value.Mob_user, 0.0);   // CH mobility scale M0: M = M0 * epsilon^2
        pp_query_default("apply_ch_companion", value.apply_ch_companion, 1); // 1: CH companion mass/mom/energy fluxes on (keeps alpharho_k consistent with eta); 0: off
        pp_query_default("ch_companion_pp_limit", value.ch_companion_pp_limit, 1); // 1: Zalesak outflow limiter on the companion (positivity -> M_floor_gas=0); 0: unlimited companion (old behavior, bit-identical)
        pp_query_default("abgrall_freeze_eos", value.abgrall_freeze_eos, 1); // 1: Abgrall-Karni double-flux (DEFAULT, production) -- freeze mixture EOS params (gamma_m,p0_eff) at start-of-step eta^n in the per-stage primitive recovery, curing the stiffened-mixture interface pressure-equilibrium instability; 0: off (legacy pre-cure path, A/B only). See ../Shock-droplet_5-Equation.tex sec:eos-instability.
        if (value.epsilon <= 0.0)
        {
            Util::Abort(INFO, "epsilon must be positive for Hydro2 Cahn-Hilliard mobility; got ", value.epsilon);
        }

        // CURVATURE
        pp_query_default("kappa_method", value.kappa_method, 1); // 1: Smooth Normals (default)  2: legacy Hessian-based
        pp_query_default("smooth_kernel_size", value.smooth_kernel_size, 3); // Gaussian normal-smoothing kernel: 3 (3x3) or 5 (5x5)
        pp_query_default("kappa_grad_frac", value.kappa_grad_frac, 0.02); // |grad eta| floor for curvature, as a fraction of the equilibrium peak 1/(2 sqrt2 eps); below this kappa is forced to 0. Lower keeps stretched/thinned interface (necks); ~0.28 was the old 0.1/dx_eff default.

        pp_query_default("write_integrals", value.write_integrals, 1); // 1: write volume-integrated conserved quantities to <plot_file>/integrals.dat each step

        // ===========================================================
        // 7-EQUATION (Baer-Nunziato) SCAFFOLDING SWITCHES
        //   equation_count = 5  -> existing Allaire-style model (default)
        //   equation_count = 7  -> per-phase mass/momentum/energy + alpha
        //
        // Phase A: parsing wired, MultiFabs registered, Riemann/NSCBC stubs
        //          available, but the RHS, Mix, and BC paths still operate
        //          on the 5-eqn state. Selecting equation_count=7 will
        //          abort with a clear "not yet implemented" message in the
        //          first integrator stage that touches the 7-eqn path.
        // ===========================================================
        pp_query_default("equation_count",       value.equation_count,       5);
        pp_query_default("relax_pressure_stiff", value.relax_pressure_stiff, 1);
        pp_query_default("relax_velocity_stiff", value.relax_velocity_stiff, 1);
        pp_query_default("surface_tension_form", value.surface_tension_form, 0);
        if (value.equation_count != 5 && value.equation_count != 7)
        {
            Util::Abort(INFO, "equation_count must be 5 or 7; got ", value.equation_count);
        }

        // INTERFACE COMPRESSION
        pp_query_default("apply_sharpening", value.apply_sharpening, false);
        pp_query_default("sharpening_frequency", value.sharpening_frequency, 10);
        pp_query_default("reinit_max_iter", value.reinit_max_iter, 10);
        pp_query_default("reinit_tolerance", value.reinit_tolerance, 1e-6);
        pp_query_default("density_max_iter", value.density_max_iter, value.reinit_max_iter); // Density correction iterations (papers use 5-10 iterations)
        pp_query_default("density_tolerance", value.density_tol, value.reinit_tolerance);    // Density correction tolerance
        pp_query_default("density_relax", value.omega_relax, 0.5);                  // Relaxation parameter (0.3-0.7 typical)

    
        // Boundry Conditions
        pp_query_default("nghost", value.nghost, 2); // Number of Ghost Cells

        // The curvature pipeline fills the derived fields (eta_x, smoothed
        // normals) into the ghost layer so kappa is coarse-fine/physical-
        // boundary correct. That needs nghost >= krad+2 (3 for a 3x3 smoothing
        // kernel, 4 for 5x5). Below that, patch-edge curvature ghosts fall back
        // to same-level FillBoundary (today's behavior) and can be stale.
        {
            const int krad_req = (value.smooth_kernel_size >= 5) ? 2 : 1;
            if (value.nghost < krad_req + 2)
                Util::Message(INFO, "WARNING: nghost=", value.nghost,
                              " < krad+2=", krad_req + 2,
                              " for smooth_kernel_size=", value.smooth_kernel_size,
                              "; curvature at coarse-fine/physical boundaries falls back to FillBoundary (same-level only).");
        }

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
            if (value.nghost == 2)
            {
                value.nscbc_bc = new BC::NSCBC(pp);
                value.nscbc4_bc = nullptr;
                Util::Message(INFO, "Parsing NSCBC (2-cell)");
            }
            else if (value.nghost == 4)
            {
                value.nscbc_bc = nullptr;
                value.nscbc4_bc = new BC::NSCBC4(pp);
                Util::Message(INFO, "Parsing NSCBC4 (4-cell)");
            }
            else
            {
                Util::Abort(INFO, "NSCBC requires nghost = 2 or 4");
            }

            // Use BC::Nothing for standard BC pointers
            value.density_bc = &value.bc_nothing;
            value.energy_bc = &value.bc_nothing;
            value.momentum_bc = &value.bc_nothing;

            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
            Util::Message(INFO, "nscbc4_bc Pointer=", value.nscbc4_bc);
        }
        else
        {
            // Standard mode: Use Expression BCs
            value.nscbc_bc = nullptr;

            value.density_bc = new BC::Expression(1, pp, "density.bc");
            value.energy_bc = new BC::Expression(1, pp, "energy.bc");
            value.momentum_bc = new BC::Expression(2, pp, "momentum.bc");

            Util::Message(INFO, "Parsing Reg");
            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
        }

        // Eta BC: parse from "eta.bc" if user provides it, otherwise zero neumann.
        // Eta is a volume fraction transported by advection + Cahn-Hilliard;
        // zero-neumann (ghost = interior) is the physically correct default.
        if (pp.contains("eta.bc.type.xlo") || pp.contains("eta.bc.type.ylo"))
        {
            value.eta_bc = new BC::Expression(1, pp, "eta.bc");
        }
        else
        {
            value.eta_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
        }
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,           value.eta_bc, 1, nghost, "eta", true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.eta_bc, 1, nghost, "eta_old", false, true);
        value.RegisterNewFab(value.alpharho0_mf,      value.density_bc, 1, nghost, "alpharho0", true, true);
        value.RegisterNewFab(value.alpharho1_mf,      value.density_bc, 1, nghost, "alpharho1", true, true);
        value.RegisterNewFab(value.alpharho0_old_mf,  value.density_bc, 1, nghost, "alpharho0_old", false, true);
        value.RegisterNewFab(value.alpharho1_old_mf,  value.density_bc, 1, nghost, "alpharho1_old", false, true);
        // Stage 3: dodecane-vapor partial density within the gas (carrier+vapor).
        // Zero-gradient (neumann) BC, NOT density_bc: the gas inflow is pure
        // carrier (no vapor), so the total-density dirichlet value density_bc
        // carries would be wrong as a rho_vap ghost. Zero-gradient also matches
        // the species advection/diffusion stencil, which clamps to the domain
        // edge (= no-flux) at physical boundaries.
        value.RegisterNewFab(value.rho_vap_mf,       &value.neumann_bc_1, 1, nghost, "rho_vap", true, true);
        value.RegisterNewFab(value.rho_vap_old_mf,   &value.neumann_bc_1, 1, nghost, "rho_vap_old", false, true);

        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing, 1, 0, "etadot", true, false);
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing, 4, 0, "hess_eta", false, false, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing, 2, 0, "n_hat", false, false, { "x", "y" });

        // FLUID 0
        // density0_mf / density1_mf are the LIVE per-phase intrinsic densities rho_0, rho_1
        // (= alpharho_k / max(alpha_k, small)), recomputed each step in Advance and plotted.
        // They replace the retired rho_eta partition as the per-phase mass representation.
        value.RegisterNewFab(value.density0_mf,     value.density_bc,   1, nghost, "density0",     true, false );
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
        value.RegisterNewFab(value.density1_mf,     value.density_bc,   1, nghost, "density1", true, false);
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
        value.RegisterNewFab(value.density_mf,      value.density_bc,   1, nghost, "density", true, false);
        value.RegisterNewFab(value.density_old_mf,  value.density_bc,   1, nghost, "density_old", false, false);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost, "energy_per_vol", true, true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost, "energy_per_mass", true, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost, "energy_vol_old", false, true);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost, "energy_mas_old", false, true);
        value.RegisterNewFab(value.momentum_mf,     value.momentum_bc,  2, nghost, "momentum", true, true, { "x", "y" });
        value.RegisterNewFab(value.momentum_old_mf, value.momentum_bc,  2, nghost, "momentum_old", false, true, { "x", "y" });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,           &value.bc_nothing, 1, 0, "m0", false, false);
        value.RegisterNewFab(value.u0_mf,           &value.bc_nothing, 2, 0, "u0", false, false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,            &value.bc_nothing, 2, 0, "q0", false, false, { "x", "y" });
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
        value.RegisterNewFab(value.mu_chem_mf,      value.eta_bc, 1, nghost, "mu_chem", true, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,            &value.bc_nothing,  1, nghost, "a", true, false);                    // Speed of sound
        value.RegisterNewFab(value.Ma_mf,           &value.bc_nothing,  2, nghost, "Ma", true, false, { "x", "y" });   // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   value.energy_bc,  1, nghost, "UE_per_vol", true, false);         // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   value.energy_bc,  1, nghost, "UE_per_mass", true, false);        // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   value.energy_bc,  1, nghost, "KE_per_vol", true, false);         // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   value.energy_bc,  1, nghost, "KE_per_mass", true, false);        // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,           &value.bc_nothing,  1, nghost, "Spadling_Number", true, false);    // Spalding Number
        value.RegisterNewFab(value.Y_mf,            &value.bc_nothing,  1, nghost, "Mass_Fraction", true, false);       // Mass Fraction
        value.RegisterNewFab(value.mass_frac_v_mf,  &value.bc_nothing,  1, nghost, "Mass_Fraction_Vapor", true, false); // vapor mass fraction of the gas (rho_vap/alpharho0)

        // Curvature pipeline fields
        value.RegisterNewFab(value.eta_x_mf,            &value.bc_nothing,  1, nghost, "eta_x", false, false);
        value.RegisterNewFab(value.eta_y_mf,            &value.bc_nothing,  1, nghost, "eta_y", false, false);
        value.RegisterNewFab(value.gradmag_mf,          &value.bc_nothing,  1, nghost, "gradmag", false, false);
        value.RegisterNewFab(value.nx_smoothed_mf,      &value.bc_nothing,  1, nghost, "nx_smoothed", false, false);
        value.RegisterNewFab(value.ny_smoothed_mf,      &value.bc_nothing,  1, nghost, "ny_smoothed", false, false);
        value.RegisterNewFab(value.kappa_SF_mf,         &value.bc_nothing,  1, nghost, "kappa_SF", true, false);

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,     &value.bc_nothing,  2, nghost, "grad_eta", true, false, { "x", "y" });
        value.RegisterNewFab(value.kappas_mf,       &value.bc_nothing,  3, nghost, "kappa", true, false, { "Active", "SN", "Legacy" });
        value.RegisterNewFab(value.grad_mag_grad_eta_mf, &value.bc_nothing, 2, nghost, "grad_mag_grad_eta", false, false, { "x", "y" }); // grad( | grad(eta) | )
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
        value.RegisterNewFab(value.omega_adc_mf, &value.bc_nothing, 2, 0, "omega_adc", true, false, { "_xhi", "_yhi" }); // HLLC-ADC shock locator (per hi-face omega)
        value.RegisterNewFab(value.flux_dev_mf, &value.bc_nothing, 2, 0, "flux_dev", true, false, { "_xhi", "_yhi" }); // |F_used - F_HLL| rel. dev. per hi-face
        value.RegisterNewFab(value.shock_sensor_mf, &value.bc_nothing, 1, 0, "shock_sensor", true, false); // raw Ducros-weighted compression sensor (compared to adc_shock_threshold)
        value.RegisterNewFab(value.pp_theta_mf,  &value.bc_nothing, 2, 0, "pp_theta", true, false, { "_xhi", "_yhi" }); // PP limiter blend factor per hi-face (1=HLLC, 0=HLL)
        value.RegisterNewFab(value.pp_dmass_mf,  &value.bc_nothing, 1, 0, "pp_dmass", true, false); // antidiffusive mass flux removed by PP limiter at hi-faces (interface desync source)
        value.RegisterNewFab(value.pp_efloor_mf, &value.bc_nothing, 1, 0, "pp_efloor", true, false); // internal energy injected by the eps_p positivity backstop (pressure flooring)

        // ====================================================================
        // 7-EQUATION (Baer-Nunziato) PER-PHASE CONSERVATIVES
        //
        // Registered only when equation_count == 7 so the 5-eqn build is
        // bit-identical to the pre-conversion behavior. Components match
        // the layout declared in Hydro2.H.
        // ====================================================================
        if (value.equation_count == 7)
        {
            // Phase 0 (liquid)
            value.RegisterNewFab(value.alpha_rho_0_mf,     value.density_bc,  1, nghost, "alpha_rho_0",      true,  true);
            value.RegisterNewFab(value.alpha_rho_0_old_mf, value.density_bc,  1, nghost, "alpha_rho_0_old",  false, true);
            value.RegisterNewFab(value.alpha_rho_M0_mf,    value.momentum_bc, 2, nghost, "alpha_rho_M0",     true,  true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_M0_old_mf,value.momentum_bc, 2, nghost, "alpha_rho_M0_old", false, true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_E0_mf,    value.energy_bc,   1, nghost, "alpha_rho_E0",     true,  true);
            value.RegisterNewFab(value.alpha_rho_E0_old_mf,value.energy_bc,   1, nghost, "alpha_rho_E0_old", false, true);

            // Phase 1 (gas)
            value.RegisterNewFab(value.alpha_rho_1_mf,     value.density_bc,  1, nghost, "alpha_rho_1",      true,  true);
            value.RegisterNewFab(value.alpha_rho_1_old_mf, value.density_bc,  1, nghost, "alpha_rho_1_old",  false, true);
            value.RegisterNewFab(value.alpha_rho_M1_mf,    value.momentum_bc, 2, nghost, "alpha_rho_M1",     true,  true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_M1_old_mf,value.momentum_bc, 2, nghost, "alpha_rho_M1_old", false, true, { "x", "y" });
            value.RegisterNewFab(value.alpha_rho_E1_mf,    value.energy_bc,   1, nghost, "alpha_rho_E1",     true,  true);
            value.RegisterNewFab(value.alpha_rho_E1_old_mf,value.energy_bc,   1, nghost, "alpha_rho_E1_old", false, true);

            // Volume fraction (advected, non-conservative). Distinct field from
            // eta_mf so the existing 5-eqn diagnostics keep working unchanged.
            value.RegisterNewFab(value.alpha0_mf,     value.energy_bc, 1, nghost, "alpha0",     true,  true);
            value.RegisterNewFab(value.alpha0_old_mf, value.energy_bc, 1, nghost, "alpha0_old", false, true);
        }
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
                      Solver::Local::FluidRiemann::HLL,
                      Solver::Local::FluidRiemann::HLLE,
                      Solver::Local::FluidRiemann::HLLC,
                      Solver::Local::FluidRiemann::HLLCE,
                      //Solver::Local::FluidRiemann::HLLCE_WENO5, // Never verified but updated
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman, // Can't remember if this has been verified
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro,
                      Solver::Local::FluidRiemann::HLLC_LM,
                      Solver::Local::FluidRiemann::HLLC_ADC
                      //Solver::Local::FluidRiemann::Upwind,
                      //Solver::Local::FluidRiemann::Lax_Friedrich
    >("Riemann_Solver", value.riemannsolver);
    Util::Message(INFO, "Selected Riemann solver: ", typeid(*value.riemannsolver).name());

    // Low-order, positivity-preserving flux for the PP flux limiter. Always an
    // HLL solver (Davis wave speeds), independent of the selected high-order
    // solver; only consumed when pp_flux_limiter != 0.
    value.hll_solver = new Solver::Local::FluidRiemann::HLL();

    // Mirror HLLC-ADC's exponent so the omega_adc_mf diagnostic uses the same
    // sensor sharpness as the active solver. Defaults to 3.0 if not set.
    {
        // HLLC-ADC reads its parameters under the fully-qualified prefix
        // "Riemann_Solver.hllc_adc" (select_default appends ".<type>"); mirror
        // that here so the omega_adc diagnostic uses the solver's actual alpha.
        IO::ParmParse pp_rs("Riemann_Solver.hllc_adc");
        pp_rs.query_default("alpha", value.omega_adc_alpha, 3.0);
    }


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

    // Initialize step counter (shared between 5-eqn and 7-eqn paths)
    if (step_counter.size() <= (size_t)lev)
    {
        step_counter.resize(lev + 1, 0);
    }
    step_counter[lev] = 0;

    // Ensure flux register and face-flux vectors are large enough.
    if ((int)flux_reg.size() <= lev)
        flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev)
        cc_fluxes.resize(lev + 1);
    if ((int)pp_scratch.size() <= lev)
        pp_scratch.resize(lev + 1);

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
    alpharho0_mf[lev]->setVal(0.0);
    alpharho1_mf[lev]->setVal(0.0);
    alpharho0_old_mf[lev]->setVal(0.0);
    alpharho1_old_mf[lev]->setVal(0.0);
    rho_vap_mf[lev]->setVal(0.0);
    rho_vap_old_mf[lev]->setVal(0.0);

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

    // Zero Common Fields
    ZeroDerivedScratchFields(lev);

    // Initialize Riemander
    a_mf[lev]           ->setVal(0.0);
    Ma_mf[lev]          ->setVal(0.0);
    UE_per_vol_mf[lev]  ->setVal(0.0);
    UE_per_mas_mf[lev]  ->setVal(0.0);
    KE_per_vol_mf[lev]  ->setVal(0.0);
    KE_per_mas_mf[lev]  ->setVal(0.0);

    // Use BoxArray/dmap from a registered MultiFab (density_mf) because
    // grids[lev]/dmap[lev] may not yet be set when Initialize runs.
    const amrex::BoxArray& ba_init = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm_init = density_mf[lev]->DistributionMap();

    // Build FluxRegister for this level if it is a fine level.
    if (lev > 0)
    {
        int ncomp_reflux = 2 + AMREX_SPACEDIM + 1 + 1; // rho0 + rho1 + momentum + energy + rho_vap
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba_init, dm_init, refRatio(lev - 1), lev, ncomp_reflux);
    }

    // Cell-centered flux MultiFabs: cell (i,j) stores the hi-face Riemann flux.
    // 1 ghost cell so FillBoundary can propagate fluxes across box boundaries
    // for the cell-to-face conversion in Advance().
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 3, 1); // alpharho0, alpharho1, rho_vap
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba_init, dm_init, AMREX_SPACEDIM, 1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 1, 1);

        cc_fluxes[lev].mass[d]->setVal(0.0);
        cc_fluxes[lev].mom[d]->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
    }

    // PP flux-limiter scratch (see LimiterScratch). 1 ghost like cc_fluxes.
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        pp_scratch[lev].Fhi[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 5, 1);
        pp_scratch[lev].Flo[d] = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 5, 1);
        pp_scratch[lev].Fhi[d]->setVal(0.0);
        pp_scratch[lev].Flo[d]->setVal(0.0);
    }
    pp_scratch[lev].Bbase = std::make_unique<amrex::MultiFab>(ba_init, dm_init, 6, 1);
    pp_scratch[lev].theta = std::make_unique<amrex::MultiFab>(ba_init, dm_init, AMREX_SPACEDIM, 1);
    pp_scratch[lev].Bbase->setVal(0.0);
    pp_scratch[lev].theta->setVal(0.0);

    Util::ParallelMessage(INFO, "Finished initialization, begginning time iteration");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Stage 3d film-sink resolution check: the Spalding sink is sampled
    // film_eps_mult*epsilon into the gas, clamped to the ghost depth nghost. If
    // ell/dx > nghost the sample is clamped back inside the diffuse band and the
    // closure reverts to the self-quenching local-cell behavior. Warn once.
    if (spalding_film_sink == 1)
    {
        const Set::Scalar ncells = film_eps_mult * epsilon / std::min(DX[0], DX[1]);
        if (ncells > (Set::Scalar)nghost + 1.0e-6)
            Util::Message(INFO, "WARNING: spalding_film_sink film distance is ",
                          ncells, " cells but nghost = ", nghost,
                          "; the sink sample is clamped inside the band -> rate will"
                          " self-quench. Increase nghost or dx (need dx >= "
                          "film_eps_mult*epsilon/nghost).");
    }

    // Function is for the diffusive mixing terms. I.E: rho = eta*rho0 + (1-eta)*rho1
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpharho0 = alpharho0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpharho1 = alpharho1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpharho0_old = alpharho0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> alpharho1_old = alpharho1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_vap = rho_vap_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_vap_old = rho_vap_old_mf.Patch(lev, mfi);

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
        Set::Patch<Set::Scalar>         mass_frac_v = mass_frac_v_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar>         a           = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Ma          = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_vol      = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_mas      = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_vol      = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_mas      = KE_per_mas_mf.Patch(lev, mfi);

        // Local EOS Copy (eos0_base = inert carrier; eos0_local is built per-cell
        // below as carrier + dodecane vapor when species_transport is on).
        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            // Derivative Function Calls
            Set::Scalar lap_eta = Numeric::Laplacian(eta, i, j, k, 0, DX);

            // =====================================================================
            // STATE CONVENTION (hydro2-5eq, post rho_eta retirement):
            //   eta       = alpha_0 = volume fraction of phase 0 (gas); alpha_1 = 1 - eta
            //   density0  = rho_0  (intrinsic density of phase 0),  density1 = rho_1
            //   alpharho0 = alpha_0 * rho_0 = eta * rho_0   (GAS    partial density)
            //   alpharho1 = alpha_1 * rho_1 = (1-eta) * rho_1 (LIQUID partial density)
            //   => alpharho0 + alpharho1 = rho (mixture density),  Y_gas = alpharho0/rho
            // alpharho0/1 are the TIME-EVOLVED conserved mass DOF (true SQR/Allaire partial
            // densities, NOT the old rho*eta partitions). rho is reconstructed as their SUM in
            // Advance (rho = alpharho0 + alpharho1). eta is evolved separately as the volume
            // fraction (advection + CH + phase change). The per-phase intrinsic densities are
            // RECOVERED for output as rho_k = alpharho_k / max(alpha_k, small) (done in Advance,
            // written to density0_mf/density1_mf). The mixture hydro/EOS reads only (rho, eta).
            //
            // Mass conservation: the mass flux is split by the TRUE upwind mass fraction
            // Y_k = alpharho_k/rho (NOT eta), so ff_0+ff_1 = F_mass telescopes; the phase-change
            // source is +/- m_dot (telescopes); nothing is ever re-projected to rho*eta. Hence
            // d/dt INT(alpharho0+alpharho1) = -boundary flux, exactly (= 0 for sealed walls).
            //
            // density0.ic/density1.ic set the initial intrinsic rho_0/rho_1; the local name
            // `rho0` aliases density0_mf here (the intrinsic density), used to build alpharho0.
            // =====================================================================
            // Calculate State Variables
            rho(i, j, k) = eta(i, j, k) * rho0(i, j, k) + (1.0 - eta(i, j, k)) * rho1(i, j, k);
            //rho(i, j, k) = 1.0 / (eta(i, j, k) / (rho0(i, j, k)) + (1.0 - eta(i, j, k)) / (rho1(i, j, k)));
            rho_old(i, j, k) = rho(i, j, k);

            // True partial densities alpha_k * rho_k (rho0 = density0 = intrinsic rho_0).
            alpharho0(i, j, k) = eta(i, j, k) * rho0(i, j, k);
            // Stage 3: vapor mass = Y_v * gas mass. Yv_init applies in the gas
            // (eta=1); alpharho0=0 in the liquid so rho_vap=0 there automatically.
            rho_vap(i, j, k) = Yv_init * alpharho0(i, j, k);
            rho_vap_old(i, j, k) = rho_vap(i, j, k);
            alpharho1(i, j, k) = (1.0 - eta(i, j, k)) * rho1(i, j, k);

            alpharho0_old(i, j, k) = alpharho0(i, j, k);
            alpharho1_old(i, j, k) = alpharho1(i, j, k);

            // Stage 3 (3a-2): composition-dependent gas EOS -- blend the carrier
            // (eos0_base) with dodecane vapor at the local vapor mass fraction
            // Yv = rho_vap/alpharho0. Off (== carrier) when species_transport=0.
            const Solver::EOS::Tammann eos0_local = species_transport
                ? GasEOS_eff(rho_vap(i, j, k) / std::max(alpharho0(i, j, k), small), eos0_base, cv_vap, cp_vap)
                : eos0_base;

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
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = alpharho0(i, j, k) / (rho(i, j, k));
            // Vapor mass fraction of the gas (Stage 3): rho_vap / alpharho0. ~0 in
            // the liquid (both ~0); = Y_v in the gas.
            // Cap Yv <= 0.99: rho_vap can transiently overshoot alpharho0 at isolated
            // points (advection/diffusion transients), giving an unphysical Yv > 1.
            mass_frac_v(i, j, k) = ClampYv(rho_vap(i, j, k) / std::max(alpharho0(i, j, k), small), 0.99);

            // Temperature (computed before Bm: the Stage-2 saturation driving force needs it)
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y)).
            // Stage 2: spalding_saturation -> physical Antoine saturation mass fraction
            // Y_s at the interface T as the driving force (carrier R from the base gas
            // EOS, vapor R = cp_vap-cv_vap; see SaturationYs) instead of the local cell Y.
            Set::Scalar Y_drive = Y(i, j, k);
            if (spalding_saturation == 1)
                Y_drive = SaturationYs(T(i, j, k), press(i, j, k), antoine_A, antoine_B, antoine_C,
                                       cp_vap - cv_vap, eos0_base.Cp() - eos0_base.Cv(), small);
            Bm(i, j, k) = SpaldingBM(Y_drive, Y_infinity, small);

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
                { "alpharho0", alpharho0(i, j, k) }, 
                { "alpharho1", alpharho1(i, j, k) }, 
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

    if (write_integrals) WriteIntegrals(time);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// WriteIntegrals ///////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Volume-integrated conserved quantities, appended to <plot_file>/integrals.dat once per coarse
// timestep (called from TimeStepComplete). This is a 5-equation model: the conserved state is two
// partial densities (alpharho0 = phase 0 / gas, alpharho1 = phase 1 / liquid; total rho = alpharho0 +
// alpharho1) plus a single mixture momentum and a single mixture total energy. Momentum and energy are
// therefore reported as totals only -- they are not separately conserved per phase here -- while mass
// is split into both phases (mass transfer mdot_0 + mdot_1 = 0 should keep M_total flat as the phase
// masses trade off). Adapted from the two-full-states branch, which split every quantity per phase.
void Hydro2::WriteIntegrals(Set::Scalar time)
{
    BL_PROFILE("Hydro2::WriteIntegrals");

    Set::Scalar M_total = 0.0, M_phase0 = 0.0, M_phase1 = 0.0;
    Set::Scalar M_vapor = 0.0;   // dodecane vapor mass in the gas (Stage 3); carrier = M_phase0 - M_vapor
    Set::Scalar Px_total = 0.0, Py_total = 0.0;
    Set::Scalar E_total = 0.0, KE_total = 0.0;
    Set::Scalar Mdot_total = 0.0; // volume-integrated liquid->vapor rate INT m_dot_Vap dV [kg/s] (Vap_dot[1])

    for (int lev = 0; lev <= finest_level; ++lev)
    {
        const amrex::Geometry& geom = this->geom[lev];
        const amrex::Real* DX = geom.CellSize();
        const amrex::Real  dV = DX[0] * DX[1];

        // Mask out cells covered by a finer level so they are not double-counted.
        amrex::iMultiFab fine_mask;
        if (lev < finest_level)
        {
            fine_mask = amrex::makeFineMask(
                eta_mf[lev]->boxArray(),
                eta_mf[lev]->DistributionMap(),
                eta_mf[lev + 1]->boxArray(),
                ref_ratio[lev],
                1, 0);   // 1 = uncovered (count), 0 = covered (skip)
        }

        for (amrex::MFIter mfi(*eta_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();

            auto re0_arr = alpharho0_mf[lev]->const_array(mfi);
            auto re1_arr = alpharho1_mf[lev]->const_array(mfi);
            auto rvap_arr = rho_vap_mf[lev]->const_array(mfi);
            auto M_arr   = momentum_mf[lev]->const_array(mfi);
            auto E_arr   = energy_per_vol_mf[lev]->const_array(mfi);
            auto vap_arr = Vap_dot_mf[lev]->const_array(mfi);   // [1] = m_dot_Vap [kg/m^3/s]

            const auto mask_arr = (lev < finest_level)
                ? fine_mask.const_array(mfi) : amrex::Array4<const int>{};

            amrex::LoopOnCpu(bx, [&](int i, int j, int k)
            {
                if (lev < finest_level && mask_arr(i, j, k) == 0) return;

                Set::Scalar re0 = re0_arr(i, j, k);
                Set::Scalar re1 = re1_arr(i, j, k);
                Set::Scalar rho = re0 + re1;

                Set::Scalar Mx = M_arr(i, j, k, 0);
                Set::Scalar My = M_arr(i, j, k, 1);

                Set::Scalar KE = (rho > Set::Scalar(0.0))
                    ? Set::Scalar(0.5) * (Mx*Mx + My*My) / rho : Set::Scalar(0.0);

                M_total  += rho * dV;
                M_phase0 += re0 * dV;
                M_phase1 += re1 * dV;
                M_vapor  += rvap_arr(i, j, k) * dV;
                Px_total += Mx  * dV;
                Py_total += My  * dV;
                E_total  += E_arr(i, j, k) * dV;
                KE_total += KE  * dV;
                Mdot_total += vap_arr(i, j, k, 1) * dV;   // INT m_dot_Vap dV [kg/s]
            });
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(M_total);
    amrex::ParallelDescriptor::ReduceRealSum(M_phase0);
    amrex::ParallelDescriptor::ReduceRealSum(M_vapor);
    amrex::ParallelDescriptor::ReduceRealSum(M_phase1);
    amrex::ParallelDescriptor::ReduceRealSum(Px_total);
    amrex::ParallelDescriptor::ReduceRealSum(Py_total);
    amrex::ParallelDescriptor::ReduceRealSum(E_total);
    amrex::ParallelDescriptor::ReduceRealSum(KE_total);
    amrex::ParallelDescriptor::ReduceRealSum(Mdot_total);

    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // Cumulative liquid->vapor mass transformed by the phase-change source,
        // INT INT m_dot_Vap dV dt, trapezoidally integrated from the volume-integrated
        // rate Mdot_total. With the action-reaction guard this tracks -dM_phase1 (and
        // dM_vapor when Yv_init=0) up to transport. Assumes WriteIntegrals fires once per
        // level-0 step; only the IOProcessor copy is maintained (the only consumer).
        if (!std::isfinite(Mdot_total)) Mdot_total = 0.0;  // Vap_dot may be unfilled at t=0
        if (wi_time_prev >= 0.0 && time > wi_time_prev)
            m_vap_transformed += 0.5 * (Mdot_total + wi_mdot_prev) * (time - wi_time_prev);
        wi_mdot_prev = Mdot_total;
        wi_time_prev = time;

        std::string fname = plot_file + "/integrals.dat";
        std::ifstream test(fname);
        bool write_header = (test.peek() == std::ifstream::traits_type::eof() || !test.good());
        test.close();

        std::ofstream outfile(fname, std::ios_base::app);
        if (write_header)
            outfile << "# 1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq"
                    << " 5:Px_total 6:Py_total 7:E_total 8:KE_total"
                    << " 9:M_vapor 10:M_carrier 11:M_vap_transformed"
                    << " 12:M_floor_gas 13:M_floor_liq 14:E_floor"
                    << " 15:E_floor_gas 16:E_floor_liq"
                    << " 17:p_preflr_min_gas 18:p_preflr_min_liq\n";

        outfile << std::setprecision(12) << time;
        outfile << std::setprecision(8)
                << "\t" << M_total  << "\t" << M_phase0 << "\t" << M_phase1
                << "\t" << Px_total << "\t" << Py_total
                << "\t" << E_total  << "\t" << KE_total
                << "\t" << M_vapor  << "\t" << (M_phase0 - M_vapor)
                << "\t" << m_vap_transformed
                << "\t" << m_floor_gas << "\t" << m_floor_liq
                << "\t" << e_floor
                << "\t" << e_floor_gas << "\t" << e_floor_liq
                << "\t" << p_preflr_min_gas << "\t" << p_preflr_min_liq
                << "\n";
        outfile.close();
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
    Set::Scalar dt,
    amrex::MultiFab &alpharho0_rhs_mf,
    amrex::MultiFab &alpharho1_rhs_mf,
    amrex::MultiFab &M_rhs_mf,
    amrex::MultiFab &E_rhs_mf,
    amrex::MultiFab &eta_rhs_mf,
    amrex::MultiFab &rho_vap_rhs_mf,
    const amrex::MultiFab &alpharho0_mf_in,
    const amrex::MultiFab &alpharho1_mf_in,
    const amrex::MultiFab &M_mf_in,
    const amrex::MultiFab &E_mf_in,
    const amrex::MultiFab &eta_mf_in,
    const amrex::MultiFab &rho_vap_mf_in)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Converting Array to mf
    amrex::MultiFab::Copy(*alpharho0_mf[lev], alpharho0_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*alpharho1_mf[lev], alpharho1_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*rho_vap_mf[lev], rho_vap_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*momentum_mf[lev], M_mf_in, 0, 0, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_mf_in, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*eta_mf[lev], eta_mf_in, 0, 0, 1, 0);

    // Eta Fields
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(alpharho0(i, j, k) + alpharho1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }
    
    // Primitive Fields (with BC)
    FillGhost4BC(lev, time);

    // Curvature pipeline (smooth-normals method fills kappas_mf before the RHS loop)
    if (kappa_method == 1)
        ComputeKappas(lev);

    // Pre-Source Terms
    for (amrex::MFIter mfi(*(velocity_mf)[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto const alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto const rho_vap  = rho_vap_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        auto const M = momentum_mf[lev]->array(mfi);
        auto const E = energy_per_vol_mf[lev]->array(mfi);

        // PRIMITIVE
        Set::Patch<const Set::Scalar> v = velocity_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> press = pressure_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> a = a_mf.Patch(lev, mfi);

        // SINGLE PHASE
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);

        // SOURCE - ish
        Set::Patch<const Set::Scalar> KE = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> UE = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cp = cp_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> cv = cv_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T = T_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Y = Y_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mass_frac_v = mass_frac_v_mf.Patch(lev, mfi);

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

            // Chemical Potential
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            //Set::Scalar mu_chem = -epsilon * lap_eta + f_prime / epsilon;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = alpharho0(i, j, k) / (rho(i, j, k));
            // Vapor mass fraction of the gas (Stage 3): rho_vap / alpharho0. ~0 in
            // the liquid (both ~0); = Y_v in the gas.
            // Cap Yv <= 0.99: rho_vap can transiently overshoot alpharho0 at isolated
            // points (advection/diffusion transients), giving an unphysical Yv > 1.
            mass_frac_v(i, j, k) = ClampYv(rho_vap(i, j, k) / std::max(alpharho0(i, j, k), small), 0.99);

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y)).
            // Stage 2: spalding_saturation -> physical Antoine saturation mass fraction
            // Y_s at the interface T as the driving force (carrier R from eos0_local,
            // vapor R = cp_vap-cv_vap; see SaturationYs) instead of the local cell Y.
            // T/press here are the FillGhost4BC-refreshed primitives (valid neighbors).
            Set::Scalar Y_drive = Y(i, j, k);
            if (spalding_saturation == 1)
                Y_drive = SaturationYs(T(i, j, k), press(i, j, k), antoine_A, antoine_B, antoine_C,
                                       cp_vap - cv_vap, eos0_local.Cp() - eos0_local.Cv(), small);
            // Stage 3d: sink mass fraction Y_inf for the Spalding driving force B_M =
            // (Y_s - Y_inf)/(1 - Y_s). Default = constant far-field Y_infinity (fixed
            // driving -> constant, non-Stefan rate). With spalding_film_sink=1 the sink
            // is sampled at a FILM distance ell = film_eps_mult*epsilon along +n_hat
            // (grad eta points into the gas), i.e. at the gas EDGE of the diffuse band.
            // This is deliberately NOT the adjacent cell (which fills with the vapor the
            // source just produced and would self-quench the rate -- the local-cell
            // closure pathology) and NOT a fixed far field (which never self-limits): as
            // vapor accumulates and diffuses, the sampled sink rises and the driving
            // falls, giving a transport-limited (sqrt(t)) recession. Only meaningful in
            // the band (the source ~ |grad eta|); elsewhere fall back to Y_infinity. The
            // cell offset is clamped to the ghost depth (needs dx >= ell/nghost so the
            // sample clears the band -- warned once in Mix()).
            Set::Scalar Y_sink = Y_infinity;
            if (spalding_film_sink == 1 && species_transport && grad_eta_mag > small)
            {
                const Set::Scalar inv_g = 1.0 / grad_eta_mag;
                const Set::Scalar ell   = film_eps_mult * epsilon;
                const Set::Scalar fx = ell * grad_eta(0) * inv_g / DX[0];
                const Set::Scalar fy = ell * grad_eta(1) * inv_g / DX[1];
                int di = (int)(fx + (fx >= 0.0 ? 0.5 : -0.5));
                int dj = (int)(fy + (fy >= 0.0 ? 0.5 : -0.5));
                di = (di < -nghost) ? -nghost : ((di > nghost) ? nghost : di);
                dj = (dj < -nghost) ? -nghost : ((dj > nghost) ? nghost : dj);
                Y_sink = rho_vap(i + di, j + dj, k) / std::max(alpharho0(i + di, j + dj, k), small);
            }
            Bm(i, j, k) = SpaldingBM(Y_drive, Y_sink, small);

            // Curvature (legacy Hessian-based method — stored in component 2 for diagnostics)
            if (kappa_method != 1)
            {
                Set::Vector n_hat = grad_eta / (grad_eta_mag + small);
                Set::Vector t1;
                if (abs(n_hat(0)) > abs(n_hat(1)))
                    t1 = Set::Vector(-n_hat(1), n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);
                else
                    t1 = Set::Vector(n_hat(1), -n_hat(0)) / sqrt(n_hat(0) * n_hat(0) + n_hat(1) * n_hat(1) + small);

                Set::Scalar kappa1 = -n_hat.dot(hess_eta * n_hat);
                Set::Scalar kappa2 = -t1.dot(hess_eta * t1) * 2.0 * epsilon;
                kappas(i, j, k, 0) = kappa2;
                kappas(i, j, k, 1) = kappa1;
                kappas(i, j, k, 2) = kappa2;
            }

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN RHS(): Primative Field Calculation", {
                { "alpharho0", alpharho0(i, j, k) }, 
                { "alpharho1", alpharho1(i, j, k) }, 
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

    // Fill mu_chem ghost cells with zero-neumann BCs before the main loop
    // computes lap(mu_chem) via central differences. Must use eta_bc
    // (zero-neumann), NOT energy_bc — energy_bc has dirichlet values
    // (e.g. 86M at xlo) that would create a massive spurious lap(mu_chem)
    // and corrupt eta across the entire domain via the CH term.
    FillBoundariesWithBC(lev, time, eta_bc, { mu_chem_mf[lev].get() });

    // ------------------------------------------------------------
    // HLLC-ADC carbuncle shock tag: a Ducros-weighted div(u) compression
    // sensor, thresholded per cell then grown by adc_grow cells. Below it
    // drives a per-face omega (0 = full HLL on the ADC-controlled components,
    // 1 = full HLLC) into the Riemann states, replacing the solver's
    // pressure-ratio sensor. Carbuncles are pressure-quiet, so a div(u) tag
    // plus dilation lands the HLL dissipation on the post-shock band where the
    // ripple lives, not just on the thin pressure jump. adc_grow < 0 disables
    // the tag and lets the solver use its internal pressure sensor.
    // ------------------------------------------------------------
    const bool use_shock_tag = (adc_grow >= 0);
    const int  tag_grow      = std::max(adc_grow, 0);
    // Force full HLL on every face for the first hll_first_steps step(s) (step_counter
    // is incremented at the top of Advance, so the first step's RHS sees == 1). This
    // overrides the shock tag below to skip the step-1 interface-carbuncle seed.
    const bool force_hll = (hll_first_steps > 0 && step_counter[lev] <= hll_first_steps);
    amrex::MultiFab shock_tag_local(velocity_mf[lev]->boxArray(),
                                    velocity_mf[lev]->DistributionMap(), 1, 1);
    shock_tag_local.setVal(0.0);
    shock_sensor_mf[lev]->setVal(0.0);
    pp_theta_mf[lev]->setVal(0.0);
    pp_dmass_mf[lev]->setVal(0.0);
    if (use_shock_tag)
    {
        amrex::MultiFab shock_raw(velocity_mf[lev]->boxArray(),
                                  velocity_mf[lev]->DistributionMap(), 1, tag_grow + 1);
        shock_raw.setVal(0.0);

        const Set::Scalar dr         = std::sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));
        const Set::Scalar thresh     = adc_shock_threshold;
        const Set::Scalar small_l    = small;
        const Set::Scalar iface_band = adc_interface_band;

        // Raw shock indicator per valid cell.
        for (amrex::MFIter mfi(shock_raw, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& sbx = mfi.validbox();
            auto raw = shock_raw.array(mfi);
            auto vv  = velocity_mf[lev]->array(mfi);
            auto aa  = a_mf[lev]->array(mfi);
            auto ee  = eta_mf[lev]->array(mfi);
            auto sensor_out = shock_sensor_mf[lev]->array(mfi);
            amrex::ParallelFor(sbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto sten = Numeric::GetStencil(i, j, k, domain);
                Set::Matrix grad_u = Numeric::Gradient(vv, i, j, k, DX, sten);
                Set::Scalar div_u = grad_u.trace();
                Set::Scalar comp = std::max(-div_u, 0.0);
#if AMREX_SPACEDIM == 2
                Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
                Set::Scalar curl_mag2 = wz * wz;
#else
                Set::Scalar wx = grad_u(2, 1) - grad_u(1, 2);
                Set::Scalar wy = grad_u(0, 2) - grad_u(2, 0);
                Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
                Set::Scalar curl_mag2 = wx * wx + wy * wy + wz * wz;
#endif
                Set::Scalar ducros = comp / std::sqrt(div_u * div_u + curl_mag2 + small_l * small_l);
                Set::Scalar c = std::max(aa(i, j, k), small_l);
                Set::Scalar sensor = ducros * comp * dr / c;
                sensor_out(i, j, k) = sensor;
                raw(i, j, k) = (sensor > thresh) ? 1.0 : 0.0;
                // Interface-compression tag: the Ducros weight kills the sensor on the
                // sheared diffuse interface, so a shock-compressed contact never trips
                // the threshold above. Tag it directly when it is inside the band AND
                // compressing, so HLL dissipation reaches the windward-contact overshoot
                // that would otherwise feed the liquid pressure/energy floor.
                if (iface_band > 0.0)
                {
                    const Set::Scalar eb = std::min(ee(i, j, k), 1.0 - ee(i, j, k));
                    if (eb > iface_band && div_u < 0.0) raw(i, j, k) = 1.0;
                }
            });
        }
        shock_raw.FillBoundary(geom[lev].periodicity());

        // Morphological dilation: tag a cell if any raw cell within tag_grow is set.
        for (amrex::MFIter mfi(shock_tag_local, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& sbx = mfi.validbox();
            auto tag = shock_tag_local.array(mfi);
            auto raw = shock_raw.array(mfi);
            const int g = tag_grow;
            amrex::ParallelFor(sbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar m = 0.0;
                for (int dj = -g; dj <= g; dj++)
                    for (int di = -g; di <= g; di++)
#if AMREX_SPACEDIM == 3
                        for (int dk = -g; dk <= g; dk++)
                            m = std::max(m, raw(i + di, j + dj, k + dk));
#else
                        m = std::max(m, raw(i + di, j + dj, k));
#endif
                tag(i, j, k) = m;
            });
        }
        shock_tag_local.FillBoundary(geom[lev].periodicity());
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

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto const alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto const rho_vap  = rho_vap_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        auto const M = momentum_mf[lev]->array(mfi);
        auto const E = energy_per_vol_mf[lev]->array(mfi);

        // OUTPUTS
        Set::Patch<Set::Scalar> alpharho0_rhs = alpharho0_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> alpharho1_rhs = alpharho1_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_vap_rhs  = rho_vap_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> M_rhs       = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs       = E_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> eta_rhs     = eta_rhs_mf.array(mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);

        // Cell-centered flux arrays for reflux at coarse-fine boundaries.
        // cc_fluxes use density_mf's BA/dmap, so mfi access is compatible.
        bool have_cc_fluxes = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y;
        amrex::Array4<Set::Scalar> ff_mom_x,  ff_mom_y;
        amrex::Array4<Set::Scalar> ff_ene_x,  ff_ene_y;
        if (have_cc_fluxes)
        {
            ff_mass_x = cc_fluxes[lev].mass[0]->array(mfi);
            ff_mass_y = cc_fluxes[lev].mass[1]->array(mfi);
            ff_mom_x  = cc_fluxes[lev].mom[0]->array(mfi);
            ff_mom_y  = cc_fluxes[lev].mom[1]->array(mfi);
            ff_ene_x  = cc_fluxes[lev].energy[0]->array(mfi);
            ff_ene_y  = cc_fluxes[lev].energy[1]->array(mfi);
        }

        // PP flux-limiter scratch: store the selected (high-order) and HLL
        // (low-order) hi-face fluxes for use in the limiter passes below.
        const bool pp_on = (pp_flux_limiter != 0 && lev < (int)pp_scratch.size()
                            && pp_scratch[lev].Fhi[0]);
        amrex::Array4<Set::Scalar> Fhi_x, Fhi_y, Flo_x, Flo_y;
        if (pp_on)
        {
            Fhi_x = pp_scratch[lev].Fhi[0]->array(mfi);
            Fhi_y = pp_scratch[lev].Fhi[1]->array(mfi);
            Flo_x = pp_scratch[lev].Flo[0]->array(mfi);
            Flo_y = pp_scratch[lev].Flo[1]->array(mfi);
        }

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
        Set::Patch<Set::Scalar> omega_adc = omega_adc_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> flux_dev = flux_dev_mf.Patch(lev, mfi);
        const Set::Scalar omega_alpha_local = omega_adc_alpha;
        amrex::Array4<const Set::Scalar> stag = shock_tag_local.const_array(mfi);

        const auto bx_lo = amrex::lbound(bx);

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
            // Fsv =  simga * kappa * grad_eta
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
                    
                    Fsv_vector(0) = sigma_eff * kappa * grad_eta(0); // * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
                    Fsv_vector(1) = sigma_eff * kappa * grad_eta(1); // * epsilon; // / (grad_eta_mag + small)); // / (DX[1] + small);
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
            // d(eta)/dt = -u*grad(eta) + Mob * laplacian(mu)
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
            // Cahn-Hilliard
            // ------------------------------------------------------------
            // Dr. Q mobility scaling:
            //   Pe_CH = (U / epsilon) / (M * sigma / epsilon^3)
            //         = U * epsilon^2 / (M * sigma).
            // Setting Pe_CH = O(1) gives M = M0 * epsilon^2 with
            // M0 = O(U / sigma).  The chemical potential stored here is
            // mu = f'(eta) - epsilon^2 lap(eta), so the dimensional
            // coefficient multiplying lap(mu) is M * sigma / epsilon.
            Set::Scalar lap_mu_chem = Numeric::Laplacian(mu_chem_, i, j, k, 0, DX);
            Set::Scalar M0_CH = Mob_user;
            Set::Scalar M_CH = M0_CH * epsilon * epsilon;
            Set::Scalar Mob = M_CH * sigma / epsilon;
            Set::Scalar eta_dot_CH = Mob * lap_mu_chem;

            // ERROR CHECKING
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Cahn-Hillard solving", {
                { "mu_chem",  mu_chem_(i, j, k) },
                { "lap_mu_chem",  lap_mu_chem },
                { "a",  a(i, j, k) },
                { "M0_CH",  M0_CH },
                { "M_CH",  M_CH },
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
            Set::Scalar E_dot_Vap = 0.0;
            if (apply_vaporization == 1)
            {
                //m_dot_Vap = (rho0(i, j, k) * Dv * (Bm(i, j, k) / (1.0 + Bm(i, j, k) + small)) * grad_eta_mag);
                //eta_dot_Vap += (1.0 / (rho(i, j, k) * epsilon)) * m_dot_vap;

                // Spalding mass transfer number. The driving force is folded into Bm
                // upstream in the primitives loop: Bm = (Y_drive - Y_inf)/(1 - Y_drive),
                // where Y_drive is the Antoine SATURATION mass fraction Y_s at the
                // interface T when spalding_saturation = 1 (Stage 2; see SaturationYs),
                // else the legacy local cell Y. The rate below just consumes Bm.
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // Gas density from fluid 0 (eta=1 corresponds to fluid 0)
                Set::Scalar rho_g = alpharho0(i, j, k);

                // Mass-transfer rate (volumetric) -- simplified Spalding form.
                //   areal flux   m'' = (rho_g * Dv / L_film) * (B_M/(1+B_M))   [kg/m^2/s]
                //   volumetric   m_dot_Vap = m'' * |grad eta|                  [kg/m^3/s]
                // L_film is the diffusion boundary-layer thickness (a PHYSICAL length);
                // |grad eta| is the band localizer (INT|grad eta|dV = interface area), NOT
                // the diffusion length. Including 1/L_film makes m'' a true areal flux
                // (matching the const-rate path, where vap_const_mdot is already [kg/m^2/s])
                // and m_dot_Vap dimensionally consistent with rho_vap_flux/rho_vap_diff
                // [kg/m^3/s]. Fixed L_film = QUASI-STEADY (constant rate); transient
                // sqrt(t) Stefan needs L -> delta(t) ~ sqrt(Dv*t). (We keep the simplified
                // (B_M/(1+B_M)) factor instead of ln(1+B_M) for now -- see F-3.)
                // CE path (rhoD_ref>0): the areal flux uses the gas transport
                // product rho*D(T) evaluated at the interface T (p-independent,
                // ~T^0.5) instead of the legacy rho_g*Dv, which inherited the
                // spurious density/pressure scaling of the partial density rho_g.
                // |grad eta| stays the band localizer; L_film the (quasi-steady)
                // film length. Legacy (rhoD_ref<=0): rho_g*Dv with rho_g=alpharho0.
                const Set::Scalar rhoD_film = (rhoD_ref > 0.0)
                    ? RhoD_CE(T(i, j, k), rhoD_ref, rhoD_Tref, rhoD_exp, small)
                    : rho_g * Dv;
                m_dot_Vap = rhoD_film * (B_M / (1.0 + B_M + small)) * grad_eta_mag / L_film;

                // Phase-change source for eta. IMPORTANT: this branch does NOT
                // carry independent intrinsic phase densities -- rho is the single
                // mixture density and alpharho0 = eta*rho, alpharho1 = (1-eta)*rho
                // (see L847-852), so eta is effectively the gas mass fraction
                // (Y = alpharho0/rho = eta). The canonical 5-eq form
                // m_dot*(1/rho_g - 1/rho_l) assumes true partial densities and does
                // NOT apply here. To keep eta consistent with alpharho0/rho under the
                // mass source (d(alpharho0)=+m_dot, d(rho)=0 by action-reaction):
                //     eta_dot = d(alpharho0/rho)/dt = m_dot / rho   (>0: eta -> gas)
                // Bounded (no 1/eta or 1/(1-eta)), so it is stable near pure phases.
                eta_dot_Vap = m_dot_Vap / std::max(rho(i, j, k), small);

                // Energy "flux" diagnostic -- NOT applied to Source[3] (commented
                // out at line 1259-ish below). Kept inactive per F-6 ignore;
                // formula reproduced below without referencing the deleted
                // M_dot_Vap (F-5) so it stays compileable. Mathematically
                // identical to the previous u.dot(u * m_dot_Vap * |grad_eta|) * |grad_eta|.
                E_dot_Vap = m_dot_Vap * (u(0)*u(0) + u(1)*u(1)) * grad_eta_mag * grad_eta_mag;
            }

            // ------------------------------------------------------------
            // Prescribed constant-rate phase change (VERIFICATION ONLY)
            // ------------------------------------------------------------
            // Bypasses the Spalding closure to impose a KNOWN interfacial mass
            // flux, so the integrated transfer is exactly
            //     dM_liq/dt = -vap_const_mdot * (interface area),
            // (and +the same into the gas), because the localizer |grad eta|
            // integrates to the interface area:  ∫|grad eta| dV = ∫ dA.  This
            // isolates the mass-transfer plumbing (action-reaction + per-phase
            // sink/source) from the questionable Spalding driving force. Liquid
            // -> vapor for vap_const_mdot > 0. Same eta-source form as Spalding.
            if (apply_vap_const == 1)
            {
                Set::Scalar m_dot_const = vap_const_mdot * grad_eta_mag;
                // eta source = m_dot/rho keeps eta = alpharho0/rho consistent under
                // the mass source (see Spalding block above for the derivation).
                m_dot_Vap   += m_dot_const;
                eta_dot_Vap += m_dot_const / std::max(rho(i, j, k), small);
            }

            // ------------------------------------------------------------
            // HRM bulk cavitation (Phase A: mass + void; latent heat deferred)
            // ------------------------------------------------------------
            // When the local pressure falls below saturation, convert liquid ->
            // vapor at a finite rate (Bilicki / Downar-Zapolski HRM): the gas phase
            // here IS n-dodecane vapor, so cavitation is just pressure-triggered
            // vaporization. Folds into the same m_dot_Vap/eta_dot_Vap, so the
            // partial-density update preserves action-reaction (gas +m, liquid -m).
            // Evaporation only (drive clamped >= 0; no condensation of the ambient
            // vapor) and it vanishes in pure gas since the rate scales with the
            // liquid partial density. The tension is relieved through the
            // eta-blended EOS (p0_eff drops as vapor forms); no latent-heat term is
            // applied yet (mixture energy held fixed -- Phase B, pending theory).
            if (apply_cavitation == 1)
            {
                Set::Scalar p_sat = std::pow(10.0, antoine_A - antoine_B / (T(i, j, k) + antoine_C)) * 1.0e5; // Antoine (bar) -> Pa
                Set::Scalar drive = (p_sat - press(i, j, k)) / (p_sat + small);
                drive = (drive < 0.0) ? 0.0 : ((drive > 1.0) ? 1.0 : drive);
                Set::Scalar m_dot_cav = (alpharho1(i, j, k) / tau_cav) * drive; // liquid -> vapor [kg/m^3/s]
                // eta source = m_dot/rho keeps eta = alpharho0/rho consistent under
                // the mass source (see Spalding block above for the derivation).
                m_dot_Vap   += m_dot_cav;
                eta_dot_Vap += m_dot_cav / std::max(rho(i, j, k), small);
            }

            // ------------------------------------------------------------
            // Conservation / positivity guard on the evaporation sink
            // ------------------------------------------------------------
            // The sink removes from alpharho1 = rho*(1-eta), the LIQUID partition (see the
            // convention block in Mix()). alpharho1 -> 0 across the gas region and the gas-side
            // half of the diffuse band, yet the source is still nonzero there (m_dot_Vap ~
            // |grad eta|, symmetric about eta=0.5). The FE step alpharho1 += dt*(-m_dot_Vap)
            // then pushes alpharho1 below the `small` floor, and the post-update floor (Advance)
            // clamps it and INJECTS mass -- while the matching +m_dot_Vap on alpharho0 is not
            // clipped. That breaks action-reaction (gas gains ~2x the liquid loss; ~0.5*m_vapor
            // of spurious mass in the sealed-box test). Cap m_dot_Vap at the rate that just
            // empties the available liquid to the floor; the SAME capped value feeds gas(+),
            // liquid(-), vapor(+), eta_dot, and latent, so the floor never bites and
            // m_dot_0 + m_dot_1 = 0 holds at the discrete level. (Only activates in the gas-side
            // band tail; the band-center bulk evaporation where alpharho1 is large is untouched.)
            // Physically: you cannot evaporate liquid that is not present in the cell.
            if (m_dot_Vap > 0.0 && dt > 0.0)
            {
                // (1) Mass-available cap: cannot evaporate liquid not present.
                Set::Scalar mdot_cap = std::max(alpharho1(i, j, k) - small, 0.0) / dt;
                // (2) Energy-available cap (the mass cap's missing twin). The latent
                // sink E_rhs += -L_vap*m_dot_Vap cannot pull the cell's internal energy
                // below the EOS pressure floor; if it does, the Advance backstop clamps
                // UE_vol up and INJECTS energy (non-conservative -- this is the
                // "mass=0, energy!=0" floor signature). The internal-energy margin above
                // the floor is UE_vol - ue_floor = (p - p_floor)/(gamma-1), so the
                // latent-safe rate is m_dot <= margin/(L_vap*dt). The 0.9 factor leaves
                // headroom for the other energy sinks in this cell (conduction/viscous).
                // Uses the SAME p_floor expression as the Advance backstop (PressureFloor
                // -> ue_floor). Gated on apply_latent_heat so the mass-only verification
                // runs stay bit-for-bit. The single capped m_dot feeds gas(+)/liquid(-)/
                // eta_dot/latent below, preserving discrete action-reaction.
                if (apply_latent_heat == 1 && L_vap > 0.0)
                {
                    const Set::Scalar p_floor   = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0_eff(i, j, k), eps_p, p_cav);
                    const Set::Scalar ue_margin = std::max(press(i, j, k) - p_floor, 0.0) / std::max(gammaf(i, j, k) - 1.0, small);
                    const Set::Scalar edot_avail = 0.9 * ue_margin / dt;   // J/m^3/s available to the latent sink
                    mdot_cap = std::min(mdot_cap, edot_avail / (L_vap + small));
                }
                m_dot_Vap = std::min(m_dot_Vap, mdot_cap);
            }
            // Re-derive eta_dot from the (possibly capped) m_dot so eta = alpharho0/rho stays
            // consistent (eta_dot_Vap above is exactly m_dot_Vap/rho before the cap).
            eta_dot_Vap = m_dot_Vap / std::max(rho(i, j, k), small);

            // ------------------------------------------------------------
            // Latent heat of vaporization (Stage 3c)
            // ------------------------------------------------------------
            // The Tammann/CPG EOS here carries no formation/reference energy, so
            // the phase-change enthalpy is NOT encoded in the internal energy and
            // must be added explicitly. Evaporation breaks liquid bonds, drawing
            // that energy from the mixture's sensible energy pool -> the energy
            // equation gets a sink (evaporative cooling):
            //     E_rhs += -L_vap * m_dot_Vap      [J/m^3/s]
            // with m_dot_Vap > 0 for liquid -> vapor (so E drops, T drops). It is a
            // single source on the shared mixture energy, applied to the total
            // m_dot_Vap (Spalding + const-rate + cavitation), so it is consistent
            // with the same m_dot_Vap that drives the mass transfer. Gated on
            // apply_latent_heat so the Phase-A mass-only verification runs (which
            // were checked with the energy held fixed) are bit-for-bit unchanged.
            Set::Scalar E_latent = (apply_latent_heat == 1) ? (-L_vap * m_dot_Vap) : 0.0;

            // Vaporization Trackers (Vap_dot[2..3] used to hold M_dot_Vap which
            // has been removed per F-5; left as zero so the plotfile field
            // shape is preserved without changing the registration).
            Vap_dot(i, j, k, 0) = eta_dot_Vap;
            Vap_dot(i, j, k, 1) = m_dot_Vap;
            Vap_dot(i, j, k, 2) = 0.0;
            Vap_dot(i, j, k, 3) = 0.0;
            Vap_dot(i, j, k, 4) = E_dot_Vap;


            // Total:
            Set::Vector Total_Force = Set::Vector(Fsv_vector(0) + Fw_vector(0),
                                                  Fsv_vector(1) + Fw_vector(1));
            
            Source(i, j, k, 0) = mdot0;

            // Momentum source from phase change is ZERO in 5-eq velocity-equilibrium
            // (vapor leaves at the local fluid velocity -- Saurel-Petitpas JFM 2008
            // Eq. 3). M_dot_Vap was removed per F-5.
            Source(i, j, k, 1) = Pdot0(0) + Ldot(0) + div_tau(0) + Total_Force(0);
            Source(i, j, k, 2) = Pdot0(1) + Ldot(1) + div_tau(1) + Total_Force(1);
            Source(i, j, k, 3) = qdot0 + u.dot(div_tau) + u.dot(Ldot) + u.dot(Total_Force) + E_latent;// + E_dot_Vap;

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

            // Populate the perpendicular-neighbor pressures on each face state.
            // Consumed by shock-aware solvers (HLLC-ADC) for the multidim sensor;
            // ignored by all other solvers. For an x-face the perpendiculars are
            // y-neighbors; for a y-face they are x-neighbors.
            // x-face at i-1/2: L is cell (i-1,j), R is cell (i,j)
            x_leftStates[1].p_cell    = press(i - 1, j, k);
            x_leftStates[1].p_perp_lo = press(i - 1, j - 1, k);
            x_leftStates[1].p_perp_hi = press(i - 1, j + 1, k);
            x_rightStates[1].p_cell    = press(i, j, k);
            x_rightStates[1].p_perp_lo = press(i, j - 1, k);
            x_rightStates[1].p_perp_hi = press(i, j + 1, k);
            // x-face at i+1/2: L is cell (i,j), R is cell (i+1,j)
            x_leftStates[2].p_cell    = press(i, j, k);
            x_leftStates[2].p_perp_lo = press(i, j - 1, k);
            x_leftStates[2].p_perp_hi = press(i, j + 1, k);
            x_rightStates[2].p_cell    = press(i + 1, j, k);
            x_rightStates[2].p_perp_lo = press(i + 1, j - 1, k);
            x_rightStates[2].p_perp_hi = press(i + 1, j + 1, k);
            // y-face at j-1/2: L is cell (i,j-1), R is cell (i,j)
            y_leftStates[1].p_cell    = press(i, j - 1, k);
            y_leftStates[1].p_perp_lo = press(i - 1, j - 1, k);
            y_leftStates[1].p_perp_hi = press(i + 1, j - 1, k);
            y_rightStates[1].p_cell    = press(i, j, k);
            y_rightStates[1].p_perp_lo = press(i - 1, j, k);
            y_rightStates[1].p_perp_hi = press(i + 1, j, k);
            // y-face at j+1/2: L is cell (i,j), R is cell (i,j+1)
            y_leftStates[2].p_cell    = press(i, j, k);
            y_leftStates[2].p_perp_lo = press(i - 1, j, k);
            y_leftStates[2].p_perp_hi = press(i + 1, j, k);
            y_rightStates[2].p_cell    = press(i, j + 1, k);
            y_rightStates[2].p_perp_lo = press(i - 1, j + 1, k);
            y_rightStates[2].p_perp_hi = press(i + 1, j + 1, k);

            // Drive the HLLC-ADC blend from the div(u) shock tag: a face inside
            // the (dilated) tagged region gets omega = 0 (full HLL on the
            // controlled components), otherwise omega = 1 (full HLLC). Both
            // sides of a face carry the same value. When the tag is disabled
            // (adc_grow < 0) omega_ext stays -1 and the solver uses its
            // internal pressure-ratio sensor.
            if (force_hll)
            {
                // hll_first_steps active: omega = 0 on every face -> full HLL on the
                // ADC-controlled components (genuine pure HLL when adc_components=4).
                // Takes precedence over the shock tag so the step-1 carbuncle never seeds.
                x_leftStates[1].omega_ext = x_rightStates[1].omega_ext = 0.0;
                x_leftStates[2].omega_ext = x_rightStates[2].omega_ext = 0.0;
                y_leftStates[1].omega_ext = y_rightStates[1].omega_ext = 0.0;
                y_leftStates[2].omega_ext = y_rightStates[2].omega_ext = 0.0;
            }
            else if (use_shock_tag)
            {
                Set::Scalar tc  = stag(i, j, k);
                Set::Scalar txm = stag(i - 1, j, k);
                Set::Scalar txp = stag(i + 1, j, k);
                Set::Scalar tym = stag(i, j - 1, k);
                Set::Scalar typ = stag(i, j + 1, k);
                Set::Scalar om_xlo = (txm > 0.5 || tc > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_xhi = (tc > 0.5 || txp > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_ylo = (tym > 0.5 || tc > 0.5) ? 0.0 : 1.0;
                Set::Scalar om_yhi = (tc > 0.5 || typ > 0.5) ? 0.0 : 1.0;
                x_leftStates[1].omega_ext = x_rightStates[1].omega_ext = om_xlo;
                x_leftStates[2].omega_ext = x_rightStates[2].omega_ext = om_xhi;
                y_leftStates[1].omega_ext = y_rightStates[1].omega_ext = om_ylo;
                y_leftStates[2].omega_ext = y_rightStates[2].omega_ext = om_yhi;
            }

            // ------------------------------------------------------------
            // HLLC-ADC shock locator diagnostic.
            // Compute omega at the cell's two hi-faces (i+1/2, j) and
            // (i, j+1/2) using the same 5-face pressure-ratio min that
            // HLLC_ADC consumes. Written every step so the locator can be
            // inspected regardless of which Riemann solver is selected.
            //   omega = ( min_k f_k )^alpha,
            //   f_k   = min(p_a^k, p_b^k) / max(p_a^k, p_b^k).
            // ------------------------------------------------------------
            {
                const Set::Scalar omega_small = 1e-30;
                auto omega_face_ratio = [&](Set::Scalar pa, Set::Scalar pb) -> Set::Scalar
                {
                    Set::Scalar pmin = std::min(pa, pb);
                    Set::Scalar pmax = std::max(pa, pb);
                    return pmin / (pmax + omega_small);
                };
                auto omega_at_face = [&](const Solver::Local::FluidRiemann::State& L,
                                         const Solver::Local::FluidRiemann::State& R) -> Set::Scalar
                {
                    Set::Scalar fm = omega_face_ratio(L.p_cell, R.p_cell);
                    if (L.p_cell > 0.0 && L.p_perp_lo > 0.0 && L.p_perp_hi > 0.0)
                    {
                        fm = std::min(fm, omega_face_ratio(L.p_perp_lo, L.p_cell));
                        fm = std::min(fm, omega_face_ratio(L.p_cell,    L.p_perp_hi));
                    }
                    if (R.p_cell > 0.0 && R.p_perp_lo > 0.0 && R.p_perp_hi > 0.0)
                    {
                        fm = std::min(fm, omega_face_ratio(R.p_perp_lo, R.p_cell));
                        fm = std::min(fm, omega_face_ratio(R.p_cell,    R.p_perp_hi));
                    }
                    return std::pow(fm, omega_alpha_local);
                };
                if (use_shock_tag)
                {
                    // Report the tag-based omega actually fed to the solver.
                    omega_adc(i, j, k, 0) = (stag(i, j, k) > 0.5 || stag(i + 1, j, k) > 0.5) ? 0.0 : 1.0;
                    omega_adc(i, j, k, 1) = (stag(i, j, k) > 0.5 || stag(i, j + 1, k) > 0.5) ? 0.0 : 1.0;
                }
                else
                {
                    omega_adc(i, j, k, 0) = omega_at_face(x_leftStates[2], x_rightStates[2]); // x-hi face
                    omega_adc(i, j, k, 1) = omega_at_face(y_leftStates[2], y_rightStates[2]); // y-hi face
                }
            }

            // Calculate fluxes using the mixed fluid approach
            Solver::Local::FluidRiemann::Flux flux_xlo, flux_ylo, flux_xhi, flux_yhi;
            
            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Conservative Variable Check", { 
                { "eta", eta(i, j, k) },
                { "alpharho0", alpharho0(i, j, k) },
                { "alpharho1", alpharho1(i, j, k) },
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

            // ------------------------------------------------------------
            // PP flux limiter: store the high-order (selected) and low-order
            // (HLL) hi-face fluxes in fixed (x,y) momentum frame so the limiter
            // passes can form the blend F^l + theta*(F^h - F^l). Lo-face fluxes
            // are written into the lo ghost at box boundaries (FillBoundary
            // overwrites interior box boundaries; physical boundaries keep the
            // BC-based flux). Components: 0=mass,1=x-mom,2=y-mom,3=energy,
            // 4=u_interface (Fhi only).
            // ------------------------------------------------------------
            if (pp_on)
            {
                Solver::Local::FluidRiemann::Flux hll_xlo = hll_solver->Solve(x_leftStates[1], x_rightStates[1], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_ylo = hll_solver->Solve(y_leftStates[1], y_rightStates[1], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_xhi = hll_solver->Solve(x_leftStates[2], x_rightStates[2], pref, small, Spec_Vol);
                Solver::Local::FluidRiemann::Flux hll_yhi = hll_solver->Solve(y_leftStates[2], y_rightStates[2], pref, small, Spec_Vol);

                // x hi-face (i+1/2): normal=x, tangent=y
                Fhi_x(i, j, k, 0) = flux_xhi.mass;
                Fhi_x(i, j, k, 1) = flux_xhi.momentum_normal;
                Fhi_x(i, j, k, 2) = flux_xhi.momentum_tangent;
                Fhi_x(i, j, k, 3) = flux_xhi.energy;
                Fhi_x(i, j, k, 4) = flux_xhi.u_interface;
                Flo_x(i, j, k, 0) = hll_xhi.mass;
                Flo_x(i, j, k, 1) = hll_xhi.momentum_normal;
                Flo_x(i, j, k, 2) = hll_xhi.momentum_tangent;
                Flo_x(i, j, k, 3) = hll_xhi.energy;
                Flo_x(i, j, k, 4) = hll_xhi.u_interface;

                // y hi-face (j+1/2): normal=y, tangent=x
                Fhi_y(i, j, k, 0) = flux_yhi.mass;
                Fhi_y(i, j, k, 1) = flux_yhi.momentum_tangent;
                Fhi_y(i, j, k, 2) = flux_yhi.momentum_normal;
                Fhi_y(i, j, k, 3) = flux_yhi.energy;
                Fhi_y(i, j, k, 4) = flux_yhi.u_interface;
                Flo_y(i, j, k, 0) = hll_yhi.mass;
                Flo_y(i, j, k, 1) = hll_yhi.momentum_tangent;
                Flo_y(i, j, k, 2) = hll_yhi.momentum_normal;
                Flo_y(i, j, k, 3) = hll_yhi.energy;
                Flo_y(i, j, k, 4) = hll_yhi.u_interface;

                if (i == bx_lo.x) {
                    Fhi_x(i - 1, j, k, 0) = flux_xlo.mass;
                    Fhi_x(i - 1, j, k, 1) = flux_xlo.momentum_normal;
                    Fhi_x(i - 1, j, k, 2) = flux_xlo.momentum_tangent;
                    Fhi_x(i - 1, j, k, 3) = flux_xlo.energy;
                    Fhi_x(i - 1, j, k, 4) = flux_xlo.u_interface;
                    Flo_x(i - 1, j, k, 0) = hll_xlo.mass;
                    Flo_x(i - 1, j, k, 1) = hll_xlo.momentum_normal;
                    Flo_x(i - 1, j, k, 2) = hll_xlo.momentum_tangent;
                    Flo_x(i - 1, j, k, 3) = hll_xlo.energy;
                    Flo_x(i - 1, j, k, 4) = hll_xlo.u_interface;
                }
                if (j == bx_lo.y) {
                    Fhi_y(i, j - 1, k, 0) = flux_ylo.mass;
                    Fhi_y(i, j - 1, k, 1) = flux_ylo.momentum_tangent;
                    Fhi_y(i, j - 1, k, 2) = flux_ylo.momentum_normal;
                    Fhi_y(i, j - 1, k, 3) = flux_ylo.energy;
                    Fhi_y(i, j - 1, k, 4) = flux_ylo.u_interface;
                    Flo_y(i, j - 1, k, 0) = hll_ylo.mass;
                    Flo_y(i, j - 1, k, 1) = hll_ylo.momentum_tangent;
                    Flo_y(i, j - 1, k, 2) = hll_ylo.momentum_normal;
                    Flo_y(i, j - 1, k, 3) = hll_ylo.energy;
                    Flo_y(i, j - 1, k, 4) = hll_ylo.u_interface;
                }

                // Diagnostic: relative deviation of the used flux from pure HLL
                // at the two hi-faces, max over conserved components. ~0 at a
                // tagged face confirms it actually produced an HLL flux.
                auto rel_dev = [&](const Solver::Local::FluidRiemann::Flux& fh,
                                   const Solver::Local::FluidRiemann::Flux& fl_) -> Set::Scalar
                {
                    Set::Scalar num = std::max(std::max(std::abs(fh.mass            - fl_.mass),
                                                        std::abs(fh.momentum_normal - fl_.momentum_normal)),
                                               std::max(std::abs(fh.momentum_tangent- fl_.momentum_tangent),
                                                        std::abs(fh.energy          - fl_.energy)));
                    Set::Scalar den = std::max(std::max(std::abs(fl_.mass),            std::abs(fl_.momentum_normal)),
                                               std::max(std::abs(fl_.momentum_tangent),std::abs(fl_.energy)));
                    return num / (den + small);
                };
                flux_dev(i, j, k, 0) = rel_dev(flux_xhi, hll_xhi);
                flux_dev(i, j, k, 1) = rel_dev(flux_yhi, hll_yhi);
            }
            else
            {
                flux_dev(i, j, k, 0) = 0.0;
                flux_dev(i, j, k, 1) = 0.0;
            }

            // Upwind volume fractions (face-centered alpha, advected by HLLC contact wave
            // speed u*). Used ONLY by the non-conservative volume-fraction (eta) advection.
            Set::Scalar eta_face_xlo = (flux_xlo.u_interface > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
            Set::Scalar eta_face_xhi = (flux_xhi.u_interface > 0.0) ? eta(i, j, k) : eta(i + 1, j, k);
            Set::Scalar eta_face_ylo = (flux_ylo.u_interface > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);
            Set::Scalar eta_face_yhi = (flux_yhi.u_interface > 0.0) ? eta(i, j, k) : eta(i, j + 1, k);

            // Upwind GAS MASS FRACTION Y_0 = alpharho0/rho (the true mass fraction, NOT eta).
            // The partial density alpharho0 = alpha_0*rho_0 advects with the mixture mass flux
            // weighted by Y_0, so ff_0 = Y0_face*F_mass and ff_1 = (1-Y0_face)*F_mass telescope
            // to F_mass (=> total mass conserved). rho here is the reconstructed mixture density
            // (= alpharho0+alpharho1). This is the conservation-critical change vs the retired
            // rho_eta convention, which split by eta and so did not transport the true masses.
            // DONOR RULE (Larrouturou 1991): Y_0 is upwinded by the sign of the MASS FLUX it
            // multiplies, NOT by u_interface. For pure HLLC the two agree (F_mass = rho* S*),
            // but at ADC-tagged / HLL faces the diffusive S_L*S_R*(rho_R - rho_L) term can flip
            // the mass-flux sign at a gas-liquid face; an u_interface donor then drains a
            // partial density from a cell holding none of that phase (alpharho < 0 -> floored
            // -> mass created -- the shock-droplet floor leak). Flux-sign upwinding keeps the
            // partial update positivity-preserving whenever the mixture update is.
            auto Y0_up = [&](int ii, int jj) {
                return alpharho0(ii, jj, k) / std::max(rho(ii, jj, k), small);
            };
            Set::Scalar Y0_face_xlo = (flux_xlo.mass > 0.0) ? Y0_up(i - 1, j) : Y0_up(i, j);
            Set::Scalar Y0_face_xhi = (flux_xhi.mass > 0.0) ? Y0_up(i, j)     : Y0_up(i + 1, j);
            Set::Scalar Y0_face_ylo = (flux_ylo.mass > 0.0) ? Y0_up(i, j - 1) : Y0_up(i, j);
            Set::Scalar Y0_face_yhi = (flux_yhi.mass > 0.0) ? Y0_up(i, j)     : Y0_up(i, j + 1);

            // -----------------------------------------------------------
            // Store hi-face Riemann fluxes at cell centers for reflux.
            // Cell (i,j) stores the flux at face (i+1/2,j) for d=0
            // and at face (i,j+1/2) for d=1.  Converted to face-centered
            // MultiFabs in Advance() for CrseInit/FineAdd.
            // -----------------------------------------------------------
            if (have_cc_fluxes)
            {
                // x-direction hi-face: per-phase mass fluxes split by the true mass fraction Y_0
                ff_mass_x(i, j, k, 0) = Y0_face_xhi * flux_xhi.mass;           // alpharho0
                ff_mass_x(i, j, k, 1) = (1.0 - Y0_face_xhi) * flux_xhi.mass;  // alpharho1
                ff_mom_x(i, j, k, 0)  = flux_xhi.momentum_normal;   // x-mom
                ff_mom_x(i, j, k, 1)  = flux_xhi.momentum_tangent;  // y-mom
                ff_ene_x(i, j, k)     = flux_xhi.energy;

                // y-direction hi-face: per-phase mass fluxes (swap mom to fixed x,y)
                ff_mass_y(i, j, k, 0) = Y0_face_yhi * flux_yhi.mass;           // alpharho0
                ff_mass_y(i, j, k, 1) = (1.0 - Y0_face_yhi) * flux_yhi.mass;  // alpharho1
                ff_mom_y(i, j, k, 0)  = flux_yhi.momentum_tangent;  // x-mom
                ff_mom_y(i, j, k, 1)  = flux_yhi.momentum_normal;   // y-mom
                ff_ene_y(i, j, k)     = flux_yhi.energy;

                // Write lo-face fluxes into ghost cells at box boundaries.
                // At coarse-fine interfaces FillBoundary has no neighbor to
                // copy from, so the ghost would stay zero without this.
                // At interior box boundaries FillBoundary overwrites later.
                if (i == bx_lo.x) {
                    ff_mass_x(i - 1, j, k, 0) = Y0_face_xlo * flux_xlo.mass;
                    ff_mass_x(i - 1, j, k, 1) = (1.0 - Y0_face_xlo) * flux_xlo.mass;
                    ff_mom_x(i - 1, j, k, 0)  = flux_xlo.momentum_normal;
                    ff_mom_x(i - 1, j, k, 1)  = flux_xlo.momentum_tangent;
                    ff_ene_x(i - 1, j, k)     = flux_xlo.energy;
                }
                if (j == bx_lo.y) {
                    ff_mass_y(i, j - 1, k, 0) = Y0_face_ylo * flux_ylo.mass;
                    ff_mass_y(i, j - 1, k, 1) = (1.0 - Y0_face_ylo) * flux_ylo.mass;
                    ff_mom_y(i, j - 1, k, 0)  = flux_ylo.momentum_tangent;
                    ff_mom_y(i, j - 1, k, 1)  = flux_ylo.momentum_normal;
                    ff_ene_y(i, j - 1, k)     = flux_ylo.energy;
                }
            }

            // ------------------------------------------------------------
            // Non-conservative volume-fraction advection (Saurel & Abgrall 1999, Eq. 41)
            //   D(eta)/Dt = deta/dt + u*grad(eta) = 0
            // Discretized as
            //   deta/dt = -div(u eta) + eta * div(u)
            // ------------------------------------------------------------
            {
                Set::Scalar div_uA_x = (flux_xhi.u_interface * eta_face_xhi
                                      - flux_xlo.u_interface * eta_face_xlo) / DX[0];
                Set::Scalar div_uA_y = (flux_yhi.u_interface * eta_face_yhi
                                      - flux_ylo.u_interface * eta_face_ylo) / DX[1];
                Set::Scalar div_u_x  = (flux_xhi.u_interface - flux_xlo.u_interface) / DX[0];
                Set::Scalar div_u_y  = (flux_yhi.u_interface - flux_ylo.u_interface) / DX[1];
                Set::Scalar eta_adv  = -(div_uA_x + div_uA_y)
                                       + eta(i, j, k) * (div_u_x + div_u_y);
                // When the PP limiter is active the conserved fluxes are blended
                // toward HLL per face; the volume fraction must advect with the
                // SAME blended interface velocity, or eta stays razor-sharp while
                // rho diffuses and the stiffened-liquid EOS amplifies the
                // mismatch into a spurious pressure wave. Defer the advective
                // term to Pass D (recomputed with the blended u); keep only the
                // non-advective sources here. With the limiter off, advect now.
                eta_rhs(i, j, k) = (pp_on ? 0.0 : eta_adv)
                                   + eta_dot_Vap
                                   + eta_dot_CH; // CHAN HILLARD (CH)
            }


            // UPDATE MIXED FLUID VARIABLES
            rho_flux(i, j, k) = (flux_xlo.mass - flux_xhi.mass) / (DX[0]) + (flux_ylo.mass - flux_yhi.mass) / (DX[1]);
            M_flux(i, j, k, 0) = (flux_xlo.momentum_normal - flux_xhi.momentum_normal) / (DX[0]) + (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / (DX[1]);
            M_flux(i, j, k, 1) = (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / (DX[0]) + (flux_ylo.momentum_normal - flux_yhi.momentum_normal) / (DX[1]);
            E_flux(i, j, k) = (flux_xlo.energy - flux_xhi.energy) / (DX[0]) + (flux_ylo.energy - flux_yhi.energy) / (DX[1]);

            // Partial-density mass fluxes, split by the true upwind mass fraction Y_0
            // (NOT eta): F_alpharho0 = Y0_face*F_mass, F_alpharho1 = (1-Y0_face)*F_mass.
            // The two sum to F_mass per face, so the total mass flux telescopes exactly.
            Set::Scalar F_alpharho0_xlo = Y0_face_xlo * flux_xlo.mass;
            Set::Scalar F_alpharho0_xhi = Y0_face_xhi * flux_xhi.mass;
            Set::Scalar F_alpharho0_ylo = Y0_face_ylo * flux_ylo.mass;
            Set::Scalar F_alpharho0_yhi = Y0_face_yhi * flux_yhi.mass;

            Set::Scalar F_alpharho1_xlo = (1.0 - Y0_face_xlo) * flux_xlo.mass;
            Set::Scalar F_alpharho1_xhi = (1.0 - Y0_face_xhi) * flux_xhi.mass;
            Set::Scalar F_alpharho1_ylo = (1.0 - Y0_face_ylo) * flux_ylo.mass;
            Set::Scalar F_alpharho1_yhi = (1.0 - Y0_face_yhi) * flux_yhi.mass;

            Set::Scalar alpharho0_flux = (F_alpharho0_xlo - F_alpharho0_xhi) / DX[0]
                                        + (F_alpharho0_ylo - F_alpharho0_yhi) / DX[1];

            Set::Scalar alpharho1_flux = (F_alpharho1_xlo - F_alpharho1_xhi) / DX[0]
                                        + (F_alpharho1_ylo - F_alpharho1_yhi) / DX[1];

            // Mass
            alpharho0_rhs(i, j, k) = alpharho0_flux + Source(i, j, k, 0) * (eta(i, j, k)) + m_dot_Vap;
            alpharho1_rhs(i, j, k) = alpharho1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k)) - m_dot_Vap;

            // Vapor species (Stage 3a, checkpoint 2): rho_vap advects with the
            // mixture mass flux, weighted by the upwind vapor-fraction-of-total
            // (rho_vap/rho) with the SAME upwind direction as eta_face. The
            // neighbor index is clamped only at PHYSICAL domain edges (= neumann
            // for rho_vap); at box / coarse-fine interfaces it reads rho_vap
            // ghosts, which FillGhost4BC now FillBoundary's (same level) and
            // FillPatch'es (coarse-fine). Evaporation creates
            // vapor (+m_dot_Vap), matching the +m_dot_Vap added to alpharho0, so
            // carrier (= alpharho0 - rho_vap) is conserved. This is the FE/no-limiter
            // value; when the PP limiter is on, Pass D OVERWRITES rho_vap_rhs with the
            // blended-mass-flux version (consistent with alpharho0) -- see there.
            if (species_transport)
            {
                const int dlo0 = domain.smallEnd(0), dhi0 = domain.bigEnd(0);
                const int dlo1 = domain.smallEnd(1), dhi1 = domain.bigEnd(1);
                const int im = (i > dlo0) ? i - 1 : i;
                const int ip = (i < dhi0) ? i + 1 : i;
                const int jm = (j > dlo1) ? j - 1 : j;
                const int jp = (j < dhi1) ? j + 1 : j;
                // Same Larrouturou donor rule as Y0_face above: upwind by the
                // sign of the mass flux this fraction multiplies, not u_interface.
                Set::Scalar fvap_xlo = (flux_xlo.mass > 0.0)
                    ? rho_vap(im, j, k) / std::max(rho(im, j, k), small)
                    : rho_vap(i, j, k)  / std::max(rho(i, j, k), small);
                Set::Scalar fvap_xhi = (flux_xhi.mass > 0.0)
                    ? rho_vap(i, j, k)  / std::max(rho(i, j, k), small)
                    : rho_vap(ip, j, k) / std::max(rho(ip, j, k), small);
                Set::Scalar fvap_ylo = (flux_ylo.mass > 0.0)
                    ? rho_vap(i, jm, k) / std::max(rho(i, jm, k), small)
                    : rho_vap(i, j, k)  / std::max(rho(i, j, k), small);
                Set::Scalar fvap_yhi = (flux_yhi.mass > 0.0)
                    ? rho_vap(i, j, k)  / std::max(rho(i, j, k), small)
                    : rho_vap(i, jp, k) / std::max(rho(i, jp, k), small);
                Set::Scalar rho_vap_flux = (fvap_xlo * flux_xlo.mass - fvap_xhi * flux_xhi.mass) / DX[0]
                                         + (fvap_ylo * flux_ylo.mass - fvap_yhi * flux_yhi.mass) / DX[1];

                // Store the hi-face rho_vap ADVECTIVE flux for reflux (mass comp 2),
                // mirroring the alpharho0/alpharho1 storage above. Diffusion (rho_vap_diff)
                // is deliberately NOT refluxed -- same approximation as conduction, which
                // is a source rather than a cc_flux. fvap_face <= eta_face (rho_vap<=alpharho0)
                // per face, so the rho_vap reflux correction stays bounded by alpharho0's,
                // keeping the carrier (alpharho0 - rho_vap) consistent at coarse-fine edges.
                if (have_cc_fluxes)
                {
                    ff_mass_x(i, j, k, 2) = fvap_xhi * flux_xhi.mass;
                    ff_mass_y(i, j, k, 2) = fvap_yhi * flux_yhi.mass;
                    if (i == bx_lo.x) ff_mass_x(i - 1, j, k, 2) = fvap_xlo * flux_xlo.mass;
                    if (j == bx_lo.y) ff_mass_y(i, j - 1, k, 2) = fvap_ylo * flux_ylo.mass;
                }

                // Fickian vapor diffusion through the inert carrier (Stage 3b-1):
                //   d(rho_vap)/dt += div( alpharho0 * Dv * grad(Y_v) ),  Y_v = rho_vap/alpharho0.
                // alpharho0 (= alpha_0*rho_0) is the gas mass per total volume, so this is the
                // vapor mass flux per total area -- binary diffusion of vapor through the
                // carrier. Conservative central Laplacian with arithmetic-mean face
                // alpharho0; the clamped neighbor indices (im/ip/jm/jp) give a zero-gradient
                // (no-flux) diffusive boundary at the domain edges (neumann), correct for
                // the 1D Stefan tests. It only redistributes vapor within the fixed
                // alpharho0 field, so the carrier (= alpharho0 - rho_vap) counter-diffuses
                // and INT(rho_vap) is unchanged for no-flux walls. Single-grid / FE path
                // only (same rho_vap ghost-fill limitation as the advection above).
                Set::Scalar rho_vap_diff = 0.0;
                if (Dv > 0.0 || rhoD_ref > 0.0)
                {
                    // Clamp Y_v to [0,1] (NaN-safe) BEFORE differencing. The raw
                    // ratio rho_vap/alpharho0 blows up where the gas partial density
                    // collapses (lee/liquid side of the interface): alpharho0 -> floor
                    // makes Y_v ~ O(1e15), the explicit diffusion overflows to inf and
                    // then inf-inf -> NaN -- and dt_species = cfl*dx^2/Dv never sees it
                    // (that bound assumes a clean Dv-Laplacian). Clamping caps the
                    // operator at the nominal Dv scale dt_species already respects; the
                    // face coefficient below still vanishes where there is no carrier
                    // gas (eta or alpharho0 -> 0), so deep-liquid contributions stay ~0.
                    const Set::Scalar Yv_c  = ClampYv(rho_vap(i, j, k)  / std::max(alpharho0(i, j, k), small));
                    const Set::Scalar Yv_im = ClampYv(rho_vap(im, j, k) / std::max(alpharho0(im, j, k), small));
                    const Set::Scalar Yv_ip = ClampYv(rho_vap(ip, j, k) / std::max(alpharho0(ip, j, k), small));
                    const Set::Scalar Yv_jm = ClampYv(rho_vap(i, jm, k) / std::max(alpharho0(i, jm, k), small));
                    const Set::Scalar Yv_jp = ClampYv(rho_vap(i, jp, k) / std::max(alpharho0(i, jp, k), small));
                    // Face transport coefficient for the vapor-mass flux per total area.
                    // Legacy: alpharho0_face * Dv  (= alpha0*rho_0*Dv, rho_0 varying with
                    // the wrong sign). CE: alpha0_face * rho*D(T_face), with alpha0 = eta
                    // the gas volume-fraction localizer and rho*D the p-independent CE
                    // product evaluated at the arithmetic-mean face temperature.
                    Set::Scalar c_xlo, c_xhi, c_ylo, c_yhi;
                    if (rhoD_ref > 0.0)
                    {
                        c_xlo = 0.5 * (eta(im, j, k) + eta(i, j, k)) * RhoD_CE(0.5 * (T(im, j, k) + T(i, j, k)), rhoD_ref, rhoD_Tref, rhoD_exp, small);
                        c_xhi = 0.5 * (eta(i, j, k) + eta(ip, j, k)) * RhoD_CE(0.5 * (T(i, j, k) + T(ip, j, k)), rhoD_ref, rhoD_Tref, rhoD_exp, small);
                        c_ylo = 0.5 * (eta(i, jm, k) + eta(i, j, k)) * RhoD_CE(0.5 * (T(i, jm, k) + T(i, j, k)), rhoD_ref, rhoD_Tref, rhoD_exp, small);
                        c_yhi = 0.5 * (eta(i, j, k) + eta(i, jp, k)) * RhoD_CE(0.5 * (T(i, j, k) + T(i, jp, k)), rhoD_ref, rhoD_Tref, rhoD_exp, small);
                    }
                    else
                    {
                        c_xlo = 0.5 * (alpharho0(im, j, k) + alpharho0(i, j, k)) * Dv;
                        c_xhi = 0.5 * (alpharho0(i, j, k) + alpharho0(ip, j, k)) * Dv;
                        c_ylo = 0.5 * (alpharho0(i, jm, k) + alpharho0(i, j, k)) * Dv;
                        c_yhi = 0.5 * (alpharho0(i, j, k) + alpharho0(i, jp, k)) * Dv;
                    }
                    rho_vap_diff = ( (c_xhi * (Yv_ip - Yv_c) - c_xlo * (Yv_c - Yv_im)) / (DX[0] * DX[0])
                                   + (c_yhi * (Yv_jp - Yv_c) - c_ylo * (Yv_c - Yv_jm)) / (DX[1] * DX[1]) );
                }
                rho_vap_rhs(i, j, k) = rho_vap_flux + m_dot_Vap + rho_vap_diff;
            }
            else
            {
                rho_vap_rhs(i, j, k) = 0.0;
            }

            // Momentum
            M_rhs(i, j, k, 0) = M_flux(i, j, k, 0) + Source(i, j, k, 1); //(mu * (lap_ux * eta(i, j, k))) +
            M_rhs(i, j, k, 1) = M_flux(i, j, k, 1) + Source(i, j, k, 2); //(mu * (lap_uy * eta(i, j, k))) +

            // Energy
            // Fourier thermal conduction (Stage 3b-2): d(E)/dt += div( k grad T ),
            // k = eta*k0_thermal + (1-eta)*k1_thermal (eta-blended like the viscosity
            // mu_eff above). Conservative central Laplacian with arithmetic-mean face
            // k; the clamped neighbor indices give a zero-gradient (adiabatic / no-flux)
            // wall at the domain edges -- correct for the neumann energy BC and the 1D
            // Stefan setup. T is refreshed every RHS by FillGhost4BC. Gated on k>0 so
            // all conduction-off (k0=k1=0) runs are bit-for-bit unchanged. Folded into
            // Source[3] below so it flows through the source limiter (Pass D), exactly
            // like latent heat -- instead of being dropped when the limiter overwrites
            // E_rhs (same valid-neighbor/clamped access as the species diffusion above).
            Set::Scalar E_cond = 0.0;
            if (k0_thermal > 0.0 || k1_thermal > 0.0)
            {
                const int dlo0 = domain.smallEnd(0), dhi0 = domain.bigEnd(0);
                const int dlo1 = domain.smallEnd(1), dhi1 = domain.bigEnd(1);
                const int im = (i > dlo0) ? i - 1 : i;
                const int ip = (i < dhi0) ? i + 1 : i;
                const int jm = (j > dlo1) ? j - 1 : j;
                const int jp = (j < dhi1) ? j + 1 : j;
                const Set::Scalar k_c  = eta(i, j, k)  * k0_thermal + (1.0 - eta(i, j, k))  * k1_thermal;
                const Set::Scalar k_im = eta(im, j, k) * k0_thermal + (1.0 - eta(im, j, k)) * k1_thermal;
                const Set::Scalar k_ip = eta(ip, j, k) * k0_thermal + (1.0 - eta(ip, j, k)) * k1_thermal;
                const Set::Scalar k_jm = eta(i, jm, k) * k0_thermal + (1.0 - eta(i, jm, k)) * k1_thermal;
                const Set::Scalar k_jp = eta(i, jp, k) * k0_thermal + (1.0 - eta(i, jp, k)) * k1_thermal;
                const Set::Scalar k_xlo = 0.5 * (k_im + k_c);
                const Set::Scalar k_xhi = 0.5 * (k_c + k_ip);
                const Set::Scalar k_ylo = 0.5 * (k_jm + k_c);
                const Set::Scalar k_yhi = 0.5 * (k_c + k_jp);
                E_cond = ( k_xhi * (T(ip, j, k) - T(i, j, k)) - k_xlo * (T(i, j, k) - T(im, j, k)) ) / (DX[0] * DX[0])
                       + ( k_yhi * (T(i, jp, k) - T(i, j, k)) - k_ylo * (T(i, j, k) - T(i, jm, k)) ) / (DX[1] * DX[1]);
            }
            // Fold conduction into the energy source so it takes the SAME path as latent
            // heat: the FE update below adds Source[3], and when the PP limiter is on Pass
            // D adds s_src*Source[3] -- so conduction is now governed by the source limiter
            // rather than dropped. Source[3] is reassigned fresh every RHS (Source loop),
            // so this does not accumulate across RK stages. At k=0, E_cond=0 -> unchanged.
            // NB: the source limiter scales the whole energy source by one per-cell s_src,
            // so when s_src<1 (positivity active) the conservative div(k gradT) is scaled
            // and conduction is no longer strictly energy-conserving in those cells -- the
            // intended positivity-over-conservation trade-off of the source limiter.
            Source(i, j, k, 3) += E_cond;
            E_rhs(i, j, k) = E_flux(i, j, k) + Source(i, j, k, 3);

           // ------------------------------------------------------------
           // Error Checking
           // ------------------------------------------------------------
           if ( (M_rhs(i, j, k, 0) != M_rhs(i, j, k, 0))
                or (M_rhs(i, j, k, 1) != M_rhs(i, j, k, 1))
                or (E_rhs(i, j, k) != E_rhs(i, j, k))
                or (alpharho0_rhs(i, j, k) != alpharho0_rhs(i, j, k))
                or (alpharho1_rhs(i, j, k) != alpharho1_rhs(i, j, k))
                or (rho_vap_rhs(i, j, k) != rho_vap_rhs(i, j, k)))
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
                Util::ParallelMessage(INFO, "drhoeta0/dt=", alpharho0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", alpharho1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhovap/dt=", rho_vap_rhs(i, j, k));
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

                Util::ParallelMessage(INFO, "drhoeta0/dt=", alpharho0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", alpharho1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                
            }

            // Calculate vorticity for visualization
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));
        });
    }

    // ====================================================================
    // Positivity-preserving flux limiter (Hu-Adams-Shu 2013).
    // Pass A above stored the selected (Fhi) and HLL (Flo) hi-face fluxes.
    // Here we (B) build the source-inclusive low-order baseline, (C) compute
    // the per-face blend factor theta, and (D) recompute the conservative RHS
    // (and reflux fluxes) from the blended flux. Pass D also re-adds the
    // non-conservative eta advection using the SAME blended interface velocity
    // (Pass A left only the eta sources in eta_rhs when the limiter is on).
    // ====================================================================
    const bool pp_on_lev = (pp_flux_limiter != 0 && lev < (int)pp_scratch.size()
                            && pp_scratch[lev].Fhi[0]);
    if (pp_on_lev)
    {
        const Set::Scalar eps_rho_l = eps_rho;
        const Set::Scalar eps_p_l   = eps_p;
        const Set::Scalar p_cav_l   = p_cav;
        const Set::Scalar pref_l    = pref;
        const Set::Scalar small_l   = small;
        const Set::Scalar pp_factor = 2.0 * AMREX_SPACEDIM;
        // Source limiter on -> guard the per-phase partial densities in theta and
        // scale the source for positivity in Pass D. Its positivity baseline must
        // be source-free (the source is governed by s, not by theta), so override
        // the legacy source-into-baseline folding.
        const bool src_limit_on = (pp_source_limiter != 0);
        // Local copies (member access is not GPU-safe inside the device lambdas below)
        // for the Pass-D vapor-species (rho_vap) recompute.
        const bool species_on = (species_transport != 0);
        const Set::Scalar Dv_l = Dv;
        // CE variable-diffusivity model params (member access is not GPU-safe in
        // the device lambdas below; mirror the Dv_l local-copy pattern).
        const Set::Scalar rhoD_ref_l  = rhoD_ref;
        const Set::Scalar rhoD_Tref_l = rhoD_Tref;
        const Set::Scalar rhoD_exp_l  = rhoD_exp;

        // Make each cell's lo-face flux available from its neighbor's hi-face.
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            pp_scratch[lev].Fhi[d]->FillBoundary(geom[lev].periodicity());
            pp_scratch[lev].Flo[d]->FillBoundary(geom[lev].periodicity());
        }

        // ---- Pass B: B = U_low + dt*flux_div (+ dt*Source if pp_source_limit) ----
        // With pp_source_limit set, the source is folded into the positivity
        // baseline so theta also guards against the source; cleared, theta
        // limits the flux only and the source is added unlimited in Pass D.
        const Set::Scalar src_fac = src_limit_on ? 0.0 : ((pp_source_limit != 0) ? 1.0 : 0.0);
        // Seed the baseline over valid cells AND one ghost layer from the
        // already ghost-filled conserved state. FillGhost4BC (run at the top of
        // RHS) FillPatches alpharho0/alpharho1/M/E across coarse-fine boundaries
        // and applies the physical BCs, and recomputes the primitives over the
        // grown box -- so density_mf/momentum_mf/energy_per_vol_mf hold valid
        // ghosts here. The valid cells are overwritten just below with the
        // source-inclusive low-order update, and FillBoundary then overwrites
        // same-level ghosts with the neighbor's updated baseline. This leaves
        // coarse-fine and physical-domain ghosts holding a real, positive
        // neighbor state instead of zero. Without it, PPThetaCell reads an empty
        // (rho=0) baseline across every patch boundary and returns theta=0,
        // dumping spurious full-HLL dissipation on grid lines (the pp_theta=0
        // streaks seen on patch edges).
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& gbx = mfi.growntilebox(1);
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = alpharho0_mf[lev]->array(mfi);
            auto re1 = alpharho1_mf[lev]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Bb(i, j, k, 0) = rho(i, j, k);
                Bb(i, j, k, 1) = Mom(i, j, k, 0);
                Bb(i, j, k, 2) = Mom(i, j, k, 1);
                Bb(i, j, k, 3) = Ene(i, j, k);
                Bb(i, j, k, 4) = re0(i, j, k);
                Bb(i, j, k, 5) = re1(i, j, k);
            });
        }
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = alpharho0_mf[lev]->array(mfi);
            auto re1 = alpharho1_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto Src = Source_mf[lev]->array(mfi);
            auto Vap = Vap_dot_mf[lev]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar fdiv_rho = (Flx(i - 1, j, k, 0) - Flx(i, j, k, 0)) / DX[0] + (Fly(i, j - 1, k, 0) - Fly(i, j, k, 0)) / DX[1];
                Set::Scalar fdiv_mx  = (Flx(i - 1, j, k, 1) - Flx(i, j, k, 1)) / DX[0] + (Fly(i, j - 1, k, 1) - Fly(i, j, k, 1)) / DX[1];
                Set::Scalar fdiv_my  = (Flx(i - 1, j, k, 2) - Flx(i, j, k, 2)) / DX[0] + (Fly(i, j - 1, k, 2) - Fly(i, j, k, 2)) / DX[1];
                Set::Scalar fdiv_E   = (Flx(i - 1, j, k, 3) - Flx(i, j, k, 3)) / DX[0] + (Fly(i, j - 1, k, 3) - Fly(i, j, k, 3)) / DX[1];
                Bb(i, j, k, 0) = rho(i, j, k) + dt * fdiv_rho + src_fac * dt * Src(i, j, k, 0);
                Bb(i, j, k, 1) = Mom(i, j, k, 0) + dt * fdiv_mx + src_fac * dt * Src(i, j, k, 1);
                Bb(i, j, k, 2) = Mom(i, j, k, 1) + dt * fdiv_my + src_fac * dt * Src(i, j, k, 2);
                Bb(i, j, k, 3) = Ene(i, j, k) + dt * fdiv_E + src_fac * dt * Src(i, j, k, 3);

                // Per-phase low-order baselines. The partial mass flux is the
                // upwind TRUE MASS FRACTION Y_0 = alpharho0/rho times the low-order
                // mixture mass flux, so Bb[4]+Bb[5] == Bb[0] at theta=0. DONOR RULE
                // (Larrouturou): Y_0 upwinds by the sign of the LOW-ORDER mass flux
                // it multiplies -- the subsonic HLL flux's diffusive S_L*S_R*drho
                // term can oppose u_interface at a gas-liquid face, and an
                // u_interface donor then makes the theta=0 baseline itself drain a
                // partial from a pure-other-phase cell (negative -> floored -> mass
                // leak the theta limiter cannot catch). The source is folded in only
                // when the source limiter is off (src_fac); otherwise s guards it.
                // (Y_0, not eta: alpharho_k are the true partial densities.)
                Set::Scalar Yf_xlo = (Flx(i - 1, j, k, 0) > 0.0) ? re0(i - 1, j, k) / std::max(rho(i - 1, j, k), small_l) : re0(i, j, k) / std::max(rho(i, j, k), small_l);
                Set::Scalar Yf_xhi = (Flx(i, j, k, 0)     > 0.0) ? re0(i, j, k)     / std::max(rho(i, j, k), small_l)     : re0(i + 1, j, k) / std::max(rho(i + 1, j, k), small_l);
                Set::Scalar Yf_ylo = (Fly(i, j - 1, k, 0) > 0.0) ? re0(i, j - 1, k) / std::max(rho(i, j - 1, k), small_l) : re0(i, j, k) / std::max(rho(i, j, k), small_l);
                Set::Scalar Yf_yhi = (Fly(i, j, k, 0)     > 0.0) ? re0(i, j, k)     / std::max(rho(i, j, k), small_l)     : re0(i, j + 1, k) / std::max(rho(i, j + 1, k), small_l);
                Set::Scalar fdiv_re0 = (Yf_xlo * Flx(i - 1, j, k, 0) - Yf_xhi * Flx(i, j, k, 0)) / DX[0]
                                     + (Yf_ylo * Fly(i, j - 1, k, 0) - Yf_yhi * Fly(i, j, k, 0)) / DX[1];
                Set::Scalar fdiv_re1 = ((1.0 - Yf_xlo) * Flx(i - 1, j, k, 0) - (1.0 - Yf_xhi) * Flx(i, j, k, 0)) / DX[0]
                                     + ((1.0 - Yf_ylo) * Fly(i, j - 1, k, 0) - (1.0 - Yf_yhi) * Fly(i, j, k, 0)) / DX[1];
                Set::Scalar mdv = Vap(i, j, k, 1);
                Bb(i, j, k, 4) = re0(i, j, k) + dt * fdiv_re0 + src_fac * dt * (Src(i, j, k, 0) * eta(i, j, k)         + mdv);
                Bb(i, j, k, 5) = re1(i, j, k) + dt * fdiv_re1 + src_fac * dt * (Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv);
            });
        }
        pp_scratch[lev].Bbase->FillBoundary(geom[lev].periodicity());

        // ---- Pass C: per-face blend factor theta ----
        // Default to 1 (no blend) so any face not explicitly set -- e.g. hi-side
        // ghosts Pass D never reads -- stays high-order rather than full HLL.
        pp_scratch[lev].theta->setVal(1.0);
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            // Grow one cell on the low side so each box also fills the theta of
            // its lo-edge faces (held in the lo ghost as that ghost cell's hi
            // face). Pass D reads a cell's lo-face theta from there; across a
            // coarse-fine boundary FillBoundary has no same-level neighbor to
            // supply it, so we compute it from the now-valid baseline and the
            // lo-face fluxes Pass A wrote into the ghost. Growing only the low
            // side keeps every (i+1,j+1) neighbor read inside the 1-cell halo.
            amrex::Box bx = mfi.validbox();
            bx.growLo(0, 1);
            bx.growLo(1, 1);
            auto Fhx = pp_scratch[lev].Fhi[0]->array(mfi);
            auto Fhy = pp_scratch[lev].Fhi[1]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Bb  = pp_scratch[lev].Bbase->array(mfi);
            auto gam = gamma_mf[lev]->array(mfi);
            auto p0  = p0_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto re0 = alpharho0_mf[lev]->array(mfi);   // for the true mass fraction Y_0 = alpharho0/rho
            auto rhoc = density_mf[lev]->array(mfi);
            auto Th  = pp_scratch[lev].theta->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // x hi-face: L = (i,j), R = (i+1,j), lambda = dt/dx. The per-phase
                // guard takes dF_re0 = Y0h*Fh[0] - Y0l*Fl[0], the gas-partial flux
                // correction per unit theta, with each Y_0 donor chosen by ITS OWN
                // flux's mass sign (Larrouturou) -- the same per-face split Pass D
                // applies, shared by both cells.
                {
                    Set::Scalar dF[4] = { Fhx(i, j, k, 0) - Flx(i, j, k, 0), Fhx(i, j, k, 1) - Flx(i, j, k, 1),
                                          Fhx(i, j, k, 2) - Flx(i, j, k, 2), Fhx(i, j, k, 3) - Flx(i, j, k, 3) };
                    Set::Scalar BL[4] = { Bb(i, j, k, 0), Bb(i, j, k, 1), Bb(i, j, k, 2), Bb(i, j, k, 3) };
                    Set::Scalar BR[4] = { Bb(i + 1, j, k, 0), Bb(i + 1, j, k, 1), Bb(i + 1, j, k, 2), Bb(i + 1, j, k, 3) };
                    Set::Scalar lam = dt / DX[0];
                    Set::Scalar YL = re0(i, j, k)     / std::max(rhoc(i, j, k),     small_l);
                    Set::Scalar YR = re0(i + 1, j, k) / std::max(rhoc(i + 1, j, k), small_l);
                    Set::Scalar Yh = (Fhx(i, j, k, 0) > 0.0) ? YL : YR;
                    Set::Scalar Yl = (Flx(i, j, k, 0) > 0.0) ? YL : YR;
                    Set::Scalar dF_re0 = Yh * Fhx(i, j, k, 0) - Yl * Flx(i, j, k, 0);
                    Set::Scalar pfL = Solver::EOS::EOS::PressureFloor(eta(i, j, k),     p0(i, j, k),     eps_p_l, p_cav_l);
                    Set::Scalar pfR = Solver::EOS::EOS::PressureFloor(eta(i + 1, j, k), p0(i + 1, j, k), eps_p_l, p_cav_l);
                    Set::Scalar tL = PPThetaCell(BL, -pp_factor * lam, dF, gam(i, j, k),     p0(i, j, k),     eps_rho_l, pfL, pref_l,
                                                 src_limit_on, Bb(i, j, k, 4),     Bb(i, j, k, 5),     dF_re0, 0.0);
                    Set::Scalar tR = PPThetaCell(BR, +pp_factor * lam, dF, gam(i + 1, j, k), p0(i + 1, j, k), eps_rho_l, pfR, pref_l,
                                                 src_limit_on, Bb(i + 1, j, k, 4), Bb(i + 1, j, k, 5), dF_re0, 0.0);
                    Set::Scalar th = std::min(tL, tR);
                    Th(i, j, k, 0) = (th < 0.0) ? 0.0 : ((th > 1.0) ? 1.0 : th);
                }
                // y hi-face: L = (i,j), R = (i,j+1), lambda = dt/dy
                {
                    Set::Scalar dF[4] = { Fhy(i, j, k, 0) - Fly(i, j, k, 0), Fhy(i, j, k, 1) - Fly(i, j, k, 1),
                                          Fhy(i, j, k, 2) - Fly(i, j, k, 2), Fhy(i, j, k, 3) - Fly(i, j, k, 3) };
                    Set::Scalar BL[4] = { Bb(i, j, k, 0), Bb(i, j, k, 1), Bb(i, j, k, 2), Bb(i, j, k, 3) };
                    Set::Scalar BR[4] = { Bb(i, j + 1, k, 0), Bb(i, j + 1, k, 1), Bb(i, j + 1, k, 2), Bb(i, j + 1, k, 3) };
                    Set::Scalar lam = dt / DX[1];
                    Set::Scalar YL = re0(i, j, k)     / std::max(rhoc(i, j, k),     small_l);
                    Set::Scalar YR = re0(i, j + 1, k) / std::max(rhoc(i, j + 1, k), small_l);
                    Set::Scalar Yh = (Fhy(i, j, k, 0) > 0.0) ? YL : YR;
                    Set::Scalar Yl = (Fly(i, j, k, 0) > 0.0) ? YL : YR;
                    Set::Scalar dF_re0 = Yh * Fhy(i, j, k, 0) - Yl * Fly(i, j, k, 0);
                    Set::Scalar pfL = Solver::EOS::EOS::PressureFloor(eta(i, j, k),     p0(i, j, k),     eps_p_l, p_cav_l);
                    Set::Scalar pfR = Solver::EOS::EOS::PressureFloor(eta(i, j + 1, k), p0(i, j + 1, k), eps_p_l, p_cav_l);
                    Set::Scalar tL = PPThetaCell(BL, -pp_factor * lam, dF, gam(i, j, k),     p0(i, j, k),     eps_rho_l, pfL, pref_l,
                                                 src_limit_on, Bb(i, j, k, 4),     Bb(i, j, k, 5),     dF_re0, 0.0);
                    Set::Scalar tR = PPThetaCell(BR, +pp_factor * lam, dF, gam(i, j + 1, k), p0(i, j + 1, k), eps_rho_l, pfR, pref_l,
                                                 src_limit_on, Bb(i, j + 1, k, 4), Bb(i, j + 1, k, 5), dF_re0, 0.0);
                    Set::Scalar th = std::min(tL, tR);
                    Th(i, j, k, 1) = (th < 0.0) ? 0.0 : ((th > 1.0) ? 1.0 : th);
                }
            });
        }
        pp_scratch[lev].theta->FillBoundary(geom[lev].periodicity());

        // ---- Pass D: blended flux divergence -> conservative RHS + reflux ----
        const bool have_cc = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box& bx = mfi.validbox();
            const auto bxlo = amrex::lbound(bx);
            auto eta = eta_mf[lev]->array(mfi);
            auto Src = Source_mf[lev]->array(mfi);
            auto Vap = Vap_dot_mf[lev]->array(mfi);
            auto rho = density_mf[lev]->array(mfi);
            auto Mom = momentum_mf[lev]->array(mfi);
            auto Ene = energy_per_vol_mf[lev]->array(mfi);
            auto re0 = alpharho0_mf[lev]->array(mfi);
            auto re1 = alpharho1_mf[lev]->array(mfi);
            auto rho_vap = rho_vap_mf[lev]->array(mfi);
            auto T   = T_mf[lev]->array(mfi);   // interface/face temperature for the CE rho*D(T) coefficient
            auto rvaprhs = rho_vap_rhs_mf.array(mfi);
            auto gam = gamma_mf[lev]->array(mfi);
            auto p0  = p0_mf[lev]->array(mfi);
            auto Fhx = pp_scratch[lev].Fhi[0]->array(mfi);
            auto Fhy = pp_scratch[lev].Fhi[1]->array(mfi);
            auto Flx = pp_scratch[lev].Flo[0]->array(mfi);
            auto Fly = pp_scratch[lev].Flo[1]->array(mfi);
            auto Th  = pp_scratch[lev].theta->array(mfi);
            auto re0rhs = alpharho0_rhs_mf.array(mfi);
            auto re1rhs = alpharho1_rhs_mf.array(mfi);
            auto Mrhs   = M_rhs_mf.array(mfi);
            auto Erhs   = E_rhs_mf.array(mfi);
            auto etarhs = eta_rhs_mf.array(mfi);
            auto rho_flux = rho_flux_mf[lev]->array(mfi);
            auto M_flux   = M_flux_mf[lev]->array(mfi);
            auto E_flux   = E_flux_mf[lev]->array(mfi);
            auto pp_th_diag = pp_theta_mf[lev]->array(mfi);
            auto pp_dm_diag = pp_dmass_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y, ff_mom_x, ff_mom_y, ff_ene_x, ff_ene_y;
            if (have_cc)
            {
                ff_mass_x = cc_fluxes[lev].mass[0]->array(mfi);
                ff_mass_y = cc_fluxes[lev].mass[1]->array(mfi);
                ff_mom_x  = cc_fluxes[lev].mom[0]->array(mfi);
                ff_mom_y  = cc_fluxes[lev].mom[1]->array(mfi);
                ff_ene_x  = cc_fluxes[lev].energy[0]->array(mfi);
                ff_ene_y  = cc_fluxes[lev].energy[1]->array(mfi);
            }
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar th_xhi = Th(i, j, k, 0);
                Set::Scalar th_xlo = Th(i - 1, j, k, 0);
                Set::Scalar th_yhi = Th(i, j, k, 1);
                Set::Scalar th_ylo = Th(i, j - 1, k, 1);

                // Diagnostics: blend factor and the antidiffusive mass flux the
                // limiter removed at this cell's hi-faces (the desync the
                // unblended eta advection never sees).
                pp_th_diag(i, j, k, 0) = th_xhi;
                pp_th_diag(i, j, k, 1) = th_yhi;
                pp_dm_diag(i, j, k) = std::max((1.0 - th_xhi) * std::abs(Fhx(i, j, k, 0) - Flx(i, j, k, 0)),
                                               (1.0 - th_yhi) * std::abs(Fhy(i, j, k, 0) - Fly(i, j, k, 0)));

                // Blended face fluxes F^l + theta*(F^h - F^l), comps 0=mass,1=Mx,2=My,3=E.
                Set::Scalar Fxhi[4], Fxlo[4], Fyhi[4], Fylo[4];
                for (int c = 0; c < 4; c++)
                {
                    Fxhi[c] = Flx(i, j, k, c)     + th_xhi * (Fhx(i, j, k, c)     - Flx(i, j, k, c));
                    Fxlo[c] = Flx(i - 1, j, k, c) + th_xlo * (Fhx(i - 1, j, k, c) - Flx(i - 1, j, k, c));
                    Fyhi[c] = Fly(i, j, k, c)     + th_yhi * (Fhy(i, j, k, c)     - Fly(i, j, k, c));
                    Fylo[c] = Fly(i, j - 1, k, c) + th_ylo * (Fhy(i, j - 1, k, c) - Fly(i, j - 1, k, c));
                }

                // Donor-split (Larrouturou) partial-mass face fluxes: each component
                // flux upwinds Y_0 = alpharho0/rho by ITS OWN mass-flux sign, then
                // blends with the SAME theta as the mixture:
                //   F_re0 = Y0l*Fl[0] + theta*(Y0h*Fh[0] - Y0l*Fl[0]).
                // Per face F_re0 + F_re1 telescopes to the blended mixture flux
                // (Y0+Y1 = 1 for each flux), and at theta=0 it reduces to the
                // positivity-preserving low-order donor flux -- the admissible
                // baseline Pass B built and PPThetaCell's per-phase guard assumes.
                // (Upwinding by the high-order u_interface instead drained partials
                // from pure-other-phase cells wherever the HLL mass flux opposed
                // u_interface -- the shock-droplet floor leak.)
                auto Y0c = [&](int ii, int jj) { return re0(ii, jj, k) / std::max(rho(ii, jj, k), small_l); };
                // x hi-face (donors i | i+1)
                Set::Scalar Yh_xhi = (Fhx(i, j, k, 0) > 0.0) ? Y0c(i, j) : Y0c(i + 1, j);
                Set::Scalar Yl_xhi = (Flx(i, j, k, 0) > 0.0) ? Y0c(i, j) : Y0c(i + 1, j);
                Set::Scalar Fre0_xhi = Yl_xhi * Flx(i, j, k, 0) + th_xhi * (Yh_xhi * Fhx(i, j, k, 0) - Yl_xhi * Flx(i, j, k, 0));
                // x lo-face (donors i-1 | i)
                Set::Scalar Yh_xlo = (Fhx(i - 1, j, k, 0) > 0.0) ? Y0c(i - 1, j) : Y0c(i, j);
                Set::Scalar Yl_xlo = (Flx(i - 1, j, k, 0) > 0.0) ? Y0c(i - 1, j) : Y0c(i, j);
                Set::Scalar Fre0_xlo = Yl_xlo * Flx(i - 1, j, k, 0) + th_xlo * (Yh_xlo * Fhx(i - 1, j, k, 0) - Yl_xlo * Flx(i - 1, j, k, 0));
                // y hi-face (donors j | j+1)
                Set::Scalar Yh_yhi = (Fhy(i, j, k, 0) > 0.0) ? Y0c(i, j) : Y0c(i, j + 1);
                Set::Scalar Yl_yhi = (Fly(i, j, k, 0) > 0.0) ? Y0c(i, j) : Y0c(i, j + 1);
                Set::Scalar Fre0_yhi = Yl_yhi * Fly(i, j, k, 0) + th_yhi * (Yh_yhi * Fhy(i, j, k, 0) - Yl_yhi * Fly(i, j, k, 0));
                // y lo-face (donors j-1 | j)
                Set::Scalar Yh_ylo = (Fhy(i, j - 1, k, 0) > 0.0) ? Y0c(i, j - 1) : Y0c(i, j);
                Set::Scalar Yl_ylo = (Fly(i, j - 1, k, 0) > 0.0) ? Y0c(i, j - 1) : Y0c(i, j);
                Set::Scalar Fre0_ylo = Yl_ylo * Fly(i, j - 1, k, 0) + th_ylo * (Yh_ylo * Fhy(i, j - 1, k, 0) - Yl_ylo * Fly(i, j - 1, k, 0));

                Set::Scalar rho_div = (Fxlo[0] - Fxhi[0]) / DX[0] + (Fylo[0] - Fyhi[0]) / DX[1];
                Set::Scalar mx_div  = (Fxlo[1] - Fxhi[1]) / DX[0] + (Fylo[1] - Fyhi[1]) / DX[1];
                Set::Scalar my_div  = (Fxlo[2] - Fxhi[2]) / DX[0] + (Fylo[2] - Fyhi[2]) / DX[1];
                Set::Scalar E_div   = (Fxlo[3] - Fxhi[3]) / DX[0] + (Fylo[3] - Fyhi[3]) / DX[1];

                Set::Scalar re0_div = (Fre0_xlo - Fre0_xhi) / DX[0] + (Fre0_ylo - Fre0_yhi) / DX[1];
                // Exact mixture remainder: the per-face partial fluxes telescope, so
                // re0_div + re1_div == rho_div to machine precision by construction.
                Set::Scalar re1_div = rho_div - re0_div;

                // Volume-fraction advection with the SAME per-face blend factor as
                // the conserved fluxes: u_blend = u_HLL + theta*(u_HLLC - u_HLL).
                // Pass A deferred this term (leaving only sources in eta_rhs) when
                // the limiter is on, so eta now tracks the blended mass transport
                // instead of advecting at the unblended HLLC contact speed -- the
                // desync that floored the pressure in the stiff liquid. theta=1
                // faces recover full HLLC advection identically to the limiter-off
                // path.
                Set::Scalar ub_xhi = Flx(i, j, k, 4)     + th_xhi * (Fhx(i, j, k, 4)     - Flx(i, j, k, 4));
                Set::Scalar ub_xlo = Flx(i - 1, j, k, 4) + th_xlo * (Fhx(i - 1, j, k, 4) - Flx(i - 1, j, k, 4));
                Set::Scalar ub_yhi = Fly(i, j, k, 4)     + th_yhi * (Fhy(i, j, k, 4)     - Fly(i, j, k, 4));
                Set::Scalar ub_ylo = Fly(i, j - 1, k, 4) + th_ylo * (Fhy(i, j - 1, k, 4) - Fly(i, j - 1, k, 4));

                Set::Scalar ebf_xhi = (ub_xhi > 0.0) ? eta(i, j, k)     : eta(i + 1, j, k);
                Set::Scalar ebf_xlo = (ub_xlo > 0.0) ? eta(i - 1, j, k) : eta(i, j, k);
                Set::Scalar ebf_yhi = (ub_yhi > 0.0) ? eta(i, j, k)     : eta(i, j + 1, k);
                Set::Scalar ebf_ylo = (ub_ylo > 0.0) ? eta(i, j - 1, k) : eta(i, j, k);

                Set::Scalar div_uA_b = (ub_xhi * ebf_xhi - ub_xlo * ebf_xlo) / DX[0]
                                     + (ub_yhi * ebf_yhi - ub_ylo * ebf_ylo) / DX[1];
                Set::Scalar div_u_b  = (ub_xhi - ub_xlo) / DX[0]
                                     + (ub_yhi - ub_ylo) / DX[1];
                etarhs(i, j, k) += -div_uA_b + eta(i, j, k) * div_u_b;

                Set::Scalar mdv = Vap(i, j, k, 1); // m_dot_Vap stored in Pass A

                // Positivity-preserving SOURCE limiter. The flux-blended update is
                // already admissible (the per-phase theta guaranteed it); find the
                // largest single scale s in [0,1] so that adding dt*s*Source keeps
                // the cell admissible (mixture rho,p AND both partial densities).
                // One scalar s for every component preserves the action-reaction
                // split (mdv cancels in the mixture; total mass source = s*Src0).
                Set::Scalar s_src = 1.0;
                if (src_limit_on)
                {
                    Set::Scalar Bf[4] = { rho(i, j, k) + dt * rho_div, Mom(i, j, k, 0) + dt * mx_div,
                                          Mom(i, j, k, 1) + dt * my_div, Ene(i, j, k) + dt * E_div };
                    Set::Scalar Sv[4] = { Src(i, j, k, 0), Src(i, j, k, 1), Src(i, j, k, 2), Src(i, j, k, 3) };
                    // Mixture (rho, p): reuse PPThetaCell with scale dt, source as dF.
                    Set::Scalar pf = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0(i, j, k), eps_p_l, p_cav_l);
                    Set::Scalar s = PPThetaCell(Bf, dt, Sv, gam(i, j, k), p0(i, j, k), eps_rho_l, pf, pref_l);
                    // Per-phase partial densities (linear in s).
                    Set::Scalar re0f = re0(i, j, k) + dt * re0_div;
                    Set::Scalar re1f = re1(i, j, k) + dt * re1_div;
                    // Clamp target is 0, matching the post-update floor-at-0 (partials
                    // are legitimately exactly 0 in pure phases; a `small` target here
                    // would zero the sources across the whole single-phase field).
                    Set::Scalar src_re0 = Src(i, j, k, 0) * eta(i, j, k)         + mdv;
                    Set::Scalar src_re1 = Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv;
                    if (src_re0 < 0.0) { Set::Scalar s0 = re0f / (-dt * src_re0); if (s0 < s) s = s0; }
                    if (src_re1 < 0.0) { Set::Scalar s1 = re1f / (-dt * src_re1); if (s1 < s) s = s1; }
                    s_src = (s < 0.0) ? 0.0 : ((s > 1.0) ? 1.0 : s);
                }

                re0rhs(i, j, k) = re0_div + s_src * (Src(i, j, k, 0) * eta(i, j, k)         + mdv);
                re1rhs(i, j, k) = re1_div + s_src * (Src(i, j, k, 0) * (1.0 - eta(i, j, k)) - mdv);
                Mrhs(i, j, k, 0) = mx_div + s_src * Src(i, j, k, 1);
                Mrhs(i, j, k, 1) = my_div + s_src * Src(i, j, k, 2);
                Erhs(i, j, k)    = E_div  + s_src * Src(i, j, k, 3);

                // Vapor species under the PP limiter (Stage 3a): advect rho_vap with
                // the same composite donor-split face flux as alpharho0 (Fre0_*),
                // weighted by the upwind vapor-fraction-of-total rho_vap/rho with the
                // donor of each component flux chosen by its own mass-flux sign
                // (Larrouturou, matching the Y0 split). The evaporation source +mdv is scaled by
                // the SAME s_src as alpharho0's +mdv, so the carrier (alpharho0 - rho_vap)
                // source cancels exactly and action-reaction holds at the discrete
                // level. Diffusion is the same FE Fickian operator (clamped-neighbor
                // neumann at domain edges). This OVERWRITES the unblended rho_vap_rhs
                // the FE block wrote above (same pattern as the other conserved RHS).
                // Coarse-fine rho_vap ghosts are FillPatch'd and the advective flux is
                // refluxed (cc_fluxes mass comp 2); the only remaining approximation
                // where the limiter fires is that rho_vap carries no separate positivity
                // theta (it inherits the mixture per-face blend) -- see parse WARNING.
                if (species_on)
                {
                    const int dlo0 = domain.smallEnd(0), dhi0 = domain.bigEnd(0);
                    const int dlo1 = domain.smallEnd(1), dhi1 = domain.bigEnd(1);
                    const int im = (i > dlo0) ? i - 1 : i;
                    const int ip = (i < dhi0) ? i + 1 : i;
                    const int jm = (j > dlo1) ? j - 1 : j;
                    const int jp = (j < dhi1) ? j + 1 : j;
                    // Same donor-split composite as Fre0_* above: the vapor fraction
                    // rho_vap/rho upwinds by each flux's own mass sign, blended with
                    // the same per-face theta. fv = vapor-fraction-of-total.
                    auto fvc = [&](int ii, int jj) { return rho_vap(ii, jj, k) / std::max(rho(ii, jj, k), small_l); };
                    Set::Scalar fvh_xhi = (Fhx(i, j, k, 0) > 0.0) ? fvc(i, j) : fvc(ip, j);
                    Set::Scalar fvl_xhi = (Flx(i, j, k, 0) > 0.0) ? fvc(i, j) : fvc(ip, j);
                    Set::Scalar Frv_xhi = fvl_xhi * Flx(i, j, k, 0) + th_xhi * (fvh_xhi * Fhx(i, j, k, 0) - fvl_xhi * Flx(i, j, k, 0));
                    Set::Scalar fvh_xlo = (Fhx(i - 1, j, k, 0) > 0.0) ? fvc(im, j) : fvc(i, j);
                    Set::Scalar fvl_xlo = (Flx(i - 1, j, k, 0) > 0.0) ? fvc(im, j) : fvc(i, j);
                    Set::Scalar Frv_xlo = fvl_xlo * Flx(i - 1, j, k, 0) + th_xlo * (fvh_xlo * Fhx(i - 1, j, k, 0) - fvl_xlo * Flx(i - 1, j, k, 0));
                    Set::Scalar fvh_yhi = (Fhy(i, j, k, 0) > 0.0) ? fvc(i, j) : fvc(i, jp);
                    Set::Scalar fvl_yhi = (Fly(i, j, k, 0) > 0.0) ? fvc(i, j) : fvc(i, jp);
                    Set::Scalar Frv_yhi = fvl_yhi * Fly(i, j, k, 0) + th_yhi * (fvh_yhi * Fhy(i, j, k, 0) - fvl_yhi * Fly(i, j, k, 0));
                    Set::Scalar fvh_ylo = (Fhy(i, j - 1, k, 0) > 0.0) ? fvc(i, jm) : fvc(i, j);
                    Set::Scalar fvl_ylo = (Fly(i, j - 1, k, 0) > 0.0) ? fvc(i, jm) : fvc(i, j);
                    Set::Scalar Frv_ylo = fvl_ylo * Fly(i, j - 1, k, 0) + th_ylo * (fvh_ylo * Fhy(i, j - 1, k, 0) - fvl_ylo * Fly(i, j - 1, k, 0));
                    Set::Scalar rho_vap_div = (Frv_xlo - Frv_xhi) / DX[0]
                                            + (Frv_ylo - Frv_yhi) / DX[1];

                    // Store the hi-face rho_vap advective flux for reflux (mass comp 2),
                    // the SAME composite donor-split flux as the divergence above.
                    // (Stored here, inside the species block, where Frv_* are in scope;
                    // the alpharho0/alpharho1 comps are stored in the have_cc block below.)
                    if (have_cc)
                    {
                        ff_mass_x(i, j, k, 2) = Frv_xhi;
                        ff_mass_y(i, j, k, 2) = Frv_yhi;
                        if (i == bxlo.x) ff_mass_x(i - 1, j, k, 2) = Frv_xlo;
                        if (j == bxlo.y) ff_mass_y(i, j - 1, k, 2) = Frv_ylo;
                    }

                    Set::Scalar rho_vap_diff = 0.0;
                    if (Dv_l > 0.0 || rhoD_ref_l > 0.0)
                    {
                        // Clamp Y_v to [0,1] (NaN-safe) before differencing -- see the
                        // FE-path note above; same alpharho0-collapse stiffness/NaN guard
                        // for the PP-limiter (Pass D) species update.
                        const Set::Scalar Yv_c  = ClampYv(rho_vap(i, j, k)  / std::max(re0(i, j, k), small_l));
                        const Set::Scalar Yv_im = ClampYv(rho_vap(im, j, k) / std::max(re0(im, j, k), small_l));
                        const Set::Scalar Yv_ip = ClampYv(rho_vap(ip, j, k) / std::max(re0(ip, j, k), small_l));
                        const Set::Scalar Yv_jm = ClampYv(rho_vap(i, jm, k) / std::max(re0(i, jm, k), small_l));
                        const Set::Scalar Yv_jp = ClampYv(rho_vap(i, jp, k) / std::max(re0(i, jp, k), small_l));
                        // Face transport coefficient (mirrors the FE path): CE uses
                        // eta_face * rho*D(T_face); legacy uses alpharho0_face * Dv.
                        Set::Scalar c_xlo, c_xhi, c_ylo, c_yhi;
                        if (rhoD_ref_l > 0.0)
                        {
                            c_xlo = 0.5 * (eta(im, j, k) + eta(i, j, k)) * RhoD_CE(0.5 * (T(im, j, k) + T(i, j, k)), rhoD_ref_l, rhoD_Tref_l, rhoD_exp_l, small_l);
                            c_xhi = 0.5 * (eta(i, j, k) + eta(ip, j, k)) * RhoD_CE(0.5 * (T(i, j, k) + T(ip, j, k)), rhoD_ref_l, rhoD_Tref_l, rhoD_exp_l, small_l);
                            c_ylo = 0.5 * (eta(i, jm, k) + eta(i, j, k)) * RhoD_CE(0.5 * (T(i, jm, k) + T(i, j, k)), rhoD_ref_l, rhoD_Tref_l, rhoD_exp_l, small_l);
                            c_yhi = 0.5 * (eta(i, j, k) + eta(i, jp, k)) * RhoD_CE(0.5 * (T(i, j, k) + T(i, jp, k)), rhoD_ref_l, rhoD_Tref_l, rhoD_exp_l, small_l);
                        }
                        else
                        {
                            c_xlo = 0.5 * (re0(im, j, k) + re0(i, j, k)) * Dv_l;
                            c_xhi = 0.5 * (re0(i, j, k) + re0(ip, j, k)) * Dv_l;
                            c_ylo = 0.5 * (re0(i, jm, k) + re0(i, j, k)) * Dv_l;
                            c_yhi = 0.5 * (re0(i, j, k) + re0(i, jp, k)) * Dv_l;
                        }
                        rho_vap_diff = ( (c_xhi * (Yv_ip - Yv_c) - c_xlo * (Yv_c - Yv_im)) / (DX[0] * DX[0])
                                       + (c_yhi * (Yv_jp - Yv_c) - c_ylo * (Yv_c - Yv_jm)) / (DX[1] * DX[1]) );
                    }
                    rvaprhs(i, j, k) = rho_vap_div + s_src * mdv + rho_vap_diff;

                    // Catch a vapor-species RHS NaN at its source (the Pass-D limiter
                    // path) instead of letting it slip through to the EOS compositing,
                    // where it first surfaces as a misleading gamma=NaN. Reports the
                    // individual terms so the offending one (advection/source/diffusion)
                    // is immediately visible.
                    check4nans(time, lev, i, j, k, "ERROR IN Advance(): Pass-D vapor species RHS", {
                        {"rvaprhs", rvaprhs(i, j, k)},
                        {"rho_vap_div", rho_vap_div},
                        {"rho_vap_diff", rho_vap_diff},
                        {"mdv", mdv},
                        {"s_src", s_src},
                        {"rho_vap", rho_vap(i, j, k)},
                        {"re0", re0(i, j, k)}
                    });
                }

                rho_flux(i, j, k) = rho_div;
                M_flux(i, j, k, 0) = mx_div;
                M_flux(i, j, k, 1) = my_div;
                E_flux(i, j, k) = E_div;

                if (have_cc)
                {
                    // Per-phase reflux mass fluxes = the same composite donor-split face
                    // fluxes (Fre0_*) used for re0_div/re1_div above, so reflux stays
                    // consistent with the conserved partial-density update; the liquid
                    // comp is the mixture remainder (telescopes exactly).
                    ff_mass_x(i, j, k, 0) = Fre0_xhi;
                    ff_mass_x(i, j, k, 1) = Fxhi[0] - Fre0_xhi;
                    ff_mom_x(i, j, k, 0) = Fxhi[1];
                    ff_mom_x(i, j, k, 1) = Fxhi[2];
                    ff_ene_x(i, j, k) = Fxhi[3];

                    ff_mass_y(i, j, k, 0) = Fre0_yhi;
                    ff_mass_y(i, j, k, 1) = Fyhi[0] - Fre0_yhi;
                    ff_mom_y(i, j, k, 0) = Fyhi[1];
                    ff_mom_y(i, j, k, 1) = Fyhi[2];
                    ff_ene_y(i, j, k) = Fyhi[3];

                    if (i == bxlo.x) {
                        ff_mass_x(i - 1, j, k, 0) = Fre0_xlo;
                        ff_mass_x(i - 1, j, k, 1) = Fxlo[0] - Fre0_xlo;
                        ff_mom_x(i - 1, j, k, 0) = Fxlo[1];
                        ff_mom_x(i - 1, j, k, 1) = Fxlo[2];
                        ff_ene_x(i - 1, j, k) = Fxlo[3];
                    }
                    if (j == bxlo.y) {
                        ff_mass_y(i, j - 1, k, 0) = Fre0_ylo;
                        ff_mass_y(i, j - 1, k, 1) = Fylo[0] - Fre0_ylo;
                        ff_mom_y(i, j - 1, k, 0) = Fylo[1];
                        ff_mom_y(i, j - 1, k, 1) = Fylo[2];
                        ff_ene_y(i, j - 1, k) = Fylo[3];
                    }
                }
            });
        }
    } // end pp_on_lev

    // ============================================================================
    // Cahn-Hilliard COMPANION mass / momentum / energy fluxes
    // ============================================================================
    // The phase field eta (= gas volume fraction alpha_0) carries a CH diffusion
    // term  eta_dot_CH = div(M grad mu) = Mob*lap(mu_chem)  (added to eta_rhs above).
    // The conserved partial densities alpharho_k = alpha_k*rho_k are advected ONLY,
    // so CH moves the VOLUME fraction without moving the PHASE MASS: it shrinks
    // (1-eta) in a band cell without removing the stranded liquid mass alpharho1,
    // and the recovered rho1 = alpharho1/(1-eta) -> diverges as eta -> 1. The fix
    // (continuous): (1-eta)[d_t rho1 + div(rho1 u)] = rho1 * S_CH, singular as
    // eta->1; cancel the RHS by transporting each phase's mass+energy along the SAME
    // face CH flux J_CH whose divergence IS eta_dot_CH:
    //   J_CH(i+1/2) = Mob*(mu(i+1)-mu(i))/dx ,  S_CH = (J_CH(hi)-J_CH(lo))/dx == eta_dot_CH.
    // Because eta = alpha_0, J_CH moves GAS volume fraction:
    //   d_t alpharho0 += + div(rho0_donor * J_CH)     (gas gains volume -> gains mass)
    //   d_t alpharho1 += - div(rho1_donor * J_CH)     (liquid loses volume -> loses mass)
    //   d_t E         += + div(E0_donor * J_CH) - div(E1_donor * J_CH)   E_k = rho_k e_k + 0.5 rho_k|u|^2
    //   d_t (Mx,My)   += + div(u_face * F_rho_CH),    F_rho_CH = (rho0_donor - rho1_donor)*J_CH
    // The mixture density rho = alpharho0+alpharho1 is RECOVERED downstream, so the
    // mixture-mass companion div(F_rho_CH) is applied automatically (no separate rho
    // equation); it is flux-form, so global mass is still conserved. F_rho_CH is
    // nonzero (Lowengrub-Truskinovsky: volume-fraction diffusion at unequal
    // intrinsic densities really moves mixture mass) and that is correct.
    //
    // Donor rule (mirrors the advective Larrouturou sign(F) switch): the gas flux is
    // +rho0*J_CH (upwind rho0 by sign(J_CH)); the liquid flux is -rho1*J_CH (upwind
    // rho1 by sign(-J_CH)). Per-phase energy donors use that phase's mass-donor side.
    //
    // Sourcing: ALL inputs come from the CURRENT-STAGE, ghost-valid conserved member
    // fields (alpharho0/1, momentum, energy_per_vol, eta, mu_chem -- all filled by
    // FillGhost4BC / the mu_chem ghost fill before the main loop), recovered locally
    // and consistently, NOT from the frozen velocity_mf/pressure_mf primitives. At a
    // PHYSICAL domain face the zero-neumann mu_chem ghost gives mu(ghost)=mu(edge) so
    // J_CH=0 there automatically -- no special edge handling needed.
    //
    // Placement: a single unconditional parabolic post-pass with += into the conserved
    // RHS, so it lands identically on the FE path (main loop) and the PP-limiter path
    // (Pass D, which OVERWRITES the conserved RHS). eta_rhs already holds eta_dot_CH
    // and is NOT touched here. DEFERRED (after criteria pass): coarse-fine reflux of
    // the companion fluxes, and the thermodynamic div(mu_chem*J_CH) working term.
    if (apply_ch_companion != 0 && Mob_user != 0.0)
    {
        const Set::Scalar Mob_l   = Mob_user * epsilon * epsilon * sigma / epsilon; // == Mob in the main loop (constant)
        const Set::Scalar small_l = small;
        const Set::Scalar pref_l  = pref;
        const Set::Scalar eps_p_l = eps_p;
        const Set::Scalar p_cav_l = p_cav;
        // Below this volume fraction the donor intrinsic-density recovery is bounded
        // (matches the diagnostic recovery guard in Advance): the donor rule normally
        // reads the well-conditioned side, but guard defensively in case it does not.
        const Set::Scalar alpha_floor = 1.0e-3;
        const Solver::EOS::Tammann eos0_l = eos0;
        const Solver::EOS::Tammann eos1_l = eos1;
        const Set::Scalar g0 = eos0_l.Gamma(), pi0 = eos0_l.P0();
        const Set::Scalar g1 = eos1_l.Gamma(), pi1 = eos1_l.P0();

        // Per-cell recovered companion state: intrinsic densities, velocity, and
        // per-phase INTERNAL energy per volume rho_k*e_k = (p + gamma_k*pi_k)/(gamma_k-1)
        // (Tammann; independent of rho_k -- only the kinetic part 0.5 rho_k|u|^2 needs rho_k).
        struct CompCell { Set::Scalar rho0, rho1, ux, uy, ein0, ein1; };

        // Zalesak outflow limiter (positivity for the conservative companion). The companion
        // `+= dFr0` (Pass 2 below) can drain a band-tail cell's vanishing alpharho_k below 0
        // (the gas-mass "CFL" = J_CH*dt/(eta*dx) blows up as eta->0), and the post-advance
        // floor-at-0 then INJECTS mass -- the integrals.dat col-12 M_floor_gas drift. R0/R1 are
        // per-cell scale factors in [0,1]: Pass 1 caps each cell's net OUTGOING phase mass at
        // the post-advective budget A_k = max(alpharho_k + dt*re_k_rhs, 0) (>=0 by Pass C/D);
        // Pass 2 scales each face flux by the DRAINING cell's R (same scaled flux on both sides
        // -> the divergence still telescopes, so it stays conservative -- the floor injection is
        // removed at the source rather than papered over). Default 1.0 leaves domain-edge ghosts
        // (J_CH=0 there) and the limit-off (ch_companion_pp_limit=0) path bit-identical.
        const bool limit_on = (ch_companion_pp_limit != 0);
        amrex::MultiFab R0lim_mf(alpharho0_mf[lev]->boxArray(), alpharho0_mf[lev]->DistributionMap(), 1, 1);
        amrex::MultiFab R1lim_mf(alpharho1_mf[lev]->boxArray(), alpharho1_mf[lev]->DistributionMap(), 1, 1);
        R0lim_mf.setVal(1.0);
        R1lim_mf.setVal(1.0);

        if (limit_on)
        {
            for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                auto mu_chem = mu_chem_mf[lev]->array(mfi);
                auto alpharho0 = alpharho0_mf[lev]->array(mfi);
                auto alpharho1 = alpharho1_mf[lev]->array(mfi);
                auto M = momentum_mf[lev]->array(mfi);
                auto E = energy_per_vol_mf[lev]->array(mfi);
                auto eta = eta_mf[lev]->array(mfi);
                auto re0rhs = alpharho0_rhs_mf.array(mfi);   // advective+source (companion not yet added)
                auto re1rhs = alpharho1_rhs_mf.array(mfi);
                auto R0lim = R0lim_mf.array(mfi);
                auto R1lim = R1lim_mf.array(mfi);

                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    auto St = [=] AMREX_GPU_DEVICE (int ii, int jj) -> CompCell {
                        CompCell c;
                        const Set::Scalar a0 = std::max(alpharho0(ii, jj, k), 0.0);
                        const Set::Scalar a1 = std::max(alpharho1(ii, jj, k), 0.0);
                        const Set::Scalar rc = std::max(a0 + a1, small_l);
                        Set::Scalar e_ = std::max(0.0, std::min(1.0, eta(ii, jj, k)));
                        c.ux = M(ii, jj, k, 0) / rc;
                        c.uy = M(ii, jj, k, 1) / rc;
                        const Set::Scalar UE = E(ii, jj, k) - 0.5 * rc * (c.ux * c.ux + c.uy * c.uy);
                        const Set::Scalar p = Solver::EOS::EOS::MixedPressure(rc, UE, e_, eos0_l, eos1_l, pref_l, small_l, eps_p_l, p_cav_l);
                        c.rho0 = a0 / std::max(e_,        alpha_floor);
                        c.rho1 = a1 / std::max(1.0 - e_,  alpha_floor);
                        c.ein0 = (p + g0 * pi0) / (g0 - 1.0);
                        c.ein1 = (p + g1 * pi1) / (g1 - 1.0);
                        return c;
                    };
                    const CompCell C  = St(i, j);
                    const CompCell Xp = St(i + 1, j);
                    const CompCell Xm = St(i - 1, j);
                    const CompCell Yp = St(i, j + 1);
                    const CompCell Ym = St(i, j - 1);

                    const Set::Scalar Jxh = Mob_l * (mu_chem(i + 1, j, k) - mu_chem(i, j, k)) / DX[0];
                    const Set::Scalar Jxl = Mob_l * (mu_chem(i, j, k) - mu_chem(i - 1, j, k)) / DX[0];
                    const Set::Scalar Jyh = Mob_l * (mu_chem(i, j + 1, k) - mu_chem(i, j, k)) / DX[1];
                    const Set::Scalar Jyl = Mob_l * (mu_chem(i, j, k) - mu_chem(i, j - 1, k)) / DX[1];

                    // Unscaled per-face phase-mass fluxes (gas = rho0(donor)*J, liquid = rho1(donor)*J).
                    auto faceMass = [=] AMREX_GPU_DEVICE (const CompCell& A, const CompCell& B, Set::Scalar J,
                                                          Set::Scalar& Fr0, Set::Scalar& Fr1) {
                        const CompCell& gd = (J > 0.0) ? A : B;
                        const CompCell& ld = (J > 0.0) ? B : A;
                        Fr0 = gd.rho0 * J;
                        Fr1 = ld.rho1 * J;
                    };
                    // Outgoing phase mass = negative part of each face's contribution to this cell.
                    // gas:    +Fr0(hi)/dx, -Fr0(lo)/dx   (re0 += +dFr0)
                    // liquid: -Fr1(hi)/dx, +Fr1(lo)/dx   (re1 += -dFr1)
                    Set::Scalar f0, f1, out0 = 0.0, out1 = 0.0;
                    faceMass(C,  Xp, Jxh, f0, f1); out0 += std::max(-f0, 0.0) / DX[0]; out1 += std::max( f1, 0.0) / DX[0];
                    faceMass(Xm, C,  Jxl, f0, f1); out0 += std::max( f0, 0.0) / DX[0]; out1 += std::max(-f1, 0.0) / DX[0];
                    faceMass(C,  Yp, Jyh, f0, f1); out0 += std::max(-f0, 0.0) / DX[1]; out1 += std::max( f1, 0.0) / DX[1];
                    faceMass(Ym, C,  Jyl, f0, f1); out0 += std::max( f0, 0.0) / DX[1]; out1 += std::max(-f1, 0.0) / DX[1];

                    const Set::Scalar A0 = std::max(alpharho0(i, j, k) + dt * re0rhs(i, j, k), 0.0);
                    const Set::Scalar A1 = std::max(alpharho1(i, j, k) + dt * re1rhs(i, j, k), 0.0);
                    R0lim(i, j, k) = (dt * out0 > 0.0) ? std::min(1.0, A0 / (dt * out0)) : 1.0;
                    R1lim(i, j, k) = (dt * out1 > 0.0) ? std::min(1.0, A1 / (dt * out1)) : 1.0;
                });
            }
            R0lim_mf.FillBoundary(geom[lev].periodicity());
            R1lim_mf.FillBoundary(geom[lev].periodicity());
        }

        // ---- Pass 2: apply the (limited) companion divergence to the RHS ----
        for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto mu_chem = mu_chem_mf[lev]->array(mfi);
            auto alpharho0 = alpharho0_mf[lev]->array(mfi);
            auto alpharho1 = alpharho1_mf[lev]->array(mfi);
            auto M = momentum_mf[lev]->array(mfi);
            auto E = energy_per_vol_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->array(mfi);

            auto re0rhs = alpharho0_rhs_mf.array(mfi);
            auto re1rhs = alpharho1_rhs_mf.array(mfi);
            auto Mrhs   = M_rhs_mf.array(mfi);
            auto Erhs   = E_rhs_mf.array(mfi);
            auto R0lim  = R0lim_mf.array(mfi);
            auto R1lim  = R1lim_mf.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                auto St = [=] AMREX_GPU_DEVICE (int ii, int jj) -> CompCell {
                    CompCell c;
                    const Set::Scalar a0 = std::max(alpharho0(ii, jj, k), 0.0);
                    const Set::Scalar a1 = std::max(alpharho1(ii, jj, k), 0.0);
                    const Set::Scalar rc = std::max(a0 + a1, small_l);
                    Set::Scalar e_ = std::max(0.0, std::min(1.0, eta(ii, jj, k)));
                    c.ux = M(ii, jj, k, 0) / rc;
                    c.uy = M(ii, jj, k, 1) / rc;
                    const Set::Scalar UE = E(ii, jj, k) - 0.5 * rc * (c.ux * c.ux + c.uy * c.uy);
                    const Set::Scalar p = Solver::EOS::EOS::MixedPressure(rc, UE, e_, eos0_l, eos1_l, pref_l, small_l, eps_p_l, p_cav_l);
                    c.rho0 = a0 / std::max(e_,        alpha_floor);
                    c.rho1 = a1 / std::max(1.0 - e_,  alpha_floor);
                    c.ein0 = (p + g0 * pi0) / (g0 - 1.0);
                    c.ein1 = (p + g1 * pi1) / (g1 - 1.0);
                    return c;
                };

                const CompCell C  = St(i, j);
                const CompCell Xp = St(i + 1, j);
                const CompCell Xm = St(i - 1, j);
                const CompCell Yp = St(i, j + 1);
                const CompCell Ym = St(i, j - 1);

                // Face CH fluxes J_CH = Mob*(mu_nbr - mu_cell)/dx (hi-face between
                // this cell and its +neighbor; lo-face between -neighbor and this cell).
                const Set::Scalar Jxh = Mob_l * (mu_chem(i + 1, j, k) - mu_chem(i, j, k)) / DX[0];
                const Set::Scalar Jxl = Mob_l * (mu_chem(i, j, k) - mu_chem(i - 1, j, k)) / DX[0];
                const Set::Scalar Jyh = Mob_l * (mu_chem(i, j + 1, k) - mu_chem(i, j, k)) / DX[1];
                const Set::Scalar Jyl = Mob_l * (mu_chem(i, j, k) - mu_chem(i, j - 1, k)) / DX[1];

                // Per-face donor-weighted companion fluxes. Gas donor = sign(J) side;
                // liquid donor = sign(-J) side. F_rho = gas_mass - liquid_mass; momentum
                // carries F_rho at the face-averaged velocity. r0f/r1f are the Zalesak
                // factors of the DRAINING cell for each phase (1.0 when limiting is off);
                // each phase's mass+energy+its momentum share scale together, so the per-face
                // flux stays a single conservative quantity shared by both adjacent cells.
                auto faceFlux = [=] AMREX_GPU_DEVICE (const CompCell& A, const CompCell& B, Set::Scalar J,
                                                      Set::Scalar r0f, Set::Scalar r1f,
                                                      Set::Scalar& Fr0, Set::Scalar& Fr1,
                                                      Set::Scalar& FE0, Set::Scalar& FE1,
                                                      Set::Scalar& FMx, Set::Scalar& FMy) {
                    // A = this cell's "lower-index" side, B = "upper-index" side of the face.
                    const CompCell& gd = (J > 0.0) ? A : B;   // gas donor (flux +rho0*J)
                    const CompCell& ld = (J > 0.0) ? B : A;   // liquid donor (flux -rho1*J)
                    Fr0 = r0f * gd.rho0 * J;
                    Fr1 = r1f * ld.rho1 * J;
                    FE0 = r0f * (gd.ein0 + 0.5 * gd.rho0 * (gd.ux * gd.ux + gd.uy * gd.uy)) * J;
                    FE1 = r1f * (ld.ein1 + 0.5 * ld.rho1 * (ld.ux * ld.ux + ld.uy * ld.uy)) * J;
                    const Set::Scalar Frho = Fr0 - Fr1;       // F_rho_CH at this face (limited)
                    const Set::Scalar uxf = 0.5 * (A.ux + B.ux);
                    const Set::Scalar uyf = 0.5 * (A.uy + B.uy);
                    FMx = uxf * Frho;
                    FMy = uyf * Frho;
                };

                // Limiter factor of the DRAINING cell per face and phase. Since rho_k >= 0,
                // sign(Fr0)=sign(Fr1)=sign(J): the gas flux drains the sign(J) side and the
                // liquid flux drains the opposite side (mass swap), so for a face (L|U) the
                // gas factor is R0(U) if J>0 else R0(L), and the liquid factor is R1(L) if
                // J>0 else R1(U). 1.0 when limiting is off (faceFlux then reproduces the
                // unlimited companion bit-for-bit).
                const Set::Scalar r0_xh = limit_on ? ((Jxh > 0.0) ? R0lim(i + 1, j, k) : R0lim(i, j, k)) : 1.0;
                const Set::Scalar r1_xh = limit_on ? ((Jxh > 0.0) ? R1lim(i, j, k) : R1lim(i + 1, j, k)) : 1.0;
                const Set::Scalar r0_xl = limit_on ? ((Jxl > 0.0) ? R0lim(i, j, k) : R0lim(i - 1, j, k)) : 1.0;
                const Set::Scalar r1_xl = limit_on ? ((Jxl > 0.0) ? R1lim(i - 1, j, k) : R1lim(i, j, k)) : 1.0;
                const Set::Scalar r0_yh = limit_on ? ((Jyh > 0.0) ? R0lim(i, j + 1, k) : R0lim(i, j, k)) : 1.0;
                const Set::Scalar r1_yh = limit_on ? ((Jyh > 0.0) ? R1lim(i, j, k) : R1lim(i, j + 1, k)) : 1.0;
                const Set::Scalar r0_yl = limit_on ? ((Jyl > 0.0) ? R0lim(i, j, k) : R0lim(i, j - 1, k)) : 1.0;
                const Set::Scalar r1_yl = limit_on ? ((Jyl > 0.0) ? R1lim(i, j - 1, k) : R1lim(i, j, k)) : 1.0;

                Set::Scalar Fr0_xh, Fr1_xh, FE0_xh, FE1_xh, FMx_xh, FMy_xh;
                Set::Scalar Fr0_xl, Fr1_xl, FE0_xl, FE1_xl, FMx_xl, FMy_xl;
                Set::Scalar Fr0_yh, Fr1_yh, FE0_yh, FE1_yh, FMx_yh, FMy_yh;
                Set::Scalar Fr0_yl, Fr1_yl, FE0_yl, FE1_yl, FMx_yl, FMy_yl;
                faceFlux(C,  Xp, Jxh, r0_xh, r1_xh, Fr0_xh, Fr1_xh, FE0_xh, FE1_xh, FMx_xh, FMy_xh);
                faceFlux(Xm, C,  Jxl, r0_xl, r1_xl, Fr0_xl, Fr1_xl, FE0_xl, FE1_xl, FMx_xl, FMy_xl);
                faceFlux(C,  Yp, Jyh, r0_yh, r1_yh, Fr0_yh, Fr1_yh, FE0_yh, FE1_yh, FMx_yh, FMy_yh);
                faceFlux(Ym, C,  Jyl, r0_yl, r1_yl, Fr0_yl, Fr1_yl, FE0_yl, FE1_yl, FMx_yl, FMy_yl);

                // Divergences (hi - lo)/dx, matching S_CH = (J_hi - J_lo)/dx.
                const Set::Scalar dFr0 = (Fr0_xh - Fr0_xl) / DX[0] + (Fr0_yh - Fr0_yl) / DX[1];
                const Set::Scalar dFr1 = (Fr1_xh - Fr1_xl) / DX[0] + (Fr1_yh - Fr1_yl) / DX[1];
                const Set::Scalar dFE0 = (FE0_xh - FE0_xl) / DX[0] + (FE0_yh - FE0_yl) / DX[1];
                const Set::Scalar dFE1 = (FE1_xh - FE1_xl) / DX[0] + (FE1_yh - FE1_yl) / DX[1];
                const Set::Scalar dFMx = (FMx_xh - FMx_xl) / DX[0] + (FMx_yh - FMx_yl) / DX[1];
                const Set::Scalar dFMy = (FMy_xh - FMy_xl) / DX[0] + (FMy_yh - FMy_yl) / DX[1];

                re0rhs(i, j, k)  += dFr0;        // gas gains mass where it gains volume
                re1rhs(i, j, k)  += -dFr1;       // liquid loses mass where it loses volume
                Erhs(i, j, k)    += dFE0 - dFE1;
                Mrhs(i, j, k, 0) += dFMx;
                Mrhs(i, j, k, 1) += dFMy;

                check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): CH companion flux", {
                    {"dFr0", dFr0}, {"dFr1", dFr1}, {"dFE0", dFE0}, {"dFE1", dFE1},
                    {"dFMx", dFMx}, {"dFMy", dFMy}, {"Jxh", Jxh}, {"Jyh", Jyh}
                });
            });
        }
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
    std::swap(eta_old_mf[lev], eta_mf[lev]);
    std::swap(alpharho0_old_mf[lev], alpharho0_mf[lev]);
    std::swap(alpharho1_old_mf[lev], alpharho1_mf[lev]);
    std::swap(rho_vap_old_mf[lev], rho_vap_mf[lev]);

    // ------------------------------------------------------------
    // Time Integration
    // ------------------------------------------------------------

    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*alpharho0_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*alpharho1_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*eta_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_vap_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*alpharho0_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*alpharho1_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 2);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*eta_old_mf[lev].get(),            amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_vap_old_mf[lev].get(),        amrex::MakeType::make_alias, 0, 1);

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        // rhs_mf:      [0]=alpharho0_rhs, [1]=alpharho1_rhs, [2]=M_rhs, [3]=E_rhs, [4]=eta_rhs, [5]=rho_vap_rhs
        // solution_mf: [0]=alpharho0,     [1]=alpharho1,     [2]=M,     [3]=E,     [4]=eta,     [5]=rho_vap
        RHS(lev, time, dt,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], rhs_mf[4], rhs_mf[5],
            solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3], solution_mf[4], solution_mf[5]);
    });

    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        // Copy stage data to working arrays
        amrex::MultiFab::Copy(*alpharho0_mf[lev],       stage_mf[0], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*alpharho1_mf[lev],       stage_mf[1], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*momentum_mf[lev],       stage_mf[2], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], stage_mf[3], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*eta_mf[lev],            stage_mf[4], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*rho_vap_mf[lev],        stage_mf[5], 0, 0, 1,              nghost);

        // Clamp eta in domain prior to ghost fill (state can drift slightly outside [0,1])
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            });
        }

        // Fill all ghost cells
        FillGhost4BC(lev, time);

        // Copy back
        amrex::MultiFab::Copy(stage_mf[0], *alpharho0_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[1], *alpharho1_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[2], *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(stage_mf[3], *energy_per_vol_mf[lev], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[4], *eta_mf[lev],            0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[5], *rho_vap_mf[lev],        0, 0, 1,              nghost);

    });


    // Zero face-flux MultiFabs so reflux accumulation starts clean each step.
    if (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0])
    {
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            cc_fluxes[lev].mass[d]->setVal(0.0);
            cc_fluxes[lev].mom[d]->setVal(0.0);
            cc_fluxes[lev].energy[d]->setVal(0.0);
        }
    }

    timeintegrator.advance(solution_old, solution_new, time, dt);

    // ------------------------------------------------------------------
    // Feed FluxRegister for reflux at coarse-fine boundaries.
    // Convert cell-centered hi-face fluxes to face-centered MultiFabs
    // required by CrseInit/FineAdd, then accumulate.
    // ------------------------------------------------------------------
    bool need_crse = (lev < finest_level
                      && lev + 1 < (int)flux_reg.size() && flux_reg[lev + 1]
                      && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
    bool need_fine = (lev > 0
                      && lev < (int)flux_reg.size() && flux_reg[lev]
                      && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);

    if (need_crse || need_fine)
    {
        if (need_crse)
            flux_reg[lev + 1]->setVal(0.0);

        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            // FillBoundary so ghost cells have neighbor's hi-face flux
            // (needed for the lo-face of each box in the conversion below).
            cc_fluxes[lev].mass[d]->FillBoundary(geom[lev].periodicity());
            cc_fluxes[lev].mom[d]->FillBoundary(geom[lev].periodicity());
            cc_fluxes[lev].energy[d]->FillBoundary(geom[lev].periodicity());

            // Build temporary face-centered MultiFabs for FluxRegister.
            // Use cc_fluxes' own BA/dmap to stay consistent.
            amrex::BoxArray face_ba = cc_fluxes[lev].mass[d]->boxArray();
            face_ba.surroundingNodes(d);
            const amrex::DistributionMapping& dm_cc = cc_fluxes[lev].mass[d]->DistributionMap();
            amrex::MultiFab face_mass(face_ba, dm_cc, 3, 0); // alpharho0, alpharho1, rho_vap
            amrex::MultiFab face_mom(face_ba, dm_cc, AMREX_SPACEDIM, 0);
            amrex::MultiFab face_ene(face_ba, dm_cc, 1, 0);

            // Convert: face(f,j,k) = cc(f-1,j,k) for d=0, etc.
            // The hi-face flux of cell (f-1) IS the flux at face f.
            int nlocal = (int)cc_fluxes[lev].mass[d]->local_size();
            for (int li = 0; li < nlocal; li++)
            {
                auto cc_m = cc_fluxes[lev].mass[d]->atLocalIdx(li).array();
                auto cc_p = cc_fluxes[lev].mom[d]->atLocalIdx(li).array();
                auto cc_e = cc_fluxes[lev].energy[d]->atLocalIdx(li).array();

                auto f_m = face_mass.atLocalIdx(li).array();
                auto f_p = face_mom.atLocalIdx(li).array();
                auto f_e = face_ene.atLocalIdx(li).array();

                const amrex::Box& fbx = face_mass.atLocalIdx(li).box();

                int dd = d;
                amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    int ii = (dd == 0) ? i - 1 : i;
                    int jj = (dd == 1) ? j - 1 : j;
                    f_m(i, j, k, 0) = cc_m(ii, jj, k, 0);
                    f_m(i, j, k, 1) = cc_m(ii, jj, k, 1);
                    f_m(i, j, k, 2) = cc_m(ii, jj, k, 2); // rho_vap (0 when species off)
                    for (int n = 0; n < AMREX_SPACEDIM; n++)
                        f_p(i, j, k, n) = cc_p(ii, jj, k, n);
                    f_e(i, j, k) = cc_e(ii, jj, k);
                });
            }

            // Face area MultiFab for correct area-weighted flux accumulation.
            // For Cartesian grids: x-face area = dy, y-face area = dx.
            const amrex::Real* dx_lev = geom[lev].CellSize();
            amrex::Real face_area = (d == 0) ? dx_lev[1] : dx_lev[0];
            amrex::MultiFab area_mf(face_ba, dm_cc, 1, 0);
            area_mf.setVal(face_area);

            if (need_crse)
            {
                flux_reg[lev + 1]->CrseInit(face_mass, area_mf, d, 0, 0, 2, -dt);
                flux_reg[lev + 1]->CrseInit(face_mom, area_mf, d, 0, 2, AMREX_SPACEDIM, -dt);
                flux_reg[lev + 1]->CrseInit(face_ene, area_mf, d, 0, 2 + AMREX_SPACEDIM, 1, -dt);
                // rho_vap: face_mass comp 2 -> register comp 2+SPACEDIM+1 (last).
                if (species_transport)
                    flux_reg[lev + 1]->CrseInit(face_mass, area_mf, d, 2, 2 + AMREX_SPACEDIM + 1, 1, -dt);
            }
            if (need_fine)
            {
                flux_reg[lev]->FineAdd(face_mass, area_mf, d, 0, 0, 2, dt);
                flux_reg[lev]->FineAdd(face_mom, area_mf, d, 0, 2, AMREX_SPACEDIM, dt);
                flux_reg[lev]->FineAdd(face_ene, area_mf, d, 0, 2 + AMREX_SPACEDIM, 1, dt);
                if (species_transport)
                    flux_reg[lev]->FineAdd(face_mass, area_mf, d, 2, 2 + AMREX_SPACEDIM + 1, 1, dt);
            }
        }
    }

    // ENFORCE POSITIVITY after time advance. The deficit healed here is mass
    // created from nothing (non-conservative) -- accumulate it as a diagnostic
    // so the source/flux limiter's effect on conservation is measurable.
    // Split the floor injection by phase: alpharho0 (gas partition, ~0 in the dense liquid)
    // vs alpharho1 (liquid partition, ~0 in the light gas). The injection scales with the
    // local bulk density of the region where the partition vanishes, so for rho_liq/rho_gas
    // = R the gas-side injection should exceed the liquid-side by ~R -- a direct measure of
    // the convention's density-ratio asymmetry. (See integrals.dat cols 12/13.)
    Set::Scalar floor_mass_gas_local = 0.0;  // alpharho0 floor injection this step
    Set::Scalar floor_mass_liq_local = 0.0;  // alpharho1 floor injection this step
    const Set::Scalar small_l = small;
    for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto rho_vap  = rho_vap_mf[lev]->array(mfi);
        auto eta      = eta_mf[lev]->array(mfi);
        auto density0 = density0_mf[lev]->array(mfi);   // rho_0 output (recovered below)
        auto density1 = density1_mf[lev]->array(mfi);   // rho_1 output
        auto density0_old = density0_old_mf[lev]->array(mfi); // IC reference rho_0 (band-tail fallback)
        auto density1_old = density1_old_mf[lev]->array(mfi); // IC reference rho_1 (band-tail fallback)

        // Below this volume fraction the intrinsic-density recovery rho_k =
        // alpharho_k / alpha_k is ill-conditioned (alpha_k -> 0); freeze the
        // OUTPUT density to the IC reference instead of letting it blow up.
        const Set::Scalar alpha_recover_floor = 1.0e-3;

        amrex::ParallelFor(bx, [=, &floor_mass_gas_local, &floor_mass_liq_local] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Floor the TRUE partial densities alpharho_k = alpha_k*rho_k at 0, NOT `small`.
            // They are legitimately 0 in pure phases (alpharho1=0 in the gas, alpharho0=0 in
            // the liquid); a `small` floor there CREATES mass every step. 0 is safe: every
            // division by alpharho0 is std::max(.,small)-guarded and rho = max(alpharho0+
            // alpharho1, small) is guarded at its reconstruction; alpharho1 is never a
            // denominator. With the conservative Y0-split mass flux and no rho*eta
            // re-projection anywhere, this floor should essentially never fire.
            floor_mass_gas_local += std::max(0.0 - alpharho0(i, j, k), 0.0);
            floor_mass_liq_local += std::max(0.0 - alpharho1(i, j, k), 0.0);
            alpharho0(i, j, k) = std::max(alpharho0(i, j, k), 0.0);
            alpharho1(i, j, k) = std::max(alpharho1(i, j, k), 0.0);
            // Vapor: floor at 0 (vapor is legitimately 0 in the liquid; do not
            // create mass with a `small` floor). Clamp <= alpharho0 (Y_v<=1)
            // is deferred to the EOS-compositing step.
            rho_vap(i, j, k) = std::max(rho_vap(i, j, k), 0.0);

            // Recover the intrinsic per-phase densities rho_k = alpharho_k / alpha_k for
            // output (the live density0/density1 fields that replace the retired rho_eta
            // partition as the per-phase mass representation). alpha_0 = eta, alpha_1 = 1-eta.
            // These are DIAGNOSTIC-ONLY here: the mixture hydro/EOS reads only
            // rho = alpharho0+alpharho1 and eta, never the recovered rho_k.
            //
            // Band-tail guard: in the diffuse band tail the uncompanioned CH term leaves
            // alpharho_k unchanged while alpha_k -> 0, so the raw quotient diverges (rho1
            // ~ 1e6 spikes on the windward rim). Below alpha_recover_floor, hold the IC
            // reference density (density{0,1}_old, which retain the initial-condition
            // values -- they are not swapped/updated during the run) so plots stay bounded.
            // This is cosmetic: it does NOT remove the spurious alpharho1 rim that drives
            // the mixture pressure cascade -- the CH companion mass flux does that.
            const Set::Scalar alpha0 = eta(i, j, k);
            const Set::Scalar alpha1 = 1.0 - eta(i, j, k);
            density0(i, j, k) = (alpha0 > alpha_recover_floor)
                                ? alpharho0(i, j, k) / alpha0
                                : density0_old(i, j, k);
            density1(i, j, k) = (alpha1 > alpha_recover_floor)
                                ? alpharho1(i, j, k) / alpha1
                                : density1_old(i, j, k);
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
        Set::Patch<const Set::Scalar> alpharho0 = alpharho0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> alpharho1 = alpharho1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_vap = rho_vap_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);

            rho(i, j, k) = std::max(alpharho0(i, j, k) + alpharho1(i, j, k), small);
            eta_new(i, j, k) = std::max(0.0, std::min(1.0, eta_new(i, j, k)));

            // rho_vap is included here so a corrupted vapor field is caught on the
            // conserved state (both FE and PP-limiter paths) rather than downstream
            // in EOS compositing as a misleading gamma=NaN. (0 when species off.)
            check4nans(time, lev, i, j, k, "ERROR IN Advance(): Conservative Variable Check", {
                { "eta_new", eta_new(i, j, k) },
                { "alpharho0", alpharho0(i, j, k) },
                { "alpharho1", alpharho1(i, j, k) },
                { "rho", rho(i, j, k) },
                { "rho_vap", rho_vap(i, j, k) }
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
    Set::Scalar nu_max_local = 0.0;       // max per-cell kinematic viscosity mu_eff/rho (viscous dt)
    Set::Scalar alpha_max_local = 0.0;    // max per-cell thermal diffusivity k/(rho cv) (conduction dt)
    Set::Scalar Deff_max_local = 0.0;     // max per-cell vapor diffusivity rho*D(T)/rho_0 (species dt, CE model)
    Set::Scalar floor_energy_local = 0.0; // internal energy injected by the eps_p backstop
    Set::Scalar floor_energy_gas_local = 0.0; // ... split by phase: gas-dominated cells (eta>=0.5)
    Set::Scalar floor_energy_liq_local = 0.0; // ... liquid-dominated cells (eta<0.5)
    Set::Scalar p_preflr_min_gas_local = 1e30; // min pre-floor pressure among floored gas cells
    Set::Scalar p_preflr_min_liq_local = 1e30; // min pre-floor pressure among floored liquid cells

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
    
        Set::Patch<const Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
    
        Set::Patch<const Set::Scalar> alpharho0 = alpharho0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> alpharho1 = alpharho1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_vap  = rho_vap_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> T = T_mf.Patch(lev, mfi);   // for the CE species-diffusion dt bound
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
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
        Set::Patch<Set::Scalar> pp_efloor_ = pp_efloor_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);

        // Local EOS Copy (eos0_base = carrier; per-cell eos0_local built below)
        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
    
        amrex::ParallelFor(bx, [=, &c_max_local, &vx_max_local, &vy_max_local, &F_max_local, &rho_min_local, &nu_max_local, &alpha_max_local, &Deff_max_local, &floor_energy_local, &floor_energy_gas_local, &floor_energy_liq_local, &p_preflr_min_gas_local, &p_preflr_min_liq_local] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);
        
            Set::Vector grad_eta = Numeric::Gradient(eta_new, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta_new, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta_new, i, j, k, 0, DX);

            // Stage 3 (3a-2): composition-dependent gas EOS (carrier + vapor).
            const Solver::EOS::Tammann eos0_local = species_transport
                ? GasEOS_eff(rho_vap(i, j, k) / std::max(alpharho0(i, j, k), small), eos0_base, cv_vap, cp_vap)
                : eos0_base;

            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);

            etadot(i,j,k) = (eta_new(i,j,k) - eta(i,j,k)) / dt;
        
            v(i,j,k,0) = M(i,j,k,0) / (rho(i,j,k));
            v(i,j,k,1) = M(i,j,k,1) / (rho(i,j,k));

            // Limiting Velocity
            Set::Scalar u_limit = 1e8;
            v(i, j, k, 0) = (v(i, j, k, 0) < 0.0) ? std::max(v(i, j, k, 0), -u_limit) : std::min(v(i, j, k, 0), u_limit);
            v(i, j, k, 1) = (v(i, j, k, 1) < 0.0) ? std::max(v(i, j, k, 1), -u_limit) : std::min(v(i, j, k, 1), u_limit);
        
            KE_vol(i,j,k) = 0.5 * rho(i,j,k) * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
            KE_mas(i,j,k) = 0.5 * (v(i,j,k,0) * v(i,j,k,0) + v(i,j,k,1) * v(i,j,k,1));
        
            UE_vol(i,j,k) = E_vol(i,j,k) - KE_vol(i,j,k);

            // EOS-aware positivity backstop (final safety net behind the PP flux
            // limiter). Enforce a PHASE-DEPENDENT physical pressure floor: the gas
            // (eta -> 1) cannot tension, p >= eps_p; the stiffened liquid (eta -> 0)
            // may be pulled into tension down to a cavitation limit, p >= -p_cav
            // (PressureFloor blends the two). This replaces the old uniform p >= eps_p
            // (which clamped legitimate lee-side liquid tension to 1 Pa and injected
            // energy every step). UE_vol >= (p_floor + gamma_eff*p0_eff - pref)/(gamma_eff-1).
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
            {
                Set::Scalar p_floor  = Solver::EOS::EOS::PressureFloor(eta(i, j, k), p0_eff(i, j, k), eps_p, p_cav);
                Set::Scalar ue_floor = (p_floor + gammaf(i, j, k) * p0_eff(i, j, k) - pref) / (gammaf(i, j, k) - 1.0);
                // Diagnostic: energy injected by the floor (0 if not floored).
                pp_efloor_(i, j, k) = std::max(ue_floor - UE_vol(i, j, k), 0.0);
                floor_energy_local += pp_efloor_(i, j, k);
                // Attribute the injection (and record the unclamped pre-floor
                // pressure) to gas- vs liquid-dominated cells. UE_vol is still the
                // raw value here, so press_pre is the pressure the cell WOULD have
                // had with no floor -- negative on the liquid side = tension being
                // clipped; very negative = an upstream overshoot, not tension.
                if (pp_efloor_(i, j, k) > 0.0)
                {
                    const Set::Scalar press_pre = (gammaf(i, j, k) - 1.0) * UE_vol(i, j, k)
                                                - gammaf(i, j, k) * p0_eff(i, j, k) + pref;
                    if (eta(i, j, k) >= 0.5)
                    {
                        floor_energy_gas_local += pp_efloor_(i, j, k);
                        p_preflr_min_gas_local = std::min(p_preflr_min_gas_local, press_pre);
                    }
                    else
                    {
                        floor_energy_liq_local += pp_efloor_(i, j, k);
                        p_preflr_min_liq_local = std::min(p_preflr_min_liq_local, press_pre);
                    }
                }
                UE_vol(i, j, k) = std::max(UE_vol(i, j, k), ue_floor);
                // Heal the CONSERVED energy too, not just the derived UE/pressure.
                // Without this E_vol keeps its sub-floor (negative-internal-energy)
                // value: only the displayed pressure is clamped while the bad
                // conserved energy persists and propagates. Writing it back makes
                // the backstop a true (non-conservative) energy floor so the
                // conserved state stays consistent with the floored pressure.
                E_vol(i, j, k) = UE_vol(i, j, k) + KE_vol(i, j, k);
            }
            E_mas(i,j,k) = E_vol(i,j,k) / (rho(i,j,k) + small);
            UE_mas(i,j,k) = E_mas(i,j,k) - KE_mas(i,j,k);

            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE_vol(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);
        
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
                {"alpharho0", alpharho0(i,j,k)},
                {"alpharho1", alpharho1(i,j,k)},
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

            // Per-cell parabolic diffusivities for the dt limits. Use the SAME
            // eta-blended mu_eff and k as the viscous/conduction fluxes (mu_eff at
            // ~1709, k at ~2510) over the local rho (and rho*cv), so the stability
            // estimate tracks the true worst-case kinematic viscosity / thermal
            // diffusivity instead of pairing min-rho with max-mu (a cell that
            // exists nowhere and over-restricts dt by ~70x at small scales).
            {
                Set::Scalar eta_c  = eta_new(i, j, k);
                Set::Scalar mu_eff = eta_c * mu0 + (1.0 - eta_c) * mu1;
                nu_max_local = std::max(nu_max_local, mu_eff / (rho(i, j, k) + small));

                Set::Scalar k_eff  = eta_c * k0_thermal + (1.0 - eta_c) * k1_thermal;
                Set::Scalar cv_eff = eta_c * eos0_local.Cv() + (1.0 - eta_c) * eos1_local.Cv();
                alpha_max_local = std::max(alpha_max_local, k_eff / (rho(i, j, k) * cv_eff + small));

                // CE species diffusion dt: the scalar diffusivity acting on Y_v is
                // D_eff = rho*D(T)/rho_0 (rho_0 = alpharho0/eta the intrinsic gas
                // density). Restrict to the gas (eta_c > 1e-3) so the deep-liquid
                // alpharho0->0 limit can't spuriously inflate the bound.
                if (species_transport && rhoD_ref > 0.0 && eta_c > 1.0e-3)
                {
                    // Floor rho0 so a collapsed-alpharho0 interface cell cannot inflate
                    // Deff (and crush dt_species) past the nominal CE scale the
                    // ClampYv-bounded vapor operator actually respects. dt BOUND only.
                    Set::Scalar rho0_est = std::max(alpharho0(i, j, k) / eta_c, dt_species_rho0_floor); // intrinsic gas density
                    Set::Scalar Deff = RhoD_CE(T(i, j, k), rhoD_ref, rhoD_Tref, rhoD_exp, small) / (rho0_est + small);
                    Deff_max_local = std::max(Deff_max_local, Deff);
                }
            }
        });
    } // end Mixed Fields loop

    // Parallel Reduction
    amrex::ParallelDescriptor::ReduceRealMax(c_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vx_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(vy_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(F_max_local);
    amrex::ParallelDescriptor::ReduceRealMin(rho_min_local);
    amrex::ParallelDescriptor::ReduceRealMax(nu_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(alpha_max_local);
    amrex::ParallelDescriptor::ReduceRealMax(Deff_max_local);

    c_max = c_max_local;
    vx_max = vx_max_local;
    vy_max = vy_max_local;
    F_max = F_max_local;
    rho_min = rho_min_local;

    // Conservation diagnostic: total mass and internal energy this level created
    // out of nothing by the post-update positivity floors this step (cell-volume
    // weighted). Both should trend to ~0 as the source/flux limiter does its job;
    // a persistently large mass figure is the droplet-mass leak.
    amrex::ParallelDescriptor::ReduceRealSum(floor_mass_gas_local);
    amrex::ParallelDescriptor::ReduceRealSum(floor_mass_liq_local);
    amrex::ParallelDescriptor::ReduceRealSum(floor_energy_local);
    amrex::ParallelDescriptor::ReduceRealSum(floor_energy_gas_local);
    amrex::ParallelDescriptor::ReduceRealSum(floor_energy_liq_local);
    amrex::ParallelDescriptor::ReduceRealMin(p_preflr_min_gas_local);
    amrex::ParallelDescriptor::ReduceRealMin(p_preflr_min_liq_local);
    const Set::Scalar cell_vol = DX[0] * DX[1];
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        // Cumulative per-phase floor injection [kg] -> integrals.dat cols 12/13 (IOProcessor copy).
        m_floor_gas += floor_mass_gas_local * cell_vol;
        m_floor_liq += floor_mass_liq_local * cell_vol;
        // Cumulative internal energy injected [J] by the eps_p pressure backstop -> col 14.
        e_floor += floor_energy_local * cell_vol;
        // Phase-split injection (cols 15/16) + all-time min pre-floor pressure (cols 17/18).
        e_floor_gas += floor_energy_gas_local * cell_vol;
        e_floor_liq += floor_energy_liq_local * cell_vol;
        p_preflr_min_gas = std::min(p_preflr_min_gas, p_preflr_min_gas_local);
        p_preflr_min_liq = std::min(p_preflr_min_liq, p_preflr_min_liq_local);
        if (floor_mass_gas_local > 0.0 || floor_mass_liq_local > 0.0 || floor_energy_local > 0.0)
            Util::ParallelMessage(INFO, "[pp-floor] lev=", lev,
                                  " mass_gas=", floor_mass_gas_local * cell_vol,
                                  " mass_liq=", floor_mass_liq_local * cell_vol,
                                  " energy_injected=", floor_energy_local * cell_vol);
    }

    // Computing dt for next time step on all levels
    Set::Scalar dx_min = std::min(DX[0], DX[1]);

    Set::Scalar wave_speed = c_max + sqrt(vx_max * vx_max + vy_max * vy_max);
    Set::Scalar dt_acoustic = cfl * dx_min / (wave_speed + small);

    // Parabolic momentum-diffusion limit: dt <= cfl_v * dx^2 / nu, nu = mu/rho.
    // nu_max_local is the max per-cell kinematic viscosity (mu_eff/rho) tracked in
    // the loop above. This replaces the old rho_min/mu_max pairing, which combined
    // the global-min density (gas) with the global-max viscosity (liquid) -- a
    // "cell" that exists nowhere -- and over-restricted dt by ~70x at small scales.
    Set::Scalar dt_viscous = (nu_max_local > 0.0)
                                 ? cfl_v * dx_min * dx_min / (nu_max_local + small)
                                 : dt_acoustic;

    // Parabolic limit for vapor-species diffusion (Stage 3b-1). The scalar
    // diffusivity acting on Y_v is the coefficient/alpharho0:
    //   legacy (alpharho0*Dv): D_eff = Dv (constant), bound = dx^2/Dv;
    //   CE (eta*rho*D(T)):     D_eff = rho*D(T)/rho_0 (varies), bound = dx^2/Deff_max,
    //   with Deff_max the domain-max tracked in the loop above (hot/rarefied gas).
    // Falls back to the acoustic dt when species diffusion is off so it never
    // tightens the min.
    Set::Scalar dt_species;
    if (species_transport && rhoD_ref > 0.0)
        dt_species = (Deff_max_local > 0.0)
                         ? cfl_v * dx_min * dx_min / (Deff_max_local + small)
                         : dt_acoustic;
    else if (species_transport && Dv > 0.0)
        dt_species = cfl_v * dx_min * dx_min / (Dv + small);
    else
        dt_species = dt_acoustic;

    // Parabolic Fourier-conduction limit: dt <= cfl_v * dx^2 / alpha,
    // alpha = k/(rho*cv). alpha_max_local is the max per-cell thermal diffusivity
    // tracked in the loop above (same eta-blended k and per-cell rho*cv the
    // conduction flux uses), replacing the old rho_min*cv_min/k_max worst-case
    // pairing. Falls back to the acoustic dt when conduction is off.
    Set::Scalar dt_conduction = (alpha_max_local > 0.0)
                                    ? cfl_v * dx_min * dx_min / (alpha_max_local + small)
                                    : dt_acoustic;

    Set::Scalar a_max = F_max / (rho_min + small);
    Set::Scalar dt_force = cfl_v * sqrt(dx_min / (a_max + small));

    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    Set::Scalar dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn, dt_species, dt_conduction });
    dt_max = dt_max * 0.9;

    // Debugging to report cfl constants used. Change bool to show. Prints each
    // dt candidate separately so the binding constraint is obvious at a glance.
    if ((step_counter[lev] % 10 == 0) && true)
    {
        Util::Message(INFO, "=== CFL DIAGNOSTICS Level ", lev, " ===");
        Util::Message(INFO, "  c_max = ", c_max, " m/s");
        Util::Message(INFO, "  vx_max = ", vx_max, " m/s");
        Util::Message(INFO, "  vy_max = ", vy_max, " m/s");
        Util::Message(INFO, "  nu_max = ", nu_max_local, " m^2/s , alpha_max = ", alpha_max_local, " m^2/s");
        Util::Message(INFO, "  dt_acoustic   = ", dt_acoustic, " s");
        Util::Message(INFO, "  dt_viscous    = ", dt_viscous, " s");
        Util::Message(INFO, "  dt_conduction = ", dt_conduction, " s");
        Util::Message(INFO, "  dt_species    = ", dt_species, " s");
        Util::Message(INFO, "  dt_force      = ", dt_force, " s");
        Util::Message(INFO, "  dt_allen_cahn = ", dt_allen_cahn, " s");
        Util::Message(INFO, "  dt_max (x0.9) = ", dt_max, " s");
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

    // Zero fill fields
    ZeroDerivedScratchFields(lev);

    // Ensure vectors are large enough for this level.
    if ((int)flux_reg.size() <= lev)
        flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev)
        cc_fluxes.resize(lev + 1);
    if ((int)pp_scratch.size() <= lev)
        pp_scratch.resize(lev + 1);

    // Use BA/dmap from a registered MultiFab (same reason as Initialize).
    const amrex::BoxArray& ba_reg = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm_reg = density_mf[lev]->DistributionMap();

    // Rebuild FluxRegister on regridded level.
    if (lev > 0)
    {
        int ncomp_reflux = 2 + AMREX_SPACEDIM + 1 + 1; // rho0 + rho1 + momentum + energy + rho_vap
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba_reg, dm_reg, refRatio(lev - 1), lev, ncomp_reflux);
    }

    // Rebuild cell-centered flux storage (1 ghost for FillBoundary in Advance).
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 3, 1); // alpharho0, alpharho1, rho_vap
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, AMREX_SPACEDIM, 1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 1, 1);

        cc_fluxes[lev].mass[d]->setVal(0.0);
        cc_fluxes[lev].mom[d]->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
    }

    // Rebuild PP flux-limiter scratch (1 ghost, mirrors cc_fluxes).
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        pp_scratch[lev].Fhi[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 5, 1);
        pp_scratch[lev].Flo[d] = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 5, 1);
        pp_scratch[lev].Fhi[d]->setVal(0.0);
        pp_scratch[lev].Flo[d]->setVal(0.0);
    }
    pp_scratch[lev].Bbase = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, 6, 1); // 6 comp: rho, Mx, My, E, alpharho0, alpharho1 (must match Initialize)
    pp_scratch[lev].theta = std::make_unique<amrex::MultiFab>(ba_reg, dm_reg, AMREX_SPACEDIM, 1);
    pp_scratch[lev].Bbase->setVal(0.0);
    pp_scratch[lev].theta->setVal(0.0);

    // Apply BC
    FillGhost4BC(lev, regrid_time);

    Util::Message(INFO, "Regridding on level", lev);
}// end regrid

//void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray &a_tags, Set::Scalar time, int ngrow)
void Hydro2::TagCellsForRefinement(int lev, amrex::TagBoxArray& a_tags, Set::Scalar, int)
{
    BL_PROFILE("Integrator::Flame::TagCellsForRefinement");

    const Set::Scalar* DX = geom[lev].CellSize();
    Set::Scalar dr = sqrt(AMREX_D_TERM(DX[0] * DX[0], +DX[1] * DX[1], +DX[2] * DX[2]));
    amrex::Box domain = geom[lev].Domain();

    // Eta criterion for refinement
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& eta = (*eta_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            if (grad_eta.lpNorm<2>() * dr * 2 > eta_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
            // Also refine the low-|grad eta| band TAILS: the gradient tag misses
            // eta->0/1 (the tanh flattens), leaving a coarse-fine edge INSIDE the
            // diffuse band where the stiffened offset (1-eta)*g1*p0_1/(g1-1) ~ 7e8
            // is interpolated and HLL amplifies the jump into a patch-aligned halo.
            // Tag by value so the C-F edge only lands where the minority phase is
            // < eta_band_refinement (set >= 0.5 to disable -> gradient-only tagging).
            if (std::min(eta(i, j, k), 1.0 - eta(i, j, k)) > eta_band_refinement) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Vorticity criterion for refinement
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
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
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);
            if (grad_u.lpNorm<2>() * dr * 2 > gradu_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Divergence (shock) criterion for refinement.
    // Shocks are strongly compressive (div(u) << 0); clean material interfaces
    // and contacts keep u continuous so div(u) stays small there. A Ducros
    // switch additionally suppresses tagging in shear/vorticity-dominated
    // regions, so the gas-liquid interface no longer triggers refinement the
    // way the old pressure-gradient sensor did.
    for (amrex::MFIter mfi(*velocity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& v = (*velocity_mf[lev]).array(mfi);
        amrex::Array4<const Set::Scalar> const& a = (*a_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            Set::Matrix grad_u = Numeric::Gradient(v, i, j, k, DX, sten);

            // div(u) = trace; compressive part only.
            Set::Scalar div_u = grad_u.trace();
            Set::Scalar comp = std::max(-div_u, 0.0);

            // |curl(u)|^2 from the off-diagonal velocity gradients.
#if AMREX_SPACEDIM == 2
            Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
            Set::Scalar curl_mag2 = wz * wz;
#else
            Set::Scalar wx = grad_u(2, 1) - grad_u(1, 2);
            Set::Scalar wy = grad_u(0, 2) - grad_u(2, 0);
            Set::Scalar wz = grad_u(1, 0) - grad_u(0, 1);
            Set::Scalar curl_mag2 = wx * wx + wy * wy + wz * wz;
#endif

            // Ducros switch in [0,1]: ~1 in a clean shock, ~0 in shear.
            Set::Scalar ducros = comp / std::sqrt(div_u * div_u + curl_mag2 + small * small);

            // Dimensionless shock strength: compression across a cell relative
            // to the local sound speed, gated by the Ducros switch.
            Set::Scalar c = std::max(a(i, j, k), small);
            Set::Scalar sensor = ducros * comp * dr / c;

            if (sensor > divu_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Density criterion for refinement
    for (amrex::MFIter mfi(*density_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& rho = (*density_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
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
    amrex::MultiFab psi_mf(alpharho0_mf[lev]->boxArray(), alpharho0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab psi_reinit_mf(alpharho0_mf[lev]->boxArray(), alpharho0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab phi_sharp_mf(alpharho0_mf[lev]->boxArray(), alpharho0_mf[lev]->DistributionMap(), 1, 2);

    // ============================================================================
    // STEP 1: Transform phi to psi (Equation 6)
    // psi = epsilon * ln(phi / (1-phi))
    // Only operate on INTERIOR cells (exclude boundaries)
    // ============================================================================
    for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
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

        Set::Patch<const Set::Scalar> alpharho0 = alpharho0_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> alpharho1 = alpharho1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> psi = psi_mf.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Compute mixture density
            Set::Scalar rho = alpharho0(i, j, k) + alpharho1(i, j, k);

            // Compute volume fraction
            Set::Scalar phi = alpharho0(i, j, k) / (rho + small);

            // Clamp to avoid log(0)
            phi = std::max(1e-10, std::min(1.0 - 1e-10, phi));

            // Equation (6): psi = epsilon * ln(phi/(1-phi))
            psi(i, j, k) = epsilon * std::log(phi / (1.0 - phi));
        });
    }

    // Fill boundaries for psi using custom BC function
    FillBoundariesWithBC(lev, 0.0, energy_bc, { &psi_mf });

    // ============================================================================
    // STEP 2: Reinitialize psi to signed distance function (Equation 10)
    // d(psi)/d(tau) = S(psi) * (1 - |grad(psi)|)
    // ============================================================================
    amrex::MFIter::allowMultipleMFIters(true);
    ReinitializeSignedDistance(lev, psi_reinit_mf, psi_mf, reinit_max_iter);
    amrex::MFIter::allowMultipleMFIters(false);

    // Fill boundaries for reinitialized psi
    FillBoundariesWithBC(lev, 0.0, energy_bc, { &psi_reinit_mf });

    // ============================================================================
    // STEP 3: Transform psi back to phi_sharp (Inverse of Equation 6)
    // phi = 1 / (1 + exp(-psi/epsilon))
    // Only operate on INTERIOR cells
    // ============================================================================
    for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
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

    FillBoundariesWithBC(lev, 0.0, energy_bc, { &phi_sharp_mf });

    // ============================================================================
    // STEP 4: Density Correction with Compression Operators (Equations 15-16)
    // This is the key step that prevents density anomalies!
    // ============================================================================

    // Create working copies for iterative correction
    amrex::MultiFab alpharho0_work(alpharho0_mf[lev]->boxArray(), alpharho0_mf[lev]->DistributionMap(), 1, 2);
    amrex::MultiFab alpharho1_work(alpharho1_mf[lev]->boxArray(), alpharho1_mf[lev]->DistributionMap(), 1, 2);

    // Initialize with current values
    amrex::MultiFab::Copy(alpharho0_work, *alpharho0_mf[lev], 0, 0, 1, 2);
    amrex::MultiFab::Copy(alpharho1_work, *alpharho1_mf[lev], 0, 0, 1, 2);

    // Pseudo-timestep for density correction (Equation 20a)
    // From knowledge: dt <= 2*h^2 for well-resolved interface (epsilon >= h/2)
    Set::Scalar h = std::min(DX[0], DX[1]);
    Set::Scalar dt_compression = 0.5 * h * h;

    Set::Scalar omega_relax = 0.5;

    

    Util::Message(INFO, "Starting density correction iterations...");

    for (int density_iter = 0; density_iter < density_max_iter; density_iter++)
    {
        // Store old values for convergence check
        amrex::MultiFab alpharho0_old(alpharho0_work.boxArray(), alpharho0_work.DistributionMap(), 1, 2);
        amrex::MultiFab alpharho1_old(alpharho1_work.boxArray(), alpharho1_work.DistributionMap(), 1, 2);

        amrex::MultiFab::Copy(alpharho0_old, alpharho0_work, 0, 0, 1, 2);
        amrex::MultiFab::Copy(alpharho1_old, alpharho1_work, 0, 0, 1, 2);

        // Apply compression operators (INTERIOR CELLS ONLY)
        for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx_full = mfi.validbox();

            // Shrink box to exclude boundaries
            amrex::Box bx = bx_full;
            bx.grow(0);

            if (bx.isEmpty())
                continue;

            Set::Patch<const Set::Scalar> phi_sharp = phi_sharp_mf.array(mfi);
            Set::Patch<Set::Scalar> alpharho0 = alpharho0_work.array(mfi);
            Set::Patch<Set::Scalar> alpharho1 = alpharho1_work.array(mfi);
            Set::Patch<const Set::Scalar> alpharho0_original = alpharho0_mf[lev]->array(mfi);
            Set::Patch<const Set::Scalar> alpharho1_original = alpharho1_mf[lev]->array(mfi);

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

                // Gradient of alpharho0
                Set::Vector grad_alpharho0 = Numeric::Gradient(alpharho0, i, j, k, 0, DX);

                // n dot grad(rho_0*phi_0)
                Set::Scalar n_dot_grad_alpharho0 = n_hat.dot(grad_alpharho0);

                // Compute grad(epsilon * n dot grad(rho_0*phi_0))
                // Need n dot grad(rho_0*phi_0) at neighboring cells
                Set::Scalar n_dot_grad_alpharho0_ip = 0.0, n_dot_grad_alpharho0_im = 0.0;
                Set::Scalar n_dot_grad_alpharho0_jp = 0.0, n_dot_grad_alpharho0_jm = 0.0;

                // At i+1,j
                if (i + 1 <= domain.bigEnd(0))
                {
                    Set::Vector grad_phi_ip = Numeric::Gradient(phi_sharp, i + 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_ip = grad_phi_ip.lpNorm<2>();
                    if (grad_phi_mag_ip > 1e-10)
                    {
                        Set::Vector n_hat_ip = grad_phi_ip / grad_phi_mag_ip;
                        Set::Vector grad_alpharho0_ip = Numeric::Gradient(alpharho0, i + 1, j, k, 0, DX);
                        n_dot_grad_alpharho0_ip = n_hat_ip.dot(grad_alpharho0_ip);
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
                        Set::Vector grad_alpharho0_im = Numeric::Gradient(alpharho0, i - 1, j, k, 0, DX);
                        n_dot_grad_alpharho0_im = n_hat_im.dot(grad_alpharho0_im);
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
                        Set::Vector grad_alpharho0_jp = Numeric::Gradient(alpharho0, i, j + 1, k, 0, DX);
                        n_dot_grad_alpharho0_jp = n_hat_jp.dot(grad_alpharho0_jp);
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
                        Set::Vector grad_alpharho0_jm = Numeric::Gradient(alpharho0, i, j - 1, k, 0, DX);
                        n_dot_grad_alpharho0_jm = n_hat_jm.dot(grad_alpharho0_jm);
                    }
                }

                // grad(epsilon * n dot grad(rho_0*phi_0))
                Set::Scalar grad_term0_x = epsilon * (n_dot_grad_alpharho0_ip - n_dot_grad_alpharho0_im) / (2.0 * DX[0]);
                Set::Scalar grad_term0_y = epsilon * (n_dot_grad_alpharho0_jp - n_dot_grad_alpharho0_jm) / (2.0 * DX[1]);

                Set::Scalar term1_0 = n_hat(0) * grad_term0_x + n_hat(1) * grad_term0_y;

                // (1-2*phi) * grad(rho_0*phi_0)
                Set::Scalar term2_0 = (1.0 - 2.0 * phi_val) * n_dot_grad_alpharho0;

                // Compression operator R_l (Equation 15)
                Set::Scalar R_l = H * (term1_0 - term2_0);

                // ============================================================
                // Compression Operator for Fluid 1 (Gas) - Equation (16)
                // R_g = H * n dot [grad(epsilon * n dot grad(rho_1*phi_1))
                //                  - (1-2*phi)*grad(rho_1*phi_1)]
                // ============================================================

                // Gradient of alpharho1
                Set::Vector grad_alpharho1 = Numeric::Gradient(alpharho1, i, j, k, 0, DX);

                // n dot grad(rho_1*phi_1)
                Set::Scalar n_dot_grad_alpharho1 = n_hat.dot(grad_alpharho1);

                // Compute grad(epsilon * n dot grad(rho_1*phi_1)) at neighbors
                Set::Scalar n_dot_grad_alpharho1_ip = 0.0, n_dot_grad_alpharho1_im = 0.0;
                Set::Scalar n_dot_grad_alpharho1_jp = 0.0, n_dot_grad_alpharho1_jm = 0.0;

                // At i+1,j
                if (i + 1 <= domain.bigEnd(0))
                {
                    Set::Vector grad_phi_ip = Numeric::Gradient(phi_sharp, i + 1, j, k, 0, DX);
                    Set::Scalar grad_phi_mag_ip = grad_phi_ip.lpNorm<2>();
                    if (grad_phi_mag_ip > 1e-10)
                    {
                        Set::Vector n_hat_ip = grad_phi_ip / grad_phi_mag_ip;
                        Set::Vector grad_alpharho1_ip = Numeric::Gradient(alpharho1, i + 1, j, k, 0, DX);
                        n_dot_grad_alpharho1_ip = n_hat_ip.dot(grad_alpharho1_ip);
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
                        Set::Vector grad_alpharho1_im = Numeric::Gradient(alpharho1, i - 1, j, k, 0, DX);
                        n_dot_grad_alpharho1_im = n_hat_im.dot(grad_alpharho1_im);
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
                        Set::Vector grad_alpharho1_jp = Numeric::Gradient(alpharho1, i, j + 1, k, 0, DX);
                        n_dot_grad_alpharho1_jp = n_hat_jp.dot(grad_alpharho1_jp);
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
                        Set::Vector grad_alpharho1_jm = Numeric::Gradient(alpharho1, i, j - 1, k, 0, DX);
                        n_dot_grad_alpharho1_jm = n_hat_jm.dot(grad_alpharho1_jm);
                    }
                }

                // grad(epsilon * n dot grad(rho_1*phi_1))
                Set::Scalar grad_term1_x = epsilon * (n_dot_grad_alpharho1_ip - n_dot_grad_alpharho1_im) / (2.0 * DX[0]);
                Set::Scalar grad_term1_y = epsilon * (n_dot_grad_alpharho1_jp - n_dot_grad_alpharho1_jm) / (2.0 * DX[1]);

                Set::Scalar term1_1 = n_hat(0) * grad_term1_x + n_hat(1) * grad_term1_y;

                // (1-2*phi) * grad(rho_1*phi_1)
                Set::Scalar term2_1 = (1.0 - 2.0 * phi_val) * n_dot_grad_alpharho1;

                // Compression operator R_g (Equation 16)
                Set::Scalar R_g = H * (term1_1 - term2_1);

                // ============================================================
                // Update with relaxation (pseudo-time stepping)
                // ============================================================

                alpharho0(i, j, k) = alpharho0(i, j, k) - omega_relax * dt_compression * R_l;
                alpharho1(i, j, k) = alpharho1(i, j, k) - omega_relax * dt_compression * R_g;

                // Ensure positivity (floor at 0, not small: alpharho_k are partial
                // densities, legitimately 0 in pure phases; a small floor injects mass).
                alpharho0(i, j, k) = std::max(0.0, alpharho0(i, j, k));
                alpharho1(i, j, k) = std::max(0.0, alpharho1(i, j, k));

                // ============================================================
                // Enforce exact mass conservation
                // Total mass must equal original total mass
                // ============================================================

                Set::Scalar rho_total_original = alpharho0_original(i, j, k) + alpharho1_original(i, j, k);
                Set::Scalar rho_total_new = alpharho0(i, j, k) + alpharho1(i, j, k);

                Set::Scalar mass_error = std::abs(rho_total_new - rho_total_original);

                if (mass_error > 1e-12)
                {
                    // Renormalize to ensure exact mass conservation
                    Set::Scalar scale = rho_total_original / (rho_total_new + small);
                    alpharho0(i, j, k) *= scale;
                    alpharho1(i, j, k) *= scale;
                }
            });
        }

        // Fill boundaries after each iteration using custom BC function
        Util::ParallelMessage(INFO, "Filling Shrp Interface: Density Correction");

        FillBoundariesWithBC(lev, 0.0, density_bc, { &alpharho0_work, &alpharho1_work });

        // ========================================================================
        // Check convergence of density correction
        // ========================================================================

        amrex::MultiFab residual0(alpharho0_work.boxArray(), alpharho0_work.DistributionMap(), 1, 0);
        amrex::MultiFab residual1(alpharho1_work.boxArray(), alpharho1_work.DistributionMap(), 1, 0);

        amrex::MultiFab::Copy(residual0, alpharho0_work, 0, 0, 1, 0);
        amrex::MultiFab::Copy(residual1, alpharho1_work, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(residual0, alpharho0_old, 0, 0, 1, 0);
        amrex::MultiFab::Subtract(residual1, alpharho1_old, 0, 0, 1, 0);

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

    amrex::MultiFab::Copy(*alpharho0_mf[lev], alpharho0_work, 0, 0, 1, 0);
    amrex::MultiFab::Copy(*alpharho1_mf[lev], alpharho1_work, 0, 0, 1, 0);

    // ============================================================================
    // STEP 6: Update eta from corrected densities
    // ============================================================================
    // WARNING: This recovery (eta = alpharho0 / rho_total) is the legacy
    // mass-fraction form that the new state model deliberately abandons.
    // It is dead code while apply_sharpening=0. If sharpening is re-enabled,
    // the sharpening algorithm needs to be reworked to operate on the
    // independent volume-fraction state (eta_mf) directly rather than via
    // the conservative phase masses; otherwise it will silently overwrite
    // the volume fraction with the mass fraction every sharpening pass.
    // ============================================================================

    for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();

        Set::Patch<const Set::Scalar> alpharho0 = alpharho0_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> alpharho1 = alpharho1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar rho_total = alpharho0(i, j, k) + alpharho1(i, j, k);
            eta(i, j, k) = alpharho0(i, j, k) / (rho_total + small);

            // Simple clamping (NO cutoff transformation)
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ============================================================================
    // FILL BOUNDARIES WITH CUSTOM BC (FINAL - ONLY ONCE)
    // ============================================================================
    Util::ParallelMessage(INFO, "Filling Shrp Interface: Density Correction, COMPELTE");

    // Eta: use eta_bc (not energy_bc or density_bc, which have wrong dirichlet values)
    FillBoundariesWithBC(lev, 0.0, eta_bc, { eta_mf[lev].get() });
    // Density: fill the CONSERVED partial densities directly, then recompute the total
    // (same conservative pattern as FillGhost4BC -- no rho*eta re-projection).
    FillBoundariesWithBC(lev, 0.0, density_bc, { alpharho0_mf[lev].get(), alpharho1_mf[lev].get() });
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);
        auto rho  = density_mf[lev]->array(mfi);
        auto rho0 = alpharho0_mf[lev]->array(mfi);
        auto rho1 = alpharho1_mf[lev]->array(mfi);
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho0(i, j, k) + rho1(i, j, k), small);
        });
    }

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
// Example Usage: FillBoundaries(lev, time, {x_mf[lev].get(), y_mf[lev].get(), z_mf[lev].get()});
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
        bool err_verbose = false;
        if (err_verbose && (mf->contains_nan()))
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundaries");
            Util::Abort(INFO);
        }
    }
}

// FillBoundariesWithBC(): Fill Boundries with BC
// Example Usage: FillBoundariesWithBC(lev, time, x_bc, {x_mf[lev].get(), y_mf[lev].get(), z_mf[lev].get()});
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
        bool err_verbose = false;
        if (err_verbose && (mf->contains_nan()))
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundariesWithBC");
            Util::Abort(INFO);
        }
    }
}

// ZeroDerivedScratchFields(): Single source of truth for the list of
// derived/scratch MultiFabs that need to be zeroed at IC time and after
// every regrid
void Hydro2::ZeroDerivedScratchFields(int lev)
{
    BL_PROFILE("Integrator::Hydro2::ZeroDerivedScratchFields");
    FillBoundariesWithZero(lev, {
        Source_mf[lev].get(),               // RHS source / forcing
        Fsv_mf[lev].get(),                  // surface tension
        Fw_mf[lev].get(),                   // weight / body force
        Ldot_mf[lev].get(),                 // Lagrange / IB term
        Vap_dot_mf[lev].get(),              // vaporization tracker
        rho_flux_mf[lev].get(),             // Riemann mass flux divergence
        M_flux_mf[lev].get(),               // Riemann momentum flux divergence
        E_flux_mf[lev].get(),               // Riemann energy flux divergence
        div_tau_mf[lev].get(),              // viscous stress divergence
        hess_u_mf[lev].get(),               // velocity Hessian (debug)
        grad_eta_mf[lev].get(),             // grad(eta)
        kappas_mf[lev].get(),               // curvature
        grad_mag_grad_eta_mf[lev].get(),    // |grad(eta)| gradient
        eta_x_mf[lev].get(),               // eta gradient x
        eta_y_mf[lev].get(),               // eta gradient y
        gradmag_mf[lev].get(),             // |grad(eta)|
        nx_smoothed_mf[lev].get(),         // smoothed normal x
        ny_smoothed_mf[lev].get(),         // smoothed normal y
        kappa_SF_mf[lev].get(),            // smooth-normal curvature
        n_hat_mf[lev].get(),                // interface normal
        hess_eta_mf[lev].get(),             // Hessian of eta
        mu_chem_mf[lev].get(),              // chemical potential
        Bm_mf[lev].get(),                   // Spalding number
        Y_mf[lev].get()                     // mass fraction (diagnostic)
    });
}

// FillBoundariesWithZero(): Fill Boundries with Zero
// Example Usage: FillBoundariesWithZero(lev, { x_mf[lev].get(), y_mf[lev].get() });
void Hydro2::FillBoundariesWithZero(int lev, std::initializer_list<amrex::MultiFab *> mfs)
{
    BL_PROFILE("Integrator::Hydro2::FillBoundariesWithZero");
    (void)lev;
    for (auto *mf : mfs)
    {
        if (mf != nullptr)
        {
            mf->setVal(0.0);
        }

        // Checking
        bool err_verbose = false;
        if (err_verbose && mf != nullptr && mf->contains_nan())
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "NaNs after FillBoundariesWithZero");
            Util::Abort(INFO);
        }
    }
}


/// ============================================================================
/// FillGhost4BC(): Unified ghost cell filling for all boundary conditions
/// ============================================================================
///
/// This function handles ghost cell filling for both NSCBC and standard BCs.
/// It ensures proper ordering: compute domain quantities -> fill conservatives
/// -> compute ghost primitives, maintaining consistency between all fields.
///
/// CRITICAL ORDERING (to prevent NaN propagation):
///   1. Compute rho, eta in domain
///   2. Fill eta in ghosts (NSCBC needs this for EOS)
///   3. Compute primitives in domain (NSCBC needs these for characteristics)
///   4. Call NSCBC or standard BCs to fill conservative ghosts
///   5. Update alpharho0, alpharho1 from NSCBC-modified rho_total
///   6. Compute primitives in ghosts (using NSCBC's gamma/p if available)
///   7. Enforce consistency and check for NaN/Inf
///
/// @param lev   AMR level
/// @param time  Current simulation time
/// ============================================================================
void Hydro2::FillGhost4BC(int lev, Set::Scalar time)
{
    BL_PROFILE("Integrator::Hydro2::FillGhost4BC");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Ensure eta_bc has a valid geometry for FillBoundary calls.
    // The base Integrator calls define(geom) on registered BCs after
    // Initialize(), but eta_bc may not yet be defined on this level
    // if this is the first call. define() is idempotent.
    eta_bc->define(geom[lev]);

    // ------------------------------------------------------------
    // STEP 1: Determine BC strategy based on nghost and NSCBC flag
    // ------------------------------------------------------------
    bool use_nscbc = (nscbc_bc != nullptr) || (nscbc4_bc != nullptr);
    int effective_nghost = nghost;

    if (use_nscbc)
    {
        // NSCBC objects were already created in Parse()
        // Just verify the correct one exists for this nghost
        if (nghost == 2 && nscbc_bc == nullptr)
        {
            Util::Abort(INFO, "nghost=2 but nscbc_bc is null - check Parse()");
        }
        else if (nghost == 4 && nscbc4_bc == nullptr)
        {
            Util::Abort(INFO, "nghost=4 but nscbc4_bc is null - check Parse()");
        }

        if (nghost == 2)
        {
            Util::Message(INFO, "FillGhost4BC: Using NSCBC with 2 ghost cells");
        }
        else if (nghost == 4)
        {
            Util::Message(INFO, "FillGhost4BC: Using NSCBC4 with 4 ghost cells");
        }
    }
    else
    {
        effective_nghost = 2;
    }

    // Fill Patches for regridding
    if (lev > 0)
    {
        FillPatch(lev, time, alpharho0_mf,        *alpharho0_mf[lev],       *density_bc,   0);
        FillPatch(lev, time, alpharho1_mf,        *alpharho1_mf[lev],       *density_bc,   0);
        FillPatch(lev, time, momentum_mf,        *momentum_mf[lev],       *momentum_bc,  0);
        FillPatch(lev, time, energy_per_vol_mf,  *energy_per_vol_mf[lev], *energy_bc,    0);
        FillPatch(lev, time, eta_mf,             *eta_mf[lev],            *eta_bc,       0);
        // Vapor species: interpolate rho_vap across coarse-fine boundaries from
        // the coarse level (two-time-level FillPatch, zero-gradient physical BC).
        // The species advection/diffusion stencils read rho_vap at +/-1 across
        // coarse-fine interfaces (interior, NOT clamped), so without this the CF
        // ghosts are stale and the vapor transport is wrong at refinement edges.
        if (species_transport)
            FillPatch(lev, time, rho_vap_mf,     *rho_vap_mf[lev],        neumann_bc_1,  0);
    }

    // Same-level FillBoundary for rho_vap (ALL levels, both the NSCBC and
    // non-NSCBC paths below). The CF FillPatch above only runs for lev>0; this
    // fills box-to-box ghosts so multi-box grids (even at max_level=0) don't
    // read stale rho_vap ghosts. rho_vap is not touched elsewhere in this
    // routine, so these ghosts persist to the RHS. Physical-domain edges are
    // left to the stencil clamp (= no-flux), consistent with neumann_bc_1.
    if (species_transport)
        rho_vap_mf[lev]->FillBoundary(geom[lev].periodicity());

    // ------------------------------------------------------------
    // STEP 2: Compute total density in DOMAIN; clamp eta.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(alpharho0(i, j, k) + alpharho1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ------------------------------------------------------------
    // STEP 3: Fill Eta ghost cells
    // ------------------------------------------------------------
    // Eta must use its own BC (eta_bc), NOT energy_bc.
    // energy_bc has dirichlet values meant for the total energy field
    // (e.g. 1685653 at xlo), which would overwrite eta ghost cells with
    // nonsensical values (clamped to 1.0 or 0.0). eta_bc defaults to
    // zero-neumann: the flow carries eta into the domain via advection;
    // ghost cells should mirror the interior gradient.
    FillBoundariesWithBC(lev, time, eta_bc, {
            eta_mf[lev].get()
     });


    // ------------------------------------------------------------
    // STEP 4: Compute primitives in DOMAIN
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox(); // DOMAIN ONLY

        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);
        auto UE = UE_per_vol_mf[lev]->array(mfi);
        auto KE = KE_per_vol_mf[lev]->array(mfi);

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto rho_vap  = rho_vap_mf[lev]->array(mfi);
        // Start-of-step volume fraction eta^n for the Abgrall-Karni frozen-EOS recovery
        // (eta_old_mf is solution_old -- untouched through the RK stages, so it holds eta^n).
        auto eta_old = eta_old_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;
        const int freeze_eos = abgrall_freeze_eos; // local copy for the lambda

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Velocity
            v(i, j, k, 0) = M(i, j, k, 0) / rho(i, j, k);
            v(i, j, k, 1) = M(i, j, k, 1) / rho(i, j, k);

            // Stage 3 (3a-2): composition-dependent gas EOS (carrier + vapor).
            const Solver::EOS::Tammann eos0_local = species_transport
                ? GasEOS_eff(rho_vap(i, j, k) / std::max(alpharho0(i, j, k), small), eos0_base, cv_vap, cp_vap)
                : eos0_base;

            // Limiting Velocity
            /*
            Set::Scalar u_limit = 1e8;
            v(i, j, k, 0) = (v(i, j, k, 0) < 0.0) ? std::max(v(i, j, k, 0), -u_limit) : std::min(v(i, j, k, 0), u_limit);
            v(i, j, k, 1) = (v(i, j, k, 1) < 0.0) ? std::max(v(i, j, k, 1), -u_limit) : std::min(v(i, j, k, 1), u_limit);
            */

            // Kinetic energy
            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            // Abgrall-Karni double-flux: evaluate the mixture EOS PARAMETERS from the
            // start-of-step eta^n (eta_old) instead of the current RK-stage eta when
            // abgrall_freeze_eos is on. Freezing gamma_m,p0_eff over the step removes the
            // huge dp/deta ~ gamma_m*p0_eff (~5e8 for stiffened dodecane) coupling that
            // otherwise lets a roundoff eta perturbation blow the interface pressure up
            // (../Shock-droplet_5-Equation.tex, sec:eos-instability). UE/KE/velocity below
            // still use the true conserved state, so mass and momentum stay conservative.
            const Set::Scalar eta_eos = (freeze_eos != 0) ? eta_old(i, j, k) : eta(i, j, k);

            // Mixed EOS properties
            gamma(i, j, k) = Solver::EOS::EOS::MixedGamma(eta_eos, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta_eos, eos0_local, eos1_local);

            // Internal energy, floored to the SAME eps_p-equivalent ue_floor the
            // post-update backstop uses. The old clamp to `small` only caught
            // UE < 0; but for the stiffened liquid p = (gamma-1)UE - gamma*p0 is
            // already hugely negative for any UE below ue_floor (~7e8), so the
            // weak clamp let negative pressure into the Riemann reconstruction
            // even while the plotted (post-update) pressure read eps_p. This is
            // the pressure the solver actually consumes, so it must carry the
            // same floor or the backstop never reaches the flux computation.
            Set::Scalar p_floor = Solver::EOS::EOS::PressureFloor(eta_eos, p0_eff(i, j, k), eps_p, p_cav);
            Set::Scalar ue_floor = (p_floor + gamma(i, j, k) * p0_eff(i, j, k) - pref) / (gamma(i, j, k) - 1.0);
            UE(i, j, k) = std::max(E(i, j, k) - KE(i, j, k), ue_floor);

            // Pressure from Tammann EOS (frozen-eta params under abgrall_freeze_eos)
            press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta_eos, eos0_local, eos1_local, pref, small, eps_p, p_cav);

            // Temperature
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta_eos, eos0_local, eos1_local, pref);

            // Sound speed
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gamma(i, j, k), p0_eff(i, j, k), small);
        });
    }

    // ------------------------------------------------------------
    // STEP 5: Fill CONSERVATIVE ghost cells (rho, M, E)
    // ------------------------------------------------------------
    if (use_nscbc)
    {
        // ====================================================================
        // NSCBC PATH: Use characteristic-based boundary conditions
        // ====================================================================

        // NSCBC operates on total density, not phase densities
        // Compute rho_total in domain + ghosts
        amrex::MultiFab rho_total(alpharho0_mf[lev]->boxArray(),
                                  alpharho0_mf[lev]->DistributionMap(),
                                  1,
                                  nghost);

        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.growntilebox(nghost);
            auto rho = rho_total.array(mfi);
            auto rho0 = alpharho0_mf[lev]->array(mfi);
            auto rho1 = alpharho1_mf[lev]->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }

        // Create working copies for NSCBC (it modifies them in-place)
        amrex::MultiFab M_copy(momentum_mf[lev]->boxArray(),
                               momentum_mf[lev]->DistributionMap(),
                               AMREX_SPACEDIM,
                               nghost);
        amrex::MultiFab E_copy(energy_per_vol_mf[lev]->boxArray(),
                               energy_per_vol_mf[lev]->DistributionMap(),
                               1,
                               nghost);

        amrex::MultiFab::Copy(M_copy, *momentum_mf[lev], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(E_copy, *energy_per_vol_mf[lev], 0, 0, 1, nghost);

        // ====================================================================
        // CALL NSCBC TO FILL GHOST CELLS
        // ====================================================================
        // NSCBC will:
        //   - Read eta, gamma, p0, pressure from domain (computed above)
        //   - Compute characteristic waves from interior gradients
        //   - Modify incoming waves based on BC type
        //   - Fill rho_total, M, E in ghost cells
        //   - Write gamma, p0, pressure to ghost cells
        // ====================================================================
        if (nghost == 2 && nscbc_bc != nullptr)
        {
            nscbc_bc->FillBoundary(rho_total, M_copy, E_copy, *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev], eos0, eos1, geom[lev], time, pref);
        }
        else if (nghost == 4 && nscbc4_bc != nullptr)
        {
            nscbc4_bc->FillBoundary(rho_total, M_copy, E_copy, *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev], eos0, eos1, geom[lev], time, pref);
        }

        // Copy modified conservatives back to main arrays
        amrex::MultiFab::Copy(*momentum_mf[lev], M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_copy, 0, 0, 1, nghost);

        // ====================================================================
        // Update phase densities from NSCBC-modified rho_total
        // ====================================================================
        // NSCBC is a characteristic BC on the MIXTURE: it fills the total density
        // rho_total (and M, E) in the open-boundary ghosts. The per-phase partial
        // densities must sum to that total, so split it by the TRUE upwind mass
        // fraction Y_0 = alpharho0/(alpharho0+alpharho1) carried from the existing
        // (FillPatch'd) ghost masses -- NOT by the volume fraction eta. At a
        // pure-phase open boundary (eta -> 0 or 1, the usual case) Y_0 == eta and
        // this is exact; in the rare interface-at-boundary case it transports the
        // true masses. eta is only the degenerate fallback where both partials ~0.
        // (This replaces the retired rho*eta volume-split.)
        const Set::Scalar small_l = small;
        for (amrex::MFIter mfi(rho_total); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);

            auto rho = rho_total.array(mfi);
            auto eta = eta_mf[lev]->array(mfi);
            auto rho0 = alpharho0_mf[lev]->array(mfi);
            auto rho1 = alpharho1_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar a0 = rho0(i, j, k), a1 = rho1(i, j, k);
                const Set::Scalar Y0 = (a0 + a1 > small_l) ? a0 / (a0 + a1) : eta(i, j, k);
                rho0(i, j, k) = rho(i, j, k) * Y0;
                rho1(i, j, k) = rho(i, j, k) * (1.0 - Y0);
            });
        }

        // Update density_mf from rho_total
        amrex::MultiFab::Copy(*density_mf[lev], rho_total, 0, 0, 1, nghost);
    }
    else
    {
        // Density: apply density_bc to the TOTAL density (density_mf),
        // then partition into alpharho0/alpharho1 using eta.
        // The user-specified dirichlet value (e.g. 39.0 at xlo) is the
        // total mixture density. Applying it directly to both alpharho0
        // AND alpharho1 would double the total density in ghost cells.
        // Inter-fab exchange for phase densities still needed:
        alpharho0_mf[lev]->FillBoundary(geom[lev].periodicity());
        alpharho1_mf[lev]->FillBoundary(geom[lev].periodicity());

        // Apply the physical density BC to the CONSERVED partial densities directly
        // (alpharho0/alpharho1 = alpha_k*rho_k), NOT to the total followed by an eta
        // re-partition. For REFLECT walls each partial density reflects conservatively
        // (zero advective wall flux -> exact total-mass conservation); periodic/processor
        // edges were exchanged by the FillBoundary above and coarse-fine ghosts were
        // FillPatch'd. Re-projecting rho*eta here (the retired convention) overwrote the
        // conservatively-transported masses with the INDEPENDENT volume fraction every step
        // -- the sealed-box mass leak. The density-BC ncomp matches (1 per partial density).
        FillBoundariesWithBC(lev, time, density_bc, {
            alpharho0_mf[lev].get(), alpharho1_mf[lev].get()
        });
        // Dirichlet-total repair. density_bc prescribes the TOTAL mixture density,
        // but it was just applied to EACH partial, so a dirichlet ghost holds
        // alpharho0 = alpharho1 = rho_BC: composition Y0 = 1/2 regardless of what
        // is flowing in, and the rebuilt ghost mixture density DOUBLES. A
        // supersonic inflow then feeds half its mass into the wrong phase (the
        // shock-droplet liquid-mass growth, ~rho*u*H/2) and a rho-doubled
        // momentum/energy state (u and p far off the intended post-shock values).
        // Detect those ghosts by applying the same BC to a scratch TOTAL-density
        // fab: neumann/reflect ghosts reproduce the per-partial sum exactly (the
        // BC copies/mirrors the interior), so only dirichlet-total ghosts
        // disagree. The BC carries no composition, so split rho_BC by the eta
        // ghost (own BC, zero-neumann default) -- exact for a pure-phase inflow
        // (eta = Y0 there), an approximation only if a dirichlet boundary cuts
        // the interface (the same eta fallback the NSCBC branch uses).
        {
            amrex::MultiFab rho_bc_total(density_mf[lev]->boxArray(),
                                         density_mf[lev]->DistributionMap(), 1, nghost);
            amrex::MultiFab::Copy(rho_bc_total, *density_mf[lev], 0, 0, 1, nghost);
            FillBoundariesWithBC(lev, time, density_bc, { &rho_bc_total });
            const Set::Scalar small_l = small;
            for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &ghostbox = mfi.growntilebox(nghost);
                auto rbc  = rho_bc_total.array(mfi);
                auto rho0 = alpharho0_mf[lev]->array(mfi);
                auto rho1 = alpharho1_mf[lev]->array(mfi);
                auto eta  = eta_mf[lev]->array(mfi);
                amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    if (domain.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) return; // physical ghosts only
                    const Set::Scalar sum = rho0(i, j, k) + rho1(i, j, k);
                    if (std::abs(sum - rbc(i, j, k)) > 1.0e-10 * std::max(std::abs(rbc(i, j, k)), small_l))
                    {
                        rho0(i, j, k) = rbc(i, j, k) * eta(i, j, k);
                        rho1(i, j, k) = rbc(i, j, k) * (1.0 - eta(i, j, k));
                    }
                });
            }
        }
        // Total density ghosts = sum of the filled partial densities, so (rho, M, E) stay
        // mutually consistent at domain edges and coarse-fine boundaries (the Riemann
        // reconstruction reads UE = E - |M|^2/2rho with this rho).
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho  = density_mf[lev]->array(mfi);
            auto rho0 = alpharho0_mf[lev]->array(mfi);
            auto rho1 = alpharho1_mf[lev]->array(mfi);
            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = std::max(rho0(i, j, k) + rho1(i, j, k), small);
            });
        }
        // Momentum
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        // Energy
        FillBoundariesWithBC(lev, time, energy_bc, {
            energy_per_vol_mf[lev].get()
        });

        // Zero Gradient Fill
        if (nghost > effective_nghost)
        {
            for (amrex::MFIter mfi(*alpharho0_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &validbox = mfi.validbox();
                const amrex::Box &ghostEffbox = mfi.growntilebox(effective_nghost);
                const amrex::Box &ghostNbox = mfi.growntilebox(nghost);

                auto rho0 = alpharho0_mf[lev]->array(mfi);
                auto rho1 = alpharho1_mf[lev]->array(mfi);
                auto M = momentum_mf[lev]->array(mfi);
                auto E = energy_per_vol_mf[lev]->array(mfi);

                amrex::ParallelFor(ghostNbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // If in outer ghost layers (beyond layer 2)
                    if (!ghostEffbox.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                    {
                        // Find nearest layer-2 ghost cell and copy its value
                        int i_copy = i;
                        int j_copy = j;

                        // Clamp to layer-2 boundary
                        if (i < ghostEffbox.smallEnd(0))
                            i_copy = ghostEffbox.smallEnd(0);
                        if (i > ghostEffbox.bigEnd(0))
                            i_copy = ghostEffbox.bigEnd(0);
                        if (j < ghostEffbox.smallEnd(1))
                            j_copy = ghostEffbox.smallEnd(1);
                        if (j > ghostEffbox.bigEnd(1))
                            j_copy = ghostEffbox.bigEnd(1);

                        // Zero-gradient extrapolation (copy from layer 2)
                        rho0(i, j, k) = rho0(i_copy, j_copy, k);
                        rho1(i, j, k) = rho1(i_copy, j_copy, k);
                        M(i, j, k, 0) = M(i_copy, j_copy, k, 0);
                        M(i, j, k, 1) = M(i_copy, j_copy, k, 1);
                        E(i, j, k) = E(i_copy, j_copy, k);
                    }
                });
            }
        }
    }

    // ------------------------------------------------------------
    // STEP 6: Update total density in GHOST CELLS; clamp eta.
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(alpharho0(i, j, k) + alpharho1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ------------------------------------------------------------
    // STEP 7: Compute primitives in GHOST CELLS
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);
        auto UE = UE_per_vol_mf[lev]->array(mfi);
        auto KE = KE_per_vol_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        // Compute 
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            v(i, j, k, 0) = M(i, j, k, 0) / rho(i, j, k);
            v(i, j, k, 1) = M(i, j, k, 1) / rho(i, j, k);

            KE(i, j, k) = 0.5 * rho(i, j, k) * (v(i, j, k, 0) * v(i, j, k, 0) + v(i, j, k, 1) * v(i, j, k, 1));

            UE(i, j, k) = E(i, j, k) - KE(i, j, k);
            UE(i, j, k) = (UE(i, j, k) < 0.0) ? small : UE(i, j, k);

            //if (!(use_nscbc))
            //{
                gamma(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);
                p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(eta(i, j, k), eos0_local, eos1_local);
                press(i, j, k) = Solver::EOS::EOS::MixedPressure(rho(i, j, k), UE(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref, small, eps_p, p_cav);
            //}
            T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);
            a(i, j, k) = Solver::EOS::EOS::TammannSoundSpeed(rho(i, j, k), press(i, j, k), gamma(i, j, k), p0_eff(i, j, k), small);
        });
    }

    // ------------------------------------------------------------
    // STEP 8: Repair any remaining NaN in ALL cells (domain + ghosts)
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto T = T_mf[lev]->array(mfi);
        auto a = a_mf[lev]->array(mfi);
        auto gamma = gamma_mf[lev]->array(mfi);
        auto p0_eff = p0_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Check for NaN in any field
            bool has_nan = !std::isfinite(alpharho0(i, j, k)) || !std::isfinite(alpharho1(i, j, k)) || !std::isfinite(rho(i, j, k)) || !std::isfinite(eta(i, j, k)) || !std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1)) || !std::isfinite(E(i, j, k)) || !std::isfinite(press(i, j, k)) || !std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1)) || !std::isfinite(T(i, j, k)) || !std::isfinite(a(i, j, k)) || !std::isfinite(gamma(i, j, k)) || !std::isfinite(p0_eff(i, j, k));

            if (has_nan)
            {
                // Repair by copying from nearest valid neighbor

                // Try left neighbor (i-1)
                if (i > ghostbox.smallEnd(0))
                {
                    bool left_valid = std::isfinite(alpharho0(i - 1, j, k)) && std::isfinite(E(i - 1, j, k)) && std::isfinite(press(i - 1, j, k));
                    if (left_valid)
                    {
                        alpharho0(i, j, k) = alpharho0(i - 1, j, k);
                        alpharho1(i, j, k) = alpharho1(i - 1, j, k);
                        rho(i, j, k) = rho(i - 1, j, k);
                        eta(i, j, k) = eta(i - 1, j, k);
                        M(i, j, k, 0) = M(i - 1, j, k, 0);
                        M(i, j, k, 1) = M(i - 1, j, k, 1);
                        E(i, j, k) = E(i - 1, j, k);
                        press(i, j, k) = press(i - 1, j, k);
                        v(i, j, k, 0) = v(i - 1, j, k, 0);
                        v(i, j, k, 1) = v(i - 1, j, k, 1);
                        T(i, j, k) = T(i - 1, j, k);
                        a(i, j, k) = a(i - 1, j, k);
                        gamma(i, j, k) = gamma(i - 1, j, k);
                        p0_eff(i, j, k) = p0_eff(i - 1, j, k);
                        return; // Fixed, exit early
                    }
                }

                // Try right neighbor (i+1)
                if (i < ghostbox.bigEnd(0))
                {
                    bool right_valid = std::isfinite(alpharho0(i + 1, j, k)) && std::isfinite(E(i + 1, j, k)) && std::isfinite(press(i + 1, j, k));
                    if (right_valid)
                    {
                        alpharho0(i, j, k) = alpharho0(i + 1, j, k);
                        alpharho1(i, j, k) = alpharho1(i + 1, j, k);
                        rho(i, j, k) = rho(i + 1, j, k);
                        eta(i, j, k) = eta(i + 1, j, k);
                        M(i, j, k, 0) = M(i + 1, j, k, 0);
                        M(i, j, k, 1) = M(i + 1, j, k, 1);
                        E(i, j, k) = E(i + 1, j, k);
                        press(i, j, k) = press(i + 1, j, k);
                        v(i, j, k, 0) = v(i + 1, j, k, 0);
                        v(i, j, k, 1) = v(i + 1, j, k, 1);
                        T(i, j, k) = T(i + 1, j, k);
                        a(i, j, k) = a(i + 1, j, k);
                        gamma(i, j, k) = gamma(i + 1, j, k);
                        p0_eff(i, j, k) = p0_eff(i + 1, j, k);
                        return;
                    }
                }

                // Try bottom neighbor (j-1)
                if (j > ghostbox.smallEnd(1))
                {
                    bool bottom_valid = std::isfinite(alpharho0(i, j - 1, k)) && std::isfinite(E(i, j - 1, k)) && std::isfinite(press(i, j - 1, k));
                    if (bottom_valid)
                    {
                        alpharho0(i, j, k) = alpharho0(i, j - 1, k);
                        alpharho1(i, j, k) = alpharho1(i, j - 1, k);
                        rho(i, j, k) = rho(i, j - 1, k);
                        eta(i, j, k) = eta(i, j - 1, k);
                        M(i, j, k, 0) = M(i, j - 1, k, 0);
                        M(i, j, k, 1) = M(i, j - 1, k, 1);
                        E(i, j, k) = E(i, j - 1, k);
                        press(i, j, k) = press(i, j - 1, k);
                        v(i, j, k, 0) = v(i, j - 1, k, 0);
                        v(i, j, k, 1) = v(i, j - 1, k, 1);
                        T(i, j, k) = T(i, j - 1, k);
                        a(i, j, k) = a(i, j - 1, k);
                        gamma(i, j, k) = gamma(i, j - 1, k);
                        p0_eff(i, j, k) = p0_eff(i, j - 1, k);
                        return;
                    }
                }

                // Try top neighbor (j+1)
                if (j < ghostbox.bigEnd(1))
                {
                    bool top_valid = std::isfinite(alpharho0(i, j + 1, k)) && std::isfinite(E(i, j + 1, k)) && std::isfinite(press(i, j + 1, k));
                    if (top_valid)
                    {
                        alpharho0(i, j, k) = alpharho0(i, j + 1, k);
                        alpharho1(i, j, k) = alpharho1(i, j + 1, k);
                        rho(i, j, k) = rho(i, j + 1, k);
                        eta(i, j, k) = eta(i, j + 1, k);
                        M(i, j, k, 0) = M(i, j + 1, k, 0);
                        M(i, j, k, 1) = M(i, j + 1, k, 1);
                        E(i, j, k) = E(i, j + 1, k);
                        press(i, j, k) = press(i, j + 1, k);
                        v(i, j, k, 0) = v(i, j + 1, k, 0);
                        v(i, j, k, 1) = v(i, j + 1, k, 1);
                        T(i, j, k) = T(i, j + 1, k);
                        a(i, j, k) = a(i, j + 1, k);
                        gamma(i, j, k) = gamma(i, j + 1, k);
                        p0_eff(i, j, k) = p0_eff(i, j + 1, k);
                        return;
                    }
                }

                Util::Message(INFO, "=== NaN DETECTED IN FillGhost4BC ===");
                Util::Message(INFO, "Level: ", lev);
                Util::Message(INFO, "Time: ", time);
                Util::Message(INFO, "NSCBC mode: ", use_nscbc);
                Util::Message(INFO, "All neighbors are NaNs, unable to patch cell");
                Util::Abort(INFO, "NaN detected in FillGhost4BC - see details above");
            }
        });
    }
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto rho0 = alpharho0_mf[lev]->array(mfi);
        auto rho1 = alpharho1_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
        });
    }

    // ------------------------------------------------------------
    // STEP 9: Enforce consistency and positivity in ALL cells
    // ------------------------------------------------------------
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto alpharho0 = alpharho0_mf[lev]->array(mfi);
        auto alpharho1 = alpharho1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Enforce rho = alpharho0 + alpharho1
            Set::Scalar rho_calc = alpharho0(i, j, k) + alpharho1(i, j, k);
            rho(i, j, k) = rho_calc;

            // Enforce positivity
            rho(i, j, k) = std::max(rho(i, j, k), small);
            alpharho0(i, j, k) = std::max(alpharho0(i, j, k), small);
            alpharho1(i, j, k) = std::max(alpharho1(i, j, k), small);
            press(i, j, k) = std::max(press(i, j, k), 1.0e-6);
            E(i, j, k) = std::max(E(i, j, k), 1.0e-10);

            // Clamp eta to [0, 1]
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ========================================================================
    // STEP 10: NaN/Inf detection and reporting
    // ========================================================================
    // Check all critical fields for NaN/Inf in ghost cells.
    // If found, report location and abort with diagnostic info.
    // ========================================================================
    bool found_nan = false;
    std::string nan_details = "";

    const amrex::Box &domain_box = geom[lev].Domain();
    int domain_ilo = domain_box.smallEnd(0);
    int domain_ihi = domain_box.bigEnd(0);
    int domain_jlo = domain_box.smallEnd(1);
    int domain_jhi = domain_box.bigEnd(1);

    // Lambda to describe cell location
    auto describe_location = [&](int i, int j, int k, const amrex::Box &validbox) -> std::string {
        std::stringstream ss;

        // Distance from domain boundaries
        int dist_from_xlo = i - domain_ilo;
        int dist_from_xhi = domain_ihi - i;
        int dist_from_ylo = j - domain_jlo;
        int dist_from_yhi = domain_jhi - j;

        // Distance from valid box boundaries
        int dist_from_valid_xlo = i - validbox.smallEnd(0);
        int dist_from_valid_xhi = validbox.bigEnd(0) - i;
        int dist_from_valid_ylo = j - validbox.smallEnd(1);
        int dist_from_valid_yhi = validbox.bigEnd(1) - j;

        ss << "Cell (" << i << "," << j << "," << k << ")\n";
        ss << "  Domain: [" << domain_ilo << "," << domain_ihi << "] x ["
           << domain_jlo << "," << domain_jhi << "]\n";
        ss << "  ValidBox: [" << validbox.smallEnd(0) << "," << validbox.bigEnd(0) << "] x ["
           << validbox.smallEnd(1) << "," << validbox.bigEnd(1) << "]\n";

        // Determine location type
        bool in_domain = (i >= domain_ilo && i <= domain_ihi && j >= domain_jlo && j <= domain_jhi);
        bool in_valid = validbox.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));

        if (in_valid)
        {
            ss << "  Location: INTERIOR (valid box)\n";
        }
        else if (in_domain)
        {
            ss << "  Location: DOMAIN but outside valid box\n";
        }
        else
        {
            ss << "  Location: GHOST CELL\n";

            // Determine which boundary
            if (i < domain_ilo)
            {
                ss << "    X-LO boundary (ghost layer " << (domain_ilo - i) << ")\n";
            }
            else if (i > domain_ihi)
            {
                ss << "    X-HI boundary (ghost layer " << (i - domain_ihi) << ")\n";
            }

            if (j < domain_jlo)
            {
                ss << "    Y-LO boundary (ghost layer " << (domain_jlo - j) << ")\n";
            }
            else if (j > domain_jhi)
            {
                ss << "    Y-HI boundary (ghost layer " << (j - domain_jhi) << ")\n";
            }

            // Check if corner
            if ((i < domain_ilo || i > domain_ihi) && (j < domain_jlo || j > domain_jhi))
            {
                ss << "    CORNER CELL\n";
            }
        }

        ss << "  Distance from domain boundaries:\n";
        ss << "    X-LO: " << dist_from_xlo << " cells\n";
        ss << "    X-HI: " << dist_from_xhi << " cells\n";
        ss << "    Y-LO: " << dist_from_ylo << " cells\n";
        ss << "    Y-HI: " << dist_from_yhi << " cells\n";

        return ss.str();
    };

    // Check each field and report first NaN location
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &validbox = mfi.validbox();
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);
        auto p = pressure_mf[lev]->array(mfi);
        auto v = velocity_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        // Scan for NaN (CPU loop for detailed reporting)
        amrex::LoopOnCpu(ghostbox, [&](int i, int j, int k) {
            // Check density
            if (!found_nan && !std::isfinite(rho(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: density\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(rho(i, j, k)) + "\n";
            }

            // Check momentum
            if (!found_nan && (!std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1))))
            {
                found_nan = true;
                nan_details = "Field: momentum\n" + describe_location(i, j, k, validbox);
                nan_details += "M[0]: " + std::to_string(M(i, j, k, 0)) + "\n";
                nan_details += "M[1]: " + std::to_string(M(i, j, k, 1)) + "\n";
            }

            // Check energy
            if (!found_nan && !std::isfinite(E(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: energy\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(E(i, j, k)) + "\n";
            }

            // Check pressure
            if (!found_nan && !std::isfinite(p(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: pressure\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(p(i, j, k)) + "\n";
            }

            // Check velocity
            if (!found_nan && (!std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1))))
            {
                found_nan = true;
                nan_details = "Field: velocity\n" + describe_location(i, j, k, validbox);
                nan_details += "v[0]: " + std::to_string(v(i, j, k, 0)) + "\n";
                nan_details += "v[1]: " + std::to_string(v(i, j, k, 1)) + "\n";
            }

            // Check eta
            if (!found_nan && !std::isfinite(eta(i, j, k)))
            {
                found_nan = true;
                nan_details = "Field: eta\n" + describe_location(i, j, k, validbox);
                nan_details += "Value: " + std::to_string(eta(i, j, k)) + "\n";
            }
        });

        // If NaN found, stop scanning
        if (found_nan)
            break;
    }

    if (found_nan)
    {
        Util::Message(INFO, "=== NaN DETECTED IN FillGhost4BC ===");
        Util::Message(INFO, "Level: ", lev);
        Util::Message(INFO, "Time: ", time);
        Util::Message(INFO, "NSCBC mode: ", use_nscbc);
        Util::Message(INFO, "\n" + nan_details);
        Util::Abort(INFO, "NaN detected in FillGhost4BC - see details above");
    }
} // end FillGhost4BC()


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// POST-SUBCYCLE REFLUX /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::PostSubcycleReflux(int lev, Set::Scalar /*time*/, Set::Scalar /*dt_coarse*/)
{
    BL_PROFILE("Integrator::Hydro2::PostSubcycleReflux");

    if (lev >= finest_level) return;

    int fine_lev = lev + 1;
    if (fine_lev >= (int)flux_reg.size() || !flux_reg[fine_lev]) return;

    // Apply the reflux correction to coarse-level conserved variables.
    // The FluxRegister accumulated:
    //   fine_flux * dt_fine  (positive, from FineAdd across all sub-steps)
    //   coarse_flux * dt_coarse (negative, from CrseInit)
    // Reflux() applies:  U_coarse += scale * (1/vol) * sum_faces(register)
    // Register layout: [alpharho0, alpharho1, mom_x, mom_y, energy, rho_vap]

    // Reflux per-phase densities directly
    flux_reg[fine_lev]->Reflux(*alpharho0_mf[lev],
                               1.0,        // scale
                               0,          // src component in register
                               0,          // dst component
                               1,          // ncomp
                               geom[lev]);

    flux_reg[fine_lev]->Reflux(*alpharho1_mf[lev],
                               1.0,
                               1,          // src component in register
                               0,          // dst component
                               1,
                               geom[lev]);

    // Reflux momentum
    flux_reg[fine_lev]->Reflux(*momentum_mf[lev],
                               1.0,
                               2,          // src component in register
                               0,          // dst component
                               AMREX_SPACEDIM,
                               geom[lev]);

    // Reflux energy
    flux_reg[fine_lev]->Reflux(*energy_per_vol_mf[lev],
                               1.0,
                               2 + AMREX_SPACEDIM,  // src component in register
                               0,                    // dst component
                               1,
                               geom[lev]);

    // Reflux vapor species (register comp 2 + AMREX_SPACEDIM + 1, the last one).
    // Conservative correction of rho_vap at coarse-fine boundaries with the same
    // advective flux that updated alpharho0, so the carrier stays consistent.
    if (species_transport)
        flux_reg[fine_lev]->Reflux(*rho_vap_mf[lev],
                                   1.0,
                                   2 + AMREX_SPACEDIM + 1,  // src component in register
                                   0,                        // dst component
                                   1,
                                   geom[lev]);

    // Recompute derived fields from the refluxed conserved variables.
    const bool spec = (species_transport != 0);
    for (amrex::MFIter mfi(*density_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto rho  = density_mf[lev]->array(mfi);
        auto rho0 = alpharho0_mf[lev]->array(mfi);
        auto rho1 = alpharho1_mf[lev]->array(mfi);
        auto mom  = momentum_mf[lev]->array(mfi);
        auto vel  = velocity_mf[lev]->array(mfi);
        auto rvap = rho_vap_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            vel(i, j, k, 0) = mom(i, j, k, 0) / rho(i, j, k);
            vel(i, j, k, 1) = mom(i, j, k, 1) / rho(i, j, k);
            // Keep the refluxed vapor in [0, alpharho0] so Yv = rho_vap/alpharho0
            // stays physical (the flux correction is bounded by alpharho0's, but
            // guard against round-off pushing it slightly out of range).
            if (spec)
                rvap(i, j, k) = std::max(0.0, std::min(rvap(i, j, k), rho0(i, j, k)));
        });
    }
}


///////////////////////////////////////////////////////////////////////////
///////////////////////// CURVATURE PIPELINE ////////////////////////////
///////////////////////////////////////////////////////////////////////////

void Hydro2::ComputeGradEta(int lev)
{
    BL_PROFILE("Hydro2::ComputeGradEta");

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    // Compute eta gradients into the ghost layer (not just valid cells) so the
    // smoothing/curvature stages downstream are coarse-fine- and physical-
    // boundary-correct: eta is FillPatch'd to nghost in FillGhost4BC, but the
    // derived fields only get same-level FillBoundary, which leaves patch-edge
    // ghosts stale. gE is the ghost depth we fill; reading eta at +/-1 needs eta
    // valid to gE+1 (<= nghost). Targets normals 1-deep -> eta_x krad+1 deep.
    // Non-tiled MFIter avoids the grown-tile write race.
    const int krad = (smooth_kernel_size >= 5) ? 2 : 1;
    const int gE = std::max(0, std::min(krad + 1, nghost - 1));

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.growntilebox(gE);

        auto eta_arr  = eta_mf[lev]->const_array(mfi);
        auto etax_arr = eta_x_mf[lev]->array(mfi);
        auto etay_arr = eta_y_mf[lev]->array(mfi);
        auto gmag_arr = gradmag_mf[lev]->array(mfi);
        auto ge_arr   = grad_eta_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Set::Vector G = Numeric::Gradient(eta_arr, i, j, k, 0, dx);
            amrex::Real gx = G(0);
            amrex::Real gy = G(1);

            etax_arr(i,j,k,0) = gx;
            etay_arr(i,j,k,0) = gy;
            gmag_arr(i,j,k,0) = std::sqrt(gx*gx + gy*gy);
            ge_arr(i,j,k,0)   = gx;
            ge_arr(i,j,k,1)   = gy;
        });
    }

    eta_x_mf[lev]->FillBoundary(geom.periodicity());
    eta_y_mf[lev]->FillBoundary(geom.periodicity());
    gradmag_mf[lev]->FillBoundary(geom.periodicity());
    grad_eta_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeSmoothNormals(int lev)
{
    BL_PROFILE("Hydro2::ComputeSmoothNormals");

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    const int krad = (smooth_kernel_size >= 5) ? 2 : 1;

    // Smooth normals one ghost cell deep so ComputeKappas' divergence (reads n
    // at +/-1) is coarse-fine/physical-boundary correct. Needs eta_x valid to
    // gN+krad, which ComputeGradEta fills (gE = krad+1 when nghost >= krad+2).
    // Falls back to 0 (valid-only, same as before) when the halo is too small.
    const int gN = std::max(0, std::min(1, (nghost - 1) - krad));

    amrex::GpuArray<amrex::Real, 3> gw;
    gw[0] = 1.0;
    gw[1] = std::exp(-0.5);
    gw[2] = std::exp(-2.0);

    for (amrex::MFIter mfi(*nx_smoothed_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.growntilebox(gN);

        auto eta_x   = eta_x_mf[lev]->const_array(mfi);
        auto eta_y   = eta_y_mf[lev]->const_array(mfi);
        auto gradmag = gradmag_mf[lev]->const_array(mfi);
        auto nx_s    = nx_smoothed_mf[lev]->array(mfi);
        auto ny_s    = ny_smoothed_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            amrex::Real accx = 0.0, accy = 0.0, wsum = 0.0;
            const amrex::Box tb_g = amrex::grow(tb, krad);

            for (int di = -krad; di <= krad; ++di)
            for (int dj = -krad; dj <= krad; ++dj)
            {
                int ii = i + di, jj = j + dj;
                if (!tb_g.contains(amrex::IntVect(AMREX_D_DECL(ii, jj, 0)))) continue;

                amrex::Real w    = gw[std::abs(di)] * gw[std::abs(dj)];
                amrex::Real gmN  = gradmag(ii, jj, k) + 1e-14;
                accx += w * eta_x(ii, jj, k) / gmN;
                accy += w * eta_y(ii, jj, k) / gmN;
                wsum += w;
            }

            amrex::Real nx = accx / (wsum + 1e-14);
            amrex::Real ny = accy / (wsum + 1e-14);
            amrex::Real m  = std::sqrt(nx*nx + ny*ny) + 1e-14;
            nx_s(i, j, k) = nx / m;
            ny_s(i, j, k) = ny / m;
        });
    }

    nx_smoothed_mf[lev]->FillBoundary(geom.periodicity());
    ny_smoothed_mf[lev]->FillBoundary(geom.periodicity());
}

void Hydro2::ComputeKappas(int lev)
{
    BL_PROFILE("Hydro2::ComputeKappas");

    ComputeGradEta(lev);
    ComputeSmoothNormals(lev);

    const amrex::Geometry& geom = this->geom[lev];
    const amrex::Real* dx = geom.CellSize();
    const amrex::Box& domain = geom.Domain();

    // |grad eta| floor below which the normal is too weak to define a curvature.
    // Expressed as a fraction (kappa_grad_frac) of the equilibrium peak gradient
    // |grad eta|_eq = 1/(2 sqrt2 eps), so it tracks the interface width, not the
    // mesh: low enough (~2-3%) to keep a stretched/thinned interface (e.g. a neck
    // where |grad eta| has collapsed and the SQR sources should still act), high
    // enough to reject the flat bulk. The max(.,1.0) is an absolute 0/0 guard so a
    // genuinely gradient-free cell can never define a normal even if frac -> 0.
    // (Old floor was 0.1/dx_eff ~ 0.28 * peak, which cut anything below ~28%.)
    const amrex::Real Cg = std::max(kappa_grad_frac / (2.0 * std::sqrt(2.0) * epsilon), 1.0);
    const amrex::Real small = 1e-14;

    for (amrex::MFIter mfi(*kappas_mf[lev], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& tb = mfi.tilebox();

        auto gradmag_arr = gradmag_mf[lev]->const_array(mfi);
        auto nx_s     = nx_smoothed_mf[lev]->const_array(mfi);
        auto ny_s     = ny_smoothed_mf[lev]->const_array(mfi);
        auto kappas   = kappas_mf[lev]->array(mfi);
        auto kappa_sf = kappa_SF_mf[lev]->array(mfi);

        amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real gm = gradmag_arr(i,j,k);

            if (gm < Cg)
            {
                kappas(i,j,k,0) = 0.0;
                kappas(i,j,k,1) = 0.0;
                kappas(i,j,k,2) = 0.0;
                kappa_sf(i,j,k) = 0.0;
                return;
            }

            int il = amrex::max(i-1, domain.smallEnd(0));
            int ir = amrex::min(i+1, domain.bigEnd(0));
            int jl = amrex::max(j-1, domain.smallEnd(1));
            int jr = amrex::min(j+1, domain.bigEnd(1));

            amrex::Real nx_x = 0.5 * (nx_s(ir,j,k) - nx_s(il,j,k)) / dx[0];
            amrex::Real ny_y = 0.5 * (ny_s(i,jr,k) - ny_s(i,jl,k)) / dx[1];

            amrex::Real kSN = -(nx_x + ny_y);

            const amrex::Real kappa_max = 2.0 / amrex::min(dx[0], dx[1]);
            kSN = amrex::max(-kappa_max, amrex::min(kappa_max, kSN));

            kappas(i,j,k,0) = kSN;
            kappas(i,j,k,1) = kSN;
            kappas(i,j,k,2) = 0.0;
            kappa_sf(i,j,k) = kSN;
        });
    }

    kappas_mf[lev]->FillBoundary(geom.periodicity());
}


} // end of Integrator namespace

//#endif
