#include <map>
// Base
#include "Hydro2.H"
#include <memory>
#include <fstream>
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
#include "IC/Profile1D.H"
#include "IC/BMP.H"
#include "IC/PNG.H"
#include "IC/STL.H"   // 3D STL solid geometry (self-guarded: no-op unless USE_EB + 3D)
// Solvers
#include "Solver/Local/FluidRiemann/Roe.H"
#include "Solver/Local/FluidRiemann/HLLE.H"
#include "Solver/Local/FluidRiemann/HLLC.H"
#include "Solver/Local/FluidRiemann/HLLC_Elastic.H"
#include "Solver/Local/FluidRiemann/HLLC_Solid.H"
#ifdef HYDRO2_PELE_MECH
#include "mechanism.cpp"      // the non-inline part of the PelePhysics mechanism (species names, ...), this file only
#endif
#include "Solver/Local/FluidRiemann/HLLC_Oomar_Jaiman.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach.H"
#include "Solver/Local/FluidRiemann/HLLC_All_Mach_Furfaro.H"
#include "Solver/Local/FluidRiemann/HLLCE.H"
//#include "Solver/Local/FluidRiemann/PartiallyParabolic.H"
#include "Solver/Local/FluidRiemann/Upwind.H"
#include "Solver/Local/FluidRiemann/Lax_Friedrich.H"
// Limiters / primitive-variable reconstruction
#include "Solver/Local/Limiter/Limiter.H"
#include "Solver/Local/Limiter/Godunov.H"
#include "Solver/Local/Limiter/Minmod.H"
#include "Solver/Local/Limiter/VanLeer.H"
#include "Solver/Local/Limiter/WENO3.H"
#include "Solver/Local/Limiter/WENO5.H"
#include "Solver/Local/Limiter/MUSCL.H"
#include "Solver/Local/Limiter/MUSCL2.H"
#include "Solver/Local/Limiter/THINC.H"
//EOS
#include "Solver/EOS/EOS.H"
#include <eigen3/Eigen/Eigenvalues>   // SelfAdjointEigenSolver (InterfaceCurvature)
#include "Solver/EOS/Tammann.H"
#include "Solver/EOS/CPG.H"
// Generic
#include <AMReX_Math.H>

#include "AMReX_TimeIntegrator.H"
#include <eigen3/Eigen/Eigenvalues>


namespace Integrator
{

// temperature.phase_consistent (see Parse): phase temperatures from the phase energies and the common temperature.
//   T_k = (E_k - alpha_k pi_k) / (m_k c_v,k), or E_k / (m_k c_v,k) when es_k (species.eos = pele phase: c_v = e_s / T).
//   T   = sum_k m_k c_v,k T_k / sum_k m_k c_v,k over the non-trace phases (mass fraction >= 1e-6); a trace phase
//         reports the other phase's temperature.
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar PhaseConsistentT(Set::Scalar E0, Set::Scalar E1, Set::Scalar m0, Set::Scalar m1, Set::Scalar a0,
                                    const Solver::EOS::Tammann &eos0, const Solver::EOS::Tammann &eos1,
                                    bool es0, bool es1, Set::Scalar &T0, Set::Scalar &T1)
{
    const Set::Scalar a1 = 1.0 - a0;
    const Set::Scalar mt = std::max(m0, 0.0) + std::max(m1, 0.0);
    const bool tr0 = !(mt > 0.0) || m0 < 1.0e-6 * mt, tr1 = !(mt > 0.0) || m1 < 1.0e-6 * mt;
    const Set::Scalar cv0 = eos0.Cv(), cv1 = eos1.Cv();
    T0 = tr0 ? 0.0 : (es0 ? E0 : E0 - a0 * eos0.P0()) / (m0 * cv0);
    T1 = tr1 ? 0.0 : (es1 ? E1 : E1 - a1 * eos1.P0()) / (m1 * cv1);
    if (tr0 && tr1) return 0.0;
    if (tr0) { T0 = T1; return T1; }
    if (tr1) { T1 = T0; return T0; }
    const Set::Scalar w0 = m0 * cv0, w1 = m1 * cv1;
    return (w0 * T0 + w1 * T1) / (w0 + w1);
}

// ----------------------------------------------------------------------
// CAPILLARY ENERGY helper (retained for diagnostics only).
//
// e_cap = sigma_eff |grad eta| is a STATE FUNCTION of (eta, Gamma), never an
// independently transported field -- so it can be recomputed identically at
// every site that assembles or inverts rho E, and can never drift out of
// step with eta.  Both helpers mirror the sigma_eff evaluation in RHS
// exactly; keep them in sync if the Marmottant branch structure changes.
// ----------------------------------------------------------------------
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// Interface gate: a cell with |grad eta| (or |grad psi|) below g_gem_min [1/m] is OFF the
// interface (no shell, no normal, no curvature).  Set in Parse from interface.gem_min,
// default 1e-13/epsilon: scale-free, so round-off noise in eta (~1e-16/dx) is gated out at
// every scale.  The old fixed 1e-10 passed that noise at the micro scale (dx ~ 4e-8 m).
static Set::Scalar g_gem_min = 1.0e-10;
// shell.slip5_rupture (see Parse): Gamma = 0 marks a broken shell -> water tension, no shell viscosity.
static int g_shell_rupture = 0;
static void FaceForceDomainGhosts(amrex::MultiFab &J, const amrex::Geometry &gm);   // TEMP jump.face_force

// [OPT 2026-09-24] Tiling policy for the Hydro2 MFIter loops.  Tiles are only
// useful when OpenMP threads share a box; with one thread (flat-MPI build, or
// OMP_NUM_THREADS=1) the loops run untiled exactly as before, so results and
// cost are unchanged.  NOTE: TagCellsForRefinement keeps its original
// MFIter(mf, true) because its gradient stencils are tile-bounded.
static bool HydroTiling()
{
    return amrex::TilingIfNotGPU() && (amrex::OpenMP::get_max_threads() > 1);
}

static Set::Scalar SigmaEffFromGamma(Set::Scalar Gamma, Set::Scalar grad_eta_mag,
                                     int marm, Set::Scalar sig_const,
                                     Set::Scalar chi_, Set::Scalar Gb,
                                     Set::Scalar sbrk, Set::Scalar sigw)
{
    if (grad_eta_mag <= g_gem_min) return 0.0; // off the interface: no shell
    if (!marm) return sig_const;
    if (g_shell_rupture && Gamma <= 0.0) return sigw;   // broken shell: clean interface (Gamma -> 0+ limit)
    Set::Scalar se = 0.0;
    if (Gamma > 0.0 && Gamma < Gb)             // stretched -> elastic branch
    {
        const Set::Scalar el = chi_ * (Gb / Gamma - 1.0);
        se = (el >= sbrk) ? sigw : el;
    }
    return se;                                 // Gamma >= Gb: buckled, sigma = 0
}

// Factor on the shell viscous tension kappa_s div_s u (see Parse shell.kappa_s_alpha):
// Cross-law shear thinning 1/(1 + alpha |div_s u|/2); alpha = 0 -> 1 (constant kappa_s).
// Face viscosity across a viscosity jump: harmonic mean (series resistance; 0 if either side is inviscid).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar HarmonicMean(Set::Scalar a, Set::Scalar b)
{
    return (a > 0.0 && b > 0.0) ? 2.0 * a * b / (a + b) : 0.0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar ShellViscFactor(Set::Scalar ds, Set::Scalar alpha)
{
    return (alpha > 0.0) ? 1.0 / (1.0 + 0.5 * alpha * std::abs(ds)) : 1.0;
}

// ----------------------------------------------------------------------
// shell.rate_local = 1: interface area rate from a liquid-side band cell's OWN
// fluid stretch (no extension along the normal).  The cell sits at signed
// distance d = eps atanh(2 eta - 1) from eta = 1/2; flux-tube continuity of the
// incompressible liquid maps its surface dilatation to the interface's:
//     r = [div_s u - kappa u_n] + (kappa - 2 K d) u_n / (1 - kappa d + K d^2)^2
// kappa, K from the local level-set tensor S = grad n (two-cell stencil).
// Exact for normal flow on any surface; r = div_s u at d = 0, so the switch at
// eta = 1/2 is continuous.  Returns ds_raw where the map is not valid (gas side,
// near-pure cells, or 1 - kappa d + K d^2 < 1/4).
// ----------------------------------------------------------------------
template <class ARR>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar LiquidInterfaceRate(const ARR &et, int i, int j, int k, const Set::Scalar *DXl,
                                       Set::Scalar eps_l, Set::Scalar cut_l,
                                       const Set::Matrix &gu, const Set::Vector &u, Set::Scalar ds_raw)
{
    const Set::Scalar ec = et(i, j, k);
    if (ec <= 0.5 || ec >= 1.0 - cut_l) return ds_raw;
    const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXl);
    const Set::Scalar g = ge.lpNorm<2>();
    if (g < g_gem_min) return ds_raw;
    const Set::Vector n = ge / g;
    Set::Matrix S = Set::Matrix::Zero();
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
        const int di = (d == 0), dj = (d == 1), dk = (d == 2);
        const Set::Vector gp = Numeric::Gradient(et, i + di, j + dj, k + dk, 0, DXl);
        const Set::Vector gm = Numeric::Gradient(et, i - di, j - dj, k - dk, 0, DXl);
        const Set::Scalar gpm = gp.lpNorm<2>(), gmm = gm.lpNorm<2>();
        if (gpm < g_gem_min || gmm < g_gem_min) return ds_raw;
        const Set::Vector np_ = gp / gpm, nm_ = gm / gmm;
        for (int c = 0; c < AMREX_SPACEDIM; ++c) S(d, c) = (np_(c) - nm_(c)) / (2.0 * DXl[d]);
    }
    const Set::Scalar kap = S.trace();
#if AMREX_SPACEDIM == 3
    const Set::Scalar KG = 0.5 * (kap * kap - (S * S).trace());
#else
    const Set::Scalar KG = 0.0;
#endif
    Set::Scalar dd = eps_l * std::atanh(2.0 * ec - 1.0);
    const Set::Scalar disc = std::sqrt(std::max(0.25 * kap * kap - KG, 0.0));
    const Set::Scalar kd = (AMREX_SPACEDIM == 3) ? std::max((0.5 * kap + disc) * dd, (0.5 * kap - disc) * dd) : kap * dd;
    if (kd > 0.5) dd *= 0.5 / kd;                       // cap k_i d <= 1/2 (thin necks)
    const Set::Scalar den = 1.0 - kap * dd + KG * dd * dd;
    if (den < 0.25) return ds_raw;
    const Set::Scalar un = u.dot(n);
    const Set::Scalar ds_fl = gu.trace() - n.dot(gu * n);
    return (ds_fl - kap * un) + (kap - 2.0 * KG * dd) * un / (den * den);
}

// shell.foot_rate = 1 (see Parse): source field for the foot-point interface rate.  On the well-conditioned
// band core (0.02 < eta < 0.98) comp 0 = the cell's own surface dilatation div_s u = tr(grad u) - n.(grad u) n
// (no flux-tube map), comp 1 = the tensor curvature Kt.  NaN elsewhere.  nGrow 12 for FootExtend.
static void FootRateSource(const amrex::MultiFab &eta, const amrex::MultiFab &vel, const amrex::MultiFab &Kt,
                           const Set::Scalar *DXl, amrex::MultiFab &L)
{
    L.define(eta.boxArray(), eta.DistributionMap(), 2, 12);
    L.setVal(std::numeric_limits<Set::Scalar>::quiet_NaN());
    for (amrex::MFIter mfi(L, false); mfi.isValid(); ++mfi)
    {
        auto et = eta.const_array(mfi); auto v = vel.const_array(mfi); auto kt = Kt.const_array(mfi); auto l = L.array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
            const Set::Scalar ec = et(i, j, k);
            if (!(ec > 0.02 && ec < 0.98)) return;
            const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXl);
            const Set::Scalar g = ge.lpNorm<2>();
            if (g < g_gem_min) return;
            const Set::Vector nh = ge / g;
            Set::Matrix gu = Set::Matrix::Zero();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) gu.col(d) = Numeric::Gradient(v, i, j, k, d, DXl);
            l(i, j, k, 0) = gu.trace() - nh.dot(gu * nh);
            l(i, j, k, 1) = kt(i, j, k);
        });
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar CapEnergyAt(const amrex::Array4<const Set::Scalar> &eta,
                               const amrex::Array4<const Set::Scalar> &shell,
                               int i, int j, int k, const Set::Scalar *DX,
                               int marm, Set::Scalar sig_const, Set::Scalar chi_,
                               Set::Scalar Gb, Set::Scalar sbrk, Set::Scalar sigw)
{
    // The centred gradient reads i+-1, so the outermost ghost ring has no
    // valid stencil.  e_cap is identically zero away from the interface, and
    // the interface never reaches the outer ghosts in any case, so returning
    // 0 there is exact rather than merely safe.
    if (!eta.contains(i - 1, j, k) || !eta.contains(i + 1, j, k)) return 0.0;
#if AMREX_SPACEDIM > 1
    if (!eta.contains(i, j - 1, k) || !eta.contains(i, j + 1, k)) return 0.0;
#endif
#if AMREX_SPACEDIM > 2
    if (!eta.contains(i, j, k - 1) || !eta.contains(i, j, k + 1)) return 0.0;
#endif
    const Set::Vector ge = Numeric::Gradient(eta, i, j, k, 0, DX);
    const Set::Scalar gem = ge.lpNorm<2>();
    const Set::Scalar G = marm ? shell(i, j, k) : 0.0;
    return SigmaEffFromGamma(G, gem, marm, sig_const, chi_, Gb, sbrk, sigw) * gem;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar SpaldingBM(Set::Scalar Y_local, Set::Scalar Y_inf, Set::Scalar small)
{
    return (Y_local - Y_inf) / (1.0 - Y_local + small);
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
        pp.query_default("eta_refinement_criterion", value.eta_refinement_criterion, 0.001);        // eta-based refinement
        pp.query_default("omega_refinement_criterion", value.omega_refinement_criterion, 0.01);     // vorticity-based refinement
        pp.query_default("gradu_refinement_criterion", value.gradu_refinement_criterion, 0.01);     // velocity gradient-based refinement
        pp.query_default("p_refinement_criterion", value.p_refinement_criterion, 1e-3);             // pressure-based refinement
        // Highest level that grad(p) (shock) tags may create; -1 = no cap.  Lets shocks sit on a coarser level than the interface.
        pp.query_default("p_refinement_max_level", value.p_refinement_max_level, -1);
        pp.query_default("rho_refinement_criterion", value.rho_refinement_criterion, 1e-6);         // density-based refinement
        pp.query_default("phi_refinement_criterion", value.embedded.refinement_criterion, 0.001);   // solid-boundary refinement
        {   // optional geometric refinement box: force refinement to max_level inside [lo,hi]
            amrex::ParmParse ppb("refine_box");
            std::vector<Set::Scalar> lo, hi;
            if (ppb.queryarr("lo", lo) && ppb.queryarr("hi", hi)
                && (int)lo.size() >= AMREX_SPACEDIM && (int)hi.size() >= AMREX_SPACEDIM)
            {
                value.refine_box_on = 1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { value.refine_box_lo[d] = lo[d]; value.refine_box_hi[d] = hi[d]; }
            }
            // [2026-10-02] highest level the box forces (default: amr.max_level).  Levels above
            // it refine only where the other criteria tag (e.g. a phi band hugging the body).
            ppb.query("max_level", value.refine_box_max_level);
        }

        // SOLVER AND REFRENCE CONDITIONS
        pp_query_required("cfl", value.cfl);                // cfl condition
        pp_query_default("cfl_v", value.cfl_v, value.cfl);  // cfl condition (viscous specific)
        // visc_subcycle = 1: BULK viscosity (mu0, mu1, mu0_b, mu1_b) is taken out
        // of the RK stages and advanced once per step as its own operator with
        // N explicit (forward-Euler) sub-steps, N = ceil(dt / dtau),
        //     dtau = visc_subcycle_cfl * dx^2 / (2 nu_L),
        //     nu_L = max over cells of (|lambda| + 4/3 mu + (D-1) mu) / rho,
        // the forward-Euler bound of the discrete stress operator (longitudinal
        // + transverse modes).  The global dt is then set by acoustics alone, so
        // the dx^2 viscous limit stops binding at high refinement.  Same
        // discrete stress and dissipation as the RHS path (momentum + total
        // energy only; RelaxAndReinit re-partitions E).  First order in time
        // (Lie split).  DEFAULT 1 since 2026-10-23 (A/B on Couette, Poiseuille,
        // Rayleigh, Taylor-Green, Lamb-Oseen: same answer to <1e-4, 1.3-6x faster
        // when viscosity sets dt).  Inviscid runs are unaffected (early return).
        // visc_subcycle = 0 restores the RK-stage viscous path.
        pp_query_default("visc_subcycle", value.visc_subcycle, 1);
        pp_query_default("visc_subcycle_cfl", value.visc_subcycle_cfl, 0.9);
        pp_query_default("visc_subcycle_max", value.visc_subcycle_max, 1000);
        // visc.flux_form (2026-11-19): the bulk viscous stress of the sub-cycled operator in CONSERVATIVE
        // face-flux form, div(tau) = sum_q [tau_.q]_{-1/2}^{+1/2} / dx_q, with tau built at each face from the
        // VELOCITY field (normal derivative across the face, tangential derivatives averaged from the two cells)
        // and the face viscosity the harmonic mean of the two cells (series resistance: the shear stress is
        // continuous across a viscosity jump).  The energy source is the face flux of tau.u (conservative).
        //   0 = expanded form mu lap(u) + grad(mu).grad(u), lap(u) rebuilt from M and rho by the product rule.
        //       Where mu jumps across a cell (gas-side band tail next to liquid) its central grad(mu).grad(u)
        //       couples a light cell to its far neighbour with a NEGATIVE weight ~ (mu_n - mu_c)/4, and the
        //       product-rule lap(u) amplifies errors by rho_n/rho_c, while the sub-step limit uses the cell's
        //       own mu/rho only: the update is unstable there.  2D driven bubble at R0/dx 32 (claude_scratch/jet2d):
        //       tangential velocity of a 99%-gas band cell at the grid-axis tip doubled every step -> NaN at 0.62 us.
        //   1 = flux form (exact in the continuum, positive definite; sub-step limit from the face viscosities).
        pp_query_validate("visc.flux_form", value.visc_flux_form, {1, 0});   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        pp_query_default("visc_subcycle_verbose", value.visc_subcycle_verbose, 0);
        pp_query_default("dt_diag", value.dt_diag, 0);   // >0: print the dt limits every dt_diag steps
        // [2026-11-18] temperature.phase_consistent, default 0 (legacy).  The legacy T field is the blended single-fluid
        // formula (p + gamma_eff p0_eff) / (rho c_v,eff (gamma_eff - 1)) (EOS::MixedTemperature).  For a stiffened gas
        // that uses gamma pi where the phase temperature has pi, T = (p + pi) / (rho c_v (gamma - 1)): 300 K liquid
        // n-dodecane read 688 K.  It is also undefined in a cell out of temperature equilibrium.  With 1:
        //   T_k = (E_k - alpha_k pi_k) / (alpha_k rho_k c_v,k)   (the phase temperature the vaporization film model and
        //         the chemistry already use; e_s / c_v for a species.eos = pele phase, whose c_v is defined as e_s / T),
        //   T   = sum_k m_k c_v,k T_k / sum_k m_k c_v,k  (the common temperature with the same thermal energy),
        // a phase with mass fraction below 1e-6 (trace) is left out of T and takes the other phase's T_k.  T_phase0 and
        // T_phase1 are written to the plotfile.  T feeds thermal.conduction, so this changes the solution when
        // conduction is on.
        pp_query_default("temperature.phase_consistent", value.T_phase_consistent, 0);
        if (value.T_phase_consistent != 0 && value.T_phase_consistent != 1) Util::Abort(INFO, "temperature.phase_consistent must be 0 or 1");
        // [merge 2026-11-23] DropletDetonation ported three flames2 fixes behind flags that flames2 has always on:
        // cfl_dimensional (summed-direction acoustic dt, see the CFL block), relax.refill_eta_ghosts (eta/shell/cfun
        // ghosts refilled after RelaxAndReinit) and nscbc.ghost_clamp_domain (NSCBC ghost clamp only outside the
        // domain).  All three are always on here; the keys are accepted and setting one to 0 is ignored with a warning.
        for (const char *alw : {"cfl_dimensional", "relax.refill_eta_ghosts", "nscbc.ghost_clamp_domain"})
        {
            int v = 1;
            pp.query(alw, v);
            if (v == 0) Util::Warning(INFO, alw, " = 0 is IGNORED: the behaviour is always on (merge 2026-11-23).");
        }
        pp_query_default("pref", value.pref, 0.0);          // reference pressure for Roe solver
        pp_query_default("small", value.small, 1.0E-8);     // small regularization value
        pp_query_default("cutoff", value.cutoff, 1.0E-8);   // eta cutoff value
        // REMOVED OPTIONS.  Decks that still set one get a warning and the key is ignored.
        // Full history (what each did, why it was dropped): claude_scratch/flag_audit.md.
        //  2026-10-10..27 (broken, rule-out only, or band-weighting fits): shell.divs_kinematic,
        //    trace_volume_relax, shell.kappa_s_area_rate, pressure_relax, capillary_use_eta,
        //    shell_gate_free, shell.kappa_s_liqweight, shell.kappa_s_narrow, mu_mix, mu_mix_p,
        //    density_relax, shell.kappa_s_contour_rate(_cap), shell.contour_rate_gamma,
        //    shell.kappa_s_interface, shell.kappa_s_heat, shell.kappa_s_subcycle.
        //  2026-11-05 (now always on, see INTERFACE JUMP / PRESSURE RELAXATION / SHELL AREA
        //    RATE): jump.consistent, jump.tangential, jump.R0, jump.curvature, jump.curvature_y,
        //    jump.curvature_iso, capillary_jump, shell.kappa_s_jump, relax.closure,
        //    relax.allow_tension, relax.trace, relax.reinit_split, shell.rate_local,
        //    shell_bulk_extend, shell.band_eta, eta_consistent_advect, omega_sym_mirror,
        //    sym_face_central.
        //  2026-11-05 (failed or superseded): shell.gamma_uniform (+ _sweeps; normal extension),
        //    shell.rate_liquid_eta, shell.rate_liquid_sweeps, shell.rate_liquid_debug,
        //    shell.kappa_s_rate, jump.curvature_cp (no effect), kappa_method (plot only),
        //    shell.heat_to_liquid.
        for (const char *removed : {"shell.divs_kinematic", "trace_volume_relax", "shell.kappa_s_area_rate",
                                    "pressure_relax", "capillary_use_eta", "shell_gate_free",
                                    "shell.kappa_s_liqweight", "shell.kappa_s_narrow", "mu_mix", "mu_mix_p",
                                    "density_relax", "shell.kappa_s_contour_rate", "shell.kappa_s_contour_rate_cap",
                                    "shell.contour_rate_gamma", "shell.kappa_s_interface", "shell.kappa_s_heat",
                                    "shell.kappa_s_subcycle",
                                    "jump.consistent", "jump.tangential", "jump.R0", "jump.curvature",
                                    "jump.curvature_y", "jump.curvature_iso", "capillary_jump", "shell.kappa_s_jump",
                                    "relax.closure", "relax.allow_tension", "relax.trace", "relax.reinit_split",
                                    "shell.rate_local", "shell_bulk_extend", "shell.band_eta",
                                    "omega_sym_mirror", "sym_face_central",
                                    "shell.gamma_uniform", "shell.gamma_uniform_sweeps", "shell.rate_liquid_eta",
                                    "shell.rate_liquid_sweeps", "shell.rate_liquid_debug", "shell.kappa_s_rate",
                                    "jump.curvature_cp", "kappa_method", "shell.heat_to_liquid",
                                    // merge 2026-11-23: DropletDetonation's PolymerVisc-port options (superseded in flames2)
                                    "capillary.jump_scheme", "eta_regularize", "eta_regularize_speed", "eta_trace_kapila",
                                    "shell.divs_limit_upper", "shell.gamma_interface", "shell.no_compression",
                                    "shell.visc_true_area", "shell.visc_layer", "shell.band_weight",
                                    "shell.visc_liquid_side", "omega_sym_mirror", "sym_face_central"})
            if (pp.contains(removed))
                Util::Warning(INFO, removed, " is IGNORED: the option was removed (see Hydro2::Parse, "
                              "REMOVED OPTIONS).  Delete it from the input file.");
        pp_query_default("lagrange", value.lagrange, 0.0);  // lagrange no-penetration factor
        pp_query_default("grav", value.g, 9.81);            // Gravitational Acceletation

        // OPTIONAL SOURCE TERMS
        pp_query_default("apply_surface_tension", value.apply_surface_tension, false);  // Apply surface tension when solving, default: true --> "Apply Surface Tension"
        // Boussinesq--Scriven interfacial viscosity (see Hydro2.H).  kappa_s is
        // applied through sigma_tot; mu_s is parsed only so a request for the
        // unimplemented shear term is caught here rather than silently ignored.
        //
        // ALWAYS ON (flags removed 2026-11-05, see REMOVED OPTIONS above):
        //  * symmetry (REFLECT) faces: the ghost Omega is the mirror of the first valid
        //    cell (diagonal and in-plane shear even, the two shear components involving
        //    the face normal odd), and the eta/rho Hessians use central differences in
        //    the first cell (one-sided made the row inconsistent with its mirror image:
        //    quarter vs full domain differed by 7.4e-3 in eta, now 1e-7)
        //    (were omega_sym_mirror, sym_face_central).
        //
        // eta_consistent_advect (restored as a flag in the 2026-11-23 merge):
        //   1 (default) = alpha row from the limiter-reconstructed face alpha, upwinded on S_M, so it
        //       has the same truncation error as the mass rows (flames2 hard-wired this 2026-11-05:
        //       donor cell made pressure-floor cells in the gas core; DropletDetonation needs it for
        //       higher-order limiters at metal|gas interfaces).
        //   0 = donor-cell alpha row (the DropletDetonation default before the merge; its AIAA,
        //       droplet, Riemann and harness reference runs used it).
        pp_query_validate("eta_consistent_advect", value.eta_consistent_advect, {1, 0});
        // RK-stage bulk viscous path (from DropletDetonation, II.55 / II.63; all default 0).  They act
        // where the bulk viscosity is in the RK stages (visc_subcycle = 0, forced with the embedded solid);
        // the sub-cycled operator has its own flux form (visc.flux_form, see above).
        //   gradu_from_velocity = 1: grad u / Hess u by central differences of the velocity field, not the
        //       product rule on M and rho (which amplifies by rho_{i+1}/rho_i at a sharp interface foot).
        //   visc_flux_form = 1: div(tau) as a difference of FACE stresses (face mu = arithmetic mean).
        //   visc_harmonic = 1 (needs visc_flux_form = 1): harmonic face mu 1/(e/mu0 + (1-e)/mu1).
        pp_query_default("gradu_from_velocity", value.gradu_from_velocity, 0);
        pp_query_default("visc_flux_form", value.visc_flux_form_rk, 0);
        pp_query_default("visc_harmonic", value.visc_harmonic, 0);
        if (value.visc_harmonic && !value.visc_flux_form_rk)
            Util::Abort(INFO, "visc_harmonic = 1 needs visc_flux_form = 1");
        pp_query_default("shell.kappa_s", value.shell_kappa_s, 0.0);
        pp_query_default("shell.mu_s",    value.shell_mu_s,    0.0);
        if (value.shell_mu_s != 0.0)
            Util::Message(INFO, "Shell SHEAR viscosity ON: mu_s = ", value.shell_mu_s,
                          " kg/s (2 mu_s D_s, D_s = P sym(grad u) P).  Note this term "
                          "cancels identically for spherical motion -- it only acts on "
                          "shape change at constant area.");
        if (value.shell_kappa_s != 0.0)
            Util::Message(INFO, "Shell dilatational viscosity ON: kappa_s = ", value.shell_kappa_s,
                          " kg/s (sigma_tot = sigma_eff + kappa_s div_s u)");
        pp_query_default("apply_weight", value.apply_weight, false);                    // Apply weight when solving, default: false --> "No Weight"
        pp_query_default("apply_vaporization", value.apply_vaporization, false);        // Enforces Eta boundry to be prescribed constant: false --> "moveable boundry"

        // EMBEDDED SOLID BOUNDARY (see FlowWedge for examples)
        pp_query_default("apply_embedded_solid", value.embedded.apply, 0);  // Apply Solid Boundry 1 --> "Domain has solid boundry"
        if (value.visc_subcycle && value.embedded.apply)
        {
            // not implemented with the embedded solid: fall back to the RK-stage path
            value.visc_subcycle = 0;
            Util::Warning(INFO, "visc_subcycle not supported with apply_embedded_solid: using the RK-stage viscous path");
        }
                // [2026-09-25] default 1e6 (stiff sharp wall, used with solid.implicit = 1).
        // <= 0 selects the old auto momentum-projection wall.
        pp_query_default("solid.brinkman", value.embedded.brinkman, 1.0e6); // Yang(2023) momentum-only Brinkman no-penetration
        pp_query_default("solid.slip", value.embedded.slip, 0);             // 1 = slip wall (penalize only wall-normal momentum); 0 = no-slip
        // [2026-09-25] solid viscosity = solid.mu_factor * fluid viscosity, blended by
        // (1 - phi) into the Cauchy stress (single-phase Hydro uses 100).
        // 1 = off (default).  Only acts where the fluid is viscous.
        pp_query_default("solid.mu_factor", value.embedded.mu_factor, 1.0);
        if (value.embedded.mu_factor < 1.0)
            Util::Abort(INFO, "solid.mu_factor must be >= 1 (1 = no solid viscosity)");
        // [2026-09-24] 1 = apply the Brinkman penalty IMPLICITLY (exact exponential
        // decay, once per level step after the RK advance) instead of as an
        // explicit RHS source.  The explicit source is only stable for
        // brinkman*dt < ~2.5 on EVERY level, which caps brinkman at O(1/dt_coarse)
        // (~150 on a 0.06-cell base grid) -- far too weak for a sharp no-slip
        // wall (penalization layer sqrt(nu/brinkman) must be << dx).  The exact
        // update M <- Ms + (M-Ms) exp(-brinkman (1-phi) dt) is unconditionally
        // stable, so brinkman can be 1e4-1e8.  [2026-09-25] 1 is the default;
        // 0 restores the old explicit RHS source.
        pp_query_default("solid.implicit", value.embedded.implicit, 1);
        // [2026-09-24] 1 = SHARP-WALL face flux.  The embedded solid is "full-flux
        // porous": every face, including fluid/solid ones, gets the ordinary
        // Riemann flux, and only momentum is penalized.  With a sharp wall
        // (phi a step) and a stiff penalty that is a mass trap: at a
        // stagnation point the fluid->solid face carries mass INTO the first
        // solid cell (u* > 0), the penalty zeroes its momentum, and nothing
        // carries it out -- measured on the viscous NACA 0012 (M 0.5, Re 5000,
        // max_level 4): LE solid cell rho 1.0 -> 1.35 (t=2) -> 1.74 (t=4) ->
        // singular at t = 5.6.  With wall_flux = 1, a face whose two cells lie
        // on opposite sides of phi = 0.5 is solved as a wall: fluid cell vs its
        // mirror (normal velocity reflected about the solid's), so u* = 0, no
        // mass/energy crosses and only pressure acts; faces with both cells in
        // the solid carry zero flux.  Tangential no-slip still comes from the
        // Brinkman term + viscous stress.  0 = diffuse (porous full-flux) wall.
        // [2026-09-25] AUTO SELECTION when solid.wall_flux is NOT given: the wall
        // stays diffuse (0) unless the phi interface is thinner than
        // solid.wall_flux_cells (default 2) cells on the finest level built at
        // initialization (thickness ~ 1 / max |phi(i+1) - phi(i)|; a step is
        // ~1 cell, a tanh profile ~2 eps/dx), in which case the mirrored sharp-wall
        // flux (1) is switched on -- see TimeStepBegin.  An explicit
        // solid.wall_flux = 0 or 1 always wins.
        if (pp.contains("solid.wall_flux"))
        {
            pp_query_default("solid.wall_flux", value.embedded.wall_flux, 0);
            value.embedded.wall_flux_auto = 0;
        }
        else
        {
            value.embedded.wall_flux = 0;
            value.embedded.wall_flux_auto = 1;
        }
        if (value.embedded.wall_flux != 0 && value.embedded.wall_flux != 1)
            Util::Abort(INFO, "solid.wall_flux must be 0 or 1");
        pp_query_default("solid.wall_flux_cells", value.embedded.wall_flux_cells, 2.0);
        // [2026-09-27] Sharp-wall viscous stencil (see RHS): 1 = viscous terms see the
        // no-slip wall on the fluid|solid FACE (mirrored velocity), consistent with the
        // mirrored Riemann flux; 0 = solid cells' own velocity one cell away (validated
        // default -- on a staircase body the full mirrored shear acts on every step
        // face, i.e. on up to 4/pi of the true area: Re40 cylinder friction 0.576 vs
        // 0.520 with 0).  Only read when solid.wall_flux = 1.
        // [2026-10-02] 2 = TRUE-SURFACE viscous ghost: the wall is placed at phi = 0.5
        // (ratio r = (0.5 - phi_s)/(phi_f - 0.5), cap 4), which removes the staircase
        // over-count of 1; identical to 1 for a 0/1 phi.  See RHS.
        pp_query_default("solid.visc_mirror", value.embedded.visc_mirror, 0);
        // [2026-09-29] Sharp-wall geometry (see RHS): 1 = SLIP walls reflect the ghost
        // (wall-face Riemann problem and reconstruction window) about the TRUE surface
        // normal n = grad(phi)/|grad(phi)| (Gaussian-smoothed, radius 3) -- ghost-cell
        // immersed boundary; 0 = the old face-normal mirror (staircase: every step riser
        // of an inclined wall stagnates the along-surface flow).  No-slip walls are
        // unaffected (their mirror 2 u_s - u does not depend on n).
        // Only read when solid.wall_flux = 1.  Needs nghost >= 4 for the full kernel
        // (indices are clamped to the fab otherwise).
        pp_query_default("solid.wall_normal", value.embedded.wall_normal, 1);
        // [2026-09-29] with wall_normal + slip: hand the mass/energy that crosses riser
        // faces into solid cells back to their fluid neighbours (exact conservation).
        // OFF by default: at a one-cell-thick trailing edge it moves mass between the
        // pressure and suction sides (trailing-edge "blowing"): Euler NACA 0008 a4 Cl
        // 1.16 (v4) / -0.26 (v7) / -0.34 (v8) vs 0.22 (v8b, off) at 2 c/U.
        pp_query_default("solid.wall_redistribute", value.embedded.wall_redistribute, 0);
        // [2026-09-29] wall_normal: a fluid|solid face whose solid is <= this many cells
        // thick along the face axis (a plate: sharp trailing edges, wedge tips) uses the
        // face normal -- the smoothed normal is unreliable there.  0 = never.
        pp_query_default("solid.wall_thin_cells", value.embedded.wall_thin_cells, 2);
        // [2026-09-29] wall_normal slip: place the slip plane at the true surface (phi = 0.5
        // interpolated between the fluid and solid cell) instead of the staircase face.
        pp_query_default("solid.wall_image", value.embedded.wall_image, 1);
        // [2026-09-30] same for NO-SLIP walls (ghost velocity -r (u - u_s), linear to zero at
        // phi = 0.5).  Identical to the plain mirror for a 0/1 phi.
        pp_query_default("solid.wall_image_noslip", value.embedded.wall_image_noslip, 1);
        // [2026-10-05] MOVING RIGID SOLID.
        // solid.wall_vel_surface (default 1): wall velocity evaluated at the true surface
        //   point between the fluid and solid cells (see wall_vel in RHS).  Only matters
        //   when the solid velocity varies in space (rotation).
        // solid.moving (default 0): 1 = the geometry itself moves.  solid.phi.ic and
        //   solid.momentum.ic are re-evaluated at the current time at the start of
        //   every step (give them as expressions of x,y,z,t), and cells the body has
        //   just vacated are filled from the neighbouring fluid (UpdateMovingSolid).
        //   Prescribed rigid motion only; the body does not respond to the flow.
        pp_query_default("solid.wall_vel_surface", value.embedded.wall_vel_surface, 1);
        pp_query_default("solid.moving", value.embedded.moving, 0);

        // [2026-10-05] ISOTHERMAL SHARP WALL for the gas heat conduction (thermal.conduction = 1):
        // solid.T_wall > 0 puts the wall at that temperature (surface at phi = 0.5); default adiabatic.
        pp_query_default("solid.T_wall", value.embedded.T_wall, -1.0);
        // [2026-10-05] SURFACE ABLATION (default off), heat-of-ablation model -- see Ablation.H.
        pp_query_default("ablation.on", value.ablation.on, 0);
        if (value.ablation.on)
        {
            pp_query_required("ablation.T_wall", value.ablation.T_wall);   // wall (ablation) temperature
            pp_query_required("ablation.Qstar", value.ablation.Qstar);     // heat of ablation per unit mass of solid
            pp_query_required("ablation.rho_s", value.ablation.rho_s);     // solid density
            pp_query_default("ablation.accel", value.ablation.accel, 1.0); // recession speed multiplier
            pp_query_default("ablation.extend", value.ablation.extend, 8); // passes extending v_abl off the wall
            pp_query_default("ablation.t_start", value.ablation.t_start, 0.0); // no recession before this time
            value.embedded.T_wall = value.ablation.T_wall;
            // [2026-10-06] in-depth conduction in the solid; T_wall is then the ablation temperature
            pp_query_default("ablation.conduction", value.ablation.conduction, 0);
            if (value.ablation.conduction)
            {
                pp_query_required("ablation.k_s", value.ablation.k_s);     // solid thermal conductivity
                pp_query_required("ablation.cp_s", value.ablation.cp_s);   // solid specific heat
                pp_query_default("ablation.T_solid0", value.ablation.T_solid0, value.ablation.T_wall); // initial solid temperature
            }
        }
        ElasticSolid::Parse(value.elastic, pp, value.geom);     // deforming solid phase (elastic.*), see Hydro2_Solid.H
        // [2026-10-06] SPECIES inside the two phases (default off), stage 1 = passive transport -- see Hydro2_Species.H.
        pp_query_default("species.on", value.species.on, 0);
        if (value.species.on)
        {
            std::vector<std::string> n0, n1;
            pp.queryarr("species.phase0.names", n0);      // species of phase 0 (the eta = 1 phase)
            pp.queryarr("species.phase1.names", n1);      // species of phase 1
            value.species.n0 = (int)n0.size(); value.species.n1 = (int)n1.size();
            value.species.names = n0; value.species.names.insert(value.species.names.end(), n1.begin(), n1.end());
            if (value.species.ns() < 1) Util::Abort(INFO, "species.on = 1 needs species.phase0.names and/or species.phase1.names");
            // [2026-10-06] stage 2a: species-dependent gamma, c_v inside a phase (Hydro2_Species.H); default fixed
            {
                std::string seos = "fixed"; pp.query("species.eos", seos);
                if (seos != "fixed" && seos != "mixture" && seos != "pele") Util::Abort(INFO, "species.eos must be fixed, mixture or pele");
                value.species.eos_mix = (seos == "fixed") ? 0 : 1;
                if (seos == "pele")     // thermally perfect species of the PelePhysics mechanism of this build (Hydro2_Pele.H)
                {
#ifdef HYDRO2_PELE_MECH
                    const int ph = (value.species.n1 == Pele::NS) ? 1 : ((value.species.n0 == Pele::NS) ? 0 : -1);
                    if (ph < 0) Util::Abort(INFO, "species.eos = pele: one phase must list the ", Pele::NS, " species of the mechanism, in its order");
                    amrex::Vector<std::string> kn; CKSYMS_STR(kn);
                    const int c0 = ph ? value.species.n0 : 0;
                    for (int s = 0; s < Pele::NS; ++s)
                        if (value.species.names[c0 + s] != kn[s])
                            Util::Abort(INFO, "species.eos = pele: species ", s, " of phase ", ph, " must be ", kn[s], " (mechanism order), found ", value.species.names[c0 + s]);
                    { std::string ps = "effective"; pp.query("species.pele_sound", ps);     // effective | frozen (Hydro2_Pele.H)
                      if (ps != "effective" && ps != "frozen") Util::Abort(INFO, "species.pele_sound must be effective or frozen");
                      value.species.pele.frozen_c = (ps == "frozen") ? 1 : 0; }
                    value.species.pele.on = 1; value.species.pele.ph = ph; value.species.pele.c0 = c0;
                    value.species.pele.Init();
                    Util::Message(INFO, "species.eos = pele: ", Pele::NS, " thermally perfect species in phase ", ph, " (deck must be SI)");
#else
                    Util::Abort(INFO, "species.eos = pele needs a binary built with HYDRO2_MECH=<PelePhysics mechanism directory>");
#endif
                }
                else if (value.species.eos_mix)
                {
            if (value.species.n0 > Species::NMAX || value.species.n1 > Species::NMAX) Util::Abort(INFO, "species: too many species in one phase");
                    for (int ph = 0; ph < 2; ++ph)
                    {
                        const int n = ph ? value.species.n1 : value.species.n0, c0 = ph ? value.species.n0 : 0;
                        if (n == 0) continue;
                        const std::string pre = ph ? "species.phase1." : "species.phase0.";
                        std::vector<Set::Scalar> g, cv;
                        pp.queryarr((pre + "gamma").c_str(), g); pp.queryarr((pre + "cv").c_str(), cv);
                        if ((int)g.size() != n || (int)cv.size() != n)
                            Util::Abort(INFO, "species.eos = mixture needs ", pre, "gamma and ", pre, "cv with one value per species");
                        for (int s = 0; s < n; ++s)
                        {
                            if (!(g[s] > 1.0) || !(cv[s] > 0.0)) Util::Abort(INFO, "species.eos = mixture: gamma must be > 1 and cv > 0");
                            value.species.sgam[c0 + s] = g[s]; value.species.scv[c0 + s] = cv[s];
                        }
                        // stage 2b: different stiffness per species of this phase (volume-weighted closure, Hydro2_Species.H)
                        std::vector<Set::Scalar> pin, rr;
                        pp.queryarr((pre + "pinf").c_str(), pin); pp.queryarr((pre + "rho_ref").c_str(), rr);
                        if (!pin.empty() || !rr.empty())
                        {
                            if ((int)pin.size() != n || (int)rr.size() != n)
                                Util::Abort(INFO, pre, "pinf and ", pre, "rho_ref must both be given, one value per species");
                            for (int s = 0; s < n; ++s)
                            {
                                if (!(rr[s] > 0.0) || pin[s] < 0.0) Util::Abort(INFO, pre, "rho_ref must be > 0 and pinf >= 0");
                                value.species.spi[c0 + s] = pin[s]; value.species.srho[c0 + s] = rr[s];
                            }
                            value.species.has_pi[ph] = 1;
                        }
                    }
                }
            }
            pp_query_default("species.order", value.species.order, 1);     // 1 = upwind cell, 2 = van Leer limited face mass fractions
            // [2026-10-10] species.order = 3: THINC face mass fraction (Wallis, Barton & Nikiforakis, Comput. Struct. 254
            // (2021) 106578, Appendix A, eqs. A.1-A.7 and A.10) in the cells where the mass fraction is strictly between
            // species.thinc_eps and 1 - species.thinc_eps and monotone across the three cells; the van Leer value of
            // order 2 elsewhere.  beta = species.thinc_beta |n_d| + 0.01, n = normal of the species boundary from the
            // central differences of the mass fraction (their beta_0 = 2.5).  Keeps the boundary between two species
            // of one phase (an explosive against its confiner) about two cells wide.  Their BVD selection between
            // the THINC and the MUSCL value (A.8-A.9) is NOT implemented.
            pp_query_default("species.thinc_beta", value.species.thinc_beta, 2.5);
            pp_query_default("species.thinc_eps", value.species.thinc_eps, 1.0e-4);
            if (value.species.order < 1 || value.species.order > 3) Util::Abort(INFO, "species.order must be 1, 2 or 3");
            if (value.species.n0 > Species::NMAX || value.species.n1 > Species::NMAX) Util::Abort(INFO, "species: more than ", Species::NMAX, " species in one phase (raise Species::NMAX)");
            // [2026-11-18] species.trace_rho [kg/m^3], default 0 (off).  A phase whose partial density alpha_k rho_k is
            // below this value keeps the base EOS (eos0 / eos1) instead of the composition closure (Species::Eos::Mix).
            // Why: where a phase is only a trace (alpha_k rho_k at the `small` floor) its composition and e_s = E_k/m_k
            // are noise; with species.eos = pele the closure turned them into gamma ~ 1.01 (Ms 5 dodecane column, gas
            // emptied from the rear band: 3-10 K in the T output).  Suggested value: 100 x small.
            pp_query_default("species.trace_rho", value.species.trace_rho, 0.0);
            if (value.species.trace_rho < 0.0) Util::Abort(INFO, "species.trace_rho must be >= 0");
            // [2026-11-18] SPECIES DIFFUSION (design stage 6), default 0.  Fickian diffusion of the species of ONE phase
            // inside that phase, operator-split once per step after the hyperbolic advance (Hydro2::DiffuseSpecies):
            //     j_i = -(alpha rho)_k D grad Y_i + Y_i sum_l (alpha rho)_k D grad Y_l    (the correction keeps sum_i j_i = 0,
            //     so the phase mass and the flow are untouched), with the energy flux sum_i h_i j_i on E_k and rho E
            // (h_i = gamma_i c_v,i T for species.eos = mixture, the NASA sensible enthalpy for species.eos = pele; 0 when
            // all species share one EOS).  D is one constant (species.diffusion.D) or mu_k/(rho_k Sc) (species.diffusion.Sc,
            // the phase viscosity mu0 / mu1).  Faces only between cells with alpha_k >= species.diffusion.alpha_min (no
            // diffusion into the other phase).  Explicit sub-steps dtau <= species.diffusion.cfl dx^2/(2 d D_max).
            pp_query_default("species.diffusion", value.species.diff_on, 0);
            if (value.species.diff_on)
            {
                pp_query_default("species.diffusion.phase", value.species.diff_phase, 0);
                pp_query_default("species.diffusion.D", value.species.diff_D, 0.0);
                pp_query_default("species.diffusion.Sc", value.species.diff_Sc, 0.0);
                pp_query_default("species.diffusion.alpha_min", value.species.diff_alpha_min, 1.0e-3);
                pp_query_default("species.diffusion.cfl", value.species.diff_cfl, 0.9);
                pp_query_default("species.diffusion.max_sub", value.species.diff_max_sub, 1000);
                pp_query_default("species.diffusion.verbose", value.species.diff_verbose, 0);
                if (value.species.diff_phase != 0 && value.species.diff_phase != 1) Util::Abort(INFO, "species.diffusion.phase must be 0 or 1");
                if ((value.species.diff_D > 0.0) == (value.species.diff_Sc > 0.0)) Util::Abort(INFO, "species.diffusion: give exactly one of species.diffusion.D > 0 or species.diffusion.Sc > 0");
                if (((value.species.diff_phase == 0) ? value.species.n0 : value.species.n1) < 2) Util::Abort(INFO, "species.diffusion: the phase needs at least two species");
            }
        }
        // [2026-10-06] several SOLIDS as species of one phase (stage 2b): per-species shear modulus, yield stress and
        // plastic viscosity -- see Hydro2_Solid.H (struct Mat).  Needs species.eos = mixture with pinf / rho_ref.
        if (value.elastic.on && pp.contains("elastic.species.mu"))
        {
            const int ph = value.species.has_pi[0] ? 0 : (value.species.has_pi[1] ? 1 : -1);
            if (!value.species.on || ph < 0)
                Util::Abort(INFO, "elastic.species.mu needs species.eos = mixture with species.phaseK.pinf and rho_ref for the solid phase");
            const int n = ph ? value.species.n1 : value.species.n0, c0 = ph ? value.species.n0 : 0;
            std::vector<Set::Scalar> m_, sy_, ky_;
            pp.queryarr("elastic.species.mu", m_); pp.queryarr("elastic.species.sigma_y", sy_); pp.queryarr("elastic.species.Ky", ky_);
            if ((int)m_.size() != n || (!sy_.empty() && (int)sy_.size() != n) || (!ky_.empty() && (int)ky_.size() != n))
                Util::Abort(INFO, "elastic.species.mu / sigma_y / Ky: one value per species of phase ", ph);
            if (n > ElasticSolid::SPMAX) Util::Abort(INFO, "elastic.species: too many species");
            value.elastic.sp_on = 1; value.elastic.sp_ph = ph; value.elastic.sp_c0 = c0; value.elastic.sp_n = n;
            Set::Scalar mmax = 0.0;
            for (int s_ = 0; s_ < n; ++s_)
            {
                value.elastic.sp_mu[s_] = m_[s_]; value.elastic.sp_r0[s_] = value.species.srho[c0 + s_];
                value.elastic.sp_sy[s_] = sy_.empty() ? 0.0 : sy_[s_]; value.elastic.sp_ky[s_] = ky_.empty() ? 0.0 : ky_[s_];
                mmax = std::max(mmax, m_[s_]);
            }
            if (!(value.elastic.mu[ph] > 0.0)) value.elastic.mu[ph] = mmax;      // marks the phase as solid; the cell values replace it
            std::vector<Set::Scalar> ch_, gt_;
            pp.queryarr("elastic.species.chi", ch_); pp.queryarr("elastic.species.gtn", gt_);    // compaction modulus, GTN (1) or von Mises (0)
            if ((!ch_.empty() && (int)ch_.size() != n) || (!gt_.empty() && (int)gt_.size() != n))
                Util::Abort(INFO, "elastic.species.chi / gtn: one value per species of phase ", ph);
            for (int s_ = 0; s_ < n; ++s_) { value.elastic.sp_chi[s_] = ch_.empty() ? 0.0 : ch_[s_]; value.elastic.sp_gtn[s_] = gt_.empty() ? 0.0 : gt_[s_]; }
        }
        if (value.elastic.on && value.elastic.comp_on)
        {
            if (!(value.elastic.mu[0] > 0.0) || value.elastic.mu[1] > 0.0)
                Util::Abort(INFO, "elastic.compaction = 1: the solid must be phase 0 (eta = 1) and phase 1 a fluid");
            if (value.marmottant) Util::Abort(INFO, "elastic.compaction = 1 uses the colour function field and cannot be combined with marmottant = 1");
        }
        // [2026-10-06] CHEMISTRY on the species (default off): native one-step Arrhenius reaction -- see Hydro2_Chem.H.
        pp_query_default("chem.on", value.chem.on, 0);
        pp_query_default("chem.alpha_min", value.chem.alpha_min, 0.0);   // no reaction below this volume fraction of the reacting phase
        if (value.chem.on)
        {
            if (!value.species.on) Util::Abort(INFO, "chem.on = 1 needs species.on = 1");
            {
                std::string be = "native"; pp.query("chem.backend", be);      // native (one-step) | pele (mechanism of the build)
                if (be != "native" && be != "pele") Util::Abort(INFO, "chem.backend must be native or pele");
                value.chem.pele = (be == "pele") ? 1 : 0;
            }
            if (value.chem.pele)
            {
                if (!value.species.pele.on) Util::Abort(INFO, "chem.backend = pele needs species.eos = pele");
                pp_query_default("chem.T_min", value.chem.T_min, 0.0);     // no reaction below this temperature [K]
                pp_query_default("chem.rtol", value.chem.rtol, 1.0e-6);    // relative tolerance of the reactor (mass fractions)
                pp_query_default("chem.atol", value.chem.atol, 1.0e-12);   // absolute tolerance of the reactor (mass fractions)
                pp_query_default("chem.max_sub", value.chem.max_sub, 20000); // sub-step limit per cell and step
                value.chem.phase = value.species.pele.ph;
            }
            else {
            pp_query_required("chem.reactant", value.chem.reactant);   // species consumed
            pp_query_required("chem.product", value.chem.product);     // species produced (same phase)
            pp_query_required("chem.A", value.chem.A);                 // pre-exponential factor [1/s]
            pp_query_default("chem.Ta", value.chem.Ta, 0.0);           // activation temperature E_a/R
            pp_query_required("chem.Q", value.chem.Q);                 // heat released per unit mass of reactant
            pp_query_default("chem.T_min", value.chem.T_min, 0.0);     // no reaction below this temperature
            pp_query_default("chem.dY_max", value.chem.dY_max, 0.05);  // sub-step limit on the relative change of Y
            pp_query_default("chem.dT_max", value.chem.dT_max, 0.02);  // sub-step limit on the relative change of T
            const auto &nm = value.species.names;
            for (int n = 0; n < (int)nm.size(); ++n) { if (nm[n] == value.chem.reactant) value.chem.ir = n; if (nm[n] == value.chem.product) value.chem.ip = n; }
            if (value.chem.ir < 0 || value.chem.ip < 0) Util::Abort(INFO, "chem: reactant / product must be species names");
            const int phr = (value.chem.ir < value.species.n0) ? 0 : 1, php = (value.chem.ip < value.species.n0) ? 0 : 1;
            if (phr != php) Util::Abort(INFO, "chem: reactant and product must belong to the same phase (phase change is a separate process)");
            value.chem.phase = phr;
            // global fuel + oxidizer step (see Hydro2_Chem.H), default off
            pp.query("chem.oxidizer", value.chem.oxidizer);
            if (!value.chem.oxidizer.empty())
            {
                for (int n = 0; n < (int)nm.size(); ++n) if (nm[n] == value.chem.oxidizer) value.chem.iox = n;
                if (value.chem.iox < 0) Util::Abort(INFO, "chem.oxidizer must be a species name");
                if (((value.chem.iox < value.species.n0) ? 0 : 1) != phr) Util::Abort(INFO, "chem.oxidizer must be in the phase of the reactant");
                pp_query_required("chem.s", value.chem.s_ox);
                pp_query_default("chem.order_fuel", value.chem.a_F, 1.0);
                pp_query_default("chem.order_ox", value.chem.b_O, 1.0);
                if (!(value.chem.s_ox > 0.0)) Util::Abort(INFO, "chem.s (oxidizer-to-fuel mass ratio) must be > 0");
            }
            }
        }
        // [2026-10-09] ENERGETIC MATERIAL (default off): reactant species of phase 0 -> product species of phase 1 --
        // see Hydro2_Energetic.H.
        pp_query_default("energetic.on", value.energetic.on, 0);
        if (value.energetic.on)
        {
            if (!value.species.on) Util::Abort(INFO, "energetic.on = 1 needs species.on = 1");
            pp_query_required("energetic.reactant", value.energetic.reactant);   // species of phase 0 that reacts
            pp_query_required("energetic.product", value.energetic.product);     // species of phase 1 it becomes
            std::string elaw = "sqrt";
            pp.query_validate("energetic.law", elaw, {"sqrt", "pressure"});     // reaction-progress law
            value.energetic.law = (elaw == "pressure") ? 1 : 0;
            pp_query_required("energetic.k", value.energetic.k);                 // rate constant [1/time]
            pp_query_required("energetic.Q", value.energetic.Q);                 // heat of reaction per unit mass
            pp_query_default("energetic.p_ign", value.energetic.p_ign, 0.0);     // the reaction runs where p > p_ign
            pp_query_default("energetic.p_ref", value.energetic.p_ref, 1.0);     // law = pressure: reference pressure
            pp_query_default("energetic.n", value.energetic.n, 1.0);             // law = pressure: pressure exponent
            const auto &nm = value.species.names;
            for (int n = 0; n < (int)nm.size(); ++n) { if (nm[n] == value.energetic.reactant) value.energetic.ir = n; if (nm[n] == value.energetic.product) value.energetic.ip = n; }
            if (value.energetic.ir < 0 || value.energetic.ip < 0) Util::Abort(INFO, "energetic: reactant / product must be species names");
            if (value.energetic.ir >= value.species.n0 || value.energetic.ip < value.species.n0)
                Util::Abort(INFO, "energetic: the reactant must be a species of phase 0 and the product a species of phase 1");
        }
        if (value.embedded.wall_normal != 0 && value.embedded.wall_normal != 1)
            Util::Abort(INFO, "solid.wall_normal must be 0 or 1");
        // [2026-09-24] force / pressure-probe time history, see Hydro2.H.
        pp_query_default("solid.force_int", value.solid_force_int, 0);
        pp.queryarr("solid.probe.x", value.probe_x);
        pp.queryarr("solid.probe.y", value.probe_y);
        if (value.probe_x.size() != value.probe_y.size())
            Util::Abort(INFO, "solid.probe.x and solid.probe.y must have the same length");
#if AMREX_SPACEDIM == 3
        pp.queryarr("solid.probe.z", value.probe_z);
        if (value.probe_z.size() != value.probe_x.size())
            Util::Abort(INFO, "in 3D solid.probe.z must be given, with the same length as solid.probe.x");
#endif
        if (value.embedded.apply)
        {
            if (value.embedded.brinkman <= 0.0)
                Util::Message(INFO, "embedded solid: auto momentum-projection wall (solid.brinkman <= 0)");
            else
                Util::Message(INFO, "embedded solid: solid.brinkman=", value.embedded.brinkman,
                              " implicit=", value.embedded.implicit, " wall_flux=",
                              value.embedded.wall_flux_auto ? std::string("auto") : std::to_string(value.embedded.wall_flux));
            // Skips pressure relaxation within the solid (acts likes Hydro do-not-solve type solid)
            value.embedded.relax_skip = 0.5;
        }

        // Artificial heat exchange (AHE) on per-phase internal energy rows.
        // Sources:
        //  Schmidmayer 2020 eq. 13 r-source
        //  Saurel-Petitpas-Berry 2009 Sec 5.3 Fig. 21
        // ahe.method: top-level switch that maps to (apply_ahe, ahe.form).
        //   0 = Sch20 stiff-limit only (no explicit AHE source)
        //   1 = Sch20 finite-mu r-source (+/- mu p_I Dp)
        //   2 = Sau09 sec.5.3 q*@u/@x form
        // When set (>= 0) overrides the legacy apply_ahe / ahe.form parses.
        int ahe_method_in = -1;
        pp_query_default("ahe.method",          ahe_method_in, -1);
        pp_query_default("apply_ahe",           value.apply_ahe,           0);
        pp_query_default("ahe.use_const_mu",    value.ahe_use_const_mu,    0);
        pp_query_default("ahe.mu_const",        value.ahe_mu_const,        0.0);
        pp_query_default("ahe.mu_scale",        value.ahe_mu_scale,        1.0);
        pp_query_default("ahe.mu_a",            value.ahe_mu_a,           -5.64e7);
        pp_query_default("ahe.mu_b",            value.ahe_mu_b,            5.34e3);
        pp_query_default("ahe.mu_c",            value.ahe_mu_c,           -25.6);
        pp_query_default("ahe.v_min",           value.ahe_v_min,           2.65e-4);
        pp_query_default("ahe.v_max",           value.ahe_v_max,           4.61e-4);
        pp_query_default("ahe.compression_only",value.ahe_compression_only,0);
        pp_query_default("ahe.apply_alpha_src", value.ahe_apply_alpha_src, 1);
        pp_query_default("ahe.max_frac",        value.ahe_max_frac,        0.1);
        pp_query_default("ahe.form",            value.ahe_form,            1);

        // Apply ahe.method override if the user set it.
        if (ahe_method_in == 0)
        {
            value.ahe_method = 0;
            value.apply_ahe  = 0;            // Sch20 stiff limit -- RelaxAndReinit only (default)
        }
        else if (ahe_method_in == 1)
        {
            value.ahe_method = 1;
            value.apply_ahe  = 1;
            value.ahe_form   = 0;            // Sch20 r-source (antisymmetric)
        }
        else if (ahe_method_in == 2)
        {
            value.ahe_method = 2;
            value.apply_ahe  = 1;
            value.ahe_form   = 1;            // Sau09 q*du/dx
        }
        //else (ahe_method_in == -1): unset -> fall back to legacy flags as parsed.

        // FLUID 0
        pp_query_required("mu0", value.mu0);            // linear viscosity coefficient
        pp_query_default("mu0_b", value.mu0_b, 0.0);    // bulk viscosity coefficient
        
        // FLUID 1
        pp_query_required("mu1", value.mu1);            // linear viscosity coefficient
        pp_query_default("mu1_b", value.mu1_b, 0.0);    // bulk viscosity coefficient
        // [2026-09-24] Fourier heat conduction.  Without it a viscous no-slip wall
        // has nowhere to put its dissipation heat (Pr -> infinity: the adiabatic-
        // wall recovery temperature grows without bound): NACA 0012 at M 0.8,
        // Re 500 reached T_wall/T_inf = 2.6 (physical ~1.1) by t = 0.5 and
        // crashed; at M 0.5, Re 5000 it crept to 1.30 by t = 14 (physical 1.04).
        // k = mu(eta) cp(eta) / Pr.  At embedded-solid faces (phi < 0.5 side)
        // the heat flux is zero (adiabatic wall).
        //
        // thermal.conduction (merge 2026-11-23: both branches had their own conduction under this key):
        //   1 = MIXTURE temperature (DropletDetonation, 2026-09-24/11-19): div(k grad T) with the face conductivity
        //       the mean of k(eta) = e k0 + (1-e) k1 (thermal.k0/k1) or mu(eta) cp(eta) / Pr (thermal.Pr), deposited in
        //       rho E and split to E0/E1 by volume fraction; adiabatic / isothermal embedded walls and the ablation
        //       in-depth coupling.  T is the legacy blended T unless temperature.phase_consistent = 1 (recommended
        //       with a liquid phase: the legacy T of a 300 K stiffened-gas liquid reads ~690 K).
        //   2 = PER-PHASE Fourier conduction (flames2, 2026-11-10): each phase conducts its own temperature
        //       T_k = (E_k - alpha_k pi_k)/(rho_k cv_k), face flux k_k min(alpha_k) dT_k/dx, phases below
        //       thermal.alpha_min excluded; the phases exchange heat only through thermal.exchange.  thermal.k0/k1
        //       required.  No embedded-wall handling (aborts with apply_embedded_solid / ablation.conduction).
        // thermal.k0 / thermal.k1 are the conductivities of PHASE 0 / PHASE 1 [W/(m K)], i.e. of whatever fluid the
        // deck puts in eos0 / eos1.  The MarmPaper decks have liquid = phase 0, the droplet decks gas = phase 0: do
        // not copy a k0/k1 pair between the two families without swapping it.
        pp_query_validate("thermal.conduction", value.thermal_conduction, {0, 1, 2});
        pp_query_default("entropy.on", value.entropy_on, 0);   // 1 = plot the entropy per unit volume (diagnostic only)
        pp_query_default("thermal.Pr", value.thermal_Pr, 0.72);
        // thermal.k0 / thermal.k1 (2026-11-19; default off = k = mu cp / Pr, unchanged): constant conductivity of each
        // phase [W/(m K)], k = alpha_0 k0 + alpha_1 k1.  One Pr for both phases gives a liquid k = mu_l cp_l / Pr that can
        // be 50x the real one (n-dodecane: mu 1.34e-3, stiffened-gas cp 4179 -> 7.8 W/(m K) against 0.14).  Give both.
        pp_query_default("thermal.k0", value.thermal_k0, -1.0);
        pp_query_default("thermal.k1", value.thermal_k1, -1.0);
        if ((value.thermal_k0 >= 0.0) != (value.thermal_k1 >= 0.0))
            Util::Abort(INFO, "thermal.k0 and thermal.k1: give both (>= 0) or neither");

        // EOS
        Solver::EOS::Tammann::Parse(value.eos0, pp, "eos0.");
        Solver::EOS::Tammann::Parse(value.eos1, pp, "eos1.");
        // liquid <-> vapour phase change by thermodynamic relaxation (phasechange.*, default off)
        PhaseChange::Parse(value.phasechange, pp, value.eos0.Gamma(), value.eos0.P0(), value.eos0.Cv(),
                           value.eos1.Gamma(), value.eos1.P0(), value.eos1.Cv());

        // PeleC EOS implementation (only works for non-tamman fluids "p0=0")
        //   eos.backend = tammann   (default; equivalent to "native")
        //   eos.backend = pelephysics
        std::string eos_backend_str = "tammann";
        pp_query_default("eos.backend", eos_backend_str, "tammann");
        Solver::EOS::SetBackend(eos_backend_str);

        // INTERACTIONS
        pp_query_default("sigma", value.sigma, 0.0);            // Surface tension condition (= sigma_water when marmottant=1)
        // Marmottant (2005) coated-bubble surface tension -- ADVECTED-SHELL form.
        // sigma is a function of the shell's areal strain Gamma_0/Gamma, carried by
        // the conserved shell density c (see Hydro2.H).  No curvature is involved,
        // so the eta/kappa/sigma smoothing passes and the height-function curvature
        // that the old sigma(R(x)) path needed are GONE.
        pp_query_default("marmottant", value.marmottant, 0);                       // 0=const sigma, 1=Marmottant sigma(Gamma)
        pp_query_default("marmottant.R_buckling", value.marmottant_R_buckling, 0.0);   // buckling radius [m]
        pp_query_default("marmottant.chi",        value.marmottant_chi, 0.0);          // shell elastic modulus [N/m]
        pp_query_default("marmottant.sigma_break",value.marmottant_sigma_break, 1.0e30); // rupture tension (default huge = never rupture)
        pp_query_default("marmottant.R0",         value.marmottant_R0, 0.0);           // reference radius: shell starts UNSTRAINED here
        if (value.marmottant)
        {
            if (!(value.marmottant_R0 > 0.0))
                Util::Abort(INFO, "marmottant=1 requires marmottant.R0 > 0 "
                                  "(the radius at which the shell is unstrained, Gamma=Gamma_0=1)");
            if (!(value.marmottant_R_buckling > 0.0))
                Util::Abort(INFO, "marmottant=1 requires marmottant.R_buckling > 0");
            // Gamma_0 == 1 by construction (c is initialized to |grad eta|), so the
            // buckling areal density follows from Marmottant's A/A_buck = (R/R_buck)^2
            // evaluated at the reference state.  This reproduces the previous
            // calibration exactly:  sigma(t=0) = chi ((R0/R_buck)^2 - 1).
            value.marmottant_Gamma_buck = (value.marmottant_R0 * value.marmottant_R0)
                                        / (value.marmottant_R_buckling * value.marmottant_R_buckling);
            Util::Message(INFO, "Marmottant ON (advected shell): R0=", value.marmottant_R0,
                          " R_buckling=", value.marmottant_R_buckling,
                          " chi=", value.marmottant_chi,
                          " Gamma_buck=", value.marmottant_Gamma_buck,
                          " sigma(t=0)=", value.marmottant_chi * (value.marmottant_Gamma_buck - 1.0),
                          " sigma_break=", value.marmottant_sigma_break,
                          " sigma_water=", value.sigma);
        }
        pp_query_default("Dv", value.Dv, 0.0);                  // Vapor Diffusivity
        pp_query_default("Lv", value.Lv, 0.0);                  // Latent heat of vaporization [J/kg] (evaporative cooling sink)
        pp_query_required("epsilon", value.epsilon);            // diffuse interface thickness Y_infinity
        // interface.gem_min [1/m]: interface gate on |grad eta| (see g_gem_min).  Default
        // 1e-13/epsilon (scale-free); interface.gem_min = 1e-10 restores the old fixed gate.
        value.interface_gem_min = 1.0e-13 / value.epsilon;
        pp_query_default("interface.gem_min", value.interface_gem_min, value.interface_gem_min);
        g_gem_min = value.interface_gem_min;
        Util::Message(INFO, "interface gate |grad eta| > ", g_gem_min, " 1/m (gem_min*epsilon = ", g_gem_min * value.epsilon, ")");
        pp_query_default("Y_infinity", value.Y_infinity, 0.0);  // Far Field Vapor Mass Fraction
        // Film evaporation model (default 0 = original surrogate; see Hydro2.H)
        pp_query_default("vaporization.model", value.vap_model, 0);
        pp_query_default("vaporization.legacy_energy", value.vap_legacy_energy, 0);
        if (value.vap_model < 0 || value.vap_model > 3) Util::Abort(INFO, "vaporization.model must be 0..3, got ", value.vap_model);
        if (value.vap_model >= 1)
        {
            if (!value.apply_vaporization) Util::Abort(INFO, "vaporization.model >= 1 needs apply_vaporization = 1");
            pp_query_required("vaporization.film_thickness", value.vap_delta);
            pp_query_default("vaporization.film_thickness_T", value.vap_delta_T, value.vap_delta);
            if (value.vap_model >= 2) pp_query_required("vaporization.k_gas", value.vap_k_gas);
            pp_query_default("vaporization.cp_vapor", value.vap_cp_vapor, 2000.0);
            pp_query_default("vaporization.W_vapor", value.vap_W_vapor, 18.015);
            pp_query_default("vaporization.W_gas", value.vap_W_gas, 28.965);
            std::string ps = "buck";
            pp.query("vaporization.psat", ps);
            if (ps == "buck") value.vap_psat = 0;
            else if (ps == "clausius_clapeyron") value.vap_psat = 1;
            else if (ps == "antoine") value.vap_psat = 2;
            else Util::Abort(INFO, "vaporization.psat must be buck, clausius_clapeyron or antoine, got ", ps);
            if (value.vap_psat == 2)    // NIST WebBook form: log10(p / bar) = A - B / (T / K + C)
            {
                pp_query_required("vaporization.antoine_A", value.vap_antoine_A);
                pp_query_required("vaporization.antoine_B", value.vap_antoine_B);
                pp_query_required("vaporization.antoine_C", value.vap_antoine_C);
            }
            pp_query_default("vaporization.T_ref", value.vap_T_ref, 373.15);
            pp_query_default("vaporization.p_ref", value.vap_p_ref, 101325.0);
            pp_query_default("vaporization.q_liquid", value.vap_q_liquid, 0.0);
            pp_query_default("vaporization.q_gas", value.vap_q_gas, 0.0);
            pp_query_default("vaporization.alpha_min", value.vap_alpha_min, 0.05);
            pp_query_default("vaporization.T_gas_max", value.vap_T_gas_max, 1.0e30);
            pp_query_default("vaporization.volume_source", value.vap_volume_source, 0);
            pp_query_default("vaporization.rho_gas_min", value.vap_rho_gas_min, 0.01);
            if (value.vap_alpha_min < 0.0 || value.vap_alpha_min >= 0.5) Util::Abort(INFO, "vaporization.alpha_min must be in [0, 0.5)");
            if (value.vap_delta <= 0.0 || value.vap_delta_T <= 0.0) Util::Abort(INFO, "vaporization.film_thickness must be > 0");
            if (value.Lv <= 0.0) Util::Abort(INFO, "vaporization.model >= 1 needs Lv > 0");
            if (value.vap_model != 2 && value.Dv <= 0.0) Util::Abort(INFO, "vaporization.model 1/3 needs Dv > 0");
        }
        // [2026-11-18] EVAPORATION INTO A SPECIES (default off).  With species.on = 1 the evaporated mass m_dot (any
        // vaporization model) is booked on named species instead of the phase totals only:
        //     vaporization.vapor_species  = <name>   gas-phase (phase 0) species that receives m_dot   (the fuel vapour)
        //     vaporization.liquid_species = <name>   liquid-phase (phase 1) species that loses it      (optional; with a
        //                                            single liquid species the phase closure does it anyway)
        // Without vapor_species the evaporated mass goes to the gas phase total and the species closure hands it to the
        // MAJORITY gas species (air), i.e. the vapour is invisible to the chemistry.
        //     vaporization.Y_inf_local = 1 (default when vapor_species is given): the film models use the cell's own
        //     vapour mass fraction Y_v = m_v / (alpha rho)_g as the far-field value in B_M = (Y_s - Y_inf)/(1 - Y_s)
        //     instead of the constant Y_infinity, so evaporation slows and stops as the gas saturates.
        {
            std::string vs, ls;
            pp.query("vaporization.vapor_species", vs);
            pp.query("vaporization.liquid_species", ls);
            if (!vs.empty() || !ls.empty())
            {
                if (!value.species.on) Util::Abort(INFO, "vaporization.vapor_species / liquid_species need species.on = 1");
                if (!value.apply_vaporization) Util::Abort(INFO, "vaporization.vapor_species / liquid_species need apply_vaporization = 1");
                for (int n = 0; n < value.species.ns(); ++n)
                {
                    if (!vs.empty() && value.species.names[n] == vs) value.vap_sp_gas = n;
                    if (!ls.empty() && value.species.names[n] == ls) value.vap_sp_liq = n;
                }
                if (!vs.empty() && (value.vap_sp_gas < 0 || value.vap_sp_gas >= value.species.n0))
                    Util::Abort(INFO, "vaporization.vapor_species = ", vs, " must be a species of phase 0 (the gas)");
                if (!ls.empty() && (value.vap_sp_liq < value.species.n0))
                    Util::Abort(INFO, "vaporization.liquid_species = ", ls, " must be a species of phase 1 (the liquid)");
            }
            pp_query_default("vaporization.Y_inf_local", value.vap_Yinf_local, (value.vap_sp_gas >= 0) ? 1 : 0);
            if (value.vap_Yinf_local && value.vap_sp_gas < 0) Util::Abort(INFO, "vaporization.Y_inf_local = 1 needs vaporization.vapor_species");
        }
        pp_query_default("Mob", value.Mob_user, 0.0);           // CH mobility scale M0: M = M0 * epsilon^2
        if (value.epsilon <= 0.0)
        {
            Util::Abort(INFO, "epsilon must be positive for Hydro2 Cahn-Hilliard mobility; got ", value.epsilon);
        }

        // IC pressure equalization during initialization step to prevenent spurious oscilation at the start
        pp_query_default("equalize_ic_pressure", value.equalize_ic_pressure, 1);   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        
        // Newton diagnostic for stiff pressure relaxation.
        pp_query_default("nan_check", value.nan_check, 0);  // 1 = per-cell NaN/Inf guard (slow; debugging only)
        pp_query_default("verbose", value.verbose, 0);    // 1 = per-call informational messages

        // Opt-in bubble time series in thermo.dat (see Hydro2.H).  Sampling a
        // kHz bubble oscillation from 3D plotfiles would need hundreds of GB per
        // run; these scalars cost one masked pass over the grid per sample.
        {
            amrex::ParmParse pp_amr("amr");
            int tp_int = -1;
            Set::Scalar tp_dt = -1.0;
            pp_amr.query("thermo.plot_int", tp_int);
            pp_amr.query("thermo.plot_dt", tp_dt);
            if (tp_int > 0 || tp_dt > 0.0)
            {
                value.RegisterIntegratedVariable(&value.thermo_gas_volume,     "gas_volume");
                value.RegisterIntegratedVariable(&value.thermo_gas_pressure,   "gas_pressure_int");
                value.RegisterIntegratedVariable(&value.thermo_kinetic_energy, "kinetic_energy");
                value.RegisterIntegratedVariable(&value.thermo_interface_area, "interface_area");
                Util::Message(INFO, "thermo.dat: gas_volume, gas_pressure_int, kinetic_energy, interface_area");
            }
        }
        // The kappa_s surface-viscous stress is sub-cycled together with the
        // bulk viscosity: visc_subcycle = 1 also advances it in its own
        // sub-cycle (N = dt/dt_visc sub-steps on the band, nearest-neighbour
        // FillBoundary per sub-step, no global reductions) instead of letting
        // its dx^3 stability limit throttle the GLOBAL timestep.  When on, the
        // shell viscous term is removed from Omega and from the dt limiter.
        // visc_subcycle = 0 restores the dt-limited path for both.
        value.shell_visc_subcycle = value.visc_subcycle;
        // Guard: if nu_s ever blows up, cap the sub-step count rather than
        // spinning forever inside one hydro step.  512 is ~6 refinement levels
        // past the measured L5 value of 21 (N ~ 1/dx^2, so it quadruples/level).
        pp_query_default("shell.kappa_s_subcycle_max", value.shell_visc_subcycle_max, 8192);   // [2026-11-24] default = the 06 Cattaneo configuration (was 512)
        // Level-set slip of the shell after each step (2026-11-10, 06 Cattaneo tests):
        //   0 = off: Gamma moves with the fluid only (Gamma row, raw div_s u)
        //   5 = option 5: implicit upwind shift + exp(-kappa s dt) area stretch (prints SLIP5)
        //   6 = option 6: conservative flux of Gamma|grad eta| (prints SLIP6)
        //   7 = option 7: MATERIAL shell.  The field is m = Gamma |grad psi|, psi the volume
        //       fraction at fixed reference densities (a function of the mass fraction, so its
        //       level sets move with the fluid; relaxation cannot change it).  For any material
        //       level set D Gamma/Dt = -Gamma div_s u and D|grad psi|/Dt = -|grad psi| n.grad u.n,
        //       so exactly  dm/dt + div(m u) = 0:  conservative advection, no slip, no curvature.
        //       Gamma = m/|grad psi| is read only for sigma(Gamma) (ShellPsiGrad).
        //  11 = option 11 (2026-11-11): CONSERVATIVE level-set shell.  The field is the shell amount per
        //       volume m = Gamma |grad eta| (option 7 storage, eta instead of psi).  The level sets of eta
        //       move with V = u + s n, and for any field carried by V exactly
        //           dm/dt + div(m V) = 0
        //       (D|grad eta|/Dt = -|grad eta| n.grad V.n, D Gamma/Dt = -Gamma div_s V, div_s V = div_s u + kappa s).
        //       Fluid part: option 7's upwind flux on u*.  Slip part (ShellSlipMass): upwind flux of m with
        //       s n, s n = w grad eta/|grad eta|^2, w = -(eta_t + u.grad eta); explicit MUSCL, sub-cycled to CFL .45
        //       (positive, total shell conserved).  No curvature: the area law comes out of the divergence.
        //       Option 5's exp(-kappa s dt) on Gamma created shell on trace level sets collapsing to a point
        //       (06 c12 bubble centre, Gamma 1e11 -> interface, 2026-11-11); here such a level set carries
        //       only its own (vanishing) share of m.  Gamma = m/|grad eta| is read only for sigma(Gamma).
        //  12 = option 12: option 11 WITHOUT the slip flux -- m = Gamma |grad eta| carried with the fluid only
        //       (conservative, option 7's flux), Gamma = m/|grad eta| read on the eta band.  The slip flux of
        //       option 11 lost Gamma(eta=.5) (1.0 -> .80 by 1.8 us, 06 c12; 2nd order .85): s = w/|grad eta| is
        //       steep across the band and the discrete div(m s n) does not track |grad eta|; Gamma itself only
        //       feels kappa s, small at eta = .5 (option 5 and slip-free m both keep Gamma/exact ~ 1).
        //  13 = option 13: option 5 with the shift done on the LEVEL SET.  Gamma lives on its own eta level
        //       set; the departure value of cell i is Gamma_old where the fluid-advected old field
        //       eta* = eta^n - dt u.grad eta equals eta^{n+1}_i, found by linear interpolation in eta* between
        //       cell i and its upwind neighbour (theta = (eta^{n+1}_i - eta*_i)/(eta*_up - eta*_i) in [0,1]),
        //       then the exact exp(-kappa s dt) stretch.  Option 5's implicit upwind weights ignore which level
        //       the donor sits on: at the 06 c12 160 kPa deepest collapse they copied Gamma ~1e11 from the
        //       eta ~ 1e-4 bubble-centre level set into the eta = 0.5 cell (permanently buckled shell).
        // Locally (RADIAL slab) slip 5 and 6 split the slab columns at 19.2 cells/R0; 0 did not.
        pp_query_default("shell.slip", value.shell_slip, 12);   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        // shell.slip5_rupture (2026-11-12, option 5 only): where the level set through a band cell has
        // shrunk to a point -- eta has an extremum ALONG THE NORMAL inside the cell (one-sided normal
        // differences change sign: bubble centre after a deep collapse, isolated tail extrema, pinch-off) --
        // that level set has no area left, so it carries no shell: Gamma = 0 (polymer chains torn apart).
        // Gamma = 0 means a clean interface (sigma_water, the Marmottant Gamma -> 0+ limit) and no shell
        // viscosity.  Intact neighbours re-cover the cell through the slip's upwind transport (rejoining).
        pp_query_default("shell.slip5_rupture", value.shell_slip5_rupture, 0);
        g_shell_rupture = value.shell_slip5_rupture;
        // option 11 face slip velocity: 0 = mean of the two cells, 1 = the donor (upwind) cell's own s n,
    // 2 = flux Gamma_donor (w n)_face (m s n = Gamma w n: both factors smooth across the band, s = w/|grad eta| is not)
        pp_query_default("shell.slip11_face", value.shell_slip11_face, 0);
        // relax.opt8 (option 8, 2026-11-10): the fixed-alpha relaxation branches (cells beyond the
        // cutoff, or with a floored pre-relax pressure) used to put the gas at its EOS floor whenever
        // the common pressure was below it (liquid in tension): for an ideal gas p = 0 means e = 0,
        // i.e. a 0 K gas, and its thermal energy went to the liquid.  Repeated each rarefaction this
        // built cold, crushed trace gas (rho_g 20-50x the core) next to the band and the 160 kPa runs
        // blew up there.  With opt8 = 1 such a cell has no admissible equal-pressure state, so the
        // gas KEEPS its own energy (its own p >= 0) and the liquid takes the rest of rho e (carries
        // the tension): pressure non-equilibrium for that step (Schmidmayer et al. 2023, mu = 0
        // limit); the full isobaric relaxation is tried again every step.  E0 + E1 = rho e exactly.
        pp_query_validate("relax.opt8", value.relax_opt8, {1, 0, 2});   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        // relax.trace_pT (2026-11-11): a liquid cell beyond the cutoff (eta > 1 - cutoff) that holds gas
        // mass is treated as liquid with dispersed micro-gas, in pressure AND temperature equilibrium with
        // it (the 4-equation / HRM state, solved from the conserved masses and rho e only, so no trace E_k
        // noise enters).  Before, alpha was frozen and the gas was set to the liquid pressure at fixed
        // volume: compressed trace gas could not re-expand as the pressure fell, the closure removed its
        // heat instead (isochoric), and it went to ~1 K at 6-18 kg/m3; on re-entering the band the
        // relaxation reheated it and it expanded ~25x in one step (06 160 kPa c19 crash at 6.70 us, with
        // or without the shell slip).  Applied only when the frozen-alpha common pressure is >= 0: an
        // ideal gas cannot be in equilibrium with liquid in tension (that needs the nuclei's own Laplace
        // pressure), so liquid in tension keeps the previous guard and holds its tension.
        pp_query_validate("relax.trace_pT", value.relax_trace_pT, {3, 0, 1, 2});   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        // relax.blake (2026-11-12): an isolated gas pocket in liquid (eta > 1/2 and the gas fraction peaks along
        // the eta normal, i.e. eta has a minimum there: gas not connected to a resolved interface) is a sub-grid
        // nucleus of radius r = (3 alpha_g dV / 4 pi)^(1/3).  The 6-eq relaxation has no capillary term, so in
        // liquid tension it expanded such gas to the liquid pressure (0.01% -> 5% gas, 160 kPa slab, 2.8 us).
        // A real nucleus carries its Laplace pressure 2 sigma/r (MPa at nm radii) and only grows past the Blake
        // threshold p_l < p_B = -4 sigma / (3 R_c), R_c = sqrt(3 p_g r^3 / (2 sigma)) (isothermal gas, no vapour).
        // Below it the nucleus volume is held (stiff limit of 2 sigma/r >> |dp|) and only the temperature is
        // shared with the liquid (same closed form as trace_pT = 3; E0 + E1 = rho e).  Above it the cell relaxes
        // normally (cavitation).  sigma = clean-water value (sigma).  No tunable parameter.
        pp_query_validate("relax.blake", value.relax_blake, {0, 1});
        // relax.trace_pT = 3: alpha and phase densities stay frozen (no trace-gas compressibility in the
        // liquid), the trace gas takes the common temperature at fixed volume (energy conserved, closed form).
        // relax.trace_pT = 2: same cells and same admissibility (frozen-alpha p >= 0), but instead of the p-T state
        // they go through the ordinary closure of the band cells (each phase isentropic to the common pressure plus
        // the same specific heat q, Dj = 0): pressure equilibrium, gas keeps its own temperature.   // 2: trace gas keeps its energy in every beyond-cutoff liquid cell
        // relax.opt9 (option 9, 2026-11-10): which cells skip the isobaric (alpha-free) relaxation.
        // Default: eta beyond the cutoff (eta < cutoff or > 1 - cutoff) is treated as a single fluid with
        // alpha FROZEN -- so the liquid can hold tension, but its pressure is then set by the advected
        // eta: water's rho c^2 turns an advection error of 2e-5 in eta into ~44 kPa, which nothing
        // removes and which grew into the 160 kPa band-tail blow-up.  With opt9 = 1 alpha is frozen only
        // where a phase has NO mass; every cell holding both phases gets the infinite-rate relaxation of
        // Saurel et al. 2009 / Sch20 (alpha free), so a trace absorbs eta errors.  Pure liquid (eta = 1,
        // no gas mass) still carries tension.  Cells with a floored pre-relax pressure keep the guard
        // (combine with relax.opt8).
        pp_query_default("relax.opt9", value.relax_opt9, 0);
        // relax.sch20 (2026-11-11): skip the isentropic + mass-specific-heat closure and use the Sch20 /
        // Saurel et al. 2009 closure (each phase e_k(p, v_k) - e_k^0 + p (v_k - v_k^0) = 0, the path below
        // that is otherwise only the fallback).  The default adds the energy residual as the SAME specific
        // heat q to every phase: q of order the gas's own c_v T (2e5 J/kg) is negligible for the liquid
        // but zeroes (q < 0) or multiplies (q > 0) the trace gas energy in band-tail cells (160 kPa
        // diagnosis: tail gas at 34 K, then 1e5 K, before the blow-up).
        pp_query_default("relax.sch20", value.relax_sch20, 0);
        // face.two_pressure (option 10, 2026-11-11): the face reconstruction assumed ONE pressure
        // (p_mix) for both phases, but the cells keep the gas at p_liq + Delta (Laplace jump).  Where the
        // liquid is in tension p_mix < 0 while the gas pressure is still positive, so the face put the gas
        // at its EOS floor (p = 0, zero energy) and gas mass crossed with no energy -> cold trace gas in
        // the liquid-side band tail (160 kPa blow-up).  With 1 the faces also reconstruct dp = p_gas - p_liq
        // and rebuild each phase at its own pressure (alpha-weighted sum = p_mix exactly; cell-centre
        // roundtrip exact for both phase energies).  See Limiter.H Primitive::dp.
        pp_query_default("face.two_pressure", value.face_two_pressure, 1);   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        if (value.shell_slip != 0 && value.shell_slip != 5 && value.shell_slip != 6 && value.shell_slip != 7 && value.shell_slip != 11 && value.shell_slip != 12 && value.shell_slip != 13 && value.shell_slip != 15)
            Util::Abort(INFO, "shell.slip must be 0, 5, 6, 7, 11, 12, 13 or 15");
        pp_query_default("shell.kappa_s_verbose", value.shell_visc_verbose, 0);
        // Where the shell viscous dissipation (heat) is booked between the phases:
        //   0 = by volume fraction (default, previous behaviour)
        //   2 = by thermal effusivity e_k = sqrt(k_k rho_k cp_k), the heat-flux split
        //       of a heat source on the contact of two semi-infinite media
        //       (air/water: 99.65% to the water).  Needs thermal.k0 / thermal.k1.
        //   (1 = all to the liquid was removed 2026-11-05: identical to 2 to 4 digits.)
        pp_query_validate("shell.heat_split", value.shell_heat_split, {0, 2});
        // HEAT CONDUCTION (2026-10-28).  Temperatures are the stiffened-gas
        //     T_k = (p_k + pi_k) / ((gamma_k - 1) rho_k cv_k) = (E_k - alpha_k pi_k) / (alpha_k rho_k cv_k),
        // so eos{0,1}.cv must be CALIBRATED to give equal temperatures at the
        // initial state (cv enters nothing but T).
        //   thermal.conduction = 1 : Fourier conduction inside each phase,
        //        d(alpha rho e)_k/dt += div(alpha_k k_k grad T_k)   (also in rho E)
        //   thermal.exchange   = 1 : gas-liquid heat exchange in mixed cells,
        //        Q = H (T_1 - T_0) into phase 0, H = |grad eta|^2 / (eta (1-eta) (alpha_0/k_0 + alpha_1/k_1))
        //        (conduction across the two phase layers of a cell of band width
        //         eps_loc = 2 eta (1-eta)/|grad eta|, interface area |grad eta|)
        //                      = 2 : same with a constant coefficient, H = thermal.h |grad eta|
        //                      = 3 : instantaneous temperature equilibrium, T_0 = T_1
        //        (H -> infinity), in cells where both phases exceed thermal.alpha_min
        //   Q is applied once per step as an exact exponential relaxation.
        // SHELL AREA RATE (2026-11-02, was shell.rate_local).  Every liquid-side band cell
        // (eta > 1/2) maps its OWN fluid surface dilatation to the interface area rate by
        // flux-tube continuity of the incompressible liquid (LiquidInterfaceRate):
        //     r = [div_s u - kappa u_n] + (kappa - 2 K d) u_n / (1 - kappa d + K d^2)^2,
        // d = eps atanh(2 eta - 1) the distance to eta = 1/2, kappa and K the mean and Gaussian
        // curvature of the local level set (S = grad n), |k_i d| capped at 1/2.  Exact for
        // normal flow on any surface (2 Rdot/R on a sphere).  Used in the Gamma row
        // (DGamma/Dt = -Gamma r) and in the viscous shell rate.  Purely local: nothing is
        // copied along the normal, so it is safe through topology change.  Gas-side cells
        // keep the raw div_s u.  (Without it the liquid-half parcels dilate as (R/r)^3 and
        // Gamma lags there by 25-40%.)  Gamma is extended into the adjacent bulk
        // (ShellBandUniform); band/bulk threshold = cutoff.
        // (thermal.conduction is parsed with the fluid parameters above: 1 = mixture T, 2 = per-phase.)
        //                      = 4 : JOINT p-T relaxation (2026-11-11): in cells where both phases exceed
        //        thermal.alpha_min the pressure relaxation solves for ONE state with equal pressures
        //        (gas at p + Dj), equal temperatures, and conserved mass, volume and energy -- the
        //        p-T equilibrium of the 4-equation (HRM) limit (Saurel / Pelanti & Shyue).  For two
        //        stiffened gases it is a quadratic in p.  ThermalExchange does nothing in this mode.
        pp_query_validate("thermal.exchange",   value.thermal_exchange,   {0, 1, 2, 3, 4});
        pp_query_default("thermal.diag",        value.thermal_diag,       0);
        // thermal.dt_limit (2026-11-14): Fourier conduction is EXPLICIT inside the RK right-hand side (it is NOT
        // sub-cycled), so its diffusion limit dt <= cfl_v dx^2 / (2 d alpha_k), alpha_k = k_k / (rho_k cv_k), must bound
        // the global step.  visc_subcycle = 1 used to zero the whole nu_total (bulk viscosity AND conduction): in the
        // expanded gas core of the 160 kPa slabs the diffusion number reached 0.265-0.268 and an odd-even T/p mode grew
        // at the RADIAL axis -> NaN (R0/dx 19 at 9.08 us, 25 at 8.03 us).  1 (default) = conduction always enters
        // dt_viscous; 0 = previous behaviour.
        pp_query_default("thermal.dt_limit", value.thermal_dt_limit, 1);
        // thermal.subcycle (2026-11-26, gate flag, default 0): the per-phase Fourier conduction (thermal.conduction = 2)
        // is taken OUT of the RK right-hand side and advanced as its own operator after the hyperbolic step (Lie split,
        // same place as the bulk / shell viscous sub-cycles): N forward-Euler sub-steps of dtau = dt/N on E_0, E_1 and
        // rho E, with eta, rho_k fixed.  N from the exact max-principle bound of the discrete operator,
        //     dtau <= thermal.subcycle_cfl / max_cell sum_faces k_k min(alpha_k) / (alpha_k rho_k cv_k dx^2),
        // so the conduction diffusivity no longer bounds the global step (thermal.dt_limit is then not needed: the
        // 06Exp INCLINE run sat at 40x below the acoustic dt on the finest level).  Same discrete fluxes as the RHS
        // form (conservative, trace phases below thermal.alpha_min do not conduct).  Mode 2 only.
        pp_query_default("thermal.subcycle", value.thermal_subcycle, 0);
        pp_query_default("thermal.subcycle_cfl", value.thermal_subcycle_cfl, 0.9);
        pp_query_default("thermal.subcycle_max", value.thermal_subcycle_max, 10000);
        pp_query_default("thermal.subcycle_verbose", value.thermal_subcycle_verbose, 0);
        if (value.thermal_subcycle && value.thermal_conduction != 2)
            Util::Abort(INFO, "thermal.subcycle = 1 sub-cycles the per-phase conduction: it needs thermal.conduction = 2");
        pp_query_default("thermal.alpha_min",   value.thermal_alpha_min,  1.0e-4);   // [2026-11-24] default = the 06 Cattaneo configuration (was 1e-3)  // phase fraction below which a phase takes no part
        if (value.thermal_exchange == 2) pp_query_required("thermal.h", value.thermal_h);   // [W/m^2/K]
        if (value.shell_heat_split == 2 || value.thermal_conduction == 2 || value.thermal_exchange == 1 || value.thermal_exchange == 2)
        {
            pp_query_required("thermal.k0", value.thermal_k0);   // phase-0 thermal conductivity [W/m/K]
            pp_query_required("thermal.k1", value.thermal_k1);   // phase-1 thermal conductivity [W/m/K]
            if (value.thermal_k0 < 0.0 || value.thermal_k1 < 0.0)
                Util::Abort(INFO, "thermal.k0 / thermal.k1 must be >= 0 (needed by thermal.conduction = 2, thermal.exchange 1/2 or shell.heat_split = 2)");
        }
        // The per-phase heat options belong to thermal.conduction = 2 (or no conduction); with the mixture-T mode they
        // would act on a different temperature field.
        if (value.thermal_conduction == 1 && (value.thermal_exchange != 0 || value.shell_heat_split == 2))
            Util::Abort(INFO, "thermal.exchange / shell.heat_split = 2 are per-phase options: use thermal.conduction = 2 (not 1)");
        if (value.thermal_conduction == 2 && (value.embedded.apply || (value.ablation.on && value.ablation.conduction)))
            Util::Abort(INFO, "thermal.conduction = 2 (per phase) has no embedded-wall / ablation coupling: use thermal.conduction = 1");
        if (value.thermal_conduction == 1 && !value.T_phase_consistent && (value.eos0.P0() > 0.0 || value.eos1.P0() > 0.0))
            Util::Warning(INFO, "thermal.conduction = 1 with a stiffened (liquid) phase conducts the legacy blended T; "
                          "set temperature.phase_consistent = 1 for the phase temperature");
        // Density floor for the nu_s stability estimate (NOT for the physics).
        // Defaults to 1.0 kg/m^3, the gas reference density of the Sch20 decks.
        pp_query_default("shell.nus_rho_floor", value.shell_nus_rho_floor, 1.0);
        // shell.foot_rate (2026-11-18): interface area rate of the shell viscous jump evaluated ON the interface.
        //   0 (default): every liquid-side band cell maps its own stretch through the flux tube
        //       (LiquidInterfaceRate, gain ~ (1 + kappa d/2)^3, d = eps atanh(2 eta - 1)).
        //   1: every band cell takes r and the curvature Kt from its foot point x - d n on eta = 1/2, trilinearly
        //       interpolated from the band core (0.02 < eta < 0.98), where r = div_s u of the fluid there.
        //   Why: the flux-tube map is a Taylor expansion in kappa d.  At a grid-axis tip (cubic l=4 mode in a
        //   collapse, kappa ~ 4 x 2/r) d at eta ~ 0.999 (~3.8 eps) exceeds 1/kappa; the kd <= 1/2 cap and the
        //   den >= 1/4 guard then return 16-190x gain with the wrong sign, and the local tail curvature is 5-20x
        //   the interface value.  The path integral carries that J_F (-0.3..-0.4 MPa) into eta ~ 0.998 cells,
        //   relaxation inflates the trace gas, an axis finger grows and the conduction dt collapses (C4F10
        //   160 kPa octant, 3.235 us).  With 1 the jump is a function of the interface only, so the band carries
        //   the same J at every eta.  Cells whose foot stencil is unavailable (coarse-fine edge, thin neck) keep
        //   the local map.  Used in the sub-cycled viscous force and in the relaxation path integral.
        pp_query_validate("shell.foot_rate", value.shell_foot_rate, {1, 0});   // [2026-11-24] default = the 06 Cattaneo configuration (was 0)
        // shell.tension_floor [N/m]: TEST CLAMP on the total shell tension,
        //   sigma_tot = max(sigma_eff + (kappa_s - mu_s) div_s u, tension_floor),
        // applied to the viscous normal force + tangential stress (sub-cycle) and to the
        // relaxation jump.  Off by default.  It removes the compressive viscous resistance
        // wherever it is active (expect a deeper R_min).
        pp_query_default("shell.tension_floor", value.shell_tension_floor, -1.0e300);
        if (value.shell_tension_floor > -1.0e299)
            Util::Message(INFO, "shell.tension_floor ON: sigma_tot >= ", value.shell_tension_floor, " N/m");
        // shell.kappa_s_alpha [s]: rate-dependent (shear-thinning) shell dilatational viscosity,
        //   kappa_eff = kappa_s / (1 + alpha |div_s u| / 2),
        // the Cross law of Doinikov, Haac & Dayton (Ultrasonics 49:269, 2009) written with the
        // local surface dilatation rate (div_s u = 2 Rdot/R on a sphere).  Their fit to lipid
        // contrast agents gives alpha = 4e-6 s.  0 = off (constant kappa_s, Marmottant 2005).
        pp_query_default("shell.kappa_s_alpha", value.shell_kappa_s_alpha, 0.0);
        if (value.shell_kappa_s_alpha > 0.0)
            Util::Message(INFO, "shell.kappa_s_alpha ON: kappa_eff = kappa_s/(1 + alpha|div_s u|/2), alpha = ",
                          value.shell_kappa_s_alpha, " s");
        // INTERFACE JUMP (always on with apply_surface_tension = 1; the flags jump.consistent,
        // jump.tangential, capillary_jump, shell.kappa_s_jump, jump.curvature(_y,_iso,_cp) and
        // jump.R0 were removed 2026-11-05).  The momentum force and the pressure relaxation
        // see the SAME per-cell jump
        //     Delta = sigma_eff kappa  +  [(kappa_s - mu_s) + 2 mu_s/(D-1)] (div_s u) kappa,
        // sigma_eff the constant sigma or Marmottant sigma(Gamma), div_s u the shell area
        // rate (SHELL AREA RATE above).  For a sphere this is 2 sigma/R + 4 kappa_s Rdot/R^2
        // (Keller-Miksis).
        //  * Momentum: the NORMAL part enters as F_n = -Delta grad(eta) (static part in the RHS,
        //    viscous part in the sub-cycler); the capillary tensor div(Omega) keeps only its
        //    TANGENTIAL part (Marangoni stress of sigma(Gamma), tangential viscous stress).
        //  * Relaxation: band cells relax to p_gas - p_liq = Delta instead of p_gas = p_liq,
        //    so p_mix = p_liq + (1 - eta) Delta and -grad(p_mix) + F_n = -grad(p_liq) wherever
        //    Delta is uniform: a bubble at rest is an exact discrete equilibrium, and the band
        //    gas sits at the core pressure (without the jump it sat at the mixture pressure and
        //    absorbed part of the shell stress, kappa_eff ~ 1 - 1.9 eps/R).
        //  * Energy: the total energy gains u.F, plus the viscous dissipation
        //    keff |grad eta| (div_s u)^2 as heat; the surface energy is not stored in the fluid.
        //  * IC: the phases keep their own IC pressures (the Laplace jump).
        // CURVATURE kappa (TensorCurvature): kappa = div(n), n = grad Y/|grad Y|, Y = (alpha rho)_0/rho0_ref the liquid mass (2026-11-13; 11-02..11-13 psi below), with
        //     psi = (a rho)_0/rho0_ref / ((a rho)_0/rho0_ref + (a rho)_1/rho1_ref),
        // the volume fraction at fixed reference densities (mean pure-phase densities of the
        // IC).  The relaxation conserves each (a rho)_k, so it cannot change psi: with
        // kappa(eta) the loop relaxation -> alpha -> kappa -> J -> relaxation diverged in the
        // band tail under AMR subcycling (sigma = 500 Light_Mili, step 1, 2026-11-03).  Every
        // derivative is a central difference averaged over the transverse neighbours with
        // [1,2,1]/4 weights (isotropic 3^D stencil): the compact face-neighbour form has a
        // grid-axis truncation error that gave kappa < 0 -> J < 0 on the axis rows in the
        // band tail (Light_Mili rebound crash, 2026-11-04).  Any shape; the Gamma-based
        // sphere-only curvature was removed (no l >= 2 restoring force).
        {
            const bool visc = (value.shell_kappa_s != 0.0 || value.shell_mu_s != 0.0);
            if (visc && value.apply_surface_tension && !value.shell_visc_subcycle)
                Util::Abort(INFO, "a viscous shell (shell.kappa_s / shell.mu_s) needs visc_subcycle = 1 "
                            "(the sub-cycler carries the viscous jump force).");
        }
        // ENERGY-CONSISTENT ALPHA in the pure-cell relaxation guard.
        //
        // THE BUG.  In the pure-cell guard of RelaxAndReinit, p_pure was
        // solved from the alpha_floor-CLAMPED a1 (>= 1e-12) while the two
        // per-phase energies were written with the RAW a1g (= eta clamped
        // only to [0,1]).  ReinitMixturePressure inverts
        //     rho_e = p*A(alpha) + B(alpha),
        // so writing E_k back with a different alpha breaks E0+E1 = rho_e by
        //     dE = (a1g - a1) * [ (p + gam0*pi0)/(gam0-1) - (p + gam1*pi1)/(gam1-1) ].
        // Inside a gas bubble eta ~ 1e-13 < alpha_floor, so a1g - a1 < 0 and,
        // because the liquid is stiff (pi0 = 1e9), the bracket is ~1.7e9:
        // dE ~ -1.5e-3 J/m^3 of internal energy REMOVED from every core cell
        // on every relaxation call.  One-signed, so it accumulates.  The gas
        // cools; its pressure is pinned to the liquid by the stiff
        // relaxation, so the bubble shrinks at constant pressure -- which is
        // exactly the measured drift (gas mass conserved to 3.4e-14, volume
        // -0.69 %/cycle, polytropic exponent 0.077 on the drift against 1.402
        // on the oscillation).  The sink scales with pi0, so it is ~100x
        // weaker at eos0.p0 = 1e7 than at 1e9 -- the drift_unit/stiffc pair.
        //
        // THE FIX.  Use the SAME alpha for the pressure solve and the energy
        // write.  a1g is the right one: the comment below explains that E_k
        // must be built with the frozen raw eta so the later E_k/eta recovery
        // divide cancels exactly.  Making p_pure agree restores
        // E0 + E1 = rho_e identically without reintroducing that mispricing.
        // PRESSURE RELAXATION (always on; the flags relax.closure, relax.allow_tension,
        // relax.trace and relax.reinit_split were removed 2026-11-05):
        //  * Closure: each phase goes ISENTROPICALLY to its final pressure (liquid p, gas
        //    p + Delta); the energy residual Q = rho e - sum (alpha rho)_k e_k^s is added as
        //    the SAME specific heat q = Q/rho to every phase, at fixed p_k:
        //    v_k = v_k^s + (gam_k - 1) q/(p_k + gam_k pi_k).  The volume constraint
        //    sum (alpha rho)_k v_k = 1 fixes p.  Conserves rho e exactly.  No root / v_k <= 0
        //    -> the Sch20 SC path + reinit below (counted in relax_split with relax.heat_diag).
        //  * Root bracket over the full EOS range, p_liq > -pi_liq and p_liq + Delta > -pi_gas
        //    (liquid tension is physical: KM wall pressure p_g - 2 sigma/R - 4 kappa Rdot/R^2
        //    < 0 during expansion), but the mixture pressure p_liq + alpha_gas Delta stays > 0
        //    (Riemann solver).  A positive bracket gave no root -> equal-pressure reinit ->
        //    gas heating (2026-11-01).
        //  * Reinit (fallback path): the defect between rho e and the relaxed phasic energies
        //    is split by mass fraction Y_k (Sch20 eq. 26 split it by alpha_k/(gamma_k - 1),
        //    ~90% to the gas in a 50/50 cell).
        //  * Trace phase: in pure-cell guard cells (eta beyond cutoff) whose dominant phase is
        //    the softer one (bulk modulus gam (p + pi); trace water in gas), the trace phase
        //    keeps its own (alpha rho e)_k and its volume fraction becomes
        //    alpha_t = (gam_t - 1) E_t / (p + gam_t pi_t) at the common pressure (one 1D root).
        //    Frozen alpha let converging flow compress trace liquid far beyond its EOS and the
        //    reinit handed its gam pi/(gam - 1) offset energy to the gas (centre heating).
        pp_query_default("limiter.radial_base", value.lim_radial_base, 0);   // TEMP: 1 = base limiter on RADIAL slab transverse faces
        pp_query_default("diag.gasmass", value.diag_gasmass, 0);        // TEMP gas-mass ledger (see Hydro2.H)
        pp_query_default("diag.gasmass_dt", value.diag_gasmass_dt, 1.0e-8);
        pp_query_default("diag.trace_x", value.diag_trace_x, -1.0);      // TEMP: print cell state at each ledger point
        pp_query_default("jump.face_force", value.jump_face_force, 0);   // TEMP: 1 = jump forces J grad(eta) from face differences (balanced-force)
        // RELAXATION VISCOUS JUMP (2026-11-14): 0 (default) = eta J_R = int J_F d eta integrated along the
        // grad-eta path inside the band (see RelaxAndReinit); 1 = old per-cell jump J_R = J_F (cell-local form,
        // bit-for-bit the previous behaviour), kept for a future cell-local approach.
        pp_query_default("visc.relax_cell_local", value.visc_relax_cell_local, 0);
        pp_query_default("visc.relax_path_log", value.visc_relax_path_log, 0);   // diagnostic: log every N calls (0 = off)
        pp_query_default("diag.trace_t0", value.diag_trace_t0, 0.0);
        pp_query_default("diag.trace_t1", value.diag_trace_t1, -1.0);
        pp_query_default("relax_diag", value.relax_diag, 0); // 1 = print per-stage {max_iters, max_residual, count_unconverged}.
        // relax.heat_diag = 1: plot field relax_heat_{gas,liq} = cumulative energy each phase
        // received in RelaxAndReinit BEYOND isentropic compression, (alpha rho e)_post minus
        // alpha_post (p_s + gam pi)/(gam - 1) with p_s + pi = (p_pre + pi)(rho_post/rho_pre)^gam
        // [J/m^3, summed over every call].  Output only; does not change the solution.
        pp_query_validate("relax.heat_diag", value.relax_heat_diag, {0, 1});
        pp_query_default("clip_ghost_only", value.clip_ghost_only, 0); // 1 = FillGhost STEP-9 positivity clip touches GHOST cells only (per-phase mass conservation)

        // Symmetry-face detection: a domain face whose normal-momentum BC is
        // REFLECT_ODD is a symmetry plane; its advective fluxes are enforced
        // to the exact u*=0 solution in RHS (see symmetrize()).
        {
            const char* faces[6] = {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"};
            for (int f = 0; f < 2 * AMREX_SPACEDIM; ++f)
            {
                const int d = f / 2;
                const bool is_lo = (f % 2 == 0);
                std::vector<std::string> mom_bc;
                std::string key = std::string("momentum.bc.type.") + faces[d * 2 + (is_lo ? 0 : 1)];
                if (pp.queryarr(key.c_str(), mom_bc) && (int)mom_bc.size() > d)
                {
                    std::string t = mom_bc[d];
                    for (auto &c : t) c = toupper(c);
                    const bool sym = (t == "REFLECT_ODD");
                    if (is_lo) value.sym_face_lo[d] = sym;
                    else       value.sym_face_hi[d] = sym;
                    if (sym)
                        Util::Message(INFO, "symmetry face detected: ", faces[d * 2 + (is_lo ? 0 : 1)],
                                      " (exact u*=0 flux enforced)");
                }
            }
        }


        // BOUNDRY CONDITITIONS
        pp_query_default("nghost", value.nghost, 4);   // [2026-11-24] default = the 06 Cattaneo configuration (was 2) // Number of Ghost Cells (NOTE: NSCBC can only use 2 or 4 nghost) ### WIP ### Add nghost cabability for nghost

        // GUARD: FillPatch of a fine level's ghost region interpolates from
        // the PARENT level, and needs the parent's valid data to cover the
        // fine box grown by nghost (coarsened by the ref ratio) PLUS one
        // coarse cell for the conservative-interp slope stencil.  AMReX's
        // proper nesting only guarantees coverage at blocking_factor
        // granularity, so amr.blocking_factor < 2*nghost lets a fine box's
        // ghost region poke past the parent's coverage -- and
        // FillPatchTwoLevels then SILENTLY interpolates from UNFILLED
        // memory into coarse-fine ghosts (measured 2026-09-03 on
        // FlowMarmottant/input_Marmottant_Laplace_Static, nghost=4:
        // blocking_factor 2 and 4 trap on snan-poisoned coarse cells inside
        // amrex::mf_compute_slopes; 8 runs clean).  Whatever the heap held
        // becomes ghost data: silent corruption at best, NaN blow-ups at
        // worst, worse at small radii / deep refinement.  Fail loudly here
        // instead.
        {
            // With amr.blocking_factor < 2*nghost the DEFAULT proper-nesting
            // buffer would let coarse-fine FillPatch read unfilled parent
            // cells (silent ghost corruption).  hydro2.cc auto-raises
            // amr.n_proper before the mesh is constructed, so small blocking
            // factors are safe without input edits; this is informational.
            amrex::ParmParse pp_amr("amr");
            int max_level_in = 0, nproper_in = 1;
            pp_amr.query("max_level", max_level_in);
            pp_amr.query("n_proper", nproper_in);
            std::vector<int> bf;
            pp_amr.queryarr("blocking_factor", bf);
            if (max_level_in > 0)
                for (int b : bf)
                    if (b * nproper_in < 2 * value.nghost)
                        Util::Warning(INFO, "amr.blocking_factor=", b, " with amr.n_proper=",
                                      nproper_in, " gives a proper-nesting width below nghost=",
                                      value.nghost, " + interp stencil: coarse-fine ghost fills "
                                      "may read unfilled parent cells. Raise amr.n_proper "
                                      "(hydro2.cc auto-sets it unless overridden).");
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

        // Primitive (density, momentum, pressure, eta) vs conservative (density, momentum, energy) BC path. Default 0 keeps the existing conservative
        pp_query_default("bc.primitive", value.bc_primitive, 0);

        // Initialize boundary conditions based on whether NSCBC is used
        Util::Message(INFO, "uses_nscbc=", uses_nscbc);
        if (uses_nscbc)
        {
            if (value.nghost == 2)
            {
#if AMREX_SPACEDIM == 3
                // The simple (nghost=2) NSCBC variant is 2D-only: single-tangential
                // transverse LODI + a 2D corner closure that were never ported to 3D/
                // Use NSCBC4 (nghost=4) for 3D, which is validated (faces+edges+corners).
                Util::Abort(INFO, "Simple NSCBC (nghost=2) is 2D-only; set nghost=4 to use NSCBC4 for 3D runs.");
#endif
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

            // PER-FACE EXPRESSION BCs (2026-09-08), not a global fill.  The
            // previous global ZeroNeumann (and BC::Nothing before it) DISCARDED
            // the input's per-face types in NSCBC mode -- in particular
            // REFLECT_ODD on the normal momentum at symmetry planes.  NSCBC4
            // skips faces typed 'none', so nothing ever wrote MIRRORED ghosts
            // at the symmetry planes: the reconstruction stencils at the first
            // interior faces read unreflected (+u instead of -u) ghosts, and
            // the resulting 2-3-cell boundary layer error made cells on the
            // symmetry AXES oscillate at ~2x the off-axis amplitude (~7x at
            // the triple corner) -- driving rectified trace-liquid transport
            // along the sealed axes into the origin trap (driven-bubble
            // deaths at t~8-9e-3, amplitude-independent; the user spotted the
            // axis pattern in the plotfiles).  BC::Expression honours
            // REFLECT_* per face, clamp-fills nscbc_* faces (finite data for
            // FillPatch; NSCBC4 overwrites the real characteristic ghosts),
            // and its GetBCRec translates NSCBC types to foextrap.
            value.density_bc = new BC::Expression(1, pp, "density.bc");
            value.energy_bc = new BC::Expression(1, pp, "energy.bc");
            value.momentum_bc = new BC::Expression(AMREX_SPACEDIM, pp, "momentum.bc");
            // No-slip / slip WALLS on the domain face (2026-10-29): the momentum ghost is the odd reflection
            // 2 u_w - u_mirror on faces whose wall-normal momentum is a Dirichlet 0, so the wall sits on the face
            // and not at the ghost-cell centre half a cell outside the domain (which made the single-phase Couette
            // profile 0.76 % wrong on 64 cells, first order).  Inflow faces (non-zero normal value) keep ghost =
            // value.  momentum.bc.dirichlet_at_face = 0 restores the old placement (see BC/Expression.H).
            if (!pp.contains("momentum.bc.dirichlet_at_face"))
                static_cast<BC::Expression*>(value.momentum_bc)->SetDirichletFace(2);
            /* superseded:
            // ZERO-NEUMANN (not BC::Nothing) for the standard BC pointers.
            // These objects are ALSO the physbc handed to FillPatch /
            // RemakeLevel / InterpFromCoarseLevel for every registered fab.
            // With BC::Nothing, any fine level that touches a PHYSICAL
            // boundary interpolates its new cells from coarse slope stencils
            // whose out-of-domain neighbors were NEVER FILLED (snan trap in
            // amrex::mf_cell_cons_lin_interp the moment refinement reaches an
            // NSCBC face; unpoisoned builds instead inherit heap garbage that
            // the NSCBC face fill never repairs -- measured as a zero-pressure
            // gas-like state in the last boundary cell breaching the Tammann
            // floor at t~4.7e-4 on the FlowMarmottant sweep's smallest
            // radius).  Zero-neumann gives the interpolation finite,
            // smooth out-of-domain data; NSCBC4 still overwrites the real
            // characteristic ghosts afterward in FillGhost4BC.
            value.density_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
            value.energy_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
            value.momentum_bc = new BC::Constant(BC::Constant::ZeroNeumann(AMREX_SPACEDIM));
            */

            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
            Util::Message(INFO, "nscbc4_bc Pointer=", value.nscbc4_bc);
        }
        else
        {
            // Standard mode: Use Expression BCs
            value.nscbc_bc = nullptr;

            value.density_bc = new BC::Expression(1, pp, "density.bc");
            value.energy_bc = new BC::Expression(1, pp, "energy.bc");
            value.momentum_bc = new BC::Expression(AMREX_SPACEDIM, pp, "momentum.bc");
            // No-slip / slip WALLS on the domain face (2026-10-29): the momentum ghost is the odd reflection
            // 2 u_w - u_mirror on faces whose wall-normal momentum is a Dirichlet 0, so the wall sits on the face
            // and not at the ghost-cell centre half a cell outside the domain (which made the single-phase Couette
            // profile 0.76 % wrong on 64 cells, first order).  Inflow faces (non-zero normal value) keep ghost =
            // value.  momentum.bc.dirichlet_at_face = 0 restores the old placement (see BC/Expression.H).
            if (!pp.contains("momentum.bc.dirichlet_at_face"))
                static_cast<BC::Expression*>(value.momentum_bc)->SetDirichletFace(2);

            // Primitive path: prescribe pressure at the boundary and reconstruct
            // per-phase energies from the EOS in FillGhost4BC. Left nullptr in
            // the conservative path (delete nullptr is safe).
            if (value.bc_primitive)
                value.pressure_bc = new BC::Expression(1, pp, "pressure.bc");

            Util::Message(INFO, "Parsing Reg");
            Util::Message(INFO, "nscbc_bc Pointer=", value.nscbc_bc);
        }

        // ------------------------------------------------------------------
        // PER-PHASE density BCs for the conserved partial densities.
        // ------------------------------------------------------------------
        // rho_eta0/1 are filled directly from their own BC (FillGhost4BC no longer
        // partitions the mixture density by eta).  A DIRICHLET "density.bc" value is
        // therefore written into BOTH phases and the mixture ghost becomes their SUM --
        // 2x the intended density for a 50/50 mixture, or the whole liquid density
        // injected into the gas phase for a pure-phase far field.  Declaring
        // "density0.bc" / "density1.bc" gives each phase its own (alpha rho)_k value.
        // Unset => the shared mixture BC, i.e. unchanged behaviour (exact for
        // neumann / reflect / periodic faces, where the sum of the per-phase
        // extrapolations equals the extrapolated mixture).
        const bool has_rho0_bc = pp.contains("density0.bc.type.xlo") || pp.contains("density0.bc.type.ylo");
        const bool has_rho1_bc = pp.contains("density1.bc.type.xlo") || pp.contains("density1.bc.type.ylo");
        value.density0_bc = has_rho0_bc ? new BC::Expression(1, pp, "density0.bc") : value.density_bc;
        value.density1_bc = has_rho1_bc ? new BC::Expression(1, pp, "density1.bc") : value.density_bc;
        if (has_rho0_bc || has_rho1_bc)
            Util::Message(INFO, "Per-phase density BCs: density0.bc=", has_rho0_bc,
                          " density1.bc=", has_rho1_bc);
        else
        {
            // Warn if any face prescribes a DIRICHLET mixture density without per-phase
            // values: that face will apply the same value to both partial densities.
            for (const auto &face : bc_faces)
            {
                std::string t;
                pp.query(("density.bc.type." + face).c_str(), t);
                if (t.find("dirichlet") != std::string::npos || t.find("DIRICHLET") != std::string::npos)
                    Util::Warning(INFO, "density.bc.type.", face, " is dirichlet but no density0.bc/",
                                  "density1.bc given: the mixture value is applied to BOTH partial ",
                                  "densities (ghost rho = sum).  Declare per-phase density BCs.");
            }
        }

        // Eta BC: parse from "eta.bc" if user provides it, otherwise zero neumann.
        // Eta is a volume fraction transported by advection + Cahn-Hilliard;
        if (pp.contains("eta.bc.type.xlo") || pp.contains("eta.bc.type.ylo"))
        {
            value.eta_bc = new BC::Expression(1, pp, "eta.bc");
            const char *fnm[3][2] = {{"xlo", "xhi"}, {"ylo", "yhi"}, {"zlo", "zhi"}};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int sd = 0; sd < 2; ++sd)
                {
                    std::string t;
                    pp.query((std::string("eta.bc.type.") + fnm[d][sd]).c_str(), t);
                    value.foot_mirror[d][sd] = (t.rfind("REFLECT", 0) == 0);   // shell.foot_rate ghosts
                }
        }
        else
        {
            // [2026-10-06] DEFAULT eta BC: zero gradient on physical boundaries, PERIODIC where the domain is.  It used to
            // be zero gradient in every direction; in a periodic direction that disagrees with the periodic density /
            // momentum / energy and is a momentum source wherever alpha matters to the pressure (stiff phases): the
            // slab-vibration "instability" of II.19 / II.20 was exactly this (changelog II.29).
            amrex::Vector<int> eta_per(AMREX_SPACEDIM, 0);
            { amrex::ParmParse ppg("geometry"); ppg.queryarr("is_periodic", eta_per, 0, AMREX_SPACEDIM); }
            bool any_per = false; for (int d = 0; d < AMREX_SPACEDIM; ++d) any_per = any_per || eta_per[d];
            if (!any_per) value.eta_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
            else
            {
                amrex::Vector<std::string> eta_bstr;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) eta_bstr.push_back(eta_per[d] ? "periodic" : "neumann");
                amrex::Vector<amrex::Real> eta_z(1, 0.0);
                value.eta_bc = new BC::Constant(1, eta_bstr, eta_bstr, AMREX_D_DECL(eta_z, eta_z, eta_z), AMREX_D_DECL(eta_z, eta_z, eta_z));
                Util::Message(INFO, "eta.bc not given: zero gradient on physical boundaries, periodic in the periodic direction(s)");
            }
        }
        // WALL CONTACT ANGLE (2026-11-19).  eta.contact_angle.<face> = theta [degrees], measured
        // through the LIQUID; 0 = liquid wets the wall completely (hydrophilic substrate: a liquid film
        // stays between gas and wall), 90 = neutral, 180 = gas wets the wall.  Unset = off (the mirror
        // ghosts of a REFLECT face then give exactly 90 degrees).  Wall adhesion of Brackbill, Kothe &
        // Zemach (JCP 100, 1992): at the wall the interface normal is n_w = cos(theta) e + sin(theta) t,
        // e = outward unit wall normal, t = wall-tangential direction of the adjacent interior normal.
        // It is imposed on the ghost normals of the capillary curvature (TensorCurvature), so
        // kappa = div n at the wall cells carries the contact-line force; the hydro fluxes and the
        // wall's mass/momentum conditions are untouched.  Requires a wall (REFLECT) face.
        {
            const char *fnm[3][2] = {{"xlo", "xhi"}, {"ylo", "yhi"}, {"zlo", "zhi"}};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int sd = 0; sd < 2; ++sd)
                {
                    const std::string key = std::string("eta.contact_angle.") + fnm[d][sd];
                    if (!pp.contains(key.c_str())) continue;
                    Set::Scalar deg = 90.0;
                    pp.query(key.c_str(), deg);
                    if (!(deg >= 0.0 && deg <= 180.0))
                        Util::Abort(INFO, key, " = ", deg, ": contact angle must be in [0, 180] degrees");
                    if (!value.foot_mirror[d][sd])
                        Util::Abort(INFO, key, " is set but eta.bc.type.", fnm[d][sd], " is not a REFLECT (wall) face");
                    value.contact_angle[d][sd] = deg * M_PI / 180.0;
                }
            // eta.contact_angle_profile = 1 (2026-11-20): impose the angle in proportion to how much the
            // wall cell is an INTERFACE.  The resolved diffuse interface is the tanh profile of the IC,
            // for which |grad eta| = 2 eta (1 - eta) / epsilon exactly.  A uniform gas-liquid mixture
            // (sub-band gas film trapped by a jet on the wall) has eta(1-eta) > 0 but |grad eta| ~ 0: no
            // contact line there, yet mode 0 forced n_w on its noise normals (row-by-row eta
            // alternation in the wall foam, 2D jet 2026-11-19).  Mode 1 uses
            //     w = min(1, epsilon |grad eta| / (2 eta (1 - eta)))   at the mirror interior cell,
            //     n_ghost = normalise(w n_w + (1 - w) n_mirror):
            // w = 1 on a resolved interface (mode 0 exactly), w -> 0 in a uniform mixture (mirror = neutral).
            pp_query_validate("eta.contact_angle_profile", value.contact_angle_profile, {0, 1});
        }

        // RADIALLY SYMMETRIC SLAB (2026-10-23).  density.bc.type.yhi (and zhi
        // in 3D) = RADIAL turns a thin slab x in [0,L], y (,z) in [0,h] into a
        // 1D radial problem: cylindrical in 2D, spherical in 3D, centred on the
        // origin.  Every ghost cell outside a RADIAL face is refilled from the
        // near-axis row (j = k = 0) at the SAME radius r = |x| (scalars), and
        // momentum as u_r(r) x/|x|.  The ordinary Cartesian stencils then see
        // an exactly radial field, so the (D-1)/r geometry comes out of them
        // with no new source terms.  See RadialGhostFill.  AMR (2026-10-24): a
        // fine level reads its axis row over its own patches (plus their
        // coarse-fine x-ghosts), so the fine patches must contain the axis row
        // -- eta tagging of a thin slab refines its whole width.
        {
            const char *tf[2] = {"yhi", "zhi"};
            for (int d = 1; d < AMREX_SPACEDIM; ++d)
            {
                std::string t;
                pp.query((std::string("density.bc.type.") + tf[d - 1]).c_str(), t);
                if (BC::BCUtil::IsRadial(BC::BCUtil::ReadString(t)))
                {
                    value.radial_face_hi[d] = true;
                    value.radial_bc = true;
                }
            }
            if (value.radial_bc)
            {
                std::vector<Set::Scalar> plo;
                amrex::ParmParse pp_geom("geometry");
                pp_geom.queryarr("prob_lo", plo);
                for (int d = 1; d < AMREX_SPACEDIM; ++d)
                {
                    if (!value.radial_face_hi[d])
                        Util::Abort(INFO, "RADIAL: every transverse hi face (yhi", AMREX_SPACEDIM == 3 ? ", zhi" : "",
                                    ") must be RADIAL");
                    if (!value.sym_face_lo[d])
                        Util::Abort(INFO, "RADIAL: the transverse lo faces must be symmetry planes "
                                    "(momentum REFLECT_ODD on the normal component)");
                }
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    if ((int)plo.size() > d && plo[d] != 0.0)
                        Util::Abort(INFO, "RADIAL: geometry.prob_lo must be 0 (the centre is the origin)");
                if (!value.sym_face_lo[0])
                    Util::Abort(INFO, "RADIAL: xlo must be a symmetry plane (the centre)");
                // NSCBC is allowed on xhi only (far field of the slab); the
                // transverse and centre faces stay with the field BCs.
                if (uses_nscbc)
                    for (const char *f : {"xlo", "ylo", "yhi", "zlo", "zhi"})
                    {
                        std::string t = "none";
                        amrex::ParmParse pp_ns("nscbc");
                        pp_ns.query((std::string(f) + ".type").c_str(), t);
                        if (t != "none")
                            Util::Abort(INFO, "RADIAL: NSCBC is allowed on xhi only (nscbc.", f, ".type = ", t, ")");
                    }
                if (value.embedded.apply)
                    Util::Abort(INFO, "RADIAL: not combined with apply_embedded_solid");
                // [2026-11-24] RADIAL ghost source: 0 = near-axis profile at the same r (default);
                // 1 = local mirror: interior cell at the same r on the rows next to the face.
                pp_query_validate("radial.ghost_fill", value.radial_ghost_fill, {0, 1});
                // every registered field BC must say RADIAL on those faces too
                for (const char *f : {"eta", "energy", "momentum"})
                    for (int d = 1; d < AMREX_SPACEDIM; ++d)
                    {
                        std::vector<std::string> tt;
                        pp.queryarr((std::string(f) + ".bc.type." + tf[d - 1]).c_str(), tt);
                        for (auto &x : tt)
                            if (!BC::BCUtil::IsRadial(BC::BCUtil::ReadString(x)))
                                Util::Abort(INFO, "RADIAL: ", f, ".bc.type.", tf[d - 1], " must also be RADIAL");
                    }
                Util::Message(INFO, "RADIAL slab: ", AMREX_SPACEDIM == 3 ? "spherical" : "cylindrical",
                              " symmetry about the origin, ghosts on the transverse hi faces from the axis row");
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                value.cen_face_lo[d] = value.sym_face_lo[d];
                value.cen_face_hi[d] = value.sym_face_hi[d] || value.radial_face_hi[d];
                if (value.cen_face_lo[d] || value.cen_face_hi[d]) value.cen_face_any = 1;
            }
        }

        // Phi (solid indicator) BC.  The solid geometry is static, so the
        // indicator should not vary at the domain edge: zero-neumann
        // (ghost = interior) is the correct default.  An explicit
        // "solid.phi.bc" expression may override it.
        if (value.embedded.apply)
        {
            if (pp.contains("solid.phi.bc.type.xlo") || pp.contains("solid.phi.bc.type.ylo"))
                value.embedded.phi_bc = new BC::Expression(1, pp, "solid.phi.bc");
            else
                value.embedded.phi_bc = new BC::Constant(BC::Constant::ZeroNeumann(1));
        }
    }

    // Register FabFields:
    // Toggle the last boolean to true/false to track the variable or not.
    {
        int nghost = value.nghost;

        // plot_minimal (default 0 = OFF, nothing changes): drop the pure
        // diagnostic fields from the plotfiles and keep only what the bubble
        // analyses actually read.  A 3D plotfile carries ~69 components; the
        // set kept here is ~19, so the files shrink by roughly 70%.
        //
        // This exists because the Sch20 microbubble inputs are 160^3 at level
        // 0 for an R0 = 2e-6 bubble in a 2e-4 box: level 0 alone is 2.2 GB per
        // plotfile (2.5 GB with the refined levels), and at their plot_dt that
        // is ~500 GB for the oscillating case and ~1 TB for the collapsing
        // one -- more than the filesystem will take.
        //
        // KEPT: eta, rho_eta0/1, shell, Gamma, pressure, velocity, density,
        //       momentum, energy0/1, energy_per_vol, T, UE_per_vol, KE_per_vol
        // DROPPED: cfun, etadot, vorticity, energy_per_mass, Source, Fsv, Fw,
        //       Ldot, gamma, p0, mu_chem, a, Ma, UE/KE_per_mass, Spalding,
        //       Mass_Fraction, rho/M/E_flux, div_tau, Vap_dot
        //
        // kappa (surface curvature) is KEPT: reference/amr_analyze.py reads
        // kappa2 for the Marmottant sigma(R), and its except-branch silently
        // substitutes zeros -- a dropped field would look like sigma = 0.
        //
        // T, UE_per_vol and KE_per_vol are deliberately KEPT: they are what a
        // thermal or energy-budget diagnosis needs.
        int plot_minimal = 0;
        pp.query("plot_minimal", plot_minimal);
        const bool diag = (plot_minimal == 0);
        if (plot_minimal) Util::Message(INFO, "plot_minimal = 1: diagnostic fields omitted from plotfiles");

        // BC carried by the (derived) pressure fields for AMR coarse-fine
        // FillPatch. In the primitive path this is the user pressure BC; else
        // fall back to energy_bc (pressure fields are non-evolving, so this is
        // mostly cosmetic -- the physical-boundary fill happens in FillGhost4BC).
        BC::BC<Set::Scalar>* pbc =
            (value.bc_primitive && value.pressure_bc) ? value.pressure_bc : value.energy_bc;

        // DIFFUSE PARAMETERS
        value.RegisterNewFab(value.eta_mf,           value.eta_bc,      1, nghost,  "eta",          true, true);
        value.RegisterNewFab(value.eta_old_mf,       value.eta_bc,      1, nghost,  "eta_old",      false,true);
        value.RegisterNewFab(value.rho_eta0_mf,      value.density0_bc,  1, nghost,  "rho_eta0",     true, true);
        value.RegisterNewFab(value.rho_eta1_mf,      value.density1_bc,  1, nghost,  "rho_eta1",     true, true);
        value.RegisterNewFab(value.rho_eta0_old_mf,  value.density0_bc,  1, nghost,  "rho_eta0_old", false,true);
        value.RegisterNewFab(value.rho_eta1_old_mf,  value.density1_bc,  1, nghost,  "rho_eta1_old", false,true);

        // Advected shell density c = Gamma|grad eta| (8th RK primary) + its
        // diagnostic areal density Gamma = c/|grad eta|.  Shares eta's BC: c is a
        // passive advected scalar with the same zero-gradient far-field behaviour.
        value.RegisterNewFab(value.shell_mf,         value.eta_bc,      1, nghost,  "shell",        true, true);
        value.RegisterNewFab(value.shell_old_mf,     value.eta_bc,      1, nghost,  "shell_old",    false,true);
        // Colour function c (Schmidmayer 2017 eq. 3): 9th RK primary, pure advection.
        value.RegisterNewFab(value.cfun_mf,          value.eta_bc,      1, nghost,  "cfun",         diag, true);
        value.RegisterNewFab(value.cfun_old_mf,      value.eta_bc,      1, nghost,  "cfun_old",     false,true);
        value.RegisterNewFab(value.Gamma_mf,        &value.bc_nothing,  1, 0,       "Gamma",        true, false);

        value.RegisterNewFab(value.etadot_mf,       &value.bc_nothing,  1, 0,       "etadot",       diag, false);
        value.RegisterNewFab(value.relax_heat_mf,   &value.bc_nothing,  2, 0,       "relax_heat",   value.relax_heat_diag, false, { "gas", "liq" });
        // relax.heat_diag: target jump J seen by each relaxation call, the previous call's J,
        // and running sums of |J - J_prev| and (J - J_prev)^2 over calls [Pa, Pa, Pa^2]
        // relax.heat_diag: where the gas energy change in RelaxAndReinit comes from, summed over calls
        //   gasdefect/liqdefect = each phase's share of the reinit defect (E - KE) - (E0r + E1r)
        //   gasnewton = gas energy at the relaxed pressure minus isentropic (the Newton step)
        //   gasguard  = gas energy change in pure-cell / floored-guard cells (equal-pressure reset)
        //   nguard, neq26 = number of calls taking the guard path / the eq. 26 fallback
        value.RegisterNewFab(value.relax_split_mf,  &value.bc_nothing,  10, 0,       "relax_split",  value.relax_heat_diag, false, { "gasdefect", "liqdefect", "gasnewton", "gasguard", "nguard", "neq26", "nmassfallback", "nqneg", "nplneg", "nraised" });
        value.RegisterNewFab(value.relax_J_mf,      &value.bc_nothing,  4, 0,       "relax_J",      value.relax_heat_diag, false, { "J", "Jprev", "sumabsdJ", "sumdJ2" });
        value.RegisterNewFab(value.hess_eta_mf,     &value.bc_nothing,  4, 0,       "hess_eta",     false,false, { "00", "01", "10", "11" });
        value.RegisterNewFab(value.n_hat_mf,        &value.bc_nothing,  AMREX_SPACEDIM, 0,       "n_hat",        false,false, { AMREX_D_DECL("x", "y", "z") });

        if (value.phasechange.on)
        {
            value.RegisterNewFab(value.phasechange.vapor_mf,     value.eta_bc, 1, nghost, "vapor",     true,  true);
            value.RegisterNewFab(value.phasechange.vapor_old_mf, value.eta_bc, 1, nghost, "vapor_old", false, true);
            value.RegisterNewFab(value.phasechange.diag_mf, &value.bc_nothing, 4, 0, "pc", true, false, { "_T", "_Tsat", "_Yv", "_dm" });
        }
        if (value.species.on)
        {
            const int ns = value.species.ns();
            if (pp.contains("species.bc.type.xlo") || pp.contains("species.bc.type.ylo"))
                value.species.bc = new BC::Expression(ns, pp, "species.bc");
            else
            {
                if (AMREX_D_TERM(value.geom[0].isPeriodic(0), || value.geom[0].isPeriodic(1), || value.geom[0].isPeriodic(2)))
                    Util::Abort(INFO, "species.on on a periodic domain needs species.bc.type.* (periodic in the periodic directions): "
                                "the default is zero gradient in every direction");
                value.species.bc = new BC::Constant(BC::Constant::ZeroNeumann(ns));
            }
            std::vector<std::string> sfx; for (const auto &n : value.species.names) sfx.push_back("_" + n);
            value.RegisterNewFab(value.species.mf,     value.species.bc, ns, nghost, "species",     true,  true, sfx);
            value.RegisterNewFab(value.species.old_mf, value.species.bc, ns, nghost, "species_old", false, true, sfx);
            value.RegisterNewFab(value.species.Y_mf, &value.bc_nothing,  ns, 0,      "Y",           true,  false, sfx);
            if (value.chem.on) value.RegisterNewFab(value.chem.heat_mf, &value.bc_nothing, 1, 0, "chem_heat", true, false);
        }
        if (value.entropy_on)
            value.RegisterNewFab(value.entropy_mf, &value.bc_nothing, 1, 0, "entropy", true, false);
        if (value.ablation.on)
        {
            value.RegisterNewFab(value.ablation.q_mf, &value.bc_nothing, 1, 2, "ablation_q", true, false);
            value.RegisterNewFab(value.ablation.v_mf, &value.bc_nothing, 1, 2, "ablation_v", true, false);
            if (value.ablation.conduction)
            {
                // the solid temperature shares the indicator's boundary condition (zero-gradient by default,
                // periodic / expression when solid.phi.bc is given): adiabatic at the domain edge
                value.ablation.Ts_bc = value.embedded.phi_bc;
                if (!value.ablation.Ts_bc) Util::Abort(INFO, "ablation.conduction needs apply_embedded_solid = 1");
                value.RegisterNewFab(value.ablation.Ts_mf, value.ablation.Ts_bc, 1, 2, "ablation_Ts", true, false);
                value.RegisterNewFab(value.ablation.qs_mf, &value.bc_nothing, 2 * AMREX_SPACEDIM, 1, "ablation_qs", false, false);
            }
        }
        // ELASTIC SOLID PHASE (elastic.on): averaged cobasis E^b_i + stress diagnostics
        if (value.elastic.on)
        {
            value.RegisterNewFab(value.elastic.ebasis_mf,     value.elastic.bc, ElasticSolid::NEB, nghost, "ebasis",     true,  true, ElasticSolid::cobasisNames());
            value.RegisterNewFab(value.elastic.ebasis_old_mf, value.elastic.bc, ElasticSolid::NEB, nghost, "ebasis_old", false, true);
            value.RegisterNewFab(value.elastic.stress_mf,    &value.bc_nothing, 7, 0, "elastic", true, false,
                                 { "_Sxx", "_Syy", "_Szz", "_Sxy", "_Sxz", "_Syz", "_W" });
            if (value.elastic.plastic)
                value.RegisterNewFab(value.elastic.plastic_mf, &value.bc_nothing, 2, 0, "plastic", true, false, { "_rate", "_seq" });
        }

        // EMBEDDED SOLID BOUNDARY
        if (value.embedded.apply)
        {
            value.RegisterNewFab(value.embedded.phi_mf,         value.embedded.phi_bc, 1, nghost,   "phi",      true, false);
            value.RegisterNewFab(value.embedded.phi_old_mf,     value.embedded.phi_bc, 1, nghost,   "phi_old",  false,false);
            value.RegisterNewFab(value.embedded.grad_phi_mf,   &value.bc_nothing,      2, 0,        "grad_phi", true, false, { "x", "y" });

            // Prescribed per-phase solid target state (quiescent solid).
            value.RegisterNewFab(value.embedded.density0_mf, value.density_bc,  1, nghost, "solid_rho_eta0",    false,false);
            value.RegisterNewFab(value.embedded.density1_mf, value.density_bc,  1, nghost, "solid_rho_eta1",    false,false);
            value.RegisterNewFab(value.embedded.momentum_mf, value.momentum_bc, AMREX_SPACEDIM, nghost, "solid_momentum",    false,false, { AMREX_D_DECL("x", "y", "z") });
            value.RegisterNewFab(value.embedded.energy0_mf,  value.energy_bc,   1, nghost, "solid_energy0",     false,false);
            value.RegisterNewFab(value.embedded.energy1_mf,  value.energy_bc,   1, nghost, "solid_energy1",     false,false);
        }

        // FLUID 0
        value.RegisterNewFab(value.density0_mf,     value.density_bc,   1, nghost,  "density0",     false,false);
        value.RegisterNewFab(value.density0_old_mf, value.density_bc,   1, nghost,  "density0_old", false,false);

        // E_0 / E_1 are 6-eq conserved primaries (per-phase internal energies).
        value.RegisterNewFab(value.energy0_mf,      value.energy_bc,    1, nghost, "energy0",       true, true);
        value.RegisterNewFab(value.energy0_old_mf,  value.energy_bc,    1, nghost, "energy0_old",   false,true);

        // PER-PHASE momenta are diagnostic only (read-only patches M0/M1) and are
        // built from the 2-component per-phase velocity ICs; kept 2-component in 3D.
        // They use bc_nothing (component-agnostic): momentum_bc now has SPACEDIM
        // components and would assert on these 2-component fabs in 3D.
        value.RegisterNewFab(value.momentum0_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum0",     false,false, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum0_old_mf,&value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum0_old", false,false);
 
        //value.RegisterNewFab(value.T0_mf,           value.temperature_bc, 1, nghost, "T0", false, false);
        //value.RegisterNewFab(value.k0_thermal_mf,   &value.bc_nothing, 1, nghost, "k0_thermal", false, false);
        //value.RegisterNewFab(value.h0_thermal_mf,   &value.bc_nothing, 1, nghost, "h0_thermal", false, false);

        value.RegisterNewFab(value.pressure0_mf,    pbc,                1, nghost, "pressure0",     false,false);
        value.RegisterNewFab(value.velocity0_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "velocity0",     false,false, { AMREX_D_DECL("x", "y", "z") });

        // FLUID 1
        value.RegisterNewFab(value.density1_mf,     value.density_bc,   1, nghost, "density1",      false,false);
        value.RegisterNewFab(value.density1_old_mf, value.density_bc,   1, nghost, "density1_old",  false,false);

        value.RegisterNewFab(value.energy1_mf,      value.energy_bc,    1, nghost, "energy1",       true, true);
        value.RegisterNewFab(value.energy1_old_mf,  value.energy_bc,    1, nghost, "energy1_old",   false,true);

        value.RegisterNewFab(value.momentum1_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum1",     false,false, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum1_old_mf,&value.bc_nothing,  AMREX_SPACEDIM, nghost, "momentum1_old", false,false);

        //value.RegisterNewFab(value.T1_mf,           value.temperature_bc, 1, nghost, "T1", false, false);
        //value.RegisterNewFab(value.k1_thermal_mf,   &value.bc_nothing, 1, nghost, "k1_thermal", false, false);
        //value.RegisterNewFab(value.h1_thermal_mf,   &value.bc_nothing, 1, nghost, "h1_thermal", false, false);

        value.RegisterNewFab(value.pressure1_mf,    pbc,                1, nghost, "pressure1",     false,true);
        value.RegisterNewFab(value.velocity1_mf,    &value.bc_nothing,  AMREX_SPACEDIM, nghost, "velocity1",     false,true, { AMREX_D_DECL("x", "y", "z") });

        // MIXTURE
        value.RegisterNewFab(value.pressure_mf,    pbc,                 1,              nghost,  "pressure",    true, false);
        value.RegisterNewFab(value.velocity_mf,    &value.bc_nothing,   AMREX_SPACEDIM, nghost,  "velocity",    true, false, { AMREX_D_DECL("x", "y", "z") });
        // Vorticity is the curl of velocity: a scalar (omega_z) in 2D, a full
        // 3-vector (omega_x, omega_y, omega_z) in 3D.
#if AMREX_SPACEDIM == 2
        value.RegisterNewFab(value.vorticity_mf,           &value.bc_nothing,   1, 0,       "vorticity",        diag, false);
#else
        value.RegisterNewFab(value.vorticity_mf,           &value.bc_nothing,   3, 0,       "vorticity",        diag, false, { "x", "y", "z" });
#endif
        value.RegisterNewFab(value.density_mf,              value.density_bc,   1, nghost,  "density",          true, false);
        value.RegisterNewFab(value.density_old_mf,          value.density_bc,   1, nghost,  "density_old",      false,false);
        value.RegisterNewFab(value.energy_per_vol_mf,       value.energy_bc,    1, nghost,  "energy_per_vol",   true, true);
        value.RegisterNewFab(value.energy_per_mas_mf,       value.energy_bc,    1, nghost,  "energy_per_mass",  diag, true);
        value.RegisterNewFab(value.energy_per_vol_old_mf,   value.energy_bc,    1, nghost,  "energy_vol_old",   false,true);
        value.RegisterNewFab(value.energy_per_mas_old_mf,   value.energy_bc,    1, nghost,  "energy_mas_old",   false,true);
        value.RegisterNewFab(value.momentum_mf,             value.momentum_bc,  AMREX_SPACEDIM, nghost,  "momentum",         true, true, { AMREX_D_DECL("x", "y", "z") });
        value.RegisterNewFab(value.momentum_old_mf,         value.momentum_bc,  AMREX_SPACEDIM, nghost,  "momentum_old",     false,true, { AMREX_D_DECL("x", "y", "z") });

        // SOURCES
        value.RegisterNewFab(value.m0_mf,          &value.bc_nothing,   1, 0,       "m0",               false,false);
        value.RegisterNewFab(value.u0_mf,          &value.bc_nothing,   2, 0,       "u0",               false,false, { "x", "y" });
        value.RegisterNewFab(value.q_mf,           &value.bc_nothing,   2, 0,       "q0",               false,false, { "x", "y" });
        value.RegisterNewFab(value.Source_mf,      &value.bc_nothing,   AMREX_SPACEDIM + 2, 0,  "Source",diag, false, { "_rho", AMREX_D_DECL("_Mx", "_My", "_Mz"), "_E" });
        value.RegisterNewFab(value.Fsv_mf,         &value.bc_nothing,   AMREX_SPACEDIM, 0,      "Fsv",  diag, false, { AMREX_D_DECL("x", "y", "z") }); // Surface Tension
        value.RegisterNewFab(value.Fw_mf,          &value.bc_nothing,   AMREX_SPACEDIM, 0,      "Fw",   diag, false, { AMREX_D_DECL("x", "y", "z") }); // Weight
        value.RegisterNewFab(value.Ldot_mf,        &value.bc_nothing,   AMREX_SPACEDIM, 0, "Ldot",      diag, false, { AMREX_D_DECL("x", "y", "z") }); // Ldot (3-comp: z-momentum source develops dynamically)
        value.RegisterNewFab(value.T_mf,            value.energy_bc,    1, nghost, "T",                 true, false);               // Temperature
        value.RegisterNewFab(value.cp_mf,          &value.bc_nothing,   1, nghost, "cp",                false,true);                // Constant Pressure Specific Heat
        value.RegisterNewFab(value.cv_mf,          &value.bc_nothing,   1, nghost, "cv",                false,true);                // Constant Volume Specific Heat
        //value.RegisterNewFab(value.k_thermal_mf,    &value.bc_nothing,  1, nghost, "k_thermal", false, true);         // Thermal Conductivity
        //value.RegisterNewFab(value.h_thermal_mf,    &value.bc_nothing,  1, nghost, "h_thermal", false, true);         // Thermal Convectivity
        value.RegisterNewFab(value.gamma_mf,        value.energy_bc,    1, nghost, "gamma",             diag, false);               // Specific Heat Ratio
        value.RegisterNewFab(value.p0_mf,           value.energy_bc,    1, nghost, "p0",                diag, true);                // Tamman Pressure
        value.RegisterNewFab(value.mu_chem_mf,      value.energy_bc,    1, nghost, "mu_chem",           diag, false);               // Chemical Potential
        value.RegisterNewFab(value.a_mf,           &value.bc_nothing,   1, nghost, "a",                 diag, false);               // Speed of sound
        value.RegisterNewFab(value.Ma_mf,          &value.bc_nothing,   2, nghost, "Ma",                diag, false, { "x", "y" }); // Mach
        value.RegisterNewFab(value.UE_per_vol_mf,   value.energy_bc,    1, nghost, "UE_per_vol",        true, false);               // Internal Energy (per unit volume)
        value.RegisterNewFab(value.UE_per_mas_mf,   value.energy_bc,    1, nghost, "UE_per_mass",       diag, false);               // Internal Energy (per unit mass)
        value.RegisterNewFab(value.KE_per_vol_mf,   value.energy_bc,    1, nghost, "KE_per_vol",        true, false);               // Kinetic Energy (per unit volume)
        value.RegisterNewFab(value.KE_per_mas_mf,   value.energy_bc,    1, nghost, "KE_per_mass",       diag, false);               // Kinetic Energy (per unit mass)
        value.RegisterNewFab(value.Bm_mf,          &value.bc_nothing,   1, nghost, "Spadling_Number",   diag, false);               // Spalding Number
        value.RegisterNewFab(value.Y_mf,           &value.bc_nothing,   1, nghost, "Mass_Fraction",     diag, false);               // Mass Fraction
        if (value.T_phase_consistent)
            value.RegisterNewFab(value.Tph_mf, &value.bc_nothing, 2, nghost, "T", true, false, { "_phase0", "_phase1" });
        if (value.vap_model >= 1)
            value.RegisterNewFab(value.Vap_film_mf, &value.bc_nothing, 5, 0, "Vap_film", true, false, { "_T_liquid", "_T_gas", "_B_M", "_B_T", "_Q_L" });

        // EXTRAS & DEBUGGING
        value.RegisterNewFab(value.grad_eta_mf,         &value.bc_nothing,  AMREX_SPACEDIM, 0, "grad_eta",           true, false, { AMREX_D_DECL("x", "y", "z") }); // grad(eta) -- KEPT by plot_minimal: it is the surface measure dA = |grad eta| dV, without which an areal density like Gamma cannot be averaged correctly
        value.RegisterNewFab(value.kappas_mf,           &value.bc_nothing,  3,              0, "kappa",              true, false, { "Avg", "1", "2" });             // Surface curvature
        value.RegisterNewFab(value.grad_mag_grad_eta_mf,&value.bc_nothing,  AMREX_SPACEDIM, 0, "grad_mag_grad_eta",  false,false, { AMREX_D_DECL("x", "y", "z") }); // grad( | grad(eta) | )
        value.RegisterNewFab(value.rho_flux_mf,         &value.bc_nothing,  1,              0, "rho_flux",           diag, false);                                  // Density Flux
        value.RegisterNewFab(value.M_flux_mf,           &value.bc_nothing,  AMREX_SPACEDIM, 0, "M_flux",             diag, false, { AMREX_D_DECL("x", "y", "z") }); // Momentum Flux
        value.RegisterNewFab(value.E_flux_mf,           &value.bc_nothing,  1,              0, "E_flux",             diag, false);                                  // Energy Flux
        value.RegisterNewFab(value.div_tau_mf,          &value.bc_nothing,  AMREX_SPACEDIM, 0, "div_tau",            diag, false, { AMREX_D_DECL("x", "y", "z") }); // Energy Flux
        value.RegisterNewFab(value.hess_u_mf,           &value.bc_nothing,  8,              0, "hess_u",             false,false, {"000","001",
                                                                                                                                   "010","011",
                                                                                                                                   "100","101",
                                                                                                                                   "110","111"});                   // hess_u Flux
        value.RegisterNewFab(value.Vap_dot_mf, &value.bc_nothing, AMREX_SPACEDIM + 3, 0, "Vap_dot", diag, false, { "_eta", "_rho", AMREX_D_DECL("_Mx", "_My", "_Mz"), "_E" });    // Momentum Flux
        
    }

    // INITIAL CONDITIONS
    // Eta
    pp.select_default<IC::Constant,IC::Laminate,IC::Expression,IC::BMP,IC::PNG>("eta.ic",value.eta_ic,value.geom);
    // Fluid 0
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("velocity0.ic",      value.velocity0_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("pressure0.ic",      value.pressure0_ic, value.geom);
    if (value.species.on) pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("species.ic", value.species.ic, value.geom);   // mass fractions within the phase
    // [2026-10-06] optional non-uniform initial solid temperature (in-depth conduction): ablation.Ts.ic.type = expression ...
    if (value.ablation.on && value.ablation.conduction && pp.contains("ablation.Ts.ic.type"))
        pp.select_default<IC::Constant,IC::Expression>("ablation.Ts.ic", value.ablation_Ts_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("density0.ic",       value.density0_ic,  value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("temperature0.ic",  value.temperature0_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("k0_thermal.ic",    value.k0_thermal_ic, value.geom);
    //pp.select_default<IC::Constant, IC::Expression>("h1_thermal.ic",    value.h0_thermal_ic, value.geom);

    // Fluid 1
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("velocity1.ic",      value.velocity1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("pressure1.ic",      value.pressure1_ic, value.geom);
    pp.select_default<IC::Constant,IC::Expression,IC::Profile1D>("density1.ic",       value.density1_ic,  value.geom);
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

    // EMBEDDED SOLID BOUNDARY initial conditions.
    //   solid.phi.ic       -> indicator field (1 fluid, 0 solid), e.g. a tanh
    //                         circle/airfoil expression (see tests/FlowCylinder).
    //   solid.density{0,1} -> prescribed phase masses (alpha_k rho_k) in solid.
    //   solid.momentum     -> prescribed mixture momentum in solid (0 = static wall).
    //   solid.energy{0,1}  -> prescribed per-phase internal energies in solid.
    if (value.embedded.apply)
    {
#if (AMREX_SPACEDIM == 3)
        // 3D build: also offer "stl" geometry (IC::STL -> self-contained Util::STL, no EB).
        pp.select_default<IC::Constant,IC::Expression,IC::BMP,IC::PNG,IC::STL>("solid.phi.ic", value.embedded.phi_ic, value.geom);
#else
        pp.select_default<IC::Constant,IC::Expression,IC::BMP,IC::PNG>("solid.phi.ic",       value.embedded.phi_ic,      value.geom);
#endif
        // Single-phase solid input: total density + pressure (+ momentum).  The
        // per-phase target slots are derived from these by InitEmbeddedSolidTarget.
        pp.select_default<IC::Constant,IC::Expression>("solid.density.ic",                  value.embedded.density_ic,  value.geom);
        pp.select_default<IC::Constant,IC::Expression>("solid.pressure.ic",                 value.embedded.pressure_ic, value.geom);
        pp.select_default<IC::Constant,IC::Expression>("solid.momentum.ic",                 value.embedded.momentum_ic, value.geom);
    }


    // SOLVERS
    // Riemann solver
    std::string solver_name;
    pp.query("Riemann_Solver.type", solver_name);
    Util::Message(INFO, "Input file has Riemann_Solver.type = ", solver_name);
    // [2026-11-24] default = HLLC (the 06 Cattaneo configuration; was Roe)
    pp.select_default<Solver::Local::FluidRiemann::HLLC,
                      Solver::Local::FluidRiemann::Roe,
                      Solver::Local::FluidRiemann::HLLE,
                      Solver::Local::FluidRiemann::HLLCE,
                      //Solver::Local::FluidRiemann::PartiallyParabolic, // WIP - very outdated - never verified
                      Solver::Local::FluidRiemann::HLLC_Oomar_Jaiman,   // Doesn't really work
                      Solver::Local::FluidRiemann::HLLC_All_Mach,
                      Solver::Local::FluidRiemann::HLLC_All_Mach_Furfaro,
                      Solver::Local::FluidRiemann::HLLC_Elastic,
                      Solver::Local::FluidRiemann::HLLC_Solid           // Favrie & Schmidmayer (2026); by name only
                      //Solver::Local::FluidRiemann::Upwind,            // Super bad - do not use
                      //Solver::Local::FluidRiemann::Lax_Friedrich      // Super bad - do not use
    >("Riemann_Solver", value.riemannsolver);
    if (value.elastic.on && !dynamic_cast<Solver::Local::FluidRiemann::HLLC_Elastic *>(value.riemannsolver)
                         && !dynamic_cast<Solver::Local::FluidRiemann::HLLC_Solid *>(value.riemannsolver))
    {
        // The solid stress only exists in the hllc_elastic (default) and hllc_solid fluxes.
        Util::Message(INFO, "elastic.on = 1: Riemann solver replaced by hllc_elastic");
        delete value.riemannsolver;
        value.riemannsolver = new Solver::Local::FluidRiemann::HLLC_Elastic();
    }
    Util::Message(INFO, "Selected Riemann solver: ", typeid(*value.riemannsolver).name());
    // Shock fix: HLLE on the faces lying along a detected shock (see Hydro2.H)
    pp.query_default("shock_fix.on", value.shock_fix, 0);
    // Finite-rate pressure relaxation coefficient omega [1/(pressure x time)]; < 0 (default) = instantaneous
    pp.query_default("relax.omega", value.relax_omega, -1.0);
    // relax.trace_pT (ported from flames2, 2026-11-19; default 0 = off, bit-identical).  A cell whose SOFTER phase
    // (smaller pi: the gas) holds mass but occupies less than relax.trace_alpha of the volume is liquid carrying
    // dispersed sub-grid gas nuclei.  The ordinary relaxation gives that gas a share of the energy defect in
    // proportion to its volume, and with ~1e-6 of the mass this shock-heated the trace air in a dodecane droplet
    // to ~1.4e4 K (port1118/tfix_on).  Nuclei that small reach the liquid temperature in well under a step.
    //   1 = pressure AND temperature equilibrium (homogeneous-relaxation / 4-equation state, closed form from
    //       the masses and rho e only): the gas compresses and expands with the liquid at the liquid T.
    //       Applied only where that state has p >= 0; liquid in tension takes the ordinary relaxation (cavitation).
    //   3 = alpha and phase densities frozen, the trace gas takes the common temperature (closed form).
    // Total energy is conserved exactly (E0 + E1 = rho e).  Phase convention independent (flames2 had phase 0 =
    // liquid; here the trace phase is identified by its stiffness).
    // (relax.trace_pT itself is parsed with the other relaxation options, modes 0 - 3; merge 2026-11-23.  The trace
    // phase is the softer one, below relax.trace_alpha by volume; default 1e-4 = the flames2 decks' cutoff.)
    pp.query_default("relax.trace_alpha", value.relax_trace_alpha, 1.0e-4);
    if (!(value.relax_trace_alpha > 0.0) || value.relax_trace_alpha >= 0.5)
        Util::Abort(INFO, "relax.trace_alpha must lie in (0, 0.5) (got ", value.relax_trace_alpha, ")");
    {
        // How the finite-rate step shares the total-energy defect between the phases: alpha (volume fractions,
        // Favrie & Schmidmayer 2026 eq. 113; default) or mass (mass fractions, their remark below eq. 119)
        std::string share = "alpha";
        pp.query_validate("relax.energy_share", share, {"alpha", "mass"});
        value.relax_share_mass = (share == "mass") ? 1 : 0;
        // relax.reinit_share was removed in the 2026-11-23 merge: the merged relaxation always shares the
        // total-energy defect of the instantaneous step by MASS fraction (flames2), which was its "mass" option.
        if (pp.contains("relax.reinit_share"))
            Util::Warning(INFO, "relax.reinit_share is IGNORED: the energy defect is always shared by mass (merge 2026-11-23)");
        // Path of each phase to the relaxed pressure: sc (Schmidmayer 2020 eq. 25, default) or isentropic (see EOS.H)
        std::string rpath = "sc";
        pp.query_validate("relax.path", rpath, {"sc", "isentropic"});
        Solver::EOS::EOS::relax_isentropic = (rpath == "isentropic") ? 1 : 0;
    }
    pp.query_default("shock_fix.ratio", value.shock_fix_ratio, 0.5);
    if (value.shock_fix && value.elastic.on)
    {
        Util::Warning(INFO, "shock_fix.on ignored: the HLLE flux carries no solid stress (elastic.on = 1)");
        value.shock_fix = 0;
    }
    if (value.shock_fix) value.shockfix_solver = new Solver::Local::FluidRiemann::HLLE();


    // LIMITER / primitive-variable reconstruction.
    // Selected by name: Limiter.type = godunov | minmod | vanleer | weno3 | weno5 | muscl | muscl2 | thinc
    // Default = godunov (no reconstruction; first-order behavior preserved).
    {
        std::string limiter_name;
        pp.query("Limiter.type", limiter_name);
        Util::Message(INFO, "Input file has Limiter.type = ", limiter_name);
        // [OPT/FIX 2026-09-24] Each face stencil spans 3 cells beyond the box;
        // WENO5 is the only limiter that uses the outermost two, so it needs
        // nghost >= 3 (use 4).  With fewer ghosts those points used to be
        // out-of-bounds reads; they are now clamped to the last ghost cell.
        if ((limiter_name == "weno5" || limiter_name == "WENO5") && value.nghost < 3)
            Util::Warning(INFO, "Limiter.type = weno5 with nghost = ", value.nghost,
                          ": the 5-point stencil needs nghost >= 3 (set nghost = 4). "
                          "Box-edge faces will use clamped (zero-gradient) ghost values.");
        // [2026-11-24] default = MUSCL2 (the 06 Cattaneo configuration; was Godunov)
        pp.select_default<Solver::Local::Limiter::MUSCL2,   // Sch20 MUSCL2: MC slope (pair with the midpoint RK tableau)
                          Solver::Local::Limiter::Godunov,  // 1st Order (i.e. no limiter)
                          Solver::Local::Limiter::Minmod,
                          Solver::Local::Limiter::VanLeer,
                          Solver::Local::Limiter::WENO3,    // 3rd Order
                          Solver::Local::Limiter::WENO5,    // 5th Order
                          Solver::Local::Limiter::MUSCL,    // 2nd order, Limiter.slope = minmod|mc|vanleer|superbee
                          Solver::Local::Limiter::THINC     // THINC alpha sharpening around Limiter.thinc.thinc_base
        >("Limiter", value.limiter);
        Util::Message(INFO, "Selected Limiter: ", typeid(*value.limiter).name());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// INITIALIZE //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Initialize(int lev)
{
    BL_PROFILE("Integrator::Hydro2::Initialize");
    // Initialize step counter
    if (step_counter.size() <= (size_t)lev)
    {
        step_counter.resize(lev + 1, 0);
    }
    step_counter[lev] = 0;


    // Initialize individual fluid variables
    // DIFFUSIVE BOUNDRY
    eta_ic          ->Initialize(lev, eta_mf, 0.0);
    eta_ic          ->Initialize(lev, eta_old_mf, 0.0);
    etadot_mf[lev]  ->setVal(0.0);
    relax_heat_mf[lev]->setVal(0.0);
    relax_J_mf[lev]->setVal(0.0);
    relax_split_mf[lev]->setVal(0.0);
    hess_eta_mf[lev]->setVal(0.0);

    // ------------------------------------------------------------------
    // SHELL (Marmottant): the AREAL shell density Gamma is the transported
    // primary.  The shell starts unstrained everywhere, Gamma_0 = 1, so
    // sigma(t=0) = chi(Gamma_buck - 1) exactly and uniformly across the whole
    // band.  Setting it to 1 in the bulk too costs nothing -- Gamma is only ever
    // read where |grad eta| is significant -- and avoids a spurious Gamma
    // gradient at the band edge that advection would otherwise pull inward.
    // ------------------------------------------------------------------
    shell_mf[lev]    ->setVal(1.0);
    shell_old_mf[lev]->setVal(1.0);
    // c is initialised FROM eta and normalised by its jump.  eta already runs
    // 0 -> 1 across the band, so [c] = 1 and c = eta at t = 0.  They diverge
    // afterwards: eta is relaxed and clamped, c only advects.
    amrex::MultiFab::Copy(*cfun_mf[lev],     *eta_mf[lev], 0, 0, 1, nghost);
    amrex::MultiFab::Copy(*cfun_old_mf[lev], *eta_mf[lev], 0, 0, 1, nghost);
    Gamma_mf[lev]    ->setVal(0.0);

    // SOURCE TERMS
    Fw_mf[lev]      ->setVal(0.0);
    Fsv_mf[lev]     ->setVal(0.0);
    Vap_dot_mf[lev] ->setVal(0.0);

    // Pre-zero per-phase velocity/momentum so inputs that specify only the in-plane
    // components (region0/region1) leave the out-of-plane (z) component at 0.  The
    // expression IC writes exactly as many components as regionN entries provided,
    // so a 3-component field driven by a 2-region IC keeps vz = 0 here; supply
    // velocity{0,1}.ic.expression.region2 to drive a genuine 3D velocity field.
    velocity0_mf[lev]->setVal(0.0);
    velocity1_mf[lev]->setVal(0.0);
    momentum0_mf[lev]->setVal(0.0);
    momentum1_mf[lev]->setVal(0.0);

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

    // EMBEDDED SOLID BOUNDARY: indicator + prescribed solid target state.
    if (embedded.apply)
    {
        embedded.phi_ic     ->Initialize(lev, embedded.phi_mf, 0.0);
        embedded.phi_ic     ->Initialize(lev, embedded.phi_old_mf, 0.0);
        embedded.grad_phi_mf[lev]    ->setVal(0.0);
        // Derive the per-phase solid target (density0/1, energy0/1, momentum)
        // from the single solid density + pressure, split by local eta.
        // Must run BEFORE the velocity blend below (it reads density0/1, M).
        InitEmbeddedSolidTarget(lev, 0.0);

        // ----------------------------------------------------------------
        // Initialize the embedded solid AT REST (Mix-style velocity blend).
        // ----------------------------------------------------------------
        // The fluid IC sets the velocity over the WHOLE domain, including
        // inside the solid.  Blend it toward the solid velocity u_solid by phi
        //   velocity <- phi*velocity + (1-phi)*u_solid
        // so the solid region starts quiescent.  Without this, the Yang
        // full-flux porous wall (no cutoff freeze) starts the dense solid
        // interior moving at the freestream and shocks itself apart at startup
        // (the low-Mach viscous Riemann abort).  u_solid = solid_M/rho_solid
        // (= 0 for a static solid).  Mix() below then builds the conserved
        // state from the blended primitives.
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi   = embedded.phi_mf[lev]->array(mfi);
            auto vel0  = velocity0_mf[lev]->array(mfi);
            auto vel1  = velocity1_mf[lev]->array(mfi);
            auto s_re0 = embedded.density0_mf[lev]->array(mfi);
            auto s_re1 = embedded.density1_mf[lev]->array(mfi);
            auto s_M   = embedded.momentum_mf[lev]->array(mfi);
            const Set::Scalar small_loc = small;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar w     = std::min(std::max(phi(i, j, k), 0.0), 1.0);
                const Set::Scalar rho_s = std::max(s_re0(i, j, k) + s_re1(i, j, k), small_loc);
                const Set::Scalar us_x  = s_M(i, j, k, 0) / rho_s;
                const Set::Scalar us_y  = s_M(i, j, k, 1) / rho_s;
                vel0(i, j, k, 0) = w * vel0(i, j, k, 0) + (1.0 - w) * us_x;
                vel0(i, j, k, 1) = w * vel0(i, j, k, 1) + (1.0 - w) * us_y;
                vel1(i, j, k, 0) = w * vel1(i, j, k, 0) + (1.0 - w) * us_x;
                vel1(i, j, k, 1) = w * vel1(i, j, k, 1) + (1.0 - w) * us_y;
            });
        }

        // grad(phi) diagnostic (static; for the boundary-refinement plot).
        const Set::Scalar *DXp = geom[lev].CellSize();
        embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto grad_phi = embedded.grad_phi_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector g = Numeric::Gradient(phi, i, j, k, 0, DXp);
                grad_phi(i, j, k, 0) = g(0);
                grad_phi(i, j, k, 1) = g(1);
            });
        }

        // [2026-09-25] phi interface thickness on this level, in cells:
        // ~ 1 / (largest jump of the clamped phi between face neighbours).
        // Used by the solid.wall_flux auto selection (TimeStepBegin).
        {
            Set::Scalar jmax = 0.0;
            for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                auto phi = embedded.phi_mf[lev]->const_array(mfi);
                amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                    const Set::Scalar pc = EmbeddedSolid::clampPhi(phi(i, j, k));
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int ii = i + (d == 0), jj = j + (d == 1), kk = k + (d == 2);
                        if (!bx.contains(amrex::IntVect(AMREX_D_DECL(ii, jj, kk)))) continue;
                        jmax = std::max(jmax, std::abs(EmbeddedSolid::clampPhi(phi(ii, jj, kk)) - pc));
                    }
                });
            }
            amrex::ParallelDescriptor::ReduceRealMax(jmax);
            if ((int)phi_thickness_cells.size() <= lev) phi_thickness_cells.resize(lev + 1, NAN);
            phi_thickness_cells[lev] = (jmax > 0.0) ? 1.0 / jmax : std::numeric_limits<Set::Scalar>::infinity();
        }
    }

    // FILLING GHOST CELLS
    rho_eta0_mf[lev]->setVal(0.0);
    rho_eta1_mf[lev]->setVal(0.0);
    rho_eta0_old_mf[lev]->setVal(0.0);
    rho_eta1_old_mf[lev]->setVal(0.0);

    // Pre-zero the mixture momentum/velocity (like the conserved fields above) so
    // every cell of the grown box is finite before the time=0 Mix() runs over it
    // (Mix() then overwrites valid+ghost from the per-phase fields).
    momentum_mf[lev]->setVal(0.0);
    momentum_old_mf[lev]->setVal(0.0);
    velocity_mf[lev]->setVal(0.0);

    energy_per_vol_mf[lev]->setVal(0.0);
    energy_per_mas_mf[lev]->setVal(0.0);
    energy_per_vol_old_mf[lev]->setVal(0.0);
    energy_per_mas_old_mf[lev]->setVal(0.0);

    UE_per_vol_mf[lev]->setVal(0.0);
    UE_per_mas_mf[lev]->setVal(0.0);
    KE_per_vol_mf[lev]->setVal(0.0);
    KE_per_mas_mf[lev]->setVal(0.0);

    // Calculate mixed variables based on individual fluid variables
    if (species.on && species.eos_mix)      // Mix() builds the phase energies from the pressure: it needs the mass
    {                                       // fractions (as weights, ghost cells included) before the partial densities exist
        species.mf[lev]->setVal(0.0);
        species.ic->Initialize(lev, species.mf, 0.0);
    }
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

    // Reflux scratch: FluxRegister (lev>0) + per-direction cc_fluxes.
    // Initialize covers lev=0 (Regrid is never called on level 0); Regrid
    // covers lev>0 via MakeNewLevelFromCoarse / RemakeLevel.
    AllocateRefluxScratch(lev);

    if (elastic.on && elastic.plastic) elastic.plastic_mf[lev]->setVal(0.0);
    if (elastic.on) elastic.Init(lev, geom[lev], *rho_eta0_mf[lev], *rho_eta1_mf[lev], *energy_per_vol_mf[lev], *energy_per_vol_old_mf[lev], (species.on ? species.mf[lev].get() : nullptr));
    if (phasechange.on)     // dry gas at t = 0 (phasechange.Yv0 > 0: that vapour mass fraction everywhere)
    {
        const Set::Scalar Yv0 = phasechange.Yv0;
        amrex::MultiFab::Copy(*phasechange.vapor_mf[lev], (phasechange.liquid == 1) ? *rho_eta0_mf[lev] : *rho_eta1_mf[lev], 0, 0, 1, 0);
        phasechange.vapor_mf[lev]->mult(Yv0, 0, 1, 0);
        eta_bc->define(geom[lev]);      // (an undefined BC treats cells beyond ITS domain as ghosts and overwrites valid data)
        FillBoundariesWithBC(lev, 0.0, eta_bc, { phasechange.vapor_mf[lev].get() });
        amrex::MultiFab::Copy(*phasechange.vapor_old_mf[lev], *phasechange.vapor_mf[lev], 0, 0, 1, phasechange.vapor_mf[lev]->nGrow());
        phasechange.diag_mf[lev]->setVal(0.0);
    }
    if (ablation.on)
    {
        ablation.q_mf[lev]->setVal(0.0); ablation.v_mf[lev]->setVal(0.0);
        if (!embedded.apply || !embedded.wall_flux || !thermal_conduction)
            Util::Abort(INFO, "ablation.on needs apply_embedded_solid = 1, solid.wall_flux = 1 and thermal.conduction = 1");
        if (embedded.moving) Util::Abort(INFO, "ablation.on with solid.moving is not supported");
        if (ablation.conduction)
        {
            ablation.Ts_mf[lev]->setVal(ablation.T_solid0);
            ablation.qs_mf[lev]->setVal(0.0);
            if (ablation_Ts_ic) ablation_Ts_ic->Initialize(lev, ablation.Ts_mf, 0.0);
        }
    }

    if (elastic.on && elastic.comp_on && marmottant)
        Util::Abort(INFO, "elastic.compaction = 1 uses the colour function field and cannot be combined with marmottant = 1");
    if (species.on && species.pele.on && ((species.pele.ph == 0 ? eos0.P0() : eos1.P0()) != 0.0))
        Util::Abort(INFO, "species.eos = pele: the phase that carries the mechanism species must have p0 = 0 (ideal gases)");
    if (species.on && species.eos_mix && (nscbc_bc || nscbc4_bc))
        Util::Abort(INFO, "species.eos = mixture is not available with the NSCBC boundaries yet (they use the phase constants)");
    if (species.on)     // mass fractions from species.ic -> partial densities, majority species closed to the phase mass
    {
        species.Y_mf[lev]->setVal(0.0);
        species.ic->Initialize(lev, species.Y_mf, 0.0);
        species.mf[lev]->setVal(0.0);
        for (int n = 0; n < species.ns(); ++n)
            amrex::MultiFab::Copy(*species.mf[lev], (n < species.n0) ? *rho_eta0_mf[lev] : *rho_eta1_mf[lev], 0, n, 1, 0);
        amrex::MultiFab::Multiply(*species.mf[lev], *species.Y_mf[lev], 0, 0, species.ns(), 0);
        if (chem.on) chem.heat_mf[lev]->setVal(0.0);
        CloseSpecies(lev, false);     // majority species of each cell takes the remainder of the phase mass
        CloseSpecies(lev, true);      // mass fractions for the plot file of step 0
        species.bc->define(geom[lev]);
        FillBoundariesWithBC(lev, 0.0, species.bc, { species.mf[lev].get() });
        amrex::MultiFab::Copy(*species.old_mf[lev], *species.mf[lev], 0, 0, species.ns(), species.mf[lev]->nGrow());
    }
    if (entropy_on) ComputeEntropy(lev);     // after the solid temperature is set (the plot file of step 0)

    Util::ParallelMessage(INFO, "Finished initialization, begginning time iteration");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// MIX /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Mix(int lev)
{
    // Six-equation initial-condition construction.
    // ------------------------------------------------------------------------
    // Inputs (per-phase IC fields populated by Initialize()):
    //   eta_mf       = alpha_1            (Schmidmayer 2020 eq. 13 first row)
    //   density{0,1} = rho_k (pure)       (per-phase IC)
    //   velocity{0,1}= u_k    (per-phase IC; mechanical equilibrium assumed)
    //   pressure{0,1}= p_k    (per-phase IC; usually equal at IC)
    //
    // Outputs (canonical 6-eq conservative primaries):
    //   eta            = alpha_1                          (Sch20 eq. 13)
    //   rho_eta0       = (alpha_1 rho_1)                  (canonical phase mass)
    //   rho_eta1       = (alpha_2 rho_2)
    //   density (mix)  = alpha_1 rho_1 + alpha_2 rho_2    (Sch20 eq. 8)
    //   M              = mixture momentum  rho u
    //   energy0        = alpha_1 (p + gam0 pi0)/(gam0-1)  (Sau09 eq. III.5)
    //   energy1        = alpha_2 (p + gam1 pi1)/(gam1-1)
    //   energy_per_vol = E0 + E1 + KE     = redundant rho E   (Sch20 eq. 16)
    // ------------------------------------------------------------------------
    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();
    (void)domain;

    // Function is for the diffusive mixing terms (6-eq canonical form).
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox();

        const int cj_ic = apply_surface_tension;   // phases keep their IC pressures (see below)
        // Capillary-energy closure inputs (see Hydro2.H).
        const int marm_ic = marmottant;
        const Set::Scalar sig_ic = sigma, chi_ic = marmottant_chi;
        const Set::Scalar Gb_ic = marmottant_Gamma_buck;
        const Set::Scalar sbrk_ic = marmottant_sigma_break, sigw_ic = sigma;
        amrex::Array4<const Set::Scalar> const &shell_ic = shell_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> const cfun_ic  = eta_mf[lev]->const_array(mfi);

        // DIFFUSIVE BOUNDRY
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta0_old = rho_eta0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_eta1_old = rho_eta1_old_mf.Patch(lev, mfi);

        // FLUID 0 (per-phase IC and primary canonical (alpha rho e)_0 written)
        Set::Patch<const Set::Scalar>   v0          = velocity0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p0          = pressure0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0        = density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho0_old    = density0_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0          = momentum0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M0_old      = momentum0_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0_arr      = energy0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E0_old_arr  = energy0_old_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   T0          = T0_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   k0_thermal  = k0_thermal_mf.Patch(lev, mfi);
        //Set::Patch<const Set::Scalar>   h0_thermal  = h0_thermal_mf.Patch(lev, mfi);

        // FLUID 1 (per-phase IC and primary canonical (alpha rho e)_1 written)
        Set::Patch<const Set::Scalar>   v1          = velocity1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   p1          = pressure1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1        = density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   rho1_old    = density1_old_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1          = momentum1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar>   M1_old      = momentum1_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_arr      = energy1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         E1_old_arr  = energy1_old_mf.Patch(lev, mfi);
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
        const int Tpc = T_phase_consistent;     // temperature.phase_consistent
        auto tph = Tpc ? Tph_mf[lev]->array(mfi) : amrex::Array4<Set::Scalar>{};
        Set::Patch<Set::Scalar>         cp          = cp_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         cv          = cv_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar>         k_thermal   = k_thermal_mf.Patch(lev, mfi);
        //Set::Patch<Set::Scalar>         h_thermal   = h_thermal_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         gammaf      = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         p0_eff      = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         mu_chem_    = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Bm          = Bm_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Y           = Y_mf.Patch(lev, mfi);

        // EXTRAS & DEBUGGING
        Set::Patch<Set::Scalar>         a           = a_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         Ma          = Ma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_vol      = UE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         UE_mas      = UE_per_mas_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_vol      = KE_per_vol_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar>         KE_mas      = KE_per_mas_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_base = eos1;

        // EOS constants (Tammann/SG) for the per-phase initialization.
        
        const Species::Eos spe_l = species.eos();     // species.eos = mixture: gamma, c_v of each phase per cell
        amrex::Array4<const Set::Scalar> sp_l; if (spe_l.on) sp_l = species.mf[lev]->const_array(mfi);

        // [FIX 2026-09-24] bx is the FULL ghost-grown box, so on the outermost
        // ghost ring the +-1 Laplacian stencil read past the end of eta's
        // allocation -- undefined values there, and an intermittent SEGFAULT
        // at initialization when the fab sits at the end of its memory block
        // (reproduced: UNIT_TEST_3D_Garrick on 1 rank crashed 2 of 4 runs, old
        // and new binaries alike).  Only that ring is affected, and mu_chem
        // ghosts are refilled by FillBoundariesWithBC before any use.
        amrex::Array4<const Set::Scalar> const eta_c = eta_mf[lev]->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Solver::EOS::Tammann eos0_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 0, eos0_base, spe_l.EsFromPRho(sp_l, i, j, k, 0, p0(i, j, k), rho0(i, j, k))) : eos0_base;
            const Solver::EOS::Tammann eos1_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 1, eos1_base, spe_l.EsFromPRho(sp_l, i, j, k, 1, p1(i, j, k), rho1(i, j, k))) : eos1_base;
            const Set::Scalar gam0 = eos0_local.Gamma(), gam1 = eos1_local.Gamma();
            const Set::Scalar pi0_ = eos0_local.P0(), pi1_ = eos1_local.P0();

            // Eta laplacian (used below for the chemical-potential diagnostic).
            const bool lap_fits = eta_c.contains(i - 1, j, k) && eta_c.contains(i + 1, j, k)
                               && eta_c.contains(i, j - 1, k) && eta_c.contains(i, j + 1, k)
#if AMREX_SPACEDIM > 2
                               && eta_c.contains(i, j, k - 1) && eta_c.contains(i, j, k + 1)
#endif
                               ;
            Set::Scalar lap_eta = lap_fits ? Numeric::Laplacian(eta, i, j, k, 0, DX) : 0.0;

            // --- 6-equation canonical state at IC (Schmidmayer 2020 Sec2.3) ---
            // Volume fractions:
            const Set::Scalar a1 = eta(i, j, k);              // alpha_1
            const Set::Scalar a2 = 1.0 - a1;                  // alpha_2

            // Phase masses (canonical):
            rho_eta0(i, j, k)     = a1 * rho0(i, j, k);       // (alpha_1 rho_1)
            rho_eta1(i, j, k)     = a2 * rho1(i, j, k);       // (alpha_2 rho_2)
            rho_eta0_old(i, j, k) = rho_eta0(i, j, k);
            rho_eta1_old(i, j, k) = rho_eta1(i, j, k);

            // Mixture density (Schmidmayer 2020 eq. 8):
            rho(i, j, k)     = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            rho_old(i, j, k) = rho(i, j, k);

            // Mixture momentum from per-phase mass-weighted velocities:
            //   M = (alpha_1 rho_1) u_0 + (alpha_2 rho_2) u_1
            // PER-PHASE velocity fields (v0/v1) are full AMREX_SPACEDIM vectors; the
            // IC may drive every component (region0/region1/region2).  Build all
            // momentum components from them -- in 3D the z-momentum is a genuine DOF.
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                M(i, j, k, d) = rho_eta0(i, j, k) * v0(i, j, k, d) + rho_eta1(i, j, k) * v1(i, j, k, d);

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                M_old(i, j, k, d) = M(i, j, k, d);

            // Mixture velocity (diagnostic):
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            // Mixture kinetic energy (consistent with rho E):
            KE_vol(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM( v(i, j, k, 0) * v(i, j, k, 0),
                                                                + v(i, j, k, 1) * v(i, j, k, 1),
                                                                + v(i, j, k, 2) * v(i, j, k, 2)));
            KE_mas(i, j, k) = (rho(i, j, k) > small) ? KE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // ===========================================================
            // MECHANICAL-EQUILIBRIUM INITIAL CONDITION
            //   p_mix = alpha_1 p0_ic + alpha_2 p1_ic    (Schmidmayer eq. 8)           
            //   (alpha rho e)_k = alpha_k (p_mix + gamma_k pi_k) / (gamma_k - 1)
            //                                              (Sau09 eq. III.5)
            // equalize_ic_pressure:
            //   0 (legacy)  : p_mix = a1*p0 + a2*p1                 (linear average)
            //   1 (matches  : p_mix = (a1*p0/(g0-1) + a2*p1/(g1-1)) /
            //     RelaxAnd                (a1/(g0-1) + a2/(g1-1))
            //     Reinit)     gamma-weighted form -- this is exactly the
            //                 pressure that the energy-conserving relaxation
            //                 (Sch20 eq. 26 / Sau09 eq. III.5) would yield
            //                 from these per-phase IC pressures, so step-1
            //                 RelaxAndReinit produces no IC pressure kick.
            // ===========================================================
            const Set::Scalar p0_ij = p0(i, j, k);
            const Set::Scalar p1_ij = p1(i, j, k);
            Set::Scalar p_mix_IC;
            if (equalize_ic_pressure == 0)
            {
                p_mix_IC = a1 * p0_ij + a2 * p1_ij;
            }
            else
            {
                const Set::Scalar denom_a = a1 / (gam0 - 1.0) + a2 / (gam1 - 1.0);
                const Set::Scalar numer_a = a1 * p0_ij / (gam0 - 1.0)
                                          + a2 * p1_ij / (gam1 - 1.0);
                p_mix_IC = numer_a / std::max(denom_a, small);
            }

            // With surface tension the phases keep their own IC pressures (p_gas -
            // p_liq is the Laplace jump the relaxation then maintains); the
            // equal-pressure p_mix_IC would make the first jump relaxation drop
            // the band liquid by the full jump.
            const bool keep_phase_p = (cj_ic != 0);
            E0_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(keep_phase_p ? p0_ij : p_mix_IC, a1, gam0, pi0_, small);
            E1_arr(i, j, k)     = Solver::EOS::EOS::PhasicEnergyFromPressure(keep_phase_p ? p1_ij : p_mix_IC, a2, gam1, pi1_, small);
            E0_old_arr(i, j, k) = E0_arr(i, j, k);
            E1_old_arr(i, j, k) = E1_arr(i, j, k);

            // Mixture internal energy (sum of canonical per-phase energies):
            UE_vol(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);
            UE_mas(i, j, k) = (rho(i, j, k) > small) ? UE_vol(i, j, k) / rho(i, j, k) : 0.0;

            // Redundant total energy rho E (Schmidmayer 2020 eq. 16), plus the
            // capillary energy when the closure is on (Schmidmayer 2017).
            E_vol(i, j, k)     = KE_vol(i, j, k) + UE_vol(i, j, k);
            E_vol_old(i, j, k) = E_vol(i, j, k);
            E_mas(i, j, k)     = KE_mas(i, j, k) + UE_mas(i, j, k);
            E_mas_old(i, j, k) = E_mas(i, j, k);

            // Diagnostic specific-heat-ratio and ref-pressure (for plotfile only).
            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);

            // Mixture pressure
            press(i, j, k) = p_mix_IC;

            // Chemical Potential
            // Set::Scalar f_prime = 4.0 * eta(i, j, k) * (eta(i, j, k) - 0.5) * (eta(i, j, k) - 1.0); // Double-well potential derivative: f'(eta) = 4*eta*(eta-0.5)*(eta-1)
            Set::Scalar f_prime = 4.0 * eta(i, j, k) * (0.5 - eta(i, j, k)) * (1.0 - eta(i, j, k)); // Flipped Sign?
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i, j, k) = mu_chem;

            // Mass Fraction
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y))
            Bm(i, j, k) = SpaldingBM(Y(i, j, k), Y_infinity, small);

            // Temperature (diagnostic; not used by hyperbolic step)
            if (Tpc)
            {
                const Species::Eos &se_t = spe_l;
                const bool pe0 = se_t.on && se_t.th.on && se_t.th.ph == 0, pe1 = se_t.on && se_t.th.on && se_t.th.ph == 1;
                Set::Scalar t0, t1;
                T(i, j, k) = PhaseConsistentT(E0_arr(i, j, k), E1_arr(i, j, k), rho_eta0(i, j, k), rho_eta1(i, j, k), std::min(std::max(eta(i, j, k), 0.0), 1.0), eos0_local, eos1_local, pe0, pe1, t0, t1);
                tph(i, j, k, 0) = t0; tph(i, j, k, 1) = t1;
            }
            else T(i, j, k) = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), eta(i, j, k), eos0_local, eos1_local, pref);

            // Speed of sound -- FROZEN mixture sound speed.
            // Schmidmayer 2020 eq. 17 / Saurel 2009 eq. III.2:
            //   c^2 = Y_1 c_1^2 + Y_2 c_2^2
            // Use the mechanical-equilibrium IC pressure p_mix_IC (both
            // per-phase pressures are = p_mix_IC at this point).
            {
                const Set::Scalar Y0 = rho_eta0(i, j, k) / std::max(rho(i, j, k), small);
                const Set::Scalar Y1 = 1.0 - Y0;
                const Set::Scalar rho1pure = rho_eta0(i, j, k) / std::max(a1, small);
                const Set::Scalar rho2pure = rho_eta1(i, j, k) / std::max(a2, small);
                const Set::Scalar c0_ph = Solver::EOS::EOS::PhasicSoundSpeed(rho1pure, p_mix_IC, gam0, pi0_, small);
                const Set::Scalar c1_ph = Solver::EOS::EOS::PhasicSoundSpeed(rho2pure, p_mix_IC, gam1, pi1_, small);
                a(i, j, k) = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_ph, c1_ph);
            }

            // Mach Number
            Ma(i, j, k, 0) = v(i, j, k, 0) / a(i, j, k);
            Ma(i, j, k, 1) = v(i, j, k, 1) / a(i, j, k);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(0, lev, i, j, k, "ERROR IN Mix(): Primative Field Calculation", {
                { "rho_eta0", rho_eta0(i, j, k) }, 
                { "rho_eta1", rho_eta1(i, j, k) }, 
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
    vz_max = 0.0;
    F_max = 0.0;
    rho_min = 1e10;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////// TIMESTEPBEGIN ////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::TimeStepBegin(Set::Scalar, int /*iter*/)
{
    // [2026-09-25] solid.wall_flux AUTO selection (see Parse), decided once,
    // before the first step, from the finest level measured at initialization.
    if (embedded.apply && embedded.wall_flux_auto && !wall_flux_decided)
    {
        wall_flux_decided = true;
        int L = -1;
        for (int l = (int)phi_thickness_cells.size() - 1; l >= 0; --l)
            if (phi_thickness_cells[l] == phi_thickness_cells[l]) { L = l; break; }
        if (L < 0)
        {
            Util::Warning(INFO, "embedded solid: phi interface thickness not measured (restart?) -- "
                                "keeping the diffuse wall (solid.wall_flux = 0); set solid.wall_flux explicitly if needed");
            return;
        }
        const Set::Scalar th = phi_thickness_cells[L];
        if (th < embedded.wall_flux_cells)
        {
            embedded.wall_flux = 1;
            Util::Message(INFO, "embedded solid: phi interface thickness = ", th, " cells on level ", L,
                          " (< solid.wall_flux_cells = ", embedded.wall_flux_cells, "): the solid is too thin to be "
                          "resolved as a diffuse wall -- switching to the MIRRORED sharp-wall flux (solid.wall_flux = 1)");
        }
        else
        {
            embedded.wall_flux = 0;
            Util::Message(INFO, "embedded solid: phi interface thickness = ", th, " cells on level ", L,
                          " (>= solid.wall_flux_cells = ", embedded.wall_flux_cells, "): keeping the diffuse wall (solid.wall_flux = 0)");
        }
    }
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
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////// RHS /////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void
Hydro2::RHS(int lev,
    Set::Scalar time,
    amrex::MultiFab &rho_eta0_rhs_mf,
    amrex::MultiFab &rho_eta1_rhs_mf,
    amrex::MultiFab &M_rhs_mf,
    amrex::MultiFab &E_rhs_mf,
    amrex::MultiFab &eta_rhs_mf,
    amrex::MultiFab &E0_rhs_mf,
    amrex::MultiFab &E1_rhs_mf,
    amrex::MultiFab &shell_rhs_mf,
    amrex::MultiFab &cfun_rhs_mf,
    const amrex::MultiFab &rho_eta0_mf_in,
    const amrex::MultiFab &rho_eta1_mf_in,
    const amrex::MultiFab &M_mf_in,
    const amrex::MultiFab &E_mf_in,
    const amrex::MultiFab &eta_mf_in,
    const amrex::MultiFab &E0_mf_in,
    const amrex::MultiFab &E1_mf_in,
    const amrex::MultiFab &shell_mf_in,
    const amrex::MultiFab &cfun_mf_in,
    amrex::MultiFab *ebasis_rhs_mf,
    const amrex::MultiFab *ebasis_mf_in,
    amrex::MultiFab *vapor_rhs_mf,
    const amrex::MultiFab *vapor_mf_in,
    amrex::MultiFab *species_rhs_mf,
    const amrex::MultiFab *species_mf_in)
{
    BL_PROFILE("Integrator::Hydro2::RHS");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Converting Array to mf  (6-eq primaries -- Sch20 eq. 13 + 16)
    amrex::MultiFab::Copy(*rho_eta0_mf[lev],       rho_eta0_mf_in, 0, 0, 1,              0);
    amrex::MultiFab::Copy(*rho_eta1_mf[lev],       rho_eta1_mf_in, 0, 0, 1,              0);
    amrex::MultiFab::Copy(*momentum_mf[lev],       M_mf_in,        0, 0, AMREX_SPACEDIM, 0);
    amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_mf_in,        0, 0, 1,              0);
    amrex::MultiFab::Copy(*eta_mf[lev],            eta_mf_in,      0, 0, 1,              0);
    amrex::MultiFab::Copy(*energy0_mf[lev],        E0_mf_in,       0, 0, 1,              0);
    amrex::MultiFab::Copy(*energy1_mf[lev],        E1_mf_in,       0, 0, 1,              0);
    amrex::MultiFab::Copy(*shell_mf[lev],          shell_mf_in,    0, 0, 1,              0);
    shell_mf[lev]->FillBoundary(geom[lev].periodicity());
    amrex::MultiFab::Copy(*cfun_mf[lev],           cfun_mf_in,     0, 0, 1,              0);
    cfun_mf[lev]->FillBoundary(geom[lev].periodicity());
    if (ablation.on) ablation.q_mf[lev]->setVal(0.0);   // wall heat flux of this RHS evaluation (set in the conduction block)
    if (phasechange.on)   // ghosts are filled by FillGhost4BC below
        amrex::MultiFab::Copy(*phasechange.vapor_mf[lev], *vapor_mf_in, 0, 0, 1, 0);
    if (species.on)       // ghosts are filled by FillGhost4BC below
        amrex::MultiFab::Copy(*species.mf[lev], *species_mf_in, 0, 0, species.ns(), 0);
    if (elastic.on)   // ghosts are filled by FillGhost4BC below
        amrex::MultiFab::Copy(*elastic.ebasis_mf[lev], *ebasis_mf_in, 0, 0, ElasticSolid::NEB, 0);

    // Eta Fields

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(velocity_mf)[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> eta = eta_mf.Patch(lev, mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // Primitive Fields (with BCs)
    FillGhost4BC(lev, time);

    // Pre-Source Terms
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*(velocity_mf)[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto const rho_eta1 = rho_eta1_mf[lev]->array(mfi);
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

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_local = eos0;
        const Solver::EOS::Tammann eos1_local = eos1;

        const int symc = cen_face_any;   // symmetry + RADIAL faces
        const bool symlo_c[3] = {cen_face_lo[0], cen_face_lo[1], cen_face_lo[2]};
        const bool symhi_c[3] = {cen_face_hi[0], cen_face_hi[1], cen_face_hi[2]};
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            if (symc)        // mirrored ghosts -> central is exact (see Parse)
            {
                const amrex::Dim3 dlo = amrex::lbound(domain), dhi = amrex::ubound(domain);
                const int ijk[3] = {i, j, k}, lo3[3] = {dlo.x, dlo.y, dlo.z}, hi3[3] = {dhi.x, dhi.y, dhi.z};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    if ((symlo_c[d] && ijk[d] == lo3[d]) || (symhi_c[d] && ijk[d] == hi3[d]))
                        sten[d] = Numeric::StencilType::Central;
            }

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
            Y(i, j, k) = rho_eta0(i, j, k) / (rho(i, j, k));

            // Spalding Number  (F-1 / F-10: single canonical helper, denominator (1 - Y))
            Bm(i, j, k) = SpaldingBM(Y(i, j, k), Y_infinity, small);

            // Curvature (CSF formula on eta; PLOT FIELD kappas[0] only -- the force and
            // the relaxation jump use TensorCurvature).  kappas[1] is overwritten with the
            // total tension by the surface-tension block.
            {
                // (grad|grad eta|)_a = (H grad_eta)_a/|grad eta|  (hess_eta symmetric)
                Set::Vector grad_mag_grad_eta = (1.0 / (grad_eta_mag + small)) * (hess_eta * grad_eta);
                kappas(i, j, k, 0) = -((lap_eta / (grad_eta_mag + small)) - (grad_eta.dot(grad_mag_grad_eta) / (grad_eta_mag * grad_eta_mag + small)));
                kappas(i, j, k, 1) = 0.0;
                kappas(i, j, k, 2) = 0.0;
            }

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN RHS(): Primative Field Calculation", {
                { "rho_eta0", rho_eta0(i, j, k) }, 
                { "rho_eta1", rho_eta1(i, j, k) }, 
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

    // Fill Boundries
    FillBoundariesWithBC(lev, time, energy_bc, { mu_chem_mf[lev].get() });

    // ------------------------------------------------------------
    // MARMOTTANT (2005) coated-bubble surface tension -- ADVECTED SHELL
    // ------------------------------------------------------------
    // sigma is a function of the shell's AREAL STRAIN, not of the local curvature.
    // The lipid monolayer is a MATERIAL carried by the interface, so we transport a
    // conserved shell density  c = Gamma |grad eta|  as an RK primary
    // (d(c)/dt + div(c u) = 0, discretized with the interface velocity below), and
    //
    //     Gamma = c / |grad eta|                        shell per unit AREA
    //     sigma = chi (Gamma_buck/Gamma - 1)            elastic
    //           = 0                                     buckled  (Gamma >= Gamma_buck)
    //           = sigma_water                           ruptured (elastic >= sigma_break)
    //
    // Gamma_buck/Gamma IS Marmottant's area ratio A/A_buck, so this is the same
    // sigma(A) law -- with A carried by the shell instead of re-derived from a
    // second derivative of eta every step.  What that buys, vs. the old sigma(R(x)):
    //
    //  * No curvature enters the tension, so there is nothing for the elastic law
    //    to amplify.  sigma(R) had d(sigma)/sigma = 2(R/Rb)^2/((R/Rb)^2-1) dR/R
    //    = 10.5x at R0/Rb = 1.11, which turned a 1% kappa error into 10% tension.
    //  * Gamma is uniform ACROSS the band by construction: c and |grad eta| share
    //    the same profile, so c/|grad eta| is flat through the band thickness.  The
    //    old per-cell R = (d-1)/kappa gave every cell in the band its OWN level-set
    //    radius, so with the band spanning R0 +- 3eps and R_buck inside that range,
    //    ~24% of band cells were permanently mis-classified "buckled" (sigma=0) even
    //    on a perfect circle -- and that fraction did NOT converge under refinement.
    //  * Shell material is conserved, so stretching genuinely dilutes Gamma and the
    //    rupture branch is reached for the right physical reason.
    //  * Tangential (Marangoni) redistribution of shell material is represented;
    //    eta alone cannot express it, since sliding material along a level set
    //    leaves eta identically unchanged.
    //  * Nonspherical shells and multiple bubbles need no labelling or geometry.
    //
    // sigma_eff carries one ghost cell so the capillary tensor Omega below can be
    // built in the ghost ring too (kappas_mf has zero ghosts).  It is evaluated on
    // the GROWN box directly from eta/shell (both of which have filled ghosts), so
    // the ghost ring is genuinely computed rather than left zero at physical faces.
    amrex::MultiFab sig_eff_mf;
    // shell.slip = 7: the shell field is m = Gamma |grad psi|; Gamma = m / Gp7 (ShellPsiGrad).
    const int slip7 = (shell_slip == 7 || shell_slip == 11 || shell_slip == 12 || shell_slip == 15) && (marmottant || apply_surface_tension);
    const int slip15 = (shell_slip == 15);   // shell.slip = 15: no fluid flux of m in the RHS (ShellRemap15)
    amrex::MultiFab Gp7;
    if (slip7) ShellPsiGrad(lev, time, Gp7);

    if (marmottant)
    {
        const amrex::BoxArray &ba = eta_mf[lev]->boxArray();
        const amrex::DistributionMapping &dm = eta_mf[lev]->DistributionMap();
        sig_eff_mf.define(ba, dm, 1, 1);
        sig_eff_mf.setVal(0.0);

        const Set::Scalar chi_ = marmottant_chi, Gb = marmottant_Gamma_buck;
        const Set::Scalar sbrk = marmottant_sigma_break, sigw = sigma;
        for (amrex::MFIter mfi(sig_eff_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &vbx = mfi.validbox();
            const amrex::Box gbx  = mfi.growntilebox(1);
            // The interface gate must use the SAME field the capillary tensor
            // is built from, or the shell tension and Omega end up gated on
            // two different interfaces once c and eta separate.
            amrex::Array4<const Set::Scalar> const &et = eta_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const &cs = shell_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const gp7 = slip7 ? Gp7.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            amrex::Array4<Set::Scalar> const &sg  = sig_eff_mf.array(mfi);
            amrex::Array4<Set::Scalar> const &Gm  = Gamma_mf[lev]->array(mfi);
            amrex::Array4<Set::Scalar> const &kap = kappas_mf[lev]->array(mfi);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                Set::Scalar gem = ge.lpNorm<2>();
                Set::Scalar se = 0.0, G = 0.0;
                // Gate on |grad eta|: off the interface there is no shell and no
                // area to normalise by.  Omega is identically zero there anyway.
                if (gem > g_gem_min)
                {
                    // Gamma is a PRIMARY -- read straight off the advected field.
                    // No division by |grad eta|, which is what made the previous
                    // c/|grad eta| reconstruction blow up in the band tails.
                    G = cs(i, j, k);
                    if (slip7) G = (gp7(i, j, k) > 0.0) ? cs(i, j, k) / gp7(i, j, k) : 1.0;   // m / |grad psi|
                    if (G > 0.0 && G < Gb)             // stretched -> elastic branch
                    {
                        Set::Scalar el = chi_ * (Gb / G - 1.0);
                        se = (el >= sbrk) ? sigw : el;
                    }
                    else if (g_shell_rupture && G <= 0.0) se = sigw;   // broken shell (shell.slip5_rupture)
                    // G >= Gb: compressed past buckling -> sigma = 0 (se stays 0)
                }
                sg(i, j, k) = se;
                if (vbx.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                {
                    Gm(i, j, k)     = G;
                    kap(i, j, k, 2) = se;   // sigma_eff diagnostic (plottable)
                }
            });
        }
        // Interior/periodic ghosts from neighbours; physical-face ghosts keep the
        // values computed on the grown box above.
        sig_eff_mf.FillBoundary(geom[lev].periodicity());
    }

    // ------------------------------------------------------------
    // CAPILLARY STRESS TENSOR  (Schmidmayer et al. 2017, Eq.3-4) -- the conservative
    // "continuum surface stress" form of surface tension.  The momentum force is
    //   F = div(Omega),   Omega_ij = sigma_eff (|grad eta| delta_ij - d_i eta d_j eta/|grad eta|).
    // div(Omega) reduces to sigma*kappa*grad(eta) for CONSTANT sigma, but for a VARYING
    // sigma it also carries the Marangoni (grad sigma) term automatically -- i.e. the
    // physically complete surface stress.  sigma_eff is the local Marmottant tension
    // kappas(.,2) (marmottant) or the constant sigma.  [The capillary energy
    // eps_sigma = sigma|grad eta| in the energy flux is the next step toward full
    // discrete conservation; it does not change the static momentum/Laplace balance.]
    // Symmetric-tensor storage:  2D [xx,yy,xy] ; 3D [xx,yy,zz,xy,xz,yz].
    amrex::MultiFab Omega;
    if (apply_surface_tension)
    {
        const amrex::BoxArray &baO = eta_mf[lev]->boxArray();
        const amrex::DistributionMapping &dmO = eta_mf[lev]->DistributionMap();
        const int nO = (AMREX_SPACEDIM == 2) ? 3 : 6;
        // Components nO .. nO+D-1 carry the normal jump force
        // F_n = -Delta_s grad(eta) per cell (see Parse, CONSISTENT JUMP).
        Omega.define(baO, dmO, nO + AMREX_SPACEDIM, 1);
        Omega.setVal(0.0);

        const Set::Scalar sig0 = sigma; const int marm = marmottant;
        // Boussinesq--Scriven dilatational shell viscosity.  sigma_tot =
        // sigma_eff + (kappa_s - mu_s) div_s u; mu_s is zero (the shear term is
        // not implemented and Parse aborts on a nonzero value), so kappa_s is
        // the whole of it.  This is the ONE place sigma enters Omega, so adding
        // it here covers the momentum source and the capillary work together.
        // sigma_tot = sigma_eff + (kappa_s - mu_s)(div_s u)  multiplies the
        // projector P; 2 mu_s D_s is a genuine tensor and is added on top.
        const Set::Scalar kap_s = shell_kappa_s - shell_mu_s;
        const int subcyc_visc = shell_visc_subcycle && (shell_kappa_s != 0.0);
        const Set::Scalar mu_s  = shell_mu_s;
        const int visc_shell = (kap_s != 0.0) || (mu_s != 0.0);
        const Set::Scalar chi_o = marmottant_chi, Gb_o = marmottant_Gamma_buck;
        const Set::Scalar sbrk_o = marmottant_sigma_break, sigw_o = sigma;
        // curvature of the normal jump force: the capillary-tensor curvature (TensorCurvature)
        amrex::MultiFab Kt_o(baO, dmO, 1, 0);
        TensorCurvature(lev, Kt_o, time);
        const int ffo = jump_face_force;
        amrex::MultiFab Jst(baO, dmO, 1, 1);   // TEMP jump.face_force: J = -sigma_eff kappa per valid cell
        Jst.setVal(0.0);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box bx = mfi.growntilebox(1);
            amrex::Array4<const Set::Scalar> const &kt_o = Kt_o.const_array(mfi);
            amrex::Array4<const Set::Scalar> const &shl_o = shell_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const gp7o = slip7 ? Gp7.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            amrex::Array4<const Set::Scalar> const &et = eta_mf[lev]->const_array(mfi);
            amrex::Array4<const Set::Scalar> const &sg = marm ? sig_eff_mf.const_array(mfi)
                                                              : amrex::Array4<const Set::Scalar>{};
            amrex::Array4<const Set::Scalar> const &vel = velocity_mf[lev]->const_array(mfi);
            amrex::Array4<Set::Scalar> const &om = Omega.array(mfi);
            amrex::Array4<Set::Scalar> const &jst = Jst.array(mfi);
            // Diagnostic: kappa1 carries the TOTAL surface tension actually
            // used in Omega, sigma(Gamma) + kappa_s div_s(u).  kappa2 is written in the Marmottant block
            // above and is the BARE sigma(Gamma) only, so it cannot show the
            // viscous contribution -- which is the whole question for the
            // coated collapse.
            amrex::Array4<Set::Scalar> const &kapd = kappas_mf[lev]->array(mfi);
            const amrex::Box vbx_om = mfi.validbox();
            // Omega is built on the VOLUME FRACTION.  Schmidmayer 2017 uses the
            // colour function instead, to keep the capillary energy consistent
            // with a conservative energy flux; that closure has been removed, so
            // grad(eta) is the only path and c is no longer read here.
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                Set::Scalar gem = ge.lpNorm<2>();
                // Normal static force F_n = -Delta_s grad(eta), Delta_s = sigma_eff(Gamma) kappa,
                // this cell's value only (only valid cells are read).
                {
                    const Set::Scalar Gs = slip7 ? ((gp7o(i, j, k) > 0.0) ? std::max(shl_o(i, j, k), 0.0) / gp7o(i, j, k) : 1.0)
                                                 : std::max(shl_o(i, j, k), 0.0);
                    const Set::Scalar kT = vbx_om.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))) ? kt_o(i, j, k) : 0.0;
                    const Set::Scalar sE = SigmaEffFromGamma(marm ? Gs : 0.0, 1.0, marm, sig0, chi_o, Gb_o, sbrk_o, sigw_o);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) om(i, j, k, nO + d) = -sE * kT * ge(d);
                    if (ffo && vbx_om.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)))) jst(i, j, k) = -sE * kT;
                }
                if (gem < g_gem_min) return; // Omega = 0 off the interface
                // The tensor carries only the tangential (Marangoni) part of the static
                // tension, which is nonzero only for a varying sigma(Gamma).
                Set::Scalar se = marm ? sg(i, j, k) : 0.0;
                // Shear part of the Boussinesq--Scriven stress.  Unlike the
                // dilatational part it is NOT proportional to P, so it cannot be
                // folded into the scalar tension and is accumulated separately.
                Set::Matrix Ds = Set::Matrix::Zero();
                if (visc_shell
                    && vel.contains(i - 1, j, k) && vel.contains(i + 1, j, k)
                    && vel.contains(i, j - 1, k) && vel.contains(i, j + 1, k)
#if AMREX_SPACEDIM > 2
                    && vel.contains(i, j, k - 1) && vel.contains(i, j, k + 1)
#endif
                   )
                {
                    const Set::Vector nh = ge / gem;
                    Set::Matrix gu = Set::Matrix::Zero();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        gu.col(d) = Numeric::Gradient(vel, i, j, k, d, DX);
                    // div_s u = tr(P.grad u) = tr(grad u) - n.grad u.n
                    // When visc_subcycle is on, SubcycleShellViscous
                    // owns this term and Omega carries the ELASTIC branch only,
                    // so it must not be added twice.
                    const Set::Scalar divs_u = gu.trace() - nh.dot(gu * nh);
                    if (!subcyc_visc) se += kap_s * divs_u;
                    if (vbx_om.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))))
                    {
                        // total sigma
                        kapd(i, j, k, 1) = se;
                    }
                    if (mu_s != 0.0)
                    {
                        // D_s = P sym(grad u) P.  P is symmetric and idempotent,
                        // so the double projection of the symmetric part is the
                        // full surface rate-of-deformation tensor.
                        const Set::Matrix P =
                            Set::Matrix::Identity() - nh * nh.transpose();
                        Ds = P * (0.5 * (gu + gu.transpose())) * P;
                    }
                }
                // Omega = ||grad eta|| * T_s,  T_s = se * P + 2 mu_s D_s.
                // ||grad eta|| * P_ab = (gem d_ab - ge_a ge_b / gem).
                const Set::Scalar tw = 2.0 * mu_s * gem;
                om(i, j, k, 0) = se * (gem - ge(0) * ge(0) / gem) + tw * Ds(0, 0); // xx
                om(i, j, k, 1) = se * (gem - ge(1) * ge(1) / gem) + tw * Ds(1, 1); // yy
#if AMREX_SPACEDIM == 2
                om(i, j, k, 2) = se * (-ge(0) * ge(1) / gem) + tw * Ds(0, 1); // xy
#else
                om(i, j, k, 2) = se * (gem - ge(2) * ge(2) / gem) + tw * Ds(2, 2); // zz
                om(i, j, k, 3) = se * (-ge(0) * ge(1) / gem) + tw * Ds(0, 1); // xy
                om(i, j, k, 4) = se * (-ge(0) * ge(2) / gem) + tw * Ds(0, 2); // xz
                om(i, j, k, 5) = se * (-ge(1) * ge(2) / gem) + tw * Ds(1, 2); // yz
#endif
            });
        }
        if (ffo)
        {
            // TEMP jump.face_force: F_d = [J_i (e+ - e-) + J+ (e+ - e_i) + J- (e_i - e-)] / (4 dx_d),
            // i.e. the average of the two face forces J_f (e_R - e_L)/dx, J_f = (J_L + J_R)/2,
            // on the same faces as the pressure flux (balanced-force).  Same net impulse as the
            // centred form; only the distribution over the band changes.
            Jst.FillBoundary(geom[lev].periodicity());
            FaceForceDomainGhosts(Jst, geom[lev]);
            for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box bx = mfi.validbox();
                auto et = eta_mf[lev]->const_array(mfi);
                auto jf = Jst.const_array(mfi);
                auto om = Omega.array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int ip = i + (d == 0), jp = j + (d == 1), kp = k + (d == 2);
                        const int im = i - (d == 0), jm = j - (d == 1), km = k - (d == 2);
                        const Set::Scalar e0 = et(i, j, k), ep = et(ip, jp, kp), em = et(im, jm, km);
                        om(i, j, k, nO + d) = (jf(i, j, k) * (ep - em) + jf(ip, jp, kp) * (ep - e0)
                                               + jf(im, jm, km) * (e0 - em)) / (4.0 * DX[d]);
                    }
                });
            }
        }
        Omega.FillBoundary(geom[lev].periodicity());
        // Ghost Omega at every symmetry face = mirror of the first valid cell (see Parse).
        {
            // Symmetric-tensor storage: 2D [xx,yy,xy]; 3D [xx,yy,zz,xy,xz,yz].
            // sgn[d][c] = parity of component c under reflection across a face normal to d.
#if AMREX_SPACEDIM == 2
            const Set::Scalar sgn[2][3] = {{1, 1, -1}, {1, 1, -1}};
            const int nO = 3;
#else
            const Set::Scalar sgn[3][6] = {{1, 1, 1, -1, -1, 1}, {1, 1, 1, -1, 1, -1}, {1, 1, 1, 1, -1, -1}};
            const int nO = 6;
#endif
            const amrex::Box dom = geom[lev].Domain();
            for (amrex::MFIter mfi(Omega, false); mfi.isValid(); ++mfi)
            {
                const amrex::Box vb = mfi.validbox();
                amrex::Array4<Set::Scalar> const &om = Omega.array(mfi);
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    for (int side = 0; side < 2; ++side)
                    {
                        const bool sym = side ? sym_face_hi[d] : sym_face_lo[d];
                        const int edge = side ? dom.bigEnd(d) : dom.smallEnd(d);
                        if (!sym || (side ? vb.bigEnd(d) : vb.smallEnd(d)) != edge) continue;
                        amrex::Box gb = amrex::grow(vb, 1);          // ghost slab of width 1 on this face
                        if (side) { gb.setSmall(d, edge + 1); gb.setBig(d, edge + 1); }
                        else      { gb.setSmall(d, edge - 1); gb.setBig(d, edge - 1); }
                        const int sh = side ? -1 : 1;                  // ghost (edge -+ 1) mirrors valid cell edge
                        amrex::LoopOnCpu(gb, [&](int i, int j, int k) {
                            amrex::IntVect g(AMREX_D_DECL(i, j, k)), v = g;
                            v[d] += sh;
                            if (!om.contains(v[0], v[1], AMREX_SPACEDIM > 2 ? v[2] : 0)) return;
                            for (int c = 0; c < nO; ++c)
                                om(i, j, k, c) = sgn[d][c] * om(v[0], v[1], AMREX_SPACEDIM > 2 ? v[2] : 0, c);
                        });
                    }
            }
        }
    }

    // Main time integration loop
    // [DIAG RHSDECOMP] per-cell boundary-flux + source stash
    // [OPT] These five scratch MultiFabs feed only the relax_diag RHSDECOMP
    // printout; they were allocated + zeroed on every RHS call regardless.
    // With relax_diag off they stay undefined and the kernel never touches
    // them (empty Array4 -> contains() is false).
    amrex::MultiFab rhsp_fxlo_mf, rhsp_fxhi_mf, rhsp_fylo_mf, rhsp_fyhi_mf, rhsp_src_mf;
    if (relax_diag)
    {
        for (amrex::MultiFab *m : {&rhsp_fxlo_mf, &rhsp_fxhi_mf, &rhsp_fylo_mf, &rhsp_fyhi_mf, &rhsp_src_mf})
        {
            m->define(rho_eta1_rhs_mf.boxArray(), rho_eta1_rhs_mf.DistributionMap(), 1, 0);
            m->setVal(0.0);
        }
    }
    // Per-box face caches (see FACE-FLUX CACHE below).  resize() only
    // reallocates when a box is larger than any seen before.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
    // Face/primitive caches are per-thread scratch (declared inside the
    // parallel region so each thread owns its own).
    amrex::BaseFab<Solver::Local::FluidRiemann::Flux> face_flux_fab[AMREX_SPACEDIM];
    amrex::FArrayBox shell_face_fab[AMREX_SPACEDIM];
    amrex::FArrayBox prim_cache_fab;
    for (amrex::MFIter mfi(*(velocity_mf)[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        // PRIMARY FLUIDS
        // FLUID 0
        Set::Patch<const Set::Scalar> rho0 = density0_mf.Patch(lev, mfi);

        // FLUID 1
        Set::Patch<const Set::Scalar> rho1 = density1_mf.Patch(lev, mfi);

        // CONSERVATIVE
        Set::Patch<const Set::Scalar> eta = eta_mf.Patch(lev, mfi);
        auto const rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto const rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        auto const M = momentum_mf[lev]->array(mfi);
        auto const E = energy_per_vol_mf[lev]->array(mfi);


        // OUTPUTS
        Set::Patch<Set::Scalar> rho_eta0_rhs = rho_eta0_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> rho_eta1_rhs = rho_eta1_rhs_mf.array(mfi);
        amrex::Array4<Set::Scalar> rhsp_fxlo, rhsp_fxhi, rhsp_fylo, rhsp_fyhi, rhsp_src;
        if (relax_diag)
        {
            rhsp_fxlo = rhsp_fxlo_mf.array(mfi);
            rhsp_fxhi = rhsp_fxhi_mf.array(mfi);
            rhsp_fylo = rhsp_fylo_mf.array(mfi);
            rhsp_fyhi = rhsp_fyhi_mf.array(mfi);
            rhsp_src  = rhsp_src_mf.array(mfi);
        }
        Set::Patch<Set::Scalar> M_rhs       = M_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> E_rhs       = E_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> eta_rhs     = eta_rhs_mf.array(mfi);
        Set::Patch<Set::Scalar> shell_rhs   = shell_rhs_mf.array(mfi);
        Set::Patch<const Set::Scalar> shell = shell_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> cfun_rhs    = cfun_rhs_mf.array(mfi);
        Set::Patch<const Set::Scalar> cfun  = cfun_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E0_rhs      = E0_rhs_mf.array(mfi);    // per-phase internal energy
        Set::Patch<Set::Scalar> E1_rhs      = E1_rhs_mf.array(mfi);
        // PHASE CHANGE: vapour partial density (empty Array4s when phasechange.on = 0)
        const bool pc_on = phasechange.on != 0;
        const int  pc_gas = 1 - phasechange.liquid;
        amrex::Array4<const Set::Scalar> vap_arr;
        amrex::Array4<Set::Scalar> vap_rhs;
        if (pc_on) { vap_arr = phasechange.vapor_mf[lev]->const_array(mfi); vap_rhs = vapor_rhs_mf->array(mfi); }
        // SPECIES partial densities (empty Array4s when species.on = 0)
        const bool sp_on = species.on != 0;
        const int  sp_n0 = species.n0, sp_n1 = species.n1, sp_order = species.order;
        const Set::Scalar sp_tb = species.thinc_beta, sp_te = species.thinc_eps;
        amrex::Array4<const Set::Scalar> sp_arr;
        amrex::Array4<Set::Scalar> sp_rhs;
        if (sp_on) { sp_arr = species.mf[lev]->const_array(mfi); sp_rhs = species_rhs_mf->array(mfi); }
        // species face fluxes for the coarse-fine reflux (hi face of each cell; lo face too at the box lo side)
        const bool sp_cc = sp_on && lev < (int)cc_fluxes.size() && cc_fluxes[lev].spec[0];
        amrex::Array4<Set::Scalar> ff_sp[3];
        if (sp_cc) for (int d = 0; d < AMREX_SPACEDIM; ++d) ff_sp[d] = cc_fluxes[lev].spec[d]->array(mfi);
        const amrex::Array4<Set::Scalar> ff_sp0 = ff_sp[0], ff_sp1 = ff_sp[1], ff_sp2 = ff_sp[2];
        const int vspg = sp_on ? vap_sp_gas : -1, vspl = sp_on ? vap_sp_liq : -1, vYloc = vap_Yinf_local;
        // ELASTIC SOLID PHASE (empty Array4s when elastic.on = 0)
        const bool elastic_on = elastic.on != 0;
        amrex::Array4<const Set::Scalar> eb_arr;
        amrex::Array4<Set::Scalar> eb_rhs, el_diag;
        if (elastic_on)
        {
            eb_arr  = elastic.ebasis_mf[lev]->const_array(mfi);
            eb_rhs  = ebasis_rhs_mf->array(mfi);
            el_diag = elastic.stress_mf[lev]->array(mfi);
        }

        // PER-PHASE INTERNAL ENERGIES (canonical primaries)
        Set::Patch<const Set::Scalar> E0_arr = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1_arr = energy1_mf.Patch(lev, mfi);

        // SOURCES
        Set::Patch<Set::Scalar> omega = vorticity_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho_flux = rho_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> M_flux = M_flux_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> E_flux = E_flux_mf.Patch(lev, mfi);

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
        Set::Patch<Set::Scalar> Vap_film = (vap_model >= 1) ? Vap_film_mf.Patch(lev, mfi) : Vap_dot;  // fallback never written
        const Set::Scalar vg0 = eos0.Gamma(), vpi0 = eos0.P0(), vcv0 = eos0.Cv();
        const Set::Scalar vpi1 = eos1.P0(), vcv1 = eos1.Cv();

        Set::Patch<const Set::Scalar> m0 = m0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> q0 = q_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> _u0 = u0_mf.Patch(lev, mfi);

        // DEBUGGING PLOTS
        Set::Patch<const Set::Scalar> kappas = kappas_mf.Patch(lev, mfi);
        amrex::Array4<const Set::Scalar> om_cap; // capillary stress tensor (if surface tension on)
        if (apply_surface_tension) om_cap = Omega.const_array(mfi);
        const int nO_f = (AMREX_SPACEDIM == 2) ? 3 : 6;
        Set::Patch<Set::Scalar> div_tau_ = div_tau_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> hess_u_ = hess_u_mf.Patch(lev, mfi);

        // ---------------- Cell-centered hi-face flux storage for reflux ---
        // cc_fluxes use density_mf's BA/dmap (same as this MFIter source),
        // so mfi access is compatible.  Each cell stores the Riemann flux
        // at its HI face (i+1/2 for d=0, j+1/2 for d=1); at box-lo
        // boundaries we also write into ghost cell (i-1) / (j-1) so
        // coarse-fine boundary fluxes survive FillBoundary (Matt Q
        // commit 8a1633977 "Fix missing lo-face fluxes...").
        bool have_cc_fluxes = (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        amrex::Array4<Set::Scalar> ff_mass_x, ff_mass_y;
        amrex::Array4<Set::Scalar> ff_mom_x,  ff_mom_y;
        amrex::Array4<Set::Scalar> ff_ene_x,  ff_ene_y;
        amrex::Array4<Set::Scalar> ff_ene_k_x, ff_ene_k_y;
#if AMREX_SPACEDIM == 3
        amrex::Array4<Set::Scalar> ff_mass_z, ff_mom_z, ff_ene_z, ff_ene_k_z;
#endif
        if (have_cc_fluxes)
        {
            ff_mass_x  = cc_fluxes[lev].mass  [0]->array(mfi);
            ff_mass_y  = cc_fluxes[lev].mass  [1]->array(mfi);
            ff_mom_x   = cc_fluxes[lev].mom   [0]->array(mfi);
            ff_mom_y   = cc_fluxes[lev].mom   [1]->array(mfi);
            ff_ene_x   = cc_fluxes[lev].energy[0]->array(mfi);
            ff_ene_y   = cc_fluxes[lev].energy[1]->array(mfi);
            ff_ene_k_x = cc_fluxes[lev].ene_k [0]->array(mfi);
            ff_ene_k_y = cc_fluxes[lev].ene_k [1]->array(mfi);
#if AMREX_SPACEDIM == 3
            ff_mass_z  = cc_fluxes[lev].mass  [2]->array(mfi);
            ff_mom_z   = cc_fluxes[lev].mom   [2]->array(mfi);
            ff_ene_z   = cc_fluxes[lev].energy[2]->array(mfi);
            ff_ene_k_z = cc_fluxes[lev].ene_k [2]->array(mfi);
#endif
        }

        // EMBEDDED SOLID BOUNDARY patches (empty Array4 when the feature is
        // off; never dereferenced unless apply_embedded_solid is set).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);          // solid indicator (1 fluid, 0 solid)
        Set::Patch<const Set::Scalar> s_re0  = embedded.density0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_re1  = embedded.density1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_M    = embedded.momentum_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_E0   = embedded.energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> s_E1   = embedded.energy1_mf.Patch(lev, mfi);

        // Gate the SHELL and COLOUR-FUNCTION rows on the physics that reads
        // them.  With marmottant=0 the shell is a pure passenger, but its bulk
        // evolution D(Gamma)/Dt = -Gamma div_s(u) integrates div(u)
        // exponentially inside a compressing/expanding bubble (measured on the
        // 3D driven bubble: shell_min 1.0 -> -4e-4 by t=3.5e-3, NaN by ~5e-3,
        // at which point WritePlotFile's NaN check ABORTS an otherwise healthy
        // run -- the cluster LowAmp_NSCBC death).  Gating also removes ~12-18
        // wasted limiter reconstructions per cell per stage.
        const int shell_row_on = marmottant || apply_surface_tension;
        const int face_diag = (thermal_diag >= 2);   // thermal.diag = 2: print faces whose gas is put at its floor
        const Set::Scalar two_p_flag = face_two_pressure ? 1.0 : 0.0;
        const Set::Scalar fd_time = time;
        const Set::Scalar eps_rl = epsilon, cut_rl = std::max(cutoff, 1.0e-12);
        const int cfun_row_on  = elastic.comp_on ? 1 : 0;   // retired with the capillary closure; carries alpha_ref of the compaction model (Hydro2_Solid.H)

        // Device-safe copies of the symmetry-face flags: the [=] kernel lambda
        // captures member arrays through `this`, which is a host pointer on a
        // GPU build.
        const bool symlo0 = sym_face_lo[0], symhi0 = sym_face_hi[0];
        const bool symlo1 = sym_face_lo[1], symhi1 = sym_face_hi[1];
#if AMREX_SPACEDIM == 3
        const bool symlo2 = sym_face_lo[2], symhi2 = sym_face_hi[2];
#endif

        // ============================================================
        // [OPT 2026-09-24] FACE-FLUX CACHE.  Every face used to be
        // reconstructed + Riemann-solved TWICE: once as the hi face of the
        // cell below it and once as the lo face of the cell above it.  Each
        // face is now solved exactly once, into a per-box face array, and the
        // cell loop below reads flux_lo = F(i), flux_hi = F(i+1).  The face
        // kernel is the same code the cell loop ran (compute_face moved out
        // verbatim; the per-cell State->Primitive conversion that make_state +
        // ToPrimitive did is now the PRIMITIVE CACHE below), so results are
        // bit-identical.
        // ============================================================
        using FluxT = Solver::Local::FluidRiemann::Flux;
        // [OPT] PRIMITIVE CACHE.  ToPrimitive(make_state(cell, dir)) -- the old
        // per-cell lambda that packed (eta, rho_eta0/1, M permuted to
        // normal/tangent, E0, E1, E, eos constants) into a State -- was
        // evaluated 6x per cell per direction (each cell sits in 6 face
        // stencils).  Its only direction dependence is which momentum
        // component is called normal/tangent, so the direction-invariant
        // parts are computed once per cell here and permuted on lookup --
        // the same arithmetic as Limiter::ToPrimitive, bit for bit.  Cells
        // outside the cached box (only possible when nghost < 3) are clamped
        // (see get_prim).
        //   comps: 0 alpha, 1 rho0_pure, 2 rho1_pure, 3 p_mix, 4.. u_d (d<SD),
        //          [2D only] 4+SD = 0/max(rho,DIV_FLOOR)  (ToPrimitive's p.w)
        constexpr int NPC = 4 + AMREX_SPACEDIM + ((AMREX_SPACEDIM == 2) ? 1 : 0);
        const int ng_pc = std::min({3, eta_mf[lev]->nGrow(), rho_eta0_mf[lev]->nGrow(), rho_eta1_mf[lev]->nGrow(),
                                    momentum_mf[lev]->nGrow(), energy0_mf[lev]->nGrow(), energy1_mf[lev]->nGrow()});
        const amrex::Box pbx = amrex::grow(bx, ng_pc);
        // + gamma / pi of phase 0 / 1 (species.eos = mixture: per cell), + dp = p_1 - p_0 (face.two_pressure)
        prim_cache_fab.resize(pbx, NPC + 5);
        amrex::Array4<Set::Scalar> const &PCw = prim_cache_fab.array();
        {
            const Set::Scalar g0c_ = eos0.Gamma(), p0c_ = eos0.P0();
            const Set::Scalar g1c_ = eos1.Gamma(), p1c_ = eos1.P0();
            const Species::Eos spe = species.eos();
            const bool el_comp = elastic.on && elastic.comp_on;
            amrex::ParallelFor(pbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                Set::Scalar g0c = g0c_, g1c = g1c_, p0c = p0c_, p1c = p1c_;
                if (spe.on) { Set::Scalar cvd = 1.0; const Set::Scalar af_ = std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12); spe.Mix(sp_arr, i, j, k, 0, g0c, cvd, E0_arr(i, j, k) / std::max(rho_eta0(i, j, k), 1.0e-300), &p0c, rho_eta0(i, j, k) / af_); spe.Mix(sp_arr, i, j, k, 1, g1c, cvd, E1_arr(i, j, k) / std::max(rho_eta1(i, j, k), 1.0e-300), &p1c, rho_eta1(i, j, k) / (1.0 - af_)); }
                if (el_comp) p0c += elastic.compaction(elastic.matAt(sp_arr, i, j, k), std::min(std::max(eta(i, j, k), 0.0), 1.0), rho_eta0(i, j, k), cfun(i, j, k)).Pc / g0c;   // effective solid pressure (Hydro2_Solid.H)
                PCw(i, j, k, NPC) = g0c; PCw(i, j, k, NPC + 1) = g1c; PCw(i, j, k, NPC + 2) = p0c; PCw(i, j, k, NPC + 3) = p1c;
                // Mirrors Limiter::ToPrimitive exactly (keep in sync).
                constexpr Set::Scalar alpha_floor = 1.0e-12;
                constexpr Set::Scalar DIV_FLOOR   = 1.0e-30;
                constexpr Set::Scalar RHO_PURE_CAP = 1.0e4;
                const Set::Scalar s_alpha = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar s_ar0 = rho_eta0(i, j, k), s_ar1 = rho_eta1(i, j, k);
                const Set::Scalar pa = std::min(std::max(s_alpha, alpha_floor), 1.0 - alpha_floor);
                const Set::Scalar a2 = 1.0 - pa;
                PCw(i, j, k, 0) = pa;
                PCw(i, j, k, 1) = std::min(s_ar0 / pa, RHO_PURE_CAP);
                PCw(i, j, k, 2) = std::min(s_ar1 / a2, RHO_PURE_CAP);
                const Set::Scalar rho_ = s_ar0 + s_ar1;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    PCw(i, j, k, 4 + d) = M(i, j, k, d) / std::max(rho_, DIV_FLOOR);
#if AMREX_SPACEDIM == 2
                PCw(i, j, k, 4 + AMREX_SPACEDIM) = 0.0 / std::max(rho_, DIV_FLOOR);
#endif
                Set::Scalar pp0 = (g0c - 1.0) * E0_arr(i, j, k) / std::max(pa, DIV_FLOOR) - g0c * p0c;
                Set::Scalar pp1 = (g1c - 1.0) * E1_arr(i, j, k) / std::max(a2, DIV_FLOOR) - g1c * p1c;
                pp0 = std::max(pp0, -p0c + DIV_FLOOR);
                pp1 = std::max(pp1, -p1c + DIV_FLOOR);
                PCw(i, j, k, 3) = pa * pp0 + a2 * pp1;
                PCw(i, j, k, NPC + 4) = pp1 - pp0;          // Limiter::ToPrimitive's p.dp
            });
        }
        amrex::Array4<const Set::Scalar> const PC = prim_cache_fab.const_array();
        auto get_prim = [=](int ii, int jj, int kk, int dir) -> Solver::Local::Limiter::Primitive
        {
            // Outside the cache means outside the allocated ghost region
            // (only when nghost < 3).  The old code read that memory out of
            // bounds; clamp to the nearest cached cell instead.  Only WENO5
            // ever uses these points (see the Parse warning), so every other
            // limiter is bit-identical.
            if (!PC.contains(ii, jj, kk))
            {
                const auto plo = amrex::lbound(pbx), phi = amrex::ubound(pbx);
                ii = amrex::min(amrex::max(ii, plo.x), phi.x);
                jj = amrex::min(amrex::max(jj, plo.y), phi.y);
                kk = amrex::min(amrex::max(kk, plo.z), phi.z);
            }
            Solver::Local::Limiter::Primitive q;
            q.alpha       = PC(ii, jj, kk, 0);
            q.alpha_rho_0 = PC(ii, jj, kk, 1);
            q.alpha_rho_1 = PC(ii, jj, kk, 2);
            q.u           = PC(ii, jj, kk, 4 + dir);
            q.v           = PC(ii, jj, kk, 4 + (dir + 1) % AMREX_SPACEDIM);
#if AMREX_SPACEDIM == 3
            q.w           = PC(ii, jj, kk, 4 + (dir + 2) % AMREX_SPACEDIM);
#else
            q.w           = PC(ii, jj, kk, 4 + AMREX_SPACEDIM);
#endif
            q.p_mix  = PC(ii, jj, kk, 3);
            q.gamma0 = PC(ii, jj, kk, NPC);      q.pi0 = PC(ii, jj, kk, NPC + 2);
            q.gamma1 = PC(ii, jj, kk, NPC + 1);  q.pi1 = PC(ii, jj, kk, NPC + 3);
            q.dp     = PC(ii, jj, kk, NPC + 4);
            q.two_p  = two_p_flag;                     // face.two_pressure (metadata, see Limiter.H)
            return q;
        };

        // ------------------------------------------------------------
        // Reconstruct face states using the selected limiter (Sch20
        // Sec 3.2: PRIMITIVE-variable reconstruction; never conservatives).
        // For face between cell `lo` and cell `hi` in normal direction
        // `dir`, gather 6-cell window {lo-2, lo-1, lo, hi, hi+1, hi+2},
        // convert to primitives, and reconstruct Q_L (right edge of
        // `lo`) and Q_R (left edge of `hi`, via reversed-stencil trick).
        // Default Limiter=Godunov returns the cell-center value
        // unchanged -- equivalent to the original first-order flux.
        // ------------------------------------------------------------
        const bool sharp_wall = embedded.apply && embedded.wall_flux;
        const bool visc_mirror = embedded.visc_mirror != 0;
        const bool noslip_wall = sharp_wall && !embedded.slip;
        // [2026-09-29] SURFACE-NORMAL WALL (solid.wall_normal = 1, sharp wall only).
        // The staircase faces are x/y(/z) faces, but the body surface is not: the
        // old ghost-mirror reflected the velocity about the FACE normal, so every
        // step riser of an inclined wall stagnated the along-surface flow (Euler
        // NACA 0008 a4: spurious separation, Cl 0.34 vs potential 0.477, Cd 0.025).
        // Here the wall normal is the TRUE surface normal n = grad(phi)/|grad(phi)|
        // (pointing into the fluid) from a Gaussian-smoothed gradient of phi, radius
        // WN_R cells, sigma WN_R/2 -- wide enough to average over the steps.
        const bool wall_normal = sharp_wall && embedded.wall_normal;
        const int wall_thin_cells = embedded.wall_thin_cells;
        const Set::Scalar wall_ratio_max = embedded.wall_image ? 4.0 : 1.0;   // 1 = plain mirror (r = 1 cap -> exactly the old reflection when phi is 0/1)
        // [2026-09-30] no-slip walls: same true-surface ghost (solid.wall_image_noslip)
        const bool noslip_image = noslip_wall && embedded.wall_image_noslip;
        const Set::Scalar noslip_ratio_max = 4.0;
        // raw smoothed gradient of phi at cell (i,j,k) (indices clamped to the fab)
        // Unit wall normal at the fluid|solid face between fluid cell f and solid
        // cell s, pointing into the fluid: Gaussian-smoothed gradient of phi (radius
        // WN_R, sigma 1.5 cells) summed over kernels centred on f and on s.  The
        // kernel is ONE-SIDED: a fluid cell counts as fluid only if it is connected
        // to f through fluid inside the local window (flood fill), so across a thin
        // section (trailing edge, wedge apex) the far surface does not cancel the
        // normal (v3: n -> +x on both faces of the NACA 0008 TE, the downwash
        // leaked into the solid), while cells along the SAME surface stay connected
        // (a line-of-sight test fails there: it grazes the step corners, v6).
        // Falls back to the face normal where the gradient is degenerate or points
        // back into the solid through this face.
        auto face_wall_normal = [=](int fi, int fj, int fk, int si, int sj, int sk, int dir, Set::Scalar n[3])
        {
            constexpr int WN_R = 3, W = WN_R + 1, N = 2 * W + 1;
#if AMREX_SPACEDIM == 3
            constexpr int WZ = W, NZ = N;
#else
            constexpr int WZ = 0, NZ = 1;
#endif
            constexpr Set::Scalar inv2s2 = 1.0 / (2.0 * 1.5 * 1.5);
            const auto blo = amrex::lbound(phisol); const auto bhi = amrex::ubound(phisol);   // inclusive
            auto ph_at = [&](int a, int b, int c) -> Set::Scalar {
                const int ii = std::min(std::max(fi + a, blo.x), bhi.x);
                const int jj = std::min(std::max(fj + b, blo.y), bhi.y);
                const int kk = std::min(std::max(fk + c, blo.z), bhi.z);
                return phisol(ii, jj, kk);
            };
            auto id = [&](int a, int b, int c) { return ((c + WZ) * N + (b + W)) * N + (a + W); };
            unsigned char mark[N * N * NZ];
            for (int q = 0; q < N * N * NZ; ++q) mark[q] = 0;
            mark[id(0, 0, 0)] = 1;
            for (int pass = 0; pass < 2 * N; ++pass)
            {
                bool changed = false;
                for (int c = -WZ; c <= WZ; ++c)
                for (int b = -W; b <= W; ++b)
                for (int a = -W; a <= W; ++a)
                {
                    if (mark[id(a, b, c)] || ph_at(a, b, c) < 0.5) continue;
                    const bool nb = (a > -W && mark[id(a - 1, b, c)]) || (a < W && mark[id(a + 1, b, c)])
                                 || (b > -W && mark[id(a, b - 1, c)]) || (b < W && mark[id(a, b + 1, c)])
                                 || (c > -WZ && mark[id(a, b, c - 1)]) || (c < WZ && mark[id(a, b, c + 1)]);
                    if (nb) { mark[id(a, b, c)] = 1; changed = true; }
                }
                if (!changed) break;
            }
            Set::Scalar g[3] = {0.0, 0.0, 0.0};
            const int ctr[2][3] = {{0, 0, 0}, {si - fi, sj - fj, sk - fk}};
            for (int m = 0; m < 2; ++m)
                for (int c = -(WZ ? WN_R : 0); c <= (WZ ? WN_R : 0); ++c)
                for (int b = -WN_R; b <= WN_R; ++b)
                for (int a = -WN_R; a <= WN_R; ++a)
                {
                    const int oa = ctr[m][0] + a, ob = ctr[m][1] + b, oc = ctr[m][2] + c;
                    Set::Scalar ph = ph_at(oa, ob, oc);
                    if (ph >= 0.5 && !mark[id(oa, ob, oc)]) ph = 0.0;          // fluid on the far side
                    const Set::Scalar w = std::exp(-(Set::Scalar)(a * a + b * b + c * c) * inv2s2);
                    g[0] += w * a * ph / DX[0];
                    g[1] += w * b * ph / DX[1];
#if AMREX_SPACEDIM == 3
                    g[2] += w * c * ph / DX[2];
#endif
                }
            // [2026-09-29] CROSSING-POINT FIT (replaces the kernel direction when possible).
            // The Gaussian-weighted gradient is exact only for a LINEAR phi; across the
            // steep, saturating wall profile (tanh / volume fraction) the square lattice's
            // anisotropy tilts it: 15 deg wedge, median +0.69 deg -> the flow turned
            // 14.23 deg, beta -0.78 deg.  Instead collect the phi = 0.5 crossings on every
            // grid edge of the window whose fluid end is connected to f (same side), and
            // take the normal of the least-squares line/plane through them: wedge bias
            // +0.01 deg (p90 0.11), NACA 0008 mid-chord bias +0.01 deg.
            {
                Set::Scalar S1[3] = {0.0, 0.0, 0.0}, S2[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                int np = 0;
                for (int c = -WZ; c <= WZ; ++c)
                for (int b = -W; b <= W; ++b)
                for (int a = -W; a <= W; ++a)
                {
                    const Set::Scalar p0 = ph_at(a, b, c);
                    for (int ed = 0; ed < AMREX_SPACEDIM; ++ed)
                    {
                        const int a2 = a + (ed == 0), b2 = b + (ed == 1), c2 = c + (ed == 2);
                        if (a2 > W || b2 > W || c2 > WZ) continue;
                        const Set::Scalar p1 = ph_at(a2, b2, c2);
                        if ((p0 >= 0.5) == (p1 >= 0.5)) continue;
                        const bool f0 = p0 >= 0.5;
                        if (!mark[f0 ? id(a, b, c) : id(a2, b2, c2)]) continue;   // crossing on the far side
                        const Set::Scalar t = (p0 - 0.5) / (p0 - p1);
                        const Set::Scalar q[3] = {(a + t * (ed == 0)) * DX[0], (b + t * (ed == 1)) * DX[1],
                                                  (AMREX_SPACEDIM == 3 ? (c + t * (ed == 2)) * DX[AMREX_SPACEDIM - 1] : 0.0)};
                        for (int d1 = 0; d1 < 3; ++d1) { S1[d1] += q[d1]; for (int d2 = 0; d2 < 3; ++d2) S2[d1][d2] += q[d1] * q[d2]; }
                        ++np;
                    }
                }
                if (np >= AMREX_SPACEDIM + 1)
                {
                    Set::Scalar C[3][3];
                    for (int d1 = 0; d1 < 3; ++d1) for (int d2 = 0; d2 < 3; ++d2) C[d1][d2] = S2[d1][d2] - S1[d1] * S1[d2] / np;
                    Set::Scalar m[3];
#if AMREX_SPACEDIM == 2
                    const Set::Scalar thm = 0.5 * std::atan2(2.0 * C[0][1], C[0][0] - C[1][1]);   // line direction
                    m[0] = -std::sin(thm); m[1] = std::cos(thm); m[2] = 0.0;
#else
                    Eigen::Matrix3d Cm;
                    for (int d1 = 0; d1 < 3; ++d1) for (int d2 = 0; d2 < 3; ++d2) Cm(d1, d2) = C[d1][d2];
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Cm);
                    const Eigen::Vector3d v = es.eigenvectors().col(0);
                    m[0] = v(0); m[1] = v(1); m[2] = v(2);
#endif
                    const Set::Scalar gmag = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
                    const Set::Scalar sgn = (m[0] * g[0] + m[1] * g[1] + m[2] * g[2]) >= 0.0 ? 1.0 : -1.0;
                    if (gmag > 0.0) for (int d = 0; d < 3; ++d) g[d] = sgn * m[d] * gmag;       // keep |g| for the degeneracy test
                }
            }
            Set::Scalar e[3] = {0.0, 0.0, 0.0};
            e[dir] = (dir == 0 ? (Set::Scalar)(fi - si) : dir == 1 ? (Set::Scalar)(fj - sj) : (Set::Scalar)(fk - sk));
            // PLATE-THIN FACE: if the solid through s is <= wall_thin_cells thick ALONG
            // THIS FACE'S AXIS (fluid on both sides), s is part of a plate one or two
            // cells thick and the face normal IS the plate normal.  The kernel cannot
            // resolve it there: within ~5 cells of a sharp trailing edge it sees the
            // fluid beyond the tip (connected to both sides) and tilts 30-50 deg toward
            // the tip (NACA 0008 L4), and the ghost then leaked the downwash into the
            // solid: a trailing-edge sink that pumps up the circulation (v3: Cl 0.58 at
            // 8 c/U vs potential 0.477).  Only the face's own axis: testing every axis
            // (v10) also switched the RISER faces of the last 7 % chord back to the
            // staircase mirror and cost 27 % of the circulation.
            {
                const int TL = wall_thin_cells;
                const int ax = dir;
                // solid thickness along ax through cell c (0 if c is fluid); > TL if not bounded by fluid within TL
                auto thick = [&](int ci, int cj, int ck) -> int {
                    ci = std::min(std::max(ci, blo.x), bhi.x); cj = std::min(std::max(cj, blo.y), bhi.y); ck = std::min(std::max(ck, blo.z), bhi.z);
                    if (phisol(ci, cj, ck) >= 0.5) return 0;
                    int run = 1; bool hit_p = false, hit_m = false;
                    for (int sgn = -1; sgn <= 1; sgn += 2)
                        for (int m = 1; m <= TL; ++m)
                        {
                            const int ii = ci + (ax == 0 ? sgn * m : 0), jj = cj + (ax == 1 ? sgn * m : 0), kk = ck + (ax == 2 ? sgn * m : 0);
                            if (ii < blo.x || ii > bhi.x || jj < blo.y || jj > bhi.y || kk < blo.z || kk > bhi.z) break;
                            if (phisol(ii, jj, kk) >= 0.5) { (sgn > 0 ? hit_p : hit_m) = true; break; }
                            ++run;
                        }
                    return (hit_p && hit_m) ? run : TL + 1;
                };
                // a PLATE stays thin (or ends) 2 cells away along every tangential axis; a
                // rounded NOSE (whose tip is also 1-2 cells thick) thickens right away --
                // there the smoothed normal is accurate and the plate axis is not (it lies
                // along the chord: v12 created 2.3e-3 of mass at the NACA 0008 nose).
                bool plate = TL > 0 && thick(si, sj, sk) <= TL;
                for (int tax = 0; tax < AMREX_SPACEDIM && plate; ++tax)
                {
                    if (tax == ax) continue;
                    for (int sgn = -2; sgn <= 2 && plate; sgn += 4)
                        if (thick(si + (tax == 0 ? sgn : 0), sj + (tax == 1 ? sgn : 0), sk + (tax == 2 ? sgn : 0)) > TL) plate = false;
                }
                if (plate)
                {
                    // The plate's normal = the direction of LEAST spread of its solid
                    // cells (principal axis of the solid-fraction-weighted positions in
                    // the window around s): the midline normal, so the flow leaves a
                    // sharp trailing edge along its bisector.  The face normal instead
                    // makes the last cells horizontal -- a small flap deflected by the
                    // local midline angle (-4 deg at a4: ~ -0.1 Cl, v11 at 0.70 x Wagner).
                    Set::Scalar wsum = 0.0, c[3] = {0.0, 0.0, 0.0}, C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                    const int sa = si - fi, sb = sj - fj, sc = sk - fk;
                    for (int pass = 0; pass < 2; ++pass)
                    {
                        for (int cc = -WZ; cc <= WZ; ++cc)
                        for (int bb = -W; bb <= W; ++bb)
                        for (int aa = -W; aa <= W; ++aa)
                        {
                            const Set::Scalar ws = 1.0 - std::min(std::max(ph_at(sa + aa, sb + bb, sc + cc), 0.0), 1.0);
                            if (ws <= 0.0) continue;
                            const Set::Scalar q[3] = {(Set::Scalar)aa * DX[0], (Set::Scalar)bb * DX[1], (Set::Scalar)cc * (AMREX_SPACEDIM == 3 ? DX[AMREX_SPACEDIM - 1] : 0.0)};
                            if (pass == 0) { wsum += ws; for (int d = 0; d < 3; ++d) c[d] += ws * q[d]; }
                            else for (int d1 = 0; d1 < 3; ++d1) for (int d2 = 0; d2 < 3; ++d2)
                                C[d1][d2] += ws * (q[d1] - c[d1]) * (q[d2] - c[d2]);
                        }
                        if (pass == 0 && wsum > 0.0) for (int d = 0; d < 3; ++d) c[d] /= wsum;
                    }
                    Set::Scalar m[3] = {e[0], e[1], e[2]};
#if AMREX_SPACEDIM == 2
                    {   // smallest eigenvector of the 2x2 covariance
                        const Set::Scalar th = 0.5 * std::atan2(2.0 * C[0][1], C[0][0] - C[1][1]);   // major axis
                        m[0] = -std::sin(th); m[1] = std::cos(th); m[2] = 0.0;
                    }
#else
                    {
                        Eigen::Matrix3d Cm;
                        for (int d1 = 0; d1 < 3; ++d1) for (int d2 = 0; d2 < 3; ++d2) Cm(d1, d2) = C[d1][d2];
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Cm);
                        const Eigen::Vector3d v = es.eigenvectors().col(0);          // ascending eigenvalues
                        m[0] = v(0); m[1] = v(1); m[2] = v(2);
                    }
#endif
                    Set::Scalar mn = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
                    Set::Scalar me = m[0] * e[0] + m[1] * e[1] + m[2] * e[2];
                    if (wsum <= 0.0 || mn < 1e-12 || std::abs(me) < 1e-12) { for (int d = 0; d < 3; ++d) n[d] = e[d]; return; }
                    for (int d = 0; d < 3; ++d) n[d] = (me > 0.0 ? 1.0 : -1.0) * m[d] / mn;
                    return;
                }
            }
            const Set::Scalar gm = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
            const Set::Scalar ref = 0.1 / DX[dir];          // |grad phi| of a unit jump spread over ~10 cells
            if (gm > ref)
            {
                for (int d = 0; d < 3; ++d) n[d] = g[d] / gm;
                if (n[0] * e[0] + n[1] * e[1] + n[2] * e[2] > 0.0) return;
            }
            for (int d = 0; d < 3; ++d) n[d] = e[d];
        };
        // [2026-10-05] WALL VELOCITY AT THE TRUE SURFACE (solid.wall_vel_surface, default 1).
        // u_s is the solid target velocity M_s/rho_s, a field defined in every cell
        // (rigid motion: translation + rotation, set by solid.momentum.ic).  The wall
        // sits at phi = 0.5, a fraction t = (phi_f - 0.5)/(phi_f - phi_s) of the way
        // from the fluid cell f to the solid cell s, so the wall velocity is the
        // linear interpolation u_w = u_s(f) + t (u_s(s) - u_s(f)) -- exact for a
        // rigid body.  0 = the solid cell's own value (a spinning wall is then
        // evaluated up to a cell inside the body: O(Omega dx) error).  A static or
        // uniformly translating solid is unaffected (u_s uniform).
        const bool wall_vel_surface = embedded.wall_vel_surface != 0;
        auto wall_vel = [=](int fi, int fj, int fk, int si, int sj, int sk, Set::Scalar uw[3])
        {
            const Set::Scalar rs = std::max(s_re0(si, sj, sk) + s_re1(si, sj, sk), small);
            uw[0] = uw[1] = uw[2] = 0.0;
            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) uw[dd] = s_M(si, sj, sk, dd) / rs;
            if (wall_vel_surface)
            {
                const Set::Scalar rf = std::max(s_re0(fi, fj, fk) + s_re1(fi, fj, fk), small);
                const Set::Scalar phf = phisol(fi, fj, fk), phs = phisol(si, sj, sk);
                const Set::Scalar t = std::min(std::max((phf - 0.5) / std::max(phf - phs, 1e-12), 0.0), 1.0);
                for (int dd = 0; dd < AMREX_SPACEDIM; ++dd)
                {
                    const Set::Scalar uf = s_M(fi, fj, fk, dd) / rf;
                    uw[dd] = uf + t * (uw[dd] - uf);
                }
            }
        };
        // THINC slope weighting needs the interface normal, which the limiter cannot see (see compute_face)
        // Interface geometry for geometry-aware limiters (multi-D THINC;
        // Limiter::WantsGeom).  From central differences of the clamped
        // eta: n = grad eta/|grad eta|, K = P Hess(eta) P/|grad eta|
        // (P = I - n n^T, the shape tensor grad n), rotated into the face
        // frame [dir, dir+1, dir+2] and flipped along dir for sgn = -1.
        const bool lim_geom = limiter->WantsGeom();
        auto cell_geom = [=](int ii, int jj, int kk, int dir, int sgn) -> Solver::Local::Limiter::Geom
        {
            Solver::Local::Limiter::Geom g;
            auto et = [&](int a, int b, int c) { return std::min(1.0, std::max(0.0, eta(ii + a, jj + b, kk + c))); };
            auto off = [](int d, int s_) { amrex::IntVect v(AMREX_D_DECL(0, 0, 0)); v[d] = s_; return v; };
            Set::Vector gr; Set::Matrix He;
            const Set::Scalar e0 = et(0, 0, 0);
            for (int a = 0; a < AMREX_SPACEDIM; ++a)
            {
                amrex::IntVect pa = off(a, 1);
#if AMREX_SPACEDIM == 3
                const Set::Scalar ep = et(pa[0], pa[1], pa[2]), em = et(-pa[0], -pa[1], -pa[2]);
#else
                const Set::Scalar ep = et(pa[0], pa[1], 0), em = et(-pa[0], -pa[1], 0);
#endif
                gr(a) = (ep - em) / (2.0 * DX[a]);
                He(a, a) = (ep - 2.0 * e0 + em) / (DX[a] * DX[a]);
                for (int b = a + 1; b < AMREX_SPACEDIM; ++b)
                {
                    amrex::IntVect pb = off(b, 1);
                    auto E2 = [&](int sa, int sb) {
#if AMREX_SPACEDIM == 3
                        return et(sa * pa[0] + sb * pb[0], sa * pa[1] + sb * pb[1], sa * pa[2] + sb * pb[2]);
#else
                        return et(sa * pa[0] + sb * pb[0], sa * pa[1] + sb * pb[1], 0);
#endif
                    };
                    He(a, b) = He(b, a) = (E2(1, 1) - E2(1, -1) - E2(-1, 1) + E2(-1, -1)) / (4.0 * DX[a] * DX[b]);
                }
            }
            const Set::Scalar gn = gr.norm();
            Set::Scalar hmin = DX[0];
            for (int a = 1; a < AMREX_SPACEDIM; ++a) hmin = std::min(hmin, (Set::Scalar)DX[a]);
            if (!(gn * hmin > 1.0e-8)) return g;           // no usable normal
            const Set::Vector n = gr / gn;
            const Set::Matrix P = Set::Matrix::Identity() - n * n.transpose();
            const Set::Matrix K = P * He * P / gn;
            int ax[AMREX_SPACEDIM];
            for (int a = 0; a < AMREX_SPACEDIM; ++a) ax[a] = (dir + a) % AMREX_SPACEDIM;
            for (int a = 0; a < AMREX_SPACEDIM; ++a)
            {
                const Set::Scalar fa = (a == 0) ? sgn : 1.0;
                g.n(a) = fa * n(ax[a]);
                g.h(a) = DX[ax[a]];
                for (int b = 0; b < AMREX_SPACEDIM; ++b)
                    g.K(a, b) = fa * ((b == 0) ? sgn : 1.0) * K(ax[a], ax[b]);
            }
            g.eps = epsilon;
            g.valid = true;
            return g;
        };

        auto compute_face = [=](int lo_i, int lo_j, int lo_k, int hi_i, int hi_j, int hi_k, int dir)
            -> FluxT
        {
            const int di = hi_i - lo_i;
            const int dj = hi_j - lo_j;
            const int dk = hi_k - lo_k;
            Solver::Local::Limiter::Primitive prim[6];
            for (int s = -2; s <= 3; ++s)
                prim[s + 2] = get_prim(lo_i + s * di, lo_j + s * dj, lo_k + s * dk, dir);
            // [2026-09-24] GHOST-MIRROR RECONSTRUCTION (sharp wall, solid.wall_flux).
            // compute_face is only called for fluid|fluid faces in that mode, but
            // its 6-cell window can reach into the solid.  Using the solid cells'
            // OWN state there (u = 0, p = p_solid, frozen rho) puts an artificial
            // jump inside the limiter stencil that is inconsistent with the mirror
            // state the wall face itself uses.  Measured on the AoA = 0 NACA 0012:
            // a CFL-dependent instability (cfl 0.3: mirror asymmetry 1e-14 -> 2e-3
            // in 5 steps, growing x3-40 per fine step; cfl 0.1: 1e-6) that drove a
            // spurious, slowly growing lift.  Replace each solid cell in the window
            // by the mirror image of the fluid cell across the wall (normal velocity
            // reflected about the solid's), the standard ghost-cell treatment.
            if (sharp_wall)
            {
                bool sol[6];
                for (int s = -2; s <= 3; ++s)
                    sol[s + 2] = phisol(lo_i + s * di, lo_j + s * dj, lo_k + s * dk) < 0.5;
                auto mirror = [&](int ghost, int src) {
                    // wall = first solid cell of the window on this side (ws) | its fluid neighbour (wf)
                    const int ws_ = (ghost <= 1) ? (sol[1] ? 1 : 0) : (sol[4] ? 4 : 5);
                    const int wf_ = (ghost <= 1) ? ws_ + 1 : ws_ - 1;
                    Set::Scalar Uw[3];
                    wall_vel(lo_i + (wf_ - 2) * di, lo_j + (wf_ - 2) * dj, lo_k + (wf_ - 2) * dk,
                             lo_i + (ws_ - 2) * di, lo_j + (ws_ - 2) * dj, lo_k + (ws_ - 2) * dk, Uw);
                    const Set::Scalar us_n = Uw[dir];
                    Solver::Local::Limiter::Primitive q = prim[src];
                    q.u = 2.0 * us_n - prim[src].u;
                    if (wall_normal && !noslip_wall)
                    {
                        // [2026-09-29] slip: reflect about the TRUE surface normal,
                        // u_g = u - 2 ((u - u_s).n) n, not about the face normal.
                        // The wall is the first solid cell of the window (1 or 4).
                        // first solid cell (ws) and its fluid neighbour (wf) on this side
                        const int ws = (ghost <= 1) ? (sol[1] ? 1 : 0) : (sol[4] ? 4 : 5);
                        const int wf = (ghost <= 1) ? ws + 1 : ws - 1;
                        Set::Scalar n[3];
                        face_wall_normal(lo_i + (wf - 2) * di, lo_j + (wf - 2) * dj, lo_k + (wf - 2) * dk,
                                         lo_i + (ws - 2) * di, lo_j + (ws - 2) * dj, lo_k + (ws - 2) * dk, dir, n);
                        const int c1 = (dir + 1) % AMREX_SPACEDIM, c2 = (dir + 2) % AMREX_SPACEDIM;
                        Set::Scalar U[3] = {0.0, 0.0, 0.0}, Us[3] = {0.0, 0.0, 0.0};
                        U[dir] = prim[src].u; U[c1] = prim[src].v;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d) Us[d] = Uw[d];
#if AMREX_SPACEDIM == 3
                        U[c2] = prim[src].w;
#endif
                        Set::Scalar un = 0.0;
                        for (int d = 0; d < 3; ++d) un += (U[d] - Us[d]) * n[d];
                        q.u = U[dir] - 2.0 * un * n[dir];
                        q.v = U[c1]  - 2.0 * un * n[c1];
#if AMREX_SPACEDIM == 3
                        q.w = U[c2]  - 2.0 * un * n[c2];
#endif
                        (void)c2;
                    }
                    if (noslip_wall)   // no-slip: reflect the tangential velocity about the solid's too
                    {
                        q.v = 2.0 * Uw[(dir + 1) % AMREX_SPACEDIM] - prim[src].v;
#if AMREX_SPACEDIM == 3
                        q.w = 2.0 * Uw[(dir + 2) % AMREX_SPACEDIM] - prim[src].w;
#endif
                        if (noslip_image)
                        {
                            // [2026-09-30] NO-SLIP WALL AT THE TRUE SURFACE: velocity (relative to
                            // the solid) linear and zero at phi = 0.5, which sits a fraction t of
                            // the f -> s spacing from the wall fluid cell.  A ghost mg cells deep
                            // mirrors a source ms cells out: u_g - u_s = -(mg - t)/(t + ms - 1)
                            // (u_src - u_s).  Plain mirror (ratio 1) for a 0/1 phi (t = 1/2).
                            const int ws = (ghost <= 1) ? (sol[1] ? 1 : 0) : (sol[4] ? 4 : 5);
                            const int wf = (ghost <= 1) ? ws + 1 : ws - 1;
                            const Set::Scalar phf = phisol(lo_i + (wf - 2) * di, lo_j + (wf - 2) * dj, lo_k + (wf - 2) * dk);
                            const Set::Scalar phs = phisol(lo_i + (ws - 2) * di, lo_j + (ws - 2) * dj, lo_k + (ws - 2) * dk);
                            const Set::Scalar t = std::min(std::max((phf - 0.5) / std::max(phf - phs, 1e-12), 0.05), 0.95);
                            const int mg = std::abs(ghost - ws) + 1, ms = std::abs(src - wf) + 1;
                            const Set::Scalar ratio = std::min((mg - t) / (t + ms - 1), noslip_ratio_max);
                            const Set::Scalar usn = Uw[dir];
                            const Set::Scalar ust = Uw[(dir + 1) % AMREX_SPACEDIM];
                            q.u = usn - ratio * (prim[src].u - usn);
                            q.v = ust - ratio * (prim[src].v - ust);
#if AMREX_SPACEDIM == 3
                            const Set::Scalar ust2 = Uw[(dir + 2) % AMREX_SPACEDIM];
                            q.w = ust2 - ratio * (prim[src].w - ust2);
#endif
                        }
                    }
                    prim[ghost] = q;
                };
                // lo side: window cells 1 (lo-1) and 0 (lo-2); lo = 2 and hi = 3 are fluid
                if (sol[1])      { mirror(1, 2); mirror(0, 3); }
                else if (sol[0]) { mirror(0, 1); }
                // hi side: window cells 4 (hi+1) and 5 (hi+2)
                if (sol[4])      { mirror(4, 3); mirror(5, 2); }
                else if (sol[5]) { mirror(5, 4); }
            }
            // Q_L: right-edge of cell `lo` (stencil centered on prim[2]).
            // Q_L: right-edge of cell `lo` (stencil centered on prim[2]).
            Solver::Local::Limiter::Primitive stencil_L[5] =
                { prim[0], prim[1], prim[2], prim[3], prim[4] };
            // Q_R: left-edge of cell `hi` (stencil centered on prim[3],
            //      reversed so right-edge reconstruction returns left-edge).
            Solver::Local::Limiter::Primitive stencil_R[5] =
                { prim[5], prim[4], prim[3], prim[2], prim[1] };
            Solver::Local::Limiter::Primitive pL, pR;
            if (lim_radial_base && radial_face_hi[dir])
            {
                // RADIAL slab transverse direction: plain base-limiter face values (no THINC
                // sharpening; the two transverse cells only emulate the spherical divergence).
                pL = limiter->ReconstructPassive(stencil_L);
                pR = limiter->ReconstructPassive(stencil_R);
            }
            else if (lim_geom)
            {
                // interface geometry of each stencil's center cell, in the
                // face frame of that stencil (stencil_R is reversed: sgn -1)
                pL = limiter->ReconstructGeom(stencil_L, cell_geom(lo_i, lo_j, lo_k, dir, +1));
                pR = limiter->ReconstructGeom(stencil_R, cell_geom(hi_i, hi_j, hi_k, dir, -1));
            }
            else
            {
                pL = limiter->Reconstruct(stencil_L);
                pR = limiter->Reconstruct(stencil_R);
            }
            Solver::Local::FluidRiemann::State sL_face = Solver::Local::Limiter::ToState(pL, small);
            Solver::Local::FluidRiemann::State sR_face = Solver::Local::Limiter::ToState(pR, small);
            if (elastic_on)     // solid stress, elastic sound speed, shear-dissipation input (Hydro2_Solid.H)
            {
                elastic.faceState(sL_face, eb_arr, eta, rho_eta0, rho_eta1, lo_i, lo_j, lo_k, dir, +1, sp_arr, cfun);
                elastic.faceState(sR_face, eb_arr, eta, rho_eta0, rho_eta1, hi_i, hi_j, hi_k, dir, -1, sp_arr, cfun);
                if (elastic.objective_shear && sL_face.solid && sR_face.solid)
                {
                    auto un = [&](int ci, int cj, int ck) { return get_prim(ci, cj, ck, dir).u; };
                    const int t1 = (dir + 1) % AMREX_SPACEDIM;
                    const Set::Scalar g1 = elastic.shearCorrection(eta, un, lo_i, lo_j, lo_k, hi_i, hi_j, hi_k, t1, DX[dir], DX[t1], pR.v - pL.v);
                    sL_face.dvt = g1; sR_face.dvt = g1;
#if AMREX_SPACEDIM == 3
                    const int t2 = (dir + 2) % AMREX_SPACEDIM;
                    const Set::Scalar g2 = elastic.shearCorrection(eta, un, lo_i, lo_j, lo_k, hi_i, hi_j, hi_k, t2, DX[dir], DX[t2], pR.w - pL.w);
                    sL_face.dvt2 = g2; sR_face.dvt2 = g2;
#endif
                }
            }

            bool at_shock = false;
            if (shock_fix)      // a strong pressure jump transverse to this face: the face lies along a shock
            {
                const int cc[2][3] = {{lo_i, lo_j, lo_k}, {hi_i, hi_j, hi_k}};
                for (int c = 0; c < 2 && !at_shock; ++c)
                {
                    const Set::Scalar ac = PC(cc[c][0], cc[c][1], cc[c][2], 0);
                    const Set::Scalar pc = PC(cc[c][0], cc[c][1], cc[c][2], 3) + ac * PC(cc[c][0], cc[c][1], cc[c][2], NPC + 2) + (1.0 - ac) * PC(cc[c][0], cc[c][1], cc[c][2], NPC + 3);
                    for (int t = 1; t < AMREX_SPACEDIM && !at_shock; ++t)
                    {
                        const int td = (dir + t) % AMREX_SPACEDIM;
                        for (int sg = -1; sg <= 1 && !at_shock; sg += 2)
                        {
                            const int ni = cc[c][0] + (td == 0 ? sg : 0), nj = cc[c][1] + (td == 1 ? sg : 0), nk = cc[c][2] + (td == 2 ? sg : 0);
                            if (!PC.contains(ni, nj, nk)) continue;
                            if (sharp_wall && phisol(ni, nj, nk) < 0.5) continue;      // not across an embedded wall
                            const Set::Scalar an = PC(ni, nj, nk, 0);
                            const Set::Scalar pn = PC(ni, nj, nk, 3) + an * PC(ni, nj, nk, NPC + 2) + (1.0 - an) * PC(ni, nj, nk, NPC + 3);
                            at_shock = std::abs(pn - pc) > shock_fix_ratio * std::max(std::min(pn, pc), small);
                        }
                    }
                }
            }
            FluxT fl_ = at_shock ? shockfix_solver->Solve(sL_face, sR_face, pref, small)
                                 : riemannsolver->Solve(sL_face, sR_face, pref, small);
            if (face_diag && dir == 0 && lo_j == 0 && lo_k == 0
                && ((pL.p_mix + pL.pi1 <= 0.0 && pL.p_mix + pL.pi0 > 0.0) || (pR.p_mix + pR.pi1 <= 0.0 && pR.p_mix + pR.pi0 > 0.0)))
                printf("FACEFLOOR t=%.7e face %d|%d aL=%.6f aR=%.6f pmixL=%.4e pmixR=%.4e gas_mass_flux=%.4e gas_energy_flux=%.4e e1_carried=%.4e\n",
                       fd_time, lo_i, hi_i, pL.alpha, pR.alpha, pL.p_mix, pR.p_mix, fl_.mass1, fl_.energy1,
                       (std::abs(fl_.mass1) > 0.0) ? fl_.energy1 / fl_.mass1 : 0.0);
            // Carry the RECONSTRUCTED alpha out, upwinded on the contact
            // speed, so the alpha row can be advected at the same order as
            // the mass rows (see the alpha-row note in Advance).
            fl_.alpha_face = (fl_.u_interface > 0.0) ? pL.alpha : pR.alpha;
            return fl_;
        };

        // ------------------------------------------------------
        // SYMMETRY-FACE FLUX ENFORCEMENT.  At a REFLECT domain
        // face the exact Riemann solution has u* = 0: the only
        // nonzero flux component is the normal-momentum (pressure)
        // term.  Computing it numerically instead leaves a
        // roundoff residual of order eps*|S_L|*rho_eta (the stiff
        // sound speed amplifies machine eps ~1e-16 to ~1e-13),
        // applied with the SAME SIGN at the SAME cells every step
        // -- a coherent mass/energy pump through the symmetry
        // plane that feeds back exponentially in driven octant/
        // quadrant runs (measured: 100%% of the gas-mass
        // instability enters via the reflect faces; RHSDECOMP
        // budget closes to 1e-25 with them zeroed analytically).
        // ------------------------------------------------------
        auto symmetrize = [](FluxT &f) {
            f.mass0 = 0.0;
            f.mass1 = 0.0;
            f.momentum_tangent  = 0.0;
            f.momentum_tangent2 = 0.0;
            f.energy_total = 0.0;
            f.energy0 = 0.0;
            f.energy1 = 0.0;
            // u* = 0 is exact at the symmetry plane, and u_interface is
            // the ONLY quantity the eta/E0/E1 non-conservative rows read
            // from this flux.  Leaving its roundoff residual while
            // zeroing the mass rows makes the alpha row discretely
            // inconsistent with the phase-mass rows at the same face --
            // a coherent same-sign alpha/mass mismatch at the collapse
            // focus (octant corner) every stage.
            f.u_interface = 0.0;
        };

        // Face node f in direction d separates cells f-1 (lo) and f (hi).
        // The domain lo face is node smallEnd(d); the domain hi face is node
        // bigEnd(d)+1 -- the same faces the per-cell code symmetrized as
        // flux_*lo of cell smallEnd and flux_*hi of cell bigEnd.
        BL_PROFILE_VAR("Hydro2::RHS::face_fluxes", prof_faces);
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
        {
            const amrex::Box fbx = amrex::surroundingNodes(bx, d);
            face_flux_fab[d].resize(fbx, 1);
            amrex::Array4<FluxT> const &Ff = face_flux_fab[d].array();
            const int di = (d == 0), dj = (d == 1), dk = (d == 2);
            const int dom_lo = domain.smallEnd(d), dom_hi_face = domain.bigEnd(d) + 1;
            bool sl = false, sh = false;
            if (d == 0) { sl = symlo0; sh = symhi0; }
            if (d == 1) { sl = symlo1; sh = symhi1; }
#if AMREX_SPACEDIM == 3
            if (d == 2) { sl = symlo2; sh = symhi2; }
#endif
            // [2026-09-24] sharp-wall flux (solid.wall_flux, see Parse).
            const bool wallflux = embedded.apply && embedded.wall_flux;
            amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                const int f = (d == 0) ? i : ((d == 1) ? j : k);
                try
                {
                    FluxT fl;
                    const bool fl_lo = !wallflux || phisol(i - di, j - dj, k - dk) >= 0.5;
                    const bool fl_hi = !wallflux || phisol(i, j, k) >= 0.5;
                    if (fl_lo && fl_hi)
                    {
                        fl = compute_face(i - di, j - dj, k - dk, i, j, k, d);
                    }
                    else if (!fl_lo && !fl_hi)
                    {
                        fl = FluxT{};                                  // inside the solid: no flux
                        fl.alpha_face = get_prim(i, j, k, d).alpha;
                    }
                    else
                    {
                        // Wall face: first-order Riemann problem between the fluid
                        // cell and its mirror.  Normal velocity reflected about the
                        // solid's normal velocity (u_g = 2 u_s - u_f), tangential kept
                        // (the no-slip part is the Brinkman term + viscous stress).
                        const int fi = fl_lo ? i - di : i, fj = fl_lo ? j - dj : j, fk = fl_lo ? k - dk : k;
                        const int si = fl_lo ? i : i - di, sj = fl_lo ? j : j - dj, sk = fl_lo ? k : k - dk;
                        const Solver::Local::Limiter::Primitive pf = get_prim(fi, fj, fk, d);
                        Set::Scalar Uw[3];                       // wall velocity at the true surface (see wall_vel)
                        wall_vel(fi, fj, fk, si, sj, sk, Uw);
                        const Set::Scalar us_n = Uw[d];
                        Solver::Local::Limiter::Primitive pg = pf;
                        pg.u = 2.0 * us_n - pf.u;
                        if (noslip_wall)
                        {
                            pg.v = 2.0 * Uw[(d + 1) % AMREX_SPACEDIM] - pf.v;
#if AMREX_SPACEDIM == 3
                            pg.w = 2.0 * Uw[(d + 2) % AMREX_SPACEDIM] - pf.w;
#endif
                            if (noslip_image)
                            {
                                // [2026-09-30] no-slip wall at phi = 0.5 (see compute_face): ghost
                                // velocity relative to the solid = -r (u_f - u_s),
                                // r = (0.5 - phi_s)/(phi_f - 0.5); r = 1 for a 0/1 phi.
                                const Set::Scalar phf = phisol(fi, fj, fk), phs = phisol(si, sj, sk);
                                Set::Scalar r = (0.5 - phs) / std::max(phf - 0.5, 1e-3);
                                r = std::min(std::max(r, 0.0), noslip_ratio_max);
                                const Set::Scalar ust = Uw[(d + 1) % AMREX_SPACEDIM];
                                pg.u = us_n - r * (pf.u - us_n);
                                pg.v = ust  - r * (pf.v - ust);
#if AMREX_SPACEDIM == 3
                                const Set::Scalar ust2 = Uw[(d + 2) % AMREX_SPACEDIM];
                                pg.w = ust2 - r * (pf.w - ust2);
#endif
                            }
                        }
                        if (wall_normal && !noslip_wall)
                        {
                            // [2026-09-29] GHOST-CELL MIRROR ABOUT THE TRUE NORMAL (slip).
                            // The ghost is the fluid state reflected about the surface
                            // normal n (relative to the solid): u_g = u - 2 ((u - u_s).n) n,
                            // and the face flux is the ordinary Riemann flux fluid|ghost.
                            // Flow tangent to the TRUE surface therefore passes a riser
                            // face of the staircase as if the wall were smooth (ghost-cell
                            // immersed boundary).  A pressure-only flux with p* along n was
                            // tried first and is WRONG at low Mach: the staircase still
                            // forces the cell velocity parallel to the steps, so u.n ~ U
                            // sin(theta) and p* picks up rho c U sin(theta) (~0.7 q at
                            // M 0.2, alpha 4): Euler Cl 1.05, Cd -0.05.  No-slip needs no
                            // normal: its mirror 2 u_s - u is frame independent.
                            Set::Scalar n[3];
                            face_wall_normal(fi, fj, fk, si, sj, sk, d, n);
                            const int c1 = (d + 1) % AMREX_SPACEDIM, c2 = (d + 2) % AMREX_SPACEDIM;
                            Set::Scalar U[3] = {0.0, 0.0, 0.0}, Us[3] = {0.0, 0.0, 0.0};
                            U[d] = pf.u; U[c1] = pf.v;
#if AMREX_SPACEDIM == 3
                            U[c2] = pf.w;
#endif
                            (void)c2;
                            for (int dd = 0; dd < AMREX_SPACEDIM; ++dd) Us[dd] = Uw[dd];
                            Set::Scalar un = 0.0;
                            for (int dd = 0; dd < 3; ++dd) un += (U[dd] - Us[dd]) * n[dd];
                            // [2026-09-29] WALL AT THE TRUE SURFACE, not at the face.  Along
                            // the face axis the surface (phi = 0.5) sits a fraction
                            // t = (phi_f - 0.5)/(phi_f - phi_s) from f toward s, so the normal
                            // velocity, linear and zero at the surface, is -r u_n at the ghost
                            // with r = |d_s|/d_f = (0.5 - phi_s)/(phi_f - 0.5).  The plain mirror
                            // (r = 1) puts the slip plane on the staircase face: the 15 deg wedge
                            // turned the flow only 14.23 deg (beta -0.78 deg at M 3 and 5).
                            // r = 1 exactly for a 0/1 phi; needs phi carrying sub-cell geometry
                            // (volume fraction / smooth signed-distance profile).
                            const Set::Scalar phf = phisol(fi, fj, fk), phs = phisol(si, sj, sk);
                            Set::Scalar r = (0.5 - phs) / std::max(phf - 0.5, 1e-3);
                            r = std::min(std::max(r, 0.0), wall_ratio_max);
                            pg.u = U[d]  - (1.0 + r) * un * n[d];
                            pg.v = U[c1] - (1.0 + r) * un * n[c1];
#if AMREX_SPACEDIM == 3
                            pg.w = U[c2] - (1.0 + r) * un * n[c2];
#endif
                        }
                        {
                        const Solver::Local::FluidRiemann::State sf = Solver::Local::Limiter::ToState(pf, small);
                        const Solver::Local::FluidRiemann::State sg = Solver::Local::Limiter::ToState(pg, small);
                        fl = fl_lo ? riemannsolver->Solve(sf, sg, pref, small)
                                   : riemannsolver->Solve(sg, sf, pref, small);
                        fl.alpha_face = pf.alpha;
                        }
                    }
                    if (sl && f == dom_lo)      symmetrize(fl);
                    if (sh && f == dom_hi_face) symmetrize(fl);
                    Ff(i, j, k) = fl;
                }
                catch (...)
                {
                    Util::ParallelMessage(INFO, "-------------------------------");
                    Util::ParallelMessage(INFO, "ERROR IN RIEMANN SOLVERS (6-eq)");
                    Util::ParallelMessage(INFO, "lev=", lev, " face dir=", d, " i=", i, " j=", j, " k=", k);
                    Util::Abort(INFO);
                }
            });
        }
        BL_PROFILE_VAR_STOP(prof_faces);
        amrex::Array4<const FluxT> const Fx = face_flux_fab[0].const_array();
        amrex::Array4<const FluxT> const Fy = face_flux_fab[1].const_array();
#if AMREX_SPACEDIM == 3
        amrex::Array4<const FluxT> const Fz = face_flux_fab[2].const_array();
#endif

        // Shell (Gamma) face reconstruction, cached the same way: comp 0 =
        // right edge of the lo cell (qL), comp 1 = left edge of the hi cell
        // (qR).  Only built when the shell row is live.
        const bool shell_faces = shell_row_on;
        amrex::Array4<const Set::Scalar> Sx, Sy;
#if AMREX_SPACEDIM == 3
        amrex::Array4<const Set::Scalar> Sz;
#endif
        if (shell_faces)
        {
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const amrex::Box fbx = amrex::surroundingNodes(bx, d);
                shell_face_fab[d].resize(fbx, 2);
                amrex::Array4<Set::Scalar> const &Sf = shell_face_fab[d].array();
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    // lo cell = (i,j,k) - e_d ; window lo-2 .. lo+3
                    Solver::Local::Limiter::Primitive prim[6];
                    for (int sft = -2; sft <= 3; ++sft)
                        prim[sft + 2].alpha = shell(i + (sft - 1) * di, j + (sft - 1) * dj, k + (sft - 1) * dk);
                    Solver::Local::Limiter::Primitive stL[5] = { prim[0], prim[1], prim[2], prim[3], prim[4] };
                    Solver::Local::Limiter::Primitive stR[5] = { prim[5], prim[4], prim[3], prim[2], prim[1] };
                    Sf(i, j, k, 0) = limiter->ReconstructPassive(stL).alpha;
                    Sf(i, j, k, 1) = limiter->ReconstructPassive(stR).alpha;
                });
            }
            Sx = shell_face_fab[0].const_array();
            Sy = shell_face_fab[1].const_array();
#if AMREX_SPACEDIM == 3
            Sz = shell_face_fab[2].const_array();
#endif
        }

        // [OPT] Viscous stress needs the momentum Hessian (the widest, most
        // expensive stencil in this kernel).  With every viscosity zero,
        // div_tau and visc_diss are identically 0 whatever the Hessian is, so
        // skip it.  hess_u_ is a non-plotted debug field and is zeroed.
        // visc_subcycle: the bulk stress is advanced by SubcycleBulkViscous after the hyperbolic step instead.
        const bool viscous = !visc_subcycle && ((mu0 != 0.0) || (mu1 != 0.0) || (mu0_b != 0.0) || (mu1_b != 0.0));
        const int conduct_l = (thermal_conduction == 1);   // mode 1 = mixture T (mode 2 = per-phase, below)
        const Set::Scalar Pr_l = thermal_Pr;
        const Set::Scalar kph0_l = thermal_k0, kph1_l = thermal_k1;   // thermal.k0/k1 (< 0: mu cp / Pr)
        const Set::Scalar Twall_l = embedded.T_wall;
        const bool abl_on = ablation.on != 0;
        amrex::Array4<Set::Scalar> abl_q;
        if (abl_on) { abl_q = ablation.q_mf[lev]->array(mfi); }
        const bool abl_cond = abl_on && ablation.conduction;
        const Set::Scalar abl_ks = ablation.k_s;
        amrex::Array4<const Set::Scalar> abl_Ts;
        amrex::Array4<Set::Scalar> abl_qs;
        if (abl_cond) { abl_Ts = ablation.Ts_mf[lev]->const_array(mfi); abl_qs = ablation.qs_mf[lev]->array(mfi); }
        BL_PROFILE_VAR("Hydro2::RHS::cell_update", prof_cells);
        const amrex::Box vbx_cc = mfi.validbox();
        const auto vbx_lo = amrex::lbound(vbx_cc);
        // RK stage weight b_i for the reflux flux accumulation (see Advance).
        const Set::Scalar rw = reflux_stage_weight;

        const int symc = cen_face_any;   // symmetry + RADIAL faces
        const bool symlo_c[3] = {cen_face_lo[0], cen_face_lo[1], cen_face_lo[2]};
        const bool symhi_c[3] = {cen_face_hi[0], cen_face_hi[1], cen_face_hi[2]};
        const int gu_vel = gradu_from_velocity, visc_ff = visc_flux_form_rk;
        const int visc_harm = visc_harmonic;
        const int visc_subcycle_l = visc_subcycle;
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            if (symc)        // mirrored ghosts -> central is exact (see Parse)
            {
                const amrex::Dim3 dlo = amrex::lbound(domain), dhi = amrex::ubound(domain);
                const int ijk[3] = {i, j, k}, lo3[3] = {dlo.x, dlo.y, dlo.z}, hi3[3] = {dhi.x, dhi.y, dhi.z};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    if ((symlo_c[d] && ijk[d] == lo3[d]) || (symhi_c[d] && ijk[d] == hi3[d]))
                        sten[d] = Numeric::StencilType::Central;
            }

            // Diffuse Sources
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Vector n_hat = grad_eta / (grad_eta_mag + g_gem_min); // Normal Vector (taper below the interface gate)

            // Extract velocity from momentum and density.  The mixture velocity u
            // is fully dimensioned (momentum spine carries z); the prescribed-flow
            // u0 keeps a 0.0 z-placeholder (its IC field stays 2-component).
            Set::Vector u = Set::Vector(AMREX_D_DECL(v(i, j, k, 0), v(i, j, k, 1), v(i, j, k, 2)));
            Set::Vector u0 = Set::Vector(AMREX_D_DECL(_u0(i, j, k, 0), _u0(i, j, k, 1), 0.0));

            Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
            Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
            Set::Matrix gradu = (gradM - u * gradrho.transpose()) / (rho(i, j, k));
            if (gu_vel) gradu = Numeric::Gradient(v, i, j, k, DX);          // gradu_from_velocity (see Parse)

            Set::Vector q0_ = Set::Vector(AMREX_D_DECL(q0(i, j, k, 0), q0(i, j, k, 1), 0.0));  // q0 IC field stays 2-component

            /// Calculate Source Terms
            // Shear:
            Set::Scalar mdot0 = -m0(i, j, k) * grad_eta_mag;
            Set::Vector Pdot0 = Set::Vector::Zero();
            Set::Scalar qdot0 = q0_.dot(grad_eta);

            Set::Matrix3 hess_u = Set::Matrix3::Zero();
            if (viscous)
            {
                Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
                Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
                for (int p = 0; p < AMREX_SPACEDIM; p++)
                    for (int q = 0; q < AMREX_SPACEDIM; q++)
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                        {
                            hess_u(r, p, q) = (hess_M(r, p, q) - gradu(r, q) * gradrho(p) - gradu(r, p) * gradrho(q) - u(r) * hess_rho(p, q))
                                              / (rho(i, j, k));
                        }
                if (gu_vel) hess_u = Numeric::Hessian(v, i, j, k, DX);          // gradu_from_velocity (see Parse)
            }

            // [2026-09-27] SHARP-WALL VISCOUS STENCIL (solid.wall_flux = 1, solid.visc_mirror = 1).
            // The mirrored Riemann flux puts the wall on the fluid|solid FACE, but
            // the stencils above read the solid cells' own velocity (u_s ~ 0) one
            // full cell away: the viscous terms saw the no-slip wall dx/2 inside
            // the solid, half the wall shear, and a spurious face slip velocity
            // -- inconsistent with the inviscid wall (first order in dx; measured
            // as a lift deficit growing with Re on the NACA 0008).  Use the same
            // ghost-mirror as compute_face: every stencil cell across the wall
            // takes u_g = 2 u_wall - u_c, so u = u_wall exactly on the face.
            //   fluid cell: gradu / hess_u from central differences of that
            //     mirrored local field (only cells with a solid neighbour).
            //   solid cell (inert; its div_tau only feeds rhs_force): per fluid
            //     neighbour n the face shear is mu (u_n - u_g)/dx with
            //     u_g = 2 u_s - u_n, i.e. exactly what the fluid cell loses, so
            //     the reported viscous force is the one the fluid feels.
            if (viscous && sharp_wall && visc_mirror)
            {
                const bool self_solid = phisol(i, j, k) < 0.5;
                bool near_wall = false;
                for (int a = -1; a <= 1; ++a)
                for (int b = -1; b <= 1; ++b)
#if AMREX_SPACEDIM == 3
                for (int c = -1; c <= 1; ++c)
#else
                for (int c = 0; c <= 0; ++c)
#endif
                    if ((phisol(i + a, j + b, k + c) < 0.5) != self_solid) near_wall = true;

                if (near_wall)
                {
                    auto uraw = [&](int a, int b, int c, int r) {
                        return M(i + a, j + b, k + c, r) / std::max(rho(i + a, j + b, k + c), small);
                    };
                    // wall velocity = the solid's prescribed velocity (as compute_face)
                    auto uwall = [&](int a, int b, int c, int r) {
                        return s_M(i + a, j + b, k + c, r)
                               / std::max(s_re0(i + a, j + b, k + c) + s_re1(i + a, j + b, k + c), small);
                    };
                    Set::Scalar uc[AMREX_SPACEDIM];
                    for (int r = 0; r < AMREX_SPACEDIM; r++) uc[r] = uraw(0, 0, 0, r);
                    int e[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                    // [2026-10-02] TRUE-SURFACE VISCOUS GHOST (solid.visc_mirror = 2).  The
                    // wall sits at phi = 0.5 between the fluid cell (phi_f) and the solid
                    // stencil cell (phi_s), a fraction t = (phi_f - 0.5)/(phi_f - phi_s) of
                    // the spacing from the fluid cell.  Linear ghost through u_wall there:
                    //   u_g = u_w - r (u_f - u_w),  r = (1 - t)/t = (0.5 - phi_s)/(phi_f - 0.5),
                    // so (u_f - u_g)/dx = (u_f - u_w)/(t dx): the face shear is the exact
                    // one-sided gradient to the true surface along that axis.  Summed over
                    // the staircase faces this gives mu tau L (sum n_a^2 = 1) instead of the
                    // ~4/pi over-count of the plain mirror (r = 1, visc_mirror = 1), and r = 1
                    // for a 0/1 phi.  Conservative: the solid cell's bookkeeping below uses
                    // the same (1 + r).  r capped at visc_ratio_max (explicit viscous limit:
                    // diagonal (2 + 2 + r)/dx^2 vs 4/dx^2; cfl_v dx^2/(4 nu) stays inside).
                    const bool visc_true = embedded.visc_mirror == 2;
                    const Set::Scalar visc_ratio_max = 4.0;
                    auto visc_ratio = [&](Set::Scalar ph_f, Set::Scalar ph_s) {
                        if (!visc_true) return 1.0;
                        const Set::Scalar r = (0.5 - ph_s) / std::max(ph_f - 0.5, 1e-3);
                        return std::min(std::max(r, 0.0), visc_ratio_max);
                    };
                    gradu = Set::Matrix::Zero();
                    hess_u = Set::Matrix3::Zero();
                    if (!self_solid)
                    {
                        const Set::Scalar ph_c = phisol(i, j, k);
                        // mirrored velocity at offset (a,b,c): plain (r = 1) or true-surface ghost
                        auto um = [&](int a, int b, int c, int r) {
                            const Set::Scalar ph = phisol(i + a, j + b, k + c);
                            if (ph >= 0.5) return uraw(a, b, c, r);
                            const Set::Scalar uw = uwall(a, b, c, r);
                            return uw - visc_ratio(ph_c, ph) * (uc[r] - uw);
                        };
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                            for (int p = 0; p < AMREX_SPACEDIM; p++)
                            {
                                const int* ep = e[p];
                                gradu(r, p) = (um(ep[0], ep[1], ep[2], r) - um(-ep[0], -ep[1], -ep[2], r)) / (2.0 * DX[p]);
                                for (int q = 0; q < AMREX_SPACEDIM; q++)
                                {
                                    const int* eq = e[q];
                                    if (p == q)
                                        hess_u(r, p, p) = (um(ep[0], ep[1], ep[2], r) - 2.0 * uc[r] + um(-ep[0], -ep[1], -ep[2], r)) / (DX[p] * DX[p]);
                                    else
                                        hess_u(r, p, q) = (um( ep[0] + eq[0],  ep[1] + eq[1],  ep[2] + eq[2], r)
                                                         - um( ep[0] - eq[0],  ep[1] - eq[1],  ep[2] - eq[2], r)
                                                         - um(-ep[0] + eq[0], -ep[1] + eq[1], -ep[2] + eq[2], r)
                                                         + um(-ep[0] - eq[0], -ep[1] - eq[1], -ep[2] - eq[2], r)) / (4.0 * DX[p] * DX[q]);
                                }
                            }
                    }
                    else
                    {
                        // wall-shear exchange only: (u_n - u_g)/dx^2 = (1 + r)(u_n - u_s)/dx^2 per
                        // fluid neighbour n (r = 1: plain mirror) -- exactly what n loses
                        const Set::Scalar ph_c = phisol(i, j, k);
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                            for (int p = 0; p < AMREX_SPACEDIM; p++)
                                for (int sgn = -1; sgn <= 1; sgn += 2)
                                {
                                    const int a = sgn * e[p][0], b = sgn * e[p][1], c = sgn * e[p][2];
                                    const Set::Scalar ph_n = phisol(i + a, j + b, k + c);
                                    if (ph_n >= 0.5)
                                        hess_u(r, p, p) += (1.0 + visc_ratio(ph_n, ph_c)) * (uraw(a, b, c, r) - uwall(0, 0, 0, r)) / (DX[p] * DX[p]);
                                }
                    }
                }
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

            // DIAGNOSTIC interface-localised artificial viscosity (see Hydro2.H).
            // Body-force pair: momentum gets F_art, energy gets u.F_art below.

            // EMBEDDED SOLID
            Set::Scalar phi_c = 1.0;
            if (embedded.apply)
            {
                phi_c = embedded.clampPhi(phisol(i, j, k));
                // [2026-09-25] SOLID VISCOSITY (solid.mu_factor, as in single-phase
                // Hydro): mu_solid = mu_factor * mu_f(eta), blended by the solid
                // fraction, mu_eff = phi mu_f + (1 - phi) mu_solid = w mu_f with
                // w = phi + (1 - phi) mu_factor.  Enforces no-slip through the
                // viscous stress across the diffuse wall band.  The gradient
                // term picks up mu_f dw = mu_f (1 - mu_factor) grad(phi).  Same
                // factor on the bulk viscosity.  Inviscid (mu_f = 0): no effect.
                const Set::Scalar fac = embedded.mu_factor;
                if (fac != 1.0)
                {
                    const Set::Scalar w = phi_c + (1.0 - phi_c) * fac;
                    const Set::Vector gphi = Numeric::Gradient(phisol, i, j, k, 0, DX);
                    grad_mu     = w * grad_mu     + (1.0 - fac) * mu_eff     * gphi;
                    grad_lambda = w * grad_lambda + (1.0 - fac) * lambda_eff * gphi;
                    mu_eff     *= w;
                    lambda_eff *= w;
                }
            }

            // Solving (every term is proportional to a viscosity -> skip when inviscid)
            if (viscous)
            for (int p = 0; p < AMREX_SPACEDIM; p++)             // i
                for (int q = 0; q < AMREX_SPACEDIM; q++)         // j
                    for (int r = 0; r < AMREX_SPACEDIM; r++)     // k
                        for (int s = 0; s < AMREX_SPACEDIM; s++) // l
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
                            Ldot(p) += 0.0;

                            // Grad visc terms
                            div_tau(p) += dMpqrs * gradu(r, s);
                        }

            if (viscous && visc_ff)         // visc_flux_form (see Parse): div tau = sum_q (tau_q+ - tau_q-)/dx_q
            {
                div_tau = Set::Vector::Zero();
                const int c0[3] = {i, j, k};
                auto vel = [&](const int a[3], int n) { return v(a[0], a[1], a[2], n); };
                auto dcen = [&](const int a[3], int n, int dir)     // central difference of u_n along dir at cell a
                {
                    int ap[3] = {a[0], a[1], a[2]}, am[3] = {a[0], a[1], a[2]}; ap[dir] += 1; am[dir] -= 1;
                    return (vel(ap, n) - vel(am, n)) / (2.0 * DX[dir]);
                };
                for (int q = 0; q < AMREX_SPACEDIM; q++)
                    for (int sgn = -1; sgn <= 1; sgn += 2)
                    {
                        int nb[3] = {c0[0], c0[1], c0[2]}; nb[q] += sgn;
                        const Set::Scalar e_f = 0.5 * (eta(i, j, k) + eta(nb[0], nb[1], nb[2]));
                        const Set::Scalar mu_f = (visc_harm && mu0 > 0.0 && mu1 > 0.0)
                                               ? 1.0 / (e_f / mu0 + (1.0 - e_f) / mu1)
                                               : e_f * mu0 + (1.0 - e_f) * mu1;
                        const Set::Scalar la_f = e_f * mu0_b + (1.0 - e_f) * mu1_b;
                        Set::Scalar divu_f = 0.0;
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                            divu_f += (r == q) ? sgn * (vel(nb, r) - vel(c0, r)) / DX[q] : 0.5 * (dcen(c0, r, r) + dcen(nb, r, r));
                        for (int p = 0; p < AMREX_SPACEDIM; p++)
                        {
                            const Set::Scalar dq_up = sgn * (vel(nb, p) - vel(c0, p)) / DX[q];                                 // d_q u_p at the face
                            const Set::Scalar dp_uq = (p == q) ? dq_up : 0.5 * (dcen(c0, q, p) + dcen(nb, q, p));              // d_p u_q at the face
                            Set::Scalar tau = mu_f * (dq_up + dp_uq);
                            if (p == q) tau += (la_f - (2.0 / 3.0) * mu_f) * divu_f;
                            div_tau(p) += sgn * tau / DX[q];
                        }
                    }
            }

            // Debugging feild for div_tau and Ldot
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                div_tau_(i, j, k, d) = div_tau(d);

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Ldot_(i, j, k, d) = Ldot(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Viscosity solving", {
                { "Ldot[0]", Ldot(0) },
                { "Ldot[1]", Ldot(1) },
                { "div_tau[0]", div_tau(0) },
                { "div_tau[1]", div_tau(1) }
            }); // end check4nans
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Ldot[2]", Ldot(2) },
                { "div_tau[2]", div_tau(2) },
            }); // end check4nans
#endif  
            
            // ------------------------------------------------------------
            // Surface Tension
            // ------------------------------------------------------------
            // Fsv =  simga * kappa * grad_eta
            Set::Vector Fsv_vector = Set::Vector::Zero();
            // Under the split scheme the capillary terms belong to L_cap
            // (Schmidmayer eq. 17); applying them here too would double-count.
            if (apply_surface_tension)
            {
                // Conservative continuum-surface-stress force  F = div(Omega)
                // (Schmidmayer 2017).  Centered divergence of the capillary stress
                // tensor stored in om_cap.
                const Set::Scalar idx = 0.5 / DX[0], idy = 0.5 / DX[1];
#if AMREX_SPACEDIM == 2
                Fsv_vector(0) = (om_cap(i+1,j,k,0) - om_cap(i-1,j,k,0)) * idx
                              + (om_cap(i,j+1,k,2) - om_cap(i,j-1,k,2)) * idy;
                Fsv_vector(1) = (om_cap(i+1,j,k,2) - om_cap(i-1,j,k,2)) * idx
                              + (om_cap(i,j+1,k,1) - om_cap(i,j-1,k,1)) * idy;
#else
                const Set::Scalar idz = 0.5 / DX[2];
                Fsv_vector(0) = (om_cap(i+1,j,k,0) - om_cap(i-1,j,k,0)) * idx
                              + (om_cap(i,j+1,k,3) - om_cap(i,j-1,k,3)) * idy
                              + (om_cap(i,j,k+1,4) - om_cap(i,j,k-1,4)) * idz;
                Fsv_vector(1) = (om_cap(i+1,j,k,3) - om_cap(i-1,j,k,3)) * idx
                              + (om_cap(i,j+1,k,1) - om_cap(i,j-1,k,1)) * idy
                              + (om_cap(i,j,k+1,5) - om_cap(i,j,k-1,5)) * idz;
                Fsv_vector(2) = (om_cap(i+1,j,k,4) - om_cap(i-1,j,k,4)) * idx
                              + (om_cap(i,j+1,k,5) - om_cap(i,j-1,k,5)) * idy
                              + (om_cap(i,j,k+1,2) - om_cap(i,j,k-1,2)) * idz;
#endif
                {
                    // Tangential part of div(Omega) (Marangoni; zero unless
                    // sigma(Gamma)) + the normal jump force -Delta_s grad(eta)
                    // stored per cell.
                    if (grad_eta_mag > g_gem_min)
                    {
                        const Set::Vector nh = grad_eta / grad_eta_mag;
                        Fsv_vector -= nh * nh.dot(Fsv_vector);
                    }
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) Fsv_vector(d) += om_cap(i, j, k, nO_f + d);
                }
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Fsv(i, j, k, d) = Fsv_vector(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fsv_vector[0]", Fsv_vector(0) },
                { "Fsv_vector[1]", Fsv_vector(1) }
            });
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fsv_vector[2]", Fsv_vector(2) }
            }); // end check4nans
#endif  

            // ------------------------------------------------------------
            // Weight
            // ------------------------------------------------------------
            // Fw = - rho * g
            Set::Vector Fw_vector = Set::Vector::Zero();
            if (apply_weight)
            {
                // Gravity acts along the y-axis (component 1), as in the 2D model.
                Fw_vector(1) = -rho(i, j, k) * g;
            }
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Fw(i, j, k, d) = Fw_vector(d);

            // ERROR CHECKING
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Weight solving", {
                { "Fw_vector[0]", Fw_vector(0) },
                { "Fw_vector[1]", Fw_vector(1) }
            });
#if AMREX_SPACEDIM == 3
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Surface Tension solving", {
                { "Fw_vector[2]", Fw_vector(2) }
            }); // end check4nans
#endif

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
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Cahn-Hillard solving", {
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
            Set::Scalar Q_L_Vap = 0.0;      // heat into the liquid [W/m^3] (model 3 only)
            Set::Scalar e_l_Vap = 0.0;      // liquid specific internal energy carried by the vaporized mass
            if (apply_vaporization == 1 && vap_model >= 1)
            {
                // FILM EVAPORATION (vaporization.model 1-3).  Phase temperatures from the
                // stiffened-gas caloric law of Le Metayer, Massoni & Saurel (2004):
                //     T_k = (e_k - q_k - pi_k / rho_k) / cv_k   (= (p_k + pi_k)/((gamma_k-1) rho_k cv_k))
                // NOT the plotted Tammann T = (p + gamma pi)/(...), which is wrong for a liquid.
                const Set::Scalar a_g = eta(i, j, k), a_l = 1.0 - eta(i, j, k);
                // Only where both phases hold at least vaporization.alpha_min: the phase energies of a trace
                // phase are not reliable (the energy re-sync after relaxation gives a few-percent liquid in
                // the wake mist temperatures of 1e5 K even with vaporization off).  wnorm restores the
                // interface area the cut tails held: int 6 eta(1-eta) over [0,a]U[1-a,1] = 2(3a^2 - 2a^3).
                const Set::Scalar floor_a = vap_alpha_min;
                const Set::Scalar wnorm = 1.0 / (1.0 - 2.0 * (3.0 * floor_a * floor_a - 2.0 * floor_a * floor_a * floor_a));
                Set::Scalar Ts = 0.0, Tg = 0.0, BMv = 0.0, BTv = 0.0;
                // Also skipped where the gas is near vacuum (rho_g < vaporization.rho_gas_min): the wake axis
                // behind a column carries gas at 1e-5..1e-4 kg/m^3 even with vaporization off, and drawing
                // the film heat from that little gas drives its pressure to zero.
                if (a_g > floor_a && a_l > floor_a && rho_eta0(i, j, k) > small && rho_eta1(i, j, k) > small
                    && rho_eta0(i, j, k) / a_g >= vap_rho_gas_min)
                {
                    const Set::Scalar rho_g = rho_eta0(i, j, k) / a_g, rho_l = rho_eta1(i, j, k) / a_l;
                    const Set::Scalar e_g = E0_arr(i, j, k) / rho_eta0(i, j, k), e_l = E1_arr(i, j, k) / rho_eta1(i, j, k);
                    // Capped at vaporization.T_gas_max: mixed-cell gas energies carry outliers far above any
                    // physical gas temperature (top 1 % of band cells at 15-25x the post-shock T in Ms 1.47
                    // runs, vaporization off).  The physical bound is the stagnation temperature of the flow.
                    Tg = std::min((e_g - vap_q_gas - vpi0 / rho_g) / vcv0, vap_T_gas_max);
                    Ts = (e_l - vap_q_liquid - vpi1 / rho_l) / vcv1;
                    const Set::Scalar cp_g = vg0 * vcv0;
                    const Set::Scalar pmix = std::max(press(i, j, k), 1.0);
                    // saturation pressure at the liquid temperature
                    Set::Scalar psat;
                    if (vap_psat == 0) { const Set::Scalar Tc = Ts - 273.15;            // Buck (1996), over water
                                         psat = 611.21 * std::exp((18.678 - Tc / 234.5) * (Tc / (257.14 + Tc))); }
                    else if (vap_psat == 1) psat = vap_p_ref * std::exp(Lv * vap_W_vapor / R * (1.0 / vap_T_ref - 1.0 / std::max(Ts, 1.0)));
                    else psat = 1.0e5 * std::pow(10.0, vap_antoine_A - vap_antoine_B / std::max(Ts + vap_antoine_C, 1.0));
                    const Set::Scalar Xs = std::min(psat / pmix, 0.999);
                    const Set::Scalar Ys = Xs * vap_W_vapor / (Xs * vap_W_vapor + (1.0 - Xs) * vap_W_gas);
                    // far-field vapour fraction: the constant Y_infinity, or the cell's own vapour (vaporization.Y_inf_local)
                    const Set::Scalar Yinf = (vYloc && sp_on) ? std::min(std::max(sp_arr(i, j, k, vspg), 0.0) / rho_eta0(i, j, k), 1.0)
                                                              : Y_infinity;
                    BMv = std::max((Ys - Yinf) / (1.0 - Ys), 0.0);    // evaporation only
                    BTv = std::max(cp_g * (Tg - Ts) / Lv, 0.0);
                    // Interface-area density.  6 a_g a_l |grad eta| integrates to the interface area across a
                    // tanh profile (int_0^1 6 eta (1 - eta) d eta = 1), like |grad eta|, but vanishes where a
                    // phase does: the mass taken from (and the heat put into) the liquid per unit liquid mass
                    // stays bounded in cells that hold almost no liquid.
                    const Set::Scalar wband = 6.0 * a_g * a_l * grad_eta_mag * wnorm;
                    Set::Scalar j_flux = 0.0;                               // [kg/m^2/s]
                    if (vap_model == 2) j_flux = vap_k_gas / (cp_g * vap_delta_T) * std::log1p(BTv);
                    else                j_flux = rho_g * Dv / vap_delta * std::log1p(BMv);
                    if (vap_model == 3)
                    {
                        // Abramzon & Sirignano (1989): (1 + B_T') = (1 + B_M)^phi,
                        // phi = (cp_v / cp_g) (Sh / Nu) / Le,  Sh/Nu = delta_T/delta,  Le = k/(rho cp_g Dv);
                        // heat into the liquid Q_L = j [cp_v (T_g - T_s)/B_T' - Lv]   (per unit area)
                        const Set::Scalar Le = vap_k_gas / (rho_g * cp_g * Dv);
                        const Set::Scalar phi = (vap_cp_vapor / cp_g) * (vap_delta_T / vap_delta) / Le;
                        const Set::Scalar BTp = std::expm1(phi * std::log1p(BMv));
                        Set::Scalar qL;
                        if (BTp > 1.0e-8) qL = j_flux * (vap_cp_vapor * (Tg - Ts) / BTp - Lv);
                        else              qL = vap_k_gas * (Tg - Ts) / vap_delta_T - j_flux * Lv;  // B -> 0 limit
                        Q_L_Vap = qL * wband;
                    }
                    m_dot_Vap = j_flux * wband;
                    // Mass moves at fixed volume fractions; the mechanical (pressure) relaxation after every
                    // stage then sets the volumes.  The explicit volume source m_dot (1/rho_g - 1/rho_l)
                    // (vaporization.volume_source = 1) runs away where the gas is rarefied: it adds gas
                    // volume without gas mass, rho_g -> 0, p -> 0, which raises Y_s and m_dot further.
                    if (vap_volume_source) eta_dot_Vap = m_dot_Vap * (1.0 / rho_g - 1.0 / rho_l);
                    E_dot_Vap = -m_dot_Vap * Lv;
                    e_l_Vap = e_l;
                }
                Vap_film(i, j, k, 0) = Ts; Vap_film(i, j, k, 1) = Tg; Vap_film(i, j, k, 2) = BMv;
                Vap_film(i, j, k, 3) = BTv; Vap_film(i, j, k, 4) = Q_L_Vap;
            }
            else if (apply_vaporization == 1)
            {
                // Spalding mass transfer number (from the vapor-mass-fraction
                // surrogate Y = (alpha rho)_0 / rho computed above).
                Set::Scalar B_M = Bm(i, j, k);
                //B_M = std::max(B_M, 0.0); // Only evaporation, no condensation in this formulation

                // PURE-phase densities rho_k = (alpha rho)_k / alpha_k (the
                // density0/1_mf primitives from FillGhost4BC).  The old code
                // used the PARTIAL densities (alpha rho)_k here, which (i)
                // scaled the evaporation rate by a spurious factor alpha_g and
                // (ii) made the volume-transfer term blow up where a phase
                // vanishes (1/max(alpha*rho, small) -> 1e8).
                Set::Scalar rho_g = std::max(rho0(i, j, k), small);   // gas    (fluid 0; eta = alpha_gas)
                Set::Scalar rho_l = std::max(rho1(i, j, k), small);   // liquid (fluid 1)

                // Mass-transfer rate (volumetric) -- simplified Spalding form,
                // localized to the band by |grad(eta)|.  Dv is a calibrated
                // input that carries the film-length division of the classical
                // d^2-law rate (rho_g D_v / delta_film) ln(1+B_M).
                m_dot_Vap = rho_g * Dv * (B_M / (1.0 + B_M + small)) * grad_eta_mag;

                // Volume-fraction source: evaporation (m_dot > 0) GROWS the
                // gas volume fraction -- gas gains volume at 1/rho_g per unit
                // transferred mass while liquid loses it at 1/rho_l:
                //     d(alpha_g)/dt += m_dot (1/rho_g - 1/rho_l) > 0.
                // (The old (1/rho_l - 1/rho_g) partial-density form had the
                // sign INVERTED: evaporation shrank the gas volume fraction.)
                eta_dot_Vap = m_dot_Vap * (1.0 / rho_g - 1.0 / rho_l);

                // Evaporative cooling: the stiffened-gas EOSs carry no heat-of-
                // vaporization offset, so the latent heat enters as an explicit
                // sink -m_dot * Lv [W/m^3] on the mixture energy (applied in
                // the Source energy row below; Lv = 0 disables).  Replaces the
                // old m_dot |u|^2 |grad eta|^2 term, which was dimensionally
                // inconsistent and never actually applied.
                E_dot_Vap = -m_dot_Vap * Lv;
                if (rho_eta1(i, j, k) > small) e_l_Vap = E1_arr(i, j, k) / rho_eta1(i, j, k);
            }
            // Vaporization Trackers.  Layout: [_eta, _rho, M..., _E].
            Vap_dot(i, j, k, 0) = eta_dot_Vap;
            Vap_dot(i, j, k, 1) = m_dot_Vap;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Vap_dot(i, j, k, 2 + d) = 0.0;
            Vap_dot(i, j, k, AMREX_SPACEDIM + 2) = E_dot_Vap;


            // Total:
            Set::Vector Total_Force = Fsv_vector + Fw_vector;

            // ------------------------------------------------------------
            // Energy work terms in conservative (divergence) form.
            //
            // Viscous:   div(tau.u) = u.div(tau) + tau:grad(u).  The second
            // term is the dissipation Phi >= 0 (arrested KE -> heat); with
            // only u.div(tau) the total energy is not conserved.
            //   Phi = 2 mu_eff S:S + (lambda_eff - 2/3 mu_eff) (div u)^2,
            //   S = sym(grad u).
            //
            // Capillary: the FLUID energy (rho E = internal + kinetic, which
            // does NOT contain the surface energy sigma|grad eta|) receives
            // exactly the capillary-force work u.div(Omega) -- already inside
            // u.dot(Total_Force) below.  The identity
            //   u.div(Omega) = div(Omega.u) - Omega:grad(u)
            // says the fluid absorbs the transported capillary work MINUS the
            // Omega:grad(u) exchange, which is the rate the SURFACE reservoir
            // gains:  d(sigma|grad eta|)/dt + div(sigma|grad eta| u)
            //       = Omega:grad(u)   (exact for advected eta).
            // A previous revision added cap_work = Omega:grad(u) ON TOP of
            // u.dot(Total_Force), handing the fluid div(Omega.u) total -- i.e.
            // the surface-exchange leg was double-counted into rho E, CREATING
            // energy whenever the interface stretched and destroying it on
            // shrink (measured on the oscillating-bubble test as a per-cycle
            // amplitude pump).  The reservoir itself needs no PDE here: it is
            // the state function sigma_eff |grad eta| (auditable offline from
            // the plotted kappa[2] and grad_eta fields).  Folding it INTO the
            // conserved state (full Schmidmayer 2017 E_t) remains the deeper
            // upgrade -- it touches the Riemann state, EOS inversions, and
            // relaxation, and is deferred deliberately.
            // ------------------------------------------------------------
            Set::Matrix Ssym = 0.5 * (gradu + gradu.transpose());
            Set::Scalar divu_cell = gradu.trace();
            Set::Scalar visc_diss = visc_subcycle_l ? 0.0          // sub-cycled: SubcycleBulkViscous books it
                                  : 2.0 * mu_eff * Ssym.squaredNorm()
                                  + (lambda_eff - (2.0 / 3.0) * mu_eff) * divu_cell * divu_cell;

            // SOURCES.  Layout: [0]=mdot, [1..SD]=momentum, [SD+1]=energy.
            Source(i, j, k, 0) = mdot0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Source(i, j, k, 1 + d) = Pdot0(d) + Ldot(d) + div_tau(d) + Total_Force(d);
            // Capillary/shell energy: div(Omega . u), see above.
            Set::Scalar cap_w = 0.0;
            // Work of the capillary force only (see Parse, CONSISTENT JUMP)
            if (apply_surface_tension) cap_w = u.dot(Fsv_vector);
            Source(i, j, k, AMREX_SPACEDIM + 1) = qdot0 + u.dot(div_tau) + visc_diss + u.dot(Ldot)
                                                + u.dot(Fw_vector) + cap_w + E_dot_Vap;

            // Lagrange terms to enforce no-penetration
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                Source(i, j, k, 1 + d) -= lagrange * u.dot(grad_eta) * grad_eta(d);

            // ------------------------------------------------------------
            // Error Checking
            // ------------------------------------------------------------
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Source solving", { 
                { "Total_Force[0]",  Total_Force(0) },
                { "Total_Force[1]",  Total_Force(1) },
                { "Source[0]",  Source(i, j, k, 0) },
                { "Source[1]",  Source(i, j, k, 1) },
                { "Source[2]",  Source(i, j, k, 2) },
                { "Source[3]",  Source(i, j, k, 3) }
            }); // end check4nans

            // ============================================================
            // 6-equation HLLC face fluxes (Saurel 2009 Sec 3.1.2 / Schmidmayer 2020 Sec 3).
            // [OPT] Read from the per-box face cache built above (each face
            // solved once; symmetry faces already enforced there).
            // ============================================================
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Hydro2()::RHS(): Conservative Variable Check", {
                { "eta", eta(i, j, k) },
                { "rho_eta0", rho_eta0(i, j, k) },
                { "rho_eta1", rho_eta1(i, j, k) },
                { "rho", rho(i, j, k) },
                { "M[0]", M(i, j, k, 0) },
                { "M[1]", M(i, j, k, 1) },
                { "E", E(i, j, k) },
                { "E0", E0_arr(i, j, k) },
                { "E1", E1_arr(i, j, k) },
                { "press", press(i, j, k) },
            }); // end check4nans

            const FluxT &flux_xlo = Fx(i,     j,     k);
            const FluxT &flux_xhi = Fx(i + 1, j,     k);
            const FluxT &flux_ylo = Fy(i,     j,     k);
            const FluxT &flux_yhi = Fy(i,     j + 1, k);
#if AMREX_SPACEDIM == 3
            const FluxT &flux_zlo = Fz(i,     j,     k);
            const FluxT &flux_zhi = Fz(i,     j,     k + 1);
#endif
            // NOTE: two rearrangements of the capillary force were implemented and
            // REMOVED.  (1) div(Omega) from face-evaluated Omega; (2) Omega carried
            // in the momentum flux, F = rho u u + p I - Omega, subtracted from the
            // Riemann fluxes.  These are ALGEBRAICALLY IDENTICAL -- differencing
            // face fluxes reproduces the face divergence exactly -- and measured
            // identical to 5 s.f. at t=20us on the static R0=24mm bubble (|u|max
            // 0.05015 both).  Both are WORSE than the cell-centred 2*dx source
            // (|u|max at 0.3ms: 29.9 source vs 78.0 face-Fsv), and (2) additionally
            // crashed before 0.18 ms because subtracting Omega AFTER the Riemann
            // solve leaves the solver's wave speeds unaware of it, so the fluxes
            // stop being consistent with the reconstructed states.
            // A real flux-form treatment needs Omega inside the WAVE STRUCTURE
            // (modified signal speeds), i.e. a change to the Riemann solver itself
            // -- not a rearrangement of where div(Omega) is evaluated.

            // ============================================================
            // Saurel-Abgrall non-conservative discretization for alpha and
            // per-phase internal energies (Saurel 2009 lines 836 / 852):
            //
            //   div(alpha u) computed from FACE-UPWIND alpha_face and
            //                S_M (= flux.u_interface).
            //   div(u)       from S_M.
            //   eta_rhs      = -div(alpha u) + alpha_C * div(u)
            //                = - u . grad(alpha)             (Sch20 row 1)
            //   E_k_rhs      = -div(alpha rho e_k u)
            //                  - alpha_k_C * p_k_C * div(u)  (Sau09 line 852)
            // where C subscript indicates CELL-CENTERED, TIME-n FROZEN
            // (the Saurel-Abgrall trick that makes the Abgrall uniform-flow
            //  test pass exactly).
            // ============================================================

            // Cell-centered per-phase pressures at time n (frozen for the
            // non-conservative source).
            const Set::Scalar a1_C   = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a2_C   = 1.0 - a1_C;
            const Set::Scalar p0_C   = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1_C, PC(i, j, k, NPC), PC(i, j, k, NPC + 2), small);
            const Set::Scalar p1_C   = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2_C, PC(i, j, k, NPC + 1), PC(i, j, k, NPC + 3), small);

            // ALPHA ROW (eta_consistent_advect, see Parse).  1 (default): the limiter-reconstructed
            // face alpha, upwinded on the contact speed S_M, consistent with the mass rows (flames2:
            // donor cell made pressure-floor cells in the gas core).  0: first-order donor cell (the
            // DropletDetonation default before the 2026-11-23 merge).  On outflow faces the
            // reconstructed value is not bound by the upwind maximum principle, so it is clamped to [0,1].
            Set::Scalar a_face_xlo = (flux_xlo.u_interface > 0.0) ? eta(i - 1, j, k) : eta(i,     j, k);
            Set::Scalar a_face_xhi = (flux_xhi.u_interface > 0.0) ? eta(i,     j, k) : eta(i + 1, j, k);
            Set::Scalar a_face_ylo = (flux_ylo.u_interface > 0.0) ? eta(i, j - 1, k) : eta(i, j,     k);
            Set::Scalar a_face_yhi = (flux_yhi.u_interface > 0.0) ? eta(i, j,     k) : eta(i, j + 1, k);
            if (eta_consistent_advect == 1)
            {
                a_face_xlo = flux_xlo.alpha_face;
                a_face_xhi = flux_xhi.alpha_face;
                a_face_ylo = flux_ylo.alpha_face;
                a_face_yhi = flux_yhi.alpha_face;
            }
            // [MERGE 2026-10-05] SHARP WALL + donor cell.  Across a fluid|solid face the
            // donor value must not be the inert solid cell's eta (with u* <= 0 the
            // plain upwind pick on a hi-side wall face is exactly that cell, and a
            // moving wall has u* != 0 on both sides).  The wall-face flux already
            // carries the FLUID cell's alpha (alpha_face = pf.alpha), so use it there.
            if (sharp_wall)
            {
                if (phisol(i - 1, j, k) < 0.5) a_face_xlo = flux_xlo.alpha_face;
                if (phisol(i + 1, j, k) < 0.5) a_face_xhi = flux_xhi.alpha_face;
                if (phisol(i, j - 1, k) < 0.5) a_face_ylo = flux_ylo.alpha_face;
                if (phisol(i, j + 1, k) < 0.5) a_face_yhi = flux_yhi.alpha_face;
            }
            a_face_xlo = std::min(std::max(a_face_xlo, 0.0), 1.0);
            a_face_xhi = std::min(std::max(a_face_xhi, 0.0), 1.0);
            a_face_ylo = std::min(std::max(a_face_ylo, 0.0), 1.0);
            a_face_yhi = std::min(std::max(a_face_yhi, 0.0), 1.0);
            // Colour function c: face-upwind on the SAME contact velocity as
            // alpha, so c and eta share one discrete advection operator.
            // Schmidmayer 2017 sec. 4.2.1: the c row of H_nc is -c, exactly as
            // the alpha_1 row is -alpha_1, and both are driven by the Riemann
            // star velocity u*.  Advecting c with the cell-centred u instead
            // (as the first implementation did) gives it a different truncation
            // error from eta, and the two separate -- measured at 2.28% in R
            // over 15 ms.  NOT clamped to [0,1]: c is a marker, and clipping it
            // would inject exactly the non-material change we are avoiding.
            const Set::Scalar c_face_xlo = (flux_xlo.u_interface > 0.0) ? cfun(i - 1, j, k) : cfun(i,     j, k);
            const Set::Scalar c_face_xhi = (flux_xhi.u_interface > 0.0) ? cfun(i,     j, k) : cfun(i + 1, j, k);
            const Set::Scalar c_face_ylo = (flux_ylo.u_interface > 0.0) ? cfun(i, j - 1, k) : cfun(i, j,     k);
            const Set::Scalar c_face_yhi = (flux_yhi.u_interface > 0.0) ? cfun(i, j,     k) : cfun(i, j + 1, k);
#if AMREX_SPACEDIM == 3
            const Set::Scalar c_face_zlo = (flux_zlo.u_interface > 0.0) ? cfun(i, j, k - 1) : cfun(i, j, k);
            const Set::Scalar c_face_zhi = (flux_zhi.u_interface > 0.0) ? cfun(i, j, k)     : cfun(i, j, k + 1);
            Set::Scalar a_face_zlo = (flux_zlo.u_interface > 0.0) ? eta(i, j, k - 1) : eta(i, j, k);
            Set::Scalar a_face_zhi = (flux_zhi.u_interface > 0.0) ? eta(i, j, k)     : eta(i, j, k + 1);
            if (eta_consistent_advect == 1)
            {
                a_face_zlo = flux_zlo.alpha_face;
                a_face_zhi = flux_zhi.alpha_face;
            }
            if (sharp_wall)   // see the x/y note above
            {
                if (phisol(i, j, k - 1) < 0.5) a_face_zlo = flux_zlo.alpha_face;
                if (phisol(i, j, k + 1) < 0.5) a_face_zhi = flux_zhi.alpha_face;
            }
            a_face_zlo = std::min(std::max(a_face_zlo, 0.0), 1.0);
            a_face_zhi = std::min(std::max(a_face_zhi, 0.0), 1.0);
#endif

            // -----------------------------------------------------------
            // Store hi-face Riemann fluxes at cell centers for reflux.
            // Cell (i,j) -> face (i+1/2,j) flux for d=0, (i,j+1/2) for d=1.
            // HEAD's Riemann solver already returns per-phase mass and
            // per-phase energy components, so we feed the FluxRegister
            // directly without an extra eta-weighted split (Matt Q's
            // a5783ab32 split is built into our flux struct).
            //
            // At box-lo boundaries also write the lo-face flux into the
            // (i-1) / (j-1) ghost cell -- FillBoundary in Advance() will
            // overwrite this from a neighbor where one exists, but at
            // coarse-fine boundaries no neighbor exists and these ghost
            // values are what FineAdd picks up (Matt Q 8a1633977 fix).
            // -----------------------------------------------------------
            // ff_mom_d stores, in component c, the flux of mixture-momentum
            // component c through the d-face: component d gets momentum_normal,
            // (d+1)%SD gets momentum_tangent, (d+2)%SD gets momentum_tangent2 --
            // the SAME cyclic mapping as make_state / the M_flux scatter, so the
            // reflux is self-consistent in any dimension.  In 2D this reduces to
            // the old explicit (x: 0=n,1=t ; y: 0=t,1=n) layout.
            if (have_cc_fluxes)
            {
                // x-direction hi-face (normal->0, tangent->1, tangent2->2)
                ff_mass_x  (i, j, k, 0) += rw * flux_xhi.mass0;
                ff_mass_x  (i, j, k, 1) += rw * flux_xhi.mass1;
                ff_mom_x   (i, j, k, 0) += rw * flux_xhi.momentum_normal;
                ff_mom_x   (i, j, k, 1) += rw * flux_xhi.momentum_tangent;
                ff_ene_x   (i, j, k)    += rw * flux_xhi.energy_total;
                ff_ene_k_x (i, j, k, 0) += rw * flux_xhi.energy0;
                ff_ene_k_x (i, j, k, 1) += rw * flux_xhi.energy1;

                // y-direction hi-face (normal->1, tangent->2%SD, tangent2->0)
                ff_mass_y  (i, j, k, 0) += rw * flux_yhi.mass0;
                ff_mass_y  (i, j, k, 1) += rw * flux_yhi.mass1;
                ff_mom_y   (i, j, k, 1)                  += rw * flux_yhi.momentum_normal;
                ff_mom_y   (i, j, k, 2 % AMREX_SPACEDIM) += rw * flux_yhi.momentum_tangent;
                ff_ene_y   (i, j, k)    += rw * flux_yhi.energy_total;
                ff_ene_k_y (i, j, k, 0) += rw * flux_yhi.energy0;
                ff_ene_k_y (i, j, k, 1) += rw * flux_yhi.energy1;
#if AMREX_SPACEDIM == 3
                ff_mom_x   (i, j, k, 2) += rw * flux_xhi.momentum_tangent2;  // x tangent2 -> 2
                ff_mom_y   (i, j, k, 0) += rw * flux_yhi.momentum_tangent2;  // y tangent2 -> 0

                // z-direction hi-face (normal->2, tangent->0, tangent2->1)
                ff_mass_z  (i, j, k, 0) += rw * flux_zhi.mass0;
                ff_mass_z  (i, j, k, 1) += rw * flux_zhi.mass1;
                ff_mom_z   (i, j, k, 2) += rw * flux_zhi.momentum_normal;
                ff_mom_z   (i, j, k, 0) += rw * flux_zhi.momentum_tangent;
                ff_mom_z   (i, j, k, 1) += rw * flux_zhi.momentum_tangent2;
                ff_ene_z   (i, j, k)    += rw * flux_zhi.energy_total;
                ff_ene_k_z (i, j, k, 0) += rw * flux_zhi.energy0;
                ff_ene_k_z (i, j, k, 1) += rw * flux_zhi.energy1;
#endif

                if (i == vbx_lo.x) {
                    ff_mass_x  (i - 1, j, k, 0) += rw * flux_xlo.mass0;
                    ff_mass_x  (i - 1, j, k, 1) += rw * flux_xlo.mass1;
                    ff_mom_x   (i - 1, j, k, 0) += rw * flux_xlo.momentum_normal;
                    ff_mom_x   (i - 1, j, k, 1) += rw * flux_xlo.momentum_tangent;
                    ff_ene_x   (i - 1, j, k)    += rw * flux_xlo.energy_total;
                    ff_ene_k_x (i - 1, j, k, 0) += rw * flux_xlo.energy0;
                    ff_ene_k_x (i - 1, j, k, 1) += rw * flux_xlo.energy1;
#if AMREX_SPACEDIM == 3
                    ff_mom_x   (i - 1, j, k, 2) += rw * flux_xlo.momentum_tangent2;
#endif
                }
                if (j == vbx_lo.y) {
                    ff_mass_y  (i, j - 1, k, 0) += rw * flux_ylo.mass0;
                    ff_mass_y  (i, j - 1, k, 1) += rw * flux_ylo.mass1;
                    ff_mom_y   (i, j - 1, k, 1)                  += rw * flux_ylo.momentum_normal;
                    ff_mom_y   (i, j - 1, k, 2 % AMREX_SPACEDIM) += rw * flux_ylo.momentum_tangent;
                    ff_ene_y   (i, j - 1, k)    += rw * flux_ylo.energy_total;
                    ff_ene_k_y (i, j - 1, k, 0) += rw * flux_ylo.energy0;
                    ff_ene_k_y (i, j - 1, k, 1) += rw * flux_ylo.energy1;
#if AMREX_SPACEDIM == 3
                    ff_mom_y   (i, j - 1, k, 0) += rw * flux_ylo.momentum_tangent2;
#endif
                }
#if AMREX_SPACEDIM == 3
                if (k == vbx_lo.z) {
                    ff_mass_z  (i, j, k - 1, 0) += rw * flux_zlo.mass0;
                    ff_mass_z  (i, j, k - 1, 1) += rw * flux_zlo.mass1;
                    ff_mom_z   (i, j, k - 1, 2) += rw * flux_zlo.momentum_normal;
                    ff_mom_z   (i, j, k - 1, 0) += rw * flux_zlo.momentum_tangent;
                    ff_mom_z   (i, j, k - 1, 1) += rw * flux_zlo.momentum_tangent2;
                    ff_ene_z   (i, j, k - 1)    += rw * flux_zlo.energy_total;
                    ff_ene_k_z (i, j, k - 1, 0) += rw * flux_zlo.energy0;
                    ff_ene_k_z (i, j, k - 1, 1) += rw * flux_zlo.energy1;
                }
#endif
            }

            // div(alpha u) and div(u) (from S_M = flux.u_interface).
            const Set::Scalar div_uA_x = (flux_xhi.u_interface * a_face_xhi
                                        - flux_xlo.u_interface * a_face_xlo) / DX[0];
            const Set::Scalar div_uA_y = (flux_yhi.u_interface * a_face_yhi
                                        - flux_ylo.u_interface * a_face_ylo) / DX[1];
            const Set::Scalar div_u_x  = (flux_xhi.u_interface - flux_xlo.u_interface) / DX[0];
            const Set::Scalar div_u_y  = (flux_yhi.u_interface - flux_ylo.u_interface) / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar div_uA_z = (flux_zhi.u_interface * a_face_zhi
                                        - flux_zlo.u_interface * a_face_zlo) / DX[2];
            const Set::Scalar div_u_z  = (flux_zhi.u_interface - flux_zlo.u_interface) / DX[2];
#endif
            const Set::Scalar div_u    = AMREX_D_TERM(div_u_x, + div_u_y, + div_u_z);

            // ------------------------------------------------------------
            // Volume fraction (alpha_1) row -- Saurel 2009 line 836:
            //   d(alpha)/dt + u . grad(alpha) = 0
            //   discretized as -div(alpha u) + alpha_C div(u).
            // The 6-eq model REPLACES the 5-eq Kapila K-source with the
            // stiff relaxation source (handled in the post-stage hook).
            // ------------------------------------------------------------
            Set::Scalar eta_advect = -(AMREX_D_TERM(div_uA_x, + div_uA_y, + div_uA_z)) + eta(i, j, k) * div_u;
            eta_rhs(i, j, k) = eta_advect + eta_dot_Vap;

            // ------------------------------------------------------------
            // SHELL row -- the AREAL density Gamma is the primary, transported by
            // the exact surfactant-on-a-deforming-interface law (Stone 1990):
            //
            //     D(Gamma)/Dt = -Gamma (div_s u)      div_s u = div(u) - n.grad(u).n
            //  => Gamma_rhs   = -div(Gamma u) + Gamma (n.grad(u).n)
            //
            // WHY THIS FORM.  The previous version stored the volumetric density
            // c = Gamma|grad eta| and recovered Gamma = c/|grad eta|.  That
            // division is ill-conditioned exactly where |grad eta| -> 0, so any
            // residual mismatch between how c and eta are discretely transported
            // showed up first in the band tails: measured Gamma -> 1.79 mean /
            // 11.9 max in the gas-side tail at 5 ms while the band core was still
            // 0.95-1.01, after which the piecewise sigma(Gamma) rectified that
            // noise into force and it leaked inward.  Damping or gating the tail
            // only masks it.
            //
            // Here NO |grad eta| appears at all.  The only coupling to eta is the
            // UNIT normal n_hat, which stays well-conditioned right through the
            // tails, and Gamma uses the same face-upwind values and the same
            // interface velocity S_M as the eta row -- so the two are transported
            // by the identical operator and cannot drift apart.  It also makes
            // Gamma immune to NON-MATERIAL changes in eta (pressure relaxation,
            // clamping, ghost reconstruction) for free, since Gamma is never
            // reconstructed from eta.
            //
            // Trade-off: this is the non-conservative form, so total shell mass is
            // conserved only to truncation rather than to machine precision.  The
            // conservative form conserved it exactly and still failed, because a
            // well-conditioned Gamma -- not exact conservation -- is what the
            // sigma(Gamma) map actually needs.
            // Gamma is reconstructed at faces with the SAME limiter eta uses for
            // its Riemann reconstruction (Limiter.type = godunov|minmod|vanleer|
            // weno3|weno5).  Every limiter reconstructs componentwise, so we route
            // Gamma through the existing Primitive API in the `alpha` slot -- the
            // other fields are inert.  With Limiter=Godunov this collapses exactly
            // to the first-order upwind difference, so the scheme is unchanged at
            // the default and only improves when a real limiter is selected.
            //
            // The VELOCITY stays cell-centered, matching gradu in the stretch term
            // below.  An earlier version paired the face interface velocity S_M
            // with cell-centered gradu; the O(dx) disagreement biased Gamma upward
            // ~2%/ms (sigma decayed to 0.54 at 10 ms while the bubble expanded --
            // the shell read as COMPRESSING while its surface grew).  Only the
            // Gamma INTERPOLATION is raised in order here, never the velocity.
            if (!shell_row_on)
            {
                shell_rhs(i, j, k) = 0.0;   // frozen at init value; nothing reads it
            }
            else
            {
            //   D(Gamma)/Dt = -Gamma (div u - n.grad(u).n)
            //
            // GAMMA ROW.  Advected with EXACTLY the operator the alpha row
            // uses -- limiter-reconstructed face value
            // upwinded on the RIEMANN CONTACT SPEED u* (flux.u_interface), in
            // flux-divergence form  -div(u* G_face) + Gamma div(u*).
            //
            // Gamma was already limiter-reconstructed, but it upwound on the
            // CELL-CENTRED velocity u and differenced a gradient, while alpha
            // uses u* at the faces.  Two rows that must stay locked together
            // then carry different truncation error and separate -- the exact
            // failure the colour function c hit (see the note by a_face above:
            // "Advecting c with the cell-centred u instead ... gives it a
            // different truncation error from eta, and the two separate --
            // measured at 2.28% in R over 15 ms").  c was fixed; Gamma was not.
            //
            // Measured cost of the mismatch on the coated Sch20 collapse, where
            // the exact solution is Gamma = (R0/R)^2:
            //     R/R0 0.939 -> Gamma  -1.5 %
            //     R/R0 0.707 -> Gamma  -3.1 %
            //     R/R0 0.430 -> Gamma -23.7 %
            //     R/R0 0.270 -> Gamma -26.7 %
            //     R/R0 0.360 (rebound) -> -18.0 %   <-- does NOT recover
            // Since sigma(Gamma) = chi (Gamma_buck/Gamma - 1), a Gamma 27% low
            // makes sigma too high AND crosses the buckling threshold at the
            // wrong radius, so the whole Marmottant law is fed a biased input.
            // [OPT] Face values come from the per-box shell face cache (comp 0
            // = qL, right edge of the lo cell; comp 1 = qR, left edge of the hi
            // cell) -- the identical reconstruction, done once per face.
            const Set::Scalar aL = Sx(i + 1, j, k, 0), aR = Sx(i + 1, j, k, 1);   // face i+1/2
            const Set::Scalar cL = Sx(i,     j, k, 0), cR = Sx(i,     j, k, 1);   // face i-1/2
            const Set::Scalar bL = Sy(i, j + 1, k, 0), bR = Sy(i, j + 1, k, 1);   // face j+1/2
            const Set::Scalar dL = Sy(i, j,     k, 0), dR = Sy(i, j,     k, 1);   // face j-1/2
#if AMREX_SPACEDIM == 3
            const Set::Scalar eL = Sz(i, j, k + 1, 0), eR = Sz(i, j, k + 1, 1);   // face k+1/2
            const Set::Scalar fL = Sz(i, j, k,     0), fR = Sz(i, j, k,     1);   // face k-1/2
#endif
            // G_face upwinded on u* at each face, mirroring a_face exactly.
            // REVERTED to the cell-centred velocity (see the note above).  Pairing
            // the face interface velocity S_M with the cell-centred `gradu` used
            // by the stretch term below is an O(dx) inconsistency that was
            // already tried and rejected: it biased Gamma upward ~2%/ms, so the
            // shell read as COMPRESSING while its surface grew.  Only the Gamma
            // INTERPOLATION is limiter-raised here; the velocity stays cell
            // centred so advection and stretch share one velocity.
            const Set::Scalar ux = u(0), uy = u(1);
            const Set::Scalar dGx = (ux > 0.0) ? (aL - cL) / DX[0] : (aR - cR) / DX[0];
            const Set::Scalar dGy = (uy > 0.0) ? (bL - dL) / DX[1] : (bR - dR) / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar uz = u(2);
            const Set::Scalar dGz = (uz > 0.0) ? (eL - fL) / DX[2] : (eR - fR) / DX[2];
#endif
            const Set::Scalar u_dot_gradG = AMREX_D_TERM(ux * dGx, + uy * dGy, + uz * dGz);
            // Surface divergence.  In the bulk grad_eta -> 0 so n_hat -> 0 and this
            // reduces to div(u); Gamma there is unused.
            // NOTE: differencing the ratio u = M/rho directly, instead of the
            // product-rule form (grad M - u grad rho)/rho that `gradu` uses, was
            // tried on the suspicion that the 10:1 density swing across the band
            // spoils the reconstruction.  It changed nothing measurable (Gamma
            // response 82.4% / 83.1% vs 83.0%), so gradu is not the issue and the
            // simpler shared expression is kept.
            // SURFACE DIVERGENCE, gate-free.
            //
            // div_s u = P : grad u  with  P = I - n n.  That identity assumes
            // |n| = 1.  Here n_hat = grad_eta/(|grad_eta| + small), so |n_hat|
            // is ~1 on the band and -> 0 in the bulk.  Using the consistent
            // projector
            //        P = |n|^2 I - n n        (tr P = 2|n|^2)
            // reproduces div_s u EXACTLY where |n| = 1, and makes the whole
            // source vanish smoothly where there is no interface -- instead of
            // degenerating to div(u), which is what forced the old DX-scaled
            // freeze.  No threshold, no new constant: |n_hat|^2 is built from
            // quantities already in scope, and it is mesh-independent.
            const Set::Scalar nn = n_hat.squaredNorm();
            // LEVEL-SET SHELL (option 4, 2026-11-10): each band cell carries the shell of its OWN
            // eta level set.  Here the fluid part, -u.grad(Gamma) - Gamma div_s u; the difference
            // between the level set's normal speed V_n and the fluid's u_n is applied once per step
            // after the relaxation (ShellLevelSetSlip), from the actual change of eta.
            (void)eps_rl; (void)cut_rl;
            const Set::Scalar div_s_u = nn * gradu.trace() - n_hat.dot(gradu * n_hat);
            if (slip7 && slip15)
            {
                // shell.slip = 15: m moves with the eta level sets, applied once per step after the
                // relaxation (ShellRemap15).  No fluid flux here.
                shell_rhs(i, j, k) = 0.0;
            }
            else if (slip7)
            {
                // shell.slip = 7: m = Gamma |grad psi| on material level sets, dm/dt = -div(m u*),
                // limiter-reconstructed m upwinded on the Riemann contact speed u* (as the eta row).
                auto upw = [](Set::Scalar us, Set::Scalar qL, Set::Scalar qR) { return us * ((us > 0.0) ? qL : qR); };
                shell_rhs(i, j, k) = -(AMREX_D_TERM(
                      (upw(flux_xhi.u_interface, aL, aR) - upw(flux_xlo.u_interface, cL, cR)) / DX[0],
                    + (upw(flux_yhi.u_interface, bL, bR) - upw(flux_ylo.u_interface, dL, dR)) / DX[1],
                    + (upw(flux_zhi.u_interface, eL, eR) - upw(flux_zlo.u_interface, fL, fR)) / DX[2]));
            }
            else
            shell_rhs(i, j, k) = -u_dot_gradG - shell(i, j, k) * div_s_u;
            }   // end shell_row_on / band gate

            // VAPOUR (phase change): conserved partial density of vapour in the gas phase,
            // flux = gas-phase mass flux x upwind vapour mass fraction.
            if (pc_on)
            {
                auto Yv = [&](int ii, int jj, int kk) {
                    const Set::Scalar mg = (pc_gas == 0) ? rho_eta0(ii, jj, kk) : rho_eta1(ii, jj, kk);
                    return (mg > 1.0e-300) ? std::min(std::max(vap_arr(ii, jj, kk) / mg, 0.0), 1.0) : 0.0;
                };
                auto gflux = [&](const FluxT &f) { return (pc_gas == 0) ? f.mass0 : f.mass1; };
                Set::Scalar r_v = 0.0;
                { const Set::Scalar fl_ = gflux(flux_xlo), fh_ = gflux(flux_xhi);
                  r_v += (fl_ * (fl_ > 0.0 ? Yv(i - 1, j, k) : Yv(i, j, k)) - fh_ * (fh_ > 0.0 ? Yv(i, j, k) : Yv(i + 1, j, k))) / DX[0]; }
                { const Set::Scalar fl_ = gflux(flux_ylo), fh_ = gflux(flux_yhi);
                  r_v += (fl_ * (fl_ > 0.0 ? Yv(i, j - 1, k) : Yv(i, j, k)) - fh_ * (fh_ > 0.0 ? Yv(i, j, k) : Yv(i, j + 1, k))) / DX[1]; }
#if AMREX_SPACEDIM == 3
                { const Set::Scalar fl_ = gflux(flux_zlo), fh_ = gflux(flux_zhi);
                  r_v += (fl_ * (fl_ > 0.0 ? Yv(i, j, k - 1) : Yv(i, j, k)) - fh_ * (fh_ > 0.0 ? Yv(i, j, k) : Yv(i, j, k + 1))) / DX[2]; }
#endif
                vap_rhs(i, j, k) = r_v;
            }

            // SPECIES (species.on): partial density of each species of each phase.  Flux = phase mass flux of the
            // Riemann solver x mass fraction of the UPWIND cell, small fractions first, the majority species of the
            // upwind cell takes the remainder (Species::FaceFluxes): the species fluxes add up to the phase flux exactly.
            if (sp_on)
            {
                for (int ph = 0; ph < 2; ++ph)
                {
                    const int n = (ph == 0) ? sp_n0 : sp_n1, c0 = (ph == 0) ? 0 : sp_n0;
                    if (n == 0) continue;
                    Set::Scalar r[Species::NMAX];
                    for (int s_ = 0; s_ < n; ++s_) r[s_] = 0.0;
                    auto face = [&](const FluxT &f, int il, int jl, int kl, int ih, int jh, int kh, Set::Scalar sign, Set::Scalar dx) {
                        const Set::Scalar F = (ph == 0) ? f.mass0 : f.mass1;
                        const int iu = (F > 0.0) ? il : ih, ju = (F > 0.0) ? jl : jh, ku = (F > 0.0) ? kl : kh;      // upwind cell
                        const Set::Scalar mk = (ph == 0) ? rho_eta0(iu, ju, ku) : rho_eta1(iu, ju, ku);
                        Set::Scalar Y[Species::NMAX], Ff[Species::NMAX];
                        for (int s_ = 0; s_ < n; ++s_) Y[s_] = (mk > 1.0e-300) ? std::max(sp_arr(iu, ju, ku, c0 + s_), 0.0) / mk : 0.0;
                        if (sp_order >= 2)      // limited linear face value from the upwind side (species.order = 2), THINC (3)
                        {
                            const int id = (F > 0.0) ? ih : il, jd = (F > 0.0) ? jh : jl, kd = (F > 0.0) ? kh : kl;   // downwind cell
                            const int iw = 2 * iu - id, jw = 2 * ju - jd, kw = 2 * ku - kd;                           // one further upwind
                            const Set::Scalar md = (ph == 0) ? rho_eta0(id, jd, kd) : rho_eta1(id, jd, kd);
                            const Set::Scalar mw = (ph == 0) ? rho_eta0(iw, jw, kw) : rho_eta1(iw, jw, kw);
                            if (mk > 1.0e-300 && md > 1.0e-300 && mw > 1.0e-300)
                                for (int s_ = 0; s_ < n; ++s_)
                                {
                                    const Set::Scalar a = Y[s_] - std::max(sp_arr(iw, jw, kw, c0 + s_), 0.0) / mw;
                                    const Set::Scalar b = std::max(sp_arr(id, jd, kd, c0 + s_), 0.0) / md - Y[s_];
                                    if (sp_order == 3 && a * b > 0.0 && Y[s_] > sp_te && Y[s_] < 1.0 - sp_te)
                                    {
                                        // THINC (Wallis 2021, App. A): tanh profile through the three cells, value on the
                                        // downwind face of the upwind cell.  qm / qp = mass fraction one cell upwind / downwind.
                                        auto Yc = [&](int ii, int jj, int kk) {
                                            const Set::Scalar mm = (ph == 0) ? rho_eta0(ii, jj, kk) : rho_eta1(ii, jj, kk);
                                            return (mm > 1.0e-300) ? std::max(sp_arr(ii, jj, kk, c0 + s_), 0.0) / mm : Y[s_]; };
                                        Set::Scalar g2 = 0.0, gd = 0.0;
                                        { const Set::Scalar gx = Yc(iu + 1, ju, ku) - Yc(iu - 1, ju, ku); g2 += gx * gx; if (id != iu) gd = gx; }
                                        { const Set::Scalar gy = Yc(iu, ju + 1, ku) - Yc(iu, ju - 1, ku); g2 += gy * gy; if (jd != ju) gd = gy; }
#if AMREX_SPACEDIM == 3
                                        { const Set::Scalar gz = Yc(iu, ju, ku + 1) - Yc(iu, ju, ku - 1); g2 += gz * gz; if (kd != ku) gd = gz; }
#endif
                                        const Set::Scalar beta = sp_tb * ((g2 > 0.0) ? std::abs(gd) / std::sqrt(g2) : 1.0) + 0.01;
                                        const Set::Scalar qm = Y[s_] - a, qp = Y[s_] + b, qmin = std::min(qm, qp), dq = std::abs(qp - qm);
                                        const Set::Scalar th = (qp > qm) ? 1.0 : -1.0, C = (Y[s_] - qmin + 1.0e-20) / (dq + 1.0e-20);
                                        const Set::Scalar Bq = std::exp(th * beta * (2.0 * C - 1.0)), tb = std::tanh(beta);
                                        const Set::Scalar A = (Bq / std::cosh(beta) - 1.0) / tb;
                                        Y[s_] = qmin + 0.5 * dq * (1.0 + th * (tb + A) / (1.0 + A * tb));
                                    }
                                    else
                                    if (a * b > 0.0) Y[s_] += a * b / (a + b);          // half the van Leer slope 2ab/(a+b)
                                }
                        }
                        Species::FaceFluxes(F, Y, n, Ff);
                        for (int s_ = 0; s_ < n; ++s_) r[s_] += sign * Ff[s_] / dx;
                        if (sp_cc)      // reflux capture: the flux of the face is stored in its LOW cell (il, jl, kl)
                        {
                            const int dir = (ih != il) ? 0 : ((jh != jl) ? 1 : 2);
                            const int ic = (dir == 0) ? i : ((dir == 1) ? j : k);
                            const int lo_d = (dir == 0) ? vbx_lo.x : ((dir == 1) ? vbx_lo.y : vbx_lo.z);
                            if (sign < 0.0 || ic == lo_d)
                            {
                                const amrex::Array4<Set::Scalar> &fs = (dir == 0) ? ff_sp0 : ((dir == 1) ? ff_sp1 : ff_sp2);
                                for (int s_ = 0; s_ < n; ++s_) fs(il, jl, kl, c0 + s_) += rw * Ff[s_];
                            }
                        }
                    };
                    face(flux_xlo, i - 1, j, k, i, j, k, +1.0, DX[0]); face(flux_xhi, i, j, k, i + 1, j, k, -1.0, DX[0]);
                    face(flux_ylo, i, j - 1, k, i, j, k, +1.0, DX[1]); face(flux_yhi, i, j, k, i, j + 1, k, -1.0, DX[1]);
#if AMREX_SPACEDIM == 3
                    face(flux_zlo, i, j, k - 1, i, j, k, +1.0, DX[2]); face(flux_zhi, i, j, k, i, j, k + 1, -1.0, DX[2]);
#endif
                    for (int s_ = 0; s_ < n; ++s_) sp_rhs(i, j, k, c0 + s_) = r[s_];
                }
                // evaporation into named species (vaporization.vapor_species / liquid_species, see Parse)
                if (vspg >= 0) sp_rhs(i, j, k, vspg) += m_dot_Vap;
                if (vspl >= 0) sp_rhs(i, j, k, vspl) -= m_dot_Vap;
            }
            if (elastic_on)     // cobasis row + stress diagnostics (Hydro2_Solid.H)
            {
                const FluxT *const Fl[AMREX_SPACEDIM] = { AMREX_D_DECL(&flux_xlo, &flux_ylo, &flux_zlo) };
                const FluxT *const Fh[AMREX_SPACEDIM] = { AMREX_D_DECL(&flux_xhi, &flux_yhi, &flux_zhi) };
                elastic.cobasisRHS(eb_rhs, el_diag, eb_arr, eta, rho_eta0, rho_eta1, Fl, Fh, DX, i, j, k, sp_arr);
            }

            // ------------------------------------------------------------
            // COLOUR FUNCTION  dc/dt + u.grad(c) = 0   (Schmidmayer 2017 eq. 3)
            // ------------------------------------------------------------
            // Discretised EXACTLY as the alpha_1 row above: -div(c u*) + c div(u*),
            // with u* the HLLC contact velocity.  Same flux, same star velocity,
            // same truncation error -- so c and eta cannot drift apart for
            // numerical reasons, only through the physical difference between
            // them (alpha_1 carries the relaxation source mu(P1-P2); c does not).
            if (!cfun_row_on)
            {
                cfun_rhs(i, j, k) = 0.0;    // colour function unused without the closure
            }
            else
            {
            const Set::Scalar div_uC_x = (flux_xhi.u_interface * c_face_xhi
                                        - flux_xlo.u_interface * c_face_xlo) / DX[0];
            const Set::Scalar div_uC_y = (flux_yhi.u_interface * c_face_yhi
                                        - flux_ylo.u_interface * c_face_ylo) / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar div_uC_z = (flux_zhi.u_interface * c_face_zhi
                                        - flux_zlo.u_interface * c_face_zlo) / DX[2];
#endif
            cfun_rhs(i, j, k) = -(AMREX_D_TERM(div_uC_x, + div_uC_y, + div_uC_z))
                              + cfun(i, j, k) * div_u;
            }

            // NOTE: a surface-diffusion term  + D_s grad_s^2(Gamma)  (Stone 1990)
            // was implemented and tested here, then removed.  It does not fix the
            // Gamma feedback instability.  Measured at eps = 4 finest cells / 4 AMR
            // levels, sigma spread at t = 0.5 ms:
            //   tangential grad_s^2:  D_s = 0 / 1e-5 / 1e-4 / 1e-3 -> 226 / 225 / 204 / 192 %
            //   full Laplacian:       D_s = 1e-4 / 1e-3 / 1e-2     -> 194 / 168 / 73 %
            // Only D_s = 1e-2 bites, and that sits AT the explicit diffusion limit
            // (dt_max = dx^2/4D_s = 9.5e-7 vs the subcycled fine dt of 6.9e-7) while
            // still growing exponentially -- delayed, not stabilized, at a diffusivity
            // ~10 orders above real lipid values.  The unstable modes live in the
            // band-NORMAL degrees of freedom of Gamma, which are unphysical for a
            // surface field; damping cannot fix a mode whose root cause is that the
            // degree of freedom should not exist.


            // ------------------------------------------------------------
            // Phase-mass rows (pure conservation, no source from h):
            //   d(alpha rho)_k / dt + div((alpha rho)_k u) = 0
            // ------------------------------------------------------------
            const Set::Scalar rho_eta0_flux = AMREX_D_TERM(
                                              (flux_xlo.mass0 - flux_xhi.mass0) / DX[0],
                                            + (flux_ylo.mass0 - flux_yhi.mass0) / DX[1],
                                            + (flux_zlo.mass0 - flux_zhi.mass0) / DX[2]);
            const Set::Scalar rho_eta1_flux = AMREX_D_TERM(
                                              (flux_xlo.mass1 - flux_xhi.mass1) / DX[0],
                                            + (flux_ylo.mass1 - flux_yhi.mass1) / DX[1],
                                            + (flux_zlo.mass1 - flux_zhi.mass1) / DX[2]);

            rho_eta0_rhs(i, j, k) = rho_eta0_flux + Source(i, j, k, 0) * (eta(i, j, k))         + m_dot_Vap;
            rho_eta1_rhs(i, j, k) = rho_eta1_flux + Source(i, j, k, 0) * (1.0 - eta(i, j, k))   - m_dot_Vap;
            // [DIAG RHSDECOMP] stash boundary-face mass1 fluxes + source terms
            if (rhsp_fxlo.contains(i, j, k))
            {
                rhsp_fxlo(i, j, k) = flux_xlo.mass1;
                rhsp_fxhi(i, j, k) = flux_xhi.mass1;
                rhsp_fylo(i, j, k) = flux_ylo.mass1;
                rhsp_fyhi(i, j, k) = flux_yhi.mass1;
                rhsp_src (i, j, k) = Source(i, j, k, 0) * (1.0 - eta(i, j, k)) - m_dot_Vap;
            }

            // Diagnostic mass flux (kept for plotfile compatibility):
            rho_flux(i, j, k) = rho_eta0_flux + rho_eta1_flux;

            // ------------------------------------------------------------
            // Mixture momentum (pure conservation, with body sources):
            // ------------------------------------------------------------
            // In RHS we accumulate as (F_lo - F_hi)/dx.  On each face with normal
            // `dir`, momentum_normal feeds M-component `dir` and the tangential
            // momentum fluxes feed the cyclically-next components (dir+1, dir+2) --
            // mirroring make_state's normal/tangent mapping so the round-trip is
            // self-consistent.  The `% AMREX_SPACEDIM` keeps the 2D path identical
            // (x-tangent->1, y-tangent->0) while the 3D path closes the cycle.
            for (int n = 0; n < AMREX_SPACEDIM; ++n) M_flux(i, j, k, n) = 0.0;
            // x-faces (normal->0, tangent->1)
            M_flux(i, j, k, 0)                  += (flux_xlo.momentum_normal  - flux_xhi.momentum_normal ) / DX[0];
            M_flux(i, j, k, 1 % AMREX_SPACEDIM) += (flux_xlo.momentum_tangent - flux_xhi.momentum_tangent) / DX[0];
            // y-faces (normal->1, tangent->2%SD)
            M_flux(i, j, k, 1)                  += (flux_ylo.momentum_normal  - flux_yhi.momentum_normal ) / DX[1];
            M_flux(i, j, k, 2 % AMREX_SPACEDIM) += (flux_ylo.momentum_tangent - flux_yhi.momentum_tangent) / DX[1];
#if AMREX_SPACEDIM == 3
            // second tangents of the x/y faces, plus the full z-faces.
            M_flux(i, j, k, 2) += (flux_xlo.momentum_tangent2 - flux_xhi.momentum_tangent2) / DX[0]; // x t2 -> 2
            M_flux(i, j, k, 0) += (flux_ylo.momentum_tangent2 - flux_yhi.momentum_tangent2) / DX[1]; // y t2 -> 0
            // z-faces (normal->2, tangent->0, tangent2->1)
            M_flux(i, j, k, 2) += (flux_zlo.momentum_normal   - flux_zhi.momentum_normal  ) / DX[2];
            M_flux(i, j, k, 0) += (flux_zlo.momentum_tangent  - flux_zhi.momentum_tangent ) / DX[2];
            M_flux(i, j, k, 1) += (flux_zlo.momentum_tangent2 - flux_zhi.momentum_tangent2) / DX[2];
#endif

            for (int n = 0; n < AMREX_SPACEDIM; ++n)
                M_rhs(i, j, k, n) = M_flux(i, j, k, n) + Source(i, j, k, 1 + n);

            // ------------------------------------------------------------
            // Redundant mixture total energy (pure conservation):
            //   d(rho E)/dt + div((rho E + p) u) = 0     (Sch20 eq. 16)
            // ------------------------------------------------------------
            E_flux(i, j, k) = AMREX_D_TERM(
                              (flux_xlo.energy_total - flux_xhi.energy_total) / DX[0],
                            + (flux_ylo.energy_total - flux_yhi.energy_total) / DX[1],
                            + (flux_zlo.energy_total - flux_zhi.energy_total) / DX[2]);

            // CAPILLARY ENERGY CLOSURE (Schmidmayer 2017): the energy flux
            // carries -Omega.u, i.e. the RHS gains +div(Omega.u).  G is built
            // cell-centred and averaged to faces, so the face value is shared
            // between neighbours and the sum telescopes -- discretely
            // conservative.  Note the (hi - lo) ordering here is the OPPOSITE
            // of the (lo - hi) convention above, because that term is -div(F)
            // while this one is +div(G).
            if (false && apply_surface_tension)   // superseded by L_cap
            {
                auto Gcap = [&](int ii, int jj, int kk, int d) -> Set::Scalar {
#if AMREX_SPACEDIM == 2
                    const Set::Scalar oxx = om_cap(ii,jj,kk,0), oyy = om_cap(ii,jj,kk,1);
                    const Set::Scalar oxy = om_cap(ii,jj,kk,2);
                    const Set::Scalar ux = v(ii,jj,kk,0), uy = v(ii,jj,kk,1);
                    return (d == 0) ? (oxx * ux + oxy * uy) : (oxy * ux + oyy * uy);
#else
                    const Set::Scalar oxx = om_cap(ii,jj,kk,0), oyy = om_cap(ii,jj,kk,1);
                    const Set::Scalar ozz = om_cap(ii,jj,kk,2), oxy = om_cap(ii,jj,kk,3);
                    const Set::Scalar oxz = om_cap(ii,jj,kk,4), oyz = om_cap(ii,jj,kk,5);
                    const Set::Scalar ux = v(ii,jj,kk,0), uy = v(ii,jj,kk,1), uz = v(ii,jj,kk,2);
                    if (d == 0) return oxx * ux + oxy * uy + oxz * uz;
                    if (d == 1) return oxy * ux + oyy * uy + oyz * uz;
                    return              oxz * ux + oyz * uy + ozz * uz;
#endif
                };
                Set::Scalar cap_div = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    const Set::Scalar g_hi = 0.5 * (Gcap(i, j, k, d)
                                                  + Gcap(i + di, j + dj, k + dk, d));
                    const Set::Scalar g_lo = 0.5 * (Gcap(i - di, j - dj, k - dk, d)
                                                  + Gcap(i, j, k, d));
                    cap_div += (g_hi - g_lo) / DX[d];
                }
                E_flux(i, j, k) += cap_div;
            }
            E_rhs(i, j, k)  = E_flux(i, j, k) + Source(i, j, k, AMREX_SPACEDIM + 1);

            // ------------------------------------------------------------
            // Per-phase internal energies (Sch20 eq. 13 last two rows /
            // Sau09 line 852):
            //   d(alpha rho e)_k/dt + div((alpha rho e)_k u)
            //                       + alpha_k_C p_k_C div(u) = 0
            // (The +/- mu p_I (p_1-p_2) relaxation source is deferred to
            //  the post-stage hook, in the stiff-relaxation limit.)
            // ------------------------------------------------------------
            const Set::Scalar E0_flux_div = AMREX_D_TERM(
                                            (flux_xlo.energy0 - flux_xhi.energy0) / DX[0],
                                          + (flux_ylo.energy0 - flux_yhi.energy0) / DX[1],
                                          + (flux_zlo.energy0 - flux_zhi.energy0) / DX[2]);
            const Set::Scalar E1_flux_div = AMREX_D_TERM(
                                            (flux_xlo.energy1 - flux_xhi.energy1) / DX[0],
                                          + (flux_ylo.energy1 - flux_yhi.energy1) / DX[1],
                                          + (flux_zlo.energy1 - flux_zhi.energy1) / DX[2]);

            E0_rhs(i, j, k) = (E0_flux_div - a1_C * p0_C * div_u);
            E1_rhs(i, j, k) = (E1_flux_div - a2_C * p1_C * div_u);

            // thermal.conduction = 2: Fourier conduction inside each phase (see Parse).
            // Face flux k_k min(alpha_k) dT_k/dx is computed identically from both
            // sides, so the phase energy (and rho E) are conserved.  A phase below
            // thermal.alpha_min on either side of a face does not conduct there:
            // its temperature is ill-conditioned (trace phase).
            if (thermal_conduction == 2 && !thermal_subcycle)   // thermal.subcycle: SubcycleConduction does it
            {
                const Set::Scalar amin = thermal_alpha_min;
                auto Tph = [&](int ii, int jj, int kk, int ph) -> Set::Scalar {
                    const Set::Scalar a0 = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                    return ph == 0
                        ? (E0_arr(ii, jj, kk) - a0 * eos0.P0()) / (std::max(rho_eta0(ii, jj, kk), small) * eos0.Cv())
                        : (E1_arr(ii, jj, kk) - (1.0 - a0) * eos1.P0()) / (std::max(rho_eta1(ii, jj, kk), small) * eos1.Cv());
                };
                auto aph = [&](int ii, int jj, int kk, int ph) -> Set::Scalar {
                    const Set::Scalar a0 = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                    return ph == 0 ? a0 : 1.0 - a0;
                };
                Set::Scalar cond[2] = {0.0, 0.0};
                for (int ph = 0; ph < 2; ++ph)
                {
                    const Set::Scalar kk_ = ph ? thermal_k1 : thermal_k0;
                    if (aph(i, j, k, ph) <= amin) continue;
                    const Set::Scalar Tc = Tph(i, j, k, ph);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                        const Set::Scalar ahi = std::min(aph(i, j, k, ph), aph(i + di, j + dj, k + dk, ph));
                        const Set::Scalar alo = std::min(aph(i, j, k, ph), aph(i - di, j - dj, k - dk, ph));
                        const Set::Scalar Fhi = (ahi > amin) ? kk_ * ahi * (Tph(i + di, j + dj, k + dk, ph) - Tc) / DX[d] : 0.0;
                        const Set::Scalar Flo = (alo > amin) ? kk_ * alo * (Tc - Tph(i - di, j - dj, k - dk, ph)) / DX[d] : 0.0;
                        cond[ph] += (Fhi - Flo) / DX[d];
                    }
                }
                E0_rhs(i, j, k) += cond[0];
                E1_rhs(i, j, k) += cond[1];
                E_rhs(i, j, k)  += cond[0] + cond[1];
            }

            // Vaporization: the vaporized mass leaves the liquid with its own internal energy and
            // enters the gas with it; the latent heat is taken from the gas, Q_L moves from gas
            // to liquid (model 3).  E0 + E1 changes by -m_dot Lv, the same as the total-energy row.
            if (apply_vaporization == 1 && !vap_legacy_energy)
            {
                E1_rhs(i, j, k) += -m_dot_Vap * e_l_Vap + Q_L_Vap;
                E0_rhs(i, j, k) +=  m_dot_Vap * e_l_Vap - m_dot_Vap * Lv - Q_L_Vap;
            }

            // [2026-09-24] FOURIER HEAT CONDUCTION (thermal.conduction = 1).
            // div(k grad T) with face conductivity = mean of the two cell values;
            // each face term appears with opposite sign in its two cells, so the
            // level is conservative.  Faces to an embedded-solid cell carry no
            // heat (adiabatic wall); solid cells get nothing.  Deposited in rho E
            // and split into E0/E1 by volume fraction, like the shell dissipation.
            if (conduct_l && !(embedded.apply && phisol(i, j, k) < 0.5))
            {
                auto kcell = [&](int ii, int jj, int kk) -> Set::Scalar {
                    const Set::Scalar e = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                    if (kph0_l >= 0.0) return e * kph0_l + (1.0 - e) * kph1_l;
                    return (e * mu0 + (1.0 - e) * mu1) * Solver::EOS::EOS::MixedCp(e, eos0, eos1) / Pr_l;
                };
                const Set::Scalar kc = kcell(i, j, k);
                const Set::Scalar Tc = T(i, j, k);
                Set::Scalar qcond = 0.0, qwall_abl = 0.0, nwall_abl = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    for (int sg = -1; sg <= 1; sg += 2)
                    {
                        const int ii = i + (d == 0 ? sg : 0), jj = j + (d == 1 ? sg : 0), kk = k + (d == 2 ? sg : 0);
                        if (embedded.apply && phisol(ii, jj, kk) < 0.5)
                        {
                            if (Twall_l <= 0.0) continue;                              // adiabatic wall
                            // ISOTHERMAL WALL (solid.T_wall): surface at phi = 0.5, a fraction t of the
                            // spacing from this cell centre (t = 1/2 for a 0/1 phi); one-sided gradient.
                            const Set::Scalar phf = phisol(i, j, k), phs = phisol(ii, jj, kk);
                            const Set::Scalar t = std::min(std::max((phf - 0.5) / std::max(phf - phs, 1e-12), 0.0), 1.0);
                            // one-sided wall gradient.  When the surface is closer than half a cell to this
                            // cell centre the two-point quotient (T_w - T_c)/(t dx) is ill-conditioned (and a
                            // clamp on t under-predicts the flux: -3 % measured on the gap-conduction test), so
                            // take the gradient to the NEXT fluid cell away from the wall, a distance (1 + t) dx.
                            const int io = i - (d == 0 ? sg : 0), jo = j - (d == 1 ? sg : 0), ko = k - (d == 2 ? sg : 0);
                            const bool far_ok = (t < 0.5) && !(phisol(io, jo, ko) < 0.5);
                            Set::Scalar qf = far_ok ? kc * (Twall_l - T(io, jo, ko)) / ((1.0 + t) * DX[d])
                                                    : kc * (Twall_l - Tc) / (std::max(t, 0.25) * DX[d]);  // heat flux INTO this cell
                            Set::Scalar qs = 0.0;                                          // heat flux INTO the solid
                            if (abl_cond)
                            {
                                // [2026-10-06] IN-DEPTH CONDUCTION: surface temperature from flux continuity
                                // with the solid's one-sided gradient (same far-cell rule on the solid side),
                                // capped at the ablation temperature Twall_l.  Mirrored in AdvanceSolidTemperature.
                                const Set::Scalar Tg = far_ok ? T(io, jo, ko) : Tc;
                                const Set::Scalar hg = kc / ((far_ok ? 1.0 + t : std::max(t, 0.25)) * DX[d]);
                                const int i2 = ii + (d == 0 ? sg : 0), j2 = jj + (d == 1 ? sg : 0), k2 = kk + (d == 2 ? sg : 0);
                                const bool sfar = (1.0 - t < 0.5) && (phisol(i2, j2, k2) < 0.5);
                                const Set::Scalar Tsr = sfar ? abl_Ts(i2, j2, k2) : abl_Ts(ii, jj, kk);
                                const Set::Scalar hs = abl_ks / ((sfar ? 2.0 - t : std::max(1.0 - t, 0.25)) * DX[d]);
                                const Set::Scalar Tw = Ablation::WallTemperature(hg, Tg, hs, Tsr, Twall_l);
                                qf = hg * (Tw - Tg);
                                qs = hs * (Tw - Tsr);
                                abl_qs(i, j, k, 2 * d + (sg > 0 ? 1 : 0)) += rw * qs;   // stage-weighted: what the solid receives
                            }
                            qcond += qf / DX[d];
                            // NORMAL wall heat flux from the staircase faces (n = grad phi / |grad phi|): a face
                            // normal to d carries qf_d ~ q_n |n_d|, so over the wall faces this cell HAS the
                            // least-squares estimate is q_n = sum_d qf_d |n_d| / sum_d n_d^2.  [2026-10-05] the
                            // sum_d n_d^2 was missing: a cell with only one wall face on an inclined wall reported
                            // q_n n_d^2 (-13 % recession on the 26.6 deg inclined Stefan test, exact when aligned).
                            Set::Scalar g[3] = {0.0, 0.0, 0.0}, gn = 0.0;
                            for (int e = 0; e < AMREX_SPACEDIM; ++e)
                            {
                                const int a = (e == 0), b = (e == 1), c = (e == 2);
                                g[e] = (phisol(i + a, j + b, k + c) - phisol(i - a, j - b, k - c)) / (2.0 * DX[e]);
                                gn += g[e] * g[e];
                            }
                            const Set::Scalar nd = (gn > 0.0 ? std::abs(g[d]) / std::sqrt(gn) : 1.0);
                            qwall_abl += (-qf - qs) * nd;       // what is left to remove solid (= -qf without in-depth conduction)
                            nwall_abl += nd * nd;
                            continue;
                        }
                        qcond += 0.5 * (kc + kcell(ii, jj, kk)) * (T(ii, jj, kk) - Tc) / (DX[d] * DX[d]);
                    }
                // faces nearly parallel to the wall (sum n_d^2 < 0.1) give no usable estimate: leave the cell
                // to the neighbour average of RecedeAblatingSurface
                if (abl_on) abl_q(i, j, k) = (nwall_abl >= 0.1) ? qwall_abl / nwall_abl : 0.0;
                E_rhs(i, j, k)  += qcond;
                E0_rhs(i, j, k) += a1_C * qcond;
                E1_rhs(i, j, k) += a2_C * qcond;
            }


            // ------------------------------------------------------------
            // Artificial heat exchange (AHE) -- Schmidmayer 2020 eq. 13
            // r-source on per-phase internal energies:
            //     E0_rhs += -mu p_I (p_0 - p_1)
            //     E1_rhs += +mu p_I (p_0 - p_1)
            //
            // Sch20 does NOT specify a finite mu
            // (they take the stiff limit inside operator-split Newton); the
            // Saurel-Petitpas-Berry 2009 sec. 5.3 curve fit used here for
            // mu(v) was originally calibrated for the q*@u/@x form, NOT the
            // antisymmetric Sch20 form.  Magnitudes are a starting point
            // and must be re-calibrated for the target mixture.
            //
            // NOT the most conservative method in this codebase.  Pairwise
            // conservative across the two energy rows (sources sum to zero),
            // but per-phase energy is only conserved pairwise; total mixture
            // energy is maintained via the redundant rho-E reinit step.  If
            // the reinit step were disabled the per-phase energies would
            // drift -- the AHE source intentionally trades that property
            // for non-stiff pressure relaxation dynamics.
            // ------------------------------------------------------------
            if (apply_ahe)
            {
                // ----------------------------------------------------------
                // Local quantities shared by both AHE forms.
                // ----------------------------------------------------------
                const Set::Scalar rho0_pure = std::max(rho_eta0(i,j,k) / std::max(a1_C, small), small);
                const Set::Scalar rho1_pure = std::max(rho_eta1(i,j,k) / std::max(a2_C, small), small);

                const Set::Scalar c0_C = Solver::EOS::EOS::PhasicSoundSpeed(
                    rho0_pure, p0_C, PC(i, j, k, NPC), PC(i, j, k, NPC + 2), small);
                const Set::Scalar c1_C = Solver::EOS::EOS::PhasicSoundSpeed(
                    rho1_pure, p1_C, PC(i, j, k, NPC + 1), PC(i, j, k, NPC + 3), small);

                // mu (or q*-coefficient in Sau09 form) -- either constant or
                // Sau09 Sec.5.3 curve fit.
                Set::Scalar mu_ahe;
                if (ahe_use_const_mu)
                {
                    mu_ahe = ahe_mu_const;
                }
                else
                {
                    const Set::Scalar rho_mix_loc = std::max(rho_eta0(i,j,k) + rho_eta1(i,j,k), small);
                    const Set::Scalar v_mix       = 1.0 / rho_mix_loc;
                    mu_ahe = ahe_mu_scale * Solver::EOS::EOS::AHE_mu_Saurel(
                        v_mix, ahe_mu_a, ahe_mu_b, ahe_mu_c, ahe_v_min, ahe_v_max);
                }

                // Acoustic-CFL based source cap (used by both forms).
                const Set::Scalar E0_loc = std::max(E0_arr(i, j, k), small);
                const Set::Scalar E1_loc = std::max(E1_arr(i, j, k), small);
                const Set::Scalar E_min  = std::min(E0_loc, E1_loc);
                const Set::Scalar rho_loc_lim = std::max(rho_eta0(i,j,k) + rho_eta1(i,j,k), small);
                const Set::Scalar u_mag_lim = std::sqrt(AMREX_D_TERM(  M(i, j, k, 0) * M(i, j, k, 0),
                                                                    +  M(i, j, k, 1) * M(i, j, k, 1),
                                                                    +  M(i, j, k, 2) * M(i, j, k, 2))) / rho_loc_lim;
                const Set::Scalar c_local = std::max(c0_C, c1_C);
                const Set::Scalar dx_min  = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});
                const Set::Scalar src_cap = ahe_max_frac * E_min * (u_mag_lim + c_local) / std::max(dx_min, small);

                if (ahe_form == 0)
                {
                    // ======================================================
                    // Sch20 r-source form (antisymmetric +/- mu p_I Dp).
                    // ======================================================
                    const Set::Scalar Z0_C = rho0_pure * c0_C;
                    const Set::Scalar Z1_C = rho1_pure * c1_C;
                    const Set::Scalar p_I  = Solver::EOS::EOS::InterfacialPressureZ(
                        p0_C, p1_C, Z0_C, Z1_C, small);
                    const Set::Scalar delta_p = p0_C - p1_C;

                    const bool gate_on = (!ahe_compression_only) || (div_u < 0.0);
                    Set::Scalar ahe_src = gate_on ? (mu_ahe * p_I * delta_p) : 0.0;
                    ahe_src = std::max(std::min(ahe_src, src_cap), -src_cap);

                    E0_rhs(i, j, k) += -ahe_src;
                    E1_rhs(i, j, k) += +ahe_src;

                    if (ahe_apply_alpha_src && gate_on)
                    {
                        Set::Scalar eta_src = mu_ahe * delta_p;
                        const Set::Scalar eta_src_cap = ahe_max_frac * (u_mag_lim + c_local) / std::max(dx_min, small);
                        eta_src = std::max(std::min(eta_src, eta_src_cap), -eta_src_cap);
                        eta_rhs(i, j, k) += eta_src;
                    }
                }
                else
                {
                    // ======================================================
                    // Sau09 sec.5.3 q*@u/@x form (default).
                    //
                    // Sau09 eq. (above 5.3):
                    //   d(a_k rho_k e_k)/dt + d(a_k rho_k e_k u)/dx
                    //                       + (a_k p_k + q*) du/dx = 0,
                    //   q* = g(du/dx) * mu(v),
                    //   g  = 1 if du/dx < 0 else 0       (compression only).
                    //
                    // Both phases get the SAME -q* div(u) on the RHS (NOT
                    // antisymmetric).  q* acts as an extra effective pressure
                    // active only in the shock layer.  The Newton inside
                    // RelaxAndReinit sees Sau09-modified per-phase energies
                    // and produces a consistent shocked state -- this is
                    // the form used to obtain Sau09 Fig. 22 convergence.
                    //
                    // No alpha source in this form.
                    // ======================================================
                    const Set::Scalar g_ind = (div_u < 0.0) ? 1.0 : 0.0;
                    const Set::Scalar q_star = g_ind * mu_ahe;

                    Set::Scalar q_src = q_star * div_u;   // = -q*|div_u| <= 0 in compression
                    q_src = std::max(std::min(q_src, src_cap), -src_cap);

                    E0_rhs(i, j, k) += -q_src;
                    E1_rhs(i, j, k) += -q_src;
                }
            }

            // ============================================================
            // EMBEDDED SOLID BOUNDARY -- Yang(2023) diffuse-domain
            // no-penetration via MOMENTUM-ONLY Brinkman penalization.
            // ------------------------------------------------------------
            // MOMENTUM diffuse-interface no-penetration (Yang et al. 2023,
            // term phi0/kappa (u_S - u); orig. Angot 1999 / Liu & Vasilyev
            // 2007)
            // ============================================================
            if (embedded.apply && embedded.brinkman > 0.0 && !embedded.implicit && !embedded.wall_flux)
            {
                const Set::Scalar chi = 1.0 - phi_c;
                if (chi > 0.0)
                {
                    const Set::Scalar lam = embedded.brinkman * chi;
                    Set::Vector nrm = embedded.slip ? Numeric::Gradient(phisol, i, j, k, 0, DX)
                                                    : Set::Vector::Zero();
                    const Set::Scalar nmag = embedded.slip ? nrm.norm() : 0.0;
                    if (embedded.slip && nmag > 1e-8)
                    {
                        // SLIP wall: penalize only the wall-NORMAL momentum
                        // (no-penetration), leaving tangential free.  Normal from
                        // grad(phi).  (Deep in the solid grad~0 -> falls through to
                        // the full no-slip branch below to hold the static solid.)
                        const Set::Vector nhat = nrm / nmag;
                        Set::Scalar dMn = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            dMn += (M(i, j, k, d) - s_M(i, j, k, d)) * nhat(d);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            M_rhs(i, j, k, d) += -lam * dMn * nhat(d);
                    }
                    else
                    {
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            M_rhs(i, j, k, d) += -lam * (M(i, j, k, d) - s_M(i, j, k, d));
                    }
                    // momentum-only: no mass/energy penalty (pressure builds,
                    // arrested KE -> internal energy; energy-conserving).
                }
            }

           // ------------------------------------------------------------
           // Error Checking
           // ------------------------------------------------------------
           if ( (M_rhs(i, j, k, 0) != M_rhs(i, j, k, 0))
                or (M_rhs(i, j, k, 1) != M_rhs(i, j, k, 1))
                or (E_rhs(i, j, k) != E_rhs(i, j, k))
                or (E0_rhs(i, j, k) != E0_rhs(i, j, k))
                or (E1_rhs(i, j, k) != E1_rhs(i, j, k))
                or (eta_rhs(i, j, k) != eta_rhs(i, j, k))
                or (rho_eta0_rhs(i, j, k) != rho_eta0_rhs(i, j, k))
                or (rho_eta1_rhs(i, j, k) != rho_eta1_rhs(i, j, k)))
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HYDRO2 (6-eq RHS)");
                Util::ParallelMessage(INFO, "time=", time, " lev=", lev, " i=", i, " j=", j);
                Util::ParallelMessage(INFO, "flux_xlo: ", flux_xlo);
                Util::ParallelMessage(INFO, "flux_xhi: ", flux_xhi);
                Util::ParallelMessage(INFO, "flux_ylo: ", flux_ylo);
                Util::ParallelMessage(INFO, "flux_yhi: ", flux_yhi);
                Util::ParallelMessage(INFO, "Source=", Source(i, j, k, 0), ", ", Source(i, j, k, 1), ", ", Source(i, j, k, 2), ", ", Source(i, j, k, 3));
                Util::ParallelMessage(INFO, "drhoeta0/dt=", rho_eta0_rhs(i, j, k));
                Util::ParallelMessage(INFO, "drhoeta1/dt=", rho_eta1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dM/dt=", M_rhs(i, j, k, 0), ", ", M_rhs(i, j, k, 1));
                Util::ParallelMessage(INFO, "dE/dt=", E_rhs(i, j, k));
                Util::ParallelMessage(INFO, "dE0/dt=", E0_rhs(i, j, k), " dE1/dt=", E1_rhs(i, j, k));
                Util::ParallelMessage(INFO, "deta/dt=", eta_rhs(i, j, k));
                Util::Abort(INFO);
            }

            // Calculate vorticity (curl of velocity) for visualization.
#if AMREX_SPACEDIM == 2
            omega(i, j, k) = (gradu(1, 0) - gradu(0, 1));           // omega_z
#else
            omega(i, j, k, 0) = gradu(2, 1) - gradu(1, 2);          // omega_x = du_z/dy - du_y/dz
            omega(i, j, k, 1) = gradu(0, 2) - gradu(2, 0);          // omega_y = du_x/dz - du_z/dx
            omega(i, j, k, 2) = gradu(1, 0) - gradu(0, 1);          // omega_z = du_y/dx - du_x/dy
#endif
        });
        BL_PROFILE_VAR_STOP(prof_cells);
    }
    } // end omp parallel (RHS main loop)
    // [2026-09-25] SHARP-WALL SOLID CELLS ARE INERT (solid.wall_flux = 1).
    // The wall-face flux (fluid vs mirror) is added to BOTH cells of the face, so
    // the solid cell next to the wall received the wall pressure (plus viscous
    // stress) on one face and zero flux on its solid-solid faces: a large one-sided
    // push every stage.  A stiff penalty (brinkman 1e6, the airfoil decks) wiped it
    // out each step, but a soft one (Re40 cylinder, brinkman 70) let those cells
    // accelerate (|u| 4 -> 20 -> 1e8 in 15 steps at phi = 0.49) and the viscous
    // stress fed that motion back into the fluid -> blow-up.  The solid is not part
    // of the fluid domain here: record the momentum it would have received (the
    // pressure + viscous force ON THE BODY, used by solid.force_int) and zero its
    // right-hand side.  The solid state is held by the implicit update in Advance.
    if (embedded.apply && embedded.wall_flux)
    {
        const bool want_force = (solid_force_int > 0 && lev == finest_level);
        const bool force_carry_corr = embedded.wall_normal && embedded.slip && !embedded.wall_redistribute;
        Set::Scalar vol = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) vol *= DX[d];
        // [2026-09-29] CONSERVATIVE REDISTRIBUTION (slip wall + solid.wall_normal).
        // With the ghost reflected about the TRUE normal, flow crosses riser faces of
        // the staircase: mass/energy entering a solid cell would be deleted when its
        // RHS is zeroed (Euler NACA 0008: 2.4 % of the chord mass flux lost, and its
        // momentum read as +0.049 Cd by the force diagnostic).  Hand each solid cell's
        // net mass / energy RHS to its face-neighbour fluid cells in equal shares, with
        // the carried momentum (mass share x receiving-cell velocity).  Exactly
        // conservative for mass and energy; the carried momentum is taken off the force.
        Set::Scalar redis_force[3] = {0.0, 0.0, 0.0};
        if (embedded.wall_normal && embedded.slip && embedded.wall_redistribute)
        {
            constexpr int NR = 5;                          // rho_eta0, rho_eta1, E, E0, E1
            amrex::MultiFab rs(M_rhs_mf.boxArray(), M_rhs_mf.DistributionMap(), NR, 1);
            rs.setVal(0.0);
            amrex::MultiFab::Copy(rs, rho_eta0_rhs_mf, 0, 0, 1, 0);
            amrex::MultiFab::Copy(rs, rho_eta1_rhs_mf, 0, 1, 1, 0);
            amrex::MultiFab::Copy(rs, E_rhs_mf,        0, 2, 1, 0);
            amrex::MultiFab::Copy(rs, E0_rhs_mf,       0, 3, 1, 0);
            amrex::MultiFab::Copy(rs, E1_rhs_mf,       0, 4, 1, 0);
            rs.FillBoundary(geom[lev].periodicity());
            embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
            amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> rd_op;
            amrex::ReduceData<Set::Scalar, Set::Scalar, Set::Scalar> rd_data(rd_op);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(M_rhs_mf, HydroTiling()); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.tilebox();
                auto phi = embedded.phi_mf[lev]->const_array(mfi);
                auto R   = rs.const_array(mfi);
                auto r0  = rho_eta0_rhs_mf.array(mfi);
                auto r1  = rho_eta1_rhs_mf.array(mfi);
                auto Mr  = M_rhs_mf.array(mfi);
                auto Er  = E_rhs_mf.array(mfi);
                auto E0r = E0_rhs_mf.array(mfi);
                auto E1r = E1_rhs_mf.array(mfi);
                auto Ms  = M_mf_in.const_array(mfi);
                auto q0  = rho_eta0_mf_in.const_array(mfi);
                auto q1  = rho_eta1_mf_in.const_array(mfi);
                rd_op.eval(bx, rd_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar, Set::Scalar, Set::Scalar> {
                    if (phi(i, j, k) < 0.5) return {0.0, 0.0, 0.0};          // receivers are fluid cells
                    Set::Scalar add[NR] = {0.0, 0.0, 0.0, 0.0, 0.0};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        for (int sg = -1; sg <= 1; sg += 2)
                        {
                            const int si = i + (d == 0 ? sg : 0), sj = j + (d == 1 ? sg : 0), sk = k + (d == 2 ? sg : 0);
                            if (phi(si, sj, sk) >= 0.5) continue;              // neighbour is fluid
                            int nf = 0;                                         // fluid face-neighbours of that solid cell
                            for (int e = 0; e < AMREX_SPACEDIM; ++e)
                                for (int tg = -1; tg <= 1; tg += 2)
                                    if (phi(si + (e == 0 ? tg : 0), sj + (e == 1 ? tg : 0), sk + (e == 2 ? tg : 0)) >= 0.5) ++nf;
                            if (nf == 0) continue;
                            for (int c = 0; c < NR; ++c) add[c] += R(si, sj, sk, c) / nf;
                        }
                    r0(i, j, k) += add[0]; r1(i, j, k) += add[1]; Er(i, j, k) += add[2];
                    E0r(i, j, k) += add[3]; E1r(i, j, k) += add[4];
                    const Set::Scalar rho = std::max(q0(i, j, k) + q1(i, j, k), 1e-30);
                    const Set::Scalar dm = add[0] + add[1];
                    Set::Scalar f[3] = {0.0, 0.0, 0.0};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const Set::Scalar dM = dm * Ms(i, j, k, d) / rho;
                        Mr(i, j, k, d) += dM;
                        f[d] = dM * vol;
                    }
                    return {f[0], f[1], f[2]};
                });
            }
            auto rdv = rd_data.value(rd_op);
            redis_force[0] = amrex::get<0>(rdv); redis_force[1] = amrex::get<1>(rdv); redis_force[2] = amrex::get<2>(rdv);
        }
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> rf_op;
        amrex::ReduceData<Set::Scalar, Set::Scalar, Set::Scalar> rf_data(rf_op);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(M_rhs_mf, HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            auto phi = embedded.phi_mf[lev]->const_array(mfi);
            auto r0  = rho_eta0_rhs_mf.array(mfi);
            auto r1  = rho_eta1_rhs_mf.array(mfi);
            auto Mr  = M_rhs_mf.array(mfi);
            auto Er  = E_rhs_mf.array(mfi);
            auto er  = eta_rhs_mf.array(mfi);
            auto E0r = E0_rhs_mf.array(mfi);
            auto E1r = E1_rhs_mf.array(mfi);
            auto Ms  = M_mf_in.const_array(mfi);
            auto q0  = rho_eta0_mf_in.const_array(mfi);
            auto q1  = rho_eta1_mf_in.const_array(mfi);
            rf_op.eval(bx, rf_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar, Set::Scalar, Set::Scalar> {
                if (phi(i, j, k) >= 0.5) return {0.0, 0.0, 0.0};
                Set::Scalar f[3] = {0.0, 0.0, 0.0};
                // [2026-09-30] slip + wall_normal (ghost-cell wall, no redistribution): mass
                // crosses riser faces, and the solid cell's momentum RHS then also holds the
                // momentum that mass carries -- not a force on the body.  (Euler NACA 0008:
                // ~2.3e-3 of mass created at the nose read as thrust, diagnostic Cd -0.0085
                // vs +0.0135 from a control-volume momentum balance.)  Remove it: mass RHS
                // times the mean velocity of the cell's fluid face-neighbours.
                Set::Scalar ucar[3] = {0.0, 0.0, 0.0};
                if (force_carry_corr)
                {
                    int nf = 0;
                    for (int dd = 0; dd < AMREX_SPACEDIM; ++dd)
                        for (int sg = -1; sg <= 1; sg += 2)
                        {
                            const int ii = i + (dd == 0 ? sg : 0), jj = j + (dd == 1 ? sg : 0), kk = k + (dd == 2 ? sg : 0);
                            if (phi(ii, jj, kk) < 0.5) continue;
                            const Set::Scalar rn = std::max(q0(ii, jj, kk) + q1(ii, jj, kk), 1e-30);
                            for (int d = 0; d < AMREX_SPACEDIM; ++d) ucar[d] += Ms(ii, jj, kk, d) / rn;
                            ++nf;
                        }
                    if (nf > 0) for (int d = 0; d < AMREX_SPACEDIM; ++d) ucar[d] /= nf;
                }
                const Set::Scalar dm = force_carry_corr ? (r0(i, j, k) + r1(i, j, k)) : 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) { f[d] = (Mr(i, j, k, d) - dm * ucar[d]) * vol; Mr(i, j, k, d) = 0.0; }
                r0(i, j, k) = 0.0; r1(i, j, k) = 0.0; Er(i, j, k) = 0.0;
                er(i, j, k) = 0.0; E0r(i, j, k) = 0.0; E1r(i, j, k) = 0.0;
                return {f[0], f[1], f[2]};
            });
        }
        auto rfv = rf_data.value(rf_op);
        // body force = momentum pushed into the solid - momentum handed back with the
        // redistributed mass (0 unless the slip + wall_normal redistribution ran)
        if (want_force) { rhs_force[0] = amrex::get<0>(rfv) - redis_force[0]; rhs_force[1] = amrex::get<1>(rfv) - redis_force[1]; rhs_force[2] = amrex::get<2>(rfv) - redis_force[2]; }
    }
    // [DIAG RHSDECOMP] mass1 RHS budget: sum(rhs)*dV must equal
    // net boundary influx + interior sources.  Printed every 300 calls.
    if (relax_diag)
    {
        const amrex::Box &dom = geom[lev].Domain();
        Set::Scalar bx_lo = 0, bx_hi = 0, by_lo = 0, by_hi = 0, src_sum = 0;
        for (amrex::MFIter pmfi(rhsp_fxlo_mf, false); pmfi.isValid(); ++pmfi)
        {
            const amrex::Box &pbx = pmfi.validbox();
            auto flo = rhsp_fxlo_mf.array(pmfi);
            auto fhi = rhsp_fxhi_mf.array(pmfi);
            auto glo = rhsp_fylo_mf.array(pmfi);
            auto ghi = rhsp_fyhi_mf.array(pmfi);
            auto sr  = rhsp_src_mf.array(pmfi);
            amrex::LoopOnCpu(pbx, [&](int i, int j, int k) {
                if (i == dom.smallEnd(0)) bx_lo += flo(i, j, k) * DX[1];
                if (i == dom.bigEnd(0))   bx_hi -= fhi(i, j, k) * DX[1];
                if (j == dom.smallEnd(1)) by_lo += glo(i, j, k) * DX[0];
                if (j == dom.bigEnd(1))   by_hi -= ghi(i, j, k) * DX[0];
                src_sum += sr(i, j, k) * DX[0] * DX[1];
            });
        }
        Set::Scalar rhs_sum = rho_eta1_rhs_mf.sum(0) * DX[0] * DX[1];
        static int _rd_n = 0;
        if ((_rd_n++ % 300) == 0)
        {
            char b[260];
            snprintf(b, 260, "RHSDECOMP rhs=%.6e bxlo=%.6e bxhi=%.6e bylo=%.6e byhi=%.6e src=%.6e gap=%.3e",
                     rhs_sum, bx_lo, bx_hi, by_lo, by_hi, src_sum,
                     rhs_sum - (bx_lo + bx_hi + by_lo + by_hi + src_sum));
            Util::Message(INFO, b);
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

    // a restart does not call Initialize, which is where step_counter is sized
    if (step_counter.size() <= (size_t)lev) step_counter.resize(lev + 1, 0);
    step_counter[lev]++;
    if (diag_gasmass) GasLedger(lev, "advance_entry", time);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK advance_entry lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }

    // MOVING RIGID SOLID (solid.moving): bring the geometry to this step's start time.
    if (embedded.apply && (embedded.moving || ablation.on)) UpdateMovingSolid(lev, time, dt);

    // shell.slip = 7: on the first step from t = 0 turn the IC areal density Gamma (shell field)
    // into the transported m = Gamma |grad psi|.  Levels made later by regrid get m by
    // interpolation from the coarser level, so they need nothing here.
    if ((shell_slip == 7 || shell_slip == 11 || shell_slip == 12 || shell_slip == 15) && (marmottant || apply_surface_tension))
    {
        if ((int)shell_psi_init.size() <= lev) shell_psi_init.resize(lev + 1, 0);
        if (!shell_psi_init[lev] && time <= 0.0)
        {
            amrex::MultiFab Gp;
            ShellPsiGrad(lev, time, Gp);
            for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &vb = mfi.validbox();
                auto g  = Gp.const_array(mfi);
                auto sh = shell_mf[lev]->array(mfi);
                amrex::ParallelFor(vb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    sh(i, j, k) = (g(i, j, k) > 0.0) ? sh(i, j, k) * g(i, j, k) : 0.0;
                });
            }
            FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
            Util::Message(INFO, (shell_slip >= 11 ? "shell.slip = 11/12/15: shell field set to m = Gamma |grad eta| on level " : "shell.slip = 7: shell field set to m = Gamma |grad psi| on level "), lev);
        }
        shell_psi_init[lev] = 1;
    }

    // Swapping pointers (6-eq primaries -- canonical set)
    std::swap(density_old_mf[lev],         density_mf[lev]);
    std::swap(momentum_old_mf[lev],        momentum_mf[lev]);
    std::swap(energy_per_vol_old_mf[lev],  energy_per_vol_mf[lev]);
    std::swap(energy_per_mas_old_mf[lev],  energy_per_mas_mf[lev]);
    std::swap(eta_old_mf[lev],             eta_mf[lev]);
    std::swap(rho_eta0_old_mf[lev],        rho_eta0_mf[lev]);
    std::swap(rho_eta1_old_mf[lev],        rho_eta1_mf[lev]);
    std::swap(energy0_old_mf[lev],         energy0_mf[lev]);
    std::swap(energy1_old_mf[lev],         energy1_mf[lev]);
    std::swap(shell_old_mf[lev],           shell_mf[lev]);
    std::swap(cfun_old_mf[lev],            cfun_mf[lev]);
    if (elastic.on) std::swap(elastic.ebasis_old_mf[lev], elastic.ebasis_mf[lev]);
    if (phasechange.on) std::swap(phasechange.vapor_old_mf[lev], phasechange.vapor_mf[lev]);
    if (species.on) std::swap(species.old_mf[lev], species.mf[lev]);
    const int ix_el = 9, ix_pc = 9 + (elastic.on ? 1 : 0);      // optional solution components
    const int ix_sp = ix_pc + (phasechange.on ? 1 : 0);

    // ------------------------------------------------------------
    // Time Integration
    //
    // 6-equation primary set (Schmidmayer 2020 eqs. 13 + 16):
    //   [0] rho_eta0       = alpha_1 rho_1
    //   [1] rho_eta1       = alpha_2 rho_2
    //   [2] momentum       = rho u (2-component)
    //   [3] energy_per_vol = rho E (redundant)
    //   [4] eta            = alpha_1
    //   [5] energy0        = alpha_1 rho_1 e_1
    //   [6] energy1        = alpha_2 rho_2 e_2
    // ------------------------------------------------------------

    amrex::Vector<amrex::MultiFab> solution_new;
    solution_new.emplace_back(*rho_eta0_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*rho_eta1_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*momentum_mf[lev].get(),          amrex::MakeType::make_alias, 0, AMREX_SPACEDIM);
    solution_new.emplace_back(*energy_per_vol_mf[lev].get(),    amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*eta_mf[lev].get(),               amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy0_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*energy1_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*shell_mf[lev].get(),             amrex::MakeType::make_alias, 0, 1);
    solution_new.emplace_back(*cfun_mf[lev].get(),              amrex::MakeType::make_alias, 0, 1);
    if (elastic.on) solution_new.emplace_back(*elastic.ebasis_mf[lev].get(), amrex::MakeType::make_alias, 0, ElasticSolid::NEB);   // [ix_el]
    if (phasechange.on) solution_new.emplace_back(*phasechange.vapor_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);            // [ix_pc]
    if (species.on) solution_new.emplace_back(*species.mf[lev].get(), amrex::MakeType::make_alias, 0, species.ns());                  // [ix_sp]

    amrex::Vector<amrex::MultiFab> solution_old;
    solution_old.emplace_back(*rho_eta0_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*rho_eta1_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*momentum_old_mf[lev].get(),      amrex::MakeType::make_alias, 0, AMREX_SPACEDIM);
    solution_old.emplace_back(*energy_per_vol_old_mf[lev].get(),amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*eta_old_mf[lev].get(),           amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy0_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*energy1_old_mf[lev].get(),       amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*shell_old_mf[lev].get(),         amrex::MakeType::make_alias, 0, 1);
    solution_old.emplace_back(*cfun_old_mf[lev].get(),          amrex::MakeType::make_alias, 0, 1);
    if (elastic.on) solution_old.emplace_back(*elastic.ebasis_old_mf[lev].get(), amrex::MakeType::make_alias, 0, ElasticSolid::NEB);
    if (phasechange.on) solution_old.emplace_back(*phasechange.vapor_old_mf[lev].get(), amrex::MakeType::make_alias, 0, 1);
    if (species.on) solution_old.emplace_back(*species.old_mf[lev].get(), amrex::MakeType::make_alias, 0, species.ns());

    amrex::TimeIntegrator timeintegrator(solution_new, time);

    // Butcher weights b_i of the explicit integrator AMReX will run (same
    // integration.type / integration.rk.type it parses).  User tableaus read
    // integration.rk.weights.  See reflux_stage_weight in Hydro2.H.
    if (reflux_rk_weights.empty())
    {
        amrex::ParmParse pp("integration");
        std::string itype = "ForwardEuler";
        pp.query("type", itype);
        if (itype == "RungeKutta")
        {
            int rktype = 3;
            pp.query("rk.type", rktype);
            if      (rktype == 1) reflux_rk_weights = {1.0};
            else if (rktype == 2) reflux_rk_weights = {0.5, 0.5};
            else if (rktype == 3) reflux_rk_weights = {1./6., 1./6., 2./3.};
            else if (rktype == 4) reflux_rk_weights = {1./6., 1./3., 1./3., 1./6.};
            else { amrex::Vector<Set::Scalar> w; pp.getarr("rk.weights", w); reflux_rk_weights.assign(w.begin(), w.end()); }
        }
        else reflux_rk_weights = {1.0};
    }
    int reflux_rk_stage = 0;

    timeintegrator.set_rhs([&](
                               amrex::Vector<amrex::MultiFab> &rhs_mf,
                               amrex::Vector<amrex::MultiFab> &solution_mf,
                               const Set::Scalar time) {
        // AMReX's explicit RK evaluates the RHS exactly once per stage, in
        // stage order (AMReX_RKIntegrator.H advance()).
        reflux_stage_weight = reflux_rk_weights[std::min<int>(reflux_rk_stage, (int)reflux_rk_weights.size() - 1)];
        reflux_rk_stage++;
        // rhs_mf:      [0]=rho_eta0_rhs, [1]=rho_eta1_rhs, [2]=M_rhs, [3]=E_rhs, [4]=eta_rhs, [5]=E0_rhs, [6]=E1_rhs, [7]=shell_rhs
        // solution_mf: [0]=rho_eta0,     [1]=rho_eta1,     [2]=M,     [3]=E,     [4]=eta,     [5]=E0,     [6]=E1,     [7]=shell
        RHS(lev, time,
            rhs_mf[0], rhs_mf[1], rhs_mf[2], rhs_mf[3], rhs_mf[4], rhs_mf[5], rhs_mf[6], rhs_mf[7], rhs_mf[8],
            solution_mf[0], solution_mf[1], solution_mf[2], solution_mf[3], solution_mf[4], solution_mf[5], solution_mf[6], solution_mf[7], solution_mf[8],
            elastic.on ? &rhs_mf[ix_el] : nullptr, elastic.on ? &solution_mf[ix_el] : nullptr,
            phasechange.on ? &rhs_mf[ix_pc] : nullptr, phasechange.on ? &solution_mf[ix_pc] : nullptr,
            species.on ? &rhs_mf[ix_sp] : nullptr, species.on ? &solution_mf[ix_sp] : nullptr);
    });

    timeintegrator.set_post_stage_action([&](amrex::Vector<amrex::MultiFab> &stage_mf, Set::Scalar time) {
        // Copy stage data to working arrays
        amrex::MultiFab::Copy(*rho_eta0_mf[lev],       stage_mf[0], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*rho_eta1_mf[lev],       stage_mf[1], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*momentum_mf[lev],       stage_mf[2], 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], stage_mf[3], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*eta_mf[lev],            stage_mf[4], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*energy0_mf[lev],        stage_mf[5], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*energy1_mf[lev],        stage_mf[6], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*shell_mf[lev],          stage_mf[7], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(*cfun_mf[lev],           stage_mf[8], 0, 0, 1,              nghost);
        if (elastic.on) amrex::MultiFab::Copy(*elastic.ebasis_mf[lev], stage_mf[ix_el], 0, 0, ElasticSolid::NEB, nghost);
        if (phasechange.on) amrex::MultiFab::Copy(*phasechange.vapor_mf[lev], stage_mf[ix_pc], 0, 0, 1, nghost);
        if (species.on) { amrex::MultiFab::Copy(*species.mf[lev], stage_mf[ix_sp], 0, 0, species.ns(), nghost); CloseSpecies(lev, false); }

        // Everything from here to the end of this hook changes eta WITHOUT moving
        // material (clamping to [0,1], per-phase ghost reconstruction, stiff
        // pressure relaxation inside FillGhost4BC).  Hold the shell's areal
        // density Gamma fixed across all of it; c is rescaled to the new |grad eta|.

        // Clamp eta in domain prior to ghost fill (state can drift slightly outside [0,1])
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            auto eta = eta_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            });
        }

        // Fill all ghost cells (uses 6-eq primitive recovery internally)
        FillGhost4BC(lev, time);

        // Copy back
        amrex::MultiFab::Copy(stage_mf[0], *rho_eta0_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[1], *rho_eta1_mf[lev],       0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[2], *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(stage_mf[3], *energy_per_vol_mf[lev], 0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[4], *eta_mf[lev],            0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[5], *energy0_mf[lev],        0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[6], *energy1_mf[lev],        0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[7], *shell_mf[lev],          0, 0, 1,              nghost);
        amrex::MultiFab::Copy(stage_mf[8], *cfun_mf[lev],           0, 0, 1,              nghost);
        if (elastic.on) amrex::MultiFab::Copy(stage_mf[ix_el], *elastic.ebasis_mf[lev], 0, 0, ElasticSolid::NEB, nghost);
        if (phasechange.on) amrex::MultiFab::Copy(stage_mf[ix_pc], *phasechange.vapor_mf[lev], 0, 0, 1, nghost);
        if (species.on) amrex::MultiFab::Copy(stage_mf[ix_sp], *species.mf[lev], 0, 0, species.ns(), nghost);

    });


    // Zero face-flux scratch so reflux accumulation starts clean each step.
    if (lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0])
    {
        for (int d = 0; d < AMREX_SPACEDIM; d++)
        {
            cc_fluxes[lev].mass[d]  ->setVal(0.0);
            cc_fluxes[lev].mom[d]   ->setVal(0.0);
            cc_fluxes[lev].energy[d]->setVal(0.0);
            cc_fluxes[lev].ene_k[d] ->setVal(0.0);
            if (cc_fluxes[lev].spec[d]) cc_fluxes[lev].spec[d]->setVal(0.0);
        }
    }

    // ------------------------------------------------------------------
    // OPERATOR CHAIN  (Schmidmayer 2017 sec. 4.2):
    //     U^{n+1} = L_relax ( L_cap ( L_hyper (U^n) ) )
    // L_hyper is the SSPRK3 advance of the hyperbolic submodel (16) -- the
    // capillary terms are gated out of RHS above.  L_cap and L_relax then run
    // ONCE per step, in that order.
    // ------------------------------------------------------------------
    defer_relax = false;
    if (ablation.on && ablation.conduction) ablation.qs_mf[lev]->setVal(0.0);
    timeintegrator.advance(solution_old, solution_new, time, dt);
    if (chem.on) ReactChemistry(lev, dt);       // operator-split reaction step (the primitives are refreshed below)
    if (energetic.on) ReactEnergetic(lev, dt);  // condensed reactant -> gas products (the relaxation follows below)
    // plastic relaxation of the elastic deformation (Favrie & Schmidmayer 2026); the released elastic energy becomes
    // internal energy at the re-initialisation inside the FillGhost4BC below
    if (elastic.on && elastic.plastic) elastic.PlasticRelax(lev, dt, *eta_mf[lev], *rho_eta0_mf[lev], *rho_eta1_mf[lev], (species.on ? species.mf[lev].get() : nullptr), cfun_mf[lev].get());
    // in-depth conduction of the solid with the surface fluxes of the stages just taken (energy-conservative)
    if (ablation.on && ablation.conduction) AdvanceSolidTemperature(lev, time, dt);
    // (bulk viscosity: SubcycleBulkViscous runs after the positivity clip below, flames2 order)
    // species diffusion inside one phase (species.diffusion = 1; returns at once otherwise)
    DiffuseSpecies(lev, time, dt);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(reflux_rk_stage == (int)reflux_rk_weights.size(),
        "Hydro2 reflux: RHS evaluations per step != number of RK weights");
    reflux_stage_weight = 1.0;

    // ------------------------------------------------------------------
    // Feed FluxRegister for reflux at coarse-fine boundaries.
    // Convert cell-centered hi-face fluxes (written by RHS) to face-centered
    // MultiFabs, then accumulate into the appropriate register.
    //
    // Register layout matches PostSubcycleReflux:
    //   [0]               rho_eta0
    //   [1]               rho_eta1
    //   [2..2+SD-1]       momentum_k
    //   [2+SD]            rho*E
    //   [3+SD..4+SD]      E_0, E_1
    // ------------------------------------------------------------------
    {
        const bool need_crse = (lev < finest_level
                                && lev + 1 < (int)flux_reg.size() && flux_reg[lev + 1]
                                && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);
        const bool need_fine = (lev > 0
                                && lev < (int)flux_reg.size() && flux_reg[lev]
                                && lev < (int)cc_fluxes.size() && cc_fluxes[lev].mass[0]);

        if (need_crse || need_fine)
        {
            if (need_crse) flux_reg[lev + 1]->setVal(0.0);

            constexpr int IDX_RHO_ETA0 = 0;
            constexpr int IDX_RHO_ETA1 = 1;
            constexpr int IDX_M        = 2;
            constexpr int IDX_E_VOL    = 2 + AMREX_SPACEDIM;
            constexpr int IDX_E0       = 3 + AMREX_SPACEDIM;
            constexpr int IDX_E1       = 4 + AMREX_SPACEDIM;
            constexpr int IDX_SP       = 5 + AMREX_SPACEDIM;   // species partial densities (species.on)
            const int nsp = (species.on && cc_fluxes[lev].spec[0]) ? species.ns() : 0;

            for (int d = 0; d < AMREX_SPACEDIM; d++)
            {
                if (nsp > 0)    // species: cc hi-face flux -> face MultiFab -> register (same conversion as below)
                {
                    cc_fluxes[lev].spec[d]->FillBoundary(geom[lev].periodicity());
                    amrex::BoxArray fba = cc_fluxes[lev].spec[d]->boxArray(); fba.surroundingNodes(d);
                    const amrex::DistributionMapping &dmc = cc_fluxes[lev].spec[d]->DistributionMap();
                    amrex::MultiFab face_sp(fba, dmc, nsp, 0), area_sp(fba, dmc, 1, 0);
                    for (int li = 0; li < (int)cc_fluxes[lev].spec[d]->local_size(); li++)
                    {
                        auto cc_s = cc_fluxes[lev].spec[d]->atLocalIdx(li).array();
                        auto f_s = face_sp.atLocalIdx(li).array();
                        const int dd = d;
                        amrex::ParallelFor(face_sp.atLocalIdx(li).box(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                            const int ii = (dd == 0) ? i - 1 : i, jj = (dd == 1) ? j - 1 : j, kk = (dd == 2) ? k - 1 : k;
                            for (int n = 0; n < nsp; ++n) f_s(i, j, k, n) = cc_s(ii, jj, kk, n);
                        });
                    }
                    const amrex::Real *dxl = geom[lev].CellSize();
                    amrex::Real fa = 1.0;
                    for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2) if (d2 != d) fa *= dxl[d2];
                    area_sp.setVal(fa);
                    if (need_crse) flux_reg[lev + 1]->CrseInit(face_sp, area_sp, d, 0, IDX_SP, nsp, -dt, amrex::FluxRegister::ADD);
                    if (need_fine) flux_reg[lev]->FineAdd(face_sp, area_sp, d, 0, IDX_SP, nsp, dt);
                }
                // FillBoundary so ghost cells inherit a neighbor's hi-face
                // flux for the cc->face conversion below.  At c/f boundaries
                // FillBoundary has nothing to copy from -- the lo-face values
                // written by RHS (commit 8a1633977 fix) persist there.
                cc_fluxes[lev].mass[d]  ->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].mom[d]   ->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].energy[d]->FillBoundary(geom[lev].periodicity());
                cc_fluxes[lev].ene_k[d] ->FillBoundary(geom[lev].periodicity());

                // Build face-centered MFs on cc_fluxes' BA/dmap (surroundingNodes).
                amrex::BoxArray face_ba = cc_fluxes[lev].mass[d]->boxArray();
                face_ba.surroundingNodes(d);
                const amrex::DistributionMapping& dm_cc = cc_fluxes[lev].mass[d]->DistributionMap();
                amrex::MultiFab face_mass (face_ba, dm_cc, 2,              0);
                amrex::MultiFab face_mom  (face_ba, dm_cc, AMREX_SPACEDIM, 0);
                amrex::MultiFab face_ene  (face_ba, dm_cc, 1,              0);
                amrex::MultiFab face_ene_k(face_ba, dm_cc, 2,              0);

                // Convert cell-centered hi-face flux to face-centered:
                //   face(f, j) = cc(f - 1, j)   for d=0
                // i.e., the hi-face flux of cell f-1 IS the flux at face f.
                int nlocal = (int)cc_fluxes[lev].mass[d]->local_size();
                for (int li = 0; li < nlocal; li++)
                {
                    auto cc_m  = cc_fluxes[lev].mass  [d]->atLocalIdx(li).array();
                    auto cc_p  = cc_fluxes[lev].mom   [d]->atLocalIdx(li).array();
                    auto cc_e  = cc_fluxes[lev].energy[d]->atLocalIdx(li).array();
                    auto cc_ek = cc_fluxes[lev].ene_k [d]->atLocalIdx(li).array();

                    auto f_m  = face_mass  .atLocalIdx(li).array();
                    auto f_p  = face_mom   .atLocalIdx(li).array();
                    auto f_e  = face_ene   .atLocalIdx(li).array();
                    auto f_ek = face_ene_k .atLocalIdx(li).array();

                    const amrex::Box& fbx = face_mass.atLocalIdx(li).box();
                    const int dd = d;
                    amrex::ParallelFor(fbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        const int ii = (dd == 0) ? i - 1 : i;
                        const int jj = (dd == 1) ? j - 1 : j;
                        const int kk = (dd == 2) ? k - 1 : k;   // z-face shift (3D)
                        f_m (i, j, k, 0) = cc_m (ii, jj, kk, 0);
                        f_m (i, j, k, 1) = cc_m (ii, jj, kk, 1);
                        for (int n = 0; n < AMREX_SPACEDIM; n++)
                            f_p(i, j, k, n) = cc_p(ii, jj, kk, n);
                        f_e (i, j, k)    = cc_e (ii, jj, kk);
                        f_ek(i, j, k, 0) = cc_ek(ii, jj, kk, 0);
                        f_ek(i, j, k, 1) = cc_ek(ii, jj, kk, 1);
                    });
                }

                // Face area for area-weighted accumulation: product of the cell
                // sizes in all directions EXCEPT the face normal d.
                //   2D: x-face area = dy, y-face area = dx.
                //   3D: x-face = dy*dz, y-face = dx*dz, z-face = dx*dy.
                const amrex::Real* dx_lev = geom[lev].CellSize();
                amrex::Real face_area = 1.0;
                for (int dd2 = 0; dd2 < AMREX_SPACEDIM; ++dd2)
                    if (dd2 != d) face_area *= dx_lev[dd2];
                amrex::MultiFab area_mf(face_ba, dm_cc, 1, 0);
                area_mf.setVal(face_area);

                // Sign convention: CrseInit gets -dt (subtracts coarse flux),
                // FineAdd gets +dt (adds fine flux). Reflux = (fine - coarse).
                if (need_crse)
                {
                    flux_reg[lev + 1]->CrseInit(face_mass,  area_mf, d, 0, IDX_RHO_ETA0, 2,              -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_mom,   area_mf, d, 0, IDX_M,        AMREX_SPACEDIM, -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_ene,   area_mf, d, 0, IDX_E_VOL,    1,              -dt, amrex::FluxRegister::ADD);
                    flux_reg[lev + 1]->CrseInit(face_ene_k, area_mf, d, 0, IDX_E0,       2,              -dt, amrex::FluxRegister::ADD);
                }
                if (need_fine)
                {
                    flux_reg[lev]->FineAdd(face_mass,  area_mf, d, 0, IDX_RHO_ETA0, 2,              dt);
                    flux_reg[lev]->FineAdd(face_mom,   area_mf, d, 0, IDX_M,        AMREX_SPACEDIM, dt);
                    flux_reg[lev]->FineAdd(face_ene,   area_mf, d, 0, IDX_E_VOL,    1,              dt);
                    flux_reg[lev]->FineAdd(face_ene_k, area_mf, d, 0, IDX_E0,       2,              dt);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // [2026-09-24] IMPLICIT BRINKMAN PENALIZATION (solid.implicit = 1).
    // Exact solution of dM/dt = -lam (M - Ms), lam = brinkman (1 - phi),
    // over this level's dt, applied once per level step (first-order
    // operator split; exact and unconditionally stable in the stiff part).
    // No-slip: every momentum component relaxes to the solid momentum.
    // Slip:    only the wall-normal component (n = grad phi / |grad phi|),
    //          same n and same no-slip fallback as the explicit RHS form.
    // Energy is left untouched, exactly as in the explicit form: the kinetic
    // energy removed is returned as internal energy (energy-conserving), and
    // the post-step FillGhost4BC -> RelaxAndReinit re-derives p, E0, E1.
    // ------------------------------------------------------------------
    // [2026-09-25] sharp-wall mode always takes this path (the solid RHS is zeroed
    // in RHS(); here the solid cells are set exactly to the solid state).
    if (embedded.apply && ((embedded.implicit && embedded.brinkman > 0.0) || embedded.wall_flux))
    {
        // Target the solid VELOCITY u_s = M_s / rho_s (M -> rho u_s), not the
        // solid momentum: the solid is porous to mass, so its density drifts
        // and a fixed-momentum target makes a moving solid run at M_s/rho != u_s
        // (Brinkman Couette unit test: top slab 1-4% fast).  And add the work
        // done BY the solid, dE = u_s . dM: dM/dt = F with the solid moving at
        // u_s puts u_s.F into the fluid's total energy (kinetic u.F plus
        // dissipation (u_s-u).F >= 0).  For a STATIC solid u_s = 0 and both
        // reduce exactly to the previous form (M -> 0, E unchanged: the
        // arrested KE becomes heat).
        const Set::Scalar lam0 = embedded.brinkman;
        const int slip_l = embedded.slip;
        const int sharp_l = embedded.wall_flux;
        const Set::Scalar small_l = small;
        embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
        // Force on the body = - (momentum the penalty adds to the fluid) / dt.
        amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> pf_op;
        amrex::ReduceData<Set::Scalar, Set::Scalar, Set::Scalar> pf_data(pf_op);
        Set::Scalar vol = 1.0;
        for (int d = 0; d < AMREX_SPACEDIM; ++d) vol *= DX[d];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*momentum_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox();
            auto M    = momentum_mf[lev]->array(mfi);
            auto E    = energy_per_vol_mf[lev]->array(mfi);
            auto r0   = rho_eta0_mf[lev]->const_array(mfi);
            auto r1   = rho_eta1_mf[lev]->const_array(mfi);
            auto phi  = embedded.phi_mf[lev]->const_array(mfi);
            auto s_M  = embedded.momentum_mf[lev]->const_array(mfi);
            auto s_r0 = embedded.density0_mf[lev]->const_array(mfi);
            auto s_r1 = embedded.density1_mf[lev]->const_array(mfi);
            auto s_E0 = embedded.energy0_mf[lev]->const_array(mfi);
            auto s_E1 = embedded.energy1_mf[lev]->const_array(mfi);
            auto E0s  = energy0_mf[lev]->array(mfi);
            auto E1s  = energy1_mf[lev]->array(mfi);
            pf_op.eval(bx, pf_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar, Set::Scalar, Set::Scalar> {
                // With the sharp-wall flux on, a cell is fluid (phi >= 0.5) or solid,
                // and the penalty must use the SAME binary classification.  Bilinear
                // sampling of a sharp bitmap leaves 0.5 < phi < 1 in cells within a
                // pixel of the surface; with chi = 1 - phi and a stiff brinkman
                // (f = 1 - exp(-lam chi dt) ~ 1 even for chi ~ 1e-3) those FLUID
                // cells were fully arrested every step and their kinetic energy
                // dumped as heat -- 2.25x freestream temperature at the M 0.8
                // leading edge, then a crash at t = 0.55 (NACA 0012, M 0.8, Re 500).
                const Set::Scalar phic = EmbeddedSolid::clampPhi(phi(i, j, k));
                const Set::Scalar chi = sharp_l ? ((phic < 0.5) ? 1.0 : 0.0) : 1.0 - phic;
                if (chi <= 0.0) return {0.0, 0.0, 0.0};
                // [2026-09-25] sharp wall: solid cells projected exactly (f = 1), so the
                // wall no longer depends on solid.brinkman.
                const Set::Scalar f = sharp_l ? 1.0 : 1.0 - std::exp(-lam0 * chi * dt);   // fraction removed
                const Set::Scalar rho_c = std::max(r0(i, j, k) + r1(i, j, k), small_l);
                const Set::Scalar rho_s = std::max(s_r0(i, j, k) + s_r1(i, j, k), small_l);
                Set::Scalar us[AMREX_SPACEDIM], dM[AMREX_SPACEDIM];
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    us[d] = s_M(i, j, k, d) / rho_s;
                    dM[d] = -f * (M(i, j, k, d) - rho_c * us[d]);        // no-slip: all components
                }
                if (slip_l)
                {
                    const Set::Vector nrm = Numeric::Gradient(phi, i, j, k, 0, DX);
                    const Set::Scalar nmag = nrm.norm();
                    if (nmag > 1e-8)
                    {
                        const Set::Vector nhat = nrm / nmag;
                        Set::Scalar dMn = 0.0;
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            dMn += (M(i, j, k, d) - rho_c * us[d]) * nhat(d);
                        for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            dM[d] = -f * dMn * nhat(d);                  // slip: normal only
                    }
                }
                Set::Scalar work = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    M(i, j, k, d) += dM[d];
                    work += us[d] * dM[d];
                }
                E(i, j, k) += work;
                // [2026-09-24] Sharp-wall mode: hold the SOLID cell's internal energy at
                // its reference state.  Otherwise the momentum that pressure and
                // viscous stress push into the solid is turned into heat there every
                // step (static solid: dE = 0 while KE is removed) and never leaves --
                // measured T_solid/T_inf = 10 after 4 convective times on the NACA
                // 0012, sound speed 3x the fluid's, in cells the CFL does not see.
                if (sharp_l && phic < 0.5)
                {
                    E0s(i, j, k) = s_E0(i, j, k);
                    E1s(i, j, k) = s_E1(i, j, k);
                    Set::Scalar ke = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) ke += M(i, j, k, d) * M(i, j, k, d);
                    E(i, j, k) = s_E0(i, j, k) + s_E1(i, j, k) + 0.5 * ke / rho_c;
                }
                const Set::Scalar w = -vol / dt;
                return { w * dM[0], w * dM[1], (AMREX_SPACEDIM == 3) ? w * dM[AMREX_SPACEDIM - 1] : 0.0 };
            });
        }
        auto pfv = pf_data.value(pf_op);
        pen_force[0] = amrex::get<0>(pfv); pen_force[1] = amrex::get<1>(pfv); pen_force[2] = amrex::get<2>(pfv);
    }

    if (diag_gasmass) GasLedger(lev, "pre_clip", time);
    // ENFORCE POSITIVITY after time advance
    // [DIAG] measure clip-created mass (CLIPLEDGER, gated on relax_diag)
    Set::Scalar _clip0 = 0.0, _clip1 = 0.0;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !relax_diag)
#endif
    for (amrex::MFIter mfi(*rho_eta0_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        if (relax_diag)
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                if (rho_eta0(i, j, k) < 0.0) _clip0 -= rho_eta0(i, j, k);
                if (rho_eta1(i, j, k) < 0.0) _clip1 -= rho_eta1(i, j, k);
            });

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // NEGATIVITY clip only -- flooring at `small` (1e-6) parks an
            // artificial discontinuity in the trace-phase partial density at
            // the contour where the tanh tail crosses 1e-6 (eta ~ 1e-9);
            // the recovery rho_k = arh_k/alpha then reads a fake 1e6 kg/m^3
            // liquid there and the biased per-phase fluxes PUMP gas mass at
            // that contour (measured: +16%/period gas-mass creation in
            // UT-RA on the pre-fix code, concentrated in the two columns
            // straddling the floor contour).  Divides downstream are all
            // DIV_FLOOR/max-protected, so zero is safe here.
            rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), 0.0);
            rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), 0.0);
        });
    }
    if (relax_diag && (_clip0 > 0.0 || _clip1 > 0.0))
    {
        static Set::Scalar _clip0_cum = 0.0, _clip1_cum = 0.0;
        static int _clip_n = 0;
        _clip0_cum += _clip0; _clip1_cum += _clip1;
        if ((_clip_n++ % 200) == 0)
        {
            char b[200];
            snprintf(b, 200, "CLIPLEDGER step-created0=%.4e step-created1=%.4e cum0=%.4e cum1=%.4e",
                     _clip0, _clip1, _clip0_cum, _clip1_cum);
            Util::Message(INFO, b);
        }
    }
    if (diag_gasmass) GasLedger(lev, "post_clip", time);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK advance_exit  lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
    // Bulk viscosity as a separate, sub-cycled operator (visc_subcycle; Lie
    // split U^{n+1} = L_relax L_visc,bulk L_hyper U^n).  Primitives (v, rho) and
    // ghosts are current from the last RK post-stage hook; the refresh below
    // rebuilds them from the updated M, E.  (Returns at once with visc_subcycle = 0.)
    SubcycleBulkViscous(lev, time, dt);
    if (diag_gasmass) GasLedger(lev, "bulkvisc_done", time);
    // Shell viscous stress (visc_subcycle), once per step.  The split
    // under-damps by a fraction ~ dt: kappa_eff/kappa_s 0.925 / 0.955 / 0.973 at
    // cfl 0.5 / 0.25 / 0.125 against 0.988 dt-limited (single-phase pulse,
    // R0/dx 16, eps/dx 1; 2026-10-23).  A Strang split gave identical numbers --
    // this is stiff-relaxation order reduction, not the splitting order.
    SubcycleShellViscous(lev, time, dt);
    if (diag_gasmass) GasLedger(lev, "shellvisc_done", time);
    // Per-phase conduction as its own sub-cycled operator (thermal.subcycle; returns at once when off).
    SubcycleConduction(lev, time, dt);
    // Gas-liquid heat exchange (thermal.exchange), exact relaxation over dt.
    if (thermal_exchange) ThermalExchange(lev, dt);
    if (diag_gasmass) GasLedger(lev, "thermal_done", time);

    // ============================================================
    // POST-INTEGRATION PRIMITIVE REFRESH.
    // ============================================================
    FillGhost4BC(lev, time + dt);
    // Level-set shell: move Gamma with the eta level sets, not the fluid (option 4).
    if (marmottant || apply_surface_tension)
    {
        if (shell_slip == 5) ShellLevelSetSlip5(lev, time + dt, dt);
        else if (shell_slip == 6) ShellLevelSetSlip(lev, time + dt, dt);
        else if (shell_slip == 11) ShellSlipMass(lev, time + dt, dt);
        else if (shell_slip == 13) ShellLevelSetSlip13(lev, time + dt, dt);
        else if (shell_slip == 15) ShellRemap15(lev, time + dt, dt);
    }

    // ------------------------------------------------------------------
    // [2026-09-24] FORCE / PROBE TIME HISTORY (solid.force_int > 0), finest
    // level only (it must cover the body -- use a refine_box).  Columns:
    //   step lev time  F_total[d]  F_pressure[d]  p(probe_n)...
    // F_total  = momentum removed by the implicit Brinkman penalty / dt
    //            (pressure + viscous: both reach the solid cells and are taken
    //            out by the penalty; exact for a static solid in steady state).
    // F_press  = sum over fluid/solid faces of -p n_out dA (n_out solid->fluid).
    // ------------------------------------------------------------------
    if (embedded.apply && solid_force_int > 0 && lev == finest_level
        && (step_counter[lev] % solid_force_int) == 0)
    {
        const int np = (int)probe_x.size();
        std::vector<Set::Scalar> red(2 * AMREX_SPACEDIM + 2 * np, 0.0);
        const Set::Scalar *plo = geom[lev].ProbLo();
        Set::Scalar area[AMREX_SPACEDIM];
        for (int d = 0; d < AMREX_SPACEDIM; ++d) { area[d] = 1.0; for (int e = 0; e < AMREX_SPACEDIM; ++e) if (e != d) area[d] *= DX[e]; }
        for (amrex::MFIter mfi(*pressure_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi = embedded.phi_mf[lev]->const_array(mfi);
            auto p   = pressure_mf[lev]->const_array(mfi);
            amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                if (phi(i, j, k) < 0.5) return;                    // fluid cells only
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    for (int sg = -1; sg <= 1; sg += 2)
                    {
                        const int ii = i + (d == 0 ? sg : 0), jj = j + (d == 1 ? sg : 0), kk = k + (d == 2 ? sg : 0);
                        if (phi(ii, jj, kk) >= 0.5) continue;
                        // solid neighbour on side sg: n_out (solid -> fluid) = -sg e_d
                        red[AMREX_SPACEDIM + d] += p(i, j, k) * sg * area[d];   // -p n_out dA
                    }
            });
            for (int n = 0; n < np; ++n)
            {
                amrex::IntVect iv(AMREX_D_DECL((int)std::floor((probe_x[n] - plo[0]) / DX[0]),
                                               (int)std::floor((probe_y[n] - plo[1]) / DX[1]),
                                               (int)std::floor((probe_z[n] - plo[2]) / DX[2])));
                if (bx.contains(iv)) { red[2 * AMREX_SPACEDIM + 2 * n] += p(iv); red[2 * AMREX_SPACEDIM + 2 * n + 1] += 1.0; }
            }
        }
        // [2026-09-25] sharp wall: force = momentum the fluid pushes into the (inert)
        // solid cells, from the last RHS evaluation of this step; porous: the
        // momentum removed by the implicit penalty / dt.
        for (int d = 0; d < AMREX_SPACEDIM; ++d) red[d] = embedded.wall_flux ? rhs_force[d] : pen_force[d];
        amrex::ParallelDescriptor::ReduceRealSum(red.data() + AMREX_SPACEDIM, (int)red.size() - AMREX_SPACEDIM);
        amrex::ParallelDescriptor::ReduceRealSum(red.data(), AMREX_SPACEDIM);
        if (amrex::ParallelDescriptor::IOProcessor())
        {
            const std::string fn = plot_file + "_forces.dat";
            static bool header = false;
            std::ofstream fo(fn, std::ios::app);
            if (!header)
            {
                fo << "# step lev time";
                for (int d = 0; d < AMREX_SPACEDIM; ++d) fo << " Ftot_" << d;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) fo << " Fpres_" << d;
                for (int n = 0; n < np; ++n)
                {
                    fo << " p_probe" << n << "(" << probe_x[n] << "," << probe_y[n];
#if AMREX_SPACEDIM == 3
                    fo << "," << probe_z[n];
#endif
                    fo << ")";
                }
                fo << "\n"; header = true;
            }
            char b[64];
            fo << step_counter[lev] << " " << lev;
            snprintf(b, 64, " %.10e", time + dt); fo << b;
            for (int d = 0; d < 2 * AMREX_SPACEDIM; ++d) { snprintf(b, 64, " %.10e", red[d]); fo << b; }
            for (int n = 0; n < np; ++n)
            {
                const Set::Scalar c = red[2 * AMREX_SPACEDIM + 2 * n + 1];
                snprintf(b, 64, " %.10e", c > 0.0 ? red[2 * AMREX_SPACEDIM + 2 * n] / c : std::nan(""));
                fo << b;
            }
            fo << "\n";
        }
    }



    // ------------------------------------------------------------
    // Interface Sharpenging
    // ------------------------------------------------------------

    // ------------------------------------------------------------
    // Mixed Fields
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();

        Set::Patch<Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> rho = density_mf.Patch(lev, mfi);

        const int symc = cen_face_any;   // symmetry + RADIAL faces
        const bool symlo_c[3] = {cen_face_lo[0], cen_face_lo[1], cen_face_lo[2]};
        const bool symhi_c[3] = {cen_face_hi[0], cen_face_hi[1], cen_face_hi[2]};
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, domain);
            if (symc)        // mirrored ghosts -> central is exact (see Parse)
            {
                const amrex::Dim3 dlo = amrex::lbound(domain), dhi = amrex::ubound(domain);
                const int ijk[3] = {i, j, k}, lo3[3] = {dlo.x, dlo.y, dlo.z}, hi3[3] = {dhi.x, dhi.y, dhi.z};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    if ((symlo_c[d] && ijk[d] == lo3[d]) || (symhi_c[d] && ijk[d] == hi3[d]))
                        sten[d] = Numeric::StencilType::Central;
            }

            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta_new(i, j, k) = std::max(0.0, std::min(1.0, eta_new(i, j, k)));

            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Advance(): Conservative Variable Check", {
                { "eta_new", eta_new(i, j, k) },
                { "rho_eta0", rho_eta0(i, j, k) },
                { "rho_eta1", rho_eta1(i, j, k) },
                { "rho", rho(i, j, k) }
            });
        });
    } // end rho, eta solver loop

    // ------------------------------------------------------------
    // Compute CFL for next time step on this level
    // ------------------------------------------------------------
    Set::Scalar c_max_local = 0.0;
    Set::Scalar vx_max_local = 0.0;
    Set::Scalar vy_max_local = 0.0;
    Set::Scalar vz_max_local = 0.0;
    Set::Scalar F_max_local = 0.0;
    Set::Scalar rho_min_local = 1e10;
    // Max LOCAL kinematic viscosity nu = mu(eta)/rho.  Reduced exactly rather
    // than estimated as mu_max/rho_min: that pairing takes the liquid's mu with
    // the gas's rho, two values that never occur in the same cell, and for the
    // standard setup (mu 0.15/0.015, rho 10/1) it overestimates nu by 10x --
    // costing a factor of ~6 in dt once the capillary limit takes over.
    Set::Scalar nu_max_local = 0.0;
    Set::Scalar nuth_max_local = 0.0;   // conduction diffusivity alone (thermal.dt_limit)
    const int th_cond_l = (thermal_conduction == 2) && !thermal_subcycle;   // per-phase conduction (mode 2, explicit in the RHS); mode 1 is bounded below
    const Set::Scalar th_amin_l = thermal_alpha_min, th_k0_l = thermal_k0, th_k1_l = thermal_k1;
    const Set::Scalar th_cv0_l = eos0.Cv(), th_cv1_l = eos1.Cv();
    // Max LOCAL kinematic diffusivity of the SHELL DILATATIONAL stress,
    //     nu_s = kappa_eff * |grad eta| / rho.
    // Reduced exactly, for the same reason nu_max is: the surface stress is
    // identically zero where |grad eta| = 0, i.e. throughout the gas interior
    // that supplies rho_min, so pairing kappa_s with the global rho_min takes
    // a density and a gradient that never occur in the same cell.  Measured on
    // Sch20-Collapsing that pairing was 1000x too conservative -- it drove
    // dt[0] to 2.0e-15 instead of ~2.2e-12.
    Set::Scalar nus_max_local = 0.0;
    const Set::Scalar kap_eff_l = std::abs(shell_kappa_s - shell_mu_s)
                                + 2.0 * std::abs(shell_mu_s);

    amrex::ReduceOps<amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax,
                     amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMax, amrex::ReduceOpMin,
                     amrex::ReduceOpMax> cfl_reduce_op;
    amrex::ReduceData<Set::Scalar, Set::Scalar, Set::Scalar, Set::Scalar,
                      Set::Scalar, Set::Scalar, Set::Scalar, Set::Scalar,
                      Set::Scalar> cfl_reduce_data(cfl_reduce_op);
    using CFLTuple = typename decltype(cfl_reduce_data)::Type;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
    
        Set::Patch<const Set::Scalar> eta_new = eta_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> eta = eta_old_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> etadot = etadot_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> grad_eta_ = grad_eta_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> n_hat_ = n_hat_mf.Patch(lev, mfi);
    
        Set::Patch<const Set::Scalar> rho_eta0 = rho_eta0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho_eta1 = rho_eta1_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> rho = density_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E_vol = energy_per_vol_mf.Patch(lev, mfi);
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
        Set::Patch<const Set::Scalar> Source = Source_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> gammaf = gamma_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> p0_eff = p0_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> mu_chem_ = mu_chem_mf.Patch(lev, mfi);
        Set::Patch<Set::Scalar> Bm = Bm_mf.Patch(lev, mfi);
        // 6-eq per-phase internal energies (canonical primaries)
        Set::Patch<const Set::Scalar> E0_p = energy0_mf.Patch(lev, mfi);
        Set::Patch<const Set::Scalar> E1_p = energy1_mf.Patch(lev, mfi);

        // Capillary-energy closure inputs (see Hydro2.H).  NOTE: this loop
        // aliases `eta` to eta_OLD; rho E here is current, so e_cap must be
        // built from eta_new.
        const int marm_p = marmottant;
        const Set::Scalar sig_p = sigma, chi_p = marmottant_chi;
        const Set::Scalar Gb_p = marmottant_Gamma_buck;
        const Set::Scalar sbrk_p = marmottant_sigma_break, sigw_p = sigma;
        amrex::Array4<const Set::Scalar> const &shell_p = shell_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> const cfun_p  = eta_mf[lev]->const_array(mfi);

        // EMBEDDED SOLID indicator (empty Array4 when feature is off).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);

        // Local EOS Copy
        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_base = eos1;
        
        const Species::Eos spe_l = species.eos();     // species.eos = mixture: gamma, c_v of each phase per cell
        amrex::Array4<const Set::Scalar> sp_l; if (spe_l.on) sp_l = species.mf[lev]->const_array(mfi);
        auto cf_l = cfun_mf[lev]->const_array(mfi);     // alpha_ref of the compaction model
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;
        // Local copies: class members are not addressable inside a GPU lambda.
        const Set::Scalar mu0_l = mu0, mu1_l = mu1, small_l = small;
        const int symc = cen_face_any;   // symmetry + RADIAL faces
        const bool symlo_c[3] = {cen_face_lo[0], cen_face_lo[1], cen_face_lo[2]};
        const bool symhi_c[3] = {cen_face_hi[0], cen_face_hi[1], cen_face_hi[2]};
        cfl_reduce_op.eval(bx, cfl_reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> CFLTuple
        {
            const Solver::EOS::Tammann eos0_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 0, eos0_base, E0_p(i, j, k) / std::max(rho_eta0(i, j, k), 1.0e-300), rho_eta0(i, j, k) / std::min(std::max(eta_new(i, j, k), 1.0e-12), 1.0 - 1.0e-12)) : eos0_base;
            const Solver::EOS::Tammann eos1_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 1, eos1_base, E1_p(i, j, k) / std::max(rho_eta1(i, j, k), 1.0e-300), rho_eta1(i, j, k) / (1.0 - std::min(std::max(eta_new(i, j, k), 1.0e-12), 1.0 - 1.0e-12))) : eos1_base;
            const Set::Scalar gam0_ = eos0_local.Gamma(), gam1_ = eos1_local.Gamma();
            const Set::Scalar pi1_ = eos1_local.P0();
            const Set::Scalar pi0_ = eos0_local.P0() + ((elastic.on && elastic.comp_on) ? elastic.compaction(elastic.matAt(sp_l, i, j, k), std::min(std::max(eta_new(i, j, k), 0.0), 1.0), rho_eta0(i, j, k), cf_l(i, j, k)).Pc / gam0_ : 0.0);
            auto sten = Numeric::GetStencil(i, j, k, domain);
            if (symc)        // mirrored ghosts -> central is exact (see Parse)
            {
                const amrex::Dim3 dlo = amrex::lbound(domain), dhi = amrex::ubound(domain);
                const int ijk[3] = {i, j, k}, lo3[3] = {dlo.x, dlo.y, dlo.z}, hi3[3] = {dhi.x, dhi.y, dhi.z};
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    if ((symlo_c[d] && ijk[d] == lo3[d]) || (symhi_c[d] && ijk[d] == hi3[d]))
                        sten[d] = Numeric::StencilType::Central;
            }

            Set::Vector grad_eta = Numeric::Gradient(eta_new, i, j, k, 0, DX);
            Set::Scalar grad_eta_mag = grad_eta.lpNorm<2>();
            Set::Matrix hess_eta = Numeric::Hessian(eta_new, i, j, k, 0, DX, sten);
            Set::Scalar lap_eta = Numeric::Laplacian(eta_new, i, j, k, 0, DX);

            gammaf(i, j, k) = Solver::EOS::EOS::MixedGamma(eta(i, j, k), eos0_local, eos1_local);

            etadot(i,j,k) = (eta_new(i,j,k) - eta(i,j,k)) / dt;

            // Limiting Velocity
            Set::Scalar u_limit = 1e8;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                v(i, j, k, d) = M(i, j, k, d) / (rho(i, j, k));
                v(i, j, k, d) = (v(i, j, k, d) < 0.0) ? std::max(v(i, j, k, d), -u_limit) : std::min(v(i, j, k, d), u_limit);
            }

            KE_vol(i,j,k) = 0.5 * rho(i,j,k) * (AMREX_D_TERM(v(i,j,k,0) * v(i,j,k,0), + v(i,j,k,1) * v(i,j,k,1), + v(i,j,k,2) * v(i,j,k,2)));
            KE_mas(i,j,k) = 0.5 * (AMREX_D_TERM(v(i,j,k,0) * v(i,j,k,0), + v(i,j,k,1) * v(i,j,k,1), + v(i,j,k,2) * v(i,j,k,2)));

            // Debit the capillary energy: with the closure on, rho E carries
            // e_cap, so the internal energy is rho E - KE - e_cap.
            UE_vol(i,j,k) = E_vol(i,j,k) - KE_vol(i,j,k);
            UE_vol(i, j, k) = (UE_vol(i, j, k) < 0.0) ? small : UE_vol(i, j, k);
            E_mas(i,j,k) = E_vol(i,j,k) / (rho(i,j,k) + small);
            UE_mas(i,j,k) = E_mas(i,j,k) - KE_mas(i,j,k);

            // 6-eq mixture pressure and frozen sound speed.
            // p   = alpha_1 p_1 + alpha_2 p_2          (Sch20 eq. 8)
            // c^2 = Y_1 c_1^2  + Y_2 c_2^2             (Sch20 eq. 17)
            {
                Set::Scalar a1 = std::min(std::max(eta_new(i, j, k), alpha_floor), 1.0 - alpha_floor);
                Set::Scalar a2 = 1.0 - a1;
                Set::Scalar arh0 = std::max(rho_eta0(i, j, k), small);
                Set::Scalar arh1 = std::max(rho_eta1(i, j, k), small);
                Set::Scalar r0p  = arh0 / a1;
                Set::Scalar r1p  = arh1 / a2;
                Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_p(i, j, k), a1, gam0_, pi0_, small);
                Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_p(i, j, k), a2, gam1_, pi1_, small);
                // TRACE-PHASE SLAVING: recovering a vanished phase's pressure
                // from E_k/alpha_k is ill-conditioned by gam*pi/p (~2.4e4 for
                // water at 1 bar) -- any alpha inconsistency becomes ~1e5 Pa of
                // garbage that the limiter reconstructs across the band tail
                // (the residual secular leak after the alpha-floor fix).  A
                // phase below the eta cutoff is dynamically irrelevant: slave
                // its pressure to the dominant phase (mechanical equilibrium).
                {
                    // SMOOTH slaving: a hard threshold leaves a discontinuous
                    // seam in the p_k field at the switch contour, which the
                    // limiter reads as a real pressure feature (measured: a
                    // persistent -30% dip parked at the eta = 1e-2 contour
                    // pumping gas from the bubble center into the band).
                    // Blend raw -> dominant over alpha in [SLAVE_LO, SLAVE_HI]
                    // with a C1 smoothstep: no seam, fully slaved below 1%,
                    // fully raw above 10%.
                    const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                    Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                    w0 = w0 * w0 * (3.0 - 2.0 * w0);
                    Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                    w1 = w1 * w1 * (3.0 - 2.0 * w1);
                    const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                    p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                    p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
                }
                Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(r0p, p0_loc, gam0_, pi0_, small);
                Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(r1p, p1_loc, gam1_, pi1_, small);
                Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
                Set::Scalar Y1 = 1.0 - Y0;
                press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
                a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);
                p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local); // diagnostic only
            }

            Set::Scalar f_prime = 4.0 * eta_new(i,j,k) * (eta_new(i,j,k) - 0.5) * (eta_new(i,j,k) - 1.0);
            Set::Scalar mu_chem = -epsilon * epsilon * lap_eta + f_prime;
            mu_chem_(i,j,k) = mu_chem;

            // Spalding number from the vapor MASS fraction Y = (alpha rho)_0 / rho
            // (same definition as RHS); the old eta/(1-eta) volume-fraction ratio
            // was not the Spalding B_M and clobbered the RHS diagnostic.
            Bm(i,j,k) = SpaldingBM(rho_eta0(i,j,k) / std::max(rho(i,j,k), small), Y_infinity, small);

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
        
            if (nan_check) check4nans(time, lev, i, j, k, "ERROR IN Advance(): Visualization", { 
                {"eta_new", eta_new(i,j,k)},
                {"rho_eta0", rho_eta0(i,j,k)},
                {"rho_eta1", rho_eta1(i,j,k)},
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
        
            // Track CFL quantities.  Skip solid cells (their frozen state must
            // not drive the global timestep).
            // [OPT] Returned as a ReduceOps tuple instead of accumulated into
            // by-reference captures (thread-safe; same max/min result).  NaN
            // entries are mapped to the identity to reproduce std::max's old
            // behaviour of silently skipping them.
            const bool is_fluid = (!embedded.apply) || (phisol(i, j, k) > embedded.relax_skip);
            Set::Scalar t_c = 0.0, t_vx = 0.0, t_vy = 0.0, t_vz = 0.0, t_F = 0.0, t_nu = 0.0, t_nus = 0.0, t_nuth = 0.0;
            Set::Scalar t_rho = 1e10;
            if (is_fluid)
            {
                t_c  = a(i,j,k);
                if (elastic.on)   // longitudinal elastic speed (Favrie 2009 Sec. 3.5)
                {
                    const Set::Scalar a0_ = std::min(std::max(eta_new(i, j, k), 0.0), 1.0);
                    t_c = std::sqrt(t_c * t_c + elastic.c2(elastic.matAt(sp_l, i, j, k), a0_, rho_eta0(i, j, k) + rho_eta1(i, j, k)));
                }
                t_vx = std::abs(v(i,j,k,0));
                t_vy = std::abs(v(i,j,k,1));
#if AMREX_SPACEDIM == 3
                t_vz = std::abs(v(i,j,k,2));
#endif
                t_F = sqrt(Source(i,j,k,1) * Source(i,j,k,1) +
                           Source(i,j,k,2) * Source(i,j,k,2));
                t_rho = rho(i,j,k);
                {
                    const Set::Scalar a1n = std::min(std::max(eta_new(i,j,k), 0.0), 1.0);
                    // [2026-09-25] the solid.mu_factor viscosity is deliberately NOT
                    // included here (user request): the timestep follows the fluid
                    // viscosity only, as in single-phase Hydro.
                    const Set::Scalar mu_c = a1n * mu0_l + (1.0 - a1n) * mu1_l;
                    t_nu = mu_c / (rho(i,j,k) + small_l);
                    // thermal diffusivity k_k / (rho_k cv_k) of each conducting phase (thermal.conduction = 2)
                    if (th_cond_l)
                    {
                        if (a1n > th_amin_l)
                            t_nuth = std::max(t_nuth, th_k0_l * a1n / (std::max(rho_eta0(i,j,k), small_l) * th_cv0_l));
                        if (1.0 - a1n > th_amin_l)
                            t_nuth = std::max(t_nuth, th_k1_l * (1.0 - a1n) / (std::max(rho_eta1(i,j,k), small_l) * th_cv1_l));
                        t_nu = std::max(t_nu, t_nuth);
                    }
                    if (kap_eff_l > 0.0)
                        t_nus = kap_eff_l * grad_eta_mag / (rho(i,j,k) + small_l);
                }
            }
            auto mx = [](Set::Scalar x) { return (x == x) ? x : 0.0; };
            return { mx(t_c), mx(t_vx), mx(t_vy), mx(t_vz), mx(t_F), mx(t_nu), mx(t_nus),
                     (t_rho == t_rho) ? t_rho : 1e10, mx(t_nuth) };
        });
    } // end Mixed Fields loop

    // Parallel Reduction.  [OPT] One packed Allreduce instead of eight
    // (the min is folded in as the max of -rho; negation is exact).
    {
        CFLTuple hv = cfl_reduce_data.value(cfl_reduce_op);
        Set::Scalar red[9] = {
            std::max(c_max_local,   amrex::get<0>(hv)),
            std::max(vx_max_local,  amrex::get<1>(hv)),
            std::max(vy_max_local,  amrex::get<2>(hv)),
            std::max(vz_max_local,  amrex::get<3>(hv)),
            std::max(F_max_local,   amrex::get<4>(hv)),
            std::max(nu_max_local,  amrex::get<5>(hv)),
            std::max(nus_max_local, amrex::get<6>(hv)),
            -std::min(rho_min_local, amrex::get<7>(hv)),
            std::max(nuth_max_local, amrex::get<8>(hv)) };
        amrex::ParallelDescriptor::ReduceRealMax(red, 9);
        c_max_local   = red[0];
        vx_max_local  = red[1];
        vy_max_local  = red[2];
        vz_max_local  = red[3];
        F_max_local   = red[4];
        nu_max_local  = red[5];
        nus_max_local = red[6];
        rho_min_local = -red[7];
        nuth_max_local = red[8];
    }

    c_max = c_max_local;
    vx_max = vx_max_local;
    vy_max = vy_max_local;
    vz_max = vz_max_local;
    F_max = F_max_local;
    rho_min = rho_min_local;
    nu_max  = nu_max_local;

    // Computing dt for next time step on all levels
    Set::Scalar dx_min = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});

    // ACOUSTIC.  Unsplit (method-of-lines) form, summed over directions:
    //     dt = cfl / sum_d (c + |u_d|) / dx_d.
    // The old 1D form cfl dx / (c + |u|) has no dimension factor: the
    // checkerboard (pi,pi,pi) mode gets upwind damping ~2 d c/dx, and RK3
    // (real-axis limit 2.51) is then stable only for c dt/dx <~ 1.25/d, i.e.
    // ~0.42 in 3D.  At cfl 0.5 the Mili 3D runs sat at ~0.45 and grew a liquid
    // rho/E/p checkerboard (CollapseDebug, 2026-11-09).  In 1D this is identical
    // to the old form; for equal dx and u = 0 it is cfl dx / (d c).  (Was the
    // opt-in cfl_dimensional on DropletDetonation; always on since the merge.)
    Set::Scalar acoustic_rate = AMREX_D_TERM((c_max + vx_max) / DX[0],
                                           + (c_max + vy_max) / DX[1],
                                           + (c_max + vz_max) / DX[2]);
    Set::Scalar dt_acoustic = cfl / (acoustic_rate + small);

    // VISCOUS (explicit diffusion).  The stability limit for a (2d+1)-point
    // Laplacian in d dimensions is dt <= dx^2 / (2 d nu), nu = mu/rho.  The
    // previous form omitted the 2d, so with the default cfl_v = cfl = 0.3 it was
    // ~20%% too permissive in 2D (limit coefficient 0.25) and 6x too permissive
    // in 3D.  That is why mu0=mu1=0.05 blew up immediately at eps/dx=4 -- dt
    // stayed at the acoustic 6.9e-7 against a true gas-phase limit of 3.8e-7,
    // Total_Force reached -3.7e7 and dt collapsed to the 1e-12 floor.
    // rho_min with mu_max is the conservative pairing: the binding phase is the
    // one with the largest nu = mu/rho.
    Set::Scalar mu_max = std::max(mu0, mu1);
    // The diagnostic artificial viscosity mu_art = C rho a dx w has kinematic
    // value nu_art = C a dx w <= C c_max dx_min, independent of rho, so it adds
    // directly to nu.  Without this the C > ~1 runs would blow up on a stale
    // timestep rather than on the physics under test.
    // nu_max is the exact max of mu(eta)/rho over the domain (reduced above).
    // Fall back to the old conservative pairing if it was never set.
    Set::Scalar nu_phys = (nu_max > 0.0) ? nu_max : (mu_max / (rho_min + small));
    // visc_subcycle: the bulk viscosity is sub-cycled (SubcycleBulkViscous) and no longer limits the global step.
    Set::Scalar nu_total = visc_subcycle ? 0.0 : nu_phys;
    // Heat conduction: the internal-energy diffusivity is k/(rho cv) = gamma nu / Pr.  It is never sub-cycled, so it
    // stays in the limit with visc_subcycle too (flames2 thermal.dt_limit: dropping it gave NaN on the axis).
    // thermal.conduction = 2 (per-phase): conduction is never sub-cycled, so it stays in the limit
    // (thermal.dt_limit = 1, default).
    if (visc_subcycle && thermal_dt_limit && thermal_conduction == 2) nu_total = nuth_max_local;
    if (thermal_conduction == 1 && thermal_k0 < 0.0)
        nu_total = nu_phys * std::max(1.0, std::max(eos0.Gamma(), eos1.Gamma()) / thermal_Pr);
    else if (thermal_conduction == 1)    // thermal.k0/k1: diffusivity k/(rho cv), lowest density with the largest k (conservative)
        nu_total = std::max(nu_total, std::max(thermal_k0, thermal_k1) / ((rho_min + small) * std::min(eos0.Cv(), eos1.Cv())));
    // in-depth conduction of the ablating solid (explicit): alpha_s = k_s / (rho_s cp_s)
    if (ablation.on && ablation.conduction)
        nu_total = std::max(nu_total, ablation.k_s / (ablation.rho_s * ablation.cp_s));
    // SHELL DILATATIONAL VISCOSITY.  kappa_s enters the momentum equation as
    //     div( |grad eta| * kappa_s * (div_s u) * P ),
    // i.e. a SURFACE momentum diffusion whose kinematic diffusivity is
    //     nu_s = kappa_s * |grad eta| / rho  ~  kappa_s / (rho * eps),
    // eps being the interface width.  Because eps is held at ~1 cell
    // (eps/dx = 1.02 on the Sch20 decks), |grad eta| ~ 1/dx_min and nu_s
    // therefore GROWS as the mesh is refined:
    //     nu_s ~ 1/dx   =>   dt_visc ~ dx^2 / nu_s ~ dx^3,
    // while the acoustic limit only scales as dx.  Refinement makes this the
    // binding constraint even though it is invisible at coarse resolution.
    //
    // Measured on Sch20-Collapsing (kappa_s = 7.2e-9, rho = 1000):
    //   level  dx         nu_s       dt_limit   substep    violation
    //   L3     7.81e-8    9.0e-5     4.52e-12   1.16e-11     2.6x
    //   L4     3.91e-8    1.80e-4    5.65e-13   4.11e-12     7.3x
    //   L5     1.95e-8    3.60e-4    7.06e-14   1.45e-12    20.6x  -> NaN at step 3
    // nu_s at L5 is 360x the bulk mu0/rho, so omitting it from nu_total made
    // dt_viscous meaningless for any coated run.  1/dx_min is used as the
    // bound on |grad eta| because the band cannot be thinner than a cell.
    // nus_max_local is the exact max of kappa_eff*|grad eta|/rho, reduced over
    // the whole level and taken only where the surface stress actually lives.
    // When the viscous stress is sub-cycled it no longer constrains the GLOBAL
    // step -- SubcycleShellViscous takes N = dt/dt_visc sub-steps internally --
    // so the constraint is dropped here and enforced there instead.
    if (kap_eff_l > 0.0 && !(shell_visc_subcycle && shell_kappa_s != 0.0))
        nu_total += nus_max_local;
    Set::Scalar dt_viscous = cfl_v * dx_min * dx_min
                           / (2.0 * AMREX_SPACEDIM * (nu_total + small));

    // CAPILLARY (Brackbill, Kothe & Zemach, JCP 100 (1992) 335).  An explicit
    // surface-tension force cannot resolve capillary waves faster than
    //     dt <= sqrt(rho_bar dx^3 / (2 pi sigma)).
    // This scales as dx^1.5 while the acoustic CFL scales as dx, so REFINING a
    // surface-tension problem eventually violates it no matter how small the
    // acoustic dt is.  Measured on the static R0=24mm bubble (inviscid,
    // sigma=7.28): dt_acoustic/dt_cap = 0.70 / 0.99 / 1.40 at eps/dx = 2 / 4 / 8,
    // tracking the observed parasitic-current growth 3365 / 9630 / 103169 /s.
    // NOTE this constraint alone does NOT cure those currents -- capping dt at
    // 0.25 dt_cap cut the growth rate only 33%% (eps/dx=4) and 9%% (eps/dx=8).
    // It is a correctness fix; the residual grid-aligned (n=4) force imbalance is
    // a separate, unsolved problem.
    // sigma is the largest tension the capillary term can apply: with
    // marmottant=1 the elastic branch is clamped at sigma_break -> sigma_water,
    // which is this same `sigma`.
    Set::Scalar dt_capillary = 1.0e30;
    if (apply_surface_tension && sigma > 0.0)
        dt_capillary = cfl_v * std::sqrt(rho_min * dx_min * dx_min * dx_min
                                       / (2.0 * Set::Constant::Pi * sigma));

    Set::Scalar a_max = F_max / (rho_min + small);
    Set::Scalar dt_force = cfl_v * sqrt(dx_min / (a_max + small));

    Set::Scalar Mob = 0.01 * dx_min * dx_min;
    Set::Scalar dt_allen_cahn = 0.5 * dx_min * dx_min / (Mob + small);

    Set::Scalar dt_max = std::min({ dt_acoustic, dt_viscous, dt_force, dt_allen_cahn, dt_capillary });
    dt_max = dt_max * 0.9;

    // dt_diag > 0: report the limits every dt_diag steps (level 0)
    if (dt_diag > 0 && lev == 0 && (step_counter[lev] % dt_diag == 0))
    {
        Util::Message(INFO, "=== CFL DIAGNOSTICS Level ", lev, " ===");
        Util::Message(INFO, "  c_max = ", c_max, " m/s");
        Util::Message(INFO, "  vx_max = ", vx_max, " m/s");
        Util::Message(INFO, "  vy_max = ", vy_max, " m/s");
        Util::Message(INFO, "  vz_max = ", vz_max, " m/s");
        Util::Message(INFO, "  dt_acoustic  = ", dt_acoustic,  " s");
        Util::Message(INFO, "  dt_viscous   = ", dt_viscous,   " s");
        Util::Message(INFO, "  dt_capillary = ", dt_capillary, " s");
        Util::Message(INFO, "  dt_force     = ", dt_force, " s   nu_phys = ", nu_phys, (visc_subcycle ? "  (visc_subcycle: viscous limit off)" : ""));
        Util::Message(INFO, "  dt_max = ", dt_max, " s");
    }

    if (dynamictimestep.on)
    {
        this->DynamicTimestep_SyncTimeStep(lev, dt_max);
    }

} 


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////// EMBEDDED-SOLID TARGET DERIVE //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// Build the per-phase embedded-solid target (density0/1_mf, energy0/1_mf,
// momentum_mf) from the SINGLE-phase solid input (density_ic, pressure_ic,
// momentum_ic), splitting mass + energy by the LOCAL eta (option 3a) and using
// the stiffened-gas EOS for the per-phase internal energies:
//   rho_eta0 = eta*rho_s ,  rho_eta1 = (1-eta)*rho_s
//   E_k      = alpha_k*(p_s + gamma_k*pi_k)/(gamma_k - 1)
// Called at IC time and after every regrid.  Requires eta_mf populated first.
void Hydro2::InitEmbeddedSolidTarget(int lev, Set::Scalar time)
{
    if (!embedded.apply) return;

    // Single-phase solid inputs.  Stage the total density into the phase-0
    // density slot and the pressure into the phase-0 energy slot as scratch,
    // then split per-cell (read into locals before overwriting).
    embedded.momentum_ic->Initialize(lev, embedded.momentum_mf, time);
    embedded.density_ic ->Initialize(lev, embedded.density0_mf, time);   // scratch: rho_s
    embedded.pressure_ic->Initialize(lev, embedded.energy0_mf, time);    // scratch: p_s

    const Set::Scalar g0 = eos0.Gamma(), pi0 = eos0.P0();
    const Set::Scalar g1 = eos1.Gamma(), pi1 = eos1.P0();
    const Set::Scalar sm = small;
    for (amrex::MFIter mfi(*embedded.density0_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.growntilebox(nghost);
        auto eta = eta_mf[lev]->array(mfi);
        auto d0  = embedded.density0_mf[lev]->array(mfi);
        auto d1  = embedded.density1_mf[lev]->array(mfi);
        auto e0  = embedded.energy0_mf[lev]->array(mfi);
        auto e1  = embedded.energy1_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar rho_s = d0(i, j, k);   // scratch (total solid density)
            const Set::Scalar p_s   = e0(i, j, k);   // scratch (solid pressure)
            const Set::Scalar a1 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a2 = 1.0 - a1;
            d0(i, j, k) = a1 * rho_s;
            d1(i, j, k) = a2 * rho_s;
            e0(i, j, k) = a1 * (p_s + g0 * pi0) / std::max(g0 - 1.0, sm);
            e1(i, j, k) = a2 * (p_s + g1 * pi1) / std::max(g1 - 1.0, sm);
        });
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// MOVING RIGID SOLID (solid.moving = 1) -- prescribed motion, sharp embedded wall.
//
// Called at the start of Advance(lev, time, dt), i.e. once per level step, with the
// state at `time`:
//   1. phi_old <- phi;  phi <- solid.phi.ic(x, time)   (analytic: no advection error,
//      the body keeps its shape exactly; the expression must carry the motion, and be
//      periodic if the domain is)
//   2. solid target (density, pressure, momentum -> wall velocity u_s) at `time`
//   3. FRESH CELLS (phi_old < 0.5 <= phi: solid a step ago, fluid now).  They hold the
//      inert solid state, not a fluid one, so every state field is copied from the
//      face/edge neighbour that was fluid before AND is fluid now, the one deepest in
//      the fluid (largest phi).  One donor, all fields: the fresh cell gets a
//      thermodynamically consistent state (first order in space -- the standard
//      ghost-cell / cut-cell "fresh cell" treatment).  CFL keeps the wall from
//      crossing more than one cell per step, so a donor always exists.
//   4. DEAD CELLS (fluid -> solid) need nothing: they become inert and the Brinkman
//      term relaxes them to the solid target.
// The method is NOT conservative: the fluid in dead cells is deleted and fresh cells
// are created (compensated on average by the wall-face flux rho* u_s.n of the
// moving-wall Riemann problem).  Track the mass drift in any validation.
///////////////////////////////////////////////////////////////////////////////////////////////////////
// ABLATION: recede the embedded surface, d(phi)/dt = v_abl |grad phi| (phi = 1 fluid, 0 solid).
// v_abl = accel q_w / (rho_s Q*) on the wall-adjacent fluid cells (q_w from the last RHS
// evaluation), extended to the neighbourhood by averaging; first-order upwind |grad phi|.
void Hydro2::RecedeAblatingSurface(int lev, Set::Scalar time, Set::Scalar dt)
{
    amrex::MultiFab &q = *ablation.q_mf[lev], &v = *ablation.v_mf[lev];
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar coef = (time >= ablation.t_start) ? ablation.accel / (ablation.rho_s * ablation.Qstar) : 0.0;
    amrex::iMultiFab has(q.boxArray(), q.DistributionMap(), 1, 2);
    for (amrex::MFIter mfi(q, false); mfi.isValid(); ++mfi)
    {
        auto qa = q.const_array(mfi); auto va = v.array(mfi); auto ha = has.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            ha(i, j, k) = (qa(i, j, k) != 0.0) ? 1 : 0;
            va(i, j, k) = coef * std::max(qa(i, j, k), 0.0);
        });
    }
    amrex::MultiFab vn(q.boxArray(), q.DistributionMap(), 1, 0);
    amrex::iMultiFab hn(q.boxArray(), q.DistributionMap(), 1, 0);
    for (int pass = 0; pass < ablation.extend; ++pass)
    {
        v.FillBoundary(geom[lev].periodicity()); has.FillBoundary(geom[lev].periodicity());
        for (amrex::MFIter mfi(q, false); mfi.isValid(); ++mfi)
        {
            auto va = v.const_array(mfi); auto ha = has.const_array(mfi); auto vo = vn.array(mfi); auto ho = hn.array(mfi);
            const amrex::Box gb = v[mfi].box();
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                vo(i, j, k) = va(i, j, k); ho(i, j, k) = ha(i, j, k);
                if (ha(i, j, k)) return;
                Set::Scalar sum = 0.0; int n = 0;
#if AMREX_SPACEDIM == 3
                for (int c = -1; c <= 1; ++c)
#else
                for (int c = 0; c <= 0; ++c)
#endif
                for (int b = -1; b <= 1; ++b)
                for (int a = -1; a <= 1; ++a)
                    if (gb.contains(i + a, j + b, k + c) && ha(i + a, j + b, k + c)) { sum += va(i + a, j + b, k + c); ++n; }
                if (n > 0) { vo(i, j, k) = sum / n; ho(i, j, k) = 1; }
            });
        }
        amrex::MultiFab::Copy(v, vn, 0, 0, 1, 0);
        amrex::iMultiFab::Copy(has, hn, 0, 0, 1, 0);
    }
    embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
    amrex::MultiFab pn(q.boxArray(), q.DistributionMap(), 1, 0);
    Set::Scalar cmax = 0.0;
    for (amrex::MFIter mfi(q, false); mfi.isValid(); ++mfi)
    {
        auto va = v.const_array(mfi); auto ph = embedded.phi_mf[lev]->const_array(mfi); auto po = pn.array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
            // phi_t - v |grad phi| = 0: the front moves toward lower phi, information comes from the
            // higher-phi (fluid) side -> Godunov upwinding with max(D-, 0)... on the opposite sides
            Set::Scalar g2 = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int a = (d == 0), b = (d == 1), c = (d == 2);
                const Set::Scalar dm = (ph(i, j, k) - ph(i - a, j - b, k - c)) / DX[d];
                const Set::Scalar dp = (ph(i + a, j + b, k + c) - ph(i, j, k)) / DX[d];
                const Set::Scalar m = std::max(std::max(-dm, 0.0), std::max(dp, 0.0));
                g2 += m * m;
            }
            po(i, j, k) = std::min(std::max(ph(i, j, k) + dt * va(i, j, k) * std::sqrt(g2), 0.0), 1.0);
            for (int d = 0; d < AMREX_SPACEDIM; ++d) cmax = std::max(cmax, va(i, j, k) * dt / DX[d]);
        });
    }
    amrex::ParallelDescriptor::ReduceRealMax(cmax);
    if (cmax > 0.5) Util::Warning(INFO, "ablation: surface moves ", cmax, " cells per step on level ", lev, " (reduce ablation.accel)");
    amrex::MultiFab::Copy(*embedded.phi_mf[lev], pn, 0, 0, 1, 0);
    embedded.phi_bc->define(geom[lev]);
    if (lev == 0)
        FillBoundariesWithBC(lev, time, embedded.phi_bc, { embedded.phi_mf[lev].get() });
    else
        // Refined level (2026-10-29): phi is not an evolving field, so the framework does not refill its
        // coarse-fine ghosts each step; take them from the (already receded) coarse level here.  The coarse phi
        // under this level is replaced by the average of this one after the fine steps (Integrator::TimeStep).
        FillPatch(lev, time, embedded.phi_mf, *embedded.phi_mf[lev], *embedded.phi_bc, 0);
}

// ABLATION, in-depth conduction (ablation.conduction): rho_s cp_s dT_s/dt = div(k_s grad T_s) in the
// solid cells (phi < 0.5), explicit Euler once per step, called at the END of the step.  Solid|solid faces:
// standard two-point flux.  Solid|fluid faces: the heat flux the gas conduction block computed (surface
// temperature from flux continuity, capped at the ablation temperature -- see Ablation.H), accumulated over
// the Runge-Kutta stages with their weights in ablation.qs_mf on the fluid cell.  The solid therefore gains
// exactly what the gas lost minus what ablation consumed: energy is conserved to round-off (checked by
// tests/FlowConjugateHeat/input_1D_ClosedBoxEquilibrium).  Adiabatic at the domain boundary.  Single level.
void Hydro2::AdvanceSolidTemperature(int lev, Set::Scalar time, Set::Scalar dt)
{
    amrex::MultiFab &Ts = *ablation.Ts_mf[lev]; amrex::MultiFab &Qs = *ablation.qs_mf[lev];
    const Set::Scalar *DX = geom[lev].CellSize();
    const amrex::Box dom = geom[lev].Domain();
    const Set::Scalar ks = ablation.k_s, rc = ablation.rho_s * ablation.cp_s;
    bool per[3] = {false, false, false};
    for (int d = 0; d < AMREX_SPACEDIM; ++d) per[d] = geom[lev].isPeriodic(d);
    ablation.Ts_bc->define(geom[lev]);
    if (lev == 0) FillBoundariesWithBC(lev, time, ablation.Ts_bc, { &Ts }); else FillPatch(lev, time, ablation.Ts_mf, Ts, *ablation.Ts_bc, 0);   // coarse-fine ghosts from the coarse level (AMR, 2026-10-29)
    embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
    Qs.FillBoundary(geom[lev].periodicity());
    amrex::MultiFab Tn(Ts.boxArray(), Ts.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(Ts, false); mfi.isValid(); ++mfi)
    {
        auto ts = Ts.const_array(mfi); auto tn = Tn.array(mfi);
        auto ph = embedded.phi_mf[lev]->const_array(mfi);
        auto qa = Qs.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            tn(i, j, k) = ts(i, j, k);
            if (!(ph(i, j, k) < 0.5)) return;
            Set::Scalar q = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int sg = -1; sg <= 1; sg += 2)
                {
                    const int a = (d == 0 ? sg : 0), b = (d == 1 ? sg : 0), c = (d == 2 ? sg : 0);
                    const int fi = i + a, fj = j + b, fk = k + c;                  // neighbour
                    if (ph(fi, fj, fk) < 0.5) { q += ks * (ts(fi, fj, fk) - ts(i, j, k)) / (DX[d] * DX[d]); continue; }
                    const amrex::IntVect fiv(AMREX_D_DECL(fi, fj, fk));
                    if (!per[d] && !dom.contains(fiv)) continue;                   // adiabatic domain boundary
                    // seen from the fluid cell this solid cell is on the side -sg of direction d
                    q += qa(fi, fj, fk, 2 * d + (sg < 0 ? 1 : 0)) / DX[d];
                }
            tn(i, j, k) = ts(i, j, k) + dt * q / rc;
        });
    }
    amrex::MultiFab::Copy(Ts, Tn, 0, 0, 1, 0);
    if (lev == 0) FillBoundariesWithBC(lev, time, ablation.Ts_bc, { &Ts }); else FillPatch(lev, time, ablation.Ts_mf, Ts, *ablation.Ts_bc, 0);   // coarse-fine ghosts from the coarse level (AMR, 2026-10-29)
}

// CHEMISTRY (chem.on): operator-split one-step reaction in the valid cells, at constant volume and phase mass.
// [2026-10-06] Default off.  Verified with the reactor and Chapman-Jouguet tests of tests/FlowChemistry.
// Reactant -> product inside phase `chem.phase`; the heat of reaction goes into that phase's energy and into rho E
// (sensible-energy convention, see Hydro2_Chem.H).  The reaction keeps sum(species) = phase mass exactly.
void Hydro2::ReactChemistry(int lev, Set::Scalar dt)
{
    if (chem.pele)      // finite-rate mechanism (Hydro2_Pele.H): constant-volume reactor in every valid cell
    {
        const Pele::Thermo th = species.pele;
        const int ph = th.ph, c0 = th.c0, max_sub = chem.max_sub;
        const Set::Scalar Tmin = chem.T_min, rtol = chem.rtol, atol = chem.atol;
        const bool solid = embedded.apply != 0;
        long nfail = 0, nsub = 0;
        for (amrex::MFIter mfi(*species.mf[lev], false); mfi.isValid(); ++mfi)
        {
            auto m = species.mf[lev]->array(mfi); auto H = chem.heat_mf[lev]->array(mfi);
            auto eta = eta_mf[lev]->const_array(mfi);
            auto Ek = (ph == 0) ? energy0_mf[lev]->array(mfi) : energy1_mf[lev]->array(mfi);
            auto E  = energy_per_vol_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> phi; if (solid) phi = embedded.phi_mf[lev]->const_array(mfi);
            const amrex::Box &vb = mfi.validbox();
            const auto lo = amrex::lbound(vb), hi = amrex::ubound(vb);
            for (int k = lo.z; k <= hi.z; ++k) for (int j = lo.y; j <= hi.y; ++j) for (int i = lo.x; i <= hi.x; ++i)
            {
                H(i, j, k) = 0.0;
                if (solid && phi(i, j, k) < 0.5) continue;
                Set::Scalar Y[Pele::NS], mph = 0.0;
                for (int s = 0; s < Pele::NS; ++s) { Y[s] = std::max(m(i, j, k, c0 + s), 0.0); mph += Y[s]; }
                if (!(mph > 1.0e-300)) continue;
                for (int s = 0; s < Pele::NS; ++s) Y[s] /= mph;
                const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12), ak = (ph == 0) ? a0 : 1.0 - a0;
                if (ak < chem.alpha_min) continue;                                   // chem.alpha_min (default 0: never)
                Set::Scalar es = Ek(i, j, k) / mph;
                if (!(es > 0.0) || th.TfromEs(es, Y) < Tmin) continue;
                const Set::Scalar es0 = es;
                const int n = th.React(Y, mph / ak, es, dt, rtol, atol, max_sub);
                if (n < 0) ++nfail;
                nsub += std::abs(n);
                for (int s = 0; s < Pele::NS; ++s) m(i, j, k, c0 + s) = Y[s] * mph;
                const Set::Scalar dq = (es - es0) * mph;        // heat released per volume = - sum dY e0
                Ek(i, j, k) += dq; E(i, j, k) += dq;
                H(i, j, k) = dq / dt;
            }
        }
        if (nfail > 0) Util::Warning(INFO, "chem (pele): the sub-step limit chem.max_sub was reached in ", nfail, " cells of this rank; their reaction step is incomplete");
        amrex::ignore_unused(nsub);
        return;
    }
    const Chem::Rate ch = chem.rate(); const int ph = chem.phase, ir = chem.ir, ip = chem.ip;
    const Chem::Rate2 ch2 = chem.rate2(); const int iox = chem.iox;
    const Set::Scalar amin_c = chem.alpha_min;
    const Set::Scalar pik = (ph == 0) ? eos0.P0() : eos1.P0(), cvk_c = (ph == 0) ? eos0.Cv() : eos1.Cv();
    const Set::Scalar gk_c = (ph == 0) ? eos0.Gamma() : eos1.Gamma();
    const Species::Eos spe = species.eos();     // species.eos = mixture: c_v of the reacting phase from its composition
    const bool solid = embedded.apply != 0;
    for (amrex::MFIter mfi(*species.mf[lev], false); mfi.isValid(); ++mfi)
    {
        auto m = species.mf[lev]->array(mfi); auto H = chem.heat_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->const_array(mfi);
        auto mk = (ph == 0) ? rho_eta0_mf[lev]->const_array(mfi) : rho_eta1_mf[lev]->const_array(mfi);
        auto Ek = (ph == 0) ? energy0_mf[lev]->array(mfi) : energy1_mf[lev]->array(mfi);
        auto E  = energy_per_vol_mf[lev]->array(mfi);
        amrex::Array4<const Set::Scalar> phi; if (solid) phi = embedded.phi_mf[lev]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            H(i, j, k) = 0.0;
            if (solid && phi(i, j, k) < 0.5) return;
            const Set::Scalar mph = mk(i, j, k), mr = m(i, j, k, ir);
            if (!(mph > 1.0e-300) || !(mr > 0.0)) return;
            const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12), ak = (ph == 0) ? a0 : 1.0 - a0;
            if (ak < amin_c) return;                                                 // chem.alpha_min (default 0: never)
            Set::Scalar cvk = cvk_c, gk = gk_c;
            if (spe.on) spe.Mix(amrex::Array4<const Set::Scalar>(m), i, j, k, ph, gk, cvk);   // c_v at the start of the step (frozen over it)
            const Set::Scalar T0 = (Ek(i, j, k) - ak * pik) / (mph * cvk);          // phase temperature (stiffened gas)
            const Set::Scalar Y0 = mr / mph;
            const Set::Scalar Y1 = (iox >= 0) ? ch2.React(Y0, std::max(m(i, j, k, iox), 0.0) / mph, mph / ak, T0, cvk, dt)
                                              : ch.React(Y0, T0, cvk, dt);
            const Set::Scalar dm = (Y0 - Y1) * mph;                                  // reactant mass consumed per volume
            if (!(dm > 0.0)) return;
            if (iox >= 0) { const Set::Scalar dmo = std::min(ch2.s * dm, std::max(m(i, j, k, iox), 0.0));
                            m(i, j, k, ir) -= dm; m(i, j, k, iox) -= dmo; m(i, j, k, ip) += dm + dmo; }
            else { m(i, j, k, ir) -= dm; m(i, j, k, ip) += dm; }
            Ek(i, j, k) += ch.Q * dm; E(i, j, k) += ch.Q * dm;
            H(i, j, k) = ch.Q * dm / dt;
        });
    }
}

// ENERGETIC MATERIAL (energetic.on): operator-split burn of the reactant species of phase 0 into the product species
// of phase 1 in the valid cells -- masses, volume and energy as described in Hydro2_Energetic.H.  The pressure used by
// the ignition switch and the rate is the mixture pressure of the last primitive update.
void Hydro2::ReactEnergetic(int lev, Set::Scalar dt)
{
    const int ir = energetic.ir, ip = energetic.ip, law = energetic.law;
    const Set::Scalar kr = energetic.k, Q = energetic.Q, p_ign = energetic.p_ign, p_ref = energetic.p_ref, pn = energetic.n;
    const Species::Eos spe = species.eos();
    for (amrex::MFIter mfi(*species.mf[lev], false); mfi.isValid(); ++mfi)
    {
        auto m = species.mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto m0 = rho_eta0_mf[lev]->array(mfi); auto m1 = rho_eta1_mf[lev]->array(mfi);
        auto E0 = energy0_mf[lev]->array(mfi); auto E1 = energy1_mf[lev]->array(mfi);
        auto E  = energy_per_vol_mf[lev]->array(mfi);
        auto p  = pressure_mf[lev]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar mr = m(i, j, k, ir), mp = std::max(m(i, j, k, ip), 0.0), mph = m0(i, j, k);
            // no reactant to speak of (tails of the initial profile inside an inert solid): nothing to burn
            if (!(mr > 1.0e-12 * mph) || !(mph > 1.0e-300) || !(p(i, j, k) > p_ign)) return;
            const Set::Scalar lam0 = mr / (mr + mp), lam1 = Energetic::Step(law, lam0, p(i, j, k), kr, p_ref, pn, dt);
            if (!(lam0 > 1.0e-12)) return;                              // burnt out: leave the last trace, do not drive phase 0 to a denormal
            // never empty phase 0 of a cell completely: the last 1e-9 of it burns in the following steps
            const Set::Scalar dm = std::min((lam0 - lam1) * (mr + mp), std::min(mr, mph * (1.0 - 1.0e-9)));
            if (!(dm > 0.0)) return;
            // share of the VOLUME and of the ENERGY of phase 0 that the reacting mass takes with it.  One species
            // equation of state in the phase: its mass share.  Species with their own stiffness (species.phase0.pinf,
            // a confiner sharing the phase): the reactant's OWN volume z_r (reference-volume closure) and its own
            // energy z_r (p + gamma_r pi_r)/(gamma_r - 1) at the pressure of the phase -- the mass share of the
            // phase energy would hand the cold energy of the confiner to the products in the mixed cells.
            Set::Scalar fv = dm / mph, fe = fv;
            if (spe.on && spe.has_pi[0])
            {
                Set::Scalar sv = 0.0, a = 0.0, b = 0.0;
                for (int s_ = 0; s_ < spe.n0; ++s_)
                {
                    const Set::Scalar vi = std::max(m(i, j, k, s_), 0.0) / spe.rr[s_];
                    sv += vi; a += vi / (spe.g[s_] - 1.0); b += vi * spe.g[s_] * spe.pi[s_] / (spe.g[s_] - 1.0);
                }
                const Set::Scalar a0 = std::max(eta(i, j, k), 1.0e-300), vr = mr / spe.rr[ir];
                const Set::Scalar pk = (E0(i, j, k) / a0 * sv - b) / a;                    // pressure of phase 0
                const Set::Scalar er = vr * (pk + spe.g[ir] * spe.pi[ir]) / (spe.g[ir] - 1.0), et = pk * a + b;
                fv = (dm / mr) * vr / sv;
                fe = (et > 0.0 && er > 0.0) ? std::min((dm / mr) * er / et, 1.0 - 1.0e-9) : fv;
            }
            const Set::Scalar f = fv, de = E0(i, j, k) * fe;
            eta(i, j, k) -= eta(i, j, k) * f;
            m0(i, j, k) -= dm; m1(i, j, k) += dm;
            m(i, j, k, ir) -= dm; m(i, j, k, ip) += dm;
            E0(i, j, k) -= de; E1(i, j, k) += de + Q * dm; E(i, j, k) += Q * dm;
        });
    }
}

// SPECIES (species.on): close the species of each phase to the phase partial density in the valid cells
// (Species::CloseCell: majority species = phase mass - sum of the others).  diag = true ALSO writes the mass
// fractions Y_i = m_i / (alpha_k rho_k) for the plot file and does not touch the partial densities.
void Hydro2::CloseSpecies(int lev, bool diag)
{
    const int n0 = species.n0, n1 = species.n1;
    for (amrex::MFIter mfi(*species.mf[lev], false); mfi.isValid(); ++mfi)
    {
        auto m = species.mf[lev]->array(mfi); auto Y = species.Y_mf[lev]->array(mfi);
        auto r0 = rho_eta0_mf[lev]->const_array(mfi); auto r1 = rho_eta1_mf[lev]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            for (int ph = 0; ph < 2; ++ph)
            {
                const int n = (ph == 0) ? n0 : n1, c0 = (ph == 0) ? 0 : n0;
                if (n == 0) continue;
                const Set::Scalar mk = (ph == 0) ? r0(i, j, k) : r1(i, j, k);
                if (!diag)
                {
                    Set::Scalar v[Species::NMAX];
                    for (int s_ = 0; s_ < n; ++s_) v[s_] = m(i, j, k, c0 + s_);
                    Species::CloseCell(v, n, mk);
                    for (int s_ = 0; s_ < n; ++s_) m(i, j, k, c0 + s_) = v[s_];
                }
                else
                    for (int s_ = 0; s_ < n; ++s_) Y(i, j, k, c0 + s_) = (mk > 1.0e-300) ? m(i, j, k, c0 + s_) / mk : 0.0;
            }
        });
    }
}

// ENTROPY DIAGNOSTIC (entropy.on): entropy per unit volume from the conserved state -- see Hydro2.H.
void Hydro2::ComputeEntropy(int lev)
{
    const Set::Scalar g0_c = eos0.Gamma(), pi0 = eos0.P0(), cv0_c = eos0.Cv(), g1_c = eos1.Gamma(), pi1 = eos1.P0(), cv1_c = eos1.Cv();
    const Species::Eos spe = species.eos();     // mixture: gamma, c_v per cell (the entropy of MIXING of the species is not included)
    const bool solid = embedded.apply != 0, sol_T = solid && ablation.on && ablation.conduction;
    const Set::Scalar rc = sol_T ? ablation.rho_s * ablation.cp_s : 0.0;
    for (amrex::MFIter mfi(*entropy_mf[lev], false); mfi.isValid(); ++mfi)
    {
        auto S = entropy_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->const_array(mfi);
        auto r0 = rho_eta0_mf[lev]->const_array(mfi); auto r1 = rho_eta1_mf[lev]->const_array(mfi);
        auto E0 = energy0_mf[lev]->const_array(mfi);  auto E1 = energy1_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> phi, Ts;
        if (solid) phi = embedded.phi_mf[lev]->const_array(mfi);
        if (sol_T) Ts = ablation.Ts_mf[lev]->const_array(mfi);
        amrex::Array4<const Set::Scalar> sp_e; if (spe.on) sp_e = species.mf[lev]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (solid && phi(i, j, k) < 0.5) { S(i, j, k) = sol_T ? rc * std::log(std::max(Ts(i, j, k), 1.0e-300)) : 0.0; return; }
            const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12), a1 = 1.0 - a0;
            Set::Scalar s = 0.0;
            Set::Scalar g0 = g0_c, cv0 = cv0_c, g1 = g1_c, cv1 = cv1_c;
            if (spe.on) { spe.Mix(sp_e, i, j, k, 0, g0, cv0, E0(i, j, k) / std::max(r0(i, j, k), 1.0e-300)); spe.Mix(sp_e, i, j, k, 1, g1, cv1, E1(i, j, k) / std::max(r1(i, j, k), 1.0e-300)); }
            if (r0(i, j, k) > 0.0)
            {
                const Set::Scalar rho = r0(i, j, k) / a0, p = (g0 - 1.0) * E0(i, j, k) / a0 - g0 * pi0;
                s += r0(i, j, k) * cv0 * (std::log(std::max(p + pi0, 1.0e-300)) - g0 * std::log(rho));
            }
            if (r1(i, j, k) > 0.0)
            {
                const Set::Scalar rho = r1(i, j, k) / a1, p = (g1 - 1.0) * E1(i, j, k) / a1 - g1 * pi1;
                s += r1(i, j, k) * cv1 * (std::log(std::max(p + pi1, 1.0e-300)) - g1 * std::log(rho));
            }
            S(i, j, k) = s;
        });
    }
}

void Hydro2::UpdateMovingSolid(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::UpdateMovingSolid");
    if (!embedded.apply || !(embedded.moving || ablation.on)) return;

    amrex::MultiFab::Copy(*embedded.phi_old_mf[lev], *embedded.phi_mf[lev], 0, 0, 1, nghost);
    if (embedded.moving)
    {
        embedded.phi_ic->Initialize(lev, embedded.phi_mf, time);
        InitEmbeddedSolidTarget(lev, time);
    }
    if (ablation.on) RecedeAblatingSurface(lev, time, dt);

    // grad(phi) diagnostic follows the body
    const Set::Scalar *DX = geom[lev].CellSize();
    embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
    for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto phi      = embedded.phi_mf[lev]->array(mfi);
        auto grad_phi = embedded.grad_phi_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector g = Numeric::Gradient(phi, i, j, k, 0, DX);
            grad_phi(i, j, k, 0) = g(0);
            grad_phi(i, j, k, 1) = g(1);
        });
    }

    // Fresh-cell fill.  State + derived fields RHS reads (ghosts are current: they
    // were filled by the last post-stage FillGhost4BC of the previous step).
    amrex::Vector<amrex::MultiFab *> fields = {
        rho_eta0_mf[lev].get(), rho_eta1_mf[lev].get(), momentum_mf[lev].get(),
        energy_per_vol_mf[lev].get(), energy_per_mas_mf[lev].get(), eta_mf[lev].get(),
        energy0_mf[lev].get(), energy1_mf[lev].get(), shell_mf[lev].get(), cfun_mf[lev].get(),
        density_mf[lev].get(), velocity_mf[lev].get(), pressure_mf[lev].get(),
        T_mf[lev].get(), cp_mf[lev].get(), cv_mf[lev].get(),
        gamma_mf[lev].get(), p0_mf[lev].get(), a_mf[lev].get()};
    long nfresh = 0, norphan = 0;
    for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto phn = embedded.phi_mf[lev]->const_array(mfi);
        auto pho = embedded.phi_old_mf[lev]->const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            if (!(pho(i, j, k) < 0.5 && phn(i, j, k) >= 0.5)) return;
            int di = 0, dj = 0, dk = 0; Set::Scalar best = -1.0;
#if AMREX_SPACEDIM == 3
            for (int c = -1; c <= 1; ++c)
#else
            for (int c = 0; c <= 0; ++c)
#endif
            for (int b = -1; b <= 1; ++b)
            for (int a = -1; a <= 1; ++a)
            {
                if (a == 0 && b == 0 && c == 0) continue;
                if (!(pho(i + a, j + b, k + c) >= 0.5 && phn(i + a, j + b, k + c) >= 0.5)) continue;
                // prefer face neighbours, then the one deepest in the fluid
                const Set::Scalar score = phn(i + a, j + b, k + c) - 0.25 * (std::abs(a) + std::abs(b) + std::abs(c) - 1);
                if (score > best) { best = score; di = a; dj = b; dk = c; }
            }
            if (best < 0.0) { ++norphan; return; }
            ++nfresh;
            for (amrex::MultiFab *mf : fields)
            {
                if (!mf) continue;                       // (MultiFab::ok() runs its own MFIter: not allowed in here)
                auto f = mf->array(mfi);
                for (int n = 0; n < mf->nComp(); ++n) f(i, j, k, n) = f(i + di, j + dj, k + dk, n);
            }
        });
    }
    amrex::ParallelDescriptor::ReduceLongSum(nfresh);
    amrex::ParallelDescriptor::ReduceLongSum(norphan);
    if (norphan > 0)
        Util::Warning(INFO, "solid.moving: ", norphan, " fresh cell(s) with no fluid donor on level ", lev,
                      " (wall moved more than one cell in a step?) -- left at the solid state");
    if (nfresh > 0) FillGhost4BC(lev, time);     // neighbours' ghost copies of the filled cells
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////// REGRIDDING //////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
void Hydro2::Regrid(int lev, Set::Scalar regrid_time)
{
    BL_PROFILE("Integrator::Hydro2::Regrid");

    // Zero fill fields
    ZeroDerivedScratchFields(lev);

    // Re-allocate reflux scratch on the freshly-regridded level. Any previously-accumulated fluxes are discarded.
    AllocateRefluxScratch(lev);

    // EMBEDDED SOLID: Re-evaluate phi from its IC on the new grid rather
    // than relying on coarse-fine interpolation, which would smear the
    // sharp solid boundary and create spurious partial-solid cells.
    if (embedded.apply)
    {
        // ABLATION + AMR (2026-10-29): the receding surface is NOT the initial shape any more.  Keep the phi the
        // framework transferred to the new grid (RemakeLevel: copy where the old level existed, conservative
        // interpolation from the coarse level elsewhere; MakeNewLevelFromCoarse: interpolation), instead of
        // resetting it to phi_ic, which would undo the recession on every regrid of a refined run.
        if (!ablation.on)
        {
            embedded.phi_ic   ->Initialize(lev, embedded.phi_mf,            regrid_time);
            embedded.phi_ic   ->Initialize(lev, embedded.phi_old_mf,        regrid_time);
        }
        else
        {
            amrex::MultiFab::Copy(*embedded.phi_old_mf[lev], *embedded.phi_mf[lev], 0, 0, 1, embedded.phi_mf[lev]->nGrow());
        }
        // Re-derive the per-phase solid target from the single density+pressure
        // (split by the regridded eta).
        InitEmbeddedSolidTarget(lev, regrid_time);

        // grad(phi) diagnostic (static; recomputed for the new grid).
        const Set::Scalar *DX = geom[lev].CellSize();
        embedded.phi_mf[lev]->FillBoundary(geom[lev].periodicity());
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto grad_phi = embedded.grad_phi_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector g = Numeric::Gradient(phi, i, j, k, 0, DX);
                grad_phi(i, j, k, 0) = g(0);
                grad_phi(i, j, k, 1) = g(1);
            });
        }
    }

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

    // Eta criterion for refinement
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& eta = (*eta_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
            if (grad_eta.lpNorm<2>() * dr * 2 > eta_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
        });
    }

    // Geometric refinement box: force refinement to max_level inside [lo,hi]
    // (targets a region, e.g. the trailing edge, without refining the whole domain)
    if (refine_box_on && (refine_box_max_level < 0 || lev < refine_box_max_level))
    {
        const Set::Scalar* problo = geom[lev].ProbLo();
        const Set::Scalar bl0 = refine_box_lo[0], bh0 = refine_box_hi[0];
        const Set::Scalar bl1 = refine_box_lo[1], bh1 = refine_box_hi[1];
#if AMREX_SPACEDIM == 3
        const Set::Scalar bl2 = refine_box_lo[2], bh2 = refine_box_hi[2];
#endif
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], true); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<char> const& tags = a_tags.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar x = problo[0] + (i + 0.5) * DX[0];
                Set::Scalar y = problo[1] + (j + 0.5) * DX[1];
                bool inside = (x >= bl0 && x <= bh0 && y >= bl1 && y <= bh1);
#if AMREX_SPACEDIM == 3
                Set::Scalar z = problo[2] + (k + 0.5) * DX[2];
                inside = inside && (z >= bl2 && z <= bh2);
#endif
                if (inside) tags(i, j, k) = amrex::TagBox::SET;
            });
        }
    }

    // EMBEDDED SOLID: refine the diffuse solid boundary
    if (embedded.apply)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*embedded.phi_mf[lev], true); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            amrex::Array4<char> const& tags = a_tags.array(mfi);
            amrex::Array4<const Set::Scalar> const& phi = (*embedded.phi_mf[lev]).array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector grad_phi = Numeric::Gradient(phi, i, j, k, 0, DX);
                if (grad_phi.lpNorm<2>() * dr * 2 > embedded.refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
            });
        }
    }

    // Vorticity criterion for refinement
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*vorticity_mf[lev], true); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        amrex::Array4<char> const& tags = a_tags.array(mfi);
        amrex::Array4<const Set::Scalar> const& omega = (*vorticity_mf[lev]).array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            auto sten = Numeric::GetStencil(i, j, k, bx);
#if AMREX_SPACEDIM == 2
            Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, 0, DX, sten);
            if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion) tags(i, j, k) = amrex::TagBox::SET;
#else
            // 3D vorticity has 3 components; tag if any component's gradient is steep.
            for (int c = 0; c < AMREX_SPACEDIM; ++c)
            {
                Set::Vector grad_omega = Numeric::Gradient(omega, i, j, k, c, DX, sten);
                if (grad_omega.lpNorm<2>() * dr * 2 > omega_refinement_criterion)
                {
                    tags(i, j, k) = amrex::TagBox::SET;
                    break;
                }
            }
#endif
        });
    }
    
    // Gradu criterion for refinement
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    if (p_refinement_max_level < 0 || lev < p_refinement_max_level)
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
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



///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// INTERFACE SHARPENING /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////// REINITIALIZE SIGNED DISTANCE ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


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
            // Radial slab: overwrite the transverse-face ghosts (the BC above
            // only clamp-copied them).  Momentum is the one vector field.
            if (radial_bc) RadialGhostFill(lev, *mf, bc == momentum_bc);
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
///   5. Update rho_eta0, rho_eta1 from NSCBC-modified rho_total
///   6. Compute primitives in ghosts (using NSCBC's gamma/p if available)
///   7. Enforce consistency and check for NaN/Inf
///
/// ============================================================================
void Hydro2::FillGhost4BC(int lev, Set::Scalar time)
{
    if (species.on && species.eos_mix)      // the phase constants of every cell (ghosts too) come from the species
    {
        species.bc->define(geom[lev]);
        // [2026-11-18] coarse-fine ghosts from the coarse level on lev > 0 (was a same-level fill only: species on AMR
        // read stale ghosts at coarse-fine faces).  Single-level runs are unchanged.
        if (lev > 0) FillPatch(lev, time, species.mf, *species.mf[lev], *species.bc, 0);
        else FillBoundariesWithBC(lev, time, species.bc, { species.mf[lev].get() });
    }
    BL_PROFILE("Integrator::Hydro2::FillGhost4BC");

    const Set::Scalar *DX = geom[lev].CellSize();
    amrex::Box domain = geom[lev].Domain();

    // Ensure eta_bc has a valid geometry for FillBoundary calls.
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
            if (verbose) Util::Message(INFO, "FillGhost4BC: Using NSCBC with 2 ghost cells");
        }
        else if (nghost == 4)
        {
            if (verbose) Util::Message(INFO, "FillGhost4BC: Using NSCBC4 with 4 ghost cells");
        }
    }
    else
    {
        effective_nghost = 2;
    }

    // Fill Patches for regridding -- AMR coarse-fine ghost interpolation
    // for the FULL 6-eq conservative state.  Missing energy0_mf or
    // energy1_mf here causes the flux-loop make_state to pick up STALE
    // (unfilled) per-phase E_k ghost values at coarse-fine interfaces;
    // when a wave reaches a fine patch whose ghosts sit at the corner
    // of (coarse-fine) and (physical-boundary), the BC fills neighbor
    // ghosts on top of those garbage E_k cells -> garbage flux -> NaN.
    if (lev > 0)
    {
        FillPatch(lev, time, rho_eta0_mf,        *rho_eta0_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, rho_eta1_mf,        *rho_eta1_mf[lev],       *density_bc,  0);
        FillPatch(lev, time, momentum_mf,        *momentum_mf[lev],       *momentum_bc, 0);
        FillPatch(lev, time, energy_per_vol_mf,  *energy_per_vol_mf[lev], *energy_bc,   0);
        FillPatch(lev, time, eta_mf,             *eta_mf[lev],            *eta_bc,      0);
        FillPatch(lev, time, shell_mf,           *shell_mf[lev],          *eta_bc,      0);
        FillPatch(lev, time, cfun_mf,            *cfun_mf[lev],           *eta_bc,      0);
        FillPatch(lev, time, energy0_mf,         *energy0_mf[lev],        *energy_bc,   0);
        FillPatch(lev, time, energy1_mf,         *energy1_mf[lev],        *energy_bc,   0);
        if (elastic.on) FillPatch(lev, time, elastic.ebasis_mf, *elastic.ebasis_mf[lev], *elastic.bc, 0);
    }
    if (elastic.on)   // fluid cells carry no deformation memory; then cobasis ghosts (needed before any stress evaluation)
    {
        elastic.ResetFluidCells(lev, *eta_mf[lev], *rho_eta0_mf[lev], *rho_eta1_mf[lev], (species.on ? species.mf[lev].get() : nullptr));
        elastic.FillGhosts(lev, geom[lev], time);
    }

    // ------------------------------------------------------------
    // STEP 2: Compute total density in DOMAIN; clamp eta.
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox(); // DOMAIN ONLY

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
        });
    }

    // ------------------------------------------------------------
    // STEP 2b: EMBEDDED SOLID BOUNDARY auto wall + ghost fill.
    // ------------------------------------------------------------
    // Fill the static indicator's ghosts, then (only when solid.brinkman was
    // not provided) apply a dt-independent, unconditionally-stable momentum-
    // only projection toward the solid velocity, graded by phi.  When
    // solid.brinkman > 0 the explicit RHS penalty is the wall and this is
    // skipped.  This runs BEFORE the STEP 4 primitive recovery so the EOS sees
    // a sensible (finite p, c) state inside the solid, and it is IDEMPOTENT --
    // safe to call from RHS, the post-stage hook, and the post-integration
    // refresh.
    if (embedded.apply)
    {
        embedded.phi_bc->define(geom[lev]);
        FillBoundariesWithBC(lev, time, embedded.phi_bc, { embedded.phi_mf[lev].get() });

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.tilebox(); // DOMAIN ONLY
            auto phi      = embedded.phi_mf[lev]->array(mfi);
            auto M        = momentum_mf[lev]->array(mfi);
            auto E        = energy_per_vol_mf[lev]->array(mfi);
            auto E0_arr   = energy0_mf[lev]->array(mfi);
            auto E1_arr   = energy1_mf[lev]->array(mfi);
            auto rho      = density_mf[lev]->array(mfi);
            auto s_M      = embedded.momentum_mf[lev]->array(mfi);

            const Set::Scalar small_  = small;
            const Set::Scalar brink_  = embedded.brinkman;

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // Auto wall (used only when solid.brinkman not provided): a
                // dt-independent, unconditionally-stable momentum-only
                // projection toward the solid velocity, graded by phi:
                //   M <- phi*M + (1-phi)*M_solid  (= the rest-init blend applied
                // every step).  When solid.brinkman > 0 the explicit RHS penalty
                // is the wall instead and this is skipped.  Mass/per-phase
                // energy keep their evolved (full-flux) values; only the
                // momentum is projected and the redundant total energy E is
                // recomputed consistently (rho already = rho_eta0+rho_eta1 from
                // STEP 2 above).
                if (brink_ <= 0.0)
                {
                    const Set::Scalar w = std::min(std::max(phi(i, j, k), 0.0), 1.0);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        M(i, j, k, d) = w * M(i, j, k, d) + (1.0 - w) * s_M(i, j, k, d);
                    const Set::Scalar KE = 0.5 * (AMREX_D_TERM(M(i, j, k, 0) * M(i, j, k, 0),
                                                             + M(i, j, k, 1) * M(i, j, k, 1),
                                                             + M(i, j, k, 2) * M(i, j, k, 2))) / std::max(rho(i, j, k), small_);
                    E(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k) + KE;
                }
            });
        }
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
    // shell and cfun ride along: they share eta's zero-gradient far-field
    // behaviour, and nothing else fills their DOMAIN-face ghosts (RHS only
    // does the periodic/same-level exchange), so without this the shell
    // stencil read stale t=0 values at every physical boundary.
    FillBoundariesWithBC(lev, time, eta_bc, {
            eta_mf[lev].get(),
            shell_mf[lev].get(),
            cfun_mf[lev].get()
     });


    // ------------------------------------------------------------
    // STEP 4: Compute primitives in DOMAIN (6-equation recovery).
    //
    // From the canonical primaries (eta, (alpha rho)_k, M, rho E, E_k) we
    // derive per-phase rho_k, e_k, p_k, c_k and the mixture rho, u, p, c.
    //   - p_k uses Sau09 eq. (II.3) inverted (PhasicPressureFromEnergy).
    //   - p   uses Schmidmayer 2020 eq. (8): p = alpha_1 p_1 + alpha_2 p_2.
    //   - c   uses Schmidmayer 2020 eq. (17): c^2 = Y_1 c_1^2 + Y_2 c_2^2.
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*velocity_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox(); // DOMAIN ONLY

        auto rho       = density_mf[lev]->array(mfi);
        auto eta       = eta_mf[lev]->array(mfi);
        auto arh0_arr  = rho_eta0_mf[lev]->array(mfi);
        auto arh1_arr  = rho_eta1_mf[lev]->array(mfi);
        auto M         = momentum_mf[lev]->array(mfi);
        auto E         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_arr    = energy0_mf[lev]->array(mfi);
        auto E1_arr    = energy1_mf[lev]->array(mfi);
        auto v         = velocity_mf[lev]->array(mfi);
        auto press     = pressure_mf[lev]->array(mfi);
        auto T         = T_mf[lev]->array(mfi);
        const int Tpc = T_phase_consistent;     // temperature.phase_consistent
        auto tph = Tpc ? Tph_mf[lev]->array(mfi) : amrex::Array4<Set::Scalar>{};
        auto a         = a_mf[lev]->array(mfi);
        auto gamma     = gamma_mf[lev]->array(mfi);
        auto p0_eff    = p0_mf[lev]->array(mfi);
        auto UE        = UE_per_vol_mf[lev]->array(mfi);
        auto KE        = KE_per_vol_mf[lev]->array(mfi);
        auto rho0_arr  = density0_mf[lev]->array(mfi);
        auto rho1_arr  = density1_mf[lev]->array(mfi);
        auto p0_arr    = pressure0_mf[lev]->array(mfi);
        auto p1_arr    = pressure1_mf[lev]->array(mfi);
        auto v0_arr    = velocity0_mf[lev]->array(mfi);
        auto v1_arr    = velocity1_mf[lev]->array(mfi);

        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_base = eos1;
        
        const Species::Eos spe_l = species.eos();     // species.eos = mixture: gamma, c_v of each phase per cell
        amrex::Array4<const Set::Scalar> sp_l; if (spe_l.on) sp_l = species.mf[lev]->const_array(mfi);
        auto cf_l = cfun_mf[lev]->const_array(mfi);     // alpha_ref of the compaction model
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Solver::EOS::Tammann eos0_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 0, eos0_base, E0_arr(i, j, k) / std::max(arh0_arr(i, j, k), 1.0e-300), arh0_arr(i, j, k) / std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12)) : eos0_base;
            const Solver::EOS::Tammann eos1_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 1, eos1_base, E1_arr(i, j, k) / std::max(arh1_arr(i, j, k), 1.0e-300), arh1_arr(i, j, k) / (1.0 - std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12))) : eos1_base;
            const Set::Scalar gam0 = eos0_local.Gamma(), gam1 = eos1_local.Gamma();
            const Set::Scalar pi1_ = eos1_local.P0();
            const Set::Scalar pi0_ = eos0_local.P0() + ((elastic.on && elastic.comp_on) ? elastic.compaction(elastic.matAt(sp_l, i, j, k), std::min(std::max(eta(i, j, k), 0.0), 1.0), arh0_arr(i, j, k), cf_l(i, j, k)).Pc / gam0 : 0.0);
            // Volume fraction with floor.
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            // Phase masses (canonical).
            Set::Scalar arh0 = std::max(arh0_arr(i, j, k), small);
            Set::Scalar arh1 = std::max(arh1_arr(i, j, k), small);

            // Mixture density (Sch20 eq. 8 RHS).
            rho(i, j, k) = arh0 + arh1;

            // Velocity from mixture momentum.
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            // Kinetic energy.
            KE(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                          + v(i, j, k, 1) * v(i, j, k, 1),
                                                          + v(i, j, k, 2) * v(i, j, k, 2)));

            // Pure-phase densities for per-phase EOS evaluation.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;

            // Per-phase pressures from canonical (alpha rho e)_k (Sau09 II.3).
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);
            // TRACE-PHASE SLAVING (see flux-loop recovery for rationale):
            // a vanished phase's E_k/alpha_k pressure is gam*pi/p-amplified
            // noise; slave it to the dominant phase below the eta cutoff.
            {
                // SMOOTH slaving (see flux-loop recovery block for rationale).
                const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w0 = w0 * w0 * (3.0 - 2.0 * w0);
                Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w1 = w1 * w1 * (3.0 - 2.0 * w1);
                const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
            }

            // Per-phase sound speeds.
            Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho0_pure, p0_loc, gam0, pi0_, small);
            Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho1_pure, p1_loc, gam1, pi1_, small);

            // Mass fractions.
            Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
            Set::Scalar Y1 = 1.0 - Y0;

            // Mixture pressure (Sch20 eq. 8) and frozen sound speed (Sch20 eq. 17).
            press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
            a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);

            // Per-phase diagnostic primitives (plotfile).
            rho0_arr(i, j, k) = rho0_pure;
            rho1_arr(i, j, k) = rho1_pure;
            p0_arr(i, j, k)   = p0_loc;
            p1_arr(i, j, k)   = p1_loc;
            v0_arr(i, j, k, 0) = v(i, j, k, 0);   // mechanical equilibrium (one velocity)
            v0_arr(i, j, k, 1) = v(i, j, k, 1);
            v1_arr(i, j, k, 0) = v(i, j, k, 0);
            v1_arr(i, j, k, 1) = v(i, j, k, 1);

            // Internal energy (sum of canonical per-phase energies).
            UE(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);

            // Diagnostic gamma / p0_eff (NOT used by hyperbolic step).
            gamma(i, j, k)  = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);
            if (Tpc)
            {
                const Species::Eos &se_t = spe_l;
                const bool pe0 = se_t.on && se_t.th.on && se_t.th.ph == 0, pe1 = se_t.on && se_t.th.on && se_t.th.ph == 1;
                Set::Scalar t0, t1;
                T(i, j, k) = PhaseConsistentT(E0_arr(i, j, k), E1_arr(i, j, k), arh0_arr(i, j, k), arh1_arr(i, j, k), a1, eos0_local, eos1_local, pe0, pe1, t0, t1);
                tph(i, j, k, 0) = t0; tph(i, j, k, 1) = t1;
            }
            else T(i, j, k)      = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), a1, eos0_local, eos1_local, pref);
        });
    }

    // ============================================================
    // 6-eq STIFF PRESSURE RELAXATION + REINIT (Sch20 Sec.3.3 / Sau09 Sec.3.5)
    // Primitives are computed first (STEP 4 above), then the Newton enforces 
    // p_0 = p_1 on thecanonical primaries (eta, E_0, E_1) before any source/flux
    // ============================================================
    // MERGE (AcousticBC): mass-tracking diagnostic, gated on relax_diag.  This
    // supersedes the unconditional pre-relax Message that was here on the
    // Marmottant side -- that one printed every call and swamped the logs.
    if (diag_gasmass) GasLedger(lev, "fillghost_in", time);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK fillghost_in  lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
    // MERGE (Marmottant): Schmidmayer 2017 splitting -- U^{n+1} = L_relax L_cap
    // L_hyper(U^n), so relaxation runs ONCE per step after the capillary
    // operator, not between RK stages.  defer_relax is false unless the split
    // scheme is active, so the AcousticBC behaviour is unchanged by default.
    // The sub-cycled shell viscous stress (visc_subcycle) used to be
    // advanced HERE.  FillGhost4BC runs at every RK stage (RHS start and the
    // post-stage hook), so it applied a full dt of shell viscous impulse 6x per
    // step with RK3 (measured 2026-10-23: 6 calls/step).  It now runs once
    // per step from Advance, before the post-integration refresh, whose
    // RelaxAndReinit still follows it (L_relax o L_visc o L_hyper).

    relax_time = time;
    if (!defer_relax)
    {
        RelaxAndReinit(lev);
        // The relaxation changes eta (and ShellBandUniform inside it, shell) in the valid
        // cells, but their ghosts were filled in STEP 3, before it.  STEP 5 refills only
        // rho_eta0/1, M and E_k, so STEP 7 rebuilt the ghost primitives from post-relax
        // rho_eta_k over PRE-relax eta (mirror audit: eta asym 2e-5, all others 0).  On
        // the octant the axis rows (two mirrors) grew a gas pocket in the liquid tail from
        // it (INCLINE Mili crash, 11-03).  Same fill as STEP 3.
        FillBoundariesWithBC(lev, time, eta_bc, {
                eta_mf[lev].get(),
                shell_mf[lev].get(),
                cfun_mf[lev].get()
        });
    }
    if (phasechange.on)
    {
        if (!defer_relax) RelaxPhaseChange(lev);      // thermodynamic (p-T-g) relaxation, after the mechanical one
        // eta's ghosts were filled in STEP 3, BEFORE the relaxations; phase change moves eta by O(1),
        // and a stale ghost eta next to fresh ghost masses / energies makes the two sides of a
        // periodic or box boundary see different states (measured: mass drift 5e-7 per 10 steps in a
        // uniform periodic box).  Refill it together with the vapour field.
        FillBoundariesWithBC(lev, time, eta_bc, { eta_mf[lev].get(), phasechange.vapor_mf[lev].get() });
    }

    // ------------------------------------------------------------
    // STEP 5: Fill CONSERVATIVE ghost cells (rho, M, E)
    // ------------------------------------------------------------
    if (use_nscbc)
    {
        // ====================================================================
        // NSCBC PATH: Use characteristic-based boundary conditions
        // ====================================================================

        const int ib_lo = geom[lev].Domain().smallEnd(0);
        const int ib_hi = geom[lev].Domain().bigEnd(0);
        const int jb_lo = geom[lev].Domain().smallEnd(1);
        const int jb_hi = geom[lev].Domain().bigEnd(1);
        const bool x_periodic = geom[lev].isPeriodic(0);
        const bool y_periodic = geom[lev].isPeriodic(1);
#if AMREX_SPACEDIM == 3
        const int kb_lo = geom[lev].Domain().smallEnd(2);
        const int kb_hi = geom[lev].Domain().bigEnd(2);
        const bool z_periodic = geom[lev].isPeriodic(2);
#endif

        // Faces with nscbc.<face>.type = none (e.g. REFLECT symmetry planes)
        // are owned by the registered field BCs: their ghosts were mirror-
        // filled by the Expression BCs above and must NOT be overwritten by
        // the eta clamp-extrapolation or the post-NSCBC equilibrium
        // partition below.  The partition at such a face used CLAMPED eta
        // (killing the layer-2 mirror) and STALE pressure ghosts (STEP 4
        // computes pressure in the valid box only; only NSCBC faces get
        // ghost pressure), producing a one-cell overpressure dipole at the
        // wall (+15% of the local signal, quadrant-vs-full-domain A/B).
        const bool ns_xlo = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(0, 0) : true;
        const bool ns_xhi = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(0, 1) : true;
        const bool ns_ylo = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(1, 0) : true;
        const bool ns_yhi = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(1, 1) : true;
#if AMREX_SPACEDIM == 3
        const bool ns_zlo = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(2, 0) : true;
        const bool ns_zhi = (nscbc4_bc != nullptr) ? nscbc4_bc->FaceActive(2, 1) : true;
#endif

        // --------------------------------------------------------------------
        // PRE-NSCBC: refresh periodic ghosts on ALL 6-eq conservative
        // primaries.  Without this, periodic ghosts of (alpha rho)_k,
        // momentum, rho E, E_0, E_1 carry stale values from the previous
        // timestep (the post-stage MultiFab::Copy copies the integrator's
        // stage ghosts which are never advanced).  The downstream RHS reads
        // ghost cells at y-edge interior cells to build Riemann states; if
        // those are stale, the flux divergence is wrong, the error feeds
        // back into the domain, and after O(60) steps the state explodes.
        //
        // FillBoundariesWithBC unconditionally runs mf->FillBoundary(
        // periodicity) on each MF; the bc->FillBoundary call is a no-op
        // for nscbc_outflow / nscbc_inflow faces (Expression doesn't handle
        // those types) — so this DOES fill periodic ghosts and is harmless
        // for the NSCBC faces (NSCBC overwrites those further down).
        // --------------------------------------------------------------------
        FillBoundariesWithBC(lev, time, density_bc, {
            rho_eta0_mf[lev].get(),
            rho_eta1_mf[lev].get()
        });
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        FillBoundariesWithBC(lev, time, energy_bc, {
            energy_per_vol_mf[lev].get(),
            energy0_mf[lev].get(),
            energy1_mf[lev].get()
        });

        // --------------------------------------------------------------------
        // PRE-NSCBC: extrapolate eta into NSCBC ghost cells by constant
        // copy from the boundary cell.  Periodic-direction ghosts are
        // skipped because they are already filled by FillBoundary's
        // periodic copy in STEP 3 above.
        // --------------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto eta = eta_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const bool x_outside = (i < ib_lo) || (i > ib_hi);
                const bool y_outside = (j < jb_lo) || (j > jb_hi);
#if AMREX_SPACEDIM == 3
                const bool z_outside = (k < kb_lo) || (k > kb_hi);
#else
                const bool z_outside = false;
#endif
                if (!x_outside && !y_outside && !z_outside) return;  // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone
#if AMREX_SPACEDIM == 3
                if (z_outside && z_periodic)  return;                // periodic — leave alone
#endif
                // Ghost touching a non-NSCBC (mirror) face: keep the STEP-3
                // Expression fill (mirror in the reflect direction, clamp in
                // any composed nscbc direction).
                if ((i < ib_lo && !ns_xlo) || (i > ib_hi && !ns_xhi)) return;
                if ((j < jb_lo && !ns_ylo) || (j > jb_hi && !ns_yhi)) return;
#if AMREX_SPACEDIM == 3
                if ((k < kb_lo && !ns_zlo) || (k > kb_hi && !ns_zhi)) return;
                const int kb = std::min(std::max(k, kb_lo), kb_hi);
#else
                const int kb = k;
#endif
                const int ib = std::min(std::max(i, ib_lo), ib_hi);
                const int jb = std::min(std::max(j, jb_lo), jb_hi);
                eta(i, j, k) = eta(ib, jb, kb);
            });
        }

        // --------------------------------------------------------------------
        // Build mixture rho_total in domain + ghosts.  NSCBC's LODI uses
        // ONLY mixture quantities; the per-phase fields are re-derived after.
        // --------------------------------------------------------------------
        amrex::MultiFab rho_total(rho_eta0_mf[lev]->boxArray(),
                                  rho_eta0_mf[lev]->DistributionMap(),
                                  1,
                                  nghost);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rho_total, HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.growntilebox(nghost);
            auto rho  = rho_total.array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }

        // Working copies for momentum and total energy (NSCBC modifies in-place).
        amrex::MultiFab M_copy(momentum_mf[lev]->boxArray(),
                               momentum_mf[lev]->DistributionMap(),
                               AMREX_SPACEDIM,
                               nghost);
        amrex::MultiFab E_copy(energy_per_vol_mf[lev]->boxArray(),
                               energy_per_vol_mf[lev]->DistributionMap(),
                               1,
                               nghost);
        amrex::MultiFab::Copy(M_copy, *momentum_mf[lev],       0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(E_copy, *energy_per_vol_mf[lev], 0, 0, 1,              nghost);

        // --------------------------------------------------------------------
        // Run NSCBC LODI.  Uses the mixture-effective EOS internally (Tammann
        // form, validated against the 5-equation code path).  Modifies
        // rho_total, M_copy, E_copy in ghosts; writes gamma_mf, p0_mf,
        // pressure_mf in ghosts as well.
        // --------------------------------------------------------------------
        if (nghost == 2 && nscbc_bc != nullptr)
        {
            nscbc_bc->FillBoundary(rho_total, M_copy, E_copy,
                                   *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev],
                                   eos0, eos1, geom[lev], time, pref);
        }
        else if (nghost == 4 && nscbc4_bc != nullptr)
        {
            nscbc4_bc->FillBoundary(rho_total, M_copy, E_copy,
                                    *eta_mf[lev], *gamma_mf[lev], *p0_mf[lev], *pressure_mf[lev],
                                    eos0, eos1, geom[lev], time, pref);
        }

        amrex::MultiFab::Copy(*momentum_mf[lev],       M_copy, 0, 0, AMREX_SPACEDIM, nghost);
        amrex::MultiFab::Copy(*energy_per_vol_mf[lev], E_copy, 0, 0, 1,              nghost);

        // --------------------------------------------------------------------
        // POST-NSCBC partition: build the 6-eq per-phase ghosts from the
        // NSCBC-filled mixture and the (pre-extrapolated) eta.  Identical
        // ideology to the 5-equation form — split mass and internal energy
        // by alpha alone, with mechanical-equilibrium pressure:
        //
        //   (alpha rho)_0_ghost = rho_total_ghost * eta_ghost
        //   (alpha rho)_1_ghost = rho_total_ghost * (1 - eta_ghost)
        //   E_0_ghost           = eta_ghost     * (p_g + gamma_0 pi_0)/(gamma_0 - 1)
        //   E_1_ghost           = (1-eta_ghost) * (p_g + gamma_1 pi_1)/(gamma_1 - 1)
        //
        // The E_k formulas are Sau09 eq. III.5 / Sch20 eq. 13 init evaluated
        // at the NSCBC-set ghost pressure and the boundary-extrapolated eta.
        // For periodic-direction ghosts the fields are already correct via
        // FillBoundary periodicity — skip.
        // --------------------------------------------------------------------
        const Set::Scalar g0_ns = eos0.Gamma();
        const Set::Scalar pi0_ns = eos0.P0();
        const Set::Scalar g1_ns = eos1.Gamma();
        const Set::Scalar pi1_ns = eos1.P0();
        const Set::Scalar small_ns = small;
        // DIAGNOSTIC (nscbc.ghost_mode=1): bypass the equilibrium partition and
        // zero-gradient-copy the per-phase ghosts instead, making the NSCBC
        // branch's ghost state functionally identical to the Neumann branch.
        const int gm_copy = (nscbc4_bc != nullptr) ? nscbc4_bc->ghost_mode : 0;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(rho_total, HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho   = rho_total.array(mfi);
            auto eta   = eta_mf[lev]->array(mfi);
            auto rho0  = rho_eta0_mf[lev]->array(mfi);
            auto rho1  = rho_eta1_mf[lev]->array(mfi);
            auto press = pressure_mf[lev]->array(mfi);
            auto E0    = energy0_mf[lev]->array(mfi);
            auto E1    = energy1_mf[lev]->array(mfi);

            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const bool x_outside = (i < ib_lo) || (i > ib_hi);
                const bool y_outside = (j < jb_lo) || (j > jb_hi);
#if AMREX_SPACEDIM == 3
                const bool z_outside = (k < kb_lo) || (k > kb_hi);
#else
                const bool z_outside = false;
#endif
                if (!x_outside && !y_outside && !z_outside) return;  // interior — leave alone
                if (x_outside && x_periodic)  return;                // periodic — leave alone
                if (y_outside && y_periodic)  return;                // periodic — leave alone
#if AMREX_SPACEDIM == 3
                if (z_outside && z_periodic)  return;                // periodic — leave alone
#endif
                // Ghost touching a non-NSCBC (mirror) face: the per-phase
                // ghosts were mirror-filled by the pre-NSCBC Expression BCs;
                // the partition would rebuild them from clamped eta and stale
                // ghost pressure.  Keep the mirrors.
                if ((i < ib_lo && !ns_xlo) || (i > ib_hi && !ns_xhi)) return;
                if ((j < jb_lo && !ns_ylo) || (j > jb_hi && !ns_yhi)) return;
#if AMREX_SPACEDIM == 3
                if ((k < kb_lo && !ns_zlo) || (k > kb_hi && !ns_zhi)) return;
#endif

                if (gm_copy == 1)
                {
                    const int ic = amrex::min(amrex::max(i, ib_lo), ib_hi);
                    const int jc = amrex::min(amrex::max(j, jb_lo), jb_hi);
                    int kc = k;
#if AMREX_SPACEDIM == 3
                    kc = amrex::min(amrex::max(k, kb_lo), kb_hi);
#endif
                    rho0(i, j, k) = rho0(ic, jc, kc);
                    rho1(i, j, k) = rho1(ic, jc, kc);
                    E0(i, j, k)   = E0(ic, jc, kc);
                    E1(i, j, k)   = E1(ic, jc, kc);
                    return;
                }

                const Set::Scalar a1 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2 = 1.0 - a1;
                const Set::Scalar rho_g = std::max(rho(i, j, k), small_ns);
                const Set::Scalar p_g   = std::max(press(i, j, k), small_ns);

                // Per-phase densities: preserve the (BC-filled) per-phase RATIO and
                // rescale to the LODI-updated mixture rho_g, instead of re-splitting
                // rho_g by eta -- the latter mis-assigns liquid mass to the gas at a
                // mixed ghost (alpha_g*alpha_l*(rho_l-rho_g)); pure-phase ghosts are
                // identical either way.  Energies stay the equilibrium re-derivation.
                const Set::Scalar r0   = std::max(rho0(i, j, k), 0.0);
                const Set::Scalar r1   = std::max(rho1(i, j, k), 0.0);
                const Set::Scalar scale = rho_g / std::max(r0 + r1, small_ns);
                rho0(i, j, k) = r0 * scale;
                rho1(i, j, k) = r1 * scale;
                E0(i, j, k)   = a1 * (p_g + g0_ns * pi0_ns) / std::max(g0_ns - 1.0, small_ns);
                E1(i, j, k)   = a2 * (p_g + g1_ns * pi1_ns) / std::max(g1_ns - 1.0, small_ns);
            });
        }

        // Update density_mf from rho_total
        amrex::MultiFab::Copy(*density_mf[lev], rho_total, 0, 0, 1, nghost);
    }
    else
    {
        // Density: apply density_bc to the TOTAL density (density_mf),
        // then partition into rho_eta0/rho_eta1 using eta.
        // The user-specified dirichlet value (e.g. 39.0 at xlo) is the
        // total mixture density. Applying it directly to both rho_eta0
        // AND rho_eta1 would double the total density in ghost cells.
        // Inter-fab exchange for phase densities still needed:
        rho_eta0_mf[lev]->FillBoundary(geom[lev].periodicity());
        rho_eta1_mf[lev]->FillBoundary(geom[lev].periodicity());
        // Per-phase density GHOSTS via their own BC.  Do NOT re-partition the mixture
        // density by eta: rho_eta_k = rho_mix*(eta | 1-eta) is exact only at PURE-phase
        // cells.  At a mixed-eta INTERFACE cell it injects alpha_g*alpha_l*(rho_l-rho_g)
        // (~250 for water/air at 50/50) of LIQUID mass into the GAS partial density --
        // and growntilebox(nghost) clobbered the VALID, flux-conserved cells too, so it
        // was a per-step gas-mass SOURCE at the interface that destroyed bubble collapse
        // (gas mass blew up ~119x; see tests/FlowRayleighPlesset mass-conservation check).
        // rho_eta0/1 are conserved -- fill only their ghosts, directly, per phase.
        FillBoundariesWithBC(lev, time, density0_bc, { rho_eta0_mf[lev].get() });
        FillBoundariesWithBC(lev, time, density1_bc, { rho_eta1_mf[lev].get() });
        // Mixture density is the consistent sum of the partial densities.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho  = density_mf[lev]->array(mfi);
            auto rho0 = rho_eta0_mf[lev]->array(mfi);
            auto rho1 = rho_eta1_mf[lev]->array(mfi);
            amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            });
        }
        // Mixture momentum.
        FillBoundariesWithBC(lev, time, momentum_bc, {
            momentum_mf[lev].get()
        });
        if (!bc_primitive)
        {
            // CONSERVATIVE PATH: redundant total energy AND per-phase internal
            // energies (6-eq primaries -- Sch20 eq. 13 last two rows + eq. 16).
            FillBoundariesWithBC(lev, time, energy_bc, {
                energy_per_vol_mf[lev].get(),
                energy0_mf[lev].get(),
                energy1_mf[lev].get()
            });
        }
        else
        {
            // PRIMITIVE PATH: prescribe pressure at the boundary, then
            // reconstruct per-phase internal energies from (alpha, p) via the
            // EOS -- identical formula to the NSCBC4 branch above. eta ghosts
            // are already filled (STEP 3); rho/momentum ghosts from above.
            // rho E (energy_per_vol) is rebuilt in STEP 7 as E0+E1+KE.
            FillBoundariesWithBC(lev, time, pressure_bc, {
                pressure_mf[lev].get()
            });

            const Set::Scalar g0p = eos0.Gamma(), pi0p = eos0.P0();
            const Set::Scalar g1p = eos1.Gamma(), pi1p = eos1.P0();
            const Set::Scalar smp = small;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
            {
                const amrex::Box &ghostbox = mfi.growntilebox(nghost);
                auto eta   = eta_mf[lev]->array(mfi);
                auto press = pressure_mf[lev]->array(mfi);
                auto E0    = energy0_mf[lev]->array(mfi);
                auto E1    = energy1_mf[lev]->array(mfi);
                amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const Set::Scalar a1  = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                    const Set::Scalar a2  = 1.0 - a1;
                    const Set::Scalar p_g = std::max(press(i, j, k), smp);
                    E0(i, j, k) = a1 * (p_g + g0p * pi0p) / std::max(g0p - 1.0, smp);
                    E1(i, j, k) = a2 * (p_g + g1p * pi1p) / std::max(g1p - 1.0, smp);
                });
            }
        }

        // Zero Gradient Fill
        // Layers 3..nghost of an OUT-OF-DOMAIN ghost on a non-mirror, non-periodic
        // face get a copy of layer 2 (the validated far-field behaviour).  Every
        // other ghost is left as FillBoundariesWithBC wrote it.  The first version
        // clamped every ghost outside growntilebox(2): that overwrote the exact
        // MIRROR in layers 3-4 at REFLECT faces (ghost -3 got cell 1 instead of
        // cell 2, and lost the REFLECT_ODD sign change) and the same-level exchange
        // data in layers 3-4 between neighbouring boxes.  Radius-1 limiters
        // (MUSCL/MUSCL2/THINC/WENO3) never read layer 3; WENO5 does.
        if (nghost > effective_nghost)
        {
            const amrex::Box dom = geom[lev].Domain();
            const int eff = effective_nghost;
            int clamp_lo[3] = {0, 0, 0}, clamp_hi[3] = {0, 0, 0};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                clamp_lo[d] = !geom[lev].isPeriodic(d) && !sym_face_lo[d];
                clamp_hi[d] = !geom[lev].isPeriodic(d) && !sym_face_hi[d] && !radial_face_hi[d];
            }
            const int cxl = clamp_lo[0], cxh = clamp_hi[0], cyl = clamp_lo[1], cyh = clamp_hi[1];
            const int czl = clamp_lo[2], czh = clamp_hi[2];
            for (amrex::MFIter mfi(*rho_eta0_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &ghostNbox = mfi.growntilebox(nghost);

                auto rho0 = rho_eta0_mf[lev]->array(mfi);
                auto rho1 = rho_eta1_mf[lev]->array(mfi);
                auto M = momentum_mf[lev]->array(mfi);
                auto E = energy_per_vol_mf[lev]->array(mfi);
                auto eta_zg = eta_mf[lev]->array(mfi);
                auto E0_zg = energy0_mf[lev]->array(mfi);
                auto E1_zg = energy1_mf[lev]->array(mfi);

                amrex::ParallelFor(ghostNbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    // Clamp only the directions in which this ghost lies more than
                    // `eff` cells outside the DOMAIN through a clampable face.
                    int i_copy = i;
                    int j_copy = j;
                    int k_copy = k;
                    if (cxl && i < dom.smallEnd(0) - eff) i_copy = dom.smallEnd(0) - eff;
                    if (cxh && i > dom.bigEnd(0) + eff)   i_copy = dom.bigEnd(0) + eff;
                    if (cyl && j < dom.smallEnd(1) - eff) j_copy = dom.smallEnd(1) - eff;
                    if (cyh && j > dom.bigEnd(1) + eff)   j_copy = dom.bigEnd(1) + eff;
#if AMREX_SPACEDIM == 3
                    if (czl && k < dom.smallEnd(2) - eff) k_copy = dom.smallEnd(2) - eff;
                    if (czh && k > dom.bigEnd(2) + eff)   k_copy = dom.bigEnd(2) + eff;
#endif
                    if (i_copy != i || j_copy != j || k_copy != k)
                    {

                        // Zero-gradient extrapolation (copy from layer 2).
                        // MUST cover the FULL 6-eq state: this block originally
                        // copied only {rho_eta, M, E}, leaving eta/E0/E1 in
                        // ghost layers 3-4 FROZEN AT t=0 while the copied
                        // fields evolved.  WENO5's boundary-face stencil reads
                        // layer 3; ToPrimitive on the mismatched state (gam*pi
                        // amplified) corrupted the boundary Riemann solve until
                        // the OUTFLOW face pumped mass INTO the domain (UT_ADV:
                        // xlo boundary flux drifted from the exact -8.0 to
                        // +11.3 while adjacent cells stayed healthy).  Third
                        // defect fixed in this block (after the 3D k-clamp and
                        // the missing z-momentum).
                        rho0(i, j, k) = rho0(i_copy, j_copy, k_copy);
                        rho1(i, j, k) = rho1(i_copy, j_copy, k_copy);
                        M(i, j, k, 0) = M(i_copy, j_copy, k_copy, 0);
                        M(i, j, k, 1) = M(i_copy, j_copy, k_copy, 1);
#if AMREX_SPACEDIM == 3
                        M(i, j, k, 2) = M(i_copy, j_copy, k_copy, 2);
#endif
                        E(i, j, k) = E(i_copy, j_copy, k_copy);
                        eta_zg(i, j, k) = eta_zg(i_copy, j_copy, k_copy);
                        E0_zg(i, j, k) = E0_zg(i_copy, j_copy, k_copy);
                        E1_zg(i, j, k) = E1_zg(i_copy, j_copy, k_copy);
                    }
                });
            }
        }
    }

    // ------------------------------------------------------------
    // STEP 6: Update total density in GHOST CELLS; clamp eta.
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto M = momentum_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = std::max(rho_eta0(i, j, k) + rho_eta1(i, j, k), small);
            eta(i, j, k) = std::max(0.0, std::min(1.0, eta(i, j, k)));
            // VACUUM-CONSISTENT MOMENTUM: a cell whose density sits at/near
            // the floor carries no physical momentum -- retaining finite M
            // there manufactures u = M/rho_floor ~ 1e8 velocities (CFL
            // collapse to dt ~ 1e-12, then NaN; the divide-floor disease,
            // 4th appearance -- observed as the focal-corner vacuum crash
            // in the driven quadrant/octant).  1e-5 kg/m^3 is 4+ orders
            // below any physical state in the water/air system.
            if (rho(i, j, k) < 1.0e-5)
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    M(i, j, k, d) = 0.0;
        });
    }

    // ------------------------------------------------------------
    // STEP 7: Compute primitives in GHOST CELLS (6-eq primitive recovery).
    //
    // Also OVERWRITE the BC-supplied rho E in ghosts with the
    // thermodynamically consistent value  rho E = E0 + E1 + KE.
    // This avoids the prior BC-inconsistency bug class where a
    // user-supplied rho E in ghosts disagrees with the per-phase
    // energies, leading the reinit step to overwrite p with garbage.
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*velocity_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        // Capillary-energy closure inputs (see Hydro2.H).
        const int marm_g = marmottant;
        const Set::Scalar sig_g = sigma, chi_g = marmottant_chi;
        const Set::Scalar Gb_g = marmottant_Gamma_buck;
        const Set::Scalar sbrk_g = marmottant_sigma_break, sigw_g = sigma;
        auto shell_g   = shell_mf[lev]->const_array(mfi);
        auto cfun_g    = eta_mf[lev]->const_array(mfi);

        auto rho       = density_mf[lev]->array(mfi);
        auto eta       = eta_mf[lev]->array(mfi);
        auto arh0_arr  = rho_eta0_mf[lev]->array(mfi);
        auto arh1_arr  = rho_eta1_mf[lev]->array(mfi);
        auto M         = momentum_mf[lev]->array(mfi);
        auto E         = energy_per_vol_mf[lev]->array(mfi);
        auto E0_arr    = energy0_mf[lev]->array(mfi);
        auto E1_arr    = energy1_mf[lev]->array(mfi);
        auto v         = velocity_mf[lev]->array(mfi);
        auto press     = pressure_mf[lev]->array(mfi);
        auto T         = T_mf[lev]->array(mfi);
        const int Tpc = T_phase_consistent;     // temperature.phase_consistent
        auto tph = Tpc ? Tph_mf[lev]->array(mfi) : amrex::Array4<Set::Scalar>{};
        auto a         = a_mf[lev]->array(mfi);
        auto gamma     = gamma_mf[lev]->array(mfi);
        auto p0_eff    = p0_mf[lev]->array(mfi);
        auto UE        = UE_per_vol_mf[lev]->array(mfi);
        auto KE        = KE_per_vol_mf[lev]->array(mfi);
        auto rho0_arr  = density0_mf[lev]->array(mfi);
        auto rho1_arr  = density1_mf[lev]->array(mfi);
        auto p0_arr    = pressure0_mf[lev]->array(mfi);
        auto p1_arr    = pressure1_mf[lev]->array(mfi);
        const bool el_on_g = elastic.on != 0;
        amrex::Array4<const Set::Scalar> eb_g;
        if (el_on_g) eb_g = elastic.ebasis_mf[lev]->const_array(mfi);

        const Solver::EOS::Tammann eos0_base = eos0;
        const Solver::EOS::Tammann eos1_base = eos1;
        
        const Species::Eos spe_l = species.eos();     // species.eos = mixture: gamma, c_v of each phase per cell
        amrex::Array4<const Set::Scalar> sp_l; if (spe_l.on) sp_l = species.mf[lev]->const_array(mfi);
        auto cf_l = cfun_mf[lev]->const_array(mfi);     // alpha_ref of the compaction model
        // DIVIDE floor only -- must NOT be a physical clamp.  With alpha_floor
        // = 1e-6 the per-phase recovery p_k = (gam-1)E_k/alpha_k - gam*pi
        // divides E_k (built with the TRUE alpha) by the CLAMPED alpha; the
        // 1e-6 mismatch times pi = 1e9 misprices pure-liquid pressure by
        // +2.35 kPa per pass (and -1 kPa in gas via the floored trace phase),
        // a secular per-stage ratchet that dug a vacuum at the bubble center
        // in FlowDrivenBubble (NaN at step 9123).  1e-30 keeps the divide
        // finite while preserving the exact E_k/alpha_k cancellation; trace-
        // phase garbage is then weighted by the TRUE tiny alpha and cancels.
        // (Same lesson as the limiter ToPrimitive/ToState divide floor.)
        const Set::Scalar alpha_floor = 1.0e-30;

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Solver::EOS::Tammann eos0_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 0, eos0_base, E0_arr(i, j, k) / std::max(arh0_arr(i, j, k), 1.0e-300), arh0_arr(i, j, k) / std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12)) : eos0_base;
            const Solver::EOS::Tammann eos1_local = spe_l.on ? spe_l.Phase(sp_l, i, j, k, 1, eos1_base, E1_arr(i, j, k) / std::max(arh1_arr(i, j, k), 1.0e-300), arh1_arr(i, j, k) / (1.0 - std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12))) : eos1_base;
            const Set::Scalar gam0 = eos0_local.Gamma(), gam1 = eos1_local.Gamma();
            const Set::Scalar pi1_ = eos1_local.P0();
            const Set::Scalar pi0_ = eos0_local.P0() + ((elastic.on && elastic.comp_on) ? elastic.compaction(elastic.matAt(sp_l, i, j, k), std::min(std::max(eta(i, j, k), 0.0), 1.0), arh0_arr(i, j, k), cf_l(i, j, k)).Pc / gam0 : 0.0);
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            Set::Scalar arh0 = std::max(arh0_arr(i, j, k), small);
            Set::Scalar arh1 = std::max(arh1_arr(i, j, k), small);
            rho(i, j, k) = arh0 + arh1;

            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), small);

            KE(i, j, k) = 0.5 * rho(i, j, k) * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                          + v(i, j, k, 1) * v(i, j, k, 1),
                                                          + v(i, j, k, 2) * v(i, j, k, 2)));

            // Per-phase pure densities, pressures, and sound speeds.
            Set::Scalar rho0_pure = arh0 / a1;
            Set::Scalar rho1_pure = arh1 / a2;
            Set::Scalar p0_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E0_arr(i, j, k), a1, gam0, pi0_, small);
            Set::Scalar p1_loc = Solver::EOS::EOS::PhasicPressureFromEnergy(E1_arr(i, j, k), a2, gam1, pi1_, small);
            // TRACE-PHASE SLAVING (see flux-loop recovery for rationale).
            {
                // SMOOTH slaving (see flux-loop recovery block for rationale).
                const Set::Scalar SLAVE_LO = 1.0e-2, SLAVE_HI = 1.0e-1;
                Set::Scalar w0 = std::min(std::max((a1 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w0 = w0 * w0 * (3.0 - 2.0 * w0);
                Set::Scalar w1 = std::min(std::max((a2 - SLAVE_LO) / (SLAVE_HI - SLAVE_LO), 0.0), 1.0);
                w1 = w1 * w1 * (3.0 - 2.0 * w1);
                const Set::Scalar p0_raw = p0_loc, p1_raw = p1_loc;
                p0_loc = w0 * p0_raw + (1.0 - w0) * p1_raw;
                p1_loc = w1 * p1_raw + (1.0 - w1) * p0_raw;
            }
            Set::Scalar c0_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho0_pure, p0_loc, gam0, pi0_, small);
            Set::Scalar c1_loc = Solver::EOS::EOS::PhasicSoundSpeed(rho1_pure, p1_loc, gam1, pi1_, small);

            Set::Scalar Y0 = arh0 / std::max(rho(i, j, k), small);
            Set::Scalar Y1 = 1.0 - Y0;

            press(i, j, k) = a1 * p0_loc + a2 * p1_loc;
            a(i, j, k)     = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0_loc, c1_loc);

            // Total energy in the ghost CONSISTENT with the per-phase energies:
            // rho E = sum_k (alpha rho e)_k + 0.5 rho |u|^2  (Sch20 eq. 16).
            // This overwrite is the BC-consistency guarantee for the 6-eq model.
            UE(i, j, k) = E0_arr(i, j, k) + E1_arr(i, j, k);
            E(i, j, k)  = UE(i, j, k) + KE(i, j, k);
            if (el_on_g)   // elastic solid: rho E also carries the elastic energy W (Favrie 2009 eq. 42)
                E(i, j, k) += elastic.energy(eb_g, i, j, k, arh0_arr(i, j, k), arh1_arr(i, j, k), sp_l, std::min(std::max(eta(i, j, k), 0.0), 1.0), cf_l(i, j, k));

            // Diagnostic per-phase primitives.
            rho0_arr(i, j, k) = rho0_pure;
            rho1_arr(i, j, k) = rho1_pure;
            p0_arr(i, j, k)   = p0_loc;
            p1_arr(i, j, k)   = p1_loc;

            // Diagnostic mixture gamma / p0 / T (NOT used by hyperbolic step).
            gamma(i, j, k)  = Solver::EOS::EOS::MixedGamma(a1, eos0_local, eos1_local);
            p0_eff(i, j, k) = Solver::EOS::EOS::MixedP0(a1, eos0_local, eos1_local);
            if (Tpc)
            {
                const Species::Eos &se_t = spe_l;
                const bool pe0 = se_t.on && se_t.th.on && se_t.th.ph == 0, pe1 = se_t.on && se_t.th.on && se_t.th.ph == 1;
                Set::Scalar t0, t1;
                T(i, j, k) = PhaseConsistentT(E0_arr(i, j, k), E1_arr(i, j, k), arh0_arr(i, j, k), arh1_arr(i, j, k), a1, eos0_local, eos1_local, pe0, pe1, t0, t1);
                tph(i, j, k, 0) = t0; tph(i, j, k, 1) = t1;
            }
            else T(i, j, k)      = Solver::EOS::EOS::MixedTemperature(rho(i, j, k), press(i, j, k), a1, eos0_local, eos1_local, pref);
        });
    }

    // ------------------------------------------------------------
    // STEP 8: Repair any remaining NaN in ALL cells (domain + ghosts)
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*density_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
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

        const amrex::Box vbx8 = mfi.validbox();
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Check for NaN in any field
            bool has_nan = !std::isfinite(rho_eta0(i, j, k)) || !std::isfinite(rho_eta1(i, j, k)) || !std::isfinite(rho(i, j, k)) || !std::isfinite(eta(i, j, k)) || !std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1)) || !std::isfinite(E(i, j, k)) || !std::isfinite(press(i, j, k)) || !std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1)) || !std::isfinite(T(i, j, k)) || !std::isfinite(a(i, j, k)) || !std::isfinite(gamma(i, j, k)) || !std::isfinite(p0_eff(i, j, k));

            if (has_nan)
            {
                // [DIAG] STEP-8 repair ledger: this overwrite REPLACES the
                // cell's per-phase MASSES with a neighbor's -- silent mass
                // creation/destruction if it ever fires on valid cells.
                {
                    char b[220];
                    snprintf(b, 220, "STEP8REPAIR lev=%d i=%d j=%d valid=%d re0=%.3e re1=%.3e "
                             "finite[re0=%d re1=%d rho=%d p=%d v=%d T=%d a=%d gam=%d]",
                             lev, i, j,
                             (int)vbx8.contains(amrex::IntVect(AMREX_D_DECL(i, j, k))),
                             (double)rho_eta0(i, j, k), (double)rho_eta1(i, j, k),
                             (int)std::isfinite(rho_eta0(i, j, k)), (int)std::isfinite(rho_eta1(i, j, k)),
                             (int)std::isfinite(rho(i, j, k)), (int)std::isfinite(press(i, j, k)),
                             (int)std::isfinite(v(i, j, k, 0)), (int)std::isfinite(T(i, j, k)),
                             (int)std::isfinite(a(i, j, k)), (int)std::isfinite(gamma(i, j, k)));
                    Util::Message(INFO, b);
                }
                // Repair by copying from nearest valid neighbor

                // Try left neighbor (i-1)
                if (i > ghostbox.smallEnd(0))
                {
                    bool left_valid = std::isfinite(rho_eta0(i - 1, j, k)) && std::isfinite(E(i - 1, j, k)) && std::isfinite(press(i - 1, j, k));
                    if (left_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i - 1, j, k);
                        rho_eta1(i, j, k) = rho_eta1(i - 1, j, k);
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
                    bool right_valid = std::isfinite(rho_eta0(i + 1, j, k)) && std::isfinite(E(i + 1, j, k)) && std::isfinite(press(i + 1, j, k));
                    if (right_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i + 1, j, k);
                        rho_eta1(i, j, k) = rho_eta1(i + 1, j, k);
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
                    bool bottom_valid = std::isfinite(rho_eta0(i, j - 1, k)) && std::isfinite(E(i, j - 1, k)) && std::isfinite(press(i, j - 1, k));
                    if (bottom_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i, j - 1, k);
                        rho_eta1(i, j, k) = rho_eta1(i, j - 1, k);
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
                    bool top_valid = std::isfinite(rho_eta0(i, j + 1, k)) && std::isfinite(E(i, j + 1, k)) && std::isfinite(press(i, j + 1, k));
                    if (top_valid)
                    {
                        rho_eta0(i, j, k) = rho_eta0(i, j + 1, k);
                        rho_eta1(i, j, k) = rho_eta1(i, j + 1, k);
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
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*density_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho = density_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);

        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
        });
    }

    // ------------------------------------------------------------
    // STEP 9: Enforce consistency and positivity in ALL cells
    // ------------------------------------------------------------
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion() && !relax_diag)
#endif
    for (amrex::MFIter mfi(*density_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &ghostbox = mfi.growntilebox(nghost);

        auto rho_eta0 = rho_eta0_mf[lev]->array(mfi);
        auto rho_eta1 = rho_eta1_mf[lev]->array(mfi);
        auto rho = density_mf[lev]->array(mfi);
        auto eta = eta_mf[lev]->array(mfi);
        auto press = pressure_mf[lev]->array(mfi);
        auto E = energy_per_vol_mf[lev]->array(mfi);

        // [DIAG] STEP-9 clip ledger on VALID cells (the uninstrumented mass
        // source: this clip runs EVERY RK STAGE via post_stage FillGhost and
        // growntilebox INCLUDES the valid region).  Gated on relax_diag.
        if (relax_diag)
        {
            const amrex::Box &vbx9 = mfi.tilebox();   // tile (== validbox when untiled)
            Set::Scalar c0 = 0.0, c1 = 0.0;
            amrex::LoopOnCpu(vbx9, [&](int i, int j, int k) {
                if (rho_eta0(i, j, k) < 0.0) c0 -= rho_eta0(i, j, k);
                if (rho_eta1(i, j, k) < 0.0) c1 -= rho_eta1(i, j, k);
            });
            if (c0 > 0.0 || c1 > 0.0)
            {
                static Set::Scalar cum0 = 0.0, cum1 = 0.0;
                static int nrep = 0;
                cum0 += c0; cum1 += c1;
                if ((nrep++ % 500) == 0)
                {
                    char b[200];
                    snprintf(b, 200, "STEP9CLIP valid-created0=%.4e valid-created1=%.4e cum0=%.4e cum1=%.4e",
                             c0, c1, cum0, cum1);
                    Util::Message(INFO, b);
                }
            }
        }
        // FIX (gated on clip_ghost_only): restrict the STEP-9 positivity clip
        // to GHOST cells.  Valid-cell mid-stage undershoots are transient RK
        // stage states -- zeroing them CREATES phase mass every stage (the
        // quadrant/octant driven-bubble mass instability); the final combined
        // state is protected by the post-advance clip, which is measured and
        // near-silent.
        const amrex::Box vbx_keep = mfi.validbox();
        const int ghost_only = clip_ghost_only;
        amrex::ParallelFor(ghostbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const bool in_valid = vbx_keep.contains(amrex::IntVect(AMREX_D_DECL(i, j, k)));
            // Enforce rho = rho_eta0 + rho_eta1
            Set::Scalar rho_calc = rho_eta0(i, j, k) + rho_eta1(i, j, k);
            rho(i, j, k) = rho_calc;

            // Enforce positivity (rho_eta: NEGATIVITY clip only; and with
            // clip_ghost_only=1, ghosts only -- valid-cell stage states pass
            // through untouched to preserve per-phase mass conservation)
            rho(i, j, k) = std::max(rho(i, j, k), small);
            if (!ghost_only || !in_valid)
            {
                rho_eta0(i, j, k) = std::max(rho_eta0(i, j, k), 0.0);
                rho_eta1(i, j, k) = std::max(rho_eta1(i, j, k), 0.0);
            }
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

    // [OPT] Cheap parallel pre-check.  The detailed serial scan below used
    // to run over every grown box on every call (7x per step); it now runs
    // only when this rank actually holds a non-finite value in one of the
    // same fields, so the report and the abort are unchanged.
    bool any_nonfinite = false;
    {
        amrex::ReduceOps<amrex::ReduceOpMax> nf_op;
        amrex::ReduceData<int> nf_data(nf_op);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(*density_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
        {
            const amrex::Box &ghostbox = mfi.growntilebox(nghost);
            auto rho = density_mf[lev]->const_array(mfi);
            auto M = momentum_mf[lev]->const_array(mfi);
            auto E = energy_per_vol_mf[lev]->const_array(mfi);
            auto p = pressure_mf[lev]->const_array(mfi);
            auto v = velocity_mf[lev]->const_array(mfi);
            auto eta = eta_mf[lev]->const_array(mfi);
            nf_op.eval(ghostbox, nf_data, [=] AMREX_GPU_DEVICE (int i, int j, int k) -> amrex::GpuTuple<int> {
                const bool bad = !std::isfinite(rho(i, j, k))
                              || !std::isfinite(M(i, j, k, 0)) || !std::isfinite(M(i, j, k, 1))
                              || !std::isfinite(E(i, j, k))
                              || !std::isfinite(p(i, j, k))
                              || !std::isfinite(v(i, j, k, 0)) || !std::isfinite(v(i, j, k, 1))
                              || !std::isfinite(eta(i, j, k));
                return { bad ? 1 : 0 };
            });
        }
        any_nonfinite = (amrex::get<0>(nf_data.value(nf_op)) > 0);
    }

    // Check each field and report first NaN location
    if (any_nonfinite)
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

    // ================================================================
    // REFLECT-MIRROR AUDIT (temporary diagnostic, gated on relax_diag):
    // at xlo/ylo domain faces with REFLECT BCs, ghost layer -g must be
    // the exact mirror of interior layer +g-1: even fields equal, normal
    // momentum negated.  Any asymmetry here seeds a nonzero mass flux
    // through the symmetry plane (quadrant/octant gas-mass instability).
    // ================================================================
    if (relax_diag)
    {
        const amrex::Box &dom = geom[lev].Domain();
        Set::Scalar a_eta = 0, a_r0 = 0, a_r1 = 0, a_Mn = 0, a_E0 = 0, a_E1 = 0;
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &vbx = mfi.validbox();
            if (vbx.smallEnd(0) != dom.smallEnd(0)) continue;
            auto ge = eta_mf[lev]->array(mfi);
            auto g0 = rho_eta0_mf[lev]->array(mfi);
            auto g1 = rho_eta1_mf[lev]->array(mfi);
            auto gM = momentum_mf[lev]->array(mfi);
            auto gE0 = energy0_mf[lev]->array(mfi);
            auto gE1 = energy1_mf[lev]->array(mfi);
            const int klo = (AMREX_SPACEDIM == 3) ? vbx.smallEnd(2) : 0;
            const int khi = (AMREX_SPACEDIM == 3) ? vbx.bigEnd(2) : 0;
            for (int k = klo; k <= khi; ++k)
                for (int j = vbx.smallEnd(1); j <= vbx.bigEnd(1); ++j)
                    for (int g = 1; g <= nghost; ++g)
                    {
                        const int ig = dom.smallEnd(0) - g;      // ghost
                        const int im = dom.smallEnd(0) + g - 1;  // mirror
                        a_eta = std::max(a_eta, std::abs(ge(ig,j,k) - ge(im,j,k)));
                        a_r0  = std::max(a_r0,  std::abs(g0(ig,j,k) - g0(im,j,k)));
                        a_r1  = std::max(a_r1,  std::abs(g1(ig,j,k) - g1(im,j,k)));
                        a_Mn  = std::max(a_Mn,  std::abs(gM(ig,j,k,0) + gM(im,j,k,0)));
                        a_E0  = std::max(a_E0,  std::abs(gE0(ig,j,k) - gE0(im,j,k)));
                        a_E1  = std::max(a_E1,  std::abs(gE1(ig,j,k) - gE1(im,j,k)));
                    }
        }
        static int _ma_count = 0;
        if ((_ma_count++ % 200) == 0 && (a_eta + a_r0 + a_r1 + a_Mn + a_E0 + a_E1) > 0)
        {
            char b[240];
            snprintf(b, 240, "MIRRORAUDIT lev=%d asym: eta=%.3e r0=%.3e r1=%.3e Mn=%.3e E0=%.3e E1=%.3e",
                     lev, a_eta, a_r0, a_r1, a_Mn, a_E0, a_E1);
            Util::Message(INFO, b);
        }
    }
    if (diag_gasmass) GasLedger(lev, "fillghost_out", time);
    if (species.on)
    {
        species.bc->define(geom[lev]);
        if (lev > 0) FillPatch(lev, time, species.mf, *species.mf[lev], *species.bc, 0);   // coarse-fine ghosts (2026-11-18)
        else FillBoundariesWithBC(lev, time, species.bc, { species.mf[lev].get() });
        CloseSpecies(lev, true);      // diagnostic mass fractions only (valid cells were closed after the stage)
    }
    if (entropy_on) ComputeEntropy(lev);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK fillghost_out lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
} // end FillGhost4BC()


////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////// RELAXATION + REINITIALIZATION //////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
//
// Stiff pressure-relaxation (mu -> infinity) + reinitialization step for the
// 6-equation diffuse-interface model.
//
//   Step 1 (relaxation, Schmidmayer 2020 Sec.3.3, eq. (24)+(25); Saurel 2009 Sec.3.3,
//   eq. (III.4)):  per-cell Newton on
//                     f(p) = sum_k (alpha rho)_k v_k(p) - 1 = 0
//   with v_k from Saurel eq. (III.4) (frozen-p_hat_I form).  Updates alpha_k
//   holding (alpha rho)_k, momentum, and rho E fixed.
//
//   Step 2 (reinit, Schmidmayer 2020 eq. (26); Saurel 2009 eq. (III.5)):
//   recompute mixture p from the redundant rho E using the post-relaxation
//   alpha_k; then reset each per-phase (alpha rho e)_k from
//   (alpha rho e)_k = alpha_k (p + gamma_k pi_k)/(gamma_k - 1).
//
// Gamma is now a transported PRIMARY that never references |grad eta|, so it is
// already immune to non-material changes in eta and needs no snapshot/restore.
// Kept as no-ops so the call sites document where eta is rewritten without any
// material motion, should a future formulation need them again.



// ======================================================================
// L_cap : Schmidmayer 2017 eq. (17) / sec. 4.2.2.  See Hydro2.H.
// ======================================================================
//
// SUB-CYCLED SHELL DILATATIONAL VISCOUS UPDATE
// ---------------------------------------------------------------------------
// kappa_s enters momentum as div(Omega_visc), Omega_visc = |grad eta| kappa_s
// (div_s u) P.  That is a SURFACE momentum diffusion with kinematic diffusivity
// nu_s = kappa_s |grad eta| / rho.  Because the decks hold eps/dx ~ 1,
// |grad eta| ~ 1/dx, so nu_s ~ 1/dx and its explicit stability limit
// dt <= dx^2/(2 d nu_s) scales as dx^3 while the acoustic limit scales as dx.
// Letting it set the global step made max_level 5 cost ~83,000 base steps.
//
// Instead this advances ONLY that term, on the band, in N sub-steps of dt/N
// with N from the local stability limit.  Structure: nearest-neighbour FillBoundary per sub-step, no
// global reductions, band cells only -- so the extra work is <1% of a hydro
// step (the band is ~5e4 cells against 6.4M at L5) and it adds no collective.
//
// ENERGY.  The bulk Omega is treated as a REVERSIBLE capillary stress whose
// Omega:grad(u) exchange feeds the surface reservoir sigma_eff|grad eta|.  The
// viscous part is NOT reversible and has no reservoir: its work rate is
//     Omega_visc : grad u = kappa_s |grad eta| (div_s u)^2  >= 0,
// positive-definite, so it is deposited as HEAT in the internal energy.  This
// is why splitting it out is more correct than folding it into the scalar
// tension, where it could drive sigma_tot negative and had to be clamped.
//
// COARSE-FINE.  The coarse level does not advance during the sub-cycle, so its
// contribution to the band ghost velocity is held fixed across the N sub-steps
// -- first order over an interval of one hydro dt, which is consistent with the
// subcycled-AMR treatment of the other source terms.
//
// TEMP jump.face_force: ghost cells outside the physical domain take the adjacent valid value
static void FaceForceDomainGhosts(amrex::MultiFab &J, const amrex::Geometry &gm)
{
    const amrex::Box dom = gm.Domain();
    for (amrex::MFIter mfi(J, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box gb = mfi.growntilebox(1);
        auto a = J.array(mfi);
        const amrex::IntVect lo = dom.smallEnd(), hi = dom.bigEnd();
        amrex::LoopOnCpu(gb, [&](int i, int j, int k) {
            amrex::IntVect iv(AMREX_D_DECL(i, j, k));
            if (dom.contains(iv)) return;
            amrex::IntVect c = iv;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (gm.isPeriodic(d)) continue;
                c[d] = std::min(std::max(c[d], lo[d]), hi[d]);
            }
            if (c == iv) return;
            if (!mfi.validbox().contains(c)) return;
            a(iv) = a(c);
        });
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// DiffuseSpecies (species.diffusion = 1, 2026-11-18; design stage 6).  Fickian diffusion of the species of one phase k
// inside that phase, once per step after the hyperbolic advance, in N explicit sub-steps.  Face flux between cells L|R:
//     j_i = -Dm (Y_i,R - Y_i,L)/dx + Yf_i sum_l Dm (Y_l,R - Y_l,L)/dx,      Dm = (alpha rho)_k,f D   or   alpha_f mu_k/Sc,
// Yf = face mean of the mass fractions (normalised), so sum_i j_i = 0 exactly: the phase mass, eta and the flow are
// untouched.  Energy flux q = sum_i h_i(T_f) j_i on E_k and rho E (zero when the species share one EOS).  Faces only
// where both cells hold alpha_k >= alpha_min and phase mass.  Each face flux is computed identically from both sides
// (conservative to round-off).  Single level: no reflux of these fluxes at coarse-fine faces (as the species advection).
// ---------------------------------------------------------------------------------------------------------------------
void Hydro2::DiffuseSpecies(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::DiffuseSpecies");
    if (!species.on || !species.diff_on || !(dt > 0.0)) return;
    const int ph = species.diff_phase;
    const int n = ph ? species.n1 : species.n0, c0 = ph ? species.n0 : 0;
    if (n < 2) return;
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar dxm = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});
    const Set::Scalar amin = species.diff_alpha_min, Dc = species.diff_D, Sc = species.diff_Sc;
    const Set::Scalar muk = ph ? mu1 : mu0;
    const Set::Scalar tn = time + dt;
    if (Sc > 0.0 && !(muk > 0.0)) Util::Abort(INFO, "species.diffusion.Sc needs a viscosity for the phase (mu", ph, " > 0)");

    // enthalpy of species s at T:  mixture h = gamma_s cv_s T;  pele: NASA sensible enthalpy;  otherwise none
    const Species::Eos spe = species.eos();
    const int hmode = (spe.on && spe.th.on && spe.th.ph == ph) ? 2 : ((spe.on && !spe.has_pi[ph]) ? 1 : 0);
    const Solver::EOS::Tammann base = ph ? eos1 : eos0;

    auto fill = [&](bool all) {
        if (lev > 0)
        {
            FillPatch(lev, tn, species.mf, *species.mf[lev], *species.bc, 0);
            if (all)
            {
                FillPatch(lev, tn, rho_eta0_mf, *rho_eta0_mf[lev], *density_bc, 0);
                FillPatch(lev, tn, rho_eta1_mf, *rho_eta1_mf[lev], *density_bc, 0);
                FillPatch(lev, tn, eta_mf,      *eta_mf[lev],      *eta_bc,     0);
            }
            FillPatch(lev, tn, energy0_mf, *energy0_mf[lev], *energy_bc, 0);
            FillPatch(lev, tn, energy1_mf, *energy1_mf[lev], *energy_bc, 0);
        }
        else
        {
            FillBoundariesWithBC(lev, tn, species.bc, { species.mf[lev].get() });
            if (all)
            {
                FillBoundariesWithBC(lev, tn, density_bc, { rho_eta0_mf[lev].get(), rho_eta1_mf[lev].get() });
                FillBoundariesWithBC(lev, tn, eta_bc, { eta_mf[lev].get() });
            }
            FillBoundariesWithBC(lev, tn, energy_bc, { energy0_mf[lev].get(), energy1_mf[lev].get() });
        }
    };
    fill(true);

    // ---- stability bound ----
    Set::Scalar Dmax = Dc;
    if (Sc > 0.0)
    {
        amrex::ReduceOps<amrex::ReduceOpMax> rop;
        amrex::ReduceData<Set::Scalar> rdat(rop);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            auto eta = eta_mf[lev]->const_array(mfi);
            auto mk  = ph ? rho_eta1_mf[lev]->const_array(mfi) : rho_eta0_mf[lev]->const_array(mfi);
            rop.eval(mfi.validbox(), rdat, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar> {
                const Set::Scalar a = ph ? 1.0 - eta(i, j, k) : eta(i, j, k);
                if (a < amin || !(mk(i, j, k) > 1.0e-300)) return { 0.0 };
                return { muk / (Sc * mk(i, j, k) / a) };
            });
        }
        Dmax = amrex::get<0>(rdat.value());
        amrex::ParallelDescriptor::ReduceRealMax(Dmax);
    }
    if (!(Dmax > 0.0)) return;
    const Set::Scalar dtau_max = species.diff_cfl * dxm * dxm / (2.0 * AMREX_SPACEDIM * Dmax);
    long long Nll = (long long)std::ceil((long double)dt / (long double)dtau_max);
    if (Nll < 1LL) Nll = 1LL;
    if (Nll > (long long)species.diff_max_sub)
    {
        Util::Warning(INFO, "DiffuseSpecies lev=", lev, ": required N=", Nll, " > species.diffusion.max_sub=", species.diff_max_sub, "; clamping (UNSTABLE)");
        Nll = species.diff_max_sub;
    }
    const int N = (int)Nll;
    const Set::Scalar dtau = dt / (Set::Scalar)N;
    if (species.diff_verbose)
        Util::Message(INFO, "DiffuseSpecies lev=", lev, " N=", N, " dtau=", dtau, " (dt=", dt, ", D_max=", Dmax, ")");

    amrex::MultiFab dS(species.mf[lev]->boxArray(), species.mf[lev]->DistributionMap(), n + 1, 0);
    for (int it = 0; it < N; ++it)
    {
        if (it > 0) fill(false);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            auto eta = eta_mf[lev]->const_array(mfi);
            auto mk  = ph ? rho_eta1_mf[lev]->const_array(mfi) : rho_eta0_mf[lev]->const_array(mfi);
            auto Ek  = ph ? energy1_mf[lev]->const_array(mfi)  : energy0_mf[lev]->const_array(mfi);
            auto m   = species.mf[lev]->const_array(mfi);
            auto out = dS.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                // phase temperature of a cell (Le Metayer form, the phase constants of its species), or < 0
                auto Tcell = [&](int ii, int jj, int kk) -> Set::Scalar {
                    const Set::Scalar a = ph ? 1.0 - eta(ii, jj, kk) : eta(ii, jj, kk), mm = mk(ii, jj, kk);
                    if (hmode == 0 || a < amin || !(mm > 1.0e-300)) return -1.0;
                    const Set::Scalar es = Ek(ii, jj, kk) / mm;
                    Set::Scalar gam = base.Gamma(), cvm = base.Cv();
                    spe.Mix(m, ii, jj, kk, ph, gam, cvm, es);
                    return (es - base.P0() * a / mm) / cvm;
                };
                Set::Scalar acc[Species::NMAX + 1];
                for (int s = 0; s <= n; ++s) acc[s] = 0.0;
                // flux from L to R across the face between L and R (direction d), added with sign sg to this cell
                auto face = [&](int iL, int jL, int kL, int iR, int jR, int kR, int d, Set::Scalar sg) {
                    const Set::Scalar aL = ph ? 1.0 - eta(iL, jL, kL) : eta(iL, jL, kL);
                    const Set::Scalar aR = ph ? 1.0 - eta(iR, jR, kR) : eta(iR, jR, kR);
                    const Set::Scalar mL = mk(iL, jL, kL), mR = mk(iR, jR, kR);
                    if (aL < amin || aR < amin || !(mL > 1.0e-300) || !(mR > 1.0e-300)) return;
                    const Set::Scalar Dm = (Sc > 0.0) ? 0.5 * (aL + aR) * muk / Sc : 0.5 * (mL + mR) * Dc;
                    Set::Scalar YL[Species::NMAX], YR[Species::NMAX], Yf[Species::NMAX], jf[Species::NMAX];
                    Set::Scalar sL = 0.0, sR = 0.0;
                    for (int s = 0; s < n; ++s)
                    {
                        YL[s] = std::max(m(iL, jL, kL, c0 + s), 0.0); sL += YL[s];
                        YR[s] = std::max(m(iR, jR, kR, c0 + s), 0.0); sR += YR[s];
                    }
                    if (!(sL > 0.0) || !(sR > 0.0)) return;
                    Set::Scalar sj = 0.0, sf = 0.0;
                    for (int s = 0; s < n; ++s)
                    {
                        YL[s] /= sL; YR[s] /= sR;
                        jf[s] = -Dm * (YR[s] - YL[s]) / DX[d]; sj += jf[s];
                        Yf[s] = 0.5 * (YL[s] + YR[s]); sf += Yf[s];
                    }
                    for (int s = 0; s < n; ++s) jf[s] -= (Yf[s] / sf) * sj;
                    Set::Scalar q = 0.0;
                    if (hmode)
                    {
                        const Set::Scalar TL = Tcell(iL, jL, kL), TR = Tcell(iR, jR, kR);
                        if (TL > 0.0 && TR > 0.0)
                        {
                            const Set::Scalar Tf = 0.5 * (TL + TR);
                            if (hmode == 2)
                            {
                                Set::Scalar h[Pele::NS > 0 ? Pele::NS : 1];
                                spe.th.Hs(Tf, h);
                                for (int s = 0; s < n && s < Pele::NS; ++s) q += h[s] * jf[s];
                            }
                            else
                                for (int s = 0; s < n; ++s) q += spe.g[c0 + s] * spe.cv[c0 + s] * Tf * jf[s];
                        }
                    }
                    for (int s = 0; s < n; ++s) acc[s] += sg * jf[s] / DX[d];
                    acc[n] += sg * q / DX[d];
                };
                face(i - 1, j, k, i, j, k, 0, +1.0); face(i, j, k, i + 1, j, k, 0, -1.0);
#if AMREX_SPACEDIM > 1
                face(i, j - 1, k, i, j, k, 1, +1.0); face(i, j, k, i, j + 1, k, 1, -1.0);
#endif
#if AMREX_SPACEDIM > 2
                face(i, j, k - 1, i, j, k, 2, +1.0); face(i, j, k, i, j, k + 1, 2, -1.0);
#endif
                for (int s = 0; s <= n; ++s) out(i, j, k, s) = acc[s];
            });
        }
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            auto m  = species.mf[lev]->array(mfi);
            auto Ek = ph ? energy1_mf[lev]->array(mfi) : energy0_mf[lev]->array(mfi);
            auto E  = energy_per_vol_mf[lev]->array(mfi);
            auto d_ = dS.const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                for (int s = 0; s < n; ++s) m(i, j, k, c0 + s) += dtau * d_(i, j, k, s);
                Ek(i, j, k) += dtau * d_(i, j, k, n);
                E(i, j, k)  += dtau * d_(i, j, k, n);
            });
        }
    }
    CloseSpecies(lev, false);
}

// ---------------------------------------------------------------------------------------------------------------------
// SubcycleBulkViscous (visc_subcycle = 1; ported 2026-11-18 from flames2, SymFix branch).  The bulk viscous stress as
// its own operator after the hyperbolic advance:  N forward-Euler sub-steps of dtau = dt/N on momentum and total energy,
//     M += dtau div(tau),   rho E += dtau (u . div(tau) + Phi),
// with N from the explicit limit of the discrete stress operator, dtau <= visc_subcycle_cfl dx^2 / (2 nu_L),
// nu_L = max (|lambda| + 4/3 mu + (d-1) mu)/rho.  rho_k, eta and E_k are untouched (RelaxAndReinit re-partitions E).
// The discrete stress is the RHS one, including gradu_from_velocity, visc_flux_form and visc_harmonic.  Differences
// from flames2: those three options are honoured (flames2 has the product-rule gradients only, which amplify the
// viscous force by the density ratio at an interface), and the density / eta ghosts are refreshed first (after the RK
// advance they are one stage old).
// ---------------------------------------------------------------------------------------------------------------------
void Hydro2::SubcycleBulkViscous(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::SubcycleBulkViscous");
    if (!visc_subcycle) return;
    if (mu0 == 0.0 && mu1 == 0.0 && mu0_b == 0.0 && mu1_b == 0.0) return;
    if (!(dt > 0.0)) return;

    const Set::Scalar *DX = geom[lev].CellSize();
    const amrex::Box domain = geom[lev].Domain();
    const Set::Scalar dxm = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});
    const Set::Scalar sm = small;
    const Set::Scalar m0 = mu0, m1 = mu1, l0 = mu0_b, l1 = mu1_b;
    const int fflux = visc_flux_form;

    // mixture viscosity, same closures as RHS
    auto mu_of = [=] AMREX_GPU_DEVICE(Set::Scalar e_raw, Set::Scalar &dmu_de) -> Set::Scalar {
        dmu_de = (m0 - m1);
        return e_raw * m0 + (1.0 - e_raw) * m1;
    };

    // ---- stability bound -> number of sub-steps ----------------------------
    Set::Scalar nuL = 0.0;
    {
        amrex::ReduceOps<amrex::ReduceOpMax> rop;
        amrex::ReduceData<Set::Scalar> rdat(rop);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto rho = density_mf[lev]->const_array(mfi);
            rop.eval(bx, rdat, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar> {
                Set::Scalar dm;
                Set::Scalar mu = mu_of(eta(i, j, k), dm);
                Set::Scalar lam = eta(i, j, k) * l0 + (1.0 - eta(i, j, k)) * l1;
                if (fflux)   // flux form: the largest face viscosity of the cell sets its diffusion rate
                {
                    const Set::Scalar mc = mu, lc = lam;
                    mu = 0.0; lam = 0.0;
                    for (int q = 0; q < AMREX_SPACEDIM; ++q)
                        for (int sg = -1; sg <= 1; sg += 2)
                        {
                            const int a = i + (q == 0 ? sg : 0), b = j + (q == 1 ? sg : 0), c = k + (q == 2 ? sg : 0);
                            Set::Scalar d2;
                            const Set::Scalar mn = mu_of(eta(a, b, c), d2);
                            const Set::Scalar ln = eta(a, b, c) * l0 + (1.0 - eta(a, b, c)) * l1;
                            mu  = std::max(mu, HarmonicMean(mc, mn));
                            lam = std::max(lam, std::abs(HarmonicMean(lc, ln)));
                        }
                }
                const Set::Scalar nu = (std::abs(lam) + (4.0 / 3.0) * std::abs(mu) + (AMREX_SPACEDIM - 1) * std::abs(mu))
                                       / std::max(rho(i, j, k), sm);
                return {nu};
            });
        }
        nuL = amrex::get<0>(rdat.value());
        amrex::ParallelDescriptor::ReduceRealMax(nuL);
    }
    if (!(nuL > 0.0)) return;
    const Set::Scalar dtau_max = visc_subcycle_cfl * dxm * dxm / (2.0 * nuL);
    long long Nll = (long long)std::ceil((long double)dt / (long double)dtau_max);
    if (Nll < 1LL) Nll = 1LL;
    if (Nll > (long long)visc_subcycle_max)
    {
        Util::Warning(INFO, "SubcycleBulkViscous lev=", lev, ": required N=", Nll,
                      " > visc_subcycle_max=", visc_subcycle_max, "; clamping (UNSTABLE -- raise the cap or lower dt)");
        Nll = visc_subcycle_max;
    }
    const int N = (int)Nll;
    const Set::Scalar dtau = dt / (Set::Scalar)N;
    if (visc_subcycle_verbose)
        Util::Message(INFO, "SubcycleBulkViscous lev=", lev, " N=", N, " dtau=", dtau,
                      " (dt=", dt, ", dtau_max=", dtau_max, ", nu_L=", nuL, ")");

    const int symc = cen_face_any;   // symmetry + RADIAL faces
    const bool symlo_c[3] = {cen_face_lo[0], cen_face_lo[1], cen_face_lo[2]};
    const bool symhi_c[3] = {cen_face_hi[0], cen_face_hi[1], cen_face_hi[2]};

    for (int it = 0; it < N; ++it)
    {
        // momentum ghosts (rho ghosts are fixed during the viscous operator)
        // field BCs on every face, NSCBC runs included (see SubcycleShellViscous)
        if (lev > 0)
        {
            FillPatch(lev, time + dt, momentum_mf, *momentum_mf[lev], *momentum_bc, 0);
            if (radial_bc) RadialGhostFill(lev, *momentum_mf[lev], true);
        }
        else
            FillBoundariesWithBC(lev, time + dt, momentum_bc, { momentum_mf[lev].get() });

        // div(tau) and Phi from the start-of-sub-step momentum (scratch), then update
        amrex::MultiFab Fv(momentum_mf[lev]->boxArray(), momentum_mf[lev]->DistributionMap(), AMREX_SPACEDIM + 1, 0);
        if (fflux)
        {
            // velocity and cell viscosities on valid + 1 ghost (corners included: FillPatch fills them)
            amrex::MultiFab Uv(momentum_mf[lev]->boxArray(), momentum_mf[lev]->DistributionMap(), AMREX_SPACEDIM, 1);
            amrex::MultiFab Mu(momentum_mf[lev]->boxArray(), momentum_mf[lev]->DistributionMap(), 2, 1);
            for (amrex::MFIter mfi(Uv, false); mfi.isValid(); ++mfi)
            {
                const amrex::Box gb = mfi.growntilebox(1);
                auto M = momentum_mf[lev]->const_array(mfi);
                auto rho = density_mf[lev]->const_array(mfi);
                auto eta = eta_mf[lev]->const_array(mfi);
                auto U = Uv.array(mfi);
                auto W = Mu.array(mfi);
                amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    const Set::Scalar rc = std::max(rho(i, j, k), sm);
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) U(i, j, k, d) = M(i, j, k, d) / rc;
                    Set::Scalar dm;
                    W(i, j, k, 0) = mu_of(eta(i, j, k), dm);
                    W(i, j, k, 1) = eta(i, j, k) * l0 + (1.0 - eta(i, j, k)) * l1;
                });
            }
            for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                auto U = Uv.const_array(mfi);
                auto W = Mu.const_array(mfi);
                auto F = Fv.array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
                {
                    Set::Scalar divtau[3] = {0, 0, 0}, divw = 0.0;
                    for (int q = 0; q < AMREX_SPACEDIM; ++q)
                        for (int side = 0; side < 2; ++side)
                        {
                            // face between L and R = L + e_q;  side 0: L = c - e_q (lower face), side 1: L = c (upper face)
                            const int L[3] = {i - (q == 0 && !side), j - (q == 1 && !side), k - (q == 2 && !side)};
                            const int R[3] = {L[0] + (q == 0), L[1] + (q == 1), L[2] + (q == 2)};
                            Set::Scalar g[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};   // g[r][s] = d u_r / d x_s at the face
                            for (int r = 0; r < AMREX_SPACEDIM; ++r)
                            {
                                g[r][q] = (U(R[0], R[1], R[2], r) - U(L[0], L[1], L[2], r)) / DX[q];
                                for (int t = 0; t < AMREX_SPACEDIM; ++t)
                                {
                                    if (t == q) continue;
                                    const int e0 = (t == 0), e1 = (t == 1), e2 = (t == 2);
                                    g[r][t] = 0.25 * (U(L[0] + e0, L[1] + e1, L[2] + e2, r) - U(L[0] - e0, L[1] - e1, L[2] - e2, r)
                                                    + U(R[0] + e0, R[1] + e1, R[2] + e2, r) - U(R[0] - e0, R[1] - e1, R[2] - e2, r)) / DX[t];
                                }
                            }
                            Set::Scalar divu = 0.0;
                            for (int t = 0; t < AMREX_SPACEDIM; ++t) divu += g[t][t];
                            const Set::Scalar muf = HarmonicMean(W(L[0], L[1], L[2], 0), W(R[0], R[1], R[2], 0));
                            const Set::Scalar laf = HarmonicMean(W(L[0], L[1], L[2], 1), W(R[0], R[1], R[2], 1));
                            const Set::Scalar sgn = side ? 1.0 : -1.0;
                            Set::Scalar w = 0.0;
                            for (int p = 0; p < AMREX_SPACEDIM; ++p)
                            {
                                const Set::Scalar tau = muf * (g[p][q] + g[q][p]) + (p == q ? (laf - (2.0 / 3.0) * muf) * divu : 0.0);
                                divtau[p] += sgn * tau / DX[q];
                                w += tau * 0.5 * (U(L[0], L[1], L[2], p) + U(R[0], R[1], R[2], p));
                            }
                            divw += sgn * w / DX[q];
                        }
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) F(i, j, k, d) = divtau[d];
                    F(i, j, k, AMREX_SPACEDIM) = divw;   // div(tau.u) = u.div(tau) + Phi
                });
            }
        }
        else
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto rho = density_mf[lev]->const_array(mfi);
            auto M   = momentum_mf[lev]->const_array(mfi);
            auto F   = Fv.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                auto sten = Numeric::GetStencil(i, j, k, domain);
                if (symc)
                {
                    const amrex::Dim3 dlo = amrex::lbound(domain), dhi = amrex::ubound(domain);
                    const int ijk[3] = {i, j, k}, lo3[3] = {dlo.x, dlo.y, dlo.z}, hi3[3] = {dhi.x, dhi.y, dhi.z};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        if ((symlo_c[d] && ijk[d] == lo3[d]) || (symhi_c[d] && ijk[d] == hi3[d]))
                            sten[d] = Numeric::StencilType::Central;
                }
                const Set::Scalar rc = std::max(rho(i, j, k), sm);
                Set::Vector u;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) u(d) = M(i, j, k, d) / rc;
                Set::Vector grad_eta = Numeric::Gradient(eta, i, j, k, 0, DX);
                Set::Matrix gradM = Numeric::Gradient(M, i, j, k, DX);
                Set::Vector gradrho = Numeric::Gradient(rho, i, j, k, 0, DX);
                Set::Matrix hess_rho = Numeric::Hessian(rho, i, j, k, 0, DX, sten);
                Set::Matrix gradu = (gradM - u * gradrho.transpose()) / rc;
                Set::Matrix3 hess_M = Numeric::Hessian(M, i, j, k, DX);
                Set::Matrix3 hess_u = Set::Matrix3::Zero();
                for (int p = 0; p < AMREX_SPACEDIM; p++)
                    for (int q = 0; q < AMREX_SPACEDIM; q++)
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                            hess_u(r, p, q) = (hess_M(r, p, q) - gradu(r, q) * gradrho(p) - gradu(r, p) * gradrho(q)
                                               - u(r) * hess_rho(p, q)) / rc;
                Set::Scalar dmu;
                const Set::Scalar mu_eff = mu_of(eta(i, j, k), dmu);
                const Set::Scalar lambda_eff = eta(i, j, k) * l0 + (1.0 - eta(i, j, k)) * l1;
                const Set::Vector grad_mu = dmu * grad_eta;
                const Set::Vector grad_lambda = (l0 - l1) * grad_eta;
                Set::Vector div_tau = Set::Vector::Zero();
                for (int p = 0; p < AMREX_SPACEDIM; p++)
                    for (int q = 0; q < AMREX_SPACEDIM; q++)
                        for (int r = 0; r < AMREX_SPACEDIM; r++)
                            for (int s = 0; s < AMREX_SPACEDIM; s++)
                            {
                                Set::Scalar Mpqrs = 0.0, dMpqrs = 0.0;
                                if ((p == r) && (q == s)) { Mpqrs += mu_eff; dMpqrs += grad_mu(q); }
                                if ((p == s) && (q == r)) { Mpqrs += mu_eff; dMpqrs += grad_mu(q); }
                                if ((p == q) && (r == s))
                                {
                                    Mpqrs += lambda_eff - (2.0 / 3.0) * mu_eff;
                                    dMpqrs += grad_lambda(q) - (2.0 / 3.0) * grad_mu(q);
                                }
                                div_tau(p) += Mpqrs * hess_u(r, s, q) + dMpqrs * gradu(r, s);
                            }
                const Set::Matrix Ssym = 0.5 * (gradu + gradu.transpose());
                const Set::Scalar divu = gradu.trace();
                const Set::Scalar diss = 2.0 * mu_eff * Ssym.squaredNorm()
                                       + (lambda_eff - (2.0 / 3.0) * mu_eff) * divu * divu;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) F(i, j, k, d) = div_tau(d);
                F(i, j, k, AMREX_SPACEDIM) = u.dot(div_tau) + diss;
            });
        }
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto M = momentum_mf[lev]->array(mfi);
            auto E = energy_per_vol_mf[lev]->array(mfi);
            auto v = velocity_mf[lev]->array(mfi);
            auto rho = density_mf[lev]->const_array(mfi);
            auto F = Fv.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
            {
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    M(i, j, k, d) += dtau * F(i, j, k, d);
                    v(i, j, k, d) = M(i, j, k, d) / std::max(rho(i, j, k), sm);
                }
                E(i, j, k) += dtau * F(i, j, k, AMREX_SPACEDIM);
            });
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// SubcycleConduction (thermal.subcycle = 1, thermal.conduction = 2; 2026-11-26).  The per-phase Fourier conduction of
// the RHS (same face fluxes k_k min(alpha_k) dT_k/dx, zero where either side is below thermal.alpha_min) advanced after
// the hyperbolic step in N forward-Euler sub-steps on E_0, E_1 and rho E; eta and rho_k are fixed, so T_k changes only
// through E_k.  Per cell the update is T_c += dtau sum_f c_f (T_f - T_c), c_f = k a_f / (a_c rho_c cv dx^2); with
// dtau sum_f c_f <= 1 it is a convex combination (max principle, no new extrema), hence stable for any diffusivity.
// N = ceil(dt / dtau), dtau = thermal.subcycle_cfl / max_cell sum_f c_f, reduced over the level.
// ---------------------------------------------------------------------------------------------------------------------
void Hydro2::SubcycleConduction(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::SubcycleConduction");
    if (!thermal_subcycle || thermal_conduction != 2) return;
    if (!(dt > 0.0)) return;
    if (thermal_k0 == 0.0 && thermal_k1 == 0.0) return;

    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar tn = time + dt;
    const Set::Scalar amin = thermal_alpha_min, sm = small;
    const Set::Scalar kph[2] = {thermal_k0, thermal_k1};
    const Set::Scalar cvp[2] = {eos0.Cv(), eos1.Cv()};
    const Set::Scalar pip[2] = {eos0.P0(), eos1.P0()};
    const Set::Scalar dx2[3] = {AMREX_D_DECL(DX[0] * DX[0], DX[1] * DX[1], DX[2] * DX[2])};

    // ghosts: eta / rho_k once (fixed during the operator), E_k every sub-step
    auto fill = [&](bool all) {
        if (lev > 0)
        {
            if (all)
            {
                FillPatch(lev, tn, rho_eta0_mf, *rho_eta0_mf[lev], *density_bc, 0);
                FillPatch(lev, tn, rho_eta1_mf, *rho_eta1_mf[lev], *density_bc, 0);
                FillPatch(lev, tn, eta_mf,      *eta_mf[lev],      *eta_bc,     0);
            }
            FillPatch(lev, tn, energy0_mf, *energy0_mf[lev], *energy_bc, 0);
            FillPatch(lev, tn, energy1_mf, *energy1_mf[lev], *energy_bc, 0);
        }
        else
        {
            if (all)
            {
                FillBoundariesWithBC(lev, tn, density_bc, { rho_eta0_mf[lev].get(), rho_eta1_mf[lev].get() });
                FillBoundariesWithBC(lev, tn, eta_bc, { eta_mf[lev].get() });
            }
            FillBoundariesWithBC(lev, tn, energy_bc, { energy0_mf[lev].get(), energy1_mf[lev].get() });
        }
        if (radial_bc && lev > 0)   // (level 0: FillBoundariesWithBC applies the RADIAL face, as for the viscous sub-cycles)
        {
            if (all)
            {
                RadialGhostFill(lev, *rho_eta0_mf[lev], false);
                RadialGhostFill(lev, *rho_eta1_mf[lev], false);
                RadialGhostFill(lev, *eta_mf[lev], false);
            }
            RadialGhostFill(lev, *energy0_mf[lev], false);
            RadialGhostFill(lev, *energy1_mf[lev], false);
        }
    };
    fill(true);

    // ---- N from the exact max-principle bound of the operator ----------------
    Set::Scalar cmax = 0.0;
    {
        amrex::ReduceOps<amrex::ReduceOpMax> rop;
        amrex::ReduceData<Set::Scalar> rdat(rop);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto r0 = rho_eta0_mf[lev]->const_array(mfi);
            auto r1 = rho_eta1_mf[lev]->const_array(mfi);
            rop.eval(bx, rdat, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Set::Scalar> {
                Set::Scalar best = 0.0;
                const Set::Scalar ac0 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                for (int ph = 0; ph < 2; ++ph)
                {
                    const Set::Scalar ac = ph ? 1.0 - ac0 : ac0;
                    if (ac <= amin) continue;
                    const Set::Scalar C = std::max(ph ? r1(i, j, k) : r0(i, j, k), sm) * cvp[ph];   // a rho cv
                    Set::Scalar sum = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        for (int sg = -1; sg <= 1; sg += 2)
                        {
                            const Set::Scalar en = std::min(std::max(eta(i + (d == 0) * sg, j + (d == 1) * sg, k + (d == 2) * sg), 0.0), 1.0);
                            const Set::Scalar af = std::min(ac, ph ? 1.0 - en : en);
                            if (af > amin) sum += kph[ph] * af / (C * dx2[d]);
                        }
                    best = std::max(best, sum);
                }
                return {(best == best) ? best : 0.0};
            });
        }
        cmax = amrex::get<0>(rdat.value());
        amrex::ParallelDescriptor::ReduceRealMax(cmax);
    }
    if (!(cmax > 0.0)) return;
    const Set::Scalar dtau_max = thermal_subcycle_cfl / cmax;
    long long Nll = (long long)std::ceil((long double)dt / (long double)dtau_max);
    if (Nll < 1LL) Nll = 1LL;
    if (Nll > (long long)thermal_subcycle_max)
    {
        Util::Warning(INFO, "SubcycleConduction lev=", lev, ": required N=", Nll, " > thermal.subcycle_max=",
                      thermal_subcycle_max, "; clamping (UNSTABLE -- raise the cap or lower dt)");
        Nll = thermal_subcycle_max;
    }
    const int N = (int)Nll;
    const Set::Scalar dtau = dt / (Set::Scalar)N;
    if (thermal_subcycle_verbose)
        Util::Message(INFO, "SubcycleConduction lev=", lev, " N=", N, " dtau=", dtau, " (dt=", dt, ", dtau_max=", dtau_max, ")");

    amrex::MultiFab Q(energy0_mf[lev]->boxArray(), energy0_mf[lev]->DistributionMap(), 2, 0);
    for (int it = 0; it < N; ++it)
    {
        if (it > 0) fill(false);
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto r0 = rho_eta0_mf[lev]->const_array(mfi);
            auto r1 = rho_eta1_mf[lev]->const_array(mfi);
            auto E0 = energy0_mf[lev]->const_array(mfi);
            auto E1 = energy1_mf[lev]->const_array(mfi);
            auto q = Q.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                // phase temperature, exactly as the RHS form: T_k = (E_k - a_k pi_k) / (max(a rho, small) cv_k)
                auto Tph = [&](int ii, int jj, int kk, int ph) -> Set::Scalar {
                    const Set::Scalar a0 = std::min(std::max(eta(ii, jj, kk), 0.0), 1.0);
                    return ph == 0 ? (E0(ii, jj, kk) - a0 * pip[0]) / (std::max(r0(ii, jj, kk), sm) * cvp[0])
                                   : (E1(ii, jj, kk) - (1.0 - a0) * pip[1]) / (std::max(r1(ii, jj, kk), sm) * cvp[1]);
                };
                const Set::Scalar ac0 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                for (int ph = 0; ph < 2; ++ph)
                {
                    q(i, j, k, ph) = 0.0;
                    const Set::Scalar ac = ph ? 1.0 - ac0 : ac0;
                    if (ac <= amin) continue;
                    const Set::Scalar Tc = Tph(i, j, k, ph);
                    Set::Scalar c = 0.0;
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                        const Set::Scalar eh = std::min(std::max(eta(i + di, j + dj, k + dk), 0.0), 1.0);
                        const Set::Scalar el = std::min(std::max(eta(i - di, j - dj, k - dk), 0.0), 1.0);
                        const Set::Scalar ahi = std::min(ac, ph ? 1.0 - eh : eh);
                        const Set::Scalar alo = std::min(ac, ph ? 1.0 - el : el);
                        const Set::Scalar Fhi = (ahi > amin) ? kph[ph] * ahi * (Tph(i + di, j + dj, k + dk, ph) - Tc) / DX[d] : 0.0;
                        const Set::Scalar Flo = (alo > amin) ? kph[ph] * alo * (Tc - Tph(i - di, j - dj, k - dk, ph)) / DX[d] : 0.0;
                        c += (Fhi - Flo) / DX[d];
                    }
                    q(i, j, k, ph) = c;
                }
            });
        }
        for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto E0 = energy0_mf[lev]->array(mfi);
            auto E1 = energy1_mf[lev]->array(mfi);
            auto E = energy_per_vol_mf[lev]->array(mfi);
            auto q = Q.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                E0(i, j, k) += dtau * q(i, j, k, 0);
                E1(i, j, k) += dtau * q(i, j, k, 1);
                E(i, j, k)  += dtau * (q(i, j, k, 0) + q(i, j, k, 1));
            });
        }
    }
}

// Gas-liquid heat exchange in mixed cells (thermal.exchange, see Parse).
// With C_k = alpha_k rho_k cv_k (dE_k = C_k dT_k at fixed alpha_k, rho_k) the
// linear exchange dE_0/dt = H (T_1 - T_0) = -dE_1/dt has the exact solution
//     q = C0 C1/(C0 + C1) (T_1 - T_0) (1 - exp(-H (1/C0 + 1/C1) dt)),
// E_0 += q, E_1 -= q; rho E is unchanged.  Unconditionally stable, and never
// overshoots the common temperature.
void Hydro2::ThermalExchange(int lev, Set::Scalar dt)
{
    const Set::Scalar amin = thermal_alpha_min, k0 = thermal_k0, k1 = thermal_k1, hc = thermal_h;
    const int mode = thermal_exchange;
    if (mode == 4) return;   // joint p-T relaxation inside RelaxAndReinit
    const int tdiag = thermal_diag;
    const Set::Scalar g0d = eos0.Gamma(), g1d = eos1.Gamma(), tnow = t_new[lev];
    const Set::Scalar cv0 = eos0.Cv(), cv1 = eos1.Cv(), pi0 = eos0.P0(), pi1 = eos1.P0();
    const Set::Scalar small_l = small;
    const Set::Scalar *DX = geom[lev].CellSize();
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto ar0 = rho_eta0_mf[lev]->const_array(mfi);
        auto ar1 = rho_eta1_mf[lev]->const_array(mfi);
        auto E0  = energy0_mf[lev]->array(mfi);
        auto E1  = energy1_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar a0 = std::min(std::max(et(i, j, k), 0.0), 1.0), a1 = 1.0 - a0;
            if (a0 <= amin || a1 <= amin) return;
            const Set::Scalar r0 = std::max(ar0(i, j, k), small_l), r1 = std::max(ar1(i, j, k), small_l);
            const Set::Scalar T0 = (E0(i, j, k) - a0 * pi0) / (r0 * cv0);
            const Set::Scalar T1 = (E1(i, j, k) - a1 * pi1) / (r1 * cv1);
            Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
            const Set::Scalar g = ge.norm();
            if (!(g > 0.0)) return;
            const Set::Scalar C0 = r0 * cv0, C1 = r1 * cv1;
            if (mode == 3)   // instantaneous equilibrium
            {
                const Set::Scalar q = C0 * C1 / (C0 + C1) * (T1 - T0);
                E0(i, j, k) += q;
                E1(i, j, k) -= q;
                return;
            }
            const Set::Scalar H = (mode == 1) ? g * g / (a0 * a1 * (a0 / k0 + a1 / k1)) : hc * g;
            const Set::Scalar q = C0 * C1 / (C0 + C1) * (T1 - T0) * (1.0 - std::exp(-H * (1.0 / C0 + 1.0 / C1) * dt));
            if (tdiag && std::abs(q) > 0.1 * std::abs(E1(i, j, k)))
            {
                const Set::Scalar p0b = ((g0d - 1.0) * E0(i, j, k) - g0d * pi0 * a0) / a0, p1b = (g1d - 1.0) * E1(i, j, k) / a1;
                const Set::Scalar p0a = ((g0d - 1.0) * (E0(i, j, k) + q) - g0d * pi0 * a0) / a0, p1a = (g1d - 1.0) * (E1(i, j, k) - q) / a1;
                printf("TXDIAG t=%.7e i=%d j=%d k=%d eta=%.6f T0=%.1f T1=%.1f dE1/E1=%.3f p_liq %.4e->%.4e p_gas %.4e->%.4e\n",
                       tnow, i, j, k, a0, T0, T1, -q / E1(i, j, k), p0b, p0a, p1b, p1a);
            }
            E0(i, j, k) += q;
            E1(i, j, k) -= q;
        });
    }
}

// TensorCurvature(): mean curvature of the local eta level set from the unit
// capillary tensor (jump.curvature = 1).
//   Omega_u = |grad eta| I - grad eta grad eta / |grad eta|,
//   n . div(Omega_u) = -|grad eta| div(n)   =>   kappa = -n . div(Omega_u) / |grad eta|.
// kappa = div(n) (n = grad eta/|grad eta|, gas -> liquid), i.e. 2/r on a sphere.
// Omega_u at the 2*DIM neighbours from central gradients (eta needs 2 ghosts),
// central divergence at the cell -- the same stencil as the capillary force.
// Valid cells only; 0 where |grad eta| < 1e-10.
// jump.curvature_iso: d/dx_d of component c of F, central along d, averaged over the transverse
// neighbours with weights [1,2,1]/4 per transverse direction (isotropic 3^DIM stencil, reach 1).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
static Set::Scalar IsoD(amrex::Array4<const Set::Scalar> const &F, int i, int j, int k, int c, int d, const Set::Scalar *DX)
{
    const Set::Scalar w1[3] = {0.25, 0.5, 0.25};
    const int t0 = (d + 1) % AMREX_SPACEDIM;
#if AMREX_SPACEDIM == 3
    const int t1 = (d + 2) % 3;
    const int bmax = 1;
#else
    const int t1 = 0;
    const int bmax = 0;
#endif
    Set::Scalar s = 0.0;
    for (int a = -1; a <= 1; ++a)
        for (int b = -bmax; b <= bmax; ++b)
        {
            int o[3] = {0, 0, 0};
            o[t0] += a;
            if (bmax) o[t1] += b;
            const Set::Scalar w = w1[a + 1] * (bmax ? w1[b + 1] : 1.0);
            s += w * (F(i + o[0] + (d == 0), j + o[1] + (d == 1), k + o[2] + (d == 2), c)
                    - F(i + o[0] - (d == 0), j + o[1] - (d == 1), k + o[2] - (d == 2), c));
        }
    return s / (2.0 * DX[d]);
}

// psi reference densities: mean density of each pure phase on level 0 at the first call (the IC).
void Hydro2::PsiRefDensities()
{
    if (jump_y_rho0 > 0.0 && jump_y_rho1 > 0.0) return;
    {
        // reference densities: mean density of each pure phase on level 0 (eta > 0.99 / < 0.01)
        Set::Scalar s0 = 0.0, n0 = 0.0, s1 = 0.0, n1 = 0.0;
        for (amrex::MFIter mfi(*eta_mf[0], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &vb = mfi.validbox();
            auto e0 = eta_mf[0]->const_array(mfi);
            auto q0 = rho_eta0_mf[0]->const_array(mfi);
            auto q1 = rho_eta1_mf[0]->const_array(mfi);
            amrex::LoopOnCpu(vb, [&](int i, int j, int k) {
                const Set::Scalar e = e0(i, j, k);
                if (e > 0.99) { s0 += q0(i, j, k) / e; n0 += 1.0; }
                if (e < 0.01) { s1 += q1(i, j, k) / (1.0 - e); n1 += 1.0; }
            });
        }
        amrex::ParallelDescriptor::ReduceRealSum(s0); amrex::ParallelDescriptor::ReduceRealSum(n0);
        amrex::ParallelDescriptor::ReduceRealSum(s1); amrex::ParallelDescriptor::ReduceRealSum(n1);
        jump_y_rho0 = (n0 > 0.0) ? s0 / n0 : 1.0;
        jump_y_rho1 = (n1 > 0.0) ? s1 / n1 : 1.0;
        Util::Message(INFO, "tensor curvature: psi reference densities rho0 = ", jump_y_rho0, "  rho1 = ", jump_y_rho1);
    }
}

// ----------------------------------------------------------------------
// shell.slip = 7 (MATERIAL shell): Gp = |grad psi| on the valid box + 1 ghost, where
// psi = (a rho)_0/rho0_ref / ((a rho)_0/rho0_ref + (a rho)_1/rho1_ref) (see Parse CURVATURE);
// -1 outside the psi band (cut < psi < 1 - cut) or where |grad psi| < g_gem_min.
// Gamma = m / Gp.  The phase-mass ghosts are refreshed exactly as in TensorCurvature.
// ----------------------------------------------------------------------
void Hydro2::ShellPsiGrad(int lev, Set::Scalar time, amrex::MultiFab &Gp)
{
    BL_PROFILE("Integrator::Hydro2::ShellPsiGrad");
    PsiRefDensities();
    const Set::Scalar *DXp = geom[lev].CellSize();
    if (lev == 0)
    {
        FillBoundariesWithBC(lev, time, density0_bc, { rho_eta0_mf[lev].get() });
        FillBoundariesWithBC(lev, time, density1_bc, { rho_eta1_mf[lev].get() });
    }
    else
    {
        FillPatch(lev, time, rho_eta0_mf, *rho_eta0_mf[lev], *density0_bc, 0);
        FillPatch(lev, time, rho_eta1_mf, *rho_eta1_mf[lev], *density1_bc, 0);
        if (radial_bc)
        {
            RadialGhostFill(lev, *rho_eta0_mf[lev], false);
            RadialGhostFill(lev, *rho_eta1_mf[lev], false);
        }
    }
    const Set::Scalar ir0 = 1.0 / jump_y_rho0, ir1 = 1.0 / jump_y_rho1;
    const bool use_eta = (shell_slip == 11 || shell_slip == 12 || shell_slip == 15);   // option 11: m = Gamma |grad eta|
    amrex::MultiFab Y(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, eta_mf[lev]->nGrow());
    for (amrex::MFIter mfi(Y, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box gb = mfi.fabbox();
        auto r0 = rho_eta0_mf[lev]->const_array(mfi);
        auto r1 = rho_eta1_mf[lev]->const_array(mfi);
        auto y  = Y.array(mfi);
        auto et = eta_mf[lev]->const_array(mfi);
        amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar a = std::max(r0(i, j, k), 0.0) * ir0, b = std::max(r1(i, j, k), 0.0) * ir1;
            y(i, j, k) = use_eta ? et(i, j, k) : ((a + b > 0.0) ? a / (a + b) : 0.0);   // option 11: eta
        });
    }
    Gp.define(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, 1);
    const Set::Scalar cut = std::max(cutoff, 1.0e-12);
    for (amrex::MFIter mfi(Gp, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box gb = mfi.growntilebox(1);
        auto y = Y.const_array(mfi);
        auto g = Gp.array(mfi);
        amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar yc = y(i, j, k);
            Set::Scalar gg = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar gd = (y(i + di, j + dj, k + dk) - y(i - di, j - dj, k - dk)) / (2.0 * DXp[d]);
                gg += gd * gd;
            }
            gg = std::sqrt(gg);
            g(i, j, k) = (yc > cut && yc < 1.0 - cut && gg > g_gem_min) ? gg : -1.0;
        });
    }
}

// shell.foot_rate: interpolate the nc-component field L (set on band-core valid cells, NaN elsewhere, nGrow >= 1)
// to the foot point x - d n, d = eps atanh(2 eta - 1), of every band cell (cutoff < eta < 1 - cutoff).
// Ghosts: same-level exchange; physical faces mirror where eta's BC is REFLECT_*, else zero-gradient.
// Gf (nGrow 0) is NaN where any stencil point with nonzero weight is NaN or off the fab (caller keeps the local value).
void Hydro2::FootExtend(int lev, amrex::MultiFab &L, amrex::MultiFab &Gf)
{
    BL_PROFILE("Integrator::Hydro2::FootExtend");
    const Set::Scalar *DXl = geom[lev].CellSize();
    const Set::Scalar nan = std::numeric_limits<Set::Scalar>::quiet_NaN();
    const Set::Scalar cut_l = std::max(cutoff, 1.0e-12), eps_l = epsilon;
    const int nc = L.nComp();
    L.FillBoundary(geom[lev].periodicity());
    const amrex::Box dom = geom[lev].Domain();
    for (amrex::MFIter mfi(L, false); mfi.isValid(); ++mfi)
    {
        auto l = L.array(mfi);
        amrex::LoopOnCpu(mfi.fabbox(), [&](int i, int j, int k) {
            const int ijk[3] = {i, j, k};
            int s[3] = {i, j, k};
            bool out = false;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                if (ijk[d] < dom.smallEnd(d))
                { s[d] = foot_mirror[d][0] ? 2 * dom.smallEnd(d) - 1 - ijk[d] : dom.smallEnd(d); out = true; }
                else if (ijk[d] > dom.bigEnd(d))
                { s[d] = foot_mirror[d][1] ? 2 * dom.bigEnd(d) + 1 - ijk[d] : dom.bigEnd(d); out = true; }
            }
            if (!out || !l.contains(s[0], s[1], s[2])) return;
            for (int c = 0; c < nc; ++c) l(i, j, k, c) = l(s[0], s[1], s[2], c);
        });
    }
    Gf.define(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), nc, 0);
    Gf.setVal(nan);
    constexpr int NB = (AMREX_SPACEDIM == 3) ? 2 : 1;   // stencil extent in z
    for (amrex::MFIter mfi(Gf, false); mfi.isValid(); ++mfi)
    {
        auto et = eta_mf[lev]->const_array(mfi); auto l = L.const_array(mfi); auto gf = Gf.array(mfi);
        amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
            const Set::Scalar ec = et(i, j, k);
            if (!(ec > cut_l && ec < 1.0 - cut_l)) return;
            const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXl);
            const Set::Scalar g = ge.lpNorm<2>();
            if (g < g_gem_min) return;
            const Set::Scalar dd = eps_l * std::atanh(2.0 * ec - 1.0);
            const int ijk[3] = {i, j, k};
            int i0[3] = {i, j, k};
            Set::Scalar w[3] = {0.0, 0.0, 0.0};
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const Set::Scalar f = ijk[d] - dd * ge(d) / g / DXl[d];
                i0[d] = (int)std::floor(f); w[d] = f - i0[d];
            }
            Set::Scalar acc[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < NB; ++c)
            {
                const Set::Scalar wt = (a ? w[0] : 1 - w[0]) * (b ? w[1] : 1 - w[1]) * (NB == 1 ? 1.0 : (c ? w[2] : 1 - w[2]));
                if (wt == 0.0) continue;
                const int ii = i0[0] + a, jj = i0[1] + b, kk = i0[2] + c;
                if (!l.contains(ii, jj, kk)) return;
                for (int q = 0; q < nc; ++q)
                {
                    const Set::Scalar v = l(ii, jj, kk, q);
                    if (!(v == v)) return;
                    acc[q] += wt * v;
                }
            }
            for (int q = 0; q < nc; ++q) gf(i, j, k, q) = acc[q];
        });
    }
}

void Hydro2::TensorCurvature(int lev, amrex::MultiFab &K, Set::Scalar time)
{
    BL_PROFILE("Integrator::Hydro2::TensorCurvature");
    const Set::Scalar *DXc = geom[lev].CellSize();
    // Curvature of the level sets of the LIQUID MASS, Y = (alpha rho)_0 / rho0_ref (2026-11-13).
    //  * The pressure relaxation never changes (alpha rho)_k, so Y has no relaxation -> alpha ->
    //    kappa -> jump -> relaxation loop.  kappa(eta) has that loop: the static Laplace bubble
    //    (soft phases, p 10 Pa, jump 3 Pa) drifted 17% with |u| ~ 1 m/s; at p 10 kPa it held.
    //  * The liquid is the stiff phase, so its level sets are the eta level sets to O(dp/(rho c^2)).
    //  * Trace gas plays no part: psi = Y_l/(Y_l + Y_g) (11-02 .. 11-13) went non-monotone where
    //    frozen trace gas had different densities in neighbouring cells -> normal flip -> one-cell
    //    kappa ~ -2/(2 dx) -> wrong-sign jump (06 160 kPa spherical-BC crash, 4.87 us).
    amrex::MultiFab Yc;
    PsiRefDensities();
    {
        // Current ghosts for the liquid mass.  The callers refresh eta's ghosts but not always
        // rho_eta0's (RelaxAndReinit runs inside FillGhost4BC before it is filled; the viscous
        // sub-cycle refreshes eta and rho only): a stale RADIAL / mirror / coarse-fine ghost
        // gave erratic kappa on the 2-cell-wide RADIAL slab.  Same fill as eta's at each caller.
        if (lev == 0)
            FillBoundariesWithBC(lev, time, density0_bc, { rho_eta0_mf[lev].get() });
        else
        {
            FillPatch(lev, time, rho_eta0_mf, *rho_eta0_mf[lev], *density0_bc, 0);
            if (radial_bc) RadialGhostFill(lev, *rho_eta0_mf[lev], false);
        }
        const Set::Scalar ir0 = 1.0 / jump_y_rho0;
        Yc.define(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, eta_mf[lev]->nGrow());
        for (amrex::MFIter mfi(Yc, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box gb = mfi.fabbox();
            auto r0 = rho_eta0_mf[lev]->const_array(mfi);
            auto Y  = Yc.array(mfi);
            amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Y(i, j, k) = std::max(r0(i, j, k), 0.0) * ir0;
            });
        }
    }
    // Unit normals n = grad Y/|grad Y| from the isotropic gradient on valid + 1 ghost (reach 2 = nghost)
    amrex::MultiFab Nf(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), AMREX_SPACEDIM, 1);
    for (amrex::MFIter mfi(Nf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box gb = mfi.growntilebox(1);
        auto et = Yc.const_array(mfi);
        auto nf = Nf.array(mfi);
        amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector gn;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) gn(d) = IsoD(et, i, j, k, 0, d, DXc);
            const Set::Scalar gm = gn.lpNorm<2>();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) nf(i, j, k, d) = gm < g_gem_min ? 0.0 : gn(d) / gm;
        });
    }
    // Wall contact angle (eta.contact_angle.<face>, see Parse): the ghost normal across a wall face is
    // the Brackbill wall normal n_w = cos(theta) e + sin(theta) t built from the mirror interior cell's
    // normal n_m (t = tangential part of n_m, normalised).  No interface at the wall (n_m = 0) or an
    // interface parallel to the wall (no tangential part: no contact line in that cell) keeps the mirror.
    {
        const amrex::Box dom = geom[lev].Domain();
        for (int d = 0; d < AMREX_SPACEDIM; ++d)
            for (int sd = 0; sd < 2; ++sd)
            {
                const Set::Scalar th = contact_angle[d][sd];
                if (th < 0.0) continue;
                const Set::Scalar ct = std::cos(th), st = std::sin(th), sgn = sd ? 1.0 : -1.0;
                const int face = sd ? dom.bigEnd(d) : dom.smallEnd(d);
                for (amrex::MFIter mfi(Nf, false); mfi.isValid(); ++mfi)
                {
                    const amrex::Box vb = mfi.validbox();
                    if ((sd ? vb.bigEnd(d) : vb.smallEnd(d)) != face) continue;   // box not on this wall
                    amrex::Box gb = amrex::grow(vb, 1);
                    if (sd) gb.setSmall(d, face + 1); else gb.setBig(d, face - 1);   // ghost layer beyond the wall
                    auto nf = Nf.array(mfi);
                    auto eta = eta_mf[lev]->const_array(mfi);
                    const int prof = contact_angle_profile;
                    const Set::Scalar eps = epsilon;
                    amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        int m[3] = {i, j, k};
                        m[d] = 2 * face + (sd ? 1 : -1) - m[d];   // mirror interior cell
                        Set::Scalar tg[3] = {0, 0, 0}, tm = 0.0;
                        for (int e = 0; e < AMREX_SPACEDIM; ++e)
                            if (e != d) { tg[e] = nf(m[0], m[1], m[2], e); tm += tg[e] * tg[e]; }
                        tm = std::sqrt(tm);
                        if (tm <= 0.0) return;
                        Set::Scalar nw[3] = {0, 0, 0};
                        for (int e = 0; e < AMREX_SPACEDIM; ++e)
                            nw[e] = (e == d) ? sgn * ct : st * tg[e] / tm;
                        Set::Scalar w = 1.0;
                        if (prof)
                        {
                            // interface fraction of the mirror cell: |grad eta| relative to the tanh profile's 2 eta(1-eta)/eps
                            Set::Vector ge = Numeric::Gradient(eta, m[0], m[1], m[2], 0, DXc);
                            // wall-normal part: one-sided 2nd-order into the domain (the central difference
                            // would use the even-reflected ghost and return half the slope)
                            {
                                int m1[3] = {m[0], m[1], m[2]}, m2[3] = {m[0], m[1], m[2]};
                                const int in = sd ? -1 : 1;
                                m1[d] += in; m2[d] += 2 * in;
                                ge(d) = in * (-3.0 * eta(m[0], m[1], m[2]) + 4.0 * eta(m1[0], m1[1], m1[2])
                                              - eta(m2[0], m2[1], m2[2])) / (2.0 * DXc[d]);
                            }
                            const Set::Scalar em = eta(m[0], m[1], m[2]);
                            const Set::Scalar den = 2.0 * em * (1.0 - em);
                            w = den > 0.0 ? std::min(1.0, eps * ge.lpNorm<2>() / den) : 0.0;
                        }
                        Set::Scalar nb[3] = {0, 0, 0}, nn = 0.0;
                        for (int e = 0; e < AMREX_SPACEDIM; ++e)
                        {
                            nb[e] = w * nw[e] + (1.0 - w) * nf(i, j, k, e);   // nf(i,j,k) = mirror ghost normal
                            nn += nb[e] * nb[e];
                        }
                        nn = std::sqrt(nn);
                        for (int e = 0; e < AMREX_SPACEDIM; ++e)
                            nf(i, j, k, e) = nn > 0.0 ? nb[e] / nn : 0.0;
                    });
                }
            }
    }
    // kappa = div(n), isotropic stencil; zero where the compact |grad Y| vanishes
    for (amrex::MFIter mfi(K, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et = Yc.const_array(mfi);
        auto Kc = K.array(mfi);
        auto nf = Nf.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Kc(i, j, k) = 0.0;
            const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXc);
            if (ge.lpNorm<2>() < g_gem_min) return;
            Set::Scalar kap = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) kap += IsoD(nf, i, j, k, d, d, DXc);
            Kc(i, j, k) = kap;
        });
    }
}

// Shell Gamma bulk extension (called from RelaxAndReinit).
// ----------------------------------------------------------------------
// LEVEL-SET SHELL SLIP (option 4, 2026-11-10).
// A shell on an eta level set moves with the LEVEL SET (normal speed V_n), while the
// Gamma row moves it with the FLUID (normal speed u_n).  In the diffuse band the two
// differ (Kapila band; pressure relaxation and heat exchange move eta without fluid
// motion): measured |dR05/dt| > |u(eta=.5)|, +-1.4 cells of slip per deep collapse, which
// accounts for the per-collapse Gamma ratchet (attr/nonmat.py).  Exact correction for
// s = V_n - u_n over the step, Eulerian for a field defined on the level sets:
//     area :  d ln Gamma = -kappa s dt        (kappa = div n, level-set area law)
//     shift:  d Gamma    = -s dt n.grad Gamma (upwind)
// V_n = -(eta^{n+1} - eta^n) / (dt |grad eta|), n = grad eta/|grad eta| (into the liquid).
// Band cells only (cut < eta < 1 - cut, |grad eta| > g_gem_min, stencil inside the box).
// ----------------------------------------------------------------------
void Hydro2::ShellLevelSetSlip(int lev, Set::Scalar time, Set::Scalar dt)
{
    // OPTION 6 (2026-11-10): CONSERVATIVE level-set slip.
    // The shell carried by the eta level sets moves with V = u + s n.  The fluid
    // part is in the Gamma row; here the slip part, written for the shell amount
    // per volume m = Gamma |grad eta| (Gamma times the interface delta):
    //     m_t + div(Gamma w n) = 0,   w = s |grad eta| = -(eta_t + u.grad eta).
    // w needs no division by |grad eta|, and the area change of a moving level
    // set (the -kappa s Gamma of options 4/5) comes out of the flux divergence
    // exactly, so no curvature (second derivatives of eta) is ever formed.
    //   eta* = eta^{n+1} + dt w   is eta advected by u alone (= eta^n - dt u.grad eta),
    //   m*   = Gamma* |grad eta*|,  m^{n+1} = m* - dt div(F),  Gamma^{n+1} = m^{n+1}/|grad eta^{n+1}|.
    // Uniform Gamma check: Gamma^{n+1} = Gamma (1 - kappa s dt) + O(dt^2) -- the area law.
    // Face flux F_d = (w n_d)_face * Gamma_upwind: conservative, total shell is conserved.
    BL_PROFILE("Integrator::Hydro2::ShellLevelSetSlip");
    if (!(dt > 0.0)) return;
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar cut = std::max(cutoff, 1.0e-12);
    const int ng = shell_mf[lev]->nGrow();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ng >= 2 && eta_mf[lev]->nGrow() >= 2 && velocity_mf[lev]->nGrow() >= 1,
                                     "ShellLevelSetSlip needs >= 2 ghost cells");
    amrex::MultiFab G0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, ng);
    amrex::MultiFab::Copy(G0, *shell_mf[lev], 0, 0, 1, ng);
    // W = w, ES = eta*, NV = w n (3 comps), on the valid box grown by one
    amrex::MultiFab W(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1 + 1 + AMREX_SPACEDIM, 1);
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = amrex::grow(mfi.validbox(), 1);
        auto et  = eta_mf[lev]->const_array(mfi);
        auto eo  = eta_old_mf[lev]->const_array(mfi);
        auto vel = velocity_mf[lev]->const_array(mfi);
        auto ww  = W.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector gm = Set::Vector::Zero();      // grad of the step-average eta
            Set::Vector g1 = Set::Vector::Zero();      // grad eta^{n+1}
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                g1(d) = (et(i + di, j + dj, k + dk) - et(i - di, j - dj, k - dk)) / (2.0 * DX[d]);
                gm(d) = 0.5 * (g1(d) + (eo(i + di, j + dj, k + dk) - eo(i - di, j - dj, k - dk)) / (2.0 * DX[d]));
            }
            Set::Scalar ug = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) ug += vel(i, j, k, d) * gm(d);
            const Set::Scalar w = -(et(i, j, k) - eo(i, j, k)) / dt - ug;
            ww(i, j, k, 0) = w;
            ww(i, j, k, 1) = et(i, j, k) + dt * w;     // eta*
            const Set::Scalar g = g1.lpNorm<2>();
            for (int d = 0; d < AMREX_SPACEDIM; ++d) ww(i, j, k, 2 + d) = (g > g_gem_min) ? w * g1(d) / g : 0.0;
        });
    }
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et = eta_mf[lev]->const_array(mfi);
        auto ww = W.const_array(mfi);
        auto g0 = G0.const_array(mfi);
        auto sh = shell_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar ec = et(i, j, k);
            if (!(ec > cut && ec < 1.0 - cut)) return;
            Set::Scalar g1 = 0.0, gs = 0.0, div = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar a1 = (et(i + di, j + dj, k + dk) - et(i - di, j - dj, k - dk)) / (2.0 * DX[d]);
                const Set::Scalar as = (ww(i + di, j + dj, k + dk, 1) - ww(i - di, j - dj, k - dk, 1)) / (2.0 * DX[d]);
                g1 += a1 * a1; gs += as * as;
                // face fluxes (w n_d) at i+1/2 and i-1/2, Gamma upwinded
                const Set::Scalar vp = 0.5 * (ww(i, j, k, 2 + d) + ww(i + di, j + dj, k + dk, 2 + d));
                const Set::Scalar vm = 0.5 * (ww(i - di, j - dj, k - dk, 2 + d) + ww(i, j, k, 2 + d));
                const Set::Scalar Fp = vp * ((vp > 0.0) ? g0(i, j, k) : g0(i + di, j + dj, k + dk));
                const Set::Scalar Fm = vm * ((vm > 0.0) ? g0(i - di, j - dj, k - dk) : g0(i, j, k));
                div += (Fp - Fm) / DX[d];
            }
            g1 = std::sqrt(g1); gs = std::sqrt(gs);
            if (g1 < g_gem_min) return;
            const Set::Scalar Gn = (g0(i, j, k) * gs - dt * div) / g1;
            if (std::fabs(Gn - g0(i, j, k)) > 0.2 * std::fabs(g0(i, j, k)))
                printf("SLIP6 t=%.6e ijk=%d %d %d eta=%.4f g1dx=%.3e gsdx=%.3e G0=%.4f Gn=%.4f\n",
                       time, i, j, k, ec, g1 * DX[0], gs * DX[0], g0(i, j, k), Gn);
            sh(i, j, k) = Gn;
        });
    }
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
}

void Hydro2::ShellBandUniform(int lev)
{
    BL_PROFILE("Integrator::Hydro2::ShellBandUniform");
    // eta below cutoff (or above 1 - cutoff): bulk cell, takes the constant normal extension.
    const Set::Scalar bth = cutoff;
    // Ghosts with the PHYSICAL BCs (REFLECT / RADIAL / far field), not only
    // interior + periodic: the bulk extension below reads eta and Gamma one
    // cell outside the valid box, and on a symmetry face or at the bubble
    // centre a stale ghost drags the band value back
    // (coarse slab: Gamma grew at 0.48 of the area rate, 2026-11-02).
    FillBoundariesWithBC(lev, t_new[lev], eta_bc, { eta_mf[lev].get(), shell_mf[lev].get() });
    if (shell_slip == 7 || shell_slip == 11 || shell_slip == 12 || shell_slip == 15) return;   // m = Gamma |grad psi| is ~0 in the bulk: an extension would create shell
    if (!(marmottant || apply_surface_tension)) return;   // no shell row
    // SHELL BULK EXTENSION.  Outside the eta band the areal density Gamma is
    // physically meaningless and its -Gamma div_s(u) evolution integrates the
    // bulk dilatation without bound (measured: 1.0 -> negative -> NaN by
    // ~5e-3 on the driven bubble).  The RHS freezes it in the bulk (see the
    // shell row gate).  Here a bulk cell adjacent to the band inherits the
    // band value (constant extension along the normal), so a cell entrained
    // by interface motion starts from the shell state actually there; cells
    // with no band neighbour are anchored at Gamma = 1.  (Pinning every bulk
    // cell to 1, the pre-2026-10 behaviour, fed a collapsing bubble's band
    // unstrained shell: Gamma -23.7% at R/R0 = 0.43, no recovery on rebound.)
    // Race-free: only BULK cells are written and only BAND cells are read.
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bxs = mfi.validbox();
        auto shl  = shell_mf[lev]->array(mfi);
        auto etas = eta_mf[lev]->const_array(mfi);
        amrex::ParallelFor(bxs, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar e = etas(i, j, k);
            const bool bulk = (e < bth || e > 1.0 - bth);
            if (!bulk) return;
            Set::Scalar acc = 0.0; int n = 0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                for (int sgn = -1; sgn <= 1; sgn += 2)
                {
                    const int ii = i + (d == 0 ? sgn : 0);
                    const int jj = j + (d == 1 ? sgn : 0);
                    const int kk = k + (d == 2 ? sgn : 0);
                    if (!etas.contains(ii, jj, kk)) continue;
                    const Set::Scalar en = etas(ii, jj, kk);
                    if (en > bth && en < 1.0 - bth) { acc += shl(ii, jj, kk); ++n; }
                }
            shl(i, j, k) = n ? acc / Set::Scalar(n) : 1.0;
        });
    }
}

void Hydro2::SubcycleShellViscous(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::SubcycleShellViscous");
    if (!shell_visc_subcycle) return;
    if (shell_kappa_s == 0.0 && shell_mu_s == 0.0) return;
    if (!apply_surface_tension) return;

    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar kap_s = shell_kappa_s - shell_mu_s;
    // shell.slip = 7: Gamma = m / |grad psi| (the phase masses do not change in the sub-cycles)
    const int slip7v = (shell_slip == 7 || shell_slip == 11 || shell_slip == 12 || shell_slip == 15) && marmottant;
    amrex::MultiFab Gp7v;
    if (slip7v) ShellPsiGrad(lev, time, Gp7v);
    const Set::Scalar dxm = std::min({AMREX_D_DECL(DX[0], DX[1], DX[2])});
    const Set::Scalar kap_eff = std::abs(kap_s) + 2.0 * std::abs(shell_mu_s);
    const Set::Scalar sm = small;

    // Gas reference density: the smallest density the stability estimate is
    // allowed to see.  Taken from the phase-1 (gas) IC scale rather than
    // `small`, which is a numerical regulariser, not a physical density.
    const Set::Scalar rho_floor_l = std::max(shell_nus_rho_floor, 1.0e-12);
    // Normal force -Delta_v grad(eta), Delta_v = keff div_s(u) kappa (tensor
    // curvature), the jump the relaxation imposes; the tensor below supplies
    // only its tangential part.
    const int heat_split = shell_heat_split;
    const Set::Scalar eps_rv = epsilon, cut_rv = std::max(cutoff, 1.0e-12);
    // effusivity split: e_k^2 = k_k rho_k cp_k (rho_k pure density, from the cell)
    const Set::Scalar kc0 = thermal_k0 * eos0.Cp(), kc1 = thermal_k1 * eos1.Cp();
    const Set::Scalar cut_h = cutoff;
    const Set::Scalar keff_v = kap_s + 2.0 * shell_mu_s / (AMREX_SPACEDIM - 1);
    // shell.tension_floor: the viscous tension actually applied is raised so that
    // sigma_eff + T_visc >= floor (sigma_eff from the cell's Gamma, as in the static force).
    const Set::Scalar tfl = shell_tension_floor;
    const bool tfl_on = tfl > -1.0e299;
    const int marm_v = marmottant;
    const Set::Scalar sig_v = sigma, chi_v = marmottant_chi, Gb_v = marmottant_Gamma_buck;
    const Set::Scalar sbrk_v = marmottant_sigma_break, sigw_v = sigma;
    const Set::Scalar alpha_v = shell_kappa_s_alpha;   // shell.kappa_s_alpha (Cross law)
    // kappa from the capillary tensor (eta is frozen over the sub-cycles)
    amrex::MultiFab Kt_v;     // filled below, once the eta ghosts are current

    // Refresh the eta and rho ghosts.  This routine runs after the RK update,
    // where the ghosts of eta/rho are not current: on a RADIAL slab the face
    // cuts through the shell, and the stale ghosts there cut kappa_eff to 0.34
    // (vs 0.988 dt-limited).  On a fine level the coarse-fine ghosts are stale
    // the same way, so they come from FillPatch (coarse interpolation + the
    // physical BC), as the momentum ghosts below do.  Valid cells are unchanged.
    // The field BCs are applied on every face, NSCBC runs included: the
    // symmetry-plane (mirror) and RADIAL ghosts must be current for the
    // viscous stencils, and an NSCBC face only gets a zero-gradient copy here,
    // which the next FillGhost4BC replaces.  (Skipping them under NSCBC left
    // the axis mirror ghosts stale and grew a grid-scale v sawtooth on the
    // axis row, 2026-10-24.)
    {
        if (lev == 0)
        {
            FillBoundariesWithBC(lev, time + dt, eta_bc, { eta_mf[lev].get() });
            FillBoundariesWithBC(lev, time + dt, density_bc, { density_mf[lev].get() });
        }
        else
        {
            FillPatch(lev, time + dt, eta_mf, *eta_mf[lev], *eta_bc, 0);
            FillPatch(lev, time + dt, density_mf, *density_mf[lev], *density_bc, 0);
            if (radial_bc)
            {
                RadialGhostFill(lev, *eta_mf[lev], false);
                RadialGhostFill(lev, *density_mf[lev], false);
            }
        }
    }
    // (Refresh moved ahead of the nu_s estimate below: the estimate's
    // gradient stencil reads the eta ghosts, which right after a regrid were
    // unfilled and gave nu_s ~ 3e11 on INCLINE, 2026-11-02.)

    // ---- N from the SAME stability measure the dt limiter uses -------------
    // nu_s = kap_eff |grad eta| / rho, reduced over the band, then
    // N = ceil( dt / (cfl_v dx^2 / (2 d nu_s)) ).
    Set::Scalar nus_max = 0.0;
    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto rho = density_mf[lev]->const_array(mfi);
        amrex::LoopOnCpu(bx, [&] (int i, int j, int k) {
            Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
            const Set::Scalar gem = ge.lpNorm<2>();
            if (gem < g_gem_min) return;
            // FLOOR rho at the gas reference density.  `small` (1e-8) is far
            // below any physical density here, so without this floor a single
            // near-vacuum band cell dominates the reduction: at max_level 5 one
            // cell with rho ~ 3e-8 drove nu_s to 1.06e7 (1e10 x the healthy
            // 1.085e-3), which is what crashed the sub-cycled run.
            const Set::Scalar rho_f = std::max(rho(i,j,k), rho_floor_l);
            nus_max = std::max(nus_max, kap_eff * gem / rho_f);
        });
    }
    amrex::ParallelDescriptor::ReduceRealMax(nus_max);
    if (nus_max <= 0.0) return;

    const Set::Scalar cflv   = (cfl_v == cfl_v) ? cfl_v : 0.4;   // NaN-safe
    // NOTE: `small` (1e-8) is a DENSITY/PRESSURE regulariser and must never be
    // added to a time.  dt_sub here is ~1e-12, so `dt_sub + small` is dominated
    // by small and drives N to 1 -- i.e. one explicit step at the UNTHROTTLED
    // dt, precisely the instability this routine exists to avoid.  nus_max > 0
    // is already guaranteed by the early return above, so no guard is needed.
    const Set::Scalar dt_sub = cflv * dxm * dxm
                             / (2.0 * AMREX_SPACEDIM * nus_max);
    // 64-bit: ceil(dt/dt_sub) reached ~6e11 when nu_s blew up, and the cast to
    // int overflowed to a negative value, which the `N < 1` clamp then turned
    // into N = 1 -- the sub-cycle silently disabled itself exactly when it was
    // most needed, taking one explicit step at the unthrottled dt (89 NaNs at
    // max_level 5).  Compute in long long, clamp, THEN narrow.
    long long Nll = (dt_sub > 0.0)
                  ? (long long)std::ceil((long double)dt / (long double)dt_sub)
                  : 1LL;
    if (Nll < 1LL) Nll = 1LL;
    if (Nll > (long long)shell_visc_subcycle_max)
    {
        if (amrex::ParallelDescriptor::IOProcessor())
            Util::Warning(INFO, "SubcycleShellViscous lev=", lev, ": required N=", Nll,
                          " exceeds shell.kappa_s_subcycle_max=", shell_visc_subcycle_max,
                          " (nu_s=", nus_max, ").  Clamping -- the sub-step is NO LONGER "
                          "stability-bounded; raise the cap or check for vacuum cells.");
        Nll = (long long)shell_visc_subcycle_max;
    }
    const int N = (int)Nll;
    const Set::Scalar dtau = dt / (Set::Scalar)N;

    if (shell_visc_verbose && amrex::ParallelDescriptor::IOProcessor())
        Util::Message(INFO, "SubcycleShellViscous lev=", lev, " N=", N,
                      " dtau=", dtau, " nu_s=", nus_max);

    // ---- N explicit sub-steps on the band ---------------------------------
    // Omega_visc is built ONCE PER CELL into scratch, then differenced.  The
    // first version rebuilt it at all six neighbours inside the divergence,
    // which recomputed an eta-gradient AND the full 3x3 velocity Jacobian seven
    // times per cell (~168 stencil reads) instead of once (~30).  The scratch is
    // allocated outside the sub-step loop so the saving is not paid back in
    // allocation churn.
    const int nOm = (AMREX_SPACEDIM == 2) ? 3 : 6;
    amrex::MultiFab Omv(momentum_mf[lev]->boxArray(),
                        momentum_mf[lev]->DistributionMap(), nOm, 1);

    Kt_v.define(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, 0);
    const int ffv = jump_face_force;
    amrex::MultiFab Jvf(momentum_mf[lev]->boxArray(), momentum_mf[lev]->DistributionMap(), 1, 1);   // TEMP jump.face_force
    TensorCurvature(lev, Kt_v, time + dt);
    const int footv = shell_foot_rate;
    for (int it = 0; it < N; ++it)
    {
        // Momentum ghosts through the field BCs (mirror at symmetry planes,
        // radial on RADIAL faces), then v = M/rho on the whole fab.  A plain
        // velocity FillBoundary left the domain-face ghosts at their
        // start-of-step values for every sub-step.  rho is fixed here.
        if (lev > 0)
        {
            FillPatch(lev, time + dt, momentum_mf, *momentum_mf[lev], *momentum_bc, 0);
            if (radial_bc) RadialGhostFill(lev, *momentum_mf[lev], true);
        }
        else
            FillBoundariesWithBC(lev, time + dt, momentum_bc, { momentum_mf[lev].get() });
        for (amrex::MFIter mfi(*velocity_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box gb = mfi.growntilebox(velocity_mf[lev]->nGrow());
            auto M = momentum_mf[lev]->const_array(mfi);
            auto r = density_mf[lev]->const_array(mfi);
            auto v = velocity_mf[lev]->array(mfi);
            amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar rc = std::max(r(i, j, k), sm);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) v(i, j, k, d) = M(i, j, k, d) / rc;
            });
        }
        amrex::MultiFab Gfv;   // shell.foot_rate: {r, Kt} at each band cell's foot point (this sub-step's velocity)
        if (footv) { amrex::MultiFab Lr; FootRateSource(*eta_mf[lev], *velocity_mf[lev], Kt_v, DX, Lr); FootExtend(lev, Lr, Gfv); }
        Omv.setVal(0.0);

        // ---- build Omega_visc on a grown box (band cells only) -------------
        for (amrex::MFIter mfi(Omv, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &gbx = mfi.growntilebox(1);
            auto et  = eta_mf[lev]->const_array(mfi);
            auto vel = velocity_mf[lev]->const_array(mfi);
            auto om  = Omv.array(mfi);
            auto shv = shell_mf[lev]->const_array(mfi);
            auto gpv = slip7v ? Gp7v.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                const Set::Scalar gem = ge.lpNorm<2>();
                if (gem < g_gem_min) return;        // off-band: Omega_visc = 0
                const Set::Vector nh = ge / gem;
                Set::Matrix gu = Set::Matrix::Zero();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    gu.col(d) = Numeric::Gradient(vel, i, j, k, d, DX);
                const Set::Scalar ds = gu.trace() - nh.dot(gu * nh);
                if (g_shell_rupture && !slip7v && shv(i, j, k) <= 0.0) return;   // broken shell: no shell viscosity
                Set::Scalar se = kap_s * ds          // viscous ONLY (its tangential part is kept below)
                               * ShellViscFactor(ds, alpha_v);
                if (tfl_on)
                    se = std::max(se, tfl - SigmaEffFromGamma(marm_v ? (slip7v ? ((gpv(i, j, k) > 0.0) ? shv(i, j, k) / gpv(i, j, k) : 1.0) : shv(i, j, k)) : 0.0, 1.0, marm_v,
                                                             sig_v, chi_v, Gb_v, sbrk_v, sigw_v));
                // |grad eta| P_ab = gem d_ab - ge_a ge_b / gem
                om(i,j,k,0) = se * (gem - ge(0)*ge(0)/gem);     // xx
                om(i,j,k,1) = se * (gem - ge(1)*ge(1)/gem);     // yy
#if AMREX_SPACEDIM == 2
                om(i,j,k,2) = se * (-ge(0)*ge(1)/gem);          // xy
#else
                om(i,j,k,2) = se * (gem - ge(2)*ge(2)/gem);     // zz
                om(i,j,k,3) = se * (-ge(0)*ge(1)/gem);          // xy
                om(i,j,k,4) = se * (-ge(0)*ge(2)/gem);          // xz
                om(i,j,k,5) = se * (-ge(1)*ge(2)/gem);          // yz
#endif
            });
        }
        Omv.FillBoundary(geom[lev].periodicity());

        // TEMP jump.face_force: per-cell viscous jump Jv = -Tv kappa first, then face form below
        if (ffv)
        {
            Jvf.setVal(0.0);
            for (amrex::MFIter mfi(*momentum_mf[lev], false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                auto et  = eta_mf[lev]->const_array(mfi);
                auto vel = velocity_mf[lev]->const_array(mfi);
                auto ktv = Kt_v.const_array(mfi);
                auto shv = shell_mf[lev]->const_array(mfi);
                auto gpv = slip7v ? Gp7v.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
                auto jv  = Jvf.array(mfi);
                auto gfv = footv ? Gfv.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                    const Set::Scalar gem = ge.lpNorm<2>();
                    if (gem < g_gem_min) return;
                    const Set::Vector nh = ge / gem;
                    Set::Matrix gu = Set::Matrix::Zero();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        gu.col(d) = Numeric::Gradient(vel, i, j, k, d, DX);
                    const Set::Scalar ds_fl = gu.trace() - nh.dot(gu * nh);
                    const bool ft = footv && gfv(i, j, k, 0) == gfv(i, j, k, 0);   // shell.foot_rate: foot value available
                    const Set::Scalar ds = ft ? gfv(i, j, k, 0)
                                              : LiquidInterfaceRate(et, i, j, k, DX, eps_rv, cut_rv, gu,
                                                    Set::Vector(AMREX_D_DECL(vel(i,j,k,0), vel(i,j,k,1), vel(i,j,k,2))), ds_fl);
                    const Set::Scalar kt = ft ? gfv(i, j, k, 1) : ktv(i, j, k);
                    Set::Scalar Tv = keff_v * ShellViscFactor(ds, alpha_v) * ds;
                    if (tfl_on)
                    {
                        const Set::Scalar Tlo = tfl - SigmaEffFromGamma(marm_v ? (slip7v ? ((gpv(i, j, k) > 0.0) ? shv(i, j, k) / gpv(i, j, k) : 1.0) : shv(i, j, k)) : 0.0, 1.0, marm_v,
                                                                        sig_v, chi_v, Gb_v, sbrk_v, sigw_v);
                        if (Tv < Tlo) Tv = Tlo;
                    }
                    jv(i, j, k) = -Tv * kt;
                });
            }
            Jvf.FillBoundary(geom[lev].periodicity());
            FaceForceDomainGhosts(Jvf, geom[lev]);
        }
        // ---- apply div(Omega_visc) to momentum, dissipation to energy ------
        for (amrex::MFIter mfi(*momentum_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto et  = eta_mf[lev]->const_array(mfi);
            auto vel = velocity_mf[lev]->const_array(mfi);
            auto om  = Omv.const_array(mfi);
            auto M   = momentum_mf[lev]->array(mfi);
            auto E   = energy_per_vol_mf[lev]->array(mfi);
            // The RK state carries BOTH the total energy and the phasic
            // internal energies (solution vector entries [3], [5], [6]).  The
            // dissipation must go into E0/E1 as well, or the invariant
            //     E_vol = KE + E0 + E1
            // is broken and RelaxAndReinit -- which enforces p0 = p1 from
            // (eta, E0, E1) immediately after this routine -- never sees the
            // deposited heat.  Writing E_vol alone was why max_level 5 diverged
            // while max_level 3 (64x fewer sub-steps) stayed within 0.05%.
            auto E0  = energy0_mf[lev]->array(mfi);
            auto E1  = energy1_mf[lev]->array(mfi);
            auto ar0 = rho_eta0_mf[lev]->const_array(mfi);
            auto ar1 = rho_eta1_mf[lev]->const_array(mfi);
            auto ktv  = Kt_v.const_array(mfi);
            auto gfv  = footv ? Gfv.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            auto shv  = shell_mf[lev]->const_array(mfi);
            auto gpv  = slip7v ? Gp7v.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            auto jvf  = Jvf.const_array(mfi);
            const Set::Scalar idx = 0.5 / DX[0], idy = 0.5 / DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar idz = 0.5 / DX[2];
#endif
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                const Set::Scalar gem = ge.lpNorm<2>();
                if (gem < g_gem_min) return;        // off-band: nothing to do

                Set::Vector F = Set::Vector::Zero();
#if AMREX_SPACEDIM == 2
                F(0) = (om(i+1,j,k,0) - om(i-1,j,k,0)) * idx
                     + (om(i,j+1,k,2) - om(i,j-1,k,2)) * idy;
                F(1) = (om(i+1,j,k,2) - om(i-1,j,k,2)) * idx
                     + (om(i,j+1,k,1) - om(i,j-1,k,1)) * idy;
#else
                F(0) = (om(i+1,j,k,0) - om(i-1,j,k,0)) * idx
                     + (om(i,j+1,k,3) - om(i,j-1,k,3)) * idy
                     + (om(i,j,k+1,4) - om(i,j,k-1,4)) * idz;
                F(1) = (om(i+1,j,k,3) - om(i-1,j,k,3)) * idx
                     + (om(i,j+1,k,1) - om(i,j-1,k,1)) * idy
                     + (om(i,j,k+1,5) - om(i,j,k-1,5)) * idz;
                F(2) = (om(i+1,j,k,4) - om(i-1,j,k,4)) * idx
                     + (om(i,j+1,k,5) - om(i,j-1,k,5)) * idy
                     + (om(i,j,k+1,2) - om(i,j,k-1,2)) * idz;
#endif
                const Set::Vector nh = ge / gem;
                Set::Matrix gu = Set::Matrix::Zero();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    gu.col(d) = Numeric::Gradient(vel, i, j, k, d, DX);
                const Set::Scalar ds_fl = gu.trace() - nh.dot(gu * nh);
                // interface area rate: liquid-side band cells map their own stretch
                // through the flux tube (LiquidInterfaceRate, see Parse shell.rate_local),
                // or take r and Kt from the foot point on eta = 1/2 (shell.foot_rate = 1)
                const bool ft = footv && gfv(i, j, k, 0) == gfv(i, j, k, 0);
                const Set::Scalar ds = ft ? gfv(i, j, k, 0)
                                          : LiquidInterfaceRate(et, i, j, k, DX, eps_rv, cut_rv, gu,
                                                Set::Vector(AMREX_D_DECL(vel(i,j,k,0), vel(i,j,k,1), vel(i,j,k,2))), ds_fl);
                const Set::Scalar kt = ft ? gfv(i, j, k, 1) : ktv(i, j, k);
                // tangential part of div(Omega_visc) + normal jump force -Delta_v grad(eta)
                F -= nh * nh.dot(F);
                const Set::Scalar fvis = ShellViscFactor(ds, alpha_v);
                Set::Scalar Tv = keff_v * fvis * ds;    // viscous tension of the normal jump force
                bool clamped = false;
                if (tfl_on)
                {
                    const Set::Scalar Tlo = tfl - SigmaEffFromGamma(marm_v ? (slip7v ? ((gpv(i, j, k) > 0.0) ? shv(i, j, k) / gpv(i, j, k) : 1.0) : shv(i, j, k)) : 0.0, 1.0, marm_v,
                                                                    sig_v, chi_v, Gb_v, sbrk_v, sigw_v);
                    if (Tv < Tlo) { Tv = Tlo; clamped = true; }
                }
                if (!ffv) F -= (Tv * kt) * ge;
                else
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    {
                        const int ip = i + (d == 0), jp = j + (d == 1), kp = k + (d == 2);
                        const int im = i - (d == 0), jm = j - (d == 1), km = k - (d == 2);
                        const Set::Scalar e0 = et(i, j, k), ep = et(ip, jp, kp), em = et(im, jm, km);
                        F(d) += (jvf(i, j, k) * (ep - em) + jvf(ip, jp, kp) * (ep - e0)
                                 + jvf(im, jm, km) * (e0 - em)) / (4.0 * DX[d]);
                    }
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    M(i,j,k,d) += dtau * F(d);

                // Dissipation -> heat.  Omega_visc:grad u = kap_s |grad eta| (div_s u)^2
                // is positive-definite, so the kinetic energy removed by F is
                // returned as internal energy rather than stored in a reservoir.
                Set::Scalar udotF = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) udotF += vel(i,j,k,d) * F(d);
                // Total energy gains the work done on the fluid (u.F, which
                // shows up as kinetic) plus the dissipation (internal).
                // clamped cells: the floor acts as an elastic (non-dissipative) tension
                const Set::Scalar diss = clamped ? 0.0 : keff_v * fvis * gem * ds * ds_fl;
                E(i,j,k) += dtau * (udotF + diss);
                // The dissipation is INTERNAL energy, so it must also be split
                // into the phasic reservoirs, by volume fraction as elsewhere
                // in this file (eta weights phase 0).  u.F is kinetic and does
                // NOT enter E0/E1.
                // shell.heat_split (see Parse).  A volume-fraction share heats the
                // band gas strongly (it holds little heat); the physical split of a
                // heat source on a gas/liquid contact is by thermal effusivity.
                Set::Scalar a0 = std::min(std::max(et(i,j,k), 0.0), 1.0);
                if (heat_split == 2)
                {
                    // Both phases present above the cutoff: effusivity share.
                    // Otherwise the heat stays with the phase that is there.
                    if (a0 > cut_h && a0 < 1.0 - cut_h)
                    {
                        const Set::Scalar r0p = std::max(ar0(i,j,k), 0.0) / a0;
                        const Set::Scalar r1p = std::max(ar1(i,j,k), 0.0) / (1.0 - a0);
                        const Set::Scalar e0 = std::sqrt(kc0 * r0p), e1 = std::sqrt(kc1 * r1p);
                        a0 = e0 / std::max(e0 + e1, 1.0e-300);
                    }
                    else a0 = (a0 >= 0.5) ? 1.0 : 0.0;
                }
                E0(i,j,k) += dtau * diss * a0;
                E1(i,j,k) += dtau * diss * (1.0 - a0);
            });
        }

        // Refresh the velocity the next sub-step differentiates (band only).
        for (amrex::MFIter mfi(*momentum_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto et  = eta_mf[lev]->const_array(mfi);
            auto M   = momentum_mf[lev]->const_array(mfi);
            auto rho = density_mf[lev]->const_array(mfi);
            auto v   = velocity_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
                if (ge.lpNorm<2>() < g_gem_min) return;
                const Set::Scalar r = rho(i,j,k) + sm;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) v(i,j,k,d) = M(i,j,k,d) / r;
            });
        }
    }
}

// RadialGhostFill(): radially symmetric slab ghosts (2026-10-23).
// Domain x in [0,L], y (,z) in [0,h]; centre at the origin; ylo (,zlo) and
// xlo are symmetry planes; yhi (,zhi) are RADIAL.  The near-axis row
// (j = k = 0, at y0 = dy/2, z0 = dz/2) is a 1D radial profile f(r_i),
// r_i = sqrt(x_i^2 + y0^2 + z0^2).  A ghost at radius r takes f(r) by cubic
// Lagrange interpolation in x_eff = sqrt(r^2 - y0^2 - z0^2), which maps r back
// onto the row's uniform x spacing.  The profile is even in x (x_{-1-i} has
// the same r as x_i), so the stencil mirrors through the centre; past the
// last cell it is held constant.  The interpolant is clipped to the two
// bracketing samples so eta and the densities cannot overshoot.
// Vectors (momentum): the profile is u_r = u.(x_i, y0, z0)/r_i and the ghost
// gets u_r(r) (x, y, z)/r -- no swirl.
// The profile is gathered over all ranks (nx values per component), so no
// assumption is made about how the boxes split the slab.
void Hydro2::RadialGhostFill(int lev, amrex::MultiFab &mf, bool is_vector)
{
    BL_PROFILE("Integrator::Hydro2::RadialGhostFill");
    if (radial_ghost_fill == 1) { RadialGhostFillLocal(lev, mf, is_vector); return; }
    const amrex::Box dom = geom[lev].Domain();
    const Set::Scalar *DX = geom[lev].CellSize();
    const int ilo = dom.smallEnd(0), nx = dom.length(0);
    const int jax = dom.smallEnd(1);
    const int kax = (AMREX_SPACEDIM == 3) ? dom.smallEnd(2) : 0;
    const int nc = is_vector ? 1 : mf.nComp();
    const Set::Scalar y0 = 0.5 * DX[1];
    const Set::Scalar z0 = (AMREX_SPACEDIM == 3) ? 0.5 * DX[2] : 0.0;
    const Set::Scalar off2 = y0 * y0 + z0 * z0;

    // ---- gather the near-axis profile ------------------------------------
    // On a fine level the patches cover only part of x.  The axis row is read
    // over the valid cells AND the x-ghosts of each patch (coarse-fine ghosts,
    // already interpolated by FillPatch); a valid value wins over a ghost one,
    // and x-cells no patch reaches take the nearest covered value.  On level 0
    // every x-cell is valid, so this is the plain axis row.
    std::vector<Set::Scalar> prof((size_t)nx * nc, 0.0), pghost((size_t)nx * nc, 0.0);
    std::vector<Set::Scalar> wval((size_t)nx, 0.0), wgh((size_t)nx, 0.0);
    for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &vb = mfi.validbox();
        if (jax < vb.smallEnd(1) || jax > vb.bigEnd(1)) continue;
#if AMREX_SPACEDIM == 3
        if (kax < vb.smallEnd(2) || kax > vb.bigEnd(2)) continue;
#endif
        const amrex::Box fb = mf[mfi].box();
        auto a = mf.const_array(mfi);
        for (int i = std::max(fb.smallEnd(0), ilo); i <= std::min(fb.bigEnd(0), ilo + nx - 1); ++i)
        {
            const int ii = i - ilo;
            const bool valid = (i >= vb.smallEnd(0) && i <= vb.bigEnd(0));
            std::vector<Set::Scalar> &dst = valid ? prof : pghost;
            (valid ? wval : wgh)[ii] += 1.0;
            if (is_vector)
            {
                const Set::Scalar x = (ii + 0.5) * DX[0];
                const Set::Scalar r = std::sqrt(x * x + off2);
                Set::Scalar ur = a(i, jax, kax, 0) * x + a(i, jax, kax, 1) * y0;
#if AMREX_SPACEDIM == 3
                ur += a(i, jax, kax, 2) * z0;
#endif
                dst[ii] += ur / r;
            }
            else
                for (int n = 0; n < nc; ++n) dst[(size_t)ii * nc + n] += a(i, jax, kax, n);
        }
    }
    amrex::ParallelDescriptor::ReduceRealSum(prof.data(), (int)prof.size());
    amrex::ParallelDescriptor::ReduceRealSum(pghost.data(), (int)pghost.size());
    amrex::ParallelDescriptor::ReduceRealSum(wval.data(), (int)wval.size());
    amrex::ParallelDescriptor::ReduceRealSum(wgh.data(), (int)wgh.size());
    {
        std::vector<char> have((size_t)nx, 0);
        for (int ii = 0; ii < nx; ++ii)
        {
            if (wval[ii] > 0.0)
                for (int n = 0; n < nc; ++n) prof[(size_t)ii * nc + n] /= wval[ii];
            else if (wgh[ii] > 0.0)
                for (int n = 0; n < nc; ++n) prof[(size_t)ii * nc + n] = pghost[(size_t)ii * nc + n] / wgh[ii];
            have[ii] = (wval[ii] > 0.0 || wgh[ii] > 0.0);
        }
        // nearest covered value for the x-cells no patch reaches
        int last = -1;
        std::vector<int> nearest((size_t)nx, -1);
        for (int ii = 0; ii < nx; ++ii) { if (have[ii]) last = ii; nearest[ii] = last; }
        last = -1;
        for (int ii = nx - 1; ii >= 0; --ii)
        {
            if (have[ii]) last = ii;
            if (!have[ii] && (nearest[ii] < 0 || (last >= 0 && last - ii < ii - nearest[ii]))) nearest[ii] = last;
        }
        for (int ii = 0; ii < nx; ++ii)
            if (!have[ii] && nearest[ii] >= 0)
                for (int n = 0; n < nc; ++n) prof[(size_t)ii * nc + n] = prof[(size_t)nearest[ii] * nc + n];
    }

    // ---- fill every ghost outside a RADIAL face ---------------------------
    const int jhi = dom.bigEnd(1);
    const bool ry = radial_face_hi[1];
#if AMREX_SPACEDIM == 3
    const int khi = dom.bigEnd(2);
    const bool rz = radial_face_hi[2];
#endif
    auto sample = [&](int n, Set::Scalar r) -> Set::Scalar {
        const Set::Scalar xe = std::sqrt(std::max(r * r - off2, 0.0));
        const Set::Scalar s = xe / DX[0] - 0.5;
        if (s >= (Set::Scalar)(nx - 1)) return prof[(size_t)(nx - 1) * nc + n];
        const int i0 = (int)std::floor(s);
        const Set::Scalar t = s - i0;
        auto at = [&](int i) -> Set::Scalar {
            if (i < 0) i = -1 - i;              // even through the centre
            if (i > nx - 1) i = nx - 1;
            return prof[(size_t)i * nc + n];
        };
        const Set::Scalar fm = at(i0 - 1), f0 = at(i0), f1 = at(i0 + 1), f2 = at(i0 + 2);
        const Set::Scalar v = -t * (t - 1.0) * (t - 2.0) / 6.0 * fm
                            + (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0 * f0
                            - (t + 1.0) * t * (t - 2.0) / 2.0 * f1
                            + (t + 1.0) * t * (t - 1.0) / 6.0 * f2;
        return std::min(std::max(v, std::min(f0, f1)), std::max(f0, f1));
    };
    for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box fb = mf[mfi].box();
        auto a = mf.array(mfi);
        amrex::LoopOnCpu(fb, [&](int i, int j, int k) {
            bool out = ry && j > jhi;
#if AMREX_SPACEDIM == 3
            out = out || (rz && k > khi);
#endif
            if (!out) return;
            const Set::Scalar x = (i - ilo + 0.5) * DX[0];
            const Set::Scalar y = (j - jax + 0.5) * DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar z = (k - kax + 0.5) * DX[2];
#else
            const Set::Scalar z = 0.0;
#endif
            const Set::Scalar r = std::sqrt(x * x + y * y + z * z);
            if (is_vector)
            {
                const Set::Scalar ur = sample(0, r) / r;
                a(i, j, k, 0) = ur * x;
                a(i, j, k, 1) = ur * y;
#if AMREX_SPACEDIM == 3
                a(i, j, k, 2) = ur * z;
#endif
            }
            else
                for (int n = 0; n < nc; ++n) a(i, j, k, n) = sample(n, r);
        });
    }
}

// RadialGhostFillLocal(): radial.ghost_fill = 1 (2026-11-24).
//
// Why: RadialGhostFill fills every ghost outside a RADIAL face from the
// near-axis row.  The face ghosts are then slaved to cells half a slab width
// away, not to their neighbours: a transverse disturbance at the face is
// overwritten with the axis value, the axis feels the face through the
// interior a crossing time later, and the loop closes.  On the 3D AMR slab
// (input_Multiphase_Normalized_Slab3D) this grows a standing transverse
// acoustic mode (x2.4 per 0.25 time units from t~1.5) that NaNs the run at
// t~3.4, with or without shell viscosity; the same deck with symmetry walls
// instead keeps the mode decaying.  On fine levels the axis row also stops at
// the patch edge, and ghosts at larger r got the edge value held constant.
//
// What: a ghost at P = (x, y, z) outside the face y = h takes the field at
// the mirror point P' = (x', 2h - y, z) with |P'| = |P| -- the interior point
// at the SAME radius on the mirrored row next to the face, x' = sign(x)
// sqrt(r^2 - y'^2 - z^2) (z likewise for the z face; both for the corner).
// A radial field f(r) is reproduced exactly (P and P' have the same r), so
// the 1D radial problem is unchanged; anything non-radial sees the mirror,
// i.e. an angular symmetry condition next to the face instead of a
// prescription from the axis.
// Vectors: u' at P' is split into u_r' = u'.P'/r and the non-radial part
// t' = u' - u_r' P'/r.  The ghost gets u_r' P/r (radial part continuous, as
// in the exact radial flow) plus t' with its face-normal component(s)
// reversed (the wall-like part).
// Interpolation along the mirrored row is the same clipped cubic in x as
// RadialGhostFill; x < 0 maps through the xlo symmetry plane (momentum x
// odd), and rows below the lo faces through those symmetry planes.
// The rows within nGrow of the RADIAL faces are gathered over all ranks
// (valid cells first, then the patches' x-ghosts, then the nearest covered
// x), as RadialGhostFill does for the axis row.
void Hydro2::RadialGhostFillLocal(int lev, amrex::MultiFab &mf, bool is_vector)
{
    BL_PROFILE("Integrator::Hydro2::RadialGhostFillLocal");
    const amrex::Box dom = geom[lev].Domain();
    const Set::Scalar *DX = geom[lev].CellSize();
    const int ilo = dom.smallEnd(0), nx = dom.length(0);
    const int jlo = dom.smallEnd(1), jhi = dom.bigEnd(1), ny = dom.length(1);
#if AMREX_SPACEDIM == 3
    const int klo = dom.smallEnd(2), khi = dom.bigEnd(2), nz = dom.length(2);
    const bool rz = radial_face_hi[2];
#else
    const int klo = 0, khi = 0, nz = 1;
    const bool rz = false;
#endif
    const bool ry = radial_face_hi[1];
    const int nc = mf.nComp();
    const int ng = std::min(mf.nGrow(1), ny);
#if AMREX_SPACEDIM == 3
    const int ngz = std::min(mf.nGrow(2), nz);
#else
    const int ngz = 0;
#endif
    if (ng == 0 && ngz == 0) return;

    // ---- rows (jj, kk) inside the domain within ng of a RADIAL face ---------
    auto in_band = [&](int jj, int kk) -> bool {
        return (ry && jj > jhi - ng) || (rz && kk > khi - ngz);
    };
    std::vector<int> rowid((size_t)ny * nz, -1);
    int nrow = 0;
    for (int kk = 0; kk < nz; ++kk)
        for (int jj = 0; jj < ny; ++jj)
            if (in_band(jj + jlo, kk + klo)) rowid[(size_t)kk * ny + jj] = nrow++;
    if (nrow == 0) return;

    // x range the level's fabs reach (a fine level covers only part of the
    // slab): gather only that, offset ioff = first covered global x index
    int ixlo = ilo + nx - 1, ixhi = ilo;
    for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box fb = mf[mfi].box();
        ixlo = std::min(ixlo, std::max(fb.smallEnd(0), ilo));
        ixhi = std::max(ixhi, std::min(fb.bigEnd(0), ilo + nx - 1));
    }
    amrex::ParallelDescriptor::ReduceIntMin(ixlo);
    amrex::ParallelDescriptor::ReduceIntMax(ixhi);
    if (ixhi < ixlo) return;
    const int ioff = ixlo - ilo, nxl = ixhi - ixlo + 1;

    // buffer layout per (row, i): [valid sum (nc) | ghost sum (nc) | w_valid | w_ghost]
    const int stride = 2 * nc + 2;
    std::vector<Set::Scalar> buf((size_t)nrow * nxl * stride, 0.0);
    auto B = [&](int row, int ii) -> Set::Scalar * { return &buf[((size_t)row * nxl + ii) * stride]; };
    for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &vb = mfi.validbox();
        const amrex::Box fb = mf[mfi].box();
        auto a = mf.const_array(mfi);
#if AMREX_SPACEDIM == 3
        const int kb0 = std::max(vb.smallEnd(2), klo), kb1 = std::min(vb.bigEnd(2), khi);
#else
        const int kb0 = 0, kb1 = 0;
#endif
        for (int kk = kb0; kk <= kb1; ++kk)
            for (int jj = std::max(vb.smallEnd(1), jlo); jj <= std::min(vb.bigEnd(1), jhi); ++jj)
            {
                const int row = rowid[(size_t)(kk - klo) * ny + (jj - jlo)];
                if (row < 0) continue;
                for (int i = std::max(fb.smallEnd(0), ilo); i <= std::min(fb.bigEnd(0), ilo + nx - 1); ++i)
                {
                    const bool valid = (i >= vb.smallEnd(0) && i <= vb.bigEnd(0));
                    Set::Scalar *b = B(row, i - ixlo);
                    for (int n = 0; n < nc; ++n) b[(valid ? 0 : nc) + n] += a(i, jj, kk, n);
                    b[2 * nc + (valid ? 0 : 1)] += 1.0;
                }
            }
    }
    amrex::ParallelDescriptor::ReduceRealSum(buf.data(), (int)buf.size());

    // resolve each row: valid average, else ghost average, else nearest covered x
    std::vector<Set::Scalar> prof((size_t)nrow * nxl * nc, 0.0);
    for (int row = 0; row < nrow; ++row)
    {
        std::vector<char> have((size_t)nxl, 0);
        for (int ii = 0; ii < nxl; ++ii)
        {
            const Set::Scalar *b = B(row, ii);
            Set::Scalar *p = &prof[((size_t)row * nxl + ii) * nc];
            if (b[2 * nc] > 0.0)          { for (int n = 0; n < nc; ++n) p[n] = b[n] / b[2 * nc]; have[ii] = 1; }
            else if (b[2 * nc + 1] > 0.0) { for (int n = 0; n < nc; ++n) p[n] = b[nc + n] / b[2 * nc + 1]; have[ii] = 1; }
        }
        std::vector<int> nearest((size_t)nxl, -1);
        int last = -1;
        for (int ii = 0; ii < nxl; ++ii) { if (have[ii]) last = ii; nearest[ii] = last; }
        last = -1;
        for (int ii = nxl - 1; ii >= 0; --ii)
        {
            if (have[ii]) last = ii;
            if (!have[ii] && (nearest[ii] < 0 || (last >= 0 && last - ii < ii - nearest[ii]))) nearest[ii] = last;
        }
        for (int ii = 0; ii < nxl; ++ii)
            if (!have[ii] && nearest[ii] >= 0)
                for (int n = 0; n < nc; ++n)
                    prof[((size_t)row * nxl + ii) * nc + n] = prof[((size_t)row * nxl + nearest[ii]) * nc + n];
    }

    // value of component n on row `row` at x (x may be < 0: xlo symmetry plane;
    // odd for the x momentum), clipped cubic as in RadialGhostFill
    auto sample = [&](int row, int n, Set::Scalar x, bool odd) -> Set::Scalar {
        const Set::Scalar sgn = (x < 0.0 && odd) ? -1.0 : 1.0;
        const Set::Scalar s = std::fabs(x) / DX[0] - 0.5;      // global x index (from xlo)
        const Set::Scalar *P = &prof[(size_t)row * nxl * nc];
        const int i0 = (int)std::floor(s);
        const Set::Scalar t = s - i0;
        // global index -> gathered sample: through the xlo plane (odd for the x
        // momentum), then held at the ends of the gathered range
        auto at = [&](int i) -> Set::Scalar {
            Set::Scalar f = 1.0;
            if (i < 0) { i = -1 - i; if (odd) f = -1.0; }
            int l = i - ioff;
            l = std::max(0, std::min(nxl - 1, l));
            return f * P[(size_t)l * nc + n];
        };
        const Set::Scalar fm = at(i0 - 1), f0 = at(i0), f1 = at(i0 + 1), f2 = at(i0 + 2);
        const Set::Scalar v = -t * (t - 1.0) * (t - 2.0) / 6.0 * fm
                            + (t + 1.0) * (t - 1.0) * (t - 2.0) / 2.0 * f0
                            - (t + 1.0) * t * (t - 2.0) / 2.0 * f1
                            + (t + 1.0) * t * (t - 1.0) / 6.0 * f2;
        return sgn * std::min(std::max(v, std::min(f0, f1)), std::max(f0, f1));
    };

    for (amrex::MFIter mfi(mf, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box fb = mf[mfi].box();
        auto a = mf.array(mfi);
        amrex::LoopOnCpu(fb, [&](int i, int j, int k) {
            const bool oy = ry && j > jhi;
#if AMREX_SPACEDIM == 3
            const bool oz = rz && k > khi;
#else
            const bool oz = false;
#endif
            if (!oy && !oz) return;
            // mirrored row; rows below a lo face map through that symmetry plane
            int jm = oy ? 2 * jhi + 1 - j : j;
            int km = oz ? 2 * khi + 1 - k : k;
            Set::Scalar fy = 1.0, fz = 1.0;               // sign of the y / z component from lo-face reflection
            if (jm < jlo) { jm = 2 * jlo - 1 - jm; fy = -1.0; }
#if AMREX_SPACEDIM == 3
            if (km < klo) { km = 2 * klo - 1 - km; fz = -1.0; }
#endif
            jm = std::max(jlo, std::min(jhi, jm));
#if AMREX_SPACEDIM == 3
            km = std::max(klo, std::min(khi, km));
#else
            km = 0;
#endif
            const int row = rowid[(size_t)(km - klo) * ny + (jm - jlo)];
            if (row < 0) return;                         // ghost deeper than the gathered band (ng > ny)

            const Set::Scalar x = (i - ilo + 0.5) * DX[0];
            const Set::Scalar y = (j - jlo + 0.5) * DX[1];
#if AMREX_SPACEDIM == 3
            const Set::Scalar z = (k - klo + 0.5) * DX[2];
#else
            const Set::Scalar z = 0.0;
#endif
            const Set::Scalar r2 = x * x + y * y + z * z;
            // mirror point: same radius, reflected across the RADIAL face(s)
            const Set::Scalar yp = oy ? 2.0 * (jhi - jlo + 1) * DX[1] - y : y;
#if AMREX_SPACEDIM == 3
            const Set::Scalar zp = oz ? 2.0 * (khi - klo + 1) * DX[2] - z : z;
#else
            const Set::Scalar zp = 0.0;
#endif
            const Set::Scalar xp = std::copysign(std::sqrt(std::max(r2 - yp * yp - zp * zp, 0.0)), x);

            if (!is_vector)
            {
                for (int n = 0; n < nc; ++n) a(i, j, k, n) = sample(row, n, xp, false);
                return;
            }
            // vector: the row stores the field at (|x'|-row, jm, km); undo the lo-face reflections
            Set::Scalar u[3] = {0.0, 0.0, 0.0};
            u[0] = sample(row, 0, xp, true);
            u[1] = fy * sample(row, 1, xp, false);
#if AMREX_SPACEDIM == 3
            u[2] = fz * sample(row, 2, xp, false);
#endif
            const Set::Scalar r = std::sqrt(r2);
            if (r <= 0.0) { for (int d = 0; d < AMREX_SPACEDIM; ++d) a(i, j, k, d) = u[d]; return; }
            const Set::Scalar Pp[3] = {xp, yp, zp}, Pg[3] = {x, y, z};
            Set::Scalar ur = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) ur += u[d] * Pp[d] / r;
            Set::Scalar tg[3];
            for (int d = 0; d < 3; ++d) tg[d] = u[d] - ur * Pp[d] / r;      // non-radial part at P'
            if (oy) tg[1] = -tg[1];                                      // reflected across the face
            if (oz) tg[2] = -tg[2];
            for (int d = 0; d < AMREX_SPACEDIM; ++d) a(i, j, k, d) = ur * Pg[d] / r + tg[d];
        });
    }
}

// ============================================================================
// SubcycleBulkViscous(): bulk (mu, lambda) viscosity as its own operator
// (visc_subcycle = 1).  Called ONCE per step from Advance, after the RK
// advance of the hyperbolic + capillary system and before the primitive
// refresh, i.e. the Lie split  U^{n+1} = L_relax L_visc L_hyper U^n.
//
// N forward-Euler sub-steps of
//     dM/dt = div(tau),      dE/dt = u.div(tau) + Phi,
// with the SAME discrete stress as the RHS path (Mpqrs/dMpqrs contraction of
// hess_u and grad_u built from M and rho; Phi = 2 mu S:S + (lambda-2/3 mu)
// (div u)^2).  rho, eta, E0, E1 are untouched (as in RHS: the dissipation
// enters only the total-energy row; RelaxAndReinit re-partitions it).
//
// Sub-step size from the forward-Euler bound of that operator: the stiffest
// discrete mode has eigenvalue 4 [ (|lambda| + 4/3 mu) + (D-1) mu ] / (rho dx^2),
//     dtau = visc_subcycle_cfl * dx^2 / (2 nu_L),
//     nu_L = max_cells (|lambda| + 4/3 mu + (D-1) mu) / rho.
// Momentum ghosts are refilled every sub-step from momentum_bc (moving walls,
// no-slip) -- FillPatch on fine levels -- so the wall shear is exact; with
// NSCBC only the interior/periodic ghosts are refreshed.
// ============================================================================
void Hydro2::RelaxAndReinit(int lev)
{
    BL_PROFILE("Integrator::Hydro2::RelaxAndReinit");
    if (diag_gasmass) GasLedger(lev, "relax_entry", relax_time);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK relax_entry   lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }

    // Constants for iteration
    const Set::Scalar alpha_floor = 1.0e-12;
    const Set::Scalar DIV_FLOOR   = 1.0e-30;
    const Set::Scalar small_loc   = small;        // kept for legacy callers
    const Set::Scalar cutoff_loc  = cutoff;       // pure-cell guard threshold
    const int tracePT = relax_trace_pT;           // relax.trace_pT: HRM p-T state for trace gas in liquid
    const int opt8 = relax_opt8;                  // relax.opt8: gas keeps its energy instead of the 0 K floor
    const int opt9 = relax_opt9;                  // relax.opt9: freeze alpha only where a phase has no mass
    const int sch20c = relax_sch20;               // relax.sch20: Sch20 closure only
    const int         max_iter    = 30;
    const Set::Scalar newton_tol  = 1.0e-10;
    const int         bisect_max  = 120;
    const Set::Scalar bisect_tol  = 1.0e-12;
    // Practical |f| threshold for the "unconverged_cells" diagnostic.
    // Newton's strict newton_tol=1e-10 over-flags bisection cells that
    // are physically converged at |f| ~ 1e-8.  unconv_threshold relaxes
    // the warning to a value that matches engineering convergence
    // (volume constraint Sum_k (alpha rho)_k v_k(p) = 1 satisfied to
    // 1 part in 10^6).
    const Set::Scalar unconv_threshold = 1.0e-6;


    // Phase EOS constants.  With species.eos = mixture (and the compaction model of an elastic solid) they are
    // per cell; the kernel below takes local copies (gam0, pi0_, ...) from these (DropletDetonation, merged 11-23).
    const Set::Scalar gam0_c = eos0.Gamma();
    const Set::Scalar pi0_c = eos0.P0();
    const int pT4 = (thermal_exchange == 4);                       // joint p-T relaxation
    const Set::Scalar cv0_c = eos0.Cv(), cv1_c = eos1.Cv(), amin_pT = thermal_alpha_min;
    const Set::Scalar gam1_c = eos1.Gamma();
    const Set::Scalar pi1_c = eos1.P0();
    // base-EOS aliases for the diagnostic loops (the main kernel shadows them with per-cell values)
    const Set::Scalar gam0 = gam0_c, pi0_ = pi0_c, gam1 = gam1_c, pi1_ = pi1_c;
    const Species::Eos spe = species.eos();      // species.eos = mixture: gamma of each phase per cell
    const Set::Scalar trace_a = relax_trace_alpha;
    // FINITE-RATE PRESSURE RELAXATION (relax.omega >= 0; DropletDetonation, see the kernel).  < 0 = instantaneous.
    const Set::Scalar omega_r = relax_omega;
    const int share_mass = relax_share_mass;      // finite-rate step: energy defect shared by mass (1) or volume (0)
    const Set::Scalar dt_r = dt[lev] / (defer_relax ? 1.0 : 3.0);     // relaxation runs after every RK stage unless deferred

    // Optional Newton diagnostic.  Per cell write {iter_count, |f(p_final)|};
    // after the ParallelFor reduce to {max iters, max residual, unconverged
    // cell count} and print a one-line summary.
    // MERGE (Marmottant): preserve the shell's areal density across the
    // pressure relaxation below (it moves volume between phases, not lipid).

    ShellBandUniform(lev);

    // MERGE (AcousticBC): honour relax_diag instead of the hardcoded `true`
    // that was on the Marmottant side.  That leftover forced the Newton
    // diagnostic on for every run and is why the logs were full of
    // RelaxAndReinit lines.
    const bool diag_on = (relax_diag != 0);
    std::unique_ptr<amrex::MultiFab> diag_mf;
    if (diag_on)
    {
        // 3 components: [0]=Newton iter count, [1]=|f(p_relaxed)|,
        //               [2]=|p_relaxed - p_reinit|
        // Component 2 detects whether the eq.26 reinit step is amplifying
        // cell-local noise.  If max(|p_relaxed - p_reinit|) is large or
        // correlates with checker artifacts, the reinit step is the noise
        // source (Newton's volume-constraint p disagrees with the energy-
        // consistent p from rho E and the new alpha).
        diag_mf = std::make_unique<amrex::MultiFab>(
            eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 3, 0);
        diag_mf->setVal(0.0);
    }

    // ------------------------------------------------------------------
    // Viscous shell pressure jump (shell.kappa_s_jump, see Parse).
    // ------------------------------------------------------------------
    const bool use_vjump = (shell_kappa_s != 0.0 || shell_mu_s != 0.0);
    const bool use_jump  = apply_surface_tension;
    std::unique_ptr<amrex::MultiFab> jump_mf;
    if (use_jump)
    {
        // Velocity from M/rho needs current ghosts on every face (mirror,
        // RADIAL; an NSCBC face gets a zero-gradient copy that STEP 5 replaces).
        // The eta ghosts were filled at the top of FillGhost4BC.
        FillBoundariesWithBC(lev, relax_time, momentum_bc, { momentum_mf[lev].get() });
        FillBoundariesWithBC(lev, relax_time, density_bc,  { density_mf[lev].get() });
        // Density is a derived field, so its coarse-fine ghosts are never FillPatch'd (momentum's are, in
        // FillGhost4BC); v = M/rho on the grown box below read stale density there (after a restart:
        // uninitialised memory).  Fill them from the coarse level like momentum.
        if (lev > 0) FillPatch(lev, relax_time, density_mf, *density_mf[lev], *density_bc, 0);
        const amrex::BoxArray &ba = eta_mf[lev]->boxArray();
        const amrex::DistributionMapping &dm = eta_mf[lev]->DistributionMap();
        amrex::MultiFab vt(ba, dm, AMREX_SPACEDIM, 1), nt(ba, dm, AMREX_SPACEDIM, 1);
        jump_mf = std::make_unique<amrex::MultiFab>(ba, dm, 1, 0);
        jump_mf->setVal(0.0);
        const Set::Scalar *DXj = geom[lev].CellSize();
        const Set::Scalar sm_j = small;
        for (amrex::MFIter mfi(vt, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box gb = mfi.growntilebox(1);
            auto M   = momentum_mf[lev]->const_array(mfi);
            auto rho = density_mf[lev]->const_array(mfi);
            auto et  = eta_mf[lev]->const_array(mfi);
            auto v   = vt.array(mfi);
            auto n   = nt.array(mfi);
            amrex::ParallelFor(gb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar rc = std::max(rho(i, j, k), sm_j);
                for (int d = 0; d < AMREX_SPACEDIM; ++d) v(i, j, k, d) = M(i, j, k, d) / rc;
                const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXj);
                const Set::Scalar gem = ge.lpNorm<2>();
                for (int d = 0; d < AMREX_SPACEDIM; ++d) n(i, j, k, d) = (gem > g_gem_min) ? ge(d) / gem : 0.0;
            });
        }
        const Set::Scalar kmm = use_vjump ? shell_kappa_s - shell_mu_s : 0.0;
        const Set::Scalar mus = use_vjump ? shell_mu_s : 0.0;
        const Set::Scalar cut_j = cutoff;
        const int marm_j = marmottant;
        const Set::Scalar sig_j = sigma, chi_j = marmottant_chi, Gb_j = marmottant_Gamma_buck;
        const Set::Scalar sbrk_j = marmottant_sigma_break, sigw_j = sigma;
        const Set::Scalar tfl_jv = shell_tension_floor;
        const bool tfl_j = tfl_jv > -1.0e299;
        const Set::Scalar alpha_j = shell_kappa_s_alpha;
        // kappa from the capillary tensor (same as the force)
        amrex::MultiFab Kt_j(ba, dm, 1, 0);
        TensorCurvature(lev, Kt_j, relax_time);
        // shell.slip = 7: Gamma = m / |grad psi| (relaxation conserves the phase masses, so psi is fixed)
        const int slip7j = (shell_slip == 7 || shell_slip == 11 || shell_slip == 12 || shell_slip == 15) && marmottant;
        amrex::MultiFab Gp7j;
        if (slip7j) ShellPsiGrad(lev, relax_time, Gp7j);
        const Set::Scalar eps_j = epsilon, cut_rj = std::max(cutoff, 1.0e-12);   // flux-tube map (LiquidInterfaceRate), as the force
        // ------------------------------------------------------------------------------------------
        // VISCOUS RELAXATION JUMP BY PATH INTEGRATION ALONG THE NORMAL (default; visc.relax_cell_local = 0)
        //
        // This block INTEGRATES ALONG THE NORMAL (the grad-eta direction) WITHIN THE BAND.
        // Why: the sub-cycler applies the viscous jump as the body force F = -J_F grad(eta) (J_F = keff r kappa
        // per cell).  The relaxation splits the mixture pressure as p_gas - p_liq = J_R per cell.  In a
        // quasi-static band, d p_gas/dn = (J_R - J_F) d eta/dn + eta dJ_R/dn, so the band gas stays at the core
        // gas pressure only if
        //         d(eta J_R) = J_F d eta   along the grad-eta path,   i.e.  eta J_R = int J_F d eta.
        // A per-cell J_R = J_F (the old form, kept behind visc.relax_cell_local = 1 for a future cell-local
        // approach) violates this wherever J_F varies across the band: the band gas then carries an
        // Rdot-phased pressure, a reversible spring in series with the shell dashpot (measured: it returns
        // ~3e-12 J per rebound at 160 kPa, the size of the whole shell dissipation).  A global least-squares
        // potential (tested 2026-11-14) fails off a sphere because J_F grad(eta) is not curl-free (J_F varies
        // along the surface) and Phi/eta amplifies the rotational residual in near-gas cells.  Integrating
        // along each cell's own upwind grad-eta path has no global solve, so no residual reaches the core.
        //
        // Scheme (upwind, eikonal-type, cells with cut < eta < 1 - cut only; nothing outside the band is
        // touched; the force is unchanged):
        //   n = grad eta/|grad eta|; for each direction d the upwind neighbour is the one against n_d; it is
        //   used if it is in the band (or pure gas, eta <= cut), on the level (valid or exchanged ghost) and has
        //   LOWER eta.  Phi_i = sum_d w_d [Phi_nb + 0.5 (J_F,i + J_F,nb)(eta_i - eta_nb)], w_d = |n_d| / sum|n_d|
        //   over the used directions.
        //   Path start (deterministic): a cell with no usable upwind BAND neighbour (its lower neighbours are pure
        //   gas, or it is a local eta minimum, or |grad eta| < gem_min, e.g. at a saddle where two bands meet)
        //   starts with Phi = 0: the integral starts there, nothing is carried in.  At the gas edge eta_start ~ cutoff,
        //   so this differs from the exact int_0 by O(cutoff J_F).  A local minimum INSIDE the band or the liquid tail
        //   (a trace-gas pocket, eta non-monotone) is a gas region not connected to pure gas; it gets no jump carried
        //   from elsewhere.  (The earlier start value Phi = eta J_F injected the tail's flux-tube-amplified J_F, up to
        //   -150 kPa at eta ~ 0.99, carried it downstream, expanded the trace gas to rho_g 0.03 vs the core 0.53 and
        //   collapsed the conduction dt: 160 kPa slab, 3.07 us.)
        //   Sweeps with ghost exchange until no Phi changes (exact: the dependency graph follows increasing eta).
        //   J_R,visc = Phi/eta in relaxation band cells.  shell.mu_s != 0 keeps the per-cell form.
        // ------------------------------------------------------------------------------------------
        const int rpath = (!visc_relax_cell_local && kmm != 0.0 && mus == 0.0);
        amrex::MultiFab Pth;
        amrex::MultiFab Gfj;   // shell.foot_rate: {r, Kt} at the foot point, same as the force
        const int footj = shell_foot_rate && rpath;
        if (footj) { amrex::MultiFab Lr; FootRateSource(*eta_mf[lev], vt, Kt_j, DXj, Lr); FootExtend(lev, Lr, Gfj); }
        if (rpath)
        {
            const Set::Scalar cu = cut_j;
            amrex::MultiFab JFp(ba, dm, 1, 1), Ip(ba, dm, 1, 1);
            JFp.setVal(0.0); Ip.setVal(0.0);
            Pth.define(ba, dm, 1, 1); Pth.setVal(0.0);
            for (amrex::MFIter mfi(JFp, false); mfi.isValid(); ++mfi)
            {
                const amrex::Box &bx = mfi.validbox();
                auto et = eta_mf[lev]->const_array(mfi);
                auto v  = vt.const_array(mfi);
                auto ktj = Kt_j.const_array(mfi);
                auto jf = JFp.array(mfi);
                auto in = Ip.array(mfi);
                auto gfj = footj ? Gfj.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    in(i, j, k) = 1.0;
                    const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXj);
                    const Set::Scalar gem = ge.lpNorm<2>();
                    if (gem < g_gem_min) return;
                    const Set::Vector nh = ge / gem;
                    Set::Matrix gu = Set::Matrix::Zero();
                    for (int d = 0; d < AMREX_SPACEDIM; ++d) gu.col(d) = Numeric::Gradient(v, i, j, k, d, DXj);
                    const Set::Scalar dsf = gu.trace() - nh.dot(gu * nh);
                    const Set::Vector uu(AMREX_D_DECL(v(i, j, k, 0), v(i, j, k, 1), v(i, j, k, 2)));
                    const bool ft = footj && gfj(i, j, k, 0) == gfj(i, j, k, 0);
                    const Set::Scalar ds = ft ? gfj(i, j, k, 0) : LiquidInterfaceRate(et, i, j, k, DXj, eps_j, cut_rj, gu, uu, dsf);
                    jf(i, j, k) = kmm * ShellViscFactor(ds, alpha_j) * ds * (ft ? gfj(i, j, k, 1) : ktj(i, j, k));   // same J_F as the force
                });
            }
            auto fillp = [&](amrex::MultiFab &F) {
                if (lev == 0) FillBoundariesWithBC(lev, relax_time, eta_bc, { &F });
                else { F.FillBoundary(geom[lev].periodicity()); if (radial_bc) RadialGhostFill(lev, F, false); }
            };
            fillp(JFp); fillp(Ip);
            // Ghost cells outside the physical domain are not path neighbours.  A mirror or zero-gradient ghost is the
            // cell itself (equal eta, never upwind anyway); a RADIAL ghost is INTERPOLATED from the axis-row profile, so
            // using it as an upwind neighbour closes a cycle (cell -> transverse neighbour -> radial ghost -> axis row
            // -> cell) and the sweeps never settle (160 kPa slab, 3.07 us).  Only cells that exist on the level (valid,
            // exchanged or periodic ghosts) take part, so every path is strictly eta-increasing.
            for (amrex::MFIter mfi(Ip, false); mfi.isValid(); ++mfi)
            {
                auto in = Ip.array(mfi);
                const amrex::Box dom = geom[lev].Domain();
                amrex::LoopOnCpu(mfi.growntilebox(1), [&](int i, int j, int k) {
                    const int ijk[3] = {i, j, k};
                    for (int d = 0; d < AMREX_SPACEDIM; ++d)
                        if (!geom[lev].isPeriodic(d) && (ijk[d] < dom.smallEnd(d) || ijk[d] > dom.bigEnd(d))) in(i, j, k) = 0.0; });
            }
            // initial: Phi = 0 everywhere; sweeps then propagate along the paths
            for (amrex::MFIter mfi(Pth, false); mfi.isValid(); ++mfi)
            {
                auto et = eta_mf[lev]->const_array(mfi); auto jf = JFp.const_array(mfi); auto ph = Pth.array(mfi);
                amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
                    const Set::Scalar ea = et(i, j, k);
                    ph(i, j, k) = 0.0; });
            }
            int sweeps = 0; Set::Scalar nchg = 1.0;
            for (; sweeps < 100000 && nchg > 0.0; ++sweeps)
            {
                fillp(Pth);
                nchg = 0.0;
                for (amrex::MFIter mfi(Pth, false); mfi.isValid(); ++mfi)
                {
                    auto et = eta_mf[lev]->const_array(mfi); auto jf = JFp.const_array(mfi);
                    auto in = Ip.const_array(mfi); auto ph = Pth.array(mfi);
                    amrex::Box bx = mfi.validbox();
                    amrex::FArrayBox newf(bx, 1); auto nw = newf.array();
                    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                        const Set::Scalar ea = et(i, j, k);
                        nw(i, j, k) = ph(i, j, k);
                        if (!(ea > cu && ea < 1.0 - cu)) return;
                        const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXj);
                        const Set::Scalar gem = ge.lpNorm<2>();
                        Set::Scalar acc = 0.0, wsum = 0.0;
                        if (gem >= g_gem_min)
                            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                            {
                                const Set::Scalar nd = ge(d) / gem;
                                if (nd == 0.0) continue;
                                const int sg = (nd > 0.0) ? -1 : 1;           // upwind = against n_d
                                const int ib = i + sg * (d == 0), jb = j + sg * (d == 1), kb = k + sg * (d == 2);
                                if (in(ib, jb, kb) < 0.5) continue;           // off the level
                                const Set::Scalar eb = et(ib, jb, kb);
                                if (!(eb < ea) || eb <= cu) continue;          // need a lower-eta BAND neighbour
                                const Set::Scalar w = std::abs(nd);
                                acc += w * (ph(ib, jb, kb) + 0.5 * (jf(i, j, k) + jf(ib, jb, kb)) * (ea - eb));
                                wsum += w;
                            }
                        const Set::Scalar pn = (wsum > 0.0) ? acc / wsum : 0.0;   // path start: Phi = 0
                        nw(i, j, k) = pn;
                    });
                    amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
                        if (nw(i, j, k) != ph(i, j, k)) { nchg += 1.0; ph(i, j, k) = nw(i, j, k); } });
                }
                amrex::ParallelDescriptor::ReduceRealSum(nchg);
            }
            if (sweeps >= 100000) Util::Warning(INFO, "relaxation path integral: sweeps did not settle (lev ", lev, ")");
            rp_calls += 1; rp_sweeps += sweeps;
            if (visc_relax_path_log > 0 && rp_calls % visc_relax_path_log == 0)
            {
                Set::Scalar jrmax = 0.0, jrmax_g = 0.0;   // all band cells / near-gas band cells (eta < 1e-2)
                for (amrex::MFIter mfi(Pth, false); mfi.isValid(); ++mfi)
                {
                    auto et = eta_mf[lev]->const_array(mfi); auto ph = Pth.const_array(mfi);
                    amrex::LoopOnCpu(mfi.validbox(), [&](int i, int j, int k) {
                        const Set::Scalar ea = et(i, j, k);
                        if (ea > cu && ea < 1.0 - cu)
                        {
                            const Set::Scalar a = std::abs(ph(i, j, k) / ea);
                            jrmax = std::max(jrmax, a); if (ea < 1.0e-2) jrmax_g = std::max(jrmax_g, a);
                        }
                    });
                }
                amrex::ParallelDescriptor::ReduceRealMax(jrmax);
                amrex::ParallelDescriptor::ReduceRealMax(jrmax_g);
                const Set::Scalar jfn = JFp.norm0(0, 0);   // collective: outside the IOProcessor branch
                if (amrex::ParallelDescriptor::IOProcessor())
                    Util::Message(INFO, "relax path: t=", relax_time, " sweeps ", sweeps, " (mean ", (double)rp_sweeps / (double)rp_calls,
                                  ") max|J_R| ", jrmax, " (eta<.01: ", jrmax_g, ") max|J_F| ", jfn);
            }
        }
        for (amrex::MFIter mfi(*jump_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bxj = mfi.validbox();
            auto et = eta_mf[lev]->const_array(mfi);
            auto v  = vt.const_array(mfi);
            auto n  = nt.const_array(mfi);
            auto J  = jump_mf->array(mfi);
            auto shl = shell_mf[lev]->const_array(mfi);
            auto gpj = slip7j ? Gp7j.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            auto ktj = Kt_j.const_array(mfi);
            auto pth = rpath ? Pth.const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            amrex::ParallelFor(bxj, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar e = et(i, j, k);
                if (!(e > cut_j && e < 1.0 - cut_j)) return;
                Set::Vector nh;
                for (int d = 0; d < AMREX_SPACEDIM; ++d) nh(d) = n(i, j, k, d);
                if (nh.lpNorm<2>() < 0.5) return;
                // gu(a,b) = d_a u_b
                Set::Matrix gu = Set::Matrix::Zero();
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                    gu.col(d) = Numeric::Gradient(v, i, j, k, d, DXj);
                const Set::Matrix P = Set::Matrix::Identity() - nh * nh.transpose();
                const Set::Scalar ds = gu.trace() - nh.dot(gu * nh);
                const Set::Scalar fvis_j = ShellViscFactor(ds, alpha_j);
                Set::Matrix Ts = (kmm * fvis_j * ds) * P;
                if (mus != 0.0) Ts += (2.0 * mus) * (P * (0.5 * (gu + gu.transpose())) * P);
                {
                    // static part: same sigma_eff as the momentum force (Gamma from the shell field)
                    const Set::Scalar G = marm_j ? (slip7j ? ((gpj(i, j, k) > 0.0) ? shl(i, j, k) / gpj(i, j, k) : 1.0) : shl(i, j, k)) : 0.0;
                    const Set::Scalar se_j = SigmaEffFromGamma(G, 1.0, marm_j, sig_j, chi_j, Gb_j, sbrk_j, sigw_j);
                    Ts += se_j * P;
                    // shell.tension_floor: raise the isotropic tension to the floor
                    if (tfl_j && kmm * fvis_j * ds + se_j < tfl_jv) Ts += (tfl_jv - kmm * fvis_j * ds - se_j) * P;
                }
                // p_gas - p_liq = T_s : grad(n) with grad(n) -> (kappa/(D-1)) P:
                //   J = kappa tr(T_s)/(D-1),  kappa = tensor curvature
                J(i, j, k) = ktj(i, j, k) * Ts.trace() / (AMREX_SPACEDIM - 1);
                // viscous part from the band path integral (see above); static part unchanged
                if (rpath)
                    J(i, j, k) = ktj(i, j, k) * (Ts.trace() / (AMREX_SPACEDIM - 1) - kmm * fvis_j * ds) + pth(i, j, k) / e;
            });
        }
    }

    // relax.heat_diag: pre-relaxation state {alpha_gas, p_liq_pre, p_gas_pre} (output only)
    amrex::MultiFab rh_pre;
    if (relax_heat_diag)
    {
        // target jump bookkeeping (J = 0 outside the band / with the jump off)
        for (amrex::MFIter mfi(*relax_J_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto Jd = relax_J_mf[lev]->array(mfi);
            amrex::Array4<const Set::Scalar> Jc = use_jump ? jump_mf->const_array(mfi) : amrex::Array4<const Set::Scalar>{};
            const bool uj = use_jump;
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar Jn = uj ? Jc(i, j, k) : 0.0;
                if (!(Jn == Jn)) Jn = 0.0;
                const Set::Scalar dJ = Jn - Jd(i, j, k, 0);
                Jd(i, j, k, 1) = Jd(i, j, k, 0);
                Jd(i, j, k, 0) = Jn;
                Jd(i, j, k, 2) += std::abs(dJ);
                Jd(i, j, k, 3) += dJ * dJ;
            });
        }
        rh_pre.define(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 3, 0);
        for (amrex::MFIter mfi(rh_pre, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto E0_ = energy0_mf[lev]->const_array(mfi);
            auto E1_ = energy1_mf[lev]->const_array(mfi);
            auto q   = rh_pre.array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
                const Set::Scalar a2 = 1.0 - a1;
                q(i, j, k, 0) = a2;
                q(i, j, k, 1) = std::max((gam0 - 1.0) * E0_(i, j, k) / a1 - gam0 * pi0_, -pi0_ + DIV_FLOOR);
                q(i, j, k, 2) = std::max((gam1 - 1.0) * E1_(i, j, k) / a2 - gam1 * pi1_, -pi1_ + DIV_FLOOR);
            });
        }
    }

    // relax.blake: flag isolated gas pockets (eta minimum along the eta normal, eta > 1/2) from the
    // pre-relaxation eta, so the main kernel below never reads neighbours it is modifying.
    const int blake = relax_blake;
    std::unique_ptr<amrex::MultiFab> nuc_mf;
    if (blake)
    {
        nuc_mf = std::make_unique<amrex::MultiFab>(eta_mf[lev]->boxArray(), eta_mf[lev]->DistributionMap(), 1, 0);
        const Set::Scalar *DXb = geom[lev].CellSize();
        for (amrex::MFIter mfi(*nuc_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bxb = mfi.validbox();
            auto et = eta_mf[lev]->const_array(mfi);
            auto nf = nuc_mf->array(mfi);
            amrex::ParallelFor(bxb, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                nf(i, j, k) = 0.0;
                const Set::Scalar ec = et(i, j, k);
                if (!(ec > 0.5 && ec < 1.0)) return;
                const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DXb);
                const Set::Scalar g2 = ge.squaredNorm();
                Set::Scalar up = 0.0, um = 0.0;   // eta rise to each side along the normal (n_d^2 weights)
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    const Set::Scalar w = (g2 > 0.0) ? ge(d) * ge(d) / g2 : 1.0 / AMREX_SPACEDIM;
                    up += w * (et(i + di, j + dj, k + dk) - ec);
                    um += w * (et(i - di, j - dj, k - dk) - ec);
                }
                if (up > 0.0 && um > 0.0) nf(i, j, k) = AMREX_D_TERM(DXb[0], * DXb[1], * DXb[2]);   // cell volume
            });
        }
    }
    const Set::Scalar sig_blake = sigma;

    for (amrex::MFIter mfi(*eta_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        amrex::Array4<const Set::Scalar> Jarr = use_jump ? jump_mf->const_array(mfi)
                                                         : amrex::Array4<const Set::Scalar>{};

        amrex::Array4<const Set::Scalar> nucA = blake ? nuc_mf->const_array(mfi) : amrex::Array4<const Set::Scalar>{};
        auto eta   = eta_mf[lev]->array(mfi);
        auto arh0  = rho_eta0_mf[lev]->array(mfi);
        auto arh1  = rho_eta1_mf[lev]->array(mfi);
        auto M_    = momentum_mf[lev]->array(mfi);
        auto E_    = energy_per_vol_mf[lev]->array(mfi);
        auto E0_   = energy0_mf[lev]->array(mfi);
        auto E1_   = energy1_mf[lev]->array(mfi);
        // ELASTIC SOLID: the conserved rho E carries the elastic energy W (Favrie
        // 2009 eq. 42), which is debited before the hydrodynamic EOS inversion.
        // The cobasis is unchanged by the relaxation (eq. 61d).
        const bool el_on = elastic.on != 0;
        amrex::Array4<const Set::Scalar> eb_r;
        if (el_on) eb_r = elastic.ebasis_mf[lev]->const_array(mfi);

        // EMBEDDED SOLID indicator (empty Array4 when feature is off).
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);

        // Diagnostic array (only used if diag_on).
        amrex::Array4<Set::Scalar> diag_arr = diag_on
            ? (*diag_mf)[mfi].array()
            : amrex::Array4<Set::Scalar>{};

        const int rhd = relax_heat_diag;
        amrex::Array4<Set::Scalar> rsp = rhd ? relax_split_mf[lev]->array(mfi) : amrex::Array4<Set::Scalar>{};
        amrex::Array4<const Set::Scalar> sp_r;
        if (spe.on) sp_r = species.mf[lev]->const_array(mfi);
        const bool el_comp = elastic.on && elastic.comp_on;
        auto cf_r = cfun_mf[lev]->const_array(mfi);     // alpha_ref of the compaction model
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            // Skip solid cells: pressure relaxation must not touch the
            // frozen solid state (its per-phase energies are prescribed,
            // not in thermodynamic equilibrium with the alpha there).
            if (embedded.apply && embedded.isSolid(phisol(i, j, k))) return;
            Set::Scalar gam0 = gam0_c, gam1 = gam1_c, pi0_ = pi0_c, pi1_ = pi1_c;
            Set::Scalar cvT0 = cv0_c, cvT1 = cv1_c;      // phase c_v (trace p-T and joint p-T)
            if (spe.on) { Set::Scalar cvd = cv0_c; const Set::Scalar af_ = std::min(std::max(eta(i, j, k), 1.0e-12), 1.0 - 1.0e-12); spe.Mix(sp_r, i, j, k, 0, gam0, cvd, E0_(i, j, k) / std::max(arh0(i, j, k), 1.0e-300), &pi0_, arh0(i, j, k) / af_); cvT0 = cvd; cvd = cv1_c; spe.Mix(sp_r, i, j, k, 1, gam1, cvd, E1_(i, j, k) / std::max(arh1(i, j, k), 1.0e-300), &pi1_, arh1(i, j, k) / (1.0 - af_)); cvT1 = cvd; }
            if (el_comp) pi0_ += elastic.compaction(elastic.matAt(sp_r, i, j, k), std::min(std::max(eta(i, j, k), 0.0), 1.0), arh0(i, j, k), cf_r(i, j, k)).Pc / gam0;
            // elastic energy W in rho E (Favrie 2009 eq. 42), debited before every hydrodynamic EOS inversion (0 for fluids)
            auto w_elastic = [&]() -> Set::Scalar {
                return el_on ? elastic.energy(eb_r, i, j, k, std::max(arh0(i, j, k), 0.0), std::max(arh1(i, j, k), 0.0), sp_r,
                                              std::min(std::max(eta(i, j, k), 0.0), 1.0), cf_r(i, j, k)) : 0.0;
            };
            
            Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
            Set::Scalar a2 = 1.0 - a1;

            // Conservative partial densities
            Set::Scalar arh0_loc = std::max(arh0(i, j, k), 0.0);
            Set::Scalar arh1_loc = std::max(arh1(i, j, k), 0.0);

            // Pure-phase densities
            Set::Scalar rho0_pre = arh0_loc / std::max(a1, DIV_FLOOR);
            Set::Scalar rho1_pre = arh1_loc / std::max(a2, DIV_FLOOR);

            // Pre-relax per-phase pressures (Sau09 eq. II.3 inverted)
            Set::Scalar p0_pre = (gam0 - 1.0) * E0_(i, j, k) / std::max(a1, DIV_FLOOR) - gam0 * pi0_;
            Set::Scalar p1_pre = (gam1 - 1.0) * E1_(i, j, k) / std::max(a2, DIV_FLOOR) - gam1 * pi1_;
            // SG positivity floor (Sau09 Sec.3): p_k + pi_k > 0.
            p0_pre = std::max(p0_pre, -pi0_ + DIV_FLOOR);
            p1_pre = std::max(p1_pre, -pi1_ + DIV_FLOOR);

            // -----------------------------------------------------------
            // TRACE-PHASE SLAVING (relax): same ill-conditioning as the
            // recovery paths -- p_pre = (gam-1)E_k/alpha - gam*pi turns a
            // 4e-5 relative E/alpha inconsistency into a 100% pressure
            // error at gam*pi/p ~ 2.4e4.  Below the guard cutoff those
            // cells skip the Newton entirely, but in the band tail ABOVE
            // cutoff (eta ~ 1e-4..1e-2) the Newton ingests the mispriced
            // trace p_pre and MOVES alpha off its garbage relaxation
            // volume -- a secular interface-erosion loop (the residual
            // exponential drain after the recovery slaving landed).
            // Slave the trace phase's p_pre to the dominant phase's:
            // with p0_pre == p1_pre the volume constraint is satisfied
            // at p_pre exactly (v_k = v_k^0 -> f = a1+a2-1 = 0), so the
            // Newton converges immediately and alpha is NOT moved --
            // which is the correct physical outcome for a cell whose
            // trace phase has no independent pressure to relax against.
            // -----------------------------------------------------------

            // -----------------------------------------------------------
            // PURE-CELL GUARD: relaxation is an interface operation
            // (Sau09/Sch20 relax alpha between phases -- meaningless when
            // one phase is below the eta cutoff).  In near-pure cells the
            // trace phase's E_k is alpha-amplified noise: p_pre =
            // (gam-1)E_k/alpha - gam pi turns roundoff into MPa at
            // alpha ~ 1e-12, and when the trace pressure floors at -pi
            // the volume constraint f(p) goes CONSTANT (plateau at
            // |f| ~ 1/gamma_gas: no root exists, Newton and bisection
            // both spin to max_iter).  The floored state is NOT confined
            // to sub-cutoff cells -- band-TAIL cells (eta ~ 1e-4..1e-2)
            // hit it through advection over/undershoot of the steep
            // trace E_k profile -- so the guard fires on EITHER
            // (a) eta beyond cutoff, or (b) either phase's pre-relax
            // pressure sitting at its positivity floor (that phase's
            // v(p) has lost EOS meaning: the solve is garbage-in).
            // Skip the solve: freeze alpha AND eta (no eta stomp to the
            // alpha_floor clamp), and do only the energy-consistent
            // reinit so both per-phase energies return to mechanical
            // equilibrium with the conserved mixture state
            // (self-healing: next step's p_pre is then consistent and
            // the cell rejoins the Newton path).
            // -----------------------------------------------------------
            const bool p0_floored = (p0_pre <= -pi0_ + std::max(2.0 * DIV_FLOOR, 1.0e-9 * pi0_));
            const bool p1_floored = (p1_pre <= -pi1_ + std::max(2.0 * DIV_FLOOR, 1.0e-9 * pi1_));
            const bool single_phase = opt9 ? (arh0_loc <= 0.0 || arh1_loc <= 0.0)
                                           : (eta(i, j, k) < cutoff_loc || eta(i, j, k) > 1.0 - cutoff_loc);
            // relax.trace_pT (see Parse): liquid beyond the cutoff with gas mass -> p-T equilibrium, Dj = 0.
            bool trace_eq = false;   // trace_pT = 2: this trace cell takes the ordinary closure below
            // The trace phase is the SOFTER one (smaller pi; gas in liquid), below relax.trace_alpha by volume:
            // phase 1 soft (flames2 decks): eta > 1 - trace_alpha; phase 0 soft (DropletDetonation): eta < trace_alpha.
            const bool soft1 = (pi1_ < pi0_);
            const bool trace_cell = (pi0_ != pi1_) && (soft1 ? (eta(i, j, k) > 1.0 - trace_a) : (eta(i, j, k) < trace_a));
            if (tracePT && trace_cell && arh0_loc > 0.0 && arh1_loc > 0.0)
            {
                const Set::Scalar rhoT = arh0_loc + arh1_loc;
                const Set::Scalar keT  = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                           + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                           + M_(i, j, k, 2) * M_(i, j, k, 2))) / rhoT;
                const Set::Scalar rheT = E_(i, j, k) - keT - w_elastic();
                const Set::Scalar a1f = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar p_frz = Solver::EOS::EOS::ReinitMixturePressure(rheT, a1f, 1.0 - a1f,
                                                                                 gam0, pi0_, gam1, pi1_, small_loc);
                if (tracePT == 2 && p_frz >= 0.0 && rheT > 0.0 && !p0_floored && !p1_floored) trace_eq = true;
                if (tracePT == 3)
                {
                    // Frozen alpha AND frozen phase densities (the guard's premise: no volume dynamics for the
                    // trace), but the trace gas shares the liquid temperature.  At fixed alpha, rho_k:
                    //   E0 = arh0 cv0 T + a1 pi0,  E1 = arh1 cv1 T,  E0 + E1 = rho e  ->  T in closed form.
                    // Uses only masses and rho e (never the noisy trace E1).
                    // (Written for either trace phase: the trace energy is set, the dominant phase takes the rest.)
                    const Set::Scalar Cm = arh0_loc * cvT0 + arh1_loc * cvT1;
                    const Set::Scalar TT = (rheT - a1f * pi0_ - (1.0 - a1f) * pi1_) / Cm;
                    if (TT > 0.0 && rheT > 0.0)
                    {
                        if (soft1)
                        {
                            E1_(i, j, k) = arh1_loc * cvT1 * TT + (1.0 - a1f) * pi1_;
                            E0_(i, j, k) = rheT - E1_(i, j, k);
                        }
                        else
                        {
                            E0_(i, j, k) = arh0_loc * cvT0 * TT + a1f * pi0_;
                            E1_(i, j, k) = rheT - E0_(i, j, k);
                        }
                        if (rhd) { rsp(i, j, k, 4) += 1.0; }
                        if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = 0.0; diag_arr(i, j, k, 2) = 0.0; }
                        return;
                    }
                }
                if (tracePT == 1 && p_frz >= 0.0 && rheT > 0.0)
                {
                    const Set::Scalar vT = 1.0 / rhoT, eT = rheT / rhoT;
                    const Set::Scalar Y0 = arh0_loc / rhoT, Y1 = arh1_loc / rhoT;
                    const Set::Scalar A0 = Y0 * (gam0 - 1.0) * cvT0, A1 = Y1 * (gam1 - 1.0) * cvT1;
                    const Set::Scalar CT = Y0 * cvT0 + Y1 * cvT1;
                    const Set::Scalar B0 = A0 * (eT - vT * pi0_), B1 = A1 * (eT - vT * pi1_);
                    const Set::Scalar bb = pi1_ - pi0_;
                    const Set::Scalar qa = vT * CT, qb = vT * CT * bb - B0 - B1, qc = -B0 * bb;
                    const Set::Scalar disc = qb * qb - 4.0 * qa * qc;
                    if (disc >= 0.0)
                    {
                        const Set::Scalar sq = std::sqrt(disc);
                        const Set::Scalar qq = -0.5 * (qb + (qb >= 0.0 ? sq : -sq));
                        const Set::Scalar r1 = qq / qa, r2 = (qq != 0.0) ? qc / qq : r1;
                        Set::Scalar aa = -1.0;
                        if (r1 > 0.0 && r1 + bb > 0.0) aa = r1;
                        if (r2 > 0.0 && r2 + bb > 0.0 && r2 > aa) aa = r2;
                        if (aa > 0.0)
                        {
                            const Set::Scalar ag = aa + bb;
                            const Set::Scalar TT = vT / (A0 / aa + A1 / ag);
                            const Set::Scalar v0T = (gam0 - 1.0) * cvT0 * TT / aa, v1T = (gam1 - 1.0) * cvT1 * TT / ag;
                            const Set::Scalar al0 = arh0_loc * v0T, al1 = arh1_loc * v1T;
                            if (TT > 0.0 && al0 > alpha_floor && al1 > 0.0 && std::abs(al0 + al1 - 1.0) < 1.0e-8)
                            {
                                eta(i, j, k) = al0 / (al0 + al1);
                                E0_(i, j, k) = arh0_loc * (cvT0 * TT + pi0_ * v0T);
                                E1_(i, j, k) = rheT - E0_(i, j, k);
                                if (rhd) rsp(i, j, k, 4) += 1.0;
                                if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = 0.0; diag_arr(i, j, k, 2) = 0.0; }
                                return;
                            }
                        }
                    }
                }
            }
            // relax.blake (see Parse): sub-grid nucleus below the Blake threshold keeps its volume.
            if (blake && nucA(i, j, k) > 0.0 && arh0_loc > 0.0 && arh1_loc > 0.0 && !p1_floored)
            {
                const Set::Scalar ag = 1.0 - std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar rn = std::cbrt(3.0 * ag * nucA(i, j, k) / (4.0 * M_PI));
                const Set::Scalar pg = std::max(p1_pre, 0.0);
                const Set::Scalar Rc = std::sqrt(3.0 * pg * rn * rn * rn / (2.0 * sig_blake));
                const Set::Scalar pB = (Rc > 0.0) ? -4.0 * sig_blake / (3.0 * Rc) : -1.0e300;
                if (rn > 0.0 && p0_pre > pB)
                {
                    const Set::Scalar rhoB = arh0_loc + arh1_loc;
                    const Set::Scalar keB  = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                               + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                               + M_(i, j, k, 2) * M_(i, j, k, 2))) / rhoB;
                    const Set::Scalar rheB = E_(i, j, k) - keB - w_elastic();
                    const Set::Scalar TT = (rheB - (1.0 - ag) * pi0_) / (arh0_loc * cvT0 + arh1_loc * cvT1);
                    if (TT > 0.0 && rheB > 0.0)
                    {
                        E1_(i, j, k) = arh1_loc * cvT1 * TT;
                        E0_(i, j, k) = rheB - E1_(i, j, k);
                        if (rhd) { rsp(i, j, k, 4) += 1.0; }
                        if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = 0.0; diag_arr(i, j, k, 2) = 0.0; }
                        return;
                    }
                }
            }
            if (!trace_eq && (single_phase || p0_floored || p1_floored))
            {
                if (!p0_floored && !p1_floored && (eta(i, j, k) < cutoff_loc || eta(i, j, k) > 1.0 - cutoff_loc))
                {
                    // Trace phase t, dominant phase d.  Keep E_t; E_d = rho e - E_t.  Root in alpha_t of
                    //   (gam_t-1) E_t / alpha_t - gam_t pi_t  =  (gam_d-1) E_d / (1 - alpha_t) - gam_d pi_d
                    // (LHS falls, RHS rises with alpha_t: unique).  Bisection in log alpha_t.
                    const bool liqtrace = eta(i, j, k) < 0.5;
                    const Set::Scalar rho_t = arh0_loc + arh1_loc;
                    const Set::Scalar ke_t = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                                + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                                + M_(i, j, k, 2) * M_(i, j, k, 2))) / std::max(rho_t, small_loc);
                    const Set::Scalar rhoe_t = std::max(E_(i, j, k) - ke_t - w_elastic(), small_loc);
                    const Set::Scalar Et = liqtrace ? E0_(i, j, k) : E1_(i, j, k);
                    const Set::Scalar mt = liqtrace ? arh0_loc : arh1_loc;
                    const Set::Scalar gt = liqtrace ? gam0 : gam1, pit = liqtrace ? pi0_ : pi1_;
                    const Set::Scalar gd = liqtrace ? gam1 : gam0, pid = liqtrace ? pi1_ : pi0_;
                    const Set::Scalar Ed = rhoe_t - Et;
                    const Set::Scalar at0 = liqtrace ? std::min(std::max(eta(i, j, k), 0.0), 1.0) : 1.0 - std::min(std::max(eta(i, j, k), 0.0), 1.0);
                    // Well-posed only when the dominant phase is the SOFTER one (bulk modulus gam (p + pi)):
                    // then a change of the trace volume barely moves the common pressure.  A soft trace
                    // (gas) inside a stiff dominant phase (water) turns any noise in E_t into a pressure jump
                    // rho c^2 d(alpha) of the dominant phase (2026-11-02: fine slab pressure spikes, blow-up);
                    // that case keeps the original guard.
                    const Set::Scalar pdg = (gd - 1.0) * Ed / std::max(1.0 - at0, 1.0e-30) - gd * pid;
                    const bool soft_dom = gd * (pdg + pid) < gt * (pdg + pit);
                    if (soft_dom && Et > 0.0 && Ed > 0.0 && mt > 0.0 && at0 > 0.0)
                    {
                        auto ff = [=](Set::Scalar at) { return (gt - 1.0) * Et / at - gt * pit - ((gd - 1.0) * Ed / (1.0 - at) - gd * pid); };
                        // bracket: at most a factor 2 change per call (robust to noisy trace E_t), never above 1/2
                        Set::Scalar lo = 0.5 * at0, hi = std::min(2.0 * at0, 0.5);
                        Set::Scalar at = at0;
                        const Set::Scalar flo = ff(lo), fhi = ff(hi);
                        if (flo <= 0.0) at = lo;
                        else if (fhi >= 0.0) at = hi;
                        else
                        {
                            for (int it = 0; it < 60; ++it)
                            {
                                const Set::Scalar mid = std::sqrt(lo * hi);
                                if (ff(mid) > 0.0) lo = mid; else hi = mid;
                                if (hi - lo < 1.0e-12 * hi) break;
                            }
                            at = std::sqrt(lo * hi);
                        }
                        eta(i, j, k) = liqtrace ? at : 1.0 - at;
                        if (liqtrace) E1_(i, j, k) = Ed; else E0_(i, j, k) = Ed;
                        if (rhd) rsp(i, j, k, 4) += 1.0;
                        if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = 0.0; diag_arr(i, j, k, 2) = 0.0; }
                        return;
                    }
                }
                Set::Scalar rho_p  = arh0_loc + arh1_loc;
                Set::Scalar ke_p   = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                       + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                       + M_(i, j, k, 2) * M_(i, j, k, 2))) / std::max(rho_p, small_loc);
                Set::Scalar rhoe_p = std::max(E_(i, j, k) - ke_p - w_elastic(), small_loc);
                // Solve for p with the SAME alpha the energies are written
                // with, so E0 + E1 = rho_e holds identically.
                const Set::Scalar a1p = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2p = 1.0 - a1p;
                Set::Scalar p_pure = Solver::EOS::EOS::ReinitMixturePressure(rhoe_p, a1p, a2p,
                                                                            gam0, pi0_, gam1, pi1_, small_loc);
                // Pressure floor.  A cell beyond the cutoff is a single fluid
                // (the dominant phase) carrying a dynamically irrelevant trace:
                // floor at the DOMINANT phase's EOS limit -pi_d, so pure Tammann
                // liquid holds tension like bulk liquid.  The old floor
                // -min(pi0, pi1) is 0 whenever the other phase is an ideal gas,
                // which held every pure-liquid cell at p >= 0 (04 INCLINE: the
                // drive tension never reached the liquid; 2026-11-09).  The
                // trace phase sits at its own floor and the dominant phase takes
                // the rest of rho e, so E0 + E1 = rho e exactly.  Mixed cells
                // that reach here only because a pre-relax pressure is floored
                // keep the equal-pressure floor of both phases.
                const bool beyond_cut = (eta(i, j, k) < cutoff_loc || eta(i, j, k) > 1.0 - cutoff_loc);
                const bool dom0 = eta(i, j, k) >= 0.5;    // phase 0 (liquid) dominant
                const Set::Scalar pi_dom = dom0 ? pi0_ : pi1_;
                // Mixed cell here (a pre-relax pressure is floored): same per-phase rule as the
                // post-relax reinit -- if the equal-pressure state p_raw is below exactly one
                // phase's EOS limit, that phase sits at its floor and the other takes the rest
                // of rho e (liquid may go negative).  Otherwise equal pressures as before.
                const Set::Scalar p_raw = p_pure;
                const bool mout0 = (p_raw + pi0_ <= small_loc), mout1 = (p_raw + pi1_ <= small_loc);
                const bool mixed_split = !beyond_cut && (mout0 != mout1);
                // Elastic solid: a (near-)pure solid cell must carry hydrostatic tension -> dominant-phase floor
                // in every cell of this branch (DropletDetonation).
                p_pure = (beyond_cut || el_on) ? std::max(p_pure, -pi_dom + small_loc)
                                               : std::max(p_pure, -std::min(pi0_, pi1_) + small_loc);
                // Write E_k with the FROZEN raw eta (clamped only to [0,1]),
                // NOT the alpha_floor-clamped a1: eta stays frozen in this
                // branch, so the recovery divide E_k/eta only cancels exactly
                // when E_k was built with the same eta.  Using the 1e-12-
                // clamped a1 here re-creates the divide-floor mispricing in
                // the band tail (eta ~ 1e-14..1e-6) that this guard exists
                // to prevent.
                const Set::Scalar a1g = std::min(std::max(eta(i, j, k), 0.0), 1.0);
                const Set::Scalar a2g = 1.0 - a1g;
                const Set::Scalar E1g_old = E1_(i, j, k);
                if (beyond_cut)
                {
                    // opt8: liquid-dominant cell whose fixed-alpha common pressure is below the gas
                    // floor -> gas keeps its own (positive-pressure) energy, liquid takes the rest.
                    const Set::Scalar E0k8 = rhoe_p - E1g_old;
                    // opt8 = 2: the trace gas keeps its own energy in EVERY beyond-cutoff liquid cell (not only
                    // below its floor): forcing it to the liquid pressure ignores the Laplace jump and leaves a
                    // cold trace (17 K at 11 kPa, 2 kg/m3) that advects into the band.
                    const bool keep8 = opt8 && dom0 && (opt8 >= 2 || p_pure + pi1_ <= small_loc) && (E1g_old > 0.0)
                                       && ((gam0 - 1.0) * E0k8 / std::max(a1g, DIV_FLOOR) - gam0 * pi0_ + pi0_ > small_loc);
                    if (keep8)
                    {
                        E1_(i, j, k) = E1g_old;
                        E0_(i, j, k) = E0k8;
                    }
                    else if (dom0)
                    {
                        E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(std::max(p_pure, -pi1_ + small_loc), a2g, gam1, pi1_, small_loc);
                        E0_(i, j, k) = rhoe_p - E1_(i, j, k);
                    }
                    else
                    {
                        E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(std::max(p_pure, -pi0_ + small_loc), a1g, gam0, pi0_, small_loc);
                        E1_(i, j, k) = rhoe_p - E0_(i, j, k);
                    }
                }
                else if (mixed_split && mout1 && opt8 && E1g_old > 0.0
                         && ((gam0 - 1.0) * (rhoe_p - E1g_old) / std::max(a1g, DIV_FLOOR) - gam0 * pi0_ + pi0_ > small_loc))
                {
                    // opt8: mixed cell, common pressure below the gas floor -> gas keeps its energy
                    E0_(i, j, k) = rhoe_p - E1g_old;
                }
                else if (mixed_split && mout1)
                {
                    const Set::Scalar E1f = a2g * (-pi1_ + small_loc + gam1 * pi1_) / (gam1 - 1.0);
                    const Set::Scalar E0f = rhoe_p - E1f;
                    if ((gam0 - 1.0) * E0f / std::max(a1g, DIV_FLOOR) - gam0 * pi0_ + pi0_ > small_loc)
                    { E1_(i, j, k) = E1f; E0_(i, j, k) = E0f; }
                    else
                    {
                        E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a1g, gam0, pi0_, small_loc);
                        E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a2g, gam1, pi1_, small_loc);
                    }
                }
                else if (mixed_split && mout0)
                {
                    const Set::Scalar E0f = a1g * (-pi0_ + small_loc + gam0 * pi0_) / (gam0 - 1.0);
                    const Set::Scalar E1f = rhoe_p - E0f;
                    if ((gam1 - 1.0) * E1f / std::max(a2g, DIV_FLOOR) - gam1 * pi1_ + pi1_ > small_loc)
                    { E0_(i, j, k) = E0f; E1_(i, j, k) = E1f; }
                    else
                    {
                        E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a1g, gam0, pi0_, small_loc);
                        E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a2g, gam1, pi1_, small_loc);
                    }
                }
                else
                {
                    E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a1g, gam0, pi0_, small_loc);
                    E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_pure, a2g, gam1, pi1_, small_loc);
                }
                if (rhd) { rsp(i, j, k, 3) += E1_(i, j, k) - E1g_old; rsp(i, j, k, 4) += 1.0; }
                if (diag_on)
                {
                    diag_arr(i, j, k, 0) = 0.0;
                    diag_arr(i, j, k, 1) = 0.0;
                    diag_arr(i, j, k, 2) = 0.0;
                }
                return;
            }

            // -----------------------------------------------------------
            // Newton on f(p) = arh0*v0(p) + arh1*v1(p) - 1 = 0, with
            // bisection fallback when Newton stalls.
            //
            // SC form (Sch20 eq. 24+25):
            //   v_k(p) = v_k^0 * [p_k^0 + gam pi + (gam-1) p] / [gam (p + pi)]
            //
            // Monotonicity (proven in Pressure_Relaxation.tex):
            //   dv_k/dp = -v_k^0 (p_k^0+pi_k)/(gam_k (p+pi_k)^2) < 0
            // strictly, given SG positivity p_k^0 + pi_k > 0.  Hence
            // f(p) is monotone-decreasing and its root is UNIQUE on
            // (-min_k(pi_k), +infty).  Therefore bisection is bulletproof
            // for cells where Newton fails to converge -- which it does
            // when the Tammann liquid Jacobian arh0*dv0/dp ~ 4e-13
            // dwarfs the convergence test scale (problematic for
            // interface cells with conflicting per-phase pressures).
            // -----------------------------------------------------------
            // Phase-pressure jump: liquid (phase 0) at p, gas (phase 1) at p + Dj.
            Set::Scalar Dj = use_jump ? Jarr(i, j, k) : 0.0;
            if (!(Dj == Dj)) Dj = 0.0;
            // ---------------- thermal.exchange = 4: joint p-T relaxation ----------------
            // Unknowns p (liquid), gas at p + Dj, common T.  SG phases: v_k = (g_k-1) cv_k T/(p_k+pi_k),
            // e_k = cv_k T + pi_k v_k.  With Y_k mass fractions, v = 1/rho, e = rho e/rho,
            // A_k = Y_k (g_k-1) cv_k, C = sum Y_k cv_k:
            //     sum_k A_k (e - v pi_k)/(p_k + pi_k) = v C,   T = v / sum_k A_k/(p_k + pi_k).
            // With a = p + pi_0 and b = Dj + pi_1 - pi_0 (gas: p_1 + pi_1 = a + b) this is the quadratic
            //     v C a^2 + [v C b - A0 (e - v pi0) - A1 (e - v pi1)] a - A0 (e - v pi0) b = 0,
            // root with a > 0 and a + b > 0.  Mass, volume (alpha0 + alpha1 = 1) and energy exact.
            if (pT4 && a1 > amin_pT && a2 > amin_pT && arh0_loc > 0.0 && arh1_loc > 0.0)
            {
                const Set::Scalar rhoT = arh0_loc + arh1_loc;
                const Set::Scalar keT  = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                           + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                           + M_(i, j, k, 2) * M_(i, j, k, 2))) / rhoT;
                const Set::Scalar rheP4 = E_(i, j, k) - keT - w_elastic();
                const Set::Scalar vT = 1.0 / rhoT, eT = rheP4 / rhoT;
                const Set::Scalar Y0 = arh0_loc / rhoT, Y1 = arh1_loc / rhoT;
                const Set::Scalar A0 = Y0 * (gam0 - 1.0) * cvT0, A1 = Y1 * (gam1 - 1.0) * cvT1;
                const Set::Scalar CT = Y0 * cvT0 + Y1 * cvT1;
                const Set::Scalar B0 = A0 * (eT - vT * pi0_), B1 = A1 * (eT - vT * pi1_);
                const Set::Scalar bb = Dj + pi1_ - pi0_;
                const Set::Scalar qa = vT * CT, qb = vT * CT * bb - B0 - B1, qc = -B0 * bb;
                const Set::Scalar disc = qb * qb - 4.0 * qa * qc;
                if (disc >= 0.0)
                {
                    const Set::Scalar sq = std::sqrt(disc);
                    const Set::Scalar qq = -0.5 * (qb + (qb >= 0.0 ? sq : -sq));   // stable roots
                    const Set::Scalar r1 = qq / qa, r2 = (qq != 0.0) ? qc / qq : r1;
                    Set::Scalar aa = -1.0;
                    if (r1 > 0.0 && r1 + bb > 0.0) aa = r1;
                    if (r2 > 0.0 && r2 + bb > 0.0 && r2 > aa) aa = r2;           // largest admissible root
                    if (aa > 0.0)
                    {
                        const Set::Scalar ag = aa + bb;              // p_gas + pi_1
                        const Set::Scalar TT = vT / (A0 / aa + A1 / ag);
                        const Set::Scalar v0T = (gam0 - 1.0) * cvT0 * TT / aa, v1T = (gam1 - 1.0) * cvT1 * TT / ag;
                        const Set::Scalar al0 = arh0_loc * v0T, al1 = arh1_loc * v1T;
                        if (TT > 0.0 && al0 > alpha_floor && al1 > alpha_floor && std::abs(al0 + al1 - 1.0) < 1.0e-8)
                        {
                            eta(i, j, k) = al0 / (al0 + al1);
                            E0_(i, j, k) = arh0_loc * (cvT0 * TT + pi0_ * v0T);
                            E1_(i, j, k) = rheP4 - E0_(i, j, k);   // energy exact
                            if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = 0.0; diag_arr(i, j, k, 2) = 0.0; }
                            return;
                        }
                    }
                }
                // no admissible p-T state: fall through to the pressure relaxation
            }
            const Set::Scalar p_min = std::max(-pi0_, -pi1_ - Dj) + DIV_FLOOR;
            // Root bracket over the full EOS range of BOTH phases (liquid tension allowed).
            // The former extra limit "mixture p + a2 Dj > 0" existed only because the face
            // rebuild (Limiter::ToState) clamped the gas energy at p_mix < 0; ToState now
            // splits per phase (gas at its floor, liquid the rest), so the mixture may go
            // negative down to -alpha_l pi_l, as in the HLLC per-phase floors (2026-11-09).
            const Set::Scalar p_tfloor = p_min;

            // ---------------- closure: isentropic + mass-specific heat ----------------
            if (!sch20c && !(omega_r >= 0.0) && arh0_loc > 0.0 && arh1_loc > 0.0)
            {
                const Set::Scalar rhoM = arh0_loc + arh1_loc;
                const Set::Scalar keM  = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                           + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                           + M_(i, j, k, 2) * M_(i, j, k, 2))) / rhoM;
                const Set::Scalar rheM = E_(i, j, k) - keM - w_elastic();
                const Set::Scalar v00 = a1 / arh0_loc, v10 = a2 / arh1_loc;     // pre specific volumes
                const Set::Scalar c0 = p0_pre + pi0_, c1 = p1_pre + pi1_;       // > 0 (SG floor)
                // f(p) and the state at liquid pressure p (gas at p + Dj)
                auto mstate = [=](Set::Scalar pl, Set::Scalar &v0, Set::Scalar &v1, Set::Scalar &qq) -> Set::Scalar {
                    const Set::Scalar pg = pl + Dj;
                    const Set::Scalar v0s = v00 * std::pow(c0 / (pl + pi0_), 1.0 / gam0);
                    const Set::Scalar v1s = v10 * std::pow(c1 / (pg + pi1_), 1.0 / gam1);
                    const Set::Scalar e0s = (pl + gam0 * pi0_) * v0s / (gam0 - 1.0);
                    const Set::Scalar e1s = (pg + gam1 * pi1_) * v1s / (gam1 - 1.0);
                    qq = (rheM - arh0_loc * e0s - arh1_loc * e1s) / rhoM;
                    v0 = v0s + (gam0 - 1.0) * qq / (pl + gam0 * pi0_);
                    v1 = v1s + (gam1 - 1.0) * qq / (pg + gam1 * pi1_);
                    return arh0_loc * v0 + arh1_loc * v1 - 1.0;
                };
                Set::Scalar v0, v1, qq;
                // Same lower bracket as the fallback path (no root -> fall back to it).
                const Set::Scalar pbase = p_tfloor;
                Set::Scalar lo = pbase + 1.0e-9 * std::max(std::abs(pbase), 1.0);
                Set::Scalar hi = 1.0e4 * std::max(std::max(std::abs(p0_pre), std::abs(p1_pre)), 1.0e5);
                Set::Scalar flo = mstate(lo, v0, v1, qq), fhi = mstate(hi, v0, v1, qq);
                // With q < 0 a spurious root sits within ~1 Pa of the gas EOS pole (p_gas -> -pi_gas),
                // so f(lo) < 0.  Raise lo (x4 steps from the pole) until f > 0: the bracket
                // [lo, hi] then holds only the physical root.
                if (!(flo > 0.0) && fhi < 0.0)
                {
                    const Set::Scalar scl = std::max(std::max(std::abs(p0_pre), std::abs(p1_pre)), 1.0e5);
                    Set::Scalar dd = 1.0e-9 * scl;
                    for (int s = 0; s < 80 && pbase + dd < hi; ++s, dd *= 4.0)
                    {
                        const Set::Scalar ft = mstate(pbase + dd, v0, v1, qq);
                        if (ft > 0.0) { lo = pbase + dd; flo = ft; if (rhd) rsp(i, j, k, 9) += 1.0; break; }
                    }
                }
                bool ok = (flo == flo) && (fhi == fhi) && (flo > 0.0) && (fhi < 0.0);
                Set::Scalar pm = lo, fm = flo;
                if (ok)
                {
                    // Illinois regula falsi, bisection every 4th step for safety
                    int side = 0;
                    for (int it = 0; it < 200; ++it)
                    {
                        pm = (it % 4 == 3) ? 0.5 * (lo + hi) : (lo * fhi - hi * flo) / (fhi - flo);
                        if (!(pm > lo && pm < hi)) pm = 0.5 * (lo + hi);
                        fm = mstate(pm, v0, v1, qq);
                        if (std::abs(fm) < 1.0e-13 || (hi - lo) < 1.0e-14 * std::max(std::abs(pm), 1.0)) break;
                        if (fm > 0.0) { lo = pm; flo = fm; if (side == -1) fhi *= 0.5; side = -1; }
                        else          { hi = pm; fhi = fm; if (side == +1) flo *= 0.5; side = +1; }
                    }
                    fm = mstate(pm, v0, v1, qq);
                    ok = (std::abs(fm) < 1.0e-8) && (v0 > 0.0) && (v1 > 0.0) && (qq == qq);
                }
                if (ok)
                {
                    Set::Scalar a1m = arh0_loc * v0, a2m = arh1_loc * v1;
                    const Set::Scalar asm_ = a1m + a2m; a1m /= asm_; a2m /= asm_;
                    if (a1m > alpha_floor && a2m > alpha_floor)
                    {
                        const Set::Scalar pg = pm + Dj;
                        const Set::Scalar v0s = v00 * std::pow(c0 / (pm + pi0_), 1.0 / gam0);
                        const Set::Scalar v1s = v10 * std::pow(c1 / (pg + pi1_), 1.0 / gam1);
                        // E_k = (alpha rho)_k (e_k^s + q): sum = rho e exactly
                        const Set::Scalar E0n = arh0_loc * ((pm + gam0 * pi0_) * v0s / (gam0 - 1.0) + qq);
                        const Set::Scalar E1n = arh1_loc * ((pg + gam1 * pi1_) * v1s / (gam1 - 1.0) + qq);
                        eta(i, j, k) = a1m;
                        E0_(i, j, k) = E0n;
                        E1_(i, j, k) = E1n;
                        if (rhd)
                        {
                            rsp(i, j, k, 0) += arh1_loc * qq;
                            rsp(i, j, k, 1) += arh0_loc * qq;
                            if (qq < 0.0) rsp(i, j, k, 7) += 1.0;
                            if (pm < 0.0) rsp(i, j, k, 8) += 1.0;
                        }
                        if (diag_on) { diag_arr(i, j, k, 0) = 0.0; diag_arr(i, j, k, 1) = std::abs(fm); diag_arr(i, j, k, 2) = 0.0; }
                        return;
                    }
                }
                if (rhd) rsp(i, j, k, 6) += 1.0;     // fall back to closure 0
            }

            // Sch20 eq. 8 mixture pressure as the initial guess --
            // exact for the mechanical-equilibrium IC (p0_pre == p1_pre
            // -> p_init == p_mix == root), and a sensible interpolation
            // for general states.  Bounded away from p=0 by virtue of
            // SG positivity on both pre-relax phases.
            Set::Scalar p_init = a1 * p0_pre + a2 * (p1_pre - Dj);
            // Lower bracket: a hair above p_min (avoid the gas v -> inf
            // pole at p -> -pi_gas).  For typical gas/liquid: p_min ~ 0,
            // p_lo ~ small positive.  For all-liquid pair: p_min ~ -pi,
            // p_lo > p_min.
            Set::Scalar p_lo = p_tfloor + 1.0e-9 * std::max(std::abs(p_tfloor), 1.0);
            // Upper bracket: generously above the maximum pre-relax p.
            Set::Scalar p_hi = 1.0e4 * std::max(std::max(std::abs(p0_pre), std::abs(p1_pre)), 1.0);

            // Local evaluator (captures pre-relax state from this cell).
            auto eval_f = [&](Set::Scalar p) -> Set::Scalar
            {
                Set::Scalar v0 = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1 = Solver::EOS::EOS::RelaxationVolume_SG_SC(p + Dj, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                return arh0_loc * v0 + arh1_loc * v1 - 1.0;
            };

            // Start Newton at p_init clamped into [p_lo, p_hi].
            Set::Scalar p = std::max(std::min(p_init, p_hi), p_lo);

            int iters_used = max_iter;
            Set::Scalar f_final = eval_f(p);
            bool newton_ok = false;
            for (int it = 0; it < max_iter; ++it)
            {
                Set::Scalar v0_p = Solver::EOS::EOS::RelaxationVolume_SG_SC(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1_p = Solver::EOS::EOS::RelaxationVolume_SG_SC(p + Dj, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                Set::Scalar f    = arh0_loc * v0_p + arh1_loc * v1_p - 1.0;
                f_final = f;

                Set::Scalar dv0 = Solver::EOS::EOS::RelaxationVolume_SG_SC_dvdp(p, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar dv1 = Solver::EOS::EOS::RelaxationVolume_SG_SC_dvdp(p + Dj, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                Set::Scalar df  = arh0_loc * dv0 + arh1_loc * dv1;

                if (std::abs(df) < DIV_FLOOR) break;       // bail to bisection
                Set::Scalar dp = -f / df;
                Set::Scalar p_new = p + dp;
                // Clamp Newton step into the bracket -- prevents the
                // Tammann-stiff cells from taking 10^9-scale steps that
                // overshoot into the gas singularity at p -> 0.
                if (p_new < p_lo) p_new = 0.5 * (p + p_lo);
                if (p_new > p_hi) p_new = 0.5 * (p + p_hi);
                if (std::abs(dp) < newton_tol * std::max(std::abs(p_new), 1.0))
                {
                    p = p_new;
                    f_final = eval_f(p);
                    iters_used = it + 1;
                    newton_ok = true;
                    break;
                }
                p = p_new;
            }

            // -----------------------------------------------------------
            // Bisection fallback.  Triggered when Newton ran the full
            // max_iter without satisfying the convergence test OR when
            // Newton bailed early on a degenerate Jacobian.  f(p) is
            // monotone-decreasing so bracketing is robust: we just need
            // f(p_lo) > 0 and f(p_hi) < 0.  If the bracket is broken
            // (rare degenerate states), expand outward.
            // -----------------------------------------------------------
            if (!newton_ok)
            {
                Set::Scalar f_lo = eval_f(p_lo);
                Set::Scalar f_hi = eval_f(p_hi);

                // Walk the upper bound out if we don't bracket -- happens
                // if pre-relax state has anomalously high partial densities.
                int expand = 0;
                while (f_hi > 0.0 && expand < 20)
                {
                    p_hi *= 10.0;
                    f_hi  = eval_f(p_hi);
                    ++expand;
                }
                // Walk lower bound in if needed.
                while (f_lo < 0.0 && p_lo > p_min && expand < 40)
                {
                    p_lo = std::max(0.5 * p_lo, p_min);
                    f_lo = eval_f(p_lo);
                    ++expand;
                }

                if (f_lo * f_hi <= 0.0)
                {
                    // Standard bisection -- f monotone-decreasing.
                    Set::Scalar p_mid = 0.5 * (p_lo + p_hi);
                    Set::Scalar f_mid = 0.0;
                    int bit = 0;
                    for (bit = 0; bit < bisect_max; ++bit)
                    {
                        p_mid = 0.5 * (p_lo + p_hi);
                        f_mid = eval_f(p_mid);
                        if (f_mid > 0.0) p_lo = p_mid;
                        else             p_hi = p_mid;
                        if ((p_hi - p_lo) < bisect_tol * std::max(std::abs(p_mid), 1.0)) break;
                    }
                    p = p_mid;
                    f_final = f_mid;
                    iters_used = max_iter + bit + 1;   // distinguishable from Newton-only count
                }
                // else: degenerate (no root in bracket); leave p at last Newton iterate.
            }

            Set::Scalar p_relaxed = p;
            const Set::Scalar Dj_relax = Dj;   // jump used by the solve (reinit may zero Dj)

            // Write diagnostic: (iter count, |f| at final p).
            if (diag_on)
            {
                diag_arr(i, j, k, 0) = static_cast<Set::Scalar>(iters_used);
                diag_arr(i, j, k, 1) = std::abs(f_final);
            }

            // Post-relax volume fractions (using self-consistent v_k(p)).
            // DIV_FLOOR matches the pre-relax / Newton-loop convention.
            // ROBUST FALLBACK: if the volume-constraint solve did NOT converge
            // (unphysical per-phase input -- e.g. a coarse cell whose per-phase
            // ENERGY split came out with the wrong sign, while the conserved
            // MIXTURE energy E is still correct), do NOT trust the failed solve's
            // alpha.  Keep the prior alpha and let the energy-consistent reinit
            // (Sch20 eq.26, below) rebuild a valid per-phase state from E.  Without
            // this the masked-garbage alpha poisons the interface (rho_eta0/1) and
            // the per-phase energies at the bubble center on coarse grids.
            const bool relax_failed = std::abs(f_final) > unconv_threshold;
            Set::Scalar a1_new, a2_new;
            if (relax_failed)
            {
                a1_new = a1;   // prior (clamped) volume fraction -- trust the conserved E instead
                a2_new = a2;
            }
            else
            {
                Set::Scalar v0_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed, p0_pre, rho0_pre, gam0, pi0_, DIV_FLOOR);
                Set::Scalar v1_r = Solver::EOS::EOS::RelaxationVolume_SG_SC(p_relaxed + Dj, p1_pre, rho1_pre, gam1, pi1_, DIV_FLOOR);
                a1_new = arh0_loc * v0_r;
                a2_new = arh1_loc * v1_r;
                Set::Scalar asum = a1_new + a2_new;
                if (asum > DIV_FLOOR) { a1_new /= asum; a2_new /= asum; }
            }
            // FINITE-RATE PRESSURE RELAXATION (relax.omega >= 0; default < 0 = instantaneous, unchanged; from
            // DropletDetonation).  d(alpha_1)/dt = omega alpha_1 alpha_2 (p_1 - p_2), the law of Favrie & Schmidmayer
            // (2026, Sec. 4) and of Biasiori-Poulanges & Schmidmayer (2023).  Linearised about the equilibrium found
            // above, p_1 - p_2 = -(K_1/alpha_1 + K_2/alpha_2)(alpha_1 - alpha_eq) with K_k = rho_k c_k^2, so the volume
            // fraction covers the fraction theta = 1 - exp(-omega (alpha_2 K_1 + alpha_1 K_2) dt) of the way to
            // equilibrium.  The phases then KEEP THEIR OWN PRESSURES (below): the volume change does work at the
            // interface pressure p_I = (Z_2 p_1 + Z_1 p_2)/(Z_1 + Z_2) (their eqs. 32 - 33) and the remaining defect of
            // the total energy is shared by volume or mass fractions (relax.energy_share; their eqs. 113 - 115).
            // theta -> 1 with equal pressures recovers the instantaneous scheme.  These cells take the Sch20 solve
            // above (the isentropic closure is skipped for them).
            bool finite_rate = false;
            if (omega_r >= 0.0)
            {
                const Set::Scalar K0 = gam0 * std::max(p0_pre + pi0_, 0.0), K1 = gam1 * std::max(p1_pre + pi1_, 0.0);
                const Set::Scalar th = 1.0 - std::exp(-omega_r * (a2 * K0 + a1 * K1) * dt_r);
                a1_new = a1 + th * (a1_new - a1);
                finite_rate = true;
            }
            a1_new = std::min(std::max(a1_new, alpha_floor), 1.0 - alpha_floor);
            a2_new = 1.0 - a1_new;

            // -----------------------------------------------------------
            // Reinit: recompute mixture p from the (CONSERVED) redundant
            // rho E using the POST-relaxation alpha_k.
            // Schmidmayer 2020 eq. 26 / Saurel 2009 eq. III.5.
            // -----------------------------------------------------------
            Set::Scalar rho_loc = arh0_loc + arh1_loc;
            Set::Scalar ke      = 0.5 * (AMREX_D_TERM(M_(i, j, k, 0) * M_(i, j, k, 0),
                                                    + M_(i, j, k, 1) * M_(i, j, k, 1),
                                                    + M_(i, j, k, 2) * M_(i, j, k, 2))) / std::max(rho_loc, small_loc);
            // Debit the capillary energy before the EOS inversion: with the
            // closure on, rho E carries e_cap, and reading it as internal
            // energy inflates p at the interface (dt collapse).  This is the
            // THIRD rho E inversion site -- it names its local `rho_e`, which
            // is why the E_vol - KE grep did not surface it.
            Set::Scalar rho_e   = std::max(E_(i, j, k) - ke - w_elastic(), small_loc);

            if (finite_rate)
            {
                const Set::Scalar Z0 = std::sqrt(rho0_pre * gam0 * std::max(p0_pre + pi0_, 0.0)), Z1 = std::sqrt(rho1_pre * gam1 * std::max(p1_pre + pi1_, 0.0));
                const Set::Scalar pI = (Z0 + Z1 > 0.0) ? (Z1 * p0_pre + Z0 * p1_pre) / (Z0 + Z1) : 0.5 * (p0_pre + p1_pre);
                const Set::Scalar da = a1_new - a1;
                Set::Scalar E0n = E0_(i, j, k) - pI * da, E1n = E1_(i, j, k) + pI * da;
                const Set::Scalar eps = rho_e - (E0n + E1n);
                // share of the defect: volume fractions (their eq. 113) or mass fractions (below their eq. 119).
                // By volume the rarefied gas of an opening spall void (alpha_gas ~ 0.9, mass 1e-5 of the cell)
                // receives nearly all of the solid's defect; by mass it does not.
                const Set::Scalar w0 = share_mass ? arh0_loc / std::max(rho_loc, small_loc) : a1_new;
                E0n += w0 * eps; E1n += (1.0 - w0) * eps;
                const Set::Scalar p0n = (gam0 - 1.0) * E0n / a1_new - gam0 * pi0_, p1n = (gam1 - 1.0) * E1n / a2_new - gam1 * pi1_;
                if (p0n > -pi0_ + small_loc && p1n > -pi1_ + small_loc)       // both phases physical: done
                {
                    eta(i, j, k) = a1_new; E0_(i, j, k) = E0n; E1_(i, j, k) = E1n;
                    if (diag_on) diag_arr(i, j, k, 2) = std::abs(p0n - p1n);
                    return;
                }
                // otherwise fall back to the one-pressure re-initialisation below
            }

            // rho_e = a1 (p + g0 pi0)/(g0-1) + a2 (p + Dj + g1 pi1)/(g1-1): the
            // equal-pressure reinit of rho_e - a2 Dj/(g1-1) gives the liquid p.
            if (relax_failed) Dj = 0.0;
            Set::Scalar p_reinit = Solver::EOS::EOS::ReinitMixturePressure(rho_e - a2_new * Dj / (gam1 - 1.0), a1_new, a2_new,
                                                                          gam0, pi0_, gam1, pi1_, small_loc);
            // Per-phase floors.  If the requested state (liquid p_reinit, gas p_reinit + Dj)
            // puts exactly ONE phase below its EOS limit (p_k + pi_k <= 0), that phase sits
            // at its own floor and the other phase takes the rest of rho_e:
            //     E_o = alpha_o (-pi_o + gam_o pi_o)/(gam_o - 1),   E_v = rho_e - E_o.
            // Exact in energy; the valid phase keeps whatever pressure that gives (liquid
            // may be negative down to -pi_l).  Replaces dropping the jump (Dj := 0) and
            // flooring both phases at ~0, which left 0 Pa holes in the 02 band next to
            // liquid at 1-2e6 Pa and created energy (2026-11-09).
            bool split_ok = false;
            Set::Scalar E0s = 0.0, E1s = 0.0;
            {
                const bool out0 = (p_reinit + pi0_ <= small_loc);
                const bool out1 = (p_reinit + Dj + pi1_ <= small_loc);
                if (out0 != out1)
                {
                    if (out1)
                    {
                        E1s = a2_new * (-pi1_ + small_loc + gam1 * pi1_) / (gam1 - 1.0);
                        E0s = rho_e - E1s;
                        split_ok = ((gam0 - 1.0) * E0s / std::max(a1_new, DIV_FLOOR) - gam0 * pi0_ + pi0_ > small_loc);
                    }
                    else
                    {
                        E0s = a1_new * (-pi0_ + small_loc + gam0 * pi0_) / (gam0 - 1.0);
                        E1s = rho_e - E0s;
                        split_ok = ((gam1 - 1.0) * E1s / std::max(a2_new, DIV_FLOOR) - gam1 * pi1_ + pi1_ > small_loc);
                    }
                }
            }
            if (!split_ok && Dj != 0.0 && (p_reinit + pi0_ <= small_loc || p_reinit + Dj + pi1_ <= small_loc))
            {
                Dj = 0.0;   // both phases out of range: equal pressures here (old fallback)
                p_reinit = Solver::EOS::EOS::ReinitMixturePressure(rho_e, a1_new, a2_new,
                                                                   gam0, pi0_, gam1, pi1_, small_loc);
            }

            // Floor: keep p + pi_k > 0 for both phases (SG positivity).  The former
            // "mixture >= 0" term (-a2 Dj) is gone with the per-phase face rebuild.
            // (with a finite relaxation rate or an elastic solid the dominant phase sets the floor: a solid that the
            // gas has not yet relieved carries tension)
            const Set::Scalar p_min_ph = ((finite_rate || el_on) ? -((a1_new >= 0.5) ? pi0_ : pi1_) : std::max(-pi0_, -pi1_ - Dj)) + small_loc;
            if (!split_ok) p_reinit = std::max(p_reinit, p_min_ph);

            // Diagnostic: gap between Newton's converged p and reinit p.
            // Nonzero gap means the volume-constraint root and the energy-
            // consistent root disagree -- per-cell noise from the upwind
            // flux feeding into a cell-local Newton, or vice versa.
            if (diag_on)
            {
                diag_arr(i, j, k, 2) = std::abs(p_relaxed - p_reinit);
            }

            // -----------------------------------------------------------
            // Reset per-phase canonical energies from p_reinit.
            // (Sau09 eq. III.5: e_k = (p + gam pi)/((gam-1) rho_k), so
            //  (alpha rho e)_k = alpha_k (p + gam pi)/(gam - 1).)
            // -----------------------------------------------------------
            eta(i, j, k) = a1_new;
            if (!relax_failed && !finite_rate)   // energy defect split by mass fraction
            {
                // Relaxed phasic energies, then the defect by mass fraction.
                const Set::Scalar E0r = Solver::EOS::EOS::PhasicEnergyFromPressure(p_relaxed, a1_new, gam0, pi0_, small_loc);
                const Set::Scalar E1r = Solver::EOS::EOS::PhasicEnergyFromPressure(p_relaxed + Dj_relax, a2_new, gam1, pi1_, small_loc);
                const Set::Scalar dlt = rho_e - (E0r + E1r);
                const Set::Scalar Y0 = arh0_loc / std::max(rho_loc, small_loc), Y1 = 1.0 - Y0;
                const Set::Scalar E0m = E0r + Y0 * dlt, E1m = E1r + Y1 * dlt;
                // keep both phases inside the SG range, else fall through to eq. 26
                const Set::Scalar q0 = (gam0 - 1.0) * E0m / a1_new - gam0 * pi0_ + pi0_;
                const Set::Scalar q1 = (gam1 - 1.0) * E1m / a2_new - gam1 * pi1_ + pi1_;
                if (q0 > small_loc && q1 > small_loc)
                {
                    E0_(i, j, k) = E0m;
                    E1_(i, j, k) = E1m;
                    if (rhd)
                    {
                        const Set::Scalar ps1 = (p1_pre + pi1_) * std::pow(a2 / a2_new, gam1) - pi1_;
                        rsp(i, j, k, 0) += Y1 * dlt;
                        rsp(i, j, k, 1) += Y0 * dlt;
                        rsp(i, j, k, 2) += E1r - a2_new * (ps1 + gam1 * pi1_) / (gam1 - 1.0);
                    }
                    return;
                }
            }
            if (split_ok)
            {
                E0_(i, j, k) = E0s;
                E1_(i, j, k) = E1s;
            }
            else
            {
                E0_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_reinit, a1_new, gam0, pi0_, small_loc);
                E1_(i, j, k) = Solver::EOS::EOS::PhasicEnergyFromPressure(p_reinit + Dj, a2_new, gam1, pi1_, small_loc);
            }
            if (rhd)
            {
                // eq. 26 fallback: split the change into Newton (relaxed p) and the reset to p_reinit
                const Set::Scalar E0q = Solver::EOS::EOS::PhasicEnergyFromPressure(p_relaxed, a1_new, gam0, pi0_, small_loc);
                const Set::Scalar E1q = Solver::EOS::EOS::PhasicEnergyFromPressure(p_relaxed + Dj_relax, a2_new, gam1, pi1_, small_loc);
                const Set::Scalar ps1 = (p1_pre + pi1_) * std::pow(a2 / a2_new, gam1) - pi1_;
                rsp(i, j, k, 0) += E1_(i, j, k) - E1q;
                rsp(i, j, k, 1) += E0_(i, j, k) - E0q;
                rsp(i, j, k, 2) += E1q - a2_new * (ps1 + gam1 * pi1_) / (gam1 - 1.0);
                rsp(i, j, k, 5) += 1.0;
            }

            // rho_eta{0,1}, momentum, energy_per_vol are NOT touched
            // (they are conserved through relaxation by construction).
        });
    }

    // relax.heat_diag: energy beyond isentropic compression, per phase (output only).
    // Partial densities are unchanged by relaxation, so rho_post/rho_pre = alpha_pre/alpha_post.
    if (relax_heat_diag)
    {
        for (amrex::MFIter mfi(rh_pre, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto eta = eta_mf[lev]->const_array(mfi);
            auto E0_ = energy0_mf[lev]->const_array(mfi);
            auto E1_ = energy1_mf[lev]->const_array(mfi);
            auto q   = rh_pre.const_array(mfi);
            auto H   = relax_heat_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                const Set::Scalar a1 = std::min(std::max(eta(i, j, k), alpha_floor), 1.0 - alpha_floor);
                const Set::Scalar a2 = 1.0 - a1;
                const Set::Scalar a2p = q(i, j, k, 0), a1p = 1.0 - a2p;
                const Set::Scalar ps0 = (q(i, j, k, 1) + pi0_) * std::pow(a1p / a1, gam0) - pi0_;
                const Set::Scalar ps1 = (q(i, j, k, 2) + pi1_) * std::pow(a2p / a2, gam1) - pi1_;
                H(i, j, k, 1) += E0_(i, j, k) - a1 * (ps0 + gam0 * pi0_) / (gam0 - 1.0);
                H(i, j, k, 0) += E1_(i, j, k) - a2 * (ps1 + gam1 * pi1_) / (gam1 - 1.0);
            });
        }
    }


    // ----------------------------------------------------------------
    // Diagnostic summary.
    // ----------------------------------------------------------------
    if (diag_on)
    {
        const Set::Scalar max_iters = diag_mf->max(0);   // already MPI-reduced
        const Set::Scalar max_res   = diag_mf->max(1);
        const Set::Scalar max_pgap  = diag_mf->max(2);   // |p_relaxed - p_reinit|

        // Count cells that hit max_iter without converging.
        int unconv = 0;
        for (amrex::MFIter mfi(*diag_mf, false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            const auto arr = (*diag_mf)[mfi].array();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);
            for (int k = lo.z; k <= hi.z; ++k)
                for (int j = lo.y; j <= hi.y; ++j)
                    for (int i = lo.x; i <= hi.x; ++i)
                    {
                        // Count as truly unconverged only if iters hit the max AND |f| exceeds the physical-engineering tolerance.
                        if (static_cast<int>(arr(i, j, k, 0)) >= max_iter
                            && arr(i, j, k, 1) > unconv_threshold)
                            ++unconv;
                    }
        }
        amrex::ParallelDescriptor::ReduceIntSum(unconv);

        Util::Message(INFO, "RelaxAndReinit lev=", lev,
                      " step=", step_counter[lev],
                      " max_iters=", static_cast<int>(max_iters),
                      "/", max_iter,
                      " max_residual=", max_res,
                      " max|p_relaxed-p_reinit|=", max_pgap,
                      " unconverged_cells=", unconv);

        if (unconv > 0)
        {
            Util::Message(INFO,
                "  ** WARNING: ", unconv,
                " cell(s) hit max_iter without satisfying |f(p)| < ",
                newton_tol, ".  Newton is failing to relax pressure in those cells.");
        }
    }
    if (diag_gasmass) GasLedger(lev, "relax_exit", relax_time);
    if (relax_diag) { char _mt[128]; snprintf(_mt, 128, "MASSTRACK relax_exit    lev=%d sum1=%.14e", lev, rho_eta1_mf[lev]->sum(0)); Util::Message(INFO, _mt); }
} // end RelaxAndReinit()


///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////// REFLUX SCRATCH ALLOC ////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Conserved primaries refluxed at coarse-fine boundaries:
//   rho_eta0, rho_eta1, M_x, M_y, rho*E, E_0, E_1   (7 components total)
//
// We split mass per-phase (eta*F, (1-eta)*F at the face) rather than refluxing
// total mass + repartitioning, because rho_eta_k are the actual evolved fields
// (not density_mf, which is derived).  See Matt Q's commit a5783ab32.
//
// E_0, E_1 are 6-eq additions not present in the 5-eq line; refluxed alongside
// the mixture energy so per-phase internal energies stay consistent across c/f.
//
// Uses density_mf[lev]'s BoxArray / DistributionMap because grids[lev]/dmap[lev]
// is not guaranteed populated when this runs (see memory note
// feedback_amr_perlevel_alloc_must_cover_level0.md -- prior reflux attempt
// segfaulted by reading empty grids[lev] inside MakeNewLevelFromScratch).
// density_mf[lev] is guaranteed allocated by RegisterNewFab before Initialize.
//
// LIQUID <-> VAPOUR PHASE CHANGE: thermodynamic relaxation of every cell that holds both liquid
// and gas (Hydro2_PhaseChange.H).  Mass and total energy of the cell are conserved; the latent
// heat appears as the change dE of the hydrodynamic energy sum and of rho E.
void Hydro2::RelaxPhaseChange(int lev)
{
    BL_PROFILE("Integrator::Hydro2::RelaxPhaseChange");
    const int liq = phasechange.liquid;
    const Set::Scalar amin = phasechange.alpha_min;
    const PhaseChange::Phase Lp = phasechange.L, Vp = phasechange.V;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.tilebox();
        auto eta  = eta_mf[lev]->array(mfi);
        auto m0   = rho_eta0_mf[lev]->array(mfi);
        auto m1   = rho_eta1_mf[lev]->array(mfi);
        auto E0_  = energy0_mf[lev]->array(mfi);
        auto E1_  = energy1_mf[lev]->array(mfi);
        auto E_   = energy_per_vol_mf[lev]->array(mfi);
        auto vap  = phasechange.vapor_mf[lev]->array(mfi);
        auto dg   = phasechange.diag_mf[lev]->array(mfi);
        Set::Patch<const Set::Scalar> phisol = embedded.phi_mf.Patch(lev, mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            if (embedded.apply && embedded.isSolid(phisol(i, j, k))) return;
            const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar al = (liq == 0) ? a0 : 1.0 - a0, ag = 1.0 - al;
            Set::Scalar ml = std::max((liq == 0) ? m0(i, j, k) : m1(i, j, k), 0.0);
            Set::Scalar mg = std::max((liq == 0) ? m1(i, j, k) : m0(i, j, k), 0.0);
            Set::Scalar mv = std::min(std::max(vap(i, j, k), 0.0), mg);
            const Set::Scalar El = (liq == 0) ? E0_(i, j, k) : E1_(i, j, k), Eg = (liq == 0) ? E1_(i, j, k) : E0_(i, j, k);
            const Set::Scalar pl = (Lp.g - 1.0) * El / std::max(al, 1.0e-300) - Lp.g * Lp.pi;
            const Set::Scalar pg = (Vp.g - 1.0) * Eg / std::max(ag, 1.0e-300) - Vp.g * Vp.pi;
            const Set::Scalar pmix = al * pl + ag * pg;
            dg(i, j, k, 2) = (mg > 1.0e-300) ? mv / mg : 0.0;
            dg(i, j, k, 3) = 0.0;
            dg(i, j, k, 1) = phasechange.Tsat(std::max(pmix, 1.0e-300 - Vp.pi));
            dg(i, j, k, 0) = (al > 0.5) ? PhaseChange::temp(Lp, pmix, ml / std::max(al, 1.0e-300)) : PhaseChange::temp(Vp, pmix, mg / std::max(ag, 1.0e-300));
            vap(i, j, k) = mv;
            if (al < amin || ag < amin) return;
            Set::Scalar aln, p, T, dE;
            const Set::Scalar mg_old = mg;
            if (!phasechange.relax(ml, mg, mv, El + Eg, pmix, aln, p, T, dE)) return;
            const Set::Scalar agn = 1.0 - aln;
            const Set::Scalar Eln = aln * (p + Lp.g * Lp.pi) / (Lp.g - 1.0), Egn = agn * (p + Vp.g * Vp.pi) / (Vp.g - 1.0);
            if (liq == 0) { m0(i, j, k) = ml; m1(i, j, k) = mg; E0_(i, j, k) = Eln; E1_(i, j, k) = Egn; eta(i, j, k) = aln; }
            else          { m1(i, j, k) = ml; m0(i, j, k) = mg; E1_(i, j, k) = Eln; E0_(i, j, k) = Egn; eta(i, j, k) = agn; }
            E_(i, j, k) += (Eln + Egn) - (El + Eg);      // = dE up to the solver tolerance: the latent heat
            vap(i, j, k) = mv;
            dg(i, j, k, 0) = T; dg(i, j, k, 1) = phasechange.Tsat(std::max(p, 1.0e-300 - Vp.pi));
            dg(i, j, k, 2) = mv / std::max(mg, 1.0e-300); dg(i, j, k, 3) = mg - mg_old;
            (void)dE;
        });
    }
}

void Hydro2::AllocateRefluxScratch(int lev)
{
    BL_PROFILE("Integrator::Hydro2::AllocateRefluxScratch");

    if ((int)flux_reg.size() <= lev)  flux_reg.resize(lev + 1);
    if ((int)cc_fluxes.size() <= lev) cc_fluxes.resize(lev + 1);

    const amrex::BoxArray& ba           = density_mf[lev]->boxArray();
    const amrex::DistributionMapping& dm = density_mf[lev]->DistributionMap();

    // FluxRegister sits between lev-1 (coarse) and lev (fine).
    if (lev > 0)
    {
        // rho0, rho1, M_x, M_y, rhoE, E_0, E_1 [, species partial densities (species.on, 2026-11-18)]
        const int ncomp_reflux = 2 + AMREX_SPACEDIM + 1 + 2 + (species.on ? species.ns() : 0);
        flux_reg[lev] = std::make_unique<amrex::FluxRegister>(
            ba, dm, refRatio(lev - 1), lev, ncomp_reflux);
        flux_reg[lev]->setVal(0.0);
    }

    // Cell-centered hi-face flux scratch.  1 ghost so FillBoundary can
    // propagate the hi-face flux of one box to the lo-face of its neighbor
    // when converting to face-centered MultiFabs in Advance().
    for (int d = 0; d < AMREX_SPACEDIM; d++)
    {
        cc_fluxes[lev].mass[d]   = std::make_unique<amrex::MultiFab>(ba, dm, 2,                1);
        cc_fluxes[lev].mom[d]    = std::make_unique<amrex::MultiFab>(ba, dm, AMREX_SPACEDIM,   1);
        cc_fluxes[lev].energy[d] = std::make_unique<amrex::MultiFab>(ba, dm, 1,                1);
        cc_fluxes[lev].ene_k[d]  = std::make_unique<amrex::MultiFab>(ba, dm, 2,                1);
        if (species.on) { cc_fluxes[lev].spec[d] = std::make_unique<amrex::MultiFab>(ba, dm, species.ns(), 1); cc_fluxes[lev].spec[d]->setVal(0.0); }

        cc_fluxes[lev].mass[d]  ->setVal(0.0);
        cc_fluxes[lev].mom[d]   ->setVal(0.0);
        cc_fluxes[lev].energy[d]->setVal(0.0);
        cc_fluxes[lev].ene_k[d] ->setVal(0.0);
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////// POST-SUBCYCLE REFLUX /////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Called from base Integrator::TimeStep AFTER all fine subcycles complete
// and BEFORE average_down.  Applies the FluxRegister correction to the
// coarse-level conserved primaries, then re-derives density/velocity from
// the corrected conserved state.
//
// Register layout (must match component offsets in Advance/RHS):
//   [0]               rho_eta0       (per-phase mass 0)
//   [1]               rho_eta1       (per-phase mass 1)
//   [2 .. 2+SD-1]     momentum_k     (k = x, y, ...)
//   [2+SD]            energy_per_vol (total rho*E)
//   [3+SD .. 4+SD]    E_0, E_1       (per-phase internal energies; 6-eq)
//
void Hydro2::PostSubcycleReflux(int lev, Set::Scalar /*time*/, Set::Scalar /*dt_coarse*/)
{
    BL_PROFILE("Integrator::Hydro2::PostSubcycleReflux");

    if (lev >= finest_level) return;
    const int fine_lev = lev + 1;
    if (fine_lev >= (int)flux_reg.size() || !flux_reg[fine_lev]) return;

    constexpr int IDX_RHO_ETA0 = 0;
    constexpr int IDX_RHO_ETA1 = 1;
    constexpr int IDX_M        = 2;
    constexpr int IDX_E_VOL    = 2 + AMREX_SPACEDIM;
    constexpr int IDX_E0       = 3 + AMREX_SPACEDIM;
    constexpr int IDX_E1       = 4 + AMREX_SPACEDIM;

    flux_reg[fine_lev]->Reflux(*rho_eta0_mf[lev],       1.0, IDX_RHO_ETA0, 0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*rho_eta1_mf[lev],       1.0, IDX_RHO_ETA1, 0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*momentum_mf[lev],       1.0, IDX_M,        0, AMREX_SPACEDIM, geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy_per_vol_mf[lev], 1.0, IDX_E_VOL,    0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy0_mf[lev],        1.0, IDX_E0,       0, 1,              geom[lev]);
    flux_reg[fine_lev]->Reflux(*energy1_mf[lev],        1.0, IDX_E1,       0, 1,              geom[lev]);
    if (species.on && flux_reg[fine_lev]->nComp() > 5 + AMREX_SPACEDIM)
    {
        flux_reg[fine_lev]->Reflux(*species.mf[lev], 1.0, 5 + AMREX_SPACEDIM, 0, species.ns(), geom[lev]);
        CloseSpecies(lev, false);
    }

    // Recompute derived fields from the refluxed conserved state.
    // density and velocity are not refluxed directly -- they're derived.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*density_mf[lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto rho  = density_mf[lev]->array(mfi);
        auto rho0 = rho_eta0_mf[lev]->array(mfi);
        auto rho1 = rho_eta1_mf[lev]->array(mfi);
        auto M    = momentum_mf[lev]->array(mfi);
        auto v    = velocity_mf[lev]->array(mfi);
        const Set::Scalar small_local = small;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            const Set::Scalar rho_safe = std::max(rho(i, j, k), small_local);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / rho_safe;
        });
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////// POST-AVERAGE-DOWN //////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
//
// AMReX average_down independently averages every conserved primary into
// each coarse-under-fine cell.  For the 6-eq state that includes E_0 and E_1
// (per-phase internal energies), which are nonlinear functions of (rho*E,
// rho_eta_k, alpha) via the EOS.  Independent averaging breaks the EOS
// relation: the averaged (E_0, E_1) don't satisfy mech equilibrium with the
// averaged (alpha, rho_eta_k, rho*E). 
// 
// After average_down, recompute (E_0, E_1) on the coarse level from
// the averaged mixture state via Saurel III.5 mech-equilibrium:
//   e_int = sum_k alpha_k * (p_mix + gamma_k * pi_k) / (gamma_k - 1)
//   solve for p_mix, then E_k = alpha_k * (p_mix + gamma_k * pi_k) / (gamma_k - 1).
//
void Hydro2::PostAverageDown(int coarse_lev)
{
    BL_PROFILE("Integrator::Hydro2::PostAverageDown");

    // Only coarse-under-fine cells got touched by average_down -- those are
    // the cells whose EOS consistency we need to restore.  Other coarse
    // cells are already at mech equilibrium from the prior RelaxAndReinit;
    // overwriting them introduces a small per-step bias that compounds in
    // low-Mach tests (~11 um interface drift over 3 us in pure advection).
    //
    // makeFineMask builds an iMultiFab on the coarse grid: 1 where covered
    // by the fine BoxArray (coarsened to coarse resolution), 0 elsewhere.
    if (coarse_lev + 1 > finest_level) return;

    amrex::iMultiFab fine_cover_mask = amrex::makeFineMask(
        eta_mf[coarse_lev]->boxArray(),
        eta_mf[coarse_lev]->DistributionMap(),
        eta_mf[coarse_lev + 1]->boxArray(),
        refRatio(coarse_lev),
        /*crse_value=*/ 0,
        /*fine_value=*/ 1);

    const Set::Scalar gam0 = eos0.Gamma();
    const Set::Scalar pi0_ = eos0.P0();
    const Set::Scalar gam1 = eos1.Gamma();
    const Set::Scalar pi1_ = eos1.P0();
    const Set::Scalar small_local = small;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*eta_mf[coarse_lev], HydroTiling()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        auto eta   = eta_mf[coarse_lev]        ->array(mfi);
        auto rho0  = rho_eta0_mf[coarse_lev]   ->array(mfi);
        auto rho1  = rho_eta1_mf[coarse_lev]   ->array(mfi);
        auto M     = momentum_mf[coarse_lev]   ->array(mfi);
        auto E_vol = energy_per_vol_mf[coarse_lev]->array(mfi);
        auto E0    = energy0_mf[coarse_lev]    ->array(mfi);
        auto E1    = energy1_mf[coarse_lev]    ->array(mfi);
        auto rho   = density_mf[coarse_lev]    ->array(mfi);
        auto v     = velocity_mf[coarse_lev]   ->array(mfi);
        auto mask  = fine_cover_mask.const_array(mfi);
        // Capillary-energy closure inputs (see Hydro2.H).
        const int marm_r = marmottant;
        const Set::Scalar sig_r = sigma, chi_r = marmottant_chi;
        const Set::Scalar Gb_r = marmottant_Gamma_buck;
        const Set::Scalar sbrk_r = marmottant_sigma_break, sigw_r = sigma;
        auto shell_r = shell_mf[coarse_lev]->const_array(mfi);
        auto cfun_r  = eta_mf[coarse_lev]->const_array(mfi);
        const Set::Scalar *DXr = geom[coarse_lev].CellSize();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            // Skip cells that average_down didn't touch.
            if (mask(i, j, k) == 0) return;
            // Re-derive density and velocity from refluxed/averaged conserved.
            rho(i, j, k) = rho0(i, j, k) + rho1(i, j, k);
            const Set::Scalar rho_safe = std::max(rho(i, j, k), small_local);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                v(i, j, k, d) = M(i, j, k, d) / rho_safe;

            // Volume fractions (clamped).
            const Set::Scalar a0 = std::min(std::max(eta(i, j, k), 0.0), 1.0);
            const Set::Scalar a1 = 1.0 - a0;

            // Mixture internal energy per volume:  e_int = rho*E - KE
            const Set::Scalar KE = 0.5 * rho_safe * (AMREX_D_TERM(v(i, j, k, 0) * v(i, j, k, 0),
                                                               + v(i, j, k, 1) * v(i, j, k, 1),
                                                               + v(i, j, k, 2) * v(i, j, k, 2)));
            const Set::Scalar e_int = E_vol(i, j, k) - KE;

            // Sau09 III.5:  e_int = sum_k alpha_k * (p + gamma_k * pi_k) / (gamma_k - 1)
            //   p_mix * [a0/(g0-1) + a1/(g1-1)] = e_int - [a0*g0*pi0/(g0-1) + a1*g1*pi1/(g1-1)]
            const Set::Scalar denom = a0 / std::max(gam0 - 1.0, small_local)
                                    + a1 / std::max(gam1 - 1.0, small_local);
            const Set::Scalar stiff_offset = a0 * gam0 * pi0_ / std::max(gam0 - 1.0, small_local)
                                           + a1 * gam1 * pi1_ / std::max(gam1 - 1.0, small_local);
            const Set::Scalar p_mix = (e_int - stiff_offset) / std::max(denom, small_local);

            // Recompute E_k consistent with p_mix.  This overwrites whatever
            // independent average_down produced.
            E0(i, j, k) = a0 * (p_mix + gam0 * pi0_) / std::max(gam0 - 1.0, small_local);
            E1(i, j, k) = a1 * (p_mix + gam1 * pi1_) / std::max(gam1 - 1.0, small_local);
        });
    }
}


void Hydro2::Integrate(int amrlev, Set::Scalar /*time*/, int /*step*/,
                       const amrex::MFIter& mfi, const amrex::Box& box)
{
    BL_PROFILE("Hydro2::Integrate");
    const Set::Scalar* DX = geom[amrlev].CellSize();
    const Set::Scalar dv = AMREX_D_TERM(DX[0], *DX[1], *DX[2]);

    const amrex::Array4<const Set::Scalar> eta = eta_mf[amrlev]->const_array(mfi);
    const amrex::Array4<const Set::Scalar> p   = pressure_mf[amrlev]->const_array(mfi);
    const amrex::Array4<const Set::Scalar> rho = density_mf[amrlev]->const_array(mfi);
    const amrex::Array4<const Set::Scalar> vel = velocity_mf[amrlev]->const_array(mfi);

    // Host loop into locals (no member access inside a device lambda).
    Set::Scalar vol = 0.0, pint = 0.0, ke = 0.0, area = 0.0;
    amrex::Loop(box, [&](int i, int j, int k)
    {
        const Set::Scalar g = 1.0 - eta(i, j, k);
        vol  += g * dv;
        pint += g * p(i, j, k) * dv;

        Set::Scalar u2 = AMREX_D_TERM(vel(i, j, k, 0) * vel(i, j, k, 0),
                                      + vel(i, j, k, 1) * vel(i, j, k, 1),
                                      + vel(i, j, k, 2) * vel(i, j, k, 2));
        ke += 0.5 * rho(i, j, k) * u2 * dv;

        Set::Scalar gx = (eta(i + 1, j, k) - eta(i - 1, j, k)) / (2.0 * DX[0]);
        Set::Scalar gy = (eta(i, j + 1, k) - eta(i, j - 1, k)) / (2.0 * DX[1]);
#if AMREX_SPACEDIM == 3
        Set::Scalar gz = (eta(i, j, k + 1) - eta(i, j, k - 1)) / (2.0 * DX[2]);
#else
        Set::Scalar gz = 0.0;
#endif
        area += std::sqrt(gx * gx + gy * gy + gz * gz) * dv;
    });

    thermo_gas_volume     += vol;
    thermo_gas_pressure   += pint;
    thermo_kinetic_energy += ke;
    thermo_interface_area += area;
}

// ----------------------------------------------------------------------
// OPTION 5 level-set slip (shell.slip = 5): same correction as ShellLevelSetSlip, but
// applied as an implicit (backward-Euler) upwind shift plus an exp(-kappa s dt) area stretch.
// ----------------------------------------------------------------------
void Hydro2::ShellLevelSetSlip5(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ShellLevelSetSlip5");
    if (!(dt > 0.0)) return;
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar cut = std::max(cutoff, 1.0e-12);
    const int rupt = shell_slip5_rupture;
    amrex::MultiFab G0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, shell_mf[lev]->nGrow());
    amrex::MultiFab::Copy(G0, *shell_mf[lev], 0, 0, 1, shell_mf[lev]->nGrow());
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto eo  = eta_old_mf[lev]->const_array(mfi);
        auto vel = velocity_mf[lev]->const_array(mfi);
        auto g0  = G0.const_array(mfi);
        auto sh  = shell_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar ec = et(i, j, k);
            if (!(ec > cut && ec < 1.0 - cut)) return;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = 2 * (d == 0), dj = 2 * (d == 1), dk = 2 * (d == 2);
                if (!et.contains(i + di, j + dj, k + dk) || !et.contains(i - di, j - dj, k - dk)) return;
            }
            const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
            const Set::Scalar g = ge.lpNorm<2>();
            if (g < g_gem_min) return;
            const Set::Vector nv = ge / g;
            // kappa = div n (central differences of the unit normal)
            Set::Scalar kap = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Vector gp = Numeric::Gradient(et, i + di, j + dj, k + dk, 0, DX);
                const Set::Vector gm = Numeric::Gradient(et, i - di, j - dj, k - dk, 0, DX);
                const Set::Scalar gpm = gp.lpNorm<2>(), gmm = gm.lpNorm<2>();
                if (gpm < g_gem_min || gmm < g_gem_min) return;
                kap += (gp(d) / gpm - gm(d) / gmm) / (2.0 * DX[d]);
            }
            if (rupt)
            {
                // extremum of eta along the normal inside this cell -> the level set has shrunk to a point
                Set::Scalar Dp = 0.0, Dm = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    Dp += nv(d) * (et(i + di, j + dj, k + dk) - ec) / DX[d];
                    Dm += nv(d) * (ec - et(i - di, j - dj, k - dk)) / DX[d];
                }
                if (Dp * Dm <= 0.0)
                {
                    if (g0(i, j, k) > 0.0)
                        printf("RUPT5 t=%.6e ijk=%d %d %d eta=%.6f g*dx=%.3e G0=%.4g\n", time, i, j, k, ec, g * DX[0], g0(i, j, k));
                    sh(i, j, k) = 0.0;
                    return;
                }
            }
            const Set::Scalar Vn = -(ec - eo(i, j, k)) / (dt * g);
            Set::Scalar un = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) un += vel(i, j, k, d) * nv(d);
            const Set::Scalar sl = Vn - un;
            // OPTION 5 (2026-11-10): implicit upwind shift.  The explicit
            // update G - s dt n.grad(G) amplifies once |s dt|/dx -> 1 (06 c12
            // 160 kPa: 0.02 -> 0.85 cells/step in 5 steps at 2.888 us, then NaN).
            // Backward-Euler in the shift with upwind neighbours at the old level:
            //   G_new (1 + sum c_d) = G0 exp(-kap s dt) + sum c_d G_up,d,
            //   c_d = |s n_d| dt / dx_d.
            // Same first-order consistency, a convex combination for any dt
            // (no new extrema, Gamma stays positive); the stretch keeps its exact exponential.
            Set::Scalar csum = 0.0, cup = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar a = sl * nv(d);
                const Set::Scalar c = std::fabs(a) * dt / DX[d];
                const Set::Scalar gup = (a > 0.0) ? g0(i - di, j - dj, k - dk) : g0(i + di, j + dj, k + dk);
                csum += c; cup += c * gup;
            }
            if (std::fabs(sl) * dt > 0.3 * DX[0])
                printf("SLIP5 t=%.6e ijk=%d %d %d eta=%.4f eta_old=%.4f g*dx=%.3e sdt/dx=%.3f kap*dx=%.3f G0=%.4f\n",
                       time, i, j, k, ec, eo(i, j, k), g * DX[0], sl * dt / DX[0], kap * DX[0], g0(i, j, k));
            sh(i, j, k) = (g0(i, j, k) * std::exp(-kap * sl * dt) + cup) / (1.0 + csum);
        });
    }
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
}

// OPTION 11 slip part (shell.slip = 11): the shell field is m = Gamma |grad eta|.
// dm/dt + div(m s n) = 0 with the level-set slip s n = w grad eta / |grad eta|^2,
// w = -(eta^{n+1} - eta^n)/dt - u.grad eta (step-average gradient, as option 6).
// Face velocity = mean of the two cell values, m upwinded with a MUSCL (minmod) face value:
// conservative, TVD for sum of outflow Courant numbers <= 1/2 (sub-cycled to 0.45).  Band cells (cut < eta < 1-cut,
// |grad eta| > g_gem_min) carry s; elsewhere s = 0.
void Hydro2::ShellSlipMass(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ShellSlipMass");
    if (!(dt > 0.0)) return;
    const bool donor = (shell_slip11_face == 1);
    const bool gw = (shell_slip11_face == 2);   // flux = Gamma_upwind (w n)_face  (m s n = Gamma w n)
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar cut = std::max(cutoff, 1.0e-12);
    const int ng = shell_mf[lev]->nGrow();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ng >= 2 && eta_mf[lev]->nGrow() >= 2 && velocity_mf[lev]->nGrow() >= 1,
                                     "ShellSlipMass needs ghost cells");
    // w = -(eta_t + u.grad eta) on the VALID cells only, then its ghosts filled with eta's BCs (w is a
    // scalar: RADIAL / mirror / far-field fills are exact for it).  Building w in the ghosts from ghost
    // eta_old and ghost velocity (a vector) pumped shell into the off-axis slab column (x100 by 0.1 us).
    amrex::MultiFab W(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, 1);
    W.setVal(0.0);
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto eo  = eta_old_mf[lev]->const_array(mfi);
        auto vel = velocity_mf[lev]->const_array(mfi);
        auto ww  = W.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar ug = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar gm = 0.5 * ((et(i + di, j + dj, k + dk) - et(i - di, j - dj, k - dk))
                                            + (eo(i + di, j + dj, k + dk) - eo(i - di, j - dj, k - dk))) / (2.0 * DX[d]);
                ug += vel(i, j, k, d) * gm;
            }
            ww(i, j, k) = -(et(i, j, k) - eo(i, j, k)) / dt - ug;
        });
    }
    FillBoundariesWithBC(lev, time, eta_bc, { &W });
    // VS = s n = w grad eta / |grad eta|^2 on the valid box grown by one (band cells only)
    amrex::MultiFab VS(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), AMREX_SPACEDIM + 1, 1);
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box bx = amrex::grow(mfi.validbox(), 1);
        auto et = eta_mf[lev]->const_array(mfi);
        auto ww = W.const_array(mfi);
        auto vs = VS.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Vector g1 = Set::Vector::Zero();
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                g1(d) = (et(i + di, j + dj, k + dk) - et(i - di, j - dj, k - dk)) / (2.0 * DX[d]);
            }
            const Set::Scalar ec = et(i, j, k);
            const Set::Scalar g2 = g1.squaredNorm();
            const bool band = (ec > cut && ec < 1.0 - cut && std::sqrt(g2) > g_gem_min);
            const Set::Scalar gn = std::sqrt(g2);
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
                vs(i, j, k, d) = band ? (gw ? ww(i, j, k) * g1(d) / gn : ww(i, j, k) * g1(d) / g2) : 0.0;
            vs(i, j, k, AMREX_SPACEDIM) = gn;
        });
    }
    // CFL: largest sum of outflow Courant numbers
    amrex::MultiFab CF(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(CF, false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto vs = VS.const_array(mfi);
        auto cf = CF.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar c = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                Set::Scalar vp = 0.5 * (vs(i, j, k, d) + vs(i + di, j + dj, k + dk, d));
                Set::Scalar vm = 0.5 * (vs(i - di, j - dj, k - dk, d) + vs(i, j, k, d));
                if (donor)
                {
                    vp = (vp > 0.0) ? std::max(vs(i, j, k, d), 0.0) : std::min(vs(i + di, j + dj, k + dk, d), 0.0);
                    vm = (vm > 0.0) ? std::max(vs(i - di, j - dj, k - dk, d), 0.0) : std::min(vs(i, j, k, d), 0.0);
                }
                if (gw)   // outflow Courant of m: (w n)_face / |grad eta|_donor
                {
                    const Set::Scalar gi = vs(i, j, k, AMREX_SPACEDIM);
                    c += (gi > g_gem_min) ? (std::max(vp, 0.0) - std::min(vm, 0.0)) / gi * dt / DX[d] : 0.0;
                }
                else c += (std::max(vp, 0.0) - std::min(vm, 0.0)) * dt / DX[d];
            }
            cf(i, j, k) = c;
        });
    }
    const Set::Scalar cmax = CF.max(0);
    int nsub = std::max(1, (int)std::ceil(cmax / 0.45));   // MUSCL + forward Euler: TVD for outflow Courant <= 1/2
    if (nsub > 64)
    {
        amrex::Print() << "SLIP11 t=" << time << " lev=" << lev << " outflow Courant " << cmax << " -> " << nsub << " substeps (capped 64)\n";
        nsub = 64;
    }
    else if (nsub > 1) amrex::Print() << "SLIP11 t=" << time << " lev=" << lev << " outflow Courant " << cmax << " -> " << nsub << " substeps\n";
    const Set::Scalar ds = dt / nsub;
    amrex::MultiFab M0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, ng);
    for (int sub = 0; sub < nsub; ++sub)
    {
        FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
        amrex::MultiFab::Copy(M0, *shell_mf[lev], 0, 0, 1, ng);
        for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto vs = VS.const_array(mfi);
            auto m0 = M0.const_array(mfi);
            auto sh = shell_mf[lev]->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                Set::Scalar div = 0.0;
                for (int d = 0; d < AMREX_SPACEDIM; ++d)
                {
                    const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                    Set::Scalar vp = 0.5 * (vs(i, j, k, d) + vs(i + di, j + dj, k + dk, d));
                    Set::Scalar vm = 0.5 * (vs(i - di, j - dj, k - dk, d) + vs(i, j, k, d));
                    if (donor)
                    {
                        vp = (vp > 0.0) ? std::max(vs(i, j, k, d), 0.0) : std::min(vs(i + di, j + dj, k + dk, d), 0.0);
                        vm = (vm > 0.0) ? std::max(vs(i - di, j - dj, k - dk, d), 0.0) : std::min(vs(i, j, k, d), 0.0);
                    }
                    // MUSCL face values, minmod slope of the upwind cell (first-order upwind smeared the
                    // shell out of the band: Gamma(eta=.5)/exact 1.0 -> 0.8 by 1.8 us, 06 c12).
                    auto mm = [](Set::Scalar a, Set::Scalar b) { return (a * b <= 0.0) ? 0.0 : ((std::fabs(a) < std::fabs(b)) ? a : b); };
                    auto face = [&](int ic, int jc, int kc, int sgn) {   // value of cell c extrapolated half a cell toward sgn
                        const Set::Scalar sl = mm(m0(ic + di, jc + dj, kc + dk) - m0(ic, jc, kc), m0(ic, jc, kc) - m0(ic - di, jc - dj, kc - dk));
                        return m0(ic, jc, kc) + 0.5 * sgn * sl;
                    };
                    if (gw)
                    {
                        // first-order upwind Gamma = m/|grad eta| of the donor cell (a cell without a level set,
                        // |grad eta| <= g_gem_min, holds no shell to give)
                        auto gam = [&](int ic, int jc, int kc) {
                            const Set::Scalar gg = vs(ic, jc, kc, AMREX_SPACEDIM);
                            return (gg > g_gem_min) ? m0(ic, jc, kc) / gg : 0.0;
                        };
                        const Set::Scalar Fp = vp * ((vp > 0.0) ? gam(i, j, k) : gam(i + di, j + dj, k + dk));
                        const Set::Scalar Fm = vm * ((vm > 0.0) ? gam(i - di, j - dj, k - dk) : gam(i, j, k));
                        div += (Fp - Fm) / DX[d];
                        continue;
                    }
                    const Set::Scalar Fp = vp * ((vp > 0.0) ? face(i, j, k, +1) : face(i + di, j + dj, k + dk, -1));
                    const Set::Scalar Fm = vm * ((vm > 0.0) ? face(i - di, j - dj, k - dk, +1) : face(i, j, k, -1));
                    div += (Fp - Fm) / DX[d];
                }
                sh(i, j, k) = m0(i, j, k) - ds * div;
            });
        }
    }
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
}

// OPTION 13 (shell.slip = 13): level-set departure point in eta space (see the shell.slip comment in Parse).
void Hydro2::ShellLevelSetSlip13(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ShellLevelSetSlip13");
    if (!(dt > 0.0)) return;
    const Set::Scalar *DX = geom[lev].CellSize();
    const Set::Scalar cut = std::max(cutoff, 1.0e-12);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(shell_mf[lev]->nGrow() >= 1 && eta_mf[lev]->nGrow() >= 2 && velocity_mf[lev]->nGrow() >= 1,
                                     "ShellLevelSetSlip13 needs ghost cells");
    // eta* = eta^n - dt u.grad eta (step-average gradient) on the valid cells, ghosts by eta's BCs (scalar).
    amrex::MultiFab ES(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, 1);
    ES.setVal(0.0);
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto eo  = eta_old_mf[lev]->const_array(mfi);
        auto vel = velocity_mf[lev]->const_array(mfi);
        auto es  = ES.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar ug = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar gm = 0.5 * ((et(i + di, j + dj, k + dk) - et(i - di, j - dj, k - dk))
                                            + (eo(i + di, j + dj, k + dk) - eo(i - di, j - dj, k - dk))) / (2.0 * DX[d]);
                ug += vel(i, j, k, d) * gm;
            }
            es(i, j, k) = eo(i, j, k) - dt * ug;
        });
    }
    FillBoundariesWithBC(lev, time, eta_bc, { &ES });
    amrex::MultiFab G0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, shell_mf[lev]->nGrow());
    amrex::MultiFab::Copy(G0, *shell_mf[lev], 0, 0, 1, shell_mf[lev]->nGrow());
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto et  = eta_mf[lev]->const_array(mfi);
        auto es  = ES.const_array(mfi);
        auto g0  = G0.const_array(mfi);
        auto sh  = shell_mf[lev]->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            const Set::Scalar ec = et(i, j, k);
            if (!(ec > cut && ec < 1.0 - cut)) return;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = 2 * (d == 0), dj = 2 * (d == 1), dk = 2 * (d == 2);
                if (!et.contains(i + di, j + dj, k + dk) || !et.contains(i - di, j - dj, k - dk)) return;
            }
            const Set::Vector ge = Numeric::Gradient(et, i, j, k, 0, DX);
            const Set::Scalar g = ge.lpNorm<2>();
            if (g < g_gem_min) return;
            const Set::Vector nv = ge / g;
            Set::Scalar kap = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Vector gp = Numeric::Gradient(et, i + di, j + dj, k + dk, 0, DX);
                const Set::Vector gm = Numeric::Gradient(et, i - di, j - dj, k - dk, 0, DX);
                const Set::Scalar gpm = gp.lpNorm<2>(), gmm = gm.lpNorm<2>();
                if (gpm < g_gem_min || gmm < g_gem_min) return;
                kap += (gp(d) / gpm - gm(d) / gmm) / (2.0 * DX[d]);
            }
            // level-set speed relative to the fluid: s = -(eta^{n+1} - eta*)/(dt |grad eta|)
            const Set::Scalar sl = -(ec - es(i, j, k)) / (dt * g);
            // upwind neighbour along the normal: per-direction donors, |n_d| geometric stencil weights
            Set::Scalar nsum = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d) nsum += std::fabs(nv(d));
            Set::Scalar gup = 0.0, eup = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                const Set::Scalar wd = std::fabs(nv(d)) / nsum;
                const bool fromminus = (sl * nv(d) > 0.0);
                const int ii = fromminus ? i - di : i + di, jj = fromminus ? j - dj : j + dj, kk = fromminus ? k - dk : k + dk;
                gup += wd * g0(ii, jj, kk);
                eup += wd * es(ii, jj, kk);
            }
            // departure point: eta*(x_dep) = eta^{n+1}_i, linear in eta* between cell i and its upwind neighbour
            const Set::Scalar den = eup - es(i, j, k);
            Set::Scalar th = (std::fabs(den) > 0.0) ? (ec - es(i, j, k)) / den : 0.0;
            if (th > 1.0)
                printf("SLIP13 t=%.6e ijk=%d %d %d eta=%.4f eta*=%.4f eta*_up=%.4f theta=%.3f sdt/dx=%.3f\n",
                       time, i, j, k, ec, es(i, j, k), eup, th, sl * dt / DX[0]);
            th = std::min(std::max(th, 0.0), 1.0);   // the departure point is inside the two-cell stencil
            sh(i, j, k) = ((1.0 - th) * g0(i, j, k) + th * gup) * std::exp(-kap * sl * dt);
        });
    }
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
}


// OPTION 15 (shell.slip = 15, 2026-11-12): the shell lives on the eta level sets.
// m = Gamma |grad eta| is the shell amount per volume.  The shell between two neighbouring level sets
// eta and eta + d eta is Gamma dA d eta / |grad eta| ... i.e. per unit eta it is Gamma per unit area, so the
// shell that crosses a face in one step is exactly the shell lying, at the OLD time, between the face and the
// old position of the level set that sits on the face at the NEW time (a Lagrangian remap in eta).  Flux
// density Gamma (-eta_t) n: no division by |grad eta|, no Courant limit, positive, conservative; Gamma
// changes only by the divergence of this flux (area law, curvature and dilatation all included).
//   Normal part, per direction d:  F_d = n_d^2 * (shell swept along d), computed in eta-space: cell c holds its
//     shell m_c dx_d spread uniformly over its OLD eta variation along d (face -> centre -> face, piecewise
//     linear), so a uniform Gamma is carried exactly whatever eta does (translation, numerical or physical
//     band widening/narrowing, compressibility).  Along d the swept length is d_eta/|d_d eta| and the swept
//     shell Gamma d_eta/|n_d|; times n_d^2 gives Gamma d_eta |n_d| = the d-component of Gamma(-eta_t dt) n.
//     The departure point is searched in the face's own neighbour cell only (reach 1): then a cell's outflows
//     are bounded by Cauchy-Schwarz, sum_d |n_d,face| V_d/(h G_c) <= 1, i.e. never more than its shell (positive
//     without a limiter).  Reach 2 let shell pass THROUGH a cell between faces of different |n_d| and drove
//     flat-eta cells negative (06 c6 slab).  In the band the level sets move ~0.1 cell per step.  A level set that is not found there
//     (eta flat: bulk, trace core) carries a vanishing share of the shell and does not move it.
//   Tangential part: m (u - (u.n) n), first-order upwind (surface flow along the interface).
// Pinch-off / a level set shrinking to a point: the level set disappears (eta extremum), no flux crosses
// the extremum, its share of shell stays where it is.  No geometry, no labelling: any shape or topology.
void Hydro2::ShellRemap15(int lev, Set::Scalar time, Set::Scalar dt)
{
    BL_PROFILE("Integrator::Hydro2::ShellRemap15");
    if (!(dt > 0.0)) return;
    const Set::Scalar *DX = geom[lev].CellSize();
    const int ng = shell_mf[lev]->nGrow();
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ng >= 3 && eta_mf[lev]->nGrow() >= 3 && eta_old_mf[lev]->nGrow() >= 3
                                     && velocity_mf[lev]->nGrow() >= 1, "ShellRemap15 needs 3 ghost cells (nghost >= 3)");
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
    amrex::MultiFab M0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, ng);
    amrex::MultiFab::Copy(M0, *shell_mf[lev], 0, 0, 1, ng);
    // Own copies of eta^n and eta^{n+1} with ghosts filled from eta's physical BCs (mirror / RADIAL / far
    // field).  eta_old's ghosts are whatever they were before this step's relaxation (option 11 note:
    // ghost values built from them pumped shell into the off-axis slab column).
    amrex::MultiFab E0(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, ng);
    amrex::MultiFab E1(shell_mf[lev]->boxArray(), shell_mf[lev]->DistributionMap(), 1, ng);
    amrex::MultiFab::Copy(E0, *eta_old_mf[lev], 0, 0, 1, 0);
    amrex::MultiFab::Copy(E1, *eta_mf[lev], 0, 0, 1, 0);
    FillBoundariesWithBC(lev, time, eta_bc, { &E0, &E1 });
    const Set::Scalar gmin = g_gem_min;
    // RADIAL slab transverse faces use the same level-set sweep as every other face (ghost eta and shell
    // come from the axis profile at the ghost radius).  The fluid flux m u_d there (tried 11-13) mixes two
    // incompatible transports: slab proto unstable, C++ c12 Gamma(.5) 30-40% over the area law on rebound.
    long nneg = 0; Set::Scalar mneg = 0.0;
    for (amrex::MFIter mfi(*shell_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto e1  = E1.const_array(mfi);
        auto e0  = E0.const_array(mfi);
        auto vel = velocity_mf[lev]->const_array(mfi);
        auto m0  = M0.const_array(mfi);
        auto sh  = shell_mf[lev]->array(mfi);
        // central gradient of eta^{n+1}
        auto grad = [=](int i, int j, int k) {
            Set::Vector g = Set::Vector::Zero();
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                g(d) = (e1(i + di, j + dj, k + dk) - e1(i - di, j - dj, k - dk)) / (2.0 * DX[d]);
            }
            return g;
        };
        // signed shell flux (per face area) from cell (iL) to cell (iL + e_d) through their common face
        auto flux = [=](int d, int i, int j, int k) -> Set::Scalar {
            const int di = (d == 0), dj = (d == 1), dk = (d == 2);
            const int iR = i + di, jR = j + dj, kR = k + dk;
            const Set::Scalar h = DX[d];
            // normal weight n_d^2 at the face
            const Set::Vector gf = 0.5 * (grad(i, j, k) + grad(iR, jR, kR));
            const Set::Scalar g2 = gf.squaredNorm();
            Set::Scalar F = 0.0;
            if (g2 > gmin * gmin)
            {
                // |n_d| at the face; each swept cell contributes its shell per unit eta (m / G_c, G_c the
                // cell's own |grad eta| from its old eta variations) times the swept eta: flux = Gamma d_eta |n_d|.
                // (m h / V_d times n_d^2 used the face n_d against the cell's d-slope: 0.8 of the dilatation
                // on the RADIAL slab sides, where eta is quadratic across the 2 transverse cells.)
                const Set::Scalar nd2 = std::fabs(gf(d)) / std::sqrt(g2);
                const Set::Scalar T  = 0.5 * (e1(i, j, k) + e1(iR, jR, kR));   // level set on the face now
                const Set::Scalar cf = 0.5 * (e0(i, j, k) + e0(iR, jR, kR));   // old value on the face
                if (T != cf && nd2 > 0.0)
                {
                    // departure point: walk away from the face on both sides through face/centre/face points
                    Set::Scalar best = 0.0; bool found = false; int nside = 0;
                    for (int s = -1; s <= 1; s += 2)
                    {
                        // cell next to the face on side s: s<0 -> (i,j,k), s>0 -> (iR,jR,kR)
                        const int bi = (s < 0) ? i : iR, bj = (s < 0) ? j : jR, bk = (s < 0) ? k : kR;
                        Set::Scalar xa = 0.0, va = cf;
                        bool hit = false;
                        for (int c = 0; c < 1 && !hit; ++c)
                        {
                            const int ci = bi + s * c * di, cj = bj + s * c * dj, ck = bk + s * c * dk;
                            const Set::Scalar vc = e0(ci, cj, ck);
                            const Set::Scalar vf = 0.5 * (vc + e0(ci + s * di, cj + s * dj, ck + s * dk));
                            const Set::Scalar xs[2] = { (c + 0.5) * h, (c + 1.0) * h };
                            const Set::Scalar vs[2] = { vc, vf };
                            for (int q = 0; q < 2 && !hit; ++q)
                            {
                                const Set::Scalar xb = xs[q], vb = vs[q];
                                if (va != vb && (va - T) * (vb - T) <= 0.0)
                                {
                                    const Set::Scalar x = xa + (T - va) / (vb - va) * (xb - xa);
                                    if (!found || x < std::fabs(best)) { best = s * x; found = true; }
                                    hit = true; ++nside;
                                }
                                xa = xb; va = vb;
                            }
                        }
                    }
                    // Departure point on BOTH sides: eta^n has an extremum at the face along d, the level set
                    // T is pinching there (converging from both sides) -- it does not cross, no shell crosses.
                    // (Picking one side pulled shell in from a mirror image: unbounded growth at a wall.)
                    if (found && nside == 1)
                    {
                        // shell swept between the face and the departure point, integrated in eta-space
                        const int s = (best < 0.0) ? -1 : 1;
                        const int bi = (s < 0) ? i : iR, bj = (s < 0) ? j : jR, bk = (s < 0) ? k : kR;
                        Set::Scalar dist = std::fabs(best), tot = 0.0;
                        for (int c = 0; c < 1 && dist > 0.0; ++c)
                        {
                            const int ci = bi + s * c * di, cj = bj + s * c * dj, ck = bk + s * c * dk;
                            const Set::Scalar vc = e0(ci, cj, ck);
                            const Set::Scalar vn = (c == 0) ? cf : 0.5 * (vc + e0(ci - s * di, cj - s * dj, ck - s * dk));
                            const Set::Scalar vfar = 0.5 * (vc + e0(ci + s * di, cj + s * dj, ck + s * dk));
                            const Set::Scalar a1 = std::fabs(vc - vn), a2 = std::fabs(vfar - vc), V = a1 + a2;
                            const Set::Scalar w = std::min(dist, h);
                            const Set::Scalar sw = (w <= 0.5 * h) ? a1 * w / (0.5 * h) : a1 + a2 * (w - 0.5 * h) / (0.5 * h);
                            const Set::Scalar mc = m0(ci, cj, ck);
                            // G_c: |grad eta^n| of cell c from its own face->centre->face variations
                            Set::Scalar Gc2 = 0.0;
                            for (int q = 0; q < AMREX_SPACEDIM; ++q)
                            {
                                const int qi = (q == 0), qj = (q == 1), qk = (q == 2);
                                const Set::Scalar ec = e0(ci, cj, ck);
                                const Set::Scalar Vq = 0.5 * std::fabs(ec - e0(ci - qi, cj - qj, ck - qk))
                                                     + 0.5 * std::fabs(e0(ci + qi, cj + qj, ck + qk) - ec);
                                Gc2 += (Vq / DX[q]) * (Vq / DX[q]);
                            }
                            (void)V;
                            if (Gc2 > 0.0) tot += mc * sw / std::sqrt(Gc2);
                            dist -= w;
                        }
                        F = nd2 * ((s < 0) ? tot : -tot);
                    }
                }
            }
            // tangential surface flow m (u - (u.n) n), first-order upwind, face velocity = mean of the cells
            auto ut = [=](int a, int b, int cc) {
                const Set::Vector g = grad(a, b, cc);
                const Set::Scalar gn = g.norm();
                Set::Scalar un = 0.0, nd = 0.0;
                if (gn > gmin)
                {
                    for (int q = 0; q < AMREX_SPACEDIM; ++q) un += vel(a, b, cc, q) * g(q) / gn;
                    nd = g(d) / gn;
                }
                return vel(a, b, cc, d) - un * nd;
            };
            const Set::Scalar uf = 0.5 * (ut(i, j, k) + ut(iR, jR, kR));
            F += dt * uf * ((uf > 0.0) ? m0(i, j, k) : m0(iR, jR, kR));
            return F;
        };
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
            Set::Scalar div = 0.0;
            for (int d = 0; d < AMREX_SPACEDIM; ++d)
            {
                const int di = (d == 0), dj = (d == 1), dk = (d == 2);
                div += (flux(d, i, j, k) - flux(d, i - di, j - dj, k - dk)) / DX[d];
            }
            sh(i, j, k) = m0(i, j, k) - div;
        });
        // diagnostic only: count negative shell (no clipping)
        const Set::Scalar mmax = 1.0;
        (void)mmax;
        for (amrex::BoxIterator bit(bx); bit.ok(); ++bit)
        {
            const amrex::IntVect iv = bit();
            const Set::Scalar v = sh(iv[0], iv[1], AMREX_SPACEDIM == 3 ? iv[2] : 0);
            if (v < 0.0)
            {
                ++nneg;
                if (v < mneg)
                {
                    mneg = v;
                    if (v < -1.0e-6 * std::fabs(m0(iv[0], iv[1], AMREX_SPACEDIM == 3 ? iv[2] : 0)) - 1.0e-30)
                        printf("SLIP15LOC t=%.6e ijk=%d %d %d m0=%.3e m=%.3e eta0(i-1..i+1)=%.4e %.4e %.4e eta1=%.4e %.4e %.4e\n",
                               time, iv[0], iv[1], AMREX_SPACEDIM == 3 ? iv[2] : 0, m0(iv[0], iv[1], iv[2]), v,
                               e0(iv[0] - 1, iv[1], iv[2]), e0(iv[0], iv[1], iv[2]), e0(iv[0] + 1, iv[1], iv[2]),
                               e1(iv[0] - 1, iv[1], iv[2]), e1(iv[0], iv[1], iv[2]), e1(iv[0] + 1, iv[1], iv[2]));
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceLongSum(nneg);
    amrex::ParallelDescriptor::ReduceRealMin(mneg);
    if (nneg > 0)
    {
        const Set::Scalar mx = shell_mf[lev]->max(0);
        if (-mneg > 1.0e-8 * mx)
            amrex::Print() << "SLIP15 t=" << time << " lev=" << lev << " negative shell cells " << nneg << " min " << mneg << " (max " << mx << ")\n";
    }
    FillBoundariesWithBC(lev, time, eta_bc, { shell_mf[lev].get() });
}

// TEMP diagnostic (2026-11-13): gas-mass ledger -- see Hydro2.H diag_gasmass.
void Hydro2::GasLedger(int lev, const char *where, Set::Scalar time)
{
    static std::map<std::string, Set::Scalar> seg;
    static std::string last;
    static Set::Scalar mlast = 0.0, m0 = -1.0, tnext = -1.0;
    const auto dx = geom[lev].CellSizeArray();
    const Set::Scalar xlo = geom[lev].ProbLo(0);
    const bool w2 = (diag_gasmass == 2);
    const bool wm = (diag_gasmass == 3);   // TEMP: radial momentum int Mx dx over x < diag_trace_x (RADIAL slab row 0)
    Set::Scalar m = 0.0;
    for (amrex::MFIter mfi(*rho_eta1_mf[lev], false); mfi.isValid(); ++mfi)
    {
        const amrex::Box &bx = mfi.validbox();
        auto r1 = rho_eta1_mf[lev]->const_array(mfi);
        auto Mm = momentum_mf[lev]->const_array(mfi);
        amrex::LoopOnCpu(bx, [&](int i, int j, int k) {
            Set::Scalar x = xlo + (i + 0.5) * dx[0];
            if (wm) { if (j == 0 && k == 0 && x < diag_trace_x) m += Mm(i, j, k, 0) * dx[0]; }
            else m += r1(i, j, k) * (w2 ? x * x : 1.0);
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(m);
    if (wm)
    {
        static std::string lastm; static Set::Scalar mprev = 0.0, tn = -1.0; static std::map<std::string, Set::Scalar> sm;
        if (!lastm.empty()) sm[lastm + ">" + where] += m - mprev;
        lastm = where; mprev = m;
        if (std::string(where) == "advance_entry" && time >= tn)
        {
            if (tn < 0.0) { tn = time + diag_gasmass_dt; sm.clear(); return; }
            tn = time + diag_gasmass_dt;
            std::string out = "MOMLEDGER t=" + std::to_string(time * 1e6) + "us intM=" + std::to_string(m);
            char b[96];
            for (auto &kv : sm) { snprintf(b, 96, " | %s %+.4e", kv.first.c_str(), kv.second); out += b; }
            Util::Message(INFO, out);
            sm.clear();
        }
        return;
    }
    if (diag_trace_x > 0.0 && time >= diag_trace_t0 && time <= diag_trace_t1)
    {
        const int ic = (int)std::floor((diag_trace_x - xlo) / dx[0]);
        for (amrex::MFIter mfi(*rho_eta1_mf[lev], false); mfi.isValid(); ++mfi)
        {
            const amrex::Box &bx = mfi.validbox();
            auto et = eta_mf[lev]->const_array(mfi);
            auto r0 = rho_eta0_mf[lev]->const_array(mfi);
            auto r1 = rho_eta1_mf[lev]->const_array(mfi);
            auto e0 = energy0_mf[lev]->const_array(mfi);
            auto e1 = energy1_mf[lev]->const_array(mfi);
            auto sh = shell_mf[lev]->const_array(mfi);
            auto M  = momentum_mf[lev]->const_array(mfi);
            for (int i = ic - 1; i <= ic + 1; ++i)
            {
                if (i < bx.smallEnd(0) || i > bx.bigEnd(0) || bx.smallEnd(1) != 0 || bx.smallEnd(2) != 0) continue;
                const Set::Scalar ge = (et(i + 1, 0, 0) - et(i - 1, 0, 0)) / (2.0 * dx[0]);
                std::printf("CTRACE t=%.9e %-15s i=%d eta=%.9f r0=%.6e r1=%.6e E0=%.6e E1=%.6e sh=%.6e Mx=%.6e gradeta=%.4e\n",
                            time, where, i, et(i, 0, 0), r0(i, 0, 0), r1(i, 0, 0), e0(i, 0, 0), e1(i, 0, 0), sh(i, 0, 0), M(i, 0, 0, 0), ge);
            }
        }
        std::fflush(stdout);
    }
    if (m0 < 0.0) { m0 = m; tnext = time + diag_gasmass_dt; }
    if (!last.empty()) seg[last + ">" + where] += (m - mlast) / m0;
    last = where; mlast = m;
    if (std::string(where) == "advance_entry" && time >= tnext)
    {
        tnext = time + diag_gasmass_dt;
        std::string out = "GASLEDGER t=" + std::to_string(time) + " m/m0=" + std::to_string(m / m0);
        char b[96];
        for (auto &kv : seg) { snprintf(b, 96, " | %s %+.3e", kv.first.c_str(), kv.second); out += b; }
        Util::Message(INFO, out);
        seg.clear();
    }
}

} // end of Integrator namespace

//#endif


