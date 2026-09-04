// 6-equation diffuse-interface reconstruction (limiter / WENO) framework.
//
// ARCHITECTURE NOTE -- READ BEFORE EDITING:
//
//   Sch20 §3.2 prescribes PRIMITIVE-variable reconstruction for the
//   6-eq diffuse-interface model (reconstructing conservatives produces
//   spurious oscillations at the diffuse interface).  We use the
//   ROBUST primitive set:
//       (alpha, alpha_rho_0, alpha_rho_1, u, v, p_mix)
//
//   Why partial densities + mixture pressure (not pure-phase rho_k, p_k):
//
//   1. Pure-phase rho_k = alpha_rho_k / alpha_k goes singular as the
//      "other" phase is essentially absent (alpha_k -> alpha_floor).
//      Tiny floor differences between cells produce HUGE swings in the
//      reconstructed face value -- which the limiter then amplifies.
//      Garrick test post-mortem: liquid pressure oscillated ~1000x the
//      background because Tammann stiffness amplifies cell-local p_k
//      perturbations, while gas was clean (low gamma, no pi).
//
//   2. p_mix = alpha_1 p_1 + alpha_2 p_2 (Sch20 eq. 8) is well-defined
//      everywhere.  After RelaxAndReinit (every step), p_1 = p_2 = p_mix
//      exactly, so reconstructing on p_mix loses NO information that
//      isn't already enforced by the stiff relaxation step.
//
//   3. alpha_rho_k (partial density) is conservative -- zero in pure-
//      other-phase cells, smoothly varies across the interface, no
//      EOS inversion needed for State <-> Primitive.
//
//   4. ToState rebuilds E_k = alpha_k (p_mix + gamma_k pi_k)/(gamma_k - 1)
//      directly (Sau09 III.5).  No divide-by-alpha, no pressure inversion.
//
//   See feedback_limiter_eos_divide_floor.md and the partial-density
//   discussion in MEMORY.md for the full rationale.
//
//   All concrete limiters share one signature:
//       Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const
//   returning the face value at the RIGHT edge of the center cell
//   (stencil[RADIUS]).  For the LEFT-edge face value, call with the
//   stencil reversed.
//
//   Lower-order classes ignore outer cells; only WENO5 uses all 5.

#ifndef SOLVER_LOCAL_LIMITER_LIMITER_H
#define SOLVER_LOCAL_LIMITER_LIMITER_H

#include <algorithm>
#include <cmath>

#include "Set/Base.H"
#include "Set/Set.H"
#include "Solver/Local/FluidRiemann/Riemann.H"

namespace Solver
{
namespace Local
{
namespace Limiter
{

/// Robust primitive set for 6-eq reconstruction.
/// (alpha, alpha_rho_0, alpha_rho_1, u, v, p_mix).
/// EOS metadata (gamma_k, pi_k) is constant across the stencil and
/// not reconstructed.
struct Primitive
{
    Set::Scalar alpha       = 0.0;  // alpha_1 (volume fraction of phase 0)
    Set::Scalar alpha_rho_0 = 0.0;  // PURE density rho_1 (slot name legacy; see ToPrimitive)
    Set::Scalar alpha_rho_1 = 0.0;  // PURE density rho_2 (slot name legacy; see ToPrimitive)
    Set::Scalar u           = 0.0;  // mixture velocity, normal component
    Set::Scalar v           = 0.0;  // mixture velocity, tangent component
    Set::Scalar w           = 0.0;  // mixture velocity, 2nd tangent (3D; 0 in 2D)
    Set::Scalar p_mix       = 0.0;  // mixture pressure (Sch20 eq. 8)
    // Metadata (NOT reconstructed; carried for back-conversion):
    Set::Scalar gamma0      = 0.0;
    Set::Scalar pi0         = 0.0;
    Set::Scalar gamma1      = 0.0;
    Set::Scalar pi1         = 0.0;
};

/// Component-wise arithmetic on the reconstructed fields only.
/// Metadata is taken from the LHS / first arg.
inline Primitive operator-(const Primitive &a, const Primitive &b)
{
    return { a.alpha       - b.alpha,
             a.alpha_rho_0 - b.alpha_rho_0,
             a.alpha_rho_1 - b.alpha_rho_1,
             a.u           - b.u,
             a.v           - b.v,
             a.w           - b.w,
             a.p_mix       - b.p_mix,
             a.gamma0, a.pi0, a.gamma1, a.pi1 };
}
inline Primitive operator+(const Primitive &a, const Primitive &b)
{
    return { a.alpha       + b.alpha,
             a.alpha_rho_0 + b.alpha_rho_0,
             a.alpha_rho_1 + b.alpha_rho_1,
             a.u           + b.u,
             a.v           + b.v,
             a.w           + b.w,
             a.p_mix       + b.p_mix,
             a.gamma0, a.pi0, a.gamma1, a.pi1 };
}
inline Primitive operator*(Set::Scalar s, const Primitive &p)
{
    return { s*p.alpha, s*p.alpha_rho_0, s*p.alpha_rho_1,
             s*p.u, s*p.v, s*p.w, s*p.p_mix,
             p.gamma0, p.pi0, p.gamma1, p.pi1 };
}
inline Primitive operator*(const Primitive &p, Set::Scalar s) { return s * p; }

/// State (conservative, 6-eq) -> Primitive.
/// p_mix is computed from per-phase pressures via Sch20 eq. 8.
/// We invert each E_k -> p_k WITH a tiny div-floor so that the
/// per-phase pressures are well-defined even in pure-other-phase
/// cells (where alpha_k = alpha_floor).  Then p_mix = sum alpha_k p_k.
/// After RelaxAndReinit p_0 = p_1 = p_mix exactly, so this is
/// information-preserving.
inline Primitive
ToPrimitive(const Solver::Local::FluidRiemann::State &s, Set::Scalar /*small*/)
{
    constexpr Set::Scalar alpha_floor = 1.0e-12;
    constexpr Set::Scalar DIV_FLOOR   = 1.0e-30;
    // Cap on the reconstructed pure densities (see comment below).
    constexpr Set::Scalar RHO_PURE_CAP = 1.0e4;
    Primitive p;
    p.alpha       = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
    Set::Scalar a2 = 1.0 - p.alpha;
    // Mass channel: reconstruct PURE densities rho_k = (alpha rho)_k/alpha_k,
    // NOT the partial densities.  The partial densities jump 1000x across a
    // few band cells; component-wise high-order reconstruction of that
    // profile under/overshoots, the post-stage positivity clip converts each
    // undershoot into CREATED MASS, and a mass "snowplow" parks at any
    // moving band edge (UT_ADV: +150% total gas mass in 8 ms of pure
    // translation; first-order Godunov, which cannot overshoot, is exact to
    // 0.00%).  The pure densities are ~constant fields (all tanh steepness
    // lives in alpha, which is bounded), so the reconstructed product
    // alpha_face * rho_pure_face is smooth, non-negative by construction,
    // and telescoping-consistent.  Slot names kept (alpha_rho_*) so the
    // concrete limiters need no changes; ToState rebuilds the partials with
    // the SAME clamped alpha, making the Godunov (cell-center) roundtrip an
    // exact identity.
    //
    // RHO_PURE_CAP: bound the RESULT of the divide, never the divisor.
    // For CONSISTENT tail cells arh_k = alpha_k rho_k with both -> 0, so
    // the raw ratio arh/alpha stays smooth at ~rho_true all the way down --
    // that smoothness IS the snowplow fix, and flooring the divisor
    // destroys it: dividing by max(alpha, 1e-6) ramps the field from
    // rho_true to ~0 across the outer tail, WENO overshoots the ramp, and
    // the snowplow returns (measured: UT_ADV m1 drift +212%/period with a
    // 1e-6 divisor floor vs 0.00% with the raw ratio).  The ONLY
    // pathological case is an INCONSISTENT cell -- residual advected mass
    // coexisting with alpha at the 1e-12 clamp (eta clamped to exactly 0/1
    // while the positivity clip only zeroes NEGATIVE mass) -- whose fake
    // rho_pure is 1e4..1e6 kg/m^3.  Cap the ratio at 1e4 (physical
    // densities are <= ~1.5e3 even for shock-compressed Tammann water):
    // consistent cells are never capped, so the cell-center roundtrip
    // through ToState stays an exact identity everywhere it matters, and
    // the capped cells' states were unphysical to begin with.
    p.alpha_rho_0 = std::min(s.alpha_rho_0 / p.alpha, RHO_PURE_CAP);   // rho_0 pure
    p.alpha_rho_1 = std::min(s.alpha_rho_1 / a2,      RHO_PURE_CAP);   // rho_1 pure
    // Mixture velocity (one velocity for all phases in the equilibrium-
    // velocity 6-eq model; Sch20 eq. 13).
    Set::Scalar rho = s.alpha_rho_0 + s.alpha_rho_1;
    p.u = s.M_normal   / std::max(rho, DIV_FLOOR);
    p.v = s.M_tangent  / std::max(rho, DIV_FLOOR);
    p.w = s.M_tangent2 / std::max(rho, DIV_FLOOR);  // 2nd tangent (3D; 0 in 2D)
    // Per-phase pressures (inlined, not via EOS::PhasicPressureFromEnergy
    // which uses `small` as divide-floor and would re-introduce the
    // 10000x discontinuity in pure-other-phase cells).
    Set::Scalar p0 = (s.gamma0 - 1.0) * s.E0 / std::max(p.alpha, DIV_FLOOR) - s.gamma0 * s.pi0;
    Set::Scalar p1 = (s.gamma1 - 1.0) * s.E1 / std::max(a2,      DIV_FLOOR) - s.gamma1 * s.pi1;
    // SG positivity floor (Sch20 §3 / Sau09 §3): p + gamma*pi > 0.
    p0 = std::max(p0, -s.pi0 + DIV_FLOOR);
    p1 = std::max(p1, -s.pi1 + DIV_FLOOR);
    // Mixture pressure (Sch20 eq. 8).  Smooth across the diffuse
    // interface even when individual p_k are wild in pure-other-phase
    // cells, because alpha_k weights them down.
    p.p_mix  = p.alpha * p0 + a2 * p1;
    p.gamma0 = s.gamma0;  p.pi0 = s.pi0;
    p.gamma1 = s.gamma1;  p.pi1 = s.pi1;
    return p;
}

/// Primitive -> State (conservative).
/// Uses Sch20 eq. 8 / Sau09 III.5 to rebuild per-phase E_k from p_mix
/// directly (assumes mechanical equilibrium at the face -- which is what
/// the relaxation step enforces every timestep).  No EOS inversion, no
/// divide-by-alpha.
inline Solver::Local::FluidRiemann::State
ToState(const Primitive &p, Set::Scalar /*small*/)
{
    constexpr Set::Scalar alpha_floor = 1.0e-12;
    constexpr Set::Scalar DIV_FLOOR   = 1.0e-30;
    Solver::Local::FluidRiemann::State s;
    Set::Scalar a1 = std::min(std::max(p.alpha, alpha_floor), 1.0 - alpha_floor);
    Set::Scalar a2 = 1.0 - a1;
    // Rebuild partial densities from the reconstructed PURE densities and
    // the clamped face alpha (see ToPrimitive): non-negative by
    // construction, and an exact roundtrip when the limiter returns the
    // cell-center value (both directions use the same clamped alpha; the
    // ToPrimitive RHO_PURE_CAP only fires on unphysical residual-mass
    // cells, so it never breaks the roundtrip of a physical state).
    s.alpha       = a1;
    s.alpha_rho_0 = a1 * std::max(p.alpha_rho_0, 0.0);
    s.alpha_rho_1 = a2 * std::max(p.alpha_rho_1, 0.0);
    Set::Scalar rho = s.alpha_rho_0 + s.alpha_rho_1;
    s.M_normal    = rho * p.u;
    s.M_tangent   = rho * p.v;
    s.M_tangent2  = rho * p.w;   // 2nd tangent (3D; 0 in 2D)
    // E_k = alpha_k (p_mix + gamma_k pi_k) / (gamma_k - 1).
    Set::Scalar num0 = std::max(p.p_mix + p.gamma0 * p.pi0, DIV_FLOOR);
    Set::Scalar num1 = std::max(p.p_mix + p.gamma1 * p.pi1, DIV_FLOOR);
    s.E0          = a1 * num0 / (p.gamma0 - 1.0);
    s.E1          = a2 * num1 / (p.gamma1 - 1.0);
    s.E_total     = s.E0 + s.E1 + 0.5 * rho * (p.u*p.u + p.v*p.v + p.w*p.w);
    s.gamma0 = p.gamma0;  s.pi0 = p.pi0;
    s.gamma1 = p.gamma1;  s.pi1 = p.pi1;
    return s;
}

/// Limiter base class.  All concrete classes return the face value at
/// the RIGHT edge of stencil[RADIUS] (the center cell).
class Limiter
{
public:
    static constexpr int STENCIL_SIZE = 5;
    static constexpr int RADIUS       = 2;

    virtual Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const = 0;
    virtual ~Limiter() = default;
};

}}}

#endif
