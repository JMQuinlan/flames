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
    Set::Scalar alpha_rho_0 = 0.0;  // partial density 0  (= alpha_1 rho_1)
    Set::Scalar alpha_rho_1 = 0.0;  // partial density 1  (= alpha_2 rho_2)
    Set::Scalar u           = 0.0;  // mixture velocity, normal component
    Set::Scalar v           = 0.0;  // mixture velocity, tangent component
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
             a.p_mix       + b.p_mix,
             a.gamma0, a.pi0, a.gamma1, a.pi1 };
}
inline Primitive operator*(Set::Scalar s, const Primitive &p)
{
    return { s*p.alpha, s*p.alpha_rho_0, s*p.alpha_rho_1,
             s*p.u, s*p.v, s*p.p_mix,
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
    Primitive p;
    p.alpha       = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
    Set::Scalar a2 = 1.0 - p.alpha;
    // Partial densities: NO division, NO floor.  Conservative quantities.
    p.alpha_rho_0 = s.alpha_rho_0;
    p.alpha_rho_1 = s.alpha_rho_1;
    // Mixture velocity (one velocity for all phases in the equilibrium-
    // velocity 6-eq model; Sch20 eq. 13).
    Set::Scalar rho = s.alpha_rho_0 + s.alpha_rho_1;
    p.u = s.M_normal  / std::max(rho, DIV_FLOOR);
    p.v = s.M_tangent / std::max(rho, DIV_FLOOR);
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
    // Limiter overshoot can drive partial densities slightly negative;
    // clamp to 0 (NOT to small=1e-8 -- that re-introduces the floor
    // distortion in pure-other-phase cells).
    s.alpha       = a1;
    s.alpha_rho_0 = std::max(p.alpha_rho_0, 0.0);
    s.alpha_rho_1 = std::max(p.alpha_rho_1, 0.0);
    Set::Scalar rho = s.alpha_rho_0 + s.alpha_rho_1;
    s.M_normal    = rho * p.u;
    s.M_tangent   = rho * p.v;
    // E_k = alpha_k (p_mix + gamma_k pi_k) / (gamma_k - 1).
    Set::Scalar num0 = std::max(p.p_mix + p.gamma0 * p.pi0, DIV_FLOOR);
    Set::Scalar num1 = std::max(p.p_mix + p.gamma1 * p.pi1, DIV_FLOOR);
    s.E0          = a1 * num0 / (p.gamma0 - 1.0);
    s.E1          = a2 * num1 / (p.gamma1 - 1.0);
    s.E_total     = s.E0 + s.E1 + 0.5 * rho * (p.u*p.u + p.v*p.v);
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
