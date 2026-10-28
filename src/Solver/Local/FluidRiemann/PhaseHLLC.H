// Single-phase HLLC flux for ONE phase of the 7-equation (two-velocity,
// two-pressure) model.
//
// The 7-equation model (Baer-Nunziato 1986; Saurel-Abgrall JCP 150, 1999)
// carries, per phase k, the conservative vector
//     U_k = alpha_k ( rho_k, rho_k u_k, rho_k E_k )
// with flux
//     F_k = alpha_k ( rho_k u_k, rho_k u_k u_k + p_k, u_k (rho_k E_k + p_k) )
// plus the non-conservative interfacial terms p_I grad(alpha_k) and
// p_I u_I . grad(alpha_k), which Hydro2::RHS adds cell-wise.
//
// The face flux is written  F_k = alpha_face,k * f_k,  where f_k is the plain
// Euler HLLC flux of phase k's pure state (rho_k, u_k, p_k) and alpha_face,k
// is one face volume fraction shared by both phases (alpha_face for phase 0,
// 1 - alpha_face for phase 1, upwinded on the interface velocity).  Sharing it
// gives two properties exactly:
//   * the two non-conservative terms cancel, so mixture mass, momentum and
//     total energy are discretely conservative;
//   * uniform p and u are preserved for any alpha profile (Abgrall 1996):
//     with p_I = p the pressure part of -div(alpha_f p) cancels against
//     p_I grad_h(alpha_f) term by term.
// This routine returns f_k (per unit volume fraction).  Acoustic speeds are the
// Davis estimates from phase k's own sound speed; star states follow Toro,
// "Riemann Solvers and Numerical Methods for Fluid Dynamics", sec. 10.4.
//
// CONTACT SPEED.  The contact is PRESCRIBED (S_star), not phase k's own HLLC
// estimate: Hydro2 passes the mixture contact speed u* -- the speed alpha_face
// is upwinded on -- plus phase k's share of the face drift velocity.  The face
// flux alpha_face f_k must take alpha and phase k's material from the SAME
// side of the face.  With each phase's own contact estimate the two choices
// disagree wherever the normal velocity is ~0 (the phase estimates and u*
// differ by solver error only): phase-k mass then leaves a cell weighted with
// the NEIGHBOUR's alpha, an anti-diffusive (downwind) alpha flux.  Measured on
// the two-fluid Couette deck: a 4-cell interface wave growing ~x10 every 2 s
// until NaN at t = 31.  With S_star = u* + drift share, phases in velocity
// equilibrium share one contact exactly (as the 6-equation HLLC does), and a
// genuine drift still carries phase k through the interface at its own speed.
// The acoustic speeds are widened to bracket S_star, so rho* > 0.
//
// EOS: stiffened gas (Tammann)  p = (gamma - 1) rho e - gamma pi.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_PHASEHLLC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_PHASEHLLC_H

#include <algorithm>
#include <cmath>
#include "Set/Base.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Pure-phase face state (normal + two tangential velocity components).
struct PhaseState
{
    Set::Scalar rho = 0.0;
    Set::Scalar un  = 0.0;
    Set::Scalar ut  = 0.0;
    Set::Scalar ut2 = 0.0;   // 3D second tangent (0 in 2D)
    Set::Scalar p   = 0.0;
};

/// Pure-phase flux per unit volume fraction.
struct PhaseFlux
{
    Set::Scalar mass   = 0.0;   // rho u_n
    Set::Scalar mom_n  = 0.0;   // rho u_n^2 + p
    Set::Scalar mom_t  = 0.0;   // rho u_n u_t
    Set::Scalar mom_t2 = 0.0;   // rho u_n u_t2
    Set::Scalar energy = 0.0;   // u_n (rho E + p)
    Set::Scalar S_M    = 0.0;   // phase contact speed (diagnostic)
};

inline PhaseFlux PhaseHLLC(const PhaseState &L, const PhaseState &R,
                           Set::Scalar gamma, Set::Scalar pi, Set::Scalar small,
                           Set::Scalar S_star)
{
    const Set::Scalar rL = std::max(L.rho, small), rR = std::max(R.rho, small);
    // Stiffened-gas positivity p + pi > 0 (callers floor, this only guards).
    const Set::Scalar pL = std::max(L.p, -pi + small), pR = std::max(R.p, -pi + small);
    const Set::Scalar cL = std::sqrt(std::max(gamma * (pL + pi) / rL, 0.0));
    const Set::Scalar cR = std::sqrt(std::max(gamma * (pR + pi) / rR, 0.0));

    // Total energy per volume:  rho E = (p + gamma pi)/(gamma - 1) + rho |u|^2 / 2.
    auto rhoE = [&](Set::Scalar r, Set::Scalar p, const PhaseState &s) {
        return (p + gamma * pi) / (gamma - 1.0)
             + 0.5 * r * (s.un * s.un + s.ut * s.ut + s.ut2 * s.ut2);
    };
    const Set::Scalar EL = rhoE(rL, pL, L), ER = rhoE(rR, pR, R);

    const Set::Scalar S_M = S_star;
    const Set::Scalar S_L = std::min(std::min(L.un - cL, R.un - cR), S_M);
    const Set::Scalar S_R = std::max(std::max(L.un + cL, R.un + cR), S_M);

    auto physical = [&](Set::Scalar r, Set::Scalar p, Set::Scalar E, const PhaseState &s) {
        PhaseFlux f;
        f.mass   = r * s.un;
        f.mom_n  = r * s.un * s.un + p;
        f.mom_t  = r * s.un * s.ut;
        f.mom_t2 = r * s.un * s.ut2;
        f.energy = s.un * (E + p);
        return f;
    };
    // F*_K = F_K + S_K (U*_K - U_K)
    auto star = [&](Set::Scalar r, Set::Scalar p, Set::Scalar E, const PhaseState &s, Set::Scalar S) {
        PhaseFlux f = physical(r, p, E, s);
        Set::Scalar dS = S - S_M;
        if (std::abs(dS) < small) dS = (dS >= 0.0 ? small : -small);
        const Set::Scalar fac = r * (S - s.un) / dS;   // rho*_K
        Set::Scalar dsu = r * (S - s.un);
        if (std::abs(dsu) < small) dsu = (dsu >= 0.0 ? small : -small);
        const Set::Scalar Estar = fac * (E / r + (S_M - s.un) * (S_M + p / dsu));
        f.mass   += S * (fac              - r);
        f.mom_n  += S * (fac * S_M        - r * s.un);
        f.mom_t  += S * (fac * s.ut       - r * s.ut);
        f.mom_t2 += S * (fac * s.ut2      - r * s.ut2);
        f.energy += S * (Estar            - E);
        return f;
    };

    PhaseFlux f;
    if (0.0 <= S_L)       f = physical(rL, pL, EL, L);
    else if (0.0 <= S_M)  f = star(rL, pL, EL, L, S_L);
    else if (0.0 <= S_R)  f = star(rR, pR, ER, R, S_R);
    else                  f = physical(rR, pR, ER, R);
    f.S_M = S_M;
    return f;
}

}
}
}

#endif
