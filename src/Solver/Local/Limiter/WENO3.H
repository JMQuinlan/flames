// 3rd-order WENO reconstruction (Jiang-Shu and WENO-Z formulations).
//
// 3-cell stencil; outer cells of the 5-cell window are ignored.
//
// For face at right edge of cell i (i+1/2), using cells {i-1, i, i+1}:
//   Substencil 0 (cells i-1, i):  u^0 = -1/2 Q_{i-1} + 3/2 Q_i
//   Substencil 1 (cells i, i+1):  u^1 =  1/2 Q_i     + 1/2 Q_{i+1}
//   Smoothness:  beta_0 = (Q_i - Q_{i-1})^2,   beta_1 = (Q_{i+1} - Q_i)^2
//   Linear weights: d_0 = 1/3, d_1 = 2/3
//
// Nonlinear weights:
//
//   Jiang-Shu (formulation = "jiang_shu"):
//     alpha_k = d_k / (eps_eff + beta_k)^2
//
//   WENO-Z (formulation = "weno_z", DEFAULT; Borges, Carmona, Costa,
//   Don, JCP 2008, "An improved weighted essentially non-oscillatory
//   scheme for hyperbolic conservation laws", eqs. (28)-(30)):
//     tau   = |beta_0 - beta_1|                      (global indicator)
//     alpha_k = d_k * (1 + (tau / (eps_eff + beta_k))^2)
//   The Z formulation is provably less dissipative in smooth regions
//   (recovers the optimal upwind 3rd-order linear weights to one
//   order higher than Jiang-Shu) while retaining shock-capturing.
//
// eps regularization:
//   eps is dimensionally (variable)^2.  For a multiphase code the
//   primitives span many orders of magnitude (alpha ~ O(1),
//   p_mix ~ O(1e6)), so a single absolute eps cannot serve all
//   variables.  When rel_eps = true (DEFAULT) the regularization
//   becomes eps_eff = eps * max(|Q_k|)^2 over the stencil, so eps
//   plays the role of a RELATIVE tolerance and the same numerical
//   value (1e-6) is meaningful across all primitives.  When
//   rel_eps = false the historical absolute Jiang-Shu behavior is
//   restored.
//
// Input parameters:
//   Limiter.eps         (Real)   default 1e-6
//   Limiter.formulation (string) "jiang_shu" | "weno_z" (default "weno_z")
//   Limiter.rel_eps     (bool)   default true
//
// Reference: Shu, "Essentially Non-Oscillatory and Weighted ENO Schemes
// for Hyperbolic Conservation Laws", ICASE 97-65 (1997).
//            Borges et al., JCP 227 (2008) 3191-3211.

#ifndef SOLVER_LOCAL_LIMITER_WENO3_H
#define SOLVER_LOCAL_LIMITER_WENO3_H

#include <algorithm>
#include <cmath>

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class WENO3 : public Limiter
{
public:
    static constexpr const char *name = "weno3";

    enum class Formulation { JiangShu, WenoZ };

    WENO3(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    WENO3(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    WENO3() { IO::ParmParse pp; pp_queryclass(*this); }

    Set::Scalar eps       = 1.0e-6;            // smoothness regularization
    Formulation formulation = Formulation::WenoZ;
    int         rel_eps   = 1;                 // eps_eff = eps * max(|Q|)^2 when nonzero

    static void Parse(WENO3 &value, IO::ParmParse &pp)
    {
        pp.query_default("eps", value.eps, 1.0e-6);
        pp.query_default("rel_eps", value.rel_eps, true);
        std::string formstr = "weno_z";
        pp.query_validate("formulation", formstr, {"jiang_shu", "weno_z"});
        value.formulation = (formstr == "jiang_shu")
                            ? Formulation::JiangShu
                            : Formulation::WenoZ;
    }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        const Primitive &Qm1 = stencil[RADIUS - 1];
        const Primitive &Q0  = stencil[RADIUS    ];
        const Primitive &Qp1 = stencil[RADIUS + 1];
        Primitive r;
        r.alpha       = scalar(Qm1.alpha,       Q0.alpha,       Qp1.alpha      );
        r.alpha_rho_0 = scalar(Qm1.alpha_rho_0, Q0.alpha_rho_0, Qp1.alpha_rho_0);
        r.alpha_rho_1 = scalar(Qm1.alpha_rho_1, Q0.alpha_rho_1, Qp1.alpha_rho_1);
        r.u           = scalar(Qm1.u,           Q0.u,           Qp1.u          );
        r.v           = scalar(Qm1.v,           Q0.v,           Qp1.v          );
        r.w           = scalar(Qm1.w,           Q0.w,           Qp1.w          );
        r.p_mix       = scalar(Qm1.p_mix,       Q0.p_mix,       Qp1.p_mix      );
        r.gamma0 = Q0.gamma0;  r.pi0 = Q0.pi0;
        r.gamma1 = Q0.gamma1;  r.pi1 = Q0.pi1;
        return r;
    }

private:
    Set::Scalar scalar(Set::Scalar qm1, Set::Scalar q0, Set::Scalar qp1) const
    {
        // Substencil polynomial reconstructions at the right edge of Q0.
        const Set::Scalar u0 = -0.5 * qm1 + 1.5 * q0;
        const Set::Scalar u1 =  0.5 * q0  + 0.5 * qp1;

        // Smoothness indicators (Jiang-Shu, eqs. (3.10)-(3.11)).
        const Set::Scalar b0 = (q0  - qm1) * (q0  - qm1);
        const Set::Scalar b1 = (qp1 - q0)  * (qp1 - q0);

        // Per-variable eps regularization. With rel_eps = true, eps becomes
        // a relative tolerance: eps_eff = eps * max(|Q_k|)^2. This makes a
        // single eps value meaningful for primitives whose magnitudes span
        // many orders (alpha vs p_mix in a multiphase compressible code).
        constexpr Set::Scalar tiny = 1.0e-30;
        Set::Scalar eps_eff = eps;
        if (rel_eps != 0)
        {
            const Set::Scalar mag = std::max({std::abs(qm1),
                                              std::abs(q0),
                                              std::abs(qp1),
                                              tiny});
            eps_eff = eps * mag * mag;
        }
        // Floor so the division below is never by zero even for a
        // constant zero stencil.
        eps_eff = std::max(eps_eff, tiny);

        Set::Scalar a0, a1;
        if (formulation == Formulation::WenoZ)
        {
            // WENO-Z (Borges et al. 2008). The global indicator
            // tau = |beta_0 - beta_1| measures the disagreement between
            // the two substencils and is ZERO at smooth extrema, which
            // recovers the optimal linear weights and avoids the
            // Jiang-Shu order loss at critical points.
            const Set::Scalar tau = std::abs(b0 - b1);
            const Set::Scalar r0  = tau / (eps_eff + b0);
            const Set::Scalar r1  = tau / (eps_eff + b1);
            a0 = (1.0 / 3.0) * (1.0 + r0 * r0);
            a1 = (2.0 / 3.0) * (1.0 + r1 * r1);
        }
        else
        {
            // Classical Jiang-Shu nonlinear weights.
            a0 = (1.0 / 3.0) / ((eps_eff + b0) * (eps_eff + b0));
            a1 = (2.0 / 3.0) / ((eps_eff + b1) * (eps_eff + b1));
        }

        const Set::Scalar asum = a0 + a1;
        return (a0 * u0 + a1 * u1) / asum;
    }
};

}}}

#endif
