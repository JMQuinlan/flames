// 5th-order WENO reconstruction (Jiang-Shu and WENO-Z formulations).
//
// 5-cell stencil; uses all of stencil[0..4].
//
// For face at right edge of cell i (i+1/2), using cells {i-2, i-1, i, i+1, i+2}:
//   Substencil 0 (cells i-2, i-1, i):  u^0 =  1/3 Q_{i-2} - 7/6 Q_{i-1} + 11/6 Q_i
//   Substencil 1 (cells i-1, i, i+1):  u^1 = -1/6 Q_{i-1} + 5/6 Q_i    +  1/3 Q_{i+1}
//   Substencil 2 (cells i, i+1, i+2):  u^2 =  1/3 Q_i     + 5/6 Q_{i+1} -  1/6 Q_{i+2}
//
//   Smoothness (Jiang-Shu):
//     beta_0 = 13/12 (Q_{i-2} - 2 Q_{i-1} + Q_i)^2 + 1/4 (Q_{i-2} - 4 Q_{i-1} + 3 Q_i)^2
//     beta_1 = 13/12 (Q_{i-1} - 2 Q_i + Q_{i+1})^2 + 1/4 (Q_{i-1} - Q_{i+1})^2
//     beta_2 = 13/12 (Q_i - 2 Q_{i+1} + Q_{i+2})^2 + 1/4 (3 Q_i - 4 Q_{i+1} + Q_{i+2})^2
//
//   Linear weights: d_0 = 1/10, d_1 = 6/10, d_2 = 3/10
//
// Nonlinear weights:
//
//   Jiang-Shu (formulation = "jiang_shu"):
//     alpha_k = d_k / (eps_eff + beta_k)^2
//
//   WENO-Z (formulation = "weno_z", DEFAULT; Borges, Carmona, Costa,
//   Don, JCP 2008, "An improved weighted essentially non-oscillatory
//   scheme for hyperbolic conservation laws", eqs. (28)-(30)):
//     tau_5   = |beta_0 - beta_2|
//     alpha_k = d_k * (1 + (tau_5 / (eps_eff + beta_k))^2)
//   The Z formulation is provably less dissipative in smooth regions
//   (recovers the optimal upwind 5th-order linear weights at critical
//   points where Jiang-Shu drops to 3rd order) while retaining
//   shock-capturing.
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
// Reference: Jiang & Shu, "Efficient Implementation of Weighted ENO
// Schemes", J. Comput. Phys. 126 (1996) 202-228, eqs. (2.51)-(2.61).
//            Borges et al., JCP 227 (2008) 3191-3211.

#ifndef SOLVER_LOCAL_LIMITER_WENO5_H
#define SOLVER_LOCAL_LIMITER_WENO5_H

#include <algorithm>
#include <cmath>

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class WENO5 : public Limiter
{
public:
    static constexpr const char *name = "weno5";

    enum class Formulation { JiangShu, WenoZ };

    WENO5(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    WENO5(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    WENO5() { IO::ParmParse pp; pp_queryclass(*this); }

    Set::Scalar eps       = 1.0e-6;            // smoothness regularization
    Formulation formulation = Formulation::WenoZ;
    int         rel_eps   = 1;                 // eps_eff = eps * max(|Q|)^2 when nonzero

    static void Parse(WENO5 &value, IO::ParmParse &pp)
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
        const Primitive &Qm2 = stencil[RADIUS - 2];
        const Primitive &Qm1 = stencil[RADIUS - 1];
        const Primitive &Q0  = stencil[RADIUS    ];
        const Primitive &Qp1 = stencil[RADIUS + 1];
        const Primitive &Qp2 = stencil[RADIUS + 2];
        Primitive r;
        r.alpha       = scalar(Qm2.alpha,       Qm1.alpha,       Q0.alpha,       Qp1.alpha,       Qp2.alpha      );
        r.alpha_rho_0 = scalar(Qm2.alpha_rho_0, Qm1.alpha_rho_0, Q0.alpha_rho_0, Qp1.alpha_rho_0, Qp2.alpha_rho_0);
        r.alpha_rho_1 = scalar(Qm2.alpha_rho_1, Qm1.alpha_rho_1, Q0.alpha_rho_1, Qp1.alpha_rho_1, Qp2.alpha_rho_1);
        r.u           = scalar(Qm2.u,           Qm1.u,           Q0.u,           Qp1.u,           Qp2.u          );
        r.v           = scalar(Qm2.v,           Qm1.v,           Q0.v,           Qp1.v,           Qp2.v          );
        r.p_mix       = scalar(Qm2.p_mix,       Qm1.p_mix,       Q0.p_mix,       Qp1.p_mix,       Qp2.p_mix      );
        r.gamma0 = Q0.gamma0;  r.pi0 = Q0.pi0;
        r.gamma1 = Q0.gamma1;  r.pi1 = Q0.pi1;
        return r;
    }

private:
    Set::Scalar scalar(Set::Scalar qm2, Set::Scalar qm1,
                       Set::Scalar q0,
                       Set::Scalar qp1, Set::Scalar qp2) const
    {
        // Substencil polynomial reconstructions at i+1/2.
        const Set::Scalar u0 =  (1.0/3.0) * qm2 - (7.0/6.0) * qm1 + (11.0/6.0) * q0;
        const Set::Scalar u1 = -(1.0/6.0) * qm1 + (5.0/6.0) * q0  + (1.0/3.0)  * qp1;
        const Set::Scalar u2 =  (1.0/3.0) * q0  + (5.0/6.0) * qp1 - (1.0/6.0)  * qp2;

        // Jiang-Shu smoothness indicators.
        const Set::Scalar t0a = qm2 - 2.0 * qm1 + q0;
        const Set::Scalar t0b = qm2 - 4.0 * qm1 + 3.0 * q0;
        const Set::Scalar b0  = (13.0/12.0) * t0a * t0a + 0.25 * t0b * t0b;

        const Set::Scalar t1a = qm1 - 2.0 * q0 + qp1;
        const Set::Scalar t1b = qm1 - qp1;
        const Set::Scalar b1  = (13.0/12.0) * t1a * t1a + 0.25 * t1b * t1b;

        const Set::Scalar t2a = q0 - 2.0 * qp1 + qp2;
        const Set::Scalar t2b = 3.0 * q0 - 4.0 * qp1 + qp2;
        const Set::Scalar b2  = (13.0/12.0) * t2a * t2a + 0.25 * t2b * t2b;

        // Per-variable eps regularization. With rel_eps = true, eps becomes
        // a relative tolerance: eps_eff = eps * max(|Q_k|)^2. This makes a
        // single eps value meaningful for primitives whose magnitudes span
        // many orders (alpha vs p_mix in a multiphase compressible code).
        constexpr Set::Scalar tiny = 1.0e-30;
        Set::Scalar eps_eff = eps;
        if (rel_eps != 0)
        {
            const Set::Scalar mag = std::max({std::abs(qm2),
                                              std::abs(qm1),
                                              std::abs(q0),
                                              std::abs(qp1),
                                              std::abs(qp2),
                                              tiny});
            eps_eff = eps * mag * mag;
        }
        // Floor so the division below is never by zero even for a
        // constant zero stencil.
        eps_eff = std::max(eps_eff, tiny);

        Set::Scalar a0, a1, a2;
        if (formulation == Formulation::WenoZ)
        {
            // WENO-Z (Borges et al. 2008). For WENO5 the global indicator
            // tau_5 = |beta_0 - beta_2| measures the disagreement of the
            // most upwind- and downwind-biased substencils. tau_5 vanishes
            // at smooth extrema, restoring optimal linear weights and
            // recovering full 5th-order accuracy where Jiang-Shu drops
            // to 3rd at critical points.
            const Set::Scalar tau = std::abs(b0 - b2);
            const Set::Scalar r0  = tau / (eps_eff + b0);
            const Set::Scalar r1  = tau / (eps_eff + b1);
            const Set::Scalar r2  = tau / (eps_eff + b2);
            a0 = (1.0 / 10.0) * (1.0 + r0 * r0);
            a1 = (6.0 / 10.0) * (1.0 + r1 * r1);
            a2 = (3.0 / 10.0) * (1.0 + r2 * r2);
        }
        else
        {
            // Classical Jiang-Shu nonlinear weights.
            a0 = (1.0 / 10.0) / ((eps_eff + b0) * (eps_eff + b0));
            a1 = (6.0 / 10.0) / ((eps_eff + b1) * (eps_eff + b1));
            a2 = (3.0 / 10.0) / ((eps_eff + b2) * (eps_eff + b2));
        }

        const Set::Scalar asum = a0 + a1 + a2;
        return (a0 * u0 + a1 * u1 + a2 * u2) / asum;
    }
};

}}}

#endif
