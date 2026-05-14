// 5th-order WENO (Jiang-Shu) reconstruction.
// 5-cell stencil; uses all of stencil[0..4].
//
// For face at right edge of cell i (i+1/2), using cells {i-2, i-1, i, i+1, i+2}:
//   Substencil 0 (cells i-2, i-1, i):  u^0 =  1/3 Q_{i-2} - 7/6 Q_{i-1} + 11/6 Q_i
//   Substencil 1 (cells i-1, i, i+1):  u^1 = -1/6 Q_{i-1} + 5/6 Q_i    +  1/3 Q_{i+1}
//   Substencil 2 (cells i, i+1, i+2):  u^2 =  1/3 Q_i     + 5/6 Q_{i+1} -  1/6 Q_{i+2}
//   Smoothness (Jiang-Shu):
//       beta_0 = 13/12 (Q_{i-2} - 2 Q_{i-1} + Q_i)^2 + 1/4 (Q_{i-2} - 4 Q_{i-1} + 3 Q_i)^2
//       beta_1 = 13/12 (Q_{i-1} - 2 Q_i + Q_{i+1})^2 + 1/4 (Q_{i-1} - Q_{i+1})^2
//       beta_2 = 13/12 (Q_i - 2 Q_{i+1} + Q_{i+2})^2 + 1/4 (3 Q_i - 4 Q_{i+1} + Q_{i+2})^2
//   Linear weights: d_0 = 1/10, d_1 = 6/10, d_2 = 3/10
//   Nonlinear weights: alpha_k = d_k / (eps + beta_k)^2,  omega_k = alpha_k / Sum(alpha_j)
//   Reconstructed:  u_face = omega_0 u^0 + omega_1 u^1 + omega_2 u^2
//
// Reference: Jiang & Shu, "Efficient Implementation of Weighted ENO
// Schemes", J. Comput. Phys. 126 (1996) 202-228, eqs. (2.51)-(2.61).

#ifndef SOLVER_LOCAL_LIMITER_WENO5_H
#define SOLVER_LOCAL_LIMITER_WENO5_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class WENO5 : public Limiter
{
public:
    static constexpr const char *name = "weno5";

    WENO5(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    WENO5(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    WENO5() { IO::ParmParse pp; pp_queryclass(*this); }

    Set::Scalar eps = 1.0e-6;  // Jiang-Shu smoothness regularization

    static void Parse(WENO5 &value, IO::ParmParse &pp)
    {
        pp.query_default("eps", value.eps, 1.0e-6);
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

        // Nonlinear weights.
        const Set::Scalar a0 = (1.0/10.0) / ((eps + b0) * (eps + b0));
        const Set::Scalar a1 = (6.0/10.0) / ((eps + b1) * (eps + b1));
        const Set::Scalar a2 = (3.0/10.0) / ((eps + b2) * (eps + b2));
        const Set::Scalar asum = a0 + a1 + a2;
        const Set::Scalar w0 = a0 / asum;
        const Set::Scalar w1 = a1 / asum;
        const Set::Scalar w2 = a2 / asum;

        return w0 * u0 + w1 * u1 + w2 * u2;
    }
};

}}}

#endif
