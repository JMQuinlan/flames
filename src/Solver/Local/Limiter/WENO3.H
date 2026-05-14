// 3rd-order WENO (Jiang-Shu) reconstruction.
// 3-cell stencil; outer cells of the 5-cell window are ignored.
//
// For face at right edge of cell i (i+1/2), using cells {i-1, i, i+1}:
//   Substencil 0 (cells i-1, i):  u^0 = -1/2 Q_{i-1} + 3/2 Q_i
//   Substencil 1 (cells i, i+1):  u^1 =  1/2 Q_i     + 1/2 Q_{i+1}
//   Smoothness:  beta_0 = (Q_i - Q_{i-1})^2,   beta_1 = (Q_{i+1} - Q_i)^2
//   Linear weights: d_0 = 1/3, d_1 = 2/3
//   Nonlinear weights: alpha_k = d_k / (eps + beta_k)^2,  omega_k = alpha_k / Sum(alpha_j)
//   Reconstructed:  u_face = omega_0 u^0 + omega_1 u^1
//
// Reference: Shu, "Essentially Non-Oscillatory and Weighted ENO Schemes
// for Hyperbolic Conservation Laws", ICASE 97-65 (1997).

#ifndef SOLVER_LOCAL_LIMITER_WENO3_H
#define SOLVER_LOCAL_LIMITER_WENO3_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class WENO3 : public Limiter
{
public:
    static constexpr const char *name = "weno3";

    WENO3(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    WENO3(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    WENO3() { IO::ParmParse pp; pp_queryclass(*this); }

    Set::Scalar eps = 1.0e-6;  // Jiang-Shu smoothness regularization

    static void Parse(WENO3 &value, IO::ParmParse &pp)
    {
        pp.query_default("eps", value.eps, 1.0e-6);
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
        r.p_mix       = scalar(Qm1.p_mix,       Q0.p_mix,       Qp1.p_mix      );
        r.gamma0 = Q0.gamma0;  r.pi0 = Q0.pi0;
        r.gamma1 = Q0.gamma1;  r.pi1 = Q0.pi1;
        return r;
    }

private:
    Set::Scalar scalar(Set::Scalar qm1, Set::Scalar q0, Set::Scalar qp1) const
    {
        const Set::Scalar u0 = -0.5 * qm1 + 1.5 * q0;
        const Set::Scalar u1 =  0.5 * q0  + 0.5 * qp1;
        const Set::Scalar b0 = (q0  - qm1) * (q0  - qm1);
        const Set::Scalar b1 = (qp1 - q0)  * (qp1 - q0);
        const Set::Scalar a0 = (1.0 / 3.0) / ((eps + b0) * (eps + b0));
        const Set::Scalar a1 = (2.0 / 3.0) / ((eps + b1) * (eps + b1));
        const Set::Scalar w0 = a0 / (a0 + a1);
        const Set::Scalar w1 = a1 / (a0 + a1);
        return w0 * u0 + w1 * u1;
    }
};

}}}

#endif
