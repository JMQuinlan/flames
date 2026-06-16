// MUSCL2 with van Leer (harmonic average) slope limiter.
// 3-cell stencil; outer cells of the 5-cell window are ignored.
//
//   slope_i = vanleer(Q_i - Q_{i-1}, Q_{i+1} - Q_i)
//           = 2ab/(a+b)   if a*b > 0,  0 otherwise
//   Q_face_right = Q_i + 0.5 * slope_i
//
// Less dissipative than minmod; slightly more compressive.

#ifndef SOLVER_LOCAL_LIMITER_VANLEER_H
#define SOLVER_LOCAL_LIMITER_VANLEER_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class VanLeer : public Limiter
{
public:
    static constexpr const char *name = "vanleer";

    VanLeer(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    VanLeer(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    VanLeer() { IO::ParmParse pp; pp_queryclass(*this); }

    static void Parse(VanLeer & /*value*/, IO::ParmParse & /*pp*/) { /* no params */ }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        const Primitive &Lm = stencil[RADIUS - 1];
        const Primitive &C  = stencil[RADIUS    ];
        const Primitive &Rm = stencil[RADIUS + 1];
        Primitive bwd   = C  - Lm;
        Primitive fwd   = Rm - C;
        Primitive slope = limit(bwd, fwd);
        return C + 0.5 * slope;
    }

private:
    static inline Set::Scalar vl(Set::Scalar a, Set::Scalar b)
    {
        constexpr Set::Scalar small = 1.0e-30;
        if (a * b <= 0.0) return 0.0;
        return 2.0 * a * b / (a + b + small);
    }
    static Primitive limit(const Primitive &a, const Primitive &b)
    {
        Primitive r;
        r.alpha       = vl(a.alpha,       b.alpha      );
        r.alpha_rho_0 = vl(a.alpha_rho_0, b.alpha_rho_0);
        r.alpha_rho_1 = vl(a.alpha_rho_1, b.alpha_rho_1);
        r.u           = vl(a.u,           b.u          );
        r.v           = vl(a.v,           b.v          );
        r.p_mix       = vl(a.p_mix,       b.p_mix      );
        r.gamma0 = a.gamma0;  r.pi0 = a.pi0;
        r.gamma1 = a.gamma1;  r.pi1 = a.pi1;
        return r;
    }
};

}}}

#endif
