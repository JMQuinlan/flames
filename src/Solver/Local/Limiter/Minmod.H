// MUSCL2 with classical minmod (Roe) slope limiter.
// 3-cell stencil; outer cells of the 5-cell window are ignored.
//
//   slope_i = minmod(Q_i - Q_{i-1}, Q_{i+1} - Q_i)
//   Q_face_right = Q_i + 0.5 * slope_i

#ifndef SOLVER_LOCAL_LIMITER_MINMOD_H
#define SOLVER_LOCAL_LIMITER_MINMOD_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class Minmod : public Limiter
{
public:
    static constexpr const char *name = "minmod";

    Minmod(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Minmod(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    Minmod() { IO::ParmParse pp; pp_queryclass(*this); }

    static void Parse(Minmod & /*value*/, IO::ParmParse & /*pp*/) { /* no params */ }

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
    static inline Set::Scalar mm(Set::Scalar a, Set::Scalar b)
    {
        if (a * b <= 0.0) return 0.0;
        return (std::abs(a) < std::abs(b)) ? a : b;
    }
    static Primitive limit(const Primitive &a, const Primitive &b)
    {
        Primitive r;
        r.alpha       = mm(a.alpha,       b.alpha      );
        r.alpha_rho_0 = mm(a.alpha_rho_0, b.alpha_rho_0);
        r.alpha_rho_1 = mm(a.alpha_rho_1, b.alpha_rho_1);
        r.u           = mm(a.u,           b.u          );
        r.v           = mm(a.v,           b.v          );
        r.w           = mm(a.w,           b.w          );
        r.p_mix       = mm(a.p_mix,       b.p_mix      );
        r.gamma0 = a.gamma0;  r.pi0 = a.pi0;
        r.gamma1 = a.gamma1;  r.pi1 = a.pi1;
        return r;
    }
};

}}}

#endif
