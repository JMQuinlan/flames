// MUSCL: second-order piecewise-linear reconstruction of the primitives with a
// selectable slope limiter.  3-cell stencil; the outer cells of the 5-cell
// window are ignored.
//
//   slope_i = phi(Q_i - Q_{i-1}, Q_{i+1} - Q_i)
//   Q_face_right = Q_i + 0.5 * slope_i
//
// Input parameters (prefix Limiter.):
//   slope (string)  minmod (default) | mc | vanleer | superbee
//
//   minmod   : minmod(a, b)                                   (most dissipative)
//   mc       : minmod(2a, (a+b)/2, 2b)  monotonized central   (Sch20's choice, see MUSCL2)
//   vanleer  : 2ab/(a+b) for ab > 0
//   superbee : max(minmod(2a, b), minmod(a, 2b))              (most compressive)
//
// Reference: van Leer, J. Comput. Phys. 32 (1979) 101; Toro, Riemann Solvers
// and Numerical Methods for Fluid Dynamics (2009), ch. 13-14.

#ifndef SOLVER_LOCAL_LIMITER_MUSCL_H
#define SOLVER_LOCAL_LIMITER_MUSCL_H

#include <algorithm>
#include <cmath>
#include <string>

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

enum class Slope { Minmod, MC, VanLeer, Superbee };

inline Set::Scalar SlopeLimit(Slope s, Set::Scalar a, Set::Scalar b)
{
    auto mm = [](Set::Scalar x, Set::Scalar y) -> Set::Scalar {
        if (x * y <= 0.0) return 0.0;
        return (std::abs(x) < std::abs(y)) ? x : y;
    };
    switch (s)
    {
    case Slope::Minmod:
        return mm(a, b);
    case Slope::MC:
    {
        if (a * b <= 0.0) return 0.0;
        const Set::Scalar c = 0.5 * (a + b);
        const Set::Scalar m = std::min({std::abs(2.0 * a), std::abs(c), std::abs(2.0 * b)});
        return std::copysign(m, c);
    }
    case Slope::VanLeer:
    {
        if (a * b <= 0.0) return 0.0;
        const Set::Scalar den = a + b;
        if (std::abs(den) < 1.0e-300) return 0.0;
        return 2.0 * a * b / den;
    }
    case Slope::Superbee:
    {
        if (a * b <= 0.0) return 0.0;
        const Set::Scalar s1 = mm(2.0 * a, b), s2 = mm(a, 2.0 * b);
        return (std::abs(s1) > std::abs(s2)) ? s1 : s2;
    }
    }
    return 0.0;
}

inline Primitive SlopeLimit(Slope s, const Primitive &a, const Primitive &b)
{
    Primitive r;
    r.alpha       = SlopeLimit(s, a.alpha,       b.alpha      );
    r.alpha_rho_0 = SlopeLimit(s, a.alpha_rho_0, b.alpha_rho_0);
    r.alpha_rho_1 = SlopeLimit(s, a.alpha_rho_1, b.alpha_rho_1);
    r.u           = SlopeLimit(s, a.u,           b.u          );
    r.v           = SlopeLimit(s, a.v,           b.v          );
    r.w           = SlopeLimit(s, a.w,           b.w          );
    r.p_mix       = SlopeLimit(s, a.p_mix,       b.p_mix      );
    r.gamma0 = a.gamma0;  r.pi0 = a.pi0;
    r.gamma1 = a.gamma1;  r.pi1 = a.pi1;
    return r;
}

inline Primitive PiecewiseLinear(Slope s, const Primitive stencil[Limiter::STENCIL_SIZE])
{
    const Primitive &Lm = stencil[Limiter::RADIUS - 1];
    const Primitive &C  = stencil[Limiter::RADIUS    ];
    const Primitive &Rm = stencil[Limiter::RADIUS + 1];
    return C + 0.5 * SlopeLimit(s, C - Lm, Rm - C);
}

class MUSCL : public Limiter
{
public:
    static constexpr const char *name = "muscl";

    MUSCL(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    MUSCL(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    MUSCL() { IO::ParmParse pp; pp_queryclass(*this); }

    Slope slope = Slope::Minmod;

    static void Parse(MUSCL &value, IO::ParmParse &pp)
    {
        std::string s = "minmod";
        pp.query_validate("slope", s, {"minmod", "mc", "vanleer", "superbee"});
        value.slope = (s == "mc") ? Slope::MC : (s == "vanleer") ? Slope::VanLeer
                    : (s == "superbee") ? Slope::Superbee : Slope::Minmod;
    }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        return PiecewiseLinear(slope, stencil);
    }
};

}}}

#endif
