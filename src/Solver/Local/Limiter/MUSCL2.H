// MUSCL2: the second-order MUSCL scheme of Schmidmayer, Bryngelson & Colonius,
// JCP 402 (2020) 109080, Sec. 3.1 -- piecewise-linear reconstruction of the
// primitive variables with the monotonized central (MC) slope limiter.
//
//   slope_i = minmod(2 dQ_-, (dQ_- + dQ_+)/2, 2 dQ_+)
//   Q_face_right = Q_i + 0.5 * slope_i
//
// Sch20 pair this reconstruction with a TWO-STEP (midpoint) time integration,
//   q^{n+1/2} = q^n + dt/2 L(q^n),   q^{n+1} = q^n + dt L(q^{n+1/2}).
// Hydro2 integrates through the AMReX Runge-Kutta integrator, so the paper's
// scheme is obtained from the input file with a user-defined Butcher tableau:
//   integration.rk.type     = 0
//   integration.rk.weights  = 0.0 1.0
//   integration.rk.nodes    = 0.0 0.5
//   integration.rk.tableau  = 0.0 0.5 0.0
// (integration.rk.type = 3, SSPRK3, also works; it is just not the paper's.)
//
// No parameters.  Equivalent to Limiter.type = muscl with Limiter.slope = mc.

#ifndef SOLVER_LOCAL_LIMITER_MUSCL2_H
#define SOLVER_LOCAL_LIMITER_MUSCL2_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/MUSCL.H"

namespace Solver { namespace Local { namespace Limiter {

class MUSCL2 : public Limiter
{
public:
    static constexpr const char *name = "muscl2";

    MUSCL2(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    MUSCL2(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    MUSCL2() { IO::ParmParse pp; pp_queryclass(*this); }

    static void Parse(MUSCL2 & /*value*/, IO::ParmParse & /*pp*/) { /* no params */ }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        return PiecewiseLinear(Slope::MC, stencil);
    }
};

}}}

#endif
