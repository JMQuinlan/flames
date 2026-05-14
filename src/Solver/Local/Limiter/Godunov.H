// First-order Godunov (no reconstruction) -- returns the center cell.
// Default selection in Hydro2; preserves the original first-order behavior.

#ifndef SOLVER_LOCAL_LIMITER_GODUNOV_H
#define SOLVER_LOCAL_LIMITER_GODUNOV_H

#include "IO/ParmParse.H"
#include "Solver/Local/Limiter/Limiter.H"

namespace Solver { namespace Local { namespace Limiter {

class Godunov : public Limiter
{
public:
    static constexpr const char *name = "godunov";

    Godunov(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Godunov(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    Godunov() { IO::ParmParse pp; pp_queryclass(*this); }

    static void Parse(Godunov & /*value*/, IO::ParmParse & /*pp*/) { /* no params */ }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        return stencil[RADIUS];
    }
};

}}}

#endif
