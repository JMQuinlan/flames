// PORT-PENDING: this Riemann solver has not yet been ported to the 6-equation
// diffuse-interface formulation (Saurel-Petitpas-Berry 2009 / Schmidmayer 2020).
// Use HLLC for now.  The class remains here only so the registration list in
// Hydro2::Parse() still compiles; calling Solve() aborts at runtime.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H
#define SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class Lax_Friedrich : public FluidRiemann
{
public:
    static constexpr const char *name = "lax_friedrich";

    Lax_Friedrich(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Lax_Friedrich(IO::ParmParse &pp) { pp_queryclass(*this); }
    Lax_Friedrich()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Lax_Friedrich & /*value*/, IO::ParmParse & /*pp*/) {}

    virtual Flux Solve(State /*lo*/, State /*hi*/, Set::Scalar /*p_ref*/, Set::Scalar /*small*/, int /*Spec_Vol*/ = 1) override
    {
        Util::Abort(INFO, "Solver " + std::string(name) + " is not yet ported to the 6-equation diffuse-interface model. Use HLLC.");
        return Flux();
    }
};

}
}
}

#endif
