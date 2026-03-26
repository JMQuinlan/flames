// This implements a mixed Riemann HLLC/HLLE solver for weakly incompressible flow using the Tammann EOS.
//      HLLC solver uses the fastest left and right wave speeds, along with the middle contact wave.
//          Restores some accuracy when compared to HLLE with a slightly increased computational time, while providing same robustness.
//      HLLE solver uses a less diffusive energy model.
//
// State variables of density and momentum are calculated via 3-wave HLLC, energy is calculated via 2-wave HLLE.
//      This will give a more accurate model than HLLC for energy, and the same accuracy of rho, M, as HLLC. Best of two worlds (hopefully :) )
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-dimensional conservation laws, Journal of Computational Physics, Volume 201, Issue 1, 2004, Pages 238-260, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2004.05.015.
//          URL: https://www.sciencedirect.com/science/article/pii/S0021999104002281
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible and Almost Incompressible Media," SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486�B507, Jan. 2017, doi: 10.1137/16m1067834.
//          URL: https://arxiv.org/abs/1604.08544
//          If you wish to use the regular gas HLLCE sovler and not the stiffened gas modeled by the Tammann EOS, set P0 = 0

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLCE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLCE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLCE Riemann Solver (Hybrid HLLC/HLLE)
class HLLCE : public FluidRiemann
{
public:
    static constexpr const char *name = "hllce";

    HLLCE(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLCE(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLCE()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLCE & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    virtual Flux Solve(State lo, State hi,
                   Set::Scalar p_ref,
                   Set::Scalar small,
                   int Spec_Vol = 1) override
    {
        // First compute HLLC flux (corrected stiffened-gas version)
        HLLC hllc_solver;
        Flux F_HLLC = hllc_solver.Solve(lo, hi, p_ref, small, Spec_Vol);

        // Compute HLLE flux (corrected stiffened-gas version)
        HLLE hlle_solver;
        Flux F_HLLE = hlle_solver.Solve(lo, hi, p_ref, small, Spec_Vol);

        // Now construct hybrid flux:
        // Mass & momentum from HLLC (better accuracy)
        // Energy from HLLE (more robust)
        Flux F;
        F.mass             = F_HLLC.mass;
        F.momentum_normal  = F_HLLC.momentum_normal;
        F.momentum_tangent = F_HLLC.momentum_tangent;

        // Use HLLE energy (2-wave model)
        F.energy = F_HLLE.energy;

        return F;
    }

    static int
    Test()
    {
        HLLCE solver;
        int failed = 0;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 1.0, 1.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 2.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxlo = solver.Solve(center, right, pref, small);
            Flux fluxhi = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:    ", left);
                Util::Warning(INFO, "center:  ", center);
                Util::Warning(INFO, "right:   ", right);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Exception(INFO, "Tangential velocity difference incorrectly affecting normal flux.");
            }
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Tangential velocity should induce no normal flux", 1);
        }

        // Test 2: Pure Transverse Velocity Difference
        try
        {
            State left(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 0.0, 1.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 0.0, 2.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxlo = solver.Solve(left, center, pref, small);
            Flux fluxhi = solver.Solve(center, right, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Pure transverse velocity difference affecting normal flux.");
            }
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Pure transverse velocity difference", 1);
        }

        // Test 3: Symmetry Test (no flux across identical states)
        try
        {
            State left(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 0.0, 0.0, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10                          // no change in mass flux
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 // no change in momentum flux
                || fabs(fluxhi.momentum_tangent) > 1E-10                         // zero tangent flux
                || fabs(fluxlo.momentum_tangent) > 1E-10                         // zero tangent flux
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10                   // no change in energy flux
            )
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Symmetric states should result in zero flux.");
            }
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Constant states induces no flux difference", 1);
        }

        // Test 4: Uniform Flow Test (no flux across uniform flow)
        try
        {
            State left(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State center(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            State right(1.0, 1.0, 0.5, 1.0, 1.4, 0.0, NAN);
            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:   ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right:  ", right);
                Util::Warning(INFO, "Fluxlo: ", fluxlo);
                Util::Warning(INFO, "Fluxhi: ", fluxhi);
                Util::Exception(INFO, "Uniform flow should result in no flux.");
            }
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: Uniform flow should maintain constant flux", 1);
        }

        return failed;
    }
};

}
}
}

#endif
