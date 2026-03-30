// This implements a modified Riemann HLLC solver for weakly incompressible flow using the Tammann EOS.
//      HLLC solver uses the fastest left and right wave speeds, along with the middle contact wave.
//      Restores some accuracy when compared to HLLE with a slightly increased computational time, while providing same robustness.
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-dimensional conservation laws, Journal of Computational Physics, Volume 201, Issue 1, 2004, Pages 238-260, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2004.05.015.
//          URL: https://www.sciencedirect.com/science/article/pii/S0021999104002281
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible and Almost Incompressible Media," SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486�B507, Jan. 2017, doi: 10.1137/16m1067834.
//          URL: https://arxiv.org/abs/1604.08544
//          If you wish to use the regular gas HLLC sovler and not the stiffened gas modeled by the Tammann EOS, set pi = 0

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLC Riemann Solver
class HLLC : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc";

    HLLC(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref,
                   Set::Scalar small, int Spec_Vol = 1) override
    {
        //---------------------------------------------------------------------
        // Extract left/right states
        //---------------------------------------------------------------------
        Set::Scalar rhoL = std::max(lo.rho, small);
        Set::Scalar rhoR = std::max(hi.rho, small);

        Set::Scalar uL = lo.M_normal / rhoL;
        Set::Scalar uR = hi.M_normal / rhoR;

        Set::Scalar vL = lo.M_tangent / rhoL;
        Set::Scalar vR = hi.M_tangent / rhoR;

        Set::Scalar EL = lo.E;
        Set::Scalar ER = hi.E;

        Set::Scalar gamL = lo.gam;
        Set::Scalar gamR = hi.gam;

        Set::Scalar piL = lo.pi;
        Set::Scalar piR = hi.pi;

        //---------------------------------------------------------------------
        // Internal energy
        //---------------------------------------------------------------------
        Set::Scalar KEL = 0.5 * (uL*uL + vL*vL) * rhoL;
        Set::Scalar UEL = EL - KEL;

        Set::Scalar KER = 0.5 * (uR*uR + vR*vR) * rhoR;
        Set::Scalar UER = ER - KER;

        //---------------------------------------------------------------------
        // Stiffened‑gas physical pressure
        //---------------------------------------------------------------------
        Set::Scalar pL = (gamL - 1.0) * UEL - gamL * piL + p_ref;
        Set::Scalar pR = (gamR - 1.0) * UER - gamR * piR + p_ref;

        pL = std::max(pL, small);
        pR = std::max(pR, small);

        //---------------------------------------------------------------------
        // Correct stiffened‑gas sound speeds
        //---------------------------------------------------------------------
        Set::Scalar aL = std::sqrt(gamL * ((pL + gamL*piL) / rhoL));
        Set::Scalar aR = std::sqrt(gamR * ((pR + gamR*piR) / rhoR));

        //---------------------------------------------------------------------
        // Wave speed estimates (HLLE)
        //---------------------------------------------------------------------
        Set::Scalar SL = std::min(uL - aL, uR - aR);
        Set::Scalar SR = std::max(uL + aL, uR + aR);

        //---------------------------------------------------------------------
        // Contact wave speed
        //---------------------------------------------------------------------
        Set::Scalar S_star =
            (pR - pL + rhoL*uL*(SL - uL) - rhoR*uR*(SR - uR))
            / (rhoL*(SL - uL) - rhoR*(SR - uR) + small);

        //---------------------------------------------------------------------
        // Star region densities
        //---------------------------------------------------------------------
        Set::Scalar rhoL_star = rhoL * (SL - uL) / (SL - S_star + small);
        Set::Scalar rhoR_star = rhoR * (SR - uR) / (SR - S_star + small);

        //---------------------------------------------------------------------
        // Star region energies
        //---------------------------------------------------------------------
        Set::Scalar EL_star =
            (EL/rhoL + (S_star - uL)*(S_star + pL/(rhoL*(SL - uL) + small)))
            * rhoL_star;

        Set::Scalar ER_star =
            (ER/rhoR + (S_star - uR)*(S_star + pR/(rhoR*(SR - uR) + small)))
            * rhoR_star;

        //---------------------------------------------------------------------
        // Compute fluxes fL and fR
        //---------------------------------------------------------------------
        Flux fL, fR;

        fL.mass = rhoL * uL;
        fL.momentum_normal  = rhoL*uL*uL + pL;
        fL.momentum_tangent = rhoL*uL*vL;
        fL.energy = uL * (EL + pL);

        fR.mass = rhoR * uR;
        fR.momentum_normal  = rhoR*uR*uR + pR;
        fR.momentum_tangent = rhoR*uR*vR;
        fR.energy = uR * (ER + pR);

        //---------------------------------------------------------------------
        // Star fluxes
        //---------------------------------------------------------------------
        Flux fL_star, fR_star;

        fL_star.mass = fL.mass + SL*(rhoL_star - rhoL);
        fL_star.momentum_normal =
            fL.momentum_normal + SL*(rhoL_star*S_star - rhoL*uL);
        fL_star.momentum_tangent =
            fL.momentum_tangent + SL*(rhoL_star*vL - rhoL*vL);
        fL_star.energy = fL.energy + SL*(EL_star - EL);

        fR_star.mass = fR.mass + SR*(rhoR_star - rhoR);
        fR_star.momentum_normal =
            fR.momentum_normal + SR*(rhoR_star*S_star - rhoR*uR);
        fR_star.momentum_tangent =
            fR.momentum_tangent + SR*(rhoR_star*vR - rhoR*vR);
        fR_star.energy = fR.energy + SR*(ER_star - ER);

        //---------------------------------------------------------------------
        // Select final flux
        //---------------------------------------------------------------------
        if (0.0 <= SL)
            return fL;
        else if (SL <= 0.0 && 0.0 <= S_star)
            return fL_star;
        else if (S_star <= 0.0 && 0.0 <= SR)
            return fR_star;
        else
            return fR;
    }

    static int
    Test()
    {
        HLLC solver;
        int failed = 0;
        Set::Scalar pref = 0.0;
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
