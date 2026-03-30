// This implements the Riemann HLLE solver.
//      HLLE only uses the fastest left and right wave speeds, ignoring contact waves
//      As a result it very robust for near-vacuum (low density, pressure) problems
//      More diffusive which caues a decrease in acuracy when compared to Roe and HLLC, especially around boundries, but it is also the fastest solver.
//
// Source: Einfeldt, Bernd. (1988). On Godunov-Type Methods for Gas Dynamics. Siam Journal on Numerical Analysis - SIAM J NUMER ANAL. 25. 294-318. 10.1137/0725021.
//          URL: https://www.researchgate.net/publication/243786215_On_Godunov-Type_Methods_for_Gas_Dynamics
//
// General Notes:
//  RL Subscript denotes average for the roe averaged method HLLE uses.
//

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
//#include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLE Riemann Solver (Harten-Lax-van Leer-Einfeldt)
class HLLE : public FluidRiemann
{
public:
    static constexpr const char *name = "hlle";

    HLLE(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLE(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLE()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLE & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    virtual Flux Solve(State lo, State hi,
                   Set::Scalar p_ref,
                   Set::Scalar small,
                   int Spec_Vol = 1) override
    {
        //----------------------------------------------------------------------
        // Extract states
        //----------------------------------------------------------------------
        Set::Scalar rhoL = std::max(lo.rho, small);
        Set::Scalar rhoR = std::max(hi.rho, small);

        Set::Scalar uL = lo.M_normal  / rhoL;
        Set::Scalar vL = lo.M_tangent / rhoL;
        Set::Scalar uR = hi.M_normal  / rhoR;
        Set::Scalar vR = hi.M_tangent / rhoR;

        Set::Scalar EL = lo.E;
        Set::Scalar ER = hi.E;

        Set::Scalar gamL = lo.gam;
        Set::Scalar gamR = hi.gam;

        Set::Scalar piL = lo.pi;
        Set::Scalar piR = hi.pi;

        //----------------------------------------------------------------------
        // Internal energy and stiffened‑gas pressure
        //----------------------------------------------------------------------
        Set::Scalar KEL = 0.5*rhoL*(uL*uL + vL*vL);
        Set::Scalar UEL = EL - KEL;

        Set::Scalar KER = 0.5*rhoR*(uR*uR + vR*vR);
        Set::Scalar UER = ER - KER;

        Set::Scalar pL = (gamL-1.0)*UEL - gamL*piL + p_ref;
        Set::Scalar pR = (gamR-1.0)*UER - gamR*piR + p_ref;

        pL = std::max(pL, small);
        pR = std::max(pR, small);

        //----------------------------------------------------------------------
        // Correct stiffened-gas sound speeds:  a² = γ (p + γ pi) / ρ
        //----------------------------------------------------------------------
        Set::Scalar aL = std::sqrt(gamL * (pL + gamL*piL) / rhoL);
        Set::Scalar aR = std::sqrt(gamR * (pR + gamR*piR) / rhoR);

        //----------------------------------------------------------------------
        // HLLE wave speed estimates
        //----------------------------------------------------------------------
        Set::Scalar SL = std::min(uL - aL, uR - aR);
        Set::Scalar SR = std::max(uL + aL, uR + aR);

        //----------------------------------------------------------------------
        // Compute fluxes (now stiffened-gas corrected)
        //----------------------------------------------------------------------
        Flux FL, FR;

        // left state fluxes
        FL.mass = rhoL*uL;
        FL.momentum_normal  = rhoL*uL*uL + pL;
        FL.momentum_tangent = rhoL*uL*vL;
        FL.energy = uL*(EL + pL + piL);

        // right state fluxes
        FR.mass = rhoR*uR;
        FR.momentum_normal  = rhoR*uR*uR + pR;
        FR.momentum_tangent = rhoR*uR*vR;
        FR.energy = uR*(ER + pR + piR);

        //----------------------------------------------------------------------
        // Select region
        //----------------------------------------------------------------------
        if (SL >= 0.0)
            return FL;

        if (SR <= 0.0)
            return FR;

        //----------------------------------------------------------------------
        // HLLE intermediate (subsonic region)
        //----------------------------------------------------------------------
        Set::Scalar inv = 1.0 / (SR - SL + small);

        Flux F;

        F.mass =
            (SR*FL.mass - SL*FR.mass + SL*SR*(rhoR - rhoL)) * inv;

        F.momentum_normal =
            (SR*FL.momentum_normal - SL*FR.momentum_normal
            + SL*SR*(hi.M_normal - lo.M_normal)) * inv;

        F.momentum_tangent =
            (SR*FL.momentum_tangent - SL*FR.momentum_tangent
            + SL*SR*(hi.M_tangent - lo.M_tangent)) * inv;

        // energy jump uses corrected total E
        F.energy =
            (SR*FL.energy - SL*FR.energy
            + SL*SR*(ER - EL)) * inv;

        return F;
    }

    static int Test()
    {
        HLLE solver;
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
                Util::Exception(INFO, "Uniform flow should result in constant flux.");
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
