//
// This implements the Riemann Roe solver.
//
// Notation and algorithm follow the presentation in Section 5.3.3
// of *Computational Gasdynamics* by Culbert B. Laney (page 88)
//
// This solver uses an optional entropy fix
//   Option 1: chimeracfd method https://chimeracfd.com/programming/gryphon/fluxroe.html
//   Option 2: Eq. 4.3.67 in *Computational Fluid Dynamics for Engineers and Scientists* by Sreenivas Jayanti
//
// Modified for Tammann (Tait-Murnaghan) EOS for weakly incompressible flows
// including microbubble oscillation and underwater detonation applications.
//

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_ROE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_ROE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
//#include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Roe Riemann Solver based on Gas Dynamics - Culbert B. Laney
class Roe : public FluidRiemann
{
public:
    static constexpr const char *name = "roe";

    Roe(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    Roe(IO::ParmParse &pp) { pp_queryclass(*this); }
    Roe()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    int entropy_fix = 0;

    static void Parse(Roe &value, IO::ParmParse &pp)
    {
        pp.query_default("entropy_fix", value.entropy_fix, 0);

        if (value.entropy_fix == 1)
            Util::Warning(INFO, "Roe entropy fix option 1 (chimeracfd) is experimental");
        else if (value.entropy_fix == 2)
            Util::Warning(INFO, "Roe entropy fix option 2 (Jayanti) is experimental");
    }

    virtual Flux Solve(State lo, State hi,
                   Set::Scalar p_ref,
                   Set::Scalar small,
                   int Spec_Vol=1) override
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
        // Internal energy and pressure
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
        // Roe averages
        //----------------------------------------------------------------------
        Set::Scalar rL = std::sqrt(rhoL);
        Set::Scalar rR = std::sqrt(rhoR);
        Set::Scalar denom = rL + rR;
        Set::Scalar rho_RL = rL * rR; // Roe average density for momentum flux

        Set::Scalar uRoe = (rL*uL + rR*uR)/denom;
        Set::Scalar vRoe = (rL*vL + rR*vR)/denom;

        Set::Scalar hL = (EL + pL) / rhoL;
        Set::Scalar hR = (ER + pR) / rhoR;

        Set::Scalar hRoe = (rL*hL + rR*hR)/denom;

        Set::Scalar piRoe = (rL*piL + rR*piR)/denom;
        Set::Scalar gamRoe = (rL*gamL + rR*gamR)/denom;

        //----------------------------------------------------------------------
        // Roe-average sound speed: a² = (γ-1)(H - ½|u|²), H = (E+p)/ρ
        // Works for both ideal and stiffened gas (π absorbed into H via EOS).
        //----------------------------------------------------------------------
        Set::Scalar a2 = (gamRoe-1.0) * ( hRoe - 0.5*(uRoe*uRoe + vRoe*vRoe) );
        a2 = std::max(a2, small);
        Set::Scalar aRoe = std::sqrt(a2);

        //----------------------------------------------------------------------
        // Wave strengths
        //----------------------------------------------------------------------
        Set::Scalar d_rho = rhoR - rhoL;
        Set::Scalar d_u   = uR - uL;
        Set::Scalar d_v   = vR - vL;
        Set::Scalar d_p   = pR - pL;

        // Project deltas into eigenbasis
        Set::Scalar alpha2 = (d_p - rho_RL*aRoe*d_u)/(2*a2);
        Set::Scalar alpha3 = (d_p + rho_RL*aRoe*d_u)/(2*a2);
        Set::Scalar alpha1 = d_rho - alpha2 - alpha3;

        //----------------------------------------------------------------------
        // Wave speeds
        //----------------------------------------------------------------------
        Set::Scalar lambda1 = std::fabs(uRoe);
        Set::Scalar lambda2 = std::fabs(uRoe + aRoe);
        Set::Scalar lambda3 = std::fabs(uRoe - aRoe);

        //----------------------------------------------------------------------
        // Compute flux
        //----------------------------------------------------------------------
        Flux FL, FR;
        FL.mass = rhoL*uL;
        FR.mass = rhoR*uR;

        FL.momentum_normal  = rhoL*uL*uL + pL;
        FR.momentum_normal  = rhoR*uR*uR + pR;

        FL.momentum_tangent = rhoL*uL*vL;
        FR.momentum_tangent = rhoR*uR*vR;

        FL.energy = uL*(EL + pL);
        FR.energy = uR*(ER + pR);

        //----------------------------------------------------------------------
        // Average flux
        //----------------------------------------------------------------------
        Flux F;
        F.mass = 0.5*(FL.mass + FR.mass)
            - 0.5*( lambda1*alpha1
                    + lambda2*alpha2
                    + lambda3*alpha3 );

        // normal momentum
        F.momentum_normal =
            0.5*(FL.momentum_normal + FR.momentum_normal)
            -0.5*( lambda1*alpha1*uRoe
                +lambda2*alpha2*(uRoe + aRoe)
                +lambda3*alpha3*(uRoe - aRoe) );

        // tangential momentum
        F.momentum_tangent =
            0.5*(FL.momentum_tangent + FR.momentum_tangent)
            -0.5*( lambda1*alpha1*vRoe
                +lambda2*alpha2*vRoe
                +lambda3*alpha3*vRoe );

        // energy
        F.energy =
            0.5*(FL.energy + FR.energy)
            -0.5*( lambda1*alpha1*hRoe
                +lambda2*alpha2*(hRoe + aRoe*uRoe)
                +lambda3*alpha3*(hRoe - aRoe*uRoe) );

        return F;
    }

    static int Test()
    {
        Roe solver;
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

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.momentum_tangent) > 1E-10 || fabs(fluxlo.momentum_tangent) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
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

        // Test 4: Uniform Flow Test
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
