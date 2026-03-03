// This implements a simple first-order upwind Riemann solver for weakly incompressible flow using the Tammann EOS.
//      Upwind solver uses the sign of the velocity to determine which state to use for flux computation.
//      This is the SIMPLEST possible Riemann solver - useful for debugging and isolating numerical artifacts.
//
// Purpose: Diagnostic tool to test if HLLC solver is causing pressure equilibration at interface
//          If upwind solver eliminates the pressure leak, then HLLC flux computation is the culprit.
//
// Note: This solver is FIRST ORDER and NOT recommended for production simulations.
//       It is intentionally simple to isolate numerical issues.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_UPWIND_H
#define SOLVER_LOCAL_FLUIDRIEMANN_UPWIND_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Simple Upwind Riemann Solver (First Order)
class Upwind : public FluidRiemann
{
public:
    static constexpr const char *name = "upwind";

    Upwind(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    Upwind(IO::ParmParse &pp) { pp_queryclass(*this); }
    Upwind()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Upwind & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small, int Spec_Vol = 1) override
    {
        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;
        Set::Scalar gamma_L = lo.gam, gamma_R = hi.gam;
        Set::Scalar p0_L = lo.p0, p0_R = hi.p0;
        Set::Scalar T_L = lo.T, T_R = hi.T;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // Compute velocities
        Set::Scalar u_L = Mn_L / (rho_L + small);
        Set::Scalar u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small);
        Set::Scalar v_R = Mt_R / (rho_R + small);

        // Compute interface velocity (simple average)
        Set::Scalar u_interface = 0.5 * (u_L + u_R);

        // Select upwind state based on interface velocity
        State upwind_state;
        Set::Scalar rho, u, v, E, gamma, p0;

        if (u_interface >= 0.0)
        {
            // Flow is left-to-right, use left state
            upwind_state = lo;
            rho = rho_L;
            u = u_L;
            v = v_L;
            E = E_L;
            gamma = gamma_L;
            p0 = p0_L;
        }
        else
        {
            // Flow is right-to-left, use right state
            upwind_state = hi;
            rho = rho_R;
            u = u_R;
            v = v_R;
            E = E_R;
            gamma = gamma_R;
            p0 = p0_R;
        }

        // Compute pressure from upwind state
        Set::Scalar ke, ue, p;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke = 0.5 * (u * u + v * v) * rho;
            ue = E - ke;
            p = (gamma - 1.0) * ue - gamma * p0 + p_ref;
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke = 0.5 * (u * u + v * v);
            ue = E - ke;
            p = (gamma - 1.0) * rho * ue - gamma * p0 + p_ref;
        }

        // Ensure positive pressure
        p = std::max(small, p);

        // Compute flux using upwind state
        Flux fl;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m^3]
            fl.mass = rho * u;
            fl.momentum_normal = rho * u * u + p;
            fl.momentum_tangent = rho * u * v;
            fl.energy = u * (E + p);
        }
        else
        {
            // Energies per unit mass [J/kg]
            fl.mass = rho * u;
            fl.momentum_normal = rho * u * u + p;
            fl.momentum_tangent = rho * u * v;
            fl.energy = u * (E + p / (rho + small));
        }

        // ERROR CHECKING
        if ((fl.mass != fl.mass)
            or (fl.momentum_normal != fl.momentum_normal)
            or (fl.momentum_tangent != fl.momentum_tangent)
            or (fl.energy != fl.energy))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN UPWIND SOLVER:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "flux: ", fl);
                Util::ParallelMessage(INFO, "rho=  ", rho);
                Util::ParallelMessage(INFO, "u=    ", u);
                Util::ParallelMessage(INFO, "v=    ", v);
                Util::ParallelMessage(INFO, "ke=   ", ke);
                Util::ParallelMessage(INFO, "ue=   ", ue);
                Util::ParallelMessage(INFO, "p=    ", p);
                Util::ParallelMessage(INFO, "u_interface= ", u_interface);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    static int Test()
    {
        Upwind solver;
        int failed = 0;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0);
            State center(1.0, 1.0, 1.0, 1.0);
            State right(1.0, 1.0, 2.0, 1.0);

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
            State left(1.0, 0.0, 0.0, 1.0);
            State center(1.0, 0.0, 1.0, 1.0);
            State right(1.0, 0.0, 2.0, 1.0);

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
            State left(1.0, 0.0, 0.0, 1.0);
            State center(1.0, 0.0, 0.0, 1.0);
            State right(1.0, 0.0, 0.0, 1.0);

            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.momentum_tangent) > 1E-10
                || fabs(fluxlo.momentum_tangent) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
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
            State left(1.0, 1.0, 0.5, 1.0);
            State center(1.0, 1.0, 0.5, 1.0);
            State right(1.0, 1.0, 0.5, 1.0);

            Flux fluxhi = solver.Solve(center, right, pref, small);
            Flux fluxlo = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10
                || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10
                || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
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

