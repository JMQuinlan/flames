// This implements a Lax-Friedrichs (Rusanov) Riemann solver for weakly incompressible flow using the Tammann EOS.
//      Lax-Friedrichs is a simple, robust, and diffusive solver that uses the maximum wave speed.
//      It's less accurate than HLLC but extremely stable and computationally efficient.
//
// Source: E.F. Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction,"
//         3rd Edition, Springer, 2009, Chapter 10.
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible and Almost Incompressible Media,"
//         SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486�B507, Jan. 2017, doi: 10.1137/16m1067834.
//         URL: https://arxiv.org/abs/1604.08544
//         If you wish to use the regular gas solver and not the stiffened gas modeled by the Tammann EOS, set pi = 0

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H
#define SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Lax-Friedrichs (Rusanov) Riemann Solver
class Lax_Friedrich : public FluidRiemann
{
public:
    static constexpr const char *name = "lax_friedrich";

    Lax_Friedrich(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    Lax_Friedrich(IO::ParmParse &pp) { pp_queryclass(*this); }
    Lax_Friedrich()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Lax_Friedrich & /*value*/, IO::ParmParse & /*pp*/)
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
        Set::Scalar pi_L = lo.pi, pi_R = hi.pi;
        Set::Scalar T_L = lo.T, T_R = hi.T;

        // Ensure no negative densities
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // Fluid primitives
        Set::Scalar ke_L, ke_R;
        Set::Scalar ue_L, ue_R;
        Set::Scalar p_L, p_R;
        Set::Scalar h_TL, h_TR;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small); // KE per unit volume
            ue_L = E_L - ke_L;                                          // IE per unit volume
            p_L = (gamma_L - 1.0) * ue_L - gamma_L * pi_L + p_ref;
            h_TL = (ke_L + ue_L + p_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * ue_R - gamma_R * pi_R + p_ref;
            h_TR = (ke_R + ue_R + p_R);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small); // KE per unit volume
            ue_L = E_L - ke_L;                                                  // IE per unit volume
            p_L = (gamma_L - 1.0) * rho_L * ue_L - gamma_L * pi_L + p_ref;
            h_TL = ke_L + ue_L + (p_L / rho_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * rho_R * ue_R - gamma_R * pi_R + p_ref;
            h_TR = ke_R + ue_R + (p_R / rho_R);
        }

        // Acoustic Speeds
        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // Sound Speeds
        Set::Scalar a_L = std::sqrt(gamma_L * (p_L + pi_L) / (rho_L));
        Set::Scalar a_R = std::sqrt(gamma_R * (p_R + pi_R) / (rho_R));

        // Maximum wave speed (Lax-Friedrichs characteristic)
        Set::Scalar S_max = std::max(std::abs(u_L) + a_L, std::abs(u_R) + a_R);

        // Conservative variables
        State U_L, U_R;
        U_L.rho = rho_L;
        U_L.M_normal = Mn_L;
        U_L.M_tangent = Mt_L;
        U_L.E = E_L;

        U_R.rho = rho_R;
        U_R.M_normal = Mn_R;
        U_R.M_tangent = Mt_R;
        U_R.E = E_R;

        // Fluxes for left and right states
        Flux f_L, f_R;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m^3]
            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L); // [W/m^2]

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);
        }
        else
        {
            // Energies per unit mass [J/kg]
            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L / (rho_L + small)); // [W/kg]

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R / (rho_R + small));
        }

        // Lax-Friedrichs flux: F = 0.5 * (F_L + F_R) - 0.5 * S_max * (U_R - U_L)
        Flux fl;
        fl.mass = 0.5 * (f_L.mass + f_R.mass) - 0.5 * S_max * (U_R.rho - U_L.rho);
        fl.momentum_normal = 0.5 * (f_L.momentum_normal + f_R.momentum_normal) - 0.5 * S_max * (U_R.M_normal - U_L.M_normal);
        fl.momentum_tangent = 0.5 * (f_L.momentum_tangent + f_R.momentum_tangent) - 0.5 * S_max * (U_R.M_tangent - U_L.M_tangent);
        fl.energy = 0.5 * (f_L.energy + f_R.energy) - 0.5 * S_max * (U_R.E - U_L.E);

        // ERROR CHECKING
        if ((fl.mass != fl.mass)
            or (fl.momentum_normal != fl.momentum_normal)
            or (fl.momentum_tangent != fl.momentum_tangent)
            or (fl.energy != fl.energy)
            or (a_L != a_L)
            or (a_R != a_R))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN LAX-FRIEDRICHS SOLVER:");
                Util::ParallelMessage(INFO, "hi:   ", hi);
                Util::ParallelMessage(INFO, "lo:   ", lo);
                Util::ParallelMessage(INFO, "flux: ", fl);
                Util::ParallelMessage(INFO, "rho_L=", rho_L);
                Util::ParallelMessage(INFO, "rho_R=", rho_R);
                Util::ParallelMessage(INFO, "u_L=  ", u_L);
                Util::ParallelMessage(INFO, "u_R=  ", u_R);
                Util::ParallelMessage(INFO, "ke_R= ", ke_R);
                Util::ParallelMessage(INFO, "ke_L= ", ke_L);
                Util::ParallelMessage(INFO, "ue_L= ", ue_L);
                Util::ParallelMessage(INFO, "ue_R= ", ue_R);
                Util::ParallelMessage(INFO, "p_L=  ", p_L);
                Util::ParallelMessage(INFO, "p_R=  ", p_R);
                Util::ParallelMessage(INFO, "a_L=  ", a_L);
                Util::ParallelMessage(INFO, "a_R=  ", a_R);
                Util::ParallelMessage(INFO, "S_max=", S_max);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    static int Test()
    {
        Lax_Friedrich solver;
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
