// This implements the classic two-wave Harten-Lax-van Leer (HLL) Riemann solver
// for weakly incompressible flow using the Tammann EOS.
//      HLL averages the left- and right-going waves only; it does not resolve the
//      middle contact wave (unlike HLLC). This makes it more diffusive on contact
//      discontinuities but inherently free of the carbuncle-type instabilities that
//      contact-resolving solvers (HLLC, Roe) can exhibit on grid-aligned strong shocks.
//      Wave-speed estimates follow Davis (1988): S_L = min(u_L - a_L, u_R - a_R),
//      S_R = max(u_L + a_L, u_R + a_R).
//
// Source: A. Harten, P. D. Lax, and B. van Leer, "On Upstream Differencing and
//         Godunov-Type Schemes for Hyperbolic Conservation Laws,"
//         SIAM Review, vol. 25, no. 1, pp. 35-61, 1983.
//
// Source: S. F. Davis, "Simplified second-order Godunov-type methods,"
//         SIAM J. Sci. Stat. Comput., vol. 9, no. 3, pp. 445-473, 1988.
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling
//         of Compressible and Almost Incompressible Media," SIAM J. Sci. Comput.,
//         vol. 39, no. 3, pp. B486-B507, 2017.
//         If you wish to use the regular gas HLL solver instead of the stiffened gas
//         modeled by the Tammann EOS, set P0 = 0.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLL_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLL_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLL Riemann Solver (Harten-Lax-van Leer, two-wave)
class HLL : public FluidRiemann
{
public:
    static constexpr const char *name = "hll";

    HLL(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLL(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLL()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLL & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters to parse
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar p_ref, Set::Scalar small, int Spec_Vol = 1) override
    {
        Set::Scalar rho_L = lo.rho,         rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal,     Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent,    Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E,             E_R = hi.E;
        Set::Scalar gamma_L = lo.gam,       gamma_R = hi.gam;
        Set::Scalar p0_L = lo.p0,           p0_R = hi.p0;
        Set::Scalar T_L = lo.T,             T_R = hi.T;

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
            ue_L = std::max(E_L - ke_L, small);                         // IE per unit volume
            p_L = Solver::EOS::EOS::TammannPressure(rho_L, ue_L, gamma_L, p0_L, p_ref, small);
            h_TL = (ke_L + ue_L + p_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = std::max(E_R - ke_R, small);
            p_R = Solver::EOS::EOS::TammannPressure(rho_R, ue_R, gamma_R, p0_R, p_ref, small);
            h_TR = (ke_R + ue_R + p_R);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small);
            ue_L = std::max(E_L - ke_L, 0.0);
            p_L = (gamma_L - 1.0) * rho_L * ue_L - gamma_L * p0_L + p_ref;
            h_TL = ke_L + ue_L + (p_L / rho_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * rho_R * ue_R - gamma_R * p0_R + p_ref;
            h_TR = ke_R + ue_R + (p_R / rho_R);
        }

        // Velocities
        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // Limiting
        Set::Scalar u_limit = 1e8;
        u_L = (u_L < 0.0) ? std::max(u_L, -u_limit) : std::min(u_L, u_limit);
        v_L = (v_L < 0.0) ? std::max(v_L, -u_limit) : std::min(v_L, u_limit);
        u_R = (u_R < 0.0) ? std::max(u_R, -u_limit) : std::min(u_R, u_limit);
        v_R = (v_R < 0.0) ? std::max(v_R, -u_limit) : std::min(v_R, u_limit);

        // Sound Speeds
        Set::Scalar a_L = Solver::EOS::EOS::TammannSoundSpeed(rho_L, p_L, gamma_L, p0_L, small);
        Set::Scalar a_R = Solver::EOS::EOS::TammannSoundSpeed(rho_R, p_R, gamma_R, p0_R, small);

        // Davis wave-speed estimates
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

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

        // Compute HLL flux based on wave positions
        //   S_L >= 0           : pure left flux  (supersonic from the left)
        //   S_R <= 0           : pure right flux (supersonic from the right)
        //   S_L < 0 < S_R      : HLL average (subsonic)
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = f_L;
            fl.u_interface = u_L;
        }
        else if (S_R <= 0.0)
        {
            fl = f_R;
            fl.u_interface = u_R;
        }
        else
        {
            // HLL averaging: F = (S_R*F_L - S_L*F_R + S_L*S_R*(U_R - U_L)) / (S_R - S_L)
            Set::Scalar denom = 1.0 / (S_R - S_L + small);
            fl.mass             = (S_R * f_L.mass             - S_L * f_R.mass             + S_L * S_R * (rho_R - rho_L)) * denom;
            fl.momentum_normal  = (S_R * f_L.momentum_normal  - S_L * f_R.momentum_normal  + S_L * S_R * (Mn_R  - Mn_L )) * denom;
            fl.momentum_tangent = (S_R * f_L.momentum_tangent - S_L * f_R.momentum_tangent + S_L * S_R * (Mt_R  - Mt_L )) * denom;
            fl.energy           = (S_R * f_L.energy           - S_L * f_R.energy           + S_L * S_R * (E_R   - E_L  )) * denom;
            fl.u_interface      = (S_R * u_L - S_L * u_R) * denom;
        }

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
                Util::ParallelMessage(INFO, "ERROR IN HLL SOLVER:");
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
                Util::ParallelMessage(INFO, "S_L=  ", S_L);
                Util::ParallelMessage(INFO, "S_R=  ", S_R);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    static int Test()
    {
        HLL solver;
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
