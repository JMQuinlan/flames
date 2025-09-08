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
//


#ifndef SOLVER_LOCAL_RIEMANN_HLLE_H
#define SOLVER_LOCAL_RIEMANN_HLLE_H

#include "IO/ParmParse.H"
#include "Solver/Local/Riemann/Riemann.H"

/// A bunch of solvers
namespace Solver
{
/// Local solvers
namespace Local
{
namespace Riemann
{
/// HLLE Riemann Solver (Harten-Lax-van Leer-Einfeldt)
class HLLE
{
public:
    static constexpr const char *name = "hlle";

    HLLE(IO::ParmParse &pp, std::string name)
    {
        pp_queryclass(name, *this);
    }

    HLLE(IO::ParmParse &pp)
    {
        pp_queryclass(*this);
    }

    HLLE()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    int verbose = 0;

    static void
    Parse(HLLE &value, IO::ParmParse &pp)
    {
        // enable to dump diagnostic data if the hlle solver fails
        pp.query_default("verbose", value.verbose, 1);
    }

    Flux Solve(State lo, State hi, Set::Scalar gamma, Set::Scalar p_ref, Set::Scalar small, Set::Scalar p_0 = 0.0, Set::Scalar gamma_alt = 0.0)
    {
        Set::Scalar rho_L = lo.rho, rho_R = hi.rho;
        Set::Scalar Mn_L = lo.M_normal, Mn_R = hi.M_normal;
        Set::Scalar Mt_L = lo.M_tangent, Mt_R = hi.M_tangent;
        Set::Scalar E_L = lo.E, E_R = hi.E;

        if (gamma_alt == 0.0)
        {
            gamma_alt = gamma;
        }

        // Ensure no negative densities
        rho_L = std::max(0.0+small, rho_L);
        rho_R = std::max(0.0+small, rho_R);

        // STEP 1: Compute fluid primitives
        //Set::Scalar ke_L = 0.5 * (Mn_L * Mn_L /*+ Mt_L * Mt_L*/) / (rho_L + small); // KE per unit volume
        Set::Scalar ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small); // KE per unit volume
        Set::Scalar ue_L = E_L - ke_L;                                          // IE per unit volume
        Set::Scalar p_L = (gamma - 1.0) * ue_L - gamma * (p_0 + p_ref);         // pressure Tammann EOS modification
        //p_L = std::max(0.0+small, p_L);

        //Set::Scalar ke_R = 0.5 * (Mn_R * Mn_R /*+ Mt_R * Mt_R*/) / (rho_R + small);
        Set::Scalar ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
        Set::Scalar ue_R = E_R - ke_R;
        Set::Scalar p_R = (gamma - 1.0) * ue_R - gamma * (p_0 + p_ref);
        //p_R = std::max(0.0+small, p_R);

        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        // STEP 2: Compute sound speeds
        //Set::Scalar a_L = std::sqrt(gamma * (p_L + p_ref) / (rho_L + small)); // MODIFIED
        //Set::Scalar a_R = std::sqrt(gamma * (p_R + p_ref) / (rho_R + small)); // MODIFIED
        Set::Scalar a_L = std::sqrt(gamma_alt * (p_L + p_ref) / (rho_L + small));
        Set::Scalar a_R = std::sqrt(gamma_alt * (p_R + p_ref) / (rho_R + small));

        // STEP 3: Compute Roe-averaged quantities for wave speed estimates
        Set::Scalar rho_RL = std::sqrt(rho_L * rho_R);
        Set::Scalar u_RL = (std::sqrt(rho_L) * u_L + std::sqrt(rho_R) * u_R) / (std::sqrt(rho_L) + std::sqrt(rho_R) + small);
        Set::Scalar v_RL = (std::sqrt(rho_L) * v_L + std::sqrt(rho_R) * v_R) / (std::sqrt(rho_L) + std::sqrt(rho_R) + small);
        Set::Scalar h_TL = (E_L + p_L) / (rho_L + small);
        Set::Scalar h_TR = (E_R + p_R) / (rho_R + small);
        Set::Scalar h_RL = (std::sqrt(rho_L) * h_TL + std::sqrt(rho_R) * h_TR) / (std::sqrt(rho_L) + std::sqrt(rho_R) + small);
        //Set::Scalar a_RL_sq = std::max(0.0, (gamma - 1.0) * (h_RL - 0.5 * (u_RL * u_RL + v_RL * v_RL)) + (gamma * p_ref / (std::sqrt(rho_L) + std::sqrt(rho_R) + small))); // MODIFIED
        Set::Scalar a_RL_sq = std::max(0.0, (gamma_alt - 1.0) * (h_RL - 0.5 * (u_RL * u_RL + v_RL * v_RL)) + (gamma_alt * p_ref / (std::sqrt(rho_L) + std::sqrt(rho_R) + small)));
        Set::Scalar a_RL = std::sqrt(a_RL_sq);

        // STEP 4: Compute wave speed estimates (Einfeldt estimates)
        Set::Scalar S_L = std::min(u_L - a_L, u_RL - a_RL);
        Set::Scalar S_R = std::max(u_R + a_R, u_RL + a_RL);

        // STEP 5: Compute physical fluxes
        Flux f_L, f_R;

        f_L.mass = rho_L * u_L;
        f_L.momentum_normal = rho_L * u_L * u_L + p_L;
        f_L.momentum_tangent = rho_L * u_L * v_L;
        f_L.energy = u_L * (E_L + p_L);

        f_R.mass = rho_R * u_R;
        f_R.momentum_normal = rho_R * u_R * u_R + p_R;
        f_R.momentum_tangent = rho_R * u_R * v_R;
        f_R.energy = u_R * (E_R + p_R);

        // STEP 6: Compute HLLE flux
        Flux fl;

        if (0 <= S_L)
        {
            // Supersonic flow to the right
            fl = f_L;
        }
        else if (S_R <= 0)
        {
            // Supersonic flow to the left
            fl = f_R;
        }
        else
        {
            // Subsonic flow - use HLLE formula
            Set::Scalar inv_SR_minus_SL = 1.0 / (S_R - S_L + small);

            // Compute the HLLE flux: (S_R*F_L - S_L*F_R + S_R*S_L*(U_R - U_L))/(S_R - S_L)
            fl.mass = (S_R * f_L.mass - S_L * f_R.mass + S_R * S_L * (rho_R - rho_L)) * inv_SR_minus_SL;

            fl.momentum_normal = (S_R * f_L.momentum_normal - S_L * f_R.momentum_normal + S_R * S_L * (Mn_R - Mn_L)) * inv_SR_minus_SL;

            fl.momentum_tangent = (S_R * f_L.momentum_tangent - S_L * f_R.momentum_tangent + S_R * S_L * (Mt_R - Mt_L)) * inv_SR_minus_SL;

            fl.energy = (S_R * f_L.energy - S_L * f_R.energy + S_R * S_L * (E_R - E_L)) * inv_SR_minus_SL;
        }

        // ERROR CHECKING
        if (   (fl.mass != fl.mass)
            or (fl.momentum_normal != fl.momentum_normal)
            or (fl.momentum_tangent != fl.momentum_tangent)
            or (fl.energy != fl.energy) )
        {
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN HLLE SOLVER:");
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

    static int
    Test()
    {
        HLLE solver;

        int failed = 0;
        Set::Scalar gamma = 1.4;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;

        // Test 1: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0);
            State center(1.0, 1.0, 1.0, 1.0);
            State right(1.0, 1.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(center, right, gamma, pref, small);
            Flux fluxhi = solver.Solve(left, center, gamma, pref, small);
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
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small);
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small);
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
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small);
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small);
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
            State left(1.0, 1.0, 0.5, 1.0);
            State center(1.0, 1.0, 0.5, 1.0);
            State right(1.0, 1.0, 0.5, 1.0);
            Flux fluxhi = solver.Solve(center, right, gamma, pref, small);
            Flux fluxlo = solver.Solve(left, center, gamma, pref, small);
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
