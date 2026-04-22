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
#include "Solver/EOS/EOS.H"


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

        // STEP 1: Compute fluid primitives
        Set::Scalar ke_L, ke_R;
        Set::Scalar ue_L, ue_R;
        Set::Scalar p_L, p_R;
        Set::Scalar h_TL, h_TR;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
            ue_L = std::max(E_L - ke_L, small);                                                // IE per unit volume
            p_L = Solver::EOS::EOS::TammannPressure(rho_L, ue_L, gamma_L, p0_L, p_ref, small); // (gamma_L - 1.0) * ue_L - gamma_L * p0_L + p_ref;
            h_TL = (E_L + p_L) / (rho_L + small);                                              // Specific enthalpy

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * ue_R - gamma_R * p0_R + p_ref;
            h_TR = (E_R + p_R) / (rho_R + small);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small);
            ue_R = std::max(E_R - ke_R, small);
            p_R = Solver::EOS::EOS::TammannPressure(rho_R, ue_R, gamma_R, p0_R, p_ref, small); // (gamma_R - 1.0) * ue_R - gamma_R * p0_R + p_ref;
            h_TL = E_L + p_L / (rho_L + small);                                                // Specific enthalpy

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R * rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * rho_R * ue_R - gamma_R * p0_R + p_ref;
            h_TR = E_R + p_R / (rho_R + small);
        }

        Set::Scalar u_L = Mn_L / (rho_L + small), u_R = Mn_R / (rho_R + small);
        Set::Scalar v_L = Mt_L / (rho_L + small), v_R = Mt_R / (rho_R + small);

        //
        // STEP 2: Compute Roe-averaged quantities
        //
        Set::Scalar sqrt_rho_L = std::sqrt(rho_L);
        Set::Scalar sqrt_rho_R = std::sqrt(rho_R);
        Set::Scalar rho_RL = sqrt_rho_L * sqrt_rho_R;

        Set::Scalar u_RL = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R + small);
        Set::Scalar v_RL = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R + small);
        Set::Scalar h_RL = (sqrt_rho_L * h_TL + sqrt_rho_R * h_TR) / (sqrt_rho_L + sqrt_rho_R + small);

        Set::Scalar p_RL = (sqrt_rho_L * p_L + sqrt_rho_R * p_R) / (sqrt_rho_L + sqrt_rho_R + small);
        Set::Scalar p0_RL = (sqrt_rho_L * p0_L + sqrt_rho_R * p0_R) / (sqrt_rho_L + sqrt_rho_R + small);
        Set::Scalar gamma_RL = (sqrt_rho_L * gamma_L + sqrt_rho_R * gamma_R) / (sqrt_rho_L + sqrt_rho_R + small);

        // Roe-averaged sound speed with Tammann EOS (CORRECTED)
        Set::Scalar a_RL_sq = gamma_RL * (p_RL + p0_RL) / (rho_RL + small);

        Util::AssertException(INFO, TEST(a_RL_sq == a_RL_sq), "a_RL_sq is nan/inf; (a_RL_sq=", a_RL_sq, ")");
        Util::AssertException(INFO, TEST(a_RL_sq >= 0), "a_RL_sq is negative; (a_RL_sq=", a_RL_sq, ")");

        Set::Scalar a_RL = std::sqrt(a_RL_sq);

        //
        // STEP 3: Compute Roe-averaged wave speeds (eigenvalues)
        //
        Set::Scalar lambda1 = u_RL;        // Contact wave (entropy/shear)
        Set::Scalar lambda2 = u_RL + a_RL; // Right-moving acoustic wave
        Set::Scalar lambda3 = u_RL - a_RL; // Left-moving acoustic wave

        //
        // STEP 4: Compute wave strengths (characteristic variables)
        //
        Set::Scalar deltarho = rho_R - rho_L;
        Set::Scalar deltap = p_R - p_L;
        Set::Scalar deltau = u_R - u_L;
        Set::Scalar deltav = v_R - v_L;

        Set::Scalar deltav1 = deltarho - deltap / (a_RL_sq + small);     // Entropy wave strength
        Set::Scalar deltav2 = deltau + deltap / (rho_RL * a_RL + small); // Right acoustic wave strength
        Set::Scalar deltav3 = deltau - deltap / (rho_RL * a_RL + small); // Left acoustic wave strength

        //
        // STEP 5: Compute the right eigenvectors (CORRECTED for Spec_Vol)
        //
        Set::Scalar r11, r12, r13;
        Set::Scalar r21, r22, r23;
        Set::Scalar r31, r32, r33;

        if (Spec_Vol == 1)
        {
            // Volume-specific energy formulation
            r11 = 1.0;
            r12 = u_RL;
            r13 = 0.5 * u_RL * u_RL + 0.5 * v_RL * v_RL;

            r21 = 0.5 * rho_RL / a_RL;
            r22 = 0.5 * rho_RL / a_RL * (u_RL + a_RL);
            r23 = 0.5 * rho_RL / a_RL * (h_RL + a_RL * u_RL);

            r31 = -0.5 * rho_RL / a_RL;
            r32 = -0.5 * rho_RL / a_RL * (u_RL - a_RL);
            r33 = -0.5 * rho_RL / a_RL * (h_RL - a_RL * u_RL); // CORRECTED
        }
        else
        {
            // Mass-specific energy formulation
            r11 = 1.0;
            r12 = u_RL;
            r13 = 0.5 * u_RL * u_RL + 0.5 * v_RL * v_RL;

            r21 = 0.5 * rho_RL / a_RL;
            r22 = 0.5 * rho_RL / a_RL * (u_RL + a_RL);
            r23 = 0.5 * (h_RL + a_RL * u_RL); // No rho_RL factor for mass-specific

            r31 = -0.5 * rho_RL / a_RL;
            r32 = -0.5 * rho_RL / a_RL * (u_RL - a_RL);
            r33 = -0.5 * (h_RL - a_RL * u_RL); // CORRECTED - no rho_RL factor
        }

        //
        // ROE ENTROPY FIX (prevents expansion shocks)
        //
        if (entropy_fix == 1)
        {
            // Option 1: chimeracfd method
            lambda1 = fabs(lambda1);
            lambda2 = fabs(lambda2);
            lambda3 = fabs(lambda3);

            if (lambda1 < deltau)
                lambda1 = 0.5 * (lambda1 * lambda1 + deltau * deltau) / (deltau + small);
            if (lambda2 < deltau)
                lambda2 = 0.5 * (lambda2 * lambda2 + deltau * deltau) / (deltau + small);
            if (lambda3 < deltau)
                lambda3 = 0.5 * (lambda3 * lambda3 + deltau * deltau) / (deltau + small);
        }
        else if (entropy_fix == 2)
        {
            // Option 2: Jayanti method with Tammann EOS (CORRECTED)
            Set::Scalar a_L = std::sqrt(gamma_L * (p_L + p0_L) / (rho_L + small));
            Set::Scalar a_R = std::sqrt(gamma_R * (p_R + p0_R) / (rho_R + small));

            Set::Scalar lambda1_L = u_L;
            Set::Scalar lambda1_R = u_R;
            Set::Scalar lambda2_L = u_L + a_L;
            Set::Scalar lambda2_R = u_R + a_R;
            Set::Scalar lambda3_L = u_L - a_L;
            Set::Scalar lambda3_R = u_R - a_R;

            Set::Scalar fix1 = std::max(0.0, std::max(lambda1 - lambda1_L, lambda1_R - lambda1));
            Set::Scalar fix2 = std::max(0.0, std::max(lambda2 - lambda2_L, lambda2_R - lambda2));
            Set::Scalar fix3 = std::max(0.0, std::max(lambda3 - lambda3_L, lambda3_R - lambda3));

            lambda1 = std::max(fabs(lambda1), fix1);
            lambda2 = std::max(fabs(lambda2), fix2);
            lambda3 = std::max(fabs(lambda3), fix3);
        }
        else
        {
            // No entropy fix - just take absolute values
            lambda1 = fabs(lambda1);
            lambda2 = fabs(lambda2);
            lambda3 = fabs(lambda3);
        }

        //
        // STEP 7: Compute fluxes using Roe averaging
        //
        Flux fl;

        // Mass flux
        fl.mass = 0.5 * (rho_L * u_L + rho_R * u_R) - 0.5 * (r11 * lambda1 * deltav1 + r21 * lambda2 * deltav2 + r31 * lambda3 * deltav3);

        // Normal momentum flux
        fl.momentum_normal = 0.5 * (rho_L * u_L * u_L + p_L + rho_R * u_R * u_R + p_R) - 0.5 * (r12 * lambda1 * deltav1 + r22 * lambda2 * deltav2 + r32 * lambda3 * deltav3);

        // Tangential momentum flux (no wave contribution for transverse velocity)
        fl.momentum_tangent = 0.5 * (rho_L * u_L * v_L + rho_R * u_R * v_R) - 0.5 * rho_RL * lambda1 * deltav * deltav1;

        // Energy flux
        if (Spec_Vol == 1)
        {
            // Volume-specific energy
            fl.energy = 0.5 * (u_L * (E_L + p_L) + u_R * (E_R + p_R)) - 0.5 * (r13 * lambda1 * deltav1 + r23 * lambda2 * deltav2 + r33 * lambda3 * deltav3);
        }
        else
        {
            // Mass-specific energy
            fl.energy = 0.5 * (u_L * (E_L + p_L / (rho_L + small)) + u_R * (E_R + p_R / (rho_R + small))) - 0.5 * (r13 * lambda1 * deltav1 + r23 * lambda2 * deltav2 + r33 * lambda3 * deltav3);
        }

        fl.u_interface = u_L;

        // ERROR CHECKING
        if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN ROE SOLVER:");
                Util::ParallelMessage(INFO, "hi:      ", hi);
                Util::ParallelMessage(INFO, "lo:      ", lo);
                Util::ParallelMessage(INFO, "flux:    ", fl);
                Util::ParallelMessage(INFO, "rho_L=   ", rho_L);
                Util::ParallelMessage(INFO, "rho_R=   ", rho_R);
                Util::ParallelMessage(INFO, "rho_RL=  ", rho_RL);
                Util::ParallelMessage(INFO, "u_L=     ", u_L);
                Util::ParallelMessage(INFO, "u_R=     ", u_R);
                Util::ParallelMessage(INFO, "u_RL=    ", u_RL);
                Util::ParallelMessage(INFO, "p_L=     ", p_L);
                Util::ParallelMessage(INFO, "p_R=     ", p_R);
                Util::ParallelMessage(INFO, "a_RL=    ", a_RL);
                Util::ParallelMessage(INFO, "lambda1= ", lambda1);
                Util::ParallelMessage(INFO, "lambda2= ", lambda2);
                Util::ParallelMessage(INFO, "lambda3= ", lambda3);
                Util::ParallelMessage(INFO, "deltav1= ", deltav1);
                Util::ParallelMessage(INFO, "deltav2= ", deltav2);
                Util::ParallelMessage(INFO, "deltav3= ", deltav3);
            }
            Util::Exception(INFO);
        }

        return fl;
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
            State left(1.0, 1.0, 0.5, 1.0);
            State center(1.0, 1.0, 0.5, 1.0);
            State right(1.0, 1.0, 0.5, 1.0);
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
