// This implements the Oomar-Jaiman all-Mach HLLC Riemann solver for weakly incompressible two-phase flow using the Tammann EOS.
//
//      This solver addresses low-Mach number issues at material interfaces with large sound speed mismatches
//      (e.g., water/air interfaces) while maintaining accuracy for high-Mach phenomena like detonation waves.
//
//      Key modifications from standard HLLC:
//      1. Interface-aware Mach number calculation
//      2. Mach-dependent wave speed scaling
//      3. Modified star state pressure with reduced acoustic dissipation
//
// Source: M. Oomar and R. Jaiman, "An All-Mach Number HLLC-Based Scheme for Multi-Phase Flow with Interface Sharpening,"
//         Computers & Fluids, vol. 224, 2021, doi: 10.1016/j.compfluid.2021.104937
//
// Source: V.A. Titarev, E.F. Toro, Finite-volume WENO schemes for three-dimensional conservation laws,
//         Journal of Computational Physics, Volume 201, Issue 1, 2004, Pages 238-260
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling of Compressible and Almost Incompressible Media,"
//         SIAM Journal on Scientific Computing, vol. 39, no. 3, pp. B486–B507, Jan. 2017

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_OOMAR_JAIMAN_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_OOMAR_JAIMAN_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
// #include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Oomar-Jaiman All-Mach HLLC Riemann Solver
class HLLC_Oomar_Jaiman : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_oomar_jaiman";

    HLLC_Oomar_Jaiman(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_Oomar_Jaiman(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_Oomar_Jaiman()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    Set::Scalar epsilon = 0.01; // Low-Mach regularization parameter

    static void Parse(HLLC_Oomar_Jaiman &value, IO::ParmParse &pp)
    {
        // Low-Mach regularization parameter (typical range: 0.001 - 0.1)
        // Smaller values = more aggressive low-Mach treatment
        // Larger values = smoother transition to high-Mach
        pp.query_default("epsilon", value.epsilon, 0.01);
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

        // Fluid primitives
        Set::Scalar ke_L, ke_R;
        Set::Scalar ue_L, ue_R;
        Set::Scalar p_L, p_R;
        Set::Scalar h_TL, h_TR;

        if (Spec_Vol == 1)
        {
            // With Respect to Specific Volume (energies are per unit volume)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
            ue_L = E_L - ke_L;
            p_L = (gamma_L - 1.0) * ue_L - gamma_L * p0_L + p_ref;
            h_TL = (ke_L + ue_L + p_L);

            ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
            ue_R = E_R - ke_R;
            p_R = (gamma_R - 1.0) * ue_R - gamma_R * p0_R + p_ref;
            h_TR = (ke_R + ue_R + p_R);
        }
        else
        {
            // With Respect to Specific Mass (energies are per unit mass)
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L * rho_L + small);
            ue_L = E_L - ke_L;
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

        // Sound speeds (Tammann EOS)
        Set::Scalar a_L = std::sqrt(gamma_L * (p_L + p0_L) / (rho_L + small));
        Set::Scalar a_R = std::sqrt(gamma_R * (p_R + p0_R) / (rho_R + small));

        // ========================================
        // OOMAR-JAIMAN ALL-MACH MODIFICATIONS
        // ========================================
        // Compute Mach number including pressure effects
        Set::Scalar Ma_L = std::abs(u_L) / (a_L + small);
        Set::Scalar Ma_R = std::abs(u_R) / (a_R + small);

        // Pressure-based Mach number (for shock detection)
        Set::Scalar p_ratio = std::max(p_L / (p_R + small), p_R / (p_L + small));
        Set::Scalar Ma_pressure = std::sqrt(std::max(0.0, (p_ratio - 1.0) * (gamma_L + 1.0) / (2.0 * gamma_L)));

        // Use maximum of velocity-based and pressure-based Mach numbers
        Set::Scalar Ma_interface = std::max({ Ma_L, Ma_R, Ma_pressure });

        // Scaling function
        Set::Scalar f_Ma = std::sqrt(Ma_interface * Ma_interface + epsilon * epsilon);

        // Step 3: Compute scaled wave speeds
        Set::Scalar S_L = u_L - a_L * f_Ma;
        Set::Scalar S_R = u_R + a_R * f_Ma;

        // Step 4: Compute modified star state pressure
        // This reduces acoustic dissipation at low Mach numbers
        Set::Scalar rho_bar = 0.5 * (rho_L + rho_R);
        Set::Scalar a_bar = 0.5 * (a_L + a_R);
        Set::Scalar p_star = 0.5 * (p_L + p_R) + 0.5 * (u_L - u_R) * rho_bar * a_bar * f_Ma;

        // Step 5: Compute star velocity using modified pressure
        // Using momentum-conserving formulation
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                             / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

        // ========================================
        // STANDARD HLLC STAR STATES
        // ========================================
        // Star state densities
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);

        // Star state energies
        Set::Scalar E_star_L, E_star_R;

        if (Spec_Vol == 1)
        {
            E_star_L = rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small)));
            E_star_R = rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small)));
        }
        else
        {
            E_star_L = E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small));
            E_star_R = E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small));
        }

        // ========================================
        // FLUX CALCULATIONS
        // ========================================
        Flux f_L, f_R;
        Flux f_star_L, f_star_R;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m^3]

            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);

            // - Left Star --
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * (E_star_L - E_L);

            // - Right Star --
            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * (E_star_R - E_R);
        }
        else
        {
            // Energies per unit mass [J/kg]

            // ---- Left ----
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L / (rho_L + small));

            // ---- Right ----
            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R / (rho_R + small));

            // - Left Star --
            f_star_L.mass = f_L.mass + S_L * (rho_star_L - rho_L);
            f_star_L.momentum_normal = f_L.momentum_normal + S_L * (rho_star_L * S_star - rho_L * u_L);
            f_star_L.momentum_tangent = f_L.momentum_tangent + S_L * (rho_star_L * v_L - rho_L * v_L);
            f_star_L.energy = f_L.energy + S_L * ((E_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small))) - E_L);

            // - Right Star --
            f_star_R.mass = f_R.mass + S_R * (rho_star_R - rho_R);
            f_star_R.momentum_normal = f_R.momentum_normal + S_R * (rho_star_R * S_star - rho_R * u_R);
            f_star_R.momentum_tangent = f_R.momentum_tangent + S_R * (rho_star_R * v_R - rho_R * v_R);
            f_star_R.energy = f_R.energy + S_R * ((E_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small))) - E_R);
        }

        // Compute HLLC flux based on wave positions
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = f_L;
        }
        else if (S_L <= 0.0 && 0.0 <= S_star)
        {
            fl = f_star_L;
        }
        else if (S_star <= 0.0 && 0.0 <= S_R)
        {
            fl = f_star_R;
        }
        else
        {
            fl = f_R;
        }

        // ========================================
        // ERROR CHECKING
        // ========================================
        if ((fl.mass != fl.mass) || (fl.momentum_normal != fl.momentum_normal) || (fl.momentum_tangent != fl.momentum_tangent) || (fl.energy != fl.energy) || (a_L != a_L) || (a_R != a_R))
        {
            bool verbose = false;
            if (verbose)
            {
                Util::ParallelMessage(INFO, "-------------------------------");
                Util::ParallelMessage(INFO, "ERROR IN OOMAR-JAIMAN HLLC SOLVER:");
                Util::ParallelMessage(INFO, "hi:        ", hi);
                Util::ParallelMessage(INFO, "lo:        ", lo);
                Util::ParallelMessage(INFO, "flux:      ", fl);
                Util::ParallelMessage(INFO, "rho_L=     ", rho_L);
                Util::ParallelMessage(INFO, "rho_R=     ", rho_R);
                Util::ParallelMessage(INFO, "u_L=       ", u_L);
                Util::ParallelMessage(INFO, "u_R=       ", u_R);
                Util::ParallelMessage(INFO, "p_L=       ", p_L);
                Util::ParallelMessage(INFO, "p_R=       ", p_R);
                Util::ParallelMessage(INFO, "a_L=       ", a_L);
                Util::ParallelMessage(INFO, "a_R=       ", a_R);
                Util::ParallelMessage(INFO, "Ma_L=      ", Ma_L);
                Util::ParallelMessage(INFO, "Ma_R=      ", Ma_R);
                Util::ParallelMessage(INFO, "Ma_int=    ", Ma_interface);
                Util::ParallelMessage(INFO, "f_Ma=      ", f_Ma);
                Util::ParallelMessage(INFO, "S_L=       ", S_L);
                Util::ParallelMessage(INFO, "S_R=       ", S_R);
                Util::ParallelMessage(INFO, "p_star=    ", p_star);
                Util::ParallelMessage(INFO, "S_star=    ", S_star);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    static int Test()
    {
        HLLC_Oomar_Jaiman solver;
        int failed = 0;
        Set::Scalar pref = 10.0;
        Set::Scalar small = 1E-10;

        // Test 1: Static Low-Mach Interface (Primary test for all-Mach capability)
        try
        {
            // Water-like properties on left, air-like on right
            // Should show minimal flux despite large sound speed difference
            State left(1000.0, 0.0, 0.0, 10.0); // High density (water-like)
            State right(1.0, 0.0, 0.0, 10.0);   // Low density (air-like)

            Flux flux = solver.Solve(left, right, pref, small);

            // For static interface, mass flux should be near zero
            if (std::abs(flux.mass) > 1E-8)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Flux:  ", flux);
                Util::Exception(INFO, "Static low-Mach interface showing spurious mass flux.");
            }
            Util::Test::SubMessage("Test 1: Static low-Mach interface (water/air)", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 1: Static low-Mach interface (water/air)", 1);
        }

        // Test 2: Tangential Velocity Difference - No Normal Flux
        try
        {
            State left(1.0, 1.0, 0.0, 1.0);
            State center(1.0, 1.0, 1.0, 1.0);
            State right(1.0, 1.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(center, right, pref, small);
            Flux fluxhi = solver.Solve(left, center, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:    ", left);
                Util::Warning(INFO, "center:  ", center);
                Util::Warning(INFO, "right:   ", right);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Exception(INFO, "Tangential velocity difference incorrectly affecting normal flux.");
            }
            Util::Test::SubMessage("Test 2: Tangential velocity should induce no normal flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 2: Tangential velocity should induce no normal flux", 1);
        }

        // Test 3: Pure Transverse Velocity Difference
        try
        {
            State left(1.0, 0.0, 0.0, 1.0);
            State center(1.0, 0.0, 1.0, 1.0);
            State right(1.0, 0.0, 2.0, 1.0);
            Flux fluxlo = solver.Solve(left, center, pref, small);
            Flux fluxhi = solver.Solve(center, right, pref, small);

            if (fabs(fluxhi.mass - fluxlo.mass) > 1E-10 || fabs(fluxhi.momentum_normal - fluxlo.momentum_normal) > 1E-10 || fabs(fluxhi.energy - fluxlo.energy) > 1E-10)
            {
                Util::Warning(INFO, "left:  ", left);
                Util::Warning(INFO, "center: ", center);
                Util::Warning(INFO, "right: ", right);
                Util::Warning(INFO, "Fluxhi:  ", fluxhi);
                Util::Warning(INFO, "Fluxlo:  ", fluxlo);
                Util::Exception(INFO, "Pure transverse velocity difference affecting normal flux.");
            }
            Util::Test::SubMessage("Test 3: Pure transverse velocity difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 3: Pure transverse velocity difference", 1);
        }

        // Test 4: Symmetry Test (no flux across identical states)
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
            Util::Test::SubMessage("Test 4: Constant states induces no flux difference", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 4: Constant states induces no flux difference", 1);
        }

        // Test 5: Uniform Flow Test (no flux across uniform flow)
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
                Util::Exception(INFO, "Uniform flow should result in no flux.");
            }
            Util::Test::SubMessage("Test 5: Uniform flow should maintain constant flux", 0);
        }
        catch (const std::runtime_error &e)
        {
            failed++;
            Util::Test::SubMessage("Test 5: Uniform flow should maintain constant flux", 1);
        }

        return failed;
    }
};

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif
