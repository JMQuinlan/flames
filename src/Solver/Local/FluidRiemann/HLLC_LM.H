// HLLC-LM Riemann solver: shock-stable, low-dissipation modification of HLLC for
// weakly incompressible flow using the Tammann EOS.
//
//      The grid-aligned shock instability (carbuncle) in HLLC is traced to an
//      inappropriate scaling of the acoustic dissipation relative to the
//      advection dissipation when the local transverse Mach number is small.
//      Fleischmann, Adami, Adams (2020) recast the HLLC subsonic flux in a
//      "centralized" form and reduce the acoustic wave speeds S_L, S_R by an
//      activation function phi(Ma_local) only in the final flux assembly. The
//      contact wave speed S_star, the star-state densities/energies, and the
//      supersonic branches are untouched, so the classical HLLC is recovered
//      exactly when Ma_local >= Ma_limit (default 0.1).
//
//      Subsonic flux (S_L < 0 < S_R):
//          F* = 1/2 (F_L + F_R)
//               - 1/2 [ phi*|S_L|*(U*_L - U_L)
//                     +     |S*| *(U*_R - U*_L)
//                     + phi*|S_R|*(U_R   - U*_R) ]
//
//      Activation:
//          Ma_local = max(|u_L|/c_L, |u_R|/c_R)
//          phi      = sin( min(1, Ma_local / Ma_limit) * pi/2 )
//
// Source: N. Fleischmann, S. Adami, N. A. Adams, "A shock-stable modification of
//         the HLLC Riemann solver with reduced numerical dissipation,"
//         Journal of Computational Physics, vol. 423, 109762, 2020.
//         doi: 10.1016/j.jcp.2020.109762
//
// Source: V.A. Titarev, E.F. Toro, "Finite-volume WENO schemes for three-dimensional
//         conservation laws," J. Comput. Phys., vol. 201, no. 1, pp. 238-260, 2004.
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface Coupling
//         of Compressible and Almost Incompressible Media," SIAM J. Sci. Comput.,
//         vol. 39, no. 3, pp. B486-B507, 2017.
//         If you wish to use the regular gas HLLC solver instead of the stiffened gas
//         modeled by the Tammann EOS, set P0 = 0.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_LM_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_LM_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLC-LM Riemann Solver (Fleischmann, Adami, Adams 2020)
class HLLC_LM : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_lm";

    HLLC_LM(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_LM(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_LM()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    // Reference Mach number for the low-Mach activation. Standard HLLC is recovered
    // when Ma_local >= Ma_limit. Paper default: 0.1.
    Set::Scalar Ma_limit = 0.1;

    static void Parse(HLLC_LM &value, IO::ParmParse &pp)
    {
        pp.query_default("ma_limit", value.Ma_limit, 0.1);
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
            ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
            ue_L = std::max(E_L - ke_L, small);
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

        // Wave speed estimates (Davis) - UNCHANGED from standard HLLC
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // Middle wave speed (contact discontinuity) - UNCHANGED
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                           / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

        // Star-state conservative components (UNCHANGED - computed with original S_L, S_R)
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);
        Set::Scalar E_star_L = rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))));
        Set::Scalar E_star_R = rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))));

        // Physical fluxes for left and right states
        Flux f_L, f_R;

        if (Spec_Vol == 1)
        {
            // Energies per unit volume [J/m^3]
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L);

            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R);
        }
        else
        {
            // Energies per unit mass [J/kg]
            f_L.mass = rho_L * u_L;
            f_L.momentum_normal = rho_L * u_L * u_L + p_L;
            f_L.momentum_tangent = rho_L * u_L * v_L;
            f_L.energy = u_L * (E_L + p_L / (rho_L + small));

            f_R.mass = rho_R * u_R;
            f_R.momentum_normal = rho_R * u_R * u_R + p_R;
            f_R.momentum_tangent = rho_R * u_R * v_R;
            f_R.energy = u_R * (E_R + p_R / (rho_R + small));
        }

        // ========================================
        // HLLC-LM LOW-MACH MODIFICATION
        // ========================================
        // Activation function phi(Ma_local) reduces the *acoustic* wave speeds
        // S_L, S_R only in the final centralized flux assembly. S_star and the
        // star states above are computed with the original (un-reduced) wave
        // speeds, so the contact wave is undisturbed.
        const Set::Scalar half_pi = 1.5707963267948966; // pi/2
        Set::Scalar Ma_local = std::max(std::abs(u_L) / (a_L + small),
                                        std::abs(u_R) / (a_R + small));
        Set::Scalar phi = std::sin(std::min(1.0, Ma_local / (Ma_limit + small)) * half_pi);

        // Compute HLLC-LM flux based on wave positions
        Flux fl;
        if (0.0 <= S_L)
        {
            // Supersonic from the left: pure left flux (UNCHANGED)
            fl = f_L;
            fl.u_interface = u_L;
        }
        else if (S_R <= 0.0)
        {
            // Supersonic from the right: pure right flux (UNCHANGED)
            fl = f_R;
            fl.u_interface = u_R;
        }
        else
        {
            // Subsonic: centralized HLLC formulation with LM-reduced acoustic
            // dissipation. Recovers classical HLLC when phi = 1.
            Set::Scalar abs_SL_LM = phi * std::abs(S_L);
            Set::Scalar abs_SR_LM = phi * std::abs(S_R);
            Set::Scalar abs_Sstar = std::abs(S_star);

            // Conservative-state differences:
            //   U_L     = (rho_L,      Mn_L,                Mt_L,                E_L     )
            //   U*_L    = (rho_star_L, rho_star_L * S_star, rho_star_L * v_L,    E_star_L)
            //   U*_R    = (rho_star_R, rho_star_R * S_star, rho_star_R * v_R,    E_star_R)
            //   U_R     = (rho_R,      Mn_R,                Mt_R,                E_R     )
            Set::Scalar drho_L = rho_star_L - rho_L;
            Set::Scalar drho_C = rho_star_R - rho_star_L;
            Set::Scalar drho_R = rho_R      - rho_star_R;

            Set::Scalar dMn_L  = rho_star_L * S_star - Mn_L;
            Set::Scalar dMn_C  = (rho_star_R - rho_star_L) * S_star;
            Set::Scalar dMn_R  = Mn_R - rho_star_R * S_star;

            Set::Scalar dMt_L  = (rho_star_L - rho_L) * v_L;
            Set::Scalar dMt_C  = rho_star_R * v_R - rho_star_L * v_L;
            Set::Scalar dMt_R  = Mt_R - rho_star_R * v_R;

            Set::Scalar dE_L   = E_star_L - E_L;
            Set::Scalar dE_C   = E_star_R - E_star_L;
            Set::Scalar dE_R   = E_R      - E_star_R;

            fl.mass             = 0.5 * (f_L.mass             + f_R.mass            )
                                - 0.5 * (abs_SL_LM * drho_L + abs_Sstar * drho_C + abs_SR_LM * drho_R);
            fl.momentum_normal  = 0.5 * (f_L.momentum_normal  + f_R.momentum_normal )
                                - 0.5 * (abs_SL_LM * dMn_L  + abs_Sstar * dMn_C  + abs_SR_LM * dMn_R );
            fl.momentum_tangent = 0.5 * (f_L.momentum_tangent + f_R.momentum_tangent)
                                - 0.5 * (abs_SL_LM * dMt_L  + abs_Sstar * dMt_C  + abs_SR_LM * dMt_R );
            fl.energy           = 0.5 * (f_L.energy           + f_R.energy          )
                                - 0.5 * (abs_SL_LM * dE_L   + abs_Sstar * dE_C   + abs_SR_LM * dE_R  );
            fl.u_interface      = S_star;
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
                Util::ParallelMessage(INFO, "ERROR IN HLLC-LM SOLVER:");
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
                Util::ParallelMessage(INFO, "Ma_local=  ", Ma_local);
                Util::ParallelMessage(INFO, "phi=       ", phi);
                Util::ParallelMessage(INFO, "S_L=       ", S_L);
                Util::ParallelMessage(INFO, "S_R=       ", S_R);
                Util::ParallelMessage(INFO, "S_star=    ", S_star);
            }
            Util::Exception(INFO);
        }

        return fl;
    }

    static int Test()
    {
        HLLC_LM solver;
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

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif
