// HLLC-ADC Riemann solver: HLLC with Anti-Diffusion Control via a pressure-ratio
// shock sensor. Cures the carbuncle / grid-aligned shock instability of HLLC
// while preserving full HLLC accuracy on contact and shear waves away from shocks.
//
//      Simon & Mandal (2018) recast HLLC as F_HLLC = F_HLL + S * (U*_HLLC - U_HLL),
//      identifying the second term as an "antidiffusive correction" that restores
//      contact/shear resolution to the underlying dissipative HLL flux. The HLLC
//      carbuncle instability is traced (via linear perturbation and matrix
//      analysis) to inadvertent activation of this antidiffusive correction in
//      the mass and interface-normal momentum components on transverse faces
//      near a shock. The cure is to multiply that correction by a pressure-ratio
//      switch omega(p_L, p_R) that drops to zero across strong jumps and stays at
//      one in smooth regions, applied only on the two critical components:
//
//          F^ADC_mass     = (1 - omega) F_HLL_mass     + omega F_HLLC_mass
//          F^ADC_momnorm  = (1 - omega) F_HLL_momnorm  + omega F_HLLC_momnorm
//          F^ADC_momtan   = F_HLLC_momtan        (unmodified)
//          F^ADC_energy   = F_HLLC_energy        (unmodified)
//
//      For strong shocks (high Mach, stiffened two-phase) the partial cure can
//      leave enough antidiffusion in the energy and tangential-momentum fluxes
//      to keep feeding a carbuncle even though omega -> 0. Setting
//      adc_components = 4 extends the same omega blend to those two components,
//      so omega -> 0 yields a genuine HLL flux on all four; omega -> 1 still
//      recovers full HLLC away from shocks. adc_components = 2 (default) is the
//      paper-faithful cure.
//
//      Pressure-ratio sensor (Simon & Mandal Eqs. 49-50):
//          omega = ( min_k f_k )^alpha,   f_k = min(p_R^k, p_L^k) / max(p_R^k, p_L^k)
//      with alpha = 3 recommended by the paper for stability over a wide Mach
//      range. omega -> 1 in smooth flow (recovers HLLC); omega -> 0 across a
//      shock (recovers HLLE on mass and normal-momentum, full HLLC elsewhere).
//
//      Stencil: the multidim minimum runs over five face pressure ratios when
//      the perpendicular-neighbor pressures are populated on the L and R
//      states (State::p_cell, p_perp_lo, p_perp_hi):
//          k=1: across the current face       -> ratio(L.p_cell, R.p_cell)
//          k=2: perpendicular face below L    -> ratio(L.p_perp_lo, L.p_cell)
//          k=3: perpendicular face above L    -> ratio(L.p_cell, L.p_perp_hi)
//          k=4: perpendicular face below R    -> ratio(R.p_perp_lo, R.p_cell)
//          k=5: perpendicular face above R    -> ratio(R.p_cell, R.p_perp_hi)
//      This is the multidimensional flavor Simon & Mandal advocate (Sec. 7):
//      ensures omega drops near grid-aligned shocks even on the transverse
//      faces where p_L = p_R locally. If the L/R states do not carry valid
//      perpendicular pressures (p_cell <= 0), the solver silently falls back
//      to the single-face (k=1 only) sensor.
//
// Source: S. Simon and J. C. Mandal, "A cure for numerical shock instability in
//         HLLC Riemann solver using antidiffusion control," Computers & Fluids,
//         vol. 174, pp. 144-166, 2018.  doi: 10.1016/j.compfluid.2018.07.001
//
// Source: M. J. Del Razo and R. J. LeVeque, "Numerical Methods for Interface
//         Coupling of Compressible and Almost Incompressible Media," SIAM J.
//         Sci. Comput., vol. 39, no. 3, pp. B486-B507, 2017.
//         If you wish to use the regular gas HLLC-ADC solver instead of the
//         stiffened gas modeled by the Tammann EOS, set P0 = 0.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ADC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ADC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// HLLC-ADC Riemann Solver (Simon & Mandal 2018)
class HLLC_ADC : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_adc";

    HLLC_ADC(IO::ParmParse &pp, std::string name) { pp_queryclass(name, *this); }
    HLLC_ADC(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_ADC()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    // Sharpness exponent of the pressure-ratio sensor. alpha=0 recovers HLLC
    // (no cure); alpha>0 activates antidiffusion control. Paper's matrix
    // stability analysis shows alpha=3.0 reaches the asymptotic stability
    // limit and works over a wide Mach range; alpha=0.5 is the practical
    // minimum to render the scheme strictly stable.
    Set::Scalar alpha = 3.0;

    // Number of flux components the omega switch is applied to at a shock.
    //   2 = Simon & Mandal's paper-faithful cure: blend only mass and normal
    //       momentum toward HLL; tangential momentum and energy stay full HLLC.
    //   4 = full-HLL cure: also blend tangential momentum and energy, so
    //       omega -> 0 yields a genuine HLL flux on all components. This adds
    //       the extra dissipation needed for strong (high-Mach, stiffened
    //       two-phase) shocks where the partial cure leaves enough
    //       antidiffusion in the energy/tangential fluxes to feed a carbuncle.
    // Away from shocks omega -> 1, so both settings recover full HLLC there.
    int adc_components = 2;

    static void Parse(HLLC_ADC &value, IO::ParmParse &pp)
    {
        pp.query_default("alpha", value.alpha, 3.0);
        pp.query_default("adc_components", value.adc_components, 2);
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

        // Wave speed estimates (Davis)
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // Middle wave speed (contact discontinuity)
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                           / (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + small);

        // Star-state conservative components (classical HLLC, Batten et al.)
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

        Flux fl;
        if (0.0 <= S_L)
        {
            // Supersonic from the left: pure left flux (unchanged)
            fl = f_L;
            fl.u_interface = u_L;
        }
        else if (S_R <= 0.0)
        {
            // Supersonic from the right: pure right flux (unchanged)
            fl = f_R;
            fl.u_interface = u_R;
        }
        else
        {
            // Subsonic: blend HLL (diffusive) and HLLC (contact-resolving) on
            // the two critical components using the pressure-ratio sensor.

            // ============================================================
            // ADC pressure-ratio shock sensor (Simon & Mandal Eqs. 49-50)
            //
            // f_k = min(p_a^k, p_b^k) / max(p_a^k, p_b^k) on face k, then
            // omega = (min_k f_k)^alpha. When perpendicular neighbor pressures
            // are populated on the L/R states, use the multidim 5-face min;
            // otherwise fall back to the single-face (current-face) ratio.
            // omega ~ 1 in smooth flow, omega ~ 0 across strong shocks.
            // ============================================================
            Set::Scalar omega;
            if (lo.omega_ext >= 0.0 || hi.omega_ext >= 0.0)
            {
                // Caller drives the blend from an external shock tag (e.g. a
                // div(u) compression sensor). Both sides carry the same per-face
                // value; take the smaller (more dissipative) of any provided.
                Set::Scalar o_lo = (lo.omega_ext >= 0.0) ? lo.omega_ext : 1.0;
                Set::Scalar o_hi = (hi.omega_ext >= 0.0) ? hi.omega_ext : 1.0;
                omega = std::min(o_lo, o_hi);
            }
            else
            {
                auto face_ratio = [&](Set::Scalar pa, Set::Scalar pb) -> Set::Scalar
                {
                    Set::Scalar pmin = std::min(pa, pb);
                    Set::Scalar pmax = std::max(pa, pb);
                    return pmin / (pmax + small);
                };

                // k=1: ratio across the current face (always available)
                Set::Scalar f_min = face_ratio(p_L, p_R);

                // k=2..5: ratios across the four perpendicular faces of the L and
                // R cells, available only if the caller populated p_cell + perps.
                if (lo.p_cell > 0.0 && lo.p_perp_lo > 0.0 && lo.p_perp_hi > 0.0)
                {
                    f_min = std::min(f_min, face_ratio(lo.p_perp_lo, lo.p_cell));
                    f_min = std::min(f_min, face_ratio(lo.p_cell,    lo.p_perp_hi));
                }
                if (hi.p_cell > 0.0 && hi.p_perp_lo > 0.0 && hi.p_perp_hi > 0.0)
                {
                    f_min = std::min(f_min, face_ratio(hi.p_perp_lo, hi.p_cell));
                    f_min = std::min(f_min, face_ratio(hi.p_cell,    hi.p_perp_hi));
                }

                omega = std::pow(f_min, alpha);
            }

            // Classical HLLC flux (mass and normal momentum components only -
            // the tangent and energy components use full HLLC directly below)
            Set::Scalar f_HLLC_mass, f_HLLC_Mn;
            if (S_star >= 0.0)
            {
                // Wave config: S_L < 0 <= S_star < S_R, use F_L + S_L*(U*_L - U_L)
                f_HLLC_mass = f_L.mass             + S_L * (rho_star_L         - rho_L);
                f_HLLC_Mn   = f_L.momentum_normal  + S_L * (rho_star_L * S_star - Mn_L);
            }
            else
            {
                // Wave config: S_L < S_star < 0 <= S_R, use F_R + S_R*(U*_R - U_R)
                f_HLLC_mass = f_R.mass             + S_R * (rho_star_R         - rho_R);
                f_HLLC_Mn   = f_R.momentum_normal  + S_R * (rho_star_R * S_star - Mn_R);
            }

            //   F_HLL = ( S_R F_L - S_L F_R + S_L S_R (U_R - U_L) ) / (S_R - S_L)
            Set::Scalar inv_dS = 1.0 / (S_R - S_L + small);
            Set::Scalar f_HLL_mass = (S_R * f_L.mass            - S_L * f_R.mass            + S_L * S_R * (rho_R - rho_L)) * inv_dS;
            Set::Scalar f_HLL_Mn   = (S_R * f_L.momentum_normal - S_L * f_R.momentum_normal + S_L * S_R * (Mn_R  - Mn_L )) * inv_dS;

            // ADC blend on mass and normal momentum (always cured).
            fl.mass            = (1.0 - omega) * f_HLL_mass + omega * f_HLLC_mass;
            fl.momentum_normal = (1.0 - omega) * f_HLL_Mn   + omega * f_HLLC_Mn;

            // Classical HLLC tangential momentum and energy.
            Set::Scalar f_HLLC_Mt, f_HLLC_E;
            if (S_star >= 0.0)
            {
                f_HLLC_Mt = f_L.momentum_tangent + S_L * (rho_star_L * v_L - Mt_L);
                f_HLLC_E  = f_L.energy           + S_L * (E_star_L         - E_L );
            }
            else
            {
                f_HLLC_Mt = f_R.momentum_tangent + S_R * (rho_star_R * v_R - Mt_R);
                f_HLLC_E  = f_R.energy           + S_R * (E_star_R         - E_R );
            }

            if (adc_components >= 4)
            {
                // Full-HLL cure: also blend tangential momentum and energy, so
                // omega -> 0 gives a genuine HLL flux on all components.
                Set::Scalar f_HLL_Mt = (S_R * f_L.momentum_tangent - S_L * f_R.momentum_tangent + S_L * S_R * (Mt_R - Mt_L)) * inv_dS;
                Set::Scalar f_HLL_E  = (S_R * f_L.energy           - S_L * f_R.energy           + S_L * S_R * (E_R  - E_L )) * inv_dS;
                fl.momentum_tangent = (1.0 - omega) * f_HLL_Mt + omega * f_HLLC_Mt;
                fl.energy           = (1.0 - omega) * f_HLL_E  + omega * f_HLLC_E;
            }
            else
            {
                // Paper-faithful cure: tangential momentum and energy stay full
                // HLLC (antidiffusion control is not applied to these).
                fl.momentum_tangent = f_HLLC_Mt;
                fl.energy           = f_HLLC_E;
            }
            fl.u_interface = S_star;
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
                Util::ParallelMessage(INFO, "ERROR IN HLLC-ADC SOLVER:");
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
        HLLC_ADC solver;
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
