// HLLE (Harten-Lax-van Leer-Einfeldt) Riemann solver for the 6-equation
// diffuse-interface model (Saurel-Petitpas-Berry JCP 228, 2009;
// Schmidmayer-Bryngelson-Colonius JCP 402, 2020).
//
// HLLE uses ONLY the two acoustic wave speeds (S_L, S_R) and no explicit
// contact-resolving wave, so it is more dissipative than HLLC but more
// robust near vacuum / cavitation. Star-state flux (Einfeldt 1988):
//     F = ( S_R F_L - S_L F_R + S_L S_R (U_R - U_L) ) / (S_R - S_L)
// applied component-by-component to the 6-eq conservative vector.
//
// Wave speed estimates: Davis (frozen mixture sound speed; Sau09 eq. III.2 /
// Sch20 eq. 17).
//
// For the non-conservative alpha and per-phase internal-energy rows the
// integrator needs a contact-resolving "u_interface" (Saurel-Abgrall;
// Sau09 line 836).  HLLE has no contact wave per se, but the HLL contact
// estimate S_M (same algebra as HLLC) is the natural choice and is what we
// expose via Flux::u_interface.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class HLLE : public FluidRiemann
{
public:
    static constexpr const char *name = "hlle";

    HLLE(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLE(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLE()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLE & /*value*/, IO::ParmParse & /*pp*/) {}

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small, int /*Spec_Vol*/ = 1) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        // ---------- Primitive recovery on each side ----------------------
        auto recover = [&](const State &s,
                           Set::Scalar &alpha,  Set::Scalar &alpha2,
                           Set::Scalar &rho,    Set::Scalar &u,    Set::Scalar &v,
                           Set::Scalar &p,      Set::Scalar &c,
                           Set::Scalar &Y0,     Set::Scalar &Y1)
        {
            alpha  = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            alpha2 = 1.0 - alpha;
            const Set::Scalar arh0 = std::max(s.alpha_rho_0, small);
            const Set::Scalar arh1 = std::max(s.alpha_rho_1, small);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            const Set::Scalar rho0 = std::max(arh0 / alpha,  small);
            const Set::Scalar rho1 = std::max(arh1 / alpha2, small);
            const Set::Scalar p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, small);
            const Set::Scalar p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, small);
            const Set::Scalar c0 = Solver::EOS::EOS::PhasicSoundSpeed(rho0, p0, s.gamma0, s.pi0, small);
            const Set::Scalar c1 = Solver::EOS::EOS::PhasicSoundSpeed(rho1, p1, s.gamma1, s.pi1, small);

            Y0 = arh0 / rho;
            Y1 = arh1 / rho;
            p  = alpha * p0 + alpha2 * p1;                                // Sch20 eq. 8
            c  = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1), small); // Sch20 eq. 17
        };

        Set::Scalar aL, a2L, rhoL, uL, vL, pL, cL, Y0L, Y1L;
        Set::Scalar aR, a2R, rhoR, uR, vR, pR, cR, Y0R, Y1R;
        recover(lo, aL, a2L, rhoL, uL, vL, pL, cL, Y0L, Y1L);
        recover(hi, aR, a2R, rhoR, uR, vR, pR, cR, Y0R, Y1R);

        // ---------- Davis / Einfeldt wave-speed estimates ----------------
        // Einfeldt's signal velocities: bound by both L/R characteristics
        // and by zero so the star state is non-empty (key for vacuum
        // robustness; Einfeldt 1988 §3).
        Set::Scalar S_L = std::min(std::min(uL - cL, uR - cR), 0.0);
        Set::Scalar S_R = std::max(std::max(uL + cL, uR + cR), 0.0);

        // HLL contact estimate (used as u_interface for the non-cons rows;
        // Saurel-Abgrall, Sau09 line 836).
        Set::Scalar denSM = rhoL * uL - rhoR * uR - S_L * rhoL + S_R * rhoR;
        const Set::Scalar S_M = (rhoL * uL * uL + pL - rhoR * uR * uR - pR
                                 - S_L * rhoL * uL + S_R * rhoR * uR)
                                / (denSM + (denSM >= 0.0 ? small : -small));

        // ---------- Build L and R fluxes ---------------------------------
        auto build_flux = [&](Set::Scalar u, Set::Scalar v, Set::Scalar p,
                              Set::Scalar arh0, Set::Scalar arh1,
                              Set::Scalar E0, Set::Scalar E1, Set::Scalar Etot,
                              Set::Scalar rho) -> Flux
        {
            Flux f;
            f.mass0            = arh0 * u;
            f.mass1            = arh1 * u;
            f.momentum_normal  = rho * u * u + p;
            f.momentum_tangent = rho * u * v;
            f.energy_total     = u * (Etot + p);
            f.energy0          = E0 * u;
            f.energy1          = E1 * u;
            return f;
        };

        const Flux F_L = build_flux(uL, vL, pL, lo.alpha_rho_0, lo.alpha_rho_1, lo.E0, lo.E1, lo.E_total, rhoL);
        const Flux F_R = build_flux(uR, vR, pR, hi.alpha_rho_0, hi.alpha_rho_1, hi.E0, hi.E1, hi.E_total, rhoR);

        // ---------- Dispatch ---------------------------------------------
        Flux fl;
        if (S_L >= 0.0)
        {
            fl = F_L;
            fl.u_interface = uL;
        }
        else if (S_R <= 0.0)
        {
            fl = F_R;
            fl.u_interface = uR;
        }
        else
        {
            // Einfeldt central flux:  F = ( S_R F_L - S_L F_R + S_L S_R (U_R - U_L) ) / (S_R - S_L).
            const Set::Scalar inv  = 1.0 / std::max(S_R - S_L, small);
            const Set::Scalar slsr = S_L * S_R;

            // Component-wise; the conserved vector U_K corresponds to each Flux
            // component below in the same order F_K.<x> uses.
            fl.mass0            = inv * (S_R * F_L.mass0            - S_L * F_R.mass0            + slsr * (hi.alpha_rho_0 - lo.alpha_rho_0));
            fl.mass1            = inv * (S_R * F_L.mass1            - S_L * F_R.mass1            + slsr * (hi.alpha_rho_1 - lo.alpha_rho_1));
            fl.momentum_normal  = inv * (S_R * F_L.momentum_normal  - S_L * F_R.momentum_normal  + slsr * (hi.M_normal    - lo.M_normal));
            fl.momentum_tangent = inv * (S_R * F_L.momentum_tangent - S_L * F_R.momentum_tangent + slsr * (hi.M_tangent   - lo.M_tangent));
            fl.energy_total     = inv * (S_R * F_L.energy_total     - S_L * F_R.energy_total     + slsr * (hi.E_total     - lo.E_total));
            fl.energy0          = inv * (S_R * F_L.energy0          - S_L * F_R.energy0          + slsr * (hi.E0          - lo.E0));
            fl.energy1          = inv * (S_R * F_L.energy1          - S_L * F_R.energy1          + slsr * (hi.E1          - lo.E1));
            // HLL contact estimate for the non-cons discretization.
            fl.u_interface      = S_M;
        }
        return fl;
    }
};

}
}
}

#endif
