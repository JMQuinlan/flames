// All-Mach HLLC (Guillard-Viozat) for the 6-equation diffuse-interface
// model (Saurel-Petitpas-Berry JCP 228, 2009; Schmidmayer-Bryngelson-Colonius
// JCP 402, 2020).
//
// Difference vs. plain HLLC:
//   Guillard & Viozat (Comp. Fluids 28, 1999) show that the upwind
//   pressure flux carries an O(rho c |Delta u|) dissipation term that
//   breaks the p' = O(M^2) pressure scaling in the low-Mach limit
//   (checkerboard modes / excessive diffusion).  The cure scales that
//   DISSIPATION term by a Mach-dependent factor
//
//       theta = min( M_loc / Ma_cutoff , 1 )         floored at alpha_min
//       M_loc = max( |u_L|/c_L , |u_R|/c_R )
//
//   applied to the star PRESSURE (the pressure flux):
//
//       p*    = consistent HLLC star pressure          (contains rho c Du)
//       p*_LM = pbar + theta (p* - pbar),  pbar = (p_L + p_R)/2
//
//   so at M_loc -> 0 the pressure flux tends to the incompressible-limit
//   central discretization while at theta = 1 the scheme reduces EXACTLY
//   to the 6-eq HLLC of Saurel 2009 §3.1.2.
//
//   S_M keeps the FULL (p_R - p_L) term of standard HLLC.  (An earlier
//   version of this file scaled the Delta-p term in S_M instead; that
//   removed the entire physical pressure-difference response of the
//   interface velocity at stagnation -- theta = 0 for quiescent data --
//   so a pressure-driven collapse initialized from rest was never
//   driven.  The Delta-p term in S_M is the acoustic-solver interface
//   velocity, not dissipation; only p* is Mach-scaled.)
//
// Wave speeds, star-state algebra, frozen mixture sound speed, phasic
// Hugoniot star pressures, and Saurel-Abgrall u_interface (= S_M) are
// unchanged from HLLC.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ALL_MACH_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class HLLC_All_Mach : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_all_mach";

    /// Mach number below which the low-Mach pressure scaling activates.
    Set::Scalar Ma_cutoff = 0.3;
    /// Minimum scaling factor (floor on theta).
    Set::Scalar alpha_min = 0.0;

    HLLC_All_Mach(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLC_All_Mach(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_All_Mach()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC_All_Mach &value, IO::ParmParse &pp)
    {
        pp.query_default("ma_cutoff", value.Ma_cutoff, 0.3);
        pp.query_default("alpha_min", value.alpha_min, 0.0);
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        // ---------- Per-side primitive recovery ----------
        auto recover = [&](const State &s,
                           Set::Scalar &alpha,  Set::Scalar &alpha2,
                           Set::Scalar &arh0,   Set::Scalar &arh1,
                           Set::Scalar &rho,    Set::Scalar &u,    Set::Scalar &v,
                           Set::Scalar &p,      Set::Scalar &c,
                           Set::Scalar &rho0,   Set::Scalar &rho1,
                           Set::Scalar &p0,     Set::Scalar &p1)
        {
            alpha  = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            alpha2 = 1.0 - alpha;
            // Clamp phase masses at ZERO only (matches HLLC.H): a positive
            // floor would export fictitious trace-phase mass in the fluxes.
            arh0   = std::max(s.alpha_rho_0, 0.0);
            arh1   = std::max(s.alpha_rho_1, 0.0);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            rho0 = std::max(arh0 / alpha,  small);
            rho1 = std::max(arh1 / alpha2, small);
            // alpha_floor (not small) as the divide floor -- alpha is already
            // clamped at alpha_floor; see the matching comment in HLLC.H.
            p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, alpha_floor);
            p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, alpha_floor);
            const Set::Scalar c0 = Solver::EOS::EOS::PhasicSoundSpeed(rho0, p0, s.gamma0, s.pi0, small);
            const Set::Scalar c1 = Solver::EOS::EOS::PhasicSoundSpeed(rho1, p1, s.gamma1, s.pi1, small);
            const Set::Scalar Y0 = arh0 / rho;
            const Set::Scalar Y1 = arh1 / rho;
            p = alpha * p0 + alpha2 * p1;                                  // Sch20 eq. 8
            c = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1), small); // Sch20 eq. 17
        };

        Set::Scalar aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL;
        Set::Scalar rho0L, rho1L, p0L, p1L;
        recover(lo, aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL, rho0L, rho1L, p0L, p1L);

        Set::Scalar aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR;
        Set::Scalar rho0R, rho1R, p0R, p1R;
        recover(hi, aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR, rho0R, rho1R, p0R, p1R);

        // ---------- All-Mach (Guillard-Viozat) scaling factor ----------
        // Applied to the star PRESSURE below, NOT to S_M.
        const Set::Scalar M_loc = std::max(std::abs(uL) / cL, std::abs(uR) / cR);
        const Set::Scalar Ma_cutoff_safe = std::max(Ma_cutoff, small);
        Set::Scalar theta = std::min(M_loc / Ma_cutoff_safe, 1.0);
        theta = std::max(theta, alpha_min);                                 // floor

        // ---------- Davis wave speeds ----------
        const Set::Scalar S_L = std::min(uL - cL, uR - cR);
        const Set::Scalar S_R = std::max(uL + cL, uR + cR);

        // ---------- Contact speed S_M: standard HLLC (full Delta-p) ----------
        // The (p_R - p_L) term is the PHYSICAL acoustic-solver response of the
        // interface velocity; scaling it (the old form) removed all pressure
        // driving at stagnation (theta = 0 for quiescent data), so pressure-
        // driven flow initialized from rest was never started.
        const Set::Scalar denSM = rhoL * (S_L - uL) - rhoR * (S_R - uR);
        const Set::Scalar denSM_safe = (std::abs(denSM) < small)
                                       ? (denSM >= 0.0 ? small : -small)
                                       : denSM;
        const Set::Scalar S_M = ( (pR - pL)
                                + rhoL * uL * (S_L - uL)
                                - rhoR * uR * (S_R - uR) )
                                / denSM_safe;

        // ---------- Supersonic fast paths ----------
        Flux fl;
        if (S_L >= 0.0)
        {
            fl.mass0            = arh0L * uL;
            fl.mass1            = arh1L * uL;
            fl.momentum_normal  = rhoL * uL * uL + pL;
            fl.momentum_tangent = rhoL * uL * vL;
            fl.energy_total     = uL * (lo.E_total + pL);
            fl.energy0          = lo.E0 * uL;
            fl.energy1          = lo.E1 * uL;
            fl.u_interface      = uL;
            fl.momentum_tangent2 = (fl.mass0 + fl.mass1) * (lo.M_tangent2 / rhoL); // 3D 2nd tangent; 0 in 2D
            return fl;
        }
        if (S_R <= 0.0)
        {
            fl.mass0            = arh0R * uR;
            fl.mass1            = arh1R * uR;
            fl.momentum_normal  = rhoR * uR * uR + pR;
            fl.momentum_tangent = rhoR * uR * vR;
            fl.energy_total     = uR * (hi.E_total + pR);
            fl.energy0          = hi.E0 * uR;
            fl.energy1          = hi.E1 * uR;
            fl.u_interface      = uR;
            fl.momentum_tangent2 = (fl.mass0 + fl.mass1) * (hi.M_tangent2 / rhoR); // 3D 2nd tangent; 0 in 2D
            return fl;
        }

        // ---------- All-Mach (Guillard-Viozat) pressure-flux scaling ----------
        // Consistent HLLC star pressure (identical from both sides by
        // construction of S_M; averaged for symmetry):
        const Set::Scalar p_star_c = 0.5 * ( (pL + rhoL * (uL - S_L) * (uL - S_M))
                                           + (pR + rhoR * (uR - S_R) * (uR - S_M)) );
        // Damp only the ACOUSTIC deviation of p* from the central average --
        // the O(rho c Delta-u) dissipation that violates the low-Mach p' =
        // O(M^2) scaling.  theta = 1 recovers standard HLLC exactly.
        const Set::Scalar p_bar     = 0.5 * (pL + pR);
        const Set::Scalar p_star_LM = p_bar + theta * (p_star_c - p_bar);

        // ---------- Star-state algebra (standard HLLC, Sau09 §3.1.2) ----------
        const Set::Scalar denL = (std::abs(S_L - S_M) < small)
                                 ? (S_L - S_M >= 0.0 ? small : -small)
                                 : (S_L - S_M);
        const Set::Scalar denR = (std::abs(S_R - S_M) < small)
                                 ? (S_R - S_M >= 0.0 ? small : -small)
                                 : (S_R - S_M);
        const Set::Scalar facL = (S_L - uL) / denL;
        const Set::Scalar facR = (S_R - uR) / denR;

        // Star states from the SAME clamped basis as the fluxes and the
        // add_star U_K deltas (arh*), so F* = F + S(U* - U) is internally
        // consistent (previously star used raw lo./hi. values while the
        // deltas used floored ones).
        const Set::Scalar arh0_starL = arh0L * facL;
        const Set::Scalar arh1_starL = arh1L * facL;
        const Set::Scalar arh0_starR = arh0R * facR;
        const Set::Scalar arh1_starR = arh1R * facR;
        const Set::Scalar rho_starL  = rhoL * facL;
        const Set::Scalar rho_starR  = rhoR * facR;

        const Set::Scalar rho0_starL = std::max(arh0_starL / aL,  small);
        const Set::Scalar rho1_starL = std::max(arh1_starL / a2L, small);
        const Set::Scalar rho0_starR = std::max(arh0_starR / aR,  small);
        const Set::Scalar rho1_starR = std::max(arh1_starR / a2R, small);

        const Set::Scalar p0_starL = Solver::EOS::EOS::PhasicHugoniotPressure(rho0_starL, rho0L, p0L, lo.gamma0, lo.pi0, small);
        const Set::Scalar p1_starL = Solver::EOS::EOS::PhasicHugoniotPressure(rho1_starL, rho1L, p1L, lo.gamma1, lo.pi1, small);
        const Set::Scalar p0_starR = Solver::EOS::EOS::PhasicHugoniotPressure(rho0_starR, rho0R, p0R, hi.gamma0, hi.pi0, small);
        const Set::Scalar p1_starR = Solver::EOS::EOS::PhasicHugoniotPressure(rho1_starR, rho1R, p1R, hi.gamma1, hi.pi1, small);

        const Set::Scalar e0_starL = std::max(p0_starL + lo.gamma0 * lo.pi0, small) / std::max((lo.gamma0 - 1.0) * rho0_starL, small);
        const Set::Scalar e1_starL = std::max(p1_starL + lo.gamma1 * lo.pi1, small) / std::max((lo.gamma1 - 1.0) * rho1_starL, small);
        const Set::Scalar e0_starR = std::max(p0_starR + hi.gamma0 * hi.pi0, small) / std::max((hi.gamma0 - 1.0) * rho0_starR, small);
        const Set::Scalar e1_starR = std::max(p1_starR + hi.gamma1 * hi.pi1, small) / std::max((hi.gamma1 - 1.0) * rho1_starR, small);

        const Set::Scalar E0_starL = aL  * rho0_starL * e0_starL;
        const Set::Scalar E1_starL = a2L * rho1_starL * e1_starL;
        const Set::Scalar E0_starR = aR  * rho0_starR * e0_starR;
        const Set::Scalar E1_starR = a2R * rho1_starR * e1_starR;

        const Set::Scalar Mt_starL = rho_starL * vL;
        const Set::Scalar Mt_starR = rho_starR * vR;
        const Set::Scalar Mn_starL = rho_starL * S_M;
        const Set::Scalar Mn_starR = rho_starR * S_M;

        // Star total energy in the EXPLICIT-p* form (algebraically identical
        // to the Toro form when p* is the consistent value), so the low-Mach
        // p*_LM enters the energy flux:  F*_E = S_M (E* + p*_LM).
        const Set::Scalar Etot_starL = (lo.E_total * (S_L - uL) + p_star_LM * S_M - pL * uL) / denL;
        const Set::Scalar Etot_starR = (hi.E_total * (S_R - uR) + p_star_LM * S_M - pR * uR) / denR;

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

        const Flux F_L = build_flux(uL, vL, pL, arh0L, arh1L, lo.E0, lo.E1, lo.E_total, rhoL);
        const Flux F_R = build_flux(uR, vR, pR, arh0R, arh1R, hi.E0, hi.E1, hi.E_total, rhoR);

        auto add_star = [&](const Flux &F_K,
                            Set::Scalar S,
                            Set::Scalar arh0_star, Set::Scalar arh1_star,
                            Set::Scalar Mn_star, Set::Scalar Mt_star,
                            Set::Scalar E_star, Set::Scalar E0_star, Set::Scalar E1_star,
                            Set::Scalar arh0, Set::Scalar arh1,
                            Set::Scalar Mn_K, Set::Scalar Mt_K,
                            Set::Scalar Etot_K, Set::Scalar E0_K, Set::Scalar E1_K) -> Flux
        {
            Flux f;
            f.mass0            = F_K.mass0            + S * (arh0_star - arh0);
            f.mass1            = F_K.mass1            + S * (arh1_star - arh1);
            f.momentum_normal  = F_K.momentum_normal  + S * (Mn_star   - Mn_K);
            f.momentum_tangent = F_K.momentum_tangent + S * (Mt_star   - Mt_K);
            f.energy_total     = F_K.energy_total     + S * (E_star    - Etot_K);
            f.energy0          = F_K.energy0          + S * (E0_star   - E0_K);
            f.energy1          = F_K.energy1          + S * (E1_star   - E1_K);
            return f;
        };

        if (0.0 <= S_M)
        {
            fl = add_star(F_L, S_L,
                          arh0_starL, arh1_starL, Mn_starL, Mt_starL,
                          Etot_starL, E0_starL,   E1_starL,
                          arh0L, arh1L, lo.M_normal, lo.M_tangent,
                          lo.E_total, lo.E0, lo.E1);
        }
        else
        {
            fl = add_star(F_R, S_R,
                          arh0_starR, arh1_starR, Mn_starR, Mt_starR,
                          Etot_starR, E0_starR,   E1_starR,
                          arh0R, arh1R, hi.M_normal, hi.M_tangent,
                          hi.E_total, hi.E0, hi.E1);
        }
        // Inject the low-Mach star pressure into the momentum flux.  In the
        // star region F*_mom = rho* S_M^2 + p* (exact identity of the
        // F + S(U* - U) construction), so shifting p* -> p*_LM is the pure
        // additive correction below.  No-op at theta = 1.
        fl.momentum_normal += (p_star_LM - p_star_c);

        fl.u_interface = S_M;
        fl.momentum_tangent2 = (fl.mass0 + fl.mass1) * ((S_M >= 0.0) ? lo.M_tangent2 / rhoL : hi.M_tangent2 / rhoR); // 3D 2nd tangent; 0 in 2D
        return fl;
    }
};

}
}
}

#endif
