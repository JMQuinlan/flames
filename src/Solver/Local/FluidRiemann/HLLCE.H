// HLLCE — HLLC with Einfeldt-bounded wave speeds — for the 6-equation
// diffuse-interface model (Saurel-Petitpas-Berry JCP 228, 2009;
// Schmidmayer-Bryngelson-Colonius JCP 402, 2020).
//
// HLLCE differs from plain HLLC in the wave-speed estimates:
//
//   * HLLC (Davis): S_L = min(u_L - c_L, u_R - c_R)
//                   S_R = max(u_L + c_L, u_R + c_R)
//
//   * HLLCE (Einfeldt 1988 §3, Roe-Einfeldt):
//                   S_L = min(u_L - c_L, u_hat - c_hat, 0)
//                   S_R = max(u_R + c_R, u_hat + c_hat, 0)
//
//     u_hat, c_hat are Roe averages (Vinokur-Liu sqrt-rho weighting).  The
//     clamp at zero ensures the HLL star region is non-empty even when both
//     characteristics share a sign, which is the positivity-preserving
//     property required near vacuum / cavitation.
//
// Everything else is identical to HLLC: same S_M contact formula on the
// boundary states (Saurel 2009 line 769), same star algebra (Sau09 §3.1.2
// lines 777-815) with α constant across acoustics (line 798), phasic
// Hugoniot for star E_k (eq. II.8), Toro form for redundant ρE (line 786).
// u_interface = S_M for the Saurel-Abgrall non-conservative discretization
// of α and the per-phase internal-energy rows (Sau09 line 836 / Sch20 eq. 18).

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLCE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLCE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class HLLCE : public FluidRiemann
{
public:
    static constexpr const char *name = "hllce";

    HLLCE(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLCE(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLCE()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLCE & /*value*/, IO::ParmParse & /*pp*/) {}

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        // ---------- Per-side primitive recovery -------------------------
        auto recover = [&](const State &s,
                           Set::Scalar &alpha,  Set::Scalar &alpha2,
                           Set::Scalar &arh0,   Set::Scalar &arh1,
                           Set::Scalar &rho,    Set::Scalar &u,    Set::Scalar &v,
                           Set::Scalar &p,      Set::Scalar &c,
                           Set::Scalar &Y0,     Set::Scalar &Y1,
                           Set::Scalar &rho0,   Set::Scalar &rho1,
                           Set::Scalar &p0,     Set::Scalar &p1,
                           Set::Scalar &c0,     Set::Scalar &c1,
                           Set::Scalar &H)
        {
            alpha  = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            alpha2 = 1.0 - alpha;
            arh0   = std::max(s.alpha_rho_0, small);
            arh1   = std::max(s.alpha_rho_1, small);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            rho0 = std::max(arh0 / alpha,  small);
            rho1 = std::max(arh1 / alpha2, small);
            p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, small);
            p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, small);
            c0 = Solver::EOS::EOS::PhasicSoundSpeed(rho0, p0, s.gamma0, s.pi0, small);
            c1 = Solver::EOS::EOS::PhasicSoundSpeed(rho1, p1, s.gamma1, s.pi1, small);

            Y0 = arh0 / rho;
            Y1 = arh1 / rho;
            p  = alpha * p0 + alpha2 * p1;                                 // Sch20 eq. 8
            c  = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1), small); // Sch20 eq. 17
            H  = (s.E_total + p) / rho;                                    // mixture stagnation enthalpy
        };

        Set::Scalar aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL;
        Set::Scalar rho0L, rho1L, p0L, p1L, c0L, c1L, Y0L, Y1L, HL;
        recover(lo, aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL,
                Y0L, Y1L, rho0L, rho1L, p0L, p1L, c0L, c1L, HL);

        Set::Scalar aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR;
        Set::Scalar rho0R, rho1R, p0R, p1R, c0R, c1R, Y0R, Y1R, HR;
        recover(hi, aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR,
                Y0R, Y1R, rho0R, rho1R, p0R, p1R, c0R, c1R, HR);

        // ---------- Roe averages (Vinokur-Liu sqrt-rho weighting) -------
        const Set::Scalar sqrtL = std::sqrt(std::max(rhoL, small));
        const Set::Scalar sqrtR = std::sqrt(std::max(rhoR, small));
        const Set::Scalar denom = std::max(sqrtL + sqrtR, small);

        const Set::Scalar rho_hat = sqrtL * sqrtR;
        const Set::Scalar u_hat   = (sqrtL * uL + sqrtR * uR) / denom;
        const Set::Scalar v_hat   = (sqrtL * vL + sqrtR * vR) / denom;
        const Set::Scalar H_hat   = (sqrtL * HL + sqrtR * HR) / denom;
        const Set::Scalar alpha_hat  = std::min(std::max((sqrtL * aL + sqrtR * aR) / denom,
                                                          alpha_floor),
                                                1.0 - alpha_floor);
        const Set::Scalar alpha2_hat = 1.0 - alpha_hat;

        // Roe-state pressure from H_hat (mixture-effective stiffened-gas inversion).
        const Set::Scalar Aeff = alpha_hat  / (lo.gamma0 - 1.0)
                                + alpha2_hat / (lo.gamma1 - 1.0);
        const Set::Scalar Beff = alpha_hat  * lo.gamma0 * lo.pi0 / (lo.gamma0 - 1.0)
                                + alpha2_hat * lo.gamma1 * lo.pi1 / (lo.gamma1 - 1.0);
        const Set::Scalar gamma_eff = 1.0 + 1.0 / std::max(Aeff, small);
        const Set::Scalar pi_eff    = (Beff / std::max(Aeff, small)) / gamma_eff;
        const Set::Scalar ke_hat    = 0.5 * (u_hat * u_hat + v_hat * v_hat);
        Set::Scalar p_hat = (gamma_eff - 1.0) / gamma_eff
                          * rho_hat * (H_hat - ke_hat)
                          - pi_eff;
        const Set::Scalar p_floor = -std::min(lo.pi0, lo.pi1) + small;
        p_hat = std::max(p_hat, p_floor);

        // Frozen mixture sound speed at the Roe state (Sch20 eq. 17 algebraic form).
        const Set::Scalar c2_hat = (alpha_hat  * lo.gamma0 * (p_hat + lo.pi0)
                                  + alpha2_hat * lo.gamma1 * (p_hat + lo.pi1))
                                  / std::max(rho_hat, small);
        const Set::Scalar c_hat = std::sqrt(std::max(c2_hat, small * small));

        // ---------- Einfeldt-bounded wave speeds ------------------------
        // Both bounded by L/R signal speeds AND clamped past zero so the
        // star region is non-empty.
        const Set::Scalar S_L = std::min(std::min(uL - cL, u_hat - c_hat), 0.0);
        const Set::Scalar S_R = std::max(std::max(uR + cR, u_hat + c_hat), 0.0);

        // HLLC contact S_M (Saurel 2009 line 769; Toro/Batten form on boundary states).
        const Set::Scalar denSM = rhoL * (S_L - uL) - rhoR * (S_R - uR);
        const Set::Scalar denSM_safe = (std::abs(denSM) < small)
                                       ? (denSM >= 0.0 ? small : -small)
                                       : denSM;
        const Set::Scalar S_M = (pR - pL + rhoL * uL * (S_L - uL)
                                          - rhoR * uR * (S_R - uR))
                                / denSM_safe;

        // ---------- Supersonic fast paths -------------------------------
        Flux fl;
        if (S_L >= 0.0)
        {
            fl.mass0            = lo.alpha_rho_0 * uL;
            fl.mass1            = lo.alpha_rho_1 * uL;
            fl.momentum_normal  = rhoL * uL * uL + pL;
            fl.momentum_tangent = rhoL * uL * vL;
            fl.energy_total     = uL * (lo.E_total + pL);
            fl.energy0          = lo.E0 * uL;
            fl.energy1          = lo.E1 * uL;
            fl.u_interface      = uL;
            return fl;
        }
        if (S_R <= 0.0)
        {
            fl.mass0            = hi.alpha_rho_0 * uR;
            fl.mass1            = hi.alpha_rho_1 * uR;
            fl.momentum_normal  = rhoR * uR * uR + pR;
            fl.momentum_tangent = rhoR * uR * vR;
            fl.energy_total     = uR * (hi.E_total + pR);
            fl.energy0          = hi.E0 * uR;
            fl.energy1          = hi.E1 * uR;
            fl.u_interface      = uR;
            return fl;
        }

        // ---------- Star-state algebra (Saurel 2009 §3.1.2) -------------
        const Set::Scalar denL = (std::abs(S_L - S_M) < small)
                                 ? (S_L - S_M >= 0.0 ? small : -small)
                                 : (S_L - S_M);
        const Set::Scalar denR = (std::abs(S_R - S_M) < small)
                                 ? (S_R - S_M >= 0.0 ? small : -small)
                                 : (S_R - S_M);
        const Set::Scalar facL = (S_L - uL) / denL;
        const Set::Scalar facR = (S_R - uR) / denR;

        // Star phase masses; alpha unchanged across acoustic waves (Sau09 line 798).
        const Set::Scalar arh0_starL = lo.alpha_rho_0 * facL;
        const Set::Scalar arh1_starL = lo.alpha_rho_1 * facL;
        const Set::Scalar arh0_starR = hi.alpha_rho_0 * facR;
        const Set::Scalar arh1_starR = hi.alpha_rho_1 * facR;
        const Set::Scalar rho_starL  = rhoL * facL;
        const Set::Scalar rho_starR  = rhoR * facR;

        const Set::Scalar rho0_starL = std::max(arh0_starL / aL,  small);
        const Set::Scalar rho1_starL = std::max(arh1_starL / a2L, small);
        const Set::Scalar rho0_starR = std::max(arh0_starR / aR,  small);
        const Set::Scalar rho1_starR = std::max(arh1_starR / a2R, small);

        // Phasic Hugoniot star pressures (Sau09 eq. II.8).
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

        // Redundant ρE in the star (Toro form, Sau09 line 786):
        const Set::Scalar Etot_starL = rho_starL
                                       * (lo.E_total / rhoL
                                          + (S_M - uL) * (S_M + pL / (rhoL * (S_L - uL))));
        const Set::Scalar Etot_starR = rho_starR
                                       * (hi.E_total / rhoR
                                          + (S_M - uR) * (S_M + pR / (rhoR * (S_R - uR))));

        // L and R Euler fluxes for each conservative row.
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

        // Star fluxes via F*_K = F_K + S_K (U*_K - U_K).
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
                          arh0_starL, arh1_starL,
                          Mn_starL,   Mt_starL,
                          Etot_starL, E0_starL,    E1_starL,
                          arh0L, arh1L,
                          lo.M_normal, lo.M_tangent,
                          lo.E_total,  lo.E0, lo.E1);
        }
        else
        {
            fl = add_star(F_R, S_R,
                          arh0_starR, arh1_starR,
                          Mn_starR,   Mt_starR,
                          Etot_starR, E0_starR,    E1_starR,
                          arh0R, arh1R,
                          hi.M_normal, hi.M_tangent,
                          hi.E_total,  hi.E0, hi.E1);
        }
        fl.u_interface = S_M;

        return fl;
    }
};

}
}
}

#endif
