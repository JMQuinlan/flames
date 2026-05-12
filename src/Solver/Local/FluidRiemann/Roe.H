// Roe approximate Riemann solver for the 6-equation diffuse-interface
// model (Saurel-Petitpas-Berry JCP 228, 2009; Schmidmayer-Bryngelson-Colonius
// JCP 402, 2020).
//
// Roe linearization with multispecies (Larrouturou JCP 1991 / Toumi JCP 1996)
// extension for the per-phase masses:
//
//     F_Roe.k = ½ (F_L.k + F_R.k) − ½ Σ_w |λ_w| α_w r_w.k
//
// Wave structure at the Roe state (8 conservative components in 2D):
//     λ_1 = û − ĉ      (mixture acoustic, normal direction)
//     λ_2 = û          (LD; multiplicity 6: entropy, shear, two species, two phasic-energy)
//     λ_3 = û + ĉ      (mixture acoustic, normal direction)
//
// Wave amplitudes (left-eigenvector projections of ΔU):
//     α_1     = (Δp − ρ̂ĉ Δu) / (2ĉ²)             [u−c acoustic]
//     α_3     = (Δp + ρ̂ĉ Δu) / (2ĉ²)             [u+c acoustic]
//     α_2_ρ   = Δρ − Δp/ĉ²                        [entropy / contact density]
//     α_2_v   = ρ̂ Δu_t                            [shear]
//     α_Y_k   = Δ(αρ)_k − Y_k_hat Δρ              [species k LD]
//     α_E_k   = ΔE_k                              [phasic internal-energy LD]
//
// Component dissipations:
//     D_ρ      = |λ_1| α_1 + |λ_2| α_2_ρ + |λ_3| α_3                                (mixture mass)
//     D_M_n    = |λ_1| α_1 (û−ĉ) + |λ_2| α_2_ρ û + |λ_3| α_3 (û+ĉ)                  (mixture normal momentum)
//     D_M_t    = D_ρ v̂ + |λ_2| α_2_v                                                (mixture tangential momentum)
//     D_ρE     = |λ_1| α_1 (Ĥ−ûĉ) + |λ_2| α_2_ρ ke_hat + |λ_2| α_2_v v̂ + |λ_3| α_3 (Ĥ+ûĉ)
//     D_(αρ)_k = Y_k_hat D_ρ + |λ_2| α_Y_k                                          (Larrouturou)
//     D_E_k    = |λ_2| α_E_k                                                        (pure LD)
//
// Roe-averaged primitives use Vinokur-Liu sqrt-ρ weighting; the Roe-state
// pressure p̂ is recovered from H_hat through the mixture-effective
// stiffened-gas EOS (γ_eff, π_eff weighted by α_hat).  The Roe sound speed
// is the FROZEN mixture form (Sch20 eq. 17 algebraic identity).  An optional
// Harten entropy fix on the acoustic eigenvalues guards against vanishing
// dissipation at transonic rarefactions when entropy_fix > 0.
//
// u_interface for the integrator's Saurel-Abgrall non-conservative
// discretization (Sau09 line 836) is the upwind face velocity û in the
// subsonic regime, or u_L / u_R in the supersonic fast paths.
//
// References for the multispecies / multiphase Roe extension:
//   Larrouturou, JCP 95 (1991) 59-84 (multi-species Roe).
//   Toumi, JCP 124 (1996) 286-300 (two-fluid Roe).
//   Pelanti & Shyue, JCP 259 (2014) §3 (6-eq mixture Roe-type).

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_ROE_H
#define SOLVER_LOCAL_FLUIDRIEMANN_ROE_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class Roe : public FluidRiemann
{
public:
    static constexpr const char *name = "roe";

    int            entropy_fix      = 0;     // 0 = off, 1 = Harten fix on acoustics
    Set::Scalar    entropy_fix_eps  = 0.05;  // Harten threshold parameter (fraction of c_hat)

    Roe(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Roe(IO::ParmParse &pp) { pp_queryclass(*this); }
    Roe()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Roe &value, IO::ParmParse &pp)
    {
        pp.query_default("entropy_fix",     value.entropy_fix,     0);
        pp.query_default("entropy_fix_eps", value.entropy_fix_eps, 0.05);
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small, int /*Spec_Vol*/ = 1) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        // ============================================================
        // 1. Per-side primitive recovery
        // ============================================================
        auto recover = [&](const State &s,
                           Set::Scalar &alpha,  Set::Scalar &alpha2,
                           Set::Scalar &rho,    Set::Scalar &u,    Set::Scalar &v,
                           Set::Scalar &p,      Set::Scalar &c,    Set::Scalar &H,
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
            p  = alpha * p0 + alpha2 * p1;                                  // Sch20 eq. 8
            c  = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1), small); // Sch20 eq. 17
            H  = (s.E_total + p) / rho;                                     // mixture stagnation enthalpy
        };

        Set::Scalar aL, a2L, rhoL, uL, vL, pL, cL, HL, Y0L, Y1L;
        Set::Scalar aR, a2R, rhoR, uR, vR, pR, cR, HR, Y0R, Y1R;
        recover(lo, aL, a2L, rhoL, uL, vL, pL, cL, HL, Y0L, Y1L);
        recover(hi, aR, a2R, rhoR, uR, vR, pR, cR, HR, Y0R, Y1R);

        // ============================================================
        // 2. Roe averages (Vinokur-Liu sqrt-ρ weighting)
        // ============================================================
        const Set::Scalar sqrtL = std::sqrt(rhoL);
        const Set::Scalar sqrtR = std::sqrt(rhoR);
        const Set::Scalar wsum  = std::max(sqrtL + sqrtR, small);

        const Set::Scalar rho_hat   = sqrtL * sqrtR;
        const Set::Scalar u_hat     = (sqrtL * uL + sqrtR * uR) / wsum;
        const Set::Scalar v_hat     = (sqrtL * vL + sqrtR * vR) / wsum;
        const Set::Scalar H_hat     = (sqrtL * HL + sqrtR * HR) / wsum;
        const Set::Scalar Y0_hat    = (sqrtL * Y0L + sqrtR * Y0R) / wsum;
        const Set::Scalar Y1_hat    = 1.0 - Y0_hat;
        const Set::Scalar alpha_hat = std::min(std::max((sqrtL * aL + sqrtR * aR) / wsum, alpha_floor), 1.0 - alpha_floor);
        const Set::Scalar alpha2_hat = 1.0 - alpha_hat;

        // ============================================================
        // 3. Roe-state pressure (via mixture-effective stiffened-gas
        //    inversion of H_hat) and FROZEN mixture sound speed.
        // ============================================================
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

        // Frozen mixture sound speed at the Roe state (Sch20 eq. 17 in
        // mechanical-equilibrium algebraic form).
        const Set::Scalar c2_hat = (alpha_hat  * lo.gamma0 * (p_hat + lo.pi0)
                                  + alpha2_hat * lo.gamma1 * (p_hat + lo.pi1))
                                  / std::max(rho_hat, small);
        const Set::Scalar c_hat  = std::sqrt(std::max(c2_hat, small * small));

        // ============================================================
        // 4. Jumps in the conservative state and primitives.
        // ============================================================
        const Set::Scalar Drho   = rhoR - rhoL;
        const Set::Scalar Du     = uR   - uL;
        const Set::Scalar Dv     = vR   - vL;
        const Set::Scalar Dp     = pR   - pL;
        const Set::Scalar Darh0  = hi.alpha_rho_0 - lo.alpha_rho_0;
        const Set::Scalar Darh1  = hi.alpha_rho_1 - lo.alpha_rho_1;
        const Set::Scalar DE0    = hi.E0      - lo.E0;
        const Set::Scalar DE1    = hi.E1      - lo.E1;

        // ============================================================
        // 5. Wave amplitudes at the Roe state.
        // ============================================================
        const Set::Scalar a1     = (Dp - rho_hat * c_hat * Du) / (2.0 * c2_hat);    // u−c
        const Set::Scalar a3     = (Dp + rho_hat * c_hat * Du) / (2.0 * c2_hat);    // u+c
        const Set::Scalar a2_rho = Drho - Dp / c2_hat;                              // entropy density
        const Set::Scalar a2_v   = rho_hat * Dv;                                    // shear momentum
        const Set::Scalar aY0    = Darh0 - Y0_hat * Drho;                           // species 0 LD
        const Set::Scalar aY1    = Darh1 - Y1_hat * Drho;                           // species 1 LD
        // E_k waves are pure LD with amplitude ΔE_k (eigenvector is the unit
        // basis vector for E_k); they appear only in D_{E_k} below.

        // ============================================================
        // 6. Wave-speed magnitudes (with optional Harten entropy fix on
        //    the acoustic eigenvalues to avoid vanishing dissipation at
        //    transonic rarefactions).
        // ============================================================
        Set::Scalar abs_l1 = std::abs(u_hat - c_hat);
        Set::Scalar abs_l2 = std::abs(u_hat);
        Set::Scalar abs_l3 = std::abs(u_hat + c_hat);
        if (entropy_fix == 1)
        {
            const Set::Scalar epsH = entropy_fix_eps * c_hat;
            if (abs_l1 < epsH) abs_l1 = 0.5 * (abs_l1 * abs_l1 / epsH + epsH);
            if (abs_l3 < epsH) abs_l3 = 0.5 * (abs_l3 * abs_l3 / epsH + epsH);
        }

        // ============================================================
        // 7. Conservative-component dissipations.
        // ============================================================
        // Mixture mass: dissipation from acoustic+entropy waves only.
        const Set::Scalar D_rho = abs_l1 * a1 + abs_l2 * a2_rho + abs_l3 * a3;

        // Mixture normal momentum: each wave contributes (eigenvalue) × amplitude.
        const Set::Scalar D_Mn = abs_l1 * a1 * (u_hat - c_hat)
                               + abs_l2 * a2_rho * u_hat
                               + abs_l3 * a3 * (u_hat + c_hat);

        // Mixture tangential momentum: acoustic+entropy waves carry v̂·D_ρ;
        // the shear wave adds the LD ρ̂·Δv jump.
        const Set::Scalar D_Mt = v_hat * D_rho + abs_l2 * a2_v;

        // Mixture total energy ρE: acoustic ± ûĉ Hugoniot dispersion, entropy ½(û²+v̂²),
        // plus shear advection of v̂.
        const Set::Scalar D_rhoE = abs_l1 * a1 * (H_hat - u_hat * c_hat)
                                 + abs_l2 * a2_rho * ke_hat
                                 + abs_l2 * a2_v * v_hat
                                 + abs_l3 * a3 * (H_hat + u_hat * c_hat);

        // Per-phase masses (αρ)_k: Larrouturou-style — the acoustic waves
        // carry mass in proportion to the Roe-averaged mass fraction
        // Y_k_hat, plus a pure LD term for the species jump residual.
        const Set::Scalar D_arh0 = Y0_hat * D_rho + abs_l2 * aY0;
        const Set::Scalar D_arh1 = Y1_hat * D_rho + abs_l2 * aY1;

        // Per-phase internal energies E_k: pure LD transport (the
        // pressure-work α_k p_k ∇·u is the non-conservative source that
        // the integrator applies separately, not part of the Riemann flux).
        const Set::Scalar D_E0 = abs_l2 * DE0;
        const Set::Scalar D_E1 = abs_l2 * DE1;

        // ============================================================
        // 8. Build L and R Euler-like fluxes.
        // ============================================================
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

        // ============================================================
        // 9. Roe flux assembly: F.k = ½(F_L.k + F_R.k) − ½ D.k.
        // ============================================================
        Flux fl;
        fl.mass0            = 0.5 * (F_L.mass0            + F_R.mass0)            - 0.5 * D_arh0;
        fl.mass1            = 0.5 * (F_L.mass1            + F_R.mass1)            - 0.5 * D_arh1;
        fl.momentum_normal  = 0.5 * (F_L.momentum_normal  + F_R.momentum_normal)  - 0.5 * D_Mn;
        fl.momentum_tangent = 0.5 * (F_L.momentum_tangent + F_R.momentum_tangent) - 0.5 * D_Mt;
        fl.energy_total     = 0.5 * (F_L.energy_total     + F_R.energy_total)     - 0.5 * D_rhoE;
        fl.energy0          = 0.5 * (F_L.energy0          + F_R.energy0)          - 0.5 * D_E0;
        fl.energy1          = 0.5 * (F_L.energy1          + F_R.energy1)          - 0.5 * D_E1;

        // ============================================================
        // 10. u_interface for Saurel-Abgrall non-conservative rows.
        //     Supersonic upwind on each side; otherwise the Roe-averaged
        //     face velocity û (the LD eigenvalue carrying α and the
        //     per-phase fields).
        // ============================================================
        if      (uL - cL >= 0.0 && uR - cR >= 0.0) fl.u_interface = uL;
        else if (uL + cL <= 0.0 && uR + cR <= 0.0) fl.u_interface = uR;
        else                                       fl.u_interface = u_hat;

        return fl;
    }
};

}
}
}

#endif
