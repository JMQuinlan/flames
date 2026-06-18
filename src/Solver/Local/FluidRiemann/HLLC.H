// HLLC Riemann solver for the 6-equation diffuse-interface model
// (Saurel-Petitpas-Berry JCP 228, 2009; Schmidmayer-Bryngelson-Colonius
// JCP 402, 2020).
//
// Star-state algebra: Saurel 2009 Section 3.1.2 (lines 740-815).
//   * Wave speeds: Davis estimates from FROZEN mixture sound speed
//     c^2 = Y_1 c_1^2 + Y_2 c_2^2  (Saurel 2009 eq. III.2 /
//     Schmidmayer 2020 eq. 17).  This is the *single* feature that
//     gives the 6-eq HLLC its monotone-in-alpha robustness vs the
//     5-eq Wood form.
//   * Contact speed S_M: standard HLL average from mixture rho, u, p.
//   * Phase masses across the contact: (alpha rho)*_K = (alpha rho)_K *
//     (S_K - u_K)/(S_K - S_M).  (Saurel 2009 line 777.)
//   * Volume fraction is CONSTANT across acoustic waves:
//     alpha*_kL = alpha_kL, alpha*_kR = alpha_kR (Saurel 2009 line 798).
//     The alpha "row" is purely a contact-wave-advected color, handled
//     by upwinding on S_M in the integrator (Saurel 2009 line 836).
//   * Total energy star: Toro form (Saurel 2009 line 786).
//   * Per-phase internal energies across the acoustic star: phasic
//     Hugoniot p_k*(rho_k*) from Saurel 2009 eq. II.8 inverted for the
//     stiffened-gas EOS, then e_k* = (p_k* + gamma_k pi_k) /
//     ((gamma_k-1) rho_k*).
//
// Schmidmayer 2020 eq. 13 confirms the same flux structure.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// 6-equation HLLC Riemann solver.
class HLLC : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc";

    HLLC(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLC(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No tunable parameters.
    }

    /// HLLC star-state algebra per Saurel 2009 §3.1.2.
    /// The face is conceptually between the L (low) and R (hi) state.
    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        // ====================================================================
        // 1. Primitive recovery on each side.
        //    Each State carries (alpha, (alpha rho)_k, M_n, M_t, E_k, rho E,
        //    gamma_k, pi_k).  Derive: mixture rho/u/v/p, per-phase rho_k/e_k/p_k/c_k,
        //    mass fractions Y_k, FROZEN mixture sound speed c.
        //
        // Use a single, *consistent* alpha floor (alpha_floor) for EVERY
        // per-phase divide so the (rho_k_pure, p_k) pair stays
        // thermodynamically consistent.  An aggressive floor (1e-6) feeds
        // measurable noise into c_k, Z_k, Y_k that biases velocity in
        // mixed gas/liquid Riemann problems (Garrick).  1e-12 is the value
        // used in the prior 6-eq stash that produced correct velocities.
        // ====================================================================
        const Set::Scalar alpha_floor = 1.0e-12;

        auto recover = [&](const State &s,
                           Set::Scalar &alpha,    Set::Scalar &alpha2,
                           Set::Scalar &arh0,     Set::Scalar &arh1,
                           Set::Scalar &rho,      Set::Scalar &u,      Set::Scalar &v,
                           Set::Scalar &p_mix,    Set::Scalar &c,
                           Set::Scalar &rho0,     Set::Scalar &rho1,
                           Set::Scalar &p0,       Set::Scalar &p1,
                           Set::Scalar &c0,       Set::Scalar &c1,
                           Set::Scalar &Y0,       Set::Scalar &Y1)
        {
            alpha  = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            alpha2 = 1.0 - alpha;
            arh0   = std::max(s.alpha_rho_0, small);
            arh1   = std::max(s.alpha_rho_1, small);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            // Per-phase pure densities (alpha_floor protects the divide).
            rho0   = std::max(arh0 / alpha,  small);
            rho1   = std::max(arh1 / alpha2, small);

            // Per-phase pressures from canonical (alpha rho e)_k.
            // Saurel 2009 eq. II.3: p_k = (gamma-1) rho_k e_k - gamma pi_k.
            p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, small);
            p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, small);

            // Per-phase sound speeds.
            c0 = Solver::EOS::EOS::PhasicSoundSpeed(rho0, p0, s.gamma0, s.pi0, small);
            c1 = Solver::EOS::EOS::PhasicSoundSpeed(rho1, p1, s.gamma1, s.pi1, small);

            // Mass fractions for the frozen mixture sound speed.
            Y0 = arh0 / rho;
            Y1 = arh1 / rho;

            // Mixture pressure: Schmidmayer 2020 eq. 8.
            p_mix = alpha * p0 + alpha2 * p1;

            // Frozen mixture sound speed: Saurel 2009 eq. III.2 / Schmidmayer 2020 eq. 17.
            c = Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1);
            c = std::max(c, small);
        };

        Set::Scalar aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL;
        Set::Scalar rho0L, rho1L, p0L, p1L, c0L, c1L, Y0L, Y1L;
        recover(lo, aL, a2L, arh0L, arh1L, rhoL, uL, vL, pL, cL,
                rho0L, rho1L, p0L, p1L, c0L, c1L, Y0L, Y1L);

        Set::Scalar aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR;
        Set::Scalar rho0R, rho1R, p0R, p1R, c0R, c1R, Y0R, Y1R;
        recover(hi, aR, a2R, arh0R, arh1R, rhoR, uR, vR, pR, cR,
                rho0R, rho1R, p0R, p1R, c0R, c1R, Y0R, Y1R);

        // Redundant total energy rho E. Use the INDEPENDENTLY-advected
        // primary directly, per Schmidmayer 2020 line 330-333:
        //   "We note that (16) is redundant when the internal energy
        //    equations are also computed. However, in practice we include
        //    it in our computations to ensure that the total energy is
        //    numerically conserved, and thus preserve a correct treatment
        //    of shock waves."
        // The post-stage reinit step (Sch20 eq. 26 / Sau09 eq. III.5) then
        // uses this rho E to recover p and reset the per-phase E_k.  If we
        // overrode the HLLC ρE flux with E_0+E_1+KE we would conservatively
        // advect a different quantity than the "ρE" the reinit step expects.
        const Set::Scalar ETotL = lo.E_total;
        const Set::Scalar ETotR = hi.E_total;

        // ====================================================================
        // 2. Davis wave-speed estimates (Saurel 2009 §3.1.2 line 764):
        //         S_L = min(u_L - c_L, u_R - c_R)
        //         S_R = max(u_L + c_L, u_R + c_R)
        // The prior 6-eq stash used Davis and produced correct velocities.
        // ====================================================================
        Set::Scalar S_L = std::min(uL - cL, uR - cR);
        Set::Scalar S_R = std::max(uL + cL, uR + cR);

        // ====================================================================
        // 3. Contact wave speed S_M (HLL average; Saurel 2009 line 769;
        //    standard Toro / Batten formula):
        //    S_M = [ (rho u^2 + p)_L - (rho u^2 + p)_R - S_L (rho u)_L + S_R (rho u)_R ]
        //          / [ (rho u)_L - (rho u)_R - S_L rho_L + S_R rho_R ]
        // ====================================================================
        Set::Scalar num_SM = (rhoL * uL * uL + pL) - (rhoR * uR * uR + pR)
                           - S_L * rhoL * uL + S_R * rhoR * uR;
        Set::Scalar den_SM = rhoL * uL - rhoR * uR - S_L * rhoL + S_R * rhoR;
        Set::Scalar S_M = num_SM / (den_SM + (den_SM >= 0.0 ? small : -small));

        // ====================================================================
        // 4. Star-state pressure (Toro / Saurel 2009 line 783):
        //    p* = p_K + rho_K (u_K - S_K)(u_K - S_M).
        // ====================================================================
        Set::Scalar p_starL = pL + rhoL * (uL - S_L) * (uL - S_M);
        Set::Scalar p_starR = pR + rhoR * (uR - S_R) * (uR - S_M);
        Set::Scalar p_star  = 0.5 * (p_starL + p_starR);
        (void)p_star; // (we use p_starK below in the flux deltas via U*-U convention)

        // ====================================================================
        // 5. Star phase masses:
        //    (alpha_k rho_k)*_K = (alpha_k rho_k)_K * (S_K - u_K) / (S_K - S_M).
        //    Volume fraction held constant across acoustic waves
        //    (Saurel 2009 line 798): alpha*_K = alpha_K.
        // ====================================================================
        Set::Scalar denL = (S_L - S_M);  if (std::abs(denL) < small) denL = small;
        Set::Scalar denR = (S_R - S_M);  if (std::abs(denR) < small) denR = small;
        Set::Scalar facL = (S_L - uL) / denL;
        Set::Scalar facR = (S_R - uR) / denR;

        Set::Scalar arh0_starL = arh0L * facL;
        Set::Scalar arh1_starL = arh1L * facL;
        Set::Scalar arh0_starR = arh0R * facR;
        Set::Scalar arh1_starR = arh1R * facR;

        Set::Scalar rho_starL = arh0_starL + arh1_starL;
        Set::Scalar rho_starR = arh0_starR + arh1_starR;

        // Per-phase pure-density star (alpha unchanged across acoustic):
        Set::Scalar rho0_starL = std::max(arh0_starL / aL,  small);
        Set::Scalar rho1_starL = std::max(arh1_starL / a2L, small);
        Set::Scalar rho0_starR = std::max(arh0_starR / aR,  small);
        Set::Scalar rho1_starR = std::max(arh1_starR / a2R, small);

        // ====================================================================
        // 6. Star total energy (Toro form, Saurel 2009 line 786):
        //    (rho E)*_K = rho*_K * [ E_K/rho_K + (S_M - u_K) ( S_M + p_K / (rho_K (S_K - u_K)) ) ]
        // ====================================================================
        auto energy_star = [&](Set::Scalar rho, Set::Scalar u, Set::Scalar p, Set::Scalar S,
                               Set::Scalar E, Set::Scalar rho_star) -> Set::Scalar
        {
            Set::Scalar denom_inner = rho * (S - u);
            if (std::abs(denom_inner) < small) denom_inner = (denom_inner >= 0.0 ? small : -small);
            return rho_star * ( E / rho + (S_M - u) * (S_M + p / denom_inner) );
        };
        Set::Scalar E_starL = energy_star(rhoL, uL, pL, S_L, ETotL, rho_starL);
        Set::Scalar E_starR = energy_star(rhoR, uR, pR, S_R, ETotR, rho_starR);

        // Star tangential momentum (pure linear advection: v unchanged across acoustic).
        Set::Scalar Mt_starL = rho_starL * vL;
        Set::Scalar Mt_starR = rho_starR * vR;

        // Star normal momentum (Toro: rho* * S_M).
        Set::Scalar Mn_starL = rho_starL * S_M;
        Set::Scalar Mn_starR = rho_starR * S_M;

        // ====================================================================
        // 7. Per-phase internal energies in the star via phasic Hugoniot.
        //    Saurel 2009 eq. II.8 -> p_k*(rho_k*) for SG, then
        //    e_k* = (p_k* + gamma_k pi_k) / ((gamma_k - 1) rho_k*).
        //    Build  (alpha_k rho_k e_k)*  =  alpha_k_K * rho_k*_K * e_k*_K.
        // ====================================================================
        auto phasic_E_star = [&](Set::Scalar rho_k_star, Set::Scalar rho_k, Set::Scalar p_k,
                                 Set::Scalar gam, Set::Scalar pi_k, Set::Scalar alpha_k) -> Set::Scalar
        {
            Set::Scalar p_k_star = Solver::EOS::EOS::PhasicHugoniotPressure(rho_k_star, rho_k, p_k, gam, pi_k, small);
            // Floor:
            Set::Scalar pp_star = std::max(p_k_star + gam * pi_k, small);
            Set::Scalar e_k_star = pp_star / std::max((gam - 1.0) * rho_k_star, small);
            return alpha_k * rho_k_star * e_k_star;
        };

        Set::Scalar E0_starL = phasic_E_star(rho0_starL, rho0L, p0L, lo.gamma0, lo.pi0, aL);
        Set::Scalar E1_starL = phasic_E_star(rho1_starL, rho1L, p1L, lo.gamma1, lo.pi1, a2L);
        Set::Scalar E0_starR = phasic_E_star(rho0_starR, rho0R, p0R, hi.gamma0, hi.pi0, aR);
        Set::Scalar E1_starR = phasic_E_star(rho1_starR, rho1R, p1R, hi.gamma1, hi.pi1, a2R);

        // ====================================================================
        // 8. Build L and R Euler-like fluxes for the conservative components.
        //    All per-phase advection fluxes use the L/R velocity; the HLLC
        //    dispatch in step 9 chooses which side to use.
        // ====================================================================
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

        Flux F_L = build_flux(uL, vL, pL, arh0L, arh1L, lo.E0, lo.E1, ETotL, rhoL);
        Flux F_R = build_flux(uR, vR, pR, arh0R, arh1R, hi.E0, hi.E1, ETotR, rhoR);

        // Star fluxes via the HLLC formula F*_K = F_K + S_K (U*_K - U_K).
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

        Flux Fstar_L = add_star(F_L, S_L,
                                arh0_starL, arh1_starL,
                                Mn_starL,   Mt_starL,
                                E_starL,    E0_starL,    E1_starL,
                                arh0L, arh1L,
                                lo.M_normal, lo.M_tangent,
                                ETotL,  lo.E0, lo.E1);
        Flux Fstar_R = add_star(F_R, S_R,
                                arh0_starR, arh1_starR,
                                Mn_starR,   Mt_starR,
                                E_starR,    E0_starR,    E1_starR,
                                arh0R, arh1R,
                                hi.M_normal, hi.M_tangent,
                                ETotR,  hi.E0, hi.E1);

        // ====================================================================
        // 9. HLLC dispatch by wave positions.
        //
        // u_interface is the contact-wave velocity used by the Saurel-Abgrall
        // non-conservative discretization (alpha row and per-phase internal-
        // energy rows). Saurel 2009 line 836 prescribes the HLLC contact
        // speed S_M in the star region, and the upwind cell velocity u_K
        // outside the star fan. Schmidmayer 2020 eq. 18 follows the same
        // convention (u_s = HLLC contact velocity).
        // ====================================================================
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = F_L;
            fl.u_interface = uL;
        }
        else if (0.0 <= S_M)
        {
            fl = Fstar_L;
            fl.u_interface = S_M;   // Sau09 line 836 / Sch20 eq. 18
        }
        else if (0.0 <= S_R)
        {
            fl = Fstar_R;
            fl.u_interface = S_M;   // Sau09 line 836 / Sch20 eq. 18
        }
        else
        {
            fl = F_R;
            fl.u_interface = uR;
        }

        // Second tangential momentum (3D): the tangential velocity is constant
        // across the acoustic waves and is simply advected across the contact,
        // exactly like momentum_tangent.  In every wave case the existing
        // momentum_tangent equals (total mass flux) * v_t(upwind of contact), so
        // the second tangent is that same mass flux times w_t2(upwind).  In 2D
        // M_tangent2 == 0, so this is identically zero (no-op).
        {
            const Set::Scalar wL = lo.M_tangent2 / rhoL;
            const Set::Scalar wR = hi.M_tangent2 / rhoR;
            fl.momentum_tangent2 = (fl.mass0 + fl.mass1) * ((S_M >= 0.0) ? wL : wR);
        }

        // ====================================================================
        // Error checking
        // ====================================================================
        if ( (fl.mass0 != fl.mass0)
          or (fl.mass1 != fl.mass1)
          or (fl.momentum_normal  != fl.momentum_normal)
          or (fl.momentum_tangent != fl.momentum_tangent)
          or (fl.momentum_tangent2 != fl.momentum_tangent2)
          or (fl.energy_total     != fl.energy_total)
          or (fl.energy0          != fl.energy0)
          or (fl.energy1          != fl.energy1)
          or (cL != cL) or (cR != cR) )
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "ERROR IN 6-eq HLLC SOLVER:");
            Util::ParallelMessage(INFO, "lo:   ", lo);
            Util::ParallelMessage(INFO, "hi:   ", hi);
            Util::ParallelMessage(INFO, "flux: ", fl);
            Util::ParallelMessage(INFO, "rhoL=", rhoL, " rhoR=", rhoR);
            Util::ParallelMessage(INFO, "uL=",  uL,  " uR=",  uR);
            Util::ParallelMessage(INFO, "pL=",  pL,  " pR=",  pR);
            Util::ParallelMessage(INFO, "p0L=", p0L, " p1L=", p1L);
            Util::ParallelMessage(INFO, "p0R=", p0R, " p1R=", p1R);
            Util::ParallelMessage(INFO, "cL=",  cL,  " cR=",  cR);
            Util::ParallelMessage(INFO, "S_L=", S_L, " S_M=", S_M, " S_R=", S_R);
            Util::Exception(INFO);
        }

        return fl;
    }
};

}
}
}

#endif
