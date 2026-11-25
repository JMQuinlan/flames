// HLLC Riemann solver for the 6-equation diffuse-interface model with a
// HYPERELASTIC SOLID phase (Hydro2 elastic.on).
//
// Reference: N. Favrie, S. L. Gavrilyuk, R. Saurel, "Solid-fluid diffuse
// interface model in cases of extreme deformations", JCP 228 (2009) 6037-6077,
// Section 4.2 "HLLC type Riemann solver" -- the extension of the
// Saurel-Petitpas-Berry solver in HLLC.H to a mixture stress
//     sigma = -p I + S,     S = alpha_s dev(sigma_s)   (State::S_nn, S_nt, S_nt2).
//
//   * wave speeds: Davis, with the longitudinal mixture speed
//       c_l^2 = Y_s (c_s^2 + (4/3) mu_s / rho_s) + Y_g c_g^2        (Sec. 3.5)
//   * contact speed:  S_M = HLL average with p -> -sigma_11          (p. 6056)
//   * sigma*_11K = sigma_11K - rho_K (u_K - S_K)(u_K - S_M)
//   * sigma*_12 (and 13): continuous across the contact ONLY for a
//     solid|solid face (both sides alpha_s > 1/2); zero otherwise, i.e. a
//     fluid slides on a solid and a solid surface next to fluid is
//     traction-free in shear (Sec. 5.6, the Phi level-set rule):
//       sigma*_12 = [m_R s12_L - m_L s12_R + m_L m_R (v_R - v_L)] / (m_R - m_L),
//       m_K = rho_K (u_K - S_K)
//   * v*_K = v_K + (sigma*_12 - sigma_12K) / m_K
//   * (rho E)*_K = [ (rho E)_K (u_K - S_K) - sigma_11K u_K - sigma_12K v_K
//                    + sigma*_11 S_M + sigma*_12 v*_K ] / (S_M - S_K)
//   * F = ( (alpha rho)_k u, rho u^2 - sigma_11, rho u v - sigma_12,
//           (rho E - sigma_11) u - sigma_12 v ),   F*_K = F_K + S_K (U*_K - U_K)
// With S = 0 and c2_el = 0 this is the fluid solver of HLLC.H.
//
// Phase masses, volume fraction and the per-phase (hydrodynamic) internal
// energies are treated exactly as in HLLC.H.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ELASTIC_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_ELASTIC_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// 6-equation HLLC Riemann solver with elastic solid stress (Favrie 2009).
class HLLC_Elastic : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_elastic";

    HLLC_Elastic(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLC_Elastic(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_Elastic()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC_Elastic & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No tunable parameters.
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        // 1. Primitive recovery on each side (identical to HLLC.H).
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
            // Phase masses clamped at ZERO only: these feed the advective
            // fluxes directly, and a positive floor (small) would export
            // ~small*u of fictitious trace-phase mass through every
            // pure-phase face.  Divides below are protected by rho/alpha.
            arh0   = std::max(s.alpha_rho_0, 0.0);
            arh1   = std::max(s.alpha_rho_1, 0.0);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            // Per-phase pure densities (alpha_floor protects the divide).
            rho0   = std::max(arh0 / alpha,  small);
            rho1   = std::max(arh1 / alpha2, small);

            // Per-phase pressures from canonical (alpha rho e)_k.
            // Saurel 2009 eq. II.3: p_k = (gamma-1) rho_k e_k - gamma pi_k.
            // Pass alpha_floor (not small=1e-8) as the divide floor: alpha is
            // already clamped at alpha_floor above, and the helper's internal
            // max(alpha, small) would otherwise re-floor trace-phase cells at
            // 1e-8, railing p_k to -pi_k there (see the alpha_floor comment).
            p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, alpha_floor);
            p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, alpha_floor);

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

        const Set::Scalar ETotL = lo.E_total;
        const Set::Scalar ETotR = hi.E_total;

        // Longitudinal mixture speed with the elastic part (Favrie 2009 Sec. 3.5).
        cL = std::sqrt(cL * cL + std::max(lo.c2_el, 0.0));
        cR = std::sqrt(cR * cR + std::max(hi.c2_el, 0.0));

        // Mixture stress on the face: sigma_11 = -p + S_nn, sigma_12 = S_nt, sigma_13 = S_nt2.
        const Set::Scalar s11L = -pL + lo.S_nn, s11R = -pR + hi.S_nn;
        const Set::Scalar s12L = lo.S_nt,       s12R = hi.S_nt;
        const Set::Scalar s13L = lo.S_nt2,      s13R = hi.S_nt2;
        const Set::Scalar wL = lo.M_tangent2 / rhoL;
        const Set::Scalar wR = hi.M_tangent2 / rhoR;

        // 2. Davis wave-speed estimates.
        Set::Scalar S_L = std::min(uL - cL, uR - cR);
        Set::Scalar S_R = std::max(uL + cL, uR + cR);

        // 3. Contact speed (HLL average, p -> -sigma_11).
        Set::Scalar num_SM = (rhoL * uL * uL - s11L) - (rhoR * uR * uR - s11R)
                           - S_L * rhoL * uL + S_R * rhoR * uR;
        Set::Scalar den_SM = rhoL * uL - rhoR * uR - S_L * rhoL + S_R * rhoR;
        Set::Scalar S_M = num_SM / (den_SM + (den_SM >= 0.0 ? small : -small));

        // 4. Star normal stress on each side of the contact.
        const Set::Scalar s11_starL = s11L - rhoL * (uL - S_L) * (uL - S_M);
        const Set::Scalar s11_starR = s11R - rhoR * (uR - S_R) * (uR - S_M);

        // Star shear stress and tangential velocities.  m_L > 0 > m_R.
        Set::Scalar mL = rhoL * (uL - S_L);  if (std::abs(mL) < small) mL = small;
        Set::Scalar mR = rhoR * (uR - S_R);  if (std::abs(mR) < small) mR = -small;
        const bool stick = (lo.solid != 0) && (hi.solid != 0);
        Set::Scalar s12_star = 0.0, s13_star = 0.0;
        if (stick)
        {
            Set::Scalar den = mR - mL;  if (std::abs(den) < small) den = -small;
            s12_star = (mR * s12L - mL * s12R + mL * mR * (vR - vL)) / den;
            s13_star = (mR * s13L - mL * s13R + mL * mR * (wR - wL)) / den;
        }
        // OBJECTIVE SHEAR DISSIPATION.  The term m_L m_R (v_R - v_L)/(m_R - m_L) in sigma*_12 is the
        // upwind dissipation of the shear wave; taken alone it acts on d(v)/dx only, i.e. it is a
        // numerical viscosity on ROTATION as well as on shear (measured: a spinning elastic disc lost
        // 10 % of its spin per quarter turn, 96 % at first order, at constant total energy).  ds12 is
        // the same coefficient applied to dx d(u)/dy (State::dvt), so that the total acts on the
        // symmetric gradient d(v)/dx + d(u)/dy.  It is added to the momentum / energy fluxes only:
        // the star velocities, which feed the cobasis rows, keep the 1D jump relations.
        Set::Scalar ds12 = 0.0, ds13 = 0.0;
        if (stick)
        {
            Set::Scalar den = mR - mL;  if (std::abs(den) < small) den = -small;
            ds12 = mL * mR * 0.5 * (lo.dvt  + hi.dvt ) / den;
            ds13 = mL * mR * 0.5 * (lo.dvt2 + hi.dvt2) / den;
        }
        const Set::Scalar v_starL = vL + (s12_star - s12L) / mL;
        const Set::Scalar v_starR = vR + (s12_star - s12R) / mR;
        const Set::Scalar w_starL = wL + (s13_star - s13L) / mL;
        const Set::Scalar w_starR = wR + (s13_star - s13R) / mR;

        // 5. Star phase masses; alpha constant across the acoustic waves.
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

        Set::Scalar rho0_starL = std::max(arh0_starL / aL,  small);
        Set::Scalar rho1_starL = std::max(arh1_starL / a2L, small);
        Set::Scalar rho0_starR = std::max(arh0_starR / aR,  small);
        Set::Scalar rho1_starR = std::max(arh1_starR / a2R, small);

        // 6. Star total energy (Favrie 2009 p. 6056).
        //    (rho E)*_K (S_M - S_K) = (rho E)_K (u_K - S_K) - sigma_K . v_K + sigma*_K . v*_K
        const Set::Scalar E_starL = ( ETotL * (uL - S_L) - s11L * uL - s12L * vL - s13L * wL
                                    + s11_starL * S_M + s12_star * v_starL + s13_star * w_starL ) / (-denL);
        const Set::Scalar E_starR = ( ETotR * (uR - S_R) - s11R * uR - s12R * vR - s13R * wR
                                    + s11_starR * S_M + s12_star * v_starR + s13_star * w_starR ) / (-denR);

        // 7. Per-phase (hydrodynamic) internal energies in the star: phasic Hugoniot, as HLLC.H.
        auto phasic_E_star = [&](Set::Scalar rho_k_star, Set::Scalar rho_k, Set::Scalar p_k,
                                 Set::Scalar gam, Set::Scalar pi_k, Set::Scalar alpha_k) -> Set::Scalar
        {
            Set::Scalar p_k_star = Solver::EOS::EOS::PhasicHugoniotPressure(rho_k_star, rho_k, p_k, gam, pi_k, small);
            Set::Scalar pp_star = std::max(p_k_star + gam * pi_k, small);
            Set::Scalar e_k_star = pp_star / std::max((gam - 1.0) * rho_k_star, small);
            return alpha_k * rho_k_star * e_k_star;
        };
        Set::Scalar E0_starL = phasic_E_star(rho0_starL, rho0L, p0L, lo.gamma0, lo.pi0, aL);
        Set::Scalar E1_starL = phasic_E_star(rho1_starL, rho1L, p1L, lo.gamma1, lo.pi1, a2L);
        Set::Scalar E0_starR = phasic_E_star(rho0_starR, rho0R, p0R, hi.gamma0, hi.pi0, aR);
        Set::Scalar E1_starR = phasic_E_star(rho1_starR, rho1R, p1R, hi.gamma1, hi.pi1, a2R);

        // 8. Left / right physical fluxes and star fluxes F*_K = F_K + S_K (U*_K - U_K).
        auto build_flux = [&](Set::Scalar u, Set::Scalar v, Set::Scalar w,
                              Set::Scalar s11, Set::Scalar s12, Set::Scalar s13,
                              Set::Scalar arh0, Set::Scalar arh1,
                              Set::Scalar E0, Set::Scalar E1, Set::Scalar Etot,
                              Set::Scalar rho) -> Flux
        {
            Flux f;
            f.mass0             = arh0 * u;
            f.mass1             = arh1 * u;
            f.momentum_normal   = rho * u * u - s11;
            f.momentum_tangent  = rho * u * v - s12;
            f.momentum_tangent2 = rho * u * w - s13;
            f.energy_total      = u * (Etot - s11) - s12 * v - s13 * w;
            f.energy0           = E0 * u;
            f.energy1           = E1 * u;
            return f;
        };
        Flux F_L = build_flux(uL, vL, wL, s11L, s12L, s13L, arh0L, arh1L, lo.E0, lo.E1, ETotL, rhoL);
        Flux F_R = build_flux(uR, vR, wR, s11R, s12R, s13R, arh0R, arh1R, hi.E0, hi.E1, ETotR, rhoR);

        auto add_star = [&](const Flux &F_K, Set::Scalar S,
                            Set::Scalar arh0_star, Set::Scalar arh1_star, Set::Scalar rho_star,
                            Set::Scalar v_star, Set::Scalar w_star,
                            Set::Scalar E_star, Set::Scalar E0_star, Set::Scalar E1_star,
                            Set::Scalar arh0, Set::Scalar arh1, const State &s,
                            Set::Scalar Etot_K) -> Flux
        {
            Flux f;
            f.mass0             = F_K.mass0             + S * (arh0_star - arh0);
            f.mass1             = F_K.mass1             + S * (arh1_star - arh1);
            f.momentum_normal   = F_K.momentum_normal   + S * (rho_star * S_M    - s.M_normal);
            f.momentum_tangent  = F_K.momentum_tangent  + S * (rho_star * v_star - s.M_tangent);
            f.momentum_tangent2 = F_K.momentum_tangent2 + S * (rho_star * w_star - s.M_tangent2);
            f.energy_total      = F_K.energy_total      + S * (E_star  - Etot_K);
            f.energy0           = F_K.energy0           + S * (E0_star - s.E0);
            f.energy1           = F_K.energy1           + S * (E1_star - s.E1);
            return f;
        };

        // 9. HLLC dispatch.  u_interface / vt / wt feed the non-conservative
        //    rows (alpha, phase energies, cobasis) in the integrator.
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = F_L;
            fl.u_interface = uL;
            fl.vt_L = vL;  fl.vt_R = vL;
            fl.wt_L = wL;  fl.wt_R = wL;
        }
        else if (0.0 <= S_M)
        {
            fl = add_star(F_L, S_L, arh0_starL, arh1_starL, rho_starL, v_starL, w_starL,
                          E_starL, E0_starL, E1_starL, arh0L, arh1L, lo, ETotL);
            fl.u_interface = S_M;
            fl.vt_L = v_starL;  fl.vt_R = v_starR;
            fl.wt_L = w_starL;  fl.wt_R = w_starR;
            fl.momentum_tangent  -= ds12;
            fl.momentum_tangent2 -= ds13;
            fl.energy_total      -= ds12 * 0.5 * (v_starL + v_starR) + ds13 * 0.5 * (w_starL + w_starR);
        }
        else if (0.0 <= S_R)
        {
            fl = add_star(F_R, S_R, arh0_starR, arh1_starR, rho_starR, v_starR, w_starR,
                          E_starR, E0_starR, E1_starR, arh0R, arh1R, hi, ETotR);
            fl.u_interface = S_M;
            fl.vt_L = v_starL;  fl.vt_R = v_starR;
            fl.wt_L = w_starL;  fl.wt_R = w_starR;
            fl.momentum_tangent  -= ds12;
            fl.momentum_tangent2 -= ds13;
            fl.energy_total      -= ds12 * 0.5 * (v_starL + v_starR) + ds13 * 0.5 * (w_starL + w_starR);
        }
        else
        {
            fl = F_R;
            fl.u_interface = uR;
            fl.vt_L = vR;  fl.vt_R = vR;
            fl.wt_L = wR;  fl.wt_R = wR;
        }

        if ( (fl.mass0 != fl.mass0)
          or (fl.mass1 != fl.mass1)
          or (fl.momentum_normal   != fl.momentum_normal)
          or (fl.momentum_tangent  != fl.momentum_tangent)
          or (fl.momentum_tangent2 != fl.momentum_tangent2)
          or (fl.energy_total      != fl.energy_total)
          or (fl.energy0           != fl.energy0)
          or (fl.energy1           != fl.energy1)
          or (cL != cL) or (cR != cR) )
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "ERROR IN 6-eq ELASTIC HLLC SOLVER:");
            Util::ParallelMessage(INFO, "lo:   ", lo);
            Util::ParallelMessage(INFO, "hi:   ", hi);
            Util::ParallelMessage(INFO, "flux: ", fl);
            Util::ParallelMessage(INFO, "S_nn L/R=", lo.S_nn, " ", hi.S_nn, " S_nt L/R=", lo.S_nt, " ", hi.S_nt);
            Util::ParallelMessage(INFO, "rhoL=", rhoL, " rhoR=", rhoR, " uL=", uL, " uR=", uR);
            Util::ParallelMessage(INFO, "pL=",  pL,  " pR=",  pR, " cL=", cL, " cR=", cR);
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
