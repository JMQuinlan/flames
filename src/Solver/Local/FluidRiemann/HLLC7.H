// Phase-by-phase HLLC Riemann solver for the 7-equation Baer-Nunziato model.
//
// PHASE B implementation (stepping stone, not the final solver).
//
// Strategy: solve a separate single-phase HLLC for each phase using its own
// (rho_k, u_k, E_k) and (gam_k, p_inf_k). Return per-phase fluxes plus an
// interface star-state (u_I*, p_I*). This is "Option B" in the conversion
// plan: it is exact when one of alpha_k -> 0 or 1 (single-phase regions),
// which is enough to satisfy the Phase-B gate (Toro/Garrick must reproduce
// 5-eqn results).
//
// In the diffuse band where both phases co-exist with non-trivial alpha, this
// solver loses the BN cross-phase wave coupling and the relaxation loop has
// to do more work to drag p_0 -> p_1 and u_0 -> u_1. The full BN-HLLC
// (Tokareva-Toro 2010 / SPB-2009) lands in Phase C and replaces this code.
//
// Closures for u_I* and p_I*:
//     u_I* = (alpha_0 rho_0 u_0* + alpha_1 rho_1 u_1*) / (alpha_0 rho_0 + alpha_1 rho_1)
//             (mass-weighted average of per-phase HLLC contact velocities)
//     p_I* = (Z_1 p_0* + Z_0 p_1*) / (Z_0 + Z_1),   Z_k = rho_k a_k
//             (acoustic-impedance-weighted average; Schmidmayer 2017 Eq. 14
//              p. 10 line 643; SPB-2009 Eq. 15)
// where p_k* and u_k* come from each phase's HLLC star state (taking the
// state on the side of the contact corresponding to the sign of u_I*).
//
#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC7_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC7_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"
#include "Util/Util.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class HLLC7 : public FluidRiemann7
{
public:
    static constexpr const char *name = "hllc7";

    HLLC7() {}
    HLLC7(IO::ParmParse &pp, std::string a_name) { pp_queryclass(a_name, *this); }
    HLLC7(IO::ParmParse &pp) { pp_queryclass(*this); }

    static void Parse(HLLC7 & /*value*/, IO::ParmParse & /*pp*/)
    {
        // No parameters in Phase B. Phase C will introduce wavespeed
        // estimator switches (Davis vs HLL vs roe-averaged).
    }

    /// Per-phase HLLC result returned by SolveOnePhase().
    struct PhaseFlux
    {
        Set::Scalar mass     = 0.0;
        Set::Scalar momn     = 0.0;
        Set::Scalar momt     = 0.0;
        Set::Scalar energy   = 0.0;
        Set::Scalar u_star   = 0.0;   // contact wave speed S* (or u_L/u_R if supersonic)
        Set::Scalar p_star   = 0.0;   // pressure on the upwind side of the contact
        Set::Scalar rho_face = 0.0;   // rho on the upwind side of the contact (for Z_k)
        Set::Scalar a_face   = 0.0;   // sound speed on the upwind side (for Z_k)
    };

    virtual Flux7 Solve(const State7 &L, const State7 &R,
                        Set::Scalar small) override
    {
        // Single-phase HLLC, applied independently to each phase. The
        // alpha-weighting is applied later by the integrator's RHS when it
        // assembles d(alpha_k * X)/dt = -div(alpha_face * F_k) + ...
        PhaseFlux f0 = SolveOnePhase(
            L.rho0, L.M0_normal, L.M0_tangent, L.E0, L.gam0, L.p_inf_0,
            R.rho0, R.M0_normal, R.M0_tangent, R.E0, R.gam0, R.p_inf_0,
            small);

        PhaseFlux f1 = SolveOnePhase(
            L.rho1, L.M1_normal, L.M1_tangent, L.E1, L.gam1, L.p_inf_1,
            R.rho1, R.M1_normal, R.M1_tangent, R.E1, R.gam1, R.p_inf_1,
            small);

        Flux7 out;
        out.mass0 = f0.mass;
        out.momn0 = f0.momn;
        out.momt0 = f0.momt;
        out.E_0   = f0.energy;
        out.mass1 = f1.mass;
        out.momn1 = f1.momn;
        out.momt1 = f1.momt;
        out.E_1   = f1.energy;

        // ---- Interface velocity: mass-weighted average of per-phase
        // contact velocities, using the alpha that is closest to the contact
        // (here we just use the L and R alphas symmetrically; in single-phase
        // regions one term vanishes and the other gives the right answer).
        Set::Scalar alpha0_L = std::max(L.alpha0, 0.0);
        Set::Scalar alpha0_R = std::max(R.alpha0, 0.0);
        Set::Scalar alpha1_L = std::max(1.0 - L.alpha0, 0.0);
        Set::Scalar alpha1_R = std::max(1.0 - R.alpha0, 0.0);

        Set::Scalar ar0 = 0.5 * (alpha0_L * f0.rho_face + alpha0_R * f0.rho_face);
        Set::Scalar ar1 = 0.5 * (alpha1_L * f1.rho_face + alpha1_R * f1.rho_face);
        Set::Scalar denom = ar0 + ar1;
        if (denom <= small)
        {
            // Both phases vanishingly thin: fall back to plain average.
            out.u_I = 0.5 * (f0.u_star + f1.u_star);
        }
        else
        {
            out.u_I = (ar0 * f0.u_star + ar1 * f1.u_star) / denom;
        }

        // ---- Interface pressure: acoustic-impedance weighted.
        Set::Scalar Z0 = std::max(f0.rho_face * f0.a_face, small);
        Set::Scalar Z1 = std::max(f1.rho_face * f1.a_face, small);
        out.p_I = (Z1 * f0.p_star + Z0 * f1.p_star) / (Z0 + Z1);

        return out;
    }

    virtual ~HLLC7() = default;

private:
    /// Single-phase HLLC kernel. Inlined copy of the existing 5-eqn HLLC
    /// algorithm (Toro/Titarev), specialized to "specific volume" mode
    /// (Spec_Vol == 1) which is the default and only used path in this
    /// branch's tests.
    static PhaseFlux SolveOnePhase(
        Set::Scalar rho_L, Set::Scalar Mn_L, Set::Scalar Mt_L, Set::Scalar E_L,
        Set::Scalar gamma_L, Set::Scalar p0_L,
        Set::Scalar rho_R, Set::Scalar Mn_R, Set::Scalar Mt_R, Set::Scalar E_R,
        Set::Scalar gamma_R, Set::Scalar p0_R,
        Set::Scalar small)
    {
        // Floor density. When a phase is essentially absent (alpha -> 0)
        // the integrator should still pass a sensible ambient (rho, p) for
        // its primitives so this routine doesn't see zero/negative inputs.
        rho_L = std::max(small, rho_L);
        rho_R = std::max(small, rho_R);

        // Per-volume kinetic / internal energies.
        Set::Scalar ke_L = 0.5 * (Mn_L * Mn_L + Mt_L * Mt_L) / (rho_L + small);
        Set::Scalar ue_L = std::max(E_L - ke_L, small);
        Set::Scalar p_L  = Solver::EOS::EOS::TammannPressure(rho_L, ue_L, gamma_L, p0_L, small);

        Set::Scalar ke_R = 0.5 * (Mn_R * Mn_R + Mt_R * Mt_R) / (rho_R + small);
        Set::Scalar ue_R = std::max(E_R - ke_R, small);
        Set::Scalar p_R  = Solver::EOS::EOS::TammannPressure(rho_R, ue_R, gamma_R, p0_R, small);

        // Velocities (with limiter, matches existing HLLC).
        Set::Scalar u_L = Mn_L / (rho_L + small);
        Set::Scalar v_L = Mt_L / (rho_L + small);
        Set::Scalar u_R = Mn_R / (rho_R + small);
        Set::Scalar v_R = Mt_R / (rho_R + small);

        const Set::Scalar u_limit = 1e8;
        u_L = (u_L < 0.0) ? std::max(u_L, -u_limit) : std::min(u_L, u_limit);
        v_L = (v_L < 0.0) ? std::max(v_L, -u_limit) : std::min(v_L, u_limit);
        u_R = (u_R < 0.0) ? std::max(u_R, -u_limit) : std::min(u_R, u_limit);
        v_R = (v_R < 0.0) ? std::max(v_R, -u_limit) : std::min(v_R, u_limit);

        Set::Scalar a_L = Solver::EOS::EOS::TammannSoundSpeed(rho_L, p_L, gamma_L, p0_L, small);
        Set::Scalar a_R = Solver::EOS::EOS::TammannSoundSpeed(rho_R, p_R, gamma_R, p0_R, small);

        // Davis wave speed estimates.
        Set::Scalar S_L = std::min(u_L - a_L, u_R - a_R);
        Set::Scalar S_R = std::max(u_L + a_L, u_R + a_R);

        // Contact wave speed.
        Set::Scalar denom_star = (rho_L * (S_L - u_L) - rho_R * (S_R - u_R));
        Set::Scalar S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R))
                             / (denom_star + small);

        // Star states.
        Set::Scalar rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star + small);
        Set::Scalar rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star + small);
        Set::Scalar E_star_L   = rho_star_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L) + small)));
        Set::Scalar E_star_R   = rho_star_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R) + small)));

        // Left and right physical fluxes (per-volume convention).
        Set::Scalar f_L_mass = rho_L * u_L;
        Set::Scalar f_L_momn = rho_L * u_L * u_L + p_L;
        Set::Scalar f_L_momt = rho_L * u_L * v_L;
        Set::Scalar f_L_E    = u_L * (E_L + p_L);

        Set::Scalar f_R_mass = rho_R * u_R;
        Set::Scalar f_R_momn = rho_R * u_R * u_R + p_R;
        Set::Scalar f_R_momt = rho_R * u_R * v_R;
        Set::Scalar f_R_E    = u_R * (E_R + p_R);

        // Star-state fluxes via HLLC jump.
        Set::Scalar fSL_mass = f_L_mass + S_L * (rho_star_L - rho_L);
        Set::Scalar fSL_momn = f_L_momn + S_L * (rho_star_L * S_star - rho_L * u_L);
        Set::Scalar fSL_momt = f_L_momt + S_L * (rho_star_L * v_L     - rho_L * v_L);
        Set::Scalar fSL_E    = f_L_E    + S_L * (E_star_L - E_L);

        Set::Scalar fSR_mass = f_R_mass + S_R * (rho_star_R - rho_R);
        Set::Scalar fSR_momn = f_R_momn + S_R * (rho_star_R * S_star - rho_R * u_R);
        Set::Scalar fSR_momt = f_R_momt + S_R * (rho_star_R * v_R     - rho_R * v_R);
        Set::Scalar fSR_E    = f_R_E    + S_R * (E_star_R - E_R);

        PhaseFlux out;
        if (0.0 <= S_L)
        {
            out.mass = f_L_mass; out.momn = f_L_momn; out.momt = f_L_momt; out.energy = f_L_E;
            out.u_star = u_L; out.p_star = p_L;
            out.rho_face = rho_L; out.a_face = a_L;
        }
        else if (S_L <= 0.0 && 0.0 <= S_star)
        {
            out.mass = fSL_mass; out.momn = fSL_momn; out.momt = fSL_momt; out.energy = fSL_E;
            out.u_star = S_star; out.p_star = p_L;
            out.rho_face = rho_L; out.a_face = a_L;
        }
        else if (S_star <= 0.0 && 0.0 <= S_R)
        {
            out.mass = fSR_mass; out.momn = fSR_momn; out.momt = fSR_momt; out.energy = fSR_E;
            out.u_star = S_star; out.p_star = p_R;
            out.rho_face = rho_R; out.a_face = a_R;
        }
        else
        {
            out.mass = f_R_mass; out.momn = f_R_momn; out.momt = f_R_momt; out.energy = f_R_E;
            out.u_star = u_R; out.p_star = p_R;
            out.rho_face = rho_R; out.a_face = a_R;
        }
        return out;
    }
};

} // namespace FluidRiemann
} // namespace Local
} // namespace Solver

#endif
