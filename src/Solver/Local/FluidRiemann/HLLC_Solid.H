// REDUCED HLLC Riemann solver for a multiphase mixture with elasto-viscoplastic solids
// (Hydro2 elastic.on, Riemann_Solver.type = hllc_solid).
//
// Reference: N. Favrie, K. Schmidmayer, "A multiphase compressible flow model including
// elasto-viscoplastic solids with irreversible compaction", CMAME 455 (2026) 118880,
// Section 4.1.1 "HLLC-type Riemann solver", eqs. (75)-(93).
//
// THIS IS ITS OWN SOLVER.  It is selected by name only; the fluid solvers (hllc, hlle, ...) and
// the Favrie-Gavrilyuk-Saurel (2009) solver hllc_elastic are untouched and elastic.on keeps
// defaulting to hllc_elastic.
//
// Three waves (S_L, S_M, S_R) for the 7-wave hyperelastic system: the two longitudinal waves and
// the contact; shear waves are not resolved separately.  With sigma = -p I + S the mixture stress
// (State::S_nn, S_nt, S_nt2), m_K = rho_K (u_K - S_K)::
//
//   (75) S_R = max(u_L + c_L, u_R + c_R),  S_L = min(u_L - c_L, u_R - c_R)                 (Davis)
//   (76) c^2 = sum_k Y_k c_k^2,   c_k^2 = dp_k/drho_k + zeta (4/3) mu_k / rho_k            (simple form below
//        eq. 77; zeta >= 1 is the paper's safety coefficient, input `zeta`, default 1)
//   (78) S_M = u* = [ (rho u^2 - sigma_11)_L - (rho u^2 - sigma_11)_R - S_L (rho u)_L + S_R (rho u)_R ]
//                   / [ (rho u)_L - (rho u)_R - S_L rho_L + S_R rho_R ]
//   (79) (alpha_k rho_k)*_K = (alpha_k rho_k)_K (S_K - u_K)/(S_K - u*)
//   (81) sigma*_11 = [ m_R sigma_11L - m_L sigma_11R + m_L m_R (u_R - u_L) ] / (m_R - m_L)
//   (82) sigma*_12 = [ m_R sigma_12L - m_L sigma_12R + m_L m_R (v_R - v_L) ] / (m_R - m_L)      (83: same for 13)
//   (84) v* = [ (rho u v - sigma_12)_L - (rho u v - sigma_12)_R - S_L (rho v)_L + S_R (rho v)_R ] / [same as 78]
//   (86) E*_K = [ rho_K E_K (u_K - S_K) - sigma_11K u_K - sigma_12K v_K - sigma_13K w_K
//                 + sigma*_11 u* + sigma*_12 v* + sigma*_13 w* ] / [ rho*_K (u* - S_K) ]
//   (90) alpha*_k = alpha_k,   (91) rho*_k = rho_k (u_K - S_K)/(u* - S_K)
//   (93) p*_k = (p_k + pi_k)(rho*_k/rho_k)^gamma_k - pi_k,   (92) e*_k = e_k(p*_k, rho*_k)   (hydrodynamic part)
//   (88)-(89) cobasis: the integrator builds the star cobasis from u_interface = u* and vt = v*, wt = w*.
//
// WHAT IS DIFFERENT FROM hllc_elastic (Favrie et al. 2009, HLLC_Elastic.H):
//
//   * the shear stress sigma*_12 and the tangential velocity v* are ONE value across the contact at EVERY face
//     (eqs. 82, 84).  hllc_elastic does that only at solid|solid faces and lets a fluid slide on a solid
//     (sigma*_12 = 0).  Here a face with a fluid side carries the term m_L m_R (v_R - v_L)/(m_R - m_L): for
//     pure fluids the tangential momentum then has the two-wave (HLL) flux, i.e. shear layers and slip lines in
//     the FLUID are diffused.  That is the paper's solver; it is the reason this class is kept apart.
//   * star phase pressures follow the isentrope (93), not the phasic Hugoniot.
//   * no objective-shear correction (State::dvt is ignored).
//
// The correction of the phase energies to the total energy (their eqs. 113-115) is not part of the Riemann
// solver; Hydro2 does the equivalent in its re-initialisation from rho E.
//
// With S = 0, c2_el = 0 and v_L = v_R this reduces to the fluid solver of HLLC.H except for (93).

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_HLLC_SOLID_H
#define SOLVER_LOCAL_FLUIDRIEMANN_HLLC_SOLID_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

/// Reduced 3-wave HLLC for fluids and elasto-plastic solids (Favrie & Schmidmayer 2026).
class HLLC_Solid : public FluidRiemann
{
public:
    static constexpr const char *name = "hllc_solid";

    HLLC_Solid(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    HLLC_Solid(IO::ParmParse &pp) { pp_queryclass(*this); }
    HLLC_Solid()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(HLLC_Solid &value, IO::ParmParse &pp)
    {
        // Safety coefficient on the elastic part of the longitudinal sound speed (>= 1)
        pp_query_default("zeta", value.zeta, 1.0);
    }

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        // Primitive recovery of one side (same conventions as HLLC.H: phase masses clamped at zero only).
        struct Side
        {
            Set::Scalar a0, a1, m0, m1, rho, u, v, w, r0, r1, p0, p1, p, c, s11, s12, s13, E;
        };
        auto recover = [&](const State &s) -> Side
        {
            Side q;
            q.a0 = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            q.a1 = 1.0 - q.a0;
            q.m0 = std::max(s.alpha_rho_0, 0.0);
            q.m1 = std::max(s.alpha_rho_1, 0.0);
            q.rho = std::max(q.m0 + q.m1, small);
            q.u = s.M_normal / q.rho;
            q.v = s.M_tangent / q.rho;
            q.w = s.M_tangent2 / q.rho;
            q.r0 = std::max(q.m0 / q.a0, small);
            q.r1 = std::max(q.m1 / q.a1, small);
            q.p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, q.a0, s.gamma0, s.pi0, alpha_floor);
            q.p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, q.a1, s.gamma1, s.pi1, alpha_floor);
            const Set::Scalar c0 = Solver::EOS::EOS::PhasicSoundSpeed(q.r0, q.p0, s.gamma0, s.pi0, small);
            const Set::Scalar c1 = Solver::EOS::EOS::PhasicSoundSpeed(q.r1, q.p1, s.gamma1, s.pi1, small);
            q.p = q.a0 * q.p0 + q.a1 * q.p1;
            // (76): frozen mixture speed plus the elastic part, State::c2_el = Y_s (4/3) mu_s / rho_s
            const Set::Scalar cf = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(q.m0 / q.rho, q.m1 / q.rho, c0, c1), small);
            q.c = std::sqrt(cf * cf + zeta * std::max(s.c2_el, 0.0));
            q.s11 = -q.p + s.S_nn;
            q.s12 = s.S_nt;
            q.s13 = s.S_nt2;
            q.E = s.E_total;
            return q;
        };
        const Side L = recover(lo), R = recover(hi);

        // (75) wave speeds
        const Set::Scalar S_L = std::min(L.u - L.c, R.u - R.c);
        const Set::Scalar S_R = std::max(L.u + L.c, R.u + R.c);

        // (78), (84), (85): one contact speed and one tangential velocity
        Set::Scalar den = L.rho * L.u - R.rho * R.u - S_L * L.rho + S_R * R.rho;
        den += (den >= 0.0 ? small : -small);
        const Set::Scalar S_M = ((L.rho * L.u * L.u - L.s11) - (R.rho * R.u * R.u - R.s11) - S_L * L.rho * L.u + S_R * R.rho * R.u) / den;
        const Set::Scalar v_s = ((L.rho * L.u * L.v - L.s12) - (R.rho * R.u * R.v - R.s12) - S_L * L.rho * L.v + S_R * R.rho * R.v) / den;
        const Set::Scalar w_s = ((L.rho * L.u * L.w - L.s13) - (R.rho * R.u * R.w - R.s13) - S_L * L.rho * L.w + S_R * R.rho * R.w) / den;

        // (81)-(83): star stresses.  m_L > 0 > m_R
        Set::Scalar mL = L.rho * (L.u - S_L);  if (std::abs(mL) < small) mL = small;
        Set::Scalar mR = R.rho * (R.u - S_R);  if (std::abs(mR) < small) mR = -small;
        Set::Scalar dm = mR - mL;              if (std::abs(dm) < small) dm = -small;
        const Set::Scalar s11_s = (mR * L.s11 - mL * R.s11 + mL * mR * (R.u - L.u)) / dm;
        const Set::Scalar s12_s = (mR * L.s12 - mL * R.s12 + mL * mR * (R.v - L.v)) / dm;
        const Set::Scalar s13_s = (mR * L.s13 - mL * R.s13 + mL * mR * (R.w - L.w)) / dm;

        // Star state of one side: (79), (86), (87), (90)-(93)
        struct Star { Set::Scalar m0, m1, rho, E, E0, E1; };
        auto star = [&](const Side &q, const State &s, Set::Scalar S) -> Star
        {
            Star t;
            Set::Scalar d = S - S_M;  if (std::abs(d) < small) d = small;
            const Set::Scalar fac = (S - q.u) / d;
            t.m0 = q.m0 * fac;  t.m1 = q.m1 * fac;  t.rho = t.m0 + t.m1;
            // (86) written for (rho E)*: [ (rho E)(u - S) - sigma.v + sigma*.v* ] / (u* - S)
            t.E = (q.E * (q.u - S) - q.s11 * q.u - q.s12 * q.v - q.s13 * q.w + s11_s * S_M + s12_s * v_s + s13_s * w_s) / (-d);
            auto phase = [&](Set::Scalar mk_s, Set::Scalar ak, Set::Scalar rk, Set::Scalar pk, Set::Scalar g, Set::Scalar pi) -> Set::Scalar
            {
                const Set::Scalar rs = std::max(mk_s / ak, small);
                const Set::Scalar ps = std::max(pk + pi, small) * std::pow(rs / rk, g) - pi;      // (93)
                return ak * std::max(ps + g * pi, small) / (g - 1.0);                             // (92): alpha_k rho_k e_k
            };
            t.E0 = phase(t.m0, q.a0, q.r0, q.p0, s.gamma0, s.pi0);
            t.E1 = phase(t.m1, q.a1, q.r1, q.p1, s.gamma1, s.pi1);
            return t;
        };

        // Physical flux of one side (their eq. 96) and F*_K = F_K + S_K (U*_K - U_K)
        auto phys = [&](const Side &q, const State &s) -> Flux
        {
            Flux f;
            f.mass0             = q.m0 * q.u;
            f.mass1             = q.m1 * q.u;
            f.momentum_normal   = q.rho * q.u * q.u - q.s11;
            f.momentum_tangent  = q.rho * q.u * q.v - q.s12;
            f.momentum_tangent2 = q.rho * q.u * q.w - q.s13;
            f.energy_total      = q.u * (q.E - q.s11) - q.s12 * q.v - q.s13 * q.w;
            f.energy0           = s.E0 * q.u;
            f.energy1           = s.E1 * q.u;
            return f;
        };
        auto starflux = [&](const Side &q, const State &s, Set::Scalar S) -> Flux
        {
            const Star t = star(q, s, S);
            Flux f = phys(q, s);
            f.mass0             += S * (t.m0 - q.m0);
            f.mass1             += S * (t.m1 - q.m1);
            f.momentum_normal   += S * (t.rho * S_M - s.M_normal);
            f.momentum_tangent  += S * (t.rho * v_s - s.M_tangent);
            f.momentum_tangent2 += S * (t.rho * w_s - s.M_tangent2);
            f.energy_total      += S * (t.E - q.E);
            f.energy0           += S * (t.E0 - s.E0);
            f.energy1           += S * (t.E1 - s.E1);
            return f;
        };

        // Sampling at x/t = 0.  u_interface, vt, wt feed the non-conservative rows (alpha, phase energies,
        // cobasis, their eqs. 88-89 and 103) in the integrator.
        Flux fl;
        if (0.0 <= S_L)
        {
            fl = phys(L, lo);
            fl.u_interface = L.u;  fl.vt_L = fl.vt_R = L.v;  fl.wt_L = fl.wt_R = L.w;
        }
        else if (0.0 <= S_M)
        {
            fl = starflux(L, lo, S_L);
            fl.u_interface = S_M;  fl.vt_L = fl.vt_R = v_s;  fl.wt_L = fl.wt_R = w_s;
        }
        else if (0.0 <= S_R)
        {
            fl = starflux(R, hi, S_R);
            fl.u_interface = S_M;  fl.vt_L = fl.vt_R = v_s;  fl.wt_L = fl.wt_R = w_s;
        }
        else
        {
            fl = phys(R, hi);
            fl.u_interface = R.u;  fl.vt_L = fl.vt_R = R.v;  fl.wt_L = fl.wt_R = R.w;
        }

        if ((fl.mass0 != fl.mass0) or (fl.mass1 != fl.mass1) or (fl.momentum_normal != fl.momentum_normal)
            or (fl.momentum_tangent != fl.momentum_tangent) or (fl.momentum_tangent2 != fl.momentum_tangent2)
            or (fl.energy_total != fl.energy_total) or (fl.energy0 != fl.energy0) or (fl.energy1 != fl.energy1))
        {
            Util::ParallelMessage(INFO, "-------------------------------");
            Util::ParallelMessage(INFO, "ERROR IN hllc_solid (Favrie & Schmidmayer 2026):");
            Util::ParallelMessage(INFO, "lo:   ", lo);
            Util::ParallelMessage(INFO, "hi:   ", hi);
            Util::ParallelMessage(INFO, "flux: ", fl);
            Util::ParallelMessage(INFO, "S_L=", S_L, " S_M=", S_M, " S_R=", S_R, " cL=", L.c, " cR=", R.c);
            Util::Exception(INFO);
        }
        return fl;
    }

private:
    Set::Scalar zeta = 1.0;
};

}
}
}

#endif
