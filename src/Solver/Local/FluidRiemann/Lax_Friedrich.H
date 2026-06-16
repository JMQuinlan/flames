// Rusanov / local Lax-Friedrichs flux for the 6-equation diffuse-interface
// model (Saurel-Petitpas-Berry JCP 228, 2009; Schmidmayer-Bryngelson-Colonius
// JCP 402, 2020).
//
//     F = 0.5 (F_L + F_R) - 0.5 S_max (U_R - U_L)
//     S_max = max(|u_L| + c_L, |u_R| + c_R)
//
// with frozen mixture sound speed c (Sch20 eq. 17 / Sau09 eq. III.2).  This
// is the most dissipative of the upwind family; useful as a near-vacuum or
// debugging fallback.  Exposes the HLL contact estimate as u_interface so
// the integrator's Saurel-Abgrall non-conservative rows (Sau09 line 836)
// stay self-consistent.

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H
#define SOLVER_LOCAL_FLUIDRIEMANN_LAX_FRIEDRICH_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class Lax_Friedrich : public FluidRiemann
{
public:
    static constexpr const char *name = "lax_friedrich";

    Lax_Friedrich(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Lax_Friedrich(IO::ParmParse &pp) { pp_queryclass(*this); }
    Lax_Friedrich()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Lax_Friedrich & /*value*/, IO::ParmParse & /*pp*/) {}

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small, int /*Spec_Vol*/ = 1) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        auto recover = [&](const State &s,
                           Set::Scalar &alpha, Set::Scalar &alpha2,
                           Set::Scalar &rho,   Set::Scalar &u,   Set::Scalar &v,
                           Set::Scalar &p,     Set::Scalar &c)
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
            const Set::Scalar Y0 = arh0 / rho;
            const Set::Scalar Y1 = arh1 / rho;
            p = alpha * p0 + alpha2 * p1;
            c = std::max(Solver::EOS::EOS::FrozenMixtureSoundSpeed(Y0, Y1, c0, c1), small);
        };

        Set::Scalar aL, a2L, rhoL, uL, vL, pL, cL;
        Set::Scalar aR, a2R, rhoR, uR, vR, pR, cR;
        recover(lo, aL, a2L, rhoL, uL, vL, pL, cL);
        recover(hi, aR, a2R, rhoR, uR, vR, pR, cR);

        // Rusanov speed: largest |u| + c across L and R.
        const Set::Scalar S_max = std::max(std::abs(uL) + cL, std::abs(uR) + cR);

        // Build L and R fluxes.
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

        Flux fl;
        fl.mass0            = 0.5 * (F_L.mass0            + F_R.mass0            - S_max * (hi.alpha_rho_0 - lo.alpha_rho_0));
        fl.mass1            = 0.5 * (F_L.mass1            + F_R.mass1            - S_max * (hi.alpha_rho_1 - lo.alpha_rho_1));
        fl.momentum_normal  = 0.5 * (F_L.momentum_normal  + F_R.momentum_normal  - S_max * (hi.M_normal    - lo.M_normal));
        fl.momentum_tangent = 0.5 * (F_L.momentum_tangent + F_R.momentum_tangent - S_max * (hi.M_tangent   - lo.M_tangent));
        fl.energy_total     = 0.5 * (F_L.energy_total     + F_R.energy_total     - S_max * (hi.E_total     - lo.E_total));
        fl.energy0          = 0.5 * (F_L.energy0          + F_R.energy0          - S_max * (hi.E0          - lo.E0));
        fl.energy1          = 0.5 * (F_L.energy1          + F_R.energy1          - S_max * (hi.E1          - lo.E1));

        // u_interface for Saurel-Abgrall: 0.5(u_L + u_R) is the natural choice
        // for a Rusanov scheme.  In supersonic limits, override with u_L / u_R.
        if      (uL - cL >= 0.0) fl.u_interface = uL;
        else if (uR + cR <= 0.0) fl.u_interface = uR;
        else                     fl.u_interface = 0.5 * (uL + uR);

        return fl;
    }
};

}
}
}

#endif
