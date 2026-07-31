// First-order upwind Riemann solver for the 6-equation diffuse-interface
// model (Saurel-Petitpas-Berry JCP 228, 2009; Schmidmayer-Bryngelson-Colonius
// JCP 402, 2020).
//
// Upwind is the simplest "Riemann" scheme: select F_L or F_R based on the
// sign of the face velocity (no acoustic decomposition, no dissipation
// scaled to a wave speed).  Provided as a debugging fallback; should NOT be
// used for problems with strong shocks (no jump capture).  u_interface is
// the face-upwinded mixture velocity, consistent with Saurel-Abgrall
// (Sau09 line 836).

#ifndef SOLVER_LOCAL_FLUIDRIEMANN_UPWIND_H
#define SOLVER_LOCAL_FLUIDRIEMANN_UPWIND_H

#include "IO/ParmParse.H"
#include "Solver/Local/FluidRiemann/Riemann.H"
#include "Solver/EOS/EOS.H"

namespace Solver
{
namespace Local
{
namespace FluidRiemann
{

class Upwind : public FluidRiemann
{
public:
    static constexpr const char *name = "upwind";

    Upwind(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    Upwind(IO::ParmParse &pp) { pp_queryclass(*this); }
    Upwind()
    {
        IO::ParmParse pp;
        pp_queryclass(*this);
    }

    static void Parse(Upwind & /*value*/, IO::ParmParse & /*pp*/) {}

    virtual Flux Solve(State lo, State hi, Set::Scalar /*p_ref*/, Set::Scalar small) override
    {
        const Set::Scalar alpha_floor = 1.0e-12;

        auto recover = [&](const State &s,
                           Set::Scalar &alpha, Set::Scalar &alpha2,
                           Set::Scalar &rho,   Set::Scalar &u,   Set::Scalar &v,
                           Set::Scalar &p)
        {
            alpha  = std::min(std::max(s.alpha, alpha_floor), 1.0 - alpha_floor);
            alpha2 = 1.0 - alpha;
            const Set::Scalar arh0 = std::max(s.alpha_rho_0, small);
            const Set::Scalar arh1 = std::max(s.alpha_rho_1, small);
            rho    = std::max(arh0 + arh1, small);
            u      = s.M_normal  / rho;
            v      = s.M_tangent / rho;

            const Set::Scalar p0 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E0, alpha,  s.gamma0, s.pi0, small);
            const Set::Scalar p1 = Solver::EOS::EOS::PhasicPressureFromEnergy(s.E1, alpha2, s.gamma1, s.pi1, small);
            p = alpha * p0 + alpha2 * p1;
        };

        Set::Scalar aL, a2L, rhoL, uL, vL, pL;
        Set::Scalar aR, a2R, rhoR, uR, vR, pR;
        recover(lo, aL, a2L, rhoL, uL, vL, pL);
        recover(hi, aR, a2R, rhoR, uR, vR, pR);

        // Roe-like density-weighted average velocity at the face.
        const Set::Scalar sqrtL = std::sqrt(std::max(rhoL, small));
        const Set::Scalar sqrtR = std::sqrt(std::max(rhoR, small));
        const Set::Scalar u_face = (sqrtL * uL + sqrtR * uR) / std::max(sqrtL + sqrtR, small);

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

        Flux fl;
        if (u_face >= 0.0)
        {
            fl = build_flux(uL, vL, pL, lo.alpha_rho_0, lo.alpha_rho_1, lo.E0, lo.E1, lo.E_total, rhoL);
            fl.u_interface = uL;
        }
        else
        {
            fl = build_flux(uR, vR, pR, hi.alpha_rho_0, hi.alpha_rho_1, hi.E0, hi.E1, hi.E_total, rhoR);
            fl.u_interface = uR;
        }
        // Second tangential momentum (3D): pure upwind advection, mirroring
        // momentum_tangent = (mass flux) * v_t(upwind).  Zero in 2D.
        fl.momentum_tangent2 = (fl.mass0 + fl.mass1)
                             * ((u_face >= 0.0) ? lo.M_tangent2 / rhoL : hi.M_tangent2 / rhoR);
        return fl;
    }
};

}
}
}

#endif
