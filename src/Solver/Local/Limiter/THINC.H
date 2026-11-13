// THINC: interface-sharpening wrapper around ANY limiter for the volume
// fraction.  Every primitive is reconstructed by the base limiter; in cells
// that hold an interface, the face value of alpha is then replaced by the
// THINC (tanh) reconstruction of Shyue & Xiao, JCP 268 (2014) 326, eqs. (10)-(13),
// applied direction by direction as in Schmidmayer, Bryngelson & Colonius,
// JCP 402 (2020) 109080, Sec. 5.4.  The partial densities follow from
// alpha_face * rho_pure_face in ToState, so mass and volume stay consistent.
//
// For cell i with neighbours a_{i-1}, a_i, a_{i+1}, THINC acts only where
//   eps < a_i < 1 - eps   and   (a_{i+1} - a_i)(a_i - a_{i-1}) > 0
// (an interface cell that is monotone across the stencil); otherwise the base
// value is kept.  With a_min = min(a_{i-1}, a_{i+1}), a_max = max(...),
//   theta = sign(a_{i+1} - a_{i-1}),
//   C = (a_i - a_min + eps) / (a_max - a_min + eps),
//   B = exp(theta beta (2C - 1)),   A = (B / cosh(beta) - 1) / tanh(beta),
//   a_{i+1/2}^L = a_min + (a_max - a_min)/2 (1 + theta (tanh(beta) + A)/(1 + A tanh(beta))).
// The left-edge value comes from the same formula on the reversed stencil
// (the framework's convention), which is THINC's a_{i-1/2}^R.
//
// Input parameters (prefix Limiter.):
//   thinc_base (string) godunov | minmod | vanleer | muscl | muscl2 | weno3 | weno5
//                       (default muscl2); its own parameters are read as usual
//   thinc_beta (Real)   steepness, default 1.6 (Shyue & Xiao; ~1.6 cells across 0.01..0.99)
//   thinc_eps  (Real)   interface-cell threshold, default 1e-4
//   thinc_sw   (int)    1 = slope weighting, beta |n_d| + 0.01 per direction (default 0)
// Both Limiter.thinc_beta and Limiter.thinc.thinc_beta are read (likewise the other keys).
//
// PASSIVE scalars (Gamma and the like) that the code routes through the alpha
// slot must not be sharpened: call ReconstructPassive, which skips THINC.

#ifndef SOLVER_LOCAL_LIMITER_THINC_H
#define SOLVER_LOCAL_LIMITER_THINC_H

#include <cmath>
#include <memory>
#include <string>

#include "IO/ParmParse.H"
#include "Util/Util.H"
#include "Solver/Local/Limiter/Limiter.H"
#include "Solver/Local/Limiter/Godunov.H"
#include "Solver/Local/Limiter/Minmod.H"
#include "Solver/Local/Limiter/VanLeer.H"
#include "Solver/Local/Limiter/MUSCL.H"
#include "Solver/Local/Limiter/MUSCL2.H"
#include "Solver/Local/Limiter/WENO3.H"
#include "Solver/Local/Limiter/WENO5.H"

namespace Solver { namespace Local { namespace Limiter {

class THINC : public Limiter
{
public:
    static constexpr const char *name = "thinc";

    THINC(IO::ParmParse &pp, std::string nm) { pp_queryclass(nm, *this); }
    THINC(IO::ParmParse &pp)                 { pp_queryclass(*this); }
    THINC() { IO::ParmParse pp; pp_queryclass(*this); }

    std::shared_ptr<Limiter> base;
    std::string base_name = "muscl2";
    Set::Scalar beta = 1.6;
    Set::Scalar eps  = 1.0e-4;
    // 1 = slope weighting (THINC/SW of Xiao, Ii & Chen, JCP 230 (2011) 7086): the steepness used in direction d is
    // beta |n_d| + 0.01 with n the unit normal of the interface, so the tanh profile is only as steep along a
    // coordinate direction as the interface is aligned with it.  The directionally-split THINC with one beta
    // sharpens every direction alike and pulls a curved interface towards the grid axes (Schmidmayer et al. 2020,
    // Sec. 5.4, name the directional splitting as the likely cause of their loss of sphericity).
    int sw = 0;

    static void Parse(THINC &value, IO::ParmParse &pp)
    {
        pp.query_validate("thinc_base", value.base_name,
                          {"muscl2", "muscl", "minmod", "vanleer", "weno3", "weno5", "godunov"});
        pp.query_default("thinc_beta", value.beta, 1.6);
        pp.query_default("thinc_eps", value.eps, 1.0e-4);
        pp.query_default("thinc_sw", value.sw, 0);
        {
            // [2026-10-08] The keys are also accepted as Limiter.thinc_beta etc., which is how the decks and the
            // header above write them.  Until now only the nested form Limiter.thinc.thinc_beta was read and the
            // short form was silently ignored (listed under "Unused ParmParse Variables"; every run got beta = 1.6).
            amrex::ParmParse pl("Limiter");
            pl.query("thinc_beta", value.beta); pl.query("thinc_eps", value.eps); pl.query("thinc_sw", value.sw);
            std::string b2; if (pl.query("thinc_base", b2)) value.base_name = b2;
        }
        const std::string &b = value.base_name;
        if      (b == "muscl2")  value.base = std::make_shared<MUSCL2>(pp);
        else if (b == "muscl")   value.base = std::make_shared<MUSCL>(pp);
        else if (b == "minmod")  value.base = std::make_shared<Minmod>(pp);
        else if (b == "vanleer") value.base = std::make_shared<VanLeer>(pp);
        else if (b == "weno3")   value.base = std::make_shared<WENO3>(pp);
        else if (b == "weno5")   value.base = std::make_shared<WENO5>(pp);
        else                     value.base = std::make_shared<Godunov>(pp);
        Util::Message(INFO, "THINC wrapper: base limiter = ", b, ", beta = ", value.beta, ", eps = ", value.eps, ", slope weighting = ", value.sw);
    }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        Primitive f = base->Reconstruct(stencil);
        const Set::Scalar w = stencil[RADIUS].thinc_w;
        f.alpha = Sharpen(stencil[RADIUS - 1].alpha, stencil[RADIUS].alpha, stencil[RADIUS + 1].alpha, f.alpha,
                          (sw && w >= 0.0) ? beta * w + 0.01 : beta);
        return f;
    }

    Primitive ReconstructPassive(const Primitive stencil[STENCIL_SIZE]) const override
    {
        return base->ReconstructPassive(stencil);
    }

private:
    Set::Scalar Sharpen(Set::Scalar am, Set::Scalar a, Set::Scalar ap, Set::Scalar fallback, Set::Scalar beta) const
    {
        if (!(a > eps && a < 1.0 - eps)) return fallback;
        if (!((ap - a) * (a - am) > 0.0)) return fallback;
        const Set::Scalar amin = std::min(am, ap), amax = std::max(am, ap);
        const Set::Scalar theta = (ap > am) ? 1.0 : -1.0;
        const Set::Scalar C = (a - amin + eps) / (amax - amin + eps);
        const Set::Scalar B = std::exp(theta * beta * (2.0 * C - 1.0));
        const Set::Scalar tb = std::tanh(beta);
        const Set::Scalar A = (B / std::cosh(beta) - 1.0) / tb;
        return amin + 0.5 * (amax - amin) * (1.0 + theta * (tb + A) / (1.0 + A * tb));
    }
};

}}}

#endif
