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
//
//   thinc_mode (string) how the interface is reconstructed (default cartesian):
//     cartesian : the scheme above, direction by direction, fixed beta (cell units).
//     mthinc    : multi-dimensional THINC with a PLANAR surface (Ii et al., JCP 231
//                 (2012) 2328): H(x) = (1 + tanh((n.x + d)/eps))/2 in the cell, d from
//                 the cell average = alpha (Gauss quadrature + safeguarded Newton), face
//                 value = face average of H.
//     qq        : THINC/QQ (Xie & Xiao, JCP 349 (2017) 415): as mthinc with the
//                 QUADRATIC surface n.x + x^T K x / 2, K = grad n (curvature).
//   thinc_eps_tie (int) 1 (default): mthinc/qq take their thickness from the model
//                 eps (Hydro2 `epsilon`); 0: from thinc_beta (eps_T = h_min/beta).
//   thinc_fade (int) 0 (default) / 1: switch sharpening off where the tanh-of-a-
//                 surface picture fails, k = eps_T |K|_F (|K|_F = sqrt(k1^2 + k2^2)):
//                 full THINC for k <= thinc_fade_on (0.5), base limiter for
//                 k >= thinc_fade_off (1.0), linear blend of face values between.
//   thinc_quad (int) Gauss points per direction for mthinc/qq (2 or 3, default 3).
//   thinc_guard (int) 1 (default) / 0: monotonicity guard for mthinc/qq.  The single-
//                 surface fit is used only where alpha is monotone across the cell
//                 and its two neighbours (as in the cartesian scheme), and the face
//                 value is bounded by those neighbours.  At a local extremum -- a
//                 feature thinner than the band (thin jet, neck, sheet) -- the base
//                 limiter's face value is used instead.
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
    enum Mode { CART = 0, MTH = 2, QQ = 3 };
    std::string mode_name = "cartesian";
    int mode = CART;
    int eps_tie = 1;
    int fade = 0;
    Set::Scalar fade_on = 0.5, fade_off = 1.0;
    int quad = 3;
    int guard = 1;

    static void Parse(THINC &value, IO::ParmParse &pp)
    {
        pp.query_validate("thinc_base", value.base_name,
                          {"muscl2", "muscl", "minmod", "vanleer", "weno3", "weno5", "godunov"});
        pp.query_default("thinc_beta", value.beta, 1.6);
        pp.query_default("thinc_eps", value.eps, 1.0e-4);
        pp.query_validate("thinc_mode", value.mode_name, {"cartesian", "mthinc", "qq"});
        value.mode = value.mode_name == "mthinc" ? MTH : value.mode_name == "qq" ? QQ : CART;
        pp.query_validate("thinc_eps_tie", value.eps_tie, {1, 0});   // first entry is the default
        pp.query_validate("thinc_fade", value.fade, {0, 1});
        pp.query_default("thinc_fade_on", value.fade_on, 0.5);
        pp.query_default("thinc_fade_off", value.fade_off, 1.0);
        pp.query_validate("thinc_quad", value.quad, {3, 2});         // first entry is the default
        pp.query_validate("thinc_guard", value.guard, {1, 0});       // first entry is the default
        if (value.fade && !(value.fade_off > value.fade_on))
            Util::Abort(INFO, "Limiter thinc_fade_off must exceed thinc_fade_on");
        const std::string &b = value.base_name;
        if      (b == "muscl2")  value.base = std::make_shared<MUSCL2>(pp);
        else if (b == "muscl")   value.base = std::make_shared<MUSCL>(pp);
        else if (b == "minmod")  value.base = std::make_shared<Minmod>(pp);
        else if (b == "vanleer") value.base = std::make_shared<VanLeer>(pp);
        else if (b == "weno3")   value.base = std::make_shared<WENO3>(pp);
        else if (b == "weno5")   value.base = std::make_shared<WENO5>(pp);
        else                     value.base = std::make_shared<Godunov>(pp);
        Util::Message(INFO, "THINC wrapper: base limiter = ", b, ", beta = ", value.beta, ", eps = ", value.eps,
                      ", mode = ", value.mode_name, ", eps_tie = ", value.eps_tie, ", fade = ", value.fade,
                      " (", value.fade_on, ", ", value.fade_off, "), quad = ", value.quad, ", guard = ", value.guard);
    }

    bool WantsGeom() const override { return mode != CART || fade; }

    // Geometry-aware reconstruction (all modes; cartesian + fade lands here too).
    Primitive ReconstructGeom(const Primitive stencil[STENCIL_SIZE], const Geom &g) const override
    {
        Primitive f = base->Reconstruct(stencil);
        const Set::Scalar am = stencil[RADIUS - 1].alpha, a = stencil[RADIUS].alpha, ap = stencil[RADIUS + 1].alpha;
        if (!(a > eps && a < 1.0 - eps)) return f;
        if (!g.valid) {
            // no usable normal (|grad eta| ~ 0): only the legacy scheme can act
            if (mode == CART) f.alpha = Sharpen(am, a, ap, f.alpha, beta);
            return f;
        }
        const Set::Scalar hmin = g.h.minCoeff();
        const Set::Scalar epsT = (eps_tie && g.eps > 0.0) ? g.eps : hmin / beta;
        Set::Scalar w = 1.0;
        if (fade)
        {
            const Set::Scalar k = epsT * g.K.norm();
            w = std::min(1.0, std::max(0.0, (fade_off - k) / (fade_off - fade_on)));
            if (w <= 0.0) return f;
        }
        Set::Scalar sh = f.alpha;
        if (mode == CART) sh = Sharpen(am, a, ap, f.alpha, beta);
        else
        {
            // guard: no single-surface fit at a local extremum of alpha
            if (guard && !((ap - a) * (a - am) > 0.0)) return f;
            sh = FaceMD(a, g, epsT, mode == QQ);
            if (guard) sh = std::min(std::max(am, ap), std::max(std::min(am, ap), sh));
        }
        f.alpha = w * sh + (1.0 - w) * f.alpha;
        return f;
    }

    Primitive Reconstruct(const Primitive stencil[STENCIL_SIZE]) const override
    {
        Primitive f = base->Reconstruct(stencil);
        f.alpha = Sharpen(stencil[RADIUS - 1].alpha, stencil[RADIUS].alpha, stencil[RADIUS + 1].alpha, f.alpha, beta);
        return f;
    }

    Primitive ReconstructPassive(const Primitive stencil[STENCIL_SIZE]) const override
    {
        return base->ReconstructPassive(stencil);
    }

private:
    // Multi-dimensional THINC face value (mthinc / qq).  In the cell (offset x from the
    // center, face frame) H(x) = (1 + tanh((phi(x) + d)/epsT))/2 with
    // phi = n.x (+ x^T K x / 2 for qq); d solves <H>_cell = a by Gauss quadrature
    // (H is monotone in d: safeguarded Newton); returns <H> over the face x_0 = +h_0/2.
    Set::Scalar FaceMD(Set::Scalar a, const Geom &g, Set::Scalar epsT, bool quadratic) const
    {
        constexpr int D = AMREX_SPACEDIM;
        Set::Scalar xi[3], wq[3]; int nq = quad;
        if (nq == 2) { xi[0] = -0.5 / std::sqrt(3.0); xi[1] = -xi[0]; wq[0] = wq[1] = 0.5; }
        else { xi[0] = -0.5 * std::sqrt(0.6); xi[1] = 0.0; xi[2] = -xi[0];
               wq[0] = wq[2] = 5.0 / 18.0; wq[1] = 8.0 / 18.0; }
        auto phi = [&](const Set::Vector &x) {
            Set::Scalar p = g.n.dot(x);
            if (quadratic) p += 0.5 * x.dot(g.K * x);
            return p;
        };
        // cell quadrature points
        constexpr int NQMAX = (D == 3) ? 27 : 9;
        Set::Scalar ph[NQMAX], W[NQMAX]; int np = 0; Set::Scalar pmax = 0.0;
        for (int q0 = 0; q0 < nq; ++q0) for (int q1 = 0; q1 < nq; ++q1)
#if AMREX_SPACEDIM == 3
        for (int q2 = 0; q2 < nq; ++q2)
#endif
        {
            Set::Vector x;
            x(0) = xi[q0] * g.h(0); x(1) = xi[q1] * g.h(1);
            Set::Scalar ww = wq[q0] * wq[q1];
#if AMREX_SPACEDIM == 3
            x(2) = xi[q2] * g.h(2); ww *= wq[q2];
#endif
            ph[np] = phi(x); W[np] = ww; pmax = std::max(pmax, std::abs(ph[np])); ++np;
        }
        auto Hm = [&](Set::Scalar d, Set::Scalar &dH) {
            Set::Scalar m = 0.0; dH = 0.0;
            for (int q = 0; q < np; ++q) {
                const Set::Scalar t = std::tanh((ph[q] + d) / epsT);
                m += W[q] * 0.5 * (1.0 + t); dH += W[q] * 0.5 * (1.0 - t * t) / epsT;
            }
            return m;
        };
        Set::Scalar lo = -pmax - 20.0 * epsT, hi = pmax + 20.0 * epsT;
        Set::Scalar d = epsT * std::atanh(std::min(1.0 - 1.0e-12, std::max(-1.0 + 1.0e-12, 2.0 * a - 1.0)));
        d = std::min(hi, std::max(lo, d));
        for (int it = 0; it < 60; ++it)
        {
            Set::Scalar dH; const Set::Scalar r = Hm(d, dH) - a;
            if (std::abs(r) < 1.0e-13) break;
            if (r > 0.0) hi = d; else lo = d;
            Set::Scalar dn = (dH > 1.0e-300) ? d - r / dH : 0.5 * (lo + hi);
            if (!(dn > lo && dn < hi)) dn = 0.5 * (lo + hi);
            d = dn;
            if (hi - lo < 1.0e-14 * (std::abs(d) + epsT)) break;
        }
        // face average at x_0 = +h_0/2
        Set::Scalar fsum = 0.0;
        for (int q1 = 0; q1 < nq; ++q1)
#if AMREX_SPACEDIM == 3
        for (int q2 = 0; q2 < nq; ++q2)
#endif
        {
            Set::Vector x; x(0) = 0.5 * g.h(0); x(1) = xi[q1] * g.h(1);
            Set::Scalar ww = wq[q1];
#if AMREX_SPACEDIM == 3
            x(2) = xi[q2] * g.h(2); ww *= wq[q2];
#endif
            fsum += ww * 0.5 * (1.0 + std::tanh((phi(x) + d) / epsT));
        }
        return std::min(1.0, std::max(0.0, fsum));
    }

    Set::Scalar Sharpen(Set::Scalar am, Set::Scalar a, Set::Scalar ap, Set::Scalar fallback, Set::Scalar bt) const
    {
        if (!(a > eps && a < 1.0 - eps)) return fallback;
        if (!((ap - a) * (a - am) > 0.0)) return fallback;
        const Set::Scalar amin = std::min(am, ap), amax = std::max(am, ap);
        const Set::Scalar theta = (ap > am) ? 1.0 : -1.0;
        const Set::Scalar C = (a - amin + eps) / (amax - amin + eps);
        const Set::Scalar B = std::exp(theta * bt * (2.0 * C - 1.0));
        const Set::Scalar tb = std::tanh(bt);
        const Set::Scalar A = (B / std::cosh(bt) - 1.0) / tb;
        return amin + 0.5 * (amax - amin) * (1.0 + theta * (tb + A) / (1.0 + A * tb));
    }
};

}}}

#endif
