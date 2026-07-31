# -*- coding: utf-8 -*-
"""
===============================================================================
MARMOTTANT (2005) MODEL -- shared reference module for the FlowMarmottant tests
===============================================================================

Mirrors the C++ `MarmottantSigma()` in src/Integrator/Hydro2.cpp EXACTLY, so
the analysis scripts validate the *implementation* (curvature -> R_eff -> sigma)
rather than the physics of the 2D area law.

Three regimes (Marmottant et al., JASA 117(6), 2005):

    sigma(R_eff) =
        0,                              R_eff <= R_buck        (buckled)
        chi (R_eff^2/R_buck^2 - 1),     R_buck < R_eff < R_rupt (elastic ramp)
        sigma_break,                    R_eff >= R_rupt        (ruptured)

    R_rupt = R_buck * sqrt(1 + sigma_break/chi)

NOTE (deferred, see bin/Marmottant.md sec. 6): the C++ uses the spherical R^2
area law even in 2D-planar runs. This module copies that R^2 form on purpose so
the test checks code==model. The "correct 2D cylinder area law" (linear in R) is
a separate physics decision tracked for the 2D/3D compile split.

In a 2D-planar sim the clean mean curvature of a circle is kappa = 1/R, and the
code reconstructs R_eff = geom_factor/|kappa|, so with geom_factor = 1,
R_eff = R. The analysis extracts R independently (eta=0.5 crossing) and compares
sigma_model(R) against the code's sigma_eff field.

Public API:
    marmottant_sigma(R_eff, chi, R_buck, sigma_break)      -> scalar/array
    r_rupture(R_buck, chi, sigma_break)                    -> scalar
    sigma_rest(R0, chi, R_buck, sigma_break)               -> sigma_0 = sigma(R0)
    reff_from_kappa(kappa, geom_factor=1.0, small=1e-30)   -> R_eff
    laplace_jump(R, sigma, geom_factor=1.0)                -> sigma * (c_d / R)
    solve_coated(P, t_end)                                 -> (t, R, Rdot)
    MarmParams dataclass
===============================================================================
"""

from dataclasses import dataclass
import numpy as np

__all__ = [
    "marmottant_sigma", "r_rupture", "sigma_rest", "reff_from_kappa",
    "laplace_jump", "solve_coated", "MarmParams",
]


# ---------------------------------------------------------------------------
# sigma(R_eff) -- exact mirror of C++ MarmottantSigma()
# ---------------------------------------------------------------------------
def marmottant_sigma(R_eff, chi, R_buck, sigma_break):
    """Three-regime Marmottant surface tension. Accepts scalar or ndarray."""
    R_eff = np.asarray(R_eff, dtype=float)
    R_rupt = R_buck * np.sqrt(1.0 + sigma_break / chi)
    elastic = chi * ((R_eff * R_eff) / (R_buck * R_buck) - 1.0)
    sig = np.where(R_eff <= R_buck, 0.0,
                   np.where(R_eff >= R_rupt, sigma_break, elastic))
    return float(sig) if sig.ndim == 0 else sig


def r_rupture(R_buck, chi, sigma_break):
    """Rupture radius R_rupt = R_buck sqrt(1 + sigma_break/chi)."""
    return R_buck * np.sqrt(1.0 + sigma_break / chi)


def sigma_rest(R0, chi, R_buck, sigma_break):
    """Rest tension sigma_0 = sigma(R0) on the elastic ramp."""
    return marmottant_sigma(R0, chi, R_buck, sigma_break)


def reff_from_kappa(kappa, geom_factor=1.0, small=1e-30):
    """Effective radius from curvature: R_eff = geom_factor / |kappa|.
    Mirrors the C++ reconstruction (geom_factor = 1 for 2D-planar, 2 for sphere).
    """
    return geom_factor / (np.abs(kappa) + small)


def laplace_jump(R, sigma, geom_factor=1.0):
    """Pressure jump across the interface, sigma * (c_d / R).
    c_d = geom_factor: 1 for 2D-cylinder (sigma/R), 2 for sphere (2 sigma/R)."""
    return sigma * geom_factor / R


# ---------------------------------------------------------------------------
# Coated-bubble Chen-2D cylindrical RPE reference ODE
# ---------------------------------------------------------------------------
# Same Chen-2D form used by the FlowRayleighPlesset reference solver, with the
# constant sigma/R Laplace term replaced by marmottant_sigma(R)/R:
#
#   rho_l R ln(r_inf/R) Rdd + rho_l Rd^2 (0.5 - ln(r_inf/R))
#       = p_g(R) - p_inf - sigma(R)/R - 2 mu_l Rd/R
#   p_g(R) = p_g0 (R0/R)^(2 gamma)        (cylindrical adiabatic polytropic)
#
# This is the geometric match for the 2D-Cartesian (cylindrical) simulation. It
# is a qualitative reference -- it shares the code's R^2 area law via
# marmottant_sigma, but uses an incompressible-liquid far field, so expect a
# structured offset from the fully-compressible sim.
# ---------------------------------------------------------------------------
@dataclass
class MarmParams:
    # Liquid
    rho_l: float = 10.0        # scaled liquid density
    p_inf: float = 500.0       # far-field liquid pressure
    mu_l:  float = 0.0
    r_inf: float = 0.20        # 2D log far-field cutoff (~ half-domain)
    # Gas
    gamma_gas: float = 1.4
    p_g0:  float = 600.0       # initial gas pressure (the drive)
    R0:    float = 0.02
    Rdot0: float = 0.0
    # Marmottant shell
    chi:         float = 14.56
    R_buck:      float = 0.018
    sigma_break: float = 7.28
    geom_factor: float = 1.0


def _coated_chen2d_ode(t, y, P):
    R, Rd = y
    R = max(R, 1e-9)
    L = np.log(max(P.r_inf / R, 1.0 + 1e-12))   # ln(r_inf/R) > 0
    p_g = P.p_g0 * (P.R0 / R) ** (2.0 * P.gamma_gas)
    sig = marmottant_sigma(R, P.chi, P.R_buck, P.sigma_break)
    rhs = (p_g - P.p_inf
           - sig / R                 # 2D cylindrical Laplace term, sigma(R)/R
           - 2.0 * P.mu_l * Rd / R)
    Rdd = (rhs - P.rho_l * Rd * Rd * (0.5 - L)) / (P.rho_l * R * L)
    return [Rd, Rdd]


def solve_coated(P, t_end, n_eval=4000):
    """Integrate the coated Chen-2D RPE. Returns (t, R, Rdot).
    Uses scipy's LSODA when available; otherwise a fixed-step RK4 fallback
    (dt = t_end/2e5, fine enough for the collapse/rebound spikes)."""
    try:
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(0.0, t_end, n_eval)
        sol = solve_ivp(_coated_chen2d_ode, (0.0, t_end), [P.R0, P.Rdot0],
                        t_eval=t_eval, args=(P,), method="LSODA",
                        rtol=1e-8, atol=1e-12, max_step=t_end / 500.0)
        return sol.t, sol.y[0], sol.y[1]
    except ImportError:
        n_steps = 200_000
        dt = t_end / n_steps
        keep = max(1, n_steps // n_eval)
        y = np.array([P.R0, P.Rdot0])
        T, R, Rd = [0.0], [y[0]], [y[1]]
        f = lambda t, y: np.asarray(_coated_chen2d_ode(t, y, P))
        t = 0.0
        for s in range(n_steps):
            k1 = f(t, y)
            k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = f(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt
            if (s + 1) % keep == 0:
                T.append(t); R.append(y[0]); Rd.append(y[1])
        return np.array(T), np.array(R), np.array(Rd)


# ---------------------------------------------------------------------------
# self-test / quick sigma(R) plot when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    chi, R_buck, sigma_break, R0 = 14.56, 0.018, 7.28, 0.02
    R_rupt = r_rupture(R_buck, chi, sigma_break)
    s0 = sigma_rest(R0, chi, R_buck, sigma_break)
    print("Marmottant canonical (FlowMarmottant) parameters:")
    print(f"  chi          = {chi}")
    print(f"  R_buckling   = {R_buck}")
    print(f"  sigma_break  = {sigma_break}")
    print(f"  R0           = {R0}")
    print(f"  R_rupture    = {R_rupt:.6f}")
    print(f"  sigma_0(R0)  = {s0:.6f}")
    print(f"  Laplace jump = sigma_0/R0 = {laplace_jump(R0, s0):.4f}")
    print(f"  -> p_gas balance = 500 + jump = {500 + laplace_jump(R0, s0):.4f}")

    R = np.linspace(0.5 * R_buck, 1.4 * R_rupt, 600)
    sig = marmottant_sigma(R, chi, R_buck, sigma_break)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R * 1e3, sig, "b-", lw=2, label=r"$\sigma(R_{\rm eff})$")
    ax.axvline(R_buck * 1e3, color="0.5", ls="--", label=r"$R_{\rm buck}$")
    ax.axvline(R_rupt * 1e3, color="0.5", ls=":", label=r"$R_{\rm rupt}$")
    ax.plot([R0 * 1e3], [s0], "ro", ms=8, label=r"rest $(R_0,\sigma_0)$")
    ax.axhline(sigma_break, color="g", ls=":", alpha=0.6,
               label=r"$\sigma_{\rm break}$")
    ax.set_xlabel(r"$R_{\rm eff}$ [mm]")
    ax.set_ylabel(r"$\sigma$ (scaled)")
    ax.set_title("Marmottant (2005) three-regime surface tension")
    ax.grid(True, alpha=0.3)
    ax.legend()
    here = os.path.dirname(os.path.abspath(__file__))
    img = os.path.join(here, "Images")
    os.makedirs(img, exist_ok=True)
    out = os.path.join(img, "Marmottant_sigma_curve.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"\n  wrote {out}")
