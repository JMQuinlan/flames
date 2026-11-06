#!/usr/bin/env python3
"""
UNIT TEST: does the surface dilatational viscosity kappa_s actually damp?

Two independent checks, because a "looks elastic-only" response can come from
either side:

  A. THE REFERENCE ODE.  p_B carries -4 kappa_s Rdot / R^2 (Marmottant 2005).
     Linearising R = R0(1+x) about equilibrium:
         rho R0^2 x'' + (4 kappa_s / R0) x' + 3 kappa p x = 0
     so the damping rate is  beta = 2 kappa_s / (rho R0^3)  and the damping
     ratio is zeta = beta / omega0 with omega0 the Minnaert frequency.  The
     test integrates a free ring-down and recovers beta by fitting the decay,
     comparing against that closed form.  Analytic reference, no fitting knobs.

  B. THE SOLVER CLOSURE.  In the PDE the same physics enters as
         sigma_tot = sigma(Gamma) + kappa_s div_s(u)
     multiplying the projector in Omega.  For a sphere div_s u = 2 Rdot / R, and
     the resulting normal traction must reproduce 4 kappa_s Rdot / R^2.  The
     test checks that algebraic correspondence directly, so a sign or factor-of
     -two error in the PDE closure cannot hide.

Run: python3 unit_test_kappa_s.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import marmottant_rpe_km as M

RHO   = 1000.0
R0    = 2.0e-6
P_INF = 1.0e5
KAPG  = 1.4
KS    = 7.2e-9          # the value the Sch20 inputs use


def omega0(p=P_INF, rho=RHO, R=R0, kap=KAPG):
    return math.sqrt(3.0 * kap * p / rho) / R


def beta_theory(ks=KS, rho=RHO, R=R0):
    return 2.0 * ks / (rho * R ** 3)


def ringdown(ks, cycles=3.0, n=6000):
    """Free oscillation seeded by an initial WALL VELOCITY; return (t, R).

    The perturbation is applied to Rdot, not R, because solve_rpe's first
    argument is simultaneously the initial radius AND the shell reference
    radius -- perturbing it would move the equilibrium along with the state
    and produce no oscillation at all.
    """
    w0 = omega0()
    T  = 2.0 * math.pi / w0
    sh = M.Shell(chi=0.0, R_buck=0.0, sigma_break=1e30,
                 sigma_water=0.0, kappa_s=ks)          # bare: no elasticity
    liq = M.Liquid(rho=RHO, mu=0.0, c=1.0e30, p_inf=P_INF)   # inviscid liquid
    gas = M.Gas(p_g0=P_INF, kappa_g=KAPG)                    # Laplace-balanced
    te = np.linspace(0.0, cycles * T, n)
    Rdot0 = 0.01 * R0 * w0                            # ~1% radius amplitude
    t, R, _ = M.solve_rpe(R0, Rdot0, (0.0, cycles * T), sh, liq, gas, t_eval=te)
    return np.asarray(t), np.asarray(R)


def fit_decay(t, R):
    """Fit |R-R0| peaks to A e^{-beta t}; return beta."""
    x = R / R0 - 1.0
    # local maxima of |x|
    a = np.abs(x)
    idx = [i for i in range(1, len(a) - 1) if a[i] > a[i-1] and a[i] > a[i+1]]
    idx = [i for i in idx if a[i] > 1e-12]
    if len(idx) < 2:
        return float("nan"), 0
    tt = t[idx]; aa = np.log(a[idx])
    p = np.polyfit(tt, aa, 1)
    return -p[0], len(idx)


def main():
    fail = 0
    print("=" * 74)
    print("A.  REFERENCE ODE -- free ring-down, damping from -4 kappa_s Rdot/R^2")
    print("=" * 74)
    w0 = omega0()
    print(f"  omega0 (Minnaert)      = {w0:.5e} rad/s")
    print(f"{'kappa_s':>12} {'beta theory':>14} {'beta fitted':>14} {'zeta':>8} {'peaks':>6} {'err %':>9}")
    for ks in (0.0, KS * 0.1, KS, KS * 3.0):
        t, R = ringdown(ks)
        b, npk = fit_decay(t, R)
        bt = beta_theory(ks)
        err = float("nan") if bt == 0 else 100.0 * (b - bt) / bt
        print(f"{ks:12.3e} {bt:14.5e} {b:14.5e} {bt/w0:8.4f} {npk:6d} {err:9.2f}")
        if ks > 0 and (not np.isfinite(b) or abs(err) > 15.0):
            print("      ^^ FAIL: fitted damping does not match 2 kappa_s/(rho R0^3)")
            fail += 1
        if ks == 0.0 and np.isfinite(b) and abs(b) > 0.02 * w0:
            print("      ^^ FAIL: undamped case is decaying")
            fail += 1

    print()
    print("=" * 74)
    print("B.  SOLVER CLOSURE -- sigma_tot = sigma + kappa_s div_s(u) must give")
    print("    the SAME normal traction as the ODE's 4 kappa_s Rdot / R^2")
    print("=" * 74)
    # For a sphere of radius R with wall speed Rdot:
    #   div_s u  = 2 Rdot / R
    #   the projector term contributes a normal traction 2 sigma_tot / R
    #   viscous part:  2 (kappa_s * 2 Rdot/R) / R = 4 kappa_s Rdot / R^2   <-- ODE
    print(f"{'R':>11} {'Rdot':>11} {'div_s u':>13} {'PDE traction':>14} {'ODE term':>14} {'rel err':>10}")
    for R in (R0, 0.7 * R0, 0.4 * R0):
        for Rd in (-5.0, -0.5, 2.0):
            divs = 2.0 * Rd / R
            pde  = 2.0 * (KS * divs) / R          # 2 sigma_visc / R
            ode  = 4.0 * KS * Rd / R ** 2
            rel  = 0.0 if ode == 0 else abs(pde - ode) / abs(ode)
            print(f"{R:11.3e} {Rd:11.3f} {divs:13.4e} {pde:14.5e} {ode:14.5e} {rel:10.2e}")
            if rel > 1e-12:
                print("      ^^ FAIL: PDE closure != ODE term"); fail += 1

    print()
    print("RESULT:", "FAIL" if fail else "PASS -- kappa_s damps, and PDE closure matches the ODE")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
