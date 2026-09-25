#!/usr/bin/env python3
"""
Analysis for tests/FlowMarmottant/input_Linear_ShellDamping_UNIT.

Fits  R(t) = R_eq + A exp(-beta t) cos(omega t + phi)  to each run's
gas-volume radius and reports the damping each viscous term actually adds:

    beta_mu  - beta_base   vs  analytic 2 mu      / (rho R0^2)
    beta_kap - beta_base   vs  analytic 2 kappa_s / (rho R0^3)

The two viscous runs are set up with the SAME analytic beta, so their ratio
is independent of domain geometry: if the bulk-viscous path (mu) and the
surface-stress path (kappa_s) are both right, ratio = 1.

Usage:
    analyze_linear_shell_damping.py --base DIR --mu DIR --kap DIR \\
        [--mu0 2e-3] [--kappa 4e-9] [--R0 2e-6] [--rho 1000] [--tmin 0]
"""
import argparse, glob, os, sys
import numpy as np


def radius_series(d):
    import yt
    yt.funcs.mylog.setLevel(50)
    fr = sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))
    t, R = [], []
    for pf in fr:
        ds = yt.load(pf)
        ad = ds.all_data()
        eta = np.asarray(ad["eta"], float)
        vol = np.asarray(ad["index", "cell_volume"], float)
        sym = 2 ** int(np.sum(np.abs(ds.domain_left_edge.v) < 1e-30))   # octant -> 8
        V = float(np.sum((1.0 - eta) * vol)) * sym
        t.append(float(ds.current_time))
        R.append((3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0))
    return np.array(t), np.array(R)


def fit(t, R):
    from scipy.optimize import curve_fit
    f = lambda tt, Req, A, b, w, ph: Req + A * np.exp(-b * tt) * np.cos(w * tt + ph)
    Req0 = R[-len(R) // 3:].mean()
    A0 = R[0] - Req0
    # frequency guess from zero crossings of R - Req
    s = np.sign(R - Req0)
    zc = t[1:][s[1:] != s[:-1]]
    w0 = np.pi / np.mean(np.diff(zc)) if len(zc) > 1 else 1.0e7
    p, cov = curve_fit(f, t, R, p0=[Req0, A0, 1.0e6, w0, 0.0], maxfev=20000)
    err = np.sqrt(np.diag(cov))
    resid = R - f(t, *p)
    return p, err, np.sqrt(np.mean(resid ** 2)) / abs(p[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mu", required=True)
    ap.add_argument("--kap", required=True)
    ap.add_argument("--mu0", type=float, default=2.0e-3)
    ap.add_argument("--kappa", type=float, default=4.0e-9)
    ap.add_argument("--R0", type=float, default=2.0e-6)
    ap.add_argument("--rho", type=float, default=1000.0)
    ap.add_argument("--pb", type=float, default=1.1e5, help="initial gas pressure")
    ap.add_argument("--pinf", type=float, default=1.0e5)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--tmin", type=float, default=0.0,
                    help="ignore frames before this time (start-up transient)")
    a = ap.parse_args()

    res = {}
    for name, d in (("base", a.base), ("mu", a.mu), ("kap", a.kap)):
        t, R = radius_series(d)
        m = t >= a.tmin
        (Req, A, b, w, ph), err, rel = fit(t[m], R[m])
        res[name] = (b, err[2], w, Req, A, rel, len(t))
        print("%-5s frames=%3d  R_eq/R0=%.4f  A/R0=%+.4f  omega=%.4e  beta=%.4e +- %.1e  fit rms/A=%.3f"
              % (name, len(t), Req / a.R0, A / a.R0, w, b, err[2], rel))

    # Physical equilibrium radius (sigma = 0): p_gas(R_eq) = p_inf.  The
    # gas-volume radius R_V carries a diffuse-band bias at coarse resolution,
    # and kappa_s's beta goes as 1/R^3, so the reference uses the physical R_eq.
    Req = a.R0 * (a.pb / a.pinf) ** (1.0 / (3.0 * a.gamma))
    print("\nR_eq physical/R0 = %.4f ; fitted R_V,eq/R0 = %.4f  (diffuse-band bias %.1f%%)"
          % (Req / a.R0, res["base"][3] / a.R0, 100 * (res["base"][3] / Req - 1)))
    beta_mu_an = 2.0 * a.mu0 / (a.rho * Req ** 2)
    beta_k_an = 2.0 * a.kappa / (a.rho * Req ** 3)
    bm = res["mu"][0] - res["base"][0]
    bk = res["kap"][0] - res["base"][0]
    print("\nanalytic (at R_eq): beta_mu = %.4e   beta_kappa = %.4e" % (beta_mu_an, beta_k_an))
    print("measured added   : beta_mu = %.4e   beta_kappa = %.4e" % (bm, bk))
    print("\n  bulk viscosity  mu      : measured/analytic = %.3f" % (bm / beta_mu_an))
    print("  shell viscosity kappa_s : measured/analytic = %.3f" % (bk / beta_k_an))
    print("  kappa_s / mu (geometry cancels)             = %.3f" % ((bk / beta_k_an) / (bm / beta_mu_an)))
    print("\nReading: 1.0 = correct.  ~0.55 for kappa_s with ~1.0 for mu isolates the")
    print("deficit to the surface-stress (Omega) path.")


if __name__ == "__main__":
    main()
