#!/usr/bin/env python3
"""
kappa_eff for the coated gas bubble on the RADIAL slab (input_Multiphase_Normalized_Slab3D).

R_vol(t) comes from the near-axis eta profile of the finest level, R^3 = 3 int (1 - eta) r^2 dr.
The kappa_s = 0 run on the same mesh (--baseline) gives the numerical error of the mesh,
e0(t) = R_K0(t) - KM_0(t); it is subtracted from the coated run, and kappa_eff/kappa_s is the
factor f for which Keller-Miksis with f kappa_s best fits R_K(t) - e0(t).

    analyze_bubble_slab.py RUN [RUN ...] --baseline RUN_K0 [--input DECK] [--tmax 10] [--plot out.png]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
from scipy.optimize import minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "FlowMarmottant", "reference")))
from marmottant_rpe_km import Shell, Liquid, Gas, solve_km   # noqa: E402

DECK = os.path.normpath(os.path.join(_HERE, "..", "input_Multiphase_Normalized_Slab3D"))


def parse(path, extra=()):
    kv = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    for o in extra:
        if "=" in o:
            k, v = o.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    return kv


def overrides(run):
    p = run.rstrip("/") + ".cmdline"
    if not os.path.exists(p):
        return []
    # slotrun writes "$@" joined by spaces; values with spaces were quoted pairs
    toks, out = open(p).read().split(), []
    for t in toks:
        if "=" in t or not out:
            out.append(t)
        else:
            out[-1] += " " + t
    return out


def series(run):
    import yt
    yt.funcs.mylog.setLevel(50)
    fs = sorted(glob.glob(os.path.join(run, "*cell")), key=lambda s: int(re.findall(r"(\d+)cell", s)[-1]))
    T, R = [], []
    for f in fs:
        ds = yt.load(f)
        lv = ds.index.max_level
        # finest level only near the band; build the axis row from a ray instead of a full covering grid
        dx = ds.domain_width.d / ds.domain_dimensions / 2 ** lv
        y0, z0 = 0.5 * dx[1], (0.5 * dx[2] if ds.dimensionality == 3 else 0.0)
        ray = ds.r[:, y0, z0] if ds.dimensionality == 3 else ds.r[:, y0]
        x = np.asarray(ray["index", "x"]); e = np.asarray(ray["boxlib", "eta"])
        o = np.argsort(x); x, e = x[o], e[o]
        r = np.sqrt(x ** 2 + y0 ** 2 + z0 ** 2)
        g = np.clip(1.0 - e, 0.0, 1.0)
        # cell widths of the ray samples (mixed levels)
        edges = np.concatenate(([0.0], 0.5 * (x[1:] + x[:-1]), [x[-1] + 0.5 * (x[-1] - x[-2])]))
        R.append((3.0 * np.sum(g * r ** 2 * np.diff(edges))) ** (1.0 / 3.0))
        T.append(float(ds.current_time))
    return np.array(T), np.array(R)


def km(kv, kappa, R0, t):
    rho = float(kv["density0.ic.expression.region0"])
    g, pi = float(kv["eos0.gamma"]), float(kv["eos0.p0"])
    p_inf = float(kv["pressure0.ic.expression.constant.p_inf"])
    p_b = float(kv["pressure1.ic.expression.constant.p_b"])
    c = np.sqrt(g * (p_inf + pi) / rho)
    sh = Shell(chi=0.0, sigma_water=0.0, kappa_s=kappa)
    liq = Liquid(rho=rho, mu=0.0, c=c, p_inf=p_inf)
    gas = Gas(p_g0=p_b, kappa_g=float(kv["eos1.gamma"]))
    tt, RR, _ = solve_km(R0, 0.0, (0.0, t[-1]), sh, liq, gas, t_eval=t)
    return RR


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--input", default=DECK)
    ap.add_argument("--tmax", type=float, default=None)
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    kv0 = parse(a.input, overrides(a.baseline))
    t0, R_0 = series(a.baseline)
    if a.tmax:
        m = t0 <= a.tmax; t0, R_0 = t0[m], R_0[m]
    R0 = R_0[0]
    e0 = R_0 - km(kv0, 0.0, R0, t0)
    print("%-26s %6s %6s %6s %7s | %8s %8s | %7s %7s" % ("run", "R0/dx", "eps/dx", "eps/R", "kappa",
                                                         "Rmin_sim", "Rmin_KM", "rmsK0", "k_eff/k"))
    fig = None
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t0, km(kv0, 0.0, R0, t0), "k-", lw=0.8, label="KM, kappa 0")
    for run in a.runs:
        kv = parse(a.input, overrides(run))
        t, R = series(run)
        # plot times jitter by a fraction of a step between runs: interpolate the baseline error
        m = (t >= t0[0]) & (t <= t0[-1])
        if a.tmax:
            m &= t <= a.tmax
        if m.sum() < 3:
            print("%-26s  no frames inside the baseline's time window; skipped" % os.path.basename(run)); continue
        t, R = t[m], R[m]
        e = np.interp(t, t0, e0)
        corr = R - e
        kap = float(kv["shell.kappa_s"])
        err = lambda f: np.sum((corr - km(kv, f * kap, R0, t)) ** 2)
        f = minimize_scalar(err, bounds=(0.0, 3.0), method="bounded", options={"xatol": 1e-3}).x if kap else np.nan
        Rk = km(kv, kap, R0, t)
        N = float(kv["amr.n_cell"].split()[0]) / (float(kv["geometry.prob_hi"].split()[0]))
        dxf = 1.0 / (N * 2 ** int(kv.get("amr.max_level", "0")))
        eps = float(kv["eta.ic.expression.constant.epsilon"])
        print("%-26s %6.0f %6.2f %6.4f %7.3g | %8.4f %8.4f | %7.4f %7.4f"
              % (os.path.basename(run.rstrip("/")), 1 / dxf, eps / dxf, eps, kap, corr.min(), Rk.min(),
                 np.sqrt(np.mean(e ** 2)), f))
        if fig is not None:
            ax.plot(t, corr, ".", ms=3, label=os.path.basename(run.rstrip("/")))
            ax.plot(t, Rk, "-", lw=0.8)
    if fig is not None:
        ax.set_xlabel("t"); ax.set_ylabel("R_vol (baseline-corrected)"); ax.legend(fontsize="x-small")
        fig.savefig(a.plot, dpi=140, bbox_inches="tight"); print("wrote", a.plot)


if __name__ == "__main__":
    main()
