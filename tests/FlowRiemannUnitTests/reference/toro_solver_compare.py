#!/usr/bin/env python3
"""
One-line-per-Riemann-solver comparison of the single-phase Toro shock tubes against the exact solution.

  python3 toro_solver_compare.py RUN_DIR --out FIG_DIR [--tests Toro1a,Toro2] [--solvers hllc,hlle] [--lim weno3]

Runs are RUN_DIR/<test>_<solver>_<lim> (decks from bin/tests/Riemann1D/make_decks.py).  The exact solution is
Toro's exact Riemann solver (Toro 2009, Ch. 4) for a stiffened gas: with p -> p + pi the stiffened-gas problem is
the ideal-gas one, so pi = 0 gives the Toro tests and pi > 0 the cavitation tube.
Writes <test>_comparison.{png,eps} per test (density, velocity, pressure) and prints / saves the relative L1 error
  E_q = sum |q - q_exact| / sum |q_exact|   (q = rho, u, p; for u the denominator is sum |u_exact| + N max|u_exact|*1e-12)
per test and solver to errors.csv.
"""
import os, sys, glob, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)

CASES = {  # rhoL, uL, pL, rhoR, uR, pR, t_end, gamma, pi
    "Toro1a": (1.0, 0.0, 1.0, 0.125, 0.0, 0.1, 0.25),
    "Toro1b": (1.0, 0.75, 1.0, 0.125, 0.0, 0.1, 0.2),
    "Toro1r": (0.125, 0.0, 0.1, 1.0, 0.0, 1.0, 0.25),
    "Toro2": (1.0, -2.0, 0.4, 1.0, 2.0, 0.4, 0.15),
    "Toro3": (1.0, 0.0, 1000.0, 1.0, 0.0, 0.01, 0.012),
    "Toro4a": (1.0, 0.0, 0.01, 1.0, 0.0, 100.0, 0.035),
    "Toro4b": (5.99924, 19.5975, 460.894, 5.99924, -6.19633, 46.0950, 0.035),
    "Toro5": (1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01, 0.035),
    "Toro6": (1.4, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0),
    "Toro7": (1.4, 0.1, 1.0, 1.0, 0.1, 1.0, 2.0),
    "Cavitation": (1000.0, -100.0, 1.0e5, 1000.0, 100.0, 1.0e5, 5.0e-4),   # stiffened gas: --gamma 7.15 --pi 3e8
}
NAME = dict(roe="Roe", hlle="HLLE", hllc="HLLC", hllce="HLLC-E", hllc_oomar_jaiman="HLLC (Oomar–Jaiman)", hllc_all_mach="HLLC All-Mach",
            hllc_all_mach_furfaro="HLLC All-Mach (Furfaro)", hllc_elastic="HLLC Elastic", hllc_solid="HLLC Solid")
MARK = "osD^v<>ph*"


def exact(case, x, g=1.4, pi=0.0):
    rl, ul, pl, rr, ur, pr, t = case[:7]
    pl, pr = pl + pi, pr + pi
    cl, cr = np.sqrt(g * pl / rl), np.sqrt(g * pr / rr)
    def f(p, rk, pk, ck):
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return (p - pk) * np.sqrt(A / (p + B))
        return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)
    def df(p, rk, pk, ck):
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
            return np.sqrt(A / (B + p)) * (1 - (p - pk) / (2 * (B + p)))
        return 1 / (rk * ck) * (p / pk) ** (-(g + 1) / (2 * g))
    p = max(1e-12 * min(pl, pr), 0.5 * (pl + pr))
    for _ in range(200):
        r = f(p, rl, pl, cl) + f(p, rr, pr, cr) + ur - ul
        pn = max(p - r / (df(p, rl, pl, cl) + df(p, rr, pr, cr)), 1e-14 * min(pl, pr))
        if abs(pn - p) < 1e-14 * (pn + p): p = pn; break
        p = pn
    us = 0.5 * (ul + ur) + 0.5 * (f(p, rr, pr, cr) - f(p, rl, pl, cl))
    R, U, P = np.empty_like(x), np.empty_like(x), np.empty_like(x)
    gm = (g - 1) / (g + 1)
    for i, s in enumerate(x / t):
        if s <= us:  # left of contact
            if p > pl:
                SL = ul - cl * np.sqrt((g + 1) / (2 * g) * p / pl + (g - 1) / (2 * g))
                st = (rl * (p / pl + gm) / (gm * p / pl + 1), us, p) if s >= SL else (rl, ul, pl)
            else:
                csl = cl * (p / pl) ** ((g - 1) / (2 * g)); SH, ST = ul - cl, us - csl
                if s <= SH: st = (rl, ul, pl)
                elif s >= ST: st = (rl * (p / pl) ** (1 / g), us, p)
                else:
                    c = 2 / (g + 1) * (cl + (g - 1) / 2 * (ul - s))
                    st = (rl * (c / cl) ** (2 / (g - 1)), 2 / (g + 1) * (cl + (g - 1) / 2 * ul + s), pl * (c / cl) ** (2 * g / (g - 1)))
        else:
            if p > pr:
                SR = ur + cr * np.sqrt((g + 1) / (2 * g) * p / pr + (g - 1) / (2 * g))
                st = (rr * (p / pr + gm) / (gm * p / pr + 1), us, p) if s <= SR else (rr, ur, pr)
            else:
                csr = cr * (p / pr) ** ((g - 1) / (2 * g)); SH, ST = ur + cr, us + csr
                if s >= SH: st = (rr, ur, pr)
                elif s <= ST: st = (rr * (p / pr) ** (1 / g), us, p)
                else:
                    c = 2 / (g + 1) * (cr - (g - 1) / 2 * (ur - s))
                    st = (rr * (c / cr) ** (2 / (g - 1)), 2 / (g + 1) * (-cr + (g - 1) / 2 * ur + s), pr * (c / cr) ** (2 * g / (g - 1)))
        R[i], U[i], P[i] = st
    return R, U, P - pi


def load(d):
    pfs = sorted(glob.glob(os.path.join(d, "*cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    if not pfs: return None
    ds = yt.load(pfs[-1]); ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    x = np.array(ad["boxlib", "x"])[:, 0, 0] if ("boxlib", "x") in ds.field_list else np.array(ad["index", "x"])[:, 0, 0]
    j = ds.domain_dimensions[1] // 2
    q = lambda n: np.array(ad["boxlib", n])[:, j, 0]
    return dict(t=float(ds.current_time), x=x, rho=q("density"), u=q("velocityx"), p=q("pressure"))


def main():
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    base = a[0]; out = opt("--out", "."); lim = opt("--lim", "weno3"); os.makedirs(out, exist_ok=True)
    tests = opt("--tests", ",".join(CASES)).split(","); solvers = opt("--solvers", ",".join(NAME)).split(",")
    g, pi = float(opt("--gamma", "1.4")), float(opt("--pi", "0"))
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 12, "xtick.labelsize": 13, "ytick.labelsize": 13})
    rows = []
    for t in tests:
        runs = [(s, load(os.path.join(base, f"{t}_{s}_{lim}"))) for s in solvers]; runs = [(s, r) for s, r in runs if r is not None]
        if not runs: continue
        xe = np.linspace(-1, 1, 4001); Re, Ue, Pe = exact(CASES[t], xe, g, pi)
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for k, (q, e, lab) in enumerate((("rho", Re, r"Density, $\rho$"), ("u", Ue, r"Velocity, $u$"), ("p", Pe, r"Pressure, $p$"))):
            ax[k].plot(xe, e, "k-", lw=2.0, label="Exact", zorder=10)
            ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
        for j, (s, r) in enumerate(runs):
            c = f"C{j}"; good = abs(r["t"] - CASES[t][6]) < 1e-3 * CASES[t][6]
            er = exact(CASES[t][:6] + (r["t"],), r["x"], g, pi); errs = []
            for q, e in zip(("rho", "u", "p"), er):
                den = np.abs(e).sum() if np.abs(e).max() > 0 else len(e)
                errs.append(np.abs(r[q] - e).sum() / den)
            rows.append((t, s, r["t"], *errs, "" if good else "STOPPED EARLY"))
            for k, q in enumerate(("rho", "u", "p")):
                ax[k].plot(r["x"], r[q], "-", color=c, lw=1.0, marker=MARK[j % len(MARK)], markevery=(j * 3, 25), ms=6, mfc="none", mew=1.1,
                           label="Numerical, " + NAME.get(s, s) + ("" if good else " (Stopped t = %.3g)" % r["t"]))
        ax[2].set_xlabel(r"Position, $x$"); ax[0].legend(loc="best", ncol=1, framealpha=0.9)
        ax[0].set_title(("%s Shock Tube" if t != "Cavitation" else "Stiffened-Gas %s Tube") % t + " at $t$ = %g" % CASES[t][6])
        fig.tight_layout()
        for ext in ("png", "eps"): fig.savefig(os.path.join(out, f"{t}_comparison.{ext}"), dpi=200 if ext == "png" else None)
        plt.close(fig)
    with open(os.path.join(out, "errors.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["test", "solver", "t", "L1_rho", "L1_u", "L1_p", "note"]); w.writerows(rows)
    print("%-8s %-24s %8s  %8s %8s %8s" % ("test", "solver", "t", "L1 rho%", "L1 u%", "L1 p%"))
    for r in rows: print("%-8s %-24s %8.4g  %8.3f %8.3f %8.3f  %s" % (r[0], r[1], r[2], 100 * r[3], 100 * r[4], 100 * r[5], r[6]))


if __name__ == "__main__":
    main()
