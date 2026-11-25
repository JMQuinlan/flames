#!/usr/bin/env python3
"""
Scheme x resolution x pressure-ratio matrix of the 2D cylinder collapse unit runs (Stage 2 dynamics study).

  python3 osc2d_matrix.py <Osc2D dir> <prefix: s2u | s2c> --out DIR [--coated] [--np 8]

Runs are named <prefix>_pr<ratio>_L<level>_<scheme>_<d|c>  (d = donor alpha row, c = consistent).
Per run (cached in <run>/dynamics.npz): R_vol(t) from thermo.dat; from the plot files the mean radius of eta = 0.5,
rms(r)/<r> and the axis / diagonal radius ratio.  FAILED = the log shows a Riemann-solver error or a collapsed time step; "(at ... ns)" = still running.
Prints, per pressure ratio, minimum R_vol, first rebound maximum and the largest departure from round for every
scheme and resolution, and writes one figure per pressure ratio: R_vol(t) at each resolution, the departure from
round, and the convergence of the minimum radius.  Reference: cylindrical Rayleigh-Plesset (guide, see osc2d_dynamics.py).
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import osc2d_dynamics as D
R0 = 2.0e-6; TSTOP = 8.0e-7
NAME = dict(weno3="WENO3", weno5="WENO5", vanleer="MUSCL van Leer", minmod="MUSCL Minmod", thincvl="THINC / van Leer", thincw3="THINC / WENO3", muscl2="MUSCL2 (MC)", thinc="THINC / MUSCL2",
            nocomp="Shell, No Net Compression", elastic="Elastic Shell Only", plainst="Plain Surface Tension", visconly="Viscous Shell Only",
            fullP="Full Shell (Not Sub-Cycled)", nocompP="No Net Compression (Not Sub-Cycled)", visconlyP="Viscous Shell Only (Not Sub-Cycled)")
ROW = dict(d="Donor", c="Consistent")


def metrics(d):
    c = os.path.join(d, "dynamics.npz")
    th = os.path.join(d, "output", "thermo.dat")
    if not os.path.exists(th): return None
    if os.path.exists(c) and os.path.getmtime(c) > os.path.getmtime(th): z = np.load(c); return dict(t=z["t"], Rv=z["Rv"], sh=z["sh"])
    a = np.loadtxt(th, skiprows=1)
    if a.ndim < 2 or len(a) < 10: return None
    sh = D.shape(d); np.savez(c, t=a[:, 0], Rv=np.sqrt(4 * a[:, 1] / np.pi) / R0, sh=sh)
    return dict(t=a[:, 0], Rv=np.sqrt(4 * a[:, 1] / np.pi) / R0, sh=sh)


def state(d):
    """finished | failed | running, from the log next to the run directory."""
    try: txt = open(d + ".log", errors="ignore").read()
    except OSError: return "finished"
    if "finalized" in txt: return "finished"
    return "failed" if ("ERROR IN" in txt or "DT = 1e-15" in txt or "DT = 1e-17" in txt) else "running"


def main():
    args = sys.argv[1:]; opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    base, prefix = args[0], args[1]; out = opt("--out", os.path.join(HERE, "Images", "Osc2D")); coated = "--coated" in args; os.makedirs(out, exist_ok=True)
    dirs = sorted(d for d in glob.glob(os.path.join(base, prefix + "_pr*_L*_*")) if os.path.isdir(d))
    with Pool(int(opt("--np", "8"))) as p: M = p.map(metrics, dirs)
    R = {}
    for d, m in zip(dirs, M):
        g = re.match(prefix + r"_pr(\d+)_L(\d)_(\w+)_([dc])$", os.path.basename(d))
        if g and m is not None: m["state"] = state(d); R[(int(g.group(1)), int(g.group(2)), g.group(3), g.group(4))] = m
    prs = sorted({k[0] for k in R}); Ls = sorted({k[1] for k in R}); schemes = [s for s in NAME if any(k[2] == s for k in R)]
    for pr in prs:
        ref = D.reference(1.0e5 / pr, coated, TSTOP); print(f"\n=== pressure ratio {pr} ({'coated' if coated else 'uncoated'})")
        if ref is not None:
            i = int(np.argmin(ref[1])); k = i + int(np.argmax(ref[1][i:])); print("    cylindrical Rayleigh-Plesset (guide): R_min %.4f at %.0f ns, rebound %.4f at %.0f ns" % (ref[1][i] / R0, 1e9 * ref[0][i], ref[1][k] / R0, 1e9 * ref[0][k]))
        print("    %-30s" % "scheme, alpha row" + "".join("|  %5.1f cells/R: Rmin   t[ns]  rebound  round%%  axis/diag  " % (12.8 * 2 ** L) for L in Ls))
        fig, ax = plt.subplots(2, len(Ls), figsize=(5.6 * len(Ls), 9.0), squeeze=False); j = 0
        for s in schemes:
            for r in "dc":
                line = "    %-30s" % (NAME[s] + ", " + ROW[r]); c = f"C{j % 10}"; mk = D.MARK[j % len(D.MARK)]; j += 1
                for q, L in enumerate(Ls):
                    m = R.get((pr, L, s, r))
                    if m is None: line += "|" + " " * 56; continue
                    t, Rv, sh = m["t"], m["Rv"], m["sh"]; fail = m["state"] == "failed"; run = m["state"] == "running"; i = int(np.argmin(Rv)); k = i + int(np.argmax(Rv[i:]))
                    rnd = 100 * sh[:, 2].max() if len(sh) else np.nan; qd = sh[int(np.argmin(np.abs(sh[:, 0] - t[i]))), 3] if len(sh) else np.nan
                    line += "|  %s  %.4f  %5.0f  %.4f  %6.2f   %.3f     " % ("FAILED %3.0f ns" % (1e9 * t[-1]) if fail else "(at %3.0f ns)   " % (1e9 * t[-1]) if run else "              ", Rv[i], 1e9 * t[i], Rv[k], rnd, qd)
                    kw = dict(color=c, lw=0.7, ms=3.5, mfc="none", mec=c, mew=0.7)
                    ax[0, q].plot(1e9 * t, Rv, "-" + mk, markevery=max(1, len(t) // 30), label=NAME[s] + ", " + ROW[r] + (" (Failed)" if fail else ""), **kw)
                    if len(sh): ax[1, q].plot(1e9 * sh[:, 0], 100 * sh[:, 2], "-" + mk, markevery=max(1, len(sh) // 30), **kw)
                print(line)
        for q, L in enumerate(Ls):
            if ref is not None: ax[0, q].plot(1e9 * ref[0], ref[1] / R0, "k-", lw=1.5, zorder=0, label="Cylindrical Rayleigh-Plesset (Guide)")
            ax[0, q].set_title("Volumetric Radius, %.1f Cells per Radius" % (12.8 * 2 ** L)); ax[0, q].set_xlabel("Time [ns]"); ax[0, q].set_ylabel(r"$R_{vol}/R_0$ [-]"); ax[0, q].grid(alpha=0.3)
            ax[1, q].set_title("Departure From Round, %.1f Cells per Radius" % (12.8 * 2 ** L)); ax[1, q].set_xlabel("Time [ns]"); ax[1, q].set_ylabel(r"rms($r$)/$\langle r \rangle$ of $\eta$ = 0.5 [%]")
            ax[1, q].set_yscale("log"); ax[1, q].set_ylim(1e-2, 1e2); ax[1, q].grid(alpha=0.3)
        ax[0, 0].legend(fontsize=6, ncol=2)
        fig.suptitle("%s Cylinder Collapse Unit Test (Numerical), Pressure Ratio %d" % ("Coated" if coated else "Uncoated", pr)); fig.tight_layout()
        o = os.path.join(out, f"{prefix}_pr{pr}.png"); fig.savefig(o, dpi=170); plt.close(fig); print("    wrote", o)


if __name__ == "__main__":
    main()
