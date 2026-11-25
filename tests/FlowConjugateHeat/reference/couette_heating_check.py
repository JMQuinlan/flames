#!/usr/bin/env python3
"""
Plane Couette flow with viscous heating between two isothermal walls
(tests/FlowConjugateHeat/input_1D_CouetteHeating).

  python3 couette_heating_check.py <run_dir> [<run_dir> ...] [--outdir DIR]

Exact steady state (constant mu, k):
    u(y) = U (1 - y/h),   T(y) = T_w + mu U^2/(2 k) (y/h)(1 - y/h),   q_w = mu U^2/(2 h) into each wall.
Several run directories (different resolutions) give the convergence table and plot.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import yt
yt.funcs.mylog.setLevel(40)
CP, H = 3.5, 0.1


def meta(d, key):
    val = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def load(d):
    pf = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    ny = ds.domain_dimensions[1]; lo, hi = float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1])
    y = lo + (np.arange(ny) + 0.5) * (hi - lo) / ny
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[0, :, 0]
    fl = q("phi") >= 0.5
    mu, Pr, U, Tw = meta(d, "mu0"), meta(d, "thermal.Pr"), meta(d, "solid.momentum.ic.expression.constant.U"), meta(d, "solid.T_wall")
    k = mu * CP / Pr; dy = y[1] - y[0]
    yf, u, T = y[fl], q("velocityx")[fl], q("T")[fl]
    uex = U * (1 - yf / H); Tex = Tw + mu * U * U / (2 * k) * (yf / H) * (1 - yf / H); dT = mu * U * U / (8 * k)
    qw_lo = k * (T[0] - Tw) / (0.5 * dy); qw_hi = k * (T[-1] - Tw) / (0.5 * dy); qex = mu * U * U / (2 * H)
    return dict(t=float(ds.current_time), n=int(fl.sum()), dy=dy, y=yf, u=u, T=T, uex=uex, Tex=Tex, U=U, dT=dT, Tw=Tw,
                eu=np.abs(u - uex).max() / U, eT=np.abs(T - Tex).max() / dT, qlo=qw_lo, qhi=qw_hi, qex=qex,
                p=q("pressure")[fl], vy=np.abs(q("velocityy")[fl]).max())


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    R = sorted((load(d) for d in dirs), key=lambda r: r["n"])
    if outdir is None: outdir = os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[-1])))
    os.makedirs(outdir, exist_ok=True)
    print(f"{'cells':>6} {'t':>6} {'max |u - exact|/U':>18} {'max |T - exact|/dT':>19} {'q_w lower':>12} {'q_w upper':>12} {'exact':>10}")
    for r in R:
        print(f"{r['n']:6d} {r['t']:6.2f} {100 * r['eu']:16.3f} % {100 * r['eT']:17.3f} % {r['qlo']:12.5e} {r['qhi']:12.5e} {r['qex']:10.4e}"
              f"   ({100 * (r['qlo'] / r['qex'] - 1):+.2f} %, {100 * (r['qhi'] / r['qex'] - 1):+.2f} %)")
    for a, b in zip(R[:-1], R[1:]):
        f = np.log(b["n"] / a["n"])
        ou = f"{np.log(a['eu'] / b['eu']) / f:.2f}" if min(a["eu"], b["eu"]) > 1e-4 else "exact to round-off (linear profile)"
        print(f"  order {a['n']} -> {b['n']}: temperature {np.log(a['eT'] / b['eT']) / f:.2f}, wall heat flux "
              f"{np.log(abs(a['qlo'] / a['qex'] - 1) / abs(b['qlo'] / b['qex'] - 1)) / f:.2f}, velocity {ou}")
    r = R[-1] if len(R) == 1 else R[len(R) // 2]
    print(f"  pressure across the gap {r['p'].min():.5f} .. {r['p'].max():.5f}; max |v| {r['vy']:.1e}")
    with open(os.path.join(outdir, "couette_heating_summary.csv"), "w") as fo:
        fo.write("cells,err_u_pct,err_T_pct,qw_lower,qw_upper,qw_exact\n")
        for q in R: fo.write(f"{q['n']},{100 * q['eu']:.5g},{100 * q['eT']:.5g},{q['qlo']:.6g},{q['qhi']:.6g},{q['qex']:.6g}\n")

    ncol = 3 if len(R) > 1 else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.3))
    yy = np.linspace(0, H, 300)
    ax[0].plot(r["U"] * (1 - yy / H), yy, "C3-", lw=1.6, label="Exact"); ax[0].plot(r["u"], r["y"], "ko", ms=4, mfc="none", label="Numerical")
    ax[0].set_xlabel("Velocity [m/s]"); ax[0].set_ylabel("Height [m]"); ax[0].set_title("Velocity Profile"); ax[0].legend()
    ax[1].plot(r["Tw"] + 4 * r["dT"] * (yy / H) * (1 - yy / H), yy, "C3-", lw=1.6, label="Exact"); ax[1].plot(r["T"], r["y"], "ko", ms=4, mfc="none", label="Numerical")
    ax[1].set_xlabel("Temperature [K]"); ax[1].set_ylabel("Height [m]"); ax[1].set_title("Temperature Profile"); ax[1].legend()
    if len(R) > 1:
        n = np.array([q["n"] for q in R], float); a = ax[2]
        a.loglog(n, [100 * q["eT"] for q in R], "ko-", mfc="none", label="Temperature")
        a.loglog(n, [100 * q["eu"] for q in R], "ks--", mfc="none", label="Velocity")
        a.loglog(n, [100 * abs(q["qlo"] / q["qex"] - 1) for q in R], "k^:", mfc="none", label="Wall Heat Flux")
        a.loglog(n, 100 * R[0]["eT"] * (n / n[0]) ** -2, "-", color="0.6", lw=1, label="Second Order")
        a.set_xlabel("Cells Across the Gap"); a.set_ylabel("Maximum Error [%]"); a.set_title("Convergence"); a.legend(fontsize=8)
        a.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); a.grid(which="minor", alpha=0.15)
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle("Plane Couette Flow with Viscous Heating Between Isothermal Walls"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "couette_heating.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/couette_heating.png")


if __name__ == "__main__":
    main()
