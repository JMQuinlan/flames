#!/usr/bin/env python3
"""
Structure of a viscous, heat-conducting shock at Pr = 3/4 (Becker 1922)
(tests/FlowConjugateHeat/input_1D_BeckerShock).

  python3 becker_shock_check.py <run_dir> [<run_dir> ...] [--outdir DIR]

Shock frame, w = S - u (speed of the gas relative to the shock), m = rho1 w1:
    (4/3) mu dw/dx = m (w1 - w) + p1 - p,    p = (m/w) R T,    T = (H - w^2/2)/cp,   H = cp T1 + w1^2/2
(the total enthalpy is uniform for Pr = 3/4).  Integrated with scipy from the mid-point of the wave; the
numerical profile is shifted to its best-fit position.  Also reported: shock speed from the positions of
two plot files, the Rankine-Hugoniot state behind the wave, the uniformity of the total enthalpy.
Errors are taken within 3 shock thicknesses of the wave (the start-up entropy spot left by the step initial
condition travels with the gas behind the shock and is excluded).
Several run directories (resolutions) give the convergence table.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import yt
yt.funcs.mylog.setLevel(40)
GAM, CP, CV, RHO1, P1, MACH = 1.4, 3.5, 2.5, 1.0, 1.0, 2.0
R = CP - CV


def meta(d, key):
    val = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def becker(mu):
    c1 = np.sqrt(GAM * P1 / RHO1); w1 = MACH * c1; m = RHO1 * w1; T1 = P1 / (RHO1 * R); H = CP * T1 + 0.5 * w1 * w1
    r = (GAM + 1) * MACH**2 / ((GAM - 1) * MACH**2 + 2); w2 = w1 / r
    p = lambda w: (m / w) * R * (H - 0.5 * w * w) / CP
    f = lambda x, w: [(m * (w1 - w[0]) + P1 - p(w[0])) / (4.0 / 3.0 * mu)]
    wm = 0.5 * (w1 + w2); L = 60 * (4.0 / 3.0 * mu) / m
    a = solve_ivp(f, (0, L), [wm], rtol=1e-11, atol=1e-13, dense_output=True)
    b = solve_ivp(f, (0, -L), [wm], rtol=1e-11, atol=1e-13, dense_output=True)
    def w(x):
        x = np.clip(np.asarray(x, float), -L, L)
        return np.where(x >= 0, a.sol(np.maximum(x, 0))[0], b.sol(np.minimum(x, 0))[0])
    return dict(w=w, w1=w1, w2=w2, m=m, H=H, S=w1, T=lambda ww: (H - 0.5 * ww * ww) / CP, thick=(w1 - w2) / f(0, [wm])[0])


def profile(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; Lx = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * Lx / n
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    return float(ds.current_time), x, q("velocityx"), q("T"), q("density"), q("pressure")


def analyse(d):
    mu = meta(d, "mu0"); B = becker(mu)
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    def fit(pf):
        t, x, u, T, rho, p = profile(pf); w = B["S"] - u
        xg = x[np.argmin(np.abs(w - 0.5 * (B["w1"] + B["w2"])))]
        res = minimize_scalar(lambda x0: np.sum((w - B["w"](x - x0)) ** 2), bounds=(xg - 0.05, xg + 0.05), method="bounded", options=dict(xatol=1e-9))
        return t, x, u, T, rho, p, w, res.x
    t, x, u, T, rho, p, w, x0 = fit(pfs[-1]); tb, _, _, _, _, _, _, xb = fit(pfs[len(pfs) // 2])
    wex = B["w"](x - x0); Tex = B["T"](wex); rex = B["m"] / wex
    # compare inside the wave only: the step initial condition leaves an entropy spot that is carried with the gas
    # behind the shock (6 - 12 thicknesses back at t = 0.5) and is not part of the steady structure
    win = np.abs(x - x0) < 3 * B["thick"]
    Hn = CP * T + 0.5 * w * w
    k = np.argmin(np.abs(x - (x0 - 12 * B["thick"])))          # a point well behind the wave
    return dict(d=d, t=t, dx=x[1] - x[0], x=x - x0, w=w, T=T, rho=rho, wex=wex, Tex=Tex, rex=rex, B=B, win=win,
                ew=np.abs(w - wex)[win].max() / (B["w1"] - B["w2"]), eT=np.abs(T - Tex)[win].max() / (B["T"](B["w2"]) - B["T"](B["w1"])),
                er=np.abs(rho - rex)[win].max() / (B["m"] / B["w2"] - RHO1), eH=np.abs(Hn - B["H"])[win].max() / B["H"],
                S=(x0 - xb) / (t - tb), behind=(rho[k], p[k], u[k]), cells=B["thick"] / (x[1] - x[0]))


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    A = sorted((analyse(d) for d in dirs), key=lambda r: -r["dx"])
    if outdir is None: outdir = os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[-1])))
    os.makedirs(outdir, exist_ok=True)
    B = A[0]["B"]; r2 = B["m"] / B["w2"]; p2 = P1 * (2 * GAM * MACH**2 - (GAM - 1)) / (GAM + 1); u2 = B["S"] - B["w2"]
    print(f"Becker shock, Mach {MACH:g}: thickness (w1 - w2)/max(dw/dx) = {B['thick']:.4f};  Rankine-Hugoniot rho2 {r2:.4f}, p2 {p2:.4f}, u2 {u2:.4f}, S {B['S']:.4f}")
    print(f"{'cells/thickness':>16} {'velocity':>10} {'temperature':>12} {'density':>9} {'total enthalpy':>15} {'shock speed':>12}   state behind (rho, p, u)")
    for a in A:
        print(f"{a['cells']:16.1f} {100 * a['ew']:8.3f} % {100 * a['eT']:10.3f} % {100 * a['er']:7.3f} % {100 * a['eH']:13.4f} % {100 * (a['S'] / B['S'] - 1):+10.3f} %"
              f"   {a['behind'][0]:.4f} {a['behind'][1]:.4f} {a['behind'][2]:.4f}")
    for a, b in zip(A[:-1], A[1:]):
        f = np.log(a["dx"] / b["dx"])
        print(f"  order: velocity {np.log(a['ew'] / b['ew']) / f:.2f}, temperature {np.log(a['eT'] / b['eT']) / f:.2f}, density {np.log(a['er'] / b['er']) / f:.2f}")
    with open(os.path.join(outdir, "becker_shock_summary.csv"), "w") as fo:
        fo.write("cells_per_thickness,err_velocity_pct,err_temperature_pct,err_density_pct,err_total_enthalpy_pct,err_shock_speed_pct\n")
        for a in A: fo.write(f"{a['cells']:.4g},{100 * a['ew']:.5g},{100 * a['eT']:.5g},{100 * a['er']:.5g},{100 * a['eH']:.5g},{100 * (a['S'] / B['S'] - 1):.5g}\n")

    a = A[len(A) // 2] if len(A) > 1 else A[0]; th = B["thick"]; ncol = 3 if len(A) > 1 else 2
    fig, ax = plt.subplots(1, ncol, figsize=(5.0 * ncol, 4.3))
    xx = np.linspace(-6 * th, 6 * th, 600); wx = B["w"](xx); m = np.abs(a["x"]) < 6 * th
    ax[0].plot(xx / th, (B["S"] - wx), "C3-", lw=1.6, label="Exact"); ax[0].plot(a["x"][m] / th, B["S"] - a["w"][m], "ko", ms=4, mfc="none", label="Numerical")
    ax[0].set_xlabel("Distance / Shock Thickness [-]"); ax[0].set_ylabel("Velocity [m/s]"); ax[0].set_title("Velocity"); ax[0].legend()
    ax[1].plot(xx / th, B["T"](wx), "C3-", lw=1.6, label="Exact"); ax[1].plot(a["x"][m] / th, a["T"][m], "ko", ms=4, mfc="none", label="Numerical")
    ax[1].set_xlabel("Distance / Shock Thickness [-]"); ax[1].set_ylabel("Temperature [K]"); ax[1].set_title("Temperature"); ax[1].legend()
    if len(A) > 1:
        n = np.array([q["cells"] for q in A]); c = ax[2]
        c.loglog(n, [100 * q["eT"] for q in A], "ko-", mfc="none", label="Temperature")
        c.loglog(n, [100 * q["ew"] for q in A], "ks--", mfc="none", label="Velocity")
        c.loglog(n, [100 * q["er"] for q in A], "k^:", mfc="none", label="Density")
        c.loglog(n, 100 * A[0]["eT"] * (n / n[0]) ** -2, "-", color="0.6", lw=1, label="Second Order")
        c.set_xlabel("Cells per Shock Thickness"); c.set_ylabel("Maximum Error [%]"); c.set_title("Convergence"); c.legend(fontsize=8)
        c.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); c.grid(which="minor", alpha=0.15)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle(f"Structure of a Mach {MACH:g} Shock with Viscosity and Heat Conduction (Pr = 3/4)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "becker_shock.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/becker_shock.png")


if __name__ == "__main__":
    main()
