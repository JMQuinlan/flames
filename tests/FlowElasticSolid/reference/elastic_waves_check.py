#!/usr/bin/env python3
"""
1D elastic waves in a solid (tests/FlowElasticSolid/input_1D_ElasticWaves).

  python3 elastic_waves_check.py <run_dir> [--outdir DIR]

A band |x| < w of solid is given a small velocity (du, dv).  Linear theory
(Favrie, Gavrilyuk & Saurel, JCP 228, 2009, Sec. 3.5): each band edge sends
out waves of half amplitude at
    c_l = sqrt(gamma (p + p_inf)/rho + (4/3) mu/rho)   (longitudinal, carries u)
    c_t = sqrt(mu/rho)                                  (shear, carries v)
with stresses  S_11 = -+ (4/3) mu u/c_l  and  S_12 = -+ rho c_t v  behind the
right/left-going fronts.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

RHO, GAM, PINF, P0, MU, W, DU, DV = 1.0, 3.4, 10.0, 0.1, 12.0, 0.1, 1.0e-3, 1.0e-3


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; lo, hi = float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0])
    x = lo + (np.arange(n) + 0.5) * (hi - lo) / n; t = float(ds.current_time)
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    u, v, Sxx, Sxy = g("velocityx"), g("velocityy"), g("elastic_Sxx"), g("elastic_Sxy")
    cl = np.sqrt(GAM * (P0 + PINF) / RHO + 4.0 / 3.0 * MU / RHO); ct = np.sqrt(MU / RHO)

    def front(q, amp):          # outermost half-height crossing of the amp/2 plateau, linear interpolation
        i = np.where(q > amp / 4)[0][-1]
        return x[i] + (x[i + 1] - x[i]) * (q[i] - amp / 4) / (q[i] - q[i + 1])
    xl, xt = front(u, DU), front(v, DV)
    # the right-going wave is the band itself, translated: it occupies [-W + c t, W + c t]
    pl = (x > cl * t - 0.6 * W) & (x < cl * t + 0.6 * W)       # right-going longitudinal plateau
    pt = (x > ct * t - 0.6 * W) & (x < ct * t + 0.6 * W)       # right-going shear plateau
    pl_l = (x < -cl * t + 0.6 * W) & (x > -cl * t - 0.6 * W)
    rows = [("c_l (front speed)", (xl - W) / t, cl), ("c_t (front speed)", (xt - W) / t, ct),
            ("u behind the right longitudinal front", u[pl].mean(), DU / 2),
            ("u behind the left longitudinal front", u[pl_l].mean(), DU / 2),
            ("v behind the shear front", v[pt].mean(), DV / 2),
            ("S_11 behind the right longitudinal front", Sxx[pl].mean(), -4.0 / 3.0 * MU * (DU / 2) / cl),
            ("S_12 behind the right shear front", Sxy[pt].mean(), -RHO * ct * DV / 2)]
    print(f"{d}: t = {t:.4f}")
    with open(os.path.join(outdir, "elastic_waves_summary.csv"), "w") as fo:
        fo.write("quantity,Hydro2,theory,error_pct\n")
        for k, a, b in rows:
            print(f"  {k:42s} {a:12.5e}   theory {b:12.5e}   ({100 * (a / b - 1):+.2f} %)")
            fo.write(f"{k},{a:.6g},{b:.6g},{100 * (a / b - 1):.3f}\n")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a, q, lab in ((ax[0, 0], u, r"$u$"), (ax[0, 1], v, r"$v$"), (ax[1, 0], Sxx, r"$S_{11}$"), (ax[1, 1], Sxy, r"$S_{12}$")):
        a.plot(x, q, "k-", lw=1); a.set_ylabel(lab); a.grid(alpha=0.3)
    for s in (-1, 1):
        for a in (ax[0, 0], ax[1, 0]): a.axvline(s * (W + cl * t), color="C3", ls="--", lw=0.8)
        for a in (ax[0, 1], ax[1, 1]): a.axvline(s * (W + ct * t), color="C0", ls="--", lw=0.8)
    for a in ax[1]: a.set_xlabel("x")
    fig.suptitle(rf"Elastic waves, t = {t:.3f}: dashed = theory fronts ($c_l$ = {cl:.3f}, $c_t$ = {ct:.3f})")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "elastic_waves.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/elastic_waves.png and elastic_waves_summary.csv")


if __name__ == "__main__":
    main()
