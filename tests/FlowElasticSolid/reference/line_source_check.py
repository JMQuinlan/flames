#!/usr/bin/env python3
"""
Cylindrical wave from a line source (tests/FlowElasticSolid/input_2D_LineSource).

  python3 line_source_check.py <run_dir> [--outdir DIR]

Reference: the axisymmetric plane-strain linear elastodynamic equations
    rho v_t = d(s_rr)/dr + (s_rr - s_tt)/r,
    (s_rr)_t = (lam + 2 mu) dv/dr + lam v/r,   (s_tt)_t = lam dv/dr + (lam + 2 mu) v/r
integrated with a staggered leapfrog scheme on a radial grid 20x finer than the 2D grid
(v = A (r/r0) exp(-r^2/r0^2), zero stress perturbation at t = 0).
Compared: v_r(r), sigma_rr = -(p - p0) + S_rr and sigma_tt = -(p - p0) + S_tt sampled along rays at
0, 26.57 and 45 deg; the azimuthal velocity (exactly 0).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import yt
yt.funcs.mylog.setLevel(40)
RHO, GAM, PINF, P0, MU, A, R0, XC = 1.0, 3.4, 10.0, 0.1, 12.0, 1.0e-3, 0.1, 1.0


def reference(t_end, dr=2.5e-4, rmax=0.98):
    M = GAM * (P0 + PINF) + 4.0 / 3.0 * MU; lam = M - 2 * MU
    n = int(rmax / dr); rv = np.arange(n + 1) * dr; rs = (np.arange(n) + 0.5) * dr      # v at nodes, stress at centres
    v = A * rv / R0 * np.exp(-(rv / R0)**2); srr = np.zeros(n); stt = np.zeros(n)
    dt = 0.4 * dr / np.sqrt(M / RHO); nst = int(np.ceil(t_end / dt)); dt = t_end / nst
    for _ in range(nst):
        dv = (v[1:] - v[:-1]) / dr; vc = 0.5 * (v[1:] + v[:-1]) / rs
        srr += dt * (M * dv + lam * vc); stt += dt * (lam * dv + M * vc)
        v[1:-1] += dt / RHO * ((srr[1:] - srr[:-1]) / dr + 0.5 * ((srr - stt)[1:] + (srr - stt)[:-1]) / rv[1:-1])
    return rv, v, rs, srr, stt


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n; t = float(ds.current_time)
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
    F = {k: RegularGridInterpolator((x, x), q(k)) for k in ("velocityx", "velocityy", "pressure", "elastic_Sxx", "elastic_Syy", "elastic_Sxy")}
    rv, vref, rs, srr_ref, stt_ref = reference(t)
    r = np.linspace(0.02, 0.9, 441)
    vi = np.interp(r, rv, vref); si = np.interp(r, rs, srr_ref); ti = np.interp(r, rs, stt_ref)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    ax[0].plot(r, vi, "k-", lw=2, label="1D radial reference"); ax[1].plot(r, si, "k-", lw=2); ax[2].plot(r, ti, "k-", lw=2)
    print(f"{d}: t = {t:.4f}; reference peak v_r {np.abs(vref).max():.4e} at r = {rv[np.abs(vref).argmax()]:.4f}")
    with open(os.path.join(outdir, "line_source_summary.csv"), "w") as fo:
        fo.write("angle_deg,peak_vr,peak_vr_ref,r_peak,r_peak_ref,rel_L2_vr,rel_L2_srr,rel_L2_stt,max_vtheta\n")
        for ang in (0.0, np.degrees(np.arctan(0.5)), 45.0):
            c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            P = np.column_stack([XC + r * c, XC + r * s])
            u, v = F["velocityx"](P), F["velocityy"](P); vr, vt = u * c + v * s, -u * s + v * c
            dp = F["pressure"](P) - P0; Sxx, Syy, Sxy = F["elastic_Sxx"](P), F["elastic_Syy"](P), F["elastic_Sxy"](P)
            srr = -dp + Sxx * c * c + Syy * s * s + 2 * Sxy * s * c; stt = -dp + Sxx * s * s + Syy * c * c - 2 * Sxy * s * c
            e = lambda a, b: np.sqrt(((a - b)**2).sum() / (b**2).sum())
            k, kr = np.abs(vr).argmax(), np.abs(vi).argmax()
            print(f"  ray {ang:5.2f} deg: peak v_r {vr[k]:+.4e} (ref {vi[kr]:+.4e}, {100 * (vr[k] / vi[kr] - 1):+.1f} %) at r = {r[k]:.3f} (ref {r[kr]:.3f});"
                  f"  L2 error v_r {100 * e(vr, vi):.1f} %, sigma_rr {100 * e(srr, si):.1f} %, sigma_tt {100 * e(stt, ti):.1f} %;  max |v_theta| {np.abs(vt).max():.1e}")
            fo.write(f"{ang:.2f},{vr[k]:.6g},{vi[kr]:.6g},{r[k]:.4f},{r[kr]:.4f},{e(vr, vi):.4f},{e(srr, si):.4f},{e(stt, ti):.4f},{np.abs(vt).max():.3g}\n")
            ax[0].plot(r, vr, "--", lw=1, label=f"Hydro2, {ang:.1f} deg"); ax[1].plot(r, srr, "--", lw=1); ax[2].plot(r, stt, "--", lw=1)
    for a_, yl in zip(ax, (r"$v_r$", r"$\sigma_{rr}$ perturbation", r"$\sigma_{\theta\theta}$ perturbation")):
        a_.set_xlabel("r"); a_.set_ylabel(yl); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=8); fig.suptitle(f"Cylindrical elastic wave from a line source, t = {t:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "line_source.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/line_source.png")


if __name__ == "__main__":
    main()
