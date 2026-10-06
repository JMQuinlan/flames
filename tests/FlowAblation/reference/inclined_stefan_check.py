#!/usr/bin/env python3
"""
Stefan problem with the wall at 26.57 deg to the grid (tests/FlowAblation/input_2D_InclinedStefan).

  python3 inclined_stefan_check.py <run_dir> [<aligned_closed_gap_run_dir>] [--outdir DIR]

Each face of the periodic slabs |frac(2x + y) - 1/2| < w must recede along its normal as in 1D:
    s(t) = 2 lam sqrt(alpha t),  lam exp(lam^2)(1 + erf lam) = rho cp (T_inf - T_w)/(sqrt(pi) rho_s Q*).
Measured two ways: from the solid area, s = (2 w0 - mean(1 - phi)) / (2 sqrt 5); and from the
phi = 0.5 contour, whose points give the half-width in the variable frac(2x + y) and its scatter
(flatness of the face, in cells).

The box is closed (periodic): the pressure and the far-field temperature fall as the gas cools
and sound waves cross the gas layer, so the run falls below the constant-pressure solution
(about -6 % at t = 2) whatever the wall orientation.  Give the grid-aligned run of the same
layer (input_1D_ClosedGap) as second argument for the like-for-like comparison.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
RHO, CP, MU, PR, TINF, TW, RHOS, QSTAR, W0 = 1.0, 3.5, 1.0e-3, 0.72, 1.0, 0.9, 1.0, 1.757, 0.16


def main():
    d = sys.argv[1]; dref = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") and sys.argv[1 if len(sys.argv) < 2 else 1] != "--outdir" else None
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    k = MU * CP / PR; al = k / (RHO * CP)
    lam = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * (TINF - TW) / (np.sqrt(np.pi) * RHOS * QSTAR), 1e-6, 5)
    H = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
        phi = np.asarray(cg[("boxlib", "phi")])[:, :, 0]
        s_area = (2 * W0 - (1 - phi).mean()) / (2 * np.sqrt(5))
        cs = plt.figure().gca().contour(x, x, phi.T, levels=[0.5]); plt.close()
        P = np.vstack([s for s in cs.allsegs[0] if len(s)])
        f = np.mod(2 * P[:, 0] + P[:, 1], 1.0) - 0.5          # +-w on the two faces
        s_cont = (W0 - np.abs(f).mean()) / np.sqrt(5); flat = np.abs(f).std() / np.sqrt(5) * n
        qv = np.asarray(cg[("boxlib", "ablation_q")])[:, :, 0]
        H.append((t, s_area, s_cont, flat, np.abs(np.asarray(cg[("boxlib", "velocityx")])).max(), qv[qv > 0].mean() if (qv > 0).any() else 0.0))
    H = np.array(H); t = H[:, 0]; s_ex = 2 * lam * np.sqrt(al * t); dx = 1.0 / n
    q_ex = k * (TINF - TW) * np.exp(-lam**2) / ((1 + erf(lam)) * np.sqrt(np.pi * al * t))
    print(f"{d}: lam = {lam:.5f}; {len(H)} plot files to t = {t[-1]:.2f}; dx = {dx:.4f}")
    for j in sorted(set(np.linspace(0, len(H) - 1, 4).round().astype(int))):
        print(f"  t = {t[j]:4.2f}: recession from the solid area {H[j, 1]:.5f}, from the contour {H[j, 2]:.5f}, 1D exact {s_ex[j]:.5f}"
              f"  ({100 * (H[j, 1] / s_ex[j] - 1):+.1f} % / {100 * (H[j, 2] / s_ex[j] - 1):+.1f} %);  face scatter {H[j, 3]:.2f} cells")
    print(f"  max gas speed {H[:, 4].max():.2e}")
    A = None
    if dref:
        A = []
        for pf in sorted(glob.glob(os.path.join(dref, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
            ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            ph = np.asarray(cg[("boxlib", "phi")])[:, :, 0]; A.append((float(ds.current_time), ph.mean() * float(ds.domain_width[0]) / 2))
        A = np.array(A); A[:, 1] -= A[0, 1]; s_al = np.interp(t, A[:, 0], A[:, 1])
        print(f"  against the grid-aligned closed layer {dref}:")
        for j in sorted(set(np.linspace(0, len(H) - 1, 4).round().astype(int))):
            print(f"    t = {t[j]:4.2f}: recession (solid area) {H[j, 1]:.5f}, aligned {s_al[j]:.5f}  ({100 * (H[j, 1] / s_al[j] - 1):+.1f} %)")
        print(f"    largest difference over the run {100 * np.abs(H[:, 1] / s_al - 1).max():.1f} %")
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(t, H[:, 1], "ko", ms=4, label="Hydro2 (solid area)"); ax.plot(t, H[:, 2], "C0s", ms=3, label="Hydro2 (contour)"); ax.plot(t, s_ex, "C3-", label="1D Stefan solution (constant pressure)")
    if A is not None: ax.plot(A[:, 0], A[:, 1], "C2--", label="Hydro2, grid-aligned closed layer")
    ax.set_xlabel("t"); ax.set_ylabel("recession along the wall normal"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "inclined_stefan.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/inclined_stefan.png")


if __name__ == "__main__":
    main()
