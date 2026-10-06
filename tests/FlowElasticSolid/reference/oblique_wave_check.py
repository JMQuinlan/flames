#!/usr/bin/env python3
"""
Plane elastic waves oblique to the grid (tests/FlowElasticSolid/input_2D_ObliqueWave).

  python3 oblique_wave_check.py <run_dir> [--outdir DIR]

Velocity projected on the longitudinal (n) and shear (t) polarisations of the mode
sin(k . x), k = 2 pi (2, 1).  Theory: a_L = UL cos(2 pi f_l t), a_T = UT cos(2 pi f_t t),
f = c |k| / (2 pi).  Fit: A exp(-t/tau) cos(2 pi f t + phi).  Also the amplitude left in
the cos(k . x) quadrature and the largest cross-talk when only one polarisation is excited.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import yt
yt.funcs.mylog.setLevel(40)
RHO, GAM, PINF, P0, MU = 1.0, 3.4, 10.0, 0.1, 12.0


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    cl = np.sqrt(GAM * (P0 + PINF) / RHO + 4.0 / 3.0 * MU / RHO); ct = np.sqrt(MU / RHO)
    nx_, ny_ = 2 / np.sqrt(5), 1 / np.sqrt(5)
    T = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*cell"))):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x, indexing="ij")
        u = np.asarray(cg[("boxlib", "velocityx")])[:, :, 0]; v = np.asarray(cg[("boxlib", "velocityy")])[:, :, 0]
        s = np.sin(2 * np.pi * (2 * X + Y))
        aL = 2 * ((u * nx_ + v * ny_) * s).mean(); aT = 2 * ((-u * ny_ + v * nx_) * s).mean()
        res = np.sqrt(((u - (aL * nx_ - aT * ny_) * s)**2 + (v - (aL * ny_ + aT * nx_) * s)**2).mean())
        T.append((float(ds.current_time), aL, aT, res))
    T = np.array(T); t = T[:, 0]
    model = lambda t, A, tau, f, ph: A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + ph)
    out = []
    print(f"{d}: {len(T)} plot files to t = {t[-1]:.3f}")
    for name, a, f0 in (("longitudinal", T[:, 1], cl * np.sqrt(5)), ("shear", T[:, 2], ct * np.sqrt(5))):
        if np.abs(a).max() < 1e-9: print(f"  {name:13s} not excited; max amplitude {np.abs(a).max():.1e}"); out.append(None); continue
        (A, tau, f, ph), _ = curve_fit(model, t, a, p0=(a[0], 1.0, f0, 0.0), maxfev=40000)
        out.append((name, f, f0, tau))
        print(f"  {name:13s} f = {f:8.4f}   theory {f0:8.4f}   ({100 * (f / f0 - 1):+.2f} %);  loss per period {100 * (1 - np.exp(-1 / (tau * f))):.2f} %")
    print(f"  rms velocity not in the two modes: max {T[:, 3].max():.2e}  (initial amplitude {max(abs(T[0, 1]), abs(T[0, 2])):.1e})")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a_, col, o, c in zip(ax, (1, 2), out, (cl, ct)):
        a_.plot(t, T[:, col], "k-", lw=0.9, label="Hydro2")
        if o: a_.plot(t, T[0, col] * np.cos(2 * np.pi * o[2] * t), "C3--", lw=0.7, label=f"theory, f = {o[2]:.3f}"); a_.set_ylabel(o[0] + " amplitude")
        a_.grid(alpha=0.3); a_.legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("t"); fig.tight_layout(); fig.savefig(os.path.join(outdir, "oblique_wave.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/oblique_wave.png")


if __name__ == "__main__":
    main()
