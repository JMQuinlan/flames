#!/usr/bin/env python3
"""
Free vibration of an elastic slab (tests/FlowElasticSolid/input_1D_SlabVibration).

  python3 slab_vibration_check.py <run_dir> [--outdir DIR]

Fundamental thickness modes of a slab of thickness L with traction-free faces:
    stretch  u = U sin(pi (x - x0)/L) cos(2 pi f_l t),  f_l = c_l/(2 L)
    shear    v = V sin(pi (x - x0)/L) cos(2 pi f_t t),  f_t = c_t/(2 L)
The modal amplitudes a_u(t), a_v(t) are the projections of u, v on sin(pi (x - x0)/L)
over the slab; frequency from a least-squares fit of A exp(-t/tau) cos(2 pi f t + phi).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import yt
yt.funcs.mylog.setLevel(40)
RHO, GAM, PINF, P0, MU, L, X0, U, RHOG = 10.0, 4.4, 100.0, 1.0, 50.0, 0.4, 0.5, 1.0e-3, 0.1


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    cl = np.sqrt(GAM * (P0 + PINF) / RHO + 4.0 / 3.0 * MU / RHO); ct = np.sqrt(MU / RHO)
    T = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*cell"))):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        m = q("rho_eta0"); sh = np.sin(np.pi * (x - X0) / L)
        w = m * sh / (m * sh * sh).sum()
        T.append((float(ds.current_time), (w * q("velocityx")).sum(), (w * q("velocityy")).sum(), q("pressure").min()))
    T = np.array(T); t = T[:, 0]
    model = lambda t, A, tau, f, ph: A * np.exp(-t / tau) * np.cos(2 * np.pi * f * t + ph)
    out = []
    for name, a, f0 in (("thickness-stretch", T[:, 1], cl / (2 * L)), ("thickness-shear", T[:, 2], ct / (2 * L))):
        try:
            (A, tau, f, ph), _ = curve_fit(model, t, a, p0=(U, 5.0, f0, 0.0), maxfev=20000)
        except RuntimeError:      # heavily damped (first-order runs): fit the first three periods only
            k = t < 3.0 / f0
            (A, tau, f, ph), _ = curve_fit(model, t[k], a[k], p0=(U, 0.5, f0, 0.0), maxfev=40000)
        out.append((name, f, f0, tau, A))
        print(f"  {name:18s} f = {f:8.4f}   theory {f0:8.4f}   ({100 * (f / f0 - 1):+.2f} %);  amplitude decay time {tau:7.3f}"
              f" = {tau * f:6.1f} periods  (loss per period {100 * (1 - np.exp(-1 / (tau * f))):.2f} %)")
    Zs, Zg = RHO * cl, np.sqrt(1.4 * P0 * RHOG); Rr = (Zs - Zg) / (Zs + Zg)
    print(f"  physical loss of the stretch mode to sound in the gas: 1 - R^2 = {100 * (1 - Rr**2):.2f} % per period; shear mode: 0")
    print(f"  minimum pressure over the run {T[:, 3].min():.4f}  (initial {P0})")
    with open(os.path.join(outdir, "slab_vibration_summary.csv"), "w") as fo:
        fo.write("mode,f,f_theory,error_pct,decay_time\n")
        for name, f, f0, tau, A in out: fo.write(f"{name},{f:.6g},{f0:.6g},{100 * (f / f0 - 1):.3f},{tau:.5g}\n")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for a_, col, (name, f, f0, tau, A) in zip(ax, (1, 2), out):
        a_.plot(t, T[:, col] / U, "k-", lw=0.9, label="Hydro2 modal amplitude")
        a_.plot(t, np.cos(2 * np.pi * f0 * t), "C3--", lw=0.7, label=f"theory, f = {f0:.3f}")
        a_.set_ylabel(name + " / initial"); a_.grid(alpha=0.3); a_.legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("t"); fig.tight_layout(); fig.savefig(os.path.join(outdir, "slab_vibration.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/slab_vibration.png")


if __name__ == "__main__":
    main()
