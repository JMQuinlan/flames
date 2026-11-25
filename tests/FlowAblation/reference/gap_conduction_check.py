#!/usr/bin/env python3
"""
Steady conduction across a gas gap to the ablating wall (tests/FlowAblation/input_1D_GapConduction).

  python3 gap_conduction_check.py <run_dir> [--outdir DIR]

Quasi-steady solution: q_w = k (T_h - T_w)/h,  rho_s Q* dh/dt = q_w  ->  h(t) = sqrt(h0^2 + 2 k dT t/(rho_s Q*)).
Gap width in the run = (x of the open end) - (phi = 0.5 crossing).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
CP, MU, PR, TH, TW, RSQ, XW, XE = 3.5, 1.0e-3, 0.72, 1.0, 0.9, 5.0, 0.1, 0.2


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    k = MU * CP / PR; H = []; prof = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) * XE / n; t = float(ds.current_time)
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1])
        H.append((t, XE - xs, q("ablation_q").max(), np.abs(q("velocityx")).max(), q("pressure")[phi > 0.5].mean()))
        prof = (t, x, q("T"), phi, xs)
    H = np.array(H); t = H[:, 0]; h0 = XE - XW
    hex_ = np.sqrt(h0**2 + 2 * k * (TH - TW) * t / RSQ)
    print(f"{d}: k = {k:.4e}; {len(H)} plot files to t = {t[-1]:.1f}")
    for tt in (5.0, 10.0, 20.0, t[-1]):
        j = np.argmin(np.abs(t - tt))
        print(f"  t = {t[j]:5.1f}: gap h = {H[j, 1]:.5f}   exact {hex_[j]:.5f}   (recession {H[j, 1] - h0:.5f} vs {hex_[j] - h0:.5f}, {100 * ((H[j, 1] - h0) / (hex_[j] - h0) - 1):+.1f} %);"
              f"   q_w {H[j, 2]:.4e} vs k dT/h(run) {k * (TH - TW) / H[j, 1]:.4e}  ({100 * (H[j, 2] * H[j, 1] / (k * (TH - TW)) - 1):+.1f} %)")
    tp, x, T, phi, xs = prof; fl = phi > 0.5
    Tlin = TW + (TH - TW) * (x - xs) / (XE - xs)
    print(f"  temperature at t = {tp:.1f}: max departure from the linear profile {np.abs(T - Tlin)[fl].max():.2e} of dT = {TH - TW:g};"
          f"  max gas speed {H[:, 3].max():.2e};  pressure {H[:, 4].min():.4f} .. {H[:, 4].max():.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(t, H[:, 1], "ko", ms=3, label="Hydro2"); ax[0].plot(t, hex_, "C3-", label=r"$\sqrt{h_0^2 + 2k\Delta T\,t/(\rho_sQ^*)}$"); ax[0].set_xlabel("t"); ax[0].set_ylabel("gap width h"); ax[0].legend()
    ax[1].plot(x[fl], T[fl], "k-", label="Hydro2"); ax[1].plot(x[fl], Tlin[fl], "C3--", label="linear"); ax[1].set_xlabel(f"x (t = {tp:.1f})"); ax[1].set_ylabel("T"); ax[1].legend()
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "gap_conduction.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/gap_conduction.png")


if __name__ == "__main__":
    main()
