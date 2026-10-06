#!/usr/bin/env python3
"""
1D Stefan problem for the ablating wall (tests/FlowAblation/input_1D_Stefan).

  python3 stefan_check.py <run_dir> [--outdir DIR] [--qstar Q*]     (Q* if the run overrides ablation.Qstar)

One-phase Stefan similarity solution (gas half-space at T_inf, wall at T_w, rho_s Q* ds/dt = q_w):
    s(t) = 2 lam sqrt(alpha t),   lam exp(lam^2) (1 + erf lam) = rho cp (T_inf - T_w)/(sqrt(pi) rho_s Q*)
    T(x, t) = T_w + (T_inf - T_w) (erf((x - x0)/(2 sqrt(alpha t))) + erf lam)/(1 + erf lam)
    q_w(t) = k (T_inf - T_w) exp(-lam^2)/((1 + erf lam) sqrt(pi alpha t))
Surface position in the run = the phi = 0.5 crossing.
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
RHO, CP, MU, PR, TINF, TW, RHOS, QSTAR, X0 = 1.0, 3.5, 1.0e-3, 0.72, 1.0, 0.9, 1.0, 1.757, 0.3


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    k = MU * CP / PR; al = k / (RHO * CP)
    QSTAR = float(sys.argv[sys.argv.index("--qstar") + 1]) if "--qstar" in sys.argv else globals()["QSTAR"]
    TW = float(sys.argv[sys.argv.index("--twall") + 1]) if "--twall" in sys.argv else globals()["TW"]
    lam = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * (TINF - TW) / (np.sqrt(np.pi) * RHOS * QSTAR), 1e-6, 5)
    H = []; prof = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1])
        H.append((t, X0 - xs, q("ablation_q").max(), np.abs(q("velocityx")).max(), q("pressure")[phi > 0.5].min()))
        prof = (t, x, q("T"), phi)
    H = np.array(H); t = H[:, 0]
    s_ex = 2 * lam * np.sqrt(al * t); q_ex = k * (TINF - TW) * np.exp(-lam**2) / ((1 + erf(lam)) * np.sqrt(np.pi * al * t))
    print(f"{d}: k = {k:.4e}, alpha = {al:.4e}, lam = {lam:.5f}; {len(H)} plot files to t = {t[-1]:.2f}")
    for tt in (1.0, 2.0, 4.0, t[-1]):
        j = np.argmin(np.abs(t - tt))
        print(f"  t = {t[j]:5.2f}: recession s = {H[j, 1]:.5f}   exact {s_ex[j]:.5f}   ({100 * (H[j, 1] / s_ex[j] - 1):+.1f} %);"
              f"   wall heat flux {H[j, 2]:.4e}   exact {q_ex[j]:.4e}   ({100 * (H[j, 2] / q_ex[j] - 1):+.1f} %)")
    tp, x, T, phi = prof; fl = phi > 0.5
    Tex = TW + (TINF - TW) * (erf((x - X0) / (2 * np.sqrt(al * tp))) + erf(lam)) / (1 + erf(lam))
    print(f"  temperature profile at t = {tp:.2f}: max error {np.abs(T - Tex)[fl].max():.2e} of dT = {TINF - TW:g}  ({100 * np.abs(T - Tex)[fl].max() / (TINF - TW):.1f} %)")
    print(f"  max gas speed over the run {H[:, 3].max():.2e}; min pressure {H[:, 4].min():.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(t, H[:, 1], "ko", ms=3, label="Hydro2"); ax[0].plot(t, s_ex, "C3-", label="Stefan solution"); ax[0].set_ylabel("surface recession s"); ax[0].set_xlabel("t")
    ax[1].loglog(t, H[:, 2], "ko", ms=3); ax[1].loglog(t, q_ex, "C3-"); ax[1].set_ylabel(r"wall heat flux $q_w$"); ax[1].set_xlabel("t")
    ax[2].plot(x[fl], T[fl], "k-", lw=1.5, label="Hydro2"); ax[2].plot(x[fl], Tex[fl], "C3--", label="Stefan solution")
    ax[2].axvspan(0, x[fl][0], color="0.8"); ax[2].set_xlim(0.2, 0.8); ax[2].set_ylabel("T"); ax[2].set_xlabel(f"x   (t = {tp:.2f}; grey = solid)")
    for a in ax: a.grid(alpha=0.3)
    ax[0].legend(); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "stefan.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/stefan.png")


if __name__ == "__main__":
    main()
