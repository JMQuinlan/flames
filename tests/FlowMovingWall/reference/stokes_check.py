#!/usr/bin/env python3
"""
Stokes' first problem (tests/FlowMovingWall/input_1D_Stokes): wall y = 0 started impulsively at U.

  python3 stokes_check.py <run_dir> [--outdir DIR]

Exact (incompressible): u(y, t) = U erfc(y / (2 sqrt(nu t))),  tau_w = mu U / sqrt(pi nu t).
Reports the profile error (max and L2, in units of U), the wall shear from the momentum the gas
has gained, d/dt int rho u dy (exact: tau_w; independent of where the wall is assumed to be), and
the position of the no-slip plane: the shift y0 that minimises the error of U erfc((y - y0)/...),
in cells (0 = on the fluid|solid face, -0.5 = centre of the first solid cell).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import minimize_scalar
import yt
yt.funcs.mylog.setLevel(40)
U, MU, RHO = 0.1, 2.0e-4, 1.0


def main():
    d = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True); nu = MU / RHO; H = []; prof = {}
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    for pf in pfs[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); t = float(ds.current_time)
        ny = ds.domain_dimensions[1]; dy = float(ds.domain_width[1]) / ny; y = float(ds.domain_left_edge[1]) + (np.arange(ny) + 0.5) * dy
        u = np.asarray(cg[("boxlib", "velocityx")])[0, :, 0]; mx = np.asarray(cg[("boxlib", "momentumx")])[0, :, 0]
        phi = np.asarray(cg[("boxlib", "phi")])[0, :, 0]; fl = phi > 0.5
        ex = lambda y0: U * erfc(np.maximum(y[fl] - y0, 0.0) / (2 * np.sqrt(nu * t)))
        y0 = minimize_scalar(lambda s: ((u[fl] - ex(s))**2).sum(), bounds=(-2 * dy, 2 * dy), method="bounded").x
        e = u[fl] - ex(0.0)
        H.append((t, np.abs(e).max() / U, np.sqrt((e**2).mean()) / U, y0 / dy, (mx[fl]).sum() * dy, np.abs(np.asarray(cg[("boxlib", "velocityy")])).max()))
        prof[t] = (y[fl], u[fl], ex(0.0))
    H = np.array(H); t = H[:, 0]; I_ex = 2 * MU * U * np.sqrt(t / (np.pi * nu))          # int_0^t tau_w dt
    print(f"{d}: {len(H)} plot files to t = {t[-1]:.2f}; dy = {dy:.4f}, delta(t_end) = {2 * np.sqrt(nu * t[-1]) / dy:.1f} cells")
    for j in sorted(set(np.linspace(0, len(H) - 1, 4).round().astype(int))):
        print(f"  t = {t[j]:4.2f}: profile error max {100 * H[j, 1]:.2f} % of U, L2 {100 * H[j, 2]:.2f} %;  no-slip plane at {H[j, 3]:+.2f} dy;"
              f"  momentum gained {H[j, 4]:.5e}   exact {I_ex[j]:.5e}   ({100 * (H[j, 4] / I_ex[j] - 1):+.1f} %)")
    print(f"  max |v| {H[:, 5].max():.2e}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, tt in enumerate(sorted(prof)[::max(1, len(prof) // 4)]):
        y, u, ex = prof[tt]; ax[0].plot(u / U, y, "o", ms=3, color=f"C{k}", label=f"t = {tt:.2f}"); ax[0].plot(ex / U, y, "-", color=f"C{k}", lw=1)
    ax[0].set_ylim(0, 0.12); ax[0].set_xlabel("u / U   (symbols Hydro2, lines exact)"); ax[0].set_ylabel("y"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(t, H[:, 4], "ko", ms=4, label="Hydro2"); ax[1].plot(t, I_ex, "C3-", label="exact"); ax[1].set_xlabel("t"); ax[1].set_ylabel(r"momentum gained $\int\rho u\,dy$")
    ax[1].legend(); ax[1].grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(outdir, "stokes.png"), dpi=180); plt.close(fig)
    print(f"  wrote {outdir}/stokes.png")


if __name__ == "__main__":
    main()
