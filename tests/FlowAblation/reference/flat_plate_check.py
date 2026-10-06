#!/usr/bin/env python3
"""
Laminar boundary-layer heat flux on a flat plate at Mach 2 (tests/FlowAblation/input_2D_FlatPlate).

  python3 flat_plate_check.py <run_dir> [--outdir DIR]

Reference (compressible laminar flat plate, Eckert reference temperature):
    q_w(x) = St rho_e U cp (T_aw - T_w),   St = 0.332 sqrt(C*) Pr^(-2/3)/sqrt(Re_x),
    T_aw = T_e (1 + sqrt(Pr) (gamma - 1)/2 M^2),  T* = T_e + 0.5 (T_w - T_e) + 0.22 (T_aw - T_e),
    C* = rho* mu*/(rho_e mu_e) = T_e/T*   (constant viscosity).
Run: `ablation_q` on the wall-adjacent fluid cells of the last plot files (time average over the
last third of the run); also the fitted exponent n of q_w ~ x^n over 0.2 < x < 0.9 (theory -1/2)
and the skin friction c_f = 0.664 sqrt(C*)/sqrt(Re_x) from mu du/dy at the wall.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
GAM, CP, MU, PR, RHO, TE, TW, MACH = 1.4, 3.5, 2.0e-4, 0.72, 1.0, 1.0, 1.0, 2.0


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    U = MACH * np.sqrt(GAM * 1.0 / RHO)
    Taw = TE * (1 + np.sqrt(PR) * (GAM - 1) / 2 * MACH**2); Ts = TE + 0.5 * (TW - TE) + 0.22 * (Taw - TE); Cs = TE / Ts
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    use = pfs[-max(1, len(pfs) // 3):]; Q = []; TAU = []; Y0 = []
    for pf in use:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        nx, ny = ds.domain_dimensions[:2]; x = (np.arange(nx) + 0.5) / nx
        y = float(ds.domain_left_edge[1]) + (np.arange(ny) + 0.5) * float(ds.domain_width[1]) / ny
        q = np.asarray(cg[("boxlib", "ablation_q")])[:, :, 0]; phi = np.asarray(cg[("boxlib", "phi")])[:, :, 0]; u = np.asarray(cg[("boxlib", "velocityx")])[:, :, 0]
        j = np.where(phi[nx // 2] >= 0.5)[0][0]                 # first fluid row
        Q.append(q[:, j]); TAU.append(MU * u[:, j] / y[j])      # wall at y = 0, cell centre at y[j] = dy/2
        dy = y[1] - y[0]; Y0.append((y[j] - u[:, j] * dy / np.maximum(u[:, j + 1] - u[:, j], 1e-30)) / dy)   # where the first two fluid cells put u = 0, in cells
    qw = np.mean(Q, axis=0); tau = np.mean(TAU, axis=0); t0, t1 = float(yt.load(use[0]).current_time), float(yt.load(use[-1]).current_time)
    Rex = RHO * U * x / MU
    qref = 0.332 * np.sqrt(Cs) * PR**(-2.0 / 3.0) / np.sqrt(Rex) * RHO * U * CP * (Taw - TW)
    cfref = 0.664 * np.sqrt(Cs) / np.sqrt(Rex); cf = tau / (0.5 * RHO * U * U)
    m = (x > 0.2) & (x < 0.9); n_fit = np.polyfit(np.log(x[m]), np.log(qw[m]), 1)[0]
    print(f"{d}: averaged over t = {t0:.2f} .. {t1:.2f} ({len(use)} plot files); U = {U:.4f}, T_aw = {Taw:.4f}, T* = {Ts:.4f}, C* = {Cs:.4f}; time scatter of q_w {np.std(Q, axis=0)[m].max():.1e}")
    for xx in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9):
        i = np.argmin(np.abs(x - xx))
        print(f"  x = {x[i]:.2f} (Re_x = {Rex[i]:7.0f}): q_w = {qw[i]:.5f}   reference {qref[i]:.5f}   ({100 * (qw[i] / qref[i] - 1):+.1f} %);"
              f"   c_f = {cf[i]:.5f}   reference {cfref[i]:.5f}   ({100 * (cf[i] / cfref[i] - 1):+.1f} %)")
    print(f"  no-slip position implied by the first two fluid cells, 0.2 < x < 0.9: y = {np.mean(np.mean(Y0, axis=0)[m]):+.2f} dy"
          "   (0 = on the wall face; -0.5 = solid cell centre, i.e. solid.visc_mirror = 0, and c_f above is then ~2x too high)")
    print(f"  q_w ~ x^n over 0.2 < x < 0.9: n = {n_fit:.3f}   (theory -0.500);   mean q_w / reference there: {np.mean(qw[m] / qref[m]):.3f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].loglog(x, qw, "k-", label="Hydro2"); ax[0].loglog(x, qref, "C3--", label="laminar flat plate (reference temperature)"); ax[0].set_ylabel(r"wall heat flux $q_w$")
    ax[1].loglog(x, cf, "k-", label="Hydro2"); ax[1].loglog(x, cfref, "C3--", label="reference"); ax[1].set_ylabel(r"skin friction $c_f$")
    for a in ax: a.set_xlabel("x from the leading edge"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8); a.set_xlim(0.02, 1)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "flat_plate.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/flat_plate.png")


if __name__ == "__main__":
    main()
