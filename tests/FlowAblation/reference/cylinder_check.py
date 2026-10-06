#!/usr/bin/env python3
"""
Circular body ablating in still hot gas (tests/FlowAblation/input_2D_Cylinder).

  python3 cylinder_check.py <run_dir> [--outdir DIR]

Reference: axisymmetric Stefan problem, integrated numerically (front-fixing transform
xi = (r - R)/(Ro - R), explicit finite differences, 600 points):
    T_t = alpha (T_rr + T_r/r),  T(R, t) = T_w,  T_r(Ro) = 0,  rho_s Q* dR/dt = -k T_r(R).
Run: radius from the solid area, R = sqrt(sum(1 - phi) dA / pi); roundness from the phi = 0.5
contour (rms and peak-to-peak departure of the contour radius, in cells).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
RHO, CP, MU, PR, TINF, TW, RSQ, R0, RO = 1.0, 3.5, 1.0e-3, 0.72, 1.0, 0.9, 1.757, 0.2, 0.5


def reference(tend, N=600):
    k = MU * CP / PR; al = k / (RHO * CP)
    xi = np.linspace(0, 1, N + 1); T = np.full(N + 1, TINF); T[0] = TW; R = R0; t = 0.0; out = [(0.0, R0)]
    while t < tend:
        Lw = RO - R; dr = Lw / N; dt = min(0.2 * dr * dr / al, tend - t)
        r = R + xi * Lw
        Tr = np.zeros_like(T); Trr = np.zeros_like(T)
        Tr[1:-1] = (T[2:] - T[:-2]) / (2 * dr); Trr[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dr**2
        Trr[-1] = 2 * (T[-2] - T[-1]) / dr**2                       # insulated outer boundary
        gw = (-3 * T[0] + 4 * T[1] - T[2]) / (2 * dr)               # wall gradient, 2nd order
        Rd = -k * gw / RSQ
        Tn = T + dt * (al * (Trr + Tr / r) + Tr * Rd * (1 - xi))
        Tn[0] = TW; T = Tn; R += dt * Rd; t += dt; out.append((t, R))
    return np.array(out)


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    H = []; last = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
        phi = np.asarray(cg[("boxlib", "phi")])[:, :, 0]
        Ra = np.sqrt((1 - phi).mean() / np.pi)
        cs = plt.figure().gca().contour(x, x, phi.T, levels=[0.5]); plt.close()
        P = max(cs.allsegs[0], key=len); rc = np.hypot(P[:, 0] - 0.5, P[:, 1] - 0.5)
        H.append((t, Ra, rc.mean(), rc.std() * n, (rc.max() - rc.min()) * n)); last = (t, P)
    H = np.array(H); t = H[:, 0]; ref = reference(t[-1]); Rref = np.interp(t, ref[:, 0], ref[:, 1])
    print(f"{d}: {len(H)} plot files to t = {t[-1]:.2f}; dx = {1.0 / n:.4f}")
    for j in sorted(set(np.linspace(1, len(H) - 1, 4).round().astype(int))):
        print(f"  t = {t[j]:4.2f}: recession {R0 - H[j, 1]:.5f} (area) / {H[0, 2] - H[j, 2]:.5f} (contour), radial reference {R0 - Rref[j]:.5f}"
              f"  ({100 * ((R0 - H[j, 1]) / (R0 - Rref[j]) - 1):+.1f} %);  out-of-round rms {H[j, 3]:.2f} cells, peak-to-peak {H[j, 4]:.2f} cells")
    print(f"  initial out-of-round (staircase): rms {H[0, 3]:.2f}, peak-to-peak {H[0, 4]:.2f} cells")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(t, R0 - H[:, 1], "ko", ms=4, label="Hydro2 (solid area)"); ax[0].plot(ref[:, 0], R0 - ref[:, 1], "C3-", label="axisymmetric Stefan (1D numerical)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("recession of the radius"); ax[0].legend(); ax[0].grid(alpha=0.3)
    th = np.linspace(0, 2 * np.pi, 400); tp, P = last
    ax[1].plot(0.5 + R0 * np.cos(th), 0.5 + R0 * np.sin(th), "0.6", lw=0.8, label="initial"); ax[1].plot(P[:, 0], P[:, 1], "k-", lw=1.2, label=f"Hydro2, t = {tp:.1f}")
    ax[1].plot(0.5 + Rref[-1] * np.cos(th), 0.5 + Rref[-1] * np.sin(th), "C3--", lw=0.8, label="reference radius"); ax[1].set_aspect("equal"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cylinder.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/cylinder.png")


if __name__ == "__main__":
    main()
