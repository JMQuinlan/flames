#!/usr/bin/env python3
"""
Rigid rotation of a stress-free elastic disc (tests/FlowElasticSolid/input_2D_RotatingDisc).

  python3 rotating_disc_check.py <run_dir> [--outdir DIR]

Exact kinematics for a rotation by theta = Omega t:
    e^1 = (cos theta, sin theta), e^2 = (-sin theta, cos theta)   ->   G = I, S = 0.
The disc does not keep Omega exactly: the solid | gas interface diffuses outward (and the
rim drags gas along), so its moment of inertia grows and it slows while the TOTAL angular
momentum is conserved.  The kinematic statement tested is therefore
    cobasis angle = integral of the measured material angular velocity.
Measured in the core of the disc (r < 0.7 R, alpha_s > 0.99):
  * rotation angle of the cobasis, atan2(e^1_y - e^2_x, e^1_x + e^2_y)   vs  int w dt  (and Omega t)
  * total angular momentum sum rho (x v - y u) dA of the whole domain
  * angular velocity of the material, least-squares fit of u = -w (y - yc), v = w (x - xc)
  * the stretch part: singular values of the cobasis (1 for a rigid motion)
  * deviatoric stress against the centrifugal scale rho_s Omega^2 R^2
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
OM, R, RHOS, MU, XC = 0.5, 0.25, 10.0, 50.0, 0.5


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    H = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*cell")))[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x, indexing="ij")
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
        t = float(ds.current_time)
        m0 = q("rho_eta0"); M = m0.sum(); xc, yc = (m0 * X).sum() / M, (m0 * Y).sum() / M
        core = (np.hypot(X - xc, Y - yc) < 0.7 * R) & (q("eta") > 0.99)
        A1, A2, B1, B2 = q("ebasis1x")[core], q("ebasis1y")[core], q("ebasis2x")[core], q("ebasis2y")[core]
        th = np.arctan2(A2 - B1, A1 + B2)
        # singular values of [[A1, A2], [B1, B2]]
        a_, b_ = 0.5 * np.hypot(A1 + B2, A2 - B1), 0.5 * np.hypot(A1 - B2, A2 + B1)
        s1, s2 = a_ + b_, a_ - b_
        u, v = q("velocityx")[core], q("velocityy")[core]; dx_, dy_ = X[core] - xc, Y[core] - yc
        w = (v * dx_ - u * dy_).sum() / (dx_**2 + dy_**2).sum()
        S = np.sqrt(q("elastic_Sxx")**2 + q("elastic_Syy")**2 + 2 * q("elastic_Sxy")**2)[core]
        H.append((t, th.mean(), th.std(), w, s1.max() - 1, s2.min() - 1, S.mean(), S.max(), np.hypot(xc - XC, yc - XC), M / 1.0,
                  (q("density") * ((X - XC) * q("velocityy") - (Y - XC) * q("velocityx"))).sum() / n**2))
    H = np.array(H); t = H[:, 0]
    # material rotation = int w dt (w starts at OM at t = 0; trapezoid on the plot times)
    tt, ww = np.concatenate([[0.0], t]), np.concatenate([[OM], H[:, 3]])
    phi = np.cumsum(0.5 * (ww[1:] + ww[:-1]) * np.diff(tt))
    scale = RHOS * OM**2 * R**2
    print(f"{d}: {len(H)} plot files to t = {t[-1]:.3f} (theta = {np.degrees(OM * t[-1]):.1f} deg); centrifugal stress scale {scale:.3f}")
    print(f"  cobasis rotation angle   {np.degrees(H[-1, 1]):9.4f} deg   exact {np.degrees(OM * t[-1]):9.4f}   ({100 * (H[-1, 1] / (OM * t[-1]) - 1):+.3f} %),"
          f" spread over the core {np.degrees(H[-1, 2]):.4f} deg")
    print(f"  cobasis angle vs material rotation int w dt = {np.degrees(phi[-1]):.4f} deg: error {np.degrees(H[-1, 1] - phi[-1]):+.4f} deg"
          f" (max over the run {np.degrees(np.abs(H[:, 1] - phi).max()):.4f} deg)")
    print(f"  total angular momentum {H[0, 10]:.6f} -> {H[-1, 10]:.6f}  ({100 * (H[-1, 10] / H[0, 10] - 1):+.2f} %)")
    print(f"  material angular velocity: mean {H[:, 3].mean():.5f}, final {H[-1, 3]:.5f}   (initial {OM})")
    print(f"  stretch (singular values - 1): max {H[:, 4].max():+.2e}, min {H[:, 5].min():+.2e}   (rigid: 0; centrifugal strain ~ {scale / MU:.1e})")
    print(f"  deviatoric stress in the core: mean {H[:, 6].mean():.4f}, max {H[:, 7].max():.4f}   = {H[:, 6].mean() / scale:.2f} / {H[:, 7].max() / scale:.2f} x rho Om^2 R^2")
    print(f"  centroid drift {H[:, 8].max():.2e}, solid mass drift {100 * (H[-1, 9] / H[0, 9] - 1):+.4f} %")
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(t, np.degrees(H[:, 1]), "ko", ms=3, label="cobasis"); ax[0].plot(t, np.degrees(OM * t), "C3-", label=r"$\Omega t$"); ax[0].plot(t, np.degrees(phi), "C0--", label=r"$\int\omega\,dt$ (measured)")
    ax[0].set_ylabel("rotation angle [deg]"); ax[0].legend()
    ax[1].plot(t, H[:, 3], "k-"); ax[1].axhline(OM, color="C3", ls="--"); ax[1].set_ylabel("material angular velocity")
    ax[2].plot(t, H[:, 6] / scale, "k-", label="mean"); ax[2].plot(t, H[:, 7] / scale, "C0-", label="max")
    ax[2].set_ylabel(r"$|S|$ in the core / $\rho_s\Omega^2R^2$"); ax[2].legend()
    for a in ax: a.set_xlabel("t"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "rotating_disc.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/rotating_disc.png")


if __name__ == "__main__":
    main()
