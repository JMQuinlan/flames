#!/usr/bin/env python3
"""
Analysis of tests/FlowMixingCylinder: a rigid cylinder translating left -> right
along the interface between two gases (light y > 0, heavy y < 0) in an
x-periodic channel (solid.moving = 1).

  python3 mixing_cylinder_check.py <run_dir> [--outdir DIR] [--no-gif] [--stride N]

<run_dir> holds output/ (plot files) and output_forces.dat.

There is no closed-form solution; the test checks that the moving solid is
well behaved:
  * mass of each gas in the fluid region, M_k = sum rho_eta_k dA over phi >= 0.5.
    The moving-wall update is NOT conservative (cells uncovered behind the body
    are filled from a neighbour, cells covered ahead of it are overwritten), so
    the drift of M_k is the main quality metric.  The body area is constant, so
    an exact scheme would give zero drift.
  * integral mixing thickness  h = (1/Lx) sum 4 eta (1 - eta) dA  (fluid cells)
  * extent of each gas across y = 0 (5 % / 95 % levels of the x-averaged eta)
  * drag / lift on the cylinder, Cd = Fx / (0.5 rho_mean Uc^2 D)
    (negative Fx = force opposing the +x motion)
  * no NaN, pressure and density stay positive

Writes to <outdir> (default Images/<run_dir name>/ next to this script):
  mixing_history.png   mass drift, mixing thickness, forces
  mixing_panels.png    density at 6 times
  Contours/density_#####.png and Contours/mixing_cylinder.gif
  mixing_summary.csv
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

UC, D, RHO_L, RHO_H = 1.0, 1.0, 25.0, 100.0
FIELDS = ["eta", "rho_eta0", "rho_eta1", "phi", "density", "pressure", "energy_per_vol", "momentumx", "momentumy"]


def load(pf):
    ds = yt.load(pf)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    f = {k: np.asarray(cg[("boxlib", k)])[:, :, 0] for k in FIELDS}
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge)
    nx, ny = ds.domain_dimensions[:2]
    return float(ds.current_time), f, (lo[0], hi[0], lo[1], hi[1]), (hi[0] - lo[0]) / nx, (hi[1] - lo[1]) / ny


def main():
    args = sys.argv[1:]
    d = args[0]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 1
    make_gif = "--no-gif" not in args
    cdir = os.path.join(outdir, "Contours"); os.makedirs(cdir, exist_ok=True)
    pfs = sorted(p for p in glob.glob(os.path.join(d, "output", "*cell")) if re.search(r"\d+cell$", p))
    H = []; frames = []; snaps = {}
    want = np.linspace(0, len(pfs) - 1, 6).round().astype(int)
    for n, pf in enumerate(pfs):
        t, f, ext, dx, dy = load(pf)
        fl = f["phi"] >= 0.5
        dA = dx * dy
        Lx = ext[1] - ext[0]
        y = ext[2] + (np.arange(f["eta"].shape[1]) + 0.5) * dy
        eta = np.clip(f["eta"], 0, 1)
        nfl = np.maximum(fl.sum(0), 1)
        ebar = (eta * fl).sum(0) / nfl                     # x-average over fluid cells
        up = y[ebar < 0.95]; dn = y[ebar > 0.05]           # heavy gas reaching up / light gas reaching down
        H.append((t, (f["rho_eta0"] * fl).sum() * dA, (f["rho_eta1"] * fl).sum() * dA,
                  (f["energy_per_vol"] * fl).sum() * dA, (f["momentumx"] * fl).sum() * dA,
                  (4 * eta * (1 - eta) * fl).sum() * dA / Lx,
                  up.max() if up.size else 0.0, dn.min() if dn.size else 0.0,
                  f["pressure"][fl].min(), f["density"][fl].min(), fl.sum() * dA,
                  int(not np.isfinite(f["density"]).all())))
        if n in want:
            snaps[n] = (t, np.where(fl, f["density"], np.nan), ext)
        if make_gif and n % stride == 0:
            fig, ax = plt.subplots(figsize=(9, 5.4))
            im = ax.imshow(np.where(fl, f["density"], np.nan).T, origin="lower", extent=ext, cmap="viridis",
                           vmin=RHO_L * 0.8, vmax=RHO_H * 1.1, interpolation="nearest")
            ax.contour(f["phi"].T, levels=[0.5], colors="w", linewidths=1.0, extent=ext)
            ax.set_facecolor("0.25"); ax.set_aspect("equal")
            ax.set_xlabel("x / D"); ax.set_ylabel("y / D"); ax.set_title(rf"$t\,U_c/D$ = {t:6.2f}")
            fig.colorbar(im, ax=ax, label=r"$\rho$", shrink=0.85)
            fn = os.path.join(cdir, f"density_{n:05d}.png")
            fig.savefig(fn, dpi=110); plt.close(fig); frames.append(fn)
    H = np.array(H); t = H[:, 0]
    M0, M1 = H[:, 1], H[:, 2]
    dM0, dM1 = 100 * (M0 / M0[0] - 1), 100 * (M1 / M1[0] - 1)

    F = np.loadtxt(os.path.join(d, "output_forces.dat")); F = F[F[:, 1] == F[:, 1].max()]
    q = 0.5 * 0.5 * (RHO_L + RHO_H) * UC * UC * D
    tf, Cd, Cl = F[:, 2], F[:, 3] / q, F[:, 4] / q
    # box-filter the step-to-step force noise of the moving wall over 0.1 D/Uc
    def smooth(a):
        n = max(1, int(0.1 / np.median(np.diff(tf)))); k = np.ones(n) / n
        return np.convolve(a, k, mode="same")
    late = tf > 0.5 * tf[-1]

    rows = [("t_end", t[-1]), ("plotfiles", len(pfs)), ("nan_frames", int(H[:, 11].sum())),
            ("mass_drift_light_pct_end", dM0[-1]), ("mass_drift_light_pct_max", np.abs(dM0).max()),
            ("mass_drift_heavy_pct_end", dM1[-1]), ("mass_drift_heavy_pct_max", np.abs(dM1).max()),
            ("fluid_area_variation_pct", 100 * (H[:, 10].max() - H[:, 10].min()) / H[0, 10]),
            ("mixing_thickness_start", H[0, 5]), ("mixing_thickness_end", H[-1, 5]),
            ("heavy_reach_up_end", H[-1, 6]), ("light_reach_down_end", H[-1, 7]),
            ("p_min", H[:, 8].min()), ("rho_min", H[:, 9].min()),
            ("Cd_mean_late", Cd[late].mean()), ("Cl_mean_late", Cl[late].mean()),
            ("Cd_step_noise_rms", float(np.std(np.diff(Cd[late])) / np.sqrt(2)))]
    with open(os.path.join(outdir, "mixing_summary.csv"), "w") as fo:
        fo.write("metric,value\n")
        for k, v in rows:
            fo.write(f"{k},{v:.6g}\n"); print(f"  {k:30s} {v:.6g}")

    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax[0].plot(t, dM0, label=r"light gas ($y>0$)"); ax[0].plot(t, dM1, label=r"heavy gas ($y<0$)")
    ax[0].set_ylabel("mass drift [%]"); ax[0].legend()
    ax[1].plot(t, H[:, 5], "k", label=r"$h=\frac{1}{L_x}\int 4\eta(1-\eta)\,dA$")
    ax[1].plot(t, H[:, 6], "C3", label="heavy gas reach (+y)"); ax[1].plot(t, -H[:, 7], "C0", label="light gas reach (−y)")
    ax[1].set_ylabel("mixing extent / D"); ax[1].legend(fontsize=9)
    ax[2].plot(tf, Cd, color="0.8", lw=0.4); ax[2].plot(tf, smooth(Cd), "k", lw=0.9); ax[2].set_ylabel(r"$C_d$ ($F_x/q$)")
    ax[3].plot(tf, Cl, color="0.8", lw=0.4); ax[3].plot(tf, smooth(Cl), "C0", lw=0.9); ax[3].set_ylabel(r"$C_l$")
    for a, c in ((ax[2], Cd), (ax[3], Cl)):
        lo, hi = np.percentile(smooth(c)[tf > min(1.0, 0.2 * tf[-1])], [0.5, 99.5]); a.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    ax[3].set_xlabel(r"$t\,U_c/D$")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "mixing_history.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(3, 2, figsize=(13, 12.5), sharex=True, sharey=True)
    for a, n in zip(ax.ravel(), sorted(snaps)):
        ts, rho, ext = snaps[n]
        im = a.imshow(rho.T, origin="lower", extent=ext, cmap="viridis", vmin=RHO_L * 0.8, vmax=RHO_H * 1.1)
        a.set_facecolor("0.25"); a.set_aspect("equal"); a.set_title(rf"$t\,U_c/D$ = {ts:.2f}")
    fig.colorbar(im, ax=ax, label=r"$\rho$", shrink=0.6)
    fig.savefig(os.path.join(outdir, "mixing_panels.png"), dpi=180, bbox_inches="tight"); plt.close(fig)

    if make_gif and frames:
        from PIL import Image
        ims = [Image.open(fn).convert("RGB") for fn in frames]
        # one shared palette so colours do not flicker between frames
        pal = ims[len(ims) // 2].quantize(colors=256, dither=Image.Dither.NONE)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(cdir, "mixing_cylinder.gif"), save_all=True, append_images=ims[1:], duration=67, loop=0)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
