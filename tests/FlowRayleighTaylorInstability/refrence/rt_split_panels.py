#!/usr/bin/env python3
"""
Rayleigh-Taylor density as sharp|diffuse split images, one panel per time, side by side.
Left half (x < 0.25): sharp THINC reference (rt_reference_hllc.py output); right half: Hydro2 run.
Each half is drawn on its own grid, so the two solutions may have different resolutions.
White line: phi = 0.5 (reference) and eta = 0.5 (Hydro2).

  python3 rt_split_panels.py SHARP.npz HYDRO_PLOTDIR OUT.png [--times 10,14.5]
         [--sharp-label "..."] [--diffuse-label "..."]
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)

LX, LY = 0.5, 1.5
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})


def main():
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    npz, hdir, out = a[0], a[1], a[2]
    times = [float(t) for t in opt("--times", "10,14.5").split(",")]
    slab = opt("--sharp-label", "Sharp"); dlab = opt("--diffuse-label", "Diffuse")
    S = np.load(npz); ts = S["t"]
    pfs = sorted(glob.glob(os.path.join(hdir, "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    tp = np.array([float(yt.load(p).current_time) for p in pfs])

    fig, ax = plt.subplots(1, len(times), figsize=(3.6 * len(times) + 1.6, 9.4), squeeze=False)
    ax = ax[0]
    for A, t in zip(ax, times):
        k = int(np.argmin(np.abs(ts - t))); j = int(np.argmin(np.abs(tp - t)))
        if abs(ts[k] - t) > 1e-3 or abs(tp[j] - t) > 1e-3:
            print(f"WARNING t = {t}: reference frame {ts[k]:.4f}, Hydro2 frame {tp[j]:.4f}")
        # sharp half
        x, y = S["x"], S["y"]; rs = S["rho"][k].T; ph = S["phi"][k].T   # (ny, nx)
        h = x <= LX / 2 + 1e-12; dx = x[1] - x[0]
        im = A.imshow(rs[:, h], origin="lower", extent=(0, x[h][-1] + dx / 2, 0, LY), cmap="viridis",
                      vmin=0.9, vmax=2.1, interpolation="nearest")
        A.contour(x[h], y, ph[:, h], [0.5], colors="w", linewidths=1.0)
        # diffuse half
        ds = yt.load(pfs[j]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        rh = np.asarray(cg[("boxlib", "density")])[:, :, 0].T; eh = np.asarray(cg[("boxlib", "eta")])[:, :, 0].T
        ny, nx = rh.shape; xh = (np.arange(nx) + 0.5) * LX / nx; yh = (np.arange(ny) + 0.5) * LY / ny
        g = xh >= LX / 2 - 1e-12; dxh = LX / nx
        A.imshow(rh[:, g], origin="lower", extent=(xh[g][0] - dxh / 2, LX, 0, LY), cmap="viridis",
                 vmin=0.9, vmax=2.1, interpolation="nearest")
        A.contour(xh[g], yh, eh[:, g], [0.5], colors="w", linewidths=1.0)
        A.axvline(LX / 2, color="k", lw=2.5, zorder=10)
        A.set_xlim(0, LX); A.set_ylim(0, LY); A.set_aspect("equal")
        A.set_xticks([0, 0.25, 0.5]); A.set_title("$t$ = %g s" % t)
        A.set_xlabel("$x$ [m]")
        A.text(LX / 4, -0.075, slab, transform=A.get_xaxis_transform(), ha="center", va="top", fontsize=13, fontweight="bold")
        A.text(3 * LX / 4, -0.075, dlab, transform=A.get_xaxis_transform(), ha="center", va="top", fontsize=13, fontweight="bold")
    ax[0].set_ylabel("$y$ [m]")
    for A in ax[1:]: A.set_yticklabels([])
    fig.subplots_adjust(left=0.10, right=0.84, top=0.90, bottom=0.11, wspace=0.08)
    cax = fig.add_axes([0.87, 0.2, 0.025, 0.66])
    fig.colorbar(im, cax=cax, label=r"Density, $\rho$ [kg/m$^3$]")
    fig.suptitle("Rayleigh-Taylor Instability: Sharp | Diffuse Density", fontsize=16, y=0.975)
    for ext in ("png", "eps"): fig.savefig(os.path.splitext(out)[0] + "." + ext, dpi=170 if ext == "png" else None)
    print("wrote", out)


if __name__ == "__main__":
    main()
