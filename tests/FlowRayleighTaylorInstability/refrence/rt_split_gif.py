#!/usr/bin/env python3
"""
Left/right split density animation of the Rayleigh-Taylor test, in the layout of
rayleightaylor_UNIT_TEST.py (plot_density_comparison): sharp reference on x < 0.25, diffuse
Hydro2 run on x > 0.25, black line at the split.

  python3 rt_split_gif.py SHARP.npz HYDRO_PLOTDIR OUT.gif [--fps 15] [--tmax 20.4]

SHARP.npz is an output of rt_reference_hllc.py (keys t, rho of shape (nt, nx, ny)).
Frames are the Hydro2 plot files; each is paired with the reference frame nearest in time.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt
yt.funcs.mylog.setLevel(50)

LX, LY = 0.5, 1.5
plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13})


def main():
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    npz, hdir, out = a[0], a[1], a[2]
    fps, tmax = int(opt("--fps", "15")), float(opt("--tmax", "1e30"))
    S = np.load(npz); ts, rs = S["t"], S["rho"]
    pfs = sorted(glob.glob(os.path.join(hdir, "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    pfs = [p for p in pfs if float(yt.load(p).current_time) <= tmax + 1e-9]
    # drop the duplicate step-0 file if two files share t = 0
    if len(pfs) > 1 and float(yt.load(pfs[1]).current_time) < 1e-12: pfs = pfs[1:]

    def hydro(p):
        ds = yt.load(p); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        return float(ds.current_time), np.asarray(cg[("boxlib", "density")])[:, :, 0].T

    fig, ax = plt.subplots(figsize=(6.2, 9.6))
    fig.subplots_adjust(left=0.16, right=0.80, top=0.94, bottom=0.11)
    cax = fig.add_axes([0.83, 0.18, 0.03, 0.66])
    t0, h0 = hydro(pfs[0]); nx = h0.shape[1]; half = nx // 2
    im = ax.imshow(np.zeros_like(h0), origin="lower", extent=(0, LX, 0, LY), cmap="viridis", vmin=0.9, vmax=2.1, interpolation="nearest")
    fig.colorbar(im, cax=cax, label=r"Density, $\rho$ [kg/m$^3$]")
    ax.axvline(LX / 2, color="k", lw=2.5, zorder=10)
    ax.text(LX / 4, -0.08, "Sharp", ha="center", va="top", fontsize=16, fontweight="bold")
    ax.text(3 * LX / 4, -0.08, "Diffuse", ha="center", va="top", fontsize=16, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("$y$ [m]"); ax.set_xticks([0, 0.25, 0.5])
    ttl = ax.set_title("")

    def draw(i):
        t, h = hydro(pfs[i]); k = int(np.argmin(np.abs(ts - t)))
        img = h.copy(); img[:, :half] = rs[k].T[:, :half]
        im.set_data(img); ttl.set_text("Rayleigh-Taylor Instability, $t$ = %.1f s" % t)
        return (im,)

    anim = FuncAnimation(fig, draw, frames=len(pfs), interval=1000 // fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=80); print("wrote", out, len(pfs), "frames")
    draw(len(pfs) - 1); fig.savefig(out.replace(".gif", "_final.png"), dpi=150); print("wrote", out.replace(".gif", "_final.png"))


if __name__ == "__main__":
    main()
