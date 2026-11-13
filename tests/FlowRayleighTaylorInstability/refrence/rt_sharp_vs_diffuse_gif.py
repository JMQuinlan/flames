#!/usr/bin/env python3
"""
Side-by-side density animation of the Rayleigh-Taylor test: sharp-interface single-fluid references
(rt_reference_hllc.py output .npz) against the diffuse-interface Hydro2 run (plotfiles).

  python3 rt_sharp_vs_diffuse_gif.py OUT.gif "Label=file.npz" ... "Label=hydro:PLOTDIR" ... [--fps 6]

The 0.5 contour of the colour function (phi for the references, eta for Hydro2) is drawn on each panel.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt
yt.funcs.mylog.setLevel(50)

plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})


def load_npz(f):
    d = np.load(f)
    return dict(t=d["t"], frames=lambda k: (d["rho"][k].T, d["phi"][k].T), ext=(0, 0.5, 0, 1.5))


def load_hydro(dr):
    pfs = sorted(glob.glob(os.path.join(dr, "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    ts = np.array([float(yt.load(p).current_time) for p in pfs])
    def fr(k):
        ds = yt.load(pfs[k]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        g = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0].T
        return g("density"), g("eta")
    ds = yt.load(pfs[0]); lo, hi = ds.domain_left_edge.v, ds.domain_right_edge.v
    return dict(t=ts, frames=fr, ext=(lo[0], hi[0], lo[1], hi[1]))


def main():
    a = sys.argv[1:]; out = a[0]; fps = int(a[a.index("--fps") + 1]) if "--fps" in a else 6
    runs = []
    for s in a[1:]:
        if "=" not in s: continue
        lab, src = s.split("=", 1)
        runs.append((lab, load_hydro(src[6:]) if src.startswith("hydro:") else load_npz(src)))
    times = runs[0][1]["t"]
    fig, ax = plt.subplots(1, len(runs), figsize=(3.3 * len(runs) + 1.2, 8.2), squeeze=False); ax = ax[0]
    fig.subplots_adjust(left=0.05, right=0.88, top=0.88, bottom=0.07, wspace=0.35)
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
    sup = fig.suptitle("")

    def draw(i):
        t = times[i]
        for a_, (lab, r) in zip(ax, runs):
            a_.cla(); k = int(np.argmin(np.abs(r["t"] - t))); rho, c = r["frames"](k)
            im = a_.imshow(rho, origin="lower", extent=r["ext"], cmap="viridis", vmin=0.9, vmax=2.1, interpolation="nearest")
            ny, nx = rho.shape
            a_.contour(np.linspace(r["ext"][0], r["ext"][1], nx), np.linspace(r["ext"][2], r["ext"][3], ny), c, [0.5], colors="w", linewidths=0.8)
            a_.set_title(lab); a_.set_xlabel("$x$")
            a_.set_xticks([0, 0.25, 0.5])
        ax[0].set_ylabel("$y$")
        fig.colorbar(im, cax=cax, label=r"Density, $\rho$")
        sup.set_text("Rayleigh-Taylor Instability, Sharp vs Diffuse Interface, $t$ = %.1f" % t)

    anim = FuncAnimation(fig, draw, frames=len(times), interval=1000 // fps)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=90); print("wrote", out)
    draw(len(times) - 1); fig.savefig(out.replace(".gif", "_final.png"), dpi=150); print("wrote", out.replace(".gif", "_final.png"))


if __name__ == "__main__":
    main()
