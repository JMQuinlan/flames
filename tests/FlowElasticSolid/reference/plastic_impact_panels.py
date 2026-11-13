#!/usr/bin/env python3
"""
Copper projectile on a copper plate (input_2D_PlateImpact_Copper[_Plastic]): numerical schlieren and von Mises
stress at chosen times, optionally under the frames of a published figure.

  python3 plastic_impact_panels.py <run_dir> --times 0.1,0.25,0.6 --out FIG.png [--lit PAGE.png --crops x0,y0,x1,y1;...]
                                   [--lit-label "(Favrie and Gavrilyuk 2011)"] [--title "..."] [--vmax 3] [--eta-mask 0.5]

Units of the decks: length m, time ms, stress GPa.  Von Mises stress sigma_eq = sqrt(3/2 S:S) from the deviator
S of the plot file (elastic_Sxx ... elastic_Szz), shown where the solid volume fraction eta > --eta-mask (default 0.5); one colour bar
for the whole row.  The eta = 0.5 outline is drawn on every numerical panel.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)


def frames(d):
    pfs = sorted(glob.glob(os.path.join(d, "*cell")) + glob.glob(os.path.join(d, "output", "*cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    return pfs


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    g = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0].T
    S = {k: g("elastic_S" + k) for k in ("xx", "yy", "zz", "xy", "xz", "yz")}
    vm = np.sqrt(1.5 * (S["xx"] ** 2 + S["yy"] ** 2 + S["zz"] ** 2 + 2 * (S["xy"] ** 2 + S["xz"] ** 2 + S["yz"] ** 2)))
    L = ds.domain_right_edge.v[:2]; lo = ds.domain_left_edge.v[:2]
    return dict(t=float(ds.current_time), rho=g("density"), eta=g("eta"), vm=vm, ext=(lo[0], L[0], lo[1], L[1]))


def main():
    a = sys.argv[1:]; opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    run = a[0]; times = [float(t) for t in opt("--times", "0.1,0.25,0.6").split(",")]; out = opt("--out", "plastic_impact_panels.png")
    lit = opt("--lit", None); crops = [tuple(int(v) for v in c.split(",")) for c in opt("--crops", "").split(";") if c]
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 13})
    pfs = frames(run); tt = [float(yt.load(p).current_time) for p in pfs]
    nrow = 3 if lit else 2; n = len(times)
    fig, ax = plt.subplots(nrow, n, figsize=(4.4 * n, 4.4 * nrow + 0.9), squeeze=False)
    r0 = 0
    if lit:
        img = plt.imread(lit)
        for j, c in enumerate(crops[:n]):
            ax[0, j].imshow(img[c[1]:c[3], c[0]:c[2]]); ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
            ax[0, j].set_title("%s, $t$ = %g $\\mu$s" % (opt("--lit-label", "Literature"), 1e3 * times[j]))
        r0 = 1
    vmax = float(opt("--vmax", "0")) or None; ims = []; em = float(opt("--eta-mask", "0.5"))
    for j, t in enumerate(times):
        k = int(np.argmin([abs(x - t) for x in tt])); f = load(pfs[k])
        gy, gx = np.gradient(f["rho"]); G = np.hypot(gx, gy); sch = np.exp(-4.0 * G / max(np.percentile(G, 99.0), 1e-30))
        ax[r0, j].imshow(sch, origin="lower", extent=f["ext"], cmap="gray", vmin=0, vmax=1)
        ax[r0, j].set_title("Numerical Schlieren, $t$ = %.0f $\\mu$s" % (1e3 * f["t"]))
        vm = np.ma.masked_where(f["eta"] < em, f["vm"])
        if vmax is None: vmax = float(np.nanpercentile(f["vm"][f["eta"] > 0.5], 99.5)) if (f["eta"] > 0.5).any() else 1.0
        ims.append(ax[r0 + 1, j].imshow(vm, origin="lower", extent=f["ext"], cmap="viridis", vmin=0, vmax=vmax))
        ax[r0 + 1, j].set_title("Von Mises Stress, $t$ = %.0f $\\mu$s" % (1e3 * f["t"]))
        for r in (r0, r0 + 1):
            ax[r, j].contour(np.linspace(f["ext"][0], f["ext"][1], f["eta"].shape[1]), np.linspace(f["ext"][2], f["ext"][3], f["eta"].shape[0]), f["eta"], [0.5 if r == r0 else em], colors="k" if r == r0 + 1 else "C3", linewidths=0.8)
            ax[r, j].set_xlabel("$x$ [m]"); ax[r, j].set_ylabel("$y$ [m]")
        print("t = %.4f ms: max von Mises in the solid %.3f GPa" % (f["t"], float(f["vm"][f["eta"] > 0.5].max())))
    fig.suptitle(opt("--title", "Copper Projectile on a Copper Plate at 800 m/s"))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    cax = fig.add_axes([0.25, 0.045, 0.5, 0.016]); cb = fig.colorbar(ims[-1], cax=cax, orientation="horizontal"); cb.set_label(r"Von Mises Stress, $\sigma_{eq}$ [GPa]")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out, dpi=170); print("wrote", out)


if __name__ == "__main__":
    main()
