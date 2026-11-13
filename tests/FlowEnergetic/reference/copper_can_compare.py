#!/usr/bin/env python3
"""
Flyer-plate ignited copper vessel with a partition (tests/FlowEnergetic/input_2D_CopperCan_Flyer): the frames of
Fig. 4 of Wallis, Barton & Nikiforakis, Comput. Struct. 254 (2021) 106578 next to the same frames of a run.

  python3 copper_can_compare.py <run_dir> --page PAGE.png [--outdir DIR]

PAGE.png is page 10 of the arXiv version (2009.14555v2) rendered at 220 dpi:
   pdftoppm -f 10 -l 10 -r 220 -png 2009.14555v2.pdf page      (the paper is not kept in the repository)
Left column: the paper's panel (cylindrically symmetric; pressure above the axis, density below).  Right column:
this run (PLANAR), same layout, the paper's colour ranges (pressure -8 to 38 GPa, density 0 to 10230 kg/m^3),
with the copper outline (copper volume fraction 0.5) in black.  Times 10, 19.5, 29, 33.5, 54.5 us.
Also prints, per frame: detonation / shock position on the axis, copper extent, masses.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from copper_can_plots import load
# panels of the page image (pixels at 220 dpi): x0, y0, x1, y1 of the plot area [0, 0.4] x [-0.2, 0.2] m
PANELS = [(10.0, (1025, 273, 1447, 693)), (19.5, (475, 730, 896, 1150)), (29.0, (1029, 730, 1450, 1150)), (33.5, (477, 1190, 898, 1611)), (54.5, (1033, 1196, 1454, 1617))]


def main():
    args = sys.argv[1:]; opt = lambda k, d: args[args.index(k) + 1] if k in args else d; d = args[0]
    page = Image.open(opt("--page", "page-10.png")).convert("RGB"); outdir = opt("--outdir", os.path.join(HERE, "Images", "CopperCanFlyer")); os.makedirs(outdir, exist_ok=True)
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])); tt = []
    for pf in pfs:
        with open(os.path.join(pf, "Header")) as fh: L = fh.read().split("\n"); tt.append(float(L[2 + int(L[1]) + 1]))
    tt = np.array(tt); F0 = load(pfs[0]); fig, ax = plt.subplots(len(PANELS), 2, figsize=(11.5, 5.3 * len(PANELS)))
    print("   t [us]   wave on the axis [cm]   copper x from .. to [cm], largest y [cm]   explosive left   copper mass - 1   p max [GPa]")
    for (tq, box), (a, b) in zip(PANELS, ax):
        a.imshow(np.asarray(page.crop(box)), extent=(0, 40, -20, 20), aspect="equal"); a.set_title(r"(Wallis 2021), Cylindrically Symmetric, t = %g $\mu$s" % tq, fontsize=10)
        f = load(pfs[int(np.argmin(np.abs(tt - 1e-6 * tq)))]); nx, ny = f["p"].shape; x = (np.arange(nx) + 0.5) * 100 * f["Lx"] / nx; y = (np.arange(ny) + 0.5) * 100 * f["Ly"] / ny
        im1 = b.imshow(f["p"].T / 1e9, origin="lower", extent=(0, 40, 0, 20), cmap="jet", vmin=-8, vmax=38, aspect="equal")
        im2 = b.imshow(f["rho"].T[::-1], origin="lower", extent=(0, 40, -20, 0), cmap="jet", vmin=0, vmax=10230, aspect="equal")
        b.contour(x, y, f["acu"].T, [0.5], colors="k", linewidths=0.6); b.contour(x, -y, f["acu"].T, [0.5], colors="k", linewidths=0.6)
        b.set_xlim(0, 40); b.set_ylim(-20, 20); b.set_title(r"Numerical, Planar, t = %.1f $\mu$s" % (1e6 * f["t"]), fontsize=10)
        fig.colorbar(im1, ax=b, fraction=0.04, pad=0.02, label="Pressure [GPa] (Upper Half)"); fig.colorbar(im2, ax=b, fraction=0.04, pad=0.05, location="left", label=r"Density [kg/m$^3$] (Lower Half)")
        a.set_xlabel("z [cm]"); a.set_ylabel("r [cm]"); b.set_xlabel("x [cm]"); b.set_yticklabels([])
        c = f["acu"] > 0.5; xi = np.where(c.any(axis=1))[0]; yi = np.where(c.any(axis=0))[0]; w = np.where(f["p"][:, 2] > 1e9)[0]
        print("  %6.1f        %s              %.2f .. %.2f, %.2f                    %.4f         %+.2e       %.2f"
              % (1e6 * f["t"], ("%.2f" % x[w.max()]) if len(w) else "  -  ", x[xi.min()], x[xi.max()], y[yi.max()], f["mr"].sum() / F0["mr"].sum(), f["mcu"].sum() / F0["mcu"].sum() - 1, f["p"].max() / 1e9))
    fig.suptitle("Flyer-Plate Ignited Explosive-Filled Copper Vessel: (Wallis 2021) Fig. 4 and Numerical", y=0.995); fig.tight_layout()
    o = os.path.join(outdir, "copper_can_flyer_compare.png"); fig.savefig(o, dpi=150); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
