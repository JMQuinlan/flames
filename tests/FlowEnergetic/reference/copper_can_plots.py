#!/usr/bin/env python3
"""
Explosive-filled copper vessel, 2D planar (tests/FlowEnergetic/input_2D_CopperCan; after test 4.3 of Wallis,
Barton & Nikiforakis 2021, their Fig. 4): six times, each panel with the pressure in the upper half and the mixture
density in the lower half (mirror image), and the outline of the copper (copper volume fraction 0.5).

  python3 copper_can_plots.py <run_dir> [--outdir DIR] [--times 5,10,20,30,40,55]      (microseconds)

Copper volume fraction = eta x reference-volume share of the species Cu in phase 0.  Also printed per plotted time:
copper mass, total mass, unreacted explosive, the largest pressure and the extent of the copper.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
HERE = os.path.dirname(os.path.abspath(__file__))
RCU, RRE = 8930.0, 1600.0


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); g = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0]
    vc, vr = g("species_Cu") / RCU, g("species_Reactant") / RRE; eta = np.clip(g("eta"), 0, 1)
    return dict(t=float(ds.current_time), Lx=float(ds.domain_right_edge[0]), Ly=float(ds.domain_right_edge[1]), p=g("pressure"), rho=g("density"),
                acu=eta * np.where(vc + vr > 0, vc / np.maximum(vc + vr, 1e-300), 0.0), mcu=g("species_Cu"), mr=g("species_Reactant"))


def main():
    args = sys.argv[1:]; opt = lambda k, d: args[args.index(k) + 1] if k in args else d; d = args[0]
    outdir = opt("--outdir", os.path.join(HERE, "Images", "CopperCan")); os.makedirs(outdir, exist_ok=True)
    times = [float(v) for v in opt("--times", "5,10,20,30,40,55").split(",")]
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    tt = []
    for pf in pfs:
        with open(os.path.join(pf, "Header")) as fh: L = fh.read().split("\n"); tt.append(float(L[2 + int(L[1]) + 1]))
    tt = np.array(tt); F0 = load(pfs[0]); fig, ax = plt.subplots(3, 2, figsize=(15.5, 14.5))
    print("   t [us]   copper mass/initial - 1   total mass/initial - 1   explosive left   p max [Pa]    copper: x from .. to [cm], largest y [cm]")
    for a, tq in zip(ax.ravel(), times):
        f = load(pfs[int(np.argmin(np.abs(tt - 1e-6 * tq)))]); ext = (0, 100 * f["Lx"], 0, 100 * f["Ly"])
        im1 = a.imshow(f["p"].T / 1e9, origin="lower", extent=ext, cmap="turbo", vmin=0, vmax=30, aspect="equal")
        im2 = a.imshow(f["rho"].T[::-1], origin="lower", extent=(0, 100 * f["Lx"], -100 * f["Ly"], 0), cmap="viridis", vmin=0, vmax=10000, aspect="equal")
        nx, ny = f["acu"].shape; x = (np.arange(nx) + 0.5) * 100 * f["Lx"] / nx; y = (np.arange(ny) + 0.5) * 100 * f["Ly"] / ny
        a.contour(x, y, f["acu"].T, [0.5], colors="w", linewidths=0.6); a.contour(x, -y, f["acu"].T, [0.5], colors="w", linewidths=0.6)
        a.set_xlim(0, 100 * f["Lx"]); a.set_ylim(-100 * f["Ly"], 100 * f["Ly"]); a.axhline(0, color="w", lw=0.4)
        a.set_title(r"t = %.1f $\mu$s" % (1e6 * f["t"])); a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]")
        fig.colorbar(im1, ax=a, fraction=0.04, pad=0.02, location="right", label="Pressure [GPa] (Upper Half)")
        fig.colorbar(im2, ax=a, fraction=0.04, pad=0.14, location="left", label=r"Density [kg/m$^3$] (Lower Half)")
        c = f["acu"] > 0.5; xi = np.where(c.any(axis=1))[0]; yi = np.where(c.any(axis=0))[0]
        print("  %6.1f      %+.3e              %+.3e            %.4f        %.3e      %.2f .. %.2f, %.2f"
              % (1e6 * f["t"], f["mcu"].sum() / F0["mcu"].sum() - 1, f["rho"].sum() / F0["rho"].sum() - 1, f["mr"].sum() / F0["mr"].sum(), f["p"].max(), x[xi.min()], x[xi.max()], y[yi.max()]))
    fig.suptitle("Explosive-Filled Copper Vessel, Booster Ignited, Planar (Numerical; Setup After Wallis 2021)\nWhite Line: Copper Outline"); fig.tight_layout()
    o = os.path.join(outdir, "copper_can.png"); fig.savefig(o, dpi=170); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
