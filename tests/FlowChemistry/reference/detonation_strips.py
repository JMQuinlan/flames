#!/usr/bin/env python3
"""
Colour-map animations of the detonation tube (tests/FlowChemistry/input_1D_H2O2Detonation_ZND): every quantity is
drawn as a strip, the tube seen from the side and coloured by that quantity.

  python3 detonation_strips.py <run_dir> [--outdir DIR] [--zoom 30] [--every 1]

Writes two GIFs (and a still of the middle frame of each):
  h2o2_tube_species.gif    mass fraction of every species of the mixture that is present, one strip per species
  h2o2_tube_p_T.gif        pressure and temperature
Left column: the whole tube.  Right column: the reaction zone, in the frame of the front (the front is at 0 and the
flow behind it to the left), where the induction zone and the radicals can be seen.  Row 0 of the two-row strip.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt
yt.funcs.mylog.setLevel(50)
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = dict(H2=r"H$_2$", O2=r"O$_2$", H2O=r"H$_2$O", OH="OH", H="H", O="O", HO2=r"HO$_2$", H2O2=r"H$_2$O$_2$", AR="Ar", N2=r"N$_2$", HE="He", CO="CO", CO2=r"CO$_2$")
ORDER = ["H2", "O2", "H2O", "OH", "H", "O", "HO2", "H2O2", "AR", "N2", "HE", "CO", "CO2"]


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def load(pf, species):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    q = lambda n: np.asarray(cg[("boxlib", n)])[:, 0, 0]
    n = int(ds.domain_dimensions[0]); L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n
    d = dict(t=float(ds.current_time), x=x, L=L, p=q("pressure") / 1e3, T=q("T"))
    for s in species: d[s] = q("Y_" + s)
    k = np.where(d["p"] > 0.5 * d["p"].max())[0].max(); d["xf"] = x[k]
    return d


def animate(F, rows, zoom, title, out, cells):
    """rows: [(key, label, cmap, vmin, vmax)]"""
    nr = len(rows); L = F[0]["L"]
    fig, ax = plt.subplots(nr, 2, figsize=(12.5, 0.62 * nr + 1.75), gridspec_kw=dict(width_ratios=[1.55, 1.0]), squeeze=False)
    fig.subplots_adjust(left=0.085, right=0.915, bottom=0.62 / (0.62 * nr + 1.75), top=1.0 - 1.02 / (0.62 * nr + 1.75), hspace=0.42, wspace=0.26)
    ims = []
    sgrid = np.linspace(-0.1 * zoom, zoom, 700)          # mm behind the front
    for r, (key, lab, cm, v0, v1) in enumerate(rows):
        a, b = ax[r]
        im1 = a.imshow(F[0][key][None, :], extent=(0, L, 0, 1), aspect="auto", cmap=cm, vmin=v0, vmax=v1, interpolation="nearest")
        im2 = b.imshow(np.zeros((1, len(sgrid))), extent=(zoom, -0.1 * zoom, 0, 1), aspect="auto", cmap=cm, vmin=v0, vmax=v1, interpolation="bilinear")
        for c in (a, b): c.set_yticks([])
        a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        if r < nr - 1: a.set_xticklabels([]); b.set_xticklabels([])
        cb = fig.colorbar(im2, ax=[a, b], fraction=0.018, pad=0.012); cb.ax.tick_params(labelsize=7)
        ims.append((im1, im2, a.axvline(F[0]["xf"], color="w", lw=0.6, ls=":"), b.axvline(0.0, color="w", lw=0.6, ls=":")))
    ax[-1, 0].set_xlabel("Position Along the Tube [m]"); ax[-1, 1].set_xlabel("Distance Behind the Front [mm]")
    ax[0, 0].set_title("Whole Tube", fontsize=10); ax[0, 1].set_title("Reaction Zone, in the Frame of the Front", fontsize=10)
    sup = fig.suptitle("", fontsize=11)

    def draw(i):
        f = F[i]; s = 1e3 * (f["xf"] - f["x"])               # increasing to the left
        for (key, lab, cm, v0, v1), (im1, im2, l1, l2) in zip(rows, ims):
            im1.set_data(f[key][None, :]); l1.set_xdata([f["xf"], f["xf"]])
            im2.set_data(np.interp(sgrid, s[::-1], f[key][::-1])[None, ::-1])
        sup.set_text(title + f" ({cells:.0f} Cells per Induction Length)\n" + rf"t = {1e6 * f['t']:.0f} $\mu$s,  Front at {f['xf']:.3f} m (Numerical)")
        return []

    anim = FuncAnimation(fig, draw, frames=len(F), interval=160)
    anim.save(out, writer=PillowWriter(fps=6), dpi=110); print("wrote", out)
    draw(len(F) // 2); o = out.replace(".gif", "_frame.png"); fig.savefig(o, dpi=200); print("wrote", o); plt.close(fig)


def main():
    args = sys.argv[1:]; flags = ("--outdir", "--zoom", "--every")
    skip = {args.index(k) + 1 for k in flags if k in args}
    d = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip][0]
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "H2O2_Detonation_Movie")); os.makedirs(outdir, exist_ok=True)
    zoom = float(opt(args, "--zoom", "30")); every = int(opt(args, "--every", "1"))
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[::every]
    ds = yt.load(pfs[0]); have = [f[1][2:] for f in ds.field_list if f[1].startswith("Y_")]
    species = [s for s in ORDER if s in have]
    F = [load(pf, species) for pf in pfs]
    if len(F) > 1 and abs(F[1]["t"] - F[0]["t"]) < 1e-12: F = F[1:]
    cells = 1.521e-3 / (F[0]["x"][1] - F[0]["x"][0])
    present = [s for s in species if max(f[s].max() for f in F) > 1e-7]           # leave out the species of the mechanism that never appear
    rows = [(s, LAB.get(s, s), "viridis", 0.0, max(f[s].max() for f in F)) for s in present]
    mix = r"2H$_2$ + O$_2$ + 7Ar at 6.67 kPa, 298 K"
    animate(F, rows, zoom, "Species Mass Fractions [-] in the Detonation Tube, " + mix, os.path.join(outdir, "h2o2_tube_species.gif"), cells)
    rows = [("p", "Pressure\n[kPa]", "inferno", 0.0, max(f["p"].max() for f in F)), ("T", "Temperature\n[K]", "magma", 0.0, max(f["T"].max() for f in F))]
    animate(F, rows, zoom, "Pressure and Temperature in the Detonation Tube, " + mix, os.path.join(outdir, "h2o2_tube_p_T.gif"), cells)
    print("species shown:", present, "; not present (left out):", [s for s in species if s not in present])


if __name__ == "__main__":
    main()
