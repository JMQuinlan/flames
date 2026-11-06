#!/usr/bin/env python3
"""Normalised radius R/R0 vs time: no-closure (cst) vs split scheme.

Same axes on both panels so the comparison is honest -- the split panel really
is that flat.  Colour is R0 (continuous, ordered -> one perceptually-uniform
ramp + colourbar).  Crosses mark runs that diverged before 15 ms.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
import yt
yt.set_log_level(50)

B = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant"
OUT = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
TAGS = ["140.0", "165.0", "180.0", "185.0", "190.0",
        "200.0", "210.0", "220.0", "240.0", "280.0"]
INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 10,
    "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "figure.facecolor": "white",
    "savefig.facecolor": "white"})


def series(d):
    t, R = [], []
    for f in sorted(glob.glob(d + "/*cell"))[1:]:
        ds = yt.load(f); ad = ds.all_data()
        eta = np.asarray(ad["boxlib", "eta"]); vol = np.asarray(ad["index", "cell_volume"])
        t.append(float(ds.current_time)); R.append(np.sqrt(((1 - eta) * vol).sum() / np.pi))
    return np.array(t), np.array(R)


data = {}
for tag in TAGS:
    R0 = float(tag) * 1e-4
    data[tag] = {"R0": R0,
                 "cst": series(f"{B}/ctl2/cst_{tag}"),
                 "split": series(f"{B}/sweep_split/cc_{tag}")}
    print(f"  {tag}: cst n={len(data[tag]['cst'][0])}  split n={len(data[tag]['split'][0])}")

R0s = np.array([data[t]["R0"] for t in TAGS])
norm = mcolors.Normalize(R0s.min(), R0s.max())
cmap = plt.get_cmap("viridis")

fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(14.6, 4.5))
fig.subplots_adjust(left=0.058, right=0.885, top=0.90, bottom=0.135, wspace=0.20)
a1.sharey(a0)

for a in (a0, a1, a2):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.axhline(1.0, color=AXIS, lw=1.0, ls="--", zorder=1)
    a.set_xlabel("Time  (ms)"); a.set_xlim(0, 15)

for key, ax, title in (("cst", a0, "(a)  no closure"),
                       ("split", a1, "(b)  split scheme  $L_{relax}L_{cap}L_{hyper}$")):
    for tag in TAGS:
        d = data[tag]; t, R = d[key]
        c = cmap(norm(d["R0"]))
        ax.plot(t * 1e3, R / R[0], lw=1.6, color=c, solid_capstyle="round", zorder=3)
        if t[-1] < 1.4e-2:      # diverged before the horizon
            ax.plot(t[-1] * 1e3, R[-1] / R[0], "X", ms=11, color=c,
                    mec="white", mew=1.3, zorder=6)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)

a0.set_ylabel(r"Normalised radius  $R/R_0$")
a0.plot([], [], "X", ms=9, color=AXIS, mec="white", mew=1.0,
        label="diverged before 15 ms")
a0.legend(frameon=False, loc="upper left")

# Third panel: the split scheme on its OWN scale, so the residual structure is
# visible rather than hidden by the shared axis.  Same data as (b).
for tag in TAGS:
    d = data[tag]; t, R = d["split"]
    a2.plot(t * 1e3, R / R[0], lw=1.6, color=cmap(norm(d["R0"])),
            solid_capstyle="round", zorder=3)
a2.set_title("(c)  split scheme, own scale", loc="left", fontsize=10.5, color=INK)
a2.set_ylabel(r"$R/R_0$")
mx = max(abs(data[t]["split"][1] / data[t]["split"][1][0] - 1).max() for t in TAGS)
a2.set_ylim(0.998, 1.0 + 1.15 * mx)
a0.text(0.97, 0.04, f"peak {100*max(abs(data[t]['cst'][1]/data[t]['cst'][1][0]-1).max() for t in TAGS):.0f}%",
        transform=a0.transAxes, ha="right", fontsize=9.5, color=INK)
a1.text(0.97, 0.04, f"peak {100*mx:.2f}%  (see (c))", transform=a1.transAxes,
        ha="right", fontsize=9.5, color=INK)

cax = fig.add_axes([0.905, 0.135, 0.017, 0.765])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$R_0$  (mm)", fontsize=10.5)
cb.set_ticks(R0s); cb.set_ticklabels([f"{v*1e3:.1f}" for v in R0s])
cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
cb.outline.set_edgecolor(AXIS); cb.outline.set_linewidth(0.7)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, f"fig_split_vs_cst.{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
