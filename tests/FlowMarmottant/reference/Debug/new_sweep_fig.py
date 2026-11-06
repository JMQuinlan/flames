#!/usr/bin/env python3
"""NEW configuration only: normalised radius over time, all ten radii.

Split capillary closure (Schmidmayer 2017 operator chain) + vanleer limiter,
constant sigma, eps/dx = 4, 5 ms.

Plotted as PERCENT DEVIATION from R0 rather than R/R0 with an axis offset --
the whole family spans +-0.03%, and an offset axis makes that unreadable.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
import yt
yt.set_log_level(50)

B = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant/" + os.environ.get("NEWDIR", "sweep5ms_vl")
TCUT = float(os.environ.get("TCUT", "5.0e-3"))
OUT = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
TAGS = ["140.0", "165.0", "180.0", "185.0", "190.0",
        "200.0", "210.0", "220.0", "240.0", "280.0"]
CHI, RB, SB = 14.56, 0.018, 7.28
INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 10,
    "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "legend.fontsize": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def series(d):
    t, R = [], []
    for f in sorted(glob.glob(d + "/*cell"))[1:]:
        ds = yt.load(f); ad = ds.all_data()
        vol = np.asarray(ad["index", "cell_volume"]); eta = np.asarray(ad["boxlib", "eta"])
        t.append(float(ds.current_time))
        R.append(np.sqrt(((1 - eta) * vol).sum() / np.pi))
    return np.array(t), np.array(R)


data = {}
for tag in TAGS:
    d = f"{B}/cc_{tag}"
    if glob.glob(d + "/*cell"):
        data[tag] = series(d)
        print(f"  loaded {tag}: n={len(data[tag][0])}")

R0s = np.array([float(t) * 1e-4 for t in TAGS])
norm = mcolors.Normalize(R0s.min(), R0s.max()); cmap = plt.get_cmap("viridis")


def regime(R0):
    s = CHI * ((R0 / RB) ** 2 - 1)
    if s <= 0:      return (1, 1.6)     # buckled  -> dotted
    if s >= SB:     return (5, 2)       # ruptured -> dashed
    return (None, None)                 # elastic  -> solid


fig, a = plt.subplots(figsize=(7.6, 4.6))
fig.subplots_adjust(left=0.115, right=0.80, top=0.935, bottom=0.13)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
a.axhline(0.0, color=AXIS, lw=1.0, ls="--", zorder=1)

peak = 0.0
for tag in TAGS:
    if tag not in data: continue
    t, R = data[tag]
    R0 = float(tag) * 1e-4
    y = 100 * (R / R[0] - 1.0)
    peak = max(peak, np.abs(y).max())
    a.plot(t * 1e3, y, lw=1.7, color=cmap(norm(R0)),
           dashes=regime(R0)[0] and regime(R0) or (None, None),
           solid_capstyle="round", zorder=3)

a.set_xlabel("Time  (ms)")
a.set_ylabel(r"Radius deviation  $100\,(R/R_0-1)$   (%)")
a.set_xlim(0, TCUT * 1e3)
a.text(0.985, 0.04, f"peak |deviation| = {peak:.3f}%", transform=a.transAxes,
       ha="right", fontsize=9.5, color=INK)

from matplotlib.lines import Line2D
key = [Line2D([], [], color=AXIS, lw=1.5, dashes=d, label=l)
       for l, d in [("buckled", (1, 1.6)), ("elastic", (None, None)),
                    ("ruptured", (5, 2))]]
a.legend(handles=key, frameon=False, loc="upper left", handlelength=2.6,
         title="initial regime", title_fontproperties={"size": 9, "weight": "600"})

cax = fig.add_axes([0.825, 0.13, 0.028, 0.805])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$R_0$  (mm)", fontsize=10.5)
cb.set_ticks(R0s); cb.set_ticklabels([f"{v*1e3:.1f}" for v in R0s])
cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
cb.outline.set_edgecolor(AXIS); cb.outline.set_linewidth(0.7)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, f"fig_new_sweep_{int(TCUT*1e3)}ms.{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
