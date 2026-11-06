#!/usr/bin/env python3
"""Publication figures for the Marmottant shell paper.

Two figures, both vector (PDF) plus a PNG proof:
  fig_radius.pdf   R(t) for the initial-radius sweep
  fig_tension.pdf  sigma_eff vs areal strain, LINEAR axes

Colour encodes R0, a continuous ordered quantity, so it uses a single
perceptually-uniform sequential ramp (viridis) with a colourbar -- never
cycled categorical hues.  Line style encodes regime, so series identity
never rests on colour alone.  Reference lines are neutral grey, not red,
so nothing in the figure depends on red/green discrimination.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("SWEEP_NPZ", os.path.join(HERE, "sweep_data.npz"))
OUT = os.environ.get("FIG_OUT", HERE)

CHI, RB, SBRK, SIGW = 14.56, 0.018, 7.28, 7.28

INK = "#1a1a1a"
AXIS = "#555555"
GRID = "#dcdcdc"
REF = "#777777"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load():
    d = np.load(DATA)
    tags = sorted({k.split("|")[0] for k in d.files})
    runs = []
    for t in tags:
        g = {k.split("|")[1]: d[k] for k in d.files if k.startswith(t + "|")}
        g["R0"] = float(g["R0"])
        # The t=0 plotfile is written before the first RHS, so the Gamma and
        # sigma diagnostics are still zero there.  Drop it.
        n = len(g["t"])
        keep = np.ones(n, bool)
        keep[0] = False
        for k, v in list(g.items()):
            if isinstance(v, np.ndarray) and v.shape == (n,):
                g[k] = v[keep]
        runs.append(g)
    runs.sort(key=lambda r: r["R0"])
    return runs


def regime(R0):
    s = CHI * ((R0 / RB) ** 2 - 1.0)
    if s <= 0:
        return "buckled", (1, 1.6)
    if s >= SBRK:
        return "ruptured", (5, 2)
    return "elastic", ()


def style(a):
    """Publication axes: no chart junk, only the two spines that carry data."""
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6, alpha=0.9)
    a.set_axisbelow(True)
    a.tick_params(length=3.2, width=0.7)


def regime_key(fig, **kw):
    handles = [Line2D([], [], color=AXIS, lw=1.5, dashes=regime(R0)[1], label=lab)
               for lab, R0 in [("buckled", 0.014), ("elastic", 0.020),
                               ("ruptured", 0.028)]]
    return fig.legend(handles=handles, frameon=False, handlelength=2.6, **kw)


def colourbar(fig, rect, norm, cmap, R0s):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$R_0$  (mm)", fontsize=10.5)
    cb.set_ticks(R0s)
    cb.set_ticklabels([f"{v * 1e3:.1f}" for v in R0s])
    cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
    cb.outline.set_edgecolor(AXIS)
    cb.outline.set_linewidth(0.7)
    return cb


def save(fig, stem):
    for ext, dpi in (("pdf", None), ("png", 400)):
        p = os.path.join(OUT, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches=None)
        print("wrote", p)


runs = load()
R0s = np.array([r["R0"] for r in runs])
norm = mcolors.Normalize(R0s.min(), R0s.max())
cmap = plt.get_cmap("viridis")

# ======================================================================
# Figure 1 -- radius history
# ======================================================================
fig, a = plt.subplots(figsize=(6.6, 4.3))
fig.subplots_adjust(left=0.105, right=0.815, top=0.965, bottom=0.135)
style(a)

a.axhline(RB * 1e3, color=REF, lw=0.9, ls="-", zorder=1)
a.text(0.012, RB * 1e3, r"$R_{\mathrm{buck}}$", color=REF, fontsize=9.5,
       va="bottom", ha="left", transform=a.get_yaxis_transform(), zorder=1)

for r in runs:
    a.plot(r["t"] * 1e3, r["R"] * 1e3, lw=1.6, color=cmap(norm(r["R0"])),
           dashes=regime(r["R0"])[1] or (None, None),
           solid_capstyle="round", zorder=3)

a.set_xlabel("Time  (ms)")
a.set_ylabel(r"Bubble radius $R$  (mm)")
a.set_xlim(left=0)
colourbar(fig, [0.845, 0.135, 0.024, 0.83], norm, cmap, R0s)
regime_key(fig, loc="upper left", bbox_to_anchor=(0.145, 0.955))
save(fig, "fig_radius")

# ======================================================================
# Figure 2 -- effective tension against areal strain, LINEAR axes
# ======================================================================
fig, a = plt.subplots(figsize=(6.6, 4.3))
fig.subplots_adjust(left=0.105, right=0.815, top=0.965, bottom=0.135)
style(a)

# Constitutive law first, so the data fall on top of it.
s = np.linspace(0.45, 2.6, 600)
a.plot(s, np.clip(CHI * (s - 1.0), 0.0, SIGW), color=INK, lw=2.2, zorder=2,
       label=r"$\sigma=\chi\,(A/A_{\mathrm{buck}}-1)$")

for r in runs:
    gb = (r["R0"] / RB) ** 2          # Gamma_buck / Gamma_0
    x = gb / r["gam"]                  # A / A_buck
    y = r["sig"]
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        continue
    c = cmap(norm(r["R0"]))
    a.plot(x[m], y[m], lw=1.5, color=c, dashes=regime(r["R0"])[1] or (None, None),
           zorder=4)
    a.plot(x[m][0], y[m][0], "o", ms=5.5, color=c, mec="white", mew=0.9, zorder=6)

a.set_xlabel(r"Areal strain  $A/A_{\mathrm{buck}}$")
a.set_ylabel(r"Effective tension  $\sigma_{\mathrm{eff}}$")
a.set_xlim(0.45, 2.6)
a.set_ylim(-0.35, 8.4)
a.set_yticks([0, 2, 4, 6, SIGW])
a.set_yticklabels(["0", "2", "4", "6", r"$\sigma_\ell$"])
a.legend(frameon=False, loc="upper left", handlelength=2.2)
colourbar(fig, [0.845, 0.135, 0.024, 0.83], norm, cmap, R0s)
regime_key(fig, loc="lower right", bbox_to_anchor=(0.80, 0.16))
save(fig, "fig_tension")
