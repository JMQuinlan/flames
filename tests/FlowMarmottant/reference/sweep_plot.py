#!/usr/bin/env python3
"""Marmottant R0 sweep figure.

Colour = R0 (a CONTINUOUS ordered quantity -> sequential ramp + colourbar, not
10 cycled categorical hues).  Line style = regime (buckled / elastic / ruptured),
so series identity never rests on colour alone.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
CHI, RB, SBRK, SIGW = 14.56, 0.018, 7.28, 7.28
GAM0 = 1.0

d = np.load(os.path.join(HERE, "sweep_data.npz"))
tags = sorted({k.split("|")[0] for k in d.files})
runs = []
for t in tags:
    g = {k.split("|")[1]: d[k] for k in d.files if k.startswith(t + "|")}
    g["R0"] = float(g["R0"])
    runs.append(g)
runs.sort(key=lambda r: r["R0"])

# t=0 plotfiles are written BEFORE the first RHS, so Gamma and sigma are still 0
# there (they are diagnostics filled during RHS, not initial conditions).  Drop
# that first sample from every series so it cannot poison the strain axis
# (strain = Gamma_buck/Gamma -> inf) or drag the sigma curves to zero.
for r in runs:
    n = len(r["t"])
    keep = np.ones(n, bool); keep[0] = False
    for k, v in list(r.items()):
        if isinstance(v, np.ndarray) and v.shape == (n,):
            r[k] = v[keep]

R0s = np.array([r["R0"] for r in runs])
norm = mcolors.Normalize(R0s.min(), R0s.max())
cmap = cm.viridis
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])


def regime(R0):
    el = CHI * ((R0 / RB) ** 2 - 1.0)
    if el <= 0:      return "buckled",  ":"
    if el >= SBRK:   return "ruptured", "--"
    return "elastic", "-"


plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.8,
    "grid.color": "#d7dce2", "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "xtick.color": "#4a525b", "ytick.color": "#4a525b",
    "axes.labelcolor": "#1d2329", "text.color": "#1d2329",
})

fig, ax = plt.subplots(4, 2, figsize=(13.0, 16.2))
fig.subplots_adjust(left=0.075, right=0.90, top=0.945, bottom=0.045,
                    hspace=0.30, wspace=0.24)
A = ax.ravel()
for a in A:
    a.grid(True, zorder=0)
    a.set_axisbelow(True)

LW = 1.7


def plot(a, xk, yk, **kw):
    for r in runs:
        _, ls = regime(r["R0"])
        a.plot(r[xk], r[yk], ls, lw=LW, color=cmap(norm(r["R0"])),
               solid_capstyle="round", **kw)


# (a) R(t) --------------------------------------------------------------
a = A[0]
for r in runs:
    _, ls = regime(r["R0"])
    a.plot(r["t"] * 1e3, r["R"] * 1e3, ls, lw=LW, color=cmap(norm(r["R0"])))
    a.axhline(r["R0"] * 1e3, color=cmap(norm(r["R0"])), lw=0.5, alpha=0.35)
a.axhline(RB * 1e3, color="#b2182b", lw=1.2, ls="-.", zorder=5)
a.text(0.985, RB * 1e3, " R$_{buckling}$", color="#b2182b", fontsize=9,
       va="bottom", ha="right", transform=a.get_yaxis_transform())
a.set_xlabel("t  (ms)"); a.set_ylabel("R  (mm)")
a.set_title("Bubble radius", loc="left", fontsize=11, fontweight=600)

# (b) R/R0(t) -----------------------------------------------------------
a = A[1]
for r in runs:
    _, ls = regime(r["R0"])
    a.plot(r["t"] * 1e3, r["R"] / r["R0"], ls, lw=LW, color=cmap(norm(r["R0"])))
a.axhline(1.0, color="#5b6673", lw=1.0, ls="--")
a.set_xlabel("t  (ms)"); a.set_ylabel("R / R$_0$")
a.set_title("Normalised radius", loc="left", fontsize=11, fontweight=600)

# (c) sigma(t) ----------------------------------------------------------
a = A[2]
for r in runs:
    a.plot(r["t"] * 1e3, r["sig"], regime(r["R0"])[1], lw=LW,
           color=cmap(norm(r["R0"])))
    a.axhline(CHI * ((r["R0"] / RB) ** 2 - 1.0) if 0 < CHI * ((r["R0"] / RB) ** 2 - 1.0) < SBRK
              else (SIGW if CHI * ((r["R0"] / RB) ** 2 - 1.0) >= SBRK else 0.0),
              color=cmap(norm(r["R0"])), lw=0.5, alpha=0.35)
a.axhline(SIGW, color="#b2182b", lw=1.2, ls="-.")
a.text(0.985, SIGW, " $\\sigma_{water}$", color="#b2182b", fontsize=9,
       va="bottom", ha="right", transform=a.get_yaxis_transform())
a.set_xlabel("t  (ms)"); a.set_ylabel("$\\sigma_{eff}$")
a.set_title("Effective shell tension   (faint = analytic $\\sigma(R_0)$)",
            loc="left", fontsize=11, fontweight=600)

# (d) Gamma(t) ----------------------------------------------------------
a = A[3]
for r in runs:
    a.plot(r["t"] * 1e3, r["gam"], regime(r["R0"])[1], lw=LW,
           color=cmap(norm(r["R0"])))
a.axhline(GAM0, color="#5b6673", lw=1.0, ls="--")
a.text(0.985, GAM0, " $\\Gamma_0$ (unstrained)", color="#5b6673", fontsize=9,
       va="bottom", ha="right", transform=a.get_yaxis_transform())
a.set_xlabel("t  (ms)"); a.set_ylabel("$\\Gamma$  (areal shell density)")
a.set_title("Shell areal density", loc="left", fontsize=11, fontweight=600)

# (e) sigma vs R with error bars ---------------------------------------
a = A[4]
for r in runs:
    ev = max(1, len(r["t"]) // 14)
    a.errorbar(r["R"][::ev] * 1e3, r["sig"][::ev], yerr=r["sig_std"][::ev],
               fmt="o", ms=4.5, lw=0, elinewidth=1.1, capsize=2.4,
               color=cmap(norm(r["R0"])), ecolor=cmap(norm(r["R0"])),
               alpha=0.95, zorder=3)
    a.plot(r["R"] * 1e3, r["sig"], regime(r["R0"])[1], lw=1.0,
           color=cmap(norm(r["R0"])), alpha=0.5, zorder=2)
a.axhline(SIGW, color="#b2182b", lw=1.2, ls="-.")
a.axvline(RB * 1e3, color="#b2182b", lw=1.0, ls=":")
a.set_xlabel("R  (mm)"); a.set_ylabel("$\\sigma_{eff}$")
a.set_title("Tension vs radius   (bars = $\\sigma$ variation around the interface)",
            loc="left", fontsize=11, fontweight=600)

# (f) parasitic-current health ----------------------------------------
# |u|max on the band.  The capillary stress tensor drives parasitic currents
# that, with mu = 0, grow EXPONENTIALLY and eventually corrupt the shell field;
# with adequate viscosity (Oh ~ 0.1) they saturate and decay.  A Laplace-
# balanced bubble should not move, so anything but a small, decaying |u| here
# means the run is being driven by a numerical artifact.
a = A[5]
for r in runs:
    if "umax" not in r:
        continue
    a.semilogy(r["t"] * 1e3, np.maximum(r["umax"], 1e-6), regime(r["R0"])[1],
               lw=LW, color=cmap(norm(r["R0"])))
a.set_xlabel("t  (ms)"); a.set_ylabel("$|u|_{max}$ on the band")
a.set_title("Parasitic-current health", loc="left", fontsize=11, fontweight=600)

# (g) sigma spread (instability indicator) ------------------------------
a = A[6]
for r in runs:
    rel = np.where(np.abs(r["sig"]) > 1e-6, r["sig_std"] / np.abs(r["sig"]), np.nan)
    a.semilogy(r["t"] * 1e3, 100 * rel, regime(r["R0"])[1], lw=LW,
               color=cmap(norm(r["R0"])))
a.set_xlabel("t  (ms)"); a.set_ylabel("$\\sigma$ spread  (%, std/mean)")
a.set_title("Tension-field uniformity", loc="left", fontsize=11, fontweight=600)

# (h) shape mode e4 -----------------------------------------------------
a = A[7]
for r in runs:
    a.semilogy(r["t"] * 1e3, np.maximum(r["e4"], 1e-6), regime(r["R0"])[1],
               lw=LW, color=cmap(norm(r["R0"])))
a.set_xlabel("t  (ms)"); a.set_ylabel("$\\epsilon_4$  (4-fold shape mode)")
a.set_title("Shape distortion", loc="left", fontsize=11, fontweight=600)

# colourbar + regime key ------------------------------------------------
cax = fig.add_axes([0.915, 0.35, 0.012, 0.42])
cb = fig.colorbar(sm, cax=cax)
cb.set_label("R$_0$  (mm)", fontsize=10)
cb.set_ticks(R0s)
cb.set_ticklabels([f"{v*1e3:.2f}" for v in R0s])
cb.ax.tick_params(labelsize=8)
cb.outline.set_edgecolor("#9aa3ad"); cb.outline.set_linewidth(0.8)

from matplotlib.lines import Line2D
key = [Line2D([], [], color="#4a525b", ls=ls, lw=1.7, label=lab)
       for lab, ls in [("buckled", ":"), ("elastic", "-"), ("ruptured", "--")]]
fig.legend(handles=key, loc="lower right", bbox_to_anchor=(0.935, 0.14),
           frameon=False, fontsize=9, title="initial regime",
           title_fontproperties={"size": 9, "weight": "600"})

fig.suptitle("Marmottant coated bubble — initial-radius sweep, buckled to ruptured",
             fontsize=14, fontweight=600, x=0.075, ha="left", y=0.977)

out = os.path.join(HERE, "marmottant_sweep.png")
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)


# ======================================================================
# STANDALONE: collapse onto the constitutive law
# ======================================================================
fig2, a = plt.subplots(figsize=(8.2, 6.6))
fig2.subplots_adjust(left=0.11, right=0.80, top=0.90, bottom=0.11)
a.grid(True); a.set_axisbelow(True)
# Plotted against (A/A_buck - 1) on LOG-LOG.  The elastic law sigma = chi(s-1)
# is then a STRAIGHT LINE OF SLOPE 1, and the rupture branch a horizontal
# plateau at sigma_water -- so departure from the constitutive law reads off
# directly as departure from a line.  On linear (or log-sigma vs linear-s) axes
# the law is near-vertical at s=1 and every trajectory piles up there.
# The buckled branch (s <= 1, sigma = 0) cannot appear on log axes by
# construction; those three cases sit exactly at sigma = 0 for all 15 ms.
sm1 = np.logspace(-3, 0.45, 400)
ideal = np.minimum(CHI * sm1, SIGW)
a.loglog(sm1, ideal, color="#1d2329", lw=2.6, zorder=1,
         label="$\\sigma=\\chi\\,(A/A_{buck}-1)$, capped at $\\sigma_w$")
for r in runs:
    gb = (r["R0"] / RB) ** 2
    x = gb / r["gam"] - 1.0
    y = r["sig"]
    m = (x > 0) & (y > 1e-3)
    if not m.any():
        continue
    c = cmap(norm(r["R0"]))
    a.plot(x[m], y[m], regime(r["R0"])[1], lw=1.5, color=c, alpha=0.95, zorder=4)
    a.plot(x[m][0], y[m][0], "o", ms=6.5, color=c, mec="white", mew=1.0, zorder=6)
    a.plot(x[m][-1], y[m][-1], "s", ms=5.5, color=c, mec="white", mew=1.0, zorder=6)
a.axhline(SIGW, color="#b2182b", lw=1.0, ls="-.", zorder=2)
a.set_xlim(1e-3, 3.0); a.set_ylim(1e-2, 12)
a.set_xlabel("$A/A_{buck} - 1$   (areal strain above buckling)")
a.set_ylabel("$\\sigma_{eff}$")
a.legend(frameon=False, fontsize=8.5, loc="lower right")
a.set_title("Collapse onto the constitutive law", loc="left",
            fontsize=11, fontweight=600)
a.text(0.985, SIGW, " $\\sigma_{water}$", color="#b2182b", fontsize=9, va="bottom",
       ha="right", transform=a.get_yaxis_transform())
a.text(0.03, 0.80, "circle = t start,  square = t end\nbuckled cases ($\\sigma$=0) off-scale",
       transform=a.transAxes, fontsize=8, color="#5b6673", va="top")


a.set_title("Collapse onto the Marmottant constitutive law", loc="left",
            fontsize=13, fontweight=600)
cax2 = fig2.add_axes([0.83, 0.11, 0.022, 0.79])
cb2 = fig2.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax2)
cb2.set_label("R$_0$  (mm)", fontsize=10)
cb2.set_ticks(R0s); cb2.set_ticklabels([f"{v*1e3:.2f}" for v in R0s])
cb2.ax.tick_params(labelsize=8)
out2 = os.path.join(HERE, "marmottant_collapse.png")
fig2.savefig(out2, dpi=190, facecolor="white")
print("wrote", out2)
