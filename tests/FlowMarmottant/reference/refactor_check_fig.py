#!/usr/bin/env python3
"""Post-refactor sanity check: Laplace bubble, refactored build vs pre-refactor.

Same case both sides (R0 = 20 mm, constant sigma = 3.4153, split closure +
vanleer, eps/dx = 4).  The refactor removed cfun_resync, art_visc_coeff,
capillary_work, ShellGammaSnapshot/Restore, InterfaceSharpening and the orphaned
sharpening params, and gated check4nans behind nan_check (default 0).
None of that should change the physics.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.set_log_level(50)

B = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant"
OUT = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 10,
    "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "legend.fontsize": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def series(d):
    t, R, U, E = [], [], [], []
    for f in sorted(glob.glob(d + "/*cell"))[1:]:
        ds = yt.load(f); ad = ds.all_data()
        vol = np.asarray(ad["index", "cell_volume"]); eta = np.asarray(ad["boxlib", "eta"])
        ux = np.asarray(ad["boxlib", "velocityx"]); uy = np.asarray(ad["boxlib", "velocityy"])
        m = (eta * (1 - eta)) > 0.05
        t.append(float(ds.current_time))
        R.append(np.sqrt(((1 - eta) * vol).sum() / np.pi))
        U.append(np.hypot(ux[m], uy[m]).max())
        E.append((np.asarray(ad["boxlib", "energy1"]) * vol).sum())
    return np.array(t), np.array(R), np.array(U), np.array(E)


RUNS = [("pre-refactor",  f"{B}/sweep15ms_merged/cc_200.0", "#0072B2", (None, None)),
        ("post-refactor", f"{B}/refactor_laplace",          "#D55E00", (5.5, 2.0))]
data = {}
for lab, d, c, dash in RUNS:
    if glob.glob(d + "/*cell"):
        data[lab] = series(d)
        print(f"  {lab}: n={len(data[lab][0])}  t_end={data[lab][0][-1]*1e3:.3f} ms")

TMAX = min(v[0][-1] for v in data.values())
print(f"\n  common window: 0 -> {TMAX*1e3:.3f} ms")
print(f"  {'run':16s} {'dR%':>9} {'|u|max':>10} {'dE1%':>9}")
for lab, _, _, _ in RUNS:
    if lab not in data: continue
    t, R, U, E = data[lab]
    i = int(np.argmin(np.abs(t - TMAX)))
    print(f"  {lab:16s} {100*(R[i]/R[0]-1):9.4f} {U[:i+1].max():10.5f} {100*(E[i]/E[0]-1):9.4f}")

fig, ax = plt.subplots(1, 3, figsize=(13.2, 4.0))
fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.145, wspace=0.29)
for a in ax:
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.set_xlabel("Time  (ms)"); a.set_xlim(0, TMAX * 1e3)

for lab, d, c, dash in RUNS:
    if lab not in data: continue
    t, R, U, E = data[lab]
    m = t <= TMAX * 1.001
    kw = dict(lw=1.8, color=c, label=lab)
    if dash[0]: kw["dashes"] = dash
    ax[0].plot(t[m] * 1e3, 100 * (R[m] / R[0] - 1), **kw)
    ax[1].plot(t[m] * 1e3, U[m], **kw)
    ax[2].plot(t[m] * 1e3, 100 * (E[m] / E[0] - 1), **kw)

ax[0].axhline(0, color=AXIS, lw=1.0, ls="--", zorder=1)
ax[0].set_ylabel(r"$100\,(R/R_0-1)$   (%)")
ax[0].set_title("(a)  radius deviation", loc="left", fontsize=10.5)
ax[0].legend(frameon=False, loc="best")
ax[1].set_ylabel(r"$|u|_{\mathrm{max}}$ on the band")
ax[1].set_title("(b)  parasitic velocity", loc="left", fontsize=10.5)
ax[2].axhline(0, color=AXIS, lw=1.0, ls="--", zorder=1)
ax[2].set_ylabel(r"gas $\Delta E_1$   (%)")
ax[2].set_title("(c)  gas internal energy", loc="left", fontsize=10.5)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, f"fig_refactor_check.{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
