#!/usr/bin/env python3
"""Where we were vs where we are: normalised radius over time.

All at R0 = 20 mm, constant sigma = 3.4153, eps/dx = 4, 15 ms, same mesh.
The only differences between the rows are the ones named.
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

# label -> (path, colour, dashes, what changed)
RUNS = [
    ("no closure, 1st order",       f"{B}/ctl2/cst_200.0",        "#7a7a7a", (1.2, 1.8)),
    ("split closure, 1st order",    f"{B}/limiter15/godunov",     "#0072B2", (None, None)),
    ("split closure, vanleer",      f"{B}/limiter15/vanleer",     "#D55E00", (5.5, 2.0)),
    ("split closure, weno3",        f"{B}/limiter15/weno3",       "#009E73", (2.5, 1.6)),
]

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
        w = eta * (1 - eta); m = w > 0.05
        t.append(float(ds.current_time))
        R.append(np.sqrt(((1 - eta) * vol).sum() / np.pi))
        U.append(np.hypot(ux[m], uy[m]).max())
        E.append((np.asarray(ad["boxlib", "energy1"]) * vol).sum())
    return np.array(t), np.array(R), np.array(U), np.array(E)


data = {}
for lab, d, c, dash in RUNS:
    if not glob.glob(d + "/*cell"):
        print(f"  MISSING: {lab} ({d})"); continue
    data[lab] = series(d)
    print(f"  loaded {lab}: n={len(data[lab][0])}, t_end={data[lab][0][-1]*1e3:.2f} ms")

# ------------------------------------------------------------------ table ---
print("\n" + "=" * 88)
print("WHERE IT WAS  vs  WHERE IT IS      R0 = 20 mm, sigma = 3.4153, eps/dx = 4, 15 ms")
print("=" * 88)
print(f"  {'configuration':<28} {'dR %':>10} {'|u|max':>10} {'gas dE1 %':>11} {'t_end ms':>9}")
print("  " + "-" * 84)
for lab, d, c, dash in RUNS:
    if lab not in data: continue
    t, R, U, E = data[lab]
    print(f"  {lab:<28} {100*(R[-1]/R[0]-1):10.4f} {U.max():10.5f} "
          f"{100*(E[-1]/E[0]-1):11.4f} {t[-1]*1e3:9.2f}")
print("  " + "-" * 84)
if "no closure, 1st order" in data and "split closure, weno3" in data:
    a = data["no closure, 1st order"]; b = data["split closure, weno3"]
    fa = 100*abs(a[1][-1]/a[1][0]-1); fb = 100*abs(b[1][-1]/b[1][0]-1)
    print(f"  net improvement, first row -> last:  dR {fa/max(fb,1e-12):.0f}x   "
          f"|u|max {a[2].max()/max(b[2].max(),1e-12):.0f}x   "
          f"gas energy {abs(a[3][-1]/a[3][0]-1)/max(abs(b[3][-1]/b[3][0]-1),1e-12):.0f}x")
print("=" * 88 + "\n")

# ----------------------------------------------------------------- figure ---
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.6, 4.5))
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.135, wspace=0.24)
for a in (a0, a1):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.axhline(1.0, color=AXIS, lw=1.0, ls="--", zorder=1)
    a.set_xlabel("Time  (ms)"); a.set_xlim(0, 15)

for lab, d, c, dash in RUNS:
    if lab not in data: continue
    t, R, U, E = data[lab]
    for ax in (a0, a1):
        ax.plot(t * 1e3, R / R[0], lw=1.8, color=c,
                dashes=dash if dash[0] else (None, None),
                label=lab, solid_capstyle="round", zorder=3)

a0.set_ylabel(r"Normalised radius  $R/R_0$")
a0.set_title("(a)  full scale", loc="left", fontsize=10.5, color=INK)
a0.legend(frameon=False, loc="upper left")

# zoom on everything except the no-closure run
zs = [data[l][1] for l, _, _, _ in RUNS if l in data and l != "no closure, 1st order"]
if zs:
    lo = min((z / z[0]).min() for z in zs); hi = max((z / z[0]).max() for z in zs)
    pad = 0.15 * max(hi - lo, 1e-4)
    a1.set_ylim(lo - pad, hi + pad)
a1.set_ylabel(r"$R/R_0$")
a1.set_title("(b)  zoom: closure runs only", loc="left", fontsize=10.5, color=INK)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, f"fig_progress.{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
