#!/usr/bin/env python3
"""Physical air/water microbubble sweep: table + normalised-radius figure.

Marmottant 2005 shell on a real UCA: sigma_w = 0.073 N/m, chi = 0.55 N/m,
R_buckling = 2.0 um, R_rupture = 2.1286 um.  Water/air at real properties.
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.lines import Line2D
import yt
yt.set_log_level(50)

B = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant/" + os.environ.get("MBDIR","microbubble")
OUT = os.environ.get("FIG_OUT", os.path.dirname(os.path.abspath(__file__)))
SIGW = float(os.environ.get("SIGW", "0.073"))
CHI  = float(os.environ.get("CHI", "0.55"))
RB   = float(os.environ.get("RB", "2.0e-6"))
RSCALE = float(os.environ.get("RSCALE", "1e6"))  # metres -> display units
RR = RB * np.sqrt(1.0 + SIGW / CHI)
INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 10,
    "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS, "legend.fontsize": 9,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def series(d):
    rows = []
    for f in sorted(glob.glob(d + "/*cell"))[1:]:
        ds = yt.load(f); ad = ds.all_data()
        vol = np.asarray(ad["index", "cell_volume"]); eta = np.asarray(ad["boxlib", "eta"])
        ux = np.asarray(ad["boxlib", "velocityx"]); uy = np.asarray(ad["boxlib", "velocityy"])
        sig = np.asarray(ad["boxlib", "kappa2"]); gam = np.asarray(ad["boxlib", "Gamma"])
        w = eta * (1 - eta); m = w > 0.05
        if not m.any():
            continue
        ww = w[m] * vol[m]; ww /= ww.sum()
        sm = float((ww * sig[m]).sum())
        sd = float(np.sqrt((ww * (sig[m] - sm) ** 2).sum()))
        rows.append(dict(t=float(ds.current_time),
                         R=np.sqrt(((1 - eta) * vol).sum() / np.pi),
                         u=np.hypot(ux[m], uy[m]).max(),
                         sig=sm, spr=100 * sd / abs(sm) if abs(sm) > 1e-14 else 0.0,
                         gam=float((ww * gam[m]).sum())))
    return rows or None


def regime(R0):
    s = CHI * ((R0 / RB) ** 2 - 1.0)
    if s <= 0:   return "buckled", (1, 1.6)
    if s >= SIGW: return "ruptured", (5, 2)
    return "elastic", (None, None)


runs = {}
for d in sorted(glob.glob(B + "/mb_*")):
    if not os.path.isdir(d):
        continue
    R0 = float(os.path.basename(d).split("_")[1]) * float(os.environ.get("TAGSCALE","1e-9"))
    s = series(d)
    if s:
        runs[R0] = s
        print(f"  loaded R0={R0*RSCALE:.1f}  n={len(s)}  t_end={s[-1]['t']*1e9:.1f} ns")

if not runs:
    raise SystemExit("no microbubble runs found")

print("\n" + "=" * 100)
print("PHYSICAL AIR/WATER MICROBUBBLE   Marmottant 2005 shell")
print(f"  sigma_w={SIGW} N/m  chi={CHI} N/m  R_buck={RB*RSCALE:.1f}  "
      f"R_rupt={RR*RSCALE:.2f}  (elastic band {100*(RR/RB-1):.2f}% wide)")
print("=" * 100)
print(f"  {'R0 um':>8} {'R/Rb':>6} {'regime':>9} {'sig_exact':>10} | "
      f"{'sig_end':>10} {'spread%':>8} {'Gam_end':>9} {'dR%':>9} {'|u|max':>10} {'t_end ns':>9}")
print("  " + "-" * 96)
for R0 in sorted(runs):
    r = runs[R0]
    reg, _ = regime(R0)
    se = CHI * ((R0 / RB) ** 2 - 1.0); se = SIGW if se >= SIGW else max(se, 0.0)
    print(f"  {R0*RSCALE:8.2f} {R0/RB:6.3f} {reg:>9} {se:10.6f} | "
          f"{r[-1]['sig']:10.6f} {r[-1]['spr']:8.3f} {r[-1]['gam']:9.6f} "
          f"{100*(r[-1]['R']/r[0]['R']-1):9.4f} {max(x['u'] for x in r):10.5f} "
          f"{r[-1]['t']*float(os.environ.get('TSCALE','1e9')):9.1f}")
print("  " + "-" * 96)
TEND = float(os.environ.get('TEND','1.0e-6'))
full = sum(1 for R0 in runs if runs[R0][-1]['t'] > 0.95*TEND)
print(f"  reached 1.0 us:  {full}/{len(runs)}")
print("=" * 100 + "\n")

R0s = np.array(sorted(runs))
norm = mcolors.Normalize(R0s.min(), R0s.max()); cmap = plt.get_cmap("viridis")
fig, a = plt.subplots(figsize=(7.8, 4.7))
fig.subplots_adjust(left=0.125, right=0.80, top=0.935, bottom=0.13)
a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
a.axhline(0.0, color=AXIS, lw=1.0, ls="--", zorder=1)

peak = 0.0
for R0 in R0s:
    r = runs[R0]
    t = np.array([x["t"] for x in r]) * float(os.environ.get("TSCALE","1e9"))
    y = 100 * (np.array([x["R"] for x in r]) / r[0]["R"] - 1.0)
    peak = max(peak, np.abs(y).max())
    a.plot(t, y, lw=1.7, color=cmap(norm(R0)),
           dashes=regime(R0)[1] if regime(R0)[1][0] else (None, None),
           solid_capstyle="round", zorder=3)

a.set_xlabel(os.environ.get("TLABEL","Time  (ns)"))
a.set_ylabel(r"Radius deviation  $100\,(R/R_0-1)$   (%)")
a.text(0.985, 0.04, f"peak |deviation| = {peak:.3f}%", transform=a.transAxes,
       ha="right", fontsize=9.5, color=INK)
key = [Line2D([], [], color=AXIS, lw=1.5, dashes=d, label=l)
       for l, d in [("buckled", (1, 1.6)), ("elastic", (None, None)), ("ruptured", (5, 2))]]
a.legend(handles=key, frameon=False, loc="upper left", handlelength=2.6,
         title="initial regime", title_fontproperties={"size": 9, "weight": "600"})

cax = fig.add_axes([0.825, 0.13, 0.028, 0.805])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(os.environ.get("RLABEL", r"$R_0$  ($\mu$m)"), fontsize=10.5)
cb.set_ticks(R0s); cb.set_ticklabels([f"{v*1e6:.3f}" for v in R0s])
cb.ax.tick_params(labelsize=8, length=2.5, width=0.6)
cb.outline.set_edgecolor(AXIS); cb.outline.set_linewidth(0.7)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, os.environ.get("FIGNAME","fig_microbubble")+f".{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
