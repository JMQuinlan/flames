#!/usr/bin/env python3
"""Final sweep analysis: OLD (no closure, 1st order) vs NEW (split + vanleer).

Table at the TCUT mark for every radius, plus normalised-radius histories.
Runs that diverged before TCUT are reported with their death time, never as a
blank or an extrapolated number.
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
CHI, RB, SB = 14.56, 0.018, 7.28
INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"
TCUT = float(os.environ.get("TCUT", "5.0e-3"))

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
    if not t:
        return None
    return np.array(t), np.array(R), np.array(U), np.array(E)


def at_cut(s):
    """Metrics at TCUT, or None if the run died first."""
    t, R, U, E = s
    if t[-1] < TCUT * 0.98:
        return None
    i = int(np.argmin(np.abs(t - TCUT)))
    return (100*(R[i]/R[0]-1), U[:i+1].max(), 100*(E[i]/E[0]-1), t[i])


old, new = {}, {}
for tag in TAGS:
    o = series(f"{B}/{os.environ.get('OLDDIR','ctl2')}/{os.environ.get('OLDPRE','cst')}_{tag}")
    n = series(f"{B}/{os.environ.get('NEWDIR','sweep5ms_vl')}/cc_{tag}")
    if o: old[tag] = o
    if n: new[tag] = n

print("\n" + "=" * 96)
print(f"10-CASE LAPLACE SWEEP  at {TCUT*1e3:.0f} ms      "
      + os.environ.get("LABELS", "NEW = split closure + vanleer"))
print("=" * 96)
print(f"  {'R0 mm':>6} {'sigma0':>7} | {'OLD dR%':>9} {'NEW dR%':>9} | "
      f"{'OLD |u|max':>11} {'NEW |u|max':>11} | {'OLD dE1%':>9} {'NEW dE1%':>9}")
print("  " + "-" * 92)
for tag in TAGS:
    R0 = float(tag) * 1e-4
    s0 = CHI * ((R0 / RB) ** 2 - 1); s0 = SB if s0 >= SB else max(s0, 0.0)
    o = at_cut(old[tag]) if tag in old else None
    n = at_cut(new[tag]) if tag in new else None
    def f(v, w, src):
        # BUGFIX: the death time must come from the run being reported, not
        # always from OLD.  The previous form printed OLD's death time in the
        # NEW column, which made surviving NEW runs look like they died early.
        wid = 11 if w == 1 else 9
        if v is not None:
            return f"{v[w]:{wid}.{5 if w==1 else 4}f}"
        if src is None:
            return f"{'absent':>{wid}}"
        return f"{'died '+format(src[0][-1]*1e3,'.1f')+'ms':>{wid}}"

    ostr = [f(o, k, old.get(tag)) for k in (0, 1, 2)]
    nstr = [f(n, k, new.get(tag)) for k in (0, 1, 2)]
    print(f"  {R0*1e3:6.1f} {s0:7.3f} | {ostr[0]} {nstr[0]} | "
          f"{ostr[1]} {nstr[1]} | {ostr[2]} {nstr[2]}")
print("  " + "-" * 92)
no = sum(1 for t in TAGS if t in old and at_cut(old[t]))
nn = sum(1 for t in TAGS if t in new and at_cut(new[t]))
print(f"  reached {TCUT*1e3:.0f} ms:   OLD {no}/10     NEW {nn}/10")
print("=" * 96 + "\n")

# ------------------------------------------------------------------ figure --
R0s = np.array([float(t) * 1e-4 for t in TAGS])
norm = mcolors.Normalize(R0s.min(), R0s.max()); cmap = plt.get_cmap("viridis")
fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.8, 4.5))
fig.subplots_adjust(left=0.072, right=0.868, top=0.90, bottom=0.135, wspace=0.22)
for a in (a0, a1):
    a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    a.axhline(1.0, color=AXIS, lw=1.0, ls="--", zorder=1)
    a.set_xlabel("Time  (ms)"); a.set_xlim(0, TCUT*1e3)

for key, ax, ttl, src in (("old", a0, "(a)  OLD: no closure, 1st order", old),
                          ("new", a1, "(b)  NEW: split closure + vanleer", new)):
    for tag in TAGS:
        if tag not in src: continue
        t, R, U, E = src[tag]
        m = t <= TCUT * 1.001
        c = cmap(norm(float(tag) * 1e-4))
        ax.plot(t[m] * 1e3, R[m] / R[0], lw=1.6, color=c, zorder=3)
        if t[-1] < TCUT * 0.98:
            ax.plot(t[-1] * 1e3, R[-1] / R[0], "X", ms=10, color=c,
                    mec="white", mew=1.2, zorder=6)
    ax.set_title(ttl, loc="left", fontsize=10.5, color=INK)
a0.set_ylabel(r"Normalised radius  $R/R_0$")
a1.set_ylabel(r"$R/R_0$")
a0.plot([], [], "X", ms=9, color=AXIS, mec="white", mew=1.0, label="diverged")
a0.legend(frameon=False, loc="upper left")

cax = fig.add_axes([0.892, 0.135, 0.019, 0.765])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cb.set_label(r"$R_0$  (mm)", fontsize=10.5)
cb.set_ticks(R0s); cb.set_ticklabels([f"{v*1e3:.1f}" for v in R0s])
cb.ax.tick_params(labelsize=8.5, length=2.5, width=0.6)
cb.outline.set_edgecolor(AXIS); cb.outline.set_linewidth(0.7)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, os.environ.get("FIGNAME","fig_sweep_final")+f"_{int(TCUT*1e3)}ms.{ext}")
    fig.savefig(p, dpi=dpi); print("wrote", p)
