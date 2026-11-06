#!/usr/bin/env python3
"""Control figure: is the Laplace drift the Marmottant shell, or the capillary
stress tensor underneath it?

Three families at the same ten radii, identical in every other respect:
  marm : Marmottant advected shell   (sweep_dt)
  cst  : marmottant=0, sigma frozen at the analytic sigma(R0)
  nost : apply_surface_tension=0, uniform pressure (no jump to balance)

Colour is categorical (three families), so it uses the Okabe-Ito
colour-blind-safe palette, and each family also carries a distinct line style
and marker so identity never rests on colour alone.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

S = os.environ.get("SCRATCH", ".")
OUT = os.environ.get("FIG_OUT", ".")
CHI, RB, SBRK = 14.56, 0.018, 7.28

INK, AXIS, GRID = "#1a1a1a", "#555555", "#dcdcdc"
FAM = {                                    # Okabe-Ito, CVD-safe
    "marm": ("Marmottant shell",        "#0072B2", (None, None), "o"),
    "cst":  (r"Constant $\sigma$",      "#D55E00", (5.5, 2.0),   "s"),
    "nost": ("No surface tension",      "#009E73", (1.2, 1.8),   "^"),
}
TAGS = ["140.0", "165.0", "180.0", "185.0", "190.0",
        "200.0", "210.0", "220.0", "240.0", "280.0"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 10,
    "axes.labelsize": 11, "axes.edgecolor": AXIS, "axes.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": AXIS, "ytick.color": AXIS,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "legend.fontsize": 9, "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def load(path):
    d = np.load(path)
    out = {}
    for t in sorted({k.split("|")[0] for k in d.files}):
        g = {k.split("|")[1]: d[k] for k in d.files if k.startswith(t + "|")}
        n = len(g["t"])
        out[t] = {k: (v[1:] if getattr(v, "shape", ()) == (n,) else v)
                  for k, v in g.items()}
    return out


marm, ctl = load(os.path.join(S, "dt/sweep_data.npz")), load(os.path.join(S, "ctl/sweep_data.npz"))
series = {"marm": lambda t: marm["sw_" + t],
          "cst":  lambda t: ctl["cst_" + t],
          "nost": lambda t: ctl["nost_" + t]}


def sigma0(R0):
    s = CHI * ((R0 / RB) ** 2 - 1.0)
    return SBRK if s >= SBRK else max(s, 0.0)


def style(a):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(True, color=GRID, lw=0.6, alpha=0.9)
    a.set_axisbelow(True)
    a.tick_params(length=3.2, width=0.7)


fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.2, 4.4))
fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.135, wspace=0.235)
for a in (a0, a1):
    style(a)

# ---- (a) the reference case, R0 = 20 mm -------------------------------------
REF = "200.0"
for k, (lab, c, dash, mk) in FAM.items():
    r = series[k](REF)
    a0.semilogy(r["t"] * 1e3, np.maximum(r["umax"], 1e-14), lw=1.9, color=c,
                dashes=dash, label=lab, solid_capstyle="round")
a0.set_xlabel("Time  (ms)")
a0.set_ylabel(r"$|u|_{\mathrm{max}}$ on the interface band")
a0.set_xlim(0, 15)
a0.set_ylim(1e-11, 1e2)
a0.legend(frameon=False, loc="center right", handlelength=2.8)
a0.set_title(r"(a)  $R_0=20$ mm,  $\sigma_0=3.415$", loc="left",
             fontsize=10.5, color=INK)

# ---- (b) scaling with tension across all radii ------------------------------
for k, (lab, c, dash, mk) in FAM.items():
    xs, ys, blew = [], [], []
    for t in TAGS:
        r = series[k](t)
        s0 = sigma0(float(r["R0"]))
        xs.append(s0)
        ys.append(max(float(np.nanmax(r["umax"])), 1e-14))
        blew.append(r["t"][-1] < 1.4e-2)
    xs, ys, blew = np.array(xs), np.array(ys), np.array(blew)
    a1.semilogy(xs, ys, lw=1.5, color=c, dashes=dash, alpha=0.85, zorder=2)
    a1.semilogy(xs[~blew], ys[~blew], mk, ms=7, color=c, mec="white", mew=1.0,
                zorder=4, label=lab)
    if blew.any():
        a1.semilogy(xs[blew], ys[blew], "X", ms=10, color=c, mec="white",
                    mew=1.1, zorder=5)
a1.semilogy([], [], "X", ms=9, color=AXIS, mec="white", mew=1.0,
            label="diverged before 15 ms")
a1.set_xlabel(r"Prescribed tension  $\sigma_0$")
a1.set_ylabel(r"peak $|u|_{\mathrm{max}}$ over the run")
a1.set_xlim(-0.35, 7.9)
a1.set_ylim(1e-11, 1e2)
a1.legend(frameon=False, loc="lower right", numpoints=1)
a1.set_title("(b)  all ten radii", loc="left", fontsize=10.5, color=INK)

for ext, dpi in (("pdf", None), ("png", 400)):
    p = os.path.join(OUT, f"fig_control.{ext}")
    fig.savefig(p, dpi=dpi)
    print("wrote", p)
