"""c-based vs eta-based capillary field: Laplace sweep comparison.

Reads data/laplace_cfun_R*.csv and data/laplace_eta_R*.csv (identical metrics,
produced by extract_laplace.py with LAPLACE_TAG) and writes

    figures/fig_cfun_vs_eta.{png,pdf}   three-panel c-vs-eta comparison
    figures/fig_radius_eta.{png,pdf}    eta-only twin of the paper's fig_radius

Style follows make_figures.py: Okabe-Ito regime colors, regime line styles,
viridis for initial radius.
"""
import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
FIG  = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

CHI, RB, SBRK = 0.55, 1.0e-3, 0.073
R_RUP = RB * np.sqrt(1.0 + SBRK / CHI)
C_BUCK, C_ELAS, C_RUPT = "#0072B2", "#009E73", "#D55E00"
C_INK, C_MUTED = "#222222", "#777777"
LS = {"buckled": ":", "elastic": "-", "ruptured": "--"}
REG_C = {"buckled": C_BUCK, "elastic": C_ELAS, "ruptured": C_RUPT}
TW = 6.5

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white"})


def regime(R0):
    if R0 <= RB:      return "buckled"
    if R0 < R_RUP:    return "elastic"
    return "ruptured"


def load(tag):
    out = []
    for f in sorted(glob.glob(os.path.join(DATA, f"laplace_{tag}R*.csv"))):
        hdr = open(f).readline()
        R0 = float(re.search(r"R0=([0-9.eE+-]+)", hdr).group(1))
        d = np.genfromtxt(f, delimiter=",", names=True, skip_header=1)
        out.append(dict(name=os.path.basename(f), R0=R0, d=d))
    return sorted(out, key=lambda r: r["R0"])


def save(fig, stem):
    for ext in ("png", "pdf"):
        p = os.path.join(FIG, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight"); print("wrote", p)
    plt.close(fig)


cf, et = load("cfun_"), load("eta_")
pairs = []
for a in cf:
    b = next((x for x in et if abs(x["R0"] - a["R0"]) < 1e-12), None)
    if b is None:
        continue
    n = min(len(a["d"]), len(b["d"]))
    if n < 2:
        continue
    pairs.append((a["R0"], a["d"], b["d"], n))
print(f"{len(pairs)} matched pairs")

# =========================================================== FIGURE 1 ======
fig, ax = plt.subplots(1, 3, figsize=(TW * 1.62, 3.05))

def thresh_lines(a, ylog=False):
    for xv, lab in ((RB, r"$R_{\rm buck}$"), (R_RUP, r"$R_{\rm rup}$")):
        a.axvline(xv * 1e3, color=C_MUTED, lw=0.9, ls=(0, (4, 2)), zorder=0)
        a.annotate(lab, xy=(xv * 1e3, 0.0), xycoords=("data", "axes fraction"),
                   xytext=(3, 4), textcoords="offset points",
                   fontsize=7.5, color=C_MUTED)

# -- (a) final radius error ------------------------------------------------
a = ax[0]
R0s = np.array([p[0] for p in pairs]) * 1e3
ec = np.array([100 * (p[1]["R_vol"][p[3]-1] / p[1]["R_vol"][0] - 1) for p in pairs])
ee = np.array([100 * (p[2]["R_vol"][p[3]-1] / p[2]["R_vol"][0] - 1) for p in pairs])
thresh_lines(a)
a.axhline(0, color=C_MUTED, lw=0.8)
a.plot(R0s, ec, "o-", color=C_INK,  ms=5, lw=1.4, mfc="white", mew=1.3, label="color function $c$")
a.plot(R0s, ee, "s--", color=C_RUPT, ms=5, lw=1.4, mfc="white", mew=1.3, label=r"volume fraction $\eta$")
a.set_xlabel(r"Initial radius $R_0$  (mm)")
a.set_ylabel(r"Radius error at 10 ms  (%)")
a.set_title("(a)  Final radius error", loc="left")
a.legend(frameon=False, loc="upper left")

# -- (b) peak spurious velocity -------------------------------------------
b = ax[1]
uc = np.array([p[1]["umax"][:p[3]].max() for p in pairs])
ue = np.array([p[2]["umax"][:p[3]].max() for p in pairs])
thresh_lines(b)
b.semilogy(R0s, uc, "o-", color=C_INK,  ms=5, lw=1.4, mfc="white", mew=1.3, label="$c$")
b.semilogy(R0s, ue, "s--", color=C_RUPT, ms=5, lw=1.4, mfc="white", mew=1.3, label=r"$\eta$")
b.set_xlabel(r"Initial radius $R_0$  (mm)")
b.set_ylabel(r"peak $|u|$  (m/s)")
b.set_title("(b)  Peak spurious velocity", loc="left")
b.legend(frameon=False, loc="lower left")

# -- (c) the three threshold cases in time --------------------------------
c = ax[2]
want = [(1.000e-3, "1.000", C_BUCK), (1.012e-3, "1.012", C_ELAS), (1.064e-3, "1.064", C_RUPT)]
for R0t, lab, col in want:
    hit = min(pairs, key=lambda q: abs(q[0] - R0t))
    if abs(hit[0] - R0t) > 1e-6:
        continue
    _, dc, de, n = hit
    t = dc["t"][:n] * 1e3
    c.semilogy(t, np.maximum(dc["umax"][:n], 1e-8), "-",  color=col, lw=1.5)
    c.semilogy(t, np.maximum(de["umax"][:n], 1e-8), "--", color=col, lw=1.5)
    c.plot([], [], "-", color=col, lw=1.5, label=rf"$R_0 = {lab}$ mm")
c.plot([], [], "-",  color=C_INK, lw=1.4, label="$c$ (solid)")
c.plot([], [], "--", color=C_INK, lw=1.4, label=r"$\eta$ (dashed)")
c.set_xlabel("Time  (ms)")
c.set_ylabel(r"$|u|_{\max}$  (m/s)")
c.set_title("(c)  Threshold cases: onset of noise", loc="left")
c.legend(frameon=False, fontsize=7.2, loc="lower right", ncol=1)
fig.tight_layout()
save(fig, "fig_cfun_vs_eta")

# =========================================================== FIGURE 2 ======
fig, a = plt.subplots(figsize=(TW, 4.8))
fig.subplots_adjust(left=0.105, right=0.815, top=0.955, bottom=0.135)
allR = np.array([r["R0"] for r in et])
norm = mcolors.Normalize(allR.min() * 1e3, allR.max() * 1e3)
cmap = plt.get_cmap("viridis")
for r in et:
    d, R0 = r["d"], r["R0"]
    a.plot(d["t"] * 1e3, 100 * (d["R_vol"] / d["R_vol"][0] - 1),
           lw=1.6, color=cmap(norm(R0 * 1e3)), ls=LS[regime(R0)], solid_capstyle="round")
a.axhline(0, color=C_MUTED, lw=0.8)
a.set_xlabel(r"Time, $t$  (ms)")
a.set_ylabel(r"Radius error, $100\,[R(t)/R(0)-1]$   (%)")
a.set_xlim(left=0)
for reg in ("buckled", "elastic", "ruptured"):
    a.plot([], [], ls=LS[reg], color=C_MUTED, lw=1.6, label=reg.capitalize())
a.legend(frameon=False, title="Initial regime", loc="upper left")
cax = fig.add_axes([0.845, 0.135, 0.024, 0.82])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, cax=cax); cb.set_label(r"Initial radius $R_0$  (mm)")
cb.set_ticks(allR * 1e3); cb.ax.set_yticklabels([f"{v*1e3:.3f}" for v in allR])
cb.ax.tick_params(labelsize=7)
save(fig, "fig_radius_eta")
