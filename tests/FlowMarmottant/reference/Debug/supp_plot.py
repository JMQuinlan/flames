#!/usr/bin/env python3
"""Early-time behaviour of the 5 elastic (transition-region) cases.

The main sweep samples every 0.5 ms, by which point the tension field has
already destabilized.  This resolves the first 1.5 ms at 25 us to show
(a) how accurate sigma is before the instability bites, and
(b) the exponential growth rate of the sigma spread vs R0.
"""
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
import yt
yt.set_log_level(50)

HERE = os.path.dirname(os.path.abspath(__file__))
SUPP = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant/supp"
CHI, RB, SBRK, SIGW = 14.56, 0.018, 7.28, 7.28


def series(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*cell"))):
        ds = yt.load(f); ad = ds.all_data()
        eta = np.asarray(ad["boxlib", "eta"]); vol = np.asarray(ad["index", "cell_volume"])
        sig = np.asarray(ad["boxlib", "kappa2"]); gam = np.asarray(ad["boxlib", "Gamma"])
        x = np.asarray(ad["index", "x"]); y = np.asarray(ad["index", "y"])
        gas = (1 - eta) * vol; A = gas.sum()
        w = eta * (1 - eta); m = w > 0.05
        if not m.any():
            continue
        ww = w[m] * vol[m]; ww = ww / ww.sum()
        sm_ = float((ww * sig[m]).sum())
        st = float(np.sqrt((ww * (sig[m] - sm_) ** 2).sum()))
        rows.append(dict(t=float(ds.current_time), R=np.sqrt(A / np.pi),
                         sig=sm_, sig_std=st, gam=float((ww * gam[m]).sum()),
                         e4=abs((gas * np.exp(4j * np.arctan2(y, x))).sum()) / A))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else None


runs = []
for d in sorted(glob.glob(os.path.join(SUPP, "e_*"))) + sorted(glob.glob(os.path.join(SUPP, "r_*"))):
    if not os.path.isdir(d):
        continue
    R0 = float(os.path.basename(d).split("_")[1]) * 1e-4
    s = series(d)
    if s is None:
        continue
    s["R0"] = R0
    s["ruptured"] = os.path.basename(d).startswith("r_")
    runs.append(s)
    print(f"{os.path.basename(d)}  R0={R0:.4f}  n={len(s['t'])}")

R0s = np.array([r["R0"] for r in runs])
norm = mcolors.Normalize(R0s.min(), R0s.max()); cmap = cm.viridis
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": "#9aa3ad", "axes.linewidth": 0.8,
                     "grid.color": "#d7dce2", "grid.linewidth": 0.6, "grid.alpha": 0.7,
                     "xtick.color": "#4a525b", "ytick.color": "#4a525b"})
fig, A = plt.subplots(2, 2, figsize=(12.4, 8.6))
fig.subplots_adjust(left=0.075, right=0.895, top=0.90, bottom=0.085, hspace=0.33, wspace=0.24)
A = A.ravel()
for a in A:
    a.grid(True); a.set_axisbelow(True)

rates = []
for r in runs:
    c = cmap(norm(r["R0"]))
    sig0 = CHI * ((r["R0"] / RB) ** 2 - 1.0)
    sig0 = SIGW if sig0 >= SBRK else max(sig0, 0.0)
    ls = "--" if r.get("ruptured") else "-"
    t = r["t"] * 1e3
    A[0].plot(t, r["sig"], ls, lw=1.8, color=c)
    A[0].axhline(sig0, color=c, lw=0.6, alpha=0.4)
    A[1].plot(t, 100 * (r["sig"] / sig0 - 1), ls, lw=1.8, color=c)
    rel = np.where(np.abs(r["sig"]) > 1e-9, r["sig_std"] / np.abs(r["sig"]), np.nan)
    A[2].semilogy(t, np.maximum(100 * rel, 1e-3), ls, lw=1.8, color=c)
    # exponential fit over the growth window (0.1%..30% spread)
    g = np.isfinite(rel) & (rel > 1e-3) & (rel < 0.3)
    if g.sum() > 4:
        k = np.polyfit(r["t"][g], np.log(rel[g]), 1)[0]
        rates.append((r["R0"], k, bool(r.get("ruptured"))))
        A[2].plot(t[g], 100 * np.exp(np.polyval(np.polyfit(r["t"][g], np.log(rel[g]), 1), r["t"][g])),
                  "--", lw=1.0, color="#1d2329", alpha=0.5)
A[0].axhline(SIGW, color="#b2182b", lw=1.3, ls="-.", zorder=1)
A[0].text(0.012, SIGW, " $\\sigma_{water}$ = 7.28", color="#b2182b", fontsize=9.5,
          va="bottom", ha="left", fontweight=600)
A[0].set_xscale("log"); A[0].set_xlim(1e-2, 1.5); A[0].set_ylim(0, 8.2)
A[0].set_xlabel("t  (ms, log)"); A[0].set_ylabel("$\\sigma_{eff}$")
A[0].set_title("Tension vs time   (dashed = ruptured; log t to expose the plateau)", loc="left",
               fontsize=11, fontweight=600)
A[1].axhline(0, color="#5b6673", lw=1.0, ls="--")
A[1].set_ylim(-30, 10); A[1].set_xscale("log"); A[1].set_xlim(1e-2, 1.5)
A[1].set_xlabel("t  (ms, log)"); A[1].set_ylabel("$\\sigma$ error  (%)")
A[1].set_title("Deviation from the analytic tension", loc="left", fontsize=11, fontweight=600)
A[2].set_xscale("log"); A[2].set_xlim(1e-2, 1.5)
A[2].set_xlabel("t  (ms, log)"); A[2].set_ylabel("$\\sigma$ spread  (%, std/mean)")
A[2].set_title("Instability growth   (dashed = exponential fit)", loc="left",
               fontsize=11, fontweight=600)

el = [(R0, k) for R0, k, rup in rates if not rup]
ru = [(R0, k) for R0, k, rup in rates if rup]
if el:
    e = np.array(el)
    sig0 = CHI * ((e[:, 0] / RB) ** 2 - 1.0)
    q = np.mean(e[:, 1] / sig0 ** 2)
    xs = np.linspace(18.3, 22.2, 100)
    s0 = CHI * ((xs * 1e-3 / RB) ** 2 - 1.0)
    A[3].plot(xs, q * s0 ** 2 / 1e3, "-", lw=2.0, color="#1d2329", zorder=2,
              label=f"$k={q:.0f}\\,\\sigma_0^2$  (fit, elastic only)")
    A[3].plot(e[:, 0] * 1e3, e[:, 1] / 1e3, "o", ms=9, color="#2166ac",
              mec="white", mew=1.2, zorder=4, label="elastic")
if ru:
    rr = np.array(ru)
    A[3].plot(rr[:, 0] * 1e3, rr[:, 1] / 1e3, "s", ms=9, mfc="none", mew=1.8,
              color="#b2182b", zorder=4, label="ruptured (plateau first)")
    A[3].annotate("on the rupture plateau d$\\sigma$/d$\\Gamma$=0,\nso these do NOT follow $\\sigma_0^2$",
                  (rr[0, 0] * 1e3, rr[0, 1] / 1e3), textcoords="offset points",
                  xytext=(-12, 26), ha="right", fontsize=8, color="#b2182b")
A[3].set_yscale("log")
A[3].legend(frameon=False, fontsize=8.5, loc="lower right")
A[3].set_xlabel("R$_0$  (mm)"); A[3].set_ylabel("growth rate  (10$^3$ s$^{-1}$)")
A[3].set_title("Instability growth rate vs initial radius", loc="left",
               fontsize=11, fontweight=600)

cax = fig.add_axes([0.912, 0.20, 0.013, 0.58])
cb = fig.colorbar(sm, cax=cax); cb.set_label("R$_0$  (mm)", fontsize=10)
cb.set_ticks(R0s); cb.set_ticklabels([f"{v*1e3:.2f}" for v in R0s])
cb.ax.tick_params(labelsize=8)
fig.suptitle("Resolved early time — elastic (25 $\\mu$s) and ruptured (5 $\\mu$s) cases",
             fontsize=13.5, fontweight=600, x=0.075, ha="left", y=0.965)
out = os.path.join(HERE, "marmottant_supp.png")
fig.savefig(out, dpi=170, facecolor="white")
print("wrote", out)
for R0, k, rup in rates:
    print(f"  R0={R0*1e3:.1f} mm  {'RUPTURED' if rup else 'elastic '}  rate = {k:8.0f} /s   e-fold = {1e6/k:6.1f} us")
