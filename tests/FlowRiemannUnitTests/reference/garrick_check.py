#!/usr/bin/env python3
"""
Gas-water shock tube of Coralic & Colonius (2014) / Garrick et al. (2017) (tests/FlowRiemannUnitTests/input_Garrick),
Numerical against the exact two-material stiffened-gas Riemann solution.

  python3 garrick_check.py --out FIG_DIR "Label=run_dir" ...

Left (x < 0): air, rho 1.241, p 2.753, gamma 1.4;  right: water, rho 0.991, p 3.059e-4, gamma 5.5, pi 1.505; t = 0.2.
Errors: relative L1 over the domain, and the maximum relative error on the plateaus (points where the exact
solution is constant within +-0.05).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
sys.path.insert(0, os.path.expanduser("~/Desktop/flames/tests/FlowPhaseChange/reference"))
import co2_depressurization_check as C
GAS, WAT = dict(g=1.4, pi=0.0, cv=1.0), dict(g=5.5, pi=1.505, cv=1.0)
C.X0 = 0.0; TEND = 0.2


def load(d):
    pfs = sorted(glob.glob(os.path.join(d, "*cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    ds = yt.load(pfs[-1]); ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); j = ds.domain_dimensions[1] // 2
    q = lambda n: np.array(ad["boxlib", n])[:, j, 0]
    return dict(t=float(ds.current_time), x=np.array(ad["index", "x"])[:, j, 0], rho=q("density"), u=q("velocityx"), p=q("pressure"), eta=q("eta"))


def main():
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "."; os.makedirs(out, exist_ok=True)
    runs = [s.split("=", 1) for s in a if "=" in s and not s.startswith("--")]
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 12})
    xe = np.linspace(-1, 1, 20001); re, ue, pe, Te, ps, us = C.sg_riemann((1.241, 0.0, 2.753), (0.991, 0.0, 3.059e-4), GAS, WAT, xe, TEND)
    print("exact: p* = %.6f, u* = %.6f" % (ps, us))
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for k, (e, lab) in enumerate(((re, r"Density, $\rho$"), (ue, r"Velocity, $u$"), (pe, r"Pressure, $p$"))):
        ax[k].plot(xe, e, "k-", lw=2.0, label="Exact"); ax[k].set_ylabel(lab); ax[k].grid(alpha=0.3)
    for j, (lab, d) in enumerate(runs):
        r = load(d); ok = np.array([np.ptp(pe[(xe > xx - 0.05) & (xe < xx + 0.05)]) < 1e-9 and np.ptp(re[(xe > xx - 0.05) & (xe < xx + 0.05)]) < 1e-9 for xx in r["x"]])
        msg = []
        for k, (q, e) in enumerate((("rho", re), ("u", ue), ("p", pe))):
            ei = np.interp(r["x"], xe, e); L1 = np.abs(r[q] - ei).sum() / np.abs(ei).sum()
            sc = np.abs(ei[ok]) if q != "u" else np.full(ok.sum(), abs(us))
            mx = np.max(np.abs(r[q][ok] - ei[ok]) / np.maximum(sc, 1e-300))
            msg.append("%s L1 %.3f %% plateau max %.3f %%" % (q, 100 * L1, 100 * mx))
            ax[k].plot(r["x"], r[q], "-", color=f"C{j}", lw=1.0, marker="os^"[j % 3], markevery=(7 * j, max(1, len(r["x"]) // 40)), ms=6, mfc="none", mew=1.1, label="Numerical, " + lab)
        print("%-20s t = %.4f  " % (lab, r["t"]) + " | ".join(msg))
    ax[2].set_xlabel(r"Position, $x$"); ax[0].legend(); ax[0].set_title(r"Gas--Water Shock Tube at $t$ = %g" % TEND)
    fig.tight_layout()
    for ext in ("png", "eps"): fig.savefig(os.path.join(out, "garrick_comparison." + ext), dpi=200 if ext == "png" else None)
    print("wrote", os.path.join(out, "garrick_comparison.png"))


if __name__ == "__main__":
    main()
