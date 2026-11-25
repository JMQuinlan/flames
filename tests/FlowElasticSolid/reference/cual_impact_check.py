#!/usr/bin/env python3
"""
Impact of an aluminium plate on a copper plate at 600 m/s, elastic and elasto-plastic
(tests/FlowElasticSolid/input_1D_CuAlImpact) against Fig. 4 of Favrie & Schmidmayer (2026).

  python3 cual_impact_check.py <plastic_run_dir> [<elastic_run_dir>] [--outdir DIR]

Reference curves: literature/favrie_schmidmayer2026_fig4_<rho|p|s11|u>_<plastic|elastic>.dat -- the vector paths of
the published figure (read from the PDF, axes calibrated on the tick marks), t = 50 microseconds.
Also reported, independent of the paper: the elastic-run plateau against the exact two-shock solution of the linear-
elastic impact is NOT used (finite strain); instead the plateau values (p, sigma_11, u in each plate), the positions of
the elastic and plastic fronts, and the von Mises stress behind the plastic fronts against the yield stress.
Units of the run: g/cm^3, km/s, GPa, m, ms  -> converted to SI for the comparison.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
HERE = os.path.dirname(os.path.abspath(__file__)); LIT = os.path.join(HERE, "literature")
REF = "(Favrie and Schmidmayer 2026)"
Q = dict(rho=("density", 1e3, r"Density [kg/m$^3$]", "Density"), p=("pressure", 1e9, "Pressure [Pa]", "Pressure"),
         s11=(None, 1e9, r"Normal Stress $\sigma_{11}$ [Pa]", "Normal Stress"), u=("velocityx", 1e3, "Velocity [m/s]", "Velocity"))


def meta(d, key):
    val = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def metalist(d, key):
    out = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*(?:\(nvals\s*=\s*\d+\)\s*::\s*\[(.*)\]|=\s*(.*))$", ln)
        if m:
            try: out = [float(v) for v in (m.group(1) or m.group(2)).replace(",", " ").split()]
            except ValueError: pass
    return out


def material(d, side):
    """(mu, sigma_y, gamma, p_inf) in GPa of 'copper' / 'aluminium': phases 1 / 0 of the two-phase deck, or
    species Cu / Al (index 0 / 1) of the solid phase in input_1D_CuAlImpact_Species."""
    L = metalist(d, "elastic.species.mu")
    if L:
        i = 0 if side == "copper" else 1
        return L[i], metalist(d, "elastic.species.sigma_y")[i], metalist(d, "species.phase0.gamma")[i], metalist(d, "species.phase0.pinf")[i]
    k = 1 if side == "copper" else 0
    return meta(d, f"elastic.mu{k}"), meta(d, f"elastic.sigma_y{k}"), meta(d, f"eos{k}.gamma"), meta(d, f"eos{k}.p0")


def load(d):
    pf = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    f = dict(t=float(ds.current_time), x=x, rho=q("density") * 1e3, p=q("pressure") * 1e9, u=q("velocityx") * 1e3, s11=(q("elastic_Sxx") - q("pressure")) * 1e9, eta=q("eta"))
    if ("boxlib", "plastic_seq") in ds.field_list: f["seq"] = q("plastic_seq") * 1e9
    return f


def lit(name, case):
    a = np.loadtxt(os.path.join(LIT, f"favrie_schmidmayer2026_fig4_{name}_{case}.dat")); return a[:, 0], a[:, 1]


def fronts(x, y, lo, hi):
    """positions where y crosses the mid-levels of its plateaux, scanning left to right"""
    d = np.abs(np.diff(y)); thr = 0.02 * (y.max() - y.min()); k = np.where(d > thr)[0]
    if len(k) == 0: return []
    groups = np.split(k, np.where(np.diff(k) > 8)[0] + 1)
    return [float(x[g].mean()) for g in groups if lo < x[g].mean() < hi]


def main():
    args = sys.argv[1:]
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    if outdir is None: outdir = os.path.join(HERE, "Images", os.path.basename(os.path.normpath(dirs[0])))
    os.makedirs(outdir, exist_ok=True)
    runs = {}
    for d in dirs:
        runs["plastic" if (meta(d, "elastic.plastic") or 0) > 0 else "elastic"] = (d, load(d))
    rows = []
    for case, (d, f) in runs.items():
        x = f["x"]; print(f"{d}  ({case}), t = {f['t'] * 1e3:.2f} microseconds, {len(x)} cells")
        for name in ("p", "s11", "u", "rho"):
            xl, yl = lit(name, case); yr = np.interp(x, xl, yl)
            # plateau values: copper side 0.40-0.48, aluminium side 0.52-0.60 (behind every front in both runs)
            for side, (a, b) in (("copper", (0.40, 0.48)), ("aluminium", (0.52, 0.60))):
                m = (x > a) & (x < b); v, r = f[name][m].mean(), yr[m].mean()
                rows.append((case, name, side, v, r)); 
            if name in ("p", "u"):
                print(f"  {Q[name][3]:>13}: copper side {rows[-2][3]:.5g}  paper {rows[-2][4]:.5g}  ({100 * (rows[-2][3] / rows[-2][4] - 1):+.2f} %);"
                      f"   aluminium side {rows[-1][3]:.5g}  paper {rows[-1][4]:.5g}  ({100 * (rows[-1][3] / rows[-1][4] - 1):+.2f} %)")
            elif name == "s11":
                print(f"  {Q[name][3]:>13}: {rows[-2][3]:.5g} / {rows[-1][3]:.5g}  paper {rows[-2][4]:.5g} / {rows[-1][4]:.5g}  ({100 * (rows[-2][3] / rows[-2][4] - 1):+.2f} % / {100 * (rows[-1][3] / rows[-1][4] - 1):+.2f} %)")
            else:
                print(f"  {Q[name][3]:>13}: {rows[-2][3]:.5g} / {rows[-1][3]:.5g}  paper {rows[-2][4]:.5g} / {rows[-1][4]:.5g}  ({100 * (rows[-2][3] / rows[-2][4] - 1):+.2f} % / {100 * (rows[-1][3] / rows[-1][4] - 1):+.2f} %)")
        xl, yl = lit("s11", case)
        fn, fp = fronts(x, f["s11"], 0.05, 0.95), fronts(xl, yl, 0.05, 0.95) if len(xl) > 500 else None
        print("  stress fronts at x = " + ", ".join(f"{v:.4f}" for v in fn))
        # the paper's polyline: corner points with a large jump
        jl = np.where(np.abs(np.diff(yl)) > 0.02 * (yl.max() - yl.min()))[0]
        gl = np.split(jl, np.where(np.diff(jl) > 1)[0] + 1) if len(jl) else []
        print("  paper's fronts at  x = " + ", ".join(f"{0.5 * (xl[g[0]] + xl[g[-1] + 1]):.4f}" for g in gl))
        if "seq" in f:
            # Hugoniot elastic limit (uniaxial strain, von Mises): sigma_HEL = sigma_y (K + 4 mu/3)/(2 mu), K = gamma (p + p_inf);
            # measured on the plateau between the elastic precursor and the plastic shock
            for side, k, (a, b) in (("copper", 1, (fn[0] - 0.045, fn[0] - 0.015)), ("aluminium", 0, (fn[-1] + 0.010, fn[-1] + 0.020))):
                mu_, sy_, g_, pi_ = material(d, side); K_ = g_ * pi_
                hel = sy_ * (K_ + 4.0 * mu_ / 3.0) / (2.0 * mu_) * 1e9; m = (x > a) & (x < b)
                print(f"  elastic precursor in the {side}: |sigma_11| = {-f['s11'][m].mean() / 1e6:.1f} MPa   Hugoniot elastic limit {hel / 1e6:.1f} MPa  ({100 * (-f['s11'][m].mean() / hel - 1):+.1f} %)")
            for side, (a, b), sy in (("copper", (0.40, 0.48), material(d, "copper")[1]), ("aluminium", (0.52, 0.60), material(d, "aluminium")[1])):
                m = (x > a) & (x < b); print(f"  von Mises stress behind the plastic shock, {side}: {f['seq'][m].mean() / 1e6:.2f} MPa  (yield {sy * 1e3:.0f} MPa, {100 * (f['seq'][m].mean() / (sy * 1e9) - 1):+.2f} %)")
    with open(os.path.join(outdir, "cual_impact_summary.csv"), "w") as fo:
        fo.write("case,quantity,side,numerical,paper,diff_pct\n")
        for c, n, s, v, r in rows: fo.write(f"{c},{n},{s},{v:.8g},{r:.8g},{100 * (v / r - 1):.4f}\n")

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.2)); col = dict(elastic="C3", plastic="C0"); lab = dict(elastic="No Plasticity", plastic="Plasticity")
    for a, name in zip(ax.ravel(), ("rho", "p", "s11", "u")):
        for case, (d, f) in runs.items():
            xl, yl = lit(name, case); st = max(1, len(f["x"]) // 110)
            a.plot(xl, yl, "-", color=col[case], lw=1.4, label=f"{lab[case]} {REF}")
            a.plot(f["x"][::st], f[name][::st], "o", color=col[case], ms=3.2, mfc="none", mew=0.7, label=f"{lab[case]}, Numerical")
        a.axvspan(0, 0.5, color="#B87333", alpha=0.08, lw=0); a.axvspan(0.5, 1, color="#C0C0C0", alpha=0.15, lw=0)
        a.set_xlim(0, 1); a.set_xlabel("Position [m]"); a.set_ylabel(Q[name][2]); a.set_title(Q[name][3]); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Impact of an Aluminum Plate on a Copper Plate at 600 m/s, Elastic and Elasto-Plastic, at 50 Microseconds"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cual_impact.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/cual_impact.png")


if __name__ == "__main__":
    main()
