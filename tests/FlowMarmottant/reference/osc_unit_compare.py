#!/usr/bin/env python3
"""
Coated-bubble unit test (tests/FlowMarmottant/input_osc_unit) against the coated Keller-Miksis model:
gas-volume radius from thermo.dat for several runs, and the bubble outline on the first z plane.

  python3 osc_unit_compare.py <run_dir> [<run_dir> ...] --labels "A;B;..." [--deck DECK] [--outdir DIR]
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "FlowRayleighPlesset", "reference")))


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def main():
    args = sys.argv[1:]; flags = ("--labels", "--deck", "--outdir"); skip = {args.index(k) + 1 for k in flags if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    labels = opt(args, "--labels", ";".join(os.path.basename(os.path.normpath(d)) for d in dirs)).split(";")
    deck = opt(args, "--deck", os.path.join(HERE, "..", "input_osc_unit"))
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "OscUnit")); os.makedirs(outdir, exist_ok=True)
    from sch20_analysis import Sch20Analysis
    A = Sch20Analysis(deck, os.path.join(dirs[0], "output"), case="Oscillating", coated=True, out_dir=os.path.join(outdir, "_km"))
    tk, Rk, _ = A.km; tc = A.tc; Rk = Rk / Rk[0]
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9)); mk = "os^Dv<>"
    rows = []
    for j, (d, lab) in enumerate(zip(dirs, labels)):
        a = np.genfromtxt(os.path.join(d, "output", "thermo.dat"), names=True); t = a["time"]; R = (a["gas_volume"] / a["gas_volume"][0]) ** (1 / 3)
        st = max(1, len(t) // 70); col = f"C{j}"
        for b in ax: b.plot(t / tc, R, "-", color=col, lw=0.8); b.plot(t[::st] / tc, R[::st], mk[j % 7], ms=4, mfc="none", mec=col, mew=0.8, label=lab)
        w = t < 3.5e-7; i = int(np.argmin(R[w])); m2 = (t > t[i]) & (t < 5.5e-7); jx = i + int(np.argmax(R[i:][m2[i:]])) if m2.any() else i
        rows.append((lab, t[-1], R[i], t[i], R[jx], t[jx], np.sqrt(np.mean((R - np.interp(t, tk, Rk)) ** 2)), R[-1]))
    for b in ax: b.plot(tk / tc, Rk, "k-", lw=1.8, label="Coated Keller-Miksis (Exact Model)", zorder=5)
    ax[0].set_xlim(0, 1.0e-6 / tc); ax[0].set_ylim(0.45, 1.03); ax[1].set_xlim(0.7, 2.2); ax[1].set_ylim(0.47, 0.72)
    for b in ax: b.set_xlabel(r"Time t / t$_c$ [-]"); b.set_ylabel(r"Gas-Volume Radius R / R(0) [-]"); b.grid(alpha=0.3)
    ax[0].set_title("Whole Run"); ax[1].set_title("Collapse and First Rebound"); ax[0].legend(fontsize=7.5)
    fig.suptitle("Coated Bubble (Marmottant Shell with Dilatational Viscosity), Unit Test at 6.4 Cells per Radius"); fig.tight_layout()
    o = os.path.join(outdir, "osc_unit_radius.png"); fig.savefig(o, dpi=200); plt.close(fig); print("wrote", o)
    k = int(np.argmin(Rk)); print(f"coated Keller-Miksis: minimum {Rk[k]:.4f} at {1e9 * tk[k]:.0f} ns; t_c = {1e9 * tc:.0f} ns")
    with open(os.path.join(outdir, "osc_unit_summary.csv"), "w") as fo:
        fo.write("run,t_end_ns,R_min,t_min_ns,R_rebound_max,t_rebound_ns,rms_R_minus_KM,R_end\n")
        for r in rows:
            fo.write(f"{r[0]},{1e9 * r[1]:.0f},{r[2]:.4f},{1e9 * r[3]:.0f},{r[4]:.4f},{1e9 * r[5]:.0f},{r[6]:.4f},{r[7]:.4f}\n")
            print(f"  {r[0]}: minimum {r[2]:.4f} at {1e9 * r[3]:.0f} ns, rebound maximum {r[4]:.4f} at {1e9 * r[5]:.0f} ns, rms (R - KM) {r[6]:.4f}, end {r[7]:.4f}")
    # outlines on the first z plane, for the runs that have plot files at several times
    import yt
    yt.funcs.mylog.setLevel(50)
    sets = []
    for d, lab in zip(dirs, labels):
        pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
        if len(pfs) >= 6: sets.append((lab, pfs))
    if sets:
        want = [2.0e-7, 3.0e-7, 4.0e-7, 6.0e-7, 8.0e-7, 1.0e-6]
        fig, ax = plt.subplots(len(sets), len(want), figsize=(2.5 * len(want), 2.75 * len(sets) + 0.5), squeeze=False)
        for r, (lab, pfs) in enumerate(sets):
            tt = []
            for pf in pfs:
                L = open(os.path.join(pf, "Header")).read().split("\n"); nc = int(L[1]); tt.append(float(L[2 + nc + 1]))
            for c, tw in enumerate(want):
                k = int(np.argmin(np.abs(np.array(tt) - tw))); ds = yt.load(pfs[k]); lev = ds.index.max_level
                n = int(ds.domain_dimensions[0]) * 2 ** lev; m = n // 5
                cg = ds.covering_grid(lev, ds.domain_left_edge, [m, m, 1]); e = np.asarray(cg[("boxlib", "eta")])[:, :, 0]; dx = float(ds.domain_width[0]) / n; R0 = 2.0e-6
                a = ax[r, c]; ext = m * dx / R0
                a.imshow(e.T, origin="lower", extent=(0, ext, 0, ext), cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
                xs = (np.arange(m) + 0.5) * dx / R0; a.contour(xs, xs, e.T, levels=[0.5], colors="k", linewidths=0.8)
                th = np.linspace(0, np.pi / 2, 60); rk = np.interp(tt[k], tk, Rk) * 1.0
                a.plot(rk * np.cos(th), rk * np.sin(th), "w--", lw=0.8)
                a.set_xlim(0, 1.1); a.set_ylim(0, 1.1); a.set_title(rf"t = {1e9 * tt[k]:.0f} ns", fontsize=8); a.tick_params(labelsize=7)
                if c == 0: a.set_ylabel(lab + "\n" + r"y / R$_0$ [-]", fontsize=7)
                if r == len(sets) - 1: a.set_xlabel(r"x / R$_0$ [-]", fontsize=8)
        fig.suptitle(r"Volume Fraction on the First z Plane (Numerical; Black: $\eta$ = 0.5; White Dashed: Coated Keller-Miksis Radius)", fontsize=10)
        fig.tight_layout(); o = os.path.join(outdir, "osc_unit_shapes.png"); fig.savefig(o, dpi=170); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
