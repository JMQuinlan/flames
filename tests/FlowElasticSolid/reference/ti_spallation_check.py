#!/usr/bin/env python3
"""
Spallation of a titanium plate hit by an aluminium plate at 700 m/s (tests/FlowElasticSolid/input_1D_TiSpallation)
against Favrie & Schmidmayer, CMAME 455 (2026) 118880, Fig. 6 (von Mises, 0.01 % air).

  python3 ti_spallation_check.py <run_dir> [--outdir DIR] [--fig 6|7]      (6: von Mises, the default; 7: GTN)

The published density and pressure profiles (0.2 ... 20 microseconds) were digitised from the vector figure
(literature/favrie_schmidmayer2026_fig6_*.dat).  Deck units: GPa, g/cm^3, cm, 10 microseconds.
Reported: peak pressure and shock positions while the waves are in the plates (0.2 - 1.2 microseconds), then the
solid pieces (runs of cells with density above half the titanium density) at 3.6, 10 and 20 microseconds: where the
intact plate ends and where the spall fragments are, for the run and for the paper.
There is no exact solution; the paper's model also differs (finite-rate pressure relaxation, three volume
fractions), so the fragment pattern is a comparison of two models, not a verification.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
HERE = os.path.dirname(os.path.abspath(__file__))
TIMES = ["0.2", "0.4", "0.8", "1.2", "2.0", "3.6", "10", "20"]          # microseconds
RHO_TI = 4527.0


FIG = "6"


def lit(q, t):
    return np.loadtxt(os.path.join(HERE, "literature", f"favrie_schmidmayer2026_fig{FIG}_{q}_t{t}us.dat"))


def pieces(x, rho, thr=0.5 * RHO_TI):
    """[(x_start, x_end)] of the runs with rho > thr."""
    s = rho > thr; out = []; i = 0
    while i < len(s):
        if s[i]:
            j = i
            while j + 1 < len(s) and s[j + 1]: j += 1
            out.append((x[i], x[j])); i = j + 1
        else: i += 1
    return out


def main():
    global FIG
    args = sys.argv[1:]
    if "--fig" in args: FIG = args[args.index("--fig") + 1]
    times = TIMES if FIG in ("6", "7") else ["0.2", "0.4", "0.8", "1.2", "2.0", "2.8", "3.6", "10", "20"]
    case = {"6": "von Mises, 0.01 % Air", "7": "GTN, 0.01 % Air", "8": "GTN, 1 % Air", "9": "GTN, 10 % Air"}[FIG]
    skip = {args.index(k) + 1 for k in ("--fig", "--outdir") if k in args}
    d = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip][0]
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(HERE, "Images", os.path.basename(os.path.normpath(d)))
    os.makedirs(outdir, exist_ok=True)
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    tt = []
    for pf in pfs:
        with open(os.path.join(pf, "Header")) as fh:
            L = fh.read().split("\n"); nc = int(L[1]); tt.append(float(L[2 + nc + 1]))
    tt = np.array(tt)
    R = {}
    for t in times:
        k = int(np.argmin(np.abs(tt - float(t) / 10.0)))
        if abs(tt[k] - float(t) / 10.0) > 0.011:        # more than half a plot interval away: the run did not reach this time
            sys.exit(f"{d}: no plot file at t = {t} microseconds (the run ends at {10 * tt.max():.2f}) -- incomplete run, not evaluated")
        ds = yt.load(pfs[k]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = int(ds.domain_dimensions[0]); Lx = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * Lx / n
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        R[t] = dict(t=10 * float(ds.current_time), x=x, rho=1e3 * q("density"), p=1e9 * q("pressure"), eta=q("eta"), u=1e3 * q("velocityx"))
    print(f"{d}: {len(R['20']['x'])} cells")
    print("  waves in the plates:      peak pressure [GPa] numerical / paper        rightmost point with p > 1 GPa [cm] numerical / paper")
    for t in ("0.2", "0.4", "0.8", "1.2"):
        r = R[t]; P = lit("p", t); f = lambda xx, pp: xx[pp > 1e9].max() if (pp > 1e9).any() else np.nan
        print(f"    t = {t:>4} us              {r['p'].max() / 1e9:6.3f} / {P[:, 1].max() / 1e9:6.3f}  ({100 * (r['p'].max() / P[:, 1].max() - 1):+5.1f} %)                  {f(r['x'], r['p']):.3f} / {f(P[:, 0], P[:, 1]):.3f}")
    print(f"  lowest pressure in the run at the plotted times: {min(r['p'].min() for r in R.values()) / 1e9:+.4f} GPa; paper {min(lit('p', t)[:, 1].min() for t in times) / 1e9:+.4f} GPa")
    rows = []
    for t in ("3.6", "10", "20"):
        r = R[t]; Lr = lit("rho", t); a = pieces(r["x"], r["rho"]); b = pieces(Lr[:, 0], Lr[:, 1])
        fmt = lambda pc: ", ".join(f"{s:.3f}-{e:.3f}" for s, e in pc)
        print(f"  t = {t:>4} us  solid pieces [cm]   numerical: {fmt(a)}")
        print(f"                                    paper:     {fmt(b)}")
        rows.append((t, a, b))
    up = os.path.join(HERE, "literature", f"favrie_schmidmayer2026_fig{FIG}_uprobe.dat")
    if os.path.exists(up):
        # Lagrangian probe on the rear face of the titanium plate: integrate dx/dt = u(x, t) from x = 1.28 - one cell
        U = np.loadtxt(up); xp = 1.28 - 2.0 * (R[times[0]]["x"][1] - R[times[0]]["x"][0]); hist = []; tp = 0.0
        for k, pf in enumerate(pfs):
            ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            n = int(ds.domain_dimensions[0]); Lx = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * Lx / n
            u = np.asarray(cg[("boxlib", "velocityx")])[:, 0, 0]; t = float(ds.current_time)
            ui = float(np.interp(xp, x, u)); hist.append((10.0 * t, 1e3 * ui)); xp += ui * (tt[k + 1] - t) if k + 1 < len(pfs) else 0.0
        hist = np.array(hist); np.savetxt(os.path.join(outdir, "rear_face_velocity.txt"), hist, header="t [microseconds], u [m/s] of a Lagrangian probe starting two cells inside the rear face")
        print(f"  rear-face probe: peak {hist[:, 1].max():.0f} m/s at {hist[np.argmax(hist[:, 1]), 0]:.2f} us (paper {U[:, 1].max():.0f} at {U[np.argmax(U[:, 1]), 0]:.2f}); at 20 us {hist[-1, 1]:.0f} m/s (paper {U[-1, 1]:.0f})")
        figp, axp = plt.subplots(figsize=(5.4, 4.2))
        axp.plot(U[:, 0], U[:, 1], "C3-", lw=1.5, label="(Favrie 2026)"); axp.plot(hist[:, 0], hist[:, 1], "ko", ms=3.5, mfc="none", label="Numerical")
        axp.set_xlabel("Time [$\\mu$s]"); axp.set_ylabel("Velocity [m/s]"); axp.set_title(f"Rear-Face Velocity of the Titanium Plate ({case})"); axp.legend(); axp.grid(alpha=0.3)
        figp.tight_layout(); figp.savefig(os.path.join(outdir, "ti_rear_face_velocity.png"), dpi=200); plt.close(figp)
    with open(os.path.join(outdir, "ti_spallation_summary.csv"), "w") as fo:
        fo.write("t_us,source,piece,x_start_cm,x_end_cm\n")
        for t, a, b in rows:
            for src, pc in (("numerical", a), ("paper", b)):
                for i, (s, e) in enumerate(pc): fo.write(f"{t},{src},{i},{s:.4f},{e:.4f}\n")

    show = ["0.4", "1.2", "2.0", "3.6", "10", "20"]
    fig, ax = plt.subplots(2, len(show), figsize=(3.6 * len(show), 6.6), sharex=True)
    for c, t in enumerate(show):
        r = R[t]; Lr, Lp = lit("rho", t), lit("p", t); st = max(1, len(r["x"]) // 400)
        for row, (key, L, sc, lab) in enumerate((("rho", Lr, 1.0, "Density [kg/m$^3$]"), ("p", Lp, 1e-9, "Pressure [GPa]"))):
            a = ax[row, c]
            a.axvspan(0.1, 0.3, color="0.6", alpha=0.10, lw=0); a.axvspan(0.3, 1.28, color="C1", alpha=0.07, lw=0)   # initial aluminium / titanium
            a.plot(L[:, 0], sc * L[:, 1], "C3-", lw=1.3, label="(Favrie 2026)")
            a.plot(r["x"][::st], sc * r[key][::st], "k-", lw=0.8, marker="o", ms=2.2, mfc="none", markevery=3, label="Numerical")
            a.grid(alpha=0.3)
            if c == 0: a.set_ylabel(lab)
            if row == 0: a.set_title(f"t = {t} $\\mu$s")
            else: a.set_xlabel("Position [cm]")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Spallation of a Titanium Plate Hit by an Aluminium Plate at 700 m/s ({case})"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "ti_spallation.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/ti_spallation.png")


if __name__ == "__main__":
    main()
