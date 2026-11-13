#!/usr/bin/env python3
"""
Mach 2 isothermal flat plate (tests/FlowAblation/input_2D_FlatPlate) against the EXACT compressible laminar
similarity solution (flat_plate_similarity.py: constant mu, Pr 0.72, gamma 1.4, M_e 2, T_w = T_e), with the
Eckert reference-temperature correlation of flat_plate_check.py shown for comparison.  flat_plate_check.py is
unchanged (old behaviour).

  python3 flat_plate_exact_check.py RUN [RUN ...] [--labels L1,L2,..] [--outdir DIR] [--plot RUN_INDEX] [--nonsimilar]

--nonsimilar  also march the non-similar boundary-layer equations (flat_plate_nonsimilar.py) with the wall pressure
              p_w(x) of the plotted run (fit 1 + a x^-1/2 + b over 0.1 < x < 0.95): the size of the displacement-
              induced-pressure effect that the zero-pressure-gradient similarity solution leaves out.  Drawn dotted
              in the error panel.

Each RUN is a run directory with output/*cell plot files (time average over the last third of them).
Numerical wall quantities, on the first fluid row j (wall face at y = 0, cell centre at dy/2):
  q_w  solver   : ablation_q, the conductive flux the scheme itself applies, k (T_w - T_1)/(dy/2)  (one-sided, O(dy))
  q_w  2nd order: k (9 (T_1 - T_w) - (T_2 - T_w)) / (3 dy) -- quadratic through T_w, T_1, T_2 (post-processing only)
  c_f  solver   : mu u_1/(dy/2) / (rho U^2/2)          (u_yy = 0 at a zero-pressure-gradient no-slip wall, so O(dy^2))
  c_f  2nd order: mu (9 u_1 - u_2)/(3 dy) / (rho U^2/2)
k = mu cp / Pr.  The figure uses the solver's own (conservative) wall flux and mu u_1/(dy/2).  The "2nd-order" heat
flux is a post-processing diagnostic only: on the EXACT profile the one-sided estimate is biased by -tau_w^2 dy/(4 mu)
(k T_yy = -mu u_y^2 at the wall; -6 % at x = 0.4 on 400 x 200), and in the discrete solution the near-wall
temperature adjusts so that the conservative flux itself is right, so the quadratic fit to T_w, T_1, T_2 over-reads.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flat_plate_similarity import flat_plate_constants, eckert_constants
yt.funcs.mylog.setLevel(40)
GAM, CP, MU, PR, RHO, TE, TW, MACH = 1.4, 3.5, 2.0e-4, 0.72, 1.0, 1.0, 1.0, 2.0
K = MU * CP / PR
U = MACH * np.sqrt(GAM * 1.0 / RHO)
STATIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
XFIT = (0.2, 0.9)


def load(d):
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    use = pfs[-max(1, len(pfs) // 3):]
    acc = {k: [] for k in ("q1", "q2", "c1", "c2", "pw", "Tw")}; times = []
    for pf in use:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); times.append(float(ds.current_time))
        nx, ny = ds.domain_dimensions[:2]
        x = float(ds.domain_left_edge[0]) + (np.arange(nx) + 0.5) * float(ds.domain_width[0]) / nx
        dy = float(ds.domain_width[1]) / ny
        y = float(ds.domain_left_edge[1]) + (np.arange(ny) + 0.5) * dy
        g = lambda f: np.asarray(cg[("boxlib", f)])[:, :, 0]
        T, phi, u, p, q = g("T"), g("phi"), g("velocityx"), g("pressure"), g("ablation_q")
        j = np.where(phi[nx // 2] >= 0.5)[0][0]
        assert abs(y[j] - 0.5 * dy) < 1e-9 * max(1, dy), "wall face is not at y = 0"
        acc["q1"].append(q[:, j])
        acc["q2"].append(K * (9 * (T[:, j] - TW) - (T[:, j + 1] - TW)) / (3 * dy))
        acc["c1"].append(MU * u[:, j] / y[j] / (0.5 * RHO * U * U))
        acc["c2"].append(MU * (9 * u[:, j] - u[:, j + 1]) / (3 * dy) / (0.5 * RHO * U * U))
        acc["pw"].append(p[:, j]); acc["Tw"].append(T[:, j])
    r = {k: np.mean(v, axis=0) for k, v in acc.items()}
    h = len(use) // 2   # steadiness: first vs second half of the averaging window
    r["drift_q"] = np.mean(acc["q1"][h:], axis=0) / np.mean(acc["q1"][:max(h, 1)], axis=0) - 1 if len(use) > 1 else 0 * x
    r.update(x=x, nx=nx, ny=ny, dx=float(ds.domain_width[0]) / nx, dy=dy, t0=times[0], t1=times[-1], nfiles=len(use))
    return r


def at(x, f, xs):
    return np.interp(xs, x, f)


def main():
    args = [a for a in sys.argv[1:]]
    opt = lambda k, dflt: (args[args.index(k) + 1] if k in args else dflt)
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = opt("--outdir", os.path.join(here, "Images", "R29"))
    VAL = ("--labels", "--outdir", "--plot")
    runs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in VAL)]
    labels = opt("--labels", ",".join(os.path.basename(os.path.normpath(r)) for r in runs)).split(",")
    iplot = int(opt("--plot", len(runs) - 1))
    os.makedirs(outdir, exist_ok=True)
    X = flat_plate_constants(MACH, GAM, PR, 0.0, TW); E = eckert_constants(MACH, GAM, PR, TW)
    print(f"exact similarity: c_f sqrt(Re_x) = {X['A_cf']:.5f}, q_w sqrt(Re_x)/(rho U cp T_e) = {X['A_q']:.5f}, T_aw = {X['Taw']:.4f}")
    print(f"Eckert (ref. T):  c_f sqrt(Re_x) = {E['A_cf']:.5f}, q_w sqrt(Re_x)/(rho U cp T_e) = {E['A_q']:.5f}, T_aw = {E['Taw']:.4f}")
    qex = lambda x: X["A_q"] * RHO * U * CP * TE / np.sqrt(RHO * U * x / MU)
    cex = lambda x: X["A_cf"] / np.sqrt(RHO * U * x / MU)
    qec = lambda x: E["A_q"] * RHO * U * CP * TE / np.sqrt(RHO * U * x / MU)
    cec = lambda x: E["A_cf"] / np.sqrt(RHO * U * x / MU)
    R = []
    for d, lab in zip(runs, labels):
        r = load(d); R.append(r); x = r["x"]
        print(f"\n{lab}: {r['nx']} x {r['ny']} (dx = {r['dx']:.5f}, dy = {r['dy']:.5f}); t = {r['t0']:.2f}..{r['t1']:.2f} ({r['nfiles']} files)")
        print("    x     Re_x |  q_w solver   2nd-ord |  c_f solver   2nd-ord |  vs Eckert: q_w(solver) c_f(solver) |  p_w/p_inf")
        L = x[-1] + 0.5 * r["dx"]
        for xs in [s_ for s_ in STATIONS if s_ < L]:
            e = lambda f, ref: 100 * (at(x, f, xs) / ref(xs) - 1)
            print(f"  {xs:4.2f} {RHO * U * xs / MU:7.0f} | {e(r['q1'], qex):+7.2f} %  {e(r['q2'], qex):+7.2f} % | {e(r['c1'], cex):+7.2f} %  {e(r['c2'], cex):+7.2f} %"
                  f" |       {e(r['q1'], qec):+7.2f} %    {e(r['c1'], cec):+7.2f} %   |  {at(x, r['pw'], xs):.4f}")
        m = (x > XFIT[0]) & (x < XFIT[1] * L)
        for k in ("q1", "q2", "c1", "c2"):
            n = np.polyfit(np.log(x[m]), np.log(np.abs(r[k][m])), 1)[0]
            ref = qex if k[0] == "q" else cex
            err = 100 * (r[k][m] / ref(x[m]) - 1)
            print(f"    {k}: exponent over {XFIT[0]} < x < {XFIT[1] * L:.2f}: {n:+.4f} (exact -0.5);  error there: max |.| {np.max(np.abs(err)):.2f} %, mean {err.mean():+.2f} %")
        print(f"    steadiness: q_w change between halves of the averaging window, max over 0.1<x<0.9: {100 * np.max(np.abs(r['drift_q'][(x > 0.1) & (x < 0.9)])):.3f} %")

    NS = None
    if "--nonsimilar" in args:
        from flat_plate_nonsimilar import march
        r = R[iplot]; x = r["x"]; m = (x > 0.1) & (x < 0.95)
        A = np.vstack([x[m] ** -0.5, np.ones(m.sum())]).T; a, b = np.linalg.lstsq(A, r["pw"][m] - 1, rcond=None)[0]
        xs = np.linspace(0.05, 1.0, 96)
        _, tau, qn = march(lambda z: 1 + a * z ** -0.5 + b, lambda z: -0.5 * a * z ** -1.5, xs, x0=0.02, nsteps=200)
        NS = dict(x=xs, eq=100 * (qn / qex(xs) - 1), ec=100 * (tau / (0.5 * RHO * U * U) / cex(xs) - 1))
        print(f"\nnon-similar boundary layer with the induced wall pressure of {labels[iplot]} (p_w = 1 + {a:.4f} x^-1/2 {b:+.4f}),"
              " relative to the zero-pressure-gradient exact solution:")
        for xs_ in STATIONS:
            print(f"  x = {xs_:4.2f}:  q_w {np.interp(xs_, NS['x'], NS['eq']):+6.2f} %   c_f {np.interp(xs_, NS['x'], NS['ec']):+6.2f} %")

    # ---------------- paper figure: 2 x 2, top = q_w, c_f (log-log), bottom = relative error per resolution ----------------
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "font.family": "serif", "mathtext.fontset": "cm"})
    r = R[iplot]; x = r["x"]; xx = np.geomspace(0.02, 1.0, 400)
    sel = np.unique(np.round(np.geomspace(np.searchsorted(x, 0.03), len(x) - 1, 22)).astype(int))   # markers evenly spaced in log x
    keep = [sel[0]]
    for k_ in sel[1:]:
        if np.log(x[k_] / x[keep[-1]]) > 0.12: keep.append(k_)
    sel = np.array(keep)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    for a, key, ex, ec, yl, tt in ((ax[0, 0], "q1", qex, qec, r"Wall Heat Flux, $q_w$", "Wall Heat Flux"),
                                   (ax[0, 1], "c1", cex, cec, r"Skin Friction Coefficient, $c_f$", "Skin Friction")):
        a.loglog(xx, ex(xx), "k-", lw=1.8, label="Exact (Compressible Similarity Solution)", zorder=2)
        a.loglog(xx, ec(xx), "--", color="0.55", lw=1.5, label="Reference Temperature (Eckert 1955)", zorder=1)
        a.loglog(x[sel], r[key][sel], "o", ls="none", ms=7, mfc="none", mec="C0", mew=1.4, label="Numerical", zorder=3)
        a.set_xlabel(r"Position, $x$"); a.set_ylabel(yl); a.set_xlim(0.02, 1); a.grid(alpha=0.3, which="both")
        a.set_ylim(0.85 * ex(1.0), 2.2 * ex(0.02))         # head-room so the legend sits above the data
        from matplotlib.ticker import LogLocator, FormatStrFormatter, NullFormatter
        a.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 4.0, 6.0))); a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        a.yaxis.set_minor_formatter(NullFormatter())
        a.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0))); a.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        a.xaxis.set_minor_formatter(NullFormatter())
        a.legend(loc="upper right", frameon=False); a.set_title(tt)
    mk = ["s", "^", "o", "D", "v"]
    for a, key, ex, nk, tt in ((ax[1, 0], "q1", qex, "eq", r"Heat Flux Error"), (ax[1, 1], "c1", cex, "ec", r"Skin Friction Error")):
        for i, (rr, lab) in enumerate(zip(R, labels)):
            e = 100 * (rr[key] / ex(rr["x"]) - 1)
            mev = (max(1, len(rr["x"]) // 20) // 2, max(1, len(rr["x"]) // 20))
            a.plot(rr["x"], e, "-", color=f"C{i}", lw=1.1, marker=mk[i % len(mk)], ms=7, mfc="none", mew=1.3, markevery=mev, label=f"Numerical, {lab}")
        if NS is not None:
            a.plot(NS["x"], NS[nk], ":", color="k", lw=2.0, label=r"Boundary Layer with Induced $p_w$")
        a.axhline(0, color="k", lw=0.8); a.axhspan(-1, 1, color="0.92", zorder=0)
        a.set_xlim(0, 1); a.set_ylim(-2, 6); a.grid(alpha=0.3)
        a.set_xlabel(r"Position, $x$"); a.set_ylabel(r"Error vs. Exact Solution [%]"); a.set_title(tt)
        a.legend(loc="upper right" if key == "q1" else "upper center", frameon=False, fontsize=11.5)
    fig.suptitle("Mach 2 Isothermal Flat Plate", fontsize=17)
    fig.tight_layout()
    for ext in ("png", "eps"):
        fig.savefig(os.path.join(outdir, f"flat_plate_exact.{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"\nwrote {outdir}/flat_plate_exact.png/.eps")

if __name__ == "__main__":
    main()
