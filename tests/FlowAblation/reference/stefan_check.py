#!/usr/bin/env python3
"""
1D Stefan problem for the ablating wall (tests/FlowAblation/input_1D_Stefan).

  python3 stefan_check.py <run_dir> [--outdir DIR] [--qstar Q*] [--twall T_w]   (if the run overrides ablation.*)
  python3 stefan_check.py --paper <run_dir> [<run_dir> ...] [--outdir DIR] [--name NAME] [--qstar Q*] [--twall T_w]
          [--times 1,2,4,7.2] [--series2 dirA,dirB,... --twall2 T_w --qstar2 Q*]
      --series2: a second resolution series (e.g. dT/T = 0.1 %) drawn in the convergence panel only.
      paper-ready figure (recession, wall heat flux, temperature profile of the finest run; error-versus-cells
      panel when several resolutions are given) + error tables.  [2026-10-29]

One-phase Stefan similarity solution (gas half-space at T_inf, wall at T_w, rho_s Q* ds/dt = q_w):
    s(t) = 2 lam sqrt(alpha t),   lam exp(lam^2) (1 + erf lam) = rho cp (T_inf - T_w)/(sqrt(pi) rho_s Q*)
    T(x, t) = T_w + (T_inf - T_w) (erf((x - x0)/(2 sqrt(alpha t))) + erf lam)/(1 + erf lam)
    q_w(t) = k (T_inf - T_w) exp(-lam^2)/((1 + erf lam) sqrt(pi alpha t))
Surface position in the run = the phi = 0.5 crossing (linear interpolation between the two cells that bracket it).
--paper also prints the recession from the solid "area", s_area = sum(phi) dx - (1 - x0) = integral of the
recession speed (the extended level-set phi spreads over several cells while the 0.5 crossing stays sharp; the
gas sees the wall at the 0.5 crossing, so that is the measure used in the figure).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
RHO, CP, MU, PR, TINF, TW, RHOS, QSTAR, X0 = 1.0, 3.5, 1.0e-3, 0.72, 1.0, 0.9, 1.0, 1.757, 0.3


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    k = MU * CP / PR; al = k / (RHO * CP)
    QSTAR = float(sys.argv[sys.argv.index("--qstar") + 1]) if "--qstar" in sys.argv else globals()["QSTAR"]
    TW = float(sys.argv[sys.argv.index("--twall") + 1]) if "--twall" in sys.argv else globals()["TW"]
    lam = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * (TINF - TW) / (np.sqrt(np.pi) * RHOS * QSTAR), 1e-6, 5)
    H = []; prof = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); Lf = ds.index.max_level; cg = ds.covering_grid(Lf, ds.domain_left_edge, ds.domain_dimensions * 2 ** Lf)   # finest level (AMR)
        n = ds.domain_dimensions[0] * 2 ** Lf; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1])
        H.append((t, X0 - xs, q("ablation_q").max(), np.abs(q("velocityx")).max(), q("pressure")[phi > 0.5].min()))
        prof = (t, x, q("T"), phi)
    H = np.array(H); t = H[:, 0]
    s_ex = 2 * lam * np.sqrt(al * t); q_ex = k * (TINF - TW) * np.exp(-lam**2) / ((1 + erf(lam)) * np.sqrt(np.pi * al * t))
    print(f"{d}: k = {k:.4e}, alpha = {al:.4e}, lam = {lam:.5f}; {len(H)} plot files to t = {t[-1]:.2f}")
    for tt in (1.0, 2.0, 4.0, t[-1]):
        j = np.argmin(np.abs(t - tt))
        print(f"  t = {t[j]:5.2f}: recession s = {H[j, 1]:.5f}   exact {s_ex[j]:.5f}   ({100 * (H[j, 1] / s_ex[j] - 1):+.1f} %);"
              f"   wall heat flux {H[j, 2]:.4e}   exact {q_ex[j]:.4e}   ({100 * (H[j, 2] / q_ex[j] - 1):+.1f} %)")
    tp, x, T, phi = prof; fl = phi > 0.5
    Tex = TW + (TINF - TW) * (erf((x - X0) / (2 * np.sqrt(al * tp))) + erf(lam)) / (1 + erf(lam))
    print(f"  temperature profile at t = {tp:.2f}: max error {np.abs(T - Tex)[fl].max():.2e} of dT = {TINF - TW:g}  ({100 * np.abs(T - Tex)[fl].max() / (TINF - TW):.1f} %)")
    print(f"  max gas speed over the run {H[:, 3].max():.2e}; min pressure {H[:, 4].min():.4f}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(t, H[:, 1], "ko", ms=3, label="Hydro2"); ax[0].plot(t, s_ex, "C3-", label="Stefan solution"); ax[0].set_ylabel("surface recession s"); ax[0].set_xlabel("t")
    ax[1].loglog(t, H[:, 2], "ko", ms=3); ax[1].loglog(t, q_ex, "C3-"); ax[1].set_ylabel(r"wall heat flux $q_w$"); ax[1].set_xlabel("t")
    ax[2].plot(x[fl], T[fl], "k-", lw=1.5, label="Hydro2"); ax[2].plot(x[fl], Tex[fl], "C3--", label="Stefan solution")
    ax[2].axvspan(0, x[fl][0], color="0.8"); ax[2].set_xlim(0.2, 0.8); ax[2].set_ylabel("T"); ax[2].set_xlabel(f"x   (t = {tp:.2f}; grey = solid)")
    for a in ax: a.grid(alpha=0.3)
    ax[0].legend(); ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "stefan.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/stefan.png")


def load(d):
    """Histories of one run: t, s (phi = 0.5 crossing), s_area, q_w, and the last temperature profile."""
    H = []; prof = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); Lf = ds.index.max_level; cg = ds.covering_grid(Lf, ds.domain_left_edge, ds.domain_dimensions * 2 ** Lf)   # finest level (AMR)
        n = int(ds.domain_dimensions[0]) * 2 ** Lf; L = float(ds.domain_width[0]); x = (np.arange(n) + 0.5) * L / n; t = float(ds.current_time)
        if t <= 1e-2: continue
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1])
        H.append((t, X0 - xs, phi.sum() * L / n - (L - X0), q("ablation_q").max()))
        prof = (t, x, q("T"), phi)
    return n, np.array(H), prof


def paper():
    args = sys.argv[1:]; opt = lambda key, dflt: args[args.index(key) + 1] if key in args else dflt
    runs = []; skip = False
    for a in args:
        if skip: skip = False; continue
        if a in ("--outdir", "--name", "--qstar", "--twall", "--times", "--series2", "--twall2", "--qstar2"): skip = True; continue
        if a.startswith("--"): continue
        runs.append(a)
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = opt("--outdir", os.path.join(here, "Images", "R29")); name = opt("--name", "stefan_paper")
    os.makedirs(outdir, exist_ok=True)
    Qs = float(opt("--qstar", QSTAR)); Tw = float(opt("--twall", TW)); dT = TINF - Tw
    k = MU * CP / PR; al = k / (RHO * CP)
    lam = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * dT / (np.sqrt(np.pi) * RHOS * Qs), 1e-6, 5)
    s_ex = lambda t: 2 * lam * np.sqrt(al * t)
    q_ex = lambda t: k * dT * np.exp(-lam**2) / ((1 + erf(lam)) * np.sqrt(np.pi * al * t))
    T_ex = lambda x, t: Tw + dT * (erf((x - X0) / (2 * np.sqrt(al * t))) + erf(lam)) / (1 + erf(lam))
    R = sorted([load(d) + (d,) for d in runs], key=lambda r: r[0])
    tend = R[-1][1][-1, 0]
    times = [float(v) for v in opt("--times", f"1,2,4,{tend:.6g}").split(",")]
    print(f"One-phase Stefan: T_w = {Tw:g} (dT/T = {dT / TINF:g}), Q* = {Qs:g}, k = {k:.4e}, alpha = {al:.4e}, lam = {lam:.5f}")
    print(f"  s_exact(t_end = {tend:.2f}) = {s_ex(tend):.5f}")
    rows = []
    for n, H, prof, d in R:
        t = H[:, 0]; es = 100 * (H[:, 1] / s_ex(t) - 1); ea = 100 * (H[:, 2] / s_ex(t) - 1); eq = 100 * (H[:, 3] / q_ex(t) - 1)
        tp, x, T, phi = prof; fl = phi > 0.5; eT = 100 * np.abs(T - T_ex(x, tp))[fl].max() / dT
        late = t >= 1.0 - 1e-9
        print(f"\n  {d}: N = {n}, dx = {1.0 / n:.3e}, {len(t)} plot files to t = {t[-1]:.2f}")
        print(f"    {'t':>6} {'s/dx':>6} {'s (phi=0.5)':>11} {'err %':>7} {'s_area':>9} {'err %':>7} {'q_w':>10} {'err %':>7}")
        for tt in times:
            j = int(np.argmin(np.abs(t - tt)))
            print(f"    {t[j]:6.2f} {s_ex(t[j]) * n:6.1f} {H[j, 1]:11.5f} {es[j]:+7.2f} {H[j, 2]:9.5f} {ea[j]:+7.2f} {H[j, 3]:10.4e} {eq[j]:+7.2f}")
        l2 = t >= 2.0 - 1e-9
        print(f"    wall heat flux error, t >= 1: {eq[late].min():+.2f} .. {eq[late].max():+.2f} % (isolated spikes when the surface uncovers a cell);"
              f"  t >= 2: median {np.median(eq[l2]):+.2f} %, max |err| {np.abs(eq[l2]).max():.2f} %")
        print(f"    T-profile max error at t = {tp:.2f}: {eT:.3f} % of dT")
        rows.append((n, es[-1], ea[-1], eq[-1], eT, np.abs(eq[late]).max(), np.abs(es[late]).max(), np.median(eq[l2])))
    rows = np.array(rows)
    if len(rows) > 1:
        print("\n  convergence at t_end (error %):   N    s(phi=0.5)   s_area    q_w    T(%dT)   q_w median(t>=2)   observed order of |err s|")
        for i, r in enumerate(rows):
            p = "" if i == 0 else f"{np.log(abs(rows[i - 1, 1]) / abs(r[1])) / np.log(r[0] / rows[i - 1, 0]):.2f}"
            print(f"  {int(r[0]):35d} {r[1]:+10.3f} {r[2]:+9.3f} {r[3]:+7.3f} {r[4]:8.3f}   {r[7]:+10.3f}        {p}")
    # ---- figure (finest run) ----
    n, H, prof, d = R[-1]; t = H[:, 0]; tp, x, T, phi = prof; fl = phi > 0.5
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 13,
                         "xtick.labelsize": 13, "ytick.labelsize": 13})
    if len(R) > 1:
        fig, ax = plt.subplots(2, 2, figsize=(11, 9)); ax = ax.ravel()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    me = max(1, len(t) // 12); tt = np.linspace(1e-3, tend, 400)
    num = dict(color="C3", lw=1.0, marker="o", mfc="none", ms=6, label="Numerical")
    ax[0].plot(tt, s_ex(tt), "k-", lw=1.8, label="Exact"); ax[0].plot(t, H[:, 1], markevery=me, **num)
    ax[0].set_xlabel("Time, $t$"); ax[0].set_ylabel("Surface Recession, $s$"); ax[0].set_title("Surface Recession")
    ax[1].plot(tt[tt > 0.1], q_ex(tt[tt > 0.1]), "k-", lw=1.8, label="Exact"); ax[1].plot(t, H[:, 3], markevery=me, **num)
    ax[1].set_xlabel("Time, $t$"); ax[1].set_ylabel("Wall Heat Flux, $q_w$"); ax[1].set_title("Wall Heat Flux")
    ax[1].set_xlim(0, tend * 1.02); ax[1].set_ylim(0, 1.1 * q_ex(max(t[0], 0.2)))
    xe = np.linspace(X0 - s_ex(tp), 0.8, 400); xf = x[fl]; sel = xf <= 0.8
    ax[2].axvspan(0.2, X0 - s_ex(tp), color="#E0E0E0", lw=0, zorder=0)
    ax[2].plot(xe, T_ex(xe, tp), "k-", lw=1.8, label="Exact")
    ax[2].plot(xf[sel], T[fl][sel], markevery=max(1, sel.sum() // 15), **num)
    ax[2].set_xlim(0.2, 0.8); ax[2].set_xlabel("Position, $x$"); ax[2].set_ylabel("Temperature, $T$")
    ax[2].set_title(f"Temperature Profile, $t$ = {tp:.1f}")
    if Tw > 0.98: ax[2].ticklabel_format(axis="y", useOffset=False)
    for a in ax[:3]: a.grid(alpha=0.3); a.legend(loc="best", frameon=False)
    if len(R) > 1:
        N = rows[:, 0]
        from matplotlib.ticker import FixedLocator, NullFormatter
        lab = lambda v: f"{100 * v:g} %"
        ax[3].loglog(N, np.abs(rows[:, 1]), "C3o-", mfc="none", ms=8, lw=1.0, label=f"Recession, $\\Delta T/T$ = {lab(dT / TINF)}")
        if "--series2" in args:
            Tw2 = float(opt("--twall2", Tw)); Q2 = float(opt("--qstar2", Qs)); dT2 = TINF - Tw2
            lam2 = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * dT2 / (np.sqrt(np.pi) * RHOS * Q2), 1e-6, 5)
            S2 = []
            for d2 in opt("--series2", "").split(","):
                n2, H2, _ = load(d2); j = int(np.argmin(np.abs(H2[:, 0] - tend)))
                S2.append((n2, 100 * (H2[j, 1] / (2 * lam2 * np.sqrt(al * H2[j, 0])) - 1)))
            S2 = np.array(sorted(S2)); print("\n  series 2 (dT/T = %g): recession error at t_end [%%]: " % dT2 + ", ".join(f"N = {int(a)}: {b:+.3f}" for a, b in S2)
                                                + "; observed orders " + ", ".join(f"{np.log(abs(S2[i - 1, 1] / S2[i, 1])) / np.log(S2[i, 0] / S2[i - 1, 0]):.2f}" for i in range(1, len(S2))))
            ax[3].loglog(S2[:, 0], np.abs(S2[:, 1]), "C0^-", mfc="none", ms=8, lw=1.0, label=f"Recession, $\\Delta T/T$ = {lab(dT2 / TINF)}")
        ax[3].loglog(N, np.abs(rows[:, 7]), "C3s--", mfc="none", ms=8, lw=1.0, label=f"Wall Heat Flux, $\\Delta T/T$ = {lab(dT / TINF)}")
        ax[3].loglog(N, np.abs(rows[0, 1]) * (N[0] / N), "k:", lw=1.2, label="First Order")
        ax[3].xaxis.set_major_locator(FixedLocator(N)); ax[3].xaxis.set_minor_formatter(NullFormatter())
        ax[3].set_xticklabels([str(int(v)) for v in N])
        ax[3].set_xlabel("Number of Cells, $N$"); ax[3].set_ylabel("Relative Error [%]"); ax[3].set_title("Grid Convergence")
        ax[3].grid(which="both", alpha=0.3); ax[3].legend(frameon=False, fontsize=10.5)
    fig.suptitle("One-Dimensional Stefan Problem", fontsize=16)
    fig.tight_layout()
    for ext in ("png", "eps"): fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"\n  figure run {d} (N = {n}): max |error| for t >= 1: recession {rows[-1, 6]:.2f} %, wall heat flux {rows[-1, 5]:.2f} %, "
          f"T profile {rows[-1, 4]:.3f} % of dT")
    print(f"  wrote {outdir}/{name}.png (.eps)")


if __name__ == "__main__":
    paper() if "--paper" in sys.argv else main()
