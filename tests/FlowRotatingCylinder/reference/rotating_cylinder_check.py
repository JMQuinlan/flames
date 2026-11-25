#!/usr/bin/env python3
"""
Lift / drag of a spinning, non-translating cylinder in a uniform stream
(tests/FlowRotatingCylinder, solid.force_int = 1 -> <run_dir>/output_forces.dat).

  python3 rotating_cylinder_check.py <run_dir> [<run_dir> ...] [--outdir DIR] [--window W] [--re RE] [--kutta]

--re RE picks the Reynolds number of the reference data (default 100; Kang et al. give 40 / 60 / 100 / 160, the
Stojkovic et al. table and Fig. 13 are Re = 100 only and are left out otherwise).
--kutta adds the inviscid Kutta-Joukowski line Cl = -2 pi alpha to the lift panel and to the force history
(off by default).

The spin rate alpha = Omega R / U is read from "alpha<value>" in each run
directory name.  Counter-clockwise spin (Omega > 0) with the stream in +x
gives lift in -y.

Compared against
  * Kutta-Joukowski (inviscid, circulation Gamma = 2 pi R^2 Omega):
        Cl = -2 pi alpha            (Cd = 0)
    This is an upper bound: a viscous cylinder at Re = 100 only sheds part of
    the wall circulation into the outer flow.
  * Viscous 2D incompressible simulations at Re = 100:
    Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002), Table II and Fig. 13 (alpha up to 12) and
    Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999), Eq. (11) mean Cl = -2.48 alpha (alpha <= 1), Table I and
    Figs. 5, 8(b), 9 (also Re = 40, 60, 160).  Read from literature/*.dat (tables copied, figures digitized
    2026-10-05; see the file headers).  Shedding is suppressed above alpha ~ 1.8 in both.

Writes to <outdir> (default Images/ next to this script):
  rotating_force_history.png   Cl(t), Cd(t) for every run
  rotating_cl_cd_vs_alpha.png  mean Cl, Cd vs alpha with theory / literature
  rotating_cl_cd_phase.png     Cd - Cl phase diagram: the limit cycle of each run (outline only)
  rotating_summary.csv
q = 0.5 rho U^2 D with rho = 100, U = 1, D = 1.
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO, U, D = 100.0, 1.0, 1.0
SHED_CL_AMP = 1e-2
# Reference data: literature/*.dat (written by literature/make_literature_dat.py; tables copied from the papers,
# figures digitized 2026-10-05 -- each file's header gives the source and the uncertainty).  2D incompressible,
# alpha = surface speed / U (same definition as here), lift negative for counter-clockwise spin.
#   Stojkovic, Breuer & Durst, Phys. Fluids 14, 3160 (2002): Table II and Fig. 13 (Re = 100, alpha up to 12)
#   Kang, Choi & Lee, Phys. Fluids 11, 3312 (1999): Eq. (11), Table I, Figs. 5, 8(b), 9 (Re = 40 / 60 / 100 / 160)
LIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "literature")
REF_RE = 100          # Reynolds number of the runs: picks the *_Re<REF_RE> columns / rows of the Kang files (--re N)
if "--re" in sys.argv: REF_RE = int(sys.argv[sys.argv.index("--re") + 1])
STOJ = REF_RE == 100  # the Stojkovic table / Fig. 13 are Re = 100 only
LIT_SOURCE, LIT2_SOURCE = "Stojkovi\u0107 et al. (2002)", "Kang et al. (1999)"


def load_lit(name):
    """columns of literature/<name> as a dict of arrays (text columns are skipped)"""
    path = os.path.join(LIT_DIR, name); cols = None; rows = []
    for l in open(path):
        if l.startswith("# columns:"): cols = l.split(":", 1)[1].split()
        elif not l.startswith("#") and l.strip(): rows.append(l.split())
    out = {}
    for n, c in enumerate(cols):
        try: out[c] = np.array([float(r[n]) for r in rows])
        except ValueError: pass
    return out


_T2 = load_lit("stojkovic2002_table2_Re100.dat")
# alpha: (Cl_mean, Cd_mean, St, Cl_amplitude, Cd_amplitude) from Stojkovic Table II; nan = not given
LIT = {a: (cl, cd, st, ca, da) for a, cl, cd, st, ca, da in zip(_T2["alpha"], _T2["Cl_mean"], _T2["Cd_mean"], _T2["St"], _T2["Cl_amp"], _T2["Cd_amp"])}
_F = load_lit("kang1999_eq11_mean_lift_fit.dat"); _k = int(np.argmin(np.abs(_F["Re"] - REF_RE)))
LIT2_CL_SLOPE, LIT2_ALPHA_MAX = float(_F["slope_total"][_k]), float(_F["alpha_max"][_k])
LIT_CL_SLOPE = LIT2_CL_SLOPE          # used for the "lit" column of the printed table


def stats(d, W):
    F = np.loadtxt(os.path.join(d, "output_forces.dat"))
    F = F[F[:, 1] == F[:, 1].max()]
    q = 0.5 * RHO * U * U * D
    t, Cd, Cl = F[:, 2], F[:, 3] / q, F[:, 4] / q
    m = t >= t[-1] - W
    tu = np.linspace(t[m][0], t[m][-1], 8192)
    y = np.interp(tu, t[m], Cl[m]); y -= y.mean()
    Fq = np.abs(np.fft.rfft(y * np.hanning(len(y)))); f = np.fft.rfftfreq(len(y), tu[1] - tu[0])
    band = (f > 0.05) & (f < 1.0)
    St = f[band][np.argmax(Fq[band])] * D / U
    # [2026-10-06] the FFT bin width is 1/W (0.033 for W = 30: St came out as 5/30 or 7/40 whatever the wake did).
    # Use the mean period between upward zero crossings of the lift fluctuation instead (FFT peak only as a guard
    # against counting noise crossings: low-pass at 3x the peak frequency first).
    if Fq[band].max() > 0:
        Y = np.fft.rfft(y); Y[f > 3.0 * St] = 0.0; ys = np.fft.irfft(Y, len(y))
        k = np.where((ys[:-1] < 0) & (ys[1:] >= 0))[0]
        if len(k) >= 3:
            tc = tu[k] - ys[k] * (tu[k + 1] - tu[k]) / (ys[k + 1] - ys[k])
            St = (len(tc) - 1) / (tc[-1] - tc[0]) * D / U
    # amplitude from percentiles: robust to step-to-step force noise
    lo, hi = np.percentile(Cl[m], [1, 99])
    dlo, dhi = np.percentile(Cd[m], [1, 99])
    Cdp = float(F[m, 5].mean() / q) if F.shape[1] > 5 else float("nan")      # pressure part (Fpres_0); friction = rest
    return dict(t=t, Cd=Cd, Cl=Cl, t0=t[m][0], t1=t[-1], Cl_mean=float(Cl[m].mean()), Cdp=Cdp,
                Cd_mean=float(Cd[m].mean()), Cl_amp=0.5 * (hi - lo), Cd_amp=0.5 * (dhi - dlo), St=St,
                shedding=0.5 * (hi - lo) > SHED_CL_AMP)


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(here, "Images")
    W = float(args[args.index("--window") + 1]) if "--window" in args else 30.0
    kutta = "--kutta" in args          # draw the inviscid Kutta-Joukowski lift line (off by default)
    skip = {args.index(k) + 1 for k in ("--outdir", "--window", "--re") if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    runs = []
    for d in dirs:
        m = re.search(r"alpha[_=]?(\d+(?:\.\d+)?)", os.path.abspath(d))
        if not m:
            print(f"skip {d}: no alpha<value> in the path"); continue
        runs.append((float(m.group(1)), d, stats(d, W)))
    runs.sort(key=lambda r: r[0])
    os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, "rotating_summary.csv"), "w") as fo:
        fo.write("alpha,t0,t1,Cl_mean,Cl_KuttaJoukowski,Cl_over_KJ,Cl_lit_approx,Cd_mean,Cl_amplitude,shedding,St\n")
        print(f"{'alpha':>5} {'window':>13} {'Cl mean':>9} {'KJ -2pi a':>9} {'Cl/KJ':>6} {'lit~':>7} "
              f"{'Cd mean':>8} {'Cl amp':>8} {'St':>6}")
        for a, d, s in runs:
            kj = -2 * np.pi * a
            lit = LIT_CL_SLOPE * a
            st = f"{s['St']:.3f}" if s["shedding"] else "steady"
            ratio = s["Cl_mean"] / kj if a > 0 else float("nan")
            print(f"{a:5.2f} {s['t0']:6.1f}-{s['t1']:6.1f} {s['Cl_mean']:9.4f} {kj:9.4f} {ratio:6.3f} "
                  f"{lit:7.2f} {s['Cd_mean']:8.4f} {s['Cl_amp']:8.4f} {st:>6}")
            fo.write(f"{a:g},{s['t0']:.4g},{s['t1']:.4g},{s['Cl_mean']:.6g},{kj:.6g},{ratio:.4g},{lit:.4g},"
                     f"{s['Cd_mean']:.6g},{s['Cl_amp']:.6g},{s['shedding']},{s['St'] if s['shedding'] else ''}\n")

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for n, (a, d, s) in enumerate(runs):
        k = s["t"] > 2
        ax[0].plot(s["t"][k], s["Cl"][k], lw=0.7, color=f"C{n}")
        if kutta: ax[0].axhline(-2 * np.pi * a, color=f"C{n}", ls=":", lw=1)
        ax[1].plot(s["t"][k], s["Cd"][k], lw=0.7, color=f"C{n}")
        # no legend: each curve carries its spin ratio in plain text just past its end, in the curve's colour
        e = s["t"] >= s["t"][-1] - 10.0
        for x_, key in ((ax[0], "Cl"), (ax[1], "Cd")):
            x_.text(s["t"][-1] + 1.0, float(s[key][e].mean()), rf"$\alpha$ = {a:g}", color=f"C{n}", fontsize=9, va="center", ha="left", clip_on=False)
    ax[0].set_ylabel(r"$C_l$"); ax[1].set_ylabel(r"$C_d$"); ax[1].set_xlabel(r"$t\,U/D$")
    ax[0].set_title(rf"Rotating Cylinder $C_l$, $C_d$ History in Re = {REF_RE} Flow" + (r" (dotted: Kutta-Joukowski $-2\pi\alpha$)" if kutta else ""))
    tmax = max(r[2]["t"][-1] for r in runs); ax[1].set_xlim(0, tmax * 1.09)          # room for the labels
    for x in ax: x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "rotating_force_history.png"), dpi=200); plt.close(fig)

    al = np.array([r[0] for r in runs])
    aa = np.linspace(0, max(al.max(), 2.0) * 1.05, 50)
    a2 = np.linspace(0, LIT2_ALPHA_MAX, 2); xmax = aa[-1]
    S13 = load_lit("stojkovic2002_fig13_Re100.dat"); m13 = S13["alpha"] <= xmax          # Re = 100, alpha up to 12
    K8 = load_lit("kang1999_fig8b_mean_drag.dat"); kcd = K8.get(f"Cd_Re{REF_RE}"); la = np.array(sorted(LIT))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    if kutta: ax[0].plot(aa, -2 * np.pi * aa, "k--", label=r"Kutta-Joukowski $-2\pi\alpha$")
    if STOJ: ax[0].plot(S13["alpha"][m13], S13["Cl_mean"][m13], color="0.5", label=LIT_SOURCE)
    ax[0].plot(a2, LIT2_CL_SLOPE * a2, color="C0", ls="-.", label=LIT2_SOURCE)
    ax[0].errorbar(al, [r[2]["Cl_mean"] for r in runs], yerr=[r[2]["Cl_amp"] for r in runs],
                   fmt="o", color="C3", capsize=4, label="Numerical")
    ax[0].set_xlabel(r"$\alpha = \Omega R/U$"); ax[0].set_ylabel(r"$\overline{C_l}$"); ax[0].legend(fontsize=9)
    if STOJ: ax[1].plot(la, [LIT[a][1] for a in la], color="0.5", label=LIT_SOURCE)
    if kcd is not None:
        mk = np.isfinite(kcd) & (K8["alpha"] <= xmax); ax[1].plot(K8["alpha"][mk], kcd[mk], color="C0", ls="-.", label=LIT2_SOURCE)
    ax[1].errorbar(al, [r[2]["Cd_mean"] for r in runs], yerr=[r[2]["Cd_amp"] for r in runs],
                   fmt="o", color="C3", capsize=4, label="Numerical")
    ax[1].set_xlabel(r"$\alpha = \Omega R/U$"); ax[1].set_ylabel(r"$\overline{C_d}$"); ax[1].legend(fontsize=9)
    ax[1].set_xlim(ax[0].get_xlim())
    if STOJ: print(f"\nagainst {LIT_SOURCE}, Table II:")
    for a, d, st in runs:
        if STOJ and a in LIT:
            L = LIT[a]; f = lambda v, r: "   n/a" if (r is None or not np.isfinite(r) or r == 0) else f"{100 * (v / r - 1):+6.1f} %"
            print(f"  alpha = {a:g}: Cl {st['Cl_mean']:.3f} vs {L[0]:.3f} ({f(st['Cl_mean'], L[0])});  Cd {st['Cd_mean']:.3f} vs {L[1]:.3f} ({f(st['Cd_mean'], L[1])});"
                  f"  St {st['St']:.3f} vs {L[2] if np.isfinite(L[2]) else 'steady'} ({f(st['St'], L[2])});  Cl amplitude {st['Cl_amp']:.3f} vs {L[3] if np.isfinite(L[3]) else 0.0}")
    # pressure / friction split of the mean drag against Kang et al. (1999) Fig. 8(b) (digitized, linear in alpha between points)
    KP, KF = load_lit("kang1999_fig8b_drag_pressure.dat"), load_lit("kang1999_fig8b_drag_friction.dat")
    if f"Cdp_Re{REF_RE}" in KP:
        print(f"\nmean drag split against {LIT2_SOURCE}, Fig. 8(b), Re = {REF_RE}:")
        for a, d, st in runs:
            if a > KP["alpha"].max() or not np.isfinite(st["Cdp"]): continue
            rp = float(np.interp(a, KP["alpha"], KP[f"Cdp_Re{REF_RE}"])); rf_ = float(np.interp(a, KF["alpha"], KF[f"Cdf_Re{REF_RE}"]))
            cf = st["Cd_mean"] - st["Cdp"]
            print(f"  alpha = {a:g}: pressure {st['Cdp']:.3f} vs {rp:.3f} ({100 * (st['Cdp'] / rp - 1):+.0f} %);  friction {cf:.3f} vs {rf_:.3f} ({100 * (cf / rf_ - 1):+.0f} %);"
                  f"  total {st['Cd_mean']:.3f} vs {rp + rf_:.3f} ({100 * (st['Cd_mean'] / (rp + rf_) - 1):+.0f} %)")
    for x in ax: x.grid(alpha=0.3)
    fig.suptitle(rf"Rotating Cylinder $C_l$, $C_d$ Comparison in Re = {REF_RE} Flow", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "rotating_cl_cd_vs_alpha.png"), dpi=200); plt.close(fig)
    # C_d - C_l phase diagram: the closed loop each run settles on (last W time units), outline only
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n, (a, d, st) in enumerate(runs):
        m = st["t"] >= st["t"][-1] - W
        ax.plot(st["Cd"][m], st["Cl"][m], lw=1.0, color=f"C{n}")
        # label at the centroid of the loop (mean Cd, mean Cl) instead of a legend
        ax.text(st["Cd"][m].mean(), st["Cl"][m].mean(), rf"$\alpha$ = {a:g}", color=f"C{n}", ha="center", va="center", fontsize=10)
    ax.set_xlabel(r"$C_d$"); ax.set_ylabel(r"$C_l$"); ax.grid(alpha=0.3)
    ax.set_title(rf"Rotating Cylinder $C_d$-$C_l$ Limit Cycles in Re = {REF_RE} Flow")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "rotating_cl_cd_phase.png"), dpi=200); plt.close(fig)
    print(f"wrote {outdir}/rotating_force_history.png, rotating_cl_cd_vs_alpha.png, rotating_cl_cd_phase.png, rotating_summary.csv")


if __name__ == "__main__":
    main()
