#!/usr/bin/env python3
"""
Sch20 COATED oscillating bubble at the cm scale -- the 1-on-1 coated twin of the uncoated
tests/FlowRayleighPlesset/Sch20_Oscillating_Neumann_Large_3D (deck:
input_Sch20-Oscillating_Marmottant_Light_ContourRateMili; R0 = 2 cm, p_inf/p_b = 10,
kappa_s = 0.72 kg/s, same K* = 0.18 as the 2 um Light deck).

Two parts:
  1. the standard Sch20Analysis plots of the coated run (all plotting lives in
     tests/FlowRayleighPlesset/reference/sch20_analysis.py), exactly like
     analyze_Sch20_Oscillating_Marmottant_Light.py;
  2. a TWIN comparison with the uncoated run of the same mesh/numerics:
       <STEM>_twin_R.png      R/R0 vs t/tau_R for coated + uncoated, sim and KM
       <STEM>_twin_error.png  sim - KM for each; "coated - uncoated" error difference
     and a printed table (extrema vs KM, kappa_eff fit).  The uncoated error is the
     numerical baseline of this mesh; what the coated run adds on top of it is the
     shell model's own error.

    python3 analyze_Sch20_Oscillating_Marmottant_Mili.py                        # INCLINE paths below
    python3 analyze_Sch20_Oscillating_Marmottant_Mili.py COATED_DIR --twin-output UNCOATED_DIR
    python3 analyze_Sch20_Oscillating_Marmottant_Mili.py --twin-only            # skip part 1
    python3 analyze_Sch20_Oscillating_Marmottant_Mili.py --nproc 16 --fit-tmax 3.0

kappa_eff fit: the coated KM model is re-solved with kappa_s -> f kappa_s and f is fitted
to the simulated R(t) over t < --fit-tmax tau_R (default: the whole run).  f = 1 means the
shell damps exactly as the model; the Light deck gave f = 0.84 (ContourRate, INCLINE 10-20).
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RP = os.path.normpath(os.path.join(_HERE, "..", "..", "FlowRayleighPlesset"))
sys.path.insert(0, os.path.join(_RP, "reference"))
import sch20_analysis                                             # noqa: E402
from sch20_analysis import Sch20Analysis, plt                     # noqa: E402
from marmottant_rpe_km import solve_km                           # noqa: E402

CFG = dict(
    INPUT=os.path.normpath(os.path.join(_HERE, "..", "input_Sch20-Oscillating_Marmottant_Light_ContourRateMili")),
    OUTPUT_DIR="/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/output_Sch20_Oscillating_Marmottant_Light_ContourRateMili",
    CASE="Oscillating",
    COATED=True,
    STEM="Sch20_Oscillating_Marmottant_Mili",
    N_PROC=4,
    PLOTS=[
        "radius", "residual", "velocity", "probes", "pressure", "waves",
        "shapes", "planes", "modes", "gif", "r_volume",
        # "gamma", "eta_band", "ic_pressure", "driven", "csv",
    ],
    DEBUG=["thermo", "health"],
    STYLE=dict(
        radius_measure="ray",       # same measure as the uncoated analysis (analyze_Sch20_Oscillating_3D.py)
        shapes_times=(0.7, "Rmin", 2.0),
        planes_n=24,
        contours=True,
        c_line=True, c_origin="tc",
    ),
    TITLES=dict(),
    LEGACY_MODEL_PLOTS=True,
    # ---- uncoated twin ---------------------------------------------------------------- #
    TWIN_INPUT=os.path.join(_RP, "Sch20_Oscillating_Neumann_Large_3D"),
    TWIN_OUTPUT="/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_Large_3D",
    TWIN_STEM="Sch20_Oscillating_3D_twin",
)


def _extrema(t, R):
    """First minimum and the following maximum, refined by a parabola through 3 points."""
    ok = np.isfinite(R)
    t, R = t[ok], R[ok]
    if len(R) < 3:
        return (np.nan,) * 4

    def para(i):
        if 0 < i < len(R) - 1:
            x, y = t[i - 1:i + 2], R[i - 1:i + 2]
            a, b, c = np.polyfit(x - x[1], y, 2)
            if a != 0:
                dt = -b / (2 * a)
                if abs(dt) <= (x[2] - x[0]):
                    return x[1] + dt, c - b * b / (4 * a)
        return t[i], R[i]
    i = int(np.argmin(R))
    j = i + int(np.argmax(R[i:])) if i < len(R) - 1 else i
    return para(i) + para(j)


def _on(tm, Rm, t):
    return np.interp(t, tm, Rm)


def _kappa_fit(A, t, R, tmax):
    """Fit f in kappa_s -> f kappa_s (coated KM) to the simulated R(t), t <= tmax."""
    from scipy.optimize import minimize_scalar
    m = np.isfinite(R) & (t <= tmax)
    tt, RR = t[m], R[m]
    te = np.linspace(0.0, tt[-1], 4001)

    def err(f):
        sh = replace(A.shell, kappa_s=f * A.shell.kappa_s)
        tk, Rk, _ = solve_km(A.R0, 0.0, (0.0, te[-1]), sh, A.liq, A.gas, t_eval=te)
        if len(Rk) != len(te):
            return 1e9
        return float(np.sqrt(np.mean((_on(tk, Rk, tt) - RR) ** 2)) / A.R0)
    r = minimize_scalar(err, bounds=(0.05, 3.0), method="bounded", options={"xatol": 1e-3})
    return r.x, r.fun, err(1.0)


def twin(A, U, out_stem, fit_tmax=None):
    tauR = A.meta["tau_c"]                       # Rayleigh time, identical for both decks
    tc, Rc, labc = A.radius()
    tu, Ru, labu = U.radius()
    tkc, Rkc, _ = A.km
    tku, Rku, _ = U.km

    # ---- table -------------------------------------------------------------------------- #
    print("\n  TWIN COMPARISON (radius measure: %s; times in tau_R = %.4g s)" % (A.S["radius_measure"], tauR))
    print("  %-22s %8s %7s  %8s %7s   %s" % ("", "R_min/R0", "t/tauR", "R_max/R0", "t/tauR", "err vs own KM (Rmin, Rmax)"))
    rows = []
    # models only over the simulated window, so a partial run is compared like for like
    wu, wc = tku <= np.nanmax(tu), tkc <= np.nanmax(tc)
    tku_w, Rku_w, tkc_w, Rkc_w = tku[wu], Rku[wu], tkc[wc], Rkc[wc]
    for name, t, R, tk, Rk in (("uncoated sim", tu, Ru, tku_w, Rku_w), ("uncoated KM", tku_w, Rku_w, None, None),
                               ("coated sim", tc, Rc, tkc_w, Rkc_w), ("coated KM", tkc_w, Rkc_w, None, None)):
        e = _extrema(t, R / U.R0 if name.startswith("unc") else R / A.R0)
        s = ""
        if tk is not None:
            k = _extrema(tk, Rk / (U.R0 if name.startswith("unc") else A.R0))
            # compare only over the simulated window
            s = "%+.4f, %+.4f" % (e[1] - k[1], e[3] - k[3])
        rows.append((name, e))
        print("  %-22s %8.4f %7.3f  %8.4f %7.3f   %s" % (name, e[1], e[0] / tauR, e[3], e[2] / tauR, s))
    # amplitude
    amp = lambda e: e[3] - e[1]
    d = dict(rows)
    for side in ("uncoated", "coated"):
        es, ek = d[side + " sim"], d[side + " KM"]
        print("  %-8s rebound amplitude (Rmax - Rmin)/R0: sim %.4f  KM %.4f  (%+.1f%%)"
              % (side, amp(es), amp(ek), 100 * (amp(es) - amp(ek)) / amp(ek)))

    # ---- errors on the sim frames ----------------------------------------------------- #
    eu = (Ru - _on(tku, Rku, tu)) / U.R0
    ec = (Rc - _on(tkc, Rkc, tc)) / A.R0
    ok_u, ok_c = np.isfinite(eu), np.isfinite(ec)
    print("  rms(sim - KM)/R0:  uncoated %.4f   coated %.4f" % (np.sqrt(np.mean(eu[ok_u] ** 2)),
                                                                np.sqrt(np.mean(ec[ok_c] ** 2))))
    tmax = (fit_tmax * tauR) if fit_tmax else np.nanmax(tc)
    f, rms_f, rms_1 = _kappa_fit(A, tc, Rc, tmax)
    print("  kappa_eff/kappa_s (coated KM fit, t <= %.2f tauR) = %.3f   rms %.4f (f=1: %.4f)"
          % (tmax / tauR, f, rms_f, rms_1))

    # ---- plots ------------------------------------------------------------------------- #
    S = A.S
    fig, ax = plt.subplots(figsize=(S["width"], S["height"]))
    ax.plot(tku / tauR, Rku / U.R0, "-", color="0.55", lw=1.4, label="Keller–Miksis (uncoated)")
    ax.plot(tu / tauR, Ru / U.R0, "o", color="0.35", ms=2.5, label="Uncoated sim")
    ax.plot(tkc / tauR, Rkc / A.R0, "-", color=S["c_km"], lw=1.4, label="Coated Keller–Miksis")
    ax.plot(tc / tauR, Rc / A.R0, "o", color=S["c_sim"], ms=2.5, label="Coated sim")
    if np.isfinite(f):
        te = np.linspace(0, np.nanmax(tc), 3001)
        tk, Rk, _ = solve_km(A.R0, 0.0, (0.0, te[-1]), replace(A.shell, kappa_s=f * A.shell.kappa_s),
                             A.liq, A.gas, t_eval=te)
        ax.plot(tk / tauR, Rk / A.R0, ":", color=S["c_km"], lw=1.2,
                label=r"Coated KM, $\kappa_s \times %.2f$" % f)
    ax.set_xlim(0, 1.02 * max(np.nanmax(tc), np.nanmax(tu)) / tauR)
    ax.set_xlabel(r"$t/\tau_R$"); ax.set_ylabel(r"$R/R_0$")
    ax.legend(loc="best", fontsize="small")
    A._save(fig, "twin_R") if out_stem is None else _save(fig, out_stem + "_twin_R", S)

    fig, ax = plt.subplots(figsize=(S["width"], S["height"]))
    ax.axhline(0, color="0.7", lw=0.8)
    ax.plot(tu[ok_u] / tauR, 100 * eu[ok_u], "-o", color="0.35", ms=2, label="Uncoated: sim − KM")
    ax.plot(tc[ok_c] / tauR, 100 * ec[ok_c], "-o", color=S["c_sim"], ms=2, label="Coated: sim − coated KM")
    # coated minus uncoated error on a common time base (numerical baseline removed)
    tt = tc[ok_c][(tc[ok_c] >= np.nanmin(tu)) & (tc[ok_c] <= np.nanmax(tu))]
    if len(tt) and ok_u.sum() > 1:
        dd = _on(tc[ok_c], ec[ok_c], tt) - _on(tu[ok_u], eu[ok_u], tt)
        ax.plot(tt / tauR, 100 * dd, "--", color="k", lw=1.2, label="Coated − uncoated error")
    ax.set_xlabel(r"$t/\tau_R$"); ax.set_ylabel(r"$(R - R_\mathrm{KM})/R_0$  [%]")
    ax.legend(loc="best", fontsize="small")
    A._save(fig, "twin_error") if out_stem is None else _save(fig, out_stem + "_twin_error", S)


def _save(fig, stem, S):
    for fmt in S["formats"]:
        fig.savefig(stem + "." + fmt, dpi=S["dpi"], bbox_inches="tight", pad_inches=S["pad_inches"])
    plt.close(fig)
    print("  wrote", stem + ".png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotdir", nargs="?", default=CFG["OUTPUT_DIR"])
    ap.add_argument("--input", default=CFG["INPUT"])
    ap.add_argument("--stem", default=CFG["STEM"])
    ap.add_argument("--twin-output", default=CFG["TWIN_OUTPUT"], help="uncoated twin plotfiles")
    ap.add_argument("--twin-input", default=CFG["TWIN_INPUT"], help="uncoated twin deck")
    ap.add_argument("--twin-only", action="store_true", help="skip the standard coated plots")
    ap.add_argument("--no-twin", action="store_true", help="skip the twin comparison")
    ap.add_argument("--fit-tmax", type=float, default=None, help="kappa_eff fit window, t/tau_R")
    ap.add_argument("--nproc", type=int, default=CFG["N_PROC"])
    ap.add_argument("--out-dir", default=None, help="image directory (default ./Images next to this script)")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.join(_HERE, "Images")
    os.makedirs(out_dir, exist_ok=True)

    if not a.twin_only:
        cfg = dict(CFG, OUTPUT_DIR=a.plotdir, INPUT=a.input, STEM=a.stem, N_PROC=a.nproc, OUT_DIR=out_dir)
        sys.argv = [sys.argv[0], a.plotdir, "--input", a.input, "--stem", a.stem, "--nproc", str(a.nproc)]
        sch20_analysis.main(cfg)
    if a.no_twin:
        return
    if not os.path.isdir(a.twin_output):
        print("  [twin] uncoated output %s not found; pass --twin-output" % a.twin_output)
        return
    kw = dict(out_dir=out_dir, nproc=a.nproc, style=CFG["STYLE"], titles=CFG["TITLES"])
    A = Sch20Analysis(a.input, a.plotdir, case=CFG["CASE"], coated=True, stem=a.stem, **kw)
    U = Sch20Analysis(a.twin_input, a.twin_output, case=CFG["CASE"], coated=False, stem=CFG["TWIN_STEM"], **kw)
    if abs(A.R0 - U.R0) > 1e-12 * A.R0 or A.meta["dx_finest"] != U.meta["dx_finest"]:
        print("  [twin] WARNING: decks differ in R0 or dx_finest (%g/%g, %g/%g)"
              % (A.R0, U.R0, A.meta["dx_finest"], U.meta["dx_finest"]))
    twin(A, U, None, a.fit_tmax)


if __name__ == "__main__":
    main()
