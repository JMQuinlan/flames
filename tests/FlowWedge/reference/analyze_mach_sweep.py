#!/usr/bin/env python3
# ============================================================================
#  analyze_mach_sweep.py   (was compare_wedge_angles.py)
#  ---------------------------------------------------------------------------
#  Cross-Mach summary for the FlowWedge tests: measure the oblique-shock angle
#  beta from each run and compare to the analytical theta-beta-Mach relation.
#
#  Outputs (to ./Images):
#     wedge_angle_comparison.png  - beta vs Ma: analytical curve (theta=const)
#                                   + measured points (red), with the detached
#                                   region shaded; plus per-case wedge contour
#                                   PNGs via wedge_analysis.
#     wedge_angles.csv            - the measured/analytical table.
#
#  Reuses wedge_analysis.sample()/detect_shock() and oblique_shock_theory.
#  Robust to runs that have not been generated yet (skips with a note).
#
#  Usage:  python3 analyze_mach_sweep.py                       # INCLINE outputs -> ./Images
#          python3 analyze_mach_sweep.py <tmpl_with_{ma}> [img_dir]
#     e.g. python3 analyze_mach_sweep.py "../../../bin/tests/FlowWedge/sweep14/Ma{ma}/output"
#  Cases are auto-detected from the output dirs that contain plotfiles (see MACH_LIST).
# ============================================================================

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yt
yt.funcs.mylog.setLevel(50)

import oblique_shock_theory as ost
import wedge_analysis as wa

# ============================================================================
# ==============================  CONFIG  ====================================
# ============================================================================
# Cases to compare.  None = AUTO-DETECT (default): exactly the output directories
# that exist and match wedge_analysis.OUTPUT_TMPL with {ma} as a wildcard (e.g.
# .../output_Ma*; the Mach number is parsed from the name) and contain at least one
# plotfile past the initial condition (a run with only step 0 is skipped).
# Nothing else is listed.  Adding a case = adding input_Ma<M> and running it with
# the usual plot_file name.  Or give an explicit list, e.g. [2.0, 3.0].
# Analytical beta table: python3 oblique_shock_theory.py
MACH_LIST   = None
THETA_DEG   = wa.WEDGE_THETA_DEG         # wedge half-angle (from wedge_analysis)
GAMMA       = wa.GAMMA
OUTPUT_DIR  = "./Images"
ALSO_RENDER_EACH = True                  # also write the per-case wedge_Ma*.png overlays
LABEL_POINTS     = False                 # "Ma <M>" text next to each measured point

# beta-vs-Ma plot style
MA_CURVE_LO, MA_CURVE_HI = 1.02, None    # analytical curve x-range (None = 5% past the largest Mach)
ANALYTIC_COLOR = "#1f5fd1"               # analytical weak-shock curve (BLUE)
STRONG_COLOR   = "#9aa0a8"               # analytical strong-shock curve (grey, dashed)
MEASURED_COLOR = "#d11f1f"               # measured points (RED)
THETAMAX_COLOR = "#39ff14"               # theta_max (detachment) curve
DETACH_SHADE   = "#ffd9d9"               # shaded detached region
FIGSIZE   = (8.8, 6.0)
DPI       = 140
LINE_WIDTH = 2.2
MARKER_SIZE = 9
FONT_TITLE = 15
FONT_LABEL = 12.5
FONT_TICK  = 10.5
TITLE   = "Oblique-shock Angle vs Mach"
XLABEL  = "Mach,  $M_1$"
YLABEL  = r"Shock Angle, $\beta$ ($^\circ$)"
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_cases():
    """{Mach: output dir}.  Auto-detected from wa.OUTPUT_TMPL (existing dirs with
    plotfiles only) unless MACH_LIST is given."""
    import glob, re
    tmpl = wa._abs(wa.OUTPUT_TMPL)
    if MACH_LIST is not None:
        return {float(M): tmpl.format(ma=f"{M:.1f}") for M in MACH_LIST}
    cases = {}
    rx = re.compile("^" + re.escape(tmpl).replace(re.escape("{ma}"), r"([0-9]+(?:\.[0-9]+)?)") + "$")
    for path in glob.glob(tmpl.replace("{ma}", "*")):
        if any(t in path for t in wa.IGNORE_TOKENS):      # e.g. alamo's renamed '*.old.*' dirs
            continue
        m = rx.match(os.path.normpath(path))
        # need at least one plotfile past the initial condition (step 0 = IC, no shock yet)
        if m and os.path.isdir(path) and any(wa.step_number(p) > 0 for p in wa.discover_plotfiles(path)):
            cases[float(m.group(1))] = path
    return dict(sorted(cases.items()))


def collect():
    """Measure beta for each available case; return a list of result dicts."""
    rows = []
    cases = detect_cases()
    print(f"  cases found: {', '.join(f'{M:g}' for M in cases)}")
    for M, out_dir in cases.items():
        beta_a = ost.beta_deg(THETA_DEG, M, GAMMA, weak=True)
        attached = beta_a is not None
        row = dict(M=M, beta_analytic=beta_a, attached=attached,
                   beta_measured=None, present=os.path.isdir(out_dir))
        if not row["present"]:
            print(f"  Ma {M:g}: output dir not found ({out_dir}) -- skipping (run the sim)")
            rows.append(row)
            continue
        try:
            pfs = wa.discover_plotfiles(out_dir)
            s = wa.sample(pfs[-1])
            meas = wa.detect_shock(s)
            row["beta_measured"] = meas["beta_deg"]
            row["time"] = s["time"]
            print(f"  Ma {M:g}: beta_analytic="
                  f"{('%.2f' % beta_a) if attached else 'DETACHED':>8}  "
                  f"beta_measured={('%.2f' % meas['beta_deg']) if meas['beta_deg'] else 'n/a':>8}  ")
                  #f"(t={s['time']:.2f})")
            if ALSO_RENDER_EACH:
                wa.analyze(M, out_dir)
        except Exception as exc:
            print(f"  Ma {M:g}: analysis failed: {exc}")
        rows.append(row)
    return rows


def plot_beta_vs_mach(rows, out_png):
    hi = MA_CURVE_HI or 1.05 * max([r["M"] for r in rows] + [2.0])
    Ms = np.linspace(MA_CURVE_LO, hi, 400)
    beta_weak = np.array([ost.beta_deg(THETA_DEG, M, GAMMA, weak=True) or np.nan for M in Ms])
    beta_strong = np.array([ost.beta_deg(THETA_DEG, M, GAMMA, weak=False) or np.nan for M in Ms])
    tmax = np.array([ost.theta_max_deg(M, GAMMA) or np.nan for M in Ms])

    # detachment Mach: smallest M whose theta_max >= THETA_DEG
    attached_mask = tmax >= THETA_DEG
    M_detach = Ms[attached_mask][0] if attached_mask.any() else None

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    if M_detach is not None:
        ax.axvspan(MA_CURVE_LO, M_detach, color=DETACH_SHADE, zorder=0,
                   label=f"detached ($\\theta>\\theta_{{max}}$), $M<{M_detach:.2f}$")

    ax.plot(Ms, beta_weak, "-", color=ANALYTIC_COLOR, lw=LINE_WIDTH,
            label=rf"analytical $\beta$ (weak), $\theta={THETA_DEG:g}^\circ$")
    ax.plot(Ms, beta_strong, "--", color=STRONG_COLOR, lw=1.6,
            label=r"analytical $\beta$ (strong)")

    # measured points
    mx = [r["M"] for r in rows if r["beta_measured"] is not None and r["attached"]]
    my = [r["beta_measured"] for r in rows if r["beta_measured"] is not None and r["attached"]]
    if mx:
        ax.plot(mx, my, "o", color=MEASURED_COLOR, ms=MARKER_SIZE, zorder=6,
                label="measured (Hydro2)")
        if LABEL_POINTS:
            for r in rows:
                if r["beta_measured"] is not None and r["attached"]:
                    ax.annotate(f"  Ma {r['M']:g}", (r["M"], r["beta_measured"]),
                                fontsize=9, color=MEASURED_COLOR, va="center")
    # detached cases: mark measured points (if any) with an x
    dx = [r["M"] for r in rows if r["beta_measured"] is not None and not r["attached"]]
    dy = [r["beta_measured"] for r in rows if r["beta_measured"] is not None and not r["attached"]]
    if dx:
        ax.plot(dx, dy, "x", color="#7a1f1f", ms=MARKER_SIZE, zorder=6,
                label="measured (detached/bow)")

    ax.set_title(TITLE, fontsize=FONT_TITLE)
    ax.set_xlabel(XLABEL, fontsize=FONT_LABEL)
    ax.set_ylabel(YLABEL, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(MA_CURVE_LO, hi)
    ax.set_ylim(0, 90)
    ax.legend(fontsize=9.5, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_png}")


def write_csv(rows, out_csv):
    with open(out_csv, "w") as f:
        f.write("Mach,theta_deg,attached,beta_analytic_deg,beta_measured_deg,abs_err_deg,pct_err\n")
        for r in rows:
            ba = r["beta_analytic"]
            bm = r["beta_measured"]
            err = (bm - ba) if (ba is not None and bm is not None) else None
            pct = (100.0 * err / ba) if (err is not None and ba) else None
            f.write(f"{r['M']:.2f},{THETA_DEG:.1f},{r['attached']},"
                    f"{'' if ba is None else f'{ba:.3f}'},"
                    f"{'' if bm is None else f'{bm:.3f}'},"
                    f"{'' if err is None else f'{err:.3f}'},"
                    f"{'' if pct is None else f'{pct:.2f}'}\n")
    print(f"  wrote {out_csv}")


def print_table(rows):
    print("\n  ---- oblique-shock angle summary (theta = %.1f deg) ----" % THETA_DEG)
    print(f"  {'Ma':>5} {'attached':>9} {'beta_analytic':>14} {'beta_measured':>14} {'err':>9}")
    for r in rows:
        ba = f"{r['beta_analytic']:.2f}" if r["beta_analytic"] is not None else "DETACHED"
        bm = f"{r['beta_measured']:.2f}" if r["beta_measured"] is not None else "n/a"
        err = (f"{r['beta_measured']-r['beta_analytic']:+.2f}"
               if (r["beta_analytic"] is not None and r["beta_measured"] is not None) else "-")
        print(f"  {r['M']:5.1f} {str(r['attached']):>9} {ba:>14} {bm:>14} {err:>9}")


def main():
    # Optional overrides:  argv[1] = plotfile-dir template with {ma}
    #                      argv[2] = image/CSV output dir
    # Defaults: wedge_analysis.OUTPUT_TMPL (INCLINE /mmfs1) and OUTPUT_DIR (./Images).
    global OUTPUT_DIR
    if len(sys.argv) > 1:
        wa.OUTPUT_TMPL = os.path.normpath(os.path.abspath(sys.argv[1]))
    if len(sys.argv) > 2:
        OUTPUT_DIR = os.path.abspath(sys.argv[2])
    wa.OUTPUT_DIR = OUTPUT_DIR                  # per-case wedge_Ma*.png go to the same place
    print("analyze_mach_sweep")
    print(f"  plotfiles: {wa._abs(wa.OUTPUT_TMPL)}")
    print(f"  images   : {wa._abs(OUTPUT_DIR)}")
    out_dir = wa._abs(OUTPUT_DIR)
    wa.ensure_dir(out_dir)
    rows = collect()
    if not rows:
        print("  no output dirs with plotfiles match the template -- nothing written.")
        return
    plot_beta_vs_mach(rows, os.path.join(out_dir, "wedge_angle_comparison.png"))
    write_csv(rows, os.path.join(out_dir, "wedge_angles.csv"))
    print_table(rows)
    print("\nDone.")


if __name__ == "__main__":
    main()
