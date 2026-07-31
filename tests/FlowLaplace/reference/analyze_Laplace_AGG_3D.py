# -*- coding: utf-8 -*-
"""
===============================================================================
FlowLaplace AGGREGATE analysis (3D sphere, 2 sigma/R)  --  overlay every R<R>_Sigma<sigma> run.
===============================================================================
Walks the FlowLaplace output directory, finds every test that has produced
AMReX plotfiles, and overlays their normalized radius histories on a single
figure (plus a companion semilog error overlay).  Time is normalized by each
test's own characteristic timescale tau so curves spanning six orders of
magnitude in R line up on one axis.

This is the sensitivity view: a curve that drifts, oscillates, or runs away
flags an (R, sigma) combination that the diffuse-interface surface-tension
model resolves poorly at this (deliberately cheap) unit-test resolution.

Encoding:  color = bubble radius R0,   line style = surface tension sigma.

USAGE:
    python tests/FlowLaplace/reference/analyze_Laplace_AGG.py

OUTPUTS (in reference/Images/):
    Laplace_AGG_radius.png / .eps     R/R0(t/tau) overlay + error overlay
===============================================================================
"""

import os
import re
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from analyze_Laplace_3D import (                                    # noqa: E402
    parse_test_name, char_timescale, extract_radius_history,
    sci_latex, is_ignored, _cell_dirs, IMG_DIR, _REPO,
)

# ============================================================================
# ============================  CONFIGURATION  ===============================
# ============================================================================

# Glob roots searched for output_R*_Sigma* plot directories.
# Where to look for output_R*_Sigma* plotfile dirs.  The UNIT_TEST roots (where
# the unit-test matrix lands) are listed first; INCLINE /mmfs1 is the default
# top entry.  The aggregator can also pass an explicit root as argv[1].
SEARCH_ROOTS = [
    # --- UNIT_TEST_3D matrix (INCLINE /mmfs1 first, then desktop) ---
    "/mmfs1/home/ttryon/flames/bin/tests/FlowLaplace/UNIT_TEST_3D",
    os.path.join(_REPO, "bin", "tests", "FlowLaplace", "UNIT_TEST_3D"),
    os.path.join(os.getcwd(), "bin", "tests", "FlowLaplace", "UNIT_TEST_3D"),
    # --- legacy / standalone locations ---
    os.path.join(_REPO, "tests", "FlowLaplace"),
    os.path.join(_REPO, "bin", "tests", "FlowLaplace"),
    os.path.join(os.getcwd(), "tests", "FlowLaplace"),
    os.path.join(os.getcwd(), "bin", "tests", "FlowLaplace"),
]
if len(sys.argv) > 1:                    # explicit root override (aggregator)
    SEARCH_ROOTS.insert(0, sys.argv[1])

# Color per radius, line style per sigma (extend if the matrix grows).
R_COLORS = {1.0e-3: "#1f77b4", 1.0e-4: "#ff7f0e", 1.0e-5: "#2ca02c"}
S_STYLES = {0.0: ":", 0.036: "-", 0.073: "--"}
DEFAULT_COLOR = "#7f7f7f"
DEFAULT_STYLE = "-."

STYLE = {
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  9,
    "axes.linewidth":   1.0,
    "lines.linewidth":  1.7,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
}
FIGSIZE  = (8.0, 7.5)
SAVE_EPS = True
SAVE_NAME = "Laplace_AGG_3D"
# ============================================================================


def discover_tests():
    """Return {test_name: output_dir} for every run with plotfiles."""
    found = {}
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        for d in os.listdir(root):
            if is_ignored(d):                       # skip *.old.* backup dirs
                continue
            m = re.match(r"output_(R[^_]+_Sigma.+)", d)
            full = os.path.join(root, d)
            if m and os.path.isdir(full) and _cell_dirs(full):
                found.setdefault(m.group(1), full)
    return dict(sorted(found.items(),
                       key=lambda kv: parse_test_name(kv[0])))


def style_for(R0, sigma):
    return (R_COLORS.get(R0, DEFAULT_COLOR), S_STYLES.get(sigma, DEFAULT_STYLE))


def main():
    plt.rcParams.update(STYLE)
    print("=" * 70)
    print("FlowLaplace aggregate analysis")
    print("=" * 70)

    tests = discover_tests()
    if not tests:
        print("  [error] no FlowLaplace output found under:")
        for r in SEARCH_ROOTS:
            print(f"          {r}")
        print("  Run the sims first, e.g.:")
        print("          ./bin/hydro2-2d-g++ tests/FlowLaplace/R1.0e-4_Sigma7.3e-2")
        return 2
    print(f"  found {len(tests)} test(s): {', '.join(tests)}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True)
    ax1.axhline(1.0, color="0.4", lw=0.8, ls=":")

    summary = []
    for name, out in tests.items():
        R0, sigma = parse_test_name(name)
        try:
            t, R = extract_radius_history(out, axis="x")
        except Exception as e:
            print(f"  [warn] {name}: extraction failed ({e})")
            continue
        tau = char_timescale(R0, sigma)
        tn = t / tau
        rn = R / R0
        err = np.abs(rn - 1.0)
        color, ls = style_for(R0, sigma)
        label = rf"$R_0={sci_latex(R0)}$, $\sigma={sci_latex(sigma)}$"
        ax1.plot(tn, rn, color=color, ls=ls, label=label)
        ax2.semilogy(tn, np.maximum(err, 1e-16), color=color, ls=ls, label=label)
        summary.append((name, np.nanmax(err), rn[-1]))

    ax1.set_ylabel(r"$R/R_0$")
    ax1.set_title("FlowLaplace surface-tension sweep  "
                  "(color = $R_0$, style = $\\sigma$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", ncol=2, framealpha=0.9)

    ax2.set_ylabel(r"$|R/R_0 - 1|$")
    ax2.set_xlabel(r"$t/\tau$")
    ax2.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    png = os.path.join(IMG_DIR, f"{SAVE_NAME}_radius.png")
    fig.savefig(png)
    if SAVE_EPS:
        fig.savefig(os.path.join(IMG_DIR, f"{SAVE_NAME}_radius.eps"))
    plt.close(fig)
    print(f"  wrote {png}")

    print("\n  sensitivity summary (max |R/R0-1|, sorted worst-first):")
    for name, e, last in sorted(summary, key=lambda s: -s[1]):
        flag = "  <-- sensitive" if e > 0.05 else ""
        print(f"    {name:<22s} max_err={e * 100:7.3f} %   R/R0_final={last:.4f}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
