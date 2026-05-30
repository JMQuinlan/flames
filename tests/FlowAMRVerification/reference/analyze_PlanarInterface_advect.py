# -*- coding: utf-8 -*-
"""
===============================================================================
PLANAR INTERFACE ADVECTION -- AMR vs no-AMR comparison
===============================================================================

Companion to:
  tests/FlowAMRVerification/input_PlanarInterface_advect_amr
  tests/FlowAMRVerification/input_PlanarInterface_advect_noamr

The two runs share an identical physical setup: a planar gas/water
interface at x = -0.25 mm advected at u = 10 m/s.  The only difference
is AMR on vs off (both at the same finest resolution of 256 x 128).

This script:
  1. Reads the AMReX plotfiles from both runs.
  2. At each plot time, extracts the y-averaged eta(x) and finds
     x_interface(t) = the eta = 0.5 crossing.
  3. Overlays x_interface(t) for AMR, no-AMR, and the analytical
     x_0 + U * t.
  4. Reports the AMR-vs-no-AMR divergence and the absolute error
     of each run against the analytical line.

USAGE:
  cd <flames2-root>
  python tests/FlowAMRVerification/reference/analyze_PlanarInterface_advect.py

OUTPUTS:
  tests/FlowAMRVerification/reference/PlanarAdvect_compare.png
  Console summary.
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import yt
    yt.funcs.mylog.setLevel(40)
except ImportError:
    print("ERROR: yt not installed.  pip install yt", file=sys.stderr)
    sys.exit(1)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

# Resolve paths relative to this script (works from any cwd).
# Script lives at <repo>/tests/FlowAMRVerification/reference/analyze_*.py
# so the flames2 root is three levels up.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
# Alamo writes plotfiles relative to the binary's cwd (bin/), so the
# "./tests/..." plot_file path lands under bin/tests/..., matching the
# convention used by tests/FlowCouette/reference/couette_analysis.py.
AMR_DIR   = os.path.join(_REPO_ROOT, "bin", "tests", "FlowAMRVerification",
                         "output_PlanarInterface_advect_amr")
NOAMR_DIR = os.path.join(_REPO_ROOT, "bin", "tests", "FlowAMRVerification",
                         "output_PlanarInterface_advect_noamr")
#OUT_PNG   = os.path.join(_SCRIPT_DIR, "/Images/PlanarAdvect_compare.png")
OUT_PNG   = "./Images/PlanarAdvect_compare.png"


# Analytical reference (must match the input files).
X_INTERFACE_0 = -0.0002    # initial interface position [m]
U_FLOW        = 10.0       # advection velocity [m/s]

# Domain (must match the input files).
X_LO, X_HI = -0.00025, 0.00025
Y_LO, Y_HI = -0.000125, 0.000125
NX_FINE, NY_FINE = 128, 64


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def list_plotfiles(directory):
    """Return sorted plotfile paths from an Alamo/AMReX output directory.

    Alamo names cell-centered plotfiles "NNNNNcell" (e.g., "00021cell");
    AMReX-default is "<prefix><nnnnn>" (often "plt00021").  Match both.
    """
    if not os.path.isdir(directory):
        return []
    candidates = sorted(glob.glob(os.path.join(directory, "*cell"))
                      + glob.glob(os.path.join(directory, "*.plt*"))
                      + glob.glob(os.path.join(directory, "plt*")))
    plots = [p for p in candidates
             if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]
    return plots


def extract_interface_position(plotfile_path):
    """Return (time, x_interface) for a single plotfile.

    x_interface is the y-averaged eta = 0.5 crossing along x.
    """
    ds = yt.load(plotfile_path)
    t = float(ds.current_time)

    # Resample to a uniform grid at the finest level.  covering_grid
    # handles AMR cleanly: fine-level cells where they exist, coarse
    # cells elsewhere, all interpolated to the requested resolution.
    cg = ds.covering_grid(
        level=ds.index.max_level,
        left_edge=[X_LO, Y_LO, 0.0],
        dims=[NX_FINE, NY_FINE, 1],
    )
    eta_2d = np.asarray(cg["eta"]).squeeze()   # shape (NX, NY)

    # y-average to a 1D profile.
    eta_x = eta_2d.mean(axis=1)
    x_cell = np.linspace(X_LO + 0.5 * (X_HI - X_LO) / NX_FINE,
                         X_HI - 0.5 * (X_HI - X_LO) / NX_FINE,
                         NX_FINE)

    # Find the eta = 0.5 crossing.  eta runs from 1 (gas, left) to 0 (water,
    # right), so we look for the first index where eta drops below 0.5 and
    # linearly interpolate.
    below = np.where(eta_x < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
        # Interface left the domain or never present; fall back to NaN.
        return t, float("nan")
    i = below[0]
    e1, e2 = eta_x[i - 1], eta_x[i]
    x1, x2 = x_cell[i - 1], x_cell[i]
    x_iface = x1 + (0.5 - e1) * (x2 - x1) / (e2 - e1)
    return t, float(x_iface)


def trace(directory, label):
    plots = list_plotfiles(directory)
    if not plots:
        print(f"  [{label}] no plotfiles found in {directory}")
        return np.array([]), np.array([])
    ts, xs = [], []
    for p in plots:
        try:
            t, x = extract_interface_position(p)
        except Exception as e:
            print(f"  [{label}] skipping {os.path.basename(p)}: {e}")
            continue
        ts.append(t)
        xs.append(x)
    return np.array(ts), np.array(xs)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PLANAR INTERFACE ADVECTION -- AMR vs no-AMR")
    print("=" * 70)
    print(f"  AMR   output dir = {AMR_DIR}")
    print(f"  noAMR output dir = {NOAMR_DIR}")
    print(f"  analytical: x(t) = {X_INTERFACE_0:+.2e} + {U_FLOW}*t")
    print()

    print("  Extracting AMR trace ...")
    t_amr,   x_amr   = trace(AMR_DIR,   "AMR")
    print(f"    {len(t_amr)} samples")
    print("  Extracting no-AMR trace ...")
    t_noamr, x_noamr = trace(NOAMR_DIR, "noAMR")
    print(f"    {len(t_noamr)} samples")

    if len(t_amr) == 0 and len(t_noamr) == 0:
        print("\n  Nothing to plot.  Run the inputs first.")
        return

    # Analytical reference over the union of available times.
    t_all = np.concatenate([t_amr, t_noamr])
    t_min, t_max = t_all.min(), t_all.max()
    t_ref = np.linspace(t_min, t_max, 200)
    x_ref = X_INTERFACE_0 + U_FLOW * t_ref

    # ---- Error metrics --------------------------------------------------
    if len(t_amr):
        err_amr = x_amr - (X_INTERFACE_0 + U_FLOW * t_amr)
        print(f"\n  AMR   vs analytical:  max |err| = {np.nanmax(np.abs(err_amr))*1e6:.2f} um")
    if len(t_noamr):
        err_noamr = x_noamr - (X_INTERFACE_0 + U_FLOW * t_noamr)
        print(f"  noAMR vs analytical:  max |err| = {np.nanmax(np.abs(err_noamr))*1e6:.2f} um")
    if len(t_amr) and len(t_noamr):
        # Resample noAMR onto AMR times via linear interp for the divergence metric.
        x_noamr_at_amr = np.interp(t_amr, t_noamr, x_noamr)
        div = x_amr - x_noamr_at_amr
        print(f"  AMR vs noAMR divergence: max |dx| = {np.nanmax(np.abs(div))*1e6:.2f} um")

    # ---- Plot -----------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_ref * 1e6, x_ref * 1e6, "k:", lw=2,
            label=f"analytical: $x_0 + U t$")
    if len(t_noamr):
        ax.plot(t_noamr * 1e6, x_noamr * 1e6, "b-", lw=2, marker="o", ms=4,
                label=f"no-AMR ({NX_FINE}x{NY_FINE} uniform)")
    if len(t_amr):
        ax.plot(t_amr * 1e6, x_amr * 1e6, "r--", lw=2, marker="s", ms=4,
                label="AMR (2 levels)")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("interface position $x_{\\eta=0.5}$ [um]")
    ax.set_title(f"Planar interface, U = {U_FLOW:.0f} m/s advection (air/water)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n  Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
