# -*- coding: utf-8 -*-
"""
===============================================================================
PLANAR SHOCK-INTERFACE IMPINGEMENT -- AMR vs no-AMR comparison
===============================================================================

Companion to:
  tests/FlowAMRVerification/input_PlanarInterface_shock_amr
  tests/FlowAMRVerification/input_PlanarInterface_shock_noamr

Both runs share an identical physical setup: a Ma = 1.5 shock in air
impinging on a planar gas/water interface at x = 0.  Same finest resolution
(512 x 128), only AMR on/off differs.

This script:
  1. Reads the AMReX plotfiles from both runs.
  2. At each plot time, extracts y-averaged eta(x) and finds the
     x_interface(t) = eta = 0.5 crossing.
  3. Plots x_interface(t) for both runs.
  4. Estimates the impingement time (kink in the trace) and the
     pre/post-impingement interface velocities by finite difference.

USAGE:
  cd <flames2-root>
  python tests/FlowAMRVerification/reference/analyze_PlanarInterface_shock.py

OUTPUTS:
  tests/FlowAMRVerification/reference/PlanarShock_compare.png
  Console summary with pre- / post-impingement velocity for each run.
"""

import glob
import os
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
# Alamo writes plotfiles relative to the binary's cwd (bin/).
AMR_DIR   = os.path.join(_REPO_ROOT, "bin", "tests", "FlowAMRVerification",
                         "output_PlanarInterface_shock_amr")
NOAMR_DIR = os.path.join(_REPO_ROOT, "bin", "tests", "FlowAMRVerification",
                         "output_PlanarInterface_shock_noamr")
OUT_PNG   = "./Images/PlanarShock_compare.png"
#OUT_PNG   = os.path.join(_SCRIPT_DIR, "./Images/PlanarShock_compare.png")

# Domain (must match inputs).
X_LO, X_HI = -0.0025, 0.0025
Y_LO, Y_HI = -0.0005, 0.0005
NX_FINE, NY_FINE = 256, 64

# Known initial / shock parameters (for reference annotation only).
X_INTERFACE_0  = 0.0       # m
X_SHOCK_0      = -0.001    # m
SHOCK_SPEED    = 510.4     # m/s, Ma=1.5 in air
T_IMPACT_EST   = abs(X_INTERFACE_0 - X_SHOCK_0) / SHOCK_SPEED  # ~2.0 us


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def list_plotfiles(directory):
    """Alamo "NNNNNcell" plus AMReX-default fallbacks."""
    if not os.path.isdir(directory):
        return []
    candidates = sorted(glob.glob(os.path.join(directory, "*cell"))
                      + glob.glob(os.path.join(directory, "*.plt*"))
                      + glob.glob(os.path.join(directory, "plt*")))
    return [p for p in candidates
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]


def extract_interface_position(plotfile_path):
    ds = yt.load(plotfile_path)
    t = float(ds.current_time)
    cg = ds.covering_grid(
        level=ds.index.max_level,
        left_edge=[X_LO, Y_LO, 0.0],
        dims=[NX_FINE, NY_FINE, 1],
    )
    eta_2d = np.asarray(cg["eta"]).squeeze()
    eta_x = eta_2d.mean(axis=1)
    x_cell = np.linspace(X_LO + 0.5 * (X_HI - X_LO) / NX_FINE,
                         X_HI - 0.5 * (X_HI - X_LO) / NX_FINE,
                         NX_FINE)
    below = np.where(eta_x < 0.5)[0]
    if len(below) == 0 or below[0] == 0:
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


def impingement_velocity(t, x, t_split):
    """Estimate post-impingement velocity by linear fit to (t > t_split)."""
    mask = (t > t_split) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan")
    coeffs = np.polyfit(t[mask], x[mask], 1)
    return float(coeffs[0])  # slope = velocity


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PLANAR SHOCK-INTERFACE IMPINGEMENT -- AMR vs no-AMR")
    print("=" * 70)
    print(f"  AMR   output dir = {AMR_DIR}")
    print(f"  noAMR output dir = {NOAMR_DIR}")
    print(f"  shock speed (Ma=1.5 in air) = {SHOCK_SPEED:.1f} m/s")
    print(f"  expected impingement at t ~ {T_IMPACT_EST*1e6:.2f} us")
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

    # ---- Post-impingement velocity (use t > T_IMPACT_EST + 0.5 us) -----
    t_split = T_IMPACT_EST + 0.5e-6
    if len(t_amr):
        v_amr_post = impingement_velocity(t_amr, x_amr, t_split)
        print(f"\n  AMR   post-impingement contact velocity ~ {v_amr_post:.1f} m/s")
    if len(t_noamr):
        v_noamr_post = impingement_velocity(t_noamr, x_noamr, t_split)
        print(f"  noAMR post-impingement contact velocity ~ {v_noamr_post:.1f} m/s")
    if len(t_amr) and len(t_noamr):
        x_noamr_at_amr = np.interp(t_amr, t_noamr, x_noamr)
        div = x_amr - x_noamr_at_amr
        print(f"  AMR vs noAMR divergence:  max |dx| = {np.nanmax(np.abs(div))*1e6:.2f} um")

    # ---- Plot ----------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    if len(t_noamr):
        ax.plot(t_noamr * 1e6, x_noamr * 1e6, "b-", lw=2, marker="o", ms=4,
                label=f"no-AMR ({NX_FINE}x{NY_FINE} uniform)")
    if len(t_amr):
        ax.plot(t_amr * 1e6, x_amr * 1e6, "r--", lw=2, marker="s", ms=4,
                label="AMR (2 levels)")
    ax.axvline(T_IMPACT_EST * 1e6, color="grey", lw=1, ls=":",
               label=f"shock-arrival estimate ({T_IMPACT_EST*1e6:.2f} us)")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("interface position $x_{\\eta=0.5}$ [um]")
    ax.set_title("Ma=1.5 shock impinging on air/water interface")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n  Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
