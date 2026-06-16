# -*- coding: utf-8 -*-
"""
===============================================================================
CHEN-2D FAR-FIELD (r_inf) SENSITIVITY SWEEP  --  NO simulation data read.
===============================================================================

PURPOSE:
    The 2D-cylindrical bubble-oscillation frequency depends on the inertia
    factor  ln(r_inf / R)  (Chen-2D RPE). Unlike the 3D spherical case (whose
    inertia ~ R, geometry-insensitive far away), the 2D cylindrical frequency
    is LOGARITHMICALLY sensitive to where the effective far-field pressure
    sits. When a Sch20 spherical test is ported to a truncated 2D Cartesian
    domain with NSCBC, r_inf is NOT precisely the domain half-width -- so the
    analytical Chen-2D reference itself is uncertain.

    This script makes that sensitivity explicit: it sweeps r_inf over 10
    values and shows (a) how the nonlinear Chen-2D R(t) trace shifts in
    frequency and (b) the period / linearized natural frequency vs r_inf.

    It reads NO AMReX output. It only integrates the shared Chen-2D ODE for
    the Sch20_Oscillating physical parameters. Use it to (i) decide what
    r_inf to put in analyze_Sch20_Oscillating.py and (ii) put an error bar
    on the analytical curve in the paper.

REFERENCES:
    Schmidmayer-Bryngelson-Colonius JCP 402 (2020) Sec. 4 (problem setup)
    Chen et al. (2D cylindrical RPE; see rayleigh_plesset_solver.py docstring)

USAGE:
    python tests/FlowRayleighPlesset/reference/sweep_Sch20_rinf.py
===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Shared RPE/KM/Chen-2D ODE solver (do NOT read amrex output here)
sys.path.insert(0, os.path.join(_HERE, "..", "OtherTests", "reference"))
from rayleigh_plesset_solver import (                                   # noqa: E402
    Params, solve, rayleigh_collapse_time, linear_natural_frequency_chen2d,
)


# ============================================================================
# CONFIGURATION  (everything tunable lives here)
# ============================================================================

# ---- Curve toggles ----
SHOW_NONLINEAR_ODE  = True    # left panel: full Chen-2D R(t) per r_inf
SHOW_LINEAR_THEORY  = True    # right panel: linearized natural frequency
SHOW_ODE_PERIOD     = True    # right panel: period measured from the ODE trace

# ---- Physical parameters (MUST match input Sch20_Oscillating) ----
RHO_L     = 1000.0            # liquid density          [kg/m^3]
P_INF     = 1.0e5             # far-field liquid pressure [Pa]
P_B       = 1.0e4             # initial bubble pressure  [Pa]   (solver p_g0)
GAMMA_GAS = 1.4               # gas polytropic index
SIGMA     = 0.0               # surface tension         [N/m]
MU_L      = 0.0               # liquid viscosity        [Pa s]
C_L       = 1533.0            # liquid sound speed      [m/s]   (unused by chen2d)
R0        = 0.02              # initial bubble radius   [m]
RDOT0     = 0.0               # initial wall velocity   [m/s]

# ---- r_inf sweep (the far-field cutoff, in units of R0) ----
# Brackets the truncated sim domain (10*R0 = 0.20 m) and Sch20's 320*R0.
R_INF_OVER_R0 = [2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 200.0, 320.0]

# ---- Time window ----
# Generous enough to show >= 2 cycles even for the slowest (largest r_inf).
T_END_OVER_TAUC = 12.0        # in units of tau_c

# ---- Non-dimensionalization toggles ----
NONDIM_TIME   = True          # x-axis t/tau_c (else seconds)
NONDIM_RADIUS = True          # y-axis R/R0    (else metres)

# ---- Plot styling (publication knobs) ----
FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 9
FONT_SIZE_TICK   = 11
LINE_WIDTH       = 1.8
DPI              = 180
FIG_WIDTH        = 13.0
FIG_HEIGHT       = 5.2
CMAP_NAME        = "viridis"          # color gradient over the r_inf sweep
COLOR_LINEAR     = "tab:red"
COLOR_ODE_PERIOD = "tab:blue"
LEGEND_LOC_LEFT  = "upper right"
LEGEND_LOC_RIGHT = "best"

# ---- Strings (every label the user may rephrase for the paper) ----
SUPTITLE_STR   = "Chen-2D far-field ($r_\\infty$) sensitivity -- Sch20 Oscillating params"
LEFT_TITLE     = "Bubble-wall trace vs $r_\\infty$"
RIGHT_TITLE    = "Oscillation frequency vs $r_\\infty$"
XLABEL_LEFT    = r"$t / \tau_c$"
YLABEL_LEFT    = r"$R / R_0$"
XLABEL_RIGHT   = r"$r_\infty / R_0$"
YLABEL_RIGHT   = r"period $T\;[\mathrm{ms}]$"
LABEL_LINEAR   = r"linear theory  $T_0 = 2\pi/\omega_0$"
LABEL_ODEPER   = r"nonlinear ODE  (peak-to-peak)"
LEGEND_FMT     = r"$r_\infty/R_0 = {:g}$"
SAVE_NAME      = "Sch20_rinf_sweep"


# ============================================================================
# HELPERS
# ============================================================================

def make_params(r_inf):
    return Params(
        rho_l=RHO_L, p_inf=P_INF, mu_l=MU_L, c_l=C_L,
        gamma_gas=GAMMA_GAS, p_g0=P_B, p_v=0.0, sigma=SIGMA,
        R0=R0, Rdot0=RDOT0, r_inf=r_inf,
    )


def local_minima(t, R):
    if len(R) < 3:
        return np.array([]), np.array([])
    m = (R[1:-1] < R[:-2]) & (R[1:-1] < R[2:])
    idx = np.where(m)[0] + 1
    return t[idx], R[idx]


def measured_period(t, R):
    """Mean peak-to-peak time from successive minima (collapse-to-collapse).
    Returns nan if fewer than 2 minima are resolved in the window."""
    tm, _ = local_minima(t, R)
    if len(tm) >= 2:
        return float(np.mean(np.diff(tm)))
    return float("nan")


# ============================================================================
# MAIN
# ============================================================================

def main():
    # tau_c is r_inf-independent (depends only on R0, rho_l, p_inf, p_b)
    P_ref = make_params(R_INF_OVER_R0[0] * R0)
    tau_c = rayleigh_collapse_time(P_ref)
    if not np.isfinite(tau_c) or tau_c <= 0:
        tau_c = 1.0
        print("WARNING: tau_c non-finite; falling back to seconds.")
    t_end = T_END_OVER_TAUC * tau_c

    cmap = plt.get_cmap(CMAP_NAME)
    n = len(R_INF_OVER_R0)

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    rinf_vals, T_ode, T_lin = [], [], []

    print(f"{'r_inf/R0':>10} {'r_inf [m]':>12} "
          f"{'T_ODE [ms]':>12} {'T_lin [ms]':>12}")
    print("-" * 50)

    for i, ratio in enumerate(R_INF_OVER_R0):
        r_inf = ratio * R0
        P = make_params(r_inf)
        color = cmap(i / max(n - 1, 1))

        t, R, _ = solve(P, t_end, model="chen2d")

        T_meas = measured_period(t, R)
        omega0 = linear_natural_frequency_chen2d(P, R_eq=R0)
        T_linear = (2.0 * np.pi / omega0) if omega0 > 0 else float("nan")

        rinf_vals.append(ratio)
        T_ode.append(T_meas)
        T_lin.append(T_linear)

        print(f"{ratio:>10g} {r_inf:>12.4g} "
              f"{1e3 * T_meas:>12.4g} {1e3 * T_linear:>12.4g}")

        if SHOW_NONLINEAR_ODE:
            x = (t / tau_c) if NONDIM_TIME else t
            y = (R / R0) if NONDIM_RADIUS else R
            axL.plot(x, y, color=color, lw=LINE_WIDTH,
                     label=LEGEND_FMT.format(ratio))

    # ---- Left panel cosmetics ----
    if SHOW_NONLINEAR_ODE:
        axL.set_title(LEFT_TITLE, fontsize=FONT_SIZE_TITLE)
        axL.set_xlabel(XLABEL_LEFT, fontsize=FONT_SIZE_LABEL)
        axL.set_ylabel(YLABEL_LEFT, fontsize=FONT_SIZE_LABEL)
        axL.tick_params(labelsize=FONT_SIZE_TICK)
        axL.grid(True, alpha=0.3)
        axL.legend(loc=LEGEND_LOC_LEFT, fontsize=FONT_SIZE_LEGEND, ncol=2)

    # ---- Right panel: period vs r_inf ----
    rinf_arr = np.array(rinf_vals)
    if SHOW_ODE_PERIOD:
        axR.semilogx(rinf_arr, 1e3 * np.array(T_ode), "o-",
                     color=COLOR_ODE_PERIOD, lw=LINE_WIDTH,
                     label=LABEL_ODEPER)
    if SHOW_LINEAR_THEORY:
        axR.semilogx(rinf_arr, 1e3 * np.array(T_lin), "s--",
                     color=COLOR_LINEAR, lw=LINE_WIDTH,
                     label=LABEL_LINEAR)
    axR.set_title(RIGHT_TITLE, fontsize=FONT_SIZE_TITLE)
    axR.set_xlabel(XLABEL_RIGHT, fontsize=FONT_SIZE_LABEL)
    axR.set_ylabel(YLABEL_RIGHT, fontsize=FONT_SIZE_LABEL)
    axR.tick_params(labelsize=FONT_SIZE_TICK)
    axR.grid(True, alpha=0.3, which="both")
    axR.legend(loc=LEGEND_LOC_RIGHT, fontsize=FONT_SIZE_LEGEND)

    fig.suptitle(SUPTITLE_STR, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    img_dir = os.path.join(_HERE, "Images")
    os.makedirs(img_dir, exist_ok=True)
    png = os.path.join(img_dir, SAVE_NAME + ".png")
    eps = os.path.join(img_dir, SAVE_NAME + ".eps")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(eps, bbox_inches="tight")
    print("-" * 50)
    print(f"tau_c = {1e3 * tau_c:.4g} ms   (r_inf-independent)")
    print(f"Saved: {png}")
    print(f"Saved: {eps}")

    # Quantify the logarithmic sensitivity
    valid = np.isfinite(T_ode)
    if valid.sum() >= 2:
        T_arr = np.array(T_ode)[valid]
        spread = (T_arr.max() - T_arr.min()) / T_arr.min() * 100.0
        print(f"ODE period spread across the r_inf sweep: {spread:.1f}%")
        print("(This is the analytical uncertainty band on the Chen-2D "
              "reference for a truncated 2D domain.)")


if __name__ == "__main__":
    main()
