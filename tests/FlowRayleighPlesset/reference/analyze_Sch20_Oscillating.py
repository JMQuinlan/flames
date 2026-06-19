# -*- coding: utf-8 -*-
"""
===============================================================================
SCH20 OSCILLATING-BUBBLE ANALYSIS  --  Case 1, low pressure ratio
(p_inf / p_b = 10).  Focus: oscillation period and acoustic-radiation
amplitude-decay rate.
===============================================================================

PURPOSE:
    Validate the flames2 simulation of the Schmidmayer-Bryngelson-Colonius
    (2020) Table 1, Case 1 spherical air bubble in water against three
    analytical bubble-dynamics references:

        - 3D Keller-Miksis        (km)     -- liquid-compressibility benchmark
        - 3D Rayleigh-Plesset     (rp)     -- incompressible 3D context
        - 2D Chen cylindrical RPE (chen2d) -- geometric match for the 2D
                                              Cartesian (cylindrical) sim

    At p_inf / p_b = 10 the bubble collapses, rebounds, and oscillates with
    acoustic-radiation damping. Keller-Miksis has that radiation damping that
    incompressible RP completely lacks, so the headline differentiators are:

        - oscillation period  T   (peak-to-peak time)
        - amplitude-decay rate alpha from successive peak amplitudes
            R_n - R_eq = R_0 * exp(-alpha * t_n)

    The classical leading-order linearized result (Brennen, "Cavitation and
    Bubble Dynamics" 1995, Sec. 4.4):

        alpha_KM = omega_0^2 * R_eq / (2 * c_l)        (acoustic radiation, 3D)

    For 2D Cartesian sims the geometric reference is Chen 2D cylindrical
    (logarithmic far-field); KM remains the compressibility-correction
    benchmark even though the 2D-vs-3D geometry is offset.

USAGE:
    python tests/FlowRayleighPlesset/reference/analyze_Sch20_Oscillating.py

OUTPUTS:
    Images/Sch20_Oscillating.png   (and .eps for the paper)
    Console summary: measured vs analytical period and acoustic-radiation
    amplitude-decay rate, for sim vs KM / RP / Chen2D.

REFERENCES:
    Schmidmayer, Bryngelson, Colonius, "An assessment of multiregion methods
        for two-phase flow simulations", J. Comput. Phys. 402 (2020) 109080,
        Sec. 4 (problem setup), Table 1 (ICs).
    Keller & Miksis, "Bubble oscillations of large amplitude", JASA 68 (1980).
    Brennen, "Cavitation and Bubble Dynamics" (1995), Sec. 4.4 (linearized
        RP / KM, acoustic-radiation damping).

===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Bring in the shared RPE / KM / Chen2D ODE solver (DO NOT modify it).
sys.path.insert(0, os.path.join(_HERE, '..', 'OtherTests', 'reference'))
from rayleigh_plesset_solver import (                                  # noqa: E402
    Params, solve,
    rayleigh_collapse_time,
    linear_natural_frequency_3d, linear_natural_frequency_chen2d,
)


# ============================================================================
# ============================  CONFIGURATION  ===============================
# Everything tunable lives in this block. Adjust here -- no hunting needed.
# ============================================================================

# ===== MODEL TOGGLES =====
# Default: 2D Chen on (the only geometric match for the 2D Cartesian sim).
# Flip KM_3D / RPE_3D to True to overlay the 3D-spherical references for
# context -- they have different periods/amplitudes because the geometry
# differs, so they won't overlay the sim, but are useful as bounds.
SHOW_KM_3D   = False     # 3D Keller-Miksis
SHOW_RPE_3D  = False     # 3D Rayleigh-Plesset
SHOW_CHEN_2D = True      # 2D Chen cylindrical (matches the 2D Cartesian sim geometry)
SHOW_SIM     = True      # flames2 simulation R(t)

# ===== PHYSICAL PARAMETERS =====
# Sch20 Table 1, Case 1 (p_inf / p_b = 10). Must match Sch20_Oscillating input.
#   c_l = sqrt(gamma_l (p_inf + pi_l) / rho_l) = sqrt(2.35*(1e5+1e9)/1000)
P = Params(
    rho_l     = 1000.0,
    p_inf     = 1.0e5,       # far-field liquid pressure
    mu_l      = 0.0,
    c_l       = 1533.0,      # Tammann water sound speed (KM only)
    gamma_gas = 1.4,
    p_g0      = 1.0e4,       # initial bubble pressure p_b
    p_v       = 0.0,
    sigma     = 0.0,
    R0        = 0.02,
    Rdot0     = 0.0,
    r_inf     = 20*0.02 #20 * R0 or so :p #0.20,        # Chen-2D log far-field cutoff = half-domain
)

# ===== TIME / DOMAIN =====
# tau_c = 0.915 R0 sqrt(rho_l / p_inf) ~ 1.83e-3 s ; stop_time = 1.0e-2 s.
T_END = 1.0e-2

# Path to the AMReX plotfile output dir. Default is relative to the repo
# (run from <repo>/bin); the absolute OVERRIDE line below can be uncommented
# and edited for a different machine / scratch location.
OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowRayleighPlesset", "output_Sch20_Oscillating_minmod_NO_p0",
))
# OVERRIDE (uncomment + edit for your machine):
OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating")

# ===== PLOT STYLING (publication knobs) =====
FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11

LINE_WIDTH_KM   = 2.5
LINE_WIDTH_RPE  = 1.5
LINE_WIDTH_CHEN = 2.5
LINE_WIDTH_SIM  = 1.5
MARKER_SIZE_SIM = 4

COLOR_KM   = 'tab:blue'
COLOR_RPE  = '0.5'
COLOR_CHEN = 'tab:green'
COLOR_SIM  = 'tab:red'

DPI        = 180
FIG_WIDTH  = 11
FIG_HEIGHT = 6.5
LEGEND_LOC = 'best'

# ===== AXIS LABELS / TITLE (every string the user might rephrase) =====
TITLE_STR   = "Sch20 Oscillating Bubble: R(t)"
XLABEL_STR  = r"$t / \tau_c$"
YLABEL_STR  = r"$R / R_0$"
LABEL_KM    = "Keller--Miksis (3D)"
LABEL_RPE   = "Rayleigh--Plesset (3D)"
LABEL_CHEN  = "Chen (2D cylindrical)"
LABEL_SIM   = "Sim"
NONDIM_RADIUS = True     # plot R / R0  instead of R [m]
SAVE_NAME     = "Sch20_Oscillating"

# Time-axis units.  Options:
#   'ms'     -- milliseconds (recommended default; most readable for tau_c ~ 1-10 ms)
#   'us'     -- microseconds (use for tau_c < 1 ms)
#   'nondim' -- t / tau_c (Rayleigh non-dimensional, the v2 default)
TIME_UNIT = 'ms'

# ===== IC-DIAGNOSTIC OPTIONS =====
# Plot p_mix(R) along the x-axis ray for the first DIAG_N_FRAMES plotfiles to
# inspect the initial pressure profile and the diffuse-interface "relaxation
# ledge" that would explain a small initial wobble in R(t).  See the analysis
# in the chat: the eq-(27) liquid IC + uniform gas IC are NOT in mechanical
# equilibrium inside the diffuse eta band; the first pressure-relaxation
# Newton step pulls the gas-side pressure up toward the liquid-side IC (the
# Tammann liquid is ~1e5x stiffer in B = sum a_k gamma_k pi_k/(g-1)).  The
# predicted ledge is computed analytically from the input-file IC constants.
DIAG_IC_PRESSURE   = True
DIAG_N_FRAMES      = 5
DIAG_R_MAX_FACTOR  = 3.0      # max radius (in units of R0) to show on x-axis
DIAG_EPSILON       = 0.0016   # eta IC tanh width (must match input file)
DIAG_GAMMA_LIQ     = 2.35     # eos0 (Tammann liquid)
DIAG_PI_LIQ        = 1.0e9
DIAG_GAMMA_GAS     = 1.4      # eos1 (ideal gas)
DIAG_PI_GAS        = 0.0
DIAG_SAVE_NAME     = "Sch20_IC_PressureDiagnostic"

# ============================================================================
# ==========================  END CONFIGURATION  =============================
# ============================================================================

IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# OSCILLATION-METRIC HELPERS
# ============================================================================

def find_local_minima(t, R):
    """Return arrays (t_min, R_min) of all interior local minima of R(t).
    Uses a simple 3-point comparison; robust enough for smooth ODE output and
    moderately-noisy simulation data."""
    if len(R) < 3:
        return np.array([]), np.array([])
    is_min = (R[1:-1] < R[:-2]) & (R[1:-1] < R[2:])
    idx = np.where(is_min)[0] + 1
    return t[idx], R[idx]


def find_local_maxima(t, R):
    """Return arrays (t_max, R_max) of all interior local maxima."""
    if len(R) < 3:
        return np.array([]), np.array([])
    is_max = (R[1:-1] > R[:-2]) & (R[1:-1] > R[2:])
    idx = np.where(is_max)[0] + 1
    return t[idx], R[idx]


def estimate_period(t, R):
    """Period from average peak-to-peak time (using minima first, then maxima
    as a fallback). Returns (T, n_periods_found, list_of_dt)."""
    tm, _ = find_local_minima(t, R)
    if len(tm) >= 2:
        dts = np.diff(tm)
        return float(np.mean(dts)), len(dts), dts
    tm, _ = find_local_maxima(t, R)
    if len(tm) >= 2:
        dts = np.diff(tm)
        return float(np.mean(dts)), len(dts), dts
    return float('nan'), 0, np.array([])


def estimate_decay_rate(t, R, R_eq):
    """Fit |R - R_eq| evaluated at the local maxima to A0 * exp(-alpha * t).
    Returns (alpha [1/s], t_maxima, amplitudes); nan if < 2 maxima are found.

    We use the maxima of R(t) (i.e. the upper envelope) so the perturbation
    R_max - R_eq is positive, allowing log() without sign flips. R_eq is the
    equilibrium radius (taken as R0 here for unforced oscillations).
    """
    tm, Rm = find_local_maxima(t, R)
    amp = Rm - R_eq
    if len(tm) < 2 or np.any(amp <= 0):
        return float('nan'), tm, amp
    coef = np.polyfit(tm, np.log(amp), 1)
    return float(-coef[0]), tm, amp


def alpha_KM_radiation(R_eq, omega0, c_l):
    """Brennen Sec. 4.4 leading-order acoustic-radiation damping for KM.

    alpha_rad = omega_0^2 * R_eq / (2 * c_l)
    (often written as omega_0 * (omega_0 * R_eq / c_l) / 2; same thing.)
    """
    return 0.5 * omega0 * omega0 * R_eq / c_l


def _crossing_in_1d(x, eta, eta_threshold):
    """Helper: find the lowest-x cell where eta first crosses eta_threshold
    from below.  Returns the interpolated x (radius), or NaN if no crossing.
    Filters NaN/Inf eta cells before searching."""
    finite = np.isfinite(eta) & np.isfinite(x)
    x = x[finite]
    eta = eta[finite]
    if len(x) < 2:
        return np.nan
    # First index where eta crosses up through the threshold (eta[i-1] < t <= eta[i]).
    # The old "eta >= threshold" test triggered on the first cell when bubble
    # had drifted off the ray and the ray-origin cell was already in liquid
    # (eta=1) at x = dx/2, producing a spuriously tiny radius.  Looking for
    # an UPWARD crossing fixes that.
    below = eta < eta_threshold
    above = eta >= eta_threshold
    cross = np.where(below[:-1] & above[1:])[0]
    if len(cross) == 0:
        return np.nan
    i = cross[0] + 1            # the "above" side of the crossing
    denom = eta[i] - eta[i - 1]
    if abs(denom) < 1.0e-30:
        return float(x[i - 1])
    frac = (eta_threshold - eta[i - 1]) / denom
    return float(x[i - 1] + frac * (x[i] - x[i - 1]))


def _extract_via_covering_grid(ds, axis, x_max, eta_threshold,
                               n_y_rows=5, y_half_cells=2):
    """Fallback extraction using a thin 2D covering_grid strip at the
    finest AMR level.  Robust against (a) the ray method missing a contour
    that VisIt sees (yt's ds.ray cell-center sampling skips AMR transitions
    that bilinear isocontouring picks up), and (b) small symmetry-breaking
    drift of the bubble center off y=0 -- the strip samples 2*y_half_cells
    on either side and picks the largest valid crossing across all rows.

    Returns interpolated radius (float) or np.nan.
    """
    try:
        finest = int(ds.index.max_level)
        # finest-level cell size on the chosen axis
        dx_finest = float(ds.domain_width[0]) / (
            int(ds.domain_dimensions[0]) * (2 ** finest))
        dy_finest = float(ds.domain_width[1]) / (
            int(ds.domain_dimensions[1]) * (2 ** finest))
        L = float(ds.domain_right_edge[0]) if axis == "x" \
            else float(ds.domain_right_edge[1])
        x_lim = float(x_max) if x_max is not None else L

        n_x = max(int(np.ceil(x_lim / dx_finest)) + 2, 4)
        half = int(y_half_cells)
        n_y = 2 * half + 1

        if axis == "x":
            left_edge  = [0.0,       -half * dy_finest, 0.0]
            dims       = [n_x, n_y, 1]
            cg = ds.covering_grid(level=finest,
                                   left_edge=left_edge, dims=dims)
            eta = np.array(cg["eta"])[:, :, 0]   # shape (n_x, n_y)
            x_axis = np.linspace(0.5 * dx_finest,
                                 (n_x - 0.5) * dx_finest, n_x)
        else:
            left_edge  = [-half * dx_finest, 0.0, 0.0]
            dims       = [n_y, n_x, 1]
            cg = ds.covering_grid(level=finest,
                                   left_edge=left_edge, dims=dims)
            eta = np.array(cg["eta"])[:, :, 0].T  # rotate so axis-0 is along-axis
            x_axis = np.linspace(0.5 * dy_finest,
                                 (n_x - 0.5) * dy_finest, n_x)

        # Scan every row, pick the LARGEST valid crossing across rows.
        # Largest = furthest from origin = the +x edge of the bubble even
        # if the bubble center has drifted slightly toward +x.  If no row
        # has a crossing, return NaN.
        best = np.nan
        for j in range(n_y):
            r = _crossing_in_1d(x_axis, eta[:, j], eta_threshold)
            if np.isfinite(r) and (not np.isfinite(best) or r > best):
                best = r
        return float(best)
    except Exception:
        return np.nan


def extract_radius_history_robust(amrex_output_dir, eta_threshold=0.5,
                                  axis="x", x_max=None, verbose=False):
    """Same as rayleigh_plesset_solver.extract_radius_history but tolerant
    of individual corrupt / partially-written plotfiles.

    Walks every '*cell' subdirectory of `amrex_output_dir`, attempts to load
    each one with yt, and silently SKIPS any that raise an exception (typical
    cause: a run interrupted while a plotfile was still being written, leaving
    an incomplete Cell_H / Header behind).

    Two-stage extraction at each plotfile:
      1. Fast path -- ds.ray along the chosen axis, scan for first upward
         crossing of eta_threshold (NaN-filtered, true crossing-detection
         instead of the old eta >= t which mis-triggered on edge cells).
      2. Fallback  -- if the ray returns NaN, retry with a thin 2D
         covering_grid strip at the finest AMR level and pick the largest
         valid crossing across rows in [-2*dy, +2*dy].  Catches the cases
         where yt's ray cell-center sampling skips an AMR boundary or where
         the bubble center has drifted slightly off-axis at deep collapse
         (both have been observed reproducing as gaps in R(t) when VisIt's
         isocontour clearly shows the bubble surface in the same frame).

    Parameters
    ----------
    x_max : float or None
        If provided, restrict the eta_threshold search to the interval
        [0, x_max] along the chosen axis.  With x_max ~ 1.5*R0 the search
        stays in the near-bubble region and ignores far-field artifacts.
    verbose : bool
        If True, print a per-frame diagnostic when the fallback fires or
        the radius could not be found, and a summary tally at the end.

    Returns (times, radii, n_skipped) with times/radii sorted by time.
    """
    import yt
    yt.funcs.mylog.setLevel(40)

    if not os.path.isdir(amrex_output_dir):
        return np.array([]), np.array([]), 0

    plot_files = sorted(
        os.path.join(amrex_output_dir, d)
        for d in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell")
    )
    if not plot_files:
        return np.array([]), np.array([]), 0

    times, radii = [], []
    n_skipped = 0
    n_ray_ok   = 0
    n_fb_ok    = 0
    n_failed   = 0
    failed_list = []
    for pf in plot_files:
        try:
            ds = yt.load(pf)
            L = float(ds.domain_right_edge[0])
            if axis == "x":
                ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                             ds.arr([L,   0.0, 0.0], "code_length"))
                x = np.array(ray["x"])
            else:
                ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                             ds.arr([0.0, L,   0.0], "code_length"))
                x = np.array(ray["y"])
            eta = np.array(ray["eta"])
            order = np.argsort(x)
            x, eta = x[order], eta[order]

            # Restrict the eta-threshold search window so spurious far-field
            # eta excursions (NSCBC ghost rings, AMR c/f wiggles, BC
            # extrapolation noise) can't be mis-picked as the bubble
            # interface.  With x_max ~ 1.5*R0 the bubble cleanly fits and
            # nothing past it is considered.
            if x_max is not None:
                mask = (x >= 0.0) & (x <= float(x_max))
                x = x[mask]
                eta = eta[mask]

            # Fast path: 1D crossing on the ray cells.
            r = _crossing_in_1d(x, eta, eta_threshold) if len(x) >= 2 else np.nan

            # Fallback: thin 2D covering_grid strip at finest level.
            if not np.isfinite(r):
                r = _extract_via_covering_grid(ds, axis, x_max,
                                                eta_threshold)
                if np.isfinite(r):
                    n_fb_ok += 1
                    if verbose:
                        print(f"    [fallback] {os.path.basename(pf)}: "
                              f"ray missed eta={eta_threshold:.2f}, "
                              f"covering_grid recovered R={r:.4e}")
                else:
                    n_failed += 1
                    failed_list.append(os.path.basename(pf))
                    if verbose:
                        print(f"    [no contour] {os.path.basename(pf)}: "
                              f"both ray + covering_grid found no "
                              f"eta={eta_threshold:.2f} crossing in window")
            else:
                n_ray_ok += 1

            radii.append(r)
            times.append(float(ds.current_time))
        except Exception:
            # Bad / partial plotfile -- skip and keep going.
            n_skipped += 1
            continue

    if verbose:
        print(f"    extract_radius_history_robust eta={eta_threshold:.2f}: "
              f"ray_ok={n_ray_ok}, fallback_ok={n_fb_ok}, "
              f"no_contour={n_failed}, plotfile_skipped={n_skipped}")
        if failed_list and len(failed_list) <= 12:
            print(f"      failed frames: {', '.join(failed_list)}")
        elif failed_list:
            print(f"      failed frames: {', '.join(failed_list[:6])}, "
                  f"... ({len(failed_list)} total)")

    if not times:
        return np.array([]), np.array([]), n_skipped

    times = np.array(times)
    radii = np.array(radii)
    order = np.argsort(times)
    return times[order], radii[order], n_skipped


# ============================================================================
# IC PRESSURE DIAGNOSTIC
# ============================================================================
#
# Plots p_mix(R) along the +x ray for the first DIAG_N_FRAMES plotfiles, with
# the predicted post-relaxation "ledge" overlaid as a reference marker.
#
# At a diffuse-interface cell of volume-fraction alpha_liq=eta, alpha_gas=1-eta
# with per-phase ICs p_liq_IC(R), p_gas_IC = p_b, the energy-conserving
# pressure relaxation drives both phases to a common p* given by:
#
#     A = sum_k alpha_k / (gamma_k - 1)
#     B = sum_k alpha_k * gamma_k * pi_k / (gamma_k - 1)
#     E_int = sum_k alpha_k * (p_k_IC + gamma_k * pi_k) / (gamma_k - 1)
#     p*    = (E_int - B) / A
#
# For Sch20 Case 1 at R0+epsilon (eta ~= 0.88, p_liq_IC ~= 16667 Pa,
# p_gas_IC = 1e4 Pa, gamma_liq=2.35, pi_liq=1e9, gamma_gas=1.4) this gives
# p* ~= 14.5 kPa -- a ~4.5 kPa jump above p_b that the gas-side cells get
# yanked up to on the very first step.  If you SEE that ledge in the
# first-frame profile, that's the IC mismatch hypothesis confirmed.
# ============================================================================

def _predicted_relaxed_pressure(R, R0, p_inf, p_b, epsilon):
    """Predicted post-relaxation p* at radius R for the Sch20 IC.

    Uses the eq-(27) gradual liquid IC clamped to p_b for R<=R0, eta-tanh
    smearing with width `epsilon`, and the energy-conserving relaxation
    formula above.  Returns p* in Pa.
    """
    p_liq_IC = p_inf - (p_inf - p_b) * R0 / np.maximum(R, R0)
    p_gas_IC = p_b
    eta_liq  = 0.5 * (1.0 + np.tanh((R - R0) / epsilon))   # alpha_liq
    eta_gas  = 1.0 - eta_liq                                # alpha_gas

    g_l, pi_l = DIAG_GAMMA_LIQ, DIAG_PI_LIQ
    g_g, pi_g = DIAG_GAMMA_GAS, DIAG_PI_GAS

    A = eta_liq / (g_l - 1.0) + eta_gas / (g_g - 1.0)
    B = eta_liq * g_l * pi_l / (g_l - 1.0) + eta_gas * g_g * pi_g / (g_g - 1.0)
    E = (eta_liq * (p_liq_IC + g_l * pi_l) / (g_l - 1.0)
         + eta_gas * (p_gas_IC + g_g * pi_g) / (g_g - 1.0))
    return (E - B) / A


def plot_ic_pressure_diagnostic(amrex_output_dir, axis="x"):
    """Overlay p_mix(R) along the +x ray for the first DIAG_N_FRAMES plotfiles.

    Adds:
      - The raw IC liquid-pressure profile (no relaxation): dashed grey.
      - The analytically-predicted post-relaxation profile: solid black.
      - Reference lines at R = R0, R = R0+epsilon, and at p_b, p_inf.

    Saves Images/{DIAG_SAVE_NAME}.png and .eps.  Skips quietly with a
    message if the output dir is missing or yt fails to load.
    """
    if not DIAG_IC_PRESSURE:
        return
    if not os.path.isdir(amrex_output_dir):
        print(f"  [diag] skip IC-pressure diagnostic: {amrex_output_dir} missing")
        return

    import yt
    yt.funcs.mylog.setLevel(40)

    plot_files = sorted(
        os.path.join(amrex_output_dir, d)
        for d in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell")
    )
    if len(plot_files) < 1:
        print(f"  [diag] no plotfiles in {amrex_output_dir}")
        return
    plot_files = plot_files[:DIAG_N_FRAMES]

    R0     = P.R0
    p_inf  = P.p_inf
    p_b    = P.p_g0
    R_max  = DIAG_R_MAX_FACTOR * R0
    R_curve = np.linspace(0.5 * R0, R_max, 400)
    p_pred  = _predicted_relaxed_pressure(R_curve, R0, p_inf, p_b, DIAG_EPSILON)
    p_ic    = p_inf - (p_inf - p_b) * R0 / np.maximum(R_curve, R0)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    cmap = plt.get_cmap('viridis')

    loaded = 0
    for j, pf in enumerate(plot_files):
        try:
            ds = yt.load(pf)
            L  = float(ds.domain_right_edge[0])
            if axis == "x":
                ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                             ds.arr([L,   0.0, 0.0], "code_length"))
                r   = np.array(ray["x"])
            else:
                ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                             ds.arr([0.0, L,   0.0], "code_length"))
                r   = np.array(ray["y"])
            p   = np.array(ray["pressure"])
            order = np.argsort(r)
            r, p  = r[order], p[order]
            mask  = r <= R_max
            color = cmap(j / max(DIAG_N_FRAMES - 1, 1))
            ax.plot(r[mask] / R0, p[mask],
                    'o-', color=color, ms=3, lw=1.2, alpha=0.8,
                    label=f"sim t = {float(ds.current_time)*1e6:.2f} us")
            loaded += 1
        except Exception as exc:
            print(f"  [diag] skipped {os.path.basename(pf)}: {exc}")
            continue

    if loaded == 0:
        plt.close(fig)
        print("  [diag] no usable plotfiles -- IC-pressure diagnostic skipped")
        return

    # Reference overlays.
    ax.plot(R_curve / R0, p_ic,   '--', color='0.55', lw=1.5,
            label='IC: eq.(27) p_liq (no relax)')
    ax.plot(R_curve / R0, p_pred, '-',  color='black', lw=2.0,
            label='Predicted post-relax p* (theory)')
    ax.axhline(p_b,   color='tab:red',   ls=':', lw=1.0,
               label=f"p_b   = {p_b:.0f} Pa")
    ax.axhline(p_inf, color='tab:blue',  ls=':', lw=1.0,
               label=f"p_inf = {p_inf:.0f} Pa")
    ax.axvline(1.0,                       color='0.3', ls=':', lw=0.8)
    ax.axvline(1.0 + DIAG_EPSILON / R0,   color='0.3', ls=':', lw=0.8)

    ax.set_xlabel(r"$R / R_0$",       fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$p_\mathrm{mix}$ [Pa]", fontsize=FONT_SIZE_LABEL)
    ax.set_title("Sch20 IC-pressure diagnostic (first {} frames)".format(loaded),
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.set_xlim(0.5, DIAG_R_MAX_FACTOR)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    plt.tight_layout()

    out_png = os.path.join(IMG_DIR, f"{DIAG_SAVE_NAME}.png")
    out_eps = os.path.join(IMG_DIR, f"{DIAG_SAVE_NAME}.eps")
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(out_eps, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diag] wrote {out_png}")
    print(f"  [diag] wrote {out_eps}")

    # Console summary: predicted p* at R0+epsilon vs p_b.
    p_star = float(_predicted_relaxed_pressure(
        np.array([R0 + DIAG_EPSILON]), R0, p_inf, p_b, DIAG_EPSILON))
    print(f"  [diag] predicted p* at R0+epsilon = {p_star:.1f} Pa "
          f"(p_b = {p_b:.0f} Pa, ledge = {p_star - p_b:+.1f} Pa)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    tau_c = rayleigh_collapse_time(P)
    # Time-axis: convert simulation time (seconds) -> plot units.
    if TIME_UNIT == 'ms':
        t_factor   = 1.0e3
        t_label    = r"$t$ [ms]"
    elif TIME_UNIT == 'us':
        t_factor   = 1.0e6
        t_label    = r"$t$ [$\mu$s]"
    else:  # 'nondim'
        t_factor   = 1.0 / tau_c if (np.isfinite(tau_c) and tau_c > 0) else 1.0
        t_label    = XLABEL_STR  # r"$t / \tau_c$"
    r_scale = P.R0  if NONDIM_RADIUS else 1.0

    print("=" * 70)
    print("SCH20 OSCILLATING-BUBBLE ANALYSIS  (Case 1, p_inf/p_b = 10)")
    print("=" * 70)
    print(f"  rho_l       = {P.rho_l}")
    print(f"  p_inf       = {P.p_inf}        ({P.p_inf/P.p_g0:.1f} x p_b)")
    print(f"  p_b (p_g0)  = {P.p_g0}")
    print(f"  c_l         = {P.c_l}")
    print(f"  sigma       = {P.sigma}")
    print(f"  R0          = {P.R0}")
    print(f"  r_inf       = {P.r_inf}  (Chen-2D log far-field cutoff)")
    print(f"  T_END       = {T_END*1e3:.3f} ms")
    print(f"  tau_c       = {tau_c*1e3:.4f} ms  (Rayleigh collapse time)")
    print(f"  output dir  = {OUTPUT_DIR}")
    print()

    # ---- Linearized natural frequencies -----------------------------------
    omega0_3D     = linear_natural_frequency_3d(P)
    omega0_chen2d = linear_natural_frequency_chen2d(P)
    T0_3D     = (2 * np.pi / omega0_3D)     if omega0_3D     > 0 else float('nan')
    T0_chen2d = (2 * np.pi / omega0_chen2d) if omega0_chen2d > 0 else float('nan')
    alpha_rad_3D = alpha_KM_radiation(P.R0, omega0_3D, P.c_l)

    print("  Linearized natural-frequency predictions:")
    print(f"    3D RP / KM Minnaert : omega_0 = {omega0_3D:10.2f} rad/s  "
          f"T_0 = {T0_3D*1e3:.4f} ms")
    print(f"    Chen 2D Minnaert    : omega_0 = {omega0_chen2d:10.2f} rad/s  "
          f"T_0 = {T0_chen2d*1e3:.4f} ms")
    if alpha_rad_3D > 0:
        print(f"    KM radiation damping (3D): alpha = {alpha_rad_3D:.3f} 1/s   "
              f"(t_decay = 1/alpha = {1/alpha_rad_3D*1e3:.3f} ms)")
    print()

    # ---- Analytical references --------------------------------------------
    curves = []   # list of dicts: key, label, color, lw, ls, t, R
    if SHOW_KM_3D:
        print("  Solving Keller-Miksis ODE ...")
        t_km, R_km, _ = solve(P, T_END, model="km")
        curves.append(dict(key="km", label=LABEL_KM, color=COLOR_KM,
                            lw=LINE_WIDTH_KM, ls='-', t=t_km, R=R_km))
    if SHOW_RPE_3D:
        print("  Solving 3D Rayleigh-Plesset ODE ...")
        t_rp, R_rp, _ = solve(P, T_END, model="rp")
        curves.append(dict(key="rp", label=LABEL_RPE, color=COLOR_RPE,
                            lw=LINE_WIDTH_RPE, ls='--', t=t_rp, R=R_rp))
    if SHOW_CHEN_2D:
        print("  Solving Chen 2D cylindrical RPE ODE ...")
        t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")
        curves.append(dict(key="chen2d", label=LABEL_CHEN, color=COLOR_CHEN,
                            lw=LINE_WIDTH_CHEN, ls='-', t=t_c2d, R=R_c2d))

    print("\n  Analytical oscillation period (peak-to-peak) and decay rate:")
    for c in curves:
        T_c, n_c, _ = estimate_period(c['t'], c['R'])
        alpha_c, _, _ = estimate_decay_rate(c['t'], c['R'], P.R0)
        c['T'] = T_c
        c['alpha'] = alpha_c
        tstr = f"{T_c*1e3:.4f} ms" if np.isfinite(T_c) else "n/a (< 2 peaks)"
        astr = f"{alpha_c:9.2f} 1/s" if np.isfinite(alpha_c) else "n/a"
        print(f"    {c['label']:<26s}: T = {tstr:<18s} "
              f"alpha = {astr}  ({n_c} cycle gaps)")

    # KM - RP period split = compressibility-induced period correction.
    km = next((c for c in curves if c['key'] == 'km'), None)
    rp = next((c for c in curves if c['key'] == 'rp'), None)
    if km and rp and np.isfinite(km['T']) and np.isfinite(rp['T']) and rp['T'] > 0:
        print(f"    KM-vs-RP period shift: {(km['T']-rp['T'])/rp['T']*100:+.2f} %  "
              f"(0 -> compressibility irrelevant; nonzero -> KM correction real)")

    # ---- Simulation extraction --------------------------------------------
    sim_loaded = False
    n_skipped  = 0
    t_sim = np.array([])
    R_sim = np.array([])
    # Diffuse-band envelope curves (eta = 0.1 inner, eta = 0.9 outer).
    # Used by the second plot to shade the diffuse-band thickness.
    R_sim_eta01 = np.array([])
    R_sim_eta09 = np.array([])
    if SHOW_SIM:
        if os.path.isdir(OUTPUT_DIR):
            # Restrict the eta = 0.5 search to the near-bubble window x in
            # [0, 1.5*R0] -- keeps NSCBC ghost rings and far-field AMR c/f
            # wiggles from being mistaken for the interface in the _Large
            # NSCBC variant.
            t_sim, R_sim, n_skipped = extract_radius_history_robust(
                OUTPUT_DIR, eta_threshold=0.5, axis="x",
                #OUTPUT_DIR, eta_threshold=0.9, axis="x", # For Giggles tracks eta = 0.9 (MINIMUM MATCHES)
                x_max=1.5 * P.R0, verbose=True)
            sim_loaded = (len(t_sim) > 1)

            # Also extract the diffuse-band envelope (eta = 0.1 inside edge,
            # eta = 0.9 outside edge of the tanh transition).  Same plotfile
            # walk + skip logic, so the time arrays match t_sim element-wise.
            if sim_loaded:
                _, R_sim_eta01, _ = extract_radius_history_robust(
                    OUTPUT_DIR, eta_threshold=0.1, axis="x",
                    x_max=1.5 * P.R0, verbose=True)
                _, R_sim_eta09, _ = extract_radius_history_robust(
                    OUTPUT_DIR, eta_threshold=0.9, axis="x",
                    x_max=1.5 * P.R0, verbose=True)
            if n_skipped > 0:
                print(f"\n  [warn] skipped {n_skipped} corrupt / partial "
                      f"plotfile(s) in")
                print(f"         {OUTPUT_DIR}")
                print(f"         (loaded {len(t_sim)} usable frames.)")
        if not sim_loaded:
            if n_skipped == 0 and not os.path.isdir(OUTPUT_DIR):
                print(f"\n  [info] no AMReX output yet at {OUTPUT_DIR}")
                print(f"         analytical curves still plotted -- run sim first.")
            else:
                print(f"\n  [info] no usable plotfiles loaded from {OUTPUT_DIR}")
                print(f"         (skipped: {n_skipped}). Analytical curves only.")
    else:
        print("\n  [info] SHOW_SIM is False -- analytical curves only.")

    if sim_loaded:
        T_sim, n_sim, _ = estimate_period(t_sim, R_sim)
        alpha_sim, t_pk_sim, amp_sim = estimate_decay_rate(t_sim, R_sim, P.R0)
        print("\n  Simulation oscillation metrics:")
        tstr = f"{T_sim*1e3:.4f} ms" if np.isfinite(T_sim) else "n/a (< 2 peaks)"
        print(f"    sim T     = {tstr}   ({n_sim} cycle gaps detected)")
        if np.isfinite(alpha_sim):
            print(f"    sim alpha = {alpha_sim:.3f} 1/s")
        else:
            print(f"    sim alpha = n/a (< 2 maxima)")

        for c in curves:
            if np.isfinite(T_sim) and np.isfinite(c['T']) and c['T'] > 0:
                err = (T_sim - c['T']) / c['T'] * 100
                print(f"    period error vs {c['label']:<24s}: {err:+7.2f} %")
        if km and np.isfinite(alpha_sim) and np.isfinite(km['alpha']) \
                and km['alpha'] != 0:
            print(f"    decay (sim vs KM): alpha_sim/alpha_KM = "
                  f"{alpha_sim/km['alpha']:.3f}  "
                  f"(1.0 perfect; >1 numerical dissipation; <1 under-damped)")

    # ---- IC pressure diagnostic (first ~5 frames, p_mix(R) vs theory) ----
    if DIAG_IC_PRESSURE:
        print("\n  Running IC-pressure diagnostic...")
        plot_ic_pressure_diagnostic(OUTPUT_DIR, axis="x")

    # ---- Plot --------------------------------------------------------------
    plt.rcParams.update({
        'axes.titlesize':  FONT_SIZE_TITLE,
        'axes.labelsize':  FONT_SIZE_LABEL,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for c in curves:
        Tlbl = f", T={c['T']*1e3:.3f} ms" if np.isfinite(c['T']) else ""
        ax.plot(c['t'] * t_factor, c['R'] / r_scale,
                c['ls'], color=c['color'], lw=c['lw'],
                label=f"{c['label']}{Tlbl}")

    if sim_loaded:
        ax.plot(t_sim * t_factor, R_sim / r_scale, 'o', color=COLOR_SIM,
                ms=MARKER_SIZE_SIM, lw=LINE_WIDTH_SIM, alpha=0.7,
                label=f"{LABEL_SIM} (n={len(t_sim)})")

    # (Equilibrium R0 reference line removed at user request -- will revisit
    # once the IC / model trajectory is sorted out.)
    ax.set_xlabel(t_label, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_STR if NONDIM_RADIUS else r"$R$ [m]",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_title(TITLE_STR, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc=LEGEND_LOC)

    plt.tight_layout()
    out_png = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    out_eps = os.path.join(IMG_DIR, f"{SAVE_NAME}.eps")
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(out_eps, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out_png}")
    print(f"  wrote {out_eps}")
    plt.close(fig)

    # ---- 2nd plot: same as above + shaded diffuse-band envelope -----------
    # Background polygon spans the radii where eta crosses 0.1 (inside edge,
    # gas-rich) and 0.9 (outside edge, liquid-rich), i.e. the FULL diffuse
    # interface thickness as the bubble evolves.  Lets us see whether the
    # eta = 0.5 collapse minimum (red dots) and the analytical R_min sit
    # inside the band -- if so, the discrepancy is dominated by interface
    # smearing, not by physics, and the cure is more AMR + smaller epsilon
    # rather than chasing a model change.
    if sim_loaded and len(R_sim_eta01) == len(t_sim) and len(R_sim_eta09) == len(t_sim):
        fig2, ax2 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

        # Same analytical curves as the main plot.
        for c in curves:
            Tlbl = f", T={c['T']*1e3:.3f} ms" if np.isfinite(c['T']) else ""
            ax2.plot(c['t'] * t_factor, c['R'] / r_scale,
                     c['ls'], color=c['color'], lw=c['lw'],
                     label=f"{c['label']}{Tlbl}")

        # Shaded diffuse band envelope.  fill_between handles NaN gracefully:
        # any frame where one of the eta = {0.1, 0.9} contours wasn't found
        # (e.g. fully sub-grid bubble at deep collapse) simply leaves a hole
        # in the polygon, which is itself diagnostic of resolution loss.
        with np.errstate(invalid='ignore'):
            ax2.fill_between(t_sim * t_factor,
                             R_sim_eta01 / r_scale,
                             R_sim_eta09 / r_scale,
                             where=np.isfinite(R_sim_eta01) & np.isfinite(R_sim_eta09),
                             color='tab:orange', alpha=0.22, linewidth=0,
                             label=r'Diffuse Interface $\eta \in [0.1, 0.9]$',
                             zorder=1)

        # eta=0.5 simulation marker on top.
        ax2.plot(t_sim * t_factor, R_sim / r_scale, 'o', color=COLOR_SIM,
                 ms=MARKER_SIZE_SIM, lw=LINE_WIDTH_SIM, alpha=0.7,
                 label=f"{LABEL_SIM} ($\\eta = 0.5$) (n={len(t_sim)})",
                 zorder=3)

        ax2.set_xlabel(t_label, fontsize=FONT_SIZE_LABEL)
        ax2.set_ylabel(YLABEL_STR if NONDIM_RADIUS else r"$R$ [m]",
                       fontsize=FONT_SIZE_LABEL)
        ax2.set_title(TITLE_STR + r"  $-$  with diffuse-band envelope",
                      fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax2.tick_params(labelsize=FONT_SIZE_TICK)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=FONT_SIZE_LEGEND, loc=LEGEND_LOC)

        plt.tight_layout()
        out_png2 = os.path.join(IMG_DIR, f"{SAVE_NAME}_EtaBand.png")
        out_eps2 = os.path.join(IMG_DIR, f"{SAVE_NAME}_EtaBand.eps")
        fig2.savefig(out_png2, dpi=DPI, bbox_inches='tight')
        fig2.savefig(out_eps2, dpi=DPI, bbox_inches='tight')
        print(f"  wrote {out_png2}")
        print(f"  wrote {out_eps2}")
        plt.close(fig2)
    elif sim_loaded:
        print(f"  [warn] diffuse-band envelope skipped -- "
              f"eta=0.1 ({len(R_sim_eta01)} pts) or eta=0.9 "
              f"({len(R_sim_eta09)} pts) array length doesn't match "
              f"R_sim ({len(t_sim)} pts).")

    # ---- CSV export: sim vs Chen2D, interpolated onto sim time grid --------
    # Columns:
    #   time_s          -- simulation plotfile time
    #   time_ms         -- same in ms (convenience)
    #   R_hydro2_m      -- extracted bubble radius from simulation
    #   R_chen2d_m      -- Chen-2D RPE solution interpolated to sim time
    #   diff_m          -- R_hydro2 - R_chen2d (signed; +ve = bigger than analytic)
    #   diff_rel_pct    -- 100 * diff / R_chen2d
    # Rows skipped where either column is NaN or chen2d outside its time span.
    if sim_loaded:
        chen = next((c for c in curves if c['key'] == 'chen2d'), None)
        if chen is not None and len(chen['t']) > 1:
            # np.interp clamps outside-range to endpoint values; mask those.
            R_chen_at_sim = np.interp(t_sim, chen['t'], chen['R'],
                                      left=np.nan, right=np.nan)
            diff_m   = R_sim - R_chen_at_sim
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_pct = 100.0 * diff_m / R_chen_at_sim

            out_csv = os.path.join(IMG_DIR, f"{SAVE_NAME}.csv")
            with open(out_csv, 'w') as f:
                f.write("time_s,time_ms,R_hydro2_m,R_chen2d_m,diff_m,diff_rel_pct\n")
                for ts, rh, rc, dm, dp in zip(t_sim, R_sim,
                                              R_chen_at_sim, diff_m, diff_pct):
                    f.write(f"{ts:.9e},{ts*1e3:.6f},{rh:.9e},"
                            f"{rc:.9e},{dm:.9e},{dp:.6f}\n")
            print(f"  wrote {out_csv}")
        else:
            print("  [info] no Chen2D curve -- CSV export skipped.")
    else:
        print("  [info] no simulation data loaded -- CSV export skipped.")


if __name__ == "__main__":
    main()
