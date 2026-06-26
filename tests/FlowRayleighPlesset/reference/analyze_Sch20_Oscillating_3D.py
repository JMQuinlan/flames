# -*- coding: utf-8 -*-
"""
===============================================================================
SCH20 OSCILLATING-BUBBLE ANALYSIS (3D, spherical)  --  Case 1, low pressure ratio
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
    python tests/FlowRayleighPlesset/reference/analyze_Sch20_Oscillating_3D.py

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
SHOW_KM_3D   = True      # 3D Keller-Miksis
SHOW_RPE_3D  = True      # 3D Rayleigh-Plesset
SHOW_CHEN_2D = False      # 2D Chen cylindrical (matches the 2D Cartesian sim geometry)
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
    r_inf     = 50*0.02 #20 * R0 or so :p #0.20,        # Chen-2D log far-field cutoff = half-domain
)

# ===== TIME / DOMAIN =====
# tau_c = 0.915 R0 sqrt(rho_l / p_inf) ~ 1.83e-3 s ; stop_time = 1.0e-2 s.
T_END = 1.0e-2

# Path to the AMReX plotfile output dir. Default is relative to the repo
# (run from <repo>/bin); the absolute OVERRIDE line below can be uncommented
# and edited for a different machine / scratch location.
OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowRayleighPlesset", "output_Sch20_Oscillating_3D",
))
# OVERRIDE (uncomment + edit for your machine):
OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_3D")
#OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_Large_3D")
OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_3D_WENO3_SHARP")
#OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_3D_WENO5")

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
TITLE_STR   = "Sch20 Oscillating Bubble (3D): R(t)"
XLABEL_STR  = r"$t / \tau_c$"
YLABEL_STR  = r"$R / R_0$"
LABEL_KM    = "Keller--Miksis (3D)"
LABEL_RPE   = "Rayleigh--Plesset (3D)"
LABEL_CHEN  = "Chen (2D cylindrical)"
LABEL_SIM   = "flames2 (3D)"
NONDIM_RADIUS = True     # plot R / R0  instead of R [m]
SAVE_NAME     = "Sch20_Oscillating_3D"

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
DIAG_SAVE_NAME     = "Sch20_IC_PressureDiagnostic_3D"

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


def _offset_ray_radial(ds, axis="x"):
    """Ray PARALLEL to `axis` but offset by half a finest cell off the other two
    axes, so it passes through cell CENTERS instead of the on-axis cell EDGES.
    The exact y=z=0 axis lies on 3D cell edges, where yt's ray sampling is
    ambiguous -> spurious eta=0.5 crossings (the ~0.64 R0 stray dots at collapse).
    Returns (radius, eta) sorted by radius, with radius = sqrt(s^2 + 2*off^2)
    (the true distance from the origin along the offset ray)."""
    L = float(ds.domain_right_edge[0])
    dx_base = float(ds.domain_width[0]) / int(ds.domain_dimensions[0])
    off = 0.5 * dx_base / (2 ** int(ds.max_level))     # half a finest cell
    if axis == "x":
        a, b, key = [0.0, off, off], [L, off, off], "x"
    elif axis == "y":
        a, b, key = [off, 0.0, off], [off, L, off], "y"
    else:
        a, b, key = [off, off, 0.0], [off, off, L], "z"
    ray = ds.ray(ds.arr(a, "code_length"), ds.arr(b, "code_length"))
    s = np.array(ray[key]); eta = np.array(ray["eta"])
    rad = np.sqrt(s * s + 2.0 * off * off)
    o = np.argsort(rad)
    return rad[o], eta[o]


def extract_radius_history_robust(amrex_output_dir, eta_threshold=0.5,
                                  axis="x", x_max=None):
    """Same as rayleigh_plesset_solver.extract_radius_history but tolerant
    of individual corrupt / partially-written plotfiles.

    Walks every '*cell' subdirectory of `amrex_output_dir`, attempts to load
    each one with yt, and silently SKIPS any that raise an exception (typical
    cause: a run interrupted while a plotfile was still being written, leaving
    an incomplete Cell_H / Header behind).

    Parameters
    ----------
    x_max : float or None
        If provided, restrict the eta=0.5 search to the interval [0, x_max]
        along the chosen axis.  Useful when the far-field has spurious
        eta excursions (NSCBC ghosts, AMR coarse-fine wiggles, plotfile
        averaging artifacts) that get mistaken for the bubble interface.
        With x_max=1.5*R0 the search stays in the near-bubble region.

    Returns (times, radii, n_skipped) with times/radii sorted by time. Never
    raises -- if the whole directory is bad, returns empty arrays.
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
    for pf in plot_files:
        try:
            ds = yt.load(pf)
            x, eta = _offset_ray_radial(ds, axis)   # off-axis ray -> radial coord (Option a)

            # Restrict the eta = 0.5 search window so spurious far-field
            # eta excursions (NSCBC ghost rings, AMR c/f wiggles, BC
            # extrapolation noise) can't get mis-picked as the bubble
            # interface.  With x_max ~ 1.5*R0 the bubble cleanly fits and
            # nothing past it is considered.
            if x_max is not None:
                mask = (x >= 0.0) & (x <= float(x_max))
                x = x[mask]
                eta = eta[mask]
                if len(x) == 0:
                    radii.append(np.nan)
                    times.append(float(ds.current_time))
                    continue

            idx = np.where(eta >= eta_threshold)[0]
            if len(idx) == 0:
                radii.append(np.nan)
            else:
                i = idx[0]
                if i == 0:
                    radii.append(x[0])
                else:
                    frac = (eta_threshold - eta[i - 1]) / (eta[i] - eta[i - 1])
                    radii.append(x[i - 1] + frac * (x[i] - x[i - 1]))
            times.append(float(ds.current_time))
        except Exception:
            # Bad / partial plotfile -- skip and keep going.
            n_skipped += 1
            continue

    if not times:
        return np.array([]), np.array([]), n_skipped

    times = np.array(times)
    radii = np.array(radii)
    order = np.argsort(times)
    return times[order], radii[order], n_skipped


# ============================================================================
# DIFFUSE ETA-BAND DIAGNOSTIC
# ============================================================================
# eta rises 0 -> 1 from gas (bubble center) to liquid (outside), so along the
# +x ray eta=0.1 is the INNER band edge, eta=0.9 the OUTER edge, eta=0.5 the
# nominal R(t).  The shaded region between the outer thresholds is the diffuse
# interface band; a band that fattens over time = numerical interface smearing
# (worth watching through the collapse, where it can over/under-shoot R_min).
BAND_THRESHOLDS = (0.1, 0.5, 0.9)


def extract_eta_band_history(amrex_output_dir, thresholds=BAND_THRESHOLDS,
                             axis="x", x_max=None):
    """Per frame, the radius where eta crosses each threshold (linear-interp).
    Returns (times, {thr: radii}); skips corrupt/partial plotfiles silently."""
    import yt
    yt.funcs.mylog.setLevel(40)
    empty = {t: np.array([]) for t in thresholds}
    if not os.path.isdir(amrex_output_dir):
        return np.array([]), empty
    pfs = sorted(os.path.join(amrex_output_dir, d) for d in os.listdir(amrex_output_dir)
                 if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell"))
    times, bands = [], {t: [] for t in thresholds}
    for pf in pfs:
        try:
            ds = yt.load(pf)
            r, eta = _offset_ray_radial(ds, axis)   # off-axis ray -> radial coord (Option a)
            if x_max is not None:
                m = (r >= 0.0) & (r <= float(x_max)); r, eta = r[m], eta[m]
            for thr in thresholds:
                if len(r) == 0:
                    bands[thr].append(np.nan); continue
                idx = np.where(eta >= thr)[0]
                if len(idx) == 0:
                    bands[thr].append(np.nan)
                elif idx[0] == 0:
                    bands[thr].append(r[0])
                else:
                    i = idx[0]
                    frac = (thr - eta[i - 1]) / (eta[i] - eta[i - 1])
                    bands[thr].append(r[i - 1] + frac * (r[i] - r[i - 1]))
            times.append(float(ds.current_time))
        except Exception:
            continue
    if not times:
        return np.array([]), empty
    times = np.array(times); o = np.argsort(times)
    return times[o], {t: np.array(bands[t])[o] for t in thresholds}


def plot_eta_band(times_s, bands, curves=None):
    """Shaded diffuse eta-band radius vs time -> Images/{SAVE_NAME}_eta_band.*"""
    if len(times_s) < 2:
        print("  [eta-band] <2 usable frames -- skipped")
        return
    thrs = sorted(bands)
    tu = globals().get("TIME_UNIT", "us")
    if tu == "ms":
        tf, tl = 1.0e3, r"$t$ [ms]"
    elif tu == "us":
        tf, tl = 1.0e6, r"$t$ [$\mu$s]"
    else:
        tau = rayleigh_collapse_time(P)
        tf = 1.0 / tau if (np.isfinite(tau) and tau > 0) else 1.0
        tl = r"$t / \tau_c$"
    R0 = P.R0
    lo, hi = thrs[0], thrs[-1]
    fw = globals().get("FIG_WIDTH", 11); fh = globals().get("FIG_HEIGHT", 6.5)
    fsl = globals().get("FONT_SIZE_LABEL", 14); fst = globals().get("FONT_SIZE_TITLE", 16)
    fsk = globals().get("FONT_SIZE_TICK", 11); fsg = globals().get("FONT_SIZE_LEGEND", 12)
    dpi = globals().get("DPI", 180)
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.fill_between(times_s * tf, bands[lo] / R0, bands[hi] / R0,
                    color="tab:red", alpha=0.20,
                    label=rf"diffuse band $\eta\in[{lo:.2f},{hi:.2f}]$")
    for thr in thrs:
        ls, lw = ("-", 2.2) if abs(thr - 0.5) < 1e-9 else ("--", 1.3)
        lab = rf"$\eta={thr:.2f}$" + ("  ($R/R_0$)" if abs(thr - 0.5) < 1e-9 else "")
        ax.plot(times_s * tf, bands[thr] / R0, ls, lw=lw, label=lab)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    # overlay analytical bubble-dynamics references (RPE / KM) for comparison
    for c in (curves or []):
        if c.get("key") in ("rp", "km"):
            ax.plot(np.asarray(c["t"]) * tf, np.asarray(c["R"]) / R0,
                    c.get("ls", "-"), color=c.get("color", "k"),
                    lw=c.get("lw", 1.6), alpha=0.9, label=c.get("label", c["key"]))
    ax.set_xlabel(tl, fontsize=fsl)
    ax.set_ylabel(r"$R / R_0$", fontsize=fsl)
    ax.set_title(f"{SAVE_NAME}: diffuse eta-band radius vs time",
                 fontsize=fst, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=fsk)
    ax.legend(fontsize=fsg, loc="best")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}_eta_band")
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".eps", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    bw = (bands[hi] - bands[lo]) / R0
    with np.errstate(invalid="ignore"):
        print(f"  [eta-band] wrote {out}.png  (band width R/R0: "
              f"start={bw[0]:.4f}, end={bw[-1]:.4f}, max={np.nanmax(bw):.4f})")


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



# ============================================================================
# VOLUME-BASED RADIUS  (Sch20 eq. 29)
# ============================================================================
# R = (3 V_b / 4pi)^(1/3),  V_b = sum_cells alpha_g * V_cell,  alpha_g = 1 - eta.
# This is Sch20's actual radius definition -- a robust volume integral over the
# whole gas phase, NOT a single-ray eta=0.5 crossing.  It is immune to the
# interface smearing / grid-anisotropy that makes the eta=0.5 ray under-read R
# at the violent collapse (see the eta-band plot).  AMR-correct: yt returns only
# leaf cells, so no double-counting of covered coarse cells.  Restricted to a
# +/-BOX_HALF box about the origin (the far field is eta=1 -> alpha_g=0, no
# contribution) so it stays fast on the 320^3 _Large grids.
BOX_HALF = 0.2   # m; ~10 R0 -- the bubble (R<=R0=0.02) lives well inside this


def extract_radius_volume_history(amrex_output_dir, box_half=BOX_HALF):
    """Per frame, R from the gas-volume integral (Sch20 eq. 29).
    Returns (times, radii); skips corrupt/partial plotfiles silently."""
    import yt
    yt.funcs.mylog.setLevel(40)
    if not os.path.isdir(amrex_output_dir):
        return np.array([]), np.array([])
    pfs = sorted(os.path.join(amrex_output_dir, d) for d in os.listdir(amrex_output_dir)
                 if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell"))
    times, radii = [], []
    for pf in pfs:
        try:
            ds = yt.load(pf)
            dle = ds.domain_left_edge
            dre = ds.domain_right_edge
            # Octant/symmetry detection: a lo edge sitting at the origin (the
            # bubble center) means that axis is a symmetry plane, so only half the
            # bubble lies along it.  full domain: all lo<0 -> factor 1; octant:
            # all lo=0 -> factor 8.  Scale V_gas by 2^(#symmetry axes).
            sym_factor = 1
            for _d in range(3):
                if float(dle[_d]) > -1e-6:
                    sym_factor *= 2
            if not times:
                print("  [R-volume] geometry: %s -> V_gas x%d"
                      % ("OCTANT" if sym_factor > 1 else "FULL domain", sym_factor))
            lo = [max(-box_half, float(dle[_d])) for _d in range(3)]
            hi = [min( box_half, float(dre[_d])) for _d in range(3)]
            reg = ds.box(ds.arr(lo, "code_length"), ds.arr(hi, "code_length"))
            eta = np.array(reg["eta"])
            try:
                vol = np.array(reg["index", "cell_volume"])
            except Exception:
                vol = np.array(reg["cell_volume"])
            alpha_g = np.clip(1.0 - eta, 0.0, 1.0)
            V_b = float(np.sum(alpha_g * vol)) * sym_factor
            radii.append((3.0 * V_b / (4.0 * np.pi)) ** (1.0 / 3.0))
            times.append(float(ds.current_time))
        except Exception:
            continue
    if not times:
        return np.array([]), np.array([])
    t = np.array(times); r = np.array(radii); o = np.argsort(t)
    return t[o], r[o]


def plot_radius_volume(times_s, R_vol, curves, R_eta_t=None, R_eta=None):
    """Volume-based R/R0 vs time, overlaid with KM/RP (and the eta=0.5 ray R for
    contrast) -> Images/{SAVE_NAME}_R_volume.*"""
    if len(times_s) < 2:
        print("  [R-volume] <2 usable frames -- skipped")
        return
    tu = globals().get("TIME_UNIT", "us")
    if tu == "ms":
        tf, tl = 1.0e3, r"$t$ [ms]"
    elif tu == "us":
        tf, tl = 1.0e6, r"$t$ [$\mu$s]"
    else:
        tau = rayleigh_collapse_time(P)
        tf = 1.0 / tau if (np.isfinite(tau) and tau > 0) else 1.0
        tl = r"$t / \tau_c$"
    R0 = P.R0
    fw = globals().get("FIG_WIDTH", 11); fh = globals().get("FIG_HEIGHT", 6.5)
    fsl = globals().get("FONT_SIZE_LABEL", 14); fst = globals().get("FONT_SIZE_TITLE", 16)
    fsk = globals().get("FONT_SIZE_TICK", 11); fsg = globals().get("FONT_SIZE_LEGEND", 12)
    dpi = globals().get("DPI", 180)
    fig, ax = plt.subplots(figsize=(fw, fh))
    # analytical references
    for c in (curves or []):
        if c.get("key") in ("rp", "km"):
            ax.plot(np.asarray(c["t"]) * tf, np.asarray(c["R"]) / R0, c.get("ls", "-"),
                    color=c.get("color", "k"), lw=c.get("lw", 1.6), alpha=0.9,
                    label=c.get("label", c["key"]))
    # eta=0.5 ray radius for contrast (the old measure), if available
    if R_eta is not None and R_eta_t is not None and len(R_eta) > 1:
        ax.plot(np.asarray(R_eta_t) * tf, np.asarray(R_eta) / R0, ":", color="tab:gray",
                lw=1.3, alpha=0.7, label=r"hydro2 $\eta{=}0.5$ ray")
    # the new volume-based radius
    ax.plot(times_s * tf, R_vol / R0, "-o", color="tab:purple", lw=2.0, ms=3,
            label=r"hydro2 $R=(3V_b/4\pi)^{1/3}$ (gas volume)")
    ax.set_xlabel(tl, fontsize=fsl)
    ax.set_ylabel(r"$R / R_0$", fontsize=fsl)
    ax.set_title(f"{SAVE_NAME}: bubble radius from GAS VOLUME (Sch20 eq. 29)",
                 fontsize=fst, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=fsk)
    ax.legend(fontsize=fsg, loc="best")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}_R_volume")
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".eps", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    imin = int(np.argmin(R_vol))
    print(f"  [R-volume] wrote {out}.png  (R_min/R0={R_vol[imin] / R0:.4f} at "
          f"t={times_s[imin] * tf:.4f}; sanity R(0)/R0={R_vol[0] / R0:.4f} -- expect ~1.0)")


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
    print("SCH20 OSCILLATING-BUBBLE ANALYSIS (3D, spherical)  (Case 1, p_inf/p_b = 10)")
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
    if SHOW_SIM:
        if os.path.isdir(OUTPUT_DIR):
            # Restrict the eta = 0.5 search to the near-bubble window x in
            # [0, 1.5*R0] -- keeps NSCBC ghost rings and far-field AMR c/f
            # wiggles from being mistaken for the interface in the _Large
            # NSCBC variant.
            t_sim, R_sim, n_skipped = extract_radius_history_robust(
                OUTPUT_DIR, eta_threshold=0.5, axis="x",
                x_max=1.5 * P.R0)
            sim_loaded = (len(t_sim) > 1)
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
    # ---- Diffuse eta-band (interface thickness) vs time ----------------------
    if sim_loaded:
        print("  Extracting diffuse eta-band (eta=0.1/0.5/0.9 radii)...")
        _bt, _bands = extract_eta_band_history(OUTPUT_DIR, axis="x", x_max=1.5 * P.R0)
        plot_eta_band(_bt, _bands, curves)
        print("  Extracting volume-based radius (Sch20 eq.29: R=(3 V_gas/4pi)^1/3)...")
        _vt, _vr = extract_radius_volume_history(OUTPUT_DIR)
        plot_radius_volume(_vt, _vr, curves, t_sim, R_sim)

    out_png = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    out_eps = os.path.join(IMG_DIR, f"{SAVE_NAME}.eps")
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(out_eps, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out_png}")
    print(f"  wrote {out_eps}")
    plt.close(fig)

    # ---- CSV export: sim vs 3D references (RP + KM), interp to sim grid ------
    # Columns:
    #   time_s          -- simulation plotfile time
    #   time_ms         -- same in ms (convenience)
    #   R_hydro2_m      -- extracted bubble radius from simulation
    #   R_rp_m          -- 3D Rayleigh-Plesset solution interpolated to sim time
    #   R_km_m          -- 3D Keller-Miksis  solution interpolated to sim time
    #   diff_km_m       -- R_hydro2 - R_km (signed; +ve = bigger than KM)
    #   diff_km_rel_pct -- 100 * diff_km / R_km
    # diff is taken vs KM (the compressible 3D benchmark) when present, else RP.
    # NaN is kept where a reference is outside its solved time span.
    if sim_loaded:
        rp = next((c for c in curves if c['key'] == 'rp'), None)
        km = next((c for c in curves if c['key'] == 'km'), None)
        have_rp = rp is not None and len(rp['t']) > 1
        have_km = km is not None and len(km['t']) > 1
        if have_rp or have_km:
            R_rp_at = (np.interp(t_sim, rp['t'], rp['R'], left=np.nan, right=np.nan)
                       if have_rp else np.full_like(t_sim, np.nan))
            R_km_at = (np.interp(t_sim, km['t'], km['R'], left=np.nan, right=np.nan)
                       if have_km else np.full_like(t_sim, np.nan))
            R_ref_at = R_km_at if have_km else R_rp_at   # diff baseline
            diff_m   = R_sim - R_ref_at
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_pct = 100.0 * diff_m / R_ref_at

            out_csv = os.path.join(IMG_DIR, f"{SAVE_NAME}.csv")
            with open(out_csv, 'w') as f:
                f.write("time_s,time_ms,R_hydro2_m,R_rp_m,R_km_m,"
                        "diff_km_m,diff_km_rel_pct\n")
                for ts, rh, rr, rk, dm, dp in zip(t_sim, R_sim,
                                                  R_rp_at, R_km_at, diff_m, diff_pct):
                    f.write(f"{ts:.9e},{ts*1e3:.6f},{rh:.9e},"
                            f"{rr:.9e},{rk:.9e},{dm:.9e},{dp:.6f}\n")
            print(f"  wrote {out_csv}")
        else:
            print("  [info] no RP/KM curve -- CSV export skipped.")
    else:
        print("  [info] no simulation data loaded -- CSV export skipped.")


if __name__ == "__main__":
    main()
