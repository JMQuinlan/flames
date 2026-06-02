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
SHOW_KM_3D   = True      # 3D Keller-Miksis
SHOW_RPE_3D  = True      # 3D Rayleigh-Plesset
SHOW_CHEN_2D = True      # 2D Chen cylindrical
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
    "FlowRayleighPlesset", "output_Sch20_Oscillating_minmod",
))
# OVERRIDE (uncomment + edit for your machine):
OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Oscillating_Large")

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
LABEL_SIM   = "flames2 (2D)"
NONDIM_TIME   = True     # plot t / tau_c instead of t [s]
NONDIM_RADIUS = True     # plot R / R0  instead of R [m]
SAVE_NAME     = "Sch20_Oscillating"

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


def extract_radius_history_robust(amrex_output_dir, eta_threshold=0.5, axis="x"):
    """Same as rayleigh_plesset_solver.extract_radius_history but tolerant
    of individual corrupt / partially-written plotfiles.

    Walks every '*cell' subdirectory of `amrex_output_dir`, attempts to load
    each one with yt, and silently SKIPS any that raise an exception (typical
    cause: a run interrupted while a plotfile was still being written, leaving
    an incomplete Cell_H / Header behind).

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
# MAIN
# ============================================================================

def main():
    tau_c = rayleigh_collapse_time(P)
    t_scale = tau_c if (NONDIM_TIME and np.isfinite(tau_c) and tau_c > 0) else 1.0
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
    if SHOW_SIM:
        if os.path.isdir(OUTPUT_DIR):
            t_sim, R_sim, n_skipped = extract_radius_history_robust(
                OUTPUT_DIR, eta_threshold=0.5, axis="x")
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
        ax.plot(c['t'] / t_scale, c['R'] / r_scale,
                c['ls'], color=c['color'], lw=c['lw'],
                label=f"{c['label']}{Tlbl}")

    if sim_loaded:
        ax.plot(t_sim / t_scale, R_sim / r_scale, 'o', color=COLOR_SIM,
                ms=MARKER_SIZE_SIM, lw=LINE_WIDTH_SIM, alpha=0.7,
                label=f"{LABEL_SIM} (n={len(t_sim)})")

    ax.axhline(1.0 if NONDIM_RADIUS else P.R0, color='k', lw=0.5, ls=':',
               alpha=0.4, label='equilibrium $R_0$')
    ax.set_xlabel(XLABEL_STR if NONDIM_TIME else r"$t$ [s]",
                  fontsize=FONT_SIZE_LABEL)
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


if __name__ == "__main__":
    main()
