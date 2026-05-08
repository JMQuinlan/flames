# -*- coding: utf-8 -*-
"""
===============================================================================
KELLER-MIKSIS OSCILLATION ANALYSIS  --  mild driving, multi-cycle, focus on
period and amplitude-decay (acoustic-radiation damping) signatures.
===============================================================================

PURPOSE:
    Companion to analyze_KellerMiksis.py. The driven-collapse case targets
    R_min and tau_c; this script targets the cleanest KM-vs-RP signature for
    OSCILLATING bubbles:

        - oscillation period T from peak-to-peak time
        - amplitude decay rate alpha from successive peak amplitudes
            R_n - R_eq = R_0 * exp(-alpha * t_n)

    KM has acoustic-radiation damping that incompressible RP completely lacks,
    so the amplitude-decay rate is the headline differentiator. The classical
    leading-order linearized result (Brennen "Cavitation and Bubble Dynamics"
    Sec. 4.4):

        alpha_KM = omega_0 * R_eq / (2 * c_l)        (acoustic radiation, 3D)

    For 2D Cartesian sims, the geometric reference is Chen 2D cylindrical
    (logarithmic far-field); KM remains the compressibility-correction
    benchmark even though the 2D-vs-3D geometry is offset.

USAGE:
    python tests/FlowKellerMiskis/reference/analyze_KellerMiksis_Oscillation.py

OUTPUTS:
    Images/KellerMiksis_Oscillation.png   (R(t), |R'|/c_l, peak amplitudes)
    Console summary with measured period, decay rate, vs analytical KM/Chen2D.

REFERENCES:
    Keller-Miksis, JASA 1980
    Brennen, "Cavitation and Bubble Dynamics" (1995) Sec. 4.4 (linearized RP/KM)
    Schmidmayer-Bryngelson-Colonius JCP 402 (2020) Sec. 5.2  (2D Cartesian protocol)

===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Bring in the shared RPE/KM/Chen2D ODE solver
sys.path.insert(0, os.path.join(_HERE, '..', '..',
                                'FlowRayleighPlesset', 'OtherTests', 'reference'))
from rayleigh_plesset_solver import (                                  # noqa: E402
    Params, solve, extract_radius_history,
    linear_natural_frequency_3d, linear_natural_frequency_chen2d,
)


# ============================================================================
# PHYSICAL PARAMETERS  (must match input_KellerMiksis_MildOscillation)
# ============================================================================

P = Params(
    rho_l     = 10.0,
    p_inf     = 1500.0,      # MILD driving (3 x p_g0)
    mu_l      = 0.0,
    c_l       = 63.0,
    gamma_gas = 1.4,
    p_g0      = 500.0,
    p_v       = 0.0,
    sigma     = 7.28,
    R0        = 0.02,
    Rdot0     = 0.0,
    r_inf     = 0.10,
)

T_END = 12.0e-3      # 5 ms -- 2-3 oscillation cycles

OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowKellerMiksis", "output_KellerMiksis_MildOscillation",
))

IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# PLOT STYLING (publish-ready knobs)
# ============================================================================

FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11
LINE_WIDTH_PRIMARY = 2.5     # Chen 2D (geometric primary)
LINE_WIDTH_KM      = 2.5     # KM (compressibility primary)
LINE_WIDTH_CONTEXT = 1.0     # RP context only
MARKER_SIZE_SIM    = 4
DPI = 180

COLOR_KM     = 'tab:blue'
COLOR_CHEN2D = 'tab:green'
COLOR_RP     = '0.5'
COLOR_SIM    = 'tab:red'


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
    Returns alpha [1/s]; nan if fewer than 2 maxima are found.

    We use the maxima of R(t) (i.e. the upper envelope) so the perturbation
    R_max - R_eq is positive, allowing log() without sign flips. R_eq is the
    equilibrium radius (taken as R0 here for unforced oscillations).
    """
    tm, Rm = find_local_maxima(t, R)
    amp = Rm - R_eq
    if len(tm) < 2 or np.any(amp <= 0):
        return float('nan'), tm, amp
    # Linear fit log(amp) = log(A0) - alpha t
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
    cause: an Hydro2 run that was interrupted while a plotfile was still
    being written, leaving an incomplete Cell_H or Header file behind).

    Returns (times, radii) arrays sorted by time, plus a count of skipped
    plotfiles (so the caller can print a warning). Never raises -- if the
    whole directory is bad, returns empty arrays.
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
    print("=" * 70)
    print("KELLER-MIKSIS OSCILLATION ANALYSIS (MILD DRIVING)")
    print("=" * 70)
    print(f"  rho_l       = {P.rho_l}")
    print(f"  p_inf       = {P.p_inf}        ({P.p_inf/P.p_g0:.1f} x p_g0)")
    print(f"  p_g0        = {P.p_g0}")
    print(f"  c_l         = {P.c_l}")
    print(f"  sigma       = {P.sigma}")
    print(f"  R0          = {P.R0}")
    print(f"  T_END       = {T_END*1e3:.2f} ms")
    print(f"  output dir  = {OUTPUT_DIR}")
    print()

    # ---- Linearized natural frequencies -----------------------------------
    omega0_3D     = linear_natural_frequency_3d(P)
    omega0_chen2d = linear_natural_frequency_chen2d(P)
    T0_3D     = (2 * np.pi / omega0_3D)     if omega0_3D     > 0 else float('nan')
    T0_chen2d = (2 * np.pi / omega0_chen2d) if omega0_chen2d > 0 else float('nan')

    # KM-radiation damping rate (3D leading order; Brennen Sec. 4.4)
    alpha_rad_3D = alpha_KM_radiation(P.R0, omega0_3D, P.c_l)

    print("  Linearized natural-frequency predictions:")
    print(f"    3D RP / KM Minnaert : omega_0 = {omega0_3D:8.2f} rad/s  T_0 = {T0_3D*1e3:.4f} ms")
    print(f"    Chen 2D Minnaert    : omega_0 = {omega0_chen2d:8.2f} rad/s  T_0 = {T0_chen2d*1e3:.4f} ms")
    print(f"    KM radiation damping (3D): alpha = {alpha_rad_3D:.3f} 1/s   "
          f"(t_decay = 1/alpha = {1/alpha_rad_3D*1e3:.3f} ms)")
    print()

    # ---- Analytical references --------------------------------------------
    print("  Solving Keller-Miksis ODE ...")
    t_km,  R_km,  V_km  = solve(P, T_END, model="km")
    print("  Solving 3D Rayleigh-Plesset ODE ...")
    t_rp,  R_rp,  V_rp  = solve(P, T_END, model="rp")
    print("  Solving Chen 2D cylindrical RPE ODE ...")
    t_c2d, R_c2d, V_c2d = solve(P, T_END, model="chen2d")

    T_km,  n_km,  _ = estimate_period(t_km,  R_km)
    T_rp,  n_rp,  _ = estimate_period(t_rp,  R_rp)
    T_c2d, n_c2d, _ = estimate_period(t_c2d, R_c2d)

    print("\n  Analytical oscillation periods (peak-to-peak from ODE):")
    print(f"    KM     :  T = {T_km*1e3:.4f} ms  ({n_km} cycle gaps)")
    print(f"    RP     :  T = {T_rp*1e3:.4f} ms  ({n_rp} cycle gaps)")
    print(f"    Chen2D :  T = {T_c2d*1e3:.4f} ms  ({n_c2d} cycle gaps)")

    # KM - RP period split is the compressibility-induced period correction
    if not np.isnan(T_km) and not np.isnan(T_rp) and T_rp > 0:
        print(f"    KM-vs-RP period shift: {(T_km - T_rp)/T_rp*100:+.2f} %  "
              f"(0 -> compressibility doesn't matter; nonzero -> KM correction is real)")

    # KM amplitude-decay rate from envelope fit
    alpha_km_meas, _, _ = estimate_decay_rate(t_km, R_km, P.R0)
    alpha_rp_meas, _, _ = estimate_decay_rate(t_rp, R_rp, P.R0)
    print(f"\n  Analytical decay rate (envelope fit, R_max - R0):")
    print(f"    KM   alpha = {alpha_km_meas:8.2f} 1/s   "
          f"(theory leading-order alpha_rad = {alpha_rad_3D:.2f} 1/s)")
    print(f"    RP   alpha = {alpha_rp_meas:8.2f} 1/s   "
          f"(should be ~ 0; RP is undamped)")

    # ---- Simulation extraction --------------------------------------------
    # Use the local robust loader so a single corrupt plotfile (e.g. an
    # Hydro2 run interrupted while writing) doesn't kill the entire analysis.
    sim_loaded = False
    n_skipped  = 0
    if os.path.isdir(OUTPUT_DIR):
        t_sim, R_sim, n_skipped = extract_radius_history_robust(
            OUTPUT_DIR, eta_threshold=0.5, axis="x")
        sim_loaded = (len(t_sim) > 1)
        if n_skipped > 0:
            print(f"\n  [warn] skipped {n_skipped} corrupt / partial plotfile(s) in")
            print(f"         {OUTPUT_DIR}")
            print(f"         (typical cause: interrupted run leaving an incomplete")
            print(f"          Cell_H/Header. Loaded {len(t_sim)} usable frames.)")
    if not sim_loaded:
        t_sim = np.array([])
        R_sim = np.array([])
        if n_skipped == 0 and not os.path.isdir(OUTPUT_DIR):
            print(f"\n  [info] no AMReX output yet at {OUTPUT_DIR}")
            print(f"         analytical curves will still be plotted -- run sim first.")
        else:
            print(f"\n  [info] no usable plotfiles loaded from {OUTPUT_DIR}")
            print(f"         (skipped: {n_skipped}). Plotting analytical curves only.")

    if sim_loaded:
        V_sim = np.gradient(R_sim, t_sim)
        T_sim, n_sim, _ = estimate_period(t_sim, R_sim)
        alpha_sim, t_pk_sim, amp_sim = estimate_decay_rate(t_sim, R_sim, P.R0)

        print("\n  Simulation oscillation metrics:")
        print(f"    sim T     = {T_sim*1e3:.4f} ms   ({n_sim} cycle gaps detected)")
        print(f"    sim alpha = {alpha_sim:.3f} 1/s")

        # Errors vs Chen 2D (geometric primary for 2D Cartesian)
        if not np.isnan(T_sim) and not np.isnan(T_c2d):
            err_T_c2d = (T_sim - T_c2d) / T_c2d * 100
            print(f"    period error vs Chen 2D: {err_T_c2d:+.2f} %  "
                  f"(geometric primary for 2D Cartesian)")
        if not np.isnan(T_sim) and not np.isnan(T_km):
            err_T_km = (T_sim - T_km) / T_km * 100
            print(f"    period error vs KM       : {err_T_km:+.2f} %  "
                  f"(compressibility benchmark)")
        # Damping comparison
        if not np.isnan(alpha_sim) and not np.isnan(alpha_km_meas):
            print(f"    decay rate (sim vs KM)   : alpha_sim/alpha_KM = "
                  f"{alpha_sim/alpha_km_meas:.3f}  "
                  f"(1.0 perfect; > 1 = numerical dissipation; < 1 = under-damped)")
    else:
        V_sim = np.array([])
        T_sim = float('nan'); alpha_sim = float('nan')
        t_pk_sim = np.array([]); amp_sim = np.array([])

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))

    # (1) R(t) / R0  -- show the multi-cycle envelope
    ax = axes[0]
    ax.plot(t_c2d * 1e3, R_c2d / P.R0, '-', color=COLOR_CHEN2D,
            lw=LINE_WIDTH_PRIMARY,
            label=f'Chen 2D cylindrical (geometric primary, T = {T_c2d*1e3:.3f} ms)')
    ax.plot(t_km  * 1e3, R_km  / P.R0, '-', color=COLOR_KM,
            lw=LINE_WIDTH_KM,
            label=f'KM (compressibility primary, T = {T_km*1e3:.3f} ms)')
    ax.plot(t_rp  * 1e3, R_rp  / P.R0, '--', color=COLOR_RP,
            lw=LINE_WIDTH_CONTEXT,
            label=f'3D RP (context, T = {T_rp*1e3:.3f} ms)')
    if sim_loaded:
        ax.plot(t_sim * 1e3, R_sim / P.R0, 'o', color=COLOR_SIM,
                ms=MARKER_SIZE_SIM, alpha=0.7,
                label=f'AMReX sim (T = {T_sim*1e3:.3f} ms, n = {len(t_sim)})')
    ax.axhline(1.0, color='k', lw=0.5, ls=':', alpha=0.4, label='R0 (equilibrium)')
    ax.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('R / R0',    fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'KM mild oscillation: R(t)/R0   '
                 f'(p_inf/p_g0 = {P.p_inf/P.p_g0:.1f})',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best', ncol=1)

    # (2) Wall Mach -- should stay BELOW 0.1 throughout in the mild regime
    ax = axes[1]
    ax.plot(t_c2d * 1e3, np.abs(V_c2d) / P.c_l, '-', color=COLOR_CHEN2D,
            lw=LINE_WIDTH_PRIMARY, label='Chen 2D')
    ax.plot(t_km  * 1e3, np.abs(V_km)  / P.c_l, '-', color=COLOR_KM,
            lw=LINE_WIDTH_KM, label='KM')
    ax.plot(t_rp  * 1e3, np.abs(V_rp)  / P.c_l, '--', color=COLOR_RP,
            lw=LINE_WIDTH_CONTEXT, label='RP')
    if sim_loaded:
        ax.plot(t_sim * 1e3, np.abs(V_sim) / P.c_l, 'o', color=COLOR_SIM,
                ms=MARKER_SIZE_SIM, alpha=0.7, label='AMReX sim (FD)')
    ax.axhline(0.05, color='k', lw=0.4, ls=':', alpha=0.5, label='M = 0.05')
    ax.axhline(0.10, color='r', lw=0.4, ls=':', alpha=0.5, label='M = 0.10')
    ax.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("|R'| / c_l", fontsize=FONT_SIZE_LABEL)
    ax.set_title('Wall Mach number',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best')

    # (3) Amplitude decay (semilog y)  -- the headline KM-vs-RP signature
    ax = axes[2]
    # Analytical envelopes
    _, t_pk_km,  amp_km  = estimate_decay_rate(t_km,  R_km,  P.R0)
    _, t_pk_rp,  amp_rp  = estimate_decay_rate(t_rp,  R_rp,  P.R0)
    _, t_pk_c2d, amp_c2d = estimate_decay_rate(t_c2d, R_c2d, P.R0)
    if len(t_pk_km) and np.all(amp_km > 0):
        ax.semilogy(t_pk_km * 1e3, amp_km, 'o-', color=COLOR_KM, ms=8,
                    lw=LINE_WIDTH_KM,
                    label=f'KM peaks (alpha = {alpha_km_meas:.1f} 1/s)')
    if len(t_pk_c2d) and np.all(amp_c2d > 0):
        ax.semilogy(t_pk_c2d * 1e3, amp_c2d, 's-', color=COLOR_CHEN2D, ms=8,
                    lw=LINE_WIDTH_PRIMARY,
                    label='Chen 2D peaks')
    if len(t_pk_rp) and np.all(amp_rp > 0):
        ax.semilogy(t_pk_rp * 1e3, amp_rp, 'd--', color=COLOR_RP, ms=6,
                    lw=LINE_WIDTH_CONTEXT,
                    label=f'RP peaks (alpha = {alpha_rp_meas:.1f} 1/s, ~ 0)')
    if sim_loaded and len(t_pk_sim) and np.all(amp_sim > 0):
        ax.semilogy(t_pk_sim * 1e3, amp_sim, 'x', color=COLOR_SIM, ms=12, mew=2,
                    label=f'sim peaks (alpha = {alpha_sim:.1f} 1/s)')
    # Theoretical KM straight line
    if alpha_rad_3D > 0 and len(t_pk_km) > 0 and amp_km[0] > 0:
        t_grid = np.linspace(0, t_pk_km[-1], 100)
        env = amp_km[0] * np.exp(-alpha_rad_3D * (t_grid - t_pk_km[0]))
        ax.semilogy(t_grid * 1e3, env, ':', color='k', lw=1.0,
                    label=f'theory: A0 exp(-omega_0^2 R_eq/(2c) t),  '
                          f'alpha = {alpha_rad_3D:.1f} 1/s')
    ax.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('R_max - R0 (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Amplitude-decay envelope  (KM-vs-RP headline signature)',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, which='both')

    # Only draw the legend if at least one labeled artist was plotted.  With
    # short T_END (e.g. 3 ms vs ~ 1.3 ms period) the analytical envelopes can
    # have < 2 maxima and produce no envelope points.
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best')
    else:
        ax.text(0.5, 0.5,
                "Not enough oscillation cycles in T_END to fit a decay envelope.\n"
                "Increase T_END (>~ 4 ms) so >= 2 maxima are visible.",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=FONT_SIZE_LABEL, color='dimgray',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "KellerMiksis_Oscillation.png")
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
