# -*- coding: utf-8 -*-
"""
===============================================================================
CHEN 2D MATCH ANALYSIS  --  validate Hydro2 against the Chen 2D cylindrical
RPE on the FIRST QUARTER cycle of a small-amplitude oscillation.
===============================================================================

PURPOSE:
    Quote one clean number for "sim matches Chen 2D within X %" without
    fighting BC reflections, KM compressibility damping, or 2D-vs-3D
    geometric ambiguity.

    Strategy (first-collapse-only protocol; Saurel-Petitpas-Berry 2009 §6.3,
    Schmidmayer 2020 §5.2):
        - Use small amplitude so wall Mach < 0.02 throughout
          --> KM ~ Chen 2D ~ RP geometrically; compressibility correction tiny
        - Validate ONLY the first peak (quarter cycle, t ~ T/4)
          --> first peak occurs BEFORE the BC reflection round-trip arrives
        - Report two numbers: (R_peak / R0) and t_peak, both vs Chen 2D ODE.

    These two numbers, with sub-5 % errors, constitute a clean publishable
    Chen-2D match for a 2D Cartesian compressible 5-eq code.

USAGE:
    python tests/FlowKellerMiksis/reference/analyze_KellerMiksis_Chen2DMatch.py

OUTPUTS:
    Images/KellerMiksis_Chen2DMatch.png   (first-peak overlay + zoomed metrics)
    Console PASS/FAIL on the two headline metrics.

REFERENCES:
    Chen, Israeli, Ravid JCP 1995  (cylindrical RPE with logarithmic far-field)
    Brennen, "Cavitation and Bubble Dynamics" (1995) Sec. 4.4 (linearized RP)
    Schmidmayer-Bryngelson-Colonius JCP 402 (2020) Sec. 5.2 (2D-Cartesian protocol)

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
    Params, solve, linear_natural_frequency_chen2d,
)


# ============================================================================
# PHYSICAL PARAMETERS  (must match input_KellerMiksis_Chen2DMatch)
# ============================================================================

P = Params(
    rho_l     = 10.0,
    p_inf     = 8000.0,           # elevated so first peak fits inside round-trip
    mu_l      = 0.0,
    c_l       = 96.4,             # Tammann c at p_inf=8000, pi=5000, rho=10
    gamma_gas = 1.4,
    p_g0      = 9500.0,           # ~14% over Young-Laplace value 8364
    p_v       = 0.0,
    sigma     = 7.28,
    R0        = 0.02,
    Rdot0     = 0.0,
    r_inf     = 0.10,
)

T_END = 4.0e-3      # 4 ms covers first peak (~3 ms) + small post-peak window

OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowKellerMiksis", "output_KellerMiksis_Chen2DMatch",
))

IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# PASS/FAIL THRESHOLDS  (publish-ready knobs)
# ============================================================================

# Schmidmayer 2020 quotes 2-5 % on period for 2D Cartesian protocol; we use
# the looser 5 % for "passing" and 2 % for "tight" so the script can report
# both bands.
PASS_T_PEAK_PCT     = 5.0      # acceptable error on first-peak time
PASS_R_PEAK_PCT     = 5.0      # acceptable error on first-peak radius
TIGHT_T_PEAK_PCT    = 2.0      # tight (state-of-the-art) bound on t_peak
TIGHT_R_PEAK_PCT    = 2.0      # tight bound on R_peak


# ============================================================================
# PLOT STYLING
# ============================================================================

FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11
LINE_WIDTH       = 2.5
MARKER_SIZE_SIM  = 5
DPI              = 180

COLOR_CHEN2D = 'tab:green'
COLOR_KM     = 'tab:blue'
COLOR_RP     = '0.5'
COLOR_SIM    = 'tab:red'


# ============================================================================
# HELPERS
# ============================================================================

def first_local_max(t, R):
    """Return (t_peak, R_peak) at the first interior local maximum of R(t),
    or (nan, nan) if no clear maximum is found within the array.
    """
    if len(R) < 3:
        return float('nan'), float('nan')
    for i in range(1, len(R) - 1):
        if R[i] > R[i - 1] and R[i] >= R[i + 1]:
            # parabolic refinement around the discrete peak (3-point fit)
            denom = (R[i - 1] - 2 * R[i] + R[i + 1])
            if abs(denom) > 1e-30:
                # fractional offset in samples (-0.5 .. +0.5)
                f = 0.5 * (R[i - 1] - R[i + 1]) / denom
                t_peak = t[i] + f * (t[i + 1] - t[i]) if 0 <= i + 1 < len(t) else t[i]
                R_peak = R[i] - 0.25 * (R[i - 1] - R[i + 1]) * f
                return float(t_peak), float(R_peak)
            return float(t[i]), float(R[i])
    # Fallback: just take argmax (likely the peak hasn't been reached fully)
    i = int(np.argmax(R))
    return float(t[i]), float(R[i])


def extract_radius_history_robust(amrex_output_dir, eta_threshold=0.5, axis="x"):
    """Robust per-plotfile extractor (skips corrupt frames). Mirrors the
    helper in analyze_KellerMiksis_Oscillation.py so this script is also
    self-contained.
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
            order = np.argsort(x); x, eta = x[order], eta[order]
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
            n_skipped += 1
            continue
    if not times:
        return np.array([]), np.array([]), n_skipped
    times = np.array(times); radii = np.array(radii)
    order = np.argsort(times)
    return times[order], radii[order], n_skipped


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("CHEN 2D MATCH ANALYSIS  --  first-peak protocol")
    print("=" * 70)
    print(f"  rho_l    = {P.rho_l}")
    print(f"  p_inf    = {P.p_inf}")
    print(f"  p_g0_ic  = {P.p_g0}    (Laplace eq. value would be "
          f"{P.p_inf + P.sigma/P.R0:.0f})")
    print(f"  c_l      = {P.c_l}")
    print(f"  sigma    = {P.sigma}")
    print(f"  R0       = {P.R0}")
    print(f"  T_END    = {T_END*1e3:.2f} ms")
    print(f"  domain   = +/- {P.r_inf} m   (acoustic round-trip ~ "
          f"{2*P.r_inf/P.c_l*1e3:.2f} ms)")
    print()

    # ---- Linearized natural period (the analytical headline) --------------
    omega0_chen2d = linear_natural_frequency_chen2d(P)
    T_chen2d_lin  = (2 * np.pi / omega0_chen2d) if omega0_chen2d > 0 else float('nan')
    print(f"  Chen 2D linearized Minnaert: omega_0 = {omega0_chen2d:.2f} rad/s")
    print(f"                               T_0     = {T_chen2d_lin*1e3:.4f} ms")
    print(f"                               T_0 / 4 = {T_chen2d_lin*1e3/4:.4f} ms  "
          f"(expected first-peak time)")
    print()

    # ---- Analytical references --------------------------------------------
    print("  Solving Chen 2D, KM, RP ODEs ...")
    t_c2d, R_c2d, V_c2d = solve(P, T_END, model="chen2d")
    t_km,  R_km,  V_km  = solve(P, T_END, model="km")
    t_rp,  R_rp,  V_rp  = solve(P, T_END, model="rp")

    t_pk_c2d, R_pk_c2d = first_local_max(t_c2d, R_c2d)
    t_pk_km,  R_pk_km  = first_local_max(t_km,  R_km)
    t_pk_rp,  R_pk_rp  = first_local_max(t_rp,  R_rp)

    print("\n  Analytical first-peak summary (FROM ODE):")
    print(f"    Chen 2D :  t_peak = {t_pk_c2d*1e3:7.4f} ms,  R_peak/R0 = {R_pk_c2d/P.R0:.4f}")
    print(f"    KM      :  t_peak = {t_pk_km *1e3:7.4f} ms,  R_peak/R0 = {R_pk_km /P.R0:.4f}")
    print(f"    RP      :  t_peak = {t_pk_rp *1e3:7.4f} ms,  R_peak/R0 = {R_pk_rp /P.R0:.4f}")

    M_max_c2d = float(np.abs(V_c2d).max() / P.c_l)
    print(f"    Chen 2D peak wall Mach = {M_max_c2d:.4f}  "
          f"({'OK -- KM correction negligible' if M_max_c2d < 0.02 else 'WARN -- compressibility may matter'})")

    # ---- Simulation extraction --------------------------------------------
    sim_loaded = False
    n_skipped = 0
    if os.path.isdir(OUTPUT_DIR):
        t_sim, R_sim, n_skipped = extract_radius_history_robust(
            OUTPUT_DIR, eta_threshold=0.5, axis="x")
        sim_loaded = (len(t_sim) > 1)
        if n_skipped > 0:
            print(f"\n  [warn] skipped {n_skipped} corrupt plotfile(s) "
                  f"({len(t_sim)} usable frames loaded)")
    if not sim_loaded:
        t_sim = np.array([])
        R_sim = np.array([])
        print(f"\n  [info] no usable AMReX output at {OUTPUT_DIR}")
        print("         analytical curves will still be plotted -- run sim first.")

    # ---- Headline metrics: first peak vs Chen 2D --------------------------
    if sim_loaded:
        t_pk_sim, R_pk_sim = first_local_max(t_sim, R_sim)
        err_t = (t_pk_sim - t_pk_c2d) / t_pk_c2d * 100 if t_pk_c2d > 0 else float('nan')
        err_R = (R_pk_sim - R_pk_c2d) / R_pk_c2d * 100 if R_pk_c2d > 0 else float('nan')

        print("\n  Simulation first peak (FROM eta = 0.5 contour):")
        print(f"    sim   :  t_peak = {t_pk_sim*1e3:7.4f} ms,  R_peak/R0 = {R_pk_sim/P.R0:.4f}")
        print()
        print("  HEADLINE: Simulation vs Chen 2D ODE on the first peak")
        print(f"    delta t_peak  = {err_t:+7.2f} %    (target < {PASS_T_PEAK_PCT:.0f} %, tight < {TIGHT_T_PEAK_PCT:.0f} %)")
        print(f"    delta R_peak  = {err_R:+7.2f} %    (target < {PASS_R_PEAK_PCT:.0f} %, tight < {TIGHT_R_PEAK_PCT:.0f} %)")

        verdict_t = ("TIGHT (SOTA)" if abs(err_t) < TIGHT_T_PEAK_PCT
                     else "PASS"     if abs(err_t) < PASS_T_PEAK_PCT
                     else "FAIL")
        verdict_R = ("TIGHT (SOTA)" if abs(err_R) < TIGHT_R_PEAK_PCT
                     else "PASS"     if abs(err_R) < PASS_R_PEAK_PCT
                     else "FAIL")
        print(f"    verdict t_peak = {verdict_t}")
        print(f"    verdict R_peak = {verdict_R}")
        if verdict_t in ("PASS", "TIGHT (SOTA)") and verdict_R in ("PASS", "TIGHT (SOTA)"):
            print()
            print("    ===> CLEAN CHEN 2D MATCH on the first quarter cycle.")
            print("         This is the validation point you can quote.")
    else:
        t_pk_sim = float('nan'); R_pk_sim = float('nan')
        err_t    = float('nan'); err_R    = float('nan')

    # ---- Plot --------------------------------------------------------------
    fig = plt.figure(figsize=(11, 11))
    gs  = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.0, 1.0], hspace=0.35)
    ax_main = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_mach = fig.add_subplot(gs[2])

    # Main R(t)
    ax_main.plot(t_c2d * 1e3, R_c2d / P.R0, '-', color=COLOR_CHEN2D,
                 lw=LINE_WIDTH,
                 label=f'Chen 2D ODE  (t_peak = {t_pk_c2d*1e3:.3f} ms,  R_peak/R0 = {R_pk_c2d/P.R0:.3f})')
    ax_main.plot(t_km  * 1e3, R_km  / P.R0, '-', color=COLOR_KM,
                 lw=1.0, alpha=0.8,
                 label=f'KM ODE (context, t_peak = {t_pk_km*1e3:.3f} ms)')
    ax_main.plot(t_rp  * 1e3, R_rp  / P.R0, '--', color=COLOR_RP, lw=1.0,
                 label=f'3D RP ODE (context, t_peak = {t_pk_rp*1e3:.3f} ms)')

    if sim_loaded:
        ax_main.plot(t_sim * 1e3, R_sim / P.R0, 'o', color=COLOR_SIM,
                     ms=MARKER_SIZE_SIM, alpha=0.8,
                     label=f'AMReX sim  (t_peak = {t_pk_sim*1e3:.3f} ms,  R_peak/R0 = {R_pk_sim/P.R0:.3f})')
        ax_main.plot([t_pk_sim*1e3], [R_pk_sim/P.R0], 'x', color=COLOR_SIM,
                     ms=14, mew=3)

    # Mark Chen 2D peak with crosshair
    ax_main.plot([t_pk_c2d*1e3], [R_pk_c2d/P.R0], '+', color=COLOR_CHEN2D,
                 ms=20, mew=3)
    ax_main.axhline(1.0, color='k', lw=0.5, ls=':', alpha=0.4, label='R0')
    # Mark BC round-trip arrival as a guide
    t_rt = 2 * P.r_inf / P.c_l * 1e3
    ax_main.axvline(t_rt, color='r', lw=0.5, ls=':', alpha=0.5,
                    label=f'BC round-trip arrives ~{t_rt:.2f} ms')

    ax_main.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax_main.set_ylabel('R / R0',    fontsize=FONT_SIZE_LABEL)
    ax_main.set_title(f'Chen 2D match: small-amplitude oscillation, '
                      f'first-peak validation',
                      fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_main.tick_params(labelsize=FONT_SIZE_TICK)
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best')

    # Zoom on the first peak (the validation window)
    if not np.isnan(t_pk_c2d):
        t_lo = max(0.0, t_pk_c2d - 0.001)
        t_hi = t_pk_c2d + 0.001
        m_c2d = (t_c2d >= t_lo) & (t_c2d <= t_hi)
        m_km  = (t_km  >= t_lo) & (t_km  <= t_hi)
        m_rp  = (t_rp  >= t_lo) & (t_rp  <= t_hi)
        ax_zoom.plot(t_c2d[m_c2d] * 1e3, R_c2d[m_c2d] / P.R0, '-',
                     color=COLOR_CHEN2D, lw=LINE_WIDTH, label='Chen 2D')
        ax_zoom.plot(t_km[m_km] * 1e3, R_km[m_km] / P.R0, '-',
                     color=COLOR_KM, lw=1.0, alpha=0.8, label='KM')
        ax_zoom.plot(t_rp[m_rp] * 1e3, R_rp[m_rp] / P.R0, '--',
                     color=COLOR_RP, lw=1.0, label='RP')
        if sim_loaded:
            m_sim = (t_sim >= t_lo) & (t_sim <= t_hi)
            ax_zoom.plot(t_sim[m_sim] * 1e3, R_sim[m_sim] / P.R0, 'o',
                         color=COLOR_SIM, ms=MARKER_SIZE_SIM + 2, alpha=0.8,
                         label='sim')
        ax_zoom.axvline(t_pk_c2d * 1e3, color=COLOR_CHEN2D, lw=0.6, ls=':')
        if sim_loaded and not np.isnan(t_pk_sim):
            ax_zoom.axvline(t_pk_sim * 1e3, color=COLOR_SIM, lw=0.6, ls=':')
        ax_zoom.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
        ax_zoom.set_ylabel('R / R0',    fontsize=FONT_SIZE_LABEL)
        ax_zoom.set_title(f'Zoom on first peak  (validation window: t in '
                          f'[{t_lo*1e3:.2f}, {t_hi*1e3:.2f}] ms)',
                          fontsize=FONT_SIZE_TITLE - 2)
        ax_zoom.tick_params(labelsize=FONT_SIZE_TICK)
        ax_zoom.grid(True, alpha=0.3)
        ax_zoom.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best')

    # Wall-Mach panel: confirms KM ~ Chen 2D ~ RP regime
    ax_mach.plot(t_c2d * 1e3, np.abs(V_c2d) / P.c_l, '-', color=COLOR_CHEN2D,
                 lw=LINE_WIDTH, label='Chen 2D')
    ax_mach.plot(t_km  * 1e3, np.abs(V_km)  / P.c_l, '-', color=COLOR_KM,
                 lw=1.0, alpha=0.8, label='KM')
    ax_mach.plot(t_rp  * 1e3, np.abs(V_rp)  / P.c_l, '--', color=COLOR_RP,
                 lw=1.0, label='RP')
    if sim_loaded and len(t_sim) > 1:
        V_sim = np.gradient(R_sim, t_sim)
        ax_mach.plot(t_sim * 1e3, np.abs(V_sim) / P.c_l, 'o', color=COLOR_SIM,
                     ms=MARKER_SIZE_SIM, alpha=0.7, label='sim (FD)')
    ax_mach.axhline(0.02, color='k', lw=0.4, ls=':', alpha=0.5,
                    label='M = 0.02 (KM correction starts)')
    ax_mach.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax_mach.set_ylabel("|R'| / c_l", fontsize=FONT_SIZE_LABEL)
    ax_mach.set_title('Wall Mach number  '
                      '(confirms small-amplitude regime: M << 0.02)',
                      fontsize=FONT_SIZE_TITLE - 2)
    ax_mach.tick_params(labelsize=FONT_SIZE_TICK)
    ax_mach.grid(True, alpha=0.3)
    ax_mach.legend(fontsize=FONT_SIZE_LEGEND - 1, loc='best')

    out = os.path.join(IMG_DIR, "KellerMiksis_Chen2DMatch.png")
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
