# -*- coding: utf-8 -*-
"""
===============================================================================
KELLER-MIKSIS BENCHMARK ANALYSIS  --  compare Hydro2 simulation against the
1st-order-compressibility-corrected RP ODE.
===============================================================================

PURPOSE:
    Read the AMReX plot files from input_KellerMiksis_DrivenCollapse, extract
    R(t) and dR/dt(t) from the eta = 0.5 contour, and compare against three
    analytical references:

        - Keller-Miksis (3D)         <-- PRIMARY reference (heavy blue)
        - Chen 2D cylindrical RPE    <-- 2D-Cartesian-aware geometric match
        - 3D Rayleigh-Plesset        <-- incompressible context (gray dashed)

    KM is the headline curve because the simulation has wall Mach |R'|/c_l
    peaking around 0.20-0.25, where KM and incompressible RP differ by
    5-15 %. RP is plotted only as context to show what the compressibility
    correction is buying.

USAGE:
    python tests/FlowKellerMiskis/reference/analyze_KellerMiksis.py

OUTPUTS:
    Images/KellerMiksis_Validation.png    (R(t)/R0, |R'|/c_l, phase plot)
    Console summary with collapse-time, R_min, peak wall Mach error vs KM.

REFERENCES:
    Keller, Miksis, JASA 1980 -- the KM ODE itself
    Tiwari, Freund, Pantano JCP 252 (2013) 290-309 -- canonical KM protocol
    Schmidmayer, Bryngelson, Colonius JCP 402 (2020) 109080 -- 2D Cartesian match
    Brennen, "Cavitation and Bubble Dynamics" (1995) Ch. 2 -- KM derivation

===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Bring in the existing RPE/KM/Chen2D ODE solver from the FlowRayleighPlesset
# benchmark family. It already implements model="km", model="rp", model="chen2d".
sys.path.insert(0, os.path.join(_HERE, '..', '..',
                                'FlowRayleighPlesset', 'OtherTests', 'reference'))
from rayleigh_plesset_solver import (                                  # noqa: E402
    Params, solve, extract_radius_history, rayleigh_collapse_time,
)


# ============================================================================
# PHYSICAL PARAMETERS  (must match input_KellerMiksis_DrivenCollapse)
# ============================================================================

P = Params(
    rho_l     = 10.0,
    p_inf     = 10000.0,
    mu_l      = 0.0,
    c_l       = 63.0,        # Tammann sound speed for the scaled liquid
    gamma_gas = 1.4,
    p_g0      = 500.0,
    p_v       = 0.0,
    sigma     = 7.28,
    R0        = 0.02,
    Rdot0     = 0.0,
    r_inf     = 0.10,        # half-domain (Chen 2D far-field cutoff)
)

# Simulation window (must cover at least the first collapse + rebound start)
T_END = 3.0e-3      # 3 ms

# AMReX output directory -- match the input file's plot_file
# (relative to repo root when invoked as recommended)
OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowKellerMiksis", "output_KellerMiksis_DrivenCollapse",
))

# Image output directory
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# PLOT STYLING (publish-ready knobs at top of script)
# ============================================================================

FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11
LINE_WIDTH_PRIMARY = 2.5     # KM
LINE_WIDTH_CONTEXT = 1.0     # RP, Chen 2D
LINE_WIDTH_SIM     = 0.0     # 0 = markers only, set > 0 for line+markers
MARKER_SIZE_SIM    = 4
DPI = 180

COLOR_KM     = 'tab:blue'
COLOR_RP     = '0.5'
COLOR_CHEN2D = '0.3'
COLOR_SIM    = 'tab:red'


# ============================================================================
# HELPERS
# ============================================================================

def first_minimum(t, R):
    """Return (t_min, R_min) at the first local minimum of R(t).
    If no clear minimum is found (e.g. monotone decreasing run), return
    (t[argmin], R[argmin]).
    """
    if len(R) < 3:
        i = int(np.argmin(R))
        return t[i], R[i]
    # local min: R[i] < R[i-1] and R[i] < R[i+1]
    for i in range(1, len(R) - 1):
        if R[i] < R[i - 1] and R[i] < R[i + 1]:
            return t[i], R[i]
    i = int(np.argmin(R))
    return t[i], R[i]


def wall_mach(Rdot, c_l):
    return np.abs(Rdot) / c_l


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("KELLER-MIKSIS BENCHMARK ANALYSIS")
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

    # ---- Analytical references ---------------------------------------------
    tau_R = rayleigh_collapse_time(P)
    print(f"  Classical Rayleigh tau_R = {tau_R*1e3:.3f} ms (3D, p_g=0 limit)")

    print("\n  Solving Keller-Miksis ODE ...")
    t_km,  R_km,  V_km  = solve(P, T_END, model="km")
    print("  Solving 3D Rayleigh-Plesset ODE ...")
    t_rp,  R_rp,  V_rp  = solve(P, T_END, model="rp")
    print("  Solving Chen 2D cylindrical RPE ODE ...")
    t_c2d, R_c2d, V_c2d = solve(P, T_END, model="chen2d")

    # First-collapse statistics for each analytical model
    t_min_km,  R_min_km  = first_minimum(t_km,  R_km)
    t_min_rp,  R_min_rp  = first_minimum(t_rp,  R_rp)
    t_min_c2d, R_min_c2d = first_minimum(t_c2d, R_c2d)
    M_max_km  = wall_mach(V_km,  P.c_l).max()
    M_max_rp  = wall_mach(V_rp,  P.c_l).max()
    M_max_c2d = wall_mach(V_c2d, P.c_l).max()

    print("\n  Analytical first-collapse summary:")
    print(f"    KM   :  t_min = {t_min_km*1e3:.4f} ms,  R_min/R0 = {R_min_km/P.R0:.4f},  |M|_max = {M_max_km:.3f}")
    print(f"    RP   :  t_min = {t_min_rp*1e3:.4f} ms,  R_min/R0 = {R_min_rp/P.R0:.4f},  |M|_max = {M_max_rp:.3f}")
    print(f"    Chen2:  t_min = {t_min_c2d*1e3:.4f} ms, R_min/R0 = {R_min_c2d/P.R0:.4f}, |M|_max = {M_max_c2d:.3f}")
    print(f"    KM-vs-RP relative collapse-time difference: "
          f"{abs(t_min_km - t_min_rp) / t_min_rp * 100:.2f} %")
    print(f"    (large = compressibility matters; small = RP is fine)")

    # ---- Simulation extraction ---------------------------------------------
    sim_loaded = False
    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR,
                                                  eta_threshold=0.5, axis="x")
            sim_loaded = (len(t_sim) > 1)
        except Exception as e:
            print(f"\n  [warn] could not read AMReX output: {e}")
    if not sim_loaded:
        t_sim = np.array([])
        R_sim = np.array([])
        print(f"\n  [info] no AMReX output yet at {OUTPUT_DIR}")
        print(f"         analytical curves will still be plotted -- run the sim first.")

    # Discrete dR/dt for the simulation, for the wall-Mach plot
    if sim_loaded:
        V_sim = np.gradient(R_sim, t_sim)   # central-difference
    else:
        V_sim = np.array([])

    # ---- Error metrics: simulation vs KM -----------------------------------
    if sim_loaded:
        t_min_sim, R_min_sim = first_minimum(t_sim, R_sim)
        M_max_sim = wall_mach(V_sim, P.c_l).max() if len(V_sim) else float('nan')
        err_t = (t_min_sim - t_min_km) / t_min_km * 100
        err_R = (R_min_sim - R_min_km) / R_min_km * 100
        print("\n  Simulation vs Keller-Miksis (PRIMARY reference):")
        print(f"    sim  :  t_min = {t_min_sim*1e3:.4f} ms,  R_min/R0 = {R_min_sim/P.R0:.4f},  |M|_max ~ {M_max_sim:.3f}")
        print(f"    error: dt_min = {err_t:+.2f} %      dR_min = {err_R:+.2f} %")
        print()
        if abs(err_t) < 5.0 and abs(err_R) < 15.0:
            print("    -> within state-of-the-art band (2-5 % on tau_c, 5-15 % on R_min).")
        else:
            print("    -> outside SOTA band; expect to apply BC + resolution fixes per RPE_ValidationNotes.md")

    # ---- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # (1) R(t) / R0 vs time
    ax = axes[0]
    ax.plot(t_km  * 1e3, R_km  / P.R0, '-',  color=COLOR_KM,
            lw=LINE_WIDTH_PRIMARY, label='Keller-Miksis (3D, primary)')
    ax.plot(t_c2d * 1e3, R_c2d / P.R0, '-',  color=COLOR_CHEN2D,
            lw=LINE_WIDTH_CONTEXT,
            label=f'Chen 2D cylindrical (r_inf = {P.r_inf*1e3:.0f} mm)')
    ax.plot(t_rp  * 1e3, R_rp  / P.R0, '--', color=COLOR_RP,
            lw=LINE_WIDTH_CONTEXT, label='3D Rayleigh-Plesset (context)')
    ax.axvline(tau_R * 1e3, color='k', lw=0.5, ls=':', alpha=0.4,
               label=f'classical tau_R = {tau_R*1e3:.2f} ms')
    if sim_loaded:
        ax.plot(t_sim * 1e3, R_sim / P.R0, 'o',
                color=COLOR_SIM, ms=MARKER_SIZE_SIM, alpha=0.7,
                label=f'AMReX sim (n = {len(t_sim)})')
    ax.set_yscale('log')
    ax.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('R / R0',    fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'Keller-Miksis benchmark: R(t)/R0  '
                 f'(p_inf/p_g0 = {P.p_inf/P.p_g0:.0f})',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')

    # (2) Wall Mach |R'|/c_l vs time
    ax = axes[1]
    ax.plot(t_km  * 1e3, wall_mach(V_km,  P.c_l), '-',  color=COLOR_KM,
            lw=LINE_WIDTH_PRIMARY, label='KM (primary)')
    ax.plot(t_c2d * 1e3, wall_mach(V_c2d, P.c_l), '-',  color=COLOR_CHEN2D,
            lw=LINE_WIDTH_CONTEXT, label='Chen 2D (context)')
    ax.plot(t_rp  * 1e3, wall_mach(V_rp,  P.c_l), '--', color=COLOR_RP,
            lw=LINE_WIDTH_CONTEXT, label='RP (context)')
    if sim_loaded:
        ax.plot(t_sim * 1e3, wall_mach(V_sim, P.c_l), 'o',
                color=COLOR_SIM, ms=MARKER_SIZE_SIM, alpha=0.7,
                label='AMReX sim (finite-difference)')
    ax.axhline(0.05, color='k', lw=0.4, ls=':', alpha=0.5,
               label='M = 0.05 (RP/KM divergence threshold)')
    ax.axhline(0.10, color='r', lw=0.4, ls=':', alpha=0.5,
               label='M = 0.10 (RP loses validity)')
    ax.set_xlabel('time [ms]', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("|R'| / c_l", fontsize=FONT_SIZE_LABEL)
    ax.set_title('Wall Mach number  (compressibility diagnostic)',
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best', ncol=1)

    # (3) Phase plot: R/R0 vs R'/c_l
    ax = axes[2]
    ax.plot(R_km  / P.R0, V_km  / P.c_l, '-',  color=COLOR_KM,
            lw=LINE_WIDTH_PRIMARY, label='KM')
    ax.plot(R_c2d / P.R0, V_c2d / P.c_l, '-',  color=COLOR_CHEN2D,
            lw=LINE_WIDTH_CONTEXT, label='Chen 2D')
    ax.plot(R_rp  / P.R0, V_rp  / P.c_l, '--', color=COLOR_RP,
            lw=LINE_WIDTH_CONTEXT, label='RP')
    if sim_loaded:
        ax.plot(R_sim / P.R0, V_sim / P.c_l, 'o',
                color=COLOR_SIM, ms=MARKER_SIZE_SIM, alpha=0.7,
                label='AMReX sim')
    ax.axvline(1.0, color='k', lw=0.4, ls=':', alpha=0.5)
    ax.axhline(0.0, color='k', lw=0.4, ls=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('R / R0',         fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("R' / c_l (Mach)", fontsize=FONT_SIZE_LABEL)
    ax.set_title('Phase-space trajectory', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "KellerMiksis_Validation.png")
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
