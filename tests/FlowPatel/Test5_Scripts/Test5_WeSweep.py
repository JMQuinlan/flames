# -*- coding: utf-8 -*-
"""
===============================================================================
TEST 5 -- WEBER-NUMBER SWEEP AT FIXED MACH (Ma = 1.5)
===============================================================================

PURPOSE:
    Compare droplet deformation across three Weber-number cases at the same
    Mach number. Generates the three new Test4-style metrics
        - Streamwise Radius   R_x(t) = (max_x - min_x) / 2 of eta-band
        - Crossflow Radius    R_y(t) = (max_y - min_y) / 2 of eta-band
        - Aspect Ratio        AR(t)  = R_y / R_x
    overlaid on the same axes for all three Weber numbers, plus an eta = 0.5
    contour overlay (multi-panel by time, all We values per panel) to show
    how surface tension shapes the 2-D deformation history.

    Per-simulation single-case plots (schlieren, pressure, velocity, full
    deformation metrics, etc.) are produced by Test4_WitVap.py -- run that
    once per simulation if you want them. This script focuses on the cross-
    case comparison plots that need every case loaded simultaneously.

INPUT FILES (assumed):
    1mm_ShockDroplet_Ma1.5_We10
    1mm_ShockDroplet_Ma1.5_We100
    1mm_ShockDroplet_Ma1.5_We1000

USAGE:
    Edit the CASES / paths / plot toggles at the top, then
        python Test5_WeSweep.py
===============================================================================
"""

import os
import sys
import numpy as np

# Make the shared helpers importable regardless of where the script is invoked
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))
from shock_droplet_sweep_common import (
    load_case, plot_radii_overlay, plot_metric_overlay,
    plot_contour_overlay,
)

# ============================================================================
# PATHS
# ============================================================================

# Root directory containing all the AMReX run output folders.
# Subdirectories below should match the file-naming convention the user uses
# for the sweep:    1mm_ShockDroplet_Ma{Ma}_We{We}
amrex_root = r'/mmfs1/home/ttryon/flames/bin/tests/FlowPatel'
#amrex_root = r'../../../bin/tests/FlowPatel'
#amrex_root = r'/mmfs1/home/spatel6/flames/bin/tests/FlowPatel'

output_folder = './Test5_WeSweep_Analysis'

# ============================================================================
# SWEEP DEFINITION
# ============================================================================

FIXED_MA = 1.5
SWEEP_WE = [10, 100, 1000]

# Each case becomes one curve in the comparison plots
CASES = [
    {'label': 'We = 10',
     'Ma':    1.5, 'We':   10,
     'subdir': '1mm_ShockDroplet_Ma1.5_We10',
     'color':  'tab:blue',     'linestyle': '-'},
    {'label': 'We = 100',
     'Ma':    1.5, 'We':  100,
     'subdir': '1mm_ShockDroplet_Ma1.5_We100',
     'color':  'tab:orange',   'linestyle': '-'},
    {'label': 'We = 1000',
     'Ma':    1.5, 'We': 1000,
     'subdir': '1mm_ShockDroplet_Ma1.5_We1000',
     'color':  'tab:green',    'linestyle': '-'},
]

# ============================================================================
# DOMAIN & DROPLET (must match input files)
# ============================================================================

X_MIN, X_MAX = -0.005,  0.005
Y_MIN, Y_MAX = -0.005,  0.005
DROPLET_CENTER_X = 0.0
DROPLET_CENTER_Y = 0.0
D_DROPLET_INITIAL = 0.002       # m, = 2 * R0

# ============================================================================
# SAMPLING / EXTRACTION OPTIONS
# ============================================================================

TIME_STEP        = 4         # sample every Nth plotfile
ETA_THRESHOLD    = 0.5
RESOLUTION       = 512       # FRB resolution for the eta slab
EXTRACT_CONTOURS = True      # set False to skip the contour overlay (faster)

# ============================================================================
# PLOT TOGGLES
# ============================================================================

PLOT_RADII_OVERLAY      = 1   # overlaid R_x, R_y, AR vs t (one stacked figure)
PLOT_MDOT_OVERLAY       = 1   # mdot(t) vs time, all We overlaid (own figure)
PLOT_CUMMASS_OVERLAY    = 1   # cumulative mass vs time, all We overlaid (own figure)
PLOT_AREA_OVERLAY       = 1   # A(t)/A0 vs time, all We overlaid (own figure)
PLOT_CONTOUR_OVERLAY    = 1   # eta=0.5 multi-panel contour comparison
NUM_CONTOUR_PANELS      = 6   # number of evenly-spaced timestep panels

# ============================================================================
# PLOT STYLING (publish-ready knobs)
# ============================================================================

FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11
LINE_WIDTH       = 2.0
DPI              = 300
SAVE_FORMAT_VECTOR = 'eps'

CONTOUR_ZOOM_TO_DROPLET = True
CONTOUR_ZOOM_PAD_MM     = 2.0

# ============================================================================
# RUN
# ============================================================================

def main():
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 70)
    print(f"TEST 5 -- We sweep at Ma = {FIXED_MA}")
    print(f"  amrex_root    : {amrex_root}")
    print(f"  output_folder : {output_folder}")
    print(f"  cases         : {[c['label'] for c in CASES]}")
    print("=" * 70)

    case_data = []
    for case in CASES:
        amrex_dir = os.path.join(amrex_root, case['subdir'])
        print(f"\nLoading {case['label']}  <=  {amrex_dir}")
        if not os.path.isdir(amrex_dir):
            print(f"  WARNING: missing directory, skipping case {case['label']}")
            continue
        d = load_case(
            amrex_output_dir = amrex_dir,
            x_min = X_MIN, x_max = X_MAX,
            y_min = Y_MIN, y_max = Y_MAX,
            time_step = TIME_STEP,
            eta_threshold = ETA_THRESHOLD,
            resolution = RESOLUTION,
            droplet_center = (DROPLET_CENTER_X, DROPLET_CENTER_Y),
            extract_contours = (EXTRACT_CONTOURS and PLOT_CONTOUR_OVERLAY),
            verbose = True,
        )
        d['label']     = case['label']
        d['color']     = case['color']
        d['linestyle'] = case.get('linestyle', '-')
        d['Ma']        = case['Ma']
        d['We']        = case['We']
        case_data.append(d)

    if not case_data:
        print("ERROR: no cases loaded; aborting.")
        return

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    if PLOT_RADII_OVERLAY:
        out = os.path.join(output_folder, '01_Radii_Overlay_WeSweep')
        png, vec = plot_radii_overlay(
            case_data, out,
            D0 = D_DROPLET_INITIAL,
            title_suffix = f'  (Ma = {FIXED_MA}, We sweep)',
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

    common_kw = dict(
        title_suffix = f'  (Ma = {FIXED_MA}, We sweep)',
        font_size_title  = FONT_SIZE_TITLE,
        font_size_label  = FONT_SIZE_LABEL,
        font_size_legend = FONT_SIZE_LEGEND,
        font_size_tick   = FONT_SIZE_TICK,
        line_width = LINE_WIDTH, dpi = DPI,
        save_format_vector = SAVE_FORMAT_VECTOR,
    )

    if PLOT_MDOT_OVERLAY:
        out = os.path.join(output_folder, '02_Mdot_Overlay_WeSweep')
        png, vec = plot_metric_overlay(
            case_data, metric_key='mdot_total', output_path=out,
            ylabel='mdot (kg / s)',
            title='Vaporization Mass-Transfer Rate',
            ref_line=0.0, ref_label=None, **common_kw,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

    if PLOT_CUMMASS_OVERLAY:
        out = os.path.join(output_folder, '03_CumMass_Overlay_WeSweep')
        png, vec = plot_metric_overlay(
            case_data, metric_key='cum_mass', output_path=out,
            ylabel='Cumulative mass transferred (kg)',
            title='Cumulative Vaporization Mass Transfer',
            ref_line=0.0, ref_label=None, **common_kw,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

    if PLOT_AREA_OVERLAY:
        out = os.path.join(output_folder, '04_InterfacialArea_Overlay_WeSweep')
        png, vec = plot_metric_overlay(
            case_data, metric_key='A_normalized', output_path=out,
            ylabel='A(t) / A0',
            title='Normalized Interfacial Area',
            ref_line=1.0, ref_label='A0 (initial)', **common_kw,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

    if PLOT_CONTOUR_OVERLAY:
        out = os.path.join(output_folder, '05_Contour_Overlay_WeSweep')
        png, vec = plot_contour_overlay(
            case_data, out,
            x_min = X_MIN, x_max = X_MAX,
            y_min = Y_MIN, y_max = Y_MAX,
            num_panels = NUM_CONTOUR_PANELS,
            title_suffix = f'  (Ma = {FIXED_MA}, We sweep)',
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
            zoom_to_droplet = CONTOUR_ZOOM_TO_DROPLET,
            zoom_pad_mm     = CONTOUR_ZOOM_PAD_MM,
        )
        if png is not None:
            print(f"  Saved: {os.path.basename(png)}")
            print(f"  Saved: {os.path.basename(vec)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
