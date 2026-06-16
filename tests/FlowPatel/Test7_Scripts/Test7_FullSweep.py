# -*- coding: utf-8 -*-
"""
===============================================================================
TEST 7 -- FULL Ma X We GRID (Ma = 1.2, 1.5, 2.9  X  We = 10, 100, 1000)
===============================================================================

PURPOSE:
    Compare droplet deformation across the full 3x3 Ma-We grid (9 cases).
    Two views of the same data are produced:

      (a) "Single overlay": one figure with 9 lines for each metric (R_x, R_y,
          AR), colour = Weber, linestyle = Mach. Useful as the headline
          comparison plot.
      (b) "Grouped panels": 3 columns (one per Ma) x 3 rows (R_x, R_y, AR),
          with the 3 Weber numbers shown as 3 lines per panel. Easier to
          read for trend analysis.

    Plus an eta = 0.5 contour overlay (multi-panel by time, all 9 cases per
    panel) showing the joint Ma-We effect on 2-D deformation.

    Per-simulation single-case plots (schlieren, pressure, velocity, full
    deformation metrics, etc.) are produced by Test4_WitVap.py -- run that
    once per simulation if you want them.

INPUT FILES (assumed; 9 total):
    1mm_ShockDroplet_Ma{Ma}_We{We}    for Ma in {1.2, 1.5, 2.9}, We in {10, 100, 1000}

NOTE on Ma = 2.9:
    The user spec lists Ma = 1.2, 1.5, 2.9 (which differs from the Ma = 2.0
    in Test 6). Edit SWEEP_MA / CASES below if 2.9 is a typo.

USAGE:
    Edit the CASES / paths / plot toggles at the top, then
        python Test7_FullSweep.py
===============================================================================
"""

import os
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt

# Make the shared helpers importable regardless of where the script is invoked
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, '..'))
from shock_droplet_sweep_common import (
    load_case, plot_radii_overlay, plot_radii_overlay_grouped,
    plot_metric_overlay, plot_metric_overlay_grouped,
    plot_contour_overlay,
)

# ============================================================================
# PATHS
# ============================================================================

amrex_root = r'/mmfs1/home/ttryon/flames/bin/tests/FlowPatel'
#amrex_root = r'../../../bin/tests/FlowPatel'
#amrex_root = r'/mmfs1/home/spatel6/flames/bin/tests/FlowPatel'

output_folder = './Test7_FullSweep_Analysis'

# ============================================================================
# SWEEP DEFINITION (3 x 3 = 9 cases)
# ============================================================================

SWEEP_MA = [1.2, 1.5, 2.9]      # NOTE: matches user spec; Test 6 uses 2.0
SWEEP_WE = [10, 100, 1000]

# Colour by Weber, linestyle by Mach
WE_COLORS = {10: 'tab:blue', 100: 'tab:orange', 1000: 'tab:green'}
MA_LINESTYLES = {1.2: '-', 1.5: '--', 2.9: ':'}

def _format_case(Ma, We):
    """File-naming convention: 1mm_ShockDroplet_Ma{Ma}_We{We}.

    Adjust here if your run-time naming uses a different format
    (e.g. 'Ma15_We100' without the dot).
    """
    return f'1mm_ShockDroplet_Ma{Ma}_We{We}'

CASES = []
for Ma, We in itertools.product(SWEEP_MA, SWEEP_WE):
    CASES.append({
        'label':     f'Ma = {Ma}, We = {We}',
        'Ma':        Ma,
        'We':        We,
        'subdir':    _format_case(Ma, We),
        'color':     WE_COLORS.get(We, 'k'),
        'linestyle': MA_LINESTYLES.get(Ma, '-'),
    })

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

TIME_STEP        = 4
ETA_THRESHOLD    = 0.5
RESOLUTION       = 512
EXTRACT_CONTOURS = True

# ============================================================================
# PLOT TOGGLES
# ============================================================================

PLOT_RADII_OVERLAY_SINGLE     = 1   # all 9 lines on one stacked R_x/R_y/AR figure
PLOT_RADII_OVERLAY_GROUPED    = 1   # 3-row x 3-col paneled radii (by Ma and by We)
PLOT_VAP_OVERLAY_SINGLE       = 1   # mdot, cum mass, A/A0 -- 3 separate figures, 9 lines each
PLOT_VAP_OVERLAY_GROUPED      = 1   # mdot, cum mass, A/A0 -- 3 separate paneled figures (per grouping)
PLOT_CONTOUR_OVERLAY          = 1   # eta=0.5 multi-panel contour comparison
NUM_CONTOUR_PANELS            = 6

# ============================================================================
# PLOT STYLING (publish-ready knobs)
# ============================================================================

FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 11
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
    print("TEST 7 -- full Ma x We grid")
    print(f"  amrex_root    : {amrex_root}")
    print(f"  output_folder : {output_folder}")
    print(f"  Ma values     : {SWEEP_MA}")
    print(f"  We values     : {SWEEP_WE}")
    print(f"  total cases   : {len(CASES)}")
    print("=" * 70)

    case_data = []
    for case in CASES:
        amrex_dir = os.path.join(amrex_root, case['subdir'])
        print(f"\nLoading {case['label']}  <=  {amrex_dir}")
        if not os.path.isdir(amrex_dir):
            print(f"  WARNING: missing directory, skipping {case['label']}")
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
        d['linestyle'] = case['linestyle']
        d['Ma']        = case['Ma']
        d['We']        = case['We']
        case_data.append(d)

    if not case_data:
        print("ERROR: no cases loaded; aborting.")
        return

    print("\n" + "=" * 70)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 70)

    if PLOT_RADII_OVERLAY_SINGLE:
        out = os.path.join(output_folder, '01_Radii_Overlay_FullSweep')
        png, vec = plot_radii_overlay(
            case_data, out,
            D0 = D_DROPLET_INITIAL,
            title_suffix = '  (Ma x We grid -- color = We, linestyle = Ma)',
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

    if PLOT_RADII_OVERLAY_GROUPED:
        out = os.path.join(output_folder, '02_Radii_Grouped_byMa_FullSweep')
        png, vec = plot_radii_overlay_grouped(
            case_data, group_key='Ma', line_key='We',
            output_path = out,
            D0 = D_DROPLET_INITIAL,
            title_suffix = '  (one column per Ma, lines = We)',
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
        )
        print(f"  Saved: {os.path.basename(png)}")
        print(f"  Saved: {os.path.basename(vec)}")

        # Also make the dual-grouped variant (one column per We, lines = Ma)
        out2 = os.path.join(output_folder, '03_Radii_Grouped_byWe_FullSweep')
        png2, vec2 = plot_radii_overlay_grouped(
            case_data, group_key='We', line_key='Ma',
            output_path = out2,
            D0 = D_DROPLET_INITIAL,
            title_suffix = '  (one column per We, lines = Ma)',
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
        )
        print(f"  Saved: {os.path.basename(png2)}")
        print(f"  Saved: {os.path.basename(vec2)}")

    common_overlay_kw = dict(
        title_suffix     = '  (Ma x We grid -- color = We, linestyle = Ma)',
        font_size_title  = FONT_SIZE_TITLE,
        font_size_label  = FONT_SIZE_LABEL,
        font_size_legend = FONT_SIZE_LEGEND,
        font_size_tick   = FONT_SIZE_TICK,
        line_width = LINE_WIDTH, dpi = DPI,
        save_format_vector = SAVE_FORMAT_VECTOR,
    )

    if PLOT_VAP_OVERLAY_SINGLE:
        # mdot
        png, vec = plot_metric_overlay(
            case_data, metric_key='mdot_total',
            output_path=os.path.join(output_folder, '04_Mdot_Overlay_FullSweep'),
            ylabel='mdot (kg / s)',
            title='Vaporization Mass-Transfer Rate',
            ref_line=0.0, ref_label=None, **common_overlay_kw,
        )
        print(f"  Saved: {os.path.basename(png)}, {os.path.basename(vec)}")

        # cumulative mass
        png, vec = plot_metric_overlay(
            case_data, metric_key='cum_mass',
            output_path=os.path.join(output_folder, '05_CumMass_Overlay_FullSweep'),
            ylabel='Cumulative mass transferred (kg)',
            title='Cumulative Vaporization Mass Transfer',
            ref_line=0.0, ref_label=None, **common_overlay_kw,
        )
        print(f"  Saved: {os.path.basename(png)}, {os.path.basename(vec)}")

        # A(t)/A0
        png, vec = plot_metric_overlay(
            case_data, metric_key='A_normalized',
            output_path=os.path.join(output_folder, '06_InterfacialArea_Overlay_FullSweep'),
            ylabel='A(t) / A0',
            title='Normalized Interfacial Area',
            ref_line=1.0, ref_label='A0 (initial)', **common_overlay_kw,
        )
        print(f"  Saved: {os.path.basename(png)}, {os.path.basename(vec)}")

    if PLOT_VAP_OVERLAY_GROUPED:
        common_grp_kw = dict(
            font_size_title  = FONT_SIZE_TITLE,
            font_size_label  = FONT_SIZE_LABEL,
            font_size_legend = FONT_SIZE_LEGEND,
            font_size_tick   = FONT_SIZE_TICK,
            line_width = LINE_WIDTH, dpi = DPI,
            save_format_vector = SAVE_FORMAT_VECTOR,
        )

        # Three metrics x two groupings (by Ma, by We) = 6 figures
        for metric_key, ylabel, title, refline in [
            ('mdot_total',   'mdot (kg / s)',
             'Vaporization Mass-Transfer Rate', 0.0),
            ('cum_mass',     'Cumulative mass transferred (kg)',
             'Cumulative Vaporization Mass Transfer', 0.0),
            ('A_normalized', 'A(t) / A0',
             'Normalized Interfacial Area', 1.0),
        ]:
            for group_key, line_key, idx_byMa, suffix in [
                ('Ma', 'We', True, '  (one panel per Ma, lines = We)'),
                ('We', 'Ma', False, '  (one panel per We, lines = Ma)'),
            ]:
                tag = 'byMa' if idx_byMa else 'byWe'
                base = {
                    'mdot_total':   '07_Mdot',
                    'cum_mass':     '08_CumMass',
                    'A_normalized': '09_InterfacialArea',
                }[metric_key]
                # Append _byMa / _byWe to the file name
                out = os.path.join(output_folder,
                                   f'{base}_Grouped_{tag}_FullSweep')
                png, vec = plot_metric_overlay_grouped(
                    case_data, metric_key=metric_key,
                    group_key=group_key, line_key=line_key,
                    output_path=out, ylabel=ylabel, title=title,
                    ref_line=refline,
                    title_suffix=suffix, **common_grp_kw,
                )
                print(f"  Saved: {os.path.basename(png)}, {os.path.basename(vec)}")

    if PLOT_CONTOUR_OVERLAY:
        out = os.path.join(output_folder, '10_Contour_Overlay_FullSweep')
        png, vec = plot_contour_overlay(
            case_data, out,
            x_min = X_MIN, x_max = X_MAX,
            y_min = Y_MIN, y_max = Y_MAX,
            num_panels = NUM_CONTOUR_PANELS,
            title_suffix = '  (Ma x We grid)',
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
