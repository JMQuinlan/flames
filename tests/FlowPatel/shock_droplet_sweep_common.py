# -*- coding: utf-8 -*-
"""
===============================================================================
SHOCK / DROPLET SWEEP -- SHARED PER-CASE LOADER + COMPARISON PLOTTERS
===============================================================================

PURPOSE:
    Lean helper module used by Test5_WeSweep, Test6_MaSweep, and Test7_FullSweep
    to (a) load the per-case time-series of droplet shape metrics from each
    AMReX output directory and (b) generate comparison overlays across the
    sweep.

    Per-case ANALYSIS (schlieren, pressure, velocity, vorticity, temperature,
    full deformation metrics, etc.) is owned by Test4_WitVap.py -- run that
    once per simulation if you want the standard single-case plots. The sweep
    scripts here focus on the new comparison plots that need data from ALL
    cases simultaneously:

        - overlaid Streamwise Radius R_x(t)
        - overlaid Crossflow  Radius R_y(t)
        - overlaid Aspect Ratio AR(t) = R_y / R_x
        - eta = 0.5 contour overlay (multi-panel by time, all cases per panel)

REQUIRED FIELDS:
    Each AMReX output directory must contain plot files with at least the
    'eta' field. The loader uses the same eta-band convention as Test4
    (eta < ETA_THRESHOLD identifies the droplet interior).

PUBLIC API:
    load_case(...)               - load one simulation's time series
    plot_radii_overlay(...)      - 3-axis comparison plot for R_x, R_y, AR
    plot_radii_overlay_grouped(...) - paneled version (Test 7)
    plot_contour_overlay(...)    - eta=0.5 multi-panel contour comparison

USAGE:
    See Test5_WeSweep.py / Test6_MaSweep.py / Test7_FullSweep.py.

===============================================================================
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import yt

yt.funcs.mylog.setLevel(40)


# ============================================================================
# HELPERS
# ============================================================================

def _extract_timestep_number(filename):
    m = re.search(r'(\d+)', os.path.basename(filename))
    return int(m.group(1)) if m else 0


def _list_plot_files(amrex_output_dir):
    """Return sorted list of valid AMReX plot directories (have a Header file)."""
    plot_files = []
    if not os.path.isdir(amrex_output_dir):
        raise FileNotFoundError(f"Not a directory: {amrex_output_dir}")
    for item in os.listdir(amrex_output_dir):
        item_path = os.path.join(amrex_output_dir, item)
        if not os.path.isdir(item_path):
            continue
        if '.old' in item or 'chk' in item.lower():
            continue
        if not os.path.exists(os.path.join(item_path, 'Header')):
            continue
        plot_files.append(item_path)
    plot_files.sort(key=_extract_timestep_number)
    if not plot_files:
        raise FileNotFoundError(f"No valid AMReX plot dirs in {amrex_output_dir}")
    return plot_files


def _extract_interface_contour(eta, x_grid, y_grid, threshold=0.5):
    """Pull the eta = threshold isoline coordinates from a 2D field."""
    fig_temp = plt.figure()
    cs = plt.contour(x_grid, y_grid, eta, levels=[threshold])
    plt.close(fig_temp)
    contours = []
    if len(cs.allsegs) > 0:
        for contour_path in cs.allsegs[0]:
            if len(contour_path) > 0:
                contours.append(np.asarray(contour_path))
    return contours


def _bbox_metrics(eta, x_grid, y_grid, threshold, droplet_center):
    """Return R_x, R_y, AR_yx, centroid from the eta-band bounding box.
    Same convention as Test4 deformation metrics; AR_yx = R_y / R_x.
    """
    droplet_mask = (eta < threshold)
    if not np.any(droplet_mask):
        return 0.0, 0.0, 1.0, droplet_center
    yi, xi = np.where(droplet_mask)
    x_coords = x_grid[yi, xi]
    y_coords = y_grid[yi, xi]
    R_x = 0.5 * (float(x_coords.max()) - float(x_coords.min()))
    R_y = 0.5 * (float(y_coords.max()) - float(y_coords.min()))
    AR_yx = (R_y / R_x) if R_x > 0 else 1.0
    centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))
    return R_x, R_y, AR_yx, centroid


# ============================================================================
# CASE LOADER
# ============================================================================

def load_case(amrex_output_dir,
              x_min, x_max, y_min, y_max,
              time_step=1, eta_threshold=0.5,
              resolution=512,
              droplet_center=(0.0, 0.0),
              extract_contours=True,
              verbose=True):
    """Load a single simulation directory and return per-time arrays.

    Returns a dict with keys:
        'times'      : (N,) array of physical times [s]
        'R_x'        : (N,) array of streamwise half-extent [m]
        'R_y'        : (N,) array of crossflow  half-extent [m]
        'AR_yx'      : (N,) array of R_y / R_x
        'centroids'  : (N, 2) array of (x_c, y_c) [m]
        'contours'   : list of length N; each entry is a list of [P, 2] arrays
                       holding the eta = eta_threshold contour segments [m]
        'amrex_dir'  : echo of input path (for diagnostics)
    """
    plot_files = _list_plot_files(amrex_output_dir)
    indices = list(range(0, len(plot_files), max(1, time_step)))

    times       = []
    R_x_list    = []
    R_y_list    = []
    AR_yx_list  = []
    centroids   = []
    contours_all = []

    domain_w = x_max - x_min
    domain_h = y_max - y_min
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    if verbose:
        print(f"  [{os.path.basename(amrex_output_dir)}] sampling "
              f"{len(indices)}/{len(plot_files)} frames "
              f"(time_step={time_step}, resolution={resolution})")

    for k, idx in enumerate(indices):
        ds = yt.load(plot_files[idx])
        t = float(ds.current_time)
        slc = ds.slice('z', 0.0)
        frb = slc.to_frb((domain_w, 'code_length'), resolution,
                         center=[cx, cy, 0.0],
                         height=(domain_h, 'code_length'))
        eta = np.array(frb['eta'])
        x_1d = np.linspace(x_min, x_max, resolution)
        y_1d = np.linspace(y_min, y_max, resolution)
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)

        Rx, Ry, ARyx, centroid = _bbox_metrics(eta, x_grid, y_grid,
                                               eta_threshold, droplet_center)

        times.append(t)
        R_x_list.append(Rx)
        R_y_list.append(Ry)
        AR_yx_list.append(ARyx)
        centroids.append(centroid)

        if extract_contours:
            contours_all.append(_extract_interface_contour(
                eta, x_grid, y_grid, eta_threshold))
        else:
            contours_all.append([])

        if verbose and ((k + 1) % 10 == 0 or k == len(indices) - 1):
            print(f"    frame {k + 1:4d}/{len(indices)}: t = {t:.4e} s   "
                  f"R_x={Rx*1e3:.3f} mm  R_y={Ry*1e3:.3f} mm  AR={ARyx:.3f}")

    return {
        'times':     np.array(times),
        'R_x':       np.array(R_x_list),
        'R_y':       np.array(R_y_list),
        'AR_yx':     np.array(AR_yx_list),
        'centroids': np.array(centroids),
        'contours':  contours_all,
        'amrex_dir': amrex_output_dir,
    }


# ============================================================================
# COMPARISON PLOTS
# ============================================================================

def plot_radii_overlay(case_data, output_path,
                       D0=None, title_suffix='',
                       font_size_title=16, font_size_label=14,
                       font_size_legend=12, font_size_tick=11,
                       line_width=2.0, dpi=300,
                       save_format_vector='eps'):
    """Three-axis vertical overlay: R_x(t), R_y(t), AR(t) for every case.

    `case_data` is a list of dicts produced by load_case(), each augmented with
    'label' and 'color' keys. `D0` is the initial droplet diameter [m]; if
    given, R0 = D0/2 is drawn as a dashed reference on the radius panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for cd in case_data:
        t_us = cd['times'] * 1e6
        kw = dict(color=cd['color'], linewidth=line_width,
                  linestyle=cd.get('linestyle', '-'),
                  label=cd['label'])
        axes[0].plot(t_us, cd['R_x']  * 1e3, **kw)
        axes[1].plot(t_us, cd['R_y']  * 1e3, **kw)
        axes[2].plot(t_us, cd['AR_yx'],     **kw)

    if D0 is not None and D0 > 0:
        R0_mm = 0.5 * D0 * 1e3
        for ax in axes[:2]:
            ax.axhline(R0_mm, color='k', linestyle=':', linewidth=1.0,
                       alpha=0.6, label=f'R0 = {R0_mm:.3f} mm')
    axes[2].axhline(1.0, color='k', linestyle=':', linewidth=1.0,
                    alpha=0.6, label='Circle (AR = 1)')

    axes[0].set_ylabel('Streamwise R_x (mm)', fontsize=font_size_label)
    axes[0].set_title(f'Streamwise Radius{title_suffix}',
                      fontsize=font_size_title, fontweight='bold')
    axes[1].set_ylabel('Crossflow R_y (mm)',  fontsize=font_size_label)
    axes[1].set_title(f'Crossflow Radius{title_suffix}',
                      fontsize=font_size_title, fontweight='bold')
    axes[2].set_ylabel('AR = R_y / R_x',      fontsize=font_size_label)
    axes[2].set_title(f'Aspect Ratio{title_suffix}',
                      fontsize=font_size_title, fontweight='bold')
    axes[2].set_xlabel('Time (us)',           fontsize=font_size_label)

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=font_size_legend, loc='best', ncol=1)
        ax.tick_params(labelsize=font_size_tick)

    plt.tight_layout()
    raster_path = output_path + '.png'
    vector_path = output_path + '.' + save_format_vector
    plt.savefig(raster_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(vector_path,             bbox_inches='tight')
    plt.close()
    return raster_path, vector_path


def plot_radii_overlay_grouped(case_data, group_key, line_key, output_path,
                               D0=None, title_suffix='',
                               font_size_title=16, font_size_label=14,
                               font_size_legend=11, font_size_tick=10,
                               line_width=2.0, dpi=300,
                               save_format_vector='eps'):
    """Paneled version for Test 7: one column per `group_key` value, lines
    within each panel coloured by `line_key`.

    case_data entries must carry `group_key` and `line_key` as dict fields.
    A 3-row grid (R_x, R_y, AR) by N-column grid (one per group value) is
    produced.
    """
    group_vals = sorted({cd[group_key] for cd in case_data})
    line_vals  = sorted({cd[line_key]  for cd in case_data})
    n_cols = len(group_vals)

    cmap = plt.get_cmap('viridis')
    line_colors = {v: cmap(i / max(1, len(line_vals) - 1))
                   for i, v in enumerate(line_vals)}

    fig, axes = plt.subplots(3, n_cols, figsize=(4.5 * n_cols, 12),
                             sharex=True, sharey='row',
                             squeeze=False)

    for col, gval in enumerate(group_vals):
        for cd in case_data:
            if cd[group_key] != gval:
                continue
            t_us = cd['times'] * 1e6
            color = line_colors[cd[line_key]]
            label = f'{line_key} = {cd[line_key]}'
            kw = dict(color=color, linewidth=line_width, label=label)
            axes[0, col].plot(t_us, cd['R_x']  * 1e3, **kw)
            axes[1, col].plot(t_us, cd['R_y']  * 1e3, **kw)
            axes[2, col].plot(t_us, cd['AR_yx'],     **kw)

        axes[0, col].set_title(f'{group_key} = {gval}',
                               fontsize=font_size_title, fontweight='bold')
        axes[2, col].set_xlabel('Time (us)', fontsize=font_size_label)

    if D0 is not None and D0 > 0:
        R0_mm = 0.5 * D0 * 1e3
        for col in range(n_cols):
            axes[0, col].axhline(R0_mm, color='k', linestyle=':', alpha=0.6)
            axes[1, col].axhline(R0_mm, color='k', linestyle=':', alpha=0.6)
    for col in range(n_cols):
        axes[2, col].axhline(1.0, color='k', linestyle=':', alpha=0.6)

    axes[0, 0].set_ylabel('Streamwise R_x (mm)', fontsize=font_size_label)
    axes[1, 0].set_ylabel('Crossflow R_y (mm)',  fontsize=font_size_label)
    axes[2, 0].set_ylabel('AR = R_y / R_x',      fontsize=font_size_label)

    for ax in axes.flat:
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=font_size_tick)

    # Single legend on the top-right panel
    axes[0, -1].legend(fontsize=font_size_legend, loc='best')

    fig.suptitle(f'Sweep Comparison{title_suffix}',
                 fontsize=font_size_title + 2, fontweight='bold', y=1.005)
    plt.tight_layout()
    raster_path = output_path + '.png'
    vector_path = output_path + '.' + save_format_vector
    plt.savefig(raster_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(vector_path,             bbox_inches='tight')
    plt.close()
    return raster_path, vector_path


def plot_contour_overlay(case_data, output_path,
                         x_min, x_max, y_min, y_max,
                         num_panels=6,
                         title_suffix='',
                         font_size_title=16, font_size_label=12,
                         font_size_legend=10, font_size_tick=9,
                         line_width=2.0, dpi=300,
                         save_format_vector='eps',
                         zoom_to_droplet=True, zoom_pad_mm=2.0):
    """Multi-panel comparison of the eta = 0.5 contour across cases.

    Each panel corresponds to one (evenly-spaced) timestep; all cases are
    overlaid in the panel using their assigned colours / linestyles.

    `case_data` is the same list-of-dicts as plot_radii_overlay. Each case
    must have 'contours' populated (extract_contours=True at load time).

    Cases may have different numbers of timesteps. We use the case with the
    fewest timesteps to set the panel count cap.
    """
    n_min = min(len(cd['times']) for cd in case_data)
    if n_min == 0:
        print("  WARNING: at least one case has zero timesteps; skipping")
        return None, None
    panel_indices = np.linspace(0, n_min - 1, min(num_panels, n_min),
                                dtype=int)
    n_panels = len(panel_indices)

    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4.5 * n_rows),
                             squeeze=False)

    # If zooming, build a window around the union of all centroids over time
    if zoom_to_droplet:
        all_cx = np.concatenate([cd['centroids'][:, 0] for cd in case_data])
        all_cy = np.concatenate([cd['centroids'][:, 1] for cd in case_data])
        cx_min, cx_max = float(all_cx.min()), float(all_cx.max())
        cy_min, cy_max = float(all_cy.min()), float(all_cy.max())
        pad = zoom_pad_mm * 1e-3
        win_x = (cx_min - pad, cx_max + pad)
        win_y = (cy_min - pad, cy_max + pad)
    else:
        win_x = (x_min, x_max)
        win_y = (y_min, y_max)

    for panel_idx, t_idx in enumerate(panel_indices):
        r, c = panel_idx // n_cols, panel_idx % n_cols
        ax = axes[r, c]
        # Use the first case's time as the panel timestamp
        t_panel_us = case_data[0]['times'][t_idx] * 1e6

        for cd in case_data:
            if t_idx >= len(cd['contours']):
                continue
            color = cd['color']
            ls    = cd.get('linestyle', '-')
            first = True
            for contour in cd['contours'][t_idx]:
                if len(contour) == 0:
                    continue
                ax.plot(contour[:, 0] * 1e3, contour[:, 1] * 1e3,
                        color=color, linestyle=ls, linewidth=line_width,
                        label=cd['label'] if (first and panel_idx == 0) else None)
                first = False

        ax.set_xlim(win_x[0] * 1e3, win_x[1] * 1e3)
        ax.set_ylim(win_y[0] * 1e3, win_y[1] * 1e3)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f't = {t_panel_us:.1f} us',
                     fontsize=font_size_label, fontweight='bold')
        ax.set_xlabel('X (mm)', fontsize=font_size_label - 1)
        ax.set_ylabel('Y (mm)', fontsize=font_size_label - 1)
        ax.tick_params(labelsize=font_size_tick)
        ax.grid(alpha=0.3)
        if panel_idx == 0:
            ax.legend(fontsize=font_size_legend, loc='best')

    # Hide unused panels
    for empty in range(n_panels, n_rows * n_cols):
        r, c = empty // n_cols, empty % n_cols
        axes[r, c].axis('off')

    fig.suptitle(f'Eta = 0.5 Contour Comparison{title_suffix}',
                 fontsize=font_size_title + 1, fontweight='bold', y=1.005)
    plt.tight_layout()

    raster_path = output_path + '.png'
    vector_path = output_path + '.' + save_format_vector
    plt.savefig(raster_path, dpi=dpi, bbox_inches='tight')
    plt.savefig(vector_path,             bbox_inches='tight')
    plt.close()
    return raster_path, vector_path
