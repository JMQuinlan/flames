"""
===============================================================================
SHOCK-DROPLET INTERACTION ANALYSIS -- PAPER-QUALITY VARIANT (4Paper)
===============================================================================

Derived from shock_droplet_analysis2.py.  Same toggleable/parameter-driven
structure, but tuned for journal-paper output.

WHAT'S DIFFERENT FROM v2:
    - Monotonic time-sequencing (re-sort by ds.current_time after load).
    - Khare/Saurel-style per-phase numerical schlieren (separate normalization
      for gas-side and liquid-side gradients) -- adjustable contrast / exponent
      / floor.  Removes the "everything is grey" problem.
    - Eta = {0.01, 0.5, 0.99} fine-weighted contour overlays on every field
      plot (interface envelope, the standard paper convention).
    - Deformation metrics emitted BOTH as the 2x3 grid (overview) AND as
      separate publication-ready figures.
    - NEW: shape-evolution composite (eta=0.5 outline overlaid at multiple
      times, color-coded -- Khare 2022 Fig 6 style).
    - NEW: Schlieren-Mach split plot (top = schlieren, bottom = Mach contour),
      mirroring the existing Schlieren-Pressure split structure.
    - NEW: mass-conservation diagnostic (integrated rho_eta0, rho_eta1, total
      vs time + drift %).
    - Publication-quality matplotlib defaults (serif fonts, tight layouts,
      vector EPS alongside PNG, consistent label sizes).

USAGE:
    python shock_droplet_analysis4Paper.py

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import re
from PIL import Image
from scipy import ndimage
from scipy.interpolate import interp1d

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# PLOT CONFIGURATION - SET TO 1 (ON) OR 0 (OFF)
# ============================================================================

# Main visualization plots
PLOT_SCHLIEREN_PRESSURE_SPLIT = 1      # Individual frames in Schlieren-Pressure/
PLOT_SCHLIEREN_PRESSURE_GRID = 0       # 1x6 horizontal grid
PLOT_SCHLIEREN_PRESSURE_GIF = 1        # GIF from Schlieren-Pressure/

PLOT_SCHLIEREN_VELOCITY_SPLIT = 1       # Individual frames in Schlieren-Velocity/
PLOT_SCHLIEREN_VELOCITY_GIF = 1         # GIF from Schlieren-Velocity/

PLOT_SCHLIEREN_VAPDOTRHO_SPLIT = 1      # Individual frames in Schlieren-VapDotRho/
PLOT_SCHLIEREN_VAPDOTRHO_GIF = 1        # GIF from Schlieren-VapDotRho/

PLOT_VELOCITY_VORTICITY_SPLIT = 1       # Individual frames in Velocity-Vorticity/
PLOT_VELOCITY_VORTICITY_GIF = 1         # GIF from Velocity-Vorticity/

PLOT_SCHLIEREN_TEMPERATURE_SPLIT = 1    # Individual frames in Schlieren-Temperature/
PLOT_SCHLIEREN_TEMPERATURE_GIF = 1      # GIF from Schlieren-Temperature/

# Quantitative analysis plots
PLOT_DEFORMATION_METRICS = 1           # D(t), AR(t), centroid, area, volume
PLOT_ENERGY_EVOLUTION = 1              # Max pressure, KE, surface energy vs time
PLOT_SHOCK_TRACKING = 1                # x-t diagram

# Flow field visualizations
PLOT_VELOCITY_FIELD = 1                # Individual frames in Velocity-Streamline/
PLOT_VELOCITY_FIELD_GIF = 1            # GIF from Velocity-Streamline/
PLOT_VORTICITY_FIELD = 1               # Individual frames in Vorticity/
PLOT_VORTICITY_FIELD_GIF = 1           # GIF from Vorticity/

# Profile and cross-section plots
PLOT_CENTERLINE_PROFILES = 1           # Pressure and density along y=0
PLOT_INTERFACE_PROFILES = 1            # Density across interface
PLOT_RADIAL_PROFILES = 1               # Pressure vs radius

# Dimensionless analysis
PLOT_DIMENSIONLESS_NUMBERS = 1         # We(t), Re(t), Ma(t)
PLOT_REGIME_MAP = 1                    # We-Oh diagram

# Advanced/optional plots
PLOT_PRESSURE_CONTOURS = 1             # Pressure contours at key times
PLOT_DENSITY_CONTOURS = 1              # Density contours at key times

# ---- 4Paper additions ------------------------------------------------------
PLOT_SHAPE_EVOLUTION = 1               # Composite eta=0.5 outlines (Khare Fig 6 style)
PLOT_SCHLIEREN_MACH_SPLIT = 1          # Individual frames in Schlieren-Mach/
PLOT_SCHLIEREN_MACH_GIF   = 1          # GIF from Schlieren-Mach/
PLOT_MASS_CONSERVATION    = 1          # rho_eta0, rho_eta1, total mass vs time
SEPARATE_DEFORMATION_FIGS = 1          # In addition to 2x3 grid, save each metric standalone
PLOT_AR_VS_VAP            = 1          # AR (left) vs integrated Vap_dot_rho (right) twin-axis
PLOT_AREA_VS_VAP          = 1          # Surface area (left) vs integrated Vap_dot_rho (right) twin-axis

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Time sampling
TIME_STEP = 4  # Sample every Nth timestep (1=all, 2=every other, 5=every 5th, etc.)

# File paths
amrex_output_dir = r'../../../bin/tests/FlowShockDroplet/output_ShockDroplet'
#amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_Shock1mmDroplet'
#amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_2mm_Droplet_1-0Ma'
#amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_Shock1mmDroplet_5Ma_H20'
#amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_Shock1mmDroplet_minmod_rk3'

#amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_Shock1mmDroplet_5Ma_n-Pentane'

output_folder = './ShockDroplet_Analysis'
#output_folder = './ShockDroplet_Analysis_5Ma_H2O'
#output_folder = './ShockDroplet_Analysis_5Ma_n-Pentane'

# Physical parameters
RHO_AIR = 1.225
RHO_WATER = 1000.0
RHO_SHOCK = 2.0
MU_AIR = 1.8e-5
MU_WATER = 1.0e-3
SIGMA = 72.8e-3
GAMMA_AIR = 1.4
GAMMA_WATER = 7.15
P0_AIR = 0.0
P0_WATER = 3.0e8
U_SHOCK = 500.0
D_DROPLET_INITIAL = 0.001
P_ATM = 1.01325e5

# Domain parameters
X_MIN = -0.005
X_MAX = 0.015
Y_MIN = -0.005
Y_MAX = 0.005
DROPLET_CENTER_X = 0.0
DROPLET_CENTER_Y = 0.0

# Time selection for grid plots
USE_ALL_TIMESTEPS = 1
KEY_TIMES = [0, 40e-6, 60e-6, 100e-6, 150e-6, 200e-6]
NUM_GRID_TIMES = 6

# Interface tracking (v2 defaults; overridden below to ETA_CONTOURS_PAPER).
ETA_CONTOURS = [0.01, 0.5, 0.99]
ETA_THRESHOLD = 0.5

# Schlieren parameters (legacy v2 globals -- kept for back-compat plot calls)
SCHLIEREN_BETA = 10.0
SCHLIEREN_LOG_SCALE = 1
SCHLIEREN_USE_MIXTURE = 1

# 4Paper schlieren -- Khare 2022 / Saurel 2009 / Sch20-style per-phase
# normalization.  Each phase gets its own dynamic range so that gas-side
# shocks (small absolute |grad rho|) and liquid-side waves (large absolute
# |grad rho|) BOTH render with crisp contrast.  k_air / k_liq are the
# Quirk 1996 contrast constants; alpha_exp is the gradient exponent
# (1.0 = linear, < 1.0 = boost weak features, > 1.0 = suppress weak features).
# grad_floor sets a relative threshold below which a region renders as
# pure white -- prevents low-amplitude numerical noise from filling the
# field with grey.
SCHLIEREN_PER_PHASE       = 1
# K controls Quirk 1996 contrast: schlieren = exp(-K * |grad rho|/max).
# Lower K -> wider tonal range, weak fronts still visible (less "saturated /
# smeared" look).  Was 60 -- saturated quickly and washed out into a single
# blob.  20 keeps the strong shock dark while the surrounding compression
# fan stays mid-grey.
SCHLIEREN_K_AIR           = 20.0
SCHLIEREN_K_LIQ           = 20.0
SCHLIEREN_ALPHA_EXP       = 0.8
# Floor below which the field renders as pure background.  Lowered from 1e-2
# to 1e-3 so weak (but real) compression waves don't get wiped out.
SCHLIEREN_GRAD_FLOOR_AIR  = 1.0e-3     # relative to per-phase max
SCHLIEREN_GRAD_FLOOR_LIQ  = 1.0e-3

# Eta contour overlays on field plots (interface envelope).
ETA_CONTOURS_PAPER = [0.01, 0.5, 0.99]
ETA_CONTOUR_LW_THIN = 0.5              # weight for the 0.01 / 0.99 envelope
ETA_CONTOUR_LW_MID  = 1.0              # weight for the 0.5 centerline
ETA_CONTOUR_COLOR   = 'black'

# Shape-evolution composite (Khare Fig 6 style).
# SHAPE_EVOLUTION_N_FRAMES = number of evenly spaced outlines drawn, picked
# from the FULL plot_files range (start -> end of simulation).  Loads eta
# independently of the main TIME_STEP / KEY_TIMES pipeline.
SHAPE_EVOLUTION_N_FRAMES = 20
SHAPE_EVOLUTION_CMAP     = 'viridis'
SHAPE_EVOLUTION_LW       = 1.5

# ----------------------------------------------------------------------------
# Override v2-era contour defaults with paper-quality values.  ETA_CONTOURS
# and CONTOUR_LINE_WIDTH are referenced by ~14 plot-function call sites
# inherited from v2; reassigning them globally here propagates to all of
# them without per-site edits.  Comment these lines to revert to v2 defaults.
# ----------------------------------------------------------------------------

# Deformation calculation
DEFORMATION_METHOD = 'bbox'

# Shock detection
SHOCK_DETECTION_METHOD = 'pressure'
SHOCK_GRADIENT_THRESHOLD = 1e4

# Vorticity filtering
VORTICITY_STD_MULTIPLIER = 2.0  # Use +/-2sigma for symmetric limits

# Velocity field options
STREAMLINE_SHOW = True   # Show streamlines on velocity plots
VECTOR_SHOW = True       # Show velocity vectors
STREAMLINE_DENSITY = 1.5 # Streamline density (higher = more lines)

# Visualization settings
DPI = 300
COLORMAP_SCHLIEREN = 'gray'
COLORMAP_PRESSURE = 'jet'
COLORMAP_DENSITY = 'viridis'
COLORMAP_VORTICITY = 'RdBu_r'
COLORMAP_VELOCITY = 'plasma'
COLORMAP_TEMPERATURE = 'hot'
COLORMAP_VAPDOTRHO = 'coolwarm'
FIGURE_SIZE_SINGLE = (12, 10)
FIGURE_SIZE_GRID_1X6 = (24, 5)  # Wide for 1x6 layout
FIGURE_SIZE_TIMESERIES = (10, 8)

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
FONT_SIZE_TIMESTAMP = 12
LINE_WIDTH_THICK = 2.5
LINE_WIDTH_NORMAL = 2.0
LINE_WIDTH_THIN = 1.5
CONTOUR_LINE_WIDTH = 0.5            # paper: fine-weighted contours (was 2.0 in v2)
PLOT_ZOOM_FACTOR = 2.0
ASPECT_RATIO = 'equal'

# Output settings
GIF_FPS = 10
GIF_LOOP = 0
GIF_OPTIMIZE = True
SAVE_FORMAT_RASTER = 'png'
SAVE_FORMAT_VECTOR = 'eps'

# ----------------------------------------------------------------------------
# Publication-quality matplotlib defaults.  Applied globally so every plot
# produced by this script inherits paper-friendly typography and bbox.
# ----------------------------------------------------------------------------
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['DejaVu Serif', 'Times', 'Times New Roman'],
    'mathtext.fontset':  'dejavuserif',
    'pdf.fonttype':      42,            # TrueType - editors can recolor text in vector
    'ps.fonttype':       42,
    'axes.linewidth':    1.0,
    'savefig.dpi':       DPI,
    'savefig.bbox':      'tight',
    'figure.dpi':        110,
})

# Create output directories
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ----------------------------------------------------------------------------
# Write run-metadata: source plotfile dir, run date/time, git commit + branch.
# Saved at output_folder/RUN_METADATA.txt before any plotting work so that
# even a crashed run leaves a traceable record of what was analyzed.
# ----------------------------------------------------------------------------
def _write_run_metadata(script_name):
    import datetime as _dt
    import subprocess as _sp
    import sys as _sys

    def _git(args, cwd=None):
        try:
            return _sp.check_output(
                ['git'] + args, cwd=cwd, stderr=_sp.DEVNULL
            ).decode().strip()
        except Exception:
            return 'unavailable'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    commit  = _git(['rev-parse', 'HEAD'],            cwd=script_dir)
    commit_s = _git(['rev-parse', '--short', 'HEAD'], cwd=script_dir)
    branch  = _git(['rev-parse', '--abbrev-ref', 'HEAD'], cwd=script_dir)
    dirty   = _git(['status', '--porcelain'],         cwd=script_dir)
    dirty_flag = 'YES' if dirty and dirty != 'unavailable' else 'no'

    meta_path = os.path.join(output_folder, 'RUN_METADATA.txt')
    with open(meta_path, 'w') as f:
        f.write("# Shock-droplet analysis run metadata (4Paper variant)\n")
        f.write(f"script              : {script_name}\n")
        f.write(f"run_started         : {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"amrex_output_dir    : {amrex_output_dir}\n")
        f.write(f"output_folder       : {output_folder}\n")
        f.write(f"git_commit          : {commit}\n")
        f.write(f"git_commit_short    : {commit_s}\n")
        f.write(f"git_branch          : {branch}\n")
        f.write(f"git_uncommitted     : {dirty_flag}\n")
        f.write(f"python_version      : {_sys.version.split()[0]}\n")
        f.write(f"python_executable   : {_sys.executable}\n")
        f.write(f"working_directory   : {os.getcwd()}\n")
        # Selected analysis knobs (back-trace plot-config without diffing scripts).
        f.write(f"TIME_STEP             : {TIME_STEP}\n")
        f.write(f"USE_ALL_TIMESTEPS     : {USE_ALL_TIMESTEPS}\n")
        f.write(f"SCHLIEREN_PER_PHASE   : {SCHLIEREN_PER_PHASE}\n")
        f.write(f"SCHLIEREN_K_AIR       : {SCHLIEREN_K_AIR}\n")
        f.write(f"SCHLIEREN_K_LIQ       : {SCHLIEREN_K_LIQ}\n")
        f.write(f"SCHLIEREN_ALPHA_EXP   : {SCHLIEREN_ALPHA_EXP}\n")
        f.write(f"ETA_CONTOURS_PAPER    : {ETA_CONTOURS_PAPER}\n")
    print(f"  wrote {meta_path}")

_write_run_metadata('shock_droplet_analysis4Paper.py')

subfolder_schlieren = os.path.join(output_folder, 'Schlieren-Pressure')
subfolder_velocity = os.path.join(output_folder, 'Velocity-Streamline')
subfolder_vorticity = os.path.join(output_folder, 'Vorticity')
subfolder_schlieren_velocity = os.path.join(output_folder, 'Schlieren-Velocity')
subfolder_schlieren_vapdotrho = os.path.join(output_folder, 'Schlieren-VapDotRho')
subfolder_velocity_vorticity = os.path.join(output_folder, 'Velocity-Vorticity')
subfolder_schlieren_temperature = os.path.join(output_folder, 'Schlieren-Temperature')
# 4Paper additions
subfolder_schlieren_mach        = os.path.join(output_folder, 'Schlieren-Mach')
subfolder_deformation_separate  = os.path.join(output_folder, 'Deformation-Separate')

for folder in [subfolder_schlieren,
               subfolder_velocity,
               subfolder_vorticity,
               subfolder_schlieren_velocity,
               subfolder_schlieren_vapdotrho,
               subfolder_velocity_vorticity,
               subfolder_schlieren_temperature,
               subfolder_schlieren_mach,
               subfolder_deformation_separate]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def find_closest_timesteps(times, target_times):
    """Find indices of timesteps closest to target_times -- MONOTONIC.

    Each target picks a UNIQUE plotfile, and the chosen indices are guaranteed
    to be in increasing time order.  Without the monotonic guarantee an earlier
    target could end up matched to a LATER plotfile than a later target picks
    (when the target spacing is finer than the plotfile spacing), making the
    frame sequence non-monotonic.  Targets are sorted internally; if there are
    more targets than available later plotfiles, the trailing targets are
    silently dropped.
    """
    times = np.asarray(times, dtype=float)
    targets = np.sort(np.asarray(target_times, dtype=float))
    indices = []
    last_idx = -1
    for target in targets:
        # Only consider plotfiles strictly later than the last pick.
        if last_idx + 1 >= len(times):
            break
        sub_times = times[last_idx + 1:]
        local_idx = int(np.argmin(np.abs(sub_times - target)))
        chosen = last_idx + 1 + local_idx
        indices.append(chosen)
        last_idx = chosen
    return indices

def select_evenly_spaced(indices, num_points):
    """Select evenly spaced indices from a list"""
    if len(indices) <= num_points:
        return indices
    return [indices[i] for i in np.linspace(0, len(indices)-1, num_points, dtype=int)]

def get_frame_number(filename):
    """Extract frame number from filename like '0042_Schlieren_Pressure.png'"""
    match = re.match(r'(\d+)_', filename)
    return int(match.group(1)) if match else 0

def create_gif_from_folder(subfolder_name, output_gif_name):
    """
    Create GIF from all images in a subfolder
    
    Parameters:
        subfolder_name: Name of subfolder (e.g., 'Schlieren-Pressure')
        output_gif_name: Name for output GIF (e.g., 'ANIM_Schlieren_Pressure.gif')
    """
    subfolder_path = os.path.join(output_folder, subfolder_name)
    
    # Get all PNG files
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
    
    if not image_files:
        print(f"  WARNING: No images found in {subfolder_name}/")
        return
    
    # Sort by frame number
    image_files = sorted(image_files, key=get_frame_number)
    
    # Load images
    frames = [Image.open(os.path.join(subfolder_path, f)) for f in image_files]
    
    # Save as GIF
    gif_path = os.path.join(output_folder, output_gif_name)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000/GIF_FPS,
        loop=GIF_LOOP,
        optimize=GIF_OPTIMIZE
    )
    
    print(f"  Created GIF: {output_gif_name} ({len(frames)} frames)")

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_schlieren(rho, dx, dy, beta=5.0, log_scale=False):
    """Calculate numerical schlieren field from density (legacy v2 form)."""
    drho_dx = np.gradient(rho, dx, axis=0)
    drho_dy = np.gradient(rho, dy, axis=1)
    grad_rho_mag = np.sqrt(drho_dx**2 + drho_dy**2)

    if log_scale:
        schlieren = np.log10(1 + 100*grad_rho_mag)
    else:
        grad_max = np.max(grad_rho_mag)
        if grad_max > 0:
            schlieren = np.exp(-beta * grad_rho_mag / grad_max)
        else:
            schlieren = np.ones_like(grad_rho_mag)

    return schlieren


def compute_schlieren_per_phase(rho, eta, dx, dy,
                                k_air=60.0, k_liq=60.0,
                                alpha_exp=0.8,
                                floor_air=1e-2, floor_liq=1e-2):
    """Khare 2022 / Saurel 2009 / Sch20-style numerical schlieren.

    Computes |grad rho| separately within the gas region (eta > 0.5) and the
    liquid region (eta < 0.5), normalizes each by its OWN max, then blends
    via eta.  Each phase gets its own contrast (k) and floor (so smooth
    regions render pure white).

    schlieren(x,y) = exp( -k * (|grad rho| / max_phase_grad)^alpha_exp )
                     in each phase, with a floor that maps small gradients
                     to pure white (schlieren = 1).

    Output is in [0, 1]; rendered with cmap='gray' or 'gray_r'.  Smooth
    regions appear WHITE; shocks/contacts appear BLACK.
    """
    drho_dx = np.gradient(rho, dx, axis=0)
    drho_dy = np.gradient(rho, dy, axis=1)
    g       = np.sqrt(drho_dx * drho_dx + drho_dy * drho_dy)

    gas_mask = (eta > 0.5)
    liq_mask = ~gas_mask

    g_max_air = float(g[gas_mask].max()) if gas_mask.any() else 1.0
    g_max_liq = float(g[liq_mask].max()) if liq_mask.any() else 1.0

    # Normalize per phase, apply exponent, then exp transform.
    g_norm_air = np.zeros_like(g)
    g_norm_liq = np.zeros_like(g)
    if g_max_air > 0:
        g_norm_air[gas_mask] = (g[gas_mask] / g_max_air) ** alpha_exp
    if g_max_liq > 0:
        g_norm_liq[liq_mask] = (g[liq_mask] / g_max_liq) ** alpha_exp

    # Floor: below the threshold, no schlieren signal (pure white).
    g_norm_air[g_norm_air < floor_air] = 0.0
    g_norm_liq[g_norm_liq < floor_liq] = 0.0

    sch = np.ones_like(g)
    sch[gas_mask] = np.exp(-k_air * g_norm_air[gas_mask])
    sch[liq_mask] = np.exp(-k_liq * g_norm_liq[liq_mask])

    return sch


def overlay_eta_contours(ax, eta, x_grid, y_grid,
                         levels=ETA_CONTOURS_PAPER,
                         lw_thin=ETA_CONTOUR_LW_THIN,
                         lw_mid=ETA_CONTOUR_LW_MID,
                         color=ETA_CONTOUR_COLOR):
    """Draw eta = {0.01, 0.5, 0.99} (or user levels) on `ax`.

    Center level (closest to 0.5) gets weight `lw_mid`; envelope levels
    get the thinner `lw_thin`.  Used to add an interface band to schlieren,
    pressure, density, Mach, vorticity, velocity plots in paper figures.
    """
    levels_sorted = sorted(levels)
    # Identify which level is the center (closest to 0.5).
    center_lvl = min(levels_sorted, key=lambda v: abs(v - 0.5))
    for lvl in levels_sorted:
        lw = lw_mid if lvl == center_lvl else lw_thin
        try:
            ax.contour(x_grid, y_grid, eta, levels=[lvl],
                       colors=color, linewidths=lw,
                       linestyles='solid', alpha=0.85)
        except Exception:
            # Silently skip if a particular level produces no contour.
            pass

def extract_interface_contour(eta, x_grid, y_grid, threshold=0.5):
    """Extract interface contour coordinates from eta field"""
    fig_temp = plt.figure()
    cs = plt.contour(x_grid, y_grid, eta, levels=[threshold])
    plt.close(fig_temp)
    
    contours = []
    if len(cs.allsegs) > 0:
        for contour_path in cs.allsegs[0]:
            if len(contour_path) > 0:
                contours.append(contour_path)
    
    return contours

def compute_deformation_params(eta, x_grid, y_grid, threshold=0.5, method='bbox'):
    """Calculate droplet deformation parameters"""
    droplet_mask = (eta < threshold)
    
    if not np.any(droplet_mask):
        return 0.0, 1.0, (DROPLET_CENTER_X, DROPLET_CENTER_Y), 0.0, 0.0
    
    y_indices, x_indices = np.where(droplet_mask)
    x_c = np.mean(x_grid[y_indices, x_indices])
    y_c = np.mean(y_grid[y_indices, x_indices])
    
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]
    volume = np.sum(droplet_mask) * dx * dy
    
    if method == 'bbox':
        x_coords = x_grid[y_indices, x_indices]
        y_coords = y_grid[y_indices, x_indices]
        L = np.max(x_coords) - np.min(x_coords)
        W = np.max(y_coords) - np.min(y_coords)
    
    if L + W > 0:
        D = (L - W) / (L + W)
        AR = L / W if W > 0 else 1.0
    else:
        D = 0.0
        AR = 1.0
    
    contours = extract_interface_contour(eta, x_grid, y_grid, threshold)
    area = 0.0
    for contour in contours:
        if len(contour) > 1:
            dx_contour = np.diff(contour[:, 0])
            dy_contour = np.diff(contour[:, 1])
            area += np.sum(np.sqrt(dx_contour**2 + dy_contour**2))
    
    return D, AR, (x_c, y_c), area, volume

def track_shock_position(field, x_grid, method='pressure', threshold=1e4):
    """Identify shock front position"""
    centerline_idx = field.shape[0] // 2
    field_centerline = field[centerline_idx, :]
    x_centerline = x_grid[centerline_idx, :]
    
    dx = x_centerline[1] - x_centerline[0]
    grad_field = np.gradient(field_centerline, dx)
    max_grad_idx = np.argmax(np.abs(grad_field))
    
    if np.abs(grad_field[max_grad_idx]) > threshold:
        x_shock = x_centerline[max_grad_idx]
    else:
        x_shock = None
    
    return x_shock

def compute_dimensionless_numbers(rho_air, u_rel, D, sigma, mu_air):
    """Calculate We, Re, Ma"""
    We = rho_air * u_rel**2 * D / sigma
    Re = rho_air * u_rel * D / mu_air
    a_air = np.sqrt(GAMMA_AIR * P_ATM / rho_air)
    Ma = u_rel / a_air
    return We, Re, Ma

def compute_kinetic_energy(rho, vx, vy, dx, dy):
    """Calculate total kinetic energy"""
    KE = 0.5 * np.sum(rho * (vx**2 + vy**2)) * dx * dy
    return KE

def compute_surface_energy(eta, x_grid, y_grid, sigma, threshold=0.5):
    """Calculate surface energy"""
    contours = extract_interface_contour(eta, x_grid, y_grid, threshold)
    area = 0.0
    for contour in contours:
        if len(contour) > 1:
            dx_contour = np.diff(contour[:, 0])
            dy_contour = np.diff(contour[:, 1])
            area += np.sum(np.sqrt(dx_contour**2 + dy_contour**2))
    SE = sigma * area
    return SE

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("SHOCK-DROPLET INTERACTION ANALYSIS v2.0")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    # Exclude backup files (.old), checkpoint files, and other non-plot directories
    if os.path.isdir(item_path):
        # Check if it's a valid plot file (contains 'plt' or starts with digits)
        # Exclude .old files and other temporary files
        if '.old' not in item and 'chk' not in item.lower():
            # Additional check: valid AMReX plot directories typically contain Header file
            header_file = os.path.join(item_path, 'Header')
            if os.path.exists(header_file):
                plot_files.append(item_path)
            else:
                print(f"  Skipping (no Header): {item}")

if not plot_files:
    print(f"ERROR: No valid plot files found in {amrex_output_dir}")
    print(f"\nDirectory contents:")
    for item in os.listdir(amrex_output_dir):
        print(f"  - {item}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} valid plot files")
print(f"First file: {os.path.basename(plot_files[0])}")
print(f"Last file: {os.path.basename(plot_files[-1])}")


# ============================================================================
# EXTRACT DATA FROM ALL TIMESTEPS
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM ALL TIMESTEPS")
print("=" * 70)

times = []
all_data = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    times.append(t)
    all_data.append(ds)

    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Loaded {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)

# ------------------------------------------------------------------------
# AUTHORITATIVE TIME SORT: re-sort plot_files, all_data, times by ACTUAL
# ds.current_time, not by filename digits.  Filename sort can fail when a
# directory prefix contains digits (e.g. "output_v2_00021cell" matches
# "2", not "21"), producing scrambled frame sequences.  This pass
# guarantees the master time array is monotonically increasing.
# ------------------------------------------------------------------------
sort_idx = np.argsort(times)
if not np.array_equal(sort_idx, np.arange(len(times))):
    print(f"  [info] re-sorting {len(times)} plot files by actual time "
          f"(filename sort produced {(sort_idx != np.arange(len(times))).sum()} mis-orderings)")
    times      = times[sort_idx]
    all_data   = [all_data[i]   for i in sort_idx]
    plot_files = [plot_files[i] for i in sort_idx]

# ============================================================================
# APPLY TIME_STEP SAMPLING
# ============================================================================

print("\n" + "=" * 70)
print(f"APPLYING TIME_STEP SAMPLING (TIME_STEP = {TIME_STEP})")
print("=" * 70)

if USE_ALL_TIMESTEPS:
    analysis_indices = list(range(0, len(plot_files), TIME_STEP))
    print(f"Using every {TIME_STEP} timestep(s): {len(analysis_indices)} total")
else:
    base_indices = find_closest_timesteps(times, KEY_TIMES)
    analysis_indices = base_indices[::TIME_STEP]
    print(f"Using {len(analysis_indices)} key timesteps with TIME_STEP={TIME_STEP}")

for idx in analysis_indices[:5]:
    print(f"  t = {times[idx]:.6e} s")
if len(analysis_indices) > 5:
    print(f"  ... ({len(analysis_indices)-5} more)")

# Select timesteps for grid plots
grid_indices = select_evenly_spaced(analysis_indices, NUM_GRID_TIMES)

# ============================================================================
# EXTRACT AND COMPUTE FIELDS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING DERIVED FIELDS")
print("=" * 70)

deformation_data = []
energy_data = []
shock_positions = []
dimensionless_data = []

schlieren_fields = []
pressure_fields = []
density_fields = []
velocity_fields = []
vorticity_fields = []
temperature_fields = []
vap_dot_rho_fields = []
eta_fields = []
x_grids = []
y_grids = []

for i, idx in enumerate(analysis_indices):
    ds = all_data[idx]
    t = times[idx]
    
    slc = ds.slice('z', 0.0)
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    resolution = 512
    
    frb = slc.to_frb((domain_width, 'code_length'), resolution, 
                     center=[0.5*(X_MIN+X_MAX), 0.5*(Y_MIN+Y_MAX), 0.0],
                     height=(domain_height, 'code_length'))
    
    if SCHLIEREN_USE_MIXTURE:
        rho = np.array(frb['density'])
    else:
        eta = np.array(frb['eta'])
        rho0 = np.array(frb['density0'])
        rho1 = np.array(frb['density1'])
        rho = eta * rho0 + (1 - eta) * rho1
    
    pressure = np.array(frb['pressure'])
    eta_field = np.array(frb['eta'])
    vx = np.array(frb['velocityx'])
    vy = np.array(frb['velocityy'])
    temperature = np.array(frb['T'])
    vap_dot_rho = np.array(frb['Vap_dot_rho'])

    
    try:
        vorticity = np.array(frb['vorticity'])
    except:
        dx = domain_width / resolution
        dy = domain_height / resolution
        dvx_dy = np.gradient(vx, dy, axis=0)
        dvy_dx = np.gradient(vy, dx, axis=1)
        vorticity = dvy_dx - dvx_dy
    
    x_1d = np.linspace(X_MIN, X_MAX, resolution)
    y_1d = np.linspace(Y_MIN, Y_MAX, resolution)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    
    dx = x_1d[1] - x_1d[0]
    dy = y_1d[1] - y_1d[0]
    # 4Paper: per-phase schlieren (Khare/Saurel-style).  Falls back to the
    # legacy mixture schlieren if SCHLIEREN_PER_PHASE = 0.
    if SCHLIEREN_PER_PHASE:
        schlieren = compute_schlieren_per_phase(
            rho, eta_field, dx, dy,
            k_air=SCHLIEREN_K_AIR, k_liq=SCHLIEREN_K_LIQ,
            alpha_exp=SCHLIEREN_ALPHA_EXP,
            floor_air=SCHLIEREN_GRAD_FLOOR_AIR,
            floor_liq=SCHLIEREN_GRAD_FLOOR_LIQ,
        )
    else:
        schlieren = compute_schlieren(rho, dx, dy, SCHLIEREN_BETA, SCHLIEREN_LOG_SCALE)

    schlieren_fields.append(schlieren)
    pressure_fields.append(pressure)
    density_fields.append(rho)
    velocity_fields.append((vx, vy))
    vorticity_fields.append(vorticity)
    temperature_fields.append(temperature)
    vap_dot_rho_fields.append(vap_dot_rho)
    eta_fields.append(eta_field)
    x_grids.append(x_grid)
    y_grids.append(y_grid)
    
    D, AR, centroid, area, volume = compute_deformation_params(
        eta_field, x_grid, y_grid, ETA_THRESHOLD, DEFORMATION_METHOD
    )
    deformation_data.append({
        'D': D, 'AR': AR, 'centroid': centroid, 
        'area': area, 'volume': volume
    })
    
    KE = compute_kinetic_energy(rho, vx, vy, dx, dy)
    SE = compute_surface_energy(eta_field, x_grid, y_grid, SIGMA, ETA_THRESHOLD)
    p_max = np.max(pressure)
    energy_data.append({'KE': KE, 'SE': SE, 'p_max': p_max})
    
    x_shock = track_shock_position(pressure, x_grid, SHOCK_DETECTION_METHOD, 
                                   SHOCK_GRADIENT_THRESHOLD)
    shock_positions.append(x_shock)
    
    u_rel = np.max(np.sqrt(vx**2 + vy**2))
    D_current = 2 * np.sqrt(volume / np.pi)
    if D_current > 0:
        We, Re, Ma = compute_dimensionless_numbers(RHO_AIR, u_rel, D_current, 
                                                   SIGMA, MU_AIR)
    else:
        We, Re, Ma = 0, 0, 0
    dimensionless_data.append({'We': We, 'Re': Re, 'Ma': Ma})
    
    if (i + 1) % 5 == 0 or i == len(analysis_indices) - 1:
        print(f"  Processed {i + 1}/{len(analysis_indices)} timesteps")

print("  Derived field computation complete")

# ============================================================================
# COMPUTE GLOBAL MIN/MAX FOR FIXED COLORBARS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING GLOBAL MIN/MAX FOR FIXED COLORBARS")
print("=" * 70)

schlieren_min = np.nanmin([np.nanmin(s) for s in schlieren_fields])
schlieren_max = np.nanmax([np.nanmax(s) for s in schlieren_fields])
print(f"  Schlieren range: [{schlieren_min:.6e}, {schlieren_max:.6e}]")

pressure_min = 0.0 # np.nanmin([np.nanmin(p) for p in pressure_fields])
pressure_max = np.nanmax([np.nanmax(p) for p in pressure_fields])
print(f"  Pressure range: [{pressure_min:.6e}, {pressure_max:.6e}] Pa")

density_min = np.nanmin([np.nanmin(rho) for rho in density_fields])
density_max = np.nanmax([np.nanmax(rho) for rho in density_fields])
print(f"  Density range: [{density_min:.6e}, {density_max:.6e}] kg/m^3")

v_mag_min = 0.0
v_mag_max = np.nanmax([np.nanmax(np.sqrt(vx**2 + vy**2)) for vx, vy in velocity_fields])
print(f"  Velocity range: [{v_mag_min:.6e}, {v_mag_max:.6e}] m/s")

temperature_min = np.nanmin([np.nanmin(T) for T in temperature_fields])
temperature_max = np.nanmax([np.nanmax(T) for T in temperature_fields])
print(f"  Temperature range: [{temperature_min:.6e}, {temperature_max:.6e}] K")

vap_dot_rho_min = np.nanmin([np.nanmin(v) for v in vap_dot_rho_fields])
vap_dot_rho_max = np.nanmax([np.nanmax(v) for v in vap_dot_rho_fields])
print(f"  Vap_dot_rho range: [{vap_dot_rho_min:.6e}, {vap_dot_rho_max:.6e}] kg/m^3/s")

# For symmetric colorbar on vap_dot_rho
vap_dot_rho_lim = max(abs(vap_dot_rho_min), abs(vap_dot_rho_max))
print(f"  Vap_dot_rho symmetric range: [{-vap_dot_rho_lim:.6e}, {vap_dot_rho_lim:.6e}] kg/m^3/s")

# Smart vorticity filtering (2sigma method)
vort_all = np.concatenate([v.flatten() for v in vorticity_fields])
vort_all = vort_all[np.isfinite(vort_all)]  # Remove NaN and Inf
vort_mean = np.mean(vort_all)
vort_std = np.std(vort_all)
vort_lim = VORTICITY_STD_MULTIPLIER * vort_std
print(f"  Vorticity range (+/-{VORTICITY_STD_MULTIPLIER}sigma): [{-vort_lim:.6e}, {vort_lim:.6e}] 1/s")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_schlieren_pressure_split_single(idx, frame_num, save_folder):
    """Plot split view with timestamp overlay - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    
    t = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    pressure = pressure_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))
    
    schlieren_top = schlieren[zero_idx:, :]
    pressure_bottom = pressure[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]
    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]
    
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    # TOP - SCHLIEREN
    im1 = ax_top.imshow(schlieren_top, origin='lower', extent=extent_top,
                        cmap=COLORMAP_SCHLIEREN + '_r', vmin=schlieren_min, vmax=schlieren_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val], colors='red',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren', fontsize=FONT_SIZE_LABEL)
    
    # BOTTOM - PRESSURE
    im2 = ax_bot.imshow(pressure_bottom, origin='lower', extent=extent_bot,
                        cmap=COLORMAP_PRESSURE, vmin=pressure_min, vmax=pressure_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val], colors='white',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    
    # TIMESTAMP OVERLAY (bottom left) - NO SPECIAL CHARACTERS
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
               fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                        ax_top.get_position().width, ax_top.get_position().height])
    
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Schlieren_Pressure.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_schlieren_pressure_grid_1x6():
    """1x6 horizontal grid for publication - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_GRID_1X6)
    gs = GridSpec(2, NUM_GRID_TIMES, figure=fig, hspace=0.05, wspace=0.15,
                  height_ratios=[1, 1])
    
    for panel_idx, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        schlieren = schlieren_fields[idx]
        pressure = pressure_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        centroid = deformation_data[idx]['centroid']
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        ax_top = fig.add_subplot(gs[0, panel_idx])
        ax_bottom = fig.add_subplot(gs[1, panel_idx], sharex=ax_top)
        
        # Top: Schlieren
        mask_top = y_grid >= 0
        schlieren_top = np.where(mask_top, schlieren, np.nan)
        im1 = ax_top.contourf(x_grid, y_grid, schlieren_top, levels=20,
                             cmap=COLORMAP_SCHLIEREN + '_r', vmin=schlieren_min, vmax=schlieren_max)
        for eta_val in ETA_CONTOURS:
            ax_top.contour(x_grid, y_grid, eta, levels=[eta_val], colors='red', linewidths=1.0)
        
        # NO SPECIAL CHARACTERS in title
        ax_top.set_title(f't = {t*1e6:.1f} us', fontsize=FONT_SIZE_LABEL-2)
        ax_top.set_xlim([x_min_zoom, x_max_zoom])
        ax_top.set_ylim([max(0, y_min_zoom), y_max_zoom])
        ax_top.set_aspect(ASPECT_RATIO, adjustable='box')
        ax_top.tick_params(labelsize=FONT_SIZE_TICK-3, labelbottom=False, labelleft=(panel_idx==0))
        if panel_idx == 0:
            ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        
        # Bottom: Pressure
        mask_bottom = y_grid < 0
        pressure_bottom = np.where(mask_bottom, pressure, np.nan)
        im2 = ax_bottom.contourf(x_grid, y_grid, pressure_bottom, levels=20,
                                cmap=COLORMAP_PRESSURE, vmin=pressure_min, vmax=pressure_max)
        for eta_val in ETA_CONTOURS:
            ax_bottom.contour(x_grid, y_grid, eta, levels=[eta_val], colors='white', linewidths=1.0)
        
        ax_bottom.set_xlim([x_min_zoom, x_max_zoom])
        ax_bottom.set_ylim([y_min_zoom, min(0, y_max_zoom)])
        ax_bottom.set_aspect(ASPECT_RATIO, adjustable='box')
        ax_bottom.tick_params(labelsize=FONT_SIZE_TICK-3, labelleft=(panel_idx==0))
        ax_bottom.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
        if panel_idx == 0:
            ax_bottom.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
    
    fig.suptitle('Shock-Droplet Interaction: Schlieren (top) & Pressure (bottom)',
                fontsize=FONT_SIZE_TITLE, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_folder, f'02_Schlieren_Pressure_Grid_1x6.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'02_Schlieren_Pressure_Grid_1x6.{SAVE_FORMAT_VECTOR}'))
    plt.close()


def plot_velocity_field_single(idx, frame_num, save_folder):
    """Velocity field with streamlines and timestamp - NO SPECIAL CHARACTERS"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    t = times[analysis_indices[idx]]
    vx, vy = velocity_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    v_mag = np.sqrt(vx**2 + vy**2)
    
    im = ax.contourf(x_grid, y_grid, v_mag, levels=30, cmap=COLORMAP_VELOCITY,
                    vmin=v_mag_min, vmax=v_mag_max)
    
    if STREAMLINE_SHOW:
        ax.streamplot(x_grid[0, :], y_grid[:, 0], vx, vy, color='white',
                     density=STREAMLINE_DENSITY, linewidth=0.8, arrowsize=0.8)
    
    if VECTOR_SHOW:
        skip = 12
        ax.quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                 vx[::skip, ::skip], vy[::skip, ::skip],
                 color='black', alpha=0.5, scale=v_mag_max*20)
    
    for eta_val in ETA_CONTOURS:
        ax.contour(x_grid, y_grid, eta, levels=[eta_val], colors='black',
                  linewidths=2, linestyles='--' if eta_val != 0.5 else '-')
    
    ax.set_xlim([x_min_zoom, x_max_zoom])
    ax.set_ylim([y_min_zoom, y_max_zoom])
    ax.set_aspect(ASPECT_RATIO)
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'Velocity Field - t = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    # Timestamp overlay - NO SPECIAL CHARACTERS
    ax.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax.transAxes,
           fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('|V| (m/s)', fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Velocity_Streamline.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_vorticity_field_single(idx, frame_num, save_folder):
    """Vorticity field with timestamp - NO SPECIAL CHARACTERS"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    t = times[analysis_indices[idx]]
    vorticity = vorticity_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    im = ax.contourf(x_grid, y_grid, vorticity, levels=30, cmap=COLORMAP_VORTICITY,
                    vmin=-vort_lim, vmax=vort_lim)
    
    for eta_val in ETA_CONTOURS:
        ax.contour(x_grid, y_grid, eta, levels=[eta_val], colors='black',
                  linewidths=2, linestyles='--' if eta_val != 0.5 else '-')
    
    ax.set_xlim([x_min_zoom, x_max_zoom])
    ax.set_ylim([y_min_zoom, y_max_zoom])
    ax.set_aspect(ASPECT_RATIO)
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_title(f'Vorticity Field - t = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    # Timestamp overlay - NO SPECIAL CHARACTERS
    ax.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax.transAxes,
           fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Vorticity (1/s)', fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Vorticity.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()

def plot_schlieren_velocity_split_single(idx, frame_num, save_folder):
    """Plot Schlieren (top) and Velocity (bottom) split view - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    
    t = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    vx, vy = velocity_fields[idx]
    v_mag = np.sqrt(vx**2 + vy**2)
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))
    
    schlieren_top = schlieren[zero_idx:, :]
    velocity_bottom = v_mag[:zero_idx + 1, :]
    vx_bottom = vx[:zero_idx + 1, :]
    vy_bottom = vy[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]
    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]
    
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    # TOP - SCHLIEREN
    im1 = ax_top.imshow(schlieren_top, origin='lower', extent=extent_top,
                        cmap=COLORMAP_SCHLIEREN + '_r', vmin=schlieren_min, vmax=schlieren_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val], colors='red',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren', fontsize=FONT_SIZE_LABEL)
    
    # BOTTOM - VELOCITY MAGNITUDE
    im2 = ax_bot.imshow(velocity_bottom, origin='lower', extent=extent_bot,
                        cmap=COLORMAP_VELOCITY, vmin=v_mag_min, vmax=v_mag_max,
                        interpolation='bilinear', aspect='auto')
    
    # Add streamlines on bottom half
    if STREAMLINE_SHOW:
        # Create meshgrid for bottom half only
        x_bottom_grid = x_grid[:zero_idx + 1, :]
        y_bottom_grid = y_grid[:zero_idx + 1, :]
        ax_bot.streamplot(x_vals, y_bottom, vx_bottom, vy_bottom, 
                         color='white', density=STREAMLINE_DENSITY, 
                         linewidth=0.8, arrowsize=0.8)
    
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val], colors='white',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    
    # TIMESTAMP OVERLAY
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
               fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('|V| (m/s)', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                        ax_top.get_position().width, ax_top.get_position().height])
    
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Schlieren_Velocity.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_schlieren_vapdotrho_split_single(idx, frame_num, save_folder):
    """Plot Schlieren (top) and Vap_dot_rho (bottom) split view - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    
    t = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    vap_dot_rho = vap_dot_rho_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))
    
    schlieren_top = schlieren[zero_idx:, :]
    vapdotrho_bottom = vap_dot_rho[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]
    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]
    
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    # TOP - SCHLIEREN
    im1 = ax_top.imshow(schlieren_top, origin='lower', extent=extent_top,
                        cmap=COLORMAP_SCHLIEREN + '_r', vmin=schlieren_min, vmax=schlieren_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val], colors='red',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren', fontsize=FONT_SIZE_LABEL)
    
    # BOTTOM - VAP_DOT_RHO (symmetric colorbar)
    im2 = ax_bot.imshow(vapdotrho_bottom, origin='lower', extent=extent_bot,
                        cmap=COLORMAP_VAPDOTRHO, vmin=-vap_dot_rho_lim, vmax=vap_dot_rho_lim,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val], colors='black',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    
    # TIMESTAMP OVERLAY
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
               fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Vap_dot_rho (kg/m^3/s)', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                        ax_top.get_position().width, ax_top.get_position().height])
    
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Schlieren_VapDotRho.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_velocity_vorticity_split_single(idx, frame_num, save_folder):
    """Plot Velocity (top) and Vorticity (bottom) split view - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    
    t = times[analysis_indices[idx]]
    vx, vy = velocity_fields[idx]
    v_mag = np.sqrt(vx**2 + vy**2)
    vorticity = vorticity_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))
    
    velocity_top = v_mag[zero_idx:, :]
    vx_top = vx[zero_idx:, :]
    vy_top = vy[zero_idx:, :]
    vorticity_bottom = vorticity[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]
    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]
    
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    # TOP - VELOCITY MAGNITUDE
    im1 = ax_top.imshow(velocity_top, origin='lower', extent=extent_top,
                        cmap=COLORMAP_VELOCITY, vmin=v_mag_min, vmax=v_mag_max,
                        interpolation='bilinear', aspect='auto')
    
    # Add streamlines on top half
    if STREAMLINE_SHOW:
        ax_top.streamplot(x_vals, y_top, vx_top, vy_top, 
                         color='white', density=STREAMLINE_DENSITY, 
                         linewidth=0.8, arrowsize=0.8)
    
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val], colors='black',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('|V| (m/s)', fontsize=FONT_SIZE_LABEL)
    
    # BOTTOM - VORTICITY
    im2 = ax_bot.imshow(vorticity_bottom, origin='lower', extent=extent_bot,
                        cmap=COLORMAP_VORTICITY, vmin=-vort_lim, vmax=vort_lim,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val], colors='black',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    
    # TIMESTAMP OVERLAY
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
               fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Vorticity (1/s)', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                        ax_top.get_position().width, ax_top.get_position().height])
    
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Velocity_Vorticity.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_schlieren_temperature_split_single(idx, frame_num, save_folder):
    """Plot Schlieren (top) and Temperature (bottom) split view - NO SPECIAL CHARACTERS"""
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)
    
    t = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    temperature = temperature_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]
    
    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))
    
    schlieren_top = schlieren[zero_idx:, :]
    temperature_bottom = temperature[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]
    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]
    
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]
    
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR
    
    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2
    
    # TOP - SCHLIEREN
    im1 = ax_top.imshow(schlieren_top, origin='lower', extent=extent_top,
                        cmap=COLORMAP_SCHLIEREN + '_r', vmin=schlieren_min, vmax=schlieren_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val], colors='red',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren', fontsize=FONT_SIZE_LABEL)
    
    # BOTTOM - TEMPERATURE
    im2 = ax_bot.imshow(temperature_bottom, origin='lower', extent=extent_bot,
                        cmap=COLORMAP_TEMPERATURE, vmin=temperature_min, vmax=temperature_max,
                        interpolation='bilinear', aspect='auto')
    
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val], colors='white',
                      linewidths=CONTOUR_LINE_WIDTH, linestyles='--' if eta_val != 0.5 else '-')
    
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    
    # TIMESTAMP OVERLAY
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
               fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Temperature (K)', fontsize=FONT_SIZE_LABEL)
    
    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                        ax_top.get_position().width, ax_top.get_position().height])
    
    save_path = os.path.join(save_folder, f'{frame_num:04d}_Schlieren_Temperature.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()

def plot_deformation_metrics():
    """Plot 3: Deformation parameters vs time - NO SPECIAL CHARACTERS"""
    t_analysis = times[analysis_indices]
    D_vals = [d['D'] for d in deformation_data]
    AR_vals = [d['AR'] for d in deformation_data]
    centroid_x = [d['centroid'][0] for d in deformation_data]
    centroid_y = [d['centroid'][1] for d in deformation_data]
    area_vals = [d['area'] for d in deformation_data]
    volume_vals = [d['volume'] for d in deformation_data]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    axes[0, 0].plot(t_analysis * 1e6, D_vals, 'b-', linewidth=LINE_WIDTH_NORMAL)
    axes[0, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel('Deformation D', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_title('Deformation Parameter', fontsize=FONT_SIZE_TITLE)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t_analysis * 1e6, AR_vals, 'r-', linewidth=LINE_WIDTH_NORMAL)
    axes[0, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel('Aspect Ratio', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_title('Aspect Ratio (L/W)', fontsize=FONT_SIZE_TITLE)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t_analysis * 1e6, np.array(centroid_x) * 1e3, 'g-', linewidth=LINE_WIDTH_NORMAL)
    axes[1, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel('Centroid X (mm)', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_title('Centroid X Position', fontsize=FONT_SIZE_TITLE)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t_analysis * 1e6, np.array(centroid_y) * 1e3, 'm-', linewidth=LINE_WIDTH_NORMAL)
    axes[1, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel('Centroid Y (mm)', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_title('Centroid Y Position', fontsize=FONT_SIZE_TITLE)
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(t_analysis * 1e6, np.array(area_vals) * 1e3, 'c-', linewidth=LINE_WIDTH_NORMAL)
    axes[2, 0].axhline(y=np.pi * D_DROPLET_INITIAL * 1e3, color='k', linestyle='--', label='Initial')
    axes[2, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_ylabel('Surface Area (mm)', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_title('Interface Perimeter', fontsize=FONT_SIZE_TITLE)
    axes[2, 0].legend(fontsize=FONT_SIZE_LEGEND)
    axes[2, 0].grid(True, alpha=0.3)
    
    V0 = np.pi * (D_DROPLET_INITIAL/2)**2
    axes[2, 1].plot(t_analysis * 1e6, np.array(volume_vals) / V0, 'y-', linewidth=LINE_WIDTH_NORMAL)
    axes[2, 1].axhline(y=1.0, color='k', linestyle='--', label='Initial')
    axes[2, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_ylabel('Volume / V0', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_title('Normalized Volume', fontsize=FONT_SIZE_TITLE)
    axes[2, 1].legend(fontsize=FONT_SIZE_LEGEND)
    axes[2, 1].grid(True, alpha=0.3)
    
    fig.suptitle('Droplet Deformation Metrics', fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'03_Deformation_Metrics.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'03_Deformation_Metrics.{SAVE_FORMAT_VECTOR}'))
    plt.close()

    # ----------------------------------------------------------------------
    # 4Paper: also save each metric as its own standalone publication figure.
    # Folder: Deformation-Separate/.  Each panel gets BOTH raster and vector.
    # ----------------------------------------------------------------------
    if SEPARATE_DEFORMATION_FIGS:
        V0 = np.pi * (D_DROPLET_INITIAL / 2) ** 2
        # (name, y, ylabel, color, title, optional hline_value, hline_label)
        standalones = [
            ("D",        D_vals,                              "Deformation $D$",      "tab:blue",   "Deformation Parameter",   None,        None),
            ("AR",       AR_vals,                             "Aspect Ratio $L/W$",   "tab:red",    "Aspect Ratio",            None,        None),
            ("CentroidX", np.array(centroid_x) * 1e3,         "Centroid $x$ [mm]",    "tab:green",  "Centroid X Position",     None,        None),
            ("CentroidY", np.array(centroid_y) * 1e3,         "Centroid $y$ [mm]",    "tab:purple", "Centroid Y Position",     None,        None),
            ("Perimeter", np.array(area_vals) * 1e3,          "Surface $A$ [mm]",      "tab:cyan",   "Interface Perimeter",
                np.pi * D_DROPLET_INITIAL * 1e3, "initial"),
            ("Volume",    np.array(volume_vals) / V0,         r"$V / V_0$",            "tab:olive",  "Normalized Volume",
                1.0, "initial"),
        ]
        for tag, y, ylabel, color, title, hval, hlabel in standalones:
            fig_s, ax_s = plt.subplots(figsize=(6.5, 4.5))
            ax_s.plot(t_analysis * 1e6, y, color=color, linewidth=LINE_WIDTH_NORMAL)
            if hval is not None:
                ax_s.axhline(y=hval, color='k', linestyle='--', linewidth=1.0, label=hlabel)
                ax_s.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
            ax_s.set_xlabel(r"Time [$\mu$s]", fontsize=FONT_SIZE_LABEL)
            ax_s.set_ylabel(ylabel, fontsize=FONT_SIZE_LABEL)
            ax_s.set_title(title, fontsize=FONT_SIZE_TITLE)
            ax_s.grid(True, alpha=0.3)
            ax_s.tick_params(labelsize=FONT_SIZE_TICK)
            fig_s.tight_layout()
            base = os.path.join(subfolder_deformation_separate, f"Deformation_{tag}")
            fig_s.savefig(f"{base}.{SAVE_FORMAT_RASTER}", dpi=DPI)
            fig_s.savefig(f"{base}.{SAVE_FORMAT_VECTOR}")
            plt.close(fig_s)


# ============================================================================
# AR / SURFACE-AREA vs INTEGRATED VAPORIZATION (twin-axis charts)
# ============================================================================

def _integrate_vap_dot_rho():
    """Return array (len = len(analysis_indices)) of int(Vap_dot_rho) dV.

    Uses the resampled Vap_dot_rho field stored in vap_dot_rho_fields, with
    cell area derived from domain extent / field shape.
    """
    vap_int = np.zeros(len(analysis_indices))
    for i in range(len(analysis_indices)):
        vd = np.asarray(vap_dot_rho_fields[i])
        if vd.ndim != 2 or vd.size == 0:
            vap_int[i] = np.nan
            continue
        dx = (X_MAX - X_MIN) / vd.shape[1]
        dy = (Y_MAX - Y_MIN) / vd.shape[0]
        vap_int[i] = float(np.nansum(vd)) * dx * dy
    return vap_int


def _twin_axis_chart(t_us, left_y, right_y, left_label, right_label,
                     left_color, right_color, title, save_basename):
    """Generic twin-axis plot helper: left_y vs right_y vs time."""
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    ax1.plot(t_us, left_y, color=left_color, linewidth=LINE_WIDTH_NORMAL,
             label=left_label)
    ax1.set_xlabel(r"Time [$\mu$s]", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel(left_label, color=left_color, fontsize=FONT_SIZE_LABEL)
    ax1.tick_params(axis='y', labelcolor=left_color, labelsize=FONT_SIZE_TICK)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(t_us, right_y, color=right_color, linewidth=LINE_WIDTH_NORMAL,
             linestyle='--', label=right_label)
    ax2.set_ylabel(right_label, color=right_color, fontsize=FONT_SIZE_LABEL)
    ax2.tick_params(axis='y', labelcolor=right_color, labelsize=FONT_SIZE_TICK)

    ax1.set_title(title, fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    fig.savefig(f"{save_basename}.{SAVE_FORMAT_RASTER}", dpi=DPI)
    fig.savefig(f"{save_basename}.{SAVE_FORMAT_VECTOR}")
    plt.close(fig)


def plot_AR_vs_vap_dot_rho():
    t_us = times[analysis_indices] * 1e6
    AR_vals = np.array([d['AR'] for d in deformation_data])
    vap_int = _integrate_vap_dot_rho()
    _twin_axis_chart(t_us, AR_vals, vap_int,
                     left_label='Aspect Ratio L/W',
                     right_label=r'$\int \dot{m}_{vap}\,dV$ [kg/s]',
                     left_color='tab:blue', right_color='tab:red',
                     title='Aspect Ratio vs Integrated Vaporization',
                     save_basename=os.path.join(output_folder, '06_AR_vs_VapDotRho'))


def plot_area_vs_vap_dot_rho():
    t_us = times[analysis_indices] * 1e6
    area_vals = np.array([d['area'] for d in deformation_data]) * 1e3  # mm
    vap_int = _integrate_vap_dot_rho()
    _twin_axis_chart(t_us, area_vals, vap_int,
                     left_label='Surface Area [mm]',
                     right_label=r'$\int \dot{m}_{vap}\,dV$ [kg/s]',
                     left_color='tab:cyan', right_color='tab:red',
                     title='Surface Area vs Integrated Vaporization',
                     save_basename=os.path.join(output_folder, '07_Area_vs_VapDotRho'))


# ============================================================================
# 4PAPER PLOT FUNCTIONS
# ============================================================================

def plot_shape_evolution_composite():
    """Single figure with eta = 0.5 outline overlaid at multiple times.

    Khare 2022 Fig 6 / Sembian et al. style.  Color-codes each contour by
    time using SHAPE_EVOLUTION_CMAP.  Useful as a one-glance summary of the
    droplet's deformation history.
    """
    if len(plot_files) < 2:
        print("  [skip] shape evolution: need at least 2 frames.")
        return

    # Pick SHAPE_EVOLUTION_N_FRAMES evenly spaced indices spanning the FULL
    # plot_files range (start -> end of simulation), independent of the
    # TIME_STEP / KEY_TIMES sampling used elsewhere.
    n_frames = min(SHAPE_EVOLUTION_N_FRAMES, len(plot_files))
    sel      = list(np.linspace(0, len(plot_files) - 1, n_frames, dtype=int))
    seen = set()
    sel  = [i for i in sel if not (i in seen or seen.add(i))]
    n_frames = len(sel)
    times_sel = times[sel]
    cmap = plt.get_cmap(SHAPE_EVOLUTION_CMAP)

    print(f"  loading eta for {n_frames} frames (independent of main pipeline)")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Compute bounding box across all overlaid contours so the zoom is tight.
    x_all = []
    y_all = []

    for j, file_idx in enumerate(sel):
        ds  = all_data[file_idx]
        slc = ds.slice('z', 0.0)
        domain_width  = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        resolution    = 512
        frb = slc.to_frb((domain_width, 'code_length'), resolution,
                         center=[0.5*(X_MIN+X_MAX), 0.5*(Y_MIN+Y_MAX), 0.0],
                         height=(domain_height, 'code_length'))
        eta = np.array(frb['eta'])
        x_1d = np.linspace(X_MIN, X_MAX, resolution)
        y_1d = np.linspace(Y_MIN, Y_MAX, resolution)
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)
        color = cmap(j / max(n_frames - 1, 1))

        cs = ax.contour(x_grid, y_grid, eta, levels=[0.5],
                        colors=[color], linewidths=SHAPE_EVOLUTION_LW)
        for seg_list in cs.allsegs:
            for seg in seg_list:
                if len(seg) > 0:
                    x_all.append(seg[:, 0])
                    y_all.append(seg[:, 1])

    if x_all:
        x_all_flat = np.concatenate(x_all)
        y_all_flat = np.concatenate(y_all)
        pad_x = 0.1 * (x_all_flat.max() - x_all_flat.min() + 1e-12)
        pad_y = 0.1 * (y_all_flat.max() - y_all_flat.min() + 1e-12)
        ax.set_xlim(x_all_flat.min() - pad_x, x_all_flat.max() + pad_x)
        ax.set_ylim(y_all_flat.min() - pad_y, y_all_flat.max() + pad_y)

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r"$x$ [m]", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$y$ [m]", fontsize=FONT_SIZE_LABEL)
    ax.set_title(r"Droplet shape evolution ($\eta = 0.5$ contour)",
                 fontsize=FONT_SIZE_TITLE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Colorbar mapped to time.
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=times_sel.min() * 1e6, vmax=times_sel.max() * 1e6))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.04)
    cbar.set_label(r"Time [$\mu$s]", fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)

    fig.tight_layout()
    base = os.path.join(output_folder, "04Paper_ShapeEvolution")
    fig.savefig(f"{base}.{SAVE_FORMAT_RASTER}", dpi=DPI)
    fig.savefig(f"{base}.{SAVE_FORMAT_VECTOR}")
    plt.close(fig)


def _mach_field(pressure, rho, vx, vy, eta):
    """Compute Mach number on the resampled FRB grid.

    Uses an eta-weighted mixture sound speed (Tammann gas + Tammann liquid):
        c_eff = sqrt( gamma_eff * (p + gamma_eff * pi_eff) / rho )
    with gamma_eff = eta * GAMMA_AIR + (1-eta) * GAMMA_WATER and similar
    for pi.  Approximation -- frozen mixture c is a more proper choice but
    requires per-phase masses; this is good enough for the visualization.
    """
    eta_c = np.clip(eta, 0.0, 1.0)
    gamma_eff = eta_c * GAMMA_AIR + (1.0 - eta_c) * GAMMA_WATER
    pi_eff    = eta_c * P0_AIR    + (1.0 - eta_c) * P0_WATER
    rho_safe  = np.maximum(rho, 1.0e-12)
    c_sq      = gamma_eff * (np.maximum(pressure + gamma_eff * pi_eff, 0.0)) / rho_safe
    c_eff     = np.sqrt(np.maximum(c_sq, 1.0e-12))
    speed     = np.sqrt(vx * vx + vy * vy)
    return speed / c_eff


def plot_schlieren_mach_split_single(idx, frame_num, save_folder):
    """Schlieren (top) + Mach contour (bottom) split.

    Mirrors the structure of plot_schlieren_pressure_split_single -- same
    domain split at y = 0, same zoom logic, same overlay convention.  Mach
    colormap is 'hot' to emphasize supersonic regions.
    """
    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)
    gs = GridSpec(2, 1, figure=fig, hspace=0.0, height_ratios=[1, 1])
    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    t        = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    pressure  = pressure_fields[idx]
    rho       = density_fields[idx]
    vx, vy    = velocity_fields[idx]
    eta       = eta_fields[idx]
    x_grid    = x_grids[idx]
    y_grid    = y_grids[idx]

    mach = _mach_field(pressure, rho, vx, vy, eta)

    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]
    zero_idx = np.argmin(np.abs(y_vals))

    schlieren_top = schlieren[zero_idx:, :]
    mach_bottom   = mach[:zero_idx + 1, :]
    eta_top       = eta[zero_idx:, :]
    eta_bottom    = eta[:zero_idx + 1, :]
    y_top         = y_vals[zero_idx:]
    y_bottom      = y_vals[:zero_idx + 1]

    extent_top = [x_vals.min(), x_vals.max(), y_top.min(),    y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]

    centroid    = deformation_data[idx]['centroid']
    zoom_width  = (X_MAX - X_MIN) / PLOT_ZOOM_FACTOR
    zoom_height = (Y_MAX - Y_MIN) / PLOT_ZOOM_FACTOR
    x_min_zoom  = centroid[0] - zoom_width / 2
    x_max_zoom  = centroid[0] + zoom_width / 2
    y_min_zoom  = centroid[1] - zoom_height / 2
    y_max_zoom  = centroid[1] + zoom_height / 2

    # TOP: schlieren -- match v2 convention (white shocks on dark background).
    # NOTE: gray_r maps high schlieren -> white, so shock fronts read as bright
    # streaks against the bulk-fluid dark.
    im1 = ax_top.imshow(schlieren_top, origin='lower', extent=extent_top,
                        cmap='gray_r', vmin=0.0, vmax=1.0,
                        interpolation='bilinear', aspect='auto')
    for eta_val in ETA_CONTOURS:
        ax_top.contour(x_vals, y_top, eta_top, levels=[eta_val],
                       colors=ETA_CONTOUR_COLOR,
                       linewidths=(ETA_CONTOUR_LW_MID if eta_val == 0.5 else ETA_CONTOUR_LW_THIN))
    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y [m]', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t*1e6:.2f} us', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_top.tick_params(labelbottom=False)
    ax_top.spines['bottom'].set_visible(False)
    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren', fontsize=FONT_SIZE_LABEL)

    # BOTTOM: Mach number
    mach_max = max(float(np.nanmax(mach)), 1.0)
    im2 = ax_bot.imshow(mach_bottom, origin='lower', extent=extent_bot,
                        cmap='hot', vmin=0.0, vmax=mach_max,
                        interpolation='bilinear', aspect='auto')
    for eta_val in ETA_CONTOURS:
        ax_bot.contour(x_vals, y_bottom, eta_bottom, levels=[eta_val],
                       colors='white',
                       linewidths=(ETA_CONTOUR_LW_MID if eta_val == 0.5 else ETA_CONTOUR_LW_THIN))
    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X [m]', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y [m]', fontsize=FONT_SIZE_LABEL)
    ax_bot.spines['top'].set_visible(False)
    ax_bot.text(0.02, 0.02, f't = {t*1e6:.2f} us', transform=ax_bot.transAxes,
                fontsize=FONT_SIZE_TIMESTAMP, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Mach Number', fontsize=FONT_SIZE_LABEL)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.08, hspace=0.0)
    ax_top.set_position([ax_top.get_position().x0, ax_bot.get_position().y1,
                         ax_top.get_position().width, ax_top.get_position().height])

    save_path = os.path.join(save_folder, f'{frame_num:04d}_Schlieren_Mach.{SAVE_FORMAT_RASTER}')
    plt.savefig(save_path, dpi=DPI)
    plt.close()


def plot_mass_conservation():
    """Integrated mass diagnostic vs time.

    Computes int(rho_eta_k) dV at each plotfile time, prints final/initial
    drift % and saves a 2-panel figure (per-phase and total).  Direct
    quality-control diagnostic for paper supplementary materials.
    """
    mass0_arr = []
    mass1_arr = []
    mass_tot  = []

    for s, ds in enumerate(all_data):
        try:
            # Use a covering grid at finest level for accurate integration.
            cg = ds.covering_grid(
                level=ds.index.max_level,
                left_edge=ds.domain_left_edge,
                dims=ds.domain_dimensions * (ds.refine_by ** ds.index.max_level),
            )
            dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) / cg.shape[0])
            dy = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]) / cg.shape[1])
            dV = dx * dy
            m0 = float(np.asarray(cg["rho_eta0"]).sum()) * dV
            m1 = float(np.asarray(cg["rho_eta1"]).sum()) * dV
        except Exception as e:
            print(f"  [mass-conservation] skip frame {s}: {e}")
            mass0_arr.append(np.nan)
            mass1_arr.append(np.nan)
            mass_tot.append(np.nan)
            continue
        mass0_arr.append(m0)
        mass1_arr.append(m1)
        mass_tot.append(m0 + m1)

    mass0_arr = np.array(mass0_arr)
    mass1_arr = np.array(mass1_arr)
    mass_tot  = np.array(mass_tot)

    # Console summary
    print("\n  Mass conservation summary:")
    for name, q in [("rho_eta0", mass0_arr), ("rho_eta1", mass1_arr), ("total", mass_tot)]:
        q0 = q[0] if np.isfinite(q[0]) else float("nan")
        qN = q[-1] if np.isfinite(q[-1]) else float("nan")
        if np.isfinite(q0) and abs(q0) > 1e-30:
            drift = (qN - q0) / abs(q0) * 100
            print(f"    {name:<10s}: {q0:14.6e} -> {qN:14.6e}  drift = {drift:+.4e}%")

    # Figure: 2 panels (relative drift + absolute values)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    t_us = times * 1e6

    def _drift(q):
        return 100.0 * (q - q[0]) / max(abs(q[0]), 1e-30)

    axes[0].plot(t_us, _drift(mass0_arr), 'tab:blue',    lw=LINE_WIDTH_NORMAL, label=r"$\int \rho_{\eta 0}\,\mathrm{d}V$ (phase 0)")
    axes[0].plot(t_us, _drift(mass1_arr), 'tab:red',     lw=LINE_WIDTH_NORMAL, label=r"$\int \rho_{\eta 1}\,\mathrm{d}V$ (phase 1)")
    axes[0].plot(t_us, _drift(mass_tot),  'k--',         lw=LINE_WIDTH_THICK,  label=r"total")
    axes[0].set_ylabel(r"Mass drift [%]", fontsize=FONT_SIZE_LABEL)
    axes[0].set_title("Mass conservation diagnostic",
                      fontsize=FONT_SIZE_TITLE)
    axes[0].legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_us, mass0_arr, 'tab:blue', lw=LINE_WIDTH_NORMAL, label="phase 0")
    axes[1].plot(t_us, mass1_arr, 'tab:red',  lw=LINE_WIDTH_NORMAL, label="phase 1")
    axes[1].plot(t_us, mass_tot,  'k--',      lw=LINE_WIDTH_THICK,  label="total")
    axes[1].set_xlabel(r"Time [$\mu$s]", fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel("Integrated mass [kg/m]", fontsize=FONT_SIZE_LABEL)
    axes[1].legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.tick_params(labelsize=FONT_SIZE_TICK)
    fig.tight_layout()
    base = os.path.join(output_folder, "05Paper_MassConservation")
    fig.savefig(f"{base}.{SAVE_FORMAT_RASTER}", dpi=DPI)
    fig.savefig(f"{base}.{SAVE_FORMAT_VECTOR}")
    plt.close(fig)


# ============================================================================
# EXECUTE PLOTS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

plot_count = 0

if PLOT_SCHLIEREN_PRESSURE_SPLIT:
    print("\nGenerating Schlieren/Pressure Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_pressure_split_single(i, i+1, subfolder_schlieren)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_SCHLIEREN_PRESSURE_GRID:
    print("\nGenerating Schlieren/Pressure 1x6 Grid...")
    plot_schlieren_pressure_grid_1x6()
    print("  Saved: 02_Schlieren_Pressure_Grid_1x6")
    plot_count += 1

if PLOT_VELOCITY_FIELD:
    print("\nGenerating Velocity Field (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_velocity_field_single(i, i+1, subfolder_velocity)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_VORTICITY_FIELD:
    print("\nGenerating Vorticity Field (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_vorticity_field_single(i, i+1, subfolder_vorticity)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_DEFORMATION_METRICS:
    print("\nGenerating Deformation Metrics...")
    plot_deformation_metrics()
    print("  Saved: 03_Deformation_Metrics")
    plot_count += 1

if PLOT_AR_VS_VAP:
    print("\nGenerating AR vs integrated Vap_dot_rho...")
    plot_AR_vs_vap_dot_rho()
    print("  Saved: 06_AR_vs_VapDotRho")
    plot_count += 1

if PLOT_AREA_VS_VAP:
    print("\nGenerating Surface Area vs integrated Vap_dot_rho...")
    plot_area_vs_vap_dot_rho()
    print("  Saved: 07_Area_vs_VapDotRho")
    plot_count += 1

if PLOT_SCHLIEREN_VELOCITY_SPLIT:
    print("\nGenerating Schlieren/Velocity Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_velocity_split_single(i, i+1, subfolder_schlieren_velocity)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_SCHLIEREN_VAPDOTRHO_SPLIT:
    print("\nGenerating Schlieren/VapDotRho Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_vapdotrho_split_single(i, i+1, subfolder_schlieren_vapdotrho)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_VELOCITY_VORTICITY_SPLIT:
    print("\nGenerating Velocity/Vorticity Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_velocity_vorticity_split_single(i, i+1, subfolder_velocity_vorticity)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_SCHLIEREN_TEMPERATURE_SPLIT:
    print("\nGenerating Schlieren/Temperature Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_temperature_split_single(i, i+1, subfolder_schlieren_temperature)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1


# [Add calls to other plotting functions]

# ============================================================================
# GIF GENERATION
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING GIFS")
print("=" * 70)

if PLOT_SCHLIEREN_PRESSURE_GIF:
    print("\nCreating Schlieren/Pressure GIF...")
    create_gif_from_folder('Schlieren-Pressure', 'ANIM_Schlieren_Pressure.gif')

if PLOT_VELOCITY_FIELD_GIF:
    print("\nCreating Velocity/Streamline GIF...")
    create_gif_from_folder('Velocity-Streamline', 'ANIM_Velocity_Streamline.gif')

if PLOT_VORTICITY_FIELD_GIF:
    print("\nCreating Vorticity GIF...")
    create_gif_from_folder('Vorticity', 'ANIM_Vorticity.gif')

if PLOT_SCHLIEREN_VELOCITY_GIF:
    print("\nCreating Schlieren/Velocity GIF...")
    create_gif_from_folder('Schlieren-Velocity', 'ANIM_Schlieren_Velocity.gif')

if PLOT_SCHLIEREN_VAPDOTRHO_GIF:
    print("\nCreating Schlieren/VapDotRho GIF...")
    create_gif_from_folder('Schlieren-VapDotRho', 'ANIM_Schlieren_VapDotRho.gif')

if PLOT_VELOCITY_VORTICITY_GIF:
    print("\nCreating Velocity/Vorticity GIF...")
    create_gif_from_folder('Velocity-Vorticity', 'ANIM_Velocity_Vorticity.gif')

if PLOT_SCHLIEREN_TEMPERATURE_GIF:
    print("\nCreating Schlieren/Temperature GIF...")
    create_gif_from_folder('Schlieren-Temperature', 'ANIM_Schlieren_Temperature.gif')

# ---- 4Paper additions ----------------------------------------------------
if PLOT_SHAPE_EVOLUTION:
    print("\nGenerating shape-evolution composite (eta=0.5 overlay)...")
    plot_shape_evolution_composite()
    print("  Saved: 04Paper_ShapeEvolution")
    plot_count += 1

if PLOT_SCHLIEREN_MACH_SPLIT:
    print("\nGenerating Schlieren/Mach split (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_mach_split_single(i, i+1, subfolder_schlieren_mach)
        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_SCHLIEREN_MACH_GIF:
    print("\nCreating Schlieren/Mach GIF...")
    create_gif_from_folder('Schlieren-Mach', 'ANIM_Schlieren_Mach.gif')

if PLOT_MASS_CONSERVATION:
    print("\nGenerating mass-conservation diagnostic...")
    plot_mass_conservation()
    print("  Saved: 05Paper_MassConservation")
    plot_count += 1


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"Generated {plot_count} plot types")
print(f"Total frames: {len(analysis_indices)}")
print(f"\nSubfolders:")
print(f"  - Schlieren-Pressure/: {len(os.listdir(subfolder_schlieren))} files")
print(f"  - Schlieren-Velocity/: {len(os.listdir(subfolder_schlieren_velocity))} files")
print(f"  - Schlieren-VapDotRho/: {len(os.listdir(subfolder_schlieren_vapdotrho))} files")
print(f"  - Velocity-Streamline/: {len(os.listdir(subfolder_velocity))} files")
print(f"  - Velocity-Vorticity/: {len(os.listdir(subfolder_velocity_vorticity))} files")
print(f"  - Vorticity/: {len(os.listdir(subfolder_vorticity))} files")
print(f"  - Schlieren-Temperature/: {len(os.listdir(subfolder_schlieren_temperature))} files")
print(f"  - Schlieren-Mach/: {len(os.listdir(subfolder_schlieren_mach))} files")
print(f"  - Deformation-Separate/: {len(os.listdir(subfolder_deformation_separate))} files")
print("\n" + "=" * 70)
