"""
===============================================================================
SHOCK-DROPLET INTERACTION COMPREHENSIVE ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Perform full diagnostic of shock-droplet interaction simulations, including
    numerical schlieren visualization, droplet deformation tracking, shock 
    position analysis, and dimensionless number evolution.

FEATURES:
    - 15+ standalone diagnostic plots (.png and .eps)
    - Split-view visualization (schlieren top, pressure bottom)
    - Droplet deformation metrics (D, AR, centroid, area, volume)
    - Shock tracking and x-t diagrams
    - Dimensionless number evolution (We, Re, Ma)
    - Optional evolution GIFs/animations
    - Modular on/off control for each plot type

INPUTS:
    - AMReX plot files from shock-droplet simulation
    - Physical parameters: rho_air, rho_water, mu, sigma, U_shock, D_droplet
    - Fluid properties: gamma0, gamma1, p0_0, p0_1

OUTPUTS:
    - 15+ diagnostic plots (.png and .eps)
    - Optional evolution GIFs
    - Quantitative analysis data

USAGE:
    1. Set configuration parameters in CONFIGURATION section
    2. Enable/disable plots using PLOT_* flags (1=on, 0=off)
    3. Run script: python ShockDroplet_Analysis.py

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
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
PLOT_SCHLIEREN_PRESSURE_SPLIT = 1      # Split view: schlieren (y>=0) + pressure (y<0) with eta contours
PLOT_SCHLIEREN_PRESSURE_GRID = 1       # 2x3 grid of split views at 6 time steps

# Quantitative analysis plots
PLOT_DEFORMATION_METRICS = 1           # D(t), AR(t), centroid position, surface area, volume
PLOT_ENERGY_EVOLUTION = 1              # Max pressure, KE, surface energy vs time
PLOT_SHOCK_TRACKING = 1                # x-t diagram showing shock positions

# Flow field visualizations
PLOT_VELOCITY_FIELD = 1                # Velocity magnitude + vectors/streamlines at key times
PLOT_VORTICITY_FIELD = 1               # Vorticity contours showing vortex rings

# Profile and cross-section plots
PLOT_CENTERLINE_PROFILES = 1           # Pressure and density along y=0 at key times
PLOT_INTERFACE_PROFILES = 1            # Density profile across interface (normal direction)
PLOT_RADIAL_PROFILES = 1               # Pressure vs radius from droplet center

# Dimensionless analysis
PLOT_DIMENSIONLESS_NUMBERS = 1         # We(t), Re(t), Ma(t) evolution
PLOT_REGIME_MAP = 1                    # We-Oh diagram with current case marked

# Advanced/optional plots
PLOT_PRESSURE_CONTOURS = 1             # Pure pressure contours at key times
PLOT_DENSITY_CONTOURS = 1              # Pure density contours at key times
PLOT_COMPARISON_THEORY = 0             # Overlay theoretical predictions (requires theory data)

# Animation controls
CREATE_ANIMATIONS = 0                  # Set to 1 to generate MP4/GIF animations
SAVE_INDIVIDUAL_FRAMES = 1             # Save PNG for each timestep (for manual animation)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# File paths
#amrex_output_dir = r'../../../bin/tests/FlowShockDroplet/output_ShockDroplet'
amrex_output_dir = r'/mmfs1/home/ttryon/flames/bin/tests/FlowShockDroplet/output_ShockDroplet'

output_folder = './ShockDroplet_Analysis'

# Physical parameters (matching input file)
RHO_AIR = 1.225                        # Air density [kg/m^3]
RHO_WATER = 1000.0                     # Water density [kg/m^3]
RHO_SHOCK = 2.0                        # Post-shock air density [kg/m^3]
MU_AIR = 1.8e-5                        # Air viscosity [Pa*s]
MU_WATER = 1.0e-3                      # Water viscosity [Pa*s]
SIGMA = 72.8e-3                        # Surface tension [N/m]
GAMMA_AIR = 1.4                        # Air specific heat ratio
GAMMA_WATER = 7.15                     # Water specific heat ratio
P0_AIR = 0.0                           # Air Tammann pressure [Pa]
P0_WATER = 3.0e8                       # Water Tammann pressure [Pa]
U_SHOCK = 500.0                        # Initial shock velocity [m/s]
D_DROPLET_INITIAL = 0.01               # Initial droplet diameter [m] (2 x R0)
P_ATM = 1.01325e5                      # Atmospheric pressure [Pa]

# Domain parameters (from input file)
X_MIN = -0.025                         # Domain left edge [m]
X_MAX = 0.075                          # Domain right edge [m]
Y_MIN = -0.025                         # Domain bottom edge [m]
Y_MAX = 0.025                          # Domain top edge [m]
DROPLET_CENTER_X = 0.0                 # Initial droplet center x [m]
DROPLET_CENTER_Y = 0.0                 # Initial droplet center y [m]

# Time selection
USE_ALL_TIMESTEPS = 0                  # 1 = use all available, 0 = use KEY_TIMES
KEY_TIMES = [0, 40e-6, 60e-6, 100e-6, 150e-6, 200e-6]  # Specific times in seconds
NUM_GRID_TIMES = 6                     # Number of evenly spaced times for grid plot

# Interface tracking
ETA_CONTOURS = [0.1, 0.5, 0.9]        # Eta values for interface contours
ETA_THRESHOLD = 0.5                    # Threshold for droplet boundary definition

# Schlieren parameters
SCHLIEREN_BETA = 1.0                   # Sensitivity parameter (1-10, higher = more contrast)
SCHLIEREN_LOG_SCALE = 1                # Use log scaling for schlieren (0=linear, 1=log)
SCHLIEREN_USE_MIXTURE = 1              # Use mixture density (1) or phase densities (0)

# Deformation calculation
DEFORMATION_METHOD = 'bbox'            # 'bbox' = bounding box, 'ellipse' = fitted ellipse

# Shock detection
SHOCK_DETECTION_METHOD = 'pressure'    # 'pressure', 'density', or 'both'
SHOCK_GRADIENT_THRESHOLD = 1e4         # Threshold for shock detection [Pa/m] or [kg/m^4]

# Visualization settings
DPI = 300                              # Resolution for saved figures
COLORMAP_SCHLIEREN = 'gray'
COLORMAP_PRESSURE = 'jet'
COLORMAP_DENSITY = 'viridis'
COLORMAP_VORTICITY = 'RdBu_r'
COLORMAP_VELOCITY = 'plasma'
FIGURE_SIZE_SINGLE = (12, 10)
FIGURE_SIZE_GRID = (18, 12)
FIGURE_SIZE_TIMESERIES = (10, 8)

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_THICK = 2.5
LINE_WIDTH_NORMAL = 2.0
LINE_WIDTH_THIN = 1.5
CONTOUR_LINE_WIDTH = 2.0
PLOT_ZOOM_FACTOR = 2.0                 # How much to zoom in on droplet (1.0 = full domain, 2.0 = half width)
ASPECT_RATIO = 'equal'                 # Force equal aspect ratio


# Output settings
ANIMATION_FPS = 10
GIF_DURATION = 100                     # Duration per frame in milliseconds

# Publication mode
PUBLICATION_MODE = 0                   # Use LaTeX fonts and PDF-optimized settings
SAVE_FORMAT_RASTER = 'png'             # 'png' or 'jpg'
SAVE_FORMAT_VECTOR = 'eps'             # 'eps', 'pdf', or 'svg'

# Create output directory
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

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
    """Find indices of timesteps closest to target times"""
    indices = []
    for target in target_times:
        idx = np.argmin(np.abs(times - target))
        indices.append(idx)
    return indices

def select_evenly_spaced(indices, num_points):
    """Select evenly spaced indices from a list"""
    if len(indices) <= num_points:
        return indices
    return [indices[i] for i in np.linspace(0, len(indices)-1, num_points, dtype=int)]

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compute_schlieren(rho, dx, dy, beta=5.0, log_scale=False):
    """
    Calculate numerical schlieren field from density
    
    Parameters:
        rho: 2D density field
        dx, dy: grid spacing
        beta: sensitivity parameter
        log_scale: use logarithmic scaling
    
    Returns:
        schlieren: 2D schlieren field
    """
    # Compute gradients
    drho_dx = np.gradient(rho, dx, axis=0)
    drho_dy = np.gradient(rho, dy, axis=1)
    
    # Magnitude of gradient
    grad_rho_mag = np.sqrt(drho_dx**2 + drho_dy**2)
    
    if log_scale:
        # Logarithmic schlieren
        schlieren = np.log10(1 + 100*grad_rho_mag)
    else:
        # Exponential schlieren (traditional)
        grad_max = np.max(grad_rho_mag)
        if grad_max > 0:
            schlieren = np.exp(-beta * grad_rho_mag / grad_max)
        else:
            schlieren = np.ones_like(grad_rho_mag)
    
    return schlieren

def extract_interface_contour(eta, x_grid, y_grid, threshold=0.5):
    """
    Extract interface contour coordinates from eta field
    
    Parameters:
        eta: 2D volume fraction field
        x_grid, y_grid: 2D coordinate grids
        threshold: eta value for interface
    
    Returns:
        contours: list of contour coordinate arrays
    """
    import matplotlib.pyplot as plt
    
    # Use matplotlib's contour to extract coordinates
    fig_temp = plt.figure()
    cs = plt.contour(x_grid, y_grid, eta, levels=[threshold])
    plt.close(fig_temp)
    
    contours = []
    
    # cs.allsegs is a list of lists: [level][contour_segment][vertex_coords]
    # Since we only have one level (threshold), access the first level
    if len(cs.allsegs) > 0:
        for contour_path in cs.allsegs[0]:  # Get all paths at level 0
            if len(contour_path) > 0:
                contours.append(contour_path)
    
    return contours

def compute_deformation_params(eta, x_grid, y_grid, threshold=0.5, method='bbox'):
    """
    Calculate droplet deformation parameters
    
    Parameters:
        eta: 2D volume fraction field
        x_grid, y_grid: 2D coordinate grids
        threshold: eta value for droplet boundary
        method: 'bbox' or 'ellipse'
    
    Returns:
        D: deformation parameter
        AR: aspect ratio
        centroid: (x_c, y_c)
        area: surface area
        volume: droplet volume (2D area)
    """
    # Create binary mask for droplet (eta < threshold means water)
    droplet_mask = (eta < threshold)
    
    if not np.any(droplet_mask):
        return 0.0, 1.0, (DROPLET_CENTER_X, DROPLET_CENTER_Y), 0.0, 0.0
    
    # Calculate centroid
    y_indices, x_indices = np.where(droplet_mask)
    x_c = np.mean(x_grid[y_indices, x_indices])
    y_c = np.mean(y_grid[y_indices, x_indices])
    
    # Calculate volume (2D area)
    dx = x_grid[0, 1] - x_grid[0, 0]
    dy = y_grid[1, 0] - y_grid[0, 0]
    volume = np.sum(droplet_mask) * dx * dy
    
    if method == 'bbox':
        # Bounding box method
        x_coords = x_grid[y_indices, x_indices]
        y_coords = y_grid[y_indices, x_indices]
        
        L = np.max(x_coords) - np.min(x_coords)  # Length (x-direction)
        W = np.max(y_coords) - np.min(y_coords)  # Width (y-direction)
    
    elif method == 'ellipse':
        # Fit ellipse (more accurate for deformed droplets)
        x_coords = x_grid[y_indices, x_indices] - x_c
        y_coords = y_grid[y_indices, x_indices] - y_c
        
        # Calculate second moments
        Ixx = np.sum(y_coords**2)
        Iyy = np.sum(x_coords**2)
        Ixy = np.sum(x_coords * y_coords)
        
        # Eigenvalues give major/minor axes
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        lambda1 = (trace + np.sqrt(trace**2 - 4*det)) / 2
        lambda2 = (trace - np.sqrt(trace**2 - 4*det)) / 2
        
        L = 2 * np.sqrt(lambda1 / len(x_coords))
        W = 2 * np.sqrt(lambda2 / len(x_coords))
    
    # Deformation parameter and aspect ratio
    if L + W > 0:
        D = (L - W) / (L + W)
        AR = L / W if W > 0 else 1.0
    else:
        D = 0.0
        AR = 1.0
    
    # Surface area (perimeter in 2D)
    contours = extract_interface_contour(eta, x_grid, y_grid, threshold)
    area = 0.0
    for contour in contours:
        if len(contour) > 1:
            # Calculate perimeter
            dx_contour = np.diff(contour[:, 0])
            dy_contour = np.diff(contour[:, 1])
            area += np.sum(np.sqrt(dx_contour**2 + dy_contour**2))
    
    return D, AR, (x_c, y_c), area, volume

def track_shock_position(field, x_grid, method='pressure', threshold=1e4):
    """
    Identify shock front position from pressure or density field
    
    Parameters:
        field: 2D pressure or density field
        x_grid: 2D x-coordinate grid
        method: 'pressure' or 'density'
        threshold: gradient threshold for shock detection
    
    Returns:
        x_shock: x-position of shock front
    """
    # Take horizontal gradient along centerline
    centerline_idx = field.shape[0] // 2
    field_centerline = field[centerline_idx, :]
    x_centerline = x_grid[centerline_idx, :]
    
    # Compute gradient
    dx = x_centerline[1] - x_centerline[0]
    grad_field = np.gradient(field_centerline, dx)
    
    # Find maximum gradient location (shock front)
    max_grad_idx = np.argmax(np.abs(grad_field))
    
    # Check if gradient exceeds threshold
    if np.abs(grad_field[max_grad_idx]) > threshold:
        x_shock = x_centerline[max_grad_idx]
    else:
        x_shock = None
    
    return x_shock

def compute_dimensionless_numbers(rho_air, u_rel, D, sigma, mu_air):
    """
    Calculate Weber, Reynolds, and Mach numbers
    
    Parameters:
        rho_air: air density
        u_rel: relative velocity
        D: droplet diameter
        sigma: surface tension
        mu_air: air viscosity
    
    Returns:
        We: Weber number
        Re: Reynolds number
        Ma: Mach number
    """
    # Weber number
    We = rho_air * u_rel**2 * D / sigma
    
    # Reynolds number
    Re = rho_air * u_rel * D / mu_air
    
    # Mach number (speed of sound in air at standard conditions)
    a_air = np.sqrt(GAMMA_AIR * P_ATM / rho_air)
    Ma = u_rel / a_air
    
    return We, Re, Ma

def compute_kinetic_energy(rho, vx, vy, dx, dy):
    """
    Calculate total kinetic energy in domain
    
    Parameters:
        rho: 2D density field
        vx, vy: 2D velocity components
        dx, dy: grid spacing
    
    Returns:
        KE: total kinetic energy
    """
    KE = 0.5 * np.sum(rho * (vx**2 + vy**2)) * dx * dy
    return KE

def compute_surface_energy(eta, x_grid, y_grid, sigma, threshold=0.5):
    """
    Calculate surface energy from interface area
    
    Parameters:
        eta: 2D volume fraction field
        x_grid, y_grid: 2D coordinate grids
        sigma: surface tension
        threshold: eta value for interface
    
    Returns:
        SE: surface energy
    """
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
print("SHOCK-DROPLET INTERACTION COMPREHENSIVE ANALYSIS")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and ('plt' in item.lower() or 'cell' in item.lower()):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files")

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
    
    # Store dataset reference for later use
    all_data.append(ds)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Loaded {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)

# ============================================================================
# SELECT TIMESTEPS FOR ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("SELECTING TIMESTEPS FOR ANALYSIS")
print("=" * 70)

if USE_ALL_TIMESTEPS:
    analysis_indices = list(range(len(plot_files)))
    print(f"Using all {len(analysis_indices)} timesteps")
else:
    analysis_indices = find_closest_timesteps(times, KEY_TIMES)
    print(f"Using {len(analysis_indices)} key timesteps:")
    for idx in analysis_indices:
        print(f"  t = {times[idx]:.6e} s")

# Select timesteps for grid plots
grid_indices = select_evenly_spaced(analysis_indices, NUM_GRID_TIMES)
print(f"\nSelected {len(grid_indices)} timesteps for grid plots:")
for i, idx in enumerate(grid_indices):
    print(f"  Panel {i+1}: t = {times[idx]:.6e} s")

# ============================================================================
# EXTRACT AND COMPUTE FIELDS FOR ALL TIMESTEPS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING DERIVED FIELDS")
print("=" * 70)

# Storage for time-series data
deformation_data = []
energy_data = []
shock_positions = []
dimensionless_data = []

# Storage for spatial fields at key times
schlieren_fields = []
pressure_fields = []
density_fields = []
velocity_fields = []
vorticity_fields = []
eta_fields = []
x_grids = []
y_grids = []

for i, idx in enumerate(analysis_indices):
    ds = all_data[idx]
    t = times[idx]
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Determine appropriate resolution
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    resolution = 256  # Can be adjusted
    
    frb = slc.to_frb((domain_width, 'code_length'), resolution, 
                     center=[0.5*(X_MIN+X_MAX), 0.5*(Y_MIN+Y_MAX), 0.0],
                     height=(domain_height, 'code_length'))
    
    # Extract fields
    if SCHLIEREN_USE_MIXTURE:
        rho = np.array(frb['density'])
    else:
        # Compute mixture density from phase densities
        eta = np.array(frb['eta'])
        rho0 = np.array(frb['density0'])
        rho1 = np.array(frb['density1'])
        rho = eta * rho0 + (1 - eta) * rho1
    
    pressure = np.array(frb['pressure'])
    eta_field = np.array(frb['eta'])
    
    # Velocity components
    vx = np.array(frb['velocityx'])
    vy = np.array(frb['velocityy'])
    
    # Vorticity (if available, otherwise compute)
    try:
        vorticity = np.array(frb['vorticity'])
    except:
        # Compute vorticity from velocity
        dx = domain_width / resolution
        dy = domain_height / resolution
        dvx_dy = np.gradient(vx, dy, axis=0)
        dvy_dx = np.gradient(vy, dx, axis=1)
        vorticity = dvy_dx - dvx_dy
    
    # Create coordinate grids
    x_1d = np.linspace(X_MIN, X_MAX, resolution)
    y_1d = np.linspace(Y_MIN, Y_MAX, resolution)
    x_grid, y_grid = np.meshgrid(x_1d, y_1d)
    
    # Compute schlieren
    dx = x_1d[1] - x_1d[0]
    dy = y_1d[1] - y_1d[0]
    schlieren = compute_schlieren(rho, dx, dy, SCHLIEREN_BETA, SCHLIEREN_LOG_SCALE)
    
    # Store spatial fields
    schlieren_fields.append(schlieren)
    pressure_fields.append(pressure)
    density_fields.append(rho)
    velocity_fields.append((vx, vy))
    vorticity_fields.append(vorticity)
    eta_fields.append(eta_field)
    x_grids.append(x_grid)
    y_grids.append(y_grid)
    
    # Compute deformation parameters
    D, AR, centroid, area, volume = compute_deformation_params(
        eta_field, x_grid, y_grid, ETA_THRESHOLD, DEFORMATION_METHOD
    )
    deformation_data.append({
        'D': D, 'AR': AR, 'centroid': centroid, 
        'area': area, 'volume': volume
    })
    
    # Compute energies
    KE = compute_kinetic_energy(rho, vx, vy, dx, dy)
    SE = compute_surface_energy(eta_field, x_grid, y_grid, SIGMA, ETA_THRESHOLD)
    p_max = np.max(pressure)
    energy_data.append({'KE': KE, 'SE': SE, 'p_max': p_max})
    
    # Track shock position
    x_shock = track_shock_position(pressure, x_grid, SHOCK_DETECTION_METHOD, 
                                   SHOCK_GRADIENT_THRESHOLD)
    shock_positions.append(x_shock)
    
    # Compute dimensionless numbers
    u_rel = np.max(np.sqrt(vx**2 + vy**2))  # Maximum velocity in domain
    D_current = 2 * np.sqrt(volume / np.pi)  # Equivalent diameter
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

# Schlieren range
schlieren_min = min([np.min(s) for s in schlieren_fields])
schlieren_max = max([np.max(s) for s in schlieren_fields])
print(f"  Schlieren range: [{schlieren_min:.6e}, {schlieren_max:.6e}]")

# Pressure range
pressure_min = min([np.min(p) for p in pressure_fields])
pressure_max = max([np.max(p) for p in pressure_fields])
print(f"  Pressure range: [{pressure_min:.6e}, {pressure_max:.6e}] Pa")

# Density range
density_min = min([np.min(rho) for rho in density_fields])
density_max = max([np.max(rho) for rho in density_fields])
print(f"  Density range: [{density_min:.6e}, {density_max:.6e}] kg/m^3")

# Velocity magnitude range
v_mag_min = 0.0
v_mag_max = max([np.max(np.sqrt(vx**2 + vy**2)) for vx, vy in velocity_fields])
print(f"  Velocity range: [{v_mag_min:.6e}, {v_mag_max:.6e}] m/s")

# Vorticity range
vort_min = min([np.min(v) for v in vorticity_fields])
vort_max = max([np.max(v) for v in vorticity_fields])
vort_lim = max(abs(vort_min), abs(vort_max))
print(f"  Vorticity range: [{-vort_lim:.6e}, {vort_lim:.6e}] 1/s")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_schlieren_pressure_split_single(idx, save_name):
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig = plt.figure(figsize=FIGURE_SIZE_SINGLE)

    # Zero vertical spacing
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.0)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    # -------------------------------------------------------
    # Load data
    # -------------------------------------------------------
    t = times[analysis_indices[idx]]
    schlieren = schlieren_fields[idx]
    pressure = pressure_fields[idx]
    eta = eta_fields[idx]
    x_grid = x_grids[idx]
    y_grid = y_grids[idx]

    y_vals = y_grid[:, 0]
    x_vals = x_grid[0, :]

    # Exact y = 0 index
    zero_idx = np.argmin(np.abs(y_vals))

    # Clean split
    schlieren_top = schlieren[zero_idx:, :]
    pressure_bottom = pressure[:zero_idx + 1, :]
    eta_top = eta[zero_idx:, :]
    eta_bottom = eta[:zero_idx + 1, :]

    y_top = y_vals[zero_idx:]
    y_bottom = y_vals[:zero_idx + 1]

    # Extents
    extent_top = [x_vals.min(), x_vals.max(), y_top.min(), y_top.max()]
    extent_bot = [x_vals.min(), x_vals.max(), y_bottom.min(), y_bottom.max()]

    # -------------------------------------------------------
    # ZOOM WINDOW
    # -------------------------------------------------------
    centroid = deformation_data[idx]['centroid']
    domain_width = X_MAX - X_MIN
    domain_height = Y_MAX - Y_MIN
    zoom_width = domain_width / PLOT_ZOOM_FACTOR
    zoom_height = domain_height / PLOT_ZOOM_FACTOR

    x_min_zoom = centroid[0] - zoom_width / 2
    x_max_zoom = centroid[0] + zoom_width / 2
    y_min_zoom = centroid[1] - zoom_height / 2
    y_max_zoom = centroid[1] + zoom_height / 2

    # =======================================================
    # TOP - SCHLIEREN
    # =======================================================
    im1 = ax_top.imshow(
        schlieren_top,
        origin='lower',
        extent=extent_top,
        cmap=COLORMAP_SCHLIEREN + '_r',  # CHANGED: Added '_r' to reverse colormap
        vmin=schlieren_min,
        vmax=schlieren_max,
        interpolation='bilinear',  # CHANGED: Better interpolation for smoother gradients
        aspect='auto'
    )

    for eta_val in ETA_CONTOURS:
        ax_top.contour(
            x_vals,
            y_top,
            eta_top,
            levels=[eta_val],
            colors='red',
            linewidths=CONTOUR_LINE_WIDTH,
            linestyles='--' if eta_val != 0.5 else '-'
        )

    ax_top.set_xlim(x_min_zoom, x_max_zoom)
    ax_top.set_ylim(0, y_max_zoom)
    ax_top.set_aspect('equal', adjustable='box')
    ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_top.set_title(f't = {t:.6e} s',
                     fontsize=FONT_SIZE_TITLE,
                     fontweight='bold')
    ax_top.tick_params(labelbottom=False)

    # Remove bottom spine to avoid double line at y=0
    ax_top.spines['bottom'].set_visible(False)

    cax1 = inset_axes(ax_top, width="3%", height="80%", loc='right')
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('Numerical Schlieren',
                    fontsize=FONT_SIZE_LABEL)

    # =======================================================
    # BOTTOM - PRESSURE
    # =======================================================
    im2 = ax_bot.imshow(
        pressure_bottom,
        origin='lower',
        extent=extent_bot,
        cmap=COLORMAP_PRESSURE,
        vmin=pressure_min,
        vmax=pressure_max,
        interpolation='bilinear',  # CHANGED: Better interpolation
        aspect='auto'
    )

    for eta_val in ETA_CONTOURS:
        ax_bot.contour(
            x_vals,
            y_bottom,
            eta_bottom,
            levels=[eta_val],
            colors='white',
            linewidths=CONTOUR_LINE_WIDTH,
            linestyles='--' if eta_val != 0.5 else '-'
        )

    ax_bot.set_xlim(x_min_zoom, x_max_zoom)
    ax_bot.set_ylim(y_min_zoom, 0)
    ax_bot.set_aspect('equal', adjustable='box')
    ax_bot.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_bot.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)

    # Remove top spine to avoid double line at y=0
    ax_bot.spines['top'].set_visible(False)

    cax2 = inset_axes(ax_bot, width="3%", height="80%", loc='right')
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('Pressure (Pa)',
                    fontsize=FONT_SIZE_LABEL)

    # -------------------------------------------------------
    # Make it visually seamless
    # -------------------------------------------------------
    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        top=0.95,
        bottom=0.08,
        hspace=0.0
    )

    # Remove vertical gap completely
    ax_top.set_position([
        ax_top.get_position().x0,
        ax_bot.get_position().y1,
        ax_top.get_position().width,
        ax_top.get_position().height
    ])

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------
    plt.savefig(os.path.join(output_folder,
                f'{save_name}.{SAVE_FORMAT_RASTER}'),
                dpi=DPI)
    plt.savefig(os.path.join(output_folder,
                f'{save_name}.{SAVE_FORMAT_VECTOR}'))
    plt.close()


def plot_schlieren_pressure_grid():
    """Plot 2: 2x3 grid of split views at 6 timesteps - NO GAP, CENTERED ON DROPLET"""
    
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=FIGURE_SIZE_GRID)
    
    # Create GridSpec: 6 rows (2 per timestep), 3 columns
    gs = GridSpec(NUM_GRID_TIMES * 2, 3, figure=fig, hspace=0.15, wspace=0.3, 
                  height_ratios=[1, 1] * NUM_GRID_TIMES)
    
    for panel_idx, data_idx in enumerate(grid_indices):
        # Find index in analysis_indices
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        schlieren = schlieren_fields[idx]
        pressure = pressure_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid for centering
        centroid = deformation_data[idx]['centroid']
        
        # Calculate zoom window
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        # Determine column (0, 1, or 2)
        col = panel_idx % 3
        # Determine row pair (0-1, 2-3, 4-5, etc.)
        row_pair = (panel_idx // 3) * 2
        
        # Create subplots with shared x-axis
        ax_top = fig.add_subplot(gs[row_pair, col])
        ax_bottom = fig.add_subplot(gs[row_pair + 1, col], sharex=ax_top)
        
        # Top: Schlieren
        mask_top = y_grid >= 0
        schlieren_top = np.where(mask_top, schlieren, np.nan)
        im1 = ax_top.contourf(x_grid, y_grid, schlieren_top, levels=30, 
                             cmap=COLORMAP_SCHLIEREN)
        for eta_val in ETA_CONTOURS:
            ax_top.contour(x_grid, y_grid, eta, levels=[eta_val], 
                          colors='red', linewidths=1.5)
        
        ax_top.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        ax_top.set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE-2)
        ax_top.set_xlim([x_min_zoom, x_max_zoom])
        ax_top.set_ylim([max(0, y_min_zoom), y_max_zoom])
        ax_top.set_aspect(ASPECT_RATIO)
        ax_top.tick_params(labelsize=FONT_SIZE_TICK-2, labelbottom=False)
        
        # Bottom: Pressure
        mask_bottom = y_grid < 0
        pressure_bottom = np.where(mask_bottom, pressure, np.nan)
        im2 = ax_bottom.contourf(x_grid, y_grid, pressure_bottom, levels=30, 
                                cmap=COLORMAP_PRESSURE)
        for eta_val in ETA_CONTOURS:
            ax_bottom.contour(x_grid, y_grid, eta, levels=[eta_val], 
                            colors='white', linewidths=1.5)
        
        ax_bottom.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        ax_bottom.set_xlim([x_min_zoom, x_max_zoom])
        ax_bottom.set_ylim([y_min_zoom, min(0, y_max_zoom)])
        ax_bottom.set_aspect(ASPECT_RATIO)
        ax_bottom.tick_params(labelsize=FONT_SIZE_TICK-2)
        
        # Only show x-label on bottom row
        if panel_idx >= 3:  # Bottom row of panels
            ax_bottom.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
    
    fig.suptitle('Shock-Droplet Interaction: Schlieren (top) & Pressure (bottom)', 
                fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'02_Schlieren_Pressure_Grid.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'02_Schlieren_Pressure_Grid.{SAVE_FORMAT_VECTOR}'))
    plt.close()


def plot_deformation_metrics():
    """Plot 3: Deformation parameters vs time"""
    
    t_analysis = times[analysis_indices]
    D_vals = [d['D'] for d in deformation_data]
    AR_vals = [d['AR'] for d in deformation_data]
    centroid_x = [d['centroid'][0] for d in deformation_data]
    centroid_y = [d['centroid'][1] for d in deformation_data]
    area_vals = [d['area'] for d in deformation_data]
    volume_vals = [d['volume'] for d in deformation_data]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Deformation parameter
    axes[0, 0].plot(t_analysis * 1e6, D_vals, 'b-', linewidth=LINE_WIDTH_NORMAL)
    axes[0, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_ylabel('Deformation D', fontsize=FONT_SIZE_LABEL)
    axes[0, 0].set_title('Deformation Parameter', fontsize=FONT_SIZE_TITLE)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Aspect ratio
    axes[0, 1].plot(t_analysis * 1e6, AR_vals, 'r-', linewidth=LINE_WIDTH_NORMAL)
    axes[0, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_ylabel('Aspect Ratio', fontsize=FONT_SIZE_LABEL)
    axes[0, 1].set_title('Aspect Ratio (L/W)', fontsize=FONT_SIZE_TITLE)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Centroid X
    axes[1, 0].plot(t_analysis * 1e6, np.array(centroid_x) * 1e3, 'g-', 
                   linewidth=LINE_WIDTH_NORMAL)
    axes[1, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_ylabel('Centroid X (mm)', fontsize=FONT_SIZE_LABEL)
    axes[1, 0].set_title('Centroid X Position', fontsize=FONT_SIZE_TITLE)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Centroid Y
    axes[1, 1].plot(t_analysis * 1e6, np.array(centroid_y) * 1e3, 'm-', 
                   linewidth=LINE_WIDTH_NORMAL)
    axes[1, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_ylabel('Centroid Y (mm)', fontsize=FONT_SIZE_LABEL)
    axes[1, 1].set_title('Centroid Y Position', fontsize=FONT_SIZE_TITLE)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Surface area
    axes[2, 0].plot(t_analysis * 1e6, np.array(area_vals) * 1e3, 'c-', 
                   linewidth=LINE_WIDTH_NORMAL)
    axes[2, 0].axhline(y=np.pi * D_DROPLET_INITIAL * 1e3, color='k', 
                      linestyle='--', label='Initial')
    axes[2, 0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_ylabel('Surface Area (mm)', fontsize=FONT_SIZE_LABEL)
    axes[2, 0].set_title('Interface Perimeter', fontsize=FONT_SIZE_TITLE)
    axes[2, 0].legend(fontsize=FONT_SIZE_LEGEND)
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Volume
    V0 = np.pi * (D_DROPLET_INITIAL/2)**2
    axes[2, 1].plot(t_analysis * 1e6, np.array(volume_vals) / V0, 'y-', 
                   linewidth=LINE_WIDTH_NORMAL)
    axes[2, 1].axhline(y=1.0, color='k', linestyle='--', label='Initial')
    axes[2, 1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_ylabel('Volume / V0', fontsize=FONT_SIZE_LABEL)
    axes[2, 1].set_title('Normalized Volume', fontsize=FONT_SIZE_TITLE)
    axes[2, 1].legend(fontsize=FONT_SIZE_LEGEND)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].tick_params(labelsize=FONT_SIZE_TICK)
    
    fig.suptitle('Droplet Deformation Metrics', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'03_Deformation_Metrics.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'03_Deformation_Metrics.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_energy_evolution():
    """Plot 4: Energy components vs time"""
    
    t_analysis = times[analysis_indices]
    KE_vals = [e['KE'] for e in energy_data]
    SE_vals = [e['SE'] for e in energy_data]
    p_max_vals = [e['p_max'] for e in energy_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Kinetic energy
    axes[0].plot(t_analysis * 1e6, KE_vals, 'b-', linewidth=LINE_WIDTH_NORMAL)
    axes[0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel('Kinetic Energy (J)', fontsize=FONT_SIZE_LABEL)
    axes[0].set_title('Total Kinetic Energy', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Surface energy
    axes[1].plot(t_analysis * 1e6, SE_vals, 'r-', linewidth=LINE_WIDTH_NORMAL)
    axes[1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel('Surface Energy (J)', fontsize=FONT_SIZE_LABEL)
    axes[1].set_title('Surface Energy', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Maximum pressure
    axes[2].plot(t_analysis * 1e6, np.array(p_max_vals) / 1e5, 'g-', 
                linewidth=LINE_WIDTH_NORMAL)
    axes[2].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Max Pressure (bar)', fontsize=FONT_SIZE_LABEL)
    axes[2].set_title('Maximum Pressure', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=FONT_SIZE_TICK)
    
    fig.suptitle('Energy Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'04_Energy_Evolution.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'04_Energy_Evolution.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_shock_tracking():
    """Plot 5: x-t diagram showing shock position"""
    
    t_analysis = times[analysis_indices]
    x_shock_vals = [x if x is not None else np.nan for x in shock_positions]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TIMESERIES)
    
    # Plot shock trajectory
    ax.plot(t_analysis * 1e6, np.array(x_shock_vals) * 1e3, 'ro-', 
           linewidth=LINE_WIDTH_NORMAL, markersize=6, label='Shock Front')
    
    # Add droplet position
    ax.axhline(y=DROPLET_CENTER_X * 1e3, color='b', linestyle='--', 
              linewidth=LINE_WIDTH_THICK, label='Droplet Center')
    
    # Theoretical shock speed line
    t_impact = (DROPLET_CENTER_X - X_MIN - 0.005) / U_SHOCK  # Approximate
    ax.plot([0, t_impact * 1e6], 
           [(X_MIN + 0.005) * 1e3, DROPLET_CENTER_X * 1e3], 
           'k--', linewidth=LINE_WIDTH_THIN, label=f'Theory (u={U_SHOCK} m/s)')
    
    ax.set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Shock Position X (mm)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Shock Wave Tracking', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'05_Shock_Tracking.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'05_Shock_Tracking.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_velocity_field():
    """Plot 6: Velocity magnitude with vectors at key times - EQUAL ASPECT, CENTERED"""
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()
    
    for panel_idx, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        vx, vy = velocity_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid
        centroid = deformation_data[idx]['centroid']
        
        # Calculate zoom window
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        v_mag = np.sqrt(vx**2 + vy**2)
        
        # Plot velocity magnitude
        im = axes[panel_idx].contourf(x_grid, y_grid, v_mag, levels=30, 
                                     cmap=COLORMAP_VELOCITY)
        
        # Add velocity vectors (subsample for clarity)
        skip = 8
        axes[panel_idx].quiver(x_grid[::skip, ::skip], y_grid[::skip, ::skip],
                              vx[::skip, ::skip], vy[::skip, ::skip],
                              color='white', alpha=0.7, scale=5000)
        
        # Add eta contour
        axes[panel_idx].contour(x_grid, y_grid, eta, levels=[0.5], 
                               colors='black', linewidths=2)
        
        axes[panel_idx].set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE-2)
        axes[panel_idx].set_xlim([x_min_zoom, x_max_zoom])
        axes[panel_idx].set_ylim([y_min_zoom, y_max_zoom])
        axes[panel_idx].set_aspect(ASPECT_RATIO)
        axes[panel_idx].tick_params(labelsize=FONT_SIZE_TICK-2)
        
        cbar = plt.colorbar(im, ax=axes[panel_idx])
        cbar.set_label('|V| (m/s)', fontsize=FONT_SIZE_LABEL-4)
    
    fig.suptitle('Velocity Field Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'06_Velocity_Field.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'06_Velocity_Field.{SAVE_FORMAT_VECTOR}'))
    plt.close()


def plot_vorticity_field():
    """Plot 7: Vorticity contours at key times - EQUAL ASPECT, CENTERED"""
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()
    
    # Find global vorticity range for consistent colormap
    vort_min = min([np.min(v) for v in vorticity_fields])
    vort_max = max([np.max(v) for v in vorticity_fields])
    vort_lim = max(abs(vort_min), abs(vort_max))
    
    for panel_idx, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        vorticity = vorticity_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid
        centroid = deformation_data[idx]['centroid']
        
        # Calculate zoom window
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        # Plot vorticity
        im = axes[panel_idx].contourf(x_grid, y_grid, vorticity, levels=30, 
                                     cmap=COLORMAP_VORTICITY, 
                                     vmin=-vort_lim, vmax=vort_lim)
        
        # Add eta contour
        axes[panel_idx].contour(x_grid, y_grid, eta, levels=[0.5], 
                               colors='black', linewidths=2)
        
        axes[panel_idx].set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE-2)
        axes[panel_idx].set_xlim([x_min_zoom, x_max_zoom])
        axes[panel_idx].set_ylim([y_min_zoom, y_max_zoom])
        axes[panel_idx].set_aspect(ASPECT_RATIO)
        axes[panel_idx].tick_params(labelsize=FONT_SIZE_TICK-2)
        
        cbar = plt.colorbar(im, ax=axes[panel_idx])
        cbar.set_label('Vorticity (1/s)', fontsize=FONT_SIZE_LABEL-4)
    
    fig.suptitle('Vorticity Field Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'07_Vorticity_Field.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'07_Vorticity_Field.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_centerline_profiles():
    """Plot 8: Pressure and density along y=0 at key times"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(grid_indices)))
    
    for i, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        pressure = pressure_fields[idx]
        density = density_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Extract centerline (y=0)
        centerline_idx = np.argmin(np.abs(y_grid[:, 0]))
        x_centerline = x_grid[centerline_idx, :]
        p_centerline = pressure[centerline_idx, :]
        rho_centerline = density[centerline_idx, :]
        
        # Plot pressure
        ax1.plot(x_centerline * 1e3, p_centerline / 1e5, 
                linewidth=LINE_WIDTH_NORMAL, color=colors[i], 
                label=f't = {t*1e6:.1f} us')
        
        # Plot density
        ax2.plot(x_centerline * 1e3, rho_centerline, 
                linewidth=LINE_WIDTH_NORMAL, color=colors[i], 
                label=f't = {t*1e6:.1f} us')
    
    ax1.set_xlabel('X (mm)', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Pressure (bar)', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('Centerline Pressure', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax1.legend(fontsize=FONT_SIZE_LEGEND-2, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    
    ax2.set_xlabel('X (mm)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Centerline Density', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.legend(fontsize=FONT_SIZE_LEGEND-2, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    
    fig.suptitle('Centerline Profiles (y = 0)', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'08_Centerline_Profiles.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'08_Centerline_Profiles.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_interface_profiles():
    """Plot 9: Density profile across interface"""
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TIMESERIES)
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(grid_indices)))
    
    for i, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        density = density_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Extract vertical profile through droplet center
        centroid = deformation_data[idx]['centroid']
        x_center_idx = np.argmin(np.abs(x_grid[0, :] - centroid[0]))
        
        y_profile = y_grid[:, x_center_idx]
        rho_profile = density[:, x_center_idx]
        eta_profile = eta[:, x_center_idx]
        
        # Plot density vs eta
        ax.plot(eta_profile, rho_profile, linewidth=LINE_WIDTH_NORMAL, 
               color=colors[i], label=f't = {t*1e6:.1f} us')
    
    ax.axvline(x=0.5, color='k', linestyle='--', linewidth=LINE_WIDTH_THIN, 
              label='Interface (eta=0.5)')
    ax.set_xlabel('Volume Fraction eta', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Density Profile Across Interface', fontsize=FONT_SIZE_TITLE, 
                fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'09_Interface_Profiles.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'09_Interface_Profiles.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_radial_profiles():
    """Plot 10: Pressure vs radius from droplet center"""
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TIMESERIES)
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(grid_indices)))
    
    for i, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        pressure = pressure_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid
        centroid = deformation_data[idx]['centroid']
        
        # Compute radius from centroid
        r_grid = np.sqrt((x_grid - centroid[0])**2 + (y_grid - centroid[1])**2)
        
        # Flatten and sort by radius
        r_flat = r_grid.flatten()
        p_flat = pressure.flatten()
        sort_idx = np.argsort(r_flat)
        
        # Bin average for smoother profile
        r_sorted = r_flat[sort_idx]
        p_sorted = p_flat[sort_idx]
        
        # Plot (subsample for clarity)
        subsample = slice(0, len(r_sorted), len(r_sorted)//200)
        ax.plot(r_sorted[subsample] * 1e3, p_sorted[subsample] / 1e5, 
               linewidth=LINE_WIDTH_NORMAL, color=colors[i], alpha=0.7,
               label=f't = {t*1e6:.1f} us')
    
    ax.axvline(x=D_DROPLET_INITIAL/2 * 1e3, color='k', linestyle='--', 
              linewidth=LINE_WIDTH_THIN, label='Initial Radius')
    ax.set_xlabel('Radius from Centroid (mm)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Pressure (bar)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Radial Pressure Profile', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND-2, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_xlim([0, 15])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'10_Radial_Profiles.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'10_Radial_Profiles.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_dimensionless_numbers():
    """Plot 11: We, Re, Ma evolution"""
    
    t_analysis = times[analysis_indices]
    We_vals = [d['We'] for d in dimensionless_data]
    Re_vals = [d['Re'] for d in dimensionless_data]
    Ma_vals = [d['Ma'] for d in dimensionless_data]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Weber number
    axes[0].plot(t_analysis * 1e6, We_vals, 'b-', linewidth=LINE_WIDTH_THICK)
    axes[0].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[0].set_ylabel('Weber Number', fontsize=FONT_SIZE_LABEL)
    axes[0].set_title('We = rho u^2 D / sigma', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Reynolds number
    axes[1].plot(t_analysis * 1e6, Re_vals, 'r-', linewidth=LINE_WIDTH_THICK)
    axes[1].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[1].set_ylabel('Reynolds Number', fontsize=FONT_SIZE_LABEL)
    axes[1].set_title('Re = rho u D / mu', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(labelsize=FONT_SIZE_TICK)
    
    # Mach number
    axes[2].plot(t_analysis * 1e6, Ma_vals, 'g-', linewidth=LINE_WIDTH_THICK)
    axes[2].set_xlabel('Time (us)', fontsize=FONT_SIZE_LABEL)
    axes[2].set_ylabel('Mach Number', fontsize=FONT_SIZE_LABEL)
    axes[2].set_title('Ma = u / a', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(labelsize=FONT_SIZE_TICK)
    
    fig.suptitle('Dimensionless Number Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'11_Dimensionless_Numbers.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'11_Dimensionless_Numbers.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_regime_map():
    """Plot 12: We-Oh regime diagram"""
    
    # Calculate Ohnesorge number
    Oh = MU_WATER / np.sqrt(RHO_WATER * SIGMA * D_DROPLET_INITIAL)
    
    # Get Weber number range from simulation
    We_vals = [d['We'] for d in dimensionless_data]
    We_mean = np.mean(We_vals)
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_TIMESERIES)
    
    # Plot regime boundaries (approximate from literature)
    We_range = np.logspace(-1, 4, 100)
    
    # Vibrational breakup boundary (We ~ 12)
    ax.axvline(x=12, color='gray', linestyle='--', linewidth=LINE_WIDTH_THIN, 
              alpha=0.5, label='Vibrational')
    
    # Bag breakup boundary (We ~ 12-50)
    ax.axvspan(12, 50, alpha=0.2, color='blue', label='Bag Breakup')
    
    # Multimode breakup (We ~ 50-100)
    ax.axvspan(50, 100, alpha=0.2, color='green', label='Multimode')
    
    # Shear breakup (We > 100)
    ax.axvspan(100, 10000, alpha=0.2, color='red', label='Shear Breakup')
    
    # Plot current case
    ax.plot(We_mean, Oh, 'ko', markersize=15, markerfacecolor='yellow', 
           markeredgewidth=2, label=f'Current Case\n(We={We_mean:.1f}, Oh={Oh:.4f})', 
           zorder=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Weber Number (We)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Ohnesorge Number (Oh)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Droplet Breakup Regime Map', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.set_xlim([0.1, 10000])
    ax.set_ylim([0.001, 10])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'12_Regime_Map.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'12_Regime_Map.{SAVE_FORMAT_VECTOR}'))
    plt.close()

def plot_pressure_contours():
    """Plot 13: Pure pressure contours at key times - EQUAL ASPECT, CENTERED"""
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()
    
    # Find global pressure range
    p_min = min([np.min(p) for p in pressure_fields])
    p_max = max([np.max(p) for p in pressure_fields])
    
    for panel_idx, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        pressure = pressure_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid
        centroid = deformation_data[idx]['centroid']
        
        # Calculate zoom window
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        # Plot pressure
        im = axes[panel_idx].contourf(x_grid, y_grid, pressure / 1e5, levels=30, 
                                     cmap=COLORMAP_PRESSURE, vmin=p_min/1e5, vmax=p_max/1e5)
        
        # Add eta contour
        axes[panel_idx].contour(x_grid, y_grid, eta, levels=[0.5], 
                               colors='white', linewidths=2)
        
        axes[panel_idx].set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE-2)
        axes[panel_idx].set_xlim([x_min_zoom, x_max_zoom])
        axes[panel_idx].set_ylim([y_min_zoom, y_max_zoom])
        axes[panel_idx].set_aspect(ASPECT_RATIO)
        axes[panel_idx].tick_params(labelsize=FONT_SIZE_TICK-2)
        
        cbar = plt.colorbar(im, ax=axes[panel_idx])
        cbar.set_label('P (bar)', fontsize=FONT_SIZE_LABEL-4)
    
    fig.suptitle('Pressure Field Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'13_Pressure_Contours.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'13_Pressure_Contours.{SAVE_FORMAT_VECTOR}'))
    plt.close()


def plot_density_contours():
    """Plot 14: Pure density contours at key times - EQUAL ASPECT, CENTERED"""
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_GRID)
    axes = axes.flatten()
    
    # Find global density range
    rho_min = min([np.min(rho) for rho in density_fields])
    rho_max = max([np.max(rho) for rho in density_fields])
    
    for panel_idx, data_idx in enumerate(grid_indices):
        idx = analysis_indices.index(data_idx)
        
        t = times[data_idx]
        density = density_fields[idx]
        eta = eta_fields[idx]
        x_grid = x_grids[idx]
        y_grid = y_grids[idx]
        
        # Get droplet centroid
        centroid = deformation_data[idx]['centroid']
        
        # Calculate zoom window
        domain_width = X_MAX - X_MIN
        domain_height = Y_MAX - Y_MIN
        zoom_width = domain_width / PLOT_ZOOM_FACTOR
        zoom_height = domain_height / PLOT_ZOOM_FACTOR
        
        x_min_zoom = centroid[0] - zoom_width / 2
        x_max_zoom = centroid[0] + zoom_width / 2
        y_min_zoom = centroid[1] - zoom_height / 2
        y_max_zoom = centroid[1] + zoom_height / 2
        
        # Plot density
        im = axes[panel_idx].contourf(x_grid, y_grid, density, levels=30, 
                                     cmap=COLORMAP_DENSITY, vmin=rho_min, vmax=rho_max)
        
        # Add eta contour
        axes[panel_idx].contour(x_grid, y_grid, eta, levels=[0.5], 
                               colors='red', linewidths=2)
        
        axes[panel_idx].set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL-2)
        axes[panel_idx].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE-2)
        axes[panel_idx].set_xlim([x_min_zoom, x_max_zoom])
        axes[panel_idx].set_ylim([y_min_zoom, y_max_zoom])
        axes[panel_idx].set_aspect(ASPECT_RATIO)
        axes[panel_idx].tick_params(labelsize=FONT_SIZE_TICK-2)
        
        cbar = plt.colorbar(im, ax=axes[panel_idx])
        cbar.set_label('rho (kg/m^3)', fontsize=FONT_SIZE_LABEL-4)
    
    fig.suptitle('Density Field Evolution', fontsize=FONT_SIZE_TITLE + 2, 
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(os.path.join(output_folder, f'14_Density_Contours.{SAVE_FORMAT_RASTER}'), dpi=DPI)
    plt.savefig(os.path.join(output_folder, f'14_Density_Contours.{SAVE_FORMAT_VECTOR}'))
    plt.close()

# ============================================================================
# EXECUTE PLOTS BASED ON FLAGS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

plot_count = 0

if PLOT_SCHLIEREN_PRESSURE_SPLIT and SAVE_INDIVIDUAL_FRAMES:
    print("\nGenerating Plot 1: Schlieren/Pressure Split Views (individual frames)...")
    for i, idx in enumerate(analysis_indices):
        plot_schlieren_pressure_split_single(i, f'01_Schlieren_Pressure_Split_t{i:03d}')
        if (i + 1) % 5 == 0:
            print(f"  Saved {i + 1}/{len(analysis_indices)} frames")
    plot_count += 1

if PLOT_SCHLIEREN_PRESSURE_GRID:
    print("\nGenerating Plot 2: Schlieren/Pressure Grid...")
    plot_schlieren_pressure_grid()
    print("  Saved: 02_Schlieren_Pressure_Grid")
    plot_count += 1

if PLOT_DEFORMATION_METRICS:
    print("\nGenerating Plot 3: Deformation Metrics...")
    plot_deformation_metrics()
    print("  Saved: 03_Deformation_Metrics")
    plot_count += 1

if PLOT_ENERGY_EVOLUTION:
    print("\nGenerating Plot 4: Energy Evolution...")
    plot_energy_evolution()
    print("  Saved: 04_Energy_Evolution")
    plot_count += 1

if PLOT_SHOCK_TRACKING:
    print("\nGenerating Plot 5: Shock Tracking...")
    plot_shock_tracking()
    print("  Saved: 05_Shock_Tracking")
    plot_count += 1

if PLOT_VELOCITY_FIELD:
    print("\nGenerating Plot 6: Velocity Field...")
    plot_velocity_field()
    print("  Saved: 06_Velocity_Field")
    plot_count += 1

if PLOT_VORTICITY_FIELD:
    print("\nGenerating Plot 7: Vorticity Field...")
    plot_vorticity_field()
    print("  Saved: 07_Vorticity_Field")
    plot_count += 1

if PLOT_CENTERLINE_PROFILES:
    print("\nGenerating Plot 8: Centerline Profiles...")
    plot_centerline_profiles()
    print("  Saved: 08_Centerline_Profiles")
    plot_count += 1

if PLOT_INTERFACE_PROFILES:
    print("\nGenerating Plot 9: Interface Profiles...")
    plot_interface_profiles()
    print("  Saved: 09_Interface_Profiles")
    plot_count += 1

if PLOT_RADIAL_PROFILES:
    print("\nGenerating Plot 10: Radial Profiles...")
    plot_radial_profiles()
    print("  Saved: 10_Radial_Profiles")
    plot_count += 1

if PLOT_DIMENSIONLESS_NUMBERS:
    print("\nGenerating Plot 11: Dimensionless Numbers...")
    plot_dimensionless_numbers()
    print("  Saved: 11_Dimensionless_Numbers")
    plot_count += 1

if PLOT_REGIME_MAP:
    print("\nGenerating Plot 12: Regime Map...")
    plot_regime_map()
    print("  Saved: 12_Regime_Map")
    plot_count += 1

if PLOT_PRESSURE_CONTOURS:
    print("\nGenerating Plot 13: Pressure Contours...")
    plot_pressure_contours()
    print("  Saved: 13_Pressure_Contours")
    plot_count += 1

if PLOT_DENSITY_CONTOURS:
    print("\nGenerating Plot 14: Density Contours...")
    plot_density_contours()
    print("  Saved: 14_Density_Contours")
    plot_count += 1

if PLOT_COMPARISON_THEORY:
    print("\nPlot 15: Comparison with Theory (not implemented - requires theory data)")
    print("  Skipped")

# ============================================================================
# ANIMATION GENERATION (OPTIONAL)
# ============================================================================

if CREATE_ANIMATIONS:
    print("\n" + "=" * 70)
    print("GENERATING ANIMATIONS")
    print("=" * 70)
    
    # Animation 1: Schlieren/Pressure Split View
    if PLOT_SCHLIEREN_PRESSURE_SPLIT:
        print("\nCreating Animation 1: Schlieren/Pressure Split Evolution...")
        frames = []
        for i in range(len(analysis_indices)):
            img_path = os.path.join(output_folder, f'01_Schlieren_Pressure_Split_t{i:03d}.{SAVE_FORMAT_RASTER}')
            if os.path.exists(img_path):
                frames.append(Image.open(img_path))
        
        if frames:
            gif_path = os.path.join(output_folder, 'ANIM_01_Schlieren_Pressure.gif')
            frames[0].save(gif_path, save_all=True, append_images=frames[1:], 
                          duration=GIF_DURATION, loop=0)
            print(f"  Saved: ANIM_01_Schlieren_Pressure.gif")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nGenerated {plot_count} plot types")
print(f"Total files: {len(os.listdir(output_folder))}")

print("\n" + "=" * 70)
