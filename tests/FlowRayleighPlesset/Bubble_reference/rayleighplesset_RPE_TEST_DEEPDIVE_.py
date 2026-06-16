# -*- coding: utf-8 -*-
"""
===============================================================================
RAYLEIGH-PLESSET DEEP DIVE ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Comprehensive analysis of Rayleigh-Plesset bubble simulation including:
    - Bubble radius tracking and comparison with analytical RPE
    - Surface tension force (Fsvx) analysis vs sharp interface limit
    - Surface curvature (kappaAvg) analysis vs analytical curvature
    - Error analysis for all quantities

FEATURES:
    - Extracts bubble radius from eta field (eta=0.5 contour)
    - Integrates Fsvx along x-axis (x>0) and normalizes by radius: (1/R)*int(Fsvx*dx)
    - Plots Fsvx profile along x-axis at y=0
    - Analyzes kappaAvg with outlier filtering
    - Generates comprehensive plots vs time
    - Log-scale error plots

INPUTS:
    - AMReX plot files from Rayleigh-Plesset simulation
    - Physical parameters matching RPE analytical solution

OUTPUTS:
    - Radius comparison and error plots
    - Fsvx profile plot and analysis
    - Fsvx analysis and error plots vs time
    - Curvature analysis and error plots vs time

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH YOUR SIMULATION INPUT FILE)
rho_L = 1000.0              # Liquid density [kg/m^3]
mu_L = 0.1                # Dynamic viscosity [Pa*s]
S = 72.8                  # Surface tension [N/m]
p_v = 0.0                 # Vapor pressure [Pa]
gamma = 1.4               # Adiabatic index

# Initial conditions
p_inf = 1.0e5             # External pressure [Pa]
p_B0 = 2.0e5             # Initial bubble pressure [Pa]
R0 = 0.01                 # Initial radius [m] = 20 mm
R_dot0 = 0.0              # Initial velocity [m/s]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.5         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'../../../bin/tests/RayleighPlesset/output_RayleighPlesset'  # Directory containing AMReX plot files

# Analytical RPE solver parameters
r_inf = 5.0 * R0          # Far-field boundary for RPE
t_span_rpe = (0, 0.05)    # Time span for RPE integration [s]
n_points_rpe = 10000      # Number of time points for RPE

# Integration parameters for Fsvx
Y_INTEGRATION_WIDTH = 0.005  # Width of strip to integrate over [m]

# Curvature filtering parameters
KAPPA_FILTER_PERCENT = 50.0  # Filter kappa values outside +/- this % of analytical [%]

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './RPE_DeepDive_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL RPE SOLUTION (2D CYLINDRICAL)
# ============================================================================

def rp_equation_2d_chen(t, y):
    """
    Chen's 2D Cylindrical RPE
    """
    R, R_dot = y
    
    # Safety checks
    if R <= 1e-12 or R >= 0.9 * r_inf:
        return [0, 0]
    
    # Geometric logarithmic factor
    ln_factor = np.log(r_inf / R)
    if ln_factor < 1e-6:
        ln_factor = 1e-6
    
    # Gas pressure (2D exponent)
    p_B = p_v + (p_B0 - p_v) * (R0 / R)**(2 * gamma)
    
    # External pressure (no acoustic forcing)
    p_ext = p_inf
    
    # RHS pressure term
    pressure_term = (
        p_B
        - p_ext
        - 2 * mu_L * R_dot / R
        - S / R
    ) / rho_L
    
    # Correct cylindrical inertia structure
    numerator = (
        pressure_term
        - R_dot**2 * (0.5 - ln_factor)
    )
    
    denominator = R * ln_factor
    
    R_ddot = numerator / denominator
    
    return [R_dot, R_ddot]

print("=" * 70)
print("RAYLEIGH-PLESSET DEEP DIVE ANALYSIS")
print("=" * 70)

# Solve analytical RPE
print("\nSolving analytical 2D cylindrical RPE...")
y0 = [R0, R_dot0]
t_eval_rpe = np.linspace(*t_span_rpe, n_points_rpe)

sol = solve_ivp(rp_equation_2d_chen, t_span_rpe, y0, t_eval=t_eval_rpe,
                method='RK45', rtol=1e-9, atol=1e-11)

if not sol.success:
    print(f"WARNING: RPE integration failed - {sol.message}")
    exit(1)

radius_analytical = sol.y[0]
time_analytical = sol.t

print(f"  Analytical solution computed: {len(time_analytical)} time points")
print(f"  Time range: [{time_analytical[0]:.6e}, {time_analytical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(radius_analytical)*1000:.4f}, {np.max(radius_analytical)*1000:.4f}] mm")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def extract_bubble_radius_from_eta(ds, center_x, center_y, eta_value=0.5):
    """
    Extract bubble radius by finding the eta=0.5 contour.
    Uses radial sampling from bubble center.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge[0])
    x_max = float(ds.domain_right_edge[0])
    y_min = float(ds.domain_left_edge[1])
    y_max = float(ds.domain_right_edge[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 256  # Higher resolution for better contour detection
    
    # Create fixed resolution buffer
    width_x = x_max - x_min
    width_y = y_max - y_min
    frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
    
    # Extract eta field
    eta_field = np.array(frb['eta'])
    
    # Create coordinate arrays
    x_1d = np.linspace(x_min, x_max, eta_field.shape[1])
    y_1d = np.linspace(y_min, y_max, eta_field.shape[0])
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    
    # Calculate radial distance from bubble center
    R_grid = np.sqrt((X_grid - center_x)**2 + (Y_grid - center_y)**2)
    
    # Find points near eta = 0.5 contour
    eta_tolerance = 0.05
    contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) == 0:
        # If no points found, try larger tolerance
        eta_tolerance = 0.1
        contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) > 0:
        # Extract radii at contour points
        contour_radii = R_grid[contour_mask]
        
        # Use mean radius as bubble radius
        bubble_radius = np.mean(contour_radii)
        
        return bubble_radius
    else:
        return None

def extract_fsvx_profile_at_y0(ds):
    """
    Extract Fsvx profile along x-axis at y=0 (or near y=0).
    Returns x coordinates and Fsvx values for x > 0.
    """
    try:
        # Get domain bounds
        x_min = float(ds.domain_left_edge[0])
        x_max = float(ds.domain_right_edge[0])
        y_min = float(ds.domain_left_edge[1])
        y_max = float(ds.domain_right_edge[1])
        
        # Create 2D slice at z=0
        slc = ds.slice('z', 0.0)
        
        # Create fixed resolution buffer with high resolution
        resolution = 512
        width_x = x_max - x_min
        width_y = y_max - y_min
        frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
        
        # Extract Fsvx field
        fsvx_field = np.array(frb['Fsvx'])
        
        # Create coordinate arrays
        x_1d = np.linspace(x_min, x_max, fsvx_field.shape[1])
        y_1d = np.linspace(y_min, y_max, fsvx_field.shape[0])
        
        # Find index closest to y=0
        y_idx = np.argmin(np.abs(y_1d - 0.0))
        
        # Extract 1D profile at this y location
        fsvx_profile = fsvx_field[y_idx, :]
        
        # Filter to only x > 0
        x_positive_mask = x_1d > 0.0
        x_positive = x_1d[x_positive_mask]
        fsvx_positive = fsvx_profile[x_positive_mask]
        
        return x_positive, fsvx_positive
        
    except Exception as e:
        print(f"  WARNING: Could not extract Fsvx profile: {e}")
        return None, None

def integrate_fsvx_2d_slice(ds, y_center=0.0, y_width=0.005):
    """
    Integrate Fsvx over a 2D slice near y=y_center with width y_width.
    Computes: int(Fsvx * dx) from x=0 to x=x_max
    Returns the raw integral (to be normalized by R later).
    """
    try:
        # Get domain bounds
        x_min = float(ds.domain_left_edge[0])
        x_max = float(ds.domain_right_edge[0])
        y_min = float(ds.domain_left_edge[1])
        y_max = float(ds.domain_right_edge[1])
        
        # Create 2D slice at z=0
        slc = ds.slice('z', 0.0)
        
        # Create fixed resolution buffer with high resolution
        resolution = 512
        width_x = x_max - x_min
        width_y = y_max - y_min
        frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
        
        # Extract Fsvx field
        fsvx_field = np.array(frb['Fsvx'])
        
        # Create coordinate arrays
        x_1d = np.linspace(x_min, x_max, fsvx_field.shape[1])
        y_1d = np.linspace(y_min, y_max, fsvx_field.shape[0])
        
        # Find indices corresponding to the integration strip around y_center
        y_lower = y_center - y_width / 2.0
        y_upper = y_center + y_width / 2.0
        
        y_mask = (y_1d >= y_lower) & (y_1d <= y_upper)
        
        if np.sum(y_mask) == 0:
            print(f"  WARNING: No points found in y range [{y_lower}, {y_upper}]")
            return None
        
        # Extract the strip
        fsvx_strip = fsvx_field[y_mask, :]
        
        # Average over y direction to get 1D profile along x
        fsvx_avg_y = np.mean(fsvx_strip, axis=0)
        
        # Filter to only x > 0
        x_positive_mask = x_1d > 0.0
        x_positive = x_1d[x_positive_mask]
        fsvx_positive = fsvx_avg_y[x_positive_mask]
        
        if len(x_positive) > 1:
            # Integrate using trapezoidal rule: int(Fsvx * dx)
            fsvx_integrated = np.trapz(fsvx_positive, x_positive)
            
            # Debug output
            print(f"    Fsvx integration: max(Fsvx)={np.max(fsvx_positive):.6e}, " +
                  f"integral={fsvx_integrated:.6e}")
            
            return fsvx_integrated
        else:
            return None
            
    except Exception as e:
        print(f"  WARNING: Could not integrate Fsvx from 2D slice: {e}")
        return None

def extract_kappa_avg_filtered(ds, kappa_analytical, filter_percent=50.0):
    """
    Extract average curvature (kappaAvg) from the dataset with outlier filtering.
    Filters out values outside +/- filter_percent of analytical value.
    """
    try:
        # Get all data
        ad = ds.all_data()
        kappa_values = np.array(ad['kappaAvg'])
        
        # Calculate bounds for filtering
        kappa_lower = kappa_analytical * (1.0 - filter_percent / 100.0)
        kappa_upper = kappa_analytical * (1.0 + filter_percent / 100.0)
        
        # Filter outliers
        valid_mask = (kappa_values >= kappa_lower) & (kappa_values <= kappa_upper)
        kappa_filtered = kappa_values[valid_mask]
        
        if len(kappa_filtered) > 0:
            # Calculate mean curvature of filtered values
            kappa_avg = np.mean(kappa_filtered)
            
            print(f"    Kappa filtering: {len(kappa_filtered)}/{len(kappa_values)} values kept, avg={kappa_avg:.6e}")
            
            return kappa_avg
        else:
            print(f"    WARNING: All kappa values filtered out!")
            return None
        
    except Exception as e:
        print(f"  WARNING: Could not extract kappaAvg: {e}")
        return None

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("\n" + "=" * 70)
print("LOADING SIMULATION DATA")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    # Filter out .old files and ensure it's a directory
    if os.path.isdir(item_path) and '.old' not in item:
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

# Sort by timestep number
plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files (excluding .old files)")
print(f"Integration: (1/R) * int(Fsvx*dx) from x=0 to x=x_max")
print(f"Integration parameters: y_width={Y_INTEGRATION_WIDTH} m")
print(f"Curvature filter: +/- {KAPPA_FILTER_PERCENT}% of analytical")

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM SIMULATION")
print("=" * 70)

times_numerical = []
radii_numerical = []
fsvx_integrated_raw = []  # Store raw integrals
kappa_avg_values = []

# Store one Fsvx profile for plotting
fsvx_profile_x = None
fsvx_profile_values = None
fsvx_profile_time = None

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Extract bubble radius from eta field
        radius = extract_bubble_radius_from_eta(ds, bubble_center_x, bubble_center_y, eta_contour)
        
        if radius is not None:
            # Calculate analytical curvature for filtering
            kappa_ana = 1.0 / radius
            
            # Integrate Fsvx using 2D slice method (returns raw integral)
            fsvx_int_raw = integrate_fsvx_2d_slice(ds, y_center=bubble_center_y, y_width=Y_INTEGRATION_WIDTH)
            
            # Extract average curvature with filtering
            kappa_avg = extract_kappa_avg_filtered(ds, kappa_ana, KAPPA_FILTER_PERCENT)
            
            # Store Fsvx profile from middle timestep for plotting
            if fsvx_profile_x is None and i == len(plot_files) // 2:
                fsvx_profile_x, fsvx_profile_values = extract_fsvx_profile_at_y0(ds)
                fsvx_profile_time = t
                print(f"  Stored Fsvx profile at t={t:.6e} s for plotting")
            
            times_numerical.append(t)
            radii_numerical.append(radius)
            fsvx_integrated_raw.append(fsvx_int_raw if fsvx_int_raw is not None else np.nan)
            kappa_avg_values.append(kappa_avg if kappa_avg is not None else np.nan)
        else:
            print(f"  WARNING: Could not extract radius at t={t:.6e} s")
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times_numerical = np.array(times_numerical)
radii_numerical = np.array(radii_numerical)
fsvx_integrated_raw = np.array(fsvx_integrated_raw)
kappa_avg_values = np.array(kappa_avg_values)

# Normalize Fsvx integral by radius: (1/R) * int(Fsvx*dx)
fsvx_integrated = fsvx_integrated_raw / radii_numerical

print(f"\nSuccessfully extracted {len(times_numerical)} measurements")
print(f"  Time range: [{times_numerical[0]:.6e}, {times_numerical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(radii_numerical)*1000:.4f}, {np.max(radii_numerical)*1000:.4f}] mm")
print(f"  Fsvx raw integral range: [{np.nanmin(fsvx_integrated_raw):.6e}, {np.nanmax(fsvx_integrated_raw):.6e}]")
print(f"  Fsvx normalized range: [{np.nanmin(fsvx_integrated):.6e}, {np.nanmax(fsvx_integrated):.6e}]")

# ============================================================================
# CALCULATE ANALYTICAL VALUES
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING ANALYTICAL VALUES")
print("=" * 70)

# Interpolate analytical radius to numerical time points
analytical_interp = interp1d(time_analytical, radius_analytical, 
                             kind='cubic', fill_value='extrapolate')
radii_analytical_interp = analytical_interp(times_numerical)

# Sharp interface limit for surface tension force: F = sigma / R
fsvx_analytical = S / radii_analytical_interp

# Analytical curvature for 2D cylinder: kappa = 1 / R
kappa_analytical = 1.0 / radii_analytical_interp

print(f"  Calculated analytical values for {len(times_numerical)} points")
print(f"  Analytical sigma/R range: [{np.min(fsvx_analytical):.6e}, {np.max(fsvx_analytical):.6e}]")

# ============================================================================
# CALCULATE ERRORS
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING ERRORS")
print("=" * 70)

# Radius errors
radius_abs_error = np.abs(radii_numerical - radii_analytical_interp)
radius_rel_error = (radius_abs_error / radii_analytical_interp) * 100

# Fsvx errors (filter out NaN values)
valid_fsvx = ~np.isnan(fsvx_integrated)
fsvx_abs_error = np.abs(fsvx_integrated[valid_fsvx] - fsvx_analytical[valid_fsvx])
fsvx_rel_error = (fsvx_abs_error / fsvx_analytical[valid_fsvx]) * 100

# Kappa errors (filter out NaN values)
valid_kappa = ~np.isnan(kappa_avg_values)
kappa_abs_error = np.abs(kappa_avg_values[valid_kappa] - kappa_analytical[valid_kappa])
kappa_rel_error = (kappa_abs_error / kappa_analytical[valid_kappa]) * 100

print(f"\nRadius Error Statistics:")
print(f"  Max absolute error: {np.max(radius_abs_error)*1000:.6f} mm")
print(f"  Mean absolute error: {np.mean(radius_abs_error)*1000:.6f} mm")
print(f"  Max relative error: {np.max(radius_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(radius_rel_error):.4f}%")

if len(fsvx_abs_error) > 0:
    print(f"\nFsvx Error Statistics:")
    print(f"  Max absolute error: {np.max(fsvx_abs_error):.6e}")
    print(f"  Mean absolute error: {np.mean(fsvx_abs_error):.6e}")
    print(f"  Max relative error: {np.max(fsvx_rel_error):.4f}%")
    print(f"  Mean relative error: {np.mean(fsvx_rel_error):.4f}%")

if len(kappa_abs_error) > 0:
    print(f"\nKappa Error Statistics:")
    print(f"  Max absolute error: {np.max(kappa_abs_error):.6e} 1/m")
    print(f"  Mean absolute error: {np.mean(kappa_abs_error):.6e} 1/m")
    print(f"  Max relative error: {np.max(kappa_rel_error):.4f}%")
    print(f"  Mean relative error: {np.mean(kappa_rel_error):.4f}%")

# ============================================================================
# PLOT 1: RADIUS COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(time_analytical * 1000, radius_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, 
         label='Analytical (2D RPE)', zorder=1)
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, 
         label='Numerical (eta=0.5)', alpha=0.7, zorder=2)
ax1.axhline(y=R0*1000, color='gray', linestyle='--', 
            linewidth=1.5, alpha=0.5, label=f'R0 = {R0*1000:.1f} mm')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Bubble Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Bubble Radius Comparison\nrho={rho_L} kg/m^3, mu={mu_L} Pa*s, sigma={S} N/m',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Radius_Comparison.eps'))
print("  Saved: 01_Radius_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: RADIUS ABSOLUTE ERROR (LOG SCALE)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.semilogy(times_numerical * 1000, radius_abs_error * 1000, 
             'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Absolute Error (mm)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Radius Absolute Error: |R_numerical - R_analytical| (Log Scale)',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.tick_params(labelsize=FONT_SIZE_TICK)
textstr = f'Max: {np.max(radius_abs_error)*1000:.6f} mm\nMean: {np.mean(radius_abs_error)*1000:.6f} mm'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Radius_Absolute_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Radius_Absolute_Error.eps'))
print("  Saved: 02_Radius_Absolute_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: RADIUS RELATIVE ERROR (LOG SCALE)
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.semilogy(times_numerical * 1000, radius_rel_error, 
             'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Radius Relative Error (Log Scale)',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)
textstr = f'Max: {np.max(radius_rel_error):.4f}%\nMean: {np.mean(radius_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Radius_Relative_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Radius_Relative_Error.eps'))
print("  Saved: 03_Radius_Relative_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: FSVX PROFILE AT Y=0
# ============================================================================

if fsvx_profile_x is not None and fsvx_profile_values is not None:
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(fsvx_profile_x * 1000, fsvx_profile_values, 
             'b-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax4.set_xlabel('X Position (mm)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Fsvx (N/m)', fontsize=FONT_SIZE_LABEL)
    ax4.set_title(f'Fsvx Profile along X-axis at y=0\nt = {fsvx_profile_time:.6e} s',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=FONT_SIZE_TICK)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_Fsvx_Profile.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '04_Fsvx_Profile.eps'))
    print("  Saved: 04_Fsvx_Profile.png/.eps")
    plt.close()

# ============================================================================
# PLOT 5: FSVX VS TIME
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.plot(times_numerical * 1000, fsvx_analytical, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, 
         label='Analytical (sigma/R)', zorder=1)
ax5.plot(times_numerical[valid_fsvx] * 1000, -fsvx_integrated[valid_fsvx], 
         'ro', markersize=MARKER_SIZE, 
         label='Numerical ((1/R)*int(Fsvx*dx))', alpha=0.7, zorder=2)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Surface Tension Force (N/m)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Surface Tension Force vs Time',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Fsvx_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Fsvx_vs_Time.eps'))
print("  Saved: 05_Fsvx_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 6: FSVX ABSOLUTE ERROR VS TIME (LOG SCALE)
# ============================================================================

if len(fsvx_abs_error) > 0:
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    ax6.semilogy(times_numerical[valid_fsvx] * 1000, fsvx_abs_error, 
                 'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax6.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax6.set_ylabel('Absolute Error (N/m)', fontsize=FONT_SIZE_LABEL)
    ax6.set_title('Fsvx Absolute Error vs Time (Log Scale)',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    textstr = f'Max: {np.max(fsvx_abs_error):.6e} N/m\nMean: {np.mean(fsvx_abs_error):.6e} N/m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '06_Fsvx_Absolute_Error_Time.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '06_Fsvx_Absolute_Error_Time.eps'))
    print("  Saved: 06_Fsvx_Absolute_Error_Time.png/.eps")
    plt.close()

# ============================================================================
# PLOT 7: FSVX RELATIVE ERROR VS TIME (LOG SCALE)
# ============================================================================

if len(fsvx_rel_error) > 0:
    fig7, ax7 = plt.subplots(figsize=(12, 8))
    ax7.semilogy(times_numerical[valid_fsvx] * 1000, fsvx_rel_error, 
                 'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax7.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax7.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
    ax7.set_title('Fsvx Relative Error vs Time (Log Scale)',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax7.grid(True, alpha=0.3, which='both')
    ax7.tick_params(labelsize=FONT_SIZE_TICK)
    textstr = f'Max: {np.max(fsvx_rel_error):.4f}%\nMean: {np.mean(fsvx_rel_error):.4f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax7.text(0.05, 0.95, textstr, transform=ax7.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '07_Fsvx_Relative_Error_Time.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '07_Fsvx_Relative_Error_Time.eps'))
    print("  Saved: 07_Fsvx_Relative_Error_Time.png/.eps")
    plt.close()

# ============================================================================
# PLOT 8: KAPPA VS TIME
# ============================================================================

fig8, ax8 = plt.subplots(figsize=(12, 8))
ax8.plot(times_numerical * 1000, kappa_analytical, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, 
         label='Analytical (1/R)', zorder=1)
ax8.plot(times_numerical[valid_kappa] * 1000, kappa_avg_values[valid_kappa], 
         'ro', markersize=MARKER_SIZE, 
         label=f'Numerical (filtered +/-{KAPPA_FILTER_PERCENT}%)', alpha=0.7, zorder=2)
ax8.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax8.set_ylabel('Curvature (1/m)', fontsize=FONT_SIZE_LABEL)
ax8.set_title('Surface Curvature vs Time',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax8.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax8.grid(True, alpha=0.3)
ax8.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '08_Kappa_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '08_Kappa_vs_Time.eps'))
print("  Saved: 08_Kappa_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 9: KAPPA ABSOLUTE ERROR VS TIME (LOG SCALE)
# ============================================================================

if len(kappa_abs_error) > 0:
    fig9, ax9 = plt.subplots(figsize=(12, 8))
    ax9.semilogy(times_numerical[valid_kappa] * 1000, kappa_abs_error, 
                 'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax9.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax9.set_ylabel('Absolute Error (1/m)', fontsize=FONT_SIZE_LABEL)
    ax9.set_title('Kappa Absolute Error vs Time (Log Scale)',
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax9.grid(True, alpha=0.3, which='both')
    ax9.tick_params(labelsize=FONT_SIZE_TICK)
    textstr = f'Max: {np.max(kappa_abs_error):.6e} 1/m\nMean: {np.mean(kappa_abs_error):.6e} 1/m'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax9.text(0.05, 0.95, textstr, transform=ax9.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '09_Kappa_Absolute_Error_Time.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '09_Kappa_Absolute_Error_Time.eps'))
    print("  Saved: 09_Kappa_Absolute_Error_Time.png/.eps")
    plt.close()

# ============================================================================
# PLOT 10: KAPPA RELATIVE ERROR VS TIME (LOG SCALE)
# ============================================================================

if len(kappa_rel_error) > 0:
    fig10, ax10 = plt.subplots(figsize=(12, 8))
    ax10.semilogy(times_numerical[valid_kappa] * 1000, kappa_rel_error, 
                  'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax10.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax10.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
    ax10.set_title('Kappa Relative Error vs Time (Log Scale)',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax10.grid(True, alpha=0.3, which='both')
    ax10.tick_params(labelsize=FONT_SIZE_TICK)
    textstr = f'Max: {np.max(kappa_rel_error):.4f}%\nMean: {np.mean(kappa_rel_error):.4f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax10.text(0.05, 0.95, textstr, transform=ax10.transAxes, 
              fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '10_Kappa_Relative_Error_Time.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '10_Kappa_Relative_Error_Time.eps'))
    print("  Saved: 10_Kappa_Relative_Error_Time.png/.eps")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("DEEP DIVE ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  Radius Analysis:")
print(f"    - 01_Radius_Comparison.png/.eps")
print(f"    - 02_Radius_Absolute_Error.png/.eps (LOG SCALE)")
print(f"    - 03_Radius_Relative_Error.png/.eps (LOG SCALE)")
print(f"  Surface Tension Force (Fsvx) Analysis:")
print(f"    - 04_Fsvx_Profile.png/.eps (Profile at y=0)")
print(f"    - 05_Fsvx_vs_Time.png/.eps ((1/R)*int(Fsvx*dx) vs sigma/R)")
print(f"    - 06_Fsvx_Absolute_Error_Time.png/.eps (LOG SCALE)")
print(f"    - 07_Fsvx_Relative_Error_Time.png/.eps (LOG SCALE)")
print(f"  Curvature (Kappa) Analysis:")
print(f"    - 08_Kappa_vs_Time.png/.eps (with +/-{KAPPA_FILTER_PERCENT}% filtering)")
print(f"    - 09_Kappa_Absolute_Error_Time.png/.eps (LOG SCALE)")
print(f"    - 10_Kappa_Relative_Error_Time.png/.eps (LOG SCALE)")
print("\n" + "=" * 70)
