# -*- coding: utf-8 -*-
"""
===============================================================================
RAYLEIGH-PLESSET BUBBLE RADIUS ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Extract bubble radius from AMReX simulation data by tracking the eta=0.5
    contour and compare against analytical 2D cylindrical RPE solution.

FEATURES:
    - Extracts bubble radius from eta field (eta=0.5 contour)
    - Compares numerical radius vs analytical RPE solution
    - Generates radius comparison plot and error plot
    - Saves outputs as PNG and EPS

INPUTS:
    - AMReX plot files from Rayleigh-Plesset simulation
    - Physical parameters matching RPE analytical solution

OUTPUTS:
    - Radius comparison plot (numerical vs analytical)
    - Absolute error plot
    - Relative error plot

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
rho_L = 10.0              # Liquid density [kg/m^3]
mu_L = 0.1                # Dynamic viscosity [Pa*s]
S = 7.28                  # Surface tension [N/m]
p_v = 0.0                 # Vapor pressure [Pa]
gamma = 1.4               # Adiabatic index

# Initial conditions
p_inf = 500.0             # External pressure [Pa]
p_B0 = 1000.0             # Initial bubble pressure [Pa]
R0 = 0.02                 # Initial radius [m] = 20 mm
R_dot0 = 0.0              # Initial velocity [m/s]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.5         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'../../../bin/tests/RayleighPlesset/output_RayleighPlesset_UNIT_TEST'  # Directory containing AMReX plot files

# Analytical RPE solver parameters
r_inf = 5.0 * R0          # Far-field boundary for RPE
t_span_rpe = (0, 0.05)    # Time span for RPE integration [s]
n_points_rpe = 10000      # Number of time points for RPE

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './RPE_Analysis'
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
print("RAYLEIGH-PLESSET BUBBLE RADIUS ANALYSIS")
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

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("\n" + "=" * 70)
print("LOADING SIMULATION DATA")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files")

# ============================================================================
# EXTRACT BUBBLE RADIUS FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING BUBBLE RADIUS FROM ETA FIELD")
print("=" * 70)

times_numerical = []
radii_numerical = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Extract bubble radius from eta field
        radius = extract_bubble_radius_from_eta(ds, bubble_center_x, bubble_center_y, eta_contour)
        
        if radius is not None:
            times_numerical.append(t)
            radii_numerical.append(radius)
        else:
            print(f"  WARNING: Could not extract radius at t={t:.6e} s")
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times_numerical = np.array(times_numerical)
radii_numerical = np.array(radii_numerical)

print(f"\nSuccessfully extracted {len(times_numerical)} radius measurements")
print(f"  Time range: [{times_numerical[0]:.6e}, {times_numerical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(radii_numerical)*1000:.4f}, {np.max(radii_numerical)*1000:.4f}] mm")

# ============================================================================
# INTERPOLATE ANALYTICAL SOLUTION TO NUMERICAL TIME POINTS
# ============================================================================

print("\n" + "=" * 70)
print("INTERPOLATING ANALYTICAL SOLUTION")
print("=" * 70)

# Create interpolation function for analytical solution
analytical_interp = interp1d(time_analytical, radius_analytical, 
                             kind='cubic', fill_value='extrapolate')

# Interpolate to numerical time points
radii_analytical_interp = analytical_interp(times_numerical)

print(f"  Interpolated analytical solution to {len(times_numerical)} points")

# ============================================================================
# CALCULATE ERRORS
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING ERRORS")
print("=" * 70)

absolute_error = np.abs(radii_numerical - radii_analytical_interp)
relative_error = (absolute_error / radii_analytical_interp) * 100  # Percentage

print(f"\nError Statistics:")
print(f"  Max absolute error: {np.max(absolute_error)*1000:.6f} mm")
print(f"  Mean absolute error: {np.mean(absolute_error)*1000:.6f} mm")
print(f"  Max relative error: {np.max(relative_error):.4f}%")
print(f"  Mean relative error: {np.mean(relative_error):.4f}%")

# ============================================================================
# PLOT 1: RADIUS COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: RADIUS COMPARISON")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Plot analytical solution (full curve)
ax1.plot(time_analytical * 1000, radius_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, 
         label='Analytical (2D RPE)', zorder=1)

# Plot numerical data points
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, 
         label='Numerical (eta=0.5)', alpha=0.7, zorder=2)

# Add initial radius line
ax1.axhline(y=R0*1000, color='gray', linestyle='--', 
            linewidth=1.5, alpha=0.5, label=f'R0 = {R0*1000:.1f} mm')

ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Bubble Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Rayleigh-Plesset Bubble Radius Comparison\n' + 
              f'rho={rho_L} kg/m^3, mu={mu_L} Pa*s, sigma={S} N/m, p_inf={p_inf} Pa, p_B0={p_B0} Pa',
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
# PLOT 2: ABSOLUTE ERROR
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: ABSOLUTE ERROR")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(12, 8))

ax2.plot(times_numerical * 1000, absolute_error * 1000, 
         'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', 
         markersize=MARKER_SIZE-2)

ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Absolute Error (mm)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Rayleigh-Plesset: Absolute Radius Error\n|R_numerical - R_analytical|',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

# Add error statistics text box
textstr = f'Max Error: {np.max(absolute_error)*1000:.6f} mm\n'
textstr += f'Mean Error: {np.mean(absolute_error)*1000:.6f} mm'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Absolute_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Absolute_Error.eps'))
print("  Saved: 02_Absolute_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: RELATIVE ERROR
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: RELATIVE ERROR")
print("=" * 70)

fig3, ax3 = plt.subplots(figsize=(12, 8))

ax3.plot(times_numerical * 1000, relative_error, 
         'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', 
         markersize=MARKER_SIZE-2)

ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Rayleigh-Plesset: Relative Radius Error\n|(R_numerical - R_analytical) / R_analytical| x 100%',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)

# Add error statistics text box
textstr = f'Max Error: {np.max(relative_error):.4f}%\n'
textstr += f'Mean Error: {np.mean(relative_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Relative_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Relative_Error.eps'))
print("  Saved: 03_Relative_Error.png/.eps")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  - 01_Radius_Comparison.png/.eps")
print(f"  - 02_Absolute_Error.png/.eps")
print(f"  - 03_Relative_Error.png/.eps")
print("\n" + "=" * 70)
