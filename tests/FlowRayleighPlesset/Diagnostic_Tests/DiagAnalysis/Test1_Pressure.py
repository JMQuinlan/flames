# -*- coding: utf-8 -*-

"""
===============================================================================
UNIT TEST 1 ANALYSIS: GAS PRESSURE ADIABATIC LAW VALIDATION
===============================================================================

PURPOSE:
    Validate that gas pressure inside the bubble follows the adiabatic
    compression law for an ideal gas in 2D cylindrical geometry:
    
    p_B = p_B0 * (R0 / R)^(2*gamma)
    
    This test isolates the gas equation of state from other physics
    (no viscosity, no surface tension).

VALIDATION CHECKS:
    1. Plot p_B vs time and R vs time
    2. Plot p_B vs R on log-log scale (should be straight line)
    3. Verify slope = -2*gamma = -2.8
    4. Check if p_B * R^(2*gamma) = constant
    5. Compute relative error in adiabatic invariant

EXPECTED BEHAVIOR:
    - Bubble should oscillate with small amplitude
    - Gas pressure should increase when radius decreases
    - Adiabatic invariant (p_B * R^(2*gamma)) should be constant
    - If invariant drifts, gas EOS coupling is broken

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import linregress
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH TEST1 INPUT FILE)
gamma = 1.4               # Adiabatic index
p_B0 = 550.0              # Initial bubble pressure [Pa]
p_inf = 500.0             # External pressure [Pa]
R0 = 0.02005                 # Initial radius [m]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.50         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'../../../../bin/tests/RayleighPlesset/TEST1_GasPressure'  # Directory containing AMReX plot files

# Pressure extraction parameters
PRESSURE_FIELD_NAME = 'pressure'  # Name of pressure field in simulation
PRESSURE_SAMPLE_RADIUS = 0.001    # Radius around bubble center to average [m]

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.5
MARKER_SIZE = 6

# Output settings
output_folder = './TEST1_Analysis'
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

def extract_pressure_at_bubble_center(ds, center_x, center_y, sample_radius=0.001):
    """
    Extract pressure at bubble center by averaging over a small region.
    """
    try:
        # Create a sphere (or circle in 2D) around bubble center
        center = [center_x, center_y, 0.0]
        sp = ds.sphere(center, (sample_radius, 'code_length'))
        
        # Extract pressure field
        pressure_values = np.array(sp[PRESSURE_FIELD_NAME])
        
        # Return mean pressure
        p_B_numerical = np.mean(pressure_values)
        
        return p_B_numerical
        
    except Exception as e:
        print(f"  WARNING: Could not extract pressure: {e}")
        return None

def compute_adiabatic_pressure(R, R0, p_B0, gamma):
    """
    Compute analytical adiabatic pressure for 2D cylindrical geometry
    p_B = p_B0 * (R0 / R)^(2*gamma)
    """
    return p_B0 * (R0 / R)**(2 * gamma)

def compute_adiabatic_invariant(p_B, R, gamma):
    """
    Compute adiabatic invariant: I = p_B * R^(2*gamma)
    This should be constant if adiabatic law is followed
    """
    return p_B * R**(2 * gamma)

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("UNIT TEST 1: GAS PRESSURE ADIABATIC LAW VALIDATION")
print("=" * 70)

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

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM SIMULATION")
print("=" * 70)

times_numerical = []
radii_numerical = []
pressure_numerical = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Extract bubble radius from eta field
        radius = extract_bubble_radius_from_eta(ds, bubble_center_x, bubble_center_y, eta_contour)
        
        # Extract pressure at bubble center
        pressure = extract_pressure_at_bubble_center(ds, bubble_center_x, bubble_center_y, 
                                                     PRESSURE_SAMPLE_RADIUS)
        
        if radius is not None and pressure is not None:
            times_numerical.append(t)
            radii_numerical.append(radius)
            pressure_numerical.append(pressure)
        else:
            if radius is None:
                print(f"  WARNING: Could not extract radius at t={t:.6e} s")
            if pressure is None:
                print(f"  WARNING: Could not extract pressure at t={t:.6e} s")
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times_numerical = np.array(times_numerical)
radii_numerical = np.array(radii_numerical)
pressure_numerical = np.array(pressure_numerical)

print(f"\nSuccessfully extracted {len(times_numerical)} measurements")
print(f"  Time range: [{times_numerical[0]:.6e}, {times_numerical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(radii_numerical)*1000:.4f}, {np.max(radii_numerical)*1000:.4f}] mm")
print(f"  Pressure range: [{np.min(pressure_numerical):.2f}, {np.max(pressure_numerical):.2f}] Pa")

# ============================================================================
# COMPUTE ANALYTICAL ADIABATIC PRESSURE
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL ADIABATIC PRESSURE")
print("=" * 70)

pressure_analytical = compute_adiabatic_pressure(radii_numerical, R0, p_B0, gamma)

print(f"  Analytical pressure range: [{np.min(pressure_analytical):.2f}, {np.max(pressure_analytical):.2f}] Pa")

# ============================================================================
# COMPUTE ADIABATIC INVARIANT
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ADIABATIC INVARIANT")
print("=" * 70)

invariant_numerical = compute_adiabatic_invariant(pressure_numerical, radii_numerical, gamma)
invariant_analytical = compute_adiabatic_invariant(pressure_analytical, radii_numerical, gamma)

# Theoretical invariant (should be constant)
invariant_theory = p_B0 * R0**(2 * gamma)

print(f"  Theoretical invariant: {invariant_theory:.6e}")
print(f"  Numerical invariant range: [{np.min(invariant_numerical):.6e}, {np.max(invariant_numerical):.6e}]")
print(f"  Invariant drift: {(np.max(invariant_numerical) - np.min(invariant_numerical)) / invariant_theory * 100:.4f}%")

# ============================================================================
# COMPUTE ERRORS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ERRORS")
print("=" * 70)

pressure_abs_error = np.abs(pressure_numerical - pressure_analytical)
pressure_rel_error = (pressure_abs_error / pressure_analytical) * 100

invariant_abs_error = np.abs(invariant_numerical - invariant_theory)
invariant_rel_error = (invariant_abs_error / invariant_theory) * 100

print(f"\nPressure Error Statistics:")
print(f"  Max absolute error: {np.max(pressure_abs_error):.4f} Pa")
print(f"  Mean absolute error: {np.mean(pressure_abs_error):.4f} Pa")
print(f"  Max relative error: {np.max(pressure_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(pressure_rel_error):.4f}%")

print(f"\nAdiabatic Invariant Error Statistics:")
print(f"  Max absolute error: {np.max(invariant_abs_error):.6e}")
print(f"  Mean absolute error: {np.mean(invariant_abs_error):.6e}")
print(f"  Max relative error: {np.max(invariant_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(invariant_rel_error):.4f}%")

# ============================================================================
# LOG-LOG REGRESSION ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("LOG-LOG REGRESSION ANALYSIS")
print("=" * 70)

# Perform linear regression on log(p_B) vs log(R)
log_R = np.log(radii_numerical)
log_p_numerical = np.log(pressure_numerical)
log_p_analytical = np.log(pressure_analytical)

# Regression for numerical data
slope_num, intercept_num, r_value_num, p_value_num, std_err_num = linregress(log_R, log_p_numerical)

# Expected slope
slope_expected = -2 * gamma

print(f"\nLog-Log Regression Results:")
print(f"  Expected slope: {slope_expected:.4f}")
print(f"  Numerical slope: {slope_num:.4f}")
print(f"  Slope error: {abs(slope_num - slope_expected):.4f}")
print(f"  R-squared: {r_value_num**2:.6f}")
print(f"  Slope relative error: {abs(slope_num - slope_expected) / abs(slope_expected) * 100:.4f}%")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: RADIUS VS TIME
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax1.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'R0 = {R0*1000:.1f} mm')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Bubble Radius vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_vs_Time.png'), dpi=300)
print("  Saved: 01_Radius_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: PRESSURE VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(times_numerical * 1000, pressure_analytical, 
         'b-', linewidth=LINE_WIDTH, label='Analytical (Adiabatic)', zorder=1)
ax2.plot(times_numerical * 1000, pressure_numerical, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax2.axhline(y=p_B0, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'p_B0 = {p_B0:.1f} Pa')
ax2.axhline(y=p_inf, color='green', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'p_inf = {p_inf:.1f} Pa')
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Gas Pressure vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Pressure_vs_Time.png'), dpi=300)
print("  Saved: 02_Pressure_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 3: PRESSURE VS RADIUS (LOG-LOG)
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plot data
ax3.loglog(radii_numerical * 1000, pressure_analytical, 
           'b-', linewidth=LINE_WIDTH, label='Analytical (Adiabatic)', zorder=1)
ax3.loglog(radii_numerical * 1000, pressure_numerical, 
           'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)

# Plot regression line
R_fit = np.linspace(np.min(radii_numerical), np.max(radii_numerical), 100)
p_fit = np.exp(intercept_num) * R_fit**slope_num
ax3.loglog(R_fit * 1000, p_fit, 'k--', linewidth=1.5, 
           label=f'Fit: slope = {slope_num:.3f}', alpha=0.7)

ax3.set_xlabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax3.set_title(f'Pressure vs Radius (Log-Log)\nExpected slope: {slope_expected:.3f}, Numerical slope: {slope_num:.3f}', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Pressure_vs_Radius_LogLog.png'), dpi=300)
print("  Saved: 03_Pressure_vs_Radius_LogLog.png")
plt.close()

# ============================================================================
# PLOT 4: ADIABATIC INVARIANT VS TIME
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times_numerical * 1000, invariant_numerical, 
         'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2, label='Numerical')
ax4.axhline(y=invariant_theory, color='red', linestyle='--', linewidth=2, 
            label=f'Theory: {invariant_theory:.6e}')
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Adiabatic Invariant: p_B * R^(2*gamma)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Adiabatic Invariant vs Time (Should be Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Adiabatic_Invariant_vs_Time.png'), dpi=300)
print("  Saved: 04_Adiabatic_Invariant_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 5: PRESSURE RELATIVE ERROR VS TIME
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.semilogy(times_numerical * 1000, pressure_rel_error, 
             'r-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Pressure Relative Error vs Time (Log Scale)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max: {np.max(pressure_rel_error):.4f}%\nMean: {np.mean(pressure_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Pressure_Relative_Error.png'), dpi=300)
print("  Saved: 05_Pressure_Relative_Error.png")
plt.close()

# ============================================================================
# PLOT 6: INVARIANT RELATIVE ERROR VS TIME
# ============================================================================

fig6, ax6 = plt.subplots(figsize=(12, 8))
ax6.semilogy(times_numerical * 1000, invariant_rel_error, 
             'g-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax6.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax6.set_title('Adiabatic Invariant Relative Error vs Time (Log Scale)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6.grid(True, alpha=0.3, which='both')
ax6.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max: {np.max(invariant_rel_error):.4f}%\nMean: {np.mean(invariant_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_Invariant_Relative_Error.png'), dpi=300)
print("  Saved: 06_Invariant_Relative_Error.png")
plt.close()

# ============================================================================
# PLOT 7: PHASE SPACE (R vs p_B)
# ============================================================================

fig7, ax7 = plt.subplots(figsize=(12, 8))

# Create colormap based on time
colors = times_numerical * 1000  # Convert to ms for colorbar

scatter = ax7.scatter(radii_numerical * 1000, pressure_numerical, 
                     c=colors, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

# Plot analytical curve
R_analytical_curve = np.linspace(np.min(radii_numerical), np.max(radii_numerical), 100)
p_analytical_curve = compute_adiabatic_pressure(R_analytical_curve, R0, p_B0, gamma)
ax7.plot(R_analytical_curve * 1000, p_analytical_curve, 'r--', linewidth=2, 
         label='Analytical Adiabatic')

ax7.set_xlabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax7.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax7.set_title('Phase Space: Pressure vs Radius (colored by time)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7.legend(fontsize=FONT_SIZE_LEGEND)
ax7.grid(True, alpha=0.3)
ax7.tick_params(labelsize=FONT_SIZE_TICK)

cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Time (ms)', fontsize=FONT_SIZE_LABEL)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '07_Phase_Space_R_vs_p.png'), dpi=300)
print("  Saved: 07_Phase_Space_R_vs_p.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TEST 1 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_vs_Time.png")
print(f"  02_Pressure_vs_Time.png")
print(f"  03_Pressure_vs_Radius_LogLog.png")
print(f"  04_Adiabatic_Invariant_vs_Time.png")
print(f"  05_Pressure_Relative_Error.png")
print(f"  06_Invariant_Relative_Error.png")
print(f"  07_Phase_Space_R_vs_p.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
PASS_THRESHOLD_PRESSURE = 5.0  # 5% error threshold
PASS_THRESHOLD_INVARIANT = 2.0  # 2% drift threshold
PASS_THRESHOLD_SLOPE = 0.1  # 10% slope error threshold

pressure_pass = np.mean(pressure_rel_error) < PASS_THRESHOLD_PRESSURE
invariant_pass = (np.max(invariant_rel_error) - np.min(invariant_rel_error)) < PASS_THRESHOLD_INVARIANT
slope_pass = abs(slope_num - slope_expected) / abs(slope_expected) * 100 < PASS_THRESHOLD_SLOPE

print(f"\nTest Results:")
print(f"  [{'PASS' if pressure_pass else 'FAIL'}] Pressure accuracy: {np.mean(pressure_rel_error):.4f}% (threshold: {PASS_THRESHOLD_PRESSURE}%)")
print(f"  [{'PASS' if invariant_pass else 'FAIL'}] Invariant drift: {(np.max(invariant_rel_error) - np.min(invariant_rel_error)):.4f}% (threshold: {PASS_THRESHOLD_INVARIANT}%)")
print(f"  [{'PASS' if slope_pass else 'FAIL'}] Log-log slope error: {abs(slope_num - slope_expected) / abs(slope_expected) * 100:.4f}% (threshold: {PASS_THRESHOLD_SLOPE}%)")

if pressure_pass and invariant_pass and slope_pass:
    print("\n*** TEST 1 PASSED: Gas pressure follows adiabatic law correctly ***")
else:
    print("\n*** TEST 1 FAILED: Gas pressure does NOT follow adiabatic law ***")
    print("\nPossible issues:")
    if not pressure_pass:
        print("  - Gas equation of state not properly implemented")
        print("  - Pressure boundary condition at interface incorrect")
    if not invariant_pass:
        print("  - Adiabatic invariant drifting (energy not conserved)")
        print("  - Gas mass not conserved during compression/expansion")
    if not slope_pass:
        print("  - Wrong exponent in gas law (check 2D vs 3D formulation)")
        print("  - Pressure-radius coupling broken")

print("\n" + "=" * 70)
