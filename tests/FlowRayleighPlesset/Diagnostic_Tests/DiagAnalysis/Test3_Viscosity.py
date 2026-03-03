# -*- coding: utf-8 -*-

"""
===============================================================================
UNIT TEST 3 ANALYSIS: VISCOUS DAMPING TERM VALIDATION
===============================================================================

PURPOSE:
    Validate that viscous damping term is properly implemented in an
    overdamped bubble collapse scenario (high viscosity, no surface tension).
    
    Key viscous term in 2D cylindrical RPE:
    - Viscous damping: -4*mu_L*R_dot/(rho_L*R)

VALIDATION CHECKS:
    1. Verify monotonic collapse (no oscillation)
    2. Check viscous term magnitude and sign
    3. Compare numerical vs analytical viscous term
    4. Validate velocity decay rate
    5. Verify overdamped behavior (no R_ddot sign changes)

EXPECTED BEHAVIOR:
    - Slow monotonic collapse (no oscillation)
    - R_dot negative and decaying toward zero
    - Viscous term dominates dynamics
    - No acceleration sign changes (overdamped)

FAILURE MODES:
    - Oscillation present -> viscous damping too weak
    - Viscous term zero -> not applied at interface
    - Collapse too fast -> viscous term not computed correctly
    - R_dot doesn't decay -> viscous dissipation not working

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH TEST3 INPUT FILE)
rho_L = 10.0              # Liquid density [kg/m^3]
mu_L = 1.0                # Dynamic viscosity [Pa*s] - HIGH for this test
S = 0.0                   # Surface tension [N/m] - ZERO for this test
p_v = 0.0                 # Vapor pressure [Pa]
gamma = 1.4               # Adiabatic index

# Initial conditions
p_inf = 500.0             # External pressure [Pa]
p_B0 = 550.0              # Initial bubble pressure [Pa] - small difference
R0 = 0.02                 # Initial radius [m]
R_dot0 = 0.0              # Initial velocity [m/s]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.5         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'./tests/RayleighPlesset/TEST3_ViscousDamping'  # Directory containing AMReX plot files

# Analytical RPE solver parameters
r_inf = 5.0 * R0          # Far-field boundary for RPE
t_span_rpe = (0, 0.05)    # Time span for RPE integration [s]
n_points_rpe = 10000      # Number of time points for RPE

# Finite difference parameters
FD_METHOD = 'central'     # 'central' or 'forward' for derivatives
FD_SMOOTHING = True       # Apply Savitzky-Golay smoothing to derivatives
FD_WINDOW = 7             # Window size for smoothing (must be odd)
FD_POLYORDER = 3          # Polynomial order for smoothing

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './TEST3_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL RPE SOLUTION (2D CYLINDRICAL - VISCOUS)
# ============================================================================

def rp_equation_2d_viscous(t, y):
    """
    Chen's 2D Cylindrical RPE - Viscous version (no surface tension)
    Returns [R_dot, R_ddot]
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
    
    # External pressure
    p_ext = p_inf
    
    # RHS pressure term (WITH VISCOSITY, NO SURFACE TENSION)
    pressure_term = (
        p_B
        - p_ext
        - 4 * mu_L * R_dot / R
    ) / rho_L
    
    # Correct cylindrical inertia structure
    numerator = pressure_term - R_dot**2 * (0.5 - ln_factor)
    denominator = R * ln_factor
    
    R_ddot = numerator / denominator
    
    return [R_dot, R_ddot]

print("=" * 70)
print("UNIT TEST 3: VISCOUS DAMPING TERM VALIDATION")
print("=" * 70)

# Solve analytical RPE
print("\nSolving analytical 2D cylindrical RPE (viscous)...")
y0 = [R0, R_dot0]
t_eval_rpe = np.linspace(*t_span_rpe, n_points_rpe)
sol = solve_ivp(rp_equation_2d_viscous, t_span_rpe, y0, t_eval=t_eval_rpe,
                method='RK45', rtol=1e-9, atol=1e-11)

if not sol.success:
    print(f"WARNING: RPE integration failed - {sol.message}")
    exit(1)

R_analytical = sol.y[0]
R_dot_analytical = sol.y[1]
time_analytical = sol.t

# Compute R_ddot analytically
R_ddot_analytical = np.zeros_like(R_analytical)
for i in range(len(time_analytical)):
    _, R_ddot_analytical[i] = rp_equation_2d_viscous(time_analytical[i], 
                                                      [R_analytical[i], R_dot_analytical[i]])

print(f"  Analytical solution computed: {len(time_analytical)} time points")
print(f"  Time range: [{time_analytical[0]:.6e}, {time_analytical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(R_analytical)*1000:.4f}, {np.max(R_analytical)*1000:.4f}] mm")

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
    resolution = 256
    
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
        eta_tolerance = 0.1
        contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) > 0:
        contour_radii = R_grid[contour_mask]
        bubble_radius = np.mean(contour_radii)
        return bubble_radius
    else:
        return None

def compute_derivatives(t, R, method='central', smooth=False, window=5, polyorder=2):
    """
    Compute first and second derivatives of R with respect to t.
    """
    n = len(t)
    R_dot = np.zeros(n)
    R_ddot = np.zeros(n)
    
    if method == 'central':
        # Central difference for interior points
        for i in range(1, n-1):
            dt_forward = t[i+1] - t[i]
            dt_backward = t[i] - t[i-1]
            dt_avg = (dt_forward + dt_backward) / 2.0
            R_dot[i] = (R[i+1] - R[i-1]) / (2 * dt_avg)
        
        # Forward/backward difference for endpoints
        R_dot[0] = (R[1] - R[0]) / (t[1] - t[0])
        R_dot[-1] = (R[-1] - R[-2]) / (t[-1] - t[-2])
        
        # Second derivative
        for i in range(1, n-1):
            dt = t[i+1] - t[i]
            R_ddot[i] = (R[i+1] - 2*R[i] + R[i-1]) / dt**2
        
        R_ddot[0] = R_ddot[1]
        R_ddot[-1] = R_ddot[-2]
    
    # Apply smoothing if requested
    if smooth and n > window:
        R_dot = savgol_filter(R_dot, window, polyorder)
        R_ddot = savgol_filter(R_ddot, window, polyorder)
    
    return R_dot, R_ddot

def check_monotonic(signal):
    """
    Check if signal is monotonically decreasing (for collapse)
    Returns percentage of monotonic behavior
    """
    diffs = np.diff(signal)
    monotonic_decreasing = np.sum(diffs <= 0) / len(diffs) * 100
    return monotonic_decreasing

def count_sign_changes(signal):
    """
    Count number of times signal changes sign
    """
    sign_changes = np.sum(np.diff(np.sign(signal)) != 0)
    return sign_changes

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("\n" + "=" * 70)
print("LOADING SIMULATION DATA")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and '.old' not in item:
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)

print(f"\nFound {len(plot_files)} plot files (excluding .old files)")
print(f"Finite difference method: {FD_METHOD}")
print(f"Smoothing: {FD_SMOOTHING} (window={FD_WINDOW}, polyorder={FD_POLYORDER})")

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM SIMULATION")
print("=" * 70)

times_numerical = []
radii_numerical = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
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

print(f"\nSuccessfully extracted {len(times_numerical)} measurements")
print(f"  Time range: [{times_numerical[0]:.6e}, {times_numerical[-1]:.6e}] s")
print(f"  Radius range: [{np.min(radii_numerical)*1000:.4f}, {np.max(radii_numerical)*1000:.4f}] mm")

# ============================================================================
# COMPUTE NUMERICAL DERIVATIVES
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING NUMERICAL DERIVATIVES")
print("=" * 70)

R_dot_numerical, R_ddot_numerical = compute_derivatives(
    times_numerical, radii_numerical, 
    method=FD_METHOD, smooth=FD_SMOOTHING, 
    window=FD_WINDOW, polyorder=FD_POLYORDER
)

print(f"  R_dot range: [{np.min(R_dot_numerical):.6e}, {np.max(R_dot_numerical):.6e}] m/s")
print(f"  R_ddot range: [{np.min(R_ddot_numerical):.6e}, {np.max(R_ddot_numerical):.6e}] m/s^2")

# ============================================================================
# CHECK MONOTONIC BEHAVIOR
# ============================================================================

print("\n" + "=" * 70)
print("MONOTONIC BEHAVIOR ANALYSIS")
print("=" * 70)

monotonic_R = check_monotonic(radii_numerical)
monotonic_R_dot = check_monotonic(R_dot_numerical)

sign_changes_R_dot = count_sign_changes(R_dot_numerical)
sign_changes_R_ddot = count_sign_changes(R_ddot_numerical)

print(f"\nMonotonic behavior:")
print(f"  Radius monotonically decreasing: {monotonic_R:.2f}%")
print(f"  Velocity monotonically decreasing: {monotonic_R_dot:.2f}%")

print(f"\nSign changes:")
print(f"  R_dot sign changes: {sign_changes_R_dot}")
print(f"  R_ddot sign changes: {sign_changes_R_ddot}")

# ============================================================================
# INTERPOLATE ANALYTICAL SOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("INTERPOLATING ANALYTICAL SOLUTION")
print("=" * 70)

R_analytical_interp = interp1d(time_analytical, R_analytical, 
                               kind='cubic', fill_value='extrapolate')(times_numerical)
R_dot_analytical_interp = interp1d(time_analytical, R_dot_analytical, 
                                   kind='cubic', fill_value='extrapolate')(times_numerical)
R_ddot_analytical_interp = interp1d(time_analytical, R_ddot_analytical, 
                                    kind='cubic', fill_value='extrapolate')(times_numerical)

print(f"  Interpolated analytical solution to {len(times_numerical)} numerical time points")

# ============================================================================
# COMPUTE VISCOUS TERM
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING VISCOUS DAMPING TERM")
print("=" * 70)

# Viscous term: -4*mu_L*R_dot/(rho_L*R)
viscous_term_num = -4 * mu_L * R_dot_numerical / (rho_L * radii_numerical)
viscous_term_ana = -4 * mu_L * R_dot_analytical_interp / (rho_L * R_analytical_interp)

print("  Numerical viscous term:")
print(f"    Range: [{np.min(viscous_term_num):.6e}, {np.max(viscous_term_num):.6e}] m^2/s^2")
print(f"    Mean: {np.mean(viscous_term_num):.6e} m^2/s^2")

print("  Analytical viscous term:")
print(f"    Range: [{np.min(viscous_term_ana):.6e}, {np.max(viscous_term_ana):.6e}] m^2/s^2")
print(f"    Mean: {np.mean(viscous_term_ana):.6e} m^2/s^2")

# Compute errors
viscous_abs_error = np.abs(viscous_term_num - viscous_term_ana)
viscous_rel_error = np.abs((viscous_term_num - viscous_term_ana) / viscous_term_ana) * 100

print(f"\nViscous term error:")
print(f"  Max absolute error: {np.max(viscous_abs_error):.6e} m^2/s^2")
print(f"  Mean absolute error: {np.mean(viscous_abs_error):.6e} m^2/s^2")
print(f"  Max relative error: {np.max(viscous_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(viscous_rel_error):.4f}%")

# ============================================================================
# COMPUTE ALL RPE TERMS FOR COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ALL RPE TERMS FOR COMPARISON")
print("=" * 70)

# Logarithmic factor
ln_factor_num = np.log(r_inf / radii_numerical)

# Gas pressure
p_B_num = p_v + (p_B0 - p_v) * (R0 / radii_numerical)**(2 * gamma)

# All terms
term_pressure_num = (p_B_num - p_inf) / rho_L
term_inertial1_num = radii_numerical * R_ddot_numerical * ln_factor_num
term_inertial2_num = R_dot_numerical**2 * (ln_factor_num - 0.5)

print("  Term magnitudes (mean absolute value):")
print(f"    Pressure term: {np.mean(np.abs(term_pressure_num)):.6e} m^2/s^2")
print(f"    Viscous term: {np.mean(np.abs(viscous_term_num)):.6e} m^2/s^2")
print(f"    Inertial term 1: {np.mean(np.abs(term_inertial1_num)):.6e} m^2/s^2")
print(f"    Inertial term 2: {np.mean(np.abs(term_inertial2_num)):.6e} m^2/s^2")

# Check if viscous term dominates
viscous_dominance = np.mean(np.abs(viscous_term_num)) / (
    np.mean(np.abs(term_inertial1_num)) + np.mean(np.abs(term_inertial2_num)) + 1e-12
)
print(f"\n  Viscous/Inertial ratio: {viscous_dominance:.4f}")

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
ax1.plot(time_analytical * 1000, R_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical (Viscous)', zorder=1)
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax1.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'R0 = {R0*1000:.1f} mm')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Bubble Radius vs Time (Overdamped)\nMonotonic: {monotonic_R:.1f}%', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_vs_Time.png'), dpi=300)
print("  Saved: 01_Radius_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: VELOCITY VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(time_analytical * 1000, R_dot_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax2.plot(times_numerical * 1000, R_dot_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Velocity R_dot (mm/s)', fontsize=FONT_SIZE_LABEL)
ax2.set_title(f'Bubble Wall Velocity vs Time\nSign changes: {sign_changes_R_dot}', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Velocity_vs_Time.png'), dpi=300)
print("  Saved: 02_Velocity_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 3: ACCELERATION VS TIME
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(time_analytical * 1000, R_ddot_analytical, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax3.plot(times_numerical * 1000, R_ddot_numerical, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Acceleration R_ddot (m/s^2)', fontsize=FONT_SIZE_LABEL)
ax3.set_title(f'Bubble Wall Acceleration vs Time\nSign changes: {sign_changes_R_ddot}', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Acceleration_vs_Time.png'), dpi=300)
print("  Saved: 03_Acceleration_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 4: VISCOUS TERM VS TIME
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times_numerical * 1000, viscous_term_ana, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax4.plot(times_numerical * 1000, viscous_term_num, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Viscous Term: -4*mu*R_dot/(rho*R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Viscous Damping Term vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Viscous_Term_vs_Time.png'), dpi=300)
print("  Saved: 04_Viscous_Term_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 5: VISCOUS TERM RELATIVE ERROR
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.semilogy(times_numerical * 1000, viscous_rel_error, 
             'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Viscous Term Relative Error vs Time (Log Scale)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max: {np.max(viscous_rel_error):.4f}%\nMean: {np.mean(viscous_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Viscous_Term_Error.png'), dpi=300)
print("  Saved: 05_Viscous_Term_Error.png")
plt.close()

# ============================================================================
# PLOT 6: ALL TERMS COMPARISON
# ============================================================================

fig6, ax6 = plt.subplots(figsize=(12, 8))
ax6.plot(times_numerical * 1000, term_pressure_num, 
         'g-', linewidth=LINE_WIDTH_NUMERICAL, label='Pressure', marker='o', markersize=4)
ax6.plot(times_numerical * 1000, viscous_term_num, 
         'r-', linewidth=LINE_WIDTH_NUMERICAL, label='Viscous', marker='s', markersize=4)
ax6.plot(times_numerical * 1000, term_inertial1_num, 
         'b-', linewidth=LINE_WIDTH_NUMERICAL, label='Inertial 1', marker='^', markersize=4)
ax6.plot(times_numerical * 1000, term_inertial2_num, 
         'm-', linewidth=LINE_WIDTH_NUMERICAL, label='Inertial 2', marker='v', markersize=4)
ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax6.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6.set_ylabel('Term Value (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax6.set_title('All RPE Terms Comparison (Numerical)', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6.legend(fontsize=FONT_SIZE_LEGEND)
ax6.grid(True, alpha=0.3)
ax6.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_All_Terms_Comparison.png'), dpi=300)
print("  Saved: 06_All_Terms_Comparison.png")
plt.close()

# ============================================================================
# PLOT 7: PHASE SPACE (R_dot vs R)
# ============================================================================

fig7, ax7 = plt.subplots(figsize=(12, 8))

# Analytical phase space
ax7.plot(R_analytical * 1000, R_dot_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', alpha=0.7)

# Numerical phase space
colors = times_numerical * 1000
scatter = ax7.scatter(radii_numerical * 1000, R_dot_numerical * 1000, 
                     c=colors, cmap='viridis', s=50, alpha=0.7, 
                     edgecolors='k', linewidth=0.5, label='Numerical')

ax7.set_xlabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax7.set_ylabel('Velocity R_dot (mm/s)', fontsize=FONT_SIZE_LABEL)
ax7.set_title('Phase Space: Velocity vs Radius (Overdamped)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7.legend(fontsize=FONT_SIZE_LEGEND)
ax7.grid(True, alpha=0.3)
ax7.tick_params(labelsize=FONT_SIZE_TICK)

cbar = plt.colorbar(scatter, ax=ax7)
cbar.set_label('Time (ms)', fontsize=FONT_SIZE_LABEL)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '07_Phase_Space.png'), dpi=300)
print("  Saved: 07_Phase_Space.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TEST 3 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_vs_Time.png")
print(f"  02_Velocity_vs_Time.png")
print(f"  03_Acceleration_vs_Time.png")
print(f"  04_Viscous_Term_vs_Time.png")
print(f"  05_Viscous_Term_Error.png")
print(f"  06_All_Terms_Comparison.png")
print(f"  07_Phase_Space.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MIN_MONOTONIC = 90.0  # Minimum % monotonic behavior
MAX_SIGN_CHANGES_RDOT = 2  # Maximum R_dot sign changes
MAX_SIGN_CHANGES_RDDOT = 3  # Maximum R_ddot sign changes
MAX_VISCOUS_ERROR = 10.0  # Maximum viscous term error (%)
MIN_VISCOUS_DOMINANCE = 2.0  # Viscous should be 2x larger than inertial

monotonic_pass = monotonic_R >= MIN_MONOTONIC
sign_changes_pass = (sign_changes_R_dot <= MAX_SIGN_CHANGES_RDOT and 
                     sign_changes_R_ddot <= MAX_SIGN_CHANGES_RDDOT)
viscous_error_pass = np.mean(viscous_rel_error) < MAX_VISCOUS_ERROR
viscous_dominance_pass = viscous_dominance >= MIN_VISCOUS_DOMINANCE

print(f"\nTest Results:")
print(f"  [{'PASS' if monotonic_pass else 'FAIL'}] Monotonic collapse: {monotonic_R:.2f}% (threshold: {MIN_MONOTONIC}%)")
print(f"  [{'PASS' if sign_changes_pass else 'FAIL'}] Sign changes: R_dot={sign_changes_R_dot}, R_ddot={sign_changes_R_ddot} (max: {MAX_SIGN_CHANGES_RDOT}, {MAX_SIGN_CHANGES_RDDOT})")
print(f"  [{'PASS' if viscous_error_pass else 'FAIL'}] Viscous term accuracy: {np.mean(viscous_rel_error):.4f}% (threshold: {MAX_VISCOUS_ERROR}%)")
print(f"  [{'PASS' if viscous_dominance_pass else 'FAIL'}] Viscous dominance: {viscous_dominance:.4f}x (threshold: {MIN_VISCOUS_DOMINANCE}x)")

if monotonic_pass and sign_changes_pass and viscous_error_pass and viscous_dominance_pass:
    print("\n*** TEST 3 PASSED: Viscous damping term correctly implemented ***")
else:
    print("\n*** TEST 3 FAILED: Viscous damping term NOT functioning correctly ***")
    print("\nPossible issues:")
    if not monotonic_pass:
        print("  - Bubble oscillating instead of monotonic collapse")
        print("  - Viscous damping too weak or not applied")
    if not sign_changes_pass:
        print("  - Too many sign changes (oscillatory behavior)")
        print("  - Viscosity not suppressing oscillations")
    if not viscous_error_pass:
        print("  - Viscous term magnitude incorrect")
        print("  - Check if -4*mu*R_dot/R is being computed correctly")
        print("  - Verify viscosity value at interface")
    if not viscous_dominance_pass:
        print("  - Viscous term not dominating dynamics")
        print("  - Inertial terms too large (should be suppressed)")
        print("  - Check if viscosity is being applied at interface")

print("\n" + "=" * 70)
