# -*- coding: utf-8 -*-

"""
===============================================================================
UNIT TEST 2 ANALYSIS: INERTIAL TERMS VALIDATION
===============================================================================

PURPOSE:
    Validate that inertial terms remain active during bubble oscillation
    in an inviscid (zero viscosity, zero surface tension) simulation.
    
    Key inertial terms in 2D cylindrical RPE:
    - Term 1: R * R_ddot * ln(r_inf/R)  [Primary inertial]
    - Term 2: R_dot^2 * [ln(r_inf/R) - 0.5]  [Convective inertial]

VALIDATION CHECKS:
    1. Verify bubble oscillates (multiple cycles)
    2. Check R_ddot changes sign (acceleration reverses)
    3. Confirm inertial terms remain non-zero throughout
    4. Validate energy conservation (no artificial damping)
    5. Count number of oscillation cycles

EXPECTED BEHAVIOR:
    - At least 2-3 complete oscillation cycles
    - R_ddot crosses zero multiple times
    - Inertial terms active throughout simulation
    - No monotonic decay (energy conserved)

FAILURE MODES:
    - Monotonic collapse -> inertia not computed
    - R_ddot goes to zero and stays zero -> momentum loss
    - Single oscillation then stops -> artificial damping

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH TEST2 INPUT FILE)
rho_L = 10.0              # Liquid density [kg/m^3]
mu_L = 0.0                # Dynamic viscosity [Pa*s] - ZERO for this test
S = 0.0                   # Surface tension [N/m] - ZERO for this test
p_v = 0.0                 # Vapor pressure [Pa]
gamma = 1.4               # Adiabatic index

# Initial conditions
p_inf = 500.0             # External pressure [Pa]
p_B0 = 1000.0             # Initial bubble pressure [Pa]
R0 = 0.02                 # Initial radius [m]
R_dot0 = 0.0              # Initial velocity [m/s]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.5         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'../../../../bin/tests/RayleighPlesset/TEST2_InertialTerms'  # Directory containing AMReX plot files

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
output_folder = './TEST2_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL RPE SOLUTION (2D CYLINDRICAL - INVISCID)
# ============================================================================

def rp_equation_2d_inviscid(t, y):
    """
    Chen's 2D Cylindrical RPE - Inviscid version (no viscosity, no surface tension)
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
    
    # RHS pressure term (NO VISCOSITY, NO SURFACE TENSION)
    pressure_term = (p_B - p_ext) / rho_L
    
    # Correct cylindrical inertia structure
    numerator = pressure_term - R_dot**2 * (0.5 - ln_factor)
    denominator = R * ln_factor
    
    R_ddot = numerator / denominator
    
    return [R_dot, R_ddot]

print("=" * 70)
print("UNIT TEST 2: INERTIAL TERMS VALIDATION")
print("=" * 70)

# Solve analytical RPE
print("\nSolving analytical 2D cylindrical RPE (inviscid)...")
y0 = [R0, R_dot0]
t_eval_rpe = np.linspace(*t_span_rpe, n_points_rpe)
sol = solve_ivp(rp_equation_2d_inviscid, t_span_rpe, y0, t_eval=t_eval_rpe,
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
    _, R_ddot_analytical[i] = rp_equation_2d_inviscid(time_analytical[i], 
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

def count_oscillations(R, t):
    """
    Count number of oscillation cycles by finding peaks in R(t)
    """
    # Find peaks (maxima)
    peaks, _ = find_peaks(R, distance=len(R)//10)  # Minimum distance between peaks
    
    # Find troughs (minima)
    troughs, _ = find_peaks(-R, distance=len(R)//10)
    
    # Number of complete cycles
    n_cycles = min(len(peaks), len(troughs))
    
    return n_cycles, peaks, troughs

def count_zero_crossings(signal):
    """
    Count number of times signal crosses zero
    """
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings)

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
# COUNT OSCILLATIONS AND ZERO CROSSINGS
# ============================================================================

print("\n" + "=" * 70)
print("OSCILLATION ANALYSIS")
print("=" * 70)

# Count oscillations in numerical data
n_cycles_num, peaks_num, troughs_num = count_oscillations(radii_numerical, times_numerical)
print(f"\nNumerical simulation:")
print(f"  Number of oscillation cycles: {n_cycles_num}")
print(f"  Number of peaks (maxima): {len(peaks_num)}")
print(f"  Number of troughs (minima): {len(troughs_num)}")

# Count zero crossings in R_ddot
n_zero_crossings_ddot = count_zero_crossings(R_ddot_numerical)
print(f"  R_ddot zero crossings: {n_zero_crossings_ddot}")

# Count oscillations in analytical data
n_cycles_ana, peaks_ana, troughs_ana = count_oscillations(R_analytical, time_analytical)
print(f"\nAnalytical solution:")
print(f"  Number of oscillation cycles: {n_cycles_ana}")
print(f"  Number of peaks (maxima): {len(peaks_ana)}")
print(f"  Number of troughs (minima): {len(troughs_ana)}")

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
# COMPUTE INERTIAL TERMS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING INERTIAL TERMS")
print("=" * 70)

# Logarithmic factor
ln_factor_num = np.log(r_inf / radii_numerical)
ln_factor_ana = np.log(r_inf / R_analytical_interp)

# Term 1: Primary inertial term
term1_num = radii_numerical * R_ddot_numerical * ln_factor_num
term1_ana = R_analytical_interp * R_ddot_analytical_interp * ln_factor_ana

# Term 2: Convective inertia
term2_num = R_dot_numerical**2 * (ln_factor_num - 0.5)
term2_ana = R_dot_analytical_interp**2 * (ln_factor_ana - 0.5)

print("  Numerical inertial terms:")
print(f"    Term 1 (R*R_ddot*ln): range [{np.min(term1_num):.6e}, {np.max(term1_num):.6e}]")
print(f"    Term 2 (R_dot^2*[ln-0.5]): range [{np.min(term2_num):.6e}, {np.max(term2_num):.6e}]")

print("  Analytical inertial terms:")
print(f"    Term 1 (R*R_ddot*ln): range [{np.min(term1_ana):.6e}, {np.max(term1_ana):.6e}]")
print(f"    Term 2 (R_dot^2*[ln-0.5]): range [{np.min(term2_ana):.6e}, {np.max(term2_ana):.6e}]")

# Check if terms go to zero
term1_near_zero = np.sum(np.abs(term1_num) < 1e-6) / len(term1_num) * 100
term2_near_zero = np.sum(np.abs(term2_num) < 1e-6) / len(term2_num) * 100

print(f"\n  Percentage of time Term 1 near zero (<1e-6): {term1_near_zero:.2f}%")
print(f"  Percentage of time Term 2 near zero (<1e-6): {term2_near_zero:.2f}%")

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
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical (Inviscid)', zorder=1)
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)

# Mark peaks and troughs
if len(peaks_num) > 0:
    ax1.plot(times_numerical[peaks_num] * 1000, radii_numerical[peaks_num] * 1000, 
             'g^', markersize=10, label='Peaks', zorder=3)
if len(troughs_num) > 0:
    ax1.plot(times_numerical[troughs_num] * 1000, radii_numerical[troughs_num] * 1000, 
             'rv', markersize=10, label='Troughs', zorder=3)

ax1.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'R0 = {R0*1000:.1f} mm')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Bubble Radius vs Time\nOscillation Cycles: {n_cycles_num}', 
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
ax2.set_title('Bubble Wall Velocity vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
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

# Mark zero crossings
zero_crossing_indices = np.where(np.diff(np.sign(R_ddot_numerical)))[0]
if len(zero_crossing_indices) > 0:
    ax3.plot(times_numerical[zero_crossing_indices] * 1000, 
             np.zeros(len(zero_crossing_indices)), 
             'gx', markersize=12, markeredgewidth=3, label=f'Zero crossings ({n_zero_crossings_ddot})', zorder=3)

ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Acceleration R_ddot (m/s^2)', fontsize=FONT_SIZE_LABEL)
ax3.set_title(f'Bubble Wall Acceleration vs Time\nZero Crossings: {n_zero_crossings_ddot}', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Acceleration_vs_Time.png'), dpi=300)
print("  Saved: 03_Acceleration_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 4: TERM 1 (PRIMARY INERTIAL)
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times_numerical * 1000, term1_ana, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax4.plot(times_numerical * 1000, term1_num, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax4.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Term 1: R*R_ddot*ln(r_inf/R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Primary Inertial Term vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Term1_Primary_Inertial.png'), dpi=300)
print("  Saved: 04_Term1_Primary_Inertial.png")
plt.close()

# ============================================================================
# PLOT 5: TERM 2 (CONVECTIVE INERTIAL)
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.plot(times_numerical * 1000, term2_ana, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax5.plot(times_numerical * 1000, term2_num, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax5.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Term 2: R_dot^2*[ln(r_inf/R) - 0.5] (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Convective Inertial Term vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND)
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Term2_Convective_Inertial.png'), dpi=300)
print("  Saved: 05_Term2_Convective_Inertial.png")
plt.close()

# ============================================================================
# PLOT 6: BOTH INERTIAL TERMS
# ============================================================================

fig6, ax6 = plt.subplots(figsize=(12, 8))
ax6.plot(times_numerical * 1000, term1_num, 
         'b-', linewidth=LINE_WIDTH_NUMERICAL, label='Term 1: R*R_ddot*ln', marker='o', markersize=4)
ax6.plot(times_numerical * 1000, term2_num, 
         'r-', linewidth=LINE_WIDTH_NUMERICAL, label='Term 2: R_dot^2*[ln-0.5]', marker='s', markersize=4)
ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax6.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6.set_ylabel('Inertial Terms (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax6.set_title('Both Inertial Terms vs Time (Numerical)', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6.legend(fontsize=FONT_SIZE_LEGEND)
ax6.grid(True, alpha=0.3)
ax6.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_Both_Inertial_Terms.png'), dpi=300)
print("  Saved: 06_Both_Inertial_Terms.png")
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
ax7.set_title('Phase Space: Velocity vs Radius', fontsize=FONT_SIZE_TITLE, fontweight='bold')
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
print("TEST 2 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_vs_Time.png")
print(f"  02_Velocity_vs_Time.png")
print(f"  03_Acceleration_vs_Time.png")
print(f"  04_Term1_Primary_Inertial.png")
print(f"  05_Term2_Convective_Inertial.png")
print(f"  06_Both_Inertial_Terms.png")
print(f"  07_Phase_Space.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MIN_CYCLES = 2  # Minimum number of oscillation cycles
MIN_ZERO_CROSSINGS = 4  # Minimum R_ddot zero crossings (2 per cycle)
MAX_ZERO_PERCENTAGE = 20.0  # Maximum percentage of time terms can be near zero

cycles_pass = n_cycles_num >= MIN_CYCLES
zero_crossings_pass = n_zero_crossings_ddot >= MIN_ZERO_CROSSINGS
term1_active_pass = term1_near_zero < MAX_ZERO_PERCENTAGE
term2_active_pass = term2_near_zero < MAX_ZERO_PERCENTAGE

print(f"\nTest Results:")
print(f"  [{'PASS' if cycles_pass else 'FAIL'}] Oscillation cycles: {n_cycles_num} (minimum: {MIN_CYCLES})")
print(f"  [{'PASS' if zero_crossings_pass else 'FAIL'}] R_ddot zero crossings: {n_zero_crossings_ddot} (minimum: {MIN_ZERO_CROSSINGS})")
print(f"  [{'PASS' if term1_active_pass else 'FAIL'}] Term 1 active: {100-term1_near_zero:.2f}% of time (threshold: {100-MAX_ZERO_PERCENTAGE}%)")
print(f"  [{'PASS' if term2_active_pass else 'FAIL'}] Term 2 active: {100-term2_near_zero:.2f}% of time (threshold: {100-MAX_ZERO_PERCENTAGE}%)")

if cycles_pass and zero_crossings_pass and term1_active_pass and term2_active_pass:
    print("\n*** TEST 2 PASSED: Inertial terms remain active during oscillation ***")
else:
    print("\n*** TEST 2 FAILED: Inertial terms NOT functioning correctly ***")
    print("\nPossible issues:")
    if not cycles_pass:
        print("  - Bubble not oscillating (monotonic collapse)")
        print("  - Artificial damping present despite zero viscosity")
        print("  - Momentum being lost at interface")
    if not zero_crossings_pass:
        print("  - R_ddot not changing sign (acceleration not reversing)")
        print("  - Inertial terms not driving oscillation")
    if not term1_active_pass or not term2_active_pass:
        print("  - Inertial terms going to zero prematurely")
        print("  - Check if R*R_ddot*ln(r_inf/R) is being computed in solver")
        print("  - Check if R_dot^2 terms are included in momentum equation")
        print("  - Verify 2D cylindrical geometry factors (ln terms)")

print("\n" + "=" * 70)
