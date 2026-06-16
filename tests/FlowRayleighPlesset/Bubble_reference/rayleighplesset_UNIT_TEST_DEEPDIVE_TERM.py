# -*- coding: utf-8 -*-

"""
===============================================================================
RAYLEIGH-PLESSET TERM-BY-TERM ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Diagnose missing oscillations in bubble dynamics by validating each term
    of the 2D cylindrical Rayleigh-Plesset equation individually.
    
    Compares analytical vs numerical values for:
    - Inertial terms: R*R_ddot*ln(r_inf/R) and R_dot^2*[ln(r_inf/R) - 1/2]
    - Pressure driving term: (p_B - p_inf)/rho_L
    - Viscous damping term: -4*mu_L*R_dot/(rho_L*R)
    - Surface tension term: -sigma/(rho_L*R)
    - Gas pressure: p_B with adiabatic compression

FEATURES:
    - Extracts R(t) from eta field
    - Computes R_dot and R_ddot via finite differences
    - Extracts p_B from pressure field at bubble center
    - Calculates all RPE terms analytically and numerically
    - Generates comparison plots for each term
    - Identifies which term(s) cause missing oscillations

INPUTS:
    - AMReX plot files from Rayleigh-Plesset simulation
    - Physical parameters matching RPE analytical solution

OUTPUTS:
    - Time series plots for R, R_dot, R_ddot
    - Term-by-term comparison plots (analytical vs numerical)
    - Error analysis plots for each term
    - Summary diagnostic report

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

# Finite difference parameters
FD_METHOD = 'central'     # 'central' or 'forward' for derivatives
FD_SMOOTHING = False      # Apply Savitzky-Golay smoothing to derivatives
FD_WINDOW = 5             # Window size for smoothing (must be odd)
FD_POLYORDER = 2          # Polynomial order for smoothing

# Pressure extraction parameters
PRESSURE_FIELD_NAME = 'pressure'  # Name of pressure field in simulation
PRESSURE_SAMPLE_RADIUS = 0.001    # Radius around bubble center to average [m]

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './RPE_TermByTerm_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL RPE SOLUTION (2D CYLINDRICAL)
# ============================================================================

def rp_equation_2d_chen(t, y):
    """
    Chen's 2D Cylindrical RPE
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
    
    # External pressure (no acoustic forcing)
    p_ext = p_inf
    
    # RHS pressure term
    pressure_term = (
        p_B
        - p_ext
        - 4 * mu_L * R_dot / R
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
print("RAYLEIGH-PLESSET TERM-BY-TERM ANALYSIS")
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

R_analytical = sol.y[0]
R_dot_analytical = sol.y[1]
time_analytical = sol.t

# Compute R_ddot analytically
R_ddot_analytical = np.zeros_like(R_analytical)
for i in range(len(time_analytical)):
    _, R_ddot_analytical[i] = rp_equation_2d_chen(time_analytical[i], 
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

def compute_derivatives(t, R, method='central', smooth=False, window=5, polyorder=2):
    """
    Compute first and second derivatives of R with respect to t.
    
    Parameters:
    -----------
    t : array
        Time values
    R : array
        Radius values
    method : str
        'central' or 'forward' finite difference
    smooth : bool
        Apply Savitzky-Golay smoothing
    window : int
        Window size for smoothing (must be odd)
    polyorder : int
        Polynomial order for smoothing
    
    Returns:
    --------
    R_dot : array
        First derivative (velocity)
    R_ddot : array
        Second derivative (acceleration)
    """
    from scipy.signal import savgol_filter
    
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
        
    elif method == 'forward':
        # Forward difference
        for i in range(n-1):
            dt = t[i+1] - t[i]
            R_dot[i] = (R[i+1] - R[i]) / dt
        R_dot[-1] = R_dot[-2]
        
        # Second derivative
        for i in range(n-2):
            dt = t[i+1] - t[i]
            R_ddot[i] = (R_dot[i+1] - R_dot[i]) / dt
        R_ddot[-2:] = R_ddot[-3]
    
    # Apply smoothing if requested
    if smooth and n > window:
        R_dot = savgol_filter(R_dot, window, polyorder)
        R_ddot = savgol_filter(R_ddot, window, polyorder)
    
    return R_dot, R_ddot

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
        
        if radius is not None:
            times_numerical.append(t)
            radii_numerical.append(radius)
            pressure_numerical.append(pressure if pressure is not None else np.nan)
        else:
            print(f"  WARNING: Could not extract radius at t={t:.6e} s")
        
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
# INTERPOLATE ANALYTICAL SOLUTION TO NUMERICAL TIME POINTS
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
# COMPUTE ALL RPE TERMS (ANALYTICAL)
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL RPE TERMS")
print("=" * 70)

# Logarithmic factor
ln_factor_ana = np.log(r_inf / R_analytical_interp)

# Gas pressure (2D adiabatic)
p_B_ana = p_v + (p_B0 - p_v) * (R0 / R_analytical_interp)**(2 * gamma)

# Term 1: Primary inertial term
term1_ana = R_analytical_interp * R_ddot_analytical_interp * ln_factor_ana

# Term 2: Convective inertia
term2_ana = R_dot_analytical_interp**2 * (ln_factor_ana - 0.5)

# Term 3: Pressure driving term
term3_ana = (p_B_ana - p_inf) / rho_L

# Term 4: Viscous damping term
term4_ana = -4 * mu_L * R_dot_analytical_interp / (rho_L * R_analytical_interp)

# Term 5: Surface tension term
term5_ana = -S / (rho_L * R_analytical_interp)

print("  Analytical terms computed:")
print(f"    Term 1 (R*R_ddot*ln): range [{np.min(term1_ana):.6e}, {np.max(term1_ana):.6e}]")
print(f"    Term 2 (R_dot^2*[ln-0.5]): range [{np.min(term2_ana):.6e}, {np.max(term2_ana):.6e}]")
print(f"    Term 3 (DeltaP/rho): range [{np.min(term3_ana):.6e}, {np.max(term3_ana):.6e}]")
print(f"    Term 4 (Viscous): range [{np.min(term4_ana):.6e}, {np.max(term4_ana):.6e}]")
print(f"    Term 5 (Surface): range [{np.min(term5_ana):.6e}, {np.max(term5_ana):.6e}]")

# ============================================================================
# COMPUTE ALL RPE TERMS (NUMERICAL)
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING NUMERICAL RPE TERMS")
print("=" * 70)

# Logarithmic factor
ln_factor_num = np.log(r_inf / radii_numerical)

# Gas pressure (use extracted pressure from simulation)
p_B_num = pressure_numerical

# Term 1: Primary inertial term
term1_num = radii_numerical * R_ddot_numerical * ln_factor_num

# Term 2: Convective inertia
term2_num = R_dot_numerical**2 * (ln_factor_num - 0.5)

# Term 3: Pressure driving term
term3_num = (p_B_num - p_inf) / rho_L

# Term 4: Viscous damping term
term4_num = -4 * mu_L * R_dot_numerical / (rho_L * radii_numerical)

# Term 5: Surface tension term
term5_num = -S / (rho_L * radii_numerical)

print("  Numerical terms computed:")
print(f"    Term 1 (R*R_ddot*ln): range [{np.nanmin(term1_num):.6e}, {np.nanmax(term1_num):.6e}]")
print(f"    Term 2 (R_dot^2*[ln-0.5]): range [{np.nanmin(term2_num):.6e}, {np.nanmax(term2_num):.6e}]")
print(f"    Term 3 (DeltaP/rho): range [{np.nanmin(term3_num):.6e}, {np.nanmax(term3_num):.6e}]")
print(f"    Term 4 (Viscous): range [{np.nanmin(term4_num):.6e}, {np.nanmax(term4_num):.6e}]")
print(f"    Term 5 (Surface): range [{np.nanmin(term5_num):.6e}, {np.nanmax(term5_num):.6e}]")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: RADIUS TIME SERIES
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(time_analytical * 1000, R_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax1.plot(times_numerical * 1000, radii_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Bubble Radius vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_TimeSeries.png'), dpi=300)
print("  Saved: 01_Radius_TimeSeries.png")
plt.close()

# ============================================================================
# PLOT 2: VELOCITY TIME SERIES
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(time_analytical * 1000, R_dot_analytical * 1000, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax2.plot(times_numerical * 1000, R_dot_numerical * 1000, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Velocity R_dot (mm/s)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Bubble Wall Velocity vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Velocity_TimeSeries.png'), dpi=300)
print("  Saved: 02_Velocity_TimeSeries.png")
plt.close()

# ============================================================================
# PLOT 3: ACCELERATION TIME SERIES
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(time_analytical * 1000, R_ddot_analytical, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax3.plot(times_numerical * 1000, R_ddot_numerical, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Acceleration R_ddot (m/s^2)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Bubble Wall Acceleration vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Acceleration_TimeSeries.png'), dpi=300)
print("  Saved: 03_Acceleration_TimeSeries.png")
plt.close()

# ============================================================================
# PLOT 4: GAS PRESSURE
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times_numerical * 1000, p_B_ana, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical p_B', zorder=1)
ax4.plot(times_numerical * 1000, p_B_num, 
         'ro', markersize=MARKER_SIZE, label='Numerical p_B', alpha=0.7, zorder=2)
ax4.axhline(y=p_inf, color='gray', linestyle='--', linewidth=1.5, 
            alpha=0.5, label=f'p_inf = {p_inf} Pa')
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Gas Pressure p_B vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Gas_Pressure.png'), dpi=300)
print("  Saved: 04_Gas_Pressure.png")
plt.close()

# ============================================================================
# PLOT 5: TERM 1 - PRIMARY INERTIAL TERM
# ============================================================================

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 12))

# Top: Comparison
ax5a.plot(times_numerical * 1000, term1_ana, 
          'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax5a.plot(times_numerical * 1000, term1_num, 
          'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax5a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5a.set_ylabel('Term 1: R*R_ddot*ln(r_inf/R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax5a.set_title('Term 1: Primary Inertial Term', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5a.legend(fontsize=FONT_SIZE_LEGEND)
ax5a.grid(True, alpha=0.3)
ax5a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Relative Error
valid = ~np.isnan(term1_num) & (np.abs(term1_ana) > 1e-12)
rel_error = np.abs((term1_num[valid] - term1_ana[valid]) / term1_ana[valid]) * 100
ax5b.semilogy(times_numerical[valid] * 1000, rel_error, 
              'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax5b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5b.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5b.set_title('Term 1: Relative Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2)
ax5b.grid(True, alpha=0.3, which='both')
ax5b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Term1_Inertial_Primary.png'), dpi=300)
print("  Saved: 05_Term1_Inertial_Primary.png")
plt.close()

# ============================================================================
# PLOT 6: TERM 2 - CONVECTIVE INERTIA
# ============================================================================

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(12, 12))

# Top: Comparison
ax6a.plot(times_numerical * 1000, term2_ana, 
          'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax6a.plot(times_numerical * 1000, term2_num, 
          'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax6a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6a.set_ylabel('Term 2: R_dot^2*[ln(r_inf/R) - 0.5] (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax6a.set_title('Term 2: Convective Inertia', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6a.legend(fontsize=FONT_SIZE_LEGEND)
ax6a.grid(True, alpha=0.3)
ax6a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Relative Error
valid = ~np.isnan(term2_num) & (np.abs(term2_ana) > 1e-12)
rel_error = np.abs((term2_num[valid] - term2_ana[valid]) / term2_ana[valid]) * 100
ax6b.semilogy(times_numerical[valid] * 1000, rel_error, 
              'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax6b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6b.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax6b.set_title('Term 2: Relative Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2)
ax6b.grid(True, alpha=0.3, which='both')
ax6b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_Term2_Inertial_Convective.png'), dpi=300)
print("  Saved: 06_Term2_Inertial_Convective.png")
plt.close()

# ============================================================================
# PLOT 7: TERM 3 - PRESSURE DRIVING
# ============================================================================

fig7, (ax7a, ax7b) = plt.subplots(2, 1, figsize=(12, 12))

# Top: Comparison
ax7a.plot(times_numerical * 1000, term3_ana, 
          'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax7a.plot(times_numerical * 1000, term3_num, 
          'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax7a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax7a.set_ylabel('Term 3: (p_B - p_inf)/rho_L (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax7a.set_title('Term 3: Pressure Driving Term', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7a.legend(fontsize=FONT_SIZE_LEGEND)
ax7a.grid(True, alpha=0.3)
ax7a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Relative Error
valid = ~np.isnan(term3_num) & (np.abs(term3_ana) > 1e-12)
rel_error = np.abs((term3_num[valid] - term3_ana[valid]) / term3_ana[valid]) * 100
ax7b.semilogy(times_numerical[valid] * 1000, rel_error, 
              'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax7b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax7b.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax7b.set_title('Term 3: Relative Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2)
ax7b.grid(True, alpha=0.3, which='both')
ax7b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '07_Term3_Pressure_Driving.png'), dpi=300)
print("  Saved: 07_Term3_Pressure_Driving.png")
plt.close()

# ============================================================================
# PLOT 8: TERM 4 - VISCOUS DAMPING
# ============================================================================

fig8, (ax8a, ax8b) = plt.subplots(2, 1, figsize=(12, 12))

# Top: Comparison
ax8a.plot(times_numerical * 1000, term4_ana, 
          'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax8a.plot(times_numerical * 1000, term4_num, 
          'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax8a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax8a.set_ylabel('Term 4: -4*mu_L*R_dot/(rho_L*R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax8a.set_title('Term 4: Viscous Damping Term', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax8a.legend(fontsize=FONT_SIZE_LEGEND)
ax8a.grid(True, alpha=0.3)
ax8a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Relative Error
valid = ~np.isnan(term4_num) & (np.abs(term4_ana) > 1e-12)
rel_error = np.abs((term4_num[valid] - term4_ana[valid]) / term4_ana[valid]) * 100
ax8b.semilogy(times_numerical[valid] * 1000, rel_error, 
              'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax8b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax8b.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax8b.set_title('Term 4: Relative Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2)
ax8b.grid(True, alpha=0.3, which='both')
ax8b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '08_Term4_Viscous_Damping.png'), dpi=300)
print("  Saved: 08_Term4_Viscous_Damping.png")
plt.close()

# ============================================================================
# PLOT 9: TERM 5 - SURFACE TENSION
# ============================================================================

fig9, (ax9a, ax9b) = plt.subplots(2, 1, figsize=(12, 12))

# Top: Comparison
ax9a.plot(times_numerical * 1000, term5_ana, 
          'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax9a.plot(times_numerical * 1000, term5_num, 
          'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax9a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax9a.set_ylabel('Term 5: -sigma/(rho_L*R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax9a.set_title('Term 5: Surface Tension Term', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax9a.legend(fontsize=FONT_SIZE_LEGEND)
ax9a.grid(True, alpha=0.3)
ax9a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Relative Error
valid = ~np.isnan(term5_num) & (np.abs(term5_ana) > 1e-12)
rel_error = np.abs((term5_num[valid] - term5_ana[valid]) / term5_ana[valid]) * 100
ax9b.semilogy(times_numerical[valid] * 1000, rel_error, 
              'k-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax9b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax9b.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax9b.set_title('Term 5: Relative Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2)
ax9b.grid(True, alpha=0.3, which='both')
ax9b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '09_Term5_Surface_Tension.png'), dpi=300)
print("  Saved: 09_Term5_Surface_Tension.png")
plt.close()

# ============================================================================
# PLOT 10: ALL TERMS COMPARISON (ANALYTICAL)
# ============================================================================

fig10, ax10 = plt.subplots(figsize=(14, 8))
ax10.plot(times_numerical * 1000, term1_ana, 'b-', linewidth=2, label='Term 1: R*R_ddot*ln')
ax10.plot(times_numerical * 1000, term2_ana, 'r-', linewidth=2, label='Term 2: R_dot^2*[ln-0.5]')
ax10.plot(times_numerical * 1000, term3_ana, 'g-', linewidth=2, label='Term 3: DeltaP/rho')
ax10.plot(times_numerical * 1000, term4_ana, 'm-', linewidth=2, label='Term 4: Viscous')
ax10.plot(times_numerical * 1000, term5_ana, 'c-', linewidth=2, label='Term 5: Surface')
ax10.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax10.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax10.set_ylabel('Term Value (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax10.set_title('All RPE Terms - Analytical', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax10.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax10.grid(True, alpha=0.3)
ax10.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '10_All_Terms_Analytical.png'), dpi=300)
print("  Saved: 10_All_Terms_Analytical.png")
plt.close()

# ============================================================================
# PLOT 11: ALL TERMS COMPARISON (NUMERICAL)
# ============================================================================

fig11, ax11 = plt.subplots(figsize=(14, 8))
ax11.plot(times_numerical * 1000, term1_num, 'b-', linewidth=2, label='Term 1: R*R_ddot*ln')
ax11.plot(times_numerical * 1000, term2_num, 'r-', linewidth=2, label='Term 2: R_dot^2*[ln-0.5]')
ax11.plot(times_numerical * 1000, term3_num, 'g-', linewidth=2, label='Term 3: DeltaP/rho')
ax11.plot(times_numerical * 1000, term4_num, 'm-', linewidth=2, label='Term 4: Viscous')
ax11.plot(times_numerical * 1000, term5_num, 'c-', linewidth=2, label='Term 5: Surface')
ax11.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax11.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax11.set_ylabel('Term Value (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax11.set_title('All RPE Terms - Numerical', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax11.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax11.grid(True, alpha=0.3)
ax11.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '11_All_Terms_Numerical.png'), dpi=300)
print("  Saved: 11_All_Terms_Numerical.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("TERM-BY-TERM ERROR SUMMARY")
print("=" * 70)

def compute_error_stats(analytical, numerical, term_name):
    """Compute and print error statistics for a term"""
    valid = ~np.isnan(numerical) & (np.abs(analytical) > 1e-12)
    if np.sum(valid) > 0:
        abs_error = np.abs(numerical[valid] - analytical[valid])
        rel_error = (abs_error / np.abs(analytical[valid])) * 100
        
        print(f"\n{term_name}:")
        print(f"  Max absolute error: {np.max(abs_error):.6e}")
        print(f"  Mean absolute error: {np.mean(abs_error):.6e}")
        print(f"  Max relative error: {np.max(rel_error):.4f}%")
        print(f"  Mean relative error: {np.mean(rel_error):.4f}%")
    else:
        print(f"\n{term_name}: No valid data points")

compute_error_stats(R_analytical_interp, radii_numerical, "Radius R")
compute_error_stats(R_dot_analytical_interp, R_dot_numerical, "Velocity R_dot")
compute_error_stats(R_ddot_analytical_interp, R_ddot_numerical, "Acceleration R_ddot")
compute_error_stats(p_B_ana, p_B_num, "Gas Pressure p_B")
compute_error_stats(term1_ana, term1_num, "Term 1 (Primary Inertial)")
compute_error_stats(term2_ana, term2_num, "Term 2 (Convective Inertial)")
compute_error_stats(term3_ana, term3_num, "Term 3 (Pressure Driving)")
compute_error_stats(term4_ana, term4_num, "Term 4 (Viscous Damping)")
compute_error_stats(term5_ana, term5_num, "Term 5 (Surface Tension)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("TERM-BY-TERM ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_TimeSeries.png")
print(f"  02_Velocity_TimeSeries.png")
print(f"  03_Acceleration_TimeSeries.png")
print(f"  04_Gas_Pressure.png")
print(f"  05_Term1_Inertial_Primary.png")
print(f"  06_Term2_Inertial_Convective.png")
print(f"  07_Term3_Pressure_Driving.png")
print(f"  08_Term4_Viscous_Damping.png")
print(f"  09_Term5_Surface_Tension.png")
print(f"  10_All_Terms_Analytical.png")
print(f"  11_All_Terms_Numerical.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC GUIDANCE")
print("=" * 70)
print("\nTo diagnose missing oscillations, examine:")
print("  1. Term 1 & 2 (Inertial): These create the 'spring-like' restoring force")
print("  2. Term 3 (Pressure): Gas compression/expansion drives oscillations")
print("  3. R_ddot (Acceleration): Should show oscillatory behavior if physics is correct")
print("\nIf numerical terms show monotonic decay while analytical shows oscillation:")
print("  -> Check if inertial terms are being computed correctly in your solver")
print("  -> Verify gas pressure coupling (adiabatic compression)")
print("  -> Check if time integration scheme preserves oscillatory dynamics")
print("\n" + "=" * 70)
