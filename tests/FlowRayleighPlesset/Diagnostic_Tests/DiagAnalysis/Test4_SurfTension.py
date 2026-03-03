# -*- coding: utf-8 -*-

"""
===============================================================================
UNIT TEST 4 ANALYSIS: SURFACE TENSION TERM VALIDATION
===============================================================================

PURPOSE:
    Validate that surface tension term is properly implemented through
    capillary oscillation of a bubble with no pressure difference.
    
    Key surface tension term in 2D cylindrical RPE:
    - Surface tension: -sigma/(rho_L*R)
    
    Surface tension force (CSF model):
    - F_ST = sigma * kappa * delta * n
    - where kappa = 1/R for 2D circular bubble

VALIDATION CHECKS:
    1. Verify bubble oscillates (capillary oscillation)
    2. Check oscillation frequency matches theory
    3. Compare numerical vs analytical surface tension term
    4. Validate curvature: kappa = 1/R
    5. Verify surface tension provides restoring force

EXPECTED BEHAVIOR:
    - Capillary oscillation around equilibrium
    - Natural frequency: omega_0 = sqrt(2*sigma/(rho_L*R0^3))
    - For R0=0.02m, sigma=7.28, rho=10: omega ~ 6.77 rad/s, T ~ 0.93 ms
    - Multiple oscillation cycles over 50 ms
    - Surface tension term proportional to 1/R

FAILURE MODES:
    - No oscillation -> surface tension not applied
    - Wrong frequency -> surface tension magnitude incorrect
    - Surface tension term zero -> not coupled to momentum
    - Curvature wrong -> interface geometry not computed correctly

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH TEST4 INPUT FILE)
rho_L = 10.0              # Liquid density [kg/m^3]
mu_L = 0.01               # Dynamic viscosity [Pa*s] - small for this test
S = 7.28                  # Surface tension [N/m] - ACTIVE for this test
p_v = 0.0                 # Vapor pressure [Pa]
gamma = 1.4               # Adiabatic index

# Initial conditions
p_inf = 500.0             # External pressure [Pa]
p_B0 = 500.0              # Initial bubble pressure [Pa] - EQUAL to p_inf
R0 = 0.02                 # Initial radius [m]
R_dot0 = 0.0              # Initial velocity [m/s]

# Bubble center location
bubble_center_x = 0.0     # X-coordinate of bubble center [m]
bubble_center_y = 0.0     # Y-coordinate of bubble center [m]

# Eta contour value for interface tracking
eta_contour = 0.5         # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'./tests/RayleighPlesset/TEST4_SurfaceTension'  # Directory containing AMReX plot files

# Analytical RPE solver parameters
r_inf = 5.0 * R0          # Far-field boundary for RPE
t_span_rpe = (0, 0.05)    # Time span for RPE integration [s]
n_points_rpe = 10000      # Number of time points for RPE

# Finite difference parameters
FD_METHOD = 'central'     # 'central' or 'forward' for derivatives
FD_SMOOTHING = True       # Apply Savitzky-Golay smoothing to derivatives
FD_WINDOW = 7             # Window size for smoothing (must be odd)
FD_POLYORDER = 3          # Polynomial order for smoothing

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
output_folder = './TEST4_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# THEORETICAL CAPILLARY FREQUENCY
# ============================================================================

# Natural frequency for 2D capillary oscillation
# omega_0 = sqrt(2*sigma/(rho_L*R0^3))
omega_theory = np.sqrt(2 * S / (rho_L * R0**3))
period_theory = 2 * np.pi / omega_theory
freq_theory = 1.0 / period_theory

print("=" * 70)
print("UNIT TEST 4: SURFACE TENSION TERM VALIDATION")
print("=" * 70)
print(f"\nTheoretical Capillary Oscillation:")
print(f"  Natural frequency: omega_0 = {omega_theory:.4f} rad/s")
print(f"  Period: T = {period_theory*1000:.4f} ms")
print(f"  Frequency: f = {freq_theory:.4f} Hz")

# ============================================================================
# ANALYTICAL RPE SOLUTION (2D CYLINDRICAL - SURFACE TENSION)
# ============================================================================

def rp_equation_2d_surface_tension(t, y):
    """
    Chen's 2D Cylindrical RPE - Surface tension version (small viscosity)
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
    
    # Gas pressure (2D exponent) - equal to p_inf initially
    p_B = p_v + (p_B0 - p_v) * (R0 / R)**(2 * gamma)
    
    # External pressure
    p_ext = p_inf
    
    # RHS pressure term (WITH SURFACE TENSION, SMALL VISCOSITY)
    pressure_term = (
        p_B
        - p_ext
        - 4 * mu_L * R_dot / R
        - S / R
    ) / rho_L
    
    # Correct cylindrical inertia structure
    numerator = pressure_term - R_dot**2 * (0.5 - ln_factor)
    denominator = R * ln_factor
    
    R_ddot = numerator / denominator
    
    return [R_dot, R_ddot]

# Solve analytical RPE
print("\nSolving analytical 2D cylindrical RPE (surface tension)...")
y0 = [R0 * 1.05, R_dot0]  # Start with 5% perturbation to initiate oscillation
t_eval_rpe = np.linspace(*t_span_rpe, n_points_rpe)
sol = solve_ivp(rp_equation_2d_surface_tension, t_span_rpe, y0, t_eval=t_eval_rpe,
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
    _, R_ddot_analytical[i] = rp_equation_2d_surface_tension(time_analytical[i], 
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
            kappa_avg = np.mean(kappa_filtered)
            return kappa_avg
        else:
            return None
        
    except Exception as e:
        print(f"  WARNING: Could not extract kappaAvg: {e}")
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
    peaks, _ = find_peaks(R, distance=len(R)//10)
    
    # Find troughs (minima)
    troughs, _ = find_peaks(-R, distance=len(R)//10)
    
    # Number of complete cycles
    n_cycles = min(len(peaks), len(troughs))
    
    return n_cycles, peaks, troughs

def compute_fft_frequency(t, signal):
    """
    Compute dominant frequency using FFT
    """
    # Remove mean
    signal_centered = signal - np.mean(signal)
    
    # Compute FFT
    N = len(signal_centered)
    dt = np.mean(np.diff(t))
    yf = fft(signal_centered)
    xf = fftfreq(N, dt)
    
    # Only positive frequencies
    positive_freq_mask = xf > 0
    xf_positive = xf[positive_freq_mask]
    yf_positive = np.abs(yf[positive_freq_mask])
    
    # Find dominant frequency
    dominant_idx = np.argmax(yf_positive)
    dominant_freq = xf_positive[dominant_idx]
    dominant_omega = 2 * np.pi * dominant_freq
    
    return dominant_freq, dominant_omega, xf_positive, yf_positive

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
kappa_numerical = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        radius = extract_bubble_radius_from_eta(ds, bubble_center_x, bubble_center_y, eta_contour)
        
        if radius is not None:
            # Calculate analytical curvature for filtering
            kappa_ana = 1.0 / radius
            
            # Extract average curvature with filtering
            kappa_avg = extract_kappa_avg_filtered(ds, kappa_ana, KAPPA_FILTER_PERCENT)
            
            times_numerical.append(t)
            radii_numerical.append(radius)
            kappa_numerical.append(kappa_avg if kappa_avg is not None else np.nan)
        else:
            print(f"  WARNING: Could not extract radius at t={t:.6e} s")
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times_numerical = np.array(times_numerical)
radii_numerical = np.array(radii_numerical)
kappa_numerical = np.array(kappa_numerical)

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
# COUNT OSCILLATIONS AND COMPUTE FREQUENCY
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

# Compute frequency via FFT
if len(times_numerical) > 10:
    freq_num, omega_num, xf, yf = compute_fft_frequency(times_numerical, radii_numerical)
    print(f"\nFFT Analysis:")
    print(f"  Dominant frequency: f = {freq_num:.4f} Hz")
    print(f"  Dominant angular frequency: omega = {omega_num:.4f} rad/s")
    print(f"  Theoretical frequency: f = {freq_theory:.4f} Hz")
    print(f"  Theoretical angular frequency: omega = {omega_theory:.4f} rad/s")
    print(f"  Frequency error: {abs(freq_num - freq_theory) / freq_theory * 100:.2f}%")
else:
    freq_num = 0
    omega_num = 0
    xf = np.array([])
    yf = np.array([])

# Count oscillations in analytical data
n_cycles_ana, peaks_ana, troughs_ana = count_oscillations(R_analytical, time_analytical)
print(f"\nAnalytical solution:")
print(f"  Number of oscillation cycles: {n_cycles_ana}")

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
# COMPUTE SURFACE TENSION TERM
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING SURFACE TENSION TERM")
print("=" * 70)

# Surface tension term: -sigma/(rho_L*R)
surface_tension_term_num = -S / (rho_L * radii_numerical)
surface_tension_term_ana = -S / (rho_L * R_analytical_interp)

print("  Numerical surface tension term:")
print(f"    Range: [{np.min(surface_tension_term_num):.6e}, {np.max(surface_tension_term_num):.6e}] m^2/s^2")
print(f"    Mean: {np.mean(surface_tension_term_num):.6e} m^2/s^2")

print("  Analytical surface tension term:")
print(f"    Range: [{np.min(surface_tension_term_ana):.6e}, {np.max(surface_tension_term_ana):.6e}] m^2/s^2")
print(f"    Mean: {np.mean(surface_tension_term_ana):.6e} m^2/s^2")

# Compute errors
st_abs_error = np.abs(surface_tension_term_num - surface_tension_term_ana)
st_rel_error = np.abs((surface_tension_term_num - surface_tension_term_ana) / surface_tension_term_ana) * 100

print(f"\nSurface tension term error:")
print(f"  Max absolute error: {np.max(st_abs_error):.6e} m^2/s^2")
print(f"  Mean absolute error: {np.mean(st_abs_error):.6e} m^2/s^2")
print(f"  Max relative error: {np.max(st_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(st_rel_error):.4f}%")

# ============================================================================
# COMPUTE CURVATURE
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING CURVATURE")
print("=" * 70)

# Analytical curvature for 2D circle: kappa = 1/R
kappa_analytical = 1.0 / radii_numerical

# Filter valid kappa values
valid_kappa = ~np.isnan(kappa_numerical)

if np.sum(valid_kappa) > 0:
    kappa_abs_error = np.abs(kappa_numerical[valid_kappa] - kappa_analytical[valid_kappa])
    kappa_rel_error = (kappa_abs_error / kappa_analytical[valid_kappa]) * 100
    
    print(f"\nCurvature error:")
    print(f"  Max absolute error: {np.max(kappa_abs_error):.6e} 1/m")
    print(f"  Mean absolute error: {np.mean(kappa_abs_error):.6e} 1/m")
    print(f"  Max relative error: {np.max(kappa_rel_error):.4f}%")
    print(f"  Mean relative error: {np.mean(kappa_rel_error):.4f}%")
else:
    print("\nWARNING: No valid curvature data extracted")
    kappa_abs_error = np.array([])
    kappa_rel_error = np.array([])

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
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
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
ax1.set_title(f'Capillary Oscillation: Radius vs Time\nCycles: {n_cycles_num}, Theory: T={period_theory*1000:.2f} ms', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_vs_Time.png'), dpi=300)
print("  Saved: 01_Radius_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: FFT SPECTRUM
# ============================================================================

if len(xf) > 0:
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(xf, yf, 'b-', linewidth=LINE_WIDTH_NUMERICAL)
    ax2.axvline(x=freq_theory, color='r', linestyle='--', linewidth=2, 
                label=f'Theory: f={freq_theory:.4f} Hz')
    if freq_num > 0:
        ax2.axvline(x=freq_num, color='g', linestyle='--', linewidth=2, 
                    label=f'Numerical: f={freq_num:.4f} Hz')
    ax2.set_xlabel('Frequency (Hz)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Amplitude', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('FFT Spectrum of Radius Oscillation', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.legend(fontsize=FONT_SIZE_LEGEND)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    ax2.set_xlim([0, min(50, np.max(xf))])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_FFT_Spectrum.png'), dpi=300)
    print("  Saved: 02_FFT_Spectrum.png")
    plt.close()

# ============================================================================
# PLOT 3: SURFACE TENSION TERM VS TIME
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(times_numerical * 1000, surface_tension_term_ana, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical', zorder=1)
ax3.plot(times_numerical * 1000, surface_tension_term_num, 
         'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7, zorder=2)
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Surface Tension Term: -sigma/(rho*R) (m^2/s^2)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Surface Tension Term vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Surface_Tension_Term.png'), dpi=300)
print("  Saved: 03_Surface_Tension_Term.png")
plt.close()

# ============================================================================
# PLOT 4: CURVATURE VS TIME
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times_numerical * 1000, kappa_analytical, 
         'b-', linewidth=LINE_WIDTH_ANALYTICAL, label='Analytical (1/R)', zorder=1)
ax4.plot(times_numerical[valid_kappa] * 1000, kappa_numerical[valid_kappa], 
         'ro', markersize=MARKER_SIZE, label='Numerical (kappaAvg)', alpha=0.7, zorder=2)
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Curvature kappa (1/m)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Surface Curvature vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Curvature_vs_Time.png'), dpi=300)
print("  Saved: 04_Curvature_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 5: SURFACE TENSION TERM ERROR
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.semilogy(times_numerical * 1000, st_rel_error, 
             'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Surface Tension Term Relative Error (Log Scale)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max: {np.max(st_rel_error):.4f}%\nMean: {np.mean(st_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.5)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Surface_Tension_Error.png'), dpi=300)
print("  Saved: 05_Surface_Tension_Error.png")
plt.close()

# ============================================================================
# PLOT 6: CURVATURE ERROR
# ============================================================================

if len(kappa_rel_error) > 0:
    fig6, ax6 = plt.subplots(figsize=(12, 8))
    ax6.semilogy(times_numerical[valid_kappa] * 1000, kappa_rel_error, 
                 'g-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', markersize=MARKER_SIZE-2)
    ax6.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax6.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
    ax6.set_title('Curvature Relative Error (Log Scale)', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    
    textstr = f'Max: {np.max(kappa_rel_error):.4f}%\nMean: {np.mean(kappa_rel_error):.4f}%'
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax6.text(0.05, 0.95, textstr, transform=ax6.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '06_Curvature_Error.png'), dpi=300)
    print("  Saved: 06_Curvature_Error.png")
    plt.close()

# ============================================================================
# PLOT 7: PHASE SPACE
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
ax7.set_title('Phase Space: Capillary Oscillation', fontsize=FONT_SIZE_TITLE, fontweight='bold')
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
print("TEST 4 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_vs_Time.png")
print(f"  02_FFT_Spectrum.png")
print(f"  03_Surface_Tension_Term.png")
print(f"  04_Curvature_vs_Time.png")
print(f"  05_Surface_Tension_Error.png")
print(f"  06_Curvature_Error.png")
print(f"  07_Phase_Space.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MIN_CYCLES = 2  # Minimum oscillation cycles
MAX_FREQ_ERROR = 15.0  # Maximum frequency error (%)
MAX_ST_ERROR = 10.0  # Maximum surface tension term error (%)
MAX_KAPPA_ERROR = 10.0  # Maximum curvature error (%)

cycles_pass = n_cycles_num >= MIN_CYCLES
freq_pass = abs(freq_num - freq_theory) / freq_theory * 100 < MAX_FREQ_ERROR if freq_num > 0 else False
st_error_pass = np.mean(st_rel_error) < MAX_ST_ERROR
kappa_error_pass = np.mean(kappa_rel_error) < MAX_KAPPA_ERROR if len(kappa_rel_error) > 0 else False

print(f"\nTest Results:")
print(f"  [{'PASS' if cycles_pass else 'FAIL'}] Oscillation cycles: {n_cycles_num} (minimum: {MIN_CYCLES})")
if freq_num > 0:
    print(f"  [{'PASS' if freq_pass else 'FAIL'}] Frequency error: {abs(freq_num - freq_theory) / freq_theory * 100:.2f}% (threshold: {MAX_FREQ_ERROR}%)")
else:
    print(f"  [FAIL] Frequency: Could not compute")
print(f"  [{'PASS' if st_error_pass else 'FAIL'}] Surface tension term accuracy: {np.mean(st_rel_error):.4f}% (threshold: {MAX_ST_ERROR}%)")
if len(kappa_rel_error) > 0:
    print(f"  [{'PASS' if kappa_error_pass else 'FAIL'}] Curvature accuracy: {np.mean(kappa_rel_error):.4f}% (threshold: {MAX_KAPPA_ERROR}%)")
else:
    print(f"  [FAIL] Curvature: No valid data")

if cycles_pass and freq_pass and st_error_pass and kappa_error_pass:
    print("\n*** TEST 4 PASSED: Surface tension term correctly implemented ***")
else:
    print("\n*** TEST 4 FAILED: Surface tension term NOT functioning correctly ***")
    print("\nPossible issues:")
    if not cycles_pass:
        print("  - Bubble not oscillating (surface tension not providing restoring force)")
        print("  - Surface tension not coupled to momentum equation")
    if not freq_pass:
        print("  - Wrong oscillation frequency (surface tension magnitude incorrect)")
        print("  - Check if sigma value is correct at interface")
        print("  - Verify 2D vs 3D formulation (factor of 2 difference)")
    if not st_error_pass:
        print("  - Surface tension term magnitude incorrect")
        print("  - Check if -sigma/(rho*R) is being computed correctly")
        print("  - Verify surface tension force distribution (CSF model)")
    if not kappa_error_pass:
        print("  - Curvature not computed correctly")
        print("  - Check if kappa = 1/R for 2D circular bubble")
        print("  - Verify interface geometry calculation")

print("\n" + "=" * 70)
