# -*- coding: utf-8 -*-
"""
===============================================================================
OSCILLATING ELLIPTICAL DROPLET ANALYSIS SCRIPT
===============================================================================
PURPOSE:
    Extract droplet shape evolution from AMReX simulation data by tracking 
    the eta=0.5 contour and compare oscillation frequency against analytical 
    Lamb frequency for mode-2 elliptical oscillations.

FEATURES:
    - Extracts droplet semi-major and semi-minor axes from eta field
    - Performs FFT analysis to determine oscillation frequency
    - Compares numerical frequency vs analytical Lamb frequency
    - Generates time-series plots, FFT spectrum, and comparison plots
    - Saves outputs as PNG and EPS

INPUTS:
    - AMReX plot files from oscillating droplet simulation
    - Physical parameters matching simulation input file

OUTPUTS:
    - Droplet axis evolution plot (semi-major and semi-minor axes)
    - FFT spectrum showing dominant frequency
    - Frequency comparison (numerical vs analytical)
    - Contour shape evolution visualization

ANALYTICAL SOLUTION:
    Lamb frequency for mode-2 elliptical oscillations:
    f = (1 / 2*pi) * sqrt(8 * sigma / (rho * R^3))
    
    where:
    - sigma = surface tension [N/m]
    - rho = droplet density [kg/m^3]
    - R = equilibrium droplet radius [m]

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH YOUR SIMULATION INPUT FILE)
rho_water = 1000.0         # Water droplet density [kg/m^3]
rho_air = 1.225            # Air density [kg/m^3]
sigma = 72.8e-3            # Surface tension [N/m]
R0 = 0.001                 # Initial droplet radius [m] = 1 mm

# Droplet center location
droplet_center_x = 0.0     # X-coordinate of droplet center [m]
droplet_center_y = 0.0     # Y-coordinate of droplet center [m]

# Eta contour value for interface tracking
eta_contour = 0.5          # Interface location (0.5 = midpoint)

# File paths
amrex_output_dir = r'../../../bin/tests/FlowPatel/Test1_OscilatingDroplet'  # Directory containing AMReX plot files

# FFT parameters
fft_window = 'hann'        # Window function: 'hann', 'hamming', 'blackman', or None

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './Elliptical_Droplet_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL LAMB FREQUENCY FOR MODE-2 ELLIPTICAL OSCILLATIONS
# ============================================================================

def calculate_lamb_frequency_mode2(sigma, rho, R):
    """
    Calculate analytical Lamb frequency for mode-2 elliptical oscillations.
    
    Formula: f = (1 / 2*pi) * sqrt(8 * sigma / (rho * R^3))
    
    Parameters:
    -----------
    sigma : float
        Surface tension [N/m]
    rho : float
        Droplet density [kg/m^3]
    R : float
        Equilibrium droplet radius [m]
    
    Returns:
    --------
    f : float
        Oscillation frequency [Hz]
    """
    omega_squared = 8.0 * sigma / (rho * R**3)
    omega = np.sqrt(omega_squared)
    f = omega / (2.0 * np.pi)
    return f

print("=" * 70)
print("OSCILLATING ELLIPTICAL DROPLET ANALYSIS")
print("=" * 70)

# Calculate analytical frequency
f_analytical = calculate_lamb_frequency_mode2(sigma, rho_water, R0)
print(f"\nAnalytical Lamb Frequency (Mode-2):")
print(f"  f_analytical = {f_analytical:.4f} Hz")
print(f"  Period = {1.0/f_analytical*1000:.4f} ms")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def extract_droplet_axes_from_eta(ds, center_x, center_y, eta_value=0.5):
    """
    Extract droplet semi-major and semi-minor axes by finding the eta=0.5 contour.
    
    Parameters:
    -----------
    ds : yt dataset
        YT dataset object
    center_x : float
        X-coordinate of droplet center [m]
    center_y : float
        Y-coordinate of droplet center [m]
    eta_value : float
        Eta contour value to track (default 0.5)
    
    Returns:
    --------
    radius_x : float
        Semi-major axis in x-direction [m]
    radius_y : float
        Semi-minor axis in y-direction [m]
    contour_x : ndarray
        X-coordinates of contour points [m]
    contour_y : ndarray
        Y-coordinates of contour points [m]
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge[0])
    x_max = float(ds.domain_right_edge[0])
    y_min = float(ds.domain_left_edge[1])
    y_max = float(ds.domain_right_edge[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 512  # High resolution for accurate contour detection
    
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
    
    # Find points near eta = 0.5 contour
    eta_tolerance = 0.05
    contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) == 0:
        # If no points found, try larger tolerance
        eta_tolerance = 0.1
        contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) > 0:
        # Extract contour coordinates
        contour_x = X_grid[contour_mask]
        contour_y = Y_grid[contour_mask]
        
        # Calculate maximum extents from center
        x_distances = np.abs(contour_x - center_x)
        y_distances = np.abs(contour_y - center_y)
        
        # Semi-major and semi-minor axes
        radius_x = np.max(x_distances)
        radius_y = np.max(y_distances)
        
        return radius_x, radius_y, contour_x, contour_y
    else:
        return None, None, None, None

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
# EXTRACT DROPLET AXES FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DROPLET AXES FROM ETA FIELD")
print("=" * 70)

times_numerical = []
radii_x_numerical = []
radii_y_numerical = []
contour_data = []  # Store contour points for visualization

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Extract droplet axes from eta field
        radius_x, radius_y, contour_x, contour_y = extract_droplet_axes_from_eta(
            ds, droplet_center_x, droplet_center_y, eta_contour
        )
        
        if radius_x is not None and radius_y is not None:
            times_numerical.append(t)
            radii_x_numerical.append(radius_x)
            radii_y_numerical.append(radius_y)
            contour_data.append((contour_x, contour_y))
        else:
            print(f"  WARNING: Could not extract axes at t={t:.6e} s")
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times_numerical = np.array(times_numerical)
radii_x_numerical = np.array(radii_x_numerical)
radii_y_numerical = np.array(radii_y_numerical)

print(f"\nSuccessfully extracted {len(times_numerical)} measurements")
print(f"  Time range: [{times_numerical[0]:.6e}, {times_numerical[-1]:.6e}] s")
print(f"  X-radius range: [{np.min(radii_x_numerical)*1000:.4f}, {np.max(radii_x_numerical)*1000:.4f}] mm")
print(f"  Y-radius range: [{np.min(radii_y_numerical)*1000:.4f}, {np.max(radii_y_numerical)*1000:.4f}] mm")

# ============================================================================
# PERFORM FFT ANALYSIS ON X-DIRECTION RADIUS
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMING FFT ANALYSIS")
print("=" * 70)

# Check if we have enough data points
if len(times_numerical) < 10:
    print("ERROR: Not enough data points for FFT analysis")
    exit(1)

# Calculate time step (assume uniform sampling)
dt = np.mean(np.diff(times_numerical))
print(f"\nTime step: dt = {dt:.6e} s")

# Remove mean (detrend)
radii_x_detrended = radii_x_numerical - np.mean(radii_x_numerical)

# Apply window function if specified
if fft_window == 'hann':
    window = np.hanning(len(radii_x_detrended))
    radii_x_windowed = radii_x_detrended * window
    print(f"Applied Hann window")
elif fft_window == 'hamming':
    window = np.hamming(len(radii_x_detrended))
    radii_x_windowed = radii_x_detrended * window
    print(f"Applied Hamming window")
elif fft_window == 'blackman':
    window = np.blackman(len(radii_x_detrended))
    radii_x_windowed = radii_x_detrended * window
    print(f"Applied Blackman window")
else:
    radii_x_windowed = radii_x_detrended
    print(f"No window applied")

# Perform FFT
N = len(radii_x_windowed)
fft_values = fft(radii_x_windowed)
fft_frequencies = fftfreq(N, dt)

# Take only positive frequencies
positive_freq_mask = fft_frequencies > 0
fft_frequencies_positive = fft_frequencies[positive_freq_mask]
fft_magnitude = np.abs(fft_values[positive_freq_mask])

# Find dominant frequency
dominant_freq_index = np.argmax(fft_magnitude)
f_numerical = fft_frequencies_positive[dominant_freq_index]

print(f"\nFFT Results:")
print(f"  Dominant frequency: f_numerical = {f_numerical:.4f} Hz")
print(f"  Period: {1.0/f_numerical*1000:.4f} ms")
print(f"  Frequency resolution: {fft_frequencies_positive[1]:.4f} Hz")

# Calculate error
frequency_error_abs = np.abs(f_numerical - f_analytical)
frequency_error_rel = (frequency_error_abs / f_analytical) * 100

print(f"\nFrequency Comparison:")
print(f"  Analytical: {f_analytical:.4f} Hz")
print(f"  Numerical:  {f_numerical:.4f} Hz")
print(f"  Absolute error: {frequency_error_abs:.4f} Hz")
print(f"  Relative error: {frequency_error_rel:.2f}%")

# ============================================================================
# PLOT 1: DROPLET AXES TIME EVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: DROPLET AXES EVOLUTION")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Plot x-direction radius (semi-major axis)
ax1.plot(times_numerical * 1000, radii_x_numerical * 1000, 
         'b-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', 
         markersize=MARKER_SIZE-2, label='X-direction (Semi-major)', alpha=0.8)

# Plot y-direction radius (semi-minor axis)
ax1.plot(times_numerical * 1000, radii_y_numerical * 1000, 
         'r-', linewidth=LINE_WIDTH_NUMERICAL, marker='s', 
         markersize=MARKER_SIZE-2, label='Y-direction (Semi-minor)', alpha=0.8)

# Add initial radius line
ax1.axhline(y=R0*1000, color='gray', linestyle='--', 
            linewidth=1.5, alpha=0.5, label=f'R0 = {R0*1000:.2f} mm')

ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Droplet Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Oscillating Elliptical Droplet: Axis Evolution\n' + 
              f'rho={rho_water} kg/m^3, sigma={sigma*1000:.2f} mN/m, R0={R0*1000:.2f} mm',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Droplet_Axes_Evolution.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Droplet_Axes_Evolution.eps'))
print("  Saved: 01_Droplet_Axes_Evolution.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: FFT SPECTRUM
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: FFT SPECTRUM")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot FFT magnitude spectrum
ax2.plot(fft_frequencies_positive, fft_magnitude, 
         'b-', linewidth=LINE_WIDTH_NUMERICAL, alpha=0.8)

# Mark dominant frequency
ax2.axvline(x=f_numerical, color='red', linestyle='--', 
            linewidth=2, label=f'Numerical: {f_numerical:.4f} Hz')

# Mark analytical frequency
ax2.axvline(x=f_analytical, color='green', linestyle='--', 
            linewidth=2, label=f'Analytical: {f_analytical:.4f} Hz')

ax2.set_xlabel('Frequency (Hz)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('FFT Magnitude', fontsize=FONT_SIZE_LABEL)
ax2.set_title('FFT Spectrum: X-Direction Radius Oscillation',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

# Zoom to relevant frequency range (0 to 3x analytical frequency)
ax2.set_xlim(0, 3 * f_analytical)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_FFT_Spectrum.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_FFT_Spectrum.eps'))
print("  Saved: 02_FFT_Spectrum.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: FREQUENCY COMPARISON BAR CHART
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: FREQUENCY COMPARISON")
print("=" * 70)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 3a: Bar chart comparison
frequencies = [f_analytical, f_numerical]
labels = ['Analytical\n(Lamb Mode-2)', 'Numerical\n(FFT)']
colors = ['green', 'red']

bars = ax3a.bar(labels, frequencies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, freq in zip(bars, frequencies):
    height = bar.get_height()
    ax3a.text(bar.get_x() + bar.get_width()/2., height,
              f'{freq:.4f} Hz',
              ha='center', va='bottom', fontsize=FONT_SIZE_LEGEND, fontweight='bold')

ax3a.set_ylabel('Frequency (Hz)', fontsize=FONT_SIZE_LABEL)
ax3a.set_title('Frequency Comparison',
               fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3a.grid(True, alpha=0.3, axis='y')
ax3a.tick_params(labelsize=FONT_SIZE_TICK)

# Subplot 3b: Error metrics
error_labels = ['Absolute\nError (Hz)', 'Relative\nError (%)']
error_values = [frequency_error_abs, frequency_error_rel]
error_colors = ['orange', 'purple']

bars_error = ax3b.bar(error_labels, error_values, color=error_colors, 
                       alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars_error, error_values):
    height = bar.get_height()
    ax3b.text(bar.get_x() + bar.get_width()/2., height,
              f'{val:.4f}',
              ha='center', va='bottom', fontsize=FONT_SIZE_LEGEND, fontweight='bold')

ax3b.set_ylabel('Error Magnitude', fontsize=FONT_SIZE_LABEL)
ax3b.set_title('Frequency Error Metrics',
               fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3b.grid(True, alpha=0.3, axis='y')
ax3b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Frequency_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Frequency_Comparison.eps'))
print("  Saved: 03_Frequency_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: CONTOUR SHAPE EVOLUTION (SELECTED TIMESTEPS)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: CONTOUR SHAPE EVOLUTION")
print("=" * 70)

# Select 6 evenly spaced timesteps to visualize
n_snapshots = min(6, len(contour_data))
snapshot_indices = np.linspace(0, len(contour_data)-1, n_snapshots, dtype=int)

fig4, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, snapshot_idx in enumerate(snapshot_indices):
    ax = axes[idx]
    
    contour_x, contour_y = contour_data[snapshot_idx]
    t = times_numerical[snapshot_idx]
    
    # Plot contour
    ax.scatter(contour_x * 1000, contour_y * 1000, 
               c='blue', s=10, alpha=0.6)
    
    # Plot center
    ax.plot(droplet_center_x * 1000, droplet_center_y * 1000, 
            'rx', markersize=10, markeredgewidth=2)
    
    # Add circle for reference (initial radius)
    circle = plt.Circle((droplet_center_x * 1000, droplet_center_y * 1000), 
                        R0 * 1000, color='gray', fill=False, 
                        linestyle='--', linewidth=1.5, alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_xlabel('X (mm)', fontsize=FONT_SIZE_LABEL-2)
    ax.set_ylabel('Y (mm)', fontsize=FONT_SIZE_LABEL-2)
    ax.set_title(f't = {t*1000:.3f} ms', fontsize=FONT_SIZE_LEGEND, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=FONT_SIZE_TICK-2)

fig4.suptitle('Droplet Contour Evolution (eta = 0.5)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Contour_Evolution.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Contour_Evolution.eps'))
print("  Saved: 04_Contour_Evolution.png/.eps")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  - 01_Droplet_Axes_Evolution.png/.eps")
print(f"  - 02_FFT_Spectrum.png/.eps")
print(f"  - 03_Frequency_Comparison.png/.eps")
print(f"  - 04_Contour_Evolution.png/.eps")

print(f"\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"\nPhysical Parameters:")
print(f"  Droplet density: {rho_water} kg/m^3")
print(f"  Surface tension: {sigma*1000:.2f} mN/m")
print(f"  Initial radius: {R0*1000:.2f} mm")

print(f"\nFrequency Results:")
print(f"  Analytical (Lamb Mode-2): {f_analytical:.4f} Hz")
print(f"  Numerical (FFT):          {f_numerical:.4f} Hz")
print(f"  Absolute error:           {frequency_error_abs:.4f} Hz")
print(f"  Relative error:           {frequency_error_rel:.2f}%")

print("\n" + "=" * 70)
