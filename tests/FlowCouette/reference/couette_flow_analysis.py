"""
Couette Flow Analysis Script

This script extracts velocity profiles from AMReX Couette flow simulations,
compares them with analytical solutions, and generates visualization plots.

Features:
- Extracts 1D velocity profile along y-direction at x=0
- Compares numerical solution with analytical Couette flow
- Customizable plotting parameters (fonts, titles, line weights)
- Outputs plots as both .eps and .png formats
- Optional GIF generation showing velocity profile evolution
- Error metrics (L2 and L-infinity norms)

Required inputs:
- gamma: Ratio of specific heats
- pressure: Reference pressure (Pa)
- p0_tammann: Tammann EOS pressure modification (Pa)
- density: Fluid density (kg/m³)
- tube_height: Height of the Couette flow domain (m)
- top_plate_velocity: Velocity of the moving top plate (m/s)
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters
gamma = 1.4                      # Ratio of specific heats
pressure = 1.0e5                 # Reference pressure (Pa)
p0_tammann = 0.0                 # Tammann EOS pressure modification (Pa)
density = 1.0                    # Fluid density (kg/m³)
tube_height = 0.01               # Height of the tube (m)
top_plate_velocity = 1.0         # Velocity of the top plate (m/s)

# File paths
amrex_output_dir = r'..\..\..\bin\tests\CouetteFlow\CouetteFlow'

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_EXACT = 2.5
LINE_WIDTH_NUMERICAL = 2.0
PLOT_TITLE = 'Couette Flow: Velocity Profile'

# Output settings
output_folder = './Images'
create_gif = True                # Set to True to generate GIF animation
gif_duration = 200               # Duration per frame in milliseconds
gif_filename = 'couette_evolution.gif'

# ============================================================================
# ANALYTICAL COUETTE FLOW SOLUTION
# ============================================================================

def analytical_couette_velocity(y, U_top, H):
    """
    Analytical solution for Couette flow with moving top plate.
    
    Parameters:
    -----------
    y : array-like
        Vertical positions
    U_top : float
        Velocity of the top plate
    H : float
        Height of the channel
    
    Returns:
    --------
    u : array-like
        Velocity at each y position
    """
    return U_top * (y / H)

# ============================================================================
# SETUP
# ============================================================================

print("=" * 70)
print("COUETTE FLOW ANALYSIS")
print("=" * 70)
print(f"\nPhysical Parameters:")
print(f"  Gamma:              {gamma}")
print(f"  Pressure:           {pressure:.2e} Pa")
print(f"  Tammann p0:         {p0_tammann:.2e} Pa")
print(f"  Density:            {density:.2e} kg/m³")
print(f"  Tube Height:        {tube_height:.4f} m")
print(f"  Top Plate Velocity: {top_plate_velocity:.4f} m/s")

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"\nCreated output directory: {output_folder}")

# ============================================================================
# FIND PLOT FILES
# ============================================================================

print("\n" + "=" * 70)
print("SCANNING FOR AMReX OUTPUT FILES")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and item.endswith('cell'):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No *cell directories found in {amrex_output_dir}")
    print("Looking for directories like: 00000cell, 00100cell, etc.")
    exit(1)

# Sort plot files
plot_files.sort()
print(f"\nFound {len(plot_files)} plot files")
print(f"First: {os.path.basename(plot_files[0])}")
print(f"Last:  {os.path.basename(plot_files[-1])}")

# ============================================================================
# EXTRACT DATA FROM LAST TIMESTEP
# ============================================================================

last_plot = plot_files[-1]
print("\n" + "=" * 70)
print("EXTRACTING DATA FROM LAST TIMESTEP")
print("=" * 70)
print(f"Using: {os.path.basename(last_plot)}")

# Load the dataset
ds = yt.load(last_plot)
simulation_time = float(ds.current_time)
print(f"\nSimulation time: {simulation_time:.6e} s")
print(f"Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")

# Print available fields
print(f"\nAvailable fields:")
for field in ds.field_list:
    print(f"  {field}")

# Create a ray along y-direction at x=0, z=0
# Using the same method as the Riemann script
print(f"\nExtracting 1D slice at x=0, z=0...")
y_min = float(ds.domain_left_edge[1])
y_max = float(ds.domain_right_edge[1])

# Define the ray from y=0 to y=h at x=0, z=0
ray_start = ds.arr([0.0, y_min, 0.0], 'code_length')
ray_end = ds.arr([0.0, y_max, 0.0], 'code_length')
ray = ds.ray(ray_start, ray_end)

# Sort by y coordinate (same method as Riemann script sorts by x)
sort_indices = np.argsort(ray['y'])
y_numerical = np.array(ray['y'][sort_indices])
velocity_numerical = np.array(ray['velocityx'][sort_indices])

print(f"Extracted {len(y_numerical)} points")

# Print data ranges
print(f"\nNumerical data ranges:")
print(f"  y: [{np.min(y_numerical):.6f}, {np.max(y_numerical):.6f}]")
print(f"  velocityx: [{np.min(velocity_numerical):.6e}, {np.max(velocity_numerical):.6e}]")

# ============================================================================
# COMPUTE ANALYTICAL SOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL SOLUTION")
print("=" * 70)

# Create fine grid for analytical solution
y_analytical = np.linspace(y_min, y_max, 1000)
velocity_analytical = analytical_couette_velocity(y_analytical, top_plate_velocity, tube_height)

print(f"Analytical solution computed on {len(y_analytical)} points")
print(f"  y: [{np.min(y_analytical):.6f}, {np.max(y_analytical):.6f}]")
print(f"  velocity: [{np.min(velocity_analytical):.6e}, {np.max(velocity_analytical):.6e}]")

# ============================================================================
# CREATE COMPARISON PLOT
# ============================================================================

print("\n" + "=" * 70)
print("CREATING COMPARISON PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Plot analytical solution
ax.plot(velocity_analytical, y_analytical, 'b-', 
        linewidth=LINE_WIDTH_EXACT, label='Analytical Solution', zorder=1)

# Plot numerical solution
ax.plot(velocity_numerical, y_numerical, 'r--', 
        linewidth=LINE_WIDTH_NUMERICAL, label='Numerical Solution', 
        zorder=2, alpha=0.8)

# Formatting
ax.set_xlabel('Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
ax.set_ylabel('Height (m)', fontsize=FONT_SIZE_LABEL)
ax.set_title(f'{PLOT_TITLE}\nt = {simulation_time:.6e} s', 
             fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()

# Save plots
output_filename = os.path.join(output_folder, 'couette_comparison')
plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
print(f"\nPlots saved:")
print(f"  {output_filename}.png")
print(f"  {output_filename}.eps")

plt.show()

# ============================================================================
# COMPUTE ERROR METRICS
# ============================================================================

print("\n" + "=" * 70)
print("ERROR METRICS")
print("=" * 70)

# Interpolate analytical solution to numerical grid points
velocity_analytical_interp = np.interp(y_numerical, y_analytical, velocity_analytical)

# Calculate errors
velocity_error = velocity_numerical - velocity_analytical_interp
abs_velocity_error = np.abs(velocity_error)
l2_error = np.linalg.norm(velocity_error) / np.sqrt(len(velocity_error))
linf_error = np.max(np.abs(velocity_error))
relative_l2_error = l2_error / (np.linalg.norm(velocity_analytical_interp) / np.sqrt(len(velocity_analytical_interp)))

print(f"\nVelocity Errors:")
print(f"  L2 error:          {l2_error:.6e}")
print(f"  L-infinity error:  {linf_error:.6e}")
print(f"  Relative L2 error: {relative_l2_error:.6e} ({relative_l2_error*100:.4f}%)")

# ============================================================================
# CREATE SEMILOG ERROR PLOT
# ============================================================================

print("\n" + "=" * 70)
print("CREATING SEMILOG ERROR PLOT")
print("=" * 70)

fig, ax = plt.subplots(figsize=(10, 8))

# Semilog plot of absolute error with height on x-axis
# Add small epsilon to avoid log(0)
epsilon = 1e-16
abs_error_safe = abs_velocity_error + epsilon

ax.semilogy(y_numerical, abs_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)
ax.set_xlabel('Height (m)', fontsize=FONT_SIZE_LABEL)
ax.set_ylabel('Error (m/s)', fontsize=FONT_SIZE_LABEL)
ax.set_title(f'Absolute Error \nt = {simulation_time:.6e} s', 
             fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()

# Save error plot
error_filename = os.path.join(output_folder, 'couette_error_semilog')
plt.savefig(error_filename + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(error_filename + '.eps', format='eps', bbox_inches='tight')
print(f"\nError plot saved:")
print(f"  {error_filename}.png")
print(f"  {error_filename}.eps")

plt.show()

# ============================================================================
# GENERATE GIF ANIMATION (OPTIONAL)
# ============================================================================

if create_gif:
    print("\n" + "=" * 70)
    print("GENERATING GIF ANIMATION")
    print("=" * 70)
    
    gif_frames = []
    temp_frame_files = []
    
    # Determine which timesteps to include (use all or subsample if too many)
    max_frames = 100
    if len(plot_files) > max_frames:
        step = len(plot_files) // max_frames
        selected_plots = plot_files[::step]
    else:
        selected_plots = plot_files
    
    print(f"Processing {len(selected_plots)} timesteps for GIF...")
    
    for i, plot_file in enumerate(selected_plots):
        # Load dataset
        ds_temp = yt.load(plot_file)
        time_temp = float(ds_temp.current_time)
        
        # Extract data
        ray_start_temp = ds_temp.arr([0.0, y_min, 0.0], 'code_length')
        ray_end_temp = ds_temp.arr([0.0, y_max, 0.0], 'code_length')
        ray_temp = ds_temp.ray(ray_start_temp, ray_end_temp)
        
        sort_indices_temp = np.argsort(ray_temp['y'])
        y_temp = np.array(ray_temp['y'][sort_indices_temp])
        velocity_temp = np.array(ray_temp['velocityx'][sort_indices_temp])
        
        # Create frame
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        
        ax_temp.plot(velocity_analytical, y_analytical, 'b-', 
                    linewidth=LINE_WIDTH_EXACT, label='Analytical', zorder=1)
        ax_temp.plot(velocity_temp, y_temp, 'r--', 
                    linewidth=LINE_WIDTH_NUMERICAL, label='Numerical', 
                    zorder=2, alpha=0.8)
        
        ax_temp.set_xlabel('Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
        ax_temp.set_ylabel('Height (m)', fontsize=FONT_SIZE_LABEL)
        ax_temp.set_title(f'{PLOT_TITLE}\nt = {time_temp:.6e} s', 
                         fontsize=FONT_SIZE_TITLE, fontweight='bold')
        ax_temp.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
        ax_temp.grid(True, alpha=0.3)
        ax_temp.tick_params(labelsize=FONT_SIZE_TICK)
        
        # Set consistent axis limits
        ax_temp.set_xlim([min(0, np.min(velocity_analytical)), 
                         max(top_plate_velocity * 1.1, np.max(velocity_analytical))])
        ax_temp.set_ylim([y_min, y_max])
        
        plt.tight_layout()
        
        # Save temporary frame
        temp_filename = os.path.join(output_folder, f'temp_frame_{i:04d}.png')
        plt.savefig(temp_filename, format='png', dpi=150, bbox_inches='tight')
        temp_frame_files.append(temp_filename)
        plt.close(fig_temp)
        
        if (i + 1) % 10 == 0 or i == len(selected_plots) - 1:
            print(f"  Processed {i + 1}/{len(selected_plots)} frames")
    
    # Create GIF from frames
    print("\nAssembling GIF...")
    images = [Image.open(frame) for frame in temp_frame_files]
    gif_path = os.path.join(output_folder, gif_filename)
    images[0].save(gif_path, save_all=True, append_images=images[1:], 
                   duration=gif_duration, loop=0)
    
    # Clean up temporary files
    for temp_file in temp_frame_files:
        os.remove(temp_file)
    
    print(f"\nGIF saved: {gif_path}")
    print(f"  Frames: {len(images)}")
    print(f"  Duration per frame: {gif_duration} ms")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nFinal timestep: {simulation_time:.6e} s")
print(f"Output directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  - couette_comparison.png")
print(f"  - couette_comparison.eps")
print(f"  - couette_error_semilog.png")
print(f"  - couette_error_semilog.eps")
if create_gif:
    print(f"  - {gif_filename}")
print("\n" + "=" * 70)
