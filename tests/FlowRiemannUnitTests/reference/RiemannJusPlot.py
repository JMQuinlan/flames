"""
Extract 1D slice from AMReX data using yt and plot results
No exact solution comparison - just plots the numerical data
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

# Specify the AMReX output directory containing the *cell folders
amrex_output_dir = r'..\..\..\bin\tests\FlowRiemannUnitTests\output_Garrick_GasLiquid14'

# Specify the output folder for saving plots
output_folder = r'./Images'

# Specify the output filename (without extension)
output_filename = 'Toro1a_numerical'

# ============================================================================

print("="*60)
print("EXTRACTING DATA FROM AMReX OUTPUT USING YT")
print("="*60)

# Find the plot files in the output directory
plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and item.endswith('cell'):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No *cell directories found in {amrex_output_dir}")
    print("Looking for directories like: 00000cell, 00100cell, etc.")
    exit(1)

# Sort and use the last one
plot_files.sort()
last_plot = plot_files[-1]

print(f"\nFound {len(plot_files)} plot files")
print(f"Using last timestep: {os.path.basename(last_plot)}")

# Load the dataset
print(f"\nLoading dataset...")
ds = yt.load(last_plot)
print(f"Simulation time: {ds.current_time}")
print(f"Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")

# Get field names
print(f"\nAvailable fields:")
for field in ds.field_list:
    print(f"  {field}")

# Create a ray (1D line) through the domain at y=0
print(f"\nExtracting 1D slice at y=0...")

# Define the ray from x=-1 to x=1 at y=0, z=0
ray_start = ds.arr([-1.0, 0.0, 0.0], 'code_length')
ray_end = ds.arr([1.0, 0.0, 0.0], 'code_length')
ray = ds.ray(ray_start, ray_end)

# Sort by x coordinate
sort_indices = np.argsort(ray['x'])
x_numerical = np.array(ray['x'][sort_indices])
velocity_numerical = np.array(ray['velocityx'][sort_indices])
pressure_numerical = np.array(ray['pressure'][sort_indices])
density_numerical = np.array(ray['density'][sort_indices])

print(f"Extracted {len(x_numerical)} points")

# Print data ranges
print(f"\nNumerical data ranges:")
print(f"  x: [{np.min(x_numerical):.6f}, {np.max(x_numerical):.6f}]")
print(f"  velocity: [{np.min(velocity_numerical):.6e}, {np.max(velocity_numerical):.6e}]")
print(f"  pressure: [{np.min(pressure_numerical):.6e}, {np.max(pressure_numerical):.6e}]")
print(f"  density: [{np.min(density_numerical):.6e}, {np.max(density_numerical):.6e}]")

print("\n" + "="*60)
print("CREATING PLOTS")
print("="*60)

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Density
axes[0].set_title(f'Numerical Solution at t={float(ds.current_time):.6f}s', 
                  fontsize=14, fontweight='bold')
axes[0].plot(x_numerical, density_numerical, 'r-', linewidth=2, label='Numerical Solution')
axes[0].set_xlabel('Position (x)', fontsize=12)
axes[0].set_ylabel('Density (kg/m$^3$)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([x_numerical[0], x_numerical[-1]])

# Velocity
axes[1].plot(x_numerical, velocity_numerical, 'r-', linewidth=2, label='Numerical Solution')
axes[1].set_xlabel('Position (x)', fontsize=12)
axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([x_numerical[0], x_numerical[-1]])

# Pressure
axes[2].plot(x_numerical, pressure_numerical, 'r-', linewidth=2, label='Numerical Solution')
axes[2].set_xlabel('Position (x)', fontsize=12)
axes[2].set_ylabel('Pressure (Pa)', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([x_numerical[0], x_numerical[-1]])

plt.tight_layout()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Save plots
output_path = os.path.join(output_folder, output_filename)
plt.savefig(output_path + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_path + '.eps', format='eps', bbox_inches='tight')

print(f"\nPlots saved:")
print(f"  {output_path}.png")
print(f"  {output_path}.eps")

plt.show()

print("="*60)
print("\nPlotting complete!")
