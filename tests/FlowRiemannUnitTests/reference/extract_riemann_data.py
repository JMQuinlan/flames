"""
Extract 1D slice from AMReX data using yt and compare with exact solution
"""

import yt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# Configuration
case_name = 'Toro1a'
group_name = 'Toro_case_1_a'
hdf5_file = f'{case_name}.hdf5'
amrex_output_dir = fr'..\..\..\bin\tests\FlowRiemannUnitTests\output_{case_name}'

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
print("LOADING EXACT SOLUTION FROM HDF5")
print("="*60)

# Read exact solution
with h5py.File(hdf5_file, 'r') as f:
    print(f"Available cases: {list(f.keys())}")
    
    if group_name not in f.keys():
        print(f"Warning: '{group_name}' not found, using first available")
        group_name = list(f.keys())[0]
    
    print(f"Using group: {group_name}")
    group = f[group_name]
    
    x_exact = group['xvec'][:]
    velocity_exact = group['velocity'][:]
    pressure_exact = group['pressure'][:]
    density_exact = group['density'][:]
    internal_energy_exact = group['internal_energy'][:]

print(f"Exact solution: {len(x_exact)} points")

print(f"\nExact data ranges:")
print(f"  x: [{np.min(x_exact):.6f}, {np.max(x_exact):.6f}]")
print(f"  velocity: [{np.min(velocity_exact):.6e}, {np.max(velocity_exact):.6e}]")
print(f"  pressure: [{np.min(pressure_exact):.6e}, {np.max(pressure_exact):.6e}]")
print(f"  density: [{np.min(density_exact):.6e}, {np.max(density_exact):.6e}]")

print("\n" + "="*60)
print("CREATING COMPARISON PLOTS")
print("="*60)

# Create comparison plots
fig, axes = plt.subplots(3, 1, figsize=(12, 14))

# Velocity
axes[0].plot(x_exact, velocity_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[0].plot(x_numerical, velocity_numerical, 'r--', linewidth=1.5, 
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[0].set_xlabel('Position (x)', fontsize=12)
axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[0].set_title(f'Velocity Comparison - {case_name} at t={float(ds.current_time):.6f}s', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([x_exact[0], x_exact[-1]])

# Pressure
axes[1].plot(x_exact, pressure_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[1].plot(x_numerical, pressure_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[1].set_xlabel('Position (x)', fontsize=12)
axes[1].set_ylabel('Pressure (Pa)', fontsize=12)
axes[1].set_title('Pressure Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([x_exact[0], x_exact[-1]])

# Density
axes[2].plot(x_exact, density_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[2].plot(x_numerical, density_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[2].set_xlabel('Position (x)', fontsize=12)
axes[2].set_ylabel('Density (kg/m3)', fontsize=12)
axes[2].set_title('Density Comparison', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([x_exact[0], x_exact[-1]])

plt.tight_layout()
output_filename = f'{case_name}_comparison_yt.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nPlot saved: {output_filename}")
plt.show()

# Calculate error metrics
velocity_exact_interp = np.interp(x_numerical, x_exact, velocity_exact)
pressure_exact_interp = np.interp(x_numerical, x_exact, pressure_exact)
density_exact_interp = np.interp(x_numerical, x_exact, density_exact)

print("\n" + "="*60)
print("ERROR METRICS")
print("="*60)
print(f"Velocity L2 error:   {np.linalg.norm(velocity_numerical - velocity_exact_interp):.6e}")
print(f"Velocity L_infinity error:   {np.max(np.abs(velocity_numerical - velocity_exact_interp)):.6e}")
print(f"\nPressure L2 error:   {np.linalg.norm(pressure_numerical - pressure_exact_interp):.6e}")
print(f"Pressure L_infinity error:   {np.max(np.abs(pressure_numerical - pressure_exact_interp)):.6e}")
print(f"\nDensity L2 error:    {np.linalg.norm(density_numerical - density_exact_interp):.6e}")
print(f"Density L_infinity error:    {np.max(np.abs(density_numerical - density_exact_interp)):.6e}")
print("="*60)

print("\n Comparison complete!")
