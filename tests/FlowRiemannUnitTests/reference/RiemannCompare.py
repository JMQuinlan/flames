"""
Script to compare HyperCLaw simulation output with exact Riemann solution
Reads HyperCLaw custom format (Header + Cell_D binary files)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import struct

header_file = r'C:\Users\tryon\Desktop\BubblesResearch\flames\bin\tests\FlowRiemannUnitTests\output_Toro1a\02500cell\Header'

with open(header_file, 'r') as f:
    lines = [line.strip() for line in f.readlines()]

print(f"Total lines in Header: {len(lines)}\n")
print("Complete Header file:")
for i, line in enumerate(lines):
    print(f"Line {i}: {line}")


def read_hyperCLaw_header(header_file):
    """Read HyperCLaw Header file to get metadata"""
    with open(header_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    idx = 0
    
    # Line 0: Format version
    format_version = lines[idx]
    idx += 1
    
    # Line 1: Number of variables
    num_vars = int(lines[idx])
    idx += 1
    
    # Lines 2-66: Variable names (65 variables)
    var_names = []
    for i in range(num_vars):
        var_names.append(lines[idx])
        idx += 1
    
    # Line 67: Dimension
    dim = int(lines[idx])
    idx += 1
    
    # Line 68: Time
    time = float(lines[idx])
    idx += 1
    
    # Line 69: Unknown (0)
    idx += 1
    
    # Line 70: X bounds
    x_bounds = [float(x) for x in lines[idx].split()]
    idx += 1
    
    # Line 71: Y bounds
    y_bounds = [float(x) for x in lines[idx].split()]
    idx += 1
    
    # Line 72: Empty line - SKIP IT
    if lines[idx] == '':
        idx += 1
    
    # Line 73: Grid info
    grid_info = lines[idx]
    idx += 1
    
    # Line 74: Timestep number
    timestep = int(lines[idx])
    idx += 1
    
    # Line 75: Grid spacing (dx dy)
    dx_info = lines[idx].split()
    dx = float(dx_info[0])
    dy = float(dx_info[1]) if len(dx_info) > 1 else dx
    idx += 1
    
    # Line 76-77: Skip (0, 0)
    idx += 2
    
    # Line 78: Time info (0 1 0.24999999999998879)
    time_info = lines[idx].split()
    idx += 1
    
    # Line 79: Number of levels (2500 - wait, this looks like another timestep?)
    # Actually this might be max timesteps, let's call it num_levels for now
    num_levels = int(lines[idx])
    idx += 1
    
    # Line 80: X range
    x_range = [float(x) for x in lines[idx].split()]
    idx += 1
    
    # Line 81: Y range
    y_range = [float(x) for x in lines[idx].split()]
    idx += 1
    
    # Line 82: Level path
    level_path = lines[idx]
    
    metadata = {
        'format_version': format_version,
        'num_vars': num_vars,
        'var_names': var_names,
        'dim': dim,
        'time': time,
        'timestep': timestep,
        'x_bounds': x_bounds,
        'y_bounds': y_bounds,
        'x_range': x_range,
        'y_range': y_range,
        'dx': dx,
        'dy': dy,
        'num_levels': num_levels,
        'grid_info': grid_info,
        'level_path': level_path
    }
    
    return metadata

def read_hyperCLaw_binary_data(data_dir, metadata):
    """Read binary Cell_D file and return all variables"""
    # Find the Cell_D file
    data_files = [f for f in os.listdir(data_dir) if f.startswith('Cell_D_')]
    
    if not data_files:
        raise FileNotFoundError(f"No Cell_D files found in {data_dir}")
    
    data_file = os.path.join(data_dir, data_files[0])
    
    # Read binary data as doubles
    with open(data_file, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)
    
    num_vars = metadata['num_vars']
    total_values = len(data)
    
    print(f"Total values read: {total_values}")
    print(f"Expected values per cell: {num_vars}")
    
    # Calculate expected number of cells
    nx = int(np.round((metadata['x_range'][1] - metadata['x_range'][0]) / metadata['dx']))
    ny = int(np.round((metadata['y_range'][1] - metadata['y_range'][0]) / metadata['dy']))
    expected_cells = nx * ny
    expected_total = expected_cells * num_vars
    
    print(f"Grid dimensions: {nx} x {ny} = {expected_cells} cells")
    print(f"Expected total values: {expected_total}")
    print(f"Actual total values: {total_values}")
    print(f"Difference: {total_values - expected_total}")
    
    # Check if there's a header/footer
    if total_values > expected_total:
        # Likely has a header - try skipping values at the beginning
        offset = total_values - expected_total
        print(f"\nTrying to skip {offset} header values...")
        data = data[offset:]
        total_values = len(data)
    
    # Now try to reshape
    num_cells = total_values // num_vars
    
    if total_values % num_vars != 0:
        print(f"\nWARNING: Data size ({total_values}) is not evenly divisible by num_vars ({num_vars})")
        print(f"Truncating to {num_cells * num_vars} values")
        data = data[:num_cells * num_vars]
    
    # Reshape to (num_cells, num_vars)
    data_reshaped = data.reshape((num_cells, num_vars))
    
    print(f"Successfully reshaped to: {data_reshaped.shape}")
    
    # Try to reshape into 2D grid
    # HyperCLaw might store data in column-major or row-major order
    # Let's try both and see which makes sense
    
    if num_cells == nx * ny:
        # Perfect match - reshape to 2D grid
        print(f"Reshaping to 2D grid: ({ny}, {nx})")
        var_dict = {}
        for i, var_name in enumerate(metadata['var_names']):
            # Try row-major (C-style) first
            var_dict[var_name] = data_reshaped[:, i].reshape((ny, nx))
    else:
        # Mismatch - use what we have
        print(f"WARNING: Cell count mismatch. Using available data.")
        # Estimate grid dimensions from available cells
        ny_actual = int(np.sqrt(num_cells * ny / nx))
        nx_actual = num_cells // ny_actual
        print(f"Adjusted grid: {ny_actual} x {nx_actual}")
        
        var_dict = {}
        for i, var_name in enumerate(metadata['var_names']):
            var_dict[var_name] = data_reshaped[:, i].reshape((ny_actual, nx_actual))
        
        ny = ny_actual
        nx = nx_actual
    
    # Create coordinate arrays
    x = np.linspace(metadata['x_range'][0] + metadata['dx']/2, 
                    metadata['x_range'][1] - metadata['dx']/2, nx)
    y = np.linspace(metadata['y_range'][0] + metadata['dy']/2,
                    metadata['y_range'][1] - metadata['dy']/2, ny)
    
    return x, y, var_dict

def extract_1d_slice(x, y, var_dict, y_slice=0.0):
    """Extract 1D slice at y=y_slice from 2D data"""
    # Find closest y index
    y_idx = np.argmin(np.abs(y - y_slice))
    
    print(f"Extracting 1D slice at y={y[y_idx]:.6f} (requested y={y_slice})")
    
    # Extract 1D slices
    slice_dict = {}
    for var_name, var_data in var_dict.items():
        slice_dict[var_name] = var_data[y_idx, :]
   
    return x, slice_dict

# Configuration
case_name = 'Toro1a'
group_name = 'Toro_case_1_a'
hdf5_file = f'./{case_name}.hdf5'
output_base = fr'C:\Users\tryon\Desktop\BubblesResearch\flames\bin\tests\FlowRiemannUnitTests\output_{case_name}'

# Read exact solution from HDF5
print(f"Reading exact solution from {hdf5_file}...")
with h5py.File(hdf5_file, 'r') as f:
    print(f"Available cases in HDF5: {list(f.keys())}")
    
    if group_name not in f.keys():
        print(f"Warning: '{group_name}' not found, using first available case")
        group_name = list(f.keys())[0]
    
    print(f"Using group: {group_name}")
    group = f[group_name]
    
    x_exact = group['xvec'][:]
    velocity_exact = group['velocity'][:]
    pressure_exact = group['pressure'][:]
    internal_energy_exact = group['internal_energy'][:]

print(f"Exact solution loaded: {len(x_exact)} points\n")

# Read HyperCLaw output
timestep_dir = '02500cell'  # Last timestep
header_file = os.path.join(output_base, timestep_dir, 'Header')
data_dir = os.path.join(output_base, timestep_dir, 'Level_0')

print(f"Reading HyperCLaw output from {timestep_dir}...")
metadata = read_hyperCLaw_header(header_file)

print(f"Simulation time: {metadata['time']}")
print(f"Timestep: {metadata['timestep']}")
print(f"Grid spacing: dx={metadata['dx']}, dy={metadata['dy']}")
print(f"Number of variables: {metadata['num_vars']}\n")

# Read binary data
x_numerical, y_numerical, var_dict_2d = read_hyperCLaw_binary_data(data_dir, metadata)

# Extract 1D slice at y=0
x_numerical, var_dict_1d = extract_1d_slice(x_numerical, y_numerical, var_dict_2d, y_slice=0.0)

# Map variable names
velocity_numerical = var_dict_1d['velocityx']
pressure_numerical = var_dict_1d['pressure']
internal_energy_numerical = var_dict_1d['energy_per_mass']

print(f"\nNumerical solution loaded: {len(x_numerical)} points\n")

 # Add this right after extracting the 1D slice (around line 220)
print("\n" + "="*60)
print("DATA DIAGNOSTICS")
print("="*60)

# Check for NaN and inf values
print(f"\nNumerical data checks:")
print(f"  Velocity - min: {np.min(velocity_numerical):.6e}, max: {np.max(velocity_numerical):.6e}")
print(f"  Velocity - NaN count: {np.sum(np.isnan(velocity_numerical))}, Inf count: {np.sum(np.isinf(velocity_numerical))}")
print(f"  Pressure - min: {np.min(pressure_numerical):.6e}, max: {np.max(pressure_numerical):.6e}")
print(f"  Pressure - NaN count: {np.sum(np.isnan(pressure_numerical))}, Inf count: {np.sum(np.isinf(pressure_numerical))}")
print(f"  Internal Energy - min: {np.min(internal_energy_numerical):.6e}, max: {np.max(internal_energy_numerical):.6e}")
print(f"  Internal Energy - NaN count: {np.sum(np.isnan(internal_energy_numerical))}, Inf count: {np.sum(np.isinf(internal_energy_numerical))}")

print(f"\nExact solution checks:")
print(f"  Velocity - min: {np.min(velocity_exact):.6e}, max: {np.max(velocity_exact):.6e}")
print(f"  Pressure - min: {np.min(pressure_exact):.6e}, max: {np.max(pressure_exact):.6e}")
print(f"  Internal Energy - min: {np.min(internal_energy_exact):.6e}, max: {np.max(internal_energy_exact):.6e}")

print(f"\nGrid comparison:")
print(f"  Numerical x-range: [{np.min(x_numerical):.6f}, {np.max(x_numerical):.6f}]")
print(f"  Exact x-range: [{np.min(x_exact):.6f}, {np.max(x_exact):.6f}]")
print(f"  Numerical points: {len(x_numerical)}")
print(f"  Exact points: {len(x_exact)}")

# Print first few values
print(f"\nFirst 5 numerical values:")
print(f"  x: {x_numerical[:5]}")
print(f"  velocity: {velocity_numerical[:5]}")
print(f"  pressure: {pressure_numerical[:5]}")
print(f"  internal_energy: {internal_energy_numerical[:5]}")

print(f"\nFirst 5 exact values:")
print(f"  x: {x_exact[:5]}")
print(f"  velocity: {velocity_exact[:5]}")
print(f"  pressure: {pressure_exact[:5]}")
print(f"  internal_energy: {internal_energy_exact[:5]}")

print("="*60 + "\n")

# Create comparison plots
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Velocity plot
axes[0].plot(x_exact, velocity_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[0].plot(x_numerical, velocity_numerical, 'r--', linewidth=1.5, 
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[0].set_xlabel('Position (x)', fontsize=12)
axes[0].set_ylabel('Velocity (m/s)', fontsize=12)
axes[0].set_title(f'Velocity Comparison - t={metadata["time"]:.6f}s', 
                  fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([x_exact[0], x_exact[-1]])

# Pressure plot
axes[1].plot(x_exact, pressure_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[1].plot(x_numerical, pressure_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[1].set_xlabel('Position (x)', fontsize=12)
axes[1].set_ylabel('Pressure (Pa)', fontsize=12)
axes[1].set_title('Pressure Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([x_exact[0], x_exact[-1]])

# Internal Energy plot
axes[2].plot(x_exact, internal_energy_exact, 'b-', linewidth=2, label='Exact Solution', zorder=1)
axes[2].plot(x_numerical, internal_energy_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[2].set_xlabel('Position (x)', fontsize=12)
axes[2].set_ylabel('Internal Energy (J/kg)', fontsize=12)
axes[2].set_title('Internal Energy Comparison', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([x_exact[0], x_exact[-1]])

plt.tight_layout()
output_filename = f'{case_name}_comparison_t{metadata["timestep"]:05d}.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Comparison plot saved as {output_filename}")
plt.show()

# Calculate error metrics by interpolating exact solution to numerical grid
velocity_exact_interp = np.interp(x_numerical, x_exact, velocity_exact)
pressure_exact_interp = np.interp(x_numerical, x_exact, pressure_exact)
internal_energy_exact_interp = np.interp(x_numerical, x_exact, internal_energy_exact)

print("\n" + "="*60)
print("ERROR METRICS")
print("="*60)
print(f"Velocity L2 error:        {np.linalg.norm(velocity_numerical - velocity_exact_interp):.6e}")
print(f"Velocity L_infinity error:        {np.max(np.abs(velocity_numerical - velocity_exact_interp)):.6e}")
print(f"\nPressure L2 error:        {np.linalg.norm(pressure_numerical - pressure_exact_interp):.6e}")
print(f"Pressure L_infinity error:        {np.max(np.abs(pressure_numerical - pressure_exact_interp)):.6e}")
print(f"\nInternal Energy L2 error: {np.linalg.norm(internal_energy_numerical - internal_energy_exact_interp):.6e}")
print(f"Internal Energy L_infinity error: {np.max(np.abs(internal_energy_numerical - internal_energy_exact_interp)):.6e}")
print("="*60)

print("\nScript completed successfully!")
