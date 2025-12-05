"""
Script to compare HyperCLaw simulation output with exact Riemann solution
Reads HyperCLaw custom format (Header + Cell_D binary files)
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import struct

# Add this diagnostic before trying to read the binary file
data_dir = r'C:\Users\tryon\Desktop\BubblesResearch\flames\bin\tests\FlowRiemannUnitTests\output_Toro1a\02500cell\Level_0'
data_files = [f for f in os.listdir(data_dir) if f.startswith('Cell_D_')]
data_file = os.path.join(data_dir, data_files[0])

print(f"\n{'='*60}")
print("BINARY FILE INVESTIGATION")
print(f"{'='*60}")
print(f"File: {data_file}")
print(f"Size: {os.path.getsize(data_file)} bytes\n")

# Read first 1000 bytes as raw hex
with open(data_file, 'rb') as f:
    raw_bytes = f.read(1000)
    
print("First 100 bytes (hex):")
print(' '.join(f'{b:02x}' for b in raw_bytes[:100]))

print("\n\nTrying different data types:")

# Try reading as different types
with open(data_file, 'rb') as f:
    # Try float64 (what we've been using)
    f.seek(0)
    data_f64 = np.fromfile(f, dtype=np.float64, count=20)
    print(f"\nAs float64: {data_f64[:10]}")
    
    # Try float32
    f.seek(0)
    data_f32 = np.fromfile(f, dtype=np.float32, count=20)
    print(f"As float32: {data_f32[:10]}")
    
    # Try with different endianness
    f.seek(0)
    data_f64_be = np.fromfile(f, dtype='>f8', count=20)  # Big-endian float64
    print(f"As float64 (big-endian): {data_f64_be[:10]}")
    
    # Try int64 to see the raw bit patterns
    f.seek(0)
    data_i64 = np.fromfile(f, dtype=np.int64, count=20)
    print(f"As int64: {data_i64[:10]}")

print(f"\n{'='*60}\n")

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

def read_fab_file(fab_filename):
    """Read a FAB (Fortran Array Binary) file from BoxLib/AMReX"""
    
    with open(fab_filename, 'rb') as f:
        # Read the first line which contains all the header info
        header_line = f.readline().decode('ascii').strip()
        
        print(f"FAB Header: {header_line}\n")
        
        # Parse the header
        # Format: FAB ((type_info))((box_info)) num_components
        
        import re
        
        # Extract box dimensions: ((low_x,low_y) (high_x,high_y) (ghost_x,ghost_y))
        box_match = re.search(r'\(\((\d+),(\d+)\)\s*\((\d+),(\d+)\)\s*\((\d+),(\d+)\)\)', header_line)
        if not box_match:
            raise ValueError(f"Could not parse box dimensions from header: {header_line}")
        
        low_x, low_y = int(box_match.group(1)), int(box_match.group(2))
        high_x, high_y = int(box_match.group(3)), int(box_match.group(4))
        ghost_x, ghost_y = int(box_match.group(5)), int(box_match.group(6))
        
        nx = high_x - low_x + 1
        ny = high_y - low_y + 1
        
        print(f"Box: ({low_x},{low_y}) to ({high_x},{high_y}), ghost: ({ghost_x},{ghost_y})")
        print(f"Grid dimensions: {nx} x {ny} = {nx*ny} cells")
        
        # Extract number of components (last number in the header)
        comp_match = re.search(r'\s(\d+)\s*$', header_line)
        if not comp_match:
            raise ValueError(f"Could not parse component count from header: {header_line}")
        
        num_components = int(comp_match.group(1))
        print(f"Number of components: {num_components}")
        
        # Now we're positioned at the start of binary data
        binary_start = f.tell()
        f.seek(0, 2)  # Seek to end
        file_size = f.tell()
        binary_size = file_size - binary_start
        
        print(f"\nBinary data starts at byte: {binary_start}")
        print(f"Binary data size: {binary_size} bytes")
        
        # Read binary data
        f.seek(binary_start)
        data = np.fromfile(f, dtype=np.float32)
        
        print(f"Total values read: {len(data)}")
        
        # Calculate expected data size
        expected_size = nx * ny * num_components
        
        print(f"Expected data size: {expected_size}")
        print(f"Actual data size: {len(data)}")
        
        if len(data) >= expected_size:
            # Trim to expected size if we read too much
            data = data[:expected_size]
            
            # Reshape data - FAB format stores in Fortran column-major order
            # Data layout: all values for component 0, then all for component 1, etc.
            # Within each component: column-major (x varies fastest)
            
            # Reshape to (num_components, nx, ny) in Fortran order
            data_reshaped = data.reshape((num_components, nx, ny), order='F')
            
            # Transpose to get (ny, nx, num_components) for easier indexing
            data_reshaped = np.transpose(data_reshaped, (2, 1, 0))
            
            print(f"Successfully reshaped to: {data_reshaped.shape} (ny, nx, ncomp)")
            
            # Print statistics for first few components
            print(f"\nFirst 3 components statistics:")
            for i in range(min(3, num_components)):
                comp_data = data_reshaped[:, :, i]
                print(f"  Component {i}: min={np.min(comp_data):.6e}, max={np.max(comp_data):.6e}, mean={np.mean(comp_data):.6e}")
            
            return nx, ny, num_components, data_reshaped
        else:
            raise ValueError(f"Insufficient data in FAB file: expected {expected_size}, got {len(data)}")

def read_hyperCLaw_binary_data(data_dir, metadata):
    """Read HyperCLaw FAB format data"""
    
    # Find the Cell_D file
    data_files = [f for f in os.listdir(data_dir) if f.startswith('Cell_D_')]
    
    if not data_files:
        raise FileNotFoundError(f"No Cell_D files found in {data_dir}")
    
    data_file = os.path.join(data_dir, data_files[0])
    
    print(f"\nReading FAB file: {data_file}")
    
    # Read the FAB file
    result = read_fab_file(data_file)

    

    
    if result is None:
        raise ValueError("Failed to read FAB file")
    
    nx, ny, num_components, data_reshaped = result

    print(f"\nChecking ALL 65 components for reasonable Riemann values:")
    print(f"(Looking for: velocity ~0-1, pressure ~0.1-1, density ~0.1-1)")
    for i in range(num_components):
        comp_data = data_reshaped[:, :, i]
        non_zero = np.count_nonzero(comp_data)
        if non_zero > 0:
            min_val = np.min(comp_data)
            max_val = np.max(comp_data)
            # Check if values are in reasonable range for Riemann problem
            if -2 < min_val < 2 and 0 < max_val < 2:
                print(f"  Component {i:2d}: CANDIDATE! min={min_val:.6e}, max={max_val:.6e}")


    print(f"\nVariable name to index mapping:")
    for var_name in ['velocityx', 'pressure', 'energy_per_mass', 'density']:
        if var_name in metadata['var_names']:
            idx = metadata['var_names'].index(var_name)
            print(f"  {var_name:20s} -> index {idx}")
    
    # Verify number of components matches header
    if num_components != metadata['num_vars']:
        print(f"WARNING: Component count mismatch!")
        print(f"  FAB file: {num_components}")
        print(f"  Header: {metadata['num_vars']}")
    
    # Create variable dictionary
    var_dict = {}
    for i, var_name in enumerate(metadata['var_names']):
        if i < num_components:
            var_dict[var_name] = data_reshaped[:, :, i]
    
    # Print statistics for key variables
    print(f"\nVariable statistics:")
    for var_name in ['velocityx', 'pressure', 'energy_per_mass', 'density']:
        if var_name in var_dict:
            var_data = var_dict[var_name]
            print(f"  {var_name:20s}: min={np.min(var_data):.6e}, max={np.max(var_data):.6e}, mean={np.mean(var_data):.6e}")
    
    # Create coordinate arrays
    x = np.linspace(metadata['x_range'][0] + metadata['dx']/2, 
                    metadata['x_range'][1] - metadata['dx']/2, nx)
    y = np.linspace(metadata['y_range'][0] + metadata['dy']/2,
                    metadata['y_range'][1] - metadata['dy']/2, ny)
    
    return x, y, var_dict

def extract_1d_slice(x, y, var_dict_2d, y_slice=0.0):
    """Extract 1D slice from 2D data at specified y coordinate"""
    
    print(f"\nDEBUGGING extract_1d_slice:")
    print(f"  y array: {y}")
    print(f"  Requested y_slice: {y_slice}")
    
    # Find closest y index
    y_idx = np.argmin(np.abs(y - y_slice))
    
    print(f"  Closest y index: {y_idx}")
    print(f"  Actual y value: {y[y_idx]}")
    
    # Check if ANY row has non-zero data
    print(f"\nChecking which rows have non-zero data:")
    first_var_name = list(var_dict_2d.keys())[0]
    first_var = var_dict_2d[first_var_name]
    print(f"  2D data shape: {first_var.shape}")
    
    for row_idx in range(first_var.shape[0]):
        row_data = first_var[row_idx, :]
        non_zero = np.count_nonzero(row_data)
        if non_zero > 0:
            print(f"  Row {row_idx} (y={y[row_idx]:.6f}): {non_zero} non-zero values, min={np.min(row_data):.6e}, max={np.max(row_data):.6e}")
        else:
            print(f"  Row {row_idx} (y={y[row_idx]:.6f}): ALL ZEROS")
    
    print(f"\nExtracting 1D slice at y={y[y_idx]:.6f} (index {y_idx})")
    
    # Extract slice for each variable
    var_dict_1d = {}
    for var_name, var_data_2d in var_dict_2d.items():
        var_dict_1d[var_name] = var_data_2d[y_idx, :]
    
    # Check what we extracted
    print(f"\nExtracted 1D data:")
    for var_name in ['velocityx', 'pressure', 'energy_per_mass', 'density']:
        if var_name in var_dict_1d:
            var_data = var_dict_1d[var_name]
            non_zero = np.count_nonzero(var_data)
            print(f"  {var_name:20s}: {non_zero:4d}/{len(var_data)} non-zero, min={np.min(var_data):.6e}, max={np.max(var_data):.6e}")
    
    return x, var_dict_1d

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

# Add this right after extracting the 1D slice
print(f"\nDEBUGGING: Checking extracted 1D slice:")
for var_name in ['velocityx', 'pressure', 'energy_per_mass', 'density']:
    if var_name in var_dict_1d:
        var_data = var_dict_1d[var_name]
        non_zero = np.count_nonzero(var_data)
        print(f"  {var_name:20s}: {non_zero:4d}/{len(var_data)} non-zero, min={np.min(var_data):.6e}, max={np.max(var_data):.6e}")

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
