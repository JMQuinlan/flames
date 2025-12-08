"""
Track interface location (eta = 0.5) over time from HyperCLaw output
Uses yt to read the data (same method as Riemann comparison)
"""

import yt
import numpy as np
import h5py
import os
from scipy.interpolate import interp1d

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# Configuration
case_name = 'LargeBubbleRPE_2_Bar.GammaHigh.old.20250831204517'
group_name = ''
on_Incline = 0

if (on_Incline == 1):
    amrex_output_dir = fr'/mmfs1/home/ttryon/flames/bin/tests/{group_name}/{case_name}'
else:
    amrex_output_dir = fr'../../../bin/tests/{group_name}/{case_name}'

output_hdf5 = f'./{case_name}_interface_tracking.hdf5'

# Focus on bubble region (x >= 0)
x_min_focus = 0.0

print("="*60)
print(f"TRACKING INTERFACE (eta = 0.5) FOR {case_name}")
print(f"Focus region: x >= {x_min_focus}")
print("="*60)

# Find all timestep directories
timestep_dirs = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and 'cell' in item.lower():
        timestep_dirs.append(item_path)

if not timestep_dirs:
    print("ERROR: No timestep directories found")
    exit(1)

# Sort by timestep number
def extract_timestep_number(path):
    basename = os.path.basename(path)
    numeric_part = ''.join(filter(str.isdigit, basename))
    return int(numeric_part) if numeric_part else 0

timestep_dirs.sort(key=extract_timestep_number)
print(f"\nFound {len(timestep_dirs)} timesteps")

# Arrays to store results
times = []
interface_positions = []
timestep_numbers = []

print("\nProcessing timesteps...")

for i, timestep_dir in enumerate(timestep_dirs):
    try:
        timestep_num = extract_timestep_number(timestep_dir)
        
        # Load the timestep directory directly with yt (not the Header file)
        # yt will automatically find and read the data files
        ds = yt.load(timestep_dir)
        
        # Get domain bounds
        x_left = float(ds.domain_left_edge[0])
        x_right = float(ds.domain_right_edge[0])
        
        # Create a ray at y=0, focusing on x >= x_min_focus
        ray_start = ds.arr([max(x_left, x_min_focus), 0.0, 0.0], 'code_length')
        ray_end = ds.arr([x_right, 0.0, 0.0], 'code_length')
        
        ray = ds.ray(ray_start, ray_end)
        
        # Sort by x coordinate
        sort_indices = np.argsort(ray['x'])
        x = np.array(ray['x'][sort_indices])
        
        # Find eta field
        eta_field = None
        for field_tuple in ds.field_list:
            if 'eta' in field_tuple[1].lower():
                eta_field = field_tuple
                break
        
        if eta_field is None:
            if i == 0:
                print(f"  ERROR: 'eta' field not found")
                print(f"  Available fields: {[f[1] for f in ds.field_list[:20]]}")
            continue
        
        eta = np.array(ray[eta_field][sort_indices])
        
        # Filter to x >= x_min_focus
        mask = x >= x_min_focus
        x = x[mask]
        eta = eta[mask]
        
        if len(x) < 2:
            continue
        
        # Find eta = 0.5
        if np.min(eta) <= 0.5 <= np.max(eta):
            # Sort by eta for interpolation
            eta_sort_idx = np.argsort(eta)
            eta_sorted = eta[eta_sort_idx]
            x_sorted = x[eta_sort_idx]
            
            f = interp1d(eta_sorted, x_sorted, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
            x_interface = float(f(0.5))
        else:
            idx_closest = np.argmin(np.abs(eta - 0.5))
            x_interface = float(x[idx_closest])
        
        # Store results
        times.append(float(ds.current_time))
        interface_positions.append(x_interface)
        timestep_numbers.append(timestep_num)
        
        # Progress every 100 timesteps
        if (i + 1) % 100 == 0 or i == 0 or i == len(timestep_dirs) - 1:
            print(f"  Timestep {timestep_num:5d} (t={float(ds.current_time):.6e}s): x={x_interface:.6f}")
    
    except Exception as e:
        if i < 3:
            print(f"  ERROR: {str(e)[:80]}")
        continue

if len(times) == 0:
    print("\nERROR: No data extracted")
    exit(1)

# Convert to arrays
times = np.array(times)
interface_positions = np.array(interface_positions)
timestep_numbers = np.array(timestep_numbers)

print(f"\nProcessed {len(times)} timesteps")

# Save to HDF5
with h5py.File(output_hdf5, 'w') as f:
    grp = f.create_group(case_name)
    grp.create_dataset('time', data=times)
    grp.create_dataset('interface_position', data=interface_positions)
    grp.create_dataset('timestep_number', data=timestep_numbers)
    
    grp.attrs['case_name'] = case_name
    grp.attrs['num_timesteps'] = len(times)
    grp.attrs['time_min'] = np.min(times)
    grp.attrs['time_max'] = np.max(times)
    grp.attrs['x_min_focus'] = x_min_focus

print(f"Saved to {output_hdf5}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Timesteps: {len(times)}")
print(f"Time: {np.min(times):.6e} to {np.max(times):.6e} s")
print(f"Position: {np.min(interface_positions):.6f} to {np.max(interface_positions):.6f}")
print(f"Displacement: {np.max(interface_positions) - np.min(interface_positions):.6f}")

if len(times) > 1:
    avg_vel = (interface_positions[-1] - interface_positions[0]) / (times[-1] - times[0])
    print(f"Avg velocity: {avg_vel:.6e} m/s")

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(times, interface_positions, 'b-', linewidth=2, marker='o', markersize=2)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Interface Position (x)', fontsize=12)
ax.set_title(f'Bubble Interface (eta=0.5, x>={x_min_focus})', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plot_file = f'./{case_name}_interface_tracking.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nPlot: {plot_file}")
plt.show()

print("\nComplete!")
