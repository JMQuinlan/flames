"""
Track interface location (eta = 0.5), velocity at interface, and pressure at origin
Uses yt to read the data - OPTIMIZED VERSION
"""

import yt
import numpy as np
import h5py
import os
from scipy.interpolate import interp1d

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# Configuration
case_name = 'LargeBubbleRPE_2_Bar.GammaHigh.old.20250915191918'
group_name = ''
on_Incline = 0

if (on_Incline == 1):
    amrex_output_dir = fr'/mmfs1/home/ttryon/flames/bin/tests/{group_name}/{case_name}'
else:
    amrex_output_dir = fr'../../../bin/tests/{group_name}/{case_name}'

output_hdf5 = f'./{case_name}_tracking.hdf5'

# Focus on bubble region (x >= 0)
x_min_focus = 0.0

# Tracking locations
pressure_track_x = 0.0
pressure_track_y = 0.0

print("="*60)
print(f"TRACKING DATA FOR {case_name}")
print(f"  1. Interface position (eta = 0.5, x >= {x_min_focus})")
print(f"  2. Velocity at interface (eta = 0.5)")
print(f"  3. Pressure at origin (x={pressure_track_x}, y={pressure_track_y})")
print("="*60)

# Find all timestep directories - OPTIMIZED
timestep_dirs = []
seen_timesteps = set()

for item in os.listdir(amrex_output_dir):
    # Skip .old. directories
    if '.old.' in item.lower():
        continue
    
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and 'cell' in item.lower():
        numeric_part = ''.join(filter(str.isdigit, item))
        timestep_num = int(numeric_part) if numeric_part else -1
        
        if timestep_num in seen_timesteps:
            continue
        
        seen_timesteps.add(timestep_num)
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
print(f"\nFound {len(timestep_dirs)} unique timesteps (skipped .old. directories)")

# Arrays to store results - preallocate for speed
num_timesteps = len(timestep_dirs)
times = np.zeros(num_timesteps)
interface_positions = np.zeros(num_timesteps)
interface_velocities = np.zeros(num_timesteps)
origin_pressures = np.zeros(num_timesteps)
timestep_numbers = np.zeros(num_timesteps, dtype=int)
valid_count = 0

print("\nProcessing timesteps...")

# Cache field names after first successful read
eta_field_cached = None
pressure_field_cached = None
velocityx_field_cached = None

for i, timestep_dir in enumerate(timestep_dirs):
    try:
        timestep_num = extract_timestep_number(timestep_dir)
        
        # Load the timestep directory
        ds = yt.load(timestep_dir)
        
        # Get domain bounds
        x_left = float(ds.domain_left_edge[0])
        x_right = float(ds.domain_right_edge[0])
        
        # Find fields on first iteration
        if eta_field_cached is None:
            for field_tuple in ds.field_list:
                field_name = field_tuple[1].lower()
                if 'eta' in field_name and eta_field_cached is None:
                    eta_field_cached = field_tuple
                if 'pressure' in field_name and pressure_field_cached is None:
                    pressure_field_cached = field_tuple
                if 'velocityx' in field_name and velocityx_field_cached is None:
                    velocityx_field_cached = field_tuple
            
            if eta_field_cached is None or pressure_field_cached is None or velocityx_field_cached is None:
                print(f"  ERROR: Required fields not found")
                print(f"  Available fields: {[f[1] for f in ds.field_list[:20]]}")
                break
        
        # 1. Track interface position and velocity (eta = 0.5)
        ray_start = ds.arr([max(x_left, x_min_focus), 0.0, 0.0], 'code_length')
        ray_end = ds.arr([x_right, 0.0, 0.0], 'code_length')
        ray = ds.ray(ray_start, ray_end)
        
        sort_indices = np.argsort(ray['x'])
        x = np.array(ray['x'][sort_indices])
        eta = np.array(ray[eta_field_cached][sort_indices])
        velocityx = np.array(ray[velocityx_field_cached][sort_indices])
        
        # Filter to x >= x_min_focus
        mask = x >= x_min_focus
        x = x[mask]
        eta = eta[mask]
        velocityx = velocityx[mask]
        
        if len(x) < 2:
            continue
        
        # Find eta = 0.5 location
        eta_min = np.min(eta)
        eta_max = np.max(eta)
        
        if eta_min <= 0.5 <= eta_max:
            if not np.all(eta[:-1] <= eta[1:]):
                eta_sort_idx = np.argsort(eta)
                eta_sorted = eta[eta_sort_idx]
                x_sorted = x[eta_sort_idx]
                velocityx_sorted = velocityx[eta_sort_idx]
            else:
                eta_sorted = eta
                x_sorted = x
                velocityx_sorted = velocityx
            
            # Interpolate for interface position
            f_x = interp1d(eta_sorted, x_sorted, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
            x_interface = float(f_x(0.5))
            
            # Interpolate for velocity at interface
            f_vel = interp1d(eta_sorted, velocityx_sorted, kind='linear',
                           bounds_error=False, fill_value='extrapolate')
            vel_interface = float(f_vel(0.5))
        else:
            idx_closest = np.argmin(np.abs(eta - 0.5))
            x_interface = float(x[idx_closest])
            vel_interface = float(velocityx[idx_closest])
        
        # 2. Track pressure at origin (x=0, y=0)
        point = ds.point([pressure_track_x, pressure_track_y, 0.0])
        pressure_origin = float(point[pressure_field_cached])
        
        # Store results
        times[valid_count] = float(ds.current_time)
        interface_positions[valid_count] = x_interface
        interface_velocities[valid_count] = vel_interface
        origin_pressures[valid_count] = pressure_origin
        timestep_numbers[valid_count] = timestep_num
        valid_count += 1
        
        # Progress every 100 timesteps
        if (i + 1) % 100 == 0 or i == 0 or i == len(timestep_dirs) - 1:
            print(f"  Timestep {timestep_num:5d} (t={float(ds.current_time):.6e}s):")
            print(f"    Interface: x={x_interface:.6f}, vel={vel_interface:.6e}")
            print(f"    Origin pressure: {pressure_origin:.6e}")
    
    except Exception as e:
        if i < 3:
            print(f"  ERROR: {str(e)[:80]}")
        continue

if valid_count == 0:
    print("\nERROR: No data extracted")
    exit(1)

# Trim arrays to actual valid count
times = times[:valid_count]
interface_positions = interface_positions[:valid_count]
interface_velocities = interface_velocities[:valid_count]
origin_pressures = origin_pressures[:valid_count]
timestep_numbers = timestep_numbers[:valid_count]

print(f"\nProcessed {valid_count} timesteps")

# Save to HDF5
with h5py.File(output_hdf5, 'w') as f:
    grp = f.create_group(case_name)
    
    # Time array (shared by all quantities)
    grp.create_dataset('time', data=times, compression='gzip')
    grp.create_dataset('timestep_number', data=timestep_numbers, compression='gzip')
    
    # Interface tracking
    grp.create_dataset('interface_position', data=interface_positions, compression='gzip')
    grp.create_dataset('interface_velocity', data=interface_velocities, compression='gzip')
    
    # Origin pressure tracking
    grp.create_dataset('origin_pressure', data=origin_pressures, compression='gzip')
    
    # Metadata
    grp.attrs['case_name'] = case_name
    grp.attrs['num_timesteps'] = valid_count
    grp.attrs['time_min'] = np.min(times)
    grp.attrs['time_max'] = np.max(times)
    grp.attrs['x_min_focus'] = x_min_focus
    grp.attrs['pressure_track_x'] = pressure_track_x
    grp.attrs['pressure_track_y'] = pressure_track_y
    grp.attrs['description'] = 'Interface position (eta=0.5), velocity at interface, and pressure at origin'

print(f"Saved to {output_hdf5}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Timesteps: {valid_count}")
print(f"Time: {np.min(times):.6e} to {np.max(times):.6e} s")
print(f"\nInterface position:")
print(f"  Range: {np.min(interface_positions):.6f} to {np.max(interface_positions):.6f}")
print(f"  Displacement: {np.max(interface_positions) - np.min(interface_positions):.6f}")
print(f"\nInterface velocity:")
print(f"  Range: {np.min(interface_velocities):.6e} to {np.max(interface_velocities):.6e} m/s")
print(f"\nOrigin pressure:")
print(f"  Range: {np.min(origin_pressures):.6e} to {np.max(origin_pressures):.6e} Pa")

if valid_count > 1:
    avg_vel = (interface_positions[-1] - interface_positions[0]) / (times[-1] - times[0])
    print(f"\nAverage interface velocity: {avg_vel:.6e} m/s")

# Create plots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Interface position
axes[0].plot(times, interface_positions, 'b-', linewidth=2, marker='o', markersize=2)
axes[0].set_xlabel('Time (s)', fontsize=11)
axes[0].set_ylabel('Interface Position (x)', fontsize=11)
axes[0].set_title('Bubble Interface Position (eta=0.5)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Interface velocity
axes[1].plot(times, interface_velocities, 'r-', linewidth=2, marker='o', markersize=2)
axes[1].set_xlabel('Time (s)', fontsize=11)
axes[1].set_ylabel('Velocity (m/s)', fontsize=11)
axes[1].set_title('Velocity at Interface (eta=0.5)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Origin pressure
axes[2].plot(times, origin_pressures, 'g-', linewidth=2, marker='o', markersize=2)
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Pressure (Pa)', fontsize=11)
axes[2].set_title(f'Pressure at Origin (x={pressure_track_x}, y={pressure_track_y})', 
                 fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

plot_file = f'./{case_name}_tracking.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nPlot: {plot_file}")
plt.show()

print("\nComplete!")
