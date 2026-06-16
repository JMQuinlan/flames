"""
Simple plotting script for HDF5 tracking data
Plots interface position, velocity, and pressure over time
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

# Configuration
case_name = 'BubbleEvap'
hdf5_file = f'./{case_name}_tracking.hdf5'

print("="*60)
print(f"PLOTTING TRACKING DATA FOR {case_name}")
print("="*60)

# Load data from HDF5
print(f"\nLoading data from {hdf5_file}...")
try:
    with h5py.File(hdf5_file, 'r') as f:
        grp = f[case_name]
        times = grp['time'][:]
        interface_positions = grp['interface_position'][:]
        interface_velocities = grp['interface_velocity'][:]
        origin_pressures = grp['origin_pressure'][:]
        timestep_numbers = grp['timestep_number'][:]
        
        # Load metadata
        num_timesteps = grp.attrs['num_timesteps']
        time_min = grp.attrs['time_min']
        time_max = grp.attrs['time_max']
        x_min_focus = grp.attrs['x_min_focus']
        pressure_track_x = grp.attrs['pressure_track_x']
        pressure_track_y = grp.attrs['pressure_track_y']
    
    print(f"Loaded {num_timesteps} timesteps")
    print(f"Time range: {time_min:.6e} to {time_max:.6e} s")
    
except Exception as e:
    print(f"ERROR: Could not load HDF5 file: {e}")
    exit(1)

# Print summary statistics
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Timesteps: {num_timesteps}")
print(f"Time range: {np.min(times):.6e} to {np.max(times):.6e} s")
print(f"\nInterface position:")
print(f"  Range: {np.min(interface_positions):.6f} to {np.max(interface_positions):.6f} m")
print(f"  Displacement: {np.max(interface_positions) - np.min(interface_positions):.6f} m")
print(f"\nInterface velocity:")
print(f"  Range: {np.min(interface_velocities):.6e} to {np.max(interface_velocities):.6e} m/s")
print(f"\nOrigin pressure:")
print(f"  Range: {np.min(origin_pressures):.6e} to {np.max(origin_pressures):.6e} Pa")

if num_timesteps > 1:
    avg_vel = (interface_positions[-1] - interface_positions[0]) / (times[-1] - times[0])
    print(f"\nAverage interface velocity: {avg_vel:.6e} m/s")

# Create plots
print("\nGenerating plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Interface position
axes[0].plot(times * 1000, interface_positions * 1000, 'b-', linewidth=2, marker='o', markersize=2)
axes[0].set_xlabel('Time (ms)', fontsize=12)
axes[0].set_ylabel('Interface Position (mm)', fontsize=12)
axes[0].set_title(f'Bubble Interface Position (eta=0.5, x>={x_min_focus})', 
                 fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Interface velocity
axes[1].plot(times * 1000, interface_velocities, 'r-', linewidth=2, marker='o', markersize=2)
axes[1].set_xlabel('Time (ms)', fontsize=12)
axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1].set_title('Velocity at Interface (eta=0.5)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# Plot 3: Origin pressure
axes[2].plot(times * 1000, origin_pressures / 1e5, 'g-', linewidth=2, marker='o', markersize=2)
axes[2].set_xlabel('Time (ms)', fontsize=12)
axes[2].set_ylabel('Pressure (bar)', fontsize=12)
axes[2].set_title(f'Pressure at Origin (x={pressure_track_x}, y={pressure_track_y})', 
                 fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()

# Save plots
plot_png = f'./{case_name}_tracking_plots.png'
plot_eps = f'./{case_name}_tracking_plots.eps'
plt.savefig(plot_png, dpi=300, bbox_inches='tight')
plt.savefig(plot_eps, format='eps', bbox_inches='tight')
print(f"\nSaved: {plot_png}")
print(f"Saved: {plot_eps}")
plt.show()

print("\nComplete!")
