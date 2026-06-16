"""
Extract 1D slice from AMReX data using yt for CO2 Depressurization Test 2 (WITH Vaporization)
Visualizes numerical results and compares with reference data when available
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# Configuration
case_name = 'Test2_CO2_WithVap'
amrex_output_dir = r'../../../bin/tests/FlowDepressurization/Test2_CO2_Vap'

# CO2 Stiffened Gas EOS Parameters (from Table 1)
# Liquid
gamma_L = 1.23
p_inf_L = 1.32e8  # Pa
cv_L = 2.44e3     # J/(kg*K)
q_L = -6.23e5     # J/kg

# Vapor
gamma_V = 1.06
p_inf_V = 8.86e5  # Pa
cv_V = 2.41e3     # J/(kg*K)
q_V = -3.01e5     # J/kg

# Initial conditions (from Lund test case)
# Left side (mostly liquid)
P_L = 6.0e6       # 60 bar
T_L = 273.0       # K
rho_L = 902.0     # kg/m^3 (calculated)

# Right side (mostly vapor)
P_R = 1.0e6       # 10 bar
T_R = 273.0       # K
rho_R = 47.8      # kg/m^3 (calculated)

# Geometry
x_interface = 50.0  # m
L_total = 80.0      # m
t_final = 0.08      # s

print("="*60)
print("CO2 DEPRESSURIZATION TEST 2 - WITH VAPORIZATION")
print("="*60)
print(f"\nStiffened Gas EOS Parameters:")
print(f"  Left (liquid):  gamma = {gamma_L}, p_inf = {p_inf_L/1e6:.2f} MPa")
print(f"  Right (vapor):  gamma = {gamma_V}, p_inf = {p_inf_V/1e6:.2f} MPa")
print(f"\nInitial Conditions:")
print(f"  Left:  P = {P_L/1e5:.1f} bar, T = {T_L:.1f} K, rho = {rho_L:.1f} kg/m^3")
print(f"  Right: P = {P_R/1e5:.1f} bar, T = {T_R:.1f} K, rho = {rho_R:.1f} kg/m^3")
print(f"  Interface at x = {x_interface} m")
print(f"\nVaporization: ENABLED")
print(f"  Mass transfer between phases active")
print(f"  Chemical relaxation at metastable interfaces")

print("\n" + "="*60)
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
sim_time = float(ds.current_time)
print(f"Simulation time: {sim_time:.6f} s")
print(f"Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")

# Get field names
print(f"\nAvailable fields:")
for field in ds.field_list:
    print(f"  {field}")

# Create a ray (1D line) through the domain at y=0
print(f"\nExtracting 1D slice at y=0...")

# Define the ray from x=0 to x=80 at y=0, z=0
ray_start = ds.arr([0.0, 0.0, 0.0], 'code_length')
ray_end = ds.arr([L_total, 0.0, 0.0], 'code_length')
ray = ds.ray(ray_start, ray_end)

# Sort by x coordinate
sort_indices = np.argsort(ray['x'])
x_numerical = np.array(ray['x'][sort_indices])
velocity_numerical = np.array(ray['velocityx'][sort_indices])
pressure_numerical = np.array(ray['pressure'][sort_indices])
density_numerical = np.array(ray['density'][sort_indices])

# Try to extract phase fraction (eta) if available
try:
    eta_numerical = np.array(ray['eta'][sort_indices])
    has_eta = True
    print(f"  Extracted phase fraction (eta)")
except:
    has_eta = False
    print(f"  Phase fraction (eta) not available in output")

print(f"Extracted {len(x_numerical)} points")

# Print data ranges
print(f"\nNumerical data ranges:")
print(f"  x: [{np.min(x_numerical):.6f}, {np.max(x_numerical):.6f}] m")
print(f"  velocity: [{np.min(velocity_numerical):.6e}, {np.max(velocity_numerical):.6e}] m/s")
print(f"  pressure: [{np.min(pressure_numerical):.6e}, {np.max(pressure_numerical):.6e}] Pa")
print(f"  density: [{np.min(density_numerical):.6e}, {np.max(density_numerical):.6e}] kg/m^3")
if has_eta:
    print(f"  eta: [{np.min(eta_numerical):.6e}, {np.max(eta_numerical):.6e}]")

print("\n" + "="*60)
print("LOADING REFERENCE DATA (IF AVAILABLE)")
print("="*60)

# Placeholder for reference/truth data
# TODO: Replace this section when you have reference data from binary file
reference_data_available = False
reference_file = './reference_data_test2.hdf5'  # Change to your binary file path

if os.path.exists(reference_file):
    print(f"Loading reference data from: {reference_file}")
    try:
        with h5py.File(reference_file, 'r') as f:
            # TODO: Adjust these dataset names to match your binary file structure
            x_reference = f['x'][:]
            density_reference = f['density'][:]
            velocity_reference = f['velocity'][:]
            pressure_reference = f['pressure'][:]
            if 'eta' in f:
                eta_reference = f['eta'][:]
            else:
                eta_reference = None
        reference_data_available = True
        print(f"  Successfully loaded {len(x_reference)} reference points")
    except Exception as e:
        print(f"  Error loading reference data: {e}")
        reference_data_available = False
else:
    print(f"Reference file not found: {reference_file}")
    print("Proceeding with numerical results only")
    print("\nTO ADD REFERENCE DATA:")
    print("  1. Save your reference data to HDF5 format")
    print("  2. Update 'reference_file' path in this script")
    print("  3. Ensure datasets are named: 'x', 'density', 'velocity', 'pressure', 'eta'")

print("\n" + "="*60)
print("CREATING PLOTS")
print("="*60)

# Determine number of subplots
n_plots = 4 if has_eta else 3
fig, axes = plt.subplots(n_plots, 1, figsize=(14, 4*n_plots))

plot_idx = 0

# Density
axes[plot_idx].set_title(f'CO2 Depressurization (WITH Vaporization) at t={sim_time:.6f}s', 
                  fontsize=14, fontweight='bold')
if reference_data_available:
    axes[plot_idx].plot(x_reference, density_reference, 'b-', linewidth=2, 
                 label='Reference Solution', zorder=1)
axes[plot_idx].plot(x_numerical, density_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[plot_idx].set_xlabel('Position (m)', fontsize=12)
axes[plot_idx].set_ylabel('Density (kg/m^3)', fontsize=12)
axes[plot_idx].legend(fontsize=10)
axes[plot_idx].grid(True, alpha=0.3)
axes[plot_idx].set_xlim([0, L_total])
axes[plot_idx].axvline(x=x_interface, color='k', linestyle=':', alpha=0.5, label='Initial Interface')
plot_idx += 1

# Velocity
if reference_data_available:
    axes[plot_idx].plot(x_reference, velocity_reference, 'b-', linewidth=2, 
                 label='Reference Solution', zorder=1)
axes[plot_idx].plot(x_numerical, velocity_numerical, 'r--', linewidth=1.5, 
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[plot_idx].set_xlabel('Position (m)', fontsize=12)
axes[plot_idx].set_ylabel('Velocity (m/s)', fontsize=12)
axes[plot_idx].legend(fontsize=10)
axes[plot_idx].grid(True, alpha=0.3)
axes[plot_idx].set_xlim([0, L_total])
axes[plot_idx].axvline(x=x_interface, color='k', linestyle=':', alpha=0.5)
plot_idx += 1

# Pressure
if reference_data_available:
    axes[plot_idx].plot(x_reference, pressure_reference/1e5, 'b-', linewidth=2, 
                 label='Reference Solution', zorder=1)
axes[plot_idx].plot(x_numerical, pressure_numerical/1e5, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[plot_idx].set_xlabel('Position (m)', fontsize=12)
axes[plot_idx].set_ylabel('Pressure (bar)', fontsize=12)
axes[plot_idx].legend(fontsize=10)
axes[plot_idx].grid(True, alpha=0.3)
axes[plot_idx].set_xlim([0, L_total])
axes[plot_idx].axvline(x=x_interface, color='k', linestyle=':', alpha=0.5)
plot_idx += 1

# Phase fraction (eta) if available
if has_eta:
    if reference_data_available and eta_reference is not None:
        axes[plot_idx].plot(x_reference, eta_reference, 'b-', linewidth=2, 
                     label='Reference Solution', zorder=1)
    axes[plot_idx].plot(x_numerical, eta_numerical, 'r--', linewidth=1.5,
                 label='Numerical Solution', zorder=2, alpha=0.7)
    axes[plot_idx].set_xlabel('Position (m)', fontsize=12)
    axes[plot_idx].set_ylabel('Liquid Volume Fraction (eta)', fontsize=12)
    axes[plot_idx].legend(fontsize=10)
    axes[plot_idx].grid(True, alpha=0.3)
    axes[plot_idx].set_xlim([0, L_total])
    axes[plot_idx].set_ylim([-0.1, 1.1])
    axes[plot_idx].axvline(x=x_interface, color='k', linestyle=':', alpha=0.5)

plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('./Images', exist_ok=True)

output_filename = f'./Images/{case_name}_results'
plt.savefig(output_filename+'.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename+'.eps', format='eps', bbox_inches='tight')
print(f"\nPlot saved: {output_filename}")
plt.show()

# Calculate error metrics if reference data available
if reference_data_available:
    print("\n" + "="*60)
    print("ERROR METRICS (vs. Reference Solution)")
    print("="*60)
    
    # Interpolate reference to numerical grid
    pressure_ref_interp = np.interp(x_numerical, x_reference, pressure_reference)
    density_ref_interp = np.interp(x_numerical, x_reference, density_reference)
    velocity_ref_interp = np.interp(x_numerical, x_reference, velocity_reference)
    
    print(f"Velocity L2 error:   {np.linalg.norm(velocity_numerical - velocity_ref_interp):.6e}")
    print(f"Velocity Linf error: {np.max(np.abs(velocity_numerical - velocity_ref_interp)):.6e}")
    print(f"\nPressure L2 error:   {np.linalg.norm(pressure_numerical - pressure_ref_interp):.6e}")
    print(f"Pressure Linf error: {np.max(np.abs(pressure_numerical - pressure_ref_interp)):.6e}")
    print(f"\nDensity L2 error:    {np.linalg.norm(density_numerical - density_ref_interp):.6e}")
    print(f"Density Linf error:  {np.max(np.abs(density_numerical - density_ref_interp)):.6e}")
    
    if has_eta and eta_reference is not None:
        eta_ref_interp = np.interp(x_numerical, x_reference, eta_reference)
        print(f"\nEta L2 error:        {np.linalg.norm(eta_numerical - eta_ref_interp):.6e}")
        print(f"Eta Linf error:      {np.max(np.abs(eta_numerical - eta_ref_interp)):.6e}")
    
    print("="*60)

print("\n" + "="*60)
print("PHYSICAL OBSERVATIONS")
print("="*60)

# Analyze wave structure
print("\nWave Structure Analysis:")
print(f"  Initial interface position: {x_interface} m")

# Find approximate contact discontinuity (max density gradient)
density_gradient = np.gradient(density_numerical, x_numerical)
contact_idx = np.argmax(np.abs(density_gradient))
contact_position = x_numerical[contact_idx]
print(f"  Approximate contact position: {contact_position:.2f} m")
print(f"  Contact displacement: {contact_position - x_interface:.2f} m")

# Find velocity extrema
max_vel_idx = np.argmax(np.abs(velocity_numerical))
print(f"  Maximum velocity: {velocity_numerical[max_vel_idx]:.2f} m/s at x = {x_numerical[max_vel_idx]:.2f} m")

# Pressure range
print(f"\nPressure evolution:")
print(f"  Minimum: {np.min(pressure_numerical)/1e5:.2f} bar")
print(f"  Maximum: {np.max(pressure_numerical)/1e5:.2f} bar")
print(f"  Pressure ratio: {np.max(pressure_numerical)/np.min(pressure_numerical):.2f}")

if has_eta:
    print(f"\nPhase distribution:")
    liquid_region = np.sum(eta_numerical > 0.5)
    vapor_region = np.sum(eta_numerical < 0.5)
    print(f"  Cells with liquid (eta > 0.5): {liquid_region}")
    print(f"  Cells with vapor (eta < 0.5): {vapor_region}")
    
    # Find interface region (0.1 < eta < 0.9)
    interface_region = np.sum((eta_numerical > 0.1) & (eta_numerical < 0.9))
    print(f"  Interface cells (0.1 < eta < 0.9): {interface_region}")

print("="*60)

print("\nAnalysis complete!")
print("\nNOTE: This test includes vaporization effects:")
print("  - Mass transfer between liquid and vapor phases")
print("  - Chemical relaxation at metastable interfaces")
print("  - Additional wave structures compared to Test 1")
print("\nTo add reference data for comparison:")
print("  1. Save reference solution to HDF5 file")
print("  2. Update 'reference_file' variable in script")
print("  3. Re-run this script")
