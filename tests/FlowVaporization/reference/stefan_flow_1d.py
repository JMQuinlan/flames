"""
===============================================================================
1D STEFAN FLOW - POST-PROCESSING FROM AMREX PLOT FILES
===============================================================================

PURPOSE:
    Post-process AMReX plot files from 1D Stefan flow simulation.
    Extract and plot interface position (eta = 0.5) and mass flux over time.
    
FEATURES:
    - Read AMReX plot files from simulation
    - Track interface position (eta = 0.5 contour)
    - Calculate mass flux at interface from simulation data
    - Compare with analytical Stefan flow solution
    - Time series plots
    
PHYSICS:
    Stefan flow mass transfer with moving interface
    - Interface tracking from eta field
    - Mass flux from rho * v_x at interface
    - Analytical comparison
    
INPUTS:
    - AMReX plot files from your hydro2 simulation
    - Physical parameters from input file
    
OUTPUTS:
    - Interface position vs time plot
    - Mass flux vs time plot
    - Combined visualization
    - Data export to text file

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS (FROM YOUR INPUT FILE)
# ============================================================================

# File paths
amrex_output_dir = r'../../../bin/tests/stefan/stefan_1'
output_folder = './Stefan_Flow_1D_PostProcess'

# Physical parameters - FROM YOUR INPUT FILE
# Fluid 0 properties (vapor/gas phase, eta=1)
density0 = 1.0                   # Density of gas phase [kg/m^3]
mu0 = 1.8e-5                     # Dynamic viscosity of gas [Pa-s]
pressure0 = 101325.0             # Pressure of gas [Pa]
gamma0 = 1.4                     # Ratio of specific heats for gas
temperature0 = 293.15            # Temperature [K]
cp0 = 1005.0                     # Specific heat [J/kg/K]
cv0 = 717.86                     # Specific heat at constant volume [J/kg/K]
k_thermal0 = 0.026               # Thermal conductivity [W/m/K]

# Fluid 1 properties (liquid phase, eta=0)
density1 = 1000.0                # Density of liquid [kg/m^3]
mu1 = 1.0e-3                     # Dynamic viscosity of liquid [Pa-s]
pressure1 = 101325.0             # Pressure of liquid [Pa]
gamma1 = 1.4                     # Ratio of specific heats for liquid
temperature1 = 293.15            # Temperature [K]
cp1 = 4179.0                     # Specific heat [J/kg/K]
cv1 = 2984.3                     # Specific heat at constant volume [J/kg/K]
k_thermal1 = 0.6                 # Thermal conductivity [W/m/K]

# Mass transfer properties
D_v = 1.0e-5                     # Binary diffusion coefficient [m^2/s]
sigma = 72.8                     # Surface tension [N/m]

# Boundary conditions for mass fraction
Y_surface = 0.9                  # Mass fraction at liquid surface (eta=0.5)
Y_infinity = 0.1                 # Mass fraction far from interface

# Domain properties (FROM YOUR INPUT FILE)
x_min = 0.0                      # geometry.prob_lo
x_max = 1.0                      # geometry.prob_hi
L_domain = x_max - x_min

# Initial interface position (FROM YOUR INPUT FILE)
x_interface_initial = 0.5        # eta.ic.expression.constant.x_interface

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_EXACT = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL SOLUTION FUNCTIONS
# ============================================================================

def stefan_velocity(D_v, delta, Y_s, Y_inf):
    """
    Calculate Stefan velocity at the interface
    v_s = (D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0 or delta < 1e-12:
        return 0.0
    return (D_v / delta) * np.log((1.0 - Y_inf) / (1.0 - Y_s))

def stefan_mass_flux(rho, D_v, delta, Y_s, Y_inf):
    """
    Calculate mass flux at the interface
    m_dot = rho * v_s = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    return rho * v_s

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def find_interface_position_1d(x_coords, eta_values):
    """
    Find x-position where eta = 0.5 (interface location)
    """
    idx = np.argmin(np.abs(eta_values - 0.5))
    return x_coords[idx]

def calculate_mass_flux_at_interface(ds, x_interface):
    """
    Calculate mass flux at interface from simulation data
    m_dot = rho * v_x at interface
    """
    # Extract 1D ray along x-axis at y=0
    y_mid = 0.0
    z_mid = 0.0
    x_min_domain = float(ds.domain_left_edge[0])
    x_max_domain = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([x_min_domain, y_mid, z_mid], 'code_length')
    ray_end = ds.arr([x_max_domain, y_mid, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by x-coordinate
    sort_indices = np.argsort(ray['x'])
    x_coords = np.array(ray['x'][sort_indices])
    rho_values = np.array(ray['density'][sort_indices])
    vx_values = np.array(ray['velocityx'][sort_indices])
    
    # Interpolate to interface position
    idx = np.argmin(np.abs(x_coords - x_interface))
    rho_interface = rho_values[idx]
    vx_interface = vx_values[idx]
    
    return rho_interface * vx_interface

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("1D STEFAN FLOW - POST-PROCESSING FROM AMREX FILES")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    # Skip files with .old in the name
    if os.path.isdir(item_path) and 'cell' in item and '.old' not in item:
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files")

# ============================================================================
# EXTRACT DATA FROM ALL TIMESTEPS
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM ALL TIMESTEPS")
print("=" * 70)

times = []
interface_positions = []
numerical_mass_flux = []
stefan_velocities = []
analytical_mass_flux = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    
    # Extract 1D ray along x-axis at y=0
    y_mid = 0.0
    z_mid = 0.0
    x_min_domain = float(ds.domain_left_edge[0])
    x_max_domain = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([x_min_domain, y_mid, z_mid], 'code_length')
    ray_end = ds.arr([x_max_domain, y_mid, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by x-coordinate
    sort_indices = np.argsort(ray['x'])
    x_coords = np.array(ray['x'][sort_indices])
    eta_values = np.array(ray['eta'][sort_indices])
    
    # Find interface position (where eta = 0.5)
    x_interface = find_interface_position_1d(x_coords, eta_values)
    
    # Calculate numerical mass flux at interface
    m_dot_num = calculate_mass_flux_at_interface(ds, x_interface)
    
    # Calculate analytical mass flux
    delta = L_domain - x_interface
    m_dot_ana = stefan_mass_flux(density0, D_v, delta, Y_surface, Y_infinity)
    v_s = stefan_velocity(D_v, delta, Y_surface, Y_infinity)
    
    times.append(t)
    interface_positions.append(x_interface)
    numerical_mass_flux.append(m_dot_num)
    analytical_mass_flux.append(m_dot_ana)
    stefan_velocities.append(v_s)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
interface_positions = np.array(interface_positions)
numerical_mass_flux = np.array(numerical_mass_flux)
analytical_mass_flux = np.array(analytical_mass_flux)
stefan_velocities = np.array(stefan_velocities)

print(f"\nExtraction complete!")
print(f"  Time range: {times[0]:.6e} s to {times[-1]:.6e} s")
print(f"  Interface displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")

# ============================================================================
# PLOT 1: INTERFACE POSITION VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, interface_positions, 'bo-', linewidth=LINE_WIDTH_NUMERICAL, 
         markersize=MARKER_SIZE, label='Interface Position (eta = 0.5)', alpha=0.7)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('1D Stefan Flow: Interface Position vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '01_Interface_Position_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Interface_Position_vs_Time.eps'))
print("  Saved: 01_Interface_Position_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: MASS FLUX VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(times, analytical_mass_flux, 'b-', linewidth=LINE_WIDTH_EXACT, 
         label='Analytical Mass Flux', zorder=1)
ax2.plot(times, numerical_mass_flux, 'ro', markersize=MARKER_SIZE, 
         label='Numerical Mass Flux (rho * v_x)', alpha=0.7, zorder=2)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('1D Stefan Flow: Mass Flux vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_Mass_Flux_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Mass_Flux_vs_Time.eps'))
print("  Saved: 02_Mass_Flux_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: STEFAN VELOCITY VS TIME
# ============================================================================

# Calculate numerical interface velocity from position
if len(times) > 1:
    interface_velocity_numerical = np.gradient(interface_positions, times)
    
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.plot(times, stefan_velocities, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical Stefan Velocity', zorder=1)
    ax3.plot(times, interface_velocity_numerical, 'ro', markersize=MARKER_SIZE, 
             label='Numerical Interface Velocity', alpha=0.7, zorder=2)
    
    ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax3.set_ylabel('Interface Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
    ax3.set_title('1D Stefan Flow: Interface Velocity vs Time', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax3.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '03_Interface_Velocity_vs_Time.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '03_Interface_Velocity_vs_Time.eps'))
    print("  Saved: 03_Interface_Velocity_vs_Time.png/.eps")
    plt.close()

# ============================================================================
# PLOT 4: COMBINED PLOT (INTERFACE AND MASS FLUX)
# ============================================================================

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 12))

# Top: Interface position
ax4a.plot(times, interface_positions, 'bo-', linewidth=LINE_WIDTH_NUMERICAL, 
          markersize=MARKER_SIZE, label='Interface Position (eta = 0.5)', alpha=0.7)
ax4a.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4a.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax4a.set_title('Interface Position vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4a.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4a.grid(True, alpha=0.3)
ax4a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Mass flux
ax4b.plot(times, analytical_mass_flux, 'b-', linewidth=LINE_WIDTH_EXACT, 
          label='Analytical Mass Flux', zorder=1)
ax4b.plot(times, numerical_mass_flux, 'ro', markersize=MARKER_SIZE, 
          label='Numerical Mass Flux', alpha=0.7, zorder=2)
ax4b.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4b.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax4b.set_title('Mass Flux vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4b.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4b.grid(True, alpha=0.3)
ax4b.tick_params(labelsize=FONT_SIZE_TICK)

fig4.suptitle('1D Stefan Flow - Simulation Results', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '04_Combined_Results.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Combined_Results.eps'))
print("  Saved: 04_Combined_Results.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: MASS FLUX ERROR
# ============================================================================

mass_flux_error = np.abs(numerical_mass_flux - analytical_mass_flux)
epsilon = 1e-16
mass_flux_error_safe = mass_flux_error + epsilon

fig5, ax5 = plt.subplots(figsize=(10, 8))
ax5.semilogy(times, mass_flux_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)

ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('|m_dot_num - m_dot_analytical| (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Mass Flux Error vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '05_Mass_Flux_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Mass_Flux_Error.eps'))
print("  Saved: 05_Mass_Flux_Error.png/.eps")
plt.close()

# ============================================================================
# SAVE DATA TO TEXT FILES
# ============================================================================

print("\n" + "=" * 70)
print("SAVING DATA TO TEXT FILES")
print("=" * 70)

# Save time series data
data_file = os.path.join(output_folder, 'stefan_flow_data.txt')
header = "Time(s) Interface_Position(m) Numerical_Mass_Flux(kg/m^2/s) Analytical_Mass_Flux(kg/m^2/s) Stefan_Velocity(m/s)"
data = np.column_stack((times, interface_positions, numerical_mass_flux, analytical_mass_flux, stefan_velocities))
np.savetxt(data_file, data, header=header, fmt='%.10e')
print(f"  Saved: stefan_flow_data.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("POST-PROCESSING COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Gas density (rho0): {density0} kg/m^3")
print(f"  Liquid density (rho1): {density1} kg/m^3")
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")

print(f"\nSimulation Results:")
print(f"  Initial interface position: {interface_positions[0]:.6f} m")
print(f"  Final interface position: {interface_positions[-1]:.6f} m")
print(f"  Interface displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")
print(f"  Average numerical mass flux: {np.mean(numerical_mass_flux):.6e} kg/m^2/s")
print(f"  Average analytical mass flux: {np.mean(analytical_mass_flux):.6e} kg/m^2/s")
print(f"  Average Stefan velocity: {np.mean(stefan_velocities):.6e} m/s")
print(f"  Max mass flux error: {np.max(mass_flux_error):.6e} kg/m^2/s")

print(f"\nFiles generated:")
print(f"  - 01_Interface_Position_vs_Time.png/.eps")
print(f"  - 02_Mass_Flux_vs_Time.png/.eps")
print(f"  - 03_Interface_Velocity_vs_Time.png/.eps")
print(f"  - 04_Combined_Results.png/.eps")
print(f"  - 05_Mass_Flux_Error.png/.eps")
print(f"  - stefan_flow_data.txt")

print("\n" + "=" * 70)
