"""
===============================================================================
1D PLANAR EVAPORATION WITH STEFAN FLOW - COMPREHENSIVE ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Analyze 1D planar evaporation into stagnant gas with Stefan flow.
    Compare numerical solver results against analytical Stefan flow solution.
    
FEATURES:
    - Comparison plots of mass fraction profiles
    - Interface position tracking over time
    - Mass flux verification
    - Error analysis (absolute and relative)
    - 2x3 grid plots for temporal evolution
    - PNG and EPS output formats

ANALYTICAL SOLUTION:
    Stefan flow for 1D planar evaporation with constant properties:
    
    Mass fraction profile:
    Y(x,t) = Y_inf + (Y_s - Y_inf) * [exp(Pe*x/L) - 1] / [exp(Pe) - 1]
    
    where Pe = m_dot / (rho * D_v) is the Peclet number
    
    Interface velocity (Stefan velocity):
    v_s = (D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    
    Mass flux:
    m_dot = rho * v_s = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]

INPUTS:
    - AMReX plot files from evaporation simulation
    - Physical parameters: rho, D_v, Y_s, Y_inf

OUTPUTS:
    - Comparison plots (numerical vs analytical)
    - Error plots (absolute and relative)
    - Interface position vs time
    - Mass flux verification

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
# CONFIGURATION PARAMETERS
# ============================================================================

# File paths
amrex_output_dir = r'./plot_files'
output_folder = './Stefan_Flow_1D_Analysis'

# Physical parameters - ADJUST THESE FOR YOUR SIMULATION
# Fluid 0 properties (vapor/gas phase, eta=1)
density0 = 1.0                   # Density of gas phase [kg/m^3]
velocity0 = 0.0                  # Initial velocity of gas [m/s]
mu0 = 1.0e-5                     # Dynamic viscosity of gas [Pa-s]
pressure0 = 101325.0             # Pressure of gas [Pa]
gamma0 = 1.4                     # Ratio of specific heats for gas

# Fluid 1 properties (liquid phase, eta=0)
density1 = 1000.0                # Density of liquid [kg/m^3]
velocity1 = 0.0                  # Initial velocity of liquid [m/s]
mu1 = 1.0e-3                     # Dynamic viscosity of liquid [Pa-s]
pressure1 = 101325.0             # Pressure of liquid [Pa]
gamma1 = 1.4                     # Ratio of specific heats for liquid

# Mass transfer properties
D_v = 1.0e-5                     # Binary diffusion coefficient [m^2/s]
Y_surface = 0.9                  # Mass fraction at liquid surface (eta=0.5)
Y_infinity = 0.1                 # Mass fraction far from interface

# Domain properties
L_domain = 1.0                   # Domain length [m]

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_EXACT = 2.5
LINE_WIDTH_NUMERICAL = 2.0

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
    if Y_s >= 1.0 or Y_inf >= 1.0:
        return 0.0
    return (D_v / delta) * np.log((1.0 - Y_inf) / (1.0 - Y_s))

def stefan_mass_flux(rho, D_v, delta, Y_s, Y_inf):
    """
    Calculate mass flux at the interface
    
    m_dot = rho * v_s = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    return rho * v_s

def stefan_mass_fraction_profile(x, x_interface, D_v, Y_s, Y_inf, rho):
    """
    Analytical mass fraction profile with Stefan flow
    
    Y(x) = Y_inf + (Y_s - Y_inf) * [exp(Pe*(x-x_i)/delta) - 1] / [exp(Pe) - 1]
    
    where delta = L - x_interface, Pe = v_s * delta / D_v
    """
    delta = L_domain - x_interface
    if delta < 1e-12:
        return np.full_like(x, Y_inf)
    
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    Pe = v_s * delta / D_v
    
    # Normalized distance from interface
    xi = (x - x_interface) / delta
    
    # Mass fraction profile
    if abs(Pe) < 1e-6:
        # Linear profile for small Peclet number
        Y = Y_s + (Y_inf - Y_s) * xi
    else:
        Y = Y_inf + (Y_s - Y_inf) * (np.exp(Pe * xi) - 1.0) / (np.exp(Pe) - 1.0)
    
    # Apply only in gas region (x > x_interface)
    Y = np.where(x > x_interface, Y, Y_s)
    
    return Y

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def calculate_mass_fraction(eta, rho0, rho1):
    """
    Calculate mass fraction from volume fraction
    Y = eta * rho1 / (eta * rho1 + (1-eta) * rho0)
    """
    denominator = eta * rho1 + (1.0 - eta) * rho0
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    return eta * rho1 / denominator

def find_interface_position_1d(x_coords, eta_values):
    """
    Find x-position where eta = 0.5 (interface location)
    """
    # Find where eta crosses 0.5
    idx = np.argmin(np.abs(eta_values - 0.5))
    return x_coords[idx]

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("1D PLANAR EVAPORATION WITH STEFAN FLOW - ANALYSIS")
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
all_mass_fraction_profiles = []
all_x_coords = []
all_analytical_profiles = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    
    # Extract 1D ray along x-axis at y=0
    y_mid = 0.0
    z_mid = 0.0
    x_min = float(ds.domain_left_edge[0])
    x_max = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([x_min, y_mid, z_mid], 'code_length')
    ray_end = ds.arr([x_max, y_mid, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by x-coordinate
    sort_indices = np.argsort(ray['x'])
    x_coords = np.array(ray['x'][sort_indices])
    eta_values = np.array(ray['eta'][sort_indices])
    
    # Calculate mass fraction
    mass_fraction = calculate_mass_fraction(eta_values, density0, density1)
    
    # Find interface position (where eta = 0.5)
    x_interface = find_interface_position_1d(x_coords, eta_values)
    
    # Calculate analytical solution
    Y_analytical = stefan_mass_fraction_profile(x_coords, x_interface, D_v, 
                                                Y_surface, Y_infinity, density0)
    
    times.append(t)
    interface_positions.append(x_interface)
    all_mass_fraction_profiles.append(mass_fraction)
    all_x_coords.append(x_coords)
    all_analytical_profiles.append(Y_analytical)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
interface_positions = np.array(interface_positions)

# ============================================================================
# SELECT 6 TIMESTEPS FOR 2x3 GRIDS
# ============================================================================

num_panels = 6
if len(plot_files) <= num_panels:
    selected_indices = list(range(len(plot_files)))
else:
    selected_indices = np.linspace(0, len(plot_files) - 1, num_panels, dtype=int)

selected_times = times[selected_indices]
print(f"\nSelected {num_panels} timesteps for 2x3 grids:")
for i, idx in enumerate(selected_indices):
    print(f"  Panel {i+1}: t = {times[idx]:.6e} s, x_interface = {interface_positions[idx]:.6e} m")

# ============================================================================
# PLOT 1: INTERFACE POSITION VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: INTERFACE POSITION VS TIME")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, interface_positions, 'bo-', linewidth=LINE_WIDTH_NUMERICAL, 
         markersize=6, label='Numerical Interface Position', alpha=0.7)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Interface Position vs Time\n1D Planar Stefan Flow', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '01_Interface_Position.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Interface_Position.eps'))
print("  Saved: 01_Interface_Position.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: MASS FRACTION COMPARISON (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: MASS FRACTION COMPARISON (2x3 GRID)")
print("=" * 70)

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
axes2 = axes2.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    x_coords = all_x_coords[idx]
    Y_num = all_mass_fraction_profiles[idx]
    Y_ana = all_analytical_profiles[idx]
    x_int = interface_positions[idx]
    
    axes2[i].plot(x_coords, Y_ana, 'b-', linewidth=LINE_WIDTH_EXACT, 
                  label='Analytical', zorder=1)
    axes2[i].plot(x_coords, Y_num, 'r--', linewidth=LINE_WIDTH_NUMERICAL, 
                  label='Numerical', zorder=2, alpha=0.8)
    axes2[i].axvline(x=x_int, color='g', linestyle=':', linewidth=1.5, 
                     label='Interface', alpha=0.7)
    
    axes2[i].set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
    axes2[i].set_ylabel('Mass Fraction Y', fontsize=FONT_SIZE_LABEL)
    axes2[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes2[i].legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    axes2[i].grid(True, alpha=0.3)
    axes2[i].tick_params(labelsize=FONT_SIZE_TICK)

fig2.suptitle('Mass Fraction Profile Comparison - Stefan Flow', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '02_Mass_Fraction_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Mass_Fraction_Comparison.eps'))
print("  Saved: 02_Mass_Fraction_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: MASS FRACTION ERROR (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: MASS FRACTION ERROR (2x3 GRID)")
print("=" * 70)

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
axes3 = axes3.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    x_coords = all_x_coords[idx]
    Y_num = all_mass_fraction_profiles[idx]
    Y_ana = all_analytical_profiles[idx]
    Y_error = np.abs(Y_num - Y_ana)
    
    axes3[i].plot(x_coords, Y_error, 'k-', linewidth=LINE_WIDTH_NUMERICAL)
    
    axes3[i].set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
    axes3[i].set_ylabel('|Y_num - Y_analytical|', fontsize=FONT_SIZE_LABEL)
    axes3[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes3[i].grid(True, alpha=0.3)
    axes3[i].tick_params(labelsize=FONT_SIZE_TICK)

fig3.suptitle('Mass Fraction Error - Stefan Flow', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '03_Mass_Fraction_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Mass_Fraction_Error.eps'))
print("  Saved: 03_Mass_Fraction_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: INTERFACE VELOCITY VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: INTERFACE VELOCITY VS TIME")
print("=" * 70)

# Calculate interface velocity from position
if len(times) > 1:
    interface_velocity = np.gradient(interface_positions, times)
    
    # Calculate analytical Stefan velocity
    analytical_velocity = []
    for x_int in interface_positions:
        delta = L_domain - x_int
        v_s = stefan_velocity(D_v, delta, Y_surface, Y_infinity)
        analytical_velocity.append(v_s)
    analytical_velocity = np.array(analytical_velocity)
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.plot(times, analytical_velocity, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical Stefan Velocity', zorder=1)
    ax4.plot(times, interface_velocity, 'ro', markersize=6, 
             label='Numerical Interface Velocity', alpha=0.7, zorder=2)
    
    ax4.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Interface Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
    ax4.set_title('Interface Velocity vs Time\n1D Planar Stefan Flow', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax4.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '04_Interface_Velocity.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '04_Interface_Velocity.eps'))
    print("  Saved: 04_Interface_Velocity.png/.eps")
    plt.close()

# ============================================================================
# PLOT 5: MASS FLUX VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 5: MASS FLUX VS TIME")
print("=" * 70)

# Calculate mass flux
if len(times) > 1:
    numerical_mass_flux = density0 * interface_velocity
    
    analytical_mass_flux = []
    for x_int in interface_positions:
        delta = L_domain - x_int
        m_dot = stefan_mass_flux(density0, D_v, delta, Y_surface, Y_infinity)
        analytical_mass_flux.append(m_dot)
    analytical_mass_flux = np.array(analytical_mass_flux)
    
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.plot(times, analytical_mass_flux, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical Mass Flux', zorder=1)
    ax5.plot(times, numerical_mass_flux, 'ro', markersize=6, 
             label='Numerical Mass Flux', alpha=0.7, zorder=2)
    
    ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax5.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
    ax5.set_title('Mass Flux vs Time\n1D Planar Stefan Flow', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax5.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '05_Mass_Flux.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '05_Mass_Flux.eps'))
    print("  Saved: 05_Mass_Flux.png/.eps")
    plt.close()

# ============================================================================
# PLOT 6: MASS FLUX ERROR (SEMILOG)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 6: MASS FLUX ERROR (SEMILOG)")
print("=" * 70)

if len(times) > 1:
    mass_flux_error = np.abs(numerical_mass_flux - analytical_mass_flux)
    epsilon = 1e-16
    mass_flux_error_safe = mass_flux_error + epsilon
    
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    ax6.semilogy(times, mass_flux_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)
    
    ax6.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax6.set_ylabel('|m_dot_num - m_dot_analytical| (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
    ax6.set_title('Mass Flux Error\n1D Planar Stefan Flow', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '06_Mass_Flux_Error.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '06_Mass_Flux_Error.eps'))
    print("  Saved: 06_Mass_Flux_Error.png/.eps")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Gas density (rho0): {density0} kg/m^3")
print(f"  Liquid density (rho1): {density1} kg/m^3")
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")

if len(times) > 1:
    print(f"\nResults:")
    print(f"  Initial interface position: {interface_positions[0]:.6e} m")
    print(f"  Final interface position: {interface_positions[-1]:.6e} m")
    print(f"  Average Stefan velocity: {np.mean(analytical_velocity):.6e} m/s")
    print(f"  Average mass flux: {np.mean(analytical_mass_flux):.6e} kg/m^2/s")
    print(f"  Max mass flux error: {np.max(mass_flux_error):.6e} kg/m^2/s")

print(f"\nFiles generated:")
print(f"  - 01_Interface_Position.png/.eps")
print(f"  - 02_Mass_Fraction_Comparison.png/.eps")
print(f"  - 03_Mass_Fraction_Error.png/.eps")
print(f"  - 04_Interface_Velocity.png/.eps")
print(f"  - 05_Mass_Flux.png/.eps")
print(f"  - 06_Mass_Flux_Error.png/.eps")

print("\n" + "=" * 70)
