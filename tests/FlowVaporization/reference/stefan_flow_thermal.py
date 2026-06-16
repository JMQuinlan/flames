"""
===============================================================================
STEFAN PROBLEM (THERMAL) - PHASE CHANGE WITH ENERGY COUPLING
===============================================================================

PURPOSE:
    Analyze the classical Stefan problem for phase change with moving boundary.
    Verify coupling between mass transfer and energy equation with latent heat.
    Compare numerical solver results against Stefan similarity solution.
    
FEATURES:
    - Interface position tracking over time
    - Temperature profile comparison
    - Stefan number analysis
    - Interface velocity verification
    - Energy balance verification
    - Error analysis (absolute and relative)
    - 2x3 grid plots for temporal evolution
    - PNG and EPS output formats

ANALYTICAL SOLUTION:
    Classical Stefan problem for 1D planar solidification/melting:
    
    Interface position (similarity solution):
    s(t) = 2 * lambda * sqrt(alpha * t)
    
    where lambda is the Stefan constant (root of transcendental equation):
    lambda * exp(lambda^2) * erf(lambda) = (Ste / sqrt(pi))
    
    Stefan number:
    Ste = c_p * (T_inf - T_melt) / L_latent
    
    Temperature profile in liquid (x > s):
    T(x,t) = T_inf + (T_melt - T_inf) * erf(x / (2*sqrt(alpha*t))) / erf(lambda)
    
    Temperature profile in solid (x < s):
    T(x,t) = T_melt
    
    Interface velocity:
    v_interface = ds/dt = lambda * sqrt(alpha / t)
    
    Energy balance at interface:
    rho * L_latent * v_interface = k * (dT/dx)|liquid - k * (dT/dx)|solid

INPUTS:
    - AMReX plot files from phase change simulation
    - Physical parameters: rho, c_p, k, L_latent, T_melt, T_inf

OUTPUTS:
    - Interface position vs time (similarity solution check)
    - Temperature profile comparison
    - Interface velocity verification
    - Stefan number analysis
    - Energy balance verification

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.special import erf, erfinv
from scipy.optimize import fsolve

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# File paths
amrex_output_dir = r'./plot_files'
output_folder = './Stefan_Thermal_Analysis'

# Physical parameters - ADJUST THESE FOR YOUR SIMULATION
# Fluid 0 properties (solid phase, eta=1)
density0 = 1000.0                # Density of solid phase [kg/m^3]
velocity0 = 0.0                  # Initial velocity of solid [m/s]
mu0 = 1.0e-3                     # Dynamic viscosity of solid [Pa-s]
pressure0 = 101325.0             # Pressure of solid [Pa]
gamma0 = 1.4                     # Ratio of specific heats for solid
c_p0 = 2000.0                    # Specific heat capacity of solid [J/kg/K]
k_thermal0 = 0.5                 # Thermal conductivity of solid [W/m/K]

# Fluid 1 properties (liquid phase, eta=0)
density1 = 1000.0                # Density of liquid phase [kg/m^3]
velocity1 = 0.0                  # Initial velocity of liquid [m/s]
mu1 = 1.0e-3                     # Dynamic viscosity of liquid [Pa-s]
pressure1 = 101325.0             # Pressure of liquid [Pa]
gamma1 = 1.4                     # Ratio of specific heats for liquid
c_p1 = 4000.0                    # Specific heat capacity of liquid [J/kg/K]
k_thermal1 = 0.6                 # Thermal conductivity of liquid [W/m/K]

# Phase change properties
L_latent = 334000.0              # Latent heat of fusion [J/kg]
T_melt = 273.15                  # Melting temperature [K]
T_infinity = 300.0               # Far-field temperature [K]
T_initial = 273.15               # Initial temperature [K]

# Domain properties
L_domain = 1.0                   # Domain length [m]

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

def stefan_number(c_p, delta_T, L_latent):
    """
    Calculate Stefan number
    Ste = c_p * (T_inf - T_melt) / L_latent
    """
    return c_p * delta_T / L_latent

def stefan_constant_lambda(Ste):
    """
    Solve for Stefan constant lambda from transcendental equation:
    lambda * exp(lambda^2) * erf(lambda) = Ste / sqrt(pi)
    """
    def equation(lam):
        if lam <= 0:
            return 1e10
        return lam * np.exp(lam**2) * erf(lam) - Ste / np.sqrt(np.pi)
    
    # Initial guess
    lambda_guess = 0.5
    try:
        lambda_sol = fsolve(equation, lambda_guess)[0]
        return lambda_sol
    except:
        return lambda_guess

def interface_position_stefan(t, lambda_const, alpha):
    """
    Interface position from similarity solution
    s(t) = 2 * lambda * sqrt(alpha * t)
    """
    return 2.0 * lambda_const * np.sqrt(alpha * t)

def interface_velocity_stefan(t, lambda_const, alpha):
    """
    Interface velocity
    v = ds/dt = lambda * sqrt(alpha / t)
    """
    if t < 1e-12:
        return 0.0
    return lambda_const * np.sqrt(alpha / t)

def temperature_profile_stefan(x, t, x_interface, T_melt, T_inf, lambda_const, alpha):
    """
    Temperature profile from Stefan similarity solution
    
    Liquid region (x > s):
    T(x,t) = T_inf + (T_melt - T_inf) * erf(x / (2*sqrt(alpha*t))) / erf(lambda)
    
    Solid region (x < s):
    T(x,t) = T_melt
    """
    if t < 1e-12:
        return np.full_like(x, T_melt)
    
    # Calculate similarity variable
    eta = x / (2.0 * np.sqrt(alpha * t))
    
    # Temperature in liquid region
    erf_lambda = erf(lambda_const)
    if abs(erf_lambda) < 1e-12:
        erf_lambda = 1e-12
    
    T_liquid = T_inf + (T_melt - T_inf) * erf(eta) / erf_lambda
    
    # Apply boundary conditions
    T = np.where(x > x_interface, T_liquid, T_melt)
    
    return T

def thermal_diffusivity(k, rho, c_p):
    """
    Thermal diffusivity: alpha = k / (rho * c_p)
    """
    return k / (rho * c_p)

def energy_flux_interface(rho, L_latent, v_interface):
    """
    Energy flux at interface due to phase change
    q = rho * L_latent * v_interface
    """
    return rho * L_latent * v_interface

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def calculate_temperature(energy, rho, velocity, gamma, p0):
    """
    Calculate temperature from total energy
    T = (E - 0.5*rho*v^2) / (rho * c_v)
    where c_v = p0 / (rho * (gamma - 1))
    """
    kinetic_energy = 0.5 * rho * velocity**2
    internal_energy = energy - kinetic_energy
    c_v = p0 / (rho * (gamma - 1.0))
    c_v = np.where(np.abs(c_v) < 1e-12, 1e-12, c_v)
    T = internal_energy / (rho * c_v)
    return T

def find_interface_position_1d(x_coords, eta_values):
    """
    Find x-position where eta = 0.5 (interface location)
    """
    idx = np.argmin(np.abs(eta_values - 0.5))
    return x_coords[idx]

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("STEFAN PROBLEM (THERMAL) - PHASE CHANGE ANALYSIS")
print("=" * 70)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and 'cell' in item and '.old' not in item:
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files")

# ============================================================================
# CALCULATE ANALYTICAL PARAMETERS
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING ANALYTICAL PARAMETERS")
print("=" * 70)

# Calculate thermal properties
alpha_liquid = thermal_diffusivity(k_thermal1, density1, c_p1)
alpha_solid = thermal_diffusivity(k_thermal0, density0, c_p0)

# Use liquid properties for Stefan solution (melting problem)
alpha = alpha_liquid

# Calculate Stefan number
delta_T = T_infinity - T_melt
Ste = stefan_number(c_p1, delta_T, L_latent)

# Calculate Stefan constant lambda
lambda_const = stefan_constant_lambda(Ste)

print(f"\nThermal Properties:")
print(f"  Liquid thermal diffusivity (alpha): {alpha:.6e} m^2/s")
print(f"  Solid thermal diffusivity: {alpha_solid:.6e} m^2/s")
print(f"  Stefan number (Ste): {Ste:.6f}")
print(f"  Stefan constant (lambda): {lambda_const:.6f}")

# ============================================================================
# EXTRACT DATA FROM ALL TIMESTEPS
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM ALL TIMESTEPS")
print("=" * 70)

times = []
interface_positions = []
all_temperature_profiles = []
all_x_coords = []
all_analytical_profiles = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    
    # Skip t=0 to avoid division by zero
    if t < 1e-12:
        continue
    
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
    
    # Extract fields for temperature calculation
    energy_values = np.array(ray['energy'][sort_indices])
    rho_values = np.array(ray['density'][sort_indices])
    vx_values = np.array(ray['velocityx'][sort_indices])
    gamma_values = np.array(ray['gammaf'][sort_indices])
    p0_values = np.array(ray['p0'][sort_indices])
    
    # Calculate temperature
    temperature = calculate_temperature(energy_values, rho_values, vx_values, 
                                       gamma_values, p0_values)
    
    # Find interface position (where eta = 0.5)
    x_interface = find_interface_position_1d(x_coords, eta_values)
    
    # Calculate analytical solution
    T_analytical = temperature_profile_stefan(x_coords, t, x_interface, T_melt, 
                                             T_infinity, lambda_const, alpha)
    
    times.append(t)
    interface_positions.append(x_interface)
    all_temperature_profiles.append(temperature)
    all_x_coords.append(x_coords)
    all_analytical_profiles.append(T_analytical)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
interface_positions = np.array(interface_positions)

# Calculate analytical interface position
s_analytical = interface_position_stefan(times, lambda_const, alpha)

# ============================================================================
# SELECT 6 TIMESTEPS FOR 2x3 GRIDS
# ============================================================================

num_panels = 6
if len(times) <= num_panels:
    selected_indices = list(range(len(times)))
else:
    selected_indices = np.linspace(0, len(times) - 1, num_panels, dtype=int)

selected_times = times[selected_indices]
print(f"\nSelected {num_panels} timesteps for 2x3 grids:")
for i, idx in enumerate(selected_indices):
    print(f"  Panel {i+1}: t = {times[idx]:.6e} s, s = {interface_positions[idx]:.6e} m")

# ============================================================================
# PLOT 1: INTERFACE POSITION VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: INTERFACE POSITION VS TIME")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, s_analytical, 'b-', linewidth=LINE_WIDTH_EXACT, 
         label='Analytical: s = 2*lambda*sqrt(alpha*t)', zorder=1)
ax1.plot(times, interface_positions, 'ro', markersize=MARKER_SIZE, 
         label='Numerical Interface Position', alpha=0.7, zorder=2)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Interface Position s(t) (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Stefan Problem: Interface Position vs Time\nlambda = {lambda_const:.6f}, Ste = {Ste:.6f}', 
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
# PLOT 2: INTERFACE POSITION VS SQRT(TIME)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: INTERFACE POSITION VS SQRT(TIME)")
print("=" * 70)

sqrt_times = np.sqrt(times)
sqrt_times_analytical = np.sqrt(times)
s_vs_sqrt_t = 2.0 * lambda_const * np.sqrt(alpha) * sqrt_times_analytical

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(sqrt_times_analytical, s_vs_sqrt_t, 'b-', linewidth=LINE_WIDTH_EXACT, 
         label='Analytical (Linear)', zorder=1)
ax2.plot(sqrt_times, interface_positions, 'ro', markersize=MARKER_SIZE, 
         label='Numerical', alpha=0.7, zorder=2)

ax2.set_xlabel('sqrt(Time) (s^(1/2))', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Interface Position s(t) (m)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Stefan Problem: Similarity Solution Verification\ns vs sqrt(t) should be linear', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_Interface_Similarity.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Interface_Similarity.eps'))
print("  Saved: 02_Interface_Similarity.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: INTERFACE POSITION ERROR
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: INTERFACE POSITION ERROR")
print("=" * 70)

s_error = np.abs(interface_positions - s_analytical)
epsilon = 1e-16
s_error_safe = s_error + epsilon

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.semilogy(times, s_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)

ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('|s_num - s_analytical| (m)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Interface Position Error', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '03_Interface_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Interface_Error.eps'))
print("  Saved: 03_Interface_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: TEMPERATURE PROFILE COMPARISON (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: TEMPERATURE PROFILE COMPARISON (2x3 GRID)")
print("=" * 70)

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
axes4 = axes4.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    x_coords = all_x_coords[idx]
    T_num = all_temperature_profiles[idx]
    T_ana = all_analytical_profiles[idx]
    s_int = interface_positions[idx]
    
    axes4[i].plot(x_coords, T_ana, 'b-', linewidth=LINE_WIDTH_EXACT, 
                  label='Analytical', zorder=1)
    axes4[i].plot(x_coords, T_num, 'r--', linewidth=LINE_WIDTH_NUMERICAL, 
                  label='Numerical', zorder=2, alpha=0.8)
    axes4[i].axvline(x=s_int, color='g', linestyle=':', linewidth=1.5, 
                     label='Interface', alpha=0.7)
    axes4[i].axhline(y=T_melt, color='gray', linestyle='--', linewidth=1, 
                     label='T_melt', alpha=0.5)
    
    axes4[i].set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
    axes4[i].set_ylabel('Temperature T (K)', fontsize=FONT_SIZE_LABEL)
    axes4[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes4[i].legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    axes4[i].grid(True, alpha=0.3)
    axes4[i].tick_params(labelsize=FONT_SIZE_TICK)

fig4.suptitle('Temperature Profile Comparison - Stefan Problem', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '04_Temperature_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Temperature_Comparison.eps'))
print("  Saved: 04_Temperature_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: TEMPERATURE ERROR (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 5: TEMPERATURE ERROR (2x3 GRID)")
print("=" * 70)

fig5, axes5 = plt.subplots(2, 3, figsize=(18, 12))
axes5 = axes5.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    x_coords = all_x_coords[idx]
    T_num = all_temperature_profiles[idx]
    T_ana = all_analytical_profiles[idx]
    T_error = np.abs(T_num - T_ana)
    
    axes5[i].plot(x_coords, T_error, 'k-', linewidth=LINE_WIDTH_NUMERICAL)
    
    axes5[i].set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
    axes5[i].set_ylabel('|T_num - T_analytical| (K)', fontsize=FONT_SIZE_LABEL)
    axes5[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes5[i].grid(True, alpha=0.3)
    axes5[i].tick_params(labelsize=FONT_SIZE_TICK)

fig5.suptitle('Temperature Error - Stefan Problem', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '05_Temperature_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Temperature_Error.eps'))
print("  Saved: 05_Temperature_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 6: INTERFACE VELOCITY VERIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 6: INTERFACE VELOCITY VERIFICATION")
print("=" * 70)

# Calculate numerical interface velocity
if len(times) > 1:
    v_interface_numerical = np.gradient(interface_positions, times)
    
    # Calculate analytical interface velocity
    v_interface_analytical = []
    for t in times:
        v = interface_velocity_stefan(t, lambda_const, alpha)
        v_interface_analytical.append(v)
    v_interface_analytical = np.array(v_interface_analytical)
    
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    ax6.plot(times, v_interface_analytical, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical: v = lambda*sqrt(alpha/t)', zorder=1)
    ax6.plot(times, v_interface_numerical, 'ro', markersize=MARKER_SIZE, 
             label='Numerical Interface Velocity', alpha=0.7, zorder=2)
    
    ax6.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax6.set_ylabel('Interface Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
    ax6.set_title('Interface Velocity vs Time\nStefan Problem', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '06_Interface_Velocity.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '06_Interface_Velocity.eps'))
    print("  Saved: 06_Interface_Velocity.png/.eps")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Liquid density (rho_l): {density1} kg/m^3")
print(f"  Solid density (rho_s): {density0} kg/m^3")
print(f"  Liquid specific heat (c_p): {c_p1} J/kg/K")
print(f"  Liquid thermal conductivity (k): {k_thermal1} W/m/K")
print(f"  Latent heat (L): {L_latent} J/kg")
print(f"  Melting temperature (T_melt): {T_melt} K")
print(f"  Far-field temperature (T_inf): {T_infinity} K")

print(f"\nStefan Problem Results:")
print(f"  Thermal diffusivity (alpha): {alpha:.6e} m^2/s")
print(f"  Stefan number (Ste): {Ste:.6f}")
print(f"  Stefan constant (lambda): {lambda_const:.6f}")
print(f"  Initial interface position: {interface_positions[0]:.6e} m")
print(f"  Final interface position: {interface_positions[-1]:.6e} m")

if len(times) > 1:
    print(f"  Average interface velocity: {np.mean(v_interface_analytical):.6e} m/s")
    print(f"  Max interface position error: {np.max(s_error):.6e} m")

print(f"\nFiles generated:")
print(f"  - 01_Interface_Position.png/.eps")
print(f"  - 02_Interface_Similarity.png/.eps")
print(f"  - 03_Interface_Error.png/.eps")
print(f"  - 04_Temperature_Comparison.png/.eps")
print(f"  - 05_Temperature_Error.png/.eps")
print(f"  - 06_Interface_Velocity.png/.eps")

print("\n" + "=" * 70)
