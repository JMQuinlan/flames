"""
===============================================================================
1D PLANAR EVAPORATION WITH TRANSIENT STEFAN FLOW - COMPREHENSIVE ANALYSIS SCRIPT
===============================================================================
PURPOSE:
    Analyze 1D planar evaporation into stagnant gas with transient Stefan flow.
    Compare numerical solver results against analytical transient Stefan solution.
    
FEATURES:
    - Comparison plots of mass fraction profiles
    - Interface position tracking over time (s(t) = 2*lambda*sqrt(D_AB*t))
    - Mass flux verification
    - Error analysis (absolute and relative)
    - 2x3 grid plots for temporal evolution
    - PNG and EPS output formats

ANALYTICAL SOLUTION (TRANSIENT SIMILARITY SOLUTION):
    Transient Stefan flow for 1D planar evaporation (semi-infinite domain):
    
    Mass fraction profile:
    Y(x,t) = Y_inf + (Y_s - Y_inf) * erfc((x-s(t))/(2*sqrt(D_v*t))) / erfc(lambda)
    
    Interface position:
    s(t) = 2*lambda*sqrt(D_v*t)
    
    Stefan number lambda determined from transcendental equation:
    lambda * exp(lambda^2) * erfc(lambda) = (rho_gas/rho_liq) * (Y_s - Y_inf) / (1 - Y_s)
    
    Mass flux:
    m_dot = rho_liq * 2 * lambda * sqrt(D_v / (pi*t))

VALIDITY:
    - Semi-infinite domain: L - s(t) >> sqrt(D_v*t)
    - Early times before steady-state
    - Constant surface concentration

INPUTS:
    - AMReX plot files from evaporation simulation
    - Physical parameters: rho_gas, rho_liq, D_v, Y_s, Y_inf

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
from scipy.special import erfc, erfcx
from scipy.optimize import fsolve

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# File paths
amrex_output_dir = r'../../../bin/tests/stefan/stefan_1'
output_folder = './Stefan_Flow_1D_PostProcess_Transient'

# Physical parameters - ADJUST THESE FOR YOUR SIMULATION
# Fluid 0 properties (vapor/gas phase, eta=1)
density0 = 1.0                   # Density of gas phase [kg/m^3]
velocity0 = 0.0                  # Initial velocity of gas [m/s]
mu0 = 1.0e-5                     # Dynamic viscosity of gas [Pa-s]
pressure0 = 101325.0             # Pressure of gas [Pa]
gamma0 = 1.4                     # Ratio of specific heats for gas

# Fluid 1 properties (liquid phase, eta=0)
density1 = 100.0                # Density of liquid [kg/m^3]
velocity1 = 0.0                  # Initial velocity of liquid [m/s]
mu1 = 1.0e-3                     # Dynamic viscosity of liquid [Pa-s]
pressure1 = 101325.0             # Pressure of liquid [Pa]
gamma1 = 1.4                     # Ratio of specific heats for liquid

# Mass transfer properties
D_v = 2.0e-5                     # Binary diffusion coefficient [m^2/s]
Y_surface = density0 / (density0 + density1)                # Mass fraction at liquid surface (interface)
Y_infinity = 0.0 #density1 / (density0 + density1)                  # Mass fraction far from interface

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
# TRANSIENT STEFAN FLOW ANALYTICAL SOLUTION FUNCTIONS
# ============================================================================

def solve_stefan_number(Y_s, Y_inf, rho_gas, rho_liq):
    """
    Solve transcendental equation for Stefan number lambda:
    lambda * exp(lambda^2) * erfc(lambda) = (rho_gas/rho_liq) * (Y_s - Y_inf) / (1 - Y_s)
    
    This is the correct formulation for liquid-gas Stefan flow with density ratio.
    
    Returns:
        lambda: Stefan number (eigenvalue)
    """
    # Right-hand side of equation (includes density ratio)
    A = (rho_gas / rho_liq) * (Y_s - Y_inf) / (1.0 - Y_s)
    
    print(f"\nStefan Number Calculation:")
    print(f"  Density ratio (rho_gas/rho_liq): {rho_gas/rho_liq:.6e}")
    print(f"  Mass fraction difference (Y_s - Y_inf): {Y_s - Y_inf:.6e}")
    print(f"  Denominator (1 - Y_s): {1.0 - Y_s:.6e}")
    print(f"  Parameter A = (rho_gas/rho_liq)*(Y_s-Y_inf)/(1-Y_s): {A:.6e}")
    
    # Define equation to solve: f(lambda) = 0
    def lambda_equation(lam):
        if lam <= 0:
            return 1e10  # Penalize negative values
        # lambda * exp(lambda^2) * erfc(lambda) - A = 0
        # Use erfcx for stability: erfcx(x) = exp(x^2) * erfc(x)
        lhs = lam * erfcx(lam)
        return lhs - A
    
    # Initial guess
    lambda_init = max(0.01, A)
    
    # Solve using fsolve
    lambda_solution = fsolve(lambda_equation, lambda_init)[0]
    
    return lambda_solution

def transient_interface_position(t, D_v, lambda_stefan):
    """
    Calculate interface position for transient Stefan flow
    s(t) = 2*lambda*sqrt(D_v*t)
    
    Args:
        t: time [s]
        D_v: diffusion coefficient [m^2/s]
        lambda_stefan: Stefan number
    
    Returns:
        s(t): interface displacement from initial position [m]
    """
    if t <= 0:
        return 0.0
    return 2.0 * lambda_stefan * np.sqrt(D_v * t)

def transient_stefan_mass_fraction_profile(x, t, D_v, Y_s, Y_inf, lambda_stefan, x_initial=0.0):
    """
    Transient mass fraction profile with Stefan flow (similarity solution)
    
    Y(x,t) = Y_inf + (Y_s - Y_inf) * erfc((x-s(t))/(2*sqrt(D_v*t))) / erfc(lambda)
    
    For x < s(t): Y = Y_s (liquid surface)
    For x > s(t): Y transitions from Y_s to Y_inf
    
    Args:
        x: spatial coordinate [m]
        t: time [s]
        D_v: diffusion coefficient [m^2/s]
        Y_s: surface mass fraction
        Y_inf: far-field mass fraction
        lambda_stefan: Stefan number
        x_initial: initial interface position [m]
    
    Returns:
        Y(x,t): mass fraction profile
    """
    if t <= 1e-12:
        # At t=0, step function
        return np.where(x > x_initial, Y_inf, Y_s)
    
    # Current interface position
    sqrt_Dt = np.sqrt(D_v * t)
    s_t = x_initial + 2.0 * lambda_stefan * sqrt_Dt
    
    # Similarity variable: eta = (x - s(t)) / (2*sqrt(D_v*t))
    eta = (x - s_t) / (2.0 * sqrt_Dt)
    
    # Mass fraction profile
    erfc_eta = erfc(eta)
    erfc_lambda = erfc(lambda_stefan)
    
    # Avoid division by zero
    if abs(erfc_lambda) < 1e-12:
        erfc_lambda = 1e-12
    
    Y = Y_inf + (Y_s - Y_inf) * erfc_eta / erfc_lambda
    
    # For x < s(t), set Y = Y_s (liquid surface value)
    Y = np.where(x < s_t, Y_s, Y)
    
    # Clip to physical bounds
    Y = np.clip(Y, min(Y_s, Y_inf), max(Y_s, Y_inf))
    
    return Y

def transient_stefan_mass_flux(t, rho_liq, D_v, lambda_stefan):
    """
    Calculate transient mass flux at the interface
    
    m_dot(t) = rho_liq * 2 * lambda * sqrt(D_v / (pi*t))
    
    Args:
        t: time [s]
        rho_liq: liquid density [kg/m^3]
        D_v: diffusion coefficient [m^2/s]
        lambda_stefan: Stefan number
    
    Returns:
        m_dot: mass flux [kg/m^2/s]
    """
    if t <= 1e-12:
        return 0.0
    
    m_dot = rho_liq * 2.0 * lambda_stefan * np.sqrt(D_v / (np.pi * t))
    
    return m_dot

def transient_stefan_velocity(t, D_v, lambda_stefan):
    """
    Calculate transient interface velocity
    
    v_s(t) = ds/dt = lambda * sqrt(D_v / (pi*t))
    
    Args:
        t: time [s]
        D_v: diffusion coefficient [m^2/s]
        lambda_stefan: Stefan number
    
    Returns:
        v_s: interface velocity [m/s]
    """
    if t <= 1e-12:
        return 0.0
    
    v_s = lambda_stefan * np.sqrt(D_v / (np.pi * t))
    
    return v_s

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
    Calculate mass fraction of gas phase from volume fraction
    
    eta = 1: pure gas (Y = 1.0)
    eta = 0: pure liquid (Y = 0.0)
    
    Y = (eta * rho0) / (eta * rho0 + (1-eta) * rho1)
    """
    denominator = rho0 + rho1
    denominator = np.where(np.abs(denominator) < 1e-12, 1e-12, denominator)
    numerator = (eta) * rho0 # rho
    return  numerator / denominator

def find_interface_position_1d(x_coords, eta_values):
    """
    Find x-position where eta = 0.5 (interface location)
    """
    # Find where eta crosses 0.5
    idx = np.argmin(np.abs(eta_values - 0.5))
    return x_coords[idx]

# ============================================================================
# CALCULATE STEFAN NUMBER
# ============================================================================

print("=" * 70)
print("1D PLANAR EVAPORATION WITH TRANSIENT STEFAN FLOW - ANALYSIS")
print("=" * 70)

print("\nCalculating Stefan number lambda...")
lambda_stefan = solve_stefan_number(Y_surface, Y_infinity, density0, density1)
print(f"  Stefan number lambda = {lambda_stefan:.6f}")

# Verify solution
A = (density0 / density1) * (Y_surface - Y_infinity) / (1.0 - Y_surface)
lhs = lambda_stefan * erfcx(lambda_stefan)
print(f"  Verification: LHS = {lhs:.6e}, A = {A:.6e}, Error = {abs(lhs-A):.6e}")

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

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
x_initial = None  # Will be set from first timestep

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
    rho_values = np.array(ray['density'][sort_indices])
    
    # Calculate mass fraction
    mass_fraction = calculate_mass_fraction(eta_values, density0, density1)
    
    # Find interface position (where eta = 0.5)
    x_interface = find_interface_position_1d(x_coords, eta_values)
    
    # Set initial interface position from first timestep
    if x_initial is None:
        x_initial = x_interface
        print(f"  Initial interface position: x_initial = {x_initial:.6e} m")
    
    # Calculate analytical solution
    Y_analytical = transient_stefan_mass_fraction_profile(
        x_coords, t, D_v, Y_surface, Y_infinity, lambda_stefan, x_initial
    )
    
    times.append(t)
    interface_positions.append(x_interface)
    all_mass_fraction_profiles.append(mass_fraction)
    all_x_coords.append(x_coords)
    all_analytical_profiles.append(Y_analytical)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
interface_positions = np.array(interface_positions)

# Calculate analytical interface positions
analytical_interface_positions = np.array([
    x_initial + transient_interface_position(t, D_v, lambda_stefan) 
    for t in times
])

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

ax1.plot(times, analytical_interface_positions, 'b-', linewidth=LINE_WIDTH_EXACT, 
         label='Analytical: s(t) = x0 + 2*lambda*sqrt(Dt)', zorder=1)
ax1.plot(times, interface_positions, 'ro', markersize=6, 
         label='Numerical Interface Position', alpha=0.7, zorder=2)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Interface Position vs Time\n1D Transient Stefan Flow (Similarity Solution)', 
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
    x_int_ana = analytical_interface_positions[idx]
    
    axes2[i].plot(x_coords, Y_ana, 'b-', linewidth=LINE_WIDTH_EXACT, 
                  label='Analytical (erfc)', zorder=1)
    axes2[i].plot(x_coords, Y_num, 'r--', linewidth=LINE_WIDTH_NUMERICAL, 
                  label='Numerical', zorder=2, alpha=0.8)
    axes2[i].axvline(x=x_int, color='r', linestyle=':', linewidth=1.5, 
                     label='Interface (Num)', alpha=0.7)
    axes2[i].axvline(x=x_int_ana, color='b', linestyle=':', linewidth=1.5, 
                     label='Interface (Ana)', alpha=0.7)
    
    axes2[i].set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
    axes2[i].set_ylabel('Mass Fraction Y', fontsize=FONT_SIZE_LABEL)
    axes2[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes2[i].legend(fontsize=FONT_SIZE_LEGEND-2, loc='best')
    axes2[i].grid(True, alpha=0.3)
    axes2[i].tick_params(labelsize=FONT_SIZE_TICK)

fig2.suptitle('Mass Fraction Profile Comparison - Transient Stefan Flow', 
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

fig3.suptitle('Mass Fraction Error - Transient Stefan Flow', 
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
    
    # Calculate analytical transient Stefan velocity
    analytical_velocity = np.array([
        transient_stefan_velocity(t, D_v, lambda_stefan)
        for t in times
    ])
    
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.plot(times, analytical_velocity, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical: v_s(t) = lambda*sqrt(D/(pi*t))', zorder=1)
    ax4.plot(times, interface_velocity, 'ro', markersize=6, 
             label='Numerical Interface Velocity', alpha=0.7, zorder=2)
    
    ax4.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Interface Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
    ax4.set_title('Interface Velocity vs Time\n1D Transient Stefan Flow', 
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
    numerical_mass_flux = density1 * interface_velocity
    
    analytical_mass_flux = np.array([
        transient_stefan_mass_flux(t, density1, D_v, lambda_stefan)
        for t in times
    ])
    
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    ax5.plot(times, analytical_mass_flux, 'b-', linewidth=LINE_WIDTH_EXACT, 
             label='Analytical: m_dot = rho*2*lambda*sqrt(D/(pi*t))', zorder=1)
    ax5.plot(times, numerical_mass_flux, 'ro', markersize=6, 
             label='Numerical Mass Flux', alpha=0.7, zorder=2)
    
    ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
    ax5.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
    ax5.set_title('Mass Flux vs Time\n1D Transient Stefan Flow', 
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
    ax6.set_title('Mass Flux Error\n1D Transient Stefan Flow', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '06_Mass_Flux_Error.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '06_Mass_Flux_Error.eps'))
    print("  Saved: 06_Mass_Flux_Error.png/.eps")
    plt.close()

# ============================================================================
# PLOT 7: INTERFACE POSITION ERROR
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 7: INTERFACE POSITION ERROR")
print("=" * 70)

interface_position_error = np.abs(interface_positions - analytical_interface_positions)

fig7, ax7 = plt.subplots(figsize=(10, 8))
ax7.plot(times, interface_position_error, 'k-', linewidth=LINE_WIDTH_NUMERICAL)

ax7.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax7.set_ylabel('|s_num - s_analytical| (m)', fontsize=FONT_SIZE_LABEL)
ax7.set_title('Interface Position Error\n1D Transient Stefan Flow', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '07_Interface_Position_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '07_Interface_Position_Error.eps'))
print("  Saved: 07_Interface_Position_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 8: SQRT(t) SCALING VERIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 8: SQRT(t) SCALING VERIFICATION")
print("=" * 70)

# Plot s(t) vs sqrt(t) to verify linear relationship
sqrt_times = np.sqrt(times)

fig8, ax8 = plt.subplots(figsize=(10, 8))
ax8.plot(sqrt_times, analytical_interface_positions - x_initial, 'b-', 
         linewidth=LINE_WIDTH_EXACT, label='Analytical: 2*lambda*sqrt(Dt)', zorder=1)
ax8.plot(sqrt_times, interface_positions - x_initial, 'ro', markersize=6, 
         label='Numerical', alpha=0.7, zorder=2)

ax8.set_xlabel('sqrt(t) (s^0.5)', fontsize=FONT_SIZE_LABEL)
ax8.set_ylabel('s(t) - s0 (m)', fontsize=FONT_SIZE_LABEL)
ax8.set_title('Interface Displacement vs sqrt(t)\nVerification of Similarity Solution', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax8.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax8.grid(True, alpha=0.3)
ax8.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '08_Sqrt_t_Scaling.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '08_Sqrt_t_Scaling.eps'))
print("  Saved: 08_Sqrt_t_Scaling.png/.eps")
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
print(f"  Density ratio (rho0/rho1): {density0/density1:.6e}")
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")
print(f"\nTransient Stefan Flow Parameters:")
print(f"  Stefan number (lambda): {lambda_stefan:.6f}")
print(f"  Parameter A = (rho_gas/rho_liq)*(Y_s-Y_inf)/(1-Y_s): {A:.6e}")

if len(times) > 1:
    print(f"\nResults:")
    print(f"  Initial interface position: {interface_positions[0]:.6e} m")
    print(f"  Final interface position: {interface_positions[-1]:.6e} m")
    print(f"  Interface displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")
    print(f"  Initial Stefan velocity: {analytical_velocity[0]:.6e} m/s")
    print(f"  Final Stefan velocity: {analytical_velocity[-1]:.6e} m/s")
    print(f"  Initial mass flux: {analytical_mass_flux[0]:.6e} kg/m^2/s")
    print(f"  Final mass flux: {analytical_mass_flux[-1]:.6e} kg/m^2/s")
    print(f"\nError Metrics:")
    print(f"  Max interface position error: {np.max(interface_position_error):.6e} m")
    print(f"  Mean interface position error: {np.mean(interface_position_error):.6e} m")
    print(f"  Max mass flux error: {np.max(mass_flux_error):.6e} kg/m^2/s")
    print(f"  Mean mass flux error: {np.mean(mass_flux_error):.6e} kg/m^2/s")
    
    # Check validity of semi-infinite approximation
    final_penetration = np.sqrt(D_v * times[-1])
    final_distance_to_boundary = L_domain - interface_positions[-1]
    print(f"\nValidity Check (Semi-Infinite Approximation):")
    print(f"  Diffusion penetration depth sqrt(Dt): {final_penetration:.6e} m")
    print(f"  Distance to boundary (L - s): {final_distance_to_boundary:.6e} m")
    print(f"  Ratio (L-s)/sqrt(Dt): {final_distance_to_boundary/final_penetration:.2f}")
    if final_distance_to_boundary / final_penetration > 3:
        print(f"  Semi-infinite approximation is VALID (ratio >> 1)")
    else:
        print(f"  Semi-infinite approximation may be QUESTIONABLE (ratio should be >> 1)")

print(f"\nFiles generated:")
print(f"  - 01_Interface_Position.png/.eps")
print(f"  - 02_Mass_Fraction_Comparison.png/.eps")
print(f"  - 03_Mass_Fraction_Error.png/.eps")
print(f"  - 04_Interface_Velocity.png/.eps")
print(f"  - 05_Mass_Flux.png/.eps")
print(f"  - 06_Mass_Flux_Error.png/.eps")
print(f"  - 07_Interface_Position_Error.png/.eps")
print(f"  - 08_Sqrt_t_Scaling.png/.eps")

print("\n" + "=" * 70)
