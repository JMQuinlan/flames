"""
===============================================================================
SPHERICAL DROPLET EVAPORATION - D^2-LAW VERIFICATION
===============================================================================

PURPOSE:
    Analyze spherical droplet evaporation and verify the classical D^2-law.
    Compare numerical solver results against analytical quasi-steady solution.
    
FEATURES:
    - Droplet radius tracking over time
    - D^2-law verification (D^2 vs time should be linear)
    - Mass fraction profile comparison (radial)
    - Evaporation rate analysis
    - Error analysis (absolute and relative)
    - 2x3 grid plots for temporal evolution
    - PNG and EPS output formats

ANALYTICAL SOLUTION:
    Classical D^2-law for quasi-steady droplet evaporation:
    
    D^2(t) = D0^2 - K * t
    
    where K is the evaporation constant:
    K = (8 * rho_g * D_v / rho_l) * ln[(1 - Y_inf) / (1 - Y_s)]
    
    Droplet lifetime:
    t_life = D0^2 / K
    
    Mass fraction profile (spherical coordinates):
    Y(r) = Y_inf + (Y_s - Y_inf) * [(r_s / r) * (r_inf - r) / (r_inf - r_s)]
    
    With Stefan flow correction:
    Y(r) = Y_inf + (Y_s - Y_inf) * [(r_s / r) * ((r/r_inf)^B - 1) / ((r_s/r_inf)^B - 1)]
    where B = ln[(1 - Y_inf) / (1 - Y_s)]

INPUTS:
    - AMReX plot files from droplet evaporation simulation
    - Physical parameters: rho_l, rho_g, D_v, Y_s, Y_inf

OUTPUTS:
    - Droplet radius vs time
    - D^2 vs time (linearity check)
    - Radial mass fraction profiles
    - Evaporation rate verification
    - Error analysis plots

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
output_folder = './D2_Law_Analysis'

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
Y_surface = 0.9                  # Mass fraction at droplet surface (eta=0.5)
Y_infinity = 0.1                 # Mass fraction far from droplet

# Domain properties
L_domain = 1.0                   # Domain length [m]
droplet_center_x = 0.0           # Droplet center x-coordinate [m]
droplet_center_y = 0.0           # Droplet center y-coordinate [m]

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

def evaporation_constant_K(rho_g, rho_l, D_v, Y_s, Y_inf):
    """
    Calculate evaporation constant K for D^2-law
    K = (8 * rho_g * D_v / rho_l) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0:
        return 0.0
    return (8.0 * rho_g * D_v / rho_l) * np.log((1.0 - Y_inf) / (1.0 - Y_s))

def droplet_diameter_squared(D0_squared, K, t):
    """
    D^2-law: D^2(t) = D0^2 - K * t
    """
    return D0_squared - K * t

def droplet_lifetime(D0_squared, K):
    """
    Droplet lifetime: t_life = D0^2 / K
    """
    if K < 1e-16:
        return np.inf
    return D0_squared / K

def mass_fraction_profile_spherical(r, r_s, r_inf, Y_s, Y_inf):
    """
    Analytical mass fraction profile for spherical droplet with Stefan flow
    
    Y(r) = Y_inf + (Y_s - Y_inf) * [(r_s / r) * ((r/r_inf)^B - 1) / ((r_s/r_inf)^B - 1)]
    where B = ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0:
        return np.full_like(r, Y_inf)
    
    B = np.log((1.0 - Y_inf) / (1.0 - Y_s))
    
    # Avoid division by zero
    r = np.where(r < r_s, r_s, r)
    
    # Calculate profile
    if abs(B) < 1e-6:
        # Linear approximation for small B
        Y = Y_inf + (Y_s - Y_inf) * (r_s / r) * (r_inf - r) / (r_inf - r_s)
    else:
        numerator = (r / r_inf)**B - 1.0
        denominator = (r_s / r_inf)**B - 1.0
        if abs(denominator) < 1e-12:
            denominator = 1e-12
        Y = Y_inf + (Y_s - Y_inf) * (r_s / r) * (numerator / denominator)
    
    # Apply boundary conditions
    Y = np.where(r <= r_s, Y_s, Y)
    Y = np.where(r >= r_inf, Y_inf, Y)
    
    return Y

def evaporation_rate(rho_l, K):
    """
    Mass evaporation rate: dm/dt = -pi * rho_l * K
    """
    return -np.pi * rho_l * K

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

def find_droplet_radius_2d(ds, center_x, center_y):
    """
    Find droplet radius by locating eta = 0.5 contour
    Extract radial distance from center where eta = 0.5
    """
    # Extract 1D ray from center outward along x-axis
    z_mid = 0.0
    x_min = float(ds.domain_left_edge[0])
    x_max = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([center_x, center_y, z_mid], 'code_length')
    ray_end = ds.arr([x_max, center_y, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by distance from center
    x_coords = np.array(ray['x'])
    y_coords = np.array(ray['y'])
    r_coords = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    eta_values = np.array(ray['eta'])
    
    sort_indices = np.argsort(r_coords)
    r_sorted = r_coords[sort_indices]
    eta_sorted = eta_values[sort_indices]
    
    # Find where eta = 0.5
    idx = np.argmin(np.abs(eta_sorted - 0.5))
    radius = r_sorted[idx]
    
    return radius

def extract_radial_profile(ds, center_x, center_y):
    """
    Extract radial profiles of eta and mass fraction
    """
    z_mid = 0.0
    x_max = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([center_x, center_y, z_mid], 'code_length')
    ray_end = ds.arr([x_max, center_y, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    x_coords = np.array(ray['x'])
    y_coords = np.array(ray['y'])
    r_coords = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    eta_values = np.array(ray['eta'])
    
    sort_indices = np.argsort(r_coords)
    r_sorted = r_coords[sort_indices]
    eta_sorted = eta_values[sort_indices]
    
    return r_sorted, eta_sorted

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("SPHERICAL DROPLET EVAPORATION - D^2-LAW VERIFICATION")
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
# EXTRACT DATA FROM ALL TIMESTEPS
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM ALL TIMESTEPS")
print("=" * 70)

times = []
droplet_radii = []
all_radial_coords = []
all_mass_fraction_profiles = []
all_analytical_profiles = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    
    # Find droplet radius
    radius = find_droplet_radius_2d(ds, droplet_center_x, droplet_center_y)
    
    # Extract radial profile
    r_coords, eta_values = extract_radial_profile(ds, droplet_center_x, droplet_center_y)
    
    # Calculate mass fraction
    mass_fraction = calculate_mass_fraction(eta_values, density0, density1)
    
    # Calculate analytical solution
    r_inf = L_domain / 2.0  # Assume far-field is half domain size
    Y_analytical = mass_fraction_profile_spherical(r_coords, radius, r_inf, 
                                                    Y_surface, Y_infinity)
    
    times.append(t)
    droplet_radii.append(radius)
    all_radial_coords.append(r_coords)
    all_mass_fraction_profiles.append(mass_fraction)
    all_analytical_profiles.append(Y_analytical)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
droplet_radii = np.array(droplet_radii)

# Calculate D^2 values
droplet_diameters = 2.0 * droplet_radii
D_squared = droplet_diameters**2
D0_squared = D_squared[0]

# Calculate evaporation constant K
K_analytical = evaporation_constant_K(density0, density1, D_v, Y_surface, Y_infinity)

# Calculate analytical D^2(t)
D_squared_analytical = droplet_diameter_squared(D0_squared, K_analytical, times)

# Calculate droplet lifetime
t_life = droplet_lifetime(D0_squared, K_analytical)

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
    print(f"  Panel {i+1}: t = {times[idx]:.6e} s, R = {droplet_radii[idx]:.6e} m")

# ============================================================================
# PLOT 1: DROPLET RADIUS VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: DROPLET RADIUS VS TIME")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, droplet_radii, 'bo-', linewidth=LINE_WIDTH_NUMERICAL, 
         markersize=MARKER_SIZE, label='Numerical Radius', alpha=0.7)

# Analytical radius from D^2-law
R_analytical = np.sqrt(D_squared_analytical) / 2.0
ax1.plot(times, R_analytical, 'r-', linewidth=LINE_WIDTH_EXACT, 
         label='Analytical (D^2-law)', zorder=1)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Droplet Radius (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Droplet Radius vs Time\nSpherical Evaporation', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '01_Droplet_Radius.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Droplet_Radius.eps'))
print("  Saved: 01_Droplet_Radius.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: D^2 VS TIME (D^2-LAW VERIFICATION)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: D^2 VS TIME (D^2-LAW VERIFICATION)")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(times, D_squared, 'bo', markersize=MARKER_SIZE, 
         label='Numerical D^2', alpha=0.7, zorder=2)
ax2.plot(times, D_squared_analytical, 'r-', linewidth=LINE_WIDTH_EXACT, 
         label=f'Analytical: D^2 = D0^2 - K*t\nK = {K_analytical:.6e} m^2/s', zorder=1)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Droplet Diameter Squared D^2 (m^2)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('D^2-Law Verification\nLinear Decay of D^2 with Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

# Add lifetime annotation
if t_life < np.inf:
    ax2.axvline(x=t_life, color='g', linestyle='--', linewidth=1.5, 
                label=f'Predicted lifetime = {t_life:.4e} s', alpha=0.7)

plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_D2_Law_Verification.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_D2_Law_Verification.eps'))
print("  Saved: 02_D2_Law_Verification.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: D^2 ERROR VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: D^2 ERROR VS TIME")
print("=" * 70)

D2_error = np.abs(D_squared - D_squared_analytical)
epsilon = 1e-16
D2_error_safe = D2_error + epsilon

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.semilogy(times, D2_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)

ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('|D^2_num - D^2_analytical| (m^2)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('D^2-Law Error', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '03_D2_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_D2_Error.eps'))
print("  Saved: 03_D2_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: RADIAL MASS FRACTION COMPARISON (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: RADIAL MASS FRACTION COMPARISON (2x3 GRID)")
print("=" * 70)

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))
axes4 = axes4.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    r_coords = all_radial_coords[idx]
    Y_num = all_mass_fraction_profiles[idx]
    Y_ana = all_analytical_profiles[idx]
    R_drop = droplet_radii[idx]
    
    axes4[i].plot(r_coords, Y_ana, 'b-', linewidth=LINE_WIDTH_EXACT, 
                  label='Analytical', zorder=1)
    axes4[i].plot(r_coords, Y_num, 'r--', linewidth=LINE_WIDTH_NUMERICAL, 
                  label='Numerical', zorder=2, alpha=0.8)
    axes4[i].axvline(x=R_drop, color='g', linestyle=':', linewidth=1.5, 
                     label='Droplet Surface', alpha=0.7)
    
    axes4[i].set_xlabel('Radial Distance r (m)', fontsize=FONT_SIZE_LABEL)
    axes4[i].set_ylabel('Mass Fraction Y', fontsize=FONT_SIZE_LABEL)
    axes4[i].set_title(f't = {t:.4e} s, R = {R_drop:.4e} m', 
                       fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes4[i].legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    axes4[i].grid(True, alpha=0.3)
    axes4[i].tick_params(labelsize=FONT_SIZE_TICK)

fig4.suptitle('Radial Mass Fraction Profile - Spherical Droplet', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '04_Mass_Fraction_Radial_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Mass_Fraction_Radial_Comparison.eps'))
print("  Saved: 04_Mass_Fraction_Radial_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: RADIAL MASS FRACTION ERROR (2x3 GRID)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 5: RADIAL MASS FRACTION ERROR (2x3 GRID)")
print("=" * 70)

fig5, axes5 = plt.subplots(2, 3, figsize=(18, 12))
axes5 = axes5.flatten()

for i, idx in enumerate(selected_indices):
    t = times[idx]
    r_coords = all_radial_coords[idx]
    Y_num = all_mass_fraction_profiles[idx]
    Y_ana = all_analytical_profiles[idx]
    Y_error = np.abs(Y_num - Y_ana)
    
    axes5[i].plot(r_coords, Y_error, 'k-', linewidth=LINE_WIDTH_NUMERICAL)
    
    axes5[i].set_xlabel('Radial Distance r (m)', fontsize=FONT_SIZE_LABEL)
    axes5[i].set_ylabel('|Y_num - Y_analytical|', fontsize=FONT_SIZE_LABEL)
    axes5[i].set_title(f't = {t:.4e} s', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    axes5[i].grid(True, alpha=0.3)
    axes5[i].tick_params(labelsize=FONT_SIZE_TICK)

fig5.suptitle('Radial Mass Fraction Error - Spherical Droplet', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '05_Mass_Fraction_Radial_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Mass_Fraction_Radial_Error.eps'))
print("  Saved: 05_Mass_Fraction_Radial_Error.png/.eps")
plt.close()

# ============================================================================
# PLOT 6: EVAPORATION RATE VERIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 6: EVAPORATION RATE VERIFICATION")
print("=" * 70)

# Calculate numerical evaporation rate from D^2 slope
if len(times) > 1:
    # Fit linear regression to D^2 vs time
    coeffs = np.polyfit(times, D_squared, 1)
    K_numerical = -coeffs[0]  # Negative slope is K
    
    # Calculate evaporation rates
    m_dot_analytical = evaporation_rate(density1, K_analytical)
    m_dot_numerical = evaporation_rate(density1, K_numerical)
    
    fig6, ax6 = plt.subplots(figsize=(10, 8))
    
    # Plot K values
    ax6.bar(['Analytical K', 'Numerical K'], 
            [K_analytical, K_numerical], 
            color=['blue', 'red'], alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax6.set_ylabel('Evaporation Constant K (m^2/s)', fontsize=FONT_SIZE_LABEL)
    ax6.set_title('Evaporation Constant Comparison\nD^2-Law: D^2 = D0^2 - K*t', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(labelsize=FONT_SIZE_TICK)
    
    # Add value labels
    ax6.text(0, K_analytical, f'{K_analytical:.6e}', 
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK)
    ax6.text(1, K_numerical, f'{K_numerical:.6e}', 
             ha='center', va='bottom', fontsize=FONT_SIZE_TICK)
    
    # Add error percentage
    K_error_percent = abs(K_numerical - K_analytical) / K_analytical * 100
    ax6.text(0.5, max(K_analytical, K_numerical) * 0.5, 
             f'Error: {K_error_percent:.2f}%', 
             ha='center', fontsize=FONT_SIZE_LABEL, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_folder, '06_Evaporation_Constant.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '06_Evaporation_Constant.eps'))
    print("  Saved: 06_Evaporation_Constant.png/.eps")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Gas density (rho_g): {density0} kg/m^3")
print(f"  Liquid density (rho_l): {density1} kg/m^3")
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")

print(f"\nD^2-Law Results:")
print(f"  Initial diameter D0: {np.sqrt(D0_squared):.6e} m")
print(f"  Initial radius R0: {droplet_radii[0]:.6e} m")
print(f"  Analytical K: {K_analytical:.6e} m^2/s")

if len(times) > 1:
    print(f"  Numerical K: {K_numerical:.6e} m^2/s")
    print(f"  K error: {K_error_percent:.4f}%")
    print(f"  Predicted lifetime: {t_life:.6e} s")

print(f"\nFiles generated:")
print(f"  - 01_Droplet_Radius.png/.eps")
print(f"  - 02_D2_Law_Verification.png/.eps")
print(f"  - 03_D2_Error.png/.eps")
print(f"  - 04_Mass_Fraction_Radial_Comparison.png/.eps")
print(f"  - 05_Mass_Fraction_Radial_Error.png/.eps")
print(f"  - 06_Evaporation_Constant.png/.eps")

print("\n" + "=" * 70)
