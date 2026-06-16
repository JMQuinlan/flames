"""
===============================================================================
MASS FLUX COMPARISON - TEXTBOOK FORMULA VERIFICATION
===============================================================================

PURPOSE:
    Compare mass flux results from 1D planar evaporation simulation against
    classical textbook formulas for Stefan flow mass transfer.
    
FEATURES:
    - Multiple textbook formula comparisons
    - Mass flux evolution over time
    - Sherwood number analysis
    - Transfer coefficient verification
    - Error analysis and convergence plots
    - PNG and EPS output formats

ANALYTICAL FORMULAS:
    
    1. Stefan Flow Mass Flux (exact):
       m_dot = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    
    2. Film Theory Mass Flux:
       m_dot = k_c * rho * ln[(1 - Y_inf) / (1 - Y_s)]
       where k_c = D_v / delta is the mass transfer coefficient
    
    3. Sherwood Number:
       Sh = k_c * L / D_v = L / delta
    
    4. Dilute Approximation (Y_s << 1):
       m_dot_approx = (rho * D_v / delta) * (Y_s - Y_inf)
    
    5. Blowing Factor Correction:
       B_M = (Y_s - Y_inf) / (1 - Y_s)  [Spalding mass transfer number]
       m_dot = (rho * D_v / delta) * ln(1 + B_M)

INPUTS:
    - AMReX plot files from evaporation simulation
    - Physical parameters: rho, D_v, Y_s, Y_inf

OUTPUTS:
    - Mass flux comparison plots (multiple formulas)
    - Sherwood number evolution
    - Transfer coefficient verification
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
output_folder = './Mass_Flux_Analysis'

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
MARKER_SIZE = 6

# Create output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# TEXTBOOK FORMULA FUNCTIONS
# ============================================================================

def stefan_mass_flux_exact(rho, D_v, delta, Y_s, Y_inf):
    """
    Exact Stefan flow mass flux formula
    m_dot = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0 or delta < 1e-12:
        return 0.0
    return (rho * D_v / delta) * np.log((1.0 - Y_inf) / (1.0 - Y_s))

def mass_transfer_coefficient(D_v, delta):
    """
    Mass transfer coefficient: k_c = D_v / delta
    """
    if delta < 1e-12:
        return 0.0
    return D_v / delta

def sherwood_number(L, delta):
    """
    Sherwood number: Sh = L / delta
    """
    if delta < 1e-12:
        return 0.0
    return L / delta

def spalding_mass_transfer_number(Y_s, Y_inf):
    """
    Spalding mass transfer number (blowing factor)
    B_M = (Y_s - Y_inf) / (1 - Y_s)
    """
    if Y_s >= 1.0:
        return 0.0
    return (Y_s - Y_inf) / (1.0 - Y_s)

def mass_flux_spalding(rho, D_v, delta, Y_s, Y_inf):
    """
    Mass flux using Spalding number
    m_dot = (rho * D_v / delta) * ln(1 + B_M)
    """
    B_M = spalding_mass_transfer_number(Y_s, Y_inf)
    if delta < 1e-12:
        return 0.0
    return (rho * D_v / delta) * np.log(1.0 + B_M)

def mass_flux_dilute_approximation(rho, D_v, delta, Y_s, Y_inf):
    """
    Dilute approximation (linear profile, no Stefan flow)
    m_dot_approx = (rho * D_v / delta) * (Y_s - Y_inf)
    """
    if delta < 1e-12:
        return 0.0
    return (rho * D_v / delta) * (Y_s - Y_inf)

def mass_flux_film_theory(rho, k_c, Y_s, Y_inf):
    """
    Film theory mass flux
    m_dot = k_c * rho * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0:
        return 0.0
    return k_c * rho * np.log((1.0 - Y_inf) / (1.0 - Y_s))

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
    idx = np.argmin(np.abs(eta_values - 0.5))
    return x_coords[idx]

def calculate_numerical_mass_flux(ds, x_interface):
    """
    Calculate numerical mass flux at interface from simulation data
    m_dot = rho * v_x at interface
    """
    # Extract data at interface
    y_mid = 0.0
    z_mid = 0.0
    x_min = float(ds.domain_left_edge[0])
    x_max = float(ds.domain_right_edge[0])
    
    ray_start = ds.arr([x_min, y_mid, z_mid], 'code_length')
    ray_end = ds.arr([x_max, y_mid, z_mid], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
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
print("MASS FLUX COMPARISON - TEXTBOOK FORMULA VERIFICATION")
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
interface_positions = []
numerical_mass_flux = []
stefan_exact_flux = []
spalding_flux = []
dilute_flux = []
film_theory_flux = []
sherwood_numbers = []
transfer_coefficients = []
spalding_numbers = []

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
    
    sort_indices = np.argsort(ray['x'])
    x_coords = np.array(ray['x'][sort_indices])
    eta_values = np.array(ray['eta'][sort_indices])
    
    # Find interface position
    x_interface = find_interface_position_1d(x_coords, eta_values)
    delta = L_domain - x_interface
    
    # Calculate numerical mass flux
    m_dot_num = calculate_numerical_mass_flux(ds, x_interface)
    
    # Calculate analytical mass fluxes using different formulas
    m_dot_stefan = stefan_mass_flux_exact(density0, D_v, delta, Y_surface, Y_infinity)
    m_dot_spalding = mass_flux_spalding(density0, D_v, delta, Y_surface, Y_infinity)
    m_dot_dilute = mass_flux_dilute_approximation(density0, D_v, delta, Y_surface, Y_infinity)
    
    k_c = mass_transfer_coefficient(D_v, delta)
    m_dot_film = mass_flux_film_theory(density0, k_c, Y_surface, Y_infinity)
    
    # Calculate dimensionless numbers
    Sh = sherwood_number(L_domain, delta)
    B_M = spalding_mass_transfer_number(Y_surface, Y_infinity)
    
    times.append(t)
    interface_positions.append(x_interface)
    numerical_mass_flux.append(m_dot_num)
    stefan_exact_flux.append(m_dot_stefan)
    spalding_flux.append(m_dot_spalding)
    dilute_flux.append(m_dot_dilute)
    film_theory_flux.append(m_dot_film)
    sherwood_numbers.append(Sh)
    transfer_coefficients.append(k_c)
    spalding_numbers.append(B_M)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps")

times = np.array(times)
interface_positions = np.array(interface_positions)
numerical_mass_flux = np.array(numerical_mass_flux)
stefan_exact_flux = np.array(stefan_exact_flux)
spalding_flux = np.array(spalding_flux)
dilute_flux = np.array(dilute_flux)
film_theory_flux = np.array(film_theory_flux)
sherwood_numbers = np.array(sherwood_numbers)
transfer_coefficients = np.array(transfer_coefficients)
spalding_numbers = np.array(spalding_numbers)

# ============================================================================
# PLOT 1: MASS FLUX COMPARISON - ALL FORMULAS
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: MASS FLUX COMPARISON - ALL FORMULAS")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))

ax1.plot(times, stefan_exact_flux, 'b-', linewidth=LINE_WIDTH_EXACT, 
         label='Stefan Exact Formula', zorder=1)
ax1.plot(times, spalding_flux, 'g--', linewidth=LINE_WIDTH_EXACT, 
         label='Spalding B_M Formula', zorder=2, alpha=0.8)
ax1.plot(times, film_theory_flux, 'c-.', linewidth=LINE_WIDTH_EXACT, 
         label='Film Theory Formula', zorder=3, alpha=0.8)
ax1.plot(times, dilute_flux, 'm:', linewidth=LINE_WIDTH_EXACT, 
         label='Dilute Approximation', zorder=4, alpha=0.8)
ax1.plot(times, numerical_mass_flux, 'ro', markersize=MARKER_SIZE, 
         label='Numerical Simulation', alpha=0.7, zorder=5)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Mass Flux Comparison - Multiple Textbook Formulas', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '01_Mass_Flux_All_Formulas.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Mass_Flux_All_Formulas.eps'))
print("  Saved: 01_Mass_Flux_All_Formulas.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: MASS FLUX ERROR - STEFAN EXACT FORMULA
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: MASS FLUX ERROR - STEFAN EXACT")
print("=" * 70)

mass_flux_error = np.abs(numerical_mass_flux - stefan_exact_flux)
epsilon = 1e-16
mass_flux_error_safe = mass_flux_error + epsilon

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.semilogy(times, mass_flux_error_safe, 'k-', linewidth=LINE_WIDTH_NUMERICAL)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('|m_dot_num - m_dot_Stefan| (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Mass Flux Error - Stefan Exact Formula', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_Mass_Flux_Error_Stefan.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Mass_Flux_Error_Stefan.eps'))
print("  Saved: 02_Mass_Flux_Error_Stefan.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: RELATIVE ERROR COMPARISON - ALL FORMULAS
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: RELATIVE ERROR COMPARISON")
print("=" * 70)

# Calculate relative errors (avoid division by zero)
stefan_rel_error = np.abs(numerical_mass_flux - stefan_exact_flux) / (np.abs(stefan_exact_flux) + epsilon) * 100
spalding_rel_error = np.abs(numerical_mass_flux - spalding_flux) / (np.abs(spalding_flux) + epsilon) * 100
dilute_rel_error = np.abs(numerical_mass_flux - dilute_flux) / (np.abs(dilute_flux) + epsilon) * 100
film_rel_error = np.abs(numerical_mass_flux - film_theory_flux) / (np.abs(film_theory_flux) + epsilon) * 100

fig3, ax3 = plt.subplots(figsize=(12, 8))

ax3.semilogy(times, stefan_rel_error, 'b-', linewidth=LINE_WIDTH_NUMERICAL, 
             label='Stefan Exact', zorder=1)
ax3.semilogy(times, spalding_rel_error, 'g--', linewidth=LINE_WIDTH_NUMERICAL, 
             label='Spalding B_M', zorder=2, alpha=0.8)
ax3.semilogy(times, film_rel_error, 'c-.', linewidth=LINE_WIDTH_NUMERICAL, 
             label='Film Theory', zorder=3, alpha=0.8)
ax3.semilogy(times, dilute_rel_error, 'm:', linewidth=LINE_WIDTH_NUMERICAL, 
             label='Dilute Approximation', zorder=4, alpha=0.8)

ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Relative Error Comparison - All Formulas', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '03_Relative_Error_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Relative_Error_Comparison.eps'))
print("  Saved: 03_Relative_Error_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: SHERWOOD NUMBER VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: SHERWOOD NUMBER VS TIME")
print("=" * 70)

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(times, sherwood_numbers, 'b-', linewidth=LINE_WIDTH_NUMERICAL, marker='o', 
         markersize=MARKER_SIZE, label='Sh = L / delta')

ax4.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Sherwood Number (Sh)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Sherwood Number Evolution', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '04_Sherwood_Number.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Sherwood_Number.eps'))
print("  Saved: 04_Sherwood_Number.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: MASS TRANSFER COEFFICIENT VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 5: MASS TRANSFER COEFFICIENT VS TIME")
print("=" * 70)

fig5, ax5 = plt.subplots(figsize=(10, 8))
ax5.plot(times, transfer_coefficients, 'g-', linewidth=LINE_WIDTH_NUMERICAL, 
         marker='s', markersize=MARKER_SIZE, label='k_c = D_v / delta')

ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Mass Transfer Coefficient k_c (m/s)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Mass Transfer Coefficient Evolution', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '05_Transfer_Coefficient.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Transfer_Coefficient.eps'))
print("  Saved: 05_Transfer_Coefficient.png/.eps")
plt.close()

# ============================================================================
# PLOT 6: SPALDING NUMBER VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 6: SPALDING NUMBER VS TIME")
print("=" * 70)

fig6, ax6 = plt.subplots(figsize=(10, 8))
ax6.plot(times, spalding_numbers, 'r-', linewidth=LINE_WIDTH_NUMERICAL, 
         marker='^', markersize=MARKER_SIZE, label='B_M = (Y_s - Y_inf) / (1 - Y_s)')

ax6.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax6.set_ylabel('Spalding Number B_M', fontsize=FONT_SIZE_LABEL)
ax6.set_title('Spalding Mass Transfer Number Evolution', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax6.grid(True, alpha=0.3)
ax6.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '06_Spalding_Number.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '06_Spalding_Number.eps'))
print("  Saved: 06_Spalding_Number.png/.eps")
plt.close()

# ============================================================================
# PLOT 7: FORMULA COMPARISON BAR CHART (AVERAGE ERROR)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 7: AVERAGE ERROR BAR CHART")
print("=" * 70)

avg_errors = {
    'Stefan Exact': np.mean(stefan_rel_error),
    'Spalding B_M': np.mean(spalding_rel_error),
    'Film Theory': np.mean(film_rel_error),
    'Dilute Approx': np.mean(dilute_rel_error)
}

fig7, ax7 = plt.subplots(figsize=(10, 8))
formulas = list(avg_errors.keys())
errors = list(avg_errors.values())
colors = ['blue', 'green', 'cyan', 'magenta']

bars = ax7.bar(formulas, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

ax7.set_ylabel('Average Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax7.set_title('Average Relative Error - Formula Comparison', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')
ax7.tick_params(labelsize=FONT_SIZE_TICK)

# Add value labels on bars
for bar, error in zip(bars, errors):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'{error:.2f}%', ha='center', va='bottom', fontsize=FONT_SIZE_TICK)

plt.tight_layout()

plt.savefig(os.path.join(output_folder, '07_Average_Error_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '07_Average_Error_Comparison.eps'))
print("  Saved: 07_Average_Error_Comparison.png/.eps")
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
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")
print(f"  Spalding number (B_M): {spalding_numbers[0]:.6f}")

print(f"\nMass Flux Results:")
print(f"  Average numerical mass flux: {np.mean(numerical_mass_flux):.6e} kg/m^2/s")
print(f"  Average Stefan exact flux: {np.mean(stefan_exact_flux):.6e} kg/m^2/s")
print(f"  Average Spalding flux: {np.mean(spalding_flux):.6e} kg/m^2/s")
print(f"  Average dilute flux: {np.mean(dilute_flux):.6e} kg/m^2/s")

print(f"\nError Analysis:")
print(f"  Stefan exact - Avg relative error: {np.mean(stefan_rel_error):.4f}%")
print(f"  Spalding B_M - Avg relative error: {np.mean(spalding_rel_error):.4f}%")
print(f"  Film theory - Avg relative error: {np.mean(film_rel_error):.4f}%")
print(f"  Dilute approx - Avg relative error: {np.mean(dilute_rel_error):.4f}%")

print(f"\nDimensionless Numbers:")
print(f"  Average Sherwood number: {np.mean(sherwood_numbers):.4f}")
print(f"  Average transfer coefficient: {np.mean(transfer_coefficients):.6e} m/s")

print(f"\nFiles generated:")
print(f"  - 01_Mass_Flux_All_Formulas.png/.eps")
print(f"  - 02_Mass_Flux_Error_Stefan.png/.eps")
print(f"  - 03_Relative_Error_Comparison.png/.eps")
print(f"  - 04_Sherwood_Number.png/.eps")
print(f"  - 05_Transfer_Coefficient.png/.eps")
print(f"  - 06_Spalding_Number.png/.eps")
print(f"  - 07_Average_Error_Comparison.png/.eps")

print("\n" + "=" * 70)
