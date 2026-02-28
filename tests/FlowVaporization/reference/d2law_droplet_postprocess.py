"""
===============================================================================
SPHERICAL DROPLET D² LAW EVAPORATION - ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Analyze spherical water droplet evaporation in air.
    Compare numerical solver results against analytical D² law.
    
FEATURES:
    - D² vs time plot (classic D² law verification)
    - Diameter regression rate vs time plot
    - PNG output format

ANALYTICAL SOLUTION (CLASSICAL D² LAW):
    Spherical droplet evaporation (quasi-steady):
    
    d²(t) = d₀² - K·t
    
    Evaporation constant:
    K = (8·ρ_g·D_v/ρ_l)·ln(1 + B_M)
    
    Spalding number:
    B_M = (Y_vs - Y_inf) / (1 - Y_vs)
    
    Regression rate:
    dd/dt = -K/(2d)

VALIDITY:
    - Quasi-steady assumption
    - Constant droplet temperature
    - Spherically symmetric evaporation

INPUTS:
    - AMReX plot files from 2D axisymmetric simulation
    - Physical parameters: rho_gas, rho_liq, D_v, T_s, T_inf, p_ambient

OUTPUTS:
    - water_d2_law_numerical.png (d² vs time)
    - water_regression_rate_numerical.png (dd/dt vs time)

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
amrex_output_dir = r'../../../Output/d2law_water_droplet'
output_folder = './D2Law_Droplet_PostProcess'

# Physical parameters - MUST MATCH YOUR INPUT FILE
# Air properties (gas phase, eta=1)
density0 = 1.184                 # Air density [kg/m³]
mu0 = 1.8e-5                     # Air dynamic viscosity [Pa·s]
T_inf = 400.0                    # Ambient temperature [K]

# Water properties (liquid phase, eta=0)
density1 = 997.0                 # Water density [kg/m³]
mu1 = 1.0e-3                     # Water dynamic viscosity [Pa·s]
T_s = 373.15                     # Droplet surface temperature [K]

# Mass transfer properties
D_v = 2.6e-5                     # Water vapor diffusion coefficient [m²/s]
p_ambient = 101325.0             # Ambient pressure [Pa]
Y_infinity = 0.0                 # Far-field vapor mass fraction (dry air)

# Water vapor properties
W_v = 18.015e-3                  # Molecular weight of water [kg/mol]
W_g = 28.97e-3                   # Molecular weight of air [kg/mol]

# Antoine coefficients for water (log10(p[Pa]) = A - B/(C + T[K]))
antoine_A = 5.40221
antoine_B = 1838.675
antoine_C = -31.737

# Initial droplet diameter (from input file)
d0_initial = 100.0e-6            # 100 μm [m]

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
# ANALYTICAL D² LAW FUNCTIONS
# ============================================================================

def saturation_pressure(T):
    """
    Antoine equation for water vapor saturation pressure.
    
    Parameters:
    -----------
    T : float
        Temperature in K
        
    Returns:
    --------
    p_sat : float
        Saturation pressure in Pa
    """
    log10_p = antoine_A - antoine_B / (antoine_C + T)
    return 10**log10_p

def vapor_mass_fraction_surface(T_s, p_ambient):
    """
    Surface vapor mass fraction.
    
    Y_vs = (W_v * x_vs) / (W_v * x_vs + W_g * (1 - x_vs))
    where x_vs = p_sat(T_s) / p
    """
    p_sat = saturation_pressure(T_s)
    x_vs = min(p_sat / p_ambient, 1.0)
    
    numerator = W_v * x_vs
    denominator = W_v * x_vs + W_g * (1.0 - x_vs)
    Y_vs = numerator / denominator
    
    return Y_vs

def spalding_number(T_s, p_ambient, Y_infinity=0.0):
    """
    Spalding mass transfer number.
    
    B_M = (Y_vs - Y_inf) / (1 - Y_vs)
    """
    Y_vs = vapor_mass_fraction_surface(T_s, p_ambient)
    
    denominator = 1.0 - Y_vs
    if denominator < 1e-10:
        denominator = 1e-10
    
    B_M = (Y_vs - Y_infinity) / denominator
    
    return B_M

def evaporation_constant(T_s, p_ambient, rho_g, rho_l, D_v, Y_infinity=0.0):
    """
    Classic D² law evaporation constant.
    
    K = (8 * ρ_g * D_v / ρ_l) * ln(1 + B_M)
    
    Returns:
    --------
    K : float
        Evaporation constant in m²/s
    """
    B_M = spalding_number(T_s, p_ambient, Y_infinity)
    K = (8.0 * rho_g * D_v / rho_l) * np.log(1.0 + B_M)
    return K

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def calculate_droplet_volume_2d(eta_field, dx, dy):
    """
    Calculate droplet volume from 2D axisymmetric data.
    
    For axisymmetric geometry: V = 2π ∫∫ r * (1-eta) * dr * dz
    where r is radial coordinate (x in 2D), z is axial coordinate (y in 2D)
    
    Parameters:
    -----------
    eta_field : array
        Volume fraction field (eta=0 is liquid, eta=1 is gas)
    dx, dy : float
        Grid spacing
        
    Returns:
    --------
    volume : float
        Droplet volume in m³
    """
    # eta = 0 inside droplet (liquid), eta = 1 outside (gas)
    # Liquid volume fraction = (1 - eta)
    liquid_fraction = 1.0 - eta_field
    
    # For 2D axisymmetric: integrate with cylindrical volume element
    # This is a simplified calculation assuming uniform grid
    volume = np.sum(liquid_fraction) * dx * dy
    
    return volume

def calculate_droplet_diameter_from_eta(ds):
    """
    Calculate droplet diameter from eta field in 2D axisymmetric simulation.
    
    Method: Find the radius where eta = 0.5 (interface location)
    
    Parameters:
    -----------
    ds : yt dataset
        AMReX dataset
        
    Returns:
    --------
    diameter : float
        Droplet diameter in meters
    """
    # Create a covering grid to get uniform data
    level = 0
    dims = ds.domain_dimensions
    
    # Get the data
    ad = ds.all_data()
    
    # Get coordinates and eta values
    x_coords = np.array(ad['x'])
    y_coords = np.array(ad['y'])
    eta_values = np.array(ad['eta'])
    
    # Calculate radial distance from center (0, 0)
    r_coords = np.sqrt(x_coords**2 + y_coords**2)
    
    # Find interface location (where eta ≈ 0.5)
    # Method: find points near eta = 0.5
    interface_mask = np.abs(eta_values - 0.5) < 0.1
    
    if np.sum(interface_mask) > 0:
        # Average radius at interface
        r_interface = np.mean(r_coords[interface_mask])
        diameter = 2.0 * r_interface
    else:
        # Fallback: estimate from liquid volume
        liquid_volume = np.sum(1.0 - eta_values) * (ds.domain_width[0] / dims[0]) * (ds.domain_width[1] / dims[1])
        # V = (4/3)πr³ for sphere
        radius = (3.0 * liquid_volume / (4.0 * np.pi))**(1.0/3.0)
        diameter = 2.0 * radius
    
    return diameter

# ============================================================================
# CALCULATE ANALYTICAL D² LAW PARAMETERS
# ============================================================================

print("=" * 70)
print("SPHERICAL DROPLET D² LAW EVAPORATION - ANALYSIS")
print("=" * 70)

print("\nCalculating analytical D² law parameters...")

# Calculate Spalding number
B_M = spalding_number(T_s, p_ambient, Y_infinity)
print(f"  Spalding number: B_M = {B_M:.6f}")

# Calculate evaporation constant
K_analytical = evaporation_constant(T_s, p_ambient, density0, density1, D_v, Y_infinity)
print(f"  Evaporation constant: K = {K_analytical*1e9:.6f} × 10⁻⁹ m²/s")

# Calculate droplet lifetime
t_life_analytical = d0_initial**2 / K_analytical
print(f"  Droplet lifetime: t_life = {t_life_analytical:.4f} s")

# Surface vapor mass fraction
Y_vs = vapor_mass_fraction_surface(T_s, p_ambient)
print(f"  Surface vapor mass fraction: Y_vs = {Y_vs:.6f}")

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
diameters = []

for i, plot_file in enumerate(plot_files):
    ds = yt.load(plot_file)
    t = float(ds.current_time)
    
    # Calculate droplet diameter
    d = calculate_droplet_diameter_from_eta(ds)
    
    times.append(t)
    diameters.append(d)
    
    if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
        print(f"  Processed {i + 1}/{len(plot_files)} timesteps: t = {t:.6e} s, d = {d*1e6:.2f} μm")

times = np.array(times)
diameters = np.array(diameters)

# Calculate d²
d_squared = diameters**2

# Calculate regression rate: dd/dt
if len(times) > 1:
    dd_dt = np.gradient(diameters, times)
else:
    dd_dt = np.zeros_like(diameters)

print(f"\nData extraction complete!")
print(f"  Initial diameter: d₀ = {diameters[0]*1e6:.2f} μm")
print(f"  Final diameter: d_f = {diameters[-1]*1e6:.2f} μm")
print(f"  Total evaporation time: {times[-1]:.4f} s")

# ============================================================================
# PLOT 1: D² LAW (d² vs time)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: D² LAW (d² vs time)")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 7))

# Plot numerical data
ax1.plot(times, d_squared*1e12, 'bo', markersize=6, label='Numerical', alpha=0.7, zorder=2)

# Linear fit to verify linearity
if len(times) > 1:
    coeffs = np.polyfit(times, d_squared*1e12, 1)
    K_fitted = -coeffs[0] * 1e-12
    ax1.plot(times, np.polyval(coeffs, times), 'r--', linewidth=LINE_WIDTH_EXACT, alpha=0.7,
            label=f'Linear fit: K = {K_fitted*1e9:.6f}×10⁻⁹ m²/s')
    
    # Plot analytical solution
    d_squared_analytical = (diameters[0]**2 - K_analytical * times) * 1e12
    ax1.plot(times, d_squared_analytical, 'g-', linewidth=LINE_WIDTH_EXACT, 
            label=f'Analytical: K = {K_analytical*1e9:.6f}×10⁻⁹ m²/s', zorder=1)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax1.set_ylabel('d² (μm²)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax1.set_title('Classic D² Law: Water Droplet Evaporation in Air\n(Numerical Simulation)', 
            fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.4, linestyle='--')
ax1.tick_params(labelsize=FONT_SIZE_TICK)

# Add info text box
if len(times) > 1:
    R_squared = np.corrcoef(times, d_squared)[0, 1]**2
    error_percent = abs(K_fitted / K_analytical - 1.0) * 100
    
    textstr = f'NUMERICAL RESULTS\n\n'
    textstr += f'd₀ = {diameters[0]*1e6:.1f} μm\n'
    textstr += f'T_s = {T_s:.1f} K\n'
    textstr += f'T∞ = {T_inf:.1f} K\n\n'
    textstr += f'K_analytical = {K_analytical*1e9:.6f} × 10⁻⁹ m²/s\n'
    textstr += f'K_fitted = {K_fitted*1e9:.6f} × 10⁻⁹ m²/s\n'
    textstr += f'B_M = {B_M:.6f}\n\n'
    textstr += f'R² = {R_squared:.6f}\n'
    textstr += f'K error = {error_percent:.2f}%'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
    ax1.text(0.98, 0.97, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'water_d2_law_numerical.png'), dpi=300, bbox_inches='tight')
print("  Saved: water_d2_law_numerical.png")
plt.show()

# ============================================================================
# PLOT 2: DIAMETER REGRESSION RATE (dd/dt vs time)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: DIAMETER REGRESSION RATE (dd/dt vs time)")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(10, 7))

# Plot numerical regression rate (absolute value)
ax2.plot(times, -dd_dt*1e6, 'ro', markersize=6, label='Numerical: |dd/dt|', alpha=0.7, zorder=2)

# Plot analytical regression rate: |dd/dt| = K/(2d)
if len(times) > 1:
    dd_dt_analytical = -K_analytical / (2.0 * diameters)
    ax2.plot(times, -dd_dt_analytical*1e6, 'b-', linewidth=LINE_WIDTH_EXACT, 
            label='Analytical: |dd/dt| = K/(2d)', zorder=1)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax2.set_ylabel('Regression Rate (μm/s)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax2.set_title('Diameter Regression Rate: Water Droplet in Air\n(Numerical Simulation)', 
            fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.4, linestyle='--')
ax2.tick_params(labelsize=FONT_SIZE_TICK)

# Add info text box
if len(times) > 1:
    textstr = f'REGRESSION RATE\n\n'
    textstr += f'd₀ = {diameters[0]*1e6:.1f} μm\n'
    textstr += f'K = {K_analytical*1e9:.6f} × 10⁻⁹ m²/s\n\n'
    textstr += f'Initial rate:\n'
    textstr += f'  |dd/dt|₀ = {-dd_dt_analytical[0]*1e6:.4f} μm/s\n\n'
    textstr += f'Final rate:\n'
    textstr += f'  |dd/dt|_f = {-dd_dt_analytical[-1]*1e6:.4f} μm/s\n\n'
    textstr += f'Rate increases as\ndroplet shrinks'
    
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.9)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=props, family='monospace')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'water_regression_rate_numerical.png'), dpi=300, bbox_inches='tight')
print("  Saved: water_regression_rate_numerical.png")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Air density (rho_g): {density0} kg/m³")
print(f"  Water density (rho_l): {density1} kg/m³")
print(f"  Diffusion coefficient (D_v): {D_v} m²/s")
print(f"  Surface temperature (T_s): {T_s} K")
print(f"  Ambient temperature (T_inf): {T_inf} K")
print(f"  Ambient pressure: {p_ambient} Pa")

print(f"\nAnalytical D² Law Parameters:")
print(f"  Spalding number (B_M): {B_M:.6f}")
print(f"  Evaporation constant (K): {K_analytical*1e9:.6f} × 10⁻⁹ m²/s")
print(f"  Droplet lifetime: {t_life_analytical:.4f} s")

if len(times) > 1:
    print(f"\nNumerical Results:")
    print(f"  Initial diameter: {diameters[0]*1e6:.2f} μm")
    print(f"  Final diameter: {diameters[-1]*1e6:.2f} μm")
    print(f"  Simulation time: {times[-1]:.4f} s")
    print(f"  Fitted K: {K_fitted*1e9:.6f} × 10⁻⁹ m²/s")
    print(f"  K error: {error_percent:.2f}%")
    print(f"  R² (linearity): {R_squared:.6f}")

print(f"\nFiles generated:")
print(f"  - water_d2_law_numerical.png")
print(f"  - water_regression_rate_numerical.png")

print("\n" + "=" * 70)
