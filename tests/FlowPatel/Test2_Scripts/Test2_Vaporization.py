# -*- coding: utf-8 -*-
"""
===============================================================================
CYLINDRICAL DROPLET VAPORIZATION ANALYSIS SCRIPT
===============================================================================
PURPOSE:
    Analyze droplet vaporization from AMReX simulation data by tracking 
    diameter evolution, mass transfer rates, and comparing against analytical
    d-squared law predictions.

FEATURES:
    - Extracts droplet diameter from eta field over time
    - Performs linear regression on D^2 vs t to extract vaporization constant K
    - Compares numerical K against two analytical models (thermal and mass transfer)
    - Calculates and plots Spalding number evolution
    - Tracks mass transfer rate (mdot) and cumulative mass transfer
    - Visualizes interfacial area evolution
    - Plots temperature fields at start and end of simulation

INPUTS:
    - AMReX plot files from vaporization simulation
    - Physical parameters matching simulation input file

OUTPUTS:
    - Diameter evolution plot
    - D-squared vs time with linear fit
    - K comparison (numerical vs analytical)
    - Spalding number evolution
    - Mass transfer rate evolution
    - Cumulative mass transfer
    - Interfacial area evolution
    - Temperature field visualizations

ANALYTICAL SOLUTIONS:
    D-squared law: D^2 = D0^2 - K * t
    
    Option A - Thermal Spalding:
    K_A = (8 * lambda_gas) / (rho_liquid * cp_gas) * ln(1 + B_T)
    B_T = cp_gas * (T_inf - T_s) / L_v
    
    Option B - Mass Transfer Spalding:
    K_B = (8 * lambda_gas) / (rho_liquid * cp_gas) * ln(1 + B_M) / Le
    B_M = (Y_F,s - Y_F,inf) / (1 - Y_F,s)
    
    where:
    - lambda_gas = thermal conductivity of gas ,[W/m-K]
    - rho_liquid = liquid density ,[kg/m^3]
    - cp_gas = gas specific heat ,[J/kg-K]
    - T_inf = ambient temperature ,[K]
    - T_s = surface/saturation temperature ,[K]
    - L_v = latent heat of vaporization ,[J/kg]
    - Le = Lewis number (thermal/mass diffusivity ratio)
    - Y_F,s = fuel mass fraction at surface
    - Y_F,inf = fuel mass fraction at infinity

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.ndimage import sobel
import os
import re

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (MUST MATCH YOUR SIMULATION INPUT FILE)
# Liquid properties (water droplet)
rho_liquid = 1599.9314      # Liquid density at 373 K ,[kg/m^3]
T_s = 373.0                 # Saturation temperature ,[K]
cp_liquid = 4179.0          # Liquid specific heat ,[J/kg-K]

# Gas properties (air)
rho_gas = 0.5804            # Gas density at 600 K ,[kg/m^3]
T_inf = 600.0               # Ambient temperature ,[K]
cp_gas = 1005.0             # Gas specific heat ,[J/kg-K]
mu_gas = 1.8e-5             # Gas dynamic viscosity ,[Pa*s]

# Transport properties
lambda_gas = 0.0436         # Thermal conductivity of air at 600 K ,[W/m-K]
D_v = 2.3e-5                # Vapor diffusivity ,[m^2/s]
alpha_gas = lambda_gas / (rho_gas * cp_gas)  # Thermal diffusivity ,[m^2/s]
Le = alpha_gas / D_v        # Lewis number

# Vaporization properties
L_v = 2.257e6               # Latent heat of vaporization for water ,[J/kg]

# Initial conditions
R0 = 0.001                  # Initial droplet radius ,[m] = 1 mm
D0 = 2.0 * R0               # Initial diameter ,[m]

# Droplet center location
droplet_center_x = 0.0      # X-coordinate of droplet center ,[m]
droplet_center_y = 0.0      # Y-coordinate of droplet center ,[m]

# Eta contour value for interface tracking
eta_contour = 0.5           # Interface location (0.5 = midpoint)

# Spalding number calculation parameters
Y_F_s = 1.0                 # Fuel mass fraction at surface (pure vapor)
Y_F_inf = 0.0               # Fuel mass fraction at infinity (pure air)

# Interfacial area threshold
grad_eta_threshold = 1e-4   # Threshold for grad_eta_mag

# File paths
amrex_output_dir = r'../../../bin/tests/FlowPatel/Test2_Vaporization'  # Directory containing AMReX plot files

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH_ANALYTICAL = 2.5
LINE_WIDTH_NUMERICAL = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './Vaporization_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL VAPORIZATION CONSTANTS
# ============================================================================

def calculate_spalding_thermal(cp_gas, T_inf, T_s, L_v):
    """
    Calculate thermal Spalding number B_T.
    
    B_T = cp_gas * (T_inf - T_s) / L_v
    """
    B_T = cp_gas * (T_inf - T_s) / L_v
    return B_T

def calculate_spalding_mass(Y_F_s, Y_F_inf):
    """
    Calculate mass transfer Spalding number B_M.
    
    B_M = (Y_F,s - Y_F,inf) / (1 - Y_F,s)
    """
    B_M = (Y_F_s - Y_F_inf) / (1.0 - Y_F_s)
    return B_M

def calculate_K_thermal(lambda_gas, rho_liquid, cp_gas, B_T):
    """
    Calculate vaporization constant K using thermal Spalding number.
    
    K = (8 * lambda_gas) / (rho_liquid * cp_gas) * ln(1 + B_T)
    """
    K = (8.0 * lambda_gas) / (rho_liquid * cp_gas) * np.log(1.0 + B_T)
    return K

def calculate_K_mass(lambda_gas, rho_liquid, cp_gas, B_M, Le):
    """
    Calculate vaporization constant K using mass transfer Spalding number.
    
    K = (8 * lambda_gas) / (rho_liquid * cp_gas) * ln(1 + B_M) / Le
    """
    K = (8.0 * lambda_gas) / (rho_liquid * cp_gas) * np.log(1.0 + B_M) / Le
    return K

print("=" * 70)
print("CYLINDRICAL DROPLET VAPORIZATION ANALYSIS")
print("=" * 70)

# Calculate analytical Spalding numbers
B_T_analytical = calculate_spalding_thermal(cp_gas, T_inf, T_s, L_v)
B_M_analytical = calculate_spalding_mass(Y_F_s, Y_F_inf)

print(f"\nAnalytical Spalding Numbers:")
print(f"  B_T (thermal):       {B_T_analytical:.6f}")
print(f"  B_M (mass transfer): {B_M_analytical:.6f}")

# Calculate analytical K values
K_analytical_thermal = calculate_K_thermal(lambda_gas, rho_liquid, cp_gas, B_T_analytical)
K_analytical_mass = calculate_K_mass(lambda_gas, rho_liquid, cp_gas, B_M_analytical, Le)

print(f"\nAnalytical Vaporization Constants:")
print(f"  K_A (thermal):       {K_analytical_thermal:.6e} m^2/s")
print(f"  K_B (mass transfer): {K_analytical_mass:.6e} m^2/s")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def extract_droplet_diameter_from_eta(ds, center_x, center_y, eta_value=0.5):
    """
    Extract droplet diameter by finding the eta=0.5 contour.
    Returns average diameter from radial measurements.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge,[0])
    x_max = float(ds.domain_right_edge,[0])
    y_min = float(ds.domain_left_edge,[1])
    y_max = float(ds.domain_right_edge,[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 512  # High resolution for accurate contour detection
    
    # Create fixed resolution buffer
    width_x = x_max - x_min
    width_y = y_max - y_min
    frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
    
    # Extract eta field
    eta_field = np.array(frb,['eta'])
    
    # Create coordinate arrays
    x_1d = np.linspace(x_min, x_max, eta_field.shape,[1])
    y_1d = np.linspace(y_min, y_max, eta_field.shape,[0])
    X_grid, Y_grid = np.meshgrid(x_1d, y_1d)
    
    # Calculate radial distance from droplet center
    R_grid = np.sqrt((X_grid - center_x)**2 + (Y_grid - center_y)**2)
    
    # Find points near eta = 0.5 contour
    eta_tolerance = 0.05
    contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) == 0:
        # If no points found, try larger tolerance
        eta_tolerance = 0.1
        contour_mask = np.abs(eta_field - eta_value) < eta_tolerance
    
    if np.sum(contour_mask) > 0:
        # Extract radii at contour points
        contour_radii = R_grid,[contour_mask]
        
        # Use mean radius as droplet radius
        radius = np.mean(contour_radii)
        diameter = 2.0 * radius
        
        return diameter
    else:
        return None

def calculate_grad_eta_magnitude(ds):
    """
    Calculate magnitude of eta gradient using existing grad_eta field.
    If grad_eta doesn't exist, compute it numerically.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge,[0])
    x_max = float(ds.domain_right_edge,[0])
    y_min = float(ds.domain_left_edge,[1])
    y_max = float(ds.domain_right_edge,[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 512
    
    # Create fixed resolution buffer
    width_x = x_max - x_min
    width_y = y_max - y_min
    frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
    
    # Try to extract grad_eta field if it exists
    try:
        # Check if grad_eta components exist
        grad_eta_x = np.array(frb,['grad_eta_x'])
        grad_eta_y = np.array(frb,['grad_eta_y'])
        grad_eta_mag = np.sqrt(grad_eta_x**2 + grad_eta_y**2)
    except:
        # If grad_eta doesn't exist, compute it from eta
        eta_field = np.array(frb,['eta'])
        
        # Calculate grid spacing
        dx = (x_max - x_min) / resolution
        dy = (y_max - y_min) / resolution
        
        # Compute gradients using Sobel operator
        grad_eta_x = sobel(eta_field, axis=1) / (2.0 * dx)
        grad_eta_y = sobel(eta_field, axis=0) / (2.0 * dy)
        grad_eta_mag = np.sqrt(grad_eta_x**2 + grad_eta_y**2)
    
    return grad_eta_mag

def calculate_interfacial_area(ds, threshold=1e-4):
    """
    Calculate interfacial area as sum of grad_eta_mag * cell_area
    where grad_eta_mag > threshold.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge,[0])
    x_max = float(ds.domain_right_edge,[0])
    y_min = float(ds.domain_left_edge,[1])
    y_max = float(ds.domain_right_edge,[1])
    
    # Get resolution
    resolution = 512
    
    # Calculate cell dimensions
    dx = (x_max - x_min) / resolution
    dy = (y_max - y_min) / resolution
    cell_area = np.sqrt(dx**2 + dy**2)
    
    # Get grad_eta magnitude
    grad_eta_mag = calculate_grad_eta_magnitude(ds)
    
    # Apply threshold
    interface_mask = grad_eta_mag > threshold
    
    # Calculate interfacial area
    interfacial_area = np.sum(grad_eta_mag,[interface_mask]) * cell_area
    
    return interfacial_area

def extract_field_sum(ds, field_name):
    """
    Extract sum of a field over all cells.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge,[0])
    x_max = float(ds.domain_right_edge,[0])
    y_min = float(ds.domain_left_edge,[1])
    y_max = float(ds.domain_right_edge,[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 512
    
    # Create fixed resolution buffer
    width_x = x_max - x_min
    width_y = y_max - y_min
    frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
    
    # Extract field
    try:
        field_data = np.array(frb,[field_name])
        field_sum = np.sum(field_data)
        return field_sum
    except:
        print(f"  WARNING: Field '{field_name}' not found")
        return None

def extract_temperature_field(ds):
    """
    Extract temperature field for visualization.
    """
    # Get domain bounds
    x_min = float(ds.domain_left_edge,[0])
    x_max = float(ds.domain_right_edge,[0])
    y_min = float(ds.domain_left_edge,[1])
    y_max = float(ds.domain_right_edge,[1])
    
    # Create 2D slice at z=0
    slc = ds.slice('z', 0.0)
    
    # Get resolution for fixed resolution buffer
    resolution = 512
    
    # Create fixed resolution buffer
    width_x = x_max - x_min
    width_y = y_max - y_min
    frb = slc.to_frb((max(width_x, width_y), 'code_length'), resolution)
    
    # Extract temperature field
    try:
        temp_field = np.array(frb,['Temperature'])
        x_1d = np.linspace(x_min, x_max, temp_field.shape,[1])
        y_1d = np.linspace(y_min, y_max, temp_field.shape,[0])
        return temp_field, x_1d, y_1d
    except:
        print(f"  WARNING: Temperature field not found")
        return None, None, None

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("\n" + "=" * 70)
print("LOADING SIMULATION DATA")
print("=" * 70)

plot_files = ,[]
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)
print(f"\nFound {len(plot_files)} plot files")

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING SIMULATION DATA")
print("=" * 70)

times = ,[]
diameters = ,[]
vap_rho_sums = ,[]
rho_eta_sums = ,[]
interfacial_areas = ,[]

# Store first and last temperature fields
temp_initial = None
temp_final = None
x_coords = None
y_coords = None

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Extract diameter
        diameter = extract_droplet_diameter_from_eta(ds, droplet_center_x, droplet_center_y, eta_contour)
        
        # Extract Vap_rho0 sum (for mdot calculation)
        vap_rho_sum = extract_field_sum(ds, 'Vap_rho0')
        
        # Extract rho_eta0 sum (for Spalding number)
        rho_eta_sum = extract_field_sum(ds, 'rho_eta0')
        
        # Calculate interfacial area
        interfacial_area = calculate_interfacial_area(ds, grad_eta_threshold)
        
        # Store data
        if diameter is not None:
            times.append(t)
            diameters.append(diameter)
            vap_rho_sums.append(vap_rho_sum if vap_rho_sum is not None else 0.0)
            rho_eta_sums.append(rho_eta_sum if rho_eta_sum is not None else 0.0)
            interfacial_areas.append(interfacial_area)
        
        # Extract temperature fields (first and last)
        if i == 0:
            temp_initial, x_coords, y_coords = extract_temperature_field(ds)
        if i == len(plot_files) - 1:
            temp_final, _, _ = extract_temperature_field(ds)
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times = np.array(times)
diameters = np.array(diameters)
vap_rho_sums = np.array(vap_rho_sums)
rho_eta_sums = np.array(rho_eta_sums)
interfacial_areas = np.array(interfacial_areas)

print(f"\nSuccessfully extracted {len(times)} measurements")
print(f"  Time range: ,[{times,[0]:.6e}, {times,[-1]:.6e}] s")
print(f"  Diameter range: ,[{np.min(diameters)*1000:.4f}, {np.max(diameters)*1000:.4f}] mm")

# ============================================================================
# CALCULATE DERIVED QUANTITIES
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING DERIVED QUANTITIES")
print("=" * 70)

# Calculate D^2
D_squared = diameters**2

# Calculate mdot (time derivative of Vap_rho0 sum)
mdot = np.zeros(len(times))
if len(times) > 1:
    # Use central differences for interior points
    for i in range(1, len(times) - 1):
        dt_forward = times,[i + 1] - times,[i]
        dt_backward = times,[i] - times,[i - 1]
        mdot,[i] = (vap_rho_sums,[i + 1] - vap_rho_sums,[i - 1]) / (dt_forward + dt_backward)
    
    # Use forward difference for first point
    mdot,[0] = (vap_rho_sums,[object Object], - vap_rho_sums,[0]) / (times,[object Object], - times,[0])
    
    # Use backward difference for last point
    mdot,[-1] = (vap_rho_sums,[-1] - vap_rho_sums,[-2]) / (times,[-1] - times,[-2])

# Calculate Spalding number from rho_eta0
# B_M = (Y_F,s - Y_F,inf) / (1 - Y_F,s)
# Approximate using rho_eta0 as a proxy for vapor mass fraction
# This is a simplified calculation - adjust based on your specific field definitions
B_M_numerical = rho_eta_sums / (rho_liquid + 1e-10)  # Avoid division by zero

print(f"  Calculated mdot: range ,[{np.min(mdot):.6e}, {np.max(mdot):.6e}] kg/s")
print(f"  Total mass transfer: {vap_rho_sums,[-1]:.6e} kg")
print(f"  Spalding number range: ,[{np.min(B_M_numerical):.6f}, {np.max(B_M_numerical):.6f}]")

# ============================================================================
# LINEAR REGRESSION ON D^2 vs t TO EXTRACT K
# ============================================================================

print("\n" + "=" * 70)
print("PERFORMING LINEAR REGRESSION ON D^2 vs t")
print("=" * 70)

# Perform linear regression: D^2 = D0^2 - K*t
# y = mx + b where y = D^2, x = t, m = -K, b = D0^2
slope, intercept, r_value, p_value, std_err = linregress(times, D_squared)

K_numerical = -slope  # K is negative of slope
D0_squared_fit = intercept

print(f"\nLinear Regression Results:")
print(f"  Slope (m):        {slope:.6e} m^2/s")
print(f"  K_numerical:      {K_numerical:.6e} m^2/s")
print(f"  Intercept (D0^2): {intercept:.6e} m^2")
print(f"  D0_fit:           {np.sqrt(intercept)*1000:.4f} mm")
print(f"  R^2:              {r_value**2:.6f}")
print(f"  Std error:        {std_err:.6e}")

# Calculate errors
K_error_thermal_abs = np.abs(K_numerical - K_analytical_thermal)
K_error_thermal_rel = (K_error_thermal_abs / K_analytical_thermal) * 100

K_error_mass_abs = np.abs(K_numerical - K_analytical_mass)
K_error_mass_rel = (K_error_mass_abs / K_analytical_mass) * 100

print(f"\nComparison with Analytical K (Thermal):")
print(f"  K_analytical (thermal): {K_analytical_thermal:.6e} m^2/s")
print(f"  K_numerical:            {K_numerical:.6e} m^2/s")
print(f"  Absolute error:         {K_error_thermal_abs:.6e} m^2/s")
print(f"  Relative error:         {K_error_thermal_rel:.2f}%")

print(f"\nComparison with Analytical K (Mass Transfer):")
print(f"  K_analytical (mass):    {K_analytical_mass:.6e} m^2/s")
print(f"  K_numerical:            {K_numerical:.6e} m^2/s")
print(f"  Absolute error:         {K_error_mass_abs:.6e} m^2/s")
print(f"  Relative error:         {K_error_mass_rel:.2f}%")

# ============================================================================
# PLOT 1: DIAMETER EVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 1: DIAMETER EVOLUTION")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Plot numerical diameter
ax1.plot(times, diameters * 1000, 
         'bo-', linewidth=LINE_WIDTH_NUMERICAL, markersize=MARKER_SIZE-2,
         label='Numerical', alpha=0.8)

# Add initial diameter line
ax1.axhline(y=D0*1000, color='gray', linestyle='--', 
            linewidth=1.5, alpha=0.5, label=f'D0 = {D0*1000:.2f} mm')

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Diameter (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title(f'Droplet Diameter Evolution\n' + 
              f'T_inf={T_inf} K, T_s={T_s} K, D0={D0*1000:.2f} mm',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Diameter_Evolution.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Diameter_Evolution.eps'))
print("  Saved: 01_Diameter_Evolution.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: D-SQUARED EVOLUTION WITH LINEAR FIT
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 2: D-SQUARED EVOLUTION")
print("=" * 70)

fig2, ax2 = plt.subplots(figsize=(12, 8))

# Plot numerical D^2
ax2.plot(times, D_squared * 1e6, 
         'bo', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)

# Plot linear fit
D_squared_fit = intercept + slope * times
ax2.plot(times, D_squared_fit * 1e6, 
         'r-', linewidth=LINE_WIDTH_ANALYTICAL, 
         label=f'Linear Fit: D^2 = {intercept*1e6:.4f} - {K_numerical*1e6:.4f}*t (mm^2)')

# Plot analytical predictions
D_squared_analytical_thermal = D0**2 - K_analytical_thermal * times
D_squared_analytical_mass = D0**2 - K_analytical_mass * times

ax2.plot(times, D_squared_analytical_thermal * 1e6, 
         'g--', linewidth=LINE_WIDTH_ANALYTICAL, 
         label=f'Analytical (Thermal): K = {K_analytical_thermal*1e6:.4f} mm^2/s', alpha=0.8)

ax2.plot(times, D_squared_analytical_mass * 1e6, 
         'm--', linewidth=LINE_WIDTH_ANALYTICAL, 
         label=f'Analytical (Mass): K = {K_analytical_mass*1e6:.4f} mm^2/s', alpha=0.8)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('D^2 (mm^2)', fontsize=FONT_SIZE_LABEL)
ax2.set_title(f'D-Squared Law: Diameter Squared vs Time\n' + 
              f'R^2 = {r_value**2:.6f}',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND-1, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_D_Squared_Evolution.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_D_Squared_Evolution.eps'))
print("  Saved: 02_D_Squared_Evolution.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: K COMPARISON BAR CHART
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 3: K COMPARISON")
print("=" * 70)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 3a: Bar chart comparison
K_values = ,[K_analytical_thermal * 1e6, K_analytical_mass * 1e6, K_numerical * 1e6]
labels = ,['Analytical\n(Thermal B_T)', 'Analytical\n(Mass B_M)', 'Numerical\n(Linear Fit)']
colors = ,['green', 'magenta', 'red']

bars = ax3a.bar(labels, K_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, K_val in zip(bars, K_values):
    height = bar.get_height()
    ax3a.text(bar.get_x() + bar.get_width()/2., height,
              f'{K_val:.4f}',
              ha='center', va='bottom', fontsize=FONT_SIZE_LEGEND, fontweight='bold')

ax3a.set_ylabel('K (mm^2/s)', fontsize=FONT_SIZE_LABEL)
ax3a.set_title('Vaporization Constant Comparison',
               fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3a.grid(True, alpha=0.3, axis='y')
ax3a.tick_params(labelsize=FONT_SIZE_TICK)

# Subplot 3b: Error metrics
error_labels = ,['Thermal\nAbs Error', 'Thermal\nRel Error (%)', 'Mass\nAbs Error', 'Mass\nRel Error (%)']
error_values = ,[K_error_thermal_abs * 1e6, K_error_thermal_rel, 
                K_error_mass_abs * 1e6, K_error_mass_rel]
error_colors = ,['orange', 'orange', 'purple', 'purple']

bars_error = ax3b.bar(error_labels, error_values, color=error_colors, 
                       alpha=0.7, edgecolor='black', linewidth=2)

# Add value labels on bars
for bar, val in zip(bars_error, error_values):
    height = bar.get_height()
    ax3b.text(bar.get_x() + bar.get_width()/2., height,
              f'{val:.4f}',
              ha='center', va='bottom', fontsize=FONT_SIZE_LEGEND-1, fontweight='bold')

ax3b.set_ylabel('Error Magnitude', fontsize=FONT_SIZE_LABEL)
ax3b.set_title('K Error Metrics',
               fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3b.grid(True, alpha=0.3, axis='y')
ax3b.tick_params(labelsize=FONT_SIZE_TICK-1)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_K_Comparison.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_K_Comparison.eps'))
print("  Saved: 03_K_Comparison.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: TEMPERATURE FIELDS (INITIAL AND FINAL)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 4: TEMPERATURE FIELDS")
print("=" * 70)

if temp_initial is not None and temp_final is not None:
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Initial temperature
    im1 = ax4a.contourf(x_coords * 1000, y_coords * 1000, temp_initial, 
                        levels=50, cmap='hot')
    ax4a.set_xlabel('X (mm)', fontsize=FONT_SIZE_LABEL)
    ax4a.set_ylabel('Y (mm)', fontsize=FONT_SIZE_LABEL)
    ax4a.set_title(f'Initial Temperature (t = {times,[0]:.3f} s)',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax4a.set_aspect('equal')
    cbar1 = plt.colorbar(im1, ax=ax4a)
    cbar1.set_label('Temperature (K)', fontsize=FONT_SIZE_LABEL)
    
    # Final temperature
    im2 = ax4b.contourf(x_coords * 1000, y_coords * 1000, temp_final, 
                        levels=50, cmap='hot')
    ax4b.set_xlabel('X (mm)', fontsize=FONT_SIZE_LABEL)
    ax4b.set_ylabel('Y (mm)', fontsize=FONT_SIZE_LABEL)
    ax4b.set_title(f'Final Temperature (t = {times,[-1]:.3f} s)',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax4b.set_aspect('equal')
    cbar2 = plt.colorbar(im2, ax=ax4b)
    cbar2.set_label('Temperature (K)', fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_Temperature_Fields.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, '04_Temperature_Fields.eps'))
    print("  Saved: 04_Temperature_Fields.png/.eps")
    plt.close()
else:
    print("  WARNING: Temperature fields not available")

# ============================================================================
# PLOT 5: SPALDING NUMBER EVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 5: SPALDING NUMBER EVOLUTION")
print("=" * 70)

fig5, ax5 = plt.subplots(figsize=(12, 8))

# Plot numerical Spalding number
ax5.plot(times, B_M_numerical, 
         'bo-', linewidth=LINE_WIDTH_NUMERICAL, markersize=MARKER_SIZE-2,
         label='Numerical B_M', alpha=0.8)

# Add analytical Spalding number line
ax5.axhline(y=B_M_analytical, color='green', linestyle='--', 
            linewidth=LINE_WIDTH_ANALYTICAL, 
            label=f'Analytical B_M = {B_M_analytical:.6f}')

ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Spalding Number B_M', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Mass Transfer Spalding Number Evolution',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Spalding_Number.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Spalding_Number.eps'))
print("  Saved: 05_Spalding_Number.png/.eps")
plt.close()

# ============================================================================
# PLOT 6: MASS TRANSFER RATE (mdot)
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 6: MASS TRANSFER RATE")
print("=" * 70)

fig6, ax6 = plt.subplots(figsize=(12, 8))

# Plot mdot
ax6.plot(times, mdot, 
         'ro-', linewidth=LINE_WIDTH_NUMERICAL, markersize=MARKER_SIZE-2,
         label='mdot (d/dt of Vap_rho0)', alpha=0.8)

ax6.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax6.set_ylabel('Mass Transfer Rate (kg/s)', fontsize=FONT_SIZE_LABEL)
ax6.set_title('Mass Transfer Rate Evolution',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax6.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax6.grid(True, alpha=0.3)
ax6.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_Mass_Transfer_Rate.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '06_Mass_Transfer_Rate.eps'))
print("  Saved: 06_Mass_Transfer_Rate.png/.eps")
plt.close()

# ============================================================================
# PLOT 7: TOTAL MASS TRANSFER
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 7: TOTAL MASS TRANSFER")
print("=" * 70)

fig7, ax7 = plt.subplots(figsize=(12, 8))

# Plot cumulative mass transfer
ax7.plot(times, vap_rho_sums, 
         'go-', linewidth=LINE_WIDTH_NUMERICAL, markersize=MARKER_SIZE-2,
         label='Total Mass Transfer (Vap_rho0)', alpha=0.8)

ax7.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax7.set_ylabel('Total Mass Transfer (kg)', fontsize=FONT_SIZE_LABEL)
ax7.set_title('Cumulative Mass Transfer',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax7.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax7.grid(True, alpha=0.3)
ax7.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '07_Total_Mass_Transfer.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '07_Total_Mass_Transfer.eps'))
print("  Saved: 07_Total_Mass_Transfer.png/.eps")
plt.close()

# ============================================================================
# PLOT 8: INTERFACIAL AREA EVOLUTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOT 8: INTERFACIAL AREA EVOLUTION")
print("=" * 70)

fig8, ax8 = plt.subplots(figsize=(12, 8))

# Plot interfacial area
ax8.plot(times, interfacial_areas, 
         'mo-', linewidth=LINE_WIDTH_NUMERICAL, markersize=MARKER_SIZE-2,
         label='Interfacial Area', alpha=0.8)

ax8.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax8.set_ylabel('Interfacial Area (m)', fontsize=FONT_SIZE_LABEL)
ax8.set_title(f'Interfacial Area Evolution\n(grad_eta_mag > {grad_eta_threshold})',
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax8.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax8.grid(True, alpha=0.3)
ax8.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '08_Interfacial_Area.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '08_Interfacial_Area.eps'))
print("  Saved: 08_Interfacial_Area.png/.eps")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  - 01_Diameter_Evolution.png/.eps")
print(f"  - 02_D_Squared_Evolution.png/.eps")
print(f"  - 03_K_Comparison.png/.eps")
print(f"  - 04_Temperature_Fields.png/.eps")
print(f"  - 05_Spalding_Number.png/.eps")
print(f"  - 06_Mass_Transfer_Rate.png/.eps")
print(f"  - 07_Total_Mass_Transfer.png/.eps")
print(f"  - 08_Interfacial_Area.png/.eps")

print(f"\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"\nPhysical Parameters:")
print(f"  Liquid density:       {rho_liquid} kg/m^3")
print(f"  Gas density:          {rho_gas} kg/m^3")
print(f"  Surface temperature:  {T_s} K")
print(f"  Ambient temperature:  {T_inf} K")
print(f"  Initial diameter:     {D0*1000:.2f} mm")

print(f"\nVaporization Constant Results:")
print(f"  K_analytical (thermal):  {K_analytical_thermal*1e6:.4f} mm^2/s")
print(f"  K_analytical (mass):     {K_analytical_mass*1e6:.4f} mm^2/s")
print(f"  K_numerical:             {K_numerical*1e6:.4f} mm^2/s")
print(f"  Error vs thermal:        {K_error_thermal_rel:.2f}%")
print(f"  Error vs mass:           {K_error_mass_rel:.2f}%")

print(f"\nSpalding Numbers:")
print(f"  B_T (thermal):           {B_T_analytical:.6f}")
print(f"  B_M (mass transfer):     {B_M_analytical:.6f}")

print(f"\nRegression Quality:")
print(f"  R^2:                     {r_value**2:.6f}")

print("\n" + "=" * 70)
