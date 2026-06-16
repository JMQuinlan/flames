# -*- coding: utf-8 -*-

"""
===============================================================================
ADIABATIC TEST 3 ANALYSIS: RADIAL COMPRESSION (2D CYLINDRICAL) - FIXED
===============================================================================

PURPOSE:
    Validate 2D cylindrical adiabatic relation: p_B ~ R^(-2*gamma)
    This tests the EXACT geometry of your bubble problem without interface.
    
    For 2D cylindrical radial compression:
    
    p_B = p_B0 * (R0/R)^(2*gamma)
    
    NOT 3D spherical: p_B = p_B0 * (R0/R)^(3*gamma)

VALIDATION CHECKS:
    1. Compute effective radius from velocity field or mass conservation
    2. Verify pressure follows: p ~ R^(-2*gamma) = R^(-2.8)
    3. Check gas mass conservation: m_gas = integral(rho * dA) = constant
    4. Compute effective exponent: n_eff = d(ln p) / d(ln R)
    5. Verify radial symmetry (no azimuthal variation)

EXPECTED BEHAVIOR:
    - Radial compression (R decreases)
    - Pressure increases as R^(-2.8) (2D cylindrical)
    - Gas mass conserved
    - n_eff = -2.8 (NOT -4.2 which is 3D spherical)

FAILURE MODES:
    - n_eff = -4.2 -> using 3D spherical formula
    - Gas mass not conserved -> leaking or numerical diffusion
    - n_eff shallower than -2.8 -> mass loss
    - Asymmetry -> radial symmetry broken

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

# Physical parameters (MUST MATCH TEST INPUT FILE)
gamma_input = 1.4             # Adiabatic index
p_B0 = 550.0                  # Initial pressure [Pa]
rho_0 = 1.0                   # Initial density [kg/m^3]
alpha = 5.0                   # Compression rate [1/s]
R0 = 0.02                     # Initial effective radius [m]

# Expected exponent for 2D cylindrical
n_expected_2D = -2.0 * gamma_input  # -2.8
n_expected_3D = -3.0 * gamma_input  # -4.2

# File paths
amrex_output_dir = r'../../../../bin/tests/Adiabatic/TEST3_Radial_Compression'

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.5
MARKER_SIZE = 6

# Output settings
output_folder = './ADIABATIC_TEST3_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL SOLUTIONS
# ============================================================================

def R_analytical(t, R0, alpha):
    """Analytical effective radius: R(t) = R0 * exp(-alpha * t)"""
    return R0 * np.exp(-alpha * t)

def p_analytical_2D(R, p_B0, R0, gamma):
    """Analytical pressure for 2D cylindrical: p = p_B0 * (R0/R)^(2*gamma)"""
    return p_B0 * (R0 / R)**(2 * gamma)

def p_analytical_3D(R, p_B0, R0, gamma):
    """Analytical pressure for 3D spherical: p = p_B0 * (R0/R)^(3*gamma)"""
    return p_B0 * (R0 / R)**(3 * gamma)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def compute_effective_radius_from_mass(rho, dx, dy, rho_0, R0):
    """
    Compute effective radius from mass conservation
    For uniform density: rho * pi * R^2 = rho_0 * pi * R0^2
    Therefore: R = R0 * sqrt(rho_0 / rho)
    """
    rho_mean = np.mean(rho)
    R_eff = R0 * np.sqrt(rho_0 / rho_mean)
    return R_eff

def compute_gas_mass(rho, dx, dy):
    """Compute total gas mass: m = integral(rho * dA)"""
    dA = dx * dy
    return np.sum(rho) * dA

def compute_effective_exponent(p, R):
    """
    Compute effective exponent: n_eff = d(ln p) / d(ln R)
    For 2D cylindrical: n_eff = -2*gamma = -2.8
    For 3D spherical: n_eff = -3*gamma = -4.2
    """
    valid = (p > 1e-12) & (R > 1e-12) & np.isfinite(p) & np.isfinite(R)
    p_valid = p[valid]
    R_valid = R[valid]
    
    if len(p_valid) < 2:
        return np.nan
    
    if np.std(R_valid) < 1e-12 or np.std(p_valid) < 1e-12:
        return np.nan
    
    try:
        log_p = np.log(p_valid)
        log_R = np.log(R_valid)
        
        valid_log = np.isfinite(log_p) & np.isfinite(log_R)
        if np.sum(valid_log) < 2:
            return np.nan
        
        log_p = log_p[valid_log]
        log_R = log_R[valid_log]
        
        # Linear fit: log(p) = n_eff * log(R) + const
        coeffs = np.polyfit(log_R, log_p, 1)
        n_eff = coeffs[0]
        
        if not np.isfinite(n_eff) or n_eff > 0 or n_eff < -10:
            return np.nan
        
        return n_eff
        
    except Exception as e:
        return np.nan

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("ADIABATIC TEST 3: RADIAL COMPRESSION (2D CYLINDRICAL) ANALYSIS")
print("=" * 70)

print("\n" + "=" * 70)
print("LOADING SIMULATION DATA")
print("=" * 70)

if not os.path.exists(amrex_output_dir):
    print(f"ERROR: Directory not found: {amrex_output_dir}")
    exit(1)

plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and '.old' not in item:
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No plot files found in {amrex_output_dir}")
    exit(1)

plot_files.sort(key=extract_timestep_number)

print(f"\nFound {len(plot_files)} plot files (excluding .old files)")

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM SIMULATION")
print("=" * 70)

times = []
R_eff_list = []
p_mean_list = []
rho_mean_list = []
gas_mass_list = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        ad = ds.all_data()
        
        pressure = np.array(ad['pressure'])
        density = np.array(ad['density'])
        x_coords = np.array(ad['x'])
        y_coords = np.array(ad['y'])
        
        if len(pressure) == 0 or len(density) == 0:
            print(f"  WARNING: No data in timestep {i}")
            continue
        
        # Compute mean density and pressure
        rho_mean = np.mean(density)
        p_mean = np.mean(pressure)
        
        # Compute effective radius from mass conservation
        # For uniform compression: rho * R^2 = rho_0 * R0^2
        R_eff = compute_effective_radius_from_mass(density, 1.0, 1.0, rho_0, R0)
        
        # Compute gas mass
        dx = np.abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        dy = np.abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
        gas_mass = compute_gas_mass(density, dx, dy)
        
        # Store data
        times.append(t)
        R_eff_list.append(R_eff)
        p_mean_list.append(p_mean)
        rho_mean_list.append(rho_mean)
        gas_mass_list.append(gas_mass)
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times = np.array(times)
R_eff_list = np.array(R_eff_list)
p_mean_list = np.array(p_mean_list)
rho_mean_list = np.array(rho_mean_list)
gas_mass_list = np.array(gas_mass_list)

if len(times) == 0:
    print("\nERROR: No data successfully extracted!")
    exit(1)

# Compute effective exponent from time series
n_eff_global = compute_effective_exponent(p_mean_list, R_eff_list)

print(f"\nSuccessfully extracted {len(times)} measurements")
print(f"  Time range: [{times[0]:.6e}, {times[-1]:.6e}] s")
print(f"  Radius range: [{np.min(R_eff_list)*1000:.4f}, {np.max(R_eff_list)*1000:.4f}] mm")
print(f"  Density range: [{np.min(rho_mean_list):.4f}, {np.max(rho_mean_list):.4f}] kg/m^3")

# ============================================================================
# COMPUTE ANALYTICAL SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL SOLUTIONS")
print("=" * 70)

R_ana = R_analytical(times, R0, alpha)
p_ana_2D = p_analytical_2D(R_ana, p_B0, R0, gamma_input)
p_ana_3D = p_analytical_3D(R_ana, p_B0, R0, gamma_input)

print(f"  Initial radius: R0 = {R0*1000} mm")
print(f"  Initial pressure: p_B0 = {p_B0} Pa")
print(f"  Compression rate: alpha = {alpha} s^-1")
print(f"  Expected exponent (2D): n = {n_expected_2D:.4f}")
print(f"  Expected exponent (3D): n = {n_expected_3D:.4f}")

# ============================================================================
# COMPUTE ERRORS
# ============================================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

# Radius errors
R_abs_error = np.abs(R_eff_list - R_ana)
R_rel_error = (R_abs_error / R_ana) * 100

# Pressure errors (compare to 2D analytical)
p_abs_error = np.abs(p_mean_list - p_ana_2D)
p_rel_error = (p_abs_error / p_ana_2D) * 100

# Gas mass conservation
gas_mass_drift = (np.max(gas_mass_list) - np.min(gas_mass_list)) / np.mean(gas_mass_list) * 100

# Exponent error
if np.isfinite(n_eff_global):
    n_error = abs(n_eff_global - n_expected_2D) / abs(n_expected_2D) * 100
else:
    n_error = np.nan

print(f"\nRadius Evolution:")
print(f"  Max relative error: {np.max(R_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(R_rel_error):.4f}%")

print(f"\nPressure Evolution (vs 2D analytical):")
print(f"  Max relative error: {np.max(p_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(p_rel_error):.4f}%")

print(f"\nGas Mass Conservation:")
print(f"  Initial mass: {gas_mass_list[0]:.6e} kg")
print(f"  Final mass: {gas_mass_list[-1]:.6e} kg")
print(f"  Drift: {gas_mass_drift:.4f}%")

if np.isfinite(n_eff_global):
    print(f"\nEffective Exponent:")
    print(f"  Expected (2D): {n_expected_2D:.4f}")
    print(f"  Expected (3D): {n_expected_3D:.4f}")
    print(f"  Numerical: {n_eff_global:.4f}")
    print(f"  Error from 2D: {n_error:.4f}%")
    
    # Diagnostic
    if abs(n_eff_global - n_expected_3D) < abs(n_eff_global - n_expected_2D):
        print(f"  WARNING: Closer to 3D spherical exponent!")
else:
    print(f"\nEffective Exponent: Could not compute")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: RADIUS VS TIME
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(times * 1000, R_ana * 1000, 'b-', linewidth=LINE_WIDTH, label='Analytical: R = R0*exp(-alpha*t)')
ax1.plot(times * 1000, R_eff_list * 1000, 'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Effective Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Radial Compression: Radius vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max error: {np.max(R_rel_error):.4f}%\nMean error: {np.mean(R_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Radius_vs_Time.png'), dpi=300)
print("  Saved: 01_Radius_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: PRESSURE VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(times * 1000, p_ana_2D, 'b-', linewidth=LINE_WIDTH, label='Analytical 2D: p ~ R^(-2.8)')
ax2.plot(times * 1000, p_ana_3D, 'g--', linewidth=LINE_WIDTH, label='Analytical 3D: p ~ R^(-4.2)', alpha=0.7)
ax2.plot(times * 1000, p_mean_list, 'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Pressure Evolution: 2D vs 3D Comparison', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max error (vs 2D): {np.max(p_rel_error):.4f}%\nMean error: {np.mean(p_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Pressure_vs_Time.png'), dpi=300)
print("  Saved: 02_Pressure_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 3: PRESSURE VS RADIUS (LOG-LOG) - CRITICAL DIAGNOSTIC
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))

# Plot numerical data
ax3.loglog(R_eff_list * 1000, p_mean_list, 'ro', markersize=MARKER_SIZE, 
           alpha=0.7, label='Numerical', zorder=3)

# Plot analytical curves
R_range = np.linspace(np.min(R_eff_list), np.max(R_eff_list), 100)
p_2D_curve = p_analytical_2D(R_range, p_B0, R0, gamma_input)
p_3D_curve = p_analytical_3D(R_range, p_B0, R0, gamma_input)

ax3.loglog(R_range * 1000, p_2D_curve, 'b-', linewidth=LINE_WIDTH, 
           label=f'2D Cylindrical: p ~ R^{n_expected_2D:.1f}', zorder=1)
ax3.loglog(R_range * 1000, p_3D_curve, 'g--', linewidth=LINE_WIDTH, 
           label=f'3D Spherical: p ~ R^{n_expected_3D:.1f}', alpha=0.7, zorder=2)

ax3.set_xlabel('Radius (mm)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('CRITICAL: Pressure vs Radius (Log-Log)\nSlope reveals geometry', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3, which='both')
ax3.tick_params(labelsize=FONT_SIZE_TICK)

if np.isfinite(n_eff_global):
    textstr = f'Numerical slope: {n_eff_global:.4f}\nExpected (2D): {n_expected_2D:.4f}\nExpected (3D): {n_expected_3D:.4f}'
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.7)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Pressure_vs_Radius_LogLog.png'), dpi=300)
print("  Saved: 03_Pressure_vs_Radius_LogLog.png")
plt.close()

# ============================================================================
# PLOT 4: DENSITY VS TIME
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))

# Analytical density from mass conservation: rho = rho_0 * (R0/R)^2
rho_ana = rho_0 * (R0 / R_ana)**2

ax4.plot(times * 1000, rho_ana, 'b-', linewidth=LINE_WIDTH, label='Analytical: rho ~ R^(-2)')
ax4.plot(times * 1000, rho_mean_list, 'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Density Evolution (Mass Conservation)', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Density_vs_Time.png'), dpi=300)
print("  Saved: 04_Density_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 5: GAS MASS CONSERVATION
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.plot(times * 1000, gas_mass_list, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax5.axhline(y=gas_mass_list[0], color='r', linestyle='--', linewidth=2, 
            label=f'Initial mass: {gas_mass_list[0]:.6e} kg')
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Gas Mass (kg)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Gas Mass Conservation (Should be Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND)
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Drift: {gas_mass_drift:.4f}%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Gas_Mass_Conservation.png'), dpi=300)
print("  Saved: 05_Gas_Mass_Conservation.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ADIABATIC TEST 3 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Radius_vs_Time.png")
print(f"  02_Pressure_vs_Time.png")
print(f"  03_Pressure_vs_Radius_LogLog.png (CRITICAL DIAGNOSTIC)")
print(f"  04_Density_vs_Time.png")
print(f"  05_Gas_Mass_Conservation.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MAX_R_ERROR = 5.0  # %
MAX_P_ERROR = 10.0  # %
MAX_MASS_DRIFT = 2.0  # %
MAX_N_ERROR = 10.0  # %

R_pass = np.mean(R_rel_error) < MAX_R_ERROR
p_pass = np.mean(p_rel_error) < MAX_P_ERROR
mass_pass = gas_mass_drift < MAX_MASS_DRIFT
n_pass = (n_error < MAX_N_ERROR) if np.isfinite(n_error) else False

print(f"\nTest Results:")
print(f"  [{'PASS' if R_pass else 'FAIL'}] Radius evolution: {np.mean(R_rel_error):.4f}% error (threshold: {MAX_R_ERROR}%)")
print(f"  [{'PASS' if p_pass else 'FAIL'}] Pressure evolution: {np.mean(p_rel_error):.4f}% error (threshold: {MAX_P_ERROR}%)")
print(f"  [{'PASS' if mass_pass else 'FAIL'}] Gas mass conservation: {gas_mass_drift:.4f}% drift (threshold: {MAX_MASS_DRIFT}%)")
if np.isfinite(n_error):
    print(f"  [{'PASS' if n_pass else 'FAIL'}] Exponent accuracy: n_eff={n_eff_global:.4f}, error={n_error:.4f}% (threshold: {MAX_N_ERROR}%)")
else:
    print(f"  [FAIL] Exponent: Could not compute")

if R_pass and p_pass and mass_pass and n_pass:
    print("\n*** TEST 3 PASSED: 2D cylindrical adiabatic relation correct ***")
    print("\nYour radial geometry handling is working correctly!")
    print("Since this passes but your bubble fails, the issue is:")
    print("  - Gas-liquid INTERFACE treatment")
    print("  - Diffuse interface losing gas mass")
    print("  - Pressure jump at eta=0.5 not enforced")
    print("  - Interface thickness (epsilon) too large")
else:
    print("\n*** TEST 3 FAILED: Radial geometry problem ***")
    
    if not R_pass:
        print("\n  Issue: Radius not compressing correctly")
        print("    -> Check radial velocity field")
        print("    -> Verify boundary conditions")
    
    if not p_pass:
        print("\n  Issue: Pressure not following 2D relation")
        if np.isfinite(n_eff_global):
            if abs(n_eff_global - n_expected_3D) < abs(n_eff_global - n_expected_2D):
                print("    -> YOU ARE USING 3D SPHERICAL FORMULA!")
                print("    -> Change exponent from 3*gamma to 2*gamma")
            elif abs(n_eff_global) < abs(n_expected_2D):
                print("    -> Exponent too shallow (gas mass leaking)")
        print("    -> Check EOS implementation for radial geometry")
    
    if not mass_pass:
        print("\n  Issue: Gas mass not conserved")
        print("    -> Mass leaking through boundaries")
        print("    -> Numerical diffusion too high")
        print("    -> Check boundary conditions")
    
    if not n_pass and np.isfinite(n_error):
        print("\n  Issue: Wrong exponent")
        print(f"    -> Numerical: {n_eff_global:.4f}")
        print(f"    -> Expected (2D): {n_expected_2D:.4f}")
        print(f"    -> Expected (3D): {n_expected_3D:.4f}")

print("\n" + "=" * 70)
print("CONNECTION TO YOUR BUBBLE PROBLEM")
print("=" * 70)

print("\nLooking at your pressure plot (02_Pressure_vs_Time.png):")
print("  - Analytical (blue) follows 2D adiabatic: p ~ R^(-2.8)")
print("  - Numerical (red) drops from 550 Pa to ~497 Pa by 10 ms")
print("  - Numerical pressure equilibrates to p_inf = 500 Pa")
print("  - This suggests gas NOT compressing adiabatically")

if R_pass and p_pass and mass_pass and n_pass:
    print("\nSince Test 3 PASSES:")
    print("  -> Pure gas radial compression works correctly")
    print("  -> Problem is at the GAS-LIQUID INTERFACE")
    print("\nNext steps:")
    print("  1. Check interface thickness: epsilon = 0.004 m vs R0 = 0.02 m (20% of radius!)")
    print("  2. Verify gas mass inside bubble: m_gas = integral(rho * (1-eta) * dA)")
    print("  3. Check pressure jump at interface")
    print("  4. Look for gas diffusing into liquid phase (eta field)")

print("\n" + "=" * 70)
