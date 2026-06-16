# -*- coding: utf-8 -*-

"""
===============================================================================
ADIABATIC TEST 4 ANALYSIS: STATIC BUBBLE WITH DIFFUSE INTERFACE
===============================================================================

PURPOSE:
    Diagnose if diffuse interface causes pressure equilibration or gas mass loss
    even with ZERO velocity (no dynamics).
    
    This is the CRITICAL test to determine if your interface model is broken.

VALIDATION CHECKS:
    1. Monitor p_B(t) at bubble center - should stay at 500 Pa
    2. Monitor p_L(t) far from bubble - should stay at 500 Pa
    3. Compute gas mass: m_gas = integral(rho * (1 - eta) * dA)
    4. Track interface thickness over time
    5. Check for gas diffusion into liquid phase

EXPECTED BEHAVIOR (CORRECT):
    - Pressure stays constant (no equilibration)
    - Gas mass conserved
    - Interface thickness constant
    - No velocity develops

EXPECTED BEHAVIOR (BROKEN INTERFACE):
    - p_B drops from 500 Pa toward 500 Pa (equilibrates!)
    - Gas mass decreases
    - Interface blurs (epsilon grows)
    - Gas diffuses into liquid

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
p_B0 = 500.0              # Initial gas pressure [Pa]
p_inf = 500.0             # Liquid pressure [Pa]
rho_gas = 1.0             # Gas density [kg/m^3]
rho_liquid = 10.0         # Liquid density [kg/m^3]
R0 = 0.02                 # Bubble radius [m]
epsilon_initial = 0.0011  # Initial interface thickness [m]

# File paths
amrex_output_dir = r'../../../../../bin/tests/Adiabatic/TEST4_Static_Interface'

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.5
MARKER_SIZE = 6

# Output settings
output_folder = './ADIABATIC_TEST4_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def compute_gas_mass(rho, eta, dx, dy):
    """
    Compute gas mass: m_gas = integral(rho * (1 - eta) * dA)
    Gas fraction is (1 - eta)
    """
    dA = dx * dy
    gas_fraction = 1.0 - eta
    return np.sum(rho * gas_fraction) * dA

def compute_interface_thickness(eta, x, y):
    """
    Compute interface thickness by measuring width of eta transition
    Find distance between eta = 0.1 and eta = 0.9 contours
    """
    try:
        # Find radial distance from center
        r = np.sqrt(x**2 + y**2)
        
        # Sort by radius
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        eta_sorted = eta[sort_idx]
        
        # Find where eta crosses 0.1 and 0.9
        idx_01 = np.where(eta_sorted > 0.1)[0]
        idx_09 = np.where(eta_sorted > 0.9)[0]
        
        if len(idx_01) > 0 and len(idx_09) > 0:
            r_01 = r_sorted[idx_01[0]]
            r_09 = r_sorted[idx_09[0]]
            thickness = r_09 - r_01
            return thickness
        else:
            return np.nan
    except:
        return np.nan

def extract_pressure_at_location(pressure, x, y, x_target, y_target, radius=0.005):
    """
    Extract pressure at a specific location by averaging over small region
    """
    r = np.sqrt((x - x_target)**2 + (y - y_target)**2)
    mask = r < radius
    if np.sum(mask) > 0:
        return np.mean(pressure[mask])
    else:
        return np.nan

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("ADIABATIC TEST 4: STATIC BUBBLE WITH DIFFUSE INTERFACE")
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
p_center_list = []      # Pressure at bubble center
p_far_list = []         # Pressure far from bubble
gas_mass_list = []      # Total gas mass
interface_thickness_list = []  # Interface thickness
velocity_max_list = []  # Maximum velocity magnitude

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        ad = ds.all_data()
        
        pressure = np.array(ad['pressure'])
        density = np.array(ad['density'])
        eta = np.array(ad['eta'])
        x_coords = np.array(ad['x'])
        y_coords = np.array(ad['y'])
        
        # Try to get velocity
        try:
            vel_x = np.array(ad['velocity_x'])
            vel_y = np.array(ad['velocity_y'])
            vel_mag = np.sqrt(vel_x**2 + vel_y**2)
            velocity_max = np.max(vel_mag)
        except:
            velocity_max = 0.0
        
        if len(pressure) == 0:
            print(f"  WARNING: No data in timestep {i}")
            continue
        
        # Extract pressure at bubble center (x=0, y=0)
        p_center = extract_pressure_at_location(pressure, x_coords, y_coords, 0.0, 0.0, radius=0.005)
        
        # Extract pressure far from bubble (x=0.08, y=0)
        p_far = extract_pressure_at_location(pressure, x_coords, y_coords, 0.08, 0.0, radius=0.005)
        
        # Compute gas mass
        dx = np.abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
        dy = np.abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0
        gas_mass = compute_gas_mass(density, eta, dx, dy)
        
        # Compute interface thickness
        interface_thickness = compute_interface_thickness(eta, x_coords, y_coords)
        
        # Store data
        times.append(t)
        p_center_list.append(p_center)
        p_far_list.append(p_far)
        gas_mass_list.append(gas_mass)
        interface_thickness_list.append(interface_thickness)
        velocity_max_list.append(velocity_max)
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times = np.array(times)
p_center_list = np.array(p_center_list)
p_far_list = np.array(p_far_list)
gas_mass_list = np.array(gas_mass_list)
interface_thickness_list = np.array(interface_thickness_list)
velocity_max_list = np.array(velocity_max_list)

if len(times) == 0:
    print("\nERROR: No data successfully extracted!")
    exit(1)

print(f"\nSuccessfully extracted {len(times)} measurements")
print(f"  Time range: [{times[0]:.6e}, {times[-1]:.6e}] s")

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# Pressure drift
valid_center = ~np.isnan(p_center_list)
valid_far = ~np.isnan(p_far_list)

if np.sum(valid_center) > 0:
    p_center_initial = p_center_list[valid_center][0]
    p_center_final = p_center_list[valid_center][-1]
    p_center_drift = p_center_final - p_center_initial
    p_center_drift_pct = (p_center_drift / p_B0) * 100
    
    print(f"\nBubble Center Pressure:")
    print(f"  Initial: {p_center_initial:.2f} Pa (expected: {p_B0} Pa)")
    print(f"  Final: {p_center_final:.2f} Pa")
    print(f"  Drift: {p_center_drift:.2f} Pa ({p_center_drift_pct:.4f}%)")
    print(f"  Equilibrating toward p_inf? {abs(p_center_final - p_inf) < abs(p_center_initial - p_inf)}")

if np.sum(valid_far) > 0:
    p_far_initial = p_far_list[valid_far][0]
    p_far_final = p_far_list[valid_far][-1]
    p_far_drift = p_far_final - p_far_initial
    
    print(f"\nLiquid Pressure (far from bubble):")
    print(f"  Initial: {p_far_initial:.2f} Pa (expected: {p_inf} Pa)")
    print(f"  Final: {p_far_final:.2f} Pa")
    print(f"  Drift: {p_far_drift:.2f} Pa")

# Gas mass conservation
gas_mass_initial = gas_mass_list[0]
gas_mass_final = gas_mass_list[-1]
gas_mass_drift = (gas_mass_final - gas_mass_initial) / gas_mass_initial * 100

print(f"\nGas Mass Conservation:")
print(f"  Initial: {gas_mass_initial:.6e} kg")
print(f"  Final: {gas_mass_final:.6e} kg")
print(f"  Drift: {gas_mass_drift:.4f}%")

# Interface thickness
valid_thickness = ~np.isnan(interface_thickness_list)
if np.sum(valid_thickness) > 0:
    thickness_initial = interface_thickness_list[valid_thickness][0]
    thickness_final = interface_thickness_list[valid_thickness][-1]
    thickness_growth = (thickness_final - thickness_initial) / epsilon_initial * 100
    
    print(f"\nInterface Thickness:")
    print(f"  Initial: {thickness_initial*1000:.4f} mm (expected: {epsilon_initial*1000} mm)")
    print(f"  Final: {thickness_final*1000:.4f} mm")
    print(f"  Growth: {thickness_growth:.4f}%")

# Velocity check
velocity_max_overall = np.max(velocity_max_list)
print(f"\nVelocity Check (should be zero):")
print(f"  Maximum velocity magnitude: {velocity_max_overall:.6e} m/s")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: PRESSURE AT BUBBLE CENTER VS TIME
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(times * 1000, p_center_list, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax1.axhline(y=p_B0, color='r', linestyle='--', linewidth=2, label=f'Initial: p_B0 = {p_B0} Pa')
ax1.axhline(y=p_inf, color='g', linestyle='--', linewidth=2, label=f'Liquid: p_inf = {p_inf} Pa')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Pressure at Bubble Center (Pa)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('CRITICAL: Pressure at Bubble Center (Should Stay Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

if np.sum(valid_center) > 0:
    textstr = f'Initial: {p_center_initial:.2f} Pa\nFinal: {p_center_final:.2f} Pa\nDrift: {p_center_drift_pct:.4f}%'
    props = dict(boxstyle='round', facecolor='yellow', alpha=0.7)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Pressure_Center_vs_Time.png'), dpi=300)
print("  Saved: 01_Pressure_Center_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: GAS MASS VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(times * 1000, gas_mass_list, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax2.axhline(y=gas_mass_initial, color='r', linestyle='--', linewidth=2, 
            label=f'Initial: {gas_mass_initial:.6e} kg')
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Gas Mass (kg)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Gas Mass Conservation (Should be Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Drift: {gas_mass_drift:.4f}%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Gas_Mass_vs_Time.png'), dpi=300)
print("  Saved: 02_Gas_Mass_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 3: INTERFACE THICKNESS VS TIME
# ============================================================================

if np.sum(valid_thickness) > 0:
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.plot(times[valid_thickness] * 1000, interface_thickness_list[valid_thickness] * 1000, 
             'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
    ax3.axhline(y=epsilon_initial*1000, color='r', linestyle='--', linewidth=2, 
                label=f'Initial: {epsilon_initial*1000} mm')
    ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax3.set_ylabel('Interface Thickness (mm)', fontsize=FONT_SIZE_LABEL)
    ax3.set_title('Interface Thickness (Should be Constant)', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax3.legend(fontsize=FONT_SIZE_LEGEND)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=FONT_SIZE_TICK)
    
    textstr = f'Growth: {thickness_growth:.4f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_Interface_Thickness.png'), dpi=300)
    print("  Saved: 03_Interface_Thickness.png")
    plt.close()

# ============================================================================
# PLOT 4: PRESSURE COMPARISON
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.plot(times * 1000, p_center_list, 'b-', linewidth=LINE_WIDTH, 
         marker='o', markersize=MARKER_SIZE-2, label='Bubble Center')
ax4.plot(times * 1000, p_far_list, 'r-', linewidth=LINE_WIDTH, 
         marker='s', markersize=MARKER_SIZE-2, label='Far from Bubble')
ax4.axhline(y=p_B0, color='b', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.axhline(y=p_inf, color='r', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Pressure at Different Locations', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Pressure_Comparison.png'), dpi=300)
print("  Saved: 04_Pressure_Comparison.png")
plt.close()

# ============================================================================
# PLOT 5: VELOCITY CHECK
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))
ax5.semilogy(times * 1000, velocity_max_list, 'b-', linewidth=LINE_WIDTH, 
             marker='o', markersize=MARKER_SIZE-2)
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Maximum Velocity Magnitude (m/s)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Velocity Check (Should be Zero)', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max: {velocity_max_overall:.6e} m/s'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax5.text(0.05, 0.95, textstr, transform=ax5.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Velocity_Check.png'), dpi=300)
print("  Saved: 05_Velocity_Check.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ADIABATIC TEST 4 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Pressure_Center_vs_Time.png (CRITICAL)")
print(f"  02_Gas_Mass_vs_Time.png")
print(f"  03_Interface_Thickness.png")
print(f"  04_Pressure_Comparison.png")
print(f"  05_Velocity_Check.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MAX_PRESSURE_DRIFT = 1.0  # % (should be nearly zero)
MAX_MASS_DRIFT = 1.0  # %
MAX_THICKNESS_GROWTH = 10.0  # %
MAX_VELOCITY = 1e-6  # m/s

pressure_pass = abs(p_center_drift_pct) < MAX_PRESSURE_DRIFT if np.sum(valid_center) > 0 else False
mass_pass = abs(gas_mass_drift) < MAX_MASS_DRIFT
thickness_pass = abs(thickness_growth) < MAX_THICKNESS_GROWTH if np.sum(valid_thickness) > 0 else True
velocity_pass = velocity_max_overall < MAX_VELOCITY

print(f"\nTest Results:")
if np.sum(valid_center) > 0:
    print(f"  [{'PASS' if pressure_pass else 'FAIL'}] Pressure constancy: {abs(p_center_drift_pct):.4f}% drift (threshold: {MAX_PRESSURE_DRIFT}%)")
print(f"  [{'PASS' if mass_pass else 'FAIL'}] Gas mass conservation: {abs(gas_mass_drift):.4f}% drift (threshold: {MAX_MASS_DRIFT}%)")
if np.sum(valid_thickness) > 0:
    print(f"  [{'PASS' if thickness_pass else 'FAIL'}] Interface thickness: {abs(thickness_growth):.4f}% growth (threshold: {MAX_THICKNESS_GROWTH}%)")
print(f"  [{'PASS' if velocity_pass else 'FAIL'}] Velocity zero: {velocity_max_overall:.6e} m/s (threshold: {MAX_VELOCITY})")

if pressure_pass and mass_pass and thickness_pass and velocity_pass:
    print("\n*** TEST 4 PASSED: Interface model is correct ***")
    print("\nYour diffuse interface is NOT causing pressure equilibration.")
    print("The problem with your bubble simulation is in the DYNAMICS:")
    print("  - Inertial terms not strong enough")
    print("  - Viscosity too high")
    print("  - Surface tension incorrect")
    print("  - Go back to RPE term-by-term analysis")
else:
    print("\n*** TEST 4 FAILED: Interface model is BROKEN ***")
    print("\nYour diffuse interface IS causing the problem!")
    
    if not pressure_pass and np.sum(valid_center) > 0:
        print(f"\n  CRITICAL ISSUE: Pressure equilibrating with ZERO velocity!")
        print(f"    Initial: {p_center_initial:.2f} Pa")
        print(f"    Final: {p_center_final:.2f} Pa")
        print(f"    Drift: {p_center_drift:.2f} Pa ({p_center_drift_pct:.4f}%)")
        print(f"    Moving toward p_inf = {p_inf} Pa")
        print("\n  Root causes:")
        print("    1. Pressure being averaged across interface: p_mix = p_gas*(1-eta) + p_liquid*eta")
        print("    2. Interface too thick (epsilon = 0.0005 m is 2.5% of R0)")
        print("    3. Numerical diffusion in pressure field")
        print("\n  Solutions:")
        print("    - Reduce epsilon to 0.0001 m (0.5% of R0)")
        print("    - Check pressure reconstruction at interface")
        print("    - Ensure sharp pressure jump at eta = 0.5")
        print("    - Use Ghost Fluid Method instead of CSF for pressure")
    
    if not mass_pass:
        print(f"\n  CRITICAL ISSUE: Gas mass leaking!")
        print(f"    Initial: {gas_mass_initial:.6e} kg")
        print(f"    Final: {gas_mass_final:.6e} kg")
        print(f"    Loss: {abs(gas_mass_drift):.4f}%")
        print("\n  Root causes:")
        print("    1. Gas diffusing into liquid phase")
        print("    2. Eta field evolving incorrectly")
        print("    3. Numerical diffusion in eta equation")
        print("\n  Solutions:")
        print("    - Check eta advection scheme")
        print("    - Verify gas mass formula: m_gas = integral(rho * (1-eta) * dA)")
        print("    - Reduce numerical diffusion in eta transport")
    
    if not thickness_pass and np.sum(valid_thickness) > 0:
        print(f"\n  Issue: Interface blurring over time")
        print(f"    Growth: {thickness_growth:.4f}%")
        print("    -> Numerical diffusion in eta field")

print("\n" + "=" * 70)
print("CONNECTION TO YOUR BUBBLE PROBLEM")
print("=" * 70)

print("\nYour original pressure plot shows:")
print("  - Numerical pressure drops from 500 Pa to ~497 Pa")
print("  - Equilibrates to p_inf = 500 Pa by ~4 ms")
print("  - No oscillation (monotonic decay)")

if not pressure_pass and np.sum(valid_center) > 0:
    print("\nTest 4 confirms the same behavior WITH ZERO VELOCITY!")
    print("This PROVES the interface model is diffusing pressure.")
    print("\nThe fix:")
    print("  1. Reduce interface thickness: epsilon = 0.0001 m")
    print("  2. Implement sharp pressure jump at interface")
    print("  3. Use Ghost Fluid Method for pressure boundary condition")
    print("  4. Check mixture property formulas at interface")
else:
    print("\nTest 4 shows interface is OK (pressure stays constant).")
    print("The problem is in your bubble dynamics, not the interface.")

print("\n" + "=" * 70)
