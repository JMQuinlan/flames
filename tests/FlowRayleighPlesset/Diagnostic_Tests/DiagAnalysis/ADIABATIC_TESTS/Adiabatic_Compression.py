# -*- coding: utf-8 -*-

"""
===============================================================================
ADIABATIC TEST 1 ANALYSIS: 1D SINUSOIDAL COMPRESSION
===============================================================================

PURPOSE:
    Validate pure adiabatic gas dynamics without interface complications.
    This is the cleanest possible test of EOS coupling.
    
    For adiabatic (isentropic) flow, the quantity:
    
    S = p / rho^gamma
    
    must remain constant in space and time.

VALIDATION CHECKS:
    1. Compute isentropic invariant S = p/rho^gamma at all points
    2. Check if S is spatially uniform (constant across domain)
    3. Check if S is temporally constant (no drift over time)
    4. Compute effective gamma: gamma_eff = d(ln p) / d(ln rho)
    5. Verify gamma_eff = gamma_input (should be 1.4)

EXPECTED BEHAVIOR:
    - S should be constant everywhere and at all times
    - gamma_eff should equal 1.4 (input gamma)
    - Compression waves propagate without changing S
    - No energy dissipation (inviscid flow)

FAILURE MODES:
    - S drifts with time -> energy equation wrong
    - S varies spatially -> EOS coupling broken
    - gamma_eff != 1.4 -> wrong polytropic relation
    - gamma_eff ~ 1.0 -> isothermal instead of adiabatic

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
gamma_input = 1.4             # Adiabatic index (input value)
rho_0 = 1.0                   # Initial density [kg/m^3]
p_0 = 1.0                     # Initial pressure [Pa]

# File paths
amrex_output_dir = r'../../../../bin/tests/Adiabatic/TEST1_1D_Compression'  # Directory containing AMReX plot files

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.0
MARKER_SIZE = 6

# Output settings
output_folder = './ADIABATIC_TEST1_Analysis'
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

def compute_isentropic_invariant(p, rho, gamma):
    """
    Compute isentropic invariant: S = p / rho^gamma
    This should be constant for adiabatic flow
    """
    # Avoid division by zero
    rho_safe = np.where(rho > 1e-12, rho, 1e-12)
    return p / (rho_safe**gamma)

def compute_effective_gamma(p, rho):
    """
    Compute effective gamma from data: gamma_eff = d(ln p) / d(ln rho)
    Uses robust linear regression on log-log plot
    """
    # Remove any zeros or negative values
    valid = (p > 1e-12) & (rho > 1e-12) & np.isfinite(p) & np.isfinite(rho)
    p_valid = p[valid]
    rho_valid = rho[valid]
    
    if len(p_valid) < 2:
        return np.nan
    
    # Check if there's actually variation in the data
    if np.std(rho_valid) < 1e-12 or np.std(p_valid) < 1e-12:
        return np.nan
    
    try:
        # Log-log regression
        log_p = np.log(p_valid)
        log_rho = np.log(rho_valid)
        
        # Remove any infinities or NaNs
        valid_log = np.isfinite(log_p) & np.isfinite(log_rho)
        if np.sum(valid_log) < 2:
            return np.nan
        
        log_p = log_p[valid_log]
        log_rho = log_rho[valid_log]
        
        # Linear fit: log(p) = gamma_eff * log(rho) + const
        # Use numpy's polyfit which is more robust
        coeffs = np.polyfit(log_rho, log_p, 1)
        gamma_eff = coeffs[0]
        
        # Sanity check
        if not np.isfinite(gamma_eff) or gamma_eff < 0 or gamma_eff > 10:
            return np.nan
        
        return gamma_eff
        
    except Exception as e:
        print(f"    Warning: Could not compute gamma_eff: {e}")
        return np.nan

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("ADIABATIC TEST 1: 1D SINUSOIDAL COMPRESSION ANALYSIS")
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

# Sort by timestep number
plot_files.sort(key=extract_timestep_number)

print(f"\nFound {len(plot_files)} plot files (excluding .old files)")

# ============================================================================
# EXTRACT DATA FROM SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRACTING DATA FROM SIMULATION")
print("=" * 70)

times = []
S_mean_list = []
S_std_list = []
S_min_list = []
S_max_list = []
gamma_eff_list = []

# Store spatial profiles for selected timesteps
spatial_data = {}
selected_timesteps = [0, len(plot_files)//4, len(plot_files)//2, 3*len(plot_files)//4, len(plot_files)-1]

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        # Get all data
        ad = ds.all_data()
        
        # Extract fields
        pressure = np.array(ad['pressure'])
        density = np.array(ad['density'])
        x_coords = np.array(ad['x'])
        
        # Check if data is valid
        if len(pressure) == 0 or len(density) == 0:
            print(f"  WARNING: No data in timestep {i}")
            continue
        
        # Compute isentropic invariant
        S = compute_isentropic_invariant(pressure, density, gamma_input)
        
        # Remove any invalid values
        S_valid = S[np.isfinite(S)]
        
        if len(S_valid) == 0:
            print(f"  WARNING: No valid S values at t={t:.6e} s")
            continue
        
        # Compute statistics
        S_mean = np.mean(S_valid)
        S_std = np.std(S_valid)
        S_min = np.min(S_valid)
        S_max = np.max(S_valid)
        
        # Compute effective gamma (skip if fails)
        gamma_eff = compute_effective_gamma(pressure, density)
        
        # Store data
        times.append(t)
        S_mean_list.append(S_mean)
        S_std_list.append(S_std)
        S_min_list.append(S_min)
        S_max_list.append(S_max)
        gamma_eff_list.append(gamma_eff)
        
        # Store spatial profiles for selected timesteps
        if i in selected_timesteps:
            # Sort by x coordinate
            sort_idx = np.argsort(x_coords)
            spatial_data[i] = {
                'time': t,
                'x': x_coords[sort_idx],
                'pressure': pressure[sort_idx],
                'density': density[sort_idx],
                'S': S[sort_idx]
            }
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times = np.array(times)
S_mean_list = np.array(S_mean_list)
S_std_list = np.array(S_std_list)
S_min_list = np.array(S_min_list)
S_max_list = np.array(S_max_list)
gamma_eff_list = np.array(gamma_eff_list)

if len(times) == 0:
    print("\nERROR: No data successfully extracted!")
    print("Check that:")
    print("  1. Simulation has run and produced output files")
    print("  2. Output directory path is correct")
    print("  3. Plot files contain 'pressure' and 'density' fields")
    exit(1)

print(f"\nSuccessfully extracted {len(times)} measurements")
print(f"  Time range: [{times[0]:.6e}, {times[-1]:.6e}] s")

# ============================================================================
# COMPUTE THEORETICAL ISENTROPIC INVARIANT
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING THEORETICAL VALUES")
print("=" * 70)

# Theoretical isentropic invariant (should be constant)
S_theory = p_0 / (rho_0**gamma_input)

print(f"  Theoretical S = p_0 / rho_0^gamma = {S_theory:.6e}")
print(f"  Input gamma = {gamma_input}")

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS")
print("=" * 70)

# Temporal drift in S
S_drift_abs = np.max(S_mean_list) - np.min(S_mean_list)
S_drift_rel = (S_drift_abs / S_theory) * 100

# Spatial variation in S (average over time)
S_spatial_variation = np.mean(S_std_list) / S_theory * 100

# Effective gamma statistics (only valid values)
gamma_eff_valid = gamma_eff_list[np.isfinite(gamma_eff_list)]
if len(gamma_eff_valid) > 0:
    gamma_eff_mean = np.mean(gamma_eff_valid)
    gamma_eff_std = np.std(gamma_eff_valid)
    gamma_error = abs(gamma_eff_mean - gamma_input) / gamma_input * 100
else:
    gamma_eff_mean = np.nan
    gamma_eff_std = np.nan
    gamma_error = np.nan
    print("  WARNING: Could not compute effective gamma for any timestep")

print(f"\nIsentropic Invariant (S = p/rho^gamma):")
print(f"  Theoretical value: {S_theory:.6e}")
print(f"  Mean numerical value: {np.mean(S_mean_list):.6e}")
print(f"  Temporal drift: {S_drift_abs:.6e} ({S_drift_rel:.4f}%)")
print(f"  Spatial variation (avg std): {S_spatial_variation:.4f}%")

if np.isfinite(gamma_eff_mean):
    print(f"\nEffective Gamma:")
    print(f"  Input gamma: {gamma_input}")
    print(f"  Mean gamma_eff: {gamma_eff_mean:.6f} +/- {gamma_eff_std:.6f}")
    print(f"  Error: {gamma_error:.4f}%")
    print(f"  Valid measurements: {len(gamma_eff_valid)}/{len(gamma_eff_list)}")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: ISENTROPIC INVARIANT VS TIME
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(times * 1000, S_mean_list, 'b-', linewidth=LINE_WIDTH, label='Mean S')
ax1.fill_between(times * 1000, S_min_list, S_max_list, alpha=0.3, label='Min-Max Range')
ax1.axhline(y=S_theory, color='r', linestyle='--', linewidth=2, label=f'Theory: S = {S_theory:.6e}')
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Isentropic Invariant: S = p/rho^gamma', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Isentropic Invariant vs Time (Should be Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Drift: {S_drift_rel:.4f}%\nSpatial var: {S_spatial_variation:.4f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Isentropic_Invariant_vs_Time.png'), dpi=300)
print("  Saved: 01_Isentropic_Invariant_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: EFFECTIVE GAMMA VS TIME
# ============================================================================

if len(gamma_eff_valid) > 0:
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Plot only valid gamma values
    times_valid = times[np.isfinite(gamma_eff_list)]
    gamma_valid = gamma_eff_list[np.isfinite(gamma_eff_list)]
    
    ax2.plot(times_valid * 1000, gamma_valid, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
    ax2.axhline(y=gamma_input, color='r', linestyle='--', linewidth=2, label=f'Input gamma = {gamma_input}')
    ax2.axhline(y=1.0, color='g', linestyle='--', linewidth=2, alpha=0.5, label='Isothermal (gamma = 1.0)')
    ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Effective Gamma: d(ln p) / d(ln rho)', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Effective Gamma vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.legend(fontsize=FONT_SIZE_LEGEND)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    
    textstr = f'Mean: {gamma_eff_mean:.6f}\nStd: {gamma_eff_std:.6f}\nError: {gamma_error:.4f}%'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '02_Effective_Gamma_vs_Time.png'), dpi=300)
    print("  Saved: 02_Effective_Gamma_vs_Time.png")
    plt.close()

# ============================================================================
# PLOT 3: SPATIAL PROFILES AT SELECTED TIMES
# ============================================================================

if len(spatial_data) > 0:
    fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for timestep_idx in selected_timesteps[:4]:
        if timestep_idx in spatial_data and plot_idx < 4:
            data = spatial_data[timestep_idx]
            ax = axes[plot_idx]
            
            ax.plot(data['x'], data['S'], 'b-', linewidth=LINE_WIDTH)
            ax.axhline(y=S_theory, color='r', linestyle='--', linewidth=2, label='Theory')
            ax.set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL-2)
            ax.set_ylabel('S = p/rho^gamma', fontsize=FONT_SIZE_LABEL-2)
            ax.set_title(f't = {data["time"]*1000:.3f} ms', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
            ax.legend(fontsize=FONT_SIZE_LEGEND-2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=FONT_SIZE_TICK-2)
            
            plot_idx += 1
    
    plt.suptitle('Spatial Distribution of Isentropic Invariant', 
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '03_Spatial_Profiles.png'), dpi=300)
    print("  Saved: 03_Spatial_Profiles.png")
    plt.close()

# ============================================================================
# PLOT 4: PRESSURE VS DENSITY (LOG-LOG)
# ============================================================================

if len(plot_files)//2 in spatial_data:
    data = spatial_data[len(plot_files)//2]
    
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    # Plot data
    ax4.loglog(data['density'], data['pressure'], 'bo', markersize=MARKER_SIZE, 
               alpha=0.7, label='Numerical Data')
    
    # Theoretical line: p = const * rho^gamma
    rho_range = np.linspace(np.min(data['density']), np.max(data['density']), 100)
    p_theory = S_theory * rho_range**gamma_input
    ax4.loglog(rho_range, p_theory, 'r-', linewidth=LINE_WIDTH, 
               label=f'Theory: p ~ rho^{gamma_input}')
    
    # Fit line with effective gamma (if available)
    mid_idx = len(plot_files)//2
    if mid_idx < len(gamma_eff_list) and np.isfinite(gamma_eff_list[mid_idx]):
        gamma_eff_mid = gamma_eff_list[mid_idx]
        p_fit = S_theory * rho_range**gamma_eff_mid
        ax4.loglog(rho_range, p_fit, 'g--', linewidth=LINE_WIDTH, 
                   label=f'Fit: p ~ rho^{gamma_eff_mid:.3f}')
    
    ax4.set_xlabel('Density rho (kg/m^3)', fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel('Pressure p (Pa)', fontsize=FONT_SIZE_LABEL)
    ax4.set_title(f'Pressure vs Density (Log-Log) at t = {data["time"]*1000:.3f} ms', 
                  fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax4.legend(fontsize=FONT_SIZE_LEGEND)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.tick_params(labelsize=FONT_SIZE_TICK)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '04_Pressure_vs_Density_LogLog.png'), dpi=300)
    print("  Saved: 04_Pressure_vs_Density_LogLog.png")
    plt.close()

# ============================================================================
# PLOT 5: RELATIVE DEVIATION FROM THEORY
# ============================================================================

fig5, ax5 = plt.subplots(figsize=(12, 8))

# Compute relative deviation
S_rel_dev = (S_mean_list - S_theory) / S_theory * 100

ax5.plot(times * 1000, S_rel_dev, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Perfect Agreement')
ax5.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('Relative Deviation from Theory (%)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Isentropic Invariant: Deviation from Theory', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.legend(fontsize=FONT_SIZE_LEGEND)
ax5.grid(True, alpha=0.3)
ax5.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Relative_Deviation.png'), dpi=300)
print("  Saved: 05_Relative_Deviation.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ADIABATIC TEST 1 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Isentropic_Invariant_vs_Time.png")
if len(gamma_eff_valid) > 0:
    print(f"  02_Effective_Gamma_vs_Time.png")
if len(spatial_data) > 0:
    print(f"  03_Spatial_Profiles.png")
if len(plot_files)//2 in spatial_data:
    print(f"  04_Pressure_vs_Density_LogLog.png")
print(f"  05_Relative_Deviation.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MAX_DRIFT = 1.0  # Maximum allowed temporal drift (%)
MAX_SPATIAL_VAR = 1.0  # Maximum allowed spatial variation (%)
MAX_GAMMA_ERROR = 5.0  # Maximum allowed gamma error (%)

drift_pass = S_drift_rel < MAX_DRIFT
spatial_pass = S_spatial_variation < MAX_SPATIAL_VAR
gamma_pass = (gamma_error < MAX_GAMMA_ERROR) if np.isfinite(gamma_error) else False

print(f"\nTest Results:")
print(f"  [{'PASS' if drift_pass else 'FAIL'}] Temporal drift: {S_drift_rel:.4f}% (threshold: {MAX_DRIFT}%)")
print(f"  [{'PASS' if spatial_pass else 'FAIL'}] Spatial variation: {S_spatial_variation:.4f}% (threshold: {MAX_SPATIAL_VAR}%)")
if np.isfinite(gamma_error):
    print(f"  [{'PASS' if gamma_pass else 'FAIL'}] Gamma accuracy: error = {gamma_error:.4f}% (threshold: {MAX_GAMMA_ERROR}%)")
else:
    print(f"  [FAIL] Gamma accuracy: Could not compute")

if drift_pass and spatial_pass and gamma_pass:
    print("\n*** TEST 1 PASSED: Adiabatic gas dynamics working correctly ***")
    print("\nYour EOS coupling is correct for pure gas (no interface).")
    print("The problem with your bubble simulation is likely:")
    print("  - Interface treatment (diffuse interface model)")
    print("  - Gas mass conservation across interface")
    print("  - Pressure boundary condition at interface")
else:
    print("\n*** TEST 1 FAILED: Adiabatic gas dynamics NOT working correctly ***")
    print("\nThis is a FUNDAMENTAL issue - no interface involved!")
    print("\nPossible root causes:")
    
    if not drift_pass:
        print("\n  Issue: Temporal drift in S")
        print("    -> Energy equation not conserving entropy")
        print("    -> Check energy update scheme")
        print("    -> Verify pressure reconstruction from energy")
    
    if not spatial_pass:
        print("\n  Issue: Spatial variation in S")
        print("    -> EOS coupling broken")
        print("    -> Pressure and density not following p ~ rho^gamma")
        print("    -> Check how pressure is computed from density and energy")
    
    if not gamma_pass and np.isfinite(gamma_error):
        if gamma_eff_mean < 1.2:
            print("\n  Issue: gamma_eff ~ 1.0 (isothermal behavior)")
            print("    -> Temperature being held constant")
            print("    -> Check if you're using isothermal EOS instead of adiabatic")
            print("    -> Verify gamma is being used in pressure calculation")
        else:
            print("\n  Issue: Wrong gamma value")
            print("    -> Check if using 3D formula (3*gamma) instead of 2D (2*gamma)")
            print("    -> Verify polytropic exponent in EOS")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)

if drift_pass and spatial_pass and gamma_pass:
    print("\nSince pure gas works, the issue is at the bubble interface.")
    print("Check your bubble simulation for:")
    print("  1. Gas mass conservation (compute m_gas = rho * pi * R^2)")
    print("  2. Pressure jump at interface (should have discontinuity)")
    print("  3. Interface thickness (epsilon) relative to bubble radius")
else:
    print("\nFix the fundamental EOS issue first before proceeding.")
    print("Check your energy equation and pressure reconstruction.")

print("\n" + "=" * 70)
