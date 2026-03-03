# -*- coding: utf-8 -*-

"""
===============================================================================
ADIABATIC TEST 2 ANALYSIS: HOMOGENEOUS BOX COMPRESSION
===============================================================================

PURPOSE:
    Validate pure adiabatic compression with uniform velocity divergence.
    This is the SIMPLEST possible test - no waves, no gradients, no geometry.
    
    For uniform compression with div(u) = constant:
    
    rho(t) = rho_0 * exp(-div_u * t)
    p(t) = p_0 * (rho/rho_0)^gamma
    S = p / rho^gamma = constant

VALIDATION CHECKS:
    1. Verify density follows exponential: rho(t) = rho_0 * exp(-div_u * t)
    2. Verify pressure follows: p(t) = p_0 * (rho/rho_0)^gamma
    3. Check isentropic invariant S = p/rho^gamma remains constant
    4. Verify NO spatial variation (perfectly uniform)
    5. Compute effective gamma: gamma_eff = d(ln p) / d(ln rho)

EXPECTED BEHAVIOR:
    - Density increases exponentially
    - Pressure increases as rho^gamma
    - S constant everywhere and at all times
    - gamma_eff = 1.4 (input gamma)
    - Zero spatial variation

FAILURE MODES:
    - rho doesn't follow exponential -> mass conservation broken
    - S drifts with time -> energy equation wrong
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
div_u = -1.0                  # Velocity divergence [1/s] (negative = compression)

# File paths
amrex_output_dir = r'../../../../bin/tests/Adiabatic/TEST2_Homogeneous_Compression'  # Directory containing AMReX plot files

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.5
MARKER_SIZE = 6

# Output settings
output_folder = './ADIABATIC_TEST2_Analysis'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ============================================================================
# ANALYTICAL SOLUTIONS
# ============================================================================

def rho_analytical(t, rho_0, div_u):
    """Analytical density: rho(t) = rho_0 * exp(-div_u * t)"""
    return rho_0 * np.exp(-div_u * t)

def p_analytical(t, p_0, rho_0, div_u, gamma):
    """Analytical pressure: p(t) = p_0 * (rho/rho_0)^gamma"""
    rho_t = rho_analytical(t, rho_0, div_u)
    return p_0 * (rho_t / rho_0)**gamma

def S_analytical(p_0, rho_0, gamma):
    """Analytical isentropic invariant: S = p_0 / rho_0^gamma"""
    return p_0 / (rho_0**gamma)

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
    """Compute isentropic invariant: S = p / rho^gamma"""
    rho_safe = np.where(rho > 1e-12, rho, 1e-12)
    return p / (rho_safe**gamma)

def compute_effective_gamma(p, rho):
    """Compute effective gamma: gamma_eff = d(ln p) / d(ln rho)"""
    valid = (p > 1e-12) & (rho > 1e-12) & np.isfinite(p) & np.isfinite(rho)
    p_valid = p[valid]
    rho_valid = rho[valid]
    
    if len(p_valid) < 2:
        return np.nan
    
    if np.std(rho_valid) < 1e-12 or np.std(p_valid) < 1e-12:
        return np.nan
    
    try:
        log_p = np.log(p_valid)
        log_rho = np.log(rho_valid)
        
        valid_log = np.isfinite(log_p) & np.isfinite(log_rho)
        if np.sum(valid_log) < 2:
            return np.nan
        
        log_p = log_p[valid_log]
        log_rho = log_rho[valid_log]
        
        coeffs = np.polyfit(log_rho, log_p, 1)
        gamma_eff = coeffs[0]
        
        if not np.isfinite(gamma_eff) or gamma_eff < 0 or gamma_eff > 10:
            return np.nan
        
        return gamma_eff
        
    except Exception as e:
        return np.nan

# ============================================================================
# FIND AND SORT PLOT FILES
# ============================================================================

print("=" * 70)
print("ADIABATIC TEST 2: HOMOGENEOUS BOX COMPRESSION ANALYSIS")
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
rho_mean_list = []
rho_std_list = []
p_mean_list = []
p_std_list = []
S_mean_list = []
S_std_list = []
gamma_eff_list = []

for i, plot_file in enumerate(plot_files):
    try:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        
        ad = ds.all_data()
        
        pressure = np.array(ad['pressure'])
        density = np.array(ad['density'])
        
        if len(pressure) == 0 or len(density) == 0:
            print(f"  WARNING: No data in timestep {i}")
            continue
        
        # Compute isentropic invariant
        S = compute_isentropic_invariant(pressure, density, gamma_input)
        S_valid = S[np.isfinite(S)]
        
        if len(S_valid) == 0:
            print(f"  WARNING: No valid S values at t={t:.6e} s")
            continue
        
        # Compute statistics
        rho_mean = np.mean(density)
        rho_std = np.std(density)
        p_mean = np.mean(pressure)
        p_std = np.std(pressure)
        S_mean = np.mean(S_valid)
        S_std = np.std(S_valid)
        
        # Compute effective gamma
        gamma_eff = compute_effective_gamma(pressure, density)
        
        # Store data
        times.append(t)
        rho_mean_list.append(rho_mean)
        rho_std_list.append(rho_std)
        p_mean_list.append(p_mean)
        p_std_list.append(p_std)
        S_mean_list.append(S_mean)
        S_std_list.append(S_std)
        gamma_eff_list.append(gamma_eff)
        
        if (i + 1) % 10 == 0 or i == len(plot_files) - 1:
            print(f"  Processed {i + 1}/{len(plot_files)} timesteps")
    
    except Exception as e:
        print(f"  ERROR processing {plot_file}: {e}")
        continue

times = np.array(times)
rho_mean_list = np.array(rho_mean_list)
rho_std_list = np.array(rho_std_list)
p_mean_list = np.array(p_mean_list)
p_std_list = np.array(p_std_list)
S_mean_list = np.array(S_mean_list)
S_std_list = np.array(S_std_list)
gamma_eff_list = np.array(gamma_eff_list)

if len(times) == 0:
    print("\nERROR: No data successfully extracted!")
    exit(1)

print(f"\nSuccessfully extracted {len(times)} measurements")
print(f"  Time range: [{times[0]:.6e}, {times[-1]:.6e}] s")

# ============================================================================
# COMPUTE ANALYTICAL SOLUTIONS
# ============================================================================

print("\n" + "=" * 70)
print("COMPUTING ANALYTICAL SOLUTIONS")
print("=" * 70)

rho_ana = rho_analytical(times, rho_0, div_u)
p_ana = p_analytical(times, p_0, rho_0, div_u, gamma_input)
S_theory = S_analytical(p_0, rho_0, gamma_input)

print(f"  Theoretical S = p_0 / rho_0^gamma = {S_theory:.6e}")
print(f"  Input gamma = {gamma_input}")
print(f"  Velocity divergence = {div_u} s^-1")

# ============================================================================
# COMPUTE ERRORS
# ============================================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

# Density errors
rho_abs_error = np.abs(rho_mean_list - rho_ana)
rho_rel_error = (rho_abs_error / rho_ana) * 100

# Pressure errors
p_abs_error = np.abs(p_mean_list - p_ana)
p_rel_error = (p_abs_error / p_ana) * 100

# Isentropic invariant drift
S_drift_abs = np.max(S_mean_list) - np.min(S_mean_list)
S_drift_rel = (S_drift_abs / S_theory) * 100

# Spatial uniformity (should be zero)
rho_spatial_var = np.mean(rho_std_list) / np.mean(rho_mean_list) * 100
p_spatial_var = np.mean(p_std_list) / np.mean(p_mean_list) * 100

# Effective gamma
gamma_eff_valid = gamma_eff_list[np.isfinite(gamma_eff_list)]
if len(gamma_eff_valid) > 0:
    gamma_eff_mean = np.mean(gamma_eff_valid)
    gamma_eff_std = np.std(gamma_eff_valid)
    gamma_error = abs(gamma_eff_mean - gamma_input) / gamma_input * 100
else:
    gamma_eff_mean = np.nan
    gamma_eff_std = np.nan
    gamma_error = np.nan

print(f"\nDensity Evolution:")
print(f"  Max absolute error: {np.max(rho_abs_error):.6e} kg/m^3")
print(f"  Mean absolute error: {np.mean(rho_abs_error):.6e} kg/m^3")
print(f"  Max relative error: {np.max(rho_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(rho_rel_error):.4f}%")
print(f"  Spatial variation: {rho_spatial_var:.6f}%")

print(f"\nPressure Evolution:")
print(f"  Max absolute error: {np.max(p_abs_error):.6e} Pa")
print(f"  Mean absolute error: {np.mean(p_abs_error):.6e} Pa")
print(f"  Max relative error: {np.max(p_rel_error):.4f}%")
print(f"  Mean relative error: {np.mean(p_rel_error):.4f}%")
print(f"  Spatial variation: {p_spatial_var:.6f}%")

print(f"\nIsentropic Invariant:")
print(f"  Theoretical value: {S_theory:.6e}")
print(f"  Mean numerical value: {np.mean(S_mean_list):.6e}")
print(f"  Temporal drift: {S_drift_abs:.6e} ({S_drift_rel:.4f}%)")

if np.isfinite(gamma_eff_mean):
    print(f"\nEffective Gamma:")
    print(f"  Input gamma: {gamma_input}")
    print(f"  Mean gamma_eff: {gamma_eff_mean:.6f} +/- {gamma_eff_std:.6f}")
    print(f"  Error: {gamma_error:.4f}%")

# ============================================================================
# PLOTTING SECTION
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

# ============================================================================
# PLOT 1: DENSITY VS TIME
# ============================================================================

fig1, ax1 = plt.subplots(figsize=(12, 8))
ax1.plot(times * 1000, rho_ana, 'b-', linewidth=LINE_WIDTH, label='Analytical: rho = exp(t)')
ax1.plot(times * 1000, rho_mean_list, 'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)
ax1.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Density Evolution: Exponential Growth', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND)
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max error: {np.max(rho_rel_error):.4f}%\nMean error: {np.mean(rho_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Density_vs_Time.png'), dpi=300)
print("  Saved: 01_Density_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 2: PRESSURE VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(times * 1000, p_ana, 'b-', linewidth=LINE_WIDTH, label='Analytical: p = exp(gamma*t)')
ax2.plot(times * 1000, p_mean_list, 'ro', markersize=MARKER_SIZE, label='Numerical', alpha=0.7)
ax2.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Pressure Evolution', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND)
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Max error: {np.max(p_rel_error):.4f}%\nMean error: {np.mean(p_rel_error):.4f}%'
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '02_Pressure_vs_Time.png'), dpi=300)
print("  Saved: 02_Pressure_vs_Time.png")
plt.close()

# ============================================================================
# PLOT 3: ISENTROPIC INVARIANT VS TIME
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(12, 8))
ax3.plot(times * 1000, S_mean_list, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax3.axhline(y=S_theory, color='r', linestyle='--', linewidth=2, label=f'Theory: S = {S_theory:.6e}')
ax3.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Isentropic Invariant: S = p/rho^gamma', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Isentropic Invariant vs Time (Should be Constant)', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND)
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)

textstr = f'Drift: {S_drift_rel:.4f}%'
props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, 
         fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '03_Isentropic_Invariant.png'), dpi=300)
print("  Saved: 03_Isentropic_Invariant.png")
plt.close()

# ============================================================================
# PLOT 4: PRESSURE VS DENSITY (LOG-LOG)
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(12, 8))
ax4.loglog(rho_mean_list, p_mean_list, 'bo', markersize=MARKER_SIZE, alpha=0.7, label='Numerical')
ax4.loglog(rho_ana, p_ana, 'r-', linewidth=LINE_WIDTH, label=f'Analytical: p ~ rho^{gamma_input}')

ax4.set_xlabel('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Pressure vs Density (Log-Log)', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND)
ax4.grid(True, alpha=0.3, which='both')
ax4.tick_params(labelsize=FONT_SIZE_TICK)

if np.isfinite(gamma_eff_mean):
    textstr = f'gamma_eff = {gamma_eff_mean:.4f}\nExpected = {gamma_input}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.05, 0.95, textstr, transform=ax4.transAxes, 
             fontsize=FONT_SIZE_LEGEND, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '04_Pressure_vs_Density.png'), dpi=300)
print("  Saved: 04_Pressure_vs_Density.png")
plt.close()

# ============================================================================
# PLOT 5: RELATIVE ERRORS
# ============================================================================

fig5, (ax5a, ax5b) = plt.subplots(2, 1, figsize=(12, 12))

# Density error
ax5a.semilogy(times * 1000, rho_rel_error, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax5a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5a.set_ylabel('Density Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5a.set_title('Density Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5a.grid(True, alpha=0.3, which='both')
ax5a.tick_params(labelsize=FONT_SIZE_TICK)

# Pressure error
ax5b.semilogy(times * 1000, p_rel_error, 'r-', linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax5b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax5b.set_ylabel('Pressure Relative Error (%)', fontsize=FONT_SIZE_LABEL)
ax5b.set_title('Pressure Error (Log Scale)', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5b.grid(True, alpha=0.3, which='both')
ax5b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '05_Relative_Errors.png'), dpi=300)
print("  Saved: 05_Relative_Errors.png")
plt.close()

# ============================================================================
# PLOT 6: SPATIAL UNIFORMITY CHECK
# ============================================================================

fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(12, 12))

# Density spatial variation
ax6a.plot(times * 1000, rho_std_list / rho_mean_list * 100, 'b-', 
          linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax6a.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6a.set_ylabel('Density Spatial Std Dev (%)', fontsize=FONT_SIZE_LABEL)
ax6a.set_title('Spatial Uniformity: Density (Should be Zero)', 
               fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax6a.grid(True, alpha=0.3)
ax6a.tick_params(labelsize=FONT_SIZE_TICK)

# Pressure spatial variation
ax6b.plot(times * 1000, p_std_list / p_mean_list * 100, 'r-', 
          linewidth=LINE_WIDTH, marker='o', markersize=MARKER_SIZE-2)
ax6b.set_xlabel('Time (ms)', fontsize=FONT_SIZE_LABEL)
ax6b.set_ylabel('Pressure Spatial Std Dev (%)', fontsize=FONT_SIZE_LABEL)
ax6b.set_title('Spatial Uniformity: Pressure (Should be Zero)', 
               fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax6b.grid(True, alpha=0.3)
ax6b.tick_params(labelsize=FONT_SIZE_TICK)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '06_Spatial_Uniformity.png'), dpi=300)
print("  Saved: 06_Spatial_Uniformity.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ADIABATIC TEST 2 ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nFiles generated:")
print(f"  01_Density_vs_Time.png")
print(f"  02_Pressure_vs_Time.png")
print(f"  03_Isentropic_Invariant.png")
print(f"  04_Pressure_vs_Density.png")
print(f"  05_Relative_Errors.png")
print(f"  06_Spatial_Uniformity.png")

print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Pass/Fail criteria
MAX_RHO_ERROR = 2.0  # %
MAX_P_ERROR = 2.0  # %
MAX_S_DRIFT = 1.0  # %
MAX_SPATIAL_VAR = 0.5  # %
MAX_GAMMA_ERROR = 5.0  # %

rho_pass = np.mean(rho_rel_error) < MAX_RHO_ERROR
p_pass = np.mean(p_rel_error) < MAX_P_ERROR
S_pass = S_drift_rel < MAX_S_DRIFT
spatial_pass = (rho_spatial_var < MAX_SPATIAL_VAR) and (p_spatial_var < MAX_SPATIAL_VAR)
gamma_pass = (gamma_error < MAX_GAMMA_ERROR) if np.isfinite(gamma_error) else False

print(f"\nTest Results:")
print(f"  [{'PASS' if rho_pass else 'FAIL'}] Density evolution: {np.mean(rho_rel_error):.4f}% error (threshold: {MAX_RHO_ERROR}%)")
print(f"  [{'PASS' if p_pass else 'FAIL'}] Pressure evolution: {np.mean(p_rel_error):.4f}% error (threshold: {MAX_P_ERROR}%)")
print(f"  [{'PASS' if S_pass else 'FAIL'}] Isentropic invariant drift: {S_drift_rel:.4f}% (threshold: {MAX_S_DRIFT}%)")
print(f"  [{'PASS' if spatial_pass else 'FAIL'}] Spatial uniformity: rho={rho_spatial_var:.4f}%, p={p_spatial_var:.4f}% (threshold: {MAX_SPATIAL_VAR}%)")
if np.isfinite(gamma_error):
    print(f"  [{'PASS' if gamma_pass else 'FAIL'}] Gamma accuracy: {gamma_error:.4f}% error (threshold: {MAX_GAMMA_ERROR}%)")

if rho_pass and p_pass and S_pass and spatial_pass and gamma_pass:
    print("\n*** TEST 2 PASSED: Adiabatic compression working correctly ***")
    print("\nYour EOS is correct for uniform compression.")
    print("Since this passes, the issue with your bubble is likely:")
    print("  - Radial geometry effects (2D cylindrical vs 3D spherical)")
    print("  - Interface treatment (diffuse interface losing gas mass)")
    print("  - Pressure boundary condition at eta=0.5")
else:
    print("\n*** TEST 2 FAILED: Fundamental adiabatic problem ***")
    print("\nThis is the SIMPLEST test - no waves, no geometry!")
    
    if not rho_pass:
        print("\n  Issue: Density not following exponential growth")
        print("    -> Mass conservation broken")
        print("    -> Check continuity equation implementation")
    
    if not p_pass:
        print("\n  Issue: Pressure not following p ~ rho^gamma")
        print("    -> EOS coupling broken")
        print("    -> Check how pressure is computed from density and energy")
    
    if not S_pass:
        print("\n  Issue: Isentropic invariant drifting")
        print("    -> Energy not conserved")
        print("    -> Check energy equation update")
    
    if not spatial_pass:
        print("\n  Issue: Spatial variation appearing")
        print("    -> Should be perfectly uniform!")
        print("    -> Numerical instability or boundary condition issue")
    
    if not gamma_pass and np.isfinite(gamma_error):
        if gamma_eff_mean < 1.2:
            print("\n  Issue: gamma_eff ~ 1.0 (ISOTHERMAL behavior)")
            print("    -> You are NOT running adiabatic compression")
            print("    -> Temperature being held constant")
            print("    -> Check if gamma is being used in EOS")

print("\n" + "=" * 70)
