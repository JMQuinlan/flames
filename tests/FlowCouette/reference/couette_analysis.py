"""
Couette Flow Analysis Script
Extracts 2D AMReX data and compares with analytical solution for viscous shear flow

CONFIGURATION (modify these as needed):
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

case_name = 'Couette_Flow'
amrex_output_dir = r'../../../bin/tests/FlowCouette/output'

# Geometry (from input file)
y_min = 0.0      # m
y_max = 3.0      # m
x_min = 0.0      # m
x_max = 0.1875   # m

# Physical parameters (from input file)
mu0 = 1.0        # Viscosity phase 0
mu1 = 10.0       # Viscosity phase 1
rho = 100.0      # Density (both phases)

# Initial conditions
U_top = 0.1      # m/s - velocity at top (y > 1.5)
U_bottom = 0.0   # m/s - velocity at bottom

# Interface positions (from eta IC)
y_interface_lower = 0.5   # m
y_interface_upper = 2.5   # m

# Slice location for 1D profile extraction
x_slice = 0.09375  # Middle of x-domain

print("="*70)
print("COUETTE FLOW ANALYSIS")
print("="*70)
print("\nProblem Setup:")
print(f"  Domain: x = [{x_min}, {x_max}] m, y = [{y_min}, {y_max}] m")
print(f"  Viscosities: mu0 = {mu0}, mu1 = {mu1}")
print(f"  Density: rho = {rho} kg/m^3")
print(f"  Top plate velocity: U = {U_top} m/s (y > 1.5 m)")
print(f"  Bottom plate velocity: U = {U_bottom} m/s")
print(f"  Interface positions: y = {y_interface_lower} m and y = {y_interface_upper} m")

# ============================================================================
# LOAD AMREX DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING AMREX OUTPUT")
print("="*70)

# Find plot files
plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and item.endswith('cell'):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No *cell directories found in {amrex_output_dir}")
    exit(1)

plot_files.sort()
last_plot = plot_files[-1]

print(f"\nFound {len(plot_files)} plot files")
print(f"Using LAST timestep only: {os.path.basename(last_plot)}")

# Load dataset
ds = yt.load(last_plot)
sim_time = float(ds.current_time)
print(f"Simulation time: {sim_time:.3f} s")

# ============================================================================
# EXTRACT 1D VELOCITY PROFILE
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING VELOCITY PROFILE")
print("="*70)

# Create vertical line at x = x_slice
ray_start = ds.arr([x_slice, y_min, 0.0], 'code_length')
ray_end = ds.arr([x_slice, y_max, 0.0], 'code_length')
ray = ds.ray(ray_start, ray_end)

# Sort by y coordinate
sort_indices = np.argsort(ray['y'])
y_numerical = np.array(ray['y'][sort_indices])
u_numerical = np.array(ray['velocityx'][sort_indices])
v_numerical = np.array(ray['velocityy'][sort_indices])

# Try to extract eta and viscosity
try:
    eta_numerical = np.array(ray['eta'][sort_indices])
    has_eta = True
except:
    has_eta = False
    print("  Warning: eta field not available")

try:
    mu_numerical = np.array(ray['mu'][sort_indices])
    has_mu = True
except:
    has_mu = False
    print("  Warning: mu field not available")

print(f"\nExtracted {len(y_numerical)} points along x = {x_slice} m")
print(f"  u velocity range: [{np.min(u_numerical):.6f}, {np.max(u_numerical):.6f}] m/s")
print(f"  v velocity range: [{np.min(v_numerical):.6e}, {np.max(v_numerical):.6e}] m/s")

# ============================================================================
# ANALYTICAL SOLUTION
# ============================================================================

print("\n" + "="*70)
print("COMPUTING ANALYTICAL SOLUTION")
print("="*70)

# For Couette flow between parallel plates with different viscosities
# The analytical solution depends on whether flow has reached steady state

# Create fine grid for analytical solution
y_analytical = np.linspace(y_min, y_max, 500)
u_analytical = np.zeros_like(y_analytical)

# Steady-state Couette flow with three regions
# Region 1: y < 0.5 (phase 0, mu = mu0)
# Region 2: 0.5 < y < 2.5 (phase 1, mu = mu1)
# Region 3: y > 2.5 (phase 0, mu = mu0)

# Boundary conditions:
# u(0) = 0 (bottom wall)
# u(3) = 0 (top wall, but initial condition has u = 0.1 for y > 1.5)
# Actually, from IC: u = 0.1 for y > 1.5

# For steady Couette flow: du/dy = constant in each region (for constant mu)
# Shear stress tau = mu * du/dy must be continuous across interfaces

# Simplified analytical solution for steady state:
# Assuming the flow eventually reaches a linear profile in each region

for i, y in enumerate(y_analytical):
    if y < y_interface_lower:
        # Region 1: Bottom layer (phase 0)
        u_analytical[i] = 0.0  # Adjust based on your expected steady state
    elif y < y_interface_upper:
        # Region 2: Middle layer (phase 1, higher viscosity)
        # Linear interpolation for now
        frac = (y - y_interface_lower) / (y_interface_upper - y_interface_lower)
        u_analytical[i] = 0.0 + frac * 0.05  # Approximate
    else:
        # Region 3: Top layer (phase 0)
        frac = (y - y_interface_upper) / (y_max - y_interface_upper)
        u_analytical[i] = 0.05 + frac * 0.05  # Approximate

print("\nNote: Analytical solution is approximate for transient Couette flow")
print("      with multiple viscosity regions and complex initial conditions.")
print(f"      At t = {sim_time:.3f} s, flow may still be developing.")

# Interpolate analytical solution to numerical grid for error calculation
u_analytical_interp = np.interp(y_numerical, y_analytical, u_analytical)

# ============================================================================
# COMPUTE THEORETICAL VISCOUS DIFFUSION TIME
# ============================================================================

# Characteristic diffusion time: t_diff ~ L^2 / nu
# where nu = mu / rho is kinematic viscosity

nu0 = mu0 / rho
nu1 = mu1 / rho
L_char = y_max - y_min

t_diff_0 = L_char**2 / nu0
t_diff_1 = L_char**2 / nu1

print(f"\nViscous diffusion timescales:")
print(f"  Phase 0 (mu={mu0}): t_diff = {t_diff_0:.2f} s")
print(f"  Phase 1 (mu={mu1}): t_diff = {t_diff_1:.2f} s")
print(f"  Current time / t_diff_0 = {sim_time/t_diff_0:.3f}")

# ============================================================================
# COMPUTE ERROR METRICS
# ============================================================================

print("\n" + "="*70)
print("COMPUTING ERROR METRICS")
print("="*70)

# Absolute error
error_abs = u_numerical - u_analytical_interp

# Relative error (avoid division by zero)
u_analytical_safe = np.where(np.abs(u_analytical_interp) < 1e-10, 1e-10, u_analytical_interp)
error_rel = error_abs / u_analytical_safe

# L2 and Linf norms
L2_error = np.linalg.norm(error_abs)
Linf_error = np.max(np.abs(error_abs))

print(f"\nError Metrics:")
print(f"  L2 error:   {L2_error:.6e}")
print(f"  Linf error: {Linf_error:.6e}")
print(f"  Mean absolute error: {np.mean(np.abs(error_abs)):.6e}")
print(f"  RMS error: {np.sqrt(np.mean(error_abs**2)):.6e}")

# ============================================================================
# CREATE PLOTS
# ============================================================================

print("\n" + "="*70)
print("CREATING PLOTS")
print("="*70)

os.makedirs('./Images', exist_ok=True)

# ============================================================================
# PLOT 1: Velocity Profile Comparison
# ============================================================================

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

ax1.set_title(f'Couette Flow Velocity Profile at t = {sim_time:.3f} s', 
              fontsize=14, fontweight='bold')
ax1.plot(u_analytical, y_analytical, 'b-', linewidth=2, 
         label='Analytical (Approximate)', zorder=1)
ax1.plot(u_numerical, y_numerical, 'r--', linewidth=1.5, 
         label='Numerical', zorder=2, alpha=0.7)
ax1.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
            label='Interface')
ax1.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax1.set_xlabel('Velocity u (m/s)', fontsize=12)
ax1.set_ylabel('Height y (m)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([y_min, y_max])

plt.tight_layout()
output_file = f'./Images/{case_name}_velocity_profile'
fig1.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig1.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_file}.png and .eps")
plt.close(fig1)

# ============================================================================
# PLOT 2: Shear Rate Profile
# ============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))

du_dy_numerical = np.gradient(u_numerical, y_numerical)
ax2.plot(du_dy_numerical, y_numerical, 'r-', linewidth=2, label='du/dy (Numerical)')
ax2.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
            label='Interface')
ax2.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax2.set_xlabel('Velocity Gradient du/dy (1/s)', fontsize=12)
ax2.set_ylabel('Height y (m)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([y_min, y_max])
ax2.set_title('Shear Rate Profile', fontsize=14, fontweight='bold')

plt.tight_layout()
output_file = f'./Images/{case_name}_shear_rate'
fig2.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig2.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_file}.png and .eps")
plt.close(fig2)

# ============================================================================
# PLOT 3: Phase Fraction (if available)
# ============================================================================

if has_eta:
    fig3, ax3 = plt.subplots(1, 1, figsize=(10, 8))
    
    ax3.plot(eta_numerical, y_numerical, 'g-', linewidth=2, label='eta (Phase 0 fraction)')
    ax3.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
                label='Interface')
    ax3.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Phase Fraction eta', fontsize=12)
    ax3.set_ylabel('Height y (m)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([y_min, y_max])
    ax3.set_xlim([-0.1, 1.1])
    ax3.set_title('Phase Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = f'./Images/{case_name}_phase_fraction'
    fig3.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
    fig3.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
    print(f"Saved: {output_file}.png and .eps")
    plt.close(fig3)

# ============================================================================
# PLOT 4: Absolute Error (Linear Scale)
# ============================================================================

fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))

ax4.plot(error_abs, y_numerical, 'k-', linewidth=2, label='Absolute Error')
ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax4.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
            label='Interface')
ax4.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax4.set_xlabel('Error (u_numerical - u_analytical) [m/s]', fontsize=12)
ax4.set_ylabel('Height y (m)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([y_min, y_max])
ax4.set_title(f'Absolute Error (L2={L2_error:.3e}, Linf={Linf_error:.3e})', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
output_file = f'./Images/{case_name}_error_absolute'
fig4.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig4.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_file}.png and .eps")
plt.close(fig4)

# ============================================================================
# PLOT 5: Absolute Error (Log Scale)
# ============================================================================

fig5, ax5 = plt.subplots(1, 1, figsize=(10, 8))

# Use absolute value for log scale
error_abs_log = np.abs(error_abs)
# Replace zeros with small value for log plotting
error_abs_log = np.where(error_abs_log < 1e-12, 1e-12, error_abs_log)

ax5.semilogx(error_abs_log, y_numerical, 'k-', linewidth=2, label='|Absolute Error|')
ax5.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
            label='Interface')
ax5.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax5.set_xlabel('|Error| (log scale) [m/s]', fontsize=12)
ax5.set_ylabel('Height y (m)', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, which='both')
ax5.set_ylim([y_min, y_max])
ax5.set_title(f'Absolute Error - Log Scale (L2={L2_error:.3e}, Linf={Linf_error:.3e})', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
output_file = f'./Images/{case_name}_error_absolute_log'
fig5.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig5.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_file}.png and .eps")
plt.close(fig5)

# ============================================================================
# PLOT 6: Combined Error Metrics
# ============================================================================

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(16, 8))

# Left: Absolute error (linear)
ax6a.plot(error_abs, y_numerical, 'k-', linewidth=2, label='Absolute Error')
ax6a.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax6a.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
             label='Interface')
ax6a.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax6a.set_xlabel('Error (u_num - u_ana) [m/s]', fontsize=12)
ax6a.set_ylabel('Height y (m)', fontsize=12)
ax6a.legend(fontsize=10)
ax6a.grid(True, alpha=0.3)
ax6a.set_ylim([y_min, y_max])
ax6a.set_title('Absolute Error (Linear Scale)', fontsize=12, fontweight='bold')

# Right: Absolute error (log)
ax6b.semilogx(error_abs_log, y_numerical, 'k-', linewidth=2, label='|Absolute Error|')
ax6b.axhline(y=y_interface_lower, color='k', linestyle=':', alpha=0.5, 
             label='Interface')
ax6b.axhline(y=y_interface_upper, color='k', linestyle=':', alpha=0.5)
ax6b.set_xlabel('|Error| (log scale) [m/s]', fontsize=12)
ax6b.set_ylabel('Height y (m)', fontsize=12)
ax6b.legend(fontsize=10)
ax6b.grid(True, alpha=0.3, which='both')
ax6b.set_ylim([y_min, y_max])
ax6b.set_title('Absolute Error (Log Scale)', fontsize=12, fontweight='bold')

fig6.suptitle(f'Error Analysis at t={sim_time:.3f}s (L2={L2_error:.3e}, Linf={Linf_error:.3e})', 
              fontsize=14, fontweight='bold')

plt.tight_layout()
output_file = f'./Images/{case_name}_error_combined'
fig6.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig6.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_file}.png and .eps")
plt.close(fig6)

# ============================================================================
# ANALYSIS SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nVelocity Statistics:")
print(f"  Maximum u velocity: {np.max(u_numerical):.6f} m/s")
print(f"  Minimum u velocity: {np.min(u_numerical):.6f} m/s")
print(f"  Mean u velocity: {np.mean(u_numerical):.6f} m/s")

print(f"\nShear Rate Statistics:")
print(f"  Maximum du/dy: {np.max(du_dy_numerical):.6f} 1/s")
print(f"  Minimum du/dy: {np.min(du_dy_numerical):.6f} 1/s")
print(f"  Mean du/dy: {np.mean(du_dy_numerical):.6f} 1/s")

print(f"\nError Statistics:")
print(f"  L2 norm: {L2_error:.6e}")
print(f"  Linf norm: {Linf_error:.6e}")
print(f"  Mean absolute error: {np.mean(np.abs(error_abs)):.6e}")
print(f"  RMS error: {np.sqrt(np.mean(error_abs**2)):.6e}")

# Check for flow development
if sim_time < 0.1 * t_diff_0:
    print(f"\nFlow Status: EARLY TRANSIENT")
    print(f"  Simulation time is {sim_time/t_diff_0*100:.1f}% of diffusion time")
    print(f"  Flow is still developing")
elif sim_time < t_diff_0:
    print(f"\nFlow Status: DEVELOPING")
    print(f"  Simulation time is {sim_time/t_diff_0*100:.1f}% of diffusion time")
    print(f"  Flow approaching steady state")
else:
    print(f"\nFlow Status: NEAR STEADY STATE")
    print(f"  Simulation time exceeds characteristic diffusion time")

# Check mass conservation (v should be small)
print(f"\nMass Conservation Check:")
print(f"  Max |v| velocity: {np.max(np.abs(v_numerical)):.6e} m/s")
print(f"  Mean |v| velocity: {np.mean(np.abs(v_numerical)):.6e} m/s")
if np.max(np.abs(v_numerical)) < 1e-6:
    print(f"  Status: GOOD (v << u)")
else:
    print(f"  Status: CHECK (v may be significant)")

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print(f"\nAll plots saved to ./Images/ directory:")
print(f"  - {case_name}_velocity_profile.png/.eps")
print(f"  - {case_name}_shear_rate.png/.eps")
if has_eta:
    print(f"  - {case_name}_phase_fraction.png/.eps")
print(f"  - {case_name}_error_absolute.png/.eps")
print(f"  - {case_name}_error_absolute_log.png/.eps")
print(f"  - {case_name}_error_combined.png/.eps")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nNotes:")
print("  - Couette flow with multiple viscosity layers")
print("  - Analytical solution is approximate for transient case")
print("  - Check if simulation time is sufficient for steady state")
print("  - Shear stress should be continuous across interfaces")
print("  - Error metrics computed against approximate analytical solution")
