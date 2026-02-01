import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import h5py
import sys

# Load hydro2 data from HDF5
hydro2_case_name = 'LargeBubbleRPE_2_Bar.GammaHigh.old.20250915191918'
hydro2_hdf5_file = f'./{hydro2_case_name}_tracking.hdf5'

# Parameters for water at 25C
rho_L = 997           # Density of liquid [kg/m^3]
mu_L = 0.001          # Dynamic viscosity of liquid [Pa.s] 
S = 72.8              # Surface tension [N/m] SCALLED
p_v = 3169            # Vapor pressure [Pa] at 25C
nu_L = mu_L / rho_L   # Kinematic viscosity [m^2/s]
gamma = 1.4           # Adiabatic index

# Initial conditions
p_inf = 1.0e5         # Pa, pressure far in the domain
p_B0 = 2.0e5          # Pa, initial bubble pressure
R0 = 0.01             # Initial Radius [m]
R_dot0 = 0            # Initial velocity [m/s]
y0 = [R0, R_dot0]

# Domain boundary (CRITICAL for 2D RPE)
r_inf = 10.0 * R0     # Far-field boundary location

# Time span for integration
t_span = (0, 0.05)
t_eval = np.linspace(*t_span, 10000)

# Acoustic Driving Force
Amp = 0.0             # Amplitude of pressure forcing [Pa]
Freq = 1.0            # Frequency of pressure forcing [Hz]

def P_force(Amp, Freq, t):
    """Acoustic driving pressure"""
    if Amp == 0.0:
        return 0.0
    else:
        return Amp * np.sin(2 * np.pi * Freq * t)

def rp_equation_2d_chen(t, y):
    """
    Chen's 2D Cylindrical RPE (Equation 3.7 from Chen 2010)
    """
    R, R_dot = y
    
    # Safety checks
    if R <= 1e-12:
        return [0, 0]
    
    if R >= 0.9 * r_inf:
        return [0, 0]
    
    # Internal bubble pressure (2D polytropic gas law)
    p_B = p_v + (p_B0 - p_v) * (R0 / R)**(2 * gamma)
    
    # External pressure with optional acoustic forcing
    p_ext = p_inf + P_force(Amp, Freq, t)
    
    # Pressure difference normalized by rho_L
    dP = (p_B - p_ext) / rho_L
    
    # Geometric logarithmic factor
    ln_factor = np.log(r_inf / R)
    if ln_factor < 1e-6:
        ln_factor = 1e-6
    
    # Viscous damping term
    viscous = (2 * mu_L / (rho_L * R)) * R_dot
    
    # Surface tension term
    surface = S / (rho_L * R)
    
    # Kinetic energy term
    kinetic = 0.5 * R_dot**2
    
    # Chen's 2D cylindrical RPE acceleration
    R_ddot = (dP - viscous + surface - kinetic) / (R * ln_factor)
    
    return [R_dot, R_ddot]

# Solve the ODE
print("Solving 2D Cylindrical RPE...")
sol = solve_ivp(rp_equation_2d_chen, t_span, y0, t_eval=t_eval, 
                method='RK45', rtol=1e-9, atol=1e-11)

if not sol.success:
    print(f"Warning: Integration failed - {sol.message}")

# Extract results
radius_analytical = sol.y[0]
velocity_analytical = sol.y[1]
time_analytical = sol.t

# Calculate equilibrium radius
def equilibrium_pressure_balance_2d(R_eq):
    p_B_eq = p_v + (p_B0 - p_v) * (R0 / R_eq)**(2 * gamma)
    p_ext_eq = p_inf + S / R_eq
    return p_B_eq - p_ext_eq

try:
    R_eq = fsolve(equilibrium_pressure_balance_2d, R0)[0]
    equilibrium_found = True
except:
    R_eq = R0
    equilibrium_found = False



print("\nLoading hydro2 data from HDF5...")
try:
    with h5py.File(hydro2_hdf5_file, 'r') as f:
        grp = f[hydro2_case_name]
        hydro2_times = grp['time'][:]
        hydro2_radius = grp['interface_position'][:]
        hydro2_velocity = grp['interface_velocity'][:]
        hydro2_pressure = grp['origin_pressure'][:]
    print(f"Loaded {len(hydro2_times)} hydro2 timesteps")
    
    # Interpolate hydro2 data to analytical time points
    hydro2_radius_interp = np.interp(time_analytical, hydro2_times, hydro2_radius)
    hydro2_velocity_interp = np.interp(time_analytical, hydro2_times, hydro2_velocity)
    hydro2_pressure_interp = np.interp(time_analytical, hydro2_times, hydro2_pressure)
    
    # Calculate error
    radius_error = np.abs(hydro2_radius_interp - radius_analytical)
    relative_error = radius_error / radius_analytical * 100
    
    hydro2_loaded = True
except Exception as e:
    print(f"Warning: Could not load hydro2 data: {e}")
    print("Continuing with analytical solution only")
    hydro2_loaded = False

# Calculate pressure for analytical solution
p_B_analytical = p_v + (p_B0 - p_v) * (R0 / radius_analytical)**(2 * gamma)

# Print analysis
print(f"\n{'='*60}")
print(f"2D CYLINDRICAL RPE - CHEN 2010 FORMULATION")
print(f"{'='*60}")
print(f"\nPhysical Parameters:")
print(f"  Liquid density rho_L: {rho_L} kg/m^3")
print(f"  Dynamic viscosity mu_L: {mu_L*1e3:.3f} mPa*s")
print(f"  Surface tension S: {S*1e3:.3f} mN/m")
print(f"  Vapor pressure p_v: {p_v} Pa")
print(f"  Adiabatic index gamma: {gamma}")
print(f"\nInitial Conditions:")
print(f"  Initial radius R0: {R0*1e3:.6f} mm")
print(f"  Initial pressure p_B0: {p_B0/1e3:.3f} kPa")
print(f"  Far-field pressure p_inf: {p_inf/1e3:.3f} kPa")
print(f"  Domain boundary r_inf: {r_inf*1e3:.3f} mm")

if equilibrium_found:
    print(f"\nEquilibrium Analysis:")
    print(f"  Calculated R_eq: {R_eq*1e3:.6f} mm")
    print(f"  R_eq/R0: {R_eq/R0:.6f}")

print(f"\nSimulation Results:")
print(f"  Final radius: {radius_analytical[-1]*1e3:.6f} mm")
print(f"  Max radius: {np.max(radius_analytical)*1e3:.6f} mm")
print(f"  Min radius: {np.min(radius_analytical)*1e3:.6f} mm")

# ============================================================
# PLOT 1: Multi-subplot figure
# ============================================================
print("\nGenerating multi-subplot figure...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Radius vs Time
ax1 = axes[0, 0]
ax1.plot(time_analytical * 1000, radius_analytical * 1000, 'b-', linewidth=2, label='Analytical (Chen 2D)')
if hydro2_loaded:
    ax1.plot(time_analytical * 1000, hydro2_radius_interp * 1000, 'r--', linewidth=2, label='Hydro2')
if equilibrium_found:
    ax1.axhline(y=R_eq*1000, color='g', linestyle=':', linewidth=1.5, 
                label=f'R_eq = {R_eq*1000:.3f} mm')
ax1.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='R0')
ax1.set_xlabel('Time (ms)', fontsize=11)
ax1.set_ylabel('Radius (mm)', fontsize=11)
ax1.set_title('Bubble Radius vs Time', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Velocity vs Time
ax2 = axes[0, 1]
ax2.plot(time_analytical * 1000, velocity_analytical, 'b-', linewidth=2, label='Analytical')
if hydro2_loaded:
    ax2.plot(time_analytical * 1000, hydro2_velocity_interp, 'r--', linewidth=2, label='Hydro2')
ax2.set_xlabel('Time (ms)', fontsize=11)
ax2.set_ylabel('Velocity (m/s)', fontsize=11)
ax2.set_title('Bubble Wall Velocity vs Time', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Subplot 3: Pressure vs Time
ax3 = axes[1, 0]
ax3.plot(time_analytical * 1000, p_B_analytical / 1e5, 'b-', linewidth=2, label='Analytical')
if hydro2_loaded:
    ax3.plot(time_analytical * 1000, hydro2_pressure_interp / 1e5, 'r--', linewidth=2, label='Hydro2')
ax3.axhline(y=p_inf/1e5, color='gray', linestyle='--', linewidth=1, label='p_inf')
ax3.set_xlabel('Time (ms)', fontsize=11)
ax3.set_ylabel('Pressure (bar)', fontsize=11)
ax3.set_title('Pressure vs Time', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Subplot 4: Phase Space
ax4 = axes[1, 1]
ax4.plot(radius_analytical * 1000, velocity_analytical, 'b-', linewidth=2, label='Analytical')
if hydro2_loaded:
    ax4.plot(hydro2_radius_interp * 1000, hydro2_velocity_interp, 'r--', linewidth=2, label='Hydro2')
ax4.set_xlabel('Radius (mm)', fontsize=11)
ax4.set_ylabel('Velocity (m/s)', fontsize=11)
ax4.set_title('Phase Space (R vs U)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save multi-subplot figure
multi_png = './cylindrical_rpe_comparison_multi.png'
multi_eps = './cylindrical_rpe_comparison_multi.eps'
plt.savefig(multi_png, dpi=300, bbox_inches='tight')
plt.savefig(multi_eps, format='eps', bbox_inches='tight')
print(f"Saved: {multi_png}")
print(f"Saved: {multi_eps}")
plt.show()

# ============================================================
# PLOT 2: Standalone Radius Plot
# ============================================================
print("\nGenerating standalone radius plot...")

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(time_analytical * 1000, radius_analytical * 1000, 'b-', linewidth=2.5, label='Analytical (Chen 2D)')
if hydro2_loaded:
    ax.plot(time_analytical * 1000, hydro2_radius_interp * 1000, 'r--', linewidth=2.5, label='Hydro2')
if equilibrium_found:
    ax.axhline(y=R_eq*1000, color='g', linestyle=':', linewidth=2, 
               label=f'R_eq = {R_eq*1000:.3f} mm')
ax.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='R0')
ax.set_xlabel('Time (ms)', fontsize=13)
ax.set_ylabel('Radius (mm)', fontsize=13)
ax.set_title('Bubble Radius vs Time - 2D Cylindrical RPE', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

standalone_png = './cylindrical_rpe_radius_standalone.png'
standalone_eps = './cylindrical_rpe_radius_standalone.eps'
plt.savefig(standalone_png, dpi=300, bbox_inches='tight')
plt.savefig(standalone_eps, format='eps', bbox_inches='tight')
print(f"Saved: {standalone_png}")
print(f"Saved: {standalone_eps}")
plt.show()

# ============================================================
# PLOT 3: Semilogy Error Plot
# ============================================================
if hydro2_loaded:
    print("\nGenerating error plot...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.semilogy(time_analytical * 1000, radius_error * 1000, 'r-', linewidth=2.5)
    ax.set_xlabel('Time (ms)', fontsize=13)
    ax.set_ylabel('Absolute Error (mm)', fontsize=13)
    ax.set_title('Radius Error: Hydro2 vs Analytical (Semilogy)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Error statistics
    max_error = np.max(radius_error * 1000)
    mean_error = np.mean(radius_error * 1000)
    textstr = f'Max Error: {max_error:.4f} mm\nMean Error: {mean_error:.4f} mm'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    
    error_png = './cylindrical_rpe_error.png'
    error_eps = './cylindrical_rpe_error.eps'
    plt.savefig(error_png, dpi=300, bbox_inches='tight')
    plt.savefig(error_eps, format='eps', bbox_inches='tight')
    print(f"Saved: {error_png}")
    print(f"Saved: {error_eps}")
    plt.show()
    
    print("\n" + "="*60)
    print("ERROR STATISTICS")
    print("="*60)
    print(f"Maximum absolute error: {max_error:.6f} mm")
    print(f"Mean absolute error: {mean_error:.6f} mm")
    print(f"Maximum relative error: {np.max(relative_error):.4f}%")
    print(f"Mean relative error: {np.mean(relative_error):.4f}%")

print("\nComplete!")
