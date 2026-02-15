"""
===============================================================================
TRANSIENT 1D STEFAN FLOW - ANALYTICAL SOLUTION VISUALIZATION
===============================================================================

PURPOSE:
    Visualize the transient evolution of 1D Stefan flow mass fraction profiles.
    Plot mass fraction vs position with time represented by color.
    
FEATURES:
    - Time-marching analytical solution
    - Moving interface tracking
    - Mass fraction profiles at multiple time steps
    - Color-coded time evolution
    - Verification of Stefan flow functions
    
PHYSICS:
    Stefan flow for 1D planar evaporation with moving interface
    - Interface velocity: v_s = (D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    - Mass flux: m_dot = rho * v_s
    - Mass fraction profile with Stefan flow correction
    
OUTPUTS:
    - Mass fraction profiles colored by time
    - Interface position vs time
    - Combined visualization

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Physical parameters (matching your input file)
density0 = 1.0                   # Gas density [kg/m^3]
density1 = 10.0                # Liquid density [kg/m^3]
D_v = 1.0e-4                     # Binary diffusion coefficient [m^2/s]
Y_surface = 1.00                 # Mass fraction at liquid surface
Y_infinity = 0.00                 # Mass fraction far from interface

# Domain properties
x_min = 0.0                      # Domain start [m]
x_max = 1.0                      # Domain end [m]
x_interface_initial = 0.5        # Initial interface position [m]

# Time parameters
t_start = 0.0                    # Start time [s]
t_end = 100.0                    # End time [s] - adjust as needed
n_timesteps = 20                 # Number of time snapshots

# Spatial discretization
n_points = 500                   # Number of spatial points

# Output settings
output_folder = './Stefan_Flow_Analytical_Test'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.0

# ============================================================================
# ANALYTICAL SOLUTION FUNCTIONS (SAME AS OTHER SCRIPTS)
# ============================================================================

def stefan_velocity(D_v, delta, Y_s, Y_inf):
    """
    Calculate Stefan velocity at the interface
    v_s = (D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    if Y_s >= 1.0 or Y_inf >= 1.0 or delta < 1e-12:
        return 0.0
    return (D_v / delta) * np.log((1.0 - Y_inf) / (1.0 - Y_s))

def stefan_mass_flux(rho, D_v, delta, Y_s, Y_inf):
    """
    Calculate mass flux at the interface
    m_dot = rho * v_s = (rho * D_v / delta) * ln[(1 - Y_inf) / (1 - Y_s)]
    """
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    return rho * v_s

def mass_fraction_profile(x, x_interface, D_v, Y_s, Y_inf, rho, L_domain):
    """
    Analytical mass fraction profile with Stefan flow
    Y(x) = Y_inf + (Y_s - Y_inf) * [exp(Pe*(x-x_i)/delta) - 1] / [exp(Pe) - 1]
    where delta = L - x_interface, Pe = v_s * delta / D_v
    """
    delta = L_domain - x_interface
    if delta < 1e-12:
        return np.full_like(x, Y_inf)
    
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    Pe = v_s * delta / D_v
    
    # Normalized distance from interface
    xi = (x - x_interface) / delta
    
    # Mass fraction profile
    if abs(Pe) < 1e-6:
        # Linear profile for small Peclet number
        Y = Y_s + (Y_inf - Y_s) * xi
    else:
        Y = Y_inf + (Y_s - Y_inf) * (np.exp(Pe * xi) - 1.0) / (np.exp(Pe) - 1.0)
    
    # Apply only in gas region (x > x_interface)
    Y = np.where(x > x_interface, Y, Y_s)
    
    return Y

# ============================================================================
# TIME MARCHING SIMULATION
# ============================================================================

print("=" * 70)
print("TRANSIENT 1D STEFAN FLOW - ANALYTICAL SOLUTION")
print("=" * 70)

# Create spatial grid
x = np.linspace(x_min, x_max, n_points)
L_domain = x_max - x_min

# Time array
times = np.linspace(t_start, t_end, n_timesteps)
dt = times[1] - times[0] if len(times) > 1 else 1.0

# Storage arrays
interface_positions = []
mass_fraction_profiles = []
stefan_velocities_list = []
mass_fluxes_list = []

# Initial conditions
x_interface = x_interface_initial

print(f"\nSimulation Parameters:")
print(f"  Domain: [{x_min}, {x_max}] m")
print(f"  Initial interface: {x_interface_initial} m")
print(f"  Time range: [{t_start}, {t_end}] s")
print(f"  Number of timesteps: {n_timesteps}")
print(f"  Gas density: {density0} kg/m^3")
print(f"  Diffusion coefficient: {D_v} m^2/s")
print(f"  Y_surface: {Y_surface}")
print(f"  Y_infinity: {Y_infinity}")

print("\n" + "=" * 70)
print("COMPUTING TRANSIENT SOLUTION")
print("=" * 70)

for i, t in enumerate(times):
    # Calculate mass fraction profile at current interface position
    Y_profile = mass_fraction_profile(x, x_interface, D_v, Y_surface, Y_infinity, 
                                      density0, L_domain)
    
    # Calculate Stefan velocity and mass flux
    delta = L_domain - x_interface
    v_s = stefan_velocity(D_v, delta, Y_surface, Y_infinity)
    m_dot = stefan_mass_flux(density0, D_v, delta, Y_surface, Y_infinity)
    
    # Store results
    interface_positions.append(x_interface)
    mass_fraction_profiles.append(Y_profile)
    stefan_velocities_list.append(v_s)
    mass_fluxes_list.append(m_dot)
    
    # Update interface position for next timestep (explicit Euler)
    if i < len(times) - 1:
        x_interface = x_interface + v_s * dt
        
        # Check if interface reaches boundary
        if x_interface >= x_max - 0.01:
            print(f"  Warning: Interface approaching boundary at t = {t:.2f} s")
            # Truncate arrays
            times = times[:i+1]
            break
    
    if (i + 1) % 5 == 0 or i == 0 or i == len(times) - 1:
        print(f"  t = {t:.2f} s: x_int = {x_interface:.6f} m, v_s = {v_s:.6e} m/s, m_dot = {m_dot:.6e} kg/m^2/s")

interface_positions = np.array(interface_positions)
mass_fraction_profiles = np.array(mass_fraction_profiles)
stefan_velocities_list = np.array(stefan_velocities_list)
mass_fluxes_list = np.array(mass_fluxes_list)

print(f"\nSimulation complete!")
print(f"  Interface displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")
print(f"  Average Stefan velocity: {np.mean(stefan_velocities_list):.6e} m/s")
print(f"  Average mass flux: {np.mean(mass_fluxes_list):.6e} kg/m^2/s")

# ============================================================================
# PLOT 1: MASS FRACTION PROFILES WITH TIME COLORBAR
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(12, 8))

# Create colormap for time
cmap = cm.viridis
norm = Normalize(vmin=times[0], vmax=times[-1])
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Plot mass fraction profiles
for i, t in enumerate(times):
    color = cmap(norm(t))
    ax1.plot(x, mass_fraction_profiles[i], color=color, linewidth=LINE_WIDTH, 
             alpha=0.8, label=f't = {t:.2f} s' if i % 5 == 0 else '')
    
    # Mark interface position
    ax1.axvline(x=interface_positions[i], color=color, linestyle=':', 
                linewidth=1.0, alpha=0.5)

# Add colorbar
cbar = plt.colorbar(sm, ax=ax1, label='Time (s)')
cbar.ax.tick_params(labelsize=FONT_SIZE_TICK)
cbar.set_label('Time (s)', fontsize=FONT_SIZE_LABEL)

ax1.set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Mass Fraction Y', fontsize=FONT_SIZE_LABEL)
ax1.set_title('Transient 1D Stefan Flow: Mass Fraction Profiles\nColor represents time evolution', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
ax1.set_xlim([x_min, x_max])
ax1.set_ylim([Y_infinity - 0.1, Y_surface + 0.1])

plt.tight_layout()
plt.savefig(os.path.join(output_folder, '01_Mass_Fraction_Profiles_Transient.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Mass_Fraction_Profiles_Transient.eps'))
print("  Saved: 01_Mass_Fraction_Profiles_Transient.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: INTERFACE POSITION VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(times, interface_positions, 'b-', linewidth=LINE_WIDTH, 
         marker='o', markersize=6, label='Interface Position')

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('Interface Position vs Time\n1D Stefan Flow', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_Interface_Position_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Interface_Position_vs_Time.eps'))
print("  Saved: 02_Interface_Position_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: STEFAN VELOCITY VS TIME
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(times, stefan_velocities_list, 'g-', linewidth=LINE_WIDTH, 
         marker='s', markersize=6, label='Stefan Velocity')

ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Stefan Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('Stefan Velocity vs Time\n1D Stefan Flow', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax3.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax3.grid(True, alpha=0.3)
ax3.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '03_Stefan_Velocity_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '03_Stefan_Velocity_vs_Time.eps'))
print("  Saved: 03_Stefan_Velocity_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 4: MASS FLUX VS TIME
# ============================================================================

fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(times, mass_fluxes_list, 'r-', linewidth=LINE_WIDTH, 
         marker='^', markersize=6, label='Mass Flux')

ax4.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax4.set_title('Mass Flux vs Time\n1D Stefan Flow', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4.grid(True, alpha=0.3)
ax4.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '04_Mass_Flux_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Mass_Flux_vs_Time.eps'))
print("  Saved: 04_Mass_Flux_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: COMBINED VISUALIZATION (2x2 GRID)
# ============================================================================

fig5 = plt.figure(figsize=(16, 12))

# Subplot 1: Mass fraction profiles with colorbar
ax5a = plt.subplot(2, 2, 1)
for i, t in enumerate(times):
    color = cmap(norm(t))
    ax5a.plot(x, mass_fraction_profiles[i], color=color, linewidth=1.5, alpha=0.8)
    ax5a.axvline(x=interface_positions[i], color=color, linestyle=':', 
                 linewidth=0.8, alpha=0.5)
cbar1 = plt.colorbar(sm, ax=ax5a, label='Time (s)')
cbar1.ax.tick_params(labelsize=FONT_SIZE_TICK-2)
ax5a.set_xlabel('Position x (m)', fontsize=FONT_SIZE_LABEL-2)
ax5a.set_ylabel('Mass Fraction Y', fontsize=FONT_SIZE_LABEL-2)
ax5a.set_title('Mass Fraction Profiles', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5a.grid(True, alpha=0.3)
ax5a.tick_params(labelsize=FONT_SIZE_TICK-2)

# Subplot 2: Interface position
ax5b = plt.subplot(2, 2, 2)
ax5b.plot(times, interface_positions, 'b-', linewidth=LINE_WIDTH, marker='o', markersize=4)
ax5b.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL-2)
ax5b.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL-2)
ax5b.set_title('Interface Position', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5b.grid(True, alpha=0.3)
ax5b.tick_params(labelsize=FONT_SIZE_TICK-2)

# Subplot 3: Stefan velocity
ax5c = plt.subplot(2, 2, 3)
ax5c.plot(times, stefan_velocities_list, 'g-', linewidth=LINE_WIDTH, marker='s', markersize=4)
ax5c.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL-2)
ax5c.set_ylabel('Stefan Velocity (m/s)', fontsize=FONT_SIZE_LABEL-2)
ax5c.set_title('Stefan Velocity', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5c.grid(True, alpha=0.3)
ax5c.tick_params(labelsize=FONT_SIZE_TICK-2)

# Subplot 4: Mass flux
ax5d = plt.subplot(2, 2, 4)
ax5d.plot(times, mass_fluxes_list, 'r-', linewidth=LINE_WIDTH, marker='^', markersize=4)
ax5d.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL-2)
ax5d.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL-2)
ax5d.set_title('Mass Flux', fontsize=FONT_SIZE_TITLE-2, fontweight='bold')
ax5d.grid(True, alpha=0.3)
ax5d.tick_params(labelsize=FONT_SIZE_TICK-2)

fig5.suptitle('Transient 1D Stefan Flow - Complete Analysis', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])

plt.savefig(os.path.join(output_folder, '05_Combined_Analysis.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Combined_Analysis.eps'))
print("  Saved: 05_Combined_Analysis.png/.eps")
plt.close()

# ============================================================================
# SAVE DATA TO TEXT FILE
# ============================================================================

print("\n" + "=" * 70)
print("SAVING DATA TO TEXT FILE")
print("=" * 70)

data_file = os.path.join(output_folder, 'stefan_flow_analytical_data.txt')
header = "Time(s) Interface_Position(m) Stefan_Velocity(m/s) Mass_Flux(kg/m^2/s)"
data = np.column_stack((times, interface_positions, stefan_velocities_list, mass_fluxes_list))
np.savetxt(data_file, data, header=header, fmt='%.10e')
print(f"  Saved: stefan_flow_analytical_data.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Gas density: {density0} kg/m^3")
print(f"  Liquid density: {density1} kg/m^3")
print(f"  Diffusion coefficient: {D_v} m^2/s")
print(f"  Y_surface: {Y_surface}")
print(f"  Y_infinity: {Y_infinity}")

print(f"\nResults:")
print(f"  Time range: {times[0]:.2f} - {times[-1]:.2f} s")
print(f"  Initial interface: {interface_positions[0]:.6f} m")
print(f"  Final interface: {interface_positions[-1]:.6f} m")
print(f"  Total displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")
print(f"  Average Stefan velocity: {np.mean(stefan_velocities_list):.6e} m/s")
print(f"  Average mass flux: {np.mean(mass_fluxes_list):.6e} kg/m^2/s")

print(f"\nFiles generated:")
print(f"  - 01_Mass_Fraction_Profiles_Transient.png/.eps")
print(f"  - 02_Interface_Position_vs_Time.png/.eps")
print(f"  - 03_Stefan_Velocity_vs_Time.png/.eps")
print(f"  - 04_Mass_Flux_vs_Time.png/.eps")
print(f"  - 05_Combined_Analysis.png/.eps")
print(f"  - stefan_flow_analytical_data.txt")

print("\n" + "=" * 70)
print("VERIFICATION NOTES:")
print("=" * 70)
print("This script uses the SAME analytical functions as the post-processing script:")
print("  1. stefan_velocity(D_v, delta, Y_s, Y_inf)")
print("  2. stefan_mass_flux(rho, D_v, delta, Y_s, Y_inf)")
print("  3. mass_fraction_profile(x, x_interface, D_v, Y_s, Y_inf, rho, L_domain)")
print("\nYou can verify these functions are working correctly by:")
print("  - Checking that interface moves in positive x direction")
print("  - Verifying mass fraction is Y_s at interface, Y_inf at far-field")
print("  - Confirming Stefan velocity decreases as delta increases")
print("  - Ensuring mass flux matches m_dot = rho * v_s")
print("\n" + "=" * 70)
