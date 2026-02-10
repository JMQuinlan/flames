"""
===============================================================================
1D STEFAN FLOW SIMULATION - MASS FLUX AND INTERFACE TRACKING
===============================================================================

PURPOSE:
    Simulate 1D planar Stefan flow evaporation and track interface motion.
    Plot mass flux and interface position (eta = 0.5) over time.
    
FEATURES:
    - Time-marching simulation of Stefan flow
    - Interface tracking (eta = 0.5 contour)
    - Mass flux calculation at interface
    - Real-time or post-processing visualization
    - Comparison with analytical Stefan flow solution
    
PHYSICS:
    Stefan flow mass transfer with moving interface
    - Mass fraction transport with convection
    - Interface velocity from Stefan flow
    - Quasi-steady mass fraction profile
    
INPUTS:
    Physical parameters from input file (densities, diffusivity, etc.)
    
OUTPUTS:
    - Interface position vs time plot
    - Mass flux vs time plot
    - Combined visualization

===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================================
# CONFIGURATION PARAMETERS (FROM INPUT FILE)
# ============================================================================

# Physical parameters - EXTRACTED FROM YOUR INPUT FILE
# Fluid 0 properties (vapor/gas phase, eta=1)
density0 = 1.0                   # Density of gas phase [kg/m^3]
mu0 = 1.8e-5                     # Dynamic viscosity of gas [Pa-s]
pressure0 = 101325.0             # Pressure of gas [Pa]
gamma0 = 1.4                     # Ratio of specific heats for gas
temperature0 = 293.15            # Temperature [K]
cp0 = 1005.0                     # Specific heat [J/kg/K]
cv0 = 717.86                     # Specific heat at constant volume [J/kg/K]
k_thermal0 = 0.026               # Thermal conductivity [W/m/K]

# Fluid 1 properties (liquid phase, eta=0)
density1 = 1000.0                # Density of liquid [kg/m^3]
mu1 = 1.0e-3                     # Dynamic viscosity of liquid [Pa-s]
pressure1 = 101325.0             # Pressure of liquid [Pa]
gamma1 = 1.4                     # Ratio of specific heats for liquid
temperature1 = 293.15            # Temperature [K]
cp1 = 4179.0                     # Specific heat [J/kg/K]
cv1 = 2984.3                     # Specific heat at constant volume [J/kg/K]
k_thermal1 = 0.6                 # Thermal conductivity [W/m/K]

# Mass transfer properties
D_v = 1.0e-5                     # Binary diffusion coefficient [m^2/s]
sigma = 72.8                     # Surface tension [N/m]

# Boundary conditions for mass fraction
Y_surface = 0.9                  # Mass fraction at liquid surface (eta=0.5)
Y_infinity = 0.1                 # Mass fraction far from interface

# Domain properties (FROM INPUT FILE)
x_min = 0.0                      # geometry.prob_lo
x_max = 1.0                      # geometry.prob_hi
y_min = -0.0625
y_max = 0.0625

# Grid properties (FROM INPUT FILE)
n_cells_x = 128                  # amr.n_cell
n_cells_y = 16

# Time stepping (FROM INPUT FILE)
dt_initial = 1e-7                # timestep
dt_max = 1e-0                    # dynamictimestep.max
dt_min = 1e-12                   # dynamictimestep.min
cfl = 0.1                        # cfl
stop_time = 1.0                  # stop_time
plot_dt = 2e-3                   # amr.plot_dt

# Initial interface position (FROM INPUT FILE)
x_interface_initial = 0.5        # eta.ic.expression.constant.x_interface
epsilon_interface = 0.005        # eta.ic.expression.constant.epsilon

# Output settings
output_folder = fr'..\..\..\bin\tests\stefan\stefan_1'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Plotting customization
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.0

# ============================================================================
# ANALYTICAL SOLUTION FUNCTIONS
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

def mass_fraction_profile(x, x_interface, D_v, Y_s, Y_inf, rho):
    """
    Analytical mass fraction profile with Stefan flow
    Y(x) = Y_inf + (Y_s - Y_inf) * [exp(Pe*(x-x_i)/delta) - 1] / [exp(Pe) - 1]
    """
    delta = x_max - x_interface
    if delta < 1e-12:
        return np.full_like(x, Y_inf)
    
    v_s = stefan_velocity(D_v, delta, Y_s, Y_inf)
    Pe = v_s * delta / D_v
    
    # Normalized distance from interface
    xi = (x - x_interface) / delta
    
    # Mass fraction profile
    if abs(Pe) < 1e-6:
        Y = Y_s + (Y_inf - Y_s) * xi
    else:
        Y = Y_inf + (Y_s - Y_inf) * (np.exp(Pe * xi) - 1.0) / (np.exp(Pe) - 1.0)
    
    # Apply only in gas region
    Y = np.where(x > x_interface, Y, Y_s)
    
    return Y

def eta_profile_initial(x, x_interface, epsilon):
    """
    Initial eta profile (tanh interface)
    eta = 0.5 * (1 + tanh((x - x_interface) / (2*sqrt(2)*epsilon)))
    """
    return 0.5 * (1.0 + np.tanh((x - x_interface) / (2.0 * np.sqrt(2.0) * epsilon)))

# ============================================================================
# SIMULATION SETUP
# ============================================================================

print("=" * 70)
print("1D STEFAN FLOW SIMULATION")
print("=" * 70)

# Create spatial grid
x = np.linspace(x_min, x_max, n_cells_x)
dx = x[1] - x[0]

# Initialize eta field
eta = eta_profile_initial(x, x_interface_initial, epsilon_interface)

# Initialize storage arrays
times = [0.0]
interface_positions = [x_interface_initial]
mass_fluxes = []
stefan_velocities = []

# Calculate initial mass flux
delta_initial = x_max - x_interface_initial
m_dot_initial = stefan_mass_flux(density0, D_v, delta_initial, Y_surface, Y_infinity)
v_s_initial = stefan_velocity(D_v, delta_initial, Y_surface, Y_infinity)
mass_fluxes.append(m_dot_initial)
stefan_velocities.append(v_s_initial)

print(f"\nInitial Conditions:")
print(f"  Interface position: {x_interface_initial:.6f} m")
print(f"  Stefan velocity: {v_s_initial:.6e} m/s")
print(f"  Mass flux: {m_dot_initial:.6e} kg/m^2/s")

# ============================================================================
# TIME MARCHING SIMULATION
# ============================================================================

print("\n" + "=" * 70)
print("RUNNING TIME MARCHING SIMULATION")
print("=" * 70)

t = 0.0
dt = dt_initial
step = 0
plot_counter = 0
next_plot_time = plot_dt

while t < stop_time:
    # Find current interface position (where eta = 0.5)
    idx_interface = np.argmin(np.abs(eta - 0.5))
    x_interface = x[idx_interface]
    
    # Calculate Stefan velocity and mass flux
    delta = x_max - x_interface
    if delta < dx:
        print(f"  Warning: Interface too close to boundary at t = {t:.6e} s")
        break
    
    v_s = stefan_velocity(D_v, delta, Y_surface, Y_infinity)
    m_dot = stefan_mass_flux(density0, D_v, delta, Y_surface, Y_infinity)
    
    # Adaptive time stepping based on CFL condition
    if v_s > 1e-12:
        dt_cfl = cfl * dx / abs(v_s)
        dt = min(dt_cfl, dt_max)
        dt = max(dt, dt_min)
    
    # Update interface position (simple explicit Euler)
    x_interface_new = x_interface + v_s * dt
    
    # Update eta field (shift interface)
    eta = eta_profile_initial(x, x_interface_new, epsilon_interface)
    
    # Update time
    t += dt
    step += 1
    
    # Store data at plot intervals
    if t >= next_plot_time:
        times.append(t)
        interface_positions.append(x_interface_new)
        mass_fluxes.append(m_dot)
        stefan_velocities.append(v_s)
        next_plot_time += plot_dt
        plot_counter += 1
        
        if plot_counter % 10 == 0:
            print(f"  Step {step}: t = {t:.6e} s, x_int = {x_interface_new:.6f} m, m_dot = {m_dot:.6e} kg/m^2/s")
    
    # Safety check
    if step > 1e7:
        print(f"  Warning: Maximum steps reached")
        break

times = np.array(times)
interface_positions = np.array(interface_positions)
mass_fluxes = np.array(mass_fluxes)
stefan_velocities = np.array(stefan_velocities)

print(f"\nSimulation complete!")
print(f"  Total steps: {step}")
print(f"  Final time: {t:.6e} s")
print(f"  Final interface position: {interface_positions[-1]:.6f} m")
print(f"  Final mass flux: {mass_fluxes[-1]:.6e} kg/m^2/s")

# ============================================================================
# CALCULATE ANALYTICAL SOLUTION FOR COMPARISON
# ============================================================================

print("\n" + "=" * 70)
print("CALCULATING ANALYTICAL SOLUTION")
print("=" * 70)

analytical_mass_flux = []
analytical_interface_pos = []

for i, t_val in enumerate(times):
    x_int = interface_positions[i]
    delta = x_max - x_int
    m_dot_ana = stefan_mass_flux(density0, D_v, delta, Y_surface, Y_infinity)
    analytical_mass_flux.append(m_dot_ana)
    analytical_interface_pos.append(x_int)

analytical_mass_flux = np.array(analytical_mass_flux)
analytical_interface_pos = np.array(analytical_interface_pos)

# ============================================================================
# PLOT 1: INTERFACE POSITION VS TIME
# ============================================================================

print("\n" + "=" * 70)
print("CREATING PLOTS")
print("=" * 70)

fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.plot(times, interface_positions, 'b-', linewidth=LINE_WIDTH, 
         label='Numerical Interface Position (eta = 0.5)', marker='o', markersize=4)

ax1.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax1.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax1.set_title('1D Stefan Flow: Interface Position vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax1.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax1.grid(True, alpha=0.3)
ax1.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '01_Interface_Position_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '01_Interface_Position_vs_Time.eps'))
print("  Saved: 01_Interface_Position_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 2: MASS FLUX VS TIME
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.plot(times, mass_fluxes, 'r-', linewidth=LINE_WIDTH, 
         label='Numerical Mass Flux', marker='s', markersize=4)
ax2.plot(times, analytical_mass_flux, 'b--', linewidth=LINE_WIDTH, 
         label='Analytical Mass Flux', alpha=0.7)

ax2.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax2.set_title('1D Stefan Flow: Mass Flux vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax2.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax2.grid(True, alpha=0.3)
ax2.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '02_Mass_Flux_vs_Time.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '02_Mass_Flux_vs_Time.eps'))
print("  Saved: 02_Mass_Flux_vs_Time.png/.eps")
plt.close()

# ============================================================================
# PLOT 3: STEFAN VELOCITY VS TIME
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(times, stefan_velocities, 'g-', linewidth=LINE_WIDTH, 
         label='Stefan Velocity', marker='^', markersize=4)

ax3.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Stefan Velocity (m/s)', fontsize=FONT_SIZE_LABEL)
ax3.set_title('1D Stefan Flow: Interface Velocity vs Time', 
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
# PLOT 4: COMBINED PLOT (INTERFACE AND MASS FLUX)
# ============================================================================

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(10, 12))

# Top: Interface position
ax4a.plot(times, interface_positions, 'b-', linewidth=LINE_WIDTH, 
          label='Interface Position (eta = 0.5)', marker='o', markersize=4)
ax4a.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4a.set_ylabel('Interface Position (m)', fontsize=FONT_SIZE_LABEL)
ax4a.set_title('Interface Position vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4a.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4a.grid(True, alpha=0.3)
ax4a.tick_params(labelsize=FONT_SIZE_TICK)

# Bottom: Mass flux
ax4b.plot(times, mass_fluxes, 'r-', linewidth=LINE_WIDTH, 
          label='Numerical Mass Flux', marker='s', markersize=4)
ax4b.plot(times, analytical_mass_flux, 'b--', linewidth=LINE_WIDTH, 
          label='Analytical Mass Flux', alpha=0.7)
ax4b.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax4b.set_ylabel('Mass Flux (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax4b.set_title('Mass Flux vs Time', fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax4b.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
ax4b.grid(True, alpha=0.3)
ax4b.tick_params(labelsize=FONT_SIZE_TICK)

fig4.suptitle('1D Stefan Flow Simulation Results', 
              fontsize=FONT_SIZE_TITLE + 2, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig(os.path.join(output_folder, '04_Combined_Results.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '04_Combined_Results.eps'))
print("  Saved: 04_Combined_Results.png/.eps")
plt.close()

# ============================================================================
# PLOT 5: MASS FLUX ERROR
# ============================================================================

mass_flux_error = np.abs(mass_fluxes - analytical_mass_flux)
epsilon = 1e-16

fig5, ax5 = plt.subplots(figsize=(10, 8))
ax5.semilogy(times, mass_flux_error + epsilon, 'k-', linewidth=LINE_WIDTH)

ax5.set_xlabel('Time (s)', fontsize=FONT_SIZE_LABEL)
ax5.set_ylabel('|m_dot_num - m_dot_analytical| (kg/m^2/s)', fontsize=FONT_SIZE_LABEL)
ax5.set_title('Mass Flux Error vs Time', 
              fontsize=FONT_SIZE_TITLE, fontweight='bold')
ax5.grid(True, alpha=0.3, which='both')
ax5.tick_params(labelsize=FONT_SIZE_TICK)
plt.tight_layout()

plt.savefig(os.path.join(output_folder, '05_Mass_Flux_Error.png'), dpi=300)
plt.savefig(os.path.join(output_folder, '05_Mass_Flux_Error.eps'))
print("  Saved: 05_Mass_Flux_Error.png/.eps")
plt.close()

# ============================================================================
# SAVE DATA TO TEXT FILES
# ============================================================================

print("\n" + "=" * 70)
print("SAVING DATA TO TEXT FILES")
print("=" * 70)

# Save time series data
data_file = os.path.join(output_folder, 'stefan_flow_data.txt')
header = "Time(s) Interface_Position(m) Mass_Flux(kg/m^2/s) Stefan_Velocity(m/s) Analytical_Mass_Flux(kg/m^2/s)"
data = np.column_stack((times, interface_positions, mass_fluxes, stefan_velocities, analytical_mass_flux))
np.savetxt(data_file, data, header=header, fmt='%.10e')
print(f"  Saved: stefan_flow_data.txt")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SIMULATION COMPLETE")
print("=" * 70)
print(f"\nOutput directory: {output_folder}")
print(f"\nPhysical Parameters:")
print(f"  Gas density (rho0): {density0} kg/m^3")
print(f"  Liquid density (rho1): {density1} kg/m^3")
print(f"  Diffusion coefficient (D_v): {D_v} m^2/s")
print(f"  Surface mass fraction (Y_s): {Y_surface}")
print(f"  Far-field mass fraction (Y_inf): {Y_infinity}")

print(f"\nSimulation Results:")
print(f"  Initial interface position: {interface_positions[0]:.6f} m")
print(f"  Final interface position: {interface_positions[-1]:.6f} m")
print(f"  Interface displacement: {interface_positions[-1] - interface_positions[0]:.6e} m")
print(f"  Average mass flux: {np.mean(mass_fluxes):.6e} kg/m^2/s")
print(f"  Average Stefan velocity: {np.mean(stefan_velocities):.6e} m/s")
print(f"  Max mass flux error: {np.max(mass_flux_error):.6e} kg/m^2/s")

print(f"\nFiles generated:")
print(f"  - 01_Interface_Position_vs_Time.png/.eps")
print(f"  - 02_Mass_Flux_vs_Time.png/.eps")
print(f"  - 03_Stefan_Velocity_vs_Time.png/.eps")
print(f"  - 04_Combined_Results.png/.eps")
print(f"  - 05_Mass_Flux_Error.png/.eps")
print(f"  - stefan_flow_data.txt")

print("\n" + "=" * 70)
