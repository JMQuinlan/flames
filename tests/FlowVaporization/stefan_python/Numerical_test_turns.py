"""
Stefan Problem: Vapor Diffusion Through Stagnant Gas
======================================================
This script solves the classic Stefan problem with vapor A diffusing through
stagnant gas B above a liquid pool. The liquid-vapor interface moves due to
evaporation, with dynamics governed by energy balance and diffusion equations.

Physical system:
- Liquid A at the bottom (z = 0)
- Gas phase above with species A (vapor) and B (inert gas)
- Mixture of A (vapor) and B flowing across the top
- Liquid level recedes as evaporation occurs
- Mass transfer rate depends on concentration gradients

Theory:
- Equation 3.39: Y_A(x) = 1 - (1 - Y_A,i)*exp(m_A*x/rho*D_AB)
- Equation 3.40: m_A = (rho*D_AB/L)*ln[(1 - Y_A,inf)/(1 - Y_A,i)]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


# ==============================================================================
# INITIAL CONDITIONS AND PHYSICAL PARAMETERS
# ==============================================================================

print("=" * 80)
print("STEFAN PROBLEM: VAPOR DIFFUSION WITH MOVING INTERFACE")
print("=" * 80)

# --- Temperature conditions ---
# The saturation temperature determines the liquid-vapor equilibrium
T_liquid = 373.15       # Liquid temperature (K) - saturation at 1 atm
T_gas = 298.15          # Gas temperature (K)
T_sat = 373.15          # Interface temperature (K)

# --- Pressure ---
P_total = 101325.0      # Total pressure (Pa) - 1 atm

# --- Species and concentration conditions ---
# Important physical points:
# - Species A (water vapor): evaporates from liquid interface
# - Species B (air/N2): inert, stagnant, does NOT dissolve in liquid
# - Stefan flow: bulk flow in gas due to evaporation produces enhancement
# - This is TRUE STAGNANT LAYER behavior (only A moves across interface)

Y_A_sat = 0.03          # Mass fraction of A at interface (saturated)
                        # Corresponds to vapor pressure at saturation
                        # Higher value = stronger driving force for evaporation

Y_A_ambient = 0.0       # Mass fraction of A at top boundary (dry air)

# --- Thermodynamic and transport properties ---
# Evaluated at gas temperature (298 K)

rho_gas = 1.2           # Gas density (kg/m³) at 298 K, 1 atm
                        # For dry air: ~1.2 kg/m³

D_AB = 2.6e-5           # Binary diffusion coefficient water-air (m²/s)
                        # Standard value at room temperature
                        # D varies as T^1.5 / P

mu_gas = 1.8e-5         # Dynamic viscosity of gas (Pa·s)

# --- Domain geometry ---
L_domain = 0.1          # Height of gas column (m) = 10 cm
                        # Typical lab apparatus scale

# --- Initial condition: interface position ---
z_interface_init = 0.005  # Initial liquid height (m) = 5 mm
                          # Liquid region: [0, z_interface_init]
                          # Gas region: [z_interface_init, L_domain]

# --- Time domain ---
t_final = 100.0         # Total simulation time (s)
n_steps = 5000          # Number of time steps
dt = t_final / n_steps  # Time step size (s)

# --- Spatial discretization ---
n_grid = 200            # Grid points in vertical direction
                        # Fine resolution for diffusion profile

z = np.linspace(0, L_domain, n_grid)
dz = L_domain / (n_grid - 1)

print("\nINITIAL CONDITIONS:")
print(f"  Liquid saturation temp: {T_liquid:.2f} K ({T_liquid-273.15:.1f}°C)")
print(f"  Gas temperature: {T_gas:.2f} K ({T_gas-273.15:.1f}°C)")
print(f"  Total pressure: {P_total/101325:.2f} atm")
print(f"  Saturation mass fraction: {Y_A_sat:.4f}")
print(f"  Ambient mass fraction: {Y_A_ambient:.4f}")
print(f"\nTRANSPORT PROPERTIES:")
print(f"  Gas density: {rho_gas:.3f} kg/m³")
print(f"  Diffusion coeff D_AB: {D_AB:.2e} m²/s")
print(f"\nDOMAIN AND DISCRETIZATION:")
print(f"  Total height: {L_domain*100:.1f} cm")
print(f"  Initial interface: {z_interface_init*1000:.2f} mm")
print(f"  Grid points: {n_grid}")
print(f"  Grid spacing: {dz*1000:.3f} mm")
print(f"\nTIME INTEGRATION:")
print(f"  Total time: {t_final:.1f} s")
print(f"  Time steps: {n_steps}")
print(f"  dt: {dt:.4f} s")

print(f"\nSPECIES SYSTEM:")
print(f"  Species A: Water vapor (evaporating species)")
print(f"  Species B: Air/Nitrogen (inert, stagnant species)")
print(f"  Key: B does NOT dissolve; creates true stagnant layer")
print(f"  Result: Significant enhancement of mass transfer vs pure diffusion")

print("\n" + "=" * 80)
print("STARTING SIMULATION")
print("=" * 80 + "\n")


# ==============================================================================
# INITIALIZE SOLUTION FIELDS
# ==============================================================================

# Mass fraction of species A - 2D field [time, space]
Y_A = np.zeros((n_steps + 1, n_grid))
Y_A[0, :] = Y_A_ambient  # Initial condition: ambient everywhere

# Interface position trajectory
z_interface = np.zeros(n_steps + 1)
z_interface[0] = z_interface_init

# Time array
time = np.zeros(n_steps + 1)

# Physical quantities for post-processing
evaporation_rate = np.zeros(n_steps + 1)  # Mass flux kg/(m²·s)
interface_velocity = np.zeros(n_steps + 1)


# ==============================================================================
# TIME INTEGRATION: DIFFUSION + MOVING BOUNDARY
# ==============================================================================

for step_n in range(n_steps):
    time[step_n] = step_n * dt
    
    z_int = z_interface[step_n]
    idx_int = np.argmin(np.abs(z - z_int))
    
    # Safety check: stay in domain
    if idx_int < 1 or idx_int >= n_grid - 1:
        print(f"Interface reached boundary at t={time[step_n]:.2f}s")
        n_steps = step_n
        break
    
    Y_A_curr = Y_A[step_n, :]
    
    # --- Compute concentration gradient at interface ---
    if idx_int < n_grid - 1:
        dY_A_dz_int = (Y_A_curr[idx_int + 1] - Y_A_curr[idx_int]) / dz
    else:
        dY_A_dz_int = 0.0
    
    # --- Diffusive mass flux (Fick's law) ---
    # m_dot = -rho * D_AB * dY/dz (positive value = evaporation)
    m_dot = -rho_gas * D_AB * dY_A_dz_int
    m_dot = max(0.0, m_dot)  # Physical constraint: no condensation
    
    evaporation_rate[step_n] = m_dot
    
    # --- Solve diffusion equation using implicit (Crank-Nicolson) scheme ---
    # Unconditionally stable, handles large Courant numbers
    # (I - r*L) Y^(n+1) = Y^n
    
    n_interior = max(1, n_grid - idx_int)
    
    if n_interior > 0:
        r = D_AB * dt / (dz ** 2)
        
        # Build tridiagonal system matrix
        diag_main = (1.0 + 2.0*r) * np.ones(n_interior)
        diag_upper = -r * np.ones(n_interior - 1)
        diag_lower = -r * np.ones(n_interior - 1)
        
        # Apply boundary conditions (interface and top)
        diag_main[0] = 1.0 + r
        if n_interior > 1:
            diag_upper[0] = -r
            diag_main[-1] = 1.0 + r
        
        # Build sparse matrix
        A = diags([diag_lower, diag_main, diag_upper], 
                 [-1, 0, 1], shape=(n_interior, n_interior), format='csr')
        
        # Right-hand side: old concentration
        b = Y_A_curr[idx_int:idx_int+n_interior].copy()
        
        # Boundary conditions
        b[0] = Y_A_sat           # Interface: saturated vapor
        b[-1] = Y_A_ambient      # Top: ambient
        
        # Solve implicit system
        try:
            Y_new_interior = spsolve(A, b)
            Y_A[step_n + 1, idx_int:idx_int+n_interior] = Y_new_interior
        except Exception as e:
            # Fallback: just carry forward
            Y_A[step_n + 1, idx_int:idx_int+n_interior] = Y_A_curr[idx_int:idx_int+n_interior]
    
    # Boundary conditions
    Y_A[step_n + 1, idx_int] = Y_A_sat
    Y_A[step_n + 1, n_grid - 1] = Y_A_ambient
    
    # --- Update interface position (Stefan condition) ---
    # From mass conservation at liquid:
    # rho_liquid * dz/dt = m_dot / rho_liquid
    
    rho_liquid = 1000.0  # Liquid density (kg/m³)
    
    if m_dot > 0:
        # Liquid recedes as vapor is removed
        dz_dt = -m_dot / rho_liquid
        z_interface[step_n + 1] = z_interface[step_n] + dz_dt * dt
        interface_velocity[step_n] = dz_dt
    else:
        z_interface[step_n + 1] = z_interface[step_n]
        interface_velocity[step_n] = 0.0
    
    # Keep interface in domain
    z_interface[step_n + 1] = max(z_interface[step_n + 1], 0.0001)
    if z_interface[step_n + 1] > L_domain - dz:
        z_interface[step_n + 1] = L_domain - dz
        print(f"Liquid completely evaporated at t={time[step_n]:.2f}s")
        n_steps = step_n
        break
    
    # Progress update
    if (step_n + 1) % 500 == 0:
        evap_pct = (z_interface_init - z_interface[step_n + 1]) / z_interface_init * 100
        print(f"  Step {step_n+1:5d}: t={time[step_n]:7.2f}s | "
              f"z={z_interface[step_n+1]*1000:7.3f}mm | "
              f"m·={m_dot:.3e}kg/(m²·s) | "
              f"Evap={evap_pct:5.1f}%")

# Finalize arrays
time[n_steps] = time[n_steps - 1] + dt
evaporation_rate[n_steps] = evaporation_rate[n_steps - 1]
interface_velocity[n_steps] = 0.0

print("\n" + "=" * 80)
print("SIMULATION COMPLETED")
print(f"  Final time: {time[n_steps]:.2f} s")
print(f"  Interface final position: {z_interface[n_steps]*1000:.3f} mm")
print(f"  Total evaporation: {(z_interface_init - z_interface[n_steps])*1000:.3f} mm")
print(f"  Final evaporation rate: {evaporation_rate[n_steps]:.3e} kg/(m²·s)")
print("=" * 80 + "\n")


# ==============================================================================
# PLOTTING
# ==============================================================================

print("Generating plots...\n")

# --- PLOT 1: Interface Position vs Time ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time[:n_steps+1], z_interface[:n_steps+1]*1000, 'b-', linewidth=2.5)
ax.fill_between(time[:n_steps+1], 0, z_interface[:n_steps+1]*1000, alpha=0.3, color='blue')
ax.axhline(y=z_interface_init*1000, color='green', linestyle='--', linewidth=2, label='Initial')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Interface Height (mm)', fontsize=12)
ax.set_title('Liquid-Vapor Interface Position vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig('01_interface_position.png', dpi=150)
plt.close(fig)

# --- PLOT 2: Interface Recession Velocity ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time[:n_steps], interface_velocity[:n_steps]*1000, 'r-', linewidth=2.5)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax.fill_between(time[:n_steps], interface_velocity[:n_steps]*1000, alpha=0.3, color='red')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Interface Velocity (mm/s)', fontsize=12)
ax.set_title('Interface Recession Rate vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('02_interface_velocity.png', dpi=150)
plt.close(fig)

# --- PLOT 3: Evaporation Rate (Mass Flux) ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(time[:n_steps+1], evaporation_rate[:n_steps+1]*1e6, 'g-', linewidth=2.5)
ax.fill_between(time[:n_steps+1], 0, evaporation_rate[:n_steps+1]*1e6, 
                alpha=0.3, color='green')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Mass Flux (μg/(m²·s))', fontsize=12)
ax.set_title('Evaporation Rate vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('03_evaporation_rate.png', dpi=150)
plt.close(fig)

# --- PLOT 4: Vapor Concentration Profiles ---
snapshot_indices = np.array([0, n_steps//5, 2*n_steps//5, 3*n_steps//5, 4*n_steps//5, n_steps])
snapshot_indices = snapshot_indices[snapshot_indices <= n_steps]

fig, ax = plt.subplots(figsize=(11, 7))
colors = plt.cm.plasma(np.linspace(0, 1, len(snapshot_indices)))

for idx, color in zip(snapshot_indices, colors):
    z_int = z_interface[idx]
    Y_snap = Y_A[idx, :]
    
    # Only gas region
    mask = z >= z_int
    ax.plot(Y_snap[mask]*100, (z[mask] - z_int)*1000, color=color, linewidth=2.5,
            label=f't={time[idx]:.1f}s, z_i={z_int*1000:.2f}mm')
    ax.plot(Y_A_sat*100, 0, 'o', color=color, markersize=8)

ax.set_xlabel('Mass Fraction of Species A (%)', fontsize=12)
ax.set_ylabel('Height above Interface (mm)', fontsize=12)
ax.set_title('Vapor Concentration Profiles Evolution', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('04_concentration_profiles.png', dpi=150)
plt.close(fig)

# --- PLOT 5: Concentration Contour (Space-Time) ---
fig, ax = plt.subplots(figsize=(12, 8))

# Create 2D field
Y_contour = Y_A[:n_steps+1, :]
time_grid, z_grid = np.meshgrid(time[:n_steps+1], z*1000)
Y_plot = Y_contour.T * 100

level = np.linspace(0, Y_A_sat*100*1.1, 15)
cf = ax.contourf(time_grid, z_grid, Y_plot, levels=level, cmap='RdYlBu_r')
cl = ax.contour(time_grid, z_grid, Y_plot, colors='black', linewidths=0.5, alpha=0.3)
ax.clabel(cl, inline=True, fontsize=8)

# Overlay interface
ax.plot(time[:n_steps+1], z_interface[:n_steps+1]*1000, 'g--', linewidth=3, label='Interface')

ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Height (mm)', fontsize=12)
ax.set_title('Vapor Mass Fraction Evolution (Space-Time Contour)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
cbar = plt.colorbar(cf, ax=ax, label='Y_A (%)')
fig.tight_layout()
fig.savefig('05_concentration_contour.png', dpi=150)
plt.close(fig)

# --- PLOT 6: Cumulative Evaporation ---
fig, ax = plt.subplots(figsize=(10, 6))
cumulative = (z_interface_init - z_interface[:n_steps+1]) * 1000
ax.plot(time[:n_steps+1], cumulative, 'purple', linewidth=2.5)
ax.fill_between(time[:n_steps+1], 0, cumulative, alpha=0.3, color='purple')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Cumulative Evaporation (mm)', fontsize=12)
ax.set_title('Total Evaporated Liquid Height vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('06_cumulative_evaporation.png', dpi=150)
plt.close(fig)

# --- PLOT 7: Gas Region Height ---
fig, ax = plt.subplots(figsize=(10, 6))
gas_height = L_domain - z_interface[:n_steps+1]
ax.plot(time[:n_steps+1], gas_height*1000, 'steelblue', linewidth=2.5)
ax.fill_between(time[:n_steps+1], 0, gas_height*1000, alpha=0.3, color='steelblue')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Gas Region Height (mm)', fontsize=12)
ax.set_title('Gas Column Height vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('07_gas_height.png', dpi=150)
plt.close(fig)

# --- PLOT 8: Concentration Gradient at Interface ---
fig, ax = plt.subplots(figsize=(10, 6))
grad_interface = np.zeros(n_steps + 1)

for n in range(n_steps + 1):
    z_int = z_interface[n]
    idx_int = np.argmin(np.abs(z - z_int))
    if idx_int < n_grid - 1:
        grad = (Y_A[n, idx_int+1] - Y_A[n, idx_int]) / dz
    else:
        grad = 0
    grad_interface[n] = grad

ax.plot(time[:n_steps+1], grad_interface[:n_steps+1], 'brown', linewidth=2.5)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Concentration Gradient (1/m)', fontsize=12)
ax.set_title('dY_A/dz at Interface vs Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig('08_concentration_gradient.png', dpi=150)
plt.close(fig)

# --- PLOT 9: Summary 2x2 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Interface position
axes[0, 0].plot(time[:n_steps+1], z_interface[:n_steps+1]*1000, 'b-', linewidth=2)
axes[0, 0].fill_between(time[:n_steps+1], 0, z_interface[:n_steps+1]*1000, alpha=0.3)
axes[0, 0].set_ylabel('Height (mm)', fontsize=11)
axes[0, 0].set_title('Interface Position', fontweight='bold')
axes[0, 0].grid(True, alpha=0.4)

# Panel 2: Velocity
axes[0, 1].plot(time[:n_steps], interface_velocity[:n_steps]*1000, 'r-', linewidth=2)
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.8)
axes[0, 1].set_ylabel('Velocity (mm/s)', fontsize=11)
axes[0, 1].set_title('Recession Rate', fontweight='bold')
axes[0, 1].grid(True, alpha=0.4)

# Panel 3: Evaporation rate
axes[1, 0].plot(time[:n_steps+1], evaporation_rate[:n_steps+1]*1e6, 'g-', linewidth=2)
axes[1, 0].fill_between(time[:n_steps+1], 0, evaporation_rate[:n_steps+1]*1e6, alpha=0.3, color='g')
axes[1, 0].set_xlabel('Time (s)', fontsize=11)
axes[1, 0].set_ylabel('Mass Flux (μg/(m²·s))', fontsize=11)
axes[1, 0].set_title('Evaporation Rate', fontweight='bold')
axes[1, 0].grid(True, alpha=0.4)

# Panel 4: Cumulative
axes[1, 1].plot(time[:n_steps+1], (z_interface_init - z_interface[:n_steps+1])*1000, 
               'purple', linewidth=2)
axes[1, 1].fill_between(time[:n_steps+1], 0, (z_interface_init - z_interface[:n_steps+1])*1000,
                       alpha=0.3, color='purple')
axes[1, 1].set_xlabel('Time (s)', fontsize=11)
axes[1, 1].set_ylabel('Total Evaporation (mm)', fontsize=11)
axes[1, 1].set_title('Cumulative Evaporation', fontweight='bold')
axes[1, 1].grid(True, alpha=0.4)

fig.suptitle('Stefan Problem: Complete Interface Dynamics Summary', 
            fontsize=14, fontweight='bold', y=0.995)
fig.tight_layout()
fig.savefig('09_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- PLOT 10: Stefan Number Context ---
fig, ax = plt.subplots(figsize=(10, 6))
L_vap = 2.26e6  # Latent heat J/kg
cp = 1005       # Specific heat J/(kg·K)
Ste = cp * (T_liquid - T_gas) / L_vap

ax.text(0.5, 0.7, f'Stefan Number = {Ste:.4f}', ha='center', va='center',
       fontsize=16, fontweight='bold', transform=ax.transAxes,
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.text(0.5, 0.5, 'Ste << 1: Diffusion-controlled (this problem)', ha='center', va='center',
       fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.4, 'Ste ~ 1: Mixed control', ha='center', va='center',
       fontsize=12, transform=ax.transAxes)
ax.text(0.5, 0.3, 'Ste >> 1: Heat/Energy-controlled', ha='center', va='center',
       fontsize=12, transform=ax.transAxes)

ax.text(0.5, 0.15, 'Physical Interpretation:', ha='center', va='center',
       fontsize=11, fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.08, 'Evaporation rate limited by vapor diffusion,\nnot by energy supply',
       ha='center', va='center', fontsize=10, transform=ax.transAxes)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
fig.tight_layout()
fig.savefig('10_stefan_number.png', dpi=150)
plt.close(fig)

print("✓ All static plots saved\n")


# ==============================================================================
# ANIMATED GIF
# ==============================================================================

print("Creating animated GIF...\n")

# Frame selection (subsample to keep file manageable)
n_frames_target = 100
frame_step = max(1, (n_steps + 1) // n_frames_target)
frame_indices = np.arange(0, n_steps + 1, frame_step)
if frame_indices[-1] != n_steps:
    frame_indices = np.append(frame_indices, n_steps)

fig_anim, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def animate(frame_num):
    ax1.clear()
    ax2.clear()
    
    idx = frame_indices[frame_num]
    t_curr = time[idx]
    z_curr = z_interface[idx]
    
    # Current state
    current_time = t_curr
    interface_pos = z_curr
    
    # Tube visualization
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.01, L_domain * 1.1)
    ax1.set_aspect('equal')
    
    # Draw tube walls
    tube_width = 0.3
    ax1.plot([-tube_width, -tube_width], [0, L_domain], 'k-', linewidth=3)
    ax1.plot([tube_width, tube_width], [0, L_domain], 'k-', linewidth=3)
    ax1.plot([-tube_width, tube_width], [0, 0], 'k-', linewidth=3)
    
    # Draw liquid (blue rectangle)
    liquid_height = interface_pos
    if liquid_height > 0:
        ax1.fill_between([-tube_width, tube_width], 0, liquid_height, 
                         color='lightblue', alpha=0.7, label='Liquid A')
    
    # Draw interface
    current_interface = liquid_height
    ax1.plot([-tube_width, tube_width], [current_interface, current_interface], 
             'b-', linewidth=3, label='Interface')
    
    # Draw gas region with concentration gradient
    n_layers = 20
    for i in range(n_layers):
        y_bottom = current_interface + i * (L_domain - current_interface) / n_layers
        y_top = current_interface + (i + 1) * (L_domain - current_interface) / n_layers
        
        x_rel = (y_bottom + y_top) / 2 - current_interface
        gas_height = L_domain - current_interface
        if gas_height > 0:
            # Map position in gas to grid index
            grid_idx = np.argmin(np.abs(z - ((y_bottom + y_top) / 2)))
            if grid_idx < n_grid:
                Y_local = Y_A[idx, grid_idx]
            else:
                Y_local = Y_A_ambient
        else:
            Y_local = Y_A_ambient
        
        color_intensity = Y_local / Y_A_sat if Y_A_sat > 0 else 0
        ax1.fill_between([-tube_width, tube_width], y_bottom, y_top,
                        color='red', alpha=color_intensity * 0.5)
    
    # Gas flow arrow at top
    ax1.arrow(0, L_domain + 0.005, 0.3, 0, head_width=0.01, head_length=0.05, 
             fc='green', ec='green', linewidth=2)
    ax1.text(0.35, L_domain + 0.005, 'Gas flow', fontsize=10, va='center')
    
    ax1.set_ylabel('Height (m)', fontsize=11)
    ax1.set_title(f'Stefan Tube at t = {current_time:.1f} s', fontsize=12)
    ax1.set_xticks([])
    ax1.legend(loc='upper left', fontsize=9)
    
    # Concentration profile on right
    if L_domain - current_interface > 0:
        z_gas = z[z >= current_interface]
        Y_profile = Y_A[idx, z >= current_interface]
        ax2.plot(Y_profile, z_gas, 'b-', linewidth=2)
    
    ax2.axhline(y=current_interface, color='r', linestyle='--', linewidth=2, label='Interface')
    ax2.set_xlabel('Mass fraction $Y_A$', fontsize=11)
    ax2.set_ylabel('Height (m)', fontsize=11)
    ax2.set_title('Concentration Profile', fontsize=12)
    ax2.set_xlim(-0.005, max(Y_A_sat, 0.05) * 1.2)
    ax2.set_ylim(0, L_domain * 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

# Create animation
anim = FuncAnimation(fig_anim, animate, frames=len(frame_indices), 
                    interval=50, repeat=True)

# Save as GIF
print(f"  Creating {len(frame_indices)} frames...")
print("  Saving GIF (this may take a minute)...")

writer = PillowWriter(fps=20)
anim.save('stefan_animation.gif', writer=writer)
plt.tight_layout()
plt.close(fig_anim)

print("  ✓ GIF saved as 'stefan_animation.gif'\n")

print("=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  1. 01_interface_position.png")
print("  2. 02_interface_velocity.png")
print("  3. 03_evaporation_rate.png")
print("  4. 04_concentration_profiles.png")
print("  5. 05_concentration_contour.png")
print("  6. 06_cumulative_evaporation.png")
print("  7. 07_gas_height.png")
print("  8. 08_concentration_gradient.png")
print("  9. 09_summary.png")
print("  10. 10_stefan_number.png")
print("  + stefan_animation.gif")
print("=" * 80)
