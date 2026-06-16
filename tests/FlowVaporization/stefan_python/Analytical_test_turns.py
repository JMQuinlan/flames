import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

# Stefan Problem: Evaporation of liquid A into stagnant gas B
# This models diffusion through a tube with liquid at bottom and gas flow at top

# Physical parameters - these would typically come from experimental data
L = 0.1  # tube length in meters
rho = 1.2  # gas mixture density, kg/m^3 (roughly air at room temp)
D_AB = 2.5e-5  # binary diffusion coefficient, m^2/s (typical for vapor in air)

# Boundary conditions
Y_A_interface = 0.5  # mass fraction of A at liquid surface (x=0)
Y_A_infinity = 0.0  # mass fraction of A in flowing gas at top (x=L)

# Species info:
# A = evaporating liquid (e.g., water)
# B = carrier gas (e.g., air/nitrogen) - STAGNANT, does not move
# Key assumption: B is insoluble in liquid A, so mass flux of B = 0

# Temperature and pressure (assumed constant for this analysis)
T = 298  # K (25 C, room temperature)
P = 101325  # Pa (1 atm)

# Calculate mass flux using Equation 3.40
if Y_A_infinity >= Y_A_interface:
    print("No driving force for mass transfer")
    m_flux_A = 0
else:
    m_flux_A = (rho * D_AB / L) * np.log((1 - Y_A_infinity) / (1 - Y_A_interface))

print(f"Mass flux of species A: {m_flux_A:.6e} kg/(m^2·s)")
print(f"Dimensionless flux: {m_flux_A / (rho * D_AB / L):.4f}")

# Spatial discretization
x = np.linspace(0, L, 200)

# Calculate mass fraction profile using Equation 3.39
Y_A = 1 - (1 - Y_A_interface) * np.exp(m_flux_A * x / (rho * D_AB))

# Plot 1: Mass fraction distribution along tube
plt.figure(figsize=(10, 6))
plt.plot(x * 100, Y_A, 'b-', linewidth=2)
plt.xlabel('Distance from interface (cm)', fontsize=12)
plt.ylabel('Mass fraction of A, $Y_A$', fontsize=12)
plt.title('Concentration Profile in Stefan Diffusion Tube', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=Y_A_interface, color='r', linestyle='--', label=f'Interface: $Y_{{A,i}}$ = {Y_A_interface}')
plt.axhline(y=Y_A_infinity, color='g', linestyle='--', label=f'Top: $Y_{{A,\infty}}$ = {Y_A_infinity}')
plt.legend()
plt.tight_layout()
plt.savefig('mass_fraction_profile.png', dpi=300)
plt.show()

# Plot 2: Effect of interface mass fraction on flux (Table 3.1)
Y_A_i_values = np.array([0, 0.05, 0.10, 0.20, 0.50, 0.90, 0.999])
flux_values = []

for Y_i in Y_A_i_values:
    if Y_i == 0:
        flux_values.append(0)
    else:
        flux = (rho * D_AB / L) * np.log((1 - Y_A_infinity) / (1 - Y_i))
        flux_values.append(flux / (rho * D_AB / L))

plt.figure(figsize=(10, 6))
plt.plot(Y_A_i_values, flux_values, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Interface mass fraction, $Y_{A,i}$', fontsize=12)
plt.ylabel('Dimensionless mass flux, $\dot{m}_A^{\prime\prime}/(\\rho D_{AB}/L)$', fontsize=12)
plt.title('Effect of Interface Concentration on Mass Flux', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('flux_vs_interface_concentration.png', dpi=300)
plt.show()

# Plot 3: Concentration profiles for different interface conditions
plt.figure(figsize=(10, 6))
Y_i_test = [0.05, 0.20, 0.50, 0.90]
colors = ['blue', 'green', 'orange', 'red']

for Y_i, color in zip(Y_i_test, colors):
    m_flux = (rho * D_AB / L) * np.log((1 - Y_A_infinity) / (1 - Y_i))
    Y_profile = 1 - (1 - Y_i) * np.exp(m_flux * x / (rho * D_AB))
    plt.plot(x * 100, Y_profile, color=color, linewidth=2, label=f'$Y_{{A,i}}$ = {Y_i}')

plt.xlabel('Distance from interface (cm)', fontsize=12)
plt.ylabel('Mass fraction of A, $Y_A$', fontsize=12)
plt.title('Concentration Profiles for Different Interface Conditions', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('multiple_concentration_profiles.png', dpi=300)
plt.show()

# Plot 4: Diffusive flux vs position
# From Fick's law: diffusive component = -rho * D_AB * dY_A/dx
dY_dx = (1 - Y_A_interface) * (m_flux_A / (rho * D_AB)) * np.exp(m_flux_A * x / (rho * D_AB))
diffusive_flux = -rho * D_AB * dY_dx
convective_flux = Y_A * m_flux_A
total_flux = np.ones_like(x) * m_flux_A

plt.figure(figsize=(10, 6))
plt.plot(x * 100, diffusive_flux * 1e3, 'b-', linewidth=2, label='Diffusive flux')
plt.plot(x * 100, convective_flux * 1e3, 'r-', linewidth=2, label='Convective flux')
plt.plot(x * 100, total_flux * 1e3, 'k--', linewidth=2, label='Total flux (constant)')
plt.xlabel('Distance from interface (cm)', fontsize=12)
plt.ylabel('Mass flux (g/(m²·s))', fontsize=12)
plt.title('Flux Components in Stefan Problem', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('flux_components.png', dpi=300)
plt.show()

# Plot 5: Logarithmic relationship (linearization of Eq 3.37)
# -m_A'' * x / (rho * D_AB) = -ln(1 - Y_A) + C
left_side = -m_flux_A * x / (rho * D_AB)
right_side = -np.log(1 - Y_A) + np.log(1 - Y_A_interface)

plt.figure(figsize=(10, 6))
plt.plot(x * 100, left_side, 'b-', linewidth=2, label='$-\dot{m}_A^{\prime\prime} x / (\\rho D_{AB})$')
plt.plot(x * 100, right_side, 'r--', linewidth=2, label='$-\ln(1-Y_A) + C$')
plt.xlabel('Distance from interface (cm)', fontsize=12)
plt.ylabel('Dimensionless quantity', fontsize=12)
plt.title('Verification of Integrated Solution (Eq 3.37)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('solution_verification.png', dpi=300)
plt.show()

# ==============================================================================
# TIME-DEPENDENT ANALYSIS
# ==============================================================================

# Liquid density (for interface motion calculation)
rho_liquid = 1000  # kg/m^3 (water)

# Interface recession velocity (Stefan condition: constant in analytical solution)
interface_velocity_const = m_flux_A / rho_liquid  # m/s

# Time parameters
time_array = np.linspace(0, 3600, 500)  # 0 to 1 hour, 500 points
sqrt_time_array = np.sqrt(time_array)

# Interface position as function of time (linear recession)
interface_position_array = interface_velocity_const * time_array

print("\n" + "="*80)
print("TIME-DEPENDENT ANALYSIS (ANALYTICAL SOLUTION)")
print("="*80)
print(f"\nInterface recession velocity (constant): {interface_velocity_const*1e6:.6f} μm/s")
print(f"Interface recession velocity (constant): {interface_velocity_const*1e3:.9f} mm/s")
print(f"Time to completely evaporate (t = h₀/v): {0.1/interface_velocity_const:.1f} s")
print("\nNOTE: In this STEADY-STATE analytical solution:")
print("  • Interface velocity is CONSTANT (not √t dependent)")
print("  • Interface position is LINEAR in time")
print("  • Concentration profile is FIXED in space (doesn't change with time)")
print("  • Temperature is uniform (isothermal assumption)")
print("="*80 + "\n")

# ==============================================================================
# PLOT 6: Interface Velocity vs Time (Constant)
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.axhline(y=interface_velocity_const*1e6, color='r', linewidth=2.5, label='Constant velocity')
plt.plot(time_array, np.ones_like(time_array)*interface_velocity_const*1e6, 'r--', linewidth=0.5, alpha=0.5)
plt.fill_between(time_array, 0, interface_velocity_const*1e6, alpha=0.2, color='red')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Interface Velocity (μm/s)', fontsize=12)
plt.title('Interface Recession Velocity vs Time\n(Analytical Steady-State: Constant)', 
         fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 3600)
plt.ylim(0, interface_velocity_const*1e6 * 1.2)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('06_interface_velocity_vs_time.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 7: Interface Position vs Time (Linear)
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.plot(time_array, interface_position_array*1000, 'b-', linewidth=2.5)
plt.fill_between(time_array, 0, interface_position_array*1000, alpha=0.3, color='blue')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Interface Recession Distance (mm)', fontsize=12)
plt.title('Interface Position vs Time\n(Analytical Steady-State: Linear Recession)',
         fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.xlim(0, 3600)
plt.legend(['z(t) = v·t'], fontsize=11)
plt.tight_layout()
plt.savefig('07_interface_position_vs_time.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 8: Interface Velocity vs √Time (Still Constant)
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.plot(sqrt_time_array, np.ones_like(sqrt_time_array)*interface_velocity_const*1e6, 
        'purple', linewidth=2.5, label='Constant (not √t dependent)')
plt.fill_between(sqrt_time_array, 0, interface_velocity_const*1e6, alpha=0.2, color='purple')
plt.xlabel('√Time (√s)', fontsize=12)
plt.ylabel('Interface Velocity (μm/s)', fontsize=12)
plt.title('Interface Velocity vs √Time\n(Analytical Solution: NOT √t dependent)',
         fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim(0, interface_velocity_const*1e6 * 1.2)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('08_interface_velocity_vs_sqrt_time.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 9: Concentration Profile (Steady-State - Same at All Times)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Single profile
axes[0].plot(x*100, Y_A, 'b-', linewidth=2.5)
axes[0].scatter([0, L*100], [Y_A_interface, Y_A_infinity], color='red', s=100, zorder=5)
axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Interface (x=0)')
axes[0].axvline(x=L*100, color='g', linestyle='--', alpha=0.5, label='Top (x=L)')
axes[0].set_xlabel('Distance from Interface (cm)', fontsize=11)
axes[0].set_ylabel('Mass Fraction Y_A', fontsize=11)
axes[0].set_title('Concentration Profile (Steady-State)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)

# Right: Annotated with key values
axes[1].plot(x*100, Y_A, 'b-', linewidth=3)
axes[1].scatter([0], [Y_A_interface], color='red', s=150, zorder=5, label='Interface')
axes[1].scatter([L*100], [Y_A_infinity], color='green', s=150, zorder=5, label='Top')
axes[1].axhline(y=Y_A_interface, color='r', linestyle=':', alpha=0.3)
axes[1].axhline(y=Y_A_infinity, color='g', linestyle=':', alpha=0.3)

# Add mid-point annotation
x_mid = L * 0.5
Y_mid = 1 - (1 - Y_A_interface) * np.exp(m_flux_A * x_mid / (rho * D_AB))
axes[1].scatter([L*100*0.5], [Y_mid], color='orange', s=100, zorder=5)
axes[1].annotate(f'  Y_A at x=L/2\n  = {Y_mid:.4f}', 
                xy=(L*100*0.5, Y_mid), xytext=(L*100*0.6, Y_mid+0.05),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='orange'))

axes[1].set_xlabel('Distance from Interface (cm)', fontsize=11)
axes[1].set_ylabel('Mass Fraction Y_A', fontsize=11)
axes[1].set_title('Concentration Profile (Steady-State - Invariant in Time)',
                 fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('09_concentration_profile_steady_state.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 10: Interface-Top Concentration Difference vs Distance
# ==============================================================================
delta_Y = Y_A_interface - Y_A

plt.figure(figsize=(10, 6))
plt.plot(x*100, delta_Y, 'g-', linewidth=2.5)
plt.fill_between(x*100, 0, delta_Y, alpha=0.3, color='green')
plt.xlabel('Distance from Interface (cm)', fontsize=12)
plt.ylabel('Concentration Difference ΔY_A', fontsize=12)
plt.title('Driving Force for Diffusion vs Distance\n(ΔY = Y_{interface} - Y_A)',
         fontsize=13, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(['ΔY_A = 0.5 - Y_A(x)'], fontsize=11)
plt.tight_layout()
plt.savefig('10_driving_force_profile.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 11: Logarithmic Concentration Scale
# ==============================================================================
# Plot (1 - Y_A) on log scale to show exponential nature
fig, ax = plt.subplots(figsize=(10, 6))

# Only plot where Y_A < 1 to avoid log(0)
valid_idx = Y_A < 0.99
ax.semilogy(x[valid_idx]*100, (1 - Y_A[valid_idx]), 'b-', linewidth=2.5, label='1 - Y_A')

# Add theoretical line
ax.semilogy(x[valid_idx]*100, (1 - Y_A_interface)*np.exp(m_flux_A*x[valid_idx]/(rho*D_AB)), 
           'r--', linewidth=2, label='$(1-Y_{A,i}) \exp(m_A x/\\ \\rho D_{AB})$')

ax.set_xlabel('Distance from Interface (cm)', fontsize=12)
ax.set_ylabel('(1 - Y_A) [log scale]', fontsize=12)
ax.set_title('Logarithmic Concentration Profile\n(Exponential Solution)',
            fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('11_log_concentration_profile.png', dpi=300)
plt.close()

# ==============================================================================
# PLOT 12: Summary Table of Key Parameters
# ==============================================================================
fig, ax = plt.subplots(figsize=(11, 7))
ax.axis('tight')
ax.axis('off')

summary_data = [
    ['Parameter', 'Value', 'Units'],
    ['', '', ''],
    ['PHYSICAL SETUP', '', ''],
    ['Tube Length L', f'{L*100:.1f}', 'cm'],
    ['Gas Density ρ', f'{rho:.2f}', 'kg/m³'],
    ['Binary Diffusivity D_AB', f'{D_AB:.2e}', 'm²/s'],
    ['Liquid Density ρ_liquid', f'{rho_liquid:.0f}', 'kg/m³'],
    ['Temperature (isothermal)', f'{T:.0f}', 'K'],
    ['', '', ''],
    ['BOUNDARY CONDITIONS', '', ''],
    ['Interface conc. Y_A,i', f'{Y_A_interface:.4f}', '-'],
    ['Ambient conc. Y_A,∞', f'{Y_A_infinity:.4f}', '-'],
    ['Concentration difference', f'{Y_A_interface - Y_A_infinity:.4f}', '-'],
    ['', '', ''],
    ['SOLUTION RESULTS', '', ''],
    ['Mass flux m_A"', f'{m_flux_A:.4e}', 'kg/(m²·s)'],
    ['Dimensionless flux', f'{m_flux_A / (rho * D_AB / L):.4f}', '-'],
    ['Interface velocity (const)', f'{interface_velocity_const*1e6:.6f}', 'μm/s'],
    ['Interface velocity (const)', f'{interface_velocity_const*1e3:.9f}', 'mm/s'],
    ['Time to evaporate 0.1m', f'{0.1/interface_velocity_const:.1f}', 's'],
    ['', '', ''],
    ['KEY PROPERTIES', '', ''],
    ['Solution type', 'Steady-state', 'analytical'],
    ['Interface motion', 'Linear in time', 'v = const'],
    ['Concentration profile', 'Exponential', 'inv. in time'],
    ['Temperature variation', 'None (isothermal)', '-'],
]

table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                colWidths=[0.35, 0.35, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style section headers
section_rows = [2, 9, 14, 21]
for row in section_rows:
    for col in range(3):
        table[(row, col)].set_facecolor('#E8F5E9')
        table[(row, col)].set_text_props(weight='bold')

plt.title('Analytical Solution: Summary of Parameters & Results',
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('12_solution_summary.png', dpi=300)
plt.close()

print("\nAll important plots generated successfully!")
print("\nGenerated files:")
print("  1. mass_fraction_profile.png")
print("  2. flux_vs_interface_concentration.png")
print("  3. multiple_concentration_profiles.png")
print("  4. flux_components.png")
print("  5. solution_verification.png")
print("  6. interface_velocity_vs_time.png")
print("  7. interface_position_vs_time.png")
print("  8. interface_velocity_vs_sqrt_time.png")
print("  9. concentration_profile_steady_state.png")
print("  10. driving_force_profile.png")
print("  11. log_concentration_profile.png")
print("  12. solution_summary.png")