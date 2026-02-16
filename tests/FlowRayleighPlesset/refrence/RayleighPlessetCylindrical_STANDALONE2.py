import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters - MODIFIED FOR FAST LOW-PRESSURE SIMULATION
rho_L = 10            # Density of liquid [kg/m^3] - low density
mu_L = 0.1            # Dynamic viscosity of liquid [Pa.s] - high damping
S = 7.28              # Surface tension [N/m] - scaled down 10x
p_v = 0.0             # Vapor pressure [Pa] - zero
gamma = 1.4           # Adiabatic index

# Initial conditions
p_inf = 500           # 500 Pa external pressure
p_B0 = 1000           # 1000 Pa initial bubble pressure (2x overpressure)
R0 = 0.02             # 20 mm initial radius
R_dot0 = 0            # Starting at rest
y0 = [R0, R_dot0]

# Domain boundary (CRITICAL for 2D RPE)
r_inf = 5.0 * R0      # Far-field boundary location

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
time_analytical = sol.t

# Plot bubble radius vs time
print("Generating radius plot...")
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(time_analytical * 1000, radius_analytical * 1000, 'b-', linewidth=2.5, label='Analytical (Chen 2D RPE)')
ax.axhline(y=R0*1000, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, label='R0')

ax.set_xlabel('Time (ms)', fontsize=13)
ax.set_ylabel('Radius (mm)', fontsize=13)
ax.set_title('Bubble Radius vs Time - 2D Cylindrical RPE (Low Pressure)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3)

# Save plot
output_png = './rpe_radius_plot.png'
output_eps = './rpe_radius_plot.eps'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.savefig(output_eps, format='eps', bbox_inches='tight')

print(f"Saved: {output_png}")
print(f"Saved: {output_eps}")
plt.show()

print("\nSimulation Parameters:")
print(f"  rho_L = {rho_L} kg/m^3")
print(f"  mu_L = {mu_L} Pa-s")
print(f"  S = {S} N/m")
print(f"  p_v = {p_v} Pa")
print(f"  p_inf = {p_inf} Pa")
print(f"  p_B0 = {p_B0} Pa")
print(f"  R0 = {R0*1000} mm")
print(f"  r_inf = {r_inf*1000} mm")
print(f"\nResults:")
print(f"  Max radius: {np.max(radius_analytical)*1000:.4f} mm")
print(f"  Min radius: {np.min(radius_analytical)*1000:.4f} mm")
print(f"  Final radius: {radius_analytical[-1]*1000:.4f} mm")

print("\nComplete!")
