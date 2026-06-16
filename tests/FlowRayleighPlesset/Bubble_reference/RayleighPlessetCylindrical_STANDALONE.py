import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters for water at 25C
rho_L = 10            # Keep low density
mu_L = 0.1            # Still 100x normal (strong damping, but not overwhelming)
S = 7.28              # Scale down 10x to match pressure scale
p_v = 0.0             # Keep zero
gamma = 1.4           

p_inf = 500             # 5 Pa
p_B0 = 1000             # 10 Pa (2x ratio)
R0 = 0.02             # 20 mm
R_dot0 = 0


y0 = [R0, R_dot0]

# Domain boundary (CRITICAL for 2D RPE)
r_inf = 5.0 * R0     # Far-field boundary location

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
    R, R_dot = y

    if R <= 1e-12 or R >= 0.9 * r_inf:
        return [0, 0]

    ln_factor = np.log(r_inf / R)

    # Gas pressure (2D exponent)
    p_B = p_v + (p_B0 - p_v) * (R0 / R)**(2 * gamma)

    # External pressure
    p_ext = p_inf + P_force(Amp, Freq, t)

    # RHS pressure term
    pressure_term = (
        p_B
        - p_ext
        - 2 * mu_L * R_dot / R
        - S / R
    ) / rho_L

    # Correct cylindrical inertia structure
    numerator = (
        pressure_term
        - R_dot**2 * (0.5 - ln_factor)
    )

    denominator = R * ln_factor

    R_ddot = numerator / denominator

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
ax.set_title('Bubble Radius vs Time - 2D Cylindrical RPE', fontsize=14, fontweight='bold')
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

print("\nComplete!")
