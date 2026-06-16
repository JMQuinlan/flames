"""
Extract 1D slice from AMReX data using yt for CO2 Depressurization Test
Compares numerical results with exact Riemann solution for stiffened gas EOS
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.optimize import brentq
from math import sqrt
from collections import namedtuple

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# Configuration
case_name = 'Test1_CO2_NoVap'
amrex_output_dir = r'../../../bin/tests/FlowDepressurization/Test1_CO2_NoVap'

# CO2 Stiffened Gas EOS Parameters (from Table 1)
# Liquid
gamma_L = 1.23
p_inf_L = 1.32e8  # Pa
cv_L = 2.44e3     # J/(kg*K)
q_L = -6.23e5     # J/kg

# Vapor
gamma_V = 1.06
p_inf_V = 8.86e5  # Pa
cv_V = 2.41e3     # J/(kg*K)
q_V = -3.01e5     # J/kg

# Initial conditions (from Lund test case)
# Left side (mostly liquid)
P_L = 6.0e6       # 60 bar
T_L = 273.0       # K
rho_L = 902.0     # kg/m^3 (calculated)

# Right side (mostly vapor)
P_R = 1.0e6       # 10 bar
T_R = 273.0       # K
rho_R = 47.8      # kg/m^3 (calculated)

# Geometry
x_interface = 50.0  # m
L_total = 80.0      # m
t_final = 0.08      # s

# Use liquid properties for left state, vapor for right state
gamma_left = gamma_L
p_inf_left = p_inf_L
gamma_right = gamma_V
p_inf_right = p_inf_V

print("="*60)
print("CO2 DEPRESSURIZATION TEST - EXACT RIEMANN SOLUTION")
print("="*60)
print(f"\nStiffened Gas EOS Parameters:")
print(f"  Left (liquid):  gamma = {gamma_left}, p_inf = {p_inf_left/1e6:.2f} MPa")
print(f"  Right (vapor):  gamma = {gamma_right}, p_inf = {p_inf_right/1e6:.2f} MPa")
print(f"\nInitial Conditions:")
print(f"  Left:  P = {P_L/1e5:.1f} bar, T = {T_L:.1f} K, rho = {rho_L:.1f} kg/m^3")
print(f"  Right: P = {P_R/1e5:.1f} bar, T = {T_R:.1f} K, rho = {rho_R:.1f} kg/m^3")
print(f"  Interface at x = {x_interface} m")

# Define state structure
primitive_variables = ('Density', 'Velocity', 'Pressure', 'gamma', 'p_inf')
Primitive_State = namedtuple('State', primitive_variables)

# Create left and right states with EOS parameters
left_state = Primitive_State(Density=rho_L, Velocity=0.0, Pressure=P_L, 
                             gamma=gamma_left, p_inf=p_inf_left)
right_state = Primitive_State(Density=rho_R, Velocity=0.0, Pressure=P_R, 
                              gamma=gamma_right, p_inf=p_inf_right)

# Stiffened gas EOS functions
def speed_of_sound(State):
    """Calculate sound speed for stiffened gas EOS"""
    return sqrt(State.gamma * (State.Pressure + State.p_inf) / State.Density)

def internal_energy(State):
    """Calculate internal energy for stiffened gas EOS"""
    return (State.Pressure + State.gamma * State.p_inf) / (State.Density * (State.gamma - 1.0))

def estimate_p_star(left_state, right_state):
    """Estimate p* with the two-rarefaction solution for stiffened gas"""
    c_l = speed_of_sound(left_state)
    c_r = speed_of_sound(right_state)
    u_l = left_state.Velocity
    u_r = right_state.Velocity
    p_l = left_state.Pressure
    p_r = right_state.Pressure
    gamma_l = left_state.gamma
    gamma_r = right_state.gamma
    
    # Use average gamma for estimate
    gamma_avg = 0.5 * (gamma_l + gamma_r)
    gmmm1 = gamma_avg - 1.0
    
    p_star_TR = ((c_l + c_r - 0.5*gmmm1*(u_r - u_l))
                 /(c_l/(p_l**(gmmm1/(2.0*gamma_avg))) + c_r/(p_r**(gmmm1/(2.0*gamma_avg)))))**(2.0*gamma_avg/gmmm1)
    
    if p_star_TR < 1e-6:
        p_star_TR = 1e-6
    
    return p_star_TR

def f_k_function(p, State):
    """Pressure function for stiffened gas EOS"""
    gamma = State.gamma
    p_inf = State.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    A_k = 2.0 / (gmmp1 * State.Density)
    B_k = gmmm1 / gmmp1 * (State.Pressure + 2.0 * p_inf)
    
    if p > State.Pressure:
        # Shock wave
        f_k = (p - State.Pressure) * sqrt(A_k / (p + B_k))
    else:
        # Rarefaction wave
        c_k = speed_of_sound(State)
        f_k = (2.0 * c_k / gmmm1 * 
               (((p + p_inf) / (State.Pressure + p_inf))**(gmmm1 / (2.0 * gamma)) - 1.0))
    
    return f_k

def p_function(p, left_state, right_state):
    """Root finding function for p*"""
    Delta_u = right_state.Velocity - left_state.Velocity
    f = f_k_function(p, left_state) + f_k_function(p, right_state) + Delta_u
    return f

def calc_u_star(p_star, left_state, right_state):
    """Calculate u* from p*"""
    return 0.5 * (left_state.Velocity + right_state.Velocity) + 0.5 * (f_k_function(p_star, right_state) - f_k_function(p_star, left_state))

def calc_density_star(State, p_star):
    """Calculate density in star region for stiffened gas"""
    gamma = State.gamma
    p_inf = State.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    if p_star > State.Pressure:
        # Shock
        density_star_k = State.Density * ((p_star + gmmm1/gmmp1 * (State.Pressure + 2.0*p_inf))
                                          / (State.Pressure + gmmm1/gmmp1 * (p_star + 2.0*p_inf)))
    else:
        # Rarefaction
        density_star_k = State.Density * ((p_star + p_inf) / (State.Pressure + p_inf))**(1.0 / gamma)
    
    return density_star_k

def calc_left_shock_speed(State, p_star):
    """Calculate left shock speed for stiffened gas"""
    gamma = State.gamma
    p_inf = State.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    return State.Velocity - speed_of_sound(State) * sqrt(gmmp1 / (2.0 * gamma) * (p_star + p_inf) / (State.Pressure + p_inf) + gmmm1 / (2.0 * gamma))

def calc_right_shock_speed(State, p_star):
    """Calculate right shock speed for stiffened gas"""
    gamma = State.gamma
    p_inf = State.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    return State.Velocity + speed_of_sound(State) * sqrt(gmmp1 / (2.0 * gamma) * (p_star + p_inf) / (State.Pressure + p_inf) + gmmm1 / (2.0 * gamma))

def fan_left_state(left_state, x, t):
    """Calculate state inside left rarefaction fan for stiffened gas"""
    gamma = left_state.gamma
    p_inf = left_state.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    c_left = speed_of_sound(left_state)
    
    rho = left_state.Density * (2.0/gmmp1 + gmmm1/(gmmp1*c_left)*(left_state.Velocity - x/t))**(2.0/gmmm1)
    u = 2.0/gmmp1 * (c_left + gmmm1/2.0 * left_state.Velocity + x/t)
    p = (left_state.Pressure + p_inf) * (2.0/gmmp1 + gmmm1/(gmmp1*c_left)*(left_state.Velocity - x/t))**(2.0*gamma/gmmm1) - p_inf
    
    return rho, u, p

def fan_right_state(right_state, x, t):
    """Calculate state inside right rarefaction fan for stiffened gas"""
    gamma = right_state.gamma
    p_inf = right_state.p_inf
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    c_right = speed_of_sound(right_state)
    
    rho = right_state.Density * (2.0/gmmp1 - gmmm1/(gmmp1*c_right)*(right_state.Velocity - x/t))**(2.0/gmmm1)
    u = 2.0/gmmp1 * (-c_right + gmmm1/2.0 * right_state.Velocity + x/t)
    p = (right_state.Pressure + p_inf) * (2.0/gmmp1 - gmmm1/(gmmp1*c_right)*(right_state.Velocity - x/t))**(2.0*gamma/gmmm1) - p_inf
    
    return rho, u, p

print("\n" + "="*60)
print("SOLVING EXACT RIEMANN PROBLEM")
print("="*60)

# Solve for p* and u*
p_star_init = estimate_p_star(left_state, right_state)
print(f"\nInitial p* estimate: {p_star_init/1e5:.2f} bar")

p_star = brentq(p_function, 1.e-6, 1.e8, args=(left_state, right_state))
u_star = calc_u_star(p_star, left_state, right_state)

print(f"Solved p*: {p_star/1e5:.2f} bar")
print(f"Solved u*: {u_star:.2f} m/s")

density_star_left = calc_density_star(left_state, p_star)
density_star_right = calc_density_star(right_state, p_star)

print(f"Star region densities: rho_L* = {density_star_left:.2f}, rho_R* = {density_star_right:.2f} kg/m^3")

# Determine wave structure
left_shock = p_star > left_state.Pressure
right_shock = p_star > right_state.Pressure

if left_shock:
    print("\nLeft wave: SHOCK")
    S_left = calc_left_shock_speed(left_state, p_star)
    print(f"  Shock speed: {S_left:.2f} m/s")
    S_tail_left = np.nan
    S_head_left = np.nan
else:
    print("\nLeft wave: RAREFACTION")
    c_left = speed_of_sound(left_state)
    c_star_left = sqrt(left_state.gamma * (p_star + p_inf_left) / density_star_left)
    S_tail_left = u_star - c_star_left
    S_head_left = left_state.Velocity - c_left
    print(f"  Head speed: {S_head_left:.2f} m/s")
    print(f"  Tail speed: {S_tail_left:.2f} m/s")
    S_left = np.nan

if right_shock:
    print("\nRight wave: SHOCK")
    S_right = calc_right_shock_speed(right_state, p_star)
    print(f"  Shock speed: {S_right:.2f} m/s")
    S_tail_right = np.nan
    S_head_right = np.nan
else:
    print("\nRight wave: RAREFACTION")
    c_right = speed_of_sound(right_state)
    c_star_right = sqrt(right_state.gamma * (p_star + p_inf_right) / density_star_right)
    S_tail_right = u_star + c_star_right
    S_head_right = right_state.Velocity + c_right
    print(f"  Head speed: {S_head_right:.2f} m/s")
    print(f"  Tail speed: {S_tail_right:.2f} m/s")
    S_right = np.nan

# Generate exact solution at t = t_final
print(f"\nGenerating exact solution at t = {t_final} s")

Nx = 2000
xvec = np.linspace(0, L_total, Nx)
statevec = np.array([]).reshape(0, 3)

for x in xvec:
    x_rel = x - x_interface  # Position relative to interface
    
    if x_rel < u_star * t_final:
        if left_shock:
            if x_rel < S_left * t_final:
                statevec = np.concatenate((statevec, np.array([[left_state.Density, left_state.Velocity, left_state.Pressure]])), axis=0)
            else:
                statevec = np.concatenate((statevec, np.array([[density_star_left, u_star, p_star]])), axis=0)
        else:
            if x_rel < S_head_left * t_final:
                statevec = np.concatenate((statevec, np.array([[left_state.Density, left_state.Velocity, left_state.Pressure]])), axis=0)
            elif x_rel < S_tail_left * t_final:
                statevec = np.concatenate((statevec, [np.squeeze([fan_left_state(left_state, x_rel, t_final)])]), axis=0)
            else:
                statevec = np.concatenate((statevec, np.array([[density_star_left, u_star, p_star]])), axis=0)
    else:
        if right_shock:
            if x_rel > S_right * t_final:
                statevec = np.concatenate((statevec, np.array([[right_state.Density, right_state.Velocity, right_state.Pressure]])), axis=0)
            else:
                statevec = np.concatenate((statevec, np.array([[density_star_right, u_star, p_star]])), axis=0)
        else:
            if x_rel > S_head_right * t_final:
                statevec = np.concatenate((statevec, np.array([[right_state.Density, right_state.Velocity, right_state.Pressure]])), axis=0)
            elif x_rel > S_tail_right * t_final:
                statevec = np.concatenate((statevec, [np.squeeze([fan_right_state(right_state, x_rel, t_final)])]), axis=0)
            else:
                statevec = np.concatenate((statevec, np.array([[density_star_right, u_star, p_star]])), axis=0)

x_exact = xvec
density_exact = statevec[:, 0]
velocity_exact = statevec[:, 1]
pressure_exact = statevec[:, 2]

print(f"Generated {len(x_exact)} points for exact solution")

print("\n" + "="*60)
print("EXTRACTING DATA FROM AMReX OUTPUT USING YT")
print("="*60)

# Find the plot files in the output directory
plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and item.endswith('cell'):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No *cell directories found in {amrex_output_dir}")
    print("Looking for directories like: 00000cell, 00100cell, etc.")
    exit(1)

# Sort and use the last one
plot_files.sort()
last_plot = plot_files[-1]

print(f"\nFound {len(plot_files)} plot files")
print(f"Using last timestep: {os.path.basename(last_plot)}")

# Load the dataset
print(f"\nLoading dataset...")
ds = yt.load(last_plot)
sim_time = float(ds.current_time)
print(f"Simulation time: {sim_time:.6f} s")
print(f"Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")

# Create a ray (1D line) through the domain at y=0
print(f"\nExtracting 1D slice at y=0...")

# Define the ray from x=0 to x=80 at y=0, z=0
ray_start = ds.arr([0.0, 0.0, 0.0], 'code_length')
ray_end = ds.arr([L_total, 0.0, 0.0], 'code_length')
ray = ds.ray(ray_start, ray_end)

# Sort by x coordinate
sort_indices = np.argsort(ray['x'])
x_numerical = np.array(ray['x'][sort_indices])
velocity_numerical = np.array(ray['velocityx'][sort_indices])
pressure_numerical = np.array(ray['pressure'][sort_indices])
density_numerical = np.array(ray['density'][sort_indices])

print(f"Extracted {len(x_numerical)} points")

# Print data ranges
print(f"\nNumerical data ranges:")
print(f"  x: [{np.min(x_numerical):.6f}, {np.max(x_numerical):.6f}] m")
print(f"  velocity: [{np.min(velocity_numerical):.6e}, {np.max(velocity_numerical):.6e}] m/s")
print(f"  pressure: [{np.min(pressure_numerical):.6e}, {np.max(pressure_numerical):.6e}] Pa")
print(f"  density: [{np.min(density_numerical):.6e}, {np.max(density_numerical):.6e}] kg/m^3")

print("\n" + "="*60)
print("CREATING COMPARISON PLOTS")
print("="*60)

# Create comparison plots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Density
axes[0].set_title(f'CO2 Depressurization (No Vaporization) at t={sim_time:.6f}s', 
                  fontsize=14, fontweight='bold')
axes[0].plot(x_exact, density_exact, 'b-', linewidth=2, 
             label='Exact Solution (Stiffened Gas)', zorder=1)
axes[0].plot(x_numerical, density_numerical, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[0].set_xlabel('Position (m)', fontsize=12)
axes[0].set_ylabel('Density (kg/m^3)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, L_total])

# Velocity
axes[1].plot(x_exact, velocity_exact, 'b-', linewidth=2, 
             label='Exact Solution (Stiffened Gas)', zorder=1)
axes[1].plot(x_numerical, velocity_numerical, 'r--', linewidth=1.5, 
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[1].set_xlabel('Position (m)', fontsize=12)
axes[1].set_ylabel('Velocity (m/s)', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, L_total])

# Pressure
axes[2].plot(x_exact, pressure_exact/1e5, 'b-', linewidth=2, 
             label='Exact Solution (Stiffened Gas)', zorder=1)
axes[2].plot(x_numerical, pressure_numerical/1e5, 'r--', linewidth=1.5,
             label='Numerical Solution', zorder=2, alpha=0.7)
axes[2].set_xlabel('Position (m)', fontsize=12)
axes[2].set_ylabel('Pressure (bar)', fontsize=12)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, L_total])

plt.tight_layout()

# Create output directory if it doesn't exist
os.makedirs('./Images', exist_ok=True)

output_filename = f'./Images/{case_name}_comparison'
plt.savefig(output_filename+'.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename+'.eps', format='eps', bbox_inches='tight')
print(f"\nPlot saved: {output_filename}")
plt.show()

# Calculate error metrics (interpolate exact to numerical grid)
pressure_exact_interp = np.interp(x_numerical, x_exact, pressure_exact)
density_exact_interp = np.interp(x_numerical, x_exact, density_exact)
velocity_exact_interp = np.interp(x_numerical, x_exact, velocity_exact)

print("\n" + "="*60)
print("ERROR METRICS (vs. Exact Riemann Solution)")
print("="*60)
print(f"Velocity L2 error:   {np.linalg.norm(velocity_numerical - velocity_exact_interp):.6e}")
print(f"Velocity Linf error: {np.max(np.abs(velocity_numerical - velocity_exact_interp)):.6e}")
print(f"\nPressure L2 error:   {np.linalg.norm(pressure_numerical - pressure_exact_interp):.6e}")
print(f"Pressure Linf error: {np.max(np.abs(pressure_numerical - pressure_exact_interp)):.6e}")
print(f"\nDensity L2 error:    {np.linalg.norm(density_numerical - density_exact_interp):.6e}")
print(f"Density Linf error:  {np.max(np.abs(density_numerical - density_exact_interp)):.6e}")
print("="*60)

print("\nAnalysis complete!")
print("Exact Riemann solution computed using stiffened gas EOS for CO2.")
