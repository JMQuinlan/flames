"""
Riemann solver for 1-d finite volume solver

Started 20231005 by MQ
All rights reserved

Contains function "Riemann" with left state and right state inputs.
Each state is a named tuple with density, velocity, and pressure.
--> borrowing this structure from Clawpack:
https://github.com/clawpack/riemann_book/blob/b3c8bb94f13251bfd175b366c3fdad9c2550baa4//exact_solvers/euler.py
"""

import numpy as np
from scipy.optimize import fsolve,least_squares,brentq
from math import *
from collections import namedtuple
import argparse
import h5py
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Exact Riemann solver for Tammann EOS test cases.")
parser.add_argument("--case", type=str, help="Case number")
parser.add_argument("--plot2screen", action="store_true", help="Pass to have plots rendered on screen, if possible")
parser.add_argument("--plot2file", action="store_true", help="Pass to have plots rendered file, if possible")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
args = parser.parse_args()

if args.case:
    print(f"Requested case = {args.case}")
else:
    args.case = 'Tammann_water_air'

if args.verbose:
    print("Verbose mode is enabled.")

S_right = np.nan
S_left = np.nan
right_shock = np.nan
left_shock = np.nan
c_right = np.nan
c_left = np.nan
c_star_left = np.nan
c_star_right = np.nan
S_tail_left = np.nan
S_tail_right = np.nan
S_head_right = np.nan
S_head_left = np.nan
u_star = np.nan

# Default values for gamma and p0 (will be overridden by test cases)
gmm_left = 1.4
gmm_right = 1.4
p0_left = 0.0
p0_right = 0.0

def speed_of_sound(Primitive_State, gamma, p0):
    """Calculate sound speed for Tammann EOS: c = sqrt(gamma * (p + p0) / rho)"""
    return sqrt(gamma * (Primitive_State.Pressure + p0) / Primitive_State.Density)

def internal_energy(Primitive_State, gamma, p0):
    """Calculate specific internal energy for Tammann EOS: e = (p + gamma*p0) / (rho * (gamma - 1))"""
    return (Primitive_State.Pressure + gamma * p0) / (Primitive_State.Density * (gamma - 1.0))

def stagnation_energy(Primitive_State, gamma, p0):
    """Calculate total energy per unit volume"""
    e = internal_energy(Primitive_State, gamma, p0)
    return Primitive_State.Density * e + 0.5 * Primitive_State.Density * Primitive_State.Velocity**2

def primitive2conserved(Primitive_State, gamma, p0):
    """Convert primitive variables to conserved variables"""
    Density = Primitive_State.Density
    Momentum = Primitive_State.Density * Primitive_State.Velocity
    Energy = stagnation_energy(Primitive_State, gamma, p0)
    return Conserved_State(Density=Density, Momentum=Momentum, Energy=Energy)

def conserved2primitive(Conserved_State, gamma, p0):
    """Convert conserved variables to primitive variables"""
    Density = Conserved_State.Density
    Velocity = Conserved_State.Momentum / Conserved_State.Density
    # Rearranged Tammann EOS: p = (gamma - 1) * (E - 0.5 * rho * v^2) - gamma * p0
    Pressure = (gamma - 1.0) * (Conserved_State.Energy - 0.5 * Conserved_State.Density * Velocity**2) - gamma * p0
    return Primitive_State(Density=Density, Velocity=Velocity, Pressure=Pressure)

def estimate_p_star(left_state, right_state, gamma_left, gamma_right, p0_left, p0_right):
    """Estimate p* with the two-rarefaction solution for Tammann EOS"""
    c_l = speed_of_sound(left_state, gamma_left, p0_left)
    c_r = speed_of_sound(right_state, gamma_right, p0_right)
    u_l = left_state.Velocity
    u_r = right_state.Velocity
    p_l = left_state.Pressure
    p_r = right_state.Pressure
    
    # Adapted two-rarefaction approximation for Tammann EOS
    gmmm1_l = gamma_left - 1.0
    gmmm1_r = gamma_right - 1.0
    
    # Compute the terms for the two-rarefaction approximation
    term1 = c_l + c_r - 0.5 * (u_r - u_l) * (gmmm1_l + gmmm1_r)
    term2 = c_l / ((p_l + p0_left)**(gmmm1_l/(2.0*gamma_left)))
    term3 = c_r / ((p_r + p0_right)**(gmmm1_r/(2.0*gamma_right)))
    
    # Compute p* estimate
    p_star_TR = ((term1) / (term2 + term3))**(2.0 / (gmmm1_l/gamma_left + gmmm1_r/gamma_right))
    
    # Adjust for p0 terms
    p_star_TR = p_star_TR - 0.5 * (p0_left + p0_right)
    
    if p_star_TR < 1e-6:
        print('p* estimate too small, using alternative estimate')
        # Use pressure average as alternative estimate
        p_star_TR = max(1e-6, 0.5 * (left_state.Pressure + right_state.Pressure))
        
        # For water-air interfaces, bias toward the higher pressure
        if abs(p0_left - p0_right) > 1e6:  # Large difference in p0 indicates water-air interface
            p_star_TR = max(left_state.Pressure, right_state.Pressure)
    
    return p_star_TR

def f_k_function(p, State, gamma, p0):
    """Compute the f_k function for the pressure iteration"""
    gmmm1 = gamma - 1.0
    gmmp1 = gamma + 1.0
    
    if p > State.Pressure:
        # Shock wave
        A_k = 2.0 / (gmmp1 * State.Density)
        B_k = gmmm1 / gmmp1 * (State.Pressure + p0)
        f_k = (p - State.Pressure) * sqrt(A_k / (p + p0 + B_k))
    else:
        # Rarefaction wave
        c_k = speed_of_sound(State, gamma, p0)
        f_k = 2.0 * c_k / gmmm1 * (((p + p0) / (State.Pressure + p0))**(gmmm1/(2.0*gamma)) - 1.0)
    
    return f_k

def p_function(p, *args):
    """Function to solve for p*"""
    left_state, right_state, gamma_left, gamma_right, p0_left, p0_right = args
    
    Delta_u = right_state.Velocity - left_state.Velocity
    f = f_k_function(p, left_state, gamma_left, p0_left) + f_k_function(p, right_state, gamma_right, p0_right) + Delta_u
    
    return f

def calc_u_star(p_star, left_state, right_state, gamma_left, gamma_right, p0_left, p0_right):
    """Calculate u* from p*"""
    return 0.5 * (left_state.Velocity + right_state.Velocity) + 0.5 * (
        f_k_function(p_star, right_state, gamma_right, p0_right) - 
        f_k_function(p_star, left_state, gamma_left, p0_left)
    )

def calc_density_star(Primitive_State, p_star, gamma, p0):
    """Calculate density in star region"""
    gmmm1 = gamma - 1.0
    gmmp1 = gamma + 1.0
    
    if p_star > Primitive_State.Pressure:
        # Shock wave
        density_star_k = Primitive_State.Density * (
            (p_star + p0 + gmmm1/gmmp1 * (Primitive_State.Pressure + p0)) / 
            (Primitive_State.Pressure + p0 + gmmm1/gmmp1 * (p_star + p0))
        )
    else:
        # Rarefaction wave
        density_star_k = Primitive_State.Density * ((p_star + p0) / (Primitive_State.Pressure + p0))**(1.0/gamma)
    
    return density_star_k

def calc_left_shock_speed(Primitive_State, p_star, gamma, p0):
    """Calculate left shock speed"""
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    return Primitive_State.Velocity - speed_of_sound(Primitive_State, gamma, p0) * sqrt(
        gmmp1/(2.0*gamma) * (p_star + p0)/(Primitive_State.Pressure + p0) + gmmm1/(2.0*gamma)
    )

def calc_right_shock_speed(Primitive_State, p_star, gamma, p0):
    """Calculate right shock speed"""
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    
    return Primitive_State.Velocity + speed_of_sound(Primitive_State, gamma, p0) * sqrt(
        gmmp1/(2.0*gamma) * (p_star + p0)/(Primitive_State.Pressure + p0) + gmmm1/(2.0*gamma)
    )

def fan_left_state(left_state, x, t, gamma, p0):
    """Calculate state inside left rarefaction fan"""
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    c_left_val = speed_of_sound(left_state, gamma, p0)
    
    # Adapted for Tammann EOS
    rho = left_state.Density * (
        2.0/gmmp1 + gmmm1/(gmmp1*c_left_val)*(left_state.Velocity - x/t)
    )**(2.0/gmmm1)
    
    u = 2.0/gmmp1 * (c_left_val + gmmm1/2.0 * left_state.Velocity + x/t)
    
    # Pressure from density using isentropic relation for Tammann EOS
    p = (left_state.Pressure + p0) * (
        rho / left_state.Density
    )**gamma - p0
    
    return rho, u, p

def fan_right_state(right_state, x, t, gamma, p0):
    """Calculate state inside right rarefaction fan"""
    gmmp1 = gamma + 1.0
    gmmm1 = gamma - 1.0
    c_right_val = speed_of_sound(right_state, gamma, p0)
    
    # Adapted for Tammann EOS
    rho = right_state.Density * (
        2.0/gmmp1 - gmmm1/(gmmp1*c_right_val)*(right_state.Velocity - x/t)
    )**(2.0/gmmm1)
    
    u = 2.0/gmmp1 * (-c_right_val + gmmm1/2.0 * right_state.Velocity + x/t)
    
    # Pressure from density using isentropic relation for Tammann EOS
    p = (right_state.Pressure + p0) * (
        rho / right_state.Density
    )**gamma - p0
    
    return rho, u, p

def Riemann(left_state, right_state, gamma_left, gamma_right, p0_left, p0_right, x, t):
    """Solve the Riemann problem and return the solution at position x and time t"""
    # Estimate p*
    p_star_init = estimate_p_star(left_state, right_state, gamma_left, gamma_right, p0_left, p0_right)
    
    # Solve for p*
    args = (left_state, right_state, gamma_left, gamma_right, p0_left, p0_right)
    try:
        p_star = brentq(p_function, 1e-6, 1e6*p_star_init, args=args)
    except:
        print("Warning: brentq failed, using fsolve as fallback")
        p_star = fsolve(p_function, p_star_init, args=args)[0]
    
    # Calculate u*
    u_star = calc_u_star(p_star, left_state, right_state, gamma_left, gamma_right, p0_left, p0_right)
    
    # Calculate densities in star region
    density_star_left = calc_density_star(left_state, p_star, gamma_left, p0_left)
    density_star_right = calc_density_star(right_state, p_star, gamma_right, p0_right)
    
    # Determine wave structure
    if p_star > left_state.Pressure:
        # Left shock
        left_shock = True
        S_left = calc_left_shock_speed(left_state, p_star, gamma_left, p0_left)
        S_tail_left = np.nan
        S_head_left = np.nan
    else:
        # Left rarefaction
        left_shock = False
        S_left = np.nan
        c_left = speed_of_sound(left_state, gamma_left, p0_left)
        c_star_left = c_left * ((p_star + p0_left)/(left_state.Pressure + p0_left))**((gamma_left-1.0)/(2.0*gamma_left))
        S_tail_left = u_star - c_star_left
        S_head_left = left_state.Velocity - c_left
    
    if p_star > right_state.Pressure:
        # Right shock
        right_shock = True
        S_right = calc_right_shock_speed(right_state, p_star, gamma_right, p0_right)
        S_tail_right = np.nan
        S_head_right = np.nan
    else:
        # Right rarefaction
        right_shock = False
        S_right = np.nan
        c_right = speed_of_sound(right_state, gamma_right, p0_right)
        c_star_right = c_right * ((p_star + p0_right)/(right_state.Pressure + p0_right))**((gamma_right-1.0)/(2.0*gamma_right))
        S_tail_right = u_star + c_star_right
        S_head_right = right_state.Velocity + c_right
    
    # Determine the solution at position x and time t
    if x/t < u_star:
        if left_shock:
            if x/t < S_left:
                return left_state.Density, left_state.Velocity, left_state.Pressure
            else:
                return density_star_left, u_star, p_star
        else:
            if x/t < S_head_left:
                return left_state.Density, left_state.Velocity, left_state.Pressure
            elif x/t < S_tail_left:
                return fan_left_state(left_state, x, t, gamma_left, p0_left)
            else:
                return density_star_left, u_star, p_star
    else:
        if right_shock:
            if x/t > S_right:
                return right_state.Density, right_state.Velocity, right_state.Pressure
            else:
                return density_star_right, u_star, p_star
        else:
            if x/t > S_head_right:
                return right_state.Density, right_state.Velocity, right_state.Pressure
            elif x/t > S_tail_right:
                return fan_right_state(right_state, x, t, gamma_right, p0_right)
            else:
                return density_star_right, u_star, p_star

# Define state tuples
conserved_variables = ('Density', 'Momentum', 'Energy')
primitive_variables = ('Density', 'Velocity', 'Pressure')
Primitive_State = namedtuple('State', primitive_variables)
Conserved_State = namedtuple('State', conserved_variables)
State = Primitive_State

# Define test cases
if args.case:
    match args.case:
        case 'Tammann_water_air' | 'water_air':
            # Water-air interface test case
            conditions = 'Tammann_water_air'
            # Water (left)
            gmm_left = 7.15
            p0_left = 3.0e8  # Pa
            left_state = State(Density=1000.0, Velocity=0.0, Pressure=1.0e5)  # 1 bar
            # Air (right)
            gmm_right = 1.4
            p0_right = 0.0  # Pa
            right_state = State(Density=1.0, Velocity=0.0, Pressure=1.0e5)  # 1 bar
            t = 0.001  # s

        case 'Tammann_water_shock' | 'water_shock':
            # Water shock tube
            conditions = 'Tammann_water_shock'
            # Water (both sides)
            gmm_left = 7.15
            p0_left = 3.0e8  # Pa
            left_state = State(Density=1000.0, Velocity=0.0, Pressure=1.0e9)  # 10000 bar
            gmm_right = 7.15
            p0_right = 3.0e8  # Pa
            right_state = State(Density=1000.0, Velocity=0.0, Pressure=1.0e5)  # 1 bar
            t = 0.0001  # s

        case 'Tammann_cavitation' | 'cavitation':
            # Cavitation test case
            conditions = 'Tammann_cavitation'
            # Water (both sides)
            gmm_left = 7.15
            p0_left = 3.0e8  # Pa
            left_state = State(Density=1000.0, Velocity=-100.0, Pressure=1.0e5)  # 1 bar
            gmm_right = 7.15
            p0_right = 3.0e8  # Pa
            right_state = State(Density=1000.0, Velocity=100.0, Pressure=1.0e5)  # 1 bar
            t = 0.0005  # s

        case 'Tammann_bubble_expansion' | 'bubble_expansion':
            # Air bubble expansion in water
            conditions = 'Tammann_bubble_expansion'
            # Air (left)
            gmm_left = 1.4
            p0_left = 0.0  # Pa
            left_state = State(Density=1.0, Velocity=0.0, Pressure=2.0e5)  # 2 bar
            # Water (right)
            gmm_right = 7.15
            p0_right = 3.0e8  # Pa
            right_state = State(Density=1000.0, Velocity=0.0, Pressure=1.0e5)  # 1 bar
            t = 0.001  # s

        case 'Tammann_bubble_collapse' | 'bubble_collapse':
            # Air bubble collapse in water
            conditions = 'Tammann_bubble_collapse'
            # Air (left)
            gmm_left = 1.4
            p0_left = 0.0  # Pa
            left_state = State(Density=1.0, Velocity=0.0, Pressure=0.5e5)  # 0.5 bar
            # Water (right)
            gmm_right = 7.15
            p0_right = 3.0e8  # Pa
            right_state = State(Density=1000.0, Velocity=0.0, Pressure=1.0e5)  # 1 bar
            t = 0.001  # s

        case _:
            print(f"Unknown case: {args.case}")
            exit(1)

print('Running case: ' + conditions)

# Solve for p* and u*
p_star_init = estimate_p_star(left_state, right_state, gmm_left, gmm_right, p0_left, p0_right)
args_tuple = (left_state, right_state, gmm_left, gmm_right, p0_left, p0_right)

try:
    p_star = brentq(p_function, 1e-6, 1e6*p_star_init, args=args_tuple)
except:
    print("Warning: brentq failed, using fsolve as fallback")
    p_star = fsolve(p_function, p_star_init, args=args_tuple)[0]

u_star = calc_u_star(p_star, left_state, right_state, gmm_left, gmm_right, p0_left, p0_right)

# Calculate conserved variables
conserved_right = primitive2conserved(right_state, gmm_right, p0_right)
conserved_left = primitive2conserved(left_state, gmm_left, p0_left)

# Calculate densities in star region
density_star_left = calc_density_star(left_state, p_star, gmm_left, p0_left)
density_star_right = calc_density_star(right_state, p_star, gmm_right, p0_right)

# Determine wave structure
if p_star > left_state.Pressure:
    print('left shock')
    left_shock = True
    S_left = calc_left_shock_speed(left_state, p_star, gmm_left, p0_left)
    S_tail_left = np.nan
    S_head_left = np.nan
else:
    print('left expansion')
    S_left = np.nan
    left_shock = False
    c_left = speed_of_sound(left_state, gmm_left, p0_left)
    c_star_left = c_left * ((p_star + p0_left) / (left_state.Pressure + p0_left))**((gmm_left - 1.0) / (2.0 * gmm_left))
    S_tail_left = u_star - c_star_left
    S_head_left = left_state.Velocity - c_left

if p_star > right_state.Pressure:
    print('right shock')
    right_shock = True
    S_right = calc_right_shock_speed(right_state, p_star, gmm_right, p0_right)
    S_tail_right = np.nan
    S_head_right = np.nan
else:
    print('right expansion')
    S_right = np.nan
    right_shock = False
    c_right = speed_of_sound(right_state, gmm_right, p0_right)
    c_star_right = c_right * ((p_star + p0_right) / (right_state.Pressure + p0_right))**((gmm_right - 1.0) / (2.0 * gmm_right))
    S_tail_right = u_star + c_star_right
    S_head_right = right_state.Velocity + c_right

# Print solution information
print(f"p_star = {p_star:.6e} Pa")
print(f"u_star = {u_star:.6f} m/s")
print(f"density_star_left = {density_star_left:.6f} kg/m^3")
print(f"density_star_right = {density_star_right:.6f} kg/m^3")

# Plot exact Riemann solution
Nx = 1411
xvec = np.linspace(-1, 1, Nx)
statevec = np.zeros((Nx, 3))

for i, x in enumerate(xvec):
    statevec[i, 0], statevec[i, 1], statevec[i, 2] = Riemann(
        left_state, right_state, gmm_left, gmm_right, p0_left, p0_right, x, t
    )

# Make plots of exact solution
plt.close()
fig = plt.figure(1, figsize=(6, 5))
plt.plot(xvec, statevec[:, 0], '-b', label='Density')
plotextra = statevec[:, 0].max() - statevec[:, 0].min()
plotmin = statevec[:, 0].min() - 0.05 * plotextra
plotmax = statevec[:, 0].max() + 0.05 * plotextra

if not np.isnan(S_head_left):
    plt.plot([S_head_left * t, S_head_left * t], [plotmin, plotmax], '--g', label='Fan head')
if not np.isnan(S_tail_left):
    plt.plot([S_tail_left * t, S_tail_left * t], [plotmin, plotmax], '--c', label='Fan tail')
if not np.isnan(S_head_right):
    plt.plot([S_head_right * t, S_head_right * t], [plotmin, plotmax], '--g')
if not np.isnan(S_tail_right):
    plt.plot([S_tail_right * t, S_tail_right * t], [plotmin, plotmax], '--c')
plt.plot([u_star * t, u_star * t], [plotmin, plotmax], '--m', label='Contact')
if not np.isnan(S_right):
    plt.plot([S_right * t, S_right * t], [plotmin, plotmax], '--r', label='Shock')
if not np.isnan(S_left):
    plt.plot([S_left * t, S_left * t], [plotmin, plotmax], '--r')

plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$\rho$')
if args.plot2screen:
    plt.show()
if args.plot2file:
    fig.savefig(conditions + '_Riemann_density.eps')

# Velocity plot
plt.close()
fig = plt.figure(2, figsize=(6, 5))
plt.plot(xvec, statevec[:, 1], '-b', label='Velocity')
plotextra = statevec[:, 1].max() - statevec[:, 1].min()
plotmin = statevec[:, 1].min() - 0.05 * plotextra
plotmax = statevec[:, 1].max() + 0.05 * plotextra

if not np.isnan(S_head_left):
    plt.plot([S_head_left * t, S_head_left * t], [plotmin, plotmax], '--g', label='Fan head')
if not np.isnan(S_tail_left):
    plt.plot([S_tail_left * t, S_tail_left * t], [plotmin, plotmax], '--c', label='Fan tail')
if not np.isnan(S_head_right):
    plt.plot([S_head_right * t, S_head_right * t], [plotmin, plotmax], '--g')
if not np.isnan(S_tail_right):
    plt.plot([S_tail_right * t, S_tail_right * t], [plotmin, plotmax], '--c')
plt.plot([u_star * t, u_star * t], [plotmin, plotmax], '--m', label='Contact')
if not np.isnan(S_right):
    plt.plot([S_right * t, S_right * t], [plotmin, plotmax], '--r', label='Shock')
if not np.isnan(S_left):
    plt.plot([S_left * t, S_left * t], [plotmin, plotmax], '--r')

plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$U$')
if args.plot2screen:
    plt.show()
if args.plot2file:
    fig.savefig(conditions + '_Riemann_velocity.eps')

# Pressure plot
plt.close()
fig = plt.figure(3, figsize=(6, 5))
plt.plot(xvec, statevec[:, 2], '-b', label='Pressure')
plotextra = statevec[:, 2].max() - statevec[:, 2].min()
plotmin = statevec[:, 2].min() - 0.05 * plotextra
plotmax = statevec[:, 2].max() + 0.05 * plotextra

if not np.isnan(S_head_left):
    plt.plot([S_head_left * t, S_head_left * t], [plotmin, plotmax], '--g', label='Fan head')
if not np.isnan(S_tail_left):
    plt.plot([S_tail_left * t, S_tail_left * t], [plotmin, plotmax], '--c', label='Fan tail')
if not np.isnan(S_head_right):
    plt.plot([S_head_right * t, S_head_right * t], [plotmin, plotmax], '--g')
if not np.isnan(S_tail_right):
    plt.plot([S_tail_right * t, S_tail_right * t], [plotmin, plotmax], '--c')
plt.plot([u_star * t, u_star * t], [plotmin, plotmax], '--m', label='Contact')
if not np.isnan(S_right):
    plt.plot([S_right * t, S_right * t], [plotmin, plotmax], '--r', label='Shock')
if not np.isnan(S_left):
    plt.plot([S_left * t, S_left * t], [plotmin, plotmax], '--r')

plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$p$')
if args.plot2screen:
    plt.show()
if args.plot2file:
    fig.savefig(conditions + '_Riemann_pressure.eps')

# Internal energy plot
plt.close()
fig = plt.figure(4, figsize=(6, 5))
evec = np.zeros_like(xvec)
for j in range(len(xvec)):
    # Determine which side of the contact discontinuity we're on
    if xvec[j] < u_star * t:
        # Left side - use left gamma and p0
        x_state = State(Density=statevec[j, 0], Velocity=statevec[j, 1], Pressure=statevec[j, 2])
        evec[j] = internal_energy(x_state, gmm_left, p0_left)
    else:
        # Right side - use right gamma and p0
        x_state = State(Density=statevec[j, 0], Velocity=statevec[j, 1], Pressure=statevec[j, 2])
        evec[j] = internal_energy(x_state, gmm_right, p0_right)

plt.plot(xvec, evec, '-b', label='Internal energy')
plotextra = evec.max() - evec.min()
plotmin = evec.min() - 0.05 * plotextra
plotmax = evec.max() + 0.05 * plotextra

if not np.isnan(S_head_left):
    plt.plot([S_head_left * t, S_head_left * t], [plotmin, plotmax], '--g', label='Fan head')
if not np.isnan(S_tail_left):
    plt.plot([S_tail_left * t, S_tail_left * t], [plotmin, plotmax], '--c', label='Fan tail')
if not np.isnan(S_head_right):
    plt.plot([S_head_right * t, S_head_right * t], [plotmin, plotmax], '--g')
if not np.isnan(S_tail_right):
    plt.plot([S_tail_right * t, S_tail_right * t], [plotmin, plotmax], '--c')
plt.plot([u_star * t, u_star * t], [plotmin, plotmax], '--m', label='Contact')
if not np.isnan(S_right):
    plt.plot([S_right * t, S_right * t], [plotmin, plotmax], '--r', label='Shock')
if not np.isnan(S_left):
    plt.plot([S_left * t, S_left * t], [plotmin, plotmax], '--r')

plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$e$')
if args.plot2screen:
    plt.show()
if args.plot2file:
    fig.savefig(conditions + '_Riemann_internal_energy.eps')

# Sound speed plot
plt.close()
fig = plt.figure(5, figsize=(6, 5))
avec = np.zeros_like(xvec)
for j in range(len(xvec)):
    # Determine which side of the contact discontinuity we're on
    if xvec[j] < u_star * t:
        # Left side - use left gamma and p0
        x_state = State(Density=statevec[j, 0], Velocity=statevec[j, 1], Pressure=statevec[j, 2])
        avec[j] = speed_of_sound(x_state, gmm_left, p0_left)
    else:
        # Right side - use right gamma and p0
        x_state = State(Density=statevec[j, 0], Velocity=statevec[j, 1], Pressure=statevec[j, 2])
        avec[j] = speed_of_sound(x_state, gmm_right, p0_right)

plt.plot(xvec, avec, '-b', label='Sound speed')
plotextra = avec.max() - avec.min()
plotmin = avec.min() - 0.05 * plotextra
plotmax = avec.max() + 0.05 * plotextra

if not np.isnan(S_head_left):
    plt.plot([S_head_left * t, S_head_left * t], [plotmin, plotmax], '--g', label='Fan head')
if not np.isnan(S_tail_left):
    plt.plot([S_tail_left * t, S_tail_left * t], [plotmin, plotmax], '--c', label='Fan tail')
if not np.isnan(S_head_right):
    plt.plot([S_head_right * t, S_head_right * t], [plotmin, plotmax], '--g')
if not np.isnan(S_tail_right):
    plt.plot([S_tail_right * t, S_tail_right * t], [plotmin, plotmax], '--c')
plt.plot([u_star * t, u_star * t], [plotmin, plotmax], '--m', label='Contact')
if not np.isnan(S_right):
    plt.plot([S_right * t, S_right * t], [plotmin, plotmax], '--r', label='Shock')
if not np.isnan(S_left):
    plt.plot([S_left * t, S_left * t], [plotmin, plotmax], '--r')

plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$c$')
if args.plot2screen:
    plt.show()
if args.plot2file:
    fig.savefig(conditions + '_Riemann_sound_speed.eps')

# Save out data for this case
filename = 'Exact_Riemann_Tammann_cases.hdf5'
group_path = '/' + conditions

with h5py.File(filename, 'a') as f:
    if group_path in f:
        print(f"Group '{group_path}' found in '{filename}'. Deleting...")
        # Delete the group
        del f[group_path]
        print(f"Group '{group_path}' deleted successfully.")
    else:
        print(f"Group '{group_path}' does not exist in '{filename}'.")

    # Create the new dataset
    group = f.create_group(conditions)
    group.create_dataset('xvec', data=xvec)
    group.create_dataset('density', data=statevec[:, 0])
    group.create_dataset('velocity', data=statevec[:, 1])
    group.create_dataset('pressure', data=statevec[:, 2])
    group.create_dataset('internal_energy', data=evec)
    group.create_dataset('sound_speed', data=avec)
    group.create_dataset('S_right', data=S_right)
    group.create_dataset('S_left', data=S_left)
    group.create_dataset('right_shock', data=right_shock)
    group.create_dataset('left_shock', data=left_shock)
    group.create_dataset('c_right', data=c_right)
    group.create_dataset('c_left', data=c_left)
    group.create_dataset('c_star_left', data=c_star_left)
    group.create_dataset('c_star_right', data=c_star_right)
    group.create_dataset('S_tail_left', data=S_tail_left)
    group.create_dataset('S_tail_right', data=S_tail_right)
    group.create_dataset('S_head_right', data=S_head_right)
    group.create_dataset('S_head_left', data=S_head_left)
    group.create_dataset('u_star', data=u_star)
    group.create_dataset('p_star', data=p_star)
    group.create_dataset('density_star_left', data=density_star_left)
    group.create_dataset('density_star_right', data=density_star_right)
    group.create_dataset('gamma_left', data=gmm_left)
    group.create_dataset('gamma_right', data=gmm_right)
    group.create_dataset('p0_left', data=p0_left)
    group.create_dataset('p0_right', data=p0_right)
    
    print(f"Group '{group_path}' created/replaced successfully.")

print("\nExact Riemann solver for Tammann EOS completed successfully.")
print(f"Case: {conditions}")
print(f"Time: {t} seconds")
print(f"Left state: ρ={left_state.Density} kg/m³, u={left_state.Velocity} m/s, p={left_state.Pressure} Pa")
print(f"Right state: ρ={right_state.Density} kg/m³, u={right_state.Velocity} m/s, p={right_state.Pressure} Pa")
print(f"Left EOS: γ={gmm_left}, p₀={p0_left} Pa")
print(f"Right EOS: γ={gmm_right}, p₀={p0_right} Pa")
print(f"Solution: p*={p_star:.6e} Pa, u*={u_star:.6f} m/s")
print(f"Output saved to {filename}")
