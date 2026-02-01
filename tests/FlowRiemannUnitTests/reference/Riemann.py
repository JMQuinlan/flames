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

parser = argparse.ArgumentParser(description="Exact Riemann solver for regression test cases.")
parser.add_argument("--case", type=str, help="Case number")
parser.add_argument("--plot2screen", action="store_true", help="Pass to have plots rendered on screen, if possible")
parser.add_argument("--plot2file", action="store_true", help="Pass to have plots rendered file, if possible")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

args = parser.parse_args()

if args.case:
    print(f"Requested case = {args.case}")
else:
    args.case = 'Toro_case_6'
if args.verbose:
    print("Verbose mode is enabled.")

S_right     = np.nan
S_left     = np.nan
right_shock     = np.nan
left_shock     = np.nan
c_right     = np.nan
c_left     = np.nan
c_star_left     = np.nan
c_star_right     = np.nan
S_tail_left     = np.nan
S_tail_right     = np.nan
S_head_right     = np.nan
S_head_left     = np.nan
u_star          = np.nan

gmm = 1.4
gmmm1 = gmm - 1.
gmmp1 = gmm + 1.

def speed_of_sound(Primitive_State):
    return sqrt(gmm * Primitive_State.Pressure / Primitive_State.Density)

def internal_energy(Primitive_State):
    return Primitive_State.Pressure / (Primitive_State.Density * (gmm - 1.))

def stagnation_energy(Primitive_State):
    e = internal_energy(Primitive_State)
    return Primitive_State.Density*e + 0.5 * Primitive_State.Density * Primitive_State.Velocity**2

def primative2conserved(Primitive_State):
    Density = Primitive_State.Density
    Momentum = Primitive_State.Density * Primitive_State.Velocity
    Energy = stagnation_energy(Primitive_State)
    return Conserved_State(Density=Density, Momentum=Momentum, Energy=Energy)

def conserved2primative(Conserved_State):
    Density = Conserved_State.Density
    Velocity = Conserved_State.Momentum/Conserved_State.Density
    Pressure = (gmm - 1.) * (Conserved_State.Energy - 0.5 * Conserved_State.Density * Velocity**2)
    return Primitive_State(Density=Density, Velocity=Velocity, Pressure=Pressure)


def estimate_p_star(left_state, right_state):
    # Estimate p* with the two-rarefaction solution
    # Estimates check with Toro's test # 1 and # 2
    c_l = speed_of_sound(left_state)
    c_r = speed_of_sound(right_state)
    u_l = left_state.Velocity
    u_r = right_state.Velocity
    p_l = left_state.Pressure
    p_r = right_state.Pressure
    p_star_TR = ((c_l + c_r - 0.5*(gmmm1)*(u_r - u_l))
                 /(c_l/(p_l**(gmmm1/(2.*gmm))) + c_r/(p_r**(gmmm1/(2.*gmm)))))**(2.*gmm/gmmm1)
    if p_star_TR < 1e-6:
        print('p* estimate too small')
    return p_star_TR

def f_k_function(p,State):
    A_k = 2./(gmmp1*State.Density)
    B_k = gmmm1/gmmp1*State.Pressure
    if p > State.Pressure:
        f_k = (p - State.Pressure)*sqrt(A_k/(p+B_k))
    else:
        c_k = speed_of_sound(State)
        #print(p,State.Pressure)
        f_k = 2.*c_k/gmmm1 * ((p/State.Pressure)**(gmmm1/(2.*gmm)) - 1.)
    return f_k

def p_function(p,*states):
    Delta_u = right_state.Velocity - left_state.Velocity
    f = f_k_function(p,left_state) + f_k_function(p,right_state) + Delta_u
    return f

def calc_u_star(p_star,left_state, right_state):
    return 0.5*(left_state.Velocity + right_state.Velocity) + 0.5*(f_k_function(p_star,right_state) - f_k_function(p_star,left_state))

def calc_density_star(Primitive_State,p_star):
    if (p_star > Primitive_State.Pressure):
        density_star_k  = Primitive_State.Density*((p_star/Primitive_State.Pressure + gmmm1/gmmp1)
                                                /(gmmm1/gmmp1*p_star/Primitive_State.Pressure + 1.))
    else:
        density_star_k  = Primitive_State.Density*(p_star/Primitive_State.Pressure)**(1./gmm)
    return density_star_k

def calc_left_shock_speed(Primitive_State,p_star):
    return Primitive_State.Velocity - speed_of_sound(Primitive_State)*sqrt(gmmp1/(2.*gmm)*p_star/Primitive_State.Pressure + gmmm1/(2.*gmm))

def calc_right_shock_speed(Primitive_State,p_star):
    return Primitive_State.Velocity + speed_of_sound(Primitive_State)*sqrt(gmmp1/(2.*gmm)*p_star/Primitive_State.Pressure + gmmm1/(2.*gmm))

def fan_left_state(left_state,x,t):
    rho = left_state.Density * (2./gmmp1 + gmmm1/(gmmp1*c_left)*(left_state.Velocity - x/t))**(2./gmmm1)
    u = 2./gmmp1 * (c_left + gmmm1/2. * left_state.Velocity + x/t)
    p = left_state.Pressure * (2./gmmp1 + gmmm1/(gmmp1*c_left)*(left_state.Velocity - x/t))**((2.*gmm)/gmmm1)
    return rho, u, p

def fan_right_state(right_state,x,t):
    rho = right_state.Density * (2./gmmp1 - gmmm1/(gmmp1*c_right)*(right_state.Velocity - x/t))**(2./gmmm1)
    u = 2./gmmp1 * (- c_right + gmmm1/2. * right_state.Velocity + x/t)
    p = right_state.Pressure * (2./gmmp1 - gmmm1/(gmmp1*c_right)*(right_state.Velocity - x/t))**((2.*gmm)/gmmm1)
    return rho, u, p

def Riemann(left_state, right_state):
    return 1

conserved_variables = ('Density', 'Momentum', 'Energy')
primitive_variables = ('Density', 'Velocity', 'Pressure')

Primitive_State = namedtuple('State', primitive_variables)
Conserved_State = namedtuple('State', conserved_variables)

State = Primitive_State

if args.case:
    match args.case:
        case 'Toro_case_1_a' | 'c1a':
            # Toro Test2 - Table 4.1 - Page 129
            # Toro test # 1 - Pass
            conditions = 'Toro_case_1_a'
            left_state = State(Density = 1., Velocity = 0., Pressure = 1.)
            right_state = State(Density = 0.125, Velocity = 0., Pressure = 0.1)
            t = 0.25
        case 'Toro_case_1_b' | 'c1b' | 'c1':
            # Toro Tests - Table 10.1 - Page 334
            # Toro test # 1 -
            conditions = 'Toro_case_1_b'
            left_state = State(Density = 1., Velocity = 0.75, Pressure = 1.)
            right_state = State(Density = 0.125, Velocity = 0., Pressure = 0.1)
            t = 0.2

        case 'Toro_case_1_a_rev' | 'c1r':
            # Toro test # 1_rev - Pass
            conditions = 'Toro_case_1_a_rev'
            left_state = State(Density = 0.125, Velocity = 0., Pressure = 0.1)
            right_state = State(Density = 1., Velocity = 0., Pressure = 1.)
            t = 0.25

        case 'Toro_case_2' | 'c2':
            # Toro test # 2 - Pass
            conditions = 'Toro_case_2'
            left_state = State(Density = 1., Velocity = -2., Pressure = 0.4)
            right_state = State(Density = 1., Velocity = 2., Pressure = 0.4)
            t = 0.15

        case 'Toro_case_3' | 'c3':
            # Toro test # 3 - Pass
            conditions = 'Toro_case_3'
            left_state = State(Density = 1., Velocity = 0., Pressure = 1000.)
            right_state = State(Density = 1., Velocity = 0., Pressure = 0.01)
            t = 0.012

        case 'Toro_case_4_a' | 'c4a':
            # Toro test # 4a - Pass
            conditions = 'Toro_case_4_a'
            left_state = State(Density = 1., Velocity = 0., Pressure = 0.01)
            right_state = State(Density = 1., Velocity = 0., Pressure = 100.)
            t = 0.035

        case 'Toro_case_4_b' | 'c4b' | 'c4':
            # Toro test # 4b -
            conditions = 'Toro_case_4_b'
            left_state = State(Density = 5.99924, Velocity = 19.5975, Pressure = 460.894)
            right_state = State(Density = 5.99242, Velocity = -6.19633, Pressure = 46.0950)
            t = 0.035

        case 'Toro_case_5' | 'c5':
            # Toro test # 5 -
            conditions = 'Toro_case_5'
            left_state =  State(Density = 1.0, Velocity = -19.59745, Pressure = 1000.0)
            right_state = State(Density = 1.0, Velocity = -19.59745, Pressure = 0.01)
            t = 0.035

        case 'Toro_case_6' | 'c6':
            # Toro test # 6 -
            conditions = 'Toro_case_6'
            left_state =  State(Density = 1.4, Velocity = 0.0, Pressure = 1.0)
            right_state = State(Density = 1.0,   Velocity = 0.0, Pressure = 1.0)
            t = 2.0

        case 'Toro_case_7' | 'c7':
            # Toro test # 7 -
            conditions = 'Toro_case_7'
            left_state =  State(Density = 1.4, Velocity = 0.1, Pressure = 1.0)
            right_state = State(Density = 1.0,   Velocity = 0.1, Pressure = 1.0)
            t = 2.0

        case _:
            raise ValueError("Unknown case requested")


print('Running case: '+conditions)

p_star_init = estimate_p_star(left_state, right_state)

states = (left_state,right_state)
#p_star = fsolve(p_function, p_star_init, args=states)
#p_star = least_squares(p_function, p_star_init, args=states, bounds=(0,np.inf))
p_star = brentq(p_function,1.e-6,1.e6*p_star_init, args=states)
u_star = calc_u_star(p_star,left_state, right_state)

conserved_right = primative2conserved(right_state)
conserved_left  = primative2conserved( left_state)

density_star_left  = calc_density_star(left_state,p_star)
density_star_right = calc_density_star(right_state,p_star)

S_left  = np.nan
S_right = np.nan

if (p_star > left_state.Pressure):
    print('left shock')
    left_shock = True
    S_left = calc_left_shock_speed(left_state,p_star)
    S_tail_left = np.nan
    S_head_left = np.nan
else:
    print('left expansion')
    S_left = np.nan
    left_shock = False
    c_left      = speed_of_sound(left_state)
    c_star_left = c_left*(p_star/left_state.Pressure)**(gmmm1/(2.*gmm))
    S_tail_left = u_star - c_star_left
    S_head_left = left_state.Velocity - c_left

if (p_star > right_state.Pressure):
    print('right shock')
    right_shock = True
    S_right = calc_right_shock_speed(right_state,p_star)
    S_tail_right = np.nan
    S_head_right = np.nan
else:
    print('right expansion')
    S_right = np.nan
    right_shock = False
    c_right      = speed_of_sound(right_state)
    c_star_right = c_right*(p_star/right_state.Pressure)**(gmmm1/(2.*gmm))
    S_tail_right = u_star + c_star_right
    S_head_right = right_state.Velocity + c_right


# Plot exact Riemann solution for xvec and tvec (maybe just at t for now)
Nx = 1411
xvec = np.linspace(-1, 1, Nx)

statevec = np.array([]).reshape(0,3)

for x in xvec:
    #print(x)
    if x<u_star*t:
        if left_shock:
            if x<S_left*t:
                statevec = np.concatenate((statevec,np.array([[left_state.Density, left_state.Velocity, left_state.Pressure]])),axis = 0)
            else:
                statevec = np.concatenate((statevec,np.array([[density_star_left, u_star, p_star]])),axis = 0)
        else:
            if x<S_head_left*t:
                statevec = np.concatenate((statevec,np.array([[left_state.Density, left_state.Velocity, left_state.Pressure]])),axis = 0)
            elif x<S_tail_left*t:
                statevec = np.concatenate((statevec,[np.squeeze([fan_left_state(left_state,x,t)])]),axis = 0)
            else:
                statevec = np.concatenate((statevec,np.array([[density_star_left, u_star, p_star]])),axis = 0)
    else:
        if right_shock:
            if x>S_right*t:
                statevec = np.concatenate((statevec,np.array([[right_state.Density, right_state.Velocity, right_state.Pressure]])),axis = 0)
            else:
                statevec = np.concatenate((statevec,np.array([[density_star_right, u_star, p_star]])),axis = 0)
        else:
            if x>S_head_right*t:
                statevec = np.concatenate((statevec,np.array([[right_state.Density, right_state.Velocity, right_state.Pressure]])),axis = 0)
            elif x>S_tail_right*t:
                statevec = np.concatenate((statevec,[np.squeeze([fan_right_state(right_state,x,t)])]),axis = 0)
            else:
                statevec = np.concatenate((statevec,np.array([[density_star_right, u_star, p_star]])),axis = 0)

# Make plots of exact solution
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.set_loglevel("Error")

plt.close()
fig = plt.figure(32,figsize=(6,5))
plt.plot(xvec,statevec[:,0],'-b',label='Density')
plotextra = statevec[:,0].max() - statevec[:,0].min()
plotmin = statevec[:,0].min() - 0.05*plotextra
plotmax = statevec[:,0].max() + 0.05*plotextra
plt.plot([S_head_left*t,S_head_left*t],[plotmin,plotmax],'--g',label='Fan head')
plt.plot([S_tail_left*t,S_tail_left*t],[plotmin,plotmax],'--c',label='Fan tail')
plt.plot([S_head_right*t,S_head_right*t],[plotmin,plotmax],'--g')
plt.plot([S_tail_right*t,S_tail_right*t],[plotmin,plotmax],'--c')
plt.plot([u_star*t,u_star*t],[plotmin,plotmax],'--m',label='Contact')
plt.plot([S_right*t,S_right*t],[plotmin,plotmax],'--r',label='Shock')
plt.plot([S_left*t,S_left*t],[plotmin,plotmax],'--r')
plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$\rho$')
if args.plot2screen: plt.show()
if args.plot2file: fig.savefig(conditions+'_Riemann_density.eps')
# or output pdf file (just don't waste your time outputting jpg or png files)
#fig.savefig('Riemann_density.pdf')

plt.close()
fig = plt.figure(32,figsize=(6,5))
plt.plot(xvec,statevec[:,1],'-b',label='Velocity')
plotextra = statevec[:,1].max() - statevec[:,1].min()
plotmin = statevec[:,1].min() - 0.05*plotextra
plotmax = statevec[:,1].max() + 0.05*plotextra
plt.plot([S_head_left*t,S_head_left*t],[plotmin,plotmax],'--g',label='Fan head')
plt.plot([S_tail_left*t,S_tail_left*t],[plotmin,plotmax],'--c',label='Fan tail')
plt.plot([S_head_right*t,S_head_right*t],[plotmin,plotmax],'--g')
plt.plot([S_tail_right*t,S_tail_right*t],[plotmin,plotmax],'--c')
plt.plot([u_star*t,u_star*t],[plotmin,plotmax],'--m',label='Contact')
plt.plot([S_right*t,S_right*t],[plotmin,plotmax],'--r',label='Shock')
plt.plot([S_left*t,S_left*t],[plotmin,plotmax],'--r')
plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$U$')
if args.plot2screen: plt.show()
if args.plot2file: fig.savefig(conditions+'_Riemann_velocity.eps')

plt.close()
fig = plt.figure(32,figsize=(6,5))
plt.plot(xvec,statevec[:,2],'-b',label='Pressure')
plotextra = statevec[:,2].max() - statevec[:,2].min()
plotmin = statevec[:,2].min() - 0.05*plotextra
plotmax = statevec[:,2].max() + 0.05*plotextra
plt.plot([S_head_left*t,S_head_left*t],[plotmin,plotmax],'--g',label='Fan head')
plt.plot([S_tail_left*t,S_tail_left*t],[plotmin,plotmax],'--c',label='Fan tail')
plt.plot([S_head_right*t,S_head_right*t],[plotmin,plotmax],'--g')
plt.plot([S_tail_right*t,S_tail_right*t],[plotmin,plotmax],'--c')
plt.plot([u_star*t,u_star*t],[plotmin,plotmax],'--m',label='Contact')
plt.plot([S_right*t,S_right*t],[plotmin,plotmax],'--r',label='Shock')
plt.plot([S_left*t,S_left*t],[plotmin,plotmax],'--r')
plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$p$')
if args.plot2screen: plt.show()
if args.plot2file: fig.savefig(conditions+'_Riemann_pressure.eps')

plt.close()
fig = plt.figure(32,figsize=(6,5))
evec = np.zeros_like(xvec)
for j in range(len(xvec)):
    x_state = State(Density = statevec[j,0], Velocity = statevec[j,1], Pressure = statevec[j,2])
    evec[j] = internal_energy(x_state)
plt.plot(xvec,evec,'-b',label='Internal energy')
plotextra = evec.max() - evec.min()
plotmin = evec.min() - 0.05*plotextra
plotmax = evec.max() + 0.05*plotextra
plt.plot([S_head_left*t,S_head_left*t],[plotmin,plotmax],'--g',label='Fan head')
plt.plot([S_tail_left*t,S_tail_left*t],[plotmin,plotmax],'--c',label='Fan tail')
plt.plot([S_head_right*t,S_head_right*t],[plotmin,plotmax],'--g')
plt.plot([S_tail_right*t,S_tail_right*t],[plotmin,plotmax],'--c')
plt.plot([u_star*t,u_star*t],[plotmin,plotmax],'--m',label='Contact')
plt.plot([S_right*t,S_right*t],[plotmin,plotmax],'--r',label='Shock')
plt.plot([S_left*t,S_left*t],[plotmin,plotmax],'--r')
plt.legend(loc=6)
plt.xlabel('x')
plt.xlim([-1, 1])
plt.ylabel(r'$e$')
if args.plot2screen: plt.show()
if args.plot2file: fig.savefig(conditions+'_Riemann_internal_energy.eps')

plt.close()

# Save out data for this case (append as necessary)
filename = 'Exact_Riemann_cases.hdf5'
group_path = '/'+conditions
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
    group.create_dataset('xvec',            data = xvec)
    group.create_dataset('density',         data = statevec[:,0])
    group.create_dataset('velocity',        data = statevec[:,1])
    group.create_dataset('pressure',        data = statevec[:,2])
    group.create_dataset('internal_energy', data = evec)
    group.create_dataset('S_right', data = S_right)
    group.create_dataset('S_left', data = S_left)
    group.create_dataset('right_shock', data = right_shock)
    group.create_dataset('left_shock', data = left_shock)
    group.create_dataset('c_right', data = c_right)
    group.create_dataset('c_left', data = c_left)
    group.create_dataset('c_star_left', data = c_star_left)
    group.create_dataset('c_star_right', data = c_star_right)
    group.create_dataset('S_tail_left', data = S_tail_left)
    group.create_dataset('S_tail_right', data = S_tail_right)
    group.create_dataset('S_head_right', data = S_head_right)
    group.create_dataset('S_head_left', data = S_head_left)
    group.create_dataset('u_star', data = u_star)
    print(f"Group '{group_path}' created/replaced successfully.")

