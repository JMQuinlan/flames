"""
Exact Riemann solutions for comparison with 1-d finite volume solvers

Started 20250818 by MQ
All rights reserved

This code reads in the exact Riemann solutions from an hdf5 file
Plots the exact solutions, including the wave structure/interfaces
Reads in the numerical solution produced by FLAMES
Plots against the exact solutions
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
            #left_state = State(Density = 1., Velocity = -2., Pressure = 0.4)
            #right_state = State(Density = 1., Velocity = 2., Pressure = 0.4)
            t = 0.15

        case 'Toro_case_3' | 'c3':
            # Toro test # 3 - Pass
            conditions = 'Toro_case_3'
            #left_state = State(Density = 1., Velocity = 0., Pressure = 1000.)
            #right_state = State(Density = 1., Velocity = 0., Pressure = 0.01)
            t = 0.012

        case 'Toro_case_4_a' | 'c4a':
            # Toro test # 4a - Pass
            conditions = 'Toro_case_4_a'
            #left_state = State(Density = 1., Velocity = 0., Pressure = 0.01)
            #right_state = State(Density = 1., Velocity = 0., Pressure = 100.)
            t = 0.035

        case 'Toro_case_4_b' | 'c4b' | 'c4':
            # Toro test # 4b -
            conditions = 'Toro_case_4_b'
            #left_state = State(Density = 5.99924, Velocity = 19.5975, Pressure = 460.894)
            #right_state = State(Density = 5.99242, Velocity = -6.19633, Pressure = 46.0950)
            t = 0.035

        case 'Toro_case_5' | 'c5':
            # Toro test # 5 -
            conditions = 'Toro_case_5'
            #left_state =  State(Density = 1, Velocity = -19.59745, Pressure = 1000.0)
            #right_state = State(Density = 1, Velocity = -19.59745, Pressure = 0.01)
            t = 0.035

        case 'Toro_case_6' | 'c6':
            # Toro test # 6 -
            conditions = 'Toro_case_6'
            #left_state =  State(Density = 1.4, Velocity = 0.0, Pressure = 1.0)
            #right_state = State(Density = 1,   Velocity = 0.0, Pressure = 1.0)
            t = 2.0

        case 'Toro_case_7' | 'c7':
            # Toro test # 7 -
            conditions = 'Toro_case_7'
            left_state =  State(Density = 1.4, Velocity = 0.1, Pressure = 1.0)
            right_state = State(Density = 1,   Velocity = 0.1, Pressure = 1.0)
            t = 2.0

        case _:
            Error

print('Running case: '+conditions)

with h5py.File('Exact_Riemann_cases.hdf5', 'r') as f:
    for group_name in f.keys():
        if isinstance(f[group_name], h5py.Group):
            group = f[group_name]
            print(f"Group: {group_name}")
    dataset = f['/'+conditions+'/xvec']
    xvec = dataset[:]
    dataset = f['/'+conditions+'/density']
    density = dataset[:]
    dataset = f['/'+conditions+'/velocity']
    velocity = dataset[:]
    dataset = f['/'+conditions+'/pressure']
    pressure = dataset[:]
    dataset = f['/'+conditions+'/internal_energy']
    internal_energy = dataset[:]
    S_right     = f['/'+conditions+'/S_right'][()]
    S_left      = f['/'+conditions+'/S_left'][()]
    right_shock = f['/'+conditions+'/right_shock'][()]
    left_shock  = f['/'+conditions+'/left_shock'][()]
    c_right     = f['/'+conditions+'/c_right'][()]
    c_left      = f['/'+conditions+'/c_left'][()]
    c_star_left = f['/'+conditions+'/c_star_left'][()]
    c_star_right= f['/'+conditions+'/c_star_right'][()]
    S_tail_left = f['/'+conditions+'/S_tail_left'][()]
    S_tail_right= f['/'+conditions+'/S_tail_right'][()]
    S_head_right= f['/'+conditions+'/S_head_right'][()]
    S_head_left = f['/'+conditions+'/S_head_left'][()]
    u_star      = f['/'+conditions+'/u_star'][()]
#

### Read in numerical solution ####




### Make plots of exact solution and numerical solution ####
from matplotlib import pyplot
import matplotlib.pyplot as plt
plt.set_loglevel("Error")

plt.close()
fig = plt.figure(32,figsize=(6,5))
plt.plot(xvec,density,'-b',label='Density')
plotextra = density.max() - density.min()
plotmin = density.min() - 0.05*plotextra
plotmax = density.max() + 0.05*plotextra
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
plt.plot(xvec,velocity,'-b',label='Velocity')
plotextra = velocity.max() - velocity.min()
plotmin = velocity.min() - 0.05*plotextra
plotmax = velocity.max() + 0.05*plotextra
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
plt.plot(xvec,pressure,'-b',label='Pressure')
plotextra = pressure.max() - pressure.min()
plotmin = pressure.min() - 0.05*plotextra
plotmax = pressure.max() + 0.05*plotextra
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
plt.plot(xvec,internal_energy,'-b',label='Internal energy')
plotextra = internal_energy.max() - internal_energy.min()
plotmin = internal_energy.min() - 0.05*plotextra
plotmax = internal_energy.max() + 0.05*plotextra
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
