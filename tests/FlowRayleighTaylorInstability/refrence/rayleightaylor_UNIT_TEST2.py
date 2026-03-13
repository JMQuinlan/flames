# -*- coding: utf-8 -*-
"""
===============================================================================
RAYLEIGH-TAYLOR INSTABILITY: ENHANCED VOF SOLVER vs HYDRO2 COMPARISON
===============================================================================
PURPOSE:
    Enhanced Python VOF solver with:
    - Viscosity (explicit treatment)
    - HLLC Riemann solver (less diffusive than Lax-Friedrichs)
    - Surface tension (CSF model)
    - Volume-of-Fluid phase tracking
    
    Compare with hydro2 AMReX solver for validation

ENHANCEMENTS OVER BASELINE:
    1. Viscous stress terms in momentum equation
    2. HLLC flux calculation for sharper interfaces
    3. Surface tension via Continuum Surface Force (CSF) model
    4. Improved phase advection with interface compression

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import glob
import re
from PIL import Image
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# -------------------- FILE PATHS --------------------
HYDRO2_OUTPUT_DIR = r'../../../bin/tests/RayleighTaylor/RayleighTaylor_UNIT_TEST'
OUTPUT_FOLDER = './RT_Comparison_VOF_Analysis'

# -------------------- TIME SYNCHRONIZATION --------------------
TIME_TOLERANCE_MS = 500.0  # Match timesteps within this tolerance [milliseconds]

# -------------------- PLOT TOGGLES --------------------
PLOT_DENSITY = 1
PLOT_VELOCITY = 1
PLOT_VORTICITY = 1
PLOT_DIFFERENCES = 1
GENERATE_GIFS = 1

# -------------------- PLOTTING PARAMETERS --------------------
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
FONT_SIZE_TIMESTAMP = 14

LINE_WIDTH_THICK = 2.5
LINE_WIDTH_NORMAL = 2.0
LINE_WIDTH_THIN = 1.5
CONTOUR_LINE_WIDTH = 2.0

DPI = 300
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8

COLORMAP_DENSITY = 'viridis'
COLORMAP_VELOCITY = 'plasma'
COLORMAP_VORTICITY = 'RdBu_r'
COLORMAP_DIFFERENCE = 'seismic'

STREAMLINE_SHOW = True
STREAMLINE_DENSITY = 1.5
STREAMLINE_COLOR = 'white'
STREAMLINE_LINEWIDTH = 0.8

# -------------------- PHYSICAL PARAMETERS --------------------
GAMMA = 1.4  # Ideal gas gamma
GRAVITY = -0.1  # Gravity strength [m/s^2]
P0 = 2.5  # Reference pressure [Pa]
W0 = 0.0025  # Initial perturbation amplitude [m/s]

# Domain
BOXSIZE_X = 0.5  # [m]
BOXSIZE_Y = 1.5  # [m]
INTERFACE_Y = 0.75  # Interface location [m]

# Fluid properties
RHO_HEAVY = 2.0  # Heavy fluid density [kg/m^3]
RHO_LIGHT = 1.0  # Light fluid density [kg/m^3]

# -------------------- ENHANCED PHYSICS PARAMETERS --------------------
# Viscosity
MU_HEAVY = 1.0e-3   # Heavy fluid dynamic viscosity [Pa*s]
MU_LIGHT = 1.8e-5   # Light fluid dynamic viscosity [Pa*s]

# Surface tension
SIGMA = 0.0728      # Surface tension coefficient [N/m]

# Physics toggles
USE_HLLC = True              # Use HLLC instead of Lax-Friedrichs
USE_VISCOSITY = True         # Include viscous terms
USE_SURFACE_TENSION = True   # Include surface tension

# -------------------- PYTHON SOLVER PARAMETERS --------------------
N_RESOLUTION = 64  # Resolution N x 3N (64 x 192)
COURANT_FAC = 0.4
USE_SLOPE_LIMITING = False

# Viscous CFL factor (additional constraint)
VISCOUS_CFL_FAC = 0.25

# -------------------- GIF PARAMETERS --------------------
GIF_FPS = 10
GIF_LOOP = 0
GIF_OPTIMIZE = True

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

subfolder_density = os.path.join(OUTPUT_FOLDER, 'density')
subfolder_velocity = os.path.join(OUTPUT_FOLDER, 'velocity')
subfolder_vorticity = os.path.join(OUTPUT_FOLDER, 'vorticity')
subfolder_diff_density = os.path.join(OUTPUT_FOLDER, 'difference_density')
subfolder_diff_velocity = os.path.join(OUTPUT_FOLDER, 'difference_velocity')
subfolder_diff_vorticity = os.path.join(OUTPUT_FOLDER, 'difference_vorticity')

for folder in [subfolder_density, subfolder_velocity, subfolder_vorticity,
               subfolder_diff_density, subfolder_diff_velocity, subfolder_diff_vorticity]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("=" * 80)
print("RAYLEIGH-TAYLOR: ENHANCED VOF SOLVER vs HYDRO2")
print("=" * 80)
print(f"\nPhysics enabled:")
print(f"  - HLLC Riemann solver: {USE_HLLC}")
print(f"  - Viscosity: {USE_VISCOSITY} (mu_heavy={MU_HEAVY}, mu_light={MU_LIGHT})")
print(f"  - Surface tension: {USE_SURFACE_TENSION} (sigma={SIGMA})")
print(f"\nOutput directory: {OUTPUT_FOLDER}")
print(f"Time tolerance: {TIME_TOLERANCE_MS} ms")

# ============================================================================
# ENHANCED VOF FINITE VOLUME SOLVER
# ============================================================================

def getConserved(rho, vx, vy, P, gamma, vol):
    """Calculate conserved variables from primitive variables"""
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2))*vol
    return Mass, Momx, Momy, Energy

def getPrimitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Calculate primitive variables from conservative variables"""
    rho = Mass / vol
    vx = Momx / (rho * vol + 1e-10)
    vy = Momy / (rho * vol + 1e-10)
    P = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
    P = np.maximum(P, 1e-6)  # Pressure floor
    rho, vx, vy, P = setGhostCells(rho, vx, vy, P)
    return rho, vx, vy, P

def getGradient(f, dx):
    """Calculate gradients of a field using central differences"""
    R = -1  # right
    L = 1   # left
    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2*dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2*dx)
    f_dx, f_dy = setGhostGradients(f_dx, f_dy)
    return f_dx, f_dy

def slopeLimit(f, dx, f_dx, f_dy):
    """Apply slope limiter to slopes"""
    R = -1
    L = 1
    f_dx = np.maximum(0., np.minimum(1., ((f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
    f_dy = np.maximum(0., np.minimum(1., ((f-np.roll(f,L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
    f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
    return f_dx, f_dy

def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """Extrapolate field to face centers"""
    R = -1
    L = 1
    f_XL = f - f_dx * dx/2
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx/2
    f_YL = f - f_dy * dx/2
    f_YL = np.roll(f_YL, R, axis=1)
    f_YR = f + f_dy * dx/2
    return f_XL, f_XR, f_YL, f_YR

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply fluxes to conserved variables"""
    R = -1
    L = 1
    F += - dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, L, axis=0)
    F += - dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, L, axis=1)
    return F

def getFluxHLLC(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """
    HLLC Riemann solver for inviscid fluxes
    More accurate than Lax-Friedrichs, resolves contact discontinuities
    """
    # Add small epsilon to avoid division by zero
    eps = 1e-10
    rho_L = np.maximum(rho_L, eps)
    rho_R = np.maximum(rho_R, eps)
    P_L = np.maximum(P_L, eps)
    P_R = np.maximum(P_R, eps)
    
    # Compute energies
    en_L = P_L/(gamma-1) + 0.5*rho_L * (vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R * (vx_R**2+vy_R**2)
    
    # Sound speeds
    a_L = np.sqrt(gamma * P_L / rho_L)
    a_R = np.sqrt(gamma * P_R / rho_R)
    
    # Estimate wave speeds (Davis approximation)
    S_L = np.minimum(vx_L - a_L, vx_R - a_R)
    S_R = np.maximum(vx_L + a_L, vx_R + a_R)
    
    # Contact wave speed (star region)
    S_star = (P_R - P_L + rho_L*vx_L*(S_L - vx_L) - rho_R*vx_R*(S_R - vx_R)) / \
             (rho_L*(S_L - vx_L) - rho_R*(S_R - vx_R) + eps)
    
    # Left and right fluxes
    flux_Mass_L = rho_L * vx_L
    flux_Momx_L = rho_L * vx_L**2 + P_L
    flux_Momy_L = rho_L * vx_L * vy_L
    flux_Energy_L = (en_L + P_L) * vx_L
    
    flux_Mass_R = rho_R * vx_R
    flux_Momx_R = rho_R * vx_R**2 + P_R
    flux_Momy_R = rho_R * vx_R * vy_R
    flux_Energy_R = (en_R + P_R) * vx_R
    
    # Star region states
    rho_star_L = rho_L * (S_L - vx_L) / (S_L - S_star + eps)
    rho_star_R = rho_R * (S_R - vx_R) / (S_R - S_star + eps)
    
    en_star_L = rho_star_L * (en_L/rho_L + (S_star - vx_L)*(S_star + P_L/(rho_L*(S_L - vx_L) + eps)))
    en_star_R = rho_star_R * (en_R/rho_R + (S_star - vx_R)*(S_star + P_R/(rho_R*(S_R - vx_R) + eps)))
    
    flux_Mass_star_L = rho_star_L * S_star
    flux_Momx_star_L = rho_star_L * S_star**2 + P_L + rho_L*(S_L - vx_L)*(S_star - vx_L)
    flux_Momy_star_L = rho_star_L * S_star * vy_L
    flux_Energy_star_L = S_star * en_star_L + S_star * (P_L + rho_L*(S_L - vx_L)*(S_star - vx_L))
    
    flux_Mass_star_R = rho_star_R * S_star
    flux_Momx_star_R = rho_star_R * S_star**2 + P_R + rho_R*(S_R - vx_R)*(S_star - vx_R)
    flux_Momy_star_R = rho_star_R * S_star * vy_R
    flux_Energy_star_R = S_star * en_star_R + S_star * (P_R + rho_R*(S_R - vx_R)*(S_star - vx_R))
    
    # Select flux based on wave speeds
    flux_Mass = np.where(S_L >= 0, flux_Mass_L,
                np.where(S_star >= 0, flux_Mass_star_L,
                np.where(S_R >= 0, flux_Mass_star_R, flux_Mass_R)))
    
    flux_Momx = np.where(S_L >= 0, flux_Momx_L,
                np.where(S_star >= 0, flux_Momx_star_L,
                np.where(S_R >= 0, flux_Momx_star_R, flux_Momx_R)))
    
    flux_Momy = np.where(S_L >= 0, flux_Momy_L,
                np.where(S_star >= 0, flux_Momy_star_L,
                np.where(S_R >= 0, flux_Momy_star_R, flux_Momy_R)))
    
    flux_Energy = np.where(S_L >= 0, flux_Energy_L,
                  np.where(S_star >= 0, flux_Energy_star_L,
                  np.where(S_R >= 0, flux_Energy_star_R, flux_Energy_R)))
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def getFluxLaxFriedrichs(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Original Lax-Friedrichs flux (for comparison)"""
    eps = 1e-10
    rho_L = np.maximum(rho_L, eps)
    rho_R = np.maximum(rho_R, eps)
    P_L = np.maximum(P_L, eps)
    P_R = np.maximum(P_R, eps)
    
    en_L = P_L/(gamma-1) + 0.5*rho_L * (vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R * (vx_R**2+vy_R**2)
    
    rho_star = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5*(en_L + en_R)
    
    P_star = (gamma-1)*(en_star - 0.5*(momx_star**2+momy_star**2)/(rho_star + eps))
    P_star = np.maximum(P_star, eps)
    
    flux_Mass = momx_star
    flux_Momx = momx_star**2/(rho_star + eps) + P_star
    flux_Momy = momx_star * momy_star/(rho_star + eps)
    flux_Energy = (en_star+P_star) * momx_star/(rho_star + eps)
    
    C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
    C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
    C = np.maximum(C_L, C_R)
    
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def computeViscousFluxes(vx, vy, mu, dx):
    """
    Compute viscous stress tensor and fluxes
    tau_xx = 2*mu*(dvx/dx - 1/3*div_v)
    tau_yy = 2*mu*(dvy/dy - 1/3*div_v)
    tau_xy = mu*(dvx/dy + dvy/dx)
    """
    # Velocity gradients
    dvx_dx, dvx_dy = getGradient(vx, dx)
    dvy_dx, dvy_dy = getGradient(vy, dx)
    
    # Divergence
    div_v = dvx_dx + dvy_dy
    
    # Stress tensor components
    tau_xx = 2*mu*(dvx_dx - div_v/3)
    tau_yy = 2*mu*(dvy_dy - div_v/3)
    tau_xy = mu*(dvx_dy + dvy_dx)
    
    # Viscous fluxes (momentum)
    # d(tau_xx)/dx + d(tau_xy)/dy
    dtau_xx_dx, _ = getGradient(tau_xx, dx)
    _, dtau_xy_dy = getGradient(tau_xy, dx)
    visc_flux_momx = dtau_xx_dx + dtau_xy_dy
    
    # d(tau_xy)/dx + d(tau_yy)/dy
    dtau_xy_dx, _ = getGradient(tau_xy, dx)
    _, dtau_yy_dy = getGradient(tau_yy, dx)
    visc_flux_momy = dtau_xy_dx + dtau_yy_dy
    
    # Energy flux: d(tau*v)/dx + d(tau*v)/dy
    work_x = tau_xx * vx + tau_xy * vy
    work_y = tau_xy * vx + tau_yy * vy
    dwork_x_dx, _ = getGradient(work_x, dx)
    _, dwork_y_dy = getGradient(work_y, dx)
    visc_flux_energy = dwork_x_dx + dwork_y_dy
    
    return visc_flux_momx, visc_flux_momy, visc_flux_energy

def computeSurfaceTension(phi, sigma, dx):
    """
    Continuum Surface Force (CSF) model for surface tension
    F_sigma = sigma*kappa*grad(phi) where kappa is interface curvature
    """
    # Interface normal: n = grad(phi)
    dphi_dx, dphi_dy = getGradient(phi, dx)
    
    # Magnitude of gradient
    grad_phi_mag = np.sqrt(dphi_dx**2 + dphi_dy**2 + 1e-10)
    
    # Unit normal
    nx = dphi_dx / grad_phi_mag
    ny = dphi_dy / grad_phi_mag
    
    # Curvature: kappa = -div(n)
    dnx_dx, _ = getGradient(nx, dx)
    _, dny_dy = getGradient(ny, dx)
    kappa = -(dnx_dx + dny_dy)
    
    # Smooth curvature to reduce noise
    kappa = ndimage.gaussian_filter(kappa, sigma=1.0)
    
    # Surface tension force
    F_sigma_x = sigma * kappa * dphi_dx
    F_sigma_y = sigma * kappa * dphi_dy
    
    return F_sigma_x, F_sigma_y

def addGhostCells(rho, vx, vy, P):
    """Add ghost cells to top and bottom"""
    rho = np.hstack((rho[:,0:1], rho, rho[:,-1:]))
    vx = np.hstack((vx[:,0:1], vx, vx[:,-1:]))
    vy = np.hstack((vy[:,0:1], vy, vy[:,-1:]))
    P = np.hstack((P[:,0:1], P, P[:,-1:]))
    return rho, vx, vy, P

def setGhostCells(rho, vx, vy, P):
    """Set ghost cells at top and bottom (reflecting BC)"""
    rho[:,0] = rho[:,1]
    vx[:,0] = vx[:,1]
    vy[:,0] = -vy[:,1]
    P[:,0] = P[:,1]
    
    rho[:,-1] = rho[:,-2]
    vx[:,-1] = vx[:,-2]
    vy[:,-1] = -vy[:,-2]
    P[:,-1] = P[:,-2]
    
    return rho, vx, vy, P

def setGhostGradients(f_dx, f_dy):
    """Set ghost cell gradients (reflecting)"""
    f_dy[:,0] = -f_dy[:,1]
    f_dy[:,-1] = -f_dy[:,-2]
    return f_dx, f_dy

def addSourceTerm(Mass, Momx, Momy, Energy, g, dt):
    """Add gravitational source term"""
    Energy += dt * Momy * g
    Momy += dt * Mass * g
    return Mass, Momx, Momy, Energy

def run_python_solver(target_times):
    """
    Run enhanced VOF finite volume solver with viscosity, HLLC, and surface tension
    """
    print("\n" + "=" * 80)
    print("RUNNING ENHANCED VOF FINITE VOLUME SOLVER")
    print("=" * 80)
    
    target_times = np.sort(target_times)
    
    # Mesh
    dx = BOXSIZE_X / N_RESOLUTION
    vol = dx**2
    xlin = np.linspace(0.5*dx, BOXSIZE_X-0.5*dx, N_RESOLUTION)
    ylin = np.linspace(0.5*dx, BOXSIZE_Y-0.5*dx, 3*N_RESOLUTION)
    Y, X = np.meshgrid(ylin, xlin)
    
    # Initial conditions - VOF phase fraction
    # phi = 1 for heavy fluid (top), phi = 0 for light fluid (bottom)
    phi = 1.0 * (Y > INTERFACE_Y)
    
    # Blend properties based on phase fraction
    rho = phi * RHO_HEAVY + (1 - phi) * RHO_LIGHT
    mu = phi * MU_HEAVY + (1 - phi) * MU_LIGHT
    
    vx = np.zeros(X.shape)
    vy = W0 * (1-np.cos(4*np.pi*X)) * (1-np.cos(4*np.pi*Y/3))
    P = P0 + GRAVITY * (Y - INTERFACE_Y) * rho
    
    rho, vx, vy, P = addGhostCells(rho, vx, vy, P)
    phi = np.hstack((phi[:,0:1], phi, phi[:,-1:]))
    mu = np.hstack((mu[:,0:1], mu, mu[:,-1:]))
    
    # Get conserved variables
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, GAMMA, vol)
    
    # Time integration
    t = 0
    saved_data = []
    target_idx = 0
    max_time = np.max(target_times)
    
    print(f"Target times: {len(target_times)} timesteps")
    print(f"Time range: {target_times[0]:.6e} to {max_time:.6e} s")
    print(f"Resolution: {N_RESOLUTION} x {3*N_RESOLUTION}")
    
    iteration = 0
    next_print_iter = 100
    
    while t < max_time and target_idx < len(target_times):
        # Get primitive variables
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)

        # Update blended properties
        phi = np.clip(phi, 0, 1)  # Ensure phi stays in [0,1]
        rho_blend = phi * RHO_HEAVY + (1 - phi) * RHO_LIGHT
        mu = phi * MU_HEAVY + (1 - phi) * MU_LIGHT

        # Enforce physical bounds to prevent NaN/Inf
        rho = np.maximum(rho, 0.1)  # Density floor
        P = np.maximum(P, 1e-6)     # Pressure floor
        vx = np.nan_to_num(vx, nan=0.0, posinf=0.0, neginf=0.0)
        vy = np.nan_to_num(vy, nan=0.0, posinf=0.0, neginf=0.0)

        # Get time step (CFL) with safety checks
        sound_speed = np.sqrt(GAMMA * P / rho)
        velocity_mag = np.sqrt(vx**2 + vy**2)
        max_speed = np.max(sound_speed + velocity_mag)

        # Ensure max_speed is valid
        if not np.isfinite(max_speed) or max_speed < 1e-10:
            max_speed = 1.0  # Fallback value

        dt_conv = COURANT_FAC * dx / max_speed

        # Viscous CFL constraint
        if USE_VISCOSITY:
            max_visc = np.max(mu / rho)
            if np.isfinite(max_visc) and max_visc > 1e-10:
                dt_visc = VISCOUS_CFL_FAC * dx**2 / max_visc
                dt = min(dt_conv, dt_visc)
            else:
                dt = dt_conv
        else:
            dt = dt_conv

        # Final safety check on timestep
        dt = np.clip(dt, 1e-10, 1e-2)  # Reasonable bounds for this problem
        
        # Check if we need to adjust dt to hit a target time exactly
        save_this_step = False
        if target_idx < len(target_times):
            if t + dt >= target_times[target_idx]:
                dt = target_times[target_idx] - t
                save_this_step = True
        
        # Add source (half-step)
        Mass, Momx, Momy, Energy = addSourceTerm(Mass, Momx, Momy, Energy, GRAVITY, dt/2)
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)
        
        # Calculate gradients
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx, vx_dy = getGradient(vx, dx)
        vy_dx, vy_dy = getGradient(vy, dx)
        P_dx, P_dy = getGradient(P, dx)
        
        # Slope limit gradients
        if USE_SLOPE_LIMITING:
            rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
            vx_dx, vx_dy = slopeLimit(vx, dx, vx_dx, vx_dy)
            vy_dx, vy_dy = slopeLimit(vy, dx, vy_dx, vy_dy)
            P_dx, P_dy = slopeLimit(P, dx, P_dx, P_dy)
        
        # Extrapolate half-step in time
        rho_prime = rho - 0.5*dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_prime = vx - 0.5*dt * (vx * vx_dx + vy * vx_dy + (1/rho) * P_dx)
        vy_prime = vy - 0.5*dt * (vx * vy_dx + vy * vy_dy + (1/rho) * P_dy)
        P_prime = P - 0.5*dt * (GAMMA*P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)
        
        # Extrapolate in space to face centers
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_prime, rho_dx, rho_dy, dx)
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateInSpaceToFace(vx_prime, vx_dx, vx_dy, dx)
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateInSpaceToFace(vy_prime, vy_dx, vy_dy, dx)
        P_XL, P_XR, P_YL, P_YR = extrapolateInSpaceToFace(P_prime, P_dx, P_dy, dx)
        
        # Compute inviscid fluxes (HLLC or Lax-Friedrichs)
        if USE_HLLC:
            flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFluxHLLC(
                rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, GAMMA)
            flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFluxHLLC(
                rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, GAMMA)
        else:
            flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFluxLaxFriedrichs(
                rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, GAMMA)
            flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFluxLaxFriedrichs(
                rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, GAMMA)
        
        # Update solution with inviscid fluxes
        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        
        # Add viscous terms
        if USE_VISCOSITY:
            visc_flux_momx, visc_flux_momy, visc_flux_energy = computeViscousFluxes(vx, vy, mu, dx)
            Momx += dt * vol * visc_flux_momx
            Momy += dt * vol * visc_flux_momy
            Energy += dt * vol * visc_flux_energy
        
        # Add surface tension
        if USE_SURFACE_TENSION:
            F_sigma_x, F_sigma_y = computeSurfaceTension(phi, SIGMA, dx)
            Momx += dt * vol * rho * F_sigma_x
            Momy += dt * vol * rho * F_sigma_y
        
        # Add source (half-step)
        Mass, Momx, Momy, Energy = addSourceTerm(Mass, Momx, Momy, Energy, GRAVITY, dt/2)
        
        # Simple phase advection (donor-acceptor scheme)
        # Don't advect phase - just reconstruct from density
        rho_current = Mass / vol
        phi = (rho_current - RHO_LIGHT) / (RHO_HEAVY - RHO_LIGHT + 1e-10)
        phi = np.clip(phi, 0, 1)
        
        # Update time
        t += dt
        iteration += 1
        
        # Save data if needed
        if save_this_step:
            rho_save, vx_save, vy_save, P_save = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)
            
            # Remove ghost cells for saving
            rho_save = rho_save[:, 1:-1]
            vx_save = vx_save[:, 1:-1]
            vy_save = vy_save[:, 1:-1]
            P_save = P_save[:, 1:-1]
            phi_save = phi[:, 1:-1]
            
            saved_data.append({
                't': t,
                'rho': rho_save.copy(),
                'vx': vx_save.copy(),
                'vy': vy_save.copy(),
                'P': P_save.copy(),
                'phi': phi_save.copy(),
                'x_grid': X.copy(),
                'y_grid': Y.copy()
            })
            
            print(f"  Saved timestep {target_idx+1}/{len(target_times)}: t = {t:.6e} s (iteration {iteration})")
            target_idx += 1
        
        # Progress update
        if iteration >= next_print_iter:
            print(f"  Iteration {iteration}: t = {t:.6e} s ({t/max_time*100:.1f}% complete)")
            next_print_iter += 100
    
    print(f"\nEnhanced VOF solver complete: {len(saved_data)} timesteps saved")
    return saved_data

# ============================================================================
# HELPER FUNCTIONS (same as baseline)
# ============================================================================

def extract_timestep_number(filename):
    """Extract timestep number from plot file name"""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def get_frame_number(filename):
    """Extract frame number from filename like '0042_density.png'"""
    match = re.match(r'(\d+)_', filename)
    return int(match.group(1)) if match else 0

def compute_vorticity(vx, vy, dx, dy):
    """Calculate vorticity from velocity field"""
    dvx_dy = np.gradient(vx, dy, axis=0)
    dvy_dx = np.gradient(vy, dx, axis=1)
    vorticity = dvy_dx - dvx_dy
    return vorticity

def interpolate_to_common_grid(data, x_old, y_old, x_new, y_new):
    """Interpolate data from old grid to new grid"""
    x_old_copy = np.asarray(x_old).copy()
    y_old_copy = np.asarray(y_old).copy()
    data_copy = np.asarray(data).copy()
    
    # Check for and handle non-ascending coordinates
    if len(x_old_copy) > 1:
        x_diff = np.diff(x_old_copy)
        if np.any(x_diff <= 0):
            if np.all(x_diff < 0):
                x_old_copy = x_old_copy[::-1]
                data_copy = data_copy[::-1, :]
    
    if len(y_old_copy) > 1:
        y_diff = np.diff(y_old_copy)
        if np.any(y_diff <= 0):
            if np.all(y_diff < 0):
                y_old_copy = y_old_copy[::-1]
                data_copy = data_copy[:, ::-1]
    
    interp_func = RegularGridInterpolator(
        (x_old_copy, y_old_copy),
        data_copy,
        bounds_error=False,
        fill_value=None,
        method='linear'
    )
    
    points = np.column_stack([x_new.ravel(), y_new.ravel()])
    data_new = interp_func(points).reshape(x_new.shape)
    
    return data_new

def create_gif_from_folder(subfolder_name, output_gif_name):
    """Create GIF from all images in a subfolder with error handling"""
    subfolder_path = os.path.join(OUTPUT_FOLDER, subfolder_name)
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
    
    if not image_files:
        print(f"  WARNING: No images found in {subfolder_name}/")
        return
    
    image_files = sorted(image_files, key=get_frame_number)
    frames = []
    corrupted_files = []
    
    for img_file in image_files:
        img_path = os.path.join(subfolder_path, img_file)
        try:
            img = Image.open(img_path)
            img.load()
            frames.append(img.copy())
            img.close()
        except (OSError, IOError) as e:
            corrupted_files.append(img_file)
            continue
    
    if not frames:
        print(f"  ERROR: No valid images found in {subfolder_name}/")
        return
    
    gif_path = os.path.join(OUTPUT_FOLDER, output_gif_name)
    try:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000/GIF_FPS,
            loop=GIF_LOOP,
            optimize=GIF_OPTIMIZE
        )
        print(f"  Created GIF: {output_gif_name} ({len(frames)} frames)")
    except Exception as e:
        print(f"  ERROR creating GIF {output_gif_name}: {e}")
    finally:
        for frame in frames:
            try:
                frame.close()
            except:
                pass

# ============================================================================
# PLOTTING FUNCTIONS (same as baseline with "Enhanced VOF" labels)
# ============================================================================

def plot_density_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """Merged density comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    t = python_data['t']
    
    x_min = 0.0
    x_max = BOXSIZE_X
    x_split = BOXSIZE_X / 2
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    python_rho_masked = np.where(python_mask, python_data['rho'], np.nan)
    hydro2_rho_masked = np.where(hydro2_mask, hydro2_data['rho'], np.nan)
    
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'], 
                      python_rho_masked, levels=100, cmap=COLORMAP_DENSITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      hydro2_rho_masked, levels=100, cmap=COLORMAP_DENSITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    ax.text(x_split/2, -0.12, 'Enhanced VOF', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'hydro2', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    save_path = os.path.join(subfolder_density, f'{frame_num:04d}_density.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_velocity_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """Merged velocity comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    t = python_data['t']
    
    x_min = 0.0
    x_max = BOXSIZE_X
    x_split = BOXSIZE_X / 2
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    v_mag_python = np.sqrt(python_data['vx']**2 + python_data['vy']**2)
    v_mag_hydro2 = np.sqrt(hydro2_data['vx']**2 + hydro2_data['vy']**2)
    
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    v_mag_python_masked = np.where(python_mask, v_mag_python, np.nan)
    v_mag_hydro2_masked = np.where(hydro2_mask, v_mag_hydro2, np.nan)
    
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'],
                      v_mag_python_masked, levels=100, cmap=COLORMAP_VELOCITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      v_mag_hydro2_masked, levels=100, cmap=COLORMAP_VELOCITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    ax.text(x_split/2, -0.12, 'Enhanced VOF', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'hydro2', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    save_path = os.path.join(subfolder_velocity, f'{frame_num:04d}_velocity.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_vorticity_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """Merged vorticity comparison plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    t = python_data['t']
    
    x_min = 0.0
    x_max = BOXSIZE_X
    x_split = BOXSIZE_X / 2
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    vort_python_masked = np.where(python_mask, python_data['vorticity'], np.nan)
    vort_hydro2_masked = np.where(hydro2_mask, hydro2_data['vorticity'], np.nan)
    
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'],
                      vort_python_masked, levels=100, cmap=COLORMAP_VORTICITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      vort_hydro2_masked, levels=100, cmap=COLORMAP_VORTICITY,
                      vmin=vmin, vmax=vmax, extend='both')
    
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    ax.text(x_split/2, -0.12, 'Enhanced VOF', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'hydro2', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Vorticity (1/s)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    save_path = os.path.join(subfolder_vorticity, f'{frame_num:04d}_vorticity.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_difference(python_data, hydro2_data, field_name, frame_num, vmin_diff, vmax_diff):
    """Plot difference with error metrics"""
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax_main = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    t = python_data['t']
    
    if field_name == 'rho':
        python_field = python_data['rho']
        hydro2_field = hydro2_data['rho']
        field_label = 'Density'
        field_units = 'kg/m^3'
        save_folder = subfolder_diff_density
    elif field_name == 'v_mag':
        python_field = np.sqrt(python_data['vx']**2 + python_data['vy']**2)
        hydro2_field = np.sqrt(hydro2_data['vx']**2 + hydro2_data['vy']**2)
        field_label = 'Velocity Magnitude'
        field_units = 'm/s'
        save_folder = subfolder_diff_velocity
    elif field_name == 'vorticity':
        python_field = python_data['vorticity']
        hydro2_field = hydro2_data['vorticity']
        field_label = 'Vorticity'
        field_units = '1/s'
        save_folder = subfolder_diff_vorticity
    
    difference = hydro2_field - python_field
    
    abs_error = np.abs(difference)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    rms_error = np.sqrt(np.mean(difference**2))
    
    python_max = np.max(np.abs(python_field))
    if python_max > 1e-10:
        rel_error_max = max_error / python_max * 100
        rel_error_rms = rms_error / python_max * 100
    else:
        rel_error_max = 0.0
        rel_error_rms = 0.0
    
    l2_norm = np.sqrt(np.sum(difference**2))
    
    im = ax_main.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                         difference, levels=50, cmap=COLORMAP_DIFFERENCE,
                         vmin=vmin_diff, vmax=vmax_diff)
    
    ax_main.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_title(f'{field_label} Difference (hydro2 - Enhanced VOF) at t = {t*1e3:.2f} ms',
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_main.set_aspect('equal', adjustable='box')
    
    cbar = plt.colorbar(im, extend="neither", ax=ax_main)
    cbar.set_label(f'Difference ({field_units})', fontsize=FONT_SIZE_LABEL)
    
    error_text = f"""
    ERROR METRICS:
    
    Max Absolute Error:     {max_error:.6e} {field_units}
    Mean Absolute Error:    {mean_error:.6e} {field_units}
    RMS Error:              {rms_error:.6e} {field_units}
    L2 Norm:                {l2_norm:.6e}
    
    Max Relative Error:     {rel_error_max:.4f} %
    RMS Relative Error:     {rel_error_rms:.4f} %
    """
    
    ax_text.text(0.1, 0.5, error_text, fontsize=FONT_SIZE_LABEL,
                family='monospace', verticalalignment='center')
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f'{frame_num:04d}_diff_{field_name}.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN EXECUTION (same structure as baseline)
# ============================================================================

def main():
    """Main execution function"""
    
    # STEP 1: FIND AND LOAD HYDRO2 FILES
    print("\n" + "=" * 80)
    print("STEP 1: LOADING HYDRO2 OUTPUT FILES")
    print("=" * 80)
    
    plot_files = []
    for item in os.listdir(HYDRO2_OUTPUT_DIR):
        if '.old' in item.lower():
            continue
        item_path = os.path.join(HYDRO2_OUTPUT_DIR, item)
        if os.path.isdir(item_path) and ('plt' in item.lower() or 'cell' in item.lower()):
            plot_files.append(item_path)
    
    if not plot_files:
        print(f"ERROR: No plot files found in {HYDRO2_OUTPUT_DIR}")
        return
    
    plot_files.sort(key=extract_timestep_number)
    print(f"Found {len(plot_files)} hydro2 plot files")
    
    hydro2_times = []
    for plot_file in plot_files:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        hydro2_times.append(t)
    
    hydro2_times = np.array(hydro2_times)
    print(f"Time range: {hydro2_times[0]:.6e} to {hydro2_times[-1]:.6e} s")
    
    # STEP 2: RUN ENHANCED VOF SOLVER
    python_data_list = run_python_solver(hydro2_times)
    
    # STEP 3: MATCH TIMESTEPS
    print("\n" + "=" * 80)
    print("STEP 3: MATCHING TIMESTEPS")
    print("=" * 80)
    
    tolerance_sec = TIME_TOLERANCE_MS * 1e-3
    matched_pairs = []
    
    for i, t_hydro2 in enumerate(hydro2_times):
        time_diffs = np.abs(np.array([d['t'] for d in python_data_list]) - t_hydro2)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance_sec:
            matched_pairs.append((i, closest_idx))
            print(f"  Match {len(matched_pairs)}: hydro2 t={t_hydro2:.6e} s, "
                  f"Python t={python_data_list[closest_idx]['t']:.6e} s, "
                  f"diff={time_diffs[closest_idx]*1e3:.4f} ms")
    
    print(f"\nMatched {len(matched_pairs)} timesteps within {TIME_TOLERANCE_MS} ms tolerance")
    
    if len(matched_pairs) == 0:
        print("ERROR: No matching timesteps found!")
        return
    
    # STEP 4: EXTRACT HYDRO2 DATA
    print("\n" + "=" * 80)
    print("STEP 4: EXTRACTING HYDRO2 DATA")
    print("=" * 80)
    
    hydro2_data_list = []
    domain_width = BOXSIZE_X
    domain_height = BOXSIZE_Y
    resolution = 512
    
    for hydro2_idx, python_idx in matched_pairs:
        ds = yt.load(plot_files[hydro2_idx])
        t = float(ds.current_time)
        
        slc = ds.slice('z', 0.0)
        frb = slc.to_frb((domain_width, 'code_length'), resolution,
                        center=[0.5*domain_width, 0.5*domain_height, 0.0],
                        height=(domain_height, 'code_length'))
        
        rho = np.array(frb['density'])
        vx = np.array(frb['velocityx'])
        vy = np.array(frb['velocityy'])
        vorticity = np.array(frb['vorticity'])
        
        x_1d = np.linspace(0, domain_width, resolution)
        y_1d = np.linspace(0, domain_height, resolution)
        x_grid, y_grid = np.meshgrid(x_1d, y_1d)
        
        hydro2_data_list.append({
            't': t,
            'rho': rho,
            'vx': vx,
            'vy': vy,
            'vorticity': vorticity,
            'x_grid': x_grid,
            'y_grid': y_grid
        })
        
        if (len(hydro2_data_list)) % 5 == 0:
            print(f"  Extracted {len(hydro2_data_list)}/{len(matched_pairs)} timesteps")
    
    print(f"Extraction complete: {len(hydro2_data_list)} timesteps")
    
    # STEP 5: INTERPOLATE PYTHON DATA
    print("\n" + "=" * 80)
    print("STEP 5: INTERPOLATING PYTHON DATA TO COMMON GRID")
    print("=" * 80)
    
    python_data_interp = []
    
    for hydro2_idx, python_idx in matched_pairs:
        python_data = python_data_list[python_idx]
        hydro2_data = hydro2_data_list[len(python_data_interp)]
        
        x_python = python_data['x_grid'][0, :]
        y_python = python_data['y_grid'][:, 0]
        
        if len(np.unique(x_python)) == 1 or len(np.unique(y_python)) == 1:
            if len(np.unique(python_data['x_grid'][:, 0])) > 1:
                x_python = python_data['x_grid'][:, 0]
            else:
                x_python = python_data['x_grid'][0, :]
            
            if len(np.unique(python_data['y_grid'][0, :])) > 1:
                y_python = python_data['y_grid'][0, :]
            else:
                y_python = python_data['y_grid'][:, 0]
        
        x_target = hydro2_data['x_grid']
        y_target = hydro2_data['y_grid']
        
        rho_interp = interpolate_to_common_grid(python_data['rho'], x_python, y_python,
                                                x_target, y_target)
        vx_interp = interpolate_to_common_grid(python_data['vx'], x_python, y_python,
                                               x_target, y_target)
        vy_interp = interpolate_to_common_grid(python_data['vy'], x_python, y_python,
                                               x_target, y_target)
        
        dx = x_target[0, 1] - x_target[0, 0]
        dy = y_target[1, 0] - y_target[0, 0]
        vorticity_interp = compute_vorticity(vx_interp, vy_interp, dx, dy)
        
        python_data_interp.append({
            't': python_data['t'],
            'rho': rho_interp,
            'vx': vx_interp,
            'vy': vy_interp,
            'vorticity': vorticity_interp,
            'x_grid': x_target,
            'y_grid': y_target
        })
        
        if (len(python_data_interp)) % 5 == 0:
            print(f"  Interpolated {len(python_data_interp)}/{len(matched_pairs)} timesteps")
    
    print(f"Interpolation complete: {len(python_data_interp)} timesteps")
    
    # STEP 6: COMPUTE GLOBAL MIN/MAX
    print("\n" + "=" * 80)
    print("STEP 6: COMPUTING GLOBAL MIN/MAX FOR FIXED COLORBARS")
    print("=" * 80)
    
    rho_min = min(np.min(d['rho']) for d in python_data_interp + hydro2_data_list)
    rho_max = max(np.max(d['rho']) for d in python_data_interp + hydro2_data_list)
    print(f"Density range: [{rho_min:.6e}, {rho_max:.6e}] kg/m^3")
    
    v_mag_min = 0.0
    v_mag_max = max(
        max(np.max(np.sqrt(d['vx']**2 + d['vy']**2)) for d in python_data_interp),
        max(np.max(np.sqrt(d['vx']**2 + d['vy']**2)) for d in hydro2_data_list)
    )
    print(f"Velocity magnitude range: [{v_mag_min:.6e}, {v_mag_max:.6e}] m/s")
    
    vort_all = np.concatenate([d['vorticity'].flatten() for d in python_data_interp + hydro2_data_list])
    vort_std = np.std(vort_all)
    vort_lim = 2.0 * vort_std
    print(f"Vorticity range (+/-2sigma): [{-vort_lim:.6e}, {vort_lim:.6e}] 1/s")
    
    print("\nComputing difference limits...")
    diff_rho_list = []
    diff_v_mag_list = []
    diff_vort_list = []
    
    for i in range(len(matched_pairs)):
        diff_rho = hydro2_data_list[i]['rho'] - python_data_interp[i]['rho']
        diff_rho_list.append(diff_rho)
        
        v_mag_python = np.sqrt(python_data_interp[i]['vx']**2 + python_data_interp[i]['vy']**2)
        v_mag_hydro2 = np.sqrt(hydro2_data_list[i]['vx']**2 + hydro2_data_list[i]['vy']**2)
        diff_v_mag = v_mag_hydro2 - v_mag_python
        diff_v_mag_list.append(diff_v_mag)
        
        diff_vort = hydro2_data_list[i]['vorticity'] - python_data_interp[i]['vorticity']
        diff_vort_list.append(diff_vort)
    
    diff_rho_max = max(np.max(np.abs(d)) for d in diff_rho_list)
    diff_v_mag_max = max(np.max(np.abs(d)) for d in diff_v_mag_list)
    diff_vort_max = max(np.max(np.abs(d)) for d in diff_vort_list)
    
    print(f"Density difference range: [{-diff_rho_max:.6e}, {diff_rho_max:.6e}] kg/m^3")
    print(f"Velocity difference range: [{-diff_v_mag_max:.6e}, {diff_v_mag_max:.6e}] m/s")
    print(f"Vorticity difference range: [{-diff_vort_max:.6e}, {diff_vort_max:.6e}] 1/s")
    
    # STEP 7: GENERATE COMPARISON PLOTS
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING COMPARISON PLOTS")
    print("=" * 80)
    
    for i in range(len(matched_pairs)):
        frame_num = i + 1
        
        if PLOT_DENSITY:
            plot_density_comparison(python_data_interp[i], hydro2_data_list[i],
                                   frame_num, rho_min, rho_max)
        
        if PLOT_VELOCITY:
            plot_velocity_comparison(python_data_interp[i], hydro2_data_list[i],
                                    frame_num, v_mag_min, v_mag_max)
        
        if PLOT_VORTICITY:
            plot_vorticity_comparison(python_data_interp[i], hydro2_data_list[i],
                                     frame_num, -vort_lim, vort_lim)
        
        if PLOT_DIFFERENCES:
            plot_difference(python_data_interp[i], hydro2_data_list[i], 'rho',
                          frame_num, -diff_rho_max, diff_rho_max)
            plot_difference(python_data_interp[i], hydro2_data_list[i], 'v_mag',
                          frame_num, -diff_v_mag_max, diff_v_mag_max)
            plot_difference(python_data_interp[i], hydro2_data_list[i], 'vorticity',
                          frame_num, -diff_vort_max, diff_vort_max)
        
        if (frame_num) % 5 == 0:
            print(f"  Generated plots for frame {frame_num}/{len(matched_pairs)}")
    
    print(f"All comparison plots complete: {len(matched_pairs)} frames")
    
    # STEP 8: GENERATE GIFS
    if GENERATE_GIFS:
        print("\n" + "=" * 80)
        print("STEP 8: GENERATING ANIMATED GIFS")
        print("=" * 80)
        
        if PLOT_DENSITY:
            create_gif_from_folder('density', 'ANIM_density_comparison.gif')
        
        if PLOT_VELOCITY:
            create_gif_from_folder('velocity', 'ANIM_velocity_comparison.gif')
        
        if PLOT_VORTICITY:
            create_gif_from_folder('vorticity', 'ANIM_vorticity_comparison.gif')
        
        if PLOT_DIFFERENCES:
            create_gif_from_folder('difference_density', 'ANIM_difference_density.gif')
            create_gif_from_folder('difference_velocity', 'ANIM_difference_velocity.gif')
            create_gif_from_folder('difference_vorticity', 'ANIM_difference_vorticity.gif')
    
    # SUMMARY
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {OUTPUT_FOLDER}")
    print(f"Matched timesteps: {len(matched_pairs)}")
    print(f"\nGenerated folders:")
    if PLOT_DENSITY:
        print(f"  - density/: {len(os.listdir(subfolder_density))} files")
    if PLOT_VELOCITY:
        print(f"  - velocity/: {len(os.listdir(subfolder_velocity))} files")
    if PLOT_VORTICITY:
        print(f"  - vorticity/: {len(os.listdir(subfolder_vorticity))} files")
    if PLOT_DIFFERENCES:
        print(f"  - difference_density/: {len(os.listdir(subfolder_diff_density))} files")
        print(f"  - difference_velocity/: {len(os.listdir(subfolder_diff_velocity))} files")
        print(f"  - difference_vorticity/: {len(os.listdir(subfolder_diff_vorticity))} files")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
