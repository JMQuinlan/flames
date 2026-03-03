# -*- coding: utf-8 -*-
"""
===============================================================================
VOF RAYLEIGH-TAYLOR WITH CORRECTED REFLECTING BOUNDARY CONDITIONS
===============================================================================
FIXED: Removed periodic wrapping in Y direction
FIXED: Proper no-penetration walls at top/bottom
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os
from PIL import Image

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_FOLDER = './VOF_RT_Fixed_BCs'
PLOT_INTERVAL = 0.5

# Domain
BOXSIZE_X = 0.5
BOXSIZE_Y = 1.5
N_X = 64
N_Y = 192

# Time
T_END = 15.0
CFL = 0.2

# Physics
GAMMA = 1.4
GRAVITY = -0.1

RHO_HEAVY = 2.0
RHO_LIGHT = 1.0
MU_HEAVY = 1.0e-3
MU_LIGHT = 1.8e-5

INTERFACE_Y = 0.75
P0 = 2.5
W0 = 0.0025

SIGMA = 0.0728

USE_VISCOSITY = False
USE_SURFACE_TENSION = False

FLUX_METHOD = 'LaxFriedrichs'  # 'LaxFriedrichs' or 'HLLC'

DPI = 150

# ============================================================================
# SETUP
# ============================================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print("=" * 80)
print("VOF RAYLEIGH-TAYLOR WITH FIXED REFLECTING BCs")
print("=" * 80)
print(f"Domain: {BOXSIZE_X} x {BOXSIZE_Y} m")
print(f"Resolution: {N_X} x {N_Y}")
print(f"Flux method: {FLUX_METHOD}")
print(f"BCs: Periodic (X), Reflecting (Y) - CORRECTED")
print("=" * 80)

# ============================================================================
# BOUNDARY CONDITIONS - CORRECTED
# ============================================================================

def applyBCs(rho, vx, vy, P):
    """
    CORRECTED: Reflecting walls at top/bottom
    Key: Set normal velocity to ZERO at walls
    """
    # Bottom wall (j=0)
    rho[:, 0] = rho[:, 1]
    vx[:, 0] = vx[:, 1]
    vy[:, 0] = 0.0  # ZERO normal velocity (no penetration)
    P[:, 0] = P[:, 1]
    
    # Top wall (j=N_Y-1)
    rho[:, -1] = rho[:, -2]
    vx[:, -1] = vx[:, -2]
    vy[:, -1] = 0.0  # ZERO normal velocity (no penetration)
    P[:, -1] = P[:, -2]
    
    return rho, vx, vy, P

def getGradient(f, dx):
    """
    CORRECTED: Gradients with proper boundary treatment
    X: periodic
    Y: one-sided at walls
    """
    # X direction: periodic
    f_dx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2*dx)
    
    # Y direction: NO ROLL - use explicit indexing
    f_dy = np.zeros_like(f)
    
    # Interior: central difference
    f_dy[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2*dx)
    
    # Bottom: forward difference
    f_dy[:, 0] = (f[:, 1] - f[:, 0]) / dx
    
    # Top: backward difference
    f_dy[:, -1] = (f[:, -1] - f[:, -2]) / dx
    
    return f_dx, f_dy

def extrapolateToFace(f, f_dx, f_dy, dx):
    """
    CORRECTED: Extrapolate to faces WITHOUT periodic wrapping in Y
    """
    # X faces: periodic (this is correct)
    f_XL = f - f_dx * dx/2
    f_XL = np.roll(f_XL, -1, axis=0)
    f_XR = f + f_dx * dx/2
    
    # Y faces: NO ROLL - manual construction
    f_YL = np.zeros_like(f)
    f_YR = np.zeros_like(f)
    
    # Interior faces
    f_YL[:, 1:] = f[:, :-1] + f_dy[:, :-1] * dx/2  # Left state at face j+1/2
    f_YR[:, :-1] = f[:, 1:] - f_dy[:, 1:] * dx/2   # Right state at face j+1/2
    
    # Bottom boundary (j=0): wall
    f_YL[:, 0] = f[:, 0]  # No flux from below
    f_YR[:, 0] = f[:, 0]  # Wall state
    
    # Top boundary (j=N_Y-1): wall
    f_YL[:, -1] = f[:, -1]  # Wall state
    f_YR[:, -1] = f[:, -1]  # No flux from above
    
    return f_XL, f_XR, f_YL, f_YR

def applyFluxes(F, flux_X, flux_Y, dx, dt):
    """
    CORRECTED: Apply fluxes with proper wall treatment
    """
    # X direction: periodic (correct)
    F += -dt*dx*flux_X + dt*dx*np.roll(flux_X, 1, axis=0)
    
    # Y direction: reflecting walls (NO ROLL)
    # Interior cells: standard flux difference
    F[:, 1:-1] += dt*dx*(flux_Y[:, :-2] - flux_Y[:, 1:-1])
    
    # Bottom cell (j=0): only flux from above
    F[:, 0] += -dt*dx*flux_Y[:, 0]
    
    # Top cell (j=N_Y-1): only flux from below
    F[:, -1] += dt*dx*flux_Y[:, -2]
    
    return F

# ============================================================================
# SOLVER FUNCTIONS
# ============================================================================

def getConserved(rho, vx, vy, P, gamma, vol):
    """Primitive to conservative"""
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Energy = (P/(gamma-1) + 0.5*rho*(vx**2+vy**2)) * vol
    return Mass, Momx, Momy, Energy

def getPrimitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Conservative to primitive"""
    rho = Mass / vol
    rho = np.clip(rho, 0.5, 3.0)
    
    vx = Momx / (rho * vol + 1e-10)
    vy = Momy / (rho * vol + 1e-10)
    vx = np.clip(vx, -10, 10)
    vy = np.clip(vy, -10, 10)
    
    P = (Energy/vol - 0.5*rho*(vx**2+vy**2)) * (gamma-1)
    P = np.maximum(P, 1e-3)
    
    # Apply BCs
    rho, vx, vy, P = applyBCs(rho, vx, vy, P)
    
    return rho, vx, vy, P

def getFluxLaxFriedrichs(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Lax-Friedrichs flux"""
    rho_L = np.clip(rho_L, 0.5, 3.0)
    rho_R = np.clip(rho_R, 0.5, 3.0)
    P_L = np.maximum(P_L, 1e-3)
    P_R = np.maximum(P_R, 1e-3)
    
    en_L = P_L/(gamma-1) + 0.5*rho_L*(vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R*(vx_R**2+vy_R**2)
    
    rho_avg = 0.5*(rho_L + rho_R)
    momx_avg = 0.5*(rho_L*vx_L + rho_R*vx_R)
    momy_avg = 0.5*(rho_L*vy_L + rho_R*vy_R)
    en_avg = 0.5*(en_L + en_R)
    
    P_avg = (gamma-1)*(en_avg - 0.5*(momx_avg**2+momy_avg**2)/(rho_avg+1e-10))
    P_avg = np.maximum(P_avg, 1e-3)
    
    flux_Mass = momx_avg
    flux_Momx = momx_avg**2/(rho_avg+1e-10) + P_avg
    flux_Momy = momx_avg*momy_avg/(rho_avg+1e-10)
    flux_Energy = (en_avg+P_avg)*momx_avg/(rho_avg+1e-10)
    
    a_L = np.sqrt(gamma*P_L/rho_L)
    a_R = np.sqrt(gamma*P_R/rho_R)
    C = np.maximum(np.abs(vx_L)+a_L, np.abs(vx_R)+a_R)
    
    flux_Mass -= 0.5*C*(rho_L - rho_R)
    flux_Momx -= 0.5*C*(rho_L*vx_L - rho_R*vx_R)
    flux_Momy -= 0.5*C*(rho_L*vy_L - rho_R*vy_R)
    flux_Energy -= 0.5*C*(en_L - en_R)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def getFluxHLLC(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """HLLC Riemann solver"""
    eps = 1e-10
    
    rho_L = np.clip(rho_L, 0.5, 3.0)
    rho_R = np.clip(rho_R, 0.5, 3.0)
    P_L = np.maximum(P_L, 1e-3)
    P_R = np.maximum(P_R, 1e-3)
    vx_L = np.clip(vx_L, -10, 10)
    vx_R = np.clip(vx_R, -10, 10)
    vy_L = np.clip(vy_L, -10, 10)
    vy_R = np.clip(vy_R, -10, 10)
    
    en_L = P_L/(gamma-1) + 0.5*rho_L*(vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R*(vx_R**2+vy_R**2)
    
    a_L = np.sqrt(gamma * P_L / rho_L)
    a_R = np.sqrt(gamma * P_R / rho_R)
    
    S_L = np.minimum(vx_L - a_L, vx_R - a_R)
    S_R = np.maximum(vx_L + a_L, vx_R + a_R)
    
    num = P_R - P_L + rho_L*vx_L*(S_L-vx_L) - rho_R*vx_R*(S_R-vx_R)
    den = rho_L*(S_L-vx_L) - rho_R*(S_R-vx_R) + eps
    S_star = num / den
    S_star = np.clip(S_star, -10, 10)
    
    flux_Mass_L = rho_L * vx_L
    flux_Momx_L = rho_L * vx_L**2 + P_L
    flux_Momy_L = rho_L * vx_L * vy_L
    flux_Energy_L = (en_L + P_L) * vx_L
    
    flux_Mass_R = rho_R * vx_R
    flux_Momx_R = rho_R * vx_R**2 + P_R
    flux_Momy_R = rho_R * vx_R * vy_R
    flux_Energy_R = (en_R + P_R) * vx_R
    
    rho_star_L = rho_L * (S_L - vx_L) / (S_L - S_star + eps)
    rho_star_R = rho_R * (S_R - vx_R) / (S_R - S_star + eps)
    rho_star_L = np.clip(rho_star_L, 0.5, 3.0)
    rho_star_R = np.clip(rho_star_R, 0.5, 3.0)
    
    P_star_L = P_L + rho_L*(S_L-vx_L)*(S_star-vx_L)
    P_star_R = P_R + rho_R*(S_R-vx_R)*(S_star-vx_R)
    P_star_L = np.maximum(P_star_L, 1e-3)
    P_star_R = np.maximum(P_star_R, 1e-3)
    
    en_star_L = en_L + (S_star - vx_L)*(rho_star_L*S_star + P_star_L/(S_L-vx_L+eps))
    en_star_R = en_R + (S_star - vx_R)*(rho_star_R*S_star + P_star_R/(S_R-vx_R+eps))
    en_star_L = np.clip(en_star_L, 1e-3/(gamma-1), 1e10)
    en_star_R = np.clip(en_star_R, 1e-3/(gamma-1), 1e10)
    
    flux_Mass_star_L = rho_star_L * S_star
    flux_Momx_star_L = rho_star_L * S_star**2 + P_star_L
    flux_Momy_star_L = rho_star_L * S_star * vy_L
    flux_Energy_star_L = (en_star_L + P_star_L) * S_star
    
    flux_Mass_star_R = rho_star_R * S_star
    flux_Momx_star_R = rho_star_R * S_star**2 + P_star_R
    flux_Momy_star_R = rho_star_R * S_star * vy_R
    flux_Energy_star_R = (en_star_R + P_star_R) * S_star
    
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
    
    flux_Mass = np.clip(flux_Mass, -1e10, 1e10)
    flux_Momx = np.clip(flux_Momx, -1e10, 1e10)
    flux_Momy = np.clip(flux_Momy, -1e10, 1e10)
    flux_Energy = np.clip(flux_Energy, -1e10, 1e10)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma, method='LaxFriedrichs'):
    """Wrapper to select flux method"""
    if method == 'HLLC':
        return getFluxHLLC(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma)
    else:
        return getFluxLaxFriedrichs(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma)

def addGravity(Mass, Momx, Momy, Energy, g, dt):
    """Gravity source"""
    Energy += dt * Momy * g
    Momy += dt * Mass * g
    return Mass, Momx, Momy, Energy

def computeViscosity(vx, vy, mu, dx):
    """Viscous terms"""
    dvx_dx, dvx_dy = getGradient(vx, dx)
    dvy_dx, dvy_dy = getGradient(vy, dx)
    
    div_v = dvx_dx + dvy_dy
    
    tau_xx = 2*mu*(dvx_dx - div_v/3)
    tau_yy = 2*mu*(dvy_dy - div_v/3)
    tau_xy = mu*(dvx_dy + dvy_dx)
    
    dtau_xx_dx, _ = getGradient(tau_xx, dx)
    _, dtau_xy_dy = getGradient(tau_xy, dx)
    visc_momx = dtau_xx_dx + dtau_xy_dy
    
    dtau_xy_dx, _ = getGradient(tau_xy, dx)
    _, dtau_yy_dy = getGradient(tau_yy, dx)
    visc_momy = dtau_xy_dx + dtau_yy_dy
    
    work_x = tau_xx*vx + tau_xy*vy
    work_y = tau_xy*vx + tau_yy*vy
    dwork_x_dx, _ = getGradient(work_x, dx)
    _, dwork_y_dy = getGradient(work_y, dx)
    visc_energy = dwork_x_dx + dwork_y_dy
    
    return visc_momx, visc_momy, visc_energy

def computeSurfaceTension(phi, sigma, dx):
    """Surface tension CSF"""
    dphi_dx, dphi_dy = getGradient(phi, dx)
    
    grad_mag = np.sqrt(dphi_dx**2 + dphi_dy**2 + 1e-10)
    nx = dphi_dx / grad_mag
    ny = dphi_dy / grad_mag
    
    dnx_dx, _ = getGradient(nx, dx)
    _, dny_dy = getGradient(ny, dx)
    kappa = -(dnx_dx + dny_dy)
    
    kappa = ndimage.gaussian_filter(kappa, sigma=1.0)
    
    F_x = sigma * kappa * dphi_dx
    F_y = sigma * kappa * dphi_dy
    
    return F_x, F_y

# ============================================================================
# PLOTTING
# ============================================================================

def plot_state(rho, vx, vy, phi, x, y, t, frame_num):
    """Create plot"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    ax = axes[0, 0]
    im = ax.contourf(x, y, rho.T, levels=50, cmap='viridis')
    ax.set_title(f'Density (t={t:.2f} s)', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='kg/m^3')
    
    ax = axes[0, 1]
    v_mag = np.sqrt(vx**2 + vy**2)
    im = ax.contourf(x, y, v_mag.T, levels=50, cmap='plasma')
    ax.set_title('Velocity Magnitude', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='m/s')
    
    ax = axes[1, 0]
    im = ax.contourf(x, y, phi.T, levels=50, cmap='RdBu_r')
    ax.contour(x, y, phi.T, levels=[0.5], colors='black', linewidths=2)
    ax.set_title('Phase Fraction', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='phi')
    
    ax = axes[1, 1]
    dx_plot = x[1] - x[0]
    dy_plot = y[1] - y[0]
    dvx_dy = np.gradient(vx, dy_plot, axis=1)
    dvy_dx = np.gradient(vy, dx_plot, axis=0)
    vort = dvy_dx - dvx_dy
    vort_lim = max(2*np.std(vort), 0.1)
    im = ax.contourf(x, y, vort.T, levels=50, cmap='RdBu_r',
                     vmin=-vort_lim, vmax=vort_lim)
    ax.set_title('Vorticity', fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='1/s')
    
    plt.suptitle(f'VOF RT ({FLUX_METHOD}) FIXED BCs - Frame {frame_num}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'{frame_num:04d}.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main solver"""
    
    dx = BOXSIZE_X / N_X
    dy = BOXSIZE_Y / N_Y
    vol = dx * dy
    
    x = np.linspace(0.5*dx, BOXSIZE_X-0.5*dx, N_X)
    y = np.linspace(0.5*dy, BOXSIZE_Y-0.5*dy, N_Y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial conditions
    phi = 1.0 * (Y > INTERFACE_Y)
    rho = phi*RHO_HEAVY + (1-phi)*RHO_LIGHT
    mu = phi*MU_HEAVY + (1-phi)*MU_LIGHT
    
    vx = np.zeros_like(X)
    vy = W0 * (1 - np.cos(4*np.pi*X/BOXSIZE_X)) * (1 - np.cos(4*np.pi*Y/(3*BOXSIZE_X)))
    P = np.full_like(X, P0)
    
    print(f"\nInitial conditions:")
    print(f"  Max initial vy: {np.max(np.abs(vy)):.6f} m/s")
    print(f"  Total mass: {np.sum(rho)*vol:.6f} kg")
    
    rho, vx, vy, P = applyBCs(rho, vx, vy, P)
    
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, GAMMA, vol)
    
    t = 0
    iteration = 0
    frame_num = 0
    next_plot = 0
    
    initial_mass = np.sum(Mass)
    
    print("\nStarting simulation...")
    print(f"{'Iter':<8} {'Time':<10} {'dt':<10} {'Max|v|':<10} {'Mass err%':<12}")
    print("-" * 55)
    
    while t < T_END:
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)
        
        phi = np.clip((rho - RHO_LIGHT)/(RHO_HEAVY - RHO_LIGHT), 0, 1)
        mu = phi*MU_HEAVY + (1-phi)*MU_LIGHT
        
        a = np.sqrt(GAMMA * P / rho)
        v_mag = np.sqrt(vx**2 + vy**2)
        max_speed = np.max(a + v_mag)
        dt = CFL * min(dx, dy) / max_speed
        dt = min(dt, 0.01)
        
        if t >= next_plot:
            mass_error = (np.sum(Mass) - initial_mass) / initial_mass * 100
            plot_state(rho, vx, vy, phi, x, y, t, frame_num)
            print(f"{iteration:<8} {t:<10.3f} {dt:<10.6f} {np.max(v_mag):<10.6f} {mass_error:<12.6f}")
            frame_num += 1
            next_plot += PLOT_INTERVAL
        
        Mass, Momx, Momy, Energy = addGravity(Mass, Momx, Momy, Energy, GRAVITY, dt/2)
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)
        
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx, vx_dy = getGradient(vx, dx)
        vy_dx, vy_dy = getGradient(vy, dx)
        P_dx, P_dy = getGradient(P, dx)
        
        rho_p = rho - 0.5*dt*(vx*rho_dx + rho*vx_dx + vy*rho_dy + rho*vy_dy)
        vx_p = vx - 0.5*dt*(vx*vx_dx + vy*vx_dy + P_dx/rho)
        vy_p = vy - 0.5*dt*(vx*vy_dx + vy*vy_dy + P_dy/rho)
        P_p = P - 0.5*dt*(GAMMA*P*(vx_dx+vy_dy) + vx*P_dx + vy*P_dy)
        
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateToFace(rho_p, rho_dx, rho_dy, dx)
        vx_XL, vx_XR, vx_YL, vx_YR = extrapolateToFace(vx_p, vx_dx, vx_dy, dx)
        vy_XL, vy_XR, vy_YL, vy_YR = extrapolateToFace(vy_p, vy_dx, vy_dy, dx)
        P_XL, P_XR, P_YL, P_YR = extrapolateToFace(P_p, P_dx, P_dy, dx)
        
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(
            rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, GAMMA, FLUX_METHOD)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(
            rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, GAMMA, FLUX_METHOD)
        
        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        
        if USE_VISCOSITY:
            visc_momx, visc_momy, visc_energy = computeViscosity(vx, vy, mu, dx)
            Momx += dt * vol * visc_momx
            Momy += dt * vol * visc_momy
            Energy += dt * vol * visc_energy
        
        if USE_SURFACE_TENSION:
            F_x, F_y = computeSurfaceTension(phi, SIGMA, dx)
            Momx += dt * vol * rho * F_x
            Momy += dt * vol * rho * F_y
        
        Mass, Momx, Momy, Energy = addGravity(Mass, Momx, Momy, Energy, GRAVITY, dt/2)
        
        Mass = np.clip(Mass, 0.5*vol, 3.0*vol)
        Energy = np.maximum(Energy, 1e-3*vol/(GAMMA-1))
        
        t += dt
        iteration += 1
    
    final_mass_error = (np.sum(Mass) - initial_mass) / initial_mass * 100
    
    print("\n" + "=" * 55)
    print(f"Complete! {frame_num} frames saved")
    print(f"Final mass conservation error: {final_mass_error:.6f}%")
    
    if frame_num > 0:
        print("Creating GIF...")
        images = []
        for i in range(frame_num):
            img_path = os.path.join(OUTPUT_FOLDER, f'{i:04d}.png')
            if os.path.exists(img_path):
                images.append(Image.open(img_path))
        
        if images:
            gif_path = os.path.join(OUTPUT_FOLDER, f'RT_{FLUX_METHOD}_FIXED.gif')
            images[0].save(gif_path, save_all=True, append_images=images[1:],
                          duration=100, loop=0)
            print(f"GIF saved: {gif_path}")

if __name__ == "__main__":
    main()
