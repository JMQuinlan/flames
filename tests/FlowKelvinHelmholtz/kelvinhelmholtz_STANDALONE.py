"""
================================================================================
KELVIN-HELMHOLTZ INSTABILITY SOLVER - LOW DIFFUSION (HLLC + VAN LEER)
================================================================================
Description:
    2D inviscid compressible Euler equations solver for simulating the 
    Kelvin-Helmholtz instability using a second-order finite volume method.
    
Numerical Methods:
    - Spatial: MUSCL reconstruction with van Leer limiter (2nd order, low diffusion)
    - Riemann Solver: HLLC (Harten-Lax-van Leer-Contact) - less diffusive than HLLE
    - Temporal: 3rd-order TVD Runge-Kutta (RK3)
    - Boundary Conditions: Periodic in x, Reflective (slip wall) in y
    
Key Improvements:
    - HLLC solver resolves contact discontinuities (much sharper than HLLE)
    - Van Leer limiter (less diffusive than minmod)
    - Should preserve vortex structures much better
    
Author: Generated for Tyler Tryon
Date: March 10, 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from PIL import Image
import glob

# ============================================================================
# PHYSICAL PARAMETERS
# ============================================================================
GAMMA = 1.4                    # Specific heat ratio (air)
RHO_1 = 1.0                    # Density in upper region
RHO_2 = 2.0                    # Density in lower region
U_1 = 0.5                      # x-velocity in upper region
U_2 = -0.5                     # x-velocity in lower region
P_0 = 2.5                      # Uniform pressure
SHEAR_LAYER_THICKNESS = 0.005   # Thickness of shear layer (delta)
PERTURBATION_AMPLITUDE = 0.05  # Amplitude of velocity perturbation
PERTURBATION_WIDTH = 0.1       # Width of perturbation (sigma)

# ============================================================================
# DOMAIN AND GRID PARAMETERS
# ============================================================================
LX = 1.0                       # Domain length in x
LY = 1.0                       # Domain length in y
NX = 256                       # Number of cells in x
NY = 256                       # Number of cells in y

# ============================================================================
# TIME INTEGRATION PARAMETERS
# ============================================================================
CFL = 0.7                      # CFL number
T_FINAL = 8.0                  # Final simulation time
OUTPUT_TIME_INTERVAL = 0.2     # Save plot every this many seconds of physical time

# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================
EPSILON = 1e-10                # Small number to prevent division by zero
MIN_DENSITY = 1e-6             # Minimum allowed density
MIN_PRESSURE = 1e-6            # Minimum allowed pressure

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================
# Figure settings
FIG_WIDTH = 14                 # Figure width in inches
FIG_HEIGHT = 6                 # Figure height in inches
FIG_DPI = 150                  # Figure resolution

# Font sizes
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
COLORBAR_FONTSIZE = 12

# Color schemes (matplotlib colormaps)
DENSITY_COLORMAP = 'viridis'
VORTICITY_COLORMAP = 'RdBu_r'

# Contour levels
DENSITY_LEVELS = 50
VORTICITY_LEVELS = 50

# Line width for contours
CONTOUR_LINEWIDTH = 0.5

# Output directory
OUTPUT_DIR = 'output_direc'
DENSITY_PLOTS_DIR = os.path.join(OUTPUT_DIR, 'DensityPlots')
GIF_FILENAME = 'kelvin_helmholtz.gif'
GIF_FPS = 10                   # Frames per second for GIF

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_output_directories():
    """
    create_output_directories(): Create output directories if they don't exist
    Inputs: None
    Outputs: None
    """
    os.makedirs(DENSITY_PLOTS_DIR, exist_ok=True)


def primitive_to_conservative(rho, u, v, p, gamma):
    """
    primitive_to_conservative(): Convert primitive to conservative variables
    Inputs:
        rho   - Density array [Nx, Ny]
        u     - x-velocity array [Nx, Ny]
        v     - y-velocity array [Nx, Ny]
        p     - Pressure array [Nx, Ny]
        gamma - Specific heat ratio (scalar)
    Outputs:
        U     - Conservative variables array [4, Nx, Ny]
                U[0] = rho, U[1] = rho*u, U[2] = rho*v, U[3] = E
    """
    U = np.zeros((4, rho.shape[0], rho.shape[1]))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return U


def conservative_to_primitive(U, gamma):
    """
    conservative_to_primitive(): Convert conservative to primitive variables
    Inputs:
        U     - Conservative variables array [4, Nx, Ny]
        gamma - Specific heat ratio (scalar)
    Outputs:
        rho   - Density array [Nx, Ny]
        u     - x-velocity array [Nx, Ny]
        v     - y-velocity array [Nx, Ny]
        p     - Pressure array [Nx, Ny]
    """
    # Enforce positivity
    rho = np.maximum(U[0], MIN_DENSITY)
    
    # Compute velocities with safeguard
    u = U[1] / (rho + EPSILON)
    v = U[2] / (rho + EPSILON)
    
    # Compute pressure with positivity constraint
    kinetic_energy = 0.5 * rho * (u**2 + v**2)
    p = (gamma - 1.0) * (U[3] - kinetic_energy)
    p = np.maximum(p, MIN_PRESSURE)
    
    return rho, u, v, p


def enforce_positivity(U, gamma):
    """
    enforce_positivity(): Enforce physical constraints on conservative variables
    Inputs:
        U     - Conservative variables [4, Nx, Ny]
        gamma - Specific heat ratio
    Outputs:
        U     - Conservative variables with enforced positivity [4, Nx, Ny]
    """
    # Enforce minimum density
    U[0] = np.maximum(U[0], MIN_DENSITY)
    
    # Recompute pressure and enforce minimum
    rho, u, v, p = conservative_to_primitive(U, gamma)
    
    # If pressure is too low, adjust total energy
    mask = p < MIN_PRESSURE
    if np.any(mask):
        U[3] = np.where(mask, 
                        MIN_PRESSURE / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2),
                        U[3])
    
    return U


# ============================================================================
# GRID INITIALIZATION
# ============================================================================

def initialize_grid(Nx, Ny, Lx, Ly):
    """
    initialize_grid(): Create computational grid
    Inputs:
        Nx - Number of cells in x direction
        Ny - Number of cells in y direction
        Lx - Domain length in x
        Ly - Domain length in y
    Outputs:
        x  - x-coordinates of cell centers [Nx]
        y  - y-coordinates of cell centers [Ny]
        dx - Cell spacing in x
        dy - Cell spacing in y
    """
    dx = Lx / Nx
    dy = Ly / Ny
    x = np.linspace(dx/2, Lx - dx/2, Nx)
    y = np.linspace(dy/2, Ly - dy/2, Ny)
    return x, y, dx, dy


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

def initialize_kelvin_helmholtz(x, y, params):
    """
    initialize_kelvin_helmholtz(): Set up Kelvin-Helmholtz initial conditions
    Inputs:
        x      - x-coordinates array [Nx]
        y      - y-coordinates array [Ny]
        params - Dictionary containing:
                 'rho1', 'rho2', 'u1', 'u2', 'p0', 'gamma', 
                 'delta', 'amplitude', 'sigma'
    Outputs:
        U      - Initial conservative variables [4, Nx, Ny]
    """
    Nx = len(x)
    Ny = len(y)
    
    # Create 2D meshgrid
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Extract parameters
    rho1 = params['rho1']
    rho2 = params['rho2']
    u1 = params['u1']
    u2 = params['u2']
    p0 = params['p0']
    delta = params['delta']
    A = params['amplitude']
    sigma = params['sigma']
    
    # Smooth shear layer with tanh profile
    y_mid = LY / 2.0
    rho = rho1 + (rho2 - rho1) * 0.5 * (1.0 + np.tanh((Y - y_mid) / delta))
    u = u1 + (u2 - u1) * 0.5 * (1.0 + np.tanh((Y - y_mid) / delta))
    
    # Sinusoidal perturbation in y-velocity
    wavelength = LX  # Single wavelength across domain
    v = A * np.sin(2.0 * np.pi * X / wavelength) * np.exp(-((Y - y_mid)**2) / sigma**2)
    
    # Uniform pressure
    p = np.ones((Nx, Ny)) * p0
    
    # Convert to conservative variables
    U = primitive_to_conservative(rho, u, v, p, params['gamma'])
    
    return U


# ============================================================================
# FLUX CALCULATIONS
# ============================================================================

def compute_flux_x(U, gamma):
    """
    compute_flux_x(): Compute flux in x-direction
    Inputs:
        U     - Conservative variables [4, ...]
        gamma - Specific heat ratio
    Outputs:
        F     - Flux in x-direction [4, ...]
    """
    rho, u, v, p = conservative_to_primitive(U, gamma)
    
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = u * (U[3] + p)
    
    return F


def compute_flux_y(U, gamma):
    """
    compute_flux_y(): Compute flux in y-direction
    Inputs:
        U     - Conservative variables [4, ...]
        gamma - Specific heat ratio
    Outputs:
        G     - Flux in y-direction [4, ...]
    """
    rho, u, v, p = conservative_to_primitive(U, gamma)
    
    G = np.zeros_like(U)
    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = v * (U[3] + p)
    
    return G


# ============================================================================
# SLOPE LIMITERS
# ============================================================================

def van_leer(a, b):
    """
    van_leer(): Van Leer slope limiter (less diffusive than minmod)
    Inputs:
        a - First slope
        b - Second slope
    Outputs:
        Limited slope
    """
    # Van Leer limiter: (a*b + |a*b|) / (a + b)
    # Returns 0 if a and b have different signs
    return np.where(a * b > 0,
                    2.0 * a * b / (a + b + EPSILON),
                    0.0)


# ============================================================================
# MUSCL RECONSTRUCTION
# ============================================================================

def reconstruct_x(U, dx):
    """
    reconstruct_x(): MUSCL reconstruction in x-direction with van Leer limiter
    Inputs:
        U  - Conservative variables [4, Nx, Ny]
        dx - Cell spacing in x
    Outputs:
        U_L - Left states at interfaces [4, Nx+1, Ny]
        U_R - Right states at interfaces [4, Nx+1, Ny]
    """
    Nx = U.shape[1]
    Ny = U.shape[2]
    
    U_L = np.zeros((4, Nx+1, Ny))
    U_R = np.zeros((4, Nx+1, Ny))
    
    # Use van Leer limiter
    for k in range(4):
        # Interior cells
        for i in range(1, Nx-1):
            slope_left = (U[k, i, :] - U[k, i-1, :]) / dx
            slope_right = (U[k, i+1, :] - U[k, i, :]) / dx
            slope = van_leer(slope_left, slope_right)
            
            # Reconstruct at interfaces
            U_R[k, i, :] = U[k, i, :] + 0.5 * slope * dx
            U_L[k, i+1, :] = U[k, i, :] + 0.5 * slope * dx
        
        # Periodic boundary: cell 0 (left edge)
        slope_left = (U[k, 0, :] - U[k, -1, :]) / dx
        slope_right = (U[k, 1, :] - U[k, 0, :]) / dx
        slope = van_leer(slope_left, slope_right)
        U_R[k, 0, :] = U[k, 0, :] + 0.5 * slope * dx
        U_L[k, 1, :] = U[k, 0, :] + 0.5 * slope * dx
        
        # Periodic boundary: cell Nx-1 (right edge)
        slope_left = (U[k, -1, :] - U[k, -2, :]) / dx
        slope_right = (U[k, 0, :] - U[k, -1, :]) / dx
        slope = van_leer(slope_left, slope_right)
        U_R[k, Nx-1, :] = U[k, -1, :] + 0.5 * slope * dx
        U_L[k, Nx, :] = U[k, -1, :] + 0.5 * slope * dx
        
        # Interface at x=0 (wraps around)
        U_L[k, 0, :] = U_R[k, Nx-1, :] - slope * dx
        U_R[k, Nx, :] = U_L[k, 1, :] + slope * dx
    
    return U_L, U_R


def reconstruct_y(U, dy):
    """
    reconstruct_y(): MUSCL reconstruction in y-direction with van Leer limiter
    Inputs:
        U  - Conservative variables [4, Nx, Ny]
        dy - Cell spacing in y
    Outputs:
        U_L - Left states at interfaces [4, Nx, Ny+1]
        U_R - Right states at interfaces [4, Nx, Ny+1]
    """
    Nx = U.shape[1]
    Ny = U.shape[2]
    
    U_L = np.zeros((4, Nx, Ny+1))
    U_R = np.zeros((4, Nx, Ny+1))
    
    # Use van Leer limiter
    for k in range(4):
        # Interior cells
        for j in range(1, Ny-1):
            slope_left = (U[k, :, j] - U[k, :, j-1]) / dy
            slope_right = (U[k, :, j+1] - U[k, :, j]) / dy
            slope = van_leer(slope_left, slope_right)
            
            # Reconstruct at interfaces
            U_R[k, :, j] = U[k, :, j] + 0.5 * slope * dy
            U_L[k, :, j+1] = U[k, :, j] + 0.5 * slope * dy
    
    # Reflective (slip wall) boundary conditions at top and bottom
    # Bottom boundary (j=0)
    for k in range(4):
        if k == 2:  # rho*v component - negate for reflection
            U_R[k, :, 0] = -U[k, :, 0]
            U_L[k, :, 0] = -U[k, :, 0]
        else:  # rho, rho*u, E - copy
            U_R[k, :, 0] = U[k, :, 0]
            U_L[k, :, 0] = U[k, :, 0]
        
        # First interior interface
        U_L[k, :, 1] = U[k, :, 0]
    
    # Top boundary (j=Ny)
    for k in range(4):
        if k == 2:  # rho*v component - negate for reflection
            U_R[k, :, Ny] = -U[k, :, -1]
            U_L[k, :, Ny] = -U[k, :, -1]
        else:  # rho, rho*u, E - copy
            U_R[k, :, Ny] = U[k, :, -1]
            U_L[k, :, Ny] = U[k, :, -1]
        
        # Last interior interface
        U_R[k, :, Ny-1] = U[k, :, -1]
    
    return U_L, U_R


# ============================================================================
# HLLC RIEMANN SOLVER
# ============================================================================

def hllc_flux_x(U_L, U_R, gamma):
    """
    hllc_flux_x(): HLLC Riemann solver for x-direction flux
    Inputs:
        U_L   - Left state conservative variables [4, ...]
        U_R   - Right state conservative variables [4, ...]
        gamma - Specific heat ratio
    Outputs:
        F_hllc - HLLC flux [4, ...]
    """
    # Left state
    rho_L, u_L, v_L, p_L = conservative_to_primitive(U_L, gamma)
    a_L = np.sqrt(gamma * p_L / (rho_L + EPSILON))
    H_L = (U_L[3] + p_L) / (rho_L + EPSILON)
    
    # Right state
    rho_R, u_R, v_R, p_R = conservative_to_primitive(U_R, gamma)
    a_R = np.sqrt(gamma * p_R / (rho_R + EPSILON))
    H_R = (U_R[3] + p_R) / (rho_R + EPSILON)
    
    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    a_tilde = np.sqrt((gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2))
    
    # Wave speeds
    S_L = np.minimum(u_L - a_L, u_tilde - a_tilde)
    S_R = np.maximum(u_R + a_R, u_tilde + a_tilde)
    
    # Contact wave speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + EPSILON)
    
    # Fluxes
    F_L = compute_flux_x(U_L, gamma)
    F_R = compute_flux_x(U_R, gamma)
    
    # Star region states
    U_star_L = np.zeros_like(U_L)
    U_star_R = np.zeros_like(U_R)
    
    factor_L = rho_L * (S_L - u_L) / (S_L - S_star + EPSILON)
    factor_R = rho_R * (S_R - u_R) / (S_R - S_star + EPSILON)
    
    U_star_L[0] = factor_L
    U_star_L[1] = factor_L * S_star
    U_star_L[2] = factor_L * v_L
    U_star_L[3] = factor_L * (U_L[3] / (rho_L + EPSILON) + (S_star - u_L) * \
                              (S_star + p_L / (rho_L * (S_L - u_L) + EPSILON)))
    
    U_star_R[0] = factor_R
    U_star_R[1] = factor_R * S_star
    U_star_R[2] = factor_R * v_R
    U_star_R[3] = factor_R * (U_R[3] / (rho_R + EPSILON) + (S_star - u_R) * \
                              (S_star + p_R / (rho_R * (S_R - u_R) + EPSILON)))
    
    # HLLC flux
    F_hllc = np.zeros_like(U_L)
    
    for k in range(4):
        F_hllc[k] = np.where(S_L >= 0, F_L[k],
                    np.where(S_star >= 0, F_L[k] + S_L * (U_star_L[k] - U_L[k]),
                    np.where(S_R >= 0, F_R[k] + S_R * (U_star_R[k] - U_R[k]),
                    F_R[k])))
    
    return F_hllc


def hllc_flux_y(U_L, U_R, gamma):
    """
    hllc_flux_y(): HLLC Riemann solver for y-direction flux
    Inputs:
        U_L   - Left state conservative variables [4, ...]
        U_R   - Right state conservative variables [4, ...]
        gamma - Specific heat ratio
    Outputs:
        G_hllc - HLLC flux [4, ...]
    """
    # Left state
    rho_L, u_L, v_L, p_L = conservative_to_primitive(U_L, gamma)
    a_L = np.sqrt(gamma * p_L / (rho_L + EPSILON))
    H_L = (U_L[3] + p_L) / (rho_L + EPSILON)
    
    # Right state
    rho_R, u_R, v_R, p_R = conservative_to_primitive(U_R, gamma)
    a_R = np.sqrt(gamma * p_R / (rho_R + EPSILON))
    H_R = (U_R[3] + p_R) / (rho_R + EPSILON)
    
    # Roe averages
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    v_tilde = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    a_tilde = np.sqrt((gamma - 1.0) * (H_tilde - 0.5 * v_tilde**2))
    
    # Wave speeds
    S_L = np.minimum(v_L - a_L, v_tilde - a_tilde)
    S_R = np.maximum(v_R + a_R, v_tilde + a_tilde)
    
    # Contact wave speed
    S_star = (p_R - p_L + rho_L * v_L * (S_L - v_L) - rho_R * v_R * (S_R - v_R)) / \
             (rho_L * (S_L - v_L) - rho_R * (S_R - v_R) + EPSILON)
    
    # Fluxes
    G_L = compute_flux_y(U_L, gamma)
    G_R = compute_flux_y(U_R, gamma)
    
    # Star region states
    U_star_L = np.zeros_like(U_L)
    U_star_R = np.zeros_like(U_R)
    
    factor_L = rho_L * (S_L - v_L) / (S_L - S_star + EPSILON)
    factor_R = rho_R * (S_R - v_R) / (S_R - S_star + EPSILON)
    
    U_star_L[0] = factor_L
    U_star_L[1] = factor_L * u_L
    U_star_L[2] = factor_L * S_star
    U_star_L[3] = factor_L * (U_L[3] / (rho_L + EPSILON) + (S_star - v_L) * \
                              (S_star + p_L / (rho_L * (S_L - v_L) + EPSILON)))
    
    U_star_R[0] = factor_R
    U_star_R[1] = factor_R * u_R
    U_star_R[2] = factor_R * S_star
    U_star_R[3] = factor_R * (U_R[3] / (rho_R + EPSILON) + (S_star - v_R) * \
                              (S_star + p_R / (rho_R * (S_R - v_R) + EPSILON)))
    
    # HLLC flux
    G_hllc = np.zeros_like(U_L)
    
    for k in range(4):
        G_hllc[k] = np.where(S_L >= 0, G_L[k],
                    np.where(S_star >= 0, G_L[k] + S_L * (U_star_L[k] - U_L[k]),
                    np.where(S_R >= 0, G_R[k] + S_R * (U_star_R[k] - U_R[k]),
                    G_R[k])))
    
    return G_hllc


# ============================================================================
# BOUNDARY CONDITIONS
# ============================================================================

def apply_boundary_conditions(U):
    """
    apply_boundary_conditions(): Apply boundary conditions
    Inputs:
        U - Conservative variables [4, Nx, Ny]
    Outputs:
        U - Conservative variables with updated boundaries [4, Nx, Ny]
    """
    # Periodic in x (handled in reconstruction)
    # Reflective in y (handled in reconstruction)
    return U


# ============================================================================
# SPATIAL DISCRETIZATION
# ============================================================================

def compute_rhs(U, dx, dy, gamma):
    """
    compute_rhs(): Compute right-hand side of semi-discrete equations
    Inputs:
        U     - Conservative variables [4, Nx, Ny]
        dx    - Cell spacing in x
        dy    - Cell spacing in y
        gamma - Specific heat ratio
    Outputs:
        dU_dt - Time derivative of conservative variables [4, Nx, Ny]
    """
    Nx = U.shape[1]
    Ny = U.shape[2]
    
    # Apply boundary conditions
    U = apply_boundary_conditions(U)
    
    # Enforce positivity before reconstruction
    U = enforce_positivity(U, gamma)
    
    # X-direction fluxes (periodic BC)
    U_L_x, U_R_x = reconstruct_x(U, dx)
    F_x = np.zeros((4, Nx+1, Ny))
    for i in range(Nx+1):
        F_x[:, i, :] = hllc_flux_x(U_L_x[:, i, :], U_R_x[:, i, :], gamma)
    
    # Y-direction fluxes (reflective BC)
    U_L_y, U_R_y = reconstruct_y(U, dy)
    G_y = np.zeros((4, Nx, Ny+1))
    for j in range(Ny+1):
        G_y[:, :, j] = hllc_flux_y(U_L_y[:, :, j], U_R_y[:, :, j], gamma)
    
    # Compute RHS: -dF/dx - dG/dy
    dU_dt = np.zeros_like(U)
    for k in range(4):
        dU_dt[k] = -(F_x[k, 1:, :] - F_x[k, :-1, :]) / dx - \
                    (G_y[k, :, 1:] - G_y[k, :, :-1]) / dy
    
    return dU_dt


# ============================================================================
# TIME INTEGRATION
# ============================================================================

def compute_timestep(U, dx, dy, gamma, cfl):
    """
    compute_timestep(): Compute stable timestep based on CFL condition
    Inputs:
        U     - Conservative variables [4, Nx, Ny]
        dx    - Cell spacing in x
        dy    - Cell spacing in y
        gamma - Specific heat ratio
        cfl   - CFL number
    Outputs:
        dt    - Timestep
    """
    rho, u, v, p = conservative_to_primitive(U, gamma)
    a = np.sqrt(gamma * p / (rho + EPSILON))
    
    max_speed = np.max(np.sqrt(u**2 + v**2) + a)
    dt = cfl * min(dx, dy) / (max_speed + EPSILON)
    
    return dt


def rk3_step(U, dx, dy, gamma, dt):
    """
    rk3_step(): Third-order TVD Runge-Kutta time step
    Inputs:
        U     - Conservative variables at current time [4, Nx, Ny]
        dx    - Cell spacing in x
        dy    - Cell spacing in y
        gamma - Specific heat ratio
        dt    - Timestep
    Outputs:
        U_new - Conservative variables at next time [4, Nx, Ny]
    """
    # Stage 1
    k1 = compute_rhs(U, dx, dy, gamma)
    U1 = U + dt * k1
    U1 = enforce_positivity(U1, gamma)
    
    # Stage 2
    k2 = compute_rhs(U1, dx, dy, gamma)
    U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
    U2 = enforce_positivity(U2, gamma)
    
    # Stage 3
    k3 = compute_rhs(U2, dx, dy, gamma)
    U_new = (1.0/3.0) * U + (2.0/3.0) * (U2 + dt * k3)
    U_new = enforce_positivity(U_new, gamma)
    
    return U_new


# ============================================================================
# VISUALIZATION
# ============================================================================

def compute_vorticity(U, dx, dy, gamma):
    """
    compute_vorticity(): Compute vorticity field
    Inputs:
        U     - Conservative variables [4, Nx, Ny]
        dx    - Cell spacing in x
        dy    - Cell spacing in y
        gamma - Specific heat ratio
    Outputs:
        vorticity - Vorticity field [Nx, Ny]
    """
    rho, u, v, p = conservative_to_primitive(U, gamma)
    
    Nx = U.shape[1]
    Ny = U.shape[2]
    vorticity = np.zeros((Nx, Ny))
    
    # Central differences for interior points
    vorticity[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dx) - \
                             (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dy)
    
    # Periodic boundaries in x direction (left and right columns)
    vorticity[0, 1:-1] = (v[1, 1:-1] - v[-1, 1:-1]) / (2*dx) - \
                         (u[0, 2:] - u[0, :-2]) / (2*dy)
    vorticity[-1, 1:-1] = (v[0, 1:-1] - v[-2, 1:-1]) / (2*dx) - \
                          (u[-1, 2:] - u[-1, :-2]) / (2*dy)
    
    # Reflective boundaries in y direction (top and bottom rows) - one-sided differences
    vorticity[1:-1, 0] = (v[2:, 0] - v[:-2, 0]) / (2*dx) - \
                         (u[1:-1, 1] - u[1:-1, 0]) / dy
    vorticity[1:-1, -1] = (v[2:, -1] - v[:-2, -1]) / (2*dx) - \
                          (u[1:-1, -1] - u[1:-1, -2]) / dy
    
    # Corners
    vorticity[0, 0] = (v[1, 0] - v[-1, 0]) / (2*dx) - \
                      (u[0, 1] - u[0, 0]) / dy
    vorticity[0, -1] = (v[1, -1] - v[-1, -1]) / (2*dx) - \
                       (u[0, -1] - u[0, -2]) / dy
    vorticity[-1, 0] = (v[0, 0] - v[-2, 0]) / (2*dx) - \
                       (u[-1, 1] - u[-1, 0]) / dy
    vorticity[-1, -1] = (v[0, -1] - v[-2, -1]) / (2*dx) - \
                        (u[-1, -1] - u[-1, -2]) / dy
    
    return vorticity


def plot_results(U, x, y, dx, dy, gamma, time, frame_number):
    """
    plot_results(): Create and save visualization of density and vorticity
    Inputs:
        U            - Conservative variables [4, Nx, Ny]
        x            - x-coordinates [Nx]
        y            - y-coordinates [Ny]
        dx           - Cell spacing in x
        dy           - Cell spacing in y
        gamma        - Specific heat ratio
        time         - Current simulation time
        frame_number - Frame number for output file naming
    Outputs:
        None (saves figure to file)
    """
    rho = U[0, :, :]
    vorticity = compute_vorticity(U, dx, dy, gamma)
    
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    
    # Density plot
    im1 = ax1.contourf(X, Y, rho, levels=DENSITY_LEVELS, cmap=DENSITY_COLORMAP)
    ax1.set_xlabel('x', fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel('y', fontsize=LABEL_FONTSIZE)
    ax1.set_title(f'Density at t = {time:.3f}', fontsize=TITLE_FONTSIZE)
    ax1.tick_params(labelsize=TICK_FONTSIZE)
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.ax.tick_params(labelsize=COLORBAR_FONTSIZE)
    
    # Vorticity plot
    vort_max = np.max(np.abs(vorticity))
    if vort_max > 0:
        im2 = ax2.contourf(X, Y, vorticity, levels=VORTICITY_LEVELS, 
                           cmap=VORTICITY_COLORMAP, vmin=-vort_max, vmax=vort_max)
    else:
        im2 = ax2.contourf(X, Y, vorticity, levels=VORTICITY_LEVELS, 
                           cmap=VORTICITY_COLORMAP)
    ax2.set_xlabel('x', fontsize=LABEL_FONTSIZE)
    ax2.set_ylabel('y', fontsize=LABEL_FONTSIZE)
    ax2.set_title(f'Vorticity at t = {time:.3f}', fontsize=TITLE_FONTSIZE)
    ax2.tick_params(labelsize=TICK_FONTSIZE)
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.ax.tick_params(labelsize=COLORBAR_FONTSIZE)
    
    plt.tight_layout()
    
    # Save figure with frame number for consistent ordering
    filename = os.path.join(DENSITY_PLOTS_DIR, f'KH_frame_{frame_number:05d}.png')
    plt.savefig(filename, dpi=FIG_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {filename}")


def create_gif():
    """
    create_gif(): Create animated GIF from saved PNG files
    Inputs:
        None (reads from DENSITY_PLOTS_DIR)
    Outputs:
        None (saves GIF to OUTPUT_DIR)
    """
    # Get all PNG files sorted by name
    png_files = sorted(glob.glob(os.path.join(DENSITY_PLOTS_DIR, 'KH_frame_*.png')))
    
    if len(png_files) == 0:
        print("No PNG files found to create GIF")
        return
    
    # Load images
    images = []
    for filename in png_files:
        images.append(Image.open(filename))
    
    # Save as GIF
    gif_path = os.path.join(OUTPUT_DIR, GIF_FILENAME)
    images[0].save(gif_path, save_all=True, append_images=images[1:], 
                   duration=int(1000/GIF_FPS), loop=0)
    
    print(f"\nGIF created: {gif_path}")
    print(f"Total frames: {len(images)}")


# ============================================================================
# MAIN SOLVER
# ============================================================================

def solve_kelvin_helmholtz():
    """
    solve_kelvin_helmholtz(): Main solver routine
    Inputs:
        None (uses global parameters)
    Outputs:
        U - Final conservative variables [4, Nx, Ny]
    """
    print("="*80)
    print("KELVIN-HELMHOLTZ SOLVER - HLLC + VAN LEER (LOW DIFFUSION)")
    print("="*80)
    
    # Create output directories
    create_output_directories()
    
    # Initialize grid
    print("\nInitializing grid...")
    x, y, dx, dy = initialize_grid(NX, NY, LX, LY)
    print(f"Grid: {NX} x {NY} cells")
    print(f"Domain: [{0}, {LX}] x [{0}, {LY}]")
    print(f"Cell size: dx = {dx:.6f}, dy = {dy:.6f}")
    
    # Set up initial conditions
    print("\nSetting up initial conditions...")
    params = {
        'rho1': RHO_1,
        'rho2': RHO_2,
        'u1': U_1,
        'u2': U_2,
        'p0': P_0,
        'gamma': GAMMA,
        'delta': SHEAR_LAYER_THICKNESS,
        'amplitude': PERTURBATION_AMPLITUDE,
        'sigma': PERTURBATION_WIDTH
    }
    U = initialize_kelvin_helmholtz(x, y, params)
    
    # Print simulation parameters
    print(f"\nPhysical parameters:")
    print(f"  Density ratio: {RHO_2/RHO_1:.2f}")
    print(f"  Velocity shear: {U_1 - U_2:.2f}")
    print(f"  Pressure: {P_0:.2f}")
    print(f"  Gamma: {GAMMA:.2f}")
    
    print(f"\nBoundary conditions:")
    print(f"  x-direction: Periodic")
    print(f"  y-direction: Reflective (slip wall)")
    
    print(f"\nNumerical parameters:")
    print(f"  Riemann solver: HLLC (low diffusion)")
    print(f"  Slope limiter: Van Leer")
    print(f"  CFL number: {CFL:.2f}")
    print(f"  Final time: {T_FINAL:.2f}")
    print(f"  Output time interval: {OUTPUT_TIME_INTERVAL:.3f} seconds")
    print(f"  Expected frames: ~{int(T_FINAL/OUTPUT_TIME_INTERVAL)}")
    
    # Time integration
    print("\n" + "="*80)
    print("Starting time integration...")
    print("="*80)
    
    t = 0.0
    step = 0
    frame_number = 0
    next_output_time = 0.0
    
    # Save initial condition
    plot_results(U, x, y, dx, dy, GAMMA, t, frame_number)
    frame_number += 1
    next_output_time += OUTPUT_TIME_INTERVAL
    
    while t < T_FINAL:
        # Compute timestep
        dt = compute_timestep(U, dx, dy, GAMMA, CFL)
        
        # Check for NaN or invalid timestep
        if not np.isfinite(dt) or dt <= 0:
            print(f"\nERROR: Invalid timestep dt = {dt}")
            print("Simulation terminated early due to numerical instability")
            break
        
        dt = min(dt, T_FINAL - t)  # Don't overshoot final time
        
        # Take RK3 step
        U = rk3_step(U, dx, dy, GAMMA, dt)
        
        # Check for NaN in solution
        if not np.all(np.isfinite(U)):
            print(f"\nERROR: NaN detected in solution at step {step}")
            print("Simulation terminated early due to numerical instability")
            break
        
        # Update time and step
        t += dt
        step += 1
        
        # Output at constant time intervals
        if t >= next_output_time:
            print(f"Step {step:5d}: t = {t:.4f}, dt = {dt:.6f}")
            plot_results(U, x, y, dx, dy, GAMMA, t, frame_number)
            frame_number += 1
            next_output_time += OUTPUT_TIME_INTERVAL
    
    print("\n" + "="*80)
    print("Time integration complete!")
    print(f"Final time: {t:.4f}")
    print(f"Total steps: {step}")
    print(f"Total frames saved: {frame_number}")
    print("="*80)
    
    # Create GIF
    print("\nCreating animated GIF...")
    create_gif()
    
    print("\nSimulation complete!")
    
    return U


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    U_final = solve_kelvin_helmholtz()
