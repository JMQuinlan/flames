# -*- coding: utf-8 -*-
"""
===============================================================================
RAYLEIGH-TAYLOR INSTABILITY: PYTHON vs HYDRO2 COMPARISON ANALYSIS
===============================================================================
PURPOSE:
    Compare Python finite volume solver (reference) with hydro2 AMReX solver
    Generate side-by-side visualizations, difference plots, and animated GIFs

WORKFLOW:
    1. Extract timesteps from hydro2 VisIt files
    2. Run Python FV solver and save at matching timesteps
    3. Generate comparison plots: Python (left) vs hydro2 (right)
    4. Generate difference and error plots
    5. Create animated GIFs for all metrics

OUTPUTS:
    - Organized subfolders: density/, velocity/, vorticity/, difference_density/, etc.
    - Side-by-side comparison images (no gap between left/right)
    - Difference plots with error metrics
    - Animated GIFs for each metric

USAGE:
    1. Configure paths and parameters in CONFIGURATION section
    2. Run: python RT_Comparison_Analysis.py
    3. Check output folders for results

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
HYDRO2_OUTPUT_DIR = r'../../../bin/tests/RayleighTaylor/RayleighTaylor_UNIT_TEST'  # Path to hydro2 AMReX output
OUTPUT_FOLDER = './RT_Comparison_Analysis'

# -------------------- TIME SYNCHRONIZATION --------------------
TIME_TOLERANCE_MS = 500.0  # Match timesteps within this tolerance [milliseconds]

# -------------------- PLOT TOGGLES --------------------
PLOT_DENSITY = 1
PLOT_VELOCITY = 1
PLOT_VORTICITY = 1
PLOT_DIFFERENCES = 1
GENERATE_GIFS = 1

# -------------------- PLOTTING PARAMETERS --------------------
# Font sizes
FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
FONT_SIZE_TIMESTAMP = 14

# Line properties
LINE_WIDTH_THICK = 2.5
LINE_WIDTH_NORMAL = 2.0
LINE_WIDTH_THIN = 1.5
CONTOUR_LINE_WIDTH = 2.0

# Figure settings
DPI = 300
FIGURE_WIDTH = 16  # Width for side-by-side plots
FIGURE_HEIGHT = 8  # Height for side-by-side plots

# Colormaps
COLORMAP_DENSITY = 'viridis'
COLORMAP_VELOCITY = 'plasma'
COLORMAP_VORTICITY = 'RdBu_r'
COLORMAP_DIFFERENCE = 'seismic'  # Diverging colormap for differences

# Visualization options
STREAMLINE_SHOW = True
STREAMLINE_DENSITY = 1.5
STREAMLINE_COLOR = 'white'
STREAMLINE_LINEWIDTH = 0.8

# -------------------- PHYSICAL PARAMETERS (from Python script) --------------------
GAMMA = 1.4  # Ideal gas gamma
GRAVITY = -0.1  # Gravity strength
P0 = 2.5  # Reference pressure
W0 = 0.0025  # Initial perturbation amplitude

# Domain (from Python script)
BOXSIZE_X = 0.5
BOXSIZE_Y = 1.5
INTERFACE_Y = 0.75  # Interface location

# Fluid properties
RHO_HEAVY = 2.0  # Heavy fluid (top)
RHO_LIGHT = 1.0  # Light fluid (bottom)

# -------------------- PYTHON SOLVER PARAMETERS --------------------
N_RESOLUTION = 64  # Resolution N x 3N (64 x 192)
COURANT_FAC = 0.4
USE_SLOPE_LIMITING = False

# -------------------- GIF PARAMETERS --------------------
GIF_FPS = 10
GIF_LOOP = 0
GIF_OPTIMIZE = True

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Create subfolders for each metric
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
print("RAYLEIGH-TAYLOR INSTABILITY: PYTHON vs HYDRO2 COMPARISON")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_FOLDER}")
print(f"Time tolerance: {TIME_TOLERANCE_MS} ms")

# ============================================================================
# PYTHON FINITE VOLUME SOLVER (from provided script)
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
    vx = Momx / rho / vol
    vy = Momy / rho / vol
    P = (Energy/vol - 0.5*rho * (vx**2+vy**2)) * (gamma-1)
    rho, vx, vy, P = setGhostCells(rho, vx, vy, P)
    return rho, vx, vy, P

def getGradient(f, dx):
    """Calculate gradients of a field"""
    R = -1  # right
    L = 1   # left
    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2*dx)
    f_dy = (np.roll(f, R, axis=1) - np.roll(f, L, axis=1)) / (2*dx)
    f_dx, f_dy = setGhostGradients(f_dx, f_dy)
    return f_dx, f_dy

def slopeLimit(f, dx, f_dx, f_dy):
    """Apply slope limiter to slopes"""
    R = -1  # right
    L = 1   # left
    f_dx = np.maximum(0., np.minimum(1., ((f-np.roll(f,L,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
    f_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=0))/dx)/(f_dx + 1.0e-8*(f_dx==0)))) * f_dx
    f_dy = np.maximum(0., np.minimum(1., ((f-np.roll(f,L,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
    f_dy = np.maximum(0., np.minimum(1., (-(f-np.roll(f,R,axis=1))/dx)/(f_dy + 1.0e-8*(f_dy==0)))) * f_dy
    return f_dx, f_dy

def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """Extrapolate field to face centers"""
    R = -1  # right
    L = 1   # left
    f_XL = f - f_dx * dx/2
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx/2
    f_YL = f - f_dy * dx/2
    f_YL = np.roll(f_YL, R, axis=1)
    f_YR = f + f_dy * dx/2
    return f_XL, f_XR, f_YL, f_YR

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply fluxes to conserved variables"""
    R = -1  # right
    L = 1   # left
    F += - dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, L, axis=0)
    F += - dt * dx * flux_F_Y
    F += dt * dx * np.roll(flux_F_Y, L, axis=1)
    return F

def getFlux(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, P_L, P_R, gamma):
    """Calculate fluxes with local Lax-Friedrichs/Rusanov rule"""
    en_L = P_L/(gamma-1) + 0.5*rho_L * (vx_L**2+vy_L**2)
    en_R = P_R/(gamma-1) + 0.5*rho_R * (vx_R**2+vy_R**2)
    
    rho_star = 0.5*(rho_L + rho_R)
    momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R)
    en_star = 0.5*(en_L + en_R)
    
    P_star = (gamma-1)*(en_star - 0.5*(momx_star**2+momy_star**2)/rho_star)
    
    flux_Mass = momx_star
    flux_Momx = momx_star**2/rho_star + P_star
    flux_Momy = momx_star * momy_star/rho_star
    flux_Energy = (en_star+P_star) * momx_star/rho_star
    
    C_L = np.sqrt(gamma*P_L/rho_L) + np.abs(vx_L)
    C_R = np.sqrt(gamma*P_R/rho_R) + np.abs(vx_R)
    C = np.maximum(C_L, C_R)
    
    flux_Mass -= C * 0.5 * (rho_L - rho_R)
    flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)
    flux_Momy -= C * 0.5 * (rho_L * vy_L - rho_R * vy_R)
    flux_Energy -= C * 0.5 * (en_L - en_R)
    
    return flux_Mass, flux_Momx, flux_Momy, flux_Energy

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
    Run Python finite volume solver and save at target times
    
    Parameters:
        target_times: array of times to save output [seconds]
    
    Returns:
        saved_data: list of dicts containing {t, rho, vx, vy, P, x_grid, y_grid}
    """
    print("\n" + "=" * 80)
    print("RUNNING PYTHON FINITE VOLUME SOLVER")
    print("=" * 80)
    
    # Sort target times
    target_times = np.sort(target_times)
    
    # Mesh
    dx = BOXSIZE_X / N_RESOLUTION
    vol = dx**2
    xlin = np.linspace(0.5*dx, BOXSIZE_X-0.5*dx, N_RESOLUTION)
    ylin = np.linspace(0.5*dx, BOXSIZE_Y-0.5*dx, 3*N_RESOLUTION)
    Y, X = np.meshgrid(ylin, xlin)
    
    # Initial conditions
    rho = 1. + (Y > INTERFACE_Y)
    vx = np.zeros(X.shape)
    vy = W0 * (1-np.cos(4*np.pi*X)) * (1-np.cos(4*np.pi*Y/3))
    P = P0 + GRAVITY * (Y - INTERFACE_Y) * rho
    
    rho, vx, vy, P = addGhostCells(rho, vx, vy, P)
    
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
    
    while t < max_time:
        # Get primitive variables
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, GAMMA, vol)
        
        # Get time step (CFL)
        dt = COURANT_FAC * np.min(dx / (np.sqrt(GAMMA*P/rho) + np.sqrt(vx**2+vy**2)))
        
        # Check if we need to adjust dt to hit a target time exactly
        save_this_step = False
        if target_idx < len(target_times):
            if t + dt >= target_times[target_idx]:
                # Adjust dt to hit target time exactly
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
        
        # Compute fluxes
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = getFlux(
            rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, GAMMA)
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = getFlux(
            rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, GAMMA)
        
        # Update solution
        Mass = applyFluxes(Mass, flux_Mass_X, flux_Mass_Y, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, flux_Momx_Y, dx, dt)
        Momy = applyFluxes(Momy, flux_Momy_X, flux_Momy_Y, dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)
        
        # Add source (half-step)
        Mass, Momx, Momy, Energy = addSourceTerm(Mass, Momx, Momy, Energy, GRAVITY, dt/2)
        
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
            
            saved_data.append({
                't': t,
                'rho': rho_save.copy(),
                'vx': vx_save.copy(),
                'vy': vy_save.copy(),
                'P': P_save.copy(),
                'x_grid': X.copy(),
                'y_grid': Y.copy()
            })
            
            print(f"  Saved timestep {target_idx+1}/{len(target_times)}: t = {t:.6e} s (iteration {iteration})")
            target_idx += 1
        
        # Progress update
        if iteration >= next_print_iter:
            print(f"  Iteration {iteration}: t = {t:.6e} s ({t/max_time*100:.1f}% complete)")
            next_print_iter += 100
    
    print(f"\nPython solver complete: {len(saved_data)} timesteps saved")
    return saved_data

# ============================================================================
# HELPER FUNCTIONS
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
    # Compute gradients
    # np.gradient with scalar spacing: np.gradient(f, spacing, axis=axis)
    # axis=0 is the first dimension (rows, y-direction)
    # axis=1 is the second dimension (columns, x-direction)
    dvx_dy = np.gradient(vx, dy, axis=0)  # Derivative in y-direction
    dvy_dx = np.gradient(vy, dx, axis=1)  # Derivative in x-direction
    vorticity = dvy_dx - dvx_dy
    return vorticity


def interpolate_to_common_grid(data, x_old, y_old, x_new, y_new):
    """
    Interpolate data from old grid to new grid using RegularGridInterpolator
    
    Parameters:
        data: 2D array on old grid (shape: [nx_old, ny_old])
        x_old, y_old: 1D arrays of old grid coordinates
        x_new, y_new: 2D meshgrid arrays of new grid coordinates
    
    Returns:
        data_new: 2D array interpolated to new grid
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    
    # Make copies to avoid modifying original arrays
    x_old_copy = np.asarray(x_old).copy()
    y_old_copy = np.asarray(y_old).copy()
    data_copy = np.asarray(data).copy()
    
    # Print debug info
    print(f"\n=== INTERPOLATION DEBUG ===")
    print(f"Input data shape: {data_copy.shape}")
    print(f"x_old shape: {x_old_copy.shape}, first 5: {x_old_copy[:5]}")
    print(f"y_old shape: {y_old_copy.shape}, first 5: {y_old_copy[:5]}")
    print(f"x_old is ascending: {np.all(np.diff(x_old_copy) > 0)}")
    print(f"y_old is ascending: {np.all(np.diff(y_old_copy) > 0)}")
    
    # Check for and handle non-ascending x coordinates
    if len(x_old_copy) > 1:
        x_diff = np.diff(x_old_copy)
        if np.any(x_diff <= 0):
            print(f"WARNING: x_old is not strictly ascending!")
            print(f"  x_diff min: {np.min(x_diff)}, max: {np.max(x_diff)}")
            if np.all(x_diff < 0):
                # Descending - reverse it
                print("  Reversing x_old and data along axis 0")
                x_old_copy = x_old_copy[::-1]
                data_copy = data_copy[::-1, :]
            else:
                print("  ERROR: x_old has mixed ascending/descending or duplicates!")
                return None
    
    # Check for and handle non-ascending y coordinates
    if len(y_old_copy) > 1:
        y_diff = np.diff(y_old_copy)
        if np.any(y_diff <= 0):
            print(f"WARNING: y_old is not strictly ascending!")
            print(f"  y_diff min: {np.min(y_diff)}, max: {np.max(y_diff)}")
            if np.all(y_diff < 0):
                # Descending - reverse it
                print("  Reversing y_old and data along axis 1")
                y_old_copy = y_old_copy[::-1]
                data_copy = data_copy[:, ::-1]
            else:
                print("  ERROR: y_old has mixed ascending/descending or duplicates!")
                return None
    
    # Final check
    print(f"After processing:")
    print(f"  x_old range: [{x_old_copy[0]:.6e}, {x_old_copy[-1]:.6e}]")
    print(f"  y_old range: [{y_old_copy[0]:.6e}, {y_old_copy[-1]:.6e}]")
    print(f"  data shape: {data_copy.shape}")
    
    # Create interpolator
    # RegularGridInterpolator expects points as (dim0, dim1, ...) 
    # For 2D: (rows, cols) which corresponds to (y, x) in typical meshgrid convention
    try:
        # The data should match the grid: data[i,j] corresponds to (x_old[i], y_old[j])
        # So we need to transpose data to match (y_old, x_old) ordering
        interp_func = RegularGridInterpolator(
            (x_old_copy, y_old_copy),  # Grid points
            data_copy,                  # Values at grid points
            bounds_error=False, 
            fill_value=None,
            method='linear'
        )
        
        # Create interpolation points
        # Flatten the 2D grids and stack as (N, 2) array
        points = np.column_stack([x_new.ravel(), y_new.ravel()])
        
        # Interpolate
        data_new = interp_func(points).reshape(x_new.shape)
        
        print(f"Interpolation successful! Output shape: {data_new.shape}")
        print("=" * 30 + "\n")
        
        return data_new
        
    except ValueError as e:
        print(f"\nERROR in RegularGridInterpolator: {e}")
        print(f"  Attempted grid: ({x_old_copy.shape}, {y_old_copy.shape})")
        print(f"  Attempted data shape: {data_copy.shape}")
        raise e

 

def create_gif_from_folder(subfolder_name, output_gif_name):
    """
    Create GIF from all images in a subfolder with error handling for corrupted files
    Parameters:
        subfolder_name: Name of subfolder (e.g., 'density')
        output_gif_name: Name for output GIF (e.g., 'ANIM_density.gif')
    """
    subfolder_path = os.path.join(OUTPUT_FOLDER, subfolder_name)
    # Get all PNG files
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
    if not image_files:
        print(f"  WARNING: No images found in {subfolder_name}/")
        return
    # Sort by frame number
    image_files = sorted(image_files, key=get_frame_number)
    # Load images with error handling
    frames = []
    corrupted_files = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(subfolder_path, img_file)
        try:
            # Try to open and verify the image
            img = Image.open(img_path)
            img.load()  # Force load to catch truncated images
            frames.append(img.copy())  # Make a copy to avoid file handle issues
            img.close()
        except (OSError, IOError, struct.error) as e:
            print(f"    WARNING: Skipping corrupted image {img_file}: {e}")
            corrupted_files.append(img_file)
            continue
    if not frames:
        print(f"  ERROR: No valid images found in {subfolder_name}/")
        return
    
    if corrupted_files:
        print(f"  Skipped {len(corrupted_files)} corrupted images:")
        for cf in corrupted_files[:5]:  # Show first 5
            print(f"    - {cf}")
        if len(corrupted_files) > 5:
            print(f"    ... and {len(corrupted_files)-5} more")
    # Save as GIF
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
        # Clean up
        for frame in frames:
            try:
                frame.close()
            except:
                pass


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_density_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """
    Merged density comparison with fixed global colorbar range and smooth contours
    
    Parameters:
        python_data: dict with 'rho', 'x_grid', 'y_grid', 't'
        hydro2_data: dict with 'rho', 'x_grid', 'y_grid', 't'
        frame_num: frame number for filename
        vmin, vmax: GLOBAL colorbar limits (fixed across all frames)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t = python_data['t']
    
    # Get the full domain extent
    x_min = 0.0
    x_max = BOXSIZE_X  # 0.5
    x_split = BOXSIZE_X / 2  # 0.25
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    # Create masks for left and right halves
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    # Mask the data
    python_rho_masked = np.where(python_mask, python_data['rho'], np.nan)
    hydro2_rho_masked = np.where(hydro2_mask, hydro2_data['rho'], np.nan)
    
    # Plot Python solution (left half) with MORE LEVELS for smooth gradients
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'], 
                      python_rho_masked, levels=100, cmap=COLORMAP_DENSITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Plot hydro2 solution (right half) with MORE LEVELS
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      hydro2_rho_masked, levels=100, cmap=COLORMAP_DENSITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Add vertical divider line at x=0.25
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    # Add interface type labels below x-axis
    ax.text(x_split/2, -0.12, 'Sharp', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'Diffuse', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    # Set axis properties
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    
    # Timestamp at top
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    # Vertical colorbar on the right with FIXED range
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Density (kg/m^3)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    # Explicitly set colorbar limits to global values
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    
    save_path = os.path.join(subfolder_density, f'{frame_num:04d}_density.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_velocity_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """
    Merged velocity comparison with fixed global colorbar range and smooth contours
    
    Parameters:
        python_data: dict with 'vx', 'vy', 'x_grid', 'y_grid', 't'
        hydro2_data: dict with 'vx', 'vy', 'x_grid', 'y_grid', 't'
        frame_num: frame number for filename
        vmin, vmax: GLOBAL colorbar limits (fixed across all frames)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t = python_data['t']
    
    # Get the full domain extent
    x_min = 0.0
    x_max = BOXSIZE_X
    x_split = BOXSIZE_X / 2
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    # Compute velocity magnitude
    v_mag_python = np.sqrt(python_data['vx']**2 + python_data['vy']**2)
    v_mag_hydro2 = np.sqrt(hydro2_data['vx']**2 + hydro2_data['vy']**2)
    
    # Create masks for left and right halves
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    # Mask the data
    v_mag_python_masked = np.where(python_mask, v_mag_python, np.nan)
    v_mag_hydro2_masked = np.where(hydro2_mask, v_mag_hydro2, np.nan)
    
    # Plot Python solution (left half) with MORE LEVELS
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'],
                      v_mag_python_masked, levels=100, cmap=COLORMAP_VELOCITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Plot hydro2 solution (right half) with MORE LEVELS
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      v_mag_hydro2_masked, levels=100, cmap=COLORMAP_VELOCITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Add streamlines (optional - only in respective halves)
    if STREAMLINE_SHOW:
        try:
            # Python streamlines (left half)
            x_python_1d = python_data['x_grid'][0, :]
            y_python_1d = python_data['y_grid'][:, 0]
            mask_left = x_python_1d <= x_split
            if np.any(mask_left):
                ax.streamplot(x_python_1d[mask_left], y_python_1d,
                             python_data['vx'][:, mask_left], python_data['vy'][:, mask_left],
                             color=STREAMLINE_COLOR, density=STREAMLINE_DENSITY,
                             linewidth=STREAMLINE_LINEWIDTH, arrowsize=0.8)
            
            # hydro2 streamlines (right half)
            x_hydro2_1d = hydro2_data['x_grid'][0, :]
            y_hydro2_1d = hydro2_data['y_grid'][:, 0]
            mask_right = x_hydro2_1d > x_split
            if np.any(mask_right):
                ax.streamplot(x_hydro2_1d[mask_right], y_hydro2_1d,
                             hydro2_data['vx'][:, mask_right], hydro2_data['vy'][:, mask_right],
                             color=STREAMLINE_COLOR, density=STREAMLINE_DENSITY,
                             linewidth=STREAMLINE_LINEWIDTH, arrowsize=0.8)
        except:
            pass  # Skip streamlines if they fail
    
    # Add vertical divider line at x=0.25
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    # Add interface type labels below x-axis
    ax.text(x_split/2, -0.12, 'Sharp', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'Diffuse', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    # Set axis properties
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    
    # Timestamp at top
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    # Vertical colorbar on the right with FIXED range
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    
    save_path = os.path.join(subfolder_velocity, f'{frame_num:04d}_velocity.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_vorticity_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    """
    Merged vorticity comparison with fixed global colorbar range and smooth contours
    
    Parameters:
        python_data: dict with 'vorticity', 'x_grid', 'y_grid', 't'
        hydro2_data: dict with 'vorticity', 'x_grid', 'y_grid', 't'
        frame_num: frame number for filename
        vmin, vmax: GLOBAL colorbar limits (fixed across all frames)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t = python_data['t']
    
    # Get the full domain extent
    x_min = 0.0
    x_max = BOXSIZE_X
    x_split = BOXSIZE_X / 2
    y_min = 0.0
    y_max = BOXSIZE_Y
    
    # Create masks for left and right halves
    python_mask = python_data['x_grid'] <= x_split
    hydro2_mask = hydro2_data['x_grid'] > x_split
    
    # Mask the data
    vort_python_masked = np.where(python_mask, python_data['vorticity'], np.nan)
    vort_hydro2_masked = np.where(hydro2_mask, hydro2_data['vorticity'], np.nan)
    
    # Plot Python solution (left half) with MORE LEVELS
    im1 = ax.contourf(python_data['x_grid'], python_data['y_grid'],
                      vort_python_masked, levels=100, cmap=COLORMAP_VORTICITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Plot hydro2 solution (right half) with MORE LEVELS
    im2 = ax.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                      vort_hydro2_masked, levels=100, cmap=COLORMAP_VORTICITY,
                      vmin=vmin, vmax=vmax, extend='neither')
    
    # Add vertical divider line at x=0.25
    ax.axvline(x=x_split, color='black', linewidth=2.5, linestyle='-', zorder=10)
    
    # Add interface type labels below x-axis
    ax.text(x_split/2, -0.12, 'Sharp', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    ax.text(x_split + (x_max-x_split)/2, -0.12, 'Diffuse', 
            ha='center', va='top', fontsize=FONT_SIZE_LABEL+2, 
            fontweight='bold', transform=ax.transData)
    
    # Set axis properties
    ax.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal', adjustable='box')
    
    # Timestamp at top
    ax.set_title(f't = {t*1e3:.2f} ms', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', pad=15)
    
    # Vertical colorbar on the right with FIXED range
    cbar = fig.colorbar(im2, ax=ax, orientation='vertical', extend="neither", pad=0.02, aspect=30)
    cbar.set_label('Vorticity (1/s)', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    cbar.mappable.set_clim(vmin, vmax)
    
    plt.tight_layout()
    
    save_path = os.path.join(subfolder_vorticity, f'{frame_num:04d}_vorticity.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def plot_difference(python_data, hydro2_data, field_name, frame_num, vmin_diff, vmax_diff):
    """
    Plot difference (hydro2 - Python) with error metrics
    
    Parameters:
        python_data: dict with field data
        hydro2_data: dict with field data
        field_name: 'rho', 'v_mag', or 'vorticity'
        frame_num: frame number for filename
        vmin_diff, vmax_diff: colorbar limits for difference
    """
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
    
    ax_main = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')
    
    t = python_data['t']
    
    # Get field data
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
    
    # Compute difference
    difference = hydro2_field - python_field
    
    # Compute error metrics
    abs_error = np.abs(difference)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    rms_error = np.sqrt(np.mean(difference**2))
    
    # Relative error (avoid division by zero)
    python_max = np.max(np.abs(python_field))
    if python_max > 1e-10:
        rel_error_max = max_error / python_max * 100
        rel_error_rms = rms_error / python_max * 100
    else:
        rel_error_max = 0.0
        rel_error_rms = 0.0
    
    # L2 norm
    l2_norm = np.sqrt(np.sum(difference**2))
    
    # Plot difference
    im = ax_main.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                         difference, levels=50, cmap=COLORMAP_DIFFERENCE,
                         vmin=vmin_diff, vmax=vmax_diff)
    
    ax_main.set_xlabel('X (m)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_ylabel('Y (m)', fontsize=FONT_SIZE_LABEL)
    ax_main.set_title(f'{field_label} Difference (hydro2 - Python) at t = {t*1e3:.2f} ms',
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_main.set_aspect('equal', adjustable='box')
    
    cbar = plt.colorbar(im, extend="neither", ax=ax_main)
    cbar.set_label(f'Difference ({field_units})', fontsize=FONT_SIZE_LABEL)
    
    # Error metrics text
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
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    # ========================================================================
    # STEP 1: FIND AND LOAD HYDRO2 FILES
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING HYDRO2 OUTPUT FILES")
    print("=" * 80)
    
    plot_files = []
    for item in os.listdir(HYDRO2_OUTPUT_DIR):
        # Skip files containing '.old'
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
    
    # Extract times from hydro2 files
    hydro2_times = []
    for plot_file in plot_files:
        ds = yt.load(plot_file)
        t = float(ds.current_time)
        hydro2_times.append(t)
    
    hydro2_times = np.array(hydro2_times)
    print(f"Time range: {hydro2_times[0]:.6e} to {hydro2_times[-1]:.6e} s")
    
    # ========================================================================
    # STEP 2: RUN PYTHON SOLVER AT MATCHING TIMESTEPS
    # ========================================================================
    python_data_list = run_python_solver(hydro2_times)
    
    # ========================================================================
    # STEP 3: MATCH TIMESTEPS WITHIN TOLERANCE
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: MATCHING TIMESTEPS")
    print("=" * 80)
    
    tolerance_sec = TIME_TOLERANCE_MS * 1e-3
    matched_pairs = []
    
    for i, t_hydro2 in enumerate(hydro2_times):
        # Find closest Python timestep
        time_diffs = np.abs(np.array([d['t'] for d in python_data_list]) - t_hydro2)
        closest_idx = np.argmin(time_diffs)
        
        if time_diffs[closest_idx] <= tolerance_sec:
            matched_pairs.append((i, closest_idx))
            print(f"  Match {len(matched_pairs)}: hydro2 t={t_hydro2:.6e} s, "
                  f"Python t={python_data_list[closest_idx]['t']:.6e} s, "
                  f"diff={time_diffs[closest_idx]*1e3:.4f} ms")
        else:
            print(f"  WARNING: No match for hydro2 t={t_hydro2:.6e} s "
                  f"(closest diff = {time_diffs[closest_idx]*1e3:.4f} ms)")
    
    print(f"\nMatched {len(matched_pairs)} timesteps within {TIME_TOLERANCE_MS} ms tolerance")
    
    if len(matched_pairs) == 0:
        print("ERROR: No matching timesteps found!")
        return
    
    # ========================================================================
    # STEP 4: EXTRACT HYDRO2 DATA AND COMPUTE GLOBAL MIN/MAX
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: EXTRACTING HYDRO2 DATA AND COMPUTING GLOBAL LIMITS")
    print("=" * 80)
    
    hydro2_data_list = []
    
    # Domain parameters (from hydro2 input file)
    domain_width = BOXSIZE_X
    domain_height = BOXSIZE_Y
    resolution = 512  # High resolution for extraction
    
    for hydro2_idx, python_idx in matched_pairs:
        ds = yt.load(plot_files[hydro2_idx])
        t = float(ds.current_time)
        
        # Create slice at z=0
        slc = ds.slice('z', 0.0)
        
        # Create fixed resolution buffer
        frb = slc.to_frb((domain_width, 'code_length'), resolution,
                        center=[0.5*domain_width, 0.5*domain_height, 0.0],
                        height=(domain_height, 'code_length'))
        
        # Extract fields
        eta = np.array(frb['eta'])
        rho = np.array(frb['density'])
        
        vx = np.array(frb['velocityx'])
        vy = np.array(frb['velocityy'])
        vorticity = np.array(frb['vorticity'])
        
        x_1d = np.linspace(0, domain_width, resolution)
        y_1d = np.linspace(0, domain_height, resolution)
        
        # Create grid
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
    
    # ========================================================================
    # STEP 5: INTERPOLATE PYTHON DATA TO HYDRO2 GRID - FIXED
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: INTERPOLATING PYTHON DATA TO COMMON GRID")
    print("=" * 80)

    python_data_interp = []

    for hydro2_idx, python_idx in matched_pairs:
        python_data = python_data_list[python_idx]
        hydro2_data = hydro2_data_list[len(python_data_interp)]
    
        # Python grid extraction - FIXED!
        # The Python solver creates: Y, X = np.meshgrid(ylin, xlin)
        # So X varies along axis=1 (columns) and Y varies along axis=0 (rows)
    
        # Extract unique 1D coordinate arrays
        x_python = python_data['x_grid'][0, :]  # X varies along columns (axis 1)
        y_python = python_data['y_grid'][:, 0]  # Y varies along rows (axis 0)
    
        # DEBUG: Check if coordinates are actually varying
        print(f"\n=== CHECKING PYTHON GRID EXTRACTION ===")
        print(f"x_python shape: {x_python.shape}, unique values: {len(np.unique(x_python))}")
        print(f"y_python shape: {y_python.shape}, unique values: {len(np.unique(y_python))}")
        print(f"x_python range: [{np.min(x_python):.6e}, {np.max(x_python):.6e}]")
        print(f"y_python range: [{np.min(y_python):.6e}, {np.max(y_python):.6e}]")
    
        # If all values are the same, the grid extraction is wrong
        if len(np.unique(x_python)) == 1 or len(np.unique(y_python)) == 1:
            print("\nERROR: Grid coordinates are not varying!")
            print("Attempting alternative extraction...")
        
            # Try alternative: maybe the meshgrid is stored differently
            # Check the actual grid structure
            print(f"\nDEBUG: python_data['x_grid'] shape: {python_data['x_grid'].shape}")
            print(f"First row of x_grid: {python_data['x_grid'][0, :5]}")
            print(f"First col of x_grid: {python_data['x_grid'][:5, 0]}")
            print(f"First row of y_grid: {python_data['y_grid'][0, :5]}")
            print(f"First col of y_grid: {python_data['y_grid'][:5, 0]}")
        
            # Determine which direction varies
            if len(np.unique(python_data['x_grid'][:, 0])) > 1:
                # X varies along rows (axis 0)
                x_python = python_data['x_grid'][:, 0]
                print("  -> X varies along axis 0 (rows)")
            else:
                # X varies along columns (axis 1)
                x_python = python_data['x_grid'][0, :]
                print("  -> X varies along axis 1 (columns)")
        
            if len(np.unique(python_data['y_grid'][0, :])) > 1:
                # Y varies along columns (axis 1)
                y_python = python_data['y_grid'][0, :]
                print("  -> Y varies along axis 1 (columns)")
            else:
                # Y varies along rows (axis 0)
                y_python = python_data['y_grid'][:, 0]
                print("  -> Y varies along axis 0 (rows)")
        
            print(f"\nAfter correction:")
            print(f"x_python: {len(np.unique(x_python))} unique values, range [{np.min(x_python):.6e}, {np.max(x_python):.6e}]")
            print(f"y_python: {len(np.unique(y_python))} unique values, range [{np.min(y_python):.6e}, {np.max(y_python):.6e}]")
    
        # Target grid (2D arrays from hydro2)
        x_target = hydro2_data['x_grid']
        y_target = hydro2_data['y_grid']
    
        # Interpolate fields
        rho_interp = interpolate_to_common_grid(python_data['rho'], x_python, y_python,
                                                x_target, y_target)
        vx_interp = interpolate_to_common_grid(python_data['vx'], x_python, y_python,
                                               x_target, y_target)
        vy_interp = interpolate_to_common_grid(python_data['vy'], x_python, y_python,
                                               x_target, y_target)
    
        # Compute vorticity on interpolated grid
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

    
    # ========================================================================
    # STEP 6: COMPUTE GLOBAL MIN/MAX FOR FIXED COLORBARS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: COMPUTING GLOBAL MIN/MAX FOR FIXED COLORBARS")
    print("=" * 80)
    
    # Density
    rho_min = min(np.min(d['rho']) for d in python_data_interp + hydro2_data_list)
    rho_max = max(np.max(d['rho']) for d in python_data_interp + hydro2_data_list)
    print(f"Density range: [{rho_min:.6e}, {rho_max:.6e}] kg/m^3")
    
    # Velocity magnitude
    v_mag_min = 0.0
    v_mag_max = max(
        max(np.max(np.sqrt(d['vx']**2 + d['vy']**2)) for d in python_data_interp),
        max(np.max(np.sqrt(d['vx']**2 + d['vy']**2)) for d in hydro2_data_list)
    )
    print(f"Velocity magnitude range: [{v_mag_min:.6e}, {v_mag_max:.6e}] m/s")
    
    # Vorticity (symmetric about zero)
    vort_all = np.concatenate([d['vorticity'].flatten() for d in python_data_interp + hydro2_data_list])
    vort_std = np.std(vort_all)
    vort_lim = 2.0 * vort_std  # +/- 2 sigma
    print(f"Vorticity range (+/-2sigma): [{-vort_lim:.6e}, {vort_lim:.6e}] 1/s")
    
    # Difference limits (symmetric about zero)
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
    
    # ========================================================================
    # STEP 7: GENERATE COMPARISON PLOTS
    # ========================================================================
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
    
    # ========================================================================
    # STEP 8: GENERATE GIFS
    # ========================================================================
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
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
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
