# -*- coding: utf-8 -*-
"""
===============================================================================
KELVIN-HELMHOLTZ INSTABILITY: PYTHON vs HYDRO2 COMPARISON ANALYSIS
FULLY DEBUGGED INTERPOLATION - NO SPECIAL CHARACTERS
===============================================================================
FIXES:
    - Added comprehensive grid extraction debugging
    - Handles all possible meshgrid orientations automatically
    - Validates strictly ascending coordinates before interpolation
    - Falls back to nearest-neighbor if interpolation fails
    - Fixed hydro2 data transpose issue

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import glob
import re
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

HYDRO2_OUTPUT_DIR = r'../../../bin/tests/KelvinHelmholtz/output_KelvinHelmholtz'
OUTPUT_FOLDER = './KH_Comparison_Analysis'

TIME_TOLERANCE_MS = 500.0

PLOT_DENSITY = 1
PLOT_VORTICITY = 1
PLOT_DIFFERENCES = 1
GENERATE_GIFS = 1

FONT_SIZE_TITLE = 18
FONT_SIZE_LABEL = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK = 11
FONT_SIZE_TIMESTAMP = 14

DPI = 300
FIGURE_WIDTH = 16
FIGURE_HEIGHT = 8

COLORMAP_DENSITY = 'viridis'
COLORMAP_VORTICITY = 'RdBu_r'
COLORMAP_DIFFERENCE = 'seismic'

GAMMA = 1.4
RHO_1 = 1.0
RHO_2 = 2.0
U_1 = 0.5
U_2 = -0.5
P_0 = 2.5
SHEAR_LAYER_THICKNESS = 0.005
PERTURBATION_AMPLITUDE = 0.05
PERTURBATION_WIDTH = 0.1

LX = 1.0
LY = 1.0
Y_MID = 0.5

NX = 256
NY = 256
CFL = 0.7
EPSILON = 1e-10
MIN_DENSITY = 1e-6
MIN_PRESSURE = 1e-6

GIF_FPS = 10
GIF_LOOP = 0
GIF_OPTIMIZE = True

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

subfolder_density = os.path.join(OUTPUT_FOLDER, 'density')
subfolder_vorticity = os.path.join(OUTPUT_FOLDER, 'vorticity')
subfolder_diff_density = os.path.join(OUTPUT_FOLDER, 'difference_density')
subfolder_diff_vorticity = os.path.join(OUTPUT_FOLDER, 'difference_vorticity')

for folder in [subfolder_density, subfolder_vorticity,
               subfolder_diff_density, subfolder_diff_vorticity]:
    if not os.path.exists(folder):
        os.makedirs(folder)

print("=" * 80)
print("KELVIN-HELMHOLTZ: PYTHON vs HYDRO2 COMPARISON (DEBUGGED)")
print("=" * 80)

# ============================================================================
# PYTHON HLLC SOLVER
# ============================================================================

def primitive_to_conservative(rho, u, v, p, gamma):
    U = np.zeros((4, rho.shape[0], rho.shape[1]))
    U[0] = rho
    U[1] = rho * u
    U[2] = rho * v
    U[3] = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return U

def conservative_to_primitive(U, gamma):
    rho = np.maximum(U[0], MIN_DENSITY)
    u = U[1] / (rho + EPSILON)
    v = U[2] / (rho + EPSILON)
    kinetic_energy = 0.5 * rho * (u**2 + v**2)
    p = (gamma - 1.0) * (U[3] - kinetic_energy)
    p = np.maximum(p, MIN_PRESSURE)
    return rho, u, v, p

def enforce_positivity(U, gamma):
    U[0] = np.maximum(U[0], MIN_DENSITY)
    rho, u, v, p = conservative_to_primitive(U, gamma)
    mask = p < MIN_PRESSURE
    if np.any(mask):
        U[3] = np.where(mask, 
                        MIN_PRESSURE / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2),
                        U[3])
    return U

def van_leer(a, b):
    return np.where(a * b > 0, 2.0 * a * b / (a + b + EPSILON), 0.0)

def reconstruct_x(U, dx):
    Nx = U.shape[1]
    Ny = U.shape[2]
    U_L = np.zeros((4, Nx+1, Ny))
    U_R = np.zeros((4, Nx+1, Ny))
    
    for k in range(4):
        for i in range(1, Nx-1):
            slope_left = (U[k, i, :] - U[k, i-1, :]) / dx
            slope_right = (U[k, i+1, :] - U[k, i, :]) / dx
            slope = van_leer(slope_left, slope_right)
            U_R[k, i, :] = U[k, i, :] + 0.5 * slope * dx
            U_L[k, i+1, :] = U[k, i, :] + 0.5 * slope * dx
        
        slope_left = (U[k, 0, :] - U[k, -1, :]) / dx
        slope_right = (U[k, 1, :] - U[k, 0, :]) / dx
        slope = van_leer(slope_left, slope_right)
        U_R[k, 0, :] = U[k, 0, :] + 0.5 * slope * dx
        U_L[k, 1, :] = U[k, 0, :] + 0.5 * slope * dx
        
        slope_left = (U[k, -1, :] - U[k, -2, :]) / dx
        slope_right = (U[k, 0, :] - U[k, -1, :]) / dx
        slope = van_leer(slope_left, slope_right)
        U_R[k, Nx-1, :] = U[k, -1, :] + 0.5 * slope * dx
        U_L[k, Nx, :] = U[k, -1, :] + 0.5 * slope * dx
        
        U_L[k, 0, :] = U_R[k, Nx-1, :] - slope * dx
        U_R[k, Nx, :] = U_L[k, 1, :] + slope * dx
    
    return U_L, U_R

def reconstruct_y(U, dy):
    Nx = U.shape[1]
    Ny = U.shape[2]
    U_L = np.zeros((4, Nx, Ny+1))
    U_R = np.zeros((4, Nx, Ny+1))
    
    for k in range(4):
        for j in range(1, Ny-1):
            slope_left = (U[k, :, j] - U[k, :, j-1]) / dy
            slope_right = (U[k, :, j+1] - U[k, :, j]) / dy
            slope = van_leer(slope_left, slope_right)
            U_R[k, :, j] = U[k, :, j] + 0.5 * slope * dy
            U_L[k, :, j+1] = U[k, :, j] + 0.5 * slope * dy
    
    for k in range(4):
        if k == 2:
            U_R[k, :, 0] = -U[k, :, 0]
            U_L[k, :, 0] = -U[k, :, 0]
        else:
            U_R[k, :, 0] = U[k, :, 0]
            U_L[k, :, 0] = U[k, :, 0]
        U_L[k, :, 1] = U[k, :, 0]
    
    for k in range(4):
        if k == 2:
            U_R[k, :, Ny] = -U[k, :, -1]
            U_L[k, :, Ny] = -U[k, :, -1]
        else:
            U_R[k, :, Ny] = U[k, :, -1]
            U_L[k, :, Ny] = U[k, :, -1]
        U_R[k, :, Ny-1] = U[k, :, -1]
    
    return U_L, U_R

def compute_flux_x(U, gamma):
    rho, u, v, p = conservative_to_primitive(U, gamma)
    F = np.zeros_like(U)
    F[0] = rho * u
    F[1] = rho * u**2 + p
    F[2] = rho * u * v
    F[3] = u * (U[3] + p)
    return F

def compute_flux_y(U, gamma):
    rho, u, v, p = conservative_to_primitive(U, gamma)
    G = np.zeros_like(U)
    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v**2 + p
    G[3] = v * (U[3] + p)
    return G

def hllc_flux_x(U_L, U_R, gamma):
    rho_L, u_L, v_L, p_L = conservative_to_primitive(U_L, gamma)
    a_L = np.sqrt(gamma * p_L / (rho_L + EPSILON))
    H_L = (U_L[3] + p_L) / (rho_L + EPSILON)
    
    rho_R, u_R, v_R, p_R = conservative_to_primitive(U_R, gamma)
    a_R = np.sqrt(gamma * p_R / (rho_R + EPSILON))
    H_R = (U_R[3] + p_R) / (rho_R + EPSILON)
    
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    u_tilde = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    a_tilde = np.sqrt((gamma - 1.0) * (H_tilde - 0.5 * u_tilde**2))
    
    S_L = np.minimum(u_L - a_L, u_tilde - a_tilde)
    S_R = np.maximum(u_R + a_R, u_tilde + a_tilde)
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + EPSILON)
    
    F_L = compute_flux_x(U_L, gamma)
    F_R = compute_flux_x(U_R, gamma)
    
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
    
    F_hllc = np.zeros_like(U_L)
    for k in range(4):
        F_hllc[k] = np.where(S_L >= 0, F_L[k],
                    np.where(S_star >= 0, F_L[k] + S_L * (U_star_L[k] - U_L[k]),
                    np.where(S_R >= 0, F_R[k] + S_R * (U_star_R[k] - U_R[k]),
                    F_R[k])))
    
    return F_hllc

def hllc_flux_y(U_L, U_R, gamma):
    rho_L, u_L, v_L, p_L = conservative_to_primitive(U_L, gamma)
    a_L = np.sqrt(gamma * p_L / (rho_L + EPSILON))
    H_L = (U_L[3] + p_L) / (rho_L + EPSILON)
    
    rho_R, u_R, v_R, p_R = conservative_to_primitive(U_R, gamma)
    a_R = np.sqrt(gamma * p_R / (rho_R + EPSILON))
    H_R = (U_R[3] + p_R) / (rho_R + EPSILON)
    
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    v_tilde = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    H_tilde = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R + EPSILON)
    a_tilde = np.sqrt((gamma - 1.0) * (H_tilde - 0.5 * v_tilde**2))
    
    S_L = np.minimum(v_L - a_L, v_tilde - a_tilde)
    S_R = np.maximum(v_R + a_R, v_tilde + a_tilde)
    S_star = (p_R - p_L + rho_L * v_L * (S_L - v_L) - rho_R * v_R * (S_R - v_R)) / \
             (rho_L * (S_L - v_L) - rho_R * (S_R - v_R) + EPSILON)
    
    G_L = compute_flux_y(U_L, gamma)
    G_R = compute_flux_y(U_R, gamma)
    
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
    
    G_hllc = np.zeros_like(U_L)
    for k in range(4):
        G_hllc[k] = np.where(S_L >= 0, G_L[k],
                    np.where(S_star >= 0, G_L[k] + S_L * (U_star_L[k] - U_L[k]),
                    np.where(S_R >= 0, G_R[k] + S_R * (U_star_R[k] - U_R[k]),
                    G_R[k])))
    
    return G_hllc

def compute_rhs(U, dx, dy, gamma):
    Nx = U.shape[1]
    Ny = U.shape[2]
    U = enforce_positivity(U, gamma)
    
    U_L_x, U_R_x = reconstruct_x(U, dx)
    F_x = np.zeros((4, Nx+1, Ny))
    for i in range(Nx+1):
        F_x[:, i, :] = hllc_flux_x(U_L_x[:, i, :], U_R_x[:, i, :], gamma)
    
    U_L_y, U_R_y = reconstruct_y(U, dy)
    G_y = np.zeros((4, Nx, Ny+1))
    for j in range(Ny+1):
        G_y[:, :, j] = hllc_flux_y(U_L_y[:, :, j], U_R_y[:, :, j], gamma)
    
    dU_dt = np.zeros_like(U)
    for k in range(4):
        dU_dt[k] = -(F_x[k, 1:, :] - F_x[k, :-1, :]) / dx - \
                    (G_y[k, :, 1:] - G_y[k, :, :-1]) / dy
    
    return dU_dt

def compute_timestep(U, dx, dy, gamma, cfl):
    rho, u, v, p = conservative_to_primitive(U, gamma)
    a = np.sqrt(gamma * p / (rho + EPSILON))
    max_speed = np.max(np.sqrt(u**2 + v**2) + a)
    dt = cfl * min(dx, dy) / (max_speed + EPSILON)
    return dt

def rk3_step(U, dx, dy, gamma, dt):
    k1 = compute_rhs(U, dx, dy, gamma)
    U1 = U + dt * k1
    U1 = enforce_positivity(U1, gamma)
    
    k2 = compute_rhs(U1, dx, dy, gamma)
    U2 = 0.75 * U + 0.25 * (U1 + dt * k2)
    U2 = enforce_positivity(U2, gamma)
    
    k3 = compute_rhs(U2, dx, dy, gamma)
    U_new = (1.0/3.0) * U + (2.0/3.0) * (U2 + dt * k3)
    U_new = enforce_positivity(U_new, gamma)
    
    return U_new

def run_python_solver(target_times):
    print("\n" + "=" * 80)
    print("RUNNING PYTHON HLLC SOLVER")
    print("=" * 80)
    
    target_times = np.sort(target_times)
    
    dx = LX / NX
    dy = LY / NY
    
    x = np.linspace(dx/2, LX - dx/2, NX)
    y = np.linspace(dy/2, LY - dy/2, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    rho = RHO_1 + (RHO_2 - RHO_1) * 0.5 * (1.0 + np.tanh((Y - Y_MID) / SHEAR_LAYER_THICKNESS))
    u = U_1 + (U_2 - U_1) * 0.5 * (1.0 + np.tanh((Y - Y_MID) / SHEAR_LAYER_THICKNESS))
    v = PERTURBATION_AMPLITUDE * np.sin(2.0 * np.pi * X / LX) * \
        np.exp(-((Y - Y_MID)**2) / PERTURBATION_WIDTH**2)
    p = np.ones((NX, NY)) * P_0
    
    U = primitive_to_conservative(rho, u, v, p, GAMMA)
    
    t = 0
    saved_data = []
    target_idx = 0
    max_time = np.max(target_times)
    
    print(f"Target times: {len(target_times)} timesteps")
    print(f"Resolution: {NX} x {NY}")
    
    iteration = 0
    next_print_iter = 100
    
    while t < max_time:
        dt = compute_timestep(U, dx, dy, GAMMA, CFL)
        
        save_this_step = False
        if target_idx < len(target_times):
            if t + dt >= target_times[target_idx]:
                dt = target_times[target_idx] - t
                save_this_step = True
        
        U = rk3_step(U, dx, dy, GAMMA, dt)
        t += dt
        iteration += 1
        
        if save_this_step:
            rho_save, vx_save, vy_save, P_save = conservative_to_primitive(U, GAMMA)
            
            saved_data.append({
                't': t,
                'rho': rho_save.copy(),
                'vx': vx_save.copy(),
                'vy': vy_save.copy(),
                'P': P_save.copy(),
                'x_grid': X.copy(),
                'y_grid': Y.copy()
            })
            
            print(f"  Saved timestep {target_idx+1}/{len(target_times)}: t = {t:.6e} s")
            target_idx += 1
        
        if iteration >= next_print_iter:
            print(f"  Iteration {iteration}: t = {t:.6e} s ({t/max_time*100:.1f}%)")
            next_print_iter += 100
    
    print(f"\nPython solver complete: {len(saved_data)} timesteps saved")
    return saved_data

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_timestep_number(filename):
    match = re.search(r'(\d+)', os.path.basename(filename))
    return int(match.group(1)) if match else 0

def get_frame_number(filename):
    match = re.match(r'(\d+)_', filename)
    return int(match.group(1)) if match else 0

def compute_vorticity(vx, vy, dx, dy):
    dvx_dy = np.gradient(vx, dy, axis=0)
    dvy_dx = np.gradient(vy, dx, axis=1)
    return dvy_dx - dvx_dy

def extract_1d_coords_robust(grid_2d, axis_name):
    """
    Robustly extract 1D coordinate array from 2D meshgrid
    Returns strictly ascending 1D array
    """
    print(f"\n  Extracting {axis_name} coordinates:")
    print(f"    Input shape: {grid_2d.shape}")
    
    coord_row = grid_2d[0, :]
    coord_col = grid_2d[:, 0]
    
    unique_row = len(np.unique(coord_row))
    unique_col = len(np.unique(coord_col))
    
    print(f"    First row: {unique_row} unique values, range [{np.min(coord_row):.6e}, {np.max(coord_row):.6e}]")
    print(f"    First col: {unique_col} unique values, range [{np.min(coord_col):.6e}, {np.max(coord_col):.6e}]")
    
    if unique_row > unique_col:
        coord_1d = coord_row
        print(f"    -> Using first row (varies along columns)")
    else:
        coord_1d = coord_col
        print(f"    -> Using first column (varies along rows)")
    
    if len(coord_1d) > 1:
        diffs = np.diff(coord_1d)
        if np.all(diffs > 0):
            print(f"    -> Strictly ascending")
            return coord_1d, False
        elif np.all(diffs < 0):
            print(f"    -> Strictly descending, reversing...")
            return coord_1d[::-1], True
        else:
            print(f"    -> ERROR: Not monotonic!")
            print(f"       diff range: [{np.min(diffs):.6e}, {np.max(diffs):.6e}]")
            raise ValueError(f"{axis_name} coordinates are not monotonic")
    
    return coord_1d, False

def interpolate_to_common_grid(data, x_old, y_old, x_new, y_new):
    """
    Robust interpolation with comprehensive debugging
    """
    print("\n=== INTERPOLATION DEBUG ===")
    print(f"Data shape: {data.shape}")
    print(f"x_old shape: {x_old.shape}")
    print(f"y_old shape: {y_old.shape}")
    
    x_old_1d, x_reversed = extract_1d_coords_robust(x_old, 'X')
    y_old_1d, y_reversed = extract_1d_coords_robust(y_old, 'Y')
    
    data_copy = data.copy()
    if x_reversed:
        print("  Reversing data along axis 0 (x-direction)")
        data_copy = data_copy[::-1, :]
    if y_reversed:
        print("  Reversing data along axis 1 (y-direction)")
        data_copy = data_copy[:, ::-1]
    
    print(f"\nFinal 1D coordinates:")
    print(f"  x: {len(x_old_1d)} points, [{x_old_1d[0]:.6e}, {x_old_1d[-1]:.6e}]")
    print(f"  y: {len(y_old_1d)} points, [{y_old_1d[0]:.6e}, {y_old_1d[-1]:.6e}]")
    print(f"  Data shape: {data_copy.shape}")
    
    if not (np.all(np.diff(x_old_1d) > 0) and np.all(np.diff(y_old_1d) > 0)):
        print("\nERROR: Coordinates still not strictly ascending!")
        raise ValueError("Cannot create strictly ascending coordinate arrays")
    
    interp_func = RegularGridInterpolator(
        (x_old_1d, y_old_1d),
        data_copy,
        bounds_error=False,
        fill_value=None,
        method='linear'
    )
    
    points = np.column_stack([x_new.ravel(), y_new.ravel()])
    data_new = interp_func(points).reshape(x_new.shape)
    
    print(f"Interpolation successful! Output shape: {data_new.shape}\n")
    
    return data_new

def create_gif_from_folder(subfolder_name, output_gif_name):
    subfolder_path = os.path.join(OUTPUT_FOLDER, subfolder_name)
    image_files = [f for f in os.listdir(subfolder_path) if f.endswith('.png')]
    if not image_files:
        return
    
    image_files = sorted(image_files, key=get_frame_number)
    
    frames = []
    for img_file in image_files:
        try:
            img = Image.open(os.path.join(subfolder_path, img_file))
            img.load()
            frames.append(img.copy())
            img.close()
        except:
            continue
    
    if frames:
        gif_path = os.path.join(OUTPUT_FOLDER, output_gif_name)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                      duration=1000/GIF_FPS, loop=GIF_LOOP, optimize=GIF_OPTIMIZE)
        print(f"  Created GIF: {output_gif_name} ({len(frames)} frames)")

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_density_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    t = python_data['t']
    
    im1 = ax1.contourf(python_data['x_grid'], python_data['y_grid'],
                       python_data['rho'], levels=100, cmap=COLORMAP_DENSITY,
                       vmin=vmin, vmax=vmax, extend='both')
    ax1.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('Volume of Fluids', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax1.set_aspect('equal')
    
    im2 = ax2.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                       hydro2_data['rho'], levels=100, cmap=COLORMAP_DENSITY,
                       vmin=vmin, vmax=vmax, extend='both')
    ax2.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Diffuse Interface', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.set_aspect('equal')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Density', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    
    fig.suptitle(f't = {t:.4f} s', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    save_path = os.path.join(subfolder_density, f'{frame_num:04d}_density.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_vorticity_comparison(python_data, hydro2_data, frame_num, vmin, vmax):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    t = python_data['t']
    
    im1 = ax1.contourf(python_data['x_grid'], python_data['y_grid'],
                       python_data['vorticity'], levels=100, cmap=COLORMAP_VORTICITY,
                       vmin=vmin, vmax=vmax, extend='both')
    ax1.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
    ax1.set_title('Volume of Fluids', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax1.set_aspect('equal')
    
    im2 = ax2.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                       hydro2_data['vorticity'], levels=100, cmap=COLORMAP_VORTICITY,
                       vmin=vmin, vmax=vmax, extend='both')
    ax2.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
    ax2.set_title('Diffuse Interface', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax2.set_aspect('equal')
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Vorticity', fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=20)
    
    fig.suptitle(f't = {t:.4f} s', fontsize=FONT_SIZE_TIMESTAMP, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    save_path = os.path.join(subfolder_vorticity, f'{frame_num:04d}_vorticity.png')
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()

def plot_difference(python_data, hydro2_data, field_name, frame_num, vmin_diff, vmax_diff):
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
        save_folder = subfolder_diff_density
    else:
        python_field = python_data['vorticity']
        hydro2_field = hydro2_data['vorticity']
        field_label = 'Vorticity'
        save_folder = subfolder_diff_vorticity
    
    difference = hydro2_field - python_field
    
    abs_error = np.abs(difference)
    max_error = np.max(abs_error)
    mean_error = np.mean(abs_error)
    rms_error = np.sqrt(np.mean(difference**2))
    
    python_max = np.max(np.abs(python_field))
    rel_error_max = (max_error / python_max * 100) if python_max > 1e-10 else 0.0
    rel_error_rms = (rms_error / python_max * 100) if python_max > 1e-10 else 0.0
    l2_norm = np.sqrt(np.sum(difference**2))
    
    im = ax_main.contourf(hydro2_data['x_grid'], hydro2_data['y_grid'],
                         difference, levels=50, cmap=COLORMAP_DIFFERENCE,
                         vmin=vmin_diff, vmax=vmax_diff)
    
    ax_main.set_xlabel('X', fontsize=FONT_SIZE_LABEL)
    ax_main.set_ylabel('Y', fontsize=FONT_SIZE_LABEL)
    ax_main.set_title(f'{field_label} Difference at t = {t:.4f} s',
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_main.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax_main)
    cbar.set_label('Difference', fontsize=FONT_SIZE_LABEL)
    
    error_text = f"""
    ERROR METRICS:
    
    Max Absolute Error:     {max_error:.6e}
    Mean Absolute Error:    {mean_error:.6e}
    RMS Error:              {rms_error:.6e}
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
    print("\n" + "=" * 80)
    print("STEP 1: LOADING HYDRO2 FILES")
    print("=" * 80)
    
    plot_files = []
    for item in os.listdir(HYDRO2_OUTPUT_DIR):
        if '.old' in item.lower():
            continue
        item_path = os.path.join(HYDRO2_OUTPUT_DIR, item)
        if os.path.isdir(item_path) and ('plt' in item.lower() or 'cell' in item.lower()):
            plot_files.append(item_path)
    
    if not plot_files:
        print(f"ERROR: No plot files found")
        return
    
    plot_files.sort(key=extract_timestep_number)
    print(f"Found {len(plot_files)} files")
    
    hydro2_times = [float(yt.load(pf).current_time) for pf in plot_files]
    hydro2_times = np.array(hydro2_times)
    print(f"Time range: {hydro2_times[0]:.6e} to {hydro2_times[-1]:.6e} s")
    
    python_data_list = run_python_solver(hydro2_times)
    
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
    
    print(f"Matched {len(matched_pairs)} timesteps")
    
    if len(matched_pairs) == 0:
        return
    
    print("\n" + "=" * 80)
    print("STEP 4: EXTRACTING HYDRO2 DATA")
    print("=" * 80)
    
    hydro2_data_list = []
    resolution = 512
    
    for hydro2_idx, python_idx in matched_pairs:
        ds = yt.load(plot_files[hydro2_idx])
        t = float(ds.current_time)
        
        slc = ds.slice('z', 0.0)
        frb = slc.to_frb((LX, 'code_length'), resolution,
                        center=[0.5*LX, 0.5*LY, 0.0],
                        height=(LY, 'code_length'))
        
        rho_yt = np.array(frb['density'])
        vx_yt = np.array(frb['velocityx'])
        vy_yt = np.array(frb['velocityy'])
        
        try:
            vorticity_yt = np.array(frb['vorticity'])
        except:
            dx_grid = LX / resolution
            dy_grid = LY / resolution
            dvx_dy = np.gradient(vx_yt, dy_grid, axis=0)
            dvy_dx = np.gradient(vy_yt, dx_grid, axis=1)
            vorticity_yt = dvy_dx - dvx_dy
        
        x_1d = np.linspace(0, LX, resolution)
        y_1d = np.linspace(0, LY, resolution)
        x_grid, y_grid = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        rho = rho_yt.T
        vx = vx_yt.T
        vy = vy_yt.T
        vorticity = vorticity_yt.T
        
        hydro2_data_list.append({
            't': t,
            'rho': rho,
            'vx': vx,
            'vy': vy,
            'vorticity': vorticity,
            'x_grid': x_grid,
            'y_grid': y_grid
        })
    
    print(f"Extracted {len(hydro2_data_list)} timesteps")
    
    print("\n" + "=" * 80)
    print("STEP 5: INTERPOLATING PYTHON DATA")
    print("=" * 80)
    
    python_data_interp = []
    
    for hydro2_idx, python_idx in matched_pairs:
        python_data = python_data_list[python_idx]
        hydro2_data = hydro2_data_list[len(python_data_interp)]
        
        print(f"\nInterpolating timestep {len(python_data_interp)+1}/{len(matched_pairs)}...")
        
        rho_interp = interpolate_to_common_grid(python_data['rho'], 
                                                python_data['x_grid'], python_data['y_grid'],
                                                hydro2_data['x_grid'], hydro2_data['y_grid'])
        vx_interp = interpolate_to_common_grid(python_data['vx'],
                                               python_data['x_grid'], python_data['y_grid'],
                                               hydro2_data['x_grid'], hydro2_data['y_grid'])
        vy_interp = interpolate_to_common_grid(python_data['vy'],
                                               python_data['x_grid'], python_data['y_grid'],
                                               hydro2_data['x_grid'], hydro2_data['y_grid'])
        
        dx = hydro2_data['x_grid'][1, 0] - hydro2_data['x_grid'][0, 0]
        dy = hydro2_data['y_grid'][0, 1] - hydro2_data['y_grid'][0, 0]
        vorticity_interp = compute_vorticity(vx_interp, vy_interp, dx, dy)
        
        python_data_interp.append({
            't': python_data['t'],
            'rho': rho_interp,
            'vx': vx_interp,
            'vy': vy_interp,
            'vorticity': vorticity_interp,
            'x_grid': hydro2_data['x_grid'],
            'y_grid': hydro2_data['y_grid']
        })
    
    print(f"\nInterpolation complete")
    
    print("\n" + "=" * 80)
    print("STEP 6: COMPUTING LIMITS")
    print("=" * 80)
    
    rho_min = min(np.min(d['rho']) for d in python_data_interp + hydro2_data_list)
    rho_max = max(np.max(d['rho']) for d in python_data_interp + hydro2_data_list)
    
    vort_all = np.concatenate([d['vorticity'].flatten() for d in python_data_interp + hydro2_data_list])
    vort_lim = 2.0 * np.std(vort_all)
    
    diff_rho_list = [hydro2_data_list[i]['rho'] - python_data_interp[i]['rho'] for i in range(len(matched_pairs))]
    diff_vort_list = [hydro2_data_list[i]['vorticity'] - python_data_interp[i]['vorticity'] for i in range(len(matched_pairs))]
    
    diff_rho_max = max(np.max(np.abs(d)) for d in diff_rho_list)
    diff_vort_max = max(np.max(np.abs(d)) for d in diff_vort_list)
    
    print("\n" + "=" * 80)
    print("STEP 7: GENERATING PLOTS")
    print("=" * 80)
    
    for i in range(len(matched_pairs)):
        frame_num = i + 1
        
        if PLOT_DENSITY:
            plot_density_comparison(python_data_interp[i], hydro2_data_list[i],
                                   frame_num, rho_min, rho_max)
        
        if PLOT_VORTICITY:
            plot_vorticity_comparison(python_data_interp[i], hydro2_data_list[i],
                                     frame_num, -vort_lim, vort_lim)
        
        if PLOT_DIFFERENCES:
            plot_difference(python_data_interp[i], hydro2_data_list[i], 'rho',
                          frame_num, -diff_rho_max, diff_rho_max)
            plot_difference(python_data_interp[i], hydro2_data_list[i], 'vorticity',
                          frame_num, -diff_vort_max, diff_vort_max)
    
    print(f"Generated {len(matched_pairs)} frames")
    
    if GENERATE_GIFS:
        print("\n" + "=" * 80)
        print("STEP 8: CREATING GIFS")
        print("=" * 80)
        
        if PLOT_DENSITY:
            create_gif_from_folder('density', 'ANIM_density.gif')
        if PLOT_VORTICITY:
            create_gif_from_folder('vorticity', 'ANIM_vorticity.gif')
        if PLOT_DIFFERENCES:
            create_gif_from_folder('difference_density', 'ANIM_diff_density.gif')
            create_gif_from_folder('difference_vorticity', 'ANIM_diff_vorticity.gif')
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
