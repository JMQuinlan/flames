"""
===============================================================================
TWO-PHASE COUETTE FLOW RIEMANN SOLVER COMPARISON SCRIPT
===============================================================================

PURPOSE:
    Compare numerical solutions from multiple Riemann solvers against the
    analytical solution for two-phase Couette flow (e.g., air on top of water).
    Generates comprehensive comparison plots and error analysis to evaluate
    solver performance with interface between fluids of different viscosities.

REQUIRED INPUTS:
    Physical parameters for both phases and file paths in configuration section

DIRECTORY NAMING CONVENTION:
    The script expects output directories in the following format:
    
    CouetteFlow_2Phase_{RiemannSolver}
    
    Examples:
        CouetteFlow_2Phase_HLLC
        CouetteFlow_2Phase_Roe
        CouetteFlow_2Phase_HLLE
        CouetteFlow_2Phase_HLLC-Weno5

RIEMANN SOLVER NAMING:
    - Dashes in Riemann solver names are converted to spaces in plots
    - Example: "HLLC-Weno5" displays as "HLLC Weno5"

CONFIGURATION VARIABLES (lines 70-115):
    
    Physical Parameters - Phase 0 (Bottom Fluid):
        gamma_0                 - Ratio of specific heats (default: 1.4)
        pressure_0              - Reference pressure in Pa (default: 1.0e5)
        p0_tammann_0            - Tammann EOS pressure modification in Pa (default: 0.0)
        density_0               - Fluid density in kg/m³ (default: 1000.0 for water)
        viscosity_0             - Dynamic viscosity in Pa·s (default: 1.0e-3 for water)
    
    Physical Parameters - Phase 1 (Top Fluid):
        gamma_1                 - Ratio of specific heats (default: 1.4)
        pressure_1              - Reference pressure in Pa (default: 1.0e5)
        p0_tammann_1            - Tammann EOS pressure modification in Pa (default: 0.0)
        density_1               - Fluid density in kg/m³ (default: 1.225 for air)
        viscosity_1             - Dynamic viscosity in Pa·s (default: 1.81e-5 for air)
    
    Domain Parameters:
        tube_height             - Total height of the Couette flow domain in m (default: 0.01)
        interface_height        - Height of the interface between fluids in m (default: 0.005)
        top_plate_velocity      - Velocity of the moving top plate in m/s (default: 1.0)
        bottom_plate_velocity   - Velocity of the bottom plate in m/s (default: 0.0)
    
    Font Sizes:
        TITLE_FONTSIZE          - Plot title font size (default: 16)
        AXIS_LABEL_FONTSIZE     - Axis label font size (default: 14)
        LEGEND_FONTSIZE         - Legend font size (default: 12)
        TICK_FONTSIZE           - Tick label font size (default: 11)
    
    Line Weights:
        EXACT_SOLUTION_LINEWIDTH        - Exact solution line (default: 2.5)
        SOLVER_LINEWIDTH                - Individual solver lines (default: 2.0)
        INTERFACE_LINEWIDTH             - Interface marker line (default: 1.5)
        AGGREGATED_EXACT_LINEWIDTH      - Exact solution in aggregated plot (default: 3.0)
        AGGREGATED_SOLVER_LINEWIDTH     - Solver lines in aggregated plot (default: 1.8)
        ERROR_LINEWIDTH                 - Error plot lines (default: 2.0)
    
    File Paths:
        base_output_dir         - Path to directory containing CouetteFlow_2Phase_* folders

ANALYTICAL SOLUTION:
    Two-phase Couette flow with stationary bottom plate and moving top plate:
    
    Phase 0 (0 <= y <= h):  u_0(y) = A_0 * y + B_0
    Phase 1 (h <= y <= H):  u_1(y) = A_1 * y + B_1
    
    where:
        A_0 = mu_1 * (U_top - U_bottom) / (mu_1 * h + mu_0 * (H - h))
        A_1 = mu_0 * (U_top - U_bottom) / (mu_1 * h + mu_0 * (H - h))
        B_0 = U_bottom
        B_1 = U_top - A_1 * H
        h = interface height
        H = total channel height
        mu_0, mu_1 = dynamic viscosities of phases 0 and 1
    
    Boundary conditions:
        - u_0(0) = U_bottom (bottom plate)
        - u_1(H) = U_top (top plate)
        - u_0(h) = u_1(h) (velocity continuity at interface)
        - mu_0 * du_0/dy = mu_1 * du_1/dy (shear stress continuity at interface)

OUTPUTS:
    The script generates the following plots in ./Images/ directory:
    
    1. Individual Solver Comparisons (per Riemann solver):
       - couette_2phase_{solver}_comparison.png/eps
       - Shows analytical solution vs. numerical solution for that solver
       - Includes interface marker and phase annotations
       - couette_2phase_{solver}_error_semilog.png/eps
       - Semi-log error plot for that solver with interface marker
    
    2. Aggregated Comparison:
       - couette_2phase_aggregated_comparison.png/eps
       - All solvers compared against analytical solution
       - Includes interface marker and phase annotations
       - couette_2phase_aggregated_error_semilog.png/eps
       - Semi-log error plot for all solvers with interface marker

COLOR SCHEME:
    - Each Riemann solver gets a unique color from the palette
    - Colors: blue, red, green, purple, orange, brown, pink, gray, olive, cyan
    - Analytical solution is always black
    - Interface marker is green dotted line

FILTERING:
    - Script automatically detects maximum stop time across all simulations
    - Excludes any simulation that stopped before max time (tolerance: 1e-10)
    - Only compares simulations that completed to the same final time

ERROR METRICS:
    Calculates and reports for each solver:
        - Overall L2 norm error
        - Overall L-infinity norm error
        - Overall relative L2 error (percentage)
        - Phase 0 (bottom) L2 and L-infinity errors
        - Phase 1 (top) L2 and L-infinity errors

USAGE:
    1. Set physical parameters for both phases and file paths in configuration
    2. Ensure output directories follow naming convention
    3. Run script: python CouetteFlow2PhaseRiemannComparison.py
    4. Check ./Images/ for generated plots
    5. Review console output for error metrics

DEPENDENCIES:
    - yt (for AMReX data loading)
    - numpy
    - matplotlib
    - re, os, collections

===============================================================================
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical parameters - Phase 0 (bottom fluid, e.g., water)
gamma_0 = 1.4                    # Ratio of specific heats
pressure_0 = 1.0e5               # Reference pressure (Pa)
p0_tammann_0 = 0.0               # Tammann EOS pressure modification (Pa)
density_0 = 1000.0               # Fluid density (kg/m³)
viscosity_0 = 1.0e-3             # Dynamic viscosity (Pa·s) - water

# Physical parameters - Phase 1 (top fluid, e.g., air)
gamma_1 = 1.4                    # Ratio of specific heats
pressure_1 = 1.0e5               # Reference pressure (Pa)
p0_tammann_1 = 0.0               # Tammann EOS pressure modification (Pa)
density_1 = 1.225                # Fluid density (kg/m³)
viscosity_1 = 1.81e-5            # Dynamic viscosity (Pa·s) - air

# Domain parameters
tube_height = 0.01               # Total height of the tube (m)
interface_height = 0.005         # Height of the interface (m)
top_plate_velocity = 1.0         # Velocity of the top plate (m/s)
bottom_plate_velocity = 0.0      # Velocity of the bottom plate (m/s)

# Font sizes for plots (easily adjustable)
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 11

# Line weights for plots (easily adjustable)
EXACT_SOLUTION_LINEWIDTH = 2.5
SOLVER_LINEWIDTH = 2.0
INTERFACE_LINEWIDTH = 1.5
AGGREGATED_EXACT_LINEWIDTH = 3.0
AGGREGATED_SOLVER_LINEWIDTH = 1.8
ERROR_LINEWIDTH = 2.0

# File paths
base_output_dir = r'..\..\..\bin\tests\CouetteFlow'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_label(text):
    """Convert naming convention to display format"""
    # Replace dashes with spaces
    text = text.replace('-', ' ')
    return text

def parse_output_directory(dir_name):
    """
    Parse output directory name to extract Riemann solver
    Expected format: CouetteFlow_2Phase_{RiemannSolver}
    """
    prefix = 'CouetteFlow_2Phase_'
    if not dir_name.startswith(prefix):
        return None
    
    riemann_solver = dir_name[len(prefix):]
    
    return {
        'riemann_solver': riemann_solver
    }

def extract_timestep_number(filename):
    """
    Extract the timestep number from a plot file name.
    Handles formats like: 00000cell, 00100cell, plt00000, etc.
    """
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return 0

def load_amrex_couette_data(output_dir, y_min, y_max):
    """Load Couette flow data from AMReX output directory using yt"""
    # Find plot files
    plot_files = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith('cell'):
            plot_files.append(item_path)
    
    if not plot_files:
        return None
    
    # Sort and use the last one
    plot_files.sort(key=extract_timestep_number)
    last_plot = plot_files[-1]
    
    # Load the dataset
    ds = yt.load(last_plot)
    
    # Create a ray along y-direction at x=0, z=0
    ray_start = ds.arr([0.0, y_min, 0.0], 'code_length')
    ray_end = ds.arr([0.0, y_max, 0.0], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by y coordinate
    sort_indices = np.argsort(ray['y'])
    
    data = {
        'y': np.array(ray['y'][sort_indices]),
        'velocity': np.array(ray['velocityx'][sort_indices]),
        'time': float(ds.current_time)
    }
    
    return data

def analytical_twophase_couette_velocity(y, U_top, U_bottom, H, h_interface, mu_0, mu_1):
    """
    Analytical solution for two-phase Couette flow with moving top plate.
    
    The solution is derived from continuity of velocity and shear stress at the interface:
    - Velocity is continuous: u_0(h) = u_1(h)
    - Shear stress is continuous: mu_0 * du_0/dy = mu_1 * du_1/dy at y = h
    
    Parameters:
    -----------
    y : array-like
        Vertical positions
    U_top : float
        Velocity of the top plate
    U_bottom : float
        Velocity of the bottom plate
    H : float
        Total height of the channel
    h_interface : float
        Height of the interface between fluids
    mu_0 : float
        Dynamic viscosity of bottom fluid (Pa·s)
    mu_1 : float
        Dynamic viscosity of top fluid (Pa·s)
    
    Returns:
    --------
    u : array-like
        Velocity at each y position
    """
    # Solve for interface velocity using continuity conditions
    # For phase 0 (0 <= y <= h): u_0 = A_0 * y + B_0
    # For phase 1 (h <= y <= H): u_1 = A_1 * y + B_1
    
    # From shear stress continuity: A_0 = (mu_1 / mu_0) * A_1
    # Solving the system:
    denominator = mu_1 * h_interface + mu_0 * (H - h_interface)
    A_0 = mu_1 * (U_top - U_bottom) / denominator
    A_1 = mu_0 * (U_top - U_bottom) / denominator
    
    B_0 = U_bottom
    B_1 = U_top - A_1 * H
    
    # Calculate velocity profile
    u = np.zeros_like(y)
    
    # Phase 0 (bottom fluid)
    mask_0 = y <= h_interface
    u[mask_0] = A_0 * y[mask_0] + B_0
    
    # Phase 1 (top fluid)
    mask_1 = y > h_interface
    u[mask_1] = A_1 * y[mask_1] + B_1
    
    return u

def calculate_errors(numerical_velocity, numerical_y, analytical_velocity, analytical_y, interface_height):
    """Calculate L2 and L-infinity errors for overall and per-phase"""
    # Interpolate analytical solution to numerical grid
    velocity_analytical_interp = np.interp(numerical_y, analytical_y, analytical_velocity)
    
    # Calculate overall errors
    velocity_error = numerical_velocity - velocity_analytical_interp
    
    l2_error = np.linalg.norm(velocity_error) / np.sqrt(len(velocity_error))
    linf_error = np.max(np.abs(velocity_error))
    relative_l2_error = l2_error / (np.linalg.norm(velocity_analytical_interp) / np.sqrt(len(velocity_analytical_interp)))
    
    # Calculate errors per phase
    mask_phase0 = numerical_y <= interface_height
    mask_phase1 = numerical_y > interface_height
    
    if np.any(mask_phase0):
        l2_error_phase0 = np.linalg.norm(velocity_error[mask_phase0]) / np.sqrt(np.sum(mask_phase0))
        linf_error_phase0 = np.max(np.abs(velocity_error[mask_phase0]))
    else:
        l2_error_phase0 = 0.0
        linf_error_phase0 = 0.0
    
    if np.any(mask_phase1):
        l2_error_phase1 = np.linalg.norm(velocity_error[mask_phase1]) / np.sqrt(np.sum(mask_phase1))
        linf_error_phase1 = np.max(np.abs(velocity_error[mask_phase1]))
    else:
        l2_error_phase1 = 0.0
        linf_error_phase1 = 0.0
    
    errors = {
        'l2': l2_error,
        'linf': linf_error,
        'relative_l2': relative_l2_error,
        'l2_phase0': l2_error_phase0,
        'linf_phase0': linf_error_phase0,
        'l2_phase1': l2_error_phase1,
        'linf_phase1': linf_error_phase1
    }
    
    return errors

def generate_color_palette(riemann_solvers):
    """Generate color palette for Riemann solvers"""
    # Use distinct base colors for each solver
    base_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    color_map = {}
    for idx, solver in enumerate(riemann_solvers):
        color_map[solver] = base_colors[idx % len(base_colors)]
    
    return color_map

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("TWO-PHASE COUETTE FLOW RIEMANN SOLVER COMPARISON ANALYSIS")
print("=" * 80)
print(f"\nPhase 0 (Bottom Fluid) Parameters:")
print(f"  Gamma:              {gamma_0}")
print(f"  Pressure:           {pressure_0:.2e} Pa")
print(f"  Tammann p0:         {p0_tammann_0:.2e} Pa")
print(f"  Density:            {density_0:.2e} kg/m³")
print(f"  Viscosity:          {viscosity_0:.2e} Pa·s")
print(f"\nPhase 1 (Top Fluid) Parameters:")
print(f"  Gamma:              {gamma_1}")
print(f"  Pressure:           {pressure_1:.2e} Pa")
print(f"  Tammann p0:         {p0_tammann_1:.2e} Pa")
print(f"  Density:            {density_1:.2e} kg/m³")
print(f"  Viscosity:          {viscosity_1:.2e} Pa·s")
print(f"\nDomain Parameters:")
print(f"  Tube Height:        {tube_height:.4f} m")
print(f"  Interface Height:   {interface_height:.4f} m")
print(f"  Top Plate Velocity: {top_plate_velocity:.4f} m/s")
print(f"  Bottom Plate Vel:   {bottom_plate_velocity:.4f} m/s")
print(f"  Viscosity Ratio:    {viscosity_0/viscosity_1:.2f}")
print(f"\nBase Directory: {base_output_dir}")

# Compute analytical solution
y_min = 0.0
y_max = tube_height
y_analytical = np.linspace(y_min, y_max, 1000)
velocity_analytical = analytical_twophase_couette_velocity(
    y_analytical, top_plate_velocity, bottom_plate_velocity,
    tube_height, interface_height, viscosity_0, viscosity_1
)

# Calculate interface velocity for reference
u_interface = analytical_twophase_couette_velocity(
    np.array([interface_height]), top_plate_velocity, bottom_plate_velocity,
    tube_height, interface_height, viscosity_0, viscosity_1
)[0]

print(f"\nAnalytical solution computed on {len(y_analytical)} points")
print(f"  Interface velocity: {u_interface:.6e} m/s")

# Scan for all output directories matching the pattern
print(f"\nScanning for output directories...")
all_data = {}
all_stop_times = []

for item in os.listdir(base_output_dir):
    item_path = os.path.join(base_output_dir, item)
    
    if os.path.isdir(item_path) and item.startswith('CouetteFlow_2Phase_'):
        parsed = parse_output_directory(item)
        
        if parsed:
            riemann_solver = parsed['riemann_solver']
            print(f"\nFound: {item}")
            print(f"  Riemann Solver: {riemann_solver}")
            
            # Load data
            data = load_amrex_couette_data(item_path, y_min, y_max)
            
            if data:
                stop_time = data['time']
                all_stop_times.append(stop_time)
                all_data[riemann_solver] = data
                print(f"  Loaded data: {len(data['y'])} points, t={stop_time:.6e}s")

if not all_data:
    print("\nERROR: No matching output directories found!")
    print(f"Expected pattern: CouetteFlow_2Phase_{{RiemannSolver}}")
    exit(1)

# Determine maximum stop time
if all_stop_times:
    max_stop_time = max(all_stop_times)
    print(f"\n{'=' * 80}")
    print(f"Maximum stop time found: {max_stop_time:.6e}s")
    print(f"Filtering out simulations that stopped early...")
    
    # Filter out data that didn't reach max stop time (with small tolerance)
    tolerance = 1e-10
    filtered_data = {}
    
    for solver, solver_data in all_data.items():
        if abs(solver_data['time'] - max_stop_time) < tolerance:
            filtered_data[solver] = solver_data
        else:
            print(f"  Excluding {solver} (stopped at t={solver_data['time']:.6e}s)")
    
    # Replace all_data with filtered data
    all_data = filtered_data
    
    if not all_data:
        print("\nERROR: No simulations reached the maximum stop time!")
        exit(1)

print(f"\n{'=' * 80}")
print(f"Found {len(all_data)} Riemann solver(s) with complete simulations")
for solver in all_data.keys():
    print(f"  {solver}")

# Create output directory for images
os.makedirs('./Images', exist_ok=True)

# Get simulation time
sim_time = max_stop_time if all_stop_times else None

# Generate color palette
riemann_solvers = sorted(all_data.keys())
color_map = generate_color_palette(riemann_solvers)

# ============================================================================
# PLOT 1: Individual Riemann Solver Comparisons
# ============================================================================

print(f"\n{'=' * 80}")
print("GENERATING INDIVIDUAL RIEMANN SOLVER PLOTS")
print("=" * 80)

for solver in riemann_solvers:
    solver_data = all_data[solver]
    
    print(f"\nPlotting {solver}...")
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    title = f'Two-Phase Couette Flow - {format_label(solver)}'
    if sim_time is not None:
        title += f' at t={sim_time:.6e}s'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # Plot analytical solution
    ax.plot(velocity_analytical, y_analytical, 'k-', 
            linewidth=EXACT_SOLUTION_LINEWIDTH, label='Analytical Solution', zorder=10)
    
    # Plot numerical solution
    ax.plot(solver_data['velocity'], solver_data['y'], '--', 
            color=color_map[solver], linewidth=SOLVER_LINEWIDTH, 
            label=format_label(solver), zorder=5, alpha=0.8)
    
    # Mark the interface
    ax.axhline(y=interface_height, color='green', linestyle=':', 
               linewidth=INTERFACE_LINEWIDTH, label='Interface', alpha=0.7)
    
    ax.set_xlabel('Velocity (m/s)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Height (m)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    
    # Add text annotations for phases
    y_phase0_text = interface_height / 2
    y_phase1_text = interface_height + (tube_height - interface_height) / 2
    ax.text(0.02, y_phase0_text, f'Phase 0\nmu={viscosity_0:.2e}', 
            transform=ax.get_yaxis_transform(), fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.02, y_phase1_text, f'Phase 1\nmu={viscosity_1:.2e}', 
            transform=ax.get_yaxis_transform(), fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    output_filename = f'./Images/couette_2phase_{solver}_comparison'
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    print(f"  Saved: {output_filename}")
    plt.close()
    
    # ========================================================================
    # ERROR PLOT: Individual solver
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    title = f'Two-Phase Couette Flow - {format_label(solver)} - Error Analysis'
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    # Calculate error
    velocity_analytical_interp = np.interp(solver_data['y'], y_analytical, velocity_analytical)
    error = np.abs(solver_data['velocity'] - velocity_analytical_interp)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-16
    error_safe = error + epsilon
    
    ax.semilogy(solver_data['y'], error_safe, '-', 
                color=color_map[solver], linewidth=ERROR_LINEWIDTH, 
                label=format_label(solver), alpha=0.8)
    
    # Mark the interface
    ax.axvline(x=interface_height, color='green', linestyle=':', 
               linewidth=INTERFACE_LINEWIDTH, label='Interface', alpha=0.7)
    
    ax.set_xlabel('Height (m)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('Absolute Error (m/s)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/couette_2phase_{solver}_error_semilog'
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    print(f"  Saved: {output_filename}")
    plt.close()

# ============================================================================
# PLOT 2: Aggregated Comparison (All Solvers)
# ============================================================================

print(f"\n{'=' * 80}")
print("GENERATING AGGREGATED COMPARISON PLOT")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 10))

title = 'Two-Phase Couette Flow - Riemann Solver Comparison'
if sim_time is not None:
    title += f' at t={sim_time:.6e}s'
ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

# Plot analytical solution
ax.plot(velocity_analytical, y_analytical, 'k-', 
        linewidth=AGGREGATED_EXACT_LINEWIDTH, label='Analytical Solution', zorder=10)

# Plot all solvers
for solver in riemann_solvers:
    solver_data = all_data[solver]
    ax.plot(solver_data['velocity'], solver_data['y'], '--', 
            color=color_map[solver], linewidth=AGGREGATED_SOLVER_LINEWIDTH, 
            label=format_label(solver), alpha=0.7)

# Mark the interface
ax.axhline(y=interface_height, color='green', linestyle=':', 
           linewidth=INTERFACE_LINEWIDTH, label='Interface', alpha=0.7)

ax.set_xlabel('Velocity (m/s)', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel('Height (m)', fontsize=AXIS_LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=TICK_FONTSIZE)

# Add text annotations for phases
y_phase0_text = interface_height / 2
y_phase1_text = interface_height + (tube_height - interface_height) / 2
ax.text(0.02, y_phase0_text, f'Phase 0\nmu={viscosity_0:.2e}', 
        transform=ax.get_yaxis_transform(), fontsize=10, 
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.text(0.02, y_phase1_text, f'Phase 1\nmu={viscosity_1:.2e}', 
        transform=ax.get_yaxis_transform(), fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
output_filename = './Images/couette_2phase_aggregated_comparison'
plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_filename}")
plt.close()

# ============================================================================
# ERROR PLOT: Aggregated
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

title = 'Two-Phase Couette Flow - Error Analysis (All Solvers)'
ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

# Plot errors for all solvers
for solver in riemann_solvers:
    solver_data = all_data[solver]
    
    # Calculate error
    velocity_analytical_interp = np.interp(solver_data['y'], y_analytical, velocity_analytical)
    error = np.abs(solver_data['velocity'] - velocity_analytical_interp)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-16
    error_safe = error + epsilon
    
    ax.semilogy(solver_data['y'], error_safe, '-', 
                color=color_map[solver], linewidth=ERROR_LINEWIDTH, 
                label=format_label(solver), alpha=0.7)

# Mark the interface
ax.axvline(x=interface_height, color='green', linestyle=':', 
           linewidth=INTERFACE_LINEWIDTH, label='Interface', alpha=0.7)

ax.set_xlabel('Height (m)', fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel('Absolute Error (m/s)', fontsize=AXIS_LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
ax.grid(True, alpha=0.3, which='both')
ax.tick_params(labelsize=TICK_FONTSIZE)

plt.tight_layout()
output_filename = './Images/couette_2phase_aggregated_error_semilog'
plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_filename}")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print(f"\n{'=' * 80}")
print("ERROR METRICS SUMMARY")
print("=" * 80)

for solver in riemann_solvers:
    print(f"\n{solver}:")
    print("-" * 60)
    
    solver_data = all_data[solver]
    errors = calculate_errors(solver_data['velocity'], solver_data['y'], 
                              velocity_analytical, y_analytical, interface_height)
    
    print(f"  Overall Errors:")
    print(f"    L2 error:          {errors['l2']:.6e}")
    print(f"    L-infinity error:  {errors['linf']:.6e}")
    print(f"    Relative L2 error: {errors['relative_l2']:.6e} ({errors['relative_l2']*100:.4f}%)")
    print(f"  Phase 0 (Bottom) Errors:")
    print(f"    L2 error:          {errors['l2_phase0']:.6e}")
    print(f"    L-infinity error:  {errors['linf_phase0']:.6e}")
    print(f"  Phase 1 (Top) Errors:")
    print(f"    L2 error:          {errors['l2_phase1']:.6e}")
    print(f"    L-infinity error:  {errors['linf_phase1']:.6e}")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE!")
print(f"All plots saved to: ./Images/")
print("=" * 80)
