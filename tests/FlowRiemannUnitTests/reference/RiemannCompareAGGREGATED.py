"""
===============================================================================
RIEMANN SOLVER COMPARISON ANALYSIS SCRIPT
===============================================================================

PURPOSE:
    Compare numerical solutions from multiple Riemann solvers and interface
    thicknesses against exact solutions for shock tube problems. Generates
    comprehensive comparison plots and error analysis.

REQUIRED INPUTS:
    1. Test case name (case_name variable)
    2. HDF5 file with exact solution
    3. AMReX output directories following naming convention

DIRECTORY NAMING CONVENTION:
    The script expects output directories in the following formats:
    
    - Sharp Interface:
      output_{TestCase}_{RiemannSolver}_Sharp_Interface
      Example: output_Toro1a_HLLC_Sharp_Interface
    
    - Diffuse Interface (decimal notation):
      output_{TestCase}_{RiemannSolver}_epsilon_{value}
      Example: output_Toro1a_HLLC_epsilon_0.01
    
    - Diffuse Interface (scientific notation):
      output_{TestCase}_{RiemannSolver}_epsilon_{value}e{exponent}
      Examples: output_Toro1a_HLLC_epsilon_1e-4
                output_Toro1a_HLLC_epsilon_1.0e-4
                output_Toro1a_HLLC_epsilon_5e-05

RIEMANN SOLVER NAMING:
    - Dashes in Riemann solver names are converted to spaces in plots
    - Example: "HLLC-Weno5" displays as "HLLC Weno5"
    - Dashes in epsilon values are NOT converted (scientific notation preserved)

CONFIGURATION VARIABLES (lines 32-48):
    
    Display Options:
        PLOT_TO_SCREEN          - Display plots interactively (default: False)
                                  True: Shows plots on screen (allows zooming)
                                  False: Only saves plots without displaying
    
    Font Sizes:
        TITLE_FONTSIZE          - Plot title font size (default: 16)
        AXIS_LABEL_FONTSIZE     - Axis label font size (default: 14)
        LEGEND_FONTSIZE         - Legend font size (default: 10)
        TICK_FONTSIZE           - Tick label font size (default: 12)
    
    Line Weights:
        EXACT_SOLUTION_LINEWIDTH        - Exact solution in individual plots (default: 2.5)
        SHARP_INTERFACE_LINEWIDTH       - Sharp interface in individual plots (default: 2.0)
        EPSILON_LINEWIDTH               - Epsilon variants in individual plots (default: 1.5)
        AGGREGATED_EXACT_LINEWIDTH      - Exact solution in aggregated plot (default: 3.0)
        AGGREGATED_SHARP_LINEWIDTH      - Sharp interface in aggregated plot (default: 1.8)
        AGGREGATED_EPSILON_LINEWIDTH    - Epsilon variants in aggregated plot (default: 1.3)
        ERROR_SHARP_LINEWIDTH           - Sharp interface in error plots (default: 2.0)
        ERROR_EPSILON_LINEWIDTH         - Epsilon variants in error plots (default: 1.5)
    
    Test Case:
        case_name               - Name of test case (e.g., 'Toro1a', 'Toro2')
        Tammann                 - '.' for regular EOS, 'TammannEOS' for Tammann
        hdf5_file               - Path to exact solution HDF5 file
        base_output_dir         - Path to directory containing output folders

HDF5 FILE STRUCTURE:
    The exact solution HDF5 file must contain groups with datasets:
        - xvec              : x-coordinates
        - velocity          : velocity field
        - pressure          : pressure field
        - density           : density field
        - internal_energy   : internal energy field

OUTPUTS:
    The script generates the following plots in ./Images/ directory:
    
    1. Individual Solver Comparisons (per Riemann solver):
       - {case}__{solver}_comparison.png/eps
       - Shows exact solution, sharp interface, and all epsilon variants
       - Legend appears only on top subplot
       - {case}_{solver}_error.png/eps
       - Semi-log error plot for that solver
       - Legend appears only on top subplot
    
    2. Sharp Interface Comparison:
       - {case}_sharp_comparison.png/eps
       - Compares all sharp interfaces against exact solution
       - Legend appears only on top subplot
       - {case}_sharp_error.png/eps
       - Semi-log error plot for all sharp interfaces
       - Legend appears only on top subplot
    
    3. Aggregated Comparison:
       - {case}_aggregated_comparison.png/eps
       - All solvers and interface thicknesses on one plot
       - Legend appears only on top subplot
       - {case}_aggregated_error.png/eps
       - Semi-log error plot for all data
       - Legend appears only on top subplot

COLOR SCHEME:
    - Each Riemann solver gets a unique base color
    - Sharp interface uses the base color
    - Epsilon variants use gradient from dark to light:
      * Smallest epsilon = darkest shade (closest to sharp interface)
      * Largest epsilon = lightest shade (most diffuse)

FILTERING:
    - Script automatically detects maximum stop time across all simulations
    - Excludes any simulation that stopped before max time (tolerance: 1e-10)
    - Only compares simulations that completed to the same final time

ERROR METRICS:
    Calculates and reports for each solver/interface combination:
        - L2 norm error (density, velocity, pressure)
        - L-infinity norm error (density, velocity, pressure)

USAGE:
    1. Set case_name, Tammann, and file paths in configuration section
    2. Set PLOT_TO_SCREEN = True if you want interactive plots
    3. Ensure output directories follow naming convention
    4. Run script: python RiemannCompareAGGREGATED.py
    5. Check ./Images/ for generated plots
    6. Review console output for error metrics

DEPENDENCIES:
    - yt (for AMReX data loading)
    - numpy
    - h5py
    - matplotlib
    - re, os, collections, matplotlib.colors

===============================================================================
"""

import yt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict
import matplotlib.colors as mcolors

# Suppress yt's verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Display options
PLOT_TO_SCREEN = False  # Set to True to display plots interactively (allows zooming)

# Font sizes for plots
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 10
TICK_FONTSIZE = 12

# Line weights for plots
EXACT_SOLUTION_LINEWIDTH = 2.5
SHARP_INTERFACE_LINEWIDTH = 2.0
EPSILON_LINEWIDTH = 1.5
AGGREGATED_EXACT_LINEWIDTH = 3.0
AGGREGATED_SHARP_LINEWIDTH = 1.8
AGGREGATED_EPSILON_LINEWIDTH = 1.3
ERROR_SHARP_LINEWIDTH = 2.0
ERROR_EPSILON_LINEWIDTH = 1.5

# Test case configuration
case_name = 'Toro1a'
Tammann = '.'  # Change to "." for CPG and "TammannEOS" for Tammann

# File paths
hdf5_file = fr'{Tammann}\{case_name}.hdf5'
base_output_dir = fr'..\..\..\bin\tests\FlowRiemannUnitTestsAUTO\{case_name}'

# Case name mapping
case_to_group = {
    # Regular
    'Toro1a': 'Toro_case_1_a',
    'Toro1b': 'Toro_case_1_b',
    'Toro1r': 'Toro_case_1_r',
    'Toro2': 'Toro_case_2',
    'Toro3': 'Toro_case_3',
    'Toro4a': 'Toro_case_4a',
    'Toro4b': 'Toro_case_4b',
    'Toro5': 'Toro_case_5',
    'Toro6': 'Toro_case_6',
    'Toro7': 'Toro_case_7',
    # Tammann
    'Tammann_water_air': 'water_air',
    'Tammann_water_shock': 'water_shock',
    'Tammann_cavitation': 'cavitation',
    'Tammann_bubble_expansion': 'bubble_expansion',
    'Tammann_bubble_collapse': 'bubble_collapse',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_label(text):
    """Convert naming convention to display format with LaTeX"""
    # Replace dashes with spaces (but not in epsilon values)
    # First, protect epsilon values
    epsilon_pattern = r'epsilon[_\s]*([\d.eE+-]+)'
    epsilon_matches = list(re.finditer(epsilon_pattern, text))
    
    # Replace dashes with spaces in non-epsilon parts
    if epsilon_matches:
        parts = []
        last_end = 0
        for match in epsilon_matches:
            # Process text before epsilon
            before_text = text[last_end:match.start()].replace('-', ' ')
            parts.append(before_text)
            # Keep epsilon part as-is for now
            epsilon_val = match.group(1)
            parts.append(f'$\\epsilon = {epsilon_val}$')
            last_end = match.end()
        # Process remaining text
        parts.append(text[last_end:].replace('-', ' '))
        text = ''.join(parts)
    else:
        text = text.replace('-', ' ')
    
    # Handle Sharp Interface
    text = text.replace('Sharp Interface', 'Sharp Interface')
    
    return text

def parse_output_directory(dir_name, case_name):
    """
    Parse output directory name to extract Riemann solver and interface type
    Expected formats:
    - output_{TestCase}_{RiemannSolver}_Sharp_Interface
    - output_{TestCase}_{RiemannSolver}_epsilon_{value}
    - output_{TestCase}_{RiemannSolver}_epsilon_{value}e{exponent}
    
    Supports scientific notation: epsilon_1e-4, epsilon_1.0e-4, epsilon_5e-05, etc.
    """
    # Remove the prefix
    prefix = f'output_{case_name}_'
    if not dir_name.startswith(prefix):
        return None
    
    remainder = dir_name[len(prefix):]
    
    # Check for Sharp_Interface
    if 'Sharp_Interface' in remainder:
        riemann_solver = remainder.split('_Sharp_Interface')[0]
        return {
            'riemann_solver': riemann_solver,
            'interface_type': 'Sharp_Interface',
            'epsilon_value': None,
            'is_sharp': True
        }
    
    # Check for epsilon pattern - supports both decimal and scientific notation
    # Pattern matches: epsilon_0.01, epsilon_1e-4, epsilon_1.0e-4, epsilon_5e-05
    epsilon_pattern = r'_epsilon_([\d.]+(?:[eE][+-]?\d+)?)'
    epsilon_match = re.search(epsilon_pattern, remainder)
    
    if epsilon_match:
        epsilon_str = epsilon_match.group(1)
        try:
            epsilon_value = float(epsilon_str)
        except ValueError:
            print(f"Warning: Could not parse epsilon value '{epsilon_str}' in '{dir_name}'")
            return None
        
        # Extract riemann solver (everything before _epsilon_)
        riemann_solver = remainder.split('_epsilon_')[0]
        interface_type = f'epsilon_{epsilon_str}'
        
        return {
            'riemann_solver': riemann_solver,
            'interface_type': interface_type,
            'epsilon_value': epsilon_value,
            'is_sharp': False
        }
    
    return None

def load_amrex_data(output_dir):
    """Load data from AMReX output directory using yt"""
    # Find plot files
    plot_files = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith('cell'):
            plot_files.append(item_path)
    
    if not plot_files:
        return None
    
    # Sort and use the last one
    plot_files.sort()
    last_plot = plot_files[-1]
    
    # Load the dataset
    ds = yt.load(last_plot)
    
    # Create a ray through the domain at y=0
    ray_start = ds.arr([-1.0, 0.0, 0.0], 'code_length')
    ray_end = ds.arr([1.0, 0.0, 0.0], 'code_length')
    ray = ds.ray(ray_start, ray_end)
    
    # Sort by x coordinate
    sort_indices = np.argsort(ray['x'])
    
    data = {
        'x': np.array(ray['x'][sort_indices]),
        'velocity': np.array(ray['velocityx'][sort_indices]),
        'pressure': np.array(ray['pressure'][sort_indices]),
        'density': np.array(ray['density'][sort_indices]),
        'time': float(ds.current_time)
    }
    
    return data

def load_exact_solution(hdf5_file, group_name):
    """Load exact solution from HDF5 file"""
    with h5py.File(hdf5_file, 'r') as f:
        if group_name not in f.keys():
            print(f"Warning: '{group_name}' not found, using first available")
            group_name = list(f.keys())[0]
        
        group = f[group_name]
        
        data = {
            'x': group['xvec'][:],
            'velocity': group['velocity'][:],
            'pressure': group['pressure'][:],
            'density': group['density'][:],
            'internal_energy': group['internal_energy'][:]
        }
    
    return data

def calculate_errors(numerical_data, exact_data):
    """Calculate L2 and L_infinity errors"""
    # Interpolate exact solution to numerical grid
    velocity_exact_interp = np.interp(numerical_data['x'], exact_data['x'], exact_data['velocity'])
    pressure_exact_interp = np.interp(numerical_data['x'], exact_data['x'], exact_data['pressure'])
    density_exact_interp = np.interp(numerical_data['x'], exact_data['x'], exact_data['density'])
    
    errors = {
        'velocity_l2': np.linalg.norm(numerical_data['velocity'] - velocity_exact_interp),
        'velocity_linf': np.max(np.abs(numerical_data['velocity'] - velocity_exact_interp)),
        'pressure_l2': np.linalg.norm(numerical_data['pressure'] - pressure_exact_interp),
        'pressure_linf': np.max(np.abs(numerical_data['pressure'] - pressure_exact_interp)),
        'density_l2': np.linalg.norm(numerical_data['density'] - density_exact_interp),
        'density_linf': np.max(np.abs(numerical_data['density'] - density_exact_interp))
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

def get_color_shade(base_color, epsilon_value, epsilon_values):
    """Get color shade based on epsilon value (darker to lighter)"""
    if epsilon_value is None:  # Sharp interface
        return base_color
    
    # Sort epsilon values to create gradient
    sorted_epsilons = sorted([e for e in epsilon_values if e is not None])
    
    if len(sorted_epsilons) == 0:
        return base_color
    
    # Find position in sorted list
    idx = sorted_epsilons.index(epsilon_value)
    
    # Create gradient from dark (1.0) to light (0.4)
    # Smallest epsilon (idx=0) gets alpha=1.0 (darkest)
    # Largest epsilon gets alpha=0.4 (lightest)
    alpha = 1.0 - (0.6 * idx / max(len(sorted_epsilons) - 1, 1))
    
    # Convert color name to RGB and adjust brightness
    rgb = mcolors.to_rgb(base_color)
    # Darken the color based on alpha
    rgb_adjusted = tuple(c * alpha + (1 - alpha) for c in rgb)
    
    return rgb_adjusted

def save_and_show_plot(output_filename, plot_to_screen=False):
    """Save plot to file and optionally display on screen"""
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    
    if plot_to_screen:
        plt.show()
    
    plt.close()

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("RIEMANN SOLVER COMPARISON ANALYSIS")
print("=" * 80)
print(f"\nTest Case: {case_name}")
print(f"Base Directory: {base_output_dir}")
print(f"Plot to Screen: {PLOT_TO_SCREEN}")

# Load exact solution
group_name = case_to_group.get(case_name, case_name)
print(f"\nLoading exact solution from: {hdf5_file}")
print(f"Using group: {group_name}")
exact_data = load_exact_solution(hdf5_file, group_name)
print(f"Exact solution loaded: {len(exact_data['x'])} points")

# Scan for all output directories matching the pattern
print(f"\nScanning for output directories...")
all_data = defaultdict(lambda: {'sharp': None, 'epsilon': {}})
all_stop_times = []

for item in os.listdir(base_output_dir):
    item_path = os.path.join(base_output_dir, item)
    
    if os.path.isdir(item_path) and item.startswith(f'output_{case_name}_'):
        parsed = parse_output_directory(item, case_name)
        
        if parsed:
            riemann_solver = parsed['riemann_solver']
            print(f"\nFound: {item}")
            print(f"  Riemann Solver: {riemann_solver}")
            print(f"  Interface Type: {parsed['interface_type']}")
            
            # Load data
            data = load_amrex_data(item_path)
            
            if data:
                stop_time = data['time']
                all_stop_times.append(stop_time)
                
                if parsed['is_sharp']:
                    all_data[riemann_solver]['sharp'] = data
                    print(f"  Loaded sharp interface data: {len(data['x'])} points, t={stop_time:.6e}s")
                else:
                    all_data[riemann_solver]['epsilon'][parsed['epsilon_value']] = data
                    print(f"  Loaded epsilon={parsed['epsilon_value']} data: {len(data['x'])} points, t={stop_time:.6e}s")

if not all_data:
    print("\nERROR: No matching output directories found!")
    print(f"Expected pattern: output_{case_name}_{{RiemannSolver}}_Sharp_Interface")
    print(f"                  output_{case_name}_{{RiemannSolver}}_epsilon_{{value}}")
    exit(1)

# Determine maximum stop time
if all_stop_times:
    max_stop_time = max(all_stop_times)
    print(f"\n{'=' * 80}")
    print(f"Maximum stop time found: {max_stop_time:.6e}s")
    print(f"Filtering out simulations that stopped early...")
    
    # Filter out data that didn't reach max stop time (with small tolerance)
    tolerance = 1e-10
    filtered_data = defaultdict(lambda: {'sharp': None, 'epsilon': {}})
    
    for solver in all_data.keys():
        solver_data = all_data[solver]
        
        # Check sharp interface
        if solver_data['sharp'] and abs(solver_data['sharp']['time'] - max_stop_time) < tolerance:
            filtered_data[solver]['sharp'] = solver_data['sharp']
        elif solver_data['sharp']:
            print(f"  Excluding {solver} Sharp Interface (stopped at t={solver_data['sharp']['time']:.6e}s)")
        
        # Check epsilon variants
        for eps_val, eps_data in solver_data['epsilon'].items():
            if abs(eps_data['time'] - max_stop_time) < tolerance:
                filtered_data[solver]['epsilon'][eps_val] = eps_data
            else:
                print(f"  Excluding {solver} epsilon={eps_val} (stopped at t={eps_data['time']:.6e}s)")
    
    # Replace all_data with filtered data
    all_data = filtered_data
    
    # Remove solvers with no data left
    all_data = {k: v for k, v in all_data.items() if v['sharp'] or v['epsilon']}
    
    if not all_data:
        print("\nERROR: No simulations reached the maximum stop time!")
        exit(1)

print(f"\n{'=' * 80}")
print(f"Found {len(all_data)} Riemann solver(s) with complete simulations")
for solver in all_data.keys():
    sharp_status = "YES" if all_data[solver]['sharp'] else "NO"
    epsilon_count = len(all_data[solver]['epsilon'])
    print(f"  {solver}: Sharp={sharp_status}, Epsilon variants={epsilon_count}")

# Create output directory for images
os.makedirs('./Images', exist_ok=True)

# Get simulation time (from first available dataset)
sim_time = max_stop_time if all_stop_times else None

# Generate color palette
riemann_solvers = sorted(all_data.keys())
color_map = generate_color_palette(riemann_solvers)

# ============================================================================
# PLOT 1: Individual Riemann Solver Comparison (Sharp vs Epsilon variants)
# ============================================================================

print(f"\n{'=' * 80}")
print("GENERATING INDIVIDUAL RIEMANN SOLVER PLOTS")
print("=" * 80)

for solver in riemann_solvers:
    solver_data = all_data[solver]
    
    if not solver_data['sharp'] and not solver_data['epsilon']:
        continue
    
    print(f"\nPlotting {solver}...")
    
    # Get all epsilon values for this solver
    epsilon_values = sorted(solver_data['epsilon'].keys())
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    title = f'{format_label(case_name)} - {format_label(solver)}'
    if sim_time is not None:
        title += f' at t={sim_time:.6f}s'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    variables = ['density', 'velocity', 'pressure']
    ylabels = ['Density (kg/m$^3$)', 'Velocity (m/s)', 'Pressure (Pa)']
    
    for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
        ax = axes[idx]
        
        # Plot exact solution
        ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=EXACT_SOLUTION_LINEWIDTH, 
                label='Exact Solution', zorder=10)
        
        # Plot sharp interface
        if solver_data['sharp']:
            color = get_color_shade(color_map[solver], None, epsilon_values)
            ax.plot(solver_data['sharp']['x'], solver_data['sharp'][var], 
                   '--', color=color, linewidth=SHARP_INTERFACE_LINEWIDTH, 
                   label='Sharp Interface', zorder=5, alpha=0.8)
        
        # Plot epsilon variants (light to dark)
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'$\\epsilon = {eps_val}$'
            ax.plot(eps_data['x'], eps_data[var], 
                   '-', color=color, linewidth=EPSILON_LINEWIDTH, 
                   label=label, zorder=3, alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        
        # Only show legend on the first (top) subplot
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_{solver}_comparison'
    save_and_show_plot(output_filename, PLOT_TO_SCREEN)
    print(f"  Saved: {output_filename}")
    
    # ========================================================================
    # ERROR PLOT: Sharp vs Epsilon variants
    # ========================================================================
    
    if solver_data['sharp'] or solver_data['epsilon']:
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        
        title = f'{format_label(case_name)} - {format_label(solver)} - Error Analysis'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
        
        for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
            ax = axes[idx]
            
            # Sharp interface error
            if solver_data['sharp']:
                sharp_data = solver_data['sharp']
                exact_interp = np.interp(sharp_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(sharp_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], None, epsilon_values)
                ax.semilogy(sharp_data['x'], error, '--', color=color, 
                           linewidth=ERROR_SHARP_LINEWIDTH, label='Sharp Interface', alpha=0.8)
            
            # Epsilon variant errors
            for eps_val in epsilon_values:
                eps_data = solver_data['epsilon'][eps_val]
                exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(eps_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                label = f'$\\epsilon = {eps_val}$'
                ax.semilogy(eps_data['x'], error, '-', color=color, 
                           linewidth=ERROR_EPSILON_LINEWIDTH, label=label, alpha=0.7)
            
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
            
            # Only show legend on the first (top) subplot
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
            
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        
        plt.tight_layout()
        output_filename = f'./Images/{case_name}_{solver}_error'
        save_and_show_plot(output_filename, PLOT_TO_SCREEN)
        print(f"  Saved: {output_filename}")

# ============================================================================
# PLOT 2: All Sharp Interfaces Comparison
# ============================================================================

print(f"\n{'=' * 80}")
print("GENERATING SHARP INTERFACE COMPARISON PLOT")
print("=" * 80)

sharp_solvers = [s for s in riemann_solvers if all_data[s]['sharp']]

if sharp_solvers:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    title = f'{format_label(case_name)} - Sharp Interface Comparison'
    if sim_time is not None:
        title += f' at t={sim_time:.6f}s'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
        ax = axes[idx]
        
        # Plot exact solution
        ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=EXACT_SOLUTION_LINEWIDTH, 
                label='Exact Solution', zorder=10)
        
        # Plot each sharp interface
        for solver in sharp_solvers:
            sharp_data = all_data[solver]['sharp']
            ax.plot(sharp_data['x'], sharp_data[var], '--', 
                   color=color_map[solver], linewidth=SHARP_INTERFACE_LINEWIDTH, 
                   label=format_label(solver), alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        
        # Only show legend on the first (top) subplot
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_sharp_comparison'
    save_and_show_plot(output_filename, PLOT_TO_SCREEN)
    print(f"Saved: {output_filename}")
    
    # ========================================================================
    # ERROR PLOT: Sharp Interfaces
    # ========================================================================
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    
    title = f'{format_label(case_name)} - Sharp Interface Error Analysis'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    
    for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
        ax = axes[idx]
        
        for solver in sharp_solvers:
            sharp_data = all_data[solver]['sharp']
            exact_interp = np.interp(sharp_data['x'], exact_data['x'], exact_data[var])
            error = np.abs(sharp_data[var] - exact_interp)
            ax.semilogy(sharp_data['x'], error, '--', color=color_map[solver], 
                       linewidth=ERROR_SHARP_LINEWIDTH, label=format_label(solver), alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
        
        # Only show legend on the first (top) subplot
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_sharp_error'
    save_and_show_plot(output_filename, PLOT_TO_SCREEN)
    print(f"Saved: {output_filename}")

# ============================================================================
# PLOT 3: Aggregated Plot (All Solvers and Interface Thicknesses)
# ============================================================================

print(f"\n{'=' * 80}")
print("GENERATING AGGREGATED COMPARISON PLOT")
print("=" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

title = f'{format_label(case_name)} - Complete Comparison'
if sim_time is not None:
    title += f' at t={sim_time:.6f}s'
fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
    ax = axes[idx]
    
    # Plot exact solution
    ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=AGGREGATED_EXACT_LINEWIDTH, 
            label='Exact Solution', zorder=10)
    
    # Plot all solvers and variants
    for solver in riemann_solvers:
        solver_data = all_data[solver]
        epsilon_values = sorted(solver_data['epsilon'].keys())
        
        # Sharp interface
        if solver_data['sharp']:
            sharp_data = solver_data['sharp']
            color = get_color_shade(color_map[solver], None, epsilon_values)
            label = f'{format_label(solver)} - Sharp'
            ax.plot(sharp_data['x'], sharp_data[var], '--', 
                   color=color, linewidth=AGGREGATED_SHARP_LINEWIDTH, label=label, alpha=0.7)
        
        # Epsilon variants
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'{format_label(solver)} - $\\epsilon = {eps_val}$'
            ax.plot(eps_data['x'], eps_data[var], '-', 
                   color=color, linewidth=AGGREGATED_EPSILON_LINEWIDTH, label=label, alpha=0.6)
    
    ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    
    # Only show legend on the first (top) subplot
    if idx == 0:
        ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
    ax.tick_params(labelsize=TICK_FONTSIZE)

plt.tight_layout()
output_filename = f'./Images/{case_name}_aggregated_comparison'
save_and_show_plot(output_filename, PLOT_TO_SCREEN)
print(f"Saved: {output_filename}")

# ============================================================================
# ERROR PLOT: Aggregated
# ============================================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

title = f'{format_label(case_name)} - Complete Error Analysis'
fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

for idx, (var, ylabel) in enumerate(zip(variables, ylabels)):
    ax = axes[idx]
    
    for solver in riemann_solvers:
        solver_data = all_data[solver]
        epsilon_values = sorted(solver_data['epsilon'].keys())
        
        # Sharp interface
        if solver_data['sharp']:
            sharp_data = solver_data['sharp']
            exact_interp = np.interp(sharp_data['x'], exact_data['x'], exact_data[var])
            error = np.abs(sharp_data[var] - exact_interp)
            color = get_color_shade(color_map[solver], None, epsilon_values)
            label = f'{format_label(solver)} - Sharp'
            ax.semilogy(sharp_data['x'], error, '--', color=color, 
                       linewidth=AGGREGATED_SHARP_LINEWIDTH, label=label, alpha=0.7)
        
        # Epsilon variants
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
            error = np.abs(eps_data[var] - exact_interp)
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'{format_label(solver)} - $\\epsilon = {eps_val}$'
            ax.semilogy(eps_data['x'], error, '-', color=color, 
                       linewidth=AGGREGATED_EPSILON_LINEWIDTH, label=label, alpha=0.6)
    
    ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
    
    # Only show legend on the first (top) subplot
    if idx == 0:
        ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
    
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
    ax.tick_params(labelsize=TICK_FONTSIZE)

plt.tight_layout()
output_filename = f'./Images/{case_name}_aggregated_error'
save_and_show_plot(output_filename, PLOT_TO_SCREEN)
print(f"Saved: {output_filename}")

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
    
    # Sharp interface
    if solver_data['sharp']:
        errors = calculate_errors(solver_data['sharp'], exact_data)
        print(f"  Sharp Interface:")
        print(f"    Density  - L2: {errors['density_l2']:.6e}, Linf: {errors['density_linf']:.6e}")
        print(f"    Velocity - L2: {errors['velocity_l2']:.6e}, Linf: {errors['velocity_linf']:.6e}")
        print(f"    Pressure - L2: {errors['pressure_l2']:.6e}, Linf: {errors['pressure_linf']:.6e}")
    
    # Epsilon variants
    for eps_val in sorted(solver_data['epsilon'].keys()):
        errors = calculate_errors(solver_data['epsilon'][eps_val], exact_data)
        print(f"  epsilon = {eps_val}:")
        print(f"    Density  - L2: {errors['density_l2']:.6e}, Linf: {errors['density_linf']:.6e}")
        print(f"    Velocity - L2: {errors['velocity_l2']:.6e}, Linf: {errors['velocity_linf']:.6e}")
        print(f"    Pressure - L2: {errors['pressure_l2']:.6e}, Linf: {errors['pressure_linf']:.6e}")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE!")
print(f"All plots saved to: ./Images/")
print("=" * 80)
