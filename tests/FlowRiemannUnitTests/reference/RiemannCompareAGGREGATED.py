"""
Extract 1D slice from AMReX data using yt and compare with exact solution
Supports multiple Riemann solvers and interface thicknesses
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

# Font sizes for plots (easily adjustable)
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 10
TICK_FONTSIZE = 12

# Test case configuration
case_name = 'Toro1a'
Tammann = '.'  # Change to "." for not Tammann and "TammannEOS" for Tammann

# File paths
hdf5_file = fr'{Tammann}\{case_name}.hdf5'
base_output_dir = fr'..\..\..\bin\tests\FlowRiemannUnitTests'

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
    # Replace dashes with spaces
    text = text.replace('-', ' ')
    
    # Handle epsilon values
    epsilon_match = re.search(r'epsilon[_\s]*([\d.]+)', text)
    if epsilon_match:
        epsilon_val = epsilon_match.group(1)
        text = re.sub(r'epsilon[_\s]*[\d.]+', f'$\\epsilon = {epsilon_val}$', text)
    
    # Handle Sharp Interface
    text = text.replace('Sharp Interface', 'Sharp Interface')
    
    return text

def parse_output_directory(dir_name, case_name):
    """
    Parse output directory name to extract Riemann solver and interface type
    Expected formats:
    - output_{TestCase}_{RiemannSolver}_Sharp_Interface
    - output_{TestCase}_{RiemannSolver}_epsilon_{value}
    """
    pattern = rf'output_{re.escape(case_name)}_(.+?)_(Sharp_Interface|epsilon_[\d.]+)'
    match = re.match(pattern, dir_name)
    
    if match:
        riemann_solver = match.group(1)
        interface_type = match.group(2)
        
        # Extract epsilon value if present
        epsilon_value = None
        if interface_type.startswith('epsilon'):
            epsilon_match = re.search(r'epsilon_([\d.]+)', interface_type)
            if epsilon_match:
                epsilon_value = float(epsilon_match.group(1))
        
        return {
            'riemann_solver': riemann_solver,
            'interface_type': interface_type,
            'epsilon_value': epsilon_value,
            'is_sharp': interface_type == 'Sharp_Interface'
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
    """Get color shade based on epsilon value (lighter to darker)"""
    if epsilon_value is None:  # Sharp interface
        return base_color
    
    # Sort epsilon values to create gradient
    sorted_epsilons = sorted([e for e in epsilon_values if e is not None])
    
    if len(sorted_epsilons) == 0:
        return base_color
    
    # Find position in sorted list
    idx = sorted_epsilons.index(epsilon_value)
    
    # Create gradient from light (0.4) to dark (1.0)
    alpha = 0.4 + (0.6 * idx / max(len(sorted_epsilons) - 1, 1))
    
    # Convert color name to RGB and adjust brightness
    rgb = mcolors.to_rgb(base_color)
    # Darken the color
    rgb_adjusted = tuple(c * alpha + (1 - alpha) for c in rgb)
    
    return rgb_adjusted

# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("RIEMANN SOLVER COMPARISON ANALYSIS")
print("=" * 80)
print(f"\nTest Case: {case_name}")
print(f"Base Directory: {base_output_dir}")

# Load exact solution
group_name = case_to_group.get(case_name, case_name)
print(f"\nLoading exact solution from: {hdf5_file}")
print(f"Using group: {group_name}")
exact_data = load_exact_solution(hdf5_file, group_name)
print(f"Exact solution loaded: {len(exact_data['x'])} points")

# Scan for all output directories matching the pattern
print(f"\nScanning for output directories...")
all_data = defaultdict(lambda: {'sharp': None, 'epsilon': {}})

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
                if parsed['is_sharp']:
                    all_data[riemann_solver]['sharp'] = data
                    print(f"  Loaded sharp interface data: {len(data['x'])} points")
                else:
                    all_data[riemann_solver]['epsilon'][parsed['epsilon_value']] = data
                    print(f"  Loaded epsilon={parsed['epsilon_value']} data: {len(data['x'])} points")

if not all_data:
    print("\nERROR: No matching output directories found!")
    print(f"Expected pattern: output_{case_name}_{{RiemannSolver}}_Sharp_Interface")
    print(f"                  output_{case_name}_{{RiemannSolver}}_epsilon_{{value}}")
    exit(1)

print(f"\n{'=' * 80}")
print(f"Found {len(all_data)} Riemann solver(s)")
for solver in all_data.keys():
    sharp_status = "Y" if all_data[solver]['sharp'] else "N"
    epsilon_count = len(all_data[solver]['epsilon'])
    print(f"  {solver}: Sharp={sharp_status}, Epsilon variants={epsilon_count}")

# Create output directory for images
os.makedirs('./Images', exist_ok=True)

# Get simulation time (from first available dataset)
sim_time = None
for solver_data in all_data.values():
    if solver_data['sharp']:
        sim_time = solver_data['sharp']['time']
        break
    elif solver_data['epsilon']:
        sim_time = list(solver_data['epsilon'].values())[0]['time']
        break

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
        ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=2.5, 
                label='Exact Solution', zorder=10)
        
        # Plot sharp interface
        if solver_data['sharp']:
            color = get_color_shade(color_map[solver], None, epsilon_values)
            ax.plot(solver_data['sharp']['x'], solver_data['sharp'][var], 
                   '--', color=color, linewidth=2, 
                   label='Sharp Interface', zorder=5, alpha=0.8)
        
        # Plot epsilon variants (light to dark)
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'$\\epsilon = {eps_val}$'
            ax.plot(eps_data['x'], eps_data[var], 
                   '-', color=color, linewidth=1.5, 
                   label=label, zorder=3, alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_{solver}_comparison'
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    print(f"  Saved: {output_filename}")
    plt.close()
    
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
                           linewidth=2, label='Sharp Interface', alpha=0.8)
            
            # Epsilon variant errors
            for eps_val in epsilon_values:
                eps_data = solver_data['epsilon'][eps_val]
                exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(eps_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                label = f'$\\epsilon = {eps_val}$'
                ax.semilogy(eps_data['x'], error, '-', color=color, 
                           linewidth=1.5, label=label, alpha=0.7)
            
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
            ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        
        plt.tight_layout()
        output_filename = f'./Images/{case_name}_{solver}_error'
        plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
        plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
        print(f"  Saved: {output_filename}")
        plt.close()

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
        ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=2.5, 
                label='Exact Solution', zorder=10)
        
        # Plot each sharp interface
        for solver in sharp_solvers:
            sharp_data = all_data[solver]['sharp']
            ax.plot(sharp_data['x'], sharp_data[var], '--', 
                   color=color_map[solver], linewidth=2, 
                   label=format_label(solver), alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_sharp_comparison'
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()
    
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
                       linewidth=2, label=format_label(solver), alpha=0.7)
        
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    
    plt.tight_layout()
    output_filename = f'./Images/{case_name}_sharp_error'
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    print(f"Saved: {output_filename}")
    plt.close()

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
    ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=3, 
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
                   color=color, linewidth=1.8, label=label, alpha=0.7)
        
        # Epsilon variants
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'{format_label(solver)} - $\\epsilon = {eps_val}$'
            ax.plot(eps_data['x'], eps_data[var], '-', 
                   color=color, linewidth=1.3, label=label, alpha=0.6)
    
    ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
    ax.tick_params(labelsize=TICK_FONTSIZE)

plt.tight_layout()
output_filename = f'./Images/{case_name}_aggregated_comparison'
plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
print(f"Saved: {output_filename}")
plt.close()

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
                       linewidth=1.8, label=label, alpha=0.7)
        
        # Epsilon variants
        for eps_val in epsilon_values:
            eps_data = solver_data['epsilon'][eps_val]
            exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
            error = np.abs(eps_data[var] - exact_interp)
            color = get_color_shade(color_map[solver], eps_val, epsilon_values)
            label = f'{format_label(solver)} - $\\epsilon = {eps_val}$'
            ax.semilogy(eps_data['x'], error, '-', color=color, 
                       linewidth=1.3, label=label, alpha=0.6)
    
    ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error', fontsize=AXIS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
    ax.tick_params(labelsize=TICK_FONTSIZE)

plt.tight_layout()
output_filename = f'./Images/{case_name}_aggregated_error'
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
    
    # Sharp interface
    if solver_data['sharp']:
        errors = calculate_errors(solver_data['sharp'], exact_data)
        print(f"  Sharp Interface:")
        print(f"    Density  - L2: {errors['density_l2']:.6e}, L_inf: {errors['density_linf']:.6e}")
        print(f"    Velocity - L2: {errors['velocity_l2']:.6e}, L_inf: {errors['velocity_linf']:.6e}")
        print(f"    Pressure - L2: {errors['pressure_l2']:.6e}, L_inf: {errors['pressure_linf']:.6e}")
    
    # Epsilon variants
    for eps_val in sorted(solver_data['epsilon'].keys()):
        errors = calculate_errors(solver_data['epsilon'][eps_val], exact_data)
        print(f"  epsilon = {eps_val}:")
        print(f"    Density  - L2: {errors['density_l2']:.6e}, L_inf: {errors['density_linf']:.6e}")
        print(f"    Velocity - L2: {errors['velocity_l2']:.6e}, L_inf: {errors['velocity_linf']:.6e}")
        print(f"    Pressure - L2: {errors['pressure_l2']:.6e}, L_inf: {errors['pressure_linf']:.6e}")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE!")
print(f"All plots saved to: ./Images/")
print("=" * 80)
