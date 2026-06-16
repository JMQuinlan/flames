"""
===============================================================================
RIEMANN SOLVER + LIMITER + INTEGRATION COMPARISON ANALYSIS SCRIPT  (SINGLE-CASE)
===============================================================================

PURPOSE:
    Compare numerical solutions from multiple Riemann solvers, multiple
    spatial-reconstruction limiters, multiple time-integration schemes, and
    multiple interface thicknesses against exact solutions for shock tube
    problems.  Generates comprehensive comparison plots and error analysis on
    the cross product of these axes.

DIRECTORY NAMING CONVENTION (NEW):
    output_{TestCase}_{RiemannSolver}_{Limiter}_{Integration}_Sharp_Interface
    output_{TestCase}_{RiemannSolver}_{Limiter}_{Integration}_epsilon_{value}

    Example:
        output_Toro1a_HLLC_All_Mach_Furfaro_WENO5_RK4_Sharp_Interface
        output_Toro1a_HLLC_WENO3_SSPRK3_epsilon_1e-4

LEGACY FALLBACK:
    If the trailing token does not match a known integration label, the
    script falls back to integration='ForwardEuler'.  If the next token does
    not match a known limiter, the script falls back to limiter='Godunov'.

KNOWN LIMITERS:
    Godunov, Minmod, VanLeer, WENO3, WENO5
KNOWN INTEGRATIONS:
    ForwardEuler, Trapezoid, SSPRK3, RK4
    (case-insensitive matching against the trailing tokens)

OUTPUT LAYOUT:
    ./Images/{case}/{Integration}/{Limiter}/  - per-(integration,limiter) plots
    ./Images/{case}/{Integration}/limiters/   - per-(integration,solver) limiter cmp
    ./Images/{case}/{Integration}/solvers/    - per-(integration,limiter) solver cmp
    ./Images/{case}/integrations/             - integration-axis comparison plots
    ./Images/{case}/                          - mega aggregated plots + errors CSV

===============================================================================
"""

import yt
import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
import re
import csv
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

# Grand aggregated (busy plot) line weights
GRAND_EXACT_LINEWIDTH = 3.0
GRAND_SHARP_LINEWIDTH = 1.0
GRAND_EPSILON_LINEWIDTH = 0.7
GRAND_BASE_ALPHA = 0.55

# Limiters known to the test harness (must match RUN_RIEMANN.bash capitalization)
KNOWN_LIMITERS = ['Godunov', 'Minmod', 'VanLeer', 'WENO3', 'WENO5']
LIMITER_FALLBACK = 'Godunov'  # used when trailing token does not match any KNOWN_LIMITERS

# Linestyle scheme for limiter axis (when limiter is the secondary axis on a plot)
LIMITER_LINESTYLES = {
    'Godunov': '-',
    'Minmod':  '--',
    'VanLeer': '-.',
    'WENO3':   ':',
    'WENO5':   (0, (3, 1, 1, 1)),
}

# Integration schemes known to the test harness (must match RUN_RIEMANN.bash capitalization)
KNOWN_INTEGRATIONS = ['ForwardEuler', 'Trapezoid', 'SSPRK3', 'RK4']
INTEGRATION_FALLBACK = 'ForwardEuler'  # used when no integration token matches

# Linestyle scheme for limiter axis when used as the secondary axis on
# integration-centric plots (cycled in KNOWN_LIMITERS order).
INTEGRATION_LINESTYLES = {
    'ForwardEuler': 'solid',
    'Trapezoid':    'dashed',
    'SSPRK3':       'dashdot',
    'RK4':          'dotted',
}

# Linestyles cycled across limiters on integration-axis plots
LIMITER_CYCLE_LINESTYLES = ['solid', 'dashed', 'dotted', 'dashdot', (0, (5, 1))]

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
    # Mixed
    'Garrick' : 'Garrick',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_label(text):
    """Convert naming convention to display format with LaTeX"""
    epsilon_pattern = r'epsilon[_\s]*([\d.eE+-]+)'
    epsilon_matches = list(re.finditer(epsilon_pattern, text))

    if epsilon_matches:
        parts = []
        last_end = 0
        for match in epsilon_matches:
            before_text = text[last_end:match.start()].replace('-', ' ')
            parts.append(before_text)
            epsilon_val = match.group(1)
            parts.append(f'$\\epsilon = {epsilon_val}$')
            last_end = match.end()
        parts.append(text[last_end:].replace('-', ' '))
        text = ''.join(parts)
    else:
        text = text.replace('-', ' ')

    text = text.replace('Sharp Interface', 'Sharp Interface')
    return text


def _strip_trailing_limiter(token_str):
    """
    Given a string ending in possibly _{Limiter} (case-insensitive),
    return (solver_part, limiter_canonical_or_None).
    """
    # Try splitting on the last underscore and matching the trailing token
    # against KNOWN_LIMITERS case-insensitively.
    if '_' in token_str:
        head, tail = token_str.rsplit('_', 1)
        for known in KNOWN_LIMITERS:
            if tail.lower() == known.lower():
                return head, known
    else:
        # No underscore at all — the whole thing might be just a limiter name
        for known in KNOWN_LIMITERS:
            if token_str.lower() == known.lower():
                return '', known
    return token_str, None


def _strip_trailing_integration(token_str):
    """
    Given a string ending in possibly _{Integration} (case-insensitive),
    return (head_part, integration_canonical_or_None).
    """
    if '_' in token_str:
        head, tail = token_str.rsplit('_', 1)
        for known in KNOWN_INTEGRATIONS:
            if tail.lower() == known.lower():
                return head, known
    else:
        for known in KNOWN_INTEGRATIONS:
            if token_str.lower() == known.lower():
                return '', known
    return token_str, None


def parse_output_directory(dir_name, case_name):
    """
    Parse output directory name to extract Riemann solver, limiter, integration
    scheme, and interface type.

    Expected NEW formats:
    - output_{TestCase}_{RiemannSolver}_{Limiter}_{Integration}_Sharp_Interface
    - output_{TestCase}_{RiemannSolver}_{Limiter}_{Integration}_epsilon_{value}

    Legacy fallbacks:
    - missing integration token -> integration = INTEGRATION_FALLBACK ('ForwardEuler')
    - missing limiter     token -> limiter     = LIMITER_FALLBACK     ('Godunov')

    Returns dict with keys: riemann_solver, limiter, integration,
                            interface_type, epsilon_value, is_sharp
    """
    prefix = f'output_{case_name}_'
    if not dir_name.startswith(prefix):
        return None

    remainder = dir_name[len(prefix):]

    def _split_solver_limiter_integration(suffix):
        """Strip trailing integration first, then trailing limiter."""
        head_after_int, integration = _strip_trailing_integration(suffix)
        if integration is None:
            # No integration token — fall back, head is the whole thing
            integration = INTEGRATION_FALLBACK
            head_after_int = suffix
        solver, limiter = _strip_trailing_limiter(head_after_int)
        if limiter is None:
            solver = head_after_int
            limiter = LIMITER_FALLBACK
        return solver, limiter, integration

    # Sharp interface variant
    if 'Sharp_Interface' in remainder:
        prefix_str = remainder.split('_Sharp_Interface')[0]
        solver, limiter, integration = _split_solver_limiter_integration(prefix_str)
        return {
            'riemann_solver': solver,
            'limiter': limiter,
            'integration': integration,
            'interface_type': 'Sharp_Interface',
            'epsilon_value': None,
            'is_sharp': True,
        }

    # Epsilon variant
    epsilon_pattern = r'_epsilon_([\d.]+(?:[eE][+-]?\d+)?)'
    epsilon_match = re.search(epsilon_pattern, remainder)
    if epsilon_match:
        epsilon_str = epsilon_match.group(1)
        try:
            epsilon_value = float(epsilon_str)
        except ValueError:
            print(f"Warning: Could not parse epsilon value '{epsilon_str}' in '{dir_name}'")
            return None

        prefix_str = remainder.split('_epsilon_')[0]
        solver, limiter, integration = _split_solver_limiter_integration(prefix_str)

        return {
            'riemann_solver': solver,
            'limiter': limiter,
            'integration': integration,
            'interface_type': f'epsilon_{epsilon_str}',
            'epsilon_value': epsilon_value,
            'is_sharp': False,
        }

    return None


def load_amrex_data(output_dir):
    """Load data from AMReX output directory using yt"""
    plot_files = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and item.endswith('cell'):
            plot_files.append(item_path)

    if not plot_files:
        return None

    plot_files.sort()
    last_plot = plot_files[-1]

    ds = yt.load(last_plot)

    ray_start = ds.arr([-1.0, 0.0, 0.0], 'code_length')
    ray_end = ds.arr([1.0, 0.0, 0.0], 'code_length')
    ray = ds.ray(ray_start, ray_end)

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
    velocity_exact_interp = np.interp(numerical_data['x'], exact_data['x'], exact_data['velocity'])
    pressure_exact_interp = np.interp(numerical_data['x'], exact_data['x'], exact_data['pressure'])
    density_exact_interp  = np.interp(numerical_data['x'], exact_data['x'], exact_data['density'])

    errors = {
        'velocity_l2':  np.linalg.norm(numerical_data['velocity'] - velocity_exact_interp),
        'velocity_linf': np.max(np.abs(numerical_data['velocity'] - velocity_exact_interp)),
        'pressure_l2':  np.linalg.norm(numerical_data['pressure'] - pressure_exact_interp),
        'pressure_linf': np.max(np.abs(numerical_data['pressure'] - pressure_exact_interp)),
        'density_l2':   np.linalg.norm(numerical_data['density']  - density_exact_interp),
        'density_linf': np.max(np.abs(numerical_data['density']  - density_exact_interp)),
    }

    return errors


def generate_color_palette(riemann_solvers):
    """Generate color palette for Riemann solvers"""
    base_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {}
    for idx, solver in enumerate(riemann_solvers):
        color_map[solver] = base_colors[idx % len(base_colors)]
    return color_map


def generate_limiter_color_map(limiters):
    """Distinct colors per limiter using matplotlib tab10."""
    cmap = plt.get_cmap('tab10')
    color_map = {}
    for idx, lim in enumerate(limiters):
        color_map[lim] = cmap(idx % 10)
    return color_map


def generate_integration_color_map(integrations):
    """Distinct colors per integration scheme using matplotlib Set2."""
    cmap = plt.get_cmap('Set2')
    color_map = {}
    for idx, integ in enumerate(integrations):
        color_map[integ] = cmap(idx % 8)
    return color_map


def integration_linestyle(integration):
    return INTEGRATION_LINESTYLES.get(integration, 'solid')


def limiter_cycle_linestyle(limiter, ordered_limiters):
    """Return a linestyle from the cycle keyed by index of limiter in
    ordered_limiters; falls back to solid for unknowns."""
    try:
        idx = ordered_limiters.index(limiter)
    except ValueError:
        return 'solid'
    return LIMITER_CYCLE_LINESTYLES[idx % len(LIMITER_CYCLE_LINESTYLES)]


def interface_alpha(epsilon_value, all_epsilons, sharp_alpha=1.0):
    """Alpha by interface type: sharp = 1.0, epsilons descending."""
    if epsilon_value is None:
        return sharp_alpha
    sorted_eps = sorted([e for e in all_epsilons if e is not None])
    if not sorted_eps:
        return 0.6
    idx = sorted_eps.index(epsilon_value)
    # First (smallest) epsilon ~0.7, fading down to ~0.3 for largest
    n = max(len(sorted_eps) - 1, 1)
    return max(0.3, 0.75 - 0.45 * idx / n)


def get_color_shade(base_color, epsilon_value, epsilon_values):
    """Get color shade based on epsilon value (darker to lighter)"""
    if epsilon_value is None:  # Sharp interface
        return base_color

    sorted_epsilons = sorted([e for e in epsilon_values if e is not None])
    if len(sorted_epsilons) == 0:
        return base_color

    idx = sorted_epsilons.index(epsilon_value)
    alpha = 1.0 - (0.6 * idx / max(len(sorted_epsilons) - 1, 1))
    rgb = mcolors.to_rgb(base_color)
    rgb_adjusted = tuple(c * alpha + (1 - alpha) for c in rgb)
    return rgb_adjusted


def limiter_linestyle(limiter):
    return LIMITER_LINESTYLES.get(limiter, '-')


def epsilon_alpha(epsilon_value, all_epsilons, base_alpha=GRAND_BASE_ALPHA):
    """Map epsilon to alpha for the grand-aggregated plot."""
    if epsilon_value is None:
        return min(1.0, base_alpha + 0.35)
    sorted_eps = sorted([e for e in all_epsilons if e is not None])
    if not sorted_eps:
        return base_alpha
    idx = sorted_eps.index(epsilon_value)
    # Smaller epsilon -> higher alpha
    return max(0.2, base_alpha - 0.1 * idx)


def save_and_show_plot(output_filename, plot_to_screen=False):
    """Save plot to file and optionally display on screen"""
    plt.savefig(output_filename + '.png', format='png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.eps', format='eps', bbox_inches='tight')
    if plot_to_screen:
        plt.show()
    plt.close()


# ============================================================================
# DATA-LOADING + FILTERING
# ============================================================================

def _new_sli_dict():
    """Factory for a per-(solver,limiter,integration) data dict."""
    return {'sharp': None, 'epsilon': {}}


def collect_all_data(case_name, base_output_dir):
    """
    Walk the case directory and return:
        all_data[solver][limiter][integration] =
            {'sharp': data_or_None, 'epsilon': {eps: data, ...}}
        all_stop_times = list of all loaded stop times
    """
    all_data = defaultdict(lambda: defaultdict(lambda: defaultdict(_new_sli_dict)))
    all_stop_times = []

    for item in sorted(os.listdir(base_output_dir)):
        item_path = os.path.join(base_output_dir, item)
        if not (os.path.isdir(item_path) and item.startswith(f'output_{case_name}_')):
            continue

        parsed = parse_output_directory(item, case_name)
        if not parsed:
            continue

        solver      = parsed['riemann_solver']
        limiter     = parsed['limiter']
        integration = parsed['integration']
        print(f"\nFound: {item}")
        print(f"  Riemann Solver: {solver}")
        print(f"  Limiter:        {limiter}")
        print(f"  Integration:    {integration}")
        print(f"  Interface Type: {parsed['interface_type']}")

        data = load_amrex_data(item_path)
        if not data:
            continue

        stop_time = data['time']
        all_stop_times.append(stop_time)

        if parsed['is_sharp']:
            all_data[solver][limiter][integration]['sharp'] = data
            print(f"  Loaded sharp interface data: {len(data['x'])} points, t={stop_time:.6e}s")
        else:
            all_data[solver][limiter][integration]['epsilon'][parsed['epsilon_value']] = data
            print(f"  Loaded epsilon={parsed['epsilon_value']} data: {len(data['x'])} points, t={stop_time:.6e}s")

    return all_data, all_stop_times


def filter_by_stop_time(all_data, max_stop_time, tolerance=1e-10):
    """
    Filter per-(solver, limiter, integration) tuple. Drop combos whose data did
    not reach max_stop_time. Drop empty (solver, limiter) and empty solvers.
    """
    filtered = defaultdict(lambda: defaultdict(lambda: defaultdict(_new_sli_dict)))

    for solver, lim_dict in all_data.items():
        for limiter, integ_dict in lim_dict.items():
            for integration, sli_data in integ_dict.items():
                kept_anything = False

                if sli_data['sharp']:
                    if abs(sli_data['sharp']['time'] - max_stop_time) < tolerance:
                        filtered[solver][limiter][integration]['sharp'] = sli_data['sharp']
                        kept_anything = True
                    else:
                        print(f"  Excluding {solver} / {limiter} / {integration} Sharp_Interface "
                              f"(stopped at t={sli_data['sharp']['time']:.6e}s)")

                for eps_val, eps_data in sli_data['epsilon'].items():
                    if abs(eps_data['time'] - max_stop_time) < tolerance:
                        filtered[solver][limiter][integration]['epsilon'][eps_val] = eps_data
                        kept_anything = True
                    else:
                        print(f"  Excluding {solver} / {limiter} / {integration} epsilon={eps_val} "
                              f"(stopped at t={eps_data['time']:.6e}s)")

                if not kept_anything:
                    if integration in filtered[solver][limiter]:
                        del filtered[solver][limiter][integration]

            if limiter in filtered[solver] and not filtered[solver][limiter]:
                del filtered[solver][limiter]

        if solver in filtered and not filtered[solver]:
            del filtered[solver]

    return filtered


def sort_integrations(present_set):
    """Sort integration labels: KNOWN_INTEGRATIONS order, then unknowns alpha."""
    return ([i for i in KNOWN_INTEGRATIONS if i in present_set]
            + sorted([i for i in present_set if i not in KNOWN_INTEGRATIONS]))


def sort_limiters(present_set):
    return ([l for l in KNOWN_LIMITERS if l in present_set]
            + sorted([l for l in present_set if l not in KNOWN_LIMITERS]))


# ============================================================================
# PLOT GENERATION HELPERS
# ============================================================================

VARIABLES = ['density', 'velocity', 'pressure']
YLABELS   = ['Density (kg/m$^3$)', 'Velocity (m/s)', 'Pressure (Pa)']


def _plot_exact(ax, exact_data, var, lw):
    ax.plot(exact_data['x'], exact_data[var], 'k-', linewidth=lw,
            label='Exact Solution', zorder=10)


def _finalize_axis(ax, idx, ylabel, exact_data, legend_kwargs=None):
    ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    if idx == 0 and legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
    ax.tick_params(labelsize=TICK_FONTSIZE)


# ----------------------------------------------------------------------------
# Slice helpers: extract a {solver: {limiter: {sharp,epsilon}}} sub-view fixed
# at one integration scheme.  Lets us reuse the existing plot routines without
# rewriting their inner logic.
# ----------------------------------------------------------------------------

def slice_by_integration(all_data, integration):
    """Return {solver: {limiter: {'sharp': ..., 'epsilon': {...}}}} restricted
    to one integration scheme.  Solvers/limiters with no data are omitted."""
    sliced = {}
    for solver, lim_dict in all_data.items():
        for limiter, integ_dict in lim_dict.items():
            if integration not in integ_dict:
                continue
            sli = integ_dict[integration]
            if not sli['sharp'] and not sli['epsilon']:
                continue
            sliced.setdefault(solver, {})[limiter] = sli
    return sliced


# ----------------------------------------------------------------------------
# (A) Legacy plots — replicated per (integration, limiter), written to
#     ./Images/{case}/{Integration}/{Limiter}/
# ----------------------------------------------------------------------------

def generate_legacy_plots_for_limiter(case_name, limiter, all_data, exact_data,
                                      sim_time, color_map, base_out_dir):
    """
    Reproduce the original solver-axis plots (per-solver, sharp-comparison,
    aggregated) but only including data for the given limiter.

    `all_data` here is the integration-sliced view: {solver: {limiter: sli_dict}}.
    """
    out_dir = os.path.join(base_out_dir, limiter)
    os.makedirs(out_dir, exist_ok=True)

    # Solvers that have *any* data for this limiter
    solvers = sorted([s for s in all_data.keys() if limiter in all_data[s]])
    if not solvers:
        print(f"  [{limiter}] no data for any solver - skipping")
        return

    # ------------- Per-solver comparison & error plots -------------
    print(f"\n[{limiter}] Per-solver plots ...")
    for solver in solvers:
        sl = all_data[solver][limiter]
        if not sl['sharp'] and not sl['epsilon']:
            continue

        epsilon_values = sorted(sl['epsilon'].keys())

        # Comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        title = f'{format_label(case_name)} - {format_label(solver)} [{limiter}]'
        if sim_time is not None:
            title += f' at t={sim_time:.6f}s'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            _plot_exact(ax, exact_data, var, EXACT_SOLUTION_LINEWIDTH)

            if sl['sharp']:
                color = get_color_shade(color_map[solver], None, epsilon_values)
                ax.plot(sl['sharp']['x'], sl['sharp'][var], '--', color=color,
                        linewidth=SHARP_INTERFACE_LINEWIDTH, label='Sharp Interface',
                        zorder=5, alpha=0.8)

            for eps_val in epsilon_values:
                eps_data = sl['epsilon'][eps_val]
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                ax.plot(eps_data['x'], eps_data[var], '-', color=color,
                        linewidth=EPSILON_LINEWIDTH, label=f'$\\epsilon = {eps_val}$',
                        zorder=3, alpha=0.7)

            _finalize_axis(ax, idx, ylabel, exact_data,
                           legend_kwargs={'fontsize': LEGEND_FONTSIZE, 'loc': 'best'})

        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{solver}_{limiter}_comparison')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

        # Error
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        title = (f'{format_label(case_name)} - {format_label(solver)} '
                 f'[{limiter}] - Error Analysis')
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]

            if sl['sharp']:
                sharp_data = sl['sharp']
                exact_interp = np.interp(sharp_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(sharp_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], None, epsilon_values)
                ax.semilogy(sharp_data['x'], error, '--', color=color,
                            linewidth=ERROR_SHARP_LINEWIDTH, label='Sharp Interface', alpha=0.8)

            for eps_val in epsilon_values:
                eps_data = sl['epsilon'][eps_val]
                exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(eps_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                ax.semilogy(eps_data['x'], error, '-', color=color,
                            linewidth=ERROR_EPSILON_LINEWIDTH,
                            label=f'$\\epsilon = {eps_val}$', alpha=0.7)

            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                          fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)

        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{solver}_{limiter}_error')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

    # ------------- All-sharp comparison & error -------------
    sharp_solvers = [s for s in solvers if all_data[s][limiter]['sharp']]
    if sharp_solvers:
        # Comparison
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        title = f'{format_label(case_name)} - Sharp Interface Comparison [{limiter}]'
        if sim_time is not None:
            title += f' at t={sim_time:.6f}s'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            _plot_exact(ax, exact_data, var, EXACT_SOLUTION_LINEWIDTH)
            for solver in sharp_solvers:
                sd = all_data[solver][limiter]['sharp']
                ax.plot(sd['x'], sd[var], '--', color=color_map[solver],
                        linewidth=SHARP_INTERFACE_LINEWIDTH,
                        label=format_label(solver), alpha=0.7)
            _finalize_axis(ax, idx, ylabel, exact_data,
                           legend_kwargs={'fontsize': LEGEND_FONTSIZE, 'loc': 'best'})
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{limiter}_sharp_comparison')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

        # Error
        fig, axes = plt.subplots(3, 1, figsize=(12, 14))
        title = f'{format_label(case_name)} - Sharp Interface Error [{limiter}]'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            for solver in sharp_solvers:
                sd = all_data[solver][limiter]['sharp']
                exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                error = np.abs(sd[var] - exact_interp)
                ax.semilogy(sd['x'], error, '--', color=color_map[solver],
                            linewidth=ERROR_SHARP_LINEWIDTH,
                            label=format_label(solver), alpha=0.7)
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                          fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE, loc='best')
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{limiter}_sharp_error')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

    # ------------- Aggregated (all solvers, all interfaces) for this limiter -------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    title = f'{format_label(case_name)} - Complete Comparison [{limiter}]'
    if sim_time is not None:
        title += f' at t={sim_time:.6f}s'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        _plot_exact(ax, exact_data, var, AGGREGATED_EXACT_LINEWIDTH)
        for solver in solvers:
            sl = all_data[solver][limiter]
            epsilon_values = sorted(sl['epsilon'].keys())
            if sl['sharp']:
                sd = sl['sharp']
                color = get_color_shade(color_map[solver], None, epsilon_values)
                ax.plot(sd['x'], sd[var], '--', color=color,
                        linewidth=AGGREGATED_SHARP_LINEWIDTH,
                        label=f'{format_label(solver)} - Sharp', alpha=0.7)
            for eps_val in epsilon_values:
                eps_data = sl['epsilon'][eps_val]
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                ax.plot(eps_data['x'], eps_data[var], '-', color=color,
                        linewidth=AGGREGATED_EPSILON_LINEWIDTH,
                        label=f'{format_label(solver)} - $\\epsilon = {eps_val}$',
                        alpha=0.6)
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'{case_name}_{limiter}_aggregated_comparison')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    title = f'{format_label(case_name)} - Complete Error Analysis [{limiter}]'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        for solver in solvers:
            sl = all_data[solver][limiter]
            epsilon_values = sorted(sl['epsilon'].keys())
            if sl['sharp']:
                sd = sl['sharp']
                exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                error = np.abs(sd[var] - exact_interp)
                color = get_color_shade(color_map[solver], None, epsilon_values)
                ax.semilogy(sd['x'], error, '--', color=color,
                            linewidth=AGGREGATED_SHARP_LINEWIDTH,
                            label=f'{format_label(solver)} - Sharp', alpha=0.7)
            for eps_val in epsilon_values:
                eps_data = sl['epsilon'][eps_val]
                exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                error = np.abs(eps_data[var] - exact_interp)
                color = get_color_shade(color_map[solver], eps_val, epsilon_values)
                ax.semilogy(eps_data['x'], error, '-', color=color,
                            linewidth=AGGREGATED_EPSILON_LINEWIDTH,
                            label=f'{format_label(solver)} - $\\epsilon = {eps_val}$',
                            alpha=0.6)
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                      fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(out_dir, f'{case_name}_{limiter}_aggregated_error')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (B) Per-solver limiter comparison: ./Images/{case}/limiters/
# ----------------------------------------------------------------------------

def generate_per_solver_limiter_plots(case_name, all_data, exact_data, sim_time,
                                      limiter_color_map, base_out_dir):
    out_dir = os.path.join(base_out_dir, 'limiters')
    os.makedirs(out_dir, exist_ok=True)

    for solver in sorted(all_data.keys()):
        limiter_dict = all_data[solver]
        present_limiters = [lim for lim in KNOWN_LIMITERS if lim in limiter_dict]
        # include any unknown limiters that might appear too
        for lim in limiter_dict.keys():
            if lim not in present_limiters:
                present_limiters.append(lim)
        if not present_limiters:
            continue

        # Comparison plot
        fig, axes = plt.subplots(3, 1, figsize=(13, 15))
        title = f'{format_label(case_name)} - {format_label(solver)} - Limiter Comparison'
        if sim_time is not None:
            title += f' at t={sim_time:.6f}s'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            _plot_exact(ax, exact_data, var, EXACT_SOLUTION_LINEWIDTH)
            for limiter in present_limiters:
                sl = limiter_dict[limiter]
                color = limiter_color_map.get(limiter, 'black')
                if sl['sharp']:
                    sd = sl['sharp']
                    ax.plot(sd['x'], sd[var], '-', color=color,
                            linewidth=SHARP_INTERFACE_LINEWIDTH,
                            label=f'{limiter} - Sharp', alpha=0.9)
                eps_vals = sorted(sl['epsilon'].keys())
                for eps_val in eps_vals:
                    eps_data = sl['epsilon'][eps_val]
                    ax.plot(eps_data['x'], eps_data[var], '--', color=color,
                            linewidth=EPSILON_LINEWIDTH,
                            label=f'{limiter} - $\\epsilon={eps_val}$',
                            alpha=epsilon_alpha(eps_val, eps_vals, base_alpha=0.6))
            _finalize_axis(ax, idx, ylabel, exact_data,
                           legend_kwargs={'fontsize': LEGEND_FONTSIZE - 1,
                                          'loc': 'best', 'ncol': 2})
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{solver}_limiter_comparison')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

        # Error plot
        fig, axes = plt.subplots(3, 1, figsize=(13, 15))
        title = f'{format_label(case_name)} - {format_label(solver)} - Limiter Error Analysis'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            for limiter in present_limiters:
                sl = limiter_dict[limiter]
                color = limiter_color_map.get(limiter, 'black')
                if sl['sharp']:
                    sd = sl['sharp']
                    exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                    err = np.abs(sd[var] - exact_interp)
                    ax.semilogy(sd['x'], err, '-', color=color,
                                linewidth=ERROR_SHARP_LINEWIDTH,
                                label=f'{limiter} - Sharp', alpha=0.9)
                eps_vals = sorted(sl['epsilon'].keys())
                for eps_val in eps_vals:
                    eps_data = sl['epsilon'][eps_val]
                    exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                    err = np.abs(eps_data[var] - exact_interp)
                    ax.semilogy(eps_data['x'], err, '--', color=color,
                                linewidth=ERROR_EPSILON_LINEWIDTH,
                                label=f'{limiter} - $\\epsilon={eps_val}$',
                                alpha=epsilon_alpha(eps_val, eps_vals, base_alpha=0.6))
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                          fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{solver}_limiter_error')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (C) Per-limiter solver comparison: ./Images/{case}/solvers/
# ----------------------------------------------------------------------------

def generate_per_limiter_solver_plots(case_name, all_data, exact_data, sim_time,
                                      solver_color_map, base_out_dir, all_limiters):
    out_dir = os.path.join(base_out_dir, 'solvers')
    os.makedirs(out_dir, exist_ok=True)

    for limiter in all_limiters:
        # Solvers that have this limiter
        solvers = [s for s in sorted(all_data.keys()) if limiter in all_data[s]]
        if not solvers:
            continue

        # Comparison
        fig, axes = plt.subplots(3, 1, figsize=(13, 15))
        title = f'{format_label(case_name)} - {limiter} - Solver Comparison'
        if sim_time is not None:
            title += f' at t={sim_time:.6f}s'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            _plot_exact(ax, exact_data, var, EXACT_SOLUTION_LINEWIDTH)
            ls = limiter_linestyle(limiter)
            for solver in solvers:
                sl = all_data[solver][limiter]
                color = solver_color_map[solver]
                eps_vals = sorted(sl['epsilon'].keys())
                if sl['sharp']:
                    sd = sl['sharp']
                    ax.plot(sd['x'], sd[var], linestyle=ls, color=color,
                            linewidth=SHARP_INTERFACE_LINEWIDTH,
                            label=f'{format_label(solver)} - Sharp', alpha=0.85)
                for eps_val in eps_vals:
                    eps_data = sl['epsilon'][eps_val]
                    shaded = get_color_shade(color, eps_val, eps_vals)
                    ax.plot(eps_data['x'], eps_data[var], linestyle=ls, color=shaded,
                            linewidth=EPSILON_LINEWIDTH,
                            label=f'{format_label(solver)} - $\\epsilon={eps_val}$',
                            alpha=0.65)
            _finalize_axis(ax, idx, ylabel, exact_data,
                           legend_kwargs={'fontsize': LEGEND_FONTSIZE - 1,
                                          'loc': 'best', 'ncol': 2})
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{limiter}_solver_comparison')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

        # Error
        fig, axes = plt.subplots(3, 1, figsize=(13, 15))
        title = f'{format_label(case_name)} - {limiter} - Solver Error Analysis'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            ls = limiter_linestyle(limiter)
            for solver in solvers:
                sl = all_data[solver][limiter]
                color = solver_color_map[solver]
                eps_vals = sorted(sl['epsilon'].keys())
                if sl['sharp']:
                    sd = sl['sharp']
                    exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                    err = np.abs(sd[var] - exact_interp)
                    ax.semilogy(sd['x'], err, linestyle=ls, color=color,
                                linewidth=ERROR_SHARP_LINEWIDTH,
                                label=f'{format_label(solver)} - Sharp', alpha=0.85)
                for eps_val in eps_vals:
                    eps_data = sl['epsilon'][eps_val]
                    shaded = get_color_shade(color, eps_val, eps_vals)
                    exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                    err = np.abs(eps_data[var] - exact_interp)
                    ax.semilogy(eps_data['x'], err, linestyle=ls, color=shaded,
                                linewidth=ERROR_EPSILON_LINEWIDTH,
                                label=f'{format_label(solver)} - $\\epsilon={eps_val}$',
                                alpha=0.65)
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                          fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{limiter}_solver_error')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (D) Grand aggregated (legacy, ForwardEuler-only for backward compatibility)
# ----------------------------------------------------------------------------

def generate_grand_aggregated_plots(case_name, all_data_sliced, exact_data, sim_time,
                                    solver_color_map, base_out_dir):
    """
    Operates on integration-sliced data {solver: {limiter: sli_dict}} so it
    still produces the old grand-aggregated plot for one fixed integration
    scheme (call site passes the ForwardEuler slice).
    """
    all_epsilons = set()
    for solver, lim_dict in all_data_sliced.items():
        for limiter, sl in lim_dict.items():
            for eps in sl['epsilon'].keys():
                all_epsilons.add(eps)
    all_epsilons = sorted(all_epsilons)

    # Comparison
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    title = f'{format_label(case_name)} - Grand Aggregated Comparison (solver x limiter x interface)'
    if sim_time is not None:
        title += f' at t={sim_time:.6f}s'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        _plot_exact(ax, exact_data, var, GRAND_EXACT_LINEWIDTH)
        for solver in sorted(all_data_sliced.keys()):
            color = solver_color_map[solver]
            for limiter, sl in all_data_sliced[solver].items():
                ls = limiter_linestyle(limiter)
                if sl['sharp']:
                    sd = sl['sharp']
                    ax.plot(sd['x'], sd[var], linestyle=ls, color=color,
                            linewidth=GRAND_SHARP_LINEWIDTH,
                            label=f'{format_label(solver)} / {limiter} / Sharp',
                            alpha=epsilon_alpha(None, all_epsilons))
                for eps_val in sorted(sl['epsilon'].keys()):
                    eps_data = sl['epsilon'][eps_val]
                    ax.plot(eps_data['x'], eps_data[var], linestyle=ls, color=color,
                            linewidth=GRAND_EPSILON_LINEWIDTH,
                            label=f'{format_label(solver)} / {limiter} / $\\epsilon={eps_val}$',
                            alpha=epsilon_alpha(eps_val, all_epsilons))
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=max(6, LEGEND_FONTSIZE - 3), loc='best', ncol=3)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(base_out_dir, f'{case_name}_grand_aggregated_comparison')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")

    # Error
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    title = f'{format_label(case_name)} - Grand Aggregated Error Analysis'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        for solver in sorted(all_data_sliced.keys()):
            color = solver_color_map[solver]
            for limiter, sl in all_data_sliced[solver].items():
                ls = limiter_linestyle(limiter)
                if sl['sharp']:
                    sd = sl['sharp']
                    exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                    err = np.abs(sd[var] - exact_interp)
                    ax.semilogy(sd['x'], err, linestyle=ls, color=color,
                                linewidth=GRAND_SHARP_LINEWIDTH,
                                label=f'{format_label(solver)} / {limiter} / Sharp',
                                alpha=epsilon_alpha(None, all_epsilons))
                for eps_val in sorted(sl['epsilon'].keys()):
                    eps_data = sl['epsilon'][eps_val]
                    exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                    err = np.abs(eps_data[var] - exact_interp)
                    ax.semilogy(eps_data['x'], err, linestyle=ls, color=color,
                                linewidth=GRAND_EPSILON_LINEWIDTH,
                                label=f'{format_label(solver)} / {limiter} / $\\epsilon={eps_val}$',
                                alpha=epsilon_alpha(eps_val, all_epsilons))
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                      fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=max(6, LEGEND_FONTSIZE - 3), loc='best', ncol=3)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(base_out_dir, f'{case_name}_grand_aggregated_error')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (E) NEW: Per-(solver, limiter) integration comparison
#     ./Images/{case}/integrations/{case}_{Solver}_{Limiter}_integration_*
# ----------------------------------------------------------------------------

def generate_per_solver_limiter_integration_plots(case_name, all_data, exact_data,
                                                  sim_time, integration_color_map,
                                                  base_out_dir):
    out_dir = os.path.join(base_out_dir, 'integrations')
    os.makedirs(out_dir, exist_ok=True)

    # Build global epsilon list for alpha mapping
    all_epsilons = set()
    for solver, lim_dict in all_data.items():
        for limiter, integ_dict in lim_dict.items():
            for integration, sli in integ_dict.items():
                for eps in sli['epsilon'].keys():
                    all_epsilons.add(eps)
    all_epsilons = sorted(all_epsilons)

    for solver in sorted(all_data.keys()):
        for limiter in sort_limiters(set(all_data[solver].keys())):
            integ_dict = all_data[solver][limiter]
            present_integrations = sort_integrations(set(integ_dict.keys()))
            if not present_integrations:
                continue

            # Comparison
            fig, axes = plt.subplots(3, 1, figsize=(13, 15))
            title = (f'{format_label(case_name)} - {format_label(solver)} '
                     f'[{limiter}] - Integration Comparison')
            if sim_time is not None:
                title += f' at t={sim_time:.6f}s'
            fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

            for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
                ax = axes[idx]
                _plot_exact(ax, exact_data, var, EXACT_SOLUTION_LINEWIDTH)
                for integration in present_integrations:
                    sli = integ_dict[integration]
                    color = integration_color_map.get(integration, 'black')
                    ls = integration_linestyle(integration)
                    if sli['sharp']:
                        sd = sli['sharp']
                        ax.plot(sd['x'], sd[var], linestyle=ls, color=color,
                                linewidth=SHARP_INTERFACE_LINEWIDTH,
                                label=f'{integration} - Sharp',
                                alpha=interface_alpha(None, all_epsilons))
                    for eps_val in sorted(sli['epsilon'].keys()):
                        eps_data = sli['epsilon'][eps_val]
                        ax.plot(eps_data['x'], eps_data[var], linestyle=ls, color=color,
                                linewidth=EPSILON_LINEWIDTH,
                                label=f'{integration} - $\\epsilon={eps_val}$',
                                alpha=interface_alpha(eps_val, all_epsilons))
                _finalize_axis(ax, idx, ylabel, exact_data,
                               legend_kwargs={'fontsize': LEGEND_FONTSIZE - 1,
                                              'loc': 'best', 'ncol': 2})
            plt.tight_layout()
            fname = os.path.join(out_dir,
                                 f'{case_name}_{solver}_{limiter}_integration_comparison')
            save_and_show_plot(fname, PLOT_TO_SCREEN)
            print(f"  Saved: {fname}")

            # Error
            fig, axes = plt.subplots(3, 1, figsize=(13, 15))
            title = (f'{format_label(case_name)} - {format_label(solver)} '
                     f'[{limiter}] - Integration Error Analysis')
            fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

            for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
                ax = axes[idx]
                for integration in present_integrations:
                    sli = integ_dict[integration]
                    color = integration_color_map.get(integration, 'black')
                    ls = integration_linestyle(integration)
                    if sli['sharp']:
                        sd = sli['sharp']
                        exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                        err = np.abs(sd[var] - exact_interp)
                        ax.semilogy(sd['x'], err, linestyle=ls, color=color,
                                    linewidth=ERROR_SHARP_LINEWIDTH,
                                    label=f'{integration} - Sharp',
                                    alpha=interface_alpha(None, all_epsilons))
                    for eps_val in sorted(sli['epsilon'].keys()):
                        eps_data = sli['epsilon'][eps_val]
                        exact_interp = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                        err = np.abs(eps_data[var] - exact_interp)
                        ax.semilogy(eps_data['x'], err, linestyle=ls, color=color,
                                    linewidth=ERROR_EPSILON_LINEWIDTH,
                                    label=f'{integration} - $\\epsilon={eps_val}$',
                                    alpha=interface_alpha(eps_val, all_epsilons))
                ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
                ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                              fontsize=AXIS_LABEL_FONTSIZE)
                if idx == 0:
                    ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
                ax.grid(True, alpha=0.3, which='both')
                ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
                ax.tick_params(labelsize=TICK_FONTSIZE)
            plt.tight_layout()
            fname = os.path.join(out_dir,
                                 f'{case_name}_{solver}_{limiter}_integration_error')
            save_and_show_plot(fname, PLOT_TO_SCREEN)
            print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (F) NEW: Per-integration solver+limiter overview (sharp only)
#     ./Images/{case}/integrations/{case}_{Integration}_overview_*
# ----------------------------------------------------------------------------

def generate_per_integration_overview_plots(case_name, all_data, exact_data,
                                            sim_time, solver_color_map,
                                            ordered_limiters, base_out_dir):
    out_dir = os.path.join(base_out_dir, 'integrations')
    os.makedirs(out_dir, exist_ok=True)

    # Discover integrations present
    integ_set = set()
    for solver, lim_dict in all_data.items():
        for limiter, integ_dict in lim_dict.items():
            integ_set.update(integ_dict.keys())
    ordered_integrations = sort_integrations(integ_set)

    for integration in ordered_integrations:
        # Collect (solver, limiter) sharp pairs for this integration
        pairs = []
        for solver in sorted(all_data.keys()):
            for limiter in sort_limiters(set(all_data[solver].keys())):
                sli = all_data[solver][limiter].get(integration)
                if sli and sli['sharp']:
                    pairs.append((solver, limiter, sli['sharp']))
        if not pairs:
            continue

        # Comparison
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        title = (f'{format_label(case_name)} - {integration} - '
                 f'Solver x Limiter Overview (Sharp Interface)')
        if sim_time is not None:
            title += f' at t={sim_time:.6f}s'
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            _plot_exact(ax, exact_data, var, AGGREGATED_EXACT_LINEWIDTH)
            for solver, limiter, sd in pairs:
                color = solver_color_map[solver]
                ls = limiter_cycle_linestyle(limiter, ordered_limiters)
                ax.plot(sd['x'], sd[var], linestyle=ls, color=color,
                        linewidth=AGGREGATED_SHARP_LINEWIDTH,
                        label=f'{format_label(solver)} / {limiter}',
                        alpha=0.85)
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{integration}_overview_comparison')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")

        # Error
        fig, axes = plt.subplots(3, 1, figsize=(14, 16))
        title = (f'{format_label(case_name)} - {integration} - '
                 f'Solver x Limiter Overview Error (Sharp Interface)')
        fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')

        for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
            ax = axes[idx]
            for solver, limiter, sd in pairs:
                color = solver_color_map[solver]
                ls = limiter_cycle_linestyle(limiter, ordered_limiters)
                exact_interp = np.interp(sd['x'], exact_data['x'], exact_data[var])
                err = np.abs(sd[var] - exact_interp)
                ax.semilogy(sd['x'], err, linestyle=ls, color=color,
                            linewidth=ERROR_SHARP_LINEWIDTH,
                            label=f'{format_label(solver)} / {limiter}',
                            alpha=0.85)
            ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
            ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                          fontsize=AXIS_LABEL_FONTSIZE)
            if idx == 0:
                ax.legend(fontsize=LEGEND_FONTSIZE - 1, loc='best', ncol=2)
            ax.grid(True, alpha=0.3, which='both')
            ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
            ax.tick_params(labelsize=TICK_FONTSIZE)
        plt.tight_layout()
        fname = os.path.join(out_dir, f'{case_name}_{integration}_overview_error')
        save_and_show_plot(fname, PLOT_TO_SCREEN)
        print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# (G) NEW: Mega aggregated — every (solver, limiter, integration, interface)
#     ./Images/{case}/{case}_mega_aggregated_*
# ----------------------------------------------------------------------------

def generate_mega_aggregated_plots(case_name, all_data, exact_data, sim_time,
                                   solver_color_map, integration_color_map,
                                   ordered_limiters, base_out_dir):
    # Global epsilon set for alpha mapping
    all_epsilons = set()
    for solver, lim_dict in all_data.items():
        for limiter, integ_dict in lim_dict.items():
            for integration, sli in integ_dict.items():
                for eps in sli['epsilon'].keys():
                    all_epsilons.add(eps)
    all_epsilons = sorted(all_epsilons)

    # Color per integration, linestyle per limiter, alpha per interface.
    def _draw(ax, var, semilog):
        if not semilog:
            _plot_exact(ax, exact_data, var, GRAND_EXACT_LINEWIDTH)
        for solver in sorted(all_data.keys()):
            for limiter in sort_limiters(set(all_data[solver].keys())):
                ls = limiter_cycle_linestyle(limiter, ordered_limiters)
                for integration in sort_integrations(set(all_data[solver][limiter].keys())):
                    sli = all_data[solver][limiter][integration]
                    color = integration_color_map.get(integration, 'black')
                    if sli['sharp']:
                        sd = sli['sharp']
                        a = interface_alpha(None, all_epsilons)
                        if semilog:
                            ei = np.interp(sd['x'], exact_data['x'], exact_data[var])
                            ax.semilogy(sd['x'], np.abs(sd[var] - ei),
                                        linestyle=ls, color=color,
                                        linewidth=GRAND_SHARP_LINEWIDTH, alpha=a,
                                        label=f'{format_label(solver)} / {limiter} / '
                                              f'{integration} / Sharp')
                        else:
                            ax.plot(sd['x'], sd[var], linestyle=ls, color=color,
                                    linewidth=GRAND_SHARP_LINEWIDTH, alpha=a,
                                    label=f'{format_label(solver)} / {limiter} / '
                                          f'{integration} / Sharp')
                    for eps_val in sorted(sli['epsilon'].keys()):
                        eps_data = sli['epsilon'][eps_val]
                        a = interface_alpha(eps_val, all_epsilons)
                        if semilog:
                            ei = np.interp(eps_data['x'], exact_data['x'], exact_data[var])
                            ax.semilogy(eps_data['x'], np.abs(eps_data[var] - ei),
                                        linestyle=ls, color=color,
                                        linewidth=GRAND_EPSILON_LINEWIDTH, alpha=a,
                                        label=f'{format_label(solver)} / {limiter} / '
                                              f'{integration} / $\\epsilon={eps_val}$')
                        else:
                            ax.plot(eps_data['x'], eps_data[var], linestyle=ls, color=color,
                                    linewidth=GRAND_EPSILON_LINEWIDTH, alpha=a,
                                    label=f'{format_label(solver)} / {limiter} / '
                                          f'{integration} / $\\epsilon={eps_val}$')

    # Comparison
    fig, axes = plt.subplots(3, 1, figsize=(18, 20))
    title = (f'{format_label(case_name)} - Mega Aggregated Comparison '
             f'(solver x limiter x integration x interface)')
    if sim_time is not None:
        title += f' at t={sim_time:.6f}s'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        _draw(ax, var, semilog=False)
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=max(5, LEGEND_FONTSIZE - 4), loc='best', ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(base_out_dir, f'{case_name}_mega_aggregated_comparison')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")

    # Error
    fig, axes = plt.subplots(3, 1, figsize=(18, 20))
    title = f'{format_label(case_name)} - Mega Aggregated Error Analysis'
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight='bold')
    for idx, (var, ylabel) in enumerate(zip(VARIABLES, YLABELS)):
        ax = axes[idx]
        _draw(ax, var, semilog=True)
        ax.set_xlabel('Position (x)', fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(f'{ylabel.split("(")[0].strip()} Error',
                      fontsize=AXIS_LABEL_FONTSIZE)
        if idx == 0:
            ax.legend(fontsize=max(5, LEGEND_FONTSIZE - 4), loc='best', ncol=4)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([exact_data['x'][0], exact_data['x'][-1]])
        ax.tick_params(labelsize=TICK_FONTSIZE)
    plt.tight_layout()
    fname = os.path.join(base_out_dir, f'{case_name}_mega_aggregated_error')
    save_and_show_plot(fname, PLOT_TO_SCREEN)
    print(f"  Saved: {fname}")


# ----------------------------------------------------------------------------
# Error reporting + CSV dump
# ----------------------------------------------------------------------------

def report_and_dump_errors(case_name, all_data, exact_data, base_out_dir):
    print(f"\n{'=' * 80}")
    print("ERROR METRICS SUMMARY  (solver / limiter / integration / interface)")
    print("=" * 80)

    csv_path = os.path.join(base_out_dir, f'{case_name}_errors.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['solver', 'limiter', 'integration', 'interface_type', 'epsilon',
                         'velocity_l2', 'velocity_linf',
                         'pressure_l2', 'pressure_linf',
                         'density_l2',  'density_linf'])

        for solver in sorted(all_data.keys()):
            for limiter in sort_limiters(set(all_data[solver].keys())):
                for integration in sort_integrations(set(all_data[solver][limiter].keys())):
                    sli = all_data[solver][limiter][integration]
                    print(f"\n{solver}  /  {limiter}  /  {integration}")
                    print("-" * 60)

                    if sli['sharp']:
                        errs = calculate_errors(sli['sharp'], exact_data)
                        print(f"  Sharp Interface:")
                        print(f"    Density  - L2: {errs['density_l2']:.6e}, Linf: {errs['density_linf']:.6e}")
                        print(f"    Velocity - L2: {errs['velocity_l2']:.6e}, Linf: {errs['velocity_linf']:.6e}")
                        print(f"    Pressure - L2: {errs['pressure_l2']:.6e}, Linf: {errs['pressure_linf']:.6e}")
                        writer.writerow([solver, limiter, integration, 'Sharp_Interface', '',
                                         errs['velocity_l2'], errs['velocity_linf'],
                                         errs['pressure_l2'], errs['pressure_linf'],
                                         errs['density_l2'],  errs['density_linf']])

                    for eps_val in sorted(sli['epsilon'].keys()):
                        errs = calculate_errors(sli['epsilon'][eps_val], exact_data)
                        print(f"  epsilon = {eps_val}:")
                        print(f"    Density  - L2: {errs['density_l2']:.6e}, Linf: {errs['density_linf']:.6e}")
                        print(f"    Velocity - L2: {errs['velocity_l2']:.6e}, Linf: {errs['velocity_linf']:.6e}")
                        print(f"    Pressure - L2: {errs['pressure_l2']:.6e}, Linf: {errs['pressure_linf']:.6e}")
                        writer.writerow([solver, limiter, integration, f'epsilon_{eps_val}', eps_val,
                                         errs['velocity_l2'], errs['velocity_linf'],
                                         errs['pressure_l2'], errs['pressure_linf'],
                                         errs['density_l2'],  errs['density_linf']])

    print(f"\nWrote error CSV: {csv_path}")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("=" * 80)
print("RIEMANN SOLVER + LIMITER COMPARISON ANALYSIS")
print("=" * 80)
print(f"\nTest Case:      {case_name}")
print(f"Base Directory: {base_output_dir}")
print(f"Plot to Screen: {PLOT_TO_SCREEN}")

# Load exact solution
group_name = case_to_group.get(case_name, case_name)
print(f"\nLoading exact solution from: {hdf5_file}")
print(f"Using group: {group_name}")
exact_data = load_exact_solution(hdf5_file, group_name)
print(f"Exact solution loaded: {len(exact_data['x'])} points")

# Scan + load
print(f"\nScanning for output directories...")
all_data, all_stop_times = collect_all_data(case_name, base_output_dir)

if not all_data:
    print("\nERROR: No matching output directories found!")
    print(f"Expected pattern: output_{case_name}_{{Solver}}_{{Limiter}}_{{Integration}}_Sharp_Interface")
    print(f"                  output_{case_name}_{{Solver}}_{{Limiter}}_{{Integration}}_epsilon_{{value}}")
    exit(1)

# Filter on max stop time per (solver, limiter, integration) tuple
if all_stop_times:
    max_stop_time = max(all_stop_times)
    print(f"\n{'=' * 80}")
    print(f"Maximum stop time found: {max_stop_time:.6e}s")
    print(f"Filtering out (solver, limiter, integration) combos that stopped early ...")

    all_data = filter_by_stop_time(all_data, max_stop_time)

    if not all_data:
        print("\nERROR: No simulations reached the maximum stop time!")
        exit(1)
else:
    max_stop_time = None

print(f"\n{'=' * 80}")
print(f"Found {len(all_data)} solver(s) with complete simulations")
all_limiter_set = set()
all_integration_set = set()
for solver in sorted(all_data.keys()):
    lims = sort_limiters(set(all_data[solver].keys()))
    print(f"  {solver}: limiters = {lims}")
    for lim in lims:
        integ_dict = all_data[solver][lim]
        all_limiter_set.add(lim)
        for integ in sort_integrations(set(integ_dict.keys())):
            sli = integ_dict[integ]
            sharp_status = "YES" if sli['sharp'] else "NO"
            eps_count = len(sli['epsilon'])
            print(f"      [{lim} / {integ}]  Sharp={sharp_status},  epsilon variants={eps_count}")
            all_integration_set.add(integ)

# Output directory
out_dir = f"./Images/{case_name}"
os.makedirs(out_dir, exist_ok=True)

sim_time = max_stop_time

# Color palettes
riemann_solvers = sorted(all_data.keys())
solver_color_map = generate_color_palette(riemann_solvers)

# Sort limiters / integrations by canonical order, extras alphabetically
all_limiters_sorted     = sort_limiters(all_limiter_set)
all_integrations_sorted = sort_integrations(all_integration_set)

limiter_color_map     = generate_limiter_color_map(all_limiters_sorted)
integration_color_map = generate_integration_color_map(all_integrations_sorted)

# ----------------------------------------------------------------------------
# (A,B,C) Existing limiter/solver-axis plots — generated PER integration
# ----------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print("GENERATING LEGACY-STYLE PLOTS PER INTEGRATION x LIMITER")
print("=" * 80)
for integration in all_integrations_sorted:
    sliced = slice_by_integration(all_data, integration)
    if not sliced:
        continue
    integ_out_dir = os.path.join(out_dir, integration)
    os.makedirs(integ_out_dir, exist_ok=True)

    print(f"\n=== Integration: {integration} ===")

    # (A) Legacy plots per limiter
    for limiter in all_limiters_sorted:
        if not any(limiter in sliced[s] for s in sliced):
            continue
        print(f"\n--- [{integration}] Limiter: {limiter} ---")
        generate_legacy_plots_for_limiter(case_name, limiter, sliced, exact_data,
                                          sim_time, solver_color_map, integ_out_dir)

    # (B) Per-solver limiter comparison
    print(f"\n[{integration}] Per-solver limiter comparison ...")
    generate_per_solver_limiter_plots(case_name, sliced, exact_data, sim_time,
                                      limiter_color_map, integ_out_dir)

    # (C) Per-limiter solver comparison
    print(f"\n[{integration}] Per-limiter solver comparison ...")
    integration_limiters = sort_limiters({lim for s in sliced for lim in sliced[s]})
    generate_per_limiter_solver_plots(case_name, sliced, exact_data, sim_time,
                                      solver_color_map, integ_out_dir,
                                      integration_limiters)

# ----------------------------------------------------------------------------
# (D) Legacy grand aggregated — fixed at ForwardEuler for backward compatibility
# ----------------------------------------------------------------------------
if INTEGRATION_FALLBACK in all_integrations_sorted:
    print(f"\n{'=' * 80}")
    print(f"GENERATING GRAND AGGREGATED PLOT ({INTEGRATION_FALLBACK} only)")
    print("=" * 80)
    fe_slice = slice_by_integration(all_data, INTEGRATION_FALLBACK)
    if fe_slice:
        generate_grand_aggregated_plots(case_name, fe_slice, exact_data, sim_time,
                                        solver_color_map, out_dir)

# ----------------------------------------------------------------------------
# (E) Per-(solver, limiter) integration comparison
# ----------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print("GENERATING PER-(SOLVER, LIMITER) INTEGRATION COMPARISON PLOTS")
print("=" * 80)
generate_per_solver_limiter_integration_plots(case_name, all_data, exact_data,
                                              sim_time, integration_color_map,
                                              out_dir)

# ----------------------------------------------------------------------------
# (F) Per-integration solver+limiter overview (sharp only)
# ----------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print("GENERATING PER-INTEGRATION SOLVER+LIMITER OVERVIEW PLOTS")
print("=" * 80)
generate_per_integration_overview_plots(case_name, all_data, exact_data, sim_time,
                                        solver_color_map, all_limiters_sorted,
                                        out_dir)

# ----------------------------------------------------------------------------
# (G) Mega aggregated (full cross product)
# ----------------------------------------------------------------------------
print(f"\n{'=' * 80}")
print("GENERATING MEGA AGGREGATED PLOT (solver x limiter x integration x interface)")
print("=" * 80)
generate_mega_aggregated_plots(case_name, all_data, exact_data, sim_time,
                               solver_color_map, integration_color_map,
                               all_limiters_sorted, out_dir)

# ----------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------
report_and_dump_errors(case_name, all_data, exact_data, out_dir)

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE!")
print(f"All plots saved under: {out_dir}")
print("=" * 80)
