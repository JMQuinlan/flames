"""
2D Eta Field Visualization with Complete Cartesian Mesh Overlay
Shows all individual cell lines to visualize AMR mesh refinement

CONFIGURATION (modify these as needed):
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

# Suppress yt verbose output
yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION - MODIFY THESE PARAMETERS
# ============================================================================

# Output directory
amrex_output_dir = r'../../../bin/tests/FlowMiscPlots/output_AMReX_Example'
case_name = 'Eta_Field_Mesh'

# Plot 1: Full view bounds
x_min_full = -0.05
x_max_full = 0.05
y_min_full = -0.05
y_max_full = 0.05

# Plot 2: Zoomed view bounds
x_min_zoom = 0.011
x_max_zoom = 0.017
y_min_zoom = 0.011
y_max_zoom = 0.017

# Font sizes
title_font_size = 20
axis_font_size = 16
eta_font_size = 12
zoom_font_size = 11

# Line widths
mesh_line_width_full = 0.5
mesh_line_width_zoom = 1.0
zoom_box_width = 4.0

# Grid transparency
mesh_alpha_full = 0.6
mesh_alpha_zoom = 0.8

# Plot titles
title_full = 'Eta Field with Complete Cartesian Mesh (Full View)'
title_zoom = 'Eta Field with Complete Cartesian Mesh (Zoomed View)'
title_combined = 'Adaptive Mesh Refinement'
title_full_combined = 'Full View'
title_zoom_combined = 'Zoomed View'

# ============================================================================
# END OF USER CONFIGURATION
# ============================================================================

print("="*70)
print("2D ETA FIELD VISUALIZATION WITH COMPLETE CARTESIAN MESH")
print("="*70)
print("\nPlot Configuration:")
print(f"  Full view: x=[{x_min_full}, {x_max_full}], y=[{y_min_full}, {y_max_full}]")
print(f"  Zoom view: x=[{x_min_zoom}, {x_max_zoom}], y=[{y_min_zoom}, {y_max_zoom}]")

# ============================================================================
# LOAD AMREX DATA
# ============================================================================

print("\n" + "="*70)
print("LOADING AMREX OUTPUT")
print("="*70)

# Find plot files
plot_files = []
for item in os.listdir(amrex_output_dir):
    item_path = os.path.join(amrex_output_dir, item)
    if os.path.isdir(item_path) and item.endswith('cell'):
        plot_files.append(item_path)

if not plot_files:
    print(f"ERROR: No *cell directories found in {amrex_output_dir}")
    exit(1)

plot_files.sort()
last_plot = plot_files[-1]

print(f"\nFound {len(plot_files)} plot files")
print(f"Using LAST timestep: {os.path.basename(last_plot)}")

# Load dataset
ds = yt.load(last_plot)
sim_time = float(ds.current_time)
print(f"Simulation time: {sim_time:.6f} s")
print(f"Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")

# Check for eta field
if ('boxlib', 'eta') not in ds.field_list and ('amrex', 'eta') not in ds.field_list:
    print("\nERROR: 'eta' field not found in dataset")
    print("Available fields:")
    for field in ds.field_list:
        print(f"  {field}")
    exit(1)

print("\nEta field found successfully")

# ============================================================================
# FUNCTION TO EXTRACT ALL CARTESIAN CELL LINES
# ============================================================================

def get_all_cell_lines(ds, x_min, x_max, y_min, y_max):
    """Extract ALL individual cell lines (Cartesian grid) within specified bounds"""
    
    lines = []
    
    # Iterate through all grids at all levels
    for level in range(ds.max_level + 1):
        grids = [g for g in ds.index.grids if g.Level == level]
        
        for grid in grids:
            # Get grid boundaries
            left_edge = grid.LeftEdge.to_ndarray()
            right_edge = grid.RightEdge.to_ndarray()
            
            x_left = float(left_edge[0])
            x_right = float(right_edge[0])
            y_left = float(left_edge[1])
            y_right = float(right_edge[1])
            
            # Check if grid overlaps with plot region
            if x_right < x_min or x_left > x_max:
                continue
            if y_right < y_min or y_left > y_max:
                continue
            
            # Get number of cells in this grid
            dims = grid.ActiveDimensions
            nx = dims[0]
            ny = dims[1]
            
            # Calculate cell sizes
            dx = (x_right - x_left) / nx
            dy = (y_right - y_left) / ny
            
            # Generate vertical lines (constant x)
            for i in range(nx + 1):
                x_line = x_left + i * dx
                if x_line >= x_min and x_line <= x_max:
                    y_start = max(y_left, y_min)
                    y_end = min(y_right, y_max)
                    lines.append([(x_line, y_start), (x_line, y_end)])
            
            # Generate horizontal lines (constant y)
            for j in range(ny + 1):
                y_line = y_left + j * dy
                if y_line >= y_min and y_line <= y_max:
                    x_start = max(x_left, x_min)
                    x_end = min(x_right, x_max)
                    lines.append([(x_start, y_line), (x_end, y_line)])
    
    return lines

# ============================================================================
# CREATE PLOT 1: FULL VIEW
# ============================================================================

print("\n" + "="*70)
print("CREATING FULL VIEW PLOT")
print("="*70)

# Create slice at z=0
slc_full = ds.slice('z', 0.0)

# Create fixed resolution buffer for the full region
frb_full = slc_full.to_frb((x_max_full - x_min_full, 'code_length'), 
                           (512, 512),
                           center=[(x_min_full + x_max_full)/2, 
                                   (y_min_full + y_max_full)/2, 
                                   0.0],
                           height=(y_max_full - y_min_full, 'code_length'))

# Extract eta data
eta_full = np.array(frb_full['eta'])

# Get ALL cell lines
print("Extracting ALL cell lines for full view...")
mesh_lines_full = get_all_cell_lines(ds, x_min_full, x_max_full, y_min_full, y_max_full)
print(f"  Found {len(mesh_lines_full)} cell lines")

# Create figure
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10))

# Plot eta field with custom colormap (grey to white)
from matplotlib.colors import LinearSegmentedColormap
colors = ['grey', 'white']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('eta_cmap', colors, N=n_bins)

im1 = ax1.imshow(eta_full, 
                 extent=[x_min_full, x_max_full, y_min_full, y_max_full],
                 origin='lower',
                 cmap=cmap,
                 vmin=0.0,
                 vmax=1.0,
                 interpolation='nearest')

# Add colorbar using axes divider to match plot height
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.ax.tick_params(labelsize=eta_font_size)
cbar1.set_label('$\eta$', fontsize=axis_font_size)

# Overlay ALL mesh cell lines
if mesh_lines_full:
    lc1 = LineCollection(mesh_lines_full, colors='black', 
                         linewidths=mesh_line_width_full, alpha=mesh_alpha_full)
    ax1.add_collection(lc1)

# Add rectangle showing zoom region
zoom_rect = patches.Rectangle((x_min_zoom, y_min_zoom), 
                              x_max_zoom - x_min_zoom, 
                              y_max_zoom - y_min_zoom,
                              linewidth=zoom_box_width, 
                              edgecolor='red', 
                              facecolor='none',
                              linestyle='--',
                              label='Zoom Region')
ax1.add_patch(zoom_rect)

ax1.set_xlabel('X (m)', fontsize=axis_font_size)
ax1.set_ylabel('Y (m)', fontsize=axis_font_size)
ax1.set_title(f'{title_full}', 
              fontsize=title_font_size, fontweight='bold')
ax1.set_xlim([x_min_full, x_max_full])
ax1.set_ylim([y_min_full, y_max_full])
ax1.set_aspect('equal')
ax1.legend(loc='upper right', fontsize=zoom_font_size)
ax1.grid(False)
ax1.tick_params(labelsize=axis_font_size)

plt.tight_layout()

# Save plot
os.makedirs('./Images', exist_ok=True)
output_file = f'./Images/{case_name}_full_view'
fig1.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig1.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"\nSaved: {output_file}.png and .eps")
plt.close(fig1)

# ============================================================================
# CREATE PLOT 2: ZOOMED VIEW
# ============================================================================

print("\n" + "="*70)
print("CREATING ZOOMED VIEW PLOT")
print("="*70)

# Create fixed resolution buffer for the zoomed region
frb_zoom = slc_full.to_frb((x_max_zoom - x_min_zoom, 'code_length'), 
                           (512, 512),
                           center=[(x_min_zoom + x_max_zoom)/2, 
                                   (y_min_zoom + y_max_zoom)/2, 
                                   0.0],
                           height=(y_max_zoom - y_min_zoom, 'code_length'))

# Extract eta data
eta_zoom = np.array(frb_zoom['eta'])

# Get ALL cell lines for zoomed region
print("Extracting ALL cell lines for zoomed view...")
mesh_lines_zoom = get_all_cell_lines(ds, x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom)
print(f"  Found {len(mesh_lines_zoom)} cell lines")

# Create figure
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10))

# Plot eta field
im2 = ax2.imshow(eta_zoom, 
                 extent=[x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom],
                 origin='lower',
                 cmap=cmap,
                 vmin=0.0,
                 vmax=1.0,
                 interpolation='nearest')

# Add colorbar using axes divider to match plot height
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
cbar2 = plt.colorbar(im2, cax=cax2)
cbar2.ax.tick_params(labelsize=eta_font_size)
cbar2.set_label('$\eta$', fontsize=axis_font_size)

# Overlay ALL mesh cell lines
if mesh_lines_zoom:
    lc2 = LineCollection(mesh_lines_zoom, colors='black', 
                         linewidths=mesh_line_width_zoom, alpha=mesh_alpha_zoom)
    ax2.add_collection(lc2)

ax2.set_xlabel('X (m)', fontsize=axis_font_size)
ax2.set_ylabel('Y (m)', fontsize=axis_font_size)
ax2.set_title(f'{title_zoom}', 
              fontsize=title_font_size, fontweight='bold')
ax2.set_xlim([x_min_zoom, x_max_zoom])
ax2.set_ylim([y_min_zoom, y_max_zoom])
ax2.set_aspect('equal')
ax2.grid(False)
ax2.tick_params(labelsize=axis_font_size)

plt.tight_layout()

# Save plot
output_file = f'./Images/{case_name}_zoomed_view'
fig2.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig2.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"\nSaved: {output_file}.png and .eps")
plt.close(fig2)

# ============================================================================
# CREATE COMBINED PLOT
# ============================================================================

print("\n" + "="*70)
print("CREATING COMBINED PLOT")
print("="*70)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(18, 8))

# Left: Full view
im3a = ax3a.imshow(eta_full, 
                   extent=[x_min_full, x_max_full, y_min_full, y_max_full],
                   origin='lower',
                   cmap=cmap,
                   vmin=0.0,
                   vmax=1.0,
                   interpolation='nearest')

if mesh_lines_full:
    lc3a = LineCollection(mesh_lines_full, colors='black', 
                          linewidths=mesh_line_width_full, alpha=mesh_alpha_full)
    ax3a.add_collection(lc3a)

zoom_rect3 = patches.Rectangle((x_min_zoom, y_min_zoom), 
                               x_max_zoom - x_min_zoom, 
                               y_max_zoom - y_min_zoom,
                               linewidth=zoom_box_width, 
                               edgecolor='red', 
                               facecolor='none',
                               linestyle='--',
                               label='Zoom Region')
ax3a.add_patch(zoom_rect3)

ax3a.set_xlabel('X (m)', fontsize=axis_font_size)
ax3a.set_ylabel('Y (m)', fontsize=axis_font_size)
ax3a.set_title(title_full_combined, fontsize=axis_font_size, fontweight='bold')
ax3a.set_xlim([x_min_full, x_max_full])
ax3a.set_ylim([y_min_full, y_max_full])
ax3a.set_aspect('equal')
ax3a.legend(loc='upper right', fontsize=zoom_font_size)
ax3a.grid(False)
ax3a.tick_params(labelsize=axis_font_size)

# Right: Zoomed view
im3b = ax3b.imshow(eta_zoom, 
                   extent=[x_min_zoom, x_max_zoom, y_min_zoom, y_max_zoom],
                   origin='lower',
                   cmap=cmap,
                   vmin=0.0,
                   vmax=1.0,
                   interpolation='nearest')

if mesh_lines_zoom:
    lc3b = LineCollection(mesh_lines_zoom, colors='black', 
                          linewidths=mesh_line_width_zoom, alpha=mesh_alpha_zoom)
    ax3b.add_collection(lc3b)

ax3b.set_xlabel('X (m)', fontsize=axis_font_size)
ax3b.set_ylabel('Y (m)', fontsize=axis_font_size)
ax3b.set_title(title_zoom_combined, fontsize=axis_font_size, fontweight='bold')
ax3b.set_xlim([x_min_zoom, x_max_zoom])
ax3b.set_ylim([y_min_zoom, y_max_zoom])
ax3b.set_aspect('equal')
ax3b.grid(False)
ax3b.tick_params(labelsize=axis_font_size)

# Add main title
fig3.suptitle(f'{title_combined}', 
              fontsize=title_font_size, fontweight='bold')

# Add shared colorbar at the BOTTOM
fig3.subplots_adjust(bottom=0.15)  # Make room for colorbar
cbar_ax = fig3.add_axes([0.15, 0.05, 0.7, 0.03])  # [left, bottom, width, height]
cbar3 = fig3.colorbar(im3b, cax=cbar_ax, orientation='horizontal')
cbar3.set_label('$\eta$', fontsize=axis_font_size)
cbar3.ax.tick_params(labelsize=eta_font_size)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Adjust layout to accommodate suptitle and colorbar

# Save plot
output_file = f'./Images/{case_name}_combined'
fig3.savefig(output_file + '.png', format='png', dpi=300, bbox_inches='tight')
fig3.savefig(output_file + '.eps', format='eps', bbox_inches='tight')
print(f"\nSaved: {output_file}.png and .eps")
plt.close(fig3)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nEta Field Statistics:")
print(f"  Full view - Min: {np.min(eta_full):.6f}, Max: {np.max(eta_full):.6f}")
print(f"  Zoom view - Min: {np.min(eta_zoom):.6f}, Max: {np.max(eta_zoom):.6f}")

print(f"\nMesh Statistics:")
print(f"  Maximum AMR level: {ds.max_level}")
print(f"  Total number of grids: {len(ds.index.grids)}")
print(f"  Total cell lines in full view: {len(mesh_lines_full)}")
print(f"  Total cell lines in zoom view: {len(mesh_lines_zoom)}")

# Count grids and cells at each level
total_cells = 0
for level in range(ds.max_level + 1):
    grids_at_level = [g for g in ds.index.grids if g.Level == level]
    cells_at_level = sum([g.ActiveDimensions[0] * g.ActiveDimensions[1] for g in grids_at_level])
    total_cells += cells_at_level
    print(f"  Level {level}: {len(grids_at_level)} grids, {cells_at_level} cells")

print(f"  Total cells in domain: {total_cells}")

print("\n" + "="*70)
print("FILES SAVED")
print("="*70)
print(f"\nAll plots saved to ./Images/ directory:")
print(f"  - {case_name}_full_view.png/.eps")
print(f"  - {case_name}_zoomed_view.png/.eps")
print(f"  - {case_name}_combined.png/.eps")

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nNotes:")
print("  - Eta colormap: grey (eta=0) to white (eta=1)")
print("  - ALL individual cell lines shown in black (Cartesian grid)")
print("  - Red dashed box in full view indicates zoom region")
print("  - Zoomed view clearly shows AMR mesh refinement with all cell boundaries")
print("  - Colorbars on individual plots match the height of the plot axes")
