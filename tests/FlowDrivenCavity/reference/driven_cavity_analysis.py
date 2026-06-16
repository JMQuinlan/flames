"""
Lid-Driven Cavity analysis (transient flow visualization).

Reference for input_hydro2: square cavity with a top-wall lid moving in +x,
all other walls no-slip and stationary.  This script produces:

  - Per-frame figures with pressure field as the colored background and
    velocity streamlines overlaid.
  - A GIF stitched from all the per-frame figures.
  - An optional centerline-velocity-profile plot (u_x along the vertical
    centerline, u_y along the horizontal centerline) at the final time --
    classical Ghia-style diagnostic.

CONFIGURATION lives in the block below the imports.
"""

import yt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

yt.funcs.mylog.setLevel(40)

# ============================================================================
# USER CONFIGURATION
# ============================================================================

case_name        = 'DrivenCavity'
amrex_output_dir = r'..\..\..\bin\tests\FlowDrivenCavity\output'

# Geometry (must match input_hydro2)
x_min, x_max = -12.0, 12.0
y_min, y_max = -12.0, 12.0
L = x_max - x_min                  # cavity side length

# Sampling resolution for the AMR -> uniform-grid covering_grid.
# input_hydro2 has 32x32 base + max_level=2 => effective finest 128x128.
# Pick a sampling resolution at or above the finest mesh so we don't lose
# features inside refined patches.
NX_SAMPLE = 192
NY_SAMPLE = 192

# Streamline appearance
STREAM_DENSITY = 1.6               # spacing between seed lines (1.0 sparse, 3.0 dense)
STREAM_LW_MIN  = 0.4               # min line width
STREAM_LW_SCALE = 2.5              # |u|-driven scale factor for line widths
STREAM_COLOR   = 'white'           # over the pressure colormap

# Pressure colormap
PRESSURE_CMAP  = 'viridis'

# Plot styling
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK  = 10
DPI             = 200
FIG_INCHES      = 7.0              # square figure (cavity is square)

# Frame outputs
images_dir   = './Images'
frames_dir   = os.path.join(images_dir, 'Frames')
SAVE_FRAMES  = True                # individual PNG + EPS per plotfile
BUILD_GIF    = True                # animated GIF stitched from frames
GIF_FPS      = 6
PLOT_CENTERLINE_PROFILES = True    # final-frame Ghia-style centerline cuts

os.makedirs(images_dir, exist_ok=True)
if SAVE_FRAMES:
    os.makedirs(frames_dir, exist_ok=True)


# ============================================================================
# DATA-EXTRACTION HELPERS
# ============================================================================

def _load_plotfiles(amrex_dir):
    """Return a sorted list of *cell plotfile paths inside amrex_dir."""
    if not os.path.isdir(amrex_dir):
        raise SystemExit(f"  ERROR: output directory does not exist:\n    {amrex_dir}")
    plotfiles = sorted(
        os.path.join(amrex_dir, d)
        for d in os.listdir(amrex_dir)
        if os.path.isdir(os.path.join(amrex_dir, d)) and d.endswith('cell')
    )
    if not plotfiles:
        raise SystemExit(f"  ERROR: no *cell directories found in:\n    {amrex_dir}")
    return plotfiles


def _frb_fields(ds):
    """Build a fixed-resolution buffer covering the full domain and return
    (xg, yg, pressure, vx, vy) on a NY_SAMPLE x NX_SAMPLE grid.
    """
    slc    = ds.slice('z', 0.0)
    domain_w = x_max - x_min
    domain_h = y_max - y_min
    frb = slc.to_frb(
        (domain_w, 'code_length'),
        (NY_SAMPLE, NX_SAMPLE),                            # (height, width)
        center=[0.5 * (x_min + x_max), 0.5 * (y_min + y_max), 0.0],
        height=(domain_h, 'code_length'),
    )
    pressure = np.array(frb['pressure'])
    vx       = np.array(frb['velocityx'])
    vy       = np.array(frb['velocityy'])
    x1d = np.linspace(x_min, x_max, NX_SAMPLE)
    y1d = np.linspace(y_min, y_max, NY_SAMPLE)
    xg, yg = np.meshgrid(x1d, y1d)
    return xg, yg, pressure, vx, vy


# ============================================================================
# PER-FRAME PLOT
# ============================================================================

def plot_frame(xg, yg, pressure, vx, vy, sim_time,
               p_vmin, p_vmax,
               save_basename=None,
               ax=None):
    """Draw pressure-background + streamline-overlay for one frame.

    If `save_basename` is given, save PNG + EPS.  If `ax` is given, draw
    into it instead of creating a new figure.
    Returns the (fig, ax, im, stream) tuple for further customization.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))
    else:
        fig = ax.figure

    # Pressure background.
    im = ax.imshow(
        pressure,
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap=PRESSURE_CMAP,
        vmin=p_vmin, vmax=p_vmax,
        aspect='equal',
        interpolation='bilinear',
    )

    # Streamlines.  Linewidth scales with local |u|, clamped to a sensible range.
    speed = np.sqrt(vx ** 2 + vy ** 2)
    smax  = max(np.nanmax(speed), 1e-30)
    lw    = STREAM_LW_MIN + STREAM_LW_SCALE * (speed / smax)
    stream = ax.streamplot(
        xg, yg, vx, vy,
        color=STREAM_COLOR,
        linewidth=lw,
        density=STREAM_DENSITY,
        arrowsize=0.9,
    )

    # Axes labelling.
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('x (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel('y (m)', fontsize=FONT_SIZE_LABEL)
    ax.set_title('Lid-driven cavity', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)

    # Floating time label (not in title).
    ax.text(
        0.04, 0.96,
        f't = {sim_time:.3f} s',
        transform=ax.transAxes,
        fontsize=FONT_SIZE_LABEL,
        fontweight='bold',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.4'),
    )

    # Colorbar on the right.
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
    cb.ax.tick_params(labelsize=FONT_SIZE_TICK)

    if save_basename:
        fig.tight_layout()
        fig.savefig(save_basename + '.png', dpi=DPI, bbox_inches='tight')
        fig.savefig(save_basename + '.eps',           bbox_inches='tight')

    return fig, ax, im, stream


# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("LID-DRIVEN CAVITY ANALYSIS  (input_hydro2)")
print("=" * 70)
print(f"\n  output dir : {amrex_output_dir}")

plotfiles = _load_plotfiles(amrex_output_dir)
print(f"  found      : {len(plotfiles)} plotfiles")

# -------------------------------------------------------------------- pass 1
# Walk every plotfile once to figure out the global pressure colorbar range
# and cache fields on the way (avoids loading each yt dataset twice).
print("\n  pre-pass: extracting fields + global p range...")
cache = []
for i, pf in enumerate(plotfiles):
    try:
        ds = yt.load(pf)
        xg, yg, p, vx, vy = _frb_fields(ds)
        cache.append({
            't' : float(ds.current_time),
            'xg': xg, 'yg': yg,
            'p' : p,  'vx': vx, 'vy': vy,
            'pf_name': os.path.basename(pf),
        })
    except Exception as exc:
        print(f"    [skip] {os.path.basename(pf)}: {exc}")
        continue
if not cache:
    raise SystemExit("  ERROR: no usable plotfiles loaded.")

# Sort by physical time (filenames are usually monotonic, but be safe).
cache.sort(key=lambda d: d['t'])

p_vmin = float(min(np.nanmin(d['p']) for d in cache))
p_vmax = float(max(np.nanmax(d['p']) for d in cache))
# If the test starts with a uniform pressure field, p_vmin == p_vmax;
# pad by 0.5% to avoid a singular color map.
if not (p_vmax > p_vmin):
    pad = max(abs(p_vmax), 1.0) * 0.005
    p_vmin -= pad
    p_vmax += pad
print(f"  global p range : [{p_vmin:.4e}, {p_vmax:.4e}] Pa")

# -------------------------------------------------------------------- frames
if SAVE_FRAMES:
    print("\n  writing per-frame PNG + EPS...")
    for i, d in enumerate(cache):
        base = os.path.join(frames_dir, f'{case_name}_frame_{i:04d}')
        fig, ax, im, stream = plot_frame(
            d['xg'], d['yg'], d['p'], d['vx'], d['vy'], d['t'],
            p_vmin, p_vmax,
            save_basename=base,
        )
        plt.close(fig)
        if (i + 1) % 10 == 0 or i == len(cache) - 1:
            print(f"    {i+1:4d}/{len(cache)}")
    print(f"  frames saved to: {frames_dir}")

# -------------------------------------------------------------------- GIF
if BUILD_GIF:
    print("\n  building transient GIF...")
    fig_g, ax_g = plt.subplots(figsize=(FIG_INCHES, FIG_INCHES))

    # First frame (used to seed the imshow + cbar; streamplot recreated each frame
    # because matplotlib's streamplot is not animation-friendly out of the box).
    d0 = cache[0]
    im_g = ax_g.imshow(
        d0['p'],
        origin='lower',
        extent=[x_min, x_max, y_min, y_max],
        cmap=PRESSURE_CMAP,
        vmin=p_vmin, vmax=p_vmax,
        aspect='equal',
        interpolation='bilinear',
    )
    ax_g.set_xlim([x_min, x_max])
    ax_g.set_ylim([y_min, y_max])
    ax_g.set_xlabel('x (m)', fontsize=FONT_SIZE_LABEL)
    ax_g.set_ylabel('y (m)', fontsize=FONT_SIZE_LABEL)
    ax_g.set_title('Lid-driven cavity', fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_g.tick_params(labelsize=FONT_SIZE_TICK)
    time_text_g = ax_g.text(
        0.04, 0.96, '', transform=ax_g.transAxes,
        fontsize=FONT_SIZE_LABEL, fontweight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='0.4'),
    )
    cb_g = fig_g.colorbar(im_g, ax=ax_g, fraction=0.046, pad=0.04)
    cb_g.set_label('Pressure (Pa)', fontsize=FONT_SIZE_LABEL)
    cb_g.ax.tick_params(labelsize=FONT_SIZE_TICK)

    def _gif_update(i):
        d = cache[i]
        im_g.set_data(d['p'])
        # Streamplot can't update in place; we strip the previous artist set
        # off the axes and redraw.
        for coll in list(ax_g.collections):
            coll.remove()
        for patch in list(ax_g.patches):
            patch.remove()
        speed = np.sqrt(d['vx'] ** 2 + d['vy'] ** 2)
        smax  = max(np.nanmax(speed), 1e-30)
        lw    = STREAM_LW_MIN + STREAM_LW_SCALE * (speed / smax)
        ax_g.streamplot(
            d['xg'], d['yg'], d['vx'], d['vy'],
            color=STREAM_COLOR,
            linewidth=lw,
            density=STREAM_DENSITY,
            arrowsize=0.9,
        )
        time_text_g.set_text(f't = {d["t"]:.3f} s')
        return [im_g, time_text_g]

    anim = animation.FuncAnimation(
        fig_g, _gif_update,
        frames=len(cache), interval=1000.0 / max(GIF_FPS, 1), blit=False,
    )
    gif_path = os.path.join(images_dir, f'{case_name}_transient.gif')
    try:
        anim.save(gif_path, writer=animation.PillowWriter(fps=GIF_FPS))
        print(f"  wrote {gif_path}  ({len(cache)} frames, {GIF_FPS} fps)")
    except Exception as exc:
        print(f"  [warn] gif save failed ({exc}); try `pip install pillow`.")
    plt.close(fig_g)

# -------------------------------------------------------------------- Ghia-style centerline cuts
if PLOT_CENTERLINE_PROFILES:
    d = cache[-1]
    print(f"\n  final-time centerline profiles (t = {d['t']:.3f} s)")
    # Vertical centerline: u_x(y) along x = 0.
    # Horizontal centerline: u_y(x) along y = 0.
    ix0 = NX_SAMPLE // 2
    iy0 = NY_SAMPLE // 2
    y1d = np.linspace(y_min, y_max, NY_SAMPLE)
    x1d = np.linspace(x_min, x_max, NX_SAMPLE)
    ux_along_y = d['vx'][:, ix0]
    uy_along_x = d['vy'][iy0, :]

    fig_c, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(2.0 * FIG_INCHES, FIG_INCHES))
    ax_a.plot(ux_along_y, y1d, 'b-', lw=1.6)
    ax_a.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_a.set_xlabel('u_x (m/s)', fontsize=FONT_SIZE_LABEL)
    ax_a.set_ylabel('y (m)',     fontsize=FONT_SIZE_LABEL)
    ax_a.set_title('Vertical centerline: u_x(y) at x = 0',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_a.set_ylim([y_min, y_max])
    ax_a.tick_params(labelsize=FONT_SIZE_TICK)
    ax_a.grid(alpha=0.3)

    ax_b.plot(x1d, uy_along_x, 'r-', lw=1.6)
    ax_b.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_b.set_xlabel('x (m)',     fontsize=FONT_SIZE_LABEL)
    ax_b.set_ylabel('u_y (m/s)', fontsize=FONT_SIZE_LABEL)
    ax_b.set_title('Horizontal centerline: u_y(x) at y = 0',
                   fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax_b.set_xlim([x_min, x_max])
    ax_b.tick_params(labelsize=FONT_SIZE_TICK)
    ax_b.grid(alpha=0.3)

    fig_c.tight_layout()
    base = os.path.join(images_dir, f'{case_name}_centerlines_final')
    fig_c.savefig(base + '.png', dpi=DPI, bbox_inches='tight')
    fig_c.savefig(base + '.eps',           bbox_inches='tight')
    plt.close(fig_c)
    print(f"  wrote {base}.png / .eps")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
