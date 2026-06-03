"""
===============================================================================
FLOWVORTEX -- EMBEDDED-SOLID CONTOUR VERIFICATION
===============================================================================

PURPOSE:
    Verify the embedded-solid boundary in the FlowVortex cylinder cases by
    overlaying, on top of the streamwise-velocity field:
        * the SOLID indicator contours (default levels 0.1, 0.5, 0.9), and
        * the velocityx = 0 contour (stagnation / recirculation separatrix).

    Works for both solvers:
        SOLVER = 'hydro'   -> solid indicator field is 'eta'   (1 fluid, 0 solid)
        SOLVER = 'hydro2'  -> solid indicator field is 'phi'   (1 fluid, 0 solid)

INPUTS:
    AMReX plot files (NNNNNcell directories) produced by the FlowVortex runs.

OUTPUTS:
    PNG (+ optional EPS) figures written to a dedicated Images/ folder:
        * one contour figure per selected timestep, and
        * a multi-panel montage of selected timesteps.

USAGE:
    Edit the CONFIGURATION block below (everything is adjustable), then:
        python plot_vortex_contours.py
    or override the solver / output dir on the command line:
        python plot_vortex_contours.py hydro2 ../output_hydro2
===============================================================================
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yt
yt.funcs.mylog.setLevel(40)  # suppress yt chatter

# ============================================================================
# CONFIGURATION  (everything here is adjustable)
# ============================================================================

# Which solver's output to plot: 'hydro' (single-phase) or 'hydro2' (two-phase).
SOLVER = sys.argv[1] if len(sys.argv) > 1 else "hydro2"

# Directory containing the NNNNNcell plot directories (relative to this script).
# Defaults follow the plot_file paths in input_hydro / input_hydro2.
DEFAULT_OUTPUT_DIR = {
    "hydro":  "../output_hydro",
    "hydro2": "../output_hydro2",
}
AMREX_OUTPUT_DIR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR[SOLVER]

# Name of the solid-indicator field in the plotfile (1 = fluid, 0 = solid).
SOLID_FIELD = "eta" if SOLVER == "hydro" else "phi"

# Velocity field names in the plotfile.
VELX_FIELD = "velocityx"
VELY_FIELD = "velocityy"

# Contour levels for the solid indicator and the colour of those lines.
SOLID_CONTOUR_LEVELS = [0.1, 0.5, 0.9]
SOLID_CONTOUR_COLOR  = "k"
SOLID_CONTOUR_LW     = 1.6

# velocityx = 0 contour (recirculation separatrix).
PLOT_VELX_ZERO       = True
VELX_ZERO_COLOR      = "magenta"
VELX_ZERO_LW         = 2.0

# Background field shown as a filled colormap (set to None to disable).
BACKGROUND_FIELD     = VELX_FIELD     # e.g. "velocityx", "velocityy", "vorticity"
BACKGROUND_CMAP      = "RdBu_r"
BACKGROUND_SYMMETRIC = True           # symmetric color limits about 0

# Sampling resolution of the uniform image (frb) used for contouring.
FRB_RES = 800

# Which timesteps to plot individually: "last", "all", or an explicit list of
# integer step numbers, e.g. [0, 50, 100].
TIMESTEPS_TO_PLOT = "last"

# Montage of selected timesteps (rows x cols).  Set MONTAGE_PANELS to 0 to skip.
MONTAGE_PANELS = 6
MONTAGE_SHAPE  = (2, 3)

# Plot sizes and fonts (all adjustable).
FIG_SIZE        = (11, 6)
MONTAGE_FIGSIZE = (18, 9)
FONT_SIZE_TITLE = 15
FONT_SIZE_LABEL = 13
FONT_SIZE_TICK  = 11
DPI             = 200
SAVE_EPS        = False

# Output folder for images (created if absent).
IMAGE_DIR = "./Images"

# Optional zoom window [xlo, xhi, ylo, yhi] (None -> full domain).
ZOOM = None     # e.g. [-2.0, 6.0, -3.0, 3.0]

# ============================================================================
# HELPERS
# ============================================================================

def step_number(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_plotfiles(output_dir):
    if not os.path.isdir(output_dir):
        print(f"ERROR: output directory not found: {output_dir}")
        sys.exit(1)
    pfs = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
           if os.path.isdir(os.path.join(output_dir, d)) and d.endswith("cell")]
    pfs.sort(key=step_number)
    if not pfs:
        print(f"ERROR: no NNNNNcell plot directories in {output_dir}")
        sys.exit(1)
    return pfs


def slice_field(ds, field, bounds, res):
    """Return a uniform 2D array of `field` over `bounds`=[xlo,xhi,ylo,yhi]."""
    xlo, xhi, ylo, yhi = bounds
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    wx, wy = (xhi - xlo), (yhi - ylo)
    res_y = max(4, int(res * wy / wx))
    slc = ds.slice("z", 0.0)
    # yt resolution is (npix_x, npix_y); the returned array is shaped (ny, nx).
    frb = slc.to_frb(width=(wx, "code_length"), height=(wy, "code_length"),
                     center=ds.arr([cx, cy, 0.0], "code_length"),
                     resolution=(res, res_y))
    return np.array(frb[field])


def domain_bounds(ds):
    le = ds.domain_left_edge.to_value()
    re = ds.domain_right_edge.to_value()
    return [float(le[0]), float(re[0]), float(le[1]), float(re[1])]


def draw_panel(ax, ds, bounds):
    """Draw background + solid contours + velx=0 contour on `ax`."""
    xlo, xhi, ylo, yhi = bounds
    extent = [xlo, xhi, ylo, yhi]
    solid = slice_field(ds, SOLID_FIELD, bounds, FRB_RES)
    ny, nx = solid.shape
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)
    X, Y = np.meshgrid(x, y)

    if BACKGROUND_FIELD is not None:
        bg = slice_field(ds, BACKGROUND_FIELD, bounds, FRB_RES)
        if BACKGROUND_SYMMETRIC:
            vmax = np.nanmax(np.abs(bg)) + 1e-30
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(bg), np.nanmax(bg)
        im = ax.imshow(bg, origin="lower", extent=extent, aspect="equal",
                       cmap=BACKGROUND_CMAP, vmin=vmin, vmax=vmax)
    else:
        im = None

    # Solid-indicator contours
    ax.contour(X, Y, solid, levels=sorted(SOLID_CONTOUR_LEVELS),
               colors=SOLID_CONTOUR_COLOR, linewidths=SOLID_CONTOUR_LW)

    # velocityx = 0 contour
    if PLOT_VELX_ZERO:
        velx = slice_field(ds, VELX_FIELD, bounds, FRB_RES)
        ax.contour(X, Y, velx, levels=[0.0],
                   colors=VELX_ZERO_COLOR, linewidths=VELX_ZERO_LW)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("y", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    return im


def legend_proxies():
    from matplotlib.lines import Line2D
    proxies = [Line2D([0], [0], color=SOLID_CONTOUR_COLOR, lw=SOLID_CONTOUR_LW,
                      label=f"{SOLID_FIELD} = {SOLID_CONTOUR_LEVELS}")]
    if PLOT_VELX_ZERO:
        proxies.append(Line2D([0], [0], color=VELX_ZERO_COLOR, lw=VELX_ZERO_LW,
                              label="velocityx = 0"))
    return proxies

# ============================================================================
# MAIN
# ============================================================================

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    plot_files = find_plotfiles(AMREX_OUTPUT_DIR)
    print(f"SOLVER={SOLVER}  solid field='{SOLID_FIELD}'")
    print(f"Found {len(plot_files)} plot files in {AMREX_OUTPUT_DIR}")

    # Determine which timesteps to render individually.
    if TIMESTEPS_TO_PLOT == "last":
        selected = [plot_files[-1]]
    elif TIMESTEPS_TO_PLOT == "all":
        selected = plot_files
    else:
        wanted = set(int(s) for s in TIMESTEPS_TO_PLOT)
        selected = [p for p in plot_files if step_number(p) in wanted] or [plot_files[-1]]

    # ---- individual figures ------------------------------------------------
    for pf in selected:
        ds = yt.load(pf)
        t = float(ds.current_time)
        bounds = ZOOM if ZOOM is not None else domain_bounds(ds)

        fig, ax = plt.subplots(figsize=FIG_SIZE)
        im = draw_panel(ax, ds, bounds)
        if im is not None:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label(BACKGROUND_FIELD, fontsize=FONT_SIZE_LABEL)
        ax.legend(handles=legend_proxies(), loc="upper right",
                  fontsize=FONT_SIZE_TICK, framealpha=0.9)
        ax.set_title(f"FlowVortex {SOLVER}  (Re=40)   step {step_number(pf)}, t = {t:.4g}",
                     fontsize=FONT_SIZE_TITLE, fontweight="bold")
        plt.tight_layout()
        base = os.path.join(IMAGE_DIR, f"contours_{SOLVER}_{step_number(pf):06d}")
        fig.savefig(base + ".png", dpi=DPI)
        if SAVE_EPS:
            fig.savefig(base + ".eps")
        plt.close(fig)
        print(f"  wrote {base}.png  (t={t:.4g})")

    # ---- montage -----------------------------------------------------------
    if MONTAGE_PANELS and len(plot_files) > 1:
        n = min(MONTAGE_PANELS, len(plot_files))
        idx = np.linspace(0, len(plot_files) - 1, n, dtype=int)
        rows, cols = MONTAGE_SHAPE
        fig, axes = plt.subplots(rows, cols, figsize=MONTAGE_FIGSIZE)
        axes = np.array(axes).reshape(-1)
        for a in axes[n:]:
            a.axis("off")
        for k, i in enumerate(idx):
            ds = yt.load(plot_files[i])
            t = float(ds.current_time)
            bounds = ZOOM if ZOOM is not None else domain_bounds(ds)
            draw_panel(axes[k], ds, bounds)
            axes[k].set_title(f"t = {t:.3g}", fontsize=FONT_SIZE_LABEL)
        fig.suptitle(f"FlowVortex {SOLVER} (Re=40): {SOLID_FIELD} contours "
                     f"{SOLID_CONTOUR_LEVELS} + velocityx=0",
                     fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")
        fig.legend(handles=legend_proxies(), loc="lower center", ncol=2,
                   fontsize=FONT_SIZE_TICK)
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        base = os.path.join(IMAGE_DIR, f"montage_{SOLVER}")
        fig.savefig(base + ".png", dpi=DPI)
        if SAVE_EPS:
            fig.savefig(base + ".eps")
        plt.close(fig)
        print(f"  wrote {base}.png")

    print("Done.")


if __name__ == "__main__":
    main()
