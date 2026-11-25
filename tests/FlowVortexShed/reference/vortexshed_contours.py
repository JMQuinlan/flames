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
    for each PLOT_MODES entry (u_x -> Contours/, vorticity -> Contours_Vorticity/):
        * one contour figure per selected timestep  -> <Images>/<folder>/
        * a multi-panel montage of selected timesteps -> <Images>/
        * MAKE_GIF = True: a frame per plotfile and the wake-development
          animation wake_[vorticity_]<solver>.gif     -> <Images>/<folder>/
    Colour limits are fixed per mode (MODE_STYLE vmax), so colorbars are identical
    in every figure and GIF frame.

USAGE:
    Edit the CONFIGURATION block below (everything is adjustable), then:
        python plot_vortex_contours.py
    or override the solver / output dir on the command line:
        python plot_vortex_contours.py hydro2 ../output_hydro2
    optional third argument = image folder; VORTEX_ZOOM="xlo,xhi,ylo,yhi" | full
    sets the plot window (default: near field around the cylinder and wake).
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
# LaTeX label for the solid indicator (legend, colorbar, montage title).
SOLID_LABEL = r"$\eta$" if SOLVER == "hydro" else r"$\phi$"

# Velocity field names in the plotfile.
VELX_FIELD = "velocityx"
VELY_FIELD = "velocityy"

# Contour levels for the solid indicator and the colour of those lines.
PLOT_SOLID_CONTOURS  = False          # phi contour lines (+ legend entry); phi is shown by the colorbar
SOLID_CONTOUR_LEVELS = [0.1, 0.5, 0.9]
SOLID_CONTOUR_COLOR  = "k"
SOLID_CONTOUR_LW     = 1.6

# velocityx = 0 contour (recirculation separatrix).
PLOT_VELX_ZERO       = True
VELX_ZERO_COLOR      = "red"          # (per mode: MODE_STYLE[...]["velx0_color"])
VELX_ZERO_LW         = 2.0

# Background field shown as a filled colormap.  One set of figures (final-time
# contour, montage, GIF frames, GIF) is made per entry of PLOT_MODES, each in its
# own sub-folder of <IMAGE_DIR>:
#   "ux"         streamwise velocity u_x                        -> Contours/
#   "vorticity"  omega_z = d(u_y)/dx - d(u_x)/dy  (computed)    -> Contours_Vorticity/
# vmax is the FIXED symmetric colour limit used for EVERY figure and GIF frame of
# that mode (same colorbar for all times and all Reynolds numbers; values beyond it
# saturate).  vmax = None -> each figure uses its own max (colorbar then changes).
# Override the list with the environment variable VORTEX_MODES="ux,vorticity".
PLOT_MODES = ["ux", "vorticity"]
MODE_STYLE = {
    "ux": dict(subdir="Contours", tag="", label=r"$u_x$", cmap="RdBu", vmax=1.5,
               stream_color="white", stream_alpha=0.5, velx0_color="red"),
    "vorticity": dict(subdir="Contours_Vorticity", tag="vorticity_",
               label=r"$\omega_z$", cmap="RdBu_r", vmax=4.0,
               stream_color="0.3", stream_alpha=0.45, velx0_color="black"),
}
_m = os.environ.get("VORTEX_MODES")
if _m:
    PLOT_MODES = [v.strip() for v in _m.split(",") if v.strip()]
MODE = PLOT_MODES[0]                  # current mode (set by main)
BACKGROUND_SYMMETRIC = True           # symmetric color limits about 0
# Reynolds number for the titles.  None = read it from the output path
# (e.g. .../Re47_bigdomain_mirror/output -> 47); give a number to force it.
RE                   = None
TITLE_TMPL           = "Re = {re} Flow over a Cylinder"   # single-figure and montage title

# Solid indicator drawn ON TOP of the velocity map as a black layer whose opacity
# is SOLID_OVERLAY_ALPHA * (1 - phi): pure solid (phi = 0) is opaque black, pure
# fluid (phi = 1) is untouched, and a diffuse interface shows as a graded band
# (compare diffuse vs mirrored sharp walls).  The phi contour lines are drawn on
# top of that with SOLID_CONTOUR_ALPHA (0.8 = 20% transparent).
PLOT_SOLID_BACKGROUND = True
SOLID_CMAP            = "gray"       # colorbar: 0 = black (solid) -> 1 = white (fluid)
SOLID_COLORBAR        = True
SOLID_OVERLAY_ALPHA   = 1.0          # opacity of pure solid (1 = opaque black)
SOLID_CONTOUR_ALPHA   = 0.8          # opacity of the phi contour lines (0.8 = 20% transparent)
# DISPLAY-ONLY treatment of a SHARP phi (interface thinner than SHARP_PHI_CELLS
# finest cells -- same rule as the solver's solid.wall_flux auto selection): phi is
# 0/1 cell by cell and would draw the cell staircase, so it is Gaussian-smoothed by
# SOLID_SMOOTH_CELLS cells (round outline) and then re-sharpened so the 10-90% band
# is ~1 cell again (a sharp wall still LOOKS sharp).  A DIFFUSE phi is drawn as is
# (no smoothing), so its true band shows.  0 = never smooth.  Data are not changed.
SOLID_SMOOTH_CELLS    = 2.0
SHARP_PHI_CELLS       = 2.0

# Faint streamlines of (u_x, u_y) over the velocity map, plus extra seeds inside
# the reversed-flow region so the recirculation vortices are drawn.
PLOT_STREAMLINES      = True
STREAM_COLOR          = "white"      # (per mode: MODE_STYLE[...]["stream_color"])
STREAM_ALPHA          = 0.5
STREAM_LW             = 0.6
STREAM_DENSITY        = 1.6          # matplotlib streamplot density (background lines)
STREAM_ARROWSIZE      = 0.6
STREAM_SEED_WAKE      = True         # extra seeds where u_x < 0 (recirculation zone)
STREAM_WAKE_SEEDS     = 12           # number of wake seed points per half
STREAM_WAKE_COLOR     = "0.25"       # wake lines sit on a near-white background (u ~ 0): dark gray
STREAM_WAKE_ALPHA     = 0.7

# Sampling resolution (pixels in x) of the uniform image used for plotting and
# contouring.  Fields are read on the finest AMR level over the plot window and
# LINEARLY INTERPOLATED onto this grid (smooth contours instead of the
# cell-by-cell staircase of a nearest-neighbour pixel buffer).
FRB_RES = 800
INTERP_METHOD = "linear"             # "linear" or "nearest" (old look)

# Mask the velocityx = 0 contour inside the solid (phi < this), where u ~ 0 and
# the contour would otherwise trace noise along the wall.
VELX_ZERO_MASK_PHI = 0.5

# Which timesteps to plot individually: "last", "all", or an explicit list of
# integer step numbers, e.g. [0, 50, 100].
TIMESTEPS_TO_PLOT = "last"

# Montage of selected timesteps (rows x cols).  Set MONTAGE_PANELS to 0 to skip.
MONTAGE_PANELS = 6
MONTAGE_SHAPE  = (2, 3)

# Wake-development animation.  MAKE_GIF = True renders EVERY GIF_STRIDE-th plotfile
# (same figure as the single contour plot, time in the title) into the mode's folder
# (<IMAGE_DIR>/Contours/, <IMAGE_DIR>/Contours_Vorticity/) and assembles them into
# wake_<solver>.gif / wake_vorticity_<solver>.gif there.  All frames share the mode's
# fixed colour limit AND one GIF colour palette, so colorbars and legend stay static.
# False = only the TIMESTEPS_TO_PLOT figures (also written to those folders).  Command-line/env override: VORTEX_GIF=1 | 0.
MAKE_GIF          = True
GIF_STRIDE        = 1              # 1 = every plotfile
GIF_FPS           = 15             # playback speed, frames per second
GIF_SCALE         = 1.0            # GIF frame size relative to the PNG frames (1.0 = no resampling)
# A GIF holds at most 256 colours per frame (format limit).  Instead of letting an
# automatic quantiser pick them (which smears the colorbars: colour fringing), the
# palette is BUILT from the plot's own colours: GIF_CMAP_LEVELS evenly spaced levels
# of the field colormap + GIF_GRAY_LEVELS grays (phi colorbar, text, axes) + the line
# colours.  Every frame is mapped to that one palette with no dithering, so the
# colorbars are clean and identical in every frame.  (Sum must stay <= ~250.)
GIF_CMAP_LEVELS   = 176
GIF_GRAY_LEVELS   = 64
GIF_SKIP_EXISTING = True           # reuse frame PNGs already in Contours/ (resume / re-assemble)
_g = os.environ.get("VORTEX_GIF")
if _g:
    MAKE_GIF = _g.strip().lower() not in ("0", "false", "no", "off")

# Plot sizes and fonts (all adjustable).
FIG_SIZE        = (11, 6)
MONTAGE_FIGSIZE = (18, 9)
FONT_SIZE_TITLE = 15
FONT_SIZE_LABEL = 13
FONT_SIZE_TICK  = 11
DPI             = 200
SAVE_EPS        = False

# Output folder for images (created if absent); optional 3rd command-line argument.
IMAGE_DIR = sys.argv[3] if len(sys.argv) > 3 else "./Images"

# Zoom window [xlo, xhi, ylo, yhi] around the cylinder and its wake (None -> full
# domain).  The big-domain decks are 40 D x 30 D, so the default is a near-field
# window.  Override with the environment variable VORTEX_ZOOM, e.g.
#   VORTEX_ZOOM="-2,14,-3,3"   (long wake)      VORTEX_ZOOM=full   (whole domain)
ZOOM = [-2.0, 6.0, -2.0, 2.0]
_z = os.environ.get("VORTEX_ZOOM")
if _z:
    ZOOM = None if _z.strip().lower() == "full" else [float(v) for v in _z.split(",")]

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
    """Return a uniform 2D array (ny, nx) of `field` over `bounds`=[xlo,xhi,ylo,yhi],
    read on the finest level and interpolated (INTERP_METHOD) onto the image grid."""
    from scipy.interpolate import RegularGridInterpolator
    xlo, xhi, ylo, yhi = bounds
    L = ds.index.max_level
    dxf = (ds.domain_width / (ds.domain_dimensions * ds.refine_by**L)).to_value()
    le = ds.domain_left_edge.to_value(); re_ = ds.domain_right_edge.to_value()
    # finest-level cells covering the window (plus one cell of margin), clipped to the domain
    i0 = max(0, int(np.floor((xlo - le[0]) / dxf[0])) - 1); i1 = min(int(round((re_[0] - le[0]) / dxf[0])), int(np.ceil((xhi - le[0]) / dxf[0])) + 1)
    j0 = max(0, int(np.floor((ylo - le[1]) / dxf[1])) - 1); j1 = min(int(round((re_[1] - le[1]) / dxf[1])), int(np.ceil((yhi - le[1]) / dxf[1])) + 1)
    left = [le[0] + i0 * dxf[0], le[1] + j0 * dxf[1], le[2]]
    # smoothed covering grid: coarse-level data is interpolated (not piecewise-constant)
    # onto the finest grid, so contours stay smooth where the finest level does not reach
    try:
        cg = ds.smoothed_covering_grid(level=L, left_edge=left, dims=[i1 - i0, j1 - j0, 1])
        a = np.asarray(cg[("boxlib", field)])[:, :, 0]
    except RuntimeError:
        # yt wants ghost cells at the (non-periodic) domain edge for the coarse-level
        # interpolation.  Forcing periodicity only affects those ghost fills: a window
        # inside the domain is unchanged; one touching the edge may mix the outermost
        # coarse cells with the opposite side.
        ds.force_periodicity()
        cg = ds.smoothed_covering_grid(level=L, left_edge=left, dims=[i1 - i0, j1 - j0, 1])
        a = np.asarray(cg[("boxlib", field)])[:, :, 0]
    xc = left[0] + (np.arange(i1 - i0) + 0.5) * dxf[0]
    yc = left[1] + (np.arange(j1 - j0) + 0.5) * dxf[1]
    res_y = max(4, int(res * (yhi - ylo) / (xhi - xlo)))
    xi = np.clip(np.linspace(xlo, xhi, res), xc[0], xc[-1])
    yi = np.clip(np.linspace(ylo, yhi, res_y), yc[0], yc[-1])
    Xi, Yi = np.meshgrid(xi, yi)                      # (ny, nx)
    f = RegularGridInterpolator((xc, yc), a, method=INTERP_METHOD)
    return f(np.c_[Xi.ravel(), Yi.ravel()]).reshape(Xi.shape)


def domain_bounds(ds):
    le = ds.domain_left_edge.to_value()
    re = ds.domain_right_edge.to_value()
    return [float(le[0]), float(re[0]), float(le[1]), float(re[1])]


def draw_panel(ax, ds, bounds):
    """Draw background (u_x or vorticity, per MODE) + solid + streamlines + velx=0 contour."""
    M = MODE_STYLE[MODE]
    vmax_fixed = M["vmax"]
    xlo, xhi, ylo, yhi = bounds
    extent = [xlo, xhi, ylo, yhi]
    _cache = {}
    _sf = globals()["slice_field"]
    def slice_field(ds_, field, bounds_, res):                     # each field is read once per panel
        if field not in _cache:
            _cache[field] = _sf(ds_, field, bounds_, res)
        return _cache[field]
    solid = slice_field(ds, SOLID_FIELD, bounds, FRB_RES)
    ny, nx = solid.shape
    x = np.linspace(xlo, xhi, nx)
    y = np.linspace(ylo, yhi, ny)
    X, Y = np.meshgrid(x, y)

    phi01 = np.clip(solid, 0.0, 1.0)
    phi_disp = phi01
    if SOLID_SMOOTH_CELLS and SOLID_SMOOTH_CELLS > 0:
        from scipy.ndimage import gaussian_filter
        L = ds.index.max_level
        dxf = float((ds.domain_width[0] / (ds.domain_dimensions[0] * ds.refine_by**L)).to_value())
        px = (xhi - xlo) / nx                       # image pixel size
        gy, gx = np.gradient(phi01, px)
        jump = np.nanmax(np.hypot(gx, gy)) * dxf    # largest phi change per finest cell
        thick = 1.0 / jump if jump > 0 else np.inf  # interface thickness in cells
        if thick < SHARP_PHI_CELLS:
            sig = SOLID_SMOOTH_CELLS * dxf / px
            phs = gaussian_filter(phi01, sigma=sig, mode="nearest")
            gain = 2.56 * SOLID_SMOOTH_CELLS         # 10-90% width of a smoothed step = 2.56 sigma -> ~1 cell
            phi_disp = np.clip(0.5 + gain * (phs - 0.5), 0.0, 1.0)
    if True:
        if MODE == "vorticity":                      # omega_z on the image grid
            ux_ = slice_field(ds, VELX_FIELD, bounds, FRB_RES)
            uy_ = slice_field(ds, VELY_FIELD, bounds, FRB_RES)
            bg = np.gradient(uy_, x, axis=1) - np.gradient(ux_, y, axis=0)
            bg = np.where(solid < VELX_ZERO_MASK_PHI, 0.0, bg)
        else:
            bg = slice_field(ds, VELX_FIELD, bounds, FRB_RES)
        if vmax_fixed is not None:
            vmax = float(vmax_fixed); vmin = -vmax if BACKGROUND_SYMMETRIC else np.nanmin(bg)
        elif BACKGROUND_SYMMETRIC:
            vmax = np.nanmax(np.abs(bg)) + 1e-30
            vmin = -vmax
        else:
            vmin, vmax = np.nanmin(bg), np.nanmax(bg)
        im = ax.imshow(bg, origin="lower", extent=extent, aspect="equal",
                       cmap=M["cmap"], vmin=vmin, vmax=vmax, zorder=1)
    if PLOT_SOLID_BACKGROUND:
        rgba = np.zeros(phi_disp.shape + (4,))          # black, opacity ~ solid fraction
        rgba[..., 3] = SOLID_OVERLAY_ALPHA * (1.0 - phi_disp)
        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal", zorder=2)

    # Solid-indicator contours
    if PLOT_SOLID_CONTOURS:
        ax.contour(X, Y, phi_disp, levels=sorted(SOLID_CONTOUR_LEVELS),
                   colors=SOLID_CONTOUR_COLOR, linewidths=SOLID_CONTOUR_LW, zorder=3,
                   alpha=SOLID_CONTOUR_ALPHA)

    # streamlines (masked inside the solid)
    if PLOT_STREAMLINES:
        ux = slice_field(ds, VELX_FIELD, bounds, FRB_RES)
        uy = slice_field(ds, VELY_FIELD, bounds, FRB_RES)
        solid_mask = solid < VELX_ZERO_MASK_PHI
        uxm = np.ma.masked_where(solid_mask, ux); uym = np.ma.masked_where(solid_mask, uy)
        sp = dict(color=M["stream_color"], linewidth=STREAM_LW, arrowsize=STREAM_ARROWSIZE, zorder=3)
        st = ax.streamplot(x, y, uxm, uym, density=STREAM_DENSITY, **sp)
        st.lines.set_alpha(M["stream_alpha"]); st.arrows.set_alpha(M["stream_alpha"])
        if STREAM_SEED_WAKE:
            rev = (ux < 0) & ~solid_mask & (X > 0)
            if rev.any():
                xs_, ys_ = X[rev], Y[rev]
                xa = np.linspace(xs_.min(), xs_.max(), STREAM_WAKE_SEEDS + 2)[1:-1]
                ymax = np.abs(ys_).max()
                seeds = np.array([(xx, sgn * 0.5 * ymax) for xx in xa for sgn in (-1, 1)])
                sp2 = dict(sp, color=STREAM_WAKE_COLOR)
                st2 = ax.streamplot(x, y, uxm, uym, start_points=seeds, density=4.0,
                                    integration_direction="both", **sp2)
                st2.lines.set_alpha(STREAM_WAKE_ALPHA); st2.arrows.set_alpha(STREAM_WAKE_ALPHA)

    # velocityx = 0 contour
    if PLOT_VELX_ZERO:
        velx = slice_field(ds, VELX_FIELD, bounds, FRB_RES)
        velx = np.ma.masked_where(solid < VELX_ZERO_MASK_PHI, velx)
        ax.contour(X, Y, velx, levels=[0.0],
                   colors=M["velx0_color"], linewidths=VELX_ZERO_LW, zorder=4)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("y", fontsize=FONT_SIZE_LABEL)
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    return im


def legend_proxies():
    from matplotlib.lines import Line2D
    proxies = []
    if PLOT_SOLID_CONTOURS:
        proxies.append(Line2D([0], [0], color=SOLID_CONTOUR_COLOR, lw=SOLID_CONTOUR_LW,
                              label=f"{SOLID_LABEL} = {SOLID_CONTOUR_LEVELS}"))
    if PLOT_VELX_ZERO:
        proxies.append(Line2D([0], [0], color=MODE_STYLE[MODE]["velx0_color"], lw=VELX_ZERO_LW,
                              label="$u_x$ = 0"))
    return proxies

# ============================================================================
# MAIN
# ============================================================================

def reynolds_number():
    """RE if set, else the first 'Re<number>' in the output path (default 40)."""
    if RE is not None:
        return RE
    m = re.search(r"Re[_=]?(\d+(?:\.\d+)?)", os.path.abspath(AMREX_OUTPUT_DIR))
    return float(m.group(1)) if m else 40


def render_figure(pf, base, title=None):
    """One contour figure of plotfile `pf` -> <base>.png (+ .eps).  Returns the time."""
    import matplotlib as mpl
    ds = yt.load(pf)
    t = float(ds.current_time)
    bounds = ZOOM if ZOOM is not None else domain_bounds(ds)
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    im = draw_panel(ax, ds, bounds)
    if im is not None:
        sm = mpl.cm.ScalarMappable(norm=im.norm, cmap=im.cmap)   # opaque colorbar
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02,
                          extend="both" if MODE_STYLE[MODE]["vmax"] is not None else "neither")
        cb.set_label(MODE_STYLE[MODE]["label"], fontsize=FONT_SIZE_LABEL)
    if PLOT_SOLID_BACKGROUND and SOLID_COLORBAR:
        sm2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=SOLID_CMAP)
        cb2 = fig.colorbar(sm2, ax=ax, orientation="horizontal", fraction=0.05, pad=0.12, aspect=40)
        cb2.set_label(SOLID_LABEL, fontsize=FONT_SIZE_LABEL)
    ax.legend(handles=legend_proxies(), loc="upper right",
              fontsize=FONT_SIZE_TICK, framealpha=0.9)
    ax.set_title(TITLE if title is None else title.format(t=t),
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")
    plt.tight_layout()
    fig.savefig(base + ".png", dpi=DPI)
    if SAVE_EPS:
        fig.savefig(base + ".eps")
    plt.close(fig)
    return t


def make_gif(plot_files, contour_dir):
    """Frames of every GIF_STRIDE-th plotfile -> <contour_dir>/frame_NNNNNN.png and
    <contour_dir>/wake_[vorticity_]<solver>.gif.  Fixed colour limit (MODE_STYLE vmax)
    and ONE colour palette for all frames: colorbars, legend and axes are static."""
    from PIL import Image
    frames = plot_files[::max(1, int(GIF_STRIDE))]
    if frames[-1] != plot_files[-1]:
        frames.append(plot_files[-1])
    print(f"  GIF [{MODE}]: {len(frames)} frames, colour limit +-{MODE_STYLE[MODE]['vmax']}, -> {contour_dir}")
    pngs = []
    for k, pf in enumerate(frames):
        base = os.path.join(contour_dir, f"frame_{step_number(pf):06d}")
        if not (GIF_SKIP_EXISTING and os.path.isfile(base + ".png")):
            render_figure(pf, base, title=TITLE + "   (t = {t:.1f})")
        pngs.append(base + ".png")
        if (k + 1) % 25 == 0 or k + 1 == len(frames):
            print(f"    frame {k + 1}/{len(frames)}", flush=True)

    def load(f):
        im = Image.open(f).convert("RGB")
        if GIF_SCALE and GIF_SCALE != 1.0:
            im = im.resize((int(im.width * GIF_SCALE), int(im.height * GIF_SCALE)), Image.LANCZOS)
        return im
    # one designed palette for all frames (see GIF_CMAP_LEVELS): colormap levels + grays + line colours
    import matplotlib as mpl
    from matplotlib.colors import to_rgb
    M = MODE_STYLE[MODE]
    cols = [tuple(int(round(255 * c)) for c in mpl.colormaps[M["cmap"]](v)[:3])
            for v in np.linspace(0.0, 1.0, GIF_CMAP_LEVELS)]
    cols += [(g, g, g) for g in np.linspace(0, 255, GIF_GRAY_LEVELS).round().astype(int)]
    cols += [tuple(int(round(255 * c)) for c in to_rgb(k))
             for k in (M["velx0_color"], M["stream_color"], STREAM_WAKE_COLOR, SOLID_CONTOUR_COLOR)]
    cols = list(dict.fromkeys(cols))[:256]
    cols += [cols[-1]] * (256 - len(cols))
    pal = Image.new("P", (1, 1))
    pal.putpalette([c for rgb in cols for c in rgb])
    imgs = [load(f).quantize(palette=pal, dither=Image.NONE) for f in pngs]
    out = os.path.join(contour_dir, f"wake_{MODE_STYLE[MODE]['tag']}{SOLVER}.gif")
    dur = [int(round(1000.0 / GIF_FPS))] * len(imgs)
    dur[-1] = 2000                                       # hold the last frame 2 s
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=False)
    print(f"  wrote {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")


def main():
    global TITLE, MODE
    rn = reynolds_number()
    TITLE = TITLE_TMPL.format(re=f"{rn:g}")
    print(f"  title: {TITLE}")
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

    for MODE in PLOT_MODES:
        M = MODE_STYLE[MODE]
        contour_dir = os.path.join(IMAGE_DIR, M["subdir"])
        os.makedirs(contour_dir, exist_ok=True)

        # ---- individual figures (-> Contours[_Vorticity]/) -------------------
        for pf in selected:
            base = os.path.join(contour_dir, f"contours_{M['tag']}{SOLVER}_{step_number(pf):06d}")
            t = render_figure(pf, base)
            print(f"  wrote {base}.png  (t={t:.4g})")

        # ---- montage ---------------------------------------------------------
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
            fig.suptitle(TITLE + ("" if MODE == "ux" else "  (vorticity)"),
                         fontsize=FONT_SIZE_TITLE + 1, fontweight="bold")
            fig.legend(handles=legend_proxies(), loc="lower center", ncol=2,
                       fontsize=FONT_SIZE_TICK)
            plt.tight_layout(rect=[0, 0.04, 1, 0.97])
            base = os.path.join(IMAGE_DIR, f"montage_{M['tag']}{SOLVER}")
            fig.savefig(base + ".png", dpi=DPI)
            if SAVE_EPS:
                fig.savefig(base + ".eps")
            plt.close(fig)
            print(f"  wrote {base}.png")

        # ---- wake-development GIF --------------------------------------------
        if MAKE_GIF and len(plot_files) > 1:
            make_gif(plot_files, contour_dir)

    print("Done.")


if __name__ == "__main__":
    main()
