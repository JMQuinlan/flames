"""
===============================================================================
"CURRENTS" -- ARTISTIC FLOW RENDERER  (Tame Impala album-cover homage)
===============================================================================

PURPOSE:
    Render the vortex-street wake from the FlowCurrents simulation in the style
    of the Tame Impala "Currents" cover (Robert Beatty): a smooth indigo ->
    magenta -> orange -> cream gradient with light streamlines wrapping the
    "sphere" (cylinder) and rolling up into the swirling wake.

    This is an ARTISTIC renderer, not a scientific one -- no axes, ticks, or
    colorbars by default.  Every visual feature is a knob in the CONFIG block.

OUTPUTS:
    - One standalone PNG per selected timestep (in IMAGE_DIR)
    - An animated GIF assembled from the frames

USAGE:
    Edit the CONFIG block, then:   python TameImpalaCurrents.py
    (override the plotfile dir on the command line:  python TameImpalaCurrents.py ../output_currents)

REQUIRES: yt, numpy, matplotlib, pillow
===============================================================================
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from PIL import Image

import yt
yt.funcs.mylog.setLevel(50)

# =============================================================================
# CONFIG  --  everything here is customizable
# =============================================================================

# ---- paths -----------------------------------------------------------------
AMREX_OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "../output_currents"  # dir of NNNNNcell plotfiles
IMAGE_DIR        = "./Images"          # standalone frames go here
GIF_PATH         = "./Images/Currents.gif"

# ---- which frames -----------------------------------------------------------
# "all", "last", or an explicit list of step numbers e.g. [100, 200, 300].
FRAMES        = "all"
FRAME_STRIDE  = 1                      # use every Nth plotfile (after sorting)
FRAME_START   = 0                      # skip the first FRAME_START plotfiles (startup transient)

# ---- view / sampling --------------------------------------------------------
# Crop window [xlo, xhi, ylo, yhi] in code units; None -> full domain.
ZOOM          = None                  # e.g. [-3.0, 18.0, -5.0, 5.0]
FRB_RES       = 1400                  # horizontal pixels of the sampled image (higher = crisper, slower)

# ---- the CURRENTS palette  (indigo -> magenta -> orange -> cream) -----------
# Tweak these hex stops to taste; this is the heart of the album look.
CURRENTS_PALETTE = [
    "#0a0420",   # near-black deep indigo
    "#241455",   # deep indigo / blue-violet
    "#4b1487",   # royal purple
    "#7a1a9e",   # purple-magenta
    "#b3187f",   # magenta
    "#e23a5e",   # pink-red
    "#f4632a",   # orange-red
    "#fb9412",   # orange
    "#ffc24b",   # amber
    "#ffe9bd",   # pale cream highlight
]
PALETTE_POSITIONS = None              # None -> evenly spaced; or a list in [0,1] same length as palette

# ---- background --------------------------------------------------------------
# 'field'    : color a data field with the Currents cmap (the vortices glow)
# 'gradient' : a pure vertical Currents gradient (most album-faithful base);
#              the data only appears through the streamlines
BACKGROUND_MODE   = "field"
BACKGROUND_FIELD  = "vorticity"       # 'vorticity' | 'speed' | 'velocityx' | 'velocityy' | 'pressure' | 'density'
BACKGROUND_ABS    = True              # use |field| (good for vorticity -> a sequential glow)
BACKGROUND_SIGNED = False             # if True, map a signed field two-sided about 0 (overrides ABS)
# Color limits: None -> auto from a robust percentile of the data
VMIN = None
VMAX = None
CLIP_PERCENTILE = 99.0                # auto vmax = this percentile of |field| (ignore outliers)
GAMMA = 0.75                          # <1 brightens mid-tones (more glow); 1.0 = linear
GRADIENT_ANGLE_DEG = 90.0             # for BACKGROUND_MODE='gradient': 90 = vertical, 0 = horizontal

# ---- streamlines (the "currents") ------------------------------------------
SHOW_STREAMLINES = True
# SEED MODE fixes the frame-to-frame "bobbling": matplotlib's streamplot
# auto-seeds from the field each frame, so the lines jump around.  'inlet' pins
# the seeds to a fixed column at the left edge every frame -> coherent, steady
# lines that just deform with the flow (the album look).  'grid' seeds a fixed
# lattice; 'auto' = matplotlib's default (bobbles).
STREAM_SEED_MODE = "inlet"            # 'inlet' | 'grid' | 'auto'
STREAM_N_SEEDS   = 92                 # for 'inlet': number of lines (evenly spaced in y)
STREAM_SEED_INSET = 0.02              # seed this fraction of the width in from the left edge
STREAM_GRID_NX   = 26                 # for 'grid'
STREAM_GRID_NY   = 18
STREAM_DENSITY   = 3.0                # termination grid; higher -> lines run farther before stopping
STREAM_COLOR     = "#fbe8c8"          # light cream lines; or set STREAM_COLOR_BY_SPEED = True
STREAM_COLOR_BY_SPEED = False         # color the lines by local speed with a light cmap
STREAM_LINEWIDTH = 0.9
STREAM_LW_BY_SPEED = True             # scale linewidth by speed (thicker where fast)
STREAM_ALPHA     = 0.85
STREAM_ARROWSIZE = 0.0                # 0 = no arrowheads (cleaner / more album-like)

# ---- red dye (the "red fluid" wrapping the ball) ---------------------------
# Overlays the passive eta tracer band as warm dye.  Needs the eta-band IC in
# input_TameImpalaCurrents (eta=1 strip in front of the ball).
SHOW_DYE      = True
DYE_FIELD     = "eta"
DYE_CMAP      = ["#3a0618", "#9c1030", "#d81e3c", "#ff5a2a"]  # deep crimson -> red -> red -> orange-red
DYE_ALPHA     = 0.9                   # peak opacity of the dye
DYE_GAMMA     = 0.9                   # <1 spreads the dye (more visible after diffusion)
DYE_FLOOR     = 0.06                  # hide dye below this concentration (keeps the far wake clean)

# ---- the cylinder ("sphere") ------------------------------------------------
SHOW_CYLINDER     = True
CYLINDER_FIELD    = "phi"             # solid indicator (phi<0.5 = solid)
CYLINDER_LEVEL    = 0.5
CYLINDER_STYLE    = "chrome"          # 'chrome' = shaded metallic sphere; 'flat' = solid fill
CYLINDER_FILL     = "#9aa0a8"         # base gray (used by 'flat'; also the chrome mid-tone)
CYLINDER_LIGHT    = (-0.55, 0.6, 0.7) # chrome light direction (x,y,z); up-left toward viewer
CYLINDER_EDGE     = "#2b2f36"         # outline color (dark steel); None -> no outline
CYLINDER_EDGE_LW  = 1.2

# ---- figure / output --------------------------------------------------------
FIG_W_INCHES   = 12.0                 # height auto from the aspect ratio
DPI            = 160
FACE_COLOR     = "#0a0420"            # canvas background (behind everything)
SHOW_AXES      = False                # False = pure art (no ticks/labels/frame)
TITLE          = None                 # e.g. "Currents"; None = no title
TITLE_COLOR    = "#ffe9bd"
SAVE_FRAMES    = True
FRAME_PREFIX   = "currents_"

# ---- gif --------------------------------------------------------------------
MAKE_GIF        = True
GIF_FPS         = 20
GIF_LOOP        = 0                    # 0 = loop forever
GIF_MAX_FRAMES  = 240                 # subsample if more frames than this
GIF_BOOMERANG   = False               # play forward then reverse (seamless loop)

# =============================================================================
# helpers
# =============================================================================

def currents_cmap():
    stops = PALETTE_POSITIONS if PALETTE_POSITIONS is not None \
        else np.linspace(0, 1, len(CURRENTS_PALETTE))
    return LinearSegmentedColormap.from_list(
        "currents", list(zip(stops, CURRENTS_PALETTE)), N=512)


def step_number(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_plotfiles(d):
    if not os.path.isdir(d):
        sys.exit(f"ERROR: plotfile dir not found: {d}")
    pfs = [os.path.join(d, p) for p in os.listdir(d)
           if p.endswith("cell") and os.path.isdir(os.path.join(d, p))]
    pfs.sort(key=step_number)
    if not pfs:
        sys.exit(f"ERROR: no NNNNNcell plotfiles in {d}")
    return pfs


def domain_bounds(ds):
    le = ds.domain_left_edge.to_value(); re_ = ds.domain_right_edge.to_value()
    return [float(le[0]), float(re_[0]), float(le[1]), float(re_[1])]


def sample(ds, field, bounds, res):
    """Uniform 2D array of `field` over bounds=[xlo,xhi,ylo,yhi]; returns (arr, ny, nx)."""
    xlo, xhi, ylo, yhi = bounds
    wx, wy = (xhi - xlo), (yhi - ylo)
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    res_y = max(4, int(res * wy / wx))
    slc = ds.slice("z", 0.0)
    frb = slc.to_frb(width=(wx, "code_length"), height=(wy, "code_length"),
                     center=ds.arr([cx, cy, 0.0], "code_length"),
                     resolution=(res, res_y))      # yt: (npix_x, npix_y) -> array (ny, nx)
    return np.array(frb[field])


def get_field(ds, name, bounds, res):
    if name == "speed":
        u = sample(ds, "velocityx", bounds, res)
        v = sample(ds, "velocityy", bounds, res)
        return np.sqrt(u * u + v * v)
    return sample(ds, name, bounds, res)


def vertical_gradient(shape, cmap, angle_deg):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    yy /= max(ny - 1, 1); xx /= max(nx - 1, 1)
    a = np.deg2rad(angle_deg)
    g = np.cos(a) * xx + np.sin(a) * yy
    g = (g - g.min()) / (g.ptp() + 1e-12)
    return cmap(g)


def make_norm(arr):
    if BACKGROUND_SIGNED:
        m = VMAX if VMAX is not None else np.nanpercentile(np.abs(arr), CLIP_PERCENTILE)
        m = max(m, 1e-12)
        return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    data = np.abs(arr) if BACKGROUND_ABS else arr
    vmin = VMIN if VMIN is not None else float(np.nanmin(data))
    vmax = VMAX if VMAX is not None else float(np.nanpercentile(data, CLIP_PERCENTILE))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return Normalize(vmin=vmin, vmax=vmax), data


def render_frame(ds, bounds, cmap, fig_h):
    xlo, xhi, ylo, yhi = bounds
    extent = [xlo, xhi, ylo, yhi]

    fig = plt.figure(figsize=(FIG_W_INCHES, fig_h), facecolor=FACE_COLOR)
    ax = fig.add_axes([0, 0, 1, 1]) if not SHOW_AXES else fig.add_subplot(111)
    ax.set_facecolor(FACE_COLOR)

    # ---- background --------------------------------------------------------
    if BACKGROUND_MODE == "gradient":
        ref = sample(ds, "velocityx", bounds, FRB_RES)     # just for the shape
        rgba = vertical_gradient(ref.shape, cmap, GRADIENT_ANGLE_DEG)
        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal")
    else:
        arr = get_field(ds, BACKGROUND_FIELD, bounds, FRB_RES)
        norm_data = make_norm(arr)
        if BACKGROUND_SIGNED:
            norm, data = norm_data, arr
        else:
            norm, data = norm_data
        if GAMMA != 1.0 and not BACKGROUND_SIGNED:
            d = norm(data)
            d = np.clip(d, 0, 1) ** GAMMA
            ax.imshow(cmap(d), origin="lower", extent=extent, aspect="equal")
        else:
            ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                      cmap=cmap, norm=norm)

    # ---- red dye (passive eta tracer wrapping the ball) --------------------
    if SHOW_DYE:
        e = np.clip(np.nan_to_num(sample(ds, DYE_FIELD, bounds, FRB_RES), nan=0.0), 0.0, 1.0)
        dcm = LinearSegmentedColormap.from_list("dye", DYE_CMAP, N=256)
        rgba = dcm(np.clip(e, 0.0, 1.0) ** DYE_GAMMA)
        rgba[..., 3] = np.clip((e - DYE_FLOOR) / (1.0 - DYE_FLOOR + 1e-9), 0.0, 1.0) * DYE_ALPHA
        ax.imshow(rgba, origin="lower", extent=extent, aspect="equal", zorder=3)

    # ---- streamlines (FIXED seeds -> coherent, no frame-to-frame bobble) ---
    if SHOW_STREAMLINES:
        u = np.nan_to_num(sample(ds, "velocityx", bounds, FRB_RES), nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(sample(ds, "velocityy", bounds, FRB_RES), nan=0.0, posinf=0.0, neginf=0.0)
        ny, nx = u.shape
        x = np.linspace(xlo, xhi, nx); y = np.linspace(ylo, yhi, ny)
        spd = np.sqrt(u * u + v * v)
        lw = STREAM_LINEWIDTH
        if STREAM_LW_BY_SPEED:
            lw = STREAM_LINEWIDTH * (0.3 + 1.4 * spd / (np.nanpercentile(spd, 99) + 1e-9))
        start = None
        if STREAM_SEED_MODE == "inlet":
            xs = xlo + STREAM_SEED_INSET * (xhi - xlo)
            ys = np.linspace(ylo, yhi, STREAM_N_SEEDS + 2)[1:-1]
            start = np.column_stack([np.full_like(ys, xs), ys])
        elif STREAM_SEED_MODE == "grid":
            gx = np.linspace(xlo, xhi, STREAM_GRID_NX + 2)[1:-1]
            gy = np.linspace(ylo, yhi, STREAM_GRID_NY + 2)[1:-1]
            GX, GY = np.meshgrid(gx, gy)
            start = np.column_stack([GX.ravel(), GY.ravel()])
        kw = dict(density=STREAM_DENSITY, linewidth=lw, arrowsize=STREAM_ARROWSIZE,
                  arrowstyle="-" if STREAM_ARROWSIZE <= 0 else "->", zorder=8)
        if start is not None:
            kw["start_points"] = start
            kw["integration_direction"] = "forward"
        if STREAM_COLOR_BY_SPEED:
            sp = ax.streamplot(x, y, u, v, color=spd, cmap=cmap, **kw)
        else:
            sp = ax.streamplot(x, y, u, v, color=STREAM_COLOR, **kw)
        try: sp.lines.set_alpha(STREAM_ALPHA)
        except Exception: pass

    # ---- cylinder (shaded chrome sphere) -----------------------------------
    if SHOW_CYLINDER:
        phi = sample(ds, CYLINDER_FIELD, bounds, FRB_RES)
        ny, nx = phi.shape
        x = np.linspace(xlo, xhi, nx); y = np.linspace(ylo, yhi, ny)
        X, Y = np.meshgrid(x, y)
        inside = phi < CYLINDER_LEVEL
        if CYLINDER_STYLE == "chrome" and inside.any():
            cx = X[inside].mean(); cy = Y[inside].mean()
            dx = (xhi - xlo) / max(nx - 1, 1); dy = (yhi - ylo) / max(ny - 1, 1)
            r = max(np.sqrt(inside.sum() * dx * dy / np.pi), 1e-9)
            nxn = (X - cx) / r; nyn = (Y - cy) / r
            nz = np.sqrt(np.clip(1.0 - (nxn * nxn + nyn * nyn), 0.0, 1.0))
            L = np.array(CYLINDER_LIGHT, float); L /= (np.linalg.norm(L) + 1e-9)
            diff = np.clip(nxn * L[0] + nyn * L[1] + nz * L[2], 0.0, 1.0)
            g = np.clip(0.16 + 0.84 * diff ** 1.4 + 0.7 * diff ** 24, 0.0, 1.0)  # diffuse + specular hotspot
            rgba = np.zeros((ny, nx, 4))
            rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
            rgba[..., 3] = inside.astype(float)
            ax.imshow(rgba, origin="lower", extent=extent, aspect="equal", zorder=9)
        else:
            ax.contourf(X, Y, phi, levels=[-1e9, CYLINDER_LEVEL], colors=[CYLINDER_FILL], zorder=9)
        if CYLINDER_EDGE is not None:
            ax.contour(X, Y, phi, levels=[CYLINDER_LEVEL], colors=[CYLINDER_EDGE],
                       linewidths=CYLINDER_EDGE_LW, zorder=10)

    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    if not SHOW_AXES:
        ax.set_axis_off()
    if TITLE:
        ax.set_title(TITLE, color=TITLE_COLOR, fontsize=22, fontweight="bold", loc="left")
    return fig

# =============================================================================
# main
# =============================================================================

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    cmap = currents_cmap()
    pfs = find_plotfiles(AMREX_OUTPUT_DIR)[FRAME_START::FRAME_STRIDE]

    if FRAMES == "last":
        pfs = [pfs[-1]]
    elif FRAMES != "all":
        want = set(int(s) for s in FRAMES)
        pfs = [p for p in pfs if step_number(p) in want] or [pfs[-1]]

    print(f"Currents: {len(pfs)} frames from {AMREX_OUTPUT_DIR}")
    # consistent aspect ratio for all frames
    ds0 = yt.load(pfs[0]); b0 = ZOOM if ZOOM is not None else domain_bounds(ds0)
    fig_h = FIG_W_INCHES * (b0[3] - b0[2]) / (b0[1] - b0[0])

    frame_paths = []
    for i, pf in enumerate(pfs):
        ds = yt.load(pf)
        bounds = ZOOM if ZOOM is not None else domain_bounds(ds)
        fig = render_frame(ds, bounds, cmap, fig_h)
        out = os.path.join(IMAGE_DIR, f"{FRAME_PREFIX}{step_number(pf):06d}.png")
        fig.savefig(out, dpi=DPI, facecolor=FACE_COLOR)
        plt.close(fig)
        frame_paths.append(out)
        if (i + 1) % 10 == 0 or i == len(pfs) - 1:
            print(f"  frame {i+1}/{len(pfs)}  (t={float(ds.current_time):.3g})")

    if MAKE_GIF and len(frame_paths) > 1:
        idx = list(range(len(frame_paths)))
        if len(idx) > GIF_MAX_FRAMES:
            idx = list(np.linspace(0, len(idx) - 1, GIF_MAX_FRAMES, dtype=int))
        imgs = [Image.open(frame_paths[k]).convert("RGB") for k in idx]
        if GIF_BOOMERANG:
            imgs = imgs + imgs[-2:0:-1]
        imgs[0].save(GIF_PATH, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(GIF_FPS, 1)), loop=GIF_LOOP, optimize=True)
        print(f"  wrote {GIF_PATH}  ({len(imgs)} frames @ {GIF_FPS} fps)")

    if not SAVE_FRAMES:
        for p in frame_paths:
            try: os.remove(p)
            except Exception: pass
    print("Done.")


if __name__ == "__main__":
    main()
