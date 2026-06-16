#!/usr/bin/env python3
# ============================================================================
#  compare_hydro2.py
#  ---------------------------------------------------------------------------
#  Side-by-side comparison of the SINGLE-PHASE Hydro solver and the TWO-PHASE
#  six-equation Hydro2 solver for flow past an embedded cylinder
#  (tests/FlowVortexShed, Re = 40).  Produces:
#
#    (1) per-timestep PNG "overlay" frames: greyscale SOLID background
#        (black = solid, white = fluid) with the velocity_x = 0 contour of
#        BOTH solvers overlaid -- RED = Hydro, BLUE = Hydro2.
#    (2) a GIF assembled from those frames.
#    (3) time-series comparison charts (RED Hydro / BLUE Hydro2):
#          - max x of the velocity_x = 0 contour  (recirculation tail length)
#          - wake area enclosed by the velocity_x = 0 contour
#          - wake width (transverse extent of the recirculation)
#
#  Field conventions:
#     Hydro   : 'eta' is the SOLID indicator (1 fluid, 0 solid)
#     Hydro2  : 'phi' is the SOLID indicator (1 fluid, 0 solid); 'eta' = alpha_1
#     both    : 'velocityx' is the x-velocity
#
#  Everything is configurable in the CONFIG block below (house style, cf.
#  tests/FlowShockDroplet/refrence/shock_droplet_analysis.py).
#
#  Usage:
#     python3 compare_hydro2.py
#     python3 compare_hydro2.py <hydro_output_dir> <hydro2_output_dir>
# ============================================================================

import os
import re
import sys
import math
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import yt
yt.funcs.mylog.setLevel(50)   # silence yt

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False

# ============================================================================
# ============================  CONFIG  ======================================
# ============================================================================

# ---- INPUT / OUTPUT PATHS (relative to this script's directory) ------------
HYDRO_OUTPUT   = "../../../bin/tests/FlowVortexShed/output_hydro"            # single-phase Hydro plotfiles
HYDRO2_OUTPUT  = "../../../bin/tests/FlowVortexShed/output_hydro2"           # two-phase Hydro2 plotfiles
OUTPUT_DIR     = "./Images/Comparison"        # charts land here
FRAMES_DIR     = "./Images/Comparison/TailShapes"   # per-step overlay PNGs + gif
GIF_NAME       = "wake_overlay.gif"
FRAME_PREFIX   = "frame_"

# ---- MASTER TOGGLES --------------------------------------------------------
MAKE_FRAMES        = True    # render the per-timestep overlay PNGs
MAKE_GIF           = True    # assemble frames into a GIF (needs MAKE_FRAMES or existing frames)
PLOT_TAIL_LENGTH   = True    # max x(vx=0) vs time
PLOT_WAKE_AREA     = True    # enclosed wake area vs time
PLOT_WAKE_WIDTH    = True    # transverse wake width vs time
PLOT_SUMMARY_PANEL = True    # all three metrics stacked in one figure
WRITE_CSV          = True    # dump the metric time series to csv

# ---- FRAME SELECTION -------------------------------------------------------
FRAME_START   = 0            # first plotfile index to use
FRAME_STRIDE  = 1            # use every Nth plotfile
FRAME_END     = None         # last index (None = all)
IGNORE_TOKENS = [".old."]    # skip any plotfile whose name contains one of these

# ---- SAMPLING / DOMAIN -----------------------------------------------------
DOMAIN_AUTO   = True         # take the extent from the datasets
X_MIN, X_MAX  = -3.0, 12.0   # used only if DOMAIN_AUTO = False
Y_MIN, Y_MAX  = -3.0,  3.0
ZOOM          = None         # (xlo, xhi, ylo, yhi) to crop frames+metrics, or None
FRB_RES_X     = 1000         # fixed-resolution-buffer pixels in x (y from aspect)
SMOOTH_SIGMA  = 3.0          # Gaussian blur (in FRB pixels) applied to vx & the
                             # solid field BEFORE contouring/metrics.  De-blocks
                             # the vx=0 contour (sub-cell interpolation) AND
                             # smooths the time-series (tail/area/width change
                             # finely instead of jumping a cell at a time).
                             # 0 = off.  One coarse sim cell ~ dx/(domain/FRB_RES)
                             # pixels; ~half a cell is a good start.  Lower it for
                             # fine (max_level>=2) runs where cells are small.
SMOOTH_SOLID  = True         # also smooth the solid field (mask edge + bg).  Set
                             # False to keep the solid boundary at the raw cells.

# ---- SOLID / WAKE DEFINITION -----------------------------------------------
HYDRO_SOLID_FIELD  = "eta"   # Hydro solid indicator
HYDRO2_SOLID_FIELD = "phi"   # Hydro2 solid indicator
VELX_FIELD         = "velocityx"
SOLID_THRESHOLD    = 0.5     # cell is "solid" where indicator < this
CYL_R0             = 0.5     # cylinder radius (for the outline + tail reference)
CYL_X0, CYL_Y0     = 0.0, 0.0
WAKE_HALFWIDTH     = 2.5     # restrict vx=0 metrics to |y| < this (drop far field)
OUTFLOW_MARGIN     = 0.5     # ignore vx=0 within this distance of the x-outflow
TAIL_REFERENCE     = "absolute_x"   # 'absolute_x' = max x ; 'from_rear' = x - (X0+R0)
SUBTRACT_SOLID_AREA = True   # wake area = recirculation only (exclude the cylinder)

# ---- COLORS / SOLVER IDENTITY ----------------------------------------------
HYDRO_LABEL   = "Hydro (single-phase)"
HYDRO2_LABEL  = "Hydro2 (6-eqn)"
HYDRO_COLOR   = "#d11f1f"    # RED  -> Hydro
HYDRO2_COLOR  = "#1f5fd1"    # BLUE -> Hydro2

# ---- FRAME (overlay) STYLE -------------------------------------------------
BACKGROUND       = "solid"   # 'solid' greyscale background, or 'none'
BG_SOLVER        = "hydro2"  # which solid field to show: 'hydro' | 'hydro2'
BG_CMAP          = "gray"    # black = solid (0), white = fluid (1)
BG_VMIN, BG_VMAX = 0.0, 1.0
CONTOUR_LW       = 2.2       # vx=0 contour line width
CONTOUR_LEVEL    = 0.0       # velocity_x level to contour
DRAW_CYL_OUTLINE = False     # draw the analytic cylinder circle (green eta=0.5 line)
CYL_OUTLINE_COLOR = "#39ff14"
CYL_OUTLINE_LW   = 1.0

# ---- FIGURE / TYPOGRAPHY ---------------------------------------------------
FRAME_FIGSIZE   = (11.0, 4.6)
CHART_FIGSIZE   = (9.0, 5.4)
SUMMARY_FIGSIZE = (9.0, 11.0)
DPI             = 130
LINE_WIDTH      = 2.2
MARKER          = "o"
MARKER_SIZE     = 3.5
GRID_ALPHA      = 0.30
FONT_TITLE      = 15
FONT_LABEL      = 12.5
FONT_TICK       = 10.5
FONT_LEGEND     = 11

# ---- TITLES / AXIS LABELS --------------------------------------------------
FRAME_TITLE      = r"$v_x = 0$ wake contour   |   Hydro (red)  vs  Hydro2 (blue)"
TITLE_TAIL       = "Recirculation tail length vs time"
TITLE_AREA       = "Wake area vs time"
TITLE_WIDTH      = "Wake width vs time"
XLABEL_TIME      = "time  $t$"
YLABEL_TAIL      = r"max $x$ where $v_x=0$"
YLABEL_AREA      = "wake area enclosed by $v_x=0$"
YLABEL_WIDTH     = "wake width  (transverse extent)"

# ---- GIF -------------------------------------------------------------------
GIF_FPS        = 10
GIF_LOOP       = 0           # 0 = loop forever
GIF_BOOMERANG  = False       # play forward then backward

# ============================================================================
# ==========================  END CONFIG  ====================================
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
#  small utilities
# ---------------------------------------------------------------------------
def _abs(path):
    """Resolve a config path relative to this script's directory."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(SCRIPT_DIR, path))


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def step_number(name):
    """Leading integer in a plotfile name (for proper numeric sorting)."""
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def discover_plotfiles(output_dir):
    """Return sorted list of plotfile directories, skipping IGNORE_TOKENS."""
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"output directory not found: {output_dir}")
    items = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if not os.path.isdir(full):
            continue
        low = name.lower()
        if not ("cell" in low or "plt" in low):
            continue
        if any(tok in name for tok in IGNORE_TOKENS):
            continue
        items.append(full)
    items.sort(key=step_number)
    # frame selection
    items = items[FRAME_START:FRAME_END:FRAME_STRIDE] if FRAME_END is not None \
        else items[FRAME_START::FRAME_STRIDE]
    return items


def _read_field(frb, field):
    """Robust frb field access across yt field-name spellings."""
    for key in (field, ("boxlib", field), ("gas", field)):
        try:
            return np.array(frb[key])
        except Exception:
            continue
    raise KeyError(f"field '{field}' not found in plotfile")


def _smooth(a, sigma):
    """Gaussian blur (sigma in pixels) for sub-cell interpolation; nan-free input.
    Uses scipy if present, else a separable box-blur fallback (no scipy dep)."""
    if not sigma or sigma <= 0:
        return a
    a = np.asarray(a, dtype=float)
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(a, sigma=sigma, mode="nearest")
    except Exception:
        r = max(1, int(round(2.0 * sigma)))          # box radius ~2 sigma
        k = np.ones(2 * r + 1) / (2 * r + 1)
        out = a
        for axis in (0, 1):
            out = np.apply_along_axis(
                lambda m: np.convolve(m, k, mode="same"), axis, out)
        return out


def get_extent():
    """Resolve the sampling window (DOMAIN_AUTO / ZOOM aware)."""
    if ZOOM is not None:
        return tuple(ZOOM)
    return (X_MIN, X_MAX, Y_MIN, Y_MAX)


def sample(plotdir, solid_field, extent):
    """Load a plotfile and return a uniform sampling: vx, solid, X, Y, time."""
    ds = yt.load(plotdir)
    xlo, xhi, ylo, yhi = extent
    W, H = (xhi - xlo), (yhi - ylo)
    nx = int(FRB_RES_X)
    ny = max(1, int(round(nx * H / W)))
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    slc = ds.slice(2, 0.0)
    frb = slc.to_frb(width=(W, "code_length"), resolution=(nx, ny),
                     height=(H, "code_length"), center=[cx, cy, 0.0])
    vx    = _smooth(_read_field(frb, VELX_FIELD), SMOOTH_SIGMA)
    solid = _read_field(frb, solid_field)
    if SMOOTH_SOLID:
        solid = _smooth(solid, SMOOTH_SIGMA)
    X = np.linspace(xlo, xhi, nx)
    Y = np.linspace(ylo, yhi, ny)
    XX, YY = np.meshgrid(X, Y)
    return dict(vx=vx, solid=solid, X=X, Y=Y, XX=XX, YY=YY,
                time=float(ds.current_time), nx=nx, ny=ny, extent=extent)


# ---------------------------------------------------------------------------
#  wake metrics from a sample
# ---------------------------------------------------------------------------
def wake_metrics(s):
    """
    From a sampled field dict, compute:
      tail   : max x of the velocity_x = 0 contour (recirculation tail tip)
      area   : wake area enclosed by velocity_x = 0
      width  : transverse extent of the recirculation
    The recirculation region is { vx < 0 } in the fluid (solid is exactly 0).
    """
    vx, solid = s["vx"], s["solid"]
    XX, YY = s["XX"], s["YY"]
    xlo, xhi, ylo, yhi = s["extent"]
    dx = (xhi - xlo) / (s["nx"] - 1)
    dy = (yhi - ylo) / (s["ny"] - 1)

    fluid = solid > SOLID_THRESHOLD
    region = (np.abs(YY - CYL_Y0) < WAKE_HALFWIDTH) & (XX < (xhi - OUTFLOW_MARGIN)) \
        & (XX > (CYL_X0 - CYL_R0))
    recirc = (vx < 0.0) & fluid & region

    if not np.any(recirc):
        return dict(tail=np.nan, area=0.0, width=0.0, has_wake=False)

    xs = XX[recirc]
    ys = YY[recirc]
    tail_x = float(xs.max())
    if TAIL_REFERENCE == "from_rear":
        tail = tail_x - (CYL_X0 + CYL_R0)
    else:
        tail = tail_x

    area = float(recirc.sum()) * dx * dy
    if not SUBTRACT_SOLID_AREA:
        solid_in_region = (~fluid) & region
        area += float(solid_in_region.sum()) * dx * dy

    width = float(ys.max() - ys.min())
    return dict(tail=tail, area=area, width=width, has_wake=True)


def collect_metrics(plotfiles, solid_field, tag):
    """Sample every plotfile of one solver and gather the metric time series."""
    times, tails, areas, widths = [], [], [], []
    for k, pf in enumerate(plotfiles):
        try:
            s = sample(pf, solid_field, get_extent())
            m = wake_metrics(s)
            times.append(s["time"])
            tails.append(m["tail"])
            areas.append(m["area"])
            widths.append(m["width"])
        except Exception as exc:
            print(f"  [{tag}] WARNING: skipping {os.path.basename(pf)}: {exc}")
    print(f"  [{tag}] {len(times)} frames processed")
    return dict(t=np.array(times), tail=np.array(tails),
                area=np.array(areas), width=np.array(widths))


# ---------------------------------------------------------------------------
#  per-timestep overlay frame
# ---------------------------------------------------------------------------
def render_frame(s_h, s_h2, idx, out_path):
    """One overlay PNG: greyscale solid bg + both vx=0 contours (red/blue)."""
    extent = get_extent()
    xlo, xhi, ylo, yhi = extent
    fig, ax = plt.subplots(figsize=FRAME_FIGSIZE, dpi=DPI)

    # --- background: greyscale solid indicator ---
    if BACKGROUND == "solid":
        bg = s_h2["solid"] if BG_SOLVER == "hydro2" else s_h["solid"]
        ax.imshow(bg, origin="lower", extent=[xlo, xhi, ylo, yhi],
                  cmap=BG_CMAP, vmin=BG_VMIN, vmax=BG_VMAX,
                  aspect="equal", interpolation="bilinear", zorder=0)
    else:
        ax.set_facecolor("white")

    # --- vx = 0 contour, solid region masked out so only the wake shows ---
    def _contour(s, color):
        if s is None:
            return
        vx = np.array(s["vx"], dtype=float)
        fluid = s["solid"] > SOLID_THRESHOLD
        vx_masked = np.where(fluid, vx, np.nan)
        ax.contour(s["XX"], s["YY"], vx_masked, levels=[CONTOUR_LEVEL],
                   colors=[color], linewidths=CONTOUR_LW, zorder=5)

    _contour(s_h, HYDRO_COLOR)
    _contour(s_h2, HYDRO2_COLOR)

    # --- analytic cylinder outline ---
    if DRAW_CYL_OUTLINE:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(CYL_X0 + CYL_R0 * np.cos(th), CYL_Y0 + CYL_R0 * np.sin(th),
                color=CYL_OUTLINE_COLOR, lw=CYL_OUTLINE_LW, zorder=6)

    t = s_h2["time"] if s_h2 is not None else s_h["time"]
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x", fontsize=FONT_LABEL)
    ax.set_ylabel("y", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title(f"{FRAME_TITLE}\n$t = {t:.3f}$    (frame {idx})", fontsize=FONT_TITLE)

    legend = [Line2D([0], [0], color=HYDRO_COLOR, lw=CONTOUR_LW, label=HYDRO_LABEL),
              Line2D([0], [0], color=HYDRO2_COLOR, lw=CONTOUR_LW, label=HYDRO2_LABEL)]
    ax.legend(handles=legend, loc="upper right", fontsize=FONT_LEGEND, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def make_frames(hydro_pf, hydro2_pf, frames_dir):
    """Render paired overlay frames; returns the list of saved PNG paths."""
    ensure_dir(frames_dir)
    n = min(len(hydro_pf), len(hydro2_pf))
    if len(hydro_pf) != len(hydro2_pf):
        print(f"  NOTE: frame counts differ (Hydro {len(hydro_pf)}, "
              f"Hydro2 {len(hydro2_pf)}); pairing the first {n}.")
    paths = []
    for i in range(n):
        try:
            s_h = sample(hydro_pf[i], HYDRO_SOLID_FIELD, get_extent())
            s_h2 = sample(hydro2_pf[i], HYDRO2_SOLID_FIELD, get_extent())
        except Exception as exc:
            print(f"  WARNING: frame {i} skipped: {exc}")
            continue
        out = os.path.join(frames_dir, f"{FRAME_PREFIX}{i:05d}.png")
        render_frame(s_h, s_h2, i, out)
        paths.append(out)
        if (i % 10) == 0 or i == n - 1:
            print(f"  frame {i+1}/{n}  (t={s_h2['time']:.3f})")
    return paths


def make_gif(frame_paths, gif_path):
    """Assemble PNG frames into a looping GIF via PIL."""
    if not _HAVE_PIL:
        print("  GIF skipped: Pillow (PIL) not available.")
        return
    if not frame_paths:
        print("  GIF skipped: no frames.")
        return
    imgs = [Image.open(p).convert("RGB") for p in frame_paths]
    if GIF_BOOMERANG and len(imgs) > 2:
        imgs = imgs + imgs[-2:0:-1]
    dur_ms = int(round(1000.0 / max(GIF_FPS, 1)))
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=dur_ms, loop=GIF_LOOP, optimize=False, disposal=2)
    print(f"  wrote {gif_path}  ({len(imgs)} frames @ {GIF_FPS} fps)")


# ---------------------------------------------------------------------------
#  comparison charts
# ---------------------------------------------------------------------------
def _style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=FONT_LEGEND, framealpha=0.85)


def _plot_pair(ax, mh, mh2, key):
    ax.plot(mh["t"], mh[key], color=HYDRO_COLOR, lw=LINE_WIDTH,
            marker=MARKER, ms=MARKER_SIZE, label=HYDRO_LABEL)
    ax.plot(mh2["t"], mh2[key], color=HYDRO2_COLOR, lw=LINE_WIDTH,
            marker=MARKER, ms=MARKER_SIZE, label=HYDRO2_LABEL)


def chart(mh, mh2, key, title, ylabel, fname):
    fig, ax = plt.subplots(figsize=CHART_FIGSIZE, dpi=DPI)
    _plot_pair(ax, mh, mh2, key)
    _style_axis(ax, title, XLABEL_TIME, ylabel)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {fname}")


def summary_panel(mh, mh2, fname):
    fig, axes = plt.subplots(3, 1, figsize=SUMMARY_FIGSIZE, dpi=DPI, sharex=True)
    _plot_pair(axes[0], mh, mh2, "tail")
    _style_axis(axes[0], TITLE_TAIL, "", YLABEL_TAIL)
    _plot_pair(axes[1], mh, mh2, "area")
    _style_axis(axes[1], TITLE_AREA, "", YLABEL_AREA)
    _plot_pair(axes[2], mh, mh2, "width")
    _style_axis(axes[2], TITLE_WIDTH, XLABEL_TIME, YLABEL_WIDTH)
    fig.tight_layout()
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {fname}")


def write_csv(mh, mh2, fname):
    def _dump(path, m):
        with open(path, "w") as f:
            f.write("time,tail_length,wake_area,wake_width\n")
            for i in range(len(m["t"])):
                f.write(f"{m['t'][i]:.6f},{m['tail'][i]:.6f},"
                        f"{m['area'][i]:.6f},{m['width'][i]:.6f}\n")
    _dump(fname.replace(".csv", "_hydro.csv"), mh)
    _dump(fname.replace(".csv", "_hydro2.csv"), mh2)
    print(f"  wrote {fname.replace('.csv','_hydro.csv')} and _hydro2.csv")


def steady_state_summary(mh, mh2):
    """Print final-time metric values and the Hydro-vs-Hydro2 difference."""
    def _last(m, k):
        v = m[k]
        v = v[np.isfinite(v)]
        return v[-1] if len(v) else float("nan")
    print("\n  ---- final-frame comparison (steady-state estimate) ----")
    print(f"  {'metric':<18}{'Hydro':>12}{'Hydro2':>12}{'abs diff':>12}{'% diff':>10}")
    for k, name in (("tail", "tail length"), ("area", "wake area"), ("width", "wake width")):
        a, b = _last(mh, k), _last(mh2, k)
        d = abs(a - b)
        pct = 100.0 * d / abs(b) if (b == b and b != 0) else float("nan")
        print(f"  {name:<18}{a:>12.4f}{b:>12.4f}{d:>12.4f}{pct:>9.1f}%")


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------
def main():
    hydro_dir  = _abs(sys.argv[1]) if len(sys.argv) > 1 else _abs(HYDRO_OUTPUT)
    hydro2_dir = _abs(sys.argv[2]) if len(sys.argv) > 2 else _abs(HYDRO2_OUTPUT)
    out_dir    = _abs(OUTPUT_DIR)
    frames_dir = _abs(FRAMES_DIR)

    print("compare_hydro2")
    print(f"  Hydro  : {hydro_dir}")
    print(f"  Hydro2 : {hydro2_dir}")

    # -- error checking + dir creation --
    missing = [d for d in (hydro_dir, hydro2_dir) if not os.path.isdir(d)]
    if missing:
        print("ERROR: missing output directory(ies):")
        for d in missing:
            print(f"   - {d}")
        print("Run the sims first (Re40_input_hydro -> output_hydro, "
              "Re40_input_hydro2 -> output_hydro2).")
        sys.exit(1)
    ensure_dir(out_dir)
    ensure_dir(frames_dir)

    hydro_pf  = discover_plotfiles(hydro_dir)
    hydro2_pf = discover_plotfiles(hydro2_dir)
    print(f"  found {len(hydro_pf)} Hydro / {len(hydro2_pf)} Hydro2 plotfiles")
    if not hydro_pf or not hydro2_pf:
        print("ERROR: no plotfiles found in one or both directories.")
        sys.exit(1)

    # -- (1)(2) per-timestep overlay frames + GIF --
    frame_paths = []
    if MAKE_FRAMES:
        print("\n[frames] rendering wake overlays ...")
        frame_paths = make_frames(hydro_pf, hydro2_pf, frames_dir)
    if MAKE_GIF:
        print("\n[gif] assembling ...")
        if not frame_paths and os.path.isdir(frames_dir):
            frame_paths = sorted(
                os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
                if f.startswith(FRAME_PREFIX) and f.endswith(".png"))
        # GIF lives in the main Comparison folder; frames stay in TailShapes.
        make_gif(frame_paths, os.path.join(out_dir, GIF_NAME))

    # -- (3) metric time series + charts --
    need_metrics = (PLOT_TAIL_LENGTH or PLOT_WAKE_AREA or PLOT_WAKE_WIDTH
                    or PLOT_SUMMARY_PANEL or WRITE_CSV)
    if need_metrics:
        print("\n[metrics] sampling wake metrics ...")
        mh  = collect_metrics(hydro_pf,  HYDRO_SOLID_FIELD,  "Hydro")
        mh2 = collect_metrics(hydro2_pf, HYDRO2_SOLID_FIELD, "Hydro2")

        if PLOT_TAIL_LENGTH:
            chart(mh, mh2, "tail", TITLE_TAIL, YLABEL_TAIL,
                  os.path.join(out_dir, "tail_length_vs_time.png"))
        if PLOT_WAKE_AREA:
            chart(mh, mh2, "area", TITLE_AREA, YLABEL_AREA,
                  os.path.join(out_dir, "wake_area_vs_time.png"))
        if PLOT_WAKE_WIDTH:
            chart(mh, mh2, "width", TITLE_WIDTH, YLABEL_WIDTH,
                  os.path.join(out_dir, "wake_width_vs_time.png"))
        if PLOT_SUMMARY_PANEL:
            summary_panel(mh, mh2, os.path.join(out_dir, "wake_summary.png"))
        if WRITE_CSV:
            write_csv(mh, mh2, os.path.join(out_dir, "wake_metrics.csv"))
        steady_state_summary(mh, mh2)

    print("\nDone.")


if __name__ == "__main__":
    main()
