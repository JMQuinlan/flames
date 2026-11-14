#!/usr/bin/env python3
# ============================================================================
#  T-38 TALON -- NUMERICAL SCHLIEREN SLICE IMAGERY  (3D)
# ============================================================================
#  Renders the FINAL plotfile of a 3D plane run as numerical-schlieren (and
#  friends) slices, sweeping the slice plane along each axis and writing both
#  the individual frames and an animated GIF.
#
#      X-Slice  ->  plane normal to x, sweeping FRONT  -> BACK
#      Y-Slice  ->  plane normal to y, sweeping LEFT   -> RIGHT
#      Z-Slice  ->  plane normal to z, sweeping BOTTOM -> UP
#
#  The airframe (phi = 0.5 contour) is drawn SOLID BLACK on every frame: the
#  solid region is filled black and the isoline stroked black on top, so the
#  aircraft silhouette reads as a single opaque body rather than a coloured
#  blob of whatever the field happens to be doing inside it.
#
#  Output layout (all under ./Images/ relative to THIS file):
#
#      Images/
#        X-Slice/
#          X-Slice-Schlieren/   frame_0000.png ... + X-Slice-Schlieren.gif
#          X-Slice-Pressure/    ...
#          ...
#        Y-Slice/ ...
#        Z-Slice/ ...
#        Overview/              single-shot multi-panel summaries
#
#  Only the FINAL plotfile is used, and any path containing ".old." is ignored.
#
# ---------------------------------------------------------------------------
#  NOTE ON REUSE
#  The block marked "REUSABLE ANALYSIS PRIMITIVES" below is deliberately free
#  of module globals: every function takes what it needs as arguments and
#  returns plain numpy.  It is written to be lifted verbatim into a shared
#  analysis class/module later without edits -- the same compute_schlieren,
#  slice_frb, find_plotfiles and write_gif are what the ShockDroplet, Marmottant
#  and DrivenBubble scripts each re-implement today.
# ---------------------------------------------------------------------------
#  EXPERIMENT COMPARISON (Images/Comparison/T38_BOS_Comparison.png):
#  the numerical solution rendered like the background-oriented schlieren (BOS)
#  photograph reference/T38C_BOS_NASA.jpg (NASA; Wikipedia/Wikimedia file
#  "Shockwave pattern around a T-38C observed with Background-Oriented Schlieren
#  photography"; flight Mach number and altitude NOT checked against a source),
#  as ONE split image (aircraft flying up; numerical left of the fuselage
#  centreline, experiment right).  BOS records the light deflection integrated along the
#  line of sight, i.e. the gradient of the PATH-INTEGRATED density, as light/dark
#  bands on a grey background.  The photo looks down on the planform, so the
#  model image is int rho dz, differentiated along the flight direction,
#  mirrored about the y = 0 symmetry plane, and resampled onto the photo's pixel
#  aircraft frame together with the photo (both rotated to nose-up, same metres
#  per pixel; the aircraft position and heading are measured on the photo).  Its grey scale is matched to the photo: the photo's
#  background grey for zero gradient and its dark/light percentiles for the
#  strongest compressions/expansions.
#
#  Usage (from this reference folder):
#      python3 plane_schlieren_analysis.py
#      OUT_DIR=/mmfs1/.../output_Ma1.1_Wide python3 plane_schlieren_analysis.py
#      BOS_ONLY=1 python3 plane_schlieren_analysis.py     # just the comparison figure
#      BOS_EXP_SIDE=left  ...                              # experiment on the left half
# ============================================================================

import os
import sys
import glob
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# ==============================  CONFIG  ====================================
# ============================================================================

# Plotfile directory.  Env OUT_DIR wins, so the same script serves the local
# validation run and the big INCLINE run without editing.
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "..", "..", "bin", "tests", "FlowTryonFun",
        "Planes", "T38-Talon", "output_Ma1.1_Wide")))

_INCLINE_OUT = "/mmfs1/home/ttryon/flames/bin/tests/FlowTryonFun/Planes/T38-Talon/output_Ma1.1_Wide"
if "OUT_DIR" not in os.environ and os.path.isdir(_INCLINE_OUT):
    OUT_DIR = _INCLINE_OUT

IMG_ROOT = os.path.join(_HERE, "Images")

# ----- solid / airframe -----
PHI_FIELD      = "phi"
PHI_LEVEL      = 0.5
SOLID_IS_LOW   = True     # phi < 0.5 is solid (verified: fluid phi~1, solid phi~0)
SOLID_COLOR    = "black"  # the airframe is drawn completely black
SOLID_EDGE_LW  = 1.2

# ----- slice sweep -----
N_FRAMES       = int(os.environ.get("N_FRAMES", 48))   # frames per axis sweep
SWEEP_TRIM     = 0.02     # skip this fraction at each domain end (avoid BC cells)
RESOLUTION     = int(os.environ.get("RESOLUTION", 900))  # px along the long image axis

# Restrict the sweep to the interesting region (None -> full domain extent).
# The airframe lives in x[-6,7] y[0,3.5] z[-1.5,1.5]; sweeping the whole
# 56x16x32 box wastes most frames on undisturbed freestream.
SWEEP_LIMITS = {
    "x": (-12.0, 20.0),
    "y": (0.0,  6.0),
    "z": (-6.0, 6.0),
}

# ----- what to render per slice -----
# name: (field or derived key, colormap, log?, symmetric?)
QUANTITIES = [
    ("Schlieren",    "schlieren",   "gray",    False, False),
    ("SchlierenP",   "schlieren_p", "gray",    False, False),
    ("Pressure",     "pressure",    "inferno", False, False),
    ("Mach",         "mach",        "turbo",   False, False),
    ("Density",      "density",     "viridis", False, False),
    ("VorticityMag", "vortmag",     "magma",   True,  False),
    ("Temperature",  "T",           "plasma",  False, False),
]
ENABLE = {q[0]: 1 for q in QUANTITIES}   # flip to 0 to skip one

# ----- schlieren -----
# Two things made AMR boxes visible; both are handled:
#  1. SAMPLER.  yt's slice pixelizer (to_frb) mis-fills pixels on AMR grid
#     boundaries (O(1) value errors -> gradient spikes up to 600x true).  Slices
#     are now uniform covering_grids -- see slice_frb.  This is the real fix.
#  2. SCALING.  Shocks are thin lines (<1% of pixels), so the reference
#     gradient must be a very high percentile, and a small floor removes the
#     remaining coarse-level stair-steps (<=4x the local smooth gradient).
SCHLIEREN_MODE       = "exp"     # exp: exp(-beta g/g_ref), white bg / dark shocks
                                 # log: log(1 + g/g_floor) normalised to [0,1]
SCHLIEREN_BETA       = 6.0
SCHLIEREN_REF_PCT    = 99.9      # percentile of FLUID |grad| used as g_ref (thin shocks)
SCHLIEREN_FLOOR_FRAC = 0.03      # |grad| < FLOOR_FRAC*g_ref -> 0
SCHLIEREN_SMOOTH_PX  = 0.0       # optional gaussian pre-smoothing (off: not needed)
SLICE_LEVEL          = (int(os.environ["SLICE_LEVEL"]) if "SLICE_LEVEL" in os.environ
                        else None)   # None -> auto from RESOLUTION; set to cap memory
SCHLIEREN_CLIP_PCT   = (1.0, 99.5)   # percentile clip for the NON-schlieren fields
# Log-scaled fields (VorticityMag): never show more than this many decades below
# the peak.  A 1st-percentile floor let ~1e-10 roundoff vorticity at AMR
# coarse/fine faces render as a clear outline of every box.
LOG_DECADES_FLOOR    = 1.0e-4

# ----- BOS experiment comparison -----
BOS_PHOTO      = os.path.join(_HERE, "T38C_BOS_NASA.jpg")
BOS_EXP_SIDE   = os.environ.get("BOS_EXP_SIDE", "right")   # experiment half of the split image; the model takes the other
BOS_AHEAD, BOS_BEHIND, BOS_HALF_WIDTH = 8.0, 26.0, 20.0    # split-image window in the aircraft frame [m]
BOS_ONLY       = int(os.environ.get("BOS_ONLY", 0))
BOS_LEVEL      = (int(os.environ["BOS_LEVEL"]) if "BOS_LEVEL" in os.environ else None)   # AMR level of the projection
BOS_DX         = 0.25      # None level -> the coarsest level with dx <= BOS_DX [m] (0.25 m = shock level of the new deck)
# Aircraft in the photo (full-resolution pixel coords): nose and tail of the model's centreline
# from a least-squares similarity fit (scale, rotation, shift) of six planform landmarks of
# t38_clean_xflow.STL to the photo -- nose tip, nozzle exit, both wing-tip and both tailplane-tip
# chord midpoints, read on the photo (2026-11-14): (552,181) (603,437) (518.5,369) (661,346)
# (562,422.5) (642.7,408).  Fit: 0.05347 m/px (photo span then 7.72 m = the STL's), heading
# ~11 deg from vertical, landmark residuals 0.16 m rms (~3 px).
BOS_PHOTO_NOSE = (554.20, 180.19)
BOS_PHOTO_TAIL = (605.55, 439.70)
# Aircraft length between those two points (the cleaned STL: x = -7.0725 .. 7.0725 m).  The
# model image is anchored at the solver airframe's TAIL; this length places the nose.
BOS_AIRCRAFT_LENGTH = 14.145
BOS_DARK_PCT, BOS_LIGHT_PCT = 0.5, 99.5     # photo grey percentiles the strongest model gradients map to

# ----- output -----
DPI            = 140
GIF_MS         = 90      # per-frame duration
MAKE_GIF       = True
MAKE_OVERVIEW  = True
FIG_PAD        = 0.02

# ============================================================================
# ==================  REUSABLE ANALYSIS PRIMITIVES  ==========================
#  (no module globals used below -- ready to lift into a shared class)
# ============================================================================

def find_plotfiles(out_dir, exclude_substrings=(".old.",)):
    """Every *cell plotfile in out_dir, numerically sorted by step number.
    Entries whose path contains any exclude substring are dropped."""
    cands = sorted(glob.glob(os.path.join(out_dir, "*cell")))
    keep = []
    for p in cands:
        if any(s in os.path.basename(p) or s in p for s in exclude_substrings):
            continue
        if not os.path.isdir(p):
            continue
        keep.append(p)

    def _step(p):
        b = os.path.basename(p).replace("cell", "")
        try:
            return int(b)
        except ValueError:
            return -1
    return sorted(keep, key=_step)


def gradient_magnitude_2d(field2d, dx, dy, smooth_px=0.0):
    """|grad| of a rasterised 2D field.  Gaussian pre-smoothing (in pixels)
    suppresses the stair-steps a coarse AMR level leaves in a fine raster."""
    f = np.asarray(field2d, dtype=float)
    if smooth_px and smooth_px > 0:
        try:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, smooth_px, mode="nearest")
        except ImportError:
            pass
    d0 = np.gradient(f, dy, axis=0)
    d1 = np.gradient(f, dx, axis=1)
    return np.sqrt(d0 ** 2 + d1 ** 2)


def schlieren_reference(grads, masks=None, pct=99.0):
    """g_ref = pct-th percentile of |grad| over the (masked) pixels of one or
    more frames.  Pass fluid-only masks so solid walls don't set the scale."""
    vals = []
    for i, g in enumerate(grads):
        g = np.asarray(g, dtype=float)
        m = np.isfinite(g) & (g > 0)
        if masks is not None and masks[i] is not None:
            m &= masks[i]
        if m.any():
            vals.append(g[m])
    if not vals:
        return 0.0
    return float(np.percentile(np.concatenate(vals), pct))


def compute_schlieren(grad, g_ref, mode="exp", beta=6.0, floor_frac=0.03):
    """Map a gradient-magnitude field to a [0,1] schlieren image.

    grad       : |grad(rho)| (or any scalar) -- native AMR or rasterised
    g_ref      : reference gradient (see schlieren_reference)
    mode "exp" : exp(-beta g/g_ref)  -> 1 = undisturbed (white), 0 = shock (black)
    mode "log" : 1 - log(1+g/g_floor)/log(1+g_ref/g_floor), clipped to [0,1]
    floor_frac : gradients below floor_frac*g_ref are treated as zero; this is
                 what removes AMR seams and roundoff, which sit orders of
                 magnitude below real shock gradients.
    Generalises tests/FlowShockDroplet/refrence/shock_droplet_analysisPlay.py.
    """
    g = np.nan_to_num(np.asarray(grad, dtype=float), nan=0.0, posinf=0.0)
    if not np.isfinite(g_ref) or g_ref <= 0:
        return np.ones_like(g)
    floor = floor_frac * g_ref
    g = np.maximum(g - floor, 0.0)
    ref = max(g_ref - floor, 1e-300)
    if mode == "log":
        gf = max(floor, 1e-3 * g_ref)
        return 1.0 - np.clip(np.log1p(g / gf) / np.log1p(ref / gf), 0.0, 1.0)
    return np.exp(-beta * g / ref)


def slice_frb(ds, axis, coord, fields, resolution=900, level=None):
    """Axis-aligned slice of a 3D AMR dataset on a uniform grid.

    Extracted as a ONE-CELL-THICK yt covering_grid at a single AMR level --
    NOT with slice().to_frb().  yt's slice pixelizer mis-fills pixels at AMR
    grid boundaries: on an exactly linear test field it left ~1% of pixels
    with O(1) value errors, and differencing them gave gradient spikes up to
    600x the true gradient -- the "every AMR box is outlined" schlieren
    artefact.  covering_grid injects coarse data piecewise-constant under the
    chosen level (worst case a 4x stair-step in coarse regions, i.e. the true
    local resolution) and has no boundary defects.

    level      : AMR level to sample at; None -> the coarsest level whose long
                 image axis has >= 0.9*resolution cells, capped at max_level.
    Returns (data_dict, extent, hlabel, vlabel); data[f] is (nv, nh) for
    imshow(origin='lower').  data["_dx"] = (dh, dv) cell sizes.
    """
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    names = ["x", "y", "z"]
    NATURAL = {0: (1, 2),   # x-normal cross-section: y horizontal, z vertical
               1: (0, 2),   # y-normal side view    : x horizontal, z vertical
               2: (0, 1)}   # z-normal top view     : x horizontal, y vertical
    hi_ax, vi_ax = NATURAL[ai]

    le = np.array(ds.domain_left_edge.to_value(), dtype=float)
    re = np.array(ds.domain_right_edge.to_value(), dtype=float)
    base = np.array(ds.domain_dimensions, dtype=int)
    rr = int(getattr(ds, "refine_by", 2))

    if level is None:
        level = int(ds.max_level)
        for L in range(int(ds.max_level) + 1):
            if base[[hi_ax, vi_ax]].max() * rr ** L >= 0.9 * resolution:
                level = L
                break
    level = int(min(max(level, 0), ds.max_level))

    dims = base * rr ** level
    dx = (re - le) / dims
    k = int(np.clip(np.floor((float(coord) - le[ai]) / dx[ai]), 0, dims[ai] - 1))
    # THREE cells thick, middle layer kept.  A one-cell-thick covering_grid is
    # silently wrong in yt 4.4: it ignores left_edge along the thin axis and
    # returns the domain's FIRST cell layer for every requested coordinate
    # (measured: x=+0.5, z=-8 and z=+9 all came back at -15.75).
    nthk = int(min(3, dims[ai]))
    lo = int(np.clip(k - 1, 0, dims[ai] - nthk))
    left = le.copy(); left[ai] = le[ai] + lo * dx[ai]
    cdims = dims.copy(); cdims[ai] = nthk

    keys = [f if isinstance(f, tuple) else ("boxlib", f) for f in fields]
    cg = ds.covering_grid(level, left_edge=left, dims=cdims, fields=keys)

    out = {}
    for key in keys:
        a = np.asarray(cg[key], dtype=float)            # (nx, ny, nz)
        a2 = np.take(a, k - lo, axis=ai)                 # requested layer, ascending axis order
        # remaining axes are (min(hi,vi), max(hi,vi)); want (vi, hi)
        out[key[1]] = a2.T if hi_ax < vi_ax else a2
    out["_dx"] = (float(dx[hi_ax]), float(dx[vi_ax]))
    out["_level"] = level
    out["_coord"] = float(le[ai] + (k + 0.5) * dx[ai])   # actual sampled plane

    extent = [le[hi_ax], re[hi_ax], le[vi_ax], re[vi_ax]]
    return out, extent, names[hi_ax], names[vi_ax]

def write_gif(png_paths, gif_path, duration_ms=90):
    """Assemble PNG frames into an animated GIF (PIL; imageio not required)."""
    if not png_paths:
        return False
    try:
        from PIL import Image
    except ImportError:
        print("    [warn] PIL unavailable -- no GIF written")
        return False
    frames = [Image.open(p).convert("P", palette=Image.ADAPTIVE)
              for p in png_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    for f in frames:
        f.close()
    return True


def robust_range(a, pct=(1.0, 99.5)):
    """Percentile range ignoring NaN/inf; falls back to (0,1) when degenerate."""
    f = np.asarray(a, dtype=float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(f, pct[0]), np.percentile(f, pct[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)

# ============================================================================
# ===========================  DERIVED FIELDS  ===============================
# ============================================================================

BASE_FIELDS = ["density", "pressure", "phi", "T", "gamma", "p0",
               "velocityx", "velocityy", "velocityz",
               "vorticityx", "vorticityy", "vorticityz", "a"]


def derive(key, d, dh, dv):
    """Build a display field from the raw slice dict."""
    if key in ("schlieren", "schlieren_p"):
        # returns |grad| on the uniform slice; caller maps it with a sweep-wide g_ref
        src = "density" if key == "schlieren" else "pressure"
        return gradient_magnitude_2d(d[src], dh, dv, SCHLIEREN_SMOOTH_PX)
    if key == "mach":
        u = np.sqrt(d["velocityx"] ** 2 + d["velocityy"] ** 2 + d["velocityz"] ** 2)
        # The plotfile 'a' field is zero in ~half the cells at t=0 (it is a
        # diagnostic filled during the RHS, not an IC), which turned the whole
        # Mach panel into NaN.  Recompute from the EOS instead and keep 'a'
        # only as a fallback: a = sqrt(gamma (p + p0) / rho).
        a = None
        if all(k in d for k in ("gamma", "pressure", "density")):
            p0 = d.get("p0", 0.0)
            arg = d["gamma"] * (d["pressure"] + p0) / np.maximum(d["density"], 1e-30)
            a = np.sqrt(np.maximum(arg, 0.0))
        if a is None or not np.any(a > 1e-12):
            a = d.get("a")
        if a is None:
            return np.full_like(u, np.nan)
        return u / np.where(a > 1e-12, a, np.nan)
    if key == "vortmag":
        return np.sqrt(d["vorticityx"] ** 2 + d["vorticityy"] ** 2
                       + d["vorticityz"] ** 2)
    return d[key]

# ============================================================================
# =============================  RENDERING  ==================================
# ============================================================================

def draw_solid(ax, phi, extent, level=PHI_LEVEL, low_is_solid=SOLID_IS_LOW,
               stroke=True):
    """Airframe, completely black: EVERY pixel with phi <= level is painted.

    The solid is rasterised directly as an opaque RGBA layer rather than
    filled with contourf.  contourf interpolates a binary mask, and a solid
    only 1-2 pixels thick (a wing or fin cut edge-on in an x- or z-slice)
    yields no closed contour -- the body silently vanished from those slices.
    Direct painting cannot lose a solid pixel.  The phi = level isoline is
    stroked on top only as a cosmetic edge; the fill does not depend on it.
    """
    if phi is None:
        return
    phi = np.asarray(phi, dtype=float)
    solid = (phi <= level) if low_is_solid else (phi >= level)
    solid &= np.isfinite(phi)
    if not solid.any():
        return
    from matplotlib.colors import to_rgb
    rgba = np.zeros(phi.shape + (4,), dtype=float)
    rgba[..., :3] = to_rgb(SOLID_COLOR)
    rgba[..., 3] = solid.astype(float)
    ax.imshow(rgba, origin="lower", extent=extent, interpolation="nearest",
              aspect=ax.get_aspect(), zorder=5)
    if stroke:
        try:
            ax.contour(phi, levels=[level], colors=[SOLID_COLOR],
                       linewidths=SOLID_EDGE_LW, extent=extent, origin="lower",
                       zorder=6)
        except Exception:
            pass

def render_frame(field, phi, extent, hlabel, vlabel, title, cmap, out_png,
                 vmin=None, vmax=None, log=False, cbar_label=""):
    fig, ax = plt.subplots(figsize=(10.5, 10.5 * _aspect(extent)))
    arr = np.array(field, dtype=float)
    if log:
        pos = arr[np.isfinite(arr) & (arr > 0)]
        floor = (max(np.percentile(pos, 1.0), LOG_DECADES_FLOOR * np.percentile(pos, 99.9))
                     if pos.size else 1e-12)
        arr = np.log10(np.maximum(arr, floor))
        if vmin is not None:
            vmin = math.log10(max(vmin, floor))
        if vmax is not None:
            vmax = math.log10(max(vmax, floor * 10))
    im = ax.imshow(arr, origin="lower", extent=extent, cmap=cmap,
                   norm=Normalize(vmin=vmin, vmax=vmax), aspect="equal",
                   interpolation="bilinear")
    draw_solid(ax, phi, extent)
    ax.set_xlabel(f"{hlabel}  [m]", fontsize=11)
    ax.set_ylabel(f"{vlabel}  [m]", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(cbar_label or title, fontsize=10)
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight", pad_inches=FIG_PAD)
    plt.close(fig)


def _aspect(extent):
    w = abs(extent[1] - extent[0]); h = abs(extent[3] - extent[2])
    return max(0.18, min(2.4, h / w if w > 0 else 1.0))

# ============================================================================
# ================================  MAIN  ====================================
# ============================================================================

SWEEPS = [("x", "X-Slice", "front -> back"),
          ("y", "Y-Slice", "left -> right"),
          ("z", "Z-Slice", "bottom -> up")]


def main():
    import yt
    yt.set_log_level(50)

    print("=" * 78)
    print("T-38 TALON -- SCHLIEREN SLICE IMAGERY")
    print("=" * 78)
    print(f"  plotfile dir : {OUT_DIR}")

    pfs = find_plotfiles(OUT_DIR)
    if not pfs:
        print("  [error] no usable plotfiles (after dropping '*.old.*'). Nothing to do.")
        return 1
    final = pfs[-1]
    print(f"  plotfiles    : {len(pfs)} usable, using FINAL = {os.path.basename(final)}")

    ds = yt.load(final)
    le = np.array(ds.domain_left_edge.to_value(), dtype=float)
    re = np.array(ds.domain_right_edge.to_value(), dtype=float)
    print(f"  t            = {float(ds.current_time):.6g}")
    print(f"  domain       = x[{le[0]:g},{re[0]:g}] y[{le[1]:g},{re[1]:g}] z[{le[2]:g},{re[2]:g}]")
    print(f"  max_level    = {ds.max_level},  base = {ds.domain_dimensions}")
    have = {str(f[1]) for f in ds.field_list}
    fields = [f for f in BASE_FIELDS if f in have]
    missing = [f for f in BASE_FIELDS if f not in have]
    print("  slices       = uniform covering_grid (no yt pixelizer: AMR-seam free)")
    if missing:
        print(f"  [note] absent fields skipped: {', '.join(missing)}")
    print(f"  frames/axis  = {N_FRAMES},  resolution = {RESOLUTION}px")
    os.makedirs(IMG_ROOT, exist_ok=True)
    if BOS_ONLY:
        make_bos_comparison(ds)
        return 0

    active = [q for q in QUANTITIES if ENABLE.get(q[0], 0)]

    for axis, sweep_name, direction in SWEEPS:
        ai = {"x": 0, "y": 1, "z": 2}[axis]
        lo_d, hi_d = float(le[ai]), float(re[ai])
        lo_s, hi_s = SWEEP_LIMITS.get(axis, (lo_d, hi_d))
        lo = max(lo_d, lo_s); hi = min(hi_d, hi_s)
        span = hi - lo
        lo += SWEEP_TRIM * span; hi -= SWEEP_TRIM * span
        coords = np.linspace(lo, hi, N_FRAMES)

        print(f"\n  --- {sweep_name}  ({direction})   {axis} = {lo:.3f} .. {hi:.3f}, "
              f"{N_FRAMES} frames")

        sweep_dir = os.path.join(IMG_ROOT, sweep_name)
        os.makedirs(sweep_dir, exist_ok=True)

        # First pass: gather slices once, reuse for every quantity.
        slices = []
        for ci, c in enumerate(coords):
            try:
                d, extent, hl, vl = slice_frb(ds, axis, c, fields, RESOLUTION, SLICE_LEVEL)
            except Exception as e:
                print(f"      [warn] slice {axis}={c:.3f} failed: {e}")
                continue
            slices.append((c, d, extent, hl, vl))
        if not slices:
            print("      [warn] no slices extracted -- skipping this axis.")
            continue
        extent = slices[0][2]; hl = slices[0][3]; vl = slices[0][4]
        dh, dv = slices[0][1]["_dx"]
        print(f"      sampled at AMR level {slices[0][1]['_level']}  "
              f"({slices[0][1]['density'].shape[1]} x {slices[0][1]['density'].shape[0]} cells, "
              f"dx = {dh:.4g} m)")

        for qname, qkey, cmap, qlog, _sym in active:
            qdir = os.path.join(sweep_dir, f"{sweep_name}-{qname}")
            os.makedirs(qdir, exist_ok=True)

            # Common colour scale across the sweep so the GIF does not flicker.
            samp = []
            for c, d, ext, _h, _v in slices[:: max(1, len(slices) // 8)]:
                try:
                    samp.append(derive(qkey, d, dh, dv))
                except Exception:
                    pass
            if not samp:
                print(f"      [warn] {qname}: cannot derive -- skipped")
                continue
            is_sch = qkey in ("schlieren", "schlieren_p")
            if is_sch:
                sub = slices[:: max(1, len(slices) // 8)]
                masks = [(d.get(PHI_FIELD) > PHI_LEVEL) if (SOLID_IS_LOW and d.get(PHI_FIELD) is not None)
                         else None for _c, d, _e, _h, _v in sub]
                g_ref = schlieren_reference(samp, masks, SCHLIEREN_REF_PCT)
                vmin, vmax = 0.0, 1.0
            else:
                vmin, vmax = robust_range(np.concatenate([s.ravel() for s in samp]),
                                          SCHLIEREN_CLIP_PCT)

            pngs = []
            for fi, (c, d, ext, _h, _v) in enumerate(slices):
                try:
                    fld = derive(qkey, d, dh, dv)
                    if is_sch:
                        fld = compute_schlieren(fld, g_ref, SCHLIEREN_MODE,
                                                SCHLIEREN_BETA, SCHLIEREN_FLOOR_FRAC)
                except Exception as e:
                    print(f"      [warn] {qname} frame {fi}: {e}")
                    continue
                png = os.path.join(qdir, f"frame_{fi:04d}.png")
                render_frame(fld, d.get(PHI_FIELD), ext, _h, _v,
                             f"{qname}   {axis} = {c:+.3f} m   (t = {float(ds.current_time):.4g})",
                             cmap, png, vmin=vmin, vmax=vmax, log=qlog,
                             cbar_label=qname)
                pngs.append(png)

            msg = f"      {qname:13s} {len(pngs):3d} frames"
            if MAKE_GIF and pngs:
                gif = os.path.join(qdir, f"{sweep_name}-{qname}.gif")
                if write_gif(pngs, gif, GIF_MS):
                    msg += f"  + {os.path.basename(gif)}"
            print(msg)

    if MAKE_OVERVIEW:
        make_overview(ds, fields)
    make_bos_comparison(ds)

    print("\n  done.  images under:", IMG_ROOT)
    return 0


def make_overview(ds, fields):
    """Multi-panel single-shot summaries through the airframe centreline."""
    import itertools
    ov = os.path.join(IMG_ROOT, "Overview")
    os.makedirs(ov, exist_ok=True)
    print("\n  --- Overview panels")

    # centre-plane slices: y=0 (symmetry plane), z=0, x at mid-body
    picks = [("y", 0.05, "symmetry plane (y~0)"),
             ("z", 0.00, "horizontal plane (z=0)"),
             ("x", 0.50, "cross-section (x=0.5)")]
    for axis, c, desc in picks:
        try:
            d, extent, hl, vl = slice_frb(ds, axis, c, fields, RESOLUTION, SLICE_LEVEL)
        except Exception as e:
            print(f"      [warn] overview {axis}={c}: {e}")
            continue
        dh, dv = d["_dx"]
        panels = [("Schlieren", "schlieren", "gray", False),
                  ("Pressure", "pressure", "inferno", False),
                  ("Mach", "mach", "turbo", False),
                  ("VorticityMag", "vortmag", "magma", True)]
        fig, axes = plt.subplots(2, 2, figsize=(17, 10.5))
        for ax, (nm, key, cm, lg) in zip(axes.ravel(), panels):
            try:
                fld = derive(key, d, dh, dv)
            except Exception:
                ax.set_visible(False); continue
            arr = np.array(fld, dtype=float)
            if key in ("schlieren", "schlieren_p"):
                ph = d.get(PHI_FIELD)
                m = (ph > PHI_LEVEL) if (SOLID_IS_LOW and ph is not None) else None
                g_ref = schlieren_reference([arr], [m], SCHLIEREN_REF_PCT)
                arr = compute_schlieren(arr, g_ref, SCHLIEREN_MODE,
                                        SCHLIEREN_BETA, SCHLIEREN_FLOOR_FRAC)
                vmin, vmax = 0.0, 1.0
            else:
                if lg:
                    pos = arr[np.isfinite(arr) & (arr > 0)]
                    floor = (max(np.percentile(pos, 1.0), LOG_DECADES_FLOOR * np.percentile(pos, 99.9))
                     if pos.size else 1e-12)
                    arr = np.log10(np.maximum(arr, floor))
                vmin, vmax = robust_range(arr, SCHLIEREN_CLIP_PCT)
            im = ax.imshow(arr, origin="lower", extent=extent, cmap=cm,
                           norm=Normalize(vmin, vmax), aspect="equal",
                           interpolation="bilinear")
            draw_solid(ax, d.get(PHI_FIELD), extent)
            ax.set_title(nm, fontsize=11, fontweight="bold")
            ax.set_xlabel(f"{hl} [m]", fontsize=9)
            ax.set_ylabel(f"{vl} [m]", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        fig.suptitle(f"T-38  {desc}   t = {float(ds.current_time):.4g}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        sgn = "p" if c >= 0 else "m"
        p = os.path.join(ov, f"Overview-{axis}-{sgn}{abs(c):.2f}.png")
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"      wrote {os.path.basename(p)}")


def bos_projection(ds, level=BOS_LEVEL):
    """Path-integrated density int rho dz on the x-y plane (uniform grid at `level`), its
    derivative along the flow (+x, the flight direction reversed), and the projected solid.
    Returns dict(x, y, P, dPdx, solid) with y mirrored to [-Y, Y] (symmetry plane y = 0)."""
    le = np.array(ds.domain_left_edge.to_value(), float); re = np.array(ds.domain_right_edge.to_value(), float)
    rr = int(getattr(ds, "refine_by", 2)); dx0 = float(((re - le) / np.array(ds.domain_dimensions, int)).max())
    if level is None:
        level = next((l for l in range(ds.max_level + 1) if dx0 / rr ** l <= BOS_DX * (1 + 1e-9)), ds.max_level)
    level = int(min(level, ds.max_level))
    dims = np.array(ds.domain_dimensions, int) * rr ** level
    dx = (re - le) / dims
    cg = ds.covering_grid(level, left_edge=le, dims=dims, fields=[("boxlib", "density"), ("boxlib", PHI_FIELD)])
    rho = np.asarray(cg["boxlib", "density"], float)            # (nx, ny, nz)
    phi = np.asarray(cg["boxlib", PHI_FIELD], float)
    P = (np.where(phi > PHI_LEVEL, rho, 0.0) * dx[2]).sum(axis=2)  # fluid only: the solid's own density is not flow
    solid = (phi <= PHI_LEVEL).any(axis=2)
    dPdx = np.gradient(P, dx[0], axis=0)
    dPdx[solid] = 0.0
    x = le[0] + (np.arange(dims[0]) + 0.5) * dx[0]
    y = le[1] + (np.arange(dims[1]) + 0.5) * dx[1]
    if abs(le[1]) < 1e-9 * max(1.0, re[1]):                        # half domain: mirror about y = 0
        y = np.concatenate([-y[::-1], y])
        P = np.concatenate([P[:, ::-1], P], axis=1); dPdx = np.concatenate([dPdx[:, ::-1], dPdx], axis=1)
        solid = np.concatenate([solid[:, ::-1], solid], axis=1)
    out = dict(x=x, y=y, P=P, dPdx=dPdx, solid=solid, dx=dx, level=level)
    # airframe silhouette at the FINEST level (the coarse level misses the thin nose tip and
    # tail, which set the aircraft length and so the image scale)
    if solid.any():
        L = int(ds.max_level)
        # region = bounding box of the finest-level grids that hold solid (phi <= PHI_LEVEL).  The
        # coarse level misses thin wings and fins, so its solid cannot bound the region; and in runs
        # that refined the shocks to the finest level (e.g. the old wide set-up) the finest grids
        # fill most of the domain, so only grids containing solid are used.
        while True:
            gl = [g for g in ds.index.grids if g.Level == L and float(np.min(g["boxlib", PHI_FIELD])) <= PHI_LEVEL]
            if not gl: L -= 1; continue
            blo = np.min([g.LeftEdge.d for g in gl], axis=0); bhi = np.max([g.RightEdge.d for g in gl], axis=0)
            dxf = (re - le) / (np.array(ds.domain_dimensions, int) * int(getattr(ds, "refine_by", 2)) ** L)
            i0 = np.maximum(np.floor((blo - le) / dxf + 1e-6).astype(int), 0)
            i1 = np.minimum(np.ceil((bhi - le) / dxf - 1e-6).astype(int), np.round((re - le) / dxf).astype(int))
            if np.prod(i1 - i0) <= 6e7 or L <= level: break
            L -= 1                                                    # memory cap (~0.5 GB per field)
        cgf = ds.covering_grid(L, left_edge=le + i0 * dxf, dims=i1 - i0, fields=[("boxlib", PHI_FIELD)])
        sf = (np.asarray(cgf["boxlib", PHI_FIELD], float) <= PHI_LEVEL).any(axis=2)
        xf = le[0] + (np.arange(i0[0], i1[0]) + 0.5) * dxf[0]; yf = le[1] + (np.arange(i0[1], i1[1]) + 0.5) * dxf[1]
        if abs(le[1]) < 1e-9 * max(1.0, re[1]):
            yf = np.concatenate([-yf[::-1], yf]); sf = np.concatenate([sf[:, ::-1], sf], axis=1)
        xsf = xf[sf.any(axis=1)]
        out.update(xf=xf, yf=yf, solid_f=sf, x_nose=float(xsf.min() - 0.5 * dxf[0]), x_tail=float(xsf.max() + 0.5 * dxf[0]),
                   level_f=L)
    return out


def make_bos_comparison(ds):
    """One split image in the aircraft frame (flying up): numerical BOS-style projection on one
    side of the fuselage centreline, the NASA BOS photograph on the other, same scale and grey scale."""
    from PIL import Image
    from scipy.ndimage import map_coordinates
    if not os.path.isfile(BOS_PHOTO):
        print(f"  [warn] no BOS photo at {BOS_PHOTO}: comparison skipped"); return
    print("\n  --- BOS comparison with the experiment")
    photo = np.asarray(Image.open(BOS_PHOTO).convert("L"), float) / 255.0     # (rows, cols), 0 black .. 1 white
    H, W = photo.shape
    nose = np.array(BOS_PHOTO_NOSE); tail = np.array(BOS_PHOTO_TAIL)
    pr = bos_projection(ds)
    if "x_nose" not in pr:
        print("  [warn] no solid in the projection: comparison skipped"); return
    x_tail = pr["x_tail"]; x_nose = x_tail - BOS_AIRCRAFT_LENGTH
    print(f"      solver airframe x = [{pr['x_nose']:.2f}, {x_tail:.2f}] m ({x_tail - pr['x_nose']:.2f} m long; "
          f"scale from the {BOS_AIRCRAFT_LENGTH} m aircraft length, anchored at the tail)")
    m_per_px = BOS_AIRCRAFT_LENGTH / float(np.linalg.norm(tail - nose))
    t_hat = (tail - nose) / np.linalg.norm(tail - nose)       # photo px direction nose -> tail (downstream)
    n_hat = np.array([t_hat[1], -t_hat[0]])                   # photo direction that becomes image-right when the nose points up
    # output grid in the aircraft frame: s = distance behind the nose (down the image), w = lateral (right +)
    h = m_per_px
    s_ax = np.arange(-BOS_AHEAD, BOS_BEHIND, h); w_ax = np.arange(-BOS_HALF_WIDTH, BOS_HALF_WIDTH, h) + 0.5 * h
    Wg, Sg = np.meshgrid(w_ax, s_ax)
    # photo sample
    pc = nose[0] - 0.5 + (Sg * t_hat[0] + Wg * n_hat[0]) / m_per_px
    prr = nose[1] - 0.5 + (Sg * t_hat[1] + Wg * n_hat[1]) / m_per_px
    in_photo = (pc >= 0) & (pc <= W - 1) & (prr >= 0) & (prr <= H - 1)
    ph = map_coordinates(photo, [prr, pc], order=1, mode="nearest")
    # model sample (symmetric about y = 0, so the sign of w does not matter)
    X = x_nose + Sg; Y = Wg
    x, y = pr["x"], pr["y"]
    fi = (X - x[0]) / (x[1] - x[0]); fj = (Y - y[0]) / (y[1] - y[0])
    in_dom = (fi >= 0) & (fi <= len(x) - 1) & (fj >= 0) & (fj <= len(y) - 1)
    g = map_coordinates(pr["dPdx"], [fi, fj], order=1, mode="nearest")
    xf, yf = pr["xf"], pr["yf"]
    sol = map_coordinates(pr["solid_f"].astype(float), [(X - xf[0]) / (xf[1] - xf[0]), (Y - yf[0]) / (yf[1] - yf[0])],
                          order=1, mode="constant", cval=0.0) > 0.5
    # grey scale from the photo: background = histogram mode, extremes = percentiles
    hist, edges = np.histogram(photo, bins=256, range=(0, 1))
    bg = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
    dark, light = np.percentile(photo, [BOS_DARK_PCT, BOS_LIGHT_PCT])
    gi = g[in_dom & ~sol]; big = np.abs(gi) > 1e-3 * max(np.abs(gi).max() if gi.size else 0.0, 1e-300)
    gpos = np.percentile(gi[big & (gi > 0)], BOS_LIGHT_PCT) if np.any(big & (gi > 0)) else 1.0
    gneg = np.percentile(-gi[big & (gi < 0)], BOS_LIGHT_PCT) if np.any(big & (gi < 0)) else 1.0
    # compression (dP/dx > 0 across a shock, flow +x) -> dark, expansion -> light, as in the photo
    num = np.where(g > 0, bg - (bg - dark) * np.clip(g / gpos, 0, 1), bg + (light - bg) * np.clip(-g / gneg, 0, 1))
    num[~in_dom] = bg
    num[sol] = dark
    ph[~in_photo] = 1.0                                         # outside the photograph: white
    left_is_model = BOS_EXP_SIDE == "right"
    left = Wg < 0
    img = np.where(left == left_is_model, num, ph)
    fig, ax = plt.subplots(figsize=(10, 10 * len(s_ax) / len(w_ax) + 0.9))
    ext = (w_ax[0] - 0.5 * h, w_ax[-1] + 0.5 * h, s_ax[-1] + 0.5 * h, s_ax[0] - 0.5 * h)   # s grows downward
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="bilinear", extent=ext)
    ax.axvline(0.0, color="w", lw=1.2, ls="--")
    # computational-domain edge on the model half
    try:
        ax.contour(w_ax, s_ax, np.where(left == left_is_model, in_dom, True).astype(float), [0.5],
                   colors=["0.3"], linewidths=1.0, linestyles=":")
    except Exception:
        pass
    lab_l, lab_r = ("Numerical", "Experiment (NASA BOS)") if left_is_model else ("Experiment (NASA BOS)", "Numerical")
    kw = dict(transform=ax.transAxes, va="top", fontsize=15, fontweight="bold",
              bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3))
    ax.text(0.015, 0.985, lab_l, ha="left", **kw); ax.text(0.985, 0.985, lab_r, ha="right", **kw)
    ax.plot([w_ax[0] + 2.0, w_ax[0] + 12.0], [s_ax[-1] - 2.0] * 2, color="k", lw=3)
    ax.text(w_ax[0] + 7.0, s_ax[-1] - 2.6, "10 m", ha="center", va="bottom", fontsize=13)
    ax.set_xlabel("Lateral Distance from the Fuselage Centreline [m]", fontsize=14)
    ax.set_ylabel("Distance Behind the Nose [m]", fontsize=14)
    ax.tick_params(labelsize=13)
    t = float(ds.current_time)
    ax.set_title(f"T-38 Shock Pattern, Numerical vs Experiment\n(Numerical: $M_\\infty$ = 1.1, $t$ = {t:.3f} s)", fontsize=15)
    fig.tight_layout()
    od = os.path.join(IMG_ROOT, "Comparison"); os.makedirs(od, exist_ok=True)
    out = os.path.join(od, "T38_BOS_Comparison")
    fig.savefig(out + ".png", dpi=DPI); fig.savefig(out + ".pdf"); plt.close(fig)
    print(f"      wrote {out}.png  (AMR level {pr['level']}, silhouette level {pr['level_f']}, {m_per_px:.4f} m/px, "
          f"photo background grey {bg:.3f}; model half {100 * in_dom[left == left_is_model].mean():.0f}% inside the domain, "
          f"photo half {100 * in_photo[left != left_is_model].mean():.0f}% inside the photo)")


if __name__ == "__main__":
    sys.exit(main())
