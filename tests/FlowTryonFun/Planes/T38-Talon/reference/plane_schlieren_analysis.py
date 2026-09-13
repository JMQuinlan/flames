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
#  Usage (from this reference folder):
#      python3 plane_schlieren_analysis.py
#      OUT_DIR=/mmfs1/.../output_Ma1.1_Wide python3 plane_schlieren_analysis.py
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
    ("Schlieren",    "schlieren",   "gray_r",  False, False),
    ("SchlierenP",   "schlieren_p", "gray_r",  False, False),
    ("Pressure",     "pressure",    "inferno", False, False),
    ("Mach",         "mach",        "turbo",   False, False),
    ("Density",      "density",     "viridis", False, False),
    ("VorticityMag", "vortmag",     "magma",   True,  False),
    ("Temperature",  "T",           "plasma",  False, False),
]
ENABLE = {q[0]: 1 for q in QUANTITIES}   # flip to 0 to skip one

# ----- schlieren -----
SCHLIEREN_BETA      = 10.0
SCHLIEREN_LOG_SCALE = True     # log10(1+100|grad|) -- same form as FlowShockDroplet
SCHLIEREN_CLIP_PCT  = (1.0, 99.5)   # percentile clip for display range

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


def compute_schlieren(field2d, dx, dy, beta=10.0, log_scale=True):
    """Numerical schlieren from a 2D scalar field (classically density).

    log_scale : log10(1 + 100|grad|)  -- wide dynamic range, shows weak waves
    else      : exp(-beta |grad|/max) -- classic dark-on-light shadowgraph
    Same formulation as tests/FlowShockDroplet/refrence/shock_droplet_analysisPlay.py
    """
    d0 = np.gradient(field2d, dx, axis=0)
    d1 = np.gradient(field2d, dy, axis=1)
    g = np.sqrt(d0 ** 2 + d1 ** 2)
    if log_scale:
        return np.log10(1.0 + 100.0 * g)
    gmax = np.nanmax(g)
    return np.exp(-beta * g / gmax) if gmax > 0 else np.ones_like(g)


def slice_frb(ds, axis, coord, fields, resolution=900):
    """Axis-aligned slice of a 3D (AMR) dataset rasterized to a uniform array.

    Returns (data_dict, extent, hlabel, vlabel) with data[field] shaped
    (nv, nh) and extent = [h0, h1, v0, v1] suitable for imshow(origin='lower').
    Resolution is the pixel count on the LONGER image axis; the other is scaled
    to keep square pixels.
    """
    ai = {"x": 0, "y": 1, "z": 2}[axis]
    names = ["x", "y", "z"]
    # yt's own image axes (x_axis={0:1,1:2,2:0}) put the y-normal view on its
    # side: z horizontal, x vertical.  For a vehicle flying along +x the
    # readable convention is nose-to-tail HORIZONTAL, so use a fixed natural
    # mapping and transpose whenever it disagrees with yt.
    NATURAL = {0: (1, 2),   # x-normal cross-section: y horizontal, z vertical
               1: (0, 2),   # y-normal side view    : x horizontal, z vertical
               2: (0, 1)}   # z-normal top view     : x horizontal, y vertical
    hi_ax, vi_ax = NATURAL[ai]
    yt_h = ds.coordinates.x_axis[ai]
    yt_v = ds.coordinates.y_axis[ai]
    need_T = (yt_h, yt_v) != (hi_ax, vi_ax)

    le = np.array(ds.domain_left_edge.to_value(), dtype=float)
    re = np.array(ds.domain_right_edge.to_value(), dtype=float)
    w = float(re[hi_ax] - le[hi_ax])
    h = float(re[vi_ax] - le[vi_ax])

    yw = float(re[yt_h] - le[yt_h])
    yh = float(re[yt_v] - le[yt_v])
    if yw >= yh:
        nh = int(resolution); nv = max(8, int(round(resolution * yh / yw)))
    else:
        nv = int(resolution); nh = max(8, int(round(resolution * yw / yh)))

    center = (le + re) / 2.0
    center[ai] = float(coord)

    slc = ds.slice(ai, float(coord), center=ds.arr(center, "code_length"))
    frb = slc.to_frb(width=(yw, "code_length"), resolution=(nh, nv),
                     height=(yh, "code_length"),
                     center=ds.arr(center, "code_length"))

    out = {}
    for f in fields:
        a2 = np.array(frb[("boxlib", f)], dtype=float)
        out[f] = a2.T if need_T else a2

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
    if key == "schlieren":
        return compute_schlieren(d["density"], dh, dv,
                                 SCHLIEREN_BETA, SCHLIEREN_LOG_SCALE)
    if key == "schlieren_p":
        return compute_schlieren(d["pressure"], dh, dv,
                                 SCHLIEREN_BETA, SCHLIEREN_LOG_SCALE)
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

def draw_solid(ax, phi, extent, level=PHI_LEVEL, low_is_solid=SOLID_IS_LOW):
    """Airframe, completely black: filled interior + stroked phi=0.5 isoline."""
    if phi is None or not np.any(np.isfinite(phi)):
        return
    # contourf needs an increasing-level list; mask the solid side.
    solid = (phi < level) if low_is_solid else (phi > level)
    if solid.any():
        ax.contourf(solid.astype(float), levels=[0.5, 1.5],
                    colors=[SOLID_COLOR], extent=extent, origin="lower")
    try:
        ax.contour(phi, levels=[level], colors=[SOLID_COLOR],
                   linewidths=SOLID_EDGE_LW, extent=extent, origin="lower")
    except Exception:
        pass


def render_frame(field, phi, extent, hlabel, vlabel, title, cmap, out_png,
                 vmin=None, vmax=None, log=False, cbar_label=""):
    fig, ax = plt.subplots(figsize=(10.5, 10.5 * _aspect(extent)))
    arr = np.array(field, dtype=float)
    if log:
        pos = arr[np.isfinite(arr) & (arr > 0)]
        floor = np.percentile(pos, 1.0) if pos.size else 1e-12
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
    if missing:
        print(f"  [note] absent fields skipped: {', '.join(missing)}")
    print(f"  frames/axis  = {N_FRAMES},  resolution = {RESOLUTION}px")
    os.makedirs(IMG_ROOT, exist_ok=True)

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
                d, extent, hl, vl = slice_frb(ds, axis, c, fields, RESOLUTION)
            except Exception as e:
                print(f"      [warn] slice {axis}={c:.3f} failed: {e}")
                continue
            slices.append((c, d, extent, hl, vl))
        if not slices:
            print("      [warn] no slices extracted -- skipping this axis.")
            continue
        extent = slices[0][2]; hl = slices[0][3]; vl = slices[0][4]
        dh = (extent[1] - extent[0]) / slices[0][1]["density"].shape[1]
        dv = (extent[3] - extent[2]) / slices[0][1]["density"].shape[0]

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
            vmin, vmax = robust_range(np.concatenate([s.ravel() for s in samp]),
                                      SCHLIEREN_CLIP_PCT)

            pngs = []
            for fi, (c, d, ext, _h, _v) in enumerate(slices):
                try:
                    fld = derive(qkey, d, dh, dv)
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
            d, extent, hl, vl = slice_frb(ds, axis, c, fields, RESOLUTION)
        except Exception as e:
            print(f"      [warn] overview {axis}={c}: {e}")
            continue
        dh = (extent[1] - extent[0]) / d["density"].shape[1]
        dv = (extent[3] - extent[2]) / d["density"].shape[0]
        panels = [("Schlieren", "schlieren", "gray_r", False),
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
            if lg:
                pos = arr[np.isfinite(arr) & (arr > 0)]
                floor = np.percentile(pos, 1.0) if pos.size else 1e-12
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


if __name__ == "__main__":
    sys.exit(main())
