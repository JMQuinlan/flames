#!/usr/bin/env python3
# ============================================================================
#  impact_schlieren.py
#  ---------------------------------------------------------------------------
#  Numerical-schlieren animation of FlowTryonFun/Impact: a Ma 10, 15 deg
#  half-angle wedge (apex down) hitting a flat air/water interface.  The run is
#  in the WEDGE FRAME (air + water stream up at U), on the half domain x >= 0
#  with a symmetry plane at x = 0.  Each frame is mirrored about x = 0 to show
#  the whole wedge.
#
#  Overlays (all toggled in CONFIG):
#     - the solid wedge (phi), grey shading + outline
#     - the water surface (eta = 0.5 contour)
#     - steady oblique-shock theory from the apex (drawn downstream, +y): the AIR shock before impact
#       (Ma 10, gamma 1.4 -> beta 19.9 deg), the WATER shock after the surface
#       passes the apex (Ma 2.24, stiffened gas gamma 2.35 -> beta 53.2 deg)
#     - the undisturbed water surface position y = U t
#
#  Outputs (to ./Images):
#     impact_schlieren.gif        the animation
#     impact_schlieren_last.png   the last frame at full resolution
#
#  Usage:
#     python3 impact_schlieren.py                 # plotfiles from OUTPUT_DIR in CONFIG
#     python3 impact_schlieren.py <plotfile_dir>  # explicit plotfile dir
#     python3 impact_schlieren.py <plotfile_dir> <image_dir>
#
#  Frames are rendered in parallel (NPROC) to PNGs, then stitched into the gif,
#  so memory stays flat however many frames there are.
# ============================================================================

import os
import re
import sys
import glob
import shutil
import tempfile
from multiprocessing import Pool

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

import yt
yt.funcs.mylog.setLevel(50)

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(SCRIPT_DIR, "../../../FlowWedge/reference")))
import oblique_shock_theory as ost  # noqa: E402

# ============================================================================
# ==============================  CONFIG  ====================================
# ============================================================================

# ---- CASE (must match impact_Impact) --------------------------------------
OUTPUT_DIR    = "/mmfs1/home/ttryon/flames/bin/tests/FlowTryonFun/Impact/output_Impact"
U_STREAM      = 3432.49        # m/s, air + water stream up (+y) in the wedge frame
WEDGE_THETA_DEG = 15.0         # half-angle
WEDGE_YA      = 0.03           # apex y
WEDGE_YB      = 0.05           # base y
WATER_Y0      = 0.0            # water surface at t = 0
MACH_AIR      = 10.0
GAMMA_AIR     = 1.4
MACH_WATER    = U_STREAM / 1533.0   # c_w = sqrt(2.35 (p + 1e9) / 1000) = 1533 m/s
GAMMA_WATER   = 2.35          # stiffened gas: same theta-beta-M relation with p -> p + p0

# ---- I/O ------------------------------------------------------------------
IMAGE_DIR     = "./Images"
GIF_NAME      = "impact_schlieren.gif"
LAST_PNG_NAME = "impact_schlieren_last.png"
IGNORE_TOKENS = [".old."]
FRAME_STRIDE  = 1              # use every Nth plotfile
MAX_FRAMES    = 200            # cap after striding (evenly thinned)
NPROC         = 16             # parallel frame renderers

# ---- SAMPLING / VIEW ------------------------------------------------------
DOMAIN_AUTO   = True           # full domain from the plotfile
X_MAX_VIEW    = 0.05           # used when DOMAIN_AUTO = False
Y_MIN_VIEW, Y_MAX_VIEW = -0.03, 0.09
MIRROR        = True           # mirror about the symmetry plane x = 0
FRB_RES_X     = 500            # samples across the half width (doubled when mirrored)

# ---- SCHLIEREN ------------------------------------------------------------
# 'lnrho' : |grad ln rho|  (shows air shocks and water shocks together; the
#           830:1 density jump at the surface would swamp plain |grad rho|)
# 'rho'   : |grad rho|     (classic; water features dominate)
SCHLIEREN_FIELD = "lnrho"
SCHLIEREN_EXP   = 8.0          # exp(-k g / g_ref) contrast
SCHLIEREN_PCT   = 99.0         # g_ref = this percentile of g in the fluid (per frame) ...
SCHLIEREN_REF   = None         # ... or a fixed g_ref (1/m) to stop frame-to-frame flicker
CMAP_SCHLIEREN  = "gray"       # black gradients on white
MASK_SOLID      = True         # blank the gradient where phi < PHI_MASK (solid + its
PHI_MASK        = 0.95         #   Brinkman edge, which otherwise swamps the image)

# ---- OVERLAYS / COLORS ----------------------------------------------------
SHOW_PHI_OVERLAY  = True
PHI_OVERLAY_ALPHA = 0.75
SHOW_WEDGE_OUTLINE = True
WEDGE_COLOR       = "#00b8d4"
PHI_LEVEL         = 0.5
SHOW_SURFACE      = True       # eta = 0.5 contour (water surface)
SURFACE_COLOR     = "#1f6feb"
ETA_LEVEL         = 0.5
SHOW_SURFACE_REF  = True       # undisturbed surface y = U t (where it is still flat)
SURFACE_REF_COLOR = "#8b949e"
SHOW_THEORY       = True       # steady attached-shock lines from the apex
AIR_SHOCK_COLOR   = "#e8590c"
WATER_SHOCK_COLOR = "#2f9e44"
WEDGE_LW, SURF_LW, SHOCK_LW = 1.4, 1.3, 1.6

# ---- FIGURE / TYPOGRAPHY --------------------------------------------------
FIGSIZE    = (6.4, 7.6)
DPI        = 110               # gif frames
DPI_LAST   = 180               # last-frame png
GIF_FPS    = 10
FONT_TITLE = 13
FONT_LABEL = 11.5
FONT_TICK  = 10
LENGTH_UNIT = ("mm", 1.0e3)    # axis unit label and scale from metres
TITLE_TMPL = "Ma 10 wedge water impact   t = {t:.2f} $\\mu$s"

# ============================================================================
# ============================  END CONFIG  ==================================
# ============================================================================


def _abs(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(SCRIPT_DIR, p))


def step_number(name):
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def discover_plotfiles(d):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"plotfile dir not found: {d}")
    items = [os.path.join(d, n) for n in os.listdir(d)
             if os.path.isdir(os.path.join(d, n))
             and ("cell" in n.lower() or "plt" in n.lower())
             and os.path.isfile(os.path.join(d, n, "Header"))
             and not any(t in n for t in IGNORE_TOKENS)]
    items.sort(key=step_number)
    items = items[::max(1, FRAME_STRIDE)]
    if MAX_FRAMES and len(items) > MAX_FRAMES:
        idx = np.unique(np.linspace(0, len(items) - 1, MAX_FRAMES).round().astype(int))
        items = [items[i] for i in idx]
    return items


def _read(frb, field):
    for key in (field, ("boxlib", field), ("gas", field)):
        try:
            return np.array(frb[key])
        except Exception:
            continue
    return None


def sample(plotdir):
    ds = yt.load(plotdir)
    le, re_ = ds.domain_left_edge, ds.domain_right_edge
    if DOMAIN_AUTO:
        xlo, xhi, ylo, yhi = float(le[0]), float(re_[0]), float(le[1]), float(re_[1])
    else:
        xlo, xhi, ylo, yhi = float(le[0]), X_MAX_VIEW, Y_MIN_VIEW, Y_MAX_VIEW
    W, H = xhi - xlo, yhi - ylo
    nx = int(FRB_RES_X)
    ny = max(1, int(round(nx * H / W)))
    frb = ds.slice(2, 0.0).to_frb(width=(W, "code_length"), height=(H, "code_length"),
                                  resolution=(nx, ny),
                                  center=[0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.0])
    rho = _read(frb, "density")
    if rho is None:
        raise KeyError(f"'density' not in {plotdir}")
    phi = _read(frb, "phi")
    eta = _read(frb, "eta")
    return dict(rho=rho, phi=np.ones_like(rho) if phi is None else phi, eta=eta,
                has_phi=phi is not None, extent=(xlo, xhi, ylo, yhi), nx=nx, ny=ny,
                time=float(ds.current_time))


def mirror(a):
    """Half domain x >= 0 -> full domain (columns are x)."""
    return np.concatenate([a[:, ::-1], a], axis=1)


def schlieren(s):
    xlo, xhi, ylo, yhi = s["extent"]
    dx = (xhi - xlo) / s["nx"]
    dy = (yhi - ylo) / s["ny"]
    f = np.log(np.maximum(s["rho"], 1e-30)) if SCHLIEREN_FIELD == "lnrho" else s["rho"]
    gy, gx = np.gradient(f, dy, dx)
    g = np.hypot(gx, gy)
    fluid = s["phi"] > PHI_MASK
    if MASK_SOLID and s["has_phi"]:
        g = np.where(fluid, g, 0.0)
    if SCHLIEREN_REF is not None:
        gref = float(SCHLIEREN_REF)
    else:
        pool = g[fluid] if np.any(fluid) else g.ravel()
        gref = float(np.percentile(pool, SCHLIEREN_PCT)) if pool.size else 1.0
    return np.exp(-SCHLIEREN_EXP * g / max(gref, 1e-30)), gref


def theory():
    b_air = ost.beta_deg(WEDGE_THETA_DEG, MACH_AIR, GAMMA_AIR, weak=True)
    b_wat = ost.beta_deg(WEDGE_THETA_DEG, MACH_WATER, GAMMA_WATER, weak=True)
    return b_air, b_wat


def shock_line(beta_deg, y_lo, y_hi):
    """Attached shock from the apex.  The flow is +y, so the shock runs DOWNSTREAM (+y) from the
    apex at beta to the wedge axis.  Returns (x, y) of the right branch for y in [y_lo, y_hi]."""
    if beta_deg is None:
        return None
    y_lo = max(y_lo, WEDGE_YA)
    if y_hi <= y_lo:
        return None
    y = np.linspace(y_lo, y_hi, 60)
    x = (y - WEDGE_YA) * np.tan(np.radians(beta_deg))
    return x, y


def render(task):
    pf, out_png, dpi = task
    s = sample(pf)
    sch, gref = schlieren(s)
    phi, eta = s["phi"], s["eta"]
    xlo, xhi, ylo, yhi = s["extent"]
    if MIRROR:
        sch, phi = mirror(sch), mirror(phi)
        eta = mirror(eta) if eta is not None else None
        xlo_v = -xhi
    else:
        xlo_v = xlo
    unit, sc = LENGTH_UNIT
    ext = [xlo_v * sc, xhi * sc, ylo * sc, yhi * sc]
    nxv = sch.shape[1]
    X = np.linspace(ext[0], ext[1], nxv)
    Y = np.linspace(ext[2], ext[3], sch.shape[0])

    t = s["time"]
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=dpi)
    ax.imshow(sch, origin="lower", extent=ext, cmap=CMAP_SCHLIEREN, vmin=0.0, vmax=1.0,
              aspect="equal", interpolation="bilinear", zorder=0)

    handles = []
    if SHOW_PHI_OVERLAY and s["has_phi"]:
        tt = np.linspace(0.0, 1.0, 256)
        lut = plt.get_cmap("gray")(tt * 0.35)          # solid drawn dark grey
        lut[:, 3] = PHI_OVERLAY_ALPHA * (1.0 - tt)
        ax.imshow(np.clip(phi, 0, 1), origin="lower", extent=ext, cmap=ListedColormap(lut),
                  vmin=0, vmax=1, aspect="equal", interpolation="bilinear", zorder=1)
    if SHOW_WEDGE_OUTLINE and s["has_phi"]:
        ax.contour(X, Y, phi, levels=[PHI_LEVEL], colors=[WEDGE_COLOR],
                   linewidths=WEDGE_LW, zorder=4)
        handles.append(Line2D([], [], color=WEDGE_COLOR, lw=WEDGE_LW, label="Wedge"))
    if SHOW_SURFACE and eta is not None and np.nanmin(eta) < ETA_LEVEL < np.nanmax(eta):
        ax.contour(X, Y, eta, levels=[ETA_LEVEL], colors=[SURFACE_COLOR],
                   linewidths=SURF_LW, zorder=4)
        handles.append(Line2D([], [], color=SURFACE_COLOR, lw=SURF_LW,
                              label=f"Water surface ($\\eta$={ETA_LEVEL:g})"))

    y_surf = WATER_Y0 + U_STREAM * t                    # undisturbed surface height
    if SHOW_SURFACE_REF and ylo < y_surf < yhi:
        ax.axhline(y_surf * sc, color=SURFACE_REF_COLOR, lw=0.9, ls=":", zorder=3)
        handles.append(Line2D([], [], color=SURFACE_REF_COLOR, lw=0.9, ls=":",
                              label="Undisturbed surface $y = Ut$"))

    if SHOW_THEORY:
        b_air, b_wat = theory()
        # air shock lives above the surface; water shock only once the apex is wet
        # before the surface reaches the apex: air shock from the apex; after: water shock from
        # the apex up to the (undisturbed) surface height
        air = shock_line(b_air, WEDGE_YA, yhi) if y_surf < WEDGE_YA else None
        wat = shock_line(b_wat, WEDGE_YA, min(y_surf, yhi)) if y_surf > WEDGE_YA else None
        for seg, col, lbl in ((air, AIR_SHOCK_COLOR, f"Air shock theory $\\beta$={b_air:.1f}$^\\circ$"),
                              (wat, WATER_SHOCK_COLOR, f"Water shock theory $\\beta$={b_wat:.1f}$^\\circ$")):
            if seg is None:
                continue
            x, y = seg
            ax.plot(x * sc, y * sc, "--", color=col, lw=SHOCK_LW, zorder=6)
            if MIRROR:
                ax.plot(-x * sc, y * sc, "--", color=col, lw=SHOCK_LW, zorder=6)
            handles.append(Line2D([], [], color=col, lw=SHOCK_LW, ls="--", label=lbl))

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(f"x [{unit}]", fontsize=FONT_LABEL)
    ax.set_ylabel(f"y [{unit}]", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title(TITLE_TMPL.format(t=t * 1e6), fontsize=FONT_TITLE)
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8.5, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png, t, gref


def make_gif(pngs, gif_path):
    frames = [Image.open(p).convert("RGB") for p in pngs]
    # one shared adaptive palette keeps colours stable across frames
    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(gif_path, save_all=True, append_images=q[1:],
              duration=int(round(1000 / GIF_FPS)), loop=0, optimize=False)
    for f in frames:
        f.close()


def main():
    pdir = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else _abs(OUTPUT_DIR)
    img_dir = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else _abs(IMAGE_DIR)
    os.makedirs(img_dir, exist_ok=True)
    pfs = discover_plotfiles(pdir)
    if not pfs:
        print(f"ERROR: no plotfiles in {pdir}")
        return 1
    print(f"impact_schlieren: {len(pfs)} frames from {pdir}")
    b_air, b_wat = theory()
    print(f"  theory: air Ma {MACH_AIR:g} beta {b_air:.2f} deg | water Ma {MACH_WATER:.2f} beta "
          f"{b_wat:.2f} deg | surface reaches apex at {(WEDGE_YA - WATER_Y0) / U_STREAM * 1e6:.2f} us")

    tmp = tempfile.mkdtemp(prefix="impact_frames_", dir=img_dir)
    try:
        tasks = [(pf, os.path.join(tmp, f"f{i:05d}.png"), DPI) for i, pf in enumerate(pfs)]
        with Pool(max(1, min(NPROC, len(tasks)))) as pool:
            res = pool.map(render, tasks, chunksize=1)
        pngs = [r[0] for r in res]
        gif = os.path.join(img_dir, GIF_NAME)
        make_gif(pngs, gif)
        print(f"  wrote {gif}  ({len(pngs)} frames, {os.path.getsize(gif) / 1e6:.1f} MB)")
        last = os.path.join(img_dir, LAST_PNG_NAME)
        render((pfs[-1], last, DPI_LAST))
        print(f"  wrote {last}")
        grefs = np.array([r[2] for r in res])
        print(f"  schlieren g_ref range {grefs.min():.3g} .. {grefs.max():.3g} 1/m "
              f"(set SCHLIEREN_REF to fix it if the gif flickers)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
