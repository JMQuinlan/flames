#!/usr/bin/env python3
# ============================================================================
#  impact_schlieren_AOA30.py
#  ---------------------------------------------------------------------------
#  Numerical-schlieren animation of FlowTryonFun/Impact_AOA30: the Ma 10, 15 deg
#  half-angle wedge (axis vertical, apex down) entering a flat air/water surface
#  on a path 30 deg from straight down.  The run is in the WEDGE FRAME: the wedge
#  is fixed and air + water stream at V = (-1716.25, +2972.62) m/s (up and toward
#  -x), on the FULL width (no symmetry plane, no mirroring).
#
#  Overlays (all toggled in CONFIG):
#     - the solid wedge (phi), grey shading + outline
#     - the water surface (eta = 0.5 contour)
#     - the undisturbed surface y = V_y t (V_y = 2972.6 m/s, not |V|)
#     - free-stream direction arrow
#     - LEEWARD (-x) face: Prandtl-Meyer expansion fan from the apex, 15 deg turn
#       (air Ma 10, gamma 1.4 before the surface reaches the apex; water Ma 2.24,
#       stiffened gas gamma 2.35 after -- SG obeys the ideal-gas PM relation with p -> p + p0)
#     - WINDWARD (+x) face: the flow turns 30 + 15 = 45 deg, above the attached-shock
#       limit in air (44.4 deg) and water (15.8 deg) -> a detached bow shock.  No steady
#       line exists for it; the legend states the limit instead.
#
#  Outputs (to ./Images):
#     impact_schlieren_AOA30.gif         the animation
#     impact_schlieren_AOA30_last.png    the last frame at full resolution
#     impact_schlieren_AOA30_key.png     key frames side by side (KEY_TIMES_US)
#
#  Usage:
#     python3 impact_schlieren_AOA30.py                 # plotfiles from OUTPUT_DIR in CONFIG
#     python3 impact_schlieren_AOA30.py <plotfile_dir>
#     python3 impact_schlieren_AOA30.py <plotfile_dir> <image_dir>
#
#  Frames are rendered in parallel (NPROC) to PNGs, then stitched into the gif.
# ============================================================================

import os
import re
import sys
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

# ---- CASE (must match impact_Impact_AOA30) ---------------------------------
OUTPUT_DIR    = "/mmfs1/home/ttryon/flames/bin/tests/FlowTryonFun/Impact/output_Impact_AOA30"
V_STREAM      = (-1716.245, 2972.624)   # m/s, air + water stream in the wedge frame
WEDGE_THETA_DEG = 15.0         # half-angle
WEDGE_YA      = 0.03           # apex y (apex at x = 0)
WEDGE_YB      = 0.05           # base y
WATER_Y0      = 0.0            # water surface at t = 0
C_AIR         = 343.25         # m/s (Ma 10 at |V| = 3432.5)
C_WATER       = 1533.0         # m/s, sqrt(2.35 (p + 1e9) / 1000)
GAMMA_AIR     = 1.4
GAMMA_WATER   = 2.35           # stiffened gas: ideal-gas relations with p -> p + p0

# ---- I/O ------------------------------------------------------------------
IMAGE_DIR     = "./Images"
GIF_NAME      = "impact_schlieren_AOA30.gif"
LAST_PNG_NAME = "impact_schlieren_AOA30_last.png"
KEY_PNG_NAME  = "impact_schlieren_AOA30_key.png"
KEY_TIMES_US  = [6.0, 12.0, 18.0, 30.0]   # nearest frames; times past the run are dropped
IGNORE_TOKENS = [".old."]
FRAME_STRIDE  = 1              # use every Nth plotfile
MAX_FRAMES    = 200            # cap after striding (evenly thinned)
NPROC         = 16             # parallel frame renderers

# ---- SAMPLING / VIEW ------------------------------------------------------
DOMAIN_AUTO   = True           # full domain from the plotfile
VIEW          = (-0.06, 0.06, -0.03, 0.09)   # x0, x1, y0, y1 [m] when DOMAIN_AUTO = False
FRB_RES_X     = 900            # samples across the width

# ---- SCHLIEREN ------------------------------------------------------------
# 'lnrho_rel' : |grad ln(rho / rho_ref(eta))|, rho_ref = eta RHO_WATER + (1 - eta) RHO_AIR: the density
#               relative to the undisturbed fluid of the local mix.  The 830:1 surface jump cancels, so
#               air and water shocks show together and the surface does not flood the image.
# 'lnrho'     : |grad ln rho| (the surface jump and its eta tails show as a thick black band)
# 'rho'       : |grad rho|    (water features dominate)
SCHLIEREN_FIELD = "lnrho_rel"
RHO_WATER, RHO_AIR = 1000.0, 1.204    # undisturbed densities (deck ICs)
SCHLIEREN_EXP   = 8.0          # exp(-k g / g_ref) contrast
SCHLIEREN_PCT   = 99.8         # g_ref = this percentile of g in the fluid (per frame) ...
SCHLIEREN_REF   = None         # ... or a fixed g_ref (1/m) to stop frame-to-frame flicker
CMAP_SCHLIEREN  = "gray"
MASK_SOLID      = True         # blank the gradient where phi < PHI_MASK (solid + Brinkman edge)
PHI_MASK        = 0.95
MASK_SURFACE    = False        # blank the gradient inside the water-surface band ETA_MASK < eta < 1 - ETA_MASK:
ETA_MASK        = 0.02         #   the 830:1 density jump otherwise floods a thick black band over the
                               #   near-surface shocks; the surface itself is drawn as the eta contour

# ---- OVERLAYS / COLORS ----------------------------------------------------
SHOW_PHI_OVERLAY  = True
PHI_OVERLAY_ALPHA = 0.75
SHOW_WEDGE_OUTLINE = True
WEDGE_COLOR       = "#00b8d4"
PHI_LEVEL         = 0.5
SHOW_SURFACE      = True
SURFACE_COLOR     = "#1f6feb"
ETA_LEVEL         = 0.5
SHOW_SURFACE_REF  = True
SURFACE_REF_COLOR = "#8b949e"
SHOW_FLOW_ARROW   = True
ARROW_COLOR       = "#d6336c"
SHOW_FAN          = True       # leeward Prandtl-Meyer fan from the apex
AIR_FAN_COLOR     = "#e8590c"
WATER_FAN_COLOR   = "#2f9e44"
FAN_LENGTH        = 0.03       # m, drawn length of the fan's bounding Mach lines
WEDGE_LW, SURF_LW, FAN_LW = 1.4, 1.3, 1.5

# ---- FIGURE / TYPOGRAPHY --------------------------------------------------
FIGSIZE    = (7.2, 7.4)
DPI        = 110               # gif frames
DPI_LAST   = 180               # last-frame / key-frame png
GIF_FPS    = 10
FONT_TITLE = 13
FONT_LABEL = 11.5
FONT_TICK  = 10
LENGTH_UNIT = ("mm", 1.0e3)
TITLE_TMPL = "Ma 10 wedge, 30$^\\circ$ oblique water entry   t = {t:.2f} $\\mu$s"

# ============================================================================
# ============================  END CONFIG  ==================================
# ============================================================================

U_MAG = float(np.hypot(*V_STREAM))
FLOW_DEG = float(np.degrees(np.arctan2(-V_STREAM[0], V_STREAM[1])))   # from +y toward -x
T_APEX = (WEDGE_YA - WATER_Y0) / V_STREAM[1]
T_BASE = (WEDGE_YB - WATER_Y0) / V_STREAM[1]


def _abs(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(SCRIPT_DIR, p))


def step_number(name):
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def discover_plotfiles(d, thin=True):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"plotfile dir not found: {d}")
    items = [os.path.join(d, n) for n in os.listdir(d)
             if os.path.isdir(os.path.join(d, n))
             and ("cell" in n.lower() or "plt" in n.lower())
             and os.path.isfile(os.path.join(d, n, "Header"))
             and not any(t in n for t in IGNORE_TOKENS)]
    items.sort(key=step_number)
    if not thin:
        return items
    items = items[::max(1, FRAME_STRIDE)]
    if MAX_FRAMES and len(items) > MAX_FRAMES:
        idx = np.unique(np.linspace(0, len(items) - 1, MAX_FRAMES).round().astype(int))
        items = [items[i] for i in idx]
    return items


# ---- theory ----------------------------------------------------------------
def pm_nu(M, g):
    """Prandtl-Meyer function [rad]."""
    a = np.sqrt((g + 1.0) / (g - 1.0))
    return a * np.arctan(np.sqrt((M * M - 1.0) / (a * a))) - np.arctan(np.sqrt(M * M - 1.0))


def pm_M2(M1, turn_deg, g):
    """Mach number after an isentropic expansion by turn_deg (bisection on nu)."""
    target = pm_nu(M1, g) + np.radians(turn_deg)
    a = np.sqrt((g + 1.0) / (g - 1.0))
    if target >= 0.5 * np.pi * (a - 1.0):
        return np.inf                                        # beyond the vacuum limit
    lo, hi = M1, 1.0e4
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pm_nu(mid, g) < target else (lo, mid)
    return 0.5 * (lo + hi)


def theory():
    """Windward deflection vs attached limit, and the leeward fan in air and water."""
    out = {}
    turn_w = FLOW_DEG + WEDGE_THETA_DEG          # windward deflection
    turn_l = FLOW_DEG - WEDGE_THETA_DEG          # leeward expansion
    for name, c, g in (("air", C_AIR, GAMMA_AIR), ("water", C_WATER, GAMMA_WATER)):
        M1 = U_MAG / c
        M2 = pm_M2(M1, turn_l, g)
        mu1 = np.degrees(np.arcsin(1.0 / M1))
        mu2 = np.degrees(np.arcsin(1.0 / M2)) if np.isfinite(M2) else 0.0
        out[name] = dict(M1=M1, M2=M2, theta_max=ost.theta_max_deg(M1, g), turn_w=turn_w,
                         # bounding Mach lines of the fan, angle from +y toward -x
                         head=FLOW_DEG + mu1, tail=(FLOW_DEG - turn_l) + mu2)
    return out


def ray(angle_deg, length):
    """Segment from the apex at angle_deg from +y toward -x."""
    a = np.radians(angle_deg)
    return np.array([0.0, -length * np.sin(a)]), np.array([WEDGE_YA, WEDGE_YA + length * np.cos(a)])


# ---- sampling / schlieren --------------------------------------------------
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
        xlo, xhi, ylo, yhi = VIEW
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


def schlieren(s):
    xlo, xhi, ylo, yhi = s["extent"]
    dx = (xhi - xlo) / s["nx"]
    dy = (yhi - ylo) / s["ny"]
    if SCHLIEREN_FIELD == "lnrho_rel" and s["eta"] is not None:
        e = np.clip(s["eta"], 0.0, 1.0)
        f = np.log(np.maximum(s["rho"], 1e-30) / (e * RHO_WATER + (1.0 - e) * RHO_AIR))
    elif SCHLIEREN_FIELD in ("lnrho", "lnrho_rel"):
        f = np.log(np.maximum(s["rho"], 1e-30))
    else:
        f = s["rho"]
    gy, gx = np.gradient(f, dy, dx)
    g = np.hypot(gx, gy)
    fluid = s["phi"] > PHI_MASK
    if MASK_SOLID and s["has_phi"]:
        g = np.where(fluid, g, 0.0)
    if MASK_SURFACE and s["eta"] is not None:
        band = (s["eta"] > ETA_MASK) & (s["eta"] < 1.0 - ETA_MASK)
        g = np.where(band, 0.0, g)
        fluid = fluid & ~band
    if SCHLIEREN_REF is not None:
        gref = float(SCHLIEREN_REF)
    else:
        pool = g[fluid] if np.any(fluid) else g.ravel()
        gref = float(np.percentile(pool, SCHLIEREN_PCT)) if pool.size else 1.0
    return np.exp(-SCHLIEREN_EXP * g / max(gref, 1e-30)), gref


# ---- drawing ---------------------------------------------------------------
def draw(ax, s, legend=True, title=True, labels=(True, True)):
    sch, gref = schlieren(s)
    phi, eta = s["phi"], s["eta"]
    xlo, xhi, ylo, yhi = s["extent"]
    unit, sc = LENGTH_UNIT
    ext = [xlo * sc, xhi * sc, ylo * sc, yhi * sc]
    X = np.linspace(ext[0], ext[1], sch.shape[1])
    Y = np.linspace(ext[2], ext[3], sch.shape[0])
    t = s["time"]

    ax.imshow(sch, origin="lower", extent=ext, cmap=CMAP_SCHLIEREN, vmin=0.0, vmax=1.0,
              aspect="equal", interpolation="bilinear", zorder=0)
    handles = []
    if SHOW_PHI_OVERLAY and s["has_phi"]:
        tt = np.linspace(0.0, 1.0, 256)
        lut = plt.get_cmap("gray")(tt * 0.35)
        lut[:, 3] = PHI_OVERLAY_ALPHA * (1.0 - tt)
        ax.imshow(np.clip(phi, 0, 1), origin="lower", extent=ext, cmap=ListedColormap(lut),
                  vmin=0, vmax=1, aspect="equal", interpolation="bilinear", zorder=1)
    if SHOW_WEDGE_OUTLINE and s["has_phi"]:
        ax.contour(X, Y, phi, levels=[PHI_LEVEL], colors=[WEDGE_COLOR], linewidths=WEDGE_LW, zorder=4)
        handles.append(Line2D([], [], color=WEDGE_COLOR, lw=WEDGE_LW, label="Wedge"))
    if SHOW_SURFACE and eta is not None and np.nanmin(eta) < ETA_LEVEL < np.nanmax(eta):
        ax.contour(X, Y, eta, levels=[ETA_LEVEL], colors=[SURFACE_COLOR], linewidths=SURF_LW, zorder=4)
        handles.append(Line2D([], [], color=SURFACE_COLOR, lw=SURF_LW,
                              label=f"Water surface ($\\eta$={ETA_LEVEL:g})"))

    y_surf = WATER_Y0 + V_STREAM[1] * t
    if SHOW_SURFACE_REF and ylo < y_surf < yhi:
        ax.axhline(y_surf * sc, color=SURFACE_REF_COLOR, lw=0.9, ls=":", zorder=3)
        handles.append(Line2D([], [], color=SURFACE_REF_COLOR, lw=0.9, ls=":",
                              label="Undisturbed surface $y = V_y t$"))

    th = theory()
    if SHOW_FAN:
        wet = y_surf > WEDGE_YA
        ph, col = (th["water"], WATER_FAN_COLOR) if wet else (th["air"], AIR_FAN_COLOR)
        L = FAN_LENGTH if not wet else min(FAN_LENGTH, max(0.0, (y_surf - WEDGE_YA)) / np.cos(np.radians(ph["head"])))
        if L > 0:
            for ang in (ph["head"], ph["tail"]):
                x, y = ray(ang, L)
                ax.plot(x * sc, y * sc, "--", color=col, lw=FAN_LW, zorder=6)
            m2 = f"{ph['M2']:.1f}" if np.isfinite(ph["M2"]) else "$\\infty$"
            handles.append(Line2D([], [], color=col, lw=FAN_LW, ls="--",
                                  label=f"Leeward PM fan, {'water' if wet else 'air'} "
                                        f"Ma {ph['M1']:.2f}$\\to${m2}"))
        ph = th["water"] if wet else th["air"]
        handles.append(Line2D([], [], color="none",
                              label=f"Windward turn {ph['turn_w']:.0f}$^\\circ$ > "
                                    f"$\\theta_{{max}}$ {ph['theta_max']:.1f}$^\\circ$: detached"))

    if SHOW_FLOW_ARROW:
        x0, y0 = xhi - 0.12 * (xhi - xlo), ylo + 0.10 * (yhi - ylo)
        Lr = 0.08 * (yhi - ylo)
        ax.annotate("", xy=((x0 + Lr * V_STREAM[0] / U_MAG) * sc, (y0 + Lr * V_STREAM[1] / U_MAG) * sc),
                    xytext=(x0 * sc, y0 * sc),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=2.0), zorder=7)
        ax.text(x0 * sc, (y0 - 0.025 * (yhi - ylo)) * sc, f"{U_MAG:.0f} m/s", color=ARROW_COLOR,
                ha="center", va="top", fontsize=FONT_TICK - 1, zorder=7)

    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    if labels[0]:
        ax.set_xlabel(f"x [{unit}]", fontsize=FONT_LABEL)
    if labels[1]:
        ax.set_ylabel(f"y [{unit}]", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    if title:
        ax.set_title(TITLE_TMPL.format(t=t * 1e6), fontsize=FONT_TITLE)
    if legend and handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.85)
    return gref


def render(task):
    pf, out_png, dpi = task
    s = sample(pf)
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=dpi)
    gref = draw(ax, s)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png, s["time"], gref


def key_frames(all_pfs, out_png):
    times = np.array([float(yt.load(p).current_time) for p in all_pfs])
    pick = []
    for tk in KEY_TIMES_US:
        if tk * 1e-6 > times[-1] + 1e-9:
            continue
        i = int(np.argmin(np.abs(times - tk * 1e-6)))
        if i not in pick:
            pick.append(i)
    if not pick:
        return None
    n = len(pick)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 4.1), dpi=DPI_LAST, squeeze=False)
    for k, (ax, i) in enumerate(zip(axs[0], pick)):
        s = sample(all_pfs[i])
        draw(ax, s, legend=(k == n - 1), title=False, labels=(True, k == 0))
        ax.set_title(f"t = {s['time'] * 1e6:.1f} $\\mu$s", fontsize=FONT_LABEL)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI_LAST)
    plt.close(fig)
    return out_png


def make_gif(pngs, gif_path):
    frames = [Image.open(p).convert("RGB") for p in pngs]
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
    th = theory()
    print(f"impact_schlieren_AOA30: {len(pfs)} frames from {pdir}")
    print(f"  stream |V| {U_MAG:.1f} m/s at {FLOW_DEG:.1f} deg from +y; surface reaches apex at "
          f"{T_APEX * 1e6:.2f} us, base at {T_BASE * 1e6:.2f} us")
    for k in ("air", "water"):
        p = th[k]
        print(f"  {k:5s}: Ma {p['M1']:.2f} | windward turn {p['turn_w']:.1f} deg vs theta_max "
              f"{p['theta_max']:.1f} deg -> {'DETACHED' if p['turn_w'] > p['theta_max'] else 'attached'} | "
              f"leeward fan Ma {p['M1']:.2f} -> {p['M2']:.2f}, Mach lines at {p['head']:.1f} / {p['tail']:.1f} deg")

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
        key = key_frames(discover_plotfiles(pdir, thin=False), os.path.join(img_dir, KEY_PNG_NAME))
        if key:
            print(f"  wrote {key}")
        grefs = np.array([r[2] for r in res])
        print(f"  schlieren g_ref range {grefs.min():.3g} .. {grefs.max():.3g} 1/m "
              f"(set SCHLIEREN_REF to fix it if the gif flickers)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
