#!/usr/bin/env python3
# ============================================================================
#  wedge_analysis.py
#  ---------------------------------------------------------------------------
#  Analyze one FlowWedge (Hydro2) supersonic run: render the wedge flow, detect
#  the oblique shock, and overlay the ANALYTICAL shock line (theta-beta-Mach) on
#  the 2D contour so the measured vs theoretical shock POSITION and ANGLE can be
#  compared directly.
#
#  Outputs (to ./Images):
#     wedge_Ma<MA>.png       - background field + wedge outline + measured shock
#                              points + analytical shock line, annotated with
#                              beta_analytical vs beta_measured.
#     prints a beta comparison table to stdout.
#
#  Usage:
#     python3 wedge_analysis.py                 # uses MACH / paths in CONFIG
#     python3 wedge_analysis.py 3.0             # override Mach (loads ../output_Ma3.0)
#     python3 wedge_analysis.py 3.0 <out_dir>   # explicit output dir
#
#  Everything is configurable in the CONFIG block (house style).
# ============================================================================

import os
import re
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import yt
yt.funcs.mylog.setLevel(50)

import oblique_shock_theory as ost

# ============================================================================
# ==============================  CONFIG  ====================================
# ============================================================================

# ---- CASE -----------------------------------------------------------------
MACH          = 2.0                       # freestream Mach (overridable via argv[1])
GAMMA         = 1.4
OUTPUT_TMPL   = "../../../bin/tests/FlowWedge/output_Ma{ma}"        # plotfile dir, {ma} -> str(MACH)
FRAME         = "last"                    # 'last' or an integer plotfile index

# ---- WEDGE GEOMETRY (must match the input file) ---------------------------
WEDGE_THETA_DEG = 15.0
WEDGE_X0        = -0.5                     # apex x
WEDGE_XB        = 1.5                      # base x
WEDGE_Y0        = 0.0                      # apex y (axis of symmetry)

# ---- I/O ------------------------------------------------------------------
OUTPUT_DIR    = "./Images"
IGNORE_TOKENS = [".old."]

# ---- SAMPLING / DOMAIN ----------------------------------------------------
DOMAIN_AUTO   = True
X_MIN, X_MAX  = -2.0, 4.0
Y_MIN, Y_MAX  = -2.0, 2.0
FRB_RES_X     = 1200

# ---- BACKGROUND FIELD -----------------------------------------------------
BACKGROUND    = "schlieren"   # 'schlieren' | 'density' | 'pressure' | 'mach'
SCHLIEREN_EXP = 12.0          # exp contrast for numerical schlieren
CMAP_SCHLIEREN = "gray_r"
CMAP_FIELD     = "inferno"
SHOW_COLORBAR  = True

# ---- SHOCK DETECTION ------------------------------------------------------
DETECT_SHOCK   = True
Y_DET_LO       = 0.25         # detect along horizontal lines y in [lo,hi] (upper half)
Y_DET_HI       = 1.6
N_DET_LINES    = 60
X_DET_LO       = -1.0         # x search window for the shock
X_DET_HI       = 3.0
PHI_FLUID      = 0.85         # only detect in pure fluid (phi > this); excludes wedge surface
RHO_JUMP_FRAC  = 0.05         # shock front = first x where rho > rho_inf*(1+this) (upstream-most rise)
FIT_THROUGH_APEX = False      # True: force the measured-angle fit through the apex

# ---- OVERLAYS / COLORS ----------------------------------------------------
SHOW_WEDGE_OUTLINE = True
SHOW_MEASURED      = True
SHOW_ANALYTICAL    = True
SHOW_MACH_LINE     = False        # also draw the Mach-wave angle from the apex
WEDGE_COLOR        = "#00e5ff"    # wedge phi=0.5 outline
MEASURED_COLOR     = "#ff2d2d"    # measured shock points / fit  (RED)
ANALYTIC_COLOR     = "#39ff14"    # analytical shock line        (GREEN)
MACHLINE_COLOR     = "#ffd400"
WEDGE_LW   = 1.6
SHOCK_LW   = 2.0
MEAS_MS    = 12                   # measured-point marker size (scatter)

# ---- FIGURE / TYPOGRAPHY --------------------------------------------------
FIGSIZE    = (11.0, 6.2)
DPI        = 140
FONT_TITLE = 15
FONT_LABEL = 12.5
FONT_TICK  = 10.5
FONT_ANNOT = 12
TITLE_TMPL = "FlowWedge  Ma {ma}:  oblique shock vs theta-beta-M theory"

# ============================================================================
# ============================  END CONFIG  ==================================
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(SCRIPT_DIR, p))


def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def step_number(name):
    m = re.search(r"(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else -1


def discover_plotfiles(d):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"output dir not found: {d}")
    items = [os.path.join(d, n) for n in os.listdir(d)
             if os.path.isdir(os.path.join(d, n))
             and ("cell" in n.lower() or "plt" in n.lower())
             and not any(t in n for t in IGNORE_TOKENS)]
    items.sort(key=step_number)
    return items


def _read(frb, field):
    for key in (field, ("boxlib", field), ("gas", field)):
        try:
            return np.array(frb[key])
        except Exception:
            continue
    raise KeyError(f"field '{field}' not in plotfile")


def sample(plotdir):
    ds = yt.load(plotdir)
    if DOMAIN_AUTO:
        le, re_ = ds.domain_left_edge, ds.domain_right_edge
        xlo, xhi, ylo, yhi = float(le[0]), float(re_[0]), float(le[1]), float(re_[1])
    else:
        xlo, xhi, ylo, yhi = X_MIN, X_MAX, Y_MIN, Y_MAX
    W, H = xhi - xlo, yhi - ylo
    nx = int(FRB_RES_X)
    ny = max(1, int(round(nx * H / W)))
    slc = ds.slice(2, 0.0)
    frb = slc.to_frb(width=(W, "code_length"), resolution=(nx, ny),
                     height=(H, "code_length"),
                     center=[0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.0])
    out = dict(rho=_read(frb, "density"), p=_read(frb, "pressure"),
               phi=_read(frb, "phi"), vx=_read(frb, "velocityx"),
               vy=_read(frb, "velocityy"),
               extent=(xlo, xhi, ylo, yhi), nx=nx, ny=ny, time=float(ds.current_time))
    out["X"] = np.linspace(xlo, xhi, nx)
    out["Y"] = np.linspace(ylo, yhi, ny)
    out["XX"], out["YY"] = np.meshgrid(out["X"], out["Y"])
    return out


def background_field(s):
    """Return (image_array, cmap, label) for the chosen background."""
    xlo, xhi, ylo, yhi = s["extent"]
    dx = (xhi - xlo) / (s["nx"] - 1)
    dy = (yhi - ylo) / (s["ny"] - 1)
    if BACKGROUND == "density":
        return s["rho"], CMAP_FIELD, "density"
    if BACKGROUND == "pressure":
        return s["p"], CMAP_FIELD, "pressure"
    if BACKGROUND == "mach":
        c = np.sqrt(GAMMA * np.maximum(s["p"], 1e-12) / np.maximum(s["rho"], 1e-12))
        mach = np.sqrt(s["vx"] ** 2 + s["vy"] ** 2) / np.maximum(c, 1e-12)
        return mach, CMAP_FIELD, "Mach number"
    # numerical schlieren (default)
    gy, gx = np.gradient(s["rho"], dy, dx)
    mag = np.sqrt(gx * gx + gy * gy)
    schl = np.exp(-SCHLIEREN_EXP * mag / (mag.max() + 1e-30))
    return schl, CMAP_SCHLIEREN, "numerical schlieren"


def grad_rho_mag(s):
    xlo, xhi, ylo, yhi = s["extent"]
    dx = (xhi - xlo) / (s["nx"] - 1)
    dy = (yhi - ylo) / (s["ny"] - 1)
    gy, gx = np.gradient(s["rho"], dy, dx)
    return np.sqrt(gx * gx + gy * gy)


def detect_shock(s):
    """Scan horizontal lines (upper half) for the UPSTREAM-most density rise
    (the oblique shock front), return measured shock points + fitted beta.

    The shock front is the first x (scanning the freestream from the left) where
    the density exceeds rho_inf*(1+RHO_JUMP_FRAC).  The wedge surface sits further
    downstream inside the already-compressed region, so it is not picked.

    Returns dict: pts_x, pts_y (arrays), beta_deg (or None), intercept_x.
    """
    rho, X, Y, phi = s["rho"], s["X"], s["Y"], s["phi"]
    dx = X[1] - X[0]
    tan_t = np.tan(np.radians(WEDGE_THETA_DEG))

    # freestream density from an upstream fluid column
    iup = X < (X_DET_LO + 0.3)
    up = rho[:, iup][phi[:, iup] > PHI_FLUID]
    rho_inf = float(np.median(up)) if up.size else float(np.median(rho))
    thresh = rho_inf * (1.0 + RHO_JUMP_FRAC)

    pts_x, pts_y = [], []
    for yq in np.linspace(Y_DET_LO, Y_DET_HI, N_DET_LINES):
        j = int(np.argmin(np.abs(Y - yq)))
        # search window: ahead of the wedge surface at this height (if it exists)
        x_surf = WEDGE_X0 + (yq / tan_t) if yq <= (WEDGE_XB - WEDGE_X0) * tan_t else X_DET_HI
        xhi_lim = min(X_DET_HI, x_surf - 2.0 * dx)
        m = (X >= X_DET_LO) & (X <= xhi_lim) & (phi[j, :] > PHI_FLUID) & (rho[j, :] > thresh)
        if not np.any(m):
            continue
        i = int(np.argmax(m))             # first True = upstream-most density rise
        pts_x.append(X[i])
        pts_y.append(yq)

    pts_x, pts_y = np.array(pts_x), np.array(pts_y)
    beta_deg, intercept = None, None
    if len(pts_x) >= 3:
        if FIT_THROUGH_APEX:
            # x - x0 = (1/tan beta) * y  through the apex
            b = np.sum((pts_y) * (pts_x - WEDGE_X0)) / np.sum(pts_y * pts_y)
            intercept = WEDGE_X0
        else:
            b, intercept = np.polyfit(pts_y, pts_x, 1)   # x = b*y + intercept
        beta_deg = float(np.degrees(np.arctan2(1.0, b)))  # tan(beta) = dy/dx = 1/b
    return dict(pts_x=pts_x, pts_y=pts_y, beta_deg=beta_deg, intercept_x=intercept)


def render(s, meas, mach, out_png):
    xlo, xhi, ylo, yhi = s["extent"]
    img, cmap, label = background_field(s)

    # analytical shock angle
    beta_a = ost.beta_deg(WEDGE_THETA_DEG, mach, GAMMA, weak=True)
    attached = beta_a is not None
    tmax = ost.theta_max_deg(mach, GAMMA)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    im = ax.imshow(img, origin="lower", extent=[xlo, xhi, ylo, yhi],
                   cmap=cmap, aspect="equal", interpolation="bilinear", zorder=0)
    if SHOW_COLORBAR:
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(label, fontsize=FONT_LABEL)

    # wedge outline (phi = 0.5)
    if SHOW_WEDGE_OUTLINE:
        ax.contour(s["XX"], s["YY"], s["phi"], levels=[0.5],
                   colors=[WEDGE_COLOR], linewidths=WEDGE_LW, zorder=4)

    # measured shock points + fit
    if SHOW_MEASURED and len(meas["pts_x"]):
        ax.scatter(meas["pts_x"], meas["pts_y"], s=MEAS_MS, c=MEASURED_COLOR,
                   marker="o", zorder=6, label="measured shock")
        # mirror to lower half for visual symmetry
        ax.scatter(meas["pts_x"], -meas["pts_y"], s=MEAS_MS, c=MEASURED_COLOR,
                   marker="o", zorder=6)
        if meas["beta_deg"] is not None:
            yy = np.array([0.0, max(Y_DET_HI, yhi)])
            b = 1.0 / np.tan(np.radians(meas["beta_deg"]))
            xx = meas["intercept_x"] + b * yy
            ax.plot(xx, yy, "--", color=MEASURED_COLOR, lw=SHOCK_LW * 0.8, zorder=6)
            ax.plot(xx, -yy, "--", color=MEASURED_COLOR, lw=SHOCK_LW * 0.8, zorder=6)

    # analytical shock line from the apex
    if SHOW_ANALYTICAL and attached:
        yy = np.linspace(0.0, yhi, 50)
        xx = WEDGE_X0 + yy / np.tan(np.radians(beta_a))
        ax.plot(xx, yy, "-", color=ANALYTIC_COLOR, lw=SHOCK_LW, zorder=7,
                label=f"analytical beta={beta_a:.2f}deg")
        ax.plot(xx, -yy, "-", color=ANALYTIC_COLOR, lw=SHOCK_LW, zorder=7)

    if SHOW_MACH_LINE:
        mu = ost.mach_angle(mach)
        if mu is not None:
            yy = np.linspace(0.0, yhi, 50)
            xx = WEDGE_X0 + yy / np.tan(mu)
            ax.plot(xx, yy, ":", color=MACHLINE_COLOR, lw=1.3, zorder=5,
                    label=f"Mach wave mu={np.degrees(mu):.1f}deg")

    # annotation box
    lines = [f"Ma = {mach:.2f},  theta = {WEDGE_THETA_DEG:.1f}deg,  gamma = {GAMMA}",
             f"theta_max(Ma) = {tmax:.2f}deg" if tmax else "theta_max: n/a"]
    if attached:
        lines.append(f"beta_analytical = {beta_a:.2f}deg")
        if meas["beta_deg"] is not None:
            err = meas["beta_deg"] - beta_a
            lines.append(f"beta_measured   = {meas['beta_deg']:.2f}deg")
            lines.append(f"error           = {err:+.2f}deg ({100*err/beta_a:+.1f}%)")
        else:
            lines.append("beta_measured   = (not detected)")
    else:
        lines.append("ATTACHED SHOCK IMPOSSIBLE -> DETACHED BOW SHOCK")
        lines.append(f"(theta={WEDGE_THETA_DEG:.0f} > theta_max={tmax:.2f})")
    ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes,
            fontsize=FONT_ANNOT, va="top", ha="left", family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.3", alpha=0.85), zorder=10)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("x", fontsize=FONT_LABEL)
    ax.set_ylabel("y", fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_title(TITLE_TMPL.format(ma=f"{mach:g}") + f"    (t = {s['time']:.2f})",
                 fontsize=FONT_TITLE)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"  wrote {out_png}")
    return beta_a, meas["beta_deg"]


def analyze(mach, out_dir_override=None):
    out_dir = _abs(out_dir_override) if out_dir_override \
        else _abs(OUTPUT_TMPL.format(ma=f"{mach:.1f}"))
    if not os.path.isdir(out_dir):
        print(f"ERROR: output dir not found: {out_dir}")
        print("Run the sim first, e.g.:  ./bin/hydro2-2d-g++ tests/FlowWedge/input_Ma"
              f"{mach:.1f}")
        return None
    pfs = discover_plotfiles(out_dir)
    if not pfs:
        print(f"ERROR: no plotfiles in {out_dir}")
        return None
    pf = pfs[-1] if FRAME == "last" else pfs[int(FRAME)]
    print(f"  Ma {mach:g}: {os.path.basename(pf)}")
    s = sample(pf)
    meas = detect_shock(s) if DETECT_SHOCK else dict(pts_x=[], pts_y=[], beta_deg=None, intercept_x=None)
    ensure_dir(_abs(OUTPUT_DIR))
    out_png = os.path.join(_abs(OUTPUT_DIR), f"wedge_Ma{mach:.1f}.png")
    return render(s, meas, mach, out_png)


def main():
    mach = float(sys.argv[1]) if len(sys.argv) > 1 else MACH
    out_override = sys.argv[2] if len(sys.argv) > 2 else None
    print("wedge_analysis")
    res = analyze(mach, out_override)
    if res:
        ba, bm = res
        print("\n  ---- summary ----")
        if ba is None:
            print(f"  Ma {mach:g}: detached (no attached oblique shock)")
        else:
            bm_s = f"{bm:.2f}" if bm is not None else "n/a"
            print(f"  Ma {mach:g}: beta_analytical={ba:.2f}deg  beta_measured={bm_s}deg")
    print("\nDone.")


if __name__ == "__main__":
    main()
