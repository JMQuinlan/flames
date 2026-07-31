# -*- coding: utf-8 -*-
"""
===============================================================================
FLOW-DRIVEN BUBBLE -- RADIUS HISTORY  R(t)   (3D OCTANT)
===============================================================================

PURPOSE:
    Post-process the FlowDrivenBubble runs (input_LowAmp / input_HighAmp):
    an air bubble in water, simulated as a +x+y+z OCTANT with symmetry BCs
    on the inner faces (bubble center at the origin, same convention as the
    Sch20_*_Large_3D variants), driven by a sinusoidal wall pressure
        p_wall(t) = p_inf + A sin(2 pi f t)
    imposed on the outer faces through the primitive BC path.  For every
    plotfile this script extracts the bubble radius TWO independent ways and
    plots both:

        1) VOLUME radius   -- gas volume V = sym_factor * sum (1-eta) V_cell
           over all leaf cells, converted to an assumed-spherical radius:
               3D:  R = (3 V / (4 pi))^(1/3)
               2D:  R = sqrt(V / pi)           (cylindrical / area based)
           sym_factor = 2^(number of symmetry planes) is auto-detected from
           the domain: an axis whose lower edge sits at the bubble center is
           a symmetry plane (octant run -> sym_factor = 8; full domain -> 1).
        2) ETA=0.5 radius  -- radially-binned, volume-weighted average eta(r)
           about the bubble center; R is the interpolated eta = 0.5 crossing
           of the binned profile (angle-averaged interface radius).  The
           octant needs no volume correction here -- by symmetry the radial
           profile over one octant equals the full-sphere profile.

    Agreement between the two is a shape-integrity check: they separate when
    the bubble departs from a sphere or the interface band smears.

USAGE:
    python tests/FlowDrivenBubble/reference/analyze_radius.py

OUTPUTS:
    Images/FlowDrivenBubble_radius.png  (and .eps)
    Console summary per run: R_max/R0, R_min/R0 and their times.

===============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# ============================  CONFIGURATION  ===============================
# Everything tunable lives in this block. Adjust here -- no hunting needed.
# ============================================================================

# ===== RUNS TO ANALYZE =====
# Each entry: display label, plotfile output dir, color.  Runs whose dir does
# not exist (yet) are skipped with a note, so it is safe to leave both here.
# Default paths are relative to the repo (run from <repo>/bin); uncomment and
# edit the absolute OVERRIDE entries for INCLINE / another machine.
def _repo(*p):
    return os.path.normpath(os.path.join(_HERE, "..", "..", "..", "bin", *p))

RUNS = [
    dict(label="LowAmp (A = 0.001 bar)",
         out_dir=_repo("tests", "FlowDrivenBubble", "output_LowAmp"),
         color="tab:blue"),
    dict(label="HighAmp (A = 5 MPa)",
         out_dir=_repo("tests", "FlowDrivenBubble", "output_HighAmp"),
         color="tab:red"),
]
# OVERRIDE (uncomment + edit for INCLINE):
RUNS = [
     dict(label="LowAmp (A = 0.001 bar)",
          out_dir="/mmfs1/home/ttryon/flames/bin/tests/FlowDrivenBubble/output_LowAmp",
          color="tab:blue")
#     dict(label="HighAmp (A = 5 MPa)",
#          out_dir="/mmfs1/home/ttryon/flames/bin/tests/FlowDrivenBubble/output_HighAmp",
#          color="tab:red"),
]

# ===== PROBLEM PARAMETERS (must match the input files) =====
DIM      = 3               # 3 -> R = (3V/4pi)^(1/3); 2 -> R = sqrt(V/pi)
R0       = 0.02            # initial bubble radius [m]
CENTER   = (0.0, 0.0, 0.0) # bubble center [m] (origin for the octant runs)
P_INF    = 1.0e5           # ambient pressure [Pa]
F_DRIVE  = 81.54           # drive frequency [Hz]  (f0/2, Minnaert f0 = 163.09 Hz)
T_DRIVE  = 1.0 / F_DRIVE

# Symmetry factor: None -> auto-detect from the domain (an axis whose lower
# edge coincides with CENTER is a symmetry plane; octant -> 8).  Set to an
# integer (1, 2, 4, 8) to override the auto-detection.
SYM_FACTOR = None

# ===== EXTRACTION KNOBS =====
ETA_THRESHOLD = 0.5     # interface level for the contour radius
N_RADIAL_BINS = 400     # radial bins for the angle-averaged eta(r) profile
R_BIN_MAX     = 0.04    # outer radius of the binned profile [m] (= octant edge)

# ===== DRIVE OVERLAY =====
SHOW_DRIVE  = True            # overlay normalized drive sin(2 pi f t) (right axis)
DRIVE_COLOR = "0.6"
DRIVE_LW    = 1.0

# ===== PLOT STYLING (publication knobs) =====
FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 11
FONT_SIZE_TICK   = 11

LINE_WIDTH_VOL = 2.0    # volume-radius curves (solid)
LINE_WIDTH_ETA = 1.5    # eta=0.5-radius curves (dashed)

DPI        = 180
FIG_WIDTH  = 11
FIG_HEIGHT = 6.5
LEGEND_LOC = "best"

# ===== AXIS LABELS / TITLE (every string the user might rephrase) =====
TITLE_STR    = "Flow-Driven Bubble (3D octant): R(t)"
XLABEL_STR   = r"$t / T_{drive}$"
YLABEL_STR   = r"$R / R_0$"
DRIVE_YLABEL = r"$(p_{wall} - p_\infty)/A$"
LABEL_VOL    = "volume"       # appended to run label, e.g. "LowAmp -- volume"
LABEL_ETA    = r"$\eta=0.5$"
NONDIM_TIME   = True    # plot t / T_drive instead of t [s]
NONDIM_RADIUS = True    # plot R / R0 instead of R [m]
SAVE_NAME     = "FlowDrivenBubble_radius"

# ============================================================================
# ==========================  END CONFIGURATION  =============================
# ============================================================================

IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# EXTRACTION
# ============================================================================

def _list_plotfiles(out_dir):
    """All '*cell' plotfile subdirectories of out_dir (alphabetical; re-sorted
    by simulation time after loading)."""
    if not os.path.isdir(out_dir):
        return []
    return sorted(
        os.path.join(out_dir, d)
        for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d)) and d.endswith("cell")
    )


def _sym_factor(ds):
    """2**(number of symmetry planes).  An axis whose domain lower edge sits
    at the bubble center is a symmetry plane (octant convention of the
    Sch20_*_Large_3D tests).  Overridden by SYM_FACTOR if set."""
    if SYM_FACTOR is not None:
        return float(SYM_FACTOR)
    lo = np.array(ds.domain_left_edge.to_value())
    hi = np.array(ds.domain_right_edge.to_value())
    n_sym = 0
    for d in range(DIM):
        width = hi[d] - lo[d]
        if abs(lo[d] - CENTER[d]) < 1.0e-6 * width:
            n_sym += 1
    return float(2 ** n_sym)


def extract_radius_history(out_dir):
    """Walk every plotfile in out_dir and return
        (t, R_vol, R_eta, n_skipped)
    with all arrays sorted by time.

    R_vol : assumed-spherical radius from the symmetry-corrected gas volume
            V = sym_factor * sum (1-eta) V_cell
            (3D: cbrt(3V/4pi), 2D: sqrt(V/pi) -- see DIM).
    R_eta : eta = ETA_THRESHOLD crossing of the radially-binned, angle-
            averaged eta(r) profile about CENTER (NaN if no crossing).

    Corrupt / partially-written plotfiles (interrupted runs) are skipped
    silently and counted in n_skipped.  Never raises.
    """
    import yt
    yt.funcs.mylog.setLevel(40)

    times, R_vol, R_eta = [], [], []
    n_skipped = 0

    bin_edges = np.linspace(0.0, R_BIN_MAX, N_RADIAL_BINS + 1)
    bin_mid = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for pf in _list_plotfiles(out_dir):
        try:
            ds = yt.load(pf)
            reg = ds.all_data()
            eta = np.array(reg["eta"])
            x = np.array(reg["x"]) - CENTER[0]
            y = np.array(reg["y"]) - CENTER[1]
            try:
                vol = np.array(reg["index", "cell_volume"])
            except Exception:
                vol = np.array(reg["cell_volume"])

            # ---- 1) volume radius (symmetry-corrected) ------------------
            alpha_g = np.clip(1.0 - eta, 0.0, 1.0)
            Vg = float(np.sum(alpha_g * vol)) * _sym_factor(ds)
            if DIM == 2:
                R_vol.append(np.sqrt(Vg / np.pi))
            else:
                R_vol.append((3.0 * Vg / (4.0 * np.pi)) ** (1.0 / 3.0))

            # ---- 2) angle-averaged eta = 0.5 radius ---------------------
            # (no symmetry correction needed: the octant's radial profile
            # equals the full sphere's by symmetry)
            if DIM == 2:
                r = np.sqrt(x * x + y * y)
            else:
                z = np.array(reg["z"]) - CENTER[2]
                r = np.sqrt(x * x + y * y + z * z)
            # volume-weighted mean eta per radial bin (leaf cells only --
            # yt's all_data already excludes covered coarse cells)
            w_sum, _ = np.histogram(r, bins=bin_edges, weights=vol)
            e_sum, _ = np.histogram(r, bins=bin_edges, weights=eta * vol)
            with np.errstate(invalid="ignore", divide="ignore"):
                eta_prof = e_sum / w_sum
            valid = np.isfinite(eta_prof)
            rp, ep = bin_mid[valid], eta_prof[valid]
            # first upward crossing of the threshold (eta: 0 inside -> 1 out)
            R_c = np.nan
            above = np.where(ep >= ETA_THRESHOLD)[0]
            if len(above) > 0:
                i = above[0]
                if i == 0:
                    R_c = rp[0]
                else:
                    frac = (ETA_THRESHOLD - ep[i - 1]) / (ep[i] - ep[i - 1])
                    R_c = rp[i - 1] + frac * (rp[i] - rp[i - 1])
            R_eta.append(R_c)

            times.append(float(ds.current_time))
        except Exception:
            n_skipped += 1
            continue

    if not times:
        return np.array([]), np.array([]), np.array([]), n_skipped

    order = np.argsort(np.array(times))
    return (np.array(times)[order], np.array(R_vol)[order],
            np.array(R_eta)[order], n_skipped)


def print_extrema(label, t, R):
    """Console summary of the extremes of one R(t) curve."""
    R = np.asarray(R, dtype=float)
    finite = np.isfinite(R)
    if not np.any(finite):
        print(f"    {label:<28s}: no finite samples")
        return
    idx = np.where(finite)[0]
    j_min = idx[np.argmin(R[idx])]
    j_max = idx[np.argmax(R[idx])]
    print(f"    {label:<28s}: R_max/R0 = {R[j_max]/R0:.4f} "
          f"(t/T = {t[j_max]/T_DRIVE:.3f}),  "
          f"R_min/R0 = {R[j_min]/R0:.4f} (t/T = {t[j_min]/T_DRIVE:.3f})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_scale = T_DRIVE if NONDIM_TIME else 1.0
    r_scale = R0 if NONDIM_RADIUS else 1.0

    print("=" * 70)
    print("FLOW-DRIVEN BUBBLE -- RADIUS HISTORY  (3D octant)")
    print("=" * 70)
    print(f"  DIM       = {DIM}  "
          f"({'R = sqrt(V/pi)' if DIM == 2 else 'R = (3V/4pi)^(1/3)'})")
    print(f"  R0        = {R0}")
    print(f"  f_drive   = {F_DRIVE} Hz   (T_drive = {T_DRIVE*1e3:.3f} ms)")
    print(f"  sym       = {'auto-detect' if SYM_FACTOR is None else SYM_FACTOR}")
    print()

    plt.rcParams.update({
        "axes.titlesize":  FONT_SIZE_TITLE,
        "axes.labelsize":  FONT_SIZE_LABEL,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
    })
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    t_end_seen = 0.0
    any_loaded = False
    for run in RUNS:
        out_dir = run["out_dir"]
        if not os.path.isdir(out_dir):
            print(f"  [info] no output yet at {out_dir} -- skipped.")
            continue
        print(f"  loading {out_dir} ...")
        t, R_v, R_e, n_skip = extract_radius_history(out_dir)
        if n_skip > 0:
            print(f"  [warn] skipped {n_skip} corrupt / partial plotfile(s).")
        if len(t) < 2:
            print(f"  [info] no usable plotfiles in {out_dir} -- skipped.")
            continue
        any_loaded = True
        t_end_seen = max(t_end_seen, float(t[-1]))

        print(f"  {run['label']}: {len(t)} frames")
        print_extrema(f"{run['label']} ({LABEL_VOL})", t, R_v)
        print_extrema(f"{run['label']} ({LABEL_ETA})", t, R_e)

        ax.plot(t / t_scale, R_v / r_scale, "-", color=run["color"],
                lw=LINE_WIDTH_VOL, label=f"{run['label']} -- {LABEL_VOL}")
        ax.plot(t / t_scale, R_e / r_scale, "--", color=run["color"],
                lw=LINE_WIDTH_ETA, alpha=0.8,
                label=f"{run['label']} -- {LABEL_ETA}")

    if not any_loaded:
        print("\n  [info] nothing loaded -- run the sims first. No plot written.")
        return

    # ---- normalized drive overlay (right axis) ----------------------------
    if SHOW_DRIVE:
        t_d = np.linspace(0.0, t_end_seen, 2000)
        ax2 = ax.twinx()
        ax2.plot(t_d / t_scale, np.sin(2.0 * np.pi * F_DRIVE * t_d),
                 color=DRIVE_COLOR, lw=DRIVE_LW, ls=":", zorder=0)
        ax2.set_ylabel(DRIVE_YLABEL, color=DRIVE_COLOR,
                       fontsize=FONT_SIZE_LABEL)
        ax2.tick_params(axis="y", colors=DRIVE_COLOR,
                        labelsize=FONT_SIZE_TICK)
        ax2.set_ylim(-1.05, 1.05)

    ax.axhline(1.0 if NONDIM_RADIUS else R0, color="k", lw=0.5, ls=":",
               alpha=0.4, label="initial $R_0$")
    ax.set_xlabel(XLABEL_STR if NONDIM_TIME else r"$t$ [s]",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_STR if NONDIM_RADIUS else r"$R$ [m]",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_title(TITLE_STR, fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc=LEGEND_LOC)

    plt.tight_layout()
    out_png = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    out_eps = os.path.join(IMG_DIR, f"{SAVE_NAME}.eps")
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_eps, dpi=DPI, bbox_inches="tight")
    print(f"\n  wrote {out_png}")
    print(f"  wrote {out_eps}")
    plt.close(fig)


if __name__ == "__main__":
    main()
