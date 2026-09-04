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
# OVERRIDE — active: INCLINE / HPC runs.
RUNS = [
     dict(label="LowAmp NSCBC (A = 0.001 bar)",
          out_dir="/mmfs1/home/ttryon/flames/bin/tests/FlowDrivenBubble/output_LowAmp_NSCBC",
          color="tab:blue"),
# Local WSL diagnostic runs (2026-09-03 slaving A/B) — uncomment to re-analyze:
#     dict(label="LowAmp NSCBC (slaved; died 2.03 ms)",
#          out_dir=_repo("tests", "FlowDrivenBubble", "output_LowAmp_NSCBC"),
#          color="tab:red"),
#     dict(label="LowAmp NSCBC (no slaving)",
#          out_dir=_repo("tests", "FlowDrivenBubble", "output_LowAmp_NSCBC_noslave"),
#          color="tab:blue"),
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

# Volume-radius baseline:
#   "R0"    -- normalize R_vol by the nominal R0 (absolute; the default
#              behavior, correct when the plotfiles are trustworthy).
#   "frame" -- normalize R_vol by ITS OWN value at plotfile index
#              BASELINE_FRAME (0-based; 1 = the 2nd plotfile).  Use when the
#              absolute volume integral carries a constant contamination
#              (e.g. the INCLINE R_vol/R0 ~ 1.96 flat curve): the derived
#              baseline is printed, and the normalized curve reveals the
#              TRUE relative oscillation.  Discriminator, printed per run:
#              if the volume contamination is MULTIPLICATIVE (a factor, e.g.
#              a wrong symmetry count) the normalized volume amplitude will
#              MATCH the eta-radius amplitude; if it is ADDITIVE (constant
#              spurious volume) the normalized amplitude will be strongly
#              DILUTED relative to the eta radius.
R_VOL_BASELINE = "frame"
BASELINE_FRAME = 1

# ===== EXTRACTION KNOBS =====
ETA_THRESHOLD = 0.5     # interface level for the contour radius
N_RADIAL_BINS = 400     # radial bins for the angle-averaged eta(r) profile
R_BIN_MAX     = 0.04    # outer radius of the binned profile [m] (= octant edge)

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
LEGEND_LOC = "upper left"

# ===== AXIS LABELS / TITLE (every string the user might rephrase) =====
TITLE_STR    = "Flow-Driven Bubble (3D octant): R(t)"
XLABEL_STR   = r"$t / T_{drive}$"
YLABEL_STR   = r"$R / R_0$"
LABEL_VOL    = "volume"       # appended to run label, e.g. "LowAmp -- volume"
LABEL_ETA    = r"$\eta=0.5$"
NONDIM_TIME   = True    # plot t / T_drive instead of t [s]
NONDIM_RADIUS = True    # plot R / R0 instead of R [m]
SAVE_NAME     = "FlowDrivenBubble_radius"

# ===== BUBBLE-SHAPE GIF =====
# For every plotfile, render the eta field on the z ~ 0 midplane slice (the
# bubble cross-section through its center) with the eta = 0.5 contour
# overlaid, write one PNG per frame into Images/ShapeFrames_<label>/ (pull
# individual frames into figures from there), and assemble an animated GIF
# in Images/ (one per run).
SHAPE_GIF      = True
SHAPE_RES      = 512          # slice raster resolution [px]
SHAPE_CMAP     = "RdBu"       # eta colormap (gas red -> liquid blue)
SHAPE_MS_PER_FRAME = 80       # GIF frame duration [ms]
SHAPE_MAX_FRAMES   = 400      # safety cap (subsamples evenly if exceeded)
SHAPE_LEVELS   = [0.01, 0.1, 0.5, 0.9, 0.99]   # eta contour levels drawn
SHAPE_SMOOTH_CELLS = 0.75     # gaussian smoothing of the raster, in units of
                              # FINEST-CELL widths (sub-cell -> kills the
                              # pixel staircase without moving the contours)

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
            # First-frame forensics (always printed): every input to the
            # volume radius, so a wrong R_vol(0) identifies its own cause.
            if not times:
                lev = np.array(reg["index", "grid_level"]).astype(int)
                print(f"    [frame0] sym={_sym_factor(ds):.0f}  "
                      f"rawV={Vg / _sym_factor(ds):.4e}  "
                      f"domainV={float(np.sum(vol)):.4e}  "
                      f"R_vol/R0={(3.0 * Vg / (4.0 * np.pi)) ** (1.0/3.0) / R0:.4f}")
                for L in range(int(lev.max()) + 1):
                    m = lev == L
                    print(f"    [frame0] level {L}: cells={int(m.sum())}, "
                          f"gasV={float(np.sum((alpha_g * vol)[m])):.4e}")
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

            # ---- far-field gas-fraction diagnostic ----------------------
            # A nonzero far-field (1 - eta) inflates the volume radius with a
            # CONSTANT offset while leaving the eta=0.5 crossing untouched
            # (the crossing takes the FIRST rise from the center) -- the
            # signature is a flat, too-large R_vol over an oscillating R_eta.
            # frame-0 IC discriminator: a healthy IC has interior eta ~ 0
            # (pure gas) and far-field eta ~ 1 (pure liquid).  A FLAT-0.5
            # field (broken IC: the tanh smearing width failed to evaluate,
            # tanh(x/inf) = 0 -> eta = 0.5 everywhere) shows interior ~ 0.5
            # AND far-field ~ 0.5 -- and the run is NOT a bubble benchmark.
            if not times:
                inner = r < 0.5 * R0
                if np.any(inner):
                    ein = float(np.average(eta[inner], weights=vol[inner]))
                    print(f"    [frame0] interior mean(eta) [r<0.5R0] = "
                          f"{ein:.4f}  (healthy IC ~ 0; flat-0.5 broken IC "
                          f"~ 0.5)")
            ff = r > 1.5 * R0
            if np.any(ff):
                ff_gas = float(np.average(np.clip(1.0 - eta[ff], 0.0, 1.0),
                                          weights=vol[ff]))
                if not times:
                    print(f"    [frame0] far-field mean(eta) [r>1.5R0] = "
                          f"{1.0 - ff_gas:.4f}  (healthy IC ~ 1)")
                if ff_gas > 1.0e-3:
                    print(f"    [warn] {os.path.basename(pf)}: far-field "
                          f"mean(1-eta) = {ff_gas:.3e} at r > 1.5 R0 -- "
                          f"R_vol is inflated by spurious far-field gas "
                          f"(sym={_sym_factor(ds):.0f}, rawV={Vg:.3e})")

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
# BUBBLE-SHAPE FRAMES + GIF
# ============================================================================

def render_shape_frames(out_dir, label, color):
    """Render the z ~ 0 midplane eta slice of every plotfile to a PNG frame
    (Images/ShapeFrames_<label>/frame_NNNN.png) and assemble them into
    Images/FlowDrivenBubble_shape_<label>.gif.  Skips corrupt plotfiles."""
    import yt
    from PIL import Image
    yt.funcs.mylog.setLevel(40)

    safe = "".join(c if c.isalnum() else "_" for c in label).strip("_")
    frame_dir = os.path.join(IMG_DIR, f"ShapeFrames_{safe}")
    os.makedirs(frame_dir, exist_ok=True)

    pfs = _list_plotfiles(out_dir)
    if len(pfs) > SHAPE_MAX_FRAMES:
        idx = np.linspace(0, len(pfs) - 1, SHAPE_MAX_FRAMES).astype(int)
        pfs = [pfs[i] for i in sorted(set(idx))]

    entries = []   # (time, png_path)
    for pf in pfs:
        try:
            ds = yt.load(pf)
            lo = np.array(ds.domain_left_edge.to_value())
            hi = np.array(ds.domain_right_edge.to_value())
            # Slice just inside the domain at the bubble-center z plane
            # (octant: CENTER sits on the zlo face -> nudge half a fine cell in).
            dz_fine = (hi[2] - lo[2]) / (ds.domain_dimensions[2] * 2 ** ds.max_level)
            z0 = max(CENTER[2], lo[2]) + 0.5 * dz_fine
            slc = ds.slice(2, z0)
            cen = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z0)
            wid = ((hi[0] - lo[0], "cm"), (hi[1] - lo[1], "cm"))
            frb = slc.to_frb(width=wid[0], resolution=SHAPE_RES, center=cen,
                             height=wid[1])
            eta2d = np.array(frb["eta"])
            n_fine = int(ds.domain_dimensions[0]) * 2 ** int(ds.max_level)
            entries.append((float(ds.current_time), eta2d, lo, hi, n_fine))
        except Exception:
            continue

    if len(entries) < 2:
        print(f"  [shape] <2 usable frames for {label} -- GIF skipped")
        return

    entries.sort(key=lambda e: e[0])
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None
    png_paths = []
    for i, (t, eta2d, lo, hi, _n_fine_across) in enumerate(entries):
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        ext = [lo[0] / R0, hi[0] / R0, lo[1] / R0, hi[1] / R0]
        # Sub-cell gaussian smoothing: the FRB raster reproduces the finest
        # cells as flat blocks; smoothing by ~0.75 cell widths (in px) turns
        # the staircase into the smooth tanh band the data represents
        # without displacing the contour positions.
        eta_s = eta2d
        if gaussian_filter is not None and SHAPE_SMOOTH_CELLS > 0:
            # px per finest cell = SHAPE_RES / n_finest_cells_across
            eta_s = gaussian_filter(eta2d, sigma=SHAPE_SMOOTH_CELLS
                                    * SHAPE_RES / max(1, _n_fine_across),
                                    mode="nearest")
        xg = np.linspace(ext[0], ext[1], eta_s.shape[1])
        yg = np.linspace(ext[2], ext[3], eta_s.shape[0])
        ax.imshow(eta_s, origin="lower", extent=ext, cmap=SHAPE_CMAP,
                  vmin=0.0, vmax=1.0, interpolation="bilinear", alpha=0.85)
        # Band structure: thin lines at the outer levels, bold at eta = 0.5.
        cs = ax.contour(xg, yg, eta_s, levels=SHAPE_LEVELS,
                        colors=["0.25"] * len(SHAPE_LEVELS),
                        linewidths=[0.7 if abs(l - 0.5) > 1e-9 else 1.8
                                    for l in SHAPE_LEVELS],
                        linestyles=["--" if abs(l - 0.5) > 1e-9 else "-"
                                    for l in SHAPE_LEVELS])
        # Pin each contour label where the contour crosses the y = x
        # diagonal: fixed, predictable label positions frame to frame
        # (matplotlib's automatic placement wanders along the contour and
        # makes the GIF jitter).  Levels that never cross the diagonal are
        # left unlabeled.
        diag = np.diagonal(eta_s)          # eta along y = x (FRB is square)
        sdia = np.linspace(ext[0], ext[1], len(diag)) * np.sqrt(2.0)
        manual_pos = []
        for lvl in SHAPE_LEVELS:
            cross = np.where(np.diff(np.sign(diag - lvl)) != 0)[0]
            if len(cross) > 0:
                i0 = cross[0]
                frac = (lvl - diag[i0]) / (diag[i0 + 1] - diag[i0]
                                           if diag[i0 + 1] != diag[i0] else 1.0)
                s_lab = (sdia[i0] + frac * (sdia[i0 + 1] - sdia[i0])) / np.sqrt(2.0)
                manual_pos.append((s_lab, s_lab))
        if manual_pos:
            ax.clabel(cs, fmt="%.2g", fontsize=7, inline=True,
                      manual=manual_pos)
        # initial-radius reference circle
        th = np.linspace(0, np.pi / 2, 100)
        ax.plot(np.cos(th), np.sin(th), ls=":", color="0.35", lw=0.8)
        ax.set_xlabel(r"$x / R_0$", fontsize=FONT_SIZE_LABEL)
        ax.set_ylabel(r"$y / R_0$", fontsize=FONT_SIZE_LABEL)
        ax.set_title(f"{label}   $t/T_{{drive}}$ = {t / T_DRIVE:.3f}",
                     fontsize=FONT_SIZE_LABEL)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=FONT_SIZE_TICK)
        png = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(png)

    # assemble GIF (frames stay on disk for figure-pulling)
    from PIL import Image
    imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in png_paths]
    gif = os.path.join(IMG_DIR, f"FlowDrivenBubble_shape_{safe}.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=SHAPE_MS_PER_FRAME, loop=0)
    print(f"  [shape] wrote {len(png_paths)} frames -> {frame_dir}")
    print(f"  [shape] wrote {gif}")


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

        # Volume-curve baseline (see R_VOL_BASELINE): nominal R0, or the
        # curve's own value at BASELINE_FRAME (2nd plotfile by default).
        vol_base = r_scale
        vol_lab = f"{run['label']} -- {LABEL_VOL}"
        if R_VOL_BASELINE == "frame" and len(R_v) > BASELINE_FRAME:
            vol_base = float(R_v[BASELINE_FRAME])
            print(f"    derived volume baseline (frame {BASELINE_FRAME}): "
                  f"R_vol = {vol_base:.6e} m = {vol_base / R0:.4f} R0"
                  + ("  [!! absolute volume radius is off nominal by >2% -- "
                     "see frame0 forensics]" if abs(vol_base / R0 - 1) > 0.02
                     else ""))
            vol_lab += " (self-normalized)"
            # additive-vs-multiplicative discriminator
            fin_v = np.isfinite(R_v); fin_e = np.isfinite(R_e)
            if fin_v.sum() > 3 and fin_e.sum() > 3:
                amp_v = (np.nanmax(R_v) - np.nanmin(R_v)) / vol_base
                amp_e = (np.nanmax(R_e) - np.nanmin(R_e)) / R0
                print(f"    osc amplitude: volume {amp_v:.4f} vs eta=0.5 "
                      f"{amp_e:.4f} -> "
                      + ("consistent (multiplicative offset, e.g. symmetry "
                         "factor)" if amp_v > 0.5 * amp_e else
                         "volume DILUTED (additive spurious volume)"))
        ax.plot(t / t_scale, R_v / vol_base, "-", color=run["color"],
                lw=LINE_WIDTH_VOL, label=vol_lab)
        ax.plot(t / t_scale, R_e / r_scale, "--", color=run["color"],
                lw=LINE_WIDTH_ETA, alpha=0.8,
                label=f"{run['label']} -- {LABEL_ETA}")

        if SHAPE_GIF:
            render_shape_frames(out_dir, run["label"], run["color"])

    if not any_loaded:
        print("\n  [info] nothing loaded -- run the sims first. No plot written.")
        return

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
