# -*- coding: utf-8 -*-
"""
===============================================================================
SCH20 COLLAPSING-BUBBLE ANALYSIS (3D, spherical)  --  Case 2, high pressure ratio
(p_inf / p_b = 1427).  Focus: minimum radius R_min and collapse time.
===============================================================================

PURPOSE:
    Validate the flames2 simulation of the Schmidmayer-Bryngelson-Colonius
    (2020) Table 1, Case 2 spherical air bubble in water against three
    analytical bubble-dynamics references:

        - 3D Keller-Miksis        (km)     -- liquid-compressibility benchmark
        - 3D Rayleigh-Plesset     (rp)     -- incompressible 3D context
        - 2D Chen cylindrical RPE (chen2d) -- geometric match for the 2D
                                              Cartesian (cylindrical) sim

    At p_inf / p_b = 1427 the bubble undergoes a violent, near-singular
    collapse. The headline validation metrics are therefore the geometric
    extremes of the first collapse:

        - minimum radius            R_min / R_0
        - collapse time t / tau_c   at which R_min occurs

    where tau_c = 0.915 R0 sqrt(rho_l / p_inf) is the Rayleigh collapse time
    (Sch20 Eq. 30). For 2D Cartesian sims the geometric reference is Chen 2D
    cylindrical (logarithmic far-field); KM remains the compressibility
    benchmark even though the 2D-vs-3D geometry is offset.

USAGE:
    python tests/FlowRayleighPlesset/reference/analyze_Sch20_Collapsing_3D.py

OUTPUTS:
    Images/Sch20_Collapsing.png   (and .eps for the paper)
    Console summary: minimum radius R_min/R0 and collapse time t/tau_c at
    R_min, for sim vs KM / RP / Chen2D.

REFERENCES:
    Schmidmayer, Bryngelson, Colonius, "An assessment of multiregion methods
        for two-phase flow simulations", J. Comput. Phys. 402 (2020) 109080,
        Sec. 4 (problem setup), Table 1 (ICs).
    Keller & Miksis, "Bubble oscillations of large amplitude", JASA 68 (1980).
    Brennen, "Cavitation and Bubble Dynamics" (1995), Sec. 4.4 (linearized
        RP / KM, acoustic-radiation damping).

===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))

# Bring in the shared RPE / KM / Chen2D ODE solver (DO NOT modify it).
sys.path.insert(0, os.path.join(_HERE, '..', 'OtherTests', 'reference'))
from rayleigh_plesset_solver import (                                  # noqa: E402
    Params, solve,
    rayleigh_collapse_time,
    linear_natural_frequency_3d, linear_natural_frequency_chen2d,
)


# ============================================================================
# ============================  CONFIGURATION  ===============================
# Everything tunable lives in this block. Adjust here -- no hunting needed.
# ============================================================================

# ===== MODEL TOGGLES =====
SHOW_KM_3D   = True      # 3D Keller-Miksis
SHOW_RPE_3D  = True      # 3D Rayleigh-Plesset
SHOW_CHEN_2D = False      # 2D Chen cylindrical
SHOW_SIM     = True      # flames2 simulation R(t)

# ===== PHYSICAL PARAMETERS =====
# Sch20 Table 1, Case 2 (p_inf / p_b = 1427). Must match Sch20_Collapsing input.
#   c_l = sqrt(gamma_l (p_inf + pi_l) / rho_l) = sqrt(2.35*(5e6+1e9)/1000)
P = Params(
    rho_l     = 1000.0,
    p_inf     = 5.0e6,       # far-field liquid pressure
    mu_l      = 0.0,
    c_l       = 1539.0,      # Tammann water sound speed (KM only)
    gamma_gas = 1.4,
    p_g0      = 3550.0,      # initial bubble pressure p_b
    p_v       = 0.0,
    sigma     = 0.0,
    R0        = 0.02,
    Rdot0     = 0.0,
    r_inf     = 0.20,        # Chen-2D log far-field cutoff = half-domain
)

# ===== TIME / DOMAIN =====
# tau_c = 0.915 R0 sqrt(rho_l / p_inf) ~ 2.588e-4 s ; stop_time = 4.0e-4 s.
T_END = 4.0e-4

# Path to the AMReX plotfile output dir. Default is relative to the repo
# (run from <repo>/bin); the absolute OVERRIDE line below can be uncommented
# and edited for a different machine / scratch location.
OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowRayleighPlesset", "output_Sch20_Collapsing_3D",
))
# OVERRIDE (uncomment + edit for your machine):
OUTPUT_DIR = os.path.normpath("/mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Collapsing_3D")

# ===== PLOT STYLING (publication knobs) =====
FONT_SIZE_TITLE  = 16
FONT_SIZE_LABEL  = 14
FONT_SIZE_LEGEND = 12
FONT_SIZE_TICK   = 11

LINE_WIDTH_KM   = 2.5
LINE_WIDTH_RPE  = 1.5
LINE_WIDTH_CHEN = 2.5
LINE_WIDTH_SIM  = 1.5
MARKER_SIZE_SIM = 4

COLOR_KM   = 'tab:blue'
COLOR_RPE  = '0.5'
COLOR_CHEN = 'tab:green'
COLOR_SIM  = 'tab:red'

DPI        = 180
FIG_WIDTH  = 11
FIG_HEIGHT = 6.5
LEGEND_LOC = 'best'

# ===== AXIS LABELS / TITLE (every string the user might rephrase) =====
TITLE_STR   = "Sch20 Collapsing Bubble (3D): R(t)"
XLABEL_STR  = r"$t / \tau_c$"
YLABEL_STR  = r"$R / R_0$"
LABEL_KM    = "Keller--Miksis (3D)"
LABEL_RPE   = "Rayleigh--Plesset (3D)"
LABEL_CHEN  = "Chen (2D cylindrical)"
LABEL_SIM   = "flames2 (3D)"
NONDIM_TIME   = True     # plot t / tau_c instead of t [s]
NONDIM_RADIUS = True     # plot R / R0  instead of R [m]
SAVE_NAME     = "Sch20_Collapsing_3D"

# ============================================================================
# ==========================  END CONFIGURATION  =============================
# ============================================================================

IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# COLLAPSE-METRIC HELPERS
# ============================================================================

def find_global_minimum(t, R):
    """Return (t_at_min, R_min, idx) for the global minimum of R(t).

    NaNs (e.g. plotfiles where the eta-crossing was not found) are ignored.
    Returns (nan, nan, -1) if R has no finite samples.
    """
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    finite = np.isfinite(R)
    if not np.any(finite):
        return float('nan'), float('nan'), -1
    idx_finite = np.where(finite)[0]
    j = idx_finite[np.argmin(R[idx_finite])]
    return float(t[j]), float(R[j]), int(j)


def find_local_minima(t, R):
    """Return arrays (t_min, R_min) of all interior local minima of R(t)."""
    if len(R) < 3:
        return np.array([]), np.array([])
    is_min = (R[1:-1] < R[:-2]) & (R[1:-1] < R[2:])
    idx = np.where(is_min)[0] + 1
    return t[idx], R[idx]


def find_local_maxima(t, R):
    """Return arrays (t_max, R_max) of all interior local maxima."""
    if len(R) < 3:
        return np.array([]), np.array([])
    is_max = (R[1:-1] > R[:-2]) & (R[1:-1] > R[2:])
    idx = np.where(is_max)[0] + 1
    return t[idx], R[idx]


def _offset_ray_radial(ds, axis="x"):
    """Ray PARALLEL to `axis` but offset by half a finest cell off the other two
    axes, so it passes through cell CENTERS instead of the on-axis cell EDGES.
    The exact y=z=0 axis lies on 3D cell edges, where yt's ray sampling is
    ambiguous -> spurious eta=0.5 crossings (the ~0.64 R0 stray dots at collapse).
    Returns (radius, eta) sorted by radius, with radius = sqrt(s^2 + 2*off^2)
    (the true distance from the origin along the offset ray)."""
    L = float(ds.domain_right_edge[0])
    dx_base = float(ds.domain_width[0]) / int(ds.domain_dimensions[0])
    off = 0.5 * dx_base / (2 ** int(ds.max_level))     # half a finest cell
    if axis == "x":
        a, b, key = [0.0, off, off], [L, off, off], "x"
    elif axis == "y":
        a, b, key = [off, 0.0, off], [off, L, off], "y"
    else:
        a, b, key = [off, off, 0.0], [off, off, L], "z"
    ray = ds.ray(ds.arr(a, "code_length"), ds.arr(b, "code_length"))
    s = np.array(ray[key]); eta = np.array(ray["eta"])
    rad = np.sqrt(s * s + 2.0 * off * off)
    o = np.argsort(rad)
    return rad[o], eta[o]


def extract_radius_history_robust(amrex_output_dir, eta_threshold=0.5, axis="x"):
    """Same as rayleigh_plesset_solver.extract_radius_history but tolerant
    of individual corrupt / partially-written plotfiles.

    Walks every '*cell' subdirectory of `amrex_output_dir`, attempts to load
    each one with yt, and silently SKIPS any that raise an exception (typical
    cause: a run interrupted while a plotfile was still being written, leaving
    an incomplete Cell_H / Header behind).

    Returns (times, radii, n_skipped) with times/radii sorted by time. Never
    raises -- if the whole directory is bad, returns empty arrays.
    """
    import yt
    yt.funcs.mylog.setLevel(40)

    if not os.path.isdir(amrex_output_dir):
        return np.array([]), np.array([]), 0

    plot_files = sorted(
        os.path.join(amrex_output_dir, d)
        for d in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell")
    )
    if not plot_files:
        return np.array([]), np.array([]), 0

    times, radii = [], []
    n_skipped = 0
    for pf in plot_files:
        try:
            ds = yt.load(pf)
            x, eta = _offset_ray_radial(ds, axis)   # off-axis ray -> radial coord (Option a)
            idx = np.where(eta >= eta_threshold)[0]
            if len(idx) == 0:
                radii.append(np.nan)
            else:
                i = idx[0]
                if i == 0:
                    radii.append(x[0])
                else:
                    frac = (eta_threshold - eta[i - 1]) / (eta[i] - eta[i - 1])
                    radii.append(x[i - 1] + frac * (x[i] - x[i - 1]))
            times.append(float(ds.current_time))
        except Exception:
            # Bad / partial plotfile -- skip and keep going.
            n_skipped += 1
            continue

    if not times:
        return np.array([]), np.array([]), n_skipped

    times = np.array(times)
    radii = np.array(radii)
    order = np.argsort(times)
    return times[order], radii[order], n_skipped


# ============================================================================
# DIFFUSE ETA-BAND DIAGNOSTIC
# ============================================================================
# eta rises 0 -> 1 from gas (bubble center) to liquid (outside), so along the
# +x ray eta=0.1 is the INNER band edge, eta=0.9 the OUTER edge, eta=0.5 the
# nominal R(t).  The shaded region between the outer thresholds is the diffuse
# interface band; a band that fattens over time = numerical interface smearing
# (worth watching through the collapse, where it can over/under-shoot R_min).
BAND_THRESHOLDS = (0.1, 0.5, 0.9)


def extract_eta_band_history(amrex_output_dir, thresholds=BAND_THRESHOLDS,
                             axis="x", x_max=None):
    """Per frame, the radius where eta crosses each threshold (linear-interp).
    Returns (times, {thr: radii}); skips corrupt/partial plotfiles silently."""
    import yt
    yt.funcs.mylog.setLevel(40)
    empty = {t: np.array([]) for t in thresholds}
    if not os.path.isdir(amrex_output_dir):
        return np.array([]), empty
    pfs = sorted(os.path.join(amrex_output_dir, d) for d in os.listdir(amrex_output_dir)
                 if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell"))
    times, bands = [], {t: [] for t in thresholds}
    for pf in pfs:
        try:
            ds = yt.load(pf)
            r, eta = _offset_ray_radial(ds, axis)   # off-axis ray -> radial coord (Option a)
            if x_max is not None:
                m = (r >= 0.0) & (r <= float(x_max)); r, eta = r[m], eta[m]
            for thr in thresholds:
                if len(r) == 0:
                    bands[thr].append(np.nan); continue
                idx = np.where(eta >= thr)[0]
                if len(idx) == 0:
                    bands[thr].append(np.nan)
                elif idx[0] == 0:
                    bands[thr].append(r[0])
                else:
                    i = idx[0]
                    frac = (thr - eta[i - 1]) / (eta[i] - eta[i - 1])
                    bands[thr].append(r[i - 1] + frac * (r[i] - r[i - 1]))
            times.append(float(ds.current_time))
        except Exception:
            continue
    if not times:
        return np.array([]), empty
    times = np.array(times); o = np.argsort(times)
    return times[o], {t: np.array(bands[t])[o] for t in thresholds}


def plot_eta_band(times_s, bands, curves=None):
    """Shaded diffuse eta-band radius vs time -> Images/{SAVE_NAME}_eta_band.*"""
    if len(times_s) < 2:
        print("  [eta-band] <2 usable frames -- skipped")
        return
    thrs = sorted(bands)
    tu = globals().get("TIME_UNIT", "us")
    if tu == "ms":
        tf, tl = 1.0e3, r"$t$ [ms]"
    elif tu == "us":
        tf, tl = 1.0e6, r"$t$ [$\mu$s]"
    else:
        tau = rayleigh_collapse_time(P)
        tf = 1.0 / tau if (np.isfinite(tau) and tau > 0) else 1.0
        tl = r"$t / \tau_c$"
    R0 = P.R0
    lo, hi = thrs[0], thrs[-1]
    fw = globals().get("FIG_WIDTH", 11); fh = globals().get("FIG_HEIGHT", 6.5)
    fsl = globals().get("FONT_SIZE_LABEL", 14); fst = globals().get("FONT_SIZE_TITLE", 16)
    fsk = globals().get("FONT_SIZE_TICK", 11); fsg = globals().get("FONT_SIZE_LEGEND", 12)
    dpi = globals().get("DPI", 180)
    fig, ax = plt.subplots(figsize=(fw, fh))
    ax.fill_between(times_s * tf, bands[lo] / R0, bands[hi] / R0,
                    color="tab:red", alpha=0.20,
                    label=rf"diffuse band $\eta\in[{lo:.2f},{hi:.2f}]$")
    for thr in thrs:
        ls, lw = ("-", 2.2) if abs(thr - 0.5) < 1e-9 else ("--", 1.3)
        lab = rf"$\eta={thr:.2f}$" + ("  ($R/R_0$)" if abs(thr - 0.5) < 1e-9 else "")
        ax.plot(times_s * tf, bands[thr] / R0, ls, lw=lw, label=lab)
    ax.axhline(1.0, color="0.6", ls=":", lw=0.8)
    # overlay analytical bubble-dynamics references (RPE / KM) for comparison
    for c in (curves or []):
        if c.get("key") in ("rp", "km"):
            ax.plot(np.asarray(c["t"]) * tf, np.asarray(c["R"]) / R0,
                    c.get("ls", "-"), color=c.get("color", "k"),
                    lw=c.get("lw", 1.6), alpha=0.9, label=c.get("label", c["key"]))
    ax.set_xlabel(tl, fontsize=fsl)
    ax.set_ylabel(r"$R / R_0$", fontsize=fsl)
    ax.set_title(f"{SAVE_NAME}: diffuse eta-band radius vs time",
                 fontsize=fst, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=fsk)
    ax.legend(fontsize=fsg, loc="best")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}_eta_band")
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".eps", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    bw = (bands[hi] - bands[lo]) / R0
    with np.errstate(invalid="ignore"):
        print(f"  [eta-band] wrote {out}.png  (band width R/R0: "
              f"start={bw[0]:.4f}, end={bw[-1]:.4f}, max={np.nanmax(bw):.4f})")


# ============================================================================
# MAIN
# ============================================================================



# ============================================================================
# VOLUME-BASED RADIUS  (Sch20 eq. 29)
# ============================================================================
# R = (3 V_b / 4pi)^(1/3),  V_b = sum_cells alpha_g * V_cell,  alpha_g = 1 - eta.
# This is Sch20's actual radius definition -- a robust volume integral over the
# whole gas phase, NOT a single-ray eta=0.5 crossing.  It is immune to the
# interface smearing / grid-anisotropy that makes the eta=0.5 ray under-read R
# at the violent collapse (see the eta-band plot).  AMR-correct: yt returns only
# leaf cells, so no double-counting of covered coarse cells.  Restricted to a
# +/-BOX_HALF box about the origin (the far field is eta=1 -> alpha_g=0, no
# contribution) so it stays fast on the 320^3 _Large grids.
BOX_HALF = 0.2   # m; ~10 R0 -- the bubble (R<=R0=0.02) lives well inside this


def extract_radius_volume_history(amrex_output_dir, box_half=BOX_HALF):
    """Per frame, R from the gas-volume integral (Sch20 eq. 29).
    Returns (times, radii); skips corrupt/partial plotfiles silently."""
    import yt
    yt.funcs.mylog.setLevel(40)
    if not os.path.isdir(amrex_output_dir):
        return np.array([]), np.array([])
    pfs = sorted(os.path.join(amrex_output_dir, d) for d in os.listdir(amrex_output_dir)
                 if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell"))
    times, radii = [], []
    for pf in pfs:
        try:
            ds = yt.load(pf)
            dle = ds.domain_left_edge
            dre = ds.domain_right_edge
            # Octant/symmetry detection: a lo edge sitting at the origin (the
            # bubble center) means that axis is a symmetry plane, so only half the
            # bubble lies along it.  full domain: all lo<0 -> factor 1; octant:
            # all lo=0 -> factor 8.  Scale V_gas by 2^(#symmetry axes).
            sym_factor = 1
            for _d in range(3):
                if float(dle[_d]) > -1e-6:
                    sym_factor *= 2
            if not times:
                print("  [R-volume] geometry: %s -> V_gas x%d"
                      % ("OCTANT" if sym_factor > 1 else "FULL domain", sym_factor))
            lo = [max(-box_half, float(dle[_d])) for _d in range(3)]
            hi = [min( box_half, float(dre[_d])) for _d in range(3)]
            reg = ds.box(ds.arr(lo, "code_length"), ds.arr(hi, "code_length"))
            eta = np.array(reg["eta"])
            try:
                vol = np.array(reg["index", "cell_volume"])
            except Exception:
                vol = np.array(reg["cell_volume"])
            alpha_g = np.clip(1.0 - eta, 0.0, 1.0)
            V_b = float(np.sum(alpha_g * vol)) * sym_factor
            radii.append((3.0 * V_b / (4.0 * np.pi)) ** (1.0 / 3.0))
            times.append(float(ds.current_time))
        except Exception:
            continue
    if not times:
        return np.array([]), np.array([])
    t = np.array(times); r = np.array(radii); o = np.argsort(t)
    return t[o], r[o]


def plot_radius_volume(times_s, R_vol, curves, R_eta_t=None, R_eta=None):
    """Volume-based R/R0 vs time, overlaid with KM/RP (and the eta=0.5 ray R for
    contrast) -> Images/{SAVE_NAME}_R_volume.*"""
    if len(times_s) < 2:
        print("  [R-volume] <2 usable frames -- skipped")
        return
    tu = globals().get("TIME_UNIT", "us")
    if tu == "ms":
        tf, tl = 1.0e3, r"$t$ [ms]"
    elif tu == "us":
        tf, tl = 1.0e6, r"$t$ [$\mu$s]"
    else:
        tau = rayleigh_collapse_time(P)
        tf = 1.0 / tau if (np.isfinite(tau) and tau > 0) else 1.0
        tl = r"$t / \tau_c$"
    R0 = P.R0
    fw = globals().get("FIG_WIDTH", 11); fh = globals().get("FIG_HEIGHT", 6.5)
    fsl = globals().get("FONT_SIZE_LABEL", 14); fst = globals().get("FONT_SIZE_TITLE", 16)
    fsk = globals().get("FONT_SIZE_TICK", 11); fsg = globals().get("FONT_SIZE_LEGEND", 12)
    dpi = globals().get("DPI", 180)
    fig, ax = plt.subplots(figsize=(fw, fh))
    # analytical references
    for c in (curves or []):
        if c.get("key") in ("rp", "km"):
            ax.plot(np.asarray(c["t"]) * tf, np.asarray(c["R"]) / R0, c.get("ls", "-"),
                    color=c.get("color", "k"), lw=c.get("lw", 1.6), alpha=0.9,
                    label=c.get("label", c["key"]))
    # eta=0.5 ray radius for contrast (the old measure), if available
    if R_eta is not None and R_eta_t is not None and len(R_eta) > 1:
        ax.plot(np.asarray(R_eta_t) * tf, np.asarray(R_eta) / R0, ":", color="tab:gray",
                lw=1.3, alpha=0.7, label=r"hydro2 $\eta{=}0.5$ ray")
    # the new volume-based radius
    ax.plot(times_s * tf, R_vol / R0, "-o", color="tab:purple", lw=2.0, ms=3,
            label=r"hydro2 $R=(3V_b/4\pi)^{1/3}$ (gas volume)")
    ax.set_xlabel(tl, fontsize=fsl)
    ax.set_ylabel(r"$R / R_0$", fontsize=fsl)
    ax.set_title(f"{SAVE_NAME}: bubble radius from GAS VOLUME (Sch20 eq. 29)",
                 fontsize=fst, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=fsk)
    ax.legend(fontsize=fsg, loc="best")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}_R_volume")
    fig.savefig(out + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out + ".eps", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    imin = int(np.argmin(R_vol))
    print(f"  [R-volume] wrote {out}.png  (R_min/R0={R_vol[imin] / R0:.4f} at "
          f"t={times_s[imin] * tf:.4f}; sanity R(0)/R0={R_vol[0] / R0:.4f} -- expect ~1.0)")


def main():
    tau_c = rayleigh_collapse_time(P)
    t_scale = tau_c if (NONDIM_TIME and np.isfinite(tau_c) and tau_c > 0) else 1.0
    r_scale = P.R0  if NONDIM_RADIUS else 1.0

    print("=" * 70)
    print("SCH20 COLLAPSING-BUBBLE ANALYSIS (3D, spherical)  (Case 2, p_inf/p_b = 1427)")
    print("=" * 70)
    print(f"  rho_l       = {P.rho_l}")
    print(f"  p_inf       = {P.p_inf}        ({P.p_inf/P.p_g0:.1f} x p_b)")
    print(f"  p_b (p_g0)  = {P.p_g0}")
    print(f"  c_l         = {P.c_l}")
    print(f"  sigma       = {P.sigma}")
    print(f"  R0          = {P.R0}")
    print(f"  r_inf       = {P.r_inf}  (Chen-2D log far-field cutoff)")
    print(f"  T_END       = {T_END*1e3:.4f} ms")
    print(f"  tau_c       = {tau_c*1e3:.4f} ms  (Rayleigh collapse time)")
    print(f"  output dir  = {OUTPUT_DIR}")
    print()

    # ---- Linearized natural frequencies (context only) --------------------
    omega0_3D     = linear_natural_frequency_3d(P)
    omega0_chen2d = linear_natural_frequency_chen2d(P)
    T0_3D     = (2 * np.pi / omega0_3D)     if omega0_3D     > 0 else float('nan')
    T0_chen2d = (2 * np.pi / omega0_chen2d) if omega0_chen2d > 0 else float('nan')
    print("  Linearized natural-frequency predictions (context):")
    print(f"    3D RP / KM Minnaert : omega_0 = {omega0_3D:10.2f} rad/s  "
          f"T_0 = {T0_3D*1e3:.4f} ms")
    print(f"    Chen 2D Minnaert    : omega_0 = {omega0_chen2d:10.2f} rad/s  "
          f"T_0 = {T0_chen2d*1e3:.4f} ms")
    print()

    # ---- Analytical references --------------------------------------------
    curves = []   # list of dicts: key, label, color, lw, ls, t, R
    if SHOW_KM_3D:
        print("  Solving Keller-Miksis ODE ...")
        t_km, R_km, _ = solve(P, T_END, model="km")
        curves.append(dict(key="km", label=LABEL_KM, color=COLOR_KM,
                            lw=LINE_WIDTH_KM, ls='-', t=t_km, R=R_km))
    if SHOW_RPE_3D:
        print("  Solving 3D Rayleigh-Plesset ODE ...")
        t_rp, R_rp, _ = solve(P, T_END, model="rp")
        curves.append(dict(key="rp", label=LABEL_RPE, color=COLOR_RPE,
                            lw=LINE_WIDTH_RPE, ls='--', t=t_rp, R=R_rp))
    if SHOW_CHEN_2D:
        print("  Solving Chen 2D cylindrical RPE ODE ...")
        t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")
        curves.append(dict(key="chen2d", label=LABEL_CHEN, color=COLOR_CHEN,
                            lw=LINE_WIDTH_CHEN, ls='-', t=t_c2d, R=R_c2d))

    print("\n  Analytical collapse metrics (global minimum of R over T_END):")
    for c in curves:
        t_min, R_min, _ = find_global_minimum(c['t'], c['R'])
        c['t_min'] = t_min
        c['R_min'] = R_min
        if np.isfinite(R_min):
            print(f"    {c['label']:<26s}: R_min/R0 = {R_min/P.R0:.5f}   "
                  f"t(R_min)/tau_c = {t_min/tau_c:.4f}  "
                  f"(t = {t_min*1e3:.4f} ms)")
        else:
            print(f"    {c['label']:<26s}: R_min = n/a")

    # ---- Simulation extraction --------------------------------------------
    sim_loaded = False
    n_skipped  = 0
    t_sim = np.array([])
    R_sim = np.array([])
    if SHOW_SIM:
        if os.path.isdir(OUTPUT_DIR):
            t_sim, R_sim, n_skipped = extract_radius_history_robust(
                OUTPUT_DIR, eta_threshold=0.5, axis="x")
            sim_loaded = (len(t_sim) > 1)
            if n_skipped > 0:
                print(f"\n  [warn] skipped {n_skipped} corrupt / partial "
                      f"plotfile(s) in")
                print(f"         {OUTPUT_DIR}")
                print(f"         (loaded {len(t_sim)} usable frames.)")
        if not sim_loaded:
            if n_skipped == 0 and not os.path.isdir(OUTPUT_DIR):
                print(f"\n  [info] no AMReX output yet at {OUTPUT_DIR}")
                print(f"         analytical curves still plotted -- run sim first.")
            else:
                print(f"\n  [info] no usable plotfiles loaded from {OUTPUT_DIR}")
                print(f"         (skipped: {n_skipped}). Analytical curves only.")
    else:
        print("\n  [info] SHOW_SIM is False -- analytical curves only.")

    if sim_loaded:
        t_min_s, R_min_s, _ = find_global_minimum(t_sim, R_sim)
        print("\n  Simulation collapse metrics:")
        if np.isfinite(R_min_s):
            print(f"    sim R_min/R0       = {R_min_s/P.R0:.5f}")
            print(f"    sim t(R_min)/tau_c = {t_min_s/tau_c:.4f}  "
                  f"(t = {t_min_s*1e3:.4f} ms)")
            for c in curves:
                if np.isfinite(c['R_min']) and c['R_min'] > 0:
                    err_R = (R_min_s - c['R_min']) / c['R_min'] * 100
                    print(f"    R_min error vs {c['label']:<24s}: "
                          f"{err_R:+7.2f} %")
                if np.isfinite(c['t_min']) and c['t_min'] > 0:
                    err_t = (t_min_s - c['t_min']) / c['t_min'] * 100
                    print(f"    t(R_min) error vs {c['label']:<21s}: "
                          f"{err_t:+7.2f} %")
        else:
            print(f"    sim R_min = n/a (no finite radius samples)")

    # ---- Plot --------------------------------------------------------------
    plt.rcParams.update({
        'axes.titlesize':  FONT_SIZE_TITLE,
        'axes.labelsize':  FONT_SIZE_LABEL,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
    })

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for c in curves:
        if np.isfinite(c.get('R_min', np.nan)):
            mlbl = f", R_min/R0={c['R_min']/P.R0:.3f}"
        else:
            mlbl = ""
        ax.plot(c['t'] / t_scale, c['R'] / r_scale,
                c['ls'], color=c['color'], lw=c['lw'],
                label=f"{c['label']}{mlbl}")

    if sim_loaded:
        ax.plot(t_sim / t_scale, R_sim / r_scale, 'o', color=COLOR_SIM,
                ms=MARKER_SIZE_SIM, lw=LINE_WIDTH_SIM, alpha=0.7,
                label=f"{LABEL_SIM} (n={len(t_sim)})")

    ax.axhline(1.0 if NONDIM_RADIUS else P.R0, color='k', lw=0.5, ls=':',
               alpha=0.4, label='initial $R_0$')
    ax.set_xlabel(XLABEL_STR if NONDIM_TIME else r"$t$ [s]",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL_STR if NONDIM_RADIUS else r"$R$ [m]",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_title(TITLE_STR, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc=LEGEND_LOC)

    plt.tight_layout()
    # ---- Diffuse eta-band (interface thickness) vs time ----------------------
    if sim_loaded:
        print("  Extracting diffuse eta-band (eta=0.1/0.5/0.9 radii)...")
        _bt, _bands = extract_eta_band_history(OUTPUT_DIR, axis="x", x_max=1.5 * P.R0)
        plot_eta_band(_bt, _bands, curves)
        print("  Extracting volume-based radius (Sch20 eq.29: R=(3 V_gas/4pi)^1/3)...")
        _vt, _vr = extract_radius_volume_history(OUTPUT_DIR)
        plot_radius_volume(_vt, _vr, curves, t_sim, R_sim)

    out_png = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    out_eps = os.path.join(IMG_DIR, f"{SAVE_NAME}.eps")
    fig.savefig(out_png, dpi=DPI, bbox_inches='tight')
    fig.savefig(out_eps, dpi=DPI, bbox_inches='tight')
    print(f"\n  wrote {out_png}")
    print(f"  wrote {out_eps}")
    plt.close(fig)


if __name__ == "__main__":
    main()
