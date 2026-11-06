# -*- coding: utf-8 -*-
"""
===============================================================================
MARMOTTANT OSCILLATING COATED-BUBBLE ANALYSIS
===============================================================================
Validates input_Marmottant_Oscillating (bin/Marmottant.md step 4). Extracts the
bubble radius history R(t), maps it onto the Marmottant sigma(R) curve, and
quantifies the "compression-only" asymmetry:

    compression  = (R0 - min R) / R0
    expansion    = (max R - R0) / R0
    asymmetry    = compression / expansion        ( > 1  => compression-dominated )

Overlays the coated Chen-2D RPE reference (same R^2 area law as the code).

DEBUG CSVs (reference/Data/):
  Marmottant_Oscillating_timeseries.csv -- per-frame R, regime, sigma_model(R),
        reference R interpolated to sim time, and sim-vs-ref difference.
  Marmottant_Oscillating_profile.csv    -- radial profile of one frame
        (x, eta, sigma_eff, kappa, R_eff_local, sigma_model_local).

Runnable BEFORE any simulation (draws reference ODE + sigma(R) curve).

USAGE:
    python tests/FlowMarmottant/reference/analyze_Marmottant_Oscillating.py
===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_model import (                                       # noqa: E402
    marmottant_sigma, r_rupture, sigma_rest, reff_from_kappa,
    MarmParams, solve_coated,
)

# ============================  CONFIGURATION  ===============================
# MUST match input_Marmottant_Oscillating.
CHI         = 14.56
R_BUCK      = 0.018
SIGMA_BREAK = 7.28
R0          = 0.02
GEOM_FACTOR = 1.0
P_LIQUID    = 500.0
P_GAS0      = 400.0       # gas IC drive (pressure1)
RHO_L       = 10.0
GAMMA_GAS   = 1.4
R_INF       = 0.20        # half-domain (2D log far field)
T_END       = 20.0e-3

OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowMarmottant", "output_Marmottant_Oscillating",
))
# OVERRIDE (uncomment + edit for your machine / scratch):
# OUTPUT_DIR = "/path/to/output_Marmottant_Oscillating"

ETA_THRESHOLD  = 0.5
AXIS           = "x"
PROFILE_FRAME  = -1        # frame index for the radial-profile CSV (-1=last)
SHOW_REFERENCE = True
SAVE_NAME      = "Marmottant_Oscillating"
DPI            = 170

IMG_DIR  = os.path.join(_HERE, "Images")
DATA_DIR = os.path.join(_HERE, "Data")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# PLOTFILE HELPERS (robust; never raise on a single bad frame)
# ============================================================================
def list_plotfiles(amrex_dir):
    if not os.path.isdir(amrex_dir):
        return []
    return sorted(
        os.path.join(amrex_dir, d) for d in os.listdir(amrex_dir)
        if os.path.isdir(os.path.join(amrex_dir, d)) and d.endswith("cell")
    )


def load_ray(pf, axis="x"):
    import yt
    yt.funcs.mylog.setLevel(40)
    ds = yt.load(pf)
    L = float(ds.domain_right_edge[0])
    p0 = ds.arr([0.0, 0.0, 0.0], "code_length")
    p1 = (ds.arr([L, 0.0, 0.0], "code_length") if axis == "x"
          else ds.arr([0.0, L, 0.0], "code_length"))
    ray = ds.ray(p0, p1)
    coord = np.array(ray["x" if axis == "x" else "y"])
    order = np.argsort(coord)
    out = {"t": float(ds.current_time), "x": coord[order],
           "eta": np.array(ray["eta"])[order]}
    try:
        out["sigma_eff"] = np.array(ray["sigma_eff"])[order]
    except Exception:
        out["sigma_eff"] = None
    out["kappa"] = None
    for name in ("kappa_Avg", "kappaAvg", "kappa", "kappa_0"):
        try:
            out["kappa"] = np.array(ray[name])[order]
            break
        except Exception:
            continue
    return out


def crossing(x, eta, thr):
    idx = np.where(eta >= thr)[0]
    if len(idx) == 0:
        return np.nan, None
    i = idx[0]
    if i == 0:
        return x[0], 0
    frac = (thr - eta[i - 1]) / (eta[i] - eta[i - 1])
    return x[i - 1] + frac * (x[i] - x[i - 1]), i


def extract_radius(amrex_dir, thr=0.5, axis="x"):
    """Return (times, radii, n_skipped) from eta=thr crossings."""
    times, radii, n_skipped = [], [], 0
    for pf in list_plotfiles(amrex_dir):
        try:
            r = load_ray(pf, axis)
        except Exception:
            n_skipped += 1
            continue
        R, _ = crossing(r["x"], r["eta"], thr)
        times.append(r["t"]); radii.append(R)
    if not times:
        return np.array([]), np.array([]), n_skipped
    o = np.argsort(times)
    return np.array(times)[o], np.array(radii)[o], n_skipped


def regime_of(R):
    if not np.isfinite(R):
        return "nan"
    if R <= R_BUCK:
        return "buckled"
    if R >= r_rupture(R_BUCK, CHI, SIGMA_BREAK):
        return "ruptured"
    return "elastic"


# ============================================================================
# CSV WRITERS
# ============================================================================
def write_timeseries_csv(t, R, t_ref, R_ref):
    """Per-frame R, regime, model sigma(R), and sim-vs-reference difference
    (reference interpolated onto sim times; NaN outside its span)."""
    path = os.path.join(DATA_DIR, f"{SAVE_NAME}_timeseries.csv")
    if len(t_ref) > 1:
        R_ref_at = np.interp(t, t_ref, R_ref, left=np.nan, right=np.nan)
    else:
        R_ref_at = np.full_like(t, np.nan)
    cols = ["time_s", "time_ms", "R_sim", "R_over_R0", "regime",
            "sigma_model_at_R", "R_ref", "R_ref_over_R0",
            "diff_m", "diff_rel_pct"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for ts, rs, rr in zip(t, R, R_ref_at):
            sm = marmottant_sigma(rs, CHI, R_BUCK, SIGMA_BREAK) \
                if np.isfinite(rs) else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                dm = rs - rr
                dp = 100.0 * dm / rr
            row = [f"{ts:.9e}", f"{ts*1e3:.6f}", f"{rs:.9e}",
                   f"{rs/R0:.6f}", regime_of(rs), f"{sm:.9e}",
                   f"{rr:.9e}", f"{rr/R0:.6f}", f"{dm:.9e}", f"{dp:.6f}"]
            f.write(",".join(row) + "\n")
    print(f"  wrote {path}")
    return path


def write_profile_csv(amrex_dir, frame_index, axis="x"):
    pfs = list_plotfiles(amrex_dir)
    if not pfs:
        return None
    try:
        r = load_ray(pfs[frame_index], axis)
    except Exception as e:
        print(f"  [warn] profile frame load failed: {e}")
        return None
    x, eta = r["x"], r["eta"]
    sig = r["sigma_eff"] if r["sigma_eff"] is not None else np.full_like(x, np.nan)
    kap = r["kappa"] if r["kappa"] is not None else np.full_like(x, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        reff_local = reff_from_kappa(kap, GEOM_FACTOR)
        sig_model_local = marmottant_sigma(reff_local, CHI, R_BUCK, SIGMA_BREAK)
    Rc, _ = crossing(x, eta, ETA_THRESHOLD)
    path = os.path.join(DATA_DIR, f"{SAVE_NAME}_profile.csv")
    with open(path, "w") as f:
        f.write(f"# frame t={r['t']:.9e} s   crossing R={Rc:.9e}   "
                f"R0={R0}   axis={axis}\n")
        f.write("x,eta,sigma_eff,kappa,R_eff_local,sigma_model_local\n")
        for vals in zip(x, eta, sig, kap, reff_local, sig_model_local):
            f.write(",".join(f"{v:.9e}" for v in vals) + "\n")
    print(f"  wrote {path}  (frame t={r['t']*1e3:.4f} ms)")
    return path


# ============================================================================
# MAIN
# ============================================================================
def main():
    R_rupt = r_rupture(R_BUCK, CHI, SIGMA_BREAK)
    sigma0 = sigma_rest(R0, CHI, R_BUCK, SIGMA_BREAK)

    print("=" * 70)
    print("MARMOTTANT OSCILLATING COATED-BUBBLE ANALYSIS")
    print("=" * 70)
    print(f"  chi={CHI}  R_buck={R_BUCK}  sigma_break={SIGMA_BREAK}  R0={R0}")
    print(f"  R_rupt={R_rupt:.6f}  sigma_0={sigma0:.6f}")
    print(f"  drive: p_gas0={P_GAS0} vs balanced "
          f"{P_LIQUID + sigma0/R0:.2f}  (deficit "
          f"{P_LIQUID + sigma0/R0 - P_GAS0:+.2f})")
    print(f"  output dir = {OUTPUT_DIR}\n")

    # ---- reference coated Chen-2D RPE -------------------------------------
    t_ref = R_ref = np.array([])
    if SHOW_REFERENCE:
        try:
            P = MarmParams(rho_l=RHO_L, p_inf=P_LIQUID, r_inf=R_INF,
                           gamma_gas=GAMMA_GAS, p_g0=P_GAS0, R0=R0,
                           chi=CHI, R_buck=R_BUCK, sigma_break=SIGMA_BREAK,
                           geom_factor=GEOM_FACTOR)
            t_ref, R_ref, _ = solve_coated(P, T_END)
            print("  coated Chen-2D RPE reference solved "
                  f"(min R/R0={np.min(R_ref)/R0:.3f}, "
                  f"max R/R0={np.max(R_ref)/R0:.3f})")
        except Exception as e:
            print(f"  [warn] reference ODE failed ({e}); skipping.")

    # ---- simulation -------------------------------------------------------
    t, R, n_skip = extract_radius(OUTPUT_DIR, ETA_THRESHOLD, AXIS)
    loaded = len(t) > 1
    if n_skip:
        print(f"  [warn] skipped {n_skip} corrupt/partial plotfile(s)")
    if not loaded:
        print(f"  [info] no usable plotfiles at {OUTPUT_DIR} "
              f"-> reference curves only.\n")

    if loaded:
        Rmin, Rmax = np.nanmin(R), np.nanmax(R)
        comp = (R0 - Rmin) / R0
        expa = (Rmax - R0) / R0
        asym = comp / expa if expa > 1e-12 else float("inf")
        print("\n  Compression-only metric (sim):")
        print(f"    min R/R0   = {Rmin/R0:.4f}   (compression {comp*100:.2f} %)")
        print(f"    max R/R0   = {Rmax/R0:.4f}   (expansion   {expa*100:.2f} %)")
        print(f"    asymmetry  = compression/expansion = {asym:.3f}"
              f"   ( >1 => compression-dominated )")
        print(f"    min R reaches buckling (R<R_buck)? "
              f"{'YES' if Rmin < R_BUCK else 'no'}")
        print(f"    max R reaches rupture (R>R_rupt)?  "
              f"{'YES' if Rmax > R_rupt else 'no'}")

        # ---- DEBUG CSVs ---------------------------------------------------
        print()
        write_timeseries_csv(t, R, t_ref, R_ref)
        write_profile_csv(OUTPUT_DIR, PROFILE_FRAME, AXIS)
        print()

    # ---- plot --------------------------------------------------------------
    fig, (axT, axS) = plt.subplots(1, 2, figsize=(13, 5))
    if len(t_ref):
        axT.plot(t_ref * 1e3, R_ref / R0, "-", color="tab:green", lw=1.8,
                 label="coated Chen-2D RPE")
    if loaded:
        axT.plot(t * 1e3, R / R0, "o-", color="tab:red", ms=3.5, alpha=0.8,
                 label=f"sim (n={len(t)})")
    axT.axhline(1.0, color="k", ls=":", lw=0.8, label="R0")
    axT.axhline(R_BUCK / R0, color="0.5", ls="--", lw=0.8, label="R_buck")
    axT.axhline(R_rupt / R0, color="0.5", ls=":", lw=0.8, label="R_rupt")
    axT.set_xlabel("t [ms]"); axT.set_ylabel("R / R0")
    axT.set_title("Oscillating coated bubble R(t)")
    axT.grid(True, alpha=0.3); axT.legend(fontsize=9)

    Rg = np.linspace(0.5 * R_BUCK, 1.4 * R_rupt, 500)
    axS.plot(Rg / R0, marmottant_sigma(Rg, CHI, R_BUCK, SIGMA_BREAK),
             "b-", lw=2, label=r"$\sigma(R)$ model")
    axS.axvline(R_BUCK / R0, color="0.5", ls="--", lw=0.8, label="R_buck")
    axS.axvline(R_rupt / R0, color="0.5", ls=":", lw=0.8, label="R_rupt")
    axS.plot([1.0], [sigma0], "ko", ms=7, label=r"rest")
    if loaded:
        Rclip = np.clip(R, Rg.min(), Rg.max())
        axS.plot(Rclip / R0, marmottant_sigma(Rclip, CHI, R_BUCK, SIGMA_BREAK),
                 ".", color="tab:red", ms=4, alpha=0.5,
                 label=r"sim trajectory $\sigma(R(t))$")
    axS.set_xlabel("R / R0"); axS.set_ylabel(r"$\sigma$ (scaled)")
    axS.set_title("Trajectory on the Marmottant curve")
    axS.grid(True, alpha=0.3); axS.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"\n  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
