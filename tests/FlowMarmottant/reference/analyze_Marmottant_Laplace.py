# -*- coding: utf-8 -*-
"""
===============================================================================
MARMOTTANT LAPLACE-EQUILIBRIUM / CURVATURE-RECONSTRUCTION ANALYSIS
===============================================================================
Validates input_Marmottant_Laplace_Static. Three checks (see bin/Marmottant.md
step 2-3):

  1. RADIUS HOLD     : R(t) stays at R0 (no drift)  -> Laplace balance w/ sigma(R).
  2. SIGMA_EFF       : the code's sigma_eff field on the interface ~ sigma_0
                       -> sigma(R_eff) evaluation is correct.
  3. R_eff RECONSTRUCT: invert the measured sigma_eff (or read the kappa field)
                       -> R_eff ~ R0  -> curvature->radius mapping is correct.

DEBUG CSVs (written to reference/Data/):
  Marmottant_Laplace_timeseries.csv -- per-frame error metrics.
  Marmottant_Laplace_profile.csv    -- radial profile (x, eta, sigma_eff, kappa,
                                       R_eff_local, sigma_model_local) for one
                                       frame; reveals how sigma_eff varies ACROSS
                                       the diffuse band (R_eff=1/kappa is not
                                       constant across a finite-thickness band).

Runnable BEFORE any simulation: with no plotfiles it just prints the analytic
operating point and draws the sigma(R) curve.

USAGE:
    python tests/FlowMarmottant/reference/analyze_Marmottant_Laplace.py
===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_model import (                                       # noqa: E402
    marmottant_sigma, r_rupture, sigma_rest, reff_from_kappa, laplace_jump,
)

# ============================  CONFIGURATION  ===============================
# MUST match input_Marmottant_Laplace_Static.
CHI         = 14.56
R_BUCK      = 0.018
SIGMA_BREAK = 7.28
R0          = 0.02
GEOM_FACTOR = 1.0
P_LIQUID    = 500.0

OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests",
    "FlowMarmottant", "output_Marmottant_Laplace_Static",
))
# OVERRIDE (uncomment + edit for your machine / scratch):
# OUTPUT_DIR = "/path/to/output_Marmottant_Laplace_Static"

ETA_THRESHOLD = 0.5
AXIS          = "x"
PROFILE_FRAME = -1          # which frame to dump the radial profile for (-1=last)
SAVE_NAME     = "Marmottant_Laplace"
DPI           = 170

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
    """Load one plotfile and return a dict with the +axis ray (sorted by coord):
    {t, x, eta, sigma_eff(or None), kappa(or None)}. Raises on a corrupt file."""
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
    """Interpolated eta=thr crossing radius and the first index at/above thr."""
    idx = np.where(eta >= thr)[0]
    if len(idx) == 0:
        return np.nan, None
    i = idx[0]
    if i == 0:
        return x[0], 0
    frac = (thr - eta[i - 1]) / (eta[i] - eta[i - 1])
    return x[i - 1] + frac * (x[i] - x[i - 1]), i


def extract_frames(amrex_dir, thr=0.5, axis="x"):
    """Return (frames, n_skipped). frames: list of per-frame metric dicts sorted
    by time. Never raises -- corrupt frames are skipped."""
    frames, n_skipped = [], 0
    for pf in list_plotfiles(amrex_dir):
        try:
            r = load_ray(pf, axis)
        except Exception:
            n_skipped += 1
            continue
        R, i = crossing(r["x"], r["eta"], thr)
        sig = r["sigma_eff"]
        kap = r["kappa"]
        sig_cross = float(sig[i]) if (sig is not None and i is not None) else np.nan
        sig_peak = float(np.nanmax(sig)) if sig is not None else np.nan
        kap_cross = float(kap[i]) if (kap is not None and i is not None) else np.nan
        frames.append(dict(t=r["t"], R=R, sig_cross=sig_cross,
                           sig_peak=sig_peak, kap_cross=kap_cross))
    frames.sort(key=lambda f: f["t"])
    return frames, n_skipped


# ============================================================================
# CSV WRITERS
# ============================================================================
def write_timeseries_csv(frames, sigma0):
    """Per-frame error metrics. The headline column is sigma_err_vs_model_pct:
    the code's sigma_eff at the crossing vs the model sigma evaluated at the
    INDEPENDENTLY MEASURED radius R -- i.e. is sigma(R_eff) reconstruction right?"""
    path = os.path.join(DATA_DIR, f"{SAVE_NAME}_timeseries.csv")
    cols = ["time_s", "time_ms", "R", "R_over_R0", "drift_pct",
            "sigma_eff_cross", "sigma_eff_peak", "sigma_model_at_R",
            "sigma_eff_minus_model", "sigma_err_vs_model_pct",
            "kappa_cross", "R_eff_from_kappa", "Reff_kappa_err_pct",
            "R_eff_from_sigma_inv", "Reff_sig_err_pct"]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for fr in frames:
            R = fr["R"]
            sig_model = marmottant_sigma(R, CHI, R_BUCK, SIGMA_BREAK) \
                if np.isfinite(R) else np.nan
            with np.errstate(divide="ignore", invalid="ignore"):
                drift = 100.0 * (R - R0) / R0
                sig_err = 100.0 * (fr["sig_cross"] - sig_model) / sig_model
                reff_k = (reff_from_kappa(fr["kap_cross"], GEOM_FACTOR)
                          if np.isfinite(fr["kap_cross"]) and fr["kap_cross"] != 0
                          else np.nan)
                reff_k_err = 100.0 * (reff_k - R0) / R0
                # elastic-branch inversion of sigma_eff -> R_eff (valid only mid-ramp)
                reff_s = R_BUCK * np.sqrt(max(fr["sig_cross"], 0.0) / CHI + 1.0) \
                    if np.isfinite(fr["sig_cross"]) else np.nan
                reff_s_err = 100.0 * (reff_s - R0) / R0
            row = [fr["t"], fr["t"] * 1e3, R, R / R0, drift,
                   fr["sig_cross"], fr["sig_peak"], sig_model,
                   fr["sig_cross"] - sig_model, sig_err,
                   fr["kap_cross"], reff_k, reff_k_err, reff_s, reff_s_err]
            f.write(",".join(f"{v:.9e}" if isinstance(v, float) else str(v)
                             for v in row) + "\n")
    print(f"  wrote {path}")
    return path


def write_profile_csv(amrex_dir, frame_index, thr=0.5, axis="x"):
    """Radial profile of one frame: x, eta, sigma_eff, kappa, and the LOCAL
    reconstruction R_eff=geom/|kappa| with sigma_model(R_eff_local). Shows how
    sigma_eff varies across the diffuse band (it should not be a single value)."""
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
    Rc, _ = crossing(x, eta, thr)
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
    jump = laplace_jump(R0, sigma0, GEOM_FACTOR)

    print("=" * 70)
    print("MARMOTTANT LAPLACE / CURVATURE-RECONSTRUCTION ANALYSIS")
    print("=" * 70)
    print(f"  chi          = {CHI}")
    print(f"  R_buckling   = {R_BUCK}")
    print(f"  sigma_break  = {SIGMA_BREAK}")
    print(f"  R0           = {R0}   (regime: "
          f"{'elastic' if R_BUCK < R0 < R_rupt else 'OUT OF ELASTIC RAMP!'})")
    print(f"  R_rupture    = {R_rupt:.6f}")
    print(f"  sigma_0(R0)  = {sigma0:.6f}")
    print(f"  Laplace jump = sigma_0 * geom/R0 = {jump:.4f}")
    print(f"  p_gas balance= {P_LIQUID} + jump = {P_LIQUID + jump:.4f}")
    print(f"  output dir   = {OUTPUT_DIR}")
    print()

    frames, n_skip = extract_frames(OUTPUT_DIR, ETA_THRESHOLD, AXIS)
    loaded = len(frames) > 0
    if n_skip:
        print(f"  [warn] skipped {n_skip} corrupt/partial plotfile(s)")
    if not loaded:
        print(f"  [info] no usable plotfiles at {OUTPUT_DIR}")
        print(f"         -> drawing analytic sigma(R) only; run the sim first.\n")

    t = np.array([fr["t"] for fr in frames]) if loaded else np.array([])
    R = np.array([fr["R"] for fr in frames]) if loaded else np.array([])
    sig_cross = np.array([fr["sig_cross"] for fr in frames]) if loaded else np.array([])

    if loaded:
        drift = (R - R0) / R0
        print("  CHECK 1 -- radius hold:")
        print(f"    R(0)            = {R[0]:.6e}  (target R0 = {R0})")
        print(f"    max |drift|     = {np.nanmax(np.abs(drift))*100:.4f} %")
        print(f"    final R         = {R[-1]:.6e}  ({(R[-1]-R0)/R0*100:+.4f} %)")
        print("  CHECK 2 -- sigma_eff(cross) vs model sigma at measured R:")
        if np.any(np.isfinite(sig_cross)):
            sm0 = marmottant_sigma(R[0], CHI, R_BUCK, SIGMA_BREAK)
            print(f"    sigma_eff(t0)   = {sig_cross[0]:.6f}  "
                  f"(sigma_0 = {sigma0:.6f}, model@R(0) = {sm0:.6f})")
            with np.errstate(invalid="ignore"):
                print(f"    err vs model@R  = "
                      f"{(sig_cross[0]-sm0)/sm0*100:+.3f} %  "
                      f"(<- key reconstruction error; see CSV column)")
        else:
            print("    sigma_eff field not found -- rebuild with the diagnostic.")
        print()

        # ---- DEBUG CSVs ---------------------------------------------------
        write_timeseries_csv(frames, sigma0)
        write_profile_csv(OUTPUT_DIR, PROFILE_FRAME, ETA_THRESHOLD, AXIS)
        print()

    # ---- plot --------------------------------------------------------------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    if loaded:
        axL.plot(t * 1e3, R / R0, "o-", color="tab:red", ms=4,
                 label=f"sim R(t)/R0 (n={len(t)})")
    axL.axhline(1.0, color="k", ls=":", lw=0.8, label="R0")
    axL.axhline(R_BUCK / R0, color="0.5", ls="--", lw=0.8, label="R_buck/R0")
    axL.axhline(R_rupt / R0, color="0.5", ls=":", lw=0.8, label="R_rupt/R0")
    axL.set_xlabel("t [ms]"); axL.set_ylabel("R / R0")
    axL.set_title("Radius hold (should be flat at 1.0)")
    axL.grid(True, alpha=0.3); axL.legend(fontsize=9)

    Rg = np.linspace(0.5 * R_BUCK, 1.4 * R_rupt, 500)
    axR.plot(Rg / R0, marmottant_sigma(Rg, CHI, R_BUCK, SIGMA_BREAK),
             "b-", lw=2, label=r"$\sigma(R_{\rm eff})$ model")
    axR.axvline(R_BUCK / R0, color="0.5", ls="--", lw=0.8, label="R_buck")
    axR.axvline(R_rupt / R0, color="0.5", ls=":", lw=0.8, label="R_rupt")
    axR.plot([1.0], [sigma0], "ko", ms=8, label=r"rest $(R_0,\sigma_0)$")
    if loaded and np.any(np.isfinite(sig_cross)):
        axR.plot([R[0] / R0], [sig_cross[0]], "r*", ms=14,
                 label=r"sim $\sigma_{\rm eff}$ at $R(0)$")
    axR.set_xlabel("R / R0"); axR.set_ylabel(r"$\sigma$ (scaled)")
    axR.set_title("Marmottant curve + operating point")
    axR.grid(True, alpha=0.3); axR.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
