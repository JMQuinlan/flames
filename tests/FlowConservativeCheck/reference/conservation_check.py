# -*- coding: utf-8 -*-
"""
===============================================================================
CONSERVATION CHECK  --  Garrick-Periodic test
===============================================================================

Companion to tests/FlowConservativeCheck/input_GarrickPeriodic.

For a closed system (all-periodic BCs) the volume integrals of the
CONSERVED PRIMARIES of the 6-eq model must stay constant to machine
precision:
    rho_eta0 + rho_eta1  (total mass; per-phase masses each conserved too)
    M_x, M_y             (mixture momentum components)
    energy_per_vol       (total energy rho*E)
    energy0 + energy1    (sum of per-phase internal energies; the SUM is
                          conserved; individual E_0 / E_1 may shuffle via
                          pressure relaxation, but their sum + KE = rho*E
                          must be preserved.)

Any drift in these integrals is a bug in the integrator.

USAGE:
    python tests/FlowConservativeCheck/reference/conservation_check.py

OUTPUTS:
    tests/FlowConservativeCheck/reference/Images/conservation_check.png
    Console summary with relative drift for each conserved quantity.
"""

import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import yt
    yt.funcs.mylog.setLevel(40)
except ImportError:
    print("ERROR: yt not installed.  pip install yt", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# Case selector.  Defaults to GarrickPeriodic; pass a different name as the
# first CLI arg, e.g. "python conservation_check.py Toro1aPeriodic".
CASE = sys.argv[1] if len(sys.argv) > 1 else "GarrickPeriodic"

# Alamo writes plotfiles relative to the binary's cwd (bin/).
OUTPUT_DIR = os.path.join(_REPO_ROOT, "bin", "tests", "FlowConservativeCheck",
                          f"output_{CASE}")
# OVERRIDE (uncomment + edit for your machine):
# OUTPUT_DIR = f"/mmfs1/home/ttryon/flames/bin/tests/FlowConservativeCheck/output_{CASE}"

OUT_PNG = os.path.join(_SCRIPT_DIR, "Images", f"conservation_check_{CASE}.png")
OUT_CSV = os.path.join(_SCRIPT_DIR, "Images", f"conservation_check_{CASE}.csv")


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------

def list_plotfiles(directory):
    """Alamo NNNNNcell + AMReX-default fallbacks."""
    if not os.path.isdir(directory):
        return []
    candidates = sorted(glob.glob(os.path.join(directory, "*cell"))
                      + glob.glob(os.path.join(directory, "*.plt*"))
                      + glob.glob(os.path.join(directory, "plt*")))
    return [p for p in candidates
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "Header"))]


def integrate_fields(plotfile_path):
    """Return dict of integrated conserved quantities at this plotfile's time.

    Uses a covering_grid at the finest available level so AMR (if ever
    re-enabled) is handled correctly: fine cells where they exist, coarse
    elsewhere, uniformly resampled.  For max_level=0 (this test's default)
    that's just the base grid.
    """
    ds = yt.load(plotfile_path)
    t = float(ds.current_time)

    dom_lo = ds.domain_left_edge.in_units("code_length").v
    dom_hi = ds.domain_right_edge.in_units("code_length").v
    n_finest = (ds.domain_dimensions * (ds.refine_by ** ds.index.max_level))

    cg = ds.covering_grid(
        level=ds.index.max_level,
        left_edge=dom_lo,
        dims=n_finest,
    )
    dx = float((dom_hi[0] - dom_lo[0]) / n_finest[0])
    dy = float((dom_hi[1] - dom_lo[1]) / n_finest[1])
    dV = dx * dy   # 2D: "volume" is area

    def _sum(field):
        return float(np.asarray(cg[field]).sum()) * dV

    out = dict(time=t)
    # Per-phase + total mass (conserved by RHS independently).
    out["mass0"] = _sum("rho_eta0")
    out["mass1"] = _sum("rho_eta1")
    out["mass_total"] = out["mass0"] + out["mass1"]
    # Momentum (mixture).  Alamo plotfile field names concatenate basename
    # and suffix without an underscore: "momentum" + "x" -> "momentumx".
    out["mom_x"] = _sum("momentumx")
    out["mom_y"] = _sum("momentumy")
    # Total energy (redundant primary; should be conserved).
    out["energy_total"] = _sum("energy_per_vol")
    # Per-phase internal energies (SUM, not individual, is meaningful).
    try:
        out["energy0"] = _sum("energy0")
        out["energy1"] = _sum("energy1")
        out["energy_phase_sum"] = out["energy0"] + out["energy1"]
    except Exception:
        # If E_0 / E_1 aren't in the plotfile (writeout=false), skip.
        out["energy0"] = float("nan")
        out["energy1"] = float("nan")
        out["energy_phase_sum"] = float("nan")
    # ---------------------------------------------------------------------
    # KE -- the 6-eq state DECOUPLES kinetic from internal energy:
    #   rho*E  =  E_0  +  E_1  +  KE        where KE = (1/2) |M|^2 / rho
    # So E_0 + E_1 ALONE is NOT the conserved quantity -- KE has to be
    # included to compare against rho*E.  Integrating cell-by-cell:
    #   KE_int = sum_cells [ (1/2) (M_x^2 + M_y^2) / (rho_eta0 + rho_eta1) ] * dV
    # ---------------------------------------------------------------------
    mom_x_arr = np.asarray(cg["momentumx"])
    mom_y_arr = np.asarray(cg["momentumy"])
    # Use the conservative density sum (avoids any density_mf staleness).
    rho_arr   = np.asarray(cg["rho_eta0"]) + np.asarray(cg["rho_eta1"])
    rho_safe  = np.maximum(rho_arr, 1.0e-30)
    ke_cell   = 0.5 * (mom_x_arr * mom_x_arr + mom_y_arr * mom_y_arr) / rho_safe
    out["KE"] = float(ke_cell.sum()) * dV
    # Internal+KE total derived from per-phase form: should equal energy_total
    # under a CORRECT conservative integrator + reinit.  Direct apples-to-
    # apples check.
    if np.isfinite(out["energy_phase_sum"]):
        out["energy_full"] = out["energy_phase_sum"] + out["KE"]
        # Mismatch with the conservative ρE; pure diagnostic of the per-phase
        # reconciliation quality.
        out["energy_full_minus_total"] = out["energy_full"] - out["energy_total"]
    else:
        out["energy_full"] = float("nan")
        out["energy_full_minus_total"] = float("nan")
    # Eta volume integral (NOT a conservation law per se -- transported with
    # CH and advection -- but for periodic + no sources should also be conserved).
    out["eta_int"] = _sum("eta")
    return out


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print(f"CONSERVATION CHECK -- {CASE}")
    print("=" * 70)
    print(f"  output dir = {OUTPUT_DIR}")

    plots = list_plotfiles(OUTPUT_DIR)
    if not plots:
        print(f"\n  No plotfiles found in {OUTPUT_DIR}")
        print(f"  Run input_GarrickPeriodic first.")
        return
    print(f"  found {len(plots)} plotfiles")

    rows = []
    for i, p in enumerate(plots):
        try:
            row = integrate_fields(p)
        except Exception as e:
            print(f"  [warn] skipping {os.path.basename(p)}: {e}")
            continue
        rows.append(row)
        if i == 0 or (i + 1) % 10 == 0 or i == len(plots) - 1:
            print(f"    [{i+1}/{len(plots)}] t = {row['time']:.4e} s")

    if not rows:
        print("\n  No integrable plotfiles.")
        return

    # ---- Stack into arrays --------------------------------------------------
    keys = ["mass0", "mass1", "mass_total",
            "mom_x", "mom_y",
            "energy_total", "energy0", "energy1", "energy_phase_sum",
            "KE", "energy_full", "energy_full_minus_total",
            "eta_int"]
    t = np.array([r["time"] for r in rows])
    data = {k: np.array([r[k] for r in rows]) for k in keys}

    # ---- Print summary ------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONSERVATION DRIFT (final vs initial)")
    print("=" * 70)
    print(f"  {'quantity':<22s} {'initial':>16s} {'final':>16s} "
          f"{'abs drift':>14s} {'rel drift':>12s}")
    for k in keys:
        q = data[k]
        if not np.all(np.isfinite(q)):
            continue
        q0 = q[0]
        qN = q[-1]
        d_abs = qN - q0
        d_rel = (d_abs / q0 * 100) if abs(q0) > 1e-300 else float("nan")
        print(f"  {k:<22s} {q0:16.8e} {qN:16.8e} "
              f"{d_abs:+14.4e} {d_rel:+11.4e}%")

    # ---- CSV ----------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("time_s," + ",".join(keys) + "\n")
        for i, ti in enumerate(t):
            vals = ",".join(f"{data[k][i]:.9e}" for k in keys)
            f.write(f"{ti:.9e},{vals}\n")
    print(f"\n  wrote {OUT_CSV}")

    # ---- Plot drift vs time -------------------------------------------------
    # Three panels: mass, momentum, energy.  Each shows (q(t) - q(0)) / |q(0)|
    # so a perfect integrator gives a flat zero line.
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    def _drift(q):
        q0 = q[0]
        if abs(q0) < 1e-300:
            return q - q0          # avoid divide-by-zero; absolute drift
        return (q - q0) / abs(q0) * 100

    # MASS
    ax = axes[0]
    ax.plot(t, _drift(data["mass0"]),     "b-",  label="phase 0 mass")
    ax.plot(t, _drift(data["mass1"]),     "r-",  label="phase 1 mass")
    ax.plot(t, _drift(data["mass_total"]), "k--", label="total mass", lw=2)
    ax.set_ylabel("mass drift [%]")
    ax.set_title("Conservation check -- Garrick periodic")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # MOMENTUM (mom_x and mom_y -- IC starts at 0, so plot ABSOLUTE drift)
    ax = axes[1]
    ax.plot(t, data["mom_x"], "b-", label="x-momentum")
    ax.plot(t, data["mom_y"], "r-", label="y-momentum")
    ax.set_ylabel("momentum [abs]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # ENERGY -- correctly accounting for kinetic via the 6-eq decoupling:
    #   rho*E = E_0 + E_1 + KE
    # so "energy_full" = E_0 + E_1 + KE should OVERLAY rho*E if reinit is
    # exact-conservative.  The dashed orange line (E_0 + E_1 alone) is
    # NOT a conservation diagnostic on its own -- shown faintly for
    # context only, to make the KE shift visible.
    ax = axes[2]
    ax.plot(t, _drift(data["energy_total"]), "k-",
            label=r"$\rho E$ (mixture, conserved)", lw=2)
    if np.all(np.isfinite(data["energy_full"])):
        ax.plot(t, _drift(data["energy_full"]), "g--",
                label=r"$E_0 + E_1 + KE$ (should overlay $\rho E$)", lw=2)
    if np.all(np.isfinite(data["energy_phase_sum"])):
        ax.plot(t, _drift(data["energy_phase_sum"]), color="0.6", ls=":",
                label=r"$E_0 + E_1$ (no KE) -- KE shows here", lw=1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("energy drift [%]")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"  wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
