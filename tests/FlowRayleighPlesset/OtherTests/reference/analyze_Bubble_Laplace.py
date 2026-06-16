"""
Laplace Equilibrium Test: analyze AMReX output of Bubble_UNIT_TEST2_Laplase.

A bubble initialized with the gas pressure exactly Laplace-balanced
    p_gas - p_liquid = sigma / R0   (2D cylindrical, single curvature)
should remain at R = R0 with zero velocity field. Drift, oscillation, or
collapse indicates a surface-tension / pressure-jump implementation bug.

Run from repo root:
    python tests/FlowRayleighPlesset/OtherTests/reference/analyze_Bubble_Laplace.py

Outputs:
    Images/Bubble_UNIT_TEST2_Laplase.png   (R(t)/R0, |u|_max, pressure jump)
    Console PASS/WARN/FAIL summary.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rayleigh_plesset_solver import extract_radius_history  # noqa: E402


# Must match input file Bubble_UNIT_TEST2_Laplase
SIGMA   = 7.28
R0      = 0.02
P_LIQ   = 500.0
P_GAS   = 864.0           # = P_LIQ + sigma / R0   (2D)
DELTA_P_THEORY = SIGMA / R0   # = 364

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset",
                 "output_Bubble_UNIT_TEST2_Laplase")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def extract_field_diagnostics(amrex_output_dir):
    """Return (times, u_max, p_in, p_out) per plot file.

    u_max  : domain-wide max(|velocity|), should stay ~ 0 in a Laplace test
    p_in   : pressure averaged over r < R0 / 2 (deep inside bubble)
    p_out  : pressure averaged over r > 4 * R0 (deep in liquid)
    """
    import yt
    yt.funcs.mylog.setLevel(40)

    pfs = sorted(
        os.path.join(amrex_output_dir, d)
        for d in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell")
    )

    times, u_max, p_in, p_out = [], [], [], []
    for pf in pfs:
        ds = yt.load(pf)
        ad = ds.all_data()

        # velocity magnitude — try common field names
        try:
            ux = np.asarray(ad["velocity_x"])
            uy = np.asarray(ad["velocity_y"])
        except Exception:
            mx = np.asarray(ad["momentum_x"])
            my = np.asarray(ad["momentum_y"])
            rho = np.asarray(ad["density"])
            ux = mx / np.maximum(rho, 1e-30)
            uy = my / np.maximum(rho, 1e-30)
        u_max.append(float(np.sqrt(ux * ux + uy * uy).max()))

        # pressure inside / outside the bubble by radius mask
        x = np.asarray(ad["x"])
        y = np.asarray(ad["y"])
        r = np.sqrt(x * x + y * y)
        try:
            p = np.asarray(ad["pressure"])
        except Exception:
            p = np.asarray(ad["press"])

        in_mask = r < (0.5 * R0)
        out_mask = r > (4.0 * R0)
        p_in.append(float(p[in_mask].mean()) if in_mask.any() else np.nan)
        p_out.append(float(p[out_mask].mean()) if out_mask.any() else np.nan)

        times.append(float(ds.current_time))

    return (np.array(times), np.array(u_max),
            np.array(p_in), np.array(p_out))


def main():
    print("=" * 70)
    print("Laplace Equilibrium Test")
    print("=" * 70)
    print(f"R0          = {R0}")
    print(f"sigma       = {SIGMA}")
    print(f"sigma / R0  = {DELTA_P_THEORY}   (expected pressure jump)")
    print(f"p_liquid    = {P_LIQ}")
    print(f"p_gas       = {P_GAS}")
    print(f"output dir  = {OUTPUT_DIR}")
    print()

    if not os.path.isdir(OUTPUT_DIR):
        print(f"[error] no AMReX output at {OUTPUT_DIR}")
        print("        run the simulation first:")
        print("          ./bin/hydro2-2d-g++ tests/FlowRayleighPlesset/Bubble_UNIT_TEST2_Laplase")
        return 2

    # Radius history (from eta = 0.5 crossing)
    try:
        t_R, R_x = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
        _,   R_y = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="y")
    except Exception as e:
        print(f"[error] could not extract radius history: {e}")
        return 2

    # Field diagnostics
    try:
        t_F, u_max, p_in, p_out = extract_field_diagnostics(OUTPUT_DIR)
    except Exception as e:
        print(f"[warn] could not extract field diagnostics: {e}")
        t_F = u_max = p_in = p_out = None

    # ----- metrics -----
    R_avg = 0.5 * (R_x + R_y)
    rel_err = (R_avg - R0) / R0
    aniso = (R_x - R_y) / R0     # x-vs-y asymmetry: nonzero -> non-radial mode

    max_abs_drift = float(np.nanmax(np.abs(rel_err)))
    final_drift   = float(rel_err[-1])
    max_aniso     = float(np.nanmax(np.abs(aniso)))

    print(f"frames      = {len(t_R)}")
    print(f"t_end       = {t_R[-1] * 1e3:.3f} ms")
    print(f"R/R0 final  = {R_avg[-1] / R0:.6f}")
    print(f"R/R0 min    = {R_avg.min() / R0:.6f}")
    print(f"R/R0 max    = {R_avg.max() / R0:.6f}")
    print(f"max |dR/R0| = {max_abs_drift * 100:.3f} %")
    print(f"max |Rx-Ry|/R0 = {max_aniso * 100:.3f} %")

    if u_max is not None:
        print(f"max |u|     = {u_max.max():.4f}   (should be ~0 in equilibrium)")
    if p_in is not None and not np.isnan(p_in[-1]):
        dp_meas = p_in[-1] - p_out[-1]
        dp_init = p_in[0]  - p_out[0]
        print(f"Δp initial  = {dp_init:.2f}     (theory σ/R0 = {DELTA_P_THEORY:.2f})")
        print(f"Δp final    = {dp_meas:.2f}")

    # Verdict
    print()
    if max_abs_drift < 0.01 and max_aniso < 0.01:
        verdict = "PASS  (Laplace equilibrium held within 1%)"
    elif max_abs_drift < 0.10:
        verdict = "WARN  (drift 1-10% — surface tension scaling likely off)"
    else:
        verdict = "FAIL  (drift > 10% — bubble is collapsing or expanding)"
    print(f"Verdict: {verdict}")
    print()

    # ----- plot -----
    n_panels = 3 if u_max is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3.0 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.6, label="equilibrium R = R0")
    ax.plot(t_R * 1e3, R_x / R0, "b-", lw=1.5, label="R(t)/R0  (x-axis ray)")
    ax.plot(t_R * 1e3, R_y / R0, "r--", lw=1.0, label="R(t)/R0  (y-axis ray)")
    ax.plot(t_R * 1e3, R_avg / R0, "k-", lw=2.0, alpha=0.5, label="R_avg")
    ax.set_ylabel("R / R0")
    ax.set_title(f"Laplace Equilibrium Test  ({verdict})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ax = axes[1]
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.3)
    ax.plot(t_R * 1e3, rel_err * 100, "b-",  lw=1.2, label="(R - R0)/R0")
    ax.plot(t_R * 1e3, aniso  * 100, "r--", lw=1.0, label="(Rx - Ry)/R0  (asymmetry)")
    ax.set_ylabel("relative error [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if u_max is not None:
        ax = axes[2]
        ax.semilogy(t_F * 1e3, np.maximum(u_max, 1e-12), "g-", lw=1.2,
                    label="max |u|")
        ax.set_ylabel("|u|_max")
        ax.set_xlabel("time [ms]")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)
    else:
        axes[-1].set_xlabel("time [ms]")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "Bubble_UNIT_TEST2_Laplase.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
