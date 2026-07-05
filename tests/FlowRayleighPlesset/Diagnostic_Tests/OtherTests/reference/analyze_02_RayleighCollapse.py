"""
Test 02 - Driven Rayleigh Collapse (scaled units): compare AMReX simulation
R(t) against the Chen 2D cylindrical RPE ODE.

Run from repo root:
    python tests/FlowRayleighPlesset/OtherTests/reference/analyze_02_RayleighCollapse.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rayleigh_plesset_solver import (
    Params, solve, extract_radius_history, rayleigh_collapse_time,
)


# Must match input_RPE_02_RayleighCollapse exactly.
P = Params(
    rho_l=10.0,
    p_inf=5000.0,
    mu_l=0.0,
    c_l=63.0,
    gamma_gas=1.4,
    p_g0=500.0,
    p_v=0.0,
    sigma=7.28,
    R0=0.02,
    Rdot0=0.0,
    r_inf=0.10,
)
T_END = 1.5e-2     # 15 ms covers first collapse + rebound (matches input)

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset", "OtherTests",
                 "output_RPE_02_RayleighCollapse")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    tau_R = rayleigh_collapse_time(P)
    print(f"3D Rayleigh classical tau_R = {tau_R*1e3:.3f} ms (3D - 2D will be slower)")

    t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")
    t_rp,  R_rp,  _ = solve(P, T_END, model="rp")
    t_km,  R_km,  _ = solve(P, T_END, model="km")

    print(f"Chen 2D: R_min = {R_c2d.min()/P.R0:.4f} R0  at t = {t_c2d[R_c2d.argmin()]*1e3:.2f} ms")
    print(f"3D RP:   R_min = {R_rp.min()/P.R0:.4f} R0   at t = {t_rp[R_rp.argmin()]*1e3:.2f} ms")

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axvline(tau_R * 1e3, color="k", lw=0.5, ls=":", alpha=0.5,
               label=f"3D tau_R = {tau_R*1e3:.1f} ms")
    ax.plot(t_c2d * 1e3, R_c2d / P.R0, "b-",  lw=2.5,
            label=f"Chen 2D cylindrical (r_inf={P.r_inf*1e3:.0f} mm)")
    ax.plot(t_rp  * 1e3, R_rp  / P.R0, "0.4", lw=1.0, ls="--",
            label="3D Rayleigh-Plesset (context)")
    ax.plot(t_km  * 1e3, R_km  / P.R0, "0.6", lw=1.0, ls=":",
            label="3D Keller-Miksis (context)")

    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
            ax.plot(t_sim * 1e3, R_sim / P.R0, "ro", ms=4, alpha=0.7,
                    label=f"AMReX 2D Cartesian (n={len(t_sim)})")
        except Exception as e:
            print(f"[warn] could not read AMReX output: {e}")
    else:
        print(f"[info] no AMReX output yet at {OUTPUT_DIR}")

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("R / R0")
    ax.set_title("Test 02 - Driven Rayleigh Collapse (p_inf/p_g0 = 10, scaled units)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")
    plt.tight_layout()

    out = os.path.join(IMG_DIR, "RPE_02_RayleighCollapse.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
