"""
Test 04 - Strong driven collapse (scaled units): compare AMReX R(t) to Chen 2D.

Run from repo root:
    python tests/FlowRayleighPlesset/OtherTests/reference/analyze_04_StrongCollapse.py
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


P = Params(
    rho_l=10.0,
    p_inf=12500.0,
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
T_END = 2.5e-2

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset", "OtherTests",
                 "output_RPE_04_StrongCollapse")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    tau_R = rayleigh_collapse_time(P)
    print(f"3D Rayleigh classical tau_R = {tau_R*1e3:.3f} ms")
    print(f"Pressure ratio p_inf/p_g0 = {P.p_inf/P.p_g0:.2f}")

    t_c2d, R_c2d, V_c2d = solve(P, T_END, model="chen2d")
    t_rp,  R_rp,  V_rp  = solve(P, T_END, model="rp")
    t_km,  R_km,  V_km  = solve(P, T_END, model="km")

    M_max_c2d = np.abs(V_c2d).max() / P.c_l
    M_max_km  = np.abs(V_km).max() / P.c_l
    print(f"Chen 2D peak wall Mach |R'|/c_l = {M_max_c2d:.3f}")
    print(f"3D KM   peak wall Mach          = {M_max_km:.3f}")
    print(f"Chen 2D min R/R0  = {R_c2d.min()/P.R0:.4f}")
    print(f"3D RP   min R/R0  = {R_rp.min()/P.R0:.4f}")
    print(f"3D KM   min R/R0  = {R_km.min()/P.R0:.4f}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(t_c2d * 1e3, R_c2d / P.R0, "b-",  lw=2.5,
             label=f"Chen 2D cylindrical (r_inf={P.r_inf*1e3:.0f} mm)")
    ax1.plot(t_rp  * 1e3, R_rp  / P.R0, "0.4", lw=1.0, ls="--",
             label="3D Rayleigh-Plesset (context)")
    ax1.plot(t_km  * 1e3, R_km  / P.R0, "0.6", lw=1.0, ls=":",
             label="3D Keller-Miksis (context)")

    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
            ax1.plot(t_sim * 1e3, R_sim / P.R0, "ro", ms=4, alpha=0.7,
                     label=f"AMReX 2D (n={len(t_sim)})")
        except Exception as e:
            print(f"[warn] could not read AMReX output: {e}")

    ax1.set_ylabel("R / R0")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(loc="best")
    ax1.set_title("Test 04 - Strong Collapse (p_inf/p_g0 = 25, scaled units)")

    ax2.plot(t_c2d * 1e3, V_c2d / P.c_l, "b-",  lw=2.5, label="Chen 2D wall Mach")
    ax2.plot(t_rp  * 1e3, V_rp  / P.c_l, "0.4", lw=1.0, ls="--", label="3D RP (context)")
    ax2.plot(t_km  * 1e3, V_km  / P.c_l, "0.6", lw=1.0, ls=":",  label="3D KM (context)")
    ax2.axhline(0.0, color="k", lw=0.4, alpha=0.5)
    ax2.axhline(0.1, color="r", lw=0.4, ls=":", alpha=0.5,
                label="M = 0.1 (RP loses validity)")
    ax2.set_xlabel("time [ms]")
    ax2.set_ylabel("R' / c_l")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "RPE_04_StrongCollapse.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
