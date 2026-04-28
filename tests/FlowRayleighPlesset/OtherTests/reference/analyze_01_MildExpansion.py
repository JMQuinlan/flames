"""
Test 01 - Mild Bubble Expansion: compare AMReX simulation R(t) against
Rayleigh-Plesset ODE.

Run from repo root:
    python tests/FlowRayleighPlesset/OtherTests/reference/analyze_01_MildExpansion.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rayleigh_plesset_solver import Params, solve, extract_radius_history


# -- Test parameters (must match input_RPE_01_MildExpansion exactly) ---------
# r_inf is the far-field radius for the Chen 2D logarithmic factor:
# set to half the simulation domain (5 mm = 5 * R0).
P = Params(
    rho_l=1000.0,
    p_inf=1.01325e5,
    mu_l=0.0,
    c_l=1500.0,
    gamma_gas=1.4,
    p_g0=1.5e5,        # ~1.5 atm: Chen 2D R_max ~= 1.37 mm (well inside 5mm box)
    p_v=0.0,
    sigma=0.0,
    R0=1.0e-3,
    Rdot0=0.0,
    r_inf=5.0e-3,
)
T_END = 6.0e-4   # 600 us covers full Chen 2D oscillation cycle

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset", "OtherTests",
                 "output_RPE_01_MildExpansion")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    # -- Analytical references -----------------------------------------------
    t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")    # primary reference
    t_rp,  R_rp,  _ = solve(P, T_END, model="rp")        # 3D for context
    t_km,  R_km,  _ = solve(P, T_END, model="km")        # 3D w/ compressibility

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.plot(t_c2d * 1e6, R_c2d * 1e3, "b-",  lw=2.5,
            label=f"Chen 2D cylindrical (r_inf={P.r_inf*1e3:.1f} mm)")
    ax.plot(t_rp  * 1e6, R_rp  * 1e3, "0.4", lw=1.0, ls="--",
            label="3D Rayleigh-Plesset (context)")
    ax.plot(t_km  * 1e6, R_km  * 1e3, "0.6", lw=1.0, ls=":",
            label="3D Keller-Miksis (context)")
    ax.axhline(P.R0 * 1e3, color="k", lw=0.5, ls=":", alpha=0.6, label="R0")

    # -- AMReX simulation result --------------------------------------------
    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
            ax.plot(t_sim * 1e6, R_sim * 1e3, "ro", ms=4, alpha=0.7,
                    label=f"AMReX 2D Cartesian (n={len(t_sim)})")
        except Exception as e:
            print(f"[warn] could not read AMReX output: {e}")
    else:
        print(f"[info] no AMReX output yet at {OUTPUT_DIR} - showing reference only")

    ax.set_xlabel("time [us]")
    ax.set_ylabel("R [mm]")
    ax.set_title("Test 01 - Mild Expansion (p_g0/p_inf = 4.94, no surface tension)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    out = os.path.join(IMG_DIR, "RPE_01_MildExpansion.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)

    # -- Summary -------------------------------------------------------------
    print()
    print(f"Chen 2D reference (PRIMARY for 2D Cartesian sim):")
    print(f"  R_max     = {R_c2d.max() * 1e3:.4f} mm   (= {R_c2d.max()/P.R0:.3f} * R0)")
    print(f"  t at peak = {t_c2d[R_c2d.argmax()] * 1e6:.2f} us")
    print(f"3D Rayleigh-Plesset (context):")
    print(f"  R_max     = {R_rp.max() * 1e3:.4f} mm   (= {R_rp.max()/P.R0:.3f} * R0)")
    print(f"  t at peak = {t_rp[R_rp.argmax()] * 1e6:.2f} us")


if __name__ == "__main__":
    main()
