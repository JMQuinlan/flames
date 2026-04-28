"""
Test 03 - Small-Amplitude Oscillation (scaled units): compare AMReX R(t)
against Chen 2D linearized solution and full ODE.

Run from repo root:
    python tests/FlowRayleighPlesset/OtherTests/reference/analyze_03_LinearOscillation.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from rayleigh_plesset_solver import (
    Params, solve, extract_radius_history, linear_natural_frequency_chen2d,
)


# Match input_RPE_03_LinearOscillation. Equilibrium at R_eq=0.019, perturbed
# to R0=0.020 with under-pressured gas (p_g0 from polytropic back-mapping).
R_EQ = 0.019
P = Params(
    rho_l=10.0,
    p_inf=500.0,
    mu_l=0.0,
    c_l=63.0,
    gamma_gas=1.4,
    p_g0=765.0,        # = (p_inf + sigma/R_eq) * (R_eq/R0)^(2*gamma)
    p_v=0.0,
    sigma=7.28,
    R0=0.020,
    Rdot0=0.0,
    r_inf=0.10,
)
T_END = 3.0e-2     # 30 ms ~ 3 periods

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset", "OtherTests",
                 "output_RPE_03_LinearOscillation")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    omega0 = linear_natural_frequency_chen2d(P, R_eq=R_EQ)
    T_period = 2.0 * np.pi / omega0
    print(f"Chen 2D linearized frequency: omega_0 = {omega0:.2f} rad/s")
    print(f"Period: T = {T_period*1e3:.2f} ms")

    t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")
    t_rp,  R_rp,  _ = solve(P, T_END, model="rp")

    R_lin = R_EQ + (P.R0 - R_EQ) * np.cos(omega0 * t_c2d)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axhline(R_EQ * 1e3, color="k", lw=0.5, ls=":", alpha=0.6,
               label=f"R_eq = {R_EQ*1e3:.1f} mm")
    ax.plot(t_c2d * 1e3, R_lin  * 1e3, "g--", lw=1.5,
            label=f"Chen 2D linearized (T={T_period*1e3:.1f} ms)")
    ax.plot(t_c2d * 1e3, R_c2d * 1e3, "b-",  lw=2.5,
            label="Chen 2D full ODE")
    ax.plot(t_rp  * 1e3, R_rp  * 1e3, "0.4", lw=1.0, ls=":",
            label="3D Rayleigh-Plesset (context)")

    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
            ax.plot(t_sim * 1e3, R_sim * 1e3, "ro", ms=4, alpha=0.7,
                    label=f"AMReX 2D Cartesian (n={len(t_sim)})")
        except Exception as e:
            print(f"[warn] could not read AMReX output: {e}")

    ax.set_xlabel("time [ms]")
    ax.set_ylabel("R [mm]")
    ax.set_title("Test 03 - Small-Amplitude Oscillation (5.3% perturbation, sigma=7.28)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    out = os.path.join(IMG_DIR, "RPE_03_LinearOscillation.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
