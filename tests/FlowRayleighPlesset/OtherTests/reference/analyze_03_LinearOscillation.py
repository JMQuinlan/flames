"""
Test 03 - Small-Amplitude Oscillation: compare AMReX R(t) against linearized
RP solution and check that the oscillation period matches the Minnaert /
linearized natural frequency.

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


# Note: the input file uses R0 = 1mm as the *initial perturbed* radius.
# Chen 2D Laplace equilibrium is single-curvature: p_g_eq = p_inf + sigma/R_eq.
# With R_eq = 0.95 mm: p_g_eq = 1.01325e5 + 0.0728/0.95e-3 = 1.0140e5 Pa.
# The input file IC (input_RPE_03_LinearOscillation) sets gas pressure
# accordingly so the bubble is near 2D-Laplace equilibrium.
P = Params(
    rho_l=1000.0,
    p_inf=1.01325e5,
    mu_l=0.0,
    c_l=1500.0,
    gamma_gas=1.4,
    # Chen 2D polytropic relation: p_g(R) = p_g0 * (R0/R)^(2*gamma).
    # Choose p_g0 so that p_g(R_eq) = p_inf + sigma/R_eq at R_eq = 0.95 mm.
    #   p_g0 = (p_inf + sigma/R_eq) * (R_eq/R0)^(2 gamma)
    #        = 101401.6 * (0.95/1.0)^2.8
    #        ~= 87612 Pa
    p_g0=87612.0,
    p_v=0.0,
    sigma=0.0728,
    R0=1.0e-3,        # initial (over-expanded) radius; equilibrium is at 0.95 mm
    Rdot0=0.0,
    r_inf=5.0e-3,
)
T_END = 1.5e-3   # 1.5 ms - 2D oscillation is much slower than 3D
                 # (logarithmic factor inflates inertia by ~1.6x)

OUTPUT_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                 "FlowRayleighPlesset", "OtherTests",
                 "output_RPE_03_LinearOscillation")
)
IMG_DIR = os.path.join(_HERE, "Images")
os.makedirs(IMG_DIR, exist_ok=True)


def main():
    R_eq = 0.95e-3
    omega0 = linear_natural_frequency_chen2d(P, R_eq=R_eq)
    T_period = 2.0 * np.pi / omega0
    print(f"Chen 2D linearized frequency: omega_0 = {omega0:.2f} rad/s")
    print(f"Period: T = {T_period*1e6:.2f} us")

    t_c2d, R_c2d, _ = solve(P, T_END, model="chen2d")
    t_rp,  R_rp,  _ = solve(P, T_END, model="rp")

    # Linear approximation: R(t) = R_eq + (R0 - R_eq) cos(omega_0 t)
    R_lin = R_eq + (P.R0 - R_eq) * np.cos(omega0 * t_c2d)

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    ax.axhline(R_eq * 1e3, color="k", lw=0.5, ls=":", alpha=0.6, label=f"R_eq = {R_eq*1e3:.3f} mm")
    ax.plot(t_c2d * 1e6, R_lin  * 1e3, "g--", lw=1.5,
            label=f"Chen 2D linearized (T={T_period*1e6:.0f} us)")
    ax.plot(t_c2d * 1e6, R_c2d * 1e3, "b-",  lw=2.5,
            label="Chen 2D full ODE")
    ax.plot(t_rp  * 1e6, R_rp  * 1e3, "0.4", lw=1.0, ls=":",
            label="3D Rayleigh-Plesset (context)")

    if os.path.isdir(OUTPUT_DIR):
        try:
            t_sim, R_sim = extract_radius_history(OUTPUT_DIR, eta_threshold=0.5, axis="x")
            ax.plot(t_sim * 1e6, R_sim * 1e3, "ro", ms=4, alpha=0.7,
                    label=f"AMReX 2D Cartesian (n={len(t_sim)})")
        except Exception as e:
            print(f"[warn] could not read AMReX output: {e}")

    ax.set_xlabel("time [us]")
    ax.set_ylabel("R [mm]")
    ax.set_title("Test 03 - Small-Amplitude Oscillation (5% perturbation, sigma=0.0728)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    out = os.path.join(IMG_DIR, "RPE_03_LinearOscillation.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
