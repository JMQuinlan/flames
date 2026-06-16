"""
Standalone analytical script for 1D transient Stefan evaporation.

This file isolates the analytical model from `stefan_parse.py` so it can be run
without AMReX/yt data and still generate analytical-result plots.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.special import erfc, erfcx


# ============================================================================
# CONFIGURATION
# ============================================================================

output_folder = "./StefanProblem/output1"

# Physical parameters (match your case as needed)
density0 = 1.0      # gas density [kg/m^3]
density1 = 1000.0     # liquid density [kg/m^3]
D_v = 2.0e-5        # diffusion coefficient [m^2/s]
Y_surface = density0 / (density0 + density1)
Y_infinity = 0.0

# Domain/time sampling for analytical plotting
# Match your simulation input:
#   amr.plot_dt = 5e-2
#   stop_time   = 20.0
L_domain = 1.0
x_initial = 0.5
n_x = 500
plot_dt = 5.0e-2
stop_time = 20.0
include_t0 = True
num_profile_times = 6

# Plot style
FONT_SIZE_TITLE = 15
FONT_SIZE_LABEL = 13
FONT_SIZE_TICK = 11
LINE_WIDTH = 2.2


# ============================================================================
# ANALYTICAL MODEL
# ============================================================================

def solve_stefan_number(Y_s, Y_inf, rho_gas, rho_liq):
    """
    Solve for Stefan number lambda:
      lambda * exp(lambda^2) * erfc(lambda) =
      (rho_gas/rho_liq) * (Y_s - Y_inf) / (1 - Y_s)
    """
    A = (rho_gas / rho_liq) * (Y_s - Y_inf) / (1.0 - Y_s)

    def equation(lam):
        if lam <= 0:
            return 1.0e10
        return lam * erfcx(lam) - A

    lam0 = max(0.01, A)
    lam = float(fsolve(equation, lam0)[0])
    return lam


def transient_interface_position(t, Dv, lam):
    """s(t) = 2*lambda*sqrt(D*t)."""
    if t <= 0:
        return 0.0
    return 2.0 * lam * np.sqrt(Dv * t)


def transient_stefan_mass_fraction_profile(x, t, Dv, Y_s, Y_inf, lam, x0=0.0):
    """
    Y(x,t) = Y_inf + (Y_s - Y_inf) * erfc((x-s(t))/(2*sqrt(D*t))) / erfc(lambda)
    with Y = Y_s for x < s(t).
    """
    if t <= 1.0e-12:
        return np.where(x > x0, Y_inf, Y_s)

    sqrt_Dt = np.sqrt(Dv * t)
    s_t = x0 + 2.0 * lam * sqrt_Dt
    eta = (x - s_t) / (2.0 * sqrt_Dt)

    erfc_lambda = erfc(lam)
    if abs(erfc_lambda) < 1.0e-12:
        erfc_lambda = 1.0e-12

    Y = Y_inf + (Y_s - Y_inf) * erfc(eta) / erfc_lambda
    Y = np.where(x < s_t, Y_s, Y)
    Y = np.clip(Y, min(Y_s, Y_inf), max(Y_s, Y_inf))
    return Y


def transient_stefan_mass_flux(t, rho_liq, Dv, lam):
    """m_dot(t) = rho_liq * 2 * lambda * sqrt(D/(pi*t))."""
    if t <= 1.0e-12:
        return 0.0
    return rho_liq * 2.0 * lam * np.sqrt(Dv / (np.pi * t))


def transient_stefan_velocity(t, Dv, lam):
    """v_s(t) = lambda * sqrt(D/(pi*t))."""
    if t <= 1.0e-12:
        return 0.0
    return lam * np.sqrt(Dv / (np.pi * t))


# ============================================================================
# MAIN
# ============================================================================

def main():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print("=" * 70)
    print("STANDALONE ANALYTICAL: 1D TRANSIENT STEFAN EVAPORATION")
    print("=" * 70)

    lam = solve_stefan_number(Y_surface, Y_infinity, density0, density1)
    A = (density0 / density1) * (Y_surface - Y_infinity) / (1.0 - Y_surface)
    lhs = lam * erfcx(lam)

    x = np.linspace(0.0, L_domain, n_x)
    if include_t0:
        times = np.arange(0.0, stop_time + 0.5 * plot_dt, plot_dt)
    else:
        times = np.arange(plot_dt, stop_time + 0.5 * plot_dt, plot_dt)

    n_t = len(times)
    selected_indices = np.linspace(0, n_t - 1, num_profile_times, dtype=int)
    selected_times = times[selected_indices]

    print(f"Stefan number lambda = {lam:.8f}")
    print(f"Verification lhs=lambda*erfcx(lambda)={lhs:.8e}, A={A:.8e}")
    print(
        f"Time grid: t = {times[0]:.2f} to {times[-1]:.2f} s "
        f"(dt = {plot_dt:.2e}, n = {n_t})"
    )
    print(f"Output directory: {output_folder}")

    interface_positions = x_initial + np.array(
        [transient_interface_position(t, D_v, lam) for t in times]
    )
    interface_velocity = np.array([transient_stefan_velocity(t, D_v, lam) for t in times])
    mass_flux = np.array([transient_stefan_mass_flux(t, density1, D_v, lam) for t in times])

    # Plot 1: mass fraction profiles at selected times
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    for t in selected_times:
        Y = transient_stefan_mass_fraction_profile(
            x, t, D_v, Y_surface, Y_infinity, lam, x_initial
        )
        ax1.plot(x, Y, linewidth=LINE_WIDTH, label=f"t = {t:.3e} s")
    ax1.set_xlabel("Position x (m)", fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel("Mass fraction Y", fontsize=FONT_SIZE_LABEL)
    ax1.set_title("Analytical Mass Fraction Profiles", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=FONT_SIZE_TICK)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_folder, "01_analytical_mass_fraction_profiles.png"), dpi=300)
    plt.close(fig1)

    # Plot 2: interface position vs time
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    ax2.plot(times, interface_positions, "b-", linewidth=LINE_WIDTH)
    ax2.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
    ax2.set_ylabel("Interface position x_i(t) (m)", fontsize=FONT_SIZE_LABEL)
    ax2.set_title("Analytical Interface Position", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=FONT_SIZE_TICK)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_folder, "02_analytical_interface_position.png"), dpi=300)
    plt.close(fig2)

    # Plot 3: interface velocity vs time
    fig3, ax3 = plt.subplots(figsize=(9, 6))
    ax3.plot(times, interface_velocity, "r-", linewidth=LINE_WIDTH)
    ax3.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
    ax3.set_ylabel("Interface velocity (m/s)", fontsize=FONT_SIZE_LABEL)
    ax3.set_title("Analytical Interface Velocity", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=FONT_SIZE_TICK)
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_folder, "03_analytical_interface_velocity.png"), dpi=300)
    plt.close(fig3)

    # Plot 4: mass flux vs time
    fig4, ax4 = plt.subplots(figsize=(9, 6))
    ax4.plot(times, mass_flux, "k-", linewidth=LINE_WIDTH)
    ax4.set_xlabel("Time (s)", fontsize=FONT_SIZE_LABEL)
    ax4.set_ylabel("Mass flux (kg/m^2/s)", fontsize=FONT_SIZE_LABEL)
    ax4.set_title("Analytical Mass Flux", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=FONT_SIZE_TICK)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_folder, "04_analytical_mass_flux.png"), dpi=300)
    plt.close(fig4)

    # Plot 5: displacement vs sqrt(t)
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    sqrt_t = np.sqrt(times)
    displacement = interface_positions - x_initial
    ax5.plot(sqrt_t, displacement, "g-", linewidth=LINE_WIDTH)
    ax5.set_xlabel("sqrt(t) (s^0.5)", fontsize=FONT_SIZE_LABEL)
    ax5.set_ylabel("Interface displacement (m)", fontsize=FONT_SIZE_LABEL)
    ax5.set_title("Analytical sqrt(t) Scaling", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=FONT_SIZE_TICK)
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_folder, "05_analytical_sqrt_t_scaling.png"), dpi=300)
    plt.close(fig5)

    print("\nGenerated:")
    print("  - 01_analytical_mass_fraction_profiles.png")
    print("  - 02_analytical_interface_position.png")
    print("  - 03_analytical_interface_velocity.png")
    print("  - 04_analytical_mass_flux.png")
    print("  - 05_analytical_sqrt_t_scaling.png")
    print("\nRun with: python StefanProblem/stefan_analytical_standalone.py")


if __name__ == "__main__":
    main()
