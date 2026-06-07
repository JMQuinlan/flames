#!/usr/bin/env python3
"""
Spalding driving-force comparison: constant far-field Y_inf  vs  local gas-side
mass_frac_v as the sink concentration in B_M.

Question (Stage-3d Y_inf fork): if we replace the constant far-field Y_infinity
with the LOCAL gas-side vapor mass fraction, does the Spalding number collapse
because the cell next to the interface fills with the vapor we just produced
(before it can diffuse/advect away)?

Spalding mass-transfer number and the diffuse-interface driving group:
    B_M      = (Y_s - Y_sink) / (1 - Y_s)
    driving  = B_M / (1 + B_M) = (Y_s - Y_sink) / (1 - Y_sink)     # what m_dot scales with
    m_dot    = rho_g * Dv * driving * |grad eta|                    # code's source form
Y_s = Antoine saturation mass fraction at the interface T (same curve as the
HRM cavitation / Saturation_1D test). Y_sink = Y_infinity (constant) OR the
local gas-side mass_frac_v (self-limiting).
"""
import numpy as np

# --- constants from Saturation_1D ---
A, B, C = 4.10549, 1625.928, -92.839      # Antoine (n-dodecane), p_sat in bar
P_BAR   = 1.0                              # p = 1e5 Pa = 1 bar
CP_V, CV_V = 1994.85, 1950.0              # dodecane vapor
CP_G, CV_G = 1004.5, 717.5               # air carrier
R_V = CP_V - CV_V                          # 44.85  J/kg/K
R_G = CP_G - CV_G                          # 287.0  J/kg/K


def Ys(T):
    """Antoine saturation mass fraction at temperature T [K], p = P_BAR."""
    p_sat = 10.0 ** (A - B / (T + C))      # bar
    x_s = min(p_sat / P_BAR, 1.0)          # mole fraction (Raoult/Dalton)
    return (x_s / R_V) / (x_s / R_V + (1.0 - x_s) / R_G)   # mole -> mass


def driving(Y_s, Y_sink):
    """B_M/(1+B_M) = (Y_s - Y_sink)/(1 - Y_sink), clamped at 0."""
    return max((Y_s - Y_sink) / (1.0 - Y_sink), 0.0)


# ---------------------------------------------------------------------------
# 1) Y_s across the relevant interface-temperature range
# ---------------------------------------------------------------------------
print("=" * 70)
print("Saturation mass fraction Y_s(T)  (p = 1 bar, dodecane boiling ~489 K)")
print("=" * 70)
print(f"{'T [K]':>8} {'p_sat[bar]':>12} {'x_s (mol)':>12} {'Y_s (mass)':>12}")
for T in (300, 350, 380, 400, 420, 450, 480):
    p_sat = 10.0 ** (A - B / (T + C))
    x_s = p_sat / P_BAR
    print(f"{T:8.0f} {p_sat:12.4e} {x_s:12.4e} {Ys(T):12.4e}")

# ---------------------------------------------------------------------------
# 2) Driving force vs how saturated the LOCAL cell has become
# ---------------------------------------------------------------------------
T0 = 400.0
Y_s = Ys(T0)
d_const = driving(Y_s, 0.0)        # constant far-field Y_inf = 0
print()
print("=" * 70)
print(f"Driving force at interface T = {T0:.0f} K   (Y_s = {Y_s:.4f})")
print("  constant Y_inf=0:  driving = Y_s = %.4f  (FIXED, never self-limits)" % d_const)
print("=" * 70)
print(f"{'Y_local':>10} {'B_M_local':>12} {'driving_loc':>12} {'rate_loc/rate_const':>20}")
for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0):
    Y_local = frac * Y_s
    B_M = max((Y_s - Y_local) / (1.0 - Y_s), 0.0)
    d_loc = driving(Y_s, Y_local)
    print(f"{Y_local:10.4f} {B_M:12.4f} {d_loc:12.4f} {d_loc / d_const:20.3f}")

print()
print("Read-off: once the adjacent cell reaches a FRACTION of Y_s, the rate is")
print("already cut by ~that fraction. The cell fills from the source itself, so")
print("Y_local -> Y_s within a cell residence time and the rate self-quenches to")
print("whatever diffusion can clear from the band (~ Dv/dx -> GRID DEPENDENT).")

# ---------------------------------------------------------------------------
# 3) Plot: driving vs Y_local for both closures, two temperatures
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for T in (380.0, 400.0, 420.0):
        Ysat = Ys(T)
        yl = np.linspace(0.0, Ysat, 200)
        dl = (Ysat - yl) / (1.0 - yl)
        ax[0].plot(yl / Ysat, dl / (Ysat / 1.0), label=f"T={T:.0f} K  (Y_s={Ysat:.3f})")
    ax[0].axhline(1.0, ls="--", c="k", lw=1, label="constant Y_inf=0 (fixed)")
    ax[0].set_xlabel("local saturation  Y_local / Y_s")
    ax[0].set_ylabel("rate_local / rate_const")
    ax[0].set_title("Local-cell closure self-quenches\nas the band saturates")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    Ts = np.linspace(300, 480, 200)
    ax[1].plot(Ts, [Ys(T) for T in Ts], "b-", label="Y_s(T) = const-Y_inf driving")
    ax[1].plot(Ts, [0.5 * Ys(T) for T in Ts], "r--",
               label="local driving if Y_local=0.5 Y_s")
    ax[1].plot(Ts, [0.1 * Ys(T) for T in Ts], "r:",
               label="local driving if Y_local=0.9 Y_s")
    ax[1].axvline(489, c="grey", lw=1)
    ax[1].text(489, 0.05, " boil", rotation=90, va="bottom", fontsize=8, color="grey")
    ax[1].set_xlabel("interface temperature T [K]")
    ax[1].set_ylabel("driving  B_M/(1+B_M)")
    ax[1].set_title("Driving vs T: constant vs partially-saturated local")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = __file__.rsplit("/", 1)[0] + "/spalding_closure_demo.png"
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")
except Exception as e:
    print(f"\n(plot skipped: {e})")
