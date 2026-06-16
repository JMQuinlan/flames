"""
Rayleigh-Plesset family ODE solvers used as analytical reference for the
diffuse-interface compressible bubble simulation output.

  Chen 2D cylindrical RPE (default, matches 2D Cartesian sim):
    rho_l R ln(r_inf/R) R'' + rho_l R'^2 (0.5 - ln(r_inf/R))
        = p_g(R) - p_inf - sigma/R - 2 mu_l R'/R

    Gas pressure (cylindrical adiabatic, polytropic exponent 2 gamma):
        p_g(R) = p_v + (p_g0 - p_v) (R0/R)^(2 gamma)

    Surface tension uses one curvature (sigma/R), viscous term uses 2 mu_l R'/R.
    The ln(r_inf/R) factor encodes the 2D far-field decay; r_inf must be
    specified (typically ~ half the simulation domain size).

  3D Rayleigh-Plesset (incompressible liquid):
    rho_l (R R'' + 1.5 R'^2) = p_g(R) - p_inf - 2 sigma/R - 4 mu_l R'/R
    p_g(R) = p_g0 (R0/R)^(3 gamma)

  Keller-Miksis (1st-order liquid compressibility correction, 3D):
    (1 - R'/c_l) R R'' + 1.5 (1 - R'/(3 c_l)) R'^2
        = (1 + R'/c_l)(p_g - p_inf)/rho_l + R/(rho_l c_l) d(p_g)/dt
          - 2 sigma/(rho_l R) - 4 mu_l R'/(rho_l R)

The 2D simulation is *cylindrical* in physics, so Chen 2D is the matching
reference. RP / KM are 3D-spherical and are reported for context (they will
disagree with the 2D simulation in a structured way - faster oscillation,
faster collapse).

Reference for the 2D form:
  Used in the existing rayleighplesset_UNIT_TEST.py script in this repository,
  attributed to Chen et al. 2010. The same form (cylindrical RPE with
  logarithmic far-field) appears in textbook treatments of 2D bubble dynamics
  (e.g., Brennen, "Cavitation and Bubble Dynamics", and the cylindrical
  analogue derived from inviscid 2D Bernoulli with a far-field cutoff).

Public API:
  chen_2d_ode(t, y, P)               -> [R', R'']  -- 2D cylindrical (default)
  rp_ode(t, y, P)                    -> [R', R'']  -- 3D spherical
  km_ode(t, y, P)                    -> [R', R'']  -- 3D Keller-Miksis
  solve(P, t_end, model='chen2d')    -> (t, R, R')
  Params dataclass (now includes r_inf, p_v)
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class Params:
    # Liquid
    rho_l: float = 1000.0      # liquid density [kg/m^3]
    p_inf: float = 1.01325e5   # far-field liquid pressure [Pa]
    mu_l:  float = 0.0         # liquid dynamic viscosity [Pa*s]
    c_l:   float = 1500.0      # liquid sound speed [m/s] (KM only)

    # Gas (adiabatic polytropic)
    gamma_gas: float = 1.4     # polytropic / adiabatic index
    p_g0:  float = 5.0e5       # initial gas pressure [Pa]
    p_v:   float = 0.0         # vapour pressure (Chen 2D only) [Pa]

    # Surface tension
    sigma: float = 0.0         # [N/m]

    # Initial state
    R0:    float = 1.0e-3      # initial radius [m]
    Rdot0: float = 0.0         # initial radial velocity [m/s]

    # Chen 2D-only: far-field cutoff for the logarithmic geometric factor.
    # Set this to roughly the distance from bubble center to the boundary at
    # which p_inf is enforced (typical = half-domain size).
    r_inf: float = 5.0e-3      # [m]


def _p_gas(R, P):
    """Adiabatic gas pressure as a function of radius."""
    return P.p_g0 * (P.R0 / R) ** (3.0 * P.gamma_gas)


def _dp_gas_dt(R, Rdot, P):
    """Time derivative of gas pressure (needed for KM)."""
    # d/dt[p_g0 * (R0/R)^(3 gamma)]
    #   = p_g0 * 3 gamma * (R0/R)^(3 gamma) * (-Rdot / R)
    return -3.0 * P.gamma_gas * _p_gas(R, P) * Rdot / R


def chen_2d_ode(t, y, P):
    """Chen 2D cylindrical RPE RHS. y = [R, Rdot]. Returns [Rdot, Rddot].

    Cylindrical-bubble dynamics with a logarithmic far-field cutoff (r_inf).
    Adiabatic gas exponent is 2*gamma (cylindrical) instead of 3*gamma (3D
    spherical). Surface tension contributes sigma/R (single curvature) and
    viscous drag is 2*mu_l*R'/R (one less factor than 3D).
    """
    R, Rdot = y
    if R <= 1.0e-12 or R >= 0.9 * P.r_inf:
        return [0.0, 0.0]

    ln_factor = np.log(P.r_inf / R)
    if ln_factor < 1.0e-6:
        ln_factor = 1.0e-6

    # Cylindrical adiabatic gas pressure (note exponent 2*gamma)
    p_B = P.p_v + (P.p_g0 - P.p_v) * (P.R0 / R) ** (2.0 * P.gamma_gas)

    pressure_term = (
        p_B - P.p_inf - 2.0 * P.mu_l * Rdot / R - P.sigma / R
    ) / P.rho_l

    numerator = pressure_term - Rdot * Rdot * (0.5 - ln_factor)
    denominator = R * ln_factor
    Rddot = numerator / denominator
    return [Rdot, Rddot]


def rp_ode(t, y, P):
    """Rayleigh-Plesset RHS. y = [R, Rdot]. Returns [Rdot, Rddot]."""
    R, Rdot = y
    pg = _p_gas(R, P)
    drive = (pg - P.p_inf - 2.0 * P.sigma / R - 4.0 * P.mu_l * Rdot / R) / P.rho_l
    Rddot = (drive - 1.5 * Rdot * Rdot) / R
    return [Rdot, Rddot]


def km_ode(t, y, P):
    """Keller-Miksis RHS (1st-order compressibility-corrected RP)."""
    R, Rdot = y
    pg = _p_gas(R, P)
    dpg_dt = _dp_gas_dt(R, Rdot, P)
    M = Rdot / P.c_l                                  # bubble-wall Mach
    inv_RHS = 1.0 - M
    LHS_quad = 1.5 * (1.0 - M / 3.0) * Rdot * Rdot
    drive = (
        (1.0 + M) * (pg - P.p_inf) / P.rho_l
        + R * dpg_dt / (P.rho_l * P.c_l)
        - 2.0 * P.sigma / (P.rho_l * R)
        - 4.0 * P.mu_l * Rdot / (P.rho_l * R)
    )
    Rddot = (drive - LHS_quad) / (R * inv_RHS)
    return [Rdot, Rddot]


def solve(P, t_end, model="chen2d", t_eval=None, atol=1.0e-12, rtol=1.0e-9, R_min=1.0e-9):
    """
    Integrate the chosen model from t=0 to t_end.

    model: "chen2d" (2D cylindrical, default), "rp" (3D Rayleigh-Plesset),
           or "km" (3D Keller-Miksis).
    Returns (t, R, Rdot). Robust to extreme collapse: a stop event at R = R_min
    prevents the solver from falling off into negative-R territory.
    """
    rhs = {"chen2d": chen_2d_ode, "rp": rp_ode, "km": km_ode}[model]

    def hit_min(t, y, *_):
        return y[0] - R_min
    hit_min.terminal = True
    hit_min.direction = -1

    if t_eval is None:
        t_eval = np.linspace(0.0, t_end, 4001)

    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        [P.R0, P.Rdot0],
        t_eval=t_eval,
        method="LSODA",
        atol=atol,
        rtol=rtol,
        events=hit_min,
        args=(P,),
    )
    return sol.t, sol.y[0], sol.y[1]


def rayleigh_collapse_time(P):
    """Classical Rayleigh collapse time for an empty bubble in liquid at p_inf.

    tau_R = 0.91468 * R0 * sqrt(rho_l / (p_inf - p_g0))
    Valid only when p_inf > p_g0 (liquid drives collapse).
    """
    dp = P.p_inf - P.p_g0
    if dp <= 0:
        return np.inf
    return 0.91468 * P.R0 * np.sqrt(P.rho_l / dp)


def linear_natural_frequency_3d(P):
    """3D Rayleigh-Plesset linearized natural angular frequency around equilibrium R0.

    omega0^2 = (3 gamma p_g0 - 2 sigma / R0) / (rho_l R0^2)
    Equilibrium assumes p_g0 = p_inf + 2 sigma / R0.
    """
    arg = (3.0 * P.gamma_gas * P.p_g0 - 2.0 * P.sigma / P.R0) / (P.rho_l * P.R0 * P.R0)
    if arg <= 0:
        return 0.0
    return np.sqrt(arg)


def linear_natural_frequency_chen2d(P, R_eq=None):
    """Chen 2D linearized natural angular frequency around equilibrium R_eq.

    Linearizing chen_2d_ode around R = R_eq gives
        rho_l R_eq ln(r_inf/R_eq) delta_R'' + (something) delta_R = 0
    with restoring stiffness from -dp_g/dR + sigma/R^2 evaluated at R_eq:
        omega0^2 = [2 gamma p_g0 (R0/R_eq)^(2 gamma) - sigma / R_eq]
                   / [rho_l R_eq^2 ln(r_inf / R_eq)]
    Equilibrium for cylindrical Laplace: p_g0_eq = p_inf + sigma / R_eq,
    using the polytropic gas relation. If R_eq is None, default to R0.
    """
    if R_eq is None:
        R_eq = P.R0
    pg_eq = P.p_g0 * (P.R0 / R_eq) ** (2.0 * P.gamma_gas)
    stiffness = 2.0 * P.gamma_gas * pg_eq - P.sigma / R_eq
    inertia = P.rho_l * R_eq * R_eq * np.log(P.r_inf / R_eq)
    if inertia <= 0 or stiffness <= 0:
        return 0.0
    return np.sqrt(stiffness / inertia)


# -----------------------------------------------------------------------------
# AMReX plot-file extraction helper
# -----------------------------------------------------------------------------

def extract_radius_history(amrex_output_dir, eta_threshold=0.5, axis="x"):
    """
    Walk all *cell directories in amrex_output_dir and extract bubble radius
    by finding the eta = eta_threshold crossing along a 1D ray through origin.

    Returns (times, radii) sorted by time.
    """
    import os
    import yt
    yt.funcs.mylog.setLevel(40)

    plot_files = sorted(
        os.path.join(amrex_output_dir, d)
        for d in os.listdir(amrex_output_dir)
        if os.path.isdir(os.path.join(amrex_output_dir, d)) and d.endswith("cell")
    )
    if not plot_files:
        raise FileNotFoundError(f"no *cell directories in {amrex_output_dir}")

    times = []
    radii = []
    for pf in plot_files:
        ds = yt.load(pf)
        L = float(ds.domain_right_edge[0])
        if axis == "x":
            ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                         ds.arr([L, 0.0, 0.0], "code_length"))
            x = np.array(ray["x"])
        else:
            ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"),
                         ds.arr([0.0, L, 0.0], "code_length"))
            x = np.array(ray["y"])
        eta = np.array(ray["eta"])
        order = np.argsort(x)
        x, eta = x[order], eta[order]

        # find first index where eta crosses threshold
        idx = np.where(eta >= eta_threshold)[0]
        if len(idx) == 0:
            radii.append(np.nan)
        else:
            i = idx[0]
            if i == 0:
                radii.append(x[0])
            else:
                # linear interpolation
                frac = (eta_threshold - eta[i - 1]) / (eta[i] - eta[i - 1])
                radii.append(x[i - 1] + frac * (x[i] - x[i - 1]))
        times.append(float(ds.current_time))

    return np.array(times), np.array(radii)
