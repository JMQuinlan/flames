# -*- coding: utf-8 -*-
"""
===============================================================================
MARMOTTANT-MODIFIED RAYLEIGH--PLESSET AND KELLER--MIKSIS REFERENCES
===============================================================================

Both classical bubble-dynamics models assume a clean interface with a constant
surface tension.  A coated bubble differs in two ways, and BOTH are included
here:

  1. ELASTIC SHELL.  The tension is the piecewise Marmottant (2005) law of the
     instantaneous radius,

         sigma(R) = 0                              R <= R_buck      (buckled)
                    chi (R^2/R_buck^2 - 1)         R_buck < R < R_rup (elastic)
                    sigma_water                    R >= R_rup       (ruptured)

         R_rup = R_buck sqrt(1 + sigma_break/chi)

  2. SHELL (DILATATIONAL) VISCOSITY.  A Boussinesq--Scriven interface with
     surface dilatational viscosity kappa_s adds a damping pressure

         4 kappa_s Rdot / R^2

     to the normal stress balance.  For radial motion div_s u = 2 Rdot/R and
     the surface shear viscosity cancels identically, so kappa_s is the whole
     of the shell viscosity for spherical dynamics (see Hydro2.H).

The two model equations below are therefore the standard ones with the bubble
pressure replaced by

    p_B(R, Rdot) = p_g(R) - 2 sigma(R)/R - 4 mu_l Rdot/R - 4 kappa_s Rdot/R^2

where the gas follows a polytropic law p_g = p_g0 (R0/R)^(3 kappa_g).

Setting chi = 0 and kappa_s = 0 with sigma(R) -> sigma_water recovers the
uncoated references exactly, which is the regression check at the bottom of
this file.

USAGE
    from marmottant_rpe_km import Shell, Liquid, Gas, solve_rpe, solve_km
    sh  = Shell(chi=0.55, R_buck=0.0195, sigma_break=0.0728, kappa_s=2.0e-5)
    liq = Liquid(rho=1000.0, mu=1.0e-3, c=1500.0, p_inf=1.0e5)
    gas = Gas(p_g0=1.0e4, kappa_g=1.4)
    t, R, Rdot = solve_rpe(0.02, 0.0, (0.0, 1.0e-2), sh, liq, gas)

REFERENCES
    Marmottant et al., JASA 118(6):3499, 2005.      doi:10.1121/1.2109427
    Keller & Miksis, JASA 68(2):628, 1980.          doi:10.1121/1.384720
    Plesset, J. Appl. Mech. 16:277, 1949.           doi:10.1115/1.4009975
    Schmidmayer, Bryngelson & Colonius, JCP 402:109080, 2020.
===============================================================================
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------- #
#  Parameter containers
# --------------------------------------------------------------------------- #
@dataclass
class Shell:
    """Marmottant shell.  chi = 0 and kappa_s = 0 gives a clean interface."""
    chi: float = 0.0            # elastic modulus [N/m]
    R_buck: float = 0.0         # buckling radius [m]
    sigma_break: float = 0.0728  # rupture tension [N/m]
    sigma_water: float = 0.0728  # clean-interface tension [N/m]
    kappa_s: float = 0.0        # surface dilatational viscosity [kg/s]

    @property
    def R_rupture(self):
        if self.chi <= 0.0:
            return np.inf
        return self.R_buck * np.sqrt(1.0 + self.sigma_break / self.chi)

    def sigma(self, R):
        """Piecewise Marmottant tension sigma(R) [N/m]."""
        if self.chi <= 0.0:                 # uncoated: constant tension
            return self.sigma_water
        R = np.asarray(R, dtype=float)
        el = self.chi * (R**2 / self.R_buck**2 - 1.0)
        out = np.where(R <= self.R_buck, 0.0,
                       np.where(el >= self.sigma_break, self.sigma_water, el))
        return float(out) if out.ndim == 0 else out


@dataclass
class Liquid:
    rho: float = 1000.0         # density [kg/m^3]
    mu: float = 1.0e-3          # dynamic viscosity [Pa s]
    c: float = 1500.0           # sound speed [m/s]
    p_inf: float = 1.0e5        # ambient pressure [Pa]


@dataclass
class Gas:
    p_g0: float = 1.0e5         # initial gas pressure [Pa]
    kappa_g: float = 1.4        # polytropic exponent [-]


# --------------------------------------------------------------------------- #
#  Shared pressure terms
# --------------------------------------------------------------------------- #
def gas_pressure(R, R0, gas):
    """Polytropic gas pressure p_g(R) [Pa]."""
    return gas.p_g0 * (R0 / R) ** (3.0 * gas.kappa_g)


def bubble_pressure(R, Rdot, R0, shell, liq, gas):
    """p_B = p_g - 2 sigma(R)/R - 4 mu Rdot/R - 4 kappa_s Rdot/R^2  [Pa].

    The last term is the Boussinesq--Scriven dilatational contribution; it is
    the only shell-viscosity term that survives for spherical motion.
    """
    return (gas_pressure(R, R0, gas)
            - 2.0 * shell.sigma(R) / R
            - 4.0 * liq.mu * Rdot / R
            - 4.0 * shell.kappa_s * Rdot / R**2)


def _p_drive(t, drive):
    """Acoustic forcing p_ac(t) [Pa]; drive = (amplitude, angular frequency)."""
    if drive is None:
        return 0.0
    amp, omega = drive
    return amp * np.sin(omega * t)


# --------------------------------------------------------------------------- #
#  Marmottant-modified Rayleigh--Plesset
# --------------------------------------------------------------------------- #
def solve_rpe(R0, Rdot0, t_span, shell, liq, gas, drive=None, t_eval=None,
              rtol=1e-9, atol=1e-12):
    """Integrate

        rho (R Rddot + 3/2 Rdot^2) = p_B(R, Rdot) - p_inf - p_ac(t)

    with p_B from bubble_pressure(); returns (t, R, Rdot).
    """
    def rhs(t, y):
        R, Rd = y
        pB = bubble_pressure(R, Rd, R0, shell, liq, gas)
        Rdd = (pB - liq.p_inf - _p_drive(t, drive)) / (liq.rho * R) - 1.5 * Rd**2 / R
        return [Rd, Rdd]

    sol = solve_ivp(rhs, t_span, [R0, Rdot0], method="LSODA",
                    t_eval=t_eval, rtol=rtol, atol=atol, max_step=(t_span[1]-t_span[0])/500)
    return sol.t, sol.y[0], sol.y[1]


# --------------------------------------------------------------------------- #
#  Marmottant-modified Keller--Miksis
# --------------------------------------------------------------------------- #
def solve_km(R0, Rdot0, t_span, shell, liq, gas, drive=None, t_eval=None,
             rtol=1e-9, atol=1e-12):
    """Integrate

        (1 - Rdot/c) R Rddot + 3/2 (1 - Rdot/3c) Rdot^2
            = (1/rho)(1 + Rdot/c)(p_B - p_inf - p_ac) + (R/rho c) d/dt(p_B - p_ac)

    The dp_B/dt term is evaluated analytically for the gas and shell-elastic
    parts and by the chain rule for the viscous parts, so no numerical
    differentiation enters the right-hand side.  Returns (t, R, Rdot).
    """
    eps = 1.0e-30

    def dsigma_dR(R):
        """d(sigma)/dR; zero on the buckled and ruptured plateaus."""
        if shell.chi <= 0.0:
            return 0.0
        if R <= shell.R_buck or R >= shell.R_rupture:
            return 0.0
        return 2.0 * shell.chi * R / shell.R_buck**2

    def rhs(t, y):
        R, Rd = y
        pB = bubble_pressure(R, Rd, R0, shell, liq, gas)
        pg = gas_pressure(R, R0, gas)
        sig = shell.sigma(R)

        # dp_B/dt holding Rddot terms aside: the Rddot-dependent pieces of the
        # viscous derivatives are folded into the effective inertia below.
        dpg_dR = -3.0 * gas.kappa_g * pg / R
        dsig_term_dR = -2.0 * (dsigma_dR(R) / R - sig / R**2)
        dpB_dR = dpg_dR + dsig_term_dR + 4.0 * liq.mu * Rd / R**2 \
                 + 8.0 * shell.kappa_s * Rd / R**3
        # coefficient of Rddot inside dp_B/dt
        dpB_dRdot = -4.0 * liq.mu / R - 4.0 * shell.kappa_s / R**2

        amp_om = (0.0 if drive is None else drive[0] * drive[1] * np.cos(drive[1] * t))

        A = (1.0 - Rd / liq.c) * R - (R / (liq.rho * liq.c)) * dpB_dRdot
        B = (-1.5 * (1.0 - Rd / (3.0 * liq.c)) * Rd**2
             + (1.0 + Rd / liq.c) * (pB - liq.p_inf - _p_drive(t, drive)) / liq.rho
             + (R / (liq.rho * liq.c)) * (dpB_dR * Rd - amp_om))
        return [Rd, B / (A + eps)]

    sol = solve_ivp(rhs, t_span, [R0, Rdot0], method="LSODA",
                    t_eval=t_eval, rtol=rtol, atol=atol, max_step=(t_span[1]-t_span[0])/500)
    return sol.t, sol.y[0], sol.y[1]


# --------------------------------------------------------------------------- #
#  Self-check: uncoated limit and the linear shell-stiffened frequency
# --------------------------------------------------------------------------- #
def natural_frequency(R0, shell, liq, gas):
    """Linear natural frequency [rad/s] about R0, including the shell.

        omega_0^2 = [3 kappa_g p_g0 - 2 sigma(R0)/R0 + 2 sigma'(R0)] / (rho R0^2)
    """
    sig = shell.sigma(R0)
    dsig = (0.0 if shell.chi <= 0.0 or R0 <= shell.R_buck or R0 >= shell.R_rupture
            else 2.0 * shell.chi * R0 / shell.R_buck**2)
    return np.sqrt(max((3.0 * gas.kappa_g * gas.p_g0 - 2.0 * sig / R0 + 2.0 * dsig)
                       / (liq.rho * R0**2), 0.0))


if __name__ == "__main__":
    R0 = 0.02
    liq = Liquid()
    gas = Gas(p_g0=1.0e5)

    clean = Shell(chi=0.0, sigma_water=0.0728)
    coated = Shell(chi=0.55, R_buck=0.0195, sigma_break=0.0728,
                   sigma_water=0.0728, kappa_s=2.0e-5)

    print(f"clean  : sigma(R0) = {clean.sigma(R0):.5f} N/m   "
          f"f0 = {natural_frequency(R0, clean, liq, gas)/2/np.pi:.2f} Hz")
    print(f"coated : sigma(R0) = {coated.sigma(R0):.5f} N/m   "
          f"R_rup = {coated.R_rupture*1e3:.3f} mm   "
          f"f0 = {natural_frequency(R0, coated, liq, gas)/2/np.pi:.2f} Hz")

    # free oscillation from a 5% compression, both models, both shells
    for name, sh in (("clean", clean), ("coated", coated)):
        t, R, _ = solve_rpe(R0, 0.0, (0.0, 4.0e-3), sh, liq, Gas(p_g0=1.3e5), t_eval=np.linspace(0, 4e-3, 4001))
        tk, Rk, _ = solve_km(R0, 0.0, (0.0, 4.0e-3), sh, liq, Gas(p_g0=1.3e5), t_eval=np.linspace(0, 4e-3, 4001))
        print(f"{name:7s} RPE: R_max/R0 = {R.max()/R0:.4f}  R_min/R0 = {R.min()/R0:.4f} | "
              f"KM: R_max/R0 = {Rk.max()/R0:.4f}  R_min/R0 = {Rk.min()/R0:.4f}")
