# -*- coding: utf-8 -*-
"""
Rayleigh-Taylor Instability -- standalone VoF-style finite-volume solver.

Tracks two ideal-gas fluids of different density on a Cartesian grid using a
density-as-volume-marker approach (since both fluids share gamma here, the
density field IS the interface tracker; no separate alpha needed).

Compared to the original FV-LF-Rusanov version this implementation:
  - Uses **HLLC** flux instead of Rusanov, giving roughly an order of
    magnitude less interface diffusion.
  - Enables MUSCL slope limiting by default (minmod-style).
  - Bumps the default resolution from 64 to 96.
  - Cleaner structure / docstrings.

Companion test for the diffuse-interface hydro2 setup
`RayleighTaylorInstability_UNIT_TEST` -- run both and compare the visual
sharpness of the interface band.

Original credit: spatel6 (2023).  Modifications: 2026-06-10.
"""

import matplotlib.pyplot as plt
import numpy as np

# np.roll directions
_R = -1   # right
_L = 1    # left


# ============================================================================
# CONSERVED <-> PRIMITIVE
# ============================================================================

def getConserved(rho, vx, vy, P, gamma, vol):
    """Primitive -> conservative."""
    Mass   = rho * vol
    Momx   = rho * vx * vol
    Momy   = rho * vy * vol
    Energy = (P / (gamma - 1) + 0.5 * rho * (vx ** 2 + vy ** 2)) * vol
    return Mass, Momx, Momy, Energy


def getPrimitive(Mass, Momx, Momy, Energy, gamma, vol):
    """Conservative -> primitive, then enforce ghost-cell BCs."""
    rho = Mass / vol
    vx  = Momx / rho / vol
    vy  = Momy / rho / vol
    P   = (Energy / vol - 0.5 * rho * (vx ** 2 + vy ** 2)) * (gamma - 1)
    rho, vx, vy, P = setGhostCells(rho, vx, vy, P)
    return rho, vx, vy, P


# ============================================================================
# GRADIENTS + MUSCL RECONSTRUCTION
# ============================================================================

def getGradient(f, dx):
    """Centered gradient with reflecting y-ghost setting."""
    f_dx = (np.roll(f, _R, axis=0) - np.roll(f, _L, axis=0)) / (2 * dx)
    f_dy = (np.roll(f, _R, axis=1) - np.roll(f, _L, axis=1)) / (2 * dx)
    f_dx, f_dy = setGhostGradients(f_dx, f_dy)
    return f_dx, f_dy


def slopeLimit(f, dx, f_dx, f_dy):
    """Minmod-style limiter: clamp centered slope by forward/backward ratios."""
    eps_div = 1.0e-8
    # x
    fac_a = np.maximum(0.0, np.minimum(1.0, ( (f - np.roll(f, _L, axis=0)) / dx) / (f_dx + eps_div * (f_dx == 0))))
    fac_b = np.maximum(0.0, np.minimum(1.0, (-(f - np.roll(f, _R, axis=0)) / dx) / (f_dx + eps_div * (f_dx == 0))))
    f_dx  = fac_b * (fac_a * f_dx)
    # y
    fac_a = np.maximum(0.0, np.minimum(1.0, ( (f - np.roll(f, _L, axis=1)) / dx) / (f_dy + eps_div * (f_dy == 0))))
    fac_b = np.maximum(0.0, np.minimum(1.0, (-(f - np.roll(f, _R, axis=1)) / dx) / (f_dy + eps_div * (f_dy == 0))))
    f_dy  = fac_b * (fac_a * f_dy)
    return f_dx, f_dy


def extrapolateInSpaceToFace(f, f_dx, f_dy, dx):
    """Predict face-centered values from cell-centered values and gradients."""
    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, _R, axis=0)
    f_XR = f + f_dx * dx / 2
    f_YL = f - f_dy * dx / 2
    f_YL = np.roll(f_YL, _R, axis=1)
    f_YR = f + f_dy * dx / 2
    return f_XL, f_XR, f_YL, f_YR


# ============================================================================
# HLLC RIEMANN SOLVER (Toro)
# ============================================================================
# Standard Einfeldt wave-speed estimates + Batten-style contact wave.
# This is the *normal-direction* solver; for a y-face, swap the velocity
# arguments (vy is the normal, vx is tangential -- same as the original
# getFlux call signature).

def hllc_flux(rho_L, rho_R, vn_L, vn_R, vt_L, vt_R, P_L, P_R, gamma):
    """Return (F_mass, F_normal_mom, F_tangential_mom, F_energy) at a face.

    vn_* is the velocity normal to the face (direction of propagation),
    vt_* is the tangential velocity (no flux contribution to mass / energy
    except through advection).
    """
    eps = 1.0e-30

    # Energies
    en_L = P_L / (gamma - 1) + 0.5 * rho_L * (vn_L ** 2 + vt_L ** 2)
    en_R = P_R / (gamma - 1) + 0.5 * rho_R * (vn_R ** 2 + vt_R ** 2)

    # Sound speeds (clamp positivity for robustness)
    a_L = np.sqrt(gamma * np.maximum(P_L, 1e-12) / np.maximum(rho_L, 1e-12))
    a_R = np.sqrt(gamma * np.maximum(P_R, 1e-12) / np.maximum(rho_R, 1e-12))

    # Einfeldt wave-speed estimates
    S_L = np.minimum(vn_L - a_L, vn_R - a_R)
    S_R = np.maximum(vn_L + a_L, vn_R + a_R)

    # Contact wave (Batten et al. eq. 31)
    num = P_R - P_L + rho_L * vn_L * (S_L - vn_L) - rho_R * vn_R * (S_R - vn_R)
    den = rho_L * (S_L - vn_L) - rho_R * (S_R - vn_R)
    S_star = num / (den + eps * (den == 0))

    # Left + right physical fluxes
    F_L_mass = rho_L * vn_L
    F_L_momn = rho_L * vn_L ** 2 + P_L
    F_L_momt = rho_L * vn_L * vt_L
    F_L_en   = (en_L + P_L) * vn_L

    F_R_mass = rho_R * vn_R
    F_R_momn = rho_R * vn_R ** 2 + P_R
    F_R_momt = rho_R * vn_R * vt_R
    F_R_en   = (en_R + P_R) * vn_R

    # Conservative states
    U_L_mass = rho_L
    U_L_momn = rho_L * vn_L
    U_L_momt = rho_L * vt_L
    U_L_en   = en_L

    U_R_mass = rho_R
    U_R_momn = rho_R * vn_R
    U_R_momt = rho_R * vt_R
    U_R_en   = en_R

    # Star-region intermediate states (Toro eq. 10.39)
    coef_L = rho_L * (S_L - vn_L) / (S_L - S_star + eps * ((S_L - S_star) == 0))
    U_Ls_mass = coef_L
    U_Ls_momn = coef_L * S_star
    U_Ls_momt = coef_L * vt_L
    U_Ls_en   = coef_L * (en_L / rho_L
                           + (S_star - vn_L) * (S_star + P_L / (rho_L * (S_L - vn_L) + eps)))

    coef_R = rho_R * (S_R - vn_R) / (S_R - S_star + eps * ((S_R - S_star) == 0))
    U_Rs_mass = coef_R
    U_Rs_momn = coef_R * S_star
    U_Rs_momt = coef_R * vt_R
    U_Rs_en   = coef_R * (en_R / rho_R
                           + (S_star - vn_R) * (S_star + P_R / (rho_R * (S_R - vn_R) + eps)))

    # Star-region fluxes  F_*L = F_L + S_L*(U_*L - U_L)
    F_Ls_mass = F_L_mass + S_L * (U_Ls_mass - U_L_mass)
    F_Ls_momn = F_L_momn + S_L * (U_Ls_momn - U_L_momn)
    F_Ls_momt = F_L_momt + S_L * (U_Ls_momt - U_L_momt)
    F_Ls_en   = F_L_en   + S_L * (U_Ls_en   - U_L_en)

    F_Rs_mass = F_R_mass + S_R * (U_Rs_mass - U_R_mass)
    F_Rs_momn = F_R_momn + S_R * (U_Rs_momn - U_R_momn)
    F_Rs_momt = F_R_momt + S_R * (U_Rs_momt - U_R_momt)
    F_Rs_en   = F_R_en   + S_R * (U_Rs_en   - U_R_en)

    # Pick the correct flux region based on which wave-fan contains x/t = 0.
    def _pick(F_L, F_Ls, F_Rs, F_R):
        return np.where(
            S_L >= 0, F_L,
            np.where(
                S_star >= 0, F_Ls,
                np.where(S_R >= 0, F_Rs, F_R),
            ),
        )

    flux_mass = _pick(F_L_mass, F_Ls_mass, F_Rs_mass, F_R_mass)
    flux_momn = _pick(F_L_momn, F_Ls_momn, F_Rs_momn, F_R_momn)
    flux_momt = _pick(F_L_momt, F_Ls_momt, F_Rs_momt, F_R_momt)
    flux_en   = _pick(F_L_en,   F_Ls_en,   F_Rs_en,   F_R_en)

    return flux_mass, flux_momn, flux_momt, flux_en


# ============================================================================
# CONSERVATIVE UPDATE + SOURCES + GHOSTS
# ============================================================================

def applyFluxes(F, flux_F_X, flux_F_Y, dx, dt):
    """Apply face fluxes to one conserved-variable field."""
    F += -dt * dx * flux_F_X
    F +=  dt * dx * np.roll(flux_F_X, _L, axis=0)
    F += -dt * dx * flux_F_Y
    F +=  dt * dx * np.roll(flux_F_Y, _L, axis=1)
    return F


def addSourceTerm(Mass, Momx, Momy, Energy, g, dt):
    """Gravity source (acts on y-momentum and energy via M_y * g)."""
    Energy += dt * Momy * g
    Momy   += dt * Mass * g
    return Mass, Momx, Momy, Energy


def addGhostCells(rho, vx, vy, P):
    """Pad y-direction with reflecting ghost cells (1 layer top + bottom)."""
    rho = np.hstack((rho[:, 0:1], rho, rho[:, -1:]))
    vx  = np.hstack(( vx[:, 0:1],  vx,  vx[:, -1:]))
    vy  = np.hstack(( vy[:, 0:1],  vy,  vy[:, -1:]))
    P   = np.hstack((  P[:, 0:1],   P,   P[:, -1:]))
    return rho, vx, vy, P


def setGhostCells(rho, vx, vy, P):
    """Reflecting y-walls (mirror v_y, copy everything else)."""
    rho[:,  0] = rho[:,  1]
    vx[ :,  0] = vx[ :,  1]
    vy[ :,  0] = -vy[:,  1]
    P[  :,  0] = P[  :,  1]

    rho[:, -1] = rho[:, -2]
    vx[ :, -1] = vx[ :, -2]
    vy[ :, -1] = -vy[:, -2]
    P[  :, -1] = P[  :, -2]
    return rho, vx, vy, P


def setGhostGradients(f_dx, f_dy):
    """y-direction ghost gradients reflect (mirror about the wall)."""
    f_dy[:,  0] = -f_dy[:,  1]
    f_dy[:, -1] = -f_dy[:, -2]
    return f_dx, f_dy


# ============================================================================
# MAIN
# ============================================================================

def main():
    # -------------------- Simulation parameters --------------------
    N                = 96            # base resolution N x 3N  (was 64)
    boxsizeX         = 0.5
    boxsizeY         = 1.5
    gamma            = 1.4
    courant_fac      = 0.4
    tEnd             = 15.0
    tOut             = 0.1
    useSlopeLimiting = True          # was False; default ON for sharp interface
    plotRealTime     = True

    # -------------------- Mesh --------------------
    dx = boxsizeX / N
    vol = dx ** 2
    xlin = np.linspace(0.5 * dx, boxsizeX - 0.5 * dx, N)
    ylin = np.linspace(0.5 * dx, boxsizeY - 0.5 * dx, 3 * N)
    Y, X = np.meshgrid(ylin, xlin)

    # -------------------- Initial conditions --------------------
    g   = -0.1
    w0  = 0.0025
    P0  = 2.5

    rho = 1.0 + (Y > 0.75)                                           # heavy on top
    vx  = np.zeros(X.shape)
    vy  = w0 * (1 - np.cos(4 * np.pi * X)) * (1 - np.cos(4 * np.pi * Y / 3))
    P   = P0 + g * (Y - 0.75) * rho                                  # hydrostatic

    rho, vx, vy, P = addGhostCells(rho, vx, vy, P)
    Mass, Momx, Momy, Energy = getConserved(rho, vx, vy, P, gamma, vol)

    # -------------------- Figure setup --------------------
    fig = plt.figure(figsize=(4, 4), dpi=80)
    t = 0.0
    outputCount = 1

    # -------------------- Main loop --------------------
    while t < tEnd:
        # primitives
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, gamma, vol)

        # CFL timestep
        dt = courant_fac * np.min(dx / (np.sqrt(gamma * np.maximum(P, 1e-12)
                                                 / np.maximum(rho, 1e-12))
                                         + np.sqrt(vx ** 2 + vy ** 2)))
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            dt = outputCount * tOut - t
            plotThisTurn = True

        # gravity half-step
        Mass, Momx, Momy, Energy = addSourceTerm(Mass, Momx, Momy, Energy, g, dt / 2)

        # refresh primitives after the source split
        rho, vx, vy, P = getPrimitive(Mass, Momx, Momy, Energy, gamma, vol)

        # gradients
        rho_dx, rho_dy = getGradient(rho, dx)
        vx_dx,  vx_dy  = getGradient(vx,  dx)
        vy_dx,  vy_dy  = getGradient(vy,  dx)
        P_dx,   P_dy   = getGradient(P,   dx)

        # MUSCL slope limiting (now ON by default)
        if useSlopeLimiting:
            rho_dx, rho_dy = slopeLimit(rho, dx, rho_dx, rho_dy)
            vx_dx,  vx_dy  = slopeLimit(vx,  dx, vx_dx,  vx_dy)
            vy_dx,  vy_dy  = slopeLimit(vy,  dx, vy_dx,  vy_dy)
            P_dx,   P_dy   = slopeLimit(P,   dx, P_dx,   P_dy)

        # half-step temporal extrapolation (predictor)
        rho_p = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx + vy * rho_dy + rho * vy_dy)
        vx_p  = vx  - 0.5 * dt * (vx * vx_dx + vy * vx_dy + (1 / rho) * P_dx)
        vy_p  = vy  - 0.5 * dt * (vx * vy_dx + vy * vy_dy + (1 / rho) * P_dy)
        P_p   = P   - 0.5 * dt * (gamma * P * (vx_dx + vy_dy) + vx * P_dx + vy * P_dy)

        # spatial extrapolation to face centers
        rho_XL, rho_XR, rho_YL, rho_YR = extrapolateInSpaceToFace(rho_p, rho_dx, rho_dy, dx)
        vx_XL,  vx_XR,  vx_YL,  vx_YR  = extrapolateInSpaceToFace(vx_p,  vx_dx,  vx_dy,  dx)
        vy_XL,  vy_XR,  vy_YL,  vy_YR  = extrapolateInSpaceToFace(vy_p,  vy_dx,  vy_dy,  dx)
        P_XL,   P_XR,   P_YL,   P_YR   = extrapolateInSpaceToFace(P_p,   P_dx,   P_dy,   dx)

        # HLLC face fluxes
        # X-face: normal = vx, tangent = vy
        flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Energy_X = hllc_flux(
            rho_XL, rho_XR, vx_XL, vx_XR, vy_XL, vy_XR, P_XL, P_XR, gamma)
        # Y-face: normal = vy, tangent = vx  (swap vx <-> vy in call;
        # returned flux slots are (mass, normal=momy, tangent=momx, energy))
        flux_Mass_Y, flux_Momy_Y, flux_Momx_Y, flux_Energy_Y = hllc_flux(
            rho_YL, rho_YR, vy_YL, vy_YR, vx_YL, vx_YR, P_YL, P_YR, gamma)

        # conservative update
        Mass   = applyFluxes(Mass,   flux_Mass_X,   flux_Mass_Y,   dx, dt)
        Momx   = applyFluxes(Momx,   flux_Momx_X,   flux_Momx_Y,   dx, dt)
        Momy   = applyFluxes(Momy,   flux_Momy_X,   flux_Momy_Y,   dx, dt)
        Energy = applyFluxes(Energy, flux_Energy_X, flux_Energy_Y, dx, dt)

        # gravity half-step
        Mass, Momx, Momy, Energy = addSourceTerm(Mass, Momx, Momy, Energy, g, dt / 2)

        t += dt

        # live plot
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            plt.imshow(rho.T)
            plt.clim(0.8, 2.2)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            ax.set_title(f"VoF (HLLC + MUSCL)  t = {t:.2f}", fontsize=10)
            plt.pause(0.001)
            outputCount += 1

    plt.savefig('finitevolume2.png', dpi=240)
    plt.show()
    return 0


if __name__ == "__main__":
    main()
