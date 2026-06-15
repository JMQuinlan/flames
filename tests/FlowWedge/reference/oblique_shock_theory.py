#!/usr/bin/env python3
# ============================================================================
#  oblique_shock_theory.py
#  ---------------------------------------------------------------------------
#  Analytical oblique-shock relations for the FlowWedge tests (shared helper).
#
#    theta-beta-Mach relation (weak & strong branches), maximum deflection
#    angle, Mach angle, and the Rankine-Hugoniot jumps across an oblique shock.
#
#  Conventions (all angles in RADIANS internally; helpers accept/return deg via
#  the *_deg wrappers):
#     theta = flow deflection (= wedge half-angle for an attached shock)
#     beta  = shock-wave angle measured from the upstream flow direction
#     M1    = upstream Mach number
# ============================================================================

import numpy as np

GAMMA_DEFAULT = 1.4


def mach_angle(M):
    """Mach (Mach-wave) angle mu = asin(1/M), radians.  None if M <= 1."""
    if M <= 1.0:
        return None
    return np.arcsin(1.0 / M)


def theta_from_beta(beta, M, gamma=GAMMA_DEFAULT):
    """Deflection theta(beta, M) from the theta-beta-Mach relation (radians)."""
    s = np.sin(beta)
    num = 2.0 / np.tan(beta) * (M * M * s * s - 1.0)
    den = M * M * (gamma + np.cos(2.0 * beta)) + 2.0
    return np.arctan2(num, den)


def theta_max(M, gamma=GAMMA_DEFAULT, n=20000):
    """Maximum deflection angle theta_max(M) and the beta at which it occurs.

    Returns (theta_max_rad, beta_at_max_rad).  None if M <= 1.
    """
    mu = mach_angle(M)
    if mu is None:
        return None
    betas = np.linspace(mu + 1e-6, np.pi / 2.0 - 1e-9, n)
    th = theta_from_beta(betas, M, gamma)
    k = int(np.argmax(th))
    return float(th[k]), float(betas[k])


def beta_from_theta(theta, M, gamma=GAMMA_DEFAULT, weak=True, n=20000):
    """Shock angle beta for a given deflection theta and Mach M (radians).

    weak=True returns the weak (smaller-beta) solution, else the strong one.
    Returns None if the shock is DETACHED (theta > theta_max) or M <= 1.
    """
    mu = mach_angle(M)
    if mu is None:
        return None
    tmax, beta_star = theta_max(M, gamma, n)
    if theta > tmax:
        return None  # detached -> bow shock
    if weak:
        lo, hi = mu + 1e-7, beta_star            # rising branch
    else:
        lo, hi = beta_star, np.pi / 2.0 - 1e-9   # falling branch
    # bisection on theta_from_beta(beta) - theta (monotone on each branch)
    f = lambda b: theta_from_beta(b, M, gamma) - theta
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # fall back to nearest grid point
        betas = np.linspace(lo, hi, n)
        return float(betas[int(np.argmin(np.abs(theta_from_beta(betas, M, gamma) - theta)))])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if flo * fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def oblique_jumps(M1, beta, gamma=GAMMA_DEFAULT):
    """Rankine-Hugoniot jumps across an oblique shock of angle beta.

    Returns a dict: Mn1, Mn2, M2 (downstream Mach), p2/p1, rho2/rho1, T2/T1,
    and theta (the deflection it produces).
    """
    Mn1 = M1 * np.sin(beta)
    p_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (Mn1 * Mn1 - 1.0)
    rho_ratio = (gamma + 1.0) * Mn1 * Mn1 / ((gamma - 1.0) * Mn1 * Mn1 + 2.0)
    T_ratio = p_ratio / rho_ratio
    Mn2 = np.sqrt((1.0 + 0.5 * (gamma - 1.0) * Mn1 * Mn1) /
                  (gamma * Mn1 * Mn1 - 0.5 * (gamma - 1.0)))
    theta = theta_from_beta(beta, M1, gamma)
    M2 = Mn2 / np.sin(beta - theta)
    return dict(Mn1=Mn1, Mn2=Mn2, M2=M2, p_ratio=p_ratio,
                rho_ratio=rho_ratio, T_ratio=T_ratio, theta=theta)


# ---- degree-friendly wrappers ---------------------------------------------
def beta_deg(theta_deg, M, gamma=GAMMA_DEFAULT, weak=True):
    b = beta_from_theta(np.radians(theta_deg), M, gamma, weak)
    return None if b is None else np.degrees(b)


def theta_max_deg(M, gamma=GAMMA_DEFAULT):
    r = theta_max(M, gamma)
    return None if r is None else np.degrees(r[0])


def is_attached(theta_deg, M, gamma=GAMMA_DEFAULT):
    """True if a wedge of half-angle theta_deg gives an ATTACHED oblique shock."""
    tm = theta_max_deg(M, gamma)
    return (tm is not None) and (theta_deg <= tm)


if __name__ == "__main__":
    # quick self-test / reference table
    theta = 15.0
    print(f"theta = {theta} deg,  gamma = {GAMMA_DEFAULT}")
    print(f"{'Ma':>5} {'theta_max':>10} {'attached':>9} {'beta_weak':>10} {'beta_strong':>11}")
    for M in (1.2, 2.0, 3.0, 5.0):
        tm = theta_max_deg(M)
        att = is_attached(theta, M)
        bw = beta_deg(theta, M, weak=True)
        bs = beta_deg(theta, M, weak=False)
        bw_s = f"{bw:10.2f}" if bw is not None else f"{'DETACHED':>10}"
        bs_s = f"{bs:11.2f}" if bs is not None else f"{'-':>11}"
        print(f"{M:5.1f} {tm:10.2f} {str(att):>9} {bw_s} {bs_s}")
