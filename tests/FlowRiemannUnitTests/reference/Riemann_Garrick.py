"""
Exact Riemann solver for the Garrick et al. (2017) gas-liquid problem.

Two-fluid Riemann problem with a STIFFENED-GAS EOS on each side
(possibly with different gamma and p_inf):

    p = (gamma - 1) * rho * e  -  gamma * p_inf

This generalizes the standard ideal-gas Toro solver. With the substitution
    pi_K  = p_K  + p_inf_K        (modified pressure on side K)
    pi*   = p*   + p_inf_K
the wave-equation algebra is identical to ideal gas, with pi in place of p.
The only place this matters is the rarefaction / shock formulas, which are
implemented per-side here so each fluid can carry its own (gamma, p_inf).

Output: Garrick.hdf5 in the same directory as this script, with the same
dataset layout as the Toro reference files (xvec, density, velocity, pressure,
internal_energy, plus wave-speed scalars u_star, S_left, S_right,
S_head_*, S_tail_*, c_*, c_star_*, *_shock).

Usage:
    python Riemann_Garrick.py                 # default Garrick parameters
    python Riemann_Garrick.py --plot2screen   # also show plots
    python Riemann_Garrick.py --plot2file     # also save .eps plots
"""

import argparse
import os
from collections import namedtuple
from math import sqrt

import h5py
import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Problem definition - Garrick et al. (2017), Section 4.1, Eq. 68
# matches tests/FlowRiemannUnitTests/Angled/input_Garrick_*
# ---------------------------------------------------------------------------

State = namedtuple('State', ('Density', 'Velocity', 'Pressure', 'gamma', 'p_inf'))

# Right side - LIQUID (stiffened gas)
RIGHT = State(Density=0.991, Velocity=0.0, Pressure=3.059e-4, gamma=5.5, p_inf=1.505)
# Left side - GAS (ideal)
LEFT = State(Density=1.241, Velocity=0.0, Pressure=2.753,    gamma=1.4, p_inf=0.0)

T_FINAL = 0.2
NX = 1411
X_MIN, X_MAX = -1.0, 1.0

CASE_NAME = 'Garrick'   # group name written into the HDF5 file

# ---------------------------------------------------------------------------
# Stiffened-gas relations (per-side gamma and p_inf)
# ---------------------------------------------------------------------------

def speed_of_sound(W):
    return sqrt(W.gamma * (W.Pressure + W.p_inf) / W.Density)


def internal_energy_sg(rho, p, gamma, p_inf):
    # e = (p + gamma * p_inf) / ((gamma - 1) * rho)
    return (p + gamma * p_inf) / ((gamma - 1.0) * rho)


def f_K(p_star, W):
    """Velocity-change function across the K-wave for stiffened gas."""
    pK = W.Pressure
    pi_K = pK + W.p_inf
    pi_star = p_star + W.p_inf
    g = W.gamma

    if p_star > pK:
        # Shock branch
        A_K = 2.0 / ((g + 1.0) * W.Density)
        # B_K written so that (p_star + B_K) == (pi_star + (g-1)/(g+1) * pi_K)
        B_K = (g - 1.0) / (g + 1.0) * pK + 2.0 * g / (g + 1.0) * W.p_inf
        return (p_star - pK) * sqrt(A_K / (p_star + B_K))
    else:
        # Rarefaction branch
        c_K = speed_of_sound(W)
        return 2.0 * c_K / (g - 1.0) * ((pi_star / pi_K) ** ((g - 1.0) / (2.0 * g)) - 1.0)


def p_residual(p_star, left, right):
    return f_K(p_star, left) + f_K(p_star, right) + (right.Velocity - left.Velocity)


def density_star(W, p_star):
    g = W.gamma
    pi_K = W.Pressure + W.p_inf
    pi_star = p_star + W.p_inf
    if p_star > W.Pressure:
        # Hugoniot in pi
        beta = (g - 1.0) / (g + 1.0)
        return W.Density * (pi_star / pi_K + beta) / (beta * pi_star / pi_K + 1.0)
    else:
        # Isentropic in pi
        return W.Density * (pi_star / pi_K) ** (1.0 / g)


def shock_speed_left(W, p_star):
    g = W.gamma
    c_K = speed_of_sound(W)
    pi_K = W.Pressure + W.p_inf
    pi_star = p_star + W.p_inf
    return W.Velocity - c_K * sqrt((g + 1.0) / (2.0 * g) * (pi_star / pi_K)
                                   + (g - 1.0) / (2.0 * g))


def shock_speed_right(W, p_star):
    g = W.gamma
    c_K = speed_of_sound(W)
    pi_K = W.Pressure + W.p_inf
    pi_star = p_star + W.p_inf
    return W.Velocity + c_K * sqrt((g + 1.0) / (2.0 * g) * (pi_star / pi_K)
                                   + (g - 1.0) / (2.0 * g))


def fan_left(W, x, t, c_K):
    """Sample the LEFT rarefaction fan at (x, t)."""
    g = W.gamma
    pi_K = W.Pressure + W.p_inf
    factor = 2.0 / (g + 1.0) + (g - 1.0) / ((g + 1.0) * c_K) * (W.Velocity - x / t)
    rho = W.Density * factor ** (2.0 / (g - 1.0))
    u   = 2.0 / (g + 1.0) * (c_K + (g - 1.0) / 2.0 * W.Velocity + x / t)
    pi  = pi_K * factor ** (2.0 * g / (g - 1.0))
    p   = pi - W.p_inf
    return rho, u, p


def fan_right(W, x, t, c_K):
    """Sample the RIGHT rarefaction fan at (x, t)."""
    g = W.gamma
    pi_K = W.Pressure + W.p_inf
    factor = 2.0 / (g + 1.0) - (g - 1.0) / ((g + 1.0) * c_K) * (W.Velocity - x / t)
    rho = W.Density * factor ** (2.0 / (g - 1.0))
    u   = 2.0 / (g + 1.0) * (-c_K + (g - 1.0) / 2.0 * W.Velocity + x / t)
    pi  = pi_K * factor ** (2.0 * g / (g - 1.0))
    p   = pi - W.p_inf
    return rho, u, p


# ---------------------------------------------------------------------------
# Solve for p* and assemble the full solution
# ---------------------------------------------------------------------------

def solve(left, right, t, xvec):
    # The rarefaction formula needs pi_star > 0 i.e. p_star > -p_inf.
    # Both p_inf are >= 0 here, so any positive lower bound is safe.
    p_lo = 1e-10
    p_hi = 1e6 * max(left.Pressure + left.p_inf, right.Pressure + right.p_inf, 1.0)

    # Sanity-check the bracket
    f_lo = p_residual(p_lo, left, right)
    f_hi = p_residual(p_hi, left, right)
    if f_lo * f_hi > 0:
        raise RuntimeError(
            f"Bracket failure: f({p_lo}) = {f_lo:.3e}, f({p_hi}) = {f_hi:.3e}. "
            "Either there is no positive p* solution (cavitation) or the bracket is wrong."
        )

    p_star = brentq(p_residual, p_lo, p_hi, args=(left, right), xtol=1e-14)
    u_star = 0.5 * (left.Velocity + right.Velocity) + 0.5 * (f_K(p_star, right) - f_K(p_star, left))

    rho_star_L = density_star(left,  p_star)
    rho_star_R = density_star(right, p_star)

    c_L = speed_of_sound(left)
    c_R = speed_of_sound(right)

    left_shock  = p_star > left.Pressure
    right_shock = p_star > right.Pressure

    # Wave speeds (NaN for the branch that doesn't apply, matches Riemann.py output)
    if left_shock:
        S_left      = shock_speed_left(left, p_star)
        S_head_left = float('nan')
        S_tail_left = float('nan')
        c_star_left = float('nan')
    else:
        S_left      = float('nan')
        c_star_left = c_L * ((p_star + left.p_inf) / (left.Pressure + left.p_inf)) ** ((left.gamma - 1.0) / (2.0 * left.gamma))
        S_head_left = left.Velocity - c_L
        S_tail_left = u_star - c_star_left

    if right_shock:
        S_right      = shock_speed_right(right, p_star)
        S_head_right = float('nan')
        S_tail_right = float('nan')
        c_star_right = float('nan')
    else:
        S_right      = float('nan')
        c_star_right = c_R * ((p_star + right.p_inf) / (right.Pressure + right.p_inf)) ** ((right.gamma - 1.0) / (2.0 * right.gamma))
        S_head_right = right.Velocity + c_R
        S_tail_right = u_star + c_star_right

    # Sample the solution along xvec
    statevec = np.zeros((len(xvec), 3))   # [rho, u, p]
    for i, x in enumerate(xvec):
        if x < u_star * t:
            # Left of contact - liquid (uses LEFT EOS)
            if left_shock:
                if x < S_left * t:
                    rho, u, p = left.Density, left.Velocity, left.Pressure
                else:
                    rho, u, p = rho_star_L, u_star, p_star
            else:
                if x < S_head_left * t:
                    rho, u, p = left.Density, left.Velocity, left.Pressure
                elif x < S_tail_left * t:
                    rho, u, p = fan_left(left, x, t, c_L)
                else:
                    rho, u, p = rho_star_L, u_star, p_star
        else:
            # Right of contact - gas (uses RIGHT EOS)
            if right_shock:
                if x > S_right * t:
                    rho, u, p = right.Density, right.Velocity, right.Pressure
                else:
                    rho, u, p = rho_star_R, u_star, p_star
            else:
                if x > S_head_right * t:
                    rho, u, p = right.Density, right.Velocity, right.Pressure
                elif x > S_tail_right * t:
                    rho, u, p = fan_right(right, x, t, c_R)
                else:
                    rho, u, p = rho_star_R, u_star, p_star
        statevec[i] = (rho, u, p)

    # Internal energy uses the EOS of whichever fluid sits at this x
    evec = np.zeros(len(xvec))
    for i, x in enumerate(xvec):
        if x < u_star * t:
            evec[i] = internal_energy_sg(statevec[i, 0], statevec[i, 2], left.gamma, left.p_inf)
        else:
            evec[i] = internal_energy_sg(statevec[i, 0], statevec[i, 2], right.gamma, right.p_inf)

    return {
        'p_star': p_star, 'u_star': u_star,
        'rho_star_L': rho_star_L, 'rho_star_R': rho_star_R,
        'c_left': c_L, 'c_right': c_R,
        'c_star_left': c_star_left, 'c_star_right': c_star_right,
        'left_shock': left_shock, 'right_shock': right_shock,
        'S_left': S_left, 'S_right': S_right,
        'S_head_left': S_head_left, 'S_tail_left': S_tail_left,
        'S_head_right': S_head_right, 'S_tail_right': S_tail_right,
        'statevec': statevec, 'evec': evec,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exact solver for Garrick gas-liquid Riemann.")
    parser.add_argument("--plot2screen", action="store_true", help="Show plots")
    parser.add_argument("--plot2file",   action="store_true", help="Save plots to .eps")
    args = parser.parse_args()

    print(f"Garrick gas-liquid Riemann (stiffened gas L | ideal gas R)")
    print(f"  LEFT   (liquid): rho={LEFT.Density}, u={LEFT.Velocity}, p={LEFT.Pressure}, "
          f"gamma={LEFT.gamma}, p_inf={LEFT.p_inf}")
    print(f"  RIGHT  (gas):    rho={RIGHT.Density}, u={RIGHT.Velocity}, p={RIGHT.Pressure}, "
          f"gamma={RIGHT.gamma}, p_inf={RIGHT.p_inf}")
    print(f"  t = {T_FINAL}, xvec in [{X_MIN}, {X_MAX}], Nx = {NX}")

    xvec = np.linspace(X_MIN, X_MAX, NX)
    sol = solve(LEFT, RIGHT, T_FINAL, xvec)

    print()
    print(f"  c_L = {sol['c_left']:.6f}, c_R = {sol['c_right']:.6f}")
    print(f"  p*  = {sol['p_star']:.6e}, u* = {sol['u_star']:.6f}")
    print(f"  rho*_L = {sol['rho_star_L']:.6f}, rho*_R = {sol['rho_star_R']:.6f}")
    print(f"  left  wave: {'SHOCK' if sol['left_shock']  else 'RAREFACTION'}, "
          f"S_left  = {sol['S_left']}, S_head_left  = {sol['S_head_left']}, "
          f"S_tail_left  = {sol['S_tail_left']}")
    print(f"  right wave: {'SHOCK' if sol['right_shock'] else 'RAREFACTION'}, "
          f"S_right = {sol['S_right']}, S_head_right = {sol['S_head_right']}, "
          f"S_tail_right = {sol['S_tail_right']}")

    statevec = sol['statevec']
    evec     = sol['evec']

    # ------------------------------------------------------------------
    # Save HDF5 - same layout as Toro1a.hdf5 etc.
    # ------------------------------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(here, f'{CASE_NAME}.hdf5')
    group_path = '/' + CASE_NAME

    with h5py.File(filename, 'a') as f:
        if group_path in f:
            print(f"\nGroup '{group_path}' already in {filename}, replacing.")
            del f[group_path]
        g = f.create_group(CASE_NAME)
        g.create_dataset('xvec',            data=xvec)
        g.create_dataset('density',         data=statevec[:, 0])
        g.create_dataset('velocity',        data=statevec[:, 1])
        g.create_dataset('pressure',        data=statevec[:, 2])
        g.create_dataset('internal_energy', data=evec)
        g.create_dataset('S_left',       data=sol['S_left'])
        g.create_dataset('S_right',      data=sol['S_right'])
        g.create_dataset('S_head_left',  data=sol['S_head_left'])
        g.create_dataset('S_head_right', data=sol['S_head_right'])
        g.create_dataset('S_tail_left',  data=sol['S_tail_left'])
        g.create_dataset('S_tail_right', data=sol['S_tail_right'])
        g.create_dataset('c_left',       data=sol['c_left'])
        g.create_dataset('c_right',      data=sol['c_right'])
        g.create_dataset('c_star_left',  data=sol['c_star_left'])
        g.create_dataset('c_star_right', data=sol['c_star_right'])
        g.create_dataset('u_star',       data=sol['u_star'])
        g.create_dataset('left_shock',   data=bool(sol['left_shock']))
        g.create_dataset('right_shock',  data=bool(sol['right_shock']))

    print(f"\nWrote {filename}  (group '{CASE_NAME}')")

    # ------------------------------------------------------------------
    # Optional plots
    # ------------------------------------------------------------------
    if args.plot2screen or args.plot2file:
        import matplotlib.pyplot as plt
        plt.set_loglevel("Error")

        wave_lines = []
        for s, label, style in (
            (sol['S_head_left'],  'Fan head', '--g'),
            (sol['S_tail_left'],  'Fan tail', '--c'),
            (sol['S_head_right'], None,       '--g'),
            (sol['S_tail_right'], None,       '--c'),
            (sol['u_star'],       'Contact',  '--m'),
            (sol['S_left'],       'Shock',    '--r'),
            (sol['S_right'],      None,       '--r'),
        ):
            if not np.isnan(s):
                wave_lines.append((s * T_FINAL, label, style))

        for ydata, ylabel, fname in (
            (statevec[:, 0], r'$\rho$',  'density'),
            (statevec[:, 1], r'$U$',     'velocity'),
            (statevec[:, 2], r'$p$',     'pressure'),
            (evec,           r'$e$',     'internal_energy'),
        ):
            fig = plt.figure(figsize=(6, 5))
            plt.plot(xvec, ydata, '-b', label=ylabel)
            yextra = ydata.max() - ydata.min()
            ymin = ydata.min() - 0.05 * yextra
            ymax = ydata.max() + 0.05 * yextra
            for xline, lab, style in wave_lines:
                plt.plot([xline, xline], [ymin, ymax], style, label=lab)
            plt.xlabel('x'); plt.xlim([X_MIN, X_MAX]); plt.ylabel(ylabel)
            plt.legend(loc=6)
            if args.plot2file:
                fig.savefig(os.path.join(here, f'{CASE_NAME}_Riemann_{fname}.eps'))
            if args.plot2screen:
                plt.show()
            plt.close(fig)


if __name__ == '__main__':
    main()
