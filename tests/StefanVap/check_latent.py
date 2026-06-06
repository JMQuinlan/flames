#!/usr/bin/env python3
"""Verify the latent-heat energy coupling magnitude + sign (Stage 3c).

Reads integrals.dat from the LatentHeat_1D run. Columns (1-indexed):
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E_total 8:KE_total
    9:M_vapor 10:M_carrier

The latent-heat sink is E_rhs += -L_vap*m_dot_Vap. With the prescribed const-rate
evaporation, over the EARLY (pre acoustic-transit) window:
    d(E_total)/dt = -L_vap * INT(m_dot_Vap) dV = -L_vap * vap_const_mdot * Ly
    d(M_liq)/dt   =        - INT(m_dot_Vap) dV = -        vap_const_mdot * Ly
so the RATIO is exact and cancels the discrete |grad eta| quadrature factor:
    d(E_total) / d(M_liq) = L_vap

Decisive checks over the early window:
  (1) E_total decreases (sign)               -> evaporative cooling is active
  (2) d(E_total)/d(M_liq) ~ L_vap            -> right magnitude (quadrature-free)
  (3) M_liq decreases                        -> mass actually evaporates

CONTROL: rerun the same input with apply_latent_heat = 0; E_total then stays ~flat
over the early window (only the small acoustic redistribution), confirming the
energy drop here is the latent sink.

Usage:
    python3 check_latent.py [path/to/integrals.dat] [L_vap_expected] [early_window_s]
"""

import os
import sys
import numpy as np

X_INT = 2.0e-3
DOMAIN = 4.0e-3
RHO_L = 749.5
L_VAP_DEFAULT = 256.0e3
DEFAULT = "/mmfs1/home/jquinlan/runs/stefan/output_latent/integrals.dat"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    L_vap = float(sys.argv[2]) if len(sys.argv) > 2 else L_VAP_DEFAULT
    if not os.path.exists(path):
        sys.exit(f"integrals.dat not found at {path}")

    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 7:
        sys.exit(f"expected >=7 columns (E_total); got {d.shape[1]}.")
    t    = d[:, 0]
    Mliq = d[:, 3]
    Etot = d[:, 6]

    # Early window: first wall transit at the liquid sound speed from the interface.
    c_liq = np.sqrt(2.358 * (1.0e5 + 4.01e8) / RHO_L)
    transit = min(X_INT, DOMAIN - X_INT) / c_liq
    t_win = float(sys.argv[3]) if len(sys.argv) > 3 else transit
    m = t <= t_win
    if m.sum() < 3:
        m = t <= t[min(len(t) - 1, 50)]

    dE   = Etot[m][-1] - Etot[m][0]
    dMl  = Mliq[m][-1] - Mliq[m][0]
    ratio = dE / dMl if abs(dMl) > 0 else float("nan")
    rel = abs(ratio - L_vap) / L_vap

    print(f"file: {path}")
    print(f"t: {t[0]:.3e} -> {t[-1]:.3e} s ({len(t)} rows); "
          f"transit ~ {transit:.2e} s; early window = {t[m][-1]:.2e} s ({m.sum()} rows)\n")

    print("(1) evaporative cooling (E_total decreases):")
    print(f"    E_total: {Etot[m][0]:.8e} -> {Etot[m][-1]:.8e}  dE = {dE:+.4e} J/m   "
          f"{'PASS' if dE < 0.0 else 'CHECK (expected < 0)'}\n")

    print("(2) latent-heat magnitude  d(E_total)/d(M_liq) = L_vap:")
    print(f"    dE/dM_liq = {ratio:.4e} J/kg ; L_vap = {L_vap:.4e} J/kg ; "
          f"rel err = {rel:.2e}   {'PASS' if rel < 5e-2 else 'CHECK'}\n")

    print("(3) liquid evaporates (M_liq decreases):")
    print(f"    M_liq: {Mliq[m][0]:.6e} -> {Mliq[m][-1]:.6e}  dM_liq = {dMl:+.3e} kg/m   "
          f"{'PASS' if dMl < 0.0 else 'CHECK (expected < 0)'}")

    dE_full = Etot[-1] - Etot[0]
    print(f"\n(info) whole-run dE_total = {dE_full:+.3e} J/m (after transit the neumann "
          f"walls also exchange energy, as in the other tests -- use the early window).")


if __name__ == "__main__":
    main()
