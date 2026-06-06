#!/usr/bin/env python3
"""Verify vapor-species evolution + carrier-mass conservation (Stage 3a, ckpt 2).

Reads integrals.dat from the Species_Evap_1D run. Columns (1-indexed):
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E 8:KE
    9:M_vapor 10:M_carrier

Decisive checks over the EARLY window (before the phase-change expansion wave
reaches the walls and advects mass out -- same caveat as Stage 1):
  (1) M_carrier ~ constant            -> inert carrier conserved (species plumbing)
  (2) d(M_vapor) = -d(M_phase1_liq)   -> evaporated liquid becomes vapor, in balance
  (3) M_vapor grows from ~0           -> evaporation actually produces vapor

Usage:
    python3 check_species.py [path/to/integrals.dat] [early_window_seconds]
"""

import os
import sys
import numpy as np

X_INT = 2.0e-3
RHO_L = 749.5
EARLY_T = 1.0e-6
DEFAULT = "/mmfs1/home/jquinlan/runs/stefan/output_species_evap/integrals.dat"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    t_win = float(sys.argv[2]) if len(sys.argv) > 2 else EARLY_T
    if not os.path.exists(path):
        sys.exit(f"integrals.dat not found at {path}")

    d = np.loadtxt(path, comments="#")
    if d.shape[1] < 10:
        sys.exit(f"expected >=10 columns (M_vapor/M_carrier); got {d.shape[1]}. "
                 f"Is this a species_transport=1 run?")
    t      = d[:, 0]
    Mtot   = d[:, 1]
    Mgas   = d[:, 2]
    Mliq   = d[:, 3]
    Mvap   = d[:, 8]
    Mcar   = d[:, 9]

    c_liq = np.sqrt(2.358 * (1.0e5 + 4.01e8) / RHO_L)
    transit = min(X_INT, 4.0e-3 - X_INT) / c_liq

    m = t <= t_win
    if m.sum() < 3:
        m = t <= t[min(len(t) - 1, 50)]
    tw = t[m]

    # (1) carrier conservation over the early window
    car_drift = np.max(np.abs(Mcar[m] - Mcar[m][0]))
    car_rel = car_drift / max(abs(Mcar[m][0]), 1e-30)

    # (2) vapor gained == liquid lost
    dMvap = Mvap[m][-1] - Mvap[m][0]
    dMliq = Mliq[m][-1] - Mliq[m][0]
    bal = abs(dMvap + dMliq) / max(abs(dMliq), 1e-30)

    print(f"file: {path}")
    print(f"t: {t[0]:.3e} -> {t[-1]:.3e} s ({len(t)} rows); "
          f"transit ~ {transit:.2e} s; early window = {tw[-1]:.2e} s ({m.sum()} rows)\n")

    print("(1) inert carrier conserved over early window:")
    print(f"    M_carrier: {Mcar[m][0]:.6e} -> {Mcar[m][-1]:.6e}  "
          f"max drift {car_drift:.3e} (rel {car_rel:.2e})   "
          f"{'PASS' if car_rel < 1e-4 else 'CHECK'}\n")

    print("(2) evaporated liquid becomes vapor (balance):")
    print(f"    d(M_vapor) = {dMvap:+.3e} kg ; d(M_liq) = {dMliq:+.3e} kg ; "
          f"|sum|/|dMliq| = {bal:.2e}   {'PASS' if bal < 5e-2 else 'CHECK'}\n")

    print("(3) vapor produced:")
    print(f"    M_vapor: {Mvap[m][0]:.3e} -> {Mvap[m][-1]:.3e} kg   "
          f"{'PASS' if dMvap > 0.0 else 'CHECK'}")

    drift_full = Mcar[-1] - Mcar[0]
    print(f"\n(info) whole-run M_carrier drift = {drift_full:+.3e} "
          f"(rel {abs(drift_full)/max(abs(Mcar[0]),1e-30):.2e}) -- boundary outflow "
          f"after transit, as in Stage 1.")


if __name__ == "__main__":
    main()
