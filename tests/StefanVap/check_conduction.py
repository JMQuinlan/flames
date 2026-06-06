#!/usr/bin/env python3
"""Verify Fourier thermal conduction is conservative + stable (Stage 3b, ckpt 3b-2).

Reads integrals.dat from the Conduction_1D run. Columns (1-indexed):
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E_total 8:KE_total
    9:M_vapor 10:M_carrier

The conduction term d(E)/dt += div(k grad T) is in divergence form, so with the
neumann (adiabatic / no-flux) walls it only REDISTRIBUTES internal energy --
total energy is conserved. It is checked over the EARLY window: after the
conduction-driven thermal-expansion wave reaches the walls, the neumann BC leaks
a little (same caveat as every other test here), so conservation is a pre-transit
statement.

Decisive checks over the EARLY window:
  (1) E_total ~ constant   -> conduction is conservative (no spurious energy)
  (2) M_total ~ constant   -> quiescent, no phase change / no mass creation
  (3) run produced rows + finite values -> stable (no NaN / pressure floor)

Conduction "doing something" (the T gradient relaxing) is a VISUAL check in the
plotfiles: compare against the k0_thermal = 0 control (same input), where the T
step persists. integrals.dat has no T-gradient scalar, so this script only
verifies conservation + stability; KE_total growth is reported as the dynamical
fingerprint of the (physical) thermal expansion conduction drives.

Usage:
    python3 check_conduction.py [path/to/integrals.dat] [early_window_seconds]
"""

import os
import sys
import numpy as np

DOMAIN = 4.0e-3
X_C = 2.0e-3
# Sound speed in the cold air (gamma=1.4, p=1e5, rho=1.16144): ~347 m/s.
C_GAS = np.sqrt(1.4 * 1.0e5 / 1.16144)
EARLY_T = min(X_C, DOMAIN - X_C) / C_GAS   # ~ first wall transit from the center
DEFAULT = "/mmfs1/home/jquinlan/runs/stefan/output_conduction/integrals.dat"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        sys.exit(f"integrals.dat not found at {path}")

    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 8:
        sys.exit(f"expected >=8 columns (E_total/KE_total); got {d.shape[1]}.")
    t    = d[:, 0]
    Mtot = d[:, 1]
    Etot = d[:, 6]
    KE   = d[:, 7]

    t_win = float(sys.argv[2]) if len(sys.argv) > 2 else EARLY_T
    m = t <= t_win
    if m.sum() < 3:
        m = t <= t[min(len(t) - 1, 50)]

    finite = np.all(np.isfinite(d))

    # (1) total energy conserved over the early window
    E_drift = np.max(np.abs(Etot[m] - Etot[m][0]))
    E_rel = E_drift / max(abs(Etot[m][0]), 1e-30)

    # (2) total mass conserved
    M_drift = np.max(np.abs(Mtot[m] - Mtot[m][0]))
    M_rel = M_drift / max(abs(Mtot[m][0]), 1e-30)

    dKE = KE[m][-1] - KE[m][0]

    print(f"file: {path}")
    print(f"t: {t[0]:.3e} -> {t[-1]:.3e} s ({len(t)} rows); "
          f"early window = {t[m][-1]:.2e} s ({m.sum()} rows; transit ~ {EARLY_T:.2e} s)\n")

    print("(1) total energy conserved over early window (conduction is conservative):")
    print(f"    E_total: {Etot[m][0]:.6e} -> {Etot[m][-1]:.6e}  "
          f"max drift {E_drift:.3e} (rel {E_rel:.2e})   "
          f"{'PASS' if E_rel < 1e-3 else 'CHECK'}\n")

    print("(2) total mass conserved over early window:")
    print(f"    M_total: {Mtot[m][0]:.6e} -> {Mtot[m][-1]:.6e}  "
          f"max drift {M_drift:.3e} (rel {M_rel:.2e})   "
          f"{'PASS' if M_rel < 1e-6 else 'CHECK'}\n")

    print("(3) run stable (all integrals finite):")
    print(f"    {'PASS' if finite else 'CHECK (non-finite values present)'}\n")

    print(f"(info) KE_total: {KE[m][0]:.3e} -> {KE[m][-1]:.3e} (dKE = {dKE:+.3e}) "
          f"-- the thermal expansion conduction drives; ~0 only if conduction is off.")
    print(f"(info) compare the T profile in the plotfiles against the k0_thermal=0 "
          f"control to see the conductive smoothing.")


if __name__ == "__main__":
    main()
