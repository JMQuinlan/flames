#!/usr/bin/env python3
"""Verify the prescribed constant-rate phase change (Stage 1 plumbing check).

Reads the auto-logged integrals.dat from the ConstRate_1D run and checks the
mass-transfer plumbing. integrals.dat columns:
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq ...

The decisive, localizer-independent checks are evaluated over an EARLY window,
before the phase-change-driven expansion wave reaches the (neumann/outflow)
walls -- after that, mass advects out the domain and M_total drifts (a test-setup
artifact, not a code bug; use reflecting walls for a clean whole-run demo):

  (1) M_total conserved over the early window      -> action-reaction holds
  (2) dM_liq < 0, dM_gas > 0, dM_liq = -dM_gas     -> mass moves liquid -> gas

The absolute slope is reported as INFORMATIONAL only: it equals
-vap_const_mdot * (discrete INT|grad eta| dV), and the discrete gradient norm is
~1.07*Ly here, so expect ~7% above the analytic -vap_const_mdot*Ly. Conservation
and action-reaction do NOT depend on the localizer and are the real test.

Usage:
    python3 check_const_rate.py [path/to/integrals.dat] [early_window_seconds]
"""

import os
import sys
import numpy as np

# --- expected from the ConstRate_1D input -----------------------------------
VAP_CONST_MDOT = 1000.0     # kg/m^2/s
LY = 1.6e-4                 # m  (prob_hi[1] - prob_lo[1])
RHO_L = 749.5              # kg/m^3 (intrinsic liquid density)
RHO_G = 1.35598311723      # kg/m^3 (intrinsic vapor density)
X_INT = 2.0e-3             # interface position; nearest-wall distance for transit est.

EXPECTED_SLOPE = -VAP_CONST_MDOT * LY   # analytic d(M_liq)/dt  [kg/s]
EARLY_T = 1.0e-6           # default early window [s] (pre acoustic transit)

DEFAULT = "/mmfs1/home/jquinlan/runs/stefan/output_const_rate/integrals.dat"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    t_win = float(sys.argv[2]) if len(sys.argv) > 2 else EARLY_T
    if not os.path.exists(path):
        sys.exit(f"integrals.dat not found at {path}\n"
                 f"Pass the path explicitly: python3 check_const_rate.py <file> [t_win]")

    d = np.loadtxt(path, comments="#")
    t, Mtot, Mg, Ml = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

    # liquid-side acoustic transit estimate (when boundary outflow can begin)
    c_liq = np.sqrt(2.358 * (1.0e5 + 4.01e8) / RHO_L)
    transit = min(X_INT, 4.0e-3 - X_INT) / c_liq

    m = t <= t_win
    if m.sum() < 3:
        m = t <= t[min(len(t) - 1, 50)]  # fall back to first ~50 rows
        t_win = t[m][-1]

    tw, Mtw, Mgw, Mlw = t[m], Mtot[m], Mg[m], Ml[m]

    # (1) conservation over the early window: worst-case relative drift
    drift_w = np.max(np.abs(Mtw - Mtw[0]))
    rel_w = drift_w / Mtw[0]

    # (2) action-reaction + direction over the early window
    slope_l = np.polyfit(tw, Mlw, 1)[0]
    slope_g = np.polyfit(tw, Mgw, 1)[0]
    dMl, dMg = Mlw[-1] - Mlw[0], Mgw[-1] - Mgw[0]
    bal = abs(dMl + dMg) / max(abs(dMl), 1e-30)   # 0 = perfect action-reaction
    direction_ok = (dMl < 0.0) and (dMg > 0.0)

    err = abs(slope_l - EXPECTED_SLOPE) / abs(EXPECTED_SLOPE)

    print(f"file: {path}")
    print(f"t: {t[0]:.3e} -> {t[-1]:.3e} s  ({len(t)} rows)")
    print(f"acoustic transit interface->wall ~ {transit:.2e} s; "
          f"early window = {tw[-1]:.2e} s ({m.sum()} rows)\n")

    print("(1) M_total conserved over early window (action-reaction):")
    print(f"    max |M_total - M_total0| = {drift_w:.3e} kg  (rel {rel_w:.2e})   "
          f"{'PASS' if rel_w < 1e-5 else 'CHECK'}\n")

    print("(2) mass moves liquid -> gas, in balance, over early window:")
    print(f"    dM_liq = {dMl:+.3e} kg (<0), dM_gas = {dMg:+.3e} kg (>0)   "
          f"{'PASS' if direction_ok else 'CHECK'}")
    print(f"    |dM_liq + dM_gas|/|dM_liq| = {bal:.2e}   "
          f"{'PASS' if bal < 1e-3 else 'CHECK'}\n")

    print("(info) absolute rate (localizer-dependent, ~7% > analytic expected):")
    print(f"    analytic  d(M_liq)/dt = {EXPECTED_SLOPE:+.4e} kg/s")
    print(f"    measured  d(M_liq)/dt = {slope_l:+.4e} kg/s  (gas {slope_g:+.4e})  "
          f"[{err*100:.1f}% above analytic]\n")

    # whole-run drift = boundary outflow diagnostic
    drift_full = Mtot[-1] - Mtot[0]
    print(f"(info) whole-run M_total drift = {drift_full:+.3e} kg "
          f"(rel {abs(drift_full)/Mtot[0]:.2e}) -- boundary outflow after transit; "
          f"use reflecting walls for a clean whole-run conservation demo.")


if __name__ == "__main__":
    main()
