#!/usr/bin/env python3
"""Verify the sealed adiabatic saturation-equilibrium box (`Saturation_Box_1D`).

The box is closed (REFLECT walls -> fixed total mass) and adiabatic (energy only
moves to latent), so four relations must hold:

  (1) M_total = const                          (sealed; machine precision)
  (3) E_total(t) = E_total(0) - L_vap*M_vapor   (whole run: slope dE/dM_vapor = -L_vap)
  (2) p_vapor -> p_sat(T)  in the gas bulk      (saturation; evaporation rate -> 0)
  (4) interface (eta=0.5) position ~ M_vapor    (mass converted liquid -> vapor)

(1)+(3) come from integrals.dat and hold every step (no early-window restriction --
unlike the leaky-neumann tubes, REFLECT walls do not leak). (2)+(4) come from the
plotfiles via yt. Columns of integrals.dat (1-indexed):
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E_total 8:KE_total
    9:M_vapor 10:M_carrier

Usage:
    python3 check_saturation_box.py <output_dir> [L_vap] [stride]

<output_dir> holds both integrals.dat and the <NNNNN>cell plotfiles.
PASS (2) needs the run near equilibrium; if p_v/p_sat < ~0.8 the run just needs longer.
CONTROLS: apply_latent_heat=0 -> dE/dM_vapor ~ 0; spalding_saturation=0 -> no pinning.
"""

import os
import sys
import glob
import numpy as np

# ---- problem constants (must match Saturation_Box_1D) -----------------------
R_V       = 50.0                 # cp_vap - cv_vap
ANT_A, ANT_B, ANT_C = 1.959, 868.6, 0.0
L_VAP_DEFAULT = 1.0e5
RHO_L     = 45.83
LY        = 1.6e-4
X_INT0    = 2.0e-3


def p_sat(T):
    return np.power(10.0, ANT_A - ANT_B / (T + ANT_C)) * 1.0e5   # bar -> Pa


def interface_x(eta, x):
    """x of the eta=0.5 crossing (linear interp on the y-averaged profile)."""
    s = np.where(np.diff(np.sign(eta - 0.5)) != 0)[0]
    if s.size == 0:
        return float("nan")
    i = s[0]
    return float(np.interp(0.5, [eta[i], eta[i + 1]], [x[i], x[i + 1]]))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    L_vap = float(sys.argv[2]) if len(sys.argv) > 2 else L_VAP_DEFAULT
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # ---- (1),(3) from integrals.dat -----------------------------------------
    ipath = out if out.endswith(".dat") else os.path.join(out, "integrals.dat")
    if not os.path.exists(ipath):
        sys.exit(f"integrals.dat not found at {ipath}")
    d = np.loadtxt(ipath, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    ncol = d.shape[1]
    t, Mtot, Mliq, Etot, Mvap = d[:, 0], d[:, 1], d[:, 3], d[:, 6], d[:, 8]
    # Column 11 (M_vap_transformed) = cumulative INT INT m_dot_Vap dV dt, the source-only
    # transformed mass (added to WriteIntegrals). Preferred over M_vapor (col 9, the
    # rho_vap species mass) as the energy-slope driver: the latent sink is -L_vap*m_dot_Vap
    # on the SAME rate, so d(E_total)/d(M_transformed) = -L_vap exactly, independent of
    # vapor transport. Falls back to M_vapor on older 10-column integrals.dat.
    Mtrans = d[:, 10] if ncol > 10 else None
    drive = Mtrans if Mtrans is not None else Mvap
    drive_name = "M_transformed" if Mtrans is not None else "M_vapor"

    M_drift = np.max(np.abs(Mtot - Mtot[0]))
    M_rel = M_drift / max(abs(Mtot[0]), 1e-30)

    dDrive = drive - drive[0]
    if np.ptp(dDrive) > 0:
        coef = np.polyfit(dDrive, Etot, 1)
        slope = coef[0]                                  # dE_total/d(driver)
        r2 = 1.0 - np.var(Etot - np.polyval(coef, dDrive)) / max(np.var(Etot), 1e-30)
    else:
        slope, r2 = float("nan"), float("nan")
    rel = abs(slope + L_vap) / L_vap

    print(f"integrals: {ipath}  ({ncol} cols)")
    print(f"t: {t[0]:.3e} -> {t[-1]:.3e} s ({len(t)} rows)\n")
    print("(1) total mass conserved (sealed box):")
    print(f"    M_total drift {M_drift:.3e} (rel {M_rel:.2e})   "
          f"{'PASS' if M_rel < 1e-9 else 'CHECK'}\n")

    # (1b) discrete action-reaction (needs col 11): the cumulative transformed mass must
    # equal the liquid mass LOST and the vapor mass GAINED (Yv_init=0), since in a sealed
    # box rho_eta1/rho_vap change only by this source (wall flux = 0). After the floor-at-0
    # + m_dot-cap fix these agree to ~machine zero; a gap is residual non-conservation
    # (e.g. the old `small` partition floor injecting mass).
    if Mtrans is not None:
        dMt  = Mtrans[-1] - Mtrans[0]
        dMl  = -(Mliq[-1] - Mliq[0])        # liquid mass lost
        dMvp = Mvap[-1] - Mvap[0]           # vapor species gained
        den  = max(abs(dMt), 1e-30)
        ok = abs(dMt - dMl) / den < 5e-2 and abs(dMt - dMvp) / den < 5e-2
        print("(1b) action-reaction (col 11):  M_transformed = -dM_liq = dM_vapor")
        print(f"    M_transformed = {dMt:+.4e} kg ;  -dM_liq = {dMl:+.4e} (rel {abs(dMt-dMl)/den:.2e}) ; "
              f"dM_vapor = {dMvp:+.4e} (rel {abs(dMt-dMvp)/den:.2e})   {'PASS' if ok else 'CHECK'}\n")

    print(f"(3) energy bookkeeping  d(E_total)/d({drive_name}) = -L_vap:")
    print(f"    slope = {slope:.4e} J/kg (R^2={r2:.5f}) ; -L_vap = {-L_vap:.4e} ; "
          f"rel err {rel:.2e}   {'PASS' if rel < 5e-2 else 'CHECK'}\n")

    # ---- (2),(4) from plotfiles via yt --------------------------------------
    cells = sorted(glob.glob(os.path.join(out, "*cell")))
    if not cells:
        print("(2),(4) skipped: no *cell plotfiles found.")
        return
    import yt
    yt.set_log_level(50)

    xint_t, xint_v, sat_ratio_last, T_last = [], [], None, None
    for c in cells[::stride]:
        ds = yt.load(c)
        tc = float(ds.current_time)
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge,
                              dims=ds.domain_dimensions)
        eta = np.array(cg[("boxlib", "eta")])[:, :, 0].mean(axis=1)
        nx = eta.size
        xlo = float(ds.domain_left_edge[0]); xhi = float(ds.domain_right_edge[0])
        x = xlo + (np.arange(nx) + 0.5) * (xhi - xlo) / nx
        xi = interface_x(eta, x)
        xint_t.append(tc); xint_v.append(xi)

        if c is cells[::stride][-1]:
            T  = np.array(cg[("boxlib", "T")])[:, :, 0].mean(axis=1)
            rv = np.array(cg[("boxlib", "rho_vap")])[:, :, 0].mean(axis=1)
            gas = (eta > 0.9) & (x < xlo + 0.95 * (xhi - xlo))     # gas bulk, off the wall
            pv  = rv[gas] * R_V * T[gas]
            psat = p_sat(T[gas])
            sat_ratio_last = float(np.mean(pv) / max(np.mean(psat), 1e-30))
            T_last = float(np.mean(T[gas]))
            print("(2) saturation in the gas bulk (last plotfile, "
                  f"t={tc:.3e} s, T~{T_last:.1f} K):")
            print(f"    <p_vapor> = {np.mean(pv):.4e} Pa ;  <p_sat(T)> = {np.mean(psat):.4e} Pa")
            print(f"    p_v/p_sat = {sat_ratio_last:.3f}   "
                  f"{'PASS' if 0.8 < sat_ratio_last < 1.2 else 'CHECK (run longer / not yet saturated)'}\n")

    # (4) interface position vs the transformed mass (col 11 if present, else M_vapor)
    drive_pf = np.interp(xint_t, t, drive)
    xint_v = np.array(xint_v); xint_t = np.array(xint_t)
    good = np.isfinite(xint_v)
    if good.sum() > 2 and np.ptp(drive_pf[good]) > 0:
        s4, b4 = np.polyfit(drive_pf[good], xint_v[good], 1)
        r2_4 = 1.0 - np.var(xint_v[good] - (s4 * drive_pf[good] + b4)) / max(np.var(xint_v[good]), 1e-30)
        print(f"(4) interface position linear in {drive_name} (mass -> volume):")
        print(f"    x_int: {xint_v[good][0]:.5e} -> {xint_v[good][-1]:.5e} m   "
              f"(d x_int = {xint_v[good][-1]-xint_v[good][0]:+.3e} m)")
        print(f"    fit x_int = {s4:.3e}*{drive_name} + b ; R^2 = {r2_4:.4f}   "
              f"{'PASS' if r2_4 > 0.95 else 'CHECK'}\n")

    # equilibrium / rate decay diagnostic. Compare the tail rate to a POST-STARTUP
    # baseline (10% into the run), not the violent initial spike. True equilibrium
    # needs BOTH the rate decayed AND p_v ~ p_sat -- a decayed rate with p_v << p_sat
    # means the driving weakened (T/Y_s dropped) before the gas saturated, not equilibrium.
    n = len(t)
    if n > 10:
        rate_tail = np.polyfit(t[int(0.8 * n):], drive[int(0.8 * n):], 1)[0]
        i0 = int(0.1 * n)
        rate_base = np.polyfit(t[i0:int(0.2 * n)], drive[i0:int(0.2 * n)], 1)[0]
        frac = abs(rate_tail) / max(abs(rate_base), 1e-30)
        saturated = (sat_ratio_last is not None) and (sat_ratio_last > 0.8)
        verdict = ("near saturation equilibrium" if (frac < 0.1 and saturated)
                   else "rate decayed but p_v < p_sat (driving weakened, not saturated)"
                   if frac < 0.1 else "still evaporating (run to stop_time)")
        print(f"(info) evaporation rate: tail {rate_tail:+.3e} kg/s, "
              f"{100*frac:.0f}% of post-startup baseline -> {verdict}")


if __name__ == "__main__":
    main()
