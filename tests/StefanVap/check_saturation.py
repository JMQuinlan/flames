#!/usr/bin/env python3
"""Verify the Stage-2 physical Spalding driving force (Antoine saturation Y_s).

Reads integrals.dat from TWO Saturation_1D runs at different temperatures (cold,
hot) and checks that the evaporation rate responds to temperature the way the
Antoine saturation closure predicts. Columns (1-indexed):
    1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E_total 8:KE_total
    9:M_vapor 10:M_carrier

WHY A BAND-INTEGRATED PREDICTION (not a simple cell ratio). The volumetric rate
is m_dot_Vap = rho_g * Dv * (B_M/(1+B_M)) * |grad eta|, B_M = (Y_s-Y_inf)/(1-Y_s),
Y_s = Y_s(T,p) from Antoine. The driving force uses the INTERFACE temperature
T = MixedTemperature(rho,p,eta), now the THERMAL-EQUILIBRIUM mixing rule
T = [eta*(p+pi0)/((g0-1)cv0) + (1-eta)*(p+pi1)/((g1-1)cv1)] / rho (EOS.H). This uses
per-phase gamma/cv/pi (not mixed-effective values) and stays BOUNDED between the
two pure-phase temperatures across the band -- the old single-effective-fluid form
(p+p0_eff)/((g_eff-1) rho cv_eff) divided by the collapsing band density and
SUPERHEATED the interface (hundreds of K) which clamped this closure. With the fix,
when both phases start at the same pure-T the band is ISOTHERMAL at that T, so the
test runs at PHYSICAL temperatures with no clamping. The rate ratio is still
predicted by a band integral (the rho*eta prefactor varies across the band) using
the SAME T(eta) + Antoine the code uses -- an independent re-implementation, so a
mismatch flags a C++ coding error.

  *** Boiling clamp (x_s -> 0.99 above ~489 K) is no longer a practical worry now
      that the band no longer overshoots, but the check still warns if any band
      cell exceeds 489 K. Default pair is cold=300 / hot=400 K. ***

M_vapor starts at 0 (Yv_init=0), so over the early window M_vapor(t_win) ~ rate is
the cleanest rate proxy -> observed ratio = M_vapor_hot/M_vapor_cold.

Checks:
  (1) both runs evaporate (M_vapor up, M_liq down, carrier conserved)
  (2) observed ratio >> 1 (T-increasing: the saturation signature; legacy ~0.9)
      AND within ~2x of the band-integrated Antoine prediction

Usage:
    python3 check_saturation.py <cold>/integrals.dat <hot>/integrals.dat [T_cold] [T_hot] [t_win_s]
"""

import os
import sys
import numpy as np

# --- model constants (must match the Saturation_1D input) --------------------
P_REF = 1.0e5
ANT_A, ANT_B, ANT_C = 4.10549, 1625.928, -92.839   # n-dodecane Antoine (bar)
G0, P00, CV0 = 1.4, 0.0, 717.5                      # eos0 = air carrier
G1, P01, CV1 = 2.358, 4.01e8, 1080.0               # eos1 = dodecane (Tammann)
CP_VAP, CV_VAP = 1994.85, 1950.0
R_V = CP_VAP - CV_VAP                               # vapor specific gas const
R_G = (G0 - 1.0) * CV0                              # carrier specific gas const (cp-cv)
Y_INF = 0.0
# eta profile (matches the input: tanh, eps, x_int, dx)
EPS, X_INT, DOMAIN, NX = 4.0e-5, 2.0e-3, 4.0e-3, 200
T_BOIL = 489.0  # n-dodecane normal boiling point [K] (clamp warning threshold)


def T_interface(eta, rho):
    """Thermal-equilibrium mixture T (matches EOS.H MixedTemperature):
       T = [ eta*(p+pi0)/((g0-1)cv0) + (1-eta)*(p+pi1)/((g1-1)cv1) ] / rho.
    Per-phase (pure) gamma/cv/pi, volume(eta)-weighted, divided by rho. Stays
    BOUNDED between the pure-phase T's across the band -- no overshoot. When both
    phases start at the same pure-T, the band is isothermal at that T."""
    t0 = eta * (P_REF + P00) / ((G0 - 1.0) * CV0)
    t1 = (1.0 - eta) * (P_REF + P01) / ((G1 - 1.0) * CV1)
    return (t0 + t1) / rho


def bm_group(T):
    """Antoine -> Y_s -> B_M/(1+B_M); also returns x_s (to flag clamping)."""
    p_sat = 10.0 ** (ANT_A - ANT_B / (T + ANT_C)) * 1.0e5
    x_s = min(p_sat / P_REF, 0.99)
    nv, ng = x_s / R_V, (1.0 - x_s) / R_G
    Ys = nv / (nv + ng)
    BM = (Ys - Y_INF) / (1.0 - Ys)
    return BM / (1.0 + BM), x_s, Ys


def band_model(T_pure):
    """Pure-phase densities at T_pure -> band-integrated rate + band diagnostics.

    Replicates the code's interface rate integrand rho_g*(B_M/(1+B_M))*|grad eta|,
    rho_g = rho*eta, rho = eta*rho_air + (1-eta)*rho_dod, T from the mixed EOS.
    """
    rho_air = P_REF / ((G0 - 1.0) * CV0 * T_pure)
    rho_dod = (P_REF + P01) / ((G1 - 1.0) * CV1 * T_pure)   # textbook SG (p+p0)
    dx = DOMAIN / NX
    x = np.arange(0, DOMAIN, dx) + dx / 2.0
    eta = 0.5 * (1.0 + np.tanh((x - X_INT) / (np.sqrt(2.0) * EPS)))
    gmag = np.abs(np.gradient(eta, dx))
    rho = eta * rho_air + (1.0 - eta) * rho_dod
    Tb = T_interface(eta, rho)
    grp = np.array([bm_group(t)[0] for t in Tb])
    xs = np.array([bm_group(t)[1] for t in Tb])
    rate = np.sum(rho * eta * grp * gmag) * dx
    band = (eta > 0.01) & (eta < 0.99)
    return {"rate": rate, "rho_air": rho_air, "rho_dod": rho_dod,
            "Tmin": Tb[band].min(), "Tmax": Tb[band].max(),
            "clamp": bool((xs[band] >= 0.99).any())}


def load(path):
    if not os.path.exists(path):
        sys.exit(f"integrals.dat not found at {path}")
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] < 10:
        sys.exit(f"expected >=10 columns (need M_vapor/M_carrier); got {d.shape[1]}.")
    return d


def transit(T_pure):
    rho_l = (P_REF + P01) / ((G1 - 1.0) * CV1 * T_pure)   # textbook SG (p+p0)
    return 2.0e-3 / np.sqrt(G1 * (P_REF + P01) / rho_l)


def window(d, t_win):
    t = d[:, 0]
    m = t <= t_win
    if m.sum() < 3:
        m = t <= t[min(len(t) - 1, 50)]
    return {"t_end": t[m][-1], "n": int(m.sum()),
            "dMliq": d[m, 3][-1] - d[m, 3][0],
            "Mvap0": d[m, 8][0], "Mvap1": d[m, 8][-1], "dMvap": d[m, 8][-1] - d[m, 8][0],
            "dMcar": d[m, 9][-1] - d[m, 9][0], "Mcar0": d[m, 9][0]}


def main():
    if len(sys.argv) < 3:
        sys.exit("usage: check_saturation.py <cold>/integrals.dat <hot>/integrals.dat "
                 "[T_cold] [T_hot] [t_win_s]")
    cold_path, hot_path = sys.argv[1], sys.argv[2]
    T_cold = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    T_hot = float(sys.argv[4]) if len(sys.argv) > 4 else 400.0

    dc, dh = load(cold_path), load(hot_path)
    t_win = float(sys.argv[5]) if len(sys.argv) > 5 else 0.8 * min(transit(T_cold), transit(T_hot))
    sc, sh = window(dc, t_win), window(dh, t_win)
    bc, bh = band_model(T_cold), band_model(T_hot)

    print(f"cold: {cold_path}  (pure-T={T_cold:.0f} K)")
    print(f"hot : {hot_path}  (pure-T={T_hot:.0f} K)")
    print(f"early window = {t_win:.2e} s  (cold {sc['n']} rows, hot {sh['n']} rows)\n")

    for nm, T, s, b in (("cold", T_cold, sc, bc), ("hot", T_hot, sh, bh)):
        warn = "  *** BAND CLAMPS (>boiling): T-response will be killed ***" if b["clamp"] else ""
        print(f"[{nm}] pure-T={T:.0f}K  interface band T={b['Tmin']:.0f}-{b['Tmax']:.0f}K{warn}")
        print(f"        M_vapor: {s['Mvap0']:.4e} -> {s['Mvap1']:.4e}  (dM_vapor={s['dMvap']:+.3e})")
        print(f"        dM_liq={s['dMliq']:+.3e}   dM_carrier={s['dMcar']:+.3e} "
              f"(rel {abs(s['dMcar'])/abs(s['Mcar0']+1e-300):.1e})")

    evap_ok = (sc["dMvap"] > 0 and sh["dMvap"] > 0 and sc["dMliq"] < 0 and sh["dMliq"] < 0)
    car_ok = (abs(sc["dMcar"]) / abs(sc["Mcar0"] + 1e-300) < 1e-3 and
              abs(sh["dMcar"]) / abs(sh["Mcar0"] + 1e-300) < 1e-3)
    print(f"\n(1) both evaporate (M_vapor up, M_liq down) + carrier conserved:   "
          f"{'PASS' if (evap_ok and car_ok) else 'CHECK'}")

    pred = bh["rate"] / bc["rate"] if bc["rate"] else float("nan")
    obs = sh["dMvap"] / sc["dMvap"] if sc["dMvap"] else float("nan")
    ratio_of_ratios = obs / pred if pred else float("nan")
    ok2 = (obs > 3.0) and (0.5 < ratio_of_ratios < 2.0)
    print(f"(2) temperature response  rate_hot/rate_cold:")
    print(f"    observed (M_vapor ratio)   = {obs:.3f}")
    print(f"    band-integrated prediction = {pred:.3f}   (obs/pred = {ratio_of_ratios:.2f})")
    print(f"    -> {'PASS' if ok2 else 'CHECK'}   "
          f"(legacy spalding_saturation=0 would give ~0.9, i.e. flat/decreasing in T)")


if __name__ == "__main__":
    main()
