#!/usr/bin/env python3
"""
UNIT TEST: energy conservation of the pure-cell guard in Hydro2::RelaxAndReinit.

The guard freezes alpha and does an "energy-consistent reinit":
    p_pure = ReinitMixturePressure(rho_e, a1, a2, ...)      <-- alpha_floor-CLAMPED
    E0     = PhasicEnergyFromPressure(p_pure, a1g, ...)     <-- RAW eta
    E1     = PhasicEnergyFromPressure(p_pure, a2g, ...)
ReinitMixturePressure inverts rho_e = p*A(alpha) + B(alpha), so writing the
energies back with a DIFFERENT alpha breaks E0 + E1 = rho_e.

Inside a gas bubble eta ~ 1e-13, which is BELOW alpha_floor = 1e-12, so the
two alphas differ and the mismatch is active in every core cell.
"""
import sys

GAM0, PI0 = 2.35, 1.0e9      # liquid (Tammann water)
GAM1, PI1 = 1.4, 0.0         # gas (ideal air)
ALPHA_FLOOR = 1.0e-12


def reinit_mixture_pressure(rho_e, a0, a1):
    A = a0 / (GAM0 - 1.0) + a1 / (GAM1 - 1.0)
    B = a0 * GAM0 * PI0 / (GAM0 - 1.0) + a1 * GAM1 * PI1 / (GAM1 - 1.0)
    return (rho_e - B) / max(A, 1e-16)


def phasic_energy(p, a, gam, pi):
    return a * (p + gam * pi) / (gam - 1.0)


def guard(eta, rho_e, consistent):
    a1 = min(max(eta, ALPHA_FLOOR), 1.0 - ALPHA_FLOOR)   # clamped
    a2 = 1.0 - a1
    a1g = min(max(eta, 0.0), 1.0)                        # raw
    a2g = 1.0 - a1g
    a1p, a2p = (a1g, a2g) if consistent else (a1, a2)
    p = reinit_mixture_pressure(rho_e, a1p, a2p)
    return p, phasic_energy(p, a1g, GAM0, PI0) + phasic_energy(p, a2g, GAM1, PI1)


def main():
    global PI0
    fail = 0
    p_gas = 1.0e5
    print(f"{'eta':>10} {'path':>10} {'p_pure':>14} {'E0+E1':>16} {'rho_e':>16} {'dE':>13} {'rel':>11}")
    for eta in (1.5e-13, 1.0e-12, 1.0e-10, 1.0e-6, 1.0e-3, 0.5):
        a1g = eta
        rho_e = phasic_energy(p_gas, a1g, GAM0, PI0) + phasic_energy(p_gas, 1 - a1g, GAM1, PI1)
        for consistent in (False, True):
            p, E = guard(eta, rho_e, consistent)
            dE = E - rho_e
            lab = "FIXED" if consistent else "legacy"
            print(f"{eta:10.3e} {lab:>10} {p:14.6f} {E:16.8e} {rho_e:16.8e} "
                  f"{dE:13.4e} {dE/rho_e:11.3e}")
            if consistent and abs(dE) > 1e-9 * abs(rho_e):
                print("    ^^ FAIL: fixed path must conserve energy"); fail += 1
        print()

    # the bubble core, quantified
    eta = 1.5e-13
    rho_e = phasic_energy(p_gas, eta, GAM0, PI0) + phasic_energy(p_gas, 1 - eta, GAM1, PI1)
    dE = guard(eta, rho_e, False)[1] - rho_e
    print(f"BUBBLE CORE (eta = {eta:.1e}, pi0 = {PI0:.0e}):")
    print(f"  legacy loses {dE:+.4e} J/m^3 per relaxation call "
          f"({dE/rho_e:+.3e} of the gas internal energy)")
    pred = (min(max(eta, ALPHA_FLOOR), 1.0) - eta) * (
        (p_gas + GAM0 * PI0) / (GAM0 - 1.0) - (p_gas + GAM1 * PI1) / (GAM1 - 1.0))
    print(f"  closed form -(a1-a1g)*[(p+g0 pi0)/(g0-1) - p/(g1-1)] = {-pred:+.4e}  (matches)")
    # stiffness scaling
    print("\n  scaling with liquid stiffness pi0 (why drift_unit at p0=1e7 shows ~nothing):")
    base = None
    for pi in (1.0e7, 1.0e8, 1.0e9):
        PI0 = pi
        r = phasic_energy(p_gas, eta, GAM0, PI0) + phasic_energy(p_gas, 1 - eta, GAM1, PI1)
        d = guard(eta, r, False)[1] - r
        if base is None: base = abs(d)
        print(f"    pi0 = {pi:.0e}:  dE = {d:+.4e} J/m^3   ({abs(d)/base:6.1f}x)")
    PI0 = 1.0e9

    print("\nRESULT:", "FAIL" if fail else "PASS -- fixed path conserves energy at every eta")
    return 1 if fail else 0


_RESULT = None


# ---------------------------------------------------------------------------
# HOW BIG IS IT, REALLY?  Honest bound: the mismatch is active ONLY where
# eta < alpha_floor = 1e-12, and the table above shows it vanishes by eta=1e-12.
# For a tanh interface eta = 0.5(1+tanh((r-R0)/eps)) that is a small inner core,
# not the whole bubble -- so this bug alone cannot explain a 0.69 %/cycle drift.
# ---------------------------------------------------------------------------
def impact_estimate(R0=0.02, eps=1.28e-3, calls_per_cycle=267000):
    import math
    # eta < 1e-12  <=>  (r-R0)/eps < -atanh(1 - 2e-12)
    r_core = R0 - eps * math.atanh(1.0 - 2.0e-12)
    frac = (r_core / R0) ** 3 if r_core > 0 else 0.0
    rel_per_call = 5.918e-9                      # from the table, at eta=1.5e-13
    per_cycle = rel_per_call * frac * calls_per_cycle
    print("\n--- WHOLE-BUBBLE IMPACT (R0 = %.3g m, eps = %.3g m) ---" % (R0, eps))
    print(f"  eta < 1e-12 only inside r < {r_core:.4e} m = {100*r_core/R0:.2f}% of R0")
    print(f"  affected volume fraction of the bubble: {100*frac:.3f}%")
    print(f"  gas internal-energy loss: {100*per_cycle:.5f} %/cycle")
    print(f"  MEASURED drift (f40.8_A2.72):            0.690 %/cycle")
    print(f"  -> this bug accounts for ~1/{0.690/(100*per_cycle):.0f} of the drift.")
    print("  CONCLUSION: real energy-conservation bug, worth fixing, but NOT the")
    print("  dominant drift mechanism.  The remaining suspect is the band itself:")
    print("  the Newton relaxation conserves rho E but MOVES internal energy from")
    print("  gas to the stiff liquid in every mixed cell, which is a diffuse-")
    print("  interface artifact and should scale with (band width / R0).")


if __name__ == "__main__":
    rc = main()
    impact_estimate()
    sys.exit(rc)
