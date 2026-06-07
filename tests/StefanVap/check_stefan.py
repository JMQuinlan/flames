#!/usr/bin/env python3
"""
Stage 3d -- quantitative Stefan check: transport-limited evaporation recedes the
interface as  s(t) ~ sqrt(t), so the EVAPORATED mass grows as

        dM_liq(t) = M_liq(0) - M_liq(t)  ~  t^p ,   p = 1/2   (Stefan / d^2-law)

The decisive test of the film-sink closure (spalding_film_sink=1) is the EXPONENT:
  * film sink (transport-responsive)  -> p ~ 0.5   (sqrt(t), diffusion-limited)
  * constant Y_inf (fixed driving)    -> p ~ 1.0   (linear, constant rate -- NOT Stefan)
so running the same case with the closure on vs off should flip the slope from ~1
to ~0.5. M_liq is the cleanest proxy: the liquid sits against the quiet xlo wall,
so it is insensitive to the gas-side outflow that corrupts M_total late in the run.

integrals.dat columns:
  1:Time 2:M_total 3:M_phase0_gas 4:M_phase1_liq 5:Px 6:Py 7:E_total 8:KE_total
  9:M_vapor 10:M_carrier

Usage:
  check_stefan.py <run>/integrals.dat                       # single run, fit p
  check_stefan.py <film>/integrals.dat <ctrl>/integrals.dat # film vs constant control
  check_stefan.py ... <t0> <t1>                             # explicit fit window [s]
"""
import sys
import numpy as np

I_TIME, I_MLIQ, I_MVAP = 0, 3, 8


def load(path):
    d = np.loadtxt(path, comments="#")
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if d.shape[1] < 4:
        sys.exit(f"{path}: expected >=4 columns, got {d.shape[1]}")
    return d


def fit_slope(t, dM, t0, t1):
    """Slope of log(dM) vs log(t) over [t0, t1], dM>0."""
    m = (t >= t0) & (t <= t1) & (dM > 0.0) & (t > 0.0)
    if m.sum() < 5:
        return None, m.sum()
    p = np.polyfit(np.log(t[m]), np.log(dM[m]), 1)
    return p[0], m.sum()


def analyze(path, t0, t1, label):
    d = load(path)
    t = d[:, I_TIME]
    dMliq = d[0, I_MLIQ] - d[:, I_MLIQ]          # evaporated liquid mass (>0)
    dMvap = d[:, I_MVAP] - d[0, I_MVAP]          # vapor gained
    tmax = t[-1]
    if t0 is None:
        t0, t1 = 0.10 * tmax, tmax                # skip startup transient
    p_liq, n = fit_slope(t, dMliq, t0, t1)
    p_vap, _ = fit_slope(t, dMvap, t0, t1)
    print(f"[{label}] {path}")
    print(f"    rows={len(t)}  t in [{t[0]:.3e}, {tmax:.3e}] s   fit window [{t0:.3e}, {t1:.3e}] ({n} pts)")
    print(f"    dM_liq: {dMliq[0]:.3e} -> {dMliq[-1]:.3e} kg   (evaporated)")
    if p_liq is None:
        print("    not enough positive points to fit a slope")
        return None
    print(f"    slope p (dM_liq ~ t^p):  {p_liq:.3f}    [Stefan sqrt(t) -> 0.5 ; constant rate -> 1.0]")
    if p_vap is not None:
        print(f"    slope p (M_vapor ~ t^p): {p_vap:.3f}")
    return p_liq


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    # trailing numeric args = explicit window
    t0 = t1 = None
    nums = []
    while args and _isnum(args[-1]):
        nums.insert(0, float(args.pop()))
    if len(nums) == 2:
        t0, t1 = nums
    paths = args
    print("=" * 70)
    print("Stefan recession exponent  dM_liq ~ t^p   (p=0.5 sqrt(t), p=1.0 linear)")
    print("=" * 70)
    p_film = analyze(paths[0], t0, t1, "film/run")
    p_ctrl = None
    if len(paths) >= 2:
        print()
        p_ctrl = analyze(paths[1], t0, t1, "control")

    print("\n" + "-" * 70)
    ok = (p_film is not None) and (0.35 <= p_film <= 0.65)
    if p_ctrl is not None:
        decisive = ok and (p_ctrl - p_film > 0.25)
        print(f"film slope {p_film:.3f} (want ~0.5), control slope {p_ctrl:.3f} (want ~1.0)")
        print("-> PASS: film sink gives sqrt(t), control is steeper (constant-rate)"
              if decisive else
              "-> CHECK: slopes did not separate as expected (tune Dv / window / resolution)")
    else:
        print(f"film slope {p_film if p_film is None else round(p_film,3)} (want ~0.5 for transport-limited Stefan)")
        print("-> PASS (sqrt(t) recession)" if ok else
              "-> CHECK: slope not ~0.5 (constant-rate? tune Dv / window / film_eps_mult / resolution)")


def _isnum(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    main()
