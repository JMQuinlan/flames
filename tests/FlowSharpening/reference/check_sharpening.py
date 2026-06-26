"""Validate interface sharpening: CONSERVATIVE + actually SHARPENS.

Per plotfile we compute two global integrals (AMR-correct via yt leaf cells):
  M = integral eta dV          -> the conserved volume-fraction (should stay flat)
  S = integral eta(1-eta) dV   -> "band volume": large for a smeared interface,
                                  shrinks as it sharpens (no ray needed)

Pass criteria (compares sharpening ON vs OFF dirs):
  CONSERVATIVE: |M(t)-M(0)|/M(0) stays < TOL_CONS for the ON run
  SHARPENS:     S_on(end)/S_on(0) is meaningfully < S_off(end)/S_off(0)

Usage:
  python check_sharpening.py <OFF_dir> <ON_dir> [TOL_CONS_percent]
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

TOL_CONS = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5   # percent

def cells(d):
    return sorted(glob.glob(os.path.join(d, "*cell")))

def integrals(pf):
    ds = yt.load(pf)
    ad = ds.all_data()
    eta = np.clip(np.array(ad["eta"]), 0.0, 1.0)
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    M = float(np.sum(eta * vol))
    S = float(np.sum(eta * (1.0 - eta) * vol))
    return float(ds.current_time), M, S

def series(d, label):
    cs = cells(d)
    if not cs:
        print(f"  {label}: NO OUTPUT in {d}"); return None
    t0, M0, S0 = integrals(cs[0])
    rows = []
    for c in cs:
        t, M, S = integrals(c)
        rows.append((t, M, S))
    print(f"\n  {label}  ({len(cs)} frames)   M0={M0:.6e}  S0={S0:.6e}")
    print(f"    {'t':>10} {'dM/M0 %':>11} {'S/S0':>9}")
    for t, M, S in rows:
        print(f"    {t:>10.4e} {100*(M-M0)/M0:>+11.4f} {S/S0:>9.4f}")
    return dict(t=[r[0] for r in rows], M=[r[1] for r in rows], S=[r[2] for r in rows],
               M0=M0, S0=S0, dM_max=max(abs(100*(r[1]-M0)/M0) for r in rows),
               S_ratio_end=rows[-1][2]/S0)

def main():
    off_dir, on_dir = sys.argv[1], sys.argv[2]
    print("=" * 66)
    print("INTERFACE-SHARPENING VALIDATION")
    print("=" * 66)
    off = series(off_dir, "apply_sharpening = 0 (baseline)")
    on  = series(on_dir,  "apply_sharpening = 1 (sharpening)")
    if off is None or on is None:
        print("\n  MISSING OUTPUT -- cannot judge."); sys.exit(1)
    print("\n" + "=" * 66)
    conservative = on["dM_max"] < TOL_CONS
    sharpens = on["S_ratio_end"] < 0.9 * off["S_ratio_end"]   # ON narrows clearly more than OFF
    print(f"  CONSERVATIVE: max |dM/M0| over ON run = {on['dM_max']:.4f}%  "
          f"(tol {TOL_CONS}%)  -> {'PASS' if conservative else 'FAIL'}")
    print(f"  SHARPENS:     S/S0 end  ON={on['S_ratio_end']:.4f}  vs  OFF={off['S_ratio_end']:.4f}  "
          f"-> {'PASS' if sharpens else 'FAIL'}")
    print("=" * 66)
    print(f"  VERDICT: {'PASS -- conservative + sharpens' if (conservative and sharpens) else 'FAIL -- see above'}")
    print("=" * 66)

if __name__ == "__main__":
    main()
