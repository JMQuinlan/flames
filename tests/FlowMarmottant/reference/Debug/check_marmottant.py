"""Marmottant coated-bubble validation.

Tracks the bubble radius from the GAS VOLUME (robust, no ray):
  2D: R = sqrt(V_gas/pi),   3D: R = (3 V_gas/4pi)^(1/3),   V_gas = sum (1-eta) V_cell
and checks it stays at R0 (Laplace equilibrium -> no drift). The curvature-derived
R and sigma_eff are printed by the solver itself ("Marmottant ... R=... sigma_eff=...").

Usage: python check_marmottant.py <output_dir> [R0]
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

R0 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2

def cells(d):
    return sorted(glob.glob(os.path.join(d, "*cell")))

def radius(pf):
    ds = yt.load(pf)
    dim = int(ds.dimensionality)
    ad = ds.all_data()
    ag = np.clip(1.0 - np.array(ad["eta"]), 0.0, 1.0)   # gas fraction = 1 - eta
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    Vg = float(np.sum(ag * vol))
    R = np.sqrt(Vg / np.pi) if dim == 2 else (3.0 * Vg / (4.0 * np.pi)) ** (1.0 / 3.0)
    return float(ds.current_time), float(R), dim

def main():
    d = sys.argv[1]
    cs = cells(d)
    if not cs:
        print("NO OUTPUT in", d); sys.exit(1)
    print("=" * 56)
    print("MARMOTTANT LAPLACE-BUBBLE CHECK :", os.path.basename(d))
    print("=" * 56)
    t0, R_init, dim = radius(cs[0])
    print(f"  {dim}D, {len(cs)} frames; volume-based R(0) = {R_init:.5f}  (R0 nominal = {R0})")
    print(f"  {'t':>11} {'R':>10} {'R/R0':>8}")
    Rs = []
    for c in cs:
        t, R, _ = radius(c)
        Rs.append(R)
        print(f"  {t:>11.4e} {R:>10.5f} {R/R0:>8.4f}")
    Rs = np.array(Rs)
    drift = 100.0 * np.max(np.abs(Rs - Rs[0]) / Rs[0])
    print("\n" + "=" * 56)
    print(f"  R(0)/R0={Rs[0]/R0:.4f}  R(end)/R0={Rs[-1]/R0:.4f}  "
          f"min/max R/R0={Rs.min()/R0:.4f}/{Rs.max()/R0:.4f}  max drift={drift:.2f}%")
    print(f"  STABLE (max drift < 10%, no 1.2 runaway): "
          f"{'PASS' if drift < 10.0 else 'FAIL'}")
    print("=" * 56)

if __name__ == "__main__":
    main()
