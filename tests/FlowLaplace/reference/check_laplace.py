#!/usr/bin/env python3
"""Young-Laplace check: dp = p_gas - p_liquid against 2*sigma_eff/R (3D).

Usage: check_laplace.py DIR --sigma S --R0 R [--expect DP]
For the coated (Marmottant) case pass the EFFECTIVE sigma, e.g.
chi*(R0^2/R_buckling^2 - 1).
"""
import argparse, glob, math, re, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("d"); ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--R0", type=float, required=True)
    ap.add_argument("--tol", type=float, default=5.0, help="percent")
    a = ap.parse_args()
    import yt; yt.funcs.mylog.setLevel(50)
    pfs = sorted(glob.glob(a.d + "/*cell"),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    if not pfs:
        print(f"no plotfiles in {a.d}"); return 2
    expect = 2.0 * a.sigma / a.R0
    print(f"expected dp = 2*sigma/R = 2*{a.sigma}/{a.R0} = {expect:.6g}")
    print(f"{'frame':>8} {'t':>12} {'R/R0':>9} {'p_gas':>14} {'p_liq':>14} {'dp':>12} {'err %':>9}")
    worst = 0.0
    for pf in (pfs[0], pfs[len(pfs)//2], pfs[-1]):
        ds = yt.load(pf); ad = ds.all_data()
        eta = np.array(ad["eta"], float)
        try:   vol = np.array(ad["index", "cell_volume"], float)
        except Exception: vol = np.array(ad["cell_volume"], float)
        p = np.array(ad["pressure"], float)
        sym = 1
        for k in range(3):
            if float(ds.domain_left_edge[k]) > -1e-12: sym *= 2
        R = (3 * float((np.clip(1 - eta, 0, 1) * vol).sum()) * sym / (4 * math.pi)) ** (1/3)
        x = np.array(ad["x"], float); y = np.array(ad["y"], float); z = np.array(ad["z"], float)
        r = np.sqrt(x*x + y*y + z*z)
        gas = eta < 0.02
        liq = (eta > 0.98) & (r > 2.0 * a.R0)
        if not gas.any() or not liq.any():
            print(f"  {pf}: no pure cells"); continue
        pg = float(np.average(p[gas], weights=vol[gas]))
        pl = float(np.average(p[liq], weights=vol[liq]))
        dp = pg - pl
        err = 100 * (dp - expect) / expect
        worst = max(worst, abs(err))
        print(f"{re.search(r'(\d+)cell',pf).group(1):>8} {float(ds.current_time):12.4e} "
              f"{R/a.R0:9.5f} {pg:14.5f} {pl:14.5f} {dp:12.5f} {err:9.3f}")
    ok = worst <= a.tol
    print(f"\n  worst |error| = {worst:.3f} %   tol = {a.tol} %   -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
