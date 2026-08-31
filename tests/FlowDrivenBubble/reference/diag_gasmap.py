# -*- coding: utf-8 -*-
"""
FLOW-DRIVEN BUBBLE -- SPATIAL GAS MAP (leak localization)

For frame 0, an early frame, a mid frame, and the LAST usable frame:
  - gas mass in radial shells about the bubble center
  - gas mass within 3 finest cells of each domain face (face-localized
    accumulation: rectified/leaking faces glow here)
  - gas mass in angular cones about the +x axis vs the (1,1,1) diagonal
    (corner-directed transport vs axis-directed)
  - total m0, m1 for the ledger

usage:  python diag_gasmap.py <output_dir>
prints a compact table -- paste the whole output back.
"""
import os, re, sys
import numpy as np

R0 = 0.02

def stepnum(d):
    m = re.match(r"(\d+)cell$", os.path.basename(d))
    return int(m.group(1)) if m else -1

def main(out_dir):
    import yt
    yt.funcs.mylog.setLevel(40)
    pfs = sorted((os.path.join(out_dir, d) for d in os.listdir(out_dir)
                  if d.endswith("cell") and os.path.isdir(os.path.join(out_dir, d))),
                 key=stepnum)
    picks = sorted(set([0, min(5, len(pfs)-1), len(pfs)//2, len(pfs)-1]))
    for ip in picks:
        pf = pfs[ip]
        try:
            ds = yt.load(pf); ad = ds.all_data()
        except Exception as exc:
            print(f"[skip] {os.path.basename(pf)}: {exc}"); continue
        t = float(ds.current_time)
        x = np.array(ad["x"]); y = np.array(ad["y"]); z = np.array(ad["z"])
        r = np.sqrt(x*x + y*y + z*z)
        eta = np.array(ad["eta"]); vol = np.array(ad["index", "cell_volume"])
        re0 = np.array(ad["rho_eta0"]); re1 = np.array(ad["rho_eta1"])
        gv = np.clip(1.0 - eta, 0, 1) * vol          # gas volume per cell
        gm = re1 * vol                                # gas mass per cell
        hi = float(ds.domain_right_edge[0])
        dxf = hi / (int(ds.domain_dimensions[0]) * 2 ** int(ds.max_level))
        near = 3.0 * dxf
        print(f"\n===== {os.path.basename(pf)}  t={t:.5e}  "
              f"m0={float(np.sum(re0*vol)):.6e}  m1={float(np.sum(gm)):.6e}")
        # radial shells
        print("  gas volume by shell [R0 units]:")
        shells = [(0, .5), (.5, 1), (1, 1.5), (1.5, 2), (2, 3.5)]
        for a, b in shells:
            m = (r >= a*R0) & (r < b*R0)
            print(f"    r {a:>4.1f}-{b:<4.1f}: gasV={float(np.sum(gv[m])):.4e}"
                  f"  gasM={float(np.sum(gm[m])):.4e}")
        # face-adjacent gas
        print("  gas volume within 3 fine cells of each face:")
        faces = [("xlo", x < near), ("xhi", x > hi-near),
                 ("ylo", y < near), ("yhi", y > hi-near),
                 ("zlo", z < near), ("zhi", z > hi-near)]
        for nm, m in faces:
            print(f"    {nm}: gasV={float(np.sum(gv[m])):.4e}"
                  f"  min_eta={float(np.min(eta[m])):.4f}")
        # angular cones (cos > 0.9 about direction), r > 0.5 R0
        for nm, d in [("axis(100)", (1,0,0)), ("diag(111)",
                      (1/np.sqrt(3),)*3)]:
            ct = (x*d[0] + y*d[1] + z*d[2]) / np.maximum(r, 1e-30)
            m = (ct > 0.9) & (r > 0.5*R0)
            print(f"  cone {nm}: gasV={float(np.sum(gv[m])):.4e}")

if __name__ == "__main__":
    main(sys.argv[1])
