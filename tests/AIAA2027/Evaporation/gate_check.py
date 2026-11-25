#!/usr/bin/env python3
"""
Gate check for a film-vaporization run: interface-band (0.05 < eta < 0.95) minimum pressure and gas density
on the last plotfile.  Exit 0 if band p_min > P_MIN (default 1e3 Pa), else exit 1.
The no-vaporization Ms 10 run has band p_min about 2.5e4 Pa at 10 us; the unfloored film run had 37 Pa.

  python3 gate_check.py RUN_OUTPUT_DIR [P_MIN]
"""
import sys, glob, os
import numpy as np
import yt
yt.funcs.mylog.setLevel(50)

d = sys.argv[1]; pmin_ok = float(sys.argv[2]) if len(sys.argv) > 2 else 1e3
pfs = sorted(glob.glob(os.path.join(d, "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
ds = yt.load(pfs[-1]); ad = ds.all_data()
eta = np.asarray(ad["boxlib", "eta"]); p = np.asarray(ad["boxlib", "pressure"])
rg = np.asarray(ad["boxlib", "rho_eta0"]) / np.maximum(eta, 1e-12)
b = (eta > 0.05) & (eta < 0.95)
pm = float(p[b].min()); p1 = float(np.percentile(p[b], 1)); rm = float(rg[b].min())
print(f"{os.path.basename(pfs[-1])} t={float(ds.current_time) * 1e6:.2f}us band p min {pm:.3g} p1pct {p1:.3g} "
      f"rho_g min {rm:.3g}  -> {'PASS' if pm > pmin_ok else 'FAIL'} (threshold {pmin_ok:g} Pa)")
sys.exit(0 if pm > pmin_ok else 1)
