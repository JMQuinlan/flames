# -*- coding: utf-8 -*-
"""
Drift metrics for the FlowDrivenBubble factorial unit tests (UT-A, UT-RA,
UT-AC, UT-RC, UT-RAC).

Per plotfile: total per-phase masses, total energy, mean/corner pressures,
max |u|.  The verdict metric is the SECULAR DRIFT RATE (per drive period
T = 12.26 ms) of each quantity -- a healthy configuration oscillates about
its initial value with ~zero trend; a leaking factor combination shows a
monotone trend long before any NaN.

usage:  python unit_drift.py <plotfile_dir> [label]
"""
import os, sys
import numpy as np
import yt

yt.funcs.mylog.setLevel(40)
T_DRIVE = 1.0 / 81.545     # drive period [s]

def metrics(pf):
    ds = yt.load(pf); ad = ds.all_data()
    x = np.array(ad["x"]); y = np.array(ad["y"])
    p = np.array(ad["pressure"]); vol = np.array(ad["index", "cell_volume"])
    m0 = float(np.sum(np.array(ad["rho_eta0"]) * vol))
    m1 = float(np.sum(np.array(ad["rho_eta1"]) * vol))
    E  = float(np.sum(np.array(ad["energy_per_vol"]) * vol))
    rho = np.array(ad["density"])
    vx = np.array(ad["velocityx"]); vy = np.array(ad["velocityy"])
    umax = float(np.max(np.sqrt(vx * vx + vy * vy)))
    # corner probes: nearest cell to each domain corner
    xl, xh = float(ds.domain_left_edge[0]), float(ds.domain_right_edge[0])
    yl, yh = float(ds.domain_left_edge[1]), float(ds.domain_right_edge[1])
    def pat(cx, cy):
        return float(p[np.argmin((x - cx) ** 2 + (y - cy) ** 2)])
    return dict(t=float(ds.current_time), m0=m0, m1=m1, E=E,
                p_mean=float(np.average(p, weights=vol)), umax=umax,
                p_ll=pat(xl, yl), p_hh=pat(xh, yh), p_hl=pat(xh, yl), p_lh=pat(xl, yh))

def main(d, label=""):
    def stepnum(f):
        try: return int(f.replace("cell", ""))
        except ValueError: return -1
    pfs = [os.path.join(d, f) for f in sorted(
        (f for f in os.listdir(d) if f.endswith("cell")), key=stepnum)]
    rows = []
    for pf in pfs:
        try:
            rows.append(metrics(pf))
        except Exception:
            continue
    if len(rows) < 2:
        print(f"[{label}] <2 usable frames"); return
    r0, rN = rows[0], rows[-1]
    dT = (rN["t"] - r0["t"]) / T_DRIVE   # elapsed drive periods
    print(f"\n[{label}] {d}  frames={len(rows)}  t: {r0['t']:.4e} -> {rN['t']:.4e}  ({dT:.2f} periods)")
    print(f"{'t[ms]':>8} {'m0drift%':>9} {'m1drift%':>9} {'Edrift%':>9} {'p_mean':>10} {'p_corner_ll':>11} {'p_corner_hh':>11} {'maxU':>8}")
    for r in rows:
        print(f"{r['t']*1e3:8.3f} {(r['m0']/r0['m0']-1)*100:9.4f} {((r['m1']/r0['m1']-1)*100 if r0['m1'] else 0.0):9.4f} "
              f"{(r['E']/r0['E']-1)*100:9.4f} {r['p_mean']:10.4e} {r['p_ll']:11.4e} {r['p_hh']:11.4e} {r['umax']:8.3f}")
    print(f"--- drift per drive period: m0 {100*(rN['m0']/r0['m0']-1)/max(dT,1e-9):+.4f}%  "
          f"m1 {(100*(rN['m1']/r0['m1']-1)/max(dT,1e-9) if r0['m1'] else 0.0):+.4f}%  "
          f"E {100*(rN['E']/r0['E']-1)/max(dT,1e-9):+.4f}%  "
          f"corner_ll-p_mean(final) {rN['p_ll']-rN['p_mean']:+.3e} Pa")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
