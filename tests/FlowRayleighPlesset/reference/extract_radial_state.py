#!/usr/bin/env python3
"""Dump radial profiles of the full state near the collapse, to small CSVs.

For each plotfile whose time is within the window, bin ALL leaf cells by radius
r=sqrt(x^2+y^2+z^2) and write the per-bin mean of every diagnostic field plus the
global gas/liquid mass and the volume radius.  This preserves everything needed to
diagnose "why R_vol misses / does the interface smear / is mass conserved" AFTER the
raw plotfiles are deleted.

Usage:
    python extract_radial_state.py <output_dir> [t_lo_s] [t_hi_s] [nbins]
        default window 1.4e-3 .. 2.2e-3 s (around Sch20 case-1 t_c=1.83e-3), 200 bins
Outputs: radial_state_<frame>.csv  +  radial_state_summary.csv  (in cwd)
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)
try:
    from diag_config import resolve_dir
except Exception:
    def resolve_dir(a): return a[1] if len(a) > 1 else "."

D    = resolve_dir(sys.argv)
T_LO = float(sys.argv[2]) if len(sys.argv) > 2 else 1.4e-3
T_HI = float(sys.argv[3]) if len(sys.argv) > 3 else 2.2e-3
NB   = int(sys.argv[4])   if len(sys.argv) > 4 else 200

FIELDS = ["eta", "rho_eta0", "rho_eta1", "pressure", "density"]  # add/remove as available
def cells(d): return sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))

summary = []
for pf in cells(D):
    ds = yt.load(pf); t = float(ds.current_time)
    if not (T_LO <= t <= T_HI):
        continue
    dle = ds.domain_left_edge
    sym = 1
    for d in range(3):
        if float(dle[d]) > -1e-6: sym *= 2
    ad = ds.all_data()
    x = np.array(ad["x"]); y = np.array(ad["y"]); z = np.array(ad["z"])
    r = np.sqrt(x*x + y*y + z*z)
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    avail = [f for f in FIELDS if ("boxlib", f) in ds.field_list or f in [n for _, n in ds.field_list]]
    data = {f: np.array(ad[f]) for f in avail}
    # velocity magnitude if present
    try:
        data["umag"] = np.sqrt(sum(np.array(ad[c])**2 for c in ("velocity_x","velocity_y","velocity_z")))
        avail = avail + ["umag"]
    except Exception:
        pass
    # global masses + volume radius
    mg = float(np.sum(data.get("rho_eta1", np.zeros_like(r)) * vol)) * sym
    ml = float(np.sum(data.get("rho_eta0", np.zeros_like(r)) * vol)) * sym
    ag = np.clip(1.0 - data["eta"], 0, 1) if "eta" in data else np.zeros_like(r)
    Vg = float(np.sum(ag * vol)) * sym
    Rvol = (3.0*Vg/(4.0*np.pi))**(1.0/3.0) if Vg > 0 else 0.0
    summary.append((t, mg, ml, Rvol))
    # radial binning
    rmax = min(r.max(), 0.1)
    edges = np.linspace(0.0, rmax, NB+1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, NB-1)
    rc = 0.5*(edges[:-1] + edges[1:])
    out = {"r": rc, "count": np.bincount(idx, minlength=NB).astype(float)}
    for f in avail:
        s = np.bincount(idx, weights=data[f], minlength=NB)
        out[f] = s / np.maximum(out["count"], 1.0)
    frame = os.path.basename(pf).replace("cell", "")
    fn = f"radial_state_{frame}.csv"
    cols = ["r", "count"] + avail
    with open(fn, "w") as fh:
        fh.write("# t=%.6e s\n" % t)
        fh.write(",".join(cols) + "\n")
        for i in range(NB):
            fh.write(",".join("%.6e" % out[c][i] for c in cols) + "\n")
    print(f"  wrote {fn}  (t={t:.4e}s, Rvol={Rvol:.4e}, m_gas={mg:.4e})")

with open("radial_state_summary.csv", "w") as fh:
    fh.write("time_s,m_gas,m_liq,R_vol_m\n")
    for t, mg, ml, Rv in summary:
        fh.write("%.6e,%.6e,%.6e,%.6e\n" % (t, mg, ml, Rv))
print(f"\nwrote radial_state_summary.csv ({len(summary)} frames in [{T_LO:.1e},{T_HI:.1e}]s)")
print("Send me: radial_state_summary.csv + the radial_state_*.csv files.")
