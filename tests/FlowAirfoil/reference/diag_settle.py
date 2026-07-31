"""Diagnostic: is the ~zero subsonic lift due to under-settling or a Kutta failure?
Run AoA=4 much longer and report Cl(t) at each plotfile.  If Cl stays ~0 ->
circulation never develops (Kutta/diffuse-TE failure), not settling."""
import sys, os, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_NACA_0012 as m
import yt

m.STOP_TIME = 48.0          # ~2 flow-throughs (was 16)
m.WORK = "/tmp/naca_diag"
print("running AoA=4 to t=48 ...")
m.run_case(4)

def cl_at(pf):
    ds = yt.load(pf); L = ds.index.max_level
    dims = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
    cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=dims)
    dx = np.asarray(ds.domain_width) / dims
    phi = np.asarray(cg["phi"])[:, :, 0]; p = np.asarray(cg["pressure"])[:, :, 0]
    gy = np.gradient(phi, float(dx[1]), axis=1)
    Fy = -np.sum((p - m.P_INF) * gy) * float(dx[0]) * float(dx[1])
    return Fy / (m.Q_INF * m.CHORD)

print("Cl history (should climb toward 0.51 if it was just settling):")
for pf in sorted(glob.glob("/tmp/naca_diag/a004.0/out/*cell")):
    print(f"  {os.path.basename(pf):>12s}  Cl={cl_at(pf):+.4f}   (theory 0.51)")
