"""
Settling diagnostic: run ONE NACA0012 case (AoA=4, M=0.5, slip wall) for many
flow-throughs and record C_l(t) at each plotfile.  Tells us whether the lift is
(a) still climbing at the old stop_time (under-settled -> need longer, lift may
recover) or (b) plateaued at ~0 (settled -> Kutta/TE failure is real).

eps=0.02 (cheap-ish) is fine here -- settling is a flow timescale, not strongly
eps-dependent.  Domain 6 wide, U=0.5 -> one flow-through = 12 s.
Writes Images/settle_history.png.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_NACA_0012 as m
import yt

# config
m.DOM_LO, m.DOM_HI = (-2.0, -2.5), (4.0, 2.5)
m.N_CELL = (120, 96)
m.SLIP = 1
m.EPS = 0.02
m.BRINK = 4 * m.U / m.EPS
m.MAX_LEVEL = 3
m.NX_BMP, m.NY_BMP = 600, 500
m.STOP_TIME = 60.0            # 5 flow-throughs (flow-through = 12 s)
m.PLOT_DT = 6.0              # snapshot every 6 s -> ~10 points
m.WORK = "/tmp/naca_settle"
AOA = 4.0
import math
Cl_theory = 2 * math.pi / math.sqrt(1 - m.MACH ** 2) * math.radians(AOA)

print(f"settling run: stop_time={m.STOP_TIME} (= {m.STOP_TIME/12:.1f} flow-throughs), plot every {m.PLOT_DT}s", flush=True)
r = m.run_case(AOA, tag="settle")
if r is None:
    print("FAILED -- check /tmp/naca_settle/settle/log"); sys.exit(1)

def cl_at(pf):
    ds = yt.load(pf); L = ds.index.max_level
    dims = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
    cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=dims)
    dx = np.asarray(ds.domain_width) / dims
    phi = np.asarray(cg["phi"])[:, :, 0]; p = np.asarray(cg["pressure"])[:, :, 0]
    gy = np.gradient(phi, float(dx[1]), axis=1)
    Fy = -np.sum((p - m.P_INF) * gy) * float(dx[0]) * float(dx[1])
    return float(ds.current_time), Fy / (m.Q_INF * m.CHORD)

pfs = sorted(glob.glob("/tmp/naca_settle/settle/out/*cell"))
hist = [cl_at(pf) for pf in pfs]
print("Cl(t):")
for t, cl in hist:
    print(f"  t={t:6.2f} ({t/12:.2f} flow-through)  Cl={cl:+.4f}  (theory {Cl_theory:.3f})", flush=True)

T = [h[0] for h in hist]; CL = [h[1] for h in hist]
plt.figure(figsize=(8, 5))
plt.axhline(Cl_theory, color="r", ls="--", label=f"thin-airfoil {Cl_theory:.2f}")
plt.plot(T, CL, "o-", label="CFD Cl(t)")
plt.xlabel("time (s)   [flow-through = 12 s]"); plt.ylabel("Cl")
plt.title(f"NACA0012 AoA={AOA} M={m.MACH} slip: lift vs time (is it settled?)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(m.IMAGES, "settle_history.png"), dpi=120)
print("wrote Images/settle_history.png")
