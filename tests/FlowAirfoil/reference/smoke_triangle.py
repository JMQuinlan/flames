"""
SMOKE TEST: does the embedded slip wall actually DEFLECT subsonic flow?

A simple triangular ramp whose top surface slopes DOWN in the flow direction.
If no-penetration is enforced, flow passing over the ramp must follow it and
turn DOWNWARD (v_y < 0).  This isolates a wall-deflection bug from the Kutta
issue: if even this simple ramp doesn't turn the flow down, the wall isn't
imposing the surface; if it DOES, then the airfoil's missing downwash is
specifically a circulation/Kutta problem.

Runs at M=0.5 (subsonic, the airfoil regime), slip wall.  Plots the vertical
velocity v_y + streamlines.  Writes Images/smoke_triangle_vy.png.
"""
import os, sys, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analyze_NACA_0012 as m
import yt

# downward-ramp triangle: top edge P1->P2 slopes DOWN (high-left to low-right)
P1 = (-0.6, 0.30); P2 = (0.6, -0.20); P3 = (-0.6, -0.20)
TRI = np.array([P1, P2, P3])

# config (subsonic, slip)
m.DOM_LO, m.DOM_HI = (-2.0, -2.0), (3.0, 2.0)
m.N_CELL = (96, 80)
m.MAX_LEVEL = 2
m.EPS = 0.04
m.BRINK = 4 * m.U / m.EPS        # m.U=0.5 (M=0.5)
m.SLIP = 1
m.STOP_TIME = 10.0               # ~1 flow-through (domain 5 wide / U 0.5)
m.PLOT_DT = 0
m.NX_BMP, m.NY_BMP = 500, 400
m.REFINE_BOX = None
WORK = "/tmp/smoke_tri"; os.makedirs(WORK, exist_ok=True)
rd = os.path.join(WORK, "run"); os.makedirs(rd, exist_ok=True)
bmp = os.path.join(rd, "phi.bmp"); plot = os.path.join(rd, "out"); inp = os.path.join(rd, "input")

print(f"ramp top slope = {np.degrees(np.arctan2(P1[1]-P2[1], P2[0]-P1[0])):.1f} deg downward, M={m.U}, slip wall", flush=True)
m.write_phi_bmp(TRI, bmp)
m.write_input(bmp, plot, inp)
env = dict(os.environ, OMP_NUM_THREADS="1")
with open(os.path.join(rd, "log"), "w") as lf:
    r = subprocess.run([os.path.abspath(m.SOLVER), os.path.abspath(inp)], cwd=rd, env=env,
                       stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
pfs = sorted(glob.glob(plot + "/*cell"))
if r.returncode != 0 or not pfs:
    print("SOLVER FAILED -- check", os.path.join(rd, "log")); sys.exit(1)

ds = yt.load(pfs[-1]); L = ds.index.max_level
dims = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=dims)
g = lambda f: np.asarray(cg[f])[:, :, 0]
phi = g("phi"); ux = g("velocityx"); uy = g("velocityy")
x = np.linspace(m.DOM_LO[0], m.DOM_HI[0], phi.shape[0]); y = np.linspace(m.DOM_LO[1], m.DOM_HI[1], phi.shape[1])
vy_fluid = np.where(phi > 0.5, uy, np.nan)
# diagnostic: mean v_y in a window just downstream/above the ramp
win = (np.abs(x[:, None] - 0.8) < 0.4) & (np.abs(y[None, :] - 0.1) < 0.3) & (phi > 0.5)
print(f"max |v_y| in fluid = {np.nanmax(np.abs(vy_fluid)):.4f}  (U={m.U})", flush=True)
print(f"mean v_y just over/behind ramp = {np.nanmean(np.where(win, uy, np.nan)):.4f}  (NEGATIVE = flow turned DOWN)", flush=True)

fig, ax = plt.subplots(figsize=(10, 6))
vmax = max(0.05, np.nanmax(np.abs(vy_fluid)))
im = ax.imshow(vy_fluid.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="RdBu_r",
               aspect="equal", vmin=-vmax, vmax=vmax)
plt.colorbar(im, ax=ax, label="v_y (blue = DOWN)")
ax.streamplot(x, y, ux.T, uy.T, density=2.2, color="k", linewidth=0.6, arrowsize=0.9)
ax.contour(x, y, phi.T, levels=[0.5], colors="lime", linewidths=2)
ax.set_xlim(-1.2, 2.0); ax.set_ylim(-1.0, 1.2); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.set_title(f"Smoke test: downward ramp, M={m.U}, slip wall — does flow turn DOWN over the ramp?")
out = os.path.join(m.IMAGES, "smoke_triangle_vy.png")
plt.tight_layout(); plt.savefig(out, dpi=120); print("wrote", out)
