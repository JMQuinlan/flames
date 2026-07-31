import sys, os, re
import numpy as np
import yt
yt.funcs.mylog.setLevel(50)

out = sys.argv[1] if len(sys.argv) > 1 else "../output_hydro2"
pfs = sorted([os.path.join(out, d) for d in os.listdir(out)
              if d.endswith("cell") and os.path.isdir(os.path.join(out, d))],
             key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
ds = yt.load(pfs[-1])
print("checking", pfs[-1], " t=", float(ds.current_time))

# vertical ray at x=0 from y=0 to y=1.5 (crosses cylinder top, phi=0.5 at y~0.5)
ray = ds.ray(ds.arr([0.0, 0.0, 0.0], "code_length"), ds.arr([0.0, 1.5, 0.0], "code_length"))
y = np.array(ray["y"]); vx = np.array(ray["velocityx"]); phi = np.array(ray["phi"])
o = np.argsort(y)
y, vx, phi = y[o], vx[o], phi[o]
print(" y      phi      vx")
for yi, pi, vi in zip(y, phi, vx):
    print(f"  {yi:6.3f}  {pi:6.3f}  {vi:8.4f}")
# where does phi cross 0.5 and where does vx first become nonzero?
i_phi = np.argmin(np.abs(phi - 0.5))
print(f"\nphi=0.5 at y ~ {y[i_phi]:.3f} (cylinder surface, expect ~0.5)")
noslip_violation = float(np.max(np.abs(vx[phi < 0.5]))) if np.any(phi < 0.5) else 0.0
print(f"max|vx| where phi<0.5 (inside solid): {noslip_violation:.4e}  (expect ~0 = no-slip)")
# Standardized line the aggregator parses (the no-slip violation IS the error).
print(f"UNIT_TEST_RESULT max_err={noslip_violation:.4e} avg_err={noslip_violation:.4e}")
