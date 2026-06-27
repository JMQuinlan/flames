#!/usr/bin/env python3
# Direction/orientation verification WITHOUT running the solver: read the rotated
# STL, apply the input's scale+center (vertex -> vertex*scale + center, exactly
# what AMReX read_stl_file does), and draw the placed airframe silhouette (top +
# side views) inside the simulation domain with the +x freestream arrows.  This
# shows the plane is nose-first into a -x -> +x flow.
#   usage: verify_orientation.py <stl> <scale> <cx> <cy> <cz> <domlo x,y,z> <domhi x,y,z> <out.png> "<title>"
import struct, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

def read_tris(fn):
    d = open(fn, 'rb').read()
    n = struct.unpack('<I', d[80:84])[0]
    v = np.zeros((n, 3, 3)); off = 84
    for i in range(n):
        f = struct.unpack('<12fH', d[off:off+50]); off += 50
        v[i] = np.array(f[3:12]).reshape(3, 3)
    return v

stl, scale = sys.argv[1], float(sys.argv[2])
center = np.array([float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])])
lo = np.array([float(x) for x in sys.argv[6].split(',')])
hi = np.array([float(x) for x in sys.argv[7].split(',')])
out, title = sys.argv[8], sys.argv[9]

V = read_tris(stl) * scale + center            # placed in domain coords (m)
print(f"{title}: placed bbox x[{V[...,0].min():.1f},{V[...,0].max():.1f}] "
      f"y[{V[...,1].min():.1f},{V[...,1].max():.1f}] z[{V[...,2].min():.1f},{V[...,2].max():.1f}]")

fig, axs = plt.subplots(2, 1, figsize=(11, 7.5))
for ax, (a, b, alab, blab, dlo, dhi) in zip(axs, [
        (0, 1, "x (m)", "y (m)", (lo[0], lo[1]), (hi[0], hi[1])),   # top  view x-y
        (0, 2, "x (m)", "z (m)", (lo[0], lo[2]), (hi[0], hi[2]))]): # side view x-z
    polys = V[:, :, [a, b]]
    ax.add_collection(PolyCollection(polys, facecolors="0.25", edgecolors="none", alpha=0.5))
    ax.set_xlim(dlo[0], dhi[0]); ax.set_ylim(dlo[1], dhi[1])
    # freestream +x arrows
    ys = np.linspace(dlo[1], dhi[1], 5)[1:-1]
    for yy in ys:
        ax.annotate("", xy=(dlo[0] + 0.16*(dhi[0]-dlo[0]), yy), xytext=(dlo[0] + 0.02*(dhi[0]-dlo[0]), yy),
                    arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2))
    ax.text(dlo[0] + 0.02*(dhi[0]-dlo[0]), dhi[1]*0.86, "freestream  -x -> +x", color="green", weight="bold")
    ax.set_xlabel(alab); ax.set_ylabel(blab); ax.set_aspect("equal", adjustable="box")
axs[0].set_title(f"{title}   TOP view (x-y): airframe placed in domain")
axs[1].set_title("SIDE view (x-z)")
plt.tight_layout(); plt.savefig(out, dpi=115); print(f"  wrote {out}")
