"""
Live monitor for the overnight study.  Watches every case's output dir and:
  - on the FIRST plotfile: writes Images/initial_refine_<tag>.png (solid + AMR grid boxes)
  - on each NEW plotfile:  refreshes Images/flowfield_<tag>.png (pressure/Mach + streamlines)
Runs until overnight_study.py exits (then one final pass).  Single-threaded (yt-safe).
"""
import os, sys, glob, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, yt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "Images"); os.makedirs(IMAGES, exist_ok=True)
WORK = "/tmp/naca_overnight"
FLOW = os.path.join(HERE, "plot_flowfield.py")

def make_refine(pf, tag):
    ds = yt.load(pf); L = ds.index.max_level
    dims = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
    cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=dims)
    phi = np.asarray(cg["phi"])[:, :, 0]
    lo = ds.domain_left_edge; hi = ds.domain_right_edge
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(phi.T, origin="lower", extent=[float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])],
              cmap="gray", aspect="equal")
    colors = ["red", "orange", "yellow", "cyan", "lime", "magenta"]
    for g in ds.index.grids:
        le = g.LeftEdge; re = g.RightEdge; lev = int(g.Level)
        ax.add_patch(Rectangle((float(le[0]), float(le[1])), float(re[0]-le[0]), float(re[1]-le[1]),
                               fill=False, edgecolor=colors[lev % len(colors)], lw=0.7))
    ax.set_xlim(-1.0, 1.5); ax.set_ylim(-0.8, 0.8)
    ax.set_title(f"{tag}: solid + AMR refinement boxes (max_level={L}; colors = level)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout(); plt.savefig(os.path.join(IMAGES, f"initial_refine_{tag}.png"), dpi=120); plt.close()

seen = {}; refined = set()
print("monitor started", flush=True)
while True:
    alive = subprocess.run(["pgrep", "-f", "[o]vernight_study"], capture_output=True).returncode == 0
    for cdir in sorted(glob.glob(os.path.join(WORK, "*/"))):
        tag = os.path.basename(cdir.rstrip("/"))
        pfs = sorted(glob.glob(os.path.join(cdir, "out", "*cell")))
        if not pfs:
            continue
        if tag not in refined:
            try:
                make_refine(pfs[0], tag); refined.add(tag); print(f"  refine plot: {tag}", flush=True)
            except Exception as e:
                pass  # plotfile may be mid-write; retry next cycle
        if seen.get(tag) != pfs[-1]:
            rc = os.system(f"python3 {FLOW} {os.path.join(cdir,'out')} {tag} >/dev/null 2>&1")
            if rc == 0:
                seen[tag] = pfs[-1]; print(f"  flowfield refreshed: {tag} ({os.path.basename(pfs[-1])})", flush=True)
    if not alive:
        print("study ended; final monitor pass done.", flush=True); break
    time.sleep(25)
print("monitor done", flush=True)
