"""Dump 2D fields, radial profiles and AMR boxes for one elastic Laplace case."""
import os, numpy as np, yt
yt.set_log_level(50)
CASE = "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant/lref20/R001025000"
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
FIELDS = ("eta", "cfun", "shell", "pressure", "velocityx", "velocityy", "kappa2", "vorticity")
for tag, pf in (("early", "36340cell"), ("final", "726780cell")):
    ds = yt.load(os.path.join(CASE, pf))
    lev = ds.index.max_level
    dims = ds.domain_dimensions * 2**lev
    cg = ds.covering_grid(lev, ds.domain_left_edge, dims, fields=[("boxlib", f) for f in FIELDS])
    d = {f: np.asarray(cg["boxlib", f])[:, :, 0] for f in FIELDS}
    boxes = []
    for g in ds.index.grids:
        le, re_ = g.LeftEdge.d, g.RightEdge.d
        boxes.append((g.Level, le[0], le[1], re_[0], re_[1]))
    np.savez_compressed(os.path.join(OUT, f"fields_{tag}.npz"), t=float(ds.current_time),
                        lo=ds.domain_left_edge.d[:2], hi=ds.domain_right_edge.d[:2],
                        boxes=np.array(boxes), **d)
    print(tag, float(ds.current_time), dims, len(boxes))
