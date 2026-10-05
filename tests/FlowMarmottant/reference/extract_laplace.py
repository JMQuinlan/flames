"""Extract Laplace-sweep diagnostics from the lref20 plotfiles.

Writes data/laplace_<case>.csv with one row per plotfile:
t, R_vol, Gamma_int, sigma_int, sigma_law, p_in, p_out, sigma_laplace, umax
"""
import glob, os, re, sys
import numpy as np
from multiprocessing import Pool
import yt
yt.set_log_level(50)

ROOT = os.environ.get("LAPLACE_ROOT",
       "/home/ttryon/Desktop/flames2/bin/tests/FlowMarmottant/lref20")
TAG = os.environ.get("LAPLACE_TAG", "")
GLOB = os.environ.get("LAPLACE_GLOB", "R*/")
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
CHI, RB, SBRK, SIGW = 0.55, 1.0e-3, 0.073, 0.073

def law(G, Gb):
    if G <= 0 or G >= Gb: return 0.0
    el = CHI * (Gb / G - 1.0)
    return SIGW if el >= SBRK else el

def one(pf):
    ds = yt.load(pf)
    lev = ds.index.max_level
    dims = ds.domain_dimensions * 2**lev
    cg = ds.covering_grid(lev, ds.domain_left_edge, dims,
                          fields=[("boxlib", f) for f in
                                  ("eta", "shell", "pressure", "velocityx", "velocityy", "kappa2")])
    eta = np.asarray(cg["boxlib", "eta"])[:, :, 0]
    sh  = np.asarray(cg["boxlib", "shell"])[:, :, 0]
    p   = np.asarray(cg["boxlib", "pressure"])[:, :, 0]
    u   = np.asarray(cg["boxlib", "velocityx"])[:, :, 0]
    v   = np.asarray(cg["boxlib", "velocityy"])[:, :, 0]
    se  = np.asarray(cg["boxlib", "kappa2"])[:, :, 0]
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]) / dims[0])
    x = float(ds.domain_left_edge[0]) + (np.arange(dims[0]) + 0.5) * dx
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    Rv = np.sqrt(np.sum(1.0 - eta) * dx * dx / np.pi)
    w = eta * (1.0 - eta)
    Gi = np.sum(w * sh) / np.sum(w)
    si = np.sum(w * se) / np.sum(w)
    p_in = p[eta < 1e-3].mean()
    ring = (r > 1.5 * Rv) & (r < 2.0 * Rv)
    p_out = p[ring].mean()
    return (float(ds.current_time), Rv, Gi, si, p_in, p_out,
            float(np.sqrt(u * u + v * v).max()))

def case(d):
    name = TAG + os.path.basename(d.rstrip("/"))
    R0 = float([l.split("=")[1].split()[0] for l in open(os.path.join(d, "metadata"))
                if l.startswith("marmottant.R0")][0])
    Gb = (R0 / RB) ** 2
    pfs = sorted(glob.glob(os.path.join(d, "*cell")),
                 key=lambda s: int(re.sub(r"\D", "", os.path.basename(s))))
    pfs = [p for p in pfs if os.path.exists(os.path.join(p, "Header"))]
    rows = [one(p) for p in pfs]
    with open(os.path.join(OUT, f"laplace_{name}.csv"), "w") as f:
        f.write(f"# R0={R0:.6e} Gamma_b={Gb:.6f}\n")
        f.write("t,R_vol,Gamma_int,sigma_int,sigma_law,p_in,p_out,sigma_laplace,umax\n")
        for t, Rv, Gi, si, pi, po, um in rows:
            f.write(f"{t:.6e},{Rv:.8e},{Gi:.8f},{si:.8e},{law(Gi, Gb):.8e},"
                    f"{pi:.6f},{po:.6f},{(pi - po) * Rv:.8e},{um:.6e}\n")
    return name, len(rows)

if __name__ == "__main__":
    dirs = sorted(glob.glob(os.path.join(ROOT, GLOB)))
    with Pool(min(10, len(dirs))) as pool:
        for n, k in pool.imap_unordered(case, dirs):
            print(n, k, flush=True)
