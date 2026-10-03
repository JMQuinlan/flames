#!/usr/bin/env python3
"""
Independent lift/drag from a MOMENTUM-BALANCE control volume around the airfoil,
using only the LAST plotfile of a (steady) case -- no use of the solver's solid-cell
force diagnostic (rhs_force):

  python3 cv_force_check.py <case_dir> [h | xa,xb,ya,yb ...]     (LEVEL=5 env for boxes in the refine_box)

For a steady flow the force ON the body is
  F_i = -oint_S [ rho (u_i - U_i) (u.n) + p n_i - tau_ij n_j ] dS   (U = freestream; the U term
  vanishes for exact mass conservation and removes sensitivity to a net mass flux)
over any contour S enclosing it (tau = mu (grad u + grad u^T) - 2/3 mu div(u) I).
Several nested boxes are evaluated; they agree when the flow is steady and the
contour is resolved.  Cl = Fy/q, Cd = Fx/q (freestream along +x, body rotated by alpha),
q = 0.5 rho U^2 c with rho = 1, U = Mach, c = 1.
"""
import os, re, sys, glob
import numpy as np


def case_info(d):
    txt = open(os.path.join(d, "input"), errors="ignore").read()
    m = re.search(r"M = ([0-9.eE+-]+), Re = ([0-9.eE+-]+), AoA = ([0-9.eE+-]+) deg", txt)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    import yt
    yt.set_log_level(40)
    d = os.path.abspath(sys.argv[1].rstrip("/"))
    # boxes: half-size h (-> x [-0.5-h, 1.1+h], y [-h, h]) or explicit "xa,xb,ya,yb"
    boxes = sys.argv[2:] or ["0.8", "1.2", "1.6"]
    boxes = [tuple(float(t) for t in b.split(",")) if "," in b else
             (-0.5 - float(b), 0.5 + float(b) + 0.6, -float(b), float(b)) for b in boxes]
    M, Re, aoa = case_info(d)
    mu = M / Re
    pfs = [p for p in glob.glob(os.path.join(d, "out", "*cell")) if ".old." not in p]
    pf = max(pfs, key=lambda p: int(re.search(r"(\d+)cell$", p).group(1)))
    ds = yt.load(pf)
    lev = int(os.environ.get("LEVEL", 3))             # 3 covers +-2 c in these decks; use 5 for boxes inside the refine_box
    dxl = (ds.domain_width / (ds.domain_dimensions * ds.refine_by**lev)).to_value()
    le = ds.domain_left_edge.to_value()
    lo = np.array([min(b[0] for b in boxes), min(b[2] for b in boxes)]) - 0.05
    hi = np.array([max(b[1] for b in boxes), max(b[3] for b in boxes)]) + 0.05
    i0 = np.floor((lo - le[:2]) / dxl[:2]).astype(int); i1 = np.ceil((hi - le[:2]) / dxl[:2]).astype(int)
    left = [le[0] + i0[0] * dxl[0], le[1] + i0[1] * dxl[1], le[2]]
    dims = [i1[0] - i0[0], i1[1] - i0[1], 1]
    try:
        cg = ds.smoothed_covering_grid(level=lev, left_edge=left, dims=dims)
        g = {f: np.asarray(cg[("boxlib", f)])[:, :, 0] for f in ("density", "velocityx", "velocityy", "pressure", "phi")}
    except RuntimeError:
        ds.force_periodicity()
        cg = ds.smoothed_covering_grid(level=lev, left_edge=left, dims=dims)
        g = {f: np.asarray(cg[("boxlib", f)])[:, :, 0] for f in ("density", "velocityx", "velocityy", "pressure", "phi")}
    x = left[0] + (np.arange(dims[0]) + 0.5) * dxl[0]; y = left[1] + (np.arange(dims[1]) + 0.5) * dxl[1]
    rho, u, v, p = g["density"], g["velocityx"], g["velocityy"], g["pressure"]
    ux, uy = np.gradient(u, dxl[0], dxl[1]); vx, vy = np.gradient(v, dxl[0], dxl[1])
    div = ux + vy
    txx = mu * (2 * ux - 2.0 / 3.0 * div); tyy = mu * (2 * vy - 2.0 / 3.0 * div); txy = mu * (uy + vx)
    q = 0.5 * M * M
    t = float(ds.current_time)
    print(f"{d}\n  plotfile {os.path.basename(pf)}  t = {t:.2f} (t U/c = {t*M:.2f}), level {lev} dx = {dxl[0]:.5f}")
    for xa, xb, ya, yb in boxes:
        ia, ib = np.searchsorted(x, xa), np.searchsorted(x, xb)
        ja, jb = np.searchsorted(y, ya), np.searchsorted(y, yb)
        Fx = Fy = 0.0; mdot = 0.0
        # right face (n = +x) and left face (n = -x): integrate over y
        for i, s in ((ib, 1.0), (ia, -1.0)):
            sl = (i, slice(ja, jb))
            un = s * u[sl]
            fx = rho[sl] * (u[sl] - M) * un + s * p[sl] - s * txx[sl]      # (u - U_inf): insensitive to net mass flux
            fy = rho[sl] * v[sl] * un - s * txy[sl]
            Fx -= np.sum(fx) * dxl[1]; Fy -= np.sum(fy) * dxl[1]; mdot += np.sum(rho[sl] * un) * dxl[1]
        # top face (n = +y) and bottom face (n = -y): integrate over x
        for j, s in ((jb, 1.0), (ja, -1.0)):
            sl = (slice(ia, ib), j)
            un = s * v[sl]
            fx = rho[sl] * (u[sl] - M) * un - s * txy[sl]
            fy = rho[sl] * v[sl] * un + s * p[sl] - s * tyy[sl]
            Fx -= np.sum(fx) * dxl[0]; Fy -= np.sum(fy) * dxl[0]; mdot += np.sum(rho[sl] * un) * dxl[0]
        # the sweep decks keep the freestream along +x and rotate the BODY by alpha
        # (TEMPLATES/make_naca0012_bmp.py), so x = drag and y = lift directly
        cl = Fy / q
        cd = Fx / q
        print(f"  box x [{xa:+.2f}, {xb:+.2f}] y [{ya:+.2f}, {yb:+.2f}]:  Cl = {cl:.4f}  Cd = {cd:.4f}"
              f"   net mass flux {mdot:+.2e} (x U/q = {mdot*M/q:+.4f} in Cd)")


if __name__ == "__main__":
    main()
