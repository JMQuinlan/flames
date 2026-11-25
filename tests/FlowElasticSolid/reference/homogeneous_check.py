#!/usr/bin/env python3
"""
Homogeneous finite deformation (tests/FlowElasticSolid/input_Homogeneous).

  python3 homogeneous_check.py <run_dir> a b g

Pre-strain X = a x - g y, Y = b y: cobasis e^1 = (a, -g, 0), e^2 = (0, b, 0), e^3 = (0, 0, 1).
Reference (independent numpy evaluation of Favrie et al. 2009, eq. 12):
    G = sum_b e^b (x) e^b,  Gt = G/det(G)^(1/3),  S = -mu (rho/rho0) dev(Gt^2 - Gt),
    W = (rho/rho0)(mu/4) tr((Gt - I)^2),  rho/rho0 = det(e)
For simple shear (a = b = 1) the closed form is also printed.
Checks the first and the last plot file: the state must be steady.
"""
import sys, glob, os
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)
MU = 12.0


def exact(a, b, g):
    e = np.array([[a, -g, 0.0], [0.0, b, 0.0], [0.0, 0.0, 1.0]])
    G = e.T @ e
    Gt = G / np.linalg.det(G)**(1.0 / 3.0)
    D = Gt @ Gt - Gt
    S = -MU * np.linalg.det(e) * (D - np.trace(D) / 3 * np.eye(3))
    return S, np.linalg.det(e) * MU / 4 * np.trace((Gt - np.eye(3)) @ (Gt - np.eye(3)))


def main():
    d = sys.argv[1]; a, b, g = (float(v) for v in sys.argv[2:5])
    S, W = exact(a, b, g)
    ref = {"elastic_Sxx": S[0, 0], "elastic_Syy": S[1, 1], "elastic_Szz": S[2, 2], "elastic_Sxy": S[0, 1], "elastic_W": W}
    pfs = sorted(glob.glob(os.path.join(d, "output", "*cell")))
    print(f"{d}: a = {a:g}, b = {b:g}, g = {g:g}")
    if a == 1 and b == 1:
        print(f"  closed form (simple shear): S_xy {MU * (g + g**3):.6f}  S_xx {MU * g**4 / 3:.6f}  S_yy {-MU * (g**2 + 2 * g**4 / 3):.6f}"
              f"  S_zz {MU * (g**2 + g**4 / 3):.6f}  W {MU * (2 * g**2 + g**4) / 4:.6f}")
    for pf in (pfs[1], pfs[-1]):          # pfs[0] is written before the first stress evaluation
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
        u = np.hypot(q("velocityx"), q("velocityy")).max()
        print(f"  t = {float(ds.current_time):.3f}: max |u| = {u:.2e}, pressure {q('pressure').min():.6f} .. {q('pressure').max():.6f}")
        for k, r in ref.items():
            v = q(k)
            print(f"    {k[8:]:4s} {v.mean():13.8f}   exact {r:13.8f}   error {abs(v.mean() - r):.1e}   spread {v.max() - v.min():.1e}")


if __name__ == "__main__":
    main()
