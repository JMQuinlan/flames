#!/usr/bin/env python3
"""
Measure div_s(u) DIRECTLY in the band and compare with the exact 2 Rdot / R.

Why: Gamma obeys D(Gamma)/Dt = -Gamma div_s(u), whose exact spherical solution
is Gamma = (R0/R)^2 -- but only if div_s(u) = 2 Rdot/R.  The simulated Gamma
behaves like (R0/R)^n with n starting near 1, which implies div_s(u) is coming
out near Rdot/R.  This measures div_s(u) itself instead of inferring it.

Method: build a uniform covering grid at the finest level, then form exactly
what the solver forms,
    div_s u = tr(grad u) - n.grad(u).n ,   n = grad(eta)/|grad(eta)|
with the same central differences.  Rdot is read from the radial velocity at
the eta=0.5 surface, so the reference 2 Rdot/R needs no time differencing.

The decomposition is printed term by term, because for a bubble the two pieces
are individually LARGE and nearly cancel:
    inside  (compressible gas):  div u = 3 Rdot/R,  n.grad(u).n =  Rdot/R
    outside (incompressible)  :  div u = 0       ,  n.grad(u).n = -2 Rdot/R
both giving div_s u = 2 Rdot/R.  If the band smears them differently their
difference is wrong even when each looks plausible.
"""
import glob, math, re, sys
import numpy as np


def analyse(pf, R0, half=6.0e-6, verbose=True):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    lev = ds.index.max_level
    dx = float(ds.index.get_smallest_dx())
    n = int(round(half / dx))
    cg = ds.covering_grid(level=lev, left_edge=[0.0, 0.0, 0.0], dims=[n, n, n])
    eta = np.array(cg["eta"], float)
    ux = np.array(cg["velocityx"], float)
    uy = np.array(cg["velocityy"], float)
    uz = np.array(cg["velocityz"], float)

    g = lambda f, ax: np.gradient(f, dx, axis=ax, edge_order=2)
    ge = np.stack([g(eta, 0), g(eta, 1), g(eta, 2)])            # grad eta
    gem = np.sqrt((ge ** 2).sum(axis=0))
    small = 1.0e-30
    nh = ge / (gem + small)                                     # n_hat

    J = np.empty((3, 3) + eta.shape)                            # J[a,b] = d u_a / d x_b
    for a, u in enumerate((ux, uy, uz)):
        for b in range(3):
            J[a, b] = g(u, b)
    divu = J[0, 0] + J[1, 1] + J[2, 2]
    nJn = np.einsum("a...,ab...,b...->...", nh, J, nh)
    div_s = divu - nJn

    c = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    ur = (X * ux + Y * uy + Z * uz) / np.maximum(r, small)

    # eta = 0.5 surface: angle-averaged radial profile
    nb = 120
    ed = np.linspace(0.0, half, nb + 1)
    idx = np.clip(np.digitize(r.ravel(), ed) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    prof = lambda q: np.bincount(idx, weights=q.ravel(), minlength=nb) / np.maximum(cnt, 1)
    ep, urp, dsp, dup, nnp = (prof(eta), prof(ur), prof(div_s), prof(divu), prof(nJn))
    rc = 0.5 * (ed[:-1] + ed[1:])
    ok = cnt > 0
    R = Rd = np.nan
    for q in range(nb - 1):
        if ok[q] and ok[q + 1] and (ep[q] - 0.5) * (ep[q + 1] - 0.5) < 0:
            f = (0.5 - ep[q]) / (ep[q + 1] - ep[q])
            R = rc[q] + f * (rc[q + 1] - rc[q])
            Rd = urp[q] + f * (urp[q + 1] - urp[q])
            iR = q
            break
    else:
        return None
    exact = 2.0 * Rd / R
    meas = dsp[iR] + (0.5 - ep[iR]) / (ep[iR + 1] - ep[iR]) * (dsp[iR + 1] - dsp[iR])
    return dict(R=R, Rd=Rd, exact=exact, meas=meas,
                divu=dup[iR], nJn=nnp[iR], RR0=R / R0,
                err=100.0 * (meas - exact) / exact if exact else float("nan"))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "bin/tests/FlowMarmottant/gf_1"
    R0 = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0e-6
    pfs = sorted(glob.glob(d + "/*cell"),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    print(f"{'frame':>6} {'R/R0':>7} {'Rdot':>11} {'2Rdot/R exact':>14} "
          f"{'div_s u meas':>14} {'err %':>8} | {'div u':>12} {'n.gradu.n':>12}")
    for pf in pfs:
        try:
            o = analyse(pf, R0)
        except Exception as e:
            print(f"  {pf}: {e}"); continue
        if o is None:
            continue
        print(f"{re.search(r'(d+)cell',pf).group(1) if False else re.search(r'(\d+)cell',pf).group(1):>6} "
              f"{o['RR0']:7.4f} {o['Rd']:11.3e} {o['exact']:14.4e} {o['meas']:14.4e} "
              f"{o['err']:8.2f} | {o['divu']:12.4e} {o['nJn']:12.4e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
