#!/usr/bin/env python3
"""
Interface pressure-jump budget for coated (or uncoated) spherical-bubble runs,
straight from plotfiles.  For each frame the fields are averaged on spherical
shells (octant or full domain) and three things are measured:

  APPLIED  the jump the shell stress the code actually built would support,
           int 2 T |grad eta| / r dr, with T = kappa1 (plotted total tension
           sigma(Gamma) + kappa_s div_s u ).  Reported as
           a fraction of the sharp 4 kappa_s Rdot/R^2 after removing the
           elastic 2 sigma/R part.            -> A_kap
  EFFECTIVE the jump the flow actually carries: core gas pressure minus the
           liquid pressure fit p = C0 + C1/r + C2/r^4 extrapolated to R,
           minus 2 sigma/R and the sharp liquid-viscous 4 mu Rdot/R.
                                               -> E_kap  (regression over frames)
  LAG      fluid radial velocity at eta = 0.5 over Rdot = Q/R^2.  < 1 means
           the band fluid lags the eta contours (band-gas compression).

R is the eta = 0.5 radius of the shell-averaged profile; Q = median(u_r r^2)
in the liquid just outside the band.  E_kap is fit as D = a G (instantaneous)
and D = a G + b dG/dt (lag = b/a), over frames with |Rdot| above --rdfrac of
its max.  Use -s key=value to override deck values given on the command
line of the run (e.g. -s shell.kappa_s=4e-9 -s mu0=2e-3 -s epsilon=2.5e-7).

Usage:
  diag_applied_jump.py PLOTDIR --input DECK [-s key=val ...] [--nproc 8]
                       [--tmin 0] [--csv out.csv]
"""
import argparse, glob, os, sys
from multiprocessing import Pool
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_sch20_common import parse_input, _num, _expr_const  # noqa: E402

P = {}


def _init(p):
    P.update(p)


def sigma_el(R):
    if not P["marm"]:
        return P["sigma"]
    Rb, chi, sb = P["Rb"], P["chi"], P["sbreak"]
    if R <= Rb:
        return 0.0
    s = chi * (R * R / (Rb * Rb) - 1.0)
    return s if s < sb else P["sigma"]


def frame(pf):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    R0, eps = P["R0"], P["eps"]
    L = ds.index.max_level
    dle, dre = ds.domain_left_edge.d, ds.domain_right_edge.d
    dx = (dre[0] - dle[0]) / (ds.domain_dimensions[0] * 2 ** L)
    octant = [abs(dle[d]) < 1e-12 for d in range(3)]
    cen = np.array([0.0 if octant[d] else 0.5 * (dle[d] + dre[d]) for d in range(3)])
    h = min(P["hfac"] * R0, min(dre[d] - cen[d] for d in range(3)))
    lo = np.array([max(dle[d], cen[d] - h) for d in range(3)])
    n = np.round((np.array([min(dre[d], cen[d] + h) for d in range(3)]) - lo) / dx).astype(int)
    cg = ds.covering_grid(L, lo, n)
    f = lambda k: np.asarray(cg["boxlib", k], float)
    eta, p = f("eta"), f("pressure")
    vx, vy, vz = f("velocityx"), f("velocityy"), f("velocityz")
    try:
        T = f("kappa1")
    except Exception:
        T = None
    x = [lo[d] + (np.arange(n[d]) + 0.5) * dx - cen[d] for d in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    ur = (vx * X + vy * Y + vz * Z) / r

    dr = 0.5 * dx
    nb = int(h / dr)
    edges = np.linspace(0, h, nb + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb).astype(float)
    ok = cnt > 0
    avg = lambda a: np.bincount(idx, weights=a.ravel(), minlength=nb)[ok] / cnt[ok]
    rr = rc[ok]
    e_r, p_r, u_r = avg(eta), avg(p), avg(ur)
    R05 = np.nan
    for i in range(len(rr) - 1):
        if (e_r[i] - 0.5) * (e_r[i + 1] - 0.5) <= 0 and e_r[i + 1] != e_r[i]:
            R05 = rr[i] + (0.5 - e_r[i]) * (rr[i + 1] - rr[i]) / (e_r[i + 1] - e_r[i])
            break
    out = dict(t=float(ds.current_time), R05=R05)
    if not np.isfinite(R05):
        return out
    # liquid flux and Rdot
    r1 = R05 + 5 * eps
    liq = (r > r1) & (r < r1 + 0.6 * R0)
    Q = float(np.median((ur * r * r)[liq])) if liq.any() else np.nan
    Rd = Q / R05 ** 2
    u5 = float(np.interp(R05, rr, u_r))
    # effective jump
    core = (eta < 1e-3) & (r < 0.5 * R05)
    if not core.any():
        core = r < 0.3 * R05
    pg = float(np.mean(p[core]))
    m = (rr > r1) & (rr < min(P["pfit"] * R0, rr[-1]))
    xx = R0 / rr[m]
    A = np.vstack([np.ones_like(xx), xx, xx ** 4]).T
    C, *_ = np.linalg.lstsq(A, p_r[m], rcond=None)
    pl = C[0] + C[1] * R0 / R05 + C[2] * (R0 / R05) ** 4
    out.update(Q=Q, Rdot=Rd, u5=u5, pg=pg, pl=pl, C1=C[1])
    # applied jump from the plotted total tension
    if T is not None:
        de = np.gradient(e_r, rr)
        T_r = avg(T)
        out["J_app"] = float(np.trapezoid(2.0 * T_r * de / rr, rr))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotdir")
    ap.add_argument("--input", required=True)
    ap.add_argument("-s", action="append", default=[], help="key=value deck override")
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--tmin", type=float, default=0.0, help="ignore frames before t/tc")
    ap.add_argument("--rdfrac", type=float, default=0.3)
    ap.add_argument("--hfac", type=float, default=6.0, help="half-box size in R0")
    ap.add_argument("--pfit", type=float, default=6.0, help="liquid fit out to this many R0")
    ap.add_argument("--rdot", choices=("level", "flux"), default="level",
                    help="Rdot from d(R05)/dt (default; the speed the shell should feel) or Q/R^2")
    ap.add_argument("--csv")
    a = ap.parse_args()
    kv = parse_input(a.input)
    for s in a.s:
        k, v = s.split("=", 1)
        kv[k.strip()] = v.strip()
        if k.strip() == "epsilon":
            kv["eta.ic.expression.constant.epsilon"] = v.strip()
    R0 = _expr_const(kv, "eta.ic", "R0") or _num(kv, "marmottant.R0")
    eps = _num(kv, "eta.ic.expression.constant.epsilon")
    pinf = _expr_const(kv, "pressure0.ic", "p_inf", 1e5)
    rho = _expr_const(kv, "density0.ic", "rho", None) or 1000.0
    P.update(R0=R0, eps=eps, marm=int(_num(kv, "marmottant", 0) or 0),
             Rb=_num(kv, "marmottant.R_buckling", R0), chi=_num(kv, "marmottant.chi", 0.0),
             sbreak=_num(kv, "marmottant.sigma_break", 1e30), sigma=_num(kv, "sigma", 0.0),
             hfac=a.hfac, pfit=a.pfit)
    kap = _num(kv, "shell.kappa_s", 0.0)
    mu = _num(kv, "mu0", 0.0)
    tc = 0.915 * R0 * np.sqrt(rho / pinf)
    pfs = sorted(d for d in glob.glob(os.path.join(a.plotdir, "*cell")) if os.path.isdir(d))
    if not pfs:
        raise SystemExit("no *cell plotfiles in %s" % a.plotdir)
    with Pool(a.nproc, initializer=_init, initargs=(dict(P),)) as pool:
        rows = [d for d in pool.map(frame, pfs) if "Q" in d]
    rows.sort(key=lambda d: d["t"])
    t = np.array([d["t"] for d in rows]); R = np.array([d["R05"] for d in rows])
    Rq = np.array([d["Rdot"] for d in rows]); u5 = np.array([d["u5"] for d in rows])
    Rl = np.gradient(R, t)
    Rd = Rl if a.rdot == "level" else Rq
    sig = np.array([sigma_el(x) for x in R])
    D = np.array([d["pg"] - d["pl"] for d in rows]) - 2 * sig / R - 4 * mu * Rd / R
    G = 4 * Rd / R ** 2                       # per unit kappa_s
    Japp = np.array([d.get("J_app", np.nan) for d in rows]) - 2 * sig / R
    print("kappa_s=%g mu0=%g eps=%g R0=%g eps/R0=%.3f tc=%.3e  frames=%d" % (kap, mu, eps, R0, eps / R0, tc, len(t)))
    print("  Rdot = %s" % ("dR05/dt" if a.rdot == "level" else "Q/R^2 (liquid flux)"))
    print("  t/tc   R05/R0   Rdot     Q/R2/Rdot  u5/Rdot  A_kap(applied)  E_kap(frame)")
    for i in range(len(t)):
        ak = Japp[i] / (kap * G[i]) if kap else np.nan
        ek = D[i] / (kap * G[i]) if kap else np.nan
        print("  %5.2f  %.4f  %+8.3f   %6.3f    %6.3f   %8.3f      %8.3f" % (t[i] / tc, R[i] / R0, Rd[i], Rq[i] / Rd[i], u5[i] / Rd[i], ak, ek))
    sel = (t >= a.tmin * tc) & (np.abs(Rd) > a.rdfrac * np.max(np.abs(Rd))) & np.isfinite(D)
    print("\nSUMMARY over %d frames with |Rdot| > %.0f%% max:" % (sel.sum(), 100 * a.rdfrac))
    print("  u(eta=.5)/Rdot         median %.3f" % np.median(u5[sel] / Rd[sel]))
    if kap:
        g = kap * G
        ainst = np.sum(D[sel] * g[sel]) / np.sum(g[sel] ** 2)
        dg = np.gradient(g, t)
        Xm = np.vstack([g[sel], dg[sel] * tc]).T
        c, *_ = np.linalg.lstsq(Xm, D[sel], rcond=None)
        okA = sel & np.isfinite(Japp)
        aapp = np.sum(Japp[okA] * g[okA]) / np.sum(g[okA] ** 2) if okA.any() else np.nan
        print("  A_kap applied (regr)   %.3f" % aapp)
        print("  E_kap effective (regr) %.3f   with lag term: a=%.3f lag=%+.3f tc" % (ainst, c[0], c[1] / c[0]))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "R05", "Rdot", "u5", "D", "J_app"])
            for i in range(len(t)):
                w.writerow([t[i], R[i], Rd[i], u5[i], D[i], Japp[i]])


if __name__ == "__main__":
    main()
