#!/usr/bin/env python3
"""
Band-resolved gas bookkeeping for the coated Sch20 decks.

Measured on the Light runs (10_15_Comparison caches): Keller-Miksis driven by
the sim's own kappa_eff (0.81) and its own gas p(V_gas, t) reproduces R_min and
the rebound to 0.001 R0, and the gas term is the larger rebound error.  Its
signature is X = V_gas * rho_core / m_gas != 1, tracking Rdot (0.993 collapse,
1.007 rebound): the gas VOLUME (sum of alpha_g) leads the gas MASS.  This
script finds WHERE in the band that excess volume lives, and in which state.

Per frame, cells are binned by eta (liquid fraction).  For each bin:
  dV/Vg   gas volume beyond what the bin's gas mass would occupy at the core
          gas density, as a fraction of the total gas volume.  Sum over bins
          = X - 1.  A bin carrying dV/Vg != 0 holds gas at a different density
          than the core at (nearly) the same pressure.
  s/s0    mass-weighted p/rho_g^gamma relative to the t=0 core value.
  p-pc    bin-mean pressure minus core pressure (is the bin at the core p?).
  u/uQ    radial velocity over the incompressible liquid value Q/r^2,
          Q = median(u_r r^2) in the liquid just outside the band.  1 means the
          bin moves with the liquid volume flux; < 1 at eta=0.5 means the
          fluid lags the eta level sets.

Usage:
    python3 diag_band_gas.py PLOTFILE_DIR [--input ../input_Sch20-Oscillating_Marmottant_Light]
                             [--nproc 8] [--csv band.csv]
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_sch20_common import parse_input, _num, _expr_const  # noqa: E402

P = {}
EBINS = np.array([0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999, 1.0 + 1e-12])
NB = len(EBINS) - 1


def _init(params):
    P.update(params)


def frame(pf):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    R0, g = P["R0"], P["g1"]
    L = ds.index.max_level
    dle, dre = ds.domain_left_edge.d, ds.domain_right_edge.d
    dx = (dre[0] - dle[0]) / (ds.domain_dimensions[0] * 2 ** L)
    octant = [abs(dle[d]) < 1e-12 for d in range(3)]
    sym = 2 ** sum(octant)
    cen = np.array([0.0 if octant[d] else 0.5 * (dle[d] + dre[d]) for d in range(3)])
    h = 3.0 * R0
    lo = np.array([max(dle[d], cen[d] - h) for d in range(3)])
    n = np.round((np.array([min(dre[d], cen[d] + h) for d in range(3)]) - lo) / dx).astype(int)
    cg = ds.covering_grid(L, lo, n)
    f = lambda k: np.asarray(cg["boxlib", k], float)
    eta, p, arg = f("eta"), f("pressure"), f("rho_eta1")
    vx, vy, vz = f("velocityx"), f("velocityy"), f("velocityz")
    x = [lo[d] + (np.arange(n[d]) + 0.5) * dx - cen[d] for d in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z) + 1e-30
    ur = (vx * X + vy * Y + vz * Z) / r
    dV = dx ** 3 * sym

    nb = int(h / dx)
    edges = np.linspace(0, h, nb + 1); rc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    pe = np.bincount(idx, weights=eta.ravel(), minlength=nb) / np.maximum(cnt, 1)
    R05 = np.nan
    for i in range(nb - 1):
        if cnt[i] and cnt[i + 1] and (pe[i] - 0.5) * (pe[i + 1] - 0.5) < 0:
            R05 = rc[i] + (0.5 - pe[i]) * (rc[i + 1] - rc[i]) / (pe[i + 1] - pe[i]); break

    dom = r < 1.8 * R0                         # whole bubble + band, inside the covering box
    al = np.clip(eta, 0.0, 1.0); ag = 1.0 - al
    Vg = np.sum(ag[dom]) * dV
    mg = np.sum(arg[dom]) * dV
    pure = dom & (eta < 1e-6) & (r < 0.6 * R05)
    rho_c = np.sum(arg[pure]) / np.sum(ag[pure]) if pure.any() else np.nan
    p_c = float(np.mean(p[pure])) if pure.any() else np.nan
    rg = arg / np.maximum(ag, 1e-30)
    s = p / np.maximum(rg, 1e-30) ** g

    r_liq = R05 + 5 * P["eps"]
    liq = (r > r_liq) & (r < r_liq + 0.6 * R0)
    Q = float(np.median((ur * r * r)[liq])) if liq.any() else np.nan

    out = dict(t=float(ds.current_time), R05=R05, R_vol=(3 * Vg / (4 * np.pi)) ** (1 / 3),
               m_gas=mg, V_gas=Vg, rho_core=rho_c, p_core=p_c, Q=Q,
               X=Vg * rho_c / mg, s_core=float(np.sum((s * arg)[pure]) / np.sum(arg[pure])))
    for k in range(NB):
        b = dom & (eta >= EBINS[k]) & (eta < EBINS[k + 1])
        Vb = np.sum(ag[b]) * dV; mb = np.sum(arg[b]) * dV
        out["dV%d" % k] = (Vb - mb / rho_c) / Vg
        out["Vf%d" % k] = Vb / Vg
        out["s%d" % k] = float(np.sum((s * arg)[b]) / np.sum(arg[b])) if mb > 0 else np.nan
        out["dp%d" % k] = float(np.mean(p[b]) - p_c) if b.any() else np.nan
        out["u%d" % k] = float(np.mean((ur * r * r)[b]) / Q) if (b.any() and abs(Q) > 0) else np.nan
        out["rb%d" % k] = float(np.mean(r[b]) / R05) if b.any() else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotdir")
    ap.add_argument("--input", default=os.path.normpath(os.path.join(
        _HERE, "..", "input_Sch20-Oscillating_Marmottant_Light")))
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    kv = parse_input(a.input)
    P.update(R0=_expr_const(kv, "eta.ic", "R0") or _num(kv, "marmottant.R0"),
             eps=_expr_const(kv, "eta.ic", "epsilon"),
             g1=_num(kv, "eos1.gamma", 1.4), p_inf=_expr_const(kv, "pressure0.ic", "p_inf", 1e5))
    tau_c = 0.915 * P["R0"] * np.sqrt(1000.0 / P["p_inf"])
    pfs = sorted(d for d in glob.glob(os.path.join(a.plotdir, "*cell")) if os.path.isdir(d))
    if not pfs:
        raise SystemExit("no *cell plotfiles in %s" % a.plotdir)
    with Pool(a.nproc, initializer=_init, initargs=(dict(P),)) as pool:
        rows = sorted(pool.map(frame, pfs), key=lambda d: d["t"])
    if len({d["t"] for d in rows}) < len(rows):
        print("WARNING: repeated times -- this directory mixes runs.")
    r0 = rows[0]; s0 = r0["s_core"]; g = P["g1"]
    lab = ["[%g,%g)" % (EBINS[k], min(EBINS[k + 1], 1.0)) for k in range(NB)]

    print("\nWHOLE GAS  (X = V_gas rho_core / m_gas; s_vol = p_core V_gas^g normalised)")
    print(" t/tc   R05/R0  R_vol/R0  Rdot(Q/R^2)   X        s_core   s_vol    m_gas/m0")
    for d in rows:
        print("%5.2f  %.4f  %.4f   %8.4f   %.5f  %.5f  %.5f  %.7f" % (
            d["t"] / tau_c, d["R05"] / P["R0"], d["R_vol"] / P["R0"], d["Q"] / d["R05"] ** 2, d["X"],
            d["s_core"] / s0, d["p_core"] * d["V_gas"] ** g / (r0["p_core"] * r0["V_gas"] ** g), d["m_gas"] / r0["m_gas"]))
    for key, title, fmt in (("dV", "EXCESS GAS VOLUME dV/V_gas per eta bin (sum = X-1)  [x1e3]", "%8.3f"),
                            ("s", "GAS ENTROPY s/s0 (mass-weighted) per eta bin", "%8.4f"),
                            ("dp", "PRESSURE p - p_core [Pa] per eta bin", "%8.0f"),
                            ("u", "RADIAL VELOCITY u_r r^2 / Q per eta bin (1 = moves with liquid flux)", "%8.3f"),
                            ("Vf", "GAS VOLUME FRACTION of V_gas per eta bin", "%8.4f")):
        print("\n" + title)
        print("  t/tc " + "".join("%14s" % l for l in lab))
        for d in rows:
            sc = 1e3 if key == "dV" else (1.0 / s0 if key == "s" else 1.0)
            print("  %5.2f" % (d["t"] / tau_c) + "".join(
                ("      " + fmt % (d[key + "%d" % k] * sc)) if np.isfinite(d[key + "%d" % k]) else "%14s" % "--"
                for k in range(NB)))
    if a.csv:
        keys = list(rows[0].keys())
        with open(a.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for d in rows:
                fh.write(",".join(str(d[k]) for k in keys) + "\n")
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
