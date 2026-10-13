#!/usr/bin/env python3
"""
Gas-core thermodynamics diagnostic for the coated Sch20 decks.

The core gas of the Light run follows p ~ rho^1.446 instead of the adiabat
rho^1.4 (diag_shell_tension.py: s = p/rho_g^gamma rises 1.2% at R_min and
mostly returns), while the soft-liquid unit deck gives 1.404.  This script
locates where in the core the excess lives:

  * by TRACE-LIQUID fraction.  The stiffened liquid carries
    alpha_l gamma pi/(gamma-1) of internal energy; at pi = 1e9 that equals the
    gas's own p/(gamma-1) already at alpha_l ~ 1e-4, so a tiny error in the
    trace liquid moves the reinitialised pressure a lot.  If s grows with
    alpha_l, the stiffness is a trace-phase artefact.
  * by RADIUS.  A centre-to-edge trend points at the band (heat or mass
    exchange through the interface) or at an acoustic mode in the gas.
  * gas phasic pressure from its OWN energy, p_g = (gamma_g - 1) E1/alpha_g,
    against the mixture pressure, to separate the gas state from the
    reinitialised mixture value.

Usage:
    python3 diag_gas_core.py PLOTFILE_DIR [--input ../input_Sch20-Oscillating_Marmottant_Light]
                             [--nproc 8] [--csv core.csv]
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
ABINS = np.array([0.0, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2])   # trace-liquid bins
RBINS = np.array([0.0, 0.2, 0.4, 0.6, 0.7, 0.8])                           # r / R05 bins


def _init(params):
    P.update(params)


def frame(pf):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    R0 = P["R0"]
    L = ds.index.max_level
    dle, dre = ds.domain_left_edge.d, ds.domain_right_edge.d
    dx = (dre[0] - dle[0]) / (ds.domain_dimensions[0] * 2 ** L)
    octant = [abs(dle[d]) < 1e-12 for d in range(3)]
    sym = 2 ** sum(octant)
    cen = np.array([0.0 if octant[d] else 0.5 * (dle[d] + dre[d]) for d in range(3)])
    h = 1.3 * R0
    lo = np.array([max(dle[d], cen[d] - h) for d in range(3)])
    n = np.round((np.array([min(dre[d], cen[d] + h) for d in range(3)]) - lo) / dx).astype(int)
    cg = ds.covering_grid(L, lo, n)
    g = lambda f: np.asarray(cg["boxlib", f], float)
    eta, p = g("eta"), g("pressure")
    arg, E1 = g("rho_eta1"), g("energy1")
    E0 = g("energy0")
    x = [lo[d] + (np.arange(n[d]) + 0.5) * dx - cen[d] for d in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    # eta = 0.5 radius along the radial average
    nb = int(h / dx)
    edges = np.linspace(0, h, nb + 1); rc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    pe = np.bincount(idx, weights=eta.ravel(), minlength=nb) / np.maximum(cnt, 1)
    R05 = np.nan
    for i in range(nb - 1):
        if cnt[i] and cnt[i + 1] and (pe[i] - 0.5) * (pe[i + 1] - 0.5) < 0:
            R05 = rc[i] + (0.5 - pe[i]) * (rc[i + 1] - rc[i]) / (pe[i + 1] - pe[i]); break

    al = np.clip(eta, 0, 1); ag = 1 - al
    core = (eta < 0.02) & (r < 0.8 * R05)
    gg, pig = P["g1"], P["pi1"]
    rg = arg / np.maximum(ag, 1e-12)
    pg_own = (gg - 1) * E1 / np.maximum(ag, 1e-12) - gg * pig          # gas pressure from its own energy
    s_mix = p / rg ** gg
    s_own = pg_own / rg ** gg
    out = dict(t=float(ds.current_time), R05=R05,
               rho_core=float(rg[core].mean()), p_core=float(p[core].mean()),
               pg_own_core=float(pg_own[core].mean()),
               s_mix=float(s_mix[core].mean()), s_own=float(s_own[core].mean()),
               eta_core_mean=float(eta[core].mean()), eta_core_max=float(eta[core].max()),
               liq_energy_frac=float(E0[core].sum() / (E0[core].sum() + E1[core].sum())),
               ncore=int(core.sum()))
    # mass-weighted core entropy (what an adiabat should hold constant)
    m = arg[core]
    out["s_mix_massw"] = float(np.sum(s_mix[core] * m) / np.sum(m))
    for k in range(len(ABINS) - 1):
        b = core & (eta >= ABINS[k]) & (eta < ABINS[k + 1])
        out["sA%d" % k] = float(s_mix[b].mean()) if b.any() else np.nan
        out["nA%d" % k] = int(b.sum())
    rr = r / R05
    for k in range(len(RBINS) - 1):
        b = core & (rr >= RBINS[k]) & (rr < RBINS[k + 1])
        out["sR%d" % k] = float(s_mix[b].mean()) if b.any() else np.nan
        out["pR%d" % k] = float(p[b].mean()) if b.any() else np.nan
        out["eR%d" % k] = float(eta[b].mean()) if b.any() else np.nan
    # by LOCATION: the octant's symmetry planes / axes are where the limiter
    # axis jets live, so a hot spot there is a boundary artefact, not bulk gas.
    if any(octant):
        dpl = [np.abs([X, Y, Z][d]) for d in range(3) if octant[d]]
        near_plane = np.min(dpl, axis=0) < 3 * dx
        n_close = sum((q < 3 * dx).astype(int) for q in dpl)
        near_axis = n_close >= 2
        for lab, b in (("axis", core & near_axis), ("plane", core & near_plane & ~near_axis),
                       ("bulk", core & ~near_plane)):
            out["s_" + lab] = float(s_mix[b].mean()) if b.any() else np.nan
            out["n_" + lab] = int(b.sum())
    else:
        for lab in ("axis", "plane", "bulk"):
            out["s_" + lab] = np.nan; out["n_" + lab] = 0
    out["s_max"] = float(s_mix[core].max()); out["s_min"] = float(s_mix[core].min())
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
             g1=_num(kv, "eos1.gamma", 1.4), pi1=_num(kv, "eos1.p0", 0.0),
             pi0=_num(kv, "eos0.p0", 0.0), p_inf=_expr_const(kv, "pressure0.ic", "p_inf", 1e5))
    tau_c = 0.915 * P["R0"] * np.sqrt(1000.0 / P["p_inf"])
    pfs = sorted(d for d in glob.glob(os.path.join(a.plotdir, "*cell")) if os.path.isdir(d))
    if not pfs:
        raise SystemExit("no *cell plotfiles in %s" % a.plotdir)
    print("%d plotfiles; liquid pi %.3g, gas gamma %.3f" % (len(pfs), P["pi0"], P["g1"]))
    with Pool(a.nproc, initializer=_init, initargs=(dict(P),)) as pool:
        rows = sorted(pool.map(frame, pfs), key=lambda d: d["t"])
    if len({d["t"] for d in rows}) < len(rows):
        print("WARNING: repeated times -- this directory mixes runs.")
    r0 = rows[0]
    print("\n t/tc  rho_core  s_mix/s0  s_massw/s0  s_own/s0  p_core   p_g(own)  eta_core mean / max   liq-energy frac")
    for d in rows:
        print("%5.2f  %.5f  %.5f  %.5f  %.5f  %7.0f  %7.0f  %.2e / %.2e  %.4f"
              % (d["t"] / tau_c, d["rho_core"], d["s_mix"] / r0["s_mix"], d["s_mix_massw"] / r0["s_mix_massw"],
                 d["s_own"] / r0["s_own"], d["p_core"], d["pg_own_core"], d["eta_core_mean"], d["eta_core_max"],
                 d["liq_energy_frac"]))
    print("\ns/s0 BY TRACE-LIQUID FRACTION (columns: eta bins; count in brackets at R_min frame)")
    imin = int(np.nanargmin([d["R05"] for d in rows]))
    hdr = "  t/tc " + "".join("  [%.0e,%.0e)" % (ABINS[k], ABINS[k + 1]) for k in range(len(ABINS) - 1))
    print(hdr)
    for d in rows:
        print("  %5.2f" % (d["t"] / tau_c) + "".join(
            "  %13.5f" % (d["sA%d" % k] / r0["s_mix"]) if np.isfinite(d["sA%d" % k]) else "  %13s" % "--"
            for k in range(len(ABINS) - 1)))
    print("  cells at R_min:" + "".join("  %13d" % rows[imin]["nA%d" % k] for k in range(len(ABINS) - 1)))
    print("\ns/s0, p, eta BY RADIUS r/R05")
    print("  t/tc " + "".join("      [%.1f,%.1f) s / p / eta    " % (RBINS[k], RBINS[k + 1]) for k in range(len(RBINS) - 1)))
    for d in rows:
        print("  %5.2f" % (d["t"] / tau_c) + "".join(
            "  %.5f %7.0f %.1e" % (d["sR%d" % k] / r0["s_mix"], d["pR%d" % k], d["eR%d" % k])
            if np.isfinite(d["sR%d" % k]) else "  %24s" % "--" for k in range(len(RBINS) - 1)))
    print("\ns/s0 BY LOCATION (octant symmetry planes/axes = where limiter axis jets live)")
    print("  t/tc    axis      plane     bulk      min       max      (cells axis/plane/bulk)")
    for d in rows:
        f = lambda k: d[k] / r0["s_mix"]
        print("  %5.2f  %.5f  %.5f  %.5f  %.5f  %.5f   (%d/%d/%d)"
              % (d["t"] / tau_c, f("s_axis"), f("s_plane"), f("s_bulk"), f("s_min"), f("s_max"),
                 d["n_axis"], d["n_plane"], d["n_bulk"]))
    lr = np.log([d["rho_core"] for d in rows]); lp = np.log([d["p_core"] for d in rows])
    lpo = np.log([d["pg_own_core"] for d in rows])
    print("\ngamma_eff: mixture p %.4f   gas-own p %.4f   (EOS %.3f)"
          % (np.polyfit(lr, lp, 1)[0], np.polyfit(lr, lpo, 1)[0], P["g1"]))
    if a.csv:
        keys = list(rows[0].keys())
        with open(a.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for d in rows:
                fh.write(",".join(str(d[k]) for k in keys) + "\n")
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
