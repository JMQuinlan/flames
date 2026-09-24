#!/usr/bin/env python3
"""
Isolate the SHELL DILATATIONAL VISCOSITY (kappa_s) in a Marmottant run.

The solver builds the scalar surface tension that enters Omega as

    sigma_tot = sigma(Gamma) + kappa_s * div_s(u)        (Hydro2.cpp:1745)

and then, when shell.sigma_floor = 1, clamps it:

    if (se < 0) se = 0                                   (Hydro2.cpp:1764)

Because div_s(u) = 2 Rdot / R, the viscous term is POSITIVE while the bubble
expands and NEGATIVE while it compresses.  The clamp therefore does not damp
symmetrically -- it RECTIFIES: the expansion half-cycle keeps its viscous
contribution and the compression half-cycle loses it.  That is the signature
of "the result still looks elastic-only".

Two diagnostics are written by the solver specifically for this:
    kappa1 = sigma(Gamma) + kappa_s div_s(u)   BEFORE the clamp
    kappa2 = sigma(Gamma)                      bare elastic branch only
so the viscous contribution is measured directly as kappa1 - kappa2, and the
clamp duty cycle is the band fraction with kappa1 < 0.

Usage:  analyze_shell_viscosity.py <plotdir> [<plotdir> ...]
        (each plotdir is the directory holding the NNNNNcell frames)
"""
import glob, os, sys
import numpy as np

R0_DEFAULT = 2.0e-6


def _frames(plotdir):
    fr = sorted(glob.glob(os.path.join(plotdir, "*cell")))
    return [f for f in fr if os.path.isdir(f)]


def _band_stats(pf, R0, half):
    """Angle-averaged interface quantities from the finest-level covering grid."""
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    t = float(ds.current_time)
    lev = ds.index.max_level
    dx = float(ds.domain_width[0]) / ds.domain_dimensions[0] / 2 ** lev

    # Octant runs start at the origin; clamp the window to the domain.
    lo = [float(v) for v in ds.domain_left_edge]
    hi = [min(float(ds.domain_right_edge[d]), lo[d] + half) for d in range(3)]
    dims = [max(1, int(round((hi[d] - lo[d]) / dx))) for d in range(3)]
    cg = ds.covering_grid(level=lev, left_edge=lo, dims=dims)

    eta = np.asarray(cg["eta"])
    gam = np.asarray(cg["Gamma"])
    k1 = np.asarray(cg["kappa1"])          # sigma(Gamma) + kappa_s div_s u, pre-clamp
    k2 = np.asarray(cg["kappa2"])          # sigma(Gamma) only

    # Radius from the liquid volume fraction: eta = 1 outside, 0 inside.
    # Octant, so the measured volume is 1/8 of the sphere.
    vol = float(np.sum(1.0 - eta)) * dx ** 3
    R = (3.0 * (8.0 * vol) / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Interface band: weight by |grad eta| so the average is a surface average.
    gx = np.asarray(cg["grad_etax"]); gy = np.asarray(cg["grad_etay"])
    gz = np.asarray(cg["grad_etaz"])
    w = np.sqrt(gx * gx + gy * gy + gz * gz)
    m = w > 0.05 * w.max()
    if not m.any():
        return None
    ww = w[m]
    avg = lambda q: float(np.sum(q[m] * ww) / np.sum(ww))

    visc = k1 - k2                          # = kappa_s * div_s(u)
    clamped = float(np.sum(ww[(k1[m] < 0.0)]) / np.sum(ww))
    return dict(t=t, R=R, RR0=R / R0, Gamma=avg(gam), sig_bare=avg(k2),
                sig_tot=avg(k1), visc=avg(visc), visc_min=float(visc[m].min()),
                visc_max=float(visc[m].max()), clamp_frac=clamped)


def run(plotdir, R0=R0_DEFAULT, half=6.0e-6):
    rows = []
    for pf in _frames(plotdir):
        try:
            r = _band_stats(pf, R0, half)
        except Exception as e:                       # noqa: BLE001
            print("  !! %s: %s" % (os.path.basename(pf), e)); continue
        if r:
            rows.append(r)
    if not rows:
        print("  (no usable frames in %s)" % plotdir); return rows
    print("\n=== %s ===" % plotdir)
    print("%10s %9s %8s %10s %10s %10s %10s %8s" %
          ("t [s]", "R/R0", "Gamma", "sig(Gam)", "sig_tot", "kap_s*divs", "visc_min", "clamp%"))
    for r in rows:
        print("%10.3e %9.4f %8.4f %10.4f %10.4f %10.4f %10.4f %8.1f" %
              (r["t"], r["RR0"], r["Gamma"], r["sig_bare"], r["sig_tot"],
               r["visc"], r["visc_min"], 100.0 * r["clamp_frac"]))
    v = np.array([r["visc"] for r in rows])
    e = np.array([r["sig_bare"] for r in rows])
    print("  viscous/elastic magnitude ratio: mean %.2f  max %.2f" %
          (np.mean(np.abs(v)) / max(np.mean(np.abs(e)), 1e-30),
           np.max(np.abs(v)) / max(np.mean(np.abs(e)), 1e-30)))
    print("  band-time fraction with sigma_tot < 0 (clamp active): %.1f %%" %
          (100.0 * np.mean([r["clamp_frac"] for r in rows])))
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    all_rows = {}
    for d in sys.argv[1:]:
        all_rows[d] = run(d)
    if len(all_rows) == 2:
        (da, ra), (db, rb) = list(all_rows.items())
        n = min(len(ra), len(rb))
        if n:
            print("\n=== A/B: %s  vs  %s ===" % (os.path.basename(da), os.path.basename(db)))
            print("%10s %12s %12s %10s" % ("t [s]", "R/R0 (A)", "R/R0 (B)", "dR/R0"))
            for i in range(n):
                print("%10.3e %12.5f %12.5f %10.2e" %
                      (ra[i]["t"], ra[i]["RR0"], rb[i]["RR0"], rb[i]["RR0"] - ra[i]["RR0"]))
            amp = lambda rr: max(x["RR0"] for x in rr) - min(x["RR0"] for x in rr)
            print("  oscillation amplitude in R/R0:  A %.5f   B %.5f   ratio %.3f" %
                  (amp(ra), amp(rb), amp(rb) / max(amp(ra), 1e-30)))
