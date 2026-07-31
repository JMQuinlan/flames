#!/usr/bin/env python3
"""How well-resolved is the bubble through the collapse?

Per frame: radially-averaged (volume-weighted) eta profile -> the gas-core radius
(eta=0.1 crossing, i.e. alpha_g=0.9), the interface midpoint (eta=0.5) and outer
edge (eta=0.9), expressed in FINEST CELLS (dx_finest = domain/(n_base*2^max_level)).

  cells_core  = 2 * R(eta=0.1) / dx_finest   -- gas-core DIAMETER in cells
  cells_band  = (R(eta=0.9) - R(eta=0.1)) / dx_finest

If cells_core drops to a handful at the minimum (and band >= core), the bubble is
unresolved there -> grid imprinting (square/jet/flattening) is expected, and the
fix is resolution.  Re-run after adding AMR levels and compare these curves.

Usage: python resolution_at_collapse.py <output_dir> [n_base] [max_level]
       n_base/max_level only needed if yt can't infer dx_finest (fallback).
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)
try:
    from diag_config import resolve_dir, N_BASE as _NB, MAX_LEVEL as _ML
except Exception:
    def resolve_dir(a): return a[1] if len(a) > 1 else "."
    _NB, _ML = None, None

D = resolve_dir(sys.argv)
NB = int(sys.argv[2]) if len(sys.argv) > 2 else _NB
ML = int(sys.argv[3]) if len(sys.argv) > 3 else _ML

def cells(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))

def crossing(r, f, lvl):
    for i in range(len(r) - 1):
        if (f[i] - lvl) * (f[i + 1] - lvl) < 0:
            return r[i] + (lvl - f[i]) * (r[i + 1] - r[i]) / (f[i + 1] - f[i])
    return np.nan

def dx_finest(ds):
    try:
        return float(np.min(np.array(ds.index.get_smallest_dx().to_value())))
    except Exception:
        nb = NB if NB else int(ds.domain_dimensions[0])
        ml = ML if ML is not None else int(ds.max_level)
        return float(ds.domain_width[0]) / (nb * (2 ** ml))

rows = []
for pf in cells(D):
    ds = yt.load(pf)
    dxf = dx_finest(ds)
    dle = ds.domain_left_edge; dre = ds.domain_right_edge
    cen = [0.0 if float(dle[d]) > -1e-6 else 0.5 * float(dle[d] + dre[d]) for d in range(3)]
    ad = ds.all_data()
    x = np.array(ad["x"]) - cen[0]; y = np.array(ad["y"]) - cen[1]; z = np.array(ad["z"]) - cen[2]
    r = np.sqrt(x * x + y * y + z * z)
    eta = np.array(ad["eta"])
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    nb = 400
    rmax = min(float(r.max()), 0.1)
    edges = np.linspace(0.0, rmax, nb + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
    rc = 0.5 * (edges[:-1] + edges[1:])
    wsum = np.bincount(idx, weights=vol, minlength=nb)
    esum = np.bincount(idx, weights=eta * vol, minlength=nb)
    good = wsum > 0
    rr = rc[good]; ee = esum[good] / wsum[good]
    r01 = crossing(rr, ee, 0.1)   # gas-core edge (alpha_g=0.9)
    r05 = crossing(rr, ee, 0.5)
    r09 = crossing(rr, ee, 0.9)   # outer band edge
    core_d = 2.0 * r01 / dxf if np.isfinite(r01) else np.nan
    band_c = (r09 - r01) / dxf if (np.isfinite(r01) and np.isfinite(r09)) else np.nan
    rows.append((float(ds.current_time), dxf, r01, r05, r09, core_d, band_c))

rows.sort()
print(f"{'t (s)':>12} {'dx_fin':>9} {'R_core':>10} {'R_0.5':>10} {'core_diam_cl':>13} {'band_cells':>12}")
imin = min(range(len(rows)), key=lambda i: rows[i][3] if np.isfinite(rows[i][3]) else 9)
for i, (t, dxf, r01, r05, r09, cd, bc) in enumerate(rows):
    mk = "  <== R_0.5 MIN" if i == imin else ""
    print(f"{t:>12.4e} {dxf:>9.3e} {r01:>10.3e} {r05:>10.3e} {cd:>13.2f} {bc:>12.2f}{mk}")

t, dxf, r01, r05, r09, cd, bc = rows[imin]
print("\n" + "=" * 64)
print(f"AT TIGHTEST POINT (t={t:.4e}s, dx_finest={dxf:.3e}):")
print(f"  gas-core diameter = {cd:.2f} cells   band = {bc:.2f} cells")
if np.isfinite(cd) and cd < 8:
    print(f"  -> UNDER-RESOLVED (core < 8 cells; band {'>=' if bc>=cd else '<'} core).")
    print(f"     Grid imprinting (square/jet/flatten) expected; add AMR levels.")
else:
    print(f"  -> core >= 8 cells; if asphericity persists it's less likely pure resolution.")
print("=" * 64)
