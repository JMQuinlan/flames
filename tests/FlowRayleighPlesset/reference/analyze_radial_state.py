#!/usr/bin/env python3
"""From the radial_state_*.csv dumps: is the R_vol miss interface DIFFUSION or a
METRIC artifact?  Per frame compute R_vol (gas-volume radius), R_eta (eta=0.5
crossing of the radially-averaged profile), and the interface band width
(r where 0.1<eta<0.9) in metres and in finest-cells.

  diffusion  -> band width BLOOMS at the minimum (Sch20 stays ~2-3 cells)
  metric     -> R_vol >> R_eta (gas volume integral over-reads vs the 0.5 contour)
"""
import glob, os, sys
import numpy as np

D  = sys.argv[1] if len(sys.argv) > 1 else "."
DX = float(sys.argv[2]) if len(sys.argv) > 2 else 7.81e-4   # finest dx

def load(fn):
    t = float(open(fn).readline().split("t=")[1].split()[0])
    a = np.genfromtxt(fn, delimiter=",", names=True, skip_header=1)
    return t, a

def crossing(r, f, lvl):
    """first r where f drops through lvl (gas inside: eta low->high outward)."""
    for i in range(len(r)-1):
        if (f[i]-lvl)*(f[i+1]-lvl) < 0 and np.isfinite(f[i]) and np.isfinite(f[i+1]):
            return r[i] + (lvl-f[i])*(r[i+1]-r[i])/(f[i+1]-f[i])
    return np.nan

rows = []
for fn in sorted(glob.glob(os.path.join(D, "radial_state_*.csv"))):
    if "summary" in fn:
        continue
    t, a = load(fn)
    r, eta = a["r"], a["eta"]
    m = a["count"] > 0
    r, eta = r[m], eta[m]
    re05 = crossing(r, eta, 0.5)
    r01  = crossing(r, eta, 0.1)
    r09  = crossing(r, eta, 0.9)
    width = (r09 - r01) if (np.isfinite(r01) and np.isfinite(r09)) else np.nan
    # gas-volume radius from this profile (spherical integral of alpha_g=1-eta)
    ag = np.clip(1.0 - eta, 0, 1)
    dr = np.gradient(r)
    Vg = np.sum(ag * 4.0*np.pi*r*r*dr)
    Rvol = (3.0*Vg/(4.0*np.pi))**(1.0/3.0) if Vg > 0 else np.nan
    pk_rg = np.nanmax(a["rho_eta1"][m]) if "rho_eta1" in a.dtype.names else np.nan
    pk_p  = np.nanmax(a["pressure"][m]) if "pressure"  in a.dtype.names else np.nan
    rows.append((t, Rvol, re05, width, width/DX, pk_rg, pk_p))

rows.sort()
print(f"{'t (s)':>11} {'R_vol':>10} {'R_eta0.5':>10} {'band(m)':>10} {'band/dx':>8} "
      f"{'R_vol/R_eta':>11} {'peak_rg':>9} {'peak_p':>10}")
imin = min(range(len(rows)), key=lambda i: rows[i][1] if np.isfinite(rows[i][1]) else 9)
for i, (t, Rv, re, w, wc, prg, pp) in enumerate(rows):
    ratio = Rv/re if (np.isfinite(re) and re > 0) else np.nan
    mark = "  <== R_vol MIN" if i == imin else ""
    print(f"{t:>11.4e} {Rv:>10.4e} {re:>10.4e} {w:>10.4e} {wc:>8.2f} "
          f"{ratio:>11.3f} {prg:>9.2e} {pp:>10.3e}{mark}")

t, Rv, re, w, wc, prg, pp = rows[imin]
print("\n" + "="*70)
print(f"AT R_vol MINIMUM (t={t:.4e}s):")
print(f"  R_vol      = {Rv:.4e} m   (KM target ~5.46e-3)")
print(f"  R_eta=0.5  = {re:.4e} m   <- where the interface actually is")
print(f"  band width = {w:.4e} m = {wc:.2f} finest cells  (Sch20 keeps ~2-3)")
w0 = rows[0][3]
print(f"  band at window start = {w0:.4e} m = {w0/DX:.2f} cells  -> bloom factor {w/w0:.2f}x")
print("="*70)
if np.isfinite(re) and re > 0 and Rv/re > 1.15:
    print("VERDICT: R_vol over-reads vs the 0.5 contour -> METRIC/diffuse-band dominates.")
elif np.isfinite(w) and w/DX > 5:
    print("VERDICT: band BLOOMS to >5 cells -> physical interface DIFFUSION at collapse.")
else:
    print("VERDICT: interface tight AND R_vol~R_eta -> bubble genuinely under-collapses (physics).")
