#!/usr/bin/env python3
"""Per-phase MASS conservation over a Sch20 bubble run.

The gas mass  m_g = integral(rho_eta1 dV)  and liquid mass m_l = integral(rho_eta0 dV)
must each be CONSTANT in time (the gas compresses during collapse but its mass is
conserved; the liquid is effectively incompressible).  If m_g drifts -- especially
across the violent collapse -- the scheme/AMR is NOT conservative there, and the
bubble cannot reach the correct minimum regardless of interface sharpness.  This is
the decisive test that separates "non-conservative AMR" from "interface diffusion".

AMR-correct: yt returns leaf cells only (no double counting).  Octant auto-scaled.

Usage: python check_mass_conservation.py <output_dir>
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)
try:
    from diag_config import resolve_dir
except Exception:
    def resolve_dir(a): return a[1] if len(a) > 1 else "."

def cells(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))

def masses(pf):
    ds = yt.load(pf)
    dle = ds.domain_left_edge
    sym = 1
    for d in range(3):
        if float(dle[d]) > -1e-6:
            sym *= 2
    ad = ds.all_data()
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    mg = float(np.sum(np.array(ad["rho_eta1"]) * vol)) * sym   # gas partial density
    ml = float(np.sum(np.array(ad["rho_eta0"]) * vol)) * sym   # liquid partial density
    return float(ds.current_time), mg, ml, sym

def main():
    d = resolve_dir(sys.argv)
    cs = cells(d)
    if not cs:
        print("NO OUTPUT in", d); sys.exit(1)
    t0, mg0, ml0, sym = masses(cs[0])
    print("=" * 70)
    print(f"PER-PHASE MASS CONSERVATION : {os.path.basename(d)}  (sym x{sym})")
    print("=" * 70)
    print(f"  m_gas(0)={mg0:.6e}  m_liq(0)={ml0:.6e}")
    print(f"  {'t':>11} {'m_gas':>13} {'dm_gas/m0 %':>12} {'m_liq':>13} {'dm_liq/m0 %':>12}")
    mgs = []
    for c in cs:
        t, mg, ml, _ = masses(c)
        mgs.append(mg)
        print(f"  {t:>11.4e} {mg:>13.6e} {100*(mg-mg0)/mg0:>+12.4f} "
              f"{ml:>13.6e} {100*(ml-ml0)/ml0:>+12.4f}")
    mgs = np.array(mgs)
    drift = 100.0 * np.max(np.abs(mgs - mgs[0]) / mgs[0])
    print("\n" + "=" * 70)
    print(f"  GAS-MASS max drift = {drift:.4f}%")
    if drift < 0.1:
        print("  -> gas mass CONSERVED. The missed minimum is NOT a mass-conservation")
        print("     bug; look at interface diffusion / pressure coupling instead.")
    else:
        print("  -> gas mass NOT conserved (AMR reflux / c-f). THIS is the missed-")
        print("     minimum cause: the bubble loses/gains gas across the collapse.")
    print("=" * 70)

if __name__ == "__main__":
    main()
