#!/usr/bin/env python3
"""Full interface-sharpening diagnostics for the PLANAR (x-normal) test.

The original check tracks only int(eta) and band-volume S.  A sharpener can pass
those yet still (a) not actually change the thickness in CELLS, (b) leak momentum/
energy, or (c) break velocity/pressure EQUILIBRIUM (spurious currents) -- the three
things that actually decide "correct + conservative + stable".  This measures all of
them per frame:

  thickness_cells : 10-90% width of the x-averaged eta(x) profile / dx   (the 4-12
                    target the user wants).  Also a band-volume estimate as a check.
  CONSERVATION    : int over the domain of eta, rho_eta0, rho_eta1, momentum_x/y,
                    and total energy -- each must stay flat (sharpening must not
                    create/destroy mass, momentum or energy).
  EQUILIBRIUM     : max|u| and the pressure spread (max-min)/mean -- both must stay
                    ~0 on the quiescent test (sharpening must not kick the flow).

Usage:
    python check_sharpening_full.py <dir>             # one run, full time series
    python check_sharpening_full.py <OFF_dir> <ON_dir>  # compare baseline vs sharpening
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)


def cells(d):
    return sorted(glob.glob(os.path.join(d, "*cell")))


def _first(ad, names):
    for n in names:
        try:
            return np.array(ad[n])
        except Exception:
            pass
    return None


def frame(pf):
    ds = yt.load(pf)
    ad = ds.all_data()
    dx = float(ds.index.get_smallest_dx().to_value()) if hasattr(ds.index, "get_smallest_dx") \
        else float(ds.domain_width[0]) / ds.domain_dimensions[0]
    eta = np.clip(np.array(ad["eta"]), 0.0, 1.0)
    x = np.array(ad["x"])
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    re0 = _first(ad, ["rho_eta0"]); re1 = _first(ad, ["rho_eta1"])
    mx  = _first(ad, ["momentumx", "momentum_x"]); my = _first(ad, ["momentumy", "momentum_y"])
    E   = _first(ad, ["Energy", "energy_per_vol", "energy", "Total_Energy"])
    p   = _first(ad, ["pressure", "Pressure", "p"])
    vx  = _first(ad, ["velocityx", "velocity_x"]); vy = _first(ad, ["velocityy", "velocity_y"])

    # --- thickness from the x-averaged eta profile (interface normal = x) -------
    nb = max(32, int(round((x.max() - x.min()) / dx)))
    edges = np.linspace(x.min() - 0.5 * dx, x.max() + 0.5 * dx, nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    wsum = np.bincount(idx, weights=vol, minlength=nb)
    esum = np.bincount(idx, weights=eta * vol, minlength=nb)
    good = wsum > 0
    xc = 0.5 * (edges[:-1] + edges[1:])[good]
    ep = (esum[good] / wsum[good])

    def xcross(lvl):
        for i in range(len(ep) - 1):
            if (ep[i] - lvl) * (ep[i + 1] - lvl) < 0:
                return xc[i] + (lvl - ep[i]) * (xc[i + 1] - xc[i]) / (ep[i + 1] - ep[i])
        return np.nan
    x10, x90 = xcross(0.1), xcross(0.9)
    thick_cells = abs(x90 - x10) / dx if (np.isfinite(x10) and np.isfinite(x90)) else np.nan

    out = dict(t=float(ds.current_time), dx=dx, thick=thick_cells,
               M_eta=float(np.sum(eta * vol)),
               M_r0=float(np.sum(re0 * vol)) if re0 is not None else np.nan,
               M_r1=float(np.sum(re1 * vol)) if re1 is not None else np.nan,
               P_x=float(np.sum(mx * vol)) if mx is not None else np.nan,
               P_y=float(np.sum(my * vol)) if my is not None else np.nan,
               E=float(np.sum(E * vol)) if E is not None else np.nan,
               umax=float(np.max(np.sqrt((vx**2 if vx is not None else 0) +
                                         (vy**2 if vy is not None else 0))))
               if vx is not None else np.nan,
               pspread=(float((np.max(p) - np.min(p)) / (np.mean(p) + 1e-30))
                        if p is not None else np.nan))
    return out


def series(d, label):
    cs = cells(d)
    if not cs:
        print(f"  {label}: NO OUTPUT in {d}"); return None
    rows = [frame(c) for c in cs]
    f0 = rows[0]
    print(f"\n=== {label}  ({len(cs)} frames, dx={f0['dx']:.4e}) ===")
    print(f"  {'t':>9} {'thick_cl':>8} {'dM_eta%':>9} {'dMr0%':>8} {'dMr1%':>8} "
          f"{'dPx':>9} {'dE%':>8} {'max|u|':>9} {'p_spread':>9}")
    for r in rows:
        dme = 100 * (r['M_eta'] - f0['M_eta']) / (f0['M_eta'] + 1e-30)
        dm0 = 100 * (r['M_r0'] - f0['M_r0']) / (abs(f0['M_r0']) + 1e-30)
        dm1 = 100 * (r['M_r1'] - f0['M_r1']) / (abs(f0['M_r1']) + 1e-30)
        dpx = r['P_x'] - f0['P_x']
        de = 100 * (r['E'] - f0['E']) / (abs(f0['E']) + 1e-30)
        print(f"  {r['t']:>9.3e} {r['thick']:>8.2f} {dme:>+9.4f} {dm0:>+8.4f} {dm1:>+8.4f} "
              f"{dpx:>+9.2e} {de:>+8.4f} {r['umax']:>9.2e} {r['pspread']:>9.2e}")
    return rows


def main():
    if len(sys.argv) == 2:
        series(sys.argv[1], os.path.basename(sys.argv[1].rstrip("/")))
    elif len(sys.argv) >= 3:
        off = series(sys.argv[1], "apply_sharpening=0 (OFF)")
        on  = series(sys.argv[2], "apply_sharpening=1 (ON)")
        if off and on:
            print("\n" + "=" * 70)
            t0o, t1o = off[0]['thick'], off[-1]['thick']
            t0n, t1n = on[0]['thick'], on[-1]['thick']
            print(f"  THICKNESS  OFF: {t0o:.2f} -> {t1o:.2f} cells   "
                  f"ON: {t0n:.2f} -> {t1n:.2f} cells   (target band 4-12)")
            dm = max(abs(100 * (r['M_r0'] - on[0]['M_r0']) / (abs(on[0]['M_r0']) + 1e-30)) for r in on)
            print(f"  ON conservation: max |dM_r0/M0| = {dm:.4f}%")
            print(f"  ON equilibrium:  max|u| = {max(r['umax'] for r in on):.2e}  "
                  f"max p_spread = {max(r['pspread'] for r in on):.2e}")
            print("=" * 70)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
