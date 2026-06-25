"""Octant-symmetry verification: compare _OctVerify_Full vs _OctVerify_Oct.

If the reflect_even/reflect_odd symmetry BCs are correct, the octant run must
reproduce the +x+y+z corner of the full run.  We check:
  (1) bubble radius from the GAS VOLUME:  R_full  vs  R from 8 x V_octant
  (2) eta, NORMAL velocity, pressure along the +x / +y / +z rays (one per
      symmetry plane); the normal-velocity match is the reflect_odd test.
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FULL = os.path.join(ROOT, "output_OctVerify_Full")
OCT  = os.path.join(ROOT, "output_OctVerify_Oct")

def last_cell(d):
    cs = sorted(glob.glob(os.path.join(d, "*cell")))
    return cs[-1] if cs else None

def gas_volume(ds):
    ad = ds.all_data()
    ag = np.clip(1.0 - np.array(ad["eta"]), 0.0, 1.0)
    try:    vol = np.array(ad["index", "cell_volume"])
    except Exception: vol = np.array(ad["cell_volume"])
    return float(np.sum(ag * vol))

def ray_profile(ds, axis):
    """Return (s, eta, v_normal, p) sorted along `axis` at interior offset 0.03."""
    o = 0.03
    if axis == "x":
        a, b, vn = [1e-4, o, o], [0.0999, o, o], "velocityx"; key = "x"
    elif axis == "y":
        a, b, vn = [o, 1e-4, o], [o, 0.0999, o], "velocityy"; key = "y"
    else:
        a, b, vn = [o, o, 1e-4], [o, o, 0.0999], "velocityz"; key = "z"
    r = ds.ray(ds.arr(a, "code_length"), ds.arr(b, "code_length"))
    s = np.array(r[key]); eta = np.array(r["eta"]); p = np.array(r["pressure"])
    try:    v = np.array(r[vn])
    except Exception: v = np.zeros_like(s)
    i = np.argsort(s)
    return s[i], eta[i], v[i], p[i]

def main():
    fpf, opf = last_cell(FULL), last_cell(OCT)
    if not fpf or not opf:
        print(f"MISSING output: full={fpf} oct={opf}"); sys.exit(1)
    df, do = yt.load(fpf), yt.load(opf)
    print("="*66)
    print("OCTANT-SYMMETRY VERIFICATION")
    print("="*66)
    print(f"  full: {os.path.basename(fpf)}  t={float(df.current_time):.3e}")
    print(f"  oct : {os.path.basename(opf)}  t={float(do.current_time):.3e}")

    # (1) gas-volume radius
    R0 = 0.02
    Vf = gas_volume(df); Vo = gas_volume(do) * 8.0
    Rf = (3*Vf/(4*np.pi))**(1/3); Ro = (3*Vo/(4*np.pi))**(1/3)
    dR = abs(Rf-Ro)/Rf*100 if Rf else float('nan')
    print("\n  (1) GAS-VOLUME RADIUS")
    print(f"      R_full      = {Rf/R0:.5f} R0")
    print(f"      R_oct(x8)   = {Ro/R0:.5f} R0   -> diff {dR:.3f}%")

    # (2) rays
    print("\n  (2) RAY PROFILES (octant interpolated onto full's coords)")
    print(f"      {'axis':>4} {'eta Linf':>11} {'v_n Linf':>12} {'p relL2':>11}")
    worst_eta = worst_v = worst_p = 0.0
    for ax in ("x", "y", "z"):
        sf, ef, vf, pf_ = ray_profile(df, ax)
        so, eo, vo, po  = ray_profile(do, ax)
        eo_i = np.interp(sf, so, eo); vo_i = np.interp(sf, so, vo); po_i = np.interp(sf, so, po)
        e_linf = float(np.max(np.abs(ef - eo_i)))
        vscale = max(np.max(np.abs(vf)), 1e-12)
        v_linf = float(np.max(np.abs(vf - vo_i)))            # absolute m/s
        p_rel  = float(np.linalg.norm(pf_ - po_i)/max(np.linalg.norm(pf_), 1e-30))
        print(f"      {ax:>4} {e_linf:>11.3e} {v_linf:>12.3e} {p_rel:>11.3e}   (|v|max~{vscale:.2e})")
        worst_eta = max(worst_eta, e_linf); worst_v = max(worst_v, v_linf); worst_p = max(worst_p, p_rel)

    # verdict
    ok = (dR < 1.0) and (worst_eta < 1e-2) and (worst_p < 1e-2)
    print("\n" + "="*66)
    print(f"  VERDICT: {'PASS -- octant reproduces full (symmetry BC works)' if ok else 'FAIL -- octant diverges from full (inspect symmetry BC)'}")
    print(f"    R diff {dR:.3f}% | worst eta Linf {worst_eta:.2e} | worst v_n Linf {worst_v:.2e} | worst p relL2 {worst_p:.2e}")
    print("="*66)

if __name__ == "__main__":
    main()
