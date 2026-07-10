#!/usr/bin/env python3
"""Peak velocity vs time -- the collapse-jet signature.

A collapsing/jetting bubble drives a high-speed liquid spike that peaks sharply at
maximum compression.  Per frame this reports max|u| and WHERE it is (radius from the
bubble center, in R0 and in finest-cells).  Compare across resolutions:

  peak |u| keeps SHARPENING as you refine  -> grid/under-resolution artifact (jet is
                                              a numerical imprint of the mesh)
  peak |u| CONVERGES                        -> a real (if under-resolved) jet

Usage: python peak_velocity.py [output_dir]   (default dir from diag_config)
"""
import glob, os, sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

try:
    from diag_config import resolve_dir, R0, N_BASE, MAX_LEVEL
except Exception:
    def resolve_dir(a): return a[1] if len(a) > 1 else "."
    R0, N_BASE, MAX_LEVEL = 0.02, 160, 4

D = resolve_dir(sys.argv)

def cells(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p))

def velocity_mag(ad):
    """|u| from velocity fields, or momentum/density, whichever the plotfile has."""
    for vx, vy, vz in (("velocityx", "velocityy", "velocityz"),
                       ("velocity_x", "velocity_y", "velocity_z"),
                       ("x-velocity", "y-velocity", "z-velocity")):
        try:
            u = np.array(ad[vx]); v = np.array(ad[vy]); w = np.array(ad[vz])
            return np.sqrt(u * u + v * v + w * w)
        except Exception:
            pass
    for mx, my, mz in (("momentumx", "momentumy", "momentumz"),
                       ("momentum_x", "momentum_y", "momentum_z"),
                       ("xmom", "ymom", "zmom")):
        try:
            rho = np.array(ad["density"])
            a = np.array(ad[mx]); b = np.array(ad[my]); c = np.array(ad[mz])
            rho = np.where(rho > 0, rho, np.nan)
            return np.sqrt(a * a + b * b + c * c) / rho
        except Exception:
            pass
    raise RuntimeError("no velocity_* or momentum_* fields in plotfile")

def dx_finest(ds):
    try:
        return float(np.min(np.array(ds.index.get_smallest_dx().to_value())))
    except Exception:
        return float(ds.domain_width[0]) / (N_BASE * (2 ** MAX_LEVEL))

def main():
    cs = cells(D)
    if not cs:
        print("NO OUTPUT in", D); sys.exit(1)
    print(f"PEAK VELOCITY : {os.path.basename(D)}")
    print(f"{'t (s)':>12} {'max|u| (m/s)':>13} {'r_peak/R0':>10} {'r_peak(cells)':>13}")
    rows = []
    for pf in cs:
        ds = yt.load(pf)
        dxf = dx_finest(ds)
        dle, dre = ds.domain_left_edge, ds.domain_right_edge
        cen = [0.0 if float(dle[d]) > -1e-6 else 0.5 * float(dle[d] + dre[d]) for d in range(3)]
        ad = ds.all_data()
        try:
            umag = velocity_mag(ad)
        except RuntimeError as e:
            print("  ", e); sys.exit(1)
        j = int(np.nanargmax(umag))
        x = float(np.array(ad["x"])[j]) - cen[0]
        y = float(np.array(ad["y"])[j]) - cen[1]
        z = float(np.array(ad["z"])[j]) - cen[2]
        rpk = np.sqrt(x * x + y * y + z * z)
        t = float(ds.current_time)
        rows.append((t, float(umag[j]), rpk, rpk / dxf))
        print(f"{t:>12.4e} {umag[j]:>13.3e} {rpk / R0:>10.3f} {rpk / dxf:>13.2f}")
    rows.sort()
    ts = np.array([r[0] for r in rows]); um = np.array([r[1] for r in rows])
    jmax = int(np.argmax(um))
    print("\n" + "=" * 60)
    print(f"  GLOBAL peak |u| = {um[jmax]:.3e} m/s at t={ts[jmax]:.4e}s")
    print(f"  (re-run after refining: SHARPENS->numerical, CONVERGES->real jet)")
    print("=" * 60)
    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ts * 1e3, um, "-o", color="tab:purple", ms=3, lw=1.8)
        ax.axvline(ts[jmax] * 1e3, color="0.6", ls="--", lw=1.0)
        ax.set_xlabel("t [ms]"); ax.set_ylabel("max |u|  [m/s]")
        ax.set_title("Peak velocity vs time (collapse-jet signature)")
        ax.grid(True, alpha=0.3)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
        os.makedirs(out, exist_ok=True)
        fn = os.path.join(out, "peak_velocity.png")
        fig.savefig(fn, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {fn}")
    except Exception as e:
        print("  [plot skipped]", e)

if __name__ == "__main__":
    main()
