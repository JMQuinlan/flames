#!/usr/bin/env python3
"""
Steady Re = 40 cylinder check for tests/FlowVortexShed/UNIT_TEST_2D_Re40 (Hydro2)
and Re40_input_hydro (Hydro).

  python3 cylinder_re40_check.py <run_dir>                       # summary of the last plotfile
  python3 cylinder_re40_check.py <run_dir> --csv [--stride N]    # + metric CSVs

<run_dir> holds output/ (plotfiles) and, for Hydro2 run with solid.force_int=1,
output_forces.dat.  Metrics (per plotfile):
  * recirculation length L/D: rear stagnation point (x = R) to the u_x = 0
    crossing on the centreline (closure point x_c)
  * wake width W/D: transverse extent of the reversed-flow region (u_x < 0, x > 0,
    outside the cylinder)
  * wake area A/D^2: area of that region
  * separation angle from the rear stagnation point (sign change of the
    tangential velocity on a ring 1.5 finest cells outside the wall, upper half)
  * Cd, Cd_pressure, Cd_friction (= Cd - Cd_p), Cl from the force history at that
    time (q = 0.5 rho U^2 D, rho=100, U=1, D=1); blank for Hydro (no force file)
--csv writes <run_dir>/cylinder_metrics.csv (time series, every N-th plotfile,
default N = 4) and <run_dir>/cylinder_summary.csv (last plotfile + literature).
Literature, unbounded steady Re 40 (verify): Cd 1.50-1.60 (Tritton 1959 ~1.59,
Dennis & Chang 1970 1.52 = ~1.00 pressure + ~0.52 friction, Fornberg 1980 1.50),
L/D 2.13-2.35 (Coutanceau & Bouard 1977 2.13, Dennis & Chang 2.35),
theta_sep 53-54 deg.
"""
import glob, os, re, sys
import numpy as np

RHO, U, D, R = 100.0, 1.0, 1.0, 0.5
LIT = {"L_over_D": "2.13-2.35", "Cd": "1.50-1.60", "Cd_pressure": "~1.00",
       "Cd_friction": "~0.52", "Cl": "0", "sep_angle_deg": "53-54"}


def plotfiles(d):
    pfs = [p for p in glob.glob(os.path.join(d, "output", "*cell")) if ".old." not in p]
    return sorted(pfs, key=lambda p: int(re.search(r"(\d+)cell$", p).group(1)))   # numeric: 100000cell > 99994cell


def load_forces(d):
    fpath = os.path.join(d, "output_forces.dat")
    if not os.path.exists(fpath):
        return None
    F = np.loadtxt(fpath)
    return F[F[:, 1] == F[:, 1].max()]          # finest level only


def forces_at(F, t, q):
    """Cd, Cd_p, Cd_f, Cl at the force-history sample nearest to time t."""
    if F is None:
        return dict(Cd=None, Cd_pressure=None, Cd_friction=None, Cl=None)
    k = int(np.argmin(np.abs(F[:, 2] - t)))
    Cd, Cl, Cdp = F[k, 3] / q, F[k, 4] / q, F[k, 5] / q
    return dict(Cd=float(Cd), Cd_pressure=float(Cdp), Cd_friction=float(Cd - Cdp), Cl=float(Cl))


def wake_metrics(pf):
    import yt
    from scipy.interpolate import RegularGridInterpolator as RGI
    yt.set_log_level(40)
    ds = yt.load(pf); L = ds.index.max_level
    cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions * 2**L)
    nx, ny = cg.ActiveDimensions[:2]
    lo = ds.domain_left_edge.d; hi = ds.domain_right_edge.d
    dx = (hi[0] - lo[0]) / nx; dy = (hi[1] - lo[1]) / ny
    x = lo[0] + (np.arange(nx) + 0.5) * dx; y = lo[1] + (np.arange(ny) + 0.5) * dy
    u = np.asarray(cg["boxlib", "velocityx"])[:, :, 0]; v = np.asarray(cg["boxlib", "velocityy"])[:, :, 0]
    out = dict(time=float(ds.current_time), plotfile=os.path.basename(pf), dx=dx,
               x_closure=None, L_over_D=None, wake_width=None, wake_area=None, sep_angle_deg=None)
    # centreline: average the two rows straddling y = 0
    j = np.searchsorted(y, 0.0); uc = 0.5 * (u[:, j - 1] + u[:, j])
    xs = x > R + dx
    ii = np.where(xs[:-1] & (uc[:-1] < 0) & (uc[1:] >= 0))[0]
    if len(ii):
        i = ii[0]; x0 = x[i] - uc[i] * (x[i + 1] - x[i]) / (uc[i + 1] - uc[i])
        out["x_closure"] = float(x0); out["L_over_D"] = float((x0 - R) / D)
    # reversed-flow region behind the body (outside the cylinder)
    X, Y = np.meshgrid(x, y, indexing="ij")
    rev = (u < 0) & (X > 0) & (np.hypot(X, Y) > R + dx)
    if rev.any():
        out["wake_area"] = float(rev.sum() * dx * dy / D**2)
        out["wake_width"] = float((Y[rev].max() - Y[rev].min() + dy) / D)
    # separation angle: tangential velocity on a ring just outside the wall
    rr = R + 1.5 * dx
    th = np.radians(np.linspace(0.5, 179.5, 359))                 # from the REAR stagnation point, upper half
    px, py = rr * np.cos(th), rr * np.sin(th)
    iu = RGI((x, y), u); iv = RGI((x, y), v)
    ut = -iu(np.c_[px, py]) * np.sin(th) + iv(np.c_[px, py]) * np.cos(th)   # counter-clockwise tangent
    k = np.where(np.sign(ut[:-1]) != np.sign(ut[1:]))[0]
    if len(k):
        out["sep_angle_deg"] = float(np.degrees(th[k[0]]))
    return out


def fmt(v, nd=6):
    return "" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def main():
    args = sys.argv[1:]
    d = args[0]
    want_csv = "--csv" in args
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 4
    q = 0.5 * RHO * U * U * D
    F = load_forces(d)
    if F is not None:
        t = F[:, 2]; Cd = F[:, 3] / q; Cl = F[:, 4] / q; Cdp = F[:, 5] / q
        m = t >= t[-1] - 5.0
        print(f"{d}: force history t = {t[0]:.2f} .. {t[-1]:.2f}")
        print(f"  Cd = {Cd[m].mean():.4f} +- {Cd[m].std():.1e}  (pressure {Cdp[m].mean():.4f}, friction {Cd[m].mean()-Cdp[m].mean():.4f})"
              f"   drift {np.polyfit(t[m], Cd[m], 1)[0]:+.1e}/time")
        print(f"  Cl = {Cl[m].mean():+.2e} +- {Cl[m].std():.1e}")
    else:                                  # e.g. single-phase Hydro: no force history
        print(f"{d}: no output_forces.dat -- wake geometry only")
    pfs = plotfiles(d)
    last = wake_metrics(pfs[-1])
    if last["L_over_D"] is not None:
        print(f"  recirculation length L/D = {last['L_over_D']:.3f}   (t = {last['time']:.1f}, finest dx = {last['dx']:.4f})")
    else:
        print("  no reversed flow on the centreline")
    if last["wake_width"] is not None:
        print(f"  wake width W/D = {last['wake_width']:.3f}   wake area A/D^2 = {last['wake_area']:.3f}")
    if last["sep_angle_deg"] is not None:
        print(f"  separation angle (from rear) = {last['sep_angle_deg']:.1f} deg")
    print(f"  plotfile {last['plotfile']}")
    if not want_csv:
        return
    cols = ["time", "plotfile", "L_over_D", "x_closure", "wake_width", "wake_area", "sep_angle_deg",
            "Cd", "Cd_pressure", "Cd_friction", "Cl"]
    sel = pfs[::stride]
    if sel[-1] != pfs[-1]:
        sel.append(pfs[-1])
    rows = []
    for pf in sel:
        r = dict(last) if pf == pfs[-1] else wake_metrics(pf)
        r.update(forces_at(F, r["time"], q)); rows.append(r)
    ts = os.path.join(d, "cylinder_metrics.csv")
    with open(ts, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[c]) for c in cols) + "\n")
    print(f"  wrote {ts}  ({len(rows)} rows, every {stride} plotfiles)")
    fin = rows[-1]
    sm = os.path.join(d, "cylinder_summary.csv")
    with open(sm, "w") as f:
        f.write("metric,value,literature\n")
        for c in ["time", "L_over_D", "x_closure", "wake_width", "wake_area", "sep_angle_deg",
                  "Cd", "Cd_pressure", "Cd_friction", "Cl"]:
            f.write(f"{c},{fmt(fin[c])},{LIT.get(c, '')}\n")
        if F is not None:
            f.write(f"Cd_mean_last5,{Cd[m].mean():.6g},\nCd_drift_per_time,{np.polyfit(t[m], Cd[m], 1)[0]:.3e},\n")
    print(f"  wrote {sm}")


if __name__ == "__main__":
    main()
