#!/usr/bin/env python3
"""
hydro2_diag.py -- one-shot health diagnostics for a Hydro2 run (INCLINE or local).

    python scripts/hydro2_diag.py <plotfile_dir> [--log run.log] [--every N]
                                  [--center X Y Z] [--pinf P]

Reads every Nth plotfile (default: enough for ~15 rows) plus, if given, the
console log, and prints one row per frame with the quantities that have
fingerprinted every failure mechanism found in the 2026-09 debugging campaign:

  p_gas_min / @r     min pressure over gas cells (eta<0.5) and its radius.
                     A dip parked at r~0 that deepens through the expansion
                     half-cycle = CENTER-FOCUSING DRAIN (mechanism 5).
  u_gas_max          max |u| in the gas. Growing with the drain = same.
  dp_bdry_max        max |p - p_inf| within 8 cells of any domain face.
                     Exponential growth (e-fold ~1e-5 s) = BOUNDARY
                     AMPLIFICATION (legacy NSCBC ghosts, mechanism 1).
  re_band_max        max per-phase density in the band (0.01<eta<0.99),
                     normalized by its t=0 value. Secular growth toward
                     2-7x = SNOWPLOW / band mass pile (mechanism: slaving
                     or band-edge flux bias).
  shell mn/mx, NaN   shell field extremes + NaN count.  Drift away from
                     ~1 (or any NaN) = SHELL ROT (mechanism 3; should be
                     impossible with the row gated).
  R_vol              volume-equivalent bubble radius (auto octant factor).

Log analysis (--log): dt history sampled every 500 steps, flags the ONSET of
dt decay (first step where dt < 0.5 * median of the first 1000 steps) --
that onset time is the number that localizes a slow ratchet.

Exit code: 0 healthy, 2 if any fingerprint flag fired (greppable in job
scripts).  Requires numpy + yt.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import yt
    yt.funcs.mylog.setLevel(50)
except ImportError:
    print("ERROR: yt is required (pip install yt / module load python-yt)")
    sys.exit(1)


def list_plotfiles(out_dir):
    pfs = sorted(
        d for d in glob.glob(os.path.join(out_dir, "*cell"))
        if os.path.isdir(d) and ".old." not in d
    )
    return pfs


def analyze_frame(pf, center, pinf, re10, re00):
    ds = yt.load(pf)
    ad = ds.all_data()
    t = float(ds.current_time)
    dim = int(ds.dimensionality)

    def fld(name):
        try:
            return np.asarray(ad["boxlib", name])
        except Exception:
            return None

    eta = fld("eta")
    p = fld("pressure")
    vol = np.asarray(ad["index", "cell_volume"])
    x = np.asarray(ad["index", "x"]) - center[0]
    y = np.asarray(ad["index", "y"]) - center[1]
    if dim == 3:
        z = np.asarray(ad["index", "z"]) - center[2]
    else:
        z = np.zeros_like(x)
    r = np.sqrt(x * x + y * y + z * z)

    row = {"t": t}

    # --- gas interior health (mechanism 5: center drain) ---
    gas = eta < 0.5
    if gas.any() and p is not None:
        ig = int(np.argmin(np.where(gas, p, np.inf)))
        row["p_gas_min"] = float(p[ig])
        row["r_at_pmin"] = float(r[ig])
        vx = fld("velocityx"); vy = fld("velocityy")
        vz = fld("velocityz") if dim == 3 else None
        if vx is not None:
            u2 = vx * vx + vy * vy + (vz * vz if vz is not None else 0.0)
            row["u_gas_max"] = float(np.sqrt(np.max(np.where(gas, u2, 0.0))))

    # --- boundary health (mechanism 1: boundary amplification) ---
    if p is not None:
        lo = [float(v) for v in ds.domain_left_edge.to_value()]
        hi = [float(v) for v in ds.domain_right_edge.to_value()]
        dxc = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]).to_value()) \
            / int(ds.domain_dimensions[0])
        xa = np.asarray(ad["index", "x"]); ya = np.asarray(ad["index", "y"])
        near = (xa < lo[0] + 8 * dxc) | (xa > hi[0] - 8 * dxc) \
             | (ya < lo[1] + 8 * dxc) | (ya > hi[1] - 8 * dxc)
        if dim == 3:
            za = np.asarray(ad["index", "z"])
            near |= (za < lo[2] + 8 * dxc) | (za > hi[2] - 8 * dxc)
        if near.any():
            row["dp_bdry_max"] = float(np.abs(p[near] - pinf).max())

    # --- band per-phase mass (snowplow) ---
    re1 = fld("rho_eta1"); re0 = fld("rho_eta0")
    band = (eta > 0.01) & (eta < 0.99)
    if band.any() and re1 is not None:
        row["re1_band_max"] = float(re1[band].max()) / max(re10, 1e-300)
        row["re0_band_max"] = float(re0[band].max()) / max(re00, 1e-300)

    # --- shell / cfun rot (mechanism 3) ---
    sh = fld("shell")
    if sh is not None:
        row["shell_min"] = float(np.nanmin(sh))
        row["shell_max"] = float(np.nanmax(sh))
        row["shell_nan"] = int(np.isnan(sh).sum())

    # --- bubble radius (auto symmetry factor: axes whose lo edge is center) ---
    nsym = 0
    dle = [float(v) for v in ds.domain_left_edge.to_value()]
    for d in range(dim):
        if abs(dle[d] - center[d]) < 1e-12 + 1e-6 * abs(center[d] or 1.0):
            nsym += 1
    V = float(((1.0 - eta) * vol).sum()) * (2 ** nsym)
    if dim == 3:
        row["R_vol"] = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    else:
        row["R_vol"] = (V / np.pi) ** 0.5
    return row


def analyze_drive(pfs, amp, omega, phase, pinf, every):
    """Boundary drive-response probe: does the temporal drive p_target(t) =
    pinf + amp*sin(omega t + phase) actually express on the HI FACES, or only
    along the hi-hi EDGES/axes?  Samples first-interior-layer pressure at:
      face_x/y/z : central 50% patch of each hi face (pure-face response)
      edge_xy/xz/yz : along each hi-hi edge (double-NSCBC coupled treatment)
      corner     : the hi-hi-hi corner cell
      lo_faces   : central patch of the reflective lo faces (should follow
                   the interior, NOT the drive directly)
      origin     : cell nearest (0,0,0) -- the mech-5 vacuum site
    Ends with a sin/cos least-squares fit per probe: response amplitude as a
    FRACTION of the drive amplitude and phase lag [rad].  A healthy driven
    face shows ratio ~O(1) at the drive frequency; ratio << edges = the
    face LODI is under-driven (the user's axis-only hypothesis)."""
    picked = pfs[::max(1, every or max(1, len(pfs) // 40))]
    rows = []
    print(f"\n=== DRIVE PROBE: target = pinf + {amp:g} sin({omega:g} t + {phase:g}) ===")
    hdr = (f"{'t':>11s} {'target-pinf':>11s} {'face_x':>9s} {'face_y':>9s} {'face_z':>9s} "
           f"{'edge_xy':>9s} {'corner':>9s} {'lo_face':>9s} {'origin':>9s}")
    print(hdr); print("-" * len(hdr))
    for pf in picked:
        try:
            ds = yt.load(pf); ad = ds.all_data()
            t = float(ds.current_time)
            p = np.asarray(ad["boxlib", "pressure"]) - pinf
            dim = int(ds.dimensionality)
            lo = [float(v) for v in ds.domain_left_edge.to_value()]
            hi = [float(v) for v in ds.domain_right_edge.to_value()]
            L = [hi[d] - lo[d] for d in range(dim)]
            dxc = L[0] / int(ds.domain_dimensions[0])
            X = [np.asarray(ad["index", c]) for c in ("x", "y", "z")[:dim]]
            if dim == 2:
                X.append(np.zeros_like(X[0])); lo.append(0.0); hi.append(1.0); L.append(1.0)

            def near_hi(d):    return X[d] > hi[d] - 1.5 * dxc
            def near_lo(d):    return X[d] < lo[d] + 1.5 * dxc
            def central(d):    return (X[d] > lo[d] + 0.25 * L[d]) & (X[d] < lo[d] + 0.75 * L[d])

            row = {"t": t, "tgt": amp * np.sin(omega * t + phase)}
            m = near_hi(0) & central(1) & (central(2) if dim == 3 else True)
            row["face_x"] = float(p[m].mean()) if m.any() else np.nan
            m = near_hi(1) & central(0) & (central(2) if dim == 3 else True)
            row["face_y"] = float(p[m].mean()) if m.any() else np.nan
            if dim == 3:
                m = near_hi(2) & central(0) & central(1)
                row["face_z"] = float(p[m].mean()) if m.any() else np.nan
            else:
                row["face_z"] = np.nan
            m = near_hi(0) & near_hi(1) & (central(2) if dim == 3 else True)
            row["edge_xy"] = float(p[m].mean()) if m.any() else np.nan
            m = near_hi(0) & near_hi(1) & (near_hi(2) if dim == 3 else True)
            row["corner"] = float(p[m].mean()) if m.any() else np.nan
            m = near_lo(0) & central(1) & (central(2) if dim == 3 else True)
            row["lo_face"] = float(p[m].mean()) if m.any() else np.nan
            r2 = sum(x * x for x in X[:dim])
            row["origin"] = float(p[int(np.argmin(r2))])
            rows.append(row)
            print(f"{t:11.4e} {row['tgt']:11.3e} {row['face_x']:9.2e} {row['face_y']:9.2e} "
                  f"{row['face_z']:9.2e} {row['edge_xy']:9.2e} {row['corner']:9.2e} "
                  f"{row['lo_face']:9.2e} {row['origin']:9.2e}")
        except Exception as e:
            print(f"{os.path.basename(pf)}: FAIL {e}")
    # sin/cos LSQ fit per probe -> amplitude ratio + phase lag vs drive
    if len(rows) >= 8:
        ts = np.array([r["t"] for r in rows])
        B = np.column_stack([np.sin(omega * ts + phase), np.cos(omega * ts + phase),
                             np.ones_like(ts)])
        print(f"\n{'probe':>9s} {'resp/drive':>11s} {'phase_lag':>10s}   (fit over {len(rows)} frames)")
        for k in ("face_x", "face_y", "face_z", "edge_xy", "corner", "lo_face", "origin"):
            yv = np.array([r[k] for r in rows])
            ok = np.isfinite(yv)
            if ok.sum() < 6:
                continue
            c, *_ = np.linalg.lstsq(B[ok], yv[ok], rcond=None)
            A = float(np.hypot(c[0], c[1])); lag = float(np.arctan2(-c[1], c[0]))
            print(f"{k:>9s} {A/amp:11.3f} {lag:10.3f}")
        print("interpretation: driven hi faces should sit near resp/drive ~ O(1);")
        print("  faces << edges  -> face LODI under-driven (axis-only drive);")
        print("  lo_face/origin follow the interior response, lagged.")


def analyze_log(path):
    """dt history + decay onset + abort lines."""
    dts, times, steps = [], [], []
    aborts = []
    step_re = re.compile(r"STEP (\d+) ends\. TIME = ([0-9.eE+-]+) DT = ([0-9.eE+-]+)")
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = step_re.search(line)
            if m:
                steps.append(int(m.group(1)))
                times.append(float(m.group(2)))
                dts.append(float(m.group(3)))
            elif ("ABORT" in line or "EXCEPTION" in line or "contains nan" in line
                  or "ERROR IN RIEMANN" in line):
                aborts.append(line.strip()[:140])
    print(f"\n=== LOG: {len(steps)} steps to t={times[-1] if times else 0:.4e} ===")
    if dts:
        ref = float(np.median(dts[: min(1000, len(dts))]))
        # skip the dynamic-timestep startup ramp when hunting the decay onset
        skip = min(500, len(dts) // 4)
        onset = next((i for i in range(skip, len(dts)) if dts[i] < 0.5 * ref), None)
        for i in range(0, len(dts), max(1, len(dts) // 12)):
            print(f"  step {steps[i]:>8d}  t={times[i]:.4e}  dt={dts[i]:.3e}")
        print(f"  step {steps[-1]:>8d}  t={times[-1]:.4e}  dt={dts[-1]:.3e}")
        if onset is not None:
            print(f"  >>> dt DECAY ONSET: step {steps[onset]} t={times[onset]:.5e} "
                  f"(dt {dts[onset]:.2e} < half of early median {ref:.2e})")
        else:
            print(f"  dt healthy throughout (early median {ref:.2e})")
    for a in aborts[:6]:
        print("  ABORT-CONTEXT:", a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--log", default=None)
    ap.add_argument("--every", type=int, default=0, help="use every Nth plotfile (0=auto ~15 rows)")
    ap.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--pinf", type=float, default=101325.0)
    ap.add_argument("--drive", type=float, nargs="+", default=None,
                    metavar="AMP OMEGA [PHASE]",
                    help="probe boundary drive response: AMP OMEGA [PHASE]; "
                         "e.g. --drive 1.0e4 512.35")
    args = ap.parse_args()

    pfs = list_plotfiles(args.out_dir)
    if not pfs:
        print("no plotfiles found in", args.out_dir)
        if args.log:
            analyze_log(args.log)
        sys.exit(1)
    step = args.every or max(1, len(pfs) // 15)
    picked = pfs[::step] + ([pfs[-1]] if pfs[-1] not in pfs[::step] else [])

    # t=0 band reference for snowplow normalization
    re10 = re00 = 1.0
    try:
        ds0 = yt.load(pfs[0]); ad0 = ds0.all_data()
        eta0 = np.asarray(ad0["boxlib", "eta"])
        b0 = (eta0 > 0.01) & (eta0 < 0.99)
        if b0.any():
            re10 = float(np.asarray(ad0["boxlib", "rho_eta1"])[b0].max())
            re00 = float(np.asarray(ad0["boxlib", "rho_eta0"])[b0].max())
    except Exception:
        pass

    hdr = (f"{'t':>11s} {'p_gas_min':>10s} {'@r':>8s} {'u_gas':>8s} {'dp_bdry':>9s} "
           f"{'re1/re1_0':>9s} {'shell mn/mx':>13s} {'NaN':>4s} {'R_vol':>9s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for pf in picked:
        try:
            rw = analyze_frame(pf, args.center, args.pinf, re10, re00)
            rows.append(rw)
            print(f"{rw['t']:11.4e} {rw.get('p_gas_min', float('nan')):10.3e} "
                  f"{rw.get('r_at_pmin', float('nan')):8.5f} {rw.get('u_gas_max', float('nan')):8.2e} "
                  f"{rw.get('dp_bdry_max', float('nan')):9.2e} {rw.get('re1_band_max', float('nan')):9.3f} "
                  f"{rw.get('shell_min', float('nan')):6.3f}/{rw.get('shell_max', float('nan')):6.3f} "
                  f"{rw.get('shell_nan', 0):4d} {rw.get('R_vol', float('nan')):9.5f}")
        except Exception as e:
            print(f"{os.path.basename(pf)}: FRAME FAIL {e}")

    # --- fingerprint verdicts ---
    flags = []
    def series(k):
        return [rw[k] for rw in rows if k in rw]
    pg = series("p_gas_min")
    if len(pg) > 3 and min(pg) < 0.5 * args.pinf:
        rmin = [rw["r_at_pmin"] for rw in rows if "p_gas_min" in rw][int(np.argmin(pg))]
        flags.append(f"CENTER DRAIN (mech 5?): p_gas_min fell to {min(pg):.2e} at r={rmin:.4f}")
    db = series("dp_bdry_max")
    if len(db) > 4 and db[-1] > 50 * (abs(db[1]) + 1e-30) and db[-1] > 100:
        flags.append(f"BOUNDARY GROWTH (mech 1?): dp_bdry {db[1]:.2e} -> {db[-1]:.2e}")
    rb = series("re1_band_max")
    if rb and max(rb) > 2.0:
        flags.append(f"BAND MASS PILE (snowplow?): re1_band up {max(rb):.2f}x")
    sn = series("shell_nan")
    sm = series("shell_min")
    if (sn and max(sn) > 0) or (sm and (min(sm) < 0.5 or max(series('shell_max')) > 2.0)):
        flags.append("SHELL ROT (mech 3?): shell NaN or drifted far from 1")
    print()
    if flags:
        for fl in flags:
            print("FLAG:", fl)
    else:
        print("no fingerprint flags fired")

    if args.drive:
        amp = args.drive[0]
        omega = args.drive[1] if len(args.drive) > 1 else 512.35
        phase = args.drive[2] if len(args.drive) > 2 else 0.0
        analyze_drive(pfs, amp, omega, phase, args.pinf, args.every)

    if args.log:
        analyze_log(args.log)
    sys.exit(2 if flags else 0)


if __name__ == "__main__":
    main()
