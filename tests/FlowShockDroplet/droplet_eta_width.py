#!/usr/bin/env python3
"""
Track the diffuse-interface (eta) width at the droplet's four CARDINAL points
as a function of time, for the Droplet-advection / Droplet-stationary cases.

Cardinal points (rays cast from the droplet centre outward):
    E = +x,  W = -x,  N = +y,  S = -y.

WIDTH metric = the distance along the ray between the eta = lo and eta = hi
contours (default lo=0.02, hi=0.98 -> the "2-98%" width). eta ~ 0 inside the
liquid (near the centre) and ~ 1 in the gas, so along an outward ray eta rises
through lo then hi; width = r(hi) - r(lo).

The code's equilibrium profile is  eta = 0.5*(1 + tanh((r-R0)/(sqrt(2) eps))),
whose analytic 2-98% width is  (atanh(2 hi -1) - atanh(2 lo -1)) * sqrt(2) * eps
= 3.8918 * sqrt(2) * eps  for the default thresholds. That is drawn as a
reference line when --eps is given.

The droplet centre is found per-frame as the (1-eta)-weighted, cell-volume-
weighted centroid (AMR-correct via yt), so it works for both the moving
(advection) and stationary cases. Pass --vx/--x0/--y0 to override with the exact
analytic centre  (x0 + vx*t, y0)  instead.

Usage:
    python droplet_eta_width.py <plotfile_dir> [options]
    python droplet_eta_width.py A=<advection_dir> B=<stationary_dir>   # overlay two runs

Options:
    --lo 0.02 --hi 0.98     contour thresholds
    --eps 8e-5              epsilon (draws the analytic equilibrium width line)
    --R0 4.4e-3             droplet radius (sets ray length L = 3*R0)
    --L  <m>                explicit ray length (overrides 3*R0)
    --vx <m/s> --x0 --y0    analytic centre override (skip the centroid)
    --out <prefix>          output prefix (default: <dir>/eta_width)

<plotfile_dir> holds the AMReX plot directories (00000cell, 00050cell, ...).
"""
import os
import sys
import glob
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt

yt.set_log_level(50)  # quiet

CARDINALS = [("E", (1.0, 0.0)), ("W", (-1.0, 0.0)),
             ("N", (0.0, 1.0)), ("S", (0.0, -1.0))]


def find_plotfiles(d):
    pfs = sorted(glob.glob(os.path.join(d, "*cell")))
    if not pfs:
        sys.exit(f"ERROR: no '*cell' plot directories found in {d}")
    return pfs


def rf(obj, name):
    """Read a field by short name, falling back to the ('boxlib', name) tuple."""
    try:
        return np.array(obj[name])
    except Exception:
        return np.array(obj[("boxlib", name)])


def droplet_center(ds):
    """(1-eta)*cell_volume-weighted centroid of the liquid (eta ~ 0)."""
    ad = ds.all_data()
    eta = rf(ad, "eta")
    x = rf(ad, "x")
    y = rf(ad, "y")
    vol = rf(ad, "cell_volume")
    w = np.clip(1.0 - eta, 0.0, 1.0) * vol
    W = w.sum()
    if W <= 0.0:
        return 0.0, 0.0
    return float((x * w).sum() / W), float((y * w).sum() / W)


def ray_profile(ds, cx, cy, dirvec, L):
    """eta vs distance-from-centre along an outward ray; sorted by distance."""
    dx, dy = dirvec
    start = ds.arr([cx, cy, 0.0], "code_length")
    end = ds.arr([cx + dx * L, cy + dy * L, 0.0], "code_length")
    ray = ds.ray(start, end)
    rx = rf(ray, "x")
    ry = rf(ray, "y")
    s = np.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
    e = rf(ray, "eta")
    order = np.argsort(s)
    return s[order], e[order]


def first_crossing(s, e, level):
    """Distance s where e first crosses `level` (linear interp), else nan."""
    for i in range(len(e) - 1):
        a, b = e[i], e[i + 1]
        if (a - level) * (b - level) <= 0.0 and a != b:
            t = (level - a) / (b - a)
            return s[i] + t * (s[i + 1] - s[i])
    return float("nan")


def width_along_ray(s, e, lo, hi):
    r_lo = first_crossing(s, e, lo)
    r_hi = first_crossing(s, e, hi)
    return abs(r_hi - r_lo)


def analyze_run(plotdir, args):
    pfs = find_plotfiles(plotdir)
    times = []
    widths = {k: [] for k, _ in CARDINALS}
    print(f"\n=== {plotdir}  ({len(pfs)} frames) ===")
    for pf in pfs:
        ds = yt.load(pf)
        t = float(ds.current_time)
        if args.vx is not None:
            cx, cy = args.x0 + args.vx * t, args.y0
        else:
            cx, cy = droplet_center(ds)
        row = []
        for k, dvec in CARDINALS:
            s, e = ray_profile(ds, cx, cy, dvec, args.L)
            w = width_along_ray(s, e, args.lo, args.hi)
            widths[k].append(w)
            row.append(w)
        times.append(t)
        print(f"  t={t:.3e}s  c=({cx:+.4e},{cy:+.4e})  "
              + "  ".join(f"{k}={w*1e6:7.2f}um" for (k, _), w in zip(CARDINALS, row)))
    times = np.array(times)
    for k in widths:
        widths[k] = np.array(widths[k])
    return times, widths


def summarize(label, times, widths):
    print(f"\n--- summary [{label}] (2-98% eta width) ---")
    for k, _ in CARDINALS:
        w = widths[k]
        w0 = w[np.isfinite(w)][0] if np.isfinite(w).any() else float("nan")
        w1 = w[np.isfinite(w)][-1] if np.isfinite(w).any() else float("nan")
        growth = 100.0 * (w1 - w0) / w0 if w0 else float("nan")
        print(f"  {k}: {w0*1e6:7.2f} -> {w1*1e6:7.2f} um   ({growth:+6.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+",
                    help="plotfile dir, or A=<dir> B=<dir> to overlay two runs")
    ap.add_argument("--lo", type=float, default=0.02)
    ap.add_argument("--hi", type=float, default=0.98)
    ap.add_argument("--eps", type=float, default=8.0e-5)
    ap.add_argument("--R0", type=float, default=4.4e-3)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--vx", type=float, default=None)
    ap.add_argument("--x0", type=float, default=0.0)
    ap.add_argument("--y0", type=float, default=0.0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.L is None:
        args.L = 3.0 * args.R0

    # Parse "LABEL=dir" or bare dir.
    runs = []
    for d in args.dirs:
        if "=" in d and not os.path.isdir(d):
            label, path = d.split("=", 1)
        else:
            label, path = os.path.basename(os.path.normpath(d)), d
        runs.append((label, path))

    out = args.out or "eta_width"  # written to the current directory by default

    # Analytic equilibrium 2-98% width for the reference line.
    w_eq = (math.atanh(2 * args.hi - 1) - math.atanh(2 * args.lo - 1)) * math.sqrt(2) * args.eps

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = ["-", "--", "-.", ":"]
    colors = {"E": "C0", "W": "C1", "N": "C2", "S": "C3"}

    for ri, (label, path) in enumerate(runs):
        times, widths = analyze_run(path, args)
        summarize(label, times, widths)
        # CSV per run
        csv = np.column_stack([times] + [widths[k] for k, _ in CARDINALS])
        header = "time[s] " + " ".join(f"width_{k}[m]" for k, _ in CARDINALS)
        csv_path = (out + f"_{label}.csv") if len(runs) > 1 else (out + ".csv")
        np.savetxt(csv_path, csv, header=header)
        print(f"  wrote {csv_path}")
        for k, _ in CARDINALS:
            lbl = f"{label}:{k}" if len(runs) > 1 else k
            ax.plot(times * 1e6, widths[k] * 1e6,
                    styles[ri % len(styles)], color=colors[k], label=lbl)

    ax.axhline(w_eq * 1e6, color="k", lw=1.0, ls=":",
               label=f"equilibrium {w_eq*1e6:.1f}um")
    ax.set_xlabel("time [us]")
    ax.set_ylabel("2-98% eta width [um]")
    ax.set_title("Diffuse-interface width at cardinal points")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out + ".png"
    fig.savefig(png, dpi=140)
    print(f"\nwrote {png}")


if __name__ == "__main__":
    main()
