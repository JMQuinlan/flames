#!/usr/bin/env python3
"""
Stefan-problem post-processor for input_stefan_sqr / input_stefan_ch.

Extracts the interface position x_int(t) from each AMReX plotfile by locating
the eta = 0.5 contour along a horizontal slice (the y direction is essentially
trivial in the 1D-ish setup), then compares to the self-similar analytical
solution

    delta(t) = xhi - x_int(t) = 2 * beta * sqrt(alpha_g * (t + t_off))

Usage
-----
    ./stefan_postprocess.py <plotfile_root>            # e.g. output_stefan_sqr
    ./stefan_postprocess.py <root_sqr> <root_ch>       # compare both modes

Requires yt (https://yt-project.org/).
"""
import glob
import os
import sys
import math

import numpy as np

try:
    import yt
except ImportError:
    sys.exit("ERROR: yt is required. pip install yt")

# ---------------------------------------------------------------------------
# Stefan analytical parameters — must match the input files
# ---------------------------------------------------------------------------
BETA    = 0.058
ALPHA_G = 1.21e-3      # m^2/s   (matches input m0.ic.expression.constant.alpha_g)
T_OFF   = 5.0e-5       # s       (matches input m0.ic.expression.constant.t_off)
XHI     = 1.0e-3       # m       (matches geometry.prob_hi[0])

def delta_analytical(t):
    """Vapor-layer thickness delta(t) = xhi - x_int(t)."""
    return 2.0 * BETA * math.sqrt(ALPHA_G * (t + T_OFF))

def x_int_analytical(t):
    return XHI - delta_analytical(t)

# ---------------------------------------------------------------------------
# Interface extraction
# ---------------------------------------------------------------------------
def interface_x(plotfile):
    """Return (time, x_int) where eta crosses 0.5 along the y-midline."""
    ds = yt.load(plotfile)
    t  = float(ds.current_time)

    # Take a ray along x at the y-midpoint of the domain.
    ymid = 0.5 * (ds.domain_left_edge[1] + ds.domain_right_edge[1])
    ray  = ds.r[ds.domain_left_edge[0]:ds.domain_right_edge[0]:ds.domain_dimensions[0]*1j,
                ymid:ymid:1j, 0:0:1j]

    x   = np.array(ray["index", "x"])
    eta = np.array(ray["boxlib", "eta"])

    order = np.argsort(x)
    x, eta = x[order], eta[order]

    # Find the cell pair straddling eta = 0.5; linearly interpolate.
    crossing = np.where(np.diff(np.sign(eta - 0.5)) != 0)[0]
    if len(crossing) == 0:
        return t, float("nan")
    i = crossing[0]
    e0, e1 = eta[i], eta[i+1]
    x0, x1 = x[i],   x[i+1]
    x_int  = x0 + (0.5 - e0) * (x1 - x0) / (e1 - e0)
    return t, float(x_int)

def trajectory(root):
    plotfiles = sorted(glob.glob(f"{root}*"))
    plotfiles = [p for p in plotfiles if os.path.isdir(p)]
    if not plotfiles:
        sys.exit(f"No plotfiles found at {root}*")
    times, xs = [], []
    for pf in plotfiles:
        try:
            t, x = interface_x(pf)
        except Exception as e:
            print(f"  skip {pf}: {e}", file=sys.stderr)
            continue
        if math.isnan(x):
            continue
        times.append(t)
        xs.append(x)
    return np.array(times), np.array(xs)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(label, t, x):
    print(f"\n=== {label} ===")
    print(f"  {'t [s]':>10}  {'x_int':>12}  {'analytical':>12}  {'rel err':>10}")
    for ti, xi in zip(t, x):
        x_an = x_int_analytical(ti)
        rel  = abs(xi - x_an) / max(abs(x_an), 1e-30)
        print(f"  {ti:10.4e}  {xi:12.6e}  {x_an:12.6e}  {rel:10.3e}")

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for root in sys.argv[1:]:
        t, x = trajectory(root)
        report(root, t, x)

if __name__ == "__main__":
    main()
