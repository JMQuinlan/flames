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

    # Read all cells and average eta over y/z at each unique x position.
    ad      = ds.all_data()
    x_all   = np.array(ad["index", "x"])
    eta_all = np.array(ad["boxlib", "eta"])

    x_unique = np.unique(x_all)
    eta_mean = np.array([eta_all[x_all == xi].mean() for xi in x_unique])
    x, eta = x_unique, eta_mean

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
    plotfiles = glob.glob(f"{root}*")
    plotfiles = [p for p in plotfiles if os.path.isdir(p)]
    if not plotfiles:
        sys.exit(f"No plotfiles found at {root}*")
    pairs = []
    for pf in plotfiles:
        try:
            t, x = interface_x(pf)
        except Exception as e:
            print(f"  skip {pf}: {e}", file=sys.stderr)
            continue
        if math.isnan(x):
            continue
        pairs.append((t, x))
    pairs.sort(key=lambda tx: tx[0])
    if not pairs:
        return np.array([]), np.array([])
    times, xs = zip(*pairs)
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

# Output from the initial run 20260425 MQ
#
# === ./output_stefan_sqr/ ===
#        t [s]         x_int    analytical     rel err
#   0.0000e+00  9.097058e-04  9.714678e-04   6.358e-02
#   5.0000e-06  9.096548e-04  9.700751e-04   6.228e-02
#   7.5000e-05  9.096629e-04  9.548865e-04   4.736e-02
#   8.0000e-05  9.096633e-04  9.539931e-04   4.647e-02
#   8.5000e-05  9.096637e-04  9.531168e-04   4.559e-02
#   9.0000e-05  9.096640e-04  9.522565e-04   4.473e-02
#   9.5000e-05  9.096644e-04  9.514113e-04   4.388e-02
#   1.0000e-04  9.096647e-04  9.505808e-04   4.304e-02
#   9.9998e-06  9.096499e-04  9.687446e-04   6.100e-02
#   1.5000e-05  9.096568e-04  9.674682e-04   5.976e-02
#   2.0000e-05  9.096561e-04  9.662402e-04   5.856e-02
#   2.5000e-05  9.096572e-04  9.650553e-04   5.740e-02
#   3.0000e-05  9.096582e-04  9.639093e-04   5.628e-02
#   3.5000e-05  9.096589e-04  9.627985e-04   5.519e-02
#   4.0000e-05  9.096596e-04  9.617200e-04   5.413e-02
#   4.5000e-05  9.096602e-04  9.606711e-04   5.310e-02
#   5.0000e-05  9.096607e-04  9.596493e-04   5.209e-02
#   5.5000e-05  9.096612e-04  9.586529e-04   5.110e-02
#   6.0000e-05  9.096617e-04  9.576798e-04   5.014e-02
#   6.5000e-05  9.096621e-04  9.567287e-04   4.920e-02
#   7.0000e-05  9.096625e-04  9.557981e-04   4.827e-02
# 
# === ./output_stefan_ch/ ===
#        t [s]         x_int    analytical     rel err
#   0.0000e+00  9.097058e-04  9.714678e-04   6.358e-02
#   5.0000e-06  9.095660e-04  9.700751e-04   6.238e-02
#   7.5000e-05  9.096845e-04  9.548865e-04   4.734e-02
#   8.0000e-05  9.096854e-04  9.539931e-04   4.644e-02
#   8.5000e-05  9.096862e-04  9.531168e-04   4.557e-02
#   9.0000e-05  9.096870e-04  9.522565e-04   4.470e-02
#   9.5000e-05  9.096877e-04  9.514113e-04   4.385e-02
#   1.0000e-04  9.096883e-04  9.505807e-04   4.302e-02
#   9.9998e-06  9.096128e-04  9.687446e-04   6.104e-02
#   1.5000e-05  9.096535e-04  9.674682e-04   5.976e-02
#   2.0000e-05  9.096564e-04  9.662402e-04   5.856e-02
#   2.5000e-05  9.096630e-04  9.650553e-04   5.740e-02
#   3.0000e-05  9.096677e-04  9.639093e-04   5.627e-02
#   3.5000e-05  9.096710e-04  9.627985e-04   5.518e-02
#   4.0000e-05  9.096738e-04  9.617200e-04   5.412e-02
#   4.5000e-05  9.096760e-04  9.606711e-04   5.308e-02
#   5.0000e-05  9.096780e-04  9.596494e-04   5.207e-02
#   5.5000e-05  9.096796e-04  9.586528e-04   5.109e-02
#   6.0000e-05  9.096811e-04  9.576798e-04   5.012e-02
#   6.5000e-05  9.096823e-04  9.567287e-04   4.917e-02
#   7.0000e-05  9.096835e-04  9.557981e-04   4.825e-02
