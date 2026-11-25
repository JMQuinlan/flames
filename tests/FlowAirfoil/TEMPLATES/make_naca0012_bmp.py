#!/usr/bin/env python3
"""
Standalone NACA 4-digit solid-indicator bitmap for Hydro2 (IC::BMP), rotated to an
angle of attack.  Used by TEMPLATES/sweep_aoa.bash; no other project code needed.

  python3 make_naca0012_bmp.py <aoa_deg> <out.bmp> [naca4=0012]

  naca4 = "MPTT": max camber M % chord at P/10 chord, thickness TT % chord
  (e.g. 0012, 0008, 0002, 4402).  Default 0012 (the original sweep).

Writes <out.bmp> (phi = 0 inside the airfoil -> black, 1 outside -> white; the
sweep deck reads the green channel) and prints ONE line to stdout:

  BMP_LO_X BMP_LO_Y BMP_HI_X BMP_HI_Y PROBE_X(4, space-separated) PROBE_Y(4)

  * geometry: NACA 4-digit section (standard thickness form with a closed
    trailing edge, last coefficient -0.1036; camber line of Abbott & von
    Doenhoff with the thickness applied normal to it), chord 1, centred on
    mid-chord (x in [-0.5, 0.5]), rotated nose-up by +aoa about the origin
  * bitmap bounds: bounding box of the rotated airfoil + 0.02 margin (the IC
    clamps to the edge pixel = fluid outside the bitmap, so the crop must
    contain the whole body -- a fixed +-0.22 crop clips it above ~15 deg)
  * pixel size 0.0005 (much finer than the finest L4/L5 cells); IC::BMP maps
    [lo, hi] onto pixels [0, N-1], i.e. an INCLUSIVE linspace (see the
    2026-09-24 note in reference/viscous_naca0012.py)
  * probes in the body frame, rotated with the airfoil: ahead of the LE, over
    the upper surface at mid-chord, behind the TE, and in the wake
Requires numpy, matplotlib (Path) and pillow.
"""
import math, sys
import numpy as np

BMP_DX = 0.0005
import os as _os
BMP_DX = float(_os.environ.get("BMP_DX", BMP_DX))   # [2026-10-02] override, e.g. BMP_DX=1e-4 for L6 + supersample 8
MARGIN = 0.02


def naca4(code="0012", n=400):
    m = int(code[0]) / 100.0; p = int(code[1]) / 10.0; t = int(code[2:]) / 100.0
    xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1036 * xc**4)
    if m > 0 and p > 0:
        yc = np.where(xc < p, m / p**2 * (2 * p * xc - xc**2), m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * xc - xc**2))
        dy = np.where(xc < p, 2 * m / p**2 * (p - xc), 2 * m / (1 - p)**2 * (p - xc))
    else:
        yc = np.zeros_like(xc); dy = np.zeros_like(xc)
    th = np.arctan(dy)
    xu, yu = xc - yt * np.sin(th), yc + yt * np.cos(th)
    xl, yl = xc + yt * np.sin(th), yc - yt * np.cos(th)
    up = np.column_stack([xu - 0.5, yu]); lo = np.column_stack([xl[::-1] - 0.5, yl[::-1]])
    return np.vstack([up, lo[1:-1]])


def naca0012(n=400):          # kept for backward compatibility
    return naca4("0012", n)


def rotate(pts, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return pts @ np.array([[c, s], [-s, c]]).T          # +deg = nose-up


def main():
    aoa = float(sys.argv[1]); out = sys.argv[2]
    code = sys.argv[3] if len(sys.argv) > 3 else "0012"
    assert len(code) == 4 and code.isdigit(), "naca4 must be 4 digits, e.g. 0012"
    from matplotlib.path import Path
    from PIL import Image
    poly = rotate(naca4(code), aoa)
    lo = np.floor((poly.min(axis=0) - MARGIN) * 100) / 100
    hi = np.ceil((poly.max(axis=0) + MARGIN) * 100) / 100
    nx = int(round((hi[0] - lo[0]) / BMP_DX)); ny = int(round((hi[1] - lo[1]) / BMP_DX))
    x = np.linspace(lo[0], hi[0], nx); y = np.linspace(lo[1], hi[1], ny)
    X, Y = np.meshgrid(x, y)
    inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    g = np.where(inside, 0, 255).astype(np.uint8)[::-1]   # image row 0 = top (max y)
    Image.fromarray(np.dstack([g, g, g]), "RGB").save(out)
    probes = rotate(np.array([[-0.52, 0.0], [0.0, 0.075], [0.52, 0.0], [1.0, 0.0]]), aoa)
    print(f"{lo[0]:.2f} {lo[1]:.2f} {hi[0]:.2f} {hi[1]:.2f} "
          + " ".join(f"{p:.5f}" for p in probes[:, 0]) + " " + " ".join(f"{p:.5f}" for p in probes[:, 1]))


if __name__ == "__main__":
    main()
