#!/usr/bin/env python3
"""
make_cattaneo2025_fig1cd -- builds cattaneo2025_fig1cd.npz, the bright-field frames of Cattaneo et al. (2025)
Fig. 1c (60 kPa, 'lo') and Fig. 1d (160 kPa, 'hi') used by cattaneo_06_fig1.py.

Source: M. Cattaneo et al., Nature Physics 21:590 (2025), https://doi.org/10.1038/s41567-025-02785-0, Fig. 1c,d.
Open Access, CC BY 4.0: reuse is allowed with attribution and an indication of changes (the figures say
"adapted from Cattaneo et al. (2025), Fig. 1c/d, CC BY 4.0"; the frames are only cropped out, not altered).

What is stored, per case ('lo', 'hi'):
  <case>_img   (12, 240, 240) uint8   the twelve bright-field frames, top row 3.5 .. 6.0 us then bottom row
                                      9.5 .. 12.0 us (0.5 us apart), exactly as embedded in the PDF (no resampling)
  <case>_t     (12,) us               frame times on the paper axis (Fig. 1 labels; the same axis as Fig. 2a)
  <case>_ctr   (12, 2) px             bubble centre (column, row from the top) found by a ring template match
  <case>_rad   (12,) px               radius of that ring (a check against Fig. 2a; not used for scaling)
  px_per_um    12.0                   from the 10 um scale bar (480 px at 1200 dpi = 120 image px) -> 20 um frames
The PDF images are mapped to their panel positions by matching each one against a 300 dpi render of page 3
(mean |difference| ~1-3 grey levels for the right one, >= 4.5 for every other one); the table below is that result
and is checked again here.

Bubble centre: the frame is high-passed (Gaussian sigma 1 px minus sigma 30 px), then correlated with a dark ring
(0.55 r .. r) inside a bright annulus (r .. 1.35 r) for r = 8 .. 65 px; the best (centre, r) wins, the centre limited
to 30 px from the previous frame of the same series (the bubble drifts slowly; the 160 kPa bubble lifts away from
the cell late in the record).  60 kPa radii agree with the Fig. 2a video radii within ~0.15 um.

    python3 make_cattaneo2025_fig1cd.py [PATH/s41567-025-02785-0.pdf]      (needs pdfimages, pdftoppm)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.signal import fftconvolve

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "cattaneo2025_fig1cd.npz")
PAGE = 3
PX_PER_UM = 12.0
TIMES = np.r_[np.arange(3.5, 6.01, 0.5), np.arange(9.5, 12.01, 0.5)]
# pdfimages index of each frame (poppler order on page 3) and its top-left corner in the 300 dpi page render
IDX = {"lo": [12, 22, 20, 18, 16, 14, 0, 10, 8, 6, 4, 2],
       "hi": [46, 44, 40, 42, 38, 36, 28, 26, 30, 32, 24, 34]}
COLS = [537, 776, 1015, 1256, 1495, 1735]
ROWS = {"lo": (1429, 1671), "hi": (2075, 2317)}


def _ring_centre(a, prev, radii=np.arange(8, 66, 1.0), win=30):
    b = ndi.gaussian_filter(a, 1.0) - ndi.gaussian_filter(a, 30.0)
    H, W = a.shape
    yy, xx = np.mgrid[:H, :W]
    best = (-np.inf, 0, 0, 0.0)
    for r in radii:
        n = int(1.5 * r) + 2
        y, x = np.mgrid[-n:n + 1, -n:n + 1]
        d = np.hypot(x, y)
        ri, ro = 0.55 * r, 1.35 * r + 2
        k = (np.where((d <= r) & (d > ri), -1.0, 0.0) / (np.pi * (r * r - ri * ri))
             + np.where((d > r + 1) & (d <= ro), 1.0, 0.0) / (np.pi * (ro * ro - (r + 1) ** 2)))
        acc = fftconvolve(b, k[::-1, ::-1], mode="same")
        acc[:25] = acc[-25:] = -np.inf
        acc[:, :25] = acc[:, -25:] = -np.inf
        if prev is not None:
            acc[(yy - prev[1]) ** 2 + (xx - prev[0]) ** 2 > win ** 2] = -np.inf
        i = int(np.argmax(acc))
        if acc.flat[i] > best[0]:
            best = (acc.flat[i], i % W, i // W, r)
    return best[1:]


def main(pdf):
    out = dict(px_per_um=PX_PER_UM,
               source="Cattaneo et al., Nature Physics 21:590 (2025), doi:10.1038/s41567-025-02785-0, Fig. 1c (lo) "
                      "and 1d (hi); CC BY 4.0")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdfimages", "-f", str(PAGE), "-l", str(PAGE), "-png", pdf, os.path.join(tmp, "im")], check=True)
        subprocess.run(["pdftoppm", "-f", str(PAGE), "-l", str(PAGE), "-r", "300", "-png", "-singlefile", pdf,
                        os.path.join(tmp, "page")], check=True)
        pg = np.asarray(Image.open(os.path.join(tmp, "page.png")).convert("L"), float)
        for case, idx in IDX.items():
            imgs, ctr, rad = [], [], []
            prev = None
            for k, i in enumerate(idx):
                a = np.asarray(Image.open(os.path.join(tmp, "im-%03d.png" % i)).convert("L"))
                a = a[:240, :240]
                # check: the frame sits at its panel position (labels / scale bar excluded from the comparison)
                y0, x0 = ROWS[case][k // 6], COLS[k % 6]
                err = min(np.abs(pg[y0 + dy + 40:y0 + dy + 180, x0 + dx + 10:x0 + dx + 230] - a[40:180, 10:230]).mean()
                          for dy in range(-3, 4) for dx in range(-3, 4))
                if err > 4.0:
                    sys.exit("frame %s %.1f us (pdf image %d) does not match the page (|diff| %.1f)" % (case, TIMES[k], i, err))
                cx, cy, r = _ring_centre(a.astype(float), prev)
                prev = (cx, cy)
                imgs.append(a); ctr.append((cx, cy)); rad.append(r)
                print("%s %5.1f us  image %2d  |diff| %.2f  centre (%3d, %3d) px  r %.2f um"
                      % (case, TIMES[k], i, err, cx, cy, r / PX_PER_UM))
            out[case + "_img"] = np.array(imgs, np.uint8)
            out[case + "_t"] = TIMES.copy()
            out[case + "_ctr"] = np.array(ctr, float)
            out[case + "_rad"] = np.array(rad, float)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/s41567-025-02785-0.pdf"))
