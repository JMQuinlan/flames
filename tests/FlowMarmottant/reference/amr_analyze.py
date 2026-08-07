#!/usr/bin/env python3
"""AMR-correct analysis of a Marmottant bubble plotfile (yt handles the
level masking, so covered coarse cells are not double counted).

Reports, per plotfile:
  R/R0           equivalent radius from the gas area  sqrt(int (1-eta) dA / pi)
  Gamma          band-weighted areal shell density (target 1 at t=0)
  sigma          band-weighted effective tension  (target 3.4153 at t=0)
  spread         (max-min)/mean of sigma over the band
  e4, e8         shape Fourier modes from gas-area moments (0 = perfect circle)
"""
import sys, glob, os
import numpy as np
import yt
yt.set_log_level(50)

R0, SIG0 = 0.02, 3.415309


def analyze(path):
    ds = yt.load(path)
    ad = ds.all_data()
    eta = np.asarray(ad["boxlib", "eta"])
    vol = np.asarray(ad["index", "cell_volume"])
    x = np.asarray(ad["index", "x"])
    y = np.asarray(ad["index", "y"])
    try:
        sig = np.asarray(ad["boxlib", "kappa2"])
        gam = np.asarray(ad["boxlib", "Gamma"])
    except Exception:
        sig = np.zeros_like(eta); gam = np.zeros_like(eta)

    gas = (1.0 - eta) * vol
    A = gas.sum()
    R = np.sqrt(A / np.pi)

    th = np.arctan2(y, x)
    e = {}
    for l in (2, 4, 8):
        e[l] = abs((gas * np.exp(1j * l * th)).sum()) / A

    w = eta * (1.0 - eta)
    m = w > 0.05
    if m.any():
        ww = w[m] * vol[m]
        sg, gm = sig[m], gam[m]
        smean = (ww * sg).sum() / ww.sum()
        gmean = (ww * gm).sum() / ww.sum()
        spread = (sg.max() - sg.min()) / smean if smean != 0 else float("nan")
        gmin, gmax = gm.min(), gm.max()
    else:
        smean = gmean = spread = gmin = gmax = float("nan")
    return (float(ds.current_time), R / R0, gmean, gmin, gmax, smean, spread,
            e[4], e[8], int(ds.index.max_level))


if __name__ == "__main__":
    pats = sys.argv[1:]
    files = []
    for p in pats:
        files += sorted(glob.glob(os.path.join(p, "*cell")))
    print(f"{'t(ms)':>7} {'R/R0':>8} {'Gamma':>8} {'[min':>8} {'max]':>8} "
          f"{'sigma':>8} {'spread':>8} {'e4':>8} {'e8':>8} {'lev':>4}")
    for f in files:
        try:
            t, r, g, gl, gh, s, sp, e4, e8, lv = analyze(f)
        except Exception as ex:
            print(f"  {os.path.basename(f)}: {ex}"); continue
        print(f"{t*1e3:7.2f} {r:8.4f} {g:8.4f} {gl:8.4f} {gh:8.4f} "
              f"{s:8.4f} {100*sp:7.1f}% {e4:8.4f} {e8:8.4f} {lv:4d}")
