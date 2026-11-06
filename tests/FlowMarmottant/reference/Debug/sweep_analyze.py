#!/usr/bin/env python3
"""Extract time series from the Marmottant R0 sweep and cache to npz.

Per plotfile:  t, R (gas-area equivalent), sigma mean/min/max/std on the
interface band, Gamma mean/min/max, shape modes e2/e4/e8, max AMR level.
"""
import sys, glob, os, json
import numpy as np
import yt
yt.set_log_level(50)

CHI, RB, SBRK, SIGW = 14.56, 0.018, 7.28, 7.28


def one(path):
    ds = yt.load(path); ad = ds.all_data()
    eta = np.asarray(ad["boxlib", "eta"])
    vol = np.asarray(ad["index", "cell_volume"])
    x = np.asarray(ad["index", "x"]); y = np.asarray(ad["index", "y"])
    sig = np.asarray(ad["boxlib", "kappa2"])
    gam = np.asarray(ad["boxlib", "Gamma"])
    ux = np.asarray(ad["boxlib", "velocityx"])
    uy = np.asarray(ad["boxlib", "velocityy"])

    gas = (1.0 - eta) * vol
    A = gas.sum(); R = np.sqrt(A / np.pi)
    th = np.arctan2(y, x)
    e = {l: abs((gas * np.exp(1j * l * th)).sum()) / A for l in (2, 4, 8)}

    w = eta * (1.0 - eta)
    m = w > 0.05
    if m.any():
        ww = w[m] * vol[m]; ww /= ww.sum()
        sg, gm = sig[m], gam[m]
        smean = float((ww * sg).sum())
        sstd = float(np.sqrt((ww * (sg - smean) ** 2).sum()))
        gmean = float((ww * gm).sum())
        out = dict(sig=smean, sig_std=sstd, sig_min=float(sg.min()),
                   sig_max=float(sg.max()), gam=gmean,
                   gam_min=float(gm.min()), gam_max=float(gm.max()))
    else:
        out = dict(sig=np.nan, sig_std=np.nan, sig_min=np.nan, sig_max=np.nan,
                   gam=np.nan, gam_min=np.nan, gam_max=np.nan)
    # |u|max on the interface band: the parasitic-current health metric.  With
    # mu = 0 these grow exponentially from the capillary stress tensor and
    # eventually corrupt everything downstream; with enough viscosity they
    # saturate and decay.
    out["umax"] = float(np.hypot(ux[m], uy[m]).max()) if m.any() else float("nan")
    out.update(t=float(ds.current_time), R=float(R),
               e2=e[2], e4=e[4], e8=e[8], lev=int(ds.index.max_level))
    return out


if __name__ == "__main__":
    runs = sorted(d for d in glob.glob(sys.argv[1]) if os.path.isdir(d))
    data = {}
    for d in runs:
        tag = os.path.basename(d)
        R0 = float(tag.split("_")[1]) * 1e-4
        rows = []
        for f in sorted(glob.glob(os.path.join(d, "*cell"))):
            try:
                rows.append(one(f))
            except Exception as ex:
                print(f"  skip {f}: {ex}", file=sys.stderr)
        if not rows:
            continue
        keys = rows[0].keys()
        data[tag] = {"R0": R0, **{k: np.array([r[k] for r in rows]) for k in keys}}
        print(f"{tag}  R0={R0:.4f}  n={len(rows)}  t_end={rows[-1]['t']*1e3:.2f} ms")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_data.npz"),
             **{f"{k}|{kk}": vv for k, v in data.items()
                for kk, vv in v.items()})
    print("wrote sweep_data.npz")
