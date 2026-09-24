#!/usr/bin/env python3
"""
Drift-separated comparison across driven-bubble runs.

Fits  y(t) = c + m*t + a*sin(wt) + b*cos(wt)  to every radius measure, so the
secular DRIFT is separated from the drive oscillation.  A plain mean or an
endpoint difference cannot do this: it is taken at arbitrary drive phase and
the oscillation (0.7% here) swamps the drift (0.2%/cycle).

Needs at least ONE FULL drive cycle -- at 1.00 cycle the drift is recovered to
0.6%, at 0.32 cycles the error is 21% (verified synthetically).

Usage:  drift_scaling.py [--omega W] DIR [DIR ...]
"""
import argparse, glob, math, os, re
import numpy as np


def series(d, R0, omega):
    import yt
    yt.funcs.mylog.setLevel(50)
    rows = []
    for pf in sorted(glob.glob(d + "/*cell"),
                     key=lambda s: int(re.search(r"(\d+)cell", s).group(1))):
        try:
            ds = yt.load(pf); ad = ds.all_data()
            eta = np.array(ad["eta"], float)
            try:   vol = np.array(ad["index", "cell_volume"], float)
            except Exception: vol = np.array(ad["cell_volume"], float)
            sym = 1
            for k in range(3):
                if float(ds.domain_left_edge[k]) > -1e-12: sym *= 2
            RV = (3 * float((np.clip(1 - eta, 0, 1) * vol).sum()) * sym / (4 * math.pi)) ** (1 / 3)
            r1 = np.array(ad["rho_eta1"], float)
            mw = r1 * vol; msum = float(mw.sum())
            x = np.array(ad["x"], float); y = np.array(ad["y"], float); z = np.array(ad["z"], float)
            r = np.sqrt(x * x + y * y + z * z)
            Rm = math.sqrt(5 / 3 * float((mw * r * r).sum()) / msum)
            core = eta < 0.1
            rc = float(((r1[core] / np.maximum(1 - eta[core], 1e-12)) * vol[core]).sum()
                       / vol[core].sum())
            nb = 400; rmax = min(float(r.max()), 4 * R0)
            ed = np.linspace(0, rmax, nb + 1)
            idx = np.clip(np.digitize(r, ed) - 1, 0, nb - 1)
            ws = np.bincount(idx, weights=vol, minlength=nb)
            es = np.bincount(idx, weights=eta * vol, minlength=nb)
            ok = ws > 0
            rr = (0.5 * (ed[:-1] + ed[1:]))[ok]; ee = (es / np.maximum(ws, 1e-300))[ok]
            def cr(l):
                for i in range(len(rr) - 1):
                    if (ee[i] - l) * (ee[i + 1] - l) < 0:
                        return rr[i] + (l - ee[i]) * (rr[i + 1] - rr[i]) / (ee[i + 1] - ee[i])
                return np.nan
            rows.append((float(ds.current_time), RV, cr(0.5), Rm, rc,
                         cr(0.9) - cr(0.1), msum * sym))
        except Exception:
            pass
    return np.array(rows)


def fit(t, y, w, T):
    X = np.column_stack([np.ones_like(t), t - t.mean(), np.sin(w * t), np.cos(w * t)])
    c, m, p, q = np.linalg.lstsq(X, y, rcond=None)[0]
    return c, m * T, math.hypot(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--omega", type=float, default=256.20)
    ap.add_argument("--R0", type=float, default=0.02)
    a = ap.parse_args()
    w = a.omega; T = 2 * math.pi / w
    print(f"drive omega = {w} rad/s   T = {T:.6e} s\n")
    hdr = (f"{'run':>22} {'cyc':>5} {'R_05 %/cyc':>11} {'R_m %/cyc':>10} "
           f"{'rho %/cyc':>10} {'band %/cyc':>11} {'osc amp %':>10} {'ratio':>6}")
    print(hdr); print("-" * len(hdr))
    for d in a.dirs:
        s = series(d, a.R0, w)
        if len(s) < 8:
            print(f"{os.path.basename(d):>22}  only {len(s)} frames"); continue
        t = s[:, 0]; nc = t[-1] / T
        _, d5, amp = fit(t, s[:, 2] / a.R0, w, T)
        _, dm, _   = fit(t, s[:, 3] / a.R0, w, T)
        _, dr, _   = fit(t, s[:, 4] / s[0, 4], w, T)
        _, db, _   = fit(t, s[:, 5] / a.R0, w, T)
        ratio = (-3 * d5) / dr if dr else float("nan")
        flag = "" if nc >= 1.0 else "  <1 CYCLE, UNRELIABLE"
        print(f"{os.path.basename(d):>22} {nc:5.2f} {100*d5:+11.5f} {100*dm:+10.5f} "
              f"{100*dr:+10.5f} {100*db/ (s[0,5]/a.R0):+11.5f} {100*amp:10.5f} {ratio:6.2f}{flag}")
    print("\n  ratio = (-3 * R_05 drift) / (rho_core drift); 1.0 means the measure is")
    print("  consistent with the exactly conserved gas mass (rho ~ R^-3).")


if __name__ == "__main__":
    main()
