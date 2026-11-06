#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
NON-SPHERICAL (l = 2) SHAPE-MODE DECAY -- surface SHEAR VISCOSITY comparison
===============================================================================
The only test in the suite sensitive to mu_s.

For radial motion the Boussinesq--Scriven shear term cancels identically
(D_s = (Rdot/R) P, tr P = 2 in 3D), so every spherical case is blind to mu_s.
A shape mode deforms the interface at nearly constant area, D_s becomes
deviatoric, and mu_s dissipates.

MEASURE.  The bubble starts as R(theta) = R0[1 + a2 P2(cos theta)].  The mode
amplitude is tracked by the gas-weighted second Legendre moment

    Q(t) = int (1-eta) P2(cos theta) dV  /  int (1-eta) dV ,

which is 0 for a sphere and linear in a2 for small a2.  Q(t) is fitted to a
damped oscillation  Q = A exp(-beta t) cos(omega t + phi); beta is the decay
rate and is the quantity mu_s should change.

Inviscid reference (Lamb), gas bubble in liquid:
    omega_l^2 = (l-1)(l+1)(l+2) sigma / (rho_l R0^3)

    python3 non_spherical_decay_comparison.py
    python3 non_spherical_decay_comparison.py --root <dir with the outputs>
===============================================================================
"""
import argparse, glob, math, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG   = os.path.join(_HERE, "Images")
TESTS = os.path.normpath(os.path.join(_HERE, "..", "input_non_spherical_decay"))
DEFAULT_ROOT = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests", "FlowMarmottant", "l2_decay"))

FIG_W, FIG_H, DPI = 11, 6.5, 180
FS_T, FS_L, FS_G, FS_K = 16, 14, 12, 11


def parse_input(path):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    return kv


def _f(kv, key, default=None):
    try:    return float(kv[key].split()[0])
    except Exception: return default


def moments(pfdir):
    """Per plotfile: (t, R_equiv, Q).  Octant-aware (gas volume x 8)."""
    try:
        import yt; yt.funcs.mylog.setLevel(50)
    except ImportError:
        print("  [skip] yt not available"); return np.array([]), np.array([]), np.array([])
    pfs = sorted(glob.glob(os.path.join(pfdir, "*cell")),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    t, R, Q = [], [], []
    for pf in pfs:
        try:
            ds = yt.load(pf); ad = ds.all_data()
            eta = np.array(ad["eta"], float); g = np.clip(1.0 - eta, 0.0, 1.0)
            try:    vol = np.array(ad["index", "cell_volume"], float)
            except Exception: vol = np.array(ad["cell_volume"], float)
            dle = ds.domain_left_edge
            sym = 1
            for d in range(3):
                if float(dle[d]) > -1e-6: sym *= 2
            x = np.array(ad["x"], float); y = np.array(ad["y"], float); z = np.array(ad["z"], float)
            r2 = x*x + y*y + z*z
            P2 = 0.5*(3.0*z*z/np.maximum(r2, 1e-30) - 1.0)
            V = float(np.sum(g*vol))
            if V <= 0: continue
            t.append(float(ds.current_time))
            R.append((3.0*V*sym/(4.0*math.pi))**(1.0/3.0))
            Q.append(float(np.sum(g*P2*vol))/V)
        except Exception:
            continue
    o = np.argsort(t)
    return np.array(t)[o], np.array(R)[o], np.array(Q)[o]


def fit_decay(t, Q):
    """Q = A exp(-beta t) cos(omega t + phi).  Returns (beta, omega, rms)."""
    if len(t) < 8: return np.nan, np.nan, np.nan
    q = Q - Q[-len(Q)//4:].mean()          # remove any residual offset
    # frequency from the FFT of the (interpolated) signal
    ts = np.linspace(t[0], t[-1], max(len(t), 64))
    qi = np.interp(ts, t, q)
    F = np.fft.rfft(qi*np.hanning(len(qi))); fr = np.fft.rfftfreq(len(ts), ts[1]-ts[0])
    w0 = 2*math.pi*fr[max(1, int(np.argmax(np.abs(F[1:]))+1))]
    # decay from a linear fit to log|envelope| at the extrema
    a = np.abs(q)
    pk = [i for i in range(1, len(a)-1) if a[i] >= a[i-1] and a[i] >= a[i+1] and a[i] > 0]
    if len(pk) >= 2:
        c = np.polyfit(t[pk], np.log(a[pk]), 1); beta = -c[0]
    else:
        m = a > 0
        beta = -np.polyfit(t[m], np.log(a[m]), 1)[0] if m.sum() > 3 else np.nan
    A = np.abs(q).max()
    ph = np.arctan2(0.0, 1.0)
    rms = float(np.sqrt(np.mean((q - A*np.exp(-beta*t)*np.cos(w0*t + ph))**2))) if np.isfinite(beta) else np.nan
    return beta, w0, rms


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT, help="directory holding the run outputs")
    a = ap.parse_args()
    os.makedirs(IMG, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(TESTS, "input_l2_mus*")))
    if not inputs:
        raise SystemExit("no inputs under %s" % TESTS)

    rows, series = [], []
    for inp in inputs:
        kv  = parse_input(inp)
        mus = _f(kv, "shell.mu_s", 0.0)
        kap = _f(kv, "shell.kappa_s", 0.0)
        sig = _f(kv, "sigma", 0.0)
        rho = _f(kv, "density0.ic.expression.region0", 10.0)
        R0  = _f(kv, "eta.ic.expression.constant.R0", 0.2)
        name = os.path.basename(inp).replace("input_", "")
        out = os.path.join(a.root, "output_" + name.replace("l2_mus", "l2_decay_mus"))
        if not os.path.isdir(out):
            alt = os.path.join(a.root, name)
            out = alt if os.path.isdir(alt) else out
        t, R, Q = moments(out)
        w_lamb = math.sqrt(12.0*sig/(rho*R0**3)) if sig > 0 else float("nan")
        if len(t) < 3:
            print(f"  [skip] {name}: no usable output at {out}")
            rows.append((name, mus, kap, np.nan, np.nan, w_lamb, 0)); continue
        beta, w, _ = fit_decay(t, Q)
        rows.append((name, mus, kap, beta, w, w_lamb, len(t)))
        series.append((name, mus, t, Q))
        print(f"  {name}: {len(t)} frames  mu_s={mus:g}  beta={beta:.4f} 1/s  "
              f"omega={w:.4f} (Lamb {w_lamb:.4f})")

    print(f"\n{'case':16} {'mu_s':>10} {'kappa_s':>10} {'beta [1/s]':>12} "
          f"{'omega':>10} {'omega_Lamb':>11}")
    for n, mus, kap, b, w, wl, nf in rows:
        print(f"{n:16} {mus:10.3g} {kap:10.3g} {b:12.4f} {w:10.4f} {wl:11.4f}")
    fin = [(mus, b) for _, mus, _, b, _, _, nf in rows if nf and np.isfinite(b)]
    if len(fin) >= 2:
        fin.sort()
        d0 = fin[0][1]
        print("\n  decay rate vs mu_s (mu_s = 0 baseline):")
        for mus, b in fin:
            print(f"    mu_s = {mus:9.3g}   beta = {b:8.4f}   beta - beta_0 = {b-d0:+8.4f}")
        print("\n  EXPECTED: beta increases with mu_s while omega stays ~constant.")

    if series:
        plt.rcParams.update({"axes.titlesize": FS_T, "axes.labelsize": FS_L,
                             "legend.fontsize": FS_G, "xtick.labelsize": FS_K,
                             "ytick.labelsize": FS_K})
        fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
        for (n, mus, t, Q), c in zip(series, ("tab:blue","tab:orange","tab:green","tab:red")):
            ax.plot(t, Q, "-o", ms=3, lw=1.8, color=c, label=rf"$\mu_s$ = {mus:g}")
        ax.axhline(0.0, color="0.6", lw=0.9)
        ax.set_xlabel("t   [s]", fontsize=FS_L)
        ax.set_ylabel(r"$\langle P_2 \rangle$   ($\ell = 2$ mode amplitude)", fontsize=FS_L)
        ax.set_title(r"$\ell=2$ shape-mode decay: effect of surface shear viscosity",
                     fontsize=FS_T, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FS_K)
        ax.legend(fontsize=FS_G, loc="best")
        plt.tight_layout()
        for e in ("png", "eps"):
            fig.savefig(os.path.join(IMG, f"l2_decay_mus.{e}"), dpi=DPI, bbox_inches="tight")
        print(f"\n  wrote {os.path.join(IMG,'l2_decay_mus.png')} / .eps")


if __name__ == "__main__":
    main()
