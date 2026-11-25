#!/usr/bin/env python3
"""
Vaporization mass-transfer GIF for a shock-water-column run with apply_vaporization = 1.
Top: log10 of the mass-transfer rate m_dot (plot field Vap_dot_rho, kg/m^3/s) with the eta = 0.5 interface.
Bottom: integrated rate and cumulative vaporized mass as a fraction of the initial water mass, vs t/t*.

  python3 swc_vap_gif.py OUT.gif RUN_DIR --tstar US [--every N] [--level L] [--xlim D0,D1] [--ymax D]
         [--title "..."] [--D MM]
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import yt
yt.funcs.mylog.setLevel(50)

D = 4.8e-3
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    out, run = a[0], a[1]
    global D
    D = float(opt("--D", "4.8")) * 1e-3   # drop diameter [mm] (D_mm)
    tstar = float(opt("--tstar", "1")) * 1e-6
    every = int(opt("--every", "2")); level = int(opt("--level", "2"))
    x0, x1 = [float(s) for s in opt("--xlim", "-1,7").split(",")]; ymax = float(opt("--ymax", "3"))
    title = opt("--title", "Vaporization Mass Transfer")
    d = os.path.join(run, "output") if os.path.isdir(os.path.join(run, "output")) else run
    pfs = sorted(glob.glob(os.path.join(d, "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    idx = list(range(0, len(pfs), every))
    if idx[-1] != len(pfs) - 1: idx.append(len(pfs) - 1)

    # pass 1: history (integrated rate, water mass) on all plotfiles in idx
    T, R, M, MX = [], [], [], []
    for i in idx:
        ds = yt.load(pfs[i]); ad = ds.all_data()
        v = np.asarray(ad["index", "cell_volume"])
        md = np.asarray(ad["boxlib", "Vap_dot_rho"]); MX.append(float(md.max()))
        T.append(float(ds.current_time)); R.append(float((md * v).sum()))
        if not M:
            M.append(float(((1 - np.asarray(ad["boxlib", "eta"])) * np.asarray(ad["boxlib", "density"]) * v).sum()))
    T, R = np.array(T), np.array(R); m0 = M[0]
    cum = np.concatenate([[0], np.cumsum(0.5 * (R[1:] + R[:-1]) * np.diff(T))]) / m0   # mass per unit depth, both 2D
    print(f"{run}: initial water mass {m0:.4g} kg/m, vaporized by end {cum[-1]:.3e} of it")

    tmpdir = os.path.splitext(out)[0] + "_frames"; os.makedirs(tmpdir, exist_ok=True)
    imgs = []
    vmax = np.log10(max(max(MX), 1e-30))
    for n, i in enumerate(idx):
        ds = yt.load(pfs[i]); lev = min(level, ds.index.max_level)
        cg = ds.covering_grid(lev, ds.domain_left_edge, ds.domain_dimensions * 2 ** lev)
        md = np.asarray(cg[("boxlib", "Vap_dot_rho")])[:, :, 0].T; eta = np.asarray(cg[("boxlib", "eta")])[:, :, 0].T
        L, Rr = ds.domain_left_edge.d, ds.domain_right_edge.d
        ny, nx = md.shape
        x = (L[0] + (np.arange(nx) + 0.5) * (Rr[0] - L[0]) / nx) / D
        y = (L[1] + (np.arange(ny) + 0.5) * (Rr[1] - L[1]) / ny) / D
        kx = (x >= x0) & (x <= x1); ky = y <= ymax
        md, eta, x, y = md[np.ix_(ky, kx)], eta[np.ix_(ky, kx)], x[kx], y[ky]
        fig, (A, B) = plt.subplots(2, 1, figsize=(9.5, 8.2), gridspec_kw={"height_ratios": [1.25, 1]})
        lm = np.log10(np.maximum(md, 1e-30))
        im = A.imshow(lm, origin="lower", cmap="inferno", vmin=vmax - 5, vmax=vmax,
                      extent=(x[0], x[-1], y[0], y[-1]), interpolation="nearest")
        A.contour(x, y, eta, [0.5], colors="c", linewidths=1.0)
        A.set_aspect("equal"); A.set_xlabel("$x/D$"); A.set_ylabel("$y/D$")
        A.set_title(f"$t$ = {T[n] * 1e6:.0f} $\\mu$s,  $t/t^*$ = {T[n] / tstar:.2f}")
        cb = fig.colorbar(im, ax=A, fraction=0.03, pad=0.02); cb.set_label(r"$\log_{10}\,\dot m$ [kg/m$^3$/s]")
        B.plot(T / tstar, cum, "k-", lw=1.5, label="Cumulative Vaporized Mass / Initial Water Mass")
        B.plot(T[n] / tstar, cum[n], "ko", mfc="none", ms=8)
        B.set_xlabel("$t/t^*$"); B.set_ylabel("Vaporized Mass Fraction"); B.set_xlim(0, T[-1] / tstar)
        B.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        B2 = B.twinx(); B2.plot(T / tstar, R, color="tab:red", lw=1.2, ls="--", label=r"Integrated Rate $\int \dot m\, dA$")
        B2.set_ylabel(r"$\int \dot m\, dA$ [kg/(m s)]", color="tab:red"); B2.tick_params(axis="y", colors="tab:red")
        B2.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        h1, l1 = B.get_legend_handles_labels(); h2, l2 = B2.get_legend_handles_labels()
        B.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=11, frameon=False)
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p = os.path.join(tmpdir, f"f{n:04d}.png"); fig.savefig(p, dpi=80); plt.close(fig)
        imgs.append(Image.open(p).convert("RGB"))
        if n % 20 == 0: print(f"{out}: frame {n + 1}/{len(idx)}", flush=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
