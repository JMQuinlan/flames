#!/usr/bin/env python3
"""
Shock-water-column breakup GIF: density (log scale) with the eta = 0.5 interface, mirrored about the symmetry plane.
One row per run; with two runs (e.g. no vaporization / vaporization) the rows share the colour scale and
are matched in t*.

  python3 swc_gif.py OUT.gif "Label:RUN_DIR" ["Label:RUN_DIR" ...] [--tstar US] [--every N] [--xmax MM] [--ymax MM]
         [--title "..."] [--level L] [--split] [--D MM]
--split (two runs): one panel, first run in the top half (y > 0), second run mirrored into the bottom half.
RUN_DIR holds the plotfiles directly or an output/ subfolder with them; "dirA+dirB" joins a run and its restart.
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


def plotfiles(dirs):
    """dirs: one run folder, or several joined with '+' (a run and its restarts); a later folder's plotfile
    replaces an earlier one with the same step number."""
    byname = {}
    for d in dirs.split("+"):
        if os.path.isdir(os.path.join(d, "output")): d = os.path.join(d, "output")
        for f in glob.glob(os.path.join(d, "*[0-9]cell")): byname[os.path.basename(f)] = f
    return sorted(byname.values(), key=lambda f: int(os.path.basename(f)[:-4]))


def frame(pf, level, xmax, ymax):
    ds = yt.load(pf)
    lev = min(level, ds.index.max_level)
    dims = ds.domain_dimensions * 2 ** lev
    cg = ds.covering_grid(lev, ds.domain_left_edge, dims)
    rho = np.asarray(cg[("boxlib", "density")])[:, :, 0].T
    eta = np.asarray(cg[("boxlib", "eta")])[:, :, 0].T
    L, R = ds.domain_left_edge.d, ds.domain_right_edge.d
    ny, nx = rho.shape
    x = L[0] + (np.arange(nx) + 0.5) * (R[0] - L[0]) / nx
    y = L[1] + (np.arange(ny) + 0.5) * (R[1] - L[1]) / ny
    kx = x <= xmax; ky = y <= ymax
    rho, eta, x, y = rho[np.ix_(ky, kx)], eta[np.ix_(ky, kx)], x[kx], y[ky]
    # mirror about y = 0 so the full column is shown
    rho = np.vstack([rho[::-1], rho]); eta = np.vstack([eta[::-1], eta]); y = np.concatenate([-y[::-1], y])
    return float(ds.current_time), x, y, rho, eta


def main():
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    flags = {"--split", "--tstar", "--every", "--xmax", "--ymax", "--title", "--level", "--D"}
    vflags = flags - {"--split"}
    pos = [s for i, s in enumerate(a) if s not in flags and (i == 0 or a[i - 1] not in vflags)]
    split = "--split" in a and len(a) > 0
    out, runs = pos[0], [r.split(":", 1) for r in pos[1:]]
    global D
    D = float(opt("--D", "4.8")) * 1e-3   # drop diameter [mm] (D_mm); sets the x/D, y/D axes
    tstar = float(opt("--tstar", "0")) * 1e-6
    every = int(opt("--every", "2")); level = int(opt("--level", "2"))
    xmax = float(opt("--xmax", "40")) * 1e-3; ymax = float(opt("--ymax", "18")) * 1e-3
    title = opt("--title", "Shock-Water Column Breakup")
    pfs = [plotfiles(d) for _, d in runs]
    tlist = [np.array([float(yt.load(p).current_time) for p in P]) for P in pfs]
    # drive the frames from the first run; other runs use the nearest time
    idx = list(range(0, len(pfs[0]), every))
    if idx[-1] != len(pfs[0]) - 1: idx.append(len(pfs[0]) - 1)
    tmpdir = os.path.splitext(out)[0] + "_frames"; os.makedirs(tmpdir, exist_ok=True)
    nr = len(runs)
    imgs = []
    if split:
        assert nr == 2, "--split needs exactly two runs"
        for n, i in enumerate(idx):
            t0 = tlist[0][i]; j = int(np.argmin(np.abs(tlist[1] - t0)))
            ta, x, y, ra, ea = frame(pfs[0][i], level, xmax, ymax)
            tb, xb, yb, rb, eb = frame(pfs[1][j], level, xmax, ymax)
            h = y > 0
            rho = np.vstack([rb[~h], ra[h]]); eta = np.vstack([eb[~h], ea[h]])
            fig, A = plt.subplots(figsize=(9.5, 5.4))
            im = A.imshow(np.log10(rho), origin="lower", cmap="viridis", vmin=0, vmax=3,
                          extent=(x[0] / D, x[-1] / D, y[0] / D, y[-1] / D), interpolation="nearest")
            A.contour(x / D, y[h] / D, ea[h], [0.5], colors="r", linewidths=0.8)
            A.contour(x / D, y[~h] / D, eb[~h], [0.5], colors="r", linewidths=0.8)
            A.axhline(0, color="w", lw=1.5, ls="--")
            for yy, lab, va in ((0.97, runs[0][0], "top"), (0.03, runs[1][0], "bottom")):
                A.text(0.02, yy, lab, transform=A.transAxes, color="w", fontsize=13, fontweight="bold", va=va)
            ts = f",  $t/t^*$ = {ta / tstar:.2f}" if tstar > 0 else ""
            A.set_title(f"$t$ = {ta * 1e6:.0f} $\\mu$s{ts}")
            if abs(ta - tb) > 0.02 * max(ta, 1e-12): print(f"WARNING frame {n}: times differ {ta:.3e} vs {tb:.3e}")
            A.set_xlabel("$x/D$"); A.set_ylabel("$y/D$"); A.set_aspect("equal")
            fig.subplots_adjust(left=0.08, right=0.86, top=0.84, bottom=0.14)
            cax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
            fig.colorbar(im, cax=cax, label=r"$\log_{10}\,\rho$ [kg/m$^3$]")
            fig.suptitle(title, fontsize=15)
            p = os.path.join(tmpdir, f"f{n:04d}.png"); fig.savefig(p, dpi=80); plt.close(fig)
            imgs.append(Image.open(p).convert("RGB"))
            if n % 20 == 0: print(f"{out}: frame {n + 1}/{len(idx)}", flush=True)
        imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
        print("wrote", out); return
    for n, i in enumerate(idx):
        t0 = tlist[0][i]
        fig, ax = plt.subplots(nr, 1, figsize=(9.5, 1.4 + 4.4 * nr), squeeze=False)
        for r in range(nr):
            j = int(np.argmin(np.abs(tlist[r] - t0)))
            t, x, y, rho, eta = frame(pfs[r][j], level, xmax, ymax)
            A = ax[r, 0]
            im = A.imshow(np.log10(rho), origin="lower", cmap="viridis", vmin=0, vmax=3,
                          extent=(x[0] / D, x[-1] / D, y[0] / D, y[-1] / D), interpolation="nearest")
            A.contour(x / D, y / D, eta, [0.5], colors="r", linewidths=0.8)
            lab = runs[r][0]
            ts = f",  $t/t^*$ = {t / tstar:.2f}" if tstar > 0 else ""
            A.set_title(f"{lab}:  $t$ = {t * 1e6:.0f} $\\mu$s{ts}")
            A.set_ylabel("$y/D$"); A.set_aspect("equal")
        ax[-1, 0].set_xlabel("$x/D$")
        fig.subplots_adjust(left=0.08, right=0.86, top=0.88 if nr > 1 else 0.82, bottom=0.12 if nr > 1 else 0.2,
                            hspace=0.35)
        cax = fig.add_axes([0.89, 0.2, 0.02, 0.6])
        fig.colorbar(im, cax=cax, label=r"$\log_{10}\,\rho$ [kg/m$^3$]")
        fig.suptitle(title, fontsize=15)
        p = os.path.join(tmpdir, f"f{n:04d}.png"); fig.savefig(p, dpi=80); plt.close(fig)
        imgs.append(Image.open(p).convert("RGB"))
        if n % 20 == 0: print(f"{out}: frame {n + 1}/{len(idx)}", flush=True)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=80, loop=0, optimize=True)
    print("wrote", out)


if __name__ == "__main__":
    main()
