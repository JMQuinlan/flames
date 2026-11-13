#!/usr/bin/env python3
"""
Copper projectile on a copper plate at 800 m/s (input_2D_PlateImpact_Copper_Plastic), against Fig. 8 of
Favrie and Gavrilyuk (2010), "Dynamics of shock waves in elastic-plastic solids", HAL hal-00492411.

  python3 copper_fg2010_figures.py rows    --fluid RUN --elastic RUN --plastic RUN --plastic-low RUN --out FIG.png
  python3 copper_fg2010_figures.py overlay --fluid RUN --elastic RUN --plastic RUN --plastic-low RUN --lit DIR --out FIG.png

rows:    one row per model (fluid, elastic, elastic-plastic sigma_y = 2.49 GPa, elastic-plastic sigma_y = 0.297 GPa),
         numerical schlieren at t = 100, 250, 600 us with the eta = 0.5 outline.
overlay: the nine schlieren images of the paper's Fig. 8 (DIR/img-000.png ... img-008.png, extracted with
         `pdfimages -f 19 -l 19 -png`; columns fluid / elastic / elastic-plastic, rows a, b, c) with our eta = 0.5 outline
         drawn on each at the paper's times (fluid 100, 250, 1000 us; elastic 100, 300, 600 us; elastic-plastic
         100, 250, 600 us).  Each published image is placed using only the initial geometry of the plate: its dark
         extent along y in the 100 us image of its column is the plate length, 0.5 m, centred at y = 0.35 m, and the
         centre of the plate end at x = 0.25 m.  Nothing from the numerical solution enters that placement.
         The published images span about 1 m x 1 m; the numerical domain is 0.7 m x 0.7 m (dashed box).
Units of the deck: length m, time ms.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yt
yt.funcs.mylog.setLevel(50)

plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 13,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})
LP, YP, XE = 0.5, 0.35, 0.25          # plate length, plate centre y, centre x of the plate end (initial geometry)


def frames(run):
    pfs = sorted(glob.glob(os.path.join(run, "*cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    return pfs, np.array([float(yt.load(p).current_time) for p in pfs])


def load(run, t, cache={}):
    if run not in cache: cache[run] = frames(run)
    pfs, tt = cache[run]; k = int(np.argmin(np.abs(tt - t)))
    if abs(tt[k] - t) > 2e-3: print(f"WARNING {run}: nearest frame t = {tt[k]:.4f} ms for requested {t:.4f} ms")
    ds = yt.load(pfs[k]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    g = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0].T
    lo, hi = ds.domain_left_edge.v[:2], ds.domain_right_edge.v[:2]
    ny, nx = g("eta").shape
    x = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx; y = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return dict(t=float(ds.current_time), rho=g("density"), eta=g("eta"), x=x, y=y, ext=(lo[0], hi[0], lo[1], hi[1]))


def schlieren(f):
    gy, gx = np.gradient(f["rho"]); G = np.hypot(gx, gy)
    return np.exp(-4.0 * G / max(np.percentile(G, 99.0), 1e-30))


def opt(a, k, d=None): return a[a.index(k) + 1] if k in a else d


def rows(a):
    runs = [("Fluid ($\\mu$ = 0)", opt(a, "--fluid")), ("Elastic", opt(a, "--elastic")),
            ("Elastic-Plastic, $\\sigma_y$ = 2.49 GPa", opt(a, "--plastic")),
            ("Elastic-Plastic, $\\sigma_y$ = 0.297 GPa", opt(a, "--plastic-low"))]
    if "--no-fluid" in a: runs = runs[1:]
    times = [float(t) for t in opt(a, "--times", "0.1,0.25,0.6").split(",")]
    fig, ax = plt.subplots(len(runs), len(times), figsize=(4.3 * len(times) + 1.0, 4.3 * len(runs) + 0.6), squeeze=False)
    for i, (lab, run) in enumerate(runs):
        for j, t in enumerate(times):
            f = load(run, t); A = ax[i, j]
            A.imshow(schlieren(f), origin="lower", extent=f["ext"], cmap="gray", vmin=0, vmax=1)
            A.contour(f["x"], f["y"], f["eta"], [0.5], colors="C3", linewidths=0.9)
            A.set_xticks([0, 0.2, 0.4, 0.6]); A.set_yticks([0, 0.2, 0.4, 0.6])
            if i == 0: A.set_title("$t$ = %.0f $\\mu$s" % (1e3 * f["t"]))
            if i == len(runs) - 1: A.set_xlabel("$x$ [m]")
            else: A.set_xticklabels([])
            if j == 0: A.set_ylabel(lab + "\n$y$ [m]")
            else: A.set_yticklabels([])
    fig.suptitle("Copper Projectile on a Copper Plate at 800 m/s: Effect of the Constitutive Model (Numerical Schlieren)", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); save(fig, opt(a, "--out"))


def overlay(a):
    lit = opt(a, "--lit"); from PIL import Image
    img = [np.asarray(Image.open(os.path.join(lit, "img-%03d.png" % k)).convert("L")) for k in range(9)]
    # (published column index, name, times, runs); --columns picks a subset, --no-high drops the 2.49 GPa outline
    allc = [(0, "Fluid", [0.1, 0.25, 1.0], [("--fluid", "C3", "Numerical")]),
            (1, "Elastic", [0.1, 0.3, 0.6], [("--elastic", "C3", "Numerical")]),
            (2, "Elastic-Plastic", [0.1, 0.25, 0.6], [("--plastic-low", "C3", "Numerical, $\\sigma_y$ = 0.297 GPa"),
                                                      ("--plastic", "C0", "Numerical, $\\sigma_y$ = 2.49 GPa")])]
    keep = [c.strip().lower() for c in opt(a, "--columns", "fluid,elastic,elastic-plastic").split(",")]
    nohigh = "--no-high" in a
    cols = [(j0, n, tt, [r for r in rr if not (nohigh and r[0] == "--plastic")]) for j0, n, tt, rr in allc if n.lower() in keep]
    nc = len(cols)
    fig, ax = plt.subplots(3, nc, figsize=(5.0 * nc, 15.8), squeeze=False)
    for j, (j0, name, times, runs) in enumerate(cols):
        # placement from the 100 us image of this column (row a): dark extent along y = plate length
        a0 = img[j0].astype(float); d = a0 < 0.55 * np.median(a0)
        r = np.where(d.any(1))[0]; S = (r.max() - r.min()) / LP; rc = 0.5 * (r.min() + r.max())
        pr = int(r.min() + 0.02 * S); c = np.where(d[pr])[0]; cc = 0.5 * (c.min() + c.max())
        for i, t in enumerate(times):
            im = img[3 * i + j0]; h, w = im.shape
            L = (XE - cc / S, XE + (w - cc) / S, YP - (h - rc) / S, YP + rc / S)   # (left, right, bottom, top)
            A = ax[i, j]; A.imshow(im, cmap="gray", vmin=0, vmax=255, extent=L, origin="upper")
            for key, col, lab in runs:
                f = load(opt(a, key), t)
                A.contour(f["x"], f["y"], f["eta"], [0.5], colors=col, linewidths=1.3)
            A.plot([0, 0.7, 0.7, 0, 0], [0, 0, 0.7, 0.7, 0], "k--", lw=1.0)
            A.set_xlim(L[0], L[1]); A.set_ylim(L[2], L[3]); A.set_aspect("equal")
            A.set_title("%s, $t$ = %.0f $\\mu$s" % (name, 1e3 * t))
            A.set_xlabel("$x$ [m]"); A.set_ylabel("$y$ [m]")
        print(f"{name}: {S:.1f} px/m, published image spans {L[1]-L[0]:.3f} x {L[3]-L[2]:.3f} m, x {L[0]:.3f}..{L[1]:.3f}, y {L[2]:.3f}..{L[3]:.3f}")
    if nohigh:
        hl = [Line2D([], [], color="C3", lw=1.6, label="Numerical, $\\eta$ = 0.5")]
    else:
        hl = [Line2D([], [], color="C3", lw=1.6, label="Numerical, $\\eta$ = 0.5 (Right Column: $\\sigma_y$ = 0.297 GPa)"),
              Line2D([], [], color="C0", lw=1.6, label="Numerical, $\\sigma_y$ = 2.49 GPa")]
    hl.append(Line2D([], [], color="k", ls="--", lw=1.0, label="Numerical Domain"))
    fig.legend(handles=hl, loc="lower center", ncol=len(hl) if nc > 2 else 1, fontsize=13, framealpha=0.95)
    fig.suptitle("Copper Projectile on a Copper Plate:\nNumerical Outline on the Schlieren of (Favrie and Gavrilyuk 2010)"
                 if nc < 3 else "Copper Projectile on a Copper Plate: Numerical Outline on the Schlieren of (Favrie and Gavrilyuk 2010)", fontsize=16)
    fig.tight_layout(rect=(0, 0.035 if nc > 2 else 0.06, 1, 0.98 if nc > 2 else 0.96)); save(fig, opt(a, "--out"))


def save(fig, out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for ext in ("png", "eps"): fig.savefig(os.path.splitext(out)[0] + "." + ext, dpi=170 if ext == "png" else None)
    plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    {"rows": rows, "overlay": overlay}[a[0]](a[1:])
