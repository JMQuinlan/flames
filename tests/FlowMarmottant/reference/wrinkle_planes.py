#!/usr/bin/env python3
"""
Wrinkle check for a 3D coated-bubble run (octant domain, bubble centred at the origin).

Question it answers: does the eta = 1/2 surface wrinkle during the collapse, and is
that where the viscous shell tension is lost?  It looks at the bubble on five planes at
the finest level -- the three symmetry planes (z = 0: xy, y = 0: xz, x = 0: yz), the
diagonal plane x = y (through the body diagonal), and the plane z = R/2 (off the
symmetry planes) -- mirrors each quarter to a full disc, and for every frame measures
the shape of the eta = 1/2 contour on each.

The octant set-up is symmetric under swapping x, y, z, so the three symmetry planes show
the same picture unless that symmetry is broken; the diagonal and z = R/2 planes are the
independent views.  For R(theta) the in-plane angle is measured from the first in-plane
axis (x, x, y, the (1,1,0) direction, x).

Output (in --out, default ./wrinkles_<plotdir name>):
  planes_<frame>.png   per frame, one row per plane:
                         eta with contours 0.05 / 0.5 / 0.95,
                         sigma_tot (kappa1) in the band if the field is in the plotfile,
                         r(theta)/mean - 1 of the eta = 0.5 contour (grid axes at 0 and 90 deg)
  overview.png         eta = 0.5 contours of all three planes, selected frames side by side
  modes.png            ripple amplitudes a_m (m = 4, 8, 12, 16) vs R/R0, one line per plane,
                         with the (R0/R)^m slope for reference
  modes.csv            the numbers behind modes.png

What to look for: an elastic-only or uncoated run keeps a_8..a_16 near 0.1-0.3 % and only
squares off slowly (a_4).  A shell that crumples shows a_8..a_16 growing by orders of
magnitude during the collapse, ripples visible on the contours, and a patchy sigma_tot
(near zero where the surface has wrinkled, spikes at the folds).

Memory: one finest-level covering grid of the bubble's octant per frame (about 40^3 cells
at R0/dx = 25.6), planes interpolated from it; small enough for a login node.

Usage:
    python3 wrinkle_planes.py --plotdir /mmfs1/.../output_Sch20_Oscillating_Marmottant \\
        [--R0 2e-6] [--nframes 10 | --every 2] [--tmax 4e-7] [--out DIR]
"""
import argparse, csv, glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import yt
yt.funcs.mylog.setLevel(50)

PLANES = [(2, "xy plane (z = 0)"), (1, "xz plane (y = 0)"), (0, "yz plane (x = 0)"),
          ("diag", "diagonal plane (x = y)"), ("offset", "z = R/2 plane")]
MODES = (4, 8, 12, 16)


def plotfiles(d):
    fs = [p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p)]
    return sorted(fs, key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))


def has(ds, f):
    return ("boxlib", f) in ds.field_list


def box(ds, fields, half, dx):
    """Finest-level covering grid on [0, half]^3 (the bubble's octant), one array per field."""
    lev = ds.index.max_level
    m = int(np.ceil(half / dx)) + 2
    cg = ds.covering_grid(level=lev, left_edge=[0.0, 0.0, 0.0], dims=[m, m, m])
    return {f: np.asarray(cg[("boxlib", f)], float) for f in fields}


def plane(B, axis, half, n, dx, zoff=0.0):
    """Quarter-plane image on [0, half]^2 sampled from the box B by trilinear interpolation;
    pixel (i, j) is at in-plane coordinates ((i+.5)h, (j+.5)h), h = half/n.  The box is
    mirror-extended across the three symmetry planes so the first row is exact.
    axis 0/1/2: symmetry plane x/y/z = 0; 'diag': plane x = y (horizontal along (1,1,0)/sqrt2,
    vertical z); 'offset': plane z = zoff."""
    h = half / n
    u = (np.arange(n) + 0.5) * h
    U, V = np.meshgrid(u, u, indexing="ij")
    Z0 = np.zeros_like(U)
    if axis == 2:   pts = (U, V, Z0)
    elif axis == 1: pts = (U, Z0, V)
    elif axis == 0: pts = (Z0, U, V)
    elif axis == "diag": pts = (U / np.sqrt(2.0), U / np.sqrt(2.0), V)
    else: pts = (U, V, Z0 + zoff)
    Bp = np.pad(B, 1, mode="symmetric")[:-1, :-1, :-1]            # index 0 = mirror of cell 0
    idx = [p / dx - 0.5 + 1.0 for p in pts]                         # cell-centre index in the padded box
    return map_coordinates(Bp, idx, order=1, mode="nearest")


def mirror(q):
    top = np.concatenate([q[::-1, :], q], axis=0)
    return np.concatenate([top[:, ::-1], top], axis=1)


def contour(eta, h, lvl=0.5, nth=360):
    """eta = lvl crossing along rays theta in [0, 90 deg] from the origin; eta[i, j] at ((i+.5)h, (j+.5)h)."""
    n = eta.shape[0]
    th = np.linspace(0.0, np.pi / 2, nth)
    s = np.arange(0.0, n - 1.5, 0.02)
    r = np.full(nth, np.nan)
    for k, t in enumerate(th):
        e = map_coordinates(eta, [np.clip(s * np.cos(t) - .5, 0, None), np.clip(s * np.sin(t) - .5, 0, None)],
                            order=3, mode="nearest")
        idx = np.where((e[:-1] < lvl) & (e[1:] >= lvl))[0]
        if len(idx):
            q = idx[0]
            r[k] = (s[q] + (lvl - e[q]) / (e[q + 1] - e[q]) * 0.02) * h
    return th, r


def modes(th, r):
    ok = np.isfinite(r)
    if ok.sum() < 0.8 * len(r):
        return np.nan, [np.nan] * len(MODES)
    rb = np.mean(r[ok])
    w = r[ok] / rb - 1.0
    return rb, [2.0 * np.mean(w * np.cos(m * th[ok])) * 100.0 for m in MODES]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plotdir", required=True)
    ap.add_argument("--R0", type=float, default=2.0e-6)
    ap.add_argument("--nframes", type=int, default=10, help="frames to draw (evenly spaced) if --every is not given")
    ap.add_argument("--every", type=int, default=0, help="use every Nth plotfile for the mode history (0 = all)")
    ap.add_argument("--tmax", type=float, default=None, help="ignore plotfiles after this time")
    ap.add_argument("--window", type=float, default=1.3, help="half-width of the view in R0 (default 1.3)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fs = plotfiles(a.plotdir)
    if not fs:
        raise SystemExit("no NNNNNcell plotfiles in %s" % a.plotdir)
    out = a.out or os.path.join(os.getcwd(), "wrinkles_" + os.path.basename(os.path.normpath(a.plotdir)))
    os.makedirs(out, exist_ok=True)

    hist = fs[::a.every] if a.every > 0 else fs
    rows, drawn = [], []
    pick = set(np.linspace(0, len(hist) - 1, min(a.nframes, len(hist))).astype(int))
    for k, pf in enumerate(hist):
        ds = yt.load(pf)
        t = float(ds.current_time)
        if a.tmax is not None and t > a.tmax:
            break
        lev = ds.index.max_level
        dx = float(ds.domain_width[0]) / ds.domain_dimensions[0] / 2 ** lev
        half = a.window * a.R0
        n = int(round(half / dx))
        sig = has(ds, "kappa1")
        BX = box(ds, ["eta"] + (["kappa1"] if sig else []), half * 1.05, dx)
        data = []
        Rprev = rows[-len(PLANES)]["R_over_R0"] * a.R0 if rows else a.R0
        for axis, name in PLANES:
            zoff = 0.5 * Rprev if axis == "offset" else 0.0
            eta = plane(BX["eta"], axis, half, n, dx, zoff)
            s1 = plane(BX["kappa1"], axis, half, n, dx, zoff) if sig else None
            th, r = contour(eta, dx)
            rb, am = modes(th, r)
            data.append((name, eta, s1, th, r))
            rows.append(dict(frame=os.path.basename(pf), t=t, plane=name, R_over_R0=rb / a.R0,
                             **{"a%d_pct" % m: v for m, v in zip(MODES, am)}))
        print("%s t=%.3e  " % (os.path.basename(pf), t) + "  ".join(
            "%s: R/R0 %.3f a8 %+.2f%% a12 %+.2f%%" % (r_["plane"][:8], r_["R_over_R0"], r_["a8_pct"], r_["a12_pct"])
            for r_ in rows[-len(PLANES):]), flush=True)

        if k in pick:
            drawn.append((t, dx, n, data))
            ncol = 3 if sig else 2
            fig, axs = plt.subplots(len(PLANES), ncol, figsize=(4.2 * ncol, 4.0 * len(PLANES)))
            ext = np.array([-n, n, -n, n]) * dx * 1e6
            X = (np.arange(2 * n) - n + .5) * dx * 1e6
            for row, (name, eta, s1, th, r) in enumerate(data):
                E = mirror(eta)
                ax = axs[row, 0]
                ax.imshow(E.T, origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
                ax.contour(X, X, E.T, levels=[0.05, 0.5, 0.95], colors=["#D55E00", "k", "#009E73"], linewidths=[0.7, 1.1, 0.7])
                ax.set_aspect("equal"); ax.set_ylabel(name); ax.set_title("eta, contours 0.05 / 0.5 / 0.95", fontsize=9)
                if sig:
                    S = mirror(np.where((eta > 0.01) & (eta < 0.99), s1, np.nan))
                    ax = axs[row, 1]
                    im = ax.pcolormesh(X, X, S.T, cmap="RdBu", vmin=-0.08, vmax=0.08, shading="auto")
                    ax.contour(X, X, E.T, levels=[0.5], colors="k", linewidths=0.8)
                    ax.set_aspect("equal"); ax.set_title("sigma_tot [N/m], red = net compression", fontsize=9)
                    fig.colorbar(im, ax=ax, shrink=0.75)
                ax = axs[row, ncol - 1]
                ok = np.isfinite(r)
                ax.plot(np.degrees(th[ok]), 100 * (r[ok] / np.mean(r[ok]) - 1), color="#D55E00", lw=1.4)
                ax.axhline(0, color="0.7", lw=.6); ax.set_xticks([0, 45, 90]); ax.grid(alpha=.25, lw=.5)
                ax.set_title("eta = 0.5: r(theta)/mean - 1 [%%], R = %.3f R0" % (np.mean(r[ok]) / a.R0), fontsize=9)
                ax.set_xlabel("angle from the first in-plane axis [deg]")
            fig.suptitle("%s   t = %.3e s" % (os.path.basename(pf), t), fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out, "planes_%s.png" % os.path.basename(pf)), dpi=130)
            plt.close(fig)

    keys = list(rows[0].keys())
    with open(os.path.join(out, "modes.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)

    # overview: eta = 0.5 contours, three planes overlaid per drawn frame
    if drawn:
        fig, axs = plt.subplots(1, len(drawn), figsize=(3.0 * len(drawn), 3.3))
        axs = np.atleast_1d(axs)
        cols = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]
        for ax, (t, dx, n, data) in zip(axs, drawn):
            X = (np.arange(2 * n) - n + .5) * dx * 1e6
            for (name, eta, s1, th, r), c in zip(data, cols):
                ax.contour(X, X, mirror(eta).T, levels=[0.5], colors=c, linewidths=1.0)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title("t = %.2e s" % t, fontsize=9)
        for (axis, name), c in zip(PLANES, cols): axs[0].plot([], [], color=c, label=name)
        axs[0].legend(fontsize=7, frameon=False, loc="lower left")
        fig.suptitle("eta = 0.5 contour on each plane (the z = R/2 cut is smaller by sqrt(3)/2)", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(out, "overview.png"), dpi=140); plt.close(fig)

    # modes vs R
    fig, axs = plt.subplots(1, len(MODES), figsize=(3.6 * len(MODES), 3.4), sharey=True)
    for ax, m in zip(axs, MODES):
        for (axis, name), c in zip(PLANES, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#555555"]):
            R = np.array([r_["R_over_R0"] for r_ in rows if r_["plane"] == name])
            A = np.array([abs(r_["a%d_pct" % m]) for r_ in rows if r_["plane"] == name])
            ax.loglog(R, A + 1e-4, "-o", ms=2.5, lw=1.2, color=c, label=name)
        Rs = np.array([0.95, 0.4]); ax.loglog(Rs, 0.02 * (0.95 / Rs) ** m, "--", color="0.5", lw=0.9)
        ax.text(0.45, 0.02 * (0.95 / 0.45) ** m * 0.5, "(R0/R)^%d" % m, color="0.4", fontsize=8)
        ax.set_xlim(1.05, 0.25); ax.set_ylim(5e-3, 60); ax.set_title("mode m = %d" % m, fontsize=10)
        ax.set_xlabel("R / R0"); ax.grid(alpha=.25, lw=.5)
    axs[0].set_ylabel("|a_m| / R  [%]"); axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Ripple amplitudes of the eta = 0.5 contour on the symmetry planes (collapse runs right to left)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "modes.png"), dpi=140); plt.close(fig)
    print("wrote %s/{planes_*.png, overview.png, modes.png, modes.csv}" % out)


if __name__ == "__main__":
    main()
