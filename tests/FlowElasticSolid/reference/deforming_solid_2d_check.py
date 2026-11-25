#!/usr/bin/env python3
"""
2D deforming-solid tests of tests/FlowElasticSolid (UNIT_TEST_2D_ShockCylinder,
UNIT_TEST_2D_PlateImpact): elastic run vs the same run with elastic.mu0 = 0
(the "solid" is then a liquid of the same EOS).

  python3 deforming_solid_2d_check.py <elastic_run_dir> [<liquid_run_dir>] [--outdir DIR] [--no-gif] [--stride N]
          [--unit GPa] [--title "Copper Plate Impact"] [--solid-color copper|aluminum|steel|<colour>]
          [--literature PNG --lit-times t1,t2,t3,t4 [--lit-shift DX] [--lit-label "Author et al. (YYYY)"]]

With ONE run directory the figures show the elastic body alone (no liquid comparison):
  deforming_solid_stress.png      S_xx, S_yy, S_xy and the von Mises stress inside the solid at four times
With --literature: a published 2 x 2 image of the same problem (four times, each panel the full domain) is
compared with the run at the nearest plot times,
  deforming_solid_vs_literature.png   top: the published panels with the numerical alpha_s = 0.5 outline drawn on
                                      them (shifted by DX in x if the deck places the body elsewhere);
                                      bottom: numerical schlieren of the density.
Copper impact (Favrie, Gavrilyuk & Saurel, JCP 228 (2009) 6037, Fig. 16; times 0.036, 0.106, 0.32, 0.609 ms; their
plate starts at x = 0.2 m, the deck's at 0.4 m):
  python3 deforming_solid_2d_check.py <dir> --unit GPa --title "Copper Plate Impact" --solid-color copper \
      --literature literature/favrie2009_fig16.png --lit-times 0.036,0.106,0.32,0.609 --lit-shift -0.2 \
      --lit-label "Favrie et al. (2009)"

There is no closed-form solution.  The script reports, for each run,
  * solid mass  M_s = sum (alpha rho)_0 dA  (conserved by the scheme; boundaries are open)
  * centroid (x_c, y_c) of the solid and its velocity
  * shape: bounding box of alpha_s > 0.5 (Lx, Ly) and the rms half-widths
    sqrt(<(x-x_c)^2>), sqrt(<(y-y_c)^2>) weighted by solid mass
  * surface positions on the centreline y = y_mid: first and last x with alpha_s > 0.5
  * elastic energy sum W dA, max |S| and the kinetic energy of the solid
and plots the body in its material colour (--solid-color) on a white background with the waves in the gas in
red (pressure-gradient schlieren of the gas), the shear stress S_xy, and the histories.  The elastic body should keep its shape
(Lx, Ly ring about their initial values) while the liquid one deforms without
recovery -- the behaviour of Favrie, Gavrilyuk & Saurel, JCP 228 (2009), Figs. 16-18.

Writes to <outdir> (default Images/<elastic_run_dir name>/ next to this script):
  deforming_solid_panels.png, deforming_solid_history.png, deforming_solid_summary.csv,
  Contours/frame_#####.png and Contours/deforming_solid.gif
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

FIELDS = ["eta", "rho_eta0", "density", "pressure", "velocityx", "velocityy", "elastic_Sxx", "elastic_Syy", "elastic_Szz", "elastic_Sxy", "elastic_W"]


def load(pf):
    ds = yt.load(pf)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    f = {k: np.asarray(cg[("boxlib", k)])[:, :, 0] for k in FIELDS}
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge)
    nx, ny = ds.domain_dimensions[:2]
    x = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    y = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return float(ds.current_time), f, x, y, (lo[0], hi[0], lo[1], hi[1])


def metrics(t, f, x, y):
    dA = (x[1] - x[0]) * (y[1] - y[0])
    m = f["rho_eta0"]; M = m.sum() * dA
    X, Y = np.meshgrid(x, y, indexing="ij")
    xc, yc = (m * X).sum() * dA / M, (m * Y).sum() * dA / M
    sol = f["eta"] > 0.5
    xs, ys = x[sol.any(axis=1)], y[sol.any(axis=0)]
    jm = len(y) // 2
    cl = x[sol[:, jm] | sol[:, jm - 1]]
    S = np.sqrt(f["elastic_Sxx"]**2 + f["elastic_Syy"]**2 + 2 * f["elastic_Sxy"]**2)
    return [t, M, xc, yc, xs.max() - xs.min() if xs.size else 0.0, ys.max() - ys.min() if ys.size else 0.0,
            np.sqrt((m * (X - xc)**2).sum() * dA / M), np.sqrt((m * (Y - yc)**2).sum() * dA / M),
            cl.min() if cl.size else np.nan, cl.max() if cl.size else np.nan,
            f["elastic_W"].sum() * dA, S.max(),
            0.5 * (m * (f["velocityx"]**2 + f["velocityy"]**2)).sum() * dA,
            (m * f["velocityx"]).sum() * dA / M, f["pressure"].min(), int(not np.isfinite(f["density"]).all())]


COLS = ["t", "M_solid", "x_c", "y_c", "Lx", "Ly", "rms_x", "rms_y", "x_front", "x_back", "W_total", "S_max", "KE_solid", "u_c", "p_min", "nan"]


SOLID_COLORS = {"copper": "#B87333", "aluminum": "#C0C0C0", "aluminium": "#C0C0C0", "silver": "#C0C0C0", "steel": "#8A8D8F", "grey": "0.6"}
SOLID_COLOR = "0.6"          # set by --solid-color (a name of SOLID_COLORS or any matplotlib colour)


WAVE_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("white_red", ["#FFFFFF", "#F7B7B2", "#C81E1E"])


def waves(f, x, y):
    """waves in the FLUID, 0 (uniform) .. 1 (strong): |grad p| normalised by its 99.5th percentile over the pure gas.
    Pressure, unlike density, has no jump at the smeared solid|gas interface, so no halo is left around the body;
    the faint mixture band next to the solid (alpha_s > 0.02) is blanked because it carries solid-sized stresses."""
    gx, gy = np.gradient(f["pressure"], x, y); g = np.hypot(gx, gy)
    gas = f["eta"] < 1e-3; ref = np.percentile(g[gas], 99.5) if gas.any() else 0.0
    if gas.any(): ref = max(ref, 1e-2 * np.median(np.abs(f["pressure"][gas])) / (x[1] - x[0]))   # undisturbed gas: no round-off pattern
    w = 1.0 - np.exp(-1.2 * g / ref) if ref > 0 else np.zeros_like(g)
    return np.where(f["eta"] > 0.02, 0.0, w)


REF_SOLID = [None]            # solid schlieren scale shared by the panels / GIF frames of one script run


def schlieren_grey(ax, f, x, y, ext, dx=0.0, ref_s=None):
    """the published look (Favrie et al. 2009, Fig. 16): grey-scale schlieren of the density -- light grey
    background, waves in the gas as darker lines, the solid dark grey with a black edge AND its internal waves.
    Gas: |grad ln rho| of the mixture, scaled by its own 99.5th percentile.
    Solid: |grad rho_s| of the SOLID PHASE density rho_s = (alpha rho)_s / alpha_s -- it changes only by compression
    (~0.1 % in a stress wave) and, unlike the mixture density, has no jump across the smeared surface.  Scaled by
    ref_s (97th percentile over the body if not given; pass the value returned for the first panel to the later
    ones so that weaker, later waves are not amplified).  Returns (image, ref_s)."""
    rho = np.maximum(f["density"], 1e-30); gx, gy = np.gradient(np.log(rho), x, y); gl = np.hypot(gx, gy)
    gas = f["eta"] < 1e-3; ref_g = np.percentile(gl[gas], 99.5) if gas.any() else 1.0
    ref_g = max(ref_g, 1e-4 / (x[1] - x[0]))                       # undisturbed gas stays flat
    body = f["eta"] > 0.5; rs = np.where(f["eta"] > 0.05, f["rho_eta0"] / np.maximum(f["eta"], 0.05), np.nan)
    rs = np.where(np.isfinite(rs), rs, np.nanmean(rs[body]) if body.any() else 0.0)
    sx, sy = np.gradient(rs, x, y); gs = np.hypot(sx, sy)
    if ref_s is None:
        floor = 1e-4 * rs[body].mean() / (x[1] - x[0]) if body.any() else 1.0
        ref_s = max(np.percentile(gs[body], 97.0), floor) if body.any() else 1.0
        if ref_s > floor and REF_SOLID[0] is None: REF_SOLID[0] = ref_s          # first disturbed frame sets the scale
    sol = np.clip((f["eta"] - 0.1) / 0.8, 0.0, 1.0)
    shade = (1.0 - sol) * 0.94 * np.exp(-1.5 * gl / ref_g) + sol * 0.70 * np.exp(-3.0 * gs / ref_s)
    # dx shifts the picture in x so the body sits where it does in the published panel (blank strip: background grey)
    ax.set_facecolor("0.94")
    im = ax.imshow(shade.T, origin="lower", extent=(ext[0] + dx, ext[1] + dx, ext[2], ext[3]), cmap="gray", vmin=0, vmax=1)
    ax.contour(x + dx, y, f["eta"].T, levels=[0.5], colors="k", linewidths=1.0); ax.set_aspect("equal")
    return im, ref_s


def panel(ax, f, x, y, ext, what):
    if what == "rho":
        # grey-scale density schlieren, the same look as the literature comparison (schlieren_grey)
        im, _ = schlieren_grey(ax, f, x, y, ext, ref_s=REF_SOLID[0])
    else:
        v = max(1e-12, np.abs(f["elastic_Sxy"]).max())
        im = ax.imshow(f["elastic_Sxy"].T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.contour(x, y, f["eta"].T, levels=[0.5], colors="k", linewidths=0.8)
    ax.set_aspect("equal")
    return im


def main():
    args = sys.argv[1:]
    OPTS = ("--outdir", "--stride", "--unit", "--title", "--literature", "--lit-times", "--lit-shift", "--lit-label", "--solid-color")
    skip = {args.index(k) + 1 for k in OPTS if k in args}
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    global SOLID_COLOR
    SOLID_COLOR = SOLID_COLORS.get(opt("--solid-color", "grey").lower(), opt("--solid-color", "0.6"))
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[0]))))
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 1
    make_gif = "--no-gif" not in args
    cdir = os.path.join(outdir, "Contours"); os.makedirs(cdir, exist_ok=True)
    labels = ["elastic", "liquid ($\\mu$ = 0)"][:len(dirs)] if len(dirs) > 1 else ["Numerical"]
    pfs = [sorted(p for p in glob.glob(os.path.join(d, "output", "*cell")) if re.search(r"\d+cell$", p)) for d in dirs]
    nfr = min(len(p) for p in pfs)
    H = [[] for _ in dirs]; frames = []; snaps = {}
    want = set(np.linspace(0, nfr - 1, 4).round().astype(int))
    for n in range(nfr):
        data = [load(p[n]) for p in pfs]
        for k, (t, f, x, y, ext) in enumerate(data): H[k].append(metrics(t, f, x, y))
        if n in want: snaps[n] = data
        if make_gif and n % stride == 0:
            fig, ax = plt.subplots(len(dirs), 1, figsize=(8, 4.2 * len(dirs)), squeeze=False)
            for k, (t, f, x, y, ext) in enumerate(data):
                im = panel(ax[k, 0], f, x, y, ext, "rho"); ax[k, 0].set_title(f"{labels[k]}   t = {t:6.3f}")
                fig.colorbar(im, ax=ax[k, 0], label="Numerical Schlieren", shrink=0.85)
            fn = os.path.join(cdir, f"frame_{n:05d}.png"); fig.savefig(fn, dpi=100); plt.close(fig); frames.append(fn)
    H = [np.array(h) for h in H]

    with open(os.path.join(outdir, "deforming_solid_summary.csv"), "w") as fo:
        fo.write("run," + ",".join(COLS) + "\n")
        for k, h in enumerate(H):
            for r in h: fo.write(("elastic" if k == 0 else "liquid") + "," + ",".join(f"{v:.6g}" for v in r) + "\n")
    for k, h in enumerate(H):
        c = {n: h[:, i] for i, n in enumerate(COLS)}
        print(f"{dirs[k]}  ({'elastic' if k == 0 else 'liquid'}), t = {c['t'][-1]:.3f}, {len(h)} plot files")
        print(f"  NaN frames {int(c['nan'].sum())};  p_min {c['p_min'].min():.4g};  solid mass drift {100 * (c['M_solid'][-1] / c['M_solid'][0] - 1):+.4f} %"
              f" (max {100 * np.abs(c['M_solid'] / c['M_solid'][0] - 1).max():.4f} %)")
        print(f"  centroid x {c['x_c'][0]:.4f} -> {c['x_c'][-1]:.4f}, final solid velocity {c['u_c'][-1]:.4f}")
        print(f"  bounding box Lx {c['Lx'][0]:.4f} -> {c['Lx'][-1]:.4f} (min {c['Lx'].min():.4f}, max {c['Lx'].max():.4f});"
              f"  Ly {c['Ly'][0]:.4f} -> {c['Ly'][-1]:.4f} (min {c['Ly'].min():.4f}, max {c['Ly'].max():.4f})")
        print(f"  rms half-widths x {c['rms_x'][0]:.4f} -> {c['rms_x'][-1]:.4f}, y {c['rms_y'][0]:.4f} -> {c['rms_y'][-1]:.4f}")
        print(f"  max elastic energy {c['W_total'].max():.4g}, max |S| {c['S_max'].max():.4g}")

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for k, h in enumerate(H):
        c = {n: h[:, i] for i, n in enumerate(COLS)}; ls = "-" if k == 0 else "--"
        ax[0, 0].plot(c["t"], c["Lx"], "C0" + ls, label=f"$L_x$ {labels[k]}"); ax[0, 0].plot(c["t"], c["Ly"], "C3" + ls, label=f"$L_y$ {labels[k]}")
        ax[0, 1].plot(c["t"], c["rms_x"], "C0" + ls, label=f"rms x {labels[k]}"); ax[0, 1].plot(c["t"], c["rms_y"], "C3" + ls, label=f"rms y {labels[k]}")
        ax[1, 0].plot(c["t"], c["x_front"], "C0" + ls, label=f"front {labels[k]}"); ax[1, 0].plot(c["t"], c["x_back"], "C3" + ls, label=f"back {labels[k]}")
        ax[1, 1].plot(c["t"], c["u_c"], "k" + ls, label=labels[k])
        ax[2, 0].plot(c["t"], c["W_total"], "C2" + ls, label=f"elastic energy {labels[k]}"); ax[2, 0].plot(c["t"], c["KE_solid"], "C1" + ls, label=f"solid KE {labels[k]}")
        ax[2, 1].plot(c["t"], 100 * (c["M_solid"] / c["M_solid"][0] - 1), "k" + ls, label=labels[k])
    for a, yl in zip(ax.ravel(), ["bounding box of $\\alpha_s>0.5$", "mass-weighted rms half-width", "surface x on the centreline",
                                  "solid centroid velocity", "energy", "solid mass drift [%]"]):
        a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=7)
    for a in ax[2]: a.set_xlabel("t")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "deforming_solid_history.png"), dpi=200); plt.close(fig)

    ns = sorted(snaps); nr = 2 * len(dirs)
    fig, ax = plt.subplots(nr, len(ns), figsize=(4.6 * len(ns), 2.6 * nr * (1 if snaps[ns[0]][0][4][1] > 1.5 else 1.7)), squeeze=False)
    for j, n in enumerate(ns):
        for k, (t, f, x, y, ext) in enumerate(snaps[n]):
            panel(ax[2 * k, j], f, x, y, ext, "rho"); ax[2 * k, j].set_title(("Numerical Schlieren" if len(dirs) == 1 else f"{labels[k]}: Numerical Schlieren") + f", t = {t:.3f}", fontsize=9)
            panel(ax[2 * k + 1, j], f, x, y, ext, "sxy"); ax[2 * k + 1, j].set_title(f"{labels[k]}: $S_{{xy}}$", fontsize=9)
    for a in ax.ravel(): a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "deforming_solid_panels.png"), dpi=170); plt.close(fig)

    # ---- the elastic body alone: stresses inside the solid at four times
    unit = opt("--unit"); ulab = f" [{unit}]" if unit else ""; name = opt("--title", os.path.basename(os.path.normpath(dirs[0])))
    S0 = [snaps[n][0] for n in ns]
    def vm(f): return np.sqrt(1.5 * (f["elastic_Sxx"]**2 + f["elastic_Syy"]**2 + f["elastic_Szz"]**2 + 2 * f["elastic_Sxy"]**2))
    comps = [("elastic_Sxx", r"$S_{xx}$"), ("elastic_Syy", r"$S_{yy}$"), ("elastic_Sxy", r"$S_{xy}$"), (None, "von Mises stress")]
    sol_all = np.any([f["eta"] > 0.5 for _, f, _, _, _ in S0], axis=0); xs0, ys0 = S0[0][2], S0[0][3]
    ix, iy = np.where(sol_all.any(axis=1))[0], np.where(sol_all.any(axis=0))[0]; pad = 0.08 * max(xs0[ix[-1]] - xs0[ix[0]], ys0[iy[-1]] - ys0[iy[0]])
    win = (xs0[ix[0]] - pad, xs0[ix[-1]] + pad, ys0[iy[0]] - pad, ys0[iy[-1]] + pad)
    fig, ax = plt.subplots(len(comps), len(S0), figsize=(3.1 * len(S0) * (win[1] - win[0]) / (win[3] - win[2]) + 2.2, 3.1 * len(comps)), squeeze=False)
    for i, (key, lab) in enumerate(comps):
        vals = [np.where(f["eta"] > 0.5, vm(f) if key is None else f[key], np.nan) for _, f, _, _, _ in S0]
        v = max(1e-12, np.nanmax([np.nanmax(np.abs(q)) for q in vals]))
        for j, ((t, f, x, y, ext), q) in enumerate(zip(S0, vals)):
            im = ax[i, j].imshow(q.T, origin="lower", extent=ext, cmap="viridis" if key is None else "RdBu_r", vmin=0 if key is None else -v, vmax=v)
            ax[i, j].contour(x, y, f["eta"].T, levels=[0.5], colors="k", linewidths=0.7)
            ax[i, j].set_xlim(win[0], win[1]); ax[i, j].set_ylim(win[2], win[3]); ax[i, j].set_aspect("equal"); ax[i, j].tick_params(labelsize=7)
            if i == 0: ax[i, j].set_title(f"t = {t:.3g}", fontsize=10)
            if j == 0: ax[i, j].set_ylabel("y")
            if i == len(comps) - 1: ax[i, j].set_xlabel("x")
        fig.colorbar(im, ax=ax[i, :], label=lab + ulab, shrink=0.9, pad=0.02)
    fig.suptitle(f"Stress in an Elastic Solid {name}", fontsize=13, y=0.915)
    fig.savefig(os.path.join(outdir, "deforming_solid_stress.png"), dpi=170, bbox_inches="tight"); plt.close(fig)

    # ---- comparison with a published 2 x 2 image of the same problem
    if opt("--literature"):
        from PIL import Image
        lit = np.asarray(Image.open(opt("--literature")).convert("L")).astype(float); lt = [float(v) for v in opt("--lit-times").split(",")]
        dxs = float(opt("--lit-shift", 0.0)); src = opt("--lit-label", "literature")
        def split(a, axis):                      # the two panels along an axis: runs of non-white rows / columns
            w = (a < 250).mean(axis=1 - axis) > 0.02; d = np.diff(np.concatenate([[0], w.astype(int), [0]]))
            st, en = np.where(d == 1)[0], np.where(d == -1)[0]; k = np.argsort(en - st)[-2:]; return sorted(zip(st[k], en[k]))
        R, C = split(lit, 0), split(lit, 1); panels = [lit[r0:r1, c0:c1] for r0, r1 in R for c0, c1 in C]
        times = np.array([float(yt.load(p).current_time) for p in pfs[0][:nfr]])
        fig, ax = plt.subplots(2, len(lt), figsize=(3.6 * len(lt), 7.6), squeeze=False)
        ref_solid = None
        for j, tl in enumerate(lt):
            n = int(np.argmin(np.abs(times - tl))); t, f, x, y, ext = load(pfs[0][n]); L = (ext[0], ext[1], ext[2], ext[3])
            ax[0, j].imshow(panels[j], cmap="gray", vmin=0, vmax=255, extent=L, origin="upper")
            ax[0, j].contour(x + dxs, y, f["eta"].T, levels=[0.5], colors="r", linewidths=1.0)
            ax[0, j].set_title(f"{src}, t = {tl:.3g}", fontsize=9)
            _, ref_solid = schlieren_grey(ax[1, j], f, x, y, ext, dxs, ref_solid)
            ax[1, j].set_title(f"Numerical, t = {t:.3g}", fontsize=9)
            for a in ax[:, j]: a.set_aspect("equal"); a.tick_params(labelsize=7); a.set_xlim(L[0], L[1]); a.set_ylim(L[2], L[3])
        fig.suptitle(f"{name}: Comparison with {src}", fontsize=13)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "deforming_solid_vs_literature.png"), dpi=170); plt.close(fig)

    if make_gif and frames:
        from PIL import Image
        ims = [Image.open(fn).convert("RGB") for fn in frames]
        pal = ims[len(ims) // 2].quantize(colors=256, dither=Image.Dither.NONE)   # one shared palette: no colour flicker
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(cdir, "deforming_solid.gif"), save_all=True, append_images=ims[1:], duration=80, loop=0)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
