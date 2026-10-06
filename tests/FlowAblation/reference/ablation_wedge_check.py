#!/usr/bin/env python3
"""
Ablating sharp wedge in a supersonic viscous stream (tests/FlowAblation/UNIT_TEST_2D_Wedge).

  python3 ablation_wedge_check.py <run_dir> [--outdir DIR] [--no-gif] [--stride N]

No closed-form solution for the shape.  Reported:
  * tip position x_tip(t) (first phi < 0.5 cell on the centreline, interpolated) and recession
  * nose radius R_n(t): least-squares circle through the phi = 0.5 contour within 2.5 dx..6 dx of the tip
  * peak wall heat flux q_max(t) and where it is; the laminar expectation is q ~ 1/sqrt(R_n) once blunt
  * ablated area (growth of the fluid area) against the time integral of the surface-integrated v_abl
    (consistency of the level-set recession with the heat flux that drives it)
Figures (written to <outdir>, default Images/<run_dir name>/ next to this script):
  ablation_panels.png      temperature, Mach number, pressure and density gradient (numerical
                           schlieren) at 4 times, zoomed on the nose, with the body outline
  ablation_shapes.png      the surface (phi = 0.5) every few plot files -- the tip rounding
  ablation_history.png     x_tip, R_n, q_max, ablated area
  ablation_wall_flux.png   heat flux along the surface at the same 4 times
  Contours/frame_#####.png, Contours/ablation_wedge.gif   temperature + outline + streamlines
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

GAM, TINF, X0 = 1.4, 2.4876e-3, -0.5
ZOOM = (-0.75, 0.05, -0.3, 0.3)
FIELDS = ["phi", "T", "pressure", "density", "velocityx", "velocityy", "ablation_q", "ablation_v", "a"]


TWALL = TINF          # wall (= whole-body) temperature of the deck: ablation.T_wall = T_inf
SOLID_CMAP = "YlGnBu_r"            # solid temperature (in-depth conduction runs): distinct from the gas map
SOLID_TMAX = [1.5]                 # upper end of the solid colour bar, T_abl / T_inf (read from the run's metadata)
SOLID_LABEL = r"Solid Temperature $T_s/T_\infty$"


def load(pf):
    ds = yt.load(pf)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    f = {k: np.asarray(cg[("boxlib", k)])[:, :, 0] for k in FIELDS}
    if ("boxlib", "ablation_Ts") in ds.field_list:          # in-depth conduction run (ablation.conduction = 1)
        f["Ts"] = np.asarray(cg[("boxlib", "ablation_Ts")])[:, :, 0]
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge)
    nx, ny = ds.domain_dimensions[:2]
    x = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    y = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return float(ds.current_time), f, x, y


def contour_pts(x, y, phi):
    cs = plt.figure().gca().contour(x, y, phi.T, levels=[0.5]); plt.close()
    segs = cs.allsegs[0]
    return max(segs, key=len) if segs else np.zeros((0, 2))


def nose(x, y, phi):
    dx = x[1] - x[0]; jm = len(y) // 2
    col = 0.5 * (phi[:, jm] + phi[:, jm - 1]); i = np.where(col < 0.5)[0][0]
    xt = x[i - 1] + dx * (col[i - 1] - 0.5) / (col[i - 1] - col[i])
    P = contour_pts(x, y, phi)
    if len(P) == 0: return xt, np.nan, P
    r = np.hypot(P[:, 0] - xt, P[:, 1])
    Q = P[r < 6 * dx]
    if len(Q) < 6: return xt, np.nan, P
    # circle through the tip, centre on the axis: (x - xt - R)^2 + y^2 = R^2  ->  R = (X^2 + y^2)/(2 X)
    X = Q[:, 0] - xt; m = X > 0.3 * dx
    R = np.median((X[m]**2 + Q[m, 1]**2) / (2 * X[m])) if m.sum() >= 3 else np.nan
    return xt, R, P


def panel(ax, f, x, y, what):
    fl = f["phi"] >= 0.5
    if what == "T":
        # without in-depth conduction the solid is drawn at the wall temperature (the model holds the whole body at
        # T_wall, uniform by construction); with ablation.conduction = 1 the solid temperature field is drawn with its
        # own colour map and colour bar (ax.solid_im)
        q = np.where(fl, f["T"] / TINF, TWALL / TINF if "Ts" not in f else np.nan)
        im = ax.imshow(q.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="inferno", vmin=1.0, vmax=2.9)
        ax.solid_im = None
        if "Ts" in f:
            ax.solid_im = ax.imshow(np.where(fl, np.nan, f["Ts"] / TINF).T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]),
                                    cmap=SOLID_CMAP, vmin=1.0, vmax=SOLID_TMAX[0])
        lab = r"$T/T_\infty$"
    elif what == "M":
        q = np.where(fl, np.hypot(f["velocityx"], f["velocityy"]) / f["a"], np.nan); im = ax.imshow(q.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="viridis", vmin=0, vmax=3.2)
        lab = "Mach number"
    elif what == "p":
        q = np.where(fl, f["pressure"] * GAM, np.nan); im = ax.imshow(q.T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="cividis", vmin=0.8, vmax=12.5)
        lab = r"$p/p_\infty$"
    else:
        gx, gy = np.gradient(f["density"], x, y); q = np.where(fl, np.hypot(gx, gy), np.nan)
        im = ax.imshow(np.log10(q + 1e-3).T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]), cmap="gray_r", vmin=-0.5, vmax=2.5)
        lab = "Numerical Schlieren"
    ax.contour(x, y, f["phi"].T, levels=[0.5], colors="c" if what != "s" else "r", linewidths=0.9)
    ax.set_facecolor("0.35"); ax.set_aspect("equal"); ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
    return im, lab


def main():
    args = sys.argv[1:]; d = args[0]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 1
    make_gif = "--no-gif" not in args
    cdir = os.path.join(outdir, "Contours"); os.makedirs(cdir, exist_ok=True)
    pfs = sorted(p for p in glob.glob(os.path.join(d, "output", "*cell")) if re.search(r"\d+cell$", p))
    for ln in open(os.path.join(d, "output", "metadata")):          # ablation temperature (last entry = command-line override)
        m = re.match(r"\s*ablation\.T_wall\s*=\s*(\S+)", ln)
        if m: SOLID_TMAX[0] = max(float(m.group(1)) / TINF, 1.0 + 1e-6)
    want = set(np.linspace(0, len(pfs) - 1, 4).round().astype(int)); snaps = {}; shapes = []; H = []; frames = []
    for n, pf in enumerate(pfs):
        t, f, x, y = load(pf); dA = (x[1] - x[0]) * (y[1] - y[0])
        xt, Rn, P = nose(x, y, f["phi"])
        k = np.unravel_index(f["ablation_q"].argmax(), f["ablation_q"].shape)
        gx, gy = np.gradient(f["phi"], x, y)
        H.append((t, xt, Rn, f["ablation_q"].max(), x[k[0]], y[k[1]], (f["phi"] >= 0.5).sum() * dA,
                  (f["ablation_v"] * np.hypot(gx, gy)).sum() * dA, f["T"][f["phi"] >= 0.5].max() / TINF, f["pressure"][f["phi"] >= 0.5].min()))
        if n in want: snaps[n] = (t, f, x, y)
        if n % max(1, len(pfs) // 10) == 0 or n == len(pfs) - 1: shapes.append((t, P))
        if make_gif and n % stride == 0:
            fig, ax = plt.subplots(figsize=(8.5, 5.6))
            im, lab = panel(ax, f, x, y, "T")
            ax.set_title(f"Ablating Wedge at Mach 3, t = {t:5.2f}, Nose Radius {Rn:.4f}" if np.isfinite(Rn) else f"Ablating Wedge at Mach 3, t = {t:5.2f}")
            ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(im, ax=ax, label=("Gas Temperature " if ax.solid_im is not None else "") + lab, shrink=0.85)
            if ax.solid_im is not None: fig.colorbar(ax.solid_im, ax=ax, label=SOLID_LABEL, location="bottom", shrink=0.6, pad=0.12, aspect=40)
            fn = os.path.join(cdir, f"frame_{n:05d}.png"); fig.savefig(fn, dpi=110); plt.close(fig); frames.append(fn)
    H = np.array(H); t = H[:, 0]
    abl = H[:, 6] - H[0, 6]; pred = np.concatenate([[0.0], np.cumsum(0.5 * (H[1:, 7] + H[:-1, 7]) * np.diff(t))])
    print(f"{d}: {len(pfs)} plot files to t = {t[-1]:.3f}")
    print(f"  tip: x = {H[0, 1]:.4f} -> {H[-1, 1]:.4f}  (recession {H[-1, 1] - H[0, 1]:.4f} = {(H[-1, 1] - H[0, 1]) / (x[1] - x[0]):.1f} cells)")
    print(f"  nose radius: {H[0, 2]:.4f} -> {H[-1, 2]:.4f}  ({H[-1, 2] / (x[1] - x[0]):.1f} cells)")
    print(f"  peak wall heat flux: {np.nanmax(H[:, 3]):.3f} early -> {H[-1, 3]:.3f} at the end, at (x, y) = ({H[-1, 4]:.3f}, {H[-1, 5]:+.3f})")
    print(f"  ablated area {abl[-1]:.5f};  time integral of the surface-integrated recession speed {pred[-1]:.5f}  ({100 * (abl[-1] / pred[-1] - 1) if pred[-1] > 0 else 0:+.1f} %)")
    print(f"  max T/T_inf in the gas {H[:, 8].max():.3f} (stagnation 2.8);  min pressure {H[:, 9].min():.4f}")
    with open(os.path.join(outdir, "ablation_wedge_summary.csv"), "w") as fo:
        fo.write("t,x_tip,nose_radius,q_max,x_qmax,y_qmax,fluid_area,recession_rate_integral,Tmax_over_Tinf,p_min\n")
        for r in H: fo.write(",".join(f"{v:.6g}" for v in r) + "\n")

    ns = sorted(snaps); fig, ax = plt.subplots(4, len(ns), figsize=(4.6 * len(ns), 13.5), squeeze=False)
    for j, n in enumerate(ns):
        ts, f, x, y = snaps[n]
        for i, what in enumerate(("T", "M", "p", "s")):
            im, lab = panel(ax[i, j], f, x, y, what); ax[i, j].set_title(f"{lab},  t = {ts:.2f}", fontsize=9)
            if j == len(ns) - 1: fig.colorbar(im, ax=ax[i, :].tolist(), shrink=0.85, pad=0.01)
            if j == len(ns) - 1 and what == "T" and ax[i, j].solid_im is not None:
                # solid colour bar inside the first (t = 0, dark) panel so the row keeps the size of the others
                cax = ax[i, 0].inset_axes([0.07, 0.13, 0.55, 0.05]); cb = fig.colorbar(ax[i, j].solid_im, cax=cax, orientation="horizontal")
                cb.ax.set_title(SOLID_LABEL, color="w", fontsize=8, pad=3); cb.ax.tick_params(colors="w", labelsize=7); cb.outline.set_edgecolor("w")
    fig.savefig(os.path.join(outdir, "ablation_panels.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

    # surface outlines coloured by time on a colour bar; the map stops before the pale end of "plasma" so that
    # every outline stays dark enough to print
    import matplotlib.colors as mcolors
    cm = mcolors.LinearSegmentedColormap.from_list("plasma_dark", plt.get_cmap("plasma")(np.linspace(0.0, 0.80, 256)))
    fig, ax = plt.subplots(figsize=(8.6, 6)); tmin, tmax = shapes[0][0], shapes[-1][0]; norm = mcolors.Normalize(tmin, tmax)
    for ts, P in shapes:
        if len(P): ax.plot(P[:, 0], P[:, 1], color=cm(norm(ts)), lw=1.4)
    ax.set_xlim(-0.56, -0.25); ax.set_ylim(-0.1, 0.1); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cm); sm.set_array([]); cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.03, aspect=22); cb.set_label("Time t")
    cb.set_ticks([ts for ts, _ in shapes][::2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title("Surface of the Ablating Wedge: The Tip Blunts")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "ablation_shapes.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax[0, 0].plot(t, H[:, 1], "k-"); ax[0, 0].set_ylabel("tip position $x_{tip}$")
    ax[0, 1].plot(t, H[:, 2], "k-"); ax[0, 1].set_ylabel("nose radius $R_n$")
    ax[1, 0].plot(t, H[:, 3], "k-", label="$q_{max}$")
    ok = np.isfinite(H[:, 2]) & (H[:, 2] > 2 * (x[1] - x[0])) & (H[:, 3] > 0)
    if ok.sum() > 3:
        c = (H[ok, 3] * np.sqrt(H[ok, 2])).mean(); ax[1, 0].plot(t[ok], c / np.sqrt(H[ok, 2]), "C3--", label=r"$\propto 1/\sqrt{R_n}$ (fitted constant)")
    ax[1, 0].set_ylabel("peak wall heat flux"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(t, abl, "k-", label="ablated area"); ax[1, 1].plot(t, pred, "C3--", label=r"$\int\!\!\int v_{abl}\,ds\,dt$"); ax[1, 1].legend(fontsize=8)
    for a in ax.ravel(): a.grid(alpha=0.3)
    for a in ax[1]: a.set_xlabel("t")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "ablation_history.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for n in ns:
        ts, f, x, y = snaps[n]; q = f["ablation_q"]; up = q[:, len(y) // 2:].max(axis=1); m = up > 0
        ax.semilogy(x[m], up[m], label=f"t = {ts:.2f}")
    ax.set_xlim(-0.56, 0.6); ax.set_xlabel("x"); ax.set_ylabel("wall heat flux $q_w$ (upper surface)"); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "ablation_wall_flux.png"), dpi=200); plt.close(fig)

    if make_gif and frames:
        from PIL import Image
        ims = [Image.open(fn).convert("RGB") for fn in frames]
        pal = ims[len(ims) // 2].quantize(colors=256, dither=Image.Dither.NONE)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(cdir, "ablation_wedge.gif"), save_all=True, append_images=ims[1:], duration=90, loop=0)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
