#!/usr/bin/env python3
"""
Shock / water-droplet atomization of tests/FlowShockDroplet/Atomization (input_2mm_Ma3):
run with phase change (phasechange.on = 1) vs the same run without it.

  python3 atomization_check.py <run_dir> [<reference_run_dir>] [--outdir DIR] [--no-gif] [--stride N]

There is no closed-form solution.  For each run and plot file the script reports
  * liquid mass  M_l = sum (alpha rho)_liquid dA  and, with phase change, the vapour mass sum (vapor) dA;
    M_l + M_v is conserved by the scheme until material leaves the domain
  * droplet centroid x_c (liquid-mass weighted), its velocity, and the drift
    x_c - x_c(0) in diameters against the breakup time  t* = t u_g sqrt(rho_g/rho_l) / D
    (u_g, rho_g: post-shock gas read from the inflow side of the first plot file)
  * streamwise and cross-stream extent of alpha_l > 0.5
  * max temperature, min / max pressure, and the lowest pressure inside the liquid (alpha_l > 0.9)
  * with phase change: where gas has appeared inside the liquid, |p_v/p_sat(T) - 1| from the
    solver's own diagnostic T_sat (pc_T == pc_Tsat there)
and plots numerical schlieren, pressure, temperature and (with phase change) the vapour mass
fraction of the gas, each with the alpha = 0.5 outline.

Writes to <outdir> (default Images/<run_dir name>/ next to this script):
  atomization_panels.png, atomization_history.png, atomization_summary.csv,
  Contours/frame_#####.png and Contours/atomization.gif
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

D0 = 2.0e-3
WANT = ("eta", "rho_eta0", "rho_eta1", "density", "pressure", "velocityx", "velocityy", "T", "vapor", "pc_T", "pc_Tsat", "pc_Yv")


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    names = {f[1] for f in ds.field_list}
    f = {k: np.asarray(cg[("boxlib", k)])[:, :, 0] for k in WANT if k in names}
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge); nx, ny = ds.domain_dimensions[:2]
    x = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx; y = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    return float(ds.current_time), f, x, y, (1e3 * lo[0], 1e3 * hi[0], 1e3 * lo[1], 1e3 * hi[1])


COLS = ["t", "M_liquid", "M_vapor", "x_c", "u_c", "Lx", "Ly", "T_max", "p_min", "p_max", "p_min_liquid", "gas_in_liquid", "sat_err", "nan"]


def metrics(t, f, x, y):
    dA = (x[1] - x[0]) * (y[1] - y[0]); X, Y = np.meshgrid(x, y, indexing="ij")
    ml = f["rho_eta1"]; M = ml.sum() * dA; al = 1.0 - f["eta"]
    liq = al > 0.5; xs, ys = x[liq.any(axis=1)], y[liq.any(axis=0)]
    T = f.get("pc_T", f.get("T")); core = al > 0.9
    Mv = f["vapor"].sum() * dA if "vapor" in f else 0.0
    sat = np.nan
    if "pc_Tsat" in f:                                   # mixture cells holding vapour: saturated means T == T_sat(p_v)
        m = (al > 0.05) & (al < 0.95) & (f["pc_Yv"] > 0.5) & (f["pc_Tsat"] > 0)
        if m.any(): sat = np.abs(f["pc_T"][m] / f["pc_Tsat"][m] - 1).max()
    return [t, M, Mv, (ml * X).sum() * dA / M, (ml * f["velocityx"]).sum() * dA / M,
            xs.max() - xs.min() if xs.size else 0.0, ys.max() - ys.min() if ys.size else 0.0,
            np.nanmax(T) if T is not None else np.nan, f["pressure"].min(), f["pressure"].max(),
            f["pressure"][core].min() if core.any() else np.nan,
            ((f["eta"] > 0.01) & (np.hypot(X - (ml * X).sum() * dA / M, Y) < 0.25 * D0)).sum() * dA, sat,
            int(not np.isfinite(f["density"]).all())]


def schlieren(f, x, y):
    gx, gy = np.gradient(f["density"], x, y); g = np.hypot(gx, gy)
    return np.exp(-20.0 * g / max(g.max(), 1e-30))


def panel(ax, f, x, y, ext, what):
    if what == "schlieren": im = ax.imshow(schlieren(f, x, y).T, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=1)
    elif what == "p": im = ax.imshow(1e-5 * f["pressure"].T, origin="lower", extent=ext, cmap="viridis", vmin=0, vmax=20)
    elif what == "T": im = ax.imshow(f.get("pc_T", f.get("T")).T, origin="lower", extent=ext, cmap="inferno", vmin=280, vmax=800)
    else: im = ax.imshow(f["pc_Yv"].T, origin="lower", extent=ext, cmap="magma", vmin=0, vmax=1)
    ax.contour(1e3 * x, 1e3 * y, f["eta"].T, levels=[0.5], colors="c", linewidths=0.6); ax.set_aspect("equal")
    return im


def main():
    args = sys.argv[1:]
    skip = {args.index(k) + 1 for k in ("--outdir", "--stride") if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[0]))))
    stride = int(args[args.index("--stride") + 1]) if "--stride" in args else 1
    cdir = os.path.join(outdir, "Contours"); os.makedirs(cdir, exist_ok=True)
    pfs = [sorted((p for p in glob.glob(os.path.join(d, "output", "*cell")) if re.search(r"\d+cell$", p)),
                  key=lambda p: int(os.path.basename(p)[:-4])) for d in dirs]
    labels = ["phase change" if "pc_T" in open(os.path.join(p[0], "Header")).read().split() else "no phase change" for p in pfs]
    nfr = min(len(p) for p in pfs); H = [[] for _ in dirs]; frames = []; snaps = {}
    want = set(np.linspace(0, nfr - 1, 4).round().astype(int)); ug = rg = None
    for n in range(nfr):
        data = [load(p[n]) for p in pfs]
        if n == 0:
            f0 = data[0][1]; ug = f0["velocityx"][0, 0]; rg = f0["density"][0, 0]; rl = f0["density"].max()
        for k, (t, f, x, y, ext) in enumerate(data): H[k].append(metrics(t, f, x, y))
        if n in want: snaps[n] = data
        if "--no-gif" not in args and n % stride == 0:
            kinds = ["schlieren", "p", "T"] + (["Yv"] if "pc_Yv" in data[0][1] else [])
            fig, ax = plt.subplots(len(kinds), 1, figsize=(9, 3.6 * len(kinds)), squeeze=False)
            t, f, x, y, ext = data[0]
            for a, kd, lb in zip(ax[:, 0], kinds, ["numerical schlieren", "pressure [bar]", "temperature [K]", "vapour mass fraction of the gas"]):
                im = panel(a, f, x, y, ext, kd); a.set_title(f"{lb},  t = {1e6 * t:5.1f} $\\mu$s", fontsize=10); fig.colorbar(im, ax=a, shrink=0.85)
                a.set_ylabel("y [mm]")
            ax[-1, 0].set_xlabel("x [mm]"); fig.tight_layout()
            fn = os.path.join(cdir, f"frame_{n:05d}.png"); fig.savefig(fn, dpi=90); plt.close(fig); frames.append(fn)
    H = [np.array(h) for h in H]; tstar = lambda t: t * ug * np.sqrt(rg / rl) / D0

    with open(os.path.join(outdir, "atomization_summary.csv"), "w") as fo:
        fo.write("run,tstar," + ",".join(COLS) + "\n")
        for k, h in enumerate(H):
            for r in h: fo.write(labels[k].replace(" ", "_") + f",{tstar(r[0]):.5g}," + ",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"post-shock gas u_g = {ug:.1f} m/s, rho_g = {rg:.3f} kg/m3, rho_l = {rl:.1f};  t* = 1 at t = {1e6 * D0 / (ug * np.sqrt(rg / rl)):.1f} us")
    for k, h in enumerate(H):
        c = {n: h[:, i] for i, n in enumerate(COLS)}
        print(f"{dirs[k]}  ({labels[k]}): {len(h)} plot files to t = {1e6 * c['t'][-1]:.1f} us (t* = {tstar(c['t'][-1]):.3f})")
        print(f"  NaN frames {int(c['nan'].sum())};  liquid mass {c['M_liquid'][0]:.6e} -> {c['M_liquid'][-1]:.6e} ({100 * (c['M_liquid'][-1] / c['M_liquid'][0] - 1):+.4f} %);"
              f"  vapour mass {c['M_vapor'][0]:.3e} -> {c['M_vapor'][-1]:.3e};  liquid + vapour {100 * ((c['M_liquid'][-1] + c['M_vapor'][-1]) / (c['M_liquid'][0] + c['M_vapor'][0]) - 1):+.5f} %")
        print(f"  centroid drift {(c['x_c'][-1] - c['x_c'][0]) / D0:.4f} D, velocity {c['u_c'][-1]:.2f} m/s;  extent Lx {c['Lx'][0] / D0:.3f} -> {c['Lx'][-1] / D0:.3f} D,"
              f"  Ly {c['Ly'][0] / D0:.3f} -> {c['Ly'][-1] / D0:.3f} D")
        print(f"  T_max {c['T_max'].max():.1f} K;  p_max {1e-5 * c['p_max'].max():.2f} bar;  p_min {c['p_min'].min():.4g} Pa;  lowest pressure in the liquid {np.nanmin(c['p_min_liquid']):.4g} Pa"
              f" at t = {1e6 * c['t'][np.nanargmin(c['p_min_liquid'])]:.1f} us")
        if np.isfinite(c["sat_err"]).any(): print(f"  max |T/T_sat - 1| in vapour-bearing mixture cells {np.nanmax(c['sat_err']):.2e}")

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for k, h in enumerate(H):
        c = {n: h[:, i] for i, n in enumerate(COLS)}; ls = "-" if k == 0 else "--"; tu = 1e6 * c["t"]
        ax[0, 0].plot(tu, (c["x_c"] - c["x_c"][0]) / D0, "k" + ls, label=labels[k])
        ax[0, 1].plot(tu, c["Lx"] / D0, "C0" + ls, label=f"streamwise, {labels[k]}"); ax[0, 1].plot(tu, c["Ly"] / D0, "C3" + ls, label=f"cross-stream, {labels[k]}")
        ax[0, 2].plot(tu, 100 * (c["M_liquid"] / c["M_liquid"][0] - 1), "k" + ls, label=f"liquid, {labels[k]}")
        if c["M_vapor"].max() > 0: ax[0, 2].plot(tu, 100 * c["M_vapor"] / c["M_liquid"][0], "C1" + ls, label=f"vapour, {labels[k]}")
        ax[1, 0].plot(tu, c["T_max"], "C3" + ls, label=labels[k])
        ax[1, 1].semilogy(tu, np.maximum(c["p_min_liquid"], 1.0), "C0" + ls, label=f"min in liquid, {labels[k]}"); ax[1, 1].semilogy(tu, c["p_max"], "C3" + ls, label=f"max, {labels[k]}")
        ax[1, 2].plot(tu, c["u_c"], "k" + ls, label=labels[k])
    for a, yl in zip(ax.ravel(), ["centroid drift [D]", "extent of $\\alpha_l>0.5$ [D]", "mass change [% of initial liquid]", "max temperature [K]", "pressure [Pa]", "droplet centroid velocity [m/s]"]):
        a.set_ylabel(yl); a.grid(alpha=0.3); a.legend(fontsize=7)
    for a in ax[1]: a.set_xlabel("t [$\\mu$s]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "atomization_history.png"), dpi=180); plt.close(fig)

    ns = sorted(snaps); kinds = ["schlieren", "p", "T"]; rows = [(k, kd) for kd in kinds for k in range(len(dirs))] + ([(0, "Yv")] if "pc_Yv" in snaps[ns[0]][0][1] else [])
    fig, ax = plt.subplots(len(rows), len(ns), figsize=(5.2 * len(ns), 2.9 * len(rows)), squeeze=False)
    for j, n in enumerate(ns):
        for i, (k, kd) in enumerate(rows):
            t, f, x, y, ext = snaps[n][k]; im = panel(ax[i, j], f, x, y, ext, kd)
            ax[i, j].set_title(f"{labels[k]}: {dict(schlieren='schlieren', p='p [bar]', T='T [K]', Yv='vapour mass fraction')[kd]}, t = {1e6 * t:.0f} $\\mu$s", fontsize=8)
            ax[i, j].tick_params(labelsize=6); fig.colorbar(im, ax=ax[i, j], shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "atomization_panels.png"), dpi=150); plt.close(fig)

    if frames:
        from PIL import Image
        ims = [Image.open(fn).convert("RGB") for fn in frames]
        pal = ims[len(ims) // 2].quantize(colors=256, dither=Image.Dither.NONE)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(cdir, "atomization.gif"), save_all=True, append_images=ims[1:], duration=90, loop=0)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
