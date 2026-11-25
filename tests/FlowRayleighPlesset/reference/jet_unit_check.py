#!/usr/bin/env python3
"""
Axis-jet check for the single-level octant collapse unit test (see tests/FlowRayleighPlesset/input_JetUnit_3D).

  python3 jet_unit_check.py <run_dir> [<run_dir> ...] [--outdir DIR] [--zoom 0.45]

After the collapse a limiter that keeps the interface very thin (THINC) can throw the rebounding gas out as sheets
lying on the symmetry planes, along the coordinate axes.  For every plot file this prints the gas-volume radius and
the eta = 0.5 radius along the axes [100], the in-plane diagonals [110] and the body diagonal [111], and the
JET RATIO  R[100] / R[111]  (1 = round, > 1 = jets along the axes).  It also draws eta on the first z plane (which
contains the x and y axes) for the frames after the minimum, one row per run.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
from scipy.ndimage import map_coordinates
HERE = os.path.dirname(os.path.abspath(__file__))
R0 = 0.02


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def frame(pf):
    ds = yt.load(pf); lev = ds.index.max_level
    cg = ds.covering_grid(lev, ds.domain_left_edge, ds.domain_dimensions * 2 ** lev)
    eta = np.asarray(cg[("boxlib", "eta")]); n = eta.shape[0]; L = float(ds.domain_width[0]); dx = L / n
    V = 8.0 * (1.0 - eta).sum() * dx ** 3

    def rad(d):
        d = np.array(d, float); d /= np.linalg.norm(d); s = np.linspace(0, 0.97 * L, 3000)
        v = map_coordinates(eta, [s * d[k] / dx - 0.5 for k in range(3)], order=1, mode="nearest"); k = np.where(v < 0.5)[0]
        return 0.0 if len(k) == 0 else s[k.max()]                      # OUTERMOST gas along the ray (a jet tip counts)
    r100 = np.mean([rad(d) for d in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]); r110 = np.mean([rad(d) for d in ((1, 1, 0), (0, 1, 1), (1, 0, 1))]); r111 = rad((1, 1, 1))
    return dict(t=float(ds.current_time), Rv=(3 * V / (4 * np.pi)) ** (1 / 3) / R0, r100=r100 / R0, r110=r110 / R0, r111=r111 / R0, sl=eta[:, :, 0].copy(), dx=dx, L=L)


def main():
    args = sys.argv[1:]; skip = {args.index(k) + 1 for k in ("--outdir", "--zoom") if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "JetUnit")); os.makedirs(outdir, exist_ok=True); zoom = float(opt(args, "--zoom", "0.45"))
    rowsets = []
    for d in dirs:
        pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
        F = [frame(pf) for pf in pfs]; i = int(np.argmin([f["Rv"] for f in F])); tmin = F[i]["t"]
        print(f"== {d}: R_vol min {F[i]['Rv']:.4f} at {1e6 * tmin:.0f} us (plot-file sampling)")
        print("    t/t_min   R_vol   R[100]  R[110]  R[111]   jet ratio R[100]/R[111]")
        best = 0.0
        for f in F[max(i - 2, 0):]:
            jr = f["r100"] / f["r111"] if f["r111"] > 0 else float("nan")
            if f["t"] > tmin and np.isfinite(jr): best = max(best, jr)
            print(f"    {f['t'] / tmin:6.3f}   {f['Rv']:.4f}  {f['r100']:.4f}  {f['r110']:.4f}  {f['r111']:.4f}   {jr:6.2f}")
        print(f"    largest jet ratio after the minimum: {best:.2f}")
        rowsets.append((os.path.basename(os.path.normpath(d)), F[i:i + 7], tmin))
    nr, nc = len(rowsets), max(len(r[1]) for r in rowsets)
    fig, ax = plt.subplots(nr, nc, figsize=(2.55 * nc, 2.75 * nr + 0.5), squeeze=False)
    for r, (name, F, tmin) in enumerate(rowsets):
        for c in range(nc):
            a = ax[r, c]
            if c >= len(F): a.axis("off"); continue
            f = F[c]; m = int(zoom * R0 / f["dx"]) + 1
            a.imshow(f["sl"][:m, :m].T, origin="lower", extent=(0, m * f["dx"] / R0, 0, m * f["dx"] / R0), cmap="RdBu_r", vmin=0, vmax=1, interpolation="nearest")
            a.contour((np.arange(m) + 0.5) * f["dx"] / R0, (np.arange(m) + 0.5) * f["dx"] / R0, f["sl"][:m, :m].T, levels=[0.5], colors="k", linewidths=0.7)
            a.set_title(f"t/t$_{{min}}$ = {f['t'] / tmin:.3f}", fontsize=8); a.tick_params(labelsize=7)
            if c == 0: a.set_ylabel(name + "\ny / R$_0$ [-]", fontsize=8)
            if r == nr - 1: a.set_xlabel("x / R$_0$ [-]", fontsize=8)
    fig.suptitle(r"Volume Fraction on the First z Plane After the Minimum Radius (Numerical; Black Line: $\eta$ = 0.5)", fontsize=10)
    fig.tight_layout(); o = os.path.join(outdir, "jet_unit_slices.png"); fig.savefig(o, dpi=170); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
