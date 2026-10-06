#!/usr/bin/env python3
"""
2D cavitation bubble inception (tests/FlowPhaseChange/input_2D_CavitationBubble).

  python3 cavitation_bubble_check.py <run_dir> [<run_dir without phase change>] [--outdir DIR] [--no-gif]

Reported per plot file: cavity area (vapour volume fraction > 0.5) and equivalent radius, centre
pressure and temperature, the saturation check p / p_sat(T) in the cavity (independent scipy
evaluation of g_liq = g_vap), total vapour mass, departure from circular symmetry.
Figures: cavitation_bubble_panels.png (vapour fraction, pressure, temperature, vapour mass
fraction at 4 times), cavitation_bubble_history.png, Contours/cavitation_bubble.gif.
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
L = dict(g=2.35, pi=1.0e9, cv=1816.0, q=-1167.0e3, qp=0.0)
V = dict(g=1.43, pi=0.0, cv=1040.0, q=2030.0e3, qp=-23.4e3)
gib = lambda m, p, T: (m["g"] * m["cv"] - m["qp"]) * T - m["cv"] * T * (m["g"] * np.log(T) - (m["g"] - 1) * np.log(p + m["pi"])) + m["q"]
psat = lambda T: brentq(lambda p: gib(L, p, T) - gib(V, p, T), 1.0, 5.0e7)


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; lo, w = float(ds.domain_left_edge[0]), float(ds.domain_width[0])
    x = lo + (np.arange(n) + 0.5) * w / n
    names = [f[1] for f in ds.field_list]
    data = {nm: np.asarray(cg[("boxlib", nm)])[:, :, 0] for nm in ("eta", "pressure", "pc_T", "pc_Yv", "vapor") if nm in names}
    return float(ds.current_time), x, data.get


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0]; d0 = args[1] if len(args) > 1 and "--outdir" not in sys.argv[2:3] else None
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    cdir = os.path.join(outdir, "Contours"); os.makedirs(cdir, exist_ok=True)
    pfs = sorted(p for p in glob.glob(os.path.join(d, "output", "*cell")) if re.search(r"\d+cell$", p))
    pf0 = sorted(glob.glob(os.path.join(d0, "output", "*cell"))) if d0 else []
    want = set(np.linspace(0, len(pfs) - 1, 4).round().astype(int)); snaps = {}; H = []; frames = []
    for n, pf in enumerate(pfs):
        t, x, q = load(pf); dA = (x[1] - x[0])**2; X, Y = np.meshgrid(x, x, indexing="ij"); r = np.hypot(X, Y)
        ag, p, T = q("eta"), q("pressure"), q("pc_T")
        cav = ag > 0.5; c = r < 3 * (x[1] - x[0])
        sat = np.nan
        if cav.sum() > 20:
            core = cav & (r < 0.7 * np.sqrt(cav.sum() * dA / np.pi))
            sat = np.abs(p[core] / np.array([psat(Ti) for Ti in T[core]]) - 1).max() if core.any() else np.nan
        asym = np.abs(ag - ag.T).max()
        p_no = np.nan
        if n < len(pf0): _, _, qn = load(pf0[n]); p_no = qn("pressure")[c].mean()
        H.append((t, cav.sum() * dA, np.sqrt(cav.sum() * dA / np.pi), p[c].mean(), T[c].mean(), sat, q("vapor").sum() * dA, asym, p_no, ag[c].mean()))
        if n in want: snaps[n] = (t, x, {k: q(k) for k in ("eta", "pressure", "pc_T", "pc_Yv")})
        if "--no-gif" not in sys.argv:
            fig, ax = plt.subplots(figsize=(6.4, 5.4))
            im = ax.imshow(ag.T, origin="lower", extent=(1e3 * x[0], 1e3 * x[-1], 1e3 * x[0], 1e3 * x[-1]), cmap="Blues_r", vmin=0, vmax=1)
            ax.set_title(f"vapour volume fraction, t = {1e6 * t:6.1f} $\\mu$s"); ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); fig.colorbar(im, ax=ax, shrink=0.85)
            fn = os.path.join(cdir, f"frame_{n:05d}.png"); fig.savefig(fn, dpi=100); plt.close(fig); frames.append(fn)
    H = np.array(H); t = H[:, 0]
    print(f"{d}: {len(pfs)} plot files to t = {1e6 * t[-1]:.1f} us")
    for k in sorted(set(np.linspace(0, len(H) - 1, 6).round().astype(int))):
        print(f"  t = {1e6 * H[k, 0]:6.1f} us: cavity radius {1e3 * H[k, 2]:.3f} mm, centre p {H[k, 3]:9.1f} Pa (no phase change: {H[k, 8]:9.1f}), centre T {H[k, 4]:.2f} K,"
              f" centre vapour fraction {H[k, 9]:.3f}, |p/p_sat(T) - 1| in the cavity {H[k, 5]:.1e}")
    print(f"  p_sat(354.73 K) = {psat(354.728):.1f} Pa;  max asymmetry of the vapour fraction under x <-> y: {H[:, 7].max():.1e}")
    ns = sorted(snaps); fig, ax = plt.subplots(4, len(ns), figsize=(4.2 * len(ns), 15), squeeze=False)
    spec = (("eta", "vapour volume fraction", "Blues_r", 0, 1, 1.0), ("pressure", "pressure [bar]", "viridis", 0, 1.2, 1e-5),
            ("pc_T", "temperature [K]", "inferno", 352, 356, 1.0), ("pc_Yv", "vapour mass fraction of the gas", "magma", 0, 1, 1.0))
    for j, n in enumerate(ns):
        ts, x, f = snaps[n]
        for i, (k, lab, cm, lo, hi, sc) in enumerate(spec):
            im = ax[i, j].imshow(sc * f[k].T, origin="lower", extent=(1e3 * x[0], 1e3 * x[-1], 1e3 * x[0], 1e3 * x[-1]), cmap=cm, vmin=lo, vmax=hi)
            ax[i, j].set_title(f"{lab}, t = {1e6 * ts:.0f} $\\mu$s", fontsize=9); fig.colorbar(im, ax=ax[i, j], shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cavitation_bubble_panels.png"), dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(1e6 * t, 1e3 * H[:, 2], "k-"); ax[0].set_ylabel("cavity radius [mm]")
    ax[1].semilogy(1e6 * t, H[:, 3], "k-", label="with phase change"); ax[1].axhline(psat(354.728), color="C3", ls="--", label=r"$p_{sat}(T_0)$")
    if d0: ax[1].semilogy(1e6 * t, np.abs(H[:, 8]), "C0:", label="|p|, no mass transfer")
    ax[1].set_ylabel("centre pressure [Pa]"); ax[1].legend(fontsize=8)
    ax[2].plot(1e6 * t, H[:, 6], "k-"); ax[2].set_ylabel("vapour mass per unit depth [kg/m]")
    for a in ax: a.set_xlabel(r"t [$\mu$s]"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cavitation_bubble_history.png"), dpi=200); plt.close(fig)
    if frames:
        from PIL import Image
        ims = [Image.open(fn).convert("RGB") for fn in frames]
        pal = ims[-1].quantize(colors=256, dither=Image.Dither.NONE)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(os.path.join(cdir, "cavitation_bubble.gif"), save_all=True, append_images=ims[1:], duration=100, loop=0)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
