#!/usr/bin/env python3
"""
Per-angle surface dilatation, curvature and recovered shell viscosity of a 2D coated-bubble run
(tests/FlowMarmottant/input_osc2d_unit; quarter cylinder, symmetry planes on x = 0 and y = 0).

  python3 shell_dilatation_2d.py <run_dir> [<run_dir> ...] [--kappa_s 7.2e-9] [--nang 24]

For every plot file, in NANG angular wedges between the two symmetry planes:
  r(theta)        radius of the eta = 0.5 contour
  kappa(theta)    its curvature, from the polar form  (r^2 + 2 r'^2 - r r'') / (r^2 + r'^2)^(3/2)
  D_stored        what the solver put into the shell stress: |grad eta|-weighted mean over the band of
                  (kappa1 - kappa2) / kappa_s        [kappa1 = sigma(Gamma) + kappa_s div_s u, kappa2 = sigma(Gamma)]
  D_band          the same weighting of  P : grad u  rebuilt here with central differences (check of D_stored)
  D_kin           the dilatation of the eta = 0.5 CONTOUR itself (the reference):
                      kappa V_n + d(u_t)/ds
                  V_n = normal speed of the contour from its positions in the previous and next plot file,
                  u_t = tangential mixture velocity sampled on the contour.
  recovered       D_stored / D_kin   (1 = the shell stress sees the true dilatation of the interface)
For a round cylinder D_kin = R'/R at every angle and kappa R = 1; flattening shows as kappa R < 1 on a face.
Writes <run_dir>/shell_dilatation.npz and prints a time table (wedge-mean and spread).
"""
import os, sys, glob
import numpy as np
import yt
yt.funcs.mylog.setLevel(50)
from scipy.ndimage import map_coordinates
R0 = 2.0e-6


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def load(pf, half=1.6):
    ds = yt.load(pf); lev = ds.index.max_level; dx = float(ds.domain_width[0]) / (int(ds.domain_dimensions[0]) * 2 ** lev)
    n = int(round(half * R0 / dx)); cg = ds.covering_grid(lev, ds.domain_left_edge, [n, n, 1])
    f = lambda s: np.asarray(cg[("boxlib", s)], float)[:, :, 0]
    names = [x[1] for x in ds.field_list]
    d = dict(t=float(ds.current_time), dx=dx, n=n, eta=f("eta"), u=f("velocityx"), v=f("velocityy"))
    d["k1"] = f("kappa1") if "kappa1" in names else None; d["k2"] = f("kappa2") if "kappa2" in names else None
    d["V"] = 4.0 * (1.0 - np.asarray(ds.all_data()[("boxlib", "eta")])).dot(np.asarray(ds.all_data()[("index", "cell_volume")]))
    return d


def cdiff(a, ax, dx, odd):
    """central difference; the low face is a symmetry plane (mirror ghost, odd or even)."""
    o = np.empty_like(a)
    if ax == 0:
        o[1:-1] = (a[2:] - a[:-2]) / (2 * dx); o[0] = (a[1] - (-a[0] if odd else a[0])) / (2 * dx); o[-1] = (a[-1] - a[-2]) / dx
    else:
        o[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2 * dx); o[:, 0] = (a[:, 1] - (-a[:, 0] if odd else a[:, 0])) / (2 * dx); o[:, -1] = (a[:, -1] - a[:, -2]) / dx
    return o


def contour(d, th):
    """radius of eta = 0.5 along each ray (first crossing from the centre)."""
    s = np.linspace(0.0, 0.98 * d["n"] * d["dx"], 4000); r = np.full(len(th), np.nan)
    for i, t in enumerate(th):
        v = map_coordinates(d["eta"], [s * np.cos(t) / d["dx"] - 0.5, s * np.sin(t) / d["dx"] - 0.5], order=1, mode="nearest"); k = np.where(v >= 0.5)[0]
        if len(k) and k[0] > 0: j = k[0]; r[i] = s[j - 1] + (0.5 - v[j - 1]) / (v[j] - v[j - 1]) * (s[j] - s[j - 1])
    return r


KFIT = 8


def fit_even(r, th):
    """least-squares fit r(theta) = sum_k a_k cos(2 k theta), k = 0..KFIT (even about both symmetry planes);
    returns the fitted r, r', r''.  Taking second differences of the raw contour amplifies its interpolation noise
    (curvature x R scattered between -0.2 and 1.9 on a bubble round to 3e-4)."""
    m = np.isfinite(r); k = np.arange(KFIT + 1); A = np.cos(2 * np.outer(th, k)); c = np.linalg.lstsq(A[m], r[m], rcond=None)[0]
    return A @ c, -(np.sin(2 * np.outer(th, k)) * (2 * k)) @ c, -(A * (2 * k) ** 2) @ c


def fit_odd(q, th):
    """fit q(theta) = sum_k b_k sin(2 k theta) (odd about both planes: a tangential velocity); returns q and dq/dtheta."""
    m = np.isfinite(q); k = np.arange(1, KFIT + 1); A = np.sin(2 * np.outer(th, k)); c = np.linalg.lstsq(A[m], q[m], rcond=None)[0]
    return A @ c, (np.cos(2 * np.outer(th, k)) * (2 * k)) @ c


def polar_curv(r, th):
    rf, r1, r2 = fit_even(r, th)
    return (rf * rf + 2 * r1 * r1 - rf * r2) / (rf * rf + r1 * r1) ** 1.5, r1, rf


def analyse(d0, dm, dp, nang, ks):
    dx, n = d0["dx"], d0["n"]; th = (np.arange(nang) + 0.5) * (np.pi / 2) / nang          # wedge centres
    thf = (np.arange(8 * nang) + 0.5) * (np.pi / 2) / (8 * nang)                           # fine rays for the contour
    r = contour(d0, thf); kap, r1, r = polar_curv(r, thf)
    # normal speed of the contour from the neighbouring frames (radial speed times cos of the tilt)
    rm, rp = fit_even(contour(dm, thf), thf)[0], fit_even(contour(dp, thf), thf)[0]; rdot = (rp - rm) / (dp["t"] - dm["t"]); tilt = np.arctan2(r1, r)
    Vn = rdot * np.cos(tilt)
    # tangential mixture velocity on the contour and its derivative along the arc
    xs, ys = r * np.cos(thf), r * np.sin(thf)
    uu = map_coordinates(d0["u"], [xs / dx - 0.5, ys / dx - 0.5], order=1, mode="nearest"); vv = map_coordinates(d0["v"], [xs / dx - 0.5, ys / dx - 0.5], order=1, mode="nearest")
    nx, ny = np.cos(thf - tilt), np.sin(thf - tilt)                                         # outward normal
    ut = -uu * ny + vv * nx; un = uu * nx + vv * ny
    dut = fit_odd(ut, thf)[1] / np.hypot(r, r1)                                              # d(u_t)/ds, u_t odd across a symmetry plane
    Dkin = kap * Vn + dut; Dfield = kap * un + dut
    # band quantities
    eta = d0["eta"]; ex, ey = cdiff(eta, 0, dx, False), cdiff(eta, 1, dx, False); g = np.hypot(ex, ey) + 1e-300
    ux, uy = cdiff(d0["u"], 0, dx, True), cdiff(d0["u"], 1, dx, False); vx, vy = cdiff(d0["v"], 0, dx, False), cdiff(d0["v"], 1, dx, True)
    n1, n2 = ex / g, ey / g; Dband_c = (ux + vy) - (n1 * n1 * ux + n1 * n2 * (uy + vx) + n2 * n2 * vy)
    xc = (np.arange(n) + 0.5) * dx; X, Y = np.meshgrid(xc, xc, indexing="ij"); TH = np.arctan2(Y, X)
    band = (eta > 0.02) & (eta < 0.98); wedge = np.minimum((TH / (np.pi / 2) * nang).astype(int), nang - 1)
    out = dict(th=th)
    def wmean(q):
        o = np.full(nang, np.nan)
        for k in range(nang):
            m = band & (wedge == k)
            if m.any(): o[k] = (g[m] * q[m]).sum() / g[m].sum()
        return o
    out["Dband"] = wmean(Dband_c)
    out["Dstored"] = wmean((d0["k1"] - d0["k2"]) / ks) if d0["k1"] is not None else np.full(nang, np.nan)
    out["width"] = np.array([(band & (wedge == k)).sum() * dx * dx / max(np.nanmean(r[8 * k:8 * k + 8]) * (np.pi / 2) / nang, 1e-300) / dx for k in range(nang)])   # band width in cells
    for key, arr in (("r", r), ("kap", kap), ("Dkin", Dkin), ("Dfield", Dfield), ("Vn", Vn), ("un", un)):
        out[key] = np.nanmean(arr.reshape(nang, 8), axis=1)
    out["Rv"] = np.sqrt(d0["V"] / np.pi); out["t"] = d0["t"]
    return out


def main():
    args = sys.argv[1:]; skip = {args.index(k) + 1 for k in ("--kappa_s", "--nang") if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    ks = float(opt(args, "--kappa_s", "7.2e-9")); nang = int(opt(args, "--nang", "24"))
    for d in dirs:
        pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
        F = [load(pf) for pf in pfs]; F = [f for i, f in enumerate(F) if i == 0 or f["t"] > F[i - 1]["t"] + 1e-15]
        res = [analyse(F[i], F[i - 1], F[i + 1], nang, ks) for i in range(1, len(F) - 1)]
        keys = [k for k in res[0] if k not in ("th",)]
        np.savez(os.path.join(d, "shell_dilatation.npz"), th=res[0]["th"], **{k: np.array([r[k] for r in res]) for k in keys})
        print(f"== {d}: {len(res)} frames, R0/dx = {R0 / F[0]['dx']:.1f}")
        print("    t[ns]  R_vol/R0  roundness rms(r)/<r>  kappa R: min .. max   band cells   R'/R [1/us]   D_kin    D_field   D_band   D_stored   recovered (stored/kin): mean, min .. max over angle")
        for r in res[::max(1, len(res) // 20)]:
            rr = r["r"]; kR = r["kap"] * np.nanmean(rr); rec = r["Dstored"] / r["Dkin"]; big = np.abs(r["Dkin"]) > 0.3 * np.nanmax(np.abs(r["Dkin"]))
            m = lambda a: 1e-6 * np.nanmean(a)
            print(f"   {1e9 * r['t']:6.0f}   {r['Rv'] / R0:.4f}     {np.nanstd(rr) / np.nanmean(rr):.4f}           {np.nanmin(kR):5.2f} .. {np.nanmax(kR):5.2f}     {np.nanmean(r['width']):5.1f}      "
                  f"{m(r['Vn'] / rr):+8.3f}   {m(r['Dkin']):+8.3f} {m(r['Dfield']):+8.3f} {m(r['Dband']):+8.3f} {m(r['Dstored']):+8.3f}     "
                  + (f"{np.nanmean(r['Dstored']) / np.nanmean(r['Dkin']):5.2f}, {np.nanmin(rec[big]):5.2f} .. {np.nanmax(rec[big]):5.2f}" if big.any() and abs(np.nanmean(r['Dkin'])) > 0 else "   -"))


if __name__ == "__main__":
    main()
