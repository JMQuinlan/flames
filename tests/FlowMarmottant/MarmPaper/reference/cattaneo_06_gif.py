#!/usr/bin/env python3
"""
cattaneo_06_gif -- animated GIFs for the MarmPaper 06 analysis (cattaneo_06_common.main --gif).

Outputs (<figdir>/<stem>...):
  _bubble.gif     cross-section through the bubble centre, coloured by p - p_inf (blue-red diverging
                  colormap, symmetric-log scale), eta = 0.5 interface drawn as a black contour.
  _bubble_R.gif   the same panel (left) next to R(t) (right): experiment, Cattaneo et al. model and the
                  simulation R_vol, with a marker and a vertical line at the frame time.
  _bubble3d.gif   (--gif-3d, octant / full only) eta = 0.5 isosurface.
  _bubble3d_cut.gif (--gif-3d) the same isosurface cut open along the cross-section plane through the
                  bubble centre: the half towards the viewer is removed and the cut face shows the liquid
                  coloured by p - p_inf (gas left open, eta = 0.5 contour in black), so the inside of the
                  cavity (jet, re-entrant shapes) is visible.  Cut face +-3 R0 (--gif-cut-half).
  _bubble3d_cutshell.gif (--gif-3d) the same cut-open shell with no field: only the eta = 0.5 contour on the cut.
  Frames are kept as PNG and EPS in <figdir>/<stem>_gif_frames/{bubble,bubble_R,bubble3d,bubble3d_cut,bubble3d_cutshell}/
  (both go into the results tar, marmpaper_bundle).  --sim-shift moves the simulation time (frame times, R_vol curve) only.

Cross-section per geometry (cattaneo_06_common.geometry):
  slab    RADIAL slab: the axis-row profile p(r), eta(r) revolved into a disk.
  octant  the x-z (default) symmetry plane, first cell layer, mirrored to the full disk.
  full    no symmetry assumed: the bubble centre is the gas-volume centroid of each frame
          (weight (1 - eta) dV over cells with eta < 0.95, leaf cells only), and the actual 2-D slice
          through it is shown (centred on it); cells outside the domain are drawn grey (walls).
Field frames are the plotfiles (every --gif-every); the R(t) curve and marker come from thermo.dat
(gas volume every ~1 ns), R = R0 V/V0 (slab) or R0 (V/V0)^(1/3) (octant/full), as thermo_radius().
For a restarted run V0 is taken from the first row of the thermo.dat of the run it restarted from
(followed through <output>/metadata 'restart = ...'), or from --gif-thermo-v0.  Without a usable
thermo.dat the plotfile R_vol is used.
Time axis: --gif-time paper (default; t_fig = t_sim - exp_shift = t_sim + 1.57 us, Cattaneo Fig. 2a)
or sim (t = 0 at the start of the burst, as the static figure).
GIF assembly: ffmpeg palettegen + paletteuse (no banding); Pillow fallback.
"""
import glob
import os
import re
import shutil
import subprocess

import numpy as np
import matplotlib
from matplotlib.colors import SymLogNorm, to_rgb

import cattaneo_06_common as cc
from sch20_analysis import _yt, _pool_map, _apply_rc, plt

FIELD_P = ("boxlib", "pressure")
FIELD_E = ("boxlib", "eta")
_PLANES = {"xz": (0, 2, 1), "xy": (0, 1, 2), "yz": (1, 2, 0)}     # (horizontal, vertical, normal) axes
_AX = "xyz"

# GIF look on top of cattaneo_06_common.FIG (fonts, colours, legend size).
GIF = dict(cmap="RdBu_r", bad="0.72", iface_color="k", iface_lw=1.3,
           size_bubble=(4.3, 3.9), size_pair=(9.8, 4.0), size_3d=(4.0, 4.0), size_3dcut=(4.9, 4.2), dpi=150)


# =========================================================================================== #
#  Thermo radius with the right V0 for restarted runs
# =========================================================================================== #
def _read_thermo(path):
    d = np.loadtxt(path, skiprows=1, ndmin=2)
    if not len(d):
        return None
    # restarts / repeated rows: keep the last sample of each time, in time order
    t = d[:, 0]
    keep = np.ones(len(t), bool)
    keep[:-1] = t[:-1] < t[1:]
    d = d[keep]
    return d[np.argsort(d[:, 0], kind="stable")]


def restart_v0(plotdir, depth=10):
    """V0 (gas volume at t = 0) of a restarted run: follow <dir>/metadata 'restart = <dir'>/NNNcell'
    to the run that started at t = 0.  Returns (V0, thermo path) or None."""
    d = plotdir
    for _ in range(depth):
        f = os.path.join(d, "metadata")
        src = None
        if os.path.isfile(f):
            with open(f) as fh:
                for line in fh:
                    m = re.match(r"^\s*restart\s*=\s*(\S+)", line)
                    if m:
                        src = m.group(1)
        if not src:
            return None
        d = os.path.dirname(os.path.normpath(src))
        th = os.path.join(d, "thermo.dat")
        if os.path.isfile(th):
            T = _read_thermo(th)
            if T is not None and abs(T[0, 0]) < 1e-12:
                return float(T[0, 1]), th
    return None


def thermo_series(plotdir, R0, geom, v0=None):
    """(t, R, note) from thermo.dat; v0: None = auto (t = 0 row of this run, else the restart chain),
    a float (gas volume) or a path to a reference thermo.dat (first row)."""
    f = os.path.join(plotdir, "thermo.dat")
    if not os.path.isfile(f):
        return None
    T = _read_thermo(f)
    if T is None or len(T) < 2:
        return None
    if v0 is None:
        if abs(T[0, 0]) < 1e-12:
            V0, note = T[0, 1], "V0 from this run (t = 0)"
        else:
            r = restart_v0(plotdir)
            if r is None:
                print("  [gif] thermo.dat starts at t = %.3f us and no restart chain found; "
                      "use --gif-thermo-v0 (falling back to plotfile R_vol)" % (T[0, 0] * 1e6))
                return None
            V0, note = r[0], "V0 from %s (restart chain)" % r[1]
    elif os.path.isfile(str(v0)):
        V0, note = _read_thermo(str(v0))[0, 1], "V0 from %s" % v0
    else:
        V0, note = float(v0), "V0 = %g (command line)" % float(v0)
    q = T[:, 1] / V0
    return T[:, 0], R0 * (q if geom == "slab" else np.cbrt(q)), note


# =========================================================================================== #
#  Per-frame field extraction (runs in the worker pool)
# =========================================================================================== #
def plotfile_time(pf):
    """Simulation time of an AMReX plotfile from its Header (no yt load)."""
    with open(os.path.join(pf, "Header")) as fh:
        fh.readline()
        nv = int(fh.readline())
        for _ in range(nv + 1):            # variable names, then the dimension
            fh.readline()
        return float(fh.readline())


def _finest_dx(ds):
    return np.array([float(ds.domain_width[d]) / ds.domain_dimensions[d] / 2 ** ds.index.max_level
                     for d in range(3)])


def gas_centroid(ds):
    """Gas-volume centroid from eta over the leaf cells: weight (1 - eta) dV, cells with eta < 0.95
    only (trace gas in the liquid would bias it).  Domain centre if there is no gas."""
    ad = ds.all_data()
    eta = np.clip(ad[FIELD_E].d, 0.0, 1.0)
    w = np.where(eta < 0.95, 1.0 - eta, 0.0) * ad["index", "cell_volume"].d
    if w.sum() <= 0:
        return np.array([float(v) for v in ds.domain_center])
    return np.array([np.sum(w * ad["index", a].d) / w.sum() for a in _AX])


class _Padded(dict):
    """field -> array, filled by _cgrid."""


def _cgrid(ds, lev, c0, dims, fields):
    """Covering grid of level `lev` from cell index c0 with `dims` cells, for `fields` (dict field -> ndarray).
    A box that ends on the upper domain face makes yt fail by round-off ("read outside the boundaries of a
    non-periodic domain"); there the last layer is not read but copied from its neighbour (zero gradient,
    as at a wall)."""
    dle = np.array([float(v) for v in ds.domain_left_edge])
    dx = _finest_dx(ds) * 2 ** (ds.index.max_level - lev)
    nf = np.array(ds.domain_dimensions) * 2 ** lev
    c0 = np.asarray(c0, int); dims = np.asarray(dims, int)
    cut = (c0 + dims >= nf) & (dims > 2)
    dread = dims - cut.astype(int)
    cg = ds.covering_grid(lev, left_edge=dle + c0 * dx, dims=dread)
    out = _Padded()
    for f in fields:
        A = np.asarray(cg[f].d, float)
        out[f] = np.pad(A, [(0, int(c)) for c in cut], mode="edge") if cut.any() else A
    return out


def _plane_cut(ds, geom, plane, half, centre):
    """p, eta on the finest-level plane through `centre` (normal axis plane[2]), a (2m x 2m) canvas
    of cell-centred samples at offsets (i - m + 0.5) dx from the centre; NaN outside the domain.
    Octant: the first cell layer of the symmetry plane, mirrored."""
    iu, iv, inn = _PLANES[plane]
    dx = _finest_dx(ds)
    dle = np.array([float(v) for v in ds.domain_left_edge])
    nf = np.array(ds.domain_dimensions) * 2 ** ds.index.max_level
    lev = ds.index.max_level
    if geom == "octant":
        mu, mv = (int(min(np.ceil(half / dx[a]), nf[a])) for a in (iu, iv))
        lo = dle.copy()
        dims = [2, 2, 2]               # >= 2 along the normal: yt treats a 1-cell extent as a reduced dimension
        dims[iu], dims[iv] = mu, mv
        cg = ds.covering_grid(lev, left_edge=lo, dims=dims)
        P, E = (np.take(np.asarray(cg[f].d, float), 0, axis=inn) for f in (FIELD_P, FIELD_E))   # axes in index order
        if iu > iv:
            P, E = P.T, E.T
        P = np.concatenate([P[::-1], P], 0); P = np.concatenate([P[:, ::-1], P], 1)
        E = np.concatenate([E[::-1], E], 0); E = np.concatenate([E[:, ::-1], E], 1)
        u = (np.arange(-mu, mu) + 0.5) * dx[iu]
        v = (np.arange(-mv, mv) + 0.5) * dx[iv]
        return u, v, P, E
    # full: canvas around the cell containing the centre, clipped to the domain
    ic = np.floor((centre - dle) / dx).astype(int)
    m = [int(np.ceil(half / dx[a])) + 1 for a in range(3)]
    i0 = {a: ic[a] - m[a] for a in (iu, iv)}
    i1 = {a: ic[a] + m[a] + 1 for a in (iu, iv)}
    c0 = {a: max(i0[a], 0) for a in (iu, iv)}
    c1 = {a: min(i1[a], nf[a]) for a in (iu, iv)}
    u = dle[iu] + (np.arange(i0[iu], i1[iu]) + 0.5) * dx[iu] - centre[iu]
    v = dle[iv] + (np.arange(i0[iv], i1[iv]) + 0.5) * dx[iv] - centre[iv]
    P = np.full((len(u), len(v)), np.nan)
    E = np.full((len(u), len(v)), np.nan)
    if c1[iu] > c0[iu] and c1[iv] > c0[iv]:
        lo = dle.copy()
        dims = [2, 2, 2]               # >= 2 along the normal: yt treats a 1-cell extent as a reduced dimension
        kn = min(max(ic[inn], 0), nf[inn] - 1)                    # the cell layer containing the centre
        k0 = min(kn, nf[inn] - 2)
        lo[iu], lo[iv] = dle[iu] + c0[iu] * dx[iu], dle[iv] + c0[iv] * dx[iv]
        lo[inn] = dle[inn] + k0 * dx[inn]
        dims[iu], dims[iv] = c1[iu] - c0[iu], c1[iv] - c0[iv]
        cg = _cgrid(ds, lev, np.rint((lo - dle) / dx).astype(int), dims, (FIELD_P, FIELD_E))
        su = slice(c0[iu] - i0[iu], c1[iu] - i0[iu]); sv = slice(c0[iv] - i0[iv], c1[iv] - i0[iv])
        for A, f in ((P, FIELD_P), (E, FIELD_E)):
            x = np.take(cg[f], kn - k0, axis=inn)   # axes (iu, iv) in index order
            A[su, sv] = x if iu < iv else x.T
    return u, v, P, E


def _surface(ds, geom, half, centre):
    """eta = 0.5 isosurface triangles (ntri, 3, 3) relative to the centre: marching cubes (yt's
    march_cubes_grid; skimage is not needed) on the finest-level eta of a box +-half around the centre,
    cell centres as lattice points.  Octant: the eta box is mirrored before meshing (closed surface).
    Full: the box is clipped to the domain and padded with liquid (eta = 1) outside it.  None if there is no surface."""
    from yt.utilities.lib.marching_cubes import march_cubes_grid
    dx = _finest_dx(ds)
    dle = np.array([float(v) for v in ds.domain_left_edge])
    nf = np.array(ds.domain_dimensions) * 2 ** ds.index.max_level
    lev = ds.index.max_level
    m = np.array([int(np.ceil(half / dx[a])) + 1 for a in range(3)])
    if geom == "octant":
        m = np.minimum(m, nf)
        E = np.asarray(ds.covering_grid(lev, left_edge=dle, dims=m)[FIELD_E].d, float)
        for a in range(3):
            E = np.concatenate([np.flip(E, a), E], a)
        x0 = -(m - 0.5) * dx
    else:
        ic = np.floor((centre - dle) / dx).astype(int)
        i0, i1 = ic - m, ic + m + 1
        c0, c1 = np.maximum(i0, 0), np.minimum(i1, nf)
        if np.any(c1 - c0 < 2):
            return None
        E = _cgrid(ds, lev, c0, c1 - c0, (FIELD_E,))[FIELD_E]
        # outside the domain is wall/far field, not fluid: pad with liquid (eta = 1) so the surface closes on the
        # domain face.  Edge padding copied the wall-row gas outward and drew a gas column through the wall.
        E = np.pad(E, [(c0[a] - i0[a], i1[a] - c1[a]) for a in range(3)], mode="constant", constant_values=1.0)
        x0 = dle + (i0 + 0.5) * dx - centre
    E = np.ascontiguousarray(np.clip(E, 0.0, 1.0))
    if not (E.min() < 0.5 < E.max()):
        return None
    mask = np.ones(tuple(n - 1 for n in E.shape), np.uint8)
    V = np.asarray(march_cubes_grid(0.5, E, mask, np.asarray(x0, float), np.asarray(dx, float)), float)
    return V.reshape(-1, 3, 3).astype(np.float32) if V.size else None


def _extract(args):
    pf, geom, plane, half, want3d = args
    ds = _yt().load(pf)
    t = float(ds.current_time)
    if geom == "slab":
        cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        x = cg["index", "x"].d[:, 0, 0]
        y0 = cg["index", "y"].d[0, 0, 0]
        r = np.sqrt(x ** 2 + 2 * y0 ** 2)
        return dict(t=t, kind="radial", r=r, p=cg[FIELD_P].d[:, 0, 0].copy(), eta=cg[FIELD_E].d[:, 0, 0].copy(),
                    centre=np.zeros(3), tri=None)
    centre = gas_centroid(ds) if geom == "full" else np.zeros(3)
    u, v, P, E = _plane_cut(ds, geom, plane, half, centre)
    tri = _surface(ds, geom, half, centre) if want3d else None
    return dict(t=t, kind="plane", u=u, v=v, P=P, E=E, centre=centre, tri=tri)


# =========================================================================================== #
#  Drawing
# =========================================================================================== #
def _nice_up(x):
    """Smallest 1/2/5 x 10^k >= x."""
    if not np.isfinite(x) or x <= 0:
        return 1.0
    e = np.floor(np.log10(x))
    for f in (1.0, 2.0, 5.0, 10.0):
        if f * 10 ** e >= x * (1 - 1e-9):
            return f * 10 ** e
    return 10 ** (e + 1)


def _field_grid(fr, half, n=401):
    """(u, v, P, E) on the drawing grid; slab profiles revolved onto an n x n grid."""
    if fr["kind"] == "plane":
        return fr["u"], fr["v"], fr["P"], fr["E"]
    a = np.linspace(-half, half, n)
    rr = np.hypot(*np.meshgrid(a, a, indexing="ij"))
    out = rr > fr["r"].max()
    P = np.interp(rr, fr["r"], fr["p"]); E = np.interp(rr, fr["r"], fr["eta"])
    P[out] = np.nan; E[out] = np.nan
    return a, a, P, E


def _pressure_scale(frames, half, p_inf, pmax=None, plin=None):
    """(vmax, linthresh) [kPa]: 99.5th percentile of |p - p_inf| in the view over all frames, rounded up."""
    if pmax is None:
        vals = []
        for fr in frames:
            u, v, P, _ = _field_grid(fr, half, n=161)
            vals.append(np.abs(P[np.isfinite(P)] - p_inf) / 1e3)
        vals = np.concatenate(vals) if vals else np.array([1.0])
        pmax = _nice_up(np.percentile(vals, 99.5))
    if plin is None:
        plin = pmax / 100.0
    return float(pmax), float(plin)


def _cbar_ticks(vmax, lin):
    k = [lin * 10 ** j for j in range(0, 8) if lin * 10 ** j <= vmax * (1 + 1e-9)]
    return sorted([-x for x in k] + [0.0] + k)


def _fmt_kpa(x, _pos=None):
    """Colourbar tick label: plain numbers in 0.01 .. 1e4, otherwise m x 10^k (mathtext)."""
    if x == 0:
        return "0"
    a = abs(x)
    if 0.01 <= a < 1e4:
        s = "%g" % a
    else:
        k = int(np.floor(np.log10(a) + 1e-9)); mant = a / 10 ** k
        s = (r"$10^{%d}$" % k) if abs(mant - 1) < 1e-6 else (r"$%g\times10^{%d}$" % (round(mant, 3), k))
    return ("\u2212" if x < 0 else "") + s


def _draw_field(ax, fr, half, ctx):
    S = ctx["S"]
    u, v, P, E = _field_grid(fr, half)
    um = 1e6
    cmap = matplotlib.colormaps[ctx["cmap"]].copy()
    cmap.set_bad(GIF["bad"])
    D = (P - ctx["p_inf"]) / 1e3
    du, dv = (u[1] - u[0]) * um, (v[1] - v[0]) * um
    im = ax.imshow(np.ma.masked_invalid(D).T, origin="lower", cmap=cmap, norm=ctx["norm"],
                   extent=(u[0] * um - du / 2, u[-1] * um + du / 2, v[0] * um - dv / 2, v[-1] * um + dv / 2),
                   interpolation="bilinear" if fr["kind"] == "radial" else "nearest", rasterized=True)
    Ef = np.where(np.isfinite(E), E, 1.0)
    if np.nanmin(E) < 0.5 < np.nanmax(E):
        ax.contour(u * um, v * um, Ef.T, levels=[0.5], colors=GIF["iface_color"], linewidths=GIF["iface_lw"])
    h = half * um
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_aspect("equal")
    ax.set_facecolor(GIF["bad"])
    iu, iv, _ = _PLANES[ctx["plane"]]
    if ctx["geom"] == "slab":
        lu, lv = r"$x$", r"$z$"
    elif ctx["geom"] == "full":
        lu, lv = (r"$%s - %s_\mathrm{c}$" % (_AX[i], _AX[i]) for i in (iu, iv))
    else:
        lu, lv = (r"$%s$" % _AX[i] for i in (iu, iv))
    ax.set_xlabel(lu + "  [µm]"); ax.set_ylabel(lv + "  [µm]")
    ax.set_title(r"$t$ = %.2f µs" % (fr["t"] * um + ctx["toff_sim"]), fontsize=S["label_size"], pad=4)
    return im


def _colorbar(fig, cax, im, ctx):
    cb = fig.colorbar(im, cax=cax, ticks=_cbar_ticks(ctx["vmax"], ctx["lin"]))
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_fmt_kpa))
    cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.ax.tick_params(labelsize=ctx["S"]["tick_size"] - 0.5, direction="out", right=True, left=False)
    cb.set_label(r"$p - p_\infty$  [kPa]", fontsize=ctx["S"]["label_size"])
    cb.outline.set_linewidth(ctx["S"]["axes_width"])
    return cb


def _draw_R(ax, ctx, t_now):
    S = ctx["S"]; um = 1e6; off = ctx["toff"]; offs = ctx["toff_sim"]     # experiment/model, simulation
    w0, w1 = ctx["window"]
    inw = lambda t: (t * um + off >= w0 - 0.05) & (t * um + off <= w1 + 0.05)
    te, Re = ctx["exp"]
    m = inw(te)
    ax.plot(te[m] * um + off, Re[m] * um, "o", color=S["c_exp"], mfc=S["c_exp"] if ctx["hi"] else "white", mew=0.9,
            ms=S["ms_exp"] + 0.6, label="Experiment", zorder=5)
    tc, Rc = ctx["catt"]
    m = inw(tc)
    ax.plot(tc[m] * um + off, Rc[m] * um, "-.", color=S["c_catt"], lw=1.2, label="Cattaneo model")
    ts, Rs = ctx["Rsim"]
    m = (ts * um + offs >= w0 - 0.05) & (ts * um + offs <= w1 + 0.05)
    ax.plot(ts[m] * um + offs, Rs[m] * um, "-", color=S["c_vol"], lw=1.5, zorder=4, label=r"Num. $R_\mathrm{vol}$")
    x = t_now * um + offs
    ax.axvline(x, color="0.35", lw=0.9, ls="--", zorder=2)
    ax.plot([x], [np.interp(t_now, ts, Rs) * um], "o", color=S["c_vol"], ms=6.5, mec="white", mew=1.0, zorder=6)
    ax.set_xlim(w0, w1); ax.set_ylim(*ctx["ylim"])
    ax.grid(True, which="major", color="0.88", lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("Time  [µs]"); ax.set_ylabel(r"$R$  [µm]")


def _render(args):
    kind, fr, ctx, png = args
    S = ctx["S"]
    _apply_rc(S)
    if kind == "bubble":
        W, H = GIF["size_bubble"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.80 / W, 0.55 / H, 2.85 / W, 2.85 / H])
        cax = fig.add_axes([3.75 / W, 0.55 / H, 0.12 / W, 2.85 / H])
        im = _draw_field(ax, fr, ctx["half"], ctx)
        _colorbar(fig, cax, im, ctx)
        fig.text(0.5, 1.0 - 0.12 / H, ctx["title_short"], ha="center", va="top", fontsize=S["label_size"])
    elif kind == "pair":
        W, H = GIF["size_pair"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.75 / W, 0.55 / H, 2.85 / W, 2.85 / H])
        cax = fig.add_axes([3.70 / W, 0.55 / H, 0.12 / W, 2.85 / H])
        axr = fig.add_axes([5.45 / W, 0.55 / H, 4.10 / W, 2.85 / H])
        im = _draw_field(ax, fr, ctx["half"], ctx)
        _colorbar(fig, cax, im, ctx)
        _draw_R(axr, ctx, fr["t"])
        axr.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=S["legend_size"], frameon=False,
                   **cc.LEG)
        fig.text(0.5, 1.0 - 0.12 / H, ctx["title"], ha="center", va="top", fontsize=S["label_size"])
    elif kind == "3dcutshell":
        W, H = GIF["size_3d"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
        _draw_3d_cut(ax, fr, ctx, field=False)
        fig.text(0.5, 1.0 - 0.12 / H, ctx["title_short"], ha="center", va="top", fontsize=S["label_size"])
        fig.text(0.5, 1.0 - 0.42 / H, r"$t$ = %.2f µs" % (fr["t"] * 1e6 + ctx["toff_sim"]), ha="center", va="top",
                 fontsize=S["label_size"])
    elif kind == "3dcut":
        W, H = GIF["size_3dcut"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.0, 0.0, 3.95 / W, 0.86], projection="3d")
        cax = fig.add_axes([4.05 / W, 0.55 / H, 0.12 / W, 2.75 / H])
        im = _draw_3d_cut(ax, fr, ctx)
        _colorbar(fig, cax, im, ctx)
        fig.text(0.5, 1.0 - 0.12 / H, ctx["title_short"], ha="center", va="top", fontsize=S["label_size"])
        fig.text(0.5, 1.0 - 0.42 / H, r"$t$ = %.2f µs" % (fr["t"] * 1e6 + ctx["toff_sim"]), ha="center", va="top",
                 fontsize=S["label_size"])
    else:   # 3d
        W, H = GIF["size_3d"]
        fig = plt.figure(figsize=(W, H))
        ax = fig.add_axes([0.0, 0.0, 1.0, 0.86], projection="3d")
        _draw_3d(ax, fr, ctx)
        fig.text(0.5, 1.0 - 0.12 / H, ctx["title_short"], ha="center", va="top", fontsize=S["label_size"])
        fig.text(0.5, 1.0 - 0.42 / H, r"$t$ = %.2f µs" % (fr["t"] * 1e6 + ctx["toff_sim"]), ha="center", va="top",
                 fontsize=S["label_size"])
    fig.savefig(png, dpi=ctx["dpi"])             # fixed canvas (no tight bbox): identical frame sizes
    fig.savefig(os.path.splitext(png)[0] + ".eps")   # every frame kept as PNG and EPS (bundled in the results tar)
    plt.close(fig)
    return png


def _draw_3d(ax, fr, ctx):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    h = ctx["lim3d"] * 1e6
    elev, azim = 22.0, 35.0
    tri = fr.get("tri")
    if tri is not None and len(tri):
        T = tri.astype(float) * 1e6
        n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
        # outward normals (winding is not consistent across the octant mirrors): point away from the centre
        n *= np.where(np.einsum("ij,ij->i", n, T.mean(1)) < 0, -1.0, 1.0)[:, None]
        e, az = np.radians(elev), np.radians(azim)
        view = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
        L = view + np.array([-0.35, 0.25, 0.75]); L /= np.linalg.norm(L)          # key light, upper left of the viewer
        lam = np.clip(n @ L, 0.0, 1.0)
        spec = np.clip(n @ ((L + view) / np.linalg.norm(L + view)), 0.0, 1.0) ** 40
        base = np.array(to_rgb(ctx["bubble_color"]))
        fc = np.clip(base[None, :] * (0.30 + 0.70 * lam)[:, None] + 0.35 * spec[:, None], 0, 1)
        ax.add_collection3d(Poly3DCollection(T, facecolors=fc, edgecolors=fc, linewidths=0.15))   # edges close AA seams
    else:
        ax.text2D(0.5, 0.5, "no $\\eta = 0.5$ surface", transform=ax.transAxes, ha="center")
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.view_init(elev, azim)
    ax.set_axis_off()


def _clip_halfspace(T, axis):
    """Triangles (n, 3, 3) clipped to coordinate[axis] <= 0 (Sutherland-Hodgman, one plane): a list of
    3- or 4-vertex polygons, so the cut edge follows the plane exactly."""
    out = []
    for tri in T:
        s = tri[:, axis]
        if np.all(s <= 0.0):
            out.append(tri); continue
        if np.all(s > 0.0):
            continue
        poly = []
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]; sa, sb = a[axis], b[axis]
            if sa <= 0.0:
                poly.append(a)
            if (sa <= 0.0) != (sb <= 0.0):
                w = sa / (sa - sb); q = a + w * (b - a); q[axis] = 0.0; poly.append(q)
        if len(poly) >= 3:
            out.append(np.array(poly))
    return out


def _draw_3d_cut(ax, fr, ctx, field=True):
    """eta = 0.5 isosurface cut open along the cross-section plane through the centre.  The half on the
    viewer's side of the plane is dropped (triangles whose centroid lies there); the cut face is drawn in
    the plane from the same finest-level slice as _bubble.gif: liquid (eta >= 0.5) coloured by p - p_inf,
    gas transparent so the inner wall of the cavity shows, eta = 0.5 contour in black.  Two-sided shading
    (the inner wall is seen from inside).  Returns a ScalarMappable for the colour bar."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    hc = ctx["cut_half"] * 1e6                                   # cut-face half-width (--gif-cut-half, default 3 R0)
    h = max(ctx["lim3d"] * 1e6, hc) if field else ctx["lim3d"] * 1e6
    iu, iv, inn = _PLANES[ctx["plane"]]
    # viewer on the + side of the plane normal, looking slightly down onto the cut face
    elev = 20.0
    azim = {0: 0.0, 1: 60.0, 2: 0.0}[inn]
    if inn == 2:
        elev = 65.0
    e, az = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(az), np.cos(e) * np.sin(az), np.sin(e)])
    tri = fr.get("tri")
    if tri is not None and len(tri):
        T = tri.astype(float) * 1e6
        T = _clip_halfspace(T, inn)                               # the half behind the cut plane, clipped exactly
        if len(T):
            n = np.array([np.cross(q[1] - q[0], q[2] - q[0]) for q in T])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
            n *= np.where(n @ view < 0, -1.0, 1.0)[:, None]       # two-sided: face the viewer
            L = view + np.array([-0.35, 0.25, 0.75]); L /= np.linalg.norm(L)
            lam = np.clip(n @ L, 0.0, 1.0)
            spec = np.clip(n @ ((L + view) / np.linalg.norm(L + view)), 0.0, 1.0) ** 40
            base = np.array(to_rgb(ctx["bubble_color"]))
            fc = np.clip(base[None, :] * (0.30 + 0.70 * lam)[:, None] + 0.35 * spec[:, None], 0, 1)
            ax.add_collection3d(Poly3DCollection(T, facecolors=fc, edgecolors=fc, linewidths=0.15, zorder=1))
    # cut face: the slice itself (cell centres, relative to the centre), clipped to the 3-D box
    u, v, P, E = fr["u"] * 1e6, fr["v"] * 1e6, fr["P"], fr["E"]
    ku = np.abs(u) <= (hc if field else h); kv = np.abs(v) <= (hc if field else h)
    u, v, P, E = u[ku], v[kv], P[np.ix_(ku, kv)], E[np.ix_(ku, kv)]
    cmap = matplotlib.colormaps[ctx["cmap"]]
    sm = matplotlib.cm.ScalarMappable(norm=ctx["norm"], cmap=cmap)
    if not field and len(u) > 1 and len(v) > 1 and np.nanmin(E) < 0.5 < np.nanmax(E):
        # shell only: just the eta = 0.5 contour along the cut
        cs = matplotlib.pyplot.figure().add_subplot().contour(u, v, np.where(np.isfinite(E), E, 1.0).T, levels=[0.5])
        for seg in cs.allsegs[0]:
            Y = [None, None, None]; Y[iu], Y[iv], Y[inn] = seg[:, 0], seg[:, 1], np.full(len(seg), 0.01 * h)
            ax.plot(Y[0], Y[1], Y[2], color=GIF["iface_color"], lw=GIF["iface_lw"], zorder=3)
        matplotlib.pyplot.close(cs.axes.figure)
    elif len(u) > 1 and len(v) > 1:
        U, V = np.meshgrid(u, v, indexing="ij")
        X = [None, None, None]; X[iu], X[iv], X[inn] = U, V, np.zeros_like(U)
        rgba = cmap(ctx["norm"](np.nan_to_num((P - ctx["p_inf"]) / 1e3)))
        # facecolors per quad: average of its four corners; a quad is liquid (opaque) where its mean eta >= 0.5,
        # gas and cells outside the domain stay open
        fcq = 0.25 * (rgba[:-1, :-1] + rgba[1:, :-1] + rgba[:-1, 1:] + rgba[1:, 1:])
        Eq = 0.25 * (E[:-1, :-1] + E[1:, :-1] + E[:-1, 1:] + E[1:, 1:])
        fcq[..., 3] = np.where(np.isfinite(Eq) & (Eq >= 0.5), 1.0, 0.0)
        ax.plot_surface(X[0], X[1], X[2], facecolors=fcq, rstride=1, cstride=1, linewidth=0, shade=False,
                        antialiased=False, zorder=2)
        # outline of the cut face (near p_inf the face is white on white)
        bu = np.array([u[0], u[-1], u[-1], u[0], u[0]]); bv = np.array([v[0], v[0], v[-1], v[-1], v[0]])
        B = [None, None, None]; B[iu], B[iv], B[inn] = bu, bv, np.zeros(5)
        ax.plot(B[0], B[1], B[2], color="0.55", lw=0.8, zorder=3)
        if np.nanmin(E) < 0.5 < np.nanmax(E):
            cs = matplotlib.pyplot.figure().add_subplot().contour(u, v, np.where(np.isfinite(E), E, 1.0).T, levels=[0.5])
            for seg in cs.allsegs[0]:
                Y = [None, None, None]; Y[iu], Y[iv], Y[inn] = seg[:, 0], seg[:, 1], np.full(len(seg), 0.01 * h)
                ax.plot(Y[0], Y[1], Y[2], color=GIF["iface_color"], lw=GIF["iface_lw"], zorder=3)
            matplotlib.pyplot.close(cs.axes.figure)
    ax.computed_zorder = False         # draw order: back half-shell, then the cut face, then its contour
    ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_zlim(-h, h)
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.view_init(elev, azim)
    ax.set_axis_off()
    return sm


# =========================================================================================== #
#  GIF assembly
# =========================================================================================== #
def assemble_gif(frame_dir, out, fps):
    pngs = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    if not pngs:
        return False
    ff = shutil.which("ffmpeg")
    if ff:
        pal = os.path.join(frame_dir, "_palette.png")
        pat = os.path.join(frame_dir, "frame_%04d.png")
        try:
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                            "-vf", "palettegen=max_colors=256:stats_mode=full", pal], check=True)
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat, "-i", pal,
                            "-lavfi", "paletteuse=dither=sierra2_4a", "-loop", "0", out], check=True)
            os.remove(pal)
            return True
        except (subprocess.CalledProcessError, OSError) as e:
            print("  [gif] ffmpeg failed (%s); using Pillow" % e)
    from PIL import Image
    rgb = [Image.open(p).convert("RGB") for p in pngs]
    pick = rgb[::max(1, len(rgb) // 16)]
    strip = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
    for k, im in enumerate(pick):
        strip.paste(im, (0, k * pick[0].height))
    pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    imgs = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in rgb]
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(round(1000.0 / fps)), loop=0)
    return True


# =========================================================================================== #
#  Driver
# =========================================================================================== #
def make_gifs(a, C, case_tag, plotdir, figdir, geom, sim=None):
    """a: the argparse namespace of cattaneo_06_common.main (gif_* options, exp_shift, nproc, stem,
    rvol_norm).  sim: sim_radii() dict if already computed (fallback R(t) without thermo.dat)."""
    S = dict(cc.FIG)
    _apply_rc(S)
    nproc = max(1, a.nproc)
    toff = -a.exp_shift if a.gif_time == "paper" else 0.0          # display t [us] = t_sim [us] + toff
    toff_sim = toff + getattr(a, "sim_shift", 0.0)                  # simulation curves / frame times (06Exp: -0.25)
    fs = cc._plotfiles(plotdir)
    if getattr(a, "gif_tmax", None) is not None:                    # last frame on the display axis
        fs = [pf for pf in fs if plotfile_time(pf) * 1e6 + toff_sim <= a.gif_tmax + 1e-9]
    fs = fs[::max(1, a.gif_every)]
    if not fs:
        print("  [gif] no plotfiles"); return
    want3d = bool(a.gif_3d) and geom != "slab"
    if a.gif_3d and geom == "slab":
        print("  [gif] --gif-3d: RADIAL slab has no 3-D bubble, skipped")
    um = 1e6
    print("  [gif] %s geometry, %d frames (every %d), time axis: %s%s"
          % (geom, len(fs), a.gif_every, a.gif_time, " (t_fig = t_sim %+g us)" % toff if toff else ""))

    # ---- R(t): thermo.dat, else plotfile R_vol ---------------------------------------------- #
    th = thermo_series(plotdir, C.R0, geom, a.gif_thermo_v0)
    if th is not None:
        Rsim = (th[0], th[1]); print("  [gif] R(t) from thermo.dat, %s" % th[2])
    else:
        if sim is None:
            sim = cc.sim_radii(plotdir, a.input, geom=geom, nproc=nproc, figdir=figdir, stem=a.stem)
        Rv = sim["R_vol"] * (C.R0 / sim["R_vol"][0] if a.rvol_norm else 1.0)
        Rsim = (sim["t"], Rv); print("  [gif] R(t) from plotfile R_vol")

    # ---- frames -------------------------------------------------------------------------------- #
    Rmax = float(np.nanmax(Rsim[1]))
    half = (a.gif_half * 1e-6) if a.gif_half else max(1.6 * C.R0, 1.3 * Rmax)
    cut_half = a.gif_cut_half * C.R0
    # the 3-D cut face needs the slice out to cut_half; the 2-D views are cropped to `half` again (_field_grid)
    frames = _pool_map(_extract, [(pf, geom, a.gif_plane, max(half, cut_half) if want3d else half, want3d)
                                  for pf in fs], nproc)
    frames.sort(key=lambda f: f["t"])
    tf = np.array([f["t"] for f in frames])
    if geom == "full":
        c = np.array([f["centre"] for f in frames]) * um
        print("  [gif] bubble centroid [um]: first (%.3f, %.3f, %.3f), last (%.3f, %.3f, %.3f)" % (*c[0], *c[-1]))

    p_inf = C.liq.p_inf
    vmax, lin = _pressure_scale(frames, half, p_inf, a.gif_pmax, a.gif_plin)
    print("  [gif] view half-width %.2f um; colour scale p - p_inf in +-%g kPa, linear below %g kPa"
          % (half * um, vmax, lin))

    # ---- right panel: model, experiment, window --------------------------------------------- #
    t_end = max(tf.max(), Rsim[0].max()) + 0.3e-6
    catt = cc.solve_cattaneo(C.pa, cc.Envelope(), t_end, P=dict(R0=C.R0))
    exp = cc.load_experiment(case_tag, a.exp_shift)
    if a.gif_window:
        window = tuple(a.gif_window)
    else:
        w0, w1 = tf.min() * um + toff_sim, tf.max() * um + toff_sim
        if w1 - w0 < 1.0:
            w1 = w0 + 1.0
        pad = 0.02 * (w1 - w0)
        window = (round(w0 - pad, 2), round(w1 + pad, 2))
    sel = lambda t, o=toff: (t * um + o >= window[0]) & (t * um + o <= window[1])
    ys = [y[sel(t, o)] for (t, y), o in ((Rsim, toff_sim), (catt, toff), (exp, toff)) if np.any(sel(t, o))]
    ys = np.concatenate(ys) * um if ys else np.array([C.R0 * um])
    span = np.nanmax(ys) - np.nanmin(ys)
    ylim = (max(0.0, np.nanmin(ys) - 0.06 * span - 0.05), np.nanmax(ys) + 0.06 * span + 0.05)

    ctx = dict(S=S, geom=geom, plane=a.gif_plane, half=half, p_inf=p_inf, cmap=a.gif_cmap, vmax=vmax, lin=lin,
               norm=SymLogNorm(linthresh=lin, linscale=1.0, vmin=-vmax, vmax=vmax, base=10),
               toff=toff, toff_sim=toff_sim, window=window, ylim=ylim, catt=catt, exp=exp, Rsim=Rsim, hi=case_tag == "hi",
               dpi=a.gif_dpi, bubble_color=S["bubble_color"][True],
               title=cc.paper_title(C), title_short=cc.paper_title(C))

    ext = [float(np.abs(f["tri"]).max()) for f in frames if f.get("tri") is not None]
    ctx["lim3d"] = 1.04 * max(ext) if ext else half                   # fixed 3-D box: size changes stay visible
    ctx["cut_half"] = cut_half

    root = os.path.join(figdir, a.stem + "_gif_frames")
    outs = []
    todo = (("bubble", "_bubble"), ("pair", "_bubble_R")) + ((("3d", "_bubble3d"), ("3dcut", "_bubble3d_cut"), ("3dcutshell", "_bubble3d_cutshell")) if want3d else ())
    if getattr(a, "gif_kinds", None):
        keep = set(a.gif_kinds.split(","))
        todo = tuple(k for k in todo if k[0] in keep)
    for kind, suffix in todo:
        fdir = os.path.join(root, suffix.lstrip("_"))
        os.makedirs(fdir, exist_ok=True)
        for old in glob.glob(os.path.join(fdir, "frame_*.png")) + glob.glob(os.path.join(fdir, "frame_*.eps")):
            os.remove(old)
        _pool_map(_render, [(kind, fr, ctx, os.path.join(fdir, "frame_%04d.png" % k)) for k, fr in enumerate(frames)],
                  nproc)
        out = os.path.join(figdir, a.stem + suffix + ".gif")
        if assemble_gif(fdir, out, a.gif_fps):
            outs.append(out)
            print("  wrote %s (%d frames, %g fps; PNG + EPS frames in %s/)" % (out, len(frames), a.gif_fps, fdir))
    return outs


def add_args(ap):
    g = ap.add_argument_group("animated GIFs (cattaneo_06_gif.py)")
    g.add_argument("--gif", action="store_true", help="also write <stem>_bubble.gif and <stem>_bubble_R.gif")
    g.add_argument("--gif-only", action="store_true", help="GIFs only (skip the static figure and peak table)")
    g.add_argument("--gif-fps", type=float, default=10.0, help="frames per second (default 10)")
    g.add_argument("--gif-every", type=int, default=1, metavar="N", help="use every N-th plotfile (default 1)")
    g.add_argument("--gif-time", choices=("paper", "sim"), default="paper",
                   help="time axis: paper = Cattaneo Fig. 2a (t_sim - exp_shift, default) or sim (burst start = 0)")
    g.add_argument("--gif-window", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="R(t) panel window [us] on the chosen time axis (default: the frame span)")
    g.add_argument("--gif-plane", choices=tuple(_PLANES), default="xz", help="cross-section plane (octant/full)")
    g.add_argument("--gif-half", type=float, default=None,
                   help="view half-width [um] (default max(1.6 R0, 1.3 max R))")
    g.add_argument("--gif-pmax", type=float, default=None, help="colour limit |p - p_inf| [kPa] (default auto)")
    g.add_argument("--gif-plin", type=float, default=None, help="symlog linear range [kPa] (default pmax/100)")
    g.add_argument("--gif-cmap", default=GIF["cmap"], help="diverging colormap (default %s)" % GIF["cmap"])
    g.add_argument("--gif-3d", action="store_true", help="also <stem>_bubble3d.gif (eta = 0.5 isosurface) and <stem>_bubble3d_cut.gif (cut open along the cross-section plane; octant/full)")
    g.add_argument("--gif-cut-half", type=float, default=3.0, metavar="N",
                   help="half-width of the pressure cut face in the cut 3-D GIF, in R0 (default 3)")
    g.add_argument("--gif-tmax", type=float, default=None, metavar="T",
                   help="last frame time [us] on the --gif-time axis (default: every plotfile)")
    g.add_argument("--gif-kinds", default=None, metavar="LIST",
                   help="comma list of GIFs to write from bubble,pair,3d,3dcut,3dcutshell (default: all enabled)")
    g.add_argument("--gif-dpi", type=int, default=GIF["dpi"], help="frame dpi (default %d)" % GIF["dpi"])
    g.add_argument("--gif-thermo-v0", default=None, metavar="V0|THERMO",
                   help="gas volume at t = 0 for the thermo.dat radius (number, or a reference thermo.dat whose "
                        "first row is used); default: this run's t = 0 row, else the restart chain in metadata")
    return g
