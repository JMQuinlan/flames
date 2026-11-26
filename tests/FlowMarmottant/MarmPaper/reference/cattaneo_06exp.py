#!/usr/bin/env python3
"""
cattaneo_06exp -- extra figures for the MarmPaper 06Exp runs (full 3-D bubble under a rigid substrate, the substrate
is the TOP face z = z_hi of the domain).  Hooks for cattaneo_06_common.main (add_args, post), used by
analyze_06Exp_Cattaneo_{HighPa,LowPa}.py.

Defaults set here for 06Exp: paper time axis ending at 12 us everywhere; GIFs = the four 3-D views only
(_bubble3d full bubble, _bubble3d_cut cut open with the pressure on the cut face, _bubble3d_cutshell cut open, shell
only, _bubble3d_side level side view with the substrate at the bottom, as the experimental imagery); the
gas-thermodynamics figure is off (--thermo-fig turns it on).

Pressure map  <stem>_pmap.{png,eps}: p - p_inf [kPa] (blue = below, red = above the ambient pressure, symmetric-log
scale) against time (Cattaneo Fig. 2a axis, 0..12 us) and distance r from the bubble centre (gas-volume centroid of
each plotfile) along two rays, (a) away from the substrate (-z, the side the jet forms on) and (b) parallel to the
substrate (+x); the bubble interface (eta = 0.5 along the ray) is the black line.
<stem>_pmap_compact.{png,eps}: panel (a) only.  <stem>_pmap.npz: the sampled map (t, r, p, eta per ray).
Samples are cell values of the finest-level covering grid along the cell row through the centroid.
"""
import os

import numpy as np
import matplotlib
from matplotlib.colors import SymLogNorm

import cattaneo_06_common as cc
import cattaneo_06_gif as G
from sch20_analysis import _yt, _pool_map, _apply_rc, plt

F_E, F_P = ("boxlib", "eta"), ("boxlib", "pressure")
RAYS = (("mz", 2, -1, r"Away from the substrate ($-z$)"), ("px", 0, +1, r"Parallel to the substrate ($+x$)"))
T_PAPER_END = 12.0          # us, end of the Cattaneo Fig. 2a axis


# =========================================================================================== #
#  Ray sampling (worker)
# =========================================================================================== #
def _ray(ds, centre, axis, sgn, rmax):
    """(r, p, eta) along the cell row through `centre` in direction sgn * e_axis, 0 <= r <= rmax."""
    lev = ds.index.max_level
    dx = G._finest_dx(ds)
    dle = np.array([float(v) for v in ds.domain_left_edge])
    nf = np.array(ds.domain_dimensions) * 2 ** lev
    ic = np.clip(np.floor((centre - dle) / dx).astype(int), 0, nf - 1)
    n = int(np.ceil(rmax / dx[axis])) + 1
    # 3 cells across (yt drops 1-cell dimensions), kept off the upper domain face (yt round-off read error there)
    c0 = ic.copy(); dims = np.array([3, 3, 3])
    for a in range(3):
        if a != axis:
            c0[a] = int(np.clip(ic[a] - 1, 0, nf[a] - 4))
    if sgn > 0:
        c0[axis] = ic[axis]; dims[axis] = min(n, nf[axis] - ic[axis])
    else:
        c0[axis] = max(ic[axis] - n + 1, 0); dims[axis] = ic[axis] - c0[axis] + 1
    if dims[axis] < 2:
        return np.array([0.0]), np.array([np.nan]), np.array([np.nan])
    B = G._cgrid(ds, lev, c0, dims, (F_P, F_E))
    sel = [min(ic[a] - c0[a], 2) if a != axis else slice(None) for a in range(3)]
    p, e = B[F_P][tuple(sel)], B[F_E][tuple(sel)]
    x = dle[axis] + (np.arange(c0[axis], c0[axis] + dims[axis]) + 0.5) * dx[axis]
    sd = sgn * (x - centre[axis])                     # signed distance along the ray
    k = sd > -0.5 * dx[axis]                          # the centre cell and the cells beyond it (no cell behind the centre)
    o = np.argsort(sd[k])
    return np.maximum(sd[k][o], 0.0), p[k][o], np.clip(e[k][o], 0.0, 1.0)


def _frame(args):
    pf, rmax = args
    ds = _yt().load(pf)
    c = G.gas_centroid(ds)
    ad = ds.all_data()
    vg = float(np.sum((1.0 - np.clip(ad[F_E].d, 0.0, 1.0)) * ad["index", "cell_volume"].d))   # leaf cells
    out = dict(t=float(ds.current_time), centre=c, dx=float(G._finest_dx(ds).min()), V_gas=vg)
    for key, axis, sgn, _ in RAYS:
        out[key] = _ray(ds, c, axis, sgn, rmax)
    return out


def _interface(r, e):
    """First gas -> liquid eta = 0.5 crossing going out from the centre (NaN if the centre is liquid)."""
    if not len(e) or not e[0] < 0.5:
        return np.nan
    k = np.where((e[:-1] < 0.5) & (e[1:] >= 0.5))[0]
    if not len(k):
        return np.nan
    i = k[0]
    return r[i] + (0.5 - e[i]) / (e[i + 1] - e[i]) * (r[i + 1] - r[i])


# =========================================================================================== #
#  Figure
# =========================================================================================== #
def _panel(ax, t_disp, rg, Pm, Ri, norm, cmap):
    um = 1e6
    im = ax.pcolormesh(t_disp, rg * um, Pm.T, cmap=cmap, norm=norm, shading="nearest", rasterized=True)
    ax.plot(t_disp, Ri * um, "-", color="k", lw=1.2, label=r"Bubble interface ($\eta$ = 0.5)")
    ax.set_xlim(0.0, T_PAPER_END)
    full = np.where(np.all(np.isfinite(Pm), axis=0))[0]          # radii sampled in every frame
    ax.set_ylim(0.0, (rg[full[-1]] if len(full) else rg[-1]) * um)
    ax.set_ylabel(r"$r$  [µm]")
    return im


def pressure_map(F, C, rmax, toff, out_stem, formats):
    """toff [us]: display time = t_sim + toff for the simulation frames (includes --sim-shift)."""
    S = dict(cc.FIG); _apply_rc(S)
    um = 1e6
    t = np.array([f["t"] for f in F])
    td = t * um + toff
    dr = 0.5 * min(f["dx"] for f in F)                # half the finest cell
    rg = np.arange(0.0, rmax + 0.5 * dr, dr)
    if len(rg) > 20000:
        raise RuntimeError("pressure map: %d radial samples (dr %.3g m, rmax %.3g m); check the cell size" % (len(rg), dr, rmax))
    p_inf = C.liq.p_inf
    M, I, E = {}, {}, {}
    for key, *_ in RAYS:
        M[key] = np.array([np.interp(rg, f[key][0], f[key][1], right=np.nan) for f in F]) - p_inf
        E[key] = np.array([np.interp(rg, f[key][0], f[key][2], right=np.nan) for f in F])
        I[key] = np.array([_interface(*f[key][::2]) for f in F])
    np.savez_compressed(out_stem + ".npz", t=t, r=rg, p_inf=p_inf,
                        **{"p_" + k: M[k] for k in M}, **{"eta_" + k: E[k] for k in E},
                        **{"R_" + k: I[k] for k in I})
    allp = np.abs(np.concatenate([M[k][np.isfinite(M[k])] for k in M])) / 1e3
    vmax = G._nice_up(np.percentile(allp, 99.5)) if allp.size else 1.0
    lin = vmax / 100.0
    norm = SymLogNorm(linthresh=lin, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    cmap = matplotlib.colormaps[G.GIF["cmap"]].copy(); cmap.set_bad(G.GIF["bad"])
    ticks = G._cbar_ticks(vmax, lin)

    def cbar(fig, im, axs):
        cb = fig.colorbar(im, ax=axs, ticks=ticks, pad=0.02, fraction=0.05, aspect=30)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(G._fmt_kpa))
        cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        cb.set_label(r"$p - p_\infty$  [kPa]")
        cb.outline.set_linewidth(S["axes_width"])

    def save(fig, stem):
        for ext in formats:
            fig.savefig(stem + "." + ext, dpi=S["dpi"], bbox_inches="tight", pad_inches=S["pad_inches"])
        plt.close(fig)
        print("  wrote %s.{%s}" % (stem, ",".join(formats)))

    # two rays
    W = cc.WIDTH                                       # half-width paper figure, FIG text size
    fig, axs = plt.subplots(2, 1, figsize=(W, 4.2), sharex=True, constrained_layout=True)
    for ax, (key, _, _, title), lab in zip(axs, RAYS, "ab"):
        im = _panel(ax, td, rg, M[key] / 1e3, I[key], norm, cmap)
        ax.set_title("(%s) %s" % (lab, title), loc="left", fontsize=S["label_size"], pad=3)
    # title, then the legend, then the panels: the title is the legend's title (one artist above the panels)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside upper center", fontsize=S["legend_size"],
               frameon=False,
               title=cc.paper_title(C), title_fontsize=S["title_size"], **cc.LEG)
    axs[-1].set_xlabel("Time  [µs]")
    cbar(fig, im, axs)
    save(fig, out_stem)

    # compact: the jet side only
    key = RAYS[0][0]
    fig, ax = plt.subplots(figsize=(W, 2.3), constrained_layout=True)
    im = _panel(ax, td, rg, M[key] / 1e3, I[key], norm, cmap)
    ax.set_xlabel("Time  [µs]")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", fontsize=S["legend_size"],
               frameon=False,
               title=cc.paper_title(C), title_fontsize=S["title_size"], **cc.LEG)
    cbar(fig, im, ax)
    save(fig, out_stem + "_compact")
    print("  [pmap] colour scale +-%g kPa (linear below %g kPa), r <= %.2f um, %d frames"
          % (vmax, lin, rmax * um, len(F)))


# =========================================================================================== #
#  Hooks for cattaneo_06_common.main
# =========================================================================================== #
def add_args(ap):
    g = ap.add_argument_group("06Exp (cattaneo_06exp.py)")
    # 06Exp defaults: paper time to 12 us, the three 3-D GIFs, no thermo figure
    # sim_shift: the full-sphere 06Exp simulation runs a near-constant ~0.25 us behind the experiment (one
    # transducer; the response delay is not in the model), so its time is moved left by 0.25 us (--sim-shift 0)
    ap.set_defaults(gif=True, gif_3d=True, gif_tmax=T_PAPER_END, gif_kinds="3d,3dcut,3dcutshell,3dside", no_thermo_fig=True,
                    sim_shift=-0.25)
    g.add_argument("--no-gif", dest="gif", action="store_false", help="skip the GIFs")
    g.add_argument("--thermo-fig", dest="no_thermo_fig", action="store_false",
                   help="also write the gas-thermodynamics figure (<stem>_thermo)")
    g.add_argument("--no-pmap", action="store_true", help="skip the pressure map")
    g.add_argument("--no-fig1", action="store_true",
                   help="skip the Cattaneo Fig. 1c/d frame comparison (<stem>_fig1, <stem>_fig1_overlay, + _eta035)")
    g.add_argument("--pmap-rmax", type=float, default=4.0, metavar="N", help="pressure-map radius range in R0 (default 4)")
    return g


def _frames(a, C, plotdir):
    """The centroid-ray pass over the plotfiles up to the axis end (once; cached on `a`)."""
    if getattr(a, "_x06_frames", None) is None:
        toff = (-a.exp_shift if a.time == "paper" else 0.0) + a.sim_shift
        t_end = (T_PAPER_END - toff) if a.time == "paper" else np.inf      # simulation time of the axis end
        fs = [pf for pf in cc._plotfiles(plotdir) if G.plotfile_time(pf) * 1e6 <= t_end + 1e-9]
        F = _pool_map(_frame, [(pf, a.pmap_rmax * C.R0) for pf in fs], max(1, a.nproc)) if fs else []
        F.sort(key=lambda f: f["t"])
        a._x06_frames = F
    return a._x06_frames


def radii(a, C, plotdir, figdir):
    """Simulation radii for the R(t) figure, about the moving bubble (the Sch20 frame pass uses a fixed box at
    the origin, which reaches the substrate face and fails once the bubble attaches): R_vol = (3 V_gas / 4 pi)^(1/3)
    with V_gas = sum (1 - eta) dV over the whole domain, and R(eta = 0.5) along +x from the gas centroid."""
    geom = a.geom or cc.geometry(plotdir)
    if geom != "full":
        return cc.sim_radii(plotdir, a.input, geom=geom, nproc=a.nproc, figdir=figdir, stem=a.stem)
    F = _frames(a, C, plotdir)
    t = np.array([f["t"] for f in F])
    Rv = np.cbrt(3.0 * np.array([f["V_gas"] for f in F]) / (4.0 * np.pi))
    Rx = np.array([_interface(f["px"][0], f["px"][2]) for f in F])
    print("  [06Exp] radii about the gas centroid: %d frames, t = %.3f .. %.3f us" % (len(F), t[0] * 1e6, t[-1] * 1e6))
    return dict(t=t, R_vol=Rv, R_eta=Rx, geom=geom)


def post(a, C, plotdir, figdir, sim, case_tag="hi"):
    geom = sim["geom"] if sim else (a.geom or cc.geometry(plotdir))
    if not a.no_fig1:
        if geom == "full":
            import cattaneo_06_fig1
            cattaneo_06_fig1.fig1_figures(a, C, plotdir, figdir, case_tag)
        else:
            print("  [fig1] %s geometry: the Fig. 1 comparison needs the full bubble, skipped" % geom)
    if a.no_pmap:
        return
    if geom != "full":
        print("  [pmap] %s geometry: the 06Exp pressure map needs the full bubble, skipped" % geom); return
    toff = (-a.exp_shift if a.time == "paper" else 0.0) + a.sim_shift   # simulation frames
    F = _frames(a, C, plotdir)
    if not F:
        print("  [pmap] no plotfiles before the axis end"); return
    rmax = a.pmap_rmax * C.R0
    fmts = a.formats.split(",") if a.formats else list(cc.FIG["formats"])
    pressure_map(F, C, rmax, toff, os.path.join(figdir, a.stem + "_pmap"), fmts)
