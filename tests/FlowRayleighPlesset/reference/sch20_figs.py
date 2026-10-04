#!/usr/bin/env python3
"""
Schmidmayer, Bryngelson & Colonius (JCP 402, 2020) style figures for any Sch20 bubble run.

  fig12_pressure_tR  -- Sch20 Fig. 12: pressure in the (R, t) plane along a radial ray from the
                        bubble centre, log colour scale, with the liquid sound-speed slope 1/c
                        drawn from (R0, 0).  Shows how waves leave the interface and whether
                        the mixture region slows or traps them.
  fig14_shapes       -- Sch20 Fig. 14: nominal bubble shape (eta = 0.5) at chosen times, one
                        column per run, one row per time.  3D runs are drawn as a shaded
                        surface built from the eta = 0.5 crossing along rays (the bubble is
                        star-shaped about its centre); 2D runs as the eta = 0.5 outline.

Both read AMReX plotfiles with yt, need only numpy / matplotlib / yt (no scikit-image), and
assume the bubble is centred at CENTER (default: the origin, i.e. octant / quarter domains with
symmetry planes at 0; give the centre for full-domain runs).  Quarter / octant data are mirrored
to the full bubble.  Every look-and-feel setting is in STYLE below.

Usage from a script:
    import sch20_figs as F
    F.fig12_pressure_tR(plotdir, R0, tau_c, out_stem)
    F.fig14_shapes({"label": plotdir, ...}, R0, tau_c, times=[0.7, "Rmin", 2.0], out_stem=...)
Command line:
    python sch20_figs.py fig12 PLOTDIR R0 TAU_C [OUT_STEM]
    python sch20_figs.py fig14 R0 TAU_C OUT_STEM label=PLOTDIR [label=PLOTDIR ...] [times=0.7,Rmin,2.0]
"""
import glob
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource

# --------------------------------------------------------------------------- #
#  STYLE -- edit here
# --------------------------------------------------------------------------- #
STYLE = dict(
    dpi=170,
    font_size=10,
    # fig 12
    fig12_size=(5.2, 4.4),          # per panel (width, height) in inches
    fig12_cmap="jet",
    fig12_pmin=None,                # colour range [Pa]; None = from the data (1st / 99.9th percentile)
    fig12_pmax=None,
    fig12_rmax=4.0,                 # R/R0 shown
    fig12_tmax=None,                # t/tc shown; None = all frames
    fig12_c_liquid=None,            # liquid sound speed [m/s] for the 1/c line; None = from fig12_liquid_eos
    fig12_liquid_eos=(2.35, 1.0e9, 1000.0),   # (gamma, pi [Pa], rho [kg/m^3]) of the liquid (Sch20 water)
    fig12_line_color="#B6FF3B",
    fig12_axis="x",                 # ray direction: "x", "diag" (1,1[,1])
    # fig 14
    fig14_cell=(2.4, 2.4),          # size of one shape panel (inches)
    fig14_colors=["#2EC4B6", "#E07AD6", "#3A86C8", "#8E86D8", "#F4A261", "#6D6875"],
    fig14_nth=96, fig14_nph=48,     # ray directions per hemisphere for the 3D surface
    fig14_elev=22.0, fig14_azim=35.0,
    fig14_scale_with_R=True,        # True: panels share one scale (shows size); False: each fills its panel
)
CENTER = (0.0, 0.0, 0.0)            # bubble centre [m]


def _plotfiles(d):
    fs = [p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p)]
    return sorted(fs, key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))


def _yt():
    import yt
    yt.funcs.mylog.setLevel(50)
    return yt


def _finest_dx(ds):
    return float(ds.domain_width[0]) / ds.domain_dimensions[0] / 2 ** ds.index.max_level


def _box(ds, fields, half):
    """Finest-level covering grid of [C, C+half]^d (quarter/octant about the centre)."""
    dim = ds.dimensionality; dx = _finest_dx(ds)
    lo = [max(CENTER[d], float(ds.domain_left_edge[d])) for d in range(dim)] + [0.0] * (3 - dim)
    m = int(np.ceil(half / dx)) + 2
    dims = [m] * dim + [1] * (3 - dim)
    cg = ds.covering_grid(level=ds.index.max_level, left_edge=lo, dims=dims)
    out = {f: np.asarray(cg[("boxlib", f)], float) for f in fields}
    if dim == 2:
        out = {f: v[:, :, 0] for f, v in out.items()}
    return out, dx, np.array(lo[:dim]) - np.array(CENTER[:dim])


def _sample(B, pts, dx, off):
    from scipy.ndimage import map_coordinates
    Bp = np.pad(B, 1, mode="symmetric")[tuple(slice(0, -1) for _ in B.shape)]
    idx = [(p - o) / dx - 0.5 + 1.0 for p, o in zip(pts, off)]
    return map_coordinates(Bp, idx, order=1, mode="nearest")


# --------------------------------------------------------------------------- #
#  Fig. 12 -- pressure in (R, t)
# --------------------------------------------------------------------------- #
def fig12_pressure_tR(plotdirs, R0, tau_c, out_stem, labels=None):
    """plotdirs: one directory or a list (one panel each)."""
    yt = _yt()
    if isinstance(plotdirs, str):
        plotdirs = [plotdirs]
    labels = labels or [os.path.basename(os.path.normpath(d)) for d in plotdirs]
    plt.rcParams["font.size"] = STYLE["font_size"]
    n = len(plotdirs)
    fig, axs = plt.subplots(1, n, figsize=(STYLE["fig12_size"][0] * n, STYLE["fig12_size"][1]), squeeze=False)
    axs = axs[0]
    data = []
    for d in plotdirs:
        T, P, c_liq, rr = [], [], None, None
        for pf in _plotfiles(d):
            ds = yt.load(pf)
            t = float(ds.current_time)
            if STYLE["fig12_tmax"] is not None and t / tau_c > STYLE["fig12_tmax"]:
                break
            half = STYLE["fig12_rmax"] * R0
            B, dx, off = _box(ds, ["pressure"] + (["soundspeed"] if ("boxlib", "soundspeed") in ds.field_list else []), half)
            dim = ds.dimensionality
            s = np.arange(0.0, half, 0.5 * dx)
            u = np.ones(dim) / np.sqrt(dim) if STYLE["fig12_axis"] == "diag" else np.eye(dim)[0]
            pts = [s * u[k] for k in range(dim)]
            T.append(t); P.append(_sample(B["pressure"], pts, dx, off)); rr = s
            if c_liq is None:
                c_liq = STYLE["fig12_c_liquid"]
                if c_liq is None and "soundspeed" in B:
                    c_liq = float(np.nanmax(B["soundspeed"]))
                if c_liq is None and STYLE["fig12_liquid_eos"]:
                    g_, pi_, rho_ = STYLE["fig12_liquid_eos"]
                    c_liq = float(np.sqrt(g_ * (np.nanmedian(P[0][-10:]) + pi_) / rho_)) if P else None
        data.append((np.array(T), np.array(P), rr, c_liq))
    allp = np.concatenate([p.ravel() for _, p, _, _ in data])
    allp = allp[np.isfinite(allp) & (allp > 0)]
    vmin = STYLE["fig12_pmin"] or np.percentile(allp, 1.0)
    vmax = STYLE["fig12_pmax"] or np.percentile(allp, 99.9)
    for ax, (T, P, rr, c_liq), lab in zip(axs, data, labels):
        im = ax.pcolormesh(rr / R0, T / tau_c, np.clip(P, vmin, vmax), cmap=STYLE["fig12_cmap"],
                           norm=LogNorm(vmin=vmin, vmax=vmax), shading="auto", rasterized=True)
        if c_liq:
            tt = np.linspace(0, T.max(), 50)
            ax.plot(1.0 + c_liq * tt / R0, tt / tau_c, ":", color=STYLE["fig12_line_color"], lw=2.0)
            ax.text(0.97, 0.03, "dotted: R0 + c t,  c = %.0f m/s" % c_liq, transform=ax.transAxes,
                    ha="right", fontsize=STYLE["font_size"] - 2, color="w")
        ax.set_xlim(0, STYLE["fig12_rmax"]); ax.set_ylim(0, T.max() / tau_c)
        ax.set_xlabel("R / R$_0$"); ax.set_title(lab, fontsize=STYLE["font_size"])
    axs[0].set_ylabel("t / t$_c$")
    fig.colorbar(im, ax=list(axs), label="p [Pa]", pad=0.02)
    fig.savefig(out_stem + ".png", dpi=STYLE["dpi"], bbox_inches="tight"); plt.close(fig)
    print("wrote %s.png" % out_stem)


# --------------------------------------------------------------------------- #
#  Fig. 14 -- bubble shapes
# --------------------------------------------------------------------------- #
def _radius_history(d, R0):
    yt = _yt(); T, R = [], []
    for pf in _plotfiles(d):
        ds = yt.load(pf); dim = ds.dimensionality
        B, dx, off = _box(ds, ["eta"], 1.3 * R0)
        vol = float(np.sum(1.0 - B["eta"])) * dx ** dim * 2 ** dim
        T.append(float(ds.current_time)); R.append((3 * vol / (4 * np.pi)) ** (1 / 3) if dim == 3 else np.sqrt(vol / np.pi))
    return np.array(T), np.array(R), _plotfiles(d)


def _crossing(B, dx, off, dirs, rmax):
    """eta = 0.5 crossing radius along each unit direction (rows of dirs)."""
    s = np.arange(0.0, rmax, 0.1 * dx)
    out = np.full(len(dirs), np.nan)
    for k, u in enumerate(dirs):
        e = _sample(B, [s * u[q] for q in range(len(u))], dx, off)
        idx = np.where((e[:-1] < 0.5) & (e[1:] >= 0.5))[0]
        if len(idx):
            i = idx[0]; out[k] = s[i] + (0.5 - e[i]) / (e[i + 1] - e[i]) * (s[i + 1] - s[i])
    return out


def fig14_shapes(runs, R0, tau_c, out_stem, times=(0.7, "Rmin", 2.0)):
    """runs: dict label -> plotdir.  times: t/tc values and/or the string 'Rmin'."""
    yt = _yt()
    plt.rcParams["font.size"] = STYLE["font_size"]
    labs = list(runs)
    nr, nc = len(times), len(labs)
    fig = plt.figure(figsize=(STYLE["fig14_cell"][0] * nc + 1.0, STYLE["fig14_cell"][1] * nr + 0.4))
    ls = LightSource(azdeg=315, altdeg=45)
    Rscale = 1.0
    for c, lab in enumerate(labs):
        T, Rv, fs = _radius_history(runs[lab], R0)
        col = STYLE["fig14_colors"][c % len(STYLE["fig14_colors"])]
        for r, tt in enumerate(times):
            i = int(np.argmin(Rv)) if tt == "Rmin" else int(np.argmin(np.abs(T / tau_c - float(tt))))
            ds = yt.load(fs[i]); dim = ds.dimensionality
            B, dx, off = _box(ds, ["eta"], 1.3 * R0)
            title = "%s\nt = %.2f t$_c$%s" % (lab, T[i] / tau_c, " (R$_{min}$)" if tt == "Rmin" else "") if r == 0 else \
                    "t = %.2f t$_c$%s" % (T[i] / tau_c, " (R$_{min}$)" if tt == "Rmin" else "")
            if dim == 3:
                ax = fig.add_subplot(nr, nc, r * nc + c + 1, projection="3d")
                th = np.linspace(0, np.pi / 2, STYLE["fig14_nth"] // 4 + 1)       # octant, mirrored below
                ph = np.linspace(0, np.pi / 2, STYLE["fig14_nph"] // 2 + 1)
                TH, PH = np.meshgrid(th, ph, indexing="ij")
                dirs = np.stack([np.sin(PH) * np.cos(TH), np.sin(PH) * np.sin(TH), np.cos(PH)], -1).reshape(-1, 3)
                rad = _crossing(B["eta"], dx, off, dirs, 1.3 * R0).reshape(TH.shape)
                if np.all(np.isnan(rad)):
                    ax.set_axis_off(); ax.set_title(title + "\n(no eta = 0.5 surface)", fontsize=STYLE["font_size"] - 1); continue
                rad = np.where(np.isnan(rad), np.nanmean(rad), rad)
                Rf = np.concatenate([rad, rad[::-1][1:], rad[1:], rad[::-1][1:]], 0)     # mirror in azimuth, closed
                thf = np.arange(Rf.shape[0]) * (th[1] - th[0])
                Rf = np.concatenate([Rf, Rf[:, ::-1][:, 1:]], 1)                         # mirror in z
                phf = np.linspace(0, np.pi, Rf.shape[1])
                TT, PP = np.meshgrid(thf, phf, indexing="ij")
                X, Y, Z = Rf * np.sin(PP) * np.cos(TT), Rf * np.sin(PP) * np.sin(TT), Rf * np.cos(PP)
                rgb = ls.shade(Z / R0, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", ["#222", col, "#fff"]),
                               blend_mode="soft", vert_exag=0.5)
                ax.plot_surface(X / R0, Y / R0, Z / R0, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
                L = 1.05 * (1.0 if STYLE["fig14_scale_with_R"] else np.nanmax(Rf) / R0)
                ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L); ax.set_box_aspect((1, 1, 1))
                ax.view_init(STYLE["fig14_elev"], STYLE["fig14_azim"]); ax.set_axis_off()
            else:
                ax = fig.add_subplot(nr, nc, r * nc + c + 1)
                thq = np.linspace(0, np.pi / 2, 181)
                rad = _crossing(B["eta"], dx, off, np.stack([np.cos(thq), np.sin(thq)], -1), 1.3 * R0)
                rad = np.where(np.isnan(rad), np.nanmean(rad), rad)
                th = np.concatenate([thq, np.pi - thq[::-1], np.pi + thq, 2 * np.pi - thq[::-1]])
                rr = np.concatenate([rad, rad[::-1], rad, rad[::-1]])
                ax.fill(rr * np.cos(th) / R0, rr * np.sin(th) / R0, color=col, lw=0)
                ax.plot(rr * np.cos(th) / R0, rr * np.sin(th) / R0, color="k", lw=0.6)
                L = 1.05 * (1.0 if STYLE["fig14_scale_with_R"] else np.nanmax(rr) / R0)
                ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=STYLE["font_size"] - 1)
    fig.suptitle("Nominal bubble shape ($\\eta$ = 0.5)", fontsize=STYLE["font_size"] + 1)
    fig.savefig(out_stem + ".png", dpi=STYLE["dpi"], bbox_inches="tight"); plt.close(fig)
    print("wrote %s.png" % out_stem)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "fig12":
        fig12_pressure_tR(a[1], float(a[2]), float(a[3]), a[4] if len(a) > 4 else "Sch20_fig12_pressure_tR")
    elif a and a[0] == "fig14":
        R0, tc, stem = float(a[1]), float(a[2]), a[3]
        times = (0.7, "Rmin", 2.0)
        runs = {}
        for x in a[4:]:
            if x.startswith("times="):
                times = [t if t == "Rmin" else float(t) for t in x[6:].split(",")]
            else:
                k, v = x.split("=", 1); runs[k] = v
        fig14_shapes(runs, R0, tc, stem, times)
    else:
        print(__doc__)
