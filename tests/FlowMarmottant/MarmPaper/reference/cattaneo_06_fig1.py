#!/usr/bin/env python3
"""
cattaneo_06_fig1 -- the 06Exp bubble against the Cattaneo et al. (2025) high-speed frames (their Fig. 1c, 60 kPa,
for 06Exp LowPa; Fig. 1d, 160 kPa, for 06Exp HighPa).  Full text width (6.5 in, FONT pt), PNG + EPS.

Outputs (<figdir>/<stem>..., one set per interface level, eta = 0.5 and eta = 0.35 (files with _eta035)):
  _fig1          4 x 6 frames: experiment 3.5 .. 6.0 us, the simulation at the same times, experiment
                 9.5 .. 12.0 us, the simulation at those times.  Every frame is 20 um x 20 um at the same scale
                 (10 um scale bars), the substrate at the bottom as in the experiment.  The simulated bubble is the
                 shaded eta isosurface (orange 0.5 / blue 0.35, as the GIFs) seen side-on like the microscope:
                 orthographic, level with the bubble, 45 deg between x and y (cattaneo_06_gif GIF side_elev /
                 side_azim), the substrate drawn as a grey band at its height below the bubble.
  _fig1_overlay  2 x 6: the experimental frames with the outline (silhouette) of the same projected isosurface
                 drawn on top in red.
Registration: each simulated frame is placed so that the centroid of its silhouette sits on the bubble centre
found in that experimental frame (cattaneo2025_fig1cd.npz, ring template), i.e. position is matched and size and
shape are what is compared; the scale is the experiment's (12 px/um), nothing is rescaled.
Times: frame label t on the paper axis; the simulation frame used is the plotfile nearest to
t_sim = t + exp_shift - sim_shift (06Exp: t - 1.57 + 0.25 us), within half the plot interval; a frame past the
end of the run is left empty and says so.
Experimental frames: Cattaneo et al., Nature Physics 21:590 (2025), Fig. 1c,d, CC BY 4.0 -- the paper caption
must credit them ("adapted from Cattaneo et al. (2025), Fig. 1c/d, CC BY 4.0").
"""
import os

import numpy as np
import matplotlib
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb

import cattaneo_06_common as cc
import cattaneo_06_gif as G
from sch20_analysis import _yt, _pool_map, _apply_rc, plt

_HERE = os.path.dirname(os.path.abspath(__file__))
FIG1 = os.path.join(_HERE, "cattaneo2025_fig1cd.npz")
F1 = dict(width=6.5, label_w=0.30, gap=0.035, group_gap=0.12, title_h=0.26, legend_h=0.22, bottom=0.04,
          outline="#D62728", outline_lw=1.1, sil_res=0.02,          # silhouette raster [um]
          bg="white", tdx=0.6, tdy=0.6)                              # time label inset [um]


def load_fig1(case_tag):
    """Experimental frames: dict(img (12, n, n) uint8, t (12,) us, ctr (12, 2) px, px_per_um, size_um)."""
    Z = np.load(FIG1)
    img = Z[case_tag + "_img"]
    ppu = float(Z["px_per_um"])
    return dict(img=img, t=Z[case_tag + "_t"], ctr=Z[case_tag + "_ctr"], rad=Z[case_tag + "_rad"], px_per_um=ppu,
                size_um=img.shape[2] / ppu)


# =========================================================================================== #
#  Simulation side view
# =========================================================================================== #
def _project(T, azim):
    """Triangles (n, 3, 3) [m, about the gas centroid] -> screen (n, 3, 2) [um] of the side view (level camera at
    azimuth azim, orthographic, z flipped so the substrate is at the bottom), depth (n,) towards the viewer and the
    shading of _draw_3d_side."""
    az = np.radians(azim)
    view = np.array([np.cos(az), np.sin(az), 0.0])
    side = np.array([-np.sin(az), np.cos(az), 0.0])
    P = T.astype(float) * 1e6
    P[:, :, 2] *= -1.0
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    n *= np.where(np.einsum("ij,ij->i", n, P.mean(1)) < 0, -1.0, 1.0)[:, None]
    L = view + 0.6 * side + np.array([0.0, 0.0, 0.75]); L /= np.linalg.norm(L)
    lam = np.clip(n @ L, 0.0, 1.0)
    spec = np.clip(n @ ((L + view) / np.linalg.norm(L + view)), 0.0, 1.0) ** 40
    XY = np.stack([P @ side, P[:, :, 2]], axis=-1)
    return XY.astype(np.float32), (P.mean(1) @ view).astype(np.float32), lam.astype(np.float32), spec.astype(np.float32)


def _silhouette(XY, res):
    """Outline of the union of the projected triangles: list of (m, 2) closed polylines [um] and the area centroid."""
    from PIL import Image, ImageDraw
    lo = XY.reshape(-1, 2).min(0) - 4 * res
    hi = XY.reshape(-1, 2).max(0) + 4 * res
    n = np.ceil((hi - lo) / res).astype(int) + 1
    im = Image.new("L", (int(n[0]), int(n[1])), 0)
    d = ImageDraw.Draw(im)
    Q = (XY - lo) / res
    for q in Q:
        d.polygon([tuple(p) for p in q], fill=255, outline=255)
    M = np.asarray(im, float) / 255.0                         # rows = y index
    ys, xs = np.nonzero(M > 0.5)
    c = lo + res * np.array([xs.mean(), ys.mean()])
    x = lo[0] + res * np.arange(M.shape[1]); y = lo[1] + res * np.arange(M.shape[0])
    fig = matplotlib.figure.Figure()
    cs = fig.add_subplot().contour(x, y, M, levels=[0.5])
    segs = [s for s in cs.allsegs[0] if len(s) > 8]
    return segs, c


def _side_frame(args):
    pf, half, azim, levels, res = args
    ds = _yt().load(pf)
    centre = G.gas_centroid(ds)
    out = dict(t=float(ds.current_time), wall=float(ds.domain_right_edge[2]) - centre[2], lv={})
    for lv in levels:
        T = G._surface(ds, "full", half, centre, lv)
        if T is None or not len(T):
            out["lv"][lv] = None; continue
        XY, depth, lam, spec = _project(T, azim)
        segs, c = _silhouette(XY, res)
        out["lv"][lv] = dict(XY=XY, depth=depth, lam=lam, spec=spec, segs=segs, c=c)
    return out


# =========================================================================================== #
#  Drawing
# =========================================================================================== #
def _scale_bar(ax, size, color, S):
    L = 10.0; x1 = size - 1.0; y = 1.2
    ax.plot([x1 - L, x1], [y, y], color=color, lw=2.6, solid_capstyle="butt", zorder=8)
    ax.text(x1 - L / 2, y + 0.55, "10 µm", color=color, ha="center", va="bottom", fontsize=S["label_size"], zorder=8)


def _tile_axes(fig, x, y, w, h, size):
    ax = fig.add_axes([x, y, w, h])
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.25")
    return ax


def _exp_tile(ax, E, k, S, bar):
    size = E["size_um"]
    ax.imshow(E["img"][k], cmap="gray", vmin=0, vmax=255, extent=(0, size, 0, size), origin="upper",
              interpolation="bilinear", zorder=0)
    ax.text(F1["tdx"], size - F1["tdy"], "%.1f µs" % E["t"][k], color="white", ha="left", va="top",
            fontsize=S["label_size"], zorder=8)
    if bar:
        _scale_bar(ax, size, "white", S)


def _shift(E, k, fr, lv):
    """Screen offset placing the simulated silhouette centroid on the experimental bubble centre [um]."""
    ppu, size = E["px_per_um"], E["size_um"]
    cx, cy = E["ctr"][k]
    return np.array([cx / ppu, size - cy / ppu]) - fr["lv"][lv]["c"]


def _num_tile(ax, E, k, fr, lv, color, S, bar):
    size = E["size_um"]
    ax.set_facecolor(F1["bg"])
    ok = fr is not None and fr["lv"].get(lv) is not None
    if ok:
        d = fr["lv"][lv]
        sh = _shift(E, k, fr, lv)
        zw = -fr["wall"] * 1e6 + sh[1]                          # substrate (flipped: below the bubble)
        if zw > 0:
            ax.axhspan(0, zw, color=G.GIF["wall_color"], lw=0, zorder=1)
            ax.axhline(zw, color=G.GIF["wall_edge"], lw=1.0, zorder=2)
        o = np.argsort(d["depth"])                              # painter: far first
        base = np.array(to_rgb(color))
        fc = np.clip(base[None, :] * (0.30 + 0.70 * d["lam"][o])[:, None] + 0.35 * d["spec"][o][:, None], 0, 1)
        pc = PolyCollection(d["XY"][o] + sh, facecolors=fc, edgecolors=fc, linewidths=0.12, zorder=3)
        pc.set_rasterized(True)
        ax.add_collection(pc)
    else:
        ax.set_facecolor("0.93")
        ax.text(size / 2, size / 2, "past the end\nof the run" if fr is None else "no surface",
                ha="center", va="center", fontsize=S["label_size"], color="0.35")
    ax.text(F1["tdx"], size - F1["tdy"], "%.1f µs" % E["t"][k], color="black", ha="left", va="top",
            fontsize=S["label_size"], zorder=8)
    if bar:
        _scale_bar(ax, size, "black", S)


def _layout(nrows):
    """Tile size and figure height for nrows rows of 6 tiles in two groups (rows split in half)."""
    W = F1["width"]
    tw = (W - F1["label_w"] - 0.01 - 5 * F1["gap"]) / 6.0
    H = F1["title_h"] + F1["legend_h"] + nrows * tw + (nrows - 2) * F1["gap"] + F1["group_gap"] + F1["bottom"]
    return W, H, tw


def _row_y(r, nrows, H, tw):
    """Bottom of row r (0 = top) in inches."""
    top = H - F1["title_h"] - F1["legend_h"]
    half = nrows // 2
    y = top - (r + 1) * tw - r * F1["gap"]
    if r >= half:
        y -= F1["group_gap"] - F1["gap"]
    return y


def _save(fig, stem, formats):
    imgs = [im for ax in fig.axes for im in ax.get_images()]
    for f in formats:
        # vector formats: the experimental frames embedded at their native 240 px (no resampling, small file);
        # raster: smooth bilinear at the output dpi.  The rasterized isosurfaces at 400 dpi.
        for im in imgs:
            im.set_interpolation("bilinear" if f == "png" else "none")
        fig.savefig(stem + "." + f, dpi=600 if f == "png" else 400)
    plt.close(fig)
    print("  wrote %s.{%s}" % (stem, ",".join(formats)))


def fig_compare(E, frames, lv, color, title, out_stem, S, formats):
    W, H, tw = _layout(4)
    fig = plt.figure(figsize=(W, H))
    rows = [("Exp.", 0), ("Num.", 0), ("Exp.", 6), ("Num.", 6)]
    x0 = F1["label_w"]
    for r, (who, k0) in enumerate(rows):
        y = _row_y(r, 4, H, tw)
        for j in range(6):
            k = k0 + j
            ax = _tile_axes(fig, (x0 + j * (tw + F1["gap"])) / W, y / H, tw / W, tw / H, E["size_um"])
            if who == "Exp.":
                _exp_tile(ax, E, k, S, bar=(j == 0))
            else:
                _num_tile(ax, E, k, frames[k], lv, color, S, bar=(j == 0))
        fig.text((x0 - 0.06) / W, (y + tw / 2) / H, who, rotation=90, ha="right", va="center", fontsize=S["label_size"])
    # legend line under the title: what the simulated frames show
    from matplotlib.patches import Patch
    h = [Patch(facecolor=color, edgecolor="none", label=r"Num. $\eta$ = %g isosurface (side view)" % lv),
         Patch(facecolor=G.GIF["wall_color"], edgecolor=G.GIF["wall_edge"], lw=0.8, label="Substrate")]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(0.5 + 0.5 * x0 / W, 1 - (F1["title_h"] + F1["legend_h"] / 2) / H),
               ncol=2, frameon=False, fontsize=S["legend_size"], handlelength=1.4, **{k: v for k, v in cc.LEG.items() if k != "handlelength"})
    fig.text(0.5 + 0.5 * x0 / W, 1 - 0.04 / H, title, ha="center", va="top", fontsize=S["title_size"])
    _save(fig, out_stem, formats)


def fig_overlay(E, frames, lv, title, out_stem, S, formats):
    W, H, tw = _layout(2)
    fig = plt.figure(figsize=(W, H))
    x0 = F1["label_w"]
    for r, k0 in enumerate((0, 6)):
        y = _row_y(r, 2, H, tw)
        for j in range(6):
            k = k0 + j
            ax = _tile_axes(fig, (x0 + j * (tw + F1["gap"])) / W, y / H, tw / W, tw / H, E["size_um"])
            _exp_tile(ax, E, k, S, bar=(j == 0))
            fr = frames[k]
            if fr is not None and fr["lv"].get(lv) is not None:
                sh = _shift(E, k, fr, lv)
                for s in fr["lv"][lv]["segs"]:
                    ax.plot(s[:, 0] + sh[0], s[:, 1] + sh[1], "-", color=F1["outline"], lw=F1["outline_lw"], zorder=6)
            else:
                ax.text(E["size_um"] / 2, 1.0, "past the end\nof the run" if fr is None else "no surface",
                        ha="center", va="bottom", fontsize=S["label_size"], color="white", zorder=8)
        fig.text((x0 - 0.06) / W, (y + tw / 2) / H, "Exp. + Num.", rotation=90, ha="right", va="center",
                 fontsize=S["label_size"])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=F1["outline"], lw=F1["outline_lw"],
                label=r"Num. $\eta$ = %g outline (side view, centred on the experimental bubble)" % lv)]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(0.5 + 0.5 * x0 / W, 1 - (F1["title_h"] + F1["legend_h"] / 2) / H),
               ncol=1, frameon=False, fontsize=S["legend_size"], **cc.LEG)
    fig.text(0.5 + 0.5 * x0 / W, 1 - 0.04 / H, title, ha="center", va="top", fontsize=S["title_size"])
    _save(fig, out_stem, formats)


# =========================================================================================== #
#  Driver
# =========================================================================================== #
def fig1_figures(a, C, plotdir, figdir, case_tag, plotfiles=None):
    """a: argparse namespace (exp_shift, sim_shift, time, nproc, stem, formats).  plotfiles: list of 12 plotfiles
    to use in place of the time lookup (tests)."""
    E = load_fig1(case_tag)
    S = dict(cc.FIG)
    _apply_rc(S)
    toff = (-a.exp_shift if getattr(a, "time", "paper") == "paper" else -cc.EXP_SHIFT_US) + a.sim_shift
    if plotfiles is None:
        fs = cc._plotfiles(plotdir)
        tp = np.array([G.plotfile_time(pf) for pf in fs]) * 1e6
        dtp = np.median(np.diff(tp)) if len(tp) > 1 else 0.05
        plotfiles = []
        for t in E["t"]:
            ts = t - toff
            i = int(np.argmin(np.abs(tp - ts))) if len(tp) else -1
            ok = i >= 0 and abs(tp[i] - ts) <= 0.5 * dtp + 1e-6
            plotfiles.append(fs[i] if ok else None)
            print("  [fig1] t = %4.1f us -> t_sim %.3f us: %s" % (t, ts, os.path.basename(fs[i]) + " (t_sim %.3f)" % tp[i]
                                                              if ok else "no plotfile (run ends at %.3f us)" % (tp.max() if len(tp) else 0)))
    levels = tuple(G.GIF["iso_levels"])
    half = 8.0e-6                                             # surface box: 20 um frames, bubble well inside
    todo = sorted({pf for pf in plotfiles if pf is not None})
    got = dict(zip(todo, _pool_map(_side_frame, [(pf, half, G.GIF["side_azim"], levels, F1["sil_res"]) for pf in todo],
                                   max(1, a.nproc)))) if todo else {}
    frames = [got.get(pf) if pf is not None else None for pf in plotfiles]
    formats = a.formats.split(",") if getattr(a, "formats", None) else list(cc.FIG["formats"])
    title = cc.paper_title(C)
    outs = []
    for lv in levels:
        tag = G._iso_tag(lv)
        color = G.GIF["iso_surface"].get(lv, S["bubble_color"][True])
        st = os.path.join(figdir, a.stem + "_fig1" + tag)
        fig_compare(E, frames, lv, color, title, st, S, formats)
        st2 = os.path.join(figdir, a.stem + "_fig1_overlay" + tag)
        fig_overlay(E, frames, lv, title, st2, S, formats)
        outs += [st, st2]
    # size check: equivalent-circle radius of the silhouette vs the experimental ring radius
    for k, fr in enumerate(frames):
        if fr is None or fr["lv"].get(0.5) is None:
            continue
        r = [np.sqrt(np.abs(0.5 * np.sum(s[:-1, 0] * s[1:, 1] - s[1:, 0] * s[:-1, 1])) / np.pi) for s in fr["lv"][0.5]["segs"]]
        print("  [fig1] %4.1f us: experiment ring r %.2f um, Num. silhouette (eta 0.5) equivalent r %.2f um"
              % (E["t"][k], E["rad"][k] / E["px_per_um"], max(r) if r else np.nan))
    return outs
