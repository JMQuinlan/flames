#!/usr/bin/env python3
"""
sch20_analysis -- one analysis library for every Sch20 bubble run (uncoated or coated, collapsing,
oscillating or driven; octant or full-bubble; 2D/3D detected from the plotfiles).

    from sch20_analysis import Sch20Analysis
    A = Sch20Analysis(input_deck, plotdir, case="Collapsing", coated=False, nproc=8)
    A.run(["radius", "residual", "pressure", "waves", "shapes", "planes", "modes", "gif"],
          debug=["thermo", "health"])

Layout of this file
  1. STYLE         every look-and-feel setting (fonts, colours, sizes, formats, titles, ...)
  2. models        Keller-Miksis / Rayleigh-Plesset from marmottant_rpe_km (uncoated = no shell),
                   all parameters read from the run's own input deck
  3. data pass     ONE parallel, cached pass: every plotfile is opened once and everything the plots
                   need is extracted (ray radius, gas volume, 1-D line p/u/eta, shape radii, gas core,
                   health).  Shapes / planes / GIF read their (few) frames in their own parallel pool.
  4. Sch20Analysis the class the analyze_Sch20_* scripts call; one method per plot.

Plots (names for run()):
  radius        R(t): eta = 0.5 ray vs (Coated) KM / RPE; coated runs shade the elastic window
  residual      R(t) with the percent error vs KM on a log right axis
  velocity      wall velocity
  probes        R(t) over the liquid pressure at fixed radii (+ KM radiated pressure)
  pressure      pressure in the (r, t) plane, contours, radius line, sound-speed line
  waves         outbound / inbound (p' +- rho c u)/2 maps
  shapes        3D eta = 0.5 surface at chosen times
  planes        eta = 0.5 contours on the (001)/(010)/(100)/(111) planes over time
  modes         spherical-harmonic shape modes (+ spectrum at R_min)
  gif           shape for every frame (GIF + PNG frames folder)
  r_volume      gas-volume radius + shell-averaged eta = 0.5 + ray (Sch20 eq. 29)
  eta_band      eta = 0.1/0.5/0.9 radii along the ray
  ic_pressure   first frames' pressure vs the Sch20 eq. (27) IC and its relaxed prediction
  driven        cycle envelope, spectrum, R vs edge pressure
  gamma         (coated) band Gamma vs (R0/R)^2
  csv           radii and model curves on the frame times
Debug plots (debug=[...]): conservation, thermo, wall_balance, reflection, health
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LightSource, SymLogNorm, to_rgb

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "FlowMarmottant", "reference")))
import marmottant_sch20_common as _M                         # noqa: E402  (deck parser)
from marmottant_rpe_km import solve_km, solve_rpe            # noqa: E402

# =========================================================================================== #
#  1. STYLE -- edit here (or pass overrides to Sch20Analysis(..., style={...}))
# =========================================================================================== #
STYLE = dict(
    # ---- page ---------------------------------------------------------------------------- #
    formats=("png", "eps"), dpi=300, pad_inches=0.02,
    width=6.5, height=4.0, font="times",       # "times" (Times New Roman / Liberation Serif) or "default"
    font_size=10, label_size=11, tick_size=9.5, legend_size=8.5, title_size=11,
    line_width=1.5, axes_width=0.8, grid=False, titles=True,
    time_ref="km_min",                         # t_c: "km_min" (first KM minimum) or "rayleigh" (0.915 R0 sqrt(rho/p_inf))
    # ---- colours ------------------------------------------------------------------------- #
    c_km="tab:blue", c_rp="0.5", c_sim="tab:red", c_vol="tab:purple", c_shell="tab:red", c_ray="tab:gray",
    c_res="0.35", c_elastic="0.88",
    band_fill="tab:red", band_lo="tab:blue", band_mid="tab:orange", band_hi="tab:green",
    bubble_color={False: "#2EC4B6", True: "#E07A3A"},     # 3D shapes / GIF: uncoated, coated
    # ---- radius --------------------------------------------------------------------------- #
    radius_measure="ray",                      # "ray" (eta = 0.5 along +x, the KM comparison) or "volume"
    ray_xmax_R0=1.5,                           # eta = 0.5 search window along the ray
    show_volume_on_radius=False, residual_legend_loc="center left",
    # ---- pressure map / waves --------------------------------------------------------------- #
    p_rmax_R0=4.0, p_cmap="jet", p_shading="gouraud",
    contours=True, contours_per_decade=3, contour_color="0.25", contour_lw=0.4,
    contour_upsample=(4, 2), contour_smooth=(2.0, 8.0),
    radius_line=True, radius_lw=1.4,
    c_line=True, c_origin="tc", c_line_style="--", c_line_lw=1.5, c_label=r"$\propto 1/c_l$",
    tri_at=0.45, tri_size=0.35, tri_min=0.02, tri_ls=":", tri_lw=0.6,
    waves_cmap="jet", waves_linthresh=1e-3, waves_rmax_R0=None,
    # ---- shapes / planes / gif / modes --------------------------------------------------- #
    shapes_times=(0.7, "Rmin", 2.0), shape_cell=(2.1, 1.75), shape_zoom=1.45, shape_top=0.90,
    shape_nth=96, shape_nph=48, shape_elev=22.0, shape_azim=35.0, shape_scale_with_R=True,
    planes_times="auto", planes_n=24, planes_normalize=False, planes_size=(6.0, 5.6),
    planes_cmap="viridis", planes_res=241,
    gif_size=(3.2, 3.0), gif_dpi=130, gif_fps=10, gif_highlight=0.55, gif_shadow="#0b2e2a",
    modes_lmax=8, n_dirs=400, size_spectrum=(4.5, 3.5),
    # ---- other ----------------------------------------------------------------------------- #
    probes_R0=(2.0, 5.0, 10.0), box_R0=3.0, title_gap=0.012,
    size_tall=(6.5, 5.6), size_maps=(6.8, 3.6),
    ic_frames=5, ic_rmax_R0=3.0, band_thresholds=(0.1, 0.5, 0.9),
)

# Titles: {C} -> "Coated " or "", {case} -> "Collapsing" / "Oscillating" / ...   (None = no title)
TITLES = dict(
    radius="{C}{case} Bubble Radius",
    residual="{C}{case} Bubble Radius and Error",
    velocity="{C}{case} Bubble Wall Velocity",
    probes="{C}{case} Bubble: Radius and Liquid Pressure",
    pressure="Acoustic Echo of {C}{case} Bubble",
    waves="Outbound and Inbound Waves, {C}{case} Bubble",
    shapes=None,
    planes="{C}{case} Transient Interface Contours",
    modes="{C}{case} Bubble Shape Modes",
    modes_spectrum="{C}{case} Bubble Shape-Mode Spectrum",
    r_volume="{C}{case} Bubble Radius from Gas Volume",
    eta_band="{C}{case} Bubble Interface Band",
    ic_pressure="Initial Pressure Field",
    driven="{C}{case} Bubble Response",
    gamma="{C}{case} Bubble Surface Concentration",
    conservation="Conservation", thermo="Gas Thermodynamics", wall_balance="Pressure Balance at the Wall",
    reflection="Outer-Boundary Reflection", health="Run Health",
)


def _apply_rc(S):
    if S["font"] == "times":
        fonts = {"font.family": "serif",
                 "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                 "mathtext.fontset": "custom", "mathtext.rm": "serif", "mathtext.it": "serif:italic",
                 "mathtext.bf": "serif:bold", "mathtext.cal": "serif:italic", "mathtext.sf": "serif"}
    else:
        fonts = {"font.family": "sans-serif", "mathtext.fontset": "dejavusans"}
    matplotlib.rcParams.update(fonts)
    matplotlib.rcParams.update({
        "font.size": S["font_size"], "axes.labelsize": S["label_size"], "axes.titlesize": S["title_size"],
        "xtick.labelsize": S["tick_size"], "ytick.labelsize": S["tick_size"], "legend.fontsize": S["legend_size"],
        "legend.frameon": False, "legend.handlelength": 2.2, "lines.linewidth": S["line_width"],
        "lines.markersize": 3.5, "axes.linewidth": S["axes_width"],
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "axes.formatter.use_mathtext": True,
        "savefig.dpi": S["dpi"], "pdf.fonttype": 42, "ps.fonttype": 42})


def _light(color, a):
    """Opaque stand-in for alpha (EPS has no transparency)."""
    return tuple(1.0 - a * (1.0 - np.array(to_rgb(color))))


def _deriv(t, y):
    return np.gradient(np.asarray(y, float), np.asarray(t, float))


def _plotfiles(d):
    fs = [p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p)]
    return sorted(fs, key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))


def _yt():
    import yt
    yt.funcs.mylog.setLevel(50)
    return yt


def _pool_map(fn, jobs, nproc):
    if nproc > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(nproc) as pool:
            return pool.map(fn, jobs, chunksize=1)
    return [fn(j) for j in jobs]


def _first_crossing(s, e, lvl=0.5):
    k = np.where((e[:-1] < lvl) & (e[1:] >= lvl))[0]
    if not len(k):
        return np.nan
    i = k[0]
    return s[i] + (lvl - e[i]) / (e[i + 1] - e[i]) * (s[i + 1] - s[i])


def _sample(B, pts, dx, off):
    """Linear interpolation of a cell-centred array B at physical offsets pts (mirror-padded)."""
    from scipy.ndimage import map_coordinates
    Bp = np.pad(B, 1, mode="symmetric")[tuple(slice(0, -1) for _ in B.shape)]
    idx = [(p - o) / dx - 0.5 + 1.0 for p, o in zip(pts, off)]
    return map_coordinates(Bp, idx, order=1, mode="nearest")


def _geom(ds):
    dim = ds.dimensionality
    dle = np.array([float(v) for v in ds.domain_left_edge])
    dre = np.array([float(v) for v in ds.domain_right_edge])
    sym = np.array([dle[d] > -1e-12 for d in range(3)])      # a lo face on the centre -> mirrored axis
    cen = np.where(sym, 0.0, 0.5 * (dle + dre))
    dx = float(ds.domain_width[0]) / ds.domain_dimensions[0] / 2 ** ds.index.max_level
    return dim, dle, dre, sym, cen, dx


def _eta_around(ds, half):
    """Finest-level eta on a box around the bubble centre (from the centre outward on mirrored axes,
    both sides otherwise), corners snapped to the finest grid.  Returns (E, dx, off, sym)."""
    dim, dle, dre, sym, cen, dx = _geom(ds)
    m = int(np.ceil(half / dx)) + 2
    snap = lambda v, d: dle[d] + np.floor((v - dle[d]) / dx + 1e-9) * dx
    lo = [cen[d] if sym[d] else max(snap(cen[d] - m * dx, d), dle[d]) for d in range(dim)] + [0.0] * (3 - dim)
    dims = [m if sym[d] else 2 * m for d in range(dim)] + [1] * (3 - dim)
    E = np.asarray(ds.covering_grid(level=ds.index.max_level, left_edge=lo, dims=dims)[("boxlib", "eta")], float)
    if dim == 2:
        E = E[:, :, 0]
    return E, dx, np.array(lo[:dim]) - cen[:dim], sym[:dim]


def _radii(E, dx, off, sym, dirs, rmax):
    s = np.arange(0.0, rmax, 0.1 * dx)
    out = np.full(len(dirs), np.nan)
    for k, u in enumerate(dirs):
        pts = [np.abs(s * u[q]) if sym[q] else s * u[q] for q in range(len(u))]
        out[k] = _first_crossing(s, _sample(E, pts, dx, off))
    return out


def _fib_dirs(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * k / n); th = np.pi * (1 + 5 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


# =========================================================================================== #
#  3. DATA PASS -- one worker per plotfile
# =========================================================================================== #
def _frame(args):
    pf, R0, S = args
    try:
        yt = _yt()
        ds = yt.load(pf)
        dim, dle, dre, sym, cen, dx = _geom(ds)
        lev = ds.index.max_level
        has = lambda f: ("boxlib", f) in ds.field_list
        o = {"t": float(ds.current_time), "name": os.path.basename(pf), "mtime": os.path.getmtime(pf),
             "octant": bool(np.all(sym[:dim])), "dim": dim, "dx": dx}
        # ---- eta = 0.5 (and band) radius along the ray: +x from the centre, half a cell off-axis --
        off = 0.5 * dx
        a = [cen[0], cen[1] + off, (cen[2] + off) if dim == 3 else cen[2]]
        b = [dre[0], a[1], a[2]]
        ray = ds.ray(ds.arr(a, "code_length"), ds.arr(b, "code_length"))
        x = np.asarray(ray["index", "x"], float) - cen[0]
        er = np.asarray(ray["boxlib", "eta"], float)
        rad = np.sqrt(x * x + (2 if dim == 3 else 1) * off * off)
        srt = np.argsort(rad); rad, er = rad[srt], er[srt]
        m = rad <= S["ray_xmax_R0"] * R0; rad, er = rad[m], er[m]
        for lvl in S["band_thresholds"]:
            idx = np.where(er >= lvl)[0]
            if not len(idx):
                r = np.nan
            elif idx[0] == 0:
                r = rad[0]
            else:
                i = idx[0]; r = rad[i - 1] + (lvl - er[i - 1]) / (er[i] - er[i - 1]) * (rad[i] - rad[i - 1])
            o["ray_%.2f" % lvl] = float(r)
        o["R_ray"] = o["ray_0.50"]
        # ---- 1-D line along +x to the domain edge (2 cells across the axis) ----------------
        lo = [cen[0], cen[1] if sym[1] else cen[1] - dx, (cen[2] if sym[2] else cen[2] - dx) if dim == 3 else 0.0]
        nl = int(np.floor((dre[0] - cen[0]) / dx))
        cg = ds.covering_grid(level=lev, left_edge=lo, dims=[nl, 2, 2 if dim == 3 else 1])

        def row(fld):
            v = np.asarray(cg[("boxlib", fld)], float)
            v = v[:, 0, :] if sym[1] else v.mean(axis=1)
            return v[:, 0] if (dim == 2 or sym[2]) else v.mean(axis=1)
        o["line_s"] = (np.arange(nl) + 0.5) * dx
        o["line_p"], o["line_u"], o["line_eta"] = row("pressure"), row("velocityx"), row("eta")
        # ---- box scalars --------------------------------------------------------------------
        h = S["box_R0"] * R0
        blo = np.maximum(np.where(sym, cen, cen - h), dle); bhi = np.minimum(cen + h, dre)
        if dim == 2:
            blo[2], bhi[2] = dle[2], dre[2]
        reg = ds.box(ds.arr(blo, "code_length"), ds.arr(bhi, "code_length"))
        eta = np.asarray(reg["boxlib", "eta"], float)
        vol = np.asarray(reg["index", "cell_volume"], float)
        p = np.asarray(reg["boxlib", "pressure"], float)
        rg = np.asarray(reg["boxlib", "rho_eta1"], float) if has("rho_eta1") else None
        r = np.sqrt(sum((np.asarray(reg["index", c], float) - cen[i]) ** 2 for i, c in enumerate("xyz"[:dim])))
        u2 = sum(np.asarray(reg["boxlib", "velocity" + c], float) ** 2 for c in "xyz"[:dim])
        fac = 2 ** int(np.sum(sym[:dim]))
        ag = np.clip(1 - eta, 0, 1)
        Vg = float(np.sum(ag * vol)) * fac
        o["R_vol"] = (3 * Vg / (4 * np.pi)) ** (1 / 3) if dim == 3 else np.sqrt(Vg / np.pi)
        far = np.max(np.abs(np.stack([dle - cen, dre - cen]))[:, :dim], axis=0)
        rmax = min(float(np.sqrt(np.sum(far ** 2))), 10.0 * R0)
        nb = 400
        edges = np.linspace(0.0, rmax, nb + 1)
        ib = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
        w = np.bincount(ib, weights=vol, minlength=nb); ew = np.bincount(ib, weights=eta * vol, minlength=nb)
        g = w > 0
        o["R_shell"] = float(_first_crossing(0.5 * (edges[:-1] + edges[1:])[g], ew[g] / w[g]))
        o["gas_mass"] = float(np.sum(rg * vol)) * fac if rg is not None else np.nan
        core = eta < 0.05
        o["p_gas"] = float(np.sum(p[core] * vol[core]) / np.sum(vol[core])) if core.any() else np.nan
        o["rho_gas"] = float(np.sum(rg[core] / np.maximum(ag[core], 1e-12) * vol[core]) / np.sum(vol[core])) \
            if (rg is not None and core.any()) else np.nan
        wb = eta * (1 - eta) * vol
        o["gamma_band"] = float(np.sum(np.asarray(reg["boxlib", "Gamma"], float) * wb) / np.sum(wb)) \
            if (has("Gamma") and np.sum(wb) > 0) else np.nan
        o["umax_gas"] = float(np.sqrt(u2[eta < 0.01].max())) if (eta < 0.01).any() else np.nan
        o["n_stray"] = int(np.sum((r > 1.6 * o["R_vol"]) & (eta < 0.99)))
        o["p_min"] = float(p.min())
        # ---- shape: eta = 0.5 radius in n_dirs directions + the six axis radii ---------------
        E, dxe, offe, syme = _eta_around(ds, 1.35 * R0)
        if dim == 3:
            dirs = _fib_dirs(S["n_dirs"])
        else:
            th = np.linspace(0, 2 * np.pi, S["n_dirs"], endpoint=False)
            dirs = np.stack([np.cos(th), np.sin(th)], -1)
        o["dirs"] = dirs
        o["rad"] = _radii(E, dxe, offe, syme, dirs, 1.35 * R0)
        ax6 = []
        for d in range(dim):
            for sg in (1, -1):
                u = np.zeros(dim); u[d] = sg
                ax6.append(_radii(E, dxe, offe, syme, [u], 1.35 * R0)[0])
        o["axis_R"] = np.array(ax6)
        return o
    except Exception as exc:
        return {"error": "%s: %s" % (os.path.basename(pf), exc)}


# =========================================================================================== #
#  4. THE CLASS
# =========================================================================================== #
class Sch20Analysis:
    """One Sch20 run: reference models from the input deck, cached frame data, and one method per plot."""

    def __init__(self, input_deck, plotdir, case="Collapsing", coated=False, stem=None, out_dir=None,
                 nproc=4, style=None, titles=None, cache=True, verbose=True):
        self.S = dict(STYLE); self.S.update(style or {})
        self.T = dict(TITLES); self.T.update(titles or {})
        self.case, self.coated, self.nproc, self.verbose = case, bool(coated), int(nproc), verbose
        self.plotdir = plotdir
        self.out_dir = out_dir or os.path.join(_HERE, "Images")
        os.makedirs(self.out_dir, exist_ok=True)
        self.stem = os.path.join(self.out_dir, stem or ("Sch20_%s%s" % (case, "_Marmottant" if coated else "")))
        self._cache_path = (self.stem + "_cache.pkl") if cache else None
        _apply_rc(self.S)
        # ---- deck -> parameters and models ------------------------------------------------
        kv = _M.parse_input(input_deck)
        self.shell, self.liq, self.gas, self.meta = _M.build_case(kv, input_deck)
        self.R0 = self.meta["R0"]
        t_end = self.meta["stop_time"]
        te = np.linspace(0.0, t_end, 8001)
        tk, Rk, Vk = solve_km(self.R0, 0.0, (0.0, t_end), self.shell, self.liq, self.gas, t_eval=te)
        tr, Rr, Vr = solve_rpe(self.R0, 0.0, (0.0, t_end), self.shell, self.liq, self.gas, t_eval=te)
        self.km, self.rp = (tk, Rk, Vk), (tr, Rr, Vr)
        pre = "Coated " if self.coated else ""
        self.label_km, self.label_rp = pre + "Keller–Miksis", pre + "Rayleigh–Plesset"
        i = np.where((Rk[1:-1] < Rk[:-2]) & (Rk[1:-1] <= Rk[2:]))[0]
        t_km = float(tk[(i[0] + 1) if len(i) else int(np.argmin(Rk))])
        self.tc = t_km if self.S["time_ref"] == "km_min" else self.meta["tau_c"]
        self.tlabel = r"$t/t_c$"
        self._frames = None

    # ---- helpers ------------------------------------------------------------------------ #
    def _title(self, key):
        t = self.T.get(key)
        return t.format(C="Coated " if self.coated else "", case=self.case) if (t and self.S["titles"]) else None

    def _save(self, fig, name):
        stem = self.stem + "_" + name
        for fmt in self.S["formats"]:
            fig.savefig(stem + "." + fmt, dpi=self.S["dpi"], bbox_inches="tight", pad_inches=self.S["pad_inches"])
        plt.close(fig)
        if self.verbose:
            print("  wrote", stem + ".png")

    def _suptitle(self, fig, axs, key):
        t = self._title(key)
        if t:
            top = max(a.get_position().y1 for a in np.atleast_1d(axs))
            fig.suptitle(t, y=top + self.S["title_gap"], va="bottom", fontsize=self.S["title_size"])

    def _axtitle(self, ax, key):
        t = self._title(key)
        if t:
            ax.set_title(t)

    @property
    def frames(self):
        if self._frames is None:
            self._frames = self._load_frames()
        return self._frames

    def _load_frames(self):
        key = "v1|%s|%s|%s|%s" % (self.S["ray_xmax_R0"], self.S["box_R0"], self.S["n_dirs"], self.S["band_thresholds"])
        done = {}
        if self._cache_path and os.path.isfile(self._cache_path):
            try:
                with open(self._cache_path, "rb") as fh:
                    c = pickle.load(fh)
                if c.get("key") == key:
                    done = c["frames"]
            except Exception:
                done = {}
        fs = _plotfiles(self.plotdir)
        todo = [pf for pf in fs if os.path.basename(pf) not in done
                or abs(done[os.path.basename(pf)]["mtime"] - os.path.getmtime(pf)) > 1e-6]
        if self.verbose:
            print("  [data] %s: %d plotfiles, %d cached, %d to read (%d proc)"
                  % (self.plotdir, len(fs), len(fs) - len(todo), len(todo), self.nproc))
        for r in _pool_map(_frame, [(pf, self.R0, self.S) for pf in todo], self.nproc):
            if "error" in r:
                print("  [data] skipped", r["error"])
            else:
                done[r["name"]] = r
        if self._cache_path:
            with open(self._cache_path, "wb") as fh:
                pickle.dump({"key": key, "frames": done}, fh)
        return sorted([done[os.path.basename(pf)] for pf in fs if os.path.basename(pf) in done], key=lambda d: d["t"])

    def col(self, k):
        return np.array([f.get(k, np.nan) for f in self.frames], float)

    def radius(self, measure=None):
        """(t, R, label) of the radius compared to the models."""
        m = measure or self.S["radius_measure"]
        if m == "volume":
            return self.col("t"), self.col("R_vol"), r"$R_\mathrm{volume}$"
        return self.col("t"), self.col("R_ray"), r"$\eta = 0.5$"

    def _models_on(self, ax, rp=True, vel=False):
        tk, Rk, Vk = self.km; tr, Rr, Vr = self.rp
        ax.plot(tk / self.tc, (Vk if vel else Rk / self.R0), "-", color=self.S["c_km"], label=self.label_km)
        if rp:
            ax.plot(tr / self.tc, (Vr if vel else Rr / self.R0), "--", color=self.S["c_rp"], label=self.label_rp)

    def _xlim(self, ax, t):
        ax.set_xlim(0, 1.02 * np.nanmax(t) / self.tc)

    def _c_l(self):
        return float(self.liq.c)

    def summary(self):
        t, R, lab = self.radius()
        tk, Rk, _ = self.km
        ok = np.isfinite(R)
        i = np.argmin(np.where(ok, R, np.inf)); j = int(np.argmin(Rk))
        print("  %s %s: R_min/R0 = %.4f @ t/t_c = %.3f   |  %s R_min/R0 = %.4f @ %.3f  (R_min error %+.1f%%, time %+.1f%%)"
              % (self.case, "(coated)" if self.coated else "", R[i] / self.R0, t[i] / self.tc, self.label_km,
                 Rk[j] / self.R0, tk[j] / self.tc, 100 * (R[i] - Rk[j]) / Rk[j], 100 * (t[i] - tk[j]) / tk[j]))

    # ======================================================================================= #
    #  plots
    # ======================================================================================= #
    def plot_radius(self):
        t, R, lab = self.radius()
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        self._models_on(ax)
        ax.plot(t / self.tc, R / self.R0, "-o", color=self.S["c_sim"], ms=2.5, label=lab)
        if self.S["show_volume_on_radius"] and self.S["radius_measure"] != "volume":
            ax.plot(t / self.tc, self.col("R_vol") / self.R0, "-", color=self.S["c_vol"], label=r"$R_\mathrm{volume}$")
        if self.coated and self.shell.R_buck:
            rb, rr = self.shell.R_buck / self.R0, self.shell.R_rupture / self.R0
            y0, y1 = ax.get_ylim()
            ax.axhspan(rb, min(rr, y1) if np.isfinite(rr) else y1, color=self.S["c_elastic"], lw=0, zorder=0,
                       label="Elastic shell")
            ax.set_ylim(y0, y1)
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"$R/R_0$"); self._xlim(ax, t)
        ax.legend(loc="lower left")
        self._axtitle(ax, "radius"); self._save(fig, "R")

    def plot_residual(self):
        t, R, lab = self.radius()
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        self._models_on(ax)
        ax.plot(t / self.tc, R / self.R0, "-o", color=self.S["c_sim"], ms=2.5, label=lab)
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"$R/R_0$"); self._xlim(ax, t)
        ax2 = ax.twinx()
        Rk = np.interp(t, self.km[0], self.km[1])
        err = 100.0 * np.abs(R - Rk) / Rk
        ax2.semilogy(t / self.tc, np.where(err > 0, err, np.nan), ":", color=self.S["c_res"], lw=1.4,
                     label="Error vs " + self.label_km)
        ax2.set_ylabel("Error (%)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        lg = ax2.legend(h1 + h2, l1 + l2, loc=self.S["residual_legend_loc"], frameon=True, framealpha=1.0,
                        facecolor="w", edgecolor="none")
        lg.set_zorder(10)
        self._axtitle(ax, "residual"); self._save(fig, "R_residual")

    def plot_velocity(self):
        t, R, lab = self.radius()
        ok = np.isfinite(R); t, R = t[ok], R[ok]
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        self._models_on(ax, vel=True)
        V = _deriv(t, R)
        ax.plot(t / self.tc, V, "-o", color=self.S["c_sim"], ms=2.5, label=r"$\dot R$, " + lab)
        ax.axhline(0, color="0.6", lw=0.6)
        lim = 1.3 * np.nanmax(np.abs(V)); ax.set_ylim(-lim, lim); self._xlim(ax, t)
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"Wall Velocity, $\dot R$ (m/s)")
        ax.legend(loc="lower left")
        self._axtitle(ax, "velocity"); self._save(fig, "velocity")

    def _km_pressure(self, r, t):
        tk, Rk, Vk = self.km
        Ak = _deriv(tk, Vk); c = self._c_l(); rho = self.liq.rho
        out = np.full(len(t), np.nan)
        for i, ti in enumerate(t):
            Rn = np.interp(ti, tk, Rk)
            tr = ti - max(r - Rn, 0) / c
            if tr < tk[0]:
                continue
            Rr, V, A = np.interp(tr, tk, Rk), np.interp(tr, tk, Vk), np.interp(tr, tk, Ak)
            if r > Rr:
                out[i] = self.liq.p_inf + rho * ((Rr * Rr * A + 2 * Rr * V * V) / r - Rr ** 4 * V * V / (2 * r ** 4))
        return out

    def plot_probes(self):
        fr = self.frames; tf = self.col("t"); t, R, lab = self.radius()
        s = fr[0]["line_s"]
        radii = [r for r in self.S["probes_R0"] if r * self.R0 < 0.95 * s[-1]] + [0.95 * s[-1] / self.R0]
        fig, (a1, a2) = plt.subplots(2, 1, figsize=self.S["size_tall"], sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.06})
        self._models_on(a1, rp=False)
        a1.plot(t / self.tc, R / self.R0, "-o", color=self.S["c_sim"], ms=2.5, label=lab)
        a1.set_ylabel(r"$R/R_0$"); a1.legend(loc="lower left")
        cols = plt.get_cmap("viridis")(np.linspace(0, 0.85, len(radii)))
        for rr, cc in zip(radii, cols):
            p = np.array([np.interp(rr * self.R0, f["line_s"], f["line_p"]) for f in fr])
            lb = (r"$r = %.0f R_0$" % rr) if rr == round(rr) else (r"$r = %.1f R_0$ (edge)" % rr)
            a2.plot(tf / self.tc, p / 1e6, "-", color=cc, label=lb)
            a2.plot(tf / self.tc, self._km_pressure(rr * self.R0, tf) / 1e6, "--", color=cc, lw=0.9)
        a2.plot([], [], "k--", lw=0.9, label=self.label_km + " (radiated)")
        a2.set_yscale("log"); a2.set_xlabel(self.tlabel); a2.set_ylabel(r"Pressure, $p$ (MPa)")
        a2.legend(loc="upper left", ncol=2); self._xlim(a2, tf)
        self._suptitle(fig, (a1, a2), "probes"); self._save(fig, "probes")

    # ---- (r, t) maps ---------------------------------------------------------------------- #
    def _smooth(self, x, y, Z):
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        ft, frr = self.S["contour_upsample"]
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.log10(Z)
        bad = ~np.isfinite(W)
        Wf = np.where(bad, np.nanmin(W[~bad]) if (~bad).any() else 0.0, W)
        xf = np.linspace(x[0], x[-1], max(2, int(len(x) * frr))); yf = np.linspace(y[0], y[-1], max(2, int(len(y) * ft)))
        meth = "cubic" if (len(x) >= 4 and len(y) >= 4) else "linear"
        YY, XX = np.meshgrid(yf, xf, indexing="ij")
        W2 = RegularGridInterpolator((y, x), Wf, method=meth)((YY, XX))
        B2 = RegularGridInterpolator((y, x), bad.astype(float), method="linear")((YY, XX)) > 0.5
        if self.S["contour_smooth"]:
            W2 = gaussian_filter(W2, self.S["contour_smooth"])
        W2[B2] = np.nan
        return xf, yf, 10.0 ** W2

    def _overlays(self, ax, x, y, Z, Rline, lo, hi):
        S = self.S
        if S["contours"]:
            n = S["contours_per_decade"]
            lv = 10.0 ** (np.arange(np.ceil(n * np.log10(lo)), np.floor(n * np.log10(hi)) + 1) / n)
            xs, ys, Zs = self._smooth(x, y, Z)
            ax.contour(xs, ys, Zs, levels=lv, colors=S["contour_color"], linewidths=S["contour_lw"])
        if S["radius_line"] and Rline is not None:
            ax.plot(Rline / self.R0, y, "-", color="k", lw=S["radius_lw"])
        if S["c_line"]:
            c = self._c_l()
            r_a, t_a = (0.0, self.tc) if S["c_origin"] == "tc" else (self.R0, 0.0)
            rl = np.linspace(r_a, x[-1] * self.R0, 400); tl = t_a + (rl - r_a) / c
            vis = tl / self.tc <= y[-1]; rl, tl = rl[vis] / self.R0, tl[vis] / self.tc
            if len(rl) > 2:
                ax.plot(rl, tl, S["c_line_style"], color="k", lw=S["c_line_lw"])
                x0, x1 = rl[0], rl[-1]
                xa = x0 + S["tri_at"] * (x1 - x0); xb = xa + S["tri_size"] * (x1 - x0)
                ya, yb = np.interp(xa, rl, tl), np.interp(xb, rl, tl)
                if (yb - ya) >= S["tri_min"] * y[-1]:
                    kw = dict(color="k", lw=S["tri_lw"], ls=S["tri_ls"])
                    ax.plot([xa, xb], [ya, ya], **kw); ax.plot([xb, xb], [ya, yb], **kw)
                ax.annotate(S["c_label"], (0.5 * (xa + xb), 0.5 * (ya + yb)), xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", color="k")
            ax.set_ylim(0, y[-1])

    def plot_pressure(self):
        fr = self.frames; t = self.col("t")
        dx = min(f["dx"] for f in fr)
        s = np.arange(0.0, self.S["p_rmax_R0"] * self.R0, 0.5 * dx)
        P = np.array([np.interp(s, f["line_s"], f["line_p"]) for f in fr])
        allp = P[np.isfinite(P) & (P > 0)]
        lo, hi = np.percentile(allp, 1.0), np.percentile(allp, 99.9)
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        im = ax.pcolormesh(s / self.R0, t / self.tc, np.clip(P, lo, hi), cmap=self.S["p_cmap"],
                           norm=LogNorm(vmin=lo, vmax=hi), shading=self.S["p_shading"], rasterized=True)
        self._overlays(ax, s / self.R0, t / self.tc, np.clip(P, lo, hi), self.col("R_ray"), lo, hi)
        ax.set_xlim(0, self.S["p_rmax_R0"]); ax.set_xlabel(r"$r/R_0$"); ax.set_ylabel(self.tlabel)
        fig.colorbar(im, ax=ax, label=r"Pressure, $p$ (Pa)", pad=0.02)
        self._suptitle(fig, ax, "pressure"); self._save(fig, "pressure_tR")

    def plot_waves(self):
        fr = self.frames; t = self.col("t"); c = self._c_l(); rho = self.liq.rho; pinf = self.liq.p_inf
        s = fr[0]["line_s"]
        rmax = (self.S["waves_rmax_R0"] * self.R0) if self.S["waves_rmax_R0"] else s[-1]
        sg = s[s <= rmax]
        Jp, Jm = [], []
        for f in fr:
            p = np.interp(sg, f["line_s"], f["line_p"]) - pinf; u = np.interp(sg, f["line_s"], f["line_u"])
            Jp.append(0.5 * (p + rho * c * u)); Jm.append(0.5 * (p - rho * c * u))
        Jp, Jm = np.array(Jp), np.array(Jm)
        vmax = np.nanpercentile(np.concatenate([Jp.ravel(), Jm.ravel()]), 99.5)
        lin = vmax * self.S["waves_linthresh"]
        norm = SymLogNorm(linthresh=lin, vmin=0.0, vmax=vmax, base=10)
        fig, axs = plt.subplots(1, 2, figsize=self.S["size_maps"], sharey=True)
        for ax, J, lb in zip(axs, (Jp, Jm), ("Outbound", "Inbound")):
            im = ax.pcolormesh(sg / self.R0, t / self.tc, J, cmap=self.S["waves_cmap"], norm=norm,
                               shading="gouraud", rasterized=True)
            self._overlays(ax, sg / self.R0, t / self.tc, np.where(J > 0, J, np.nan), self.col("R_ray"), lin, vmax)
            ax.set_xlabel(r"$r/R_0$")
            ax.text(0.96, 0.04, lb, transform=ax.transAxes, ha="right", va="bottom", bbox=dict(fc="w", ec="none", pad=1.5))
        axs[0].set_ylabel(self.tlabel)
        k1 = int(np.floor(np.log10(vmax))); k0 = int(np.ceil(np.log10(lin)))
        fig.colorbar(im, ax=list(axs), label=r"$(p' \pm \rho c_l u_r)/2$ (Pa)", pad=0.02,
                     ticks=[0.0] + [10.0 ** k for k in range(k0, k1 + 1)])
        self._suptitle(fig, axs, "waves"); self._save(fig, "waves")

    # ---- shapes, planes, gif ---------------------------------------------------------------- #
    def _bubble_col(self):
        return self.S["bubble_color"][self.coated]

    def _pick(self, times):
        t = self.col("t"); R = self.col("R_vol"); fs = _plotfiles(self.plotdir)
        names = [f["name"] for f in self.frames]
        out = []
        for tt in times:
            i = int(np.nanargmin(R)) if tt == "Rmin" else int(np.argmin(np.abs(t / self.tc - float(tt))))
            out.append((tt, t[i], next(p for p in fs if os.path.basename(p) == names[i])))
        return out

    def plot_shapes(self, times=None):
        times = times or self.S["shapes_times"]
        picks = self._pick(times)
        cw, ch = self.S["shape_cell"]
        fig = plt.figure(figsize=(cw * len(picks), ch + 0.3))
        yt = _yt()
        for c, (tt, t, pf) in enumerate(picks):
            ds = yt.load(pf)
            ax = fig.add_subplot(1, len(picks), c + 1, projection="3d" if ds.dimensionality == 3 else None)
            ok = _draw_bubble(ax, ds, self.R0, self._bubble_col(), self.S)
            ax.set_title(r"$t/t_c = %.2f$%s%s" % (t / self.tc, r" ($R_\mathrm{min}$)" if tt == "Rmin" else "",
                                                     "" if ok else "\n(no $\\eta = 0.5$ surface)"), pad=-2)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=self.S["shape_top"], wspace=-0.08)
        self._suptitle(fig, fig.axes, "shapes"); self._save(fig, "shapes")

    def plot_planes(self, times=None, n=None, normalize=None):
        S = self.S
        times = times or S["planes_times"]; n = n or S["planes_n"]
        normalize = S["planes_normalize"] if normalize is None else normalize
        t = self.col("t"); R = self.col("R_vol"); fs = _plotfiles(self.plotdir); names = [f["name"] for f in self.frames]
        if times == "auto":
            pick = set(np.round(np.linspace(0, len(t) - 1, n)).astype(int)); pick.add(int(np.nanargmin(R)))
        else:
            pick = {int(np.nanargmin(R)) if x == "Rmin" else int(np.argmin(np.abs(t / self.tc - float(x)))) for x in times}
        pick = sorted(pick)
        if self.frames[0]["dim"] != 3:
            print("  [planes] 2D run -- skipped"); return
        jobs = [(next(p for p in fs if os.path.basename(p) == names[i]), self.R0, S["planes_res"]) for i in pick]
        cuts = _pool_map(_plane_frame, jobs, self.nproc)
        tn = t[pick] / self.tc; imin = int(np.nanargmin(R))
        norm = matplotlib.colors.Normalize(vmin=tn.min(), vmax=tn.max()); cmap = plt.get_cmap(S["planes_cmap"])
        fig, axs = plt.subplots(2, 2, figsize=S["planes_size"], sharex=True, sharey=True); axs = axs.ravel()
        for i, (a, Es) in zip(pick, cuts):
            col = cmap(norm(t[i] / self.tc))
            for ax, E in zip(axs, Es):
                if not (np.nanmin(E) < 0.5 < np.nanmax(E)):
                    continue
                sc = 1.0
                if normalize:
                    cs = ax.contour(a / self.R0, a / self.R0, E.T, levels=[0.5], alpha=0)
                    v = np.concatenate([p.vertices for p in cs.get_paths()]) if cs.get_paths() else None
                    cs.remove()
                    if v is None or not len(v):
                        continue
                    sc = 1.0 / np.mean(np.hypot(v[:, 0], v[:, 1]))
                ax.contour(a / self.R0 * sc, a / self.R0 * sc, E.T, levels=[0.5], colors=[col],
                           linewidths=1.6 if i == imin else 1.0)
        lim = 1.3 if normalize else 1.08
        for ax, (lab, *_r) in zip(axs, _PLANES):
            ax.set_aspect("equal"); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.text(0.04, 0.94, lab, transform=ax.transAxes, va="top", fontsize=S["font_size"] + 1)
        la = r"$\hat{r}/\bar{R}$" if normalize else r"$r/R_0$"
        for ax in axs[2:]:
            ax.set_xlabel(la)
        for ax in axs[::2]:
            ax.set_ylabel(la)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
        fig.colorbar(sm, ax=list(axs), label=self.tlabel, pad=0.03, fraction=0.04)
        self._suptitle(fig, axs, "planes"); self._save(fig, "planes")

    def make_gif(self):
        S = self.S; fs = _plotfiles(self.plotdir)
        fdir = self.stem + "_shapes_movie_frames"; os.makedirs(fdir, exist_ok=True)
        col = self._bubble_col()
        hl = tuple(1.0 - S["gif_highlight"] * (1.0 - np.array(to_rgb(col))))
        rc = {k: matplotlib.rcParams[k] for k in ("font.family", "font.serif", "mathtext.fontset", "mathtext.rm",
                                                  "mathtext.it", "mathtext.bf", "font.size")}
        jobs = []
        for k, pf in enumerate(fs):
            png = os.path.join(fdir, "frame_%04d.png" % k)
            if not (os.path.isfile(png) and os.path.getmtime(png) > os.path.getmtime(pf)):
                jobs.append((pf, self.R0, self.tc, png, col, hl, rc, S, self.tlabel))
        if self.verbose:
            print("  [gif] %d frames, %d to render" % (len(fs), len(jobs)))
        _pool_map(_gif_frame, jobs, self.nproc)
        from PIL import Image
        rgb = [Image.open(p).convert("RGB") for p in (os.path.join(fdir, "frame_%04d.png" % k) for k in range(len(fs)))
               if os.path.isfile(p)]
        if rgb:
            pick = rgb[::max(1, len(rgb) // 12)]
            strip = Image.new("RGB", (pick[0].width, pick[0].height * len(pick)))
            for k, im in enumerate(pick):
                strip.paste(im, (0, k * pick[0].height))
            pal = strip.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            imgs = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
            imgs[0].save(self.stem + "_shapes_movie.gif", save_all=True, append_images=imgs[1:],
                         duration=int(1000 / S["gif_fps"]), loop=0)
            if self.verbose:
                print("  wrote %s_shapes_movie.gif (%d frames; PNGs in %s/)" % (self.stem, len(imgs), fdir))

    # ---- modes -------------------------------------------------------------------------------- #
    def shape_modes(self):
        L = self.S["modes_lmax"]; out = []
        for f in self.frames:
            r, d = f["rad"], f["dirs"]; ok = np.isfinite(r)
            if f["dim"] != 3 or ok.sum() < 0.8 * len(r):
                out.append(np.full(L + 1, np.nan)); continue
            r, d = r[ok], d[ok]; x = r / r.mean() - 1
            th = np.arccos(np.clip(d[:, 2], -1, 1)); ph = np.arctan2(d[:, 1], d[:, 0])
            cols, idx = [], []
            for l in range(L + 1):
                for m in range(-l, l + 1):
                    cols.append(_real_ylm(l, m, th, ph)); idx.append(l)
            cf = np.linalg.lstsq(np.stack(cols, 1), x, rcond=None)[0]; idx = np.array(idx)
            out.append(np.array([np.sqrt(np.sum(cf[idx == l] ** 2) / (4 * np.pi)) for l in range(L + 1)]))
        return np.array(out)

    def plot_modes(self):
        t = self.col("t"); R = self.col("R_vol"); L = self.S["modes_lmax"]; a = self.shape_modes()
        octant = bool(self.frames[0]["octant"])
        ls = [l for l in ([4, 6, 8] if octant else range(1, L + 1)) if l <= L]
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        for l, cc in zip(ls, plt.get_cmap("viridis")(np.linspace(0, 0.9, len(ls)))):
            ax.semilogy(t / self.tc, a[:, l], "-", color=cc, label=r"$\ell = %d$" % l)
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"Mode amplitude, $a_\ell$"); self._xlim(ax, t)
        ax.legend(loc="upper left", ncol=2 if len(ls) > 4 else 1, title=("octant: cubic modes" if octant else "full bubble"))
        self._axtitle(ax, "modes"); self._save(fig, "modes")
        good = np.where(np.isfinite(a[:, ls[0]]))[0]
        if len(good):
            i = good[np.argmin(R[good])]; la = np.arange(1, L + 1)
            fig, ax = plt.subplots(figsize=self.S["size_spectrum"])
            ax.bar(la, a[i, 1:], color=["0.75" if (octant and l not in ls) else self.S["c_sim"] for l in la])
            ax.set_yscale("log"); ax.set_xlabel(r"Degree, $\ell$"); ax.set_xticks(la); ax.set_ylabel(r"Mode amplitude, $a_\ell$")
            ax.text(0.97, 0.97, r"at $R_\mathrm{min}$, $t/t_c = %.2f$" % (t[i] / self.tc)
                    + ("\ngrey: zero by octant symmetry" if octant else ""), transform=ax.transAxes, ha="right", va="top")
            self._axtitle(ax, "modes_spectrum"); self._save(fig, "modes_spectrum")

    # ---- your original diagnostics ---------------------------------------------------------- #
    def plot_r_volume(self):
        t = self.col("t")
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        self._models_on(ax)
        ax.plot(t / self.tc, self.col("R_ray") / self.R0, ":", color=self.S["c_ray"], lw=1.2, label=r"$\eta = 0.5$")
        ax.plot(t / self.tc, self.col("R_vol") / self.R0, "-o", color=self.S["c_vol"], ms=2.5, label=r"$R_\mathrm{volume}$")
        ax.plot(t / self.tc, self.col("R_shell") / self.R0, "-s", color=self.S["c_shell"], ms=2.5,
                label=r"$\langle\eta\rangle = 0.5$")
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"$R/R_0$"); self._xlim(ax, t); ax.legend(loc="lower left")
        self._axtitle(ax, "r_volume"); self._save(fig, "R_volume")

    def plot_eta_band(self):
        t = self.col("t"); th = self.S["band_thresholds"]; S = self.S
        B = {x: self.col("ray_%.2f" % x) for x in th}
        fig, ax = plt.subplots(figsize=(S["width"], S["height"]))
        ax.fill_between(t / self.tc, B[th[0]] / self.R0, B[th[-1]] / self.R0, color=_light(S["band_fill"], 0.18), lw=0,
                        label=r"$%.1f \leq \eta \leq %.1f$" % (th[0], th[-1]))
        for x, cc in zip(th, (S["band_lo"], S["band_mid"], S["band_hi"])):
            ax.plot(t / self.tc, B[x] / self.R0, "-" if abs(x - 0.5) < 1e-9 else "--", color=cc,
                    lw=1.5 if abs(x - 0.5) < 1e-9 else 0.9, label=r"$\eta = %.1f$" % x)
        self._models_on(ax)
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"$R/R_0$"); self._xlim(ax, t); ax.legend(loc="lower left")
        self._axtitle(ax, "eta_band"); self._save(fig, "eta_band")

    def plot_ic_pressure(self):
        fr = self.frames[: self.S["ic_frames"]]
        R0, pinf, pb = self.R0, self.liq.p_inf, self.gas.p_g0
        eps = self.meta.get("epsilon") or 0.08 * R0
        g_l, pi_l, g_g = self.meta["gamma_l"], self.meta["pi_l"], self.meta["gamma_g"]
        r = np.linspace(0.5 * R0, self.S["ic_rmax_R0"] * R0, 600)
        p_ic = pinf - (pinf - pb) * R0 / np.maximum(r, R0)
        el = 0.5 * (1 + np.tanh((r - R0) / eps)); eg = 1 - el
        A = el / (g_l - 1) + eg / (g_g - 1); Bc = el * g_l * pi_l / (g_l - 1)
        E = el * (p_ic + g_l * pi_l) / (g_l - 1) + eg * pb / (g_g - 1)
        p_pred = (E - Bc) / A
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        cm = plt.get_cmap("viridis")
        for j, f in enumerate(fr):
            m = f["line_s"] <= r[-1]
            ax.plot(f["line_s"][m] / R0, f["line_p"][m] / 1e6, "o-", color=cm(j / max(len(fr) - 1, 1)), ms=3, lw=1.2,
                    label=r"$t = %.1f\,\mu$s" % (f["t"] * 1e6))
        ax.plot(r / R0, p_ic / 1e6, "--", color="0.55", lw=1.5, label="initial condition, Sch20 eq. (27)")
        ax.plot(r / R0, p_pred / 1e6, "-", color="k", lw=2.0, label="predicted after pressure relaxation")
        ax.axhline(pb / 1e6, color="tab:red", ls=":", lw=1.0, label=r"$p_b$")
        ax.axhline(pinf / 1e6, color="tab:blue", ls=":", lw=1.0, label=r"$p_\infty$")
        ax.axvline(1.0, color="0.3", ls=":", lw=0.8, label=r"$r = R_0$ and $R_0 + \varepsilon$")
        ax.axvline(1.0 + eps / R0, color="0.3", ls=":", lw=0.8)
        ax.set_xlim(0.5, self.S["ic_rmax_R0"]); ax.set_xlabel(r"$r/R_0$"); ax.set_ylabel(r"Pressure, $p$ (MPa)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.25, 0.95), ncol=2)
        self._axtitle(ax, "ic_pressure"); self._save(fig, "IC_pressure")

    def plot_driven(self):
        from scipy.signal import argrelextrema
        t = self.col("t"); R = self.col("R_vol"); pe = np.array([f["line_p"][-1] for f in self.frames])
        fig, axs = plt.subplots(1, 3, figsize=(self.S["width"] * 1.5, self.S["height"] * 0.8))
        imx = argrelextrema(R, np.greater)[0]; imn = argrelextrema(R, np.less)[0]
        axs[0].plot(t / self.tc, R / self.R0, "-", color="0.7", lw=0.8)
        axs[0].plot(t[imx] / self.tc, R[imx] / self.R0, "^-", color=self.S["c_sim"], ms=4, label=r"$R_\mathrm{max}$")
        axs[0].plot(t[imn] / self.tc, R[imn] / self.R0, "v-", color=self.S["c_km"], ms=4, label=r"$R_\mathrm{min}$")
        axs[0].set_xlabel(self.tlabel); axs[0].set_ylabel(r"$R/R_0$"); axs[0].legend(loc="best")
        tu = np.linspace(t[0], t[-1], 2 * len(t)); Ru = np.interp(tu, t, R)
        sp = np.abs(np.fft.rfft((Ru - Ru.mean()) * np.hanning(len(Ru)))); fq = np.fft.rfftfreq(len(Ru), tu[1] - tu[0]) * self.tc
        axs[1].semilogy(fq[1:], sp[1:] / sp[1:].max(), "-", color=self.S["c_sim"])
        axs[1].set_xlabel(r"Frequency, $f\,t_c$"); axs[1].set_ylabel("Spectrum of $R$ (normalised)")
        axs[2].plot(pe / 1e6, R / self.R0, "-", color=self.S["c_sim"], lw=1.0)
        axs[2].set_xlabel("Edge pressure (MPa)"); axs[2].set_ylabel(r"$R/R_0$")
        fig.subplots_adjust(wspace=0.35)
        self._suptitle(fig, axs, "driven"); self._save(fig, "driven")

    def plot_gamma(self):
        t = self.col("t"); R = self.col("R_vol")
        fig, ax = plt.subplots(figsize=(self.S["width"], self.S["height"]))
        ax.plot(t / self.tc, self.col("gamma_band"), "-o", color=self.S["c_sim"], ms=2.5, label=r"$\Gamma$ (band average)")
        ax.plot(t / self.tc, (self.R0 / R) ** 2, "--", color="k", label=r"$(R_0/R_\mathrm{volume})^2$")
        ax.set_xlabel(self.tlabel); ax.set_ylabel(r"Surface concentration, $\Gamma$"); self._xlim(ax, t)
        ax.legend(loc="upper left")
        self._axtitle(ax, "gamma"); self._save(fig, "dbg_gamma")

    def write_csv(self):
        t, R, _ = self.radius("ray")
        Rk = np.interp(t, self.km[0], self.km[1]); Rr = np.interp(t, self.rp[0], self.rp[1])
        with open(self.stem + ".csv", "w") as fh:
            fh.write("time_s,t_over_tc,R_ray_m,R_volume_m,R_shell_m,R_km_m,R_rp_m,err_km_pct\n")
            for row in zip(t, t / self.tc, R, self.col("R_vol"), self.col("R_shell"), Rk, Rr, 100 * (R - Rk) / Rk):
                fh.write(",".join("%.9e" % v for v in row) + "\n")
        if self.verbose:
            print("  wrote", self.stem + ".csv")

    # ---- debugging -------------------------------------------------------------------------- #
    def plot_debug(self, which=("conservation", "thermo", "wall_balance", "reflection", "health")):
        t = self.col("t"); x = t / self.tc; Rv = self.col("R_vol"); S = self.S
        W, H = S["width"], S["height"]; gam = self.gas.kappa_g; pb = self.gas.p_g0
        if "conservation" in which:
            mg = self.col("gas_mass")
            fig, ax = plt.subplots(figsize=(W, H))
            ax.plot(x, 100 * (mg / mg[0] - 1), "-o", color=S["c_sim"], ms=2.5, label="gas mass")
            ax.axhline(0, color="0.6", lw=0.6); ax.set_xlabel(self.tlabel); ax.set_ylabel("Drift from $t = 0$ (%)")
            ax.legend(loc="best"); self._xlim(ax, t); self._axtitle(ax, "conservation"); self._save(fig, "dbg_conservation")
        if "thermo" in which:
            pg, rg = self.col("p_gas"), self.col("rho_gas")
            fig, ax = plt.subplots(figsize=(W, H))
            ax.semilogy(x, pg, "-o", color=S["c_sim"], ms=2.5, label="Mean Gas Core")
            ax.semilogy(x, pb * (Rv[0] / Rv) ** (3 * gam), "--", color="k", label=r"Adiabatic, $p_b (R_0/R)^{3\gamma}$")
            ax.semilogy(x, pb * (Rv[0] / Rv) ** 3, ":", color="k", label=r"Isothermal, $p_b (R_0/R)^{3}$")
            r0 = rg[np.isfinite(rg)][0] if np.isfinite(rg).any() else np.nan
            ax.semilogy(x, pb * (rg / r0) ** gam, "-.", color="0.45",
                        label=r"Adiabatic Core Density, $p_b (\rho_g/\rho_{g,0})^{\gamma}$")
            ax.set_xlabel(self.tlabel); ax.set_ylabel(r"Pressure, $p$ (Pa)"); ax.legend(loc="best"); self._xlim(ax, t)
            self._axtitle(ax, "thermo"); self._save(fig, "dbg_thermo")
        if "wall_balance" in which:
            meas = []
            for f in self.frames:
                j = np.where(f["line_eta"] > 0.95)[0]
                meas.append(f["p_gas"] - f["line_p"][j[0]] if len(j) else np.nan)
            V = _deriv(t, Rv); sig = np.array([self.shell.sigma(r) for r in Rv])
            terms = {r"$2\sigma/R$": 2 * sig / Rv, r"$4\mu \dot R/R$": 4 * self.liq.mu * V / Rv}
            if self.shell.kappa_s:
                terms[r"$4\kappa_s \dot R/R^2$"] = 4 * self.shell.kappa_s * V / Rv ** 2
            fig, ax = plt.subplots(figsize=(W, H))
            ax.plot(x, np.array(meas) / 1e3, "-o", color=S["c_sim"], ms=2.5, label=r"measured $p_g - p_l(R^+)$")
            for lb, v in terms.items():
                ax.plot(x, v / 1e3, "--", lw=0.9, label=lb)
            ax.plot(x, sum(terms.values()) / 1e3, "-", color="k", lw=1.2, label="model total")
            ax.axhline(0, color="0.6", lw=0.6); ax.set_xlabel(self.tlabel); ax.set_ylabel("Pressure jump at the wall (kPa)")
            ax.legend(loc="best"); self._xlim(ax, t); self._axtitle(ax, "wall_balance"); self._save(fig, "dbg_wall_balance")
        if "reflection" in which:
            c = self._c_l(); rho = self.liq.rho; s = self.frames[0]["line_s"]; re_ = 0.97 * s[-1]
            pp = np.array([np.interp(re_, f["line_s"], f["line_p"]) for f in self.frames]) - self.liq.p_inf
            uu = np.array([np.interp(re_, f["line_s"], f["line_u"]) for f in self.frames])
            jp, jm = 0.5 * (pp + rho * c * uu), 0.5 * (pp - rho * c * uu)
            fig, ax = plt.subplots(figsize=(W, H))
            ax.plot(x, jp / 1e3, "-", color=S["c_sim"], label="outbound (leaving)")
            ax.plot(x, jm / 1e3, "-", color=S["c_km"], label="inbound (reflected)")
            ax.axhline(0, color="0.6", lw=0.6)
            ax.text(0.98, 0.05, r"max inbound / max outbound $= %.2f$" % (np.nanmax(np.abs(jm)) / max(np.nanmax(np.abs(jp)), 1e-30)),
                    transform=ax.transAxes, ha="right")
            ax.set_xlabel(self.tlabel); ax.set_ylabel(r"Amplitude at $r = %.0f R_0$ (kPa)" % (re_ / self.R0))
            ax.legend(loc="best"); self._xlim(ax, t); self._axtitle(ax, "reflection"); self._save(fig, "dbg_reflection")
        if "health" in which:
            axr = np.array([f["axis_R"] for f in self.frames])
            fig, axs = plt.subplots(2, 2, figsize=S["size_tall"], sharex=True, gridspec_kw={"hspace": 0.08, "wspace": 0.35})
            axs[0, 0].semilogy(x, self.col("umax_gas"), "-", color=S["c_sim"]); axs[0, 0].set_ylabel(r"max $|u|$ in gas (m/s)")
            axs[0, 1].plot(x, self.col("n_stray"), "-", color=S["c_sim"]); axs[0, 1].set_ylabel("stray gas cells")
            axs[1, 0].plot(x, self.col("p_min") / 1e3, "-", color=S["c_sim"]); axs[1, 0].set_ylabel(r"min $p$ (kPa)")
            if self.frames[0]["octant"]:
                axs[1, 1].text(0.5, 0.5, "octant run:\n$\\pm$axis symmetry\nimposed", transform=axs[1, 1].transAxes,
                               ha="center", va="center"); axs[1, 1].set_yticks([])
            else:
                axs[1, 1].plot(x, 100 * (np.nanmax(axr, 1) - np.nanmin(axr, 1)) / np.nanmean(axr, 1), "-", color=S["c_sim"])
                axs[1, 1].set_ylabel(r"$\pm$axis radius spread (%)")
            for a in axs[1]:
                a.set_xlabel(self.tlabel)
            self._xlim(axs[0, 0], t); self._suptitle(fig, axs.ravel(), "health"); self._save(fig, "dbg_health")

    # ---- driver --------------------------------------------------------------------------- #
    PLOTS = dict(radius="plot_radius", residual="plot_residual", velocity="plot_velocity", probes="plot_probes",
                 pressure="plot_pressure", waves="plot_waves", shapes="plot_shapes", planes="plot_planes",
                 modes="plot_modes", gif="make_gif", r_volume="plot_r_volume", eta_band="plot_eta_band",
                 ic_pressure="plot_ic_pressure", driven="plot_driven", gamma="plot_gamma", csv="write_csv")

    def run(self, plots, debug=()):
        if not os.path.isdir(self.plotdir):
            print("  [skip] no plotfile directory:", self.plotdir); return
        if len(self.frames) < 2:
            print("  [skip] fewer than 2 usable frames in", self.plotdir); return
        self.summary()
        for p in plots:
            if p not in self.PLOTS:
                print("  [warn] unknown plot '%s' (choices: %s)" % (p, ", ".join(self.PLOTS))); continue
            try:
                getattr(self, self.PLOTS[p])()
            except Exception as exc:
                print("  [error] %s failed: %s" % (p, exc))
        if debug:
            self.plot_debug(tuple(debug))


# =========================================================================================== #
#  module-level workers (picklable for the process pool) and small helpers
# =========================================================================================== #
_PLANES = [("(001)", (0, 0, 1), (1, 0, 0), (0, 1, 0)), ("(010)", (0, 1, 0), (1, 0, 0), (0, 0, 1)),
           ("(100)", (1, 0, 0), (0, 1, 0), (0, 0, 1)), ("(111)", (1, 1, 1), (1, -1, 0), (1, 1, -2))]


def _plane_frame(args):
    pf, R0, n = args
    ds = _yt().load(pf)
    E, dx, off, sym = _eta_around(ds, 1.25 * R0)
    a = np.linspace(-1.25 * R0, 1.25 * R0, n); A, B = np.meshgrid(a, a, indexing="ij")
    out = []
    for lab, nrm, e1, e2 in _PLANES:
        e1 = np.array(e1, float) / np.linalg.norm(e1); e2 = np.array(e2, float) / np.linalg.norm(e2)
        P = [A * e1[k] + B * e2[k] for k in range(3)]
        P = [np.abs(p) if sym[k] else p for k, p in enumerate(P)]
        out.append(_sample(E, [p.ravel() for p in P], dx, off).reshape(A.shape))
    return a, out


def _draw_bubble(ax, ds, R0, col, S, highlight="#fff", shadow="#222"):
    E, dx, off, sym = _eta_around(ds, 1.3 * R0)
    if ds.dimensionality == 3:
        th = np.linspace(0, 2 * np.pi, S["shape_nth"] + 1); ph = np.linspace(0, np.pi, S["shape_nph"] + 1)
        TT, PP = np.meshgrid(th, ph, indexing="ij")
        d = np.stack([np.sin(PP) * np.cos(TT), np.sin(PP) * np.sin(TT), np.cos(PP)], -1).reshape(-1, 3)
        Rf = _radii(E, dx, off, sym, d, 1.3 * R0).reshape(TT.shape)
        if np.all(np.isnan(Rf)):
            ax.set_axis_off(); return False
        Rf = np.where(np.isnan(Rf), np.nanmean(Rf), Rf)
        X, Y, Z = Rf * np.sin(PP) * np.cos(TT), Rf * np.sin(PP) * np.sin(TT), Rf * np.cos(PP)
        rgb = LightSource(azdeg=315, altdeg=45).shade(
            Z / R0, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("c", [shadow, col, highlight]),
            blend_mode="soft", vert_exag=0.5)
        ax.plot_surface(X / R0, Y / R0, Z / R0, facecolors=rgb, rstride=1, cstride=1, linewidth=0,
                        antialiased=False, shade=False)
        L = 1.05 * (1.0 if S["shape_scale_with_R"] else np.nanmax(Rf) / R0)
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(-L, L)
        ax.set_box_aspect((1, 1, 1), zoom=S["shape_zoom"]); ax.view_init(S["shape_elev"], S["shape_azim"]); ax.set_axis_off()
    else:
        th = np.linspace(0, 2 * np.pi, 721)
        rr = _radii(E, dx, off, sym, np.stack([np.cos(th), np.sin(th)], -1), 1.3 * R0)
        if np.all(np.isnan(rr)):
            ax.set_axis_off(); return False
        rr = np.where(np.isnan(rr), np.nanmean(rr), rr)
        ax.fill(rr * np.cos(th) / R0, rr * np.sin(th) / R0, color=col, lw=0)
        ax.plot(rr * np.cos(th) / R0, rr * np.sin(th) / R0, color="k", lw=0.6)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    return True


def _gif_frame(args):
    pf, R0, tc, png, col, hl, rc, S, tlabel = args
    matplotlib.rcParams.update(rc)
    ds = _yt().load(pf)
    fig = plt.figure(figsize=S["gif_size"])
    ax = fig.add_subplot(111, projection="3d" if ds.dimensionality == 3 else None)
    ok = _draw_bubble(ax, ds, R0, col, S, highlight=hl, shadow=S["gif_shadow"])
    ax.set_title(r"%s$ = %.3f$%s" % (tlabel, float(ds.current_time) / tc, "" if ok else "  (no $\\eta = 0.5$ surface)"),
                 fontsize=S["font_size"] + 1, pad=-4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    fig.savefig(png, dpi=S["gif_dpi"]); plt.close(fig)
    return png


def _real_ylm(l, m, th, ph):
    from scipy.special import sph_harm_y
    if m == 0:
        return np.real(sph_harm_y(l, 0, th, ph))
    y = sph_harm_y(l, abs(m), th, ph)
    return np.sqrt(2) * (-1) ** m * (np.imag(y) if m < 0 else np.real(y))


# =========================================================================================== #
#  5. SCRIPT ENTRY POINT -- what every analyze_Sch20_* script calls
# =========================================================================================== #
def main(cfg):
    """Run one configured analysis.  cfg keys: INPUT, OUTPUT_DIR, CASE, COATED, STEM, N_PROC, PLOTS,
    DEBUG, STYLE (overrides), TITLES (overrides), LEGACY_MODEL_PLOTS (coated: sigma / damping / variant plots).
    Command line:  python3 analyze_Sch20_X.py [PLOTDIR] [--input DECK] [--stem NAME] [--nproc N] [--models-only]"""
    import argparse
    ap = argparse.ArgumentParser(description=cfg.get("DESCRIPTION", "Sch20 analysis"))
    ap.add_argument("plotdir", nargs="?", default=cfg["OUTPUT_DIR"], help="plotfile directory (default: OUTPUT_DIR)")
    ap.add_argument("--input", default=cfg["INPUT"], help="input deck the run used (default: INPUT)")
    ap.add_argument("--stem", default=cfg["STEM"])
    ap.add_argument("--nproc", type=int, default=cfg.get("N_PROC", 4))
    ap.add_argument("--models-only", action="store_true", help="reference-model plots only (no plotfiles needed)")
    a = ap.parse_args()
    out_dir = cfg.get("OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "Images")
    if cfg.get("COATED") and cfg.get("LEGACY_MODEL_PLOTS", True):
        _M.run_case(a.input, out_hint=a.plotdir, shell_only=True, models_only=a.models_only, stem=a.stem, model="KM")
    if a.models_only:
        return
    A = Sch20Analysis(a.input, a.plotdir, case=cfg["CASE"], coated=cfg.get("COATED", False), stem=a.stem,
                      out_dir=out_dir, nproc=a.nproc, style=cfg.get("STYLE"), titles=cfg.get("TITLES"))
    A.run(cfg.get("PLOTS", []), debug=cfg.get("DEBUG", ()))
    return A
