#!/usr/bin/env python3
"""
debug_02_04 -- crash / failure analysis of the MarmPaper micro runs 02_FreeCollapse and 04_ForcedNonlinear.

    python3 debug_02_04.py                              # both runs, INCLINE paths from the decks' plot_file
    python3 debug_02_04.py --only 02 --nproc 16
    python3 debug_02_04.py --o2 DIR --o4 DIR --log02 FLAMES_out/flame_123_stdout --log04 ...
    python3 debug_02_04.py --refresh                    # ignore the frame cache

Works on partial / crashed runs: every plotfile is read independently (a truncated last plotfile is
reported and skipped) and the results are cached in <out>/<stem>_dbg_cache.pkl, so a re-run only reads
new frames.  Needs yt, scipy, matplotlib (same as sch20_analysis.py).

Per run (<stem> = 02_FreeCollapse / 04_ForcedNonlinear), written to ./Images next to this script:
  <stem>_dbg_radius.png    R/R0 vs coated KM: +x ray, x/y/z axis radii, (1,1,1) diagonal, min/mean/max over
                           400 directions, R_vol (plotfiles and thermo.dat); shape spread; area excess
  <stem>_dbg_contours.png  eta = 0.5 contours on the z = 0 plane and on the x = y diagonal plane, every
                           frame coloured by time; last frames; R(theta)/<R> - 1 kymograph
  <stem>_dbg_modes.png     cos(n theta) amplitudes of the z = 0 contour vs time (wrinkle growth), last spectrum
  <stem>_dbg_health.png    domain-wide extrema per frame: p, rho, T, eta over/undershoot, |u|max, NaN/Inf,
                           kappa*R and Gamma on the band, finest level / dx, dt and step (stdout log), thermo
  <stem>_dbg_fields.png    z = 0 slices (eta, p, |u|, kappaAvg*R, Gamma, shell, T, rho) at R_min and at the
                           last frame, with the eta = 0.5 contour and the domain-wide extreme locations
  <stem>_dbg_report.txt    text summary: completion, log tail + error lines, first anomaly per check,
                           last-frame table, where the extremes sit (r/R0, angle from +z, AMR level)
  02_04_dbg_overview.png   both runs, R/R0 vs KM, with the last frame / last log time marked

stdout logs: --log02/--log04, or the newest file in ~/FLAMES_out (or $FLAMES_OUT) whose header has the
deck's NSCBC "(domain <prob_hi>)" line.  The matching err_ file is read too.
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RP = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "FlowRayleighPlesset", "reference"))
sys.path.insert(0, _RP)
import sch20_analysis as SA                                           # noqa: E402
from sch20_analysis import Sch20Analysis, plt                         # noqa: E402

RUNS = {
    "02": dict(deck="input_02_FreeCollapse", stem="02_FreeCollapse", case="Collapsing", gain=1.0, half_R0=1.3),
    "04": dict(deck="input_04_ForcedNonlinear", stem="04_ForcedNonlinear", case="Forced Non-Linear", gain=1.107,
               half_R0=1.9),   # R_max 1.49 R0 expected; box gain 1.107 (20 R0 box at f0), see ../README.md
}
SLICE_FIELDS = ("eta", "pressure", "umag", "kappaAvg", "Gamma", "shell", "T", "density")
GLOBAL_FIELDS = ("pressure", "density", "T", "eta", "rho_eta0", "rho_eta1", "energy_per_vol",
                 "kappaAvg", "kappa1", "kappa2", "Gamma", "shell")
N_THETA = 181          # z = 0 plane / diagonal plane directions over the quarter circle
CACHE_VERSION = "dbg1"


# =========================================================================================== #
#  frame pass
# =========================================================================================== #
def _plane_dirs(which, n=N_THETA):
    a = np.linspace(0.0, 0.5 * np.pi, n)
    if which == "xy":                               # angle from +x towards +y
        return a, np.stack([np.cos(a), np.sin(a), 0 * a], -1)
    s = 1 / np.sqrt(2.0)                            # x = y plane: angle from +z towards the (1,1,0) diagonal
    return a, np.stack([s * np.sin(a), s * np.sin(a), np.cos(a)], -1)


def _plane_image(E, dx, off, sym, which, half):
    """eta on a plane, sampled at dx/2 (for contour drawing).  Returns (s1, s2, img)."""
    n = int(half / (0.5 * dx))
    s = (np.arange(n) + 0.5) * 0.5 * dx
    A, B = np.meshgrid(s, s, indexing="ij")
    if which == "xy":
        pts = [A, B, 0 * A]
    else:
        pts = [A / np.sqrt(2.0), A / np.sqrt(2.0), B]   # A = in-plane distance along (1,1,0), B = z
    pts = [np.abs(p) if sym[q] else p for q, p in enumerate(pts)]
    img = SA._sample(E, [p.ravel() for p in pts], dx, off).reshape(A.shape)
    return s, s, img


def _loc(ad_xyz, lev, i, R0):
    x, y, z = (float(c[i]) for c in ad_xyz)
    r = np.sqrt(x * x + y * y + z * z)
    th = np.degrees(np.arccos(z / r)) if r > 0 else 0.0
    return dict(x=x, y=y, z=z, r_R0=r / R0, theta_deg=th, level=int(lev[i]))


def _frame(args):
    pf, R0, half = args
    try:
        yt = SA._yt()
        ds = yt.load(pf)
        dim, dle, dre, sym, cen, dx = SA._geom(ds)
        has = lambda f: ("boxlib", f) in ds.field_list
        lev = ds.index.max_level
        o = {"t": float(ds.current_time), "name": os.path.basename(pf), "mtime": os.path.getmtime(pf),
             "step": int(re.search(r"(\d+)cell$", pf).group(1)), "dx": dx, "level": int(lev)}
        o["ncell_lev"] = [int(sum(int(np.prod(g.ActiveDimensions)) for g in ds.index.select_grids(l)))
                          for l in range(lev + 1)]
        # ---- shape: 3D finest eta box around the bubble ---------------------------------------
        E, dxe, offe, syme = SA._eta_around(ds, half * R0)
        dirs = SA._fib_dirs(400)
        rad = SA._radii(E, dxe, offe, syme, dirs, half * R0)
        o["rad"] = rad
        o["R_axes"] = np.array([SA._radii(E, dxe, offe, syme, [u], half * R0)[0]
                                for u in np.eye(3)])                              # x, y, z
        o["R_diag"] = float(SA._radii(E, dxe, offe, syme, [np.ones(3) / np.sqrt(3)], half * R0)[0])
        for which in ("xy", "dg"):
            th, dd = _plane_dirs(which)
            o["Rth_" + which] = SA._radii(E, dxe, offe, syme, dd, half * R0)
            o["img_" + which] = _plane_image(E, dxe, offe, syme, which, half * R0)
        # gas outside the main bubble (fragments, axis sheets): eta < 0.5 beyond 1.15 x the local max radius
        idx = np.indices(E.shape).reshape(3, -1).T
        rr = np.sqrt(np.sum(((idx + 0.5) * dxe + offe) ** 2, axis=1))
        Rmx = np.nanmax(rad) if np.isfinite(rad).any() else 0.0
        far = (E.ravel() < 0.5) & (rr > 1.15 * Rmx)
        o["n_stray"] = int(far.sum())
        o["stray_rmax_R0"] = float(rr[far].max() / R0) if far.any() else np.nan
        # ---- z = 0 slices (finest level, first cell above z = 0) -----------------------------
        m = int(np.ceil(half * R0 / dx))
        cg = ds.covering_grid(level=lev, left_edge=[0.0, 0.0, 0.0], dims=[m, m, 1])
        sl = {}
        for f in SLICE_FIELDS:
            if f == "umag":
                sl[f] = np.sqrt(sum(np.asarray(cg[("boxlib", "velocity" + c)], float)[:, :, 0] ** 2 for c in "xyz"))
            elif has(f):
                sl[f] = np.asarray(cg[("boxlib", f)], float)[:, :, 0]
        o["slice"] = sl
        o["slice_dx"] = dx
        # ---- domain-wide extrema, NaN/Inf, locations ----------------------------------------
        ad = ds.all_data()
        xyz = [np.asarray(ad["index", c], float) for c in "xyz"]
        lvl = np.asarray(ad["index", "grid_level"], float)
        r = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2 + xyz[2] ** 2)
        g = {}
        bad = np.zeros(len(r), bool)
        for f in GLOBAL_FIELDS:
            if not has(f):
                continue
            v = np.asarray(ad["boxlib", f], float)
            nf = ~np.isfinite(v)
            bad |= nf
            vv = np.where(nf, np.nan, v)
            g[f] = dict(min=float(np.nanmin(vv)) if (~nf).any() else np.nan,
                        max=float(np.nanmax(vv)) if (~nf).any() else np.nan, nbad=int(nf.sum()))
            if f in ("pressure", "density", "T", "eta") and (~nf).any():
                g[f]["argmin"] = _loc(xyz, lvl, int(np.nanargmin(vv)), R0)
                g[f]["argmax"] = _loc(xyz, lvl, int(np.nanargmax(vv)), R0)
            if f == "eta":
                eta = vv
        u = np.sqrt(sum(np.asarray(ad["boxlib", "velocity" + c], float) ** 2 for c in "xyz"))
        nfu = ~np.isfinite(u); bad |= nfu
        uu = np.where(nfu, np.nan, u)
        g["umag"] = dict(min=0.0, max=float(np.nanmax(uu)), nbad=int(nfu.sum()),
                         argmax=_loc(xyz, lvl, int(np.nanargmax(uu)), R0))
        o["nbad"] = int(bad.sum())
        o["bad_first"] = [_loc(xyz, lvl, int(i), R0) for i in np.where(bad)[0][:5]]
        # gas-side velocity and band curvature / Gamma (band = 0.05 < eta < 0.95 within 2.5 R0)
        gas = eta < 0.05
        o["umax_gas"] = float(np.nanmax(uu[gas])) if gas.any() else np.nan
        band = (eta > 0.05) & (eta < 0.95) & (r < 2.5 * R0)
        o["band_cells"] = int(band.sum())
        Rm = float(np.nanmean(rad)) if np.isfinite(rad).any() else R0
        o["R_mean"] = Rm
        for f in ("kappaAvg", "kappa1", "kappa2", "Gamma", "shell"):
            if has(f) and band.any():
                v = np.asarray(ad["boxlib", f], float)[band]
                v = v[np.isfinite(v)]
                if len(v):
                    o["band_" + f] = (float(v.min()), float(np.median(v)), float(v.max()))
                    if f == "kappaAvg":
                        i = int(np.argmax(np.where(np.isfinite(np.asarray(ad["boxlib", f], float)) & band,
                                                   np.abs(np.asarray(ad["boxlib", f], float)), -1)))
                        o["kappa_argmax"] = _loc(xyz, lvl, i, R0)
        o["glob"] = g
        return o
    except Exception as exc:                                            # truncated / unreadable plotfile
        return {"error": "%s: %s: %s" % (os.path.basename(pf), type(exc).__name__, exc)}


def load_frames(plotdir, R0, half, cache_path, nproc, refresh=False):
    done = {}
    if cache_path and os.path.isfile(cache_path) and not refresh:
        try:
            with open(cache_path, "rb") as fh:
                c = pickle.load(fh)
            if c.get("key") == (CACHE_VERSION, half):
                done = c["frames"]
        except Exception:
            done = {}
    fs = SA._plotfiles(plotdir)
    todo = [pf for pf in fs if os.path.basename(pf) not in done
            or abs(done[os.path.basename(pf)]["mtime"] - os.path.getmtime(pf)) > 1e-6]
    print("  [data] %s: %d plotfiles, %d cached, %d to read (%d proc)" % (plotdir, len(fs), len(fs) - len(todo),
                                                                        len(todo), nproc))
    errors = []
    for res in SA._pool_map(_frame, [(pf, R0, half) for pf in todo], nproc):
        if "error" in res:
            print("  [data] SKIPPED", res["error"]); errors.append(res["error"])
        else:
            done[res["name"]] = res
    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump({"key": (CACHE_VERSION, half), "frames": done}, fh)
    fr = sorted([done[os.path.basename(pf)] for pf in fs if os.path.basename(pf) in done], key=lambda d: d["t"])
    return fr, errors


# =========================================================================================== #
#  thermo.dat and stdout log
# =========================================================================================== #
def read_thermo(plotdir):
    p = os.path.join(plotdir, "thermo.dat")
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            head = fh.readline().split()
        rows = []
        with open(p) as fh:
            next(fh)
            for ln in fh:
                w = ln.split()
                if len(w) == len(head):
                    try:
                        rows.append([float(x) for x in w])
                    except ValueError:
                        pass
        a = np.array(rows) if rows else np.zeros((0, len(head)))
        return {h: a[:, i] for i, h in enumerate(head)}
    except Exception as exc:
        print("  [thermo] unreadable:", exc)
        return None


ERR_PAT = re.compile(r"nan|inf\b|abort|assert|error|erroneous|segment|signal|sigsegv|sigfpe|killed|oom|"
                     r"cancel|time limit|terminat|floating point|negative|failed|exceed", re.I)
STEP_PAT = re.compile(r"STEP\s+(\d+)\s+ends\.\s+TIME\s*=\s*([-+0-9.eE]+)\s+DT\s*=\s*([-+0-9.eE]+)")
NOISE = re.compile(r"btl_tcp_proc|reachable pairing|Unused ParmParse|::|DynamicTimestep_Update|^\s*$")


def find_log(meta, explicit):
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    root = os.environ.get("FLAMES_OUT", os.path.expanduser("~/FLAMES_out"))
    if not os.path.isdir(root):
        return None
    key = "(domain %g)" % meta["prob_hi"][0]
    cands = sorted(glob.glob(os.path.join(root, "flame_*_stdout")), key=os.path.getmtime, reverse=True)
    for c in cands[:200]:
        try:
            with open(c, errors="replace") as fh:
                head = fh.read(200000)
        except OSError:
            continue
        if key in head and ("R0=%g" % meta["R0"]) in head:
            return c
    return None


def read_log(path):
    if not path:
        return None
    steps, tail, errs = [], [], []
    files = [path]
    m = re.search(r"flame_(\d+)_stdout$", path)
    if m:
        e = os.path.join(os.path.dirname(path), "err_flame_%s_stdout" % m.group(1))
        if os.path.isfile(e):
            files.append(e)
    for f in files:
        with open(f, errors="replace") as fh:
            for ln in fh:
                s = STEP_PAT.search(ln)
                if s:
                    steps.append((int(s.group(1)), float(s.group(2)), float(s.group(3))))
                    continue
                if NOISE.search(ln):
                    continue
                ln = re.sub(r"\x1b\[[0-9;]*m", "", ln.rstrip())
                if ERR_PAT.search(ln) and len(errs) < 400:
                    errs.append("%s: %s" % (os.path.basename(f), ln[:300]))
                tail.append("%s: %s" % (os.path.basename(f), ln[:300]))
                tail = tail[-60:]
    a = np.array(steps) if steps else np.zeros((0, 3))
    return dict(files=files, step=a[:, 0], t=a[:, 1], dt=a[:, 2], tail=tail, errors=errs)


# =========================================================================================== #
#  helpers
# =========================================================================================== #
def col(fr, k, i=None):
    out = []
    for f in fr:
        v = f.get(k, np.nan)
        out.append(v[i] if (i is not None and np.ndim(v)) else v)
    return np.array(out, float)


def gcol(fr, fld, k):
    return np.array([f["glob"].get(fld, {}).get(k, np.nan) for f in fr], float)


def rvol_thermo(th):
    V = 8.0 * th["gas_volume"]                                        # octant
    return (3 * V / (4 * np.pi)) ** (1 / 3)


def fourier(th, R):
    """cos(n theta) amplitudes of R(theta) on the quarter circle (mirror symmetric in x and y -> even n)."""
    ok = np.isfinite(R)
    if ok.sum() < 10:
        return None, None
    Rf = np.interp(th, th[ok], R[ok])
    full = np.concatenate([Rf, Rf[-2:0:-1]])                          # [0, pi] even about pi/2 ... -> period pi
    full = np.concatenate([full, full])                               # [0, 2 pi)
    c = np.fft.rfft(full) / len(full)
    a = 2 * np.abs(c); a[0] /= 2
    return np.arange(len(a)), a / a[0]


def _nm(stem):
    return os.path.basename(stem)


def _contour(ax, x, y, z, *a, **k):
    """ax.contour that skips frames whose image is too small (bubble unresolved / truncated read)."""
    z = np.asarray(z)
    if z.ndim == 2 and min(z.shape) >= 2 and len(x) == z.shape[1] and len(y) == z.shape[0]:
        ax.contour(x, y, z, *a, **k)


def save(fig, path):
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path)


# =========================================================================================== #
#  plots
# =========================================================================================== #
def plot_radius(A, fr, th, stem):
    R0 = A.R0; ns = 1e9
    t = col(fr, "t")
    tk, Rk, _ = A.km
    fig, axs = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3, 1.3]))
    ax = axs[0]
    ax.plot(tk * ns, Rk / R0, "-", color="0.2", lw=1.6, label="Coated KM")
    rad = np.array([f["rad"] for f in fr])
    with np.errstate(all="ignore"):
        lo, mn, hi = np.nanmin(rad, 1), np.nanmean(rad, 1), np.nanmax(rad, 1)
    ax.fill_between(t * ns, lo / R0, hi / R0, color="tab:red", alpha=0.18, lw=0, label="min–max, 400 dirs")
    ax.plot(t * ns, mn / R0, "-", color="tab:red", lw=1.4, label="mean, 400 dirs")
    for i, (c, n) in enumerate(zip(("tab:blue", "tab:green", "tab:purple"), "xyz")):
        ax.plot(t * ns, col(fr, "R_axes", i) / R0, ".", color=c, ms=3, label="%s axis" % n)
    ax.plot(t * ns, col(fr, "R_diag") / R0, "x", color="tab:orange", ms=3.5, label="(1,1,1) diagonal")
    if th is not None and len(th["time"]):
        Rv = rvol_thermo(th)
        Rv = Rv * R0 / Rv[np.argmax(Rv > 0)] if (Rv > 0).any() else Rv
        ax.plot(th["time"] * ns, Rv / R0, "-", color="tab:cyan", lw=0.9, label=r"$R_{vol}$ thermo.dat (norm.)")
    if A.shell.R_buck:
        ax.axhline(A.shell.R_buck / R0, color="0.6", ls=":", lw=0.8)
    ax.set_ylabel(r"$R/R_0$"); ax.legend(fontsize=7, ncol=2, loc="best")
    ax.set_title("%s — radius (last frame t = %.2f ns)" % (_nm(stem), t[-1] * ns), fontsize=10)
    ax = axs[1]
    with np.errstate(all="ignore"):
        ax.semilogy(t * ns, (hi - lo) / mn, "-o", ms=2, color="tab:red", label="(max−min)/mean, 3D")
        ax.semilogy(t * ns, np.nanstd(rad, 1) / mn, "-", color="tab:orange", label="rms/mean, 3D")
        for w, c in (("xy", "tab:blue"), ("dg", "tab:green")):
            Rt = np.array([f["Rth_" + w] for f in fr])
            ax.semilogy(t * ns, (np.nanmax(Rt, 1) - np.nanmin(Rt, 1)) / np.nanmean(Rt, 1), "--", color=c, lw=0.9,
                        label="(max−min)/mean, %s plane" % ("z=0" if w == "xy" else "x=y"))
    ax.axhline(A.meta["dx_finest"] / R0, color="0.5", ls=":", lw=0.8)
    ax.text(t[0] * ns, A.meta["dx_finest"] / R0, " dx/R0", fontsize=7, va="bottom", color="0.4")
    ax.set_ylabel("shape spread"); ax.legend(fontsize=7, loc="upper left")
    ax = axs[2]
    if th is not None and len(th["time"]) and "interface_area" in th:
        with np.errstate(all="ignore"):
            Af = 8 * th["interface_area"]; Rv = rvol_thermo(th)
            ex = Af / (4 * np.pi * Rv ** 2)
            ex = ex / ex[np.argmax(np.isfinite(ex) & (ex > 0))]
        ax.plot(th["time"] * ns, ex - 1, "-", color="k", lw=0.9)
        ax.set_ylabel("area / (4πR_vol²) − 1\n(normalised to t=0)", fontsize=8)
    n_str = col(fr, "n_stray")
    ax2 = ax.twinx(); ax2.plot(t * ns, n_str, "s", color="tab:brown", ms=2.5)
    ax2.set_ylabel("stray gas cells", color="tab:brown", fontsize=8)
    ax.set_xlabel("t [ns]")
    for a in axs:
        a.set_xlim(0, max(t[-1], th["time"][-1] if (th is not None and len(th["time"])) else 0) * ns * 1.02)
    save(fig, stem + "_dbg_radius.png")


def plot_contours(A, fr, stem, n_last):
    R0 = A.R0; ns = 1e9
    t = col(fr, "t")
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 8.6))
    for row, w in enumerate(("xy", "dg")):
        lab = ("x", "y") if w == "xy" else (r"$\sqrt{x^2+y^2}$ along (1,1,0)", "z")
        ax = axs[row, 0]
        for i, f in enumerate(fr):
            s1, s2, img = f["img_" + w]
            _contour(ax, s1 / R0, s2 / R0, img.T, [0.5], colors=[cmap(i / max(len(fr) - 1, 1))], linewidths=0.6)
        ax.set_aspect("equal"); ax.set_xlabel(lab[0] + r"$/R_0$"); ax.set_ylabel(lab[1] + r"$/R_0$")
        ax.set_title("η=0.5, %s plane, all %d frames" % ("z = 0" if w == "xy" else "x = y", len(fr)), fontsize=9)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(t[0] * ns, t[-1] * ns))
        fig.colorbar(sm, ax=ax, label="t [ns]", fraction=0.046)
        ax = axs[row, 1]
        last = fr[-n_last:]
        for i, f in enumerate(last):
            s1, s2, img = f["img_" + w]
            _contour(ax, s1 / R0, s2 / R0, img.T, [0.5], colors=[plt.get_cmap("plasma")(i / max(len(last) - 1, 1))],
                       linewidths=0.9)
            ax.plot([], [], color=plt.get_cmap("plasma")(i / max(len(last) - 1, 1)), label="%.2f ns" % (f["t"] * ns))
        if last:
            s1, s2, img = last[-1]["img_" + w]
            _contour(ax, s1 / R0, s2 / R0, img.T, [0.1, 0.9], colors="0.5", linewidths=0.5, linestyles=":")
            Rm = np.nanmax(last[-1]["Rth_" + w]) / R0
            lim = 1.25 * Rm if np.isfinite(Rm) else 1.3
            ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_aspect("equal"); ax.legend(fontsize=6.5, loc="upper right")
        ax.set_title("last %d frames (dotted: η=0.1/0.9 at the last)" % len(last), fontsize=9)
        ax = axs[row, 2]
        thg, _ = _plane_dirs(w)
        Rt = np.array([f["Rth_" + w] for f in fr])
        with np.errstate(all="ignore"):
            dev = Rt / np.nanmean(Rt, 1, keepdims=True) - 1
        v = np.nanmax(np.abs(dev)) if np.isfinite(dev).any() else 1e-3
        pc = ax.pcolormesh(np.degrees(thg), t * ns, dev, cmap="RdBu_r", vmin=-v, vmax=v, shading="nearest")
        fig.colorbar(pc, ax=ax, label=r"$R(\theta)/\langle R\rangle - 1$", fraction=0.046)
        ax.set_xlabel(r"$\theta$ [deg] from %s" % ("+x" if w == "xy" else "+z (90 = diagonal)"))
        ax.set_ylabel("t [ns]"); ax.set_title("radius deviation kymograph", fontsize=9)
    fig.suptitle("%s — interface contours" % _nm(stem), fontsize=11)
    fig.tight_layout()
    save(fig, stem + "_dbg_contours.png")


def plot_modes(A, fr, stem):
    ns = 1e9
    th, _ = _plane_dirs("xy")
    t = col(fr, "t")
    spec = []
    for f in fr:
        n, a = fourier(th, f["Rth_xy"])
        spec.append(a if a is not None else None)
    good = [s for s in spec if s is not None]
    if not good:
        return
    nmax = min(len(good[0]) - 1, 80)
    S = np.array([s[:nmax + 1] if s is not None else np.full(nmax + 1, np.nan) for s in spec])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axs[0]
    show = [n for n in range(2, nmax + 1, 2)]
    cm = plt.get_cmap("turbo")
    for k, n in enumerate(show):
        ax.semilogy(t * ns, S[:, n], "-", color=cm(k / max(len(show) - 1, 1)), lw=0.8 if n > 12 else 1.4,
                    label="n=%d" % n if (n <= 12 or n % 8 == 0) else None)
    ax.set_xlabel("t [ns]"); ax.set_ylabel(r"$|a_n|/a_0$  (z = 0 contour)")
    ax.legend(fontsize=6.5, ncol=3); ax.set_title("shape-mode amplitudes vs time", fontsize=9)
    ax = axs[1]
    Rm = fr[-1]["R_mean"]; dx = fr[-1]["dx"]
    for f, a, ls in ((fr[-1], S[-1], "-"), (fr[len(fr) // 2], S[len(fr) // 2], "--"), (fr[0], S[0], ":")):
        ax.semilogy(np.arange(0, nmax + 1, 2), a[::2], ls, marker=".", label="t = %.2f ns" % (f["t"] * ns))
    for k, lab in ((4, "λ=4dx"), (2, "λ=2dx")):
        ng = 2 * np.pi * Rm / (k * dx)
        if ng < nmax:
            ax.axvline(ng, color="0.5", ls=":", lw=0.8); ax.text(ng, ax.get_ylim()[1], lab, fontsize=7, va="top")
    ax.set_xlabel("n  (cos nθ, z = 0 plane)"); ax.legend(fontsize=7); ax.set_title("spectrum", fontsize=9)
    fig.suptitle("%s — wrinkle modes" % _nm(stem), fontsize=11)
    fig.tight_layout()
    save(fig, stem + "_dbg_modes.png")


def plot_health(A, fr, th, log, stem):
    ns = 1e9; R0 = A.R0
    t = col(fr, "t")
    fig, axs = plt.subplots(4, 3, figsize=(14, 13), sharex=True)
    axs = axs.ravel()
    k = 0

    def nxt():
        nonlocal k
        a = axs[k]; k += 1; return a
    ax = nxt()
    ax.plot(t * ns, gcol(fr, "pressure", "max"), "-o", ms=2, color="tab:red", label="p max")
    ax.plot(t * ns, gcol(fr, "pressure", "min"), "-o", ms=2, color="tab:blue", label="p min")
    ax.set_yscale("symlog", linthresh=1e4); ax.set_ylabel("p [Pa]"); ax.legend(fontsize=7)
    ax = nxt()
    ax.semilogy(t * ns, gcol(fr, "density", "max"), "-o", ms=2, color="tab:red", label="ρ max")
    ax.semilogy(t * ns, np.abs(gcol(fr, "density", "min")), "-o", ms=2, color="tab:blue", label="|ρ min|")
    ax.semilogy(t * ns, np.abs(gcol(fr, "rho_eta1", "min")) + 1e-30, ":", color="tab:green", label="|min rho_eta1| (gas)")
    ax.set_ylabel("density"); ax.legend(fontsize=7)
    ax = nxt()
    ax.plot(t * ns, gcol(fr, "T", "max"), "-o", ms=2, color="tab:red", label="T max")
    ax.plot(t * ns, gcol(fr, "T", "min"), "-o", ms=2, color="tab:blue", label="T min")
    ax.set_yscale("symlog", linthresh=1.0); ax.set_ylabel("T [K]"); ax.legend(fontsize=7)
    ax = nxt()
    ax.semilogy(t * ns, np.maximum(-gcol(fr, "eta", "min"), 1e-16), "-o", ms=2, color="tab:blue", label="−min(η)")
    ax.semilogy(t * ns, np.maximum(gcol(fr, "eta", "max") - 1, 1e-16), "-o", ms=2, color="tab:red", label="max(η)−1")
    ax.set_ylabel("η over/undershoot"); ax.legend(fontsize=7)
    ax = nxt()
    ax.semilogy(t * ns, gcol(fr, "umag", "max"), "-o", ms=2, color="k", label="|u| max (domain)")
    ax.semilogy(t * ns, col(fr, "umax_gas"), "-", color="tab:orange", label="|u| max in gas (η<0.05)")
    tk, _, Vk = A.km
    ax.semilogy(tk * ns, np.abs(Vk), "--", color="0.5", lw=0.8, label="|dR/dt| KM")
    ax.set_ylabel("|u| [m/s]"); ax.legend(fontsize=7)
    ax = nxt()
    ax.plot(t * ns, col(fr, "nbad"), "-o", ms=2, color="tab:red", label="non-finite cells")
    ax.plot(t * ns, col(fr, "n_stray"), "-s", ms=2, color="tab:brown", label="stray gas cells")
    ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("count"); ax.legend(fontsize=7)
    ax = nxt()
    Rm = col(fr, "R_mean")
    for i, (c, lab) in enumerate((("tab:blue", "min"), ("k", "median"), ("tab:red", "max"))):
        ax.plot(t * ns, col(fr, "band_kappaAvg", i) * Rm, "-", color=c, label="kappaAvg*R_mean, band " + lab)
    ax.set_yscale("symlog", linthresh=1); ax.set_ylabel("kappaAvg * R_mean"); ax.legend(fontsize=7)
    ax = nxt()
    for i, (c, lab) in enumerate((("tab:blue", "min"), ("k", "median"), ("tab:red", "max"))):
        ax.plot(t * ns, col(fr, "band_Gamma", i), "-", color=c, label="Γ band " + lab)
    ax.set_ylabel("Γ (band)"); ax.legend(fontsize=7)
    ax = nxt()
    ax.plot(t * ns, col(fr, "level"), "-o", ms=2, color="k", label="finest level present")
    ax.set_ylabel("level"); ax.set_ylim(-0.5, A.meta["max_level"] + 0.5)
    ax2 = ax.twinx()
    ax2.semilogy(t * ns, [f["ncell_lev"][-1] for f in fr], "-", color="tab:green", label="cells on finest")
    ax2.semilogy(t * ns, col(fr, "band_cells"), "--", color="tab:purple", label="band cells")
    ax2.legend(fontsize=7, loc="lower right"); ax.legend(fontsize=7, loc="upper left")
    ax = nxt()
    if log is not None and len(log["t"]):
        ax.semilogy(log["t"] * ns, log["dt"], "-", color="k", lw=0.8, label="dt (stdout)")
        ax.set_ylabel("dt [s]")
        ax2 = ax.twinx(); ax2.plot(log["t"] * ns, log["step"], ":", color="tab:blue"); ax2.set_ylabel("step", color="tab:blue")
        ax.legend(fontsize=7)
    else:
        st = col(fr, "step")
        ax.text(0.5, 0.5, "no stdout log\n(pass --log02/--log04)", ha="center", transform=ax.transAxes)
        if len(t) > 1:
            ax.semilogy(t[1:] * ns, np.diff(t) / np.maximum(np.diff(st), 1), "-o", ms=2, label="mean dt between frames")
            ax.legend(fontsize=7)
    ax = nxt()
    if th is not None and len(th["time"]):
        ke = 8 * th["kinetic_energy"]
        ax.semilogy(th["time"] * ns, np.where(ke > 0, ke, np.nan), "-", color="k", lw=0.8, label="KE (octant x8)")
        if (ke > 0).any():
            ax.set_ylim(max(ke[ke > 0].min(), 1e-6 * ke.max()), 2 * ke.max())
        ax.set_ylabel("KE [J]"); ax.legend(fontsize=7)
    ax = nxt()
    if th is not None and len(th["time"]):
        with np.errstate(all="ignore"):
            pg = th["gas_pressure_int"] / th["gas_volume"]
        ax.semilogy(th["time"] * ns, np.abs(pg), "-", color="tab:red", lw=0.8, label="mean gas p (thermo.dat)")
        Rv = rvol_thermo(th)
        pk = A.gas.p_g0 * (A.R0 / np.maximum(Rv, 1e-30)) ** (3 * A.gas.kappa_g)
        ax.semilogy(th["time"] * ns, pk, "--", color="0.5", lw=0.8, label="adiabatic p_g0 (R0/R_vol)^(3 gamma)")
        ax.set_ylabel("gas pressure [Pa]"); ax.legend(fontsize=7)
    tend = max(t[-1], log["t"][-1] if (log is not None and len(log["t"])) else 0,
               th["time"][-1] if (th is not None and len(th["time"])) else 0)
    for a in axs:
        a.set_xlim(0, tend * ns * 1.02)
        a.axvline(t[-1] * ns, color="tab:red", lw=0.6, ls="--")
    for a in axs[-3:]:
        a.set_xlabel("t [ns]")
    fig.suptitle("%s — run health (red dashed: last plotfile)" % _nm(stem), fontsize=11)
    fig.tight_layout()
    save(fig, stem + "_dbg_health.png")


def plot_fields(A, fr, stem):
    R0 = A.R0; ns = 1e9
    rm = col(fr, "R_mean")
    picks = []
    if np.isfinite(rm).any():
        picks.append(("R_min", int(np.nanargmin(rm))))
    picks.append(("last", len(fr) - 1))
    if len(picks) == 2 and picks[0][1] == picks[1][1]:
        picks = picks[1:]
    nf = len(SLICE_FIELDS)
    fig, axs = plt.subplots(2 * len(picks), nf // 2, figsize=(15, 6.6 * len(picks)))
    axs = np.atleast_2d(axs)
    for pi, (name, i) in enumerate(picks):
        f = fr[i]; sl = f["slice"]; dx = f["slice_dx"]
        for j, fld in enumerate(SLICE_FIELDS):
            ax = axs[2 * pi + j // (nf // 2), j % (nf // 2)]
            if fld not in sl or min(np.shape(sl[fld]) or (0,)) < 2:
                ax.set_axis_off(); continue
            v = sl[fld].copy()
            if fld == "kappaAvg":
                v = v * f["R_mean"]
                band = (sl["eta"] > 0.05) & (sl["eta"] < 0.95)
                lim = np.nanpercentile(np.abs(v[band]), 99) if band.any() else 1
                kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
                v = np.where(band, v, np.nan); lab = "kappaAvg*R_mean (band)"
            elif fld == "pressure":
                kw = dict(cmap="jet", norm=SA.SymLogNorm(linthresh=1e4, vmin=np.nanmin(v), vmax=np.nanmax(v)))
                lab = "p"
            else:
                kw = dict(cmap="viridis"); lab = fld
            n = v.shape[0]
            ext = [0, n * dx / R0, 0, v.shape[1] * dx / R0]
            im = ax.imshow(v.T, origin="lower", extent=ext, interpolation="nearest", **kw)
            fig.colorbar(im, ax=ax, fraction=0.046)
            s = (np.arange(n) + 0.5) * dx / R0
            _contour(ax, s, (np.arange(v.shape[1]) + 0.5) * dx / R0, sl["eta"].T, [0.5], colors="w", linewidths=0.7)
            for key, mk, c in (("pressure", "v", "b"), ("umag", "D", "m")):
                g = f["glob"].get(key, {})
                L = g.get("argmin" if key == "pressure" else "argmax")
                if L and abs(L["z"]) < 3 * dx and L["x"] / R0 < ext[1] and L["y"] / R0 < ext[3]:
                    ax.plot(L["x"] / R0, L["y"] / R0, mk, color=c, ms=7, mec="w")
            ax.set_title("%s @ %s, t = %.2f ns" % (lab, name, f["t"] * ns), fontsize=8)
    fig.suptitle("%s — z = 0 slices (white: η=0.5; triangle: p min, diamond: |u| max if on this plane)" % _nm(stem), fontsize=10)
    fig.tight_layout()
    save(fig, stem + "_dbg_fields.png")


# =========================================================================================== #
#  report
# =========================================================================================== #
def first(t, mask):
    i = np.where(mask)[0]
    return ("t = %.3f ns (frame %d)" % (t[i[0]] * 1e9, i[0])) if len(i) else "never"


def report(A, fr, th, log, logpath, errors, plotdir, stem):
    ns = 1e9; R0 = A.R0
    L = []
    P = L.append
    t = col(fr, "t")
    P("=" * 100)
    P("%s   plotdir %s" % (stem, plotdir))
    P("deck %s" % A.meta["input_path"])
    P("R0 %.3g  dx_finest %.4g (R0/dx %.1f)  eps %.3g  max_level %d  stop_time %.4g ns" %
      (R0, A.meta["dx_finest"], R0 / A.meta["dx_finest"], A.meta["epsilon"], A.meta["max_level"],
       A.meta["stop_time"] * ns))
    P("frames read %d, last plotfile %s at t = %.3f ns (%.1f%% of stop_time)" %
      (len(fr), fr[-1]["name"], t[-1] * ns, 100 * t[-1] / A.meta["stop_time"]))
    for e in errors:
        P("  UNREADABLE plotfile: " + e)
    if th is not None and len(th["time"]):
        P("thermo.dat last t = %.3f ns (%d rows)" % (th["time"][-1] * ns, len(th["time"])))
    if log is None:
        P("stdout log: none found (pass --log02 / --log04)")
    else:
        P("stdout log: %s" % ", ".join(log["files"]))
        if len(log["t"]):
            P("  last STEP %d  t = %.4f ns  dt = %.3g  (min dt %.3g at t = %.3f ns; dt last 20 steps: %s)" %
              (log["step"][-1], log["t"][-1] * ns, log["dt"][-1], log["dt"].min(),
               log["t"][np.argmin(log["dt"])] * ns, " ".join("%.2g" % d for d in log["dt"][-20:])))
        P("  error-like lines (%d, last 25):" % len(log["errors"]))
        for e in log["errors"][-25:]:
            P("    " + e)
        P("  tail:")
        for e in log["tail"][-30:]:
            P("    " + e)
    # ---- KM comparison -----------------------------------------------------------------------
    tk, Rk, _ = A.km
    rm = col(fr, "R_mean")
    i = int(np.nanargmin(rm)) if np.isfinite(rm).any() else 0
    j = int(np.argmin(Rk[tk <= max(t[-1], tk[1])])) if len(Rk) else 0
    P("R_min sim (mean of 400 dirs) %.4f R0 at %.3f ns   |  KM over the same window %.4f R0 at %.3f ns" %
      (rm[i] / R0, t[i] * ns, Rk[j] / R0, tk[j] * ns))
    # ---- first anomaly per check ---------------------------------------------------------------
    rad = np.array([f["rad"] for f in fr])
    with np.errstate(all="ignore"):
        spread = (np.nanmax(rad, 1) - np.nanmin(rad, 1)) / np.nanmean(rad, 1)
        emin, emax = gcol(fr, "eta", "min"), gcol(fr, "eta", "max")
        rgas = gcol(fr, "rho_eta1", "min")
    P("first anomaly:")
    P("  non-finite cells         %s" % first(t, col(fr, "nbad") > 0))
    P("  eta < -1e-3 or > 1+1e-3  %s" % first(t, (emin < -1e-3) | (emax > 1 + 1e-3)))
    P("  rho_eta1 (gas) < -1e-12  %s" % first(t, rgas < -1e-12))
    P("  T < 0                    %s" % first(t, gcol(fr, "T", "min") < 0))
    P("  shape spread > dx/R0     %s" % first(t, spread > A.meta["dx_finest"] / R0))
    P("  shape spread > 5%%        %s" % first(t, spread > 0.05))
    P("  stray gas cells > 0      %s" % first(t, col(fr, "n_stray") > 0))
    P("  finest level dropped     %s" % first(t, col(fr, "level") < A.meta["max_level"]))
    # ---- last frames table --------------------------------------------------------------------
    P("last frames:")
    P("  %9s %7s %7s %7s %7s %7s %8s %10s %10s %9s %9s %9s %6s %6s" %
      ("t[ns]", "Rmean", "Rx", "Ry", "Rz", "Rdiag", "spread", "p_min", "p_max", "|u|max", "|u|gas", "kR max",
       "nbad", "stray"))
    for f, sp in list(zip(fr, spread))[-10:]:
        g = f["glob"]
        P("  %9.3f %7.4f %7.4f %7.4f %7.4f %7.4f %8.4f %10.3e %10.3e %9.3g %9.3g %9.3g %6d %6d" %
          (f["t"] * ns, f["R_mean"] / R0, *(f["R_axes"] / R0), f["R_diag"] / R0, sp,
           g["pressure"]["min"], g["pressure"]["max"], g["umag"]["max"], f["umax_gas"],
           (f.get("band_kappaAvg", (np.nan,) * 3)[2]) * f["R_mean"], f["nbad"], f["n_stray"]))
    # ---- where things sit at the last frame -----------------------------------------------------
    f = fr[-1]; g = f["glob"]
    P("extreme locations at the last frame (r/R0, angle from +z [deg], AMR level):")

    def w(lab, Lc):
        if Lc:
            P("  %-14s r/R0 %6.3f  theta %5.1f  level %d   (x,y,z)/R0 = (%.3f, %.3f, %.3f)" %
              (lab, Lc["r_R0"], Lc["theta_deg"], Lc["level"], Lc["x"] / R0, Lc["y"] / R0, Lc["z"] / R0))
    w("p min", g["pressure"].get("argmin")); w("p max", g["pressure"].get("argmax"))
    w("|u| max", g["umag"].get("argmax")); w("rho min", g["density"].get("argmin"))
    w("T min", g["T"].get("argmin") if "T" in g else None); w("T max", g["T"].get("argmax") if "T" in g else None)
    w("eta min", g["eta"].get("argmin")); w("eta max", g["eta"].get("argmax"))
    w("|kappa| max", f.get("kappa_argmax"))
    for k, Lc in enumerate(f["bad_first"]):
        w("non-finite #%d" % k, Lc)
    P("  bubble mean radius at the last frame %.4f R0; stray gas out to r = %.3f R0" %
      (f["R_mean"] / R0, f["stray_rmax_R0"]))
    P("  domain extrema at the last frame:")
    for k, v in sorted(g.items()):
        P("    %-15s min %12.4e  max %12.4e  non-finite %d" % (k, v["min"], v["max"], v["nbad"]))
    txt = "\n".join(L)
    with open(stem + "_dbg_report.txt", "w") as fh:
        fh.write(txt + "\n")
    print(txt)
    print("  wrote", stem + "_dbg_report.txt")


def overview(res, out_dir):
    fig, axs = plt.subplots(1, len(res), figsize=(6.2 * len(res), 4.2), squeeze=False)
    for ax, (key, (A, fr, th, log)) in zip(axs[0], res.items()):
        ns = 1e9; R0 = A.R0
        tk, Rk, _ = A.km
        ax.plot(tk * ns, Rk / R0, "-", color="0.25", label="Coated KM")
        t = col(fr, "t")
        rad = np.array([f["rad"] for f in fr])
        with np.errstate(all="ignore"):
            ax.fill_between(t * ns, np.nanmin(rad, 1) / R0, np.nanmax(rad, 1) / R0, color="tab:red", alpha=0.2, lw=0)
            ax.plot(t * ns, np.nanmean(rad, 1) / R0, "-o", ms=2, color="tab:red", label="sim (mean, min–max)")
        ax.axvline(t[-1] * ns, color="tab:red", ls="--", lw=0.8, label="last plotfile")
        if log is not None and len(log["t"]):
            ax.axvline(log["t"][-1] * ns, color="k", ls=":", lw=0.8, label="last logged step")
        ax.set_xlim(0, A.meta["stop_time"] * ns)
        ax.set_xlabel("t [ns]"); ax.set_ylabel(r"$R/R_0$"); ax.set_title(RUNS[key]["stem"], fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, "02_04_dbg_overview.png"))


# =========================================================================================== #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--o2", default=None, help="02 plotfile dir (default: deck plot_file)")
    ap.add_argument("--o4", default=None, help="04 plotfile dir (default: deck plot_file)")
    ap.add_argument("--log02", default=None, help="02 slurm stdout (flame_<id>_stdout)")
    ap.add_argument("--log04", default=None, help="04 slurm stdout")
    ap.add_argument("--only", choices=("02", "04"), default=None)
    ap.add_argument("--nproc", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--last", type=int, default=8, help="frames in the 'last frames' contour panel")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "Images"))
    ap.add_argument("--refresh", action="store_true", help="re-read every plotfile (ignore the cache)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    for key, cfg in RUNS.items():
        if a.only and key != a.only:
            continue
        deck = os.path.normpath(os.path.join(_HERE, "..", cfg["deck"]))
        A = Sch20Analysis(deck, "", case=cfg["case"], coated=True, stem=cfg["stem"] + "_dbg", out_dir=a.out_dir,
                          nproc=a.nproc, drive_gain=cfg["gain"], cache=False, verbose=True)
        plotdir = getattr(a, "o" + key[1]) or A.meta["plot_file"]
        print("\n==== %s  (%s)" % (cfg["stem"], plotdir))
        if not os.path.isdir(plotdir):
            print("  plotfile dir not found -- pass --o%s" % key[1]); continue
        stem = os.path.join(a.out_dir, cfg["stem"])
        fr, errors = load_frames(plotdir, A.R0, cfg["half_R0"], stem + "_dbg_cache.pkl", a.nproc, a.refresh)
        if not fr:
            print("  no readable plotfiles"); continue
        th = read_thermo(plotdir)
        logpath = find_log(A.meta, getattr(a, "log" + key))
        log = read_log(logpath)
        for fn, args in ((plot_radius, (A, fr, th, stem)), (plot_contours, (A, fr, stem, a.last)),
                         (plot_modes, (A, fr, stem)), (plot_health, (A, fr, th, log, stem)),
                         (plot_fields, (A, fr, stem))):
            try:
                fn(*args)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print("  [%s] FAILED: %s" % (fn.__name__, exc))
        report(A, fr, th, log, logpath, errors, plotdir, stem)
        res[key] = (A, fr, th, log)
    if res:
        overview(res, a.out_dir)


if __name__ == "__main__":
    main()
