#!/usr/bin/env python3
"""Axis-jet diagnostic for Sch20 collapse runs (Lim vs NoLim, 2026-10-07).

The 10_6 INCLINE comparison showed the limiter run (THINC + MUSCL + eta_consistent_advect)
collapsing rounder and deeper than WENO3 + donor (R_min/R0 0.040 vs 0.079; KM 0.031), then
throwing jets along the six coordinate axes on the rebound (eta = 0.5 radius 0.46 R0 within
15 deg of an axis vs 0.17 R0 on the body diagonal; R_vol 10-18 % above KM).  This script
reads the plotfiles around collapse and answers:

  1. Is the jet a narrow column of cells on the axis line (a symmetry-corner artifact: width in
     CELLS stays fixed as it grows) or a cone (grid-orientation instability: width grows with r)?
     -> front_map: eta = 0.5 front position along x for every (j, k) cell column near the axis.
  2. Where does the rebound gas volume sit?  R_vol = sum (1 - eta) dV counts mixture/stray cells,
     so it is split into the connected gas core, the rest of the near field, and stray gas.
  3. How do eta, p, u_r, gas density and T differ along the axis, an in-plane direction, a
     near-axis off-plane direction and the diagonals as the jet forms?

Usage (on INCLINE, from this directory):
    python diagnose_axis_jets.py \
        /mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Collapsing_Large_3D_LIM \
        /mmfs1/home/ttryon/flames/bin/tests/FlowRayleighPlesset/output_Sch20_Collapsing_Large_3D_NOLIM \
        --labels Lim NoLim --input ../Sch20_Collapsing_Neumann_Large_3D --nproc 16

Options: --tmin/--tmax time window [s] (default 2.55e-4 .. 3.40e-4, collapse is at ~2.69e-4),
--stride N (every Nth plotfile), --out DIR (default ./Images/axis_jets), --half H (fine box
half-width in R0, default 0.6).  Per-run data are cached in OUT/axisjet_<label>.pkl; rerunning
only reads new plotfiles.  Send back the PNGs and the printed table (the pkl files are small).
"""
import argparse
import os
import pickle
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import sch20_analysis as SA   # noqa: E402  (geometry, sampling and pool helpers)

DIRS = {   # name -> unit vector (octant: all components >= 0)
    "axis (100)":        np.array([1.0, 0.0, 0.0]),
    "in-plane 20deg":    np.array([np.cos(np.radians(20)), np.sin(np.radians(20)), 0.0]),
    "off-plane 10deg":   np.array([np.cos(np.radians(10)), np.sin(np.radians(10)) / np.sqrt(2),
                                   np.sin(np.radians(10)) / np.sqrt(2)]),
    "face diag (110)":   np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
    "body diag (111)":   np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
}
DCOL = {"axis (100)": "#c0392b", "in-plane 20deg": "#e67e22", "off-plane 10deg": "#8e44ad",
        "face diag (110)": "#2980b9", "body diag (111)": "#27ae60"}


def _time_of(pf):
    """Plotfile time from the Header (no yt), so the window can be applied before loading."""
    with open(os.path.join(pf, "Header")) as fh:
        fh.readline()
        n = int(fh.readline())
        for _ in range(n):
            fh.readline()
        fh.readline()
        return float(fh.readline())


# ------------------------------------------------------------------------------------------- #
def _frame(args):
    pf, R0, half = args
    try:
        yt = SA._yt()
        ds = yt.load(pf)
        dim, dle, dre, sym, cen, dx = SA._geom(ds)
        if dim != 3:
            return {"error": "%s: 3D only" % pf}
        has = lambda f: ("boxlib", f) in ds.field_list
        o = {"t": float(ds.current_time), "name": os.path.basename(pf), "mtime": os.path.getmtime(pf),
             "dx": dx, "sym": sym.copy()}
        # ---- fine cube around the centre (from the centre outward on mirrored axes) ----------
        m = int(np.ceil(half * R0 / dx)) + 2
        snap = lambda v, d: dle[d] + np.floor((v - dle[d]) / dx + 1e-9) * dx
        lo = [cen[d] if sym[d] else max(snap(cen[d] - m * dx, d), dle[d]) for d in range(3)]
        dims = [m if sym[d] else 2 * m for d in range(3)]
        cg = ds.covering_grid(level=ds.index.max_level, left_edge=lo, dims=dims)
        F = {k: np.asarray(cg[("boxlib", k)], float)
             for k in ("eta", "pressure", "velocityx", "velocityy", "velocityz", "rho_eta1", "T") if has(k)}
        off = np.array(lo) - cen
        xc = [off[d] + (np.arange(dims[d]) + 0.5) * dx for d in range(3)]
        X, Y, Z = np.meshgrid(*xc, indexing="ij")
        r = np.sqrt(X * X + Y * Y + Z * Z)
        eta = F["eta"]
        ur = (F["velocityx"] * X + F["velocityy"] * Y + F["velocityz"] * Z) / np.maximum(r, 1e-30)
        ag = np.clip(1.0 - eta, 0.0, 1.0)
        rhog = F["rho_eta1"] / np.maximum(ag, 1e-12) if "rho_eta1" in F else None
        # the octant cube starts at the centre, a full axis has the centre in the middle: index of
        # the first cell on the + side of each axis
        i0 = [0 if sym[d] else m for d in range(3)]

        # ---- 1. eta = 0.5 front along +x for every (j, k) column near the axis ----------------
        nc = min(40, m - 1)
        E = eta[i0[0]:, i0[1]:i0[1] + nc, i0[2]:i0[2] + nc]
        s = (np.arange(E.shape[0]) + 0.5) * dx
        front = np.full((nc, nc), np.nan)
        for j in range(nc):
            for k in range(nc):
                front[j, k] = SA._first_crossing(s, E[:, j, k])
        o["front_x"] = front                 # [m]; column (j, k) sits at y = (j+.5)dx, z = (k+.5)dx

        # ---- 2. gas volume split -------------------------------------------------------------
        from scipy import ndimage
        dV = dx ** 3
        fac_cube = 2 ** int(np.sum(sym))     # mirrored copies; full axes are inside the cube
        lab, _ = ndimage.label(eta < 0.5)
        core_id = lab[i0[0], i0[1], i0[2]] if eta[i0[0], i0[1], i0[2]] < 0.5 else 0
        core = (lab == core_id) if core_id else np.zeros_like(eta, bool)
        core_d = ndimage.binary_dilation(core, iterations=3) if core.any() else core
        o["V_core"] = float(np.sum(ag[core_d]) * dV) * fac_cube          # core + its 3-cell band
        o["V_cube"] = float(np.sum(ag) * dV) * fac_cube                  # everything in the cube
        o["V_sharp"] = float(np.sum(eta < 0.5) * dV) * fac_cube          # eta < 0.5 cells only
        o["n_islands"] = int(lab.max() - (1 if core_id else 0))          # detached eta<0.5 blobs
        mix = (eta > 0.01) & (eta < 0.99)
        o["V_mix_cells"] = float(np.sum(mix) * dV) * fac_cube
        # whole-domain gas volume (all levels) in a 3 R0 box, as sch20_analysis does
        h = 3.0 * R0
        blo = np.maximum(np.where(sym, cen, cen - h), dle); bhi = np.minimum(cen + h, dre)
        reg = ds.box(ds.arr(blo, "code_length"), ds.arr(bhi, "code_length"))
        er = np.asarray(reg["boxlib", "eta"], float); vr = np.asarray(reg["index", "cell_volume"], float)
        o["V_total"] = float(np.sum(np.clip(1 - er, 0, 1) * vr)) * 2 ** int(np.sum(sym))
        if rhog is not None:
            o["gas_mass_cube"] = float(np.sum(F["rho_eta1"]) * dV) * fac_cube
            # gas-fraction-weighted over the connected core (eta < 0.05 cells vanish at collapse)
            w = ag * core
            sw = np.sum(w)
            o["rho_core"] = float(np.sum(F["rho_eta1"] * core) / sw) if sw > 0 else np.nan
            o["p_core"] = float(np.sum(F["pressure"] * w) / sw) if sw > 0 else np.nan
            o["T_core"] = float(np.sum(F["T"] * w) / sw) if (sw > 0 and "T" in F) else np.nan

        # ---- 3. lines along the probe directions ---------------------------------------------
        sl = np.arange(0.0, half * R0, 0.5 * dx)
        o["line_s"] = sl
        flds = {"eta": eta, "p": F["pressure"], "ur": ur}
        if rhog is not None:
            flds["rho_g"] = np.where(eta < 0.5, rhog, np.nan)
        if "T" in F:
            flds["T"] = F["T"]
        lines = {}
        for nm, u in DIRS.items():
            pts = [np.abs(sl * u[q]) if sym[q] else sl * u[q] for q in range(3)]
            lines[nm] = {f: SA._sample(np.nan_to_num(v, nan=0.0), pts, dx, off) for f, v in flds.items()}
            lines[nm]["R"] = SA._first_crossing(sl, lines[nm]["eta"])
        o["lines"] = lines

        # ---- 4. slices: z = dx/2 plane (contains x and y axes) and the j == k diagonal plane ---
        kz = i0[2]
        o["slice_z0"] = {"eta": eta[:, :, kz].copy(), "p": F["pressure"][:, :, kz].copy(),
                         "ur": ur[:, :, kz].copy()}
        n = min(dims[0] - i0[0], dims[1] - i0[1], dims[2] - i0[2])
        jj = np.arange(n)
        o["slice_diag"] = {"eta": eta[i0[0]:, i0[1] + jj, i0[2] + jj].copy(),
                           "ur": ur[i0[0]:, i0[1] + jj, i0[2] + jj].copy()}
        o["slice_x0"] = xc[0]; o["slice_y0"] = xc[1]

        # ---- 5. shape in 400 directions (same measure as sch20_analysis) ----------------------
        dirs = SA._fib_dirs(400)
        o["dirs"] = dirs
        o["rad"] = SA._radii(eta, dx, off, sym, dirs, half * R0)
        ax6 = [np.eye(3)[d] * sg for d in range(3) for sg in (1, -1)]
        o["axis6"] = SA._radii(eta, dx, off, sym, ax6, half * R0)   # +x -x +y -y +z -z
        return o
    except Exception as exc:
        import traceback
        return {"error": "%s: %s\n%s" % (os.path.basename(pf), exc, traceback.format_exc())}


# ------------------------------------------------------------------------------------------- #
def load_run(plotdir, label, R0, args):
    cache = os.path.join(args.out, "axisjet_%s.pkl" % label)
    done = {}
    if os.path.isfile(cache):
        with open(cache, "rb") as fh:
            c = pickle.load(fh)
        if c.get("key") == (args.half, R0):
            done = c["frames"]
    fs = SA._plotfiles(plotdir)
    fs = [p for p in fs if args.tmin <= _time_of(p) <= args.tmax][::args.stride]
    todo = [p for p in fs if os.path.basename(p) not in done
            or abs(done[os.path.basename(p)]["mtime"] - os.path.getmtime(p)) > 1e-6]
    print("[%s] %s: %d plotfiles in window, %d to read" % (label, plotdir, len(fs), len(todo)))
    for r in SA._pool_map(_frame, [(p, R0, args.half) for p in todo], args.nproc):
        if "error" in r:
            print("  skipped", r["error"])
        else:
            done[r["name"]] = r
    with open(cache, "wb") as fh:
        pickle.dump({"key": (args.half, R0), "frames": done}, fh)
    return sorted([done[os.path.basename(p)] for p in fs if os.path.basename(p) in done], key=lambda d: d["t"])


def _ang(dirs):
    return np.degrees(np.arccos(np.clip(np.abs(dirs).max(1), 0, 1)))


def _rv(V):
    return (3.0 * np.asarray(V) / (4.0 * np.pi)) ** (1.0 / 3.0)


def front_excess(f):
    """Front along +x minus a sphere of the body-diagonal radius, per (j, k) column [cells]."""
    dx = f["dx"]
    Rr = f["lines"]["body diag (111)"]["R"]
    n = f["front_x"].shape[0]
    yc = (np.arange(n) + 0.5) * dx
    sph = np.sqrt(np.maximum(Rr ** 2 - yc[:, None] ** 2 - yc[None, :] ** 2, 0.0))
    return (f["front_x"] - sph) / dx


def jet_width_cells(f):
    """Cells across the jet: columns in the k = 0 row whose excess over the body-diagonal sphere
    is more than half the on-axis excess (0 when the on-axis excess is under one cell)."""
    ex = front_excess(f)[:, 0]
    if not np.isfinite(ex[0]):
        return np.nan
    if ex[0] < 1.0:
        return 0.0
    return float(np.sum(ex > 0.5 * ex[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="+")
    ap.add_argument("--input", default=None, help="input deck: adds Keller-Miksis and t/t_c")
    ap.add_argument("--R0", type=float, default=0.02)
    ap.add_argument("--tmin", type=float, default=2.55e-4)
    ap.add_argument("--tmax", type=float, default=3.40e-4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--half", type=float, default=0.6)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(_HERE, "Images", "axis_jets"))
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p.rstrip("/")) for p in args.runs]
    os.makedirs(args.out, exist_ok=True)
    R0 = args.R0
    km = None
    if args.input:
        A = SA.Sch20Analysis(args.input, None, verbose=False, out_dir=args.out)
        R0, km, tc = A.R0, A.km, A.tc
    runs = {lab: load_run(p, lab, R0, args) for p, lab in zip(args.runs, labels)}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        sys.exit("no frames read")
    if km is None:   # t_c = time of the deepest R_total over all runs
        tc = min((min(fr, key=lambda f: f["V_total"])["t"] for fr in runs.values()))
    col = {lab: c for lab, c in zip(runs, ["#c0392b", "#2c3e50", "#16a085", "#8e44ad"])}

    # ---- table --------------------------------------------------------------------------------
    for lab, fr in runs.items():
        dx = fr[0]["dx"]
        V0 = 4.0 / 3.0 * np.pi * R0 ** 3
        print("\n== %s  (R0/dx %.1f, sym %s)" % (lab, R0 / dx, fr[0]["sym"]))
        print("  t/tc   Rtot  Rcore Rsharp  Rkm |  R_axis R_inpl R_offp R_110 R_111 | jet[cells] "
              "islands | ur@front axis/111   p_core    rho_core T_core")
        for f in fr:
            L = f["lines"]
            Rk = np.interp(f["t"], km[0], km[1]) / R0 if km is not None else np.nan
            def ur_front(nm):
                Rf = L[nm]["R"]
                return np.interp(Rf, fr[0]["line_s"], L[nm]["ur"]) if np.isfinite(Rf) else np.nan
            print("  %.3f %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f |  %5.1f  %4d | %7.1f %7.1f  %.2e %.2e %.0f"
                  % (f["t"] / tc, _rv(f["V_total"]) / R0, _rv(f["V_core"]) / R0, _rv(f["V_sharp"]) / R0, Rk,
                     *[L[nm]["R"] / R0 for nm in DIRS],
                     jet_width_cells(f), f["n_islands"],
                     ur_front("axis (100)"), ur_front("body diag (111)"),
                     f.get("p_core", np.nan), f.get("rho_core", np.nan), f.get("T_core", np.nan)))
        print("  six axis radii /R0 (+x -x +y -y +z -z) at the last frame:", np.round(fr[-1]["axis6"] / R0, 4))

    # ---- fig 1: gas volume split -----------------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for lab, fr in runs.items():
        t = np.array([f["t"] for f in fr]) / tc
        axs[0].plot(t, _rv([f["V_total"] for f in fr]) / R0, "-", c=col[lab], label=lab + r": $R_\mathrm{vol}$ (3 $R_0$ box)")
        axs[0].plot(t, _rv([f["V_core"] for f in fr]) / R0, "--", c=col[lab], label=lab + ": connected core")
        axs[0].plot(t, _rv([f["V_sharp"] for f in fr]) / R0, ":", c=col[lab], label=lab + r": $\eta<0.5$ cells")
        st = 1 - np.array([f["V_core"] / f["V_total"] for f in fr])
        axs[1].plot(t, 100 * st, c=col[lab], label=lab)
    if km is not None:
        axs[0].plot(km[0] / tc, km[1] / R0, "k-", lw=0.8, label="Keller–Miksis")
    t_all = np.concatenate([[f["t"] for f in fr] for fr in runs.values()]) / tc
    axs[0].set_xlim(t_all.min(), t_all.max()); axs[0].set_ylim(0, None)
    axs[0].set_xlabel(r"$t/t_c$"); axs[0].set_ylabel(r"$R/R_0$"); axs[0].legend(fontsize=7)
    axs[1].set_xlabel(r"$t/t_c$"); axs[1].set_ylabel("gas volume outside the core [%]"); axs[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "gas_volume_split.png"), dpi=150); plt.close(fig)

    # pick 6 frames: R_min of each run's total volume, then + 1, 2, 4, 8, 16 frames
    def picks(fr):
        i = int(np.argmin([f["V_total"] for f in fr]))
        return [min(i + d, len(fr) - 1) for d in (-2, 0, 1, 2, 4, 8, 16)]

    # ---- fig 2/3: lines (eta and u_r) ----------------------------------------------------------
    for fld, ylab in (("eta", r"$\eta$"), ("ur", r"$u_r$ [m/s]"), ("p", "p [Pa]")):
        fig, axs = plt.subplots(len(runs), 7, figsize=(20, 3.2 * len(runs)), squeeze=False, sharey="row")
        for r_, (lab, fr) in enumerate(runs.items()):
            s = fr[0]["line_s"] / R0
            for c_, k in enumerate(picks(fr)):
                a = axs[r_, c_]
                for nm in DIRS:
                    a.plot(s, fr[k]["lines"][nm][fld], c=DCOL[nm], lw=1.1, label=nm)
                a.set_title("%s  t/t_c=%.3f" % (lab, fr[k]["t"] / tc), fontsize=9)
                a.set_xlabel(r"$r/R_0$")
                if fld == "p":
                    a.set_yscale("symlog", linthresh=1e5)
            axs[r_, 0].set_ylabel(ylab)
        axs[0, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "lines_%s.png" % fld), dpi=130); plt.close(fig)

    # ---- fig 4: slices (eta on z = dx/2 and on the j = k diagonal plane) --------------------------
    for sname in ("slice_z0", "slice_diag"):
        fig, axs = plt.subplots(len(runs), 7, figsize=(20, 3.0 * len(runs)), squeeze=False)
        for r_, (lab, fr) in enumerate(runs.items()):
            dx = fr[0]["dx"]
            for c_, k in enumerate(picks(fr)):
                E = fr[k][sname]["eta"]
                ext = [0, E.shape[0], 0, E.shape[1]]
                a = axs[r_, c_]
                a.imshow(E.T, origin="lower", cmap="RdBu_r", vmin=0, vmax=1, extent=ext, interpolation="nearest")
                a.contour(E.T, [0.5], colors="k", linewidths=0.6, extent=ext)
                a.set_title("%s  t/t_c=%.3f" % (lab, fr[k]["t"] / tc), fontsize=9)
                a.set_xlabel("i (x cells)")
                a.set_ylabel("j (y cells)" if sname == "slice_z0" else "j = k (diag cells)")
        fig.suptitle({"slice_z0": r"$\eta$ on the z = dx/2 plane (contains the x and y axes; cell units)",
                      "slice_diag": r"$\eta$ on the plane through the x axis and (0,1,1) (cell units)"}[sname])
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "%s.png" % sname), dpi=130); plt.close(fig)

    # ---- fig 5: front map (eta = 0.5 x-front per (j, k) column minus the body-diagonal sphere) ----
    fig, axs = plt.subplots(len(runs), 7, figsize=(20, 3.0 * len(runs)), squeeze=False)
    for r_, (lab, fr) in enumerate(runs.items()):
        for c_, k in enumerate(picks(fr)):
            Ex = front_excess(fr[k])
            ok = np.where(np.isfinite(Ex).any(axis=1))[0]
            nz = int(ok.max()) + 2 if len(ok) else 4
            a = axs[r_, c_]
            vm = max(1.0, float(np.nanmax(np.abs(Ex[:nz, :nz])))) if np.isfinite(Ex[:nz, :nz]).any() else 1.0
            im = a.imshow(Ex[:nz, :nz].T, origin="lower", cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="nearest")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
            a.set_title("%s  t/t_c=%.3f\njet width %s cells" % (lab, fr[k]["t"] / tc, jet_width_cells(fr[k])), fontsize=8)
            a.set_xlabel("j (y cells from axis)"); a.set_ylabel("k (z cells)")
    fig.suptitle(r"$\eta=0.5$ front along +x per (j, k) cell column minus a sphere of the (111) radius [cells]: "
                 "a jet confined to j,k $\lesssim$ 2 is an axis-column artifact; one widening with the front is a cone",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "front_map.png"), dpi=130); plt.close(fig)

    # ---- fig 6: radius vs angle from the nearest axis ---------------------------------------------
    fig, axs = plt.subplots(1, len(runs), figsize=(5.5 * len(runs), 4), squeeze=False)
    for r_, (lab, fr) in enumerate(runs.items()):
        ang = _ang(fr[0]["dirs"])
        cm = plt.cm.viridis(np.linspace(0, 1, 7))
        for c_, k in enumerate(picks(fr)):
            axs[0, r_].plot(ang, fr[k]["rad"] / R0, ".", ms=3, c=cm[c_], label="t/t_c=%.3f" % (fr[k]["t"] / tc))
        axs[0, r_].set_xlabel("angle from nearest axis [deg]"); axs[0, r_].set_ylabel(r"$R_{\eta=0.5}/R_0$")
        axs[0, r_].set_title(lab); axs[0, r_].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "radius_vs_angle.png"), dpi=150); plt.close(fig)
    print("\nwrote figures to", args.out)


if __name__ == "__main__":
    main()
