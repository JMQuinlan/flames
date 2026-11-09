#!/usr/bin/env python3
"""
onetimedebug -- one-shot data grab for the MarmPaper 02 / 04 collapse failures (micro and Mili,
current and .old.* runs).  Run it on INCLINE with no arguments:

    python3 onetimedebug.py

It finds
  * the slurm logs  /mmfs1/home/ttryon/FLAMES_out/flame_504{05,06,07,26,27}_stdout (+ err_ files)
  * every run dir   /mmfs1/home/ttryon/flames/bin/tests/FlowMarm*tant/**/{output_,}0[24]_*  (incl. *.old.*)
and writes everything to ./CollapseDebug/ next to this script (8 worker processes):

  SUMMARY.txt                 one line per run and per log, the log -> run pairing, crash lines
  overview_R.png              R/R0 vs coated KM for every run, micro-equivalent time (Mili t x 1e-4)
  overview_logs.png           dt vs time for every log
  runs/<tag>/                 one folder per run dir
      metadata, thermo.dat, diff.patch      copied from the run
      input_<deck>                          the repo deck matched by name (used for R0 / KM)
      plotfiles.txt                         plotfile list with times
      <tag>_dbg_*.png, _dbg_report.txt      the debug_02_04 plots and report
      <tag>_dbg_cache.pkl                   debug_02_04 per-frame data (radii, z=0 slices, extrema)
      <tag>_extra.pkl                       per-frame phase stats (gas / band / near liquid), surface
                                            divergence div_s u and kappa_s*div_s u in the band, pressure-
                                            floor hits, axis + diagonal ray profiles of every field
      <tag>_extra.png                       plots of the above
      cubes/<plotfile>.npz                  finest-level 3D box around the bubble, every plotfile field
                                            (float32) + grid_level, for the last frames, R_min and a few more
  logs/<id>/                  one folder per slurm job
      flame_<id>_stdout.filtered.gz         the log minus mpi/DynamicTimestep noise and STEP lines
      err_flame_<id>_stdout(.tail)          the err file (tail if large)
      <id>_log.pkl                          step/time/dt, Advance dt limits, subcycle N, STEP8REPAIR,
                                            warnings with step/time, NaN block, header Parse lines
      <id>_log.png                          dt, dt limits and subcycle N vs time
  CollapseDebug.tar.gz        the whole folder, for download

Overrides (local testing): --roots DIR ... --runs DIR ... --logs FILE ... --out DIR --nproc N --no-tar
"""
import argparse
import glob
import gzip
import os
import pickle
import re
import shutil
import sys
import tarfile
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import debug_02_04 as D                                             # noqa: E402
SA = D.SA
plt = D.plt
from sch20_analysis import Sch20Analysis                            # noqa: E402

# =========================================================================================== #
#  CONFIG
# =========================================================================================== #
LOG_DIR = "/mmfs1/home/ttryon/FLAMES_out"
LOG_IDS = ["50405", "50406", "50407", "50426", "50427"]
RUN_ROOTS = ["/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant",
             "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmattant"]
RUN_NAME = re.compile(r"^(output_)?0[24]_")
OUT_DIR = os.path.join(_HERE, "CollapseDebug")
NPROC = 8
HALF_R0 = {"02": 1.3, "04": 1.9}          # analysis box half width / R0
GAIN = {"02": 1.0, "04": 1.107}           # KM drive box gain (see ../README.md)
CUBE_ALL_IF_LE = 20                       # save a cube for every frame if a run has <= this many
CUBE_LAST = 4                             # else: last N frames + R_min + CUBE_EXTRA evenly spaced
CUBE_EXTRA = 4
P_FLOOR_HIT = 1.0e-3                      # Pa: a cell at/below this counts as on the pressure floor
ERR_COPY_MAX = 5_000_000                  # bytes: copy the whole err file below this, else its tail
RAYS = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1), "xy": (1, 1, 0), "xz": (1, 0, 1),
        "yz": (0, 1, 1), "diag": (1, 1, 1)}

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = re.compile(r"btl_tcp_proc|reachable pairing|Unused ParmParse|DynamicTimestep_Update|^\s*$")
STEP_END = re.compile(r"STEP\s+(\d+)\s+ends\.\s+TIME\s*=\s*([-+0-9.eE]+)\s+DT\s*=\s*([-+0-9.eE]+)")
STEP_START = re.compile(r"STEP\s+(\d+)\s+starts")
ADV = re.compile(r"\(Advance\)\s+(dt_acoustic|dt_viscous|dt_capillary|dt_force|dt_max|c_max|v[xyz]_max)\s*[:=]?\s*"
                 r"([-+0-9.eE]+|nan|inf)", re.I)
SUBN = re.compile(r"(Subcycle\w+)\s+lev=(\d+):\s+required N=(\d+)")
REPAIR = re.compile(r"STEP8REPAIR lev=(\d+) i=(-?\d+) j=(-?\d+)(?: k=(-?\d+))?")
WARN = re.compile(r"^(WARNING|ABORT|ERROR)|NaN|nan detected|Abort|abort\(|SIGABRT|SIGSEGV|SIGFPE|Killed|"
                  r"CANCELLED|TIME LIMIT|oom", re.I)


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# =========================================================================================== #
#  discovery
# =========================================================================================== #
def find_runs(roots):
    out = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dns, _ in os.walk(root):
            depth = dp[len(root):].count(os.sep)
            if depth > 3:
                dns[:] = []
                continue
            keep = []
            for d in dns:
                p = os.path.join(dp, d)
                if d.endswith("cell"):
                    continue
                if RUN_NAME.match(d) and (glob.glob(os.path.join(p, "*cell")) or
                                          os.path.isfile(os.path.join(p, "thermo.dat"))):
                    out.append(p)
                else:
                    keep.append(d)
            dns[:] = keep
    return sorted(set(out))


def run_key(path):
    """'02' / '04', Mili flag, repo deck path, tag."""
    base = os.path.basename(path.rstrip("/"))
    name = re.sub(r"^output_", "", base).split(".old")[0]
    key = name[:2] if name[:2] in ("02", "04") else "02"
    mili = "Mili" in name
    deck = os.path.normpath(os.path.join(_HERE, "..", "input_" + name))
    if not os.path.isfile(deck):
        deck = os.path.normpath(os.path.join(_HERE, "..", {"02": "input_02_FreeCollapse",
                                                            "04": "input_04_ForcedNonlinear"}[key]
                                             + ("_Mili" if mili else "")))
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    if path.rstrip("/").split(os.sep)[-2] not in ("MarmPaper",):
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", path.rstrip("/").split(os.sep)[-2]) + "__" + tag
    return key, mili, deck, tag


def parse_metadata(p):
    kv = {}
    if os.path.isfile(p):
        with open(p, errors="replace") as fh:
            for ln in fh:
                ln = ln.split("#", 1)[0]
                if "=" in ln and "::" not in ln:
                    k, v = ln.split("=", 1)
                    kv[k.strip()] = v.strip()
    return kv


def fnum(s, default=np.nan):
    try:
        return float(str(s).split()[0])
    except Exception:
        return default


# =========================================================================================== #
#  log parsing
# =========================================================================================== #
def parse_log(path):
    L = dict(path=path, step=[], t=[], dt=[], adv={}, subn=[], repairs=[], n_repairs_step={}, warn=[],
             header=[], nan_block=[], tail=[], marm=None, domain=None, R0=None, err_tail=[])
    cur_step, cur_t = -1, 0.0
    filt = []
    in_nan = 0
    with open(path, errors="replace") as fh:
        for raw in fh:
            ln = ANSI.sub("", raw.rstrip("\n"))
            m = STEP_END.search(ln)
            if m:
                cur_step, cur_t = int(m.group(1)), float(m.group(2))
                L["step"].append(cur_step); L["t"].append(cur_t); L["dt"].append(float(m.group(3)))
                continue
            m = STEP_START.search(ln)
            if m:
                cur_step = int(m.group(1))
                continue
            if NOISE.search(ln):
                continue
            filt.append(ln)
            if len(L["header"]) < 600 and cur_step < 0:
                L["header"].append(ln)
            m = ADV.search(ln)
            if m:
                L["adv"].setdefault(m.group(1), []).append((cur_step, cur_t, fnum(m.group(2))))
            m = SUBN.search(ln)
            if m:
                L["subn"].append((cur_step, cur_t, m.group(1), int(m.group(2)), float(m.group(3))))
            m = REPAIR.search(ln)
            if m:
                L["n_repairs_step"][cur_step] = L["n_repairs_step"].get(cur_step, 0) + 1
                if len(L["repairs"]) < 5000:
                    L["repairs"].append((cur_step, cur_t, ln[:300]))
            if "Marmottant ON" in ln and L["marm"] is None:
                L["marm"] = ln
                r = re.search(r"R0=([-+0-9.eE]+)", ln)
                L["R0"] = float(r.group(1)) if r else None
            if L["domain"] is None:
                r = re.search(r"\(domain ([-+0-9.eE]+)\)", ln)
                if r:
                    L["domain"] = float(r.group(1))
            if "NaN DETECTED" in ln or in_nan:
                L["nan_block"].append(ln[:400])
                in_nan = 30 if "NaN DETECTED" in ln else in_nan - 1
            if WARN.search(ln) and len(L["warn"]) < 20000:
                L["warn"].append((cur_step, cur_t, ln[:400]))
            L["tail"].append(ln[:400])
            if len(L["tail"]) > 800:
                L["tail"] = L["tail"][-400:]
    for k in ("step", "t", "dt"):
        L[k] = np.array(L[k], float)
    L["tail"] = L["tail"][-400:]
    return L, filt


def do_log(lid, log_path, out_root):
    d = os.path.join(out_root, "logs", lid)
    os.makedirs(d, exist_ok=True)
    L, filt = parse_log(log_path)
    with gzip.open(os.path.join(d, os.path.basename(log_path) + ".filtered.gz"), "wt") as fh:
        fh.write("\n".join(filt) + "\n")
    m = re.search(r"flame_(\d+)_stdout$", log_path)
    if m:
        e = os.path.join(os.path.dirname(log_path), "err_flame_%s_stdout" % m.group(1))
        if os.path.isfile(e):
            if os.path.getsize(e) <= ERR_COPY_MAX:
                shutil.copy2(e, d)
            with open(e, errors="replace") as fh:
                fh.seek(max(0, os.path.getsize(e) - 200000))
                tail = [ANSI.sub("", x.rstrip()) for x in fh.readlines()[-300:]]
            L["err_tail"] = tail
            with open(os.path.join(d, os.path.basename(e) + ".tail"), "w") as fh:
                fh.write("\n".join(tail) + "\n")
    with open(os.path.join(d, "%s_log.pkl" % lid), "wb") as fh:
        pickle.dump(L, fh)
    try:
        plot_log(L, lid, os.path.join(d, "%s_log.png" % lid))
    except Exception:
        traceback.print_exc()
    say("  log %s: %d steps, last t %.6g, %d warnings, %d subcycle lines, %d repairs" %
        (lid, len(L["t"]), L["t"][-1] if len(L["t"]) else np.nan, len(L["warn"]), len(L["subn"]),
         sum(L["n_repairs_step"].values())))
    return L


def plot_log(L, lid, png):
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if len(L["t"]):
        ax[0].semilogy(L["t"], L["dt"], "-", lw=0.8, color="k", label="dt")
    for k, v in L["adv"].items():
        if k.startswith("dt_"):
            a = np.array([(x[1], x[2]) for x in v])
            if len(a):
                ax[1].semilogy(a[:, 0], np.abs(a[:, 1]), ".-", ms=2, lw=0.6, label=k)
    for nm in sorted(set(x[2] for x in L["subn"])):
        a = np.array([(x[1], x[4]) for x in L["subn"] if x[2] == nm])
        ax[2].semilogy(a[:, 0], a[:, 1], "o", ms=3, label=nm + " required N")
    if L["n_repairs_step"]:
        st = np.array(sorted(L["n_repairs_step"]))
        tt = np.interp(st, L["step"], L["t"]) if len(L["t"]) else st
        ax[2].semilogy(tt, [L["n_repairs_step"][s] for s in st], "x", color="tab:red", label="STEP8REPAIR / step")
    for a, lab in zip(ax, ("dt [s]", "Advance dt limits [s]", "count")):
        a.set_ylabel(lab)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7)
    ax[-1].set_xlabel("t [s]")
    fig.suptitle("flame_%s  R0=%s  domain=%s" % (lid, L["R0"], L["domain"]), fontsize=10)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# =========================================================================================== #
#  per-frame extras: cube, phase stats, surface divergence, ray profiles
# =========================================================================================== #
def _cube(ds, half_m):
    dim, dle, dre, sym, cen, dx = SA._geom(ds)
    m = int(np.ceil(half_m / dx)) + 2
    snap = lambda v, d: dle[d] + np.floor((v - dle[d]) / dx + 1e-9) * dx
    lo = [cen[d] if sym[d] else max(snap(cen[d] - m * dx, d), dle[d]) for d in range(3)]
    dims = [m if sym[d] else 2 * m for d in range(3)]
    cg = ds.covering_grid(level=ds.index.max_level, left_edge=lo, dims=dims)
    F = {}
    for f in sorted(f for (ft, f) in ds.field_list if ft == "boxlib"):
        F[f] = np.asarray(cg[("boxlib", f)], np.float32)
    try:
        F["grid_level"] = np.asarray(cg[("index", "grid_level")], np.float32)
    except Exception:
        pass
    off = np.array(lo) - cen
    return F, dx, off, sym


def _extra(args):
    pf, R0, half, kappa_s, save_cube, cube_dir = args
    try:
        yt = SA._yt()
        ds = yt.load(pf)
        F, dx, off, sym = _cube(ds, half * R0)
        o = {"name": os.path.basename(pf), "t": float(ds.current_time), "dx": dx}
        if save_cube:
            np.savez_compressed(os.path.join(cube_dir, os.path.basename(pf) + ".npz"),
                                dx=dx, off=off, sym=sym, R0=R0, t=o["t"], **F)
            o["cube"] = True
        eta = F["eta"].astype(float)
        sh = eta.shape
        X = [(np.arange(sh[q]) + 0.5) * dx + off[q] for q in range(3)]
        XX = np.meshgrid(*X, indexing="ij")
        r = np.sqrt(sum(x * x for x in XX))
        u = [F["velocity" + c].astype(float) for c in "xyz" if "velocity" + c in F]
        p = F.get("pressure")
        # ---- surface divergence div_s u = div u - n.(grad u).n in the band --------------------
        if len(u) == 3:
            G = [[np.gradient(u[i], dx, axis=j) for j in range(3)] for i in range(3)]
            div = G[0][0] + G[1][1] + G[2][2]
            ge = np.gradient(eta, dx)
            gm = np.sqrt(sum(g * g for g in ge)) + 1e-300
            n = [g / gm for g in ge]
            nGn = sum(n[i] * G[i][j] * n[j] for i in range(3) for j in range(3))
            divs = div - nGn
            ur = sum(u[i] * XX[i] for i in range(3)) / np.maximum(r, 1e-300)
            umag = np.sqrt(sum(x * x for x in u))
        else:
            divs = div = ur = umag = np.zeros_like(eta)
        gas, band, liq = eta < 0.05, (eta > 0.05) & (eta < 0.95), eta > 0.95
        Rb = np.median(r[(eta > 0.4) & (eta < 0.6)]) if np.any((eta > 0.4) & (eta < 0.6)) else R0
        near = liq & (r < Rb + 0.5 * R0)

        def stats(v, mask):
            v = v[mask]
            v = v[np.isfinite(v)]
            if not len(v):
                return None
            return dict(min=float(v.min()), p01=float(np.percentile(v, 1)), med=float(np.median(v)),
                        p99=float(np.percentile(v, 99)), max=float(v.max()), n=int(len(v)))

        def loc(v, mask, fn):
            w = np.where(mask & np.isfinite(v), v, np.nan)
            if not np.isfinite(w).any():
                return None
            i = np.unravel_index(fn(w), w.shape)
            x = np.array([X[q][i[q]] for q in range(3)])
            rr = float(np.sqrt((x * x).sum()))
            return dict(xyz_R0=(x / R0).tolist(), r_R0=rr / R0,
                        theta=float(np.degrees(np.arccos(x[2] / rr))) if rr > 0 else 0.0,
                        eta=float(eta[i]), level=float(F["grid_level"][i]) if "grid_level" in F else -1)

        o["R_band"] = float(Rb)
        regions = dict(gas=gas, band=band, near_liq=near)
        S = {}
        for rn, mk in regions.items():
            S[rn] = {}
            for fname in ("pressure", "density", "T", "Gamma", "shell", "kappaAvg", "rho_eta0", "rho_eta1",
                          "energy_per_vol"):
                if fname in F:
                    S[rn][fname] = stats(F[fname].astype(float), mk)
            S[rn]["umag"] = stats(umag, mk)
            S[rn]["u_r"] = stats(ur, mk)
            S[rn]["div_u"] = stats(div, mk)
            if p is not None:
                S[rn]["n_pfloor"] = int(np.sum(mk & (p <= P_FLOOR_HIT)))
                S[rn]["p_min_loc"] = loc(p.astype(float), mk, np.nanargmin)
        S["band"]["div_s_u"] = stats(divs, band)
        S["band"]["div_s_u_loc_min"] = loc(divs, band, np.nanargmin)
        if kappa_s and np.isfinite(kappa_s):
            S["band"]["kappa_s_div_s"] = stats(kappa_s * divs, band)
        if "kappaAvg" in F:
            kR = F["kappaAvg"].astype(float) * Rb
            S["band"]["kR"] = stats(kR, band)
            S["band"]["kR_loc_max"] = loc(np.abs(kR), band, np.nanargmax)
        S["all"] = dict(umag_loc_max=loc(umag, np.ones_like(gas), np.nanargmax),
                        n_nonfinite={k: int((~np.isfinite(v)).sum()) for k, v in F.items()})
        o["stats"] = S
        # ---- ray profiles (axes, face diagonals, body diagonal) -------------------------------
        s = np.arange(0.0, half * R0, 0.5 * dx)
        prof = {"s_R0": s / R0}
        extra = dict(umag=umag, u_r=ur, div_s_u=divs, div_u=div)
        for rn, dv in RAYS.items():
            dv = np.array(dv, float) / np.linalg.norm(dv)
            pts = [np.abs(s * dv[q]) if sym[q] else s * dv[q] for q in range(3)]
            pr = {}
            for fname, arr in list(F.items()) + list(extra.items()):
                if fname == "grid_level":
                    continue
                pr[fname] = SA._sample(np.asarray(arr, float), pts, dx, off).astype(np.float32)
            prof[rn] = pr
        o["prof"] = prof
        return o
    except Exception as exc:
        return {"error": "%s: %s: %s" % (os.path.basename(pf), type(exc).__name__, exc),
                "tb": traceback.format_exc()}


def plot_extra(ex, R0, tag, png, tscale, tunit):
    ex = [e for e in ex if "stats" in e]
    if not ex:
        return
    t = np.array([e["t"] for e in ex]) * tscale

    def g(reg, f, k):
        out = []
        for e in ex:
            v = e["stats"].get(reg, {}).get(f)
            out.append(v[k] if isinstance(v, dict) and k in v else np.nan)
        return np.array(out)

    fig, ax = plt.subplots(3, 3, figsize=(15, 11))
    a = ax[0, 0]
    for reg, c in (("gas", "tab:red"), ("band", "tab:purple"), ("near_liq", "tab:blue")):
        a.plot(t, g(reg, "pressure", "min"), "-o", ms=3, color=c, label=reg + " min")
        a.plot(t, g(reg, "pressure", "med"), "--", color=c, lw=0.8, label=reg + " median")
    a.set_yscale("symlog", linthresh=1e2); a.set_title("pressure by region"); a.legend(fontsize=6)
    a = ax[0, 1]
    for reg, c in (("gas", "tab:red"), ("band", "tab:purple"), ("near_liq", "tab:blue")):
        a.plot(t, [e["stats"][reg].get("n_pfloor", 0) for e in ex], "-o", ms=3, color=c, label=reg)
    a.set_title("cells with p <= %g Pa" % P_FLOOR_HIT); a.legend(fontsize=7)
    a = ax[0, 2]
    for k, ls in (("min", "-"), ("p01", "--"), ("med", ":"), ("p99", "--"), ("max", "-")):
        a.plot(t, g("band", "kappa_s_div_s", k), ls, label=k)
    a.axhline(0, color="k", lw=0.5); a.set_title(r"band $\kappa_s\,\mathrm{div}_s u$ [N/m]"); a.legend(fontsize=7)
    a = ax[1, 0]
    for k, ls in (("min", "-"), ("med", ":"), ("max", "-")):
        a.plot(t, g("band", "kR", k), ls, label=k)
    a.set_title(r"band $\kappa_{avg} R$"); a.legend(fontsize=7)
    a = ax[1, 1]
    for k, ls in (("min", "-"), ("med", ":"), ("max", "-")):
        a.plot(t, g("band", "Gamma", k), ls, label="Gamma " + k)
    a.set_yscale("log"); a.set_title("band Gamma"); a.legend(fontsize=7)
    a = ax[1, 2]
    for reg, c in (("gas", "tab:red"), ("band", "tab:purple"), ("near_liq", "tab:blue")):
        a.plot(t, g(reg, "umag", "max"), "-o", ms=3, color=c, label=reg)
    a.set_yscale("log"); a.set_title("|u| max by region"); a.legend(fontsize=7)
    # ray profiles at the last 3 frames
    for col, (fname, ttl) in enumerate((("pressure", "p along rays"), ("eta", "eta along rays"),
                                        ("u_r", "u_r along rays"))):
        a = ax[2, col]
        for e, ls in zip(ex[-3:], (":", "--", "-")):
            pr = e["prof"]
            for rn, c in (("x", "tab:blue"), ("z", "tab:green"), ("diag", "tab:red"), ("xy", "tab:orange")):
                if fname in pr[rn]:
                    a.plot(pr["s_R0"], pr[rn][fname], ls, color=c, lw=0.9,
                           label="%s t=%.4g" % (rn, e["t"] * tscale) if ls == "-" or rn == "x" else None)
        a.set_title(ttl + " (last 3 frames : -- -)"); a.set_xlabel("r/R0"); a.legend(fontsize=6)
    for a in ax[:2].ravel():
        a.set_xlabel("t [%s]" % tunit)
    fig.suptitle(tag, fontsize=11)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    plt.close(fig)


# =========================================================================================== #
#  per run
# =========================================================================================== #
def do_run(path, out_root, nproc):
    key, mili, deck, tag = run_key(path)
    d = os.path.join(out_root, "runs", tag)
    os.makedirs(os.path.join(d, "cubes"), exist_ok=True)
    say("RUN %s  (key %s, %s, deck %s)" % (path, key, "Mili" if mili else "micro", os.path.basename(deck)))
    for f in ("metadata", "thermo.dat", "diff.patch"):
        p = os.path.join(path, f)
        if os.path.isfile(p) and os.path.getsize(p) < 50_000_000:
            shutil.copy2(p, d)
    if os.path.isfile(deck):
        shutil.copy2(deck, d)
    md = parse_metadata(os.path.join(path, "metadata"))
    kappa_s = fnum(md.get("shell.kappa_s", np.nan))
    pfs = SA._plotfiles(path)
    with open(os.path.join(d, "plotfiles.txt"), "w") as fh:
        for pf in pfs:
            fh.write("%s\n" % pf)
    info = dict(path=path, tag=tag, key=key, mili=mili, deck=deck, n_plot=len(pfs), kappa_s=kappa_s)
    if not pfs:
        say("  no plotfiles")
        return info
    A = Sch20Analysis(deck, "", case="Collapsing" if key == "02" else "Forced Non-Linear", coated=True,
                      stem=tag + "_dbg", out_dir=d, nproc=nproc, drive_gain=GAIN[key], cache=False, verbose=False)
    R0, half = A.R0, HALF_R0[key]
    stem = os.path.join(d, tag)
    fr, errors = D.load_frames(path, R0, half, stem + "_dbg_cache.pkl", nproc)
    th = D.read_thermo(path)
    info.update(R0=R0, stop_time=A.meta["stop_time"], A=A, fr=fr, th=th, errors=errors, stem=stem)
    if fr:
        info["t_last"] = fr[-1]["t"]
        rm = D.col(fr, "R_mean")
        info["i_rmin"] = int(np.nanargmin(rm)) if np.isfinite(rm).any() else len(fr) - 1
    # cube selection
    n = len(pfs)
    if n <= CUBE_ALL_IF_LE:
        sel = set(range(n))
    else:
        sel = set(range(max(0, n - CUBE_LAST), n)) | set(np.linspace(0, n - 1, CUBE_EXTRA).round().astype(int))
        if fr:
            nm = fr[info["i_rmin"]]["name"]
            sel |= {i for i, pf in enumerate(pfs) if os.path.basename(pf) == nm}
    say("  extras: %d frames, %d cubes" % (n, len(sel)))
    jobs = [(pf, R0, half, kappa_s, i in sel, os.path.join(d, "cubes")) for i, pf in enumerate(pfs)]
    ex = SA._pool_map(_extra, jobs, nproc)
    for e in ex:
        if "error" in e:
            say("  [extra] SKIPPED", e["error"])
    ex = sorted([e for e in ex if "error" not in e], key=lambda e: e["t"])
    with open(stem + "_extra.pkl", "wb") as fh:
        pickle.dump(dict(frames=ex, R0=R0, kappa_s=kappa_s, metadata=md, path=path), fh)
    info["extra"] = ex
    tscale, tunit = (1e5, "micro-eq ns (Mili t x 1e-4)") if mili else (1e9, "ns")
    try:
        plot_extra(ex, R0, tag, stem + "_extra.png", tscale, tunit)
    except Exception:
        traceback.print_exc()
    return info


def debug_plots(info, log):
    A, fr, th, stem = info["A"], info["fr"], info["th"], info["stem"]
    if not fr:
        return
    lg = None
    if log is not None:
        lg = dict(files=[log["path"]], step=log["step"], t=log["t"], dt=log["dt"], tail=log["tail"][-60:],
                  errors=[w[2] for w in log["warn"]][-400:])
    for fn, args in ((D.plot_radius, (A, fr, th, stem)), (D.plot_contours, (A, fr, stem, 8)),
                     (D.plot_modes, (A, fr, stem)), (D.plot_health, (A, fr, th, lg, stem)),
                     (D.plot_fields, (A, fr, stem))):
        try:
            fn(*args)
        except Exception as exc:
            traceback.print_exc()
            say("  [%s] FAILED: %s" % (fn.__name__, exc))
    try:
        D.report(A, fr, th, lg, log["path"] if log else None, info["errors"], info["path"], stem)
    except Exception:
        traceback.print_exc()


def pair_logs(runs, logs):
    """log id -> run tag: same R0 and Mili/micro scale, then the run whose last plotfile time is the
    closest at or below the log's last step time."""
    pairs = {}
    used = set()
    for lid, L in sorted(logs.items(), key=lambda kv: -(kv[1]["t"][-1] if len(kv[1]["t"]) else 0)):
        if not len(L["t"]):
            continue
        tl = L["t"][-1]
        best, bscore = None, np.inf
        for r in runs:
            if "t_last" not in r or r["tag"] in used:
                continue
            if L["R0"] and abs(r["R0"] - L["R0"]) > 1e-3 * r["R0"]:
                continue
            if tl > 1.5 * r["stop_time"]:
                continue
            gap = tl - r["t_last"]
            score = gap if gap >= -1e-15 else 10 * abs(gap)
            if r["th"] is not None and len(r["th"].get("time", [])):
                score = min(score, abs(tl - r["th"]["time"][-1]))
            score /= max(tl, 1e-30)
            if score < bscore:
                best, bscore = r["tag"], score
        if best is not None and bscore < 0.25:
            pairs[lid] = best
            used.add(best)
    return pairs


def crash_line(L):
    for pat in (r"^ABORT", r"NaN DETECTED", r"^ERROR", r"SIGSEGV|SIGFPE|Segmentation"):
        for w in L["warn"]:
            if re.search(pat, w[2]):
                return "step %d t %.6g: %s" % (w[0], w[1], w[2][:200])
    for w in reversed(L["warn"]):
        if re.search(r"CANCELLED|TIME LIMIT|Killed|oom", w[2], re.I):
            return w[2][:200]
    return "(no abort line; see tail)"


def overview(runs, logs, out_root):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for k, key in enumerate(("02", "04")):
        a = ax[k]
        km_done = set()
        for r in runs:
            if r.get("key") != key or not r.get("fr"):
                continue
            sc = 1e5 if r["mili"] else 1e9
            A = r["A"]
            if not r["mili"] and "km" not in km_done:
                tk, Rk, _ = A.km
                a.plot(tk * sc, Rk / A.R0, "-", color="0.2", lw=1.5, label="coated KM (micro deck)")
                km_done.add("km")
            t = D.col(r["fr"], "t") * sc
            a.plot(t, D.col(r["fr"], "R_mean") / r["R0"], "-o", ms=3, lw=1, label=r["tag"])
        a.set_xlabel("t [micro-equivalent ns]  (Mili t x 1e-4)"); a.set_ylabel("R/R0 (mean of 400 dirs)")
        a.set_title({"02": "02 free collapse", "04": "04 forced nonlinear"}[key]); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out_root, "overview_R.png"), dpi=110); plt.close(fig)
    fig, a = plt.subplots(figsize=(10, 5))
    for lid, L in logs.items():
        if len(L["t"]):
            sc = 1e5 if (L["R0"] or 0) > 1e-3 else 1e9
            a.semilogy(L["t"] * sc, L["dt"] * sc, lw=0.8, label="%s (R0 %s)" % (lid, L["R0"]))
    a.set_xlabel("t [micro-equivalent ns]"); a.set_ylabel("dt [micro-equivalent ns]"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out_root, "overview_logs.png"), dpi=110); plt.close(fig)


def summary(runs, logs, pairs, out_root, t0):
    S = []
    P = S.append
    P("onetimedebug  %s   (%.1f min)" % (time.strftime("%Y-%m-%d %H:%M"), (time.time() - t0) / 60))
    P("")
    P("RUNS")
    P("%-58s %4s %5s %6s %12s %12s %7s %9s %10s %s" % ("tag", "key", "scale", "nplot", "t_last", "stop", "%done",
                                                      "R_min/R0", "kappa_s", "path"))
    for r in runs:
        if "t_last" in r:
            rm = D.col(r["fr"], "R_mean") / r["R0"]
            P("%-58s %4s %5s %6d %12.5g %12.5g %7.1f %9.4f %10.3g %s" %
              (r["tag"], r["key"], "Mili" if r["mili"] else "micro", r["n_plot"], r["t_last"], r["stop_time"],
               100 * r["t_last"] / r["stop_time"], np.nanmin(rm), r["kappa_s"], r["path"]))
        else:
            P("%-58s %4s %5s %6d   (no readable frames)  %s" % (r["tag"], r.get("key"), "Mili" if r.get("mili")
                                                                else "micro", r.get("n_plot", 0), r["path"]))
    P("")
    P("LOGS")
    for lid, L in sorted(logs.items()):
        P("flame_%s  R0 %s  domain %s  steps %d  last t %s  last dt %s  -> run %s" %
          (lid, L["R0"], L["domain"], len(L["t"]), ("%.6g" % L["t"][-1]) if len(L["t"]) else "-",
           ("%.3g" % L["dt"][-1]) if len(L["t"]) else "-", pairs.get(lid, "(unpaired)")))
        P("    crash: " + crash_line(L))
        if L["subn"]:
            s = L["subn"][0]
            P("    first subcycle-cap warning: step %d t %.6g %s lev %d N %.3g  (%d total)" %
              (s[0], s[1], s[2], s[3], s[4], len(L["subn"])))
        if L["n_repairs_step"]:
            st = min(L["n_repairs_step"])
            P("    STEP8REPAIR: %d lines over %d steps, first at step %d" %
              (sum(L["n_repairs_step"].values()), len(L["n_repairs_step"]), st))
        if L["marm"]:
            P("    " + L["marm"][:250])
    txt = "\n".join(S) + "\n"
    with open(os.path.join(out_root, "SUMMARY.txt"), "w") as fh:
        fh.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="run dirs (default: search RUN_ROOTS)")
    ap.add_argument("--roots", nargs="*", default=None, help="search roots (default: RUN_ROOTS)")
    ap.add_argument("--logs", nargs="*", default=None, help="slurm stdout files (default: LOG_IDS in LOG_DIR)")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--nproc", type=int, default=NPROC)
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)

    log_paths = a.logs if a.logs is not None else [os.path.join(LOG_DIR, "flame_%s_stdout" % i) for i in LOG_IDS]
    logs = {}
    for p in log_paths:
        m = re.search(r"flame_(\d+)_stdout", os.path.basename(p))
        lid = m.group(1) if m else re.sub(r"\W", "_", os.path.basename(p))
        if not os.path.isfile(p):
            say("log missing:", p)
            continue
        try:
            logs[lid] = do_log(lid, p, a.out)
        except Exception:
            traceback.print_exc()

    run_dirs = a.runs if a.runs is not None else find_runs(a.roots or RUN_ROOTS)
    say("%d run dirs:" % len(run_dirs))
    for p in run_dirs:
        say("   ", p)
    runs = []
    for p in run_dirs:
        try:
            runs.append(do_run(os.path.abspath(p), a.out, a.nproc))
        except Exception:
            traceback.print_exc()
            runs.append(dict(path=p, tag=os.path.basename(p)))

    pairs = pair_logs(runs, logs)
    by_tag = {v: k for k, v in pairs.items()}
    for r in runs:
        if r.get("fr"):
            say("plots", r["tag"], "log", by_tag.get(r["tag"]))
            debug_plots(r, logs.get(by_tag.get(r["tag"])))
    try:
        overview(runs, logs, a.out)
    except Exception:
        traceback.print_exc()
    summary(runs, logs, pairs, a.out, t0)

    if not a.no_tar:
        tgz = os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(a.out.rstrip("/")) + ".tar.gz")
        say("packing", tgz)
        with tarfile.open(tgz, "w:gz") as tf:
            tf.add(a.out, arcname=os.path.basename(a.out.rstrip("/")))
        say("wrote %s (%.0f MB)" % (tgz, os.path.getsize(tgz) / 1e6))
    say("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
