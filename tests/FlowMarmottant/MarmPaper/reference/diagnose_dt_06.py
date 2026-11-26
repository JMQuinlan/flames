#!/usr/bin/env python3
"""Find what is holding down the time step (and the wall time per step) of a 06 / 06Exp Cattaneo run.

Defaults to the INCLINE run of the chosen series and case:
    output dir  /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_<series>_Cattaneo_<case>
    log         the StdOut of the running slurm job whose script / job name contains <series>_Cattaneo_<case>
                (fallback: newest /mmfs1/home/ttryon/FLAMES_out/flame_*_stdout -- check it is the right one)

Four parts:
  1. LOG: dt history, when dt dropped, which AMR level sets dt (the "lev=" lines), sub-cycle / clamp warnings,
     and the live step rate (samples the log for --watch seconds) -> wall time per plotfile.
  2. REFERENCE: the dt a pure-water acoustic limit would give on this grid (c_water from the deck), so you can
     see at once whether the run is at its natural limit or something is throttling it.
  3. PLOTFILE: rebuilds every dt limit of Hydro2::TimeStepComplete level by level from the newest plotfile
     (acoustic as the code does it -- max c plus max |u_d| over the level, conduction, capillary) and the two
     sub-cycle counts (bulk viscous, shell kappa_s), and lists the cells that set each one with their state
     (eta, densities, p, T, |u|, position).  The force limit (dt_force) needs the source term, which is not in
     the plotfile.
  4. COST + VERDICT: cells / boxes per level, cell-updates per coarse step and where they go, boxes per rank,
     the binding limit, and what can be done about it.

Notes
  * The log's DT is the COARSE (level 0) step.  Level L runs at DT / prod(nsubsteps[1..L]) (default 2 per level,
    so the finest level of a max_level 4 run takes 16 steps per coarse step).
  * eta = liquid volume fraction (1 in the water, 0 in the gas core).

Usage (on INCLINE, no arguments needed for the 06Exp 160 kPa run):
    python3 diagnose_dt_06.py
    python3 diagnose_dt_06.py --case LowPa
    python3 diagnose_dt_06.py --series 06 --case HighPa                      (the octant 06 runs)
    python3 diagnose_dt_06.py --dir OUTDIR --log LOGFILE --plot OUTDIR/123456cell --top 8 --watch 120 --ranks 64
Needs only numpy (no yt).  Reads one plotfile level at a time, fab by fab (low memory).
"""
import argparse
import glob
import math
import os
import re
import subprocess
import sys
import time

import numpy as np

ROOT = "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper"
LOGDIR = "/mmfs1/home/ttryon/FLAMES_out"
ANSI = re.compile(rb"\x1b\[[0-9;]*m")
SMALL = 1e-8          # Hydro2 `small`
SAFETY = 0.9          # dt_max *= 0.9 in TimeStepComplete


# ----------------------------------------------------------------------------- locate files
def find_outdir(series, case):
    stem = f"output_{series}_Cattaneo_{case}"
    c = [d for d in glob.glob(f"{ROOT}/{stem}*") if os.path.isdir(d)
         and re.fullmatch(rf"{re.escape(stem)}(\.\d+)?", os.path.basename(d))]
    if not c:
        sys.exit(f"no output dir {ROOT}/{stem}  (use --dir)")
    return max(c, key=os.path.getmtime)


def find_log(series, case):
    """-> (path, how, ranks): the slurm job whose script / name / stdout mentions <series>_Cattaneo_<case>."""
    tag = f"{series}_Cattaneo_{case}"
    try:
        ids = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        for jid in ids:
            info = subprocess.run(["scontrol", "show", "job", jid], capture_output=True, text=True, timeout=30).stdout
            fields = " ".join(re.findall(r"(?:Command|JobName|StdOut|WorkDir)=(\S+)", info))
            out = re.search(r"StdOut=(\S+)", info)
            # "06_Cattaneo_HighPa" must not match "06Exp_Cattaneo_HighPa" and vice versa: tag must not follow "Exp"
            if out and re.search(rf"(^|[^A-Za-z0-9]){re.escape(tag)}", fields) and os.path.isfile(out.group(1)):
                nt = re.search(r"NumTasks=(\d+)", info)
                return out.group(1), f"slurm job {jid}", (int(nt.group(1)) if nt else None)
    except (OSError, subprocess.SubprocessError):
        pass
    c = glob.glob(f"{LOGDIR}/flame_*_stdout")
    if not c:
        return None, "none found", None
    return (max(c, key=os.path.getmtime),
            "NEWEST stdout in FLAMES_out (no running job matched -- check it is the right run, or pass --log)", None)


def read_metadata(outdir):
    """Alamo metadata: key = value, later lines (command line) override earlier ones."""
    kv = {}
    path = os.path.join(outdir, "metadata")
    if os.path.isfile(path):
        for line in open(path, errors="replace"):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                kv[k.strip()] = v.split("#")[0].strip()
    return kv


def num(kv, key, default):
    try:
        return float(kv[key].split()[0].strip('"'))
    except (KeyError, ValueError, IndexError):
        return default


def nsub_prod(nsub, lev):
    return float(np.prod(nsub[1:lev + 1]))


# ----------------------------------------------------------------------------- reference (pure water acoustic)
def reference_dt(kv, nsub):
    """Acoustic dt of undisturbed water on the finest grid, as TimeStepComplete would compute it."""
    g0, pi0 = num(kv, "eos0.gamma", 2.35), num(kv, "eos0.p0", 9.311895e8)
    rho_w = num(kv, "density0.ic.expression.region0", 997.8)
    p_w = num(kv, "pressure0.ic.expression.constant.p_inf", 1.022e5)
    cfl = num(kv, "cfl", 0.9)
    L = int(num(kv, "amr.max_level", 0))
    try:
        lo = np.array(kv["geometry.prob_lo"].split()[:3], float); hi = np.array(kv["geometry.prob_hi"].split()[:3], float)
        n = np.array(kv["amr.n_cell"].split()[:3], float)
    except (KeyError, ValueError):
        return None
    rr = num(kv, "amr.ref_ratio", 2)
    dxf = (hi - lo) / n / rr ** L
    c_w = math.sqrt(g0 * (p_w + pi0) / rho_w)
    dt_f = cfl / sum(c_w / d for d in dxf) * SAFETY
    return dict(c_w=c_w, dx_f=dxf, dt_f=dt_f, dt_coarse=dt_f * nsub_prod(nsub, L), L=L)


# ----------------------------------------------------------------------------- log
def scan_log(path):
    steps, times, dts = [], [], []
    lev_lines, warn, nan_lines = [], {}, 0
    pat = re.compile(rb"STEP (\d+) ends\. TIME = (\S+) DT = (\S+)")
    lev = re.compile(rb"lev=(\d+) +(\S+) \((\d+)\)")
    lev_hist = []      # (step index, binding level) once per coarse step
    cur = {}
    with open(path, "rb") as f:
        for raw in f:
            if b"ends. TIME" in raw:
                m = pat.search(raw)
                if m:
                    steps.append(int(m.group(1))); times.append(float(m.group(2))); dts.append(float(m.group(3)))
            elif b"lev=" in raw and b"-->" not in raw and b"Update" in raw:
                m = lev.search(ANSI.sub(b"", raw))
                if m:
                    l, d, ns = int(m.group(1)), float(m.group(2)), int(m.group(3))
                    if l == 0 and cur:
                        lev_hist.append((len(steps), cur)); cur = {}
                    cur[l] = d
                    lev_lines.append((l, d, ns))
                    if len(lev_lines) > 400:
                        del lev_lines[:200]
            elif b"Subcycle" in raw or b"exceeds" in raw or b"Clamping" in raw or b"clamping" in raw:
                s = ANSI.sub(b"", raw).decode(errors="replace").strip()
                key = re.sub(r"[-+]?\d[\d.e+-]*", "#", s)[:120]
                n, _ = warn.get(key, (0, s))
                warn[key] = (n + 1, s)
            elif b"nan" in raw.lower():
                nan_lines += 1
    if cur:
        lev_hist.append((len(steps), cur))
    return np.array(steps), np.array(times), np.array(dts), lev_lines, lev_hist, warn, nan_lines


def last_step(path, nbytes=2_000_000):
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - nbytes))
        tail = f.read()
    m = re.findall(rb"STEP (\d+) ends\. TIME = (\S+) DT = (\S+)", tail)
    return (int(m[-1][0]), float(m[-1][1]), float(m[-1][2])) if m else None


def report_log(path, how, plot_dt, nsub, watch, ref):
    """-> dict(dt=coarse DT now, rate=coarse steps per wall second or None, bind=binding level or None)"""
    out = dict(dt=None, rate=None, bind=None, simrate=None)
    print(f"\n=== LOG  {path}\n    ({how})")
    st = os.stat(path)
    print(f"    size {st.st_size/1e9:.2f} GB, last write {time.ctime(st.st_mtime)} ({(time.time()-st.st_mtime)/60:.1f} min ago)")
    steps, t, dt, lev_lines, lev_hist, warn, nnan = scan_log(path)
    if len(steps) == 0:
        print("    no STEP lines"); return out
    L = len(nsub) - 1
    out["dt"] = dt[-1]
    print(f"    steps {steps[0]}..{steps[-1]}, sim time {t[-1]*1e6:.6f} us, coarse DT now {dt[-1]:.4g} s "
          f"(finest level {dt[-1]/nsub_prod(nsub, L):.4g} s)")
    if ref:
        r = ref['dt_coarse'] / dt[-1]
        print(f"    pure-water acoustic reference: coarse {ref['dt_coarse']:.4g} s, finest {ref['dt_f']:.4g} s  -> "
              + (f"the run is {r:.1f}x BELOW it" if r > 1.05 else "the run is AT it (dt is the natural acoustic limit)"))
    n = len(dt)
    ref_med = np.median(dt[: max(10, n // 2)])
    print(f"    median coarse DT over the first half of the run {ref_med:.4g} s -> now {ref_med/dt[-1]:.1f}x smaller")
    skip = min(n - 1, 200)                     # start-up ramp from dt = 1e-12
    low = skip + np.where(dt[skip:] < 0.3 * ref_med)[0]
    if len(low):
        i0 = low[0]
        print(f"    DT first fell below 0.3x median at step {steps[i0]}, t = {t[i0]*1e6:.6f} us "
              f"({len(low)} of the {n-i0} steps since are below it)")
    # dt history over the whole run, by simulated time
    edges = np.unique(np.linspace(0, n - 1, 16).astype(int))
    print("    history  step        t [us]         DT [s]    (min DT in the stretch before)")
    for a, b in zip(np.r_[0, edges[:-1]], edges):
        print(f"             {steps[b]:<10d}  {t[b]*1e6:10.5f}   {dt[b]:.4g}   ({dt[a:b+1].min():.4g})")
    if plot_dt > 0:
        nxt = (np.floor(t[-1] / plot_dt + 1e-9) + 1) * plot_dt
        print(f"    next plotfile at {nxt*1e6:.4f} us: {(nxt-t[-1])/dt[-1]:.0f} more coarse steps at the current DT "
              f"({plot_dt/dt[-1]:.0f} coarse steps per plot interval)")
    if lev_lines:
        last = {}
        for l, d, ns in lev_lines:
            last[l] = d
        print("    last per-level dt limits (dt of that level; x nsubsteps above it -> coarse-equivalent):")
        for l in sorted(last):
            print(f"      lev {l}: {last[l]:.4g} s  -> coarse-equivalent {last[l]*nsub_prod(nsub, l):.4g} s")
        bind = min(last, key=lambda l: last[l] * nsub_prod(nsub, l))
        out["bind"] = bind
        print(f"      binding level: lev {bind}; the plotfile part names the limit")
        if lev_hist:
            k = lev_hist[-min(len(lev_hist), 2000):]
            cnt = {}
            for _, d in k:
                b = min(d, key=lambda l: d[l] * nsub_prod(nsub, l)); cnt[b] = cnt.get(b, 0) + 1
            print("      binding level over the last %d updates: %s" % (len(k), ", ".join(f"lev {l}: {c}" for l, c in sorted(cnt.items()))))
    else:
        print("    no 'lev=' lines (old binary)")
    if warn:
        print("    sub-cycle / clamp messages (count, latest):")
        for _, (c, s) in sorted(warn.items(), key=lambda x: -x[1][0])[:8]:
            print(f"      {c:8d}  {s[:200]}")
    if nnan:
        print(f"    lines mentioning nan: {nnan}")
    if watch > 0:
        a = last_step(path)
        print(f"    sampling the log for {watch:.0f} s ...", flush=True)
        time.sleep(watch)
        b = last_step(path)
        if a and b and b[0] > a[0]:
            rate = (b[0] - a[0]) / watch
            simrate = (b[1] - a[1]) / watch
            out["rate"], out["simrate"] = rate, simrate
            print(f"    live: {rate*60:.2f} coarse steps/min, {1/rate:.2f} s per coarse step, "
                  f"{simrate*3600*1e9:.3f} ns sim per wall hour")
            if plot_dt > 0 and simrate > 0:
                nxt = (np.floor(b[1] / plot_dt + 1e-9) + 1) * plot_dt
                print(f"    next plotfile in ~{(nxt-b[1])/simrate/3600:.1f} h; one plot interval "
                      f"({plot_dt*1e9:.0f} ns) takes ~{plot_dt/simrate/3600:.1f} h at this rate")
        else:
            print("    live: NO new steps in the window -> the job is hung (or the window is shorter than one step)")
    return out


# ----------------------------------------------------------------------------- plotfile (native reader)
def newest_plotfile(outdir):
    pf = [d for d in glob.glob(os.path.join(outdir, "*cell")) if os.path.isfile(os.path.join(d, "Header"))]
    if not pf:
        return None
    pf.sort(key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))
    for d in reversed(pf):            # newest COMPLETE one
        h = open(os.path.join(d, "Header")).read().split("\n")
        try:
            finest = int(h[int(h[1]) + 4])
            if all(os.path.isfile(os.path.join(d, f"Level_{l}", "Cell_H")) for l in range(finest + 1)):
                return d
        except (ValueError, IndexError):
            continue
    return None


def read_header(pf):
    h = open(os.path.join(pf, "Header")).read().split("\n")
    nv = int(h[1]); names = h[2:2 + nv]
    i = 2 + nv
    ndim = int(h[i]); tcur = float(h[i + 1]); finest = int(h[i + 2])
    lo = np.array(h[i + 3].split(), float); hi = np.array(h[i + 4].split(), float)
    doms = re.findall(r"\(\(([-\d,]+)\) \(([-\d,]+)\) \([-\d,]+\)\)", h[i + 6])
    dx = []
    for a, b in doms[: finest + 1]:
        n = np.array(b.split(","), int) - np.array(a.split(","), int) + 1
        dx.append((hi - lo) / n)
    return names, ndim, tcur, finest, lo, dx


def iter_fabs(pf, lev):
    d = os.path.join(pf, f"Level_{lev}")
    for line in open(os.path.join(d, "Cell_H")):
        if line.startswith("FabOnDisk:"):
            _, fn, off = line.split()
            with open(os.path.join(d, fn), "rb") as f:
                f.seek(int(off))
                hdr = f.readline().decode()
                m = re.search(r"\(\((\d+), \(([\d ]+)\)\),\((\d+), \(([\d ]+)\)\)\)\(\(([-\d,]+)\) \(([-\d,]+)\) \([-\d,]+\)\) (\d+)", hdr)
                nbyte = int(m.group(3)); order = m.group(4).split()
                endian = "<" if order[0] == str(nbyte) else ">"
                blo = np.array(m.group(5).split(","), int); bhi = np.array(m.group(6).split(","), int)
                nc = int(m.group(7)); shape = bhi - blo + 1
                a = np.fromfile(f, dtype=f"{endian}f{nbyte}", count=nc * int(np.prod(shape)))
            yield blo, a.reshape(nc, *shape[::-1])      # (comp, z, y, x)


def report_plotfile(pf, kv, top, dt_log, nsub, ranks, ref):
    """-> dict with rows (coarse-equivalent dt, limit, lev), per-level cells / boxes, sub-cycle counts"""
    names, ndim, tcur, finest, lo, dxs = read_header(pf)
    ix = {n: i for i, n in enumerate(names)}
    need = ["eta", "rho_eta0", "rho_eta1", "density", "energy0", "energy1", "velocityx", "velocityy", "velocityz",
            "grad_etax", "grad_etay", "grad_etaz", "pressure", "T"]
    miss = [n for n in need if n not in ix]
    if miss:
        print(f"    plotfile lacks {miss}; cannot rebuild limits"); return None
    g0, pi0 = num(kv, "eos0.gamma", 2.35), num(kv, "eos0.p0", 9.311895e8)
    g1, pi1 = num(kv, "eos1.gamma", 1.4), num(kv, "eos1.p0", 0.0)
    cv0, cv1 = num(kv, "eos0.cv", 2343.6161), num(kv, "eos1.cv", 714.286)
    k0, k1 = num(kv, "thermal.k0", 0.6), num(kv, "thermal.k1", 0.026)
    cond = int(num(kv, "thermal.conduction", 1)); amin = num(kv, "thermal.alpha_min", 1e-3)
    dtlim = int(num(kv, "thermal.dt_limit", 1))
    cfl = num(kv, "cfl", 0.9); cflv = num(kv, "cfl_v", cfl)
    mu0, mu1 = num(kv, "mu0", 0.0), num(kv, "mu1", 0.0)
    l0, l1 = num(kv, "mu0_b", 0.0), num(kv, "mu1_b", 0.0)
    vsub = int(num(kv, "visc_subcycle", 1)); vcfl = num(kv, "visc_subcycle_cfl", 0.9)
    vmax = num(kv, "visc_subcycle_max", 1000)
    kap = abs(num(kv, "shell.kappa_s", 0) - num(kv, "shell.mu_s", 0)) + 2 * abs(num(kv, "shell.mu_s", 0))
    sig = num(kv, "sigma", 0.0); st_on = num(kv, "apply_surface_tension", 0)
    nsub_max = num(kv, "shell.kappa_s_subcycle_max", 8192)
    centre = np.array([0.0, 0.0, 0.0])[:ndim]           # bubble centre (both series put it at the origin)
    print(f"\n=== PLOTFILE {pf}\n    t = {tcur*1e6:.6f} us, finest level {finest}; "
          f"gas gamma {g1} cv {cv1} k1 {k1}; cfl {cfl} cfl_v {cflv}; thermal.conduction {cond} alpha_min {amin}")
    res = dict(rows=[], levels=[], tcur=tcur, finest=finest, nsc={}, causes={})
    for lev in range(finest + 1):
        dx = dxs[lev]; fac = nsub_prod(nsub, lev)
        dt_lev = (dt_log / fac) if dt_log else None
        best = {"acoustic (per cell)": [], "conduction": [], "shell sub-step": [], "bulk visc sub-step": [],
                "largest c": [], "largest |u|": []}
        rho_min = np.inf; ncell = 0; nbox = 0
        cmax = 0.0; umax = np.zeros(ndim)
        for blo, A in iter_fabs(pf, lev):
            nbox += 1
            g = lambda n: A[ix[n]]
            eta = g("eta"); a1 = np.clip(eta, 1e-30, 1 - 1e-30); a2 = 1 - a1
            r0 = np.maximum(g("rho_eta0"), SMALL); r1 = np.maximum(g("rho_eta1"), SMALL)
            rho = g("density"); rho_min = min(rho_min, rho.min()); ncell += rho.size
            p0 = np.maximum((g0 - 1) * g("energy0") / a1 - g0 * pi0, -pi0 + SMALL)
            p1 = np.maximum((g1 - 1) * g("energy1") / a2 - g1 * pi1, -pi1 + SMALL)
            p0r, p1r = p0, p1                                  # raw phase pressures (before trace slaving)
            sm = lambda a: (lambda w: w * w * (3 - 2 * w))(np.clip((a - 1e-2) / (1e-1 - 1e-2), 0, 1))
            w0, w1 = sm(a1), sm(a2)
            p0, p1 = w0 * p0 + (1 - w0) * p1, w1 * p1 + (1 - w1) * p0
            c0 = np.sqrt(g0 * np.maximum(p0 + pi0, SMALL) / (r0 / a1))
            c1 = np.sqrt(g1 * np.maximum(p1 + pi1, SMALL) / (r1 / a2))
            Y0 = r0 / np.maximum(rho, SMALL)
            c = np.sqrt(Y0 * c0 ** 2 + (1 - Y0) * c1 ** 2)
            c = np.where(np.isfinite(c), c, 0.0)
            u = [np.abs(g(n)) for n in ("velocityx", "velocityy", "velocityz")[:ndim]]
            cmax = max(cmax, float(c.max()))
            umax = np.maximum(umax, [float(np.nanmax(v)) for v in u])
            rate = sum((c + u[d]) / dx[d] for d in range(ndim))
            umag = np.sqrt(sum(v ** 2 for v in u))
            e01 = np.clip(eta, 0, 1)
            lim = {"acoustic (per cell)": cfl / rate * SAFETY,
                   "largest c": 1.0 / np.maximum(c, 1e-300),            # ranked by c (smallest 1/c first)
                   "largest |u|": 1.0 / np.maximum(umag, 1e-300)}
            if cond == 2:
                m0 = e01 > amin; m1 = 1 - e01 > amin
                nth = np.where(m0, k0 * e01 / (r0 * cv0), 0.0)
                nth = np.maximum(nth, np.where(m1, k1 * (1 - e01) / (r1 * cv1), 0.0))
                lim["conduction"] = cflv * dx.min() ** 2 / (2 * ndim * nth + 1e-300) * SAFETY
            if kap > 0:
                gm = np.sqrt(sum(g(n) ** 2 for n in ("grad_etax", "grad_etay", "grad_etaz")[:ndim]))
                nus = kap * gm / np.maximum(rho, SMALL)
                lim["shell sub-step"] = cflv * dx.min() ** 2 / (2 * ndim * nus + 1e-300)
            if vsub and (mu0 or mu1):
                mu = e01 * mu0 + (1 - e01) * mu1; lam = e01 * l0 + (1 - e01) * l1
                nuL = (np.abs(lam) + (4 / 3 + ndim - 1) * np.abs(mu)) / np.maximum(rho, SMALL)
                lim["bulk visc sub-step"] = vcfl * dx.min() ** 2 / (2 * nuL + 1e-300)
            for key, Lm in lim.items():
                flat = Lm.ravel(); kk = np.argsort(flat)[:top]
                for q in kk:
                    iz, iy, ixx = np.unravel_index(q, Lm.shape)
                    x = lo + (blo + np.array([ixx, iy, iz])[:ndim] + 0.5) * dx
                    info = dict(r=np.linalg.norm(x - centre) * 1e6, x=x * 1e6, eta=eta[iz, iy, ixx], rho=rho[iz, iy, ixx],
                                re1=g("rho_eta1")[iz, iy, ixx], rg=r1[iz, iy, ixx] / a2[iz, iy, ixx],
                                p=g("pressure")[iz, iy, ixx], T=g("T")[iz, iy, ixx],
                                u=umag[iz, iy, ixx], c=c[iz, iy, ixx],
                                pl=p0r[iz, iy, ixx], pg=p1r[iz, iy, ixx])
                    best[key].append((flat[q], info))
        # acoustic exactly as the code: max c and max |u_d| over the level (they may come from different cells)
        dt_ac = cfl / sum((cmax + umax[d]) / dx[d] for d in range(ndim)) * SAFETY
        res["levels"].append(dict(lev=lev, ncell=ncell, nbox=nbox, fac=fac, dx=dx, rho_min=rho_min))
        print(f"\n  level {lev}: dx {dx[0]:.4g} m, {ncell} cells in {nbox} boxes, rho_min {rho_min:.4g}, "
              f"level dt x {fac:g} = coarse DT")
        print(f"    ACOUSTIC (as coded: max c {cmax:.5g} m/s + max |u| ({', '.join(f'{v:.4g}' for v in umax)}) m/s) "
              f"level dt {dt_ac:.4g} s -> coarse-equivalent {dt_ac*fac:.4g} s")
        res["rows"].append((dt_ac * fac, "acoustic", lev))
        res["causes"][("acoustic", lev)] = dict(cmax=cmax, umax=umax.max(), c_ref=(ref or {}).get("c_w"))
        for key in best:
            b = sorted(best[key], key=lambda z: z[0])[:top]
            if not b:
                continue
            if key == "largest c":
                print(f"    {key:20s} {1/b[0][0]:.5g} m/s")
            elif key == "largest |u|":
                print(f"    {key:20s} {1/b[0][0]:.5g} m/s")
            elif key in ("shell sub-step", "bulk visc sub-step"):
                cap = nsub_max if key.startswith("shell") else vmax
                N = max(1, int(math.ceil(dt_lev / b[0][0]))) if dt_lev else None
                res["nsc"][(key, lev)] = N
                tag = f"  -> N = {N} sub-steps per level step (cap {cap:g}){'  ** CLAMPED **' if N and N > cap else ''}" if N else ""
                print(f"    {key:20s} {b[0][0]:.4g} s{tag}  [sub-cycled: costs work, not global dt]")
            else:
                print(f"    {key:20s} level dt {b[0][0]:.4g} s  -> coarse-equivalent {b[0][0]*fac:.4g} s")
                if key == "conduction" and dtlim:
                    res["rows"].append((b[0][0] * fac, "conduction", lev))
                    res["causes"][("conduction", lev)] = b[0][1]
            for v, I in b[: (2 if key.startswith("largest") or "sub-step" in key else top)]:
                if key.startswith("largest"):
                    v = 1.0 / v                                  # print the speed itself
                xyz = " ".join(f"{q:7.3f}" for q in I['x'])
                print(f"      {v:10.3e}  r {I['r']:7.3f} um  xyz({xyz})  eta {I['eta']:9.3e}  rho {I['rho']:9.3e}  "
                      f"rho_gas {I['rg']:9.3e}  (a rho)_gas {I['re1']:9.3e}  p {I['p']:10.4g}  p_liq {I['pl']:10.4g}  "
                      f"p_gas {I['pg']:10.4g}  T {I['T']:8.1f}  |u| {I['u']:8.3g}  c {I['c']:8.4g}")
        if st_on and sig > 0:
            dcap = cflv * np.sqrt(rho_min * dx.min() ** 3 / (2 * np.pi * sig)) * SAFETY
            print(f"    {'capillary':20s} level dt {dcap:.4g} s  -> coarse-equivalent {dcap*fac:.4g} s  (uses rho_min {rho_min:.4g})")
            res["rows"].append((dcap * fac, "capillary", lev))
            res["causes"][("capillary", lev)] = dict(rho_min=rho_min)
    return res


# ----------------------------------------------------------------------------- cost + verdict
def report_cost(res, nsub, ranks, live):
    print("\n=== COST PER COARSE STEP")
    tot = sum(L["ncell"] * L["fac"] for L in res["levels"])
    for L in res["levels"]:
        w = L["ncell"] * L["fac"]
        bpr = f", {L['nbox']/ranks:.1f} boxes/rank" if ranks else ""
        print(f"    level {L['lev']}: {L['ncell']:>10d} cells x {L['fac']:>3g} steps = {w:>12.4g} cell-steps "
              f"({100*w/tot:5.1f}%), {L['nbox']} boxes{bpr}")
    print(f"    total {tot:.4g} cell-steps per coarse step (x3 RK stages, + the sub-cycles below on the band)")
    for (key, lev), N in sorted(res["nsc"].items(), key=lambda x: (x[0][1], x[0][0])):
        if N and N > 1:
            print(f"    {key} on level {lev}: N = {N} per level step")
    if live and live.get("rate"):
        s = 1 / live["rate"]
        print(f"    measured {s:.2f} s per coarse step -> {s/tot*1e6:.3g} us per cell-step"
              + (f" ({s/tot*ranks*1e6:.3g} rank-us per cell-step on {ranks} ranks)" if ranks else ""))
    Lf = res["levels"][-1]
    if ranks and Lf["nbox"] < ranks:
        print(f"    ** finest level has only {Lf['nbox']} boxes for {ranks} ranks: {ranks-Lf['nbox']} ranks idle on the finest"
              " level (it dominates the cost) -> smaller max_grid_size or fewer ranks")


def verdict(res, live, ref, nsub):
    print("\n=== VERDICT")
    rows = sorted(res["rows"])
    for v, key, lev in rows[:6]:
        print(f"    {v:.4g} s  {key} on level {lev}")
    dt = live.get("dt") if live else None
    v, key, lev = rows[0]
    if dt:
        ratio = dt / v
        print(f"    log coarse DT now {dt:.4g} s = {ratio:.2f} x the smallest rebuilt limit ({key}, level {lev})")
        if ratio < 0.5:
            print("    -> NONE of the rebuilt limits is that small: the binding one is dt_force (|source|/rho_min, not in the"
                  " plotfile), or the state changed since this plotfile (it is up to one plot interval older than the log)."
                  " Run again with --plot on a plotfile written closer to now, or turn on dt_diag=100 to print every limit.")
    if ref:
        print(f"    pure-water acoustic coarse dt {ref['dt_coarse']:.4g} s; the smallest limit is {ref['dt_coarse']/v:.1f}x below it")
    hint = {
        "acoustic": "acoustic: c or |u| far above undisturbed water (c_w {c_ref}) somewhere -- see 'largest c' / 'largest |u|'"
                    " cells above.  If those are trace-gas or band cells with unphysical |u| or c (hot rarefied gas), that is a"
                    " solver defect to fix, not a setting; if c ~ c_w and |u| is real (jet, collapse) the step is physical"
                    " and only cost (cells, ranks, box size) can be reduced.",
        "conduction": "conduction: k/(rho_k cv_k) is huge where the phase density rho_k = (a rho)_k / a_k is tiny -- usually"
                      " trace gas (1-eta just above thermal.alpha_min) or rarefied gas.  Check the cells listed under"
                      " 'conduction': if rho_gas there is << the core gas density, the limit comes from trace gas that"
                      " carries no real heat.  Raising thermal.alpha_min (deck) moves the cutoff; the physical fix is to"
                      " limit conduction in trace cells by the phase's actual heat capacity (gate flag).",
        "capillary": "capillary: sqrt(rho_min dx^3 / 2 pi sigma) uses the smallest mixture density on the level -- a near-vacuum"
                     " cell (rho_min listed per level) drags it down.  Look for cavitation / trace-gas tails.",
    }[key]
    c = res["causes"].get((key, lev), {})
    if key == "acoustic":
        hint = hint.format(c_ref=f"{ref['c_w']:.0f} m/s" if ref else "~1481 m/s")
        if c:
            print(f"    level {lev}: max c {c['cmax']:.4g} m/s, max |u| {c['umax']:.4g} m/s")
            if ref and c["cmax"] < 1.05 * ref["c_w"] and c["umax"] < 0.05 * ref["c_w"]:
                hint = ("acoustic at the undisturbed-water limit: the step is physical (c_water dx / 3 per level).  Only the"
                        " cost per step can be cut: fewer finest cells (n_error_buf, eta_refinement_criterion, blocking_factor),"
                        " better load balance (boxes per rank, max_grid_size), or more ranks.")
    print("    -> " + hint)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default="06Exp", choices=["06Exp", "06"], help="06Exp (full sphere + wall) or 06 (octant)")
    ap.add_argument("--case", default="HighPa", choices=["HighPa", "LowPa"])
    ap.add_argument("--dir", help="output (plotfile) directory")
    ap.add_argument("--log", help="stdout log of the run")
    ap.add_argument("--plot", help="plotfile to analyse (default: newest complete one in --dir)")
    ap.add_argument("--top", type=int, default=5, help="cells listed per limit")
    ap.add_argument("--watch", type=float, default=90, help="seconds to sample the log for the live step rate (0 = off)")
    ap.add_argument("--ranks", type=int, help="MPI ranks of the run (default: from slurm)")
    ap.add_argument("--no-plot", action="store_true", help="log part only")
    a = ap.parse_args()

    outdir = a.dir or find_outdir(a.series, a.case)
    kv = read_metadata(outdir)
    print(f"output dir {outdir}")
    plot_dt = num(kv, "amr.plot_dt", 0.0)
    L = int(num(kv, "amr.max_level", 0))
    ns = kv.get("amr.nsubsteps")
    nsub = [1] + ([int(float(ns.split()[0]))] * L if ns and len(ns.split()) == 1 else
                  [int(float(v)) for v in ns.split()] if ns else [int(num(kv, "amr.ref_ratio", 2))] * L)
    ref = reference_dt(kv, nsub)
    if ref:
        print(f"grid: max_level {L}, nsubsteps {nsub[1:]}, finest dx {ref['dx_f'][0]:.4g} m; water c {ref['c_w']:.1f} m/s"
              f" -> pure-water acoustic dt: finest {ref['dt_f']:.4g} s, coarse {ref['dt_coarse']:.4g} s")

    if a.log:
        log, how, ranks = a.log, "given", None
    else:
        log, how, ranks = find_log(a.series, a.case)
    ranks = a.ranks or ranks
    live = report_log(log, how, plot_dt, nsub, a.watch, ref) if log else {}
    if not log:
        print("no log found (pass --log)")
    if a.no_plot:
        return
    pf = a.plot or newest_plotfile(outdir)
    if not pf:
        print("no complete plotfile found"); return
    res = report_plotfile(pf, kv, a.top, live.get("dt"), nsub, ranks, ref)
    if res:
        report_cost(res, nsub, ranks, live)
        verdict(res, live, ref, nsub)
        print("\n    xyz in um from the bubble centre (origin); eta = liquid fraction.  p_gas << p_liq in a band cell ="
              " the relaxation is enforcing a large negative jump there.")


if __name__ == "__main__":
    with np.errstate(all="ignore"):           # trace-phase divides are expected; the code floors them the same way
        main()
