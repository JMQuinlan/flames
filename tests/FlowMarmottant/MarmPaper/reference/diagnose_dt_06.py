#!/usr/bin/env python3
"""Find what is holding down the time step of a 06 Cattaneo run.

Defaults to the INCLINE opt1 run of the chosen case:
    output dir  /mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/MarmPaper/output_06_Cattaneo_<case>_opt1*
    log         the StdOut of the running slurm job whose script is 06_Cattaneo_<case>_opt1.sh
                (fallback: newest /mmfs1/home/ttryon/FLAMES_out/flame_*_stdout)

Two parts:
  1. LOG: dt history, when dt dropped, which AMR level sets dt (the "lev=" lines),
     sub-cycle warnings, and the live step rate (samples the log for --watch seconds).
  2. PLOTFILE: rebuilds every dt limit of Hydro2::TimeStepComplete cell by cell from the newest
     plotfile (acoustic, conduction, shell-viscous sub-cycle count, capillary) and prints the
     cells that set each limit, with their state (eta, densities, p, T, |u|, radius).
     The force limit (dt_force) needs the source term, which is not in the plotfile.

Note: the log's DT is the COARSE (level 0) step.  Level L runs at DT / prod(nsubsteps[1..L]).

Usage (on INCLINE, no arguments needed for the 160 kPa opt1 run):
    python3 diagnose_dt_06.py
    python3 diagnose_dt_06.py --case LowPa
    python3 diagnose_dt_06.py --dir OUTDIR --log LOGFILE --plot OUTDIR/123456cell --top 8 --watch 120
Needs only numpy (no yt).
"""
import argparse
import glob
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
def find_outdir(case):
    c = [d for d in glob.glob(f"{ROOT}/output_06_Cattaneo_{case}_opt1*") if os.path.isdir(d) and ".old" not in d]
    if not c:
        sys.exit(f"no output dir matching {ROOT}/output_06_Cattaneo_{case}_opt1*  (use --dir)")
    return max(c, key=os.path.getmtime)


def find_log(case):
    script = f"06_Cattaneo_{case}_opt1"
    try:
        ids = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        for jid in ids:
            info = subprocess.run(["scontrol", "show", "job", jid], capture_output=True, text=True, timeout=30).stdout
            cmd = re.search(r"Command=(\S+)", info)
            out = re.search(r"StdOut=(\S+)", info)
            if cmd and out and script in cmd.group(1) and os.path.isfile(out.group(1)):
                return out.group(1), f"slurm job {jid}"
    except (OSError, subprocess.SubprocessError):
        pass
    c = glob.glob(f"{LOGDIR}/flame_*_stdout")
    if not c:
        return None, "none found"
    return max(c, key=os.path.getmtime), "NEWEST stdout in FLAMES_out (no running job matched -- check it is the right run, or pass --log)"


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


# ----------------------------------------------------------------------------- log
def scan_log(path):
    steps, times, dts = [], [], []
    lev_lines, warn, nan_lines = [], {}, 0
    pat = re.compile(rb"STEP (\d+) ends\. TIME = (\S+) DT = (\S+)")
    lev = re.compile(rb"lev=(\d+) +(\S+) \((\d+)\)")
    with open(path, "rb") as f:
        for raw in f:
            if b"ends. TIME" in raw:
                m = pat.search(raw)
                if m:
                    steps.append(int(m.group(1))); times.append(float(m.group(2))); dts.append(float(m.group(3)))
            elif b"lev=" in raw and b"-->" not in raw:
                m = lev.search(ANSI.sub(b"", raw))
                if m:
                    lev_lines.append((int(m.group(1)), float(m.group(2)), int(m.group(3))))
                    if len(lev_lines) > 200:
                        del lev_lines[:100]
            elif b"Subcycle" in raw or b"exceeds" in raw or b"Clamping" in raw:
                s = ANSI.sub(b"", raw).decode(errors="replace").strip()
                key = re.sub(r"[-+]?\d[\d.e+-]*", "#", s)[:120]
                n, _ = warn.get(key, (0, s))
                warn[key] = (n + 1, s)
            elif b"nan" in raw.lower():
                nan_lines += 1
    return np.array(steps), np.array(times), np.array(dts), lev_lines, warn, nan_lines


def last_step(path, nbytes=2_000_000):
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell(); f.seek(max(0, size - nbytes))
        tail = f.read()
    m = re.findall(rb"STEP (\d+) ends\. TIME = (\S+) DT = (\S+)", tail)
    return (int(m[-1][0]), float(m[-1][1]), float(m[-1][2])) if m else None


def report_log(path, how, plot_dt, nsub, watch):
    nsub_prod = float(np.prod(nsub))
    print(f"\n=== LOG  {path}\n    ({how})")
    st = os.stat(path)
    print(f"    size {st.st_size/1e9:.2f} GB, last write {time.ctime(st.st_mtime)} ({(time.time()-st.st_mtime)/60:.1f} min ago)")
    steps, t, dt, lev_lines, warn, nnan = scan_log(path)
    if len(steps) == 0:
        print("    no STEP lines"); return None
    print(f"    steps {steps[0]}..{steps[-1]}, sim time {t[-1]*1e6:.6f} us, coarse DT now {dt[-1]:.4g} s "
          f"(finest level {dt[-1]/nsub_prod:.4g} s)")
    n = len(dt)
    ref = np.median(dt[: max(10, n // 2)])
    print(f"    median coarse DT over the first half of the run {ref:.4g} s -> now {ref/dt[-1]:.0f}x smaller")
    skip = min(n - 1, 200)                     # start-up ramp from dt = 1e-12
    low = skip + np.where(dt[skip:] < 0.3 * ref)[0]
    if len(low):
        i0 = low[0]
        print(f"    DT first fell below 0.3x median at step {steps[i0]}, t = {t[i0]*1e6:.6f} us "
              f"({len(low)} of the {n-i0} steps since are below it)")
    # dt trend over the last stretch
    k = np.unique(np.linspace(max(0, n - 20000), n - 1, 12).astype(int))
    print("    recent  step        t [us]          DT")
    for i in k:
        print(f"           {steps[i]:<10d}  {t[i]*1e6:.7f}   {dt[i]:.4g}")
    if plot_dt > 0:
        nxt = (np.floor(t[-1] / plot_dt) + 1) * plot_dt
        print(f"    next plotfile at {nxt*1e6:.4f} us: {(nxt-t[-1])/dt[-1]:.0f} more coarse steps at the current DT")
    if lev_lines:
        L = max(l for l, _, _ in lev_lines)
        last = {}
        for l, d, ns in lev_lines:
            last[l] = (d, ns)
        print("    last per-level dt limits (dt of that level; x nsubsteps above it -> coarse-equivalent):")
        for l in sorted(last):
            d, ns = last[l]
            print(f"      lev {l}: {d:.4g} s  -> coarse-equivalent {d*float(np.prod(nsub[1:l + 1])):.4g} s")
        bind = min(last, key=lambda l: last[l][0] * float(np.prod(nsub[1:l + 1])))
        print(f"      binding level: lev {bind} (smallest coarse-equivalent); the plotfile part names the limit")
    else:
        print("    no 'lev=' lines (dynamictimestep.verbose off or old binary)")
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
            print(f"    live: {rate*60:.2f} coarse steps/min, {rate and 1/rate:.2f} s/step, {simrate*3600*1e9:.3f} ns sim per wall hour")
            if plot_dt > 0:
                nxt = (np.floor(b[1] / plot_dt) + 1) * plot_dt
                print(f"    next plotfile in ~{(nxt-b[1])/simrate/3600:.1f} h at this rate")
        else:
            print("    live: NO new steps in the window -> the job is hung, not just slow")
    return dt[-1]


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


def report_plotfile(pf, kv, top, dt_log, nsub):
    names, ndim, tcur, finest, lo, dxs = read_header(pf)
    ix = {n: i for i, n in enumerate(names)}
    need = ["eta", "rho_eta0", "rho_eta1", "density", "energy0", "energy1", "velocityx", "velocityy", "velocityz",
            "grad_etax", "grad_etay", "grad_etaz", "pressure", "T"]
    miss = [n for n in need if n not in ix]
    if miss:
        print(f"    plotfile lacks {miss}; cannot rebuild limits"); return
    g0, pi0 = num(kv, "eos0.gamma", 2.35), num(kv, "eos0.p0", 9.311895e8)
    g1, pi1 = num(kv, "eos1.gamma", 1.4), num(kv, "eos1.p0", 0.0)
    cv0, cv1 = num(kv, "eos0.cv", 2343.6161), num(kv, "eos1.cv", 714.286)
    k0, k1 = num(kv, "thermal.k0", 0.6), num(kv, "thermal.k1", 0.026)
    cond = num(kv, "thermal.conduction", 1); amin = num(kv, "thermal.alpha_min", 1e-3)
    cfl = num(kv, "cfl", 0.9); cflv = num(kv, "cfl_v", cfl)
    kap = abs(num(kv, "shell.kappa_s", 0) - num(kv, "shell.mu_s", 0)) + 2 * abs(num(kv, "shell.mu_s", 0))
    sig = num(kv, "sigma", 0.0); st_on = num(kv, "apply_surface_tension", 0)
    nsub_max = num(kv, "shell.kappa_s_subcycle_max", 512)
    print(f"\n=== PLOTFILE {pf}\n    t = {tcur*1e6:.6f} us, finest level {finest}; "
          f"gas gamma {g1} cv {cv1} k1 {k1}; cfl {cfl} cfl_v {cflv}; alpha_min {amin}")
    rows = []          # (dt_coarse_equiv, limit, lev, info)
    rho_min_all = np.inf
    for lev in range(finest + 1):
        dx = dxs[lev]; fac = float(np.prod(nsub[1:lev + 1]))   # level dt x fac = coarse-equivalent dt
        best = {"acoustic": [], "conduction": [], "shell sub-cycle": []}
        rho_min = np.inf; ncell = 0
        for blo, A in iter_fabs(pf, lev):
            g = lambda n: A[ix[n]]
            eta = g("eta"); a1 = np.clip(eta, 1e-30, 1 - 1e-30); a2 = 1 - a1
            r0 = np.maximum(g("rho_eta0"), SMALL); r1 = np.maximum(g("rho_eta1"), SMALL)
            rho = g("density"); rho_min = min(rho_min, rho.min()); ncell += rho.size
            p0 = np.maximum((g0 - 1) * g("energy0") / a1 - g0 * pi0, -pi0 + SMALL)
            p1 = np.maximum((g1 - 1) * g("energy1") / a2 - g1 * pi1, -pi1 + SMALL)
            sm = lambda a: (lambda w: w * w * (3 - 2 * w))(np.clip((a - 1e-2) / (1e-1 - 1e-2), 0, 1))
            w0, w1 = sm(a1), sm(a2)
            p0, p1 = w0 * p0 + (1 - w0) * p1, w1 * p1 + (1 - w1) * p0
            c0 = np.sqrt(g0 * np.maximum(p0 + pi0, SMALL) / (r0 / a1))
            c1 = np.sqrt(g1 * np.maximum(p1 + pi1, SMALL) / (r1 / a2))
            Y0 = r0 / np.maximum(rho, SMALL)
            c = np.sqrt(Y0 * c0 ** 2 + (1 - Y0) * c1 ** 2)
            u = [np.abs(g(n)) for n in ("velocityx", "velocityy", "velocityz")[:ndim]]
            rate = sum((c + u[d]) / dx[d] for d in range(ndim))
            lim = {"acoustic": cfl / rate * SAFETY}
            if cond:
                m0 = np.clip(eta, 0, 1) > amin; m1 = 1 - np.clip(eta, 0, 1) > amin
                nth = np.where(m0, k0 * np.clip(eta, 0, 1) / (r0 * cv0), 0.0)
                nth = np.maximum(nth, np.where(m1, k1 * (1 - np.clip(eta, 0, 1)) / (r1 * cv1), 0.0))
                lim["conduction"] = cflv * dx.min() ** 2 / (2 * ndim * nth + 1e-300) * SAFETY
            if kap > 0:
                gm = np.sqrt(sum(g(n) ** 2 for n in ("grad_etax", "grad_etay", "grad_etaz")[:ndim]))
                nus = kap * gm / np.maximum(rho, SMALL)
                lim["shell sub-cycle"] = cflv * dx.min() ** 2 / (2 * ndim * nus + 1e-300)   # sub-step, not in global dt
            for key, L in lim.items():
                flat = L.ravel(); kk = np.argsort(flat)[:top]
                for q in kk:
                    iz, iy, ixx = np.unravel_index(q, L.shape)
                    x = lo + (blo + np.array([ixx, iy, iz])[:ndim] + 0.5) * dx
                    info = dict(r=np.linalg.norm(x) * 1e6, x=x * 1e6, eta=eta[iz, iy, ixx], rho=rho[iz, iy, ixx],
                                re1=g("rho_eta1")[iz, iy, ixx], rg=r1[iz, iy, ixx] / a2[iz, iy, ixx],
                                p=g("pressure")[iz, iy, ixx], T=g("T")[iz, iy, ixx],
                                u=np.sqrt(sum(v[iz, iy, ixx] ** 2 for v in u)), c=c[iz, iy, ixx])
                    best[key].append((flat[q], info))
        rho_min_all = min(rho_min_all, rho_min)
        print(f"\n  level {lev}: dx {dx[0]:.4g} m, {ncell} cells, rho_min {rho_min:.4g}, level dt x {fac:g} = coarse DT")
        for key in best:
            b = sorted(best[key], key=lambda z: z[0])[:top]
            if not b:
                continue
            if key == "shell sub-cycle":
                dtl = (dt_log / fac) if dt_log else None
                tag = (f"  -> N = {max(1, int(np.ceil(dtl/b[0][0])))} sub-steps per level step (cap {nsub_max:g})" if dtl else "")
                print(f"    {key:16s} sub-step {b[0][0]:.4g} s{tag}  [not in the global dt]")
            else:
                print(f"    {key:16s} level dt {b[0][0]:.4g} s  -> coarse-equivalent {b[0][0]*fac:.4g} s")
                rows.append((b[0][0] * fac, key, lev))
            for v, I in b:
                print(f"      {v:10.3e}  r {I['r']:7.3f} um  eta {I['eta']:9.3e}  rho {I['rho']:9.3e}  "
                      f"rho_gas {I['rg']:9.3e}  (a rho)_gas {I['re1']:9.3e}  p {I['p']:10.4g}  T {I['T']:8.1f}  "
                      f"|u| {I['u']:8.3g}  c {I['c']:8.4g}")
        if st_on and sig > 0:
            dcap = cflv * np.sqrt(rho_min * dx.min() ** 3 / (2 * np.pi * sig)) * SAFETY
            print(f"    capillary        level dt {dcap:.4g} s  -> coarse-equivalent {dcap*fac:.4g} s  (uses rho_min {rho_min:.4g})")
            rows.append((dcap * fac, "capillary", lev))
    print("\n=== SUMMARY (coarse-equivalent dt; smallest first)")
    for v, key, lev in sorted(rows)[:6]:
        print(f"    {v:.4g} s  {key} on level {lev}")
    if dt_log:
        print(f"    log coarse DT now {dt_log:.4g} s.  If no limit above is that small, the binding one is dt_force"
              f" (|source|/rho_min, not in the plotfile) or the state changed since this plotfile.")
    print("    (plotfile state can be up to one plot interval older than the log)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="HighPa", choices=["HighPa", "LowPa"])
    ap.add_argument("--dir", help="output (plotfile) directory")
    ap.add_argument("--log", help="stdout log of the run")
    ap.add_argument("--plot", help="plotfile to analyse (default: newest complete one in --dir)")
    ap.add_argument("--top", type=int, default=5, help="cells listed per limit")
    ap.add_argument("--watch", type=float, default=60, help="seconds to sample the log for the live step rate (0 = off)")
    ap.add_argument("--no-plot", action="store_true", help="log part only")
    a = ap.parse_args()

    outdir = a.dir or find_outdir(a.case)
    kv = read_metadata(outdir)
    print(f"output dir {outdir}")
    plot_dt = num(kv, "amr.plot_dt", 0.0)
    L = int(num(kv, "amr.max_level", 0))
    ns = kv.get("amr.nsubsteps")
    nsub = [1] + ([int(float(ns.split()[0]))] * L if ns and len(ns.split()) == 1 else
                  [int(float(v)) for v in ns.split()] if ns else [int(num(kv, "amr.ref_ratio", 2))] * L)

    log, how = (a.log, "given") if a.log else find_log(a.case)
    dt_log = report_log(log, how, plot_dt, nsub, a.watch) if log else None
    if not log:
        print("no log found (pass --log)")
    if not a.no_plot:
        pf = a.plot or newest_plotfile(outdir)
        if pf:
            report_plotfile(pf, kv, a.top, dt_log, nsub)
        else:
            print("no complete plotfile found")


if __name__ == "__main__":
    with np.errstate(all="ignore"):           # trace-phase divides are expected; the code floors them the same way
        main()
