#!/usr/bin/env python3
"""Analyze 3D OCTANT bubble runs (Sch20 Oscillating Marmottant Light_Mili family) against Keller-Miksis.

The bubble sits at the octant corner (0,0,0).  Each plotfile is read as ONE uniform block at the
finest level covering [0, L]^3 (L = box_R0 * R0, default 2.25 R0), so every quantity below is a
plain sum over that block; the octant is mirrored by the factor 8.

Per frame
  R_vol    radius from the gas volume, (3 * 8 int (1-eta) dV / 4 pi)^(1/3)
  R05      eta = 0.5 crossing along the x axis, the face diagonal (x=y) and the body diagonal
  Gamma    area-weighted mean  <Gamma> = int Gamma |grad eta| dV / int |grad eta| dV  vs (R0/R_vol)^2
  shell    shell material  int Gamma |grad eta| dV  relative to t=0  (should stay 1)
  gas mass int rho_eta1 dV relative to t=0
  core     mean gas pressure (eta < 0.01) vs adiabatic p_b (R0/R_vol)^(3 gamma); max |u| in the core
Scores (vs KM at the run's kappa_s, and vs KM kappa=0)
  extrema table (each min / max of R_vol): sim, KM, error in % of the KM first-collapse amplitude
  rms error over the first period and over the whole run, best-fit kappa_eff / kappa_s (first period)

Usage
  python3 analyze_octant_bubble.py OUTDIR [OUTDIR ...] [--np 8] [--box 2.25] [--show] [--png out.png]
Run parameters come from each OUTDIR/metadata.  Results are cached as OUTDIR_octant.npz next to the
output (re-runs only read new frames), so it can be run while a job is still going.
Needs numpy, scipy, matplotlib, yt, and marmottant_rpe_km.py from this directory.
"""
import argparse, glob, os, re, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_rpe_km import Shell, Liquid, Gas, solve_km   # noqa: E402


# ----------------------------------------------------------------------------- parameters
def metadata(out):
    """Run parameters from OUTDIR/metadata.  Handles both formats Alamo writes,
    'key = value' and 'key(nvals = N)  :: [v1] [v2]'; the last entry of a key wins
    (command-line overrides come after the deck), and a value that names another
    key is resolved (e.g. epsilon :: [eta.ic.expression.constant.epsilon])."""
    kv = {}
    rx = re.compile(r"^\s*([^\s(=#]+)\(nvals\s*=\s*\d+\)\s*::\s*(.*)$")
    with open(os.path.join(out, "metadata")) as f:
        for line in f:
            if line.lstrip().startswith("#"):
                continue
            m = rx.match(line)
            if m:
                kv[m.group(1)] = " ".join(re.findall(r"\[([^\]]*)\]", m.group(2)))
            elif "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.split("#")[0].strip().strip('"')
    for k, v in list(kv.items()):
        if v in kv and v != k:
            kv[k] = kv[v]
    return kv


def fnum(kv, keys, default=None):
    for k in keys:
        if k in kv:
            try:
                return float(kv[k].split()[0])
            except ValueError:
                pass
    if default is None:
        raise KeyError("none of %s in metadata" % (keys,))
    return default


def frames(out):
    fl = glob.glob(os.path.join(out, "*cell"))
    return sorted(fl, key=lambda s: int(re.findall(r"(\d+)cell$", s)[-1]))


# ----------------------------------------------------------------------------- one frame
def one_frame(args):
    f, R0, L = args
    import yt
    yt.funcs.mylog.setLevel(50)
    from scipy.ndimage import map_coordinates
    ds = yt.load(f)
    lev = ds.index.max_level
    dx = float(ds.index.get_smallest_dx())
    n = int(np.ceil(L / dx))
    cg = ds.covering_grid(lev, left_edge=ds.domain_left_edge, dims=[n, n, n])
    g = lambda s: cg["boxlib", s].d
    have = {fn for (_, fn) in ds.field_list}
    e = np.clip(g("eta"), 0, 1)
    dV = dx ** 3
    x = (np.arange(n) + 0.5) * dx
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    out = dict(t=float(ds.current_time), dx=dx, lev=lev, n=n)
    out["Vgas"] = 8 * np.sum(1 - e) * dV
    out["edge_gas"] = float((1 - e)[-1, :, :].max())                 # must be ~0: box holds the bubble
    # eta = 0.5 crossings along three directions
    s = np.linspace(0, L * 0.98, 2000)
    for name, d in (("x", (1, 0, 0)), ("xy", (1, 1, 0)), ("xyz", (1, 1, 1))):
        d = np.array(d, float) / np.linalg.norm(d)
        pts = np.array([s * d[0], s * d[1], s * d[2]]) / dx - 0.5
        v = map_coordinates(e, pts, order=1, mode="nearest")
        i = np.where((v[:-1] - 0.5) * (v[1:] - 0.5) <= 0)[0]
        out["R05_" + name] = (s[i[0]] + (0.5 - v[i[0]]) * (s[i[0] + 1] - s[i[0]]) / (v[i[0] + 1] - v[i[0]])) if len(i) else np.nan
    # shell: the transported Gamma is the 'shell' field ('Gamma' is a diagnostic, 0 at t=0)
    G = g("shell") if "shell" in have else None
    gm = np.linalg.norm(np.array(np.gradient(e, dx)), axis=0)       # |grad eta| on the block
    out["area"] = 8 * np.sum(gm) * dV
    if G is not None:
        out["Gmean"] = np.sum(G * gm) / np.sum(gm)
        out["Gmat"] = 8 * np.sum(G * gm) * dV
        b = (e > 0.45) & (e < 0.55)
        out["G05"] = float(np.median(G[b])) if b.any() else np.nan
        bb = (e > 1e-3) & (e < 0.999)
        out["Gspread"] = float(np.std(G[bb]) / max(np.mean(G[bb]), 1e-30)) if bb.any() else np.nan
    # gas
    if "rho_eta1" in have:
        out["mgas"] = 8 * np.sum(g("rho_eta1")) * dV
    core = e < 0.01
    if core.any():
        out["pcore"] = float(np.mean(g("pressure")[core]))
        u = np.sqrt(sum(g("velocity" + c) ** 2 for c in "xyz"))
        out["umax_core"] = float(u[core].max())
        ur = (g("velocityx") * X + g("velocityy") * Y + g("velocityz") * Z) / np.maximum(r, 1e-30)
        bnd = (e > 0.45) & (e < 0.55)
        out["ur05"] = float(np.mean(ur[bnd])) if bnd.any() else np.nan
    else:
        out["pcore"] = out["umax_core"] = out["ur05"] = np.nan
    return out


def series(out, R0, L, nproc):
    cache = out.rstrip("/") + "_octant.npz"
    fl = frames(out)
    done = {}
    if os.path.exists(cache):
        try:
            for d in np.load(cache, allow_pickle=True)["F"]:
                done[d["file"]] = d
        except Exception:
            done = {}
    todo = [f for f in fl if os.path.basename(f) not in done]
    # a frame touched in the last minute may still be being written
    import time
    todo = [f for f in todo if time.time() - max(os.path.getmtime(q) for q in glob.glob(os.path.join(f, "**"), recursive=True)) > 60]
    if todo:
        print("  %s: reading %d new frame(s) (%d cached) on %d proc" % (os.path.basename(out), len(todo), len(done), nproc), flush=True)
        jobs = [(f, R0, L) for f in todo]
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as p:
                res = p.map(one_frame, jobs, chunksize=1)
        else:
            res = [one_frame(j) for j in jobs]
        for f, d in zip(todo, res):
            d["file"] = os.path.basename(f)
            done[d["file"]] = d
        F = sorted(done.values(), key=lambda d: d["t"])
        np.savez(cache, F=np.array(F, dtype=object))
    F = sorted(done.values(), key=lambda d: d["t"])
    keys = sorted({k for d in F for k in d if k != "file"})
    return {k: np.array([d.get(k, np.nan) for d in F], dtype=float) for k in keys}


# ----------------------------------------------------------------------------- KM
def km(kv, kappa, R0, t):
    rho = fnum(kv, ["density0.ic.expression.region0"])
    g0, pi0 = fnum(kv, ["eos0.gamma"]), fnum(kv, ["eos0.p0"])
    p_inf = fnum(kv, ["pressure0.ic.expression.constant.p_inf"])
    p_b = fnum(kv, ["pressure1.ic.expression.constant.p_b"])
    c = np.sqrt(g0 * (p_inf + pi0) / rho)
    sh = Shell(chi=0.0, sigma_water=fnum(kv, ["sigma"], 0.0), kappa_s=kappa)
    liq = Liquid(rho=rho, mu=0.0, c=c, p_inf=p_inf)
    gas = Gas(p_g0=p_b, kappa_g=fnum(kv, ["eos1.gamma"]))
    _, RR, _ = solve_km(R0, 0.0, (0.0, t[-1]), sh, liq, gas, t_eval=t)
    return RR


def extrema(t, R):
    i = [k for k in range(1, len(R) - 1) if (R[k] - R[k - 1]) * (R[k + 1] - R[k]) < 0]
    out = []
    for k in i:   # parabolic refinement
        a, b, c = R[k - 1], R[k], R[k + 1]
        den = a - 2 * b + c
        dk = 0.5 * (a - c) / den if den != 0 else 0.0
        out.append((t[k] + dk * (t[k + 1] - t[k - 1]) / 2, b - 0.25 * (a - c) * dk, "min" if den > 0 else "max"))
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdirs", nargs="+")
    ap.add_argument("--np", type=int, default=1, help="processes for frame reading")
    ap.add_argument("--box", type=float, default=2.25, help="analysis box edge in units of R0")
    ap.add_argument("--png", default="octant_bubble.png")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    from scipy.optimize import minimize_scalar

    runs = []
    for out in a.outdirs:
        out = out.rstrip("/")
        kv = metadata(out)
        R0n = fnum(kv, ["eta.ic.expression.constant.R0", "jump.R0", "marmottant.R0"])
        kap = fnum(kv, ["shell.kappa_s"], 0.0)
        S = series(out, R0n, a.box * R0n, a.np)
        if len(S.get("t", [])) < 3:
            print("%s: fewer than 3 frames, skipped" % out); continue
        t = S["t"]; Rv = (3 * S["Vgas"] / (4 * np.pi)) ** (1 / 3); R0 = Rv[0]
        Rk = km(kv, kap, R0, t); Rk0 = km(kv, 0.0, R0, t)
        runs.append(dict(name=os.path.basename(out), kv=kv, S=S, t=t, Rv=Rv, R0=R0, Rk=Rk, Rk0=Rk0, kap=kap))

    for rr in runs:
        t, Rv, R0, Rk, Rk0, S, kv, kap = rr["t"], rr["Rv"], rr["R0"], rr["Rk"], rr["Rk0"], rr["S"], rr["kv"], rr["kap"]
        amp = R0 - Rk.min() if t[-1] > t[np.argmin(Rk)] else R0 - Rk[-1]
        eps = fnum(kv, ["epsilon", "eta.ic.expression.constant.epsilon"], np.nan)
        print("\n" + "=" * 110)
        print("%s   t_end %.3f ms (%d frames)   kappa_s %.3g   eps %.3g m  eps/dx %.2f   finest dx %.3g m  lev %d" %
              (rr["name"], t[-1] * 1e3, len(t), kap, eps, eps / S["dx"][-1], S["dx"][-1], S["lev"][-1]))
        print("  mode: shell.gamma_uniform=%s shell.kappa_s_rate=%s  Limiter=%s  box edge max gas %.1e" %
              (kv.get("shell.gamma_uniform", "?"), kv.get("shell.kappa_s_rate", "?"), kv.get("Limiter.type", "?"), np.nanmax(S["edge_gas"])))
        # extrema
        es, ek = extrema(t, Rv), extrema(t, Rk)
        print("  extrema of R_vol/R0   (error in %% of KM first-collapse amplitude %.4f R0)" % (amp / R0))
        print("    %-4s %9s %9s | %9s %9s | %8s %8s" % ("", "t_sim ms", "R_sim", "t_KM ms", "R_KM", "dR %amp", "dt %"))
        for k in range(min(len(es), len(ek), 12)):
            (ts, Rs, ks), (tk, Rkk, kk) = es[k], ek[k]
            print("    %-4s %9.3f %9.4f | %9.3f %9.4f | %+8.2f %+8.2f%s" % (ks, ts * 1e3, Rs / R0, tk * 1e3, Rkk / R0,
                  100 * (Rs - Rkk) / amp, 100 * (ts - tk) / tk, "" if ks == kk else "  (type mismatch)"))
        # rms errors
        T1 = ek[1][0] if len(ek) > 1 else t[-1]         # end of first period = first KM max after the min
        m1 = t <= T1
        rms = lambda m, ref: 100 * np.sqrt(np.mean((Rv[m] - ref[m]) ** 2)) / amp
        print("  rms(R_vol - KM) / amp:  first period %.2f%%   whole run %.2f%%   [vs KM kappa=0: %.2f%% / %.2f%%]" %
              (rms(m1, Rk), rms(t > -1, Rk), rms(m1, Rk0), rms(t > -1, Rk0)))
        if kap > 0 and m1.sum() > 5:
            f = lambda k: np.mean((km(kv, k, R0, t[m1]) - Rv[m1]) ** 2)
            kb = minimize_scalar(f, bounds=(0.0, 2.5 * kap), method="bounded", options=dict(xatol=1e-3 * kap)).x
            print("  best-fit kappa_eff / kappa_s (first period): %.3f" % (kb / kap))
        # shape, shell, gas
        Rx, Rxy, Rxyz = S["R05_x"], S["R05_xy"], S["R05_xyz"]
        asp = (np.nanmax([Rx, Rxy, Rxyz], 0) - np.nanmin([Rx, Rxy, Rxyz], 0)) / Rv
        print("  asphericity max |R05 axis/face/body spread| / R_vol: %.2f%% (at %.2f ms);  R05_x / R_vol at R_min %.4f" %
              (100 * np.nanmax(asp), 1e3 * t[np.nanargmax(asp)], Rx[np.argmin(Rv)] / Rv.min()))
        if "Gmean" in S:
            Gex = (R0 / Rv) ** 2
            print("  Gamma: <Gamma>/(R0/R_vol)^2 at R_min %.3f  min %.3f  max %.3f | Gamma(eta .5)/exact at R_min %.3f | in-band spread (std/mean) max %.3f" %
                  ((S["Gmean"] / Gex)[np.argmin(Rv)], np.nanmin(S["Gmean"] / Gex), np.nanmax(S["Gmean"] / Gex),
                   (S["G05"] / Gex)[np.argmin(Rv)], np.nanmax(S["Gspread"])))
            print("  shell material int Gamma|grad eta| / t0: min %.4f  max %.4f  end %.4f   area int|grad eta| / 4 pi R_vol^2 at R_min %.4f" %
                  (np.nanmin(S["Gmat"] / S["Gmat"][0]), np.nanmax(S["Gmat"] / S["Gmat"][0]), S["Gmat"][-1] / S["Gmat"][0],
                   (S["area"] / (4 * np.pi * Rv ** 2))[np.argmin(Rv)]))
        if "mgas" in S:
            mg = S["mgas"] / S["mgas"][0] - 1
            print("  gas mass change: max |dm| %.2f%%  end %+.2f%%" % (100 * np.nanmax(abs(mg)), 100 * mg[-1]))
        p_b = fnum(kv, ["pressure1.ic.expression.constant.p_b"]); g1 = fnum(kv, ["eos1.gamma"])
        pad = p_b * (R0 / Rv) ** (3 * g1)
        print("  gas core: p_core/p_adiabatic at R_min %.3f  (range %.3f..%.3f);  max |u| in core %.2f m/s (at %.2f ms)" %
              ((S["pcore"] / pad)[np.argmin(Rv)], np.nanmin(S["pcore"] / pad), np.nanmax(S["pcore"] / pad),
               np.nanmax(S["umax_core"]), 1e3 * t[np.nanargmax(S["umax_core"])]))
        rr["asp"] = asp

    # ------------------------------------------------------------------ plots
    import matplotlib
    if not a.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    ax = ax.ravel()
    for i, rr in enumerate(runs):
        c = "C%d" % i; t = rr["t"] * 1e3; R0 = rr["R0"]; S = rr["S"]
        if i == 0 or not np.allclose(rr["Rk"], runs[0]["Rk"], rtol=1e-4):
            ax[0].plot(t, rr["Rk"] / R0, "k-", lw=1, label="KM, kappa_s %.3g" % rr["kap"])
            ax[0].plot(t, rr["Rk0"] / R0, "k:", lw=1, label="KM, kappa_s 0")
        ax[0].plot(t, rr["Rv"] / R0, color=c, label=rr["name"])
        amp = R0 - rr["Rk"].min()
        ax[1].plot(t, 100 * (rr["Rv"] - rr["Rk"]) / amp, color=c)
        if "Gmean" in S:
            ax[2].plot(t, S["Gmean"] / (R0 / rr["Rv"]) ** 2, color=c, label=rr["name"] + " <Gamma>")
            ax[2].plot(t, S["G05"] / (R0 / rr["Rv"]) ** 2, color=c, ls="--", lw=.8)
            ax[3].plot(t, S["Gmat"] / S["Gmat"][0], color=c, label="shell material")
        if "mgas" in S:
            ax[3].plot(t, S["mgas"] / S["mgas"][0], color=c, ls=":", label="gas mass")
        ax[4].plot(t, 100 * rr["asp"], color=c)
        ax[5].plot(t, S["umax_core"], color=c)
    ax[0].set_ylabel(r"$R_{vol}/R_0$"); ax[0].legend(fontsize=8); ax[0].set_title("radius vs Keller-Miksis")
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_ylabel("(R_vol - KM) / KM amplitude [%]"); ax[1].set_title("error")
    ax[2].axhline(1, color="k", lw=.8); ax[2].set_ylabel(r"$\Gamma/(R_0/R_{vol})^2$"); ax[2].set_title(r"$\Gamma$ tracking (solid: area mean, dashed: $\eta$=0.5)")
    ax[3].axhline(1, color="k", lw=.8); ax[3].set_ylabel("relative to t=0"); ax[3].set_title("conservation (solid: shell material, dotted: gas mass)")
    ax[4].set_ylabel("R05 spread / R_vol [%]"); ax[4].set_title("asphericity (axis / face / body diagonal)")
    ax[5].set_ylabel("max |u| in gas core [m/s]"); ax[5].set_title("gas-core velocity (instability flag)")
    for q in ax[4:]:
        q.set_xlabel("t [ms]")
    fig.tight_layout(); fig.savefig(a.png, dpi=150)
    print("\nfigure: %s" % os.path.abspath(a.png))
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
