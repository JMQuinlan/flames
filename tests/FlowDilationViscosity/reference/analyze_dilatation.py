#!/usr/bin/env python3
"""
Analysis of the SINGLE-PHASE dilatational-viscosity pulse test (input_SinglePhase_Pulse_2D):
pressure probes from the plotfiles vs the sharp-membrane reference (acoustic_shell_1d.py).

For each run directory:
  * p'(t) at r = 0.5 R0 (inside: incident + membrane REFLECTION) and r = 2 R0 (outside:
    TRANSMISSION), sampled along the x axis, the y axis and the diagonal (anisotropy check);
  * error vs the reference with the deck's kappa_s;
  * kappa_eff: the membrane kappa that best fits the simulated transmitted + reflected
    signals (least squares over the probe records).  kappa_eff/kappa_s = 1 is exact.
A kappa_s = 0 run of the same mesh, if given with --baseline, is used to subtract the
mesh's own error (kappa_eff is then fitted to  sim - (sim0 - ref0)).

usage:
  analyze_dilatation.py RUN [RUN ...] [--input DECK] [--baseline RUN0] [--plot out.png]
  (deck values on the run's command line can be overridden with -s key=value)
"""
import argparse, glob, os, re, sys
import numpy as np
from scipy.optimize import minimize_scalar

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import acoustic_shell_1d as A1                                   # noqa: E402

DECK = os.path.normpath(os.path.join(_HERE, "..", "input_SinglePhase_Pulse_2D"))


def parse(path, over=()):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip().strip('"')
    for s in over:
        k, v = s.split("=", 1); kv[k.strip()] = v.strip()
    return kv


def params(kv):
    f = lambda k, d=None: float(kv[k]) if k in kv else d
    P = dict(R0=f("eta.ic.expression.constant.R0", 1.0), eps=f("eta.ic.expression.constant.epsilon"),
             kappa=f("shell.kappa_s", 0.0), rho=float(kv.get("density0.ic.expression.region0", "0.991")),
             gamma=f("eos0.gamma"), pi=f("eos0.p0"), p_inf=f("pressure0.ic.expression.constant.p_inf", 1.0),
             A=f("pressure0.ic.expression.constant.A"), w=f("pressure0.ic.expression.constant.w"))
    P["c"] = np.sqrt(P["gamma"] * (P["p_inf"] + P["pi"]) / P["rho"])
    return P


def run_overrides(run):
    """deck overrides from the run's log (the job runner writes them to cmdline.txt)"""
    p = os.path.join(run, "cmdline.txt")
    return open(p).read().split() if os.path.exists(p) else []


def probes(run, radii):
    import yt
    yt.funcs.mylog.setLevel(50)
    fs = sorted(glob.glob(os.path.join(run, "*cell")), key=lambda s: int(re.findall(r"(\d+)cell", s)[-1]))
    T, Pk = [], []
    for f in fs:
        ds = yt.load(f)
        cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        p = np.asarray(cg["boxlib", "pressure"])[:, :, 0]
        n = p.shape[0]; dx = float(ds.domain_width[0]) / n
        xc = (np.arange(n) + 0.5) * dx
        row = []
        for r in radii:
            ax = np.interp(r, xc, p[:, 0]); ay = np.interp(r, xc, p[0, :])
            s = r / np.sqrt(2.0); i = s / dx - 0.5; i0 = int(np.floor(i)); a = i - i0
            dg = ((1 - a) ** 2 * p[i0, i0] + a * (1 - a) * (p[i0 + 1, i0] + p[i0, i0 + 1]) + a * a * p[i0 + 1, i0 + 1])
            row.append((ax, ay, dg))
        T.append(float(ds.current_time)); Pk.append(row)
    return np.array(T), np.array(Pk)                   # Pk[t, probe, (x, y, diag)]


def reference(P, kappa, t_end, radii):
    return A1.solve(2, P["R0"], kappa, P["rho"], P["c"], P["rho"], P["c"],
                    lambda x: P["A"] * np.exp(-(x / P["w"]) ** 2), 12.0 * P["R0"], t_end, radii,
                    dr=P["R0"] / 400.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--input", default=DECK)
    ap.add_argument("-s", action="append", default=[])
    ap.add_argument("--baseline", default=None, help="kappa_s = 0 run on the same mesh")
    ap.add_argument("--radii", type=float, nargs=2, default=[0.5, 2.0])
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    radii = a.radii
    base = None
    if a.baseline:
        P0 = params(parse(a.input, run_overrides(a.baseline) + a.s))
        t0, s0 = probes(a.baseline, radii)
        r0 = reference(dict(P0), 0.0, t0[-1], radii)
        base = (t0, s0.mean(axis=2) - P0["p_inf"] - np.array([np.interp(t0, r0["t"], r0["p"][:, k]) for k in range(2)]).T)
    print("%-28s %6s %5s %7s | %9s %9s | %8s %8s | %7s" % ("run", "R0/dx", "eps/dx", "kappa", "Tpk/ref", "Rmin/ref",
                                                          "L2err", "anis", "k_eff/k"))
    fig = None
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for run in a.runs:
        P = params(parse(a.input, run_overrides(run) + a.s))
        t, S = probes(run, radii)
        sim = S.mean(axis=2) - P["p_inf"]                         # direction-averaged
        anis = np.max(np.abs(S - S.mean(axis=2, keepdims=True))) / P["A"]
        ref = reference(P, P["kappa"], t[-1], radii)
        rf = np.array([np.interp(t, ref["t"], ref["p"][:, k]) for k in range(2)]).T
        corr = sim.copy()
        if base is not None and len(base[0]) == len(t):
            corr = sim - base[1]
        scale = np.max(np.abs(rf))
        L2 = np.sqrt(np.mean((corr - rf) ** 2)) / scale
        # transmitted peak / reflected minimum ratios
        Tpk = corr[:, 1].max() / rf[:, 1].max()
        Rmn = corr[:, 0].min() / rf[:, 0].min()
        keff = np.nan
        if P["kappa"] > 0:
            def err(k):
                r = reference(P, k, t[-1], radii)
                q = np.array([np.interp(t, r["t"], r["p"][:, j]) for j in range(2)]).T
                return np.sum((corr - q) ** 2)
            keff = minimize_scalar(err, bounds=(0.0, 4.0 * P["kappa"]), method="bounded",
                                   options={"xatol": 1e-3 * P["kappa"]}).x / P["kappa"]
        dx = 6.0 * P["R0"] / float(re.findall(r"amr.n_cell=(\d+)", " ".join(run_overrides(run)))[0]) \
            if re.findall(r"amr.n_cell=(\d+)", " ".join(run_overrides(run))) else np.nan
        print("%-28s %6.1f %5.2f %7.3g | %9.4f %9.4f | %8.4f %8.4f | %7.4f"
              % (os.path.basename(run.rstrip("/")), P["R0"] / dx, P["eps"] / dx, P["kappa"], Tpk, Rmn, L2, anis, keff))
        if fig is not None:
            for k in range(2):
                axs[k].plot(t, corr[:, k], "o", ms=2, label=os.path.basename(run.rstrip("/")))
                axs[k].plot(ref["t"], ref["p"][:, k], "-", lw=0.8, color="k")
    if fig is not None:
        for k, ttl in enumerate(("r = %.1f R0 (reflection)" % radii[0], "r = %.1f R0 (transmission)" % radii[1])):
            axs[k].set_title(ttl); axs[k].set_xlabel("t"); axs[k].set_ylabel("p - p_inf")
        axs[1].legend(fontsize="x-small")
        fig.savefig(a.plot, dpi=140, bbox_inches="tight")
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
