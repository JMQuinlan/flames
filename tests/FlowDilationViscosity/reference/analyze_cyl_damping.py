#!/usr/bin/env python3
"""
MULTIPHASE dilatational-viscosity test (input_Multiphase_*_2D): fit kappa_eff for 2D
quarter-domain cylinder runs against the incompressible cylindrical Rayleigh-Plesset model

    rho [ (R R'' + R'^2) ln(R_inf/R) - R'^2/2 (1 - R^2/R_inf^2) ] = p_g(R) - kappa_s R'/R^2 - 2 mu R'/R - p_inf
    p_g = p_b (R0/R)^(2 gamma_g),   R_inf = equal-area circle of the quarter box (L 2/sqrt(pi))

(bin/ShellRework/scripts/rp2d.py physics, every parameter read from the deck).
R(t) = sqrt(4 A_gas/pi), A_gas = quarter-domain integral of (1 - eta), rescaled to R0 at t = 0.

usage:  analyze_cyl_damping.py RUN [RUN ...] --input DECK [--baseline RUN0] [-s key=val]
Per run: R_max / R_min vs the model, rms(f = 1), kappa_eff/kappa_s.  With --baseline (the
kappa_s = 0 run on the same mesh) the mesh's own damping is reported as an equivalent
kappa (kappa_num) and kappa_eff is also given after removing it (k_eff - k_num).
Run overrides are read from RUN/cmdline.txt if present (written by run_sweep.sh).
"""
import argparse, glob, os, re, sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


def parse(path, over=()):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip().strip('"')
    for s in over:
        if "=" in s:
            k, v = s.split("=", 1); kv[k.strip()] = v.strip()
    return kv


def params(kv):
    f = lambda k, d=None: float(kv[k]) if k in kv else d
    L = float(kv["geometry.prob_hi"].split()[0]) - float(kv.get("geometry.prob_lo", "0 0").split()[0])
    return dict(R0=f("eta.ic.expression.constant.R0"), rho=float(kv["density0.ic.expression.region0"]),
                p_inf=f("pressure0.ic.expression.constant.p_inf"), p_b=f("pressure1.ic.expression.constant.p_b"),
                g=f("eos1.gamma", 1.4), kappa=f("shell.kappa_s", 0.0), mu=f("mu0", 0.0),
                Rinf=2.0 * L / np.sqrt(np.pi), eps=f("eta.ic.expression.constant.epsilon"),
                dx=L / float(kv["amr.n_cell"].split()[0]) / 2 ** int(f("amr.max_level", 0)))


def series(run):
    import yt
    yt.funcs.mylog.setLevel(50)
    cache = os.path.join(run, "R2d.npy")
    fs = sorted((f for f in glob.glob(os.path.join(run, "*cell")) if os.path.isdir(f)),
                key=lambda s: int(re.findall(r"(\d+)cell", s)[-1]))
    if os.path.exists(cache):
        a = np.load(cache)
        if len(a) == len(fs):
            return a[:, 0], a[:, 1]
    out = []
    for f in fs:
        ds = yt.load(f); ad = ds.all_data()
        A = float(np.sum((1 - np.asarray(ad["eta"])) * np.asarray(ad["index", "cell_volume"])))
        out.append((float(ds.current_time), np.sqrt(4 * A / np.pi)))
    out = np.array(sorted(out)); np.save(cache, out)
    return out[:, 0], out[:, 1]


def rp(t, P, ks):
    R0, rho, pinf, pb, g, mu, Rinf = P["R0"], P["rho"], P["p_inf"], P["p_b"], P["g"], P["mu"], P["Rinf"]

    def f(_, y):
        R, V = y
        pL = pb * (R0 / R) ** (2 * g) - ks * V / R / R - 2 * mu * V / R
        Lg = np.log(Rinf / R)
        return [V, ((pL - pinf) / rho + 0.5 * V * V * (1 - (R / Rinf) ** 2) - V * V * Lg) / (R * Lg)]
    s = solve_ivp(f, (0, t[-1]), [R0, 0.0], t_eval=t, rtol=1e-9, atol=1e-14 * R0, max_step=t[-1] / 3000)
    return s.y[0] if s.success and len(s.y[0]) == len(t) else np.full_like(t, np.nan)


def fit(t, R, P):
    err = lambda k: np.sqrt(np.nanmean((rp(t, P, k) - R) ** 2)) / P["R0"]
    kref = P["kappa"] if P["kappa"] > 0 else 1.0
    r = minimize_scalar(err, bounds=(0.0, 3.0 * kref), method="bounded", options={"xatol": 1e-4 * kref})
    return r.x, err


def extrema(t, R):
    i = int(np.argmax(R[: max(3, len(R) // 2)]))
    j = i + int(np.argmin(R[i:]))
    return R[i], R[j]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--input", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("-s", action="append", default=[])
    a = ap.parse_args()
    ov = lambda run: (open(os.path.join(run, "cmdline.txt")).read().split()
                      if os.path.exists(os.path.join(run, "cmdline.txt")) else [])
    knum = None
    if a.baseline:
        P0 = params(parse(a.input, ov(a.baseline) + a.s))
        t0, R0r = series(a.baseline); R0r = R0r / R0r[0] * P0["R0"]
        knum, _ = fit(t0, R0r, dict(P0, kappa=params(parse(a.input))["kappa"] or 1.0))
    print("%-26s %6s %6s %9s | %8s %8s | %8s %8s | %8s %9s" % ("run", "R0/dx", "eps/dx", "kappa", "Rmax", "(model)",
                                                                "Rmin", "(model)", "rms f=1", "k_eff/k"))
    for run in a.runs:
        P = params(parse(a.input, ov(run) + a.s))
        t, R = series(run); R = R / R[0] * P["R0"]
        Rm = rp(t, P, P["kappa"])
        k, err = fit(t, R, P)
        smx, smn = extrema(t, R / P["R0"]); mmx, mmn = extrema(t, Rm / P["R0"])
        kk = (k / P["kappa"]) if P["kappa"] > 0 else np.nan
        print("%-26s %6.1f %6.2f %9.3g | %8.4f %8.4f | %8.4f %8.4f | %8.4f %9.3f"
              % (os.path.basename(run.rstrip("/")), P["R0"] / P["dx"], P["eps"] / P["dx"], P["kappa"],
                 smx, mmx, smn, mmn, err(P["kappa"]), kk))
        if P["kappa"] == 0:
            print("%-26s   numerical damping as an equivalent kappa: %.3g" % ("", k))
        elif knum is not None:
            print("%-26s   minus the kappa_s = 0 baseline (k_num = %.3g): k_eff/k = %.3f"
                  % ("", knum, (k - knum) / P["kappa"]))


if __name__ == "__main__":
    main()
