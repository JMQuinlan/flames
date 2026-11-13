#!/usr/bin/env python3
"""
A shock reflecting from a conducting wall heats it (tests/FlowConjugateHeat/input_1D_ShockHeatedWall).

  python3 shock_heated_wall_check.py <run_dir> [<run_dir> ...] [--outdir DIR]

After the reflection (t > t_r) the gas at rest at (p5, T5) is a half-space at constant pressure against the solid
half-space at T_s0.  Similarity solution, eta = psi/sqrt(tau), psi = int rho dx from the wall, tau = t - t_r:
    gas:    T' = T F,   F' = -eta T F/(2 a),   a = k p5/(cp R),   T(0) = T_i,  T(inf) = T5        (F = d ln T/d eta)
    solid:  T_s = T_s0 + (T_i - T_s0) erfc((x - x_w)/(2 sqrt(alpha_s tau)))
    flux:   q_w sqrt(tau) = k p5 F(0)/R = e_s (T_i - T_s0)/sqrt(pi),   e_s = sqrt(k_s rho_s cp_s)
solved here by shooting (F(0) for the far-field temperature, T_i for the flux balance).  The heat stored in the
solid is Q(tau) = 2 e_s (T_i - T_s0) sqrt(tau/pi).
Reported: arrival time of the shock, reflected state (p5, T5), surface temperature, stored heat, profiles.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import erfc
import yt
yt.funcs.mylog.setLevel(40)
GAM, CP, CV, MACH, X_SHOCK0 = 1.4, 3.5, 2.5, 1.2, 0.5
R = CP - CV


def meta(d, key):
    val = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def states():
    c1 = np.sqrt(GAM); S = MACH * c1
    p2 = 1 + 2 * GAM / (GAM + 1) * (MACH**2 - 1); r2 = (GAM + 1) * MACH**2 / ((GAM - 1) * MACH**2 + 2); T2 = p2 / r2
    a = (GAM + 1) / (GAM - 1); p5 = p2 * ((3 * GAM - 1) * p2 - (GAM - 1)) / ((GAM - 1) * p2 + (GAM + 1))
    T5 = T2 * (p5 / p2) * (a + p5 / p2) / (1 + a * p5 / p2)
    return S, p5, T5


def similarity(k, p5, T5, es, Ts0):
    a = k * p5 / (CP * R); EMAX = 12 * np.sqrt(a / T5) * 1.5
    def gas(Ti):
        def far(F0):
            s = solve_ivp(lambda e, y: [y[0] * y[1], -e * y[0] * y[1] / (2 * a)], (0, EMAX), [Ti, F0], rtol=1e-10, atol=1e-13)
            return s.y[0, -1] - T5
        F0 = brentq(far, 1e-8, 50.0 / np.sqrt(a))
        s = solve_ivp(lambda e, y: [y[0] * y[1], -e * y[0] * y[1] / (2 * a)], (0, EMAX), [Ti, F0], rtol=1e-10, atol=1e-13, dense_output=True)
        return F0, s
    bal = lambda Ti: k * p5 * gas(Ti)[0] / R - es * (Ti - Ts0) / np.sqrt(np.pi)
    Ti = brentq(bal, Ts0 + 1e-6 * (T5 - Ts0), T5 - 1e-6 * (T5 - Ts0), xtol=1e-12)
    F0, s = gas(Ti)
    eta = np.linspace(0, EMAX, 4000); T = s.sol(eta)[0]
    xi = np.concatenate([[0], np.cumsum(0.5 * (R * T[1:] / p5 + R * T[:-1] / p5) * np.diff(eta))])     # (x_w - x)/sqrt(tau)
    return Ti, xi, T


def analyse(d):
    k = meta(d, "mu0") * CP / meta(d, "thermal.Pr"); ks, cps, rhos = meta(d, "ablation.k_s"), meta(d, "ablation.cp_s"), meta(d, "ablation.rho_s")
    Ts0 = meta(d, "ablation.T_solid0"); xw = meta(d, "solid.phi.ic.expression.constant.xw")
    S, p5, T5 = states(); tr = (xw - X_SHOCK0) / S; es = np.sqrt(ks * rhos * cps); als = ks / (rhos * cps)
    Ti, xi, Tg = similarity(k, p5, T5, es, Ts0)
    e5 = np.sqrt(k * (p5 / (R * T5)) * CP); Ti_lin = (e5 * T5 + es * Ts0) / (e5 + es)
    H = []; last = None
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; dx = 1.0 / n; t = float(ds.current_time)
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi, T, Ts, p = q("phi"), q("T"), q("ablation_Ts"), q("pressure")
        iw = np.where(phi < 0.5)[0][0]                               # first solid cell
        Tig, Tis = 1.5 * T[iw - 1] - 0.5 * T[iw - 2], 1.5 * Ts[iw] - 0.5 * Ts[iw + 1]
        Q = rhos * cps * ((Ts - Ts0)[phi < 0.5]).sum() * dx
        j = iw - 1 - int(round(0.12 / dx))                           # 0.12 from the wall: outside the thermal layer
        H.append((t, p[iw - 1], Tig, Tis, Q, p[j], T[j], q("velocityx")[j]))
        last = (t, x, np.where(phi >= 0.5, T, Ts), phi, iw)
    H = np.array(H); t = H[:, 0]
    # arrival: wall pressure crosses the mean of 1 and p5
    c = np.where(H[:, 1] > 0.5 * (1 + p5))[0][0]; ta = t[c - 1] + (t[c] - t[c - 1]) * (0.5 * (1 + p5) - H[c - 1, 1]) / (H[c, 1] - H[c - 1, 1])
    return dict(d=d, n=len(last[1]), H=H, tr=tr, ta=ta, p5=p5, T5=T5, Ti=Ti, Ti_lin=Ti_lin, es=es, Ts0=Ts0, xi=xi, Tg=Tg, als=als, xw=xw, last=last)


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    A = sorted((analyse(d) for d in dirs), key=lambda r: r["n"])
    if outdir is None: outdir = os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[-1])))
    os.makedirs(outdir, exist_ok=True)
    a0 = A[0]; dT = a0["T5"] - a0["Ts0"]
    print(f"Mach {MACH:g} shock on a conducting wall: t_r = {a0['tr']:.4f}, p5 = {a0['p5']:.4f}, T5 = {a0['T5']:.4f};  exact surface temperature "
          f"{a0['Ti']:.5f}  (linearised {a0['Ti_lin']:.5f})")
    print(f"{'cells':>6} {'arrival':>9} {'p5':>9} {'T5':>9} {'T_i gas side':>13} {'T_i solid side':>15} {'stored heat Q':>14} {'profile gas':>12} {'profile solid':>14}")
    rows = []
    for a in A:
        H = a["H"]; t = H[:, 0]; m = t > a["tr"] + 0.25; tau = t[-1] - a["tr"]
        Qex = 2 * a["es"] * (a["Ti"] - a["Ts0"]) * np.sqrt(tau / np.pi)
        tl, x, Tn, phi, iw = a["last"]; fl = phi >= 0.5; dx = x[1] - x[0]
        Tex = np.where(fl, np.interp((a["xw"] - x) / np.sqrt(tau), a["xi"], a["Tg"]), a["Ts0"] + (a["Ti"] - a["Ts0"]) * erfc((x - a["xw"]) / (2 * np.sqrt(a["als"] * tau))))
        near = (np.abs(x - a["xw"]) < 0.1) & (np.abs(x - a["xw"]) > 1.5 * dx); err = np.abs(Tn - Tex) / dT
        r = dict(n=a["n"], ta=100 * (a["ta"] / a["tr"] - 1), p5=100 * (H[m, 5].mean() / a["p5"] - 1), T5=100 * (H[m, 6].mean() - a["T5"]) / dT,
                 tig=100 * (H[-1, 2] - a["Ti"]) / dT, tis=100 * (H[-1, 3] - a["Ti"]) / dT, Q=100 * (H[-1, 4] / Qex - 1),
                 eg=100 * err[near & fl].max(), es=100 * err[near & ~fl].max(), x=x, Tn=Tn, Tex=Tex, near=near, fl=fl, tau=tau)
        rows.append(r)
        print(f"{r['n']:6d} {r['ta']:+7.2f} % {r['p5']:+7.3f} % {r['T5']:+7.2f} % {r['tig']:+11.2f} % {r['tis']:+13.2f} % {r['Q']:+12.2f} % {r['eg']:10.2f} % {r['es']:12.2f} %")
    print("  (temperatures in % of T5 - T_s0; state 5 sampled 0.12 from the wall for t > t_r + 0.25; surface values and profiles at the last time)")
    with open(os.path.join(outdir, "shock_heated_wall_summary.csv"), "w") as fo:
        fo.write("cells,arrival_pct,p5_pct,T5_pct,Ti_gas_pct,Ti_solid_pct,Q_pct,profile_gas_pct,profile_solid_pct\n")
        for r in rows: fo.write(",".join(f"{r[k]:.5g}" for k in ("n", "ta", "p5", "T5", "tig", "tis", "Q", "eg", "es")) + "\n")

    a = A[-1]; r = rows[-1]; H = a["H"]; t = H[:, 0]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    c = ax[0]; w = np.abs(r["x"] - a["xw"]) < 0.12; st = max(1, int(w.sum() // 60))
    c.axvspan(a["xw"], a["xw"] + 0.12, color="#B87333", alpha=0.12, lw=0)
    c.plot(r["x"][w], r["Tex"][w], "C3-", lw=1.6, label="Exact"); c.plot(r["x"][w][::st], r["Tn"][w][::st], "ko", ms=4, mfc="none", label="Numerical")
    c.set_xlabel("Position [m]"); c.set_ylabel("Temperature [K]"); c.set_title(f"Temperature {r['tau']:.2f} s After the Reflection"); c.legend()
    c = ax[1]; m = t > a["tr"]
    c.axhline(a["Ti"], color="C3", lw=1.6, label="Exact"); c.plot(t[m] - a["tr"], H[m, 3], "ko", ms=4, mfc="none", label="Numerical")
    c.set_xlabel("Time After the Reflection [s]"); c.set_ylabel("Surface Temperature [K]"); c.set_title("Surface Temperature"); c.legend()
    c.set_ylim(a["Ts0"], a["Ti"] + 0.4 * (a["Ti"] - a["Ts0"]))
    c = ax[2]; tt = np.linspace(0, t[-1] - a["tr"], 200)
    c.plot(tt, 2 * a["es"] * (a["Ti"] - a["Ts0"]) * np.sqrt(tt / np.pi), "C3-", lw=1.6, label="Exact"); c.plot(t[m] - a["tr"], H[m, 4], "ko", ms=4, mfc="none", label="Numerical")
    c.set_xlabel("Time After the Reflection [s]"); c.set_ylabel(r"Heat Stored in the Wall [J/m$^2$]"); c.set_title("Heat Transferred to the Wall"); c.legend()
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle(f"Mach {MACH:g} Shock Reflecting from a Conducting Wall"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "shock_heated_wall.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/shock_heated_wall.png")


if __name__ == "__main__":
    main()
