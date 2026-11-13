#!/usr/bin/env python3
"""
CO2 depressurization tube (input_1D_CO2Depressurization): Numerical against the exact solution without mass transfer
and against the published equilibrium (p-T-g relaxed) solution with mass transfer.

  python3 co2_depressurization_check.py --out FIG_DIR "Label=run_dir" ...   (run dirs: plot_file of the runs)

Without mass transfer and with traces of 1e-5 of the other phase, the problem is the two-material Riemann problem
liquid CO2 | CO2 vapour (stiffened gases), solved exactly here (Toro's method with p -> p + pi on each side).
With mass transfer the reference is the plateau values read from Fig. 8 of (Bacigaluppi et al. 2021) (4-equation
model with p-T-g relaxation, HLLC, 2000 cells) and the saturation pressure of this EOS pair at the local temperature.
Temperature is the phase-change convention T = (p + pi)/((gamma - 1) cv rho) of the volume-weighted phases.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)

LIQ = dict(g=1.23, pi=1.32e8, cv=2440.0, q=-6.23e5, qp=-5.3409289e3)
VAP = dict(g=1.06, pi=8.86e5, cv=2410.0, q=-3.01e5, qp=-1.0398090e4)
X0, TEND = 50.0, 0.08
T0 = 273.0
rhoT = lambda m, p, T: (p + m["pi"]) / ((m["g"] - 1) * m["cv"] * T)
# plateau values read from Fig. 8 of Bacigaluppi et al. (2021) (with mass transfer), +- half a minor tick
FIG8 = dict(p1=3.5e6, p2=1.93e6, u1=7.0, u2=76.0, T2=279.5)


def gibbs(m, p, T):
    return (m["g"] * m["cv"] - m["qp"]) * T - m["cv"] * T * np.log(T ** m["g"] / (p + m["pi"]) ** (m["g"] - 1)) + m["q"]


def psat(T):
    from scipy.optimize import brentq
    return brentq(lambda p: gibbs(LIQ, p, T) - gibbs(VAP, p, T), 1e3, 2e8)


def sg_riemann(L, R, mL, mR, x, t):
    """exact Riemann solution, two stiffened gases. L, R = (rho, u, p)."""
    rl, ul, pl = L; rr, ur, pr = R
    def fK(p, rk, pk, m):
        g, pi = m["g"], m["pi"]; P, Pk = p + pi, pk + pi; ck = np.sqrt(g * Pk / rk)
        if p > pk:
            A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * Pk
            return (P - Pk) * np.sqrt(A / (P + B))
        return 2 * ck / (g - 1) * ((P / Pk) ** ((g - 1) / (2 * g)) - 1)
    from scipy.optimize import brentq
    F = lambda p: fK(p, rl, pl, mL) + fK(p, rr, pr, mR) + ur - ul
    lo = -min(mL["pi"], mR["pi"]) + 1e-6
    ps = brentq(F, lo, 1e3 * max(pl, pr)); us = 0.5 * (ul + ur) + 0.5 * (fK(ps, rr, pr, mR) - fK(ps, rl, pl, mL))
    out = np.zeros((3, len(x))); side = np.zeros(len(x), int)
    for i, s in enumerate((x - X0) / t):
        if s <= us: m, rk, uk, pk, sg = mL, rl, ul, pl, -1
        else: m, rk, uk, pk, sg = mR, rr, ur, pr, 1
        g, pi = m["g"], m["pi"]; Pk, P = pk + pi, ps + pi; ck = np.sqrt(g * Pk / rk); gm = (g - 1) / (g + 1)
        side[i] = 0 if sg < 0 else 1
        if ps > pk:
            S = uk + sg * ck * np.sqrt((g + 1) / (2 * g) * P / Pk + (g - 1) / (2 * g))
            behind = (s >= S) if sg < 0 else (s <= S)
            st = (rk * (P / Pk + gm) / (gm * P / Pk + 1), us, ps) if behind else (rk, uk, pk)
        else:
            cs = ck * (P / Pk) ** ((g - 1) / (2 * g)); SH, ST = uk + sg * ck, us + sg * cs
            if (s <= SH) if sg < 0 else (s >= SH): st = (rk, uk, pk)
            elif (s >= ST) if sg < 0 else (s <= ST): st = (rk * (P / Pk) ** (1 / g), us, ps)
            else:
                c = 2 / (g + 1) * (ck - sg * (g - 1) / 2 * (uk - s))
                st = (rk * (c / ck) ** (2 / (g - 1)), 2 / (g + 1) * (-sg * ck + (g - 1) / 2 * uk + s), (c / ck) ** (2 * g / (g - 1)) * Pk - pi)
        out[:, i] = st
    T = np.where(side == 0, (out[2] + mL["pi"]) / ((mL["g"] - 1) * mL["cv"] * out[0]), (out[2] + mR["pi"]) / ((mR["g"] - 1) * mR["cv"] * out[0]))
    return out[0], out[1], out[2], T, ps, us


def load(d):
    pfs = sorted(glob.glob(os.path.join(d, "*cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    ds = yt.load(pfs[-1]); ad = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); j = ds.domain_dimensions[1] // 2
    q = lambda n: np.array(ad["boxlib", n])[:, j, 0]
    x = np.array(ad["index", "x"])[:, j, 0]
    eta, r0, r1, p = q("eta"), q("rho_eta0"), q("rho_eta1"), q("pressure")
    with np.errstate(all="ignore"):
        rv, rl = r0 / np.maximum(eta, 1e-300), r1 / np.maximum(1 - eta, 1e-300)
        Tv, Tl = (p + VAP["pi"]) / ((VAP["g"] - 1) * VAP["cv"] * rv), (p + LIQ["pi"]) / ((LIQ["g"] - 1) * LIQ["cv"] * rl)
        T = np.where(eta > 0.5, Tv, Tl)
    return dict(t=float(ds.current_time), x=x, rho=q("density"), u=q("velocityx"), p=p, T=T, eta=eta)


def main():
    a = sys.argv[1:]; out = a[a.index("--out") + 1] if "--out" in a else "."; os.makedirs(out, exist_ok=True)
    runs = [s.split("=", 1) for s in a if "=" in s and not s.startswith("--")]
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 12})
    L = (rhoT(LIQ, 6e6, T0), 0.0, 6e6); R = (rhoT(VAP, 1e6, T0), 0.0, 1e6)
    xe = np.linspace(0, 80, 8001); re, ue, pe, Te, ps, us = sg_riemann(L, R, LIQ, VAP, xe, TEND)
    print("exact without transfer: p* = %.4e Pa, u* = %.3f m/s;  p_sat(273 K) of this EOS pair = %.4e Pa" % (ps, us, psat(T0)))
    labs = [r"Density, $\rho$ [kg/m$^3$]", r"Velocity, $u$ [m/s]", r"Pressure, $p$ [bar]", r"Temperature, $T$ [K]"]
    plateau = lambda r, xa, xb, q: np.median(r[q][(r["x"] > xa) & (r["x"] < xb)])
    for lab, d in runs:
        r = load(d); notr = "No" in lab
        fig, ax = plt.subplots(2, 2, figsize=(14, 9.5)); ax = ax.ravel()
        if notr:
            for k, e in enumerate((re, ue, pe / 1e5, Te)): ax[k].plot(xe, e, "k-", lw=1.8, label="Exact")
        for k, q in enumerate(("rho", "u", "p", "T")):
            ax[k].plot(r["x"], r[q] / (1e5 if q == "p" else 1), "-", color="C0", lw=1.0, marker="o", markevery=60, ms=6, mfc="none", mew=1.1, label="Numerical")
            ax[k].set_xlabel(r"Position, $x$ [m]"); ax[k].set_ylabel(labs[k]); ax[k].grid(alpha=0.3)
        print("%-32s t = %.4f" % (lab, r["t"]))
        print("    liquid-side plateau (20-40 m): p = %.4e  u = %.3f  T = %.2f" % (plateau(r, 20, 40, "p"), plateau(r, 20, 40, "u"), plateau(r, 20, 40, "T")))
        print("    vapour-side plateau (58-66 m): p = %.4e  u = %.3f  T = %.2f" % (plateau(r, 58, 66, "p"), plateau(r, 58, 66, "u"), plateau(r, 58, 66, "T")))
        if notr:
            # pointwise errors on the plateaus (away from the fronts: 3 cells each side excluded)
            ei = np.interp(r["x"], xe, pe); eu = np.interp(r["x"], xe, ue); eT = np.interp(r["x"], xe, Te)
            # plateaus: points where the exact solution is constant within +-2 m (all waves and fans excluded)
            ok = np.array([np.ptp(pe[(xe > xx - 2) & (xe < xx + 2)]) < 1e-9 * ps and np.ptp(re[(xe > xx - 2) & (xe < xx + 2)]) < 1e-9 * re.max() for xx in r["x"]])
            print("    against exact (plateaus: exact constant within +-2 m): max |dp|/p %.3f %%, max |du|/u* %.3f %%, max |dT|/T %.4f %%" % (
                100 * np.max(np.abs(r["p"][ok] - ei[ok]) / ei[ok]), 100 * np.max(np.abs(r["u"][ok] - eu[ok])) / us, 100 * np.max(np.abs(r["T"][ok] - eT[ok]) / eT[ok])))
            ttl = r"CO$_2$ Depressurization Tube Without Mass Transfer at $t$ = %g s" % TEND; name = "co2_depressurization_novap"
        else:
            T1 = plateau(r, 20, 40, "T")
            print("    p_sat(T) at the liquid-side plateau temperature: %.4e  (plateau / p_sat - 1 = %+.2f %%)" % (psat(T1), 100 * (plateau(r, 20, 40, "p") / psat(T1) - 1)))
            print("    against (Bacigaluppi et al. 2021) Fig. 8: p1 %+.1f %%, p2 %+.1f %%, u1 %+.1f m/s, u2 %+.1f m/s" % (
                100 * (plateau(r, 20, 40, "p") / FIG8["p1"] - 1), 100 * (plateau(r, 58, 66, "p") / FIG8["p2"] - 1), plateau(r, 20, 40, "u") - FIG8["u1"], plateau(r, 58, 66, "u") - FIG8["u2"]))
            kw = dict(color="k", ls="--", lw=1.2)
            ax[2].axhline(FIG8["p1"] / 1e5, label="Plateaus (Bacigaluppi et al. 2021)", **kw); ax[2].axhline(FIG8["p2"] / 1e5, **kw)
            ax[1].axhline(FIG8["u1"], label="Plateaus (Bacigaluppi et al. 2021)", **kw); ax[1].axhline(FIG8["u2"], **kw)
            xs = np.linspace(0, 80, 400); liq = (1 - r["eta"]) > 1e-3; ax[2].plot(r["x"][liq], np.array([psat(T) for T in r["T"][liq]]) / 1e5, ":", color="C3", lw=2.0, label=r"Saturation Pressure $p_{sat}(T)$ Where Liquid")
            ax[2].set_ylim(0, 65)
            ttl = r"CO$_2$ Depressurization Tube With Mass Transfer at $t$ = %g s" % TEND; name = "co2_depressurization_vap"
        for k in range(4): ax[k].legend(loc="best")
        fig.suptitle(ttl); fig.tight_layout()
        for ext in ("png", "eps"): fig.savefig(os.path.join(out, name + "." + ext), dpi=200 if ext == "png" else None)
        plt.close(fig); print("wrote", os.path.join(out, name + ".png"))


if __name__ == "__main__":
    main()
