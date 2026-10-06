#!/usr/bin/env python3
"""
1D gas shock hitting an elastic solid (tests/FlowElasticSolid/input_1D_ShockImpact).

  python3 shock_impact_1d_check.py <run_dir> [--outdir DIR]

Reference: impedance matching.  The reflected shock takes the gas from the
post-incident-shock state (rho2, u2, p2) to (u_i, p3); the solid responds
acoustically, p3 - p1 = Z_s u_i with Z_s = rho_s c_l and
c_l^2 = gamma_s (p1 + p_inf)/rho_s + (4/3) mu/rho_s  (Favrie et al. 2009, Sec. 3.5).
Gas Hugoniot (left-facing shock, ideal gas):
    u2 - u_i = (p3 - p2) sqrt( A / (p3 + B) ),  A = 2/((g+1) rho2),  B = (g-1)/(g+1) p2.
In the solid behind the transmitted wave (strain e = u_i / c_l):
    p - p1 = rho_s c_s^2 e,   S_11 = -(4/3) mu e,   sigma_11 = -p + S_11 = -p3.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

G, RHO1, P1, MS = 1.4, 1.0, 1.0, 2.0
RHOS, GS, PINF, MU = 10.0, 4.4, 100.0, 50.0
XI = 0.6


def reference():
    c1 = np.sqrt(G * P1 / RHO1)
    p2 = P1 * (1 + 2 * G / (G + 1) * (MS**2 - 1))
    rho2 = RHO1 * (G + 1) * MS**2 / ((G - 1) * MS**2 + 2)
    u2 = MS * c1 * (1 - RHO1 / rho2)
    cs2 = GS * (P1 + PINF) / RHOS
    cl = np.sqrt(cs2 + 4.0 / 3.0 * MU / RHOS)
    Z = RHOS * cl
    A, B = 2 / ((G + 1) * rho2), (G - 1) / (G + 1) * p2
    f = lambda p3: u2 - (p3 - p2) * np.sqrt(A / (p3 + B)) - (p3 - P1) / Z
    lo, hi = p2, 100 * p2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    p3 = 0.5 * (lo + hi); ui = (p3 - P1) / Z; e = ui / cl
    return dict(p2=p2, rho2=rho2, u2=u2, cl=cl, Z=Z, p3=p3, ui=ui,
                p_solid=P1 + RHOS * cs2 * e, S11=-4.0 / 3.0 * MU * e, t_hit=(XI - 0.3) / (MS * c1))


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    R = reference()
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    u, p, S, eta, rho = g("velocityx"), g("pressure"), g("elastic_Sxx"), g("eta"), g("density")
    sig = -p + S
    xi = x[np.argmin(np.abs(eta - 0.5))]                       # interface now
    xw = XI + R["cl"] * (t - R["t_hit"])                       # transmitted wave front
    gas = (x > xi - 0.06) & (x < xi - 0.02)                    # gas behind the reflected shock
    sol = (x > xi + 0.03) & (x < min(xw, 0.98) - 0.05)         # solid behind the transmitted wave
    rows = [("interface velocity u_i (gas side)", u[gas].mean(), R["ui"]),
            ("interface velocity u_i (solid side)", u[sol].mean(), R["ui"]),
            ("reflected pressure p3 (gas)", p[gas].mean(), R["p3"]),
            ("-sigma_11 in the solid", -sig[sol].mean(), R["p3"]),
            ("pressure in the solid", p[sol].mean(), R["p_solid"]),
            ("deviatoric S_11 in the solid", S[sol].mean(), R["S11"]),
            ("interface position", xi, XI + R["ui"] * (t - R["t_hit"]))]
    print(f"{d}: t = {t:.4f}; shock reaches the solid at t = {R['t_hit']:.4f}; c_l = {R['cl']:.4f}")
    with open(os.path.join(outdir, "shock_impact_1d_summary.csv"), "w") as fo:
        fo.write("quantity,Hydro2,reference,error_pct\n")
        for k, a, b in rows:
            print(f"  {k:38s} {a:10.5f}   ref {b:10.5f}   ({100 * (a / b - 1):+.2f} %)")
            fo.write(f"{k},{a:.6g},{b:.6g},{100 * (a / b - 1):.3f}\n")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a, q, lab, ref in ((ax[0, 0], u, r"$u$", R["ui"]), (ax[0, 1], p, r"$p$", None),
                           (ax[1, 0], sig, r"$\sigma_{11} = -p + S_{11}$", -R["p3"]), (ax[1, 1], S, r"$S_{11}$", R["S11"])):
        a.plot(x, q, "k-", lw=1); a.set_ylabel(lab); a.grid(alpha=0.3); a.axvline(xi, color="0.6", ls=":")
        if ref is not None: a.axhline(ref, color="C3", ls="--", lw=0.9, label="impedance matching")
    ax[0, 1].axhline(R["p3"], color="C3", ls="--", lw=0.9, label=r"$p_3$ (gas)")
    ax[0, 1].axhline(R["p_solid"], color="C0", ls="--", lw=0.9, label="solid")
    for a in ax.ravel(): a.legend(fontsize=8)
    for a in ax[1]: a.set_xlabel("x")
    fig.suptitle(f"Mach {MS:g} gas shock on an elastic solid, t = {t:.3f} (dotted: interface)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "shock_impact_1d.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/shock_impact_1d.png and shock_impact_1d_summary.csv")


if __name__ == "__main__":
    main()
