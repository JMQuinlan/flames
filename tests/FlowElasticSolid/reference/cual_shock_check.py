#!/usr/bin/env python3
"""
Al | Cu solid-solid shock tube (tests/FlowElasticSolid/input_1D_CuAlShock),
Favrie, Gavrilyuk & Saurel, JCP 228 (2009), Sec. 5.2 / Fig. 5.

  python3 cual_shock_check.py <run_dir> [--outdir DIR]

The pressure ratio is small against the moduli, so the acoustic solution is
accurate:  Z = rho c_l,  c_l^2 = gamma (p + p_inf)/rho + (4/3) mu/rho
    u*        = (p_L - p_R)/(Z_L + Z_R)
    sigma_11* = -p_L + Z_L u*                 (continuous across the contact)
    p*_K      = p_K -+ rho_K c_sK^2 u*/c_lK   (jumps at the contact)
Units: density 1e3 kg/m3, pressure 1e9 Pa, velocity km/s, time ms.
Paper values (Fig. 5): u* = 1.52 m/s, sigma_11* = -74 MPa, p* = 82 / 40 MPa.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)

AL = dict(rho=2.7, gam=3.4, pinf=21.5, mu=26.0, p=0.1)
CU = dict(rho=8.9, gam=4.22, pinf=34.2, mu=92.0, p=1.0e-4)
XI = 0.5


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    for m in (AL, CU):
        m["cs2"] = m["gam"] * (m["p"] + m["pinf"]) / m["rho"]
        m["cl"] = np.sqrt(m["cs2"] + 4.0 / 3.0 * m["mu"] / m["rho"]); m["Z"] = m["rho"] * m["cl"]
    us = (AL["p"] - CU["p"]) / (AL["Z"] + CU["Z"]); sig = -AL["p"] + AL["Z"] * us
    pA = AL["p"] - AL["rho"] * AL["cs2"] * us / AL["cl"]; pC = CU["p"] + CU["rho"] * CU["cs2"] * us / CU["cl"]
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    u, p, S = g("velocityx"), g("pressure"), g("elastic_Sxx"); s11 = -p + S
    xa, xc = XI - AL["cl"] * t, XI + CU["cl"] * t
    A = (x > xa + 0.3 * (XI - xa)) & (x < XI - 0.02); C = (x > XI + 0.02) & (x < xc - 0.3 * (xc - XI))
    rows = [("u* (Al side)", u[A].mean(), us), ("u* (Cu side)", u[C].mean(), us),
            ("sigma_11* (Al side)", s11[A].mean(), sig), ("sigma_11* (Cu side)", s11[C].mean(), sig),
            ("p* in Al", p[A].mean(), pA), ("p* in Cu", p[C].mean(), pC)]
    print(f"{d}: t = {t:.4f} ms   c_l(Al) = {AL['cl']:.4f}, c_l(Cu) = {CU['cl']:.4f} km/s")
    with open(os.path.join(outdir, "cual_shock_summary.csv"), "w") as fo:
        fo.write("quantity,Hydro2,acoustic,error_pct\n")
        for k, a, b in rows:
            print(f"  {k:22s} {a:12.6e}   acoustic {b:12.6e}   ({100 * (a / b - 1):+.3f} %)")
            fo.write(f"{k},{a:.7g},{b:.7g},{100 * (a / b - 1):.4f}\n")
    print(f"  sigma_11 jump across the contact: {abs(s11[A][-1] - s11[C][0]):.2e}   (pressure jump {abs(p[A][-1] - p[C][0]):.4f})")
    # exact (acoustic) solution as a full profile: left state | Al star | contact | Cu star | right state
    xcon = XI + us * t
    def exact(left, starA, starC, right):
        xe = np.array([x[0], xa, xa, xcon, xcon, xc, xc, x[-1]]); return xe, np.array([left, left, starA, starA, starC, starC, right, right])
    rhoA, rhoC = AL["rho"] * (1 - us / AL["cl"]), CU["rho"] * (1 + us / CU["cl"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True); sk = slice(None, None, max(1, n // 125))
    for a, q, lab, ex in ((ax[0, 0], 1e3 * u, "Velocity [m/s]", exact(0.0, 1e3 * us, 1e3 * us, 0.0)),
                          (ax[0, 1], 1e3 * p, "Pressure [MPa]", exact(1e3 * AL["p"], 1e3 * pA, 1e3 * pC, 1e3 * CU["p"])),
                          (ax[1, 0], 1e3 * s11, r"Normal Stress $\sigma_{11}$ [MPa]", exact(-1e3 * AL["p"], 1e3 * sig, 1e3 * sig, -1e3 * CU["p"])),
                          (ax[1, 1], 1e3 * g("density"), r"Density [kg/m$^3$]", exact(1e3 * AL["rho"], 1e3 * rhoA, 1e3 * rhoC, 1e3 * CU["rho"]))):
        a.axvspan(x[0] - 0.5 / n, xcon, color="#C0C0C0", alpha=0.28, lw=0, zorder=0)       # aluminium side (silver)
        a.axvspan(xcon, x[-1] + 0.5 / n, color="#B87333", alpha=0.16, lw=0, zorder=0)      # copper side
        a.set_xlim(x[0] - 0.5 / n, x[-1] + 0.5 / n)
        a.plot(ex[0], ex[1], "k-", lw=1.4, label="Exact")
        a.plot(x, q, "-", color="C3", lw=0.9)                                  # every cell
        a.plot(x[sk], q[sk], "o", color="C3", ms=3.5, mfc="none", mew=0.9)   # markers on every 8th cell
        a.plot([], [], "o-", color="C3", ms=3.5, mfc="none", mew=0.9, lw=0.9, label="Numerical")
        a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=9)
    for a in ax[1]: a.set_xlabel("x [m]")
    fig.suptitle(rf"Aluminum-Copper Shock Tube at t = {1e3 * t:.0f} $\mu$s", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cual_shock.png"), dpi=200); plt.close(fig)
    # ---- pointwise error against the exact (acoustic) profile, log scale
    def exact_at(left, starA, starC, right):
        return np.where(x < xa, left, np.where(x < xcon, starA, np.where(x < xc, starC, right)))
    dp = 1e3 * (AL["p"] - CU["p"])
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a, q, lab, ex, sc in ((ax[0, 0], 1e3 * u, "Velocity Error [%]", exact_at(0.0, 1e3 * us, 1e3 * us, 0.0), 1e3 * us),
                              (ax[0, 1], 1e3 * p, "Pressure Error [%]", exact_at(1e3 * AL["p"], 1e3 * pA, 1e3 * pC, 1e3 * CU["p"]), dp),
                              (ax[1, 0], 1e3 * s11, "Normal Stress Error [%]", exact_at(-1e3 * AL["p"], 1e3 * sig, 1e3 * sig, -1e3 * CU["p"]), dp),
                              (ax[1, 1], 1e3 * g("density"), "Density Error [%]", exact_at(1e3 * AL["rho"], 1e3 * rhoA, 1e3 * rhoC, 1e3 * CU["rho"]), None)):
        # per cent of: the plateau velocity u*, the initial pressure difference p_L - p_R (pressure and stress),
        # and the local exact density
        err = 100.0 * np.abs(q - ex) / (sc if sc is not None else ex)
        a.axvspan(x[0] - 0.5 / n, xcon, color="#C0C0C0", alpha=0.28, lw=0, zorder=0)
        a.axvspan(xcon, x[-1] + 0.5 / n, color="#B87333", alpha=0.16, lw=0, zorder=0)
        a.semilogy(x, np.maximum(err, 1e-16), "-", color="C3", lw=0.9, label="Numerical")
        a.set_xlim(x[0] - 0.5 / n, x[-1] + 0.5 / n); a.set_ylim(1e-7, 200.0); a.set_ylabel(lab)
        a.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=30))
        a.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=200)); a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.grid(alpha=0.35, which="major", lw=0.5); a.grid(alpha=0.2, which="minor", lw=0.3)
    ax[0, 0].legend(fontsize=9)
    for a in ax[1]: a.set_xlabel("x [m]")
    fig.suptitle(rf"Aluminum-Copper Shock Tube Error at t = {1e3 * t:.0f} $\mu$s", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cual_shock_error.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/cual_shock.png, cual_shock_error.png and cual_shock_summary.csv")


if __name__ == "__main__":
    main()
