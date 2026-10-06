#!/usr/bin/env python3
"""
Five-wave Riemann problem in a solid (tests/FlowElasticSolid/input_1D_FiveWave).

  python3 five_wave_check.py <run_dir> [--outdir DIR]

Linear (exact for small jumps) solution of a pressure jump pL | pR plus a transverse
velocity jump 0 | V in one elastic solid at rest:
    longitudinal fronts at x0 -+ c_l t, between them u = (pL - pR)/(2 rho c_l), sigma_11 = -(pL + pR)/2
    shear fronts        at x0 -+ c_t t, between them v = V/2,                  S_12 = rho c_t V/2
    c_l^2 = gamma (p + p_inf)/rho + (4/3) mu/rho,   c_t^2 = mu/rho
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
RHO, GAM, PINF, MU, PL, PR, V, X0 = 8.9, 4.22, 34.2, 92.0, 0.1, 1.0e-4, 0.01, 0.5


def main():
    d = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    pf = sorted(glob.glob(os.path.join(d, "output", "*cell")))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; t = float(ds.current_time)
    g = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    u, v, p, Sxx, Sxy = g("velocityx"), g("velocityy"), g("pressure"), g("elastic_Sxx"), g("elastic_Sxy")
    s11 = -p + Sxx
    cl = np.sqrt(GAM * (0.5 * (PL + PR) + PINF) / RHO + 4.0 / 3.0 * MU / RHO); ct = np.sqrt(MU / RHO)
    us, vs, S12 = (PL - PR) / (2 * RHO * cl), V / 2, RHO * ct * V / 2

    def front(q, half, side):       # position where q crosses `half`, on the given side of X0
        idx = np.where((x > X0) if side > 0 else (x < X0))[0]
        qq = q[idx] - half; k = np.where(np.sign(qq[:-1]) != np.sign(qq[1:]))[0]
        k = k[-1] if side > 0 else k[0]
        return x[idx][k] + (x[idx][k + 1] - x[idx][k]) * qq[k] / (qq[k] - qq[k + 1])
    L = (np.abs(x - X0) > ct * t + 0.03) & (np.abs(x - X0) < cl * t - 0.03)      # between shear and longitudinal fronts
    T = (np.abs(x - X0) < ct * t - 0.03) & (np.abs(x - X0) > 0.02)               # between the shear fronts
    rows = [("c_l (right front)", (front(u, us / 2, +1) - X0) / t, cl), ("c_l (left front)", (X0 - front(u, us / 2, -1)) / t, cl),
            ("c_t (right front)", (front(v, 0.75 * V, +1) - X0) / t, ct), ("c_t (left front)", (X0 - front(v, 0.25 * V, -1)) / t, ct),
            ("u between longitudinal fronts", u[L].mean(), us), ("sigma_11 there", s11[L].mean(), -(PL + PR) / 2),
            ("v between shear fronts", v[T].mean(), vs), ("S_12 there", Sxy[T].mean(), S12),
            ("u between shear fronts", u[T].mean(), us), ("sigma_11 there", s11[T].mean(), -(PL + PR) / 2)]
    print(f"{d}: t = {t:.4f}")
    with open(os.path.join(outdir, "five_wave_summary.csv"), "w") as fo:
        fo.write("quantity,Numerical,linear,error_pct\n")
        for k, a, b in rows:
            print(f"  {k:32s} {a:13.6e}   linear {b:13.6e}   ({100 * (a / b - 1):+.2f} %)")
            fo.write(f"{k},{a:.7g},{b:.7g},{100 * (a / b - 1):.3f}\n")
    # exact (linear) solution as full profiles; fronts at X0 +- c_l t (longitudinal) and X0 +- c_t t (shear)
    xl, xr, tl, tr = X0 - cl * t, X0 + cl * t, X0 - ct * t, X0 + ct * t
    def prof(nodes, vals):                       # piecewise-constant profile: values between consecutive nodes
        xe = np.repeat(np.array([x[0]] + list(nodes) + [x[-1]]), 2)[1:-1]; return xe, np.repeat(np.array(vals), 2)
    def at(nodes, vals): return np.array(vals)[np.searchsorted(np.array(nodes), x)]
    Q = ((1e3 * u, "Normal Velocity [m/s]", (xl, xr), (0.0, 1e3 * us, 0.0), 1e3 * us),
         (1e3 * v, "Transverse Velocity [m/s]", (tl, tr), (0.0, 1e3 * vs, 1e3 * V), 1e3 * V),
         (1e3 * s11, r"Normal Stress $\sigma_{11}$ [MPa]", (xl, xr), (-1e3 * PL, -1e3 * (PL + PR) / 2, -1e3 * PR), 1e3 * (PL - PR)),
         (1e3 * Sxy, r"Shear Stress $S_{12}$ [MPa]", (tl, tr), (0.0, 1e3 * S12, 0.0), 1e3 * S12))
    sk = slice(None, None, max(1, n // 125))
    def shade(a):
        a.axvspan(x[0] - 0.5 / n, x[-1] + 0.5 / n, color="#B87333", alpha=0.16, lw=0, zorder=0); a.set_xlim(x[0] - 0.5 / n, x[-1] + 0.5 / n)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a_, (q, lab, nodes, vals, sc) in zip(ax.ravel(), Q):
        shade(a_); xe, ye = prof(nodes, vals)
        a_.plot(xe, ye, "k-", lw=1.4, label="Exact")
        a_.plot(x, q, "-", color="C3", lw=0.9); a_.plot(x[sk], q[sk], "o", color="C3", ms=3.5, mfc="none", mew=0.9)
        a_.plot([], [], "o-", color="C3", ms=3.5, mfc="none", mew=0.9, lw=0.9, label="Numerical")
        a_.set_ylabel(lab); a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=9)
    for a_ in ax[1]: a_.set_xlabel("x [m]")
    fig.suptitle(rf"Five-Wave Riemann Problem in Copper at t = {1e3 * t:.0f} $\mu$s", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "five_wave.png"), dpi=200); plt.close(fig)

    # pointwise error against the exact profile, in per cent of: the plateau velocity u*, the transverse velocity
    # jump V, the initial pressure difference p_L - p_R, and the plateau shear stress
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for a_, (q, lab, nodes, vals, sc) in zip(ax.ravel(), Q):
        shade(a_); err = 100.0 * np.abs(q - at(nodes, vals)) / sc
        a_.semilogy(x, np.maximum(err, 1e-16), "-", color="C3", lw=0.9, label="Numerical")
        a_.set_ylim(1e-7, 200.0); a_.set_ylabel(lab.split(" [")[0].split(" $")[0] + " Error [%]")
        a_.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=30))
        a_.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=200)); a_.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a_.grid(alpha=0.35, which="major", lw=0.5); a_.grid(alpha=0.2, which="minor", lw=0.3)
    ax[0, 0].legend(fontsize=9)
    for a_ in ax[1]: a_.set_xlabel("x [m]")
    fig.suptitle(rf"Five-Wave Riemann Problem in Copper Error at t = {1e3 * t:.0f} $\mu$s", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "five_wave_error.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/five_wave.png, five_wave_error.png and five_wave_summary.csv")


if __name__ == "__main__":
    main()
