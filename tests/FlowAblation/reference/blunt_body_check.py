#!/usr/bin/env python3
"""
Mach 3 cylinder, cold isothermal wall (tests/FlowAblation/input_2D_BluntBody).

  python3 blunt_body_check.py <run_dir> [--mu MU] [--outdir DIR]

Averages the plot files of the last quarter of the run and compares
  * stagnation pressure with the Rayleigh pitot value (exact for inviscid flow)
  * shock stand-off on the axis (p = mean of the pre / post-shock values) with Billig's
    correlation  Delta/R = 0.386 exp(4.67/M^2)
  * stagnation heat flux with Fay-Riddell for a 2D body (constant mu, Le = 1, no dissociation)
        q = 0.57 Pr^-0.6 (rho_e mu)^0.4 (rho_w mu)^0.1 sqrt(beta) cp (T_0 - T_w)
    with beta = Newtonian (1/R) sqrt(2 (p_e - p_inf)/rho_e), and with beta MEASURED from the wall
    pressure, p_w(s) = p_e - rho_e beta^2 s^2 / 2 near the stagnation point.
    A boundary-layer correlation: expect ~10 %, more when the layer is a few cells thick.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
GAM, CP, PR, RHO, P, UINF, R0 = 1.4, 1005.0, 0.72, 1.0, 0.714286, 3.0, 0.25


def main():
    d = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
    MU = float(sys.argv[sys.argv.index("--mu") + 1]) if "--mu" in sys.argv else 2.0e-3
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    Rg = CP * (GAM - 1) / GAM; Tinf = P / (RHO * Rg); M = UINF / np.sqrt(GAM * P / RHO)
    p2 = P * (1 + 2 * GAM / (GAM + 1) * (M * M - 1))
    p0 = P * ((GAM + 1) * M * M / 2)**(GAM / (GAM - 1)) * ((GAM + 1) / (2 * GAM * M * M - (GAM - 1)))**(1 / (GAM - 1))
    T0 = Tinf * (1 + (GAM - 1) / 2 * M * M); rho_e = p0 / (Rg * T0); rho_w = p0 / (Rg * Tinf)
    beta_N = np.sqrt(2 * (p0 - P) / rho_e) / R0
    qFR = lambda b: 0.57 * PR**-0.6 * (rho_e * MU)**0.4 * (rho_w * MU)**0.1 * np.sqrt(b) * CP * (T0 - Tinf)
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    use = pfs[-max(1, len(pfs) // 4):]; acc = {}; hist = []
    for pf in pfs[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        f = {k: np.asarray(cg[("boxlib", k)])[:, :, 0] for k in ("pressure", "density", "ablation_q", "phi")}
        nx, ny = ds.domain_dimensions[:2]; lo = np.asarray(ds.domain_left_edge); w = np.asarray(ds.domain_width)
        x = lo[0] + (np.arange(nx) + 0.5) * w[0] / nx; y = lo[1] + (np.arange(ny) + 0.5) * w[1] / ny
        jc = [ny // 2 - 1, ny // 2]; qs = np.mean([f["ablation_q"][:, j][f["ablation_q"][:, j] != 0][0] for j in jc if (f["ablation_q"][:, j] != 0).any()] or [0.0])
        hist.append((float(ds.current_time), qs))
        if pf in use:
            for k in f: acc[k] = acc.get(k, 0) + f[k] / len(use)
    hist = np.array(hist); dx = x[1] - x[0]; X, Y = np.meshgrid(x, y, indexing="ij")
    p, q, phi = acc["pressure"], acc["ablation_q"], acc["phi"]   # ablation_q > 0: heat into the wall
    pc = 0.5 * (p[:, jc[0]] + p[:, jc[1]]); iw = np.where(phi[:, jc[0]] > 0.5)[0]; iw = iw[x[iw] < 0][-1]     # stagnation wall cell
    i_s = np.where(pc[:iw] > 0.5 * (P + p2))[0][0]; xs = x[i_s] - dx * (pc[i_s] - 0.5 * (P + p2)) / (pc[i_s] - pc[i_s - 1])
    wall = (q != 0) & (X < 0); th = np.degrees(np.arctan2(Y[wall], -X[wall])); s = R0 * np.radians(np.abs(th)); pw = p[wall]; qw = q[wall]
    m = s < 0.35 * R0; A = np.polyfit(s[m]**2, pw[m], 1); pe_fit = A[1]; beta_m = np.sqrt(max(-2 * A[0] / rho_e, 0.0))
    q_stag = qw[np.abs(th) < 4].mean()
    print(f"{d}: M = {M:.2f}, mu = {MU:g}; averaged {len(use)} plot files, t = {float(yt.load(use[0]).current_time):.2f} .. {float(yt.load(use[-1]).current_time):.2f}; dx = {dx:.4f}")
    print(f"  stagnation pressure {pc[iw]:.4f}   Rayleigh pitot {p0:.4f}   ({100 * (pc[iw] / p0 - 1):+.2f} %)")
    print(f"  shock stand-off Delta/R = {(-R0 - xs) / R0:.3f}   Billig {0.386 * np.exp(4.67 / M**2):.3f}   ({100 * ((-R0 - xs) / R0 / (0.386 * np.exp(4.67 / M**2)) - 1):+.1f} %)")
    print(f"  velocity gradient beta: Newtonian {beta_N:.3f}, from the wall pressure {beta_m:.3f}  ({100 * (beta_m / beta_N - 1):+.1f} %)")
    print(f"  boundary-layer scale sqrt(mu/(rho_e beta)) = {np.sqrt(MU / (rho_e * beta_N)) / dx:.1f} cells")
    print(f"  stagnation heat flux {q_stag:.4f}   Fay-Riddell {qFR(beta_N):.4f} (Newtonian beta, {100 * (q_stag / qFR(beta_N) - 1):+.1f} %),  {qFR(beta_m):.4f} (measured beta, {100 * (q_stag / qFR(beta_m) - 1):+.1f} %)")
    print(f"  q_stag over the last quarter: min {hist[-len(use):, 1].min():.4f}, max {hist[-len(use):, 1].max():.4f}")
    for a in (15, 30, 45, 60):
        mm = np.abs(np.abs(th) - a) < 3
        if mm.any(): print(f"  theta = {a:2d} deg: q/q_stag = {qw[mm].mean() / q_stag:.3f}   p_w/p_0 = {pw[mm].mean() / p0:.3f}  (modified Newtonian {(P + (p0 - P) * np.cos(np.radians(a))**2) / p0:.3f})")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    o = np.argsort(th); ax[0].plot(th[o], qw[o], "k.", ms=3, label="Hydro2 wall cells"); ax[0].axhline(qFR(beta_N), color="C3", ls="--", label="Fay-Riddell, stagnation point")
    ax[0].set_xlabel(r"$\theta$ from the stagnation point [deg]"); ax[0].set_ylabel("wall heat flux"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(th[o], pw[o] / p0, "k.", ms=3, label="Hydro2"); tt = np.linspace(-90, 90, 181); ax[1].plot(tt, (P + (p0 - P) * np.cos(np.radians(tt))**2) / p0, "C3--", label="modified Newtonian")
    ax[1].set_xlabel(r"$\theta$ [deg]"); ax[1].set_ylabel("$p_w / p_{0,2}$"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[2].plot(hist[:, 0], hist[:, 1], "k-"); ax[2].axhline(qFR(beta_N), color="C3", ls="--"); ax[2].set_xlabel("t"); ax[2].set_ylabel("stagnation heat flux"); ax[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "blunt_body.png"), dpi=170); plt.close(fig)
    T = np.where(phi > 0.5, acc["pressure"] / acc["density"] / Rg / Tinf, np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(11, 6)); ext = (x[0], x[-1], y[0], y[-1])
    im = ax[0].imshow(np.where(phi > 0.5, p, np.nan).T / P, origin="lower", extent=ext, cmap="viridis"); fig.colorbar(im, ax=ax[0], shrink=0.8, label="$p/p_\\infty$")
    im = ax[1].imshow(T.T, origin="lower", extent=ext, cmap="inferno"); fig.colorbar(im, ax=ax[1], shrink=0.8, label="$T/T_\\infty$")
    for a in ax: a.set_aspect("equal"); a.set_xlabel("x"); a.set_ylabel("y")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "blunt_body_fields.png"), dpi=170); plt.close(fig)
    print(f"  wrote {outdir}/blunt_body.png, blunt_body_fields.png")


if __name__ == "__main__":
    main()
