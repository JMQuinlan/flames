#!/usr/bin/env python3
"""
Detonation wave into elastic copper, 1D (tests/FlowEnergetic/input_1D_DetonationCopper; test 4.1 of Wallis, Barton
& Nikiforakis 2021).

  python3 detonation_check.py <run_dir> [--outdir DIR]

Closed forms for the two-phase idealised explosive with complete reaction (reactant: stiffened gas gamma_r, p_inf,
rho_0; products: ideal gas gamma_p; heat Q; energy bookkeeping of Hydro2_Energetic.H):
   q   = (p_0 + gamma_r p_inf)/((gamma_r - 1) rho_0) + Q            energy of the products at the initial state
   Hugoniot of the products   p v/(gamma_p - 1) - q = (p + p_0)(v_0 - v)/2 ,   Rayleigh line  p - p_0 = m^2 (v_0 - v)
   Chapman-Jouguet = the tangent Rayleigh line (found numerically): D_CJ, p_CJ, u_CJ
   von Neumann state = the unreacted shock of speed D_CJ in the stiffened gas
   products at rest behind the Taylor wave (gamma_p isentrope, Riemann invariant):  c = c_CJ - (gamma_p - 1) u_CJ/2
Measured: the front (half of the peak pressure, ahead of the peak) in time, its speed by a straight-line fit over
the second half of the run-up, the peak pressure, the pressure at rest behind the wave, and the profiles of -sigma_xx
(= p - S_xx) at the times of the paper's Fig. 1: 10 us before the wave reaches the copper, at arrival, and 0.4 and
19.5 us after it.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 13, "axes.labelsize": 14, "legend.fontsize": 11})
HERE = os.path.dirname(os.path.abspath(__file__))
GR, PINF, RHO0, GP, Q, P0, XCU = 4.0, 1.0e9, 1600.0, 3.0, 3.68e6, 1.0e5, 0.15


def cj():
    v0 = 1.0 / RHO0; q = (P0 + GR * PINF) / ((GR - 1) * RHO0) + Q
    v = np.linspace(0.52 * v0, 0.999 * v0, 400001)     # the Hugoniot has its asymptote at v = v_0/2 for gamma_p = 3
    p = (q - 0.5 * P0 * (v0 - v)) / (v / (GP - 1) - 0.5 * (v0 - v))          # Hugoniot of the products
    m2 = (p - P0) / (v0 - v); i = int(np.argmin(m2)); D = v0 * np.sqrt(m2[i]); u = D * (1 - v[i] / v0); c = np.sqrt(GP * p[i] * v[i])
    # von Neumann: stiffened-gas shock of speed D
    M2 = D * D / (GR * (P0 + PINF) / RHO0); pvn = (P0 + PINF) * (2 * GR * M2 - (GR - 1)) / (GR + 1) - PINF
    crest = c - 0.5 * (GP - 1) * u; prest = p[i] * (crest / c) ** (2 * GP / (GP - 1))
    return dict(D=D, p=p[i], u=u, c=c, rho=1 / v[i], pvn=pvn, prest=prest, q=q)


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); g = lambda n: np.asarray(cg[("boxlib", n)])[:, 0, 0]
    n = int(ds.domain_dimensions[0]); x = (np.arange(n) + 0.5) * float(ds.domain_right_edge[0]) / n
    return dict(t=float(ds.current_time), x=x, p=g("pressure"), sxx=g("pressure") - g("elastic_Sxx"), u=g("velocityx"), rho=g("density"), mr=g("species_Reactant"), mp=g("species_Products"), mcu=g("species_Cu"))


def main():
    args = sys.argv[1:]; d = args[0]; outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(HERE, "Images", os.path.basename(os.path.normpath(d)))
    os.makedirs(outdir, exist_ok=True); C = cj()
    print("Closed forms: D_CJ = %.1f m/s, p_CJ = %.4e Pa, u_CJ = %.1f m/s, rho_CJ = %.1f, von Neumann pressure %.4e Pa, products at rest %.4e Pa" % (C["D"], C["p"], C["u"], C["rho"], C["pvn"], C["prest"]))
    F = [load(pf) for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))]
    F = [f for i, f in enumerate(F) if i == 0 or f["t"] > F[i - 1]["t"]]
    T, X, PK = [], [], []
    for f in F:
        e = (f["x"] > XCU) & (f["x"] < 0.345); p = np.where(e, f["p"], 0.0); i = int(np.argmax(p))
        if p[i] < 5e9: continue
        k = np.where(p[:i] < 0.5 * p[i])[0]
        if len(k) == 0 or k.max() < np.where(e)[0][0] + 3: break
        T.append(f["t"]); X.append(f["x"][k.max()]); PK.append(p[i])
    T, X, PK = np.array(T), np.array(X), np.array(PK); h = len(T) // 2; c = np.polyfit(T[h:], X[h:], 1); tc = (XCU - c[1]) / c[0]
    print("Front: %d frames; speed over the second half of the run-up %.1f m/s (%+.2f %% of D_CJ); peak pressure there %.3e .. %.3e Pa (von Neumann %.3e); arrival at the copper at %.2f us"
          % (len(T), -c[0], 100 * (-c[0] / C["D"] - 1), PK[h:].min(), PK[h:].max(), C["pvn"], 1e6 * tc))
    m0 = (F[0]["rho"]).sum(); 
    print("Mass drift over the run %.2e; copper mass drift %.2e; reactant left at the end %.2e of the initial" % (F[-1]["rho"].sum() / m0 - 1, F[-1]["mcu"].sum() / F[0]["mcu"].sum() - 1, F[-1]["mr"].sum() / F[0]["mr"].sum()))
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    for a, dtq in zip(ax.ravel(), (-10e-6, 0.0, 0.4e-6, 19.5e-6)):
        f = F[int(np.argmin([abs(g["t"] - (tc + dtq)) for g in F]))]; st = max(1, len(f["x"]) // 800)
        a.plot(f["x"][::st], f["sxx"][::st] / 1e9, "o", ms=2.2, mfc="none", mec="C0", mew=0.6, label="Numerical")
        a.axhline(C["pvn"] / 1e9, color="0.5", lw=0.8, ls=":", label="von Neumann Pressure (Exact)"); a.axhline(C["p"] / 1e9, color="k", lw=0.8, ls="--", label="Chapman-Jouguet Pressure (Exact)")
        a.axhline(C["prest"] / 1e9, color="k", lw=0.8, ls="-.", label="Products at Rest (Exact)"); a.axvline(XCU, color="C1", lw=0.8)
        a.set_xlim(0, 0.4); a.set_ylim(0, 100); a.set_xlabel(r"Position, $x$ [m]"); a.set_ylabel(r"Normal Stress, $-\sigma_{xx}$ [GPa]"); a.grid(alpha=0.3)
        a.set_title(r"%+.1f $\mu$s From Arrival at the Copper (t = %.1f $\mu$s)" % (1e6 * (f["t"] - tc), 1e6 * f["t"]), fontsize=12)
        cu = f["x"] < XCU - 0.01; i = int(np.argmax(np.where(cu, f["sxx"], 0)))
        print("   frame %+6.2f us: max -sigma_xx %.3e Pa at x = %.4f; in the copper %.3e Pa at x = %.4f, velocity there %.1f m/s; products near x = 0.33: %.3e Pa"
              % (1e6 * (f["t"] - tc), f["sxx"].max(), f["x"][int(np.argmax(f["sxx"]))], f["sxx"][i], f["x"][i], f["u"][i], f["p"][3300 * len(f["x"]) // 4000]))
    ax[0, 0].legend(fontsize=10)
    fig.suptitle("Detonation Wave Into Elastic Copper (Setup of Wallis et al. 2021; Copper Left of the Orange Line)"); fig.tight_layout()
    o = os.path.join(outdir, "detonation_copper_profiles.png"); fig.savefig(o, dpi=200); plt.close(fig); print("wrote", o)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(1e6 * T, C["D"] * 0 + X, "o", ms=3, mfc="none", mec="C0", label="Numerical"); ax[0].plot(1e6 * T, X[h] - C["D"] * (T - T[h]), "k-", lw=1.2, label="Chapman-Jouguet Speed (Exact)")
    ax[0].set_xlabel(r"Time [$\mu$s]"); ax[0].set_ylabel("Front Position [m]"); ax[0].set_title("Detonation Front"); ax[0].legend()
    ax[1].plot(1e6 * T, PK / 1e9, "o", ms=3, mfc="none", mec="C3", label="Numerical"); ax[1].axhline(C["pvn"] / 1e9, color="k", lw=1.2, label="von Neumann Pressure (Exact)")
    ax[1].axhline(C["p"] / 1e9, color="k", lw=1.0, ls="--", label="Chapman-Jouguet Pressure (Exact)"); ax[1].set_xlabel(r"Time [$\mu$s]"); ax[1].set_ylabel("Peak Pressure [GPa]"); ax[1].set_title("Peak Pressure of the Wave"); ax[1].legend(fontsize=10)
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); o = os.path.join(outdir, "detonation_copper_front.png"); fig.savefig(o, dpi=200); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
