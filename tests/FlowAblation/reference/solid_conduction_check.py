#!/usr/bin/env python3
"""
In-depth conduction of the ablating solid (ablation.conduction = 1): two exact 1D checks.

  python3 solid_conduction_check.py contact <run_dir> [--outdir DIR]     (tests/FlowAblation/input_1D_Contact)
  python3 solid_conduction_check.py stefan2 <run_dir> [--outdir DIR]     (tests/FlowAblation/input_1D_Stefan2)
  python3 solid_conduction_check.py inclined <run_dir> [<aligned_closed_gap_run_dir>] [--outdir DIR]    (input_2D_InclinedStefan + the contact overrides:
        stop_time=0.25 amr.plot_dt=0.05 ablation.T_wall=10.0 ablation.conduction=1 ablation.k_s=1.9444444e-2 ablation.cp_s=14.0 ablation.T_solid0=0.99)

contact : two half-spaces brought into contact at t = 0, no ablation.  The surface temperature is
          constant, T_i = (e_g T_g + e_s T_s)/(e_g + e_s), e = sqrt(k rho c), and each side is an
          error-function profile:  T = T_i + (T_far - T_i) erf(|x - x0| / (2 sqrt(alpha t))).
stefan2 : two-phase Stefan problem.  Surface held at the ablation temperature T_a, recession
          s = 2 lam sqrt(alpha_g t) with
              q_g - q_s = rho_s Q* ds/dt,
              q_g = k_g (T_inf - T_a) exp(-lam^2) / ((1 + erf lam) sqrt(pi alpha_g t)),
              q_s = k_s (T_a - T_s0) exp(-lam^2 r) / (erfc(lam sqrt r) sqrt(pi alpha_s t)),  r = alpha_g/alpha_s
          gas   T = T_a + (T_inf - T_a) (erf((x - x0)/(2 sqrt(alpha_g t))) + erf lam)/(1 + erf lam)
          solid T = T_s0 + (T_a - T_s0) erfc((x0 - x)/(2 sqrt(alpha_s t))) / erfc(lam sqrt r)
Both neglect the gas motion and density change (1 % temperature differences are used).
The parameters are read from the run's metadata file.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from scipy.special import erf, erfc
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
RHO, CP, TINF = 1.0, 3.5, 1.0            # gas of input_1D_Stefan


def meta(d, key, default=None):
    val = default                    # command-line overrides are listed after the deck value: the last one wins
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    if val is None: raise SystemExit(f"{key} not in metadata")
    return val


def load(d):
    out = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[1:]:
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        phi = q("phi"); i = np.where(phi >= 0.5)[0][0]
        xs = x[i - 1] + (x[i] - x[i - 1]) * (0.5 - phi[i - 1]) / (phi[i] - phi[i - 1])
        out.append(dict(t=float(ds.current_time), x=x, phi=phi, T=q("T"), Ts=q("ablation_Ts"), qn=q("ablation_q").max(), xs=xs, i=i))
    return out


def inclined(d, outdir):
    """contact test on the periodic slabs |frac(2x + y) - 1/2| < w of input_2D_InclinedStefan (wall at 26.57 deg)"""
    kg = meta(d, "mu0") * CP / meta(d, "thermal.Pr"); ag = kg / (RHO * CP)
    ks, cps, rhos, Ts0 = meta(d, "ablation.k_s"), meta(d, "ablation.cp_s"), meta(d, "ablation.rho_s"), meta(d, "ablation.T_solid0")
    w = meta(d, "solid.phi.ic.expression.constant.xw"); as_ = ks / (rhos * cps)
    eg = np.sqrt(kg * RHO * CP); es = np.sqrt(ks * rhos * cps); Ti = (eg * TINF + es * Ts0) / (eg + es); dT = TINF - Ts0
    pf = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[-1]
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions); t = float(ds.current_time)
    n = ds.domain_dimensions[0]; x = (np.arange(n) + 0.5) / n; X, Y = np.meshgrid(x, x, indexing="ij")
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
    phi = q("phi"); fl = phi >= 0.5; T = np.where(fl, q("T"), q("ablation_Ts"))
    z = 2 * X + Y; dn = (np.abs(z - np.floor(z) - 0.5) - w) / np.sqrt(5.0)          # normal distance from the face (> 0 gas)
    Tex = np.where(dn >= 0, Ti + (TINF - Ti) * erf(dn / (2 * np.sqrt(ag * t))), Ti + (Ts0 - Ti) * erf(-dn / (2 * np.sqrt(as_ * t))))
    dx = 1.0 / n; ok = np.abs(dn) > 1.5 * dx; err = np.abs(T - Tex) / dT
    print(f"{d}: wall at 26.57 deg, t = {t:.3f}, e_s/e_g = {es / eg:.2f}, exact contact temperature {Ti:.6f}")
    print(f"  temperature against the 1D solution along the normal: max error gas {100 * err[ok & fl].max():.2f} %, solid {100 * err[ok & ~fl].max():.2f} % of dT;"
          f"  rms {100 * np.sqrt((err[ok] ** 2).mean()):.3f} %")
    ref = [a for a in sys.argv[3:] if not a.startswith("--") and a != outdir]
    if ref:
        # like-for-like: the grid-aligned run of the same closed periodic layer (input_1D_ClosedGap + the same overrides)
        pr = sorted(glob.glob(os.path.join(ref[0], "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[-1]
        dr = yt.load(pr); cr = dr.covering_grid(0, dr.domain_left_edge, dr.domain_dimensions); L = float(dr.domain_right_edge[0])
        nr = dr.domain_dimensions[0]; xr = (np.arange(nr) + 0.5) * L / nr; qr = lambda nm: np.asarray(cr[("boxlib", nm)])[:, 0, 0]
        Tr = np.where(qr("phi") >= 0.5, qr("T"), qr("ablation_Ts")); dr_ = (np.abs(xr / L - 0.5) - w) * L
        o = np.argsort(dr_); Tal = np.interp(dn, dr_[o], Tr[o]); e2 = np.abs(T - Tal) / dT
        print(f"  against the grid-aligned run of the same closed layer ({ref[0]}, t = {float(dr.current_time):.3f}): max difference gas {100 * e2[ok & fl].max():.2f} %, "
              f"solid {100 * e2[ok & ~fl].max():.2f} % of dT;  rms {100 * np.sqrt((e2[ok] ** 2).mean()):.3f} %")
        Tex = Tal; err = e2
    print(f"  solid area change {abs((~fl).mean() - 2 * w):.2e} (must be ~0: no ablation)")
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.3))
    o = np.argsort(dn.ravel()); dd = dn.ravel()[o]; a = ax[0]
    a.axvspan(dd.min(), 0, color="#B87333", alpha=0.12, lw=0)
    o = np.argsort(dn.ravel()); a.plot(dd, Tex.ravel()[o], "C3-", lw=1.6, label="Grid-Aligned Wall" if ref else "Exact", zorder=3)
    a.plot(dn.ravel()[::37], T.ravel()[::37], "ko", ms=2.5, mfc="none", label="Numerical")
    a.set_xlabel("Distance Normal to the Wall [m]"); a.set_ylabel("Temperature [K]"); a.legend(); a.set_title(f"Temperature at t = {t:.2f} s")
    a = ax[1]; a.semilogy(dn[ok][::7], 100 * np.maximum(err[ok][::7], 1e-12), "k.", ms=1.5)
    a.axvspan(dd.min(), 0, color="#B87333", alpha=0.12, lw=0)
    a.set_xlabel("Distance Normal to the Wall [m]"); a.set_ylabel("Temperature Error [%]"); a.set_title("Difference Relative to the Temperature Difference" if ref else "Error Relative to the Temperature Difference")
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); a.grid(which="minor", alpha=0.15)
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle("Contact of a Gas and a Conducting Solid, Wall at 26.6\N{DEGREE SIGN} to the Grid"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "solid_contact_inclined.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/solid_contact_inclined.png")


def main():
    mode, d = sys.argv[1], sys.argv[2]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    if mode == "inclined": return inclined(d, outdir)
    kg = meta(d, "mu0") * CP / meta(d, "thermal.Pr"); ag = kg / (RHO * CP)
    ks, cps, rhos = meta(d, "ablation.k_s"), meta(d, "ablation.cp_s"), meta(d, "ablation.rho_s")
    Ts0, Ta, Q = meta(d, "ablation.T_solid0"), meta(d, "ablation.T_wall"), meta(d, "ablation.Qstar")
    X0 = meta(d, "solid.phi.ic.expression.constant.xw")
    as_ = ks / (rhos * cps); eg = np.sqrt(kg * RHO * CP); es = np.sqrt(ks * rhos * cps)
    F = load(d); t = np.array([f["t"] for f in F]); L = F[-1]; x = L["x"]; fl = L["phi"] >= 0.5; tp = L["t"]
    Tnum = np.where(fl, L["T"], L["Ts"])
    print(f"{d}: k_g {kg:.4e} alpha_g {ag:.4e} | k_s {ks:.4e} alpha_s {as_:.4e} | e_s/e_g {es / eg:.3f}; {len(F)} plot files to t = {tp:.2f}")

    if mode == "contact":
        Ti = (eg * TINF + es * Ts0) / (eg + es); dT = TINF - Ts0
        Tex = np.where(x >= X0, Ti + (TINF - Ti) * erf((x - X0) / (2 * np.sqrt(ag * tp))),
                                Ti + (Ts0 - Ti) * erf((X0 - x) / (2 * np.sqrt(as_ * tp))))
        # surface temperature of the run: one-sided extrapolation of each side to the surface
        Tw = []
        for f in F:
            i = f["i"]; Tw.append((1.5 * f["T"][i] - 0.5 * f["T"][i + 1], 1.5 * f["Ts"][i - 1] - 0.5 * f["Ts"][i - 2]))
        Tw = np.array(Tw)
        print(f"  exact contact temperature {Ti:.6f};  run at t = {tp:.2f}: gas side {Tw[-1, 0]:.6f}, solid side {Tw[-1, 1]:.6f}"
              f"  ({100 * (Tw[-1].mean() - Ti) / dT:+.2f} % of dT)")
        print(f"  surface moved {abs(L['xs'] - X0):.2e} (must be 0)")
        title = "Contact of a Gas and a Conducting Solid"; name = "solid_contact"
    else:
        r = ag / as_
        g = lambda l: (kg * (TINF - Ta) * np.exp(-l * l) / ((1 + erf(l)) * np.sqrt(np.pi * ag))
                       - ks * (Ta - Ts0) * np.exp(-l * l * r) / (erfc(l * np.sqrt(r)) * np.sqrt(np.pi * as_)) - rhos * Q * l * np.sqrt(ag))
        lam = brentq(g, 1e-9, 5); dT = TINF - Ts0
        lam1 = brentq(lambda l: l * np.exp(l * l) * (1 + erf(l)) - RHO * CP * (TINF - Ta) / (np.sqrt(np.pi) * rhos * Q), 1e-9, 5)
        s = np.array([X0 - f["xs"] for f in F]); s_ex = 2 * lam * np.sqrt(ag * t)
        qn = np.array([f["qn"] for f in F]); qn_ex = rhos * Q * lam * np.sqrt(ag / t)
        Tex = np.where(x >= X0 - s_ex[-1], Ta + (TINF - Ta) * (erf((x - X0) / (2 * np.sqrt(ag * tp))) + erf(lam)) / (1 + erf(lam)),
                                           Ts0 + (Ta - Ts0) * erfc((X0 - x) / (2 * np.sqrt(as_ * tp))) / erfc(lam * np.sqrt(r)))
        print(f"  lam = {lam:.5f}  (without in-depth conduction {lam1:.5f}: the solid takes {100 * (1 - lam / lam1):.0f} % of the recession)")
        for tt in (1.0, 2.0, 4.0, t[-1]):
            j = np.argmin(np.abs(t - tt))
            print(f"  t = {t[j]:5.2f}: recession s = {s[j]:.5f}   exact {s_ex[j]:.5f}   ({100 * (s[j] / s_ex[j] - 1):+.1f} %);"
                  f"   net flux q_g - q_s {qn[j]:.4e}   exact {qn_ex[j]:.4e}   ({100 * (qn[j] / qn_ex[j] - 1):+.1f} %)")
        title = "Two-Phase Stefan Problem: Ablating Solid with In-Depth Conduction"; name = "solid_stefan2"
    # compare away from the cell the surface is in
    ok = np.abs(x - L["xs"]) > 1.5 * (x[1] - x[0])
    err = np.abs(Tnum - Tex) / dT
    print(f"  temperature profile at t = {tp:.2f}: max error gas {100 * err[ok & fl].max():.2f} %, solid {100 * err[ok & ~fl].max():.2f} % of dT = {dT:g}")

    ncol = 2 if mode == "contact" else 3
    fig, ax = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.3))
    a = ax[0]
    a.axvspan(0, L["xs"], color="#B87333", alpha=0.12, lw=0)
    a.plot(x, Tex, "C3-", lw=1.6, label="Exact")
    a.plot(x[::4], Tnum[::4], "ko", ms=4, mfc="none", label="Numerical")
    a.set_xlim(X0 - 0.3, X0 + 0.3); a.set_xlabel("Position [m]"); a.set_ylabel("Temperature [K]"); a.legend()
    a.set_title(f"Temperature at t = {tp:.1f} s")
    a = ax[1]
    a.semilogy(x[ok], 100 * np.maximum(err[ok], 1e-12), "k-", lw=1.2)
    a.axvspan(0, L["xs"], color="#B87333", alpha=0.12, lw=0)
    a.set_xlim(X0 - 0.3, X0 + 0.3); a.set_xlabel("Position [m]"); a.set_ylabel("Temperature Error [%]")
    a.set_title("Error Relative to the Temperature Difference")
    a.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100)); a.grid(which="minor", alpha=0.15)
    if mode != "contact":
        a = ax[2]
        a.plot(t, s_ex * 1e3, "C3-", lw=1.6, label="Exact")
        a.plot(t, 2 * lam1 * np.sqrt(ag * t) * 1e3, "-", color="0.6", lw=1.2, label="Exact, No In-Depth Conduction")
        a.plot(t[::2], s[::2] * 1e3, "ko", ms=4, mfc="none", label="Numerical")
        a.set_xlabel("Time [s]"); a.set_ylabel("Surface Recession [mm]"); a.set_title("Surface Recession"); a.legend()
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/{name}.png")


if __name__ == "__main__":
    main()
