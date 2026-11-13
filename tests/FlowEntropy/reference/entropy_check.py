#!/usr/bin/env python3
"""
Entropy diagnostic (entropy.on = 1, plot field "entropy" = entropy per unit volume) -- four checks.

  python3 entropy_check.py heater  <run_dir> [--outdir DIR]   closed box, plate and gas at different temperatures
                                                              (FlowConjugateHeat/input_1D_ClosedBoxEquilibrium + entropy.on=1)
  python3 entropy_check.py couette <run_dir> [...]            moving wall, ADIABATIC walls: the wall does work, the gas heats
                                                              (FlowConjugateHeat/input_1D_CouetteHeating solid.T_wall=-1 entropy.on=1)
  python3 entropy_check.py shock   <run_dir> [...]            entropy through a viscous shock (FlowConjugateHeat/input_1D_BeckerShock)
  python3 entropy_check.py fan     <run_dir>                  a body stirring a closed box (FlowMixingCylinder/UNIT_TEST_2D)

heater : S = int_gas s dV + rho_s cp_s int_solid ln(T_s) dV must rise monotonically to
           S_end - S_0 = m_g cv ln(T_eq/T_g0) + C_s ln(T_eq/T_s0) > 0        (gas at constant volume, uniform at the end).
couette: steady shear between adiabatic walls: the moving wall does work  dE/dt = mu U^2/h  per unit wall area (exact),
         and  dS/dt = int (Phi/T + k |grad T|^2/T^2) dy  with Phi = mu (du/dy)^2  (evaluated from the run's own profiles).
shock  : s = cv ln(p/rho^gamma) along the Becker (Pr = 3/4) profile; jump cv ln((p2/p1)(rho1/rho2)^gamma) and the
         entropy MAXIMUM inside a heat-conducting shock.
fan    : total entropy and energy of the gas against time; the entropy must not decrease.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
CV, CP, GAM = 2.5, 3.5, 1.4


def meta(d, key, default=None):
    val = default
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def files(d):
    return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def out_dir(args, d):
    here = os.path.dirname(os.path.abspath(__file__))
    o = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(here, "Images", os.path.basename(os.path.normpath(d)))
    os.makedirs(o, exist_ok=True); return o


def heater(d, outdir):
    rc = meta(d, "ablation.rho_s") * meta(d, "ablation.cp_s"); Ts0 = meta(d, "ablation.T_solid0"); H = []
    for pf in files(d):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; dx = float(ds.domain_right_edge[0]) / n
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        fl = q("phi") >= 0.5; S = q("entropy")
        H.append((float(ds.current_time), S[fl].sum() * dx, S[~fl].sum() * dx, q("density")[fl].sum() * dx, q("T")[fl].mean(), q("ablation_Ts")[~fl].mean()))
    H = np.array(H); t = H[:, 0]; Cs = rc * (~fl).sum() * dx; mg = H[0, 3]; Tg0 = H[0, 4]
    Teq = (mg * CV * Tg0 + Cs * Ts0) / (mg * CV + Cs)
    dSg, dSs = mg * CV * np.log(Teq / Tg0), Cs * np.log(Teq / Ts0); St = H[:, 1] + H[:, 2]
    print(f"{d}: plate at {Ts0:g}, gas at {Tg0:g}  ->  T_eq = {Teq:.6f}")
    print(f"  entropy change of the gas   {H[-1, 1] - H[0, 1]:+.6e}   exact {dSg:+.6e}   ({100 * ((H[-1, 1] - H[0, 1]) / dSg - 1):+.3f} %)")
    print(f"  entropy change of the solid {H[-1, 2] - H[0, 2]:+.6e}   exact {dSs:+.6e}   ({100 * ((H[-1, 2] - H[0, 2]) / dSs - 1):+.3f} %)")
    print(f"  TOTAL entropy produced      {St[-1] - St[0]:+.6e}   exact {dSg + dSs:+.6e}   ({100 * ((St[-1] - St[0]) / (dSg + dSs) - 1):+.3f} %);"
          f"   smallest increment between plot files {np.diff(St).min():+.2e}  ({'monotone' if np.diff(St).min() >= -1e-13 else 'DECREASES'})")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(t, H[:, 1] - H[0, 1], "k--", lw=1.3, label="Gas"); ax[0].plot(t, H[:, 2] - H[0, 2], "k:", lw=1.8, label="Solid")
    ax[0].plot(t, St - St[0], "k-", lw=1.6, label="Total")
    ax[0].axhline(dSg, color="C3", lw=0.8); ax[0].axhline(dSs, color="C3", lw=0.8); ax[0].axhline(dSg + dSs, color="C3", lw=1.6, label="Exact Final Values")
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel(r"Entropy Change [J/(K m$^2$)]"); ax[0].set_title("Entropy of the Gas and of the Plate"); ax[0].legend()
    ax[1].axhline(dSg + dSs, color="C3", lw=1.6, label="Exact Final Value"); ax[1].plot(t, St - St[0], "ko", ms=3.5, mfc="none", label="Numerical")
    ax[1].set_xscale("symlog", linthresh=1.0); ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel(r"Total Entropy Produced [J/(K m$^2$)]"); ax[1].set_title("Total Entropy Rises to Its Equilibrium Value"); ax[1].legend()
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Closed Adiabatic Box: " + ("A Hot Plate Heats the Gas" if Ts0 > Tg0 else "A Cold Plate Cools the Gas")); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "entropy_heater.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/entropy_heater.png")


def couette(d, outdir):
    mu, Pr, U, h = meta(d, "mu0"), meta(d, "thermal.Pr"), meta(d, "solid.momentum.ic.expression.constant.U"), 0.1
    k = mu * CP / Pr; H = []
    for pf in files(d):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        ny = ds.domain_dimensions[1]; dy = float(ds.domain_width[1]) / ny
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[0, :, 0]
        fl = q("phi") >= 0.5; u, T = q("velocityx")[fl], q("T")[fl]
        ue = np.concatenate([[2 * U - u[0]], u, [-u[-1]]]); Te = np.concatenate([[T[0]], T, [T[-1]]])       # no-slip / adiabatic ghosts
        uy = (ue[2:] - ue[:-2]) / (2 * dy); Ty = (Te[2:] - Te[:-2]) / (2 * dy)
        prod = ((mu * uy**2) / T + k * Ty**2 / T**2).sum() * dy
        H.append((float(ds.current_time), q("energy_per_vol")[fl].sum() * dy, q("entropy")[fl].sum() * dy, prod, T.mean(), mu * (2 * (U - u[0]) / dy) * U, q("density")[fl].sum() * dy))
    H = np.array(H); t = H[:, 0]; m = t > 1.0
    dE = np.polyfit(t[m], H[m, 1], 1)[0]; Wex = mu * U * U / h
    dSdt = np.gradient(H[:, 2], t); tm = 0.5 * (t[1:] + t[:-1]); rate = np.diff(H[:, 2]) / np.diff(t); pm = 0.5 * (H[1:, 3] + H[:-1, 3])
    mm = tm > 1.0
    print(f"{d}: adiabatic walls, lower wall at U = {U:g}; {int(fl.sum())} cells across")
    print(f"  work done on the gas, dE/dt for t > 1: {dE:.6e}   exact mu U^2/h = {Wex:.6e}   ({100 * (dE / Wex - 1):+.3f} %);   gas mass change {H[-1, 6] / H[0, 6] - 1:+.1e}")
    print(f"  entropy rate dS/dt against int (Phi/T + k|grad T|^2/T^2) dy from the profiles, t > 1: mean ratio {np.mean(rate[mm] / pm[mm]):.4f}  (min {np.min(rate[mm] / pm[mm]):.4f}, max {np.max(rate[mm] / pm[mm]):.4f})")
    print(f"  total entropy rise {H[-1, 2] - H[0, 2]:+.5e};  smallest increment between plot files {np.diff(H[:, 2]).min():+.2e}  ({'monotone' if np.diff(H[:, 2]).min() >= 0 else 'DECREASES'});"
          f"  mean temperature {H[0, 4]:.4f} -> {H[-1, 4]:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(t, H[0, 1] + Wex * (t - t[m][0]) + (H[m, 1][0] - H[0, 1]), "C3-", lw=1.6, label=r"Exact Slope $\mu U^2/h$"); ax[0].plot(t[::2], H[::2, 1], "ko", ms=3.5, mfc="none", label="Numerical")
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel(r"Total Energy of the Gas [J/m$^2$]"); ax[0].set_title("Work Done by the Moving Wall"); ax[0].legend()
    ax[1].plot(tm, pm, "C3-", lw=1.6, label="Dissipation and Conduction Integral"); ax[1].plot(tm[::2], rate[::2], "ko", ms=3.5, mfc="none", label="Numerical dS/dt")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel(r"Entropy Production Rate [W/(K m$^2$)]"); ax[1].set_title("Entropy Production"); ax[1].legend(); ax[1].set_ylim(0, None)
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Shear Flow Between Adiabatic Walls: The Moving Wall Heats the Gas"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "entropy_couette.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/entropy_couette.png")


def shock(d, outdir):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "FlowConjugateHeat", "reference"))
    import becker_shock_check as bk
    a = bk.analyse(d); B = a["B"]; th = B["thick"]
    ds = yt.load(files(d)[-1]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
    s = q("entropy") / q("density"); x = a["x"]
    sex = lambda w: CV * np.log((B["m"] / w) * (CP - CV) * B["T"](w) / (B["m"] / w) ** GAM)
    s1 = sex(B["w1"]); s2 = sex(B["w2"]); ww = np.linspace(B["w2"], B["w1"], 20001); smax = sex(ww).max()
    win = np.abs(x) < 3 * th; k = np.argmin(np.abs(x + 3 * th)); k1 = np.argmin(np.abs(x - 3 * th))
    print(f"{d}: {th / (x[1] - x[0]):.1f} cells per shock thickness")
    print(f"  entropy jump across the shock  {s[k] - s[k1]:.6f}   exact cv ln((p2/p1)(rho1/rho2)^gamma) = {s2 - s1:.6f}   ({100 * ((s[k] - s[k1]) / (s2 - s1) - 1):+.2f} %)")
    print(f"  entropy maximum inside the shock, above the upstream value: {s[win].max() - s[k1]:.6f}   exact {smax - s1:.6f}   ({100 * ((s[win].max() - s[k1]) / (smax - s1) - 1):+.2f} %)"
          f"   = {(smax - s1) / (s2 - s1):.2f} x the jump")
    print(f"  profile: max |s - exact| within 3 thicknesses = {100 * np.abs(s - sex(a['wex']))[win].max() / (s2 - s1):.2f} % of the jump")
    fig, ax = plt.subplots(figsize=(6.2, 4.4)); xx = np.linspace(-6 * th, 6 * th, 800); m = np.abs(x) < 6 * th
    ax.plot(xx / th, sex(B["w"](xx)) - s1, "C3-", lw=1.6, label="Exact"); ax.plot(x[m] / th, s[m] - s1, "ko", ms=4, mfc="none", label="Numerical")
    ax.set_xlabel("Distance / Shock Thickness [-]"); ax.set_ylabel("Specific Entropy Above Upstream [J/(kg K)]"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Entropy Through a Mach 2 Viscous Shock (Pr = 3/4)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "entropy_shock.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/entropy_shock.png")


def fan(d, outdir):
    # The moving body is not conservative (cells are covered and uncovered: the gas mass drifts) and the stored entropy
    # carries an arbitrary constant per unit mass, so the raw integral follows the mass.  Entropy PRODUCED is measured
    # against the initial specific entropy of each gas:  S' = S - s_0,ref M_0(t) - s_1,ref M_1(t).
    g = [meta(d, "eos0.gamma"), meta(d, "eos1.gamma")]; cv = [meta(d, "eos0.cv"), meta(d, "eos1.cv")]; pinf = [meta(d, "eos0.p0", 0.0), meta(d, "eos1.p0", 0.0)]
    H = []; sref = None; single = False
    for pf in files(d):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        dA = float(np.prod(np.asarray(ds.domain_width[:2]) / np.asarray(ds.domain_dimensions[:2])))
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
        fl = q("phi") >= 0.5; eta = np.clip(q("eta"), 1e-12, 1 - 1e-12); r0, r1 = q("rho_eta0"), q("rho_eta1")
        if sref is None:          # initial specific entropies from the (uniform) initial phase states
            p = q("pressure")
            m0, m1 = fl & (eta > 0.5), fl & (eta < 0.5)
            single = not (m0.any() and m1.any())          # one gas carried as two identical phases (eta uniform)
            if single: m0 = m1 = fl
            sref = [cv[0] * np.log((p[fl] + pinf[0]).mean() / np.median((r0 / eta)[m0]) ** g[0]),
                    cv[1] * np.log((p[fl] + pinf[1]).mean() / np.median((r1 / (1 - eta))[m1]) ** g[1])]
            S = sref[0] * r0[fl].sum() * dA + sref[1] * r1[fl].sum() * dA       # the step-0 field of older binaries is empty
        else: S = q("entropy")[fl].sum() * dA
        M0, M1 = r0[fl].sum() * dA, r1[fl].sum() * dA
        H.append((float(ds.current_time), S - sref[0] * M0 - sref[1] * M1, q("energy_per_vol")[fl].sum() * dA, M0 + M1, (q("density") * q("T"))[fl].sum() / q("density")[fl].sum(), S))
    H = np.array(H); t = H[:, 0]; dS = np.diff(H[:, 1])
    print(f"{d}: {len(H)} plot files to t = {t[-1]:.2f}")
    print(f"  entropy produced (against the initial specific entropy of each gas): {H[-1, 1]:+.5e}  = {100 * H[-1, 1] / abs(H[0, 5]):+.4f} % of |S(0)|;"
          f"  smallest increment {dS.min():+.2e}, {int((dS < 0).sum())} of {len(dS)} intervals decrease")
    print(f"  total energy {H[0, 2]:.6e} -> {H[-1, 2]:.6e} ({100 * (H[-1, 2] / H[0, 2] - 1):+.4f} %);  gas mass {100 * (H[-1, 3] / H[0, 3] - 1):+.4f} %;  mass-mean temperature {H[0, 4]:.5f} -> {H[-1, 4]:.5f}")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(t, H[:, 1], "ko-", ms=3.5, mfc="none", lw=0.8); ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("Entropy Produced in the Gas [J/(K m)]"); ax[0].set_title("Entropy Production")
    ax[1].plot(t, H[:, 2] - H[0, 2], "ko-", ms=3.5, mfc="none", lw=0.8); ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Energy Change of the Gas [J/m]"); ax[1].set_title("Work Done on the Gas")
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("A Cylinder Stirring " + ("a Single Gas" if single else "Two Gases") + " in a Closed Channel"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "entropy_fan.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/entropy_fan.png")


if __name__ == "__main__":
    mode = sys.argv[1]; args = sys.argv[2:]
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    for d in dirs: dict(heater=heater, couette=couette, shock=shock, fan=fan)[mode](d, out_dir(args, d))
