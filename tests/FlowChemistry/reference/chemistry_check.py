#!/usr/bin/env python3
"""
One-step chemistry on the species (chem.on = 1) -- unit checks.

  python3 chemistry_check.py reactor    <run_dir> [<run_dir> ...] [--outdir DIR]    tests/FlowChemistry/input_0D_Reactor
  python3 chemistry_check.py detonation <run_dir> [<run_dir> ...] [--outdir DIR]    tests/FlowChemistry/input_1D_Detonation

reactor   : constant volume, dY/dt = -A Y exp(-T_a/T), T = T_0 + Q (1 - Y)/cv.  T_a = 0: Y = exp(-A t) (closed form);
            T_a > 0: the same scalar ODE integrated by scipy (rtol 1e-12).  Also rho E = rho cv T_0 + rho Q (1 - Y).
detonation: Chapman-Jouguet speed D_CJ = c_0 (sqrt(1 + h) + sqrt(h)), h = (gamma^2 - 1) Q/(2 gamma R T_0); von Neumann and
            CJ states from the Rankine-Hugoniot relations; reaction-zone structure from the ZND equation
            d lam/dx = A (1 - lam) exp(-T_a/T)/w  along the Rayleigh line (lam = burnt fraction, w = speed in the wave frame).
Parameters are read from the run's metadata.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import yt
yt.funcs.mylog.setLevel(40)


def meta(d, key, default=None):
    val = default
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m:
            try: val = float(m.group(1))
            except ValueError: val = m.group(1)
    return val


def files(d): return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def reactor(dirs, outdir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); cols = ["k", "C0", "C2", "C4"]
    for n, d in enumerate(dirs):
        A, Ta, Q, cv = meta(d, "chem.A"), meta(d, "chem.Ta", 0.0), meta(d, "chem.Q"), meta(d, "eos1.cv"); fuel = meta(d, "chem.reactant")
        H = []
        for pf in files(d):
            ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
            Y = q("Y_" + fuel); H.append((float(ds.current_time), Y.mean(), q("T").mean(), q("pressure").mean(), q("energy_per_vol").mean(), q("density").mean(),
                                          Y.max() - Y.min(), np.abs(q("species_" + fuel) + q("species_" + meta(d, "chem.product")) - q("rho_eta1")).max()))
        H = np.array(H); t = H[:, 0]; T0 = H[0, 2]; rho = H[0, 5]
        s = solve_ivp(lambda tt, y: [-A * y[0] * np.exp(-Ta / (T0 + Q * (1 - y[0]) / cv))], (0, t[-1]), [1.0], rtol=1e-12, atol=1e-14, dense_output=True, method="LSODA")
        Yex = s.sol(t)[0]; Tex = T0 + Q * (1 - Yex) / cv; Eex = H[0, 4] + rho * Q * (1 - Yex)
        print(f"{d}: A = {A:g}, T_a = {Ta:g}, Q = {Q:g};  {len(t)} plot files to t = {t[-1]:.3f}")
        print(f"  max |Y - exact| = {np.abs(H[:, 1] - Yex).max():.2e};  max |T - exact|/dT = {np.abs(H[:, 2] - Tex).max() / (Tex[-1] - T0):.2e};"
              f"  max |rho E - exact|/(rho Q) = {np.abs(H[:, 4] - Eex).max() / (rho * Q):.2e}")
        if Ta > 0:
            ti = lambda tt, TT: tt[np.argmax(np.gradient(TT, tt))]
            tf = np.linspace(0, t[-1], 200001); print(f"  ignition time (steepest temperature rise): {ti(t, H[:, 2]):.4f}   exact {ti(tf, T0 + Q * (1 - s.sol(tf)[0]) / cv):.4f}")
        print(f"  spatial spread of Y {H[:, 6].max():.1e};  fuel + product - phase mass {H[:, 7].max():.1e};  pressure / (rho R T) - 1 = {np.abs(H[:, 3] / (H[:, 5] * (meta(d, 'eos1.cp') - cv) * H[:, 2]) - 1).max():.1e}")
        tt = np.linspace(0, t[-1], 2000); st = max(1, len(t) // 45); lab = rf"$T_a$ = {Ta:g}"
        ax[0].plot(tt, s.sol(tt)[0], "-", color="C3", lw=1.5, label="Exact" if n == 0 else None); ax[0].plot(t[::st], H[::st, 1], "o", color=cols[n % 4], ms=3.5, mfc="none", label="Numerical, " + lab)
        ax[1].plot(tt, T0 + Q * (1 - s.sol(tt)[0]) / cv, "-", color="C3", lw=1.5, label="Exact" if n == 0 else None); ax[1].plot(t[::st], H[::st, 2], "o", color=cols[n % 4], ms=3.5, mfc="none", label="Numerical, " + lab)
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("Fuel Mass Fraction [-]"); ax[0].set_title("Fuel Consumption"); ax[0].legend()
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Temperature [K]"); ax[1].set_title("Temperature"); ax[1].legend()
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Constant-Volume Reactor with a One-Step Arrhenius Reaction"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "reactor.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/reactor.png")


def cj(g, R, Q, Ta, A, rho0=1.0, p0=1.0):
    T0 = p0 / (rho0 * R); c0 = np.sqrt(g * p0 / rho0); h = (g * g - 1) * Q / (2 * g * R * T0); M = np.sqrt(1 + h) + np.sqrt(h); D = M * c0; m = rho0 * D
    def state(lam):
        v0 = 1 / rho0; a = m * m * (0.5 - g / (g - 1)); b = g / (g - 1) * (p0 + m * m * v0); c = -(g / (g - 1) * p0 * v0 + 0.5 * m * m * v0 * v0 + lam * Q)
        disc = max(b * b - 4 * a * c, 0.0); v = min((-b + np.sqrt(disc)) / (2 * a), (-b - np.sqrt(disc)) / (2 * a))
        p = p0 + m * m * (v0 - v); return 1 / v, p, p * v / R, m * v
    f = lambda x, l: [A * (1 - l[0]) * np.exp(-Ta / state(min(l[0], 1.0))[2]) / state(min(l[0], 1.0))[3]]
    s = solve_ivp(f, (0, 60.0), [0.0], rtol=1e-10, atol=1e-12, dense_output=True)
    return dict(D=D, M=M, vN=state(0.0), CJ=state(1.0), state=state, lam=lambda x: np.clip(s.sol(np.clip(x, 0, 60.0))[0], 0, 1))


def detonation(dirs, outdir):
    rows = []
    for d in dirs:
        g = meta(d, "eos1.gamma"); R = meta(d, "eos1.cp") - meta(d, "eos1.cv"); fuel = meta(d, "chem.reactant")
        Z = cj(g, R, meta(d, "chem.Q"), meta(d, "chem.Ta", 0.0), meta(d, "chem.A")); H = []; last = None
        for pf in files(d):
            ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            n = ds.domain_dimensions[0]; L = float(ds.domain_right_edge[0]); xa = (np.arange(n) + 0.5) * L / n
            # ignited in the middle (symmetry plane = closed end): analyse the right-going wave, x measured from the mid-plane
            half = xa > 0.5 * L; x = xa[half] - 0.5 * L; n = int(half.sum()); L = 0.5 * L
            q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0][half]
            p, Y = q("pressure"), q("Y_" + fuel)
            k = np.where(p > 0.5 * (1.0 + Z["vN"][1]))[0]
            if len(k) == 0: continue
            i = k[-1]; xf = x[i] + (x[i + 1] - x[i]) * (p[i] - 0.5 * (1 + Z["vN"][1])) / max(p[i] - p[i + 1], 1e-30) if i + 1 < n else x[i]       # leading shock: last crossing of the half-spike level
            H.append((float(ds.current_time), xf, p.max(), (q("species_" + fuel)).sum() * L / n, q("energy_per_vol").sum() * L / n, q("density").sum() * L / n, p[0]))
            last = (float(ds.current_time), x, p, Y, q("density"), q("velocityx"), q("T"), xf)
        H = np.array(H); t = H[:, 0]; m = t > 0.5 * t[-1]
        D = np.polyfit(t[m], H[m, 1], 1)[0]; Dloc = np.gradient(H[:, 1], t)
        tl, x, p, Y, rho, u, T, xf = last; dx = x[1] - x[0]
        # ZND profile behind the leading shock: xi = distance behind the shock
        xi = xf - x; zone = (xi > 0) & (xi < 8.0); lam = Z["lam"](xi[zone]); S = np.array([Z["state"](l) for l in lam])
        ep = np.abs(p[zone] - S[:, 1]); eY = np.abs(Y[zone] - (1 - lam)); far = xi[zone] > 4 * dx           # skip the captured shock itself
        row = dict(d=d, n=len(x), cells=1.0 / dx, D=D, Dex=Z["D"], pmax=np.median(H[m, 2]), pvN=Z["vN"][1], ep=100 * ep[far].max() / Z["vN"][1], eY=eY[far].max(),
                   Z=Z, x=x, p=p, Y=Y, xf=xf, t=t, H=H, Dloc=Dloc, tl=tl, rho=rho, u=u)
        # burnt-gas plateau between the Taylor wave and the wall does not exist with an open boundary: report the CJ point instead
        kcj = np.argmin(np.abs(xi - 6.0)); row["pcj"] = p[kcj]
        rows.append(row)
        print(f"{d}: {len(x)} cells, {row['cells']:.0f} per half-reaction length, t = {tl:.2f}")
        print(f"  front speed over the second half of the run: {D:.4f}   D_CJ = {Z['D']:.4f}   ({100 * (D / Z['D'] - 1):+.2f} %);   last instantaneous value {Dloc[-2]:.4f}")
        print(f"  peak pressure (median) {row['pmax']:.2f}   von Neumann {Z['vN'][1]:.2f}  ({100 * (row['pmax'] / Z['vN'][1] - 1):+.1f} %);   pressure 6 lengths behind the shock {row['pcj']:.2f}  (CJ {Z['CJ'][1]:.2f})")
        print(f"  reaction zone against ZND (0 < xi < 8, beyond 4 cells from the shock): max |p - p_ZND| = {row['ep']:.1f} % of p_vN;  max |Y - Y_ZND| = {row['eY']:.3f}")
        fb = H[:, 3]; QF = (fb[0] - fb[-1]) * meta(d, "chem.Q")
        print(f"  bookkeeping (half domain): fuel burnt x Q = {QF:.3f}, energy rise {H[-1, 4] - H[0, 4]:.3f}  ({100 * ((H[-1, 4] - H[0, 4]) / QF - 1):+.4f} %);  mass change {H[-1, 5] / H[0, 5] - 1:+.2e}")
        # Taylor wave: gas at rest at the closed end, p_wall = p_CJ (1 - (gamma - 1)/2 u_CJ/c_CJ)^(2 gamma/(gamma - 1))
        rcj, pcj_, Tcj, wcj = Z["CJ"]; ccj = np.sqrt(g * pcj_ / rcj); ucj = Z["D"] - wcj; pw = pcj_ * (1 - 0.5 * (g - 1) * ucj / ccj) ** (2 * g / (g - 1))
        print(f"  pressure at the closed end {H[-1, 6]:.3f}   Taylor-wave value {pw:.3f}  ({100 * (H[-1, 6] / pw - 1):+.1f} %)")
    with open(os.path.join(outdir, "detonation_summary.csv"), "w") as fo:
        fo.write("cells_per_half_reaction_length,D,D_CJ,D_err_pct,p_peak,p_vN,zone_p_err_pct_of_pvN,zone_Y_err\n")
        for r in rows: fo.write(f"{r['cells']:.4g},{r['D']:.6g},{r['Dex']:.6g},{100 * (r['D'] / r['Dex'] - 1):.4g},{r['pmax']:.5g},{r['pvN']:.5g},{r['ep']:.4g},{r['eY']:.4g}\n")
    r = rows[-1]; Z = r["Z"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    c = ax[0]; c.axhline(Z["D"], color="C3", lw=1.5, label="Exact (Chapman-Jouguet)")
    for k, q in enumerate(rows): c.plot(q["t"][1:-1], q["Dloc"][1:-1], "o", color=["k", "C0", "C2"][k % 3], ms=3.5, mfc="none", label=f"Numerical, {q['cells']:.0f} Cells per Half-Reaction Length")
    c.set_ylim(0.7 * Z["D"], 1.3 * Z["D"]); c.set_xlabel("Time [s]"); c.set_ylabel("Front Speed [m/s]"); c.set_title("Detonation Speed"); c.legend(fontsize=8)
    c = ax[1]; xi = r["xf"] - r["x"]; m = (xi > -1.5) & (xi < 9); xx = np.linspace(0, 9, 600); S = np.array([Z["state"](l) for l in Z["lam"](xx)])
    c.plot(np.concatenate([[-1.5, 0, 0], xx]), np.concatenate([[1, 1, Z["vN"][1]], S[:, 1]]), "C3-", lw=1.5, label="Exact (ZND)"); c.plot(xi[m], r["p"][m], "ko", ms=3.5, mfc="none", label="Numerical")
    c.set_xlabel("Distance Behind the Shock / Half-Reaction Length [-]"); c.set_ylabel("Pressure [Pa]"); c.set_title("Reaction Zone: Pressure"); c.legend(); c.invert_xaxis()
    c = ax[2]; c.plot(np.concatenate([[-1.5, 0], xx]), np.concatenate([[1, 1], 1 - Z["lam"](xx)]), "C3-", lw=1.5, label="Exact (ZND)"); c.plot(xi[m], r["Y"][m], "ko", ms=3.5, mfc="none", label="Numerical")
    c.set_xlabel("Distance Behind the Shock / Half-Reaction Length [-]"); c.set_ylabel("Fuel Mass Fraction [-]"); c.set_title("Reaction Zone: Fuel"); c.legend(); c.invert_xaxis()
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Chapman-Jouguet Detonation with One-Step Chemistry"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "detonation.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/detonation.png")


if __name__ == "__main__":
    mode = sys.argv[1]; args = sys.argv[2:]
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[-1])))
    os.makedirs(outdir, exist_ok=True)
    dict(reactor=reactor, detonation=detonation)[mode](dirs, outdir)
