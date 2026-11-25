#!/usr/bin/env python3
"""
Hydrogen-oxygen chemistry with a PelePhysics mechanism (species.eos = pele, chem.backend = pele) -- tests/FlowChemistry.
Reference solutions from Cantera with the SAME mechanism (mechanism.yaml of the PelePhysics mechanism directory).
Run with the Cantera environment:   bin/tests/venv_cantera/bin/python h2o2_check.py <mode> ...

  ignition   <run_dir> [--outdir DIR]         input_0D_H2O2Ignition: T, p, species against a constant-volume reactor
  cj         [--p0 6670 --T0 298 --X "H2:2,O2:1,AR:7"]
                                              Chapman-Jouguet speed and state, von Neumann state, ZND structure
                                              (induction length = distance from the shock to the maximum thermicity)
  table      [--out FILE --length 0.3 --p0 .. --T0 .. --X ..]
                                              ZND profile as a table for the profile1d initial condition
  detonation <run_dir> [<run_dir> ...] [--outdir DIR] [--p0 .. --T0 .. --X ..] [--x0 X]
                                              input_1D_H2O2Detonation: front speed against the CJ speed, peak pressure
                                              against the von Neumann pressure, profile behind the front against ZND

CJ: the equilibrium Hugoniot is followed in specific volume; for each v2 the temperature that satisfies
h2 - h1 = (p2 - p1)(v1 + v2)/2 with the products in chemical equilibrium at (T2, v2) gives the wave speed
D^2 = v1^2 (p2 - p1)/(v1 - v2); the CJ speed is its minimum.
ZND: steady reactive Euler equations in the shock frame, thermicity form, frozen sound speed,
  d rho/dt = -rho sigma/(1 - M^2),  dY_i/dt = W_i omega_i/rho,  sigma = sum (W/W_i - h_i/(c_p T)) dY_i/dt,  dx/dt = w.
"""
import os, sys, glob
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import solve_ivp
import cantera as ct

HERE = os.path.dirname(os.path.abspath(__file__))
MECH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "ext", "AMReX-Combustion", "PelePhysics", "Mechanisms", "BurkeDryer", "mechanism.yaml"))


def opt(args, key, default):
    return type(default)(args[args.index(key) + 1]) if key in args else default


def positional(args, flags):
    out, skip = [], False
    for a in args:
        if skip: skip = False; continue
        if a in flags: skip = True; continue
        out.append(a)
    return out


def cj_state(p0, T0, X):
    g = ct.Solution(MECH); g.TPX = T0, p0, X
    h1, v1, Y1 = g.enthalpy_mass, 1.0 / g.density, g.Y.copy()

    def hug(v2):
        def f(T):
            g.TDY = T, 1.0 / v2, Y1; g.equilibrate("TV")
            return g.enthalpy_mass - h1 - 0.5 * (g.P - p0) * (v1 + v2)
        T = brentq(f, 1200.0, 6000.0, xtol=1e-9)
        g.TDY = T, 1.0 / v2, Y1; g.equilibrate("TV")
        return v1 * np.sqrt((g.P - p0) / (v1 - v2)), T
    res = minimize_scalar(lambda r: hug(v1 / r)[0], bounds=(1.4, 2.2), method="bounded", options=dict(xatol=1e-7))
    D, T2 = hug(v1 / res.x); g.TDY = T2, res.x / v1, Y1; g.equilibrate("TV")
    out = dict(D=D, p=g.P, T=g.T, rho=g.density, u=D * (1 - 1.0 / res.x), rho1=1.0 / v1, Y1=Y1, c_eq=g.sound_speed)
    # von Neumann state: frozen shock at speed D
    g.TPX = T0, p0, X; gam = g.cp / g.cv

    def vn(r):                                      # r = rho2/rho1; momentum + energy with frozen composition
        p2 = p0 + D * D / v1 * (1 - 1 / r); h2 = h1 + 0.5 * D * D * (1 - 1 / r**2)
        g.HPY = h2, p2, Y1
        return g.density * v1 - r
    r = brentq(vn, 1.5, (gam + 1) / (gam - 1) * 3)
    vn(r); out.update(p_vN=g.P, T_vN=g.T, rho_vN=g.density)
    return out


def znd(cj, p0, T0, X, t_end=None):
    g = ct.Solution(MECH); D = cj["D"]; rho1 = cj["rho1"]
    g.TDY = cj["T_vN"], cj["rho_vN"], cj["Y1"]; W = g.molecular_weights

    def rhs(t, y):
        rho, Y = y[0], y[1:-1]
        p = p0 + rho1 * D * D * (1 - rho1 / rho)
        g.DPY = rho, p, np.maximum(Y, 0)
        w = D * rho1 / rho; c = g.sound_speed; eta = 1 - (w / c) ** 2          # frozen sound speed
        dY = g.net_production_rates * W / rho
        sig = np.sum((g.mean_molecular_weight / W - g.partial_molar_enthalpies / W / (g.cp_mass * g.T)) * dY)
        return np.concatenate(([-rho * sig / eta], dY, [w]))
    y0 = np.concatenate(([cj["rho_vN"]], cj["Y1"], [0.0]))
    if t_end is None: t_end = 2.0e-3
    stop = lambda t, y: 1 - (D * rho1 / y[0]) ** 2 / _c2(g, y, p0, rho1, D) - 2e-3; stop.terminal = True
    s = solve_ivp(rhs, (0, t_end), y0, method="LSODA", rtol=1e-9, atol=1e-13, events=stop, max_step=t_end / 2000)
    rho = s.y[0]; x = s.y[-1]; p = p0 + rho1 * D * D * (1 - rho1 / rho); T = np.empty_like(rho); th = np.empty_like(rho); Yw = np.empty_like(rho)
    iw = g.species_index("H2O")
    for k in range(len(rho)):
        g.DPY = rho[k], p[k], np.maximum(s.y[1:-1, k], 0); T[k] = g.T; Yw[k] = g.Y[iw]
        th[k] = abs(rhs(0, s.y[:, k])[0]) / rho[k]
    return dict(t=s.t, x=x, rho=rho, p=p, T=T, u=D * (1 - rho1 / rho), YH2O=Yw, L_ind=x[np.argmax(th)], t_ind=s.t[np.argmax(th)])


def _c2(g, y, p0, rho1, D):
    p = p0 + rho1 * D * D * (1 - rho1 / y[0]); g.DPY = y[0], p, np.maximum(y[1:-1], 0)
    return g.sound_speed ** 2


def mode_cj(args):
    p0, T0, X = opt(args, "--p0", 6670.0), opt(args, "--T0", 298.0), opt(args, "--X", "H2:2,O2:1,AR:7")
    c = cj_state(p0, T0, X); z = znd(c, p0, T0, X)
    print(f"mixture {X} at {p0:g} Pa, {T0:g} K: rho_1 = {c['rho1']:.6f} kg/m^3")
    print(f"  Chapman-Jouguet: D = {c['D']:.2f} m/s, p = {c['p']:.1f} Pa ({c['p'] / p0:.3f} p_0), T = {c['T']:.1f} K, rho = {c['rho']:.6f}, u = {c['u']:.2f} m/s")
    print(f"  von Neumann:     p = {c['p_vN']:.1f} Pa ({c['p_vN'] / p0:.3f} p_0), T = {c['T_vN']:.1f} K, rho = {c['rho_vN']:.6f}")
    print(f"  ZND: induction length {1e3 * z['L_ind']:.4f} mm, induction time {1e6 * z['t_ind']:.3f} microseconds; reaction zone followed to x = {1e3 * z['x'][-1]:.2f} mm (M -> 1)")
    return c, z


def mode_table(args):
    """ZND profile as a table for IC::Profile1D: s [m behind the shock], rho, u (laboratory frame), p, Y_1 .. Y_N.
    First row (s < 0) is the unburnt gas; the profile is cut at --length (the last row is used beyond it)."""
    p0, T0, X = opt(args, "--p0", 6670.0), opt(args, "--T0", 298.0), opt(args, "--X", "H2:2,O2:1,AR:7")
    out, L = opt(args, "--out", "znd_table.dat"), opt(args, "--length", 0.3)
    c = cj_state(p0, T0, X); g = ct.Solution(MECH); D = c["D"]; rho1 = c["rho1"]; W = g.molecular_weights

    def rhs(t, y):
        rho, Y = y[0], y[1:-1]; p = p0 + rho1 * D * D * (1 - rho1 / rho); g.DPY = rho, p, np.maximum(Y, 0)
        w = D * rho1 / rho; eta = 1 - (w / g.sound_speed) ** 2; dY = g.net_production_rates * W / rho
        sig = np.sum((g.mean_molecular_weight / W - g.partial_molar_enthalpies / W / (g.cp_mass * g.T)) * dY)
        return np.concatenate(([-rho * sig / eta], dY, [w]))
    far = lambda t, y: y[-1] - L; far.terminal = True
    s = solve_ivp(rhs, (0, 1.0), np.concatenate(([c["rho_vN"]], c["Y1"], [0.0])), method="LSODA", rtol=1e-10, atol=1e-14, events=far, max_step=2e-7, first_step=1e-10)
    x = s.y[-1]; keep = [0]
    for k in range(1, len(x)):                       # thin to steps of at least 2 micrometres
        if x[k] - x[keep[-1]] >= 2.0e-6 or k == len(x) - 1: keep.append(k)
    with open(out, "w") as fo:
        fo.write(f"# ZND profile of the CJ detonation, {X} at {p0:g} Pa, {T0:g} K, Burke-Dryer mechanism (Cantera); D_CJ = {D:.4f} m/s\n")
        fo.write("# s [m behind the shock]  rho [kg/m^3]  u [m/s, laboratory frame]  p [Pa]  " + "  ".join("Y_" + n for n in g.species_names) + "\n")
        fo.write(" ".join(f"{v:.12e}" for v in [-1.0e-7, rho1, 0.0, p0] + list(c["Y1"])) + "\n")
        for k in keep:
            rho = s.y[0, k]; p = p0 + rho1 * D * D * (1 - rho1 / rho)
            fo.write(" ".join(f"{v:.12e}" for v in [x[k], rho, D * (1 - rho1 / rho), p] + list(np.maximum(s.y[1:-1, k], 0))) + "\n")
    g.DPY = s.y[0, -1], p0 + rho1 * D * D * (1 - rho1 / s.y[0, -1]), np.maximum(s.y[1:-1, -1], 0)
    print(f"wrote {out}: {len(keep) + 1} rows to s = {x[-1] * 1e3:.1f} mm; end state T = {g.T:.1f} K (T_CJ {c['T']:.1f}), u = {D * (1 - rho1 / s.y[0, -1]):.1f} m/s (u_CJ {c['u']:.1f}); D_CJ = {D:.2f} m/s")


def files(d):
    return sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))


def mode_ignition(args):
    import yt, matplotlib; yt.funcs.mylog.setLevel(40); matplotlib.use("Agg"); import matplotlib.pyplot as plt
    d = positional(args, ["--outdir"])[0]; outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "H2O2_Ignition")); os.makedirs(outdir, exist_ok=True)
    rows = []
    for pf in files(d):
        ds = yt.load(pf); ad = ds.all_data(); q = lambda nm: float(np.asarray(ad[("boxlib", nm)]).mean())
        rows.append((float(ds.current_time), q("T"), q("pressure"), q("Y_H2"), q("Y_O2"), q("Y_H2O"), q("Y_OH"), q("Y_H"), q("energy_per_vol"), q("density")))
    R = np.array(rows[1:]); t = R[:, 0]
    g = ct.Solution(MECH); g.TDY = R[0, 1], R[0, 9], _Y0(g, R[0]); g.TPX = 1200.0, ct.one_atm, "H2:2,O2:1,AR:7"
    r = ct.IdealGasReactor(g, clone=False); net = ct.ReactorNet([r]); net.rtol = 1e-11; net.atol = 1e-18
    C = []; tt = np.linspace(0, t[-1], 3001)[1:]; F = []
    for x in tt: net.advance(x); F.append((x, g.T, g.P, g["H2"].Y[0], g["O2"].Y[0], g["H2O"].Y[0], g["OH"].Y[0], g["H"].Y[0]))
    F = np.array(F); interp = lambda col: np.interp(t, F[:, 0], F[:, col])
    tig = lambda tx, T: tx[np.argmax(np.gradient(T, tx))]
    print(f"{d}: {len(t)} outputs to t = {1e6 * t[-1]:.1f} microseconds")
    print(f"  ignition delay (steepest temperature rise): {1e6 * tig(t, R[:, 1]):.2f} microseconds, Cantera {1e6 * tig(F[:, 0], F[:, 1]):.2f}  (output interval {1e6 * (t[1] - t[0]):.1f})")
    print(f"  final temperature {R[-1, 1]:.2f} K, Cantera {F[-1, 1]:.2f} ({100 * (R[-1, 1] / F[-1, 1] - 1):+.4f} %);  final pressure {R[-1, 2]:.1f} Pa, Cantera {F[-1, 2]:.1f} ({100 * (R[-1, 2] / F[-1, 2] - 1):+.4f} %)")
    for nm, col in (("temperature", 1), ("pressure", 2), ("Y_H2O", 5), ("Y_OH", 6)):
        ref = interp(col); print(f"  max |{nm} - Cantera| / max(Cantera) over the run: {np.abs(R[:, col] - ref).max() / np.abs(ref).max():.2e}")
    with open(os.path.join(outdir, "h2o2_ignition_summary.csv"), "w") as fo:
        fo.write("t_s,T_K,T_cantera_K,p_Pa,p_cantera_Pa,Y_H2O,Y_H2O_cantera,Y_OH,Y_OH_cantera\n")
        for k in range(len(t)): fo.write(f"{t[k]:.6e},{R[k, 1]:.4f},{interp(1)[k]:.4f},{R[k, 2]:.3f},{interp(2)[k]:.3f},{R[k, 5]:.6e},{interp(5)[k]:.6e},{R[k, 6]:.6e},{interp(6)[k]:.6e}\n")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3)); us = 1e6
    ax[0].plot(us * F[:, 0], F[:, 1], "C3-", lw=1.5, label="Cantera"); ax[0].plot(us * t, R[:, 1], "ko", ms=4, mfc="none", label="Numerical")
    ax[0].set_ylabel("Temperature [K]"); ax[0].set_title("Temperature")
    ax[1].plot(us * F[:, 0], 1e-3 * F[:, 2], "C3-", lw=1.5, label="Cantera"); ax[1].plot(us * t, 1e-3 * R[:, 2], "ko", ms=4, mfc="none", label="Numerical")
    ax[1].set_ylabel("Pressure [kPa]"); ax[1].set_title("Pressure")
    for nm, col, mk in (("H$_2$", 3, "o"), ("O$_2$", 4, "s"), ("H$_2$O", 5, "^"), ("OH", 6, "v"), ("H", 7, "D")):
        ax[2].semilogy(us * F[:, 0], np.maximum(F[:, col], 1e-12), "-", lw=1.3, color="C3"); ax[2].semilogy(us * t, np.maximum(R[:, col], 1e-12), "k" + mk, ms=4, mfc="none", label=nm)
    ax[2].set_ylim(1e-8, 1); ax[2].set_ylabel("Mass Fraction [-]"); ax[2].set_title("Species (Lines: Cantera)"); ax[2].legend(fontsize=8, ncol=2)
    for a in ax[:2]: a.legend()
    for a in ax: a.set_xlabel("Time [$\\mu$s]"); a.grid(alpha=0.3)
    fig.suptitle("Constant-Volume Ignition of 2H$_2$ + O$_2$ + 7Ar at 1200 K, 1 atm (Burke-Dryer Mechanism)"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "h2o2_ignition.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/h2o2_ignition.png")


def _Y0(g, row):
    g.TPX = 1200.0, ct.one_atm, "H2:2,O2:1,AR:7"; return g.Y


def mode_detonation(args):
    import yt, matplotlib; yt.funcs.mylog.setLevel(40); matplotlib.use("Agg"); import matplotlib.pyplot as plt
    flags = ["--outdir", "--p0", "--T0", "--X", "--x0"]
    dirs = positional(args, flags); outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "H2O2_Detonation")); os.makedirs(outdir, exist_ok=True)
    p0, T0, X = opt(args, "--p0", 6670.0), opt(args, "--T0", 298.0), opt(args, "--X", "H2:2,O2:1,AR:7")
    c = cj_state(p0, T0, X); z = znd(c, p0, T0, X)
    print(f"reference: D_CJ = {c['D']:.2f} m/s, p_vN = {c['p_vN']:.0f} Pa, p_CJ = {c['p']:.0f} Pa, T_CJ = {c['T']:.0f} K, induction length {1e3 * z['L_ind']:.3f} mm")
    A = []
    for d in dirs:
        H = []; last = None
        for pf in files(d):
            ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
            n = int(ds.domain_dimensions[0]); L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n
            q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
            p = q("pressure"); t = float(ds.current_time); x0 = opt(args, "--x0", 0.5 * L); m = x > x0
            if t == 0.0: M0 = q("density").sum(); continue
            k = np.where(m & (p > 2.0 * p0))[0]
            if len(k) == 0: continue
            kf = k.max(); xf = x[kf] + (x[1] - x[0]) * (p[kf] - 2.0 * p0) / max(p[kf] - p[min(kf + 1, n - 1)], 1e-30) * 0   # cell-centre front
            H.append((t, x[kf], p[m].max(), q("density").sum() / M0 - 1))
            last = dict(x=x, p=p, T=q("T"), u=q("velocityx"), rho=q("density"), YH2O=q("Y_H2O"), t=t, xf=x[kf], dx=x[1] - x[0])
        H = np.array(H); n2 = len(H) // 2
        D = np.polyfit(H[n2:, 0], H[n2:, 1], 1)[0]; Dl = np.polyfit(H[-max(4, len(H) // 5):, 0], H[-max(4, len(H) // 5):, 1], 1)[0]
        A.append(dict(d=d, H=H, D=D, Dl=Dl, last=last, cells=z["L_ind"] / last["dx"]))
        print(f"{d}: {last['dx'] * 1e3:.4f} mm cells = {A[-1]['cells']:.1f} per induction length, {len(H)} outputs to t = {1e6 * H[-1, 0]:.1f} microseconds")
        print(f"  front speed: second half of the run {D:.1f} m/s ({100 * (D / c['D'] - 1):+.2f} % of CJ), last fifth {Dl:.1f} m/s ({100 * (Dl / c['D'] - 1):+.2f} %)")
        print(f"  peak pressure at the end {last['p'].max():.0f} Pa ({100 * (last['p'].max() / c['p_vN'] - 1):+.1f} % of von Neumann); mean over the second half {H[n2:, 2].mean():.0f} Pa ({100 * (H[n2:, 2].mean() / c['p_vN'] - 1):+.1f} %), min {H[n2:, 2].min():.0f}, max {H[n2:, 2].max():.0f}")
        print(f"  mass change {H[-1, 3]:+.2e}")
    with open(os.path.join(outdir, "h2o2_detonation_summary.csv"), "w") as fo:
        fo.write("run,cells_per_induction_length,front_speed_second_half_m_s,front_speed_last_fifth_m_s,D_CJ_m_s,peak_pressure_end_Pa,p_vN_Pa\n")
        for a in A: fo.write(f"{os.path.basename(os.path.normpath(a['d']))},{a['cells']:.3g},{a['D']:.2f},{a['Dl']:.2f},{c['D']:.2f},{a['last']['p'].max():.1f},{c['p_vN']:.1f}\n")
    a = A[-1]; l = a["last"]; fig, ax = plt.subplots(1, 4, figsize=(19, 4.3)); mm = 1e3
    for b, mk in zip(A, ("o", "s", "^", "v")):
        Hh = b["H"]; tm = 0.5 * (Hh[1:, 0] + Hh[:-1, 0]); sp = np.diff(Hh[:, 1]) / np.diff(Hh[:, 0])
        ax[0].plot(1e6 * tm, sp, "k" + mk, ms=3.5, mfc="none", lw=0.6, ls="-", label=f"Numerical, {b['cells']:.0f} Cells per Induction Length")
    ax[0].axhline(c["D"], color="C3", lw=1.5, label="Chapman-Jouguet (Cantera)"); ax[0].set_xlabel("Time [$\\mu$s]"); ax[0].set_ylabel("Front Speed [m/s]"); ax[0].set_title("Front Speed"); ax[0].legend(fontsize=7)
    ax[0].set_ylim(0.6 * c["D"], 1.6 * c["D"])
    xi = (l["xf"] - l["x"]) * mm; m = (xi > -2 * l["dx"] * mm) & (xi < 12 * z["L_ind"] * mm)
    for b, key, zk, lab, sc in ((ax[1], "p", "p", "Pressure [kPa]", 1e-3), (ax[2], "T", "T", "Temperature [K]", 1.0), (ax[3], "YH2O", "YH2O", "H$_2$O Mass Fraction [-]", 1.0)):
        zm = z["x"] < 12 * z["L_ind"]
        b.plot(z["x"][zm] * mm, sc * z[zk][zm], "C3-", lw=1.5, label="ZND (Cantera)"); b.plot(xi[m], sc * l[key][m], "ko", ms=3.5, mfc="none", label="Numerical")
        b.set_xlabel("Distance Behind the Front [mm]"); b.set_ylabel(lab); b.set_title(lab.split(" [")[0]); b.legend(fontsize=8)
    for b in ax: b.grid(alpha=0.3)
    fig.suptitle(f"Detonation in 2H$_2$ + O$_2$ + 7Ar at {p0 / 1e3:.2f} kPa, {T0:.0f} K (Burke-Dryer Mechanism), t = {1e6 * l['t']:.0f} $\\mu$s"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "h2o2_detonation.png"), dpi=200); plt.close(fig); print(f"  wrote {outdir}/h2o2_detonation.png")


if __name__ == "__main__":
    mode, rest = sys.argv[1], sys.argv[2:]
    dict(ignition=mode_ignition, cj=mode_cj, detonation=mode_detonation, table=mode_table)[mode](rest)
