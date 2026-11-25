#!/usr/bin/env python3
"""
Closed box: a plate and a gas come to thermal equilibrium (tests/FlowConjugateHeat/input_1D_ClosedBoxEquilibrium).

  python3 closed_box_check.py <run_dir> [<run_dir> ...] [--outdir DIR]

Periodic, so the total energy E = int_gas rho E dx + rho_s cp_s int_solid T_s dx and the gas mass are constant and
    T_eq = (m_g cv T_g0 + C_s T_s0)/(m_g cv + C_s),   p_eq = p0 T_eq/T_g0     (gas at constant volume).
Further run directories (same case, smaller time step) show how the energy error scales with dt.
"""
import os, sys, glob, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(40)
CV = 2.5


def meta(d, key):
    val = None
    for ln in open(os.path.join(d, "output", "metadata")):
        m = re.match(r"\s*" + re.escape(key) + r"\s*=\s*(\S+)", ln)
        if m: val = float(m.group(1))
    return val


def analyse(d):
    rc = meta(d, "ablation.rho_s") * meta(d, "ablation.cp_s"); Ts0 = meta(d, "ablation.T_solid0"); H = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
        ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
        n = ds.domain_dimensions[0]; dx = float(ds.domain_right_edge[0]) / n
        q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0]
        fl = q("phi") >= 0.5; T, Ts = q("T")[fl], q("ablation_Ts")[~fl]
        H.append((float(ds.current_time), q("energy_per_vol")[fl].sum() * dx, rc * Ts.sum() * dx, q("density")[fl].sum() * dx,
                  T.mean(), Ts.mean(), max(T.max(), Ts.max()) - min(T.min(), Ts.min()), q("pressure")[fl].mean()))
    H = np.array(H); Cs = rc * (~fl).sum() * dx; mg = H[0, 3]
    Teq = (mg * CV * H[0, 4] + Cs * Ts0) / (mg * CV + Cs)
    return dict(d=d, H=H, Teq=Teq, peq=H[0, 7] * Teq / H[0, 4], Ts0=Ts0, Tg0=H[0, 4], dt=meta(d, "cfl"), exch=abs(H[-1, 2] - H[0, 2]))


def main():
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = args[args.index("--outdir") + 1] if "--outdir" in args else None
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--outdir")]
    A = [analyse(d) for d in dirs]
    if outdir is None: outdir = os.path.join(here, "Images", os.path.basename(os.path.normpath(dirs[0])))
    os.makedirs(outdir, exist_ok=True)
    for a in A:
        H = a["H"]; E = H[:, 1] + H[:, 2]; dT = a["Tg0"] - a["Ts0"]
        print(f"{a['d']}  (cfl {a['dt']:g}, to t = {H[-1, 0]:.1f})")
        print(f"  equilibrium temperature: gas {H[-1, 4]:.6f}, solid {H[-1, 5]:.6f}, exact {a['Teq']:.6f}   ({100 * (H[-1, 4] - a['Teq']) / dT:+.3f} % of the initial difference);"
              f"  spread left {H[-1, 6]:.1e}")
        print(f"  pressure {H[-1, 7]:.6f}, exact {a['peq']:.6f}  ({100 * (H[-1, 7] / a['peq'] - 1):+.4f} %);   gas mass change {H[-1, 3] / H[0, 3] - 1:+.1e}")
        print(f"  total energy change {E[-1] / E[0] - 1:+.2e} of the total = {100 * (E[-1] - E[0]) / a['exch']:+.3f} % of the heat exchanged ({a['exch']:.4e})")
    if len(A) > 1:
        e = [abs((a["H"][-1, 1] + a["H"][-1, 2]) - (a["H"][0, 1] + a["H"][0, 2])) for a in A]
        for a, b, ea, eb in zip(A[:-1], A[1:], e[:-1], e[1:]):
            print(f"  energy error, cfl {a['dt']:g} -> {b['dt']:g}: ratio {ea / eb:.2f}  (first order in dt = {a['dt'] / b['dt']:.2f})")
    a = A[0]; H = a["H"]; t = H[:, 0]; E = H[:, 1] + H[:, 2]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].axhline(a["Teq"], color="C3", lw=1.6, label="Exact Equilibrium")
    ax[0].plot(t, H[:, 4], "ko", ms=3.5, mfc="none", label="Numerical, Gas Mean"); ax[0].plot(t, H[:, 5], "ks", ms=3.5, mfc="none", label="Numerical, Solid Mean")
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("Mean Temperature [K]"); ax[0].set_title("Approach to Thermal Equilibrium"); ax[0].legend()
    ax[1].plot(t, (H[:, 1] - H[0, 1]), "k--", lw=1.2, label="Gas"); ax[1].plot(t, (H[:, 2] - H[0, 2]), "k:", lw=1.6, label="Solid")
    ax[1].plot(t, E - E[0], "k-", lw=1.6, label="Total")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel(r"Energy Change [J/m$^2$]"); ax[1].set_title("Energy Exchanged Between Gas and Solid"); ax[1].legend()
    for c in ax: c.grid(alpha=0.3)
    fig.suptitle("Closed Box: A Cold Plate and a Warm Gas Reach Equilibrium"); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "closed_box.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/closed_box.png")


if __name__ == "__main__":
    main()
