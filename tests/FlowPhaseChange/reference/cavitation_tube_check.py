#!/usr/bin/env python3
"""
1D cavitation tube (tests/FlowPhaseChange/input_1D_CavitationTube).

  python3 cavitation_tube_check.py <run_dir> [<run_dir without phase change>] [--outdir DIR]

Checks of the phase-change run (Saurel, Petitpas & Abgrall 2008; Pelanti & Shyue 2014, Sec. 5.2):
  * in the cavitated zone the mixture is at saturation: p = p_sat(T) with T the cell temperature
    (independent scipy evaluation of g_liq(p, T) = g_vap(p, T))
  * the pressure there stays near p_sat(T_0) ~ 0.5 bar, while without mass transfer it falls far lower
  * the vapour volume fraction grows at the centre; total mass is conserved; total energy
    including the formation energies q is conserved (the tube ends are not reached)
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
L = dict(g=2.35, pi=1.0e9, cv=1816.0, q=-1167.0e3, qp=0.0)
V = dict(g=1.43, pi=0.0, cv=1040.0, q=2030.0e3, qp=-23.4e3)
gib = lambda m, p, T: (m["g"] * m["cv"] - m["qp"]) * T - m["cv"] * T * (m["g"] * np.log(T) - (m["g"] - 1) * np.log(p + m["pi"])) + m["q"]
psat = lambda T: brentq(lambda p: gib(L, p, T) - gib(V, p, T), 1.0, 5.0e7)


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    n = ds.domain_dimensions[0]; x = float(ds.domain_left_edge[0]) + (np.arange(n) + 0.5) * float(ds.domain_width[0]) / n
    names = [f[1] for f in ds.field_list]
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, 0, 0] if nm in names else None
    return float(ds.current_time), x, q


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d = args[0]; d0 = args[1] if len(args) > 1 and "--outdir" not in sys.argv[2:3] else None
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    os.makedirs(outdir, exist_ok=True)
    pfs = sorted(glob.glob(os.path.join(d, "output", "*cell")))
    t, x, q = load(pfs[-1]); t0, _, q0 = load(pfs[0])
    p, T, ag, u = q("pressure"), q("pc_T"), q("eta"), q("velocityx")
    cav = np.abs(x) < 0.05
    ps = np.array([psat(Ti) for Ti in T[cav]])
    etot = lambda f: (f("energy_per_vol") + f("rho_eta1") * L["q"] + f("rho_eta0") * V["q"]).sum()
    mass = lambda f: (f("rho_eta0") + f("rho_eta1")).sum()
    print(f"{d}: t = {1e3 * t:.3f} ms")
    print(f"  centre (|x| < 0.05): p = {p[cav].mean():.1f} Pa, T = {T[cav].mean():.3f} K, vapour volume fraction {ag[cav].mean():.4f} (initial 0.01)")
    print(f"  saturation check there: max |p / p_sat(T) - 1| = {np.abs(p[cav] / ps - 1).max():.2e}   (p_sat(354.73 K) = {psat(354.728):.1f} Pa)")
    print(f"  minimum pressure in the tube {p.min():.1f} Pa; velocities at the ends {u[2]:+.3f}, {u[-3]:+.3f}")
    # the tube is open: fluid leaves both ends at |u| = 2 m/s, so mass and energy fall by 2 |u| t / L
    out = -2.0 * abs(u[2]) * t / (x[-1] - x[0] + (x[1] - x[0]))
    print(f"  total mass change {mass(q) / mass(q0) - 1:+.4e}, total energy (incl. q) change {etot(q) / etot(q0) - 1:+.4e};  outflow through the ends {out:+.4e}")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    ax[0, 0].plot(x, 1e-5 * p, "k-", label="with phase change"); ax[0, 1].plot(x, ag, "k-"); ax[1, 0].plot(x, u, "k-"); ax[1, 1].plot(x, T, "k-")
    ax[0, 0].axhline(1e-5 * psat(354.728), color="C3", ls="--", lw=0.9, label=r"$p_{sat}(T_0)$")
    if d0:
        _, x0, qn = load(sorted(glob.glob(os.path.join(d0, "output", "*cell")))[-1])
        print(f"  without phase change: centre p = {qn('pressure')[np.abs(x0) < 0.05].mean():.1f} Pa, vapour volume fraction {qn('eta')[np.abs(x0) < 0.05].mean():.4f}")
        ax[0, 0].plot(x0, 1e-5 * qn("pressure"), "C0:", label="no mass transfer"); ax[0, 1].plot(x0, qn("eta"), "C0:"); ax[1, 0].plot(x0, qn("velocityx"), "C0:")
    for a, yl in zip(ax.ravel(), ("pressure [bar]", "vapour volume fraction", "velocity [m/s]", "temperature [K]")): a.set_ylabel(yl); a.grid(alpha=0.3)
    ax[0, 1].set_yscale("log"); ax[0, 0].legend(fontsize=8)
    for a in ax[1]: a.set_xlabel("x [m]")
    fig.suptitle(f"Cavitation tube, t = {1e3 * t:.2f} ms"); fig.tight_layout(); fig.savefig(os.path.join(outdir, "cavitation_tube.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/cavitation_tube.png")


if __name__ == "__main__":
    main()
