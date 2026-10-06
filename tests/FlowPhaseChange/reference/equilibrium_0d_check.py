#!/usr/bin/env python3
"""
0D liquid-vapour equilibrium (tests/FlowPhaseChange/input_0D_Equilibrium).

  python3 equilibrium_0d_check.py <run_dir> [alpha_gas rho_gas rho_liq p Yv0]

Independent reference (scipy brentq, written separately from the code's solver): unknowns p, T, m_vap with
    volume   m_l v_l(p,T) + m_g v_g(p,T) = 1,   v = (gamma - 1) cv T/(p + pi)
    energy   m_l e_l + m_g e_g = initial,        e = cv T (p + gamma pi)/(p + pi) + q
    Gibbs    g_l(p,T) = g_v(x_v p, T),           x_v = m_vap/m_g
(m_air = (1 - Yv0) m_g0 fixed; if the Gibbs condition cannot be met the bound m_vap = 0 or
m_liq = 0 applies).  Compared with the last plot file; also checks mass and total-energy conservation.
"""
import sys, os, glob
import numpy as np
from scipy.optimize import brentq
import yt
yt.funcs.mylog.setLevel(40)
L = dict(g=2.35, pi=1.0e9, cv=1816.0, q=-1167.0e3, qp=0.0)
V = dict(g=1.43, pi=0.0, cv=1040.0, q=2030.0e3, qp=-23.4e3)
vol = lambda m, p, T: (m["g"] - 1) * m["cv"] * T / (p + m["pi"])
ene = lambda m, p, T: m["cv"] * T * (p + m["g"] * m["pi"]) / (p + m["pi"]) + m["q"]
gib = lambda m, p, T: (m["g"] * m["cv"] - m["qp"]) * T - m["cv"] * T * (m["g"] * np.log(T) - (m["g"] - 1) * np.log(p + m["pi"])) + m["q"]


def main():
    d = sys.argv[1]
    ag, rg, rl, p0, Y0 = (float(v) for v in sys.argv[2:7]) if len(sys.argv) >= 7 else (0.5, 0.55903, 1019.9, 1.0e5, 1.0)
    ml0, mg0 = (1 - ag) * rl, ag * rg; mair = (1 - Y0) * mg0; rho = ml0 + mg0
    Tl0 = (p0 + L["pi"]) / ((L["g"] - 1) * L["cv"] * rl); Tg0 = p0 / ((V["g"] - 1) * V["cv"] * rg)
    rhoe = ml0 * ene(L, p0, Tl0) + mg0 * ene(V, p0, Tg0)

    # nested scalar root finds (scipy brentq): for a vapour mass x, the p-T state from volume +
    # energy; then the Gibbs condition in x.  Bounds m_vap = 0+ / m_liq = 0+ apply if it has no root.
    def pT(x):
        ml, mg = rho - mair - x, mair + x
        Tof = lambda pp: (rhoe - ml * L["q"] - mg * V["q"]) / (ml * L["cv"] * (pp + L["g"] * L["pi"]) / (pp + L["pi"]) + mg * V["cv"] * (pp + V["g"] * V["pi"]) / (pp + V["pi"]))
        res = lambda pp: ml * vol(L, pp, Tof(pp)) + mg * vol(V, pp, Tof(pp)) - 1.0
        pp = brentq(res, 1e-3 * p0 * 1e-6, 1e4 * p0, xtol=1e-12, rtol=1e-14)
        return pp, Tof(pp)

    def r(x):
        pp, TT = pT(x)
        return gib(L, pp, TT) - gib(V, pp * x / (mair + x), TT)
    xa, xb = 1e-12 * rho, (rho - mair) * (1 - 1e-9)
    # the lowest vapour masses may have no p-T state (the liquid alone cannot fill the box): raise xa
    for _ in range(200):
        try:
            ra = r(xa); break
        except ValueError:
            xa *= 2.0
    if ra <= 0: x = xa
    else:
        # walk up on a log grid to the first sign change (or to where no p-T state exists)
        xs = np.geomspace(xa, xb, 4000); lo = xa; hi = None
        for xx in xs[1:]:
            try: rr = r(xx)
            except ValueError: break
            if rr <= 0: hi = xx; break
            lo = xx
        x = brentq(r, lo, hi, xtol=1e-15 * rho, rtol=1e-14) if hi is not None else lo
    p, T = pT(x)
    ref = dict(p=p, T=T, m_vap=x, m_liq=rho - mair - x, alpha_gas=1 - (rho - mair - x) * vol(L, p, T))
    print(f"{d}: initial T_liq {Tl0:.2f}, T_gas {Tg0:.2f}, p {p0:.4g}, m_liq {ml0:.4f}, m_gas {mg0:.5f} (air {mair:.5f})")
    pfs = sorted(glob.glob(os.path.join(d, "output", "*cell")))
    ds = yt.load(pfs[-1]); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    q = lambda nm: np.asarray(cg[("boxlib", nm)])[:, :, 0]
    got = dict(p=q("pressure").mean(), T=q("pc_T").mean(), m_vap=q("vapor").mean(), m_liq=q("rho_eta1").mean(), alpha_gas=q("eta").mean())
    for k in ref:
        print(f"  {k:10s} {got[k]:14.6f}   reference {ref[k]:14.6f}   ({100 * (got[k] / ref[k] - 1):+.4f} %)")
    ds0 = yt.load(pfs[0]); cg0 = ds0.covering_grid(0, ds0.domain_left_edge, ds0.domain_dimensions)
    q0 = lambda nm: np.asarray(cg0[("boxlib", nm)])[:, :, 0]
    mass = lambda f: (f("rho_eta0") + f("rho_eta1")).mean()
    etot = lambda f: (f("energy_per_vol") + f("rho_eta1") * L["q"] + f("rho_eta0") * V["q"]).mean()
    print(f"  mass drift {mass(q) / rho - 1:+.2e};  total energy (incl. q) drift {etot(q) / rhoe - 1:+.2e};"
          f"  spread of p over the box {q('pressure').max() - q('pressure').min():.2e};  max |u| {np.abs(q('velocityx')).max():.2e}")
    print(f"  latent heat at T: {(V['g'] * V['cv'] - L['g'] * L['cv']) * T + V['q'] - L['q']:.4e} J/kg;  mass evaporated {x - Y0 * mg0:+.5f} kg/m3")


if __name__ == "__main__":
    main()
