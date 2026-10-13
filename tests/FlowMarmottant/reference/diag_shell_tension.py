#!/usr/bin/env python3
"""
Shell-tension and interface force-balance diagnostic for the coated Sch20 decks.

Answers two questions from the plotfiles alone:

  1. Is there a RESIDUAL surface tension acting on the bubble?  The sharp
     normal-stress balance is
         p_gas - p_l(R) = 2 sigma/R + 4 mu Rdot/R + 4 kappa_s Rdot/R^2
     so  sigma_res = R/2 (p_gas - p_l(R) - 4 mu Rdot/R - 4 kappa_s Rdot/R^2).
     For a buckled shell (sigma = 0) this should be ~0 at every frame.  p_l(R)
     is the liquid pressure extrapolated to the interface with the
     incompressible potential-flow form  p = A + B (R0/r) + C (R0/r)^4,
     fitted outside the band.

  2. What tension does the SOLVER actually apply?  Every term of the shell
     stress is recomputed from primary fields exactly as Hydro2::RHS builds it
     (the `kappa` plot fields are overwritten by the curvature pass and cannot
     be used):
         elastic     sigma(Gamma) |grad eta|                  (shell field)
         viscous     kappa_s |grad eta| div_s u
         area-rate   kappa_s n.grad(S),  S = K div u (Kapila)  (shell.kappa_s_area_rate)
     Each is reported as an area-mean tension  int(tau)/int(|grad eta|)  and as
     the equivalent normal pressure jump  int 2 tau / r dr  it puts on a sphere.
     A nonzero CYCLE-AVERAGE of the viscous parts is a rectified (spurious
     static) tension; its sign says whether it inflates or shrinks the bubble.

Usage (any run of the coated Sch20 decks; octant or full domain):
    python3 diag_shell_tension.py PLOTFILE_DIR [--input ../input_Sch20-Oscillating_Marmottant_Light]
                                  [--nproc 8] [--csv out.csv] [--kappa-s 7.2e-9] [--area-rate 1]

Output: one line per frame, then averages; --csv writes every column.
Memory: one covering grid of (box/dx_finest)^3 cells per worker, box = --box R0
(default 3; the pressure fit uses r up to the box edge).  At R0/dx 25.6 that is
~(77)^3 = 0.5 M cells -- cheap.
"""
import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from marmottant_sch20_common import parse_input, _num, _expr_const  # noqa: E402

P = {}   # case parameters, filled in main() and handed to each worker by _init


def _init(params):
    P.update(params)


def frame(pf):
    import yt
    yt.funcs.mylog.setLevel(50)
    ds = yt.load(pf)
    R0, box = P["R0"], P["box"]
    L = ds.index.max_level
    dle = ds.domain_left_edge.d
    dre = ds.domain_right_edge.d
    dx = (dre[0] - dle[0]) / (ds.domain_dimensions[0] * 2 ** L)
    octant = [abs(dle[d]) < 1e-12 for d in range(3)]
    sym = 2 ** sum(octant)
    cen = np.array([0.0 if octant[d] else 0.5 * (dle[d] + dre[d]) for d in range(3)])
    lo = np.array([max(dle[d], cen[d] - box * R0) for d in range(3)])
    hi = np.array([min(dre[d], cen[d] + box * R0) for d in range(3)])
    n = np.round((hi - lo) / dx).astype(int)
    hi = lo + n * dx
    cg = ds.covering_grid(L, lo, n)

    eta = np.asarray(cg["boxlib", "eta"], float)
    G = np.asarray(cg["boxlib", "shell"], float)
    p = np.asarray(cg["boxlib", "pressure"], float)
    u = [np.asarray(cg["boxlib", "velocity" + c], float) for c in "xyz"]

    x = [lo[d] + (np.arange(n[d]) + 0.5) * dx - cen[d] for d in range(3)]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)
    dV = dx ** 3 * sym

    # ---- shell stress, exactly as Hydro2::RHS --------------------------- #
    ge = np.array(np.gradient(eta, dx))
    g = np.sqrt((ge ** 2).sum(0))
    on = g > 1e-10
    nh = np.where(on, ge / np.where(on, g, 1.0), 0.0)
    J = np.array([np.gradient(ui, dx) for ui in u])          # J[a][b] = d u_a / d x_b
    div = J[0][0] + J[1][1] + J[2][2]
    nJn = sum(nh[a] * J[a][b] * nh[b] for a in range(3) for b in range(3))
    divs = div - nJn

    Gb, chi, sbrk, sigw = P["Gb"], P["chi"], P["sbrk"], P["sigw"]
    if P["marm"]:
        el = np.where((G > 0) & (G < Gb), chi * (Gb / np.where(G > 0, G, 1.0) - 1.0), 0.0)
        sig_el = np.where(el >= sbrk, sigw, el)
    else:
        sig_el = np.full_like(eta, P["sig0"])
    sig_el = np.where(on, sig_el, 0.0)

    kap = P["kappa_s"]
    a0 = np.clip(eta, 0, 1); a1 = 1 - a0
    Z0 = P["g0"] * np.maximum(p + P["pi0"], 1e-8)
    Z1 = P["g1"] * np.maximum(p + P["pi1"], 1e-8)
    den = a0 * Z1 + a1 * Z0
    K = np.where(den > 0, a0 * a1 * (Z1 - Z0) / np.where(den > 0, den, 1.0), 0.0)
    S = K * div
    nS = (nh * np.array(np.gradient(S, dx))).sum(0) if P["area_rate"] else np.zeros_like(eta)

    tau_el = sig_el * g
    tau_vk = kap * g * divs
    tau_vS = kap * nS
    near = (r < box * R0 * 0.95) & on
    A = np.sum(g[near]) * dV
    rr = np.maximum(r, 0.5 * dx)
    w = 2.0 / (4 * np.pi * rr ** 3)                         # int 2 tau / r dr per unit volume
    out = dict(t=float(ds.current_time), name=os.path.basename(pf), dx=dx)
    for k, tau in (("el", tau_el), ("vk", tau_vk), ("vS", tau_vS)):
        out["sig_" + k] = np.sum(tau[near]) * dV / A
        out["dp_" + k] = np.sum((tau * w)[near]) * dV
    out["area"] = A

    # ---- radii ----------------------------------------------------------- #
    out["R_vol"] = (3 * np.sum(np.clip(1 - eta, 0, 1)) * dV / (4 * np.pi)) ** (1 / 3)
    nb = int(box * R0 / dx)
    edges = np.linspace(0, box * R0, nb + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1)
    inbox = r.ravel() < box * R0
    cnt = np.bincount(idx[inbox], minlength=nb)
    def prof(f):
        s = np.bincount(idx[inbox], weights=f.ravel()[inbox], minlength=nb)
        return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    pe, pp = prof(eta), prof(p)
    R05 = np.nan
    for i in range(nb - 1):
        if np.isfinite(pe[i]) and np.isfinite(pe[i + 1]) and (pe[i] - 0.5) * (pe[i + 1] - 0.5) < 0:
            R05 = rc[i] + (0.5 - pe[i]) * (rc[i + 1] - rc[i]) / (pe[i + 1] - pe[i]); break
    out["R05"] = R05

    # ---- pressures ------------------------------------------------------- #
    core = (eta < 0.02) & (r < 0.8 * R05)
    out["p_gas"] = float(p[core].mean()) if core.any() else np.nan
    eps = P["eps"]
    m = np.isfinite(pp) & (rc > R05 + 3 * eps) & (rc < 0.95 * box * R0)
    xx = R0 / rc[m]
    M = np.vstack([np.ones(m.sum()), xx, xx ** 4]).T
    co, *_ = np.linalg.lstsq(M, pp[m], rcond=None)
    xr = R0 / R05
    out["p_l"] = float(co @ np.array([1, xr, xr ** 4]))
    out["p_inf_fit"] = float(co[0])

    # ---- gas thermodynamics: entropy proxy s = p / rho_g^gamma ------------ #
    # rho_g = (alpha rho)_gas / alpha_gas with alpha_gas = 1 - eta (phase 1).
    # The analysis cache shows the core p tracking rho^~1.436 instead of
    # rho^1.4 (0.9% over the adiabat at R_min, reversible).  Splitting core and
    # band gas says whether that is core entropy exchange or a measurement mix.
    ag = np.clip(1 - eta, 0, 1)
    arg = np.asarray(cg["boxlib", "rho_eta1"], float)
    rg = np.where(ag > 1e-3, arg / np.maximum(ag, 1e-3), np.nan)
    s = p / rg ** P["g1"]
    band = (ag > 0.1) & (ag < 0.9)
    out["rho_core"] = float(np.nanmean(rg[core])) if core.any() else np.nan
    out["s_core"] = float(np.nanmean(s[core])) if core.any() else np.nan
    out["s_core_spread"] = float(np.nanstd(s[core]) / np.nanmean(s[core])) if core.any() else np.nan
    wb = ag[band]
    out["s_band"] = float(np.nansum(s[band] * wb) / np.sum(wb)) if band.any() else np.nan
    out["gas_band_frac"] = float(np.sum(ag[band]) / np.sum(ag))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotdir")
    ap.add_argument("--input", default=os.path.normpath(os.path.join(
        _HERE, "..", "input_Sch20-Oscillating_Marmottant_Light")))
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--box", type=float, default=3.0, help="half-width of the analysed box [R0]")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--kappa-s", type=float, default=None, help="override shell.kappa_s")
    ap.add_argument("--mu", type=float, default=None, help="override mu0")
    ap.add_argument("--area-rate", type=int, default=None,
                    help="override shell.kappa_s_area_rate (solver default 1)")
    ap.add_argument("--tmin-frac", type=float, default=0.3,
                    help="averages use t > tmin-frac * tau_c (skips the elastic start)")
    a = ap.parse_args()

    kv = parse_input(a.input)
    R0 = _expr_const(kv, "eta.ic", "R0") or _num(kv, "marmottant.R0")
    mR0 = _num(kv, "marmottant.R0", R0)
    Rb = _num(kv, "marmottant.R_buckling", 0.0)
    P.update(
        R0=R0, box=a.box, eps=_num(kv, "epsilon", 0.0),
        marm=int(_num(kv, "marmottant", 0)),
        Gb=(mR0 / Rb) ** 2 if Rb else 1.0,
        chi=_num(kv, "marmottant.chi", 0.0),
        sbrk=_num(kv, "marmottant.sigma_break", 1e30),
        sigw=_num(kv, "sigma", 0.0), sig0=_num(kv, "sigma", 0.0),
        kappa_s=a.kappa_s if a.kappa_s is not None else _num(kv, "shell.kappa_s", 0.0),
        mu=a.mu if a.mu is not None else _num(kv, "mu0", 0.0),
        area_rate=a.area_rate if a.area_rate is not None else int(_num(kv, "shell.kappa_s_area_rate", 1)),
        g0=_num(kv, "eos0.gamma", 2.35), pi0=_num(kv, "eos0.p0", 0.0),
        g1=_num(kv, "eos1.gamma", 1.4), pi1=_num(kv, "eos1.p0", 0.0),
        rho_l=1000.0, p_inf=_expr_const(kv, "pressure0.ic", "p_inf", 1e5),
    )
    tau_c = 0.915 * R0 * np.sqrt(P["rho_l"] / P["p_inf"])
    pfs = sorted(d for d in glob.glob(os.path.join(a.plotdir, "*cell")) if os.path.isdir(d))
    if not pfs:
        raise SystemExit("no *cell plotfiles in %s" % a.plotdir)
    print("%d plotfiles; kappa_s %.3g  mu %.3g  area_rate %d  marmottant %d  Gamma_buck %.4f  box %.1f R0"
          % (len(pfs), P["kappa_s"], P["mu"], P["area_rate"], P["marm"], P["Gb"], a.box))
    with Pool(a.nproc, initializer=_init, initargs=(dict(P),)) as pool:
        rows = pool.map(frame, pfs)
    rows.sort(key=lambda d: d["t"])
    if len({round(d["dx"], 15) for d in rows}) > 1 or len({d["t"] for d in rows}) < len(rows):
        print("WARNING: frames differ in dx or repeat a time -- this directory mixes runs.")

    t = np.array([d["t"] for d in rows]); R = np.array([d["R05"] for d in rows])
    Rd = np.gradient(R, t) if len(t) > 2 else np.zeros_like(t)
    mu, kap = P["mu"], P["kappa_s"]
    print("\n  t/tau_c  R05/R0  Rdot   p_gas   p_l(R)  | sigma_res(sharp)  | applied tension [N/m]: elastic  visc(div_s)  area-rate  total"
          "  | jump from applied [Pa]  sigma_res(vs applied)")
    for d, Rdi, Ri in zip(rows, Rd, R):
        dp = d["p_gas"] - d["p_l"]
        visc_l = 4 * mu * Rdi / Ri
        d["Rdot"] = Rdi
        d["sigma_res_sharp"] = 0.5 * Ri * (dp - visc_l - 4 * kap * Rdi / Ri ** 2)
        d["dp_applied"] = d["dp_el"] + d["dp_vk"] + d["dp_vS"]
        d["sigma_res_applied"] = 0.5 * Ri * (dp - visc_l - d["dp_applied"])
        d["sig_tot"] = d["sig_el"] + d["sig_vk"] + d["sig_vS"]
        print("  %6.3f  %.5f %+6.3f %7.0f %7.0f | %+9.5f | %+9.5f %+9.5f %+9.5f %+9.5f | %+8.0f  %+9.5f"
              % (d["t"] / tau_c, Ri / R0, Rdi, d["p_gas"], d["p_l"], d["sigma_res_sharp"],
                 d["sig_el"], d["sig_vk"], d["sig_vS"], d["sig_tot"], d["dp_applied"], d["sigma_res_applied"]))

    m = t > a.tmin_frac * tau_c
    if m.sum() > 2:
        av = lambda k: np.mean([d[k] for d, mm in zip(rows, m) if mm])
        print("\nAVERAGES over t > %.2f tau_c (%d frames; a true cycle average needs whole periods):"
              % (a.tmin_frac, m.sum()))
        for k, lab in (("sig_el", "elastic tension"), ("sig_vk", "viscous kappa_s div_s u"),
                       ("sig_vS", "area-rate kappa_s n.grad S"), ("sig_tot", "TOTAL applied"),
                       ("sigma_res_sharp", "residual vs sharp balance"),
                       ("sigma_res_applied", "residual vs applied stress")):
            print("  %-28s %+9.5f N/m" % (lab, av(k)))
        print("  (sigma_water = %.4f N/m; a static sigma shifts R_eq by ~ 2 sigma/(3 gamma p_inf R0) = %+.4f R0 per 0.001 N/m)"
              % (P["sigw"], -2 * 0.001 / (3 * P["g1"] * P["p_inf"] * R0)))
        print("  p_inf from the liquid fit: mean %.0f Pa (deck %.0f)" % (av("p_inf_fit"), P["p_inf"]))

    # Gas thermodynamics.  s = p/rho_g^gamma is constant for an adiabatic gas;
    # gamma_eff is the slope of ln p_core vs ln rho_core (1.4 if adiabatic).
    s0c, s0b = rows[0]["s_core"], rows[0]["s_band"]
    print("\nGAS ENTROPY PROXY s = p/rho_g^gamma (normalised by frame 0)")
    print("  t/tau_c  rho_core  s_core/s0  core spread  s_band/s0_band  band gas fraction")
    for d in rows:
        print("  %6.3f  %8.5f  %9.5f  %10.2e  %13.5f  %10.4f"
              % (d["t"] / tau_c, d["rho_core"], d["s_core"] / s0c, d["s_core_spread"],
                 d["s_band"] / s0b, d["gas_band_frac"]))
    ok = np.array([np.isfinite(d["rho_core"]) and np.isfinite(d["p_gas"]) for d in rows])
    if ok.sum() > 3:
        lr = np.log([d["rho_core"] for d, k in zip(rows, ok) if k])
        lp = np.log([d["p_gas"] for d, k in zip(rows, ok) if k])
        print("  gamma_eff (fit ln p_core vs ln rho_core) = %.4f   (EOS gamma %.3f)"
              % (np.polyfit(lr, lp, 1)[0], P["g1"]))
    if a.csv:
        keys = list(rows[0].keys())
        with open(a.csv, "w") as fh:
            fh.write(",".join(keys) + "\n")
            for d in rows:
                fh.write(",".join(str(d[k]) for k in keys) + "\n")
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
