#!/usr/bin/env python3
"""
Reference solution for the ablation-onset test (tests/FlowAblation/input_1D_AblationOnset).

A solid slab 0 < x < S(t) (S(0) = L, adiabatic back at x = 0, rho_s, cp_s, k_s, initially T_s0) is in
contact with a gas column S(t) < x < X1 (adiabatic end at x = X1, rho cp, k_g, initially T_g > T_abl).
Both phases at rest, constant properties, pure conduction; the gas fills the space the solid leaves
(no blowing, as in the two-phase Stefan test).  The surface temperature follows the capped rule of
src/Integrator/Ablation.H:
    stage 1 (T_w < T_abl): no recession, flux continuity  k_g dT/dx|gas = k_s dT/dx|solid;
    stage 2 (T_w = T_abl): rho_s Q* v = q_g - q_s,  v = -dS/dt,  q_g = k_g dT/dx|gas,  q_s = k_s dT/dx|solid.

Stage 1, semi-infinite gas (X1 -> inf), closed form by Laplace transform (images in the slab):
    T_w(t) = T_s0 + (T_g - T_s0)/(1 + sig) [1 + (1 + beta) sum_{n>=1} beta^(n-1) erfc(n L / sqrt(alpha_s t))],
    sig = e_s/e_g,  beta = (sig - 1)/(sig + 1),  e = sqrt(k rho c);
the onset time t_abl solves T_w(t_abl) = T_abl.

Stage 2 has no closed form: it is computed here by a front-fixing (Landau) transform of each phase
onto a fixed grid (xi = x/S in the solid, zeta = (x - S)/(X1 - S) in the gas), second-order central
differences, second-order one-sided gradients at the surface, and an implicit (BDF) integration of
the method-of-lines system with the stage switch as an event.  The run starts at a small t0 from the
exact contact solution (erf profiles), so the initial discontinuity is not resolved numerically.

Self-checks (python3 ablation_onset_reference.py verify):
  (a) stage 1 against the closed-form T_w(t) and onset time;
  (b) stage 2 alone (T_s0 chosen so that the surface ablates from t = 0, solid long) against the
      two-phase Stefan similarity solution of input_1D_Stefan2 (lam = 0.0872);
  (c) grid convergence of the onset time and of the recession s(t_end).

Usage as a module:  solve(par, t_out, Ns, Ng) -> dict(t, s, Tw, qg, qs, t_abl, profiles)
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf, erfc
from scipy.optimize import brentq
from scipy.sparse import lil_matrix

# parameters of input_1D_AblationOnset (gas of input_1D_Stefan/Contact: mu 1e-3, cp 3.5, Pr 0.72)
DECK = dict(rho_g=1.0, cp_g=3.5, k_g=1.0e-3 * 3.5 / 0.72, T_g=1.0,
            rho_s=1.0, cp_s=0.875, k_s=1.9444444e-2, T_s0=0.99,
            T_abl=0.9975, Qstar=0.01, L=0.1, X1=0.5)


def closed_form_Tw(par, t, nterms=200):
    """stage-1 surface temperature, semi-infinite gas (valid while T_w < T_abl)"""
    eg = np.sqrt(par["k_g"] * par["rho_g"] * par["cp_g"]); es = np.sqrt(par["k_s"] * par["rho_s"] * par["cp_s"])
    a_s = par["k_s"] / (par["rho_s"] * par["cp_s"]); sig = es / eg; beta = (sig - 1) / (sig + 1)
    t = np.atleast_1d(np.asarray(t, float)); S = np.zeros_like(t)
    for n in range(1, nterms + 1):
        S += beta ** (n - 1) * erfc(n * par["L"] / np.sqrt(a_s * np.maximum(t, 1e-300)))
    return par["T_s0"] + (par["T_g"] - par["T_s0"]) / (1 + sig) * (1 + (1 + beta) * S)


def closed_form_onset(par):
    Ti = closed_form_Tw(par, 1e-12)[0]
    if par["T_abl"] <= Ti: return 0.0
    if par["T_abl"] >= par["T_g"]: return np.inf
    return brentq(lambda t: closed_form_Tw(par, t)[0] - par["T_abl"], 1e-10, 1e6, xtol=1e-14, rtol=1e-13)


def solve(par, t_out, Ns=400, Ng=2000, t0=None, rtol=1e-10, atol=1e-14, init=None):
    """init(t0) -> (S0, T_solid(x), T_gas(x)): optional exact state at t0 (default: contact erf profiles about L)"""
    rg, cg, kg, Tg = par["rho_g"], par["cp_g"], par["k_g"], par["T_g"]
    rs, cs, ks, Ts0 = par["rho_s"], par["cp_s"], par["k_s"], par["T_s0"]
    Ta, Q, L, X1 = par["T_abl"], par["Qstar"], par["L"], par["X1"]
    ag, as_ = kg / (rg * cg), ks / (rs * cs)
    eg, es = np.sqrt(kg * rg * cg), np.sqrt(ks * rs * cs)
    Ti = (eg * Tg + es * Ts0) / (eg + es)
    xi = np.linspace(0.0, 1.0, Ns + 1)        # solid nodes, xi = 1 is the surface
    ze = np.linspace(0.0, 1.0, Ng + 1)        # gas nodes, ze = 0 is the surface
    dxi, dze = xi[1] - xi[0], ze[1] - ze[0]
    t_out = np.asarray(t_out, float)
    if t0 is None:   # start from the exact contact solution while the back wall is not felt (erfc(7) ~ 1e-23);
                     # the grid must resolve the erf layer then (an unresolved step start converges erratically)
        t0 = 0.02 * L * L / as_          # fixed (independent of t_out); output times before t0 are dropped
    # unknowns: solid nodes 0..Ns-1, gas nodes 1..Ng, S  (the surface node is algebraic)
    nS, nG = Ns, Ng
    iS = nS + nG

    def Tw_cont(T_s, T_g, S):
        ds, dg = dxi * S, dze * (X1 - S)
        # k_g (-3Tw + 4T1 - T2)/(2dg) = k_s (3Tw - 4T_{-1} + T_{-2})/(2ds)
        A = 3 * kg / (2 * dg) + 3 * ks / (2 * ds)
        B = kg * (4 * T_g[0] - T_g[1]) / (2 * dg) + ks * (4 * T_s[-1] - T_s[-2]) / (2 * ds)
        return B / A

    def fluxes(T_s, T_g, S, Tw):
        ds, dg = dxi * S, dze * (X1 - S)
        qg = kg * (-3 * Tw + 4 * T_g[0] - T_g[1]) / (2 * dg)      # heat flux gas -> surface (> 0 heats)
        qs = ks * (3 * Tw - 4 * T_s[-1] + T_s[-2]) / (2 * ds)     # heat flux surface -> solid
        return qg, qs

    def rhs_factory(stage):
        def rhs(t, y):
            T_s = y[:nS]; T_g = y[nS:nS + nG]; S = y[iS]
            Tw = Ta if stage == 2 else Tw_cont(T_s, T_g, S)
            qg, qs = fluxes(T_s, T_g, S, Tw)
            Sdot = -(qg - qs) / (rs * Q) if stage == 2 else 0.0
            dy = np.empty_like(y)
            # solid: T_t = as/S^2 T_xixi + xi Sdot/S T_xi ; adiabatic at xi = 0 (ghost T_{-1} = T_1)
            Ts_full = np.concatenate([T_s, [Tw]])
            lap = np.empty(nS); grd = np.empty(nS)
            lap[0] = 2 * (Ts_full[1] - Ts_full[0]) / dxi**2; grd[0] = 0.0
            lap[1:] = (Ts_full[2:] - 2 * Ts_full[1:-1] + Ts_full[:-2]) / dxi**2
            grd[1:] = (Ts_full[2:] - Ts_full[:-2]) / (2 * dxi)
            dy[:nS] = as_ / S**2 * lap + xi[:nS] * Sdot / S * grd
            # gas: T_t = ag/(X1-S)^2 T_zz + Sdot (1-ze)/(X1-S) T_z ; adiabatic at ze = 1
            H = X1 - S
            Tg_full = np.concatenate([[Tw], T_g])
            lap = np.empty(nG); grd = np.empty(nG)
            lap[:-1] = (Tg_full[2:] - 2 * Tg_full[1:-1] + Tg_full[:-2]) / dze**2
            grd[:-1] = (Tg_full[2:] - Tg_full[:-2]) / (2 * dze)
            lap[-1] = 2 * (Tg_full[-2] - Tg_full[-1]) / dze**2; grd[-1] = 0.0
            dy[nS:nS + nG] = ag / H**2 * lap + Sdot * (1 - ze[1:]) / H * grd
            dy[iS] = Sdot
            return dy
        return rhs

    # sparsity: tridiagonal + surface coupling (2 nodes each side) + S row/column
    n = iS + 1
    sp = lil_matrix((n, n))
    for i in range(n - 1):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n - 1: sp[i, j] = 1
    near = [nS - 2, nS - 1, nS, nS + 1]
    for i in near:
        for j in near: sp[i, j] = 1
    sp[:, iS] = 1; sp[iS, :] = 1
    sp = sp.tocsr()

    # initial state at t0: exact contact (erf) profiles about x = L
    if init is None:
        S0 = L
        fs = lambda x: Ti + (Ts0 - Ti) * erf((L - x) / (2 * np.sqrt(as_ * t0)))
        fg = lambda x: Ti + (Tg - Ti) * erf((x - L) / (2 * np.sqrt(ag * t0)))
    else:
        S0, fs, fg = init(t0)
    x_s = xi[:nS] * S0; x_g = S0 + ze[1:] * (X1 - S0)
    y0 = np.concatenate([fs(x_s), fg(x_g), [S0]])
    stage = 2 if Ti >= Ta else 1

    def ev_onset(t, y):
        return Tw_cont(y[:nS], y[nS:nS + nG], y[iS]) - Ta
    ev_onset.terminal = True; ev_onset.direction = 1

    tt = t_out[t_out >= t0]
    sols = []; t_abl = t0 if stage == 2 else None; tc, yc = t0, y0
    while True:
        ev = [ev_onset] if stage == 1 else None
        tev = tt[tt > tc]
        r = solve_ivp(rhs_factory(stage), (tc, t_out.max()), yc, method="BDF", t_eval=tev if len(tev) else None,
                      events=ev, jac_sparsity=sp, rtol=rtol, atol=atol, first_step=min(1e-8, t0))
        if not r.success: raise RuntimeError(r.message)
        sols.append((stage, r.t, r.y))
        if stage == 1 and r.t_events and len(r.t_events[0]):
            t_abl = r.t_events[0][0]; tc = t_abl; yc = r.y_events[0][0].copy(); stage = 2; continue
        break
    T_list, S_list, Tw_list, qg_list, qs_list, st_list, Ys = [], [], [], [], [], [], []
    for stg, ts, ys in sols:
        for k in range(len(ts)):
            y = ys[:, k]; T_s = y[:nS]; T_g = y[nS:nS + nG]; S = y[iS]
            Tw = Ta if stg == 2 else Tw_cont(T_s, T_g, S)
            qg, qs = fluxes(T_s, T_g, S, Tw)
            T_list.append(ts[k]); S_list.append(S); Tw_list.append(Tw); qg_list.append(qg); qs_list.append(qs); st_list.append(stg)
            Ys.append(y)
    out = dict(t=np.array(T_list), s=L - np.array(S_list), Tw=np.array(Tw_list), qg=np.array(qg_list),
               qs=np.array(qs_list), stage=np.array(st_list), t_abl=t_abl, Ti=Ti)

    def profile(k):
        y = Ys[k]; S = y[iS]; Tw = out["Tw"][k]
        x = np.concatenate([xi * S, S + ze[1:] * (X1 - S)])
        T = np.concatenate([y[:nS], [Tw], y[nS:nS + nG]])
        return x, T
    out["profile"] = profile
    return out


def verify():
    par = dict(DECK); ok = True
    # (a) stage 1 against the closed form (gas long enough to be semi-infinite to round-off over the window)
    pa = dict(par, X1=par["L"] + 0.6); ta = closed_form_onset(pa)
    t_out = np.linspace(0.1, 0.98 * ta, 30)
    for Ns, Ng in ((200, 1200), (400, 2400), (800, 4800)):
        r = solve(dict(pa, T_abl=10.0), t_out, Ns=Ns, Ng=Ng)
        e = np.abs(r["Tw"] - closed_form_Tw(pa, r["t"])) / (pa["T_g"] - pa["T_s0"])
        r2 = solve(pa, [ta * 1.01], Ns=Ns, Ng=Ng)
        print(f"(a) stage 1, Ns = {Ns}: max |T_w - closed form| = {100 * e.max():.5f} % of dT over {t_out[0]:g} <= t <= {t_out[-1]:.3f};"
              f"  onset {r2['t_abl']:.6f} vs closed form {ta:.6f} ({100 * (r2['t_abl'] / ta - 1):+.5f} %)")
    # (b) stage 2 alone against the two-phase Stefan similarity solution (input_1D_Stefan2 parameters)
    pb = dict(rho_g=1.0, cp_g=3.5, k_g=par["k_g"], T_g=1.0, rho_s=1.0, cp_s=3.5, k_s=4.8611111e-3, T_s0=0.985,
              T_abl=0.99, Qstar=0.08, L=1.0, X1=4.0)
    ag = pb["k_g"] / (pb["rho_g"] * pb["cp_g"]); as_ = pb["k_s"] / (pb["rho_s"] * pb["cp_s"]); rr = ag / as_
    g = lambda l: (pb["k_g"] * (pb["T_g"] - pb["T_abl"]) * np.exp(-l * l) / ((1 + erf(l)) * np.sqrt(np.pi * ag))
                   - pb["k_s"] * (pb["T_abl"] - pb["T_s0"]) * np.exp(-l * l * rr) / (erfc(l * np.sqrt(rr)) * np.sqrt(np.pi * as_))
                   - pb["rho_s"] * pb["Qstar"] * l * np.sqrt(ag))
    lam = brentq(g, 1e-9, 5)
    x0 = pb["L"]; t0 = 0.05
    init = lambda t0: (x0 - 2 * lam * np.sqrt(ag * t0),
                       lambda x: pb["T_s0"] + (pb["T_abl"] - pb["T_s0"]) * erfc((x0 - x) / (2 * np.sqrt(as_ * t0))) / erfc(lam * np.sqrt(rr)),
                       lambda x: pb["T_abl"] + (pb["T_g"] - pb["T_abl"]) * (erf((x - x0) / (2 * np.sqrt(ag * t0))) + erf(lam)) / (1 + erf(lam)))
    for N in (1000, 2000, 4000):
        r = solve(pb, [1.0, 5.0], Ns=N, Ng=3 * N, t0=t0, init=init)
        sx = 2 * lam * np.sqrt(ag * r["t"])
        print(f"(b) two-phase Stefan (lam = {lam:.5f}), Ns = {N}: s/s_exact - 1 = "
              + ", ".join(f"{100 * (a / b - 1):+.4f} % (t = {t:g})" for a, b, t in zip(r["s"], sx, r["t"])))
    # (c) grid convergence on the deck problem
    tend = 6.0
    for Ns, Ng in ((200, 800), (400, 1600), (800, 3200), (1600, 6400)):
        r = solve(par, [tend], Ns=Ns, Ng=Ng)
        print(f"(c) deck, Ns = {Ns}, Ng = {Ng}: t_abl = {r['t_abl']:.6f}  s(t = {tend:g}) = {r['s'][-1]:.7f}  T_w closed-form onset {closed_form_onset(par):.6f}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify()
    else:
        par = dict(DECK)
        print("closed-form onset", closed_form_onset(par))
        r = solve(par, np.linspace(0.25, 6, 24), Ns=400, Ng=1600)
        for k in range(len(r["t"])):
            print(f"t {r['t'][k]:6.3f}  stage {r['stage'][k]}  Tw {r['Tw'][k]:.6f}  s {r['s'][k]:.6f}  qg {r['qg'][k]:.4e}  qs {r['qs'][k]:.4e}")
        print("t_abl", r["t_abl"])
