#!/usr/bin/env python3
"""
Non-similar compressible laminar boundary layer with a prescribed edge pressure p_e(x) (sensitivity estimate for
the displacement-induced pressure of the Mach 2 flat plate, tests/FlowAblation/input_2D_FlatPlate).

Levy-Lees variables xi = int rho_e u_e mu dx, eta = u_e/sqrt(2 xi) int rho dy, F = u/u_e, theta = T/T_e(x),
C = rho mu/(rho_e mu_e) = 1/theta (constant mu).  With s = ln xi (2 xi d/dxi = 2 d/ds):
    (C F')' + f F' + beta (theta - F^2)            = 2 (F F_s - F' f_s)
    (C/Pr theta')' + f theta' + (gamma-1) M_e^2 C F'^2 = 2 (F theta_s - theta' f_s)
beta = (2 xi/u_e) du_e/dxi.  The static-enthalpy equation has no pressure-gradient term: u dp_e/dx cancels the
dh_e/dx term exactly because rho theta = rho_e and h_e + u_e^2/2 = const.  The edge state follows p_e(x) isentropically
from the free stream (rho = p = T = 1, U).  Marched in s with BDF2 (backward Euler first step), each station an ODE
BVP in eta solved with scipy solve_bvp; started from the local-similarity profile at x0.

  python3 flat_plate_nonsimilar.py     # self-tests: p_e = const reproduces the similarity solution; u_e ~ x^m reproduces Falkner-Skan
"""
import numpy as np
from scipy.integrate import solve_bvp
from flat_plate_similarity import solve as similar


def march(pe, dpe, xs_out, x0=0.02, gam=1.4, cp=3.5, mu=2e-4, Pr=0.72, U=2.3664319132398464, Tw=1.0, nsteps=200,
          ue_override=None):
    """pe(x), dpe(x): edge pressure and its derivative (free stream p = rho = T = 1, R = cp - cv = cp (gam-1)/gam).
    Returns arrays (x, tau_w, q_w) at xs_out (q_w = heat flux INTO the wall)."""
    Rg = cp * (gam - 1) / gam
    def edge(x):
        p = pe(x); T = p ** ((gam - 1) / gam); rho = p / (Rg * T)
        u = np.sqrt(U * U - 2 * cp * (T - 1.0))
        dT = (gam - 1) / gam * T / p * dpe(x); du = -cp * dT / u
        if ue_override is not None:
            u, du = ue_override(x)
        return p, T, rho, u, du
    # xi(x) on a fine grid
    xg = np.linspace(0.0, 1.0 + 1e-9, 200001)[1:]; xg = np.concatenate([[0.0], xg])
    E = [edge(max(x, 0.5 * x0)) for x in xg]          # edge state frozen below x0/2 (only enters xi near the leading edge)
    rue = np.array([e[2] * e[3] for e in E]) * mu
    xi_g = np.concatenate([[0.0], np.cumsum(0.5 * (rue[1:] + rue[:-1]) * np.diff(xg))])
    x_of_xi = lambda xi: np.interp(xi, xi_g, xg)
    s0, s1 = np.log(np.interp(x0, xg, xi_g)), np.log(np.interp(max(xs_out), xg, xi_g))
    S = np.linspace(s0, s1, nsteps + 1)
    out_x, out_tau, out_q = [], [], []
    hist = []      # previous solutions (sol callables)
    for n, s in enumerate(S):
        xi = np.exp(s); x = x_of_xi(xi)
        p, Te, rhoe, ue, due = edge(x)
        Me = ue / np.sqrt(gam * p / rhoe); D = (gam - 1) * Me * Me
        beta = 2 * xi * due / (rhoe * ue * ue * mu)
        thw = Tw / Te
        if n == 0:
            o = similar(M=Me, gam=gam, Pr=Pr, omega=0.0, Tw=thw, beta=beta, eta_max=14.0, n=2001)
            sol = o["sol"].sol
        else:
            ds = S[n] - S[n - 1]
            if n == 1:
                c0, cs = 1.0 / ds, [(-1.0 / ds, hist[-1])]
            else:
                c0, cs = 1.5 / ds, [(-2.0 / ds, hist[-1]), (0.5 / ds, hist[-2])]
            def rhs(eta, Y, D=D, beta=beta, c0=c0, cs=cs):
                f, F, tau, th, sv = Y
                C = 1.0 / th; Fp = tau / C; thp = Pr * sv / C
                f_s, F_s, th_s = c0 * f, c0 * F, c0 * th
                for c, h in cs:
                    Yh = h(eta); f_s = f_s + c * Yh[0]; F_s = F_s + c * Yh[1]; th_s = th_s + c * Yh[3]
                return np.vstack([F, Fp, -f * Fp - beta * (th - F * F) + 2 * (F * F_s - Fp * f_s), thp,
                                  -f * thp - D * C * Fp * Fp + 2 * (F * th_s - thp * f_s)])
            def bc(Ya, Yb, thw=thw):
                return np.array([Ya[0], Ya[1], Yb[1] - 1.0, Yb[3] - 1.0, Ya[3] - thw])
            prev = hist[-1]
            r = solve_bvp(rhs, bc, prev_x, prev(prev_x), tol=1e-8, max_nodes=100000)
            if not r.success:
                raise RuntimeError(f"x = {x}: {r.message}")
            sol = r.sol
        prev_x = np.linspace(0, 14.0, 2001)
        hist.append(sol); hist = hist[-2:]
        Y0 = sol(0.0); rhow = p / (Rg * Tw)
        tau_w = mu * rhow * ue * ue * Y0[2] * (thw) / np.sqrt(2 * xi)   # Y0[2] = C_w F'(0), C_w = 1/thw  -> F'(0) = Y0[2]*thw
        q_w = (mu * cp / Pr) * Te * (Pr * Y0[4] * thw) * ue * rhow / np.sqrt(2 * xi)
        out_x.append(x); out_tau.append(tau_w); out_q.append(q_w)
    out_x, out_tau, out_q = map(np.array, (out_x, out_tau, out_q))
    return xs_out, np.interp(xs_out, out_x, out_tau), np.interp(xs_out, out_x, out_q)


if __name__ == "__main__":
    from flat_plate_similarity import flat_plate_constants
    U, mu, cp = 2.3664319132398464, 2e-4, 3.5
    K = flat_plate_constants(); xs = np.array([0.1, 0.4, 0.8])
    Re = U * xs / mu
    x, tau, q = march(lambda x: 1.0 + 0 * x, lambda x: 0 * x, xs, nsteps=60)
    print("p_e = const:  c_f/exact - 1 =", tau / (0.5 * U * U) / (K["A_cf"] / np.sqrt(Re)) - 1,
          "  q_w/exact - 1 =", q / (K["A_q"] * U * cp / np.sqrt(Re)) - 1)
    # Falkner-Skan check, nearly incompressible: tiny Mach via override u_e = U x^m (only beta matters at M -> 0)
    m = 0.05; beta = 2 * m / (m + 1)
    o = similar(M=0.0, omega=0.0, beta=beta)
    def uo(x): return 0.01 * x ** m, 0.01 * m * x ** (m - 1)
    x, tau, q = march(lambda x: 1.0 + 0 * x, lambda x: 0 * x, xs, U=1e-2, nsteps=80, ue_override=uo)
    # similarity: tau = mu rho u_e^2 f''(0)/sqrt(2 xi), xi = rho mu U' x^{m+1}/(m+1)
    ue = 0.01 * xs ** m; xi = mu * 0.01 * xs ** (m + 1) / (m + 1)
    print(f"Falkner-Skan m = {m}: tau/similarity - 1 =", tau / (mu * ue * ue * o["fpp0"] / np.sqrt(2 * xi)) - 1)
