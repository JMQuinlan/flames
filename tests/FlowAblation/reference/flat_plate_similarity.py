#!/usr/bin/env python3
"""
Exact compressible laminar flat-plate boundary layer (similarity solution), any viscosity law mu ~ T^omega.

Zero pressure gradient, perfect gas, constant cp and Pr, constant edge state (rho_e, u_e, T_e, M_e).
Illingworth / Levy-Lees variables
    xi  = rho_e u_e mu_e x,      eta = u_e / sqrt(2 xi) * int_0^y rho dy',
    f'(eta) = u/u_e,             theta(eta) = T/T_e = h/h_e,      C = rho mu/(rho_e mu_e) = theta^(omega-1)
(p = const across the layer, so rho/rho_e = 1/theta).  The problem is exactly self-similar for any mu(T):
    (C f'')' + f f'' = 0
    (C/Pr theta')' + f theta' + (gamma-1) M_e^2 C (f'')^2 = 0
    f(0) = f'(0) = 0,  f'(inf) = 1,  theta(inf) = 1,  and theta(0) = T_w/T_e (isothermal) or theta'(0) = 0 (adiabatic).
Wall quantities (exact, no correlation):
    c_f  = tau_w/(rho_e u_e^2/2)       = sqrt(2) C_w f''(0)            / sqrt(Re_x)
    q_w  = k_w dT/dy|_w (into the wall) = C_w theta'(0)/(sqrt(2) Pr) * rho_e u_e cp T_e / sqrt(Re_x)
with Re_x = rho_e u_e x/mu_e.  Solved with scipy solve_bvp, first-order system
    Y = [f, f', tau = C f'', theta, s = (C/Pr) theta'].

  python3 flat_plate_similarity.py         # verification limits + the Mach 2 isothermal flat-plate constants
"""
import numpy as np
from scipy.integrate import solve_bvp


def solve(M=2.0, gam=1.4, Pr=0.72, omega=0.0, Tw=1.0, adiabatic=False, eta_max=14.0, n=4001, tol=1e-10, beta=0.0):
    """Return dict with f''(0), theta(0), theta'(0), C_w, A_cf = c_f sqrt(Re_x), A_q = q_w sqrt(Re_x)/(rho_e u_e cp T_e),
    and the profiles (eta, f', theta).
    beta != 0: compressible Falkner-Skan pressure-gradient term + beta (theta - f'^2) in the momentum equation
    (LOCAL similarity only, used as a sensitivity estimate; the static-enthalpy equation has no beta term because
    rho theta = rho_e across the layer)."""
    D = (gam - 1.0) * M * M
    Cof = lambda th: th ** (omega - 1.0)

    def rhs(eta, Y):
        f, fp, tau, th, s = Y
        C = Cof(th); fpp = tau / C; thp = Pr * s / C
        return np.vstack([fp, fpp, -f * fpp - beta * (th - fp * fp), thp, -f * thp - D * C * fpp * fpp])

    def bc(Ya, Yb):
        wall = Ya[4] if adiabatic else Ya[3] - Tw
        return np.array([Ya[0], Ya[1], Yb[1] - 1.0, Yb[3] - 1.0, wall])

    eta = np.linspace(0.0, eta_max, n)
    fp0 = np.tanh(eta / 1.5)                                   # crude initial guess
    th_guess = (1.0 + 0.4 * D * np.exp(-eta)) if adiabatic else (Tw + (1.0 - Tw) * fp0 + 0.15 * D * fp0 * (1 - fp0))
    Y0 = np.vstack([eta - 1.5 * np.log(np.cosh(eta / 1.5)), fp0, 0.47 * (1 - fp0**2), th_guess, 0.1 * np.exp(-eta)])
    sol = solve_bvp(rhs, bc, eta, Y0, tol=tol, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    f, fp, tau, th, s = sol.sol(0.0)
    Cw = Cof(th)
    return dict(fpp0=tau / Cw, theta_w=th, thetap0=Pr * s / Cw, Cw=Cw,
                A_cf=np.sqrt(2.0) * tau, A_q=s / np.sqrt(2.0),          # tau = C_w f''(0), s = C_w theta'(0)/Pr
                eta=sol.x, fp=sol.y[1], theta=sol.y[3], sol=sol, rms=np.max(sol.rms_residuals))


def flat_plate_constants(M=2.0, gam=1.4, Pr=0.72, omega=0.0, Tw=1.0):
    """Exact c_f sqrt(Re_x), q_w sqrt(Re_x)/(rho u cp T_e), the exact adiabatic-wall temperature T_aw/T_e
    (same M, Pr, mu law) and St sqrt(Re_x) with St = q_w/(rho u cp (T_aw - T_w))."""
    iso = solve(M, gam, Pr, omega, Tw)
    ad = solve(M, gam, Pr, omega, adiabatic=True)
    Taw = ad["theta_w"]
    return dict(A_cf=iso["A_cf"], A_q=iso["A_q"], Taw=Taw, r=(Taw - 1.0) / ((gam - 1) / 2 * M * M),
                A_St=iso["A_q"] / (Taw - Tw), iso=iso, ad=ad)


def eckert_constants(M=2.0, gam=1.4, Pr=0.72, Tw=1.0):
    """Reference-temperature (Eckert 1955) values used by flat_plate_check.py (constant mu: C* = T_e/T*)."""
    Taw = 1 + np.sqrt(Pr) * (gam - 1) / 2 * M * M
    Ts = 1 + 0.5 * (Tw - 1) + 0.22 * (Taw - 1); Cs = 1 / Ts
    A_St = 0.332 * np.sqrt(Cs) * Pr ** (-2 / 3)
    return dict(A_cf=0.664 * np.sqrt(Cs), A_St=A_St, A_q=A_St * (Taw - Tw), Taw=Taw, Cs=Cs)


if __name__ == "__main__":
    print("Verification limits")
    b = solve(M=0.0, omega=1.0, Pr=0.72)
    print(f"  (a) incompressible Blasius (M = 0):            f''(0) = {b['fpp0']:.6f}  [eta = y sqrt(U/(2 nu x)); 0.469600]"
          f"   c_f sqrt(Re_x) = {b['A_cf']:.5f}  [0.66411]")
    for Pr_ in (0.72, 1.0):
        p = solve(M=0.0, omega=1.0, Pr=Pr_, Tw=2.0)
        print(f"      Pohlhausen Pr = {Pr_:4.2f}: Nu_x/sqrt(Re_x) = {-p['thetap0'] / np.sqrt(2):.5f}"
              f"  [0.332 Pr^(1/3) = {0.332 * Pr_ ** (1 / 3):.5f}; exact tables 0.2956 (Pr 0.72), 0.33206 (Pr 1)]")
    for M_ in (2.0, 5.0):
        c = solve(M=M_, omega=1.0, Pr=0.72)
        print(f"  (b) Chapman-Rubesin C = 1, M = {M_:g}, T_w = T_e:  c_f sqrt(Re_x) = {c['A_cf']:.5f}  [0.66411 for any M]")
    a = solve(M=2.0, omega=1.0, Pr=0.72, adiabatic=True)
    print(f"      C = 1 adiabatic wall, Pr = 0.72: recovery factor r = {(a['theta_w'] - 1) / (0.2 * 4):.5f}  [sqrt(Pr) = {np.sqrt(0.72):.5f}]")
    cr = solve(M=2.0, omega=1.0, Pr=1.0, Tw=1.0)
    Taw1 = 1 + 0.2 * 4
    print(f"      C = 1, Pr = 1 (Crocco-Busemann exact, Reynolds analogy St = c_f/2): St sqrt(Re_x) = {cr['A_q'] / (Taw1 - 1):.5f}"
          f"  c_f sqrt(Re_x)/2 = {cr['A_cf'] / 2:.5f}")
    print("\nMach 2, gamma 1.4, Pr 0.72, constant mu (C = T_e/T), T_w = T_e")
    for em, nn in ((10.0, 2001), (14.0, 4001), (20.0, 8001)):
        k = solve(eta_max=em, n=nn)
        print(f"  eta_max = {em:4.0f}: f''(0) = {k['fpp0']:.8f}  theta'(0) = {k['thetap0']:.8f}  A_cf = {k['A_cf']:.8f}  A_q = {k['A_q']:.8f}  (max rms residual {k['rms']:.1e})")
    K = flat_plate_constants(); E = eckert_constants()
    print(f"  exact:  c_f sqrt(Re_x) = {K['A_cf']:.5f}   q_w sqrt(Re_x)/(rho u cp T_e) = {K['A_q']:.5f}   T_aw/T_e = {K['Taw']:.5f}"
          f" (r = {K['r']:.5f})   St sqrt(Re_x) = {K['A_St']:.5f}")
    print(f"  Eckert: c_f sqrt(Re_x) = {E['A_cf']:.5f}   q_w sqrt(Re_x)/(rho u cp T_e) = {E['A_q']:.5f}   T_aw/T_e = {E['Taw']:.5f}"
          f" (r = {np.sqrt(0.72):.5f})   St sqrt(Re_x) = {E['A_St']:.5f}   C* = {E['Cs']:.5f}")
    print(f"  Eckert relative to exact: c_f {100 * (E['A_cf'] / K['A_cf'] - 1):+.2f} %,  q_w {100 * (E['A_q'] / K['A_q'] - 1):+.2f} %,"
          f"  St {100 * (E['A_St'] / K['A_St'] - 1):+.2f} %")
