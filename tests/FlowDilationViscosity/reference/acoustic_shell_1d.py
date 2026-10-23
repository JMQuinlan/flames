#!/usr/bin/env python3
"""
SHARP-INTERFACE REFERENCE for the dilatational-viscosity unit tests:
linear acoustics in D-dimensional radial symmetry (D = 2 cylinder, D = 3 sphere)
with a massless viscous MEMBRANE (Boussinesq-Scriven dilatational viscosity kappa_s)
at r = R.

    p_t = -rho c^2 r^(1-D) d/dr ( r^(D-1) u )
    u_t = -(1/rho) dp/dr

Jump at the membrane (normal-stress balance, T = kappa_s div_s u, div_s u = (D-1) u/R):

    p_in - p_out = (D-1) T / R = kappa_s (D-1)^2 u_R / R^2        (sphere: 4 kappa_s u/R^2)

rho, c may differ inside / outside (multiphase) or be equal (single phase, where the
membrane is the ONLY thing at r = R).  Staggered grid (p at centres, u at faces), the
membrane on a face; its stiff dashpot term is integrated exactly (exponential), the rest
with leapfrog-staggered symplectic Euler at CFL 0.5.  Outer boundary: first-order
characteristic (non-reflecting) condition p = rho c u.

Library:
    ref = solve(D, R, kappa_s, rho_in, c_in, rho_out, c_out, p_init(r), r_max, t_probe,
                r_probe, dr=...)     -> dict(t, p[probe], u_R)
CLI (quick look):
    acoustic_shell_1d.py --D 2 --R 1 --kappa 2 --rho 0.991 --c 3.73 --A 0.01 --w 0.4 --tend 2
"""
import argparse
import numpy as np


def solve(D, R, kappa_s, rho_in, c_in, rho_out, c_out, p_init, r_max, t_end, r_probe,
          dr=None, n_out=2001, cfl=0.5):
    if dr is None:
        dr = R / 400.0
    nR = int(round(R / dr))
    dr = R / nR                                  # membrane exactly on face nR
    N = int(np.ceil(r_max / dr))
    rf = np.arange(N + 1) * dr                   # faces
    rc = 0.5 * (rf[1:] + rf[:-1])                # centres
    rho = np.where(rc < R, rho_in, rho_out)
    c = np.where(rc < R, c_in, c_out)
    K = rho * c * c                              # bulk modulus per cell
    rho_f = np.empty(N + 1)
    rho_f[1:-1] = 0.5 * (rho[1:] + rho[:-1]); rho_f[0] = rho[0]; rho_f[-1] = rho[-1]
    m = D - 1
    vol = (rf[1:] ** D - rf[:-1] ** D) / D       # r^(D-1) dr integrated
    area = rf ** m
    p = p_init(rc).astype(float)
    u = np.zeros(N + 1)
    dt = cfl * dr / max(c_in, c_out)
    nsteps = int(np.ceil(t_end / dt)); dt = t_end / nsteps
    # membrane: u_R face momentum   rho_f dr u_t = (p_i - p_{i+1}) - G u,   G = kappa (D-1)^2 / R^2
    G = kappa_s * m * m / (R * R)
    a = G / (rho_f[nR] * dr)                     # damping rate of the membrane face
    ip = [int(np.clip(np.searchsorted(rc, r) - 1, 0, N - 2)) for r in r_probe]
    wp = [(r - rc[i]) / dr for r, i in zip(r_probe, ip)]
    t_out = np.linspace(0.0, t_end, n_out)
    P = np.zeros((n_out, len(r_probe))); UR = np.zeros(n_out)
    k_out = 0
    t = 0.0
    for n in range(nsteps + 1):
        while k_out < n_out and t_out[k_out] <= t + 0.5 * dt:
            P[k_out] = [p[i] * (1 - w) + p[i + 1] * w for i, w in zip(ip, wp)]
            UR[k_out] = u[nR]
            k_out += 1
        if n == nsteps:
            break
        # velocity update (interior faces)
        dp = (p[:-1] - p[1:]) / dr               # faces 1..N-1
        u[1:-1] += dt * dp / rho_f[1:-1]
        # membrane face: undo the explicit update, apply exact exponential with forcing
        f = (p[nR - 1] - p[nR]) / (rho_f[nR] * dr)
        u_old = u[nR] - dt * f
        if a > 0:
            e = np.exp(-a * dt)
            u[nR] = u_old * e + f / a * (1 - e)
        u[0] = 0.0                               # symmetry
        # outer face: outgoing characteristic u = p'/(rho c)
        u[-1] = p[-1] / (rho[-1] * c[-1])          # p is the perturbation about p_inf
        # pressure update
        p -= dt * K * (area[1:] * u[1:] - area[:-1] * u[:-1]) / vol
        t += dt
    return dict(t=t_out, p=P, u_R=UR, dr=dr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--R", type=float, default=1.0)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.991)
    ap.add_argument("--c", type=float, default=3.73)
    ap.add_argument("--rho_out", type=float, default=None)
    ap.add_argument("--c_out", type=float, default=None)
    ap.add_argument("--A", type=float, default=0.01)
    ap.add_argument("--w", type=float, default=0.4)
    ap.add_argument("--tend", type=float, default=2.0)
    ap.add_argument("--rmax", type=float, default=20.0)
    ap.add_argument("--probes", type=float, nargs="+", default=[0.5, 2.0])
    a = ap.parse_args()
    ro = a.rho_out or a.rho; co = a.c_out or a.c
    for kap in (0.0, a.kappa):
        r = solve(a.D, a.R, kap, a.rho, a.c, ro, co, lambda x: a.A * np.exp(-(x / a.w) ** 2),
                  a.rmax, a.tend, a.probes)
        i = np.argmax(np.abs(r["p"][:, -1]))
        print("kappa %.3g: probe r=%.2f peak %.5f at t=%.3f;  probe r=%.2f min %.5f"
              % (kap, a.probes[-1], r["p"][i, -1], r["t"][i], a.probes[0], r["p"][:, 0].min()))


if __name__ == "__main__":
    main()
