"""
Low-diffusion single-fluid reference for the Rayleigh-Taylor unit test.

Replaces the solver embedded in rayleightaylor_UNIT_TEST.py /
Raleigh_Taylor_Instability_STANDALONE.py (a MUSCL-Hancock scheme with a
Rusanov / local Lax-Friedrichs flux and *no* slope limiter, adapted from
P. Mocz's "finitevolume" tutorial).  That scheme is not a VoF method: the
interface is just the density contact of a single-gas Euler solver, and the
Rusanov flux smears a contact over many cells while the unlimited slopes
create the 0.80 / 2.18 over/undershoots seen in the old figures.

This solver keeps the same physics (2D compressible Euler, single ideal gas
gamma = 1.4, gravity g = -0.1 as momentum/energy source, x periodic, slip
walls at y = 0 and y = 1.5) but uses
  * HLLC flux (Toro, Spruce & Speares 1994) -- resolves contacts exactly,
  * MUSCL reconstruction of primitive variables with the monotonized-central
    (van Leer 1977) limiter,
  * third-order SSP Runge-Kutta (Shu & Osher 1988), CFL 0.4,
  * optional THINC reconstruction (Xiao, Honma & Kono 2005; face formulas
    as in Deng et al. 2018, THINC-BVD) of density and colour function in
    monotone interface cells, beta = 1.6-2.0, which keeps the contact at
    about 2 cells (compressive, VoF-like interface capturing),
  * a passive colour function phi (rho*phi conserved, phi = 1 heavy) used
    only for the interface diagnostics,
and is run on grids up to 8x the deck resolution (numba-compiled).

Initial condition identical to the Hydro2 deck:
  rho = 1 + 0.5 (1 + tanh((y - 0.75)/0.004)),   u = 0,
  v   = w0 (1 - cos 4 pi x)(1 - cos 4 pi y/3),   w0 = 2.5e-3,
  p   = 2.5 - 0.1 (y - 0.75) rho_k   (rho_k = 2 above, 1 below).

Usage: python3 rt_reference_hllc.py N OUTFILE.npz [t_end] [dt_out] [w0] [thinc_beta]
"""
import sys, time
import numpy as np
from numba import njit, prange

GAM = 1.4
G = -0.1
LX, LY = 0.5, 1.5
NG = 2


@njit(cache=True)
def mc(a, b):
    if a * b <= 0.0:
        return 0.0
    s = 1.0 if a > 0 else -1.0
    return s * min(2 * abs(a), 2 * abs(b), 0.5 * abs(a + b))


@njit(cache=True)
def hllc(rL, uL, vL, pL, fL, rR, uR, vR, pR, fR, out):
    # normal velocity u, tangential v
    cL = np.sqrt(GAM * pL / rL); cR = np.sqrt(GAM * pR / rR)
    EL = pL / (GAM - 1) + 0.5 * rL * (uL * uL + vL * vL)
    ER = pR / (GAM - 1) + 0.5 * rR * (uR * uR + vR * vR)
    SL = min(uL - cL, uR - cR); SR = max(uL + cL, uR + cR)
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))
    if SL >= 0:
        out[0] = rL * uL; out[1] = rL * uL * uL + pL; out[2] = rL * uL * vL
        out[3] = (EL + pL) * uL; out[4] = rL * uL * fL
    elif SR <= 0:
        out[0] = rR * uR; out[1] = rR * uR * uR + pR; out[2] = rR * uR * vR
        out[3] = (ER + pR) * uR; out[4] = rR * uR * fR
    elif Ss >= 0:
        k = rL * (SL - uL) / (SL - Ss)
        FL0 = rL * uL; FL1 = rL * uL * uL + pL; FL2 = rL * uL * vL; FL3 = (EL + pL) * uL; FL4 = rL * uL * fL
        out[0] = FL0 + SL * (k - rL)
        out[1] = FL1 + SL * (k * Ss - rL * uL)
        out[2] = FL2 + SL * (k * vL - rL * vL)
        out[3] = FL3 + SL * (k * (EL / rL + (Ss - uL) * (Ss + pL / (rL * (SL - uL)))) - EL)
        out[4] = FL4 + SL * (k * fL - rL * fL)
    else:
        k = rR * (SR - uR) / (SR - Ss)
        FR0 = rR * uR; FR1 = rR * uR * uR + pR; FR2 = rR * uR * vR; FR3 = (ER + pR) * uR; FR4 = rR * uR * fR
        out[0] = FR0 + SR * (k - rR)
        out[1] = FR1 + SR * (k * Ss - rR * uR)
        out[2] = FR2 + SR * (k * vR - rR * vR)
        out[3] = FR3 + SR * (k * (ER / rR + (Ss - uR) * (Ss + pR / (rR * (SR - uR)))) - ER)
        out[4] = FR4 + SR * (k * fR - rR * fR)


@njit(cache=True)
def prim(U, W):
    nx, ny = U.shape[1], U.shape[2]
    for i in range(nx):
        for j in range(ny):
            r = U[0, i, j]
            if r <= 0.0:
                continue
            u = U[1, i, j] / r; v = U[2, i, j] / r
            W[0, i, j] = r; W[1, i, j] = u; W[2, i, j] = v
            W[3, i, j] = (GAM - 1) * (U[3, i, j] - 0.5 * r * (u * u + v * v))
            W[4, i, j] = U[4, i, j] / r


@njit(cache=True)
def ghosts(W):
    nx, ny = W.shape[1] - 2 * NG, W.shape[2] - 2 * NG
    for g in range(NG):
        for j in range(W.shape[2]):
            for q in range(5):
                W[q, g, j] = W[q, nx + g, j]
                W[q, nx + NG + g, j] = W[q, NG + g, j]
    for g in range(NG):
        for i in range(W.shape[1]):
            for q in range(5):
                W[q, i, NG - 1 - g] = W[q, i, NG + g]
                W[q, i, ny + NG + g] = W[q, i, ny + NG - 1 - g]
            W[2, i, NG - 1 - g] = -W[2, i, NG + g]
            W[2, i, ny + NG + g] = -W[2, i, ny + NG - 1 - g]


@njit(cache=True)
def thinc_faces(qm, q0, qp, beta):
    """THINC face values of cell 0 (Xiao, Honma & Kono 2005; form of Deng et
    al. 2018): returns (value at left face, value at right face), or a flag
    (-1e30) when the cell is not a monotone interface cell."""
    qmin = min(qm, qp); dq = abs(qp - qm)
    if dq < 1e-8 or (qp - q0) * (q0 - qm) <= 0.0:
        return -1e30, -1e30
    C = (q0 - qmin + 1e-20) / (dq + 1e-20)
    if C < 1e-4 or C > 1.0 - 1e-4:
        return -1e30, -1e30
    th = 1.0 if qp > qm else -1.0
    tb = np.tanh(beta)
    B = np.exp(th * beta * (2.0 * C - 1.0))
    A = (B / np.cosh(beta) - 1.0) / tb
    left = qmin + 0.5 * dq * (1.0 + th * A)
    right = qmin + 0.5 * dq * (1.0 + th * (tb + A) / (1.0 + A * tb))
    return left, right


@njit(cache=True, parallel=False)
def rhs(U, W, dx, R, beta):
    prim(U, W); ghosts(W)
    nx, ny = U.shape[1] - 2 * NG, U.shape[2] - 2 * NG
    R[:] = 0.0
    F = np.empty(5); sL = np.empty(5); sR = np.empty(5)
    # x faces: between cells i-1 and i, i = NG..nx+NG
    for j in range(NG, ny + NG):
        for i in range(NG, nx + NG + 1):
            for q in range(5):
                aL = W[q, i - 1, j] - W[q, i - 2, j]; bL = W[q, i, j] - W[q, i - 1, j]
                aR = W[q, i, j] - W[q, i - 1, j]; bR = W[q, i + 1, j] - W[q, i, j]
                sL[q] = W[q, i - 1, j] + 0.5 * mc(aL, bL)
                sR[q] = W[q, i, j] - 0.5 * mc(aR, bR)
            if beta > 0.0:
                for q in (0, 4):   # density and colour function: THINC in interface cells
                    l0, r0 = thinc_faces(W[q, i - 2, j], W[q, i - 1, j], W[q, i, j], beta)
                    if r0 > -1e29 and 1e-3 < W[4, i - 1, j] < 1.0 - 1e-3:
                        sL[q] = r0
                    l1, r1 = thinc_faces(W[q, i - 1, j], W[q, i, j], W[q, i + 1, j], beta)
                    if l1 > -1e29 and 1e-3 < W[4, i, j] < 1.0 - 1e-3:
                        sR[q] = l1
            hllc(sL[0], sL[1], sL[2], sL[3], sL[4], sR[0], sR[1], sR[2], sR[3], sR[4], F)
            for q in range(5):
                if i > NG:
                    R[q, i - 1, j] -= F[q] / dx
                if i < nx + NG:
                    R[q, i, j] += F[q] / dx
    # y faces
    for i in range(NG, nx + NG):
        for j in range(NG, ny + NG + 1):
            for q in range(5):
                aL = W[q, i, j - 1] - W[q, i, j - 2]; bL = W[q, i, j] - W[q, i, j - 1]
                aR = W[q, i, j] - W[q, i, j - 1]; bR = W[q, i, j + 1] - W[q, i, j]
                sL[q] = W[q, i, j - 1] + 0.5 * mc(aL, bL)
                sR[q] = W[q, i, j] - 0.5 * mc(aR, bR)
            if beta > 0.0:
                for q in (0, 4):
                    l0, r0 = thinc_faces(W[q, i, j - 2], W[q, i, j - 1], W[q, i, j], beta)
                    if r0 > -1e29 and 1e-3 < W[4, i, j - 1] < 1.0 - 1e-3:
                        sL[q] = r0
                    l1, r1 = thinc_faces(W[q, i, j - 1], W[q, i, j], W[q, i, j + 1], beta)
                    if l1 > -1e29 and 1e-3 < W[4, i, j] < 1.0 - 1e-3:
                        sR[q] = l1
            # rotate: normal = v
            hllc(sL[0], sL[2], sL[1], sL[3], sL[4], sR[0], sR[2], sR[1], sR[3], sR[4], F)
            Fy0 = F[0]; Fy1 = F[2]; Fy2 = F[1]; Fy3 = F[3]; Fy4 = F[4]
            if j > NG:
                R[0, i, j - 1] -= Fy0 / dx; R[1, i, j - 1] -= Fy1 / dx; R[2, i, j - 1] -= Fy2 / dx
                R[3, i, j - 1] -= Fy3 / dx; R[4, i, j - 1] -= Fy4 / dx
            if j < ny + NG:
                R[0, i, j] += Fy0 / dx; R[1, i, j] += Fy1 / dx; R[2, i, j] += Fy2 / dx
                R[3, i, j] += Fy3 / dx; R[4, i, j] += Fy4 / dx
    # gravity
    for i in range(NG, nx + NG):
        for j in range(NG, ny + NG):
            R[2, i, j] += U[0, i, j] * G
            R[3, i, j] += U[2, i, j] * G


@njit(cache=True)
def maxspeed(U, W):
    prim(U, W)
    s = 0.0
    nx, ny = U.shape[1] - 2 * NG, U.shape[2] - 2 * NG
    for i in range(NG, nx + NG):
        for j in range(NG, ny + NG):
            c = np.sqrt(GAM * W[3, i, j] / W[0, i, j])
            s = max(s, c + max(abs(W[1, i, j]), abs(W[2, i, j])))
    return s


def run(N, out, t_end=20.0, dt_out=0.5, cfl=0.4, w0=2.5e-3, beta=0.0):
    nx, ny = N, 3 * N
    dx = LX / nx
    x = (np.arange(nx) + 0.5) * dx; y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y, indexing="ij")
    rho = 1.0 + 0.5 * (1 + np.tanh((Y - 0.75) / 0.004))
    phi = 0.5 * (1 + np.tanh((Y - 0.75) / 0.004))
    rk = np.where(Y > 0.75, 2.0, 1.0)
    p = 2.5 + G * (Y - 0.75) * rk
    u = np.zeros_like(X)
    v = w0 * (1 - np.cos(4 * np.pi * X)) * (1 - np.cos(4 * np.pi * Y / 3))
    U = np.zeros((5, nx + 2 * NG, ny + 2 * NG))
    s = (slice(NG, nx + NG), slice(NG, ny + NG))
    U[0][s] = rho; U[1][s] = rho * u; U[2][s] = rho * v
    U[3][s] = p / (GAM - 1) + 0.5 * rho * (u * u + v * v); U[4][s] = rho * phi
    W = np.zeros_like(U); R = np.zeros_like(U); U1 = np.zeros_like(U); U2 = np.zeros_like(U)
    t = 0.0; frames = {"t": [], "rho": [], "u": [], "v": [], "p": [], "phi": []}

    def save():
        prim(U, W)
        frames["t"].append(t)
        for k, q in (("rho", 0), ("u", 1), ("v", 2), ("p", 3), ("phi", 4)):
            frames[k].append(W[q][s].astype(np.float32).copy())
    save(); nxt = dt_out; t0 = time.time(); nstep = 0
    while t < t_end - 1e-12:
        dt = cfl * dx / maxspeed(U, W)
        if t + dt >= nxt - 1e-12:
            dt = nxt - t
        rhs(U, W, dx, R, beta); U1[:] = U + dt * R
        rhs(U1, W, dx, R, beta); U2[:] = 0.75 * U + 0.25 * (U1 + dt * R)
        rhs(U2, W, dx, R, beta); U[:] = U / 3.0 + 2.0 / 3.0 * (U2 + dt * R)
        t += dt; nstep += 1
        if abs(t - nxt) < 1e-10:
            save(); nxt += dt_out
            print(f"N={N} t={t:.3f} steps={nstep} wall={time.time()-t0:.0f}s", flush=True)
            np.savez_compressed(out, x=x, y=y, **{k: np.array(vv) for k, vv in frames.items()})
    np.savez_compressed(out, x=x, y=y, **{k: np.array(vv) for k, vv in frames.items()})


if __name__ == "__main__":
    N = int(sys.argv[1]); out = sys.argv[2]
    te = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    do = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    w0 = float(sys.argv[5]) if len(sys.argv) > 5 else 2.5e-3
    beta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0   # > 0: THINC interface reconstruction
    run(N, out, te, do, w0=w0, beta=beta)
