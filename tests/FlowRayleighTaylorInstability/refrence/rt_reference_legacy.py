"""
Runs the ORIGINAL reference scheme (Raleigh_Taylor_Instability_STANDALONE.py:
MUSCL-Hancock, Rusanov flux, unlimited slopes, sharp step IC) and stores the
frames in the same .npz layout as rt_reference_hllc.py, so that the old
reference can be put on the same plots.  phi = rho - 1 (no colour function in
the original code; densities are 1 and 2).
Usage: python3 rt_reference_legacy.py N OUT.npz [t_end] [dt_out]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import Raleigh_Taylor_Instability_STANDALONE as S


def run(N, out, tEnd=20.0, tOut=0.5):
    gamma, cfl, g, w0, P0 = 1.4, 0.4, -0.1, 0.0025, 2.5
    dx = 0.5 / N; vol = dx * dx
    xlin = np.linspace(0.5 * dx, 0.5 - 0.5 * dx, N); ylin = np.linspace(0.5 * dx, 1.5 - 0.5 * dx, 3 * N)
    Y, X = np.meshgrid(ylin, xlin)
    rho = 1. + (Y > 0.75); vx = np.zeros(X.shape)
    vy = w0 * (1 - np.cos(4 * np.pi * X)) * (1 - np.cos(4 * np.pi * Y / 3))
    P = P0 + g * (Y - 0.75) * rho
    rho, vx, vy, P = S.addGhostCells(rho, vx, vy, P)
    M, Mx, My, E = S.getConserved(rho, vx, vy, P, gamma, vol)
    fr = dict(t=[], rho=[], u=[], v=[], p=[], phi=[])

    def save(t):
        r, u, v, p = S.getPrimitive(M, Mx, My, E, gamma, vol)
        fr["t"].append(t)
        for k, a in (("rho", r), ("u", u), ("v", v), ("p", p), ("phi", r - 1.0)):
            fr[k].append(a[:, 1:-1].astype(np.float32).copy())
    t = 0.0; save(t); k = 1
    while t < tEnd - 1e-12:
        rho, vx, vy, P = S.getPrimitive(M, Mx, My, E, gamma, vol)
        dt = cfl * np.min(dx / (np.sqrt(gamma * P / rho) + np.sqrt(vx**2 + vy**2)))
        hit = False
        if t + dt >= k * tOut - 1e-12:
            dt = k * tOut - t; hit = True
        M, Mx, My, E = S.addSourceTerm(M, Mx, My, E, g, dt / 2)
        rho, vx, vy, P = S.getPrimitive(M, Mx, My, E, gamma, vol)
        rdx, rdy = S.getGradient(rho, dx); udx, udy = S.getGradient(vx, dx)
        vdx, vdy = S.getGradient(vy, dx); pdx, pdy = S.getGradient(P, dx)
        rp = rho - 0.5 * dt * (vx * rdx + rho * udx + vy * rdy + rho * vdy)
        up = vx - 0.5 * dt * (vx * udx + vy * udy + (1 / rho) * pdx)
        vp = vy - 0.5 * dt * (vx * vdx + vy * vdy + (1 / rho) * pdy)
        pp = P - 0.5 * dt * (gamma * P * (udx + vdy) + vx * pdx + vy * pdy)
        rXL, rXR, rYL, rYR = S.extrapolateInSpaceToFace(rp, rdx, rdy, dx)
        uXL, uXR, uYL, uYR = S.extrapolateInSpaceToFace(up, udx, udy, dx)
        vXL, vXR, vYL, vYR = S.extrapolateInSpaceToFace(vp, vdx, vdy, dx)
        pXL, pXR, pYL, pYR = S.extrapolateInSpaceToFace(pp, pdx, pdy, dx)
        fMX, fMxX, fMyX, fEX = S.getFlux(rXL, rXR, uXL, uXR, vXL, vXR, pXL, pXR, gamma)
        fMY, fMyY, fMxY, fEY = S.getFlux(rYL, rYR, vYL, vYR, uYL, uYR, pYL, pYR, gamma)
        M = S.applyFluxes(M, fMX, fMY, dx, dt); Mx = S.applyFluxes(Mx, fMxX, fMxY, dx, dt)
        My = S.applyFluxes(My, fMyX, fMyY, dx, dt); E = S.applyFluxes(E, fEX, fEY, dx, dt)
        M, Mx, My, E = S.addSourceTerm(M, Mx, My, E, g, dt / 2)
        t += dt
        if hit:
            save(t); k += 1
    x = xlin; y = ylin
    np.savez_compressed(out, x=x, y=y, **{kk: np.array(vv) for kk, vv in fr.items()})


if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2], *(float(a) for a in sys.argv[3:]))
