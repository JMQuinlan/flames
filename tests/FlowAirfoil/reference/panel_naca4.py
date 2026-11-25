#!/usr/bin/env python3
"""
Inviscid 2D lift of a NACA 4-digit section: linear-strength vortex panel method
(Kuethe & Chow, "Foundations of Aerodynamics", ch. 5) with the Kutta condition,
on the SAME geometry the Hydro2 bitmaps use (TEMPLATES/make_naca0012_bmp.naca4:
closed trailing edge, last thickness coefficient -0.1036, chord 1).

  python3 panel_naca4.py [naca4=0008] [mach=0.2] [aoa list, default 0..8]

Prints Cl (incompressible), Cl with the Prandtl-Glauert factor 1/sqrt(1-M^2), and
thin-airfoil 2 pi alpha.  Cl from Cp integrated over the panels (Cd_p ~ 0 is the
d'Alembert check).  Self-check: NACA 0012 at 5 deg gives ~0.60 (Abbott & von
Doenhoff inviscid ~0.6).  Requires numpy only.
"""
import math, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TEMPLATES"))
from make_naca0012_bmp import naca4   # noqa: E402


def panel_cl(code, aoa_deg, n=240):
    pts = naca4(code, n)                              # LE -> upper -> TE -> lower -> LE (open loop)
    k = int(np.argmax(pts[:, 0]))                     # trailing-edge node
    # Kuethe & Chow order: TE -> lower -> LE -> upper -> TE (clockwise), closed
    loop = np.vstack([pts[k:], pts[:k + 1]])
    x, y = loop[:, 0], loop[:, 1]
    m = len(x) - 1
    a = math.radians(aoa_deg)
    xm = 0.5 * (x[:-1] + x[1:]); ym = 0.5 * (y[:-1] + y[1:])
    S = np.hypot(np.diff(x), np.diff(y)); th = np.arctan2(np.diff(y), np.diff(x))
    CN1 = np.zeros((m, m)); CN2 = np.zeros((m, m)); CT1 = np.zeros((m, m)); CT2 = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                CN1[i, j], CN2[i, j], CT1[i, j], CT2[i, j] = -1.0, 1.0, 0.5 * math.pi, 0.5 * math.pi
                continue
            A = -(xm[i] - x[j]) * math.cos(th[j]) - (ym[i] - y[j]) * math.sin(th[j])
            B = (xm[i] - x[j])**2 + (ym[i] - y[j])**2
            C = math.sin(th[i] - th[j]); D = math.cos(th[i] - th[j])
            E = (xm[i] - x[j]) * math.sin(th[j]) - (ym[i] - y[j]) * math.cos(th[j])
            F = math.log(1.0 + S[j] * (S[j] + 2.0 * A) / B)
            G = math.atan2(E * S[j], B + A * S[j])
            P = (xm[i] - x[j]) * math.sin(th[i] - 2 * th[j]) + (ym[i] - y[j]) * math.cos(th[i] - 2 * th[j])
            Q = (xm[i] - x[j]) * math.cos(th[i] - 2 * th[j]) - (ym[i] - y[j]) * math.sin(th[i] - 2 * th[j])
            CN2[i, j] = D + 0.5 * Q * F / S[j] - (A * C + D * E) * G / S[j]
            CN1[i, j] = 0.5 * D * F + C * G - CN2[i, j]
            CT2[i, j] = C + 0.5 * P * F / S[j] + (A * D - C * E) * G / S[j]
            CT1[i, j] = 0.5 * C * F - D * G - CT2[i, j]
    AN = np.zeros((m + 1, m + 1)); AT = np.zeros((m, m + 1))
    AN[:m, 0] = CN1[:, 0]; AN[:m, m] = CN2[:, m - 1]
    AT[:, 0] = CT1[:, 0]; AT[:, m] = CT2[:, m - 1]
    AN[:m, 1:m] = CN1[:, 1:] + CN2[:, :-1]
    AT[:, 1:m] = CT1[:, 1:] + CT2[:, :-1]
    AN[m, 0] = AN[m, m] = 1.0                          # Kutta: gamma_TE,lower + gamma_TE,upper = 0
    rhs = np.append(np.sin(th - a), 0.0)
    g = np.linalg.solve(AN, rhs)
    V = np.cos(th - a) + AT @ g
    Cp = 1.0 - V**2
    nx, ny = -np.sin(th), np.cos(th)                   # outward normal for clockwise order
    Fx = np.sum(-Cp * nx * S); Fy = np.sum(-Cp * ny * S)
    chord = x.max() - x.min()
    return (Fy * math.cos(a) - Fx * math.sin(a)) / chord, (Fx * math.cos(a) + Fy * math.sin(a)) / chord


def main():
    code = sys.argv[1] if len(sys.argv) > 1 else "0008"
    M = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    aoas = [float(v) for v in sys.argv[3:]] or list(range(0, 9))
    beta = math.sqrt(1.0 - M * M)
    cl5, _ = panel_cl("0012", 5.0)
    print(f"self-check NACA 0012, 5 deg, incompressible: Cl = {cl5:.4f}  (expected ~0.60)")
    print(f"NACA {code}, Prandtl-Glauert M = {M:g} (x {1/beta:.4f})")
    print(" aoa   Cl_incomp   Cl_PG(M)   2*pi*a   Cd_p(check)")
    for aoa in aoas:
        cl, cd = panel_cl(code, aoa)
        print(f"{aoa:4.1f}   {cl:9.4f}   {cl/beta:8.4f}   {2*math.pi*math.radians(aoa):6.4f}   {cd:+.1e}")


if __name__ == "__main__":
    main()
