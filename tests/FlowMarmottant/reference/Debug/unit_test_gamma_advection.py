#!/usr/bin/env python3
"""
MANUFACTURED-SOLUTION UNIT TEST for the Gamma / alpha advection operator.

WHY.  Measuring Gamma error inside a collapsing bubble proved unreliable:
Gamma lives in a ~1-cell band, so volume averages are contaminated by bulk
cells where Gamma is reset, and the "exact" value (R0/R)^2 depends on a
measured R that is itself resolution-biased.  Three metrics gave three
different answers, one non-monotone in resolution.  This removes the
simulation and tests the operator alone against an analytic solution.

WHAT.  Pure advection at constant velocity -- the part that was changed:
    dG/dt + u dG/dx = 0        exact:  G(x,t) = G0(x - u t)
A tanh interface profile of FIXED physical width eps is advected a FIXED
distance while only dx is refined, so eps/dx (cells per interface width) is
the sole convergence parameter.

TWO HARNESS BUGS THIS FILE HAD, AND WHY THEY MATTER:
  1. eps and the integration time were varied along with dx, so nothing was
     held fixed.  Donor-cell then appeared to BEAT weno5, which is impossible.
  2. The reconstruction was fed POINT VALUES at cell centres.  WENO
     reconstructs from CELL AVERAGES; with point values even the exact linear
     5th-order combination measures 2nd order.  Verified after the fix:
     linear5 -> 5.00, weno5 -> 5.00, donor -> 1.00 on a smooth sine.
Both are the same class of mistake as the one being chased in the solver:
comparing two things that were not actually the same quantity.

No fitted constants: the reference is analytic and the only knob is the mesh.
"""
import math
import numpy as np

EPSW = 0.02          # physical interface width (fixed)
L    = 1.0
U    = 1.0
DIST = 0.10          # advection distance (fixed) = 5 interface widths
X0   = 0.30
CFL  = 0.4


def tanh_avg(a, b, x0=X0, e=EPSW):
    """Exact cell average of 0.5(1+tanh((x-x0)/e)) over [a,b].

    Antiderivative: int 0.5(1+tanh(z)) dx = 0.5(x + e log cosh z), z=(x-x0)/e,
    and log cosh z = z + log(1+exp(-2z)) - log 2 (written this way so it stays
    finite for large |z|).  Verified against quadrature to 1e-9.
    """
    def F(x):
        z = (x - x0) / e
        logcosh = z + np.logaddexp(0.0, -2.0 * z) - np.log(2.0)
        return 0.5 * (x + e * logcosh)
    return (F(b) - F(a)) / (b - a)


def weno5_face(q, e=1.0e-6):
    g = (0.1, 0.6, 0.3); n = len(q) - 6; out = np.empty(n + 1)
    for i in range(n + 1):
        j = i + 2      # face i sits on the RIGHT of cell i-1 (upwind for U>0)
        a, b, c, d, f = q[j-2], q[j-1], q[j], q[j+1], q[j+2]
        p = ((2*a - 7*b + 11*c)/6.0, (-b + 5*c + 2*d)/6.0, (2*c + 5*d - f)/6.0)
        B = (13/12*(a - 2*b + c)**2 + 0.25*(a - 4*b + 3*c)**2,
             13/12*(b - 2*c + d)**2 + 0.25*(b - d)**2,
             13/12*(c - 2*d + f)**2 + 0.25*(3*c - 4*d + f)**2)
        w = [g[k] / (e + B[k])**2 for k in range(3)]; s = sum(w)
        out[i] = sum(w[k] * p[k] for k in range(3)) / s
    return out


def advect(scheme, N):
    dx = L / N
    edges = np.arange(N + 1) * dx
    G = tanh_avg(edges[:-1], edges[1:])
    T = DIST / U
    nt = max(1, int(math.ceil(T / (CFL * dx / U))))
    dt = T / nt
    gl = tanh_avg(edges[0] - 3*dx + np.arange(3)*dx,
                  edges[0] - 2*dx + np.arange(3)*dx)
    gr = tanh_avg(edges[-1] + np.arange(3)*dx,
                  edges[-1] + dx + np.arange(3)*dx)

    def rhs(Gc):
        q = np.concatenate([gl, Gc, gr])
        # donor: upwind (left) cell value; weno5: its reconstructed right face
        face = q[2:N+3] if scheme == "donor" else weno5_face(q)
        return -(U / dx) * (face[1:] - face[:-1])

    # SSP-RK3.  Forward Euler caps the WHOLE scheme at O(dt) = O(dx), which
    # hides the spatial order completely (measured: weno5 looked 1st order and
    # no better than donor).  The spatial reconstruction cannot be measured
    # with a 1st-order time integrator.
    for _ in range(nt):
        G1 = G + dt * rhs(G)
        G2 = 0.75 * G + 0.25 * (G1 + dt * rhs(G1))
        G  = (1.0/3.0) * G + (2.0/3.0) * (G2 + dt * rhs(G2))
    exact = tanh_avg(edges[:-1] - U*T, edges[1:] - U*T)
    return float(np.sqrt(np.mean((G - exact)**2)))


def main():
    print(__doc__)
    print(f"{'N':>6} {'eps/dx':>8} {'donor L2':>12} {'p':>6} {'weno5 L2':>12} {'p':>6} {'gain':>8}")
    rows = []; prev = None
    for N in (32, 64, 128, 256, 512, 1024):
        epd = EPSW / (L / N)
        ed, ew = advect("donor", N), advect("weno5", N)
        if prev:
            r = math.log(2.0)
            pd = math.log(prev[0]/ed)/r; pw = math.log(prev[1]/ew)/r
            print(f"{N:6d} {epd:8.2f} {ed:12.4e} {pd:6.2f} {ew:12.4e} {pw:6.2f} {ed/ew:7.1f}x")
        else:
            print(f"{N:6d} {epd:8.2f} {ed:12.4e} {'-':>6} {ew:12.4e} {'-':>6} {ed/ew:7.1f}x")
        rows.append((epd, ed, ew)); prev = (ed, ew)

    a = np.array(rows)
    # save for plotting
    import os
    img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Images")
    os.makedirs(img, exist_ok=True)
    np.savetxt(os.path.join(img, "gamma_advection_convergence.csv"), a,
               delimiter=",", header="eps_over_dx,donor_L2,weno5_L2",
               comments="", fmt="%.8g")
    print(f"\n  wrote {img}/gamma_advection_convergence.csv")

    print("\n  eps/dx needed for L2 error <= 2 % of the profile amplitude:")
    for nm, col in (("donor (legacy)", 1), ("weno5 (current)", 2)):
        ok = a[a[:, col] <= 0.02]
        print(f"    {nm:16s}: eps/dx >= {ok[0,0]:.1f}" if len(ok)
              else f"    {nm:16s}: not reached by eps/dx = {a[-1,0]:.0f}")
    print("\n  eps/dx needed for L2 error <= 0.2 % (i.e. ~99.8 %):")
    for nm, col in (("donor (legacy)", 1), ("weno5 (current)", 2)):
        ok = a[a[:, col] <= 0.002]
        print(f"    {nm:16s}: eps/dx >= {ok[0,0]:.1f}" if len(ok)
              else f"    {nm:16s}: not reached by eps/dx = {a[-1,0]:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
