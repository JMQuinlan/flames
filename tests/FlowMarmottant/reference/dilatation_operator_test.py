#!/usr/bin/env python3
"""
Solver-free test of the SURFACE DILATATION as the shell stress sees it, on prescribed interfaces.

The viscous shell tension is kappa_s (div_s u), with div_s u = div u - n.grad(u).n = P : grad u evaluated cell by cell
with central differences and the normal n = grad(eta)/|grad eta|, and it acts through the |grad eta| weight of the
band.  Here eta and u are PRESCRIBED, so the answer is known:
   interface   r = Rs(theta), a superellipse |x|^p + |y|^p = a^p   (p = 2 circle; p > 2 flattened on the axes,
               "square"; p < 2 flattened on the diagonals, "diamond"), with eta = (1 + tanh(d / eps)) / 2,
               d = signed distance to the curve;
   velocity    the incompressible line sink  u = -A x / |x|^2   (a collapsing cylinder for p = 2), for which
               div_s u = A (1 - 2 (n . r_hat)^2) / r^2   exactly, = -A / R^2 = R'/R on a circle.
For each case and each angular wedge: curvature x mean radius, the exact div_s u ON the eta = 0.5 curve, and the
|grad eta|-weighted band mean of the discrete P : grad u.  RECOVERED = band value / exact value.

  python3 dilatation_operator_test.py [--outdir DIR]
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))


def curve(p, a, th):
    """superellipse radius, its curvature and outward unit normal at polar angle th (first quadrant)."""
    c, s = np.cos(th), np.sin(th); r = a / (c ** p + s ** p) ** (1.0 / p)
    x, y = r * c, r * s
    fx, fy = p * x ** (p - 1), p * y ** (p - 1)                                   # gradient of F = x^p + y^p
    fxx, fyy = p * (p - 1) * x ** (p - 2), p * (p - 1) * y ** (p - 2)
    g = np.hypot(fx, fy); kap = (fxx * fy * fy + fyy * fx * fx) / g ** 3            # curvature of the level curve
    return r, kap, fx / g, fy / g


def case(p, R_over_dx, eps_over_dx, nang=18, A=1.0):
    dx = 1.0 / R_over_dx; a = 1.0; eps = eps_over_dx * dx
    n = int(1.6 / dx) + 8; xc = (np.arange(-4, n) + 0.5) * dx                      # 4 ghost cells below the symmetry planes
    X, Y = np.meshgrid(xc, xc, indexing="ij"); ax, ay = np.abs(X), np.abs(Y)
    # signed distance to the superellipse: first-order from the level-set function (exact for the circle)
    F = (ax ** p + ay ** p) ** (1.0 / p) - a
    Fx = np.gradient(F, dx, axis=0); Fy = np.gradient(F, dx, axis=1); d = F / np.maximum(np.hypot(Fx, Fy), 1e-12)
    eta = 0.5 * (1.0 + np.tanh(d / eps))
    r2 = np.maximum(X * X + Y * Y, 1e-30); u, v = -A * X / r2, -A * Y / r2
    cd = lambda q, axis: (np.roll(q, -1, axis) - np.roll(q, 1, axis)) / (2 * dx)
    ex, ey = cd(eta, 0), cd(eta, 1); g = np.hypot(ex, ey) + 1e-300; n1, n2 = ex / g, ey / g
    ux, uy, vx, vy = cd(u, 0), cd(u, 1), cd(v, 0), cd(v, 1)
    D = (ux + vy) - (n1 * n1 * ux + n1 * n2 * (uy + vx) + n2 * n2 * vy)
    sel = (X > 0) & (Y > 0) & (eta > 0.02) & (eta < 0.98);                    # ghost rows (mirror images) are outside X, Y > 0
    TH = np.arctan2(Y, X); wedge = np.minimum((TH / (np.pi / 2) * nang).astype(int), nang - 1)
    th = (np.arange(nang) + 0.5) * (np.pi / 2) / nang; r, kap, nx, ny = curve(p, a, th)
    rhat_n = nx * np.cos(th) + ny * np.sin(th); Dex = A * (1.0 - 2.0 * rhat_n ** 2) / r ** 2
    Dnum = np.full(nang, np.nan)
    for k in range(nang):
        m = sel & (wedge == k)
        if m.any(): Dnum[k] = (g[m] * D[m]).sum() / g[m].sum()
    return dict(th=th, kR=kap * np.mean(r), Dex=Dex, Dnum=Dnum, rec=Dnum / Dex, r=r)


def main():
    args = sys.argv[1:]; outdir = args[args.index("--outdir") + 1] if "--outdir" in args else os.path.join(HERE, "Images", "DilatationOperator")
    os.makedirs(outdir, exist_ok=True)
    print("ROUND interface (p = 2): recovered = band-weighted discrete div_s u / exact R'/R")
    print("   R/dx    eps/dx = 1      2       3       4")
    tab = {}
    for Rdx in (6.4, 12.8, 25.6, 51.2, 102.4):
        row = [np.nanmean(case(2.0, Rdx, e)["rec"]) for e in (1, 2, 3, 4)]; tab[Rdx] = row
        print("  %6.1f      " % Rdx + "  ".join("%.4f" % v for v in row))
    print("   model 1 - c eps/R:  c = " + ", ".join("%.2f" % ((1 - tab[Rdx][1]) * Rdx / 2) for Rdx in tab) + "  (from the eps/dx = 2 column)")
    shapes = [(1.6, "Diamond (p = 1.6)"), (1.8, "p = 1.8"), (2.0, "Circle (p = 2)"), (2.4, "p = 2.4"), (3.0, "Square-ish (p = 3)")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for j, (p, lab) in enumerate(shapes):
        c = case(p, 25.6, 2.0); col = f"C{j}"
        ax[0].plot(np.degrees(c["th"]), c["kR"], "-", color=col, label=lab)
        ax[1].plot(np.degrees(c["th"]), c["Dex"] * np.mean(c["r"]) ** 2, "-", color=col, lw=1.4, label="Exact" if j == 0 else None)
        ax[1].plot(np.degrees(c["th"]), c["Dnum"] * np.mean(c["r"]) ** 2, "o", ms=3.5, mfc="none", mec=col, mew=0.8, label="Numerical Operator" if j == 0 else None)
        ax[2].plot(c["kR"], c["rec"], "o", ms=4, mfc="none", mec=col, mew=0.8, label=lab)
        print("%-20s R/dx 25.6, eps/dx 2:  curvature x R %.2f .. %.2f   exact dilatation / (R'/R of the circle) %.2f .. %.2f   recovered %.3f .. %.3f (mean %.3f)"
              % (lab, c["kR"].min(), c["kR"].max(), (-c["Dex"] * np.mean(c["r"]) ** 2).min(), (-c["Dex"] * np.mean(c["r"]) ** 2).max(), np.nanmin(c["rec"]), np.nanmax(c["rec"]), np.nanmean(c["rec"])))
    for Rdx, m in ((12.8, "s"), (51.2, "^")):
        for j, (p, lab) in enumerate(shapes):
            c = case(p, Rdx, 2.0); ax[2].plot(c["kR"], c["rec"], m, ms=3, mfc="none", mec=f"C{j}", mew=0.5, alpha=0.6)
    ax[0].set_xlabel("Polar Angle [deg]"); ax[0].set_ylabel(r"Curvature $\times$ Mean Radius [-]"); ax[0].set_title("Interface Shape"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("Polar Angle [deg]"); ax[1].set_ylabel(r"Surface Dilatation $\times$ R$^2$ / A [-]"); ax[1].set_title("Surface Dilatation by Angle (25.6 Cells per Radius)"); ax[1].legend(fontsize=8)
    ax[2].set_xlabel(r"Curvature $\times$ Mean Radius [-]"); ax[2].set_ylabel("Recovered Dilatation: Band / Exact [-]"); ax[2].axhline(1.0, color="k", lw=1.0)
    ax[2].set_title("Recovered Dilatation (Circles 25.6, Squares 12.8, Triangles 51.2 Cells per Radius)", fontsize=9); ax[2].set_ylim(0.4, 1.4)
    for a_ in ax: a_.grid(alpha=0.3)
    fig.suptitle("Shell Surface Dilatation on Prescribed Interfaces: Discrete Operator Against the Exact Value (Interface Two Cells Wide)"); fig.tight_layout()
    o = os.path.join(outdir, "dilatation_operator.png"); fig.savefig(o, dpi=200); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
