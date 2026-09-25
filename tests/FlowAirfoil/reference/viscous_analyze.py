#!/usr/bin/env python3
"""
Post-process a viscous NACA 0012 run written by viscous_naca0012.py.

  python3 viscous_analyze.py <case_dir> [A|B] [plotfile]

Reports, for the chosen (default: last) plotfile:
  * Cl, Cd from the pressure force on the staircase wall faces (sharp phi)
  * Cl, Cd TOTAL from a steady control-volume momentum balance on a rectangle
    around the body:  F = -oint [ rho u (u.n) + (p - p_inf) n - tau.n ] dS
    (valid once the flow is steady; tau = mu (grad u + grad u^T) - 2/3 mu div u I)
  * laminar separation point on the upper surface (first reversal of the
    wall-tangential velocity, measured 1.5 cells off the wall)
  * surface Cp, and a Mach / vorticity figure  -> <case_dir>/analysis_<plt>.png
"""
import glob, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscous_naca0012 as V

def load(pf, lo, hi):
    import yt
    yt.set_log_level(40)
    ds = yt.load(pf); L = ds.index.max_level
    dx0 = (V.DOM_HI[0] - V.DOM_LO[0]) / V.N_CELL[0]; dx = dx0 / 2**L
    i0 = np.floor((np.array(lo) - np.array(V.DOM_LO)) / dx).astype(int)
    i1 = np.ceil((np.array(hi) - np.array(V.DOM_LO)) / dx).astype(int)
    le = np.array(V.DOM_LO) + i0 * dx
    dims = list(i1 - i0) + [1]
    cg = ds.covering_grid(level=L, left_edge=list(le) + [0.0], dims=dims)
    f = lambda n: np.asarray(cg[("boxlib", n)])[:, :, 0]
    x = le[0] + (np.arange(dims[0]) + 0.5) * dx; y = le[1] + (np.arange(dims[1]) + 0.5) * dx
    return ds, f, x, y, dx, L

def main():
    d = sys.argv[1]; case = sys.argv[2] if len(sys.argv) > 2 else "A"
    c = V.CASES[case]; U = c["mach"]; mu = V.RHO * U * V.CHORD / c["re"]; q = 0.5 * V.RHO * U**2
    pf = sys.argv[3] if len(sys.argv) > 3 else None
    if pf is None:   # latest plotfile in out*/ (out_L4, out, ...), by step number
        cand = glob.glob(os.path.join(d, "out*", "*cell"))
        pf = max(cand, key=lambda q: int(os.path.basename(q)[:-4]))
    lo, hi = (-0.75, -0.30), (1.30, 0.30)
    ds, f, x, y, dx, L = load(pf, lo, hi)
    t = float(ds.current_time)
    phi = np.clip(f("phi"), 0, 1); p = f("pressure"); rho = f("density")
    ux = f("velocityx"); uy = f("velocityy"); a = f("a")
    X, Y = np.meshgrid(x, y, indexing="ij")
    a_ = math.radians(c["aoa"])
    # lift/drag axes: drag along freestream (+x), lift along +y (freestream is along x)
    # --- pressure force ---
    # Pressure force on the staircase wall = what the sharp-wall flux applies:
    # sum over faces between a fluid cell (phi >= 0.5) and a solid cell of
    # -(p_fluid - p_inf) n_out dA, n_out pointing from solid to fluid.  (The
    # diffuse form -int (p-p_inf) grad(phi) dV differentiates a STEP when phi
    # is sharp and picks up the solid-side pressure -- wrong by O(1).)
    sol = phi < 0.5
    Fpx = Fpy = 0.0
    fx = sol[:-1, :] & ~sol[1:, :]      # solid | fluid  (x-face, n_out = +x)
    Fpx -= np.sum(p[1:, :][fx] - V.P) * dx
    fx = ~sol[:-1, :] & sol[1:, :]      # fluid | solid  (n_out = -x)
    Fpx += np.sum(p[:-1, :][fx] - V.P) * dx
    fy = sol[:, :-1] & ~sol[:, 1:]      # solid below fluid (n_out = +y)
    Fpy -= np.sum(p[:, 1:][fy] - V.P) * dx
    fy = ~sol[:, :-1] & sol[:, 1:]      # fluid below solid (n_out = -y)
    Fpy += np.sum(p[:, :-1][fy] - V.P) * dx
    # --- control-volume total force on the rectangle cv ---
    cvlo, cvhi = (-0.65, -0.22), (1.20, 0.22)
    il = np.searchsorted(x, cvlo[0]); ih = np.searchsorted(x, cvhi[0]); jl = np.searchsorted(y, cvlo[1]); jh = np.searchsorted(y, cvhi[1])
    dudx = np.gradient(ux, dx, axis=0); dudy = np.gradient(ux, dx, axis=1)
    dvdx = np.gradient(uy, dx, axis=0); dvdy = np.gradient(uy, dx, axis=1)
    div = dudx + dvdy
    txx = mu * (2 * dudx - 2/3 * div); tyy = mu * (2 * dvdy - 2/3 * div); txy = mu * (dudy + dvdx)
    Fx = Fy = 0.0
    # faces: (index slice, normal)
    for (sl, n) in [((ih, slice(jl, jh)), (1, 0)), ((il, slice(jl, jh)), (-1, 0)),
                    ((slice(il, ih), jh), (0, 1)), ((slice(il, ih), jl), (0, -1))]:
        un = ux[sl] * n[0] + uy[sl] * n[1]
        fx = rho[sl] * ux[sl] * un + (p[sl] - V.P) * n[0] - (txx[sl] * n[0] + txy[sl] * n[1])
        fy = rho[sl] * uy[sl] * un + (p[sl] - V.P) * n[1] - (txy[sl] * n[0] + tyy[sl] * n[1])
        Fx -= np.sum(fx) * dx; Fy -= np.sum(fy) * dx
    # --- wall samples: fluid cells with a solid neighbour ---
    solid = phi < 0.5
    nb = np.zeros_like(solid)
    nb[1:, :] |= solid[:-1, :]; nb[:-1, :] |= solid[1:, :]; nb[:, 1:] |= solid[:, :-1]; nb[:, :-1] |= solid[:, 1:]
    wall = (~solid) & nb
    # body-frame coordinates (undo AoA rotation): xb along chord LE->TE
    cs, sn = math.cos(a_), math.sin(a_)
    Xb = X * cs - Y * sn; Yb = X * sn + Y * cs
    xc = Xb + 0.5
    Cp = (p - V.P) / q
    up = wall & (Yb > 0); lw = wall & (Yb < 0)
    # separation: tangential velocity (chordwise, body frame) 1.5 cells off the upper wall
    utb = ux * cs - uy * sn
    # sample: for each upper-wall column, the cell 1 above (in body normal ~ +y)
    ind = np.argwhere(up)
    sep = None
    xs, us = [], []
    for (i, j) in ind:
        jj = j + 1
        if jj < len(y) and not solid[i, jj]:
            xs.append(xc[i, j]); us.append(0.5 * (utb[i, j] + utb[i, jj]))
    xs = np.array(xs); us = np.array(us); o = np.argsort(xs); xs, us = xs[o], us[o]
    m = (xs > 0.2) & (xs < 1.0)
    rev = np.where(m & (us < 0))[0]
    if len(rev): sep = xs[rev[0]]
    Cl_p, Cd_p = Fpy / (q * V.CHORD), Fpx / (q * V.CHORD)
    Cl_t, Cd_t = Fy / (q * V.CHORD), Fx / (q * V.CHORD)
    print(f"{os.path.basename(pf)}  t = {t:.3f}  (t U/c = {t*U:.2f})  finest level {L}, dx = {dx:.5f}")
    print(f"  pressure force : Cl = {Cl_p:+.4f}   Cd_p = {Cd_p:+.4f}")
    print(f"  CV total force : Cl = {Cl_t:+.4f}   Cd   = {Cd_t:+.4f}   (friction ~ Cd - Cd_p = {Cd_t - Cd_p:+.4f})")
    print(f"  upper-surface separation x/c = {sep if sep is not None else 'none (attached)'}")
    print(f"  max Mach in window (fluid) = {np.nanmax(np.where(solid | (a <= 0), np.nan, np.hypot(ux, uy) / np.where(a > 0, a, 1))):.3f}")
    # --- figure ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        Ma = np.ma.masked_where(solid, np.hypot(ux, uy) / a)
        im = ax[0].pcolormesh(X, Y, Ma, shading="auto", cmap="viridis"); fig.colorbar(im, ax=ax[0], label="Mach")
        ax[0].streamplot(x, y, ux.T, uy.T, density=2.0, color="w", linewidth=0.4, arrowsize=0.5)
        ax[0].set_aspect("equal"); ax[0].set_xlim(-0.7, 1.25); ax[0].set_ylim(-0.3, 0.3); ax[0].set_title(f"Mach, t={t:.2f}")
        om = np.ma.masked_where(solid, dvdx - dudy)
        v = np.nanpercentile(np.abs(om), 99)
        im = ax[1].pcolormesh(X, Y, om, shading="auto", cmap="RdBu_r", vmin=-v, vmax=v); fig.colorbar(im, ax=ax[1], label="vorticity")
        ax[1].set_aspect("equal"); ax[1].set_xlim(-0.7, 1.25); ax[1].set_ylim(-0.3, 0.3); ax[1].set_title("vorticity")
        ax[2].plot(xc[up], -Cp[up], ".", ms=2, label="upper"); ax[2].plot(xc[lw], -Cp[lw], ".", ms=2, label="lower")
        ax[2].set_xlabel("x/c"); ax[2].set_ylabel("-Cp"); ax[2].set_xlim(0, 1); ax[2].grid(alpha=.3); ax[2].legend()
        ax[2].set_title(f"Cl_p={Cl_p:+.3f} Cd_p={Cd_p:.4f} | CV: Cl={Cl_t:+.3f} Cd={Cd_t:.4f} | sep x/c={sep if sep is None else round(sep,3)}")
        out = os.path.join(d, f"analysis_{os.path.basename(pf)}.png"); fig.tight_layout(); fig.savefig(out, dpi=110)
        print("  figure:", out)
    except Exception as e:
        print("  (figure skipped:", e, ")")

if __name__ == "__main__":
    main()
