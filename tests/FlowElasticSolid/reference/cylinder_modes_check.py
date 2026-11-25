#!/usr/bin/env python3
"""
Ringing of the elastic cylinder after the shock has passed (tests/FlowElasticSolid/UNIT_TEST_2D_ShockCylinder)
against the free-vibration frequencies of an elastic cylinder in plane strain.

  python3 cylinder_modes_check.py <run_dir> [--outdir DIR] [--t0 T]

Analytical: traction-free cylinder of radius a, potentials phi = A J_n(alpha r) cos n theta,
psi = B J_n(beta r) sin n theta (alpha = omega/c_l, beta = omega/c_t); sigma_rr = sigma_rtheta = 0 at r = a gives a
2 x 2 determinant in omega for each circumferential order n (n = 0: breathing, x J0(x)/J1(x) = 2 (c_t/c_l)^2 with
x = omega a/c_l; n = 2: oval mode; ...).  Small amplitude, no surrounding gas (the gas adds some damping and a
small frequency shift; rho_gas/rho_solid = 0.1 - 0.27 here).
Numerical: the alpha_s = 0.5 contour in each plot file -> r(theta) about the solid's centroid -> Fourier
amplitudes a_n(t) = (1/pi) int r cos(n theta) d theta (a_0 = mean radius).  For t > t0 (default: after the shock
has left the body) each a_n is fitted by  c0 + c1 t + A exp(-g t) cos(2 pi f t + ph)  and f compared.
The frequency resolution of the record is ~1/(record length); with ~2-3 periods of the oval mode the fitted f is
good to roughly 10 %.

Writes cylinder_modes.png and cylinder_modes_summary.csv to <outdir> (default Images/<run_dir name>/).
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import jv, jvp
from scipy.optimize import brentq, least_squares
import yt
yt.funcs.mylog.setLevel(40)

RHO, GAM, PINF, P0, MU, A0 = 10.0, 4.4, 100.0, 1.0, 50.0, 0.2
CL = np.sqrt((GAM * (P0 + PINF) + 4.0 / 3.0 * MU) / RHO); CT = np.sqrt(MU / RHO)


def det(n, w, a=A0):
    al, be = w / CL, w / CT; lam = RHO * CL**2 - 2 * MU
    Jn, Jnp, Jnpp = jv(n, al * a), jvp(n, al * a, 1), jvp(n, al * a, 2)
    Kn, Knp, Knpp = jv(n, be * a), jvp(n, be * a, 1), jvp(n, be * a, 2)
    if n == 0: return -lam * al**2 * Jn + 2 * MU * al**2 * Jnpp
    m11 = -lam * al**2 * Jn + 2 * MU * al**2 * Jnpp; m12 = 2 * MU * n * (be * Knp / a - Kn / a**2)
    m21 = -2 * n / a * al * Jnp + 2 * n / a**2 * Jn; m22 = -be**2 * Knpp + be / a * Knp - n**2 / a**2 * Kn
    return m11 * m22 - m12 * m21


def modes(n, nmax=3, wmax=400.0):
    w = np.linspace(1e-3, wmax, 40001); d = np.array([det(n, x) for x in w]); out = []
    for i in np.where(np.sign(d[1:]) != np.sign(d[:-1]))[0]:
        r = brentq(lambda x: det(n, x), w[i], w[i + 1])
        if r > 0.5: out.append(r / (2 * np.pi))
        if len(out) >= nmax: break
    return out


def shape(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    eta = np.asarray(cg[("boxlib", "eta")])[:, :, 0]; m = np.asarray(cg[("boxlib", "rho_eta0")])[:, :, 0]
    lo, hi = np.asarray(ds.domain_left_edge), np.asarray(ds.domain_right_edge); nx, ny = eta.shape
    x = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx; y = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    X, Y = np.meshgrid(x, y, indexing="ij"); xc, yc = (m * X).sum() / m.sum(), (m * Y).sum() / m.sum()
    cs = plt.figure().gca().contour(x, y, eta.T, levels=[0.5]); plt.close()
    P = max((s for s in cs.allsegs[0] if len(s)), key=len); th = np.arctan2(P[:, 1] - yc, P[:, 0] - xc); r = np.hypot(P[:, 0] - xc, P[:, 1] - yc)
    o = np.argsort(th); th, r = th[o], r[o]; tg = np.linspace(-np.pi, np.pi, 721)[:-1]; rg = np.interp(tg, th, r, period=2 * np.pi)
    a = [rg.mean()] + [2 * (rg * np.cos(n * tg)).mean() for n in (1, 2, 3, 4)]
    return float(ds.current_time), a, x[1] - x[0]


def fit(t, y, f0):
    def model(q, t): return q[0] + q[1] * t + q[2] * np.exp(-q[3] * t) * np.cos(2 * np.pi * q[4] * t + q[5])
    best = None
    for fg in f0 * np.array([0.7, 0.85, 1.0, 1.15, 1.3]):
        for ph in (0.0, 1.5, 3.0, 4.5):
            q0 = [y.mean(), 0.0, 0.5 * (y.max() - y.min()), 0.5, fg, ph]
            r = least_squares(lambda q: model(q, t) - y, q0, bounds=([-np.inf, -np.inf, 0, 0, 0.3 * f0, -np.inf], [np.inf, np.inf, np.inf, 50, 3 * f0, np.inf]))
            if best is None or r.cost < best.cost: best = r
    q = best.x; res = np.sqrt(2 * best.cost / len(t)); return q, model(q, t), res


def main():
    d = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
    outdir = (sys.argv[sys.argv.index("--outdir") + 1] if "--outdir" in sys.argv else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    t0 = float(sys.argv[sys.argv.index("--t0") + 1]) if "--t0" in sys.argv else 0.35
    os.makedirs(outdir, exist_ok=True)
    F = {n: modes(n) for n in (0, 2, 3, 4)}
    print(f"c_l = {CL:.3f}, c_t = {CT:.3f}, a = {A0};  free plane-strain modes f = omega/2pi:")
    for n in F: print(f"  n = {n}: " + ", ".join(f"{f:.3f}" for f in F[n]))
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    H = [shape(p) for p in pfs]; t = np.array([h[0] for h in H]); A = np.array([h[1] for h in H])
    print(f"{d}: {len(t)} plot files, dt = {t[1] - t[0]:.3f}, to t = {t[-1]:.2f};  fit window t > {t0} ({(t > t0).sum()} samples, record {t[-1] - t0:.2f})")
    fig, ax = plt.subplots(3, 1, figsize=(9, 9), sharex=True); rows = []
    for a_, (k, n, name) in zip(ax, ((0, 0, "Mean Radius (n = 0, Breathing)"), (2, 2, "Oval Amplitude (n = 2)"), (3, 3, "Triangular Amplitude (n = 3)"))):
        m = t > t0; q, yfit, res = fit(t[m], A[m, k], F[n][0])
        a_.plot(t, A[:, k], "ko", ms=3, label="Numerical"); a_.plot(t[m], yfit, "C3-", lw=1.1, label=f"Damped-Sine Fit, f = {q[4]:.2f}")
        a_.axvline(t0, color="0.6", ls=":"); a_.set_ylabel(name); a_.grid(alpha=0.3); a_.legend(fontsize=8)
        amp = q[2]; sig = amp / max(res, 1e-30)
        print(f"  n = {n}: fitted f = {q[4]:.3f}  (analytical lowest {F[n][0]:.3f}, {100 * (q[4] / F[n][0] - 1):+.1f} %);  amplitude {amp:.2e} = {amp / H[0][2]:.2f} cells, decay rate {q[3]:.2f}, fit residual {res:.1e} (amplitude / residual {sig:.1f})")
        rows.append((n, q[4], F[n][0], amp, q[3], res))
    ax[-1].set_xlabel("t"); fig.suptitle("Ringing of the Elastic Cylinder After Shock Passage", fontsize=13)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "cylinder_modes.png"), dpi=180); plt.close(fig)
    with open(os.path.join(outdir, "cylinder_modes_summary.csv"), "w") as fo:
        fo.write("n,f_fit,f_analytical,amplitude,decay_rate,fit_residual\n")
        for r in rows: fo.write(",".join(f"{v:.6g}" for v in r) + "\n")
    print(f"  wrote {outdir}/cylinder_modes.png, cylinder_modes_summary.csv")


if __name__ == "__main__":
    main()
