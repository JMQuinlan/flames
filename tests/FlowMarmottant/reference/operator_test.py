"""Solver-free test of the discrete capillary operator on a static circle.

Reproduces the production L_cap stencil (CapillaryOperator in Hydro2.cpp):
centred gradient of the colour function, cell-centred tensor components,
arithmetic face averages, conservative face differences.  Compares it with the
CSF body force sigma*kappa*grad(c) (kappa from centred differences) and with
the analytic force.  Outputs data/operator_test.npz.
"""
import os, numpy as np
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
L, R0, EPS = 5.0e-3, 1.025e-3, 7.8125e-5
SIG = 0.55 * ((R0 / 1.0e-3) ** 2 - 1.0)          # sigma_0 of the R0 = 1.025 mm case

def dcen(f, ax, dx):
    return (np.roll(f, -1, ax) - np.roll(f, 1, ax)) / (2.0 * dx)

def fave_diff(q, ax, dx):
    """(q_{i+1/2} - q_{i-1/2})/dx with q_{i+1/2} = (q_i + q_{i+1})/2."""
    return (0.5 * (q + np.roll(q, -1, ax)) - 0.5 * (q + np.roll(q, 1, ax))) / dx

def helmholtz_sol(Fx, Fy, dx):
    N = Fx.shape[0]
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX**2 + KY**2; K2[0, 0] = 1.0
    fx, fy = np.fft.fft2(Fx), np.fft.fft2(Fy)
    div = KX * fx + KY * fy
    gx, gy = KX * div / K2, KY * div / K2          # irrotational part
    sx, sy = np.real(np.fft.ifft2(fx - gx)), np.real(np.fft.ifft2(fy - gy))
    # pressure balancing the irrotational part: grad p = F_irr
    phat = -1j * div / K2; phat[0, 0] = 0.0
    p = np.real(np.fft.ifft2(phat))
    return sx, sy, p

res = {}
for ratio in (1, 2, 4, 8, 16):
    N = int(round(L / (EPS / ratio)))
    dx = L / N
    x = (np.arange(N) + 0.5) * dx - L / 2
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    c = 0.5 * (1.0 + np.tanh((r - R0) / EPS))
    wx, wy = dcen(c, 0, dx), dcen(c, 1, dx)
    wn = np.hypot(wx, wy)
    live = wn > 1.0e-10
    g = np.where(live, wn, 1.0)
    s = np.where(live, SIG, 0.0)
    qN, q11, q22, q12 = (np.where(live, a, 0.0) for a in (wn, wx * wx / g, wy * wy / g, wx * wy / g))
    # production: sigma averaged to faces together with the tensor components
    Fx_css = fave_diff(s * qN, 0, dx) - fave_diff(s * q11, 0, dx) - fave_diff(s * q12, 1, dx)
    Fy_css = fave_diff(s * qN, 1, dx) - fave_diff(s * q12, 0, dx) - fave_diff(s * q22, 1, dx)
    # CSF: F = -sigma (div n) grad c, n = w/|w|
    nx, ny = np.where(live, wx / g, 0.0), np.where(live, wy / g, 0.0)
    kap = dcen(nx, 0, dx) + dcen(ny, 1, dx)
    Fx_csf, Fy_csf = -SIG * kap * wx, -SIG * kap * wy
    # analytic: F = -sigma c'(r)/r r_hat
    cp = 0.5 / EPS / np.cosh((r - R0) / EPS) ** 2
    Fx_ex, Fy_ex = -SIG * cp / np.maximum(r, dx) * X / np.maximum(r, dx), -SIG * cp / np.maximum(r, dx) * Y / np.maximum(r, dx)
    out = {}
    for name, (Fx, Fy) in (("css", (Fx_css, Fy_css)), ("csf", (Fx_csf, Fy_csf)), ("exact", (Fx_ex, Fy_ex))):
        sx, sy, p = helmholtz_sol(Fx, Fy, dx)
        xi = np.sqrt(np.sum(sx**2 + sy**2) / np.sum(Fx**2 + Fy**2))
        dp = p[r < 0.5 * R0].mean() - p[r > 2.0 * R0].mean()
        j = N // 2
        Fr = Fx[j:, j] if False else None
        # radial force along +x ray (row through centre)
        row = N // 2
        ray = slice(N // 2, N)
        out[name] = dict(xi=xi, dp=dp, xr=x[ray], Fr=Fx[ray, row], sx=sx, sy=sy)
    res[ratio] = out
    print(f"eps/dx={ratio:2d} N={N:5d}  xi css={out['css']['xi']:.3e} csf={out['csf']['xi']:.3e} "
          f"exact={out['exact']['xi']:.3e}  dp/(sig/R0): css={out['css']['dp']*R0/SIG:.5f} "
          f"csf={out['csf']['dp']*R0/SIG:.5f} exact={out['exact']['dp']*R0/SIG:.5f}")

save = {"ratios": np.array(sorted(res)), "sigma": SIG, "R0": R0, "eps": EPS}
for rt in res:
    for name in ("css", "csf", "exact"):
        save[f"xi_{name}_{rt}"] = res[rt][name]["xi"]
        save[f"dp_{name}_{rt}"] = res[rt][name]["dp"]
        save[f"xr_{name}_{rt}"] = res[rt][name]["xr"]
        save[f"Fr_{name}_{rt}"] = res[rt][name]["Fr"]
# keep the solenoidal residual map of the production operator at eps/dx = 4
save["solmag_css_4"] = np.hypot(res[4]["css"]["sx"], res[4]["css"]["sy"])
save["solmag_csf_4"] = np.hypot(res[4]["csf"]["sx"], res[4]["csf"]["sy"])
np.savez_compressed(os.path.join(OUT, "operator_test.npz"), **save)
