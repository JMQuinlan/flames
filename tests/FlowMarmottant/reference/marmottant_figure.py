#!/usr/bin/env python3
"""
Surface-tension force  F.n_hat  across curvature treatments.

Shape: LEFT half a semicircle of radius R0 (constant curvature 1/R0),
       RIGHT half a triangle (flat sides kappa=0, three sharp corners).

Columns (all curvature columns use kappa_method = 1):
  Geometry                 eta field
  Exact Fsv                ANALYTIC level-set curvature, not finite-differenced
  Capillary + Filters      Hydro2_6Eqn_Marmottant: eta8 + kappa8 + sigma4, div(Omega)
  Capillary (BASE)         Hydro2_6Eqn_BASE:       eta8 only,              div(Omega)
  Brackbill (old)          raw kappa_1, CSF force sigma*kappa*grad(eta)
  kappa_1 (old)            raw kappa_1, no filters,                        div(Omega)
  New Method               sigma from advected shell density; Gamma=Gamma0 at t=0

Rows: interface resolution, eps/dx = 2, 4, 8, 16 cells across the diffuse band.

Normalisation: F.n / [sigma0 * (1/R0) * max|grad eta|], so a perfect scheme
reads -1 on the semicircular half.  Blue = inward (correct on a convex shell),
red = outward (spurious).

NOTE ON THE TENSOR NAME: the implemented stress is
    Omega_ij = sigma (|grad eta| delta_ij - d_i eta d_j eta / |grad eta|)
             = sigma |grad eta| (I - n (x) n)
which is the CONTINUUM SURFACE STRESS / capillary stress tensor
(Lafaurie et al. 1994; Schmidmayer et al. 2017 Eq. 3-4).  The KORTEWEG tensor
is -lambda grad(eta)(x)grad(eta) + isotropic, with NO division by |grad eta|
and lambda ~ sigma*epsilon.  They agree only in the sharp-interface limit.
Edit COLS below to relabel.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ physics --
R0, EPSI = 0.02, 0.02 / 6.0
CHI, RB, SBRK, SIGW = 14.56, 0.018, 7.28, 7.28
SIG0 = CHI * (R0 * R0 / (RB * RB) - 1.0)
LTRI, HALF, SMALL = 1.5 * R0, 2.0 * R0, 1.0e-12
FREF = SIG0 * (1.0 / R0) * (1.0 / (2.0 * EPSI))
VERTS = [(0.0, R0), (0.0, -R0), (LTRI, 0.0)]
RATIOS = [2, 4, 8, 16]

COLS = [                                   # (key, column title)
    ("geometry",  "Geometry"),
    ("exact",     "Exact $F_{sv}$"),
    ("filters",   "Capillary + Filters"),
    ("base",      "Capillary (BASE)"),
    ("brackbill", "Brackbill (old)"),
    ("kappa1",    r"$\kappa_1$ (old)"),
    ("newm",      "New Method"),
]


# --------------------------------------------- geometry: sdf + analytic kappa -
def sdf_kappa(X, Y):
    r = np.hypot(X, Y)
    INF = np.inf
    d_arc = np.where(X <= 0.0, np.abs(r - R0), INF)

    def seg(ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        wx, wy = X - ax, Y - ay
        t = (wx * vx + wy * vy) / (vx * vx + vy * vy)
        tc = np.clip(t, 0.0, 1.0)
        d = np.hypot(wx - tc * vx, wy - tc * vy)
        return np.where((t > 0.0) & (t < 1.0), d, INF)

    d_s1 = seg(0.0, -R0, LTRI, 0.0)
    d_s2 = seg(LTRI, 0.0, 0.0, R0)
    d_v = np.minimum.reduce([np.hypot(X - a, Y - b) for a, b in VERTS])
    d = np.minimum.reduce([d_arc, d_s1, d_s2, d_v])

    inside = ((X <= 0.0) & (r <= R0)) | (
        (X >= 0.0) & (X <= LTRI * (1.0 - np.abs(Y) / R0)) & (np.abs(Y) <= R0))

    kap = np.select(
        [d == d_v, d == d_arc],
        [np.where(inside, 0.0, 1.0 / np.maximum(d, 1e-30)),
         1.0 / np.maximum(r, 1e-30)],
        default=0.0)
    return np.where(inside, -d, d), kap


def build(N, dx):
    c = -HALF + dx * (np.arange(N) + 0.5)
    X, Y = np.meshgrid(c, c, indexing="ij")
    d, kap = sdf_kappa(X, Y)
    t = np.tanh(d / EPSI)
    eta = 0.5 * (1.0 + t)
    exact = -SIG0 * kap * (1.0 - t * t) / (2.0 * EPSI) / FREF
    return eta, exact


# ----------------------------------------------------- stencils (edge-clamped) -
def _p(a):
    return np.pad(a, 1, mode="edge")


def box3(a):
    S = _p(a)
    return (S[:-2, :-2] + S[:-2, 1:-1] + S[:-2, 2:] +
            S[1:-1, :-2] + S[1:-1, 1:-1] + S[1:-1, 2:] +
            S[2:, :-2] + S[2:, 1:-1] + S[2:, 2:]) / 9.0


def wbox3(a, w):
    sw, sf = box3(w) * 9.0, box3(a * w) * 9.0
    return np.where(sw > 1e-12, sf / np.where(sw > 0, sw, 1.0), a)


def derivs(a, dx):
    S = _p(a)
    gx = (S[2:, 1:-1] - S[:-2, 1:-1]) / (2 * dx)
    gy = (S[1:-1, 2:] - S[1:-1, :-2]) / (2 * dx)
    gxx = (S[2:, 1:-1] - 2 * a + S[:-2, 1:-1]) / dx ** 2
    gyy = (S[1:-1, 2:] - 2 * a + S[1:-1, :-2]) / dx ** 2
    gxy = (S[2:, 2:] - S[2:, :-2] - S[:-2, 2:] + S[:-2, :-2]) / (4 * dx ** 2)
    return gx, gy, gxx, gyy, gxy


def kappa1(a, dx):
    """kappa_method = 1:  kappa = -div(grad eta / |grad eta|)"""
    gx, gy, gxx, gyy, gxy = derivs(a, dx)
    gm = np.hypot(gx, gy)
    hgx = (gxx * gx + gxy * gy) / (gm + SMALL)
    hgy = (gxy * gx + gyy * gy) / (gm + SMALL)
    return -((gxx + gyy) / (gm + SMALL) - (gx * hgx + gy * hgy) / (gm ** 2 + SMALL))


def marm_sigma(kap, w):
    Rv = -1.0 / (kap + np.where(kap >= 0, SMALL, -SMALL))
    el = CHI * (Rv * Rv / (RB * RB) - 1.0)
    sig = np.where(Rv > RB, np.where(el >= SBRK, SIGW, el), 0.0)
    return np.where(w > 1e-12, sig, 0.0)


def div_omega(eta, sig, dx):
    gx, gy = derivs(eta, dx)[:2]
    gm = np.hypot(gx, gy)
    ok = gm >= 1e-10
    g = np.where(ok, gm, 1.0)
    oxx = np.where(ok, sig * (g - gx * gx / g), 0.0)
    oyy = np.where(ok, sig * (g - gy * gy / g), 0.0)
    oxy = np.where(ok, sig * (-gx * gy / g), 0.0)
    Pxx, Pyy, Pxy = _p(oxx), _p(oyy), _p(oxy)
    i2 = 0.5 / dx
    fx = (Pxx[2:, 1:-1] - Pxx[:-2, 1:-1]) * i2 + (Pxy[1:-1, 2:] - Pxy[1:-1, :-2]) * i2
    fy = (Pxy[2:, 1:-1] - Pxy[:-2, 1:-1]) * i2 + (Pyy[1:-1, 2:] - Pyy[1:-1, :-2]) * i2
    return np.where(ok, (fx * gx + fy * gy) / g, 0.0)


# ------------------------------------------------------------------- one panel -
def panel(key, ratio):
    dx = EPSI / ratio
    N = int(round(2 * HALF / dx))
    eta, exact = build(N, dx)
    w = eta * (1.0 - eta)

    if key == "geometry":
        return N, eta, "eta"
    if key == "exact":
        return N, exact, "div"
    if key == "newm":
        F = div_omega(eta, np.where(w > 1e-12, SIG0, 0.0), dx)
    elif key == "kappa1":
        F = div_omega(eta, marm_sigma(kappa1(eta, dx), w), dx)
    elif key == "brackbill":
        kap = kappa1(eta, dx)
        gx, gy = derivs(eta, dx)[:2]
        F = marm_sigma(kap, w) * kap * np.hypot(gx, gy)
    elif key == "base":                       # 6Eqn_BASE: eta smoothing only
        es = eta
        for _ in range(8):
            es = box3(es)
        F = div_omega(eta, marm_sigma(kappa1(es, dx), w), dx)
    elif key == "filters":                    # Marmottant branch: 8 / 8 / 4
        es = eta
        for _ in range(8):
            es = box3(es)
        kap = kappa1(es, dx)
        for _ in range(8):
            kap = wbox3(kap, w)
        sig = marm_sigma(kap, w)
        for _ in range(4):
            sig = wbox3(sig, w)
        F = div_omega(eta, sig, dx)
    return N, F / FREF, "div"


# ---------------------------------------------------------------------- figure -
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.7, "axes.edgecolor": "#b8bfc7",
})

nr, nc = len(RATIOS), len(COLS)
fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 0.9, 2.05 * nr + 0.55))

# fine grid for the exact eta=0.5 contour, drawn identically on every panel
cf = np.linspace(-HALF, HALF, 600)
Xf, Yf = np.meshgrid(cf, cf, indexing="ij")
Df = sdf_kappa(Xf, Yf)[0]

ext = [-HALF, HALF, -HALF, HALF]
im_div = None
print(f"{'method':<12}{'eps/dx':>7}{'N':>6}{'arc':>9}{'peak':>9}")
for ri, ratio in enumerate(RATIOS):
    for ci, (key, title) in enumerate(COLS):
        ax = axes[ri, ci]
        N, F, mode = panel(key, ratio)
        if mode == "eta":
            ax.imshow(F.T, origin="lower", extent=ext, cmap="Greys_r",
                      vmin=0, vmax=1, interpolation="nearest")
            print(f"{key:<12}{ratio:>7}{N:>6}{'--':>9}{'--':>9}")
        else:
            im_div = ax.imshow(F.T, origin="lower", extent=ext, cmap="RdBu_r",
                               vmin=-2, vmax=2, interpolation="nearest")
            dxr = EPSI / ratio
            c = -HALF + dxr * (np.arange(N) + 0.5)
            Xg, Yg = np.meshgrid(c, c, indexing="ij")
            m = (Xg < -0.15 * R0) & (np.abs(np.hypot(Xg, Yg) - R0) < 0.5 * EPSI)
            print(f"{key:<12}{ratio:>7}{N:>6}{F[m].mean():>9.3f}"
                  f"{np.abs(F).max():>9.2f}")
        # Xf/Yf are 'ij'-indexed, so they pair with Df -- NOT Df.T (which would
        # draw the outline transposed relative to the imshow'd field).
        ax.contour(Xf, Yf, Df, levels=[0.0], colors="0.35", linewidths=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-HALF, HALF); ax.set_ylim(-HALF, HALF)
        if ri == 0:
            ax.set_title(title, fontsize=11, pad=7)
        if ci == 0:
            ax.set_ylabel(f"{ratio} Cells", fontsize=11, rotation=0,
                          ha="right", va="center", labelpad=12)

fig.suptitle("Surface Curvature Methods", fontsize=15, y=0.985)
fig.subplots_adjust(left=0.058, right=0.925, top=0.895, bottom=0.02,
                    wspace=0.06, hspace=0.06)

cax = fig.add_axes([0.938, 0.10, 0.011, 0.72])
cb = fig.colorbar(im_div, cax=cax, ticks=[-2, -1, 0, 1, 2])
cb.set_label(r"$F\cdot\hat{n}$   (normalized; $-1$ = exact on the arc)",
             fontsize=10, labelpad=8)
cb.outline.set_linewidth(0.7)
cb.outline.set_edgecolor("#b8bfc7")

import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "marmottant_curvature_figure.png")
fig.savefig(out, dpi=200, facecolor="white")
print("\nwrote", out)
