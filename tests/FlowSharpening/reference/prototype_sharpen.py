#!/usr/bin/env python3
"""PROTOTYPE (no solver change): validate the CORRECT conservative interface
sharpening on the real test eta field, to de-risk the C++ rewrite.

Method = conservative Olsson-Kreiss interface compression, applied to the VOLUME
FRACTION eta directly (which is what the 6-eqn model evolves), in pseudo-time tau:

    d(eta)/d(tau) = div[ eps*(grad eta . nhat) nhat  -  eta(1-eta) nhat ]
    nhat = grad(eta)/|grad(eta)|   (FROZEN at tau=0)

  - pure DIVERGENCE form  -> integral(eta) conserved to machine precision
  - balances normal diffusion (eps) vs compression -> steady tanh of thickness ~eps
  - 10-90% width ~= 4.4*eps, so eps in [0.9, 2.7]*dx gives a 4-12 cell band
  - sharpens a STATIC interface (it's a relaxation to target thickness), so unlike
    Tiwari it IS validated by the quiescent test.

This proves the method + picks eps for the 4-12 cell target.  Run on the test IC.

Usage: python prototype_sharpen.py <plotfile_cell_dir> [eps_over_dx] [n_iter]
"""
import sys
import numpy as np
import yt
yt.funcs.mylog.setLevel(40)

PF = sys.argv[1]
EPS_OVER_DX = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
N_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 200

ds = yt.load(PF)
nx, ny = int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1])
dx = float(ds.domain_width[0]) / nx
cg = ds.covering_grid(0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
eta = np.array(cg["eta"])[:, :, 0].astype(float)        # 2D (uniform test grid)
eps = EPS_OVER_DX * dx
dtau = 0.20 * dx                                          # pseudo-time step (stable)

def thickness_cells(e):
    prof = e.mean(axis=1)                                 # x-profile (avg over y)
    xs = (np.arange(len(prof)) + 0.5) * dx
    def xc(l):
        for i in range(len(prof) - 1):
            if (prof[i] - l) * (prof[i + 1] - l) < 0:
                return xs[i] + (l - prof[i]) * (xs[i + 1] - xs[i]) / (prof[i + 1] - prof[i])
        return np.nan
    a, b = xc(0.1), xc(0.9)
    return abs(b - a) / dx if np.isfinite(a) and np.isfinite(b) else np.nan

def grad(e):
    gx = np.zeros_like(e); gy = np.zeros_like(e)
    gx[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2 * dx)
    gy[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2 * dx)
    return gx, gy

# frozen normal
gx0, gy0 = grad(eta)
mag = np.sqrt(gx0**2 + gy0**2) + 1e-30
nx_, ny_ = gx0 / mag, gy0 / mag

M0 = eta.sum() * dx * dx
print(f"grid {nx}x{ny} dx={dx:.4e}  eps={eps:.4e} ({EPS_OVER_DX:.2f} dx)  dtau={dtau:.3e}")
print(f"{'iter':>6} {'thick_cells':>11} {'dInt(eta)%':>11} {'min_eta':>9} {'max_eta':>9}")
print(f"{0:>6} {thickness_cells(eta):>11.2f} {0.0:>+11.2e} {eta.min():>9.4f} {eta.max():>9.4f}")

for it in range(1, N_ITER + 1):
    gx, gy = grad(eta)
    ndg = gx * nx_ + gy * ny_                             # grad(eta).nhat
    # flux F = eps*(grad eta . nhat) nhat - eta(1-eta) nhat   (vector)
    Fx = eps * ndg * nx_ - eta * (1 - eta) * nx_
    Fy = eps * ndg * ny_ - eta * (1 - eta) * ny_
    # conservative divergence via face averages (no-flux at x-boundaries)
    Fx_face = 0.5 * (Fx[1:, :] + Fx[:-1, :])             # at i+1/2  (nx-1 faces)
    Fy_face = 0.5 * (Fy[:, 1:] + Fy[:, :-1])
    div = np.zeros_like(eta)
    div[1:-1, :] += (Fx_face[1:, :] - Fx_face[:-1, :]) / dx
    div[:, 1:-1] += (Fy_face[:, 1:] - Fy_face[:, :-1]) / dx
    eta = np.clip(eta + dtau * div, 0.0, 1.0)
    if it % max(1, N_ITER // 10) == 0 or it == N_ITER:
        M = eta.sum() * dx * dx
        print(f"{it:>6} {thickness_cells(eta):>11.2f} {100*(M-M0)/M0:>+11.2e} "
              f"{eta.min():>9.4f} {eta.max():>9.4f}")

print("\nVERDICT: if thickness drops to ~4.4*eps/dx and dInt(eta) stays ~0, the")
print("conservative Olsson compression on eta is the correct, conservative fix.")
print(f"  target thickness for eps={EPS_OVER_DX:.1f}dx is ~{4.4*EPS_OVER_DX:.1f} cells")
