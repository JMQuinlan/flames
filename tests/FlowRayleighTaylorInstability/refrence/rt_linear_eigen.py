"""
Exact linear growth rate of the RT deck's density profile (incompressible,
inviscid, continuously stratified; Chandrasekhar 1961, Ch. X):
    D(rho D w) - k^2 rho w = -(g k^2 / n^2) (D rho) w,   w = 0 at y = 0, L_y
for rho(y) = 1 + 0.5 (1 + tanh((y - 0.75)/eps)).  Solved as the symmetric
generalized eigenproblem  B w = (n^2/(g k^2)) L w  (L = -D(rho D) + k^2 rho,
B = diag(D rho)) on a fine grid.  eps -> 0 recovers sqrt(A g k coth-form).
Usage: python3 rt_linear_eigen.py
"""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import diags

g, LX, LY, Y0 = 0.1, 0.5, 1.5, 0.75
k = 2 * np.pi / LX
A = 1.0 / 3.0


def growth(eps, n=6000):
    # cluster points near the interface: uniform grid fine enough
    y = np.linspace(0, LY, n + 2)[1:-1]; h = y[1] - y[0]
    yf = np.concatenate(([y[0] - h / 2], y + h / 2))
    rho = lambda z: 1.0 + 0.5 * (1 + np.tanh((z - Y0) / eps)) if eps > 0 else 1.0 + (z > Y0)
    rf = rho(yf)
    main = (rf[:-1] + rf[1:]) / h**2 + k**2 * rho(y)
    off = -rf[1:-1] / h**2
    Lm = diags([off, main, off], [-1, 0, 1], format="csc")
    drho = (rf[1:] - rf[:-1]) / h
    Bm = diags([drho], [0], format="csc")
    from scipy.sparse.linalg import eigsh
    w = eigsh(Bm, k=1, M=Lm, which="LA", return_eigenvectors=False)[0]
    return np.sqrt(w * g * k**2)


if __name__ == "__main__":
    s0 = np.sqrt(A * g * k)
    hfac = np.tanh(k * Y0)
    print(f"sharp, infinite depth sqrt(Agk) = {s0:.5f}")
    for eps in (0.004, 0.002, 0.001):
        for n in (6000, 24000):
            print(f"eps={eps}: n={n}  sigma = {growth(eps, n):.5f}  ({100*(growth(eps, n)/s0-1):+.3f} % vs sqrt(Agk))")
