import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import fsolve
import matplotlib as mpl

# -----------------------------
# Physical parameters
# -----------------------------
D_v = 2e-5
rho_liq = 100.0
rho_gas = 1.2

Y_s = 0.01
Y_inf = 0.0

# -----------------------------
# Solve for lambda
# -----------------------------
A = (rho_gas / rho_liq) * (Y_s - Y_inf) / (1.0 - Y_s)

def lambda_equation(lam):
    return lam * np.exp(lam**2) * erfc(lam) - A

lam_initial_guess = 0.01
lam = fsolve(lambda_equation, lam_initial_guess)[0]

print("Lambda =", lam)

# -----------------------------
# Domain and time
# -----------------------------
x = np.linspace(0, 0.04, 500)
times = np.linspace(0.01, 2.0, 400)

fig, ax = plt.subplots(figsize=(8, 5))

norm = mpl.colors.Normalize(vmin=times.min(), vmax=times.max())
cmap = plt.cm.viridis

for t in times:
    sqrtDt = np.sqrt(D_v * t)
    s = 2.0 * lam * sqrtDt

    eta = (x - s) / (2.0 * sqrtDt)

    Y = Y_inf + (Y_s - Y_inf) * erfc(eta) / erfc(lam)
    Y = np.where(x > s, Y, Y_s)

    ax.plot(x, Y, color=cmap(norm(t)))

sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax).set_label("Time (s)")

ax.set_xlabel("Distance (m)")
ax.set_ylabel("Vapor Mass Fraction")
ax.set_title("Transient 1D Stefan Evaporation")
ax.set_ylim(0, Y_s * 1.05)
ax.grid(True)

plt.tight_layout()
plt.show()
