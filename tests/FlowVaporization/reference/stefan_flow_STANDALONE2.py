import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.optimize import fsolve
import matplotlib as mpl

# -----------------------------
# Physical parameters
# -----------------------------
D_v = 2e-1
rho_liq = 10.0
rho_gas = 1.0

Y_s = 0.1
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
times = np.linspace(0.01, 20.0, 400)

# ============================================================
# PLOT 1: Mass fraction profiles (unchanged)
# ============================================================

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

# ============================================================
# PLOT 2: Interface position vs time (NEW)
# ============================================================

# Compute interface position history
s_history = 2.0 * lam * np.sqrt(D_v * times)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(times, s_history, 'b-', linewidth=2)

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Interface Position s(t) (m)")
ax2.set_title("Interface Motion: s(t) = 2*lambda*sqrt(D t)")
ax2.grid(True)

plt.tight_layout()
plt.show()
