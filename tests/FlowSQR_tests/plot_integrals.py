import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Usage: python plot_integrals.py <folder>")
    sys.exit(1)

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, sys.argv[1])
dat_path = os.path.join(folder, "integrals.dat")

data = np.loadtxt(dat_path, comments="#")

time    = data[:, 0]
M_total = data[:, 1]
M_gas   = data[:, 2]
M_liq   = data[:, 3]

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(time, M_total)
axes[0].set_ylabel("M_total")

axes[1].plot(time, M_liq)
axes[1].set_ylabel("M_liquid")

axes[2].plot(time, M_gas)
axes[2].set_ylabel("M_gas")
axes[2].set_xlabel("Time")

fig.tight_layout()
fig.savefig(os.path.join(folder, "integrals.png"), dpi=150)
print(f"Saved to {os.path.join(folder, 'integrals.png')}")
