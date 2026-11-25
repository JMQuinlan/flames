#!/usr/bin/env python3
"""
Convergence summary of the 2D cylinder collapse matrix (after osc2d_matrix.py has written the dynamics.npz caches).

  python3 osc2d_convergence.py <Osc2D dir> <prefix> --out FIG.png [--coated] [--schemes weno3,weno5,...]

One column per pressure ratio; rows: minimum volumetric radius, first rebound maximum, largest departure from round
(rms(r)/<r> of the eta = 0.5 contour over the run), and p V^gamma of the gas at the minimum over its initial value
(1 = the gas stayed on its adiabat; p = volume mean of the gas pressure from thermo.dat), each against cells per initial radius.  One line per scheme
(colour) and alpha row (solid marker line = donor, dashed = consistent).  Horizontal line: cylindrical
Rayleigh-Plesset (guide only: the runs have a non-reflecting outer boundary and a compressible liquid).
"""
import os, re, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import osc2d_dynamics as D
from osc2d_matrix import NAME, state
R0 = 2.0e-6


def main():
    args = sys.argv[1:]; opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    base, prefix = args[0], args[1]; out = opt("--out", "osc2d_convergence.png"); coated = "--coated" in args
    R = {}
    for d in sorted(glob.glob(os.path.join(base, prefix + "_pr*_L*_*"))):
        g = re.match(prefix + r"_pr(\d+)_L(\d)_(\w+)_([dc])$", os.path.basename(d)); c = os.path.join(d, "dynamics.npz")
        if not (g and os.path.exists(c)) or state(d) != "finished": continue
        z = np.load(c); t, Rv, sh = z["t"], z["Rv"], z["sh"]; i = int(np.argmin(Rv)); k = i + int(np.argmax(Rv[i:]))
        a = np.loadtxt(os.path.join(d, "output", "thermo.dat"), skiprows=1); K = a[:, 2] / a[:, 1] * a[:, 1] ** 1.4; ia = int(np.argmin(a[:, 1]))
        R[(int(g.group(1)), int(g.group(2)), g.group(3), g.group(4))] = (Rv[i], Rv[k] if t[-1] > 7.9e-7 else np.nan, 100 * sh[:, 2].max() if len(sh) else np.nan, K[ia] / K[0])
    prs = sorted({k[0] for k in R}); want = opt("--schemes", ""); schemes = [s for s in NAME if any(k[2] == s for k in R) and (not want or s in want.split(","))]
    fig, ax = plt.subplots(4, len(prs), figsize=(4.6 * len(prs), 14.0), squeeze=False)
    for q, pr in enumerate(prs):
        ref = D.reference(1.0e5 / pr, coated, 8.0e-7)
        for j, s in enumerate(schemes):
            for r, ls, mk in (("d", "-", "o"), ("c", "--", "s")):
                Ls = sorted(k[1] for k in R if k[0] == pr and k[2] == s and k[3] == r)
                if not Ls: continue
                x = [12.8 * 2 ** L for L in Ls]
                for row in range(4):
                    ax[row, q].plot(x, [R[(pr, L, s, r)][row] for L in Ls], ls + mk, color=f"C{j}", ms=4.5, mfc="none", mew=0.9, lw=0.9,
                                    label=NAME[s] + (", Donor" if r == "d" else ", Consistent"))
        if ref is not None:
            i = int(np.argmin(ref[1])); k = i + int(np.argmax(ref[1][i:]))
            ax[0, q].axhline(ref[1][i] / R0, color="k", lw=1.3, label="Cylindrical Rayleigh-Plesset (Guide)"); ax[1, q].axhline(ref[1][k] / R0, color="k", lw=1.3)
        ax[0, q].set_title("Pressure Ratio %d\nMinimum Radius" % pr); ax[1, q].set_title("First Rebound Maximum"); ax[2, q].set_title("Largest Departure From Round")
        ax[0, q].set_ylabel(r"$R_{vol,min}/R_0$ [-]"); ax[1, q].set_ylabel(r"$R_{vol}/R_0$ at Rebound [-]"); ax[2, q].set_ylabel(r"max rms($r$)/$\langle r \rangle$ [%]"); ax[2, q].set_yscale("log")
        ax[3, q].set_title("Gas Against Its Adiabat at the Minimum"); ax[3, q].set_ylabel(r"$p V^{\gamma}$ at Minimum / Initial [-]"); ax[3, q].axhline(1.0, color="k", lw=1.3)
        for row in range(4):
            ax[row, q].set_xscale("log"); ax[row, q].set_xticks([12.8, 25.6, 51.2, 102.4]); ax[row, q].set_xticklabels(["12.8", "25.6", "51.2", "102.4"]); ax[row, q].minorticks_off()
            ax[row, q].set_xlim(10, 130); ax[row, q].set_xlabel("Cells per Initial Radius [-]"); ax[row, q].grid(alpha=0.3)
    ax[2, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("%s Cylinder Collapse Unit Test (Numerical): Convergence by Scheme and Pressure Ratio" % ("Coated" if coated else "Uncoated")); fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out, dpi=170); plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    main()
