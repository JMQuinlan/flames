#!/usr/bin/env python3
"""
Constant-volume burn of the idealised explosive (tests/FlowEnergetic/input_0D_Burn) against its closed forms.

  python3 burn_check.py <run_dir> [--k 2e6] [--Q 3.68e6]

lambda(t) = (1 - k t/2)^2 until t = 2/k;  rho E(t) = rho E(0) + Q m_r(0) (1 - lambda);  the box ends as products
at p = (gamma_p - 1) rho E.  Prints the unreacted fraction, the total mass, the energy and the pressure in time
with their errors, and writes burn_check.png next to the script's Images directory.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 14, "legend.fontsize": 11})
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]; opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
    d = args[0]; k = opt("--k", 2.0e6); Q = opt("--Q", 3.68e6)
    rows = []
    for pf in sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4])):
        ds = yt.load(pf); ad = ds.all_data(); g = lambda n: np.asarray(ad[("boxlib", n)]).mean()
        rows.append((float(ds.current_time), g("species_Reactant"), g("species_Products"), g("rho_eta0") + g("rho_eta1"), g("energy_per_vol"), g("pressure"), g("eta")))
    r = np.array(rows); r = r[np.r_[True, np.diff(r[:, 0]) > 0]]; t, mr, mp, m, E, p, eta = r.T
    lam = mr / (mr + mp - mp[0]); lex = np.where(t < 2 / k, (1 - k * t / 2) ** 2, 0.0); Eex = E[0] + Q * mr[0] * (1 - lex)
    print("   t [us]   lambda   exact     eta        mass/mass0-1    rho E / exact - 1    p [Pa]")
    for i in range(0, len(t), max(1, len(t) // 16)):
        print("  %7.3f   %.5f  %.5f   %.3e   %+.2e        %+.2e         %.5e" % (1e6 * t[i], lam[i], lex[i], eta[i], m[i] / m[0] - 1, E[i] / Eex[i] - 1, p[i]))
    pend = 2.0 * Eex[-1]
    print("max |lambda - exact| %.2e ; mass drift %.2e ; energy error %.2e ; end pressure %.5e against %.5e (%+.3f %%)"
          % (np.abs(lam - lex).max(), np.abs(m / m[0] - 1).max(), np.abs(E / Eex - 1).max(), p[-1], pend, 100 * (p[-1] / pend - 1)))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5)); tt = np.linspace(0, t[-1], 400); le = np.where(tt < 2 / k, (1 - k * tt / 2) ** 2, 0.0)
    ax[0].plot(1e6 * tt, le, "k-", lw=1.4, label="Exact"); ax[0].plot(1e6 * t, lam, "o", ms=4, mfc="none", mec="C0", label="Numerical")
    ax[0].set_xlabel(r"Time [$\mu$s]"); ax[0].set_ylabel("Unreacted Mass Fraction [-]"); ax[0].set_title("Reaction Progress"); ax[0].legend()
    ax[1].plot(1e6 * tt, 2.0 * (E[0] + Q * mr[0] * (1 - le)) / 1e9, "k--", lw=1.0, label=r"($\gamma_p - 1$) $\rho E$, Exact Energy")
    ax[1].axhline(pend / 1e9, color="k", lw=1.4, label="Exact End Pressure"); ax[1].plot(1e6 * t, p / 1e9, "o", ms=4, mfc="none", mec="C3", label="Numerical")
    ax[1].set_xlabel(r"Time [$\mu$s]"); ax[1].set_ylabel("Pressure [GPa]"); ax[1].set_title("Pressure"); ax[1].legend(fontsize=11)
    for a in ax: a.grid(alpha=0.3)
    fig.suptitle("Constant-Volume Burn of the Idealised Explosive (Petitpas et al. 2009)"); fig.tight_layout()
    o = os.path.join(HERE, "Images", "Burn0D"); os.makedirs(o, exist_ok=True); fig.savefig(os.path.join(o, "burn_check.png"), dpi=200); print("wrote", os.path.join(o, "burn_check.png"))


if __name__ == "__main__":
    main()
