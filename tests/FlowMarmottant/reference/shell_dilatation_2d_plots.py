#!/usr/bin/env python3
"""
Comparison figures for the 2D coated-bubble component study (input_osc2d_unit), from the shell_dilatation.npz files
written by shell_dilatation_2d.py.

  python3 shell_dilatation_2d_plots.py --out FIG.png  "Label=run_dir" ["Label=run_dir" ...]

The run directory name must contain _L<level>_ (cells per initial radius = 12.8 x 2^level).
RECOVERED = (dilatation the shell stress used, D_stored) / (dilatation of the bubble, d ln R_vol / dt); it is the
fraction of the shell viscosity kappa_s that acts.  Time means are taken where |d ln R_vol/dt| > half its maximum.
Model line: for a round bubble with u_r/r uniform in the gas and (R/r)^2 x the wall value in the liquid, the
|grad eta|-weighted mean over a tanh interface of thickness eps is  1 - ln(2) eps / R  (2D).
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R0 = 2.0e-6
MARK = "os^Dv<>ph*"


def main():
    args = sys.argv[1:]; out = args[args.index("--out") + 1]; runs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--")]
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.2)); rows = []
    for j, (lab, d) in enumerate(runs):
        z = np.load(os.path.join(d, "shell_dilatation.npz")); t = z["t"]; Rv = z["Rv"]; L = int(re.search(r"_L(\d)_", d).group(1)); dx = R0 / (12.8 * 2 ** L)
        Dvol = np.gradient(np.log(Rv), t); Dsa = z["Dstored"]
        if not np.nanmax(np.abs(Dsa)) > 0: Dsa = z["Dband"]             # sub-cycled shell viscosity: not in the stored tension, use the rebuilt P : grad u
        Ds = np.nanmean(Dsa, axis=1); Dk = np.nanmean(z["Dkin"], axis=1); Df = np.nanmean(z["Dfield"], axis=1)
        r = z["r"]; rm = np.nanmean(r, axis=1); rnd = np.nanstd(r, axis=1) / rm; kR = z["kap"] * rm[:, None]; w = np.nanmean(z["width"], axis=1)
        big = np.abs(Dvol) > 0.5 * np.abs(Dvol).max(); rec = Ds / Dvol; eps_R = w * dx / 3.892 / Rv            # 0.02 < eta < 0.98 spans 3.892 eps
        c = f"C{j % 10}"; m = MARK[j % len(MARK)]; kw = dict(ms=3.5, mfc="none", mec=c, mew=0.7, color=c, lw=0.6, markevery=max(1, len(t) // 40))
        ax[0, 0].plot(1e9 * t, Rv / R0, "-" + m, label=lab, **kw)
        ax[0, 1].plot(1e9 * t[big], rec[big], m, label=lab, ms=3.5, mfc="none", mec=c, mew=0.7)
        ax[0, 2].plot(np.mean(eps_R[big]), np.mean(rec[big]), m, ms=8, mfc="none", mec=c, mew=1.2, label=lab)
        ax[1, 0].plot(1e9 * t, w, "-" + m, label=lab, **kw)
        ax[1, 1].plot(1e9 * t, 100 * rnd, "-" + m, label=lab, **kw)
        # per-angle recovered against curvature x R (time-mean over the fast part)
        reca = Dsa[big] / Dvol[big, None]
        ax[1, 2].plot(np.nanmean(kR[big], axis=0), np.nanmean(reca, axis=0), m, ms=4, mfc="none", mec=c, mew=0.7, label=lab)
        rows.append((lab, 12.8 * 2 ** L, Rv.min() / R0, 1e9 * t[np.argmin(Rv)], np.mean(w[big]), np.mean(eps_R[big]), np.mean(rec[big]), (Ds[big] * Dk[big]).sum() / (Dk[big] ** 2).sum(),
                     (Df[big] * Dvol[big]).sum() / (Dvol[big] ** 2).sum(), 100 * np.nanmax(rnd), np.nanmin(kR[big]), np.nanmax(kR[big])))
    e = np.linspace(0, 0.2, 50); ax[0, 2].plot(e, 1 - np.log(2) * e, "k-", lw=1.2, label=r"Ideal Round Tanh Interface, $1 - \ln 2\,\epsilon/R$"); ax[0, 2].plot(e, 1 - e, "k--", lw=1.0, label=r"$1 - \epsilon/R$")
    ax[0, 0].set_xlabel("Time [ns]"); ax[0, 0].set_ylabel(r"Volumetric Radius $R_{vol}/R_0$ [-]"); ax[0, 0].set_title("Radius")
    ax[0, 1].set_xlabel("Time [ns]"); ax[0, 1].set_ylabel("Recovered Shell Viscosity [-]"); ax[0, 1].set_title("Recovered Shell Viscosity in Time"); ax[0, 1].axhline(1, color="k", lw=1)
    ax[0, 2].set_xlabel(r"Interface Thickness / Radius, $\epsilon/R$ [-]"); ax[0, 2].set_ylabel("Recovered Shell Viscosity [-]"); ax[0, 2].set_title("Recovered Shell Viscosity Against Interface Thickness")
    ax[1, 0].set_xlabel("Time [ns]"); ax[1, 0].set_ylabel(r"Interface Width, 0.02 < $\eta$ < 0.98 [cells]"); ax[1, 0].set_title("Interface Width")
    ax[1, 1].set_xlabel("Time [ns]"); ax[1, 1].set_ylabel(r"Departure From Round, rms($r$)/$\langle r \rangle$ [%]"); ax[1, 1].set_title("Surface Deformation")
    ax[1, 2].set_xlabel(r"Curvature $\times$ Mean Radius [-]"); ax[1, 2].set_ylabel("Recovered Shell Viscosity [-]"); ax[1, 2].set_title("Recovered Shell Viscosity Against Local Curvature")
    for a in ax.ravel(): a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=6.5, ncol=2); ax[0, 2].legend(fontsize=6, ncol=1, loc="lower left"); ax[0, 1].set_ylim(0.6, 1.1); ax[1, 2].set_ylim(0.7, 1.1)
    fig.suptitle("Coated Cylinder Unit Test (Numerical): Surface Dilatation Seen by the Shell Against the Dilatation of the Bubble"); fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out, dpi=200); plt.close(fig)
    print("%-28s R0/dx  Rmin/R0  t_min[ns]  width[cells]  eps/R   recovered  model   stored/contour  field@contour/vol  round max[%%]  kappa R min..max" % "run")
    for r in rows:
        print("%-28s %5.1f  %.4f   %5.0f      %5.1f      %.3f    %.3f     %.3f      %.3f          %.3f            %.2f       %.2f .. %.2f" % (r[:6] + (r[6], 1 - np.log(2) * r[5]) + r[7:]))
    print("wrote", out)


if __name__ == "__main__":
    main()
