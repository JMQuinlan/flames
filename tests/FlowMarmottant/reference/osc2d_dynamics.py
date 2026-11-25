#!/usr/bin/env python3
"""
Dynamics comparison of 2D quarter-cylinder bubble runs (input_osc2d_unit and its uncoated / stronger-drive variants).

  python3 osc2d_dynamics.py --out FIG.png --pb 1.0e4 [--coated] [--title "..."]  "Label=run_dir" ...

Per run: volumetric radius R_vol(t) (thermo.dat gas volume of the quarter, R = sqrt(4 V / pi)), and from the plot
files the mean radius of the eta = 0.5 contour, its departure from round rms(r)/<r>, and the axis-to-diagonal
ratio r(0 deg)/r(45 deg) (below 1: flattened or jetted along the axes; above 1: along the diagonals).
Reference: cylindrical Rayleigh-Plesset between R and a constant-pressure circle of the same area as the unfolded
domain (R_inf = L 2/sqrt(pi)); a guide only, the outer boundary of the runs is a non-reflecting outflow.
The table gives minimum radius, its time and the first rebound maximum, of R_vol.
"""
import os, sys, glob, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.funcs.mylog.setLevel(50)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shell_dilatation_2d as S
R0 = 2.0e-6
MARK = "os^Dv<>ph*"


def reference(pb, coated, tend, L=1.0e-5):
    p = "/home/ttryon/Desktop/flames2/bin/ShellRework/scripts/rp2d.py"
    if not os.path.exists(p): return None
    sp = importlib.util.spec_from_file_location("rp2d", p); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    m.pb = pb; s = m.solve("full", tend, L * 2 / np.sqrt(np.pi)) if coated else m.solve("viscous", tend, L * 2 / np.sqrt(np.pi), 0.0)
    t = np.linspace(0, tend, 4001); return t, s.sol(t)[0]


def shape(d, every=1):
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[::every]
    th = (np.arange(96) + 0.5) * (np.pi / 2) / 96; out = []
    for pf in pfs:
        try: f = S.load(pf)
        except Exception: continue
        r = S.contour(f, th)
        if np.isfinite(r).sum() < 48: continue
        out.append((f["t"], np.nanmean(r), np.nanstd(r) / np.nanmean(r), np.nanmean(np.r_[r[:6], r[-6:]]) / np.nanmean(r[42:54])))
    return np.array(out)


def main():
    args = sys.argv[1:]; opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    out = opt("--out", "osc2d_dynamics.png"); pb = float(opt("--pb", "8.2e4")); coated = "--coated" in args
    runs = [a.split("=", 1) for a in args if "=" in a and not a.startswith("--") and os.path.isdir(a.split("=", 1)[1])]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.0)); tend = 0.0; rows = []
    for j, (lab, d) in enumerate(runs):
        a = np.loadtxt(os.path.join(d, "output", "thermo.dat"), skiprows=1); t = a[:, 0]; Rv = np.sqrt(4 * a[:, 1] / np.pi) / R0; tend = max(tend, t[-1])
        c = f"C{j % 10}"; m = MARK[j % len(MARK)]; kw = dict(ms=3.5, mfc="none", mec=c, mew=0.7, color=c, lw=0.6)
        ax[0].plot(1e9 * t, Rv, "-" + m, markevery=max(1, len(t) // 35), label=lab, **kw)
        i = int(np.argmin(Rv)); k = i + int(np.argmax(Rv[i:])); sh = shape(d)
        if len(sh):
            ax[1].plot(1e9 * sh[:, 0], 100 * sh[:, 2], "-" + m, label=lab, **kw); ax[2].plot(1e9 * sh[:, 0], sh[:, 3], "-" + m, label=lab, **kw)
            q = int(np.argmin(np.abs(sh[:, 0] - t[i]))); rows.append((lab, Rv[i], 1e9 * t[i], Rv[k], 1e9 * t[k], sh[q, 1] / R0, 100 * sh[:q + 1, 2].max(), 100 * sh[:, 2].max(), sh[q, 3], t[-1] * 1e9))
        else: rows.append((lab, Rv[i], 1e9 * t[i], Rv[k], 1e9 * t[k], np.nan, np.nan, np.nan, np.nan, t[-1] * 1e9))
    ref = reference(pb, coated, tend)
    if ref is not None:
        ax[0].plot(1e9 * ref[0], ref[1] / R0, "k-", lw=1.5, label="Cylindrical Rayleigh-Plesset (Guide)", zorder=0)
        i = int(np.argmin(ref[1])); k = i + int(np.argmax(ref[1][i:])); rows.insert(0, ("Cylindrical Rayleigh-Plesset", ref[1][i] / R0, 1e9 * ref[0][i], ref[1][k] / R0, 1e9 * ref[0][k], np.nan, np.nan, np.nan, np.nan, 1e9 * tend))
    ax[0].set_xlabel("Time [ns]"); ax[0].set_ylabel(r"Volumetric Radius $R_{vol}/R_0$ [-]"); ax[0].set_title("Volumetric Radius"); ax[0].legend(fontsize=6.5, ncol=2)
    ax[1].set_xlabel("Time [ns]"); ax[1].set_ylabel(r"Departure From Round, rms($r$)/$\langle r \rangle$ [%]"); ax[1].set_title(r"Surface Deformation of the $\eta$ = 0.5 Contour"); ax[1].set_yscale("log")
    ax[2].set_xlabel("Time [ns]"); ax[2].set_ylabel("Axis Radius / Diagonal Radius [-]"); ax[2].set_title("Axis-to-Diagonal Radius Ratio"); ax[2].axhline(1, color="k", lw=1)
    for a_ in ax: a_.grid(alpha=0.3)
    fig.suptitle(opt("--title", "Cylinder Collapse Unit Test (Numerical)")); fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True); fig.savefig(out, dpi=200); plt.close(fig)
    print("%-34s Rvol min  t_min[ns]  rebound  t_reb[ns]  R(eta=.5) at min  round max to min[%%]  round max[%%]  axis/diag at min   end[ns]" % "run")
    for r in rows: print("%-34s %.4f    %6.1f    %.4f   %6.1f       %.4f            %6.2f           %6.2f        %.3f         %5.0f" % r)
    print("wrote", out)


if __name__ == "__main__":
    main()
