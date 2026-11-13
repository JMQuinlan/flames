#!/usr/bin/env python3
"""
Animation (GIF) and a multi-time still of the hydrogen-oxygen detonation started from its ZND profile
(tests/FlowChemistry/input_1D_H2O2Detonation_ZND): the wave travelling down the tube and the mixture in the
reaction zone behind the front, against the steady ZND solution (Cantera, same mechanism).

  python3 detonation_movie.py <run_dir> [--outdir DIR] [--x0 0.3] [--zoom 25] [--every 1]

Upper row: pressure and temperature along the whole tube.  Lower row, in the frame of the front (distance behind
the front, mm): pressure and temperature, the major species (H2, O2, H2O) and the radicals (H, O, OH).
Reference = lines, Numerical = open markers.  Row 0 of the two-row strip is plotted.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import yt
yt.funcs.mylog.setLevel(50)
HERE = os.path.dirname(os.path.abspath(__file__))
ZND = os.path.join(HERE, "znd_2H2_O2_7Ar_6670Pa_298K.dat")
COLS = ["s", "rho", "u", "p", "H", "H2", "O", "OH", "H2O", "O2", "HO2", "H2O2", "N2", "AR", "HE", "CO", "CO2"]
W = dict(H=1.008, H2=2.016, O=15.999, OH=17.007, H2O=18.015, O2=31.998, HO2=33.006, H2O2=34.014, N2=28.014, AR=39.948, HE=4.0026, CO=28.010, CO2=44.009)
MAJOR = [("H2", "C0", r"H$_2$"), ("O2", "C3", r"O$_2$"), ("H2O", "C2", r"H$_2$O")]
RAD = [("H", "C4", "H"), ("O", "C1", "O"), ("OH", "C5", "OH")]


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def load(pf):
    ds = yt.load(pf); cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    q = lambda n: np.asarray(cg[("boxlib", n)])[:, 0, 0]
    n = int(ds.domain_dimensions[0]); L = float(ds.domain_right_edge[0]); x = (np.arange(n) + 0.5) * L / n
    d = dict(t=float(ds.current_time), x=x, p=q("pressure"), T=q("T"), L=L)
    for s, _, _ in MAJOR + RAD: d[s] = q("Y_" + s)
    k = np.where(d["p"] > 0.5 * d["p"].max())[0].max()
    xf = x[k] + (x[k + 1] - x[k]) * (0.5 * d["p"].max() - d["p"][k]) / (d["p"][k + 1] - d["p"][k]) if k + 1 < n else x[k]
    d["xf"] = xf
    return d


def main():
    args = sys.argv[1:]; flags = ("--outdir", "--x0", "--zoom", "--every")
    skip = {args.index(k) + 1 for k in flags if k in args}
    d = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip][0]
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "H2O2_Detonation_Movie")); os.makedirs(outdir, exist_ok=True)
    zoom = float(opt(args, "--zoom", "25")); every = int(opt(args, "--every", "1"))
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))[::every]
    F = [load(pf) for pf in pfs]
    if len(F) > 1 and abs(F[1]["t"] - F[0]["t"]) < 1e-12: F = F[1:]
    Z = np.loadtxt(ZND); z = {c: Z[:, i] for i, c in enumerate(COLS)}
    Wm = 1.0 / sum(z[s] / W[s] for s in W); z["T"] = z["p"] * Wm * 1e-3 / (8.314462618 * z["rho"])          # ideal gas, K
    zs = 1e3 * z["s"]; D = float(open(ZND).readline().split("D_CJ =")[1].split()[0])
    pmax, Tmax, L = 1.08 * z["p"].max() / 1e3, 1.08 * max(z["T"].max(), max(f["T"].max() for f in F)), F[0]["L"]
    cells = (1.521e-3) / (F[0]["x"][1] - F[0]["x"][0])

    def draw(fig, ax, f):
        for a in ax.ravel(): a.cla()
        a = ax[0, 0]; a.plot(f["x"], f["p"] / 1e3, "k-", lw=0.9); a.set_xlim(0, L); a.set_ylim(0, pmax)
        a.set_xlabel("Position [m]"); a.set_ylabel("Pressure [kPa]"); a.set_title("Pressure Along the Tube")
        a.axvline(f["xf"], color="0.6", lw=0.6, ls=":")
        a = ax[0, 1]; a.plot(f["x"], f["T"], "k-", lw=0.9); a.set_xlim(0, L); a.set_ylim(0, Tmax)
        a.set_xlabel("Position [m]"); a.set_ylabel("Temperature [K]"); a.set_title("Temperature Along the Tube")
        a.axvline(f["xf"], color="0.6", lw=0.6, ls=":")
        s = 1e3 * (f["xf"] - f["x"]); m = (s > -0.08 * zoom) & (s < zoom); st = max(1, int(m.sum() // 160))
        a = ax[1, 0]; a.plot(zs, z["p"] / 1e3, "C3-", lw=1.4, label="Pressure, ZND (Cantera)")
        a.plot(s[m][::st], f["p"][m][::st] / 1e3, "o", ms=3.2, mfc="none", mec="C3", mew=0.7, label="Pressure, Numerical")
        a.set_xlim(-0.08 * zoom, zoom); a.set_ylim(0, pmax); a.set_xlabel("Distance Behind the Front [mm]"); a.set_ylabel("Pressure [kPa]", color="C3")
        b = getattr(a, "_twin", None)
        if b is None: b = a.twinx(); a._twin = b
        b.cla(); b.plot(zs, z["T"], "C0-", lw=1.4, label="Temperature, ZND (Cantera)")
        b.plot(s[m][::st], f["T"][m][::st], "s", ms=3.0, mfc="none", mec="C0", mew=0.7, label="Temperature, Numerical")
        b.set_ylim(0, Tmax); b.set_ylabel("Temperature [K]", color="C0"); b.yaxis.set_label_position("right"); b.yaxis.tick_right()
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); a.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower right")
        a.set_title("Reaction Zone: Pressure and Temperature")
        a = ax[1, 1]
        for sp, c, lab in MAJOR:
            a.plot(zs, z[sp], "-", color=c, lw=1.4, label=lab); a.plot(s[m][::st], f[sp][m][::st], "o", ms=3.0, mfc="none", mec=c, mew=0.7)
        b = getattr(a, "_twin", None)
        if b is None: b = a.twinx(); a._twin = b
        b.cla()
        for sp, c, lab in RAD:
            b.plot(zs, z[sp], "--", color=c, lw=1.2, label=lab); b.plot(s[m][::st], f[sp][m][::st], "^", ms=3.0, mfc="none", mec=c, mew=0.7)
        b.set_ylim(0, 1.15 * max(z[sp].max() for sp, _, _ in RAD)); b.set_ylabel("Radical Mass Fraction [-] (Dashed)"); b.yaxis.set_label_position("right"); b.yaxis.tick_right()
        a.set_xlim(-0.08 * zoom, zoom); a.set_ylim(0, 0.112); a.set_xlabel("Distance Behind the Front [mm]"); a.set_ylabel("Major Species Mass Fraction [-] (Solid)")
        a.set_title("Reaction Zone: Species (Lines: ZND (Cantera), Markers: Numerical)", fontsize=10)
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels(); b.legend(h1 + h2, l1 + l2, fontsize=7, ncol=2, loc="lower left", bbox_to_anchor=(0.40, 0.22), framealpha=0.95)
        for a in ax.ravel(): a.grid(alpha=0.3)
        fig.suptitle(r"Detonation in 2H$_2$ + O$_2$ + 7Ar at 6.67 kPa, 298 K" + f" ({cells:.0f} Cells per Induction Length)\n"
                     + rf"t = {1e6 * f['t']:.0f} $\mu$s,  Front at {f['xf']:.3f} m,  Chapman-Jouguet Speed {D:.1f} m/s", fontsize=11)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6)); fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.87, hspace=0.36, wspace=0.30)
    anim = FuncAnimation(fig, lambda i: draw(fig, ax, F[i]), frames=len(F), interval=160)
    out = os.path.join(outdir, "h2o2_detonation.gif"); anim.save(out, writer=PillowWriter(fps=6), dpi=110); print("wrote", out)
    for tag, i in (("first", 0), ("middle", len(F) // 2), ("last", len(F) - 1)):
        draw(fig, ax, F[i]); o = os.path.join(outdir, f"h2o2_detonation_frame_{tag}.png"); fig.savefig(o, dpi=200); print("wrote", o)
    plt.close(fig)
    # multi-time still: the same reaction-zone profiles at several times on one axis each
    pick = [F[i] for i in np.linspace(0, len(F) - 1, 5).astype(int)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for a, key, sc, lab in ((ax[0], "p", 1e-3, "Pressure [kPa]"), (ax[1], "T", 1.0, "Temperature [K]"), (ax[2], "H2O", 1.0, r"H$_2$O Mass Fraction [-]")):
        a.plot(zs, sc * z[key], "k-", lw=1.6, label="ZND (Cantera)")
        for j, f in enumerate(pick):
            s = 1e3 * (f["xf"] - f["x"]); m = (s > -0.08 * zoom) & (s < zoom); st = max(1, int(m.sum() // 90))
            a.plot(s[m][::st], sc * f[key][m][::st], "os^Dv"[j], ms=3.4, mfc="none", mec=f"C{j}", mew=0.7, label=rf"Numerical, t = {1e6 * f['t']:.0f} $\mu$s")
        a.set_xlim(-0.08 * zoom, zoom); a.set_xlabel("Distance Behind the Front [mm]"); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7.5); ax[0].set_title("Pressure"); ax[1].set_title("Temperature"); ax[2].set_title(r"H$_2$O Mass Fraction")
    fig.suptitle(r"Reaction Zone of the Detonation in 2H$_2$ + O$_2$ + 7Ar at 6.67 kPa, 298 K, in the Frame of the Front" + f" ({cells:.0f} Cells per Induction Length)")
    fig.tight_layout(); o = os.path.join(outdir, "h2o2_detonation_times.png"); fig.savefig(o, dpi=200); plt.close(fig); print("wrote", o)


if __name__ == "__main__":
    main()
