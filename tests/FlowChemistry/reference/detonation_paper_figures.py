#!/usr/bin/env python3
"""
Paper figures of the hydrogen-oxygen detonation started from its ZND profile
(tests/FlowChemistry/input_1D_H2O2Detonation_ZND), split out of the animation frames of detonation_movie.py.

  python3 detonation_paper_figures.py <run_dir> [--outdir DIR] [--times 0,260,530] [--zoom 25] [--prefix znd_h2o2ar_]

Writes (PNG and EPS):
  <prefix>tube_pressure, <prefix>tube_temperature     whole tube, about 11 times, one colour per time
  <prefix>rz_pT_row, <prefix>rz_pT_stack              reaction zone, pressure and temperature, 1x3 and 3x1
  <prefix>rz_species_row, <prefix>rz_species_stack    reaction zone, major species and radicals, 1x3 and 3x1
  <prefix>rz_species_<t>us                            the species panel of each time on its own
  <prefix>reaction_zone_times                         p, T, Y_H2O at five times on one axis each
Reference (Cantera ZND, same mechanism) = lines, Numerical = open markers.  Mass fractions carry no unit label.
"""
import os, sys, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detonation_movie import load, opt, ZND, COLS, W, MAJOR, RAD, HERE

plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12,
                     "xtick.labelsize": 13, "ytick.labelsize": 13})


def save(fig, outdir, name):
    for ext in ("png", "eps"): fig.savefig(os.path.join(outdir, name + "." + ext), dpi=200 if ext == "png" else None)
    plt.close(fig); print("wrote", os.path.join(outdir, name + ".png"))


def main():
    args = sys.argv[1:]; flags = ("--outdir", "--times", "--zoom", "--prefix")
    skip = {args.index(k) + 1 for k in flags if k in args}
    d = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip][0]
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "H2O2_Detonation_Paper")); os.makedirs(outdir, exist_ok=True)
    zoom = float(opt(args, "--zoom", "25")); pre = opt(args, "--prefix", "znd_h2o2ar_")
    times = [1e-6 * float(t) for t in opt(args, "--times", "0,260,530").split(",")]
    pfs = sorted(glob.glob(os.path.join(d, "output", "*[0-9]cell")), key=lambda f: int(os.path.basename(f)[:-4]))
    F = [load(pf) for pf in pfs]
    if len(F) > 1 and abs(F[1]["t"] - F[0]["t"]) < 1e-12: F = F[1:]
    near = lambda t: F[int(np.argmin([abs(f["t"] - t) for f in F]))]
    Z = np.loadtxt(ZND); z = {c: Z[:, i] for i, c in enumerate(COLS)}
    Wm = 1.0 / sum(z[s] / W[s] for s in W); z["T"] = z["p"] * Wm * 1e-3 / (8.314462618 * z["rho"])
    zs = 1e3 * z["s"]
    pmax, Tmax, L = 1.08 * z["p"].max() / 1e3, 1.08 * max(z["T"].max(), max(f["T"].max() for f in F)), F[0]["L"]
    radmax = 1.15 * max(z[sp].max() for sp, _, _ in RAD)
    us = lambda f: 1e6 * f["t"]

    # ---- whole tube, about 11 times
    tube = [near(t) for t in np.arange(0.0, F[-1]["t"] + 1e-9, 50e-6)]
    if abs(tube[-1]["t"] - F[-1]["t"]) > 1e-9: tube.append(F[-1])
    cmap = plt.get_cmap("viridis")
    for key, sc, lab, ttl, top, name in (("p", 1e-3, "Pressure, $p$ [kPa]", "Pressure Along the Tube", pmax, "tube_pressure"),
                                         ("T", 1.0, "Temperature, $T$ [K]", "Temperature Along the Tube", Tmax, "tube_temperature")):
        fig, a = plt.subplots(figsize=(11, 5.4))
        for j, f in enumerate(tube):
            a.plot(f["x"], sc * f[key], "-", lw=1.3, color=cmap(j / max(1, len(tube) - 1)), label=rf"$t$ = {us(f):.0f} $\mu$s")
        a.set_xlim(0, L); a.set_ylim(0, top); a.set_xlabel("Position, $x$ [m]"); a.set_ylabel(lab); a.grid(alpha=0.3)
        a.set_title(ttl + r" (Numerical), 2H$_2$ + O$_2$ + 7Ar")
        a.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11.5, framealpha=0.95)
        fig.tight_layout(); save(fig, outdir, pre + name)

    sel = [near(t) for t in times]

    def zone(f, npts=160):
        s = 1e3 * (f["xf"] - f["x"]); m = (s > -0.08 * zoom) & (s < zoom); st = max(1, int(m.sum() // npts)); return s[m][::st], m, st

    def panel_pT(a, f, legend):
        s, m, st = zone(f)
        a.plot(zs, z["p"] / 1e3, "C3-", lw=1.6, label="Pressure, ZND (Cantera)")
        a.plot(s, f["p"][m][::st] / 1e3, "o", ms=4.5, mfc="none", mec="C3", mew=0.9, label="Pressure, Numerical")
        a.set_xlim(-0.08 * zoom, zoom); a.set_ylim(0, pmax); a.set_xlabel("Distance Behind the Front [mm]")
        a.set_ylabel("Pressure, $p$ [kPa]", color="C3"); a.grid(alpha=0.3)
        b = a.twinx()
        b.plot(zs, z["T"], "C0-", lw=1.6, label="Temperature, ZND (Cantera)")
        b.plot(s, f["T"][m][::st], "s", ms=4.2, mfc="none", mec="C0", mew=0.9, label="Temperature, Numerical")
        b.set_ylim(0, Tmax); b.set_ylabel("Temperature, $T$ [K]", color="C0")
        a.set_title(rf"$t$ = {us(f):.0f} $\mu$s, Front at {f['xf']:.3f} m")
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
        if legend: a.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=11)
        return h1 + h2, l1 + l2

    def panel_sp(a, f, legend):
        s, m, st = zone(f)
        for sp, c, lab in MAJOR:
            a.plot(zs, z[sp], "-", color=c, lw=1.6, label=lab + ", ZND (Cantera)")
            a.plot(s, f[sp][m][::st], "o", ms=4.2, mfc="none", mec=c, mew=0.9, label=lab + ", Numerical")
        a.set_xlim(-0.08 * zoom, zoom); a.set_ylim(0, 0.112); a.set_xlabel("Distance Behind the Front [mm]")
        a.set_ylabel("Major Species Mass Fraction (Solid)"); a.grid(alpha=0.3)
        b = a.twinx()
        for sp, c, lab in RAD:
            b.plot(zs, z[sp], "--", color=c, lw=1.4, label=lab + ", ZND (Cantera)")
            b.plot(s, f[sp][m][::st], "^", ms=4.2, mfc="none", mec=c, mew=0.9, label=lab + ", Numerical")
        b.set_ylim(0, radmax); b.set_ylabel("Radical Mass Fraction (Dashed)")
        a.set_title(rf"$t$ = {us(f):.0f} $\mu$s, Front at {f['xf']:.3f} m")
        h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
        if legend: b.legend(h1 + h2, l1 + l2, ncol=2, fontsize=10.5, loc="lower left", bbox_to_anchor=(0.36, 0.18), framealpha=0.95)
        return h1 + h2, l1 + l2

    for panel, ttl, name, ncol in ((panel_pT, "Reaction Zone: Pressure and Temperature", "rz_pT", 4),
                                   (panel_sp, "Reaction Zone: Species", "rz_species", 6)):
        for lay, shape, size in (("row", (1, 3), (19, 6.0)), ("stack", (3, 1), (10.5, 15.5))):
            fig, ax = plt.subplots(*shape, figsize=size); ax = np.ravel(ax)
            for a, f in zip(ax, sel): h, l = panel(a, f, legend=False)
            fig.suptitle(ttl + " (Lines: ZND (Cantera), Markers: Numerical)", fontsize=17)
            if lay == "row":
                fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=12, framealpha=0.95)
                fig.tight_layout(rect=(0, 0.17 if ncol == 6 else 0.09, 1, 0.97)); fig.subplots_adjust(wspace=0.42)
            else:
                fig.legend(h, l, loc="lower center", ncol=min(ncol, 3), fontsize=12, framealpha=0.95)
                fig.tight_layout(rect=(0, 0.11 if ncol == 6 else 0.05, 1, 0.98))
            save(fig, outdir, pre + name + "_" + lay)

    for f in sel:
        fig, a = plt.subplots(figsize=(10, 6.4)); panel_sp(a, f, legend=True)
        a.set_title(rf"Reaction Zone: Species, $t$ = {us(f):.0f} $\mu$s, Front at {f['xf']:.3f} m")
        fig.tight_layout(); save(fig, outdir, pre + f"rz_species_{us(f):03.0f}us")

    # ---- reaction zone at five times on one axis per quantity
    pick = [F[i] for i in np.linspace(0, len(F) - 1, 5).astype(int)]
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8))
    for a, key, sc, lab, ttl in ((ax[0], "p", 1e-3, "Pressure, $p$ [kPa]", "Pressure"), (ax[1], "T", 1.0, "Temperature, $T$ [K]", "Temperature"),
                                 (ax[2], "H2O", 1.0, r"H$_2$O Mass Fraction", r"H$_2$O Mass Fraction")):
        a.plot(zs, sc * z[key], "k-", lw=1.8, label="ZND (Cantera)")
        for j, f in enumerate(pick):
            s, m, st = zone(f, 90)
            a.plot(s, sc * f[key][m][::st], "os^Dv"[j], ms=4.5, mfc="none", mec=f"C{j}", mew=0.9, label=rf"Numerical, $t$ = {us(f):.0f} $\mu$s")
        a.set_xlim(-0.08 * zoom, zoom); a.set_xlabel("Distance Behind the Front [mm]"); a.set_ylabel(lab); a.set_title(ttl); a.grid(alpha=0.3)
    ax[0].legend(fontsize=11)
    fig.suptitle(r"Reaction Zone of the Detonation in 2H$_2$ + O$_2$ + 7Ar at 6.67 kPa, 298 K", fontsize=17)
    fig.tight_layout(); save(fig, outdir, pre + "reaction_zone_times")


if __name__ == "__main__":
    main()
