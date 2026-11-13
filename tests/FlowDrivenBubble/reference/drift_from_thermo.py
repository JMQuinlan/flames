#!/usr/bin/env python3
"""
Radius drift of a driven-bubble run from its thermo.dat (gas volume every sample):

  python3 drift_from_thermo.py <run_dir> [<run_dir> ...] [--omega 256.20] [--labels "A;B"] [--outdir DIR]

Fits  R(t) = c + m t + a sin(wt) + b cos(wt)  over the whole run and over each drive period, and plots the
gas-volume radius, the radius with the drive oscillation removed, and p V^gamma of the gas (constant for an
adiabatic gas: a leak of gas energy shows up as a drift of this quantity).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))


def opt(args, k, d):
    return args[args.index(k) + 1] if k in args else d


def main():
    args = sys.argv[1:]; flags = ("--omega", "--labels", "--outdir"); skip = {args.index(k) + 1 for k in flags if k in args}
    dirs = [a for i, a in enumerate(args) if not a.startswith("--") and i not in skip]
    w = float(opt(args, "--omega", "256.20")); T = 2 * np.pi / w
    labels = opt(args, "--labels", ";".join(os.path.basename(os.path.normpath(d)) for d in dirs)).split(";")
    outdir = opt(args, "--outdir", os.path.join(HERE, "Images", "Drift")); os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.6)); rows = []
    for j, (d, lab) in enumerate(zip(dirs, labels)):
        a = np.genfromtxt(os.path.join(d, "output", "thermo.dat"), names=True); t = a["time"]; V = a["gas_volume"]
        R = (3 * 8 * V / (4 * np.pi)) ** (1 / 3); R0 = R[0]
        A = np.c_[np.ones(len(t)), t, np.sin(w * t), np.cos(w * t)]; c = np.linalg.lstsq(A, R, rcond=None)[0]
        per = []
        for k in range(1, int(round(t[-1] / T)) + 1):
            m = (t >= (k - 1) * T) & (t <= k * T + 1e-9); ck = np.linalg.lstsq(A[m], R[m], rcond=None)[0]; per.append(100 * ck[1] * T / R0)
        S = a["gas_pressure_int"] / V * V ** 1.4
        rows.append((lab, t[-1] / T, 100 * c[1] * T / R0, per, 100 * np.hypot(c[2], c[3]) / R0, 100 * (S[-1] / S[0] - 1)))
        col = f"C{j}"; st = max(1, len(t) // 150)
        ax[0].plot(t / T, R / R0, "-", color=col, lw=0.9); ax[0].plot(t[::st] / T, R[::st] / R0, "os^D"[j % 4], ms=3.2, mfc="none", mec=col, mew=0.7, label=lab)
        # running mean over exactly one drive period: removes the drive and every harmonic of it (incl. the natural frequency ringing)
        dtm = np.median(np.diff(t)); nw = max(3, int(round(T / dtm))); ker = np.ones(nw) / nw
        Rm = np.convolve(R, ker, mode="valid"); tm = np.convolve(t, ker, mode="valid"); sm = max(1, len(tm) // 60)
        ax[1].plot(tm[::sm] / T, 100 * (Rm / Rm[0] - 1)[::sm], "os^D"[j % 4], ms=3.6, mfc="none", mec=col, mew=0.8, label=lab)
        ax[2].plot(t[::st] / T, 100 * (S / S[0] - 1)[::st], "os^D"[j % 4], ms=3.2, mfc="none", mec=col, mew=0.7, label=lab)
    ax[0].set_xlabel("Time [Drive Periods]"); ax[0].set_ylabel(r"Gas-Volume Radius R / R(0) [-]"); ax[0].set_title("Bubble Radius")
    ax[1].set_xlabel("Time [Drive Periods]"); ax[1].set_ylabel("Change of the Period-Averaged Radius [%]"); ax[1].set_title("Radius Averaged over One Drive Period")
    ax[1].axhline(0.0, color="k", lw=1.0, label="No Drift"); ax[1].set_ylim(-0.3, 0.3)
    ax[2].set_xlabel("Time [Drive Periods]"); ax[2].set_ylabel(r"Change of p V$^{\gamma}$ of the Gas [%]"); ax[2].set_title(r"Gas Adiabat p V$^{\gamma}$")
    ax[2].axhline(0.0, color="k", lw=1.0, label="Exact (Adiabatic Gas)")
    for b in ax: b.grid(alpha=0.3)
    ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1); ax[1].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1); ax[2].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
    fig.suptitle("Driven Bubble: Radius Drift over the Drive Periods (Numerical)"); fig.tight_layout()
    o = os.path.join(outdir, "driven_bubble_drift.png"); fig.savefig(o, dpi=200); plt.close(fig)
    with open(os.path.join(outdir, "driven_bubble_drift_summary.csv"), "w") as fo:
        fo.write("run,periods,drift_percent_of_R_per_period,drift_each_period,oscillation_amplitude_percent,pV_gamma_change_percent\n")
        for r in rows:
            fo.write(f"{r[0]},{r[1]:.3f},{r[2]:+.5f},{' '.join('%+.5f' % x for x in r[3])},{r[4]:.4f},{r[5]:+.5f}\n")
            print(f"{r[0]}: {r[1]:.2f} periods, drift {r[2]:+.4f} % of R per period (each period {' '.join('%+.4f' % x for x in r[3])}), amplitude {r[4]:.3f} %, p V^gamma {r[5]:+.4f} %")
    print("wrote", o)


if __name__ == "__main__":
    main()
