#!/usr/bin/env python3
"""
Force history + vortex-shedding statistics for a Hydro2 cylinder run
(solid.force_int = 1 -> <run_dir>/output_forces.dat).

  python3 cylinder_force_history.py <run_dir> [--outdir DIR] [--window W]

Writes to <outdir> (default Images/<run_dir name>/ next to this script):
  force_history.png     Cl(t) and Cd(t) (+ |Cl| on a log axis to show the
                        growth of the shedding instability from round-off)
  shedding_summary.csv  Re, Cd mean / amplitude, Cl amplitude, Strouhal number
                        St = f D / U over the last W time units (default 30),
                        and whether the wake is shedding (Cl amplitude > 1e-3)
q = 0.5 rho U^2 D with rho = 100, U = 1, D = 1 (the FlowVortexShed decks).
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RHO, U, D = 100.0, 1.0, 1.0
SHED_CL_AMP = 1e-3          # Cl amplitude above which the wake counts as shedding


def main():
    args = sys.argv[1:]
    d = args[0]
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = (args[args.index("--outdir") + 1] if "--outdir" in args
              else os.path.join(here, "Images", os.path.basename(os.path.normpath(d))))
    W = float(args[args.index("--window") + 1]) if "--window" in args else 30.0
    m_re = re.search(r"Re[_=]?(\d+(?:\.\d+)?)", os.path.abspath(d))
    Re = float(m_re.group(1)) if m_re else None
    q = 0.5 * RHO * U * U * D
    F = np.loadtxt(os.path.join(d, "output_forces.dat"))
    F = F[F[:, 1] == F[:, 1].max()]
    t, Cd, Cl = F[:, 2], F[:, 3] / q, F[:, 4] / q
    m = t >= t[-1] - W
    # Strouhal number from the Cl spectrum over the window (uniform resample, Hann window)
    tu = np.linspace(t[m][0], t[m][-1], 8192)
    y = np.interp(tu, t[m], Cl[m]); y -= y.mean()
    Fq = np.abs(np.fft.rfft(y * np.hanning(len(y)))); f = np.fft.rfftfreq(len(y), tu[1] - tu[0])
    band = (f > 0.02) & (f < 1.0)
    fpk = f[band][np.argmax(Fq[band])] if band.any() else np.nan
    St = fpk * D / U
    cl_amp = 0.5 * (Cl[m].max() - Cl[m].min())
    cd_amp = 0.5 * (Cd[m].max() - Cd[m].min())
    shedding = cl_amp > SHED_CL_AMP
    os.makedirs(outdir, exist_ok=True)
    rows = [("Re", Re), ("window_t0", t[m][0]), ("window_t1", t[-1]),
            ("shedding", shedding), ("St", St if shedding else None),
            ("Cl_amplitude", cl_amp), ("Cl_rms", float(np.std(Cl[m]))),
            ("Cd_mean", float(Cd[m].mean())), ("Cd_amplitude", cd_amp)]
    with open(os.path.join(outdir, "shedding_summary.csv"), "w") as fo:
        fo.write("metric,value\n")
        for k, v in rows:
            fo.write(f"{k},{'' if v is None else (f'{v:.6g}' if isinstance(v, float) else v)}\n")
    print(f"{d}: t = {t[0]:.1f} .. {t[-1]:.1f}   window last {W:g}")
    print(f"  shedding = {shedding}   Cl amplitude {cl_amp:.3e}   St = {St:.4f}   "
          f"Cd mean {Cd[m].mean():.4f} +- {cd_amp:.2e}")
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    ax[0].plot(t, Cd, lw=0.8, color="k"); ax[0].set_ylabel(r"$C_d$")
    lo, hi = np.percentile(Cd[t > 5], [0.5, 99.5]); pad = 0.1 * (hi - lo) + 1e-3
    ax[0].set_ylim(lo - pad, hi + pad)
    ax[1].plot(t, Cl, lw=0.8, color="C0"); ax[1].set_ylabel(r"$C_l$")
    ax[2].semilogy(t, np.abs(Cl) + 1e-16, lw=0.6, color="C3"); ax[2].set_ylabel(r"$|C_l|$")
    ax[2].set_xlabel(r"$t\,U/D$")
    for a in ax: a.grid(alpha=0.3)
    ttl = (f"Re = {Re:g}: " if Re else "") + (f"St = {St:.3f}, $C_l$ amplitude {cl_amp:.3f}" if shedding
                                              else f"steady (|$C_l$| amplitude {cl_amp:.1e})")
    ax[0].set_title(ttl)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "force_history.png"), dpi=200); plt.close(fig)
    print(f"  wrote {outdir}/force_history.png and shedding_summary.csv")


if __name__ == "__main__":
    main()
