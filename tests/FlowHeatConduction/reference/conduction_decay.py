#!/usr/bin/env python3
"""
Hydro2 heat-conduction unit test: decay of an isobaric temperature sine wave.

  python3 conduction_decay.py <plot_dir> [mu=0.01] [Pr=0.72] [L=1.0]

Fits the amplitude of the fundamental Fourier mode of T(x) in every plotfile and
compares ln(A/A0) against the linear isobaric decay  -kappa_p (2 pi / L)^2 t,
kappa_p = mu / (rho Pr)   (rho = mean density = 1).
PASS: fitted decay rate within 5% of theory.
"""
import glob, sys
import numpy as np

def main():
    d = sys.argv[1]
    mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    Pr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.72
    L = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    import yt
    yt.set_log_level(40)
    ts, amps = [], []
    for pf in sorted(glob.glob(d + "/*cell")):
        ds = yt.load(pf)
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions)
        T = np.asarray(cg[("boxlib", "T")])[:, :, 0].mean(axis=1)
        rho = np.asarray(cg[("boxlib", "density")])[:, :, 0].mean()
        n = len(T); x = (np.arange(n) + 0.5) / n * L
        A = 2.0 / n * np.abs(np.sum((T - T.mean()) * np.exp(-2j * np.pi * x / L)))
        ts.append(float(ds.current_time)); amps.append(A / T.mean())
    ts = np.array(ts); amps = np.array(amps)
    kap = mu / (rho * Pr); rate_th = kap * (2 * np.pi / L) ** 2
    m = ts > 0.1                        # skip the fast acoustic equilibration
    rate = -np.polyfit(ts[m], np.log(amps[m]), 1)[0]
    for t, a in zip(ts, amps):
        print(f"  t={t:6.3f}  A/T0={a:.5f}   theory {amps[0]*np.exp(-rate_th*t):.5f}")
    err = rate / rate_th - 1
    print(f"decay rate: measured {rate:.4f}  theory {rate_th:.4f}  error {err*100:+.2f} %")
    print("RESULT:", "PASS" if abs(err) < 0.05 else "FAIL (|error| > 5 %)")

if __name__ == "__main__":
    main()
