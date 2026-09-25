#!/usr/bin/env python3
"""
Temporal analysis of an airfoil run's per-step force / probe history
(<plot_file>_forces.dat written by hydro2 with solid.force_int > 0).

  python3 force_history.py <case_dir> [A|B] [t_start_for_spectrum]

Columns (2D): step lev time Ftot_x Ftot_y Fpres_x Fpres_y p_probe0 .. p_probeN
  Cd = Fx / (q c), Cl = Fy / (q c), q = 0.5 rho U^2  (freestream along +x).
Reports running mean / std of Cl, Cd over the last windows, the dominant
frequency (Strouhal St = f c / U) of Cl, Cd and each probe pressure after
t_start (default: second half of the record), and a trend (linear drift per
convective time) so slow drifts are not mistaken for convergence.
Writes <case_dir>/force_history.png.
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import viscous_naca0012 as V

def spectrum(t, y):
    """Dominant frequency of y(t) on a uniform resample (Hann window)."""
    n = len(t)
    if n < 64: return np.nan, np.nan
    tu = np.linspace(t[0], t[-1], n); yu = np.interp(tu, t, y) - np.mean(y)
    yu *= np.hanning(n)
    F = np.abs(np.fft.rfft(yu)); f = np.fft.rfftfreq(n, tu[1] - tu[0])
    k = 1 + np.argmax(F[1:]); amp = 2 * F[k] / np.sum(np.hanning(n))
    return f[k], amp

def main():
    d = sys.argv[1]; case = sys.argv[2] if len(sys.argv) > 2 else "A"
    c = V.CASES[case]; U = c["mach"]; q = 0.5 * V.RHO * U**2
    fn = sorted(glob.glob(os.path.join(d, "*_forces.dat")))[-1]
    hdr = open(fn).readline().lstrip("#").split()
    D = np.loadtxt(fn)
    # keep only the finest level, drop duplicate times (restart safety)
    D = D[D[:, 1] == D[:, 1].max()]
    t = D[:, 2]; o = np.argsort(t); D = D[o]; t = t[o]
    Cd, Cl = D[:, 3] / q, D[:, 4] / q
    Cdp, Clp = D[:, 5] / q, D[:, 6] / q
    probes = [(hdr[j], D[:, j]) for j in range(7, D.shape[1])]
    tc = t * U / V.CHORD                        # convective time
    ts = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5 * t[-1]
    m = t >= ts
    print(f"{fn}: {len(t)} samples, t = {t[0]:.3f} .. {t[-1]:.3f}  (t U/c up to {tc[-1]:.2f})")
    for w in (1.0, 2.0, 5.0):                    # windows in convective times
        mw = tc >= tc[-1] - w
        if mw.sum() < 10: continue
        print(f"  last {w:3.0f} c/U : Cd {Cd[mw].mean():.5f} +- {Cd[mw].std():.2e}   Cl {Cl[mw].mean():+.5f} +- {Cl[mw].std():.2e}"
              f"   (Cd_p {Cdp[mw].mean():.5f})")
    if m.sum() > 64:
        for name, y in [("Cd", Cd), ("Cl", Cl)] + probes:
            f, amp = spectrum(t[m], y[m])
            drift = np.polyfit(tc[m], y[m], 1)[0]
            print(f"  {name:28s}: dominant St = {f * V.CHORD / U:7.3f}  amplitude {amp:.3e}   drift {drift:+.2e} per c/U")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(15, 8))
        ax[0, 0].plot(tc, Cd, lw=.6, label="Cd total"); ax[0, 0].plot(tc, Cdp, lw=.6, label="Cd pressure")
        ax[0, 0].set_ylim(0, max(0.1, 2 * np.median(Cd))); ax[0, 0].legend(); ax[0, 0].set_xlabel("t U/c"); ax[0, 0].grid(alpha=.3)
        ax[0, 1].plot(tc, Cl, lw=.6, label="Cl total"); ax[0, 1].plot(tc, Clp, lw=.6, label="Cl pressure")
        ax[0, 1].legend(); ax[0, 1].set_xlabel("t U/c"); ax[0, 1].grid(alpha=.3)
        for name, y in probes: ax[1, 0].plot(tc, y, lw=.6, label=name)
        ax[1, 0].legend(fontsize=7); ax[1, 0].set_xlabel("t U/c"); ax[1, 0].set_ylabel("p"); ax[1, 0].grid(alpha=.3)
        if m.sum() > 64:
            for name, y in [("Cd", Cd), ("Cl", Cl)] + probes:
                n = m.sum(); tu = np.linspace(t[m][0], t[m][-1], n); yu = np.interp(tu, t[m], y[m]) - y[m].mean()
                F = np.abs(np.fft.rfft(yu * np.hanning(n))); f = np.fft.rfftfreq(n, tu[1] - tu[0]) * V.CHORD / U
                ax[1, 1].semilogy(f[1:], F[1:] / F[1:].max(), lw=.6, label=name)
            ax[1, 1].set_xlim(0, 5); ax[1, 1].set_xlabel("St = f c / U"); ax[1, 1].set_title(f"spectrum, t >= {ts:.1f}")
            ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3)
        fig.tight_layout(); out = os.path.join(d, "force_history.png"); fig.savefig(out, dpi=110); print("  figure:", out)
    except Exception as e:
        print("  (figure skipped:", e, ")")

if __name__ == "__main__":
    main()
