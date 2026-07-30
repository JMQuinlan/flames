"""Shedding-frequency (Strouhal) check for the unsteady high-alpha cases.

FFT of the C_l(t) history over the post-transient window (t >= T0).  Reports
  St_c = f c / U          (chord-based)
  St_p = f c sin(a) / U   (projected-chord -- the Fage-Johansen "universal"
                           bluff-body scaling: measured 0.16-0.20 for airfoils
                           past stall across Re 1e3..1e5)
usage:
  python3 strouhal_NACA_0012.py [workdir] [--lev N] [--aoa 10,12,14,16] [--t0 8]
Frequency resolution is 1/(t_end - T0): a t=24 run resolves f to ~0.06 (crude,
~30% at f~0.2); the t=60 extensions bring that to ~0.02.
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_NACA_0012 as m
import history_NACA_0012 as h


def spectrum(T, Y):
    """Peak frequency of the detrended, Hann-windowed signal (uniform dt)."""
    dt = np.median(np.diff(T))
    y = Y - np.polyval(np.polyfit(T, Y, 1), T)      # remove mean + linear drift
    y = y * np.hanning(len(y))
    f = np.fft.rfftfreq(len(y), dt)
    a = np.abs(np.fft.rfft(y))
    a[0] = 0.0
    return f, a, f[np.argmax(a)]


def main():
    lev, subset, t0 = 3, {10.0, 12.0, 14.0, 16.0}, 8.0
    argv, positional, i = sys.argv[1:], [], 0
    while i < len(argv):
        a = argv[i]
        if not a.startswith("--"):
            positional.append(a); i += 1; continue
        v = a.split("=", 1)[1] if "=" in a else (argv[i + 1] if i + 1 < len(argv) else "")
        i += 1 if "=" in a else 2                    # consume the value token too
        if a.startswith("--lev"):
            lev = None if v.lower() in ("none", "finest") else int(v)
        if a.startswith("--aoa"):
            subset = {float(s) for s in v.split(",")}
        if a.startswith("--t0"):
            t0 = float(v)
    work = positional[0] if positional else m.WORK

    cases = [c for c in m.find_cases(work) if c["aoa"] in subset]
    if not cases:
        sys.exit(f"no cases under {work} matching {sorted(subset)}")
    U = m.parse_case_input(cases[0]["inp"]).get("u_inf", 0.3)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    greys = plt.get_cmap("Greys")
    print(f"Strouhal from {work}  (lev={lev}, window t>={t0:g}, U={U:g})")
    print(f"{'case':>8} {'n':>4} {'df':>6} {'f_peak':>7} {'St_c':>6} {'St_p':>6}   (universal St_p ~ 0.16-0.20)")
    for i, c in enumerate(cases):
        T, CL, _ = h.case_history(c, lev)
        w = T >= t0
        if w.sum() < 8:
            print(f"{os.path.basename(c['rd']):>8}  too few samples past t0"); continue
        f, a, fp = spectrum(T[w], CL[w])
        st_c = fp * m.CHORD / U
        st_p = st_c * np.sin(np.radians(c["aoa"]))
        df = f[1] * m.CHORD / U
        print(f"{os.path.basename(c['rd']):>8} {int(w.sum()):>4} {df:6.3f} {fp:7.3f} {st_c:6.2f} {st_p:6.2f}")
        col = greys(0.4 + 0.55 * (i / max(1, len(cases) - 1)))
        ax.plot(f * m.CHORD / U, a / a.max(), color=col, lw=1.6,
                label=rf"$\alpha={c['aoa']:g}^\circ$: $St_p={st_p:.2f}$")

    ax.axvspan(0.16, 0.20, color="r", alpha=0.12,
               label=r"universal $St_p$ band (as $St_c$ at $\alpha=90^\circ$)")
    ax.set_xlim(0, 1.5); ax.set_xlabel(r"$St_c = f\,c/U$", fontsize=12)
    ax.set_ylabel(r"$|\hat{C}_l|$ (normalized)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=9, frameon=False)
    tag = os.path.basename(os.path.normpath(work))
    ax.set_title(rf"{m.AIRFOIL.replace('_', ' ')} $C_l$ spectra — {tag}", fontsize=11)
    out = os.path.join(m.images_dir(tag), f"{m.PREFIX}_strouhal.png")
    plt.tight_layout(); plt.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
