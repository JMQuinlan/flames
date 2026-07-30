"""
Time histories of C_l and C_d for every case in a sweep work dir -- the
steady-state diagnostic (is the window average taken on a settled flow?).

usage:
  python3 history_NACA_0012.py [workdir] [--lev N] [--aoa 0,2,4]
    workdir : default = analyze_NACA_0012.WORK (the current sweep dir)
    --lev   : covering-grid level for the force integrals (default 3: fast;
              use finest only for publication numbers, when solvers are idle)
    --aoa   : comma-separated subset of angles to plot (default: all cases)

Writes Images/naca0012_history_<workdir>.png and prints a settledness report
(drift of each quantity between the last two 3-time-unit windows).
"""
import os, sys, glob
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_NACA_0012 as m


def case_history(case, lev):
    cfg = m.parse_case_input(case["inp"])
    qc = cfg["q_inf"] * m.CHORD
    T, CL, CD = [], [], []
    for pf in sorted(glob.glob(os.path.join(case["plot"], "*cell"))):
        try:
            t, g, dx, dy = m._grid_at(pf, lev)
            phi, p = g("phi"), g("pressure")
            gx = np.gradient(phi, dx, axis=0); gy = np.gradient(phi, dy, axis=1)
            pp = p - cfg["p_inf"]
            CL.append(-np.sum(pp * gy) * dx * dy / qc)
            CD.append(-np.sum(pp * gx) * dx * dy / qc)
            T.append(t)
        except Exception as e:
            print(f"    skip {os.path.basename(pf)}: {e!r}")
    return np.array(T), np.array(CL), np.array(CD)


def drift(T, Y, w=3.0):
    """mean(last w units) - mean(previous w units); NaN if run too short."""
    if len(T) < 4 or T[-1] - T[0] < 2 * w:
        return np.nan
    m2 = T >= T[-1] - w
    m1 = (T >= T[-1] - 2 * w) & ~m2
    return float(Y[m2].mean() - Y[m1].mean())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    work = args[0] if args else m.WORK
    lev, subset = 3, None
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a.startswith("--lev"):
            v = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            lev = None if v.lower() in ("none", "finest") else int(v)
        if a.startswith("--aoa"):
            v = a.split("=", 1)[1] if "=" in a else argv[i + 1]
            subset = {float(s) for s in v.split(",")}

    cases = m.find_cases(work)
    if subset is not None:
        cases = [c for c in cases if c["aoa"] in subset]
    if not cases:
        sys.exit(f"no cases under {work}" + (f" matching --aoa" if subset else ""))

    # Grayscale-safe styling: lightness ordered by alpha AND distinct dash
    # patterns, so the figure survives black-and-white printing.
    greys = plt.get_cmap("Greys")
    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]

    fig, ax = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    tag = os.path.basename(os.path.normpath(work))
    print(f"histories from {work}  (force grid level: {lev if lev is not None else 'finest'})")
    print(f"{'case':>8} {'t_end':>6} {'Cl_end':>8} {'dCl':>8} {'dCd':>8}   (d = last 3u mean - prev 3u mean)")
    for i, c in enumerate(cases):
        T, CL, CD = case_history(c, lev)
        if len(T) == 0:
            continue
        col = greys(0.35 + 0.60 * (i / max(1, len(cases) - 1)))
        ls = styles[i % len(styles)]
        lab = rf"$\alpha = {c['aoa']:g}^\circ$"
        ax[0].plot(T, CL, linestyle=ls, color=col, lw=1.6, label=lab)
        ax[1].plot(T, CD, linestyle=ls, color=col, lw=1.6, label=lab)
        print(f"{os.path.basename(c['rd']):>8} {T[-1]:6.1f} {CL[-1]:+8.3f} "
              f"{drift(T, CL):+8.4f} {drift(T, CD):+8.4f}")

    for a, ylab in zip(ax, (r"$C_l$", r"$C_d$")):
        a.set_ylabel(ylab, fontsize=12)
        a.grid(alpha=0.3)
        a.axvline(m.T_AVG_START, color="k", ls=":", lw=0.8)
        a.legend(fontsize=9, ncol=3, frameon=False)
    ax[0].text(m.T_AVG_START, ax[0].get_ylim()[1], " averaging window ", va="top",
               ha="left", fontsize=8, color="0.3")
    ax[1].set_xlabel(r"$t$", fontsize=12)
    fig.suptitle(f"{m.AIRFOIL.replace('_', ' ')} force histories — {tag}")
    out = os.path.join(m.images_dir(tag), f"{m.PREFIX}_history.png")
    plt.tight_layout(); plt.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
