"""
Rayleigh-Taylor linear growth rate vs. classical linear theory.

Runs seeded with the deck's velocity perturbation shape but w0 = 2.5e-5
(100x smaller than the deck) so that the growing eigenmode dominates over the
initial transient long before the amplitude leaves the linear regime.

Amplitude a(t) = |cos(kx) Fourier coefficient| of the interface height
(eta = 0.5 / phi = 0.5).  sigma from a least-squares fit of
a = c1 exp(sigma t) + c2 exp(-sigma t) for a < AMAX (k a < 0.1).

Exact (Chandrasekhar 1961, Ch. X; inviscid, incompressible):
  sharp interface, walls at k h = 9.4 (coth(kh) = 1):  sigma = sqrt(A g k) = 0.6472;
  the deck's tanh profile (eps = 0.004), from the continuously-stratified
  eigenproblem (rt_linear_eigen.py):                     sigma = 0.6326 (-2.25 %).

Usage: python3 rt_linear_R32.py OUTDIR hydro:LABEL:PLTDIR ... ref:LABEL:FILE.npz ...
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.argv, _args = [sys.argv[0]], sys.argv
import rt_compare_R32 as C
sys.argv = _args

AMAX = 0.1 / C.K
from rt_linear_eigen import growth as _growth
SIGMA_EPS = _growth(0.004, 6000)   # exact linear rate of the deck's tanh profile (eps = 0.004)


def main():
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    res = []
    for spec in sys.argv[2:]:
        kind, lab, path = spec.split(":", 2)
        D = C.load_hydro(path) if kind == "hydro" else C.load_ref(path)
        t = np.asarray(D["t"]); a = np.array([C.mode_amp(np.asarray(p, float), D["x"], D["y"]) for p in D["phi"]])
        sig, n = C.growth_fit(t, a, AMAX)
        # late-window pure exponential (transient e^{-sigma t} < 2 %)
        m = (t > 6.0) & (a < AMAX)
        sig2 = np.polyfit(t[m], np.log(a[m]), 1)[0] if m.sum() >= 3 else np.nan
        res.append(dict(kind=kind, label=lab, N=len(D["x"]), t=t.tolist(), a=a.tolist(), sigma=sig, npts=int(n),
                        sigma_late=sig2, npts_late=int(m.sum())))
        print(f"{lab:22s} N={len(D['x']):4d}  sigma(2-exp fit) = {sig:.5f} ({100*(sig/C.SIGMA-1):+.2f} % vs sharp, "
              f"{100*(sig/SIGMA_EPS-1):+.2f} % vs tanh profile; {n} pts)  "
              f"sigma(t>6, ln-fit) = {sig2:.5f} ({100*(sig2/SIGMA_EPS-1):+.2f} % vs tanh; {m.sum()} pts)")
        res[-1]["err_sharp"] = sig / C.SIGMA - 1; res[-1]["err_tanh"] = sig / SIGMA_EPS - 1
    print(f"exact sigma = sqrt(A g k) = {C.SIGMA:.5f} (sharp);  tanh profile eps = 0.004: {SIGMA_EPS:.5f}")
    with open(os.path.join(out, "rt_linear_summary.json"), "w") as fh:
        json.dump(dict(sigma_exact=C.SIGMA, sigma_tanh=SIGMA_EPS, runs=res), fh, indent=1)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    mk = ["o", "s", "^", "D"]
    ih = 0
    for r in res:
        t = np.array(r["t"]); a = np.array(r["a"])
        t, a = t[t > 0.3], a[t > 0.3]
        if r["kind"] == "hydro":
            ax.semilogy(t, a, mk[ih % 4], mfc="none", color="tab:red", ms=7, markevery=2,
                        label=f"Numerical, {r['N']}$\\times${3*r['N']}"); ih += 1
        else:
            ax.semilogy(t, a, "--", lw=1.3, color="tab:blue" if r["N"] == max(q["N"] for q in res if q["kind"] == "ref") else "0.55",
                        label=f"Reference HLLC, {r['N']}$\\times${3*r['N']}")
    ref = max([r for r in res if r["kind"] == "ref"] or res, key=lambda r: r["N"])
    t = np.array(ref["t"]); a = np.array(ref["a"])
    i0 = int(np.argmin(abs(t - 8.0)))
    tl = np.linspace(2, 12, 50)
    ax.semilogy(tl, a[i0] * np.exp(SIGMA_EPS * (tl - t[i0])), "k-", lw=1.8, label="Exact")
    ax.axhline(AMAX, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Time, $t$ [s]"); ax.set_ylabel("Interface Mode Amplitude, $a$ [m]")
    ax.set_title("Rayleigh-Taylor Linear Growth Rate")
    ax.set_ylim(1e-5, 0.1); ax.set_xlim(0, 12)
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="lower right", fontsize=12); fig.tight_layout()
    for e in ("png", "eps"):
        fig.savefig(os.path.join(out, f"RT_Linear_Growth.{e}"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
