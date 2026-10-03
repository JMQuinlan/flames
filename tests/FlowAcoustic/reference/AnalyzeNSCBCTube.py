"""NSCBC tube unit tests -- reflective vs non-reflective outflow.

A right-travelling Gaussian pressure pulse in water hits the NSCBC4 outflow
face at xhi (input_NSCBCTube_<amp>_<Default|Detuned>).  For each case this
writes one standalone figure

  Images/NSCBCTube_<case>.png   transient p'(x) vs the non-reflecting analytic
                                wave, the x-t diagram, and the peak history

and prints the measured reflection coefficient

    R = (signed peak of the returning wave) / (incident peak)

next to the linear relaxation estimate.  The outflow LODI closure relaxes the
incoming characteristic at rate K = sigma (1 - M^2) c / L_ref, which for a
plane wave gives  R(w) = -1 / (1 + 2 i w / K)  (Selle, Nicoud & Poinsot 2004),
i.e. the returning wave r(t) obeys  r' + (K/2) r = -(K/2) g(t)  for an incident
wave g(t) at the face.  K >> w: pressure-release reflection, R -> -1.
K << w: non-reflecting.

Run from anywhere:  python tests/FlowAcoustic/reference/AnalyzeNSCBCTube.py
"""
import glob, os, re, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt; yt.funcs.mylog.setLevel(50)

HERE = os.path.dirname(os.path.abspath(__file__))
IMG  = os.path.join(HERE, "Images"); os.makedirs(IMG, exist_ok=True)
TEST = os.path.normpath(os.path.join(HERE, ".."))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "..", "bin", "tests", "FlowAcoustic"))
CASES = ["10kPa_Tuned", "10kPa_Default", "10kPa_Detuned", "1MPa_Tuned", "1MPa_Default", "1MPa_Detuned"]

# NSCBC4 code defaults (src/BC/NSCBC4.H, NSCBC4.cpp) used when a deck omits them.
SIGMA_DEFAULT, LREF_DOMAIN_FRACTION = 0.25, 20.0


def parse(path):
    kv = {}
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1); kv[k.strip()] = v.strip()
    return kv


def load(case):
    pfs = sorted(glob.glob(os.path.join(ROOT, f"output_NSCBCTube_{case}", "*cell")),
                 key=lambda s: int(re.search(r"(\d+)cell", s).group(1)))
    out = []
    for pf in pfs:
        ds = yt.load(pf); ad = ds.all_data()
        x = np.array(ad["x"], float); p = np.array(ad["pressure"], float)
        xs = np.unique(np.round(x, 12))
        out.append((float(ds.current_time), xs,
                    np.array([p[np.isclose(x, v)].mean() for v in xs])))
    return out


def relaxation_estimate(A, c, pw, K, t_arr):
    """Peak of r(t), r' + (K/2) r = -(K/2) g(t), g = incident wave at the face."""
    t = np.linspace(t_arr - 6 * pw / c, t_arr + 14 * pw / c, 40001); dt = t[1] - t[0]
    g = A * np.exp(-((c * (t - t_arr)) / pw) ** 2)
    r = np.zeros_like(t); a = 0.5 * K * dt
    for i in range(1, len(t)):                       # implicit trapezoid, stable for any K
        r[i] = ((1 - 0.5 * a) * r[i - 1] - 0.5 * a * (g[i] + g[i - 1])) / (1 + 0.5 * a)
    return r[np.argmax(np.abs(r))] / A


summary = []
for case in CASES:
    kv = parse(os.path.join(TEST, f"input_NSCBCTube_{case}"))
    fr = load(case)
    if not fr:
        print(f"{case}: no plotfiles in {ROOT}"); continue
    gam, pinf = float(kv["eos1.gamma"]), float(kv["eos1.p0"])
    P0, RHO0 = 101325.0, 1000.0
    c = math.sqrt(gam * (P0 + pinf) / RHO0)
    A  = float(kv["pressure1.ic.expression.constant.pamp"])
    xc = float(kv["pressure1.ic.expression.constant.xc"]); pw = float(kv["pressure1.ic.expression.constant.pw"])
    Lx = float(kv["geometry.prob_hi"].split()[0]) - float(kv["geometry.prob_lo"].split()[0])
    sig  = float(kv.get("nscbc.xhi.sigma", SIGMA_DEFAULT))
    lref = float(kv.get("nscbc.xhi.L_ref", Lx / LREF_DOMAIN_FRACTION))
    given = "nscbc.xhi.sigma" in kv
    K = sig * c / lref
    t_arr = (Lx - xc) / c
    exact = lambda t, x: A * np.exp(-((x - xc - c * t) / pw) ** 2)      # non-reflecting solution, p'

    ts = np.array([f[0] for f in fr]); x = fr[0][1]
    P = np.array([f[2] - P0 for f in fr])                                # p'(t, x)
    # Returning wave: frames after the pulse has fully left through xhi.
    late = ts > t_arr + 3.0 * pw / c
    i_inc = np.argmin(np.abs(ts - 0.5 * t_arr))
    A_inc = P[i_inc].max()
    refl = P[late]
    k = np.unravel_index(np.argmax(np.abs(refl)), refl.shape)
    R = refl[k] / A_inc
    R_th = relaxation_estimate(A, c, pw, K, t_arr)

    unit, sc = ("kPa", 1e-3) if A < 1e5 else ("MPa", 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    sel = [fr[i] for i in np.linspace(0, len(fr) - 1, 7).astype(int)]
    cm = plt.get_cmap("viridis")
    for j, (t, xx, p) in enumerate(sel):
        ax[0].plot(xx * 1e3, (p - P0) * sc, color=cm(j / 6), lw=1.5, label=f"t={t*1e6:.1f} $\\mu$s")
        ax[0].plot(xx * 1e3, exact(t, xx) * sc, color="0.55", lw=0.9, ls=":", zorder=0)
    ax[0].axhline(0, color="0.8", lw=0.6)
    ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel(f"p - p$_0$ [{unit}]"); ax[0].grid(alpha=.25, lw=.6)
    ax[0].set_title("simulation (color) vs non-reflecting analytic (dotted)", fontsize=10)
    lo, hi = ax[0].get_ylim(); ax[0].set_ylim(lo, hi + 0.32 * (hi - lo))      # head-room for the legend
    ax[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="upper center")

    vm = 1.0
    im = ax[1].pcolormesh(x * 1e3, ts * 1e6, P / A_inc, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto", rasterized=True)
    ax[1].axhline(t_arr * 1e6, color="0.3", lw=0.8, ls="--")
    ax[1].text(0.2, t_arr * 1e6 + 0.15, "pulse reaches xhi", fontsize=8, color="0.3")
    ax[1].set_xlabel("x [mm]"); ax[1].set_ylabel("t [$\\mu$s]")
    ax[1].set_title("x-t diagram of p'/A (red +, blue -)", fontsize=10)
    fig.colorbar(im, ax=ax[1], pad=0.02)

    ax[2].plot(ts * 1e6, P.max(axis=1) / A_inc, "-", color="#D55E00", lw=1.6, label="max p' / A")
    ax[2].plot(ts * 1e6, P.min(axis=1) / A_inc, "-", color="#0072B2", lw=1.6, label="min p' / A")
    ax[2].axvline(t_arr * 1e6, color="0.3", lw=0.8, ls="--")
    ax[2].axhline(0, color="0.8", lw=0.6)
    ax[2].set_ylim(-1.15, 1.15)
    ax[2].set_xlabel("t [$\\mu$s]"); ax[2].set_ylabel("domain extreme of p' / incident peak"); ax[2].grid(alpha=.25, lw=.6)
    ax[2].legend(fontsize=8, frameon=False, loc="center left")
    ax[2].set_title(f"R = {R:+.3f}   (relaxation estimate {R_th:+.3f})", fontsize=10)

    d_or_g = lambda key: "given" if key in kv else "default"
    src = f"sigma {d_or_g('nscbc.xhi.sigma')}, L_ref {d_or_g('nscbc.xhi.L_ref')}"
    fig.suptitle(f"NSCBC tube -- {case.replace('_', ', ')}:  sigma = {sig:g}, L_ref = {lref*1e3:g} mm ({src}),  "
                 f"K = {K:.2e} 1/s,  K pw/c = {K*pw/c:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, f"NSCBCTube_{case}.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary.append((case, A, sig, lref, K, K * pw / c, R, R_th, A_inc / A))
    print(f"wrote Images/NSCBCTube_{case}.png   R = {R:+.4f}  estimate {R_th:+.4f}")

print("\n%-15s %9s %7s %9s %10s %8s %9s %9s %9s" % ("case", "A [Pa]", "sigma", "L_ref", "K [1/s]", "K pw/c", "R", "R_est", "A_inc/A"))
for s in summary:
    print("%-15s %9.3g %7g %9.2e %10.2e %8.2f %+9.4f %+9.4f %9.4f" % s)
with open(os.path.join(IMG, "NSCBCTube_summary.txt"), "w") as fh:
    for s in summary:
        fh.write("%s %g %g %g %g %g %g %g %g\n" % s)
