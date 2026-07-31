# -*- coding: utf-8 -*-
"""
===============================================================================
ACOUSTIC PRIMITIVE-BC TEST -- analysis
===============================================================================
Validates tests/FlowAcoustic/input_AcousticBC: a sinusoidal pressure wave
injected via the time-dependent primitive BC path (bc.primitive=1).

Compares the simulated pressure field p(x,t) against:
  - the IDEAL d'Alembert right-traveling wave (undamped):
        p(x,t) = p0 + amp * sin(omega*(t - x/c0)) * H(t - x/c0)
  - a DAMPED fit p0 + A0*exp(-alpha*x)*sin(omega*(t - x/c0)) where alpha is the
    spatial attenuation measured from the sim (Hilbert envelope, log-linear fit).

NOTE: with mu=0 the attenuation alpha is NUMERICAL dissipation (limiter order +
CFL + resolution), not physics. The damped fit gives a clean visual match AND
quantifies the scheme's dissipation -- a useful diagnostic. To physically reduce
alpha, raise the limiter order (Limiter.type=weno5, nghost=4), use RK3, and run
nearer CFL~0.8 (see header of input_AcousticBC / bin/PrimitiveBC.md notes).

Outputs (tests/FlowAcoustic/reference/Images/ and .../Data/), prefixed by case:
  <CASE>_snapshots.png       -- p(x): sim vs ideal vs damped-fit, several times
  <CASE>_xt.png              -- x-t diagram of p (sim)
  <CASE>.gif                 -- animated wave (sim, ideal, damped-fit)
  <CASE>_error_snapshots.png -- Delta p = p_sim - p_ref vs x (+ % axis), times
  <CASE>_error.gif           -- animated pressure error (+ % axis)
  <CASE>_error.csv           -- per-frame L2/Linf vs ideal AND vs damped, alpha

Runs BEFORE any simulation (draws the ideal wave only). Once plotfiles exist
under bin/tests/FlowAcoustic/output_AcousticBC they are auto-loaded.

USAGE:
    python tests/FlowAcoustic/reference/AcousticBC.py
===============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================  CONFIGURATION  ===============================
# MUST match input_AcousticBC.
P_BASE = 1.0
AMP    = 0.01
OMEGA  = 31.415926536        # 2*pi*f, f = 5
C0     = 1.1832159566        # sqrt(gamma*p0/rho0) = sqrt(1.4)
X_LO   = 0.0
X_HI   = 1.0
T_END  = 0.7

LAMBDA = 2.0 * np.pi * C0 / OMEGA   # wavelength

FIT_MARGIN = 0.05            # exclude this much near x=0 and near the front

# Case selector: optional CLI arg picks which run to analyze, e.g.
#   python AcousticBC.py                      -> output_AcousticBC          (baseline)
#   python AcousticBC.py AcousticBC_HighOrder -> output_AcousticBC_HighOrder
# Output images/CSV are named per-case so the two runs don't clobber each other.
CASE = sys.argv[1] if len(sys.argv) > 1 else "AcousticBC"

OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests", "FlowAcoustic", "output_" + CASE))
# OVERRIDE (uncomment + edit for your machine / scratch):
# OUTPUT_DIR = "/path/to/output_" + CASE

# Damped-envelope overlay (Option B). Useful for DISSIPATIVE runs (first-order
# baseline). For low-dissipation runs the wave is ~undamped, so compare against
# the IDEAL analytic only -> auto-OFF when the case looks high-order. Override
# with a 2nd CLI arg: "damped" forces it on, "ideal"/"nodamp" forces it off.
#   python AcousticBC.py AcousticBC_HighOrder            -> ideal only (auto)
#   python AcousticBC.py AcousticBC          ideal       -> ideal only (forced)
#   python AcousticBC.py AcousticBC_HighOrder damped     -> damped fit (forced)
FIT_DECAY = "highorder" not in CASE.lower()
if len(sys.argv) > 2:
    _opt = sys.argv[2].strip().lower()
    if _opt in ("damped", "fit", "decay"):
        FIT_DECAY = True
    elif _opt in ("ideal", "nodamp", "undamped"):
        FIT_DECAY = False

N_SNAPSHOTS = 6
DPI = 150

IMG_DIR  = os.path.join(_HERE, "Images")
DATA_DIR = os.path.join(_HERE, "Data")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# ANALYTIC
# ============================================================================
def p_ideal(x, t):
    """Undamped d'Alembert right-traveling wave; quiescent ahead of x=c0*t."""
    x = np.asarray(x, dtype=float)
    p = P_BASE + AMP * np.sin(OMEGA * (t - x / C0))
    p[(t - x / C0) < 0.0] = P_BASE
    return p


def p_damped(x, t, alpha, A0):
    """Spatially attenuated right-traveling wave: A0*exp(-alpha x)."""
    x = np.asarray(x, dtype=float)
    p = P_BASE + A0 * np.exp(-alpha * x) * np.sin(OMEGA * (t - x / C0))
    p[(t - x / C0) < 0.0] = P_BASE
    return p


def fit_decay(x, p, t):
    """Fit p-P_BASE envelope to A0*exp(-alpha x) via Hilbert + log-linear fit
    over the interior region behind the front. Returns (alpha, A0) or None."""
    try:
        from scipy.signal import hilbert
    except Exception:
        return None
    x_front = min(C0 * t, X_HI)
    mask = (x > X_LO + FIT_MARGIN) & (x < x_front - FIT_MARGIN)
    if mask.sum() < 10:
        return None
    env = np.abs(hilbert(np.asarray(p, dtype=float) - P_BASE))
    xm = x[mask]
    em = np.maximum(env[mask], 1e-14)
    coef = np.polyfit(xm, np.log(em), 1)          # log(env) = log(A0) - alpha*x
    alpha = float(-coef[0])
    A0 = float(np.exp(coef[1]))
    return alpha, A0


def choose_fit_frame(times):
    """Index of the frame with the longest propagation that hasn't yet reached
    the right boundary (cleanest decay fit)."""
    inside = [i for i, t in enumerate(times) if C0 * t <= X_HI]
    return inside[-1] if inside else len(times) - 1


# ============================================================================
# PLOTFILE EXTRACTION (robust; never raises on a single bad frame)
# ============================================================================
def extract_pressure(amrex_dir):
    """Return (times, x, P) with P shape (nframes, nx); empty if nothing loads."""
    try:
        import yt
    except Exception as e:
        print(f"  [warn] yt unavailable ({e}); analytic-only.")
        return np.array([]), np.array([]), np.zeros((0, 0))
    yt.funcs.mylog.setLevel(40)

    if not os.path.isdir(amrex_dir):
        return np.array([]), np.array([]), np.zeros((0, 0))
    pfs = sorted(
        os.path.join(amrex_dir, d) for d in os.listdir(amrex_dir)
        if os.path.isdir(os.path.join(amrex_dir, d)) and d.endswith("cell"))

    times, rows, xref, n_skip = [], [], None, 0
    for pf in pfs:
        try:
            ds = yt.load(pf)
            L = float(ds.domain_right_edge[0])
            ray = ds.ray(ds.arr([X_LO, 0.0, 0.0], "code_length"),
                         ds.arr([L,    0.0, 0.0], "code_length"))
            x = np.array(ray["x"])
            p = np.array(ray["pressure"])
            order = np.argsort(x)
            x, p = x[order], p[order]
            if xref is None:
                xref = x
            times.append(float(ds.current_time))
            rows.append(p)
        except Exception:
            n_skip += 1
            continue
    if n_skip:
        print(f"  [warn] skipped {n_skip} corrupt/partial plotfile(s)")
    if not times:
        return np.array([]), np.array([]), np.zeros((0, 0))
    times = np.array(times)
    o = np.argsort(times)
    return times[o], xref, np.array(rows)[o]


# ============================================================================
# OUTPUTS
# ============================================================================
def write_error_csv(times, x, P, alpha, A0):
    path = os.path.join(DATA_DIR, f"{CASE}_error.csv")
    with open(path, "w") as f:
        f.write("time_s,L2_vs_ideal,Linf_vs_ideal,L2_vs_damped,Linf_vs_damped,"
                "alpha_frame,A0_frame,front_x_c0t,reached_xhi\n")
        for t, p in zip(times, P):
            pi_ = p_ideal(x, t)
            pd_ = p_damped(x, t, alpha, A0)
            l2i = float(np.sqrt(np.mean((p - pi_) ** 2)))
            lii = float(np.max(np.abs(p - pi_)))
            l2d = float(np.sqrt(np.mean((p - pd_) ** 2)))
            lid = float(np.max(np.abs(p - pd_)))
            fr = fit_decay(x, p, t)
            af, a0f = (fr if fr else (np.nan, np.nan))
            front = C0 * t
            f.write(f"{t:.9e},{l2i:.9e},{lii:.9e},{l2d:.9e},{lid:.9e},"
                    f"{af:.9e},{a0f:.9e},{front:.6f},{int(front >= X_HI)}\n")
    print(f"  wrote {path}")


def plot_snapshots(times, x, P, alpha, A0, have_fit):
    n = len(times)
    idx = np.linspace(0, n - 1, min(N_SNAPSHOTS, n)).astype(int)
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    xa = np.linspace(X_LO, X_HI, 800)
    for k, fi in enumerate(idx):
        ax = axes.flat[k]; ax.set_visible(True)
        t = times[fi]
        ax.plot(xa, p_ideal(xa, t), "k--", lw=1.0, alpha=0.6, label="ideal")
        if have_fit:
            ax.plot(xa, p_damped(xa, t, alpha, A0), "b-", lw=1.6,
                    label="damped fit")
        ax.plot(x, P[fi], "r.", ms=3, alpha=0.7, label="sim")
        ax.axvline(C0 * t, color="0.6", ls=":", lw=1)
        ax.set_title(f"t = {t:.3f} s")
        ax.set_xlabel("x"); ax.set_ylabel("p")
        ax.set_ylim(P_BASE - 1.5 * AMP, P_BASE + 1.5 * AMP)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    sub = (f"  (fitted alpha={alpha:.3g} /len, decay length 1/alpha={1/alpha:.3g})"
           if have_fit and alpha > 0 else "")
    fig.suptitle("Acoustic Temporal-BC: Pressure Analyitical Comparison" + sub,
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_snapshots.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def plot_xt(times, x, P):
    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [x[0], x[-1], times[0], times[-1]]
    im = ax.imshow(P - P_BASE, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", vmin=-AMP, vmax=AMP)
    ax.plot(C0 * times, times, "k--", lw=1, label="front $x=c_0 t$")
    ax.set_xlabel("x"); ax.set_ylabel("t [s]")
    ax.set_title("x-t diagram of pressure perturbation (sim)")
    fig.colorbar(im, ax=ax, label="p - p0"); ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_xt.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def make_gif(times, x, P, alpha, A0, have_fit):
    xa = np.linspace(X_LO, X_HI, 800)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ln_i, = ax.plot([], [], "k--", lw=1.0, alpha=0.6, label="ideal")
    # Only label the damped fit (-> legend) when it's actually drawn.
    ln_d, = ax.plot([], [], "b-", lw=1.6,
                    label="damped fit" if have_fit else "_nolegend_")
    ln_s, = ax.plot([], [], "r.", ms=3, alpha=0.7, label="sim")
    vln = ax.axvline(0, color="0.6", ls=":", lw=1)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(P_BASE - 1.5 * AMP, P_BASE + 1.5 * AMP)
    ax.set_xlabel("x"); ax.set_ylabel("p"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ttl = ax.set_title("")

    def upd(i):
        t = times[i]
        ln_i.set_data(xa, p_ideal(xa, t))
        if have_fit:
            ln_d.set_data(xa, p_damped(xa, t, alpha, A0))
        ln_s.set_data(x, P[i])
        vln.set_xdata([C0 * t, C0 * t])
        ttl.set_text(f"Acoustic primitive-BC   t = {t:.3f} s")
        return ln_i, ln_d, ln_s, vln, ttl

    anim = animation.FuncAnimation(fig, upd, frames=len(times), blit=False)
    out = os.path.join(IMG_DIR, f"{CASE}.gif")
    try:
        anim.save(out, writer=animation.PillowWriter(fps=12))
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  [warn] gif writer failed ({e}); skipping gif.")
    plt.close(fig)


# ============================================================================
# ERROR PLOTS  (Delta p = p_sim - p_ref, with a secondary % axis)
# ref = damped fit (baseline) or ideal (high-order), matching the main compare.
# Since p ~ p_base, the % axis (Delta p * 100 / p_base) equals the local
# relative error to < 1%.
# ============================================================================
def _pct_axis(ax):
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * 100.0 / P_BASE,
                            lambda v: v * P_BASE / 100.0))
    sec.set_ylabel("% of $p_0$")
    return sec


def plot_error_snapshots(times, x, P, ref_fn, ref_label):
    n = len(times)
    idx = np.linspace(0, n - 1, min(N_SNAPSHOTS, n)).astype(int)
    errs = [P[fi] - ref_fn(x, times[fi]) for fi in idx]
    ymax = max((np.max(np.abs(e)) for e in errs), default=1e-12)
    ymax = max(ymax, 1e-12) * 1.2
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, fi in enumerate(idx):
        ax = axes.flat[k]; ax.set_visible(True)
        t = times[fi]
        err = P[fi] - ref_fn(x, t)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.plot(x, err, "m-", lw=1.2)
        ax.axvline(C0 * t, color="0.6", ls=":", lw=1)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\Delta p = p_{\rm sim}-p_{\rm ref}$")
        ax.set_title(f"t = {t:.3f} s")
        ax.grid(True, alpha=0.3)
        _pct_axis(ax)
    fig.suptitle(f"Acoustic primitive-BC: pressure error  (ref = {ref_label})",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_error_snapshots.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def make_error_gif(times, x, P, ref_fn, ref_label):
    errs = [P[i] - ref_fn(x, times[i]) for i in range(len(times))]
    ymax = max((np.max(np.abs(e)) for e in errs), default=1e-12)
    ymax = max(ymax, 1e-12) * 1.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ln, = ax.plot([], [], "m-", lw=1.2)
    vln = ax.axvline(0, color="0.6", ls=":", lw=1)
    ax.set_xlim(X_LO, X_HI); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\Delta p = p_{\rm sim}-p_{\rm ref}$")
    ax.grid(True, alpha=0.3)
    _pct_axis(ax)
    ttl = ax.set_title("")

    def upd(i):
        t = times[i]
        ln.set_data(x, P[i] - ref_fn(x, t))
        vln.set_xdata([C0 * t, C0 * t])
        ttl.set_text(f"Pressure error (ref={ref_label})   t = {t:.3f} s")
        return ln, vln, ttl

    anim = animation.FuncAnimation(fig, upd, frames=len(times), blit=False)
    out = os.path.join(IMG_DIR, f"{CASE}_error.gif")
    try:
        anim.save(out, writer=animation.PillowWriter(fps=12))
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  [warn] error-gif writer failed ({e}); skipping.")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print(f"ACOUSTIC PRIMITIVE-BC ANALYSIS  --  case: {CASE}")
    print("=" * 70)
    print(f"  p_base={P_BASE} amp={AMP} omega={OMEGA} (f={OMEGA/2/np.pi:.3f}) "
          f"c0={C0}  lambda={LAMBDA:.4f}")
    print(f"  domain x in [{X_LO},{X_HI}], front reaches xhi at "
          f"t = {X_HI/C0:.3f} s (T_END={T_END})")
    print(f"  output dir = {OUTPUT_DIR}\n")

    times, x, P = extract_pressure(OUTPUT_DIR)
    if len(times) == 0:
        print("  [info] no simulation data -- ideal-wave snapshots only.\n")
        xa = np.linspace(X_LO, X_HI, 800)
        ts = np.linspace(0.05 * T_END, T_END, N_SNAPSHOTS)
        plot_snapshots(ts, xa, np.array([p_ideal(xa, t) for t in ts]),
                       0.0, AMP, have_fit=False)
        return

    print(f"  loaded {len(times)} frames, nx={len(x)}")

    # ---- decay fit (global alpha from the cleanest frame) ------------------
    alpha, A0, have_fit = 0.0, AMP, False
    if FIT_DECAY:
        fi = choose_fit_frame(times)
        fr = fit_decay(x, P[fi], times[fi])
        if fr is not None and fr[0] == fr[0]:   # not NaN
            alpha, A0 = fr
            have_fit = alpha > 0
    if have_fit:
        per_lambda = np.exp(-alpha * LAMBDA)
        print(f"  decay fit (frame t={times[choose_fit_frame(times)]:.3f}): "
              f"alpha = {alpha:.4g} /len")
        print(f"    decay length 1/alpha = {1/alpha:.4g}  "
              f"(amp falls to 1/e over {1/alpha:.3g} in x)")
        print(f"    amplitude per wavelength = exp(-alpha*lambda) = "
              f"{per_lambda:.4f}  ({(1-per_lambda)*100:.2f}% loss / wavelength)")
        print(f"    fitted A0 = {A0:.4g} (boundary amp = {AMP}; "
              f"ratio {A0/AMP:.3f})")
        print(f"    NOTE: mu=0 -> this is NUMERICAL dissipation (limiter/CFL).")
    elif not FIT_DECAY:
        print("  [info] damped fit OFF for this case -- comparing vs IDEAL "
              "analytic only.")
    else:
        print("  [info] decay fit unavailable (scipy missing or too few pts).")

    pi_ = p_ideal(x, times[-1])
    pd_ = p_damped(x, times[-1], alpha, A0)
    print(f"\n  final-frame L2 vs ideal  = {np.sqrt(np.mean((P[-1]-pi_)**2)):.3e}")
    if have_fit:
        print(f"  final-frame L2 vs damped = "
              f"{np.sqrt(np.mean((P[-1]-pd_)**2)):.3e}  (should be << vs ideal)")
    print()

    # Reference for the error plots = the same curve the snapshots compare to:
    # the damped fit when fitting (baseline), else the ideal d'Alembert wave.
    if have_fit:
        ref_fn = lambda xx, tt: p_damped(xx, tt, alpha, A0)
        ref_label = "damped fit"
    else:
        ref_fn = lambda xx, tt: p_ideal(xx, tt)
        ref_label = "ideal"

    write_error_csv(times, x, P, alpha, A0)
    plot_snapshots(times, x, P, alpha, A0, have_fit)
    plot_xt(times, x, P)
    make_gif(times, x, P, alpha, A0, have_fit)
    plot_error_snapshots(times, x, P, ref_fn, ref_label)
    make_error_gif(times, x, P, ref_fn, ref_label)


if __name__ == "__main__":
    main()
