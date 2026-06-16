# -*- coding: utf-8 -*-
"""
===============================================================================
FOURIER-SERIES PRIMITIVE-BC TEST -- analysis
===============================================================================
Validates tests/FlowAcoustic/input_AcousticBC_Fourier: a multi-frequency
(square-wave partial sum) pressure wave injected via the primitive BC path
(bc.primitive=1), WENO3 + RK3.

Each harmonic is a separate right-traveling d'Alembert wave at the SAME c0
(linear acoustics, non-dispersive), so the analytic reference is the series
shifted in space:

    S(tau) = A * sum_n coef_n * sin(n*w*tau)
    p(x,t) = p0 + S(t - x/c0) * H(t - x/c0)

If the BC + reconstruction are correct the sim reproduces the full waveform;
harmonic-dependent numerical dissipation shows up as the high harmonics damping
faster (the square-ish wave rounding off). No damped fit here -- WENO3 is low
dissipation, so we compare directly against the IDEAL Fourier wave.

Outputs (tests/FlowAcoustic/reference/Images/ and .../Data/):
  AcousticBC_Fourier_snapshots.png        -- p(x): sim vs ideal, several times
  AcousticBC_Fourier_xt.png               -- x-t diagram of p (sim)
  AcousticBC_Fourier.gif                  -- animated wave (sim vs ideal)
  AcousticBC_Fourier_error_snapshots.png  -- Delta p vs x (+ % axis), times
  AcousticBC_Fourier_error.gif            -- animated pressure error (+ % axis)
  AcousticBC_Fourier_error.csv            -- per-frame L2/Linf vs ideal

Runs BEFORE any simulation (ideal-wave snapshots only).

USAGE:
    python tests/FlowAcoustic/reference/AnalyzeBC_Fourier.py
===============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================  CONFIGURATION  ===============================
# MUST match input_AcousticBC_Fourier.
P_BASE = 1.0
AMP    = 0.107               # peak |S| ~ 0.93*A ~ 0.10 -> +/-0.1 swing
OMEGA  = 25.13274123         # 2*pi*f, f = 4
# NOTE: +/-0.1 on p0=1 is a ~10% perturbation -> NONLINEAR. The linear ideal
# Fourier wave below is the SMALL-amplitude reference; expect the sim to steepen
# (sharper fronts) and run slightly faster, so the error vs "ideal" grows
# downstream. That divergence is physics (full Euler), not a BC error.
C0     = 1.1832159566        # sqrt(gamma*p0/rho0) = sqrt(1.4)
HARMONICS = [(1, 1.0), (3, 1.0 / 3.0), (5, 1.0 / 5.0)]   # square-wave partial sum
X_LO   = 0.0
X_HI   = 1.0
T_END  = 0.7

CASE = "AcousticBC_Fourier"
OUTPUT_DIR = os.path.normpath(os.path.join(
    _HERE, "..", "..", "..", "bin", "tests", "FlowAcoustic", "output_" + CASE))
# OVERRIDE (uncomment + edit for your machine / scratch):
# OUTPUT_DIR = "/path/to/output_AcousticBC_Fourier"

N_SNAPSHOTS = 6
DPI = 150

IMG_DIR  = os.path.join(_HERE, "Images")
DATA_DIR = os.path.join(_HERE, "Data")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# ANALYTIC (Fourier superposition)
# ============================================================================
def S(tau):
    """Source waveform S(tau) = A * sum coef_n sin(n w tau)."""
    tau = np.asarray(tau, dtype=float)
    out = np.zeros_like(tau)
    for n, coef in HARMONICS:
        out += coef * np.sin(n * OMEGA * tau)
    return AMP * out


def p_ideal(x, t):
    """Ideal Fourier d'Alembert wave; quiescent ahead of the front x=c0*t."""
    x = np.asarray(x, dtype=float)
    tau = t - x / C0
    p = P_BASE + S(tau)
    p[tau < 0.0] = P_BASE
    return p


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
def _pct_axis(ax):
    sec = ax.secondary_yaxis(
        "right", functions=(lambda v: v * 100.0 / P_BASE,
                            lambda v: v * P_BASE / 100.0))
    sec.set_ylabel("% of $p_0$")
    return sec


# perturbation peak (for consistent y-limits)
def _ppk():
    tau = np.linspace(0, 2 * np.pi / OMEGA, 4000)
    return float(np.max(np.abs(S(tau)))) * 1.3


def write_error_csv(times, x, P):
    path = os.path.join(DATA_DIR, f"{CASE}_error.csv")
    with open(path, "w") as f:
        f.write("time_s,L2_vs_ideal,Linf_vs_ideal,front_x_c0t,reached_xhi\n")
        for t, p in zip(times, P):
            pa = p_ideal(x, t)
            l2 = float(np.sqrt(np.mean((p - pa) ** 2)))
            linf = float(np.max(np.abs(p - pa)))
            front = C0 * t
            f.write(f"{t:.9e},{l2:.9e},{linf:.9e},{front:.6f},"
                    f"{int(front >= X_HI)}\n")
    print(f"  wrote {path}")


def plot_snapshots(times, x, P):
    n = len(times)
    idx = np.linspace(0, n - 1, min(N_SNAPSHOTS, n)).astype(int)
    ylim = _ppk()
    ncol = 3
    nrow = int(np.ceil(len(idx) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    xa = np.linspace(X_LO, X_HI, 1200)
    for k, fi in enumerate(idx):
        ax = axes.flat[k]; ax.set_visible(True)
        t = times[fi]
        ax.plot(xa, p_ideal(xa, t), "k-", lw=1.4, label="ideal Fourier")
        ax.plot(x, P[fi], "r.", ms=3, alpha=0.7, label="sim")
        ax.axvline(C0 * t, color="0.6", ls=":", lw=1)
        ax.set_title(f"t = {t:.3f} s")
        ax.set_xlabel("x"); ax.set_ylabel("p")
        ax.set_ylim(P_BASE - ylim, P_BASE + ylim)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Acoustic primitive-BC (Fourier): pressure vs ideal",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_snapshots.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def plot_xt(times, x, P):
    pk = _ppk()
    fig, ax = plt.subplots(figsize=(8, 5))
    extent = [x[0], x[-1], times[0], times[-1]]
    im = ax.imshow(P - P_BASE, origin="lower", aspect="auto", extent=extent,
                   cmap="RdBu_r", vmin=-pk, vmax=pk)
    ax.plot(C0 * times, times, "k--", lw=1, label="front $x=c_0 t$")
    ax.set_xlabel("x"); ax.set_ylabel("t [s]")
    ax.set_title("x-t diagram of pressure perturbation (sim)")
    fig.colorbar(im, ax=ax, label="p - p0"); ax.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_xt.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def make_gif(times, x, P):
    xa = np.linspace(X_LO, X_HI, 1200)
    ylim = _ppk()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ln_i, = ax.plot([], [], "k-", lw=1.4, label="ideal Fourier")
    ln_s, = ax.plot([], [], "r.", ms=3, alpha=0.7, label="sim")
    vln = ax.axvline(0, color="0.6", ls=":", lw=1)
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(P_BASE - ylim, P_BASE + ylim)
    ax.set_xlabel("x"); ax.set_ylabel("p"); ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ttl = ax.set_title("")

    def upd(i):
        t = times[i]
        ln_i.set_data(xa, p_ideal(xa, t))
        ln_s.set_data(x, P[i])
        vln.set_xdata([C0 * t, C0 * t])
        ttl.set_text(f"Acoustic primitive-BC (Fourier)   t = {t:.3f} s")
        return ln_i, ln_s, vln, ttl

    anim = animation.FuncAnimation(fig, upd, frames=len(times), blit=False)
    out = os.path.join(IMG_DIR, f"{CASE}.gif")
    try:
        anim.save(out, writer=animation.PillowWriter(fps=12))
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  [warn] gif writer failed ({e}); skipping gif.")
    plt.close(fig)


def plot_error_snapshots(times, x, P):
    n = len(times)
    idx = np.linspace(0, n - 1, min(N_SNAPSHOTS, n)).astype(int)
    errs = [P[fi] - p_ideal(x, times[fi]) for fi in idx]
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
        err = P[fi] - p_ideal(x, t)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
        ax.plot(x, err, "m-", lw=1.2)
        ax.axvline(C0 * t, color="0.6", ls=":", lw=1)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\Delta p = p_{\rm sim}-p_{\rm ideal}$")
        ax.set_title(f"t = {t:.3f} s")
        ax.grid(True, alpha=0.3)
        _pct_axis(ax)
    fig.suptitle("Acoustic primitive-BC (Fourier): pressure error (ref = ideal)",
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(IMG_DIR, f"{CASE}_error_snapshots.png")
    fig.savefig(out, dpi=DPI, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {out}")


def make_error_gif(times, x, P):
    errs = [P[i] - p_ideal(x, times[i]) for i in range(len(times))]
    ymax = max((np.max(np.abs(e)) for e in errs), default=1e-12)
    ymax = max(ymax, 1e-12) * 1.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ln, = ax.plot([], [], "m-", lw=1.2)
    vln = ax.axvline(0, color="0.6", ls=":", lw=1)
    ax.set_xlim(X_LO, X_HI); ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("x"); ax.set_ylabel(r"$\Delta p = p_{\rm sim}-p_{\rm ideal}$")
    ax.grid(True, alpha=0.3)
    _pct_axis(ax)
    ttl = ax.set_title("")

    def upd(i):
        t = times[i]
        ln.set_data(x, P[i] - p_ideal(x, t))
        vln.set_xdata([C0 * t, C0 * t])
        ttl.set_text(f"Pressure error (ref=ideal Fourier)   t = {t:.3f} s")
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
    print("FOURIER PRIMITIVE-BC ANALYSIS")
    print("=" * 70)
    harm = " + ".join(f"{c:.3g}*sin({n}wt)" for n, c in HARMONICS)
    print(f"  S(t) = {AMP} * ({harm}),  w = {OMEGA} (f={OMEGA/2/np.pi:.3f})")
    print(f"  p_base={P_BASE} c0={C0}")
    print(f"  domain x in [{X_LO},{X_HI}], front reaches xhi at "
          f"t = {X_HI/C0:.3f} s (T_END={T_END})")
    print(f"  output dir = {OUTPUT_DIR}\n")

    times, x, P = extract_pressure(OUTPUT_DIR)
    if len(times) == 0:
        print("  [info] no simulation data -- ideal Fourier snapshots only.\n")
        xa = np.linspace(X_LO, X_HI, 1200)
        ts = np.linspace(0.05 * T_END, T_END, N_SNAPSHOTS)
        plot_snapshots(ts, xa, np.array([p_ideal(xa, t) for t in ts]))
        return

    print(f"  loaded {len(times)} frames, nx={len(x)}")
    pa = p_ideal(x, times[-1])
    print(f"  final-frame L2 vs ideal   = {np.sqrt(np.mean((P[-1]-pa)**2)):.3e}")
    print(f"  final-frame Linf vs ideal = {np.max(np.abs(P[-1]-pa)):.3e}")
    if C0 * times[-1] >= X_HI:
        print("  [note] front reached xhi -- expect reflection error late.")
    print()

    write_error_csv(times, x, P)
    plot_snapshots(times, x, P)
    plot_xt(times, x, P)
    make_gif(times, x, P)
    plot_error_snapshots(times, x, P)
    make_error_gif(times, x, P)


if __name__ == "__main__":
    main()
