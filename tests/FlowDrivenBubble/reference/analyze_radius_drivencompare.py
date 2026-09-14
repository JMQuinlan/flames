#!/usr/bin/env python3
# ============================================================================
# FLOW-DRIVEN BUBBLE -- R(t) WITH DRIVEN RPE + KELLER-MIKSIS COMPARISON
# ============================================================================
# Companion to analyze_radius.py.  It produces the SAME simulation curves
# (volume radius + eta=0.5 contour radius, same extraction code, imported so
# the two scripts can never drift apart) and overlays two analytical models
# driven by the SAME sinusoidal far-field pressure the simulation sees:
#
#   RPE  -- driven Rayleigh-Plesset (incompressible liquid)
#   KM   -- Keller-Miksis (first-order liquid compressibility, sound speed c)
#
# The point of running both: RPE and KM agree while |Rdot| << c and separate
# once the wall speed becomes an appreciable fraction of the sound speed.  A
# simulation that tracks KM but not RPE is resolving acoustic radiation
# damping; one that tracks neither is not doing bubble dynamics at all.
#
# ---------------------------------------------------------------------------
# THE DRIVE IS READ FROM THE INPUT FILE -- NOT HARDCODED.
# ---------------------------------------------------------------------------
# Every run below names its Alamo input file.  The script parses it for the
# drive amplitude/frequency and the fluid properties, so the models are always
# driven by exactly what the solver was told to do.  Keys understood:
#
#   NSCBC drive path   nscbc.<face>.drive_amp / .drive_omega / .target_p
#   primitive BC path  pressure.bc.constant.{amp,w,omega,p_inf,p_amb}
#
# All driven faces are checked and must agree; a mismatch is reported loudly
# rather than silently averaged.  input_LowAmp_NSCBC and input_LowerAmp_NSCBC
# both carry drive_omega = 512.35 rad/s (f = 81.54 Hz = f0/2) and differ only
# in amplitude (1.0e4 vs 1.36e3 Pa), so they share an x-axis in t/T_drive.
#
# Usage:   cd <repo>/bin && python3 ../tests/FlowDrivenBubble/reference/analyze_radius_drivencompare.py
# ============================================================================

import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import analyze_radius as ar          # reuse the extraction verbatim

# ============================================================================
# ============================  CONFIGURATION  ===============================
# ============================================================================

# Root of the bin tree holding the plotfiles.  INCLINE by default; override
# with DRIVEN_BIN_ROOT (e.g. the local repo's bin) without editing the script.
BIN_ROOT = os.environ.get("DRIVEN_BIN_ROOT", "/mmfs1/home/ttryon/flames/bin")
#BIN_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "bin"))

def _repo(*p):
    # NOTE: must join *p.  Returning BIN_ROOT alone points every run at the bin
    # root, finds no plotfiles, and silently plots the models only.
    return os.path.normpath(os.path.join(BIN_ROOT, *p))


def _tests(*p):
    return os.path.normpath(os.path.join(_HERE, "..", *p))

# Each run: label, the input file that produced it, colour, and optionally
# out_dir.  When out_dir is omitted it is read from the input's own
# `plot_file =` line -- the same single source of truth as the drive -- so the
# plotfile path cannot drift from what the solver actually wrote.
RUNS = [
    dict(label="LowAmp NSCBC (A = 1.0e4 Pa)",
         input=_tests("input_LowAmp_NSCBC"),
         color="tab:blue"),
    dict(label="LowerAmp NSCBC (A = 1.36e3 Pa)",
         input=_tests("input_LowerAmp_NSCBC"),
         color="tab:green"),
    # explicit override example:
    # dict(label="...", input=_tests("input_LowAmp_NSCBC"), color="tab:red",
    #      out_dir=_repo("tests", "FlowDrivenBubble", "output_LowAmp_NSCBC")),
]

# ===== MODEL KNOBS =====
POLYTROPIC   = None    # gas exponent kappa; None -> use eos1.gamma (adiabatic)
N_SUBSTEP    = 4000    # RK4 substeps per drive period (models only)
SHOW_RPE     = True
SHOW_KM      = True
SHOW_DRIVE   = True    # lower panel: p_inf(t) actually applied
SHAPE_GIF    = False   # the GIF is analyze_radius.py's job; off by default

# ===== PLOT =====
DPI, FIG_W, FIG_H = 180, 11.5, 7.8
SAVE_NAME  = "FlowDrivenBubble_radius_drivencompare"
TITLE_STR  = "Flow-Driven Bubble: simulation vs driven RPE / Keller-Miksis"
COLOR_RPE, COLOR_KM = "0.35", "tab:orange"

IMG_DIR = os.path.join(_HERE, "Images")

# ============================================================================
# INPUT-FILE PARSING
# ============================================================================

def parse_input(path):
    """Alamo/AMReX input file -> {key: value_string}.  Strips comments and
    quotes; later assignments win (matching ParmParse)."""
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def _f(cfg, key, default=None):
    if key not in cfg:
        return default
    try:
        return float(cfg[key].split()[0])
    except (ValueError, IndexError):
        return default


def drive_from_input(cfg, path):
    """Return (amp, omega, p_inf, how).  Scans every driven face; all faces
    that specify a drive must agree."""
    amps, omegas, pinfs, faces = [], [], [], []
    for face in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"):
        a = _f(cfg, f"nscbc.{face}.drive_amp")
        w = _f(cfg, f"nscbc.{face}.drive_omega")
        if a is None and w is None:
            continue
        if (a or 0.0) == 0.0 and (w or 0.0) == 0.0:
            continue
        faces.append(face)
        amps.append(a or 0.0); omegas.append(w or 0.0)
        p = _f(cfg, f"nscbc.{face}.target_p")
        if p is not None:
            pinfs.append(p)
    if faces:
        if len(set(np.round(omegas, 9))) > 1 or len(set(np.round(amps, 9))) > 1:
            print(f"  [WARN] driven faces disagree in {os.path.basename(path)}:")
            for f_, a_, w_ in zip(faces, amps, omegas):
                print(f"           {f_}: amp={a_:g} omega={w_:g}")
        p_inf = pinfs[0] if pinfs else _f(cfg, "pressure0.ic.expression.constant.p_inf", 1.0e5)
        return amps[0], omegas[0], p_inf, f"nscbc drive on {','.join(faces)}"

    # primitive-BC path (bc.primitive = 1): pressure.bc.constant.{amp,w}
    a = _f(cfg, "pressure.bc.constant.amp", 0.0)
    w = _f(cfg, "pressure.bc.constant.w") or _f(cfg, "pressure.bc.constant.omega", 0.0)
    p_inf = (_f(cfg, "pressure.bc.constant.p_inf")
             or _f(cfg, "pressure.bc.constant.p_amb")
             or _f(cfg, "pressure0.ic.expression.constant.p_inf", 1.0e5))
    return a or 0.0, w or 0.0, p_inf, "primitive pressure BC"


def fluids_from_input(cfg):
    """Liquid/gas properties + R0, all straight from the input file."""
    rho_l = _f(cfg, "density0.ic.expression.region0", 1000.0)
    g_l   = _f(cfg, "eos0.gamma", 2.35)
    p0_l  = _f(cfg, "eos0.p0", 1.0e9)
    g_g   = _f(cfg, "eos1.gamma", 1.4)
    R0    = _f(cfg, "eta.ic.expression.constant.R0", 0.02)
    sigma = _f(cfg, "sigma", 0.0)
    mu    = _f(cfg, "mu0", 0.0)
    return dict(rho_l=rho_l, gamma_l=g_l, pinf_l=p0_l,
                gamma_g=g_g, R0=R0, sigma=sigma, mu=mu)


def sound_speed(rho_l, gamma_l, pinf_l, p):
    """Tammann/stiffened-gas liquid sound speed."""
    return math.sqrt(gamma_l * (p + pinf_l) / rho_l)

# ============================================================================
# DRIVEN BUBBLE MODELS
# ============================================================================

def _p_gas(R, p_g0, R0, kappa):
    return p_g0 * (R0 / R) ** (3.0 * kappa)


def _rhs(t, R, Rd, P, model):
    """Return Rddot for 'rpe' or 'km'."""
    rho, c, sig, mu = P["rho_l"], P["c"], P["sigma"], P["mu"]
    R0, kap, p_g0   = P["R0"], P["kappa"], P["p_g0"]
    A, w, p_inf     = P["A"], P["omega"], P["p_inf"]

    pg   = _p_gas(R, p_g0, R0, kap)
    pw   = pg - 2.0 * sig / R - 4.0 * mu * Rd / R      # liquid-side wall pressure
    pdr  = p_inf + A * math.sin(w * t)
    dpdr = A * w * math.cos(w * t)

    if model == "rpe":
        return ((pw - pdr) / rho - 1.5 * Rd * Rd) / R

    # Keller-Miksis.  d(pw)/dt carries a -4 mu Rddot / R term, which is moved
    # to the LHS so Rddot stays explicit.
    dpg  = -3.0 * kap * pg * Rd / R
    dpw_expl = dpg + 2.0 * sig * Rd / (R * R) + 4.0 * mu * Rd * Rd / (R * R)
    num = ((1.0 + Rd / c) * (pw - pdr) / rho
           + R / (rho * c) * (dpw_expl - dpdr)
           - 1.5 * (1.0 - Rd / (3.0 * c)) * Rd * Rd)
    den = (1.0 - Rd / c) * R + 4.0 * mu / (rho * c)
    return num / den


def integrate(model, P, t_end, n_steps):
    """Fixed-step RK4.  Returns (t, R).  Bails out (NaN-padded) if the bubble
    collapses below 1e-4 R0 -- the models are singular there, the solver is not."""
    dt = t_end / n_steps
    t  = np.linspace(0.0, t_end, n_steps + 1)
    R  = np.full(n_steps + 1, np.nan)
    y  = np.array([P["R0"], 0.0])
    R[0] = y[0]
    for i in range(n_steps):
        ti = t[i]
        def f(tt, yy):
            return np.array([yy[1], _rhs(tt, yy[0], yy[1], P, model)])
        try:
            k1 = f(ti,            y)
            k2 = f(ti + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = f(ti + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = f(ti + dt,       y + dt * k3)
            y  = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        except (OverflowError, ZeroDivisionError, ValueError):
            break
        if not np.isfinite(y[0]) or y[0] < 1.0e-4 * P["R0"]:
            break
        R[i + 1] = y[0]
    return t, R


def dim_from_input(path, cfg):
    """Spatial dimension: the '#@ dim=N' header, else inferred from prob_hi."""
    with open(path) as fh:
        for line in fh:
            if line.startswith("#@") and "dim=" in line:
                try:
                    return int(line.split("dim=", 1)[1].split()[0])
                except ValueError:
                    pass
    hi = cfg.get("geometry.prob_hi", "").split()
    return 2 if len(hi) >= 3 and float(hi[2]) == 0.0 else 3


def build_params(cfg, path):
    A, w, p_inf, how = drive_from_input(cfg, path)
    F = fluids_from_input(cfg)
    kappa = POLYTROPIC if POLYTROPIC is not None else F["gamma_g"]
    c = sound_speed(F["rho_l"], F["gamma_l"], F["pinf_l"], p_inf)
    # Laplace-equilibrium gas pressure at R0
    p_g0 = p_inf + 2.0 * F["sigma"] / F["R0"]
    P = dict(F); P.update(A=A, omega=w, p_inf=p_inf, kappa=kappa, c=c,
                          p_g0=p_g0, how=how, dim=dim_from_input(path, cfg),
                          plot_file=cfg.get("plot_file", ""))
    P["f_drive"] = w / (2.0 * math.pi) if w else 0.0
    P["T_drive"] = 1.0 / P["f_drive"] if P["f_drive"] else np.nan
    # Minnaert natural frequency (unbounded liquid, no tension correction)
    P["f0"] = (1.0 / (2.0 * math.pi * F["R0"])) * math.sqrt(
        3.0 * kappa * p_inf / F["rho_l"])
    return P


def describe(P, label, path):
    print(f"  --- {label}")
    print(f"      input      : {os.path.basename(path)}   [{P['how']}]")
    print(f"      drive      : A = {P['A']:.4g} Pa, omega = {P['omega']:.4f} rad/s"
          f"  -> f = {P['f_drive']:.4f} Hz, T = {P['T_drive']*1e3:.4f} ms")
    print(f"      Minnaert f0= {P['f0']:.4f} Hz   (f_drive/f0 = "
          f"{P['f_drive']/P['f0']:.4f})")
    print(f"      liquid     : rho = {P['rho_l']:.1f}, c = {P['c']:.2f} m/s "
          f"(gamma={P['gamma_l']}, p_inf_EOS={P['pinf_l']:.3g})")
    print(f"      gas        : kappa = {P['kappa']}, p_g0 = {P['p_g0']:.6g} Pa")
    print(f"      R0 = {P['R0']:.6g} m, sigma = {P['sigma']:.4g}, mu = {P['mu']:.4g}, dim = {P['dim']}")
    lin = P["A"] / (3.0 * P["kappa"] * P["p_inf"]
                    * abs(1.0 - (P["f_drive"] / P["f0"]) ** 2))
    print(f"      linear dR/R0 (unbounded) = {lin:.4e}")

# ============================================================================
# MAIN
# ============================================================================

def _tscale(P):
    """Time normaliser: T_drive for driven runs, milliseconds otherwise."""
    return P["T_drive"] if np.isfinite(P["T_drive"]) else 1.0e-3

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    print("=" * 74)
    print("FLOW-DRIVEN BUBBLE -- SIMULATION vs DRIVEN RPE / KELLER-MIKSIS")
    print("=" * 74)

    loaded = []
    for run in RUNS:
        if not os.path.isfile(run["input"]):
            print(f"  [skip] missing input file {run['input']}")
            continue
        cfg = parse_input(run["input"])
        P = build_params(cfg, run["input"])
        describe(P, run["label"], run["input"])
        if not run.get("out_dir"):
            run["out_dir"] = P["plot_file"]
        print(f"      plotfiles  : {run['out_dir']}")
        if P["omega"] == 0.0:
            print("      [WARN] no drive found in this input -- models will be static.")
        run["P"] = P
        loaded.append(run)
        print()

    if not loaded:
        print("  nothing to do."); return

    # The runs are meant to share a drive frequency; say so explicitly.
    fs = sorted({round(r["P"]["f_drive"], 6) for r in loaded})
    if len(fs) == 1:
        print(f"  all runs share f_drive = {fs[0]:.4f} Hz -- common t/T axis.\n")
    else:
        print(f"  [WARN] runs differ in drive frequency: {fs} Hz. "
              f"t/T uses each run's own T_drive.\n")

    nrow = 2 if SHOW_DRIVE else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(FIG_W, FIG_H), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1][:nrow]))
    ax = axes[0] if nrow > 1 else axes
    axd = axes[1] if nrow > 1 else None

    any_sim = False
    t_end_seen = 0.0
    for run in loaded:
        P = run["P"]
        # ---- simulation curves (identical extraction to analyze_radius.py)
        n_pf = (len([q for q in os.listdir(run["out_dir"]) if q.endswith("cell")])
                if run["out_dir"] and os.path.isdir(run["out_dir"]) else 0)
        if n_pf >= 2:
            ar.R0 = P["R0"]
            ar.DIM = P["dim"]
            ar.R_BIN_MAX = 2.0 * P["R0"]      # eta(r) profile out to 2 R0
            if np.isfinite(P["T_drive"]):
                ar.T_DRIVE = P["T_drive"]
                ar.F_DRIVE = P["f_drive"]
            print(f"  loading {run['out_dir']}  ({n_pf} plotfiles, dim={P['dim']}) ...")
            t, R_v, R_e, n_skip = ar.extract_radius_history(run["out_dir"])
            if n_skip:
                print(f"  [warn] skipped {n_skip} corrupt/partial plotfile(s).")
            if len(t) < 2:
                print(f"  [WARN] {run['label']}: plotfiles present but no usable radius extracted.")
            if len(t) >= 2:
                any_sim = True
                t_end_seen = max(t_end_seen, float(t[-1]))
                ar.print_extrema(f"{run['label']} (volume)", t, R_v)
                ar.print_extrema(f"{run['label']} (eta=0.5)", t, R_e)
                base = (float(R_v[ar.BASELINE_FRAME])
                        if ar.R_VOL_BASELINE == "frame" and len(R_v) > ar.BASELINE_FRAME
                        else P["R0"])
                ts = _tscale(P)
                ax.plot(t / ts, R_v / base, "-", color=run["color"], lw=2.2,
                        marker="o", ms=2.5, label=f"{run['label']} -- sim volume radius")
                ax.plot(t / ts, R_e / P["R0"], "--", color=run["color"], lw=1.6,
                        alpha=0.9, label=f"{run['label']} -- sim $\\eta=0.5$ radius")
        else:
            why = ("no out_dir" if not run["out_dir"] else
                   "directory does not exist" if not os.path.isdir(run["out_dir"]) else
                   f"only {n_pf} plotfile(s)")
            print(f"  [WARN] {run['label']}: NO SIMULATION CURVES -- {why}: {run['out_dir']}")

    # ---- models: integrate over the span the simulations actually cover
    #      (fall back to 2 drive periods when nothing has been run yet)
    P0 = loaded[0]["P"]
    Tref = P0["T_drive"] if np.isfinite(P0["T_drive"]) else 1.0e-2
    t_end = t_end_seen if t_end_seen > 0 else 2.0 * Tref
    n_per = max(1.0, t_end / Tref)
    n_steps = int(N_SUBSTEP * n_per)
    print(f"\n  integrating models to t = {t_end*1e3:.4f} ms "
          f"({n_per:.2f} drive periods, {n_steps} RK4 steps)")

    for run in loaded:
        P = run["P"]
        short = run['label'].split('(')[0].strip()
        ts = _tscale(P)
        if SHOW_RPE:
            tm, Rm = integrate("rpe", P, t_end, n_steps)
            ax.plot(tm / ts, Rm / P["R0"], ":", color=run["color"], lw=1.9,
                    label=f"{short} -- driven RPE")
            ar.print_extrema(f"RPE {run['label']}", tm, Rm)
        if SHOW_KM:
            tm, Rm = integrate("km", P, t_end, n_steps)
            ax.plot(tm / ts, Rm / P["R0"], "-.", color=run["color"], lw=1.6, alpha=0.85,
                    label=f"{short} -- Keller-Miksis")
            ar.print_extrema(f"KM  {run['label']}", tm, Rm)

    ax.set_ylabel(r"$R / R_0$", fontsize=14)
    ax.set_title(TITLE_STR, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    style_key = [Line2D([], [], color="0.2", ls="-", marker="o", ms=2.5, lw=2.2, label="sim: volume radius"),
                 Line2D([], [], color="0.2", ls="--", lw=1.6, label=r"sim: $\eta=0.5$ radius"),
                 Line2D([], [], color="0.2", ls=":", lw=1.9, label="driven RPE"),
                 Line2D([], [], color="0.2", ls="-.", lw=1.6, label="Keller-Miksis")]
    leg2 = ax.legend(handles=style_key, fontsize=8.5, loc="lower right", title="line style",
                     title_fontsize=8.5, framealpha=0.9)
    ax.add_artist(leg2)
    ax.legend(fontsize=8.5, loc="upper left", ncol=2, framealpha=0.9)

    if axd is not None:
        tt = np.linspace(0.0, t_end, 2000)
        for run in loaded:
            P = run["P"]
            axd.plot(tt / _tscale(P),
                     (P["p_inf"] + P["A"] * np.sin(P["omega"] * tt)) / P["p_inf"],
                     "-", color=run["color"], lw=1.4, label=run["label"])
        axd.axhline(1.0, color="0.6", lw=0.8, ls="--")
        axd.set_ylabel(r"$p_\infty(t)\,/\,p_\infty$", fontsize=12)
        axd.grid(True, alpha=0.3)
        axd.legend(fontsize=8, loc="upper right")
    undriven = any(not np.isfinite(r["P"]["T_drive"]) for r in loaded)
    (axd if axd is not None else ax).set_xlabel(
        r"$t$ [ms]" if undriven else r"$t / T_{drive}$", fontsize=14)

    plt.tight_layout()
    png = os.path.join(IMG_DIR, f"{SAVE_NAME}.png")
    eps = os.path.join(IMG_DIR, f"{SAVE_NAME}.eps")
    fig.savefig(png, dpi=DPI, bbox_inches="tight")
    fig.savefig(eps, dpi=DPI, bbox_inches="tight")
    print(f"\n  wrote {png}")
    print(f"  wrote {eps}")
    if not any_sim:
        print("  [note] no simulation output found -- plot shows models only.")
    plt.close(fig)


if __name__ == "__main__":
    main()
