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
import re
import glob
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
# Runs are DISCOVERED from the input filenames, which carry the drive in their
# name as input_f<freq in Hz>_A<wall amplitude in kPa> (all cases use NSCBC, so
# the tag is no longer part of the name).  The frequency and amplitude parsed
# from the name label the figure; the values actually used by the models are
# still read from the input file itself, so a mislabelled file cannot silently
# change a plotted model curve.
_RUN_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


def _parse_drive_name(fname):
    """input_f40.8_A2.72 -> (40.8, 2.72); returns (None, None) if not matching.
    Kept only as a fallback tag for inputs that carry no parsable drive."""
    m = re.match(r"input_f([0-9.]+)_A([0-9.]+)$", os.path.basename(fname))
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _slug(text):
    """Filesystem-safe tag fragment."""
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", text).strip("-")


def _discover_runs(pattern=None):
    """Every driven input under tests/FlowDrivenBubble, with the drive read from
    the FILE rather than from its name.

    The old behaviour required inputs to be named input_f<Hz>_A<kPa>; anything
    else was silently skipped.  Now any input_* is picked up and its frequency
    and amplitude come from nscbc.<face>.drive_{amp,omega} (or the primitive-BC
    keys), so a new run needs no naming convention and no edit here.  Inputs
    with no drive at all (ringdowns) are reported and skipped.

    pattern: optional glob, absolute or relative to tests/FlowDrivenBubble.
    """
    pattern = pattern or os.environ.get("DRIVE_GLOB") or "input_*"
    paths = sorted(glob.glob(pattern if os.path.isabs(pattern) else _tests(pattern)))
    runs, skipped = [], []
    for path in paths:
        if os.path.isdir(path) or path.endswith((".py", ".md", "~")):
            continue
        try:
            cfg = parse_input(path)
            A, w, _p, _how, _ph = drive_from_input(cfg, path)
        except Exception as exc:
            skipped.append((os.path.basename(path), f"unreadable ({exc})"))
            continue
        if not w or not A:
            skipped.append((os.path.basename(path), "no drive (amp or omega = 0)"))
            continue
        f_hz = w / (2.0 * math.pi)
        a_kpa = A / 1.0e3
        runs.append(dict(
            label=f"$f$ = {f_hz:.4g} Hz, $A$ = {a_kpa:.4g} kPa (wall)",
            tag=f"f{f_hz:.4g}_A{a_kpa:.4g}_{_slug(os.path.basename(path))}",
            input=path,
            color=_RUN_COLORS[len(runs) % len(_RUN_COLORS)],
        ))
    if skipped:
        print("  [discover] skipped:")
        for n, why in skipped:
            print(f"      {n:42s} {why}")
    return runs


RUNS = []   # populated in main(), once parse_input/drive_from_input exist

# ===== MODEL KNOBS =====
POLYTROPIC   = None    # gas exponent kappa; None -> use eos1.gamma (adiabatic)

# ---- FINITE-DOMAIN (CONFINEMENT) CORRECTION -------------------------------
# Rayleigh-Plesset and Keller-Miksis both assume an UNBOUNDED liquid.  These
# runs put the bubble in a box only a few R0 across, which removes most of the
# liquid inertia and stiffens the bubble.  Integrating the radial kinetic
# energy out to a finite outer radius L instead of infinity,
#     KE = 2 pi rho Rdot^2 R^3 (1 - R/L),
# and applying Lagrange's equation gives
#     rho[(1 - R/L) R Rddot + (3/2 - 2R/L) Rdot^2] = p_B - p_inf ,
# which reduces to the textbook RPE as L -> infinity and linearises to
#     f0_confined = f0_unbounded / sqrt(1 - R0/L).
#
# CONFINE  "off"            unbounded, the textbook models
#          "auto" (default)  L = radius of the sphere with the same volume as
#                           the simulation domain (symmetry planes unfolded)
#
# "auto" is the default because the confinement is REAL and measurable in
# these runs.  Switching the drive on abruptly at t = 0 rings the bubble at
# its natural frequency in BOTH the solver and the models, so f0 is directly
# observable: fitting a second tone to the f40.8_A2.72 record (516 frames,
# after removing the drive tone and the eta-smearing drift) puts the solver's
# free mode at 186 Hz +/- 10.  "auto" predicts 182.5 Hz; "off" predicts 163.1
# Hz and its ring walks out of phase with the solver's over the record.  KM
# rms against that run: 0.219% (off) -> 0.177% (auto) -> 0.171% (F0_OVERRIDE
# = 186).  Use "off" only to recover the textbook unbounded models.
# F0_OVERRIDE  <Hz>         measured natural frequency; the confinement ratio
#                           R0/L is back-solved so the model rings at exactly
#                           this frequency.  Overrides CONFINE.
# Both are read from the environment so no edit is needed per run.
CONFINE     = os.environ.get("CONFINE", "auto").lower()
F0_OVERRIDE = float(os.environ["F0_OVERRIDE"]) if os.environ.get("F0_OVERRIDE") else None
N_SUBSTEP    = 4000    # RK4 substeps per drive period (models only)
SHOW_RPE     = True
SHOW_KM      = True
SHAPE_GIF    = False   # the GIF is analyze_radius.py's job; off by default

# ===== PLOT =====
DPI, FIG_W, FIG_H = 180, 11.5, 6.2
SAVE_NAME  = "FlowDrivenBubble_radius_drivencompare"   # one figure per run: <SAVE_NAME>_<tag>.png
TITLE_STR  = "Flow-Driven Bubble: simulation vs driven RPE / Keller-Miksis"
# Analytical models are drawn in FIXED colours, never the run colour, so they
# cannot be mistaken for simulation output (run colours: blue / green).
COLOR_RPE, COLOR_KM = "black", "tab:red"

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
    amps, omegas, pinfs, faces, phases = [], [], [], [], []
    for face in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi"):
        a = _f(cfg, f"nscbc.{face}.drive_amp")
        w = _f(cfg, f"nscbc.{face}.drive_omega")
        if a is None and w is None:
            continue
        if (a or 0.0) == 0.0 and (w or 0.0) == 0.0:
            continue
        faces.append(face)
        amps.append(a or 0.0); omegas.append(w or 0.0)
        phases.append(_f(cfg, f"nscbc.{face}.drive_phase", 0.0) or 0.0)
        p = _f(cfg, f"nscbc.{face}.target_p")
        if p is not None:
            pinfs.append(p)
    if faces:
        if len(set(np.round(omegas, 9))) > 1 or len(set(np.round(amps, 9))) > 1:
            print(f"  [WARN] driven faces disagree in {os.path.basename(path)}:")
            for f_, a_, w_ in zip(faces, amps, omegas):
                print(f"           {f_}: amp={a_:g} omega={w_:g}")
        p_inf = pinfs[0] if pinfs else _f(cfg, "pressure0.ic.expression.constant.p_inf", 1.0e5)
        return amps[0], omegas[0], p_inf, f"nscbc drive on {','.join(faces)}", phases[0]

    # primitive-BC path (bc.primitive = 1): pressure.bc.constant.{amp,w}
    a = _f(cfg, "pressure.bc.constant.amp", 0.0)
    w = _f(cfg, "pressure.bc.constant.w") or _f(cfg, "pressure.bc.constant.omega", 0.0)
    p_inf = (_f(cfg, "pressure.bc.constant.p_inf")
             or _f(cfg, "pressure.bc.constant.p_amb")
             or _f(cfg, "pressure0.ic.expression.constant.p_inf", 1.0e5))
    return a or 0.0, w or 0.0, p_inf, "primitive pressure BC", 0.0


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
    ph   = P.get("phase", 0.0)          # solver: target_p + A sin(w t + phase)
    pdr  = p_inf + A * math.sin(w * t + ph)
    dpdr = A * w * math.cos(w * t + ph)

    # R/L for the finite-domain correction; 0 recovers the unbounded models.
    bl = P.get("beta", 0.0) * R / R0
    bl = min(bl, 0.95)                  # guard: L is an outer radius, not a wall

    if model == "rpe":
        return ((pw - pdr) / rho - (1.5 - 2.0 * bl) * Rd * Rd) / (R * (1.0 - bl))

    # Keller-Miksis.  d(pw)/dt carries a -4 mu Rddot / R term, which is moved
    # to the LHS so Rddot stays explicit.
    dpg  = -3.0 * kap * pg * Rd / R
    dpw_expl = dpg + 2.0 * sig * Rd / (R * R) + 4.0 * mu * Rd * Rd / (R * R)
    num = ((1.0 + Rd / c) * (pw - pdr) / rho
           + R / (rho * c) * (dpw_expl - dpdr)
           - 1.5 * (1.0 - Rd / (3.0 * c)) * Rd * Rd)
    # Confinement enters the inertia the same way it does in the RPE limit.
    # The compressible (Keller-Miksis) and finite-domain corrections are both
    # first order and are combined multiplicatively on the Rddot coefficient;
    # that is exact in each limit and leading-order when both act together.
    # Confinement changes ONLY the Rdot^2 coefficient, from the Keller-Miksis
    # 1.5(1 - Rd/3c) to [1.5(1 - Rd/3c) - 2 bl]: the finite-domain and
    # compressibility corrections are both first order and act on different
    # terms, so they superpose.  Adding 2 bl Rd^2 to the numerator is exactly
    # that substitution, and it leaves the Rdot^3/(2c) compressibility term
    # intact at bl = 0 (an earlier form overwrote the whole coefficient with
    # (1.5 - 2 bl) and silently dropped it).
    num = num + 2.0 * bl * Rd * Rd
    den = (1.0 - Rd / c) * (1.0 - bl) * R + 4.0 * mu / (rho * c)
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


def _domain_outer_radius(cfg):
    """Radius of the sphere with the same volume as the simulation domain.

    The driven cases are octants with symmetry planes through the bubble
    centre, so a lo edge sitting at 0 means that axis is only half represented;
    the volume is unfolded by 2 per such axis before converting to a radius.
    A cube has no single outer radius, so the volume-equivalent sphere is the
    natural isotropic stand-in for L in the radial-inertia integral.
    """
    try:
        lo = [float(v) for v in cfg.get("geometry.prob_lo", "").split()]
        hi = [float(v) for v in cfg.get("geometry.prob_hi", "").split()]
    except ValueError:
        return None
    if len(lo) < 3 or len(hi) < 3:
        return None
    vol, sym = 1.0, 1
    for d in range(3):
        ext = hi[d] - lo[d]
        if ext <= 0.0:
            return None
        vol *= ext
        if abs(lo[d]) < 1e-12:       # symmetry plane through the bubble centre
            sym *= 2
    return (3.0 * vol * sym / (4.0 * math.pi)) ** (1.0 / 3.0)


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
    A, w, p_inf, how, phase = drive_from_input(cfg, path)
    F = fluids_from_input(cfg)
    kappa = POLYTROPIC if POLYTROPIC is not None else F["gamma_g"]
    c = sound_speed(F["rho_l"], F["gamma_l"], F["pinf_l"], p_inf)
    # Laplace-equilibrium gas pressure at R0
    p_g0 = p_inf + 2.0 * F["sigma"] / F["R0"]
    P = dict(F); P.update(A=A, omega=w, p_inf=p_inf, kappa=kappa, c=c,
                          p_g0=p_g0, how=how, phase=phase, dim=dim_from_input(path, cfg),
                          plot_file=cfg.get("plot_file", ""))
    # --- finite-domain confinement ratio beta = R0/L -----------------------
    f0_unb = (1.0 / (2.0 * math.pi * F["R0"])) * math.sqrt(
        3.0 * kappa * p_inf / F["rho_l"])
    beta, how_c = 0.0, "off (unbounded)"
    if F0_OVERRIDE:
        r = f0_unb / F0_OVERRIDE
        beta = max(0.0, min(0.95, 1.0 - r * r))
        how_c = f"F0_OVERRIDE = {F0_OVERRIDE:g} Hz -> R0/L = {beta:.4f}"
    elif CONFINE == "auto":
        L = _domain_outer_radius(cfg)
        if L and L > F["R0"]:
            beta = min(0.95, F["R0"] / L)
            how_c = f"auto: L_eq = {L:.5g} m -> R0/L = {beta:.4f}"
        else:
            how_c = "auto requested but domain unreadable -- using unbounded"
    P["beta"] = beta
    P["confine"] = how_c
    P["f0_unbounded"] = f0_unb
    P["f_drive"] = w / (2.0 * math.pi) if w else 0.0
    P["T_drive"] = 1.0 / P["f_drive"] if P["f_drive"] else np.nan
    # Minnaert natural frequency (unbounded liquid, no tension correction)
    P["f0"] = (f0_unb / math.sqrt(1.0 - beta)) if beta else (
        1.0 / (2.0 * math.pi * F["R0"])) * math.sqrt(
        3.0 * kappa * p_inf / F["rho_l"])
    return P


def describe(P, label, path):
    print(f"  --- {label}")
    print(f"      input      : {os.path.basename(path)}   [{P['how']}]")
    print(f"      drive      : A = {P['A']:.4g} Pa, omega = {P['omega']:.4f} rad/s"
          f"  -> f = {P['f_drive']:.4f} Hz, T = {P['T_drive']*1e3:.4f} ms")
    print(f"      Minnaert f0= {P['f0']:.4f} Hz   (f_drive/f0 = "
          f"{P['f_drive']/P['f0']:.4f})")
    print(f"      confinement: {P.get('confine', 'off')}"
          + (f"   [unbounded f0 = {P['f0_unbounded']:.4f} Hz]"
             if P.get("beta") else ""))
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

def _resolve_out_dir(plot_file):
    """Locate the plotfiles for a run.

    The inputs carry the INCLINE path they were run with, e.g.
        plot_file = /mmfs1/home/ttryon/flames/bin/tests/FlowDrivenBubble/output_X
    That path is used as-is when it exists (i.e. on INCLINE).  Off-cluster the
    same basename is looked for under the local bin/ tree, so the identical
    input works in both places without editing.  DRIVE_OUT_ROOT overrides the
    search root; an explicit run["out_dir"] still wins over all of this.
    """
    if not plot_file:
        return ""
    if os.path.isdir(plot_file):
        return plot_file
    base = os.path.basename(plot_file.rstrip("/"))
    roots = []
    if os.environ.get("DRIVE_OUT_ROOT"):
        roots.append(os.environ["DRIVE_OUT_ROOT"])
    roots += [_repo("bin", "tests", "FlowDrivenBubble"),
              _repo("bin", "tests", "FlowDrivenBubble", "domaintest"),
              _repo("bin", "tests")]
    for r in roots:
        cand = os.path.join(r, base)
        if os.path.isdir(cand):
            return cand
    for r in roots:                      # one level down, for grouped runs
        for cand in sorted(glob.glob(os.path.join(r, "*", base))):
            if os.path.isdir(cand):
                return cand
    return plot_file                     # report the original path in the warning


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    global RUNS
    if not RUNS:
        RUNS = _discover_runs()
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
            run["out_dir"] = _resolve_out_dir(P["plot_file"])
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

    for run in loaded:
        plot_run(run)


def plot_run(run):
    """One figure per run: simulation (volume + eta=0.5 radius) against the
    driven RPE and Keller-Miksis solutions, with the applied drive below."""
    from matplotlib.lines import Line2D
    P = run["P"]
    ts = _tscale(P)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # ---- simulation (identical extraction to analyze_radius.py)
    t_end = 0.0
    n_pf = (len([q for q in os.listdir(run["out_dir"]) if q.endswith("cell")])
            if run["out_dir"] and os.path.isdir(run["out_dir"]) else 0)
    have_sim = False
    if n_pf >= 2:
        ar.R0 = P["R0"]; ar.DIM = P["dim"]; ar.R_BIN_MAX = 2.0 * P["R0"]
        if np.isfinite(P["T_drive"]):
            ar.T_DRIVE = P["T_drive"]; ar.F_DRIVE = P["f_drive"]
        print(f"  loading {run['out_dir']}  ({n_pf} plotfiles, dim={P['dim']}) ...")
        t, R_v, R_e, n_skip = ar.extract_radius_history(run["out_dir"])
        if n_skip:
            print(f"  [warn] skipped {n_skip} corrupt/partial plotfile(s).")
        if len(t) >= 2:
            have_sim = True
            t_end = float(t[-1])
            ar.print_extrema(f"{run['label']} (volume)", t, R_v)
            ar.print_extrema(f"{run['label']} (eta=0.5)", t, R_e)
            # ONE baseline convention for BOTH simulation curves.  Dividing
            # the volume curve by its own baseline while dividing the eta=0.5
            # curve by nominal R0 mixes conventions: the volume curve is then
            # pinned to 1 at the baseline frame while the eta=0.5 curve starts
            # wherever its contour happens to sit relative to R0.  On the
            # NONINT04 record that is a 0.022% offset (5% of the oscillation
            # amplitude) -- small, but it is pure convention, and it grows
            # with any IC whose eta=0.5 contour is not exactly at R0.
            # NOTE: this is NOT the diffuse-band bias.  The band inflates the
            # volume integral by ~0.34% of R0, but normalising the volume
            # curve by its own baseline already cancels that at the baseline
            # frame (it does not cancel the band's growth in time).
            if ar.R_VOL_BASELINE == "frame" and len(R_v) > ar.BASELINE_FRAME:
                base_v = float(R_v[ar.BASELINE_FRAME])
                base_e = float(R_e[ar.BASELINE_FRAME])
                if not np.isfinite(base_e) or base_e <= 0.0:
                    base_e = P["R0"]
                bnote = f" (baseline: frame {ar.BASELINE_FRAME})"
            else:
                base_v = base_e = P["R0"]
                bnote = " (baseline: nominal $R_0$)"
            ax.plot(t / ts, R_v / base_v, "-", color=run["color"], lw=2.4, marker="o",
                    ms=3, zorder=3, label="simulation: volume radius" + bnote)
            ax.plot(t / ts, R_e / base_e, "--", color=run["color"], lw=1.8,
                    zorder=4, label=r"simulation: $\eta=0.5$ radius" + bnote)
        else:
            print(f"  [WARN] {run['label']}: plotfiles present but no usable radius extracted.")
    else:
        why = ("no out_dir" if not run["out_dir"] else
               "directory does not exist" if not os.path.isdir(run["out_dir"]) else
               f"only {n_pf} plotfile(s)")
        print(f"  [WARN] {run['label']}: NO SIMULATION CURVES -- {why}: {run['out_dir']}")
        # Stamp it on the FIGURE too.  A models-only plot is a valid-looking
        # RPE/KM comparison with nothing to compare against, and the console
        # warning is easy to miss once a directory fills up with output.
        ax.text(0.5, 0.5, "MODELS ONLY\nno simulation data",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=26, color="0.55", alpha=0.30, rotation=18,
                zorder=10, fontweight="bold")
        ax.text(0.01, 0.015, f"expected plotfiles: {run['out_dir']}  ({why})",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=7, color="0.45", zorder=10)

    # ---- analytical models over the span this run covers
    Tref = P["T_drive"] if np.isfinite(P["T_drive"]) else 1.0e-2
    if t_end <= 0:
        t_end = 2.0 * Tref
    n_steps = int(N_SUBSTEP * max(1.0, t_end / Tref))
    print(f"  integrating models to t = {t_end*1e3:.4f} ms ({n_steps} RK4 steps)")
    if SHOW_RPE:
        tm, Rm = integrate("rpe", P, t_end, n_steps)
        ax.plot(tm / ts, Rm / P["R0"], ":", color=COLOR_RPE, lw=2.0, zorder=5,
                label="analytical: driven Rayleigh-Plesset")
        ar.print_extrema(f"RPE {run['label']}", tm, Rm)
    if SHOW_KM:
        tm, Rm = integrate("km", P, t_end, n_steps)
        ax.plot(tm / ts, Rm / P["R0"], "-.", color=COLOR_KM, lw=1.8, zorder=5,
                label="analytical: Keller-Miksis")
        ar.print_extrema(f"KM  {run['label']}", tm, Rm)

    undriven = not np.isfinite(P["T_drive"])
    ax.set_ylabel(r"$R / R_0$", fontsize=14)
    ax.set_title(f"{run['label']}:  simulation vs driven RPE / Keller-Miksis",
                 fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    ax.set_xlabel(r"$t$ [ms]" if undriven else r"$t / T_{drive}$", fontsize=14)
    ax.text(0.99, 0.02, f"drive: A = {P['A']:.4g} Pa,  f = {P['f_drive']:.4g} Hz,  "
            f"f/f0 = {P['f_drive']/P['f0']:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="0.3")
    plt.tight_layout()
    tag = run.get("tag") or "".join(ch for ch in run["label"] if ch.isalnum())[:24]
    for ext, dpi in (("png", DPI), ("eps", DPI)):
        out = os.path.join(IMG_DIR, f"{SAVE_NAME}_{tag}.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"  wrote {out}")
    if not have_sim:
        print("  [note] no simulation output for this run -- figure shows models only.")
    plt.close(fig)
    print()


if __name__ == "__main__":
    main()
