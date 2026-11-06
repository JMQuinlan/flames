#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Sch20 MARMOTTANT COATED-BUBBLE ANALYSIS -- shared module
===============================================================================
Driven by analyze_Sch20_Oscillating_Marmottant.py and
analyze_Sch20_Collapsing_Marmottant.py, one per case.  Nothing here runs on
import; call run_case().

Reads one of

    tests/FlowMarmottant/input_Sch20-Oscillating_Marmottant
    tests/FlowMarmottant/input_Sch20-Collapsing_Marmottant

parses EVERY physical parameter out of it (no hard-coded physics), extracts the
bubble radius history from the AMReX plotfiles, and overlays it on the
Marmottant-modified Rayleigh--Plesset and Keller--Miksis references from
reference/marmottant_rpe_km.py.

This mirrors tests/FlowRayleighPlesset/reference/analyze_Sch20_Oscillating_3D.py
and adds the coated-shell decomposition.

THREE MEASURED RADII (same definitions as the uncoated Sch20 analysis)
    R_V     gas-volume radius, Sch20 Eq. (29): (3 V_gas / 4 pi)^(1/3) with
            V_gas = sum alpha_g dV over every cell.  Integrates the whole
            diffuse band, so on a strongly curved interface the r^2-weighted
            outer half inflates it -- R_V OVER-reads, worst at peak compression.
    R_0.5   radially averaged eta = 0.5 contour: bin every cell by r, take the
            volume-weighted mean eta per shell, interpolate the 0.5 crossing.
            Direction-averaged, so it carries no single-ray grid anisotropy.

FOUR REFERENCE CURVES PER FORMULATION
    uncoated       sigma = sigma_water const, kappa_s = 0
    elastic only   Marmottant sigma(R), kappa_s = 0
    viscous only   sigma = sigma_water const, kappa_s on
    full shell     Marmottant sigma(R) + kappa_s      <- what the sim should hit
The decomposition is what makes the test diagnostic: if the sim misses, the
spacing of these four curves says which term is wrong.

USAGE
    python3 analyze_Sch20_Oscillating_Marmottant.py
    python3 analyze_Sch20_Oscillating_Marmottant.py --input ../input_Sch20-Collapsing_Marmottant
    python3 analyze_Sch20_Oscillating_Marmottant.py --output /path/to/plotfiles
    python3 analyze_Sch20_Oscillating_Marmottant.py --models-only    # no yt needed

OUTPUT
    Images/<stem>.png / .pdf        R(t) overlay + normalized residual
    Images/<stem>_regimes.png/.pdf  sigma(R(t)) with the buckled/elastic/
                                    ruptured thresholds, and the shell vs
                                    liquid damping-pressure split
===============================================================================
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from marmottant_rpe_km import (                                   # noqa: E402
    Shell, Liquid, Gas, solve_rpe, solve_km, natural_frequency, gas_pressure)

IMG_DIR = os.path.join(_HERE, "Images")

# =========================================================================== #
#  CASES
# =========================================================================== #
# Both coated Sch20 cases are listed here with the INCLINE plotfile directory
# they write to.  Uncomment a case to include it; the collapsing run is
# commented out for now because the oscillating one finishes first.
#
# Each entry is (input file, plotfile directory).  The plotfile directory is
# only a hint -- if it does not exist the input's own plot_file is used, and
# failing that the same basename under the local bin/ tree, so the identical
# script works on INCLINE and off it.  --input / --output still override.
CASES = [
    (os.path.normpath(os.path.join(_HERE, "..", "input_Sch20-Oscillating_Marmottant")),
     "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/output_Sch20_Oscillating_Marmottant"),

    # (os.path.normpath(os.path.join(_HERE, "..", "input_Sch20-Collapsing_Marmottant")),
    #  "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/output_Sch20_Collapsing_Marmottant"),
]

DEFAULT_INPUT = CASES[0][0]


# =========================================================================== #
#  INPUT-FILE PARSING
# =========================================================================== #
def parse_input(path):
    """AMReX/alamo `key = value` input file -> {key: value-string}.

    Strips comments, keeps the LAST assignment of a duplicated key (so a
    commented-then-reassigned parameter resolves the way the solver sees it).
    """
    kv = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def _num(kv, key, default=None, required=False):
    """Fetch a scalar. Resolves the `epsilon = eta.ic.expression.constant.epsilon`
    indirection the Sch20 inputs use."""
    if key not in kv:
        if required:
            raise KeyError("input file is missing required key '%s'" % key)
        return default
    raw = kv[key]
    seen = 0
    while raw in kv and seen < 8:          # follow key-to-key indirection
        raw = kv[raw]
        seen += 1
    try:
        return float(raw.split()[0])
    except ValueError:
        if required:
            raise ValueError("could not read '%s' as a number (got %r)" % (key, raw))
        return default


def _expr_const(kv, prefix, name, default=None):
    """Read `<prefix>.expression.constant.<name>`."""
    return _num(kv, "%s.expression.constant.%s" % (prefix, name), default)


def build_case(kv, input_path):
    """Turn the parsed input into (Shell, Liquid, Gas, meta) in SI units."""
    # ---- geometry / interface -------------------------------------------- #
    R0 = _expr_const(kv, "eta.ic", "R0")
    if R0 is None:
        R0 = _num(kv, "marmottant.R0", required=True)
    eps = _num(kv, "epsilon")

    # ---- liquid (phase 0) -- stiffened gas -------------------------------- #
    rho_l = _expr_const(kv, "density0.ic", "rho")
    if rho_l is None:                       # constant expression e.g. "1000.0"
        raw = kv.get("density0.ic.expression.region0", "1000.0")
        rho_l = float(raw.strip('"').split()[0])
    gam_l = _num(kv, "eos0.gamma", 2.35)
    pi_l = _num(kv, "eos0.p0", 0.0)
    mu_l = _num(kv, "mu0", 0.0)

    # ---- gas (phase 1) ---------------------------------------------------- #
    gam_g = _num(kv, "eos1.gamma", 1.4)

    # ---- pressures -------------------------------------------------------- #
    p_inf = _expr_const(kv, "pressure0.ic", "p_inf", 1.0e5)
    p_b = _expr_const(kv, "pressure1.ic", "p_b")
    if p_b is None:
        p_b = _expr_const(kv, "pressure0.ic", "p_b", 1.0e4)

    # liquid sound speed from the stiffened-gas EOS at the far-field state
    c_l = np.sqrt(gam_l * (p_inf + pi_l) / rho_l)

    # ---- shell ------------------------------------------------------------ #
    marm = int(_num(kv, "marmottant", 0))
    sigma_w = _num(kv, "sigma", 0.0)
    if not int(_num(kv, "apply_surface_tension", 0)):
        sigma_w = 0.0
    chi = _num(kv, "marmottant.chi", 0.0) if marm else 0.0
    R_buck = _num(kv, "marmottant.R_buckling", 0.0) if marm else 0.0
    s_break = _num(kv, "marmottant.sigma_break", 1.0e30) if marm else 1.0e30
    kappa_s = _num(kv, "shell.kappa_s", 0.0)
    mu_s = _num(kv, "shell.mu_s", 0.0)
    if mu_s:
        print("  [warn] shell.mu_s = %g is nonzero; it cancels for spherical "
              "motion and is ignored by the reference ODEs." % mu_s)

    shell = Shell(chi=chi, R_buck=R_buck, sigma_break=s_break,
                  sigma_water=sigma_w, kappa_s=kappa_s)
    liq = Liquid(rho=rho_l, mu=mu_l, c=c_l, p_inf=p_inf)
    gas = Gas(p_g0=p_b, kappa_g=gam_g)

    meta = dict(
        R0=R0, epsilon=eps, stop_time=_num(kv, "stop_time", 1.0e-6),
        plot_file=kv.get("plot_file", ""),
        prob_lo=[float(x) for x in kv.get("geometry.prob_lo", "0 0 0").split()],
        prob_hi=[float(x) for x in kv.get("geometry.prob_hi", "1 1 1").split()],
        n_cell=[int(float(x)) for x in kv.get("amr.n_cell", "1 1 1").split()],
        max_level=int(_num(kv, "amr.max_level", 0)),
        limiter=kv.get("Limiter.type", "?"), riemann=kv.get("Riemann_Solver.type", "?"),
        gamma_l=gam_l, pi_l=pi_l, gamma_g=gam_g, p_b=p_b,
        input_path=input_path,
        name=os.path.basename(input_path).replace("input_", ""),
    )
    meta["tau_c"] = 0.915 * R0 * np.sqrt(rho_l / p_inf)
    dx0 = (meta["prob_hi"][0] - meta["prob_lo"][0]) / max(meta["n_cell"][0], 1)
    meta["dx_finest"] = dx0 / (2 ** meta["max_level"])
    return shell, liq, gas, meta


def report_case(shell, liq, gas, meta):
    R0 = meta["R0"]
    print("=" * 74)
    print("CASE: %s" % meta["name"])
    print("  input        %s" % meta["input_path"])
    print("-" * 74)
    print("  R0           %.4g m   (%.4g um)" % (R0, R0 * 1e6))
    print("  epsilon      %.4g m   dx_finest %.4g m   R0/dx %.1f  eps/dx %.2f"
          % (meta["epsilon"] or np.nan, meta["dx_finest"],
             R0 / meta["dx_finest"], (meta["epsilon"] or 0) / meta["dx_finest"]))
    print("  liquid       rho %.6g kg/m^3  mu %.4g Pa s  gamma %.4g  pi %.4g Pa"
          % (liq.rho, liq.mu, meta["gamma_l"], meta["pi_l"]))
    print("               c_l = sqrt(gamma(p_inf+pi)/rho) = %.1f m/s" % liq.c)
    print("  gas          gamma %.4g   p_b %.6g Pa" % (gas.kappa_g, gas.p_g0))
    print("  pressures    p_inf %.6g Pa   p_inf/p_b %.1f" % (liq.p_inf, liq.p_inf / gas.p_g0))
    print("  tau_c        %.4e s   stop_time %.4e s  (%.2f tau_c)"
          % (meta["tau_c"], meta["stop_time"], meta["stop_time"] / meta["tau_c"]))
    print("-" * 74)
    if shell.chi > 0:
        print("  SHELL        chi %.4g N/m   kappa_s %.4g kg/s" % (shell.chi, shell.kappa_s))
        print("               R_buckling %.4g m (%.4f R0)" % (shell.R_buck, shell.R_buck / R0))
        print("               R_rupture  %.4g m (%.4f R0)"
              % (shell.R_rupture, shell.R_rupture / R0))
        print("               sigma(R0)  %.5f N/m   sigma_break %.4g   sigma_water %.4g"
              % (shell.sigma(R0), shell.sigma_break, shell.sigma_water))
        print("               elastic window R_rup/R_buck = %.4f"
              % (shell.R_rupture / shell.R_buck))
    else:
        print("  SHELL        none (uncoated, sigma = %.4g N/m)" % shell.sigma_water)
    # The linear natural frequency is only meaningful about MECHANICAL
    # EQUILIBRIUM, p_g = p_inf + 2 sigma(R0)/R0.  These ICs deliberately sit far
    # from it (that is what drives the free response), so f0 is reported at the
    # equilibrium gas pressure and the IC offset is stated separately.
    clean = Shell(chi=0.0, sigma_water=shell.sigma_water)
    eq_c = Gas(p_g0=liq.p_inf + 2 * clean.sigma(R0) / R0, kappa_g=gas.kappa_g)
    eq_s = Gas(p_g0=liq.p_inf + 2 * shell.sigma(R0) / R0, kappa_g=gas.kappa_g)
    f_c = natural_frequency(R0, clean, liq, eq_c) / 2 / np.pi
    f_s = natural_frequency(R0, shell, liq, eq_s) / 2 / np.pi
    print("  linear f0    uncoated %.4e Hz   coated %.4e Hz   shift %+.1f %%"
          % (f_c, f_s, 100 * (f_s - f_c) / f_c if f_c else 0.0))
    print("               (about equilibrium p_g; this IC starts at p_b/p_eq = "
          "%.4f, i.e. far from equilibrium -- a free response, not a "
          "linear oscillation)" % (gas.p_g0 / eq_s.p_g0))
    print("  numerics     %s + %s, max_level %d, n_cell %s"
          % (meta["riemann"], meta["limiter"], meta["max_level"], meta["n_cell"]))
    print("=" * 74)


# =========================================================================== #
#  REFERENCE MODELS
# =========================================================================== #
def shell_variants(shell):
    """The four-way decomposition, ordered weakest -> strongest damping."""
    clean = Shell(chi=0.0, sigma_water=shell.sigma_water, kappa_s=0.0)
    elas = Shell(chi=shell.chi, R_buck=shell.R_buck, sigma_break=shell.sigma_break,
                 sigma_water=shell.sigma_water, kappa_s=0.0)
    visc = Shell(chi=0.0, sigma_water=shell.sigma_water, kappa_s=shell.kappa_s)
    return [("uncoated", clean), ("elastic only", elas),
            ("viscous only", visc), ("full shell", shell)]


def solve_all(shell, liq, gas, meta, n=4001):
    """{(variant, model): (t, R)} for every variant x {RPE, KM}."""
    R0, tend = meta["R0"], meta["stop_time"]
    te = np.linspace(0.0, tend, n)
    out = {}
    for name, sh in shell_variants(shell):
        for mdl, fn in (("RPE", solve_rpe), ("KM", solve_km)):
            try:
                t, R, Rd = fn(R0, 0.0, (0.0, tend), sh, liq, gas, t_eval=te)
                out[(name, mdl)] = (t, R, Rd)
            except Exception as exc:
                print("  [warn] %s/%s failed to integrate: %s" % (name, mdl, exc))
    return out


# =========================================================================== #
#  SIMULATION EXTRACTION
# =========================================================================== #
def _plotfiles(output_dir):
    if not os.path.isdir(output_dir):
        return []
    return sorted(os.path.join(output_dir, d) for d in os.listdir(output_dir)
                  if d.endswith("cell") and
                  os.path.isdir(os.path.join(output_dir, d)))


def extract_radii(output_dir, R0, nbins=400):
    """Per plotfile, the two radii R_V and R_0.5.  Returns (t, R_V, R_05).

    Octant detection: a domain lo edge at the origin means that axis is a
    symmetry plane, so the gas volume is scaled by 2 per such axis (Sch20's
    octant -> x8), matching the uncoated Sch20 analysis.
    """
    try:
        import yt
    except ImportError:
        print("  [skip] yt not available -- run with --models-only for the "
              "reference curves alone.")
        return (np.array([]),) * 3
    yt.funcs.mylog.setLevel(40)

    pfs = _plotfiles(output_dir)
    if not pfs:
        print("  [skip] no plotfiles under %s" % output_dir)
        return (np.array([]),) * 3

    times, RV, R05 = [], [], []
    announced = False
    for pf in pfs:
        try:
            ds = yt.load(pf)
            dle, dre = ds.domain_left_edge, ds.domain_right_edge
            sym = 1
            cen = []
            for d in range(3):
                if float(dle[d]) > -1e-6:
                    sym *= 2
                    cen.append(0.0)
                else:
                    cen.append(0.5 * float(dle[d] + dre[d]))
            if not announced:
                print("  [radius] geometry: %s -> V_gas x%d"
                      % ("OCTANT" if sym > 1 else "FULL domain", sym))
                announced = True

            reg = ds.all_data()
            eta = np.array(reg["eta"], dtype=float)
            try:
                vol = np.array(reg["index", "cell_volume"], dtype=float)
            except Exception:
                vol = np.array(reg["cell_volume"], dtype=float)

            # --- R_V : Sch20 Eq. (29) gas-volume radius ---------------------
            alpha_g = np.clip(1.0 - eta, 0.0, 1.0)
            V_b = float(np.sum(alpha_g * vol)) * sym
            RV.append((3.0 * V_b / (4.0 * np.pi)) ** (1.0 / 3.0))

            # --- R_0.5 : radially averaged eta = 0.5 contour ----------------
            x = np.array(reg["x"], dtype=float) - cen[0]
            y = np.array(reg["y"], dtype=float) - cen[1]
            z = np.array(reg["z"], dtype=float) - cen[2]
            r = np.sqrt(x * x + y * y + z * z)
            rmax = min(float(r.max()), 10.0 * R0)
            edges = np.linspace(0.0, rmax, nbins + 1)
            idx = np.clip(np.digitize(r, edges) - 1, 0, nbins - 1)
            rc = 0.5 * (edges[:-1] + edges[1:])
            wsum = np.bincount(idx, weights=vol, minlength=nbins)
            esum = np.bincount(idx, weights=eta * vol, minlength=nbins)
            good = wsum > 0
            R05.append(_first_crossing(rc[good], esum[good] / wsum[good]))

            times.append(float(ds.current_time))
        except Exception:
            continue

    if not times:
        return (np.array([]),) * 3
    t = np.array(times)
    o = np.argsort(t)
    return t[o], np.array(RV)[o], np.array(R05)[o]


def _first_crossing(rr, ee):
    """First outward eta = 0.5 crossing, linearly interpolated."""
    for i in range(len(rr) - 1):
        if (ee[i] - 0.5) * (ee[i + 1] - 0.5) < 0:
            return rr[i] + (0.5 - ee[i]) * (rr[i + 1] - rr[i]) / (ee[i + 1] - ee[i])
    return np.nan





# =========================================================================== #
#  PLOTTING  -- one figure per file, .png + .eps, format matched to
#  tests/FlowRayleighPlesset/reference/analyze_Sch20_Oscillating_3D.py
# =========================================================================== #
FIG_WIDTH, FIG_HEIGHT = 11, 6.5
FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_LEGEND, FONT_SIZE_TICK = 16, 14, 12, 11
DPI = 180

# Model-curve styling.  Keyed by variant name; SHELL_ONLY keeps only the two
# marked keep_in_shell_only.
_MODEL_STYLE = {
    "uncoated":     dict(color="tab:gray",  ls="--", lw=1.6, keep=True),
    "elastic only": dict(color="tab:green", ls="-.", lw=1.4, keep=False),
    "viscous only": dict(color="tab:olive", ls=":",  lw=1.4, keep=False),
    "full shell":   dict(color="tab:blue",  ls="-",  lw=2.0, keep=True),
}


def _save(fig, stem):
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, stem)
    for ext in ("png", "eps"):
        fig.savefig(f"{out}.{ext}", dpi=DPI, bbox_inches="tight")
    print("  wrote %s.png / .eps" % out)
    plt.close(fig)


def _rc():
    plt.rcParams.update({
        "axes.titlesize":  FONT_SIZE_TITLE,
        "axes.labelsize":  FONT_SIZE_LABEL,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "xtick.labelsize": FONT_SIZE_TICK,
        "ytick.labelsize": FONT_SIZE_TICK,
    })


def _variants(shell, shell_only):
    out = []
    for name, sh in shell_variants(shell):
        st = _MODEL_STYLE[name]
        if shell_only and not st["keep"]:
            continue
        out.append((name, sh, st))
    return out


def plot_radius(models, sim, shell, meta, stem, model="KM", shell_only=False):
    """R/R0 vs t: reference models + the two measured radii."""
    _rc()
    R0, tau = meta["R0"], meta["tau_c"]
    t_s, RV, R05 = sim
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for name, _sh, st in _variants(shell, shell_only):
        key = (name, model)
        if key not in models:
            continue
        t, R, _ = models[key]
        ax.plot(t / tau, R / R0, st["ls"], color=st["color"], lw=st["lw"],
                alpha=0.9, label=f"{name} ({model})")
    if len(t_s):
        ax.plot(t_s / tau, RV / R0, "-o", color="tab:purple", lw=2.0, ms=3,
                label=r"hydro2 $R=(3V_b/4\pi)^{1/3}$ (gas volume)")
        m = np.isfinite(R05)
        if m.any():
            ax.plot(t_s[m] / tau, R05[m] / R0, "-s", color="tab:red", lw=2.0, ms=3.5,
                    label=r"hydro2 $\eta{=}0.5$ (radial avg)")
    if shell.chi > 0:
        for y, lab in ((shell.R_buck / R0, r"$R_{\rm buckling}$"),
                       (shell.R_rupture / R0, r"$R_{\rm rupture}$")):
            if 0.0 < y < 3.0:
                ax.axhline(y, color="0.75", lw=0.9, zorder=0)
                ax.text(ax.get_xlim()[1], y, " " + lab, va="center",
                        fontsize=FONT_SIZE_TICK, color="0.45")
    ax.set_xlabel(r"$t / \tau_c$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$R / R_0$", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{meta['name']}: coated bubble radius vs Marmottant-modified {model}",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    plt.tight_layout(); _save(fig, stem)


def plot_residual(models, sim, shell, meta, stem, model="KM"):
    """(R_sim - R_model)/R0 against the full-shell reference."""
    t_s, RV, R05 = sim
    key = ("full shell", model)
    if not len(t_s) or key not in models:
        return
    _rc()
    R0, tau = meta["R0"], meta["tau_c"]
    tm, Rm, _ = models[key]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for lab, arr, c, mk in ((r"$R_V$ (gas volume)", RV, "tab:purple", "o"),
                            (r"$R_{0.5}$ (radial avg)", R05, "tab:red", "s")):
        m = np.isfinite(arr)
        if not m.any():
            continue
        ax.plot(t_s[m] / tau, 100.0 * (arr[m] - np.interp(t_s[m], tm, Rm)) / R0,
                "-" + mk, color=c, lw=1.6, ms=3.5, label=lab)
    ax.axhline(0.0, color="0.3", lw=1.0)
    ax.set_xlabel(r"$t / \tau_c$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$(R_{\rm sim}-R_{\rm model})/R_0$   [\%]".replace("\\%", "%"),
                  fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{meta['name']}: residual against full-shell {model}",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    plt.tight_layout(); _save(fig, stem)


def plot_sigma(models, shell, meta, stem, model="KM"):
    """sigma(R(t)) with the buckling / rupture thresholds."""
    key = ("full shell", model)
    if shell.chi <= 0 or key not in models:
        return
    _rc()
    tau = meta["tau_c"]
    t, R, _ = models[key]
    sig = shell.sigma(R)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(t / tau, sig, "-", color="tab:blue", lw=2.0, label=r"$\sigma(R)$")
    ax.axhline(0.0, color="0.7", lw=0.9)
    ax.axhline(shell.sigma_break, color="0.7", lw=0.9)
    ax.fill_between(t / tau, 0, shell.sigma_break, where=(sig <= 0),
                    color="0.9", step="mid", label="buckled ($\\sigma=0$)")
    frac = dict(b=100*np.mean(R <= shell.R_buck),
                e=100*np.mean((R > shell.R_buck) & (R < shell.R_rupture)),
                r=100*np.mean(R >= shell.R_rupture))
    ax.text(0.02, 0.96, "buckled %.0f%%\nelastic %.0f%%\nruptured %.0f%%"
            % (frac["b"], frac["e"], frac["r"]), transform=ax.transAxes,
            va="top", fontsize=FONT_SIZE_LEGEND,
            bbox=dict(fc="w", ec="0.7", lw=0.7))
    ax.set_xlabel(r"$t / \tau_c$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(r"$\sigma(R)$   [N/m]", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{meta['name']}: shell tension and regime",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    plt.tight_layout(); _save(fig, stem)


def plot_damping(models, shell, liq, meta, stem, model="KM"):
    """Shell vs liquid interfacial pressure contributions."""
    key = ("full shell", model)
    if key not in models:
        return
    _rc()
    tau = meta["tau_c"]
    t, R, Rd = models[key]
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(t / tau, np.abs(4.0*shell.kappa_s*Rd/R**2), "-", color="tab:blue", lw=2.0,
            label=r"$4\kappa_s\dot R/R^2$  (shell)")
    ax.plot(t / tau, np.abs(4.0*liq.mu*Rd/R), "--", color="tab:orange", lw=1.6,
            label=r"$4\mu_\ell\dot R/R$  (liquid)")
    ax.plot(t / tau, np.abs(2.0*shell.sigma(R)/R), ":", color="tab:green", lw=1.6,
            label=r"$2\sigma(R)/R$  (Laplace)")
    ax.axhline(liq.p_inf, color="tab:red", lw=1.2, label=r"$p_\infty$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t / \tau_c$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("interfacial pressure   [Pa]", fontsize=FONT_SIZE_LABEL)
    ax.set_title(f"{meta['name']}: shell vs liquid contributions",
                 fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both"); ax.tick_params(labelsize=FONT_SIZE_TICK)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc="best")
    plt.tight_layout(); _save(fig, stem)


def summarize(models, sim, shell, meta, shell_only=False):
    R0 = meta["R0"]
    print("\nREFERENCE MODELS -- R_min/R0 (and R_max/R0)")
    print("  %-14s %18s %18s" % ("variant", "RPE", "KM"))
    for name, _sh, _st in _variants(shell, shell_only):
        cells = []
        for mdl in ("RPE", "KM"):
            if (name, mdl) in models:
                _, R, _ = models[(name, mdl)]
                cells.append("%7.4f / %7.4f" % (R.min()/R0, R.max()/R0))
            else:
                cells.append("%17s" % "--")
        print("  %-14s %18s %18s" % (name, cells[0], cells[1]))
    t_s, RV, R05 = sim
    if not len(t_s):
        print("\n  (no simulation data -- reference curves only)")
        return
    print("\nSIMULATION -- %d frames, t in [%.3e, %.3e] s" % (len(t_s), t_s.min(), t_s.max()))
    for lab, arr in ((r"R_V", RV), ("R_0.5", R05)):
        m = np.isfinite(arr)
        if m.any():
            print("  %-6s R_min/R0 %7.4f   R_max/R0 %7.4f   R(0)/R0 %7.4f"
                  % (lab, arr[m].min()/R0, arr[m].max()/R0, arr[m][0]/R0))
    print("\nSIM vs FULL-SHELL MODEL -- RMS of (R_sim - R_model)/R0")
    for mdl in ("RPE", "KM"):
        key = ("full shell", mdl)
        if key not in models:
            continue
        tm, Rm, _ = models[key]
        row = []
        for lab, arr in (("R_V", RV), ("R_0.5", R05)):
            m = np.isfinite(arr)
            if not m.any():
                row.append("%s --" % lab); continue
            d = (arr[m] - np.interp(t_s[m], tm, Rm)) / R0
            row.append("%s %6.3f%%" % (lab, 100*np.sqrt(np.mean(d**2))))
        print("  %-4s  %s" % (mdl, "   ".join(row)))


def run_case(input_path, out_hint=None, shell_only=False, models_only=False,
             stem=None, model="KM"):
    """Analyse ONE case end to end."""
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise SystemExit("input file not found: %s" % input_path)
    kv = parse_input(input_path)
    shell, liq, gas, meta = build_case(kv, input_path)
    report_case(shell, liq, gas, meta)
    print("\nIntegrating reference models ...")
    models = solve_all(shell, liq, gas, meta)

    sim = (np.array([]),) * 3
    if not models_only:
        cands = [out_hint, meta["plot_file"]]
        base = os.path.basename((out_hint or meta["plot_file"] or "").rstrip("/"))
        if base:
            cands.append(os.path.normpath(os.path.join(
                _HERE, "..", "..", "..", "bin", "tests", "FlowMarmottant", base)))
        for c in cands:
            if c and os.path.isdir(c):
                print("\nReading plotfiles from %s" % c)
                sim = extract_radii(c, meta["R0"])
                break
        else:
            print("\n  [skip] no plotfile directory found; tried:")
            for c in cands:
                if c: print("         %s" % c)

    stem = stem or meta["name"]
    print()
    plot_radius(models, sim, shell, meta, f"{stem}_R_volume", model, shell_only)
    plot_residual(models, sim, shell, meta, f"{stem}_residual", model)
    plot_sigma(models, shell, meta, f"{stem}_sigma", model)
    plot_damping(models, shell, liq, meta, f"{stem}_damping", model)
    summarize(models, sim, shell, meta, shell_only)
