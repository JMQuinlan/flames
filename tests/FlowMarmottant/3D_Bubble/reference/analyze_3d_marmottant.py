#!/usr/bin/env python3
# ============================================================================
#  3D MARMOTTANT BUBBLE -- STATIC LAPLACE RADIUS SWEEP ANALYSIS (batch)
# ============================================================================
#  Reads EVERY 3D sweep case it can find under ROOT (output_R*), in any state
#  (finished, running, or aborted), and produces:
#
#   1. Radius history      R/R0 - 1 (%) vs t, all cases coloured by R0
#                          (dense series from thermo.dat when present,
#                           otherwise plotfiles)
#   2. Shell state         sigma_eff measured vs the exact Marmottant law,
#                          and Gamma(t) against Gamma_buck per case
#   3. Laplace balance     p_gas - p_inf measured vs 2 sigma_eff / R0
#   4. Frequency analysis  FFT of R(t) -> dominant ring-down frequency vs the
#                          linearised Marmottant-RPE natural frequency, which
#                          CHANGES BY REGIME (shell elasticity only stiffens the
#                          elastic branch)
#   5. RPE / KM overlay    Rayleigh-Plesset and Keller-Miksis with the
#                          Marmottant shell law, seeded from the simulation's
#                          own state and integrated forward -- compares the
#                          waveform, frequency and decay directly
#   6. Summary table       console + CSV
#
#  Physics per case is read from the run's own `metadata` file (R0, chi,
#  R_buckling, sigma_break, EOS, mu, IC pressures), so nothing here is
#  hardcoded to the generator's defaults.
#
#  OCTANT runs (prob_lo = 0 on every axis) are detected automatically and the
#  integrated gas volume is multiplied by 8.
#
#  Usage (from this reference folder):
#     python3 analyze_3d_marmottant.py
#     MARM3D_ROOT=/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble python3 analyze_3d_marmottant.py
#     SKIP_PLOTFILES=1 python3 analyze_3d_marmottant.py      # thermo.dat only (fast)
# ============================================================================

import os
import sys
import glob
import math
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_HERE = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# CONFIG
# ============================================================================
_CANDIDATE_ROOTS = [
    os.environ.get("MARM3D_ROOT", ""),
    os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "bin", "tests",
                                  "FlowMarmottant", "3D_Bubble")),
    "/mmfs1/home/ttryon/flames/bin/tests/FlowMarmottant/3D_Bubble",
]
IMG_DIR         = os.environ.get("IMG_DIR", os.path.join(_HERE, "Images"))
SKIP_PLOTFILES  = int(os.environ.get("SKIP_PLOTFILES", "0"))
PLOTFILE_STRIDE = int(os.environ.get("PLOTFILE_STRIDE", "1"))
T_UNIT, T_SCALE = "ms", 1.0e3
R_UNIT, R_SCALE = "mm", 1.0e3
BAND            = 0.05       # eta(1-eta) > BAND selects the interface band
FFT_FMIN_FRAC   = 0.2        # search peak in [FMIN_FRAC, FMAX_FRAC] x f_theory
FFT_FMAX_FRAC   = 5.0
FFT_TSKIP       = 0.0        # seconds of startup excluded from the FFT
ODE_SUBSTEPS    = 400        # RK4 steps per theoretical natural period
DPI             = 170

# ============================================================================
# REUSABLE PRIMITIVES  (no module globals below this line up to MODEL)
# ============================================================================

def find_plotfiles(out_dir, exclude=(".old.",)):
    """*cell plotfiles, NUMERICALLY sorted (lexicographic sort puts 72682cell
    after 726786cell), with *.old.* dropped."""
    fs = [p for p in glob.glob(os.path.join(out_dir, "*cell"))
          if os.path.isdir(p) and not any(s in p for s in exclude)]
    def _k(p):
        try:
            return int(os.path.basename(p).replace("cell", ""))
        except ValueError:
            return -1
    return sorted(fs, key=_k)


_MD_NVALS = None


def read_metadata(out_dir):
    """Alamo `metadata` -> dict of strings.  Two on-disk formats exist:
         older builds:  key = value
         newer builds:  key(nvals = N)  :: [v1 v2 ...]
    Both are handled; a mismatch silently defaults every physics parameter."""
    import re
    global _MD_NVALS
    if _MD_NVALS is None:
        _MD_NVALS = re.compile(r"^\s*([^\s(=]+)\s*\(nvals\s*=\s*\d+\)\s*::\s*\[(.*)\]\s*$")
    md = {}
    p = os.path.join(out_dir, "metadata")
    if not os.path.isfile(p):
        return md
    with open(p, errors="ignore") as fh:
        for raw in fh:
            m = _MD_NVALS.match(raw)
            if m:
                md[m.group(1).strip()] = m.group(2).replace(",", " ").strip().strip('"')
                continue
            line = raw.split("#", 1)[0]
            if "=" in line:
                k, v = line.split("=", 1)
                md[k.strip()] = v.strip().strip('"')
    return md


def mfloat(md, key, default=None, idx=0):
    try:
        return float(md[key].split()[idx])
    except (KeyError, ValueError, IndexError):
        return default


def read_thermo(out_dir):
    """thermo.dat -> dict of columns (first column is time)."""
    p = os.path.join(out_dir, "thermo.dat")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        header = fh.readline().split()
    try:
        data = np.loadtxt(p, skiprows=1, ndmin=2)
    except Exception:
        return None
    if data.size == 0 or data.shape[0] < 4:
        return None
    cols = {}
    for i, h in enumerate(header):
        if i < data.shape[1]:
            cols[h] = data[:, i]
    if "time" not in cols:
        cols["time"] = data[:, 0]
    # A restart from plotfile NNNNN appends rows starting at that plotfile's
    # time, so rows written between it and the killed job's last step appear
    # twice.  Keep the NEWEST segment: scan backwards, keep a row only if it
    # is earlier than everything already kept.
    t = cols["time"]
    keep = np.zeros(t.size, dtype=bool)
    tmin = np.inf
    for i in range(t.size - 1, -1, -1):
        if t[i] < tmin:
            keep[i] = True
            tmin = t[i]
    return {k: v[keep] for k, v in cols.items()}


def dominant_frequency(t, y, fmin, fmax, tskip=0.0):
    """Peak of the Hann-windowed amplitude spectrum inside [fmin, fmax].
    Resamples to a uniform grid first (thermo rows follow a variable dt).
    Returns (f_peak, freqs, amp) or (nan, None, None)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    m = np.isfinite(t) & np.isfinite(y) & (t >= tskip)
    t, y = t[m], y[m]
    if t.size < 16 or t[-1] - t[0] <= 0:
        return float("nan"), None, None
    n = int(2 ** math.ceil(math.log2(max(64, t.size))))
    tu = np.linspace(t[0], t[-1], n)
    yu = np.interp(tu, t, y)
    yu = yu - np.polyval(np.polyfit(tu, yu, 1), tu)       # remove drift
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft(yu * w)) * 2.0 / w.sum()
    fr = np.fft.rfftfreq(n, d=(tu[1] - tu[0]))
    sel = (fr >= fmin) & (fr <= fmax)
    if not sel.any():
        return float("nan"), fr, amp
    i = np.argmax(np.where(sel, amp, -1.0))
    # parabolic interpolation on the log spectrum for sub-bin accuracy
    if 0 < i < len(amp) - 1 and amp[i - 1] > 0 and amp[i + 1] > 0:
        a, b, c = np.log(amp[i - 1]), np.log(amp[i]), np.log(amp[i + 1])
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float(fr[i] + off * (fr[1] - fr[0])), fr, amp
    return float(fr[i]), fr, amp

# ============================================================================
# MODEL: Marmottant shell + RPE / Keller-Miksis
# ============================================================================

def sigma_marm(R, P):
    s = P["chi"] * ((R / P["Rb"]) ** 2 - 1.0)
    return min(max(s, 0.0), P["sbrk"])


def dsigma_dR(R, P):
    s = P["chi"] * ((R / P["Rb"]) ** 2 - 1.0)
    if s <= 0.0 or s >= P["sbrk"]:
        return 0.0
    return 2.0 * P["chi"] * R / P["Rb"] ** 2


def regime_of(R0, P):
    s = P["chi"] * ((R0 / P["Rb"]) ** 2 - 1.0)
    return "buckled" if s <= 0 else ("ruptured" if s >= P["sbrk"] else "elastic")


def natural_frequency(P):
    """Linearised Marmottant-RPE about R0 (sphere).  With R = R0 + r,
         rho R0^2 r'' = -[3 k p_g0 + 2 sigma'(R0) - 2 sigma0/R0] r
       so  w0^2 = [3 k p_g0 - 2 sigma0/R0 + 2 sigma'(R0)] / (rho R0^2),
       sigma' = dsigma/dR = 2 chi R0/R_buck^2 on the elastic branch, 0 on the
       plateaus.  For R0 ~ R_buck this is Marmottant's 4 chi/R0 stiffening."""
    R0 = P["R0"]
    k  = 3.0 * P["kappa"] * P["pg0"] - 2.0 * sigma_marm(R0, P) / R0 \
         + 2.0 * dsigma_dR(R0, P)
    return math.sqrt(max(k, 0.0) / (P["rho"] * R0 * R0)) / (2.0 * math.pi)


def _rddot(R, Rd, P, model):
    rho, c, mu = P["rho"], P["c"], P["mu"]
    pg  = P["pg0"] * (P["R0"] / R) ** (3.0 * P["kappa"])
    sig = sigma_marm(R, P)
    pw  = pg - 2.0 * sig / R - 4.0 * mu * Rd / R
    dp  = pw - P["pinf"]
    if model == "rpe":
        return (dp / rho - 1.5 * Rd * Rd) / R
    dpg  = -3.0 * P["kappa"] * pg * Rd / R
    dsig = -(2.0 * dsigma_dR(R, P) * Rd / R - 2.0 * sig * Rd / (R * R))
    dpw  = dpg + dsig + 4.0 * mu * Rd * Rd / (R * R)       # -4mu Rddot/R moved to LHS
    num  = (1.0 + Rd / c) * dp / rho + R / (rho * c) * dpw - 1.5 * (1.0 - Rd / (3.0 * c)) * Rd * Rd
    den  = (1.0 - Rd / c) * R + 4.0 * mu / (rho * c)
    return num / den


def integrate_ode(model, P, R_start, Rd_start, t_start, t_end, f0):
    if t_end <= t_start or not np.isfinite(f0) or f0 <= 0:
        return np.array([]), np.array([])
    n = int(max(200, ODE_SUBSTEPS * (t_end - t_start) * f0))
    n = min(n, 2_000_000)
    dt = (t_end - t_start) / n
    t = t_start + dt * np.arange(n + 1)
    R = np.full(n + 1, np.nan); R[0] = R_start
    y = np.array([R_start, Rd_start])
    f = lambda yy: np.array([yy[1], _rddot(yy[0], yy[1], P, model)])
    for i in range(n):
        try:
            k1 = f(y); k2 = f(y + 0.5 * dt * k1); k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        except (OverflowError, ZeroDivisionError, ValueError):
            break
        if not np.isfinite(y[0]) or y[0] <= 1e-3 * P["R0"]:
            break
        R[i + 1] = y[0]
    return t, R

# ============================================================================
# PER-CASE EXTRACTION
# ============================================================================

def case_params(md):
    rho  = mfloat(md, "density0.ic.expression.region0", 1000.0)
    gl   = mfloat(md, "eos0.gamma", 2.35)
    pinfl= mfloat(md, "eos0.p0", 1.0e9)
    pinf = mfloat(md, "pressure0.ic.expression.region0", 101325.0)
    pgas = mfloat(md, "pressure1.ic.expression.region0", pinf)
    plo  = [mfloat(md, "geometry.prob_lo", 0.0, i) for i in range(3)]
    return dict(
        R0=mfloat(md, "marmottant.R0", mfloat(md, "eta.ic.expression.constant.R0", 1e-3)),
        Rb=mfloat(md, "marmottant.R_buckling", 1e-3),
        chi=mfloat(md, "marmottant.chi", 0.55),
        sbrk=mfloat(md, "marmottant.sigma_break", 0.073),
        marm=int(mfloat(md, "marmottant", 1)),
        rho=rho, mu=mfloat(md, "mu0", 0.0), kappa=mfloat(md, "eos1.gamma", 1.4),
        pinf=pinf, pg0=pgas, c=math.sqrt(gl * (pinf + pinfl) / rho),
        sym=8.0 if all(abs(v) < 1e-14 for v in plo) else 1.0,
        stop=mfloat(md, "stop_time", float("nan")),
    )


def plotfile_series(out_dir, P):
    import yt
    yt.set_log_level(50)
    rows = []
    fs = find_plotfiles(out_dir)[::max(1, PLOTFILE_STRIDE)]
    for f in fs:
        try:
            ds = yt.load(f); ad = ds.all_data()
            vol = np.asarray(ad["index", "cell_volume"])
            eta = np.asarray(ad["boxlib", "eta"])
            V = P["sym"] * float(((1.0 - eta) * vol).sum())
            R = (3.0 * V / (4.0 * math.pi)) ** (1.0 / 3.0)
            w = eta * (1.0 - eta); m = w > BAND
            row = dict(t=float(ds.current_time), R=R, sig=np.nan, sd=np.nan,
                       gam=np.nan, u=np.nan)
            # kappa2 / Gamma are filled during the first step, not at the IC:
            # reading them at t=0 reports sigma = 0 (-100% error).
            if m.any() and row["t"] > 0.0:
                ww = w[m] * vol[m]; ww /= ww.sum()
                have = {str(x[1]) for x in ds.field_list}
                if "kappa2" in have:
                    s = np.asarray(ad["boxlib", "kappa2"])[m]
                    row["sig"] = float((ww * s).sum())
                    row["sd"] = float(np.sqrt((ww * (s - row["sig"]) ** 2).sum()))
                if "Gamma" in have:
                    row["gam"] = float((ww * np.asarray(ad["boxlib", "Gamma"])[m]).sum())
                u2 = sum(np.asarray(ad["boxlib", c])[m] ** 2
                         for c in ("velocityx", "velocityy", "velocityz") if c in have)
                row["u"] = float(np.sqrt(u2).max())
            rows.append(row)
        except Exception as e:
            print(f"      [skip] {os.path.basename(f)}: {e}")
    return rows


def analyse_case(out_dir):
    name = os.path.basename(out_dir).replace("output_", "")
    md = read_metadata(out_dir)
    if not md:
        print(f"  [skip] {name}: no metadata (run never started?)")
        return None
    P = case_params(md)
    C = dict(name=name, dir=out_dir, P=P, regime=regime_of(P["R0"], P),
             sig_exact=sigma_marm(P["R0"], P), f_theory=natural_frequency(P))
    C["dp_exact"] = 2.0 * C["sig_exact"] / P["R0"]

    th = read_thermo(out_dir)
    if th is not None and "gas_volume" in th:
        t = th["time"]; V = P["sym"] * th["gas_volume"]
        C["tR"] = t
        C["R"] = (3.0 * np.maximum(V, 0.0) / (4.0 * math.pi)) ** (1.0 / 3.0)
        if "gas_pressure_int" in th:
            with np.errstate(divide="ignore", invalid="ignore"):
                C["pgas"] = th["gas_pressure_int"] / np.where(th["gas_volume"] > 0, th["gas_volume"], np.nan)
        if "kinetic_energy" in th:
            C["KE"] = P["sym"] * th["kinetic_energy"]
        C["series"] = "thermo.dat"
    C["pf"] = [] if SKIP_PLOTFILES else plotfile_series(out_dir, P)
    if "R" not in C and C["pf"]:
        C["tR"] = np.array([r["t"] for r in C["pf"]]); C["R"] = np.array([r["R"] for r in C["pf"]])
        C["series"] = "plotfiles (coarse -- FFT aliased)"
    if "R" not in C:
        print(f"  [skip] {name}: no thermo.dat and no readable plotfiles yet")
        return None

    # The volume radius of a tanh profile is biased high by ~3(pi^2/12)(eps/R)^2
    # before anything moves.  Normalise to the run's own t=0 volume radius so
    # the plotted/tabulated "drift" is motion only; report the bias separately.
    C["R_ic"] = float(C["R"][0])
    C["bias_pct"] = 100.0 * (C["R_ic"] / P["R0"] - 1.0)
    C["Rn"] = C["R"] / C["R_ic"]
    C["t_end"] = float(C["tR"][-1])
    C["complete"] = np.isfinite(P["stop"]) and C["t_end"] >= 0.99 * P["stop"]

    f0 = C["f_theory"]
    C["f_sim"], C["fr"], C["amp"] = dominant_frequency(
        C["tR"], C["R"], FFT_FMIN_FRAC * f0, FFT_FMAX_FRAC * f0, FFT_TSKIP)
    if C["series"].startswith("plotfiles"):
        dtp = np.median(np.diff(C["tR"])) if C["tR"].size > 1 else np.inf
        if f0 > 0.5 / dtp:
            C["f_sim"] = float("nan")          # below Nyquist -- do not report a bogus peak

    # seed the ODEs at the largest excursion in the first 20% of the record
    # (an extremum, so Rdot ~ 0), then integrate to the end of the run
    # seed in bias-free units: R_model = R0 * (R_sim / R_sim(t=0))
    t, Rm = C["tR"], P["R0"] * C["Rn"]
    head = t <= t[0] + 0.2 * (t[-1] - t[0])
    j = int(np.argmax(np.where(head, np.abs(Rm - P["R0"]), -1.0)))
    Rd0 = float(np.gradient(Rm, t)[j]) if t.size > 2 else 0.0
    C["seed"] = (float(t[j]), float(Rm[j]), Rd0)
    C["rpe"] = integrate_ode("rpe", P, Rm[j], Rd0, t[j], t[-1], f0)
    C["km"]  = integrate_ode("km",  P, Rm[j], Rd0, t[j], t[-1], f0)
    return C

# ============================================================================
# PLOTS
# ============================================================================

def _style():
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})


def _cmap(cases):
    R0s = np.array([c["P"]["R0"] for c in cases])
    norm = mcolors.Normalize(R0s.min(), R0s.max() if R0s.max() > R0s.min() else R0s.min() * 1.01)
    return norm, plt.get_cmap("viridis")


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(IMG_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote Images/{name}.png/.pdf")


def plot_radius(cases):
    norm, cmap = _cmap(cases)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for c in cases:
        ax.plot(c["tR"] * T_SCALE, 100 * (c["Rn"] - 1),
                color=cmap(norm(c["P"]["R0"])), lw=1.2,
                label=f"{c['P']['R0']*R_SCALE:.3f} {R_UNIT} ({c['regime']})"
                      + ("" if c["complete"] else "  [incomplete]"))
    ax.axhline(0, color="0.3", ls="--", lw=0.9)
    ax.set_xlabel(f"Time ({T_UNIT})"); ax.set_ylabel(r"$100\,(R/R_{t=0}-1)$  (%)")
    ax.set_title("3D Marmottant bubble -- normalised radius", fontweight="bold")
    ax.legend(fontsize=7.5, ncol=2, frameon=False)
    _save(fig, "fig3d_radius")


def plot_shell(cases):
    P0 = cases[0]["P"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    Rg = np.linspace(min(c["P"]["R0"] for c in cases) * 0.97,
                     max(c["P"]["R0"] for c in cases) * 1.03, 400)
    a1.plot(Rg * R_SCALE, [sigma_marm(r, P0) for r in Rg], "k-", lw=1.4, label="Marmottant law (exact)")
    for c in cases:
        live = [r for r in c["pf"] if r["t"] > 0.0 and np.isfinite(r["sig"])]
        if live:
            last = live[-1]
            a1.errorbar(c["P"]["R0"] * R_SCALE, last["sig"], yerr=last["sd"], fmt="o",
                        ms=5, color="tab:red", capsize=3)
    a1.plot([], [], "o", color="tab:red", label="simulation (final frame, band-weighted ±1sd)")
    a1.set_xlabel(f"$R_0$ ({R_UNIT})"); a1.set_ylabel(r"$\sigma_{eff}$ (N/m)")
    a1.set_title("Shell tension", fontweight="bold"); a1.legend(fontsize=8, frameon=False)
    norm, cmap = _cmap(cases)
    for c in cases:
        if c["pf"]:
            a2.plot([r["t"] * T_SCALE for r in c["pf"]], [r["gam"] for r in c["pf"]],
                    "-o", ms=3, color=cmap(norm(c["P"]["R0"])))
    a2.set_xlabel(f"Time ({T_UNIT})"); a2.set_ylabel(r"$\Gamma$ (band mean)")
    a2.set_title(r"Areal shell density $\Gamma(t)$", fontweight="bold")
    _save(fig, "fig3d_shell")


def plot_laplace(cases):
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    xs, ys = [], []
    for c in cases:
        if "pgas" in c:
            n = max(1, len(c["pgas"]) // 5)
            meas = np.nanmean(c["pgas"][-n:]) - c["P"]["pinf"]
            ax.plot(c["dp_exact"], meas, "o", color="tab:blue")
            ax.annotate(f"{c['P']['R0']*R_SCALE:.3f}", (c["dp_exact"], meas), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
            xs.append(c["dp_exact"]); ys.append(meas)
    if not xs:
        plt.close(fig); print("  [note] no thermo gas pressure -- Laplace plot skipped"); return
    lim = [min(xs + ys + [0]) - 5, max(xs + ys) + 5]
    ax.plot(lim, lim, "k--", lw=1, label="exact  $2\\sigma_{eff}/R_0$")
    ax.set_xlabel(r"exact $2\sigma_{eff}/R_0$ (Pa)"); ax.set_ylabel(r"measured $\bar p_{gas}-p_\infty$ (Pa)")
    ax.set_title("Laplace balance (last 20% of run)", fontweight="bold"); ax.legend(frameon=False)
    _save(fig, "fig3d_laplace")


def plot_frequency(cases):
    norm, cmap = _cmap(cases)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for c in cases:
        if c["fr"] is not None:
            a1.semilogy(c["fr"] / 1e3, np.maximum(c["amp"], 1e-30) / c["P"]["R0"],
                        color=cmap(norm(c["P"]["R0"])), lw=0.9)
            a1.axvline(c["f_theory"] / 1e3, color=cmap(norm(c["P"]["R0"])), ls=":", lw=0.7)
    fmax = max(c["f_theory"] for c in cases) * 3
    a1.set_xlim(0, fmax / 1e3); a1.set_xlabel("frequency (kHz)"); a1.set_ylabel(r"$|\hat R|/R_0$")
    a1.set_title("R(t) spectra  (dotted = linear theory)", fontweight="bold")
    P0 = cases[0]["P"]
    Rg = np.linspace(min(c["P"]["R0"] for c in cases) * 0.97, max(c["P"]["R0"] for c in cases) * 1.03, 400)
    th = []
    for r in Rg:
        P = dict(P0); P["R0"] = r; P["pg0"] = P0["pinf"] + 2 * sigma_marm(r, P0) / r
        th.append(natural_frequency(P))
    a2.plot(Rg * R_SCALE, np.array(th) / 1e3, "k-", lw=1.4, label="linearised Marmottant RPE")
    for c in cases:
        a2.plot(c["P"]["R0"] * R_SCALE, c["f_sim"] / 1e3, "o", ms=6, color=cmap(norm(c["P"]["R0"])))
    a2.plot([], [], "o", color="0.4", label="simulation FFT peak")
    a2.set_xlabel(f"$R_0$ ({R_UNIT})"); a2.set_ylabel("natural frequency (kHz)")
    a2.set_title("Natural frequency vs radius", fontweight="bold"); a2.legend(frameon=False, fontsize=8)
    _save(fig, "fig3d_frequency")


def plot_rpe_km(cases):
    n = len(cases); ncol = min(5, n); nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), squeeze=False)
    for ax, c in zip(axes.ravel(), cases):
        R0 = c["P"]["R0"]
        ax.plot(c["tR"] * T_SCALE, c["Rn"], "-", color="tab:blue", lw=1.0, label="sim")
        for key, sty, col in (("rpe", "--", "0.25"), ("km", ":", "tab:orange")):
            t, R = c[key]
            if t.size:
                ax.plot(t * T_SCALE, R / R0, sty, color=col, lw=1.1, label=key.upper())
        ax.axvline(c["seed"][0] * T_SCALE, color="tab:red", lw=0.6, alpha=0.6)
        ax.set_title(f"{R0*R_SCALE:.3f} {R_UNIT}  {c['regime']}", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.supxlabel(f"Time ({T_UNIT})"); fig.supylabel(r"$R/R_0$")
    fig.suptitle("Simulation vs Marmottant RPE / Keller-Miksis (seeded at red line)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig3d_rpe_km")

# ============================================================================
# MAIN
# ============================================================================

def main():
    root = next((r for r in _CANDIDATE_ROOTS if r and os.path.isdir(r)), None)
    if root is None:
        print("no 3D_Bubble output root found; set MARM3D_ROOT"); return 1
    dirs = sorted(d for d in glob.glob(os.path.join(root, "output_R*")) if os.path.isdir(d))
    print("=" * 96); print("3D MARMOTTANT BUBBLE -- LAPLACE SWEEP ANALYSIS"); print("=" * 96)
    print(f"  root: {root}\n  cases found: {len(dirs)}")
    os.makedirs(IMG_DIR, exist_ok=True)

    cases = []
    for d in dirs:
        print(f"  loading {os.path.basename(d)} ...")
        c = analyse_case(d)
        if c:
            cases.append(c)
    if not cases:
        print("  nothing analysable yet."); return 1
    cases.sort(key=lambda c: c["P"]["R0"])
    _style()

    hdr = (f"  {'R0 mm':>7} {'regime':>9} {'t_end ms':>9} {'done':>5} {'sig_exact':>10} {'sig_meas':>10} "
           f"{'err%':>7} {'bias%':>6} {'drift%':>7} {'dp_exact':>9} {'dp_meas':>9} {'f_th kHz':>9} {'f_sim kHz':>10} {'series':>10}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    rows = []
    for c in cases:
        P = c["P"]
        live = [r for r in c["pf"] if r["t"] > 0.0 and np.isfinite(r["sig"])]
        last = live[-1] if live else {}
        sm = last.get("sig", np.nan)
        err = 100 * (sm - c["sig_exact"]) / c["sig_exact"] if c["sig_exact"] > 0 and np.isfinite(sm) else np.nan
        dpm = (np.nanmean(c["pgas"][-max(1, len(c["pgas"]) // 5):]) - P["pinf"]) if "pgas" in c else np.nan
        dR = 100 * (c["Rn"][-1] - 1)
        print(f"  {P['R0']*1e3:7.3f} {c['regime']:>9} {c['t_end']*1e3:9.3f} {('yes' if c['complete'] else 'no'):>5} "
              f"{c['sig_exact']:10.6f} {sm:10.6f} {err:7.2f} {c['bias_pct']:6.3f} {dR:7.3f} {c['dp_exact']:9.3f} {dpm:9.3f} "
              f"{c['f_theory']/1e3:9.4f} {c['f_sim']/1e3:10.4f} {c['series'][:10]:>10}")
        rows.append(dict(R0_m=P["R0"], regime=c["regime"], t_end_s=c["t_end"], complete=c["complete"],
                         sigma_exact=c["sig_exact"], sigma_meas=sm, sigma_err_pct=err, radius_bias_pct=c["bias_pct"], drift_pct_end=dR,
                         dp_exact_Pa=c["dp_exact"], dp_meas_Pa=dpm, f_theory_Hz=c["f_theory"],
                         f_sim_Hz=c["f_sim"], series=c["series"]))
    csvp = os.path.join(IMG_DIR, "summary_3d.csv")
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n  wrote Images/summary_3d.csv")

    plot_radius(cases)
    if any(c["pf"] for c in cases):
        plot_shell(cases)
    plot_laplace(cases)
    plot_frequency(cases)
    plot_rpe_km(cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
