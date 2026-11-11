#!/usr/bin/env python3
"""
cattaneo_06_common -- shared analysis for the MarmPaper 06 validation decks
(input_06_Cattaneo_HighPa / input_06_Cattaneo_LowPa) against Cattaneo et al.,
Nature Physics 21:590 (2025), Fig. 2a.  Used by analyze_06_Cattaneo_{HighPa,LowPa}.py.

Curves on the figure (R in um vs t in us; t = 0 at the start of the burst, as in the deck):
  * Coated Keller-Miksis and coated Rayleigh-Plesset (Marmottant shell, polytropic gas), the
    same equations as tests/FlowMarmottant/reference/marmottant_rpe_km.py (solve_km / solve_rpe),
    re-integrated here with the MEASURED pulse envelope (that module only takes a constant-
    amplitude sine) and, when the run has shell.kappa_s_alpha > 0, the run's Cross-law shell
    viscosity kappa_eff = kappa_s / (1 + alpha |Rdot|/R).  With envelope = 1 and alpha = 0 they
    reduce exactly to solve_km / solve_rpe (checked by `python3 cattaneo_06_common.py`).
  * Cattaneo et al.'s own model: modified Rayleigh-Plesset (Keller-type R/c dp/dt term) with
    the Marmottant shell (constant kappa_s) and Zhou's heat-conducting three-layer gas
    (ModifiedRayleighPlesset_MarmottantZhou.m in github.com/cttnmrc/jetting-enables-sonoporation),
    Python port; paper parameters.
  * Experiment: Fig. 2a markers (cattaneo2025_fig2a_exp_{hi,lo}Pa.dat, t in figure time),
    shifted by EXP_SHIFT_US = -1.6 us onto the simulation time axis.
  * Simulation: volumetric radius R_vol and the eta = 0.5 radius along +x.
      RADIAL slab (n_cell N x 2 x 2): the cell row next to the axis, r = sqrt(x^2 + 2 y0^2),
          R_vol = (3 sum (1 - eta) r^2 dr)^(1/3), R(eta=0.5) by linear interpolation.
      Octant / full 3D: Sch20Analysis (tests/FlowRayleighPlesset/reference/sch20_analysis.py)
          'R_vol' (gas volume in a box around the bubble) and 'R_ray' (eta = 0.5 along +x).
"""
import glob
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))           # .../tests
_RP_REF = os.path.join(_TESTS, "FlowRayleighPlesset", "reference")
_M_REF = os.path.join(_TESTS, "FlowMarmottant", "reference")
for _p in (_RP_REF, _M_REF, os.path.join(_M_REF, "Debug")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sch20_analysis                                                       # noqa: E402  (style, octant reader)
from sch20_analysis import STYLE, _apply_rc, plt                            # noqa: E402
import marmottant_sch20_common as _M                                        # noqa: E402  (deck parser)
from marmottant_rpe_km import Shell, Liquid, Gas                            # noqa: E402

ENVELOPE = os.path.join(_HERE, "cattaneo2025_envelope.dat")
EXP = {"hi": os.path.join(_HERE, "cattaneo2025_fig2a_exp_hiPa.dat"),
       "lo": os.path.join(_HERE, "cattaneo2025_fig2a_exp_loPa.dat")}
EXP_SHIFT_US = -1.6     # experiment t_fig -> simulation t (burst start at t = 0); alignment used in our comparisons
F_DRIVE = 1.0e6         # Hz (paper)
NCP = 40                # burst length in cycles (envelope file covers 0..40 us)
CYCLE_T0 = -0.4e-6      # peak-table cycle windows [k - 0.4, k + 0.6) us

# Figure look: the Sch20Analysis STYLE, plus the colours of the extra curves here.
FIG = dict(STYLE)
FIG.update(c_catt="k", c_exp="#cb181d", c_eta="tab:orange", size=(6.5, 5.6), height_ratios=(1.0, 1.0))


# =========================================================================================== #
#  Drive: measured pulse envelope, applied as the deck / their MATLAB does
# =========================================================================================== #
class Envelope:
    """p_ac(t) = pa sin(w t) * (up(t) if sin > 0 else lo(t)), 0 outside 0..NCP/f."""

    def __init__(self, path=ENVELOPE, f=F_DRIVE, ncp=NCP):
        E = np.loadtxt(path)
        self.t, self.up, self.lo = E[:, 0], E[:, 1], E[:, 2]
        self.dup, self.dlo = np.gradient(self.up, self.t), np.gradient(self.lo, self.t)
        self.w, self.tend = 2.0 * np.pi * f, ncp / f

    def p(self, t, pa):
        if t < 0.0 or t > self.tend:
            return 0.0
        s = np.sin(self.w * t)
        return pa * s * np.interp(t, self.t, self.up if s > 0 else self.lo, right=0.0)

    def dpdt(self, t, pa):
        if t < 0.0 or t > self.tend:
            return 0.0
        s, c = np.sin(self.w * t), np.cos(self.w * t)
        e, de = ((self.up, self.dup) if s > 0 else (self.lo, self.dlo))
        return pa * (self.w * c * np.interp(t, self.t, e, right=0.0) + s * np.interp(t, self.t, de, right=0.0))


# =========================================================================================== #
#  Deck (+ the run's own metadata, which carries command-line overrides)
# =========================================================================================== #
_META_KEYS = re.compile(r"^(shell\.kappa_s|shell\.kappa_s_alpha|shell\.mu_s|marmottant(\.\w+)?|mu0|sigma|"
                        r"apply_surface_tension|eos[01]\.\w+|stop_time|nscbc\.\w+\.drive_(amp|omega)|"
                        r"thermal\.(conduction|exchange))$")


def read_metadata(plotdir):
    """Physics parameters from <plotdir>/metadata (last assignment wins, as the solver sees it)."""
    out = {}
    f = os.path.join(plotdir, "metadata")
    if not os.path.isfile(f):
        return out
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"^([\w.]+)\(nvals\s*=\s*\d+\)\s*::\s*\[(.*)\]$", line)
            if m:
                k, v = m.group(1), m.group(2)
            elif "=" in line and not line.startswith("#"):
                k, v = (s.strip() for s in line.split("=", 1))
            else:
                continue
            if _META_KEYS.match(k):
                out[k] = v
    return out


@dataclass
class Case:
    R0: float
    pa: float
    shell: Shell
    liq: Liquid
    gas: Gas
    alpha: float            # Cross-law time constant [s] (0 = constant kappa_s)
    stop_time: float
    meta: dict
    overrides: dict


def load_case(deck, plotdir=None):
    kv = _M.parse_input(deck)
    ov = read_metadata(plotdir) if plotdir else {}
    changed = {k: (kv.get(k), v) for k, v in ov.items() if k in kv and _f(kv[k]) != _f(v)}
    kv.update(ov)
    shell, liq, gas, meta = _M.build_case(kv, deck)
    amps = {float(v) for k, v in kv.items() if re.match(r"nscbc\.\w+\.drive_amp$", k) and float(v) != 0.0}
    if len(amps) != 1:
        raise ValueError("expected one nonzero nscbc.*.drive_amp, got %s" % amps)
    return Case(R0=meta["R0"], pa=amps.pop(), shell=shell, liq=liq, gas=gas,
                alpha=float(_M._num(kv, "shell.kappa_s_alpha", 0.0)), stop_time=meta["stop_time"],
                meta=meta, overrides=changed)


def _f(v):
    try:
        return float(str(v).split()[0])
    except ValueError:
        return v


# =========================================================================================== #
#  Coated KM / RPE with the enveloped drive (marmottant_rpe_km equations)
# =========================================================================================== #
def _pB(R, Rd, C):
    """p_B = p_g - 2 sigma/R - 4 mu Rd/R - 4 kappa_eff Rd/R^2, kappa_eff = kappa_s/(1 + alpha|Rd|/R);
    returns (p_B, dp_B/dR, dp_B/dRd)."""
    sh, liq, gas = C.shell, C.liq, C.gas
    pg = gas.p_g0 * (C.R0 / R) ** (3.0 * gas.kappa_g)
    sig = sh.sigma(R)
    dsig = 0.0
    if sh.chi > 0.0 and sh.R_buck < R < sh.R_rupture:
        dsig = 2.0 * sh.chi * R / sh.R_buck ** 2
    D = R * R + C.alpha * abs(Rd) * R                       # 4 kappa Rd / D  is the shell term
    pB = pg - 2.0 * sig / R - 4.0 * liq.mu * Rd / R - 4.0 * sh.kappa_s * Rd / D
    dR = (-3.0 * gas.kappa_g * pg / R - 2.0 * (dsig / R - sig / R ** 2) + 4.0 * liq.mu * Rd / R ** 2
          + 4.0 * sh.kappa_s * Rd * (2.0 * R + C.alpha * abs(Rd)) / D ** 2)
    dRd = -4.0 * liq.mu / R - 4.0 * sh.kappa_s * R * R / D ** 2
    return pB, dR, dRd


def solve_coated(C, env, t_end, model="km", t_eval=None, envelope=True, max_step=2e-9):
    """Coated Keller-Miksis ('km') or Rayleigh-Plesset ('rpe') from rest at R0, t = 0..t_end.
    envelope=False drives with the constant-amplitude sine (the marmottant_rpe_km drive)."""
    liq, pa = C.liq, C.pa
    if envelope:
        pac, dpac = (lambda t: env.p(t, pa)), (lambda t: env.dpdt(t, pa))
    else:
        w = env.w
        pac, dpac = (lambda t: pa * np.sin(w * t)), (lambda t: pa * w * np.cos(w * t))

    def rhs(t, y):
        R, Rd = y
        pB, dR, dRd = _pB(R, Rd, C)
        if model == "rpe":
            return [Rd, (pB - liq.p_inf - pac(t)) / (liq.rho * R) - 1.5 * Rd ** 2 / R]
        A = (1.0 - Rd / liq.c) * R - (R / (liq.rho * liq.c)) * dRd
        B = (-1.5 * (1.0 - Rd / (3.0 * liq.c)) * Rd ** 2
             + (1.0 + Rd / liq.c) * (pB - liq.p_inf - pac(t)) / liq.rho
             + (R / (liq.rho * liq.c)) * (dR * Rd - dpac(t)))
        return [Rd, B / (A + 1e-30)]

    te = np.linspace(0.0, t_end, int(round(t_end / 2e-9)) + 1) if t_eval is None else t_eval
    s = solve_ivp(rhs, (0.0, t_end), [C.R0, 0.0], method="LSODA", t_eval=te, rtol=1e-9, atol=1e-12,
                  max_step=max_step)
    return s.t, s.y[0]


# =========================================================================================== #
#  Cattaneo et al. (2025) model: Marmottant shell + Zhou heat-conducting gas (Python port)
# =========================================================================================== #
CATT = dict(R0=2.95e-6, p0=102200.0, c=1481.0, mu=0.000954, rho=997.8, Sw=0.0728, Gamma=1.4, chi=0.15,
            kap=5e-9, lam=0.0259, Rg=287.0, Ts=295.0, zeta=7.0)


def solve_cattaneo(pa, env, t_end, t_start=-1e-6, P=None):
    """Their ModifiedRayleighPlesset_MarmottantZhou: state (R, Rdot, p_gas, M1, M3) of a three-layer
    heat-conducting gas core (Zhou), sigma(R0) = 0 (R_buck = R0).  Returns (t, R) [s, m]."""
    P = dict(CATT, **(P or {}))
    R0, p0, c, mu, rho, Sw, G, chi, kap = (P[k] for k in ("R0", "p0", "c", "mu", "rho", "Sw", "Gamma", "chi", "kap"))
    Rg, Ts, z = P["Rg"], P["Ts"], P["zeta"]
    Lam = (G - 1.0) * P["lam"] / G
    y1 = z ** 2 / (z ** 2 + z + 1); y2 = (z ** 2 + z) / (z ** 2 + z + 1)
    a1 = y1 ** 3 / 3; a2 = (y2 ** 3 - y1 ** 3) / 3; a3 = (1 - y2 ** 3) / 3
    c1 = y1 ** 4 / (4 * a1); c2 = (y2 ** 4 - y1 ** 4) / (4 * a2); c3 = (1 - y2 ** 4) / (4 * a3)
    b1, b2, bs = c2 - c1, c3 - c2, 1 - c3
    Rb = R0; Rr = Rb * np.sqrt(1 + Sw / chi)
    M0 = p0 / (Rg * Ts) * R0 ** 3 / 3

    def rhs(t, x):
        R, Rd, p, M1, M3 = x
        M2 = M0 - M1 - M3
        r1 = M1 / (a1 * R ** 3); r2 = M2 / (a2 * R ** 3); r3 = M3 / (a3 * R ** 3)
        T1, T2, T3 = p / (r1 * Rg), p / (r2 * Rg), p / (r3 * Rg)
        G1 = (T2 - T1) / (b1 * R); G2 = (T3 - T2) / (b2 * R); Gs = (Ts - T3) / (bs * R)
        u1 = Lam / p * (G1 - y1 * Gs); u2 = Lam / p * (G2 - y2 * Gs)
        r1u = r1 if u1 >= 0 else r2; r2u = r2 if u2 >= 0 else r3
        sig = 0.0 if R < Rb else (chi * (R ** 2 / Rb ** 2 - 1) if R <= Rr else Sw)
        dp = 3 * G / R * (Lam * Gs - p * Rd)
        Rdd = (((p + R / c * dp - 2 * sig / R - 4 * mu * Rd / R - 4 * kap * Rd / R ** 2 - (p0 + env.p(t, pa)))
                / rho) - 1.5 * Rd ** 2) / R
        return [Rd, Rdd, dp, -y1 ** 2 * r1u * u1 * R ** 2, y2 ** 2 * r2u * u2 * R ** 2]

    s = solve_ivp(rhs, (t_start, t_end), [R0, 0, p0, 3 * a1 * M0, 3 * a3 * M0], method="LSODA",
                  max_step=1e-9, rtol=1e-8, atol=1e-14)
    return s.t, s.y[0]


# =========================================================================================== #
#  Experiment
# =========================================================================================== #
def load_experiment(which, shift_us=EXP_SHIFT_US):
    """(t [s], R [m]) of the Fig. 2a markers on the simulation time axis."""
    X = np.loadtxt(EXP[which], delimiter=",", comments="#")
    return (X[:, 0] + shift_us) * 1e-6, X[:, 1] * 1e-6


# =========================================================================================== #
#  Simulation radii
# =========================================================================================== #
def _plotfiles(d):
    fs = [p for p in glob.glob(os.path.join(d, "*cell")) if os.path.isdir(p)]
    return sorted(fs, key=lambda s: int(re.search(r"(\d+)cell$", s).group(1)))


def geometry(plotdir):
    """'slab' (RADIAL 1-D slab, <= 4 cells across), 'octant' or 'full'."""
    fs = _plotfiles(plotdir)
    if not fs:
        raise FileNotFoundError("no plotfiles (*cell) in %s" % plotdir)
    yt = sch20_analysis._yt()
    ds = yt.load(fs[0])
    n = np.array(ds.domain_dimensions)
    if ds.dimensionality == 3 and n[1] <= 4 and n[2] <= 4:
        return "slab"
    return "octant" if np.all(np.array([float(v) for v in ds.domain_left_edge]) > -1e-12) else "full"


def _slab_frame(pf):
    yt = sch20_analysis._yt()
    ds = yt.load(pf)
    cg = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    x = cg["index", "x"].d[:, 0, 0]
    y0 = cg["index", "y"].d[0, 0, 0]
    r = np.sqrt(x ** 2 + 2 * y0 ** 2)
    dr = np.gradient(r)
    eta = np.clip(cg["boxlib", "eta"].d[:, 0, 0], 0, 1)
    Rv = (3 * np.sum((1 - eta) * r ** 2 * dr)) ** (1 / 3)
    i = int(np.argmax(eta > 0.5))
    R5 = np.nan
    if i > 0:
        w = (0.5 - eta[i - 1]) / (eta[i] - eta[i - 1])
        R5 = r[i - 1] + w * (r[i] - r[i - 1])
    return float(ds.current_time), Rv, R5


def sim_radii(plotdir, deck, geom=None, nproc=4, figdir=None, stem="06"):
    """dict(t, R_vol, R_eta, geom) from the plotfiles; raw radii [m]."""
    geom = geom or geometry(plotdir)
    if geom == "slab":
        rows = sch20_analysis._pool_map(_slab_frame, _plotfiles(plotdir), nproc)
        t, Rv, R5 = (np.array(c, float) for c in zip(*rows))
    else:
        # Sch20Analysis frame pass: R_vol from the gas volume in a box (x 8 in the octant), R_ray along +x.
        # Windows widened for a bubble that grows to ~2.4 R0; shape directions reduced (unused here).
        A = sch20_analysis.Sch20Analysis(deck, plotdir, case="Driven", coated=True, stem=stem + "_sch20",
                                         out_dir=figdir, nproc=nproc, verbose=False,
                                         style=dict(ray_xmax_R0=3.0, box_R0=3.0, n_dirs=24))
        t, Rv, R5 = A.col("t"), A.col("R_vol"), A.col("R_ray")
    return dict(t=t, R_vol=Rv, R_eta=R5, geom=geom)


def thermo_radius(plotdir, R0, geom):
    """R(t) from thermo.dat gas_volume, normalised to R0 at t = 0 (km.sim convention).
    Slab: R0 V/V0 (the slab volume is linear in R); octant/full: R0 (V/V0)^(1/3)."""
    f = os.path.join(plotdir, "thermo.dat")
    if not os.path.isfile(f):
        return None
    d = np.loadtxt(f, skiprows=1)
    q = d[:, 1] / d[0, 1]
    return d[:, 0], R0 * (q if geom == "slab" else np.cbrt(q))


# =========================================================================================== #
#  Peaks / table
# =========================================================================================== #
def cycle_extrema(t, R, ncyc, t0=CYCLE_T0, period=1.0 / F_DRIVE):
    """Max and min of R in each drive cycle [t0 + k T, t0 + (k+1) T), k = 0..ncyc-1 (NaN if not covered).
    t0 = -0.4 us puts both the expansion maximum (~k - 0.05 us) and the collapse minimum (~k + 0.3 us)
    of cycle k well inside the window."""
    mx, mn = np.full(ncyc, np.nan), np.full(ncyc, np.nan)
    edge = np.zeros((2, ncyc), bool)          # extremum on the first/last sample of the window (not a turning point)
    t, R = np.asarray(t), np.asarray(R)
    ok = np.isfinite(R)
    for k in range(ncyc):
        a, b = t0 + k * period, t0 + (k + 1) * period
        m = ok & (t >= a) & (t < b)
        if m.sum() >= 3 and t[ok].max() >= b - 0.15 * period:
            Rm = R[m]
            mx[k], mn[k] = Rm.max(), Rm.min()
            edge[0, k] = int(np.argmax(Rm)) in (0, len(Rm) - 1)
            edge[1, k] = int(np.argmin(Rm)) in (0, len(Rm) - 1)
    return mx, mn, edge


def print_table(curves, exp_name, ncyc, first=1):
    """curves: list of (name, t, R) in s, m.  Prints per-cycle peak and trough radii [um] and % gap vs experiment."""
    ext = {n: cycle_extrema(t, R, ncyc) for n, t, R in curves}
    ex_mx, ex_mn, _ = ext[exp_name]
    names = [n for n, _, _ in curves]
    w = max(len(n) for n in names) + 2
    for lab, j, ex in (("PEAK", 0, ex_mx), ("TROUGH", 1, ex_mn)):
        print("\n  %s radius per drive cycle [um]  (cycle k = [k - 0.4, k + 0.6) us; %% = gap vs experiment)" % lab)
        print("  " + "cycle".ljust(w) + "".join("%13d " % k for k in range(first, ncyc)))
        for n in names:
            v, e = ext[n][j], ext[n][2][j]
            cells = []
            for k in range(first, ncyc):
                f = "*" if e[k] else " "
                if not np.isfinite(v[k]):
                    cells.append("%14s" % "-")
                elif n == exp_name or not np.isfinite(ex[k]):
                    cells.append("%13.3f%s" % (v[k] * 1e6, f))
                else:
                    cells.append(" %6.3f(%+5.1f%%)%s" % (v[k] * 1e6, 100 * (v[k] - ex[k]) / ex[k], f))
            print("  " + n.ljust(w) + "".join(cells))
    print("  * = extremum on the edge of the cycle window (still rising/falling there, not a turning point)")


# =========================================================================================== #
#  Figure
# =========================================================================================== #
def make_figure(case_tag, C, sim, models, exp, out_stem, xlim, zoom, rvol_norm=False, thermo=None,
                show_thermo=False, formats=None, single=False):
    """single=True: one full-run panel only (no zoom), same width, about half the height."""
    S = FIG
    _apply_rc(S)
    if single:
        fig, ax1 = plt.subplots(1, 1, figsize=(S["size"][0], 0.5 * S["size"][1] + 0.35))
        axs, windows = [ax1], (xlim,)
    else:
        fig, axs = plt.subplots(2, 1, figsize=S["size"], gridspec_kw=dict(height_ratios=S["height_ratios"]))
        windows = (xlim, zoom)
    um = 1e6
    hi = case_tag == "hi"
    Rv = sim["R_vol"] * (C.R0 / sim["R_vol"][0] if rvol_norm else 1.0)
    for k, (ax, (t0, t1)) in enumerate(zip(axs, windows)):
        mk = lambda t: (t >= (t0 - 0.05) * 1e-6) & (t <= (t1 + 0.05) * 1e-6)
        tk, Rk = models["km"]; tr, Rr = models["rpe"]; tc, Rc = models["catt"]
        ax.plot(tk[mk(tk)] * um, Rk[mk(tk)] * um, "-", color=S["c_km"], lw=1.2, label="Coated Keller–Miksis")
        ax.plot(tr[mk(tr)] * um, Rr[mk(tr)] * um, "--", color=S["c_rp"], lw=1.2, label="Coated Rayleigh–Plesset")
        ax.plot(tc[mk(tc)] * um, Rc[mk(tc)] * um, "-.", color=S["c_catt"], lw=1.2,
                label="Cattaneo et al. model")
        if show_thermo and thermo is not None:
            th, Rh = thermo
            ax.plot(th[mk(th)] * um, Rh[mk(th)] * um, ":", color=S["c_vol"], lw=1.0,
                    label=r"Simulation, $R_\mathrm{vol}$ (thermo)")
        ts = sim["t"]; m = mk(ts)
        ax.plot(ts[m] * um, Rv[m] * um, "-s", color=S["c_vol"], lw=1.3, ms=2.6,
                label=r"Simulation, $R_\mathrm{vol}$")
        ax.plot(ts[m] * um, sim["R_eta"][m] * um, "-^", color=S["c_eta"], lw=1.3, ms=2.8,
                label=r"Simulation, $R(\eta = 0.5)$")
        te, Re = exp; m = mk(te)
        ax.plot(te[m] * um, Re[m] * um, "o", color=S["c_exp"], mfc=S["c_exp"] if hi else "white", mew=0.9,
                ms=4.0 if k else 3.4, label="Experiment (Cattaneo et al.)", zorder=5)
        ax.set_xlim(t0, t1)
        ax.set_xlabel(r"$t$  [µs]")
        ax.set_ylabel(r"$R$  [µm]")
    if not single and (zoom[0] > xlim[0] or zoom[1] < xlim[1]):   # zoom window marked in (a)
        axs[0].axvspan(zoom[0], zoom[1], color="0.93", lw=0, zorder=0)
    h, l = axs[0].get_legend_handles_labels()
    axs[0].legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=S["legend_size"],
                  columnspacing=1.2, handlelength=2.4)
    for ax, s in zip(axs if not single else [], ("(a)", "(b)")):
        ax.text(0.012, 0.965, s, transform=ax.transAxes, ha="left", va="top", fontsize=S["label_size"])
    fig.tight_layout()
    for fmt in (formats or S["formats"]):
        fig.savefig(out_stem + "." + fmt, dpi=S["dpi"], bbox_inches="tight", pad_inches=S["pad_inches"])
    plt.close(fig)
    print("  wrote", out_stem + ".png")


# =========================================================================================== #
#  Driver shared by the two scripts
# =========================================================================================== #
def main(case_tag, deck, default_output, stem, zoom_default):
    import argparse
    ap = argparse.ArgumentParser(description=sys.modules["__main__"].__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plotdir", nargs="?", default=None, help="plotfile directory (same as --output)")
    ap.add_argument("--output", default=None, help="plotfile directory (default: the deck's plot_file)")
    ap.add_argument("--input", default=deck, help="input deck (models; overridden by <output>/metadata)")
    ap.add_argument("--figdir", default=None, help="image directory (default ./Images next to this script)")
    ap.add_argument("--stem", default=stem)
    ap.add_argument("--tmax", type=float, default=None, help="full-panel end time [us] (default: last frame)")
    ap.add_argument("--zoom", type=float, nargs=2, default=zoom_default, metavar=("T0", "T1"),
                    help="zoom-panel window [us] (default %s %s)" % tuple(zoom_default))
    ap.add_argument("--exp-shift", type=float, default=EXP_SHIFT_US,
                    help="experiment time shift [us], t_sim = t_fig + shift (default %g)" % EXP_SHIFT_US)
    ap.add_argument("--kappa-g", type=float, default=None,
                    help="polytropic exponent of the KM/RPE gas (default: deck eos1.gamma; 1 = isothermal)")
    ap.add_argument("--rvol-norm", action="store_true", help="rescale R_vol so R_vol(0) = R0")
    ap.add_argument("--thermo", action="store_true", help="also draw R_vol from thermo.dat (km.sim measure)")
    ap.add_argument("--geom", choices=("slab", "octant", "full"), default=None, help="override geometry detection")
    ap.add_argument("--nproc", type=int, default=4)
    ap.add_argument("--formats", default=None, help="comma list, default %s" % ",".join(FIG["formats"]))
    a = ap.parse_args()

    kv = _M.parse_input(a.input)
    plotdir = a.plotdir or a.output or kv.get("plot_file") or default_output
    figdir = a.figdir or os.path.join(_HERE, "Images")
    os.makedirs(figdir, exist_ok=True)

    C = load_case(a.input, plotdir)
    if a.kappa_g is not None:
        C.gas.kappa_g = a.kappa_g
    print("  deck   %s\n  output %s" % (a.input, plotdir))
    print("  p_a %.4g Pa  R0 %.4g m  kappa_s %.3g kg/s  Cross alpha %.3g s  chi %.3g  kappa_g(KM/RPE) %.3g"
          % (C.pa, C.R0, C.shell.kappa_s, C.alpha, C.shell.chi, C.gas.kappa_g))
    for k, (v0, v1) in C.overrides.items():
        print("  run override from metadata: %s = %s (deck %s)" % (k, v1, v0))

    sim = sim_radii(plotdir, a.input, geom=a.geom, nproc=a.nproc, figdir=figdir, stem=os.path.join(a.stem))
    print("  geometry %s, %d frames, t = %.3f .. %.3f us"
          % (sim["geom"], len(sim["t"]), sim["t"][0] * 1e6, sim["t"][-1] * 1e6))
    t_end = (a.tmax * 1e-6) if a.tmax else float(np.nanmax(sim["t"]))
    t_mod = max(t_end, a.zoom[1] * 1e-6) + 0.2e-6

    env = Envelope()
    models = {}
    for m in ("km", "rpe"):
        models[m] = solve_coated(C, env, t_mod, model=m)
    models["catt"] = solve_cattaneo(C.pa, env, t_mod)
    exp = load_experiment(case_tag, a.exp_shift)
    th = thermo_radius(plotdir, C.R0, sim["geom"])

    Rv = sim["R_vol"] * (C.R0 / sim["R_vol"][0] if a.rvol_norm else 1.0)
    curves = [("Experiment", *exp),
              ("Sim R_vol", sim["t"], Rv),
              ("Sim R(eta=0.5)", sim["t"], sim["R_eta"])]
    if th is not None:
        curves.append(("Sim thermo.dat (km.sim)", *th))
    curves += [("Coated KM", *models["km"]), ("Coated RPE", *models["rpe"]), ("Cattaneo model", *models["catt"])]
    ncyc = int(np.floor(t_end * F_DRIVE - CYCLE_T0 * F_DRIVE - 0.85 + 1e-9)) + 1   # last window ends <= t_end + 0.15 T
    print("\n  R_vol(t=0) = %.4f um (%s)" % (Rv[0] * 1e6, "rescaled to R0" if a.rvol_norm else "raw; diffuse-band offset"))
    print_table(curves, "Experiment", max(ncyc, 2), first=1)

    xlim = (-0.3, round(t_end * 1e6, 2))
    make_figure(case_tag, C, sim, models, exp, os.path.join(figdir, a.stem + "_R"), xlim, tuple(a.zoom),
                rvol_norm=a.rvol_norm, thermo=th, show_thermo=a.thermo,
                formats=a.formats.split(",") if a.formats else None)
    # compact version for page count: the full run only, no zoom panel
    make_figure(case_tag, C, sim, models, exp, os.path.join(figdir, a.stem + "_R_full"), xlim, tuple(a.zoom),
                rvol_norm=a.rvol_norm, thermo=th, show_thermo=a.thermo,
                formats=a.formats.split(",") if a.formats else None, single=True)


# =========================================================================================== #
#  Self-check: envelope-free, alpha = 0 coated KM / RPE == marmottant_rpe_km.solve_km / solve_rpe
# =========================================================================================== #
if __name__ == "__main__":
    from marmottant_rpe_km import solve_km, solve_rpe
    deck = os.path.normpath(os.path.join(_HERE, "..", "input_06_Cattaneo_HighPa"))
    C = load_case(deck)
    C.alpha = 0.0
    env = Envelope()
    te = np.linspace(0.0, 3e-6, 3001)
    for name, ref in (("km", solve_km), ("rpe", solve_rpe)):
        t1, R1 = solve_coated(C, env, te[-1], model=name, t_eval=te, envelope=False)
        t2, R2, _ = ref(C.R0, 0.0, (0.0, te[-1]), C.shell, C.liq, C.gas, drive=(C.pa, env.w), t_eval=te)
        print("  %s: max |R - %s| / R0 = %.2e" % (name, ref.__name__, np.max(np.abs(R1 - R2)) / C.R0))
