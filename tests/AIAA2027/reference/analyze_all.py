#!/usr/bin/env python3
"""
===============================================================================
AIAA SciTech 2027 -- analysis of every shock / droplet run in the AIAA output tree
===============================================================================

Scans an output tree (default: the INCLINE AIAA2027 folder) for every run (a folder holding AMReX plotfiles
NNNNNcell/), reads the initial conditions from the run's own `metadata` file (the exact parameters the solver ran
with), reads every plotfile ONCE, and then draws everything from that single pass on N worker processes:

  per run      split GIFs that move with the liquid (top half / mirrored bottom half):
                 Schlieren | Pressure, Schlieren | Temperature, Schlieren | Velocity + streamlines (drop frame),
                 Schlieren | Vorticity, Schlieren | Mach, Vaporization Rate | Gas Mass Fraction (vaporizing runs)
               interface area overlapped with vaporization rate vs time (requested twin-axis + indexed version)
               eta = 0.5 contours at several times (lab frame, overlaid) and in the drop frame (panels)
               deformation, kinematics / drag coefficient, vaporization diagnostics (flux, film T, B_M),
               fragment count, shock x-t vs Rankine-Hugoniot, liquid p/T extremes, energy / circulation,
               mass budget, AMR cell counts, per-drop metrics for the tandem pair
  per pair     vaporization ON vs OFF: overlaid time series, contour panels, split comparison GIFs
  per sweep    all Mach numbers of one layout overlaid (vaporized fraction, area, flux, width, drift, speed)
  spreadsheet  AIAA2027_Metrics.xlsx (+ CSVs): README, initial conditions, summary metrics, vap vs no-vap,
               tandem pair, full time series; analysis_log.txt holds per-run notes and warnings

USAGE
  python3 analyze_all.py [ROOT] [--out DIR] [--nproc 8] [--every 1] [--only REGEX] [--skip REGEX]
                         [--maxpix 2000] [--no-frames] [--no-gifs] [--redo]
                         [--no-tar] [--tar-frames]
  ROOT    folder to scan (default /mmfs1/home/ttryon/flames/bin/tests/AIAA2027, else the repo's bin/tests/AIAA2027)
  --out   output folder (default ROOT/Analysis)
  --nproc worker processes for the read pass and for drawing (default 8; processes, not threads, because the
          Python GIL serializes threads)
  --every draw every Nth plotfile in the GIFs (the time series always use every plotfile)
  --only / --skip  regex on the run name, e.g. --only "C5_.*_long"
  --maxpix  largest image dimension of the stored window (pixels); the finest AMR level that fits is used
  --redo  ignore the cache (<out>/_cache) and redraw every frame
  --no-tar      do not write <out>.tar.gz at the end (by default the output folder is tarred next to itself,
                without _cache/ and the per-frame PNG folders, which the GIFs already hold)
  --tar-frames  include the per-frame PNG folders in the tar

CONVENTIONS
  eta = 1 gas, 0 liquid.  Liquid mass = int rho_eta1 dA; vaporization rate = int Vap_dot_rho dA (>= 0, liquid lost);
  interface length ("surface area" per unit depth in 2D) = int |grad eta| dA (exact for the diffuse profile and
  independent of the contour level; the eta = 0.5 contour length of the window is logged as a check).
  Half / quarter domains (symmetry at a lo boundary on a coordinate 0) are mirrored: every extensive quantity is
  reported for the FULL column, per unit depth.  Time tau = (t - t_impact)/t*, t* = D/u2 sqrt(rho_l/rho2)
  (Nicholls & Ranger 1969); runs without a shock use t in us.
===============================================================================
"""
import os, sys, re, glob, json, time, math, argparse, datetime, subprocess, traceback
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
AIAA_SRC = os.path.dirname(HERE)                       # tests/AIAA2027 (decks, fluids.py)
VERSION = "4"                                          # bump to invalidate the cache after changing extract()

# ----------------------------------------------------------------------------------------------------------------
# Plot style (paper: serif, fonts >= 13, Title Case titles, "Numerical" for simulation data)
# ----------------------------------------------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif", "Times", "Times New Roman"],
    "mathtext.fontset": "dejavuserif", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.grid": True, "grid.alpha": 0.25, "savefig.bbox": "tight", "figure.dpi": 100,
})
# categorical palette, fixed order (validated default palette of the dataviz skill, light surface)
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
C_VAP, C_NOVAP = PAL[1], PAL[0]
FRAME_DPI = 100
PLOT_DPI = 200
TAU_MARKS = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
VIEW_PAD = 1.0                     # image margin around the liquid extent [D]
VIEW_MIN = (2.5, 2.0)              # smallest image half-width, height [D]
SCHLIEREN_K = 6.0                  # schlieren = exp(-K |grad ln rho| / g_ref), g_ref = 99th percentile of the window
PF_RE = re.compile(r"^(\d+)cell$")

SPLITS = {   # kind: (top field, bottom field); bottom half is the mirror image
    "Schlieren-Pressure": ("schlieren", "pressure"),
    "Schlieren-Temperature": ("schlieren", "T"),
    "Schlieren-Velocity": ("schlieren", "velocity"),
    "Schlieren-Vorticity": ("schlieren", "vorticity"),
    "Schlieren-Mach": ("schlieren", "mach"),
    "VapRate-MassFraction": ("vaprate", "massfrac"),
}
CMP_KINDS = ["schlieren", "T", "pressure", "density"]   # vap (top) vs no-vap (bottom) comparison GIFs
WIN_FIELDS = ["eta", "density", "pressure", "T", "velocityx", "velocityy", "vorticity", "Vap_dot_rho",
              "Mass_Fraction", "a"]

LOG = []
def log(msg=""):
    print(msg, flush=True); LOG.append(msg)


# ================================================================================================================
# DISCOVERY AND INITIAL CONDITIONS
# ================================================================================================================
def plotfiles(run):
    pf = [d for d in os.listdir(run) if PF_RE.match(d) and os.path.isfile(os.path.join(run, d, "Header"))]
    return [os.path.join(run, d) for d in sorted(pf, key=lambda s: int(PF_RE.match(s).group(1)))]


def header_time(pf):
    with open(os.path.join(pf, "Header")) as fh:
        fh.readline(); n = int(fh.readline())
        for _ in range(n): fh.readline()
        fh.readline(); return float(fh.readline())


def discover(root, out):
    runs, outabs = [], os.path.abspath(out)
    for dp, dns, _ in os.walk(root, followlinks=False):
        if os.path.abspath(dp).startswith(outabs): dns[:] = []; continue
        if any(PF_RE.match(d) for d in dns):
            if plotfiles(dp): runs.append(dp)
            dns[:] = [d for d in dns if not PF_RE.match(d)]
        dns[:] = [d for d in dns if not d.endswith("_frames") and d not in ("_cache", "Analysis")]
    out_runs, seen = [], {}
    for r in sorted(runs):
        b = os.path.basename(os.path.normpath(r))
        name = os.path.basename(os.path.dirname(r)) if b == "output" else (b[7:] if b.startswith("output_") else b)
        rel = os.path.relpath(r, root).split(os.sep)
        group = rel[0] if len(rel) > 1 else "Runs"
        if name in seen: seen[name] += 1; name = f"{name}_{seen[name]}"
        else: seen[name] = 0
        out_runs.append(dict(run=r, name=name, group=group))
    return out_runs


def read_metadata(run):
    """key = value pairs of <run>/metadata (later duplicates win, as in the solver's ParmParse)."""
    fn = os.path.join(run, "metadata")
    if not os.path.isfile(fn):
        cand = glob.glob(os.path.join(run, "*cell", "metadata"))
        if not cand: return {}
        fn = cand[0]
    meta = {}
    for line in open(fn, errors="replace"):
        if "=" not in line or line.lstrip().startswith("#"): continue
        k, v = line.split("=", 1); k, v = k.strip(), v.strip()
        m = re.match(r"\(nvals\s*=\s*\d+\)\s*::\s*\[(.*)\]", v)
        if m: v = m.group(1)
        k = re.sub(r"\(nvals.*$", "", k).strip()
        meta[k] = v.strip('"').strip()
    return meta


def fnum(meta, key, default=float("nan")):
    try: return float(str(meta[key]).strip('"').split()[0])
    except (KeyError, ValueError, IndexError): return default


def fvec(meta, key):
    try: return [float(s) for s in str(meta[key]).split()]
    except (KeyError, ValueError): return []


_FLUIDS = None
def liquid_viscosity(fluid):
    """Liquid viscosity at 300 K: NIST tables via tests/AIAA2027/FuelDroplet/fluids.py; water 8.54e-4 Pa s
    (NIST WebBook, 300 K, 1 atm).  The runs are inviscid; this is only for Oh."""
    global _FLUIDS
    if fluid == "Water": return 8.54e-4
    if _FLUIDS is None:
        _FLUIDS = {}
        try:
            fd = os.path.join(AIAA_SRC, "FuelDroplet")
            g = dict(os=os, HERE=fd, FLUIDS=_FLUIDS); exec(open(os.path.join(fd, "fluids.py")).read(), g)
        except Exception as e:
            log(f"  [note] fluids.py not loaded ({e}); Oh left blank for fuels")
    key = {"n-Pentane": "pentane", "n-Dodecane": "dodecane"}.get(fluid)
    return _FLUIDS.get(key, {}).get("mu", float("nan")) if key else float("nan")


def sutherland(T):   # air viscosity, Sutherland's law (White, Viscous Fluid Flow)
    return 1.716e-5 * (T / 273.15) ** 1.5 * (273.15 + 110.4) / (T + 110.4)


def initial_conditions(R):
    """Everything about the setup, from the run's metadata (what ran), cross-referenced to the deck if found."""
    meta = read_metadata(R["run"]); name = R["name"]
    C = dict(name=name, group=R["group"], run=R["run"], meta_found=bool(meta))
    C["fluid"] = ("n-Pentane" if re.search(r"(^|_)C5_", name) else "n-Dodecane" if re.search(r"(^|_)C12_", name)
                  else "Water" if abs(fnum(meta, "eos1.gamma", 0) - 7.15) < 0.05 else "Liquid")
    C["vap"] = int(fnum(meta, "apply_vaporization", 0)); C["surface_tension"] = int(fnum(meta, "apply_surface_tension", 0))
    C["vap_model"] = meta.get("vaporization.model", "")
    g0 = fnum(meta, "eos0.gamma", 1.4); cv0 = fnum(meta, "eos0.cv", 717.86)
    C["gamma_g"], C["gamma_l"], C["pinf_l"] = g0, fnum(meta, "eos1.gamma"), fnum(meta, "eos1.p0")
    C["pinf_g"] = fnum(meta, "eos0.p0", 0.0)
    C["rho_l"] = fnum(meta, "density1.ic.expression.region0")
    C["sigma"] = fnum(meta, "sigma"); C["Lv"] = fnum(meta, "Lv")
    # interface: radius and column centres from the eta expression
    R0 = fnum(meta, "eta.ic.expression.constant.R0"); C["R0"] = R0; C["D"] = 2 * R0
    C["epsilon"] = fnum(meta, "eta.ic.expression.constant.epsilon")
    ex = meta.get("eta.ic.expression.region0", "")
    cen = ([0.0] if "sqrt(x**2" in ex.replace(" ", "") else []) + \
          [float(s) for s in re.findall(r"\(x\s*-\s*([-+0-9.eE]+)\)\*\*2", ex)]
    C["centers"] = sorted(cen) if cen else [0.0]; C["n_drops"] = len(C["centers"])
    C["gap"] = (C["centers"][1] - C["centers"][0] - C["D"]) if C["n_drops"] > 1 else float("nan")
    # far-field and post-shock gas
    p1 = fnum(meta, "pressure0.ic.expression.constant.p_atm", fnum(meta, "pressure0.ic.expression.region0", 101325.0))
    rho1 = fnum(meta, "density0.ic.expression.constant.rho_air", fnum(meta, "density0.ic.expression.region0"))
    p2 = fnum(meta, "pressure0.ic.expression.constant.p_shock"); rho2 = fnum(meta, "density0.ic.expression.constant.rho_shock")
    u2 = fnum(meta, "velocity0.ic.expression.constant.u_shock"); xs = fnum(meta, "pressure0.ic.expression.constant.x_shock")
    C.update(p1=p1, rho1=rho1, T1=p1 / (rho1 * (g0 - 1) * cv0), a1=math.sqrt(g0 * p1 / rho1))
    C["shock"] = bool(np.isfinite(p2) and np.isfinite(u2))
    if C["shock"]:
        Ms = math.sqrt(1 + (g0 + 1) / (2 * g0) * (p2 / p1 - 1))
        T2 = p2 / (rho2 * (g0 - 1) * cv0); a2 = math.sqrt(g0 * p2 / rho2); Us = Ms * C["a1"]
        C.update(Ms=Ms, p2=p2, rho2=rho2, u2=u2, T2=T2, a2=a2, M2=u2 / a2, Us=Us, x_shock0=xs,
                 T0_stag=T2 * (1 + 0.5 * (g0 - 1) * (u2 / a2) ** 2))
        C["t_impact"] = (C["centers"][0] - R0 - xs) / Us
        C["tstar"] = C["D"] / u2 * math.sqrt(C["rho_l"] / rho2)
        mu2 = sutherland(T2); C["mu_g2"] = mu2
        C["We"] = rho2 * u2 ** 2 * C["D"] / C["sigma"] if C["sigma"] > 0 else float("nan")
        C["Re"] = rho2 * u2 * C["D"] / mu2
    else:
        C.update(Ms=float("nan"), p2=p1, rho2=rho1, u2=float("nan"), T2=C["T1"], a2=C["a1"], M2=float("nan"),
                 Us=float("nan"), x_shock0=float("nan"), T0_stag=float("nan"), t_impact=0.0, tstar=float("nan"),
                 mu_g2=float("nan"), We=float("nan"), Re=float("nan"))
    mul = liquid_viscosity(C["fluid"]); C["mu_l"] = mul
    C["Oh"] = mul / math.sqrt(C["rho_l"] * C["sigma"] * C["D"]) if C["sigma"] > 0 and np.isfinite(mul) else float("nan")
    C["density_ratio"] = C["rho_l"] / C["rho2"]
    # time scale for the plots
    if C["shock"]: C.update(tscale=C["tstar"], t0=C["t_impact"], tlabel=r"$t/t^*$")
    else: C.update(tscale=1e-6, t0=0.0, tlabel=r"$t$ [$\mu$s]")
    # mesh
    lo, hi, nc = fvec(meta, "geometry.prob_lo"), fvec(meta, "geometry.prob_hi"), fvec(meta, "amr.n_cell")
    ml = int(fnum(meta, "amr.max_level", 0)); rr = int(fnum(meta, "amr.ref_ratio", 2))
    C.update(prob_lo=lo[:2], prob_hi=hi[:2], n_cell=[int(n) for n in nc[:2]], max_level=ml, ref_ratio=rr)
    dx0 = (hi[0] - lo[0]) / nc[0] if nc else float("nan"); C["dx0"] = dx0; C["dx_fine"] = dx0 / rr ** ml
    C["D_dx"] = C["D"] / C["dx_fine"]
    C["domain_D"] = f"x [{lo[0] / C['D']:.3g}, {hi[0] / C['D']:.3g}] D, y [{lo[1] / C['D']:.3g}, {hi[1] / C['D']:.3g}] D" if lo else ""
    # symmetry mirrors: lo boundary reflect on a coordinate 0 -> multiply extensive quantities by 2 per direction
    mir = 1
    for d, ax in enumerate("xy"):
        if "REFLECT" in meta.get(f"eta.bc.type.{ax}lo", "").upper() and lo and abs(lo[d]) < 1e-30: mir *= 2
    C["mirror"] = mir; C["mirror_y"] = "REFLECT" in meta.get("eta.bc.type.ylo", "").upper()
    C["stop_time"] = fnum(meta, "stop_time"); C["plot_dt"] = fnum(meta, "amr.plot_dt")
    C["t_end_tau"] = (C["stop_time"] - C["t0"]) / C["tscale"]
    C["nprocs"] = int(fnum(meta, "Number_of_processors", 0)); C["status"] = meta.get("Status", "")
    C["run_time_h"] = fnum(meta, "Simulation_run_time", float("nan")) / 3600
    C["core_hours"] = C["run_time_h"] * C["nprocs"] if C["nprocs"] else float("nan")
    C["git"] = meta.get("Git_commit_hash", ""); C["start"] = meta.get("Simulation_start_time", "")
    C["p_refinement_max_level"] = meta.get("p_refinement_max_level", "-1 (all)")
    deck = os.path.join(AIAA_SRC, R["group"], "input_" + name)
    C["deck"] = deck if os.path.isfile(deck) else ""
    C["deck_header"] = "".join(l for l in open(deck) if l.startswith("#")) if C["deck"] else ""
    # tokens for grouping
    C["vapbase"] = re.sub(r"_vap(?=_|$)", "", name)                          # vap/no-vap partner key
    C["layout"] = re.sub(r"(^|_)Ms[0-9.]+", r"\1", C["vapbase"]).strip("_")  # Mach-sweep key
    C["layout"] = re.sub(r"__+", "_", C["layout"])
    return C


def tau_of(C, t): return (np.asarray(t) - C["t0"]) / C["tscale"]


def tfmt(C, t):
    us = t * 1e6
    s = f"{us * 1e3:.1f} ns" if us < 1 or (np.isfinite(C["tstar"]) and C["tstar"] < 1e-6) else f"{us:.1f} $\\mu$s"
    return (f"$t/t^*$ = {tau_of(C, t):.2f}  ($t$ = {s})") if C["shock"] else f"$t$ = {s}"


# ================================================================================================================
# THE SINGLE READ PASS (one plotfile per worker call)
# ================================================================================================================
def _wq(x, w, q):
    """weighted quantile"""
    if x.size == 0: return float("nan")
    o = np.argsort(x); cw = np.cumsum(w[o]); return float(x[o][np.searchsorted(cw, q * cw[-1])].item()) if cw[-1] > 0 else float("nan")


def sg_temperature(T, p, eta, C):
    """Plotfile T (EOS::MixedTemperature) is (p + gamma_eff p0_eff)/(rho cv_eff (gamma_eff - 1)), which reads ~2x high in
    the liquid (590 K for the 300 K pentane).  Convert to the stiffened-gas temperature (p + p0_eff)/(rho cv_eff (gamma_eff - 1))
    that the decks were built with, using the solver's own mixture rules (Solver/EOS/EOS.H MixedGamma / MixedP0).
    Pure gas (p0 = 0) is unchanged."""
    g0, g1, q0, q1 = C["gamma_g"], C["gamma_l"], C["pinf_g"], C["pinf_l"]
    A = eta / (g0 - 1) + (1 - eta) / (g1 - 1); ge = 1 + 1 / A
    p0e = (eta * g0 * q0 / (g0 - 1) + (1 - eta) * g1 * q1 / (g1 - 1)) / (A * ge)
    return T * (p + p0e) / (p + ge * p0e)


def extract(job):
    """Pass 1: read one plotfile, global scalars on the leaf cells (incl. the extent of all liquid), cached to npz."""
    pf, cfile, C, redo = job
    if not redo and os.path.isfile(cfile) and os.path.getmtime(cfile) > os.path.getmtime(os.path.join(pf, "Header")):
        try:
            z = np.load(cfile, allow_pickle=False)
            if str(z["version"]) == VERSION: return json.loads(str(z["scalars"]))
        except Exception:
            pass
    try:
        import yt
        yt.funcs.mylog.setLevel(50)
        ds = yt.load(pf); ad = ds.all_data()
        have = {f for _, f in ds.field_list}
        def g(f, default=0.0):
            return np.asarray(ad["boxlib", f], dtype=np.float64) if f in have else np.full(n, default)
        eta = np.asarray(ad["boxlib", "eta"], dtype=np.float64); n = eta.size
        dA = np.asarray(ad["index", "cell_volume"], dtype=np.float64)
        x = np.asarray(ad["index", "x"], dtype=np.float64); y = np.asarray(ad["index", "y"], dtype=np.float64)
        r1, rho, p = g("rho_eta1"), g("density"), g("pressure")
        T = sg_temperature(g("T"), p, eta, C)
        ux, uy, om = g("velocityx"), g("velocityy"), g("vorticity")
        md = g("Vap_dot_rho"); gx, gy = g("grad_etax"), g("grad_etay")
        M = C["mirror"]; D = C["D"]; p1, p2 = C["p1"], C["p2"]
        S = dict(t=float(ds.current_time), plotfile=os.path.basename(pf))
        w = r1 * dA; m = w.sum()
        S["m_l"] = M * m; S["V_l"] = M * float(((1 - eta) * dA).sum())
        S["mdot"] = M * float((md * dA).sum()); S["A_grad"] = M * float((np.hypot(gx, gy) * dA).sum())
        S["m_tot"] = M * float((rho * dA).sum())
        S["xc"] = float((w * x).sum() / m) if m > 0 else float("nan")
        S["yc"] = float((w * y).sum() / m) if m > 0 else float("nan")
        S["ul"] = float((w * ux).sum() / m) if m > 0 else float("nan")
        S["KE_l"] = M * float((0.5 * w * (ux ** 2 + uy ** 2)).sum())
        liq = eta < 0.5
        wl = w[liq]
        S["xq_lo"], S["xq_hi"] = _wq(x[liq], wl, 0.002), _wq(x[liq], wl, 0.998)
        S["yq_hi"] = _wq(y[liq], wl, 0.998)
        S["x_min"] = float(x[liq].min()) if liq.any() else float("nan")
        S["x_max"] = float(x[liq].max()) if liq.any() else float("nan")
        S["y_max"] = float(y[liq].max()) if liq.any() else float("nan")
        # extent of ALL liquid incl. the diffuse band (eta < 0.99), for the image views: thin sheets and smeared
        # liquid have no eta < 0.5 cells, so the eta < 0.5 extent would crop them
        anyl = eta < 0.99; wa = ((1 - eta) * dA)[anyl]
        S["bx_lo"], S["bx_hi"] = _wq(x[anyl], wa, 1e-5), _wq(x[anyl], wa, 1 - 1e-5)
        S["by_hi"] = _wq(y[anyl], wa, 1 - 1e-5)
        # liquid temperature from the core liquid only: in the diffuse band the mixture EOS temperature is not a phase
        # temperature (it reads far above both phases), and that band holds a few % of the liquid mass
        core = eta < 0.05; wc = w[core]
        S["T_l_mean"] = float((wc * T[core]).sum() / wc.sum()) if wc.sum() > 0 else float("nan")
        S["T_l_max"] = float(T[core].max()) if core.any() else float("nan")
        S["band_mass_frac"] = float(w[(eta >= 0.05) & (eta <= 0.95)].sum() / m) if m > 0 else float("nan")
        S["T_g_max"] = float(T[~liq].max()) if (~liq).any() else float("nan")
        S["p_l_min"] = float(p[liq].min()) if liq.any() else float("nan")
        S["p_l_max"] = float(p[liq].max()) if liq.any() else float("nan")
        S["p_max"] = float(p.max())
        mw = md * dA; ms = mw.sum()
        for k, f in (("Tfilm_l", "Vap_film_T_liquid"), ("Tfilm_g", "Vap_film_T_gas"), ("BM", "Vap_film_B_M"),
                     ("BT", "Vap_film_B_T")):
            S[k] = float((mw * g(f)).sum() / ms) if ms > 0 and f in have else float("nan")
        sh = p > p1 + 0.5 * (p2 - p1) if C["shock"] else np.zeros(n, bool)
        S["x_shock"] = float(x[sh].max()) if sh.any() else float("nan")
        S["circ_pos"] = float((np.clip(om, 0, None) * dA).sum())     # computational (upper-half) plane
        S["circ_neg"] = float((np.clip(om, None, 0) * dA).sum())
        S["ncell"] = int(n); S["ncell_fine"] = int((dA <= dA.min() * 1.0001).sum())
        # tandem pair: split the liquid at the 1D two-means point of its x distribution (mass weighted)
        if C["n_drops"] == 2 and m > 0:
            sel = r1 > 1e-3 * C["rho_l"]
            xs, ws = x[sel], w[sel]; s = float((ws * xs).sum() / ws.sum())
            for _ in range(50):
                a, b = xs < s, xs >= s
                if not a.any() or not b.any(): break
                s_new = 0.5 * ((ws[a] * xs[a]).sum() / ws[a].sum() + (ws[b] * xs[b]).sum() / ws[b].sum())
                if abs(s_new - s) < 1e-4 * D: s = s_new; break
                s = s_new
            S["x_split"] = s
            for k, msk in ((1, x < s), (2, x >= s)):
                wk = w[msk]; mk = wk.sum()
                S[f"m_l_{k}"] = M * float(mk); S[f"mdot_{k}"] = M * float((md * dA)[msk].sum())
                S[f"A_{k}"] = M * float((np.hypot(gx, gy) * dA)[msk].sum())
                S[f"xc_{k}"] = float((wk * x[msk]).sum() / mk) if mk > 0 else float("nan")
                S[f"ul_{k}"] = float((wk * ux[msk]).sum() / mk) if mk > 0 else float("nan")
        os.makedirs(os.path.dirname(cfile), exist_ok=True)
        tmp = cfile + ".tmp.npz"
        np.savez_compressed(tmp, version=VERSION, scalars=json.dumps(S))
        os.replace(tmp, cfile)
        return S
    except Exception as e:
        return dict(error=f"{os.path.basename(pf)}: {e!r}\n{traceback.format_exc()}", plotfile=os.path.basename(pf))


def window(job):
    """Pass 2: uniform covering grid of the region [x0, x1] x [y_lo, y1] that every image of this plotfile shows
    (its own split view and any on/off comparison view), at the finest AMR level that fits in maxpix; plus the
    window metrics (eta = 0.5 contour length, fragments, vorticity / vaporization scales)."""
    pf, wfile, C, region, maxpix, redo = job
    key = np.round(np.asarray(region, float) / C["D"], 6)
    if not redo and os.path.isfile(wfile) and os.path.getmtime(wfile) > os.path.getmtime(os.path.join(pf, "Header")):
        try:
            z = np.load(wfile, allow_pickle=False)
            if str(z["version"]) == VERSION and int(z["maxpix"]) == maxpix and np.array_equal(z["key"], key):
                return json.loads(str(z["scalars"]))
        except Exception:
            pass
    try:
        import yt
        yt.funcs.mylog.setLevel(50)
        ds = yt.load(pf); have = {f for _, f in ds.field_list}
        M = C["mirror"]; S = {}
        lo = ds.domain_left_edge.d; hi = ds.domain_right_edge.d
        rx0, rx1, ry1 = max(region[0], lo[0]), min(region[1], hi[0]), min(region[2], hi[1])
        maxlev = ds.index.max_level; dx0 = (hi[0] - lo[0]) / ds.domain_dimensions[0]; dy0 = (hi[1] - lo[1]) / ds.domain_dimensions[1]
        L = 0
        for l in range(maxlev, -1, -1):
            if (rx1 - rx0) / (dx0 / 2 ** l) <= maxpix and (ry1 - lo[1]) / (dy0 / 2 ** l) <= maxpix: L = l; break
        dx, dy = dx0 / 2 ** L, dy0 / 2 ** L
        nxd, nyd = int(ds.domain_dimensions[0] * 2 ** L), int(ds.domain_dimensions[1] * 2 ** L)
        i0 = int(np.clip(np.floor((rx0 - lo[0]) / dx), 0, nxd - 1)); i1 = int(np.clip(np.ceil((rx1 - lo[0]) / dx), i0 + 1, nxd))
        j0 = 0; j1 = int(np.clip(np.ceil((ry1 - lo[1]) / dy), 1, nyd))
        le = ds.arr([lo[0] + i0 * dx, lo[1] + j0 * dy, ds.domain_left_edge.d[2]], "code_length")
        cg = ds.covering_grid(L, left_edge=le, dims=[i1 - i0, j1 - j0, 1], fields=[("boxlib", f) for f in WIN_FIELDS if f in have])
        Wd = {}
        for f in WIN_FIELDS:
            if f == "T" and f in have:
                e_ = np.asarray(cg[("boxlib", "eta")])[:, :, 0].T; p_ = np.asarray(cg[("boxlib", "pressure")])[:, :, 0].T
                Wd[f] = sg_temperature(np.asarray(cg[("boxlib", "T")])[:, :, 0].T, p_, e_, C).astype(np.float32); continue
            Wd[f] = (np.asarray(cg[("boxlib", f)])[:, :, 0].T.astype(np.float32) if f in have
                     else np.zeros((j1 - j0, i1 - i0), np.float32))
        ext = np.array([lo[0] + i0 * dx, lo[0] + i1 * dx, lo[1] + j0 * dy, lo[1] + j1 * dy])
        # eta = 0.5 contour length and fragments in the window (the window holds all the liquid)
        import contourpy
        from scipy import ndimage
        xv = ext[0] + (np.arange(i1 - i0) + 0.5) * dx; yv = ext[2] + (np.arange(j1 - j0) + 0.5) * dy
        lines = contourpy.contour_generator(xv, yv, Wd["eta"].astype(np.float64)).lines(0.5)
        S["A_cont"] = M * float(sum(np.hypot(*np.diff(l, axis=0).T).sum() for l in lines if len(l) > 1))
        lab, nl = ndimage.label(Wd["eta"] < 0.5, structure=np.ones((3, 3)))
        if nl:
            vol = ndimage.sum(1.0 - Wd["eta"], lab, index=np.arange(1, nl + 1)); npx = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nl + 1))
            keep = npx >= 9
            S["n_frag"] = int(keep.sum()); S["frag_largest"] = float(vol[keep].max() / vol[keep].sum()) if keep.any() else float("nan")
        else:
            S["n_frag"], S["frag_largest"] = 0, float("nan")
        S["win_level"] = int(L)
        S["vort99"] = float(np.percentile(np.abs(Wd["vorticity"]), 99.5))
        S["vapmax_win"] = float(Wd["Vap_dot_rho"].max())
        os.makedirs(os.path.dirname(wfile), exist_ok=True)
        tmp = wfile + ".tmp.npz"
        np.savez_compressed(tmp, version=VERSION, maxpix=maxpix, key=key, ext=ext, scalars=json.dumps(S), **Wd)
        os.replace(tmp, wfile)
        return S
    except Exception as e:
        return dict(error=f"{os.path.basename(pf)} (window): {e!r}\n{traceback.format_exc()}", plotfile=os.path.basename(pf))


def pool_map(fn, jobs, nproc, label=""):
    t0 = time.time(); out = []
    if nproc > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(nproc, len(jobs)), maxtasksperchild=50) as P:
            for i, r in enumerate(P.imap(fn, jobs, chunksize=1)):
                out.append(r)
                if (i + 1) % 50 == 0 or i + 1 == len(jobs): print(f"    {label} {i + 1}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
    else:
        for i, j in enumerate(jobs):
            out.append(fn(j))
            if (i + 1) % 20 == 0 or i + 1 == len(jobs): print(f"    {label} {i + 1}/{len(jobs)}  ({time.time() - t0:.0f} s)", flush=True)
    return out


# ================================================================================================================
# TIME SERIES (derived quantities)
# ================================================================================================================
def savgol(y, n=11):
    from scipy.signal import savgol_filter
    y = np.asarray(y, float); k = min(n, (len(y) // 2) * 2 - 1)
    return savgol_filter(y, k, 2) if k >= 5 and np.all(np.isfinite(y)) else y


def build_series(C, S):
    import pandas as pd
    df = pd.DataFrame(S).sort_values("t").reset_index(drop=True)
    D = C["D"]; t = df["t"].to_numpy()
    df.insert(1, "tau", tau_of(C, t))
    # step-0 plotfiles carry no diagnostics (grad eta = 0): the reference area is the first non-zero one
    nz = df.index[df["A_grad"] > 0]; i0 = int(nz[0]) if len(nz) else 0
    m0, A0 = df["m_l"].iloc[0], df["A_grad"].iloc[i0]
    C["m0"], C["A0"] = m0, A0; C["A0_circle"] = C["n_drops"] * math.pi * D
    cum = np.concatenate([[0], np.cumsum(0.5 * (df["mdot"].to_numpy()[1:] + df["mdot"].to_numpy()[:-1]) * np.diff(t))])
    df["vap_cum_frac"] = cum / m0
    df["mass_loss_frac"] = 1 - df["m_l"] / m0
    df["A_ratio"] = df["A_grad"] / A0
    df["A_cont_ratio"] = df["A_cont"] / df["A_cont"].iloc[0]
    df["mdot_norm"] = df["mdot"] * C["tscale"] / m0 if C["shock"] else df["mdot"] / m0 * 1e-6
    df["flux"] = df["mdot"] / df["A_grad"]                                  # kg/m2/s mean over the interface
    df.loc[df["A_grad"] <= 0, ["A_ratio", "flux"]] = np.nan                # step 0: no diagnostics written
    df["W_D"] = (2 if C["mirror_y"] else 1) * df["yq_hi"] / D
    df["L_D"] = (df["xq_hi"] - df["xq_lo"]) / D
    df["drift_D"] = (df["xc"] - df["xc"].iloc[0]) / D
    u2 = C["u2"]
    df["ul_u2"] = df["ul"] / u2
    if C["shock"] and len(df) > 5:
        uls = savgol(df["ul"].to_numpy()); acc = np.gradient(uls, t)
        urel = u2 - uls
        df["accel"] = acc
        df["Cd"] = df["m_l"] * acc / (0.5 * C["rho2"] * urel ** 2 * D * (C["n_drops"] if C["n_drops"] > 1 else 1))
    else:
        df["accel"] = np.nan; df["Cd"] = np.nan
    df["p_l_min_p2"] = df["p_l_min"] / C["p2"]; df["p_l_max_p2"] = df["p_l_max"] / C["p2"]
    df["p_max_p2"] = df["p_max"] / C["p2"]
    df["E_surf"] = C["sigma"] * df["A_grad"]
    df["mass_tot_drift"] = df["m_tot"] / df["m_tot"].iloc[0] - 1
    if "mdot_1" in df:
        for k in (1, 2):
            c = np.concatenate([[0], np.cumsum(0.5 * (df[f"mdot_{k}"].to_numpy()[1:] + df[f"mdot_{k}"].to_numpy()[:-1]) * np.diff(t))])
            df[f"vap_cum_frac_{k}"] = c / df[f"m_l_{k}"].iloc[0]
            df[f"A_ratio_{k}"] = df[f"A_{k}"] / df[f"A_{k}"].iloc[0]
            df[f"drift_D_{k}"] = (df[f"xc_{k}"] - df[f"xc_{k}"].iloc[0]) / D
            df[f"ul_u2_{k}"] = df[f"ul_{k}"] / u2
    return df


def at_tau(df, col, tau):
    x = df["tau"].to_numpy(); y = df[col].to_numpy()
    return float(np.interp(tau, x, y)) if len(x) and x[0] <= tau <= x[-1] else float("nan")


def summarize(C, df):
    """one row of key metrics per run (spreadsheet 'Summary')"""
    s = dict(case=C["name"], group=C["group"], fluid=C["fluid"], Ms=C["Ms"], vaporization=C["vap"], n_drops=C["n_drops"],
             D_um=C["D"] * 1e6, D_dx=C["D_dx"], tau_end=float(df["tau"].iloc[-1]), n_plotfiles=len(df))
    for tm in TAU_MARKS:
        if tm > s["tau_end"] + 1e-9: continue
        for col, lab in (("vap_cum_frac", "VapFrac"), ("A_ratio", "A/A0"), ("W_D", "W/D"), ("L_D", "L/D"),
                         ("drift_D", "Drift/D"), ("ul_u2", "u_l/u2"), ("mass_loss_frac", "MassLossFrac")):
            s[f"{lab} @tau={tm:g}"] = at_tau(df, col, tm)
    s["VapFrac_end"] = float(df["vap_cum_frac"].iloc[-1]); s["MassLossFrac_end"] = float(df["mass_loss_frac"].iloc[-1])
    s["MassLoss_minus_Vap_end"] = s["MassLossFrac_end"] - s["VapFrac_end"]
    i = int(np.nanargmax(df["mdot"].to_numpy())) if df["mdot"].max() > 0 else None
    s["mdot_peak_kg_m_s"] = float(df["mdot"].iloc[i]) if i is not None else 0.0
    s["tau_mdot_peak"] = float(df["tau"].iloc[i]) if i is not None else float("nan")
    for col, lab in (("A_ratio", "A/A0"), ("W_D", "W/D"), ("L_D", "L/D")):
        j = int(np.nanargmax(df[col].to_numpy())); s[f"{lab}_max"] = float(df[col].iloc[j]); s[f"tau_{lab}_max"] = float(df["tau"].iloc[j])
    v = df["vap_cum_frac"].to_numpy()
    s["flux_mean_kg_m2_s"] = float(np.trapezoid(df["mdot"], df["t"]) / np.trapezoid(df["A_grad"], df["t"])) if C["vap"] else 0.0
    fr = df["n_frag"].to_numpy(); ok = np.where((fr[:-1] >= 2) & (fr[1:] >= 2))[0]
    s["tau_breakup_onset"] = float(df["tau"].iloc[ok[0]]) if len(ok) else float("nan")
    s["n_frag_end"] = int(fr[-1]); s["largest_frag_end"] = float(df["frag_largest"].iloc[-1])
    early = (df["tau"] > 0.05) & (df["tau"] < 0.5)
    s["Cd_mean_tau0.05-0.5"] = float(df.loc[early, "Cd"].mean()) if early.any() else float("nan")
    s["T_l_max_K"] = float(df["T_l_max"].max()); s["T_l_mean_end_K"] = float(df["T_l_mean"].iloc[-1])
    s["Tfilm_l_mean_K"] = float(np.nanmean(df["Tfilm_l"])) if C["vap"] else float("nan")
    s["BM_mean"] = float(np.nanmean(df["BM"])) if C["vap"] else float("nan")
    s["p_l_min/p2"] = float(df["p_l_min_p2"].min()); s["p_l_max/p2"] = float(df["p_l_max_p2"].max())
    s["total_mass_drift_end"] = float(df["mass_tot_drift"].iloc[-1])
    # shock speed, fitted where the incident shock is past the liquid and inside the domain
    if C["shock"]:
        x0 = C["centers"][-1] + C["R0"] + 2 * C["D"]; x1 = C["prob_hi"][0] - C["D"]
        k = (df["x_shock"] > x0) & (df["x_shock"] < x1)
        if k.sum() >= 3:
            Us = np.polyfit(df.loc[k, "t"], df.loc[k, "x_shock"], 1)[0]
            s["Us_numerical_m_s"] = float(Us); s["Us_RH_m_s"] = C["Us"]; s["Us_error_pct"] = 100 * (Us / C["Us"] - 1)
        else:
            s["Us_RH_m_s"] = C["Us"]
    s["ncell_max"] = int(df["ncell"].max()); s["ncell_fine_max"] = int(df["ncell_fine"].max())
    s["core_hours"] = C["core_hours"]; s["status"] = C["status"]
    return s


# ================================================================================================================
# FRAME DRAWING (workers read their window from the cache)
# ================================================================================================================
def load_win(cfile):
    z = np.load(cfile, allow_pickle=False)
    W = {f: z[f].astype(np.float64) for f in WIN_FIELDS}; W["ext"] = z["ext"]
    return W


def schlieren(W):
    ext = W["ext"]; ny, nx = W["density"].shape
    dx = (ext[1] - ext[0]) / nx; dy = (ext[3] - ext[2]) / ny
    # gradient of ln(rho): shocks (density ratio ~4) and the liquid/gas interface (ratio ~100) both stay visible
    gy, gx = np.gradient(np.log(np.maximum(W["density"], 1e-8)), dy, dx); gm = np.hypot(gx, gy)
    gref = np.percentile(gm, 99.0)
    return np.exp(-SCHLIEREN_K * gm / max(gref, 1e-30))


def field(W, kind, C, lim):
    """returns array, cmap, norm, colorbar label"""
    if kind == "schlieren": return schlieren(W), "gray", Normalize(0, 1), "Numerical Schlieren"
    if kind == "pressure": return W["pressure"] / C["p2"], "viridis", Normalize(0, lim["p"]), r"$p/p_2$"
    if kind == "T": return W["T"], "inferno", Normalize(lim["T"][0], lim["T"][1]), r"$T$ [K]"
    if kind == "velocity":
        return np.hypot(W["velocityx"], W["velocityy"]) / lim["u"], "plasma", Normalize(0, 1.5), r"$|\mathbf{u}|/u_2$"
    if kind == "vorticity":
        return W["vorticity"] * C["D"] / lim["u"], "RdBu_r", Normalize(-lim["w"], lim["w"]), r"$\omega D/u_2$"
    if kind == "mach":
        return np.hypot(W["velocityx"], W["velocityy"]) / np.maximum(W["a"], 1e-12), "cividis", Normalize(0, lim["M"]), "Local Mach Number"
    if kind == "density": return np.log10(np.maximum(W["density"], 1e-6)), "viridis", Normalize(lim["rho"][0], lim["rho"][1]), r"$\log_{10}\rho$ [kg/m$^3$]"
    if kind == "vaprate":
        v = W["Vap_dot_rho"]; vm = lim["vap"]
        return np.where(v > vm * 1e-5, v, np.nan), "magma", LogNorm(vm * 1e-4, vm), r"$\dot m'''$ [kg/m$^3$/s]"
    if kind == "massfrac": return W["Mass_Fraction"], "Blues_r", Normalize(0, 1), r"Gas Mass Fraction $Y_g$"
    raise KeyError(kind)


def _eta_lines(ax, x, y, eta, color):
    for lv, lw, ls in ((0.01, 0.5, "--"), (0.5, 1.1, "-"), (0.99, 0.5, "--")):
        try: ax.contour(x, y, eta, [lv], colors=color, linewidths=lw, linestyles=ls)
        except Exception: pass


def _half(ax, W, kind, C, lim, bottom, ul, ec):
    """draw one half; bottom = mirror image (y -> -y)"""
    A, cmap, norm, lab = field(W, kind, C, lim)
    ext = W["ext"]; ny, nx = A.shape
    x = (ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx) / C["D"]
    y = (ext[2] + (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny) / C["D"]
    eta = W["eta"]
    if bottom: A, eta, y, e = A[::-1], eta[::-1], -y[::-1], (ext[0] / C["D"], ext[1] / C["D"], -ext[3] / C["D"], -ext[2] / C["D"])
    else: e = (ext[0] / C["D"], ext[1] / C["D"], ext[2] / C["D"], ext[3] / C["D"])
    im = ax.imshow(A, origin="lower", extent=e, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    if kind == "velocity":   # streamlines in the frame moving with the liquid
        st = max(1, int(max(nx, ny) / 250))
        vx = (W["velocityx"] - ul)[::st, ::st]; vy = W["velocityy"][::st, ::st]
        xs, ys = x[::st], (y if not bottom else -y[::-1])[::st]
        if bottom: vx, vy, ys = vx[::-1], -vy[::-1], -ys[::-1]
        try: ax.streamplot(xs, ys, vx, vy, color="w", density=1.4, linewidth=0.6, arrowsize=0.6)
        except Exception: pass
    _eta_lines(ax, x, y, eta, ec)
    return im, lab


def fit_view(C, xc, h, v):
    """keep a view inside the domain: shift it back in at an x boundary (same size, so no zoom jump), cap its height"""
    lo, hi = C["prob_lo"], C["prob_hi"]
    x0, x1 = xc - h, xc + h
    if x1 > hi[0]: x0, x1 = x0 - (x1 - hi[0]), hi[0]
    if x0 < lo[0]: x0, x1 = lo[0], min(x1 + (lo[0] - x0), hi[0])
    return 0.5 * (x0 + x1), 0.5 * (x1 - x0), min(v, hi[1] - lo[1])


def _view(C, row, half=None, hy=None):
    xc = row["view_xc"] / C["D"]; h = (half if half is not None else row["view_half"]) / C["D"]
    v = (hy if hy is not None else row["view_hy"]) / C["D"]
    return xc - h, xc + h, v


def draw_split(job):
    kind, C, lim, row, cfile, png = job
    try:
        W = load_win(cfile); top, bot = SPLITS[kind]
        x0, x1, v = _view(C, row)
        figw = 10.0; figh = max(4.0, min(9.0, figw * (2 * v) / (x1 - x0) * 0.9 + 1.2))
        fig, ax = plt.subplots(figsize=(figw, figh)); ax.set_facecolor("#d9d9d9")
        imt, lt = _half(ax, W, top, C, lim, False, row["ul"], "r" if top == "schlieren" else "w")
        imb, lb = _half(ax, W, bot, C, lim, True, row["ul"], "w" if bot not in ("massfrac",) else "k")
        ax.axhline(0, color="w", lw=1.0, ls=":")
        ax.set_xlim(x0, x1); ax.set_ylim(-v, v); ax.grid(False)
        ax.set_xlabel(r"$x/D$"); ax.set_ylabel(r"$y/D$")
        ax.set_title(f"{C['label']}\n{tfmt(C, row['t'])}", fontsize=14)
        fig.subplots_adjust(right=0.84)
        c1 = ax.inset_axes([1.03, 0.53, 0.025, 0.44]); fig.colorbar(imt, cax=c1).set_label(lt, fontsize=13)
        c2 = ax.inset_axes([1.03, 0.03, 0.025, 0.44]); fig.colorbar(imb, cax=c2).set_label(lb, fontsize=13)
        fig.savefig(png, dpi=FRAME_DPI, bbox_inches=None); plt.close(fig)
        return None
    except Exception as e:
        plt.close("all"); return f"{png}: {e!r}"


def draw_cmp(job):
    """vaporization ON (top) vs OFF (bottom), same field, matched in tau, centred between the two centroids"""
    kind, Cv, Cn, lim, rv, rn, fv, fn, png = job
    try:
        Wv, Wn = load_win(fv), load_win(fn)
        D = Cv["D"]; xc, h, v = (q / D for q in rv["cmp_view"])
        figw = 10.0; figh = max(4.0, min(9.0, figw * v / h * 0.9 + 1.2))
        fig, ax = plt.subplots(figsize=(figw, figh)); ax.set_facecolor("#d9d9d9")
        im, lab = _half(ax, Wv, kind, Cv, lim, False, rv["ul"], "r" if kind == "schlieren" else "w")
        _half(ax, Wn, kind, Cn, lim, True, rn["ul"], "r" if kind == "schlieren" else "w")
        ax.axhline(0, color="w", lw=1.2, ls="--")
        tc = "k" if kind == "schlieren" else "w"
        ax.text(0.01, 0.97, "Vaporization On", transform=ax.transAxes, va="top", color=tc, fontsize=13, fontweight="bold")
        ax.text(0.01, 0.03, "Vaporization Off", transform=ax.transAxes, va="bottom", color=tc, fontsize=13, fontweight="bold")
        ax.set_xlim(xc - h, xc + h); ax.set_ylim(-v, v); ax.grid(False)
        ax.set_xlabel(r"$x/D$"); ax.set_ylabel(r"$y/D$")
        ax.set_title(f"{Cv['label_base']}\n{tfmt(Cv, rv['t'])}", fontsize=14)
        fig.subplots_adjust(right=0.86)
        cax = ax.inset_axes([1.03, 0.1, 0.025, 0.8]); fig.colorbar(im, cax=cax).set_label(lab, fontsize=13)
        fig.savefig(png, dpi=FRAME_DPI, bbox_inches=None); plt.close(fig)
        return None
    except Exception as e:
        plt.close("all"); return f"{png}: {e!r}"


def make_gif(job):
    folder, gif, fps = job
    fr = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not fr: return f"{gif}: no frames"
    if os.path.isfile(gif) and os.path.getmtime(gif) > max(os.path.getmtime(f) for f in fr): return None
    ims = [Image.open(f).convert("RGB") for f in fr]
    ims = [im.resize(ims[0].size) if im.size != ims[0].size else im for im in ims]
    ims = [im.quantize(colors=256, method=Image.Quantize.MEDIANCUT) for im in ims]
    ims[0].save(gif, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return None


# ================================================================================================================
# STATIC PLOTS
# ================================================================================================================
def save(fig, path):
    fig.savefig(path + ".png", dpi=PLOT_DPI); fig.savefig(path + ".pdf"); plt.close(fig)


def lab_ax(ax, C, ylabel, title=None):
    ax.set_xlabel(C["tlabel"]); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)


def plot_area_vs_vap(C, df, d):
    x = df["tau"]
    fig, a1 = plt.subplots(figsize=(8.5, 5.2))
    a1.plot(x, df["A_ratio"], color=PAL[0], lw=2, label=r"Interface Area $A/A_0$")
    lab_ax(a1, C, r"Interface Area $A/A_0$"); a1.tick_params(axis="y", colors=PAL[0]); a1.yaxis.label.set_color(PAL[0])
    a2 = a1.twinx(); a2.grid(False)
    a2.plot(x, df["mdot_norm"], color=PAL[1], lw=2, ls="--", label="Vaporization Rate")
    rl = r"Vaporization Rate $\dot m\, t^*/m_0$" if C["shock"] else r"Vaporization Rate $\dot m/m_0$ [1/$\mu$s]"
    a2.set_ylabel(rl, color=PAL[1]); a2.tick_params(axis="y", colors=PAL[1])
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
    a1.set_title(f"Interface Area and Vaporization Rate\n{C['label']}")
    save(fig, os.path.join(d, "01_Area_vs_VapRate"))
    # same data on one axis, each indexed to its own peak (reads without a second scale)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, df["A_grad"] / df["A_grad"].max(), color=PAL[0], lw=2, label="Interface Area")
    if df["mdot"].max() > 0: ax.plot(x, df["mdot"] / df["mdot"].max(), color=PAL[1], lw=2, ls="--", label="Vaporization Rate")
    if df["mdot"].max() > 0: ax.plot(x, df["flux"] / np.nanmax(df["flux"]), color=PAL[2], lw=1.6, ls="-.", label=r"Mean Flux $\dot m/A$")
    lab_ax(ax, C, "Value / Peak Value", f"Area, Rate and Flux Indexed to Their Peaks\n{C['label']}")
    ax.legend(frameon=False); ax.set_ylim(0, 1.05)
    save(fig, os.path.join(d, "01b_Area_VapRate_Flux_Indexed"))


def plot_mass(C, df, d):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(df["tau"], df["vap_cum_frac"], color=PAL[1], lw=2, label=r"Vaporized (Model) $\int\dot m\,dt/m_0$")
    ax.plot(df["tau"], df["mass_loss_frac"], color=PAL[0], lw=2, ls="--", label=r"Liquid Mass Loss $1 - m_l/m_0$")
    ax.plot(df["tau"], df["mass_tot_drift"], color=PAL[2], lw=1.5, ls=":", label=r"Total Mass Change $m/m_{0,\mathrm{tot}} - 1$")
    lab_ax(ax, C, "Fraction", f"Mass Budget\n{C['label']}"); ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    save(fig, os.path.join(d, "02_Mass_Budget"))


def _contours(cfile, C, mirror=True, shift=0.0):
    W = load_win(cfile); ext = W["ext"]; ny, nx = W["eta"].shape
    import contourpy
    x = (ext[0] + (np.arange(nx) + 0.5) * (ext[1] - ext[0]) / nx - shift) / C["D"]
    y = (ext[2] + (np.arange(ny) + 0.5) * (ext[3] - ext[2]) / ny) / C["D"]
    L = contourpy.contour_generator(x, y, W["eta"]).lines(0.5)
    if mirror and C["mirror_y"]: L = L + [l * np.array([1, -1]) for l in L]
    return L


def _extent(groups, pad=0.25):
    """half-width / half-height (in D) that holds every contour, about x = 0 (drop frame)"""
    h, v = 0.75, 0.75
    for L in groups:
        for l in L:
            if len(l): h = max(h, np.abs(l[:, 0]).max()); v = max(v, np.abs(l[:, 1]).max())
    return h + pad, v + pad


def pick_rows(df, n):
    taus = np.linspace(df["tau"].iloc[0], df["tau"].iloc[-1], n)
    idx = sorted({int(np.argmin(np.abs(df["tau"].to_numpy() - t))) for t in taus})
    return idx


def plot_contours(C, df, d, cache):
    idx = pick_rows(df, 10); cm = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    tmin, tmax = df["tau"].iloc[idx[0]], df["tau"].iloc[idx[-1]]
    xs = []
    for i in idx:
        col = cm((df["tau"].iloc[i] - tmin) / max(tmax - tmin, 1e-12))
        for l in _contours(cache[df["plotfile"].iloc[i]], C):
            ax.plot(l[:, 0], l[:, 1], color=col, lw=1.2); xs.append(l[:, 0])
    th = np.linspace(0, 2 * np.pi, 200)
    for c in C["centers"]: ax.plot(c / C["D"] + 0.5 * np.cos(th), 0.5 * np.sin(th), "k:", lw=1)
    ax.set_aspect("equal"); ax.set_xlabel(r"$x/D$"); ax.set_ylabel(r"$y/D$")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=Normalize(tmin, tmax)); fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02).set_label(C["tlabel"])
    ax.set_title(f"Interface ($\\eta = 0.5$) at Several Times, Lab Frame\n{C['label']}")
    save(fig, os.path.join(d, "03_Eta05_Contours_LabFrame"))
    # panels in the drop frame
    idx = pick_rows(df, 8); n = len(idx); nc = 4; nr = int(math.ceil(n / nc))
    LL = {i: _contours(cache[df["plotfile"].iloc[i]], C, shift=df["xc"].iloc[i]) for i in idx}
    hmax, vmax = _extent(LL.values())
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 4.0 * nr * min(1.0, vmax / hmax) + 0.8), squeeze=False)
    for k, i in enumerate(idx):
        a = axs.flat[k]
        for l in LL[i]:
            a.plot(l[:, 0], l[:, 1], color=PAL[0], lw=1.2)
        a.set_xlim(-hmax, hmax); a.set_ylim(-vmax, vmax); a.set_aspect("equal")
        a.set_title(f"{C['tlabel']} = {df['tau'].iloc[i]:.2f}" if C["shock"] else f"$t$ = {df['t'].iloc[i] * 1e6:.1f} $\\mu$s", fontsize=13)
        if k % nc == 0: a.set_ylabel(r"$y/D$")
        if k >= n - nc: a.set_xlabel(r"$(x - x_c)/D$")
    for a in axs.flat[n:]: a.set_visible(False)
    fig.suptitle(f"Interface ($\\eta = 0.5$) in the Frame Moving With the Liquid\n{C['label']}", fontsize=15)
    fig.tight_layout(); save(fig, os.path.join(d, "04_Eta05_Contours_DropFrame"))


def multi(C, df, d, fname, title, panels, ncol=2):
    nr = int(math.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nr, ncol, figsize=(6.2 * ncol, 3.9 * nr), squeeze=False, sharex=True)
    for a, (cols, ylab) in zip(axs.flat, panels):
        for k, (c, lbl) in enumerate(cols):
            if c in df and np.isfinite(df[c]).any(): a.plot(df["tau"], df[c], color=PAL[k], lw=1.8, label=lbl, ls=["-", "--", "-.", ":"][k % 4])
        a.set_ylabel(ylab)
        if len(cols) > 1: a.legend(frameon=False, fontsize=13)
    for a in axs[-1]: a.set_xlabel(C["tlabel"])
    for a in axs.flat[len(panels):]: a.set_visible(False)
    fig.suptitle(f"{title}\n{C['label']}", fontsize=15); fig.tight_layout(); save(fig, os.path.join(d, fname))


def plot_case_static(C, df, d, cache):
    plot_area_vs_vap(C, df, d); plot_mass(C, df, d); plot_contours(C, df, d, cache)
    multi(C, df, d, "05_Deformation", "Deformation", [
        ([("W_D", "")], r"Cross-Stream Width $W/D$"), ([("L_D", "")], r"Streamwise Length $L/D$"),
        ([("A_ratio", r"$\int|\nabla\eta|\,dA$"), ("A_cont_ratio", r"$\eta = 0.5$ Contour")], r"Interface Area $A/A_0$"),
        ([("V_l", "")], r"Liquid Volume [m$^2$]")])
    if C["shock"]:
        multi(C, df, d, "06_Kinematics_Drag", "Liquid Kinematics", [
            ([("drift_D", "")], r"Centroid Drift $(x_c - x_{c,0})/D$"), ([("ul_u2", "")], r"Liquid Velocity $\bar u_l/u_2$"),
            ([("Cd", "")], r"Drag Coefficient $C_D$"), ([("KE_l", "")], r"Liquid Kinetic Energy [J/m]")])
    if C["vap"]:
        multi(C, df, d, "07_Vaporization_Diagnostics", "Vaporization Diagnostics", [
            ([("mdot", "")], r"Vaporization Rate $\dot m$ [kg/(m s)]"), ([("flux", "")], r"Mean Flux $\dot m/A$ [kg/(m$^2$ s)]"),
            ([("Tfilm_l", "Liquid Surface"), ("Tfilm_g", "Gas Film")], r"Rate-Weighted Film $T$ [K]"),
            ([("BM", r"$B_M$"), ("BT", r"$B_T$")], "Rate-Weighted Transfer Number")])
    multi(C, df, d, "08_Fragmentation", "Fragmentation (Window, $\\eta < 0.5$ Regions)", [
        ([("n_frag", "")], "Number of Liquid Fragments"), ([("frag_largest", "")], "Largest Fragment Volume Fraction")])
    multi(C, df, d, "09_Liquid_Extremes", "Liquid Pressure and Temperature", [
        ([("p_l_min_p2", "Minimum"), ("p_l_max_p2", "Maximum")], r"Liquid Pressure $p/p_2$"),
        ([("T_l_mean", "Mass Mean"), ("T_l_max", "Maximum")], r"Core Liquid ($\eta < 0.05$) $T$ [K]"),
        ([("p_max_p2", "")], r"Peak Pressure $p_{max}/p_2$"), ([("T_g_max", "")], r"Peak Gas $T$ [K]")])
    multi(C, df, d, "10_Energy_Circulation", "Energy and Circulation", [
        ([("E_surf", "")], r"Surface Energy $\sigma A$ [J/m]"), ([("KE_l", "")], r"Liquid Kinetic Energy [J/m]"),
        ([("circ_pos", r"$\Gamma^+$"), ("circ_neg", r"$\Gamma^-$")], r"Circulation (Upper Half) [m$^2$/s]"),
        ([("mass_tot_drift", "")], "Total Mass Change (Outflow Included)"),
        ([("band_mass_frac", "")], "Liquid Mass in the Diffuse Band")])
    multi(C, df, d, "11_AMR_Cells", "AMR Cell Count", [([("ncell", "All Leaf Cells"), ("ncell_fine", "Finest Level")], "Cells")], ncol=1)
    if C["shock"]:
        fig, ax = plt.subplots(figsize=(8.5, 5.2)); k = np.isfinite(df["x_shock"])
        ax.plot(df.loc[k, "t"] * 1e6, df.loc[k, "x_shock"] / C["D"], "o", ms=4, color=PAL[0], label="Numerical")
        tt = np.linspace(0, df["t"].iloc[-1], 50)
        ax.plot(tt * 1e6, (C["x_shock0"] + C["Us"] * tt) / C["D"], "-", color=PAL[1], lw=1.5, label=f"Rankine-Hugoniot, $U_s$ = {C['Us']:.0f} m/s")
        ax.axhline(C["prob_hi"][0] / C["D"], color="0.5", lw=1, ls=":")
        ax.set_xlabel(r"$t$ [$\mu$s]"); ax.set_ylabel(r"Incident Shock $x/D$"); ax.legend(frameon=False)
        ax.set_title(f"Incident Shock Trajectory\n{C['label']}"); save(fig, os.path.join(d, "12_Shock_xt"))
    if C["n_drops"] == 2 and "mdot_1" in df:
        multi(C, df, d, "13_TandemPair_PerDrop", "Tandem Pair, Per-Column Metrics (Two-Means Split in $x$)", [
            ([("vap_cum_frac_1", "Upstream"), ("vap_cum_frac_2", "Downstream")], "Vaporized Fraction"),
            ([("A_ratio_1", "Upstream"), ("A_ratio_2", "Downstream")], r"Interface Area $A/A_0$"),
            ([("drift_D_1", "Upstream"), ("drift_D_2", "Downstream")], r"Centroid Drift $/D$"),
            ([("ul_u2_1", "Upstream"), ("ul_u2_2", "Downstream")], r"Liquid Velocity $/u_2$")])


def plot_vap_novap(Cv, Cn, dv, dn, d, cv, cn):
    def ov(fname, title, panels):
        nr = int(math.ceil(len(panels) / 2))
        fig, axs = plt.subplots(nr, 2, figsize=(12.4, 3.9 * nr), squeeze=False, sharex=True)
        for a, (c, ylab) in zip(axs.flat, panels):
            a.plot(dn["tau"], dn[c], color=C_NOVAP, lw=1.8, label="Vaporization Off")
            a.plot(dv["tau"], dv[c], color=C_VAP, lw=1.8, ls="--", label="Vaporization On")
            a.set_ylabel(ylab)
        axs.flat[0].legend(frameon=False)
        for a in axs[-1]: a.set_xlabel(Cv["tlabel"])
        for a in axs.flat[len(panels):]: a.set_visible(False)
        fig.suptitle(f"{title}\n{Cv['label_base']}", fontsize=15); fig.tight_layout(); save(fig, os.path.join(d, fname))
    cols = [("A_ratio", r"Interface Area $A/A_0$"), ("W_D", r"Width $W/D$"), ("L_D", r"Length $L/D$"),
            ("drift_D", r"Centroid Drift $/D$"), ("ul_u2", r"Liquid Velocity $\bar u_l/u_2$"),
            ("mass_loss_frac", r"Liquid Mass Loss $1 - m_l/m_0$"), ("T_l_mean", r"Core Liquid Mean $T$ [K]"), ("n_frag", "Fragments")]
    ov("CMP_TimeSeries", "Vaporization On vs Off", [c for c in cols if c[0] in dv and c[0] in dn])
    # relative differences
    fig, ax = plt.subplots(figsize=(8.5, 5.2)); tg = dn["tau"].to_numpy()
    for k, (c, lbl) in enumerate((("A_ratio", "Interface Area"), ("W_D", "Width"), ("drift_D", "Centroid Drift"), ("m_l", "Liquid Mass"))):
        with np.errstate(divide="ignore", invalid="ignore"): y = 100 * (np.interp(tg, dv["tau"], dv[c]) / dn[c].to_numpy() - 1)
        ok = (tg >= dv["tau"].iloc[0]) & (tg <= dv["tau"].iloc[-1]) & (np.abs(dn[c].to_numpy()) > 1e-3 * np.nanmax(np.abs(dn[c])))
        ax.plot(tg[ok], y[ok], color=PAL[k], lw=1.8, ls=["-", "--", "-.", ":"][k], label=lbl)
    ax.axhline(0, color="0.4", lw=0.8); lab_ax(ax, Cv, "On / Off - 1 [%]", f"Effect of Vaporization\n{Cv['label_base']}"); ax.legend(frameon=False)
    save(fig, os.path.join(d, "CMP_RelativeDifference"))
    # contours, both runs, matched tau, drop frame of the no-vap run
    idx = pick_rows(dn, 8); n = len(idx); nc = 4; nr = int(math.ceil(n / nc))
    LL = {}
    for i in idx:
        tau = dn["tau"].iloc[i]; j = int(np.argmin(np.abs(dv["tau"].to_numpy() - tau))); sh = dn["xc"].iloc[i]
        LL[i] = (tau, _contours(cn[dn["plotfile"].iloc[i]], Cn, shift=sh), _contours(cv[dv["plotfile"].iloc[j]], Cv, shift=sh))
    hmax, vmax = _extent([q for v_ in LL.values() for q in v_[1:]])
    fig, axs = plt.subplots(nr, nc, figsize=(4.0 * nc, 4.0 * nr * min(1.0, vmax / hmax) + 0.8), squeeze=False)
    for k, i in enumerate(idx):
        a = axs.flat[k]; tau, Ln, Lv = LL[i]
        for l in Ln: a.plot(l[:, 0], l[:, 1], color=C_NOVAP, lw=1.3)
        for l in Lv: a.plot(l[:, 0], l[:, 1], color=C_VAP, lw=1.1, ls="--")
        a.set_xlim(-hmax, hmax); a.set_ylim(-vmax, vmax); a.set_aspect("equal"); a.set_title(f"{Cv['tlabel']} = {tau:.2f}", fontsize=13)
        if k % nc == 0: a.set_ylabel(r"$y/D$")
        if k >= n - nc: a.set_xlabel(r"$(x - x_{c,\mathrm{off}})/D$")
    for a in axs.flat[n:]: a.set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=C_NOVAP), Line2D([], [], color=C_VAP, ls="--")], ["Vaporization Off", "Vaporization On"],
               loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"Interface ($\\eta = 0.5$), Vaporization On vs Off\n{Cv['label_base']}", fontsize=15)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, os.path.join(d, "CMP_Eta05_Contours"))


def plot_sweep(cases, d, title):
    cases = sorted(cases, key=lambda cd: cd[0]["Ms"])[:len(PAL)]
    panels = [("vap_cum_frac", "Vaporized Fraction"), ("mdot_norm", r"Vaporization Rate $\dot m\,t^*/m_0$"),
              ("A_ratio", r"Interface Area $A/A_0$"), ("flux", r"Mean Flux $\dot m/A$ [kg/(m$^2$ s)]"),
              ("W_D", r"Width $W/D$"), ("drift_D", r"Centroid Drift $/D$"), ("ul_u2", r"Liquid Velocity $\bar u_l/u_2$"),
              ("T_l_mean", r"Core Liquid Mean $T$ [K]")]
    if not any(C["vap"] for C, _ in cases): panels = [p for p in panels if p[0] not in ("vap_cum_frac", "mdot_norm", "flux")]
    nr = int(math.ceil(len(panels) / 2))
    fig, axs = plt.subplots(nr, 2, figsize=(12.4, 3.8 * nr), squeeze=False, sharex=True)
    for a, (c, ylab) in zip(axs.flat, panels):
        for k, (C, df) in enumerate(cases):
            a.plot(df["tau"], df[c], color=PAL[k], lw=1.8, ls=["-", "--", "-.", ":", "-", "--"][k], label=f"$M_s$ = {C['Ms']:.3g}")
        a.set_ylabel(ylab)
    axs.flat[0].legend(frameon=False)
    for a in axs[-1]: a.set_xlabel(r"$t/t^*$")
    for a in axs.flat[len(panels):]: a.set_visible(False)
    fig.suptitle(title, fontsize=15); fig.tight_layout(); save(fig, os.path.join(d, "SWEEP_TimeSeries"))
    # end-of-run / fixed-tau metrics vs Mach
    tcom = min(df["tau"].iloc[-1] for _, df in cases)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    Ms = [C["Ms"] for C, _ in cases]
    for a, (c, ylab) in zip(axs, (("vap_cum_frac", f"Vaporized Fraction at $t/t^*$ = {tcom:.2g}"),
                                   ("A_ratio", f"$A/A_0$ at $t/t^*$ = {tcom:.2g}"), ("W_D", f"$W/D$ at $t/t^*$ = {tcom:.2g}"))):
        a.plot(Ms, [at_tau(df, c, tcom) for _, df in cases], "o-", color=PAL[0], ms=8, label="Numerical")
        a.set_xlabel(r"Shock Mach Number $M_s$"); a.set_ylabel(ylab)
    fig.suptitle(title, fontsize=15); fig.tight_layout(); save(fig, os.path.join(d, "SWEEP_vs_Mach"))


# ================================================================================================================
# MAIN
# ================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--only", default=None); ap.add_argument("--skip", default=None)
    ap.add_argument("--maxpix", type=int, default=2000); ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--no-frames", action="store_true"); ap.add_argument("--no-gifs", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--no-tar", action="store_true"); ap.add_argument("--tar-frames", action="store_true")
    a = ap.parse_args()
    root = a.root or next((r for r in ("/mmfs1/home/ttryon/flames/bin/tests/AIAA2027",
                                       os.path.join(os.path.dirname(os.path.dirname(AIAA_SRC)), "bin", "tests", "AIAA2027"))
                           if os.path.isdir(r)), ".")
    out = os.path.abspath(a.out or os.path.join(root, "Analysis")); os.makedirs(out, exist_ok=True)
    T0 = time.time()
    log("=" * 100); log(f"AIAA2027 analysis  {datetime.datetime.now().isoformat(timespec='seconds')}")
    try: git = subprocess.check_output(["git", "-C", HERE, "describe", "--always", "--dirty"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: git = "unavailable"
    log(f"root {root}\nout  {out}\nscript git {git}, {a.nproc} worker processes, maxpix {a.maxpix}, every {a.every}")

    runs = discover(root, out)
    if a.only: runs = [r for r in runs if re.search(a.only, r["name"])]
    if a.skip: runs = [r for r in runs if not re.search(a.skip, r["name"])]
    log(f"found {len(runs)} run(s)")
    cases = []
    for R in runs:
        C = initial_conditions(R)
        if not C["meta_found"]: log(f"  SKIP {R['name']}: no metadata in {R['run']}"); continue
        if not np.isfinite(C["D"]): log(f"  SKIP {R['name']}: no droplet radius R0 in metadata"); continue
        pfs = plotfiles(R["run"]); C["plotfiles"] = pfs
        C["label"] = (f"{C['fluid']}, $D$ = {C['D'] * 1e6:.3g} $\\mu$m" + (f", $M_s$ = {C['Ms']:.3g}" if C["shock"] else "")
                      + (", Tandem Pair" if C["n_drops"] > 1 else "") + (", Vaporization On" if C["vap"] else ", Vaporization Off"))
        C["label_base"] = C["label"].replace(", Vaporization On", "").replace(", Vaporization Off", "")
        log(f"  {C['group']}/{C['name']}: {len(pfs)} plotfiles, {C['label_base']}, vap {C['vap']}, D/dx {C['D_dx']:.0f}, "
            f"t* {C['tstar'] * 1e6 if C['shock'] else float('nan'):.4g} us, end t/t* {C['t_end_tau']:.3g}, status '{C['status']}'")
        cases.append(C)
    if not cases: log("nothing to do"); return

    # ---------------- single read pass over every plotfile of every run ----------------
    log("\nREAD PASS 1 (scalars over every leaf cell; cached in _cache/)")
    jobs, owner = [], []
    for ci, C in enumerate(cases):
        C["cachedir"] = os.path.join(out, "_cache", C["group"], C["name"])
        for pf in C["plotfiles"]:
            jobs.append((pf, os.path.join(C["cachedir"], os.path.basename(pf) + ".s.npz"),
                         {k: C[k] for k in ("mirror", "mirror_y", "D", "p1", "p2", "shock", "centers", "n_drops", "rho_l", "R0",
                                         "gamma_g", "gamma_l", "pinf_g", "pinf_l")},
                         a.redo))
            owner.append(ci)
    res = pool_map(extract, jobs, a.nproc, "plotfiles")
    per = [[] for _ in cases]
    for ci, r, j in zip(owner, res, jobs):
        if "error" in r: log(f"  [error] {cases[ci]['name']} {r['error'].splitlines()[0]}"); continue
        per[ci].append(r)
    log(f"  read pass 1 done in {time.time() - T0:.0f} s")

    # ---------------- views: what each image shows ----------------
    # Each split frame is centred on the liquid's extent (eta < 0.99, incl. thin sheets and smeared liquid) with a
    # VIEW_PAD margin; its size is the running maximum, so the images follow the liquid and only zoom out.
    # On/off comparison frames hold both runs' liquid (union), again only zooming out.  Pass 2 then reads, for each
    # plotfile, exactly the union of the regions its images show, so nothing inside the domain is cropped.
    for ci, C in enumerate(cases):
        per[ci].sort(key=lambda r: r["t"]); D = C["D"]; hr = vr = 0.0; c = C["centers"][0]
        for r in per[ci]:
            if np.isfinite(r.get("bx_lo", np.nan)):
                c = 0.5 * (r["bx_lo"] + r["bx_hi"]); need, vn = 0.5 * (r["bx_hi"] - r["bx_lo"]) + VIEW_PAD * D, r["by_hi"] + VIEW_PAD * D
            else:
                c = r["xc"] if np.isfinite(r["xc"]) else c; need = vn = 0.0
            hr = max(hr, VIEW_MIN[0] * D, need); vr = max(vr, VIEW_MIN[1] * D, vn)
            r["view_xc"], r["view_half"], r["view_hy"] = fit_view(C, c, hr, vr)
    ok = [ci for ci in range(len(cases)) if len(per[ci]) >= 2]
    ipairs = [(i, j) for i in ok if cases[i]["vap"] for j in ok
              if not cases[j]["vap"] and cases[j]["vapbase"] == cases[i]["vapbase"] and cases[j]["group"] == cases[i]["group"]]
    cmpviews = {}
    for i, j in ipairs:
        Rv, Rn = per[i], per[j]; D = cases[i]["D"]; tn = tau_of(cases[j], [r["t"] for r in Rn]); hr = vr = 0.0; L = []
        for iv in list(range(0, len(Rv), a.every)):
            rv = Rv[iv]; jn = int(np.argmin(np.abs(tn - tau_of(cases[i], rv["t"])))); rn = Rn[jn]
            x0 = min(rv["view_xc"] - rv["view_half"], rn["view_xc"] - rn["view_half"])
            x1 = max(rv["view_xc"] + rv["view_half"], rn["view_xc"] + rn["view_half"])
            hr = max(hr, 0.5 * (x1 - x0)); vr = max(vr, rv["view_hy"], rn["view_hy"])
            L.append((iv, jn) + fit_view(cases[i], 0.5 * (x0 + x1), hr, vr))
        cmpviews[(i, j)] = L
    # region per plotfile = union of its split view and every comparison view it appears in
    reg = [[[r["view_xc"] - r["view_half"], r["view_xc"] + r["view_half"], r["view_hy"]] for r in per[ci]] for ci in range(len(cases))]
    for (i, j), L in list(cmpviews.items()):
        for iv, jn, xc, h, v in L:
            for ci, k in ((i, iv), (j, jn)):
                g = reg[ci][k]; g[0] = min(g[0], xc - h); g[1] = max(g[1], xc + h); g[2] = max(g[2], v)
        cmpviews[(cases[i]["name"], cases[j]["name"])] = [(per[i][iv]["plotfile"], per[j][jn]["plotfile"], xc, h, v)
                                                          for iv, jn, xc, h, v in L]
        del cmpviews[(i, j)]

    T0b = time.time(); log("\nREAD PASS 2 (uniform windows that hold every image; cached in _cache/)")
    jobs2, own2 = [], []
    for ci in ok:
        C = cases[ci]; Cs = {k: C[k] for k in ("mirror", "D", "gamma_g", "gamma_l", "pinf_g", "pinf_l")}
        pfd = {os.path.basename(pf): pf for pf in C["plotfiles"]}
        for k, r in enumerate(per[ci]):
            jobs2.append((pfd[r["plotfile"]], os.path.join(C["cachedir"], r["plotfile"] + ".w.npz"), Cs, reg[ci][k], a.maxpix, a.redo))
            own2.append((ci, k))
    bad = set()
    for (ci, k), r2 in zip(own2, pool_map(window, jobs2, a.nproc, "windows")):
        if "error" in r2: log(f"  [error] {cases[ci]['name']} {r2['error'].splitlines()[0]}"); bad.add((ci, k)); continue
        per[ci][k].update(r2)
    for ci in ok: per[ci] = [r for k, r in enumerate(per[ci]) if (ci, k) not in bad]
    log(f"  read pass 2 done in {time.time() - T0b:.0f} s")

    import pandas as pd
    series, summ = {}, []
    for ci, C in enumerate(cases):
        if len(per[ci]) < 2: log(f"  SKIP {C['name']}: fewer than 2 readable plotfiles"); continue
        df = build_series(C, per[ci])
        C["cache"] = {r["plotfile"]: os.path.join(C["cachedir"], r["plotfile"] + ".w.npz") for r in per[ci]}
        series[C["name"]] = (C, df); summ.append(summarize(C, df))

    # ---------------- per-run notes and warnings ----------------
    log("\nPER-RUN NOTES")
    for name, (C, df) in series.items():
        last = df.iloc[-1]; D = C["D"]
        log(f"- {name}: tau_end {last['tau']:.3g}; vaporized {last['vap_cum_frac']:.3e}, liquid mass loss {last['mass_loss_frac']:.3e}, "
            f"A/A0 {last['A_ratio']:.3g}, W/D {last['W_D']:.3g}, drift {last['drift_D']:.3g} D, u_l/u2 {last['ul_u2']:.3g}, "
            f"fragments {int(last['n_frag'])}")
        if C["status"] and not C["status"].startswith("Complete"): log(f"    WARNING run status '{C['status']}' (not complete)")
        if C["shock"] and last["tau"] < C["t_end_tau"] - 0.05: log(f"    WARNING last plotfile at t/t* {last['tau']:.3g} < requested {C['t_end_tau']:.3g}")
        hi = C["prob_hi"]
        if np.nanmax(df["x_max"]) > hi[0] - D: log(f"    WARNING liquid within 1 D of the outflow boundary (x_max {np.nanmax(df['x_max']) / D:.3g} D)")
        if np.nanmax(df["y_max"]) > hi[1] - D: log(f"    WARNING liquid within 1 D of the top boundary (y_max {np.nanmax(df['y_max']) / D:.3g} D)")
        nol = df["tau"][~np.isfinite(df["yq_hi"])]
        if len(nol):
            log(f"    WARNING no cell with eta < 0.5 from t/t* {nol.iloc[0]:.3g} on ({len(nol)} plotfiles): the liquid is smeared over "
                f"the diffuse band; W/D, L/D, fragments and the eta = 0.5 contours are undefined there (mass, area and rate still are)")
        if not C["vap"] and abs(last["mass_loss_frac"]) > 1e-3:
            log(f"    note: no vaporization but liquid mass changed by {last['mass_loss_frac']:.2e} (numerical exchange at the diffuse interface)")
        if C["vap"]:
            log(f"    model vaporized {last['vap_cum_frac']:.3e} vs liquid mass loss {last['mass_loss_frac']:.3e} "
                f"(difference {last['mass_loss_frac'] - last['vap_cum_frac']:.2e} = numerical exchange)")
        r = np.nanmedian(df["A_cont"] / df["A_grad"])
        log(f"    interface measures: eta=0.5 contour / int|grad eta| median {r:.3f}; window AMR level {int(df['win_level'].median())}")
        log(f"    initial liquid mass {C['m0']:.4e} kg/m (full column); initial area {C['A0']:.4e} m (circle {C['A0_circle']:.4e} m)")

    # ---------------- vap / no-vap pairs and Mach sweeps ----------------
    pairs = []
    for name, (C, df) in series.items():
        if C["vap"]:
            for n2, (C2, df2) in series.items():
                if not C2["vap"] and C2["vapbase"] == C["vapbase"] and C2["group"] == C["group"]: pairs.append((name, n2))
    sweeps = {}
    for name, (C, df) in series.items():
        if C["shock"]: sweeps.setdefault((C["group"], C["layout"], C["vap"]), []).append((C, df))
    sweeps = {k: v for k, v in sweeps.items() if len(v) >= 2}
    log(f"\n{len(pairs)} vaporization on/off pair(s), {len(sweeps)} Mach sweep(s)")

    # ---------------- frames (parallel) ----------------
    T1 = time.time(); frame_jobs, gif_jobs, cmp_jobs = [], [], []
    me = os.path.getmtime(os.path.abspath(__file__))
    for name, (C, df) in series.items():
        C["outdir"] = os.path.join(out, C["group"], name); os.makedirs(C["outdir"], exist_ok=True)
        lim = dict(p=3.0, u=C["u2"] if C["shock"] else max(1.0, float(np.nanmax(np.abs(df["ul"])))),
                   T=(min(C["T1"], 280.0), (C["T0_stag"] * 1.05 if C["shock"] else float(df["T_g_max"].max()))),
                   M=2.0, rho=(np.log10(C["rho1"]) - 0.3, np.log10(C["rho_l"]) + 0.1), vap=max(float(df["vapmax_win"].max()), 1e-30))
        lim["w"] = float(np.nanmedian(df["vort99"])) * C["D"] / lim["u"] if lim["u"] > 0 else 1.0
        C["lim"] = lim
        kinds = [k for k in SPLITS if (k != "VapRate-MassFraction" or C["vap"])]
        rows = list(range(0, len(df), a.every)) + ([len(df) - 1] if (len(df) - 1) % a.every else [])
        for kind in kinds:
            fd = os.path.join(C["outdir"], "frames", kind); os.makedirs(fd, exist_ok=True)
            for n_, i in enumerate(rows):
                row = df.iloc[i].to_dict(); cf = C["cache"][row["plotfile"]]; png = os.path.join(fd, f"{n_:04d}.png")
                if not a.redo and os.path.isfile(png) and os.path.getmtime(png) > max(os.path.getmtime(cf), me): continue
                frame_jobs.append((kind, {k: v for k, v in C.items() if k not in ("plotfiles", "cache")}, lim, row, cf, png))
            gif_jobs.append((fd, os.path.join(C["outdir"], f"{kind}.gif"), a.fps))
    for nv, nn in pairs:
        Cv, dv = series[nv]; Cn, dn = series[nn]
        d = os.path.join(out, Cv["group"], "Comparisons", Cv["vapbase"]); os.makedirs(d, exist_ok=True)
        iv_of = {p_: k for k, p_ in enumerate(dv["plotfile"])}; in_of = {p_: k for k, p_ in enumerate(dn["plotfile"])}
        V = [q for q in cmpviews.get((nv, nn), []) if q[0] in iv_of and q[1] in in_of]
        if not V: log(f"  [cmp] {nv} vs {nn}: no comparison views, skipped"); continue
        for kind in CMP_KINDS:
            fd = os.path.join(d, "frames", f"CMP-{kind}"); os.makedirs(fd, exist_ok=True)
            for n_, (pv, pn, vxc, vh, vv) in enumerate(V):
                rv = dv.iloc[iv_of[pv]].to_dict(); rn = dn.iloc[in_of[pn]].to_dict(); rv["cmp_view"] = (vxc, vh, vv)
                fv, fn = Cv["cache"][rv["plotfile"]], Cn["cache"][rn["plotfile"]]; png = os.path.join(fd, f"{n_:04d}.png")
                if not a.redo and os.path.isfile(png) and os.path.getmtime(png) > max(os.path.getmtime(fv), os.path.getmtime(fn), me): continue
                strip = lambda C: {k: v for k, v in C.items() if k not in ("plotfiles", "cache")}
                cmp_jobs.append((kind, strip(Cv), strip(Cn), Cn["lim"], rv, rn, fv, fn, png))
            gif_jobs.append((fd, os.path.join(d, f"CMP-{kind}.gif"), a.fps))
    if not a.no_frames:
        log(f"\nFRAMES: {len(frame_jobs)} split + {len(cmp_jobs)} comparison frame(s) to draw")
        errs = [e for e in pool_map(draw_split, frame_jobs, a.nproc, "split frames") if e]
        errs += [e for e in pool_map(draw_cmp, cmp_jobs, a.nproc, "comparison frames") if e]
        for e in errs[:20]: log(f"  [frame error] {e}")
        if not a.no_gifs:
            errs = [e for e in pool_map(make_gif, gif_jobs, a.nproc, "gifs") if e]
            for e in errs: log(f"  [gif] {e}")
        log(f"  frames and GIFs done in {time.time() - T1:.0f} s")

    # ---------------- static plots (one job per run / pair / sweep, parallel) ----------------
    T2 = time.time(); log("\nSTATIC PLOTS")
    global _STATIC
    _STATIC = dict(series=series, pairs=pairs, sweeps=sweeps, out=out)
    sj = [("case", n) for n in series] + [("pair", p) for p in pairs] + [("sweep", k) for k in sweeps]
    for e in [e for e in pool_map(_static_job, sj, a.nproc, "static plot sets") if e]: log(f"  [plot error] {e}")
    log(f"  static plots done in {time.time() - T2:.0f} s")

    # ---------------- spreadsheet ----------------
    write_tables(out, series, summ, pairs)
    log(f"\nTOTAL {time.time() - T0:.0f} s")
    with open(os.path.join(out, "analysis_log.txt"), "w") as fh: fh.write("\n".join(LOG) + "\n")
    if not a.no_tar: tar_output(out, a.tar_frames)


def tar_output(out, frames=False):
    """<out>.tar.gz next to the output folder, without the cache (and without the per-frame PNGs unless asked)"""
    import tarfile
    T = time.time(); dst = out.rstrip("/") + ".tar.gz"; tmp = dst + ".part"; base = os.path.basename(out.rstrip("/"))
    def keep(ti):
        parts = ti.name.split("/")
        if len(parts) > 1 and parts[1] == "_cache": return None
        if not frames and "frames" in parts[1:-1] + ([parts[-1]] if ti.isdir() else []): return None
        return ti
    try:
        with tarfile.open(tmp, "w:gz", compresslevel=6) as tf: tf.add(out, arcname=base, filter=keep)
        os.replace(tmp, dst)
        print(f"tarred {out} -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB, {time.time() - T:.0f} s"
              f"{', frames included' if frames else ', without _cache/ and frames/'})", flush=True)
    except Exception as e:
        print(f"[tar] failed: {e!r} (partial file left at {tmp})", flush=True)


_STATIC = None
def _static_job(job):
    kind, key = job; S = _STATIC
    try:
        if kind == "case":
            C, df = S["series"][key]; plot_case_static(C, df, C["outdir"], C["cache"])
        elif kind == "pair":
            Cv, dv = S["series"][key[0]]; Cn, dn = S["series"][key[1]]
            d = os.path.join(S["out"], Cv["group"], "Comparisons", Cv["vapbase"]); os.makedirs(d, exist_ok=True)
            plot_vap_novap(Cv, Cn, dv, dn, d, Cv["cache"], Cn["cache"])
        else:
            g, lay, vap = key; d = os.path.join(S["out"], g, "Sweeps", f"{lay}_{'vap' if vap else 'novap'}"); os.makedirs(d, exist_ok=True)
            C0 = S["sweeps"][key][0][0]
            plot_sweep(S["sweeps"][key], d, f"{C0['fluid']}, $D$ = {C0['D'] * 1e6:.3g} $\\mu$m, {lay}, Vaporization {'On' if vap else 'Off'}")
        plt.close("all"); return None
    except Exception as e:
        plt.close("all"); return f"{kind} {key}: {e!r}\n{traceback.format_exc()}"


README_ROWS = [
    ("tau", "(t - t_impact)/t*, t* = D/u2 sqrt(rho_l/rho2); t_impact = shock arrival at the upstream column surface"),
    ("m_l", "liquid mass int rho_eta1 dA, full column, per unit depth [kg/m]"),
    ("mdot", "vaporization rate int Vap_dot_rho dA, full column [kg/(m s)]; mdot_norm = mdot t*/m0"),
    ("vap_cum_frac", "int mdot dt / m0 (mass removed by the film model)"),
    ("mass_loss_frac", "1 - m_l/m0 (all liquid lost: model + numerical exchange at the diffuse interface)"),
    ("A_grad / A_ratio", "interface length int |grad eta| dA (2D 'surface area' per unit depth) [m], and A/A0"),
    ("A_cont", "length of the eta = 0.5 contour in the stored window (check on A_grad)"),
    ("flux", "mdot / A_grad, mean vaporization mass flux over the interface [kg/(m2 s)]"),
    ("W_D, L_D", "cross-stream width (2 x 99.8% mass quantile of y) and streamwise length (0.2..99.8% mass quantiles of x) of eta < 0.5 liquid, over D"),
    ("drift_D, ul_u2", "mass-weighted liquid centroid drift / D and mean liquid x-velocity / u2"),
    ("Cd", "m_l du_l/dt / (0.5 rho2 (u2 - u_l)^2 D) per column (u_l Savitzky-Golay smoothed); valid while the liquid is one body"),
    ("Tfilm_l, Tfilm_g, BM, BT", "film-model surface temperature, gas film temperature, Spalding mass / heat transfer numbers, weighted by the local rate"),
    ("n_frag, frag_largest", "connected eta < 0.5 regions (>= 9 pixels) in the stored window, and the largest one's share of the liquid volume"),
    ("T_l_mean, T_l_max", "mass-weighted mean and maximum T of the core liquid (eta < 0.05); band_mass_frac = share of the liquid mass in 0.05 <= eta <= 0.95"),
    ("p_l_min/max", "liquid (eta < 0.5) pressure extremes; p_l_min < 0 means tension"),
    ("x_shock, Us", "largest x with p > p1 + (p2 - p1)/2 (incident shock); Us fitted where it is past the liquid, vs Rankine-Hugoniot Ms a1"),
    ("circ_pos/neg", "circulation of positive / negative vorticity in the upper half plane [m2/s]"),
    ("_1 / _2 suffix", "tandem pair: upstream / downstream column, liquid split at the mass-weighted two-means point in x"),
    ("We, Re, Oh", "rho2 u2^2 D/sigma; rho2 u2 D/mu(T2) with Sutherland air (runs are inviscid); mu_l/sqrt(rho_l sigma D) with NIST mu_l at 300 K"),
    ("T", "stiffened-gas temperature (p + p0)/(rho cv (gamma - 1)) with the solver's mixture rules; the plotfile T field "
          "(EOS::MixedTemperature, p + gamma p0) reads ~2x high in the liquid and is converted on reading"),
    ("Mirror", "half / quarter domains are mirrored: extensive values are for the full column"),
]


def write_tables(out, series, summ, pairs):
    import pandas as pd
    ic_cols = ["name", "group", "fluid", "vap", "vap_model", "surface_tension", "n_drops", "D", "gap", "Ms", "p1", "T1", "rho1", "a1",
               "p2", "T2", "rho2", "u2", "M2", "Us", "T0_stag", "rho_l", "density_ratio", "gamma_l", "pinf_l", "sigma", "Lv", "mu_l",
               "mu_g2", "We", "Re", "Oh", "tstar", "t_impact", "t_end_tau", "stop_time", "plot_dt", "epsilon", "dx0", "dx_fine", "D_dx",
               "max_level", "n_cell", "domain_D", "p_refinement_max_level", "mirror", "nprocs", "run_time_h", "core_hours", "status",
               "start", "git", "deck", "run"]
    ic = pd.DataFrame([{k: (str(C.get(k)) if isinstance(C.get(k), list) else C.get(k)) for k in ic_cols} for C, _ in series.values()])
    sm = pd.DataFrame(summ)
    vv = []
    for nv, nn in pairs:
        Cv, dv = series[nv]; Cn, dn = series[nn]
        r = dict(case=Cv["vapbase"], Ms=Cv["Ms"])
        for tm in TAU_MARKS:
            if tm > min(dv["tau"].iloc[-1], dn["tau"].iloc[-1]) + 1e-9: continue
            for col, lab in (("A_ratio", "A/A0"), ("W_D", "W/D"), ("drift_D", "Drift/D"), ("ul_u2", "u_l/u2"), ("m_l", "m_l")):
                a_, b_ = at_tau(dv, col, tm), at_tau(dn, col, tm)
                r[f"{lab} on/off-1 [%] @tau={tm:g}"] = 100 * (a_ / b_ - 1) if b_ else float("nan")
            r[f"VapFrac on @tau={tm:g}"] = at_tau(dv, "vap_cum_frac", tm)
        vv.append(r)
    vv = pd.DataFrame(vv)
    pr = []
    for C, df in series.values():
        if C["n_drops"] == 2 and "mdot_1" in df:
            for tm in TAU_MARKS:
                if tm > df["tau"].iloc[-1] + 1e-9: continue
                pr.append(dict(case=C["name"], tau=tm, **{f"{lab}_{k}": at_tau(df, f"{c}_{k}", tm) for k in (1, 2)
                          for c, lab in (("vap_cum_frac", "VapFrac"), ("A_ratio", "A/A0"), ("drift_D", "Drift/D"), ("ul_u2", "u_l/u2"))}))
    pr = pd.DataFrame(pr)
    ts = pd.concat([df.assign(case=n) for n, (C, df) in series.items()], ignore_index=True)
    ts = ts[["case"] + [c for c in ts.columns if c != "case"]]
    rd = pd.DataFrame(README_ROWS, columns=["quantity", "definition"])
    csvd = os.path.join(out, "tables"); os.makedirs(csvd, exist_ok=True)
    for nm, t in (("Cases_IC", ic), ("Summary", sm), ("Vap_vs_NoVap", vv), ("TandemPair", pr), ("TimeSeries", ts), ("README", rd)):
        t.to_csv(os.path.join(csvd, nm + ".csv"), index=False)
    xl = os.path.join(out, "AIAA2027_Metrics.xlsx")
    try:
        with pd.ExcelWriter(xl, engine="openpyxl") as W:
            for nm, t in (("README", rd), ("Cases_IC", ic), ("Summary", sm), ("Vap_vs_NoVap", vv), ("TandemPair", pr), ("TimeSeries", ts)):
                if len(t): t.to_excel(W, sheet_name=nm, index=False)
                ws = W.sheets.get(nm)
                if ws is None: continue
                ws.freeze_panes = "B2"
                for col in ws.columns:
                    w = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 60)
        log(f"\nwrote {xl} and tables/*.csv")
    except Exception as e:
        log(f"\n[xlsx not written: {e!r}]; CSVs in {csvd}")


if __name__ == "__main__":
    main()
