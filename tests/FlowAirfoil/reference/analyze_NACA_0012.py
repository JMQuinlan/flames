"""
NACA 0012 lift/drag ANALYSIS for Hydro2 (embedded-solid, compressible).

ANALYSIS-ONLY: this script never runs the solver.  overnight_study.py runs the
AoA sweep; each case directory holds a copy of the exact input that was run,
and this script
  * auto-detects case directories and their AoA from the directory NAME
    (aoa04, aoa12, aoa07.5, a004.0, ... -- no AOAS list to keep in sync), and
  * reads the flow constants it needs (rho, U, p_inf, gamma, mu, brinkman)
    from that per-case input copy -- so it always normalizes with the q_inf
    the case actually ran, not a hard-coded one.

Forces are STEADY-WINDOW AVERAGED over all plotfiles with t >= 18 (mean and
std reported; error bars on the polar), falling back to the last plotfile for
runs that end earlier.  TWO independent methods per snapshot:
  (A) pressure surface integral   F = -integral (p - p_inf) grad(phi) dV
  (B) Brinkman penalty reaction   F =  integral brinkman (1-phi)(M - Ms) dV
      (only when the case ran an explicit constant solid.brinkman > 0;
       skipped for the AUTO wavespeed-scaled penalty, which is per-cell)
Compared against thin-airfoil + Prandtl-Glauert (Cl = 2 pi alpha / beta) and
tabulated NACA 0012 experiment (Abbott & von Doenhoff / NASA Ladson TM-4074).

usage:
  python3 analyze_NACA_0012.py [workdir]      # default bin/tests/FlowAirfoil/NACA_0012/sweep

The geometry helpers (naca0012 / rotate / write_phi_bmp) live here as a small
shared library; overnight_study.py imports them to generate per-case bitmaps.
"""
import os, sys, glob, math, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "Images")             # all plots go here; reference/ stays scripts-only
os.makedirs(IMAGES, exist_ok=True)

# ---- airfoil selection ----
# The GENERATION side (which template/test dir/work dir/profile the sweep
# uses).  The ANALYSIS side is per-case: parse_case_input() reads the NACA
# digits out of each case's input copy, so old sweeps of other sections keep
# analyzing correctly regardless of this constant.
AIRFOIL = "NACA_2408"     # thin (8%) + 2% camber: the stable low-Re regime.
PREFIX = AIRFOIL.replace("_", "").lower()      # plot filename prefix, e.g. naca2408
PROFILES = {              # name -> (m camber, p camber pos, t/c thickness)
    "NACA_0012": (0.00, 0.0, 0.12),
    "NACA_2408": (0.02, 0.4, 0.08),
    "NACA_2412": (0.02, 0.4, 0.12),
}

TESTDIR = os.path.normpath(os.path.join(HERE, "..", AIRFOIL))
FLAMES = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SOLVER = os.path.join(FLAMES, "bin", "hydro2-2d-g++")

# Default sweep work dir: alongside the test's own output (NOT /tmp -- a full
# 21-case sweep at max_level 5 writes tens of GB and filled /tmp once already).
# Case dirs: .../<WORK>/aoa00../{input, phi.bmp, out/*cell}
# NOTE: new dir per geometry generation -- the resume logic skips completed
# case dirs, so reusing the old "sweep" dir would silently keep blunt-TE
# results.  The blunt-TE 21-case polar is preserved in .../sweep.
RE_TAG = "Re50k"          # sweep tag: mu in the template must match (Re = 0.3/mu)
WORK = os.path.join(FLAMES, "bin", "tests", "FlowAirfoil", AIRFOIL,
                    f"{AIRFOIL.replace('_', '')}_{RE_TAG}")

def images_dir(tag=""):
    """Per-sweep image folder Images_<tag>/ (sibling of Images/, not nested);
    plain Images/ when no tag.  Created on demand."""
    d = os.path.join(HERE, f"Images_{tag}") if tag else IMAGES
    os.makedirs(d, exist_ok=True)
    return d

# ---- geometry / raster constants ----
CHORD = 1.0
DOM_LO, DOM_HI = (-4.0, -5.0), (8.0, 5.0)   # (legacy full-domain box; kept for sibling scripts)
# TIGHT bitmap raster box: IC::BMP clamps to the edge pixel outside coord.lo/hi
# (edge = pure fluid), so the bmp only needs to cover the foil + margin.  This
# keeps the file small at sharp skins.  MUST match solid.phi.ic.bmp.coord.lo/hi
# in the template input.  Covers all rotations to AoA ~ 20 deg.
BMP_LO, BMP_HI = (-0.75, -0.5), (1.0, 0.5)
EPS = 0.0025                       # solid tanh skin baked into generated bitmaps
                                   # (1.28 finest cells at max_level 5; sharp effective TE)
NX_BMP, NY_BMP = 1400, 800         # pixel = 0.00125 = EPS/2 over the tight box

# ---- reference/legacy constants (per-case analysis parses the case input
# ---- instead; these are defaults for sibling scripts that import us) ----
GAMMA = 1.4
MACH = 0.3
RHO, P = 1.0, 1.0 / GAMMA          # c = sqrt(gamma p/rho) = 1  ->  U = Mach
U = MACH
P_INF = P
Q_INF = 0.5 * RHO * U ** 2
BETA_PG = math.sqrt(1 - MACH ** 2)

# tabulated NACA 0012 experiment (low-speed, Re~6e6; Abbott & von Doenhoff / Ladson)
EXP_LIFT_SLOPE = 0.11              # dCl/dAoA per deg (linear range)
EXP_CD0 = 0.0065                   # min profile drag (viscous)

# ---------- geometry (shared with overnight_study.py) ----------
# SHARP TE CLOSURE: the diffuse band wraps corners with radius ~EPS, so the
# effective body of the raw section ends in a rounded base ~2*EPS thick --
# identified (splitter A/B) as the mechanism that killed circulation.  Close
# the drawn section with a shallow wedge running to a sharp APEX a short
# distance aft, so the band's rounding happens on a thin sliver behind the
# true TE instead of fattening it.  (Planform grows by TE_EXT = 2.5%c; Cl/Cd
# stay normalized by CHORD.)
TE_EXT = 0.0     # apex distance aft of the nominal TE (0 = REGULAR section, no
                 # sharp-closure wedge; 0.025 = 10*EPS was used for the sharp-TE
                 # study -- blunt vs sharp made little difference at eps=0.0025)

def naca0012(n=80):
    xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))      # cosine spacing 0..1
    yt = 0.6 * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    x = (xc - 0.5) * CHORD
    up = np.column_stack([x, yt*CHORD]); lo = np.column_stack([x[::-1], -yt[::-1]*CHORD])
    if TE_EXT > 0.0:
        apex = np.array([[0.5 * CHORD + TE_EXT, 0.0]])     # sharp closure point
        return np.vstack([up, apex, lo])
    return np.vstack([up, lo])

def naca4(m=0.02, p=0.4, tc=0.08, n=120):
    """Generic NACA 4-digit section (camber m at position p, thickness tc),
    chord CHORD centered on x=0.  Reduces to naca0012() for (0, 0, 0.12)."""
    xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 5.0 * tc * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2
                     + 0.2843*xc**3 - 0.1015*xc**4)
    if m > 0.0 and p > 0.0:
        yc  = np.where(xc < p, m/p**2 * (2*p*xc - xc**2),
                       m/(1-p)**2 * ((1-2*p) + 2*p*xc - xc**2))
        dyc = np.where(xc < p, 2*m/p**2 * (p - xc),
                       2*m/(1-p)**2 * (p - xc))
    else:
        yc = np.zeros_like(xc); dyc = np.zeros_like(xc)
    th = np.arctan(dyc)
    xu = xc - yt*np.sin(th); yu = yc + yt*np.cos(th)
    xl = xc + yt*np.sin(th); yl = yc - yt*np.cos(th)
    up = np.column_stack([(xu - 0.5)*CHORD, yu*CHORD])
    lo = np.column_stack([(xl[::-1] - 0.5)*CHORD, yl[::-1]*CHORD])
    return np.vstack([up, lo])

def profile(n=120):
    """Polygon of the SELECTED airfoil (module constant AIRFOIL)."""
    m_, p_, tc_ = PROFILES[AIRFOIL]
    return naca4(m_, p_, tc_, n)

def alpha_L0_deg(m, p):
    """Thin-airfoil zero-lift angle [deg] of the NACA 4-digit camber line
    (Glauert integral, numerically).  0 for symmetric; ~ -2.07 for m=2%,p=0.4."""
    if m <= 0.0 or p <= 0.0:
        return 0.0
    th = np.linspace(0.0, np.pi, 2001)
    xc = 0.5 * (1 - np.cos(th))
    dyc = np.where(xc < p, 2*m/p**2 * (p - xc), 2*m/(1-p)**2 * (p - xc))
    a = -(1.0/np.pi) * np.trapezoid(dyc * (np.cos(th) - 1.0), th)
    return math.degrees(a)

def rotate(poly, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return poly @ np.array([[c, s], [-s, c]]).T            # +deg = nose-up -> +lift

def write_phi_bmp(poly, path):
    from matplotlib.path import Path
    from PIL import Image
    x = np.linspace(BMP_LO[0], BMP_HI[0], NX_BMP); y = np.linspace(BMP_LO[1], BMP_HI[1], NY_BMP)
    X, Y = np.meshgrid(x, y)
    inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    d2 = np.full(X.shape, np.inf)
    n = len(poly)
    for k in range(n):
        a = poly[k]; b = poly[(k+1) % n]; ab = b - a; L2 = ab @ ab
        if L2 < 1e-30:
            seg = np.hypot(X-a[0], Y-a[1])
        else:
            t = np.clip(((X-a[0])*ab[0] + (Y-a[1])*ab[1]) / L2, 0, 1)
            seg = np.hypot(X-(a[0]+t*ab[0]), Y-(a[1]+t*ab[1]))
        d2 = np.minimum(d2, seg)
    d = np.where(inside, -d2, d2)
    phi = 0.5 * (1 + np.tanh(d / EPS))                     # 1 fluid, 0 solid
    g = np.clip(np.round(255*np.flipud(phi)), 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([g, g, g]), "RGB").save(path)

# ---------- case discovery ----------
# AoA from the directory NAME: aoa04, aoa12, aoa07.5, aoa_4, a004.0, ...
_AOA_PATTERNS = [re.compile(r"aoa[-_]?(-?\d+(?:\.\d+)?)", re.I),
                 re.compile(r"^a[-_]?(-?\d+(?:\.\d+)?)$", re.I)]

def detect_aoa(name):
    base = os.path.basename(os.path.normpath(name))
    for pat in _AOA_PATTERNS:
        mm = pat.search(base)
        if mm:
            return float(mm.group(1))
    return None

def parse_case_input(inp_path):
    """Flow constants from the input copy the case actually ran."""
    txt = open(inp_path).read()
    def val(key, default=None):
        mm = re.search(rf"^\s*{re.escape(key)}\s*=\s*([^#\n]+)", txt, re.M)
        return mm.group(1).split()[0] if mm else default
    rho = float(val("density0.ic.constant.value", "1.0"))
    u   = float(val("velocity0.ic.expression.region0", "0").strip('"\''))
    p   = float(val("pressure0.ic.constant.value", str(1.0 / 1.4)))
    gam = float(val("eos0.gamma", "1.4"))
    mu  = float(val("mu0", "0.0"))
    brk = float(val("solid.brinkman", "-1.0"))
    ppw = float(val("solid.penalty_power", "1.0"))
    c   = math.sqrt(gam * p / rho)
    # Airfoil profile from the "NACA dddd" digits in the input header (the
    # sweep clones the airfoil's template, so this identifies the section):
    #   digit 1 = camber %, digit 2 = camber position/10, digits 3-4 = t/c %.
    nm = re.search(r"NACA\s*(\d)(\d)(\d\d)", txt)
    if nm:
        m_c, p_c, t_c = int(nm.group(1))/100.0, int(nm.group(2))/10.0, int(nm.group(3))/100.0
        naca_name = f"NACA {nm.group(1)}{nm.group(2)}{nm.group(3)}"
    else:
        m_c, p_c, t_c, naca_name = 0.0, 0.0, 0.12, "NACA 0012"
    return dict(rho=rho, U=u, p_inf=p, gamma=gam, mu=mu, brinkman=brk, penalty_power=ppw,
                mach=u / c, q_inf=0.5 * rho * u * u,
                re=(rho * u * CHORD / mu) if mu > 0 else float("inf"),
                naca=naca_name, alpha_L0=alpha_L0_deg(m_c, p_c))

def find_cases(work):
    """Case dirs under `work` that have an input copy, a plotfile, and a
    parsable AoA in their name."""
    cases = []
    for rd in sorted(glob.glob(os.path.join(work, "*"))):
        if not os.path.isdir(rd):
            continue
        aoa = detect_aoa(rd)
        inp = os.path.join(rd, "input")
        pfs = sorted(glob.glob(os.path.join(rd, "out", "*cell")))
        if aoa is None or not os.path.isfile(inp) or not pfs:
            continue
        cases.append(dict(rd=rd, aoa=aoa, inp=inp, plot=os.path.join(rd, "out")))
    return sorted(cases, key=lambda c: c["aoa"])

# ---------- forces (two methods) ----------
# STEADY-STATE WINDOW AVERAGE: forces are ensemble-averaged over every plotfile
# with t >= T_AVG_START (Wagner buildup ~17 t-units + bubble equilibration ~20
# -> settled by ~18; stop_time 24 gives a 6-unit window).  Residual shedding
# oscillation (+-0.02-0.05 in Cl) averages out instead of adding snapshot
# noise to the polar; the std is reported as an error bar.  Falls back to the
# LAST plotfile if a (short/crashed) run never reaches the window.
T_AVG_START = 18.0
# Covering-grid level for the force integrals.  None = finest (accurate; the
# eps=0.0025 band needs level 5) -- but a finest-level covering grid on this
# domain is ~10.5M cells PER FIELD, so a full window-averaged analysis peaks
# around a GB of RAM per snapshot.  The overnight driver's LIVE refresh passes
# lev=3 explicitly (cheap, trend-accurate); run the standalone CLI (finest)
# once at the end for the publication polar.
FORCE_LEV = None
_MODULE_DEFAULT = object()   # sentinel: "use FORCE_LEV as it is NOW" (a plain
                             # default arg would freeze its import-time value)

# Both force integrands (grad(phi) and (1-phi)^p) are IDENTICALLY ZERO outside
# the diffuse band, so the covering grid only needs to span the foil + margin
# (superset of the BMP box for all AoAs) -- NOT the whole domain.  At finest
# level that is ~460k cells instead of ~10.5M: the full-domain version OOM'd
# an overnight analysis pass.
FORCE_BOX = ((-0.80, -0.55), (1.05, 0.55))

def _grid_box(pf, lev_cap=_MODULE_DEFAULT, box=None):
    """Covering grid over a cell-aligned sub-box (default FORCE_BOX).
    Returns (t, field-getter, dx, dy, x_left, y_left)."""
    import yt
    if lev_cap is _MODULE_DEFAULT:
        lev_cap = FORCE_LEV
    if box is None:
        box = FORCE_BOX
    ds = yt.load(pf)
    L = ds.index.max_level if lev_cap is None else min(lev_cap, ds.index.max_level)
    dom_lo = np.asarray(ds.domain_left_edge)
    dims_full = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
    dxv = np.asarray(ds.domain_width) / dims_full
    i0 = np.zeros(3, dtype=int); i1 = dims_full.astype(int).copy()
    for d in range(2):
        i0[d] = max(0, int(np.floor((box[0][d] - dom_lo[d]) / dxv[d])))
        i1[d] = min(int(dims_full[d]), int(np.ceil((box[1][d] - dom_lo[d]) / dxv[d])))
    left = dom_lo + i0 * dxv
    cg = ds.covering_grid(level=L, left_edge=left, dims=(i1 - i0))
    g = lambda f: np.asarray(cg[f])[:, :, 0]
    return float(ds.current_time), g, float(dxv[0]), float(dxv[1]), float(left[0]), float(left[1])

def _grid_at(pf, lev_cap=_MODULE_DEFAULT):
    t, g, dx, dy, _, _ = _grid_box(pf, lev_cap)
    return t, g, dx, dy

def _window_plotfiles(plot):
    """Plotfiles in the averaging window (t >= T_AVG_START), else the last one."""
    import yt
    pfs = sorted(glob.glob(os.path.join(plot, "*cell")))
    win = []
    for pf in pfs:
        try:
            if float(yt.load(pf).current_time) >= T_AVG_START:
                win.append(pf)
        except Exception:
            pass
    return win if win else pfs[-1:]

def _forces_snapshot(g, dx, dy, p_inf, brinkman=None, penalty_power=1.0):
    """(pressure-integral force, Brinkman reaction force or None) for one grid."""
    phi = g("phi"); p = g("pressure")
    gx = np.gradient(phi, dx, axis=0); gy = np.gradient(phi, dy, axis=1)
    pp = p - p_inf
    Fp = (-np.sum(pp * gx) * dx * dy, -np.sum(pp * gy) * dx * dy)
    Fk = None
    if brinkman is not None and brinkman > 0.0:
        # weight matches the solver: lambda * (1-phi)^penalty_power.  (AUTO
        # brinkman < 0 is per-cell wavespeed-scaled -> not reconstructable.)
        w = brinkman * (1 - phi) ** penalty_power
        Fk = (np.sum(w * g("momentumx")) * dx * dy,
              np.sum(w * g("momentumy")) * dx * dy)
    return Fp, Fk

# Back-compat single-snapshot wrappers (LAST plotfile only).
def forces_pressure(plot, p_inf=P_INF):
    pf = sorted(glob.glob(os.path.join(plot, "*cell")))[-1]
    _, g, dx, dy = _grid_at(pf)
    return _forces_snapshot(g, dx, dy, p_inf)[0]

def forces_penalty(plot, brinkman, penalty_power=1.0):
    if brinkman is None or brinkman <= 0.0:
        return None
    pf = sorted(glob.glob(os.path.join(plot, "*cell")))[-1]
    _, g, dx, dy = _grid_at(pf)
    return _forces_snapshot(g, dx, dy, P_INF, brinkman, penalty_power)[1]

# ---------- analysis ----------
def analyze_case(case, lev=_MODULE_DEFAULT, use_cache=True):
    """Window-averaged forces for one case.  Results are cached in
    <case>/analysis.json keyed on (plotfile set, lev, window) so the live
    in-sweep refresh never re-loads a finished case's plotfiles."""
    import json
    cfg = parse_case_input(case["inp"])
    qc = cfg["q_inf"] * CHORD
    win = _window_plotfiles(case["plot"])
    lev_key = "finest" if (lev is _MODULE_DEFAULT and FORCE_LEV is None) or lev is None \
              else str(FORCE_LEV if lev is _MODULE_DEFAULT else lev)
    key = dict(last=os.path.basename(win[-1]), n=len(win), lev=lev_key, t0=T_AVG_START, v=2)
    cpath = os.path.join(case["rd"], "analysis.json")
    if use_cache and os.path.isfile(cpath):
        try:
            cached = json.load(open(cpath))
            if cached.get("key") == key:
                row = cached["row"]; row["cfg"] = cfg; row["rd"] = case["rd"]
                return row
        except Exception:
            pass
    clp, cdp, clk, cdk = [], [], [], []
    for pf in win:
        _, g, dx, dy = _grid_at(pf, lev)
        Fp, Fk = _forces_snapshot(g, dx, dy, cfg["p_inf"],
                                  cfg.get("brinkman"), cfg.get("penalty_power", 1.0))
        clp.append(Fp[1] / qc); cdp.append(Fp[0] / qc)
        if Fk is not None:
            clk.append(Fk[1] / qc); cdk.append(Fk[0] / qc)
        del g   # release the covering grid before loading the next snapshot
    # First non-zero-time snapshot (impulsive-start loading; "First*" charts).
    cl_first = cd_first = None
    for pf in sorted(glob.glob(os.path.join(case["plot"], "*cell"))):
        t, g, dx, dy = _grid_at(pf, lev)
        if t > 0.0:
            Fp, _ = _forces_snapshot(g, dx, dy, cfg["p_inf"])
            cl_first = float(Fp[1] / qc); cd_first = float(Fp[0] / qc)
            del g
            break
        del g
    row = dict(aoa=case["aoa"], n_avg=len(win),
               Cl_p=float(np.mean(clp)), Cd_p=float(np.mean(cdp)),
               Cl_std=float(np.std(clp)), Cd_std=float(np.std(cdp)),
               Cl_k=float(np.mean(clk)) if clk else None,
               Cd_k=float(np.mean(cdk)) if cdk else None,
               Cl_first=cl_first, Cd_first=cd_first)
    try:
        json.dump(dict(key=key, row=row), open(cpath, "w"), indent=1)
    except Exception:
        pass
    row["cfg"] = cfg; row["rd"] = case["rd"]
    return row

def analyze_workdir(work=WORK, quiet=False, lev=_MODULE_DEFAULT):
    cases = find_cases(work)
    if not cases:
        if not quiet:
            print(f"no analyzable cases under {work} (need <dir with AoA in name>/input + out/*cell)")
        return []
    rows = []
    for c in cases:
        try:
            r = analyze_case(c, lev=lev)
        except Exception as e:
            if not quiet:
                print(f"  {os.path.basename(c['rd'])}: analysis FAILED ({e!r})")
            continue
        rows.append(r)
        if not quiet:
            beta = math.sqrt(max(1e-12, 1 - r["cfg"]["mach"] ** 2))
            th = 2 * math.pi / beta * math.radians(r["aoa"] - r["cfg"].get("alpha_L0", 0.0))
            pen = "" if r["Cl_k"] is None else f" Cl(pen)={r['Cl_k']:+.4f} Cd(pen)={r['Cd_k']:.4f}"
            print(f"  AoA={r['aoa']:5.1f}: Cl={r['Cl_p']:+.4f}+-{r['Cl_std']:.4f} (theory {th:+.4f})  "
                  f"Cd={r['Cd_p']:.4f}  [{r['n_avg']} snaps, t>={T_AVG_START:g}]{pen}")
    if rows:
        plot_results(rows, tag=os.path.basename(os.path.normpath(work)))
    return rows

def plot_results(rows, tag=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["aoa"])
    cfg = rows[0]["cfg"]                       # sweep shares one flow condition
    mach = cfg["mach"]; beta = math.sqrt(max(1e-12, 1 - mach ** 2))
    A = np.array([r["aoa"] for r in rows]); ar = np.radians(A)
    Clp = np.array([r["Cl_p"] for r in rows]); Cdp = np.array([r["Cd_p"] for r in rows])
    have_pen = all(r["Cl_k"] is not None for r in rows)
    aL0 = cfg.get("alpha_L0", 0.0)             # zero-lift angle of the section [deg]
    naca = cfg.get("naca", "NACA 0012")
    Cl_th = 2 * np.pi / beta * np.radians(A - aL0)
    Cl_exp = EXP_LIFT_SLOPE * (A - aL0)
    re_str = "inviscid" if not np.isfinite(cfg["re"]) else rf"$Re = {cfg['re']:.0f}$"
    outdir = images_dir(tag)
    XLAB = r"$\alpha$ [$^\circ$]"
    Cls = np.array([r.get("Cl_std", 0.0) for r in rows])
    Cds = np.array([r.get("Cd_std", 0.0) for r in rows])

    # ---- 3-panel overview ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    ax[0].errorbar(A, Clp, yerr=Cls, fmt="ko-", ms=4, capsize=2.5, lw=1.2,
                   label=rf"present ($\bar{{C}}_l \pm \sigma$, $t \geq {T_AVG_START:g}$)")
    if have_pen:
        ax[0].plot(A, [r["Cl_k"] for r in rows], "s--", color="0.5", ms=4, label="Brinkman reaction")
    th_lab = (rf"thin airfoil $2\pi(\alpha-\alpha_0)/\beta$, $\alpha_0={aL0:.2f}^\circ$"
              if abs(aL0) > 1e-6 else rf"thin airfoil $2\pi\alpha/\beta$")
    ax[0].plot(A, Cl_th, "r-", lw=1.2, label=th_lab)
    ax[0].plot(A, Cl_exp, "k:", lw=1.2, label=rf"exp.-slope $0.11(\alpha-\alpha_0)/^\circ$")
    ax[0].set_ylabel(r"$\bar{C}_l$", fontsize=12)
    ax[1].errorbar(A, Cdp, yerr=Cds, fmt="ko-", ms=4, capsize=2.5, lw=1.2, label="present")
    if have_pen:
        ax[1].plot(A, [r["Cd_k"] for r in rows], "s--", color="0.5", ms=4, label="Brinkman reaction")
    ax[1].axhline(EXP_CD0, color="k", ls=":", lw=1.2, label=rf"experiment $C_{{d0}} \approx {EXP_CD0}$")
    ax[1].set_ylabel(r"$\bar{C}_d$", fontsize=12)
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[2].plot(A, Clp / np.where(Cdp == 0, np.nan, Cdp), "ko-", ms=4, lw=1.2)
    ax[2].set_ylabel(r"$\bar{C}_l/\bar{C}_d$", fontsize=12)
    from matplotlib.ticker import MultipleLocator
    for a in ax:
        a.set_xlabel(XLAB, fontsize=12); a.grid(alpha=0.3)
        a.xaxis.set_major_locator(MultipleLocator(2))
    ax[0].legend(fontsize=8, frameon=False); ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle(rf"{naca},  $M_\infty = {mach:.2f}$,  {re_str}   ({len(rows)} cases)")
    out = os.path.join(outdir, f"{PREFIX}_polars.png")
    plt.tight_layout(); plt.savefig(out, dpi=130); plt.close()
    print("wrote", out)

    # ---- standalone square Cl and Cd polars (publication style) ----
    for ylab, Y, Ys, extras, name in [
        (r"$\bar{C}_l$", Clp, Cls,
         [(A, Cl_th, "r-", th_lab),
          (A, Cl_exp, "k:", rf"exp.-slope $0.11(\alpha-\alpha_0)/^\circ$")], "Cl"),
        (r"$\bar{C}_d$", Cdp, Cds, [], "Cd"),
    ]:
        fig, a = plt.subplots(figsize=(5.6, 5.6))
        a.errorbar(A, Y, yerr=Ys, fmt="ko", ms=5, capsize=3, lw=1.0, elinewidth=1.0,
                   label=rf"present ($t \geq {T_AVG_START:g}$)")
        for xe, ye, st, lb in extras:
            a.plot(xe, ye, st, lw=1.3, label=lb)
        a.set_xlabel(XLAB, fontsize=13)
        a.set_ylabel(ylab, fontsize=13)
        a.grid(alpha=0.3)
        a.xaxis.set_major_locator(MultipleLocator(2))
        a.set_box_aspect(1)                    # square plot area
        a.legend(fontsize=9, frameon=False, loc="upper left")
        a.set_title(rf"{naca},  $M_\infty = {mach:.2f}$,  {re_str}", fontsize=10)
        out = os.path.join(outdir, f"{PREFIX}_{name}.png")
        plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
        print("wrote", out)

    # ---- "First*" charts: force at the FIRST non-zero-time snapshot ----
    # (impulsive-start loading -- a single snapshot, so no error bars)
    if all(r.get("Cl_first") is not None for r in rows):
        Clf = np.array([r["Cl_first"] for r in rows])
        Cdf = np.array([r["Cd_first"] for r in rows])
        # Label with the actual snapshot TIME (index-only load; no grid data).
        t_first = None
        try:
            import yt as _yt
            _yt.set_log_level(50)
            for pf in sorted(glob.glob(os.path.join(rows[0]["rd"], "out", "*cell"))):
                tt = float(_yt.load(pf).current_time)
                if tt > 0.0:
                    t_first = tt
                    break
        except Exception:
            pass
        tstr = rf"$t = {t_first:.3g}$" if t_first is not None else r"first $t > 0$ snapshot"
        for ylab, Y, extras, name in [
            (rf"$C_l$ at {tstr}", Clf,
             [(A, Cl_th, "r-", th_lab),
              (A, Cl_exp, "k:", rf"exp.-slope $0.11(\alpha-\alpha_0)/^\circ$")], "FirstCl"),
            (rf"$C_d$ at {tstr}", Cdf, [], "FirstCd"),
        ]:
            fig, a = plt.subplots(figsize=(5.6, 5.6))
            a.plot(A, Y, "ko", ms=5, label=rf"present ({tstr})")
            for xe, ye, st, lb in extras:
                a.plot(xe, ye, st, lw=1.3, label=lb)
            a.set_xlabel(XLAB, fontsize=13)
            a.set_ylabel(ylab, fontsize=13)
            a.grid(alpha=0.3)
            a.xaxis.set_major_locator(MultipleLocator(2))
            a.set_box_aspect(1)
            a.legend(fontsize=9, frameon=False, loc="upper left")
            a.set_title(rf"{naca},  $M_\infty = {mach:.2f}$,  {re_str}", fontsize=10)
            out = os.path.join(outdir, f"{PREFIX}_{name}.png")
            plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
            print("wrote", out)

    if len(A) > 1:
        # fit the lift slope on the pre-stall linear range only
        lin = A <= 10.0
        if lin.sum() > 1:
            sl_p = np.polyfit(ar[lin], Clp[lin], 1)[0]
            print(f"lift slope dCl/dalpha (alpha<=10): CFD={sl_p:.2f}/rad   "
                  f"theory 2pi/beta={2*np.pi/beta:.2f}/rad   "
                  f"exp~{math.degrees(EXP_LIFT_SLOPE):.2f}/rad")

# ---------- flow-field gallery ----------
# View window and rendering level for the Cp/streamline panels.
VIEW_BOX  = ((-0.8, -0.6), (1.4, 0.6))
FIELD_LEV = 4          # smooth enough for images, cheap to extract
CP_RANGE  = (-2.0, 2.0)

FIELD_ROWS = 6         # angles shown in the gallery (rows); 4-8 fits a paper page

def plot_fields(work=WORK, aoas=None, lev=FIELD_LEV, tag=None, rows=FIELD_ROWS):
    """n x 3 gallery: rows = angles of attack, columns = t = 0 (initial
    condition), midpoint, and final snapshot.  Airfoil filled black, Cp field
    blue->red, streamlines overlaid.  Sized for a portrait paper figure."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cases = find_cases(work)
    if not cases:
        print(f"plot_fields: no cases under {work}")
        return
    if aoas is not None:
        cases = [c for c in cases if c["aoa"] in set(aoas)]
    if len(cases) > rows:                   # pick `rows` evenly spaced angles
        idx = sorted({int(round(i)) for i in np.linspace(0, len(cases) - 1, rows)})
        cases = [cases[i] for i in idx]
    if tag is None:
        tag = os.path.basename(os.path.normpath(work))

    def snap_files(plot):
        pfs = sorted(glob.glob(os.path.join(plot, "*cell")))
        first = pfs[min(1, len(pfs) - 1)]             # first t>0 snapshot (t=0.5)
        return first, pfs[len(pfs) // 2], pfs[-1]     # impulse, midpoint, final

    n = len(cases)
    # Slot aspect matched to the data aspect of VIEW_BOX so aspect-equal
    # panels fill their slots (no interior padding between rows).
    fig, axs = plt.subplots(n, 3, figsize=(10.0, 1.58 * n),
                            squeeze=False, sharex=True, sharey=True)
    im = None
    for i, c in enumerate(cases):
        cfg = parse_case_input(c["inp"])
        for j, pf in enumerate(snap_files(c["plot"])):
            t, g, dx, dy, x0, y0 = _grid_box(pf, lev, VIEW_BOX)
            a = axs[i][j]
            phi, p, rho = g("phi"), g("pressure"), g("density")
            vx = g("momentumx") / rho; vy = g("momentumy") / rho
            x = x0 + (np.arange(phi.shape[0]) + 0.5) * dx
            y = y0 + (np.arange(phi.shape[1]) + 0.5) * dy
            cp = (p - cfg["p_inf"]) / cfg["q_inf"]
            im = a.pcolormesh(x, y, cp.T, cmap="coolwarm", shading="auto",
                              vmin=CP_RANGE[0], vmax=CP_RANGE[1], rasterized=True)
            a.streamplot(x, y, vx.T, vy.T, density=0.8, color="0.2",
                         linewidth=0.5, arrowsize=0.6)
            a.contourf(x, y, phi.T, levels=[-0.5, 0.5], colors=["k"])   # solid
            a.set_aspect("equal")
            a.set_xlim(x[0], x[-1]); a.set_ylim(y[0], y[-1])
            if i == 0:
                a.set_title(rf"$t = {t:.3g}$", fontsize=11)
            if i == n - 1:
                a.set_xlabel(r"$x$", fontsize=11)
            if j == 0:
                a.set_ylabel(rf"$\alpha = {c['aoa']:g}^\circ$" + "\n" + r"$y$",
                             fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.965, bottom=0.05,
                        wspace=0.04, hspace=0.05)
    cb = fig.colorbar(im, ax=[ax for row in axs for ax in row],
                      shrink=0.75, pad=0.012, extend="both")
    cb.set_label(r"$C_p$", fontsize=12)
    out = os.path.join(images_dir(tag), f"{PREFIX}_fields.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05); plt.close()
    print("wrote", out)

if __name__ == "__main__":
    _work = sys.argv[1] if len(sys.argv) > 1 else WORK
    analyze_workdir(_work)
    plot_fields(_work)
