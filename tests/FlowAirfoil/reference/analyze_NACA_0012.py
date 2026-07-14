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

Forces, TWO independent ways per plotfile:
  (A) pressure surface integral   F = -integral (p - p_inf) grad(phi) dV
  (B) Brinkman penalty reaction   F =  integral brinkman (1-phi)(M - Ms) dV
      (only when the case ran an explicit constant solid.brinkman > 0;
       skipped for the AUTO wavespeed-scaled penalty, which is per-cell)
Compared against thin-airfoil + Prandtl-Glauert (Cl = 2 pi alpha / beta) and
tabulated NACA 0012 experiment (Abbott & von Doenhoff / NASA Ladson TM-4074).

usage:
  python3 analyze_NACA_0012.py [workdir]      # default /tmp/naca_overnight

The geometry helpers (naca0012 / rotate / write_phi_bmp) live here as a small
shared library; overnight_study.py imports them to generate per-case bitmaps.
"""
import os, sys, glob, math, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "Images")             # all plots go here; reference/ stays scripts-only
os.makedirs(IMAGES, exist_ok=True)
TESTDIR = os.path.normpath(os.path.join(HERE, "..", "NACA_0012"))
FLAMES = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SOLVER = os.path.join(FLAMES, "bin", "hydro2-2d-g++")
WORK = "/tmp/naca_overnight"                      # default sweep work dir

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
def naca0012(n=80):
    xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))      # cosine spacing 0..1
    yt = 0.6 * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1015*xc**4)
    x = (xc - 0.5) * CHORD
    up = np.column_stack([x, yt*CHORD]); lo = np.column_stack([x[::-1], -yt[::-1]*CHORD])
    return np.vstack([up, lo])

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
    return dict(rho=rho, U=u, p_inf=p, gamma=gam, mu=mu, brinkman=brk, penalty_power=ppw,
                mach=u / c, q_inf=0.5 * rho * u * u,
                re=(rho * u * CHORD / mu) if mu > 0 else float("inf"))

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
def _grids(plot):
    import yt
    pf = sorted(glob.glob(os.path.join(plot, "*cell")))[-1]
    ds = yt.load(pf); L = ds.index.max_level
    dims = (ds.domain_dimensions * ds.refine_by ** L).astype(int)
    cg = ds.covering_grid(level=L, left_edge=ds.domain_left_edge, dims=dims)
    dx = np.asarray(ds.domain_width) / dims
    g = lambda f: np.asarray(cg[f])[:, :, 0]
    return g, float(dx[0]), float(dx[1])

def forces_pressure(plot, p_inf=P_INF):
    g, dx, dy = _grids(plot)
    phi = g("phi"); p = g("pressure")
    gx = np.gradient(phi, dx, axis=0); gy = np.gradient(phi, dy, axis=1)
    pp = p - p_inf
    Fx = -np.sum(pp * gx) * dx * dy; Fy = -np.sum(pp * gy) * dx * dy
    return Fx, Fy

def forces_penalty(plot, brinkman, penalty_power=1.0):
    """Brinkman reaction force.  Only valid for an explicit CONSTANT penalty
    (brinkman > 0); the AUTO penalty is per-cell wavespeed-scaled and cannot
    be reconstructed from the plotfile -> return None.  The weight matches the
    solver: lambda * (1-phi)^penalty_power."""
    if brinkman is None or brinkman <= 0.0:
        return None
    g, dx, dy = _grids(plot)
    phi = g("phi"); Mx = g("momentumx"); My = g("momentumy")
    w = brinkman * (1 - phi) ** penalty_power
    Fx = np.sum(w * Mx) * dx * dy
    Fy = np.sum(w * My) * dx * dy
    return Fx, Fy

# ---------- analysis ----------
def analyze_case(case):
    cfg = parse_case_input(case["inp"])
    qc = cfg["q_inf"] * CHORD
    Fxp, Fyp = forces_pressure(case["plot"], cfg["p_inf"])
    row = dict(aoa=case["aoa"], rd=case["rd"], cfg=cfg,
               Cl_p=Fyp / qc, Cd_p=Fxp / qc, Cl_k=None, Cd_k=None)
    pen = forces_penalty(case["plot"], cfg["brinkman"], cfg.get("penalty_power", 1.0))
    if pen is not None:
        row["Cl_k"] = pen[1] / qc; row["Cd_k"] = pen[0] / qc
    return row

def analyze_workdir(work=WORK, quiet=False):
    cases = find_cases(work)
    if not cases:
        if not quiet:
            print(f"no analyzable cases under {work} (need <dir with AoA in name>/input + out/*cell)")
        return []
    rows = []
    for c in cases:
        try:
            r = analyze_case(c)
        except Exception as e:
            if not quiet:
                print(f"  {os.path.basename(c['rd'])}: analysis FAILED ({e!r})")
            continue
        rows.append(r)
        if not quiet:
            beta = math.sqrt(max(1e-12, 1 - r["cfg"]["mach"] ** 2))
            th = 2 * math.pi / beta * math.radians(r["aoa"])
            pen = "" if r["Cl_k"] is None else f" Cl(pen)={r['Cl_k']:+.4f} Cd(pen)={r['Cd_k']:.4f}"
            print(f"  AoA={r['aoa']:5.1f}: Cl={r['Cl_p']:+.4f} (theory {th:+.4f})  Cd={r['Cd_p']:.4f}{pen}")
    if rows:
        plot_results(rows)
    return rows

def plot_results(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["aoa"])
    cfg = rows[0]["cfg"]                       # sweep shares one flow condition
    mach = cfg["mach"]; beta = math.sqrt(max(1e-12, 1 - mach ** 2))
    A = np.array([r["aoa"] for r in rows]); ar = np.radians(A)
    Clp = np.array([r["Cl_p"] for r in rows]); Cdp = np.array([r["Cd_p"] for r in rows])
    have_pen = all(r["Cl_k"] is not None for r in rows)
    Cl_th = 2 * np.pi / beta * ar
    Cl_exp = EXP_LIFT_SLOPE * A
    re_str = "inviscid" if not np.isfinite(cfg["re"]) else f"Re={cfg['re']:.0f}"

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    ax[0].plot(A, Clp, "o-", label="CFD: pressure ∫p∇φ")
    if have_pen:
        ax[0].plot(A, [r["Cl_k"] for r in rows], "s--", label="CFD: Brinkman reaction")
    ax[0].plot(A, Cl_th, "r-", label=f"thin-airfoil 2πα/β (M={mach:.2f})")
    ax[0].plot(A, Cl_exp, "k:", label="NACA exp (0.11/deg)")
    ax[0].set_title("Lift  Cl"); ax[0].set_ylabel("Cl")
    ax[1].plot(A, Cdp, "o-", label="CFD: pressure")
    if have_pen:
        ax[1].plot(A, [r["Cd_k"] for r in rows], "s--", label="CFD: Brinkman")
    ax[1].axhline(EXP_CD0, color="k", ls=":", label=f"NACA exp Cd0≈{EXP_CD0}")
    ax[1].set_title(f"Drag  Cd ({re_str})"); ax[1].set_ylabel("Cd")
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[2].plot(A, Clp / np.where(Cdp == 0, np.nan, Cdp), "o-", label="CFD: pressure")
    ax[2].set_title("L/D"); ax[2].set_ylabel("L/D")
    for a in ax:
        a.set_xlabel("angle of attack (deg)"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"NACA 0012 @ M={mach:.2f} {re_str} — CFD vs theory vs NACA experiment "
                 f"({len(rows)} cases, AoA auto-detected)")
    out = os.path.join(IMAGES, "naca0012_polars.png")
    plt.tight_layout(); plt.savefig(out, dpi=110); plt.close()
    print("wrote", out)
    if len(A) > 1:
        # fit the lift slope on the pre-stall linear range only
        lin = A <= 10.0
        if lin.sum() > 1:
            sl_p = np.polyfit(ar[lin], Clp[lin], 1)[0]
            print(f"lift slope dCl/dα (α≤10°): CFD={sl_p:.2f}/rad   "
                  f"theory 2π/β={2*np.pi/beta:.2f}/rad   "
                  f"exp≈{math.degrees(EXP_LIFT_SLOPE):.2f}/rad")

if __name__ == "__main__":
    analyze_workdir(sys.argv[1] if len(sys.argv) > 1 else WORK)
