"""
NACA 0012 lift/drag validation for Hydro2 (embedded-solid, single-gas compressible).

Sweeps angle of attack, runs the solver, and computes Cl, Cd, L/D TWO independent
ways from each plotfile:
  (A) pressure surface integral   F = -∫ (p-p_inf) ∇phi dV        (pressure/form force)
  (B) Brinkman penalty reaction   F =  ∫ brinkman (1-phi)(M-Ms) dV (force the wall applies)
For a well-resolved inviscid body these should AGREE.  Both are compared against:
  - analytical thin-airfoil + Prandtl-Glauert:  Cl = (2*pi/beta) * alpha,  beta=sqrt(1-M^2)
  - tabulated NACA 0012 experiment (Abbott & von Doenhoff / NASA Ladson TM-4074)

NOTE (honest): the run is INVISCID (mu=0).  So:
  - Cl should match theory/experiment in the linear (pre-stall) range.
  - Cd is PRESSURE/wave drag only; subcritical-subsonic inviscid Cd ~ 0 (d'Alembert),
    so it will sit FAR below the viscous experimental Cd (~0.006). That gap quantifies
    the missing skin friction -- a known limitation, not a solver bug.

usage:
  python3 analyze_NACA_0012.py gen           # write the default BMP + standalone input (AoA=0)
  python3 analyze_NACA_0012.py run <AoA>     # single case
  python3 analyze_NACA_0012.py sweep         # full AoA sweep + plot
"""
import os, sys, glob, math, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(HERE, "Images")             # all plots go here; reference/ stays scripts-only
os.makedirs(IMAGES, exist_ok=True)
TESTDIR = os.path.normpath(os.path.join(HERE, "..", "NACA_0012"))
FLAMES = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
SOLVER = os.path.join(FLAMES, "bin", "hydro2-2d-g++")
WORK = "/tmp/naca0012"

# ---- flow + numerics ----
GAMMA = 1.4
MACH = 0.5
RHO, P = 1.0, 1.0 / GAMMA          # c = sqrt(gamma p/rho) = 1  ->  U = Mach
U = MACH
CHORD = 1.0
DOM_LO, DOM_HI = (-4.0, -5.0), (8.0, 5.0)
N_CELL = (192, 160)                # base halved-cell (was 96x80); divisible by 8
MAX_LEVEL = 3                      # +1 refinement level (was 2)
NX_BMP, NY_BMP = 1600, 1334
EPS = 0.01                         # /4 (was 0.04); finest dx 0.0078 -> band ~2-3 cells
BRINK = 4 * U / EPS
SLIP = 0                           # 0 = no-slip Brinkman (default); 1 = slip (normal-only)
REFINE_BOX = None                  # ((xlo,ylo),(xhi,yhi)) geometric box -> max_level inside (TE targeting)
P_INF = P
STOP_TIME = 16.0                   # subsonic settling (slow flow)
PLOT_DT = 0                        # 0 -> single plot at stop_time/2; else write every PLOT_DT (Cl history)
AOAS = [0, 2, 4, 6, 8, 10]
Q_INF = 0.5 * RHO * U ** 2
BETA_PG = math.sqrt(1 - MACH ** 2)

# tabulated NACA 0012 experiment (low-speed, Re~6e6; Abbott & von Doenhoff / Ladson)
EXP_LIFT_SLOPE = 0.11              # dCl/dAoA per deg (linear range)
EXP_CD0 = 0.0065                   # min profile drag (viscous)

# ---------- geometry ----------
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
    x = np.linspace(DOM_LO[0], DOM_HI[0], NX_BMP); y = np.linspace(DOM_LO[1], DOM_HI[1], NY_BMP)
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

# ---------- input ----------
def write_input(bmp, plot, inp):
    mom_x = RHO * U; e_bc = 0.5 * P / (GAMMA - 1)
    q_inf = 0.5 * RHO * U ** 2
    base_dx = (DOM_HI[0] - DOM_LO[0]) / N_CELL[0]
    wall = "no-slip (all momentum)" if not SLIP else "slip (wall-normal only)"
    plotc = f"amr.plot_dt          = {PLOT_DT}" if PLOT_DT else f"amr.plot_dt          = {STOP_TIME/2.0}"
    rbox = ""
    if REFINE_BOX is not None:
        (xl, yl), (xh, yh) = REFINE_BOX
        rbox = (f"refine_box.lo            = {xl:.4f} {yl:.4f}     # force max_level in a box (trailing-edge targeting)\n"
                f"refine_box.hi            = {xh:.4f} {yh:.4f}\n")
    txt = f"""#@ [naca0012]
#@ exe=hydro2
#@ dim=2

# =============================================================================
# FlowAirfoil -- NACA 0012, 2D subsonic flow over an EMBEDDED SOLID airfoil
# (rasterized phi bitmap -> diffuse tanh boundary), for lift/drag VALIDATION
# against thin-airfoil theory and tabulated NACA data.
# =============================================================================
# Condition: Ma {MACH} (c = 1 by construction: p = 1/gamma, rho = 1 -> a = 1, U = Ma).
#
#   FLOW:      rho = {RHO},  p = {P:.6f} (= 1/gamma),  U = {U}  (Ma {MACH}).
#              rho*U = {mom_x:.4f}.   E_per_phase = 0.5*p/(gamma-1) = {e_bc:.6f}.
#              q_inf = 0.5*rho*U^2 = {q_inf:.6f}.
#   GEOMETRY:  NACA 0012, chord {CHORD}, from a phi bitmap (fit=coord over the
#              domain box).  d>0 fluid, d<0 solid -> phi = 0.5*(1+tanh(d/eps)).
#   WALL:      {wall} Brinkman penalty (solid.slip={SLIP}).
#              eps = {EPS} skin,  brinkman = {BRINK:.1f} ~ 4*U/eps.
#   LIFT/DRAG (post-processed):  F = -integral (p-p_inf) grad(phi) dV;
#              C_l = F_y/(q_inf*c),  C_d = F_x/(q_inf*c).  INVISCID (mu=0) ->
#              pressure/form drag only (no skin friction).  Two identical ideal
#              gases (eta=0.5) => the mixture is one perfect-gas air.
#   BCs:       SUBSONIC -- Dirichlet freestream inflow at xlo, pinned back-pressure
#              at xhi (energy Dirichlet); far-field top/bottom Neumann.
# =============================================================================

alamo.program = hydro2

### OUTPUT ###
plot_file            = {plot}
{plotc}
amr.plot_int         = -1

### MESHING ###
# Base dx = {base_dx:.4f}; AMR refines the airfoil surface (grad phi) + shock.
# max_level = {MAX_LEVEL} -> dx_finest = base/2^{MAX_LEVEL}; eps = {EPS} diffuse skin.
amr.max_grid_size    = 500000
amr.blocking_factor  = 8
amr.regrid_int       = 10
amr.grid_eff         = 0.8
amr.max_level        = {MAX_LEVEL}
amr.n_cell           = {N_CELL[0]} {N_CELL[1]}     # both divisible by blocking_factor 8

### DIMENSIONS -- flow in +X ###
geometry.prob_lo     = {DOM_LO[0]} {DOM_LO[1]} 0.0
geometry.prob_hi     = {DOM_HI[0]} {DOM_HI[1]} 0.0
geometry.is_periodic = 0 0 0

### TIME STEPPING ###
timestep            = 1e-6
dynamictimestep.on  = 1
dynamictimestep.max = 5e-3
dynamictimestep.min = 1e-10
cfl                 = 0.3
cfl_v               = 0.3
stop_time           = {STOP_TIME}

# ----------------------------------------------------------------------------
# EMBEDDED SOLID -- the NACA 0012, from a phi bitmap.  Yang Brinkman wall.
# ----------------------------------------------------------------------------
apply_embedded_solid     = 1
solid.brinkman           = {BRINK}          # YANG penalty ~ 4*U/eps ({wall})
solid.slip               = {SLIP}              # 0 = no-slip (all momentum); 1 = slip (wall-normal only)
{rbox}phi_refinement_criterion = 0.1            # refine on the airfoil surface (grad phi)

# Bitmap geometry IC:  pixel/255 -> phi,  fit=coord onto the domain box.
solid.phi.ic.type            = bmp
solid.phi.ic.bmp.filename    = {bmp}
solid.phi.ic.bmp.channel     = g
solid.phi.ic.bmp.min         = 0
solid.phi.ic.bmp.max         = 255
solid.phi.ic.bmp.fit         = coord
solid.phi.ic.bmp.coord.lo    = {DOM_LO[0]} {DOM_LO[1]}
solid.phi.ic.bmp.coord.hi    = {DOM_HI[0]} {DOM_HI[1]}

# Prescribed quiescent solid target (air at rest, p = p_inf):
solid.density.ic.type            = constant
solid.density.ic.constant.value  = {RHO}
solid.momentum.ic.type           = constant
solid.momentum.ic.constant.value = 0.0 0.0
solid.pressure.ic.type           = constant
solid.pressure.ic.constant.value = {P:.6f}

### ETA (alpha_1) -- uniform 0.5 (two identical gases => single ideal-gas air) ###
eta.ic.type       = constant
eta.ic.constant.value = 0.5
eta.bc.type.xlo   = dirichlet          # subsonic inflow
eta.bc.val.xlo    = 0.5
eta.bc.type.xhi   = neumann
eta.bc.type.ylo   = neumann
eta.bc.type.yhi   = neumann

### EOS (ideal-gas air, gamma = {GAMMA}) -- two identical phases ###
eos0.gamma = {GAMMA}
eos0.p0    = 0.0
eos0.cp    = 1005.0
eos0.cv    = 717.86
eos1.gamma = {GAMMA}
eos1.p0    = 0.0
eos1.cp    = 1005.0
eos1.cv    = 717.86
mu0   = 0.0
mu0_b = 0.0
mu1   = 0.0
mu1_b = 0.0

### FLUID 0 IC (subsonic freestream in +X) ###
density0.ic.type = constant
density0.ic.constant.value = {RHO}
velocity0.ic.type = expression
velocity0.ic.expression.region0 = "{U}"      # vx = U
velocity0.ic.expression.region1 = "0.0"      # vy
pressure0.ic.type = constant
pressure0.ic.constant.value = {P:.6f}

### FLUID 1 IC (identical to fluid 0) ###
density1.ic.type = constant
density1.ic.constant.value = {RHO}
velocity1.ic.type = expression
velocity1.ic.expression.region0 = "{U}"
velocity1.ic.expression.region1 = "0.0"
pressure1.ic.type = constant
pressure1.ic.constant.value = {P:.6f}

### INTERACTIONS ###
epsilon = 0.1
sigma   = 0.0
Dv      = 0.0
pref    = 0.0
small   = 1e-6
grav    = 0.0

### BOUNDARY CONDITIONS ###
# SUBSONIC: Dirichlet freestream inflow at xlo; pinned back-pressure at xhi
# (energy Dirichlet, so the domain pressure does not float); far-field Neumann.
# density.bc = mixture inflow density; energy.bc = PER-PHASE E0=E1=0.5*p/(gamma-1).
density.bc.type.xlo  = dirichlet
density.bc.val.xlo   = {RHO}
density.bc.type.xhi  = neumann
density.bc.type.ylo  = neumann
density.bc.type.yhi  = neumann

momentum.bc.type.xlo = dirichlet dirichlet     # Mx = rho*U, My = 0
momentum.bc.val.xlo  = {mom_x} 0.0
momentum.bc.type.xhi = neumann neumann
momentum.bc.type.ylo = neumann neumann
momentum.bc.type.yhi = neumann neumann

energy.bc.type.xlo   = dirichlet
energy.bc.val.xlo    = {e_bc:.6f}
energy.bc.type.xhi   = dirichlet          # subsonic outflow: pin back-pressure
energy.bc.val.xhi    = {e_bc:.6f}
energy.bc.type.ylo   = neumann
energy.bc.type.yhi   = neumann

### REFINEMENT CRITERIA -- shock (rho/p gradients) + airfoil (grad phi) ###
eta_refinement_criterion   = 1e10
omega_refinement_criterion = 1e10
gradu_refinement_criterion = 1e10
p_refinement_criterion     = 0.1
rho_refinement_criterion   = 0.1
cutoff = 1e-16

### SOURCE TERMS ###
m0.ic.constant.value = 0.0
u0.ic.constant.value = 0.0 0.0
q.ic.constant.value  = 0.0 0.0

apply_surface_tension = 0
apply_weight          = 0
apply_vaporization    = 0

### SOLVER ###
Riemann_Solver.type = hllc
Limiter.type        = minmod
kappa_method        = 1
apply_sharpening    = 0
"""
    open(inp, "w").write(txt)

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

def forces_pressure(plot):
    g, dx, dy = _grids(plot)
    phi = g("phi"); p = g("pressure")
    gx = np.gradient(phi, dx, axis=0); gy = np.gradient(phi, dy, axis=1)
    pp = p - P_INF
    Fx = -np.sum(pp * gx) * dx * dy; Fy = -np.sum(pp * gy) * dx * dy
    return Fx, Fy

def forces_penalty(plot):
    g, dx, dy = _grids(plot)
    phi = g("phi"); Mx = g("momentumx"); My = g("momentumy")
    # solid momentum ~ 0 (static).  F_on_solid = ∫ brinkman (1-phi)(M - Ms) dV
    Fx = np.sum(BRINK * (1 - phi) * Mx) * dx * dy
    Fy = np.sum(BRINK * (1 - phi) * My) * dx * dy
    return Fx, Fy

# ---------- driver ----------
def run_case(aoa, tag=None):
    rd = os.path.join(WORK, tag or f"a{aoa:05.1f}"); os.makedirs(rd, exist_ok=True)
    bmp = os.path.join(rd, "phi.bmp"); plot = os.path.join(rd, "out"); inp = os.path.join(rd, "input")
    write_phi_bmp(rotate(naca0012(), aoa), bmp)
    write_input(bmp, plot, inp)
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(os.path.join(rd, "log"), "w") as lf:
        r = subprocess.run([os.path.abspath(SOLVER), os.path.abspath(inp)], cwd=rd, env=env,
                           stdout=lf, stderr=subprocess.STDOUT, timeout=5400)
    if r.returncode != 0 or not glob.glob(plot + "/*cell"):
        return None
    (Fxp, Fyp), (Fxk, Fyk) = forces_pressure(plot), forces_penalty(plot)
    return dict(aoa=aoa,
                Cl_p=Fyp/(Q_INF*CHORD), Cd_p=Fxp/(Q_INF*CHORD),
                Cl_k=Fyk/(Q_INF*CHORD), Cd_k=Fxk/(Q_INF*CHORD), plot=plot)

def sweep():
    os.makedirs(WORK, exist_ok=True)
    rows = []
    for a in AOAS:
        r = run_case(a)
        if r is None:
            print(f"AoA={a}: FAILED"); continue
        rows.append(r)
        print(f"AoA={a:2d}: Cl(press)={r['Cl_p']:+.4f} Cl(pen)={r['Cl_k']:+.4f} "
              f"(theory {2*math.pi/BETA_PG*math.radians(a):+.4f})  "
              f"Cd(press)={r['Cd_p']:.4f} Cd(pen)={r['Cd_k']:.4f}")
    plot_results(rows)

def plot_results(rows):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    A = np.array([r["aoa"] for r in rows]); ar = np.radians(A)
    Clp = np.array([r["Cl_p"] for r in rows]); Clk = np.array([r["Cl_k"] for r in rows])
    Cdp = np.array([r["Cd_p"] for r in rows]); Cdk = np.array([r["Cd_k"] for r in rows])
    Cl_th = 2*np.pi/BETA_PG * ar
    Cl_exp = EXP_LIFT_SLOPE * A
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    ax[0].plot(A, Clp, "o-", label="CFD: pressure ∫p∇φ")
    ax[0].plot(A, Clk, "s--", label="CFD: Brinkman reaction")
    ax[0].plot(A, Cl_th, "r-", label=f"thin-airfoil 2πα/β (M={MACH})")
    ax[0].plot(A, Cl_exp, "k:", label="NACA exp (0.11/deg)")
    ax[0].set_title("Lift  Cl"); ax[0].set_ylabel("Cl")
    ax[1].plot(A, Cdp, "o-", label="CFD: pressure")
    ax[1].plot(A, Cdk, "s--", label="CFD: Brinkman")
    ax[1].axhline(EXP_CD0, color="k", ls=":", label=f"NACA exp Cd0≈{EXP_CD0} (viscous)")
    ax[1].set_title("Drag  Cd (inviscid → no skin friction)"); ax[1].set_ylabel("Cd")
    with np.errstate(divide="ignore", invalid="ignore"):
        ax[2].plot(A, Clp/Cdp, "o-", label="CFD: pressure")
        ax[2].plot(A, Clk/Cdk, "s--", label="CFD: Brinkman")
    ax[2].set_title("L/D"); ax[2].set_ylabel("L/D")
    for a in ax:
        a.set_xlabel("angle of attack (deg)"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(f"NACA 0012 @ M={MACH} (inviscid) — CFD (2 force methods) vs theory vs NACA experiment")
    out = os.path.join(IMAGES, "naca0012_polars.png")
    plt.tight_layout(); plt.savefig(out, dpi=110); print("wrote", out)
    if len(A) > 1:
        sl_p = np.polyfit(ar, Clp, 1)[0]; sl_k = np.polyfit(ar, Clk, 1)[0]
        print(f"\nlift slope dCl/dα:  pressure={sl_p:.2f}/rad  Brinkman={sl_k:.2f}/rad  "
              f"theory 2π/β={2*np.pi/BETA_PG:.2f}/rad  exp≈{math.degrees(EXP_LIFT_SLOPE):.2f}/rad")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if cmd == "gen":
        os.makedirs(TESTDIR, exist_ok=True)
        REL = "./tests/FlowAirfoil/NACA_0012"     # relative to the flames run dir (portable, like input_Ma1.1)
        write_phi_bmp(naca0012(), os.path.join(TESTDIR, "naca0012.bmp"))
        write_input(f"{REL}/naca0012.bmp", f"{REL}/output", os.path.join(TESTDIR, "input"))
        print("wrote", TESTDIR, "/{naca0012.bmp,input}")
    elif cmd == "run":
        if len(sys.argv) > 3 and sys.argv[3] == "slip":
            SLIP = 1; print("SLIP wall enabled")
        r = run_case(float(sys.argv[2]))
        print(r if r is None else
              f"AoA={r['aoa']}: Cl_press={r['Cl_p']:+.4f} Cl_pen={r['Cl_k']:+.4f} "
              f"Cd_press={r['Cd_p']:.4f} Cd_pen={r['Cd_k']:.4f}")
    else:
        sweep()
