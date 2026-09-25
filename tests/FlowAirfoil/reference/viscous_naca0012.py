#!/usr/bin/env python3
"""
Viscous (laminar) NACA 0012 validation decks for hydro2's embedded solid.

Standard laminar benchmarks (GAMM workshop / Bristeau et al. 1987 and the many
follow-ups, e.g. Swanson & Langer; Venkatakrishnan):
    A:  M = 0.5, Re = 5000, AoA =  0 deg -> steady; laminar TE separation at x/c ~ 0.81,
                                            Cl = 0, Cd ~ 0.055 (pressure + friction)
    B:  M = 0.8, Re =  500, AoA = 10 deg -> steady, large separated region, Cl ~ 0.44 (check vs your papers)

Differences from the inviscid deck (analyze_NACA_0012.py) and why:
  * SHARP wall: phi = 0.5(1+tanh(d/eps)) with eps = 1e-6, i.e. a step.  The old
    eps = 0.02 made the aft 20% of the chord porous (min phi 0.25-0.5 there).
    The bitmap is CROPPED to the airfoil and much finer than the finest cell
    (outside the crop the BMP IC clamps to the edge pixel = fluid).
  * NO-SLIP Brinkman, solid.implicit = 1 (exact exponential update), brinkman
    = 1e6: penalization layer sqrt(nu/brinkman) ~ 1e-5 << dx.  The explicit
    source could not exceed ~150 on the coarse level.
  * solid.wall_flux = 1: fluid/solid faces solved as a wall (mirror state).  Without it
    the porous full-flux wall traps mass at the LE stagnation point and the M 0.5 / Re 5000
    run went singular at t = 5.6 (LE solid-cell rho 1 -> 1.74 by t = 4).
  * NSCBC4 (nghost 4) far-field: subsonic inflow at xlo, non-reflecting
    outflow with target p on xhi/ylo/yhi (the old deck fixed all 4 variables at
    the inflow and pinned p at the outlet -> acoustically closed box).
  * Bigger domain, refinement box over the body + near wake.
  * Closed trailing edge (last coefficient -0.1036) -- the usual CFD choice.
  * thermal.conduction = 1, Pr = 0.72, adiabatic wall (as in the GAMM runs).

Usage:  python3 viscous_naca0012.py [A|B] [max_level] [dir_suffix]   (writes into bin/tests/FlowAirfoil/Viscous/)
"""
import math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FLAMES = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
OUTROOT = os.path.join(FLAMES, "bin", "tests", "FlowAirfoil", "Viscous")

GAMMA, RHO, CV = 1.4, 1.0, 717.86
P = 1.0 / GAMMA                      # a = 1
T_INF = P / (RHO * CV * (GAMMA - 1.0))
CHORD = 1.0
EPS = 1.0e-6                         # sharp wall (step)
BRINK = 1.0e6
DOM_LO, DOM_HI = (-5.0, -6.0), (11.0, 6.0)
N_CELL = (256, 192)                  # dx0 = 0.0625
BMP_LO, BMP_HI = (-0.62, -0.22), (0.62, 0.22)
BMP_DX = 0.0005                      # pixel << finest cell (L4 0.0039, L5 0.00195)
RBOX = ((-0.65, -0.20), (1.20, 0.20))  # body + near wake at max level

CASES = {
    "A": dict(mach=0.5, re=5000.0, aoa=0.0,  stop=40.0, plot_dt=2.0),
    "B": dict(mach=0.8, re=500.0,  aoa=10.0, stop=25.0, plot_dt=1.25),
}

def naca0012(n=400):
    xc = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    yt = 0.6 * (0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1036*xc**4)  # closed TE
    x = (xc - 0.5) * CHORD
    up = np.column_stack([x, yt*CHORD]); lo = np.column_stack([x[::-1], -yt[::-1]*CHORD])
    return np.vstack([up, lo[1:-1]])

def rotate(poly, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return poly @ np.array([[c, s], [-s, c]]).T            # +deg = nose-up

def write_phi_bmp(poly, path):
    from matplotlib.path import Path
    from PIL import Image
    nx = int(round((BMP_HI[0]-BMP_LO[0]) / BMP_DX)); ny = int(round((BMP_HI[1]-BMP_LO[1]) / BMP_DX))
    # IC::BMP maps [lo, hi] onto pixel coordinates [0, N-1] (img_width = nx-1,
    # img_height = ny-1), i.e. pixel I sits at x = lo + I*(hi-lo)/(N-1): an
    # INCLUSIVE linspace.  [2026-09-24 fix: an earlier version of this script
    # used lo + I*(hi-lo)/N, which stretched the image by N/(N-1) relative to the
    # solver -- the body appeared shifted up by (y-lo)/(N-1), ~0.1-0.16 of an L5
    # cell, and the staircase came out mirror-ASYMMETRIC (279 L5 cells), giving
    # spurious lift at AoA = 0.]
    if os.environ.get("OLD_PIXEL_MAP", "0") == "1":   # reproduce the buggy mapping (A/B test only)
        x = BMP_LO[0] + np.arange(nx) * (BMP_HI[0]-BMP_LO[0]) / nx
        y = BMP_LO[1] + np.arange(ny) * (BMP_HI[1]-BMP_LO[1]) / ny
    else:
        x = np.linspace(BMP_LO[0], BMP_HI[0], nx)
        y = np.linspace(BMP_LO[1], BMP_HI[1], ny)
    X, Y = np.meshgrid(x, y)
    inside = Path(poly).contains_points(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    # eps = 1e-6 -> tanh(d/eps) is a step at any pixel farther than ~1e-5 from the surface,
    # so phi is 0 (inside) / 1 (outside); the distance field is not needed.
    phi = np.where(inside, 0.0, 1.0)
    g = np.clip(np.round(255*np.flipud(phi)), 0, 255).astype(np.uint8)
    Image.fromarray(np.dstack([g, g, g]), "RGB").save(path)
    return nx, ny

def write_input(case, lvl, d):
    c = CASES[case]; U = c["mach"]; mu = RHO * U * CHORD / c["re"]
    E_ph = 0.5 * P / (GAMMA - 1.0)
    dxf = (DOM_HI[0]-DOM_LO[0]) / N_CELL[0] / 2**lvl
    # pressure probes in the body frame, rotated with the AoA: ahead of the LE,
    # over the upper surface at mid-chord, just behind the TE, and in the wake
    # AoA = 0: refine ONLY by the (mirror-symmetric) refine_box so every level's
    # grid is fixed and symmetric.  AMReX clustering is not mirror-symmetric, and
    # gradient / phi tags outside the box produced unmirrored L2/L4 patches
    # (278 / 992 cells) -> asymmetric coarse-fine boundaries -> spurious lift.
    sym = (c["aoa"] == 0.0) and os.environ.get("ASYM_GRID", "0") != "1"
    phi_crit = "1e10" if sym else "0.1"
    grad_crit = "1e10" if sym else "0.2"
    probes = rotate(np.array([[-0.52, 0.0], [0.0, 0.075], [0.52, 0.0], [1.0, 0.0]]), c["aoa"])
    txt = f"""#@ [naca0012_viscous_{case}]
#@ exe=hydro2
#@ dim=2
# Laminar NACA 0012, case {case}: M = {U}, Re = {c['re']:.0f}, AoA = {c['aoa']} deg
# (generated by tests/FlowAirfoil/reference/viscous_naca0012.py -- edit there).
# mu = rho U c / Re = {mu:.4e}.  a = 1 (p = 1/gamma, rho = 1).  T_inf = {T_INF:.6e}.
# Finest dx = {dxf:.5f} (max_level {lvl}); sharp wall eps = {EPS}; implicit Brinkman {BRINK:.0e}.
alamo.program = hydro2
plot_file            = {os.path.join(d, 'out')}
amr.plot_dt          = {c['plot_dt']}
amr.plot_int         = -1

amr.max_grid_size    = 64
amr.blocking_factor  = 8
amr.regrid_int       = 10
amr.grid_eff         = 0.8
amr.n_error_buf      = 4
amr.max_level        = {lvl}
amr.n_cell           = {N_CELL[0]} {N_CELL[1]}
geometry.prob_lo     = {DOM_LO[0]} {DOM_LO[1]} 0.0
geometry.prob_hi     = {DOM_HI[0]} {DOM_HI[1]} 0.0
geometry.is_periodic = 0 0 0
refine_box.lo        = {RBOX[0][0]} {RBOX[0][1]}
refine_box.hi        = {RBOX[1][0]} {RBOX[1][1]}

timestep            = 1e-6
dynamictimestep.on  = 1
dynamictimestep.max = 2e-2
dynamictimestep.min = 1e-10
cfl                 = 0.3
cfl_v               = 0.3
stop_time           = {c['stop']}
nghost              = 4
relax_diag          = 0

# ---- embedded solid: sharp no-slip wall, implicit Brinkman ----
apply_embedded_solid     = 1
solid.brinkman           = {BRINK:.1e}
solid.implicit           = 1
solid.wall_flux          = 1      # sharp-wall Riemann flux (no mass trap at the LE stagnation point)
solid.force_int          = 1      # per-step force / probe history -> <plot_file>_forces.dat
solid.probe.x            = {' '.join('%.5f' % q[0] for q in probes)}
solid.probe.y            = {' '.join('%.5f' % q[1] for q in probes)}
solid.slip               = 0
phi_refinement_criterion = {phi_crit}
solid.phi.ic.type            = bmp
solid.phi.ic.bmp.filename    = {os.path.join(d, 'phi.bmp')}
solid.phi.ic.bmp.channel     = g
solid.phi.ic.bmp.min         = 0
solid.phi.ic.bmp.max         = 255
solid.phi.ic.bmp.fit         = coord
solid.phi.ic.bmp.coord.lo    = {BMP_LO[0]} {BMP_LO[1]}
solid.phi.ic.bmp.coord.hi    = {BMP_HI[0]} {BMP_HI[1]}
solid.density.ic.type            = constant
solid.density.ic.constant.value  = {RHO}
solid.momentum.ic.type           = constant
solid.momentum.ic.constant.value = 0.0 0.0
solid.pressure.ic.type           = constant
solid.pressure.ic.constant.value = {P:.6f}

# ---- single-phase air: eta = 1 (phase 0 only).  With eta = 0.5 every cell is
# "mixed" and the pressure-relaxation Newton runs everywhere; eta = 1 with
# cutoff 0.01 hits the pure-cell guard (same set-up as the TGV unit test).
eta.ic.type           = constant
eta.ic.constant.value = 1.0
eta.bc.type.xlo = neumann
eta.bc.type.xhi = neumann
eta.bc.type.ylo = neumann
eta.bc.type.yhi = neumann
eos0.gamma = {GAMMA}
eos0.p0    = 0.0
eos0.cp    = 1005.0
eos0.cv    = {CV}
eos1.gamma = {GAMMA}
eos1.p0    = 0.0
eos1.cp    = 1005.0
eos1.cv    = {CV}
mu0   = {mu:.6e}
mu1   = {mu:.6e}
mu0_b = 0.0
mu1_b = 0.0
# Fourier heat conduction, Pr = 0.72 (the GAMM laminar runs); adiabatic embedded wall.
# Without it the no-slip wall's dissipation heat cannot leave (T_wall -> 2.6 T_inf, crash).
thermal.conduction = 1
thermal.Pr         = 0.72

density0.ic.type = constant
density0.ic.constant.value = {RHO}
velocity0.ic.type = expression
velocity0.ic.expression.region0 = "{U}"
velocity0.ic.expression.region1 = "0.0"
pressure0.ic.type = constant
pressure0.ic.constant.value = {P:.6f}
density1.ic.type = constant
density1.ic.constant.value = {RHO}
velocity1.ic.type = expression
velocity1.ic.expression.region0 = "{U}"
velocity1.ic.expression.region1 = "0.0"
pressure1.ic.type = constant
pressure1.ic.constant.value = {P:.6f}

epsilon = 0.1
sigma   = 0.0
Dv      = 0.0
pref    = 0.0
small   = 1e-6
grav    = 0.0
cutoff  = 0.01

# ---- NSCBC4 far-field ----
density.bc.type.xlo  = nscbc_inflow
momentum.bc.type.xlo = nscbc_inflow nscbc_inflow
energy.bc.type.xlo   = nscbc_inflow
density.bc.type.xhi  = nscbc_outflow
momentum.bc.type.xhi = nscbc_outflow nscbc_outflow
energy.bc.type.xhi   = nscbc_outflow
density.bc.type.ylo  = nscbc_outflow
momentum.bc.type.ylo = nscbc_outflow nscbc_outflow
energy.bc.type.ylo   = nscbc_outflow
density.bc.type.yhi  = nscbc_outflow
momentum.bc.type.yhi = nscbc_outflow nscbc_outflow
energy.bc.type.yhi   = nscbc_outflow
density0.bc.type.xlo = neumann
density0.bc.type.xhi = neumann
density0.bc.type.ylo = neumann
density0.bc.type.yhi = neumann
density1.bc.type.xlo = neumann
density1.bc.type.xhi = neumann
density1.bc.type.ylo = neumann
density1.bc.type.yhi = neumann

nscbc.xlo.type     = inflow
nscbc.xlo.target_u = {U}
nscbc.xlo.target_v = 0.0
nscbc.xlo.target_T = {T_INF:.8e}
""" + "".join(f"""nscbc.{f}.type     = outflow
nscbc.{f}.target_p = {P:.6f}
nscbc.{f}.target_u = {U}
nscbc.{f}.target_v = 0.0
nscbc.{f}.sigma    = 0.5
nscbc.{f}.beta     = 0.5
""" for f in ("xhi", "ylo", "yhi")) + f"""nscbc.small = 1.0e-10

# ---- refinement: body (grad phi) + refine_box; shocks/strong gradients ----
eta_refinement_criterion   = 1e10
omega_refinement_criterion = 1e10
gradu_refinement_criterion = 1e10
p_refinement_criterion     = {grad_crit}
rho_refinement_criterion   = {grad_crit}

m0.ic.constant.value = 0.0
u0.ic.constant.value = 0.0 0.0
q.ic.constant.value  = 0.0 0.0
apply_surface_tension = 0
apply_weight          = 0
apply_vaporization    = 0

# SSPRK3.  NOTE: without these two lines alamo defaults to ForwardEuler
# (query_validate takes the first option) -- the old NACA deck ran 1st-order in time.
integration.type    = RungeKutta
integration.rk.type = 3
Riemann_Solver.type = hllc
Limiter.type        = vanleer
kappa_method        = 1
apply_sharpening    = 0
"""
    open(os.path.join(d, "input"), "w").write(txt)

if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "A"
    lvl = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    d = os.path.join(OUTROOT, f"case{case}_L{lvl}{tag}"); os.makedirs(d, exist_ok=True)
    nx, ny = write_phi_bmp(rotate(naca0012(), CASES[case]["aoa"]), os.path.join(d, "phi.bmp"))
    write_input(case, lvl, d)
    print(f"wrote {d}/input and phi.bmp ({nx}x{ny} px, {BMP_DX} px size)")
