#!/usr/bin/env python3
"""Generate the 3D angled Garrick Riemann unit-test inputs (multi-D coupling).

The Garrick gas-liquid Riemann problem is 1D along a propagation direction n;
its projected profile must be IDENTICAL for any orientation of n.  This sweeps
n over 4 planes x 5 angles = 20 oriented runs:

  plane   n(theta)                                   sweeps from -> to
  -----   ----------------------------------------   ------------------------
  XY      (cos t,  sin t,        0      )            +x  ->  +y
  YZ      ( 0,     cos t,        sin t  )            +y  ->  +z
  XZ      (cos t,  0,            sin t  )            +x  ->  +z
  XYZ     (cos t,  sin t/sqrt2,  sin t/sqrt2)        +x  ->  +(y+z) diagonal   (fully 3D)

theta in {0, 30, 45, 60, 90} deg.  All n are unit vectors, so the interface
s = n . r = nx*x + ny*y + nz*z = 0 is the signed normal distance.

Writes  ../input_Garrick_<PLANE>_<NN>   (e.g. input_Garrick_XYZ_45).
Run:  python tests/FlowRiemannUnitTests/Angled_3D/reference/gen_angled_3d_inputs.py
"""
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(_HERE, ".."))   # Angled_3D/

ANGLES = [0, 30, 45, 60, 90]
S2 = math.sqrt(2.0)
PLANES = {
    "XY":  lambda t: (math.cos(t), math.sin(t), 0.0),
    "YZ":  lambda t: (0.0, math.cos(t), math.sin(t)),
    "XZ":  lambda t: (math.cos(t), 0.0, math.sin(t)),
    "XYZ": lambda t: (math.cos(t), math.sin(t) / S2, math.sin(t) / S2),
}

TEMPLATE = """#@ [garrick3d_{plane_l}_{nn}]
#@ exe=hydro2
#@ dim=3

# Garrick et al. (2017) gas-liquid Riemann problem, ROTATED in 3D.
# Plane {PLANE}, theta = {ang} deg.  Propagation direction (unit):
#     n = ({nx:.12f}, {ny:.12f}, {nz:.12f})
# Multi-D coupling check: the 1D profile projected onto n must be identical for
# every orientation.  Overlay/error: reference/RiemannGarrickCompare_3D.py

alamo.program = hydro2

### OUTPUT ###
# plot_file = ./tests/FlowRiemannUnitTests/Angled_3D/output_Garrick_Angled_3D_{PLANE}_{nn}                                  # DESKTOP (run from bin/)
plot_file = /mmfs1/home/ttryon/flames/bin/tests/FlowRiemannUnitTests/Angled_3D/output_Garrick_Angled_3D_{PLANE}_{nn}   # INCLINE /mmfs1 (default)

### NUMERICS ###
Riemann_Solver.type = hllc
Limiter.type        = weno3
integration.type    = RungeKutta
integration.rk.type = 3
nghost = 4

### ROTATED INTERFACE: s = n . r = nx*x + ny*y + nz*z = 0 ###
eta.ic.type = expression
eta.ic.expression.constant.epsilon = "1.25e-3"
eta.ic.expression.constant.nx = "{nx:.12f}"
eta.ic.expression.constant.ny = "{ny:.12f}"
eta.ic.expression.constant.nz = "{nz:.12f}"
eta.ic.expression.region0 = "0.5 * (1.0 + tanh((nx*x + ny*y + nz*z) / epsilon))"
epsilon = eta.ic.expression.constant.epsilon

### MESHING (uniform; max_grid_size set for ~125 grids -> good MPI spread) ###
amr.plot_int        = 50
amr.max_level       = 0
amr.max_grid_size   = 40
amr.blocking_factor = 8
amr.grid_eff        = 0.8
amr.n_cell          = 200 200 200

### DIMENSIONS (unit cube) ###
geometry.prob_lo     = -1.0 -1.0 -1.0 # [ m ]
geometry.prob_hi     =  1.0  1.0  1.0 # [ m ]
geometry.is_periodic = 0 0 0

### TIME STEP (CFL-driven) ###
timestep            = 1.0e-12
dynamictimestep.on  = 1
dynamictimestep.max = 1.0e-3
dynamictimestep.min = 1.0e-9
cfl                 = 0.4
stop_time           = 0.2

### FORCED SOURCE (off) ###
m0.ic.constant.value = 0.0
u0.ic.constant.value = 0.0 0.0
q.ic.type = constant
q.ic.constant.value = 0.0 0.0

### EOS (stiffened-gas liquid + ideal gas) ###
eos0.gamma = 5.5
eos1.gamma = 1.4
eos0.p0 = 1.505
eos1.p0 = 0.0
eos0.cp = 1004.5
eos1.cp = 1004.5
eos0.cv = 717.5
eos1.cv = 717.5

mu0   = 0.0
mu1   = 0.0
sigma = 0.0
pref  = 0.0

### IC State - Fluid 0 (s < 0 side) - LIQUID ###
density0.ic.type = constant
density0.ic.constant.value = 0.991
velocity0.ic.type = constant
velocity0.ic.constant.value = 0.0 0.0 0.0
pressure0.ic.type = constant
pressure0.ic.constant.value = 3.059e-4
cp0 = 1.4
cv0 = 1.0

### IC State - Fluid 1 (s > 0 side) - GAS ###
density1.ic.type = constant
density1.ic.constant.value = 1.241
velocity1.ic.type = constant
velocity1.ic.constant.value = 0.0 0.0 0.0
pressure1.ic.type = constant
pressure1.ic.constant.value = 2.753
cp1 = 1.4
cv1 = 1.0

### BC - neumann on all 6 faces ###
density.bc.type.xlo = neumann
density.bc.type.xhi = neumann
density.bc.type.ylo = neumann
density.bc.type.yhi = neumann
density.bc.type.zlo = neumann
density.bc.type.zhi = neumann
energy.bc.type.xlo = neumann
energy.bc.type.xhi = neumann
energy.bc.type.ylo = neumann
energy.bc.type.yhi = neumann
energy.bc.type.zlo = neumann
energy.bc.type.zhi = neumann
momentum.bc.type.xlo = neumann neumann neumann
momentum.bc.type.xhi = neumann neumann neumann
momentum.bc.type.ylo = neumann neumann neumann
momentum.bc.type.yhi = neumann neumann neumann
momentum.bc.type.zlo = neumann neumann neumann
momentum.bc.type.zhi = neumann neumann neumann

### REFINEMENT (off -- uniform mesh) ###
eta_refinement_criterion   = 100000.1
omega_refinement_criterion = 100000.1
gradu_refinement_criterion = 100000.0
rho_refinement_criterion   = 100000.0
p_refinement_criterion     = 100000.0

kappa_method = 1
Dv = 0
g  = 0.0
apply_surface_tension = 0
apply_weight          = 0
apply_vaporization    = 0

apply_sharpening     = 0
sharpening_frequency = 100
reinit_max_iter      = 50
reinit_tolerance     = 1e-6
omega_relax          = 0.5
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    written = []
    for plane, nfun in PLANES.items():
        for ang in ANGLES:
            nx, ny, nz = nfun(math.radians(ang))
            nn = f"{ang:02d}"
            txt = TEMPLATE.format(plane=plane, PLANE=plane, plane_l=plane.lower(),
                                  ang=ang, nn=nn, nx=nx, ny=ny, nz=nz)
            name = f"input_Garrick_{plane}_{nn}"
            with open(os.path.join(OUT_DIR, name), "w", newline="\n") as f:
                f.write(txt)
            written.append(name)
    print(f"wrote {len(written)} 3D angled inputs to {OUT_DIR}")
    for n in written:
        print("  " + n)


if __name__ == "__main__":
    main()
