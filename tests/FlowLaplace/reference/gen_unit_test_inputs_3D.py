#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the genuinely-3D FlowLaplace UNIT_TEST set: a SPHERICAL bubble.

3D Young-Laplace (sphere, two equal principal curvatures 1/R):
    p_gas - p_liquid = 2 * sigma / R          (NOTE: 2 sigma/R, vs sigma/R in 2D)

The interface is a sphere via  r = sqrt(x*x + y*y + z*z)  (the 2D cylinder used
sqrt(x*x + y*y)).  A correct Brackbill CSF holds the sphere motionless at R0.

Static files (no runner) so the SLURM driver alone owns parallelism.  Output:
    tests/FlowLaplace/UNIT_TEST_3D_R<R>_Sigma<sigma>
with the dual desktop/INCLINE plot_file (INCLINE /mmfs1 active) into UNIT_TEST_3D/.

Matrix: 3 radii x 2 surface tensions = 6 cases (same as the 2D set).
3D is much costlier, so the base mesh is coarser (32^3 + 1 AMR level) and the
run length is shorter (N_ACOUSTIC); bump these once it passes.

    python tests/FlowLaplace/reference/gen_unit_test_inputs_3D.py
"""

import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(_HERE, ".."))

# --- import the 2D module ONLY for its shared fluid constants + helpers --------
import sys
sys.path.insert(0, _HERE)
import gen_inputs as gi   # RHO_LIQ, P_LIQUID, EOS*, C_LIQ, fmt(), name_sci()

R_VALUES     = [1.0e-3, 1.0e-4, 1.0e-5]
SIGMA_VALUES = [0.036, 0.073]

# 3D mesh / run knobs (coarser + shorter than 2D for cost)
N_CELL_BASE = 32
MAX_LEVEL   = 1
REF_RATIO   = 2
EPS_CELLS   = 2
N_ACOUSTIC  = 100      # acoustic transits R/c_l (2D used 500; reduced for 3D cost)
N_FRAMES    = 40


def make_input_3d(R, sigma):
    fmt = gi.fmt
    half = 2.0 * R
    dx_finest = (2.0 * half) / (N_CELL_BASE * REF_RATIO ** MAX_LEVEL)
    epsilon = EPS_CELLS * dx_finest
    dp = 2.0 * sigma / R                       # *** 3D sphere: 2 sigma / R ***
    p_gas = gi.P_LIQUID + dp
    tau_ac = R / gi.C_LIQ
    stop_time = N_ACOUSTIC * tau_ac
    plot_dt = stop_time / N_FRAMES
    dt_max = stop_time / 200.0
    name = f"R{gi.name_sci(R)}_Sigma{gi.name_sci(sigma)}"
    base = f"output_{name}"

    return name, f"""#@ [{name.lower()}_3d]
#@ exe=hydro2
#@ dim=3

# =============================================================================
# 3D Laplace pressure test (air/water) -- SPHERE  R = {fmt(R)} m, sigma = {fmt(sigma)} N/m
# =============================================================================
# Spherical bubble at rest; gas pressure Laplace-balanced against the liquid:
#     p_gas - p_liquid = 2 sigma / R = {fmt(dp)} Pa   (3D Young-Laplace, 2 curvatures)
# A correct Brackbill CSF holds the sphere motionless at R0 = {fmt(R)} m.
# Interface: r = sqrt(x^2 + y^2 + z^2)  (sphere).  Mesh {N_CELL_BASE}^3 + {MAX_LEVEL} AMR level.
# =============================================================================

alamo.program = hydro2

### OUTPUT ###
# plot_file = ./tests/FlowLaplace/UNIT_TEST_3D/{base}                                  # DESKTOP (run from bin/)
plot_file = /mmfs1/home/ttryon/flames/bin/tests/FlowLaplace/UNIT_TEST_3D/{base}   # INCLINE /mmfs1 (default)

### MESHING ###
amr.plot_int        = -1
amr.plot_dt         = {fmt(plot_dt)}
amr.max_grid_size   = 32
amr.blocking_factor = 2
amr.regrid_int      = 10
amr.grid_eff        = 0.8
amr.max_level       = {MAX_LEVEL}
amr.n_cell          = {N_CELL_BASE} {N_CELL_BASE} {N_CELL_BASE}

nghost = 4

### TIME STEPPING ###
timestep                = 1.0e-12
dynamictimestep.on      = 1
dynamictimestep.verbose = 0
dynamictimestep.max     = {fmt(dt_max)}
dynamictimestep.min     = 1.0e-14
cfl                     = 0.3

### DIMENSIONS (cube) ###
geometry.prob_lo     = {fmt(-half)} {fmt(-half)} {fmt(-half)}
geometry.prob_hi     =  {fmt(half)}  {fmt(half)}  {fmt(half)}
geometry.is_periodic = 0 0 0
stop_time            = {fmt(stop_time)}

### ETA IC -- sphere: eta=0 inside (air, phase 1), eta=1 outside (water, phase 0) ###
eta.ic.type = expression
eta.ic.expression.constant.epsilon = {fmt(epsilon)}
eta.ic.expression.constant.R0      = {fmt(R)}
eta.ic.expression.region0 = "0.5*(1 + tanh((sqrt(x*x + y*y + z*z) - R0)/epsilon))"
epsilon = eta.ic.expression.constant.epsilon

### EQUATION OF STATE -- Sch20 Tammann water + ideal-gas air ###
eos0.gamma = 2.35
eos0.p0    = 1.0e9
eos0.cp    = 4186.0
eos0.cv    = 1781.0
eos1.gamma = 1.4
eos1.p0    = 0.0
eos1.cp    = 1000.0
eos1.cv    = 714.286

### HYDRO IC ###
density0.ic.type = expression
density0.ic.expression.region0 = "{fmt(gi.RHO_LIQ)}"
density1.ic.type = expression
density1.ic.expression.region0 = "1.0"

velocity0.ic.type = expression
velocity0.ic.expression.region0 = "0.0"
velocity0.ic.expression.region1 = "0.0"
velocity1.ic.type = expression
velocity1.ic.expression.region0 = "0.0"
velocity1.ic.expression.region1 = "0.0"

# Laplace-balanced: p_gas - p_liquid = 2 sigma/R = {fmt(dp)} Pa
pressure0.ic.type = expression
pressure0.ic.expression.region0 = "{fmt(gi.P_LIQUID)}"
pressure1.ic.type = expression
pressure1.ic.expression.region0 = "{fmt(p_gas)}"

### VISCOSITY (zero -- pure Laplace test) ###
mu0 = 0.0
mu0_b = 0.0
mu1 = 0.0
mu1_b = 0.0

### SURFACE TENSION ###
sigma = {fmt(sigma)}
pref  = 0.0

### BC -- Neumann on all 6 faces (sphere far from boundary) ###
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

### REFINEMENT -- eta-only at the interface band ###
eta_refinement_criterion   = 1.0e-1
omega_refinement_criterion = 1.0e+10
p_refinement_criterion     = 1.0e+10
rho_refinement_criterion   = 1.0e+10
gradu_refinement_criterion = 1.0e+10
cutoff = 1.0e-4

### SOURCE TERMS ###
m0.ic.constant.value = 0.0
u0.ic.constant.value = 0.0 0.0
q.ic.constant.value  = 0.0 0.0

### PHYSICS FLAGS ###
apply_surface_tension = 1
apply_buoyancy        = 0
apply_weight          = 0
apply_vaporization    = 0
grav                  = 0.0

### NUMERICS ###
Riemann_Solver.type = hllc
Limiter.type        = weno3
integration.type    = RungeKutta
integration.rk.type = 3
kappa_method        = 1

apply_sharpening     = 0
sharpening_frequency = 50
reinit_max_iter      = 100
reinit_tolerance     = 1e-6
omega_relax          = 0.5
relax_diag           = 1
"""


def main():
    written = []
    for R in R_VALUES:
        for sigma in SIGMA_VALUES:
            name, text = make_input_3d(R, sigma)
            out = os.path.join(TEST_DIR, f"UNIT_TEST_3D_{name}")
            with open(out, "w", newline="\n") as f:
                f.write(text)
            written.append(os.path.basename(out))
    print(f"wrote {len(written)} 3D Laplace UNIT_TEST inputs to {TEST_DIR}:")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
