#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for the FlowLaplace surface-tension test matrix.

Emits one Hydro2 input file per (R, sigma) combination into ../  (the
FlowLaplace test directory), named  R<R>_Sigma<sigma>.

Laplace test (2D cylindrical bubble, single curvature 1/R):
    p_gas - p_liquid = sigma / R          (Young-Laplace, 2D)
A correctly-implemented Brackbill CSF holds the bubble motionless at R = R0.
Drift / oscillation / collapse flags a surface-tension scaling bug, or shows
that a given (R, sigma) is too stiff for the chosen resolution (the point of
the sensitivity sweep).

Fluid = Garrick (2017) gas-liquid pair (Tammann liquid + ideal gas) -- chosen
because it carries a Tammann stiffness yet runs cheaply.  All quantities are in
Garrick's consistent (non-dimensional) unit system; R is treated as a length in
those same units.

Mesh: 64x64 base + 1 AMR level (eta-only refinement) -> 128 finest cells across
the 4R box, dx_finest = R/32, bubble radius resolved by 32 finest cells.
Interface band epsilon = 12 * dx_finest = 0.375 R  (~12 cells, inside the
10-20-cell target carried over from the RPE work).

Re-run from the repo root or anywhere:
    python tests/FlowLaplace/reference/gen_inputs.py
"""

import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(_HERE, ".."))

# ============================  TEST MATRIX  =================================
R_VALUES     = [1.0, 0.001, 0.000001]      # metres (Garrick units)
SIGMA_VALUES = [0.0, 1.0, 10.0]            # surface-tension coefficient

# ============================  FLUID (Garrick)  ============================
# Liquid = phase 0 (eta = 1, outside bubble); Gas = phase 1 (eta = 0, inside).
RHO_LIQ = 0.991                            # liquid density
RHO_GAS = 1.241                            # gas density
P_LIQUID = 1.0                             # uniform far-field liquid pressure
EOS0_GAMMA, EOS0_P0 = 5.5, 1.505           # Tammann liquid
EOS1_GAMMA, EOS1_P0 = 1.4, 0.0            # ideal gas
CP, CV = 1004.5, 717.5                      # shared cp / cv (Garrick)

# liquid sound speed for the sigma = 0 (no capillary timescale) cases
C_LIQ = (EOS0_GAMMA * (P_LIQUID + EOS0_P0) / RHO_LIQ) ** 0.5

# ============================  MESH / NUMERICS  ============================
N_CELL_BASE = 64
MAX_LEVEL   = 1                            # 64 base + 1 level -> 128 finest
REF_RATIO   = 2
EPS_CELLS   = 12                           # interface band thickness, finest cells
N_PERIODS   = 15                           # capillary/acoustic times to simulate
N_FRAMES    = 50


def fmt(v):
    """Compact, round-trippable number formatting for input file bodies."""
    return f"{v:.10g}"


def name_sci(v):
    """Decimal-mantissa scientific notation for file names, e.g.
    1.0 -> '1.0e0', 0.001 -> '1.0e-3', 1e-06 -> '1.0e-6', 0.0 -> '0.0'.
    Round-trips through float() so the analysis scripts can parse it back."""
    if v == 0:
        return "0.0"
    exp = math.floor(math.log10(abs(v)))
    mant = v / 10.0 ** exp
    if abs(mant) >= 9.99995:                 # floor undershot a decade boundary
        mant /= 10.0
        exp += 1
    return f"{mant:.1f}e{exp}"


def make_input(R, sigma):
    half = 2.0 * R                                  # domain half-width = 2R
    dx_finest = (2.0 * half) / (N_CELL_BASE * REF_RATIO ** MAX_LEVEL)
    epsilon = EPS_CELLS * dx_finest                 # = 0.375 R
    dp = sigma / R                                  # Young-Laplace jump (2D)
    p_gas = P_LIQUID + dp

    # characteristic timescale: capillary if sigma > 0, else acoustic
    if sigma > 0.0:
        tau = (RHO_LIQ * R ** 3 / sigma) ** 0.5
    else:
        tau = R / C_LIQ
    stop_time = N_PERIODS * tau
    plot_dt   = stop_time / N_FRAMES
    dt_init   = stop_time / 1000.0
    dt_max    = stop_time / 200.0
    dt_min    = stop_time / 1.0e8

    apply_st = 1 if sigma > 0.0 else 0
    name = f"R{name_sci(R)}_Sigma{name_sci(sigma)}"

    return name, f"""#@ [{name.lower()}]
#@ exe=hydro2
#@ dim=2

# =============================================================================
# Laplace pressure test  --  R = {fmt(R)},  sigma = {fmt(sigma)}
# =============================================================================
# 2D cylindrical bubble at rest, gas pressure Laplace-balanced against the
# liquid:
#     p_gas - p_liquid = sigma / R = {fmt(dp)}     (Young-Laplace, single curvature)
# A correct Brackbill CSF holds the bubble motionless at R0 = {fmt(R)}.  Drift,
# oscillation, or collapse indicates either a surface-tension scaling bug or
# that this (R, sigma) is too stiff for the resolution (the sweep's purpose).
#
# Fluid: Garrick (2017) Tammann liquid + ideal gas (cheap but stiffened).
#   liquid (phase 0, eta=1, outside):  rho={fmt(RHO_LIQ)}, gamma={fmt(EOS0_GAMMA)}, p0={fmt(EOS0_P0)}
#   gas    (phase 1, eta=0, inside):   rho={fmt(RHO_GAS)}, gamma={fmt(EOS1_GAMMA)}, p0={fmt(EOS1_P0)}
#   p_liquid = {fmt(P_LIQUID)},  p_gas = p_liquid + sigma/R = {fmt(p_gas)}
#
# Mesh: {N_CELL_BASE}x{N_CELL_BASE} base + {MAX_LEVEL} AMR level (eta-only) over a {fmt(4*R)} box.
#   dx_finest = {fmt(dx_finest)} = R/{N_CELL_BASE*REF_RATIO**MAX_LEVEL//4};  R resolved by {N_CELL_BASE*REF_RATIO**MAX_LEVEL//4} finest cells.
#   epsilon   = {EPS_CELLS}*dx_finest = {fmt(epsilon)} = {fmt(epsilon/R)} R  (interface band).
#
# Timescale: {'capillary tau=sqrt(rho R^3/sigma)' if sigma>0 else 'acoustic tau=R/c_l'} = {fmt(tau)};
#   stop_time = {N_PERIODS} tau = {fmt(stop_time)} ({N_FRAMES} frames).
#
# This is a unit test -- keep it fast: low resolution, eta-only AMR, Neumann
# walls (bubble is far from the boundary in the {fmt(4*R)} box).
# =============================================================================

alamo.program = hydro2

### OUTPUT ###
plot_file = ./tests/FlowLaplace/output_{name}

### MESHING ###
amr.plot_int        = -1
amr.plot_dt         = {fmt(plot_dt)}
amr.max_grid_size   = 500000
amr.blocking_factor = 2
amr.regrid_int      = 10
amr.grid_eff        = 0.8
amr.max_level       = {MAX_LEVEL}
amr.n_cell          = {N_CELL_BASE} {N_CELL_BASE}

nghost = 4

### TIME STEPPING ###
timestep                = {fmt(dt_init)}
dynamictimestep.on      = 1
dynamictimestep.verbose = 0
dynamictimestep.max     = {fmt(dt_max)}
dynamictimestep.min     = {fmt(dt_min)}
cfl                     = 0.4

### DIMENSIONS ###
geometry.prob_lo     = {fmt(-half)} {fmt(-half)} 0.0
geometry.prob_hi     =  {fmt(half)}  {fmt(half)} 0.0
geometry.is_periodic = 0 0 0
stop_time            = {fmt(stop_time)}

### ETA INITIAL CONDITIONS ###
# eta = 0 inside bubble (gas, phase 1), eta = 1 outside (liquid, phase 0).
eta.ic.type = expression
eta.ic.expression.constant.epsilon = {fmt(epsilon)}
eta.ic.expression.constant.R0      = {fmt(R)}
eta.ic.expression.region0 = "0.5*(1 + tanh((sqrt(x*x + y*y) - R0)/epsilon))"
epsilon = eta.ic.expression.constant.epsilon

### EQUATION OF STATE -- Garrick Tammann liquid + ideal gas ###
eos0.gamma = {fmt(EOS0_GAMMA)}
eos0.p0    = {fmt(EOS0_P0)}
eos0.cp    = {fmt(CP)}
eos0.cv    = {fmt(CV)}

eos1.gamma = {fmt(EOS1_GAMMA)}
eos1.p0    = {fmt(EOS1_P0)}
eos1.cp    = {fmt(CP)}
eos1.cv    = {fmt(CV)}

### HYDRO INITIAL CONDITIONS ###
density0.ic.type = expression
density0.ic.expression.region0 = "{fmt(RHO_LIQ)}"

density1.ic.type = expression
density1.ic.expression.region0 = "{fmt(RHO_GAS)}"

velocity0.ic.type = expression
velocity0.ic.expression.region0 = "0.0"
velocity0.ic.expression.region1 = "0.0"

velocity1.ic.type = expression
velocity1.ic.expression.region0 = "0.0"
velocity1.ic.expression.region1 = "0.0"

# Laplace-balanced pressures: p_gas - p_liquid = sigma/R = {fmt(dp)}
pressure0.ic.type = expression
pressure0.ic.expression.region0 = "{fmt(P_LIQUID)}"      # liquid (outside, eta=1)

pressure1.ic.type = expression
pressure1.ic.expression.region0 = "{fmt(p_gas)}"      # gas (inside, eta=0)

### VISCOSITY (zero -- pure Laplace test, no damping) ###
mu0   = 0.0
mu0_b = 0.0
mu1   = 0.0
mu1_b = 0.0

### SURFACE TENSION ###
sigma = {fmt(sigma)}
pref  = 0.0

### BOUNDARY CONDITIONS - Neumann walls (bubble far from boundary) ###
density.bc.type.xhi = neumann
density.bc.type.xlo = neumann
density.bc.type.ylo = neumann
density.bc.type.yhi = neumann

energy.bc.type.xhi = neumann
energy.bc.type.xlo = neumann
energy.bc.type.ylo = neumann
energy.bc.type.yhi = neumann

momentum.bc.type.xhi = neumann neumann
momentum.bc.type.xlo = neumann neumann
momentum.bc.type.ylo = neumann neumann
momentum.bc.type.yhi = neumann neumann

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
apply_surface_tension = {apply_st}
apply_buoyancy        = 0
apply_weight          = 0
apply_vaporization    = 0
Spec_Vol              = 1
grav                  = 0.0

### NUMERICS -- WENO3 + SSPRK3 (project standard) ###
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
            name, text = make_input(R, sigma)
            path = os.path.join(TEST_DIR, name)
            with open(path, "w", newline="\n") as f:
                f.write(text)
            written.append(name)
    print(f"wrote {len(written)} input files to {TEST_DIR}:")
    for n in written:
        print(f"  {n}")


if __name__ == "__main__":
    main()
