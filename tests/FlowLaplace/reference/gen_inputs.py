#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator for the FlowLaplace surface-tension test matrix (realistic air/water).

Emits one Hydro2 input file per (R, sigma) combination into ../  (the
FlowLaplace test directory), named  R<R>_Sigma<sigma>.

Laplace test (2D cylindrical bubble, single curvature 1/R):
    p_gas - p_liquid = sigma / R          (Young-Laplace, 2D)
A correctly-implemented Brackbill CSF holds the bubble motionless at R = R0.
Drift / oscillation / collapse flags a surface-tension scaling bug.

Fluid = air / water (the validated Sch20 Tammann-water + ideal-gas-air pair).
Background pressure is 1 atm, so the Laplace jump sigma/R (tens to thousands of
Pa) is small relative to the ~101 kPa background -- a well-conditioned pressure
field, unlike the earlier non-dimensional setup where sigma/R dwarfed the
background and drove the solver into the sound-speed clamps.

Mesh: 64x64 base + 1 AMR level (eta-only refinement) -> 128 finest cells across
the 4R box, dx_finest = R/32, bubble radius resolved by 32 finest cells.
Interface band epsilon = 2 * dx_finest = 0.0625 R  (~2 cells, matching the
actual Sch20 RPE interface thickness).

Time: water's sound speed (~1533 m/s) makes the capillary period >> the acoustic
transit time, so simulating many capillary periods would be millions of steps.
For an equilibrium check we instead run a fixed number of acoustic transits
(N_ACOUSTIC * R/c_l), which keeps every case at ~40k steps regardless of R.

Re-run from the repo root or anywhere:
    python tests/FlowLaplace/reference/gen_inputs.py
"""

import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.normpath(os.path.join(_HERE, ".."))

# ============================  TEST MATRIX  =================================
R_VALUES     = [1.0e-3, 1.0e-4, 1.0e-5]    # bubble radii [m]: 1 mm, 100 um, 10 um
SIGMA_VALUES = [0.0, 0.036, 0.073]         # surface tension [N/m]: none,
#                                            surfactant-laden, clean air/water (~0.0728)

# ============================  FLUID (air / water)  =========================
# Liquid = water = phase 0 (eta = 1, outside bubble);
# Gas    = air   = phase 1 (eta = 0, inside bubble).
# EOS values are the validated Sch20 air/water pair used by the RPE tests.
RHO_LIQ = 1000.0                           # water density [kg/m^3]
RHO_GAS = 1.0                              # air   density [kg/m^3]
P_LIQUID = 101325.0                        # far-field liquid pressure [Pa] (1 atm)
EOS0_GAMMA, EOS0_P0 = 2.35, 1.0e9          # water (Tammann / stiffened gas)
EOS1_GAMMA, EOS1_P0 = 1.4, 0.0            # air (ideal gas)
CP0, CV0 = 4186.0, 1781.0                  # water cp / cv [J/kg/K]
CP1, CV1 = 1000.0, 714.286                 # air   cp / cv [J/kg/K]

# Liquid (water) sound speed; sets the acoustic transit time R/c_l used for
# stop_time.  c_l = sqrt(gamma (p + p0)/rho) ~ 1533 m/s.
C_LIQ = (EOS0_GAMMA * (P_LIQUID + EOS0_P0) / RHO_LIQ) ** 0.5

# ============================  MESH / NUMERICS  ============================
N_CELL_BASE = 64
MAX_LEVEL   = 1                            # 64 base + 1 level -> 128 finest
REF_RATIO   = 2
EPS_CELLS   = 2                            # interface band thickness, finest cells (~Sch20 RPE)
N_ACOUSTIC  = 500                          # acoustic transits R/c_l to simulate (~40k steps/case)
N_FRAMES    = 50


def fmt(v):
    """Compact, round-trippable number formatting for input file bodies."""
    return f"{v:.10g}"


def name_sci(v):
    """Decimal-mantissa scientific notation for file names, e.g.
    1e-3 -> '1.0e-3', 0.073 -> '7.3e-2', 0.0 -> '0.0'.
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
    epsilon = EPS_CELLS * dx_finest                 # = 0.0625 R (2 cells)
    dp = sigma / R                                  # Young-Laplace jump (2D)
    p_gas = P_LIQUID + dp

    # Acoustic-transit-based stop time (see module docstring): stiff water makes
    # the capillary period >> R/c_l, so we run a fixed number of acoustic
    # transits to keep the step count ~constant.  Capillary time is reported.
    tau_ac  = R / C_LIQ
    tau_cap = (RHO_LIQ * R ** 3 / sigma) ** 0.5 if sigma > 0.0 else float("inf")
    stop_time = N_ACOUSTIC * tau_ac
    plot_dt   = stop_time / N_FRAMES
    dt_init   = 1.0e-12                              # tiny fixed start; CFL ramps up
    dt_max    = stop_time / 200.0
    dt_min    = 1.0e-14                              # floor, kept below dt_init

    apply_st = 1 if sigma > 0.0 else 0
    name = f"R{name_sci(R)}_Sigma{name_sci(sigma)}"
    tau_cap_str = "inf (sigma=0)" if sigma == 0.0 else fmt(tau_cap)

    return name, f"""#@ [{name.lower()}]
#@ exe=hydro2
#@ dim=2

# =============================================================================
# Laplace pressure test (air/water)  --  R = {fmt(R)} m,  sigma = {fmt(sigma)} N/m
# =============================================================================
# 2D cylindrical bubble at rest, gas pressure Laplace-balanced against the
# liquid:
#     p_gas - p_liquid = sigma / R = {fmt(dp)} Pa     (Young-Laplace, single curvature)
# A correct Brackbill CSF holds the bubble motionless at R0 = {fmt(R)} m.  Drift,
# oscillation, or collapse indicates a surface-tension scaling bug.
#
# Fluid: air / water (validated Sch20 Tammann-water + ideal-gas-air pair).
#   water (phase 0, eta=1, outside):  rho={fmt(RHO_LIQ)}, gamma={fmt(EOS0_GAMMA)}, p0={fmt(EOS0_P0)}
#   air   (phase 1, eta=0, inside):   rho={fmt(RHO_GAS)}, gamma={fmt(EOS1_GAMMA)}, p0={fmt(EOS1_P0)}
#   p_liquid = {fmt(P_LIQUID)} Pa (1 atm),  p_gas = p_liquid + sigma/R = {fmt(p_gas)} Pa
#   dp/p_liquid = {fmt(dp / P_LIQUID)}  (small -> well-conditioned pressure field)
#
# Mesh: {N_CELL_BASE}x{N_CELL_BASE} base + {MAX_LEVEL} AMR level (eta-only) over a {fmt(4*R)} m box.
#   dx_finest = {fmt(dx_finest)} = R/{N_CELL_BASE*REF_RATIO**MAX_LEVEL//4};  R resolved by {N_CELL_BASE*REF_RATIO**MAX_LEVEL//4} finest cells.
#   epsilon   = {EPS_CELLS}*dx_finest = {fmt(epsilon)} = {fmt(epsilon/R)} R  (interface band).
#
# Time: acoustic transit R/c_l = {fmt(tau_ac)} s (c_l = {fmt(C_LIQ)} m/s);
#   capillary tau = sqrt(rho R^3/sigma) = {tau_cap_str} s;
#   stop_time = {N_ACOUSTIC} * R/c_l = {fmt(stop_time)} s ({N_FRAMES} frames).
#
# This is a unit test -- keep it fast: low resolution, eta-only AMR, Neumann
# walls (bubble is far from the boundary in the {fmt(4*R)} m box).
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
cfl                     = 0.3

### DIMENSIONS ###
geometry.prob_lo     = {fmt(-half)} {fmt(-half)} 0.0
geometry.prob_hi     =  {fmt(half)}  {fmt(half)} 0.0
geometry.is_periodic = 0 0 0
stop_time            = {fmt(stop_time)}

### ETA INITIAL CONDITIONS ###
# eta = 0 inside bubble (air, phase 1), eta = 1 outside (water, phase 0).
eta.ic.type = expression
eta.ic.expression.constant.epsilon = {fmt(epsilon)}
eta.ic.expression.constant.R0      = {fmt(R)}
eta.ic.expression.region0 = "0.5*(1 + tanh((sqrt(x*x + y*y) - R0)/epsilon))"
epsilon = eta.ic.expression.constant.epsilon

### EQUATION OF STATE -- Sch20 Tammann water + ideal-gas air ###
eos0.gamma = {fmt(EOS0_GAMMA)}
eos0.p0    = {fmt(EOS0_P0)}
eos0.cp    = {fmt(CP0)}
eos0.cv    = {fmt(CV0)}

eos1.gamma = {fmt(EOS1_GAMMA)}
eos1.p0    = {fmt(EOS1_P0)}
eos1.cp    = {fmt(CP1)}
eos1.cv    = {fmt(CV1)}

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

# Laplace-balanced pressures: p_gas - p_liquid = sigma/R = {fmt(dp)} Pa
pressure0.ic.type = expression
pressure0.ic.expression.region0 = "{fmt(P_LIQUID)}"      # water (outside, eta=1)

pressure1.ic.type = expression
pressure1.ic.expression.region0 = "{fmt(p_gas)}"      # air (inside, eta=0)

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
