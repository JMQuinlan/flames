#!/bin/bash
#
# =============================================================================
# 2D UNIT TEST SUITE  --  single sequential SLURM job (INCLINE)
# =============================================================================
# Runs every 2D UNIT_TEST input one after the next on the /mmfs1 high-speed
# drive.  Each case is expected to take only a few minutes.  A failing case is
# logged and SKIPPED (the suite continues) so one bad run never kills the sweep.
#
# After the job finishes, analyze + plot with:
#     python TryonSlurmFiles/2DUnitTests/aggregate_unit_tests.py
# (each test's analysis OUTPUT_DIR is toggled to /mmfs1 by default; flip the
#  comment in the analysis script to read a desktop ./tests/... copy instead.)
#
# Submit:   sbatch TryonSlurmFiles/2DUnitTests/Run2DUnitTests.sh
# =============================================================================

#SBATCH --job-name=2DUnitTests
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/unit2d_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/unit2d_%j_stderr
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 20:00:00

module purge
module load gnu9 mpich

NRANKS=128

EXE=/home/ttryon/flames/bin/hydro2-2d-g++
TESTS=/home/ttryon/flames/tests

# Run one case; never abort the suite on a single failure.
RUN() {
    echo
    echo "============================================================"
    echo "=== $(date +%T)   $(basename "$1")"
    echo "============================================================"
    local t0=$SECONDS
    srun -n "$NRANKS" --mpi=pmi2 "$EXE" "$1" || echo "  [FAILED] $1"
    echo "--- elapsed: $((SECONDS - t0)) s"
}

suite_start=$SECONDS

# ---- Riemann solvers ------------------------------------------------------
#RUN $TESTS/FlowRiemannUnitTests/UNIT_TEST_2D_Toro1a
#RUN $TESTS/FlowRiemannUnitTests/UNIT_TEST_2D_Toro2
#RUN $TESTS/FlowRiemannUnitTests/UNIT_TEST_2D_Garrick

# ---- Viscosity: Couette (single + multiphase) -----------------------------
#RUN $TESTS/FlowCouette/UNIT_TEST_2D_single
#RUN $TESTS/FlowCouette/UNIT_TEST_2D_multi

# ---- Pressure: Poiseuille -------------------------------------------------
#RUN $TESTS/FlowPoseuille/UNIT_TEST_2D

# ---- Viscosity: Taylor-Green, Lamb-Oseen ----------------------------------
#RUN $TESTS/FlowTaylorGreenVortex/UNIT_TEST_2D
#RUN $TESTS/FlowLambOseenVortex/UNIT_TEST_2D

# ---- Embedded BC: Flow Vortex Re40 ----------------------------------------
#RUN $TESTS/FlowVortexShed/UNIT_TEST_2D_Re40

# ---- Surface tension: Laplace (R x sigma matrix) --------------------------
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-3_Sigma3.6e-2
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-3_Sigma7.3e-2
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-4_Sigma3.6e-2
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-4_Sigma7.3e-2
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-5_Sigma3.6e-2
RUN $TESTS/FlowLaplace/UNIT_TEST_2D_R1.0e-5_Sigma7.3e-2

echo
echo "=== 2D UNIT TEST SUITE COMPLETE   (total: $((SECONDS - suite_start)) s) ==="
echo "    analyze: python TryonSlurmFiles/2DUnitTests/aggregate_unit_tests.py"
