#!/bin/bash
#
# =============================================================================
# FLOWWEDGE MACH SWEEP  --  SLURM submit wrapper (INCLINE)
# =============================================================================
# The sweep itself (Mach list, template substitution, run loop) lives in
#     tests/FlowWedge/TEMPLATES/sweep_mach.bash   (+ TEMPLATES/input_Wedge)
# This wrapper only sets the INCLINE launcher and the /mmfs1 output root.
#
# Submit:   sbatch TryonSlurmFiles/FlowWedge/RunWedgeMachSweep.sh
#           sbatch --export=ALL,MACH="2.0 20.0 35.0" TryonSlurmFiles/FlowWedge/RunWedgeMachSweep.sh
# Analyze:  python3 tests/FlowWedge/reference/analyze_mach_sweep.py      (defaults to /mmfs1)
# =============================================================================

#SBATCH --job-name=WedgeMachSweep
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/wedge_sweep_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/wedge_sweep_%j_stderr
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH -t 48:00:00

module purge
module load gnu9 mpich

export LAUNCH="srun -n 128 --mpi=pmi2"
export EXEC=/home/ttryon/flames/bin/hydro2-2d-g++
export OUTPUT_ROOT=/mmfs1/home/ttryon/flames/bin/tests/FlowWedge

bash /home/ttryon/flames/tests/FlowWedge/TEMPLATES/sweep_mach.bash
