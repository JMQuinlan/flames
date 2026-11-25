#!/bin/bash

#SBATCH --job-name=T38
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=96
#SBATCH --partition=compute-long
#SBATCH -t 24:00:00

# input_Ma1.1_Wide (2026-11-14: domain x[-16,32] y[0,40] z[-24,24] m, shocks to 0.25 m, airframe
# skin to 0.0625 m with a 2-cell error buffer).  Start-up: 0.57M cells (level 4: 244k).  Steady state,
# estimated: ~1.9M cells (shock bands over the 40 x 48 m cross-section ~1.5M at 0.5/0.25 m), i.e.
# ~20k cells per rank on 96 ranks (fewer cells per rank and the 4-deep ghost layers of the small
# boxes start to dominate).  Cost: ~1500 coarse steps to t = 0.3 s at ~1e7 cell updates each
# (subcycled), 6.9k updates/s/rank measured locally (shared machine) -> ~7 h; 24 h wall time.

module purge
module load gnu9 mpich

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowTryonFun/Planes/T38-Talon/input_Ma1.1
srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowTryonFun/Planes/T38-Talon/input_Ma1.1_Wide
