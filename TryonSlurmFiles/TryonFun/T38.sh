#!/bin/bash

#SBATCH --job-name=T38
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --partition=compute-long
#SBATCH -t 24:00:00

# input_Ma1.1_Wide, optimized 2026-11-14: ~1.3M cells at steady state (base 37k at 1 m; airframe
# surface 0.39M at 0.0625 m; shock bands at 0.5 / 0.25 m), i.e. ~20k cells per rank on 64 ranks.
# 128 ranks would leave ~10k cells per rank in mostly 8^3-16^3 boxes, where the 4-deep ghost
# layers outweigh the box itself and communication dominates.  Cost: ~1500 coarse steps to
# t = 0.3 s at ~1e7 cell updates each (subcycled); measured 6.9k cell updates/s/rank locally
# (16 ranks, shared machine) -> ~9 h on 64 ranks; 24 h wall time leaves margin.

module purge
module load gnu9 mpich

#srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowTryonFun/Planes/T38-Talon/input_Ma1.1
srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowTryonFun/Planes/T38-Talon/input_Ma1.1_Wide
