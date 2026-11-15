#!/bin/bash

#SBATCH --job-name=MPExLo
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06Exp_Cattaneo_LowPa (2026-11-15): Cattaneo et al. experimental configuration, FULL 3D bubble 12 um under a
## rigid no-slip substrate, 5 driven NSCBC faces, Option 1 code (see deck header).  Jetting / shape modes.
## Sizing (measured locally, 2026-11-15): 843k cells at t=0 (finest level 443k in 866 boxes, ~1650 boxes total),
## coarse dt 3.36e-10 s with 16 finest substeps; ~9e-5 core-s per finest cell update.  Estimated cost
## HighPa ~13-21k core-h (bubble area ~2x R0 on average, jets), LowPa ~0.7x that.  2 nodes x 128 tasks.
## Needs hydro2 built from the commit with visc.relax_cell_local / thermal.dt_limit (Hydro2.cpp, 2026-11-14/15).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_06Exp_Cattaneo_LowPa
