#!/bin/bash

#SBATCH --job-name=CollIso
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# UNCOATED Sch20 collapse (p_inf/p_b 1427): 11_02 LocalRate deck + jump.curvature_iso = 1. Shell/jump flags inert (sigma = 0).
# Needs hydro2 built from the 11-04 working copy (jump.curvature_iso; an older binary ignores unknown keys silently).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/11_04_Changes/input_Sch20_Collapsing_Neumann_Large_3D_Iso
