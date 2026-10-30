#!/bin/bash

#SBATCH --job-name=CollEffMass
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# UNCOATED Sch20 collapse validation, same flags as the coated EffMass deck, cfl 0.4.
# Needs hydro2 built from the 10-30 working copy (the new flags are not in commit b5e17d08f,
# and an older binary ignores them silently).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/10_30_Changes/input_Sch20_Collapsing_Neumann_Large_3D_EffMass
