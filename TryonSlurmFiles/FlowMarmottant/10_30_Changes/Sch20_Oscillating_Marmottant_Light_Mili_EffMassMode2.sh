#!/bin/bash

#SBATCH --job-name=MiliEffMassM2
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# Coated Light_Mili octant, effusivity + mass-fraction relax + mode 2, cfl 0.5, eps 0.0016 (octant twin of slab run CD6_K072).
# Needs hydro2 built from the 10-30 working copy (the new flags are not in commit b5e17d08f,
# and an older binary ignores them silently).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/10_30_Changes/input_Sch20-Oscillating_Marmottant_Light_Mili_EffMassMode2
