#!/bin/bash

#SBATCH --job-name=CollLocalRate
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# UNCOATED Sch20 collapse validation (p_inf/p_b 1427), same 11-02 flag set as the coated LocalRate deck, cfl 0.4.
# Shell/jump flags are inert (sigma = 0, kappa_s = 0); relax.closure / allow_tension / trace act on every interface cell.
# Needs hydro2 built from the 11-02 working copy (see the coated script).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/11_02_Changes/input_Sch20_Collapsing_Neumann_Large_3D_LocalRate
