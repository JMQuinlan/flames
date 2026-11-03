#!/bin/bash

#SBATCH --job-name=MiliLocalRate
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 100:00:00

module purge
module load gnu9 mpich

## 3D Variants ##
# Coated Light_Mili octant, 11-02 flag set: tensor curvature + mass-fraction relax (closure 1) + allow_tension 2
# + trace fix + LOCAL Gamma (gamma_uniform 0, no normal extension) + stretch-rate fix (kappa_s_rate 0, rate_local 1).
# Same mesh/eps/cfl as the 11-01 TensorMode2 deck (eps 0.0016, eps/dx_finest 2, cfl 0.5).
# Needs hydro2 built from the 11-02 working copy (shell.rate_local, relax.trace, relax.closure, relax.allow_tension
# are not in a58870c36, and an older binary ignores them silently).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/11_02_Changes/input_Sch20-Oscillating_Marmottant_Light_Mili_LocalRate
