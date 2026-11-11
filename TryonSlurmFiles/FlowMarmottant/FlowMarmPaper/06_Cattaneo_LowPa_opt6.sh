#!/bin/bash

#SBATCH --job-name=MP06Loo6
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06_Cattaneo_LowPa_opt6, option 6 with slip OFF (shell.slip = 0), constant kappa_s: validation vs Cattaneo et al. 2025 Fig. 2a (60 kPa, measured pulse envelope; see deck header) ##
# Needs hydro2 built from commit after 547644d5c (shell.slip flag in Hydro2.cpp).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_06_Cattaneo_LowPa_opt6
