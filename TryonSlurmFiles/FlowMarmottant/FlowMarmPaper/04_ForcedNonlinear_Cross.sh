#!/bin/bash

#SBATCH --job-name=MP04ForcedX
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 04_ForcedNonlinear_Cross (rate-dependent shell viscosity, shell.kappa_s_alpha) (see the deck header; finest dx 7.812e-08 m, eps 1.562e-07 m) ##
# Needs hydro2 built from the 2026-11-05 flag-audit code (interface jump / tensor curvature built in).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_04_ForcedNonlinear_Cross
