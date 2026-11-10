#!/bin/bash

#SBATCH --job-name=MP05JettinX
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 05_Jetting_Cross (rate-dependent shell viscosity, shell.kappa_s_alpha) (see the deck header; finest dx 3.125e-08 m, eps 6.250e-08 m) ##
# Needs hydro2 built from the 2026-11-05 flag-audit code (interface jump / tensor curvature built in).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_05_Jetting_Cross
