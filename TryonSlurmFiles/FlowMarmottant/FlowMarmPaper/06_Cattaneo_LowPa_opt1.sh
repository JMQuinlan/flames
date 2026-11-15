#!/bin/bash

#SBATCH --job-name=MP06Lo1
#SBATCH -o /mmfs1/home/ttryon/FLAMES_out/flame_%j_stdout
#SBATCH -e /mmfs1/home/ttryon/FLAMES_out/err_flame_%j_stdout
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=compute-long
#SBATCH -t 500:00:00

module purge
module load gnu9 mpich

## MarmPaper 06_Cattaneo_LowPa_opt1 (2026-11-14): gooderrun (slip 12, two_pressure, opt8, trace_pT 3) + spherical NSCBC +
## thermal.dt_limit + Option 1 path-integrated viscous relaxation jump; constant kappa_s; 60 kPa, octant (see deck header).
# Needs hydro2 built from the commit that adds visc.relax_cell_local and thermal.dt_limit (Hydro2.cpp, 2026-11-14).
cd /home/ttryon/flames

srun --mpi=pmi2 /home/ttryon/flames/bin/hydro2-3d-g++ /home/ttryon/flames/tests/FlowMarmottant/MarmPaper/input_06_Cattaneo_LowPa_opt1
